"""Diagnose layer-level BF16 differences without substituting reference activations."""

import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/store/Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="results/layer-trace")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    (output / "input.json").write_text(json.dumps(ids))
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    labels = []

    def save(label, tensor):
        labels.append(label)
        np.save(output / f"ref-{label}.npy", tensor.detach().float().cpu().numpy())

    def observe(label):
        return lambda module, inputs, value: save(label, value)

    model.model.embed_tokens.register_forward_hook(observe("embedding"))
    for index, layer in enumerate(model.model.layers):
        prefix = f"layer{index}-"
        layer.input_layernorm.register_forward_hook(observe(prefix + "input_norm"))
        layer.self_attn.q_proj.register_forward_hook(observe(prefix + "q_proj"))
        layer.self_attn.q_norm.register_forward_hook(observe(prefix + "q_norm"))
        layer.self_attn.v_proj.register_forward_hook(observe(prefix + "v_proj"))
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, inputs, label=prefix + "attn": save(label, inputs[0])
        )
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs, label=prefix + "swiglu": save(label, inputs[0])
        )
        layer.register_forward_hook(observe(prefix + "output"))
    model.model.norm.register_forward_hook(observe("final_norm"))
    original_rope = modeling_qwen3.apply_rotary_pos_emb
    layer_index = 0

    def rope(*args, **kwargs):
        nonlocal layer_index
        q, k = original_rope(*args, **kwargs)
        save(f"layer{layer_index}-q_rope", q.transpose(1, 2).contiguous())
        save(f"layer{layer_index}-k_rope", k.transpose(1, 2).contiguous())
        layer_index += 1
        return q, k

    modeling_qwen3.apply_rotary_pos_emb = rope
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        model(torch.tensor([ids], device="cuda"), use_cache=False)
    modeling_qwen3.apply_rotary_pos_emb = original_rope
    del model
    torch.cuda.empty_cache()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        [
            str(pathlib.Path.home() / ".cargo/bin/cargo"),
            "run",
            "--release",
            "-p",
            "rsglang-engine",
            "--example",
            "trace",
            "--",
            args.model,
            str(output / "input.json"),
            str(output / "rust"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    report = []
    for label in labels:
        ref = np.load(output / f"ref-{label}.npy").reshape(-1).astype(np.float64)
        got = np.fromfile(output / "rust" / f"{label}.bin", dtype="<f4").astype(
            np.float64
        )
        assert got.shape == ref.shape, label
        report.append(
            {
                "layer": label,
                "elements": len(ref),
                "different_elements": int(np.count_nonzero(got != ref)),
                "nrmse": float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
                "max_abs_error": float(np.max(np.abs(got - ref))),
            }
        )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report[:12] + report[-1:], indent=2))


if __name__ == "__main__":
    main()
