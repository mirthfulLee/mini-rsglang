"""Stream a real MXFP4 checkpoint through an independent PyTorch/HF reference.

Experts are decoded on demand; this verifier never materializes all 120B weights.
The reference uses HF attention/RMSNorm/YaRN and PyTorch expert arithmetic.
"""

import argparse
import contextlib
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, GptOssConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/store/openai/gpt-oss-120b")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", default="results/gpt-oss/checkpoint")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    model = pathlib.Path(args.model)
    cfg = GptOssConfig.from_pretrained(model, local_files_only=True)
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tok.encode(args.prompt)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.bfloat16,
    )
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    refs, continuation = [], []
    start = time.monotonic()
    with contextlib.ExitStack() as stack:
        files = {
            name: stack.enter_context(
                safe_open(model / name, framework="pt", device="cpu")
            )
            for name in set(index.values())
        }

        def tensor(name):
            return files[index[name]].get_tensor(name).cuda()

        def expert(root, name, which, table=lut):
            blocks_name = f"{root}.{name}_blocks"
            scales_name = f"{root}.{name}_scales"
            blocks = files[index[blocks_name]].get_slice(blocks_name)[which].cuda()
            scales = files[index[scales_name]].get_slice(scales_name)[which].cuda()
            decoded = torch.stack(
                (table[(blocks & 15).long()], table[(blocks >> 4).long()]), dim=-1
            ).flatten(-2)
            return torch.ldexp(decoded, scales.int().unsqueeze(-1) - 127).flatten(-2)

        def norm(x, name):
            module = GptOssRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).cuda()
            module.weight.data = tensor(name)
            return module(x)

        cache = DynamicCache()
        rotary = GptOssRotaryEmbedding(cfg).cuda()
        position = 0
        tokens = ids
        with torch.inference_mode():
            for step in range(args.steps + 1):
                # Fetch embedding rows through safetensors independently of Rust's loader.
                rows = files[index["model.embed_tokens.weight"]].get_slice(
                    "model.embed_tokens.weight"
                )
                x = torch.stack([rows[token] for token in tokens]).unsqueeze(0).cuda()
                positions = torch.arange(
                    position, position + len(tokens), device="cuda"
                )
                cos_sin = rotary(x, positions.unsqueeze(0))
                total = position + len(tokens)
                columns = torch.arange(total, device="cuda")
                for layer in range(cfg.num_hidden_layers):
                    root = f"model.layers.{layer}"
                    n = norm(x, root + ".input_layernorm.weight")
                    attn = GptOssAttention(cfg, layer).bfloat16().cuda()
                    prefix = root + ".self_attn."
                    state = {name: tensor(prefix + name) for name in attn.state_dict()}
                    attn.load_state_dict(state)
                    allowed = columns.unsqueeze(0) <= positions.unsqueeze(1)
                    if cfg.layer_types[layer] == "sliding_attention":
                        allowed &= (
                            columns.unsqueeze(0)
                            > positions.unsqueeze(1) - cfg.sliding_window
                        )
                    mask = torch.where(
                        allowed, 0.0, torch.finfo(torch.bfloat16).min
                    ).to(torch.bfloat16)[None, None]
                    attention, _ = attn(
                        n,
                        cos_sin,
                        mask,
                        past_key_values=cache,
                        cache_position=positions,
                    )
                    residual = x + attention
                    n = norm(
                        residual, root + ".post_attention_layernorm.weight"
                    ).reshape(-1, cfg.hidden_size)
                    scores = F.linear(
                        n,
                        tensor(root + ".mlp.router.weight"),
                        tensor(root + ".mlp.router.bias"),
                    )
                    values, selected = scores.topk(cfg.num_experts_per_tok, dim=-1)
                    weights = values.softmax(dim=-1)
                    # HF CUDA sums BF16 weighted contributions in FP32, then casts once.
                    mixed = torch.zeros_like(n, dtype=torch.float32)
                    expert_root = root + ".mlp.experts"
                    gate_bias = tensor(expert_root + ".gate_up_proj_bias")
                    down_bias = tensor(expert_root + ".down_proj_bias")
                    for which in selected.unique().tolist():
                        token_rows, slots = torch.where(selected == which)
                        merged = F.linear(
                            n[token_rows],
                            expert(expert_root, "gate_up_proj", which),
                            gate_bias[which],
                        )
                        gate = merged[:, ::2].clamp(max=7.0)
                        up = merged[:, 1::2].clamp(min=-7.0, max=7.0)
                        activation = (up + 1) * (gate * torch.sigmoid(gate * 1.702))
                        projected = F.linear(
                            activation,
                            expert(expert_root, "down_proj", which),
                            down_bias[which],
                        )
                        mixed.index_add_(
                            0,
                            token_rows,
                            (projected * weights[token_rows, slots, None]).float(),
                        )
                    x = residual + mixed.to(torch.bfloat16).reshape_as(residual)
                    del (
                        attn,
                        state,
                        attention,
                        n,
                        scores,
                        projected,
                        merged,
                        gate_bias,
                        down_bias,
                    )
                hidden = norm(x, "model.norm.weight")[:, -1].float()
                head = files[index["lm_head.weight"]].get_slice("lm_head.weight")
                logits = torch.cat(
                    [
                        F.linear(hidden, head[offset : offset + 4096].cuda().float())
                        for offset in range(0, cfg.vocab_size, 4096)
                    ],
                    dim=-1,
                )
                row = logits[0].cpu().numpy()
                np.save(output / f"reference-step{step}.npy", row)
                refs.append(row)
                token = int(row.argmax())
                if step < args.steps:
                    continuation.append(token)
                tokens = [token]
                position = total
                print(
                    f"HF checkpoint step {step}: top1={token}, {tok.decode([token])!r}",
                    flush=True,
                )
        del x, cache, rotary, hidden, logits, lut, mixed
    torch.cuda.empty_cache()
    reference_seconds = time.monotonic() - start
    input_path = output / "input.json"
    input_path.write_text(json.dumps([dict(prompt=ids, continuation=continuation)]))
    start = time.monotonic()
    subprocess.run(
        [
            str(ROOT / "target/release/examples/forward"),
            args.model,
            str(input_path),
            str(output / "rust"),
            "4096",
            args.devices,
        ],
        env=os.environ.copy(),
        cwd=ROOT,
        check=True,
        timeout=600,
    )
    rust_seconds = time.monotonic() - start
    reports = []
    for step, ref in enumerate(refs):
        got = np.fromfile(
            output / "rust" / f"case0-step{step}.bin", dtype="<f4"
        ).astype(np.float64)
        ref = ref.astype(np.float64)
        assert got.shape == ref.shape and np.isfinite(got).all()
        nrmse = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        cosine = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
        reports.append(
            dict(
                step=step,
                nrmse=nrmse,
                cosine=cosine,
                rust_top1=int(got.argmax()),
                hf_top1=int(ref.argmax()),
                passed=nrmse <= 0.02 and cosine >= 0.999,
            )
        )
    report = dict(
        model=args.model,
        prompt=args.prompt,
        devices=args.devices,
        prompt_ids=ids,
        continuation=continuation,
        reference="HF attention/RMSNorm/YaRN and streamed PyTorch MXFP4 experts",
        reference_seconds=reference_seconds,
        rust_seconds=rust_seconds,
        reports=reports,
        passed=all(r["passed"] for r in reports),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["passed"], "real checkpoint numerical gate failed; report saved"


if __name__ == "__main__":
    main()
