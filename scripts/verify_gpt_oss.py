"""Independent HF GPT-OSS oracle for BF16/MXFP4, YaRN, sinks, windows, and TP."""

import argparse
import copy
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", default="1,2,4")
    parser.add_argument("--output", default="results/gpt-oss/fixtures")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    worlds = [int(s) for s in args.world_sizes.split(",")]
    env = os.environ.copy()
    env.setdefault(
        "RSGLANG_NCCL_LIBRARY",
        str(
            ROOT
            / ".venv-verify/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
        ),
    )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    reports = []
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.bfloat16,
    )
    for quantized in [False, True]:
        torch.manual_seed(29)
        cfg = GptOssConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=67,
            num_local_experts=4,
            num_experts_per_tok=2,
            eos_token_id=66,
            pad_token_id=0,
            max_position_embeddings=64,
            sliding_window=8,
            attention_bias=True,
            tie_word_embeddings=False,
            layer_types=["sliding_attention", "full_attention"],
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            },
        )
        model = GptOssForCausalLM(cfg).bfloat16().eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm.weight" in name:
                    param.uniform_(0.8, 1.2)
                elif "bias" in name or "sinks" in name:
                    param.normal_(0.0, 0.05)
        state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        if quantized:
            for name, param in model.named_parameters():
                if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                    experts, inputs, outputs = param.shape
                    packed = torch.randint(
                        0, 256, (experts, outputs, inputs // 32, 16), dtype=torch.uint8
                    )
                    scales = torch.full(
                        (experts, outputs, inputs // 32), 120, dtype=torch.uint8
                    )
                    decoded = torch.stack(
                        (lut[(packed & 15).long()], lut[(packed >> 4).long()]), dim=-1
                    ).flatten(-2)
                    decoded = (
                        torch.ldexp(decoded, scales.int().unsqueeze(-1) - 127)
                        .flatten(-2)
                        .transpose(1, 2)
                        .contiguous()
                    )
                    with torch.no_grad():
                        param.copy_(decoded)
                    del state[name]
                    state[name + "_blocks"] = packed
                    state[name + "_scales"] = scales
        name = "mxfp4" if quantized else "bf16"
        checkpoint = output / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.config.architectures = ["GptOssForCausalLM"]
        config = copy.deepcopy(model.config.to_dict())
        if quantized:
            config["quantization_config"] = {"quant_method": "mxfp4"}
        (checkpoint / "config.json").write_text(json.dumps(config, indent=2))
        save_file(state, checkpoint / "model.safetensors")
        model = model.cuda()
        model.set_attn_implementation("eager")
        head = model.lm_head.weight.detach().float()
        model.lm_head.forward = lambda hidden, weight=head: torch.nn.functional.linear(
            hidden.float(), weight
        )
        cases, refs = [], []
        with torch.inference_mode():
            for length in [7, 19, 35]:
                ids = [(i * 13 + 5) % 67 for i in range(length)]
                result = model(torch.tensor([ids], device="cuda"), use_cache=True)
                rows = [result.logits[0, -1].float().cpu().numpy()]
                continuation = []
                for _ in range(3):
                    token = int(result.logits[0, -1].argmax())
                    continuation.append(token)
                    result = model(
                        torch.tensor([[token]], device="cuda"),
                        past_key_values=result.past_key_values,
                        use_cache=True,
                    )
                    rows.append(result.logits[0, -1].float().cpu().numpy())
                cases.append(dict(prompt=ids, continuation=continuation))
                refs.append(rows)
        del model, head, result
        torch.cuda.empty_cache()
        input_path = checkpoint / "input.json"
        input_path.write_text(json.dumps(cases))
        for world in worlds:
            for mode, chunk in [("full", 256), ("chunked", 5)]:
                dest = checkpoint / f"tp{world}-{mode}"
                subprocess.run(
                    [
                        str(ROOT / "target/release/examples/forward"),
                        str(checkpoint),
                        str(input_path),
                        str(dest),
                        str(chunk),
                        ",".join(map(str, range(world))),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    timeout=180,
                )
                for case, rows in enumerate(refs):
                    for step, ref in enumerate(rows):
                        got = np.fromfile(
                            dest / f"case{case}-step{step}.bin", dtype="<f4"
                        ).astype(np.float64)
                        ref = ref.astype(np.float64)
                        assert got.shape == ref.shape and np.isfinite(got).all()
                        nrmse = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
                        cosine = float(
                            np.dot(got, ref)
                            / (np.linalg.norm(got) * np.linalg.norm(ref))
                        )
                        row = dict(
                            dtype=name,
                            tp=world,
                            mode=mode,
                            case=case,
                            step=step,
                            nrmse=nrmse,
                            cosine=cosine,
                            top1_equal=int(got.argmax()) == int(ref.argmax()),
                        )
                        row["passed"] = (
                            nrmse <= 0.02 and cosine >= 0.999 and row["top1_equal"]
                        )
                        if mode == "chunked":
                            full = np.fromfile(
                                checkpoint
                                / f"tp{world}-full"
                                / f"case{case}-step{step}.bin",
                                dtype="<f4",
                            ).astype(np.float64)
                            row["chunked_top1_equal"] = int(got.argmax()) == int(
                                full.argmax()
                            )
                            row["chunked_max_abs_error"] = float(
                                np.max(np.abs(got - full))
                            )
                            row["passed"] &= row["chunked_top1_equal"]
                        reports.append(row)
                print(f"{name} TP={world} {mode}: compared", flush=True)
                summary = dict(
                    seed=29,
                    torch=torch.__version__,
                    world_sizes=worlds,
                    attention="HF eager with sink",
                    rope="YaRN",
                    window=8,
                    passed=all(r["passed"] for r in reports),
                    reports=reports,
                )
                (output.parent / "fixture-report.json").write_text(
                    json.dumps(summary, indent=2)
                )
    failures = [r for r in reports if not r["passed"]]
    print(
        json.dumps(
            dict(
                comparisons=len(reports),
                failures=len(failures),
                max_nrmse=max(r["nrmse"] for r in reports),
                min_cosine=min(r["cosine"] for r in reports),
                top1_matches=sum(r["top1_equal"] for r in reports),
            ),
            indent=2,
        )
    )
    assert not failures, "GPT-OSS numerical gate failed; report saved"


if __name__ == "__main__":
    main()
