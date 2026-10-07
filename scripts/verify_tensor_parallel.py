"""Independent HF oracle for dense/MoE TP, replicated KV heads and odd vocab padding."""

import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", default="1,2,4,8")
    parser.add_argument("--output", default="results/tensor-parallel/fixtures")
    args = parser.parse_args()
    worlds = [int(s) for s in args.world_sizes.split(",")]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault(
        "RSGLANG_NCCL_LIBRARY",
        str(
            ROOT
            / ".venv-verify/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
        ),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(1)
    reports = []
    failures = []

    def save():
        report = dict(
            gemm_token_rows=32,
            seed=19,
            torch=torch.__version__,
            world_sizes=worlds,
            attention="sdpa-math",
            reference_logits="fp32-projection",
            bf16_reduced_precision_reduction=False,
            nccl_p2p_level=env.get("NCCL_P2P_LEVEL"),
            reports=reports,
            failures=failures,
            passed=len(reports) == len(worlds) * 48 and not failures,
        )
        (output.parent / "fixture-report.json").write_text(json.dumps(report, indent=2))

    for sparse in [False, True]:
        torch.manual_seed(19)
        common = dict(
            vocab_size=17,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            rope_theta=1000000.0,
            eos_token_id=16,
            tie_word_embeddings=True,
            sliding_window=None,
        )
        if sparse:
            cfg = Qwen3MoeConfig(
                **common,
                moe_intermediate_size=32,
                num_experts=4,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                mlp_only_layers=[1],
                decoder_sparse_step=1,
            )
            model = Qwen3MoeForCausalLM(cfg).bfloat16()
        else:
            cfg = Qwen3Config(**common)
            model = Qwen3ForCausalLM(cfg).bfloat16()
        name = "moe" if sparse else "dense"
        checkpoint = output / name
        model.save_pretrained(checkpoint, safe_serialization=True)
        model = model.cuda().eval()
        model.set_attn_implementation("sdpa")
        head = model.lm_head.weight.detach().float()
        model.lm_head.forward = lambda hidden, weight=head: torch.nn.functional.linear(
            hidden.float(), weight
        )
        cases = []
        refs = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for length in [7, 19, 35]:
                ids = [(i * 7 + 3) % 17 for i in range(length)]
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
                cmd = [
                    str(ROOT / "target/release/examples/forward"),
                    str(checkpoint),
                    str(input_path),
                    str(dest),
                    str(chunk),
                    ",".join(map(str, range(world))),
                ]
                subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=120)
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
                        order = np.argsort(ref)
                        gap = float(ref[order[-1]] - ref[order[-2]])
                        max_error = float(np.max(np.abs(got - ref)))
                        equal = int(got.argmax()) == int(ref.argmax())
                        row = dict(
                            model=name,
                            tp_size=world,
                            mode=mode,
                            case=case,
                            step=step,
                            nrmse=nrmse,
                            cosine=cosine,
                            top1_equal=equal,
                            hf_top1_gap=gap,
                            max_abs_error=max_error,
                        )
                        row["passed"] = (
                            nrmse <= 0.02
                            and cosine >= 0.999
                            and (equal or gap <= 2 * max_error)
                        )
                        if mode == "chunked":
                            full = np.fromfile(
                                checkpoint
                                / f"tp{world}-full"
                                / f"case{case}-step{step}.bin",
                                dtype="<f4",
                            ).astype(np.float64)
                            row["chunked_logits_exact"] = bool(
                                np.array_equal(full, got)
                            )
                            row["chunked_max_abs_error"] = float(
                                np.max(np.abs(full - got))
                            )
                            row["chunked_top1_equal"] = int(full.argmax()) == int(
                                got.argmax()
                            )
                            row["passed"] &= row["chunked_top1_equal"]
                        reports.append(row)
                        if not row["passed"]:
                            failures.append(row)
                save()
                print(f"{name} TP={world} {mode}: exported and compared", flush=True)
    save()
    print(
        json.dumps(
            dict(
                comparisons=len(reports),
                max_nrmse=max(r["nrmse"] for r in reports),
                min_cosine=min(r["cosine"] for r in reports),
                top1_matches=sum(r["top1_equal"] for r in reports),
                failures=len(failures),
            ),
            indent=2,
        )
    )
    assert not failures, "TP fixture numerical gate failed; report saved"


if __name__ == "__main__":
    main()
