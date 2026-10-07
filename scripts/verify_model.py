"""Independent BF16 Qwen3 reference; requires only the verification Python environment."""

import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/models/store/Qwen/Qwen3-0.6B")
    p.add_argument("--output", default="results/model-0.6b")
    p.add_argument("--devices", default="0")
    p.add_argument(
        "--world-sizes", help="verify contiguous GPUs 0..N-1 at each TP size"
    )
    p.add_argument("--steps", type=int, default=4)
    a = p.parse_args()
    output = ROOT / a.output
    output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    # The engine contract is BF16 activations/weights with FP32 final logits.
    # HF normally rounds lm_head output to BF16; use an independent FP32 projection
    # of the same BF16 values so the oracle observes the same output precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    head_weight = model.lm_head.weight.detach().float()
    model.lm_head.forward = lambda hidden, weight=head_weight: (
        torch.nn.functional.linear(hidden.float(), weight)
    )
    texts = [
        "Explain why the sky is blue in one sentence.",
        "请用中文解释什么是所有权。",
        "Describe a tree. " * 40,
    ]
    cases, refs, reports, failures = [], [], [], []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for i, text in enumerate(texts):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            x = torch.tensor([ids], device="cuda")
            result = model(x, use_cache=True)
            rows = [result.logits[0, -1].float().cpu().numpy()]
            cont = []
            for _ in range(a.steps):
                token = int(result.logits[0, -1].argmax())
                cont.append(token)
                result = model(
                    torch.tensor([[token]], device="cuda"),
                    past_key_values=result.past_key_values,
                    use_cache=True,
                )
                rows.append(result.logits[0, -1].float().cpu().numpy())
            cases.append({"prompt": ids, "continuation": cont})
            refs.append(rows)
    for i, rows in enumerate(refs):
        for step, row in enumerate(rows):
            np.save(output / f"ref-case{i}-step{step}.npy", row)
    del model, head_weight
    torch.cuda.empty_cache()
    (output / "input.json").write_text(json.dumps(cases))
    cargo = str(pathlib.Path.home() / ".cargo/bin/cargo")
    layouts = (
        [a.devices]
        if not a.world_sizes
        else [",".join(map(str, range(int(n)))) for n in a.world_sizes.split(",")]
    )
    for devices in layouts:
        run_output = (
            output if not a.world_sizes else output / f"tp{len(devices.split(chr(44)))}"
        )
        for mode, chunk in [("full", 4096), ("chunked", 17)]:
            dest = run_output / mode
            run(
                cargo,
                "run",
                "--release",
                "-p",
                "rsglang-engine",
                "--example",
                "forward",
                "--",
                a.model,
                output / "input.json",
                dest,
                chunk,
                devices,
            )
            for i, rows in enumerate(refs):
                for step, ref in enumerate(rows):
                    got = np.fromfile(dest / f"case{i}-step{step}.bin", dtype="<f4")
                    assert got.shape == ref.shape
                    assert np.isfinite(got).all()
                    error = got.astype(np.float64) - ref.astype(np.float64)
                    nrmse = float(
                        np.linalg.norm(error) / np.linalg.norm(ref.astype(np.float64))
                    )
                    cosine = float(
                        np.dot(got.astype(np.float64), ref.astype(np.float64))
                        / (
                            np.linalg.norm(got.astype(np.float64))
                            * np.linalg.norm(ref.astype(np.float64))
                        )
                    )
                    order = np.argsort(ref)[-2:]
                    gap = float(ref[order[-1]] - ref[order[-2]])
                    max_error = float(np.max(np.abs(error)))
                    top1_equal = int(got.argmax()) == int(ref.argmax())
                    report = dict(
                        mode=mode,
                        tp_size=len(devices.split(",")),
                        case=i,
                        step=step,
                        prompt_tokens=len(cases[i]["prompt"]),
                        nrmse=nrmse,
                        cosine=cosine,
                        max_abs_error=max_error,
                        hf_top1=int(ref.argmax()),
                        rust_top1=int(got.argmax()),
                        top1_equal=top1_equal,
                        hf_top1_gap=gap,
                    )
                    reports.append(report)
                    report["numeric_passed"] = nrmse <= 0.02 and cosine >= 0.999
                    if not report["numeric_passed"]:
                        failures.append(report)
                    if gap > 2 * max_error:
                        if not top1_equal:
                            failures.append(report)
                    if mode == "chunked":
                        full = np.fromfile(
                            run_output / "full" / f"case{i}-step{step}.bin", dtype="<f4"
                        )
                        report["chunked_logits_exact"] = bool(np.array_equal(full, got))
                        report["chunked_max_abs_error"] = float(
                            np.max(np.abs(full - got))
                        )
                        report["chunked_top1_equal"] = int(full.argmax()) == int(
                            got.argmax()
                        )
                        if not report["chunked_top1_equal"]:
                            failures.append(report)
    (output / "report.json").write_text(
        json.dumps(
            {
                "model": a.model,
                "gemm_token_rows": 32,
                "device_layouts": layouts,
                "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL"),
                "torch": torch.__version__,
                "reference_attention": "sdpa-math",
                "reference_logits": "fp32-projection",
                "bf16_reduced_precision_reduction": False,
                "allow_tf32": False,
                "thresholds": {"max_nrmse": 0.02, "min_cosine": 0.999},
                "reports": reports,
                "failures": failures,
                "passed": not failures,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "comparisons": len(reports),
                "max_nrmse": max(r["nrmse"] for r in reports),
                "min_cosine": min(r["cosine"] for r in reports),
                "top1_equal": sum(r["top1_equal"] for r in reports),
                "failures": len(failures),
            },
            indent=2,
        )
    )
    assert not failures, (
        f"{len(failures)} comparisons failed; full evidence saved to {output}/report.json"
    )


if __name__ == "__main__":
    main()
