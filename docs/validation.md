# Single-GPU baseline validation and remaining numerical work

This document records the original single-GPU implementation before fused projections and TP were added. The current [multi-GPU validation](tensor-parallel.md) records separate results and source hashes. The baseline ran both local dense Qwen3 checkpoints, but **has not passed the complete original acceptance gate**. The strict logits verifier deliberately exits nonzero. The functional tests, operator checks, service checks, and sanitizer checks below passed on the current machine.

## Reproducible environment

Measurements were taken on 2026-10-06 with NVIDIA A100-SXM4-80GB, driver 560.35.03, CUDA Toolkit 12.6.68, Rust 1.99.0, cudarc 0.19.10, PyTorch 2.8.0+cu126, and Transformers 4.57.3. The reference checkout is mini-sglang commit `9a91cfafe754aa85daee49998176275667eb58f2`.

[Environment and checkpoint hashes](../results/environment.json) record source hashes, GPU identity, exact Toolkit/library paths, and checkpoint SHA-256 values. [Verification packages](../results/verification-packages.txt) record the separate Python environment. Python, PyTorch, FlashInfer, and mini-sglang are verification dependencies; the Rust inference binary does not execute Python.

Set the environment before using any GPU command:

```bash
source "$HOME/.cargo/env"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.6
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release --workspace --examples --locked
```

The [development guide](development.md) contains commands for constructing `.venv-verify`. Reports are written under `results/`. Large generated operator fixtures and binary/NumPy activation dumps remain available locally and are ignored by Git; the scripts regenerate them from recorded inputs. Exploratory numerical variants are kept separately under the ignored `results/experiments/` directory.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Rust formatting and Clippy, all targets | Passed, warnings denied | [Clippy](../results/clippy.log) |
| CPU tests | 22 passed | [Test log](../results/rust-tests.log) |
| Independent PyTorch operator fixtures | Passed | [Operator report](../results/kernels/report.json) |
| Compute Sanitizer memcheck | 0 errors | [Log](../results/sanitizer-memcheck.log) |
| Compute Sanitizer racecheck | 0 errors, 0 warnings, 0 hazards | [Log](../results/sanitizer-racecheck.log) |
| Checkpoint chat template vs HF | 6/6 text and token-ID arrays equal | [Template report](../results/template-report.json) |
| Real GPU HTTP/SSE integration | Passed | [Service report](../results/service-report.json) |
| CLI generation with Qwen3-1.7B | Passed | [Original events](../results/generation-1.7b.jsonl) |
| Offline CLI benchmark, cold/hot | Passed | [Original timings](../results/offline-bench.json) |
| Full-model logits vs HF | **Failed strict NRMSE gate** | Reports below |

CPU tests use a fake `ModelRunner` and cover prefill/decode alternation, chunk boundaries, admission under a tiny page pool, cancellation in waiting/prefill/decode, EOS, slow consumers, failure cleanup, and prefix reuse. Cache tests cover compressed radix splitting, active pins, duplicate prefix ownership, eviction, and 2,000 deterministic randomized request operations with ownership/reference/reservation checks. UTF-8 decoding tests include bytes split across tokens and incomplete final characters. Premature stream closure becomes an explicit error.

Operator fixtures cover BF16 GEMM and FP32 logits, non-aligned RMSNorm widths, embedding, residual, SwiGLU, split-half RoPE, KV scatter, last-row gather, and paged causal attention with GQA ratio 3. Attention mixes sequences of lengths 22 and 37 with a previously cached 17-token prefix and noncontiguous physical pages. Sanitizer runs execute the same complete operator fixture, rather than only a vector-add smoke test.

HTTP checks use a real checkpoint, 17-token prefill chunks, four active requests, eight waiting requests, and a 64 MiB KV pool (36 pages). They verify Chinese streaming/nonstreaming agreement, cached prompt reuse, text/token-ID equivalence, mixed sequence lengths, seeded sampling, explicit invalid-parameter errors, EOS and length termination, disconnect cancellation, and queue saturation. At idle, `free_pages + cached_pages == 36`. A separate 16 MiB/9-page pool rejects a permanently impossible request with 429 while remaining healthy. The EOS check returns `1`, then stops after two generated tokens, including EOS.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
.venv-verify/bin/python scripts/verify_kernels.py
.venv-verify/bin/python scripts/verify_template.py
.venv-verify/bin/python scripts/verify_service.py --device 1
compute-sanitizer --tool memcheck target/release/examples/verify_ops \
  results/kernels/input.json results/kernels/memcheck-output.json
compute-sanitizer --tool racecheck target/release/examples/verify_ops \
  results/kernels/input.json results/kernels/racecheck-output.json
```

## Strict model gate: still open

The independent reference uses HF Qwen3, BF16 weights/activations, SDPA math attention with FP32 intermediates, and teacher-forced continuation. Its LM head alone projects the same BF16 hidden states/weights in FP32, matching the engine's FP32-logits contract. TF32 and BF16 reduced-precision GEMM reductions are disabled. Rust cuBLAS also forbids reduced-precision split-K reductions. No reference activations or reference output corrections enter Rust execution.

Each checkpoint has three prompts (English, Chinese, and a longer repeated-text prompt), a prefill comparison, and four teacher-forced decode steps. Each case runs as a packed batch of three with complete prefill and with 17-token chunks: 30 comparisons per checkpoint, 60 total. NRMSE is `||rust - reference||₂ / ||reference||₂`, computed in FP64 for reporting; the gate remains NRMSE ≤ 0.02 and cosine ≥ 0.999. Top-1 IDs and reference candidate gaps are recorded separately.

| Model | Strict failures | Maximum NRMSE | Minimum cosine | Matching top-1 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1/30 | 0.0204743 | 0.9998211 | 30/30 |
| Qwen3-1.7B | 5/30 | 0.0287317 | 0.9996269 | 30/30 |

[0.6B original report](../results/model-0.6b/report.json) and [1.7B original report](../results/model-1.7b/report.json) retain every comparison and every failing row. Chunked/full prefill top-1 IDs agree for all tested rows. These results establish observed token agreement on these inputs; they do not establish general sequence agreement or satisfy the original numerical threshold.

```bash
# Both commands currently exit 1 after writing their complete reports.
.venv-verify/bin/python scripts/verify_model.py
.venv-verify/bin/python scripts/verify_model.py \
  --model /models/store/Qwen/Qwen3-1.7B --output results/model-1.7b

# Trace the earliest divergence, exporting independent HF and Rust activations.
.venv-verify/bin/python scripts/verify_trace.py
```

The [layer trace](../results/layer-trace/report.json) shows exact agreement for the first layer's embedding, input norm, Q projection, Q norm, V projection, and Q/K RoPE. Its attention output first differs in 29 of 47,104 BF16 elements, with NRMSE about 0.0000169. Later layers diverge further. This identifies an observed entry point for investigation; it does not prove that changing this single operator will resolve the complete gate. Aligning PyTorch and Rust to system cuBLAS 12.6.1 did not remove the difference. An additional HF FlashAttention comparison also retained strict failures. The implementation does not select different arithmetic per prompt or relax the verifier.

## Performance measurement

Both servers run sequentially on GPU 0 with Qwen3-0.6B BF16, context 4096, page size 16, 1,170 KV pages (~2 GiB), prefill budget 512, and at most 32 active requests. The workload is 16 requests, concurrency 8, exactly 128 input and 32 generated tokens, greedy with EOS ignored. The identical raw text bypasses chat templates. mini-sglang uses FlashInfer; CUDA Graph and overlap scheduling are disabled. Library/JIT warmup includes two waves at the measured batch shapes.

Cold runs use the naive cache, so no request can reuse a prefix. Hot runs enable radix and prime the common prompt before measurement. These definitions avoid conflating prefix-cache state with compiler or model-startup costs. HTTP arrival batching is determined by real request timing and is not artificially synchronized.

| System | Prefix cache | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | Output tokens/s | Sampled peak (GiB) | Measured cache-hit tokens |
|---|---|---:|---:|---:|---:|---:|
| mini-rsglang | naive | 37.91 / 54.15 | 7.42 / 12.33 | 854.1 | 3.58 | 0 |
| mini-rsglang | radix | 21.05 / 27.00 | 7.35 / 12.05 | 947.2 | 3.55 | 1792 |
| mini-sglang | naive | 30.95 / 37.93 | 8.81 / 9.83 | 779.1 | 3.86 | unavailable |
| mini-sglang | radix | 33.46 / 42.81 | 8.68 / 9.49 | 796.1 | 3.86 | unavailable |

The [raw comparison](../results/comparison.json) includes each server command, each request's TTFT/ITL/latency, cache configuration, memory samples, and Rust metrics before/after the measured interval. TTFT is measured from HTTP submission to the first token event; ITL is measured between token events at the client. Peak memory is sampled through `nvidia-smi` every 100 ms and subtracts the pre-server baseline; it is a sampled resident-memory peak, not an allocator-exact transient peak. mini-sglang's HTTP subset does not expose a cache-hit counter, so its hit count is explicitly unavailable rather than inferred.

```bash
uv pip install --python .venv-verify/bin/python -e ../mini-sglang ninja
.venv-verify/bin/python scripts/compare_mini.py
target/release/mini-rsglang bench --requests 16 --concurrency 8 \
  --input-tokens 128 --output-tokens 32 --output results/offline-bench.json
```

This is one short workload and one measurement per configuration. It is not a general performance ranking. Reference-library patches, scheduler arrival timing, graph replay, overlap, workload shape, and repeated trials can change results. The Rust backend currently allocates per-step intermediates and uses a straightforward SIMT attention kernel; reusable scratch buffers and optimized attention remain performance work.

The original baseline did not implement cuTile/cuda-oxide execution, multi-GPU, CUDA Graph, overlap scheduling, quantization, or MoE. TP and Qwen3 MoE are now implemented and tested in the separate [multi-GPU results](tensor-parallel.md); cuTile/cuda-oxide, CUDA Graph, overlap, and quantization remain future work. The [cuTile migration document](cutile.md) specifies the environment smoke test, owned buffers/stream integration, incremental operator replacement, and identical verification suite required before advertising that backend.
