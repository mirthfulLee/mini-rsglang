# Tensor-parallel execution

The reference is the local mini-sglang checkout at commit `9a91cfafe754aa85daee49998176275667eb58f2`. It implements tensor parallelism (TP), including TP inside MoE experts. It does not implement pipeline, data, sequence, or expert parallelism. Its TorchDistributed and PyNCCL implementations are two communication backends for the same partitioning algorithms.

## Coverage

| mini-sglang operation | mini-rsglang implementation |
|---|---|
| Vocabulary-parallel embedding | Slice checkpoint rows before upload; mask nonlocal IDs; FP32 sum and BF16 cast |
| Vocabulary-parallel LM head | Local FP32 logits; NCCL all-gather; reorder `[rank,batch,local_vocab]` into `[batch,vocab]`; remove padding |
| QKV column parallelism | Slice Q/K/V separately, fuse local projection rows, GEMM and split activations |
| GQA with fewer KV heads than ranks | Adjacent ranks replicate one KV head; query heads remain partitioned |
| Merged gate/up column parallelism | Slice each segment separately, fuse local rows, GEMM and SwiGLU |
| Row-parallel output/down projections | Slice input columns; FP32 GEMM partials and collective sum; cast once to BF16 |
| Replicated normalization and router | Read-only rank-local copies |
| MoE tensor parallelism | Keep every expert on every rank; partition each expert's intermediate width; weighted local mixture and one final collective sum |
| Collective transport | Native NCCL 2.27 ABI loaded dynamically; one explicit compute/communication stream per GPU |

Supported checkpoints are Qwen3 dense and Qwen3Moe. This covers the reference's parallel partitioning algorithms; it does not add its other model architectures, FlashInfer/CUTLASS expert kernels, CUDA Graph, overlap scheduling, or optional symmetric-window registration optimization. Expert routing executes on the GPU, then downloads selected IDs/weights to group expert rows on the CPU. Expert arithmetic and mixture accumulation execute on the GPU. This approach favors readability over MoE throughput.

## Use

Install a native NCCL runtime >= 2.27, compatible with the installed CUDA/driver, and expose `libnccl.so.2` through the loader search path or `RSGLANG_NCCL_LIBRARY`. Single-GPU execution does not load NCCL. The verification environment happens to provide NCCL 2.27.3; inference itself never invokes Python.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# Alternatively point at a system NCCL installation.
export RSGLANG_NCCL_LIBRARY="$PWD/.venv-verify/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2"
cargo build --release --workspace --bins --examples --locked

# This machine has NVLink pairs (0,1), (2,3), (4,5), (6,7), with PCIe/SYS between pairs.
# Unrestricted cross-PCIe P2P failed the communication test on this machine.
# Restrict P2P to NVLink; NCCL selects other transports between pairs.
export NCCL_P2P_LEVEL=NVL

target/release/mini-rsglang --tensor-parallel-size 2 generate \
  --chat --prompt '请用一句话解释所有权。' --max-tokens 32

target/release/mini-rsglang --devices 2,3 serve --port 8000

target/release/mini-rsglang --model /models/store/Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 8 generate --chat --prompt '你好！' --max-tokens 16
```

`--device N --tensor-parallel-size T` selects contiguous CUDA ordinals `N..N+T`. `--devices` selects explicit distinct ordinals and infers TP size; an explicit TP size must agree. Ordinals are relative to `CUDA_VISIBLE_DEVICES`. Query heads and intermediate widths must divide TP size. KV head count must divide TP size or vice versa. Vocabulary width can be uneven, including ranks holding padding only. Invalid layouts/devices fail before loading weights.

`--kv-mib` is **per GPU**. Every rank has the same logical page count, computed using its local KV head count; replicated heads stop reducing KV usage when TP exceeds the original KV head count. The public metrics' peak-memory value is the sum of rank device-memory observations, not a per-GPU value. The HTTP comparison script additionally samples each physical GPU.

```rust,ignore
let engine = rsglang_runtime::EngineHandle::load_tensor_parallel(
    std::path::Path::new("/models/store/Qwen/Qwen3-0.6B"),
    &[0, 1], 2 * 1024 * 1024 * 1024,
    rsglang_core::RuntimeConfig::default(),
)?;
// generate(), cancellation, shutdown(), and HTTP events have the same contract as single GPU.
```

No environment variables or host drivers are modified by the library. The NVL setting above is an observed machine-specific workaround; it must be supplied before NCCL initialization. A different topology should first run the collective smoke test with its normal NCCL settings. See [NCCL environment controls](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2276/user-guide/docs/env.html) for the transport settings. Tests that disable P2P entirely were diagnostic, not the configuration used for model acceptance.

cuBLAS projections run in fixed 32-token-row blocks, discarding padded rows. Keeping the projection shape fixed prevents its algorithm selection from changing merely because admission or a prefill chunk changed the batch length. This resolved the observed single-request/mixed-batch greedy discrepancy in the service fixture. Two ranks use native NCCL FP32 all-reduce. Three or more ranks all-gather partials and sum them in fixed rank order on the GPU; this avoids NCCL ring owner or algorithm changes affecting FP32 rounding with different batch shapes. It trades bandwidth and scratch storage for reproducibility.

The fixed GEMM shape adds padding/copy/launch overhead, especially for small decode batches; optimizing this must preserve the behavioral and numerical checks.

## Ownership and failure handling

`rsglang-distributed` is a new CPU-only crate: partition geometry, checkpoint slicing, and `RankGroup`. Each rank worker constructs and exclusively owns its engine, model shard, physical KV, stream, communicator, and temporary tensors. One runtime scheduler owns all logical request/page/radix state. It broadcasts the same owned CPU `StepBatch` to every rank through bounded channels. Only rank 0 samples and emits tokens. The scheduler commits the step after every rank replies and every rank has synchronized its stream.

Shared prefixes remain immutable full pages. A logical page ID identifies the corresponding head shard on every GPU; KV-head replication uses the same logical mapping. Active requests pin shared pages, and their new tokens write only private pages. Mutable access to a rank's entire KV allocation is owned by that rank's worker; metadata checks reject overlapping write slots. Concurrent requests run together within one ordered batch rather than independently borrowing the same allocation. Cancellation/eviction releases a logical page only after the complete multi-rank step finishes.

NCCL owns separate reusable communication allocations (`ncclMemAlloc`/`ncclMemFree`, [NCCL allocation source](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/allocator.cc)). Stream-ordered device copies move between safe cudarc tensors and those allocations; no external code sees their pointers. Growth synchronizes previous uses before freeing storage. Allocation and collective bounds are checked, and pointer guards preserve tensor lifetimes. All unsafe code remains inside `rsglang-kernels`.

A rank initialization, model, collective, or sampling error poisons the group, aborts all registered communicators concurrently, rejects subsequent steps, and joins rank threads. Nonblocking NCCL initialization/enqueue polling and host group deadlines avoid waiting indefinitely for missing peers. The runtime then uses its existing error cleanup to fail streams and release scheduler ownership. The native communication test injects an error before peers' next collective to exercise this path.

## Reproduce verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked

# Explicit device ordering also supports noncontiguous groups.
target/release/examples/collectives 0,1
target/release/examples/collectives 0,1,2,3
target/release/examples/collectives 0,1,2,3,4,5,6,7
.venv-verify/bin/python scripts/verify_tensor_parallel.py

# These retain the original NRMSE <= .02 / cosine >= .999 gate, and exit nonzero on failures.
.venv-verify/bin/python scripts/verify_model.py --world-sizes 1,2,4,8 \
  --output results/tensor-parallel/model-0.6b
.venv-verify/bin/python scripts/verify_model.py --model /models/store/Qwen/Qwen3-1.7B \
  --world-sizes 1,2,4,8 --output results/tensor-parallel/model-1.7b
.venv-verify/bin/python scripts/verify_model.py --model /models/store/Qwen/Qwen3-30B-A3B \
  --world-sizes 8 --steps 1 --output results/tensor-parallel/model-30b-a3b

.venv-verify/bin/python scripts/verify_kernels.py --output results/tensor-parallel/kernels
cargo test --release -p rsglang-kernels rank_ordered_sum_cancellation_and_bounds -- --ignored
.venv-verify/bin/python scripts/verify_service.py --devices 0,1 \
  --output results/tensor-parallel/service-tp2.json
.venv-verify/bin/python scripts/verify_service.py --devices 0,1,2,3,4,5,6,7 \
  --output results/tensor-parallel/service-tp8.json

# Same model, devices, precision, page counts and workload, sequential server runs.
.venv-verify/bin/python scripts/compare_mini.py --tp-size 2 \
  --output results/tensor-parallel/comparison-tp2.json
```

The synthetic HF fixtures use seed 19, tied embeddings, vocabulary 17, eight Q heads, two KV heads, explicit head dimension 8, and 7/19/35-token prompts. TP 8 therefore exercises KV replication and empty vocabulary ranks. The MoE fixture includes both a sparse and a dense layer, four experts, and top-2 routing. Full/5-token chunked prefill and three teacher-forced steps produce 192 comparisons across TP 1/2/4/8, all passing the original numerical gate. Checkpoints/binary dumps are ignored by Git and regenerated by the script.

Real-model evidence and remaining numerical failures are recorded separately below; a passing small fixture does not imply all full-size model logits pass.

## Executed results (2026-10-06)

The source/environment manifest is [environment.json](../results/tensor-parallel/environment.json). The reference is the unchanged neighboring mini-sglang commit noted above. Final commands and return codes are in [final verification](../results/tensor-parallel/final-verification-report.json); full HF numerical runs, including their intentional nonzero gate exits, are in [numerical execution](../results/tensor-parallel/deterministic-execution-report.json).

| Check | Result | Evidence |
|---|---|---|
| CPU workspace tests | 28 passed; one GPU-only test ignored by ordinary cargo test | [Rust tests](../results/tensor-parallel/rust-tests.log) |
| Formatting / Clippy, all targets | Passed, warnings denied | [Clippy](../results/tensor-parallel/clippy.log) |
| Native collectives, TP 2/4/8 and reversed devices 7,6 | Passed; includes cancellation-sensitive rank sums and peer-failure cleanup | [TP 8](../results/tensor-parallel/native-collectives-tp8.log) |
| Synthetic dense/MoE HF comparison | 192/192 passed; all full/chunked logits bitwise equal | [Fixture report](../results/tensor-parallel/fixture-report.json) |
| PyTorch operator comparison, including routing/scatter and fused projections | Passed | [Operator report](../results/tensor-parallel/kernels/report.json) |
| Custom operator memcheck / racecheck | 0 errors / 0 hazards | [Memcheck](../results/tensor-parallel/sanitizer-memcheck-verification.log), [racecheck](../results/tensor-parallel/sanitizer-racecheck-verification.log) |
| Fixed-order FP32 sum GPU test, memcheck / racecheck | Passed; 0 errors / 0 hazards | [Sum memcheck](../results/tensor-parallel/ranked-sum-memcheck.log), [sum racecheck](../results/tensor-parallel/ranked-sum-racecheck.log) |
| Real HTTP/SSE, TP 2 and 8 | Passed: Chinese streaming, mixed batching, shared prefixes, seed, EOS/length, disconnect, queue full and page conservation | [TP 2](../results/tensor-parallel/service-tp2.json), [TP 8](../results/tensor-parallel/service-tp8.json) |
| Real Qwen3-30B-A3B, TP 8, CLI generation | 16 tokens and a length finish; output agrees before/after explicit NCCL grouping | [Generation events](../results/tensor-parallel/generation-final-30b-a3b-tp8.log) |
| Native NCCL communication under Compute Sanitizer | **Not passed**; all-gather value assertions fail under instrumentation | [Grouped run](../results/tensor-parallel/collectives-group-memcheck.log), [unfiltered API diagnostics](../results/tensor-parallel/collectives-outplace-memcheck-verification.log) |

The unfiltered NCCL instrumented run reports runtime API errors (kernel-image checks, already-enabled peer access and FABRIC allocation fallback), but disabling API-error reporting still leaves an all-gather value mismatch. A separate Simple-protocol/SHM transport run also failed its value assertions. These are retained as failures, not classified as harmless false positives. Native runs and full-model runs pass their communication/value checks without instrumentation. The custom kernels, including the new fixed-rank sum, passed both sanitizer tools independently. This leaves an explicit communication-instrumentation investigation open.

The final dense-model TP-8 reruns after adding explicit NCCL groups produced byte-identical full/chunked outputs to the numerical reports. Small changes to wrapper validation/grouping did not require changing the HF reference.

## Full-model numerical gate

NRMSE remains `||Rust-HF||₂/||HF||₂ <= .02`, cosine remains `>= .999`, and both verifiers exit nonzero for failures. HF uses BF16 weights/activations and FP32 final projection, SDPA math, no TF32 and no reduced-precision BF16 reductions. No threshold was relaxed and no reference values enter inference.

| Model | TP | Failing rows | Maximum NRMSE | Minimum cosine | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6b | 1 | 2/30 | 0.0206975 | 0.9998830 | 30/30 |
| Qwen3-0.6b | 2 | 4/30 | 0.0230787 | 0.9997466 | 30/30 |
| Qwen3-0.6b | 4 | 0/30 | 0.0196592 | 0.9998142 | 30/30 |
| Qwen3-0.6b | 8 | 0/30 | 0.0154212 | 0.9998888 | 30/30 |
| Qwen3-1.7b | 1 | 6/30 | 0.0597559 | 0.9986379 | 30/30 |
| Qwen3-1.7b | 2 | 4/30 | 0.0354666 | 0.9994855 | 30/30 |
| Qwen3-1.7b | 4 | 6/30 | 0.0426030 | 0.9992949 | 30/30 |
| Qwen3-1.7b | 8 | 4/30 | 0.0290443 | 0.9996811 | 30/30 |
| Qwen3-30b-a3b | 8 | 12/12 | 0.0966207 | 0.9962682 | 12/12 |

[0.6B](../results/tensor-parallel/model-0.6b/report.json), [1.7B](../results/tensor-parallel/model-1.7b/report.json), and [30B-A3B](../results/tensor-parallel/model-30b-a3b/report.json) retain every error, candidate gap, and top-1 ID. Dense models use four teacher-forced steps; the real 30B-A3B run uses one step per prompt, explicitly recorded in its commands. Together they contain 252 comparisons, all matching top-1. Full/chunked logits are bitwise equal for every tested row at every TP size. This is observed agreement on these inputs, not general numerical acceptance: 1.7B and real MoE still fail the original gate, as do some 0.6B TP-1/2 rows.

## Measured performance and limits

Final Rust library benchmarks use Qwen3-0.6B BF16, 16 requests, concurrency 8, 128 input and 32 output tokens, two warmup waves at the measured shapes, context 4096, page size 16 and 2 GiB KV **per GPU**. Cold disables prefix caching; hot primes and shares the same prompt. Warmup is excluded from timings and cache-hit deltas. Timings are measured at the Rust event stream, not over HTTP. Peak memory is the sum of rank device-memory observations; it is not a per-GPU allocation peak.

| TP | Cache | TTFT p50/p95 ms | ITL p50/p95 ms | Output tokens/s | Peak sum GiB | Hit tokens |
|---:|---|---:|---:|---:|---:|---:|
| 1 | disabled | 106.80/107.56 | 8.76/8.99 | 675.4 | 3.851 | 0 |
| 1 | warm_prefix | 18.90/18.94 | 8.59/15.76 | 848.5 | 3.851 | 1792 |
| 2 | disabled | 188.42/189.32 | 31.00/38.41 | 216.6 | 7.684 | 0 |
| 2 | warm_prefix | 43.84/101.89 | 31.53/44.94 | 237.9 | 7.723 | 1792 |
| 4 | disabled | 162.24/178.88 | 30.08/37.33 | 225.9 | 12.357 | 0 |
| 4 | warm_prefix | 80.98/107.45 | 34.55/38.87 | 217.1 | 12.342 | 1792 |
| 8 | disabled | 358.67/533.08 | 49.10/71.14 | 126.5 | 23.174 | 0 |
| 8 | warm_prefix | 77.36/209.73 | 60.63/77.90 | 123.7 | 23.018 | 1792 |

Raw timings are in [TP 1](../results/tensor-parallel/bench-tp1.json), [TP 2](../results/tensor-parallel/bench-tp2.json), [TP 4](../results/tensor-parallel/bench-tp4.json), and [TP 8](../results/tensor-parallel/bench-tp8.json). This small model slows down with more GPUs on this topology: rank-thread/collective overhead, fixed GEMM blocks, ordered sums, and intermediate allocations outweigh sharding benefits. The results do not claim strong scaling or parity with mini-sglang.

The independent HTTP comparison script also runs mini-sglang sequentially on the same physical devices, with matching logical page counts, precision and token lengths, and samples each GPU. Attempts with both PyNCCL and TorchDistributed retained partial measurements but mini-sglang stalled in Gloo broadcasts on multi-rank workloads. Its TP-2 cold PyNCCL run completed (386.9 output tokens/s), while its hot run timed out; TP-4/8 runs did not complete all cache configurations. These trials preceded the final ordered-sum policy and are historical diagnostics, not a final performance ranking.

[PyNCCL trials](../results/tensor-parallel/comparison-tp2.json), [TorchDistributed trials](../results/tensor-parallel/comparison-torch-tp2.json), and [reference failure log](../results/tensor-parallel/compare-tp2-mini-sglang-radix.log) record commands, partial results and failures. The neighboring source was not modified. Use `--mini-disable-pynccl` to select its alternate communication backend. Reference timing cells that were not completed are unavailable; they are not inferred from Rust timings.

CUDA Graph, overlap, expert parallelism, fused expert kernels, symmetric-window registration, and cuTile remain future work. Optimizing GEMM blocks and reductions must preserve the single/batched, full/chunked and cache-on/off behavior checks. Exploratory pre-stability reports are preserved locally under the ignored `results/experiments/` directory; final reports above are the reviewable results.
