# Rust-native kernel migration

The CUDA 12.6 backend is the first runnable implementation. No empty cuTile crate or advertised-but-unimplemented feature is included.

Current [cuTile Rust](https://github.com/NVlabs/cutile-rs) supports A100/sm_80 but requires CUDA Tile IR 13.2 or newer. The current machine has Toolkit 12.6 and driver 560.35.03, so this is a separate environment transition. [cuda-oxide](https://github.com/NVIDIA/cuda-rust) currently also requires CUDA 13 and its pinned nightly toolchain. Installing a Toolkit alone must not be treated as proof that the existing driver can execute the new backend.

Implementation sequence:

1. Establish a compatible CUDA/driver environment and pin the cuTile dependency revision. Run vector addition and BF16 GEMM on the intended GPU, then memcheck/racecheck. Record actual compiler, driver and architecture versions.
2. Implement a `KernelBackend` using cuTile-native tensor, logits and metadata types. The model and engine remain generic over the associated types; CUDA C pointers and cudarc types must not escape into their APIs. Keep completion/lifetime guarantees identical.
3. Replace RMSNorm, residual, SwiGLU, embedding and RoPE first. Run the independent operator fixtures and the strict model report after each replacement; preserve the observed numerical differences instead of weakening acceptance.
4. Implement paged prefill/decode GQA and KV scatter with explicit disjoint output partitions. Shared prefix pages must be read-only. Backend-specific metadata can be constructed from the same validated `StepBatch`.
5. Retain cuBLAS for GEMM through a narrow, stream-ordered interop adapter initially. Validate context, ownership and completion across that boundary. cuTile GEMM replacement is a later performance experiment.
6. Run identical end-to-end workloads and compare raw TTFT, ITL, throughput, cache reuse, and peak device memory. Make backend selection explicit only once both implementations really run.

Prefer cuTile for regular tiled operations and its ownership-aware launch surface. Investigate cuda-oxide only for SIMT work that does not fit those operations; pin its compiler separately and preserve checked launch contracts. Neither migration introduces a second scheduler, cache policy, model implementation, or global context.
