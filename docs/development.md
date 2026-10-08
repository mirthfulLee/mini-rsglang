# Development and verification

## CPU checks

The pinned Rust toolchain includes rustfmt and Clippy. CUDA/NCCL libraries load
dynamically, so these checks can run without a GPU. GitHub Actions runs the same
Rust and Python checks.

```bash
source "$HOME/.cargo/env"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uvx ruff==0.16.10 check scripts
uvx ruff==0.16.10 format --check scripts
```

Use `cargo fmt --all` and `uvx ruff==0.16.10 format scripts` to format changes.
Ruff's scope is the independent verification scripts; it checks imports, syntax,
and undefined/unused names. Ordinary Rust tests exclude the explicitly ignored
GPU test.

## GPU build

Use CUDA Toolkit 12.6, a compatible NVIDIA driver, and one of the GPU
architectures listed in the README. Set the environment before GPU commands:

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.6
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release --workspace --bins --examples --locked
```

For multiple GPUs, expose NCCL >= 2.27 through the loader search path or
`RSGLANG_NCCL_LIBRARY`. See [tensor parallelism](tensor-parallel.md) for device
selection and the topology-specific `NCCL_P2P_LEVEL=NVL` setting used for the
recorded runs.

## Independent reference environment

Python dependencies are used only for independent validation. They are not
required by the Rust inference binary. `uv` is used here to create a separate
Python 3.11 environment.

```bash
uv venv .venv-verify --python 3.11
uv pip install --python .venv-verify/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-verify/bin/python -r scripts/requirements-verify.txt
```

Scripts default to local Qwen3 checkpoints under `/models/store/Qwen` and write
reports under `results/`. Inspect each script's `--help` for checkpoint, device,
and output options. Use a separate output directory when recording a new run
so the historical evidence stays available.

## Single-GPU checks

```bash
.venv-verify/bin/python scripts/verify_kernels.py
.venv-verify/bin/python scripts/verify_template.py
.venv-verify/bin/python scripts/verify_service.py --device 0

# These write complete reports and exit nonzero if the strict logits gate fails.
.venv-verify/bin/python scripts/verify_model.py
.venv-verify/bin/python scripts/verify_model.py \
  --model /models/store/Qwen/Qwen3-1.7B --output results/model-1.7b

# Export independent HF/Rust layer activations to investigate divergence.
.venv-verify/bin/python scripts/verify_trace.py
```

Kernel fixtures must be generated before running Compute Sanitizer:

```bash
compute-sanitizer --tool memcheck target/release/examples/verify_ops \
  results/kernels/input.json results/kernels/memcheck-output.json
compute-sanitizer --tool racecheck target/release/examples/verify_ops \
  results/kernels/input.json results/kernels/racecheck-output.json
```

The public Rust API can be exercised separately:

```bash
cargo run --release --locked -p rsglang-runtime --example generate -- \
  /models/store/Qwen/Qwen3-0.6B
```

## Multiple GPUs

The collective smoke test runs native transport/value checks, including rank
failure cleanup. Synthetic fixtures cover dense/MoE sharding, replicated KV
heads, and vocabulary padding.

```bash
target/release/examples/collectives 0,1
target/release/examples/collectives 0,1,2,3
target/release/examples/collectives 0,1,2,3,4,5,6,7
.venv-verify/bin/python scripts/verify_tensor_parallel.py

.venv-verify/bin/python scripts/verify_model.py --world-sizes 1,2,4,8 \
  --output results/tensor-parallel/model-0.6b
.venv-verify/bin/python scripts/verify_service.py --devices 0,1 \
  --output results/tensor-parallel/service-tp2.json
```

See [TP verification](tensor-parallel.md#reproduce-verification) for the complete
matrix and known NCCL sanitizer failure. CPU CI does not replace these GPU checks.

## GPT-OSS

See [GPT-OSS implementation and validation](gpt-oss.md) for BF16/MXFP4 fixture
checks, native 120B verification, template matching, and HTTP smoke commands.
`verify_service.py --smoke-only` runs a generic completion JSON/SSE check without
the Qwen-specific acceptance prompts. `verify_template.py --model PATH --output
DIR` supports other local checkpoints while keeping original reports separate.

## Optional mini-sglang comparison

Clone the reference checkout next to this repository as `../mini-sglang`, then
install it into the verification environment:

```bash
uv pip install --python .venv-verify/bin/python -e ../mini-sglang ninja
.venv-verify/bin/python scripts/compare_mini.py
```

The script runs the servers sequentially on identical devices/workloads and
records commands, per-request timings, memory samples, and cache configuration.
The recorded reference revision and measured limitations are in
[baseline validation](validation.md) and [TP validation](tensor-parallel.md).

## Results and acceptance

Committed JSON reports and text logs preserve the original measurements and
failures. Large binary/NumPy activation dumps, generated operator fixtures,
synthetic checkpoint weights, local environments, and experimental runs are
ignored by Git and can be regenerated with the scripts.

Full-model acceptance requires NRMSE <= 0.02 and cosine >= 0.999. Keep those
thresholds and failing rows when recording new results. Top-1 agreement and
successful generation are separate evidence; they do not satisfy a failing
logits gate. The current open items are documented in
[validation](validation.md) and [tensor parallelism](tensor-parallel.md).
