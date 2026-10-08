# mini-rsglang

A small Rust inference engine for Qwen3 dense/MoE and experimental GPT-OSS models, inspired by
[mini-sglang](https://github.com/sgl-project/mini-sglang). It supports single-GPU
and NCCL tensor-parallel execution, offline generation, and an OpenAI-style
HTTP API. Inference runs entirely in Rust with CUDA; Python is used for
independent verification.

## What works

- Local safetensors loading, BF16 weights/activations/KV, and FP32 logits.
- Qwen3 dense/MoE and GPT-OSS layers, paged grouped-query attention, and tensor parallelism.
- Continuous batching, chunked prefill, radix prefix sharing/eviction, and request cancellation.
- Greedy and seeded top-k/top-p sampling, checkpoint chat templates, and UTF-8 streaming.
- CLI generation/benchmarks, a Rust streaming API, and JSON/SSE serving.

This is an initial implementation. Functional checks and custom-operator checks
pass, but the full-model BF16 logits gate is still open. Some HF comparisons fail
the original NRMSE/cosine thresholds despite matching top-1 IDs. Native NCCL
checks pass; NCCL value checks under Compute Sanitizer do not. See
[validation status](#validation-status) for evidence and limits.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux |
| Rust | 1.99.0, pinned in `rust-toolchain.toml` |
| GPU | NVIDIA with supported compute capability: SM 80, 86, 89, or 90 |
| CUDA | Toolkit 12.6 and a compatible NVIDIA driver |
| NCCL | Runtime >= 2.27, required only for multiple GPUs |
| Checkpoint | A local supported Hugging Face checkpoint in safetensors format |

Rust uses pinned `cudarc` bindings, cuBLAS GEMM, and NVRTC-compiled CUDA C
kernels. CUDA libraries load dynamically; ordinary CPU checks do not require
an NVIDIA GPU or CUDA libraries.

## Quick start

```bash
git clone git@github.com:mirthfulLee/mini-rsglang.git
cd mini-rsglang
source "$HOME/.cargo/env"

export CUDA_TOOLKIT_PATH=/usr/local/cuda-12.6
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64:${LD_LIBRARY_PATH:-}"
cargo build --release --locked

# Replace the path with your local checkpoint directory.
export MODEL_PATH=/models/store/Qwen/Qwen3-0.6B
target/release/mini-rsglang --model "$MODEL_PATH" generate \
  --chat --prompt '请用一句话解释 Rust 的所有权。' --max-tokens 64
```

The checkpoint directory must contain `config.json`, `tokenizer.json`,
`tokenizer_config.json` and either `model.safetensors`
or indexed safetensors shards. The chat template may be in `tokenizer_config.json`
or a standalone `chat_template.jinja`. `generation_config.json` is optional and supplies
EOS metadata when present. The engine does not download checkpoints.

Use `--json` for token/event JSONL, or supply token IDs directly:

```bash
target/release/mini-rsglang --model "$MODEL_PATH" generate \
  --token-ids 9707,11,14990 --max-tokens 16
target/release/mini-rsglang --help
target/release/mini-rsglang generate --help
```

## Supported models

| Architecture | Checkpoints exercised | Execution |
| --- | --- | --- |
| `Qwen3ForCausalLM` | Qwen3-0.6B, Qwen3-1.7B | Single GPU and TP 2/4/8 |
| `Qwen3MoeForCausalLM` | Synthetic fixtures; Qwen3-30B-A3B | Synthetic TP 1/2/4/8; real 30B TP 8 |
| `GptOssForCausalLM` | BF16/MXFP4 fixtures; gpt-oss-120b | Synthetic TP 1/2/4; real 120B single GPU and TP 2 |

The loader dispatches by `model_type` and validates architecture and dimensions.
Qwen3 variants with RoPE scaling, sliding-window attention, biased attention, or
non-SiLU activation remain unsupported. GPT-OSS has a separate implementation
with YaRN, sliding/full attention, sinks, biases, and clipped interleaved SwiGLU.
Other architectures are not implemented.

GPT-OSS reads the Hugging Face checkpoint layout, including native MXFP4 expert
blocks/scales. Attention and embeddings stay on the GPU; selected experts are
read from checkpoint files, decoded, uploaded, and released each step. This
bounds GPU memory but adds disk/host-transfer and synchronization overhead.
The 120B short prefill/decode checks match reference top-1 predictions, while
some logits still exceed the strict numerical gate. See [GPT-OSS validation](docs/gpt-oss.md).

```bash
target/release/mini-rsglang --model /models/store/openai/gpt-oss-120b \
  --tp 2 --kv-mib 128 --max-seq-len 1024 generate \
  --chat --prompt 'Say hello.' --max-tokens 64
```

GPT-OSS uses its checkpoint's default reasoning setting. `enable_thinking` only
controls Qwen3 templates. Harmony channel/tool messages are not parsed into
separate API fields; generated reasoning and channel labels appear in the
decoded continuation. Tool use remains unsupported.

## HTTP serving

```bash
target/release/mini-rsglang --model "$MODEL_PATH" serve --port 8000
```

The server binds `127.0.0.1` by default. In another terminal:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好！"}],"max_tokens":64,"temperature":0,"stream":true}'
```

| Endpoint | Supported behavior |
| --- | --- |
| `POST /v1/completions` | String prompt or token-ID array; JSON or SSE |
| `POST /v1/chat/completions` | Text system/user/assistant messages; JSON or SSE |
| `GET /v1/models` | Loaded model metadata |
| `GET /health` | Worker health |
| `GET /metrics` | Prometheus metrics |

Sampling fields are `max_tokens`, `temperature`, `top_k`, `top_p`, `seed`, and
`ignore_eos`; only `n=1` is supported. Chat accepts assistant `reasoning_content`
and `enable_thinking`. Thinking is disabled by default. SSE ends with `[DONE]`
and supports `stream_options: {"include_usage":true}`.

Unknown or unsupported fields, including tools, images, stop strings, and
logprobs, return 400. Excessive context is rejected. Queue exhaustion or a
request exceeding the entire KV pool returns 429. Disconnecting or dropping a
stream cancels its request; a failed worker becomes unhealthy.

## Multiple GPUs

Install NCCL >= 2.27 and expose `libnccl.so.2` through the loader search path.
If needed, set `RSGLANG_NCCL_LIBRARY` to its full path before launching.

```bash
target/release/mini-rsglang --model "$MODEL_PATH" --tp 2 generate \
  --chat --prompt 'Explain Rust ownership.' --max-tokens 32

# Explicit device list; its length determines TP size.
target/release/mini-rsglang --model "$MODEL_PATH" --devices 0,1,2,3 \
  serve --port 8000

# MoE checkpoint, with tensor parallelism inside each expert.
target/release/mini-rsglang --model /models/store/Qwen/Qwen3-30B-A3B \
  --tp 8 generate --chat --prompt '你好！' --max-tokens 16
```

`--device N --tp T` selects consecutive devices starting at N. Device ordinals
are relative to `CUDA_VISIBLE_DEVICES`. `--kv-mib` applies to each GPU. Query
heads and intermediate widths must divide TP size; KV heads may be partitioned
or replicated.

On the validation machine, `NCCL_P2P_LEVEL=NVL` was required to avoid failed
cross-PCIe P2P transfers. This is a topology-specific setting. See
[tensor parallelism](docs/tensor-parallel.md) for setup, ownership, failure
handling, and measured performance.

## Configuration and benchmarks

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `/models/store/Qwen/Qwen3-0.6B` | Local checkpoint directory |
| `--device` | `0` | First GPU |
| `--kv-mib` | `2048` | KV storage per GPU in MiB |
| `--max-seq-len` | `4096` | Context limit |
| `--page-size` | `16` | Tokens per KV page |
| `--prefill-budget` | `512` | Prefill tokens per scheduler step |
| `--max-running` | `32` | Active requests |
| `--max-waiting` | `256` | Waiting requests |
| `--no-prefix-cache` | Off | Disable radix prefix reuse |

Generation defaults to greedy sampling and 128 output tokens. All limits and
sampling options are available through CLI help.

```bash
target/release/mini-rsglang --model "$MODEL_PATH" bench \
  --requests 16 --concurrency 8 --input-tokens 128 --output-tokens 32 \
  --output results/offline-bench.json
```

The benchmark runs both cold and hot cache modes by default. Cold disables
prefix sharing; hot primes a shared prompt. Recorded measurements and their
limits are in [single-GPU validation](docs/validation.md) and
[multi-GPU validation](docs/tensor-parallel.md). More GPUs slowed down the small
0.6B workload on the tested topology; these reports do not claim strong scaling
or general performance parity with mini-sglang.

## Rust API

Use `rsglang-runtime::EngineHandle::load` for a single GPU or
`load_tensor_parallel` for multiple GPUs. `generate` returns a stream of
`GenerationEvent` values: tokens/text, a finish reason with usage, or an error.
`shutdown` waits for worker cleanup and device-resource destruction.

A complete example is in [generate.rs](crates/runtime/examples/generate.rs):

```bash
cargo run --release --locked -p rsglang-runtime --example generate -- "$MODEL_PATH"
```

## Code layout

| Path | Responsibility |
| --- | --- |
| `crates/core` | Protocol, sampling parameters, runtime configuration, runner contract |
| `crates/distributed` | TP layouts, checkpoint slices, rank workers and barriers |
| `crates/kernels` | Backend interface, checked CUDA operations, cuBLAS/NVRTC/NCCL |
| `crates/models` | Configuration, safetensors loading, Qwen3/GPT-OSS execution |
| `crates/cache` | CPU page ownership, reservations, naive/radix caches |
| `crates/engine` | Physical KV, forward execution, sampling |
| `crates/runtime` | Scheduler, tokenizer/templates, streams, lifecycle and metrics |
| `crates/server` | CLI, HTTP/SSE, benchmarks |
| `scripts` | Independent Python verification and reference comparisons |
| `results` | Recorded reports and logs; large regenerated fixtures are ignored |

Model configuration, checkpoint loading, and forward execution live in separate
modules. Each model implements `InferenceModel<B>`; Engine owns a trait object
returned by `load_model`. Configurations implement `ModelConfiguration` and use
one centralized loading/registration module. `KernelBackend` isolates model code
from CUDA resources. Each GPU
worker owns its model, KV shard, and stream. The scheduler commits a step after
all ranks complete. Rust unsafe code is confined to the kernels crate. See
[architecture](docs/architecture.md) for the ownership and scheduling design.

## Validation status

Recorded GPU checks ran on 2026-10-06 with A100-SXM4-80GB GPUs and CUDA 12.6.
These reports describe the measured implementation at that time.

| Check | Recorded result |
| --- | --- |
| CPU tests | 28 passed; one GPU-only test excluded |
| Custom operators and their memcheck/racecheck | Passed |
| Synthetic dense/MoE, TP 1/2/4/8 | 192/192 HF comparisons passed |
| Real HTTP/SSE, including TP 2/8 | Passed |
| Real full-model logits | Some rows fail NRMSE <= 0.02 / cosine >= 0.999 |
| Native NCCL under Compute Sanitizer | Value assertions fail under instrumentation |

Matching top-1 predictions on the tested inputs does not establish general
sequence or numerical parity. The original thresholds remain unchanged, and
strict verifiers return nonzero when they fail. See [baseline reports](docs/validation.md)
and [TP reports](docs/tensor-parallel.md) for commands and per-model evidence.

Run local CPU checks with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
uvx ruff==0.16.10 check scripts
uvx ruff==0.16.10 format --check scripts
```

See [development and verification](docs/development.md) to set up the separate
Python environment and reproduce GPU checks. CUDA Graph, overlap scheduling,
quantization beyond GPT-OSS MXFP4, PP/DP/SP/EP, fused expert kernels, and cuTile/cuda-oxide execution
remain future work. The current attention kernel is a simple SIMT implementation
and intermediates are allocated per step. The [cuTile plan](docs/cutile.md)
describes requirements for a future backend.

## License

MIT OR Apache-2.0. Dependencies and checkpoints retain their own licenses.
mini-sglang is the architecture reference; the reference checkout is not modified
by this project.
