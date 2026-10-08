# GPT-OSS support and validation

GPT-OSS is an experimental second model family alongside Qwen3. Architecture
selection uses `model_type: gpt_oss` and `GptOssForCausalLM`. The model implements
`InferenceModel<B>` and its configuration implements `ModelConfiguration`, so the
shared engine executes through the same traits as Qwen3. The engine accepts
the Hugging Face checkpoint layout with either BF16 expert tensors or native
MXFP4 `*_blocks` / `*_scales`; the `original/` reference checkpoint layout is
not supported.

## Implementation

GPT-OSS has separate configuration and model modules. Its implementation
includes attention projection/router/expert biases, alternating full/sliding
causal attention, a learned softmax sink per query head, YaRN RoPE, FP32
normalization before BF16 output, and clipped interleaved SwiGLU with alpha
1.702 and the checkpoint limit. Query normalization used by Qwen3 is not applied.

MXFP4 uses the E2M1 nibble values and E8M0 block scales. Selected projection
ranges are fetched directly from safetensors files through checked byte reads.
The GPU decodes selected blocks to BF16 for cuBLAS GEMM. Attention, normalization,
router, embedding, and LM-head weights stay resident; expert projections are
uploaded and released as their routed rows are processed. Expert biases are
kept in host memory. This avoids expanding all 120B weights or requiring them
all to fit in GPU memory.

TP partitions query/KV heads, vocabulary, and expert intermediate width. Gate/up
pairs remain interleaved. The attention output projection adds its bias after
the collective sum. Each expert's down-projection bias contributes on rank 0
exactly once before the final mixture sum. Norms and router weights replicate;
KV heads can replicate when TP exceeds their count.

The GPT-OSS attention path follows HF eager BF16 score/probability rounding.
Qwen3 retains its existing attention and normalization arithmetic. All custom
launches remain inside `rsglang-kernels`; the model/checkpoint reader forbids
unsafe Rust.

Architecture sources are [OpenAI's reference model](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py),
[MXFP4 reference decoding](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py),
and [HF Transformers 4.57.3 GPT-OSS](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/gpt_oss/modeling_gpt_oss.py).

## Run

Use CUDA Toolkit 12.6 and the CUDA/NCCL setup in the README. The local validation
checkpoint is `/models/store/openai/gpt-oss-120b`.

```bash
# Short raw completion; a small KV budget helps on shared GPUs.
target/release/mini-rsglang --model /models/store/openai/gpt-oss-120b \
  --device 0 --kv-mib 128 --max-seq-len 1024 generate \
  --prompt 'Hello' --max-tokens 8 --json

# Chat using the checkpoint's Harmony template.
target/release/mini-rsglang --model /models/store/openai/gpt-oss-120b \
  --tp 2 --kv-mib 128 --max-seq-len 1024 generate \
  --chat --prompt 'Say hello.' --max-tokens 64

# The same loader works through the HTTP server.
target/release/mini-rsglang --model /models/store/openai/gpt-oss-120b \
  --tp 2 --kv-mib 128 serve --port 8000
```

The native 120B checkpoint has roughly 4 GiB of resident BF16 non-expert weights
on a single GPU, in addition to KV, temporary projections, and library resources.
TP reduces most resident weights. It does not eliminate per-step expert reads,
transfers, allocations, and synchronization. This is a compatibility path;
no low-latency or production-throughput claim is made.

`chat_template.jinja` takes precedence over an embedded template, matching HF's
checkpoint selection. `strftime_now` uses UTC. Input assistant
`reasoning_content` is also available to templates as `thinking`. Generation EOS
IDs come from the checkpoint metadata.

Harmony channel/tool parsing remains unimplemented. Output is the decoded token
continuation, including reasoning/channel labels. A small `max_tokens` can end
within the analysis channel before a final answer. The checkpoint's default
reasoning setting applies; `enable_thinking` is a Qwen3 template control.
Tools and multimodal requests remain unsupported by the HTTP subset.

## Verification evidence

Checks ran on the existing A100/CUDA 12.6 environment using PyTorch 2.8.0 and
Transformers 4.57.3. These measurements cover a small fixture matrix and two real
checkpoint rows; they do not establish general numerical parity.

| Check | Result | Evidence |
| --- | --- | --- |
| Synthetic BF16/MXFP4, TP 1/2/4, full/chunked prefill and three decode steps | 144/144 strict comparisons pass; all top-1 IDs match | [Report](../results/gpt-oss/fixture-report.json) |
| Existing Qwen3 dense/MoE, TP 1/2 regression | 96/96 strict comparisons pass | [Regression report](../results/gpt-oss/qwen-regression-report.json) |
| Real GPT-OSS template vs HF | 6/6 text and token arrays equal | [Template report](../results/gpt-oss/templates/template-report.json) |
| Real 120B short chat generation | Eight tokens and a length finish; continuation starts in analysis | [Events](../results/gpt-oss/generation.jsonl) |
| Real 120B completion JSON/SSE | Text/usage agree; `[DONE]`, health and page conservation pass | [Service report](../results/gpt-oss/service-report.json) |
| GPT-OSS/MXFP4 kernels under memcheck | Zero errors | [Log](../results/gpt-oss/memcheck.log) |
| GPT-OSS/MXFP4 kernels under racecheck | Zero errors/warnings/hazards | [Log](../results/gpt-oss/racecheck.log) |
| Real 120B strict logits, single GPU and TP 2 | Prefill fails NRMSE gate; decode passes; all four top-1 IDs match | Reports below |

Synthetic fixtures exercise nonzero biases/sinks, nontrivial norm weights,
YaRN positions beyond the original context, an eight-token sliding window,
uneven vocabulary padding, replicated KV heads, and interleaved expert sharding.
The numerical gate remains NRMSE <= 0.02 and cosine >= 0.999.

The real-checkpoint verifier streams expert weights through independent
PyTorch arithmetic and HF attention/RMSNorm/YaRN. It uses the HF CUDA expert
path's FP32 reduction of BF16 weighted contributions and FP32 final logits. It
never materializes all expert weights or feeds reference activations to Rust.
Its raw prompt is `Hello` (one token) with one teacher-forced decode step.

| TP | Phase | NRMSE | Cosine | Rust / reference top-1 | Strict gate |
| --- | --- | ---: | ---: | --- | --- |
| 1 | Prefill | 0.0220036 | 0.9997703 | 11 / 11 | Failed |
| 1 | Decode | 0.0085850 | 0.9999717 | 5922 / 5922 | Passed |
| 2 | Prefill | 0.0216526 | 0.9997735 | 11 / 11 | Failed |
| 2 | Decode | 0.0036197 | 0.9999935 | 5922 / 5922 | Passed |

[Single-GPU report](../results/gpt-oss/checkpoint/report.json) and
[TP-2 report](../results/gpt-oss/checkpoint/report-tp2.json) retain the failing
rows. The real-checkpoint verifier deliberately exits nonzero for the current
failure. Existing Qwen3 numerical and NCCL instrumentation limitations remain
as documented in [baseline validation](validation.md) and [TP validation](tensor-parallel.md).

## Reproduce

Build all examples and create the verification environment using the
[development guide](development.md). Select free GPUs using `CUDA_VISIBLE_DEVICES`;
the scripts interpret device ordinals relative to that list. Set NCCL transport
options appropriate to the machine before TP runs.

```bash
.venv-verify/bin/python scripts/verify_gpt_oss.py --world-sizes 1,2,4
.venv-verify/bin/python scripts/verify_template.py \
  --model /models/store/openai/gpt-oss-120b --output results/gpt-oss/templates
.venv-verify/bin/python scripts/verify_service.py --smoke-only \
  --model /models/store/openai/gpt-oss-120b --device 0 --port 24837 \
  --output results/gpt-oss/service-report.json

# Writes its complete report; currently exits 1 at the unchanged strict gate.
.venv-verify/bin/python scripts/verify_gpt_oss_checkpoint.py --steps 1

compute-sanitizer --tool memcheck --error-exitcode 99 \
  target/release/examples/forward results/gpt-oss/fixtures/mxfp4 \
  results/gpt-oss/fixtures/mxfp4/input.json results/gpt-oss/fixtures/memcheck 5 0
compute-sanitizer --tool racecheck --error-exitcode 99 \
  target/release/examples/forward results/gpt-oss/fixtures/mxfp4 \
  results/gpt-oss/fixtures/mxfp4/input.json results/gpt-oss/fixtures/racecheck 5 0
```

Synthetic checkpoint weights and binary/NumPy dumps are ignored by Git and
regenerated by these scripts. Committed reports/logs record observed results.
