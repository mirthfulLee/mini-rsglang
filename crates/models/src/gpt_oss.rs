//! GPT-OSS inference with resident attention and streamed expert projections.
use crate::{
    checkpoint::{Checkpoint, TensorSource},
    invalid,
    weights::decode_shard,
    GptOssConfig, LayerKv,
};
use half::bf16;
use rsglang_core::Result;
use rsglang_distributed::{Shard, TensorParallel};
use rsglang_kernels::KernelBackend;
use std::{ops::Range, path::Path};

struct Projection {
    weight: TensorSource,
    scales: Option<TensorSource>,
    input: usize,
    output: usize,
}
impl Projection {
    fn take(
        checkpoint: &mut Checkpoint,
        root: &str,
        name: &str,
        experts: usize,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let base = format!("{root}.{name}");
        let (weight, scales) = if checkpoint.contains(&format!("{base}_blocks")) {
            if !input.is_multiple_of(32) {
                return Err(invalid("MXFP4 input width must divide 32"));
            }
            let blocks = checkpoint.take(
                &format!("{base}_blocks"),
                &[experts, output, input / 32, 16],
            )?;
            let scales =
                checkpoint.take(&format!("{base}_scales"), &[experts, output, input / 32])?;
            if blocks.dtype != safetensors::Dtype::U8 || scales.dtype != safetensors::Dtype::U8 {
                return Err(invalid("MXFP4 blocks/scales must be U8"));
            }
            (blocks, Some(scales))
        } else {
            (checkpoint.take(&base, &[experts, input, output])?, None)
        };
        Ok(Self {
            weight,
            scales,
            input,
            output,
        })
    }
    fn upload<B: KernelBackend>(
        &self,
        b: &B,
        expert: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<B::Tensor> {
        if let Some(scales) = &self.scales {
            let start = (expert * self.output + rows.start) * self.input;
            let count = rows.len() * self.input;
            let blocks = self.weight.read_range(start / 2, count / 2)?;
            let scales = scales.read_range(start / 32, count / 32)?;
            let full = b.upload_mxfp4(&blocks, &scales, &[rows.len(), self.input])?;
            if cols.start == 0 && cols.end == self.input {
                return Ok(full);
            }
            let mut widths = vec![];
            if cols.start > 0 {
                widths.push(cols.start);
            }
            widths.push(cols.len());
            if cols.end < self.input {
                widths.push(self.input - cols.end);
            }
            let selected = usize::from(cols.start > 0);
            return Ok(b.split_columns(&full, &widths)?.remove(selected));
        }
        let width = self.weight.dtype.bitsize() / 8;
        let bytes = self.weight.read_range(
            expert * self.input * self.output * width,
            self.input * self.output * width,
        )?;
        let values = decode_shard(
            &bytes,
            self.weight.dtype,
            &[self.input, self.output],
            &Shard::Replicated,
            &[self.input, self.output],
        )?;
        let mut transposed = Vec::with_capacity(rows.len() * cols.len());
        for row in rows.clone() {
            for col in cols.clone() {
                transposed.push(values[col * self.output + row]);
            }
        }
        b.upload(&transposed, &[rows.len(), cols.len()])
    }
}
struct Layer<T> {
    input_norm: T,
    post_norm: T,
    q: T,
    k: T,
    v: T,
    o: T,
    q_bias: T,
    k_bias: T,
    v_bias: T,
    o_bias: T,
    sinks: T,
    router: T,
    router_bias: T,
    gate: Projection,
    down: Projection,
    gate_bias: Vec<bf16>,
    down_bias: Vec<bf16>,
    window: usize,
}
pub struct GptOss<B: KernelBackend> {
    config: GptOssConfig,
    embedding: B::Tensor,
    lm_head: B::Tensor,
    norm: B::Tensor,
    layers: Vec<Layer<B::Tensor>>,
    tp: TensorParallel,
    frequencies: Vec<f32>,
    magnitude: f32,
}
fn host(source: &TensorSource, shard: &Shard) -> Result<(Vec<bf16>, Vec<usize>)> {
    let full = if source.shape.len() == 1 {
        vec![source.shape[0], 1]
    } else {
        source.shape.clone()
    };
    let shape = shard.shape(&full)?;
    let values = decode_shard(&source.read()?, source.dtype, &full, shard, &shape)?;
    let shape = if source.shape.len() == 1 {
        vec![shape[0]]
    } else {
        shape
    };
    Ok((values, shape))
}
fn resident<B: KernelBackend>(
    b: &B,
    cp: &mut Checkpoint,
    name: &str,
    shape: &[usize],
    shard: Shard,
) -> Result<B::Tensor> {
    let source = cp.take(name, shape)?;
    let (values, shape) = host(&source, &shard)?;
    b.upload(&values, &shape)
}
impl<B: KernelBackend> GptOss<B> {
    pub fn load(b: &B, path: &Path, config: GptOssConfig) -> Result<Self> {
        let tp = b.tensor_parallel();
        config.validate_tp(tp)?;
        let c = config.dimensions;
        let mut cp = Checkpoint::open(path)?;
        let vocab = tp.vocab(c.vocab_size)?;
        let vocab_shard = Shard::Rows {
            start: vocab.start,
            len: vocab.end - vocab.start,
            padded: vocab.padded_rows,
        };
        let embedding = resident(
            b,
            &mut cp,
            "model.embed_tokens.weight",
            &[c.vocab_size, c.hidden_size],
            vocab_shard.clone(),
        )?;
        let lm_head = resident(
            b,
            &mut cp,
            "lm_head.weight",
            &[c.vocab_size, c.hidden_size],
            vocab_shard,
        )?;
        let norm = resident(
            b,
            &mut cp,
            "model.norm.weight",
            &[c.hidden_size],
            Shard::Replicated,
        )?;
        let q = tp.partition(c.num_attention_heads)?;
        let kv = tp.kv_heads(c.num_key_value_heads)?;
        let mut layers = vec![];
        for i in 0..c.num_hidden_layers {
            let root = format!("model.layers.{i}");
            let attn = format!("{root}.self_attn");
            let mlp = format!("{root}.mlp");
            let experts = config.num_local_experts;
            let norm = |cp: &mut Checkpoint, name: &str| {
                resident(
                    b,
                    cp,
                    &format!("{root}.{name}.weight"),
                    &[c.hidden_size],
                    Shard::Replicated,
                )
            };
            let matrix = |cp: &mut Checkpoint, name: &str, heads: usize, part: Range<usize>| {
                resident(
                    b,
                    cp,
                    &format!("{attn}.{name}.weight"),
                    &[heads * c.head_dim, c.hidden_size],
                    Shard::rows(part.start * c.head_dim..part.end * c.head_dim),
                )
            };
            let bias = |cp: &mut Checkpoint, name: &str, heads: usize, part: Range<usize>| {
                resident(
                    b,
                    cp,
                    &format!("{attn}.{name}.bias"),
                    &[heads * c.head_dim],
                    Shard::rows(part.start * c.head_dim..part.end * c.head_dim),
                )
            };
            let gate = Projection::take(
                &mut cp,
                &format!("{mlp}.experts"),
                "gate_up_proj",
                experts,
                c.hidden_size,
                2 * c.intermediate_size,
            )?;
            let down = Projection::take(
                &mut cp,
                &format!("{mlp}.experts"),
                "down_proj",
                experts,
                c.intermediate_size,
                c.hidden_size,
            )?;
            let gate_bias = cp.take(
                &format!("{mlp}.experts.gate_up_proj_bias"),
                &[experts, 2 * c.intermediate_size],
            )?;
            let down_bias = cp.take(
                &format!("{mlp}.experts.down_proj_bias"),
                &[experts, c.hidden_size],
            )?;
            layers.push(Layer {
                input_norm: norm(&mut cp, "input_layernorm")?,
                post_norm: norm(&mut cp, "post_attention_layernorm")?,
                q: matrix(&mut cp, "q_proj", c.num_attention_heads, q.clone())?,
                k: matrix(&mut cp, "k_proj", c.num_key_value_heads, kv.clone())?,
                v: matrix(&mut cp, "v_proj", c.num_key_value_heads, kv.clone())?,
                q_bias: bias(&mut cp, "q_proj", c.num_attention_heads, q.clone())?,
                k_bias: bias(&mut cp, "k_proj", c.num_key_value_heads, kv.clone())?,
                v_bias: bias(&mut cp, "v_proj", c.num_key_value_heads, kv.clone())?,
                o: resident(
                    b,
                    &mut cp,
                    &format!("{attn}.o_proj.weight"),
                    &[c.hidden_size, c.num_attention_heads * c.head_dim],
                    Shard::Columns {
                        start: q.start * c.head_dim,
                        len: q.len() * c.head_dim,
                    },
                )?,
                o_bias: resident(
                    b,
                    &mut cp,
                    &format!("{attn}.o_proj.bias"),
                    &[c.hidden_size],
                    Shard::Replicated,
                )?,
                sinks: resident(
                    b,
                    &mut cp,
                    &format!("{attn}.sinks"),
                    &[c.num_attention_heads],
                    Shard::rows(q.clone()),
                )?,
                router: resident(
                    b,
                    &mut cp,
                    &format!("{mlp}.router.weight"),
                    &[experts, c.hidden_size],
                    Shard::Replicated,
                )?,
                router_bias: resident(
                    b,
                    &mut cp,
                    &format!("{mlp}.router.bias"),
                    &[experts],
                    Shard::Replicated,
                )?,
                gate,
                down,
                gate_bias: host(&gate_bias, &Shard::Replicated)?.0,
                down_bias: host(&down_bias, &Shard::Replicated)?.0,
                window: if config.layer_types[i] == "sliding_attention" {
                    config.sliding_window
                } else {
                    0
                },
            });
            b.synchronize()?;
        }
        cp.finish()?;
        let (frequencies, magnitude) = config.rope_parameters();
        Ok(Self {
            config,
            embedding,
            lm_head,
            norm,
            layers,
            tp,
            frequencies,
            magnitude,
        })
    }
    pub fn allocate_kv(
        &self,
        b: &B,
        pages: usize,
        page_size: usize,
    ) -> Result<Vec<LayerKv<B::Tensor>>> {
        let c = self.config.dimensions;
        let shape = [
            pages,
            page_size,
            self.tp.kv_heads(c.num_key_value_heads)?.len(),
            c.head_dim,
        ];
        self.layers
            .iter()
            .map(|_| {
                Ok(LayerKv {
                    k: b.zeros(&shape)?,
                    v: b.zeros(&shape)?,
                })
            })
            .collect()
    }
    pub fn forward_hidden(
        &self,
        b: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
    ) -> Result<B::Tensor> {
        if kv.len() != self.layers.len() {
            return Err(invalid("KV layer count mismatch"));
        }
        let c = self.config.dimensions;
        let qheads = self.tp.partition(c.num_attention_heads)?.len();
        let kvheads = self.tp.kv_heads(c.num_key_value_heads)?.len();
        let part = self.tp.partition(c.intermediate_size)?;
        let mut x = b.embedding_shard(&self.embedding, meta, self.tp.vocab(c.vocab_size)?.start)?;
        for (layer, cache) in self.layers.iter().zip(kv) {
            let n = b.gpt_rms_norm(
                &x,
                &layer.input_norm,
                c.hidden_size,
                self.config.rms_norm_eps,
            )?;
            let mut q = b.linear_bias(&n, &layer.q, &layer.q_bias, false)?;
            let mut k = b.linear_bias(&n, &layer.k, &layer.k_bias, false)?;
            let v = b.linear_bias(&n, &layer.v, &layer.v_bias, false)?;
            b.rope_scaled(
                &mut q,
                meta,
                qheads,
                c.head_dim,
                &self.frequencies,
                self.magnitude,
            )?;
            b.rope_scaled(
                &mut k,
                meta,
                kvheads,
                c.head_dim,
                &self.frequencies,
                self.magnitude,
            )?;
            b.store_kv(&k, &v, &mut cache.k, &mut cache.v, meta)?;
            let attn = b.attention_sink(
                &q,
                &cache.k,
                &cache.v,
                meta,
                &layer.sinks,
                layer.window,
                c.head_dim,
            )?;
            let projected = b.linear_bias(&attn, &layer.o, &layer.o_bias, true)?;
            let residual = b.add(&x, &projected)?;
            let n = b.gpt_rms_norm(
                &residual,
                &layer.post_norm,
                c.hidden_size,
                self.config.rms_norm_eps,
            )?;
            let scores = b.linear_bias(&n, &layer.router, &layer.router_bias, false)?;
            let routes = b.router_topk(&scores, self.config.num_experts_per_tok, true)?;
            let mut out = b.accumulator(b.token_count(meta), c.hidden_size)?;
            for route in routes {
                let input = b.gather_rows(&n, &route.rows)?;
                let gate = layer.gate.upload(
                    b,
                    route.expert,
                    2 * part.start..2 * part.end,
                    0..c.hidden_size,
                )?;
                let bias_start = route.expert * 2 * c.intermediate_size;
                let gate_bias = b.upload(
                    &layer.gate_bias[bias_start + 2 * part.start..bias_start + 2 * part.end],
                    &[2 * part.len()],
                )?;
                let gate_up = b.linear_bias(&input, &gate, &gate_bias, false)?;
                let activation = b.gpt_swiglu(&gate_up, self.config.swiglu_limit)?;
                let down = layer
                    .down
                    .upload(b, route.expert, 0..c.hidden_size, part.clone())?;
                let values = if self.tp.rank() == 0 {
                    // Add each expert's output bias exactly once, before the rank sum.
                    let start = route.expert * c.hidden_size;
                    let bias = b.upload(
                        &layer.down_bias[start..start + c.hidden_size],
                        &[c.hidden_size],
                    )?;
                    b.linear_bias(&activation, &down, &bias, false)?
                } else {
                    b.linear(&activation, &down)?
                };
                b.scatter_weighted(&mut out, &values, &route)?;
                // Bound temporary expert allocations even for large prefill batches.
                b.synchronize()?;
            }
            x = b.add(&residual, &b.finish_accumulator(out)?)?;
        }
        b.gpt_rms_norm(&x, &self.norm, c.hidden_size, self.config.rms_norm_eps)
    }
    pub fn project(&self, b: &B, x: &B::Tensor, meta: &B::Metadata) -> Result<B::Logits> {
        b.all_gather_logits(
            b.logits(&b.last_hidden(x, meta)?, &self.lm_head)?,
            self.config.dimensions.vocab_size,
        )
    }
}
