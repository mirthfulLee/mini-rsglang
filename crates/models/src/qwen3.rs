//! Qwen3 dense/MoE layer construction and forward execution.
use crate::{
    invalid, weights::load_weights, InferenceModel, LayerKv, ModelConfiguration, ModelDimensions,
    Qwen3Config,
};
use rsglang_core::Result;
use rsglang_distributed::TensorParallel;
use rsglang_kernels::KernelBackend;
use std::path::Path;

struct Layer<T> {
    input_norm: T,
    post_norm: T,
    qkv: T,
    o: T,
    q_norm: T,
    k_norm: T,
    mlp: Mlp<T>,
}
struct DenseMlp<T> {
    gate_up: T,
    down: T,
    intermediate: usize,
}
enum Mlp<T> {
    Dense(DenseMlp<T>),
    Moe {
        router: T,
        experts: Vec<DenseMlp<T>>,
    },
}
pub struct Qwen3<B: KernelBackend> {
    config: Qwen3Config,
    embedding: B::Tensor,
    lm_head: Option<B::Tensor>,
    norm: B::Tensor,
    layers: Vec<Layer<B::Tensor>>,
    tp: TensorParallel,
}
impl<B: KernelBackend> Qwen3<B> {
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
    pub fn load(backend: &B, path: &Path, config: Qwen3Config) -> Result<Self> {
        let tp = backend.tensor_parallel();
        config.validate_tp(tp)?;
        let mut weights = load_weights(backend, path, &config, tp)?;
        let mut take = |name: &str, shape: &[usize]| -> Result<B::Tensor> {
            let (actual, t) = weights
                .remove(name)
                .ok_or_else(|| invalid(format!("missing weight {name}")))?;
            if actual != shape {
                return Err(invalid(format!(
                    "{name}: shape {actual:?}, expected {shape:?}"
                )));
            }
            Ok(t)
        };
        let c = &config;
        let vocab = tp.vocab(c.vocab_size)?.padded_rows;
        let qwidth = tp.partition(c.num_attention_heads)?.len() * c.head_dim;
        let kvwidth = tp.kv_heads(c.num_key_value_heads)?.len() * c.head_dim;
        let embedding = take("model.embed_tokens.weight", &[vocab, c.hidden_size])?;
        let lm_head = if c.tie_word_embeddings {
            None
        } else {
            Some(take("lm_head.weight", &[vocab, c.hidden_size])?)
        };
        let norm = take("model.norm.weight", &[c.hidden_size])?;
        let mut layers = vec![];
        for i in 0..c.num_hidden_layers {
            let root = format!("model.layers.{i}");
            let qkv = backend.concat_rows(&[
                take(
                    &format!("{root}.self_attn.q_proj.weight"),
                    &[qwidth, c.hidden_size],
                )?,
                take(
                    &format!("{root}.self_attn.k_proj.weight"),
                    &[kvwidth, c.hidden_size],
                )?,
                take(
                    &format!("{root}.self_attn.v_proj.weight"),
                    &[kvwidth, c.hidden_size],
                )?,
            ])?;
            let mlp = if c.is_sparse_layer(i) {
                let router = take(
                    &format!("{root}.mlp.gate.weight"),
                    &[c.num_experts, c.hidden_size],
                )?;
                let mut experts = vec![];
                for e in 0..c.num_experts {
                    experts.push(load_mlp(
                        backend,
                        &mut take,
                        &format!("{root}.mlp.experts.{e}"),
                        c.hidden_size,
                        c.moe_intermediate_size / tp.size(),
                    )?);
                }
                Mlp::Moe { router, experts }
            } else {
                Mlp::Dense(load_mlp(
                    backend,
                    &mut take,
                    &format!("{root}.mlp"),
                    c.hidden_size,
                    c.intermediate_size / tp.size(),
                )?)
            };
            layers.push(Layer {
                input_norm: take(&format!("{root}.input_layernorm.weight"), &[c.hidden_size])?,
                post_norm: take(
                    &format!("{root}.post_attention_layernorm.weight"),
                    &[c.hidden_size],
                )?,
                qkv,
                o: take(
                    &format!("{root}.self_attn.o_proj.weight"),
                    &[c.hidden_size, qwidth],
                )?,
                q_norm: take(&format!("{root}.self_attn.q_norm.weight"), &[c.head_dim])?,
                k_norm: take(&format!("{root}.self_attn.k_norm.weight"), &[c.head_dim])?,
                mlp,
            });
        }
        if !weights.is_empty() {
            return Err(invalid(format!(
                "unexpected weights: {:?}",
                weights.keys().collect::<Vec<_>>()
            )));
        }
        Ok(Self {
            config,
            embedding,
            lm_head,
            norm,
            layers,
            tp,
        })
    }
    pub fn allocate_kv(
        &self,
        b: &B,
        pages: usize,
        page_size: usize,
    ) -> Result<Vec<LayerKv<B::Tensor>>> {
        <Self as InferenceModel<B>>::allocate_kv(self, b, pages, page_size)
    }
    pub fn forward_hidden(
        &self,
        b: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
    ) -> Result<B::Tensor> {
        self.forward_hidden_observed(b, meta, kv, &mut |_, _| Ok(()))
    }
    /// Optional observer for independent layer-by-layer numerical validation.
    pub fn forward_hidden_observed(
        &self,
        b: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
        observe: &mut impl FnMut(&str, &B::Tensor) -> Result<()>,
    ) -> Result<B::Tensor> {
        if kv.len() != self.layers.len() {
            return Err(invalid("KV layer count mismatch"));
        }
        let c = &self.config;
        let qheads = self.tp.partition(c.num_attention_heads)?.len();
        let kvheads = self.tp.kv_heads(c.num_key_value_heads)?.len();
        let mut x = b.embedding_shard(&self.embedding, meta, self.tp.vocab(c.vocab_size)?.start)?;
        observe("embedding", &x)?;
        for (i, (layer, cache)) in self.layers.iter().zip(kv).enumerate() {
            let n = b.rms_norm(&x, &layer.input_norm, c.hidden_size, c.rms_norm_eps)?;
            observe(&format!("layer{i}-input_norm"), &n)?;
            let qkv = b.linear(&n, &layer.qkv)?;
            let mut parts = b.split_columns(
                &qkv,
                &[
                    qheads * c.head_dim,
                    kvheads * c.head_dim,
                    kvheads * c.head_dim,
                ],
            )?;
            let v = parts.pop().unwrap();
            let k = parts.pop().unwrap();
            let q = parts.pop().unwrap();
            observe(&format!("layer{i}-q_proj"), &q)?;
            let mut q = b.rms_norm(&q, &layer.q_norm, c.head_dim, c.rms_norm_eps)?;
            let mut k = b.rms_norm(&k, &layer.k_norm, c.head_dim, c.rms_norm_eps)?;
            observe(&format!("layer{i}-q_norm"), &q)?;
            b.rope(&mut q, meta, qheads, c.head_dim, c.rope_theta)?;
            b.rope(&mut k, meta, kvheads, c.head_dim, c.rope_theta)?;
            observe(&format!("layer{i}-q_rope"), &q)?;
            observe(&format!("layer{i}-k_rope"), &k)?;
            observe(&format!("layer{i}-v_proj"), &v)?;
            b.store_kv(&k, &v, &mut cache.k, &mut cache.v, meta)?;
            let attn = b.attention(&q, &cache.k, &cache.v, meta, qheads, kvheads, c.head_dim)?;
            observe(&format!("layer{i}-attn"), &attn)?;
            let projected = b.linear_reduce(&attn, &layer.o)?;
            let residual = b.add(&x, &projected)?;
            let n = b.rms_norm(&residual, &layer.post_norm, c.hidden_size, c.rms_norm_eps)?;
            let down = match &layer.mlp {
                Mlp::Dense(mlp) => {
                    let activated = mlp.activated(b, &n)?;
                    observe(&format!("layer{i}-swiglu"), &activated)?;
                    b.linear_reduce(&activated, &mlp.down)?
                }
                Mlp::Moe { router, experts } => {
                    let scores = b.linear(&n, router)?;
                    let routes = b.router_topk(&scores, c.num_experts_per_tok, c.norm_topk_prob)?;
                    let mut out = b.accumulator(b.token_count(meta), c.hidden_size)?;
                    for route in routes {
                        let expert = &experts[route.expert];
                        let input = b.gather_rows(&n, &route.rows)?;
                        let activated = expert.activated(b, &input)?;
                        let values = b.linear(&activated, &expert.down)?;
                        b.scatter_weighted(&mut out, &values, &route)?;
                    }
                    b.finish_accumulator(out)?
                }
            };
            x = b.add(&residual, &down)?;
            observe(&format!("layer{i}-output"), &x)?;
        }
        let x = b.rms_norm(&x, &self.norm, c.hidden_size, c.rms_norm_eps)?;
        observe("final_norm", &x)?;
        Ok(x)
    }
    pub fn project(&self, b: &B, hidden: &B::Tensor, meta: &B::Metadata) -> Result<B::Logits> {
        let last = b.last_hidden(hidden, meta)?;
        let local = b.logits(&last, self.lm_head.as_ref().unwrap_or(&self.embedding))?;
        b.all_gather_logits(local, self.config.vocab_size)
    }
}

impl<T> DenseMlp<T> {
    fn activated<B: KernelBackend<Tensor = T>>(&self, b: &B, x: &T) -> Result<T> {
        let merged = b.linear(x, &self.gate_up)?;
        let mut parts = b.split_columns(&merged, &[self.intermediate, self.intermediate])?;
        let up = parts.pop().unwrap();
        let gate = parts.pop().unwrap();
        b.swiglu(&gate, &up)
    }
}
fn load_mlp<B: KernelBackend>(
    b: &B,
    take: &mut impl FnMut(&str, &[usize]) -> Result<B::Tensor>,
    root: &str,
    hidden: usize,
    local: usize,
) -> Result<DenseMlp<B::Tensor>> {
    let gate = take(&format!("{root}.gate_proj.weight"), &[local, hidden])?;
    let up = take(&format!("{root}.up_proj.weight"), &[local, hidden])?;
    Ok(DenseMlp {
        gate_up: b.concat_rows(&[gate, up])?,
        down: take(&format!("{root}.down_proj.weight"), &[hidden, local])?,
        intermediate: local,
    })
}

impl<B: KernelBackend> InferenceModel<B> for Qwen3<B> {
    fn dimensions(&self) -> ModelDimensions {
        self.config.dimensions()
    }
    fn forward_hidden(
        &self,
        backend: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
    ) -> Result<B::Tensor> {
        Qwen3::forward_hidden(self, backend, meta, kv)
    }
    fn project(&self, backend: &B, hidden: &B::Tensor, meta: &B::Metadata) -> Result<B::Logits> {
        Qwen3::project(self, backend, hidden, meta)
    }
}
