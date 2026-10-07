//! Safetensors validation, rank-local checkpoint slicing, and device upload.
use crate::{invalid, Qwen3Config};
use half::bf16;
use rsglang_core::Result;
use rsglang_distributed::{Shard, TensorParallel};
use rsglang_kernels::KernelBackend;
use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
};

type LoadedWeights<T> = HashMap<String, (Vec<usize>, T)>;

pub(super) fn load_weights<B: KernelBackend>(
    backend: &B,
    path: &Path,
    config: &Qwen3Config,
    tp: TensorParallel,
) -> Result<LoadedWeights<B::Tensor>> {
    let specs = weight_specs(config, tp)?;
    let mut files = BTreeSet::new();
    let index = path.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index)?).map_err(|e| invalid(e.to_string()))?;
        let map = v
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| invalid("safetensors index missing weight_map"))?;
        for v in map.values() {
            let file = v.as_str().ok_or_else(|| invalid("invalid shard name"))?;
            if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                return Err(invalid("shard must be a local safetensors filename"));
            }
            files.insert(file.to_string());
        }
    } else {
        files.insert("model.safetensors".into());
    }
    let mut weights = LoadedWeights::new();
    for file in files {
        let bytes = std::fs::read(path.join(file))?;
        let tensors =
            safetensors::SafeTensors::deserialize(&bytes).map_err(|e| invalid(e.to_string()))?;
        for (name, tensor) in tensors.tensors() {
            if weights.contains_key(&name) {
                return Err(invalid(format!("duplicate tensor {name}")));
            }
            if config.tie_word_embeddings && name == "lm_head.weight" {
                continue;
            }
            let (full, shard) = specs
                .get(&name)
                .ok_or_else(|| invalid(format!("unexpected weight {name}")))?;
            if tensor.shape() != full {
                return Err(invalid(format!(
                    "{name}: shape {:?}, expected {full:?}",
                    tensor.shape()
                )));
            }
            let local_shape = shard.shape(full)?;
            let data = tensor.data();
            let values = decode_shard(data, tensor.dtype(), full, shard, &local_shape)?;
            let device = backend.upload(&values, &local_shape)?;
            weights.insert(name, (local_shape, device));
        }
        backend.synchronize()?;
    }
    Ok(weights)
}

type WeightSpecs = HashMap<String, (Vec<usize>, Shard)>;
pub(super) fn weight_specs(c: &Qwen3Config, tp: TensorParallel) -> Result<WeightSpecs> {
    let mut specs = HashMap::new();
    let vocab = tp.vocab(c.vocab_size)?;
    let vocab_shard = Shard::Rows {
        start: vocab.start,
        len: vocab.end - vocab.start,
        padded: vocab.padded_rows,
    };
    specs.insert(
        "model.embed_tokens.weight".into(),
        (vec![c.vocab_size, c.hidden_size], vocab_shard.clone()),
    );
    if !c.tie_word_embeddings {
        specs.insert(
            "lm_head.weight".into(),
            (vec![c.vocab_size, c.hidden_size], vocab_shard),
        );
    }
    specs.insert(
        "model.norm.weight".into(),
        (vec![c.hidden_size], Shard::Replicated),
    );
    let q = tp.partition(c.num_attention_heads)?;
    let kv = tp.kv_heads(c.num_key_value_heads)?;
    for i in 0..c.num_hidden_layers {
        let root = format!("model.layers.{i}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            specs.insert(
                format!("{root}.{norm}.weight"),
                (vec![c.hidden_size], Shard::Replicated),
            );
        }
        for norm in ["q_norm", "k_norm"] {
            specs.insert(
                format!("{root}.self_attn.{norm}.weight"),
                (vec![c.head_dim], Shard::Replicated),
            );
        }
        specs.insert(
            format!("{root}.self_attn.q_proj.weight"),
            (
                vec![c.num_attention_heads * c.head_dim, c.hidden_size],
                Shard::rows(q.start * c.head_dim..q.end * c.head_dim),
            ),
        );
        for name in ["k_proj", "v_proj"] {
            specs.insert(
                format!("{root}.self_attn.{name}.weight"),
                (
                    vec![c.num_key_value_heads * c.head_dim, c.hidden_size],
                    Shard::rows(kv.start * c.head_dim..kv.end * c.head_dim),
                ),
            );
        }
        specs.insert(
            format!("{root}.self_attn.o_proj.weight"),
            (
                vec![c.hidden_size, c.num_attention_heads * c.head_dim],
                Shard::Columns {
                    start: q.start * c.head_dim,
                    len: q.len() * c.head_dim,
                },
            ),
        );
        if c.is_sparse_layer(i) {
            specs.insert(
                format!("{root}.mlp.gate.weight"),
                (vec![c.num_experts, c.hidden_size], Shard::Replicated),
            );
            for e in 0..c.num_experts {
                add_mlp_specs(
                    &mut specs,
                    &format!("{root}.mlp.experts.{e}"),
                    c.hidden_size,
                    c.moe_intermediate_size,
                    tp,
                )?;
            }
        } else {
            add_mlp_specs(
                &mut specs,
                &format!("{root}.mlp"),
                c.hidden_size,
                c.intermediate_size,
                tp,
            )?;
        }
    }
    Ok(specs)
}
fn add_mlp_specs(
    specs: &mut WeightSpecs,
    root: &str,
    hidden: usize,
    intermediate: usize,
    tp: TensorParallel,
) -> Result<()> {
    let part = tp.partition(intermediate)?;
    for name in ["gate_proj", "up_proj"] {
        specs.insert(
            format!("{root}.{name}.weight"),
            (vec![intermediate, hidden], Shard::rows(part.clone())),
        );
    }
    specs.insert(
        format!("{root}.down_proj.weight"),
        (
            vec![hidden, intermediate],
            Shard::Columns {
                start: part.start,
                len: part.len(),
            },
        ),
    );
    Ok(())
}
pub(super) fn decode_shard(
    data: &[u8],
    dtype: safetensors::Dtype,
    full: &[usize],
    shard: &Shard,
    local: &[usize],
) -> Result<Vec<bf16>> {
    let count = local
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid("weight shard overflow"))?;
    let width = match dtype {
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
        safetensors::Dtype::F32 => 4,
        _ => return Err(invalid("unsupported weight dtype")),
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let value = if let Some(index) = shard.source_index(full, i) {
            let offset = index * width;
            match dtype {
                safetensors::Dtype::BF16 => {
                    bf16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]))
                }
                safetensors::Dtype::F16 => bf16::from_f32(
                    half::f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]]))
                        .to_f32(),
                ),
                _ => bf16::from_f32(f32::from_le_bytes(
                    data[offset..offset + 4].try_into().unwrap(),
                )),
            }
        } else {
            bf16::ZERO
        };
        out.push(value);
    }
    Ok(out)
}
