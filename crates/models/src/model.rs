//! Architecture dispatch shared by the single-GPU and tensor-parallel engines.
use crate::{invalid, GptOss, GptOssConfig, LayerKv, Qwen3, Qwen3Config};
use rsglang_core::Result;
use rsglang_distributed::TensorParallel;
use rsglang_kernels::KernelBackend;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct ModelDimensions {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}
#[derive(Clone, Debug)]
pub enum ModelConfig {
    Qwen3(Qwen3Config),
    GptOss(GptOssConfig),
}
impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path.join("config.json"))?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        match value.get("model_type").and_then(|v| v.as_str()) {
            Some("qwen3" | "qwen3_moe") => Ok(Self::Qwen3(Qwen3Config::load(path)?)),
            Some("gpt_oss") => Ok(Self::GptOss(GptOssConfig::load(path)?)),
            other => Err(invalid(format!(
                "unsupported model_type {other:?}; supported: qwen3, qwen3_moe, gpt_oss"
            ))),
        }
    }
    pub fn dimensions(&self) -> ModelDimensions {
        match self {
            Self::GptOss(c) => c.dimensions,
            Self::Qwen3(c) => ModelDimensions {
                hidden_size: c.hidden_size,
                intermediate_size: c.intermediate_size,
                num_hidden_layers: c.num_hidden_layers,
                num_attention_heads: c.num_attention_heads,
                num_key_value_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                vocab_size: c.vocab_size,
                max_position_embeddings: c.max_position_embeddings,
            },
        }
    }
    pub fn validate_tp(&self, tp: TensorParallel) -> Result<()> {
        match self {
            Self::Qwen3(c) => c.validate_tp(tp),
            Self::GptOss(c) => c.validate_tp(tp),
        }
    }
    pub fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>> {
        match self {
            Self::Qwen3(c) => c.generation_eos_ids(path),
            Self::GptOss(c) => c.generation_eos_ids(path),
        }
    }
    pub fn kv_bytes_per_page_tp(&self, page_size: usize, tp: TensorParallel) -> Result<usize> {
        let c = self.dimensions();
        [
            c.num_hidden_layers,
            2,
            page_size,
            tp.kv_heads(c.num_key_value_heads)?.len(),
            c.head_dim,
            2,
        ]
        .into_iter()
        .try_fold(1usize, |a, b| {
            a.checked_mul(b).ok_or_else(|| invalid("KV size overflow"))
        })
    }
}
enum Implementation<B: KernelBackend> {
    Qwen3(Qwen3<B>),
    GptOss(GptOss<B>),
}
pub struct Model<B: KernelBackend> {
    config: ModelConfig,
    implementation: Implementation<B>,
}
impl<B: KernelBackend> Model<B> {
    pub fn load(backend: &B, path: &Path, config: ModelConfig) -> Result<Self> {
        let implementation = match &config {
            ModelConfig::Qwen3(c) => Implementation::Qwen3(Qwen3::load(backend, path, c.clone())?),
            ModelConfig::GptOss(c) => {
                Implementation::GptOss(GptOss::load(backend, path, c.clone())?)
            }
        };
        Ok(Self {
            config,
            implementation,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn allocate_kv(
        &self,
        b: &B,
        pages: usize,
        page_size: usize,
    ) -> Result<Vec<LayerKv<B::Tensor>>> {
        match &self.implementation {
            Implementation::Qwen3(m) => m.allocate_kv(b, pages, page_size),
            Implementation::GptOss(m) => m.allocate_kv(b, pages, page_size),
        }
    }
    pub fn forward_hidden(
        &self,
        b: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
    ) -> Result<B::Tensor> {
        match &self.implementation {
            Implementation::Qwen3(m) => m.forward_hidden(b, meta, kv),
            Implementation::GptOss(m) => m.forward_hidden(b, meta, kv),
        }
    }
    pub fn project(&self, b: &B, x: &B::Tensor, meta: &B::Metadata) -> Result<B::Logits> {
        match &self.implementation {
            Implementation::Qwen3(m) => m.project(b, x, meta),
            Implementation::GptOss(m) => m.project(b, x, meta),
        }
    }
}
