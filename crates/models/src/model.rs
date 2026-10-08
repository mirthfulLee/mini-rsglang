//! Model contracts and the centralized checkpoint loading factory.
use crate::{invalid, GptOss, GptOssConfig, Qwen3, Qwen3Config};
use rsglang_core::Result;
use rsglang_distributed::TensorParallel;
use rsglang_kernels::KernelBackend;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
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

/// Model-specific metadata and validation, independent of GPU execution.
pub trait ModelConfiguration {
    fn dimensions(&self) -> ModelDimensions;
    fn validate_tp(&self, tp: TensorParallel) -> Result<()>;
    fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>>;

    /// The common paged K/V layout; architectures with other layouts can override it.
    fn kv_bytes_per_page_tp(&self, page_size: usize, tp: TensorParallel) -> Result<usize> {
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
    /// One registration point for the behavior of strongly typed configurations.
    pub fn configuration(&self) -> &dyn ModelConfiguration {
        match self {
            Self::Qwen3(c) => c,
            Self::GptOss(c) => c,
        }
    }
    pub fn dimensions(&self) -> ModelDimensions {
        self.configuration().dimensions()
    }
    pub fn validate_tp(&self, tp: TensorParallel) -> Result<()> {
        self.configuration().validate_tp(tp)
    }
    pub fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>> {
        self.configuration().generation_eos_ids(path)
    }
    pub fn kv_bytes_per_page_tp(&self, page_size: usize, tp: TensorParallel) -> Result<usize> {
        self.configuration().kv_bytes_per_page_tp(page_size, tp)
    }
}

/// Rank-local physical K/V buffers, owned by the engine and borrowed during forward.
pub struct LayerKv<T> {
    pub k: T,
    pub v: T,
}

/// Object-safe model execution contract. Each rank constructs and owns its model;
/// CUDA resources are never required to cross threads through this trait object.
pub trait InferenceModel<B: KernelBackend> {
    fn dimensions(&self) -> ModelDimensions;

    /// Allocate the standard paged K/V layout used by Qwen3 and GPT-OSS.
    fn allocate_kv(
        &self,
        backend: &B,
        pages: usize,
        page_size: usize,
    ) -> Result<Vec<LayerKv<B::Tensor>>> {
        let c = self.dimensions();
        let shape = [
            pages,
            page_size,
            backend
                .tensor_parallel()
                .kv_heads(c.num_key_value_heads)?
                .len(),
            c.head_dim,
        ];
        (0..c.num_hidden_layers)
            .map(|_| {
                Ok(LayerKv {
                    k: backend.zeros(&shape)?,
                    v: backend.zeros(&shape)?,
                })
            })
            .collect()
    }

    fn forward_hidden(
        &self,
        backend: &B,
        meta: &B::Metadata,
        kv: &mut [LayerKv<B::Tensor>],
    ) -> Result<B::Tensor>;

    fn project(&self, backend: &B, hidden: &B::Tensor, meta: &B::Metadata) -> Result<B::Logits>;
}

/// Select the concrete architecture once when loading a checkpoint.
pub fn load_model<B: KernelBackend + 'static>(
    backend: &B,
    path: &Path,
    config: &ModelConfig,
) -> Result<Box<dyn InferenceModel<B>>>
where
    B::Tensor: 'static,
{
    match config {
        ModelConfig::Qwen3(c) => Ok(Box::new(Qwen3::load(backend, path, c.clone())?)),
        ModelConfig::GptOss(c) => Ok(Box::new(GptOss::load(backend, path, c.clone())?)),
    }
}
