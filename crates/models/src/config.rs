//! Checkpoint architecture, dimensions, EOS metadata, and KV sizing.
use crate::{invalid, ModelConfiguration, ModelDimensions};
use rsglang_core::Result;
use rsglang_distributed::TensorParallel;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub hidden_act: String,
    pub eos_token_id: serde_json::Value,
    #[serde(default, alias = "dtype")]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}
fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}
impl Qwen3Config {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)
            .map_err(|e| invalid(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<()> {
        if !((self.model_type == "qwen3"
            && self.architectures == ["Qwen3ForCausalLM"]
            && self.num_experts == 0)
            || (self.model_type == "qwen3_moe"
                && self.architectures == ["Qwen3MoeForCausalLM"]
                && self.num_experts > 0))
        {
            return Err(invalid(
                "only Qwen3ForCausalLM and Qwen3MoeForCausalLM are supported",
            ));
        }
        if self.num_experts > 0
            && (self.moe_intermediate_size == 0
                || self.num_experts_per_tok == 0
                || self.num_experts_per_tok > self.num_experts
                || self.decoder_sparse_step == 0
                || self
                    .mlp_only_layers
                    .iter()
                    .any(|&i| i >= self.num_hidden_layers))
        {
            return Err(invalid("invalid Qwen3 MoE configuration"));
        }
        if self.rope_scaling.is_some()
            || self.use_sliding_window
            || self.sliding_window.is_some()
            || self.attention_bias
            || self.hidden_act != "silu"
        {
            return Err(invalid("RoPE scaling, sliding window, biased attention, or non-SiLU architecture unsupported"));
        }
        if [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.vocab_size,
            self.max_position_embeddings,
        ]
        .contains(&0)
            || self.head_dim == 0
            || self.head_dim > 256
            || !self.head_dim.is_multiple_of(2)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(invalid("invalid Qwen3 dimensions or GQA ratio"));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(invalid("invalid normalization or RoPE parameters"));
        }
        for dims in [
            (self.num_attention_heads, self.head_dim),
            (self.num_key_value_heads, self.head_dim),
            (self.vocab_size, self.hidden_size),
            (self.hidden_size, self.intermediate_size),
            (self.hidden_size, self.moe_intermediate_size),
            (self.num_experts, self.hidden_size),
        ] {
            if dims
                .0
                .checked_mul(dims.1)
                .is_none_or(|n| n > i32::MAX as usize)
            {
                return Err(invalid("model dimensions exceed backend limits"));
            }
        }
        self.eos_ids()?;
        Ok(())
    }
    pub fn is_sparse_layer(&self, layer: usize) -> bool {
        self.num_experts > 0
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
            && !self.mlp_only_layers.contains(&layer)
    }
    pub fn validate_tp(&self, tp: TensorParallel) -> Result<()> {
        self.validate()?;
        tp.partition(self.num_attention_heads)?;
        tp.kv_heads(self.num_key_value_heads)?;
        tp.partition(self.intermediate_size)?;
        tp.vocab(self.vocab_size)?;
        if self.num_experts > 0 {
            tp.partition(self.moe_intermediate_size)?;
        }
        Ok(())
    }
    pub fn eos_ids(&self) -> Result<Vec<u32>> {
        self.parse_eos_ids(&self.eos_token_id)
    }
    /// Generation metadata may specify additional EOS IDs (Qwen3 has both im_end and endoftext).
    pub fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>> {
        let file = path.join("generation_config.json");
        if !file.exists() {
            return self.eos_ids();
        }
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(file)?).map_err(|e| invalid(e.to_string()))?;
        self.parse_eos_ids(value.get("eos_token_id").unwrap_or(&self.eos_token_id))
    }
    fn parse_eos_ids(&self, value: &serde_json::Value) -> Result<Vec<u32>> {
        let values = match value {
            serde_json::Value::Array(v) => v.clone(),
            v => vec![v.clone()],
        };
        if values.is_empty() {
            return Err(invalid("empty EOS token list"));
        }
        values
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| (n as usize) < self.vocab_size)
                    .ok_or_else(|| invalid("invalid eos_token_id"))
            })
            .collect()
    }
    pub fn kv_bytes_per_page(&self, page_size: usize) -> Result<usize> {
        self.kv_bytes_per_page_tp(page_size, TensorParallel::default())
    }
    pub fn kv_bytes_per_page_tp(&self, page_size: usize, tp: TensorParallel) -> Result<usize> {
        <Self as ModelConfiguration>::kv_bytes_per_page_tp(self, page_size, tp)
    }
}

impl ModelConfiguration for Qwen3Config {
    fn dimensions(&self) -> ModelDimensions {
        ModelDimensions {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
        }
    }
    fn validate_tp(&self, tp: TensorParallel) -> Result<()> {
        Qwen3Config::validate_tp(self, tp)
    }
    fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>> {
        Qwen3Config::generation_eos_ids(self, path)
    }
}
