//! GPT-OSS architecture validation and YaRN frequencies.
use crate::{invalid, ModelDimensions};
use rsglang_core::Result;
use rsglang_distributed::TensorParallel;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct GptOssConfig {
    #[serde(flatten)]
    pub dimensions: ModelDimensions,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    #[serde(default = "default_limit")]
    pub swiglu_limit: f32,
    pub attention_bias: bool,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub eos_token_id: serde_json::Value,
    #[serde(default)]
    pub rope_scaling: Option<YarnConfig>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}
fn default_limit() -> f32 {
    7.0
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YarnConfig {
    pub rope_type: String,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    #[serde(default = "yes")]
    pub truncate: bool,
    #[serde(default)]
    pub attention_factor: Option<f32>,
}
fn yes() -> bool {
    true
}
impl GptOssConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)
            .map_err(|e| invalid(e.to_string()))?;
        c.validate_tp(TensorParallel::default())?;
        Ok(c)
    }
    pub fn validate_tp(&self, tp: TensorParallel) -> Result<()> {
        let d = self.dimensions;
        if self.model_type != "gpt_oss"
            || self.architectures != ["GptOssForCausalLM"]
            || !self.attention_bias
            || self.hidden_act != "silu"
            || self.tie_word_embeddings
        {
            return Err(invalid("unsupported GPT-OSS architecture variant"));
        }
        if [
            d.hidden_size,
            d.intermediate_size,
            d.num_hidden_layers,
            d.num_attention_heads,
            d.num_key_value_heads,
            d.head_dim,
            d.vocab_size,
            d.max_position_embeddings,
            self.num_local_experts,
            self.num_experts_per_tok,
            self.sliding_window,
        ]
        .contains(&0)
            || d.head_dim > 256
            || !d.head_dim.is_multiple_of(2)
            || !d.num_attention_heads.is_multiple_of(d.num_key_value_heads)
            || self.num_experts_per_tok > self.num_local_experts
            || self.layer_types.len() != d.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|s| !["full_attention", "sliding_attention"].contains(&s.as_str()))
        {
            return Err(invalid(
                "invalid GPT-OSS dimensions, routing, or layer types",
            ));
        }
        for (a, b) in [
            (d.vocab_size, d.hidden_size),
            (d.num_attention_heads, d.head_dim),
            (d.hidden_size, d.intermediate_size),
            (self.num_local_experts, d.hidden_size),
        ] {
            if a.checked_mul(b)
                .and_then(|n| n.checked_mul(2))
                .is_none_or(|n| n > i32::MAX as usize)
            {
                return Err(invalid("GPT-OSS dimensions exceed backend limits"));
            }
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.swiglu_limit.is_finite()
            || self.swiglu_limit <= 0.0
        {
            return Err(invalid(
                "invalid GPT-OSS normalization, RoPE, or activation parameters",
            ));
        }
        if let Some(q) = &self.quantization_config {
            if q.get("quant_method").and_then(|v| v.as_str()) != Some("mxfp4") {
                return Err(invalid("GPT-OSS quantization must be MXFP4"));
            }
        }
        if let Some(y) = &self.rope_scaling {
            if y.rope_type != "yarn"
                || !y.factor.is_finite()
                || y.factor < 1.0
                || y.original_max_position_embeddings == 0
                || !y.beta_fast.is_finite()
                || !y.beta_slow.is_finite()
                || y.beta_slow <= 0.0
                || y.beta_fast < y.beta_slow
                || y.attention_factor
                    .is_some_and(|a| !a.is_finite() || a <= 0.0)
            {
                return Err(invalid("invalid or unsupported GPT-OSS YaRN parameters"));
            }
        }
        tp.partition(d.num_attention_heads)?;
        tp.kv_heads(d.num_key_value_heads)?;
        tp.partition(d.intermediate_size)?;
        tp.vocab(d.vocab_size)?;
        self.eos_ids(&self.eos_token_id)?;
        Ok(())
    }
    fn eos_ids(&self, value: &serde_json::Value) -> Result<Vec<u32>> {
        let values = match value {
            serde_json::Value::Array(v) => v.clone(),
            v => vec![v.clone()],
        };
        if values.is_empty() {
            return Err(invalid("empty EOS list"));
        }
        values
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|n| (*n as usize) < self.dimensions.vocab_size)
                    .ok_or_else(|| invalid("invalid eos_token_id"))
            })
            .collect()
    }
    pub fn generation_eos_ids(&self, path: &Path) -> Result<Vec<u32>> {
        let p = path.join("generation_config.json");
        if !p.exists() {
            return self.eos_ids(&self.eos_token_id);
        }
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(p)?).map_err(|e| invalid(e.to_string()))?;
        self.eos_ids(value.get("eos_token_id").unwrap_or(&self.eos_token_id))
    }
    pub fn rope_parameters(&self) -> (Vec<f32>, f32) {
        let dim = self.dimensions.head_dim;
        let mut frequencies: Vec<f32> = (0..dim / 2)
            .map(|j| 1.0 / (self.rope_theta as f64).powf((2 * j) as f64 / dim as f64) as f32)
            .collect();
        let Some(y) = &self.rope_scaling else {
            return (frequencies, 1.0);
        };
        let correction = |rot: f32| {
            (dim as f64
                * (y.original_max_position_embeddings as f64
                    / (rot as f64 * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * (self.rope_theta as f64).ln())) as f32
        };
        let mut low = correction(y.beta_fast);
        let mut high = correction(y.beta_slow);
        if y.truncate {
            low = low.floor();
            high = high.ceil();
        }
        low = low.max(0.0);
        high = high.min((dim - 1) as f32);
        if low == high {
            high += 0.001;
        }
        for (j, v) in frequencies.iter_mut().enumerate() {
            let ramp = ((j as f32 - low) / (high - low)).clamp(0.0, 1.0);
            *v = (*v / y.factor) * ramp + *v * (1.0 - ramp);
        }
        (
            frequencies,
            y.attention_factor.unwrap_or(1.0 + 0.1 * y.factor.ln()),
        )
    }
}
