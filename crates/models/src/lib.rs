//! Qwen3/GPT-OSS configuration, checkpoint loading, and model execution.
#![forbid(unsafe_code)]

mod checkpoint;
mod config;
mod gpt_oss;
mod gpt_oss_config;
mod model;
mod qwen3;
mod weights;

pub use config::Qwen3Config;
pub use gpt_oss::GptOss;
pub use gpt_oss_config::GptOssConfig;
pub use model::{Model, ModelConfig, ModelDimensions};
pub use qwen3::{LayerKv, Qwen3};

use rsglang_core::Error;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[cfg(test)]
mod tests;
