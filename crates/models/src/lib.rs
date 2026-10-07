//! Qwen3 checkpoint configuration, tensor-parallel loading, and model execution.
#![forbid(unsafe_code)]

mod config;
mod qwen3;
mod weights;

pub use config::Qwen3Config;
pub use qwen3::{LayerKv, Qwen3};

use rsglang_core::Error;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[cfg(test)]
mod tests;
