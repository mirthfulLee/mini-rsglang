#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub type RequestId = u64;
pub type PageId = u32;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("capacity exhausted: {0}")]
    Capacity(String),
    #[error("backend failure: {0}")]
    Backend(String),
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("engine stopped")]
    Stopped,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub ignore_eos: bool,
}
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
            seed: 0,
            max_tokens: 128,
            ignore_eos: false,
        }
    }
}
impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::Invalid(
                "temperature must be finite and nonnegative".into(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(Error::Invalid("top_p must be in (0, 1]".into()));
        }
        if self.top_k == Some(0) || self.max_tokens == 0 {
            return Err(Error::Invalid(
                "top_k and max_tokens must be positive".into(),
            ));
        }
        Ok(())
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == Some(1)
    }
}

#[derive(Clone, Debug)]
pub enum Prompt {
    Text(String),
    TokenIds(Vec<u32>),
}
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub prompt: Prompt,
    pub sampling: SamplingParams,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
}
#[derive(Clone, Debug, Serialize)]
pub enum GenerationEvent {
    Token {
        request_id: RequestId,
        token_id: u32,
        text: String,
    },
    Text {
        request_id: RequestId,
        text: String,
    },
    Finished {
        request_id: RequestId,
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
        cached_tokens: usize,
    },
    Error {
        request_id: RequestId,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
}
/// Each sequence extends its existing KV by input_ids.len() tokens, beginning at start_pos.
/// Only a final prefill chunk / decode step carries sampling parameters.
#[derive(Clone, Debug)]
pub struct SequenceStep {
    pub request_id: RequestId,
    pub input_ids: Vec<u32>,
    pub start_pos: usize,
    pub pages: Vec<PageId>,
    pub sampling: Option<SamplingParams>,
}
#[derive(Clone, Debug)]
pub struct StepBatch {
    pub phase: BatchPhase,
    pub sequences: Vec<SequenceStep>,
}
#[derive(Clone, Debug)]
pub struct TokenOutput {
    pub request_id: RequestId,
    pub token_id: u32,
}

/// run() returns only after GPU reads/writes and result transfers have completed.
/// The scheduler may then safely reclaim pages or cancel requests.
pub trait ModelRunner {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>>;
    fn forget_request(&mut self, _id: RequestId) {}
    fn memory_bytes(&self) -> Option<u64> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub max_seq_len: usize,
    pub page_size: usize,
    pub prefill_budget: usize,
    pub max_running: usize,
    pub max_waiting: usize,
    pub prefix_cache: bool,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            page_size: 16,
            prefill_budget: 512,
            max_running: 32,
            max_waiting: 256,
            prefix_cache: true,
        }
    }
}
impl RuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        if [
            self.max_seq_len,
            self.page_size,
            self.prefill_budget,
            self.max_running,
            self.max_waiting,
        ]
        .contains(&0)
        {
            return Err(Error::Invalid("runtime limits must be positive".into()));
        }
        if !self.page_size.is_power_of_two() || self.page_size > 256 {
            return Err(Error::Invalid(
                "page_size must be a power of two <= 256".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_and_out_of_range_sampling() {
        for temperature in [f32::NAN, f32::INFINITY, -1.] {
            assert!(SamplingParams {
                temperature,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for top_p in [0., -0.1, 1.1, f32::NAN] {
            assert!(SamplingParams {
                top_p,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        assert!(SamplingParams {
            max_tokens: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            top_k: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams::default().validate().is_ok());
    }
    #[test]
    fn runtime_limits_are_checked() {
        assert!(RuntimeConfig {
            page_size: 3,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RuntimeConfig {
            max_waiting: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RuntimeConfig::default().validate().is_ok());
    }
}
