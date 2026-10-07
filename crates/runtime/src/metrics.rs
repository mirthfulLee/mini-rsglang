use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
#[derive(Default)]
pub struct Metrics {
    pub submitted: AtomicU64,
    pub completed: AtomicU64,
    pub cancelled: AtomicU64,
    pub failed: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    pub prefill_batches: AtomicU64,
    pub decode_batches: AtomicU64,
    pub waiting: AtomicU64,
    pub running: AtomicU64,
    pub free_pages: AtomicU64,
    pub cached_pages: AtomicU64,
    pub peak_memory_bytes: AtomicU64,
    pub healthy: AtomicU64,
}
#[derive(Clone, Debug, Serialize)]
pub struct MetricsSnapshot {
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub cached_tokens: u64,
    pub prefill_batches: u64,
    pub decode_batches: u64,
    pub waiting: u64,
    pub running: u64,
    pub free_pages: u64,
    pub cached_pages: u64,
    pub peak_memory_bytes: u64,
    pub healthy: bool,
}
impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        let get = |v: &AtomicU64| v.load(Ordering::Acquire);
        MetricsSnapshot {
            submitted: get(&self.submitted),
            completed: get(&self.completed),
            cancelled: get(&self.cancelled),
            failed: get(&self.failed),
            prompt_tokens: get(&self.prompt_tokens),
            generated_tokens: get(&self.generated_tokens),
            cached_tokens: get(&self.cached_tokens),
            prefill_batches: get(&self.prefill_batches),
            decode_batches: get(&self.decode_batches),
            waiting: get(&self.waiting),
            running: get(&self.running),
            free_pages: get(&self.free_pages),
            cached_pages: get(&self.cached_pages),
            peak_memory_bytes: get(&self.peak_memory_bytes),
            healthy: get(&self.healthy) != 0,
        }
    }
    pub fn prometheus(&self) -> String {
        let value = serde_json::to_value(self.snapshot()).unwrap();
        let mut out = String::new();
        for (key, v) in value.as_object().unwrap() {
            let n = if let Some(b) = v.as_bool() {
                u64::from(b)
            } else {
                v.as_u64().unwrap()
            };
            out.push_str(&format!("rsglang_{key} {n}\n"));
        }
        out
    }
}
