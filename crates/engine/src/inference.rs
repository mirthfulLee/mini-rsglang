use crate::*;
use rsglang_distributed::validate_devices;

/// Execution strategy with scheduler metadata and diagnostic logits.
/// Construct and own implementations on their worker thread; no Send/Sync is required.
pub trait ExecutionEngine: RankRunner {
    fn config(&self) -> &ModelConfig;
    fn num_pages(&self) -> usize;
}

/// Scheduler-facing runner backed by an execution strategy.
pub struct InferenceEngine {
    inner: Box<dyn ExecutionEngine>,
}
impl InferenceEngine {
    /// Wrap another execution strategy without changing scheduler or forwarding code.
    pub fn new(engine: impl ExecutionEngine + 'static) -> Self {
        Self {
            inner: Box::new(engine),
        }
    }
    pub fn load(
        path: &Path,
        devices: &[usize],
        kv_bytes: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        validate_devices(devices)?;
        if devices.len() == 1 {
            Ok(Self::new(Engine::load(
                path,
                devices[0],
                kv_bytes,
                page_size,
                max_seq_len,
            )?))
        } else {
            Ok(Self::new(ParallelEngine::load(
                path,
                devices,
                kv_bytes,
                page_size,
                max_seq_len,
            )?))
        }
    }
    pub fn config(&self) -> &ModelConfig {
        self.inner.config()
    }
    pub fn num_pages(&self) -> usize {
        self.inner.num_pages()
    }
    pub fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        self.inner.logits_rows(batch)
    }
}
impl ModelRunner for InferenceEngine {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>> {
        self.inner.run(batch)
    }
    fn forget_request(&mut self, id: RequestId) {
        self.inner.forget_request(id);
    }
    fn memory_bytes(&self) -> Option<u64> {
        self.inner.memory_bytes()
    }
}
impl RankRunner for InferenceEngine {
    fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        self.logits_rows(batch)
    }
}
