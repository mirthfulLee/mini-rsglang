#![forbid(unsafe_code)]
mod metrics;
mod scheduler;
mod tokenizer;
use metrics::Metrics;
pub use metrics::MetricsSnapshot;
use rsglang_core::*;
use rsglang_engine::InferenceEngine;
use scheduler::{Scheduler, Submission};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
pub use tokenizer::{ChatMessage, TextProcessor};
use tokio::sync::{mpsc, oneshot, Semaphore};

#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub model_id: String,
    pub vocab_size: usize,
    pub num_pages: usize,
    pub devices: Vec<usize>,
    pub tensor_parallel_size: usize,
}
enum Command {
    Generate(Submission),
    ClearCache(oneshot::Sender<()>),
    Shutdown(oneshot::Sender<()>),
}
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Command>,
    permits: Arc<Semaphore>,
    next: Arc<AtomicU64>,
    text: Arc<TextProcessor>,
    info: Arc<EngineInfo>,
    metrics: Arc<Metrics>,
    config: Arc<RuntimeConfig>,
}
/// Dropping an event stream cancels the request, even if the submission queue is full.
pub struct GenerationStream {
    pub request_id: RequestId,
    rx: mpsc::Receiver<GenerationEvent>,
    cancel: Arc<AtomicBool>,
    terminal: bool,
}
impl GenerationStream {
    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        match self.rx.recv().await {
            Some(event) => {
                if matches!(
                    event,
                    GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
                ) {
                    self.terminal = true;
                }
                Some(event)
            }
            None if !self.terminal => {
                self.terminal = true;
                Some(GenerationEvent::Error {
                    request_id: self.request_id,
                    message:
                        "generation stream closed before completion (cancelled or engine stopped)"
                            .into(),
                })
            }
            None => None,
        }
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}
impl Drop for GenerationStream {
    fn drop(&mut self) {
        self.cancel();
    }
}
impl EngineHandle {
    pub fn info(&self) -> &EngineInfo {
        &self.info
    }
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
    pub fn text(&self) -> &TextProcessor {
        &self.text
    }
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
    pub fn prometheus_metrics(&self) -> String {
        self.metrics.prometheus()
    }
    pub fn load(
        model: &Path,
        device: usize,
        kv_bytes: usize,
        config: RuntimeConfig,
    ) -> Result<Self> {
        Self::load_tensor_parallel(model, &[device], kv_bytes, config)
    }
    pub fn load_tensor_parallel(
        model: &Path,
        devices: &[usize],
        kv_bytes_per_gpu: usize,
        config: RuntimeConfig,
    ) -> Result<Self> {
        config.validate()?;
        let text = Arc::new(TextProcessor::load(model)?);
        let (tx, mut rx) = mpsc::channel::<Command>(config.max_waiting);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let metrics = Arc::new(Metrics::default());
        let worker_metrics = metrics.clone();
        let worker_text = text.clone();
        let worker_config = config.clone();
        let path: PathBuf = model.to_owned();
        let devices = devices.to_vec();
        std::thread::Builder::new()
            .name("rsglang-gpu".into())
            .spawn(move || {
                let engine = match InferenceEngine::load(
                    &path,
                    &devices,
                    kv_bytes_per_gpu,
                    worker_config.page_size,
                    worker_config.max_seq_len,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let eos = match engine.config().generation_eos_ids(&path) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let info = EngineInfo {
                    model_id: path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    vocab_size: engine.config().vocab_size,
                    num_pages: engine.num_pages(),
                    tensor_parallel_size: devices.len(),
                    devices,
                };
                if let Some(bytes) = engine.memory_bytes() {
                    worker_metrics
                        .peak_memory_bytes
                        .store(bytes, Ordering::Relaxed);
                }
                let mut scheduler = Scheduler::new(
                    engine,
                    info.num_pages,
                    worker_config,
                    eos,
                    Some(worker_text),
                    worker_metrics.clone(),
                );
                worker_metrics.healthy.store(1, Ordering::Release);
                if ready_tx.send(Ok(info)).is_err() {
                    return;
                }
                let mut stop = false;
                let mut shutdown_reply = None;
                while !stop {
                    let first = if scheduler.idle() {
                        rx.blocking_recv()
                    } else {
                        None
                    };
                    if scheduler.idle() && first.is_none() {
                        break;
                    }
                    let mut commands = vec![];
                    if let Some(c) = first {
                        commands.push(c);
                    }
                    // Bound ingress work per iteration so producers cannot starve GPU execution.
                    for _ in 0..scheduler_config_ingress_limit(&scheduler) {
                        match rx.try_recv() {
                            Ok(c) => commands.push(c),
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                stop = true;
                                break;
                            }
                            Err(mpsc::error::TryRecvError::Empty) => break,
                        }
                    }
                    for c in commands {
                        match c {
                            Command::Generate(input) => scheduler.submit(input),
                            Command::ClearCache(reply) => {
                                scheduler.cache.clear_idle_cache();
                                let _ = reply.send(());
                            }
                            Command::Shutdown(reply) => {
                                shutdown_reply = Some(reply);
                                stop = true;
                            }
                        }
                    }
                    if stop {
                        break;
                    }
                    match scheduler.step() {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::error!(error=%e,"GPU worker failed");
                            scheduler.shutdown(Some(e.to_string()));
                            stop = true;
                        }
                    }
                }
                scheduler.shutdown(None);
                drop(scheduler);
                worker_metrics.healthy.store(0, Ordering::Release);
                if let Some(reply) = shutdown_reply {
                    let _ = reply.send(());
                }
            })?;
        let info = ready_rx.recv().map_err(|_| Error::Stopped)??;
        Ok(Self {
            tx,
            permits: Arc::new(Semaphore::new(config.max_waiting)),
            next: Arc::new(AtomicU64::new(1)),
            text,
            info: Arc::new(info),
            metrics,
            config: Arc::new(config),
        })
    }
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerationStream> {
        request.sampling.validate()?;
        if !self.healthy() {
            return Err(Error::Stopped);
        }
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Capacity("waiting queue is full".into()))?;
        let ids = match request.prompt {
            Prompt::TokenIds(ids) => ids,
            Prompt::Text(text) => {
                let processor = self.text.clone();
                tokio::task::spawn_blocking(move || processor.encode(&text))
                    .await
                    .map_err(|e| Error::Backend(e.to_string()))??
            }
        };
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= self.info.vocab_size) {
            return Err(Error::Invalid(
                "prompt must contain valid nonempty token IDs".into(),
            ));
        }
        let length = ids
            .len()
            .checked_add(request.sampling.max_tokens)
            .ok_or_else(|| Error::Invalid("sequence length overflow".into()))?;
        if length > self.config.max_seq_len {
            return Err(Error::Invalid(format!(
                "prompt + max_tokens exceeds context limit {}",
                self.config.max_seq_len
            )));
        }
        if length.div_ceil(self.config.page_size) > self.info.num_pages {
            return Err(Error::Capacity("request exceeds total KV capacity".into()));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (events, rx) = mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        self.tx
            .try_send(Command::Generate(Submission {
                id,
                ids,
                sampling: request.sampling,
                events,
                cancel: cancel.clone(),
                permit: Some(permit),
            }))
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => {
                    Error::Capacity("ingress queue is full".into())
                }
                mpsc::error::TrySendError::Closed(_) => Error::Stopped,
            })?;
        Ok(GenerationStream {
            request_id: id,
            rx,
            cancel,
            terminal: false,
        })
    }
    pub fn healthy(&self) -> bool {
        self.metrics.healthy.load(Ordering::Acquire) != 0
    }
    pub async fn clear_cache(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::ClearCache(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
    pub async fn shutdown(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Shutdown(tx))
            .await
            .map_err(|_| Error::Stopped)?;
        rx.await.map_err(|_| Error::Stopped)
    }
}
fn scheduler_config_ingress_limit<R: ModelRunner>(_: &Scheduler<R>) -> usize {
    256
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    #[tokio::test]
    async fn premature_closure_is_an_error_then_eof() {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        let cancel = Arc::new(AtomicBool::new(false));
        let mut s = GenerationStream {
            request_id: 7,
            rx,
            cancel: cancel.clone(),
            terminal: false,
        };
        assert!(matches!(
            s.recv().await,
            Some(GenerationEvent::Error { request_id: 7, .. })
        ));
        assert!(s.recv().await.is_none());
        drop(s);
        assert!(cancel.load(Ordering::Acquire));
    }
    #[tokio::test]
    async fn normal_completion_does_not_synthesize_an_error() {
        let (tx, rx) = mpsc::channel(1);
        tx.send(GenerationEvent::Finished {
            request_id: 7,
            reason: FinishReason::Length,
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: 0,
        })
        .await
        .unwrap();
        drop(tx);
        let mut s = GenerationStream {
            request_id: 7,
            rx,
            cancel: Arc::new(AtomicBool::new(false)),
            terminal: false,
        };
        assert!(matches!(
            s.recv().await,
            Some(GenerationEvent::Finished { .. })
        ));
        assert!(s.recv().await.is_none());
    }
}
