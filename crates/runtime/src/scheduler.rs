use crate::{
    metrics::Metrics,
    tokenizer::{IncrementalDecoder, TextProcessor},
};
use rsglang_cache::{CacheManager, PageLease};
use rsglang_core::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::sync::{mpsc, OwnedSemaphorePermit};

pub(crate) struct Submission {
    pub id: RequestId,
    pub ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub events: mpsc::Sender<GenerationEvent>,
    pub cancel: Arc<AtomicBool>,
    pub permit: Option<OwnedSemaphorePermit>,
}
struct Request {
    input: Submission,
    lease: PageLease,
    tokens: Vec<u32>,
    prompt_len: usize,
    computed: usize,
    generated: usize,
    decoder: IncrementalDecoder,
}
pub(crate) struct Scheduler<R: ModelRunner> {
    pub runner: R,
    pub cache: CacheManager,
    config: RuntimeConfig,
    waiting: VecDeque<Submission>,
    active: Vec<Request>,
    prefer_decode: bool,
    eos: Vec<u32>,
    text: Option<Arc<TextProcessor>>,
    pub metrics: Arc<Metrics>,
}
impl<R: ModelRunner> Scheduler<R> {
    pub fn new(
        runner: R,
        num_pages: usize,
        config: RuntimeConfig,
        eos: Vec<u32>,
        text: Option<Arc<TextProcessor>>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            runner,
            cache: CacheManager::new(num_pages, config.page_size, config.prefix_cache),
            config,
            waiting: VecDeque::new(),
            active: vec![],
            prefer_decode: true,
            eos,
            text,
            metrics,
        }
    }
    pub fn submit(&mut self, input: Submission) {
        self.metrics.submitted.fetch_add(1, Ordering::Relaxed);
        self.waiting.push_back(input);
        self.update_metrics();
    }
    pub fn idle(&self) -> bool {
        self.waiting.is_empty() && self.active.is_empty()
    }
    fn update_metrics(&self) {
        self.metrics
            .free_pages
            .store(self.cache.free_pages() as u64, Ordering::Relaxed);
        self.metrics
            .cached_pages
            .store(self.cache.cached_pages() as u64, Ordering::Relaxed);
        self.metrics
            .waiting
            .store(self.waiting.len() as u64, Ordering::Release);
        self.metrics
            .running
            .store(self.active.len() as u64, Ordering::Release);
    }
    fn terminal(&mut self, mut req: Request, mut reason: FinishReason, message: Option<String>) {
        if let Some(message) = message {
            let _ = req.input.events.try_send(GenerationEvent::Error {
                request_id: req.input.id,
                message,
            });
        }
        if matches!(reason, FinishReason::Stop | FinishReason::Length) {
            self.cache
                .publish(&mut req.lease, &req.tokens, req.computed);
            if let Some(text) = &self.text {
                match req
                    .decoder
                    .flush(&text.tokenizer, &req.tokens[req.prompt_len..])
                {
                    Ok(tail) if !tail.is_empty() => {
                        let _ = req.input.events.try_send(GenerationEvent::Text {
                            request_id: req.input.id,
                            text: tail,
                        });
                    }
                    Err(e) => {
                        reason = FinishReason::Error;
                        let _ = req.input.events.try_send(GenerationEvent::Error {
                            request_id: req.input.id,
                            message: e.to_string(),
                        });
                    }
                    _ => {}
                }
            }
        }
        let _ = req.input.events.try_send(GenerationEvent::Finished {
            request_id: req.input.id,
            reason,
            prompt_tokens: req.prompt_len,
            completion_tokens: req.generated,
            cached_tokens: req.lease.cached_tokens(),
        });
        match reason {
            FinishReason::Stop | FinishReason::Length => {
                self.metrics.completed.fetch_add(1, Ordering::Relaxed);
            }
            FinishReason::Cancelled => {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            FinishReason::Error => {
                self.metrics.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.runner.forget_request(req.input.id);
        self.cache.release(req.lease);
    }
    fn cancel_disconnected(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].input.cancel.load(Ordering::Acquire)
                || self.active[i].input.events.is_closed()
            {
                let req = self.active.remove(i);
                self.terminal(req, FinishReason::Cancelled, None);
            } else {
                i += 1;
            }
        }
        let mut remaining = VecDeque::new();
        while let Some(req) = self.waiting.pop_front() {
            if req.cancel.load(Ordering::Acquire) || req.events.is_closed() {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                let _ = req.events.try_send(GenerationEvent::Finished {
                    request_id: req.id,
                    reason: FinishReason::Cancelled,
                    prompt_tokens: req.ids.len(),
                    completion_tokens: 0,
                    cached_tokens: 0,
                });
            } else {
                remaining.push_back(req);
            }
        }
        self.waiting = remaining;
    }
    fn admit(&mut self) -> Result<()> {
        while self.active.len() < self.config.max_running {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let Some(lease) = self.cache.acquire(&front.ids, front.sampling.max_tokens)? else {
                break;
            };
            let mut input = self.waiting.pop_front().unwrap();
            input.permit.take();
            self.metrics
                .prompt_tokens
                .fetch_add(input.ids.len() as u64, Ordering::Relaxed);
            self.metrics
                .cached_tokens
                .fetch_add(lease.cached_tokens() as u64, Ordering::Relaxed);
            self.active.push(Request {
                tokens: input.ids.clone(),
                prompt_len: input.ids.len(),
                computed: lease.cached_tokens(),
                generated: 0,
                input,
                lease,
                decoder: IncrementalDecoder::default(),
            });
        }
        Ok(())
    }
    pub fn step(&mut self) -> Result<bool> {
        self.cancel_disconnected();
        self.admit()?;
        let prefill = self.active.iter().any(|r| r.computed < r.prompt_len);
        let decode = self.active.iter().any(|r| r.computed >= r.prompt_len);
        if !prefill && !decode {
            self.update_metrics();
            return Ok(false);
        }
        let phase = if decode && (!prefill || self.prefer_decode) {
            BatchPhase::Decode
        } else {
            BatchPhase::Prefill
        };
        self.prefer_decode = phase == BatchPhase::Prefill;
        let mut budget = self.config.prefill_budget;
        let mut plans = vec![];
        let mut sequences = vec![];
        for (i, r) in self.active.iter_mut().enumerate() {
            let is_prefill = r.computed < r.prompt_len;
            if is_prefill != (phase == BatchPhase::Prefill) {
                continue;
            }
            let n = if is_prefill {
                if budget == 0 {
                    break;
                }
                let n = budget.min(r.prompt_len - r.computed);
                budget -= n;
                n
            } else {
                1
            };
            let end = r.computed + n;
            self.cache.extend(&mut r.lease, end);
            let sampling = if end >= r.prompt_len {
                Some(r.input.sampling.clone())
            } else {
                None
            };
            sequences.push(SequenceStep {
                request_id: r.input.id,
                input_ids: r.tokens[r.computed..end].to_vec(),
                start_pos: r.computed,
                pages: r.lease.pages().to_vec(),
                sampling,
            });
            plans.push((i, end));
        }
        let batch = StepBatch { phase, sequences };
        let outputs = self.runner.run(&batch)?;
        let output_count = outputs.len();
        let mut outputs: HashMap<_, _> = outputs
            .into_iter()
            .map(|o| (o.request_id, o.token_id))
            .collect();
        let expected = batch
            .sequences
            .iter()
            .filter(|s| s.sampling.is_some())
            .count();
        if output_count != expected
            || outputs.len() != expected
            || batch
                .sequences
                .iter()
                .filter(|s| s.sampling.is_some())
                .any(|s| !outputs.contains_key(&s.request_id))
        {
            return Err(Error::Backend(
                "runner output IDs do not match sampled batch".into(),
            ));
        }
        match phase {
            BatchPhase::Prefill => {
                self.metrics.prefill_batches.fetch_add(1, Ordering::Relaxed);
            }
            BatchPhase::Decode => {
                self.metrics.decode_batches.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut finished = vec![];
        for (i, end) in plans {
            let r = &mut self.active[i];
            r.computed = end;
            if phase == BatchPhase::Prefill && end == r.prompt_len {
                self.cache.publish(&mut r.lease, &r.tokens, r.computed);
            }
            if let Some(token_id) = outputs.remove(&r.input.id) {
                r.tokens.push(token_id);
                r.generated += 1;
                self.metrics
                    .generated_tokens
                    .fetch_add(1, Ordering::Relaxed);
                let text = if let Some(t) = &self.text {
                    r.decoder.step(&t.tokenizer, token_id)?
                } else {
                    String::new()
                };
                let event = GenerationEvent::Token {
                    request_id: r.input.id,
                    token_id,
                    text,
                };
                let reason = if r.input.cancel.load(Ordering::Acquire)
                    || r.input.events.try_send(event).is_err()
                {
                    Some(FinishReason::Cancelled)
                } else if !r.input.sampling.ignore_eos && self.eos.contains(&token_id) {
                    Some(FinishReason::Stop)
                } else if r.generated >= r.input.sampling.max_tokens {
                    Some(FinishReason::Length)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    finished.push((i, reason));
                }
            }
        }
        for (i, reason) in finished.into_iter().rev() {
            let req = self.active.remove(i);
            self.terminal(req, reason, None);
        }
        if let Some(bytes) = self.runner.memory_bytes() {
            self.metrics
                .peak_memory_bytes
                .fetch_max(bytes, Ordering::Relaxed);
        }
        self.update_metrics();
        #[cfg(debug_assertions)]
        self.cache
            .check_integrity(&self.active.iter().map(|r| &r.lease).collect::<Vec<_>>());
        Ok(true)
    }
    pub fn shutdown(&mut self, message: Option<String>) {
        while let Some(req) = self.active.pop() {
            self.terminal(
                req,
                if message.is_some() {
                    FinishReason::Error
                } else {
                    FinishReason::Cancelled
                },
                message.clone(),
            );
        }
        while let Some(req) = self.waiting.pop_front() {
            if message.is_some() {
                self.metrics.failed.fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            let _ = req.events.try_send(GenerationEvent::Error {
                request_id: req.id,
                message: message
                    .clone()
                    .unwrap_or_else(|| "engine shutting down".into()),
            });
        }
        self.cache.clear_idle_cache();
        self.update_metrics();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Fake {
        batches: Vec<StepBatch>,
    }
    impl ModelRunner for Fake {
        fn run(&mut self, b: &StepBatch) -> Result<Vec<TokenOutput>> {
            self.batches.push(b.clone());
            Ok(b.sequences
                .iter()
                .filter(|s| s.sampling.is_some())
                .map(|s| TokenOutput {
                    request_id: s.request_id,
                    token_id: 42,
                })
                .collect())
        }
    }
    fn request(
        id: u64,
        ids: Vec<u32>,
        max_tokens: usize,
    ) -> (Submission, mpsc::Receiver<GenerationEvent>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        (
            Submission {
                id,
                ids,
                sampling: SamplingParams {
                    max_tokens,
                    ..Default::default()
                },
                events: tx,
                cancel: cancel.clone(),
                permit: None,
            },
            rx,
            cancel,
        )
    }
    fn scheduler() -> Scheduler<Fake> {
        Scheduler::new(
            Fake::default(),
            128,
            RuntimeConfig {
                page_size: 4,
                prefill_budget: 4,
                ..Default::default()
            },
            vec![99],
            None,
            Arc::new(Metrics::default()),
        )
    }
    #[test]
    fn chunking_and_fairness() {
        let mut s = scheduler();
        let (a, _ra, _) = request(1, vec![1, 2], 3);
        s.submit(a);
        s.step().unwrap();
        let (b, _rb, _) = request(2, (0..10).collect(), 2);
        s.submit(b);
        for _ in 0..8 {
            s.step().unwrap();
        }
        let phases: Vec<_> = s.runner.batches.iter().map(|b| b.phase).collect();
        assert_eq!(
            &phases[..4],
            &[
                BatchPhase::Prefill,
                BatchPhase::Decode,
                BatchPhase::Prefill,
                BatchPhase::Decode
            ]
        );
        let chunks: Vec<_> = s
            .runner
            .batches
            .iter()
            .flat_map(|b| b.sequences.iter())
            .filter(|q| q.request_id == 2 && q.start_pos < 10)
            .collect();
        assert_eq!(
            chunks.iter().map(|q| q.input_ids.len()).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
        assert!(chunks[..2].iter().all(|q| q.sampling.is_none()));
        assert!(s.idle());
        s.cache.clear_idle_cache();
        s.cache.check_integrity(&[]);
    }
    #[test]
    fn cancellation_in_waiting_prefill_and_decode() {
        let mut s = scheduler();
        let (a, _ra, ca) = request(1, vec![1; 20], 3);
        s.submit(a);
        s.step().unwrap();
        ca.store(true, Ordering::Release);
        s.step().unwrap();
        let (b, _rb, cb) = request(2, vec![1; 2], 3);
        s.submit(b);
        s.step().unwrap();
        cb.store(true, Ordering::Release);
        s.step().unwrap();
        let (c, _rc, cc) = request(3, vec![1; 2], 3);
        cc.store(true, Ordering::Release);
        s.submit(c);
        s.step().unwrap();
        assert_eq!(s.metrics.snapshot().cancelled, 3);
        assert!(s.idle());
        s.cache.clear_idle_cache();
        s.cache.check_integrity(&[]);
    }
    #[test]
    fn repeated_prompt_hits_prefix_cache() {
        let mut s = scheduler();
        let p: Vec<_> = (0..12).collect();
        let (a, _ra, _) = request(1, p.clone(), 1);
        s.submit(a);
        while s.step().unwrap() {}
        let (b, _rb, _) = request(2, p, 1);
        s.submit(b);
        s.step().unwrap();
        assert_eq!(s.runner.batches.last().unwrap().sequences[0].start_pos, 8);
        assert_eq!(s.metrics.snapshot().cached_tokens, 8);
        s.cache.clear_idle_cache();
        s.cache.check_integrity(&[]);
    }
    #[test]
    fn eos_ends_without_another_decode() {
        let mut s = scheduler();
        s.eos = vec![42];
        let (a, mut rx, _) = request(1, vec![1, 2], 8);
        s.submit(a);
        s.step().unwrap();
        assert!(s.idle());
        assert!(matches!(
            rx.try_recv().unwrap(),
            GenerationEvent::Token { token_id: 42, .. }
        ));
        assert!(matches!(
            rx.try_recv().unwrap(),
            GenerationEvent::Finished {
                reason: FinishReason::Stop,
                completion_tokens: 1,
                ..
            }
        ));
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    struct Runner;
    impl ModelRunner for Runner {
        fn run(&mut self, b: &StepBatch) -> Result<Vec<TokenOutput>> {
            Ok(b.sequences
                .iter()
                .filter(|q| q.sampling.is_some())
                .map(|q| TokenOutput {
                    request_id: q.request_id,
                    token_id: 7,
                })
                .collect())
        }
    }
    fn submission(
        id: u64,
        n: usize,
        max: usize,
        capacity: usize,
    ) -> (Submission, mpsc::Receiver<GenerationEvent>) {
        let (tx, rx) = mpsc::channel(capacity);
        (
            Submission {
                id,
                ids: vec![id as u32; n],
                sampling: SamplingParams {
                    max_tokens: max,
                    ..Default::default()
                },
                events: tx,
                cancel: Arc::new(AtomicBool::new(false)),
                permit: None,
            },
            rx,
        )
    }
    #[test]
    fn tiny_pool_waits_for_reserved_request_to_finish() {
        let metrics = Arc::new(Metrics::default());
        let mut s = Scheduler::new(
            Runner,
            5,
            RuntimeConfig {
                page_size: 4,
                prefill_budget: 128,
                ..Default::default()
            },
            vec![],
            None,
            metrics.clone(),
        );
        let (a, _ra) = submission(1, 12, 8, 32);
        let (b, _rb) = submission(2, 4, 4, 32);
        s.submit(a);
        s.submit(b);
        s.step().unwrap();
        assert_eq!(metrics.snapshot().running, 1);
        assert_eq!(metrics.snapshot().waiting, 1);
        for _ in 0..20 {
            if !s.step().unwrap() {
                break;
            }
        }
        assert!(s.idle());
        assert_eq!(metrics.snapshot().completed, 2);
        s.cache.clear_idle_cache();
        s.cache.check_integrity(&[]);
    }
    #[test]
    fn slow_consumer_is_cancelled_without_blocking_other_requests() {
        let metrics = Arc::new(Metrics::default());
        let mut s = Scheduler::new(
            Runner,
            32,
            RuntimeConfig {
                page_size: 4,
                ..Default::default()
            },
            vec![],
            None,
            metrics.clone(),
        );
        let (a, _ra) = submission(1, 4, 10, 1);
        let (b, _rb) = submission(2, 4, 3, 32);
        s.submit(a);
        s.submit(b);
        for _ in 0..6 {
            s.step().unwrap();
        }
        assert!(s.idle());
        assert_eq!(metrics.snapshot().cancelled, 1);
        assert_eq!(metrics.snapshot().completed, 1);
        s.cache.clear_idle_cache();
        s.cache.check_integrity(&[]);
    }
    #[test]
    fn error_cleanup_returns_all_pages() {
        struct Broken;
        impl ModelRunner for Broken {
            fn run(&mut self, _: &StepBatch) -> Result<Vec<TokenOutput>> {
                Err(Error::Backend("injected failure".into()))
            }
        }
        let mut s = Scheduler::new(
            Broken,
            32,
            RuntimeConfig::default(),
            vec![],
            None,
            Arc::new(Metrics::default()),
        );
        let (a, _rx) = submission(1, 20, 10, 32);
        s.submit(a);
        assert!(s.step().is_err());
        s.shutdown(Some("injected failure".into()));
        s.cache.check_integrity(&[]);
        assert_eq!(s.cache.free_pages(), 32);
    }
}
