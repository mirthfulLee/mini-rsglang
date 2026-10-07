use rsglang_core::{Error, ModelRunner, RequestId, Result, StepBatch, TokenOutput};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        mpsc::{self, Receiver, SyncSender},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Optional diagnostic path, using the same forward and collectives as generation.
pub trait RankRunner: ModelRunner {
    fn logits_rows(&mut self, _batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        Err(Error::Invalid(
            "runner does not expose diagnostic logits".into(),
        ))
    }
}
enum Command {
    Step(u64, Arc<StepBatch>, bool),
    Forget(RequestId),
    Stop,
}
enum Output {
    Tokens(Vec<TokenOutput>),
    Logits(Vec<Vec<f32>>),
}
struct Reply {
    rank: usize,
    step: u64,
    output: Result<Output>,
    memory: Option<u64>,
}

/// Each rank constructs/owns its runner on its thread. Only owned CPU messages cross threads.
pub struct RankGroup {
    commands: Vec<SyncSender<Command>>,
    replies: Receiver<Reply>,
    joins: Vec<JoinHandle<()>>,
    abort: Arc<dyn Fn() + Send + Sync>,
    timeout: Duration,
    step: u64,
    failed: bool,
    memory: Vec<Option<u64>>,
}
impl RankGroup {
    pub fn spawn<R: RankRunner + 'static>(
        size: usize,
        factory: impl Fn(usize) -> Result<R> + Send + Sync + 'static,
        abort: Arc<dyn Fn() + Send + Sync>,
        timeout: Duration,
    ) -> Result<Self> {
        if size == 0 || size > i32::MAX as usize || timeout.is_zero() {
            return Err(Error::Invalid("invalid rank group size/timeout".into()));
        }
        let factory = Arc::new(factory);
        let (reply_tx, replies) = mpsc::sync_channel(size * 2);
        let mut group = Self {
            commands: vec![],
            replies,
            joins: vec![],
            abort,
            timeout,
            step: 0,
            failed: false,
            memory: vec![None; size],
        };
        for rank in 0..size {
            let (tx, rx) = mpsc::sync_channel(1);
            let reply = reply_tx.clone();
            let abort = group.abort.clone();
            let factory = factory.clone();
            let join = thread::Builder::new()
                .name(format!("rsglang-rank-{rank}"))
                .spawn(move || {
                    let mut current_step = 0;
                    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
                        let mut runner = factory(rank)?;
                        if reply
                            .send(Reply {
                                rank,
                                step: 0,
                                output: Ok(Output::Tokens(vec![])),
                                memory: runner.memory_bytes(),
                            })
                            .is_err()
                        {
                            return Ok(());
                        }
                        while let Ok(command) = rx.recv() {
                            match command {
                                Command::Step(step, batch, logits) => {
                                    current_step = step;
                                    let output = if logits {
                                        runner.logits_rows(&batch).map(Output::Logits)
                                    } else {
                                        runner.run(&batch).map(Output::Tokens)
                                    };
                                    let failed = output.is_err();
                                    if failed {
                                        abort();
                                    }
                                    if reply
                                        .send(Reply {
                                            rank,
                                            step,
                                            output,
                                            memory: runner.memory_bytes(),
                                        })
                                        .is_err()
                                        || failed
                                    {
                                        break;
                                    }
                                }
                                Command::Forget(id) => runner.forget_request(id),
                                Command::Stop => break,
                            }
                        }
                        Ok(())
                    }));
                    let error = match result {
                        Ok(Ok(())) => None,
                        Ok(Err(e)) => Some(e),
                        Err(_) => Some(Error::Backend(format!("TP rank {rank} panicked"))),
                    };
                    if let Some(error) = error {
                        abort();
                        let _ = reply.send(Reply {
                            rank,
                            step: current_step,
                            output: Err(error),
                            memory: None,
                        });
                    }
                })?;
            group.commands.push(tx);
            group.joins.push(join);
        }
        drop(reply_tx);
        // Checkpoint loading can be longer than an individual collective/step timeout.
        group.collect(0, timeout.max(Duration::from_secs(300)))?;
        Ok(group)
    }
    fn collect(&mut self, step: u64, timeout: Duration) -> Result<Output> {
        let deadline = Instant::now() + timeout;
        let mut seen = vec![false; self.commands.len()];
        let mut primary = None;
        for _ in 0..seen.len() {
            let result = (|| -> Result<Reply> {
                let reply = self
                    .replies
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                    .map_err(|e| {
                        Error::Backend(format!("TP step {step} failed waiting for ranks: {e}"))
                    })?;
                if reply.step != step || reply.rank >= seen.len() || seen[reply.rank] {
                    return Err(Error::Backend("TP rank replies are out of order".into()));
                }
                Ok(reply)
            })();
            let reply = match result {
                Ok(r) => r,
                Err(e) => {
                    self.failed = true;
                    (self.abort)();
                    return Err(e);
                }
            };
            seen[reply.rank] = true;
            self.memory[reply.rank] = reply.memory;
            match reply.output {
                Ok(out) if reply.rank == 0 => primary = Some(out),
                Ok(_) => {}
                Err(e) => {
                    self.failed = true;
                    (self.abort)();
                    return Err(Error::Backend(format!("TP rank {}: {e}", reply.rank)));
                }
            }
        }
        primary.ok_or_else(|| Error::Backend("missing primary TP reply".into()))
    }
    fn execute(&mut self, batch: &StepBatch, logits: bool) -> Result<Output> {
        if self.failed {
            return Err(Error::Stopped);
        }
        self.step += 1;
        let batch = Arc::new(batch.clone());
        for tx in &self.commands {
            if tx
                .send(Command::Step(self.step, batch.clone(), logits))
                .is_err()
            {
                self.failed = true;
                (self.abort)();
                return Err(Error::Stopped);
            }
        }
        self.collect(self.step, self.timeout)
    }
    pub fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        match self.execute(batch, true)? {
            Output::Logits(rows) => Ok(rows),
            _ => Err(Error::Backend("unexpected TP reply type".into())),
        }
    }
    pub fn memory_per_rank(&self) -> &[Option<u64>] {
        &self.memory
    }
}
impl ModelRunner for RankGroup {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>> {
        match self.execute(batch, false)? {
            Output::Tokens(tokens) => Ok(tokens),
            _ => Err(Error::Backend("unexpected TP reply type".into())),
        }
    }
    fn forget_request(&mut self, id: RequestId) {
        for tx in &self.commands {
            if tx.send(Command::Forget(id)).is_err() {
                self.failed = true;
                (self.abort)();
            }
        }
    }
    fn memory_bytes(&self) -> Option<u64> {
        self.memory
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .map(|v| v.into_iter().sum())
    }
}
impl Drop for RankGroup {
    fn drop(&mut self) {
        // Abort wakes a rank in a collective before asking threads to exit and reclaim buffers.
        (self.abort)();
        for tx in &self.commands {
            let _ = tx.send(Command::Stop);
        }
        for join in self.joins.drain(..) {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsglang_core::BatchPhase;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct Fake {
        rank: usize,
        calls: Arc<AtomicUsize>,
        fail: bool,
    }
    impl ModelRunner for Fake {
        fn run(&mut self, _: &StepBatch) -> Result<Vec<TokenOutput>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(Error::Backend("injected rank failure".into()));
            }
            Ok(vec![TokenOutput {
                request_id: 1,
                token_id: self.rank as u32,
            }])
        }
    }
    impl RankRunner for Fake {}
    #[test]
    fn all_ranks_complete_and_only_primary_returns_tokens() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let mut group = RankGroup::spawn(
            4,
            move |rank| {
                Ok(Fake {
                    rank,
                    calls: c.clone(),
                    fail: false,
                })
            },
            Arc::new(|| {}),
            Duration::from_secs(1),
        )
        .unwrap();
        let out = group
            .run(&StepBatch {
                phase: BatchPhase::Decode,
                sequences: vec![],
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(out[0].token_id, 0);
    }
    #[test]
    fn rank_failure_aborts_and_poisoned_group_rejects_more_work() {
        let aborted = Arc::new(AtomicBool::new(false));
        let flag = aborted.clone();
        let mut group = RankGroup::spawn(
            2,
            |rank| {
                Ok(Fake {
                    rank,
                    calls: Arc::new(AtomicUsize::new(0)),
                    fail: rank == 1,
                })
            },
            Arc::new(move || {
                flag.store(true, Ordering::SeqCst);
            }),
            Duration::from_secs(1),
        )
        .unwrap();
        let batch = StepBatch {
            phase: BatchPhase::Decode,
            sequences: vec![],
        };
        assert!(group.run(&batch).is_err());
        assert!(aborted.load(Ordering::SeqCst));
        assert!(matches!(group.run(&batch), Err(Error::Stopped)));
    }
}
