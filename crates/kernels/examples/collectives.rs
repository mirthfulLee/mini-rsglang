//! Real NCCL checks: odd vocabulary, batch>1 rank-major gather, and peer failure.
#![forbid(unsafe_code)]
use half::bf16;
use rsglang_core::*;
use rsglang_distributed::{RankGroup, RankRunner, TensorParallel};
use rsglang_kernels::{CudaBackend, KernelBackend, NcclTeam};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

struct Runner {
    backend: CudaBackend,
    fail: bool,
}
impl ModelRunner for Runner {
    fn run(&mut self, _: &StepBatch) -> Result<Vec<TokenOutput>> {
        let b = &self.backend;
        let tp = b.tensor_parallel();
        let x = b.upload(
            &vec![bf16::from_f32((tp.rank() + 1) as f32); 133],
            &[1, 133],
        )?;
        let reduced = b.all_reduce(x)?;
        if self.fail {
            b.synchronize()?;
            if tp.rank() == tp.size() - 1 {
                return Err(Error::Backend(
                    "injected peer failure before collective".into(),
                ));
            }
            // Peers may already be waiting in this next collective when the last rank
            // aborts. Results after an abort are undefined, so only exercise cleanup here.
            let _ = b.all_reduce(reduced)?;
            b.synchronize()?;
            return Ok(vec![]);
        }
        let expected = (tp.size() * (tp.size() + 1) / 2) as f32;
        let values = b.download(&reduced)?;
        assert!(
            values.iter().all(|&v| v == expected),
            "rank {} expected {expected}, actual {:?}",
            tp.rank(),
            &values[..10]
        );
        if tp.size() > 2 {
            let term = |rank| {
                bf16::from_f32(match rank {
                    0 => 1e8,
                    1 | 3 => 1.,
                    2 => -1e8,
                    _ => rank as f32,
                })
            };
            let expected =
                bf16::from_f32((0..tp.size()).fold(0f32, |sum, rank| sum + term(rank).to_f32()))
                    .to_f32();
            for count in [33, 133] {
                let x = b.upload(&vec![term(tp.rank()); count], &[count, 1])?;
                assert!(
                    b.download(&b.all_reduce(x)?)?
                        .iter()
                        .all(|&x| x == expected),
                    "rank-ordered cancellation sum"
                );
            }
        }
        let batch = StepBatch {
            phase: BatchPhase::Prefill,
            sequences: [0, 9, 16]
                .into_iter()
                .enumerate()
                .map(|(i, t)| SequenceStep {
                    request_id: i as u64,
                    input_ids: vec![t],
                    start_pos: 0,
                    pages: vec![i as u32],
                    sampling: None,
                })
                .collect(),
        };
        let meta = b.metadata(&batch, 16, 3, 17, 16)?;
        let shard = tp.vocab(17)?;
        let embedding: Vec<_> = (0..shard.padded_rows)
            .flat_map(|r| {
                (0..3).map(move |j| {
                    bf16::from_f32(if r + shard.start < 17 {
                        ((r + shard.start) * 10 + j) as f32
                    } else {
                        0.
                    })
                })
            })
            .collect();
        let embedding = b.upload(&embedding, &[shard.padded_rows, 3])?;
        let output = b.embedding_shard(&embedding, &meta, shard.start)?;
        assert_eq!(
            b.download(&output)?,
            [0., 1., 2., 90., 91., 92., 160., 161., 162.]
        );
        let head: Vec<_> = (0..shard.padded_rows)
            .flat_map(|r| {
                [
                    bf16::from_f32((shard.start + r) as f32),
                    bf16::from_f32(2.),
                    bf16::from_f32(3.),
                ]
            })
            .collect();
        let head = b.upload(&head, &[shard.padded_rows, 3])?;
        let x = b.upload(
            &[
                bf16::ONE,
                bf16::ZERO,
                bf16::ZERO,
                bf16::ZERO,
                bf16::ONE,
                bf16::ZERO,
                bf16::ZERO,
                bf16::ZERO,
                bf16::ONE,
            ],
            &[3, 3],
        )?;
        let logits = b.all_gather_logits(b.logits(&x, &head)?, 17)?;
        assert_eq!(
            b.download_logits_row(&logits, 0)?,
            (0..17).map(|v| v as f32).collect::<Vec<_>>()
        );
        assert_eq!(b.download_logits_row(&logits, 1)?, vec![2.; 17]);
        assert_eq!(b.download_logits_row(&logits, 2)?, vec![3.; 17]);
        assert_eq!(b.argmax(&logits)?, [16, 0, 0]);
        b.synchronize()?;
        Ok(vec![TokenOutput {
            request_id: 1,
            token_id: 16,
        }])
    }
}
impl RankRunner for Runner {}
fn group(devices: Vec<usize>, fail: bool) -> Result<RankGroup> {
    let size = devices.len();
    let team = NcclTeam::new_for_devices(&devices, Duration::from_secs(15))?;
    let abort_team = team.clone();
    let abort = Arc::new(move || abort_team.abort());
    RankGroup::spawn(
        size,
        move |rank| {
            Ok(Runner {
                backend: CudaBackend::new_rank(
                    devices[rank],
                    TensorParallel::new(rank, size)?,
                    &team,
                )?,
                fail,
            })
        },
        abort,
        Duration::from_secs(30),
    )
}
fn main() -> Result<()> {
    let devices = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0,1".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect::<Vec<usize>>();
    let batch = StepBatch {
        phase: BatchPhase::Decode,
        sequences: vec![],
    };
    let mut ranks = group(devices.clone(), false)?;
    assert_eq!(ranks.run(&batch)?[0].token_id, 16);
    drop(ranks);
    let started = Instant::now();
    let mut ranks = group(devices.clone(), true)?;
    assert!(ranks.run(&batch).is_err());
    drop(ranks);
    println!("NCCL {} ranks: sum, cancellation order, sharded embedding, padded batched logits, greedy and peer-failure cleanup passed; failure test {:.3}s",devices.len(),started.elapsed().as_secs_f64());
    Ok(())
}
