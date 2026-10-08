#![forbid(unsafe_code)]
mod parallel;
pub use parallel::{InferenceEngine, ParallelEngine};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rsglang_core::{Error, ModelRunner, RequestId, Result, SamplingParams, StepBatch, TokenOutput};
use rsglang_distributed::{RankRunner, TensorParallel};
use rsglang_kernels::{CudaBackend, KernelBackend};
use rsglang_models::{LayerKv, Model, ModelConfig};
use std::{collections::HashMap, path::Path};

pub struct Engine<B: KernelBackend> {
    backend: B,
    model: Model<B>,
    kv: Vec<LayerKv<B::Tensor>>,
    num_pages: usize,
    page_size: usize,
    max_seq_len: usize,
    rngs: HashMap<RequestId, ChaCha8Rng>,
}
impl Engine<CudaBackend> {
    pub fn load(
        path: &Path,
        device: usize,
        kv_bytes: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let config = ModelConfig::load(path)?;
        if max_seq_len == 0
            || max_seq_len > config.dimensions().max_position_embeddings
            || page_size == 0
        {
            return Err(Error::Invalid(
                "invalid configured context length/page size".into(),
            ));
        }
        Self::load_backend(
            path,
            config,
            CudaBackend::new(device)?,
            kv_bytes,
            page_size,
            max_seq_len,
        )
    }
    pub(crate) fn load_backend(
        path: &Path,
        config: ModelConfig,
        backend: CudaBackend,
        kv_bytes: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        config.validate_tp(backend.tensor_parallel())?;
        let num_pages =
            kv_bytes / config.kv_bytes_per_page_tp(page_size, backend.tensor_parallel())?;
        if num_pages == 0 || num_pages > u32::MAX as usize {
            return Err(Error::Invalid("invalid KV memory budget".into()));
        }
        let model = Model::load(&backend, path, config)?;
        let kv = model.allocate_kv(&backend, num_pages, page_size)?;
        backend.synchronize()?;
        Ok(Self {
            backend,
            model,
            kv,
            num_pages,
            page_size,
            max_seq_len,
            rngs: HashMap::new(),
        })
    }
}
impl<B: KernelBackend> Engine<B> {
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }
    pub fn forward_logits(&mut self, batch: &StepBatch) -> Result<B::Logits> {
        let meta = self.backend.metadata(
            batch,
            self.page_size,
            self.num_pages,
            self.config().dimensions().vocab_size,
            self.max_seq_len,
        )?;
        let hidden = self
            .model
            .forward_hidden(&self.backend, &meta, &mut self.kv)?;
        let logits = self.model.project(&self.backend, &hidden, &meta)?;
        self.backend.synchronize()?;
        Ok(logits)
    }
}
impl<B: KernelBackend> ModelRunner for Engine<B> {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>> {
        let result = (|| -> Result<Vec<TokenOutput>> {
            for sequence in &batch.sequences {
                if let Some(params) = &sequence.sampling {
                    params.validate()?;
                }
            }

            let meta = self.backend.metadata(
                batch,
                self.page_size,
                self.num_pages,
                self.config().dimensions().vocab_size,
                self.max_seq_len,
            )?;
            let hidden = self
                .model
                .forward_hidden(&self.backend, &meta, &mut self.kv)?;
            if batch.sequences.iter().all(|s| s.sampling.is_none()) {
                self.backend.synchronize()?;
                return Ok(vec![]);
            }
            let logits = self.model.project(&self.backend, &hidden, &meta)?;
            if self.backend.tensor_parallel().rank() != 0 {
                self.backend.synchronize()?;
                return Ok(vec![]);
            }
            let greedy = if batch
                .sequences
                .iter()
                .any(|s| s.sampling.as_ref().is_some_and(SamplingParams::is_greedy))
            {
                Some(self.backend.argmax(&logits)?)
            } else {
                None
            };
            let mut outputs = vec![];
            for (row, s) in batch.sequences.iter().enumerate() {
                if let Some(params) = &s.sampling {
                    params.validate()?;
                    let token_id = if params.is_greedy() {
                        greedy.as_ref().unwrap()[row]
                    } else {
                        let host = self.backend.download_logits_row(&logits, row)?;
                        sample(
                            &host,
                            params,
                            self.rngs
                                .entry(s.request_id)
                                .or_insert_with(|| ChaCha8Rng::seed_from_u64(params.seed)),
                        )?
                    };
                    outputs.push(TokenOutput {
                        request_id: s.request_id,
                        token_id,
                    });
                }
            }
            self.backend.synchronize()?;
            Ok(outputs)
        })();
        if result.is_err() {
            self.backend.abort();
            // Even failed steps establish a completion boundary before CPU ownership cleanup.
            let _ = self.backend.synchronize();
        }
        result
    }
    fn forget_request(&mut self, id: RequestId) {
        self.rngs.remove(&id);
    }
    fn memory_bytes(&self) -> Option<u64> {
        self.backend.memory_bytes()
    }
}
impl<B: KernelBackend> RankRunner for Engine<B> {
    fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        let result = (|| {
            let logits = self.forward_logits(batch)?;
            if self.backend.tensor_parallel().rank() != 0 {
                return Ok(vec![]);
            }
            (0..batch.sequences.len())
                .map(|row| self.backend.download_logits_row(&logits, row))
                .collect()
        })();
        if result.is_err() {
            self.backend.abort();
            let _ = self.backend.synchronize();
        }
        result
    }
}
fn sample(logits: &[f32], p: &SamplingParams, rng: &mut ChaCha8Rng) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::Backend("empty/nonfinite logits".into()));
    }
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x as f64 / p.temperature as f64))
        .collect();
    candidates.sort_unstable_by(|(ia, a), (ib, b)| b.total_cmp(a).then(ia.cmp(ib)));
    if let Some(k) = p.top_k {
        candidates.truncate(k.min(candidates.len()));
    }
    let max = candidates[0].1;
    let mut total = 0.0;
    for (_, x) in &mut candidates {
        *x = (*x - max).exp();
        total += *x;
    }
    let mut retained = 0.;
    let mut n = 0;
    for (_, x) in &candidates {
        retained += x;
        n += 1;
        if retained >= p.top_p as f64 * total {
            break;
        }
    }
    candidates.truncate(n);
    let mut target = rng.gen::<f64>() * retained;
    for (i, x) in &candidates {
        target -= x;
        if target < 0. {
            return Ok(*i as u32);
        }
    }
    Ok(candidates.last().unwrap().0 as u32)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nucleus_seed_and_top_k() {
        let p = SamplingParams {
            temperature: 1.,
            top_k: Some(2),
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        };
        let mut a = ChaCha8Rng::seed_from_u64(42);
        let mut b = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let x = sample(&[0., 2., 1.], &p, &mut a).unwrap();
            assert!(x == 1 || x == 2);
            assert_eq!(x, sample(&[0., 2., 1.], &p, &mut b).unwrap());
        }
        let p = SamplingParams { top_p: 0.01, ..p };
        assert_eq!(sample(&[0., 2., 1.], &p, &mut a).unwrap(), 1);
    }
}
