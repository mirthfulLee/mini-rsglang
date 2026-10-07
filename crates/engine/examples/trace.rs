//! Export layer activations for a single token-ID prompt.
#![forbid(unsafe_code)]
use rsglang_core::{BatchPhase, Result, SequenceStep, StepBatch};
use rsglang_kernels::{CudaBackend, KernelBackend};
use rsglang_models::{Qwen3, Qwen3Config};
use std::{io::Write, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let path = Path::new(&a[1]);
    let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(&a[2])?).unwrap();
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    let b = CudaBackend::new(0)?;
    let cfg = Qwen3Config::load(path)?;
    let vocab = cfg.vocab_size;
    let model = Qwen3::load(&b, path, cfg)?;
    let pages = ids.len().div_ceil(16);
    let mut kv = model.allocate_kv(&b, pages, 16)?;
    let batch = StepBatch {
        phase: BatchPhase::Prefill,
        sequences: vec![SequenceStep {
            request_id: 1,
            input_ids: ids,
            start_pos: 0,
            pages: (0..pages as u32).collect(),
            sampling: None,
        }],
    };
    let meta = b.metadata(&batch, 16, pages, vocab, 4096)?;
    model.forward_hidden_observed(&b, &meta, &mut kv, &mut |label, t| {
        let values = b.download(t)?;
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(out.join(format!("{label}.bin")))?);
        for v in values {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    })?;
    Ok(())
}
