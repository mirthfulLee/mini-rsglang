//! Teacher-forced batched logits export for independent HF comparisons.
#![forbid(unsafe_code)]
use rsglang_core::{BatchPhase, Result, SequenceStep, StepBatch};
use rsglang_engine::InferenceEngine;
use serde::Deserialize;
use std::{io::Write, path::PathBuf};
#[derive(Deserialize)]
struct Case {
    prompt: Vec<u32>,
    continuation: Vec<u32>,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 4,
        "forward MODEL INPUT.json OUTPUT_DIR [CHUNK_SIZE] [DEVICE_IDS]"
    );
    let model = PathBuf::from(&args[1]);
    let input = std::fs::read(&args[2])?;
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out)?;
    let cases: Vec<Case> = serde_json::from_slice(&input).expect("input cases");
    let chunk = args
        .get(4)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(usize::MAX);
    assert!(chunk > 0);
    let devices = args
        .get(5)
        .map_or("0", String::as_str)
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let config = rsglang_models::Qwen3Config::load(&model)?;
    let mut engine = InferenceEngine::load(
        &model,
        &devices,
        256 * 1024 * 1024,
        16,
        config.max_position_embeddings.min(4096),
    )?;
    let mut pages = vec![];
    let mut offset = 0u32;
    for case in &cases {
        let n = (case.prompt.len() + case.continuation.len()).div_ceil(16) as u32;
        pages.push((offset..offset + n).collect::<Vec<_>>());
        offset += n;
    }
    assert!(offset as usize <= engine.num_pages());
    let mut computed = vec![0usize; cases.len()];
    loop {
        let mut indices = vec![];
        let mut sequences = vec![];
        for (i, c) in cases.iter().enumerate() {
            let start = computed[i];
            if start == c.prompt.len() {
                continue;
            }
            let end = start.saturating_add(chunk).min(c.prompt.len());
            computed[i] = end;
            indices.push(i);
            sequences.push(SequenceStep {
                request_id: i as u64,
                input_ids: c.prompt[start..end].to_vec(),
                start_pos: start,
                pages: pages[i].clone(),
                sampling: None,
            });
        }
        if sequences.is_empty() {
            break;
        }
        let logits = engine.logits_rows(&StepBatch {
            phase: BatchPhase::Prefill,
            sequences,
        })?;
        for (row, i) in indices.into_iter().enumerate() {
            if computed[i] == cases[i].prompt.len() {
                save(&out.join(format!("case{i}-step0.bin")), &logits[row])?;
            }
        }
    }
    let steps = cases
        .iter()
        .map(|c| c.continuation.len())
        .max()
        .unwrap_or(0);
    for step in 0..steps {
        let mut indices = vec![];
        let mut sequences = vec![];
        for (i, c) in cases.iter().enumerate() {
            if let Some(&id) = c.continuation.get(step) {
                indices.push(i);
                sequences.push(SequenceStep {
                    request_id: i as u64,
                    input_ids: vec![id],
                    start_pos: c.prompt.len() + step,
                    pages: pages[i].clone(),
                    sampling: None,
                });
            }
        }
        let logits = engine.logits_rows(&StepBatch {
            phase: BatchPhase::Decode,
            sequences,
        })?;
        for (row, i) in indices.into_iter().enumerate() {
            save(
                &out.join(format!("case{i}-step{}.bin", step + 1)),
                &logits[row],
            )?;
        }
    }
    println!(
        "exported {} batched sequences, prefill plus {} decode steps",
        cases.len(),
        steps
    );
    Ok(())
}
fn save(path: &std::path::Path, values: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
