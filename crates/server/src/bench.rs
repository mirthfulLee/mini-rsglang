use rsglang_core::*;
use rsglang_runtime::EngineHandle;
use serde_json::json;
use std::time::Instant;
use tokio::task::JoinSet;

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    x[((x.len() - 1) as f64 * p).round() as usize]
}
pub async fn run(
    engine: &EngineHandle,
    input_tokens: usize,
    output_tokens: usize,
    requests: usize,
    concurrency: usize,
    warm: bool,
) -> Result<serde_json::Value> {
    if [input_tokens, output_tokens, requests, concurrency].contains(&0)
        || concurrency > engine.config().max_running
    {
        return Err(Error::Invalid(
            "positive benchmark sizes required; concurrency must be <= max_running".into(),
        ));
    }
    let base = engine
        .text()
        .encode("The quick brown fox jumps over the lazy dog. ")?;
    let ids: Vec<u32> = base.iter().copied().cycle().take(input_tokens).collect();
    let make = |max_tokens| GenerateRequest {
        prompt: Prompt::TokenIds(ids.clone()),
        sampling: SamplingParams {
            max_tokens,
            ignore_eos: true,
            ..Default::default()
        },
    };
    if warm {
        let mut events = engine.generate(make(1)).await?;
        while let Some(event) = events.recv().await {
            if let GenerationEvent::Error { message, .. } = event {
                return Err(Error::Backend(message));
            }
        }
    }
    // Warm CUDA/cuBLAS at the measured batch and token shapes in both cache modes.
    // Cold runs have prefix caching disabled by the caller, so this cannot create cache hits.
    for _ in 0..2 {
        let mut tasks = JoinSet::new();
        for _ in 0..concurrency {
            let handle = engine.clone();
            let request = make(output_tokens);
            tasks.spawn(async move {
                let mut events = handle.generate(request).await?;
                while let Some(event) = events.recv().await {
                    match event {
                        GenerationEvent::Error { message, .. } => {
                            return Err(Error::Backend(message))
                        }
                        GenerationEvent::Finished {
                            reason: FinishReason::Length,
                            completion_tokens,
                            ..
                        } if completion_tokens == output_tokens => return Ok(()),
                        GenerationEvent::Finished { .. } => {
                            return Err(Error::Backend("benchmark warmup ended early".into()))
                        }
                        _ => {}
                    }
                }
                Err(Error::Backend("benchmark warmup stream closed".into()))
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.map_err(|e| Error::Backend(e.to_string()))??;
        }
    }
    let before = engine.metrics();
    let started = Instant::now();
    let mut set = JoinSet::new();
    let mut submitted = 0usize;
    let mut ttft = vec![];
    let mut itl = vec![];
    let mut latency = vec![];
    while submitted < requests || !set.is_empty() {
        while submitted < requests && set.len() < concurrency {
            let handle = engine.clone();
            let request = make(output_tokens);
            set.spawn(async move {
                let start = Instant::now();
                let mut events = handle.generate(request).await?;
                let mut first = None;
                let mut last = None;
                let mut intervals = vec![];
                let mut finished = false;
                while let Some(event) = events.recv().await {
                    match event {
                        GenerationEvent::Token { .. } => {
                            let now = start.elapsed().as_secs_f64();
                            if let Some(prev) = last {
                                intervals.push(now - prev);
                            } else {
                                first = Some(now);
                            }
                            last = Some(now);
                        }
                        GenerationEvent::Error { message, .. } => {
                            return Err(Error::Backend(message))
                        }
                        GenerationEvent::Finished { reason, .. } => {
                            if reason != FinishReason::Length {
                                return Err(Error::Backend(
                                    "benchmark did not generate fixed output length".into(),
                                ));
                            }
                            finished = true;
                            break;
                        }
                        _ => {}
                    }
                }
                if !finished {
                    return Err(Error::Backend("benchmark stream closed".into()));
                }
                Ok::<_, Error>((
                    first.unwrap_or(0.),
                    intervals,
                    start.elapsed().as_secs_f64(),
                ))
            });
            submitted += 1;
        }
        let (first, intervals, total) = set
            .join_next()
            .await
            .unwrap()
            .map_err(|e| Error::Backend(e.to_string()))??;
        ttft.push(first);
        itl.extend(intervals);
        latency.push(total);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let after = engine.metrics();
    Ok(
        json!({"warmup_waves":2,"cache_mode":if warm{"warm_prefix"}else{"disabled"},"model":engine.info().model_id,"input_tokens":input_tokens,"output_tokens":output_tokens,"requests":requests,"concurrency":concurrency,"elapsed_s":elapsed,"output_tokens_per_s":(requests*output_tokens) as f64/elapsed,"requests_per_s":requests as f64/elapsed,"ttft_ms":{"p50":percentile(&ttft,0.5)*1000.,"p95":percentile(&ttft,0.95)*1000.},"itl_ms":{"p50":percentile(&itl,0.5)*1000.,"p95":percentile(&itl,0.95)*1000.},"latency_ms":{"p50":percentile(&latency,0.5)*1000.,"p95":percentile(&latency,0.95)*1000.},"raw_ttft_s":ttft,"raw_itl_s":itl,"raw_latency_s":latency,"cached_tokens":after.cached_tokens-before.cached_tokens,"peak_memory_bytes":after.peak_memory_bytes,"metrics":after}),
    )
}
