//! Run the public Rust API without the CLI or HTTP server.
#![forbid(unsafe_code)]
use rsglang_core::{
    GenerateRequest, GenerationEvent, Prompt, Result, RuntimeConfig, SamplingParams,
};
use rsglang_runtime::{ChatMessage, EngineHandle};
use std::{io::Write, path::PathBuf};

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "/models/store/Qwen/Qwen3-0.6B".into());
    let engine = EngineHandle::load(&model, 0, 2 * 1024 * 1024 * 1024, RuntimeConfig::default())?;
    let prompt = engine.text().chat(
        &[ChatMessage {
            role: "user".into(),
            content: "Write one sentence about Rust ownership.".into(),
            reasoning_content: None,
        }],
        false,
    )?;
    let mut events = engine
        .generate(GenerateRequest {
            prompt: Prompt::Text(prompt),
            sampling: SamplingParams {
                max_tokens: 32,
                ..Default::default()
            },
        })
        .await?;
    while let Some(event) = events.recv().await {
        match event {
            GenerationEvent::Token { text, .. } | GenerationEvent::Text { text, .. } => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            GenerationEvent::Finished { reason, .. } => {
                eprintln!("\n{reason:?}");
                break;
            }
            GenerationEvent::Error { message, .. } => {
                engine.shutdown().await?;
                return Err(rsglang_core::Error::Backend(message));
            }
        }
    }
    engine.shutdown().await
}
