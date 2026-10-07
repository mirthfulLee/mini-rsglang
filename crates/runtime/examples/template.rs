//! Render the actual checkpoint template for HF byte/token equality checks.
#![forbid(unsafe_code)]
use rsglang_runtime::{ChatMessage, TextProcessor};
use std::path::Path;
fn main() -> rsglang_core::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let t = TextProcessor::load(Path::new(&a[1]))?;
    let messages: Vec<ChatMessage> = serde_json::from_slice(&std::fs::read(&a[2])?).unwrap();
    let text = t.chat(&messages, a.get(3).is_some_and(|s| s == "true"))?;
    println!(
        "{}",
        serde_json::json!({"text":text,"ids":t.encode(&text)?})
    );
    Ok(())
}
