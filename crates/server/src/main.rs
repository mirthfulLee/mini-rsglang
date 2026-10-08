#![forbid(unsafe_code)]
mod api;
mod bench;
use clap::{Args, Parser, Subcommand, ValueEnum};
use rsglang_core::*;
use rsglang_runtime::{ChatMessage, EngineHandle};
use std::{io::Write, path::PathBuf};

#[derive(Parser)]
#[command(
    name = "mini-rsglang",
    version,
    about = "Rust LLM inference and serving with single-GPU or tensor parallel execution"
)]
struct Cli {
    #[arg(long, global = true, default_value = "/models/store/Qwen/Qwen3-0.6B")]
    model: PathBuf,
    #[arg(long, global = true, default_value_t = 0)]
    device: usize,
    #[arg(long, alias = "tp", global = true)]
    tensor_parallel_size: Option<usize>,
    #[arg(long, value_delimiter = ',', global = true, conflicts_with = "device")]
    devices: Option<Vec<usize>>,
    #[arg(long, global = true, default_value_t = 2048)]
    kv_mib: usize,
    #[arg(long, global = true, default_value_t = 4096)]
    max_seq_len: usize,
    #[arg(long, global = true, default_value_t = 16)]
    page_size: usize,
    #[arg(long, global = true, default_value_t = 512)]
    prefill_budget: usize,
    #[arg(long, global = true, default_value_t = 32)]
    max_running: usize,
    #[arg(long, global = true, default_value_t = 256)]
    max_waiting: usize,
    #[arg(long, global = true)]
    no_prefix_cache: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Args)]
struct Sampling {
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    ignore_eos: bool,
}
impl Sampling {
    fn params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
            ignore_eos: self.ignore_eos,
        }
    }
}
#[derive(Clone, Copy, ValueEnum)]
enum CacheMode {
    Cold,
    Hot,
    Both,
}
#[derive(Subcommand)]
enum Command {
    Generate {
        #[arg(
            long,
            required_unless_present = "token_ids",
            conflicts_with = "token_ids"
        )]
        prompt: Option<String>,
        #[arg(long, conflicts_with = "chat")]
        token_ids: Option<String>,
        #[arg(long)]
        chat: bool,
        #[arg(long, requires = "chat")]
        enable_thinking: bool,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        sampling: Sampling,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
    Bench {
        #[arg(long, default_value_t = 128)]
        input_tokens: usize,
        #[arg(long, default_value_t = 32)]
        output_tokens: usize,
        #[arg(long, default_value_t = 16)]
        requests: usize,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        #[arg(long,value_enum,default_value_t=CacheMode::Both)]
        cache_mode: CacheMode,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
impl Cli {
    fn config(&self) -> RuntimeConfig {
        RuntimeConfig {
            max_seq_len: self.max_seq_len,
            page_size: self.page_size,
            prefill_budget: self.prefill_budget,
            max_running: self.max_running,
            max_waiting: self.max_waiting,
            prefix_cache: !self.no_prefix_cache,
        }
    }
    fn load(&self, config: RuntimeConfig) -> Result<EngineHandle> {
        let devices = self.device_ids()?;
        EngineHandle::load_tensor_parallel(
            &self.model,
            &devices,
            self.kv_mib
                .checked_mul(1024 * 1024)
                .ok_or_else(|| Error::Invalid("kv_mib overflow".into()))?,
            config,
        )
    }
    fn device_ids(&self) -> Result<Vec<usize>> {
        let size = self
            .tensor_parallel_size
            .unwrap_or_else(|| self.devices.as_ref().map_or(1, Vec::len));
        if size == 0 {
            return Err(Error::Invalid(
                "tensor_parallel_size must be positive".into(),
            ));
        }
        if let Some(devices) = &self.devices {
            if devices.len() != size {
                return Err(Error::Invalid(
                    "--devices count must equal tensor-parallel size".into(),
                ));
            }
            Ok(devices.clone())
        } else {
            let end = self
                .device
                .checked_add(size)
                .ok_or_else(|| Error::Invalid("device range overflow".into()))?;
            Ok((self.device..end).collect())
        }
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match &cli.command {
        Command::Generate {
            prompt,
            token_ids,
            chat,
            enable_thinking,
            json,
            sampling,
        } => {
            let engine = cli.load(cli.config())?;
            let prompt = if let Some(ids) = token_ids {
                Prompt::TokenIds(
                    ids.split(',')
                        .map(|v| {
                            v.trim().parse::<u32>().map_err(|_| {
                                Error::Invalid("token_ids must be comma-separated integers".into())
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                let text = prompt.as_ref().unwrap();
                Prompt::Text(if *chat {
                    engine.text().chat(
                        &[ChatMessage {
                            role: "user".into(),
                            content: text.clone(),
                            reasoning_content: None,
                        }],
                        *enable_thinking,
                    )?
                } else {
                    text.clone()
                })
            };
            let mut events = engine
                .generate(GenerateRequest {
                    prompt,
                    sampling: sampling.params(),
                })
                .await?;
            let mut completed = false;
            while let Some(event) = events.recv().await {
                if *json {
                    println!("{}", serde_json::to_string(&event).unwrap());
                }
                match event {
                    GenerationEvent::Token { text, .. } | GenerationEvent::Text { text, .. }
                        if !json =>
                    {
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    GenerationEvent::Error { message, .. } => return Err(Error::Backend(message)),
                    GenerationEvent::Finished {
                        reason,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        ..
                    } => {
                        if !json {
                            println!();
                            eprintln!("{reason:?}: prompt={prompt_tokens}, generated={completion_tokens}, cached={cached_tokens}");
                        }
                        completed = matches!(reason, FinishReason::Stop | FinishReason::Length);
                        break;
                    }
                    _ => {}
                }
            }
            engine.shutdown().await?;
            if !completed {
                return Err(Error::Backend("generation did not complete".into()));
            }
        }
        Command::Serve { host, port } => {
            let engine = cli.load(cli.config())?;
            let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
            tracing::info!(model=%engine.info().model_id,pages=engine.info().num_pages,"serving on http://{host}:{port}");
            axum::serve(listener, api::router(engine.clone()))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
            engine.shutdown().await?;
        }
        Command::Bench {
            input_tokens,
            output_tokens,
            requests,
            concurrency,
            cache_mode,
            output,
        } => {
            let modes: &[bool] = match cache_mode {
                CacheMode::Cold => &[false],
                CacheMode::Hot => &[true],
                CacheMode::Both => &[false, true],
            };
            let mut runs = vec![];
            for &warm in modes {
                let mut config = cli.config();
                config.prefix_cache = warm;
                let engine = cli.load(config)?;
                let result = bench::run(
                    &engine,
                    *input_tokens,
                    *output_tokens,
                    *requests,
                    *concurrency,
                    warm,
                )
                .await;
                engine.shutdown().await?;
                runs.push(result?);
            }
            let report=serde_json::to_string_pretty(&serde_json::json!({"backend":"cuda12-cublas-nvrtc-nccl","devices":cli.device_ids()?,"runs":runs})).unwrap();
            if let Some(path) = output {
                std::fs::write(path, &report)?;
            }
            println!("{report}");
        }
    }
    Ok(())
}
