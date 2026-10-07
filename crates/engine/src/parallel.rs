use crate::*;
use rsglang_distributed::{validate_devices, RankGroup};
use rsglang_kernels::NcclTeam;
use std::{path::PathBuf, sync::Arc, time::Duration};

pub struct ParallelEngine {
    ranks: RankGroup,
    config: Qwen3Config,
    num_pages: usize,
    devices: Vec<usize>,
}
impl ParallelEngine {
    /// kv_bytes is a per-GPU budget; all ranks allocate the same logical page IDs.
    pub fn load(
        path: &Path,
        devices: &[usize],
        kv_bytes: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        validate_devices(devices)?;
        if devices.len() < 2 {
            return Err(Error::Invalid(
                "ParallelEngine needs at least two GPUs".into(),
            ));
        }
        let count = CudaBackend::device_count()?;
        if devices.iter().any(|&d| d >= count) {
            return Err(Error::Invalid(format!(
                "requested GPU outside available device count {count}"
            )));
        }
        let config = Qwen3Config::load(path)?;
        let tp = TensorParallel::new(0, devices.len())?;
        config.validate_tp(tp)?;
        if page_size == 0 || max_seq_len == 0 || max_seq_len > config.max_position_embeddings {
            return Err(Error::Invalid("invalid context length/page size".into()));
        }
        let num_pages = kv_bytes / config.kv_bytes_per_page_tp(page_size, tp)?;
        if num_pages == 0 || num_pages > u32::MAX as usize {
            return Err(Error::Invalid("invalid per-GPU KV budget".into()));
        }
        let team = NcclTeam::new_for_devices(devices, Duration::from_secs(120))?;
        tracing_rank_start(team.version(), devices);
        let rank_team = team.clone();
        let abort = Arc::new(move || team.abort());
        let path: PathBuf = path.into();
        let rank_devices = devices.to_vec();
        let rank_config = config.clone();
        let size = devices.len();
        let ranks = RankGroup::spawn(
            size,
            move |rank| {
                Engine::load_backend(
                    &path,
                    rank_config.clone(),
                    CudaBackend::new_rank(
                        rank_devices[rank],
                        TensorParallel::new(rank, size)?,
                        &rank_team,
                    )?,
                    kv_bytes,
                    page_size,
                    max_seq_len,
                )
            },
            abort,
            Duration::from_secs(300),
        )?;
        Ok(Self {
            ranks,
            config,
            num_pages,
            devices: devices.to_vec(),
        })
    }
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }
    pub fn devices(&self) -> &[usize] {
        &self.devices
    }
    pub fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        self.ranks.logits_rows(batch)
    }
    pub fn memory_per_rank(&self) -> &[Option<u64>] {
        self.ranks.memory_per_rank()
    }
}
fn tracing_rank_start(version: i32, devices: &[usize]) {
    eprintln!("tensor parallel devices={devices:?}, NCCL={version}");
}
impl ModelRunner for ParallelEngine {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>> {
        self.ranks.run(batch)
    }
    fn forget_request(&mut self, id: RequestId) {
        self.ranks.forget_request(id);
    }
    fn memory_bytes(&self) -> Option<u64> {
        self.ranks.memory_bytes()
    }
}

/// Scheduler-facing runner. CUDA state stays inside the owning rank worker(s).
pub enum InferenceEngine {
    Single(Box<Engine<CudaBackend>>),
    Parallel(Box<ParallelEngine>),
}
impl InferenceEngine {
    pub fn load(
        path: &Path,
        devices: &[usize],
        kv_bytes: usize,
        page_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        validate_devices(devices)?;
        if devices.len() == 1 {
            Ok(Self::Single(Box::new(Engine::load(
                path,
                devices[0],
                kv_bytes,
                page_size,
                max_seq_len,
            )?)))
        } else {
            Ok(Self::Parallel(Box::new(ParallelEngine::load(
                path,
                devices,
                kv_bytes,
                page_size,
                max_seq_len,
            )?)))
        }
    }
    pub fn config(&self) -> &Qwen3Config {
        match self {
            Self::Single(e) => e.config(),
            Self::Parallel(e) => e.config(),
        }
    }
    pub fn num_pages(&self) -> usize {
        match self {
            Self::Single(e) => e.num_pages(),
            Self::Parallel(e) => e.num_pages(),
        }
    }
    pub fn logits_rows(&mut self, batch: &StepBatch) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Single(e) => RankRunner::logits_rows(e.as_mut(), batch),
            Self::Parallel(e) => e.logits_rows(batch),
        }
    }
}
impl ModelRunner for InferenceEngine {
    fn run(&mut self, batch: &StepBatch) -> Result<Vec<TokenOutput>> {
        match self {
            Self::Single(e) => e.run(batch),
            Self::Parallel(e) => e.run(batch),
        }
    }
    fn forget_request(&mut self, id: RequestId) {
        match self {
            Self::Single(e) => e.forget_request(id),
            Self::Parallel(e) => e.forget_request(id),
        }
    }
    fn memory_bytes(&self) -> Option<u64> {
        match self {
            Self::Single(e) => e.memory_bytes(),
            Self::Parallel(e) => e.memory_bytes(),
        }
    }
}
