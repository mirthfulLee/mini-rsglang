//! Checked CUDA operations. Raw pointers and unsafe launches stay inside this crate.
mod collective;
mod interface;
mod nccl_api;
mod parallel_ops;
pub use collective::NcclTeam;
use cudarc::{
    cublas::{self, sys as blas_sys, CudaBlas},
    driver::{
        CudaContext, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
        PushKernelArg,
    },
    nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx},
};
use half::bf16;
pub use interface::{ExpertAssignment, KernelBackend};
use rsglang_core::{Error, Result, StepBatch};
use rsglang_distributed::TensorParallel;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};
static PTX_CACHE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
fn invalid(s: &str) -> Error {
    Error::Invalid(s.into())
}
fn int(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| invalid("kernel dimension exceeds i32"))
}
fn elems(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid("tensor dimensions must be positive"));
    }
    let n = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid("tensor size overflow"))?;
    int(n)?;
    Ok(n)
}

pub struct Tensor {
    data: CudaSlice<bf16>,
    shape: Vec<usize>,
}
pub struct Logits {
    data: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}
pub struct Accumulator {
    data: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}
impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
/// Constructed only after validating every token, page, and output write slot.
pub struct Metadata {
    ids: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    sequences: CudaSlice<u32>,
    slots: CudaSlice<u32>,
    table: CudaSlice<u32>,
    last_rows: CudaSlice<u32>,
    tokens: usize,
    vocab: usize,
    batch: usize,
    table_width: usize,
    page_size: usize,
    pages: usize,
}

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    module: Arc<CudaModule>,
    rope_tables: Mutex<HashMap<(usize, u32), CudaSlice<f32>>>,
    tp: TensorParallel,
    communicator: Option<collective::Communicator>,
}
impl CudaBackend {
    pub fn new(device: usize) -> Result<Self> {
        let ctx = CudaContext::new(device).map_err(backend)?;
        // An explicit compute stream avoids legacy-default-stream barriers between
        // NCCL's internal streams and rank-local allocation/kernel launches.
        let stream = ctx.new_stream().map_err(backend)?;
        let major=ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).map_err(backend)?;
        let minor=ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).map_err(backend)?;
        if major < 8 {
            return Err(invalid("BF16 backend requires sm_80 or newer"));
        }
        let arch = match (major, minor) {
            (8, 0) => "compute_80",
            (8, 6) => "compute_86",
            (8, 9) => "compute_89",
            (9, 0) => "compute_90",
            _ => return Err(invalid("architecture unsupported by CUDA 12.6 backend")),
        };
        let toolkit = std::env::var("CUDA_TOOLKIT_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        let mut nvrtc_major = 0i32;
        let mut nvrtc_minor = 0i32;
        // SAFETY: nvrtcVersion writes two valid, distinct host i32 pointers synchronously.
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) }
            .result()
            .map_err(backend)?;
        let nvrtc_version = (nvrtc_major, nvrtc_minor);
        let opts = CompileOptions {
            arch: Some(arch),
            include_paths: vec![format!("{toolkit}/include")],
            options: vec!["--fmad=false".into(), "--std=c++17".into()],
            ..Default::default()
        };
        let source = include_str!("ops.cu");
        let key = format!(
            "{:x}",
            Sha256::digest(
                format!("{source}\n{opts:?}\n{major}.{minor}\n{nvrtc_version:?}").as_bytes()
            )
        );
        let cache = std::env::var_os("RSGLANG_KERNEL_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache")
                    })
                    .join("mini-rsglang/kernels")
            });
        std::fs::create_dir_all(&cache)?;
        let path = cache.join(format!("{key}.ptx"));
        let ptx = if path.exists() {
            Ptx::from_src(std::fs::read_to_string(&path)?)
        } else {
            let ptx = compile_ptx_with_opts(source, opts).map_err(backend)?;
            let serial = PTX_CACHE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = cache.join(format!("{key}.{}.{serial}.tmp", std::process::id()));
            std::fs::write(&tmp, ptx.to_src())?;
            std::fs::rename(tmp, path)?;
            ptx
        };
        let module = ctx.load_module(ptx).map_err(backend)?;
        let blas = CudaBlas::new(stream.clone()).map_err(backend)?;
        // SAFETY: this is a newly created valid handle, not yet shared or executing.
        // Keep split-K reductions in FP32 rather than permitting BF16 partial sums.
        unsafe {
            blas_sys::cublasSetMathMode(
                *blas.handle(),
                blas_sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
        }
        .result()
        .map_err(backend)?;
        Ok(Self {
            ctx,
            stream,
            blas,
            module,
            rope_tables: Mutex::new(HashMap::new()),
            tp: TensorParallel::default(),
            communicator: None,
        })
    }
    pub fn new_rank(device: usize, tp: TensorParallel, team: &Arc<NcclTeam>) -> Result<Self> {
        let mut backend = Self::new(device)?;
        backend.communicator = Some(team.connect(backend.stream.clone(), tp)?);
        backend.tp = tp;
        Ok(backend)
    }
    pub fn device_count() -> Result<usize> {
        CudaContext::device_count()
            .map(|n| n as usize)
            .map_err(backend)
    }
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.ctx.mem_get_info().map_err(backend)
    }
    pub fn download(&self, x: &Tensor) -> Result<Vec<f32>> {
        self.check(x)?;
        Ok(self
            .stream
            .clone_dtoh(&x.data)
            .map_err(backend)?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }
    fn check(&self, t: &Tensor) -> Result<()> {
        if t.data.context().as_ref() != self.ctx.as_ref() {
            return Err(invalid("tensor belongs to another CUDA context"));
        }
        Ok(())
    }
    fn check_meta(&self, m: &Metadata) -> Result<()> {
        if m.ids.context().as_ref() != self.ctx.as_ref() {
            return Err(invalid("metadata belongs to another CUDA context"));
        }
        Ok(())
    }
    fn function(&self, name: &str) -> Result<cudarc::driver::CudaFunction> {
        self.module.load_function(name).map_err(backend)
    }
    fn flat(n: usize) -> Result<LaunchConfig> {
        Ok(LaunchConfig::for_num_elems(
            u32::try_from(n).map_err(|_| invalid("launch size overflow"))?,
        ))
    }
    fn rows(n: usize, threads: u32) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid_dim: (
                u32::try_from(n).map_err(|_| invalid("grid size overflow"))?,
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    fn linear_shape(&self, x: &Tensor, w: &Tensor) -> Result<(usize, usize, usize)> {
        self.check(x)?;
        self.check(w)?;
        if x.shape.len() != 2 || w.shape.len() != 2 || x.shape[1] != w.shape[1] {
            return Err(invalid("linear requires x[M,K], weights[N,K]"));
        }
        Ok((x.shape[0], w.shape[0], x.shape[1]))
    }
    // Keep cuBLAS's token-row shape fixed so admission, chunking and batching do not
    // change its reduction algorithm. Each block is independent; padded outputs are discarded.
    fn gemm_to<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        x: &Tensor,
        w: &Tensor,
        out: &mut CudaSlice<T>,
        kind: blas_sys::cudaDataType_t,
    ) -> Result<()> {
        const ROWS: usize = 32;
        let (m, n, k) = self.linear_shape(x, w)?;
        if out.context().as_ref() != self.ctx.as_ref() || out.len() != m * n {
            return Err(invalid("GEMM output shape/context mismatch"));
        }
        let mut input = self.zeros(&[ROWS, k])?;
        let mut output = self
            .stream
            .alloc_zeros::<T>(elems(&[ROWS, n])?)
            .map_err(backend)?;
        for start in (0..m).step_by(ROWS) {
            let end = (start + ROWS).min(m);
            let used = end - start;
            self.stream
                .memcpy_dtod(
                    &x.data.slice(start * k..end * k),
                    &mut input.data.slice_mut(..used * k),
                )
                .map_err(backend)?;
            self.gemm_block(&input, w, &mut output, kind)?;
            self.stream
                .memcpy_dtod(
                    &output.slice(..used * n),
                    &mut out.slice_mut(start * n..end * n),
                )
                .map_err(backend)?;
        }
        Ok(())
    }
    fn gemm_block<T: cudarc::driver::DeviceRepr>(
        &self,
        x: &Tensor,
        w: &Tensor,
        out: &mut CudaSlice<T>,
        kind: blas_sys::cudaDataType_t,
    ) -> Result<()> {
        let (m, n, k) = self.linear_shape(x, w)?;
        if out.len() != m * n {
            return Err(invalid("GEMM output shape mismatch"));
        }
        let (wp, _wg) = w.data.device_ptr(&self.stream);
        let (xp, _xg) = x.data.device_ptr(&self.stream);
        let (op, _og) = out.device_ptr_mut(&self.stream);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: validated row-major X[M,K] W[N,K]. In column-major,
        // C[N,M]=transpose(W[K,N])*X[K,M], with lda=ldb=K and ldc=N.
        // Guards keep all buffers alive/ordered on the same stream; C is exclusively borrowed.
        unsafe {
            cublas::result::gemm_ex(
                *self.blas.handle(),
                blas_sys::cublasOperation_t::CUBLAS_OP_T,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                int(n)?,
                int(m)?,
                int(k)?,
                (&alpha as *const f32).cast(),
                wp as *const _,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                int(k)?,
                xp as *const _,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                int(k)?,
                (&beta as *const f32).cast(),
                op as *mut _,
                kind,
                int(n)?,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(backend)
    }
}
impl KernelBackend for CudaBackend {
    type Tensor = Tensor;
    type Logits = Logits;
    type Metadata = Metadata;
    type Accumulator = Accumulator;
    fn token_count(&self, meta: &Metadata) -> usize {
        meta.tokens
    }
    fn tensor_parallel(&self) -> TensorParallel {
        self.tp
    }
    fn concat_rows(&self, tensors: &[Tensor]) -> Result<Tensor> {
        self.concat_checked(tensors)
    }
    fn linear_reduce(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        if self.communicator.is_none() {
            return self.linear(x, w);
        }
        // Preserve FP32 partial GEMMs until the global row-parallel sum completes.
        let mut partial = self.logits(x, w)?;
        self.reduce_f32(&mut partial.data)?;
        self.cast_bf16(&partial.data, &[partial.rows, partial.cols])
    }
    fn split_columns(&self, x: &Tensor, widths: &[usize]) -> Result<Vec<Tensor>> {
        self.split_checked(x, widths)
    }
    fn abort(&self) {
        if let Some(comm) = &self.communicator {
            comm.abort();
        }
    }
    fn embedding_shard(&self, w: &Tensor, m: &Metadata, start: usize) -> Result<Tensor> {
        self.embed_shard_checked(w, m, start)
    }
    fn all_reduce(&self, x: Tensor) -> Result<Tensor> {
        self.reduce_checked(x)
    }
    fn all_gather_logits(&self, x: Logits, vocab: usize) -> Result<Logits> {
        self.gather_logits_checked(x, vocab)
    }
    fn router_topk(&self, x: &Tensor, k: usize, norm: bool) -> Result<Vec<ExpertAssignment>> {
        self.route_checked(x, k, norm)
    }
    fn gather_rows(&self, x: &Tensor, rows: &[u32]) -> Result<Tensor> {
        self.gather_rows_checked(x, rows)
    }
    fn accumulator(&self, rows: usize, cols: usize) -> Result<Accumulator> {
        Ok(Accumulator {
            data: self
                .stream
                .alloc_zeros(elems(&[rows, cols])?)
                .map_err(backend)?,
            rows,
            cols,
        })
    }
    fn scatter_weighted(
        &self,
        out: &mut Accumulator,
        x: &Tensor,
        a: &ExpertAssignment,
    ) -> Result<()> {
        self.scatter_checked(out, x, a)
    }
    fn finish_accumulator(&self, mut out: Accumulator) -> Result<Tensor> {
        self.reduce_f32(&mut out.data)?;
        self.cast_bf16(&out.data, &[out.rows, out.cols])
    }
    fn upload(&self, values: &[bf16], shape: &[usize]) -> Result<Tensor> {
        if elems(shape)? != values.len() {
            return Err(invalid("upload shape mismatch"));
        }
        Ok(Tensor {
            data: self.stream.clone_htod(values).map_err(backend)?,
            shape: shape.to_vec(),
        })
    }
    fn zeros(&self, shape: &[usize]) -> Result<Tensor> {
        Ok(Tensor {
            data: self.stream.alloc_zeros(elems(shape)?).map_err(backend)?,
            shape: shape.to_vec(),
        })
    }
    fn metadata(
        &self,
        batch: &StepBatch,
        page_size: usize,
        pages: usize,
        vocab: usize,
        max_seq_len: usize,
    ) -> Result<Metadata> {
        if batch.sequences.is_empty() || page_size == 0 || pages == 0 {
            return Err(invalid("empty batch or KV pool"));
        }
        let table_width = batch.sequences.iter().map(|s| s.pages.len()).max().unwrap();
        let mut ids = vec![];
        let mut positions = vec![];
        let mut seqs = vec![];
        let mut slots = vec![];
        let mut table = vec![0u32; table_width * batch.sequences.len()];
        let mut last = vec![];
        let mut writes = HashSet::new();
        for (i, s) in batch.sequences.iter().enumerate() {
            if s.input_ids.is_empty()
                || s.start_pos
                    .checked_add(s.input_ids.len())
                    .is_none_or(|n| n > max_seq_len)
                || s.pages.len() < (s.start_pos + s.input_ids.len()).div_ceil(page_size)
            {
                return Err(invalid("batch sequence length/page coverage mismatch"));
            }
            if s.pages.iter().any(|&p| p as usize >= pages)
                || s.input_ids.iter().any(|&t| t as usize >= vocab)
            {
                return Err(invalid("token or page ID out of range"));
            }
            table[i * table_width..i * table_width + s.pages.len()].copy_from_slice(&s.pages);
            for (j, &id) in s.input_ids.iter().enumerate() {
                let pos = s.start_pos + j;
                let slot = (s.pages[pos / page_size] as usize)
                    .checked_mul(page_size)
                    .and_then(|n| n.checked_add(pos % page_size))
                    .ok_or_else(|| invalid("slot overflow"))?;
                if !writes.insert(slot) {
                    return Err(invalid("aliased KV write slots in batch"));
                }
                ids.push(id);
                positions.push(u32::try_from(pos).map_err(|_| invalid("position overflow"))?);
                seqs.push(i as u32);
                slots.push(u32::try_from(slot).map_err(|_| invalid("slot overflow"))?);
            }
            last.push((ids.len() - 1) as u32);
        }
        let tokens = ids.len();
        int(tokens)?;
        int(table.len())?;
        Ok(Metadata {
            ids: self.stream.clone_htod(&ids).map_err(backend)?,
            positions: self.stream.clone_htod(&positions).map_err(backend)?,
            sequences: self.stream.clone_htod(&seqs).map_err(backend)?,
            slots: self.stream.clone_htod(&slots).map_err(backend)?,
            table: self.stream.clone_htod(&table).map_err(backend)?,
            last_rows: self.stream.clone_htod(&last).map_err(backend)?,
            tokens,
            vocab,
            batch: batch.sequences.len(),
            table_width,
            page_size,
            pages,
        })
    }
    fn embedding(&self, w: &Tensor, m: &Metadata) -> Result<Tensor> {
        self.check(w)?;
        self.check_meta(m)?;
        if w.shape.len() != 2 || w.shape[0] != m.vocab {
            return Err(invalid("embedding weights must be a matrix"));
        }
        let d = int(w.shape[1])?;
        let n = int(m.tokens)?;
        let mut out = self.zeros(&[m.tokens, w.shape[1]])?;
        let fun = self.function("embedding")?;
        // SAFETY: metadata validated token IDs; each thread owns one checked output element.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&w.data)
                .arg(&m.ids)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&d)
                .launch(Self::flat(m.tokens * w.shape[1])?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn rms_norm(&self, x: &Tensor, w: &Tensor, width: usize, eps: f32) -> Result<Tensor> {
        self.check(x)?;
        self.check(w)?;
        if width == 0
            || !x.data.len().is_multiple_of(width)
            || w.data.len() != width
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(invalid("RMSNorm shape/epsilon mismatch"));
        }
        let mut out = self.zeros(&x.shape)?;
        let d = int(width)?;
        let fun = self.function("rms")?;
        // SAFETY: each CTA owns a row; each full warp initializes its shared reduction slot.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&w.data)
                .arg(&mut out.data)
                .arg(&d)
                .arg(&eps)
                .launch(Self::rows(x.data.len() / width, 256)?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn linear(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (m, n, _) = self.linear_shape(x, w)?;
        let mut out = self.zeros(&[m, n])?;
        self.gemm_to(x, w, &mut out.data, blas_sys::cudaDataType_t::CUDA_R_16BF)?;
        Ok(out)
    }
    fn add(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        self.check(x)?;
        self.check(y)?;
        if x.shape != y.shape {
            return Err(invalid("add shape mismatch"));
        }
        let mut out = self.zeros(&x.shape)?;
        let n = int(x.data.len())?;
        let fun = self.function("add")?;
        // SAFETY: equal buffer lengths; per-element bounds check and disjoint output.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&y.data)
                .arg(&mut out.data)
                .arg(&n)
                .launch(Self::flat(x.data.len())?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn rope(
        &self,
        x: &mut Tensor,
        m: &Metadata,
        heads: usize,
        dim: usize,
        theta: f32,
    ) -> Result<()> {
        self.check(x)?;
        self.check_meta(m)?;
        if dim == 0
            || !dim.is_multiple_of(2)
            || x.shape != [m.tokens, heads * dim]
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(invalid("RoPE shape/theta mismatch"));
        }
        let n = int(m.tokens)?;
        let h = int(heads)?;
        let d = int(dim)?;
        let fun = self.function("rope")?;
        let mut tables = self
            .rope_tables
            .lock()
            .map_err(|_| backend("RoPE cache poisoned"))?;
        if let std::collections::hash_map::Entry::Vacant(entry) =
            tables.entry((dim, theta.to_bits()))
        {
            let freqs: Vec<f32> = (0..dim / 2)
                .map(|j| 1.0f32 / ((theta as f64).powf((2 * j) as f64 / dim as f64) as f32))
                .collect();
            entry.insert(self.stream.clone_htod(&freqs).map_err(backend)?);
        }
        let frequencies = tables.get(&(dim, theta.to_bits())).unwrap();
        // SAFETY: each thread updates a unique pair of dimensions in an exclusively borrowed tensor.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&mut x.data)
                .arg(&m.positions)
                .arg(frequencies)
                .arg(&n)
                .arg(&h)
                .arg(&d)
                .launch(Self::flat(m.tokens * heads * (dim / 2))?)
        }
        .map_err(backend)?;
        Ok(())
    }
    fn swiglu(&self, g: &Tensor, u: &Tensor) -> Result<Tensor> {
        self.check(g)?;
        self.check(u)?;
        if g.shape != u.shape {
            return Err(invalid("SwiGLU shape mismatch"));
        }
        let mut out = self.zeros(&g.shape)?;
        let n = int(g.data.len())?;
        let fun = self.function("swiglu")?;
        // SAFETY: equal sizes; one checked output element per thread.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&g.data)
                .arg(&u.data)
                .arg(&mut out.data)
                .arg(&n)
                .launch(Self::flat(g.data.len())?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn store_kv(
        &self,
        k: &Tensor,
        v: &Tensor,
        kc: &mut Tensor,
        vc: &mut Tensor,
        m: &Metadata,
    ) -> Result<()> {
        for x in [k, v, &*kc, &*vc] {
            self.check(x)?;
        }
        self.check_meta(m)?;
        if k.shape.len() != 2
            || k.shape != v.shape
            || k.shape[0] != m.tokens
            || kc.shape.len() != 4
            || kc.shape != vc.shape
            || kc.shape[0] != m.pages
            || kc.shape[1] != m.page_size
            || kc.shape[2] * kc.shape[3] != k.shape[1]
        {
            return Err(invalid("KV scatter shape mismatch"));
        }
        let n = int(m.tokens)?;
        let w = int(k.shape[1])?;
        let fun = self.function("scatter")?;
        // SAFETY: metadata proves slots are in range and unique; K/V caches are exclusively borrowed.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&k.data)
                .arg(&v.data)
                .arg(&m.slots)
                .arg(&mut kc.data)
                .arg(&mut vc.data)
                .arg(&n)
                .arg(&w)
                .launch(Self::flat(k.data.len())?)
        }
        .map_err(backend)?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        m: &Metadata,
        qheads: usize,
        kvheads: usize,
        dim: usize,
    ) -> Result<Tensor> {
        for x in [q, kc, vc] {
            self.check(x)?;
        }
        self.check_meta(m)?;
        if !(1..=256).contains(&dim)
            || kvheads == 0
            || !qheads.is_multiple_of(kvheads)
            || q.shape != [m.tokens, qheads * dim]
            || kc.shape != [m.pages, m.page_size, kvheads, dim]
            || vc.shape != kc.shape
        {
            return Err(invalid("paged GQA shape mismatch"));
        }
        let mut out = self.zeros(&q.shape)?;
        let qh = int(qheads)?;
        let kh = int(kvheads)?;
        let d = int(dim)?;
        let p = int(m.page_size)?;
        let tw = int(m.table_width)?;
        let scale_q = (dim as f64).powf(-0.25) as f32;
        let fun = self.function("attention")?;
        let cfg = LaunchConfig {
            grid_dim: (m.tokens as u32, qheads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: CTA(row,head) exclusively owns its output. Metadata guarantees page coverage
        // through each causal position. Shared query <=256 and every CTA uses 4 full warps.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&q.data)
                .arg(&kc.data)
                .arg(&vc.data)
                .arg(&m.positions)
                .arg(&m.sequences)
                .arg(&m.table)
                .arg(&mut out.data)
                .arg(&qh)
                .arg(&kh)
                .arg(&d)
                .arg(&p)
                .arg(&tw)
                .arg(&scale_q)
                .launch(cfg)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn last_hidden(&self, x: &Tensor, m: &Metadata) -> Result<Tensor> {
        self.check(x)?;
        self.check_meta(m)?;
        if x.shape.len() != 2 || x.shape[0] != m.tokens {
            return Err(invalid("last hidden shape mismatch"));
        }
        let mut out = self.zeros(&[m.batch, x.shape[1]])?;
        let n = int(m.batch)?;
        let d = int(x.shape[1])?;
        let fun = self.function("gather")?;
        // SAFETY: metadata's last_rows are within the input; each output element is unique.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&x.data)
                .arg(&m.last_rows)
                .arg(&mut out.data)
                .arg(&n)
                .arg(&d)
                .launch(Self::flat(m.batch * x.shape[1])?)
        }
        .map_err(backend)?;
        Ok(out)
    }
    fn logits(&self, x: &Tensor, w: &Tensor) -> Result<Logits> {
        let (m, n, _) = self.linear_shape(x, w)?;
        let mut data = self
            .stream
            .alloc_zeros::<f32>(elems(&[m, n])?)
            .map_err(backend)?;
        self.gemm_to(x, w, &mut data, blas_sys::cudaDataType_t::CUDA_R_32F)?;
        Ok(Logits {
            data,
            rows: m,
            cols: n,
        })
    }
    fn argmax(&self, l: &Logits) -> Result<Vec<u32>> {
        if l.data.context().as_ref() != self.ctx.as_ref() {
            return Err(invalid("logits context mismatch"));
        }
        let mut out = self.stream.alloc_zeros::<u32>(l.rows).map_err(backend)?;
        let d = int(l.cols)?;
        let fun = self.function("argmax")?;
        // SAFETY: one 256-thread CTA per row, exclusive result slot and bounded logits reads.
        unsafe {
            self.stream
                .launch_builder(&fun)
                .arg(&l.data)
                .arg(&mut out)
                .arg(&d)
                .launch(Self::rows(l.rows, 256)?)
        }
        .map_err(backend)?;
        self.stream.clone_dtoh(&out).map_err(backend)
    }
    fn download_logits_row(&self, l: &Logits, row: usize) -> Result<Vec<f32>> {
        if row >= l.rows || l.data.context().as_ref() != self.ctx.as_ref() {
            return Err(invalid("logits row/context mismatch"));
        }
        self.stream
            .clone_dtoh(&l.data.slice(row * l.cols..(row + 1) * l.cols))
            .map_err(backend)
    }
    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(backend)
    }
    fn memory_bytes(&self) -> Option<u64> {
        self.memory_info()
            .ok()
            .map(|(free, total)| (total - free) as u64)
    }
}
