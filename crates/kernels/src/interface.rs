//! Backend-independent model operations; device buffers remain opaque.
use half::bf16;
use rsglang_core::{Result, StepBatch};
use rsglang_distributed::TensorParallel;

pub struct ExpertAssignment {
    pub expert: usize,
    pub rows: Vec<u32>,
    pub weights: Vec<f32>,
}
/// Model-level operations, with opaque backend-specific buffers and metadata.
/// Implementations preserve buffer lifetimes across launches and use one ordered stream.
pub trait KernelBackend {
    type Tensor;
    type Logits;
    type Metadata;
    type Accumulator;
    fn token_count(&self, meta: &Self::Metadata) -> usize;
    fn tensor_parallel(&self) -> TensorParallel {
        TensorParallel::default()
    }
    fn abort(&self) {}
    fn upload(&self, values: &[bf16], shape: &[usize]) -> Result<Self::Tensor>;
    fn zeros(&self, shape: &[usize]) -> Result<Self::Tensor>;
    fn metadata(
        &self,
        batch: &StepBatch,
        page_size: usize,
        pages: usize,
        vocab: usize,
        max_seq_len: usize,
    ) -> Result<Self::Metadata>;
    fn embedding(&self, weights: &Self::Tensor, meta: &Self::Metadata) -> Result<Self::Tensor>;
    fn embedding_shard(
        &self,
        weights: &Self::Tensor,
        meta: &Self::Metadata,
        start: usize,
    ) -> Result<Self::Tensor>;
    fn all_reduce(&self, tensor: Self::Tensor) -> Result<Self::Tensor>;
    fn all_gather_logits(&self, logits: Self::Logits, vocab: usize) -> Result<Self::Logits>;
    fn router_topk(
        &self,
        logits: &Self::Tensor,
        topk: usize,
        renormalize: bool,
    ) -> Result<Vec<ExpertAssignment>>;
    fn gather_rows(&self, tensor: &Self::Tensor, rows: &[u32]) -> Result<Self::Tensor>;
    fn accumulator(&self, rows: usize, cols: usize) -> Result<Self::Accumulator>;
    fn scatter_weighted(
        &self,
        accumulator: &mut Self::Accumulator,
        values: &Self::Tensor,
        assignment: &ExpertAssignment,
    ) -> Result<()>;
    fn finish_accumulator(&self, accumulator: Self::Accumulator) -> Result<Self::Tensor>;
    fn rms_norm(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        width: usize,
        eps: f32,
    ) -> Result<Self::Tensor>;
    fn linear(&self, x: &Self::Tensor, weights: &Self::Tensor) -> Result<Self::Tensor>;
    fn linear_reduce(&self, x: &Self::Tensor, weights: &Self::Tensor) -> Result<Self::Tensor>;
    fn concat_rows(&self, tensors: &[Self::Tensor]) -> Result<Self::Tensor>;
    fn split_columns(&self, tensor: &Self::Tensor, widths: &[usize]) -> Result<Vec<Self::Tensor>>;
    fn add(&self, x: &Self::Tensor, y: &Self::Tensor) -> Result<Self::Tensor>;
    fn rope(
        &self,
        x: &mut Self::Tensor,
        meta: &Self::Metadata,
        heads: usize,
        dim: usize,
        theta: f32,
    ) -> Result<()>;
    fn swiglu(&self, gate: &Self::Tensor, up: &Self::Tensor) -> Result<Self::Tensor>;
    fn store_kv(
        &self,
        k: &Self::Tensor,
        v: &Self::Tensor,
        kc: &mut Self::Tensor,
        vc: &mut Self::Tensor,
        meta: &Self::Metadata,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Self::Tensor,
        kc: &Self::Tensor,
        vc: &Self::Tensor,
        meta: &Self::Metadata,
        qheads: usize,
        kvheads: usize,
        dim: usize,
    ) -> Result<Self::Tensor>;
    fn last_hidden(&self, x: &Self::Tensor, meta: &Self::Metadata) -> Result<Self::Tensor>;
    fn logits(&self, x: &Self::Tensor, weights: &Self::Tensor) -> Result<Self::Logits>;
    fn argmax(&self, logits: &Self::Logits) -> Result<Vec<u32>>;
    fn download_logits_row(&self, logits: &Self::Logits, row: usize) -> Result<Vec<f32>>;
    fn synchronize(&self) -> Result<()>;
    fn memory_bytes(&self) -> Option<u64> {
        None
    }
}
