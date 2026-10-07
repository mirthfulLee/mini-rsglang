use half::bf16;
use rsglang_kernels::{CudaBackend, KernelBackend};
fn main() -> rsglang_core::Result<()> {
    let b = CudaBackend::new(0)?;
    let x = b.upload(&[1., 2., 3., 4., 5., 6.].map(bf16::from_f32), &[2, 3])?;
    let w = b.upload(&[1., 0., 1., 0., 2., 0.].map(bf16::from_f32), &[2, 3])?;
    let y = b.linear(&x, &w)?;
    let got = b.download(&y)?;
    assert_eq!(got, vec![4., 4., 10., 10.]);
    let norm = b.upload(&[bf16::ONE; 3], &[3])?;
    let z = b.rms_norm(&x, &norm, 3, 1e-6)?;
    let z = b.download(&z)?;
    assert!((z[0] - 1.0 / (14.0f32 / 3.0).sqrt()).abs() < 0.005);
    let logits = b.logits(&x, &w)?;
    assert_eq!(b.argmax(&logits)?, vec![0, 0]);
    println!(
        "CUDA BF16 GEMM, NVRTC RMSNorm, FP32 logits and tie-breaking argmax passed; memory={:?}",
        b.memory_info()?
    );
    Ok(())
}
