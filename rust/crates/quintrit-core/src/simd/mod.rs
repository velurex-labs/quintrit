//! Architecture-specific SIMD kernels and portable fallback for velurex encoding GEMV.
//!
//! This is the Quintrit implementation (5 trits per byte) and the place where kernels are optimized.

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;

pub mod scalar;
