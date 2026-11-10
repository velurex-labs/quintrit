//! # Quintrit Core (`quintrit-core`)
//!
//! High-performance 1.60 bpw ternary quantization microkernel using **velurex encoding**
//! and hardware-accelerated SIMD GEMV kernels.
//!
//! ## Overview
//!
//! This microkernel packs five ternary weights `{-1, 0, +1}` into a single byte (`0..=250`)
//! via velurex encoding ($1, 3, 9, 28, 84$ radix decomposition):
//! - **Exact 20.0% DRAM reduction** compared to 2.00 bpw baselines (`GGML_TYPE_I2_S`).
//! - **Zero-modulo SIMD decoding**: The gap at radix 28 eliminates integer modulo and division instructions.
//! - **Hardware-Accelerated GEMV**: tiled activation lookup table (QLUT) execution: 32-row tiles for AVX2 and ARM NEON, 64-row tiles for AVX-512, and a portable scalar fallback.
//! - **Silent**: the library never writes to the standard output.
//! - **Persistent Multi-Threading**: Zero-overhead worker pool parallelism via Rayon for scalable multi-core GEMV.
//!
//! ## Module layout
//!
//! - [`simd`], [`lut`], [`pack`]: the Quintrit implementation (velurex encoding, 5 trits per byte).
//!   This is where kernels are optimized and extended.

pub mod limits;
pub mod lut;
pub mod pack;
pub mod simd;
pub mod threading;

pub use lut::{
    build_activation_lut, build_activation_lut_wide, build_activation_lut_wide_into,
    gemv_lut_1xn, gemv_lut_parallel, gemv_lut_wide_1xn, gemv_lut_wide_parallel,
    split_activations_hi_lo, transform_weights_lut,
    untransform_weights_lut, wide_activation_lut_len,
};

#[cfg(target_arch = "x86_64")]
pub use lut::{
    build_activation_lut_avx512, gemv_lut_64_1xn_avx512, gemv_lut_64_parallel_avx512,
    transform_weights_lut_64, untransform_weights_lut_64,
};

pub use limits::{gemv_cols_supported, MAX_GEMV_COLS, MAX_THREADS};

pub use pack::{
    pack_5trits, pack_matrix, split_activation_planes, unpack_matrix, BLOCK_BYTES_5,
    BYTE_TO_TRITS, QK_5,
};

/// Compiles the Rust examples of the crate README as doctests, so that they cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
