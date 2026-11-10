//! # Strict 1:1 Pure-Rust Reference Implementation of Microsoft BitNet (2.00 bpw)
//!
//! ## Frozen baseline
//!
//! This module is a faithful transcription of the original and is intentionally **not** optimized,
//! extended or restructured: once a baseline has been tuned it no longer shows what the original does.
//! Optimizations belong in `quintrit_core::simd` and `quintrit_core::lut`. The widening of the planar MAD kernel to
//! AVX-512 (`avx512`) keeps BitNet's algorithm and encoding unchanged; Microsoft ships no AVX-512 kernel,
//! and the port exists so that the 4-trit encoding can be compared with the 5-trit encoding on the same
//! instruction set.
//!
//! ## Architectural Purpose and Rationale
//! This module provides a faithful, exact 1:1 port of Microsoft's official BitNet b1.58
//! C++ inference kernels (`ggml-bitnet-mad.cpp` and `ggml-bitnet-lut.cpp` / T-MAC) into pure Rust.
//!
//! ### Why Pure Rust Port instead of C++ FFI?
//! 1. **Zero External Dependencies**: Enables standalone evaluation, building, and testing
//!    on any platform (Windows, Linux, macOS, Raspberry Pi 5) without requiring CMake,
//!    C++ compilers, or MSVC build tools.
//! 2. **Reproducible Baseline Verification**: Guarantees identical execution semantics and
//!    throughput for empirical benchmarking against Quintrit (1.60 bpw velurex encoding)
//!    under the exact same compiler optimization flags and runtime environment.
//! 3. **No Unsafe FFI Boundary Overhead**: Eliminates cross-language calling overhead while
//!    preserving identical SIMD instruction sequences (AVX2 `vpmaddubsw`, `vpshufb`, NEON `vqtbl1q_s8`).

pub mod scalar;
pub mod mad;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use scalar::{
    build_activation_lut_scalar, build_activation_lut_scalar as build_bitnet_lut,
    build_pair_lut_entry, dot_ternary_int8_scalar, gemv_lut_tiled32_scalar,
    gemv_ternary_int8_scalar, pack_trits_4_scalar as pack_trits_4,
    pack_trits_tiled_32, transform_weights_tiled_32,
    transform_weights_tiled_32 as transform_weights_bitnet_lut,
    unpack_trits_4_scalar as unpack_trits_4,
    untransform_weights_tiled_32,
};

pub use mad::{
    gemv_mad, gemv_mad_parallel, gemv_mad_scalar, pack_gguf_i2_s, pack_trits_planar,
    pack_trits_planar as pack_trits_bitnet_planar, unpack_gguf_i2_s, unpack_trits_planar,
};
#[cfg(target_arch = "x86_64")]
pub use mad::{gemv_mad_1x1_avx2, gemv_mad_1xn_avx2, gemv_mad_avx2};
#[cfg(target_arch = "x86_64")]
pub use avx512::{
    gemv_mad_1x1_avx512, gemv_mad_1xn_avx512, gemv_mad_avx512, gemv_mad_parallel_avx512,
    hsum_i32_avx512,
};
#[cfg(target_arch = "aarch64")]
pub use mad::{
    gemv_mad_1x1_neon, gemv_mad_1x1_neon_dotprod, gemv_mad_1xn_neon, gemv_mad_1xn_neon_dotprod,
    gemv_mad_neon,
};

/// Evaluates single-threaded 1xN GEMV using BitNet's planar bit-slicing MAD format.
#[inline(always)]
pub fn gemv_bitnet_mad_1xn(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && rows % mad::PARALLEL_SIZE == 0 {
            unsafe {
                mad::gemv_mad_1xn_avx2(packed_weights, activations, rows, cols, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if rows % mad::PARALLEL_SIZE == 0 {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                unsafe {
                    mad::gemv_mad_1xn_neon_dotprod(packed_weights, activations, rows, cols, out);
                }
                return;
            } else {
                unsafe {
                    mad::gemv_mad_1xn_neon(packed_weights, activations, rows, cols, out);
                }
                return;
            }
        }
    }
    mad::gemv_mad(packed_weights, activations, rows, cols, out);
}

/// Evaluates multi-threaded GEMV using BitNet's planar bit-slicing MAD format.
#[inline(always)]
pub fn gemv_bitnet_mad_parallel(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    threads: usize,
) {
    mad::gemv_mad_parallel(packed_weights, activations, rows, cols, out, threads);
}

/// Evaluates single-threaded 1xN GEMV using BitNet's planar MAD kernel ported to AVX-512.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn gemv_bitnet_mad_avx512_1xn(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    avx512::gemv_mad_avx512(packed_weights, activations, rows, cols, out);
}

/// Evaluates multi-threaded GEMV using BitNet's planar MAD kernel ported to AVX-512.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn gemv_bitnet_mad_avx512_parallel(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    threads: usize,
) {
    avx512::gemv_mad_parallel_avx512(packed_weights, activations, rows, cols, out, threads);
}

/// Evaluates single-threaded 32-row tiled GEMV against precomputed BitNet activation LUT.
#[inline(always)]
pub fn gemv_bitnet_lut(
    tiled_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::gemv_lut_tiled32_avx2(tiled_weights, qlut, rows, cols, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::gemv_lut_tiled32_neon(tiled_weights, qlut, rows, cols, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    scalar::gemv_lut_tiled32_scalar(tiled_weights, qlut, rows, cols, out);
}

/// Evaluates multi-threaded 32-row tiled GEMV against precomputed BitNet activation LUT.
#[inline(always)]
pub fn gemv_bitnet_lut_parallel(
    tiled_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
    threads: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            avx2::gemv_lut_tiled32_parallel(tiled_weights, qlut, rows, cols, out, threads);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        neon::gemv_lut_tiled32_parallel(tiled_weights, qlut, rows, cols, out, threads);
        return;
    }
    #[allow(unreachable_code)]
    scalar::gemv_lut_tiled32_scalar(tiled_weights, qlut, rows, cols, out);
}

/// Precomputed Lookup Table (LUT) for an input activation vector `X` (BitNet T-MAC baseline).
#[derive(Clone, Debug)]
pub struct TernaryLut {
    tables: Vec<i8>,
    trit_count: usize,
}

impl TernaryLut {
    /// Constructs lookup tables from an input activation slice `x`.
    pub fn new(x: &[i8]) -> Self {
        let tables = scalar::build_activation_lut_scalar(x);
        Self {
            tables,
            trit_count: x.len(),
        }
    }

    /// Returns the number of activation values represented.
    #[inline]
    pub fn trit_count(&self) -> usize {
        self.trit_count
    }

    /// Returns the underlying flattened 16-byte tables.
    #[inline]
    pub fn tables(&self) -> &[i8] {
        &self.tables
    }
}
