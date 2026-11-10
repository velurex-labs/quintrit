//! Exact 1:1 Rust port of Microsoft's BitNet Multiply-Add (MAD) GEMV kernels (`ggml-bitnet-mad.cpp`).
//!
//! Features:
//! - **Planar Block-Interleaved Bit-Slicing**: 128 trits per 32-byte block (x86_64) or 64 trits per 16-byte block (aarch64).
//! - **AVX2 1xN Multi-Row Tiled GEMV**: Evaluates `PARALLEL_SIZE = 4` rows concurrently, reusing
//!   activation registers across 4 rows and saturating `_mm256_maddubs_epi16`.
//! - **AVX2 1x1 GEMV**: Single-row kernel for arbitrary row counts and tile remainders.
//! - **ARM NEON 1xN Multi-Row Tiled GEMV**: Evaluates `PARALLEL_SIZE = 4` rows concurrently using
//!   ARM NEON dot-product (`vdotq_s32`) or widening multiply-accumulate (`vmlal_s8`).
//! - **ARM NEON 1x1 GEMV**: Single-row kernel for arbitrary row counts on AArch64.
//! - **Multi-Threaded GEMV**: Zero-allocation thread-scoped work distribution matching `bitnet_ffi.cpp`.
//! - **Scalar Fallback**: Portable reference kernel for non-SIMD architectures and verification.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
pub const QK_I2_S: usize = 128;
#[cfg(not(target_arch = "x86_64"))]
pub const QK_I2_S: usize = 64;

pub const BLOCK_BYTES: usize = QK_I2_S / 4;
pub const GROUP_SIZE: usize = QK_I2_S / 4;
pub const PARALLEL_SIZE: usize = 4;

// ============================================================================
// Planar Packing / Unpacking (1:1 with bitnet_pack_planar_i8)
// ============================================================================

/// Packs ternary weights in `{-1, 0, +1}` into BitNet's planar block-interleaved bit-slicing format.
///
/// Matrix dimensions: `rows x cols` (where `cols` must be a multiple of 128).
/// Returns packed byte vector of length `(rows * cols) / 4`.
pub fn pack_trits_planar(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(
        trits.len(),
        rows * cols,
        "Input slice length must match rows * cols"
    );
    assert_eq!(
        cols % QK_I2_S,
        0,
        "cols must be a multiple of 128 (QK_I2_S) for BitNet planar packing"
    );

    let total_bytes = (rows * cols) / 4;
    let mut dst = vec![0u8; total_bytes];
    let bytes_per_row = cols / 4;
    let num_blocks = cols / QK_I2_S;

    for r in 0..rows {
        let row_trits = &trits[r * cols..(r + 1) * cols];
        let row_dst = &mut dst[r * bytes_per_row..(r + 1) * bytes_per_row];

        for b in 0..num_blocks {
            let block_trits = &row_trits[b * QK_I2_S..(b + 1) * QK_I2_S];
            let block_dst = &mut row_dst[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];

            for j in 0..QK_I2_S {
                let group_idx = j / GROUP_SIZE; // 0, 1, 2, 3
                let group_pos = j % GROUP_SIZE; // 0..GROUP_SIZE-1
                let w = block_trits[j];
                let code = match w {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8,
                };
                let shift = 6 - 2 * group_idx;
                block_dst[group_pos] |= code << shift;
            }
        }
    }

    dst
}

/// Unpacks BitNet planar bit-sliced weights back to `{-1, 0, +1}`.
pub fn unpack_trits_planar(packed: &[u8], rows: usize, cols: usize) -> Vec<i8> {
    assert_eq!(
        packed.len(),
        (rows * cols) / 4,
        "Packed buffer length must match (rows * cols) / 4"
    );
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 128");

    let mut trits = vec![0i8; rows * cols];
    let bytes_per_row = cols / 4;
    let num_blocks = cols / QK_I2_S;

    for r in 0..rows {
        let row_dst = &mut trits[r * cols..(r + 1) * cols];
        let row_src = &packed[r * bytes_per_row..(r + 1) * bytes_per_row];

        for b in 0..num_blocks {
            let block_src = &row_src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let block_dst = &mut row_dst[b * QK_I2_S..(b + 1) * QK_I2_S];

            for j in 0..QK_I2_S {
                let group_idx = j / GROUP_SIZE;
                let group_pos = j % GROUP_SIZE;
                let shift = 6 - 2 * group_idx;
                let code = (block_src[group_pos] >> shift) & 0x03;
                let w = match code {
                    0 => -1i8,
                    1 => 0i8,
                    2 => 1i8,
                    _ => 0i8,
                };
                block_dst[j] = w;
            }
        }
    }

    trits
}

/// Canonical GGUF `GGML_TYPE_I2_S` block dimensions (invariant across all host CPU architectures).
pub const QK_GGUF_I2_S: usize = 128;
pub const BLOCK_BYTES_GGUF_I2_S: usize = 32;
pub const GROUP_SIZE_GGUF_I2_S: usize = 32;

/// Packs ternary weights in `{-1, 0, +1}` into canonical GGUF `GGML_TYPE_I2_S` planar format
/// (always 128 trits / 32 bytes per block, invariant across host CPU architectures).
pub fn pack_gguf_i2_s(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(
        trits.len(),
        rows * cols,
        "Input slice length must match rows * cols"
    );
    assert_eq!(
        cols % QK_GGUF_I2_S,
        0,
        "cols must be a multiple of 128 (QK_GGUF_I2_S) for GGUF I2_S packing"
    );

    let total_bytes = (rows * cols) / 4;
    let mut dst = vec![0u8; total_bytes];
    let bytes_per_row = cols / 4;
    let num_blocks = cols / QK_GGUF_I2_S;

    for r in 0..rows {
        let row_trits = &trits[r * cols..(r + 1) * cols];
        let row_dst = &mut dst[r * bytes_per_row..(r + 1) * bytes_per_row];

        for b in 0..num_blocks {
            let block_trits = &row_trits[b * QK_GGUF_I2_S..(b + 1) * QK_GGUF_I2_S];
            let block_dst = &mut row_dst[b * BLOCK_BYTES_GGUF_I2_S..(b + 1) * BLOCK_BYTES_GGUF_I2_S];

            for j in 0..QK_GGUF_I2_S {
                let group_idx = j / GROUP_SIZE_GGUF_I2_S;
                let group_pos = j % GROUP_SIZE_GGUF_I2_S;
                let w = block_trits[j];
                let code = match w {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8,
                };
                let shift = 6 - 2 * group_idx;
                block_dst[group_pos] |= code << shift;
            }
        }
    }

    dst
}

/// Unpacks canonical GGUF `GGML_TYPE_I2_S` planar weights (always 128 trits / 32 bytes per block,
/// invariant across all host CPU architectures).
pub fn unpack_gguf_i2_s(packed: &[u8], rows: usize, cols: usize) -> Vec<i8> {
    assert_eq!(
        cols % QK_GGUF_I2_S,
        0,
        "cols must be a multiple of 128 (QK_GGUF_I2_S)"
    );
    assert_eq!(
        packed.len(),
        (rows * cols) / 4,
        "Packed buffer length must match (rows * cols) / 4"
    );

    let mut trits = vec![0i8; rows * cols];
    let bytes_per_row = cols / 4;
    let num_blocks = cols / QK_GGUF_I2_S;

    for r in 0..rows {
        let row_dst = &mut trits[r * cols..(r + 1) * cols];
        let row_src = &packed[r * bytes_per_row..(r + 1) * bytes_per_row];

        for b in 0..num_blocks {
            let block_src = &row_src[b * BLOCK_BYTES_GGUF_I2_S..(b + 1) * BLOCK_BYTES_GGUF_I2_S];
            let block_dst = &mut row_dst[b * QK_GGUF_I2_S..(b + 1) * QK_GGUF_I2_S];

            for j in 0..QK_GGUF_I2_S {
                let group_idx = j / GROUP_SIZE_GGUF_I2_S;
                let group_pos = j % GROUP_SIZE_GGUF_I2_S;
                let shift = 6 - 2 * group_idx;
                let code = (block_src[group_pos] >> shift) & 0x03;
                let w = match code {
                    0 => -1i8,
                    1 => 0i8,
                    2 => 1i8,
                    _ => 0i8,
                };
                block_dst[j] = w;
            }
        }
    }

    trits
}

// ============================================================================
// Scalar Reference Implementation
// ============================================================================

/// Computes GEMV Matrix-Vector product Y = W * X using portable scalar logic.
pub fn gemv_mad_scalar(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);
    assert_eq!(cols % QK_I2_S, 0);

    let bytes_per_row = cols / 4;
    let num_blocks = cols / QK_I2_S;

    for r in 0..rows {
        let row_src = &packed_weights[r * bytes_per_row..(r + 1) * bytes_per_row];
        let mut row_sum = 0i32;

        for b in 0..num_blocks {
            let block_src = &row_src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let act_block = &activations[b * QK_I2_S..(b + 1) * QK_I2_S];

            for j in 0..QK_I2_S {
                let group_idx = j / GROUP_SIZE;
                let group_pos = j % GROUP_SIZE;
                let shift = 6 - 2 * group_idx;
                let code = (block_src[group_pos] >> shift) & 0x03;
                let y = act_block[j] as i32;
                row_sum += (code as i32) * y;
            }
        }

        out[r] = row_sum as f32;
    }
}

// ============================================================================
// AVX2 SIMD Implementation (1:1 with ggml-bitnet-mad.cpp)
// ============================================================================

/// Computes GEMV Matrix-Vector product using BitNet's 1xN multi-row tiled AVX2 kernel (`PARALLEL_SIZE = 4`).
///
/// Rows must be a multiple of 4; cols must be a multiple of 128.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_mad_1xn_avx2(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % PARALLEL_SIZE, 0, "rows must be a multiple of 4 for 1xN");
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 128");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    unsafe {
        let mask = _mm256_set1_epi8(0x03);
        let one16 = _mm256_set1_epi16(1);
        let bytes_per_row = (cols / 4) as usize;

        let mut row = 0;
        while row < nrc {
            let mut accu = [_mm256_setzero_si256(); PARALLEL_SIZE];
            let mut x_row = [core::ptr::null::<u8>(); PARALLEL_SIZE];
            for rb in 0..PARALLEL_SIZE {
                x_row[rb] = x.add((row as usize + rb) * bytes_per_row);
            }

            for i in 0..group32_num {
                let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
                for rb in 0..PARALLEL_SIZE {
                    px[rb] = x_row[rb].add((i as usize) * 1024);
                }
                let mut py = y.add((i as usize) * 4096);

                for _j in 0..32 {
                    let yq8_0 = _mm256_loadu_si256(py as *const __m256i);
                    let yq8_1 = _mm256_loadu_si256(py.add(32) as *const __m256i);
                    let yq8_2 = _mm256_loadu_si256(py.add(64) as *const __m256i);
                    let yq8_3 = _mm256_loadu_si256(py.add(96) as *const __m256i);

                    for rb in 0..PARALLEL_SIZE {
                        let xq8_raw = _mm256_loadu_si256(px[rb] as *const __m256i);
                        let xq8_2 = _mm256_srli_epi16(xq8_raw, 2);
                        let xq8_1 = _mm256_srli_epi16(xq8_raw, 4);
                        let xq8_0 = _mm256_srli_epi16(xq8_raw, 6);

                        let q8_3 = _mm256_and_si256(xq8_raw, mask);
                        let q8_2 = _mm256_and_si256(xq8_2, mask);
                        let q8_1 = _mm256_and_si256(xq8_1, mask);
                        let q8_0 = _mm256_and_si256(xq8_0, mask);

                        let p0 = _mm256_maddubs_epi16(q8_0, yq8_0);
                        let p1 = _mm256_maddubs_epi16(q8_1, yq8_1);
                        let p2 = _mm256_maddubs_epi16(q8_2, yq8_2);
                        let p3 = _mm256_maddubs_epi16(q8_3, yq8_3);

                        let sum_01 = _mm256_add_epi16(p0, p1);
                        let sum_23 = _mm256_add_epi16(p2, p3);
                        let sum_all = _mm256_add_epi16(sum_01, sum_23);
                        accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(sum_all, one16));

                        px[rb] = px[rb].add(32);
                    }
                    py = py.add(128);
                }
            }

            for _i in 0..groupla_num {
                let mut py = y.add((group32_num as usize) * 4096);
                let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
                for rb in 0..PARALLEL_SIZE {
                    px[rb] = x_row[rb].add((group32_num as usize) * 1024);
                }

                for _j in 0..la_num {
                    let yq8_0 = _mm256_loadu_si256(py as *const __m256i);
                    let yq8_1 = _mm256_loadu_si256(py.add(32) as *const __m256i);
                    let yq8_2 = _mm256_loadu_si256(py.add(64) as *const __m256i);
                    let yq8_3 = _mm256_loadu_si256(py.add(96) as *const __m256i);

                    for rb in 0..PARALLEL_SIZE {
                        let xq8_raw = _mm256_loadu_si256(px[rb] as *const __m256i);
                        let xq8_2 = _mm256_srli_epi16(xq8_raw, 2);
                        let xq8_1 = _mm256_srli_epi16(xq8_raw, 4);
                        let xq8_0 = _mm256_srli_epi16(xq8_raw, 6);

                        let q8_3 = _mm256_and_si256(xq8_raw, mask);
                        let q8_2 = _mm256_and_si256(xq8_2, mask);
                        let q8_1 = _mm256_and_si256(xq8_1, mask);
                        let q8_0 = _mm256_and_si256(xq8_0, mask);

                        let p0 = _mm256_maddubs_epi16(q8_0, yq8_0);
                        let p1 = _mm256_maddubs_epi16(q8_1, yq8_1);
                        let p2 = _mm256_maddubs_epi16(q8_2, yq8_2);
                        let p3 = _mm256_maddubs_epi16(q8_3, yq8_3);

                        let sum_01 = _mm256_add_epi16(p0, p1);
                        let sum_23 = _mm256_add_epi16(p2, p3);
                        let sum_all = _mm256_add_epi16(sum_01, sum_23);
                        accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(sum_all, one16));

                        px[rb] = px[rb].add(32);
                    }
                    py = py.add(128);
                }
            }

            for rb in 0..PARALLEL_SIZE {
                let sumi = crate::avx2::hsum_i32_avx2(accu[rb]);
                *s.add(row as usize + rb) = sumi as f32;
            }

            row += PARALLEL_SIZE as i32;
        }
    }
}

/// Computes GEMV Matrix-Vector product using BitNet's 1x1 single-row AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_mad_1x1_avx2(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 128");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    unsafe {
        let mask = _mm256_set1_epi8(0x03);
        let one16 = _mm256_set1_epi16(1);
        let bytes_per_row = (cols / 4) as usize;

        for row in 0..nrc {
            let mut accu = _mm256_setzero_si256();
            let x_row = x.add(row as usize * bytes_per_row);

            for i in 0..group32_num {
                let mut px = x_row.add((i as usize) * 1024);
                let mut py = y.add((i as usize) * 4096);

                for _j in 0..32 {
                    let xq8_raw = _mm256_loadu_si256(px as *const __m256i);
                    let xq8_2 = _mm256_srli_epi16(xq8_raw, 2);
                    let xq8_1 = _mm256_srli_epi16(xq8_raw, 4);
                    let xq8_0 = _mm256_srli_epi16(xq8_raw, 6);

                    let q8_3 = _mm256_and_si256(xq8_raw, mask);
                    let q8_2 = _mm256_and_si256(xq8_2, mask);
                    let q8_1 = _mm256_and_si256(xq8_1, mask);
                    let q8_0 = _mm256_and_si256(xq8_0, mask);

                    let p0 = _mm256_maddubs_epi16(q8_0, yq8_0_from(py));
                    let p1 = _mm256_maddubs_epi16(q8_1, _mm256_loadu_si256(py.add(32) as *const __m256i));
                    let p2 = _mm256_maddubs_epi16(q8_2, _mm256_loadu_si256(py.add(64) as *const __m256i));
                    let p3 = _mm256_maddubs_epi16(q8_3, _mm256_loadu_si256(py.add(96) as *const __m256i));

                    let sum_01 = _mm256_add_epi16(p0, p1);
                    let sum_23 = _mm256_add_epi16(p2, p3);
                    let sum_all = _mm256_add_epi16(sum_01, sum_23);
                    accu = _mm256_add_epi32(accu, _mm256_madd_epi16(sum_all, one16));

                    px = px.add(32);
                    py = py.add(128);
                }
            }

            for _i in 0..groupla_num {
                let mut px = x_row.add((group32_num as usize) * 1024);
                let mut py = y.add((group32_num as usize) * 4096);

                for _j in 0..la_num {
                    let xq8_raw = _mm256_loadu_si256(px as *const __m256i);
                    let xq8_2 = _mm256_srli_epi16(xq8_raw, 2);
                    let xq8_1 = _mm256_srli_epi16(xq8_raw, 4);
                    let xq8_0 = _mm256_srli_epi16(xq8_raw, 6);

                    let q8_3 = _mm256_and_si256(xq8_raw, mask);
                    let q8_2 = _mm256_and_si256(xq8_2, mask);
                    let q8_1 = _mm256_and_si256(xq8_1, mask);
                    let q8_0 = _mm256_and_si256(xq8_0, mask);

                    let p0 = _mm256_maddubs_epi16(q8_0, _mm256_loadu_si256(py as *const __m256i));
                    let p1 = _mm256_maddubs_epi16(q8_1, _mm256_loadu_si256(py.add(32) as *const __m256i));
                    let p2 = _mm256_maddubs_epi16(q8_2, _mm256_loadu_si256(py.add(64) as *const __m256i));
                    let p3 = _mm256_maddubs_epi16(q8_3, _mm256_loadu_si256(py.add(96) as *const __m256i));

                    let sum_01 = _mm256_add_epi16(p0, p1);
                    let sum_23 = _mm256_add_epi16(p2, p3);
                    let sum_all = _mm256_add_epi16(sum_01, sum_23);
                    accu = _mm256_add_epi32(accu, _mm256_madd_epi16(sum_all, one16));

                    px = px.add(32);
                    py = py.add(128);
                }
            }

            let sumi = crate::avx2::hsum_i32_avx2(accu);
            *s.add(row as usize) = sumi as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn yq8_0_from(py: *const i8) -> __m256i {
    unsafe { _mm256_loadu_si256(py as *const __m256i) }
}

/// Computes GEMV Matrix-Vector product with automatic handling of remainder rows on AVX2.
#[cfg(target_arch = "x86_64")]
pub fn gemv_mad_avx2(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let full_tiles = (rows / PARALLEL_SIZE) * PARALLEL_SIZE;
    let bytes_per_row = cols / 4;

    if full_tiles > 0 {
        unsafe {
            gemv_mad_1xn_avx2(
                &packed_weights[..full_tiles * bytes_per_row],
                activations,
                full_tiles,
                cols,
                &mut out[..full_tiles],
            );
        }
    }

    if full_tiles < rows {
        let rem_rows = rows - full_tiles;
        unsafe {
            gemv_mad_1x1_avx2(
                &packed_weights[full_tiles * bytes_per_row..],
                activations,
                rem_rows,
                cols,
                &mut out[full_tiles..],
            );
        }
    }
}

// ============================================================================
// ARM NEON SIMD Implementation (1:1 with ggml-bitnet-mad.cpp)
// ============================================================================

/// Computes GEMV Matrix-Vector product using BitNet's 1xN multi-row tiled NEON kernel with DotProd acceleration (`vdotq_s32`).
///
/// Rows must be a multiple of 4; cols must be a multiple of 64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv_mad_1xn_neon_dotprod(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % PARALLEL_SIZE, 0, "rows must be a multiple of 4 for 1xN");
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 64 (QK_I2_S)");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    let mask = vdupq_n_u8(0x03);
    let bytes_per_row = cols / 4;

    let mut row = 0;
    while row < nrc {
        let mut accu = [vdupq_n_s32(0); PARALLEL_SIZE];
        let mut x_row = [core::ptr::null::<u8>(); PARALLEL_SIZE];
        for rb in 0..PARALLEL_SIZE {
            x_row[rb] = x.add((row as usize + rb) * bytes_per_row);
        }

        for i in 0..group32_num {
            let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
            for rb in 0..PARALLEL_SIZE {
                px[rb] = x_row[rb].add((i as usize) * 32 * 16);
            }

            for j in 0..32 {
                let py = y.add((i as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                for rb in 0..PARALLEL_SIZE {
                    let xq8_3 = vld1q_u8(px[rb]);
                    let xq8_2 = vshrq_n_u8(xq8_3, 2);
                    let xq8_1 = vshrq_n_u8(xq8_3, 4);
                    let xq8_0 = vshrq_n_u8(xq8_3, 6);

                    let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);

                    px[rb] = px[rb].add(16);
                }
            }
        }

        for _i in 0..groupla_num {
            let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
            for rb in 0..PARALLEL_SIZE {
                px[rb] = x_row[rb].add((group32_num as usize) * 32 * 16);
            }

            for j in 0..la_num {
                let py = y.add((group32_num as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                for rb in 0..PARALLEL_SIZE {
                    let xq8_3 = vld1q_u8(px[rb]);
                    let xq8_2 = vshrq_n_u8(xq8_3, 2);
                    let xq8_1 = vshrq_n_u8(xq8_3, 4);
                    let xq8_0 = vshrq_n_u8(xq8_3, 6);

                    let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);

                    px[rb] = px[rb].add(16);
                }
            }
        }

        for rb in 0..PARALLEL_SIZE {
            let sumi = vaddlvq_s32(accu[rb]);
            *s.add(row as usize + rb) = sumi as f32;
        }

        row += PARALLEL_SIZE as i32;
    }
}

/// Computes GEMV Matrix-Vector product using BitNet's 1xN multi-row tiled baseline NEON kernel (`vmlal_s8`).
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_mad_1xn_neon(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % PARALLEL_SIZE, 0, "rows must be a multiple of 4 for 1xN");
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 64 (QK_I2_S)");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    let mask = vdupq_n_u8(0x03);
    let bytes_per_row = cols / 4;

    let mut row = 0;
    while row < nrc {
        let mut accu = [vdupq_n_s32(0); PARALLEL_SIZE];
        let mut x_row = [core::ptr::null::<u8>(); PARALLEL_SIZE];
        for rb in 0..PARALLEL_SIZE {
            x_row[rb] = x.add((row as usize + rb) * bytes_per_row);
        }

        for i in 0..group32_num {
            let mut accu32 = [vdupq_n_s16(0); PARALLEL_SIZE];
            let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
            for rb in 0..PARALLEL_SIZE {
                px[rb] = x_row[rb].add((i as usize) * 32 * 16);
            }

            for j in 0..32 {
                let py = y.add((i as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                for rb in 0..PARALLEL_SIZE {
                    let xq8_3 = vld1q_u8(px[rb]);
                    let xq8_2 = vshrq_n_u8(xq8_3, 2);
                    let xq8_1 = vshrq_n_u8(xq8_3, 4);
                    let xq8_0 = vshrq_n_u8(xq8_3, 6);

                    let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));

                    px[rb] = px[rb].add(16);
                }
            }

            for rb in 0..PARALLEL_SIZE {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu32[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu32[rb]));
            }
        }

        for _i in 0..groupla_num {
            let mut accula = [vdupq_n_s16(0); PARALLEL_SIZE];
            let mut px = [core::ptr::null::<u8>(); PARALLEL_SIZE];
            for rb in 0..PARALLEL_SIZE {
                px[rb] = x_row[rb].add((group32_num as usize) * 32 * 16);
            }

            for j in 0..la_num {
                let py = y.add((group32_num as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                for rb in 0..PARALLEL_SIZE {
                    let xq8_3 = vld1q_u8(px[rb]);
                    let xq8_2 = vshrq_n_u8(xq8_3, 2);
                    let xq8_1 = vshrq_n_u8(xq8_3, 4);
                    let xq8_0 = vshrq_n_u8(xq8_3, 6);

                    let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));

                    px[rb] = px[rb].add(16);
                }
            }

            for rb in 0..PARALLEL_SIZE {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accula[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accula[rb]));
            }
        }

        for rb in 0..PARALLEL_SIZE {
            let sumi = vaddlvq_s32(accu[rb]);
            *s.add(row as usize + rb) = sumi as f32;
        }

        row += PARALLEL_SIZE as i32;
    }
}

/// Computes GEMV Matrix-Vector product using BitNet's 1x1 single-row NEON kernel with DotProd.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv_mad_1x1_neon_dotprod(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 64 (QK_I2_S)");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    let mask = vdupq_n_u8(0x03);
    let bytes_per_row = cols / 4;

    for row in 0..nrc {
        let mut accu = vdupq_n_s32(0);
        let x_row = x.add(row as usize * bytes_per_row);

        for i in 0..group32_num {
            let mut px = x_row.add((i as usize) * 32 * 16);

            for j in 0..32 {
                let py = y.add((i as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                let xq8_3 = vld1q_u8(px);
                let xq8_2 = vshrq_n_u8(xq8_3, 2);
                let xq8_1 = vshrq_n_u8(xq8_3, 4);
                let xq8_0 = vshrq_n_u8(xq8_3, 6);

                let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);

                px = px.add(16);
            }
        }

        for _i in 0..groupla_num {
            let mut px = x_row.add((group32_num as usize) * 32 * 16);

            for j in 0..la_num {
                let py = y.add((group32_num as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                let xq8_3 = vld1q_u8(px);
                let xq8_2 = vshrq_n_u8(xq8_3, 2);
                let xq8_1 = vshrq_n_u8(xq8_3, 4);
                let xq8_0 = vshrq_n_u8(xq8_3, 6);

                let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);

                px = px.add(16);
            }
        }

        let sumi = vaddlvq_s32(accu);
        *s.add(row as usize) = sumi as f32;
    }
}

/// Computes GEMV Matrix-Vector product using BitNet's 1x1 single-row baseline NEON kernel (`vmlal_s8`).
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_mad_1x1_neon(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 64 (QK_I2_S)");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let n = cols as i32;
    let nrc = rows as i32;
    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let nb = n / (QK_I2_S as i32);
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let groupla_num = if nb % 32 != 0 { 1 } else { 0 };

    let mask = vdupq_n_u8(0x03);
    let bytes_per_row = cols / 4;

    for row in 0..nrc {
        let mut accu = vdupq_n_s32(0);
        let x_row = x.add(row as usize * bytes_per_row);

        for i in 0..group32_num {
            let mut px = x_row.add((i as usize) * 32 * 16);
            let mut accu32 = vdupq_n_s16(0);

            for j in 0..32 {
                let py = y.add((i as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                let xq8_3 = vld1q_u8(px);
                let xq8_2 = vshrq_n_u8(xq8_3, 2);
                let xq8_1 = vshrq_n_u8(xq8_3, 4);
                let xq8_0 = vshrq_n_u8(xq8_3, 6);

                let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                accu32 = vmlal_s8(accu32, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_3), vget_high_s8(yq8_3));

                px = px.add(16);
            }

            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu32));
        }

        for _i in 0..groupla_num {
            let mut px = x_row.add((group32_num as usize) * 32 * 16);
            let mut accula = vdupq_n_s16(0);

            for j in 0..la_num {
                let py = y.add((group32_num as usize) * 32 * 64 + (j as usize) * 64);
                let yq8_0 = vld1q_s8(py);
                let yq8_1 = vld1q_s8(py.add(16));
                let yq8_2 = vld1q_s8(py.add(32));
                let yq8_3 = vld1q_s8(py.add(48));

                let xq8_3 = vld1q_u8(px);
                let xq8_2 = vshrq_n_u8(xq8_3, 2);
                let xq8_1 = vshrq_n_u8(xq8_3, 4);
                let xq8_0 = vshrq_n_u8(xq8_3, 6);

                let q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                let q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                let q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                let q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

                accula = vmlal_s8(accula, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accula = vmlal_s8(accula, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accula = vmlal_s8(accula, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accula = vmlal_s8(accula, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accula = vmlal_s8(accula, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accula = vmlal_s8(accula, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accula = vmlal_s8(accula, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accula = vmlal_s8(accula, vget_high_s8(q8_3), vget_high_s8(yq8_3));

                px = px.add(16);
            }

            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula)));
            accu = vaddq_s32(accu, vmovl_high_s16(accula));
        }

        let sumi = vaddlvq_s32(accu);
        *s.add(row as usize) = sumi as f32;
    }
}

/// Computes GEMV Matrix-Vector product on AArch64 with automatic handling of remainder rows and DotProd detection.
#[cfg(target_arch = "aarch64")]
pub fn gemv_mad_neon(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let full_tiles = (rows / PARALLEL_SIZE) * PARALLEL_SIZE;
    let bytes_per_row = cols / 4;
    let has_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");

    if full_tiles > 0 {
        unsafe {
            if has_dotprod {
                gemv_mad_1xn_neon_dotprod(
                    &packed_weights[..full_tiles * bytes_per_row],
                    activations,
                    full_tiles,
                    cols,
                    &mut out[..full_tiles],
                );
            } else {
                gemv_mad_1xn_neon(
                    &packed_weights[..full_tiles * bytes_per_row],
                    activations,
                    full_tiles,
                    cols,
                    &mut out[..full_tiles],
                );
            }
        }
    }

    if full_tiles < rows {
        let rem_rows = rows - full_tiles;
        unsafe {
            if has_dotprod {
                gemv_mad_1x1_neon_dotprod(
                    &packed_weights[full_tiles * bytes_per_row..],
                    activations,
                    rem_rows,
                    cols,
                    &mut out[full_tiles..],
                );
            } else {
                gemv_mad_1x1_neon(
                    &packed_weights[full_tiles * bytes_per_row..],
                    activations,
                    rem_rows,
                    cols,
                    &mut out[full_tiles..],
                );
            }
        }
    }
}

/// Automatically dispatches GEMV to AVX2/NEON MAD kernel if supported, otherwise uses scalar fallback.
pub fn gemv_mad(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            gemv_mad_avx2(packed_weights, activations, rows, cols, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        gemv_mad_neon(packed_weights, activations, rows, cols, out);
        return;
    }
    #[allow(unreachable_code)]
    gemv_mad_scalar(packed_weights, activations, rows, cols, out);
}

/// Multi-threaded GEMV Matrix-Vector product using BitNet MAD kernels.
pub fn gemv_mad_parallel(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    if num_threads <= 1 || rows < num_threads * PARALLEL_SIZE {
        gemv_mad(packed_weights, activations, rows, cols, out);
        return;
    }

    let bytes_per_row = cols / 4;
    let rows_per_thread = ((rows / num_threads + 3) / PARALLEL_SIZE) * PARALLEL_SIZE;

    use rayon::prelude::*;
    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..rows]
            .par_chunks_mut(rows_per_thread)
            .zip(packed_weights.par_chunks(rows_per_thread * bytes_per_row))
            .for_each(|(out_chunk, weights_chunk)| {
                let chunk_rows = out_chunk.len();
                gemv_mad(weights_chunk, activations, chunk_rows, cols, out_chunk);
            });
    });
}
