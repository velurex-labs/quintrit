#![allow(unsafe_op_in_unsafe_fn)]

//! Exact 1:1 Port of Microsoft BitNet Multiply-Add (MAD) GEMV Kernels to AVX-512 (512-bit ZMM).
//!
//! Provides the Tier 2 Algorithmic Parity baseline for 4 trits per byte (2.00 bpw)
//! on 512-bit vector architectures (AMD Zen 4/5, Intel Xeon).
//!
//! ## Architectural Principles:
//! 1. **512-bit Vector Widening**: Evaluates 256 trits (two 128-trit blocks) per column beat
//!    in a single 64-byte `_mm512_loadu_si512`.
//! 2. **8-Row Tiled Parallelism (`PARALLEL_SIZE = 8`)**: Evaluates 8 output rows concurrently,
//!    reusing 4 activation ZMM registers across all 8 rows to saturate memory load pipelines.
//! 3. **Register-File Residency**: The 8 accumulators, 4 activation registers, and temporary
//!    vectors fit cleanly within the 32 architectural ZMM registers (`zmm0`–`zmm31`) without stack spilling.
//! 4. **Identical BitNet Planar Bit-Slicing**: Uses the identical shift (2, 4, 6), mask (`0x03`),
//!    and `maddubs_epi16` arithmetic to ensure 100% numerical parity with BitNet AVX2.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub const QK_I2_S: usize = 128;
pub const BLOCK_BYTES: usize = 32;
pub const PARALLEL_SIZE_AVX512: usize = 8;

/// Horizontally adds 16 signed 32-bit integers in a 512-bit ZMM register.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn hsum_i32_avx512(a: __m512i) -> i64 {
    _mm512_reduce_add_epi32(a) as i64
}

/// Computes GEMV Matrix-Vector product using 8-row tiled BitNet planar MAD on AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemv_mad_1xn_avx512(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 128 (QK_I2_S)");
    assert_eq!(rows % PARALLEL_SIZE_AVX512, 0, "rows must be a multiple of 8 for 1x8 tiling");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let bytes_per_row = cols / 4;
    let nb = cols / QK_I2_S; // Number of 128-trit (32-byte) blocks
    let num_pairs = nb / 2; // Number of 256-trit (64-byte) pairs
    let has_remainder_block = (nb % 2) != 0;

    let mask = _mm512_set1_epi8(0x03);
    let one16 = _mm512_set1_epi16(1);

    for row in (0..rows).step_by(PARALLEL_SIZE_AVX512) {
        let mut accu = [_mm512_setzero_si512(); PARALLEL_SIZE_AVX512];
        let mut x_row = [core::ptr::null::<u8>(); PARALLEL_SIZE_AVX512];
        for rb in 0..PARALLEL_SIZE_AVX512 {
            x_row[rb] = x.add((row + rb) * bytes_per_row);
        }

        // Process in groups of up to 16 pairs (16 * 256 = 4096 trits) before flushing 16-bit to 32-bit
        let mut p_idx = 0;
        while p_idx < num_pairs {
            let chunk_pairs = usize::min(16, num_pairs - p_idx);
            let mut accu16 = [_mm512_setzero_si512(); PARALLEL_SIZE_AVX512];

            for j in 0..chunk_pairs {
                let pair_col = (p_idx + j) * 256;
                let py = y.add(pair_col);

                // Load 2 blocks of activations: block 0 (0..127) and block 1 (128..255)
                let y0_lo = _mm256_loadu_si256(py as *const __m256i);
                let y0_hi = _mm256_loadu_si256(py.add(128) as *const __m256i);
                let yq8_0 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y0_lo), y0_hi);

                let y1_lo = _mm256_loadu_si256(py.add(32) as *const __m256i);
                let y1_hi = _mm256_loadu_si256(py.add(160) as *const __m256i);
                let yq8_1 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y1_lo), y1_hi);

                let y2_lo = _mm256_loadu_si256(py.add(64) as *const __m256i);
                let y2_hi = _mm256_loadu_si256(py.add(192) as *const __m256i);
                let yq8_2 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y2_lo), y2_hi);

                let y3_lo = _mm256_loadu_si256(py.add(96) as *const __m256i);
                let y3_hi = _mm256_loadu_si256(py.add(224) as *const __m256i);
                let yq8_3 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y3_lo), y3_hi);

                let pair_byte_offset = (p_idx + j) * 64;

                for rb in 0..PARALLEL_SIZE_AVX512 {
                    let px = x_row[rb].add(pair_byte_offset);
                    let xq8_raw = _mm512_loadu_si512(px as *const __m512i);

                    let xq8_2 = _mm512_srli_epi16(xq8_raw, 2);
                    let xq8_1 = _mm512_srli_epi16(xq8_raw, 4);
                    let xq8_0 = _mm512_srli_epi16(xq8_raw, 6);

                    let q8_3 = _mm512_and_si512(xq8_raw, mask);
                    let q8_2 = _mm512_and_si512(xq8_2, mask);
                    let q8_1 = _mm512_and_si512(xq8_1, mask);
                    let q8_0 = _mm512_and_si512(xq8_0, mask);

                    let p0 = _mm512_maddubs_epi16(q8_0, yq8_0);
                    let p1 = _mm512_maddubs_epi16(q8_1, yq8_1);
                    let p2 = _mm512_maddubs_epi16(q8_2, yq8_2);
                    let p3 = _mm512_maddubs_epi16(q8_3, yq8_3);

                    let sum_01 = _mm512_add_epi16(p0, p1);
                    let sum_23 = _mm512_add_epi16(p2, p3);
                    accu16[rb] = _mm512_add_epi16(accu16[rb], _mm512_add_epi16(sum_01, sum_23));
                }
            }

            for rb in 0..PARALLEL_SIZE_AVX512 {
                accu[rb] = _mm512_add_epi32(accu[rb], _mm512_madd_epi16(accu16[rb], one16));
            }

            p_idx += chunk_pairs;
        }

        // Remainder block: if cols has an odd number of 128-trit blocks
        if has_remainder_block {
            let rem_col = num_pairs * 256;
            let py = y.add(rem_col);

            let y0_lo = _mm256_loadu_si256(py as *const __m256i);
            let yq8_0 = _mm512_castsi256_si512(y0_lo);
            let y1_lo = _mm256_loadu_si256(py.add(32) as *const __m256i);
            let yq8_1 = _mm512_castsi256_si512(y1_lo);
            let y2_lo = _mm256_loadu_si256(py.add(64) as *const __m256i);
            let yq8_2 = _mm512_castsi256_si512(y2_lo);
            let y3_lo = _mm256_loadu_si256(py.add(96) as *const __m256i);
            let yq8_3 = _mm512_castsi256_si512(y3_lo);

            let rem_byte_offset = num_pairs * 64;

            for rb in 0..PARALLEL_SIZE_AVX512 {
                let px = x_row[rb].add(rem_byte_offset);
                let xq8_raw_lo = _mm256_loadu_si256(px as *const __m256i);
                let xq8_raw = _mm512_castsi256_si512(xq8_raw_lo);

                let xq8_2 = _mm512_srli_epi16(xq8_raw, 2);
                let xq8_1 = _mm512_srli_epi16(xq8_raw, 4);
                let xq8_0 = _mm512_srli_epi16(xq8_raw, 6);

                let q8_3 = _mm512_and_si512(xq8_raw, mask);
                let q8_2 = _mm512_and_si512(xq8_2, mask);
                let q8_1 = _mm512_and_si512(xq8_1, mask);
                let q8_0 = _mm512_and_si512(xq8_0, mask);

                let p0 = _mm512_maddubs_epi16(q8_0, yq8_0);
                let p1 = _mm512_maddubs_epi16(q8_1, yq8_1);
                let p2 = _mm512_maddubs_epi16(q8_2, yq8_2);
                let p3 = _mm512_maddubs_epi16(q8_3, yq8_3);

                let sum_01 = _mm512_add_epi16(p0, p1);
                let sum_23 = _mm512_add_epi16(p2, p3);
                let sum_all = _mm512_add_epi16(sum_01, sum_23);

                accu[rb] = _mm512_add_epi32(accu[rb], _mm512_madd_epi16(sum_all, one16));
            }
        }

        for rb in 0..PARALLEL_SIZE_AVX512 {
            let sumi = hsum_i32_avx512(accu[rb]);
            *s.add(row + rb) = sumi as f32;
        }
    }
}

/// Computes GEMV Matrix-Vector product using 1-row single-row BitNet planar MAD on AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn gemv_mad_1x1_avx512(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(cols % QK_I2_S, 0, "cols must be a multiple of 128 (QK_I2_S)");
    assert_eq!(packed_weights.len(), (rows * cols) / 4);
    assert!(activations.len() >= cols);
    assert!(out.len() >= rows);

    let x = packed_weights.as_ptr();
    let y = activations.as_ptr();
    let s = out.as_mut_ptr();

    let bytes_per_row = cols / 4;
    let nb = cols / QK_I2_S;
    let num_pairs = nb / 2;
    let has_remainder_block = (nb % 2) != 0;

    let mask = _mm512_set1_epi8(0x03);
    let one16 = _mm512_set1_epi16(1);

    for row in 0..rows {
        let mut accu = _mm512_setzero_si512();
        let x_row = x.add(row * bytes_per_row);

        let mut p_idx = 0;
        while p_idx < num_pairs {
            let chunk_pairs = usize::min(16, num_pairs - p_idx);
            let mut accu16 = _mm512_setzero_si512();

            for j in 0..chunk_pairs {
                let pair_col = (p_idx + j) * 256;
                let py = y.add(pair_col);

                let y0_lo = _mm256_loadu_si256(py as *const __m256i);
                let y0_hi = _mm256_loadu_si256(py.add(128) as *const __m256i);
                let yq8_0 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y0_lo), y0_hi);

                let y1_lo = _mm256_loadu_si256(py.add(32) as *const __m256i);
                let y1_hi = _mm256_loadu_si256(py.add(160) as *const __m256i);
                let yq8_1 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y1_lo), y1_hi);

                let y2_lo = _mm256_loadu_si256(py.add(64) as *const __m256i);
                let y2_hi = _mm256_loadu_si256(py.add(192) as *const __m256i);
                let yq8_2 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y2_lo), y2_hi);

                let y3_lo = _mm256_loadu_si256(py.add(96) as *const __m256i);
                let y3_hi = _mm256_loadu_si256(py.add(224) as *const __m256i);
                let yq8_3 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(y3_lo), y3_hi);

                let pair_byte_offset = (p_idx + j) * 64;
                let px = x_row.add(pair_byte_offset);
                let xq8_raw = _mm512_loadu_si512(px as *const __m512i);

                let xq8_2 = _mm512_srli_epi16(xq8_raw, 2);
                let xq8_1 = _mm512_srli_epi16(xq8_raw, 4);
                let xq8_0 = _mm512_srli_epi16(xq8_raw, 6);

                let q8_3 = _mm512_and_si512(xq8_raw, mask);
                let q8_2 = _mm512_and_si512(xq8_2, mask);
                let q8_1 = _mm512_and_si512(xq8_1, mask);
                let q8_0 = _mm512_and_si512(xq8_0, mask);

                let p0 = _mm512_maddubs_epi16(q8_0, yq8_0);
                let p1 = _mm512_maddubs_epi16(q8_1, yq8_1);
                let p2 = _mm512_maddubs_epi16(q8_2, yq8_2);
                let p3 = _mm512_maddubs_epi16(q8_3, yq8_3);

                let sum_01 = _mm512_add_epi16(p0, p1);
                let sum_23 = _mm512_add_epi16(p2, p3);
                accu16 = _mm512_add_epi16(accu16, _mm512_add_epi16(sum_01, sum_23));
            }

            accu = _mm512_add_epi32(accu, _mm512_madd_epi16(accu16, one16));
            p_idx += chunk_pairs;
        }

        if has_remainder_block {
            let rem_col = num_pairs * 256;
            let py = y.add(rem_col);

            let y0_lo = _mm256_loadu_si256(py as *const __m256i);
            let yq8_0 = _mm512_castsi256_si512(y0_lo);
            let y1_lo = _mm256_loadu_si256(py.add(32) as *const __m256i);
            let yq8_1 = _mm512_castsi256_si512(y1_lo);
            let y2_lo = _mm256_loadu_si256(py.add(64) as *const __m256i);
            let yq8_2 = _mm512_castsi256_si512(y2_lo);
            let y3_lo = _mm256_loadu_si256(py.add(96) as *const __m256i);
            let yq8_3 = _mm512_castsi256_si512(y3_lo);

            let rem_byte_offset = num_pairs * 64;
            let px = x_row.add(rem_byte_offset);
            let xq8_raw_lo = _mm256_loadu_si256(px as *const __m256i);
            let xq8_raw = _mm512_castsi256_si512(xq8_raw_lo);

            let xq8_2 = _mm512_srli_epi16(xq8_raw, 2);
            let xq8_1 = _mm512_srli_epi16(xq8_raw, 4);
            let xq8_0 = _mm512_srli_epi16(xq8_raw, 6);

            let q8_3 = _mm512_and_si512(xq8_raw, mask);
            let q8_2 = _mm512_and_si512(xq8_2, mask);
            let q8_1 = _mm512_and_si512(xq8_1, mask);
            let q8_0 = _mm512_and_si512(xq8_0, mask);

            let p0 = _mm512_maddubs_epi16(q8_0, yq8_0);
            let p1 = _mm512_maddubs_epi16(q8_1, yq8_1);
            let p2 = _mm512_maddubs_epi16(q8_2, yq8_2);
            let p3 = _mm512_maddubs_epi16(q8_3, yq8_3);

            let sum_01 = _mm512_add_epi16(p0, p1);
            let sum_23 = _mm512_add_epi16(p2, p3);
            let sum_all = _mm512_add_epi16(sum_01, sum_23);

            accu = _mm512_add_epi32(accu, _mm512_madd_epi16(sum_all, one16));
        }

        let sumi = hsum_i32_avx512(accu);
        *s.add(row) = sumi as f32;
    }
}

/// Computes GEMV Matrix-Vector product using BitNet MAD kernels on AVX-512 (1x8 tiled + 1x1 remainder).
#[cfg(target_arch = "x86_64")]
pub fn gemv_mad_avx512(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let full_tiles = (rows / PARALLEL_SIZE_AVX512) * PARALLEL_SIZE_AVX512;
    let bytes_per_row = cols / 4;

    if full_tiles > 0 {
        unsafe {
            gemv_mad_1xn_avx512(
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
            gemv_mad_1x1_avx512(
                &packed_weights[full_tiles * bytes_per_row..],
                activations,
                rem_rows,
                cols,
                &mut out[full_tiles..],
            );
        }
    }
}

/// Multi-threaded GEMV Matrix-Vector product using BitNet MAD kernels on AVX-512.
#[cfg(target_arch = "x86_64")]
pub fn gemv_mad_parallel_avx512(
    packed_weights: &[u8],
    activations: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    if num_threads <= 1 || rows < num_threads * PARALLEL_SIZE_AVX512 {
        gemv_mad_avx512(packed_weights, activations, rows, cols, out);
        return;
    }

    let bytes_per_row = cols / 4;
    let rows_per_thread = ((rows / num_threads + PARALLEL_SIZE_AVX512 - 1) / PARALLEL_SIZE_AVX512) * PARALLEL_SIZE_AVX512;

    use rayon::prelude::*;
    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..rows]
            .par_chunks_mut(rows_per_thread)
            .zip(packed_weights.par_chunks(rows_per_thread * bytes_per_row))
            .for_each(|(out_chunk, weights_chunk)| {
                let chunk_rows = out_chunk.len();
                gemv_mad_avx512(weights_chunk, activations, chunk_rows, cols, out_chunk);
            });
    });
}
