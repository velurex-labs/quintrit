#![allow(unsafe_op_in_unsafe_fn)]

//! # 64-Row Tiled Fused AVX-512 VBMI & VNNI Activation-LUT GEMV Kernels
//!
//! Evaluates a 64-row block of weights against the precomputed QLUT on x86_64
//! architectures with AVX-512 extensions (AMD Zen 4/5, Intel Ice Lake / Sapphire Rapids).
//!
//! ## Architectural Principles
//!
//! 1. **64-Row Tiled Parallelism (BM = 64):**
//!    Matrix weights are organized in blocks of 64 rows. In the inner loop,
//!    one AVX-512 vector load `_mm512_loadu_si512` loads the weights for **64 distinct rows**
//!    at the same column group. The lookup and summation are executed across all 64 rows
//!    simultaneously with zero horizontal additions.
//!
//! 2. **Single-Instruction Cross-Lane Byte Permutation via VBMI (`vpermb`):**
//!    - Upper 2 trits (d3, d4): 4-bit quotient `h34 in 0..8` indexes 64-byte `tbl_34` via
//!      `_mm512_permutexvar_epi8(h34, zmm_tbl_34)`.
//!    - Lower 3 trits (d0, d1, d2): remainder `r3 = x - 28 * h34 in 0..26` indexes 64-byte `tbl_012`
//!      via `_mm512_permutexvar_epi8(r3, zmm_tbl_012)`.
//!    - Eliminates pure-ALU interval classification and vpsignb arithmetic.
//!
//! 3. **Accumulation Pipelines:**
//!    - **VBMI Baseline:** 2-column unrolled pairwise reduction via `_mm512_maddubs_epi16` and
//!      `_mm512_madd_epi16` into 32-bit accumulators.
//!    - **VNNI Acceleration:** Direct 32-bit hardware dot-product accumulation via `_mm512_dpbusd_epi32`.

use core::arch::x86_64::*;
use crate::pack::{tile_byte, tile_row_bytes, untile_row, BLOCK_BYTES_5, QK_5};
use crate::lut::{check_gemv_buffers, MAP_34, MAP_012};

/// Whether the CPU supports the AVX-512 subset every 64-row kernel needs (F, BW and VBMI).
pub fn avx512_supported() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")
}

/// Panics unless the CPU can run the AVX-512 kernels. Executing them on a CPU without these extensions is undefined
/// behaviour, so every safe entry point checks this before it dispatches.
#[track_caller]
fn require_avx512() {
    assert!(avx512_supported(), "the AVX-512 kernels need AVX-512 F, BW and VBMI, which this CPU does not support");
}

/// Length in bytes of the table produced by [`build_activation_lut_avx512`]: 128 bytes per tile column.
pub fn activation_lut_len_avx512(cols: usize) -> usize {
    tile_row_bytes(cols) * 128
}

/// Transforms weights from standard 5-planar layout into 64-row tiled format for AVX-512 GEMV.
///
/// Each block of 64 rows is stored consecutively. For each column byte `c`,
/// the 64 bytes for rows `0..63` are laid out contiguously in memory.
pub fn transform_weights_lut_64(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(rows % 64, 0, "rows must be a multiple of 64 for 64-row tiling");
    let src_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 64;

    let mut dst = vec![0u8; num_blocks * 64 * bytes_per_row];

    for b in 0..num_blocks {
        let r_base = b * 64;
        let block_dst = &mut dst[b * (64 * bytes_per_row)..(b + 1) * (64 * bytes_per_row)];

        for c in 0..bytes_per_row {
            let col_dst = &mut block_dst[c * 64..(c + 1) * 64];
            for r in 0..64 {
                let row = &src[(r_base + r) * src_row_bytes..(r_base + r + 1) * src_row_bytes];
                col_dst[r] = tile_byte(row, cols, c);
            }
        }
    }

    dst
}

/// Compatibility alias for `transform_weights_lut_64`.
#[inline(always)]
pub fn transform_weights_quintrit_lut_64(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    transform_weights_lut_64(src, rows, cols)
}

/// Restores standard 5-planar layout from 64-row tiled format.
pub fn untransform_weights_lut_64(tiled: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(rows % 64, 0, "rows must be a multiple of 64 for 64-row tiling");
    let dst_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 64;

    let mut dst = vec![0u8; rows * dst_row_bytes];
    let mut tile_row = vec![0u8; bytes_per_row];

    for b in 0..num_blocks {
        let r_base = b * 64;
        let block_src = &tiled[b * (64 * bytes_per_row)..(b + 1) * (64 * bytes_per_row)];

        for r in 0..64 {
            for c in 0..bytes_per_row {
                tile_row[c] = block_src[c * 64 + r];
            }
            let row_dst = &mut dst[(r_base + r) * dst_row_bytes..(r_base + r + 1) * dst_row_bytes];
            untile_row(&tile_row, cols, row_dst);
        }
    }

    dst
}

/// Compatibility alias for `untransform_weights_lut_64`.
#[inline(always)]
pub fn untransform_weights_quintrit_lut_64(tiled: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    untransform_weights_lut_64(tiled, rows, cols)
}

/// Precomputes the AVX-512 Activation Lookup Table (QLUT) for all column groups.
/// Stores 128 bytes per column: 64 bytes for `tbl_34` followed by 64 bytes for `tbl_012`.
pub fn build_activation_lut_avx512(activations: &[i8], cols: usize) -> Vec<i8> {
    let bytes_per_row = tile_row_bytes(cols);
    let mut qlut = vec![0i8; bytes_per_row * 128];

    for c in 0..bytes_per_row {

        let get_act = |p: usize| -> i16 {
            let idx = crate::pack::column_activation_index(cols, c, p);
            if idx < cols {
                activations[idx] as i16
            } else {
                0
            }
        };

        let a0 = get_act(0);
        let a1 = get_act(1);
        let a2 = get_act(2);
        let a3 = get_act(3);
        let a4 = get_act(4);

        let base = c * 128;

        // 1. Precompute Table A (Upper Trits d3, d4 via h34 in 0..8)
        let mut t34 = [0i8; 64];
        for s in 0..16usize {
            let (t3, t4) = MAP_34[s];
            let val = t3 as i16 * a3 + t4 as i16 * a4;
            t34[s] = val.clamp(-128, 127) as i8;
        }
        qlut[base..base + 64].copy_from_slice(&t34);

        // 2. Precompute Table B (Lower Trio d0, d1, d2 via r3 in 0..26)
        let mut t012 = [0i8; 64];
        for s in 0..32usize {
            let (t0, t1, t2) = MAP_012[s];
            let val = t0 as i16 * a0 + t1 as i16 * a1 + t2 as i16 * a2;
            t012[s] = val.clamp(-128, 127) as i8;
        }
        qlut[base + 64..base + 128].copy_from_slice(&t012);
    }

    qlut
}

/// Compatibility alias for `build_activation_lut_avx512`.
#[inline(always)]
pub fn build_quintrit_lut_avx512(activations: &[i8], cols: usize) -> Vec<i8> {
    build_activation_lut_avx512(activations, cols)
}

/// Decomposes 64 packed 5-trit bytes into quotient `h34 in 0..8` and remainder `r3 in 0..26`.
#[inline(always)]
pub unsafe fn decompose_quintrit_avx512(vec_w: __m512i) -> (__m512i, __m512i) {
    let c_3f = _mm512_set1_epi8(0x3F);
    let c_1f = _mm512_set1_epi8(0x1F);
    let c_0f = _mm512_set1_epi8(0x0F);

    let v11 = _mm512_and_si512(_mm512_srli_epi16(vec_w, 2), c_3f);
    let v15 = _mm512_and_si512(_mm512_srli_epi16(v11, 3), c_1f);
    let v17 = _mm512_avg_epu8(v11, v15);
    let h34 = _mm512_and_si512(_mm512_srli_epi16(v17, 2), c_0f);

    // Exact remainder calculation: r3 = vec_w - 28 * h34
    // 28 * h34 = 4 * (8 * h34 - h34)
    let h34_x8 = _mm512_slli_epi16(h34, 3);
    let h34_x7 = _mm512_sub_epi8(h34_x8, h34);
    let h34_x28 = _mm512_slli_epi16(h34_x7, 2);
    let r3 = _mm512_sub_epi8(vec_w, h34_x28);

    (h34, r3)
}

/// Evaluates a 64-row block of weights against precomputed AVX-512 QLUT using VBMI (`vpermb`)
/// with 16-bit intermediate `vpmaddubsw` reduction.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn gemv_quintrit_lut_block64_avx512_vbmi(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    let c_ones = _mm512_set1_epi8(1);

    // 4 accumulators covering rows 0..63 in groups of 16
    let mut acc0 = _mm512_setzero_si512(); // Rows 0..15
    let mut acc1 = _mm512_setzero_si512(); // Rows 16..31
    let mut acc2 = _mm512_setzero_si512(); // Rows 32..47
    let mut acc3 = _mm512_setzero_si512(); // Rows 48..63

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(16);
        let mut sum_lo = _mm512_setzero_si512();
        let mut sum_hi = _mm512_setzero_si512();

        let mut ci = 0;
        while ci + 1 < chunk_len {
            let cur_c0 = c + ci;
            let cur_c1 = c + ci + 1;

            // Column 0
            let vec_w0 = _mm512_loadu_si512(block_weights.add(cur_c0 * 64) as *const __m512i);
            let tbl_34_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128) as *const __m512i);
            let tbl_012_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128 + 64) as *const __m512i);
            let (h34_0, r3_0) = decompose_quintrit_avx512(vec_w0);
            let p34_0 = _mm512_permutexvar_epi8(h34_0, tbl_34_0);
            let p012_0 = _mm512_permutexvar_epi8(r3_0, tbl_012_0);

            // Column 1
            let vec_w1 = _mm512_loadu_si512(block_weights.add(cur_c1 * 64) as *const __m512i);
            let tbl_34_1 = _mm512_loadu_si512(qlut.add(cur_c1 * 128) as *const __m512i);
            let tbl_012_1 = _mm512_loadu_si512(qlut.add(cur_c1 * 128 + 64) as *const __m512i);
            let (h34_1, r3_1) = decompose_quintrit_avx512(vec_w1);
            let p34_1 = _mm512_permutexvar_epi8(h34_1, tbl_34_1);
            let p012_1 = _mm512_permutexvar_epi8(r3_1, tbl_012_1);

            // Pairwise 16-bit reduction
            let unpck_34_lo = _mm512_unpacklo_epi8(p34_0, p34_1);
            let unpck_34_hi = _mm512_unpackhi_epi8(p34_0, p34_1);
            let madd_34_lo = _mm512_maddubs_epi16(c_ones, unpck_34_lo);
            let madd_34_hi = _mm512_maddubs_epi16(c_ones, unpck_34_hi);

            let unpck_012_lo = _mm512_unpacklo_epi8(p012_0, p012_1);
            let unpck_012_hi = _mm512_unpackhi_epi8(p012_0, p012_1);
            let madd_012_lo = _mm512_maddubs_epi16(c_ones, unpck_012_lo);
            let madd_012_hi = _mm512_maddubs_epi16(c_ones, unpck_012_hi);

            sum_lo = _mm512_add_epi16(sum_lo, _mm512_add_epi16(madd_34_lo, madd_012_lo));
            sum_hi = _mm512_add_epi16(sum_hi, _mm512_add_epi16(madd_34_hi, madd_012_hi));

            ci += 2;
        }

        // Remainder column (if chunk_len is odd)
        if ci < chunk_len {
            let cur_c0 = c + ci;
            let vec_w0 = _mm512_loadu_si512(block_weights.add(cur_c0 * 64) as *const __m512i);
            let tbl_34_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128) as *const __m512i);
            let tbl_012_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128 + 64) as *const __m512i);
            let (h34_0, r3_0) = decompose_quintrit_avx512(vec_w0);
            let p34_0 = _mm512_permutexvar_epi8(h34_0, tbl_34_0);
            let p012_0 = _mm512_permutexvar_epi8(r3_0, tbl_012_0);

            let zero = _mm512_setzero_si512();
            let unpck_34_lo = _mm512_unpacklo_epi8(p34_0, zero);
            let unpck_34_hi = _mm512_unpackhi_epi8(p34_0, zero);
            let madd_34_lo = _mm512_maddubs_epi16(c_ones, unpck_34_lo);
            let madd_34_hi = _mm512_maddubs_epi16(c_ones, unpck_34_hi);

            let unpck_012_lo = _mm512_unpacklo_epi8(p012_0, zero);
            let unpck_012_hi = _mm512_unpackhi_epi8(p012_0, zero);
            let madd_012_lo = _mm512_maddubs_epi16(c_ones, unpck_012_lo);
            let madd_012_hi = _mm512_maddubs_epi16(c_ones, unpck_012_hi);

            sum_lo = _mm512_add_epi16(sum_lo, _mm512_add_epi16(madd_34_lo, madd_012_lo));
            sum_hi = _mm512_add_epi16(sum_hi, _mm512_add_epi16(madd_34_hi, madd_012_hi));
        }

        let dw0_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_lo));
        let dw0_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(sum_lo));
        let dw1_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_hi));
        let dw1_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(sum_hi));

        acc0 = _mm512_add_epi32(acc0, dw0_lo);
        acc1 = _mm512_add_epi32(acc1, dw1_lo);
        acc2 = _mm512_add_epi32(acc2, dw0_hi);
        acc3 = _mm512_add_epi32(acc3, dw1_hi);

        c += chunk_len;
    }

    // Recover linear row order 0..63 and store to out_64
    let mut temp = [0i32; 64];
    _mm512_storeu_si512(temp.as_mut_ptr() as *mut __m512i, acc0);
    _mm512_storeu_si512(temp.as_mut_ptr().add(16) as *mut __m512i, acc1);
    _mm512_storeu_si512(temp.as_mut_ptr().add(32) as *mut __m512i, acc2);
    _mm512_storeu_si512(temp.as_mut_ptr().add(48) as *mut __m512i, acc3);

    for i in 0..8 {
        *out_64.add(i) = temp[i] as f32;
        *out_64.add(8 + i) = temp[16 + i] as f32;
        *out_64.add(16 + i) = temp[8 + i] as f32;
        *out_64.add(24 + i) = temp[24 + i] as f32;
        *out_64.add(32 + i) = temp[32 + i] as f32;
        *out_64.add(40 + i) = temp[48 + i] as f32;
        *out_64.add(48 + i) = temp[40 + i] as f32;
        *out_64.add(56 + i) = temp[56 + i] as f32;
    }
}

/// Evaluates a 64-row block of weights against precomputed AVX-512 QLUT using VBMI + VNNI (`vpdpbusd`)
/// with direct 32-bit hardware accumulation.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
pub unsafe fn gemv_quintrit_lut_block64_avx512_vnni(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    let c_ones = _mm512_set1_epi8(1);

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let mut c = 0;
    while c + 1 < bytes_per_row {
        let cur_c0 = c;
        let cur_c1 = c + 1;

        // Column 0
        let vec_w0 = _mm512_loadu_si512(block_weights.add(cur_c0 * 64) as *const __m512i);
        let tbl_34_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128) as *const __m512i);
        let tbl_012_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128 + 64) as *const __m512i);
        let (h34_0, r3_0) = decompose_quintrit_avx512(vec_w0);
        let p34_0 = _mm512_permutexvar_epi8(h34_0, tbl_34_0);
        let p012_0 = _mm512_permutexvar_epi8(r3_0, tbl_012_0);

        // Column 1
        let vec_w1 = _mm512_loadu_si512(block_weights.add(cur_c1 * 64) as *const __m512i);
        let tbl_34_1 = _mm512_loadu_si512(qlut.add(cur_c1 * 128) as *const __m512i);
        let tbl_012_1 = _mm512_loadu_si512(qlut.add(cur_c1 * 128 + 64) as *const __m512i);
        let (h34_1, r3_1) = decompose_quintrit_avx512(vec_w1);
        let p34_1 = _mm512_permutexvar_epi8(h34_1, tbl_34_1);
        let p012_1 = _mm512_permutexvar_epi8(r3_1, tbl_012_1);

        let p_lo = _mm512_unpacklo_epi8(p34_0, p012_0);
        let p_hi = _mm512_unpackhi_epi8(p34_0, p012_0);
        let q_lo = _mm512_unpacklo_epi8(p34_1, p012_1);
        let q_hi = _mm512_unpackhi_epi8(p34_1, p012_1);

        let dw0 = _mm512_unpacklo_epi16(p_lo, q_lo);
        let dw1 = _mm512_unpackhi_epi16(p_lo, q_lo);
        let dw2 = _mm512_unpacklo_epi16(p_hi, q_hi);
        let dw3 = _mm512_unpackhi_epi16(p_hi, q_hi);

        acc0 = _mm512_dpbusd_epi32(acc0, c_ones, dw0);
        acc1 = _mm512_dpbusd_epi32(acc1, c_ones, dw1);
        acc2 = _mm512_dpbusd_epi32(acc2, c_ones, dw2);
        acc3 = _mm512_dpbusd_epi32(acc3, c_ones, dw3);

        c += 2;
    }

    if c < bytes_per_row {
        let cur_c0 = c;
        let vec_w0 = _mm512_loadu_si512(block_weights.add(cur_c0 * 64) as *const __m512i);
        let tbl_34_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128) as *const __m512i);
        let tbl_012_0 = _mm512_loadu_si512(qlut.add(cur_c0 * 128 + 64) as *const __m512i);
        let (h34_0, r3_0) = decompose_quintrit_avx512(vec_w0);
        let p34_0 = _mm512_permutexvar_epi8(h34_0, tbl_34_0);
        let p012_0 = _mm512_permutexvar_epi8(r3_0, tbl_012_0);

        let zero = _mm512_setzero_si512();
        let p_lo = _mm512_unpacklo_epi8(p34_0, p012_0);
        let p_hi = _mm512_unpackhi_epi8(p34_0, p012_0);

        let dw0 = _mm512_unpacklo_epi16(p_lo, zero);
        let dw1 = _mm512_unpackhi_epi16(p_lo, zero);
        let dw2 = _mm512_unpacklo_epi16(p_hi, zero);
        let dw3 = _mm512_unpackhi_epi16(p_hi, zero);

        acc0 = _mm512_dpbusd_epi32(acc0, c_ones, dw0);
        acc1 = _mm512_dpbusd_epi32(acc1, c_ones, dw1);
        acc2 = _mm512_dpbusd_epi32(acc2, c_ones, dw2);
        acc3 = _mm512_dpbusd_epi32(acc3, c_ones, dw3);
    }

    let mut temp = [0i32; 64];
    _mm512_storeu_si512(temp.as_mut_ptr() as *mut __m512i, acc0);
    _mm512_storeu_si512(temp.as_mut_ptr().add(16) as *mut __m512i, acc1);
    _mm512_storeu_si512(temp.as_mut_ptr().add(32) as *mut __m512i, acc2);
    _mm512_storeu_si512(temp.as_mut_ptr().add(48) as *mut __m512i, acc3);

    for lane in 0..4 {
        let base_row = lane * 16;
        for g in 0..4 {
            let src_acc_offset = g * 16;
            let src_lane_offset = lane * 4;
            for i in 0..4 {
                *out_64.add(base_row + g * 4 + i) = temp[src_acc_offset + src_lane_offset + i] as f32;
            }
        }
    }
}

/// Writes the four 16-lane accumulators of the VNNI kernels (lane-interleaved) to 64 linear rows.
#[inline(always)]
unsafe fn store_vnni_rows(acc0: __m512i, acc1: __m512i, acc2: __m512i, acc3: __m512i, out_64: *mut f32) {
    let mut temp = [0i32; 64];
    _mm512_storeu_si512(temp.as_mut_ptr() as *mut __m512i, acc0);
    _mm512_storeu_si512(temp.as_mut_ptr().add(16) as *mut __m512i, acc1);
    _mm512_storeu_si512(temp.as_mut_ptr().add(32) as *mut __m512i, acc2);
    _mm512_storeu_si512(temp.as_mut_ptr().add(48) as *mut __m512i, acc3);

    for lane in 0..4 {
        let base_row = lane * 16;
        for g in 0..4 {
            let src_acc_offset = g * 16;
            let src_lane_offset = lane * 4;
            for i in 0..4 {
                *out_64.add(base_row + g * 4 + i) = temp[src_acc_offset + src_lane_offset + i] as f32;
            }
        }
    }
}

/// Fused VBMI + VNNI kernel for a wide activation table (`x = 16 * hi + lo`).
///
/// The packed weights are loaded and decomposed (`h34`, `r3`) once per column; both activation
/// tables are looked up with the same indices (`vpermb`) and accumulated into separate 32-bit
/// accumulators (`vpdpbusd`) that are combined as `lo + 16 * hi` at the end. The result is the exact
/// integer `W lo + 16 * (W hi)`. `bytes_per_row` is a multiple of 32 for every valid layout.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
pub unsafe fn gemv_quintrit_lut_block64_wide_avx512_vnni(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    debug_assert_eq!(bytes_per_row % 2, 0);
    let c_ones = _mm512_set1_epi8(1);

    let mut lo0 = _mm512_setzero_si512();
    let mut lo1 = _mm512_setzero_si512();
    let mut lo2 = _mm512_setzero_si512();
    let mut lo3 = _mm512_setzero_si512();
    let mut hi0 = _mm512_setzero_si512();
    let mut hi1 = _mm512_setzero_si512();
    let mut hi2 = _mm512_setzero_si512();
    let mut hi3 = _mm512_setzero_si512();

    let mut c = 0;
    while c + 1 < bytes_per_row {
        let (h34_0, r3_0) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add(c * 64) as *const __m512i));
        let (h34_1, r3_1) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add((c + 1) * 64) as *const __m512i));

        // lo table
        let p34_0 = _mm512_permutexvar_epi8(h34_0, _mm512_loadu_si512(qlut_lo.add(c * 128) as *const __m512i));
        let p012_0 = _mm512_permutexvar_epi8(r3_0, _mm512_loadu_si512(qlut_lo.add(c * 128 + 64) as *const __m512i));
        let p34_1 = _mm512_permutexvar_epi8(h34_1, _mm512_loadu_si512(qlut_lo.add((c + 1) * 128) as *const __m512i));
        let p012_1 = _mm512_permutexvar_epi8(r3_1, _mm512_loadu_si512(qlut_lo.add((c + 1) * 128 + 64) as *const __m512i));
        let p_lo = _mm512_unpacklo_epi8(p34_0, p012_0);
        let p_hi = _mm512_unpackhi_epi8(p34_0, p012_0);
        let q_lo = _mm512_unpacklo_epi8(p34_1, p012_1);
        let q_hi = _mm512_unpackhi_epi8(p34_1, p012_1);
        lo0 = _mm512_dpbusd_epi32(lo0, c_ones, _mm512_unpacklo_epi16(p_lo, q_lo));
        lo1 = _mm512_dpbusd_epi32(lo1, c_ones, _mm512_unpackhi_epi16(p_lo, q_lo));
        lo2 = _mm512_dpbusd_epi32(lo2, c_ones, _mm512_unpacklo_epi16(p_hi, q_hi));
        lo3 = _mm512_dpbusd_epi32(lo3, c_ones, _mm512_unpackhi_epi16(p_hi, q_hi));

        // hi table, same indices
        let p34_0 = _mm512_permutexvar_epi8(h34_0, _mm512_loadu_si512(qlut_hi.add(c * 128) as *const __m512i));
        let p012_0 = _mm512_permutexvar_epi8(r3_0, _mm512_loadu_si512(qlut_hi.add(c * 128 + 64) as *const __m512i));
        let p34_1 = _mm512_permutexvar_epi8(h34_1, _mm512_loadu_si512(qlut_hi.add((c + 1) * 128) as *const __m512i));
        let p012_1 = _mm512_permutexvar_epi8(r3_1, _mm512_loadu_si512(qlut_hi.add((c + 1) * 128 + 64) as *const __m512i));
        let p_lo = _mm512_unpacklo_epi8(p34_0, p012_0);
        let p_hi = _mm512_unpackhi_epi8(p34_0, p012_0);
        let q_lo = _mm512_unpacklo_epi8(p34_1, p012_1);
        let q_hi = _mm512_unpackhi_epi8(p34_1, p012_1);
        hi0 = _mm512_dpbusd_epi32(hi0, c_ones, _mm512_unpacklo_epi16(p_lo, q_lo));
        hi1 = _mm512_dpbusd_epi32(hi1, c_ones, _mm512_unpackhi_epi16(p_lo, q_lo));
        hi2 = _mm512_dpbusd_epi32(hi2, c_ones, _mm512_unpacklo_epi16(p_hi, q_hi));
        hi3 = _mm512_dpbusd_epi32(hi3, c_ones, _mm512_unpackhi_epi16(p_hi, q_hi));

        c += 2;
    }

    store_vnni_rows(
        _mm512_add_epi32(lo0, _mm512_slli_epi32(hi0, 4)),
        _mm512_add_epi32(lo1, _mm512_slli_epi32(hi1, 4)),
        _mm512_add_epi32(lo2, _mm512_slli_epi32(hi2, 4)),
        _mm512_add_epi32(lo3, _mm512_slli_epi32(hi3, 4)),
        out_64,
    );
}

/// Fused VBMI kernel (no VNNI required) for a wide activation table, see
/// [`gemv_quintrit_lut_block64_wide_avx512_vnni`]. Partial sums are combined as `lo + 16 * hi` in 16 bits
/// (per column at most 40 + 16 * 40, so a chunk of 16 columns stays far below the int16 range).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn gemv_quintrit_lut_block64_wide_avx512_vbmi(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    debug_assert_eq!(bytes_per_row % 2, 0);
    let c_ones = _mm512_set1_epi8(1);

    let mut acc0 = _mm512_setzero_si512(); // Rows 0..15
    let mut acc1 = _mm512_setzero_si512(); // Rows 16..31
    let mut acc2 = _mm512_setzero_si512(); // Rows 32..47
    let mut acc3 = _mm512_setzero_si512(); // Rows 48..63

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(16);
        let mut sum_lo = _mm512_setzero_si512();
        let mut sum_hi = _mm512_setzero_si512();

        let mut ci = 0;
        while ci + 1 < chunk_len {
            let c0 = c + ci;
            let c1 = c0 + 1;
            let (h34_0, r3_0) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add(c0 * 64) as *const __m512i));
            let (h34_1, r3_1) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add(c1 * 64) as *const __m512i));

            let mut parts = [(_mm512_setzero_si512(), _mm512_setzero_si512()); 2];
            for (t, qlut) in [qlut_lo, qlut_hi].into_iter().enumerate() {
                let p34_0 = _mm512_permutexvar_epi8(h34_0, _mm512_loadu_si512(qlut.add(c0 * 128) as *const __m512i));
                let p012_0 = _mm512_permutexvar_epi8(r3_0, _mm512_loadu_si512(qlut.add(c0 * 128 + 64) as *const __m512i));
                let p34_1 = _mm512_permutexvar_epi8(h34_1, _mm512_loadu_si512(qlut.add(c1 * 128) as *const __m512i));
                let p012_1 = _mm512_permutexvar_epi8(r3_1, _mm512_loadu_si512(qlut.add(c1 * 128 + 64) as *const __m512i));

                let madd_34_lo = _mm512_maddubs_epi16(c_ones, _mm512_unpacklo_epi8(p34_0, p34_1));
                let madd_34_hi = _mm512_maddubs_epi16(c_ones, _mm512_unpackhi_epi8(p34_0, p34_1));
                let madd_012_lo = _mm512_maddubs_epi16(c_ones, _mm512_unpacklo_epi8(p012_0, p012_1));
                let madd_012_hi = _mm512_maddubs_epi16(c_ones, _mm512_unpackhi_epi8(p012_0, p012_1));
                parts[t] = (_mm512_add_epi16(madd_34_lo, madd_012_lo), _mm512_add_epi16(madd_34_hi, madd_012_hi));
            }

            sum_lo = _mm512_add_epi16(sum_lo, _mm512_add_epi16(parts[0].0, _mm512_slli_epi16(parts[1].0, 4)));
            sum_hi = _mm512_add_epi16(sum_hi, _mm512_add_epi16(parts[0].1, _mm512_slli_epi16(parts[1].1, 4)));
            ci += 2;
        }

        let dw0_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_lo));
        let dw0_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(sum_lo));
        let dw1_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_hi));
        let dw1_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(sum_hi));

        acc0 = _mm512_add_epi32(acc0, dw0_lo);
        acc1 = _mm512_add_epi32(acc1, dw1_lo);
        acc2 = _mm512_add_epi32(acc2, dw0_hi);
        acc3 = _mm512_add_epi32(acc3, dw1_hi);

        c += chunk_len;
    }

    let mut temp = [0i32; 64];
    _mm512_storeu_si512(temp.as_mut_ptr() as *mut __m512i, acc0);
    _mm512_storeu_si512(temp.as_mut_ptr().add(16) as *mut __m512i, acc1);
    _mm512_storeu_si512(temp.as_mut_ptr().add(32) as *mut __m512i, acc2);
    _mm512_storeu_si512(temp.as_mut_ptr().add(48) as *mut __m512i, acc3);
    for i in 0..8 {
        *out_64.add(i) = temp[i] as f32;
        *out_64.add(8 + i) = temp[16 + i] as f32;
        *out_64.add(16 + i) = temp[8 + i] as f32;
        *out_64.add(24 + i) = temp[24 + i] as f32;
        *out_64.add(32 + i) = temp[32 + i] as f32;
        *out_64.add(40 + i) = temp[48 + i] as f32;
        *out_64.add(48 + i) = temp[40 + i] as f32;
        *out_64.add(56 + i) = temp[56 + i] as f32;
    }
}

/// Single-threaded GEMV matrix-vector multiplication using 64-row tiled AVX-512 VNNI/VBMI kernel.
pub fn gemv_lut_64_1xn_avx512(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    require_avx512();
    check_gemv_buffers(64, transformed_weights, qlut, activation_lut_len_avx512(cols), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 64;

    let has_vnni = is_x86_feature_detected!("avx512vnni");

    unsafe {
        if has_vnni {
            for b in 0..num_blocks {
                let block_w = transformed_weights.as_ptr().add(b * (64 * bytes_per_row));
                let block_out = out.as_mut_ptr().add(b * 64);
                gemv_quintrit_lut_block64_avx512_vnni(block_w, qlut.as_ptr(), bytes_per_row, block_out);
            }
        } else {
            for b in 0..num_blocks {
                let block_w = transformed_weights.as_ptr().add(b * (64 * bytes_per_row));
                let block_out = out.as_mut_ptr().add(b * 64);
                gemv_quintrit_lut_block64_avx512_vbmi(block_w, qlut.as_ptr(), bytes_per_row, block_out);
            }
        }
    }
}

/// Compatibility alias for `gemv_lut_64_1xn_avx512`.
#[inline(always)]
pub fn gemv_quintrit_lut_64_1xn(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    gemv_lut_64_1xn_avx512(transformed_weights, qlut, rows, cols, out);
}

/// Multi-threaded GEMV matrix-vector multiplication using 64-row tiled AVX-512 VNNI/VBMI kernel.
pub fn gemv_lut_64_parallel_avx512(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    require_avx512();
    check_gemv_buffers(64, transformed_weights, qlut, activation_lut_len_avx512(cols), rows, cols, out);
    let num_blocks = rows / 64;
    if num_threads <= 1 || num_blocks < num_threads {
        gemv_lut_64_1xn_avx512(transformed_weights, qlut, rows, cols, out);
        return;
    }

    let bytes_per_row = tile_row_bytes(cols);
    let blocks_per_thread = (num_blocks + num_threads - 1) / num_threads;
    let rows_per_thread = blocks_per_thread * 64;
    let has_vnni = is_x86_feature_detected!("avx512vnni");

    let weights_per_thread = rows_per_thread * bytes_per_row;
    let main_rows = num_blocks * 64;

    use rayon::prelude::*;
    let pool = crate::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..main_rows]
            .par_chunks_mut(rows_per_thread)
            .zip(transformed_weights[..main_rows * bytes_per_row].par_chunks(weights_per_thread))
            .for_each(|(out_chunk, thread_w)| {
                let b_count = out_chunk.len() / 64;
                let qlut_ptr = qlut.as_ptr();
                unsafe {
                    if has_vnni {
                        for b in 0..b_count {
                            let block_w = thread_w.as_ptr().add(b * (64 * bytes_per_row));
                            let block_out = out_chunk.as_mut_ptr().add(b * 64);
                            gemv_quintrit_lut_block64_avx512_vnni(block_w, qlut_ptr, bytes_per_row, block_out);
                        }
                    } else {
                        for b in 0..b_count {
                            let block_w = thread_w.as_ptr().add(b * (64 * bytes_per_row));
                            let block_out = out_chunk.as_mut_ptr().add(b * 64);
                            gemv_quintrit_lut_block64_avx512_vbmi(block_w, qlut_ptr, bytes_per_row, block_out);
                        }
                    }
                }
            });
    });
}

/// Compatibility alias for `gemv_lut_64_parallel_avx512`.
#[inline(always)]
pub fn gemv_quintrit_lut_64_parallel(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    gemv_lut_64_parallel_avx512(transformed_weights, qlut, rows, cols, out, num_threads);
}


// ============================================================================
// Exact (hi/lo split) GEMV for the full int8 activation range
// ============================================================================

/// Length in bytes of the table produced by [`build_activation_lut_wide_avx512_full`]: the `lo` and the
/// `hi` AVX-512 table back to back.
pub fn wide_activation_lut_len_avx512_full(cols: usize) -> usize {
    let bytes_per_row = tile_row_bytes(cols);
    2 * bytes_per_row * 128
}

/// Builds the AVX-512 activation tables of the two parts of `x = 16 * hi + lo`, as `[lo | hi]`.
///
/// The plain table stores sums of up to three products in int8 and saturates for |x| > ~42. Both
/// parts are tiny (`|part| <= 8`), so running the kernel on each of them is exact for any int8 input.
pub fn build_activation_lut_wide_avx512_full(activations: &[i8], cols: usize) -> Vec<i8> {
    let (mut hi, mut lo) = (vec![0i8; cols], vec![0i8; cols]);
    crate::lut::split_activations_hi_lo(&activations[..cols], &mut hi, &mut lo);
    let mut wide = build_activation_lut_avx512(&lo, cols);
    wide.extend_from_slice(&build_activation_lut_avx512(&hi, cols));
    wide
}

/// Single-threaded exact GEMV against a wide table (`[lo | hi]`): `W lo + 16 * (W hi)`.
///
/// Uses the fused kernels, which read and decode each block of weights once for both parts.
pub fn gemv_lut_64_wide_1xn_avx512_full(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512_full(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    let has_vnni = is_x86_feature_detected!("avx512vnni");

    unsafe {
        for b in 0..rows / 64 {
            let block_w = transformed_weights.as_ptr().add(b * (64 * bytes_per_row));
            let block_out = out.as_mut_ptr().add(b * 64);
            if has_vnni {
                gemv_quintrit_lut_block64_wide_avx512_vnni(block_w, lo.as_ptr(), hi.as_ptr(), bytes_per_row, block_out);
            } else {
                gemv_quintrit_lut_block64_wide_avx512_vbmi(block_w, lo.as_ptr(), hi.as_ptr(), bytes_per_row, block_out);
            }
        }
    }
}

/// Multi-threaded variant of [`gemv_lut_64_wide_1xn_avx512_full`].
pub fn gemv_lut_64_wide_parallel_avx512_full(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512_full(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let num_blocks = rows / 64;
    if num_threads <= 1 || num_blocks < num_threads {
        gemv_lut_64_wide_1xn_avx512_full(transformed_weights, wide_qlut, rows, cols, out);
        return;
    }
    let bytes_per_row = tile_row_bytes(cols);
    let rows_per_thread = num_blocks.div_ceil(num_threads) * 64;

    use rayon::prelude::*;
    crate::threading::get_thread_pool(num_threads).install(|| {
        out[..rows]
            .par_chunks_mut(rows_per_thread)
            .zip(transformed_weights[..rows * bytes_per_row].par_chunks(rows_per_thread * bytes_per_row))
            .for_each(|(out_chunk, weights)| {
                gemv_lut_64_wide_1xn_avx512_full(weights, wide_qlut, out_chunk.len(), cols, out_chunk);
            });
    });
}

// ============================================================================
// Experimental variants of the fused hi/lo kernel
// ============================================================================

/// Per-column sums of the two table lookups (`p34 + p012`, magnitude <= 40) for the `lo` and the `hi` table.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn column_sums_i8_avx512(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    c: usize,
) -> (__m512i, __m512i) {
    let (h34, r3) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add(c * 64) as *const __m512i));
    let lo = _mm512_add_epi8(
        _mm512_permutexvar_epi8(h34, _mm512_loadu_si512(qlut_lo.add(c * 128) as *const __m512i)),
        _mm512_permutexvar_epi8(r3, _mm512_loadu_si512(qlut_lo.add(c * 128 + 64) as *const __m512i)),
    );
    let hi = _mm512_add_epi8(
        _mm512_permutexvar_epi8(h34, _mm512_loadu_si512(qlut_hi.add(c * 128) as *const __m512i)),
        _mm512_permutexvar_epi8(r3, _mm512_loadu_si512(qlut_hi.add(c * 128 + 64) as *const __m512i)),
    );
    (lo, hi)
}

/// Variant `i8red` of the fused AVX-512 kernels: the sums of three columns are formed in int8 (<= 120) and
/// combined as `lo + 16 * hi` by one `vpmaddubsw` with the weights (1, 16) on the interleaved bytes. Needs
/// AVX-512 BW and VBMI only (no VNNI). The result is identical to the other fused kernels.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn gemv_quintrit_lut_block64_wide_avx512_i8red(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    gemv_quintrit_lut_block64_wide_avx512_i8red_impl::<0>(block_weights, qlut_lo, qlut_hi, bytes_per_row, out_64)
}

/// `i8red` with a software prefetch `PF` bytes ahead of the weight stream (0 = none).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn gemv_quintrit_lut_block64_wide_avx512_i8red_impl<const PF: usize>(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    let combine = _mm512_set1_epi16(0x1001); // bytes (1, 16)
    // 32-bit accumulators holding 16 rows each: (a0, a1) from the low-half unpack, (b0, b1) from the high-half unpack.
    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut b0 = _mm512_setzero_si512();
    let mut b1 = _mm512_setzero_si512();

    const CHUNK: usize = 45; // 15 groups of three columns, |lo + 16 hi| <= 2040 per group
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut sum_a = _mm512_setzero_si512();
        let mut sum_b = _mm512_setzero_si512();
        let mut ci = 0;
        while ci + 3 <= chunk_len {
            if PF != 0 {
                let ahead = block_weights.add((c + ci) * 64 + PF);
                _mm_prefetch(ahead as *const i8, _MM_HINT_T0);
                _mm_prefetch(ahead.add(64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(ahead.add(128) as *const i8, _MM_HINT_T0);
            }
            let (l0, h0) = column_sums_i8_avx512(block_weights, qlut_lo, qlut_hi, c + ci);
            let (l1, h1) = column_sums_i8_avx512(block_weights, qlut_lo, qlut_hi, c + ci + 1);
            let (l2, h2) = column_sums_i8_avx512(block_weights, qlut_lo, qlut_hi, c + ci + 2);
            let slo = _mm512_add_epi8(_mm512_add_epi8(l0, l1), l2);
            let shi = _mm512_add_epi8(_mm512_add_epi8(h0, h1), h2);
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
            ci += 3;
        }
        if ci < chunk_len {
            let (mut slo, mut shi) = column_sums_i8_avx512(block_weights, qlut_lo, qlut_hi, c + ci);
            if ci + 1 < chunk_len {
                let (l, h) = column_sums_i8_avx512(block_weights, qlut_lo, qlut_hi, c + ci + 1);
                slo = _mm512_add_epi8(slo, l);
                shi = _mm512_add_epi8(shi, h);
            }
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
        }
        a0 = _mm512_add_epi32(a0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_a)));
        a1 = _mm512_add_epi32(a1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_a, 1)));
        b0 = _mm512_add_epi32(b0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_b)));
        b1 = _mm512_add_epi32(b1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_b, 1)));
        c += chunk_len;
    }

    // Row order: a0 = rows 0..7 | 16..23, a1 = 32..39 | 48..55, b0 = 8..15 | 24..31, b1 = 40..47 | 56..63.
    let mut tmp = [0i32; 16];
    let mut put = |acc: __m512i, first: usize, second: usize| {
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
        for k in 0..8 {
            *out_64.add(first + k) = tmp[k] as f32;
            *out_64.add(second + k) = tmp[8 + k] as f32;
        }
    };
    put(a0, 0, 16);
    put(b0, 8, 24);
    put(a1, 32, 48);
    put(b1, 40, 56);
}

/// Experimental driver: single-threaded wide AVX-512 GEMV with a selectable block kernel.
/// Variants: `0` = original fused kernel (VNNI if available), `1` = `i8red`, `2`/`3` = `i8red` with a 2 / 4 KiB prefetch.
#[doc(hidden)]
pub fn gemv_lut_64_wide_1xn_avx512_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512_full(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    if variant == 0 {
        gemv_lut_64_wide_1xn_avx512_full(transformed_weights, wide_qlut, rows, cols, out);
        return;
    }
    type BlockFn = unsafe fn(*const u8, *const i8, *const i8, usize, *mut f32);
    let f: BlockFn = match variant {
        1 => gemv_quintrit_lut_block64_wide_avx512_i8red,
        2 => gemv_quintrit_lut_block64_wide_avx512_i8red_impl::<2048>,
        3 => gemv_quintrit_lut_block64_wide_avx512_i8red_impl::<4096>,
        _ => panic!("unknown AVX-512 variant {variant}"),
    };
    unsafe {
        for b in 0..rows / 64 {
            f(
                transformed_weights.as_ptr().add(b * (64 * bytes_per_row)),
                lo.as_ptr(),
                hi.as_ptr(),
                bytes_per_row,
                out.as_mut_ptr().add(b * 64),
            );
        }
    }
}


// ============================================================================
// Compact activation tables (one 64-byte table per column and part)
// ============================================================================

/// Offset of the `tbl_012` entries inside a compact 64-byte table (`tbl_34` occupies indices 0..=8).
const COMPACT_012_OFFSET: u8 = 9;

/// Length in bytes of the compact wide table: `lo` and `hi` part, 64 bytes per column each.
pub fn wide_activation_lut_len_avx512(cols: usize) -> usize {
    let bytes_per_row = tile_row_bytes(cols);
    2 * bytes_per_row * 64
}

/// Compact variant of [`build_activation_lut_avx512`]: both lookups of a column share one 64-byte table
/// (`tbl_34` at indices 0..=8, `tbl_012` at indices 9..=35), halving the footprint of the activation tables.
pub fn build_activation_lut_avx512_compact(activations: &[i8], cols: usize) -> Vec<i8> {
    let full = build_activation_lut_avx512(activations, cols);
    let bytes_per_row = tile_row_bytes(cols);
    let mut qlut = vec![0i8; bytes_per_row * 64];
    for c in 0..bytes_per_row {
        let (t34, t012) = (&full[c * 128..c * 128 + 64], &full[c * 128 + 64..c * 128 + 128]);
        let dst = &mut qlut[c * 64..c * 64 + 64];
        dst[..9].copy_from_slice(&t34[..9]);
        dst[COMPACT_012_OFFSET as usize..COMPACT_012_OFFSET as usize + 27].copy_from_slice(&t012[..27]);
    }
    qlut
}

/// Compact wide table `[lo | hi]` (see [`build_activation_lut_avx512_compact`]).
pub fn build_activation_lut_wide_avx512(activations: &[i8], cols: usize) -> Vec<i8> {
    let mut wide = vec![0i8; wide_activation_lut_len_avx512(cols)];
    build_activation_lut_wide_avx512_into(activations, cols, &mut wide);
    wide
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn column_sums_i8_avx512_compact(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    c: usize,
) -> (__m512i, __m512i) {
    let (h34, r3) = decompose_quintrit_avx512(_mm512_loadu_si512(block_weights.add(c * 64) as *const __m512i));
    let r3 = _mm512_add_epi8(r3, _mm512_set1_epi8(COMPACT_012_OFFSET as i8));
    let t_lo = _mm512_loadu_si512(qlut_lo.add(c * 64) as *const __m512i);
    let t_hi = _mm512_loadu_si512(qlut_hi.add(c * 64) as *const __m512i);
    (
        _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_lo), _mm512_permutexvar_epi8(r3, t_lo)),
        _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_hi), _mm512_permutexvar_epi8(r3, t_hi)),
    )
}

/// `i8red` kernel on the compact tables, with a software prefetch `PF` bytes ahead of the weights (0 = none).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn gemv_quintrit_lut_block64_wide_avx512_compact<const PF: usize>(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_64: *mut f32,
) {
    let combine = _mm512_set1_epi16(0x1001);
    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut b0 = _mm512_setzero_si512();
    let mut b1 = _mm512_setzero_si512();
    const CHUNK: usize = 45;
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut sum_a = _mm512_setzero_si512();
        let mut sum_b = _mm512_setzero_si512();
        let mut ci = 0;
        while ci + 3 <= chunk_len {
            if PF != 0 {
                let ahead = block_weights.add((c + ci) * 64 + PF);
                _mm_prefetch(ahead as *const i8, _MM_HINT_T0);
                _mm_prefetch(ahead.add(64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(ahead.add(128) as *const i8, _MM_HINT_T0);
            }
            let (l0, h0) = column_sums_i8_avx512_compact(block_weights, qlut_lo, qlut_hi, c + ci);
            let (l1, h1) = column_sums_i8_avx512_compact(block_weights, qlut_lo, qlut_hi, c + ci + 1);
            let (l2, h2) = column_sums_i8_avx512_compact(block_weights, qlut_lo, qlut_hi, c + ci + 2);
            let slo = _mm512_add_epi8(_mm512_add_epi8(l0, l1), l2);
            let shi = _mm512_add_epi8(_mm512_add_epi8(h0, h1), h2);
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
            ci += 3;
        }
        if ci < chunk_len {
            let (mut slo, mut shi) = column_sums_i8_avx512_compact(block_weights, qlut_lo, qlut_hi, c + ci);
            if ci + 1 < chunk_len {
                let (l, h) = column_sums_i8_avx512_compact(block_weights, qlut_lo, qlut_hi, c + ci + 1);
                slo = _mm512_add_epi8(slo, l);
                shi = _mm512_add_epi8(shi, h);
            }
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
        }
        a0 = _mm512_add_epi32(a0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_a)));
        a1 = _mm512_add_epi32(a1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_a, 1)));
        b0 = _mm512_add_epi32(b0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_b)));
        b1 = _mm512_add_epi32(b1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_b, 1)));
        c += chunk_len;
    }
    let mut tmp = [0i32; 16];
    let mut put = |acc: __m512i, first: usize, second: usize| {
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
        for k in 0..8 {
            *out_64.add(first + k) = tmp[k] as f32;
            *out_64.add(second + k) = tmp[8 + k] as f32;
        }
    };
    put(a0, 0, 16);
    put(b0, 8, 24);
    put(a1, 32, 48);
    put(b1, 40, 56);
}

/// Experimental driver for the compact-table kernels: variants `4` = no prefetch, `5` = 2 KiB, `6` = 4 KiB.
/// `wide_qlut` must come from [`build_activation_lut_wide_avx512`].
#[doc(hidden)]
pub fn gemv_lut_64_wide_1xn_avx512_compact_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols), "expected a compact wide table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    type BlockFn = unsafe fn(*const u8, *const i8, *const i8, usize, *mut f32);
    let f: BlockFn = match variant {
        4 => gemv_quintrit_lut_block64_wide_avx512_compact::<0>,
        5 => gemv_quintrit_lut_block64_wide_avx512_compact::<2048>,
        6 => gemv_quintrit_lut_block64_wide_avx512_compact::<4096>,
        _ => panic!("unknown compact AVX-512 variant {variant}"),
    };
    unsafe {
        for b in 0..rows / 64 {
            f(
                transformed_weights.as_ptr().add(b * (64 * bytes_per_row)),
                lo.as_ptr(),
                hi.as_ptr(),
                bytes_per_row,
                out.as_mut_ptr().add(b * 64),
            );
        }
    }
}


/// Single-threaded exact GEMV against a wide activation table (`[lo | hi]`, see
/// [`build_activation_lut_wide_avx512`]): `W lo + 16 * (W hi)`.
///
/// Production path: the int8-accumulating kernel on the compact tables, shuffle-based weight decode, a 4 KiB software
/// prefetch of every weight stream, and `PRODUCTION_BLOCKS` consecutive 64-row blocks per pass over the tables.
pub fn gemv_lut_64_wide_1xn_avx512(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    let forced = VARIANT_OVERRIDE.load(core::sync::atomic::Ordering::Relaxed);
    if forced != 0 {
        run_variant(forced, transformed_weights, wide_qlut, rows, cols, out);
        return;
    }
    unsafe {
        gemv_compact_multi::<PRODUCTION_BLOCKS, 4096, { _MM_HINT_T0 }, 0, true>(
            transformed_weights.as_ptr(),
            lo.as_ptr(),
            hi.as_ptr(),
            rows / 64,
            bytes_per_row,
            out.as_mut_ptr(),
        );
    }
}

/// Number of 64-row blocks that the production kernel processes per pass over the activation tables.
const PRODUCTION_BLOCKS: usize = 3;

static VARIANT_OVERRIDE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Benchmark hook: makes [`gemv_lut_64_wide_1xn_avx512`] (and so the parallel driver and the engine) run the experimental
/// variant `variant` instead of the production kernel; `0` restores the production kernel. Variants `4..=6`, `8..=26`
/// and `30..` are accepted (see the variant drivers).
#[doc(hidden)]
pub fn set_variant_override(variant: u32) {
    VARIANT_OVERRIDE.store(variant, core::sync::atomic::Ordering::Relaxed);
}

fn run_variant(variant: u32, w: &[u8], wide_qlut: &[i8], rows: usize, cols: usize, out: &mut [f32]) {
    match variant {
        4..=7 => gemv_lut_64_wide_1xn_avx512_compact_variant(variant, w, wide_qlut, rows, cols, out),
        8..=29 => gemv_lut_64_wide_1xn_avx512_v2_variant(variant, w, wide_qlut, rows, cols, out),
        40..=49 => gemv_lut_64_single_table_variant(variant, w, wide_qlut, rows, cols, out),
        _ => gemv_lut_64_wide_1xn_avx512_multi_variant(variant, w, wide_qlut, rows, cols, out),
    }
}

/// Multi-threaded variant of [`gemv_lut_64_wide_1xn_avx512`]; rows are split across threads in whole 64-row tiles.
pub fn gemv_lut_64_wide_parallel_avx512(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    require_avx512();
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(64, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let num_blocks = rows / 64;
    if num_threads <= 1 || num_blocks < num_threads {
        gemv_lut_64_wide_1xn_avx512(transformed_weights, wide_qlut, rows, cols, out);
        return;
    }
    let bytes_per_row = tile_row_bytes(cols);
    let rows_per_thread = num_blocks.div_ceil(num_threads) * 64;

    use rayon::prelude::*;
    crate::threading::get_thread_pool(num_threads).install(|| {
        out[..rows]
            .par_chunks_mut(rows_per_thread)
            .zip(transformed_weights[..rows * bytes_per_row].par_chunks(rows_per_thread * bytes_per_row))
            .for_each(|(out_chunk, weights)| {
                gemv_lut_64_wide_1xn_avx512(weights, wide_qlut, out_chunk.len(), cols, out_chunk);
            });
    });
}


// ============================================================================
// Direct builder of the compact wide tables
// ============================================================================

/// Lane masks `(positive, negative)` of the coefficients of the five activation positions in a compact 64-byte table:
/// lanes 0..=8 hold `tbl_34` (coefficients of `a3`, `a4`), lanes 9..=35 hold `tbl_012` (`a0`, `a1`, `a2`).
fn compact_coefficient_masks() -> [(u64, u64); 5] {
    let mut m = [(0u64, 0u64); 5];
    let mut set = |p: usize, lane: usize, coef: i8| {
        if coef > 0 {
            m[p].0 |= 1u64 << lane;
        } else if coef < 0 {
            m[p].1 |= 1u64 << lane;
        }
    };
    for s in 0..9usize {
        set(3, s, (s % 3) as i8 - 1);
        set(4, s, (s / 3) as i8 - 1);
    }
    for s in 0..27usize {
        set(0, COMPACT_012_OFFSET as usize + s, (s % 3) as i8 - 1);
        set(1, COMPACT_012_OFFSET as usize + s, ((s / 3) % 3) as i8 - 1);
        set(2, COMPACT_012_OFFSET as usize + s, (s / 9) as i8 - 1);
    }
    m
}

/// Builds the compact wide table (`[lo | hi]`, see [`build_activation_lut_wide_avx512`]) directly into `out`, without
/// intermediate allocations. The table entries are formed by masked adds and subtracts of the broadcast activation
/// parts (the coefficients are -1, 0 or 1); the parts have magnitude <= 8, so nothing saturates.
///
/// `out` must hold at least [`wide_activation_lut_len_avx512`] bytes; returns the number of bytes written.
pub fn build_activation_lut_wide_avx512_into(activations: &[i8], cols: usize, out: &mut [i8]) -> usize {
    assert!(
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"),
        "the AVX-512 table builder needs AVX-512 F and BW"
    );
    let total = wide_activation_lut_len_avx512(cols);
    assert!(out.len() >= total, "destination too small for the compact wide table");
    assert!(activations.len() >= cols);
    unsafe { build_compact_wide_impl(activations, cols, out, total) };
    total
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn build_compact_wide_impl(acts: &[i8], cols: usize, out: &mut [i8], total: usize) {
    let bytes_per_row = tile_row_bytes(cols);
    let masks = compact_coefficient_masks();
    let single = total / 2;
    let dst_lo = out.as_mut_ptr();
    let dst_hi = out.as_mut_ptr().add(single);
    for c in 0..bytes_per_row {
        let mut lo = [0i8; 5];
        let mut hi = [0i8; 5];
        for p in 0..5 {
            let idx = crate::pack::column_activation_index(cols, c, p);
            let x = if idx < cols { *acts.get_unchecked(idx) as i16 } else { 0 };
            let l = ((x + 8) & 15) - 8;
            lo[p] = l as i8;
            hi[p] = ((x - l) >> 4) as i8;
        }
        let mut acc_lo = _mm512_setzero_si512();
        let mut acc_hi = _mm512_setzero_si512();
        for p in 0..5 {
            let (pos, neg) = masks[p];
            let bl = _mm512_set1_epi8(lo[p]);
            let bh = _mm512_set1_epi8(hi[p]);
            acc_lo = _mm512_mask_add_epi8(acc_lo, pos, acc_lo, bl);
            acc_lo = _mm512_mask_sub_epi8(acc_lo, neg, acc_lo, bl);
            acc_hi = _mm512_mask_add_epi8(acc_hi, pos, acc_hi, bh);
            acc_hi = _mm512_mask_sub_epi8(acc_hi, neg, acc_hi, bh);
        }
        _mm512_storeu_si512(dst_lo.add(c * 64) as *mut __m512i, acc_lo);
        _mm512_storeu_si512(dst_hi.add(c * 64) as *mut __m512i, acc_hi);
    }
}


// ============================================================================
// Second campaign: shuffle-based weight decode and K-blocked table reuse (experimental variants)
// ============================================================================

/// `floor(i / 7)` for the 64 possible values of `w >> 2`: the quotient `h34 = floor(w / 28)` of a weight byte, because
/// 28 is a multiple of 4 (`floor(floor(w / 4) / 7) = floor(w / 28)`).
static DECODE_H34: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (i / 7) as u8;
        i += 1;
    }
    t
};

/// `28 * floor(i / 7) - COMPACT_012_OFFSET` (mod 256) for the 64 possible values of `w >> 2`; `w` minus this value is the
/// remainder `r3` already shifted to its place in a compact table.
static DECODE_M28: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (28 * (i / 7) as u32).wrapping_sub(COMPACT_012_OFFSET as u32) as u8;
        i += 1;
    }
    t
};

/// Decodes 64 weight bytes into the two compact-table indices `(h34, r3 + COMPACT_012_OFFSET)`.
///
/// `PERM = true` uses two byte shuffles on `w >> 2` (the upper two bits of every shifted byte are garbage from the
/// neighbouring byte and are ignored by `vpermb`) and one subtraction; `PERM = false` is the arithmetic decode of
/// [`decompose_quintrit_avx512`] plus the table offset.
#[inline(always)]
unsafe fn decode_compact<const PERM: bool>(w: __m512i, th: __m512i, tm: __m512i) -> (__m512i, __m512i) {
    if PERM {
        let idx = _mm512_srli_epi16::<2>(w);
        (_mm512_permutexvar_epi8(idx, th), _mm512_sub_epi8(w, _mm512_permutexvar_epi8(idx, tm)))
    } else {
        let (h34, r3) = decompose_quintrit_avx512(w);
        (h34, _mm512_add_epi8(r3, _mm512_set1_epi8(COMPACT_012_OFFSET as i8)))
    }
}

/// One column of the compact `i8red` kernel: per-table sums of the two lookups (magnitude <= 40).
#[inline(always)]
unsafe fn compact_column<const PERM: bool>(
    w: *const u8,
    lo: *const i8,
    hi: *const i8,
    c: usize,
    th: __m512i,
    tm: __m512i,
) -> (__m512i, __m512i) {
    let (h34, r3) = decode_compact::<PERM>(_mm512_loadu_si512(w.add(c * 64) as *const __m512i), th, tm);
    let t_lo = _mm512_loadu_si512(lo.add(c * 64) as *const __m512i);
    let t_hi = _mm512_loadu_si512(hi.add(c * 64) as *const __m512i);
    (
        _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_lo), _mm512_permutexvar_epi8(r3, t_lo)),
        _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_hi), _mm512_permutexvar_epi8(r3, t_hi)),
    )
}

/// Processes `ncols` columns of one 64-row block (`w`, `lo`, `hi` point at the first column of the chunk) and adds the
/// result to the 64 int32 accumulators at `acc` (stored as four 16-lane vectors; zeroed first when `first` is set).
/// `PF` is the software prefetch distance in bytes (0 = none). `ncols` must be a multiple of 45 except in the last chunk.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn compact_chunk<const PF: usize, const HINT: i32, const PN: usize, const PERM: bool>(
    w: *const u8,
    lo: *const i8,
    hi: *const i8,
    ncols: usize,
    acc: *mut i32,
    first: bool,
) {
    let th = _mm512_loadu_si512(DECODE_H34.as_ptr() as *const __m512i);
    let tm = _mm512_loadu_si512(DECODE_M28.as_ptr() as *const __m512i);
    let combine = _mm512_set1_epi16(0x1001);
    let (mut a0, mut a1, mut b0, mut b1);
    if first {
        a0 = _mm512_setzero_si512();
        a1 = _mm512_setzero_si512();
        b0 = _mm512_setzero_si512();
        b1 = _mm512_setzero_si512();
    } else {
        a0 = _mm512_loadu_si512(acc as *const __m512i);
        a1 = _mm512_loadu_si512(acc.add(16) as *const __m512i);
        b0 = _mm512_loadu_si512(acc.add(32) as *const __m512i);
        b1 = _mm512_loadu_si512(acc.add(48) as *const __m512i);
    }
    const CHUNK: usize = 45;
    let mut c = 0;
    while c < ncols {
        let chunk_len = (ncols - c).min(CHUNK);
        let mut sum_a = _mm512_setzero_si512();
        let mut sum_b = _mm512_setzero_si512();
        let mut ci = 0;
        while ci + 3 <= chunk_len {
            if PF != 0 {
                let ahead = w.add((c + ci) * 64 + PF);
                _mm_prefetch::<HINT>(ahead as *const i8);
                _mm_prefetch::<HINT>(ahead.add(64) as *const i8);
                _mm_prefetch::<HINT>(ahead.add(128) as *const i8);
            }
            if PN != 0 {
                let near = w.add((c + ci) * 64 + PN);
                _mm_prefetch::<_MM_HINT_T0>(near as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(near.add(64) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(near.add(128) as *const i8);
            }
            let (l0, h0) = compact_column::<PERM>(w, lo, hi, c + ci, th, tm);
            let (l1, h1) = compact_column::<PERM>(w, lo, hi, c + ci + 1, th, tm);
            let (l2, h2) = compact_column::<PERM>(w, lo, hi, c + ci + 2, th, tm);
            let slo = _mm512_add_epi8(_mm512_add_epi8(l0, l1), l2);
            let shi = _mm512_add_epi8(_mm512_add_epi8(h0, h1), h2);
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
            ci += 3;
        }
        if ci < chunk_len {
            let (mut slo, mut shi) = compact_column::<PERM>(w, lo, hi, c + ci, th, tm);
            if ci + 1 < chunk_len {
                let (l, h) = compact_column::<PERM>(w, lo, hi, c + ci + 1, th, tm);
                slo = _mm512_add_epi8(slo, l);
                shi = _mm512_add_epi8(shi, h);
            }
            sum_a = _mm512_add_epi16(sum_a, _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo, shi)));
            sum_b = _mm512_add_epi16(sum_b, _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo, shi)));
        }
        a0 = _mm512_add_epi32(a0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_a)));
        a1 = _mm512_add_epi32(a1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_a, 1)));
        b0 = _mm512_add_epi32(b0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_b)));
        b1 = _mm512_add_epi32(b1, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_b, 1)));
        c += chunk_len;
    }
    _mm512_storeu_si512(acc as *mut __m512i, a0);
    _mm512_storeu_si512(acc.add(16) as *mut __m512i, a1);
    _mm512_storeu_si512(acc.add(32) as *mut __m512i, b0);
    _mm512_storeu_si512(acc.add(48) as *mut __m512i, b1);
}

/// Converts the 64 int32 accumulators of a block (layout of [`compact_chunk`]) into the 64 output rows in place.
#[inline(always)]
unsafe fn finish_block(acc: *mut i32, out: *mut f32) {
    let mut tmp = [0i32; 64];
    for (i, t) in tmp.iter_mut().enumerate() {
        *t = *acc.add(i);
    }
    // Row order: a0 = rows 0..7 | 16..23, a1 = 32..39 | 48..55, b0 = 8..15 | 24..31, b1 = 40..47 | 56..63.
    for k in 0..8 {
        *out.add(k) = tmp[k] as f32;
        *out.add(16 + k) = tmp[8 + k] as f32;
        *out.add(32 + k) = tmp[16 + k] as f32;
        *out.add(48 + k) = tmp[24 + k] as f32;
        *out.add(8 + k) = tmp[32 + k] as f32;
        *out.add(24 + k) = tmp[40 + k] as f32;
        *out.add(40 + k) = tmp[48 + k] as f32;
        *out.add(56 + k) = tmp[56 + k] as f32;
    }
}

/// Compact wide GEMV over `nblocks` 64-row blocks, K-blocked: the columns are processed in chunks of `KB` (a multiple
/// of 45, or `usize::MAX` for one chunk) and every chunk is applied to all blocks before the next one starts, so the
/// activation tables of a chunk stay in the L1 data cache while the weights stream through. The integer partial sums
/// are kept in the output buffer (as raw int32) between chunks; integer addition makes the result independent of the order.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn gemv_compact_kblock<const PF: usize, const HINT: i32, const PN: usize, const PERM: bool, const KB: usize>(
    weights: *const u8,
    lo: *const i8,
    hi: *const i8,
    nblocks: usize,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let acc = out as *mut i32;
    let mut c0 = 0;
    while c0 < bytes_per_row {
        let n = (bytes_per_row - c0).min(KB);
        for b in 0..nblocks {
            compact_chunk::<PF, HINT, PN, PERM>(
                weights.add(b * 64 * bytes_per_row + c0 * 64),
                lo.add(c0 * 64),
                hi.add(c0 * 64),
                n,
                acc.add(b * 64),
                c0 == 0,
            );
        }
        c0 += n;
    }
    for b in 0..nblocks {
        finish_block(acc.add(b * 64), out.add(b * 64));
    }
}

/// Experimental driver for the second-campaign kernels (compact tables from [`build_activation_lut_wide_avx512`]).
///
/// | id | decode | prefetch | K block (columns) |
/// |---:|---|---:|---:|
/// | 8 | shuffle | 2 KiB | none |
/// | 9 | shuffle | 4 KiB | none |
/// | 10 | arithmetic | 6 KiB | none |
/// | 11 | arithmetic | 8 KiB | none |
/// | 12 | shuffle | 2 KiB | 135 |
/// | 13 | shuffle | 2 KiB | 225 |
/// | 14 | arithmetic | 2 KiB | 135 |
/// | 15 | shuffle | 0 | none |
/// | 16..=20 | shuffle | 6, 8, 12, 16, 24 KiB | none |
/// | 21..=26 | shuffle | far prefetch with hint T1 / NTA plus an optional near T0 prefetch (see the match) | none |
#[doc(hidden)]
pub fn gemv_lut_64_wide_1xn_avx512_v2_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % 64, 0, "rows must be a multiple of 64");
    assert!(out.len() >= rows);
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols), "expected a compact wide table");
    require_avx512();
    let bytes_per_row = tile_row_bytes(cols);
    assert!(transformed_weights.len() >= rows * bytes_per_row);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    type KFn = unsafe fn(*const u8, *const i8, *const i8, usize, usize, *mut f32);
    const ALL: usize = usize::MAX;
    const T0: i32 = _MM_HINT_T0;
    const T1: i32 = _MM_HINT_T1;
    const NTA: i32 = _MM_HINT_NTA;
    let f: KFn = match variant {
        8 => gemv_compact_kblock::<2048, T0, 0, true, ALL>,
        9 => gemv_compact_kblock::<4096, T0, 0, true, ALL>,
        10 => gemv_compact_kblock::<6144, T0, 0, false, ALL>,
        11 => gemv_compact_kblock::<8192, T0, 0, false, ALL>,
        12 => gemv_compact_kblock::<2048, T0, 0, true, 135>,
        13 => gemv_compact_kblock::<2048, T0, 0, true, 225>,
        14 => gemv_compact_kblock::<2048, T0, 0, false, 135>,
        15 => gemv_compact_kblock::<0, T0, 0, true, ALL>,
        16 => gemv_compact_kblock::<6144, T0, 0, true, ALL>,
        17 => gemv_compact_kblock::<8192, T0, 0, true, ALL>,
        18 => gemv_compact_kblock::<12288, T0, 0, true, ALL>,
        19 => gemv_compact_kblock::<16384, T0, 0, true, ALL>,
        20 => gemv_compact_kblock::<24576, T0, 0, true, ALL>,
        21 => gemv_compact_kblock::<8192, T1, 0, true, ALL>,
        22 => gemv_compact_kblock::<8192, T1, 1024, true, ALL>,
        23 => gemv_compact_kblock::<16384, T1, 2048, true, ALL>,
        24 => gemv_compact_kblock::<4096, NTA, 0, true, ALL>,
        25 => gemv_compact_kblock::<16384, T1, 1024, true, ALL>,
        26 => gemv_compact_kblock::<6144, T1, 1024, true, ALL>,
        _ => panic!("unknown AVX-512 v2 variant {variant}"),
    };
    unsafe { f(transformed_weights.as_ptr(), lo.as_ptr(), hi.as_ptr(), rows / 64, bytes_per_row, out.as_mut_ptr()) };
}

/// Writes the rows of one block from its four accumulators (layout of the compact `i8red` kernels).
#[inline(always)]
unsafe fn put_block_rows(a0: __m512i, a1: __m512i, b0: __m512i, b1: __m512i, out_64: *mut f32) {
    let mut tmp = [0i32; 16];
    let mut put = |acc: __m512i, first: usize, second: usize| {
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
        for k in 0..8 {
            *out_64.add(first + k) = tmp[k] as f32;
            *out_64.add(second + k) = tmp[8 + k] as f32;
        }
    };
    put(a0, 0, 16);
    put(b0, 8, 24);
    put(a1, 32, 48);
    put(b1, 40, 56);
}

/// Compact `i8red` kernel for `NB` consecutive 64-row blocks at once: every column's activation tables are loaded once
/// and applied to the weights of all `NB` blocks (`block_stride` bytes apart), which also gives `NB` independent
/// weight streams. `PF` / `HINT` / `PN` as in [`compact_chunk`].
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn compact_multi_block<const NB: usize, const PF: usize, const HINT: i32, const PN: usize, const PERM: bool>(
    w: *const u8,
    block_stride: usize,
    lo: *const i8,
    hi: *const i8,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let th = _mm512_loadu_si512(DECODE_H34.as_ptr() as *const __m512i);
    let tm = _mm512_loadu_si512(DECODE_M28.as_ptr() as *const __m512i);
    let combine = _mm512_set1_epi16(0x1001);
    let zero = _mm512_setzero_si512();
    let mut a0 = [zero; NB];
    let mut a1 = [zero; NB];
    let mut b0 = [zero; NB];
    let mut b1 = [zero; NB];

    const CHUNK: usize = 45;
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut sum_a = [zero; NB];
        let mut sum_b = [zero; NB];
        let mut ci = 0;
        while ci < chunk_len {
            let n = (chunk_len - ci).min(3);
            if PF != 0 || PN != 0 {
                for blk in 0..NB {
                    let base = w.add(blk * block_stride + (c + ci) * 64);
                    if PF != 0 {
                        _mm_prefetch::<HINT>(base.add(PF) as *const i8);
                        _mm_prefetch::<HINT>(base.add(PF + 64) as *const i8);
                        _mm_prefetch::<HINT>(base.add(PF + 128) as *const i8);
                    }
                    if PN != 0 {
                        _mm_prefetch::<_MM_HINT_T0>(base.add(PN) as *const i8);
                        _mm_prefetch::<_MM_HINT_T0>(base.add(PN + 64) as *const i8);
                        _mm_prefetch::<_MM_HINT_T0>(base.add(PN + 128) as *const i8);
                    }
                }
            }
            let mut slo = [zero; NB];
            let mut shi = [zero; NB];
            for j in 0..n {
                let col = c + ci + j;
                let t_lo = _mm512_loadu_si512(lo.add(col * 64) as *const __m512i);
                let t_hi = _mm512_loadu_si512(hi.add(col * 64) as *const __m512i);
                for blk in 0..NB {
                    let wv = _mm512_loadu_si512(w.add(blk * block_stride + col * 64) as *const __m512i);
                    let (h34, r3) = decode_compact::<PERM>(wv, th, tm);
                    let l = _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_lo), _mm512_permutexvar_epi8(r3, t_lo));
                    let h = _mm512_add_epi8(_mm512_permutexvar_epi8(h34, t_hi), _mm512_permutexvar_epi8(r3, t_hi));
                    if j == 0 {
                        slo[blk] = l;
                        shi[blk] = h;
                    } else {
                        slo[blk] = _mm512_add_epi8(slo[blk], l);
                        shi[blk] = _mm512_add_epi8(shi[blk], h);
                    }
                }
            }
            for blk in 0..NB {
                sum_a[blk] = _mm512_add_epi16(sum_a[blk], _mm512_maddubs_epi16(combine, _mm512_unpacklo_epi8(slo[blk], shi[blk])));
                sum_b[blk] = _mm512_add_epi16(sum_b[blk], _mm512_maddubs_epi16(combine, _mm512_unpackhi_epi8(slo[blk], shi[blk])));
            }
            ci += n;
        }
        for blk in 0..NB {
            a0[blk] = _mm512_add_epi32(a0[blk], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_a[blk])));
            a1[blk] = _mm512_add_epi32(a1[blk], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_a[blk], 1)));
            b0[blk] = _mm512_add_epi32(b0[blk], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_b[blk])));
            b1[blk] = _mm512_add_epi32(b1[blk], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_b[blk], 1)));
        }
        c += chunk_len;
    }
    for blk in 0..NB {
        put_block_rows(a0[blk], a1[blk], b0[blk], b1[blk], out.add(blk * 64));
    }
}

/// Runs [`compact_multi_block`] over `nblocks` blocks in groups of `NB`; a trailing group of fewer blocks uses `NB = 1`.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn gemv_compact_multi<const NB: usize, const PF: usize, const HINT: i32, const PN: usize, const PERM: bool>(
    weights: *const u8,
    lo: *const i8,
    hi: *const i8,
    nblocks: usize,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let stride = 64 * bytes_per_row;
    let mut b = 0;
    while b + NB <= nblocks {
        compact_multi_block::<NB, PF, HINT, PN, PERM>(weights.add(b * stride), stride, lo, hi, bytes_per_row, out.add(b * 64));
        b += NB;
    }
    while b < nblocks {
        compact_multi_block::<1, PF, HINT, PN, PERM>(weights.add(b * stride), stride, lo, hi, bytes_per_row, out.add(b * 64));
        b += 1;
    }
}

/// Experimental driver for the multi-block kernels (compact tables from [`build_activation_lut_wide_avx512`]).
///
/// | id | blocks per pass | decode | far prefetch | near prefetch |
/// |---:|---:|---|---|---|
/// | 30 | 1 | shuffle | 4 KiB T0 | - |
/// | 31 | 2 | shuffle | 4 KiB T0 | - |
/// | 32 | 2 | shuffle | 2 KiB T0 | - |
/// | 33 | 2 | shuffle | 8 KiB T0 | - |
/// | 34 | 3 | shuffle | 4 KiB T0 | - |
/// | 35 | 4 | shuffle | 4 KiB T0 | - |
/// | 36 | 2 | shuffle | none | - |
/// | 37 | 2 | arithmetic | 4 KiB T0 | - |
#[doc(hidden)]
pub fn gemv_lut_64_wide_1xn_avx512_multi_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % 64, 0, "rows must be a multiple of 64");
    assert!(out.len() >= rows);
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols), "expected a compact wide table");
    require_avx512();
    let bytes_per_row = tile_row_bytes(cols);
    assert!(transformed_weights.len() >= rows * bytes_per_row);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    type KFn = unsafe fn(*const u8, *const i8, *const i8, usize, usize, *mut f32);
    const T0: i32 = _MM_HINT_T0;
    let f: KFn = match variant {
        30 => gemv_compact_multi::<1, 4096, T0, 0, true>,
        31 => gemv_compact_multi::<2, 4096, T0, 0, true>,
        32 => gemv_compact_multi::<2, 2048, T0, 0, true>,
        33 => gemv_compact_multi::<2, 8192, T0, 0, true>,
        34 => gemv_compact_multi::<3, 4096, T0, 0, true>,
        35 => gemv_compact_multi::<4, 4096, T0, 0, true>,
        36 => gemv_compact_multi::<2, 0, T0, 0, true>,
        37 => gemv_compact_multi::<2, 4096, T0, 0, false>,
        _ => panic!("unknown AVX-512 multi-block variant {variant}"),
    };
    unsafe { f(transformed_weights.as_ptr(), lo.as_ptr(), hi.as_ptr(), rows / 64, bytes_per_row, out.as_mut_ptr()) };
}

/// Upper-bound experiment for the outlier split: a kernel that looks up a **single** table (no `lo` / `hi` split) and
/// widens each pair of columns to int16, as a single-table kernel with a sparse outlier correction would have to.
/// It is exact only for activations with `|x| <= 25` (every sum of five products then fits int8); it exists to measure
/// the best speed such a design could reach. `NB` blocks share every pass over the table, like the production kernel.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn single_table_multi_block<const NB: usize, const PF: usize>(
    w: *const u8,
    block_stride: usize,
    table: *const i8,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let th = _mm512_loadu_si512(DECODE_H34.as_ptr() as *const __m512i);
    let tm = _mm512_loadu_si512(DECODE_M28.as_ptr() as *const __m512i);
    let ones = _mm512_set1_epi8(1);
    let zero = _mm512_setzero_si512();
    let mut a0 = [zero; NB];
    let mut a1 = [zero; NB];
    let mut b0 = [zero; NB];
    let mut b1 = [zero; NB];

    const CHUNK: usize = 128; // 64 pairs, each pair adds at most 250 to an int16 lane
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut sum_a = [zero; NB];
        let mut sum_b = [zero; NB];
        let mut ci = 0;
        while ci < chunk_len {
            let n = (chunk_len - ci).min(2);
            if PF != 0 {
                for blk in 0..NB {
                    let base = w.add(blk * block_stride + (c + ci) * 64);
                    _mm_prefetch::<{ _MM_HINT_T0 }>(base.add(PF) as *const i8);
                    _mm_prefetch::<{ _MM_HINT_T0 }>(base.add(PF + 64) as *const i8);
                }
            }
            let col = c + ci;
            let t0 = _mm512_loadu_si512(table.add(col * 64) as *const __m512i);
            let t1 = if n == 2 { _mm512_loadu_si512(table.add((col + 1) * 64) as *const __m512i) } else { zero };
            for blk in 0..NB {
                let wp = w.add(blk * block_stride + col * 64);
                let (h0, r0) = decode_compact::<true>(_mm512_loadu_si512(wp as *const __m512i), th, tm);
                let s0 = _mm512_add_epi8(_mm512_permutexvar_epi8(h0, t0), _mm512_permutexvar_epi8(r0, t0));
                let s1 = if n == 2 {
                    let (h1, r1) = decode_compact::<true>(_mm512_loadu_si512(wp.add(64) as *const __m512i), th, tm);
                    _mm512_add_epi8(_mm512_permutexvar_epi8(h1, t1), _mm512_permutexvar_epi8(r1, t1))
                } else {
                    zero
                };
                sum_a[blk] = _mm512_add_epi16(sum_a[blk], _mm512_maddubs_epi16(ones, _mm512_unpacklo_epi8(s0, s1)));
                sum_b[blk] = _mm512_add_epi16(sum_b[blk], _mm512_maddubs_epi16(ones, _mm512_unpackhi_epi8(s0, s1)));
            }
            ci += n;
        }
        for blk in 0..NB {
            a0[blk] = _mm512_add_epi32(a0[blk], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_a[blk])));
            a1[blk] = _mm512_add_epi32(a1[blk], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_a[blk], 1)));
            b0[blk] = _mm512_add_epi32(b0[blk], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum_b[blk])));
            b1[blk] = _mm512_add_epi32(b1[blk], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum_b[blk], 1)));
        }
        c += chunk_len;
    }
    for blk in 0..NB {
        put_block_rows(a0[blk], a1[blk], b0[blk], b1[blk], out.add(blk * 64));
    }
}

/// Driver of [`single_table_multi_block`] over `nblocks` blocks (groups of `NB`, then single blocks).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn gemv_single_table<const NB: usize, const PF: usize>(
    weights: *const u8,
    table: *const i8,
    nblocks: usize,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let stride = 64 * bytes_per_row;
    let mut b = 0;
    while b + NB <= nblocks {
        single_table_multi_block::<NB, PF>(weights.add(b * stride), stride, table, bytes_per_row, out.add(b * 64));
        b += NB;
    }
    while b < nblocks {
        single_table_multi_block::<1, PF>(weights.add(b * stride), stride, table, bytes_per_row, out.add(b * 64));
        b += 1;
    }
}

/// Experimental single-table driver (upper bound of the outlier-split idea, see [`single_table_multi_block`]). Uses the
/// `lo` half of a wide compact table as the single table (so it computes `W lo`); it has the operation count of a single-table kernel.
/// Variants `40` (1 block per pass) and `41` (3 blocks per pass), 4 KiB prefetch.
#[doc(hidden)]
pub fn gemv_lut_64_single_table_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(rows % 64, 0);
    assert!(out.len() >= rows);
    assert_eq!(wide_qlut.len(), wide_activation_lut_len_avx512(cols));
    require_avx512();
    let bytes_per_row = tile_row_bytes(cols);
    assert!(transformed_weights.len() >= rows * bytes_per_row);
    let table = wide_qlut.as_ptr();
    unsafe {
        match variant {
            40 => gemv_single_table::<1, 4096>(transformed_weights.as_ptr(), table, rows / 64, bytes_per_row, out.as_mut_ptr()),
            41 => gemv_single_table::<3, 4096>(transformed_weights.as_ptr(), table, rows / 64, bytes_per_row, out.as_mut_ptr()),
            _ => panic!("unknown single-table variant {variant}"),
        }
    }
}
