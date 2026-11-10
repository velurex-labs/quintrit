#![allow(unsafe_op_in_unsafe_fn)]

//! 32-Row Tiled Fused Dual-Hash Activation-LUT GEMV AVX2 Kernel.
//!
//! Evaluates a 32-row block of weights against the precomputed Dual-Hash QLUT
//! on x86_64 architectures using AVX2 SIMD intrinsics.

use core::arch::x86_64::*;

/// Evaluates a 32-row block of weights against the precomputed Dual-Hash QLUT
/// using a 2-column unrolled loop and pairwise vpmaddubsw reduction.
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_lut_block32_avx2(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let c_ones = _mm256_set1_epi8(1);
    let c_0f = _mm256_set1_epi8(0x0F);
    let c_3f = _mm256_set1_epi8(0x3F);
    let c_1f = _mm256_set1_epi8(0x1F);
    let c_5 = _mm256_set1_epi8(5);
    let c_9 = _mm256_set1_epi8(9);

    let a2_ptr = qlut.add(bytes_per_row * 64);

    let mut acc0 = _mm256_setzero_si256(); // Rows 0..7
    let mut acc1 = _mm256_setzero_si256(); // Rows 8..15
    let mut acc2 = _mm256_setzero_si256(); // Rows 16..23
    let mut acc3 = _mm256_setzero_si256(); // Rows 24..31

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(16);
        let mut sum_r0_r16 = _mm256_setzero_si256();
        let mut sum_r8_r24 = _mm256_setzero_si256();

        let mut ci = 0;
        while ci + 1 < chunk_len {
            let cur_c0 = c + ci;
            let cur_c1 = c + ci + 1;

            // Column 0 Evaluation: Dual-Hash + ALU
            let vec_x0 = _mm256_loadu_si256(block_weights.add(cur_c0 * 32) as *const __m256i);
            let v11_0 = _mm256_and_si256(_mm256_srli_epi16(vec_x0, 2), c_3f);
            let v15_0 = _mm256_and_si256(_mm256_srli_epi16(v11_0, 3), c_1f);
            let v17_0 = _mm256_avg_epu8(v11_0, v15_0);
            let h34_0 = _mm256_and_si256(_mm256_srli_epi16(v17_0, 2), c_0f);
            let p34_0 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c0 * 64) as *const __m256i),
                h34_0,
            );

            let h01_0 = _mm256_and_si256(
                _mm256_sub_epi8(_mm256_sub_epi8(_mm256_add_epi8(vec_x0, vec_x0), v11_0), h34_0),
                c_0f,
            );
            let p01_0 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c0 * 64 + 32) as *const __m256i),
                h01_0,
            );

            let xm1_0 = _mm256_sub_epi8(vec_x0, c_ones);
            let u_0 = _mm256_and_si256(_mm256_avg_epu8(xm1_0, v17_0), c_0f);
            let is_lo_0 = _mm256_cmpgt_epi8(c_5, u_0);
            let is_hi_0 = _mm256_cmpgt_epi8(u_0, c_9);
            let t2_0 = _mm256_sub_epi8(is_lo_0, is_hi_0);
            let a2_vec_0 = _mm256_set1_epi8(*a2_ptr.add(cur_c0));
            let p2_0 = _mm256_sign_epi8(a2_vec_0, t2_0);

            // Column 1 Evaluation: Dual-Hash + ALU
            let vec_x1 = _mm256_loadu_si256(block_weights.add(cur_c1 * 32) as *const __m256i);
            let v11_1 = _mm256_and_si256(_mm256_srli_epi16(vec_x1, 2), c_3f);
            let v15_1 = _mm256_and_si256(_mm256_srli_epi16(v11_1, 3), c_1f);
            let v17_1 = _mm256_avg_epu8(v11_1, v15_1);
            let h34_1 = _mm256_and_si256(_mm256_srli_epi16(v17_1, 2), c_0f);
            let p34_1 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c1 * 64) as *const __m256i),
                h34_1,
            );

            let h01_1 = _mm256_and_si256(
                _mm256_sub_epi8(_mm256_sub_epi8(_mm256_add_epi8(vec_x1, vec_x1), v11_1), h34_1),
                c_0f,
            );
            let p01_1 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c1 * 64 + 32) as *const __m256i),
                h01_1,
            );

            let xm1_1 = _mm256_sub_epi8(vec_x1, c_ones);
            let u_1 = _mm256_and_si256(_mm256_avg_epu8(xm1_1, v17_1), c_0f);
            let is_lo_1 = _mm256_cmpgt_epi8(c_5, u_1);
            let is_hi_1 = _mm256_cmpgt_epi8(u_1, c_9);
            let t2_1 = _mm256_sub_epi8(is_lo_1, is_hi_1);
            let a2_vec_1 = _mm256_set1_epi8(*a2_ptr.add(cur_c1));
            let p2_1 = _mm256_sign_epi8(a2_vec_1, t2_1);

            // Pairwise 16-Bit Reduction
            let unpck_34_lo = _mm256_unpacklo_epi8(p34_0, p34_1);
            let unpck_34_hi = _mm256_unpackhi_epi8(p34_0, p34_1);
            let madd_34_lo = _mm256_maddubs_epi16(c_ones, unpck_34_lo);
            let madd_34_hi = _mm256_maddubs_epi16(c_ones, unpck_34_hi);

            let unpck_01_lo = _mm256_unpacklo_epi8(p01_0, p01_1);
            let unpck_01_hi = _mm256_unpackhi_epi8(p01_0, p01_1);
            let madd_01_lo = _mm256_maddubs_epi16(c_ones, unpck_01_lo);
            let madd_01_hi = _mm256_maddubs_epi16(c_ones, unpck_01_hi);

            let unpck_2_lo = _mm256_unpacklo_epi8(p2_0, p2_1);
            let unpck_2_hi = _mm256_unpackhi_epi8(p2_0, p2_1);
            let madd_2_lo = _mm256_maddubs_epi16(c_ones, unpck_2_lo);
            let madd_2_hi = _mm256_maddubs_epi16(c_ones, unpck_2_hi);

            let sum_lo = _mm256_add_epi16(_mm256_add_epi16(madd_34_lo, madd_01_lo), madd_2_lo);
            let sum_hi = _mm256_add_epi16(_mm256_add_epi16(madd_34_hi, madd_01_hi), madd_2_hi);

            sum_r0_r16 = _mm256_add_epi16(sum_r0_r16, sum_lo);
            sum_r8_r24 = _mm256_add_epi16(sum_r8_r24, sum_hi);

            ci += 2;
        }

        // Remainder column (if chunk_len is odd)
        if ci < chunk_len {
            let cur_c0 = c + ci;
            let vec_x0 = _mm256_loadu_si256(block_weights.add(cur_c0 * 32) as *const __m256i);
            let v11_0 = _mm256_and_si256(_mm256_srli_epi16(vec_x0, 2), c_3f);
            let v15_0 = _mm256_and_si256(_mm256_srli_epi16(v11_0, 3), c_1f);
            let v17_0 = _mm256_avg_epu8(v11_0, v15_0);
            let h34_0 = _mm256_and_si256(_mm256_srli_epi16(v17_0, 2), c_0f);
            let p34_0 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c0 * 64) as *const __m256i),
                h34_0,
            );

            let h01_0 = _mm256_and_si256(
                _mm256_sub_epi8(_mm256_sub_epi8(_mm256_add_epi8(vec_x0, vec_x0), v11_0), h34_0),
                c_0f,
            );
            let p01_0 = _mm256_shuffle_epi8(
                _mm256_loadu_si256(qlut.add(cur_c0 * 64 + 32) as *const __m256i),
                h01_0,
            );

            let xm1_0 = _mm256_sub_epi8(vec_x0, c_ones);
            let u_0 = _mm256_and_si256(_mm256_avg_epu8(xm1_0, v17_0), c_0f);
            let is_lo_0 = _mm256_cmpgt_epi8(c_5, u_0);
            let is_hi_0 = _mm256_cmpgt_epi8(u_0, c_9);
            let t2_0 = _mm256_sub_epi8(is_lo_0, is_hi_0);
            let a2_vec_0 = _mm256_set1_epi8(*a2_ptr.add(cur_c0));
            let p2_0 = _mm256_sign_epi8(a2_vec_0, t2_0);

            let zero = _mm256_setzero_si256();
            let unpck_34_lo = _mm256_unpacklo_epi8(p34_0, zero);
            let unpck_34_hi = _mm256_unpackhi_epi8(p34_0, zero);
            let madd_34_lo = _mm256_maddubs_epi16(c_ones, unpck_34_lo);
            let madd_34_hi = _mm256_maddubs_epi16(c_ones, unpck_34_hi);

            let unpck_01_lo = _mm256_unpacklo_epi8(p01_0, zero);
            let unpck_01_hi = _mm256_unpackhi_epi8(p01_0, zero);
            let madd_01_lo = _mm256_maddubs_epi16(c_ones, unpck_01_lo);
            let madd_01_hi = _mm256_maddubs_epi16(c_ones, unpck_01_hi);

            let unpck_2_lo = _mm256_unpacklo_epi8(p2_0, zero);
            let unpck_2_hi = _mm256_unpackhi_epi8(p2_0, zero);
            let madd_2_lo = _mm256_maddubs_epi16(c_ones, unpck_2_lo);
            let madd_2_hi = _mm256_maddubs_epi16(c_ones, unpck_2_hi);

            let sum_lo = _mm256_add_epi16(_mm256_add_epi16(madd_34_lo, madd_01_lo), madd_2_lo);
            let sum_hi = _mm256_add_epi16(_mm256_add_epi16(madd_34_hi, madd_01_hi), madd_2_hi);

            sum_r0_r16 = _mm256_add_epi16(sum_r0_r16, sum_lo);
            sum_r8_r24 = _mm256_add_epi16(sum_r8_r24, sum_hi);
        }

        // Chunk Boundary: Widen into 32-bit accumulators
        acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r0_r16)));
        acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r8_r24)));
        acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r0_r16, 1)));
        acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r8_r24, 1)));

        c += chunk_len;
    }

    // Convert to f32 and store to output
    _mm256_storeu_ps(out_32, _mm256_cvtepi32_ps(acc0));
    _mm256_storeu_ps(out_32.add(8), _mm256_cvtepi32_ps(acc1));
    _mm256_storeu_ps(out_32.add(16), _mm256_cvtepi32_ps(acc2));
    _mm256_storeu_ps(out_32.add(24), _mm256_cvtepi32_ps(acc3));
}


// ============================================================================
// Fused hi/lo ("wide") kernel
// ============================================================================

/// Weight-dependent part of one column: the table indices and the middle-trit sign selector.
/// They are identical for the `lo` and `hi` activation tables, so they are computed once.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_column(vec_x: __m256i) -> (__m256i, __m256i, __m256i) {
    let c_ones = _mm256_set1_epi8(1);
    let c_0f = _mm256_set1_epi8(0x0F);
    let c_3f = _mm256_set1_epi8(0x3F);
    let c_1f = _mm256_set1_epi8(0x1F);
    let c_5 = _mm256_set1_epi8(5);
    let c_9 = _mm256_set1_epi8(9);

    let v11 = _mm256_and_si256(_mm256_srli_epi16(vec_x, 2), c_3f);
    let v15 = _mm256_and_si256(_mm256_srli_epi16(v11, 3), c_1f);
    let v17 = _mm256_avg_epu8(v11, v15);
    let h34 = _mm256_and_si256(_mm256_srli_epi16(v17, 2), c_0f);
    let h01 = _mm256_and_si256(
        _mm256_sub_epi8(_mm256_sub_epi8(_mm256_add_epi8(vec_x, vec_x), v11), h34),
        c_0f,
    );
    let xm1 = _mm256_sub_epi8(vec_x, c_ones);
    let u = _mm256_and_si256(_mm256_avg_epu8(xm1, v17), c_0f);
    let is_lo = _mm256_cmpgt_epi8(c_5, u);
    let is_hi = _mm256_cmpgt_epi8(u, c_9);
    let t2 = _mm256_sub_epi8(is_lo, is_hi);
    (h34, h01, t2)
}

/// Table lookups of one column against one activation table: the three per-row products.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn column_products(
    qlut: *const i8,
    a2_ptr: *const i8,
    c: usize,
    h34: __m256i,
    h01: __m256i,
    t2: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let p34 = _mm256_shuffle_epi8(_mm256_loadu_si256(qlut.add(c * 64) as *const __m256i), h34);
    let p01 = _mm256_shuffle_epi8(_mm256_loadu_si256(qlut.add(c * 64 + 32) as *const __m256i), h01);
    let p2 = _mm256_sign_epi8(_mm256_set1_epi8(*a2_ptr.add(c)), t2);
    (p34, p01, p2)
}

/// Pairwise 16-bit reduction of the products of two columns (rows 0..7/16..23 and 8..15/24..31).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce_pair(
    p34_0: __m256i,
    p34_1: __m256i,
    p01_0: __m256i,
    p01_1: __m256i,
    p2_0: __m256i,
    p2_1: __m256i,
) -> (__m256i, __m256i) {
    let c_ones = _mm256_set1_epi8(1);
    let madd_34_lo = _mm256_maddubs_epi16(c_ones, _mm256_unpacklo_epi8(p34_0, p34_1));
    let madd_34_hi = _mm256_maddubs_epi16(c_ones, _mm256_unpackhi_epi8(p34_0, p34_1));
    let madd_01_lo = _mm256_maddubs_epi16(c_ones, _mm256_unpacklo_epi8(p01_0, p01_1));
    let madd_01_hi = _mm256_maddubs_epi16(c_ones, _mm256_unpackhi_epi8(p01_0, p01_1));
    let madd_2_lo = _mm256_maddubs_epi16(c_ones, _mm256_unpacklo_epi8(p2_0, p2_1));
    let madd_2_hi = _mm256_maddubs_epi16(c_ones, _mm256_unpackhi_epi8(p2_0, p2_1));
    (
        _mm256_add_epi16(_mm256_add_epi16(madd_34_lo, madd_01_lo), madd_2_lo),
        _mm256_add_epi16(_mm256_add_epi16(madd_34_hi, madd_01_hi), madd_2_hi),
    )
}

/// Fused variant of [`gemv_lut_block32_avx2`] for a wide activation table (`x = 16 * hi + lo`).
///
/// `qlut_lo` and `qlut_hi` are the single tables of the two activation parts. The weights are decoded
/// once per column; both tables are looked up with the same indices and the 16-bit partial sums are
/// combined as `lo + 16 * hi` before widening. The output is the exact integer `W lo + 16 * (W hi)`.
/// Per column the combined magnitude is at most 40 + 16 * 40, so 16 columns fit into int16.
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_lut_block32_wide_avx2(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let a2_lo = qlut_lo.add(bytes_per_row * 64);
    let a2_hi = qlut_hi.add(bytes_per_row * 64);
    let zero = _mm256_setzero_si256();

    let mut acc0 = _mm256_setzero_si256(); // Rows 0..7
    let mut acc1 = _mm256_setzero_si256(); // Rows 8..15
    let mut acc2 = _mm256_setzero_si256(); // Rows 16..23
    let mut acc3 = _mm256_setzero_si256(); // Rows 24..31

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(16);
        let mut sum_r0_r16 = _mm256_setzero_si256();
        let mut sum_r8_r24 = _mm256_setzero_si256();

        let mut ci = 0;
        while ci < chunk_len {
            let c0 = c + ci;
            let paired = ci + 1 < chunk_len;
            let c1 = if paired { c0 + 1 } else { c0 };

            let (h34_0, h01_0, t2_0) = decode_column(_mm256_loadu_si256(block_weights.add(c0 * 32) as *const __m256i));
            let (lo34_0, lo01_0, lo2_0) = column_products(qlut_lo, a2_lo, c0, h34_0, h01_0, t2_0);
            let (hi34_0, hi01_0, hi2_0) = column_products(qlut_hi, a2_hi, c0, h34_0, h01_0, t2_0);

            let (lo_a, lo_b, hi_a, hi_b);
            if paired {
                let (h34_1, h01_1, t2_1) =
                    decode_column(_mm256_loadu_si256(block_weights.add(c1 * 32) as *const __m256i));
                let (lo34_1, lo01_1, lo2_1) = column_products(qlut_lo, a2_lo, c1, h34_1, h01_1, t2_1);
                let (hi34_1, hi01_1, hi2_1) = column_products(qlut_hi, a2_hi, c1, h34_1, h01_1, t2_1);
                (lo_a, lo_b) = reduce_pair(lo34_0, lo34_1, lo01_0, lo01_1, lo2_0, lo2_1);
                (hi_a, hi_b) = reduce_pair(hi34_0, hi34_1, hi01_0, hi01_1, hi2_0, hi2_1);
            } else {
                (lo_a, lo_b) = reduce_pair(lo34_0, zero, lo01_0, zero, lo2_0, zero);
                (hi_a, hi_b) = reduce_pair(hi34_0, zero, hi01_0, zero, hi2_0, zero);
            }

            // lo + 16 * hi, still in 16 bits
            sum_r0_r16 = _mm256_add_epi16(sum_r0_r16, _mm256_add_epi16(lo_a, _mm256_slli_epi16(hi_a, 4)));
            sum_r8_r24 = _mm256_add_epi16(sum_r8_r24, _mm256_add_epi16(lo_b, _mm256_slli_epi16(hi_b, 4)));

            ci += 2;
        }

        acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r0_r16)));
        acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r8_r24)));
        acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r0_r16, 1)));
        acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r8_r24, 1)));

        c += chunk_len;
    }

    _mm256_storeu_ps(out_32, _mm256_cvtepi32_ps(acc0));
    _mm256_storeu_ps(out_32.add(8), _mm256_cvtepi32_ps(acc1));
    _mm256_storeu_ps(out_32.add(16), _mm256_cvtepi32_ps(acc2));
    _mm256_storeu_ps(out_32.add(24), _mm256_cvtepi32_ps(acc3));
}

// ============================================================================
// Experimental variants of the fused hi/lo kernel
// ============================================================================

/// Per-column sums of the three products for the `lo` and the `hi` table in int8
/// (`|sum| <= 40` per column because both activation parts have magnitude <= 8).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn column_sums_i8(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    a2_lo: *const i8,
    a2_hi: *const i8,
    c: usize,
) -> (__m256i, __m256i) {
    let (h34, h01, t2) = decode_column(_mm256_loadu_si256(block_weights.add(c * 32) as *const __m256i));
    let (lo34, lo01, lo2) = column_products(qlut_lo, a2_lo, c, h34, h01, t2);
    let (hi34, hi01, hi2) = column_products(qlut_hi, a2_hi, c, h34, h01, t2);
    (
        _mm256_add_epi8(_mm256_add_epi8(lo34, lo01), lo2),
        _mm256_add_epi8(_mm256_add_epi8(hi34, hi01), hi2),
    )
}

/// Variant `i8red` of [`gemv_lut_block32_wide_avx2`]: the products of up to three columns are added in
/// int8 (the sum stays within +-120) and widened once per group; `lo + 16 * hi` is formed by a single
/// `vpmaddubsw` with the weights (1, 16) on the interleaved bytes of the two sums. The result is identical.
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_lut_block32_wide_avx2_i8red(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    gemv_lut_block32_wide_avx2_i8red_impl::<0>(block_weights, qlut_lo, qlut_hi, bytes_per_row, out_32)
}

/// `i8red` with a software prefetch `PF` bytes ahead of the weight stream (0 = none).
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_lut_block32_wide_avx2_i8red_impl<const PF: usize>(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let a2_lo = qlut_lo.add(bytes_per_row * 64);
    let a2_hi = qlut_hi.add(bytes_per_row * 64);
    let combine = _mm256_set1_epi16(0x1001); // bytes (1, 16)

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    // 15 groups of three columns per chunk: |lo + 16 hi| <= 2040 per group, 15 * 2040 < 32768.
    const CHUNK: usize = 45;
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut sum_r0_r16 = _mm256_setzero_si256();
        let mut sum_r8_r24 = _mm256_setzero_si256();

        let mut ci = 0;
        while ci + 3 <= chunk_len {
            if PF != 0 {
                let ahead = block_weights.add((c + ci) * 32 + PF);
                _mm_prefetch(ahead as *const i8, _MM_HINT_T0);
                _mm_prefetch(ahead.add(64) as *const i8, _MM_HINT_T0);
            }
            let (l0, h0) = column_sums_i8(block_weights, qlut_lo, qlut_hi, a2_lo, a2_hi, c + ci);
            let (l1, h1) = column_sums_i8(block_weights, qlut_lo, qlut_hi, a2_lo, a2_hi, c + ci + 1);
            let (l2, h2) = column_sums_i8(block_weights, qlut_lo, qlut_hi, a2_lo, a2_hi, c + ci + 2);
            let slo = _mm256_add_epi8(_mm256_add_epi8(l0, l1), l2);
            let shi = _mm256_add_epi8(_mm256_add_epi8(h0, h1), h2);
            sum_r0_r16 = _mm256_add_epi16(sum_r0_r16, _mm256_maddubs_epi16(combine, _mm256_unpacklo_epi8(slo, shi)));
            sum_r8_r24 = _mm256_add_epi16(sum_r8_r24, _mm256_maddubs_epi16(combine, _mm256_unpackhi_epi8(slo, shi)));
            ci += 3;
        }
        if ci < chunk_len {
            let (mut slo, mut shi) = column_sums_i8(block_weights, qlut_lo, qlut_hi, a2_lo, a2_hi, c + ci);
            if ci + 1 < chunk_len {
                let (l, h) = column_sums_i8(block_weights, qlut_lo, qlut_hi, a2_lo, a2_hi, c + ci + 1);
                slo = _mm256_add_epi8(slo, l);
                shi = _mm256_add_epi8(shi, h);
            }
            sum_r0_r16 = _mm256_add_epi16(sum_r0_r16, _mm256_maddubs_epi16(combine, _mm256_unpacklo_epi8(slo, shi)));
            sum_r8_r24 = _mm256_add_epi16(sum_r8_r24, _mm256_maddubs_epi16(combine, _mm256_unpackhi_epi8(slo, shi)));
        }

        acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r0_r16)));
        acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_r8_r24)));
        acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r0_r16, 1)));
        acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_r8_r24, 1)));

        c += chunk_len;
    }

    _mm256_storeu_ps(out_32, _mm256_cvtepi32_ps(acc0));
    _mm256_storeu_ps(out_32.add(8), _mm256_cvtepi32_ps(acc1));
    _mm256_storeu_ps(out_32.add(16), _mm256_cvtepi32_ps(acc2));
    _mm256_storeu_ps(out_32.add(24), _mm256_cvtepi32_ps(acc3));
}


// ============================================================================
// Vectorized activation-table builder for small activation parts
// ============================================================================

/// Builds the single activation table (64 bytes per column plus the `a2` tail array, see `build_activation_lut_into`)
/// for activations of magnitude <= 63, such as the `lo` / `hi` parts of the wide tables. The output is identical to
/// the scalar builder for such inputs; the table entries are formed with `vpsignb` and a saturating add that cannot
/// saturate in this range.
///
/// # Safety
/// Requires AVX2; `qlut` must hold `wide`-table length of one part (`bytes_per_row * 64 + bytes_per_row.next_multiple_of(32)`).
#[target_feature(enable = "avx2")]
pub unsafe fn build_activation_lut_small_avx2(acts: &[i8], cols: usize, qlut: &mut [i8]) {
    use crate::lut::{MAP_01, MAP_34};
    use crate::pack::tile_row_bytes;

    let bytes_per_row = tile_row_bytes(cols);
    let a2_offset = bytes_per_row * 64;
    debug_assert!(qlut.len() >= a2_offset + bytes_per_row.next_multiple_of(32));

    // Coefficient vectors: the 16 slots of a table in both 128-bit lanes.
    let mut c3 = [0i8; 32];
    let mut c4 = [0i8; 32];
    let mut c0 = [0i8; 32];
    let mut c1 = [0i8; 32];
    for s in 0..16 {
        c3[s] = MAP_34[s].0;
        c3[s + 16] = MAP_34[s].0;
        c4[s] = MAP_34[s].1;
        c4[s + 16] = MAP_34[s].1;
        c0[s] = MAP_01[s].0;
        c0[s + 16] = MAP_01[s].0;
        c1[s] = MAP_01[s].1;
        c1[s + 16] = MAP_01[s].1;
    }
    let (c3, c4, c0, c1) = (
        _mm256_loadu_si256(c3.as_ptr() as *const __m256i),
        _mm256_loadu_si256(c4.as_ptr() as *const __m256i),
        _mm256_loadu_si256(c0.as_ptr() as *const __m256i),
        _mm256_loadu_si256(c1.as_ptr() as *const __m256i),
    );

    let out = qlut.as_mut_ptr();
    for c in 0..bytes_per_row {
        let get = |p: usize| -> i8 {
            let idx = crate::pack::column_activation_index(cols, c, p);
            if idx < cols { *acts.get_unchecked(idx) } else { 0 }
        };
        let (a0, a1, a2, a3, a4) = (get(0), get(1), get(2), get(3), get(4));
        let t34 = _mm256_adds_epi8(
            _mm256_sign_epi8(_mm256_set1_epi8(a3), c3),
            _mm256_sign_epi8(_mm256_set1_epi8(a4), c4),
        );
        let t01 = _mm256_adds_epi8(
            _mm256_sign_epi8(_mm256_set1_epi8(a0), c0),
            _mm256_sign_epi8(_mm256_set1_epi8(a1), c1),
        );
        _mm256_storeu_si256(out.add(c * 64) as *mut __m256i, t34);
        _mm256_storeu_si256(out.add(c * 64 + 32) as *mut __m256i, t01);
        *out.add(a2_offset + c) = a2;
    }
    // The padding of the tail array (the column count is not a multiple of 32 with a packed tail) is zeroed like in the scalar builder.
    for i in bytes_per_row..bytes_per_row.next_multiple_of(32) {
        *out.add(a2_offset + i) = 0;
    }
}
