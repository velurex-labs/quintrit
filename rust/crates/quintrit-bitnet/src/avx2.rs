//! Optimized AVX2 SIMD implementation for 4 trits per byte (256-bit vectors).
//!
//! Strict 1:1 Reference Port of Microsoft BitNet AVX2 kernels (`ggml-bitnet-lut.cpp` / T-MAC).
//!
//! Includes:
//! - High-performance AVX2 dot product for ternary weights and int8 activations
//!   with SIMD LUT dequantization and pure-register horizontal reduction.
//! - AVX2 multi-row GEMV (Matrix-Vector) kernel leveraging `_mm256_shuffle_epi8` (`vpshufb`)
//!   against precomputed activation lookup tables (inspired by BitNet / T-MAC).

use core::arch::x86_64::*;

/// Horizontally adds 8 signed 32-bit integers in a 256-bit AVX2 register entirely in registers.
///
/// Eliminates stack writes and memory store-to-load forwarding penalties (BitNet tree reduction).
#[inline(always)]
pub unsafe fn hsum_i32_avx2(a: __m256i) -> i64 {
    unsafe {
        let sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(hi64, sum128);
        let hi32 = _mm_shuffle_epi32(sum64, 0b10_11_00_01);
        let sum32 = _mm_add_epi32(sum64, hi32);
        _mm_cvtsi128_si32(sum32) as i64
    }
}

/// Computes the dot product of ternary weights and int8 activations (Ternary x Int8) using AVX2.
///
/// Processes 8 packed bytes of weights (32 trits and 32 activations) per 256-bit iteration.
/// Uses hardware byte shuffle lookup (`_mm256_shuffle_epi8`) to unpack trits to weights
/// without scalar loops or stack allocations.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_ternary_int8_avx2(w: &[u8], x: &[i8], trit_count: usize) -> i64 {
    let num_bytes = trit_count / 4;
    let chunks = num_bytes / 8;
    let mut total: i64;

    unsafe {
        let mut acc_32 = _mm256_setzero_si256();
        let ones_16 = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);
        let mask_03 = _mm256_set1_epi8(0x03);

        // Hardware LUT mapping code in {0, 1, 2, 3} -> weight in {-1, 0, +1, 0}:
        // code 0 -> -1, code 1 -> 0, code 2 -> +1, code 3 -> 0
        let lut_weights = _mm256_setr_epi8(
            -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0,
            -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0,
        );

        for i in 0..chunks {
            let w_u64 = core::ptr::read_unaligned(w.as_ptr().add(i * 8) as *const i64);
            let vx = _mm256_loadu_si256(x.as_ptr().add(i * 32) as *const __m256i);

            // Replicate 8 bytes across 32 lanes (4 copies of each byte):
            // Lower 128 bits replicates bytes 0, 1, 2, 3 into 16 bytes.
            // Upper 128 bits replicates bytes 4, 5, 6, 7 into 16 bytes.
            let vw_raw = _mm256_set1_epi64x(w_u64);
            let shuf_expand = _mm256_setr_epi8(
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
            );
            let b_exp = _mm256_shuffle_epi8(vw_raw, shuf_expand);

            // Extract each of the 4 trits per byte in parallel:
            // Lane index % 4 == 0 -> trit 0 (shift 0)
            // Lane index % 4 == 1 -> trit 1 (shift 2)
            // Lane index % 4 == 2 -> trit 2 (shift 4)
            // Lane index % 4 == 3 -> trit 3 (shift 6)
            let s0 = b_exp;
            let s1 = _mm256_srli_epi16(b_exp, 2);
            let s2 = _mm256_srli_epi16(b_exp, 4);
            let s3 = _mm256_srli_epi16(b_exp, 6);

            // Blend based on lane index % 4:
            // mask_even_words: select between shift 0 and shift 2
            // mask_even_dwords: select between (shift 0,2) and (shift 4,6)
            let blend_mask_01 = _mm256_setr_epi8(
                0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
                0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            );
            let t01 = _mm256_blendv_epi8(s0, s1, blend_mask_01);
            let t23 = _mm256_blendv_epi8(s2, s3, blend_mask_01);

            let blend_mask_01_23 = _mm256_setr_epi8(
                0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1,
                0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1,
            );
            let shifted = _mm256_blendv_epi8(t01, t23, blend_mask_01_23);
            let codes = _mm256_and_si256(shifted, mask_03);

            // Hardware SIMD LUT: maps codes in {0, 1, 2} to weights in {-1, 0, +1}:
            let vw = _mm256_shuffle_epi8(lut_weights, codes);

            // Multiply ternary weights {-1, 0, +1} by int8 activations vx:
            let is_pos = _mm256_cmpeq_epi8(vw, _mm256_set1_epi8(1));
            let is_neg = _mm256_cmpeq_epi8(vw, _mm256_set1_epi8(-1));

            let x_pos = _mm256_and_si256(is_pos, vx);
            let x_neg = _mm256_and_si256(is_neg, vx);

            let sum_pos_16 = _mm256_maddubs_epi16(ones_u8, x_pos);
            let sum_neg_16 = _mm256_maddubs_epi16(ones_u8, x_neg);

            let diff_16 = _mm256_sub_epi16(sum_pos_16, sum_neg_16);
            let sum_32 = _mm256_madd_epi16(diff_16, ones_16);

            acc_32 = _mm256_add_epi32(acc_32, sum_32);
        }

        total = hsum_i32_avx2(acc_32);
    }

    // Scalar epilogue for remainder
    let processed_bytes = chunks * 8;
    let processed_trits = processed_bytes * 4;
    if processed_trits < trit_count {
        total += crate::scalar::dot_ternary_int8_scalar(
            &w[processed_bytes..],
            &x[processed_trits..],
            trit_count - processed_trits,
        );
    }

    total
}

/// Computes General Matrix-Vector Multiplication (GEMV: Y = W * X) using AVX2 LUT kernels.
///
/// Leverages `_mm256_shuffle_epi8` (`vpshufb`) against precomputed 16-byte activation tables.
/// Evaluates 16 rows concurrently:
/// - Lower 128-bit lane queries Table(2b) for low nibbles (trits 0, 1) across 16 rows.
/// - Upper 128-bit lane queries Table(2b + 1) for high nibbles (trits 2, 3) across 16 rows.
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_ternary_int8_avx2(
    w_matrix: &[u8],
    x: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    let tables = crate::scalar::build_activation_lut_scalar(&x[..cols]);
    let bytes_per_row = cols.div_ceil(4);
    let full_bytes = cols / 4;
    let remainder = cols % 4;

    let tile_rows = 16;
    let full_row_tiles = rows / tile_rows;

    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);

        for t in 0..full_row_tiles {
            let r0 = t * tile_rows;

            let mut acc_lo_32 = _mm256_setzero_si256(); // rows 0..7
            let mut acc_hi_32 = _mm256_setzero_si256(); // rows 8..15

            let mut acc_lo_16 = _mm_setzero_si128();
            let mut acc_hi_16 = _mm_setzero_si128();

            for b in 0..full_bytes {
                let tbl_pair = _mm256_loadu_si256(tables.as_ptr().add((2 * b) * 16) as *const __m256i);

                let mut raw_bytes = [0u8; 16];
                for i in 0..16 {
                    raw_bytes[i] = *w_matrix.get_unchecked((r0 + i) * bytes_per_row + b);
                }
                let v_bytes = _mm_loadu_si128(raw_bytes.as_ptr() as *const __m128i);

                let v_low = _mm_and_si128(v_bytes, mask_0f);
                let v_high = _mm_and_si128(_mm_srli_epi16(v_bytes, 4), mask_0f);
                let idx_reg = _mm256_set_m128i(v_high, v_low);

                let res = _mm256_shuffle_epi8(tbl_pair, idx_reg);

                let partial = _mm_add_epi8(
                    _mm256_castsi256_si128(res),
                    _mm256_extracti128_si256(res, 1),
                );

                let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), partial);
                let lo_16 = _mm_unpacklo_epi8(partial, sign);
                let hi_16 = _mm_unpackhi_epi8(partial, sign);

                acc_lo_16 = _mm_add_epi16(acc_lo_16, lo_16);
                acc_hi_16 = _mm_add_epi16(acc_hi_16, hi_16);

                if (b + 1) % 128 == 0 || b + 1 == full_bytes {
                    acc_lo_32 = _mm256_add_epi32(acc_lo_32, _mm256_cvtepi16_epi32(acc_lo_16));
                    acc_hi_32 = _mm256_add_epi32(acc_hi_32, _mm256_cvtepi16_epi32(acc_hi_16));
                    acc_lo_16 = _mm_setzero_si128();
                    acc_hi_16 = _mm_setzero_si128();
                }
            }

            let mut buf_lo = [0i32; 8];
            let mut buf_hi = [0i32; 8];
            _mm256_storeu_si256(buf_lo.as_mut_ptr() as *mut __m256i, acc_lo_32);
            _mm256_storeu_si256(buf_hi.as_mut_ptr() as *mut __m256i, acc_hi_32);

            for i in 0..8 {
                out[r0 + i] = buf_lo[i] as i64;
            }
            for i in 0..8 {
                out[r0 + 8 + i] = buf_hi[i] as i64;
            }

            if remainder > 0 {
                let x_offset = full_bytes * 4;
                for i in 0..16 {
                    let r = r0 + i;
                    let bw = *w_matrix.get_unchecked(r * bytes_per_row + full_bytes);
                    for k in 0..remainder {
                        let cw = (bw >> (k * 2)) & 0x03;
                        let weight = match cw {
                            0 => -1i64,
                            2 => 1i64,
                            _ => 0i64,
                        };
                        out[r] += weight * (x[x_offset + k] as i64);
                    }
                }
            }
        }
    }

    let processed_rows = full_row_tiles * tile_rows;
    if processed_rows < rows {
        crate::scalar::gemv_ternary_int8_scalar(
            &w_matrix[processed_rows * bytes_per_row..],
            x,
            rows - processed_rows,
            cols,
            &mut out[processed_rows..],
        );
    }
}

/// Computes General Matrix-Vector Multiplication (GEMV: Y = W * X) using AVX2 LUT kernels
/// on a 32-row block interleaved (tiled) weight matrix with Multi-Block Register Reuse (`BM = 64`).
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_lut_tiled32_avx2(
    tiled_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    let bytes_per_row = cols.div_ceil(4);
    let full_bytes = cols / 4;
    let remainder_cols = cols % 4;
    let num_blocks = rows / 32;
    let num_pairs = num_blocks / 2;

    unsafe {
        let mask_0f = _mm256_set1_epi8(0x0f);
        let qlut_ptr = qlut.as_ptr();

        for pair in 0..num_pairs {
            let b0 = pair * 2;
            let b1 = pair * 2 + 1;
            let r0 = b0 * 32;
            let ptr0 = tiled_weights.as_ptr().add(b0 * (32 * bytes_per_row));
            let ptr1 = tiled_weights.as_ptr().add(b1 * (32 * bytes_per_row));

            let (out_0, out_1) = out[r0..r0 + 64].split_at_mut(32);
            gemv_lut_block64_avx2(
                ptr0,
                ptr1,
                qlut_ptr,
                full_bytes,
                remainder_cols,
                mask_0f,
                out_0,
                out_1,
            );
        }

        if num_blocks % 2 != 0 {
            let b = num_pairs * 2;
            let r0 = b * 32;
            let ptr = tiled_weights.as_ptr().add(b * (32 * bytes_per_row));
            gemv_lut_block32_avx2(
                ptr,
                qlut_ptr,
                full_bytes,
                remainder_cols,
                mask_0f,
                &mut out[r0..r0 + 32],
            );
        }
    }

    let processed_rows = num_blocks * 32;
    if processed_rows < rows {
        let rem_rows = rows - processed_rows;
        let rem_weights = &tiled_weights[num_blocks * (32 * bytes_per_row)..];
        for r in 0..rem_rows {
            let mut sum: i64 = 0;
            for c in 0..full_bytes {
                let bw = rem_weights[r * bytes_per_row + c];
                let idx0 = (bw & 0x0F) as usize;
                let idx1 = ((bw >> 4) & 0x0F) as usize;
                let p0 = qlut[(2 * c) * 16 + idx0] as i64;
                let p1 = qlut[(2 * c + 1) * 16 + idx1] as i64;
                sum += p0 + p1;
            }
            if remainder_cols > 0 {
                let bw = rem_weights[r * bytes_per_row + full_bytes];
                for k in 0..remainder_cols {
                    let cw = (bw >> (k * 2)) & 0x03;
                    let weight = match cw {
                        0 => -1i64,
                        2 => 1i64,
                        _ => 0i64,
                    };
                    let pair_idx = full_bytes * 2 + k / 2;
                    let pair_sub = k % 2;
                    let single_val = if pair_sub == 0 {
                        qlut[pair_idx * 16 + 6] as i64
                    } else {
                        qlut[pair_idx * 16 + 9] as i64
                    };
                    sum += weight * single_val;
                }
            }
            out[processed_rows + r] = sum;
        }
    }
}

/// Helper kernel evaluating a pair of 32-row blocks (BM = 64) with shared activation tables.
#[target_feature(enable = "avx2")]
unsafe fn gemv_lut_block64_avx2(
    block_ptr_0: *const u8,
    block_ptr_1: *const u8,
    qlut: *const i8,
    full_bytes: usize,
    remainder_cols: usize,
    mask_0f: __m256i,
    out_0: &mut [i64],
    out_1: &mut [i64],
) {
    unsafe {
        let mut acc0_0 = _mm256_setzero_si256();
        let mut acc1_0 = _mm256_setzero_si256();
        let mut acc2_0 = _mm256_setzero_si256();
        let mut acc3_0 = _mm256_setzero_si256();

        let mut acc0_1 = _mm256_setzero_si256();
        let mut acc1_1 = _mm256_setzero_si256();
        let mut acc2_1 = _mm256_setzero_si256();
        let mut acc3_1 = _mm256_setzero_si256();

        let mut c = 0;
        while c < full_bytes {
            let chunk_len = (full_bytes - c).min(64);
            let mut sum16_lo_0 = _mm256_setzero_si256();
            let mut sum16_hi_0 = _mm256_setzero_si256();
            let mut sum16_lo_1 = _mm256_setzero_si256();
            let mut sum16_hi_1 = _mm256_setzero_si256();

            for ci in 0..chunk_len {
                let cur_c = c + ci;
                let vec_a0 = _mm256_loadu_si256(block_ptr_0.add(cur_c * 32) as *const __m256i);
                let vec_a1 = _mm256_loadu_si256(block_ptr_1.add(cur_c * 32) as *const __m256i);

                let vec_bot0 = _mm256_and_si256(vec_a0, mask_0f);
                let vec_top0 = _mm256_and_si256(_mm256_srli_epi16(vec_a0, 4), mask_0f);

                let vec_bot1 = _mm256_and_si256(vec_a1, mask_0f);
                let vec_top1 = _mm256_and_si256(_mm256_srli_epi16(vec_a1, 4), mask_0f);

                let lut_p0 = _mm_loadu_si128(qlut.add((2 * cur_c) * 16) as *const __m128i);
                let lut_p1 = _mm_loadu_si128(qlut.add((2 * cur_c + 1) * 16) as *const __m128i);

                let tbl0 = _mm256_set_m128i(lut_p0, lut_p0);
                let tbl1 = _mm256_set_m128i(lut_p1, lut_p1);

                let p0_0 = _mm256_shuffle_epi8(tbl0, vec_bot0);
                let p1_0 = _mm256_shuffle_epi8(tbl1, vec_top0);

                let p0_1 = _mm256_shuffle_epi8(tbl0, vec_bot1);
                let p1_1 = _mm256_shuffle_epi8(tbl1, vec_top1);

                let p0_0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p0_0));
                let p1_0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p1_0));
                sum16_lo_0 = _mm256_add_epi16(sum16_lo_0, _mm256_add_epi16(p0_0_lo, p1_0_lo));

                let p0_0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p0_0, 1));
                let p1_0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p1_0, 1));
                sum16_hi_0 = _mm256_add_epi16(sum16_hi_0, _mm256_add_epi16(p0_0_hi, p1_0_hi));

                let p0_1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p0_1));
                let p1_1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p1_1));
                sum16_lo_1 = _mm256_add_epi16(sum16_lo_1, _mm256_add_epi16(p0_1_lo, p1_1_lo));

                let p0_1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p0_1, 1));
                let p1_1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p1_1, 1));
                sum16_hi_1 = _mm256_add_epi16(sum16_hi_1, _mm256_add_epi16(p0_1_hi, p1_1_hi));
            }

            acc0_0 = _mm256_add_epi32(acc0_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_lo_0)));
            acc1_0 = _mm256_add_epi32(acc1_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_lo_0, 1)));
            acc2_0 = _mm256_add_epi32(acc2_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_hi_0)));
            acc3_0 = _mm256_add_epi32(acc3_0, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_hi_0, 1)));

            acc0_1 = _mm256_add_epi32(acc0_1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_lo_1)));
            acc1_1 = _mm256_add_epi32(acc1_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_lo_1, 1)));
            acc2_1 = _mm256_add_epi32(acc2_1, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_hi_1)));
            acc3_1 = _mm256_add_epi32(acc3_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_hi_1, 1)));

            c += chunk_len;
        }

        let mut buf0 = [0i32; 8];
        let mut buf1 = [0i32; 8];
        let mut buf2 = [0i32; 8];
        let mut buf3 = [0i32; 8];

        _mm256_storeu_si256(buf0.as_mut_ptr() as *mut __m256i, acc0_0);
        _mm256_storeu_si256(buf1.as_mut_ptr() as *mut __m256i, acc1_0);
        _mm256_storeu_si256(buf2.as_mut_ptr() as *mut __m256i, acc2_0);
        _mm256_storeu_si256(buf3.as_mut_ptr() as *mut __m256i, acc3_0);
        for r in 0..8 {
            out_0[r] = buf0[r] as i64;
            out_0[8 + r] = buf1[r] as i64;
            out_0[16 + r] = buf2[r] as i64;
            out_0[24 + r] = buf3[r] as i64;
        }

        _mm256_storeu_si256(buf0.as_mut_ptr() as *mut __m256i, acc0_1);
        _mm256_storeu_si256(buf1.as_mut_ptr() as *mut __m256i, acc1_1);
        _mm256_storeu_si256(buf2.as_mut_ptr() as *mut __m256i, acc2_1);
        _mm256_storeu_si256(buf3.as_mut_ptr() as *mut __m256i, acc3_1);
        for r in 0..8 {
            out_1[r] = buf0[r] as i64;
            out_1[8 + r] = buf1[r] as i64;
            out_1[16 + r] = buf2[r] as i64;
            out_1[24 + r] = buf3[r] as i64;
        }

        if remainder_cols > 0 {
            let rem_c = full_bytes;
            for r in 0..32 {
                let bw0 = *block_ptr_0.add(rem_c * 32 + r);
                let bw1 = *block_ptr_1.add(rem_c * 32 + r);
                for k in 0..remainder_cols {
                    let cw0 = (bw0 >> (k * 2)) & 0x03;
                    let weight0 = match cw0 {
                        0 => -1i64,
                        2 => 1i64,
                        _ => 0i64,
                    };
                    let cw1 = (bw1 >> (k * 2)) & 0x03;
                    let weight1 = match cw1 {
                        0 => -1i64,
                        2 => 1i64,
                        _ => 0i64,
                    };
                    let pair_idx = full_bytes * 2 + k / 2;
                    let pair_sub = k % 2;
                    let single_val = if pair_sub == 0 {
                        *qlut.add(pair_idx * 16 + 6) as i64
                    } else {
                        *qlut.add(pair_idx * 16 + 9) as i64
                    };
                    out_0[r] += weight0 * single_val;
                    out_1[r] += weight1 * single_val;
                }
            }
        }
    }
}

/// Helper kernel evaluating a single 32-row block (BM = 32).
#[target_feature(enable = "avx2")]
unsafe fn gemv_lut_block32_avx2(
    block_ptr: *const u8,
    qlut: *const i8,
    full_bytes: usize,
    remainder_cols: usize,
    mask_0f: __m256i,
    out: &mut [i64],
) {
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        let mut c = 0;
        while c < full_bytes {
            let chunk_len = (full_bytes - c).min(64);
            let mut sum16_lo = _mm256_setzero_si256();
            let mut sum16_hi = _mm256_setzero_si256();

            for ci in 0..chunk_len {
                let cur_c = c + ci;
                let vec_a = _mm256_loadu_si256(block_ptr.add(cur_c * 32) as *const __m256i);
                let vec_bot = _mm256_and_si256(vec_a, mask_0f);
                let vec_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), mask_0f);

                let lut_p0 = _mm_loadu_si128(qlut.add((2 * cur_c) * 16) as *const __m128i);
                let lut_p1 = _mm_loadu_si128(qlut.add((2 * cur_c + 1) * 16) as *const __m128i);

                let tbl0 = _mm256_set_m128i(lut_p0, lut_p0);
                let tbl1 = _mm256_set_m128i(lut_p1, lut_p1);

                let p0 = _mm256_shuffle_epi8(tbl0, vec_bot);
                let p1 = _mm256_shuffle_epi8(tbl1, vec_top);

                let p0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p0));
                let p1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p1));
                sum16_lo = _mm256_add_epi16(sum16_lo, _mm256_add_epi16(p0_lo, p1_lo));

                let p0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p0, 1));
                let p1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p1, 1));
                sum16_hi = _mm256_add_epi16(sum16_hi, _mm256_add_epi16(p0_hi, p1_hi));
            }

            acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_lo)));
            acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_lo, 1)));
            acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16_hi)));
            acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16_hi, 1)));

            c += chunk_len;
        }

        let mut buf0 = [0i32; 8];
        let mut buf1 = [0i32; 8];
        let mut buf2 = [0i32; 8];
        let mut buf3 = [0i32; 8];
        _mm256_storeu_si256(buf0.as_mut_ptr() as *mut __m256i, acc0);
        _mm256_storeu_si256(buf1.as_mut_ptr() as *mut __m256i, acc1);
        _mm256_storeu_si256(buf2.as_mut_ptr() as *mut __m256i, acc2);
        _mm256_storeu_si256(buf3.as_mut_ptr() as *mut __m256i, acc3);

        for r in 0..8 {
            out[r] = buf0[r] as i64;
            out[8 + r] = buf1[r] as i64;
            out[16 + r] = buf2[r] as i64;
            out[24 + r] = buf3[r] as i64;
        }

        if remainder_cols > 0 {
            let rem_c = full_bytes;
            for r in 0..32 {
                let bw = *block_ptr.add(rem_c * 32 + r);
                for k in 0..remainder_cols {
                    let cw = (bw >> (k * 2)) & 0x03;
                    let weight = match cw {
                        0 => -1i64,
                        2 => 1i64,
                        _ => 0i64,
                    };
                    let pair_idx = full_bytes * 2 + k / 2;
                    let pair_sub = k % 2;
                    let single_val = if pair_sub == 0 {
                        *qlut.add(pair_idx * 16 + 6) as i64
                    } else {
                        *qlut.add(pair_idx * 16 + 9) as i64
                    };
                    out[r] += weight * single_val;
                }
            }
        }
    }
}

/// Computes General Matrix-Vector Multiplication (GEMV: Y = W * X) on 32-row tiled weights
/// using AVX2 LUT kernels directly from an int8 activation vector.
#[target_feature(enable = "avx2")]
pub unsafe fn gemv_ternary_int8_tiled32_avx2(
    tiled_weights: &[u8],
    x: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    let qlut = crate::scalar::build_activation_lut_scalar(&x[..cols]);
    unsafe {
        gemv_lut_tiled32_avx2(tiled_weights, &qlut, rows, cols, out);
    }
}

/// Computes multi-threaded General Matrix-Vector Multiplication using AVX2 on 32-row tiled weights.
pub fn gemv_lut_tiled32_parallel(
    tiled_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
    num_threads: usize,
) {
    let num_blocks = rows / 32;
    if num_threads <= 1 || num_blocks < num_threads {
        unsafe {
            gemv_lut_tiled32_avx2(tiled_weights, qlut, rows, cols, out);
        }
        return;
    }

    let bytes_per_row = cols.div_ceil(4);
    let blocks_per_thread = (((num_blocks + num_threads - 1) / num_threads + 1) / 2) * 2;
    let rows_per_thread = blocks_per_thread * 32;
    let weights_per_thread = rows_per_thread * bytes_per_row;
    let main_rows = num_blocks * 32;

    use rayon::prelude::*;
    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..main_rows]
            .par_chunks_mut(rows_per_thread)
            .zip(tiled_weights[..main_rows * bytes_per_row].par_chunks(weights_per_thread))
            .for_each(|(out_chunk, weights_chunk)| unsafe {
                let chunk_rows = out_chunk.len();
                gemv_lut_tiled32_avx2(weights_chunk, qlut, chunk_rows, cols, out_chunk);
            });
    });

    let processed_rows = num_blocks * 32;
    if processed_rows < rows {
        let remainder_weights = &tiled_weights[num_blocks * (32 * bytes_per_row)..];
        let rem_rows = rows - processed_rows;
        let rem_out = &mut out[processed_rows..rows];
        unsafe {
            gemv_lut_tiled32_avx2(remainder_weights, qlut, rem_rows, cols, rem_out);
        }
    }
}
