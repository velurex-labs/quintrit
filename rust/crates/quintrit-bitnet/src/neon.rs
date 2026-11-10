#![allow(unsafe_op_in_unsafe_fn)]

//! Optimized ARM NEON SIMD implementation for 4 trits per byte (128-bit vectors).
//!
//! Strict 1:1 Reference Port of Microsoft BitNet NEON kernels (`ggml-bitnet-lut.cpp` / T-MAC).
//!
//! Includes:
//! - High-performance NEON GEMV kernel leveraging `vqtbl1q_s8` against precomputed
//!   activation lookup tables (faithful 1:1 port of BitNet / T-MAC).
//! - 32-row block interleaved (tiled) layout with Multi-Block Register Reuse (BM = 64).

use core::arch::aarch64::*;

/// Computes GEMV Matrix-Vector product using ARM NEON LUT kernels
/// on a 32-row block interleaved (tiled) weight matrix with Multi-Block Register Reuse (`BM = 64`).
///
/// Evaluates two 32-row blocks simultaneously (64 rows total).
/// Tables `tbl0` and `tbl1` are loaded ONCE per column and reused across both 32-row blocks.
pub unsafe fn gemv_lut_tiled32_neon(
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

    let mask_0f = vdupq_n_u8(0x0f);
    let qlut_ptr = qlut.as_ptr();

    // 1. Process pairs of 32-row blocks (BM = 64) with shared table reuse
    for pair in 0..num_pairs {
        let b0 = pair * 2;
        let b1 = pair * 2 + 1;
        let r0 = b0 * 32;
        let ptr0 = tiled_weights.as_ptr().add(b0 * (32 * bytes_per_row));
        let ptr1 = tiled_weights.as_ptr().add(b1 * (32 * bytes_per_row));

        let (out_0, out_1) = out[r0..r0 + 64].split_at_mut(32);
        gemv_lut_block64_neon(
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

    // 2. Process single remainder 32-row block if num_blocks is odd
    if num_blocks % 2 != 0 {
        let b = num_pairs * 2;
        let r0 = b * 32;
        let ptr = tiled_weights.as_ptr().add(b * (32 * bytes_per_row));
        gemv_lut_block32_neon(
            ptr,
            qlut_ptr,
            full_bytes,
            remainder_cols,
            mask_0f,
            &mut out[r0..r0 + 32],
        );
    }

    // 3. Remainder rows (rows % 32 != 0)
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
                let p0 = *qlut.get_unchecked((2 * c) * 16 + idx0) as i64;
                let p1 = *qlut.get_unchecked((2 * c + 1) * 16 + idx1) as i64;
                sum += p0 + p1;
            }
            if remainder_cols > 0 {
                let bw = rem_weights[r * bytes_per_row + full_bytes];
                for k in 0..remainder_cols {
                    let cw = (bw >> (k * 2)) & 0x03;
                    sum += remainder_product(qlut.as_ptr(), full_bytes, k, cw);
                }
            }
            out[processed_rows + r] = sum;
        }
    }
}

/// Helper kernel evaluating a pair of 32-row blocks (BM = 64) with shared activation tables.
#[inline(always)]
unsafe fn gemv_lut_block64_neon(
    block_ptr_0: *const u8,
    block_ptr_1: *const u8,
    qlut: *const i8,
    full_bytes: usize,
    remainder_cols: usize,
    mask_0f: uint8x16_t,
    out_0: &mut [i64],
    out_1: &mut [i64],
) {
    let mut acc0_0 = vdupq_n_s32(0);
    let mut acc1_0 = vdupq_n_s32(0);
    let mut acc2_0 = vdupq_n_s32(0);
    let mut acc3_0 = vdupq_n_s32(0);
    let mut acc4_0 = vdupq_n_s32(0);
    let mut acc5_0 = vdupq_n_s32(0);
    let mut acc6_0 = vdupq_n_s32(0);
    let mut acc7_0 = vdupq_n_s32(0);

    let mut acc0_1 = vdupq_n_s32(0);
    let mut acc1_1 = vdupq_n_s32(0);
    let mut acc2_1 = vdupq_n_s32(0);
    let mut acc3_1 = vdupq_n_s32(0);
    let mut acc4_1 = vdupq_n_s32(0);
    let mut acc5_1 = vdupq_n_s32(0);
    let mut acc6_1 = vdupq_n_s32(0);
    let mut acc7_1 = vdupq_n_s32(0);

    let mut c = 0;
    while c < full_bytes {
        let chunk_len = (full_bytes - c).min(16);

        let mut sum16_0_0 = vdupq_n_s16(0);
        let mut sum16_1_0 = vdupq_n_s16(0);
        let mut sum16_2_0 = vdupq_n_s16(0);
        let mut sum16_3_0 = vdupq_n_s16(0);

        let mut sum16_0_1 = vdupq_n_s16(0);
        let mut sum16_1_1 = vdupq_n_s16(0);
        let mut sum16_2_1 = vdupq_n_s16(0);
        let mut sum16_3_1 = vdupq_n_s16(0);

        for ci in 0..chunk_len {
            let cur_c = c + ci;
            let tbl0 = vld1q_s8(qlut.add((2 * cur_c) * 16));
            let tbl1 = vld1q_s8(qlut.add((2 * cur_c + 1) * 16));

            // Block 0 (Rows 0..31)
            let w0_lo = vld1q_u8(block_ptr_0.add(cur_c * 32));
            let w0_hi = vld1q_u8(block_ptr_0.add(cur_c * 32 + 16));

            let bot0_lo = vandq_u8(w0_lo, mask_0f);
            let top0_lo = vandq_u8(vshrq_n_u8(w0_lo, 4), mask_0f);
            let bot0_hi = vandq_u8(w0_hi, mask_0f);
            let top0_hi = vandq_u8(vshrq_n_u8(w0_hi, 4), mask_0f);

            let p0_0_lo = vqtbl1q_s8(tbl0, bot0_lo);
            let p1_0_lo = vqtbl1q_s8(tbl1, top0_lo);
            let p0_0_hi = vqtbl1q_s8(tbl0, bot0_hi);
            let p1_0_hi = vqtbl1q_s8(tbl1, top0_hi);

            sum16_0_0 = vaddq_s16(sum16_0_0, vaddl_s8(vget_low_s8(p0_0_lo), vget_low_s8(p1_0_lo)));
            sum16_1_0 = vaddq_s16(sum16_1_0, vaddl_s8(vget_high_s8(p0_0_lo), vget_high_s8(p1_0_lo)));
            sum16_2_0 = vaddq_s16(sum16_2_0, vaddl_s8(vget_low_s8(p0_0_hi), vget_low_s8(p1_0_hi)));
            sum16_3_0 = vaddq_s16(sum16_3_0, vaddl_s8(vget_high_s8(p0_0_hi), vget_high_s8(p1_0_hi)));

            // Block 1 (Rows 32..63)
            let w1_lo = vld1q_u8(block_ptr_1.add(cur_c * 32));
            let w1_hi = vld1q_u8(block_ptr_1.add(cur_c * 32 + 16));

            let bot1_lo = vandq_u8(w1_lo, mask_0f);
            let top1_lo = vandq_u8(vshrq_n_u8(w1_lo, 4), mask_0f);
            let bot1_hi = vandq_u8(w1_hi, mask_0f);
            let top1_hi = vandq_u8(vshrq_n_u8(w1_hi, 4), mask_0f);

            let p0_1_lo = vqtbl1q_s8(tbl0, bot1_lo);
            let p1_1_lo = vqtbl1q_s8(tbl1, top1_lo);
            let p0_1_hi = vqtbl1q_s8(tbl0, bot1_hi);
            let p1_1_hi = vqtbl1q_s8(tbl1, top1_hi);
            sum16_0_1 = vaddq_s16(sum16_0_1, vaddl_s8(vget_low_s8(p0_1_lo), vget_low_s8(p1_1_lo)));
            sum16_1_1 = vaddq_s16(sum16_1_1, vaddl_s8(vget_high_s8(p0_1_lo), vget_high_s8(p1_1_lo)));
            sum16_2_1 = vaddq_s16(sum16_2_1, vaddl_s8(vget_low_s8(p0_1_hi), vget_low_s8(p1_1_hi)));
            sum16_3_1 = vaddq_s16(sum16_3_1, vaddl_s8(vget_high_s8(p0_1_hi), vget_high_s8(p1_1_hi)));
        }

        // Accumulate into 32-bit registers (Block 0)
        acc0_0 = vaddw_s16(acc0_0, vget_low_s16(sum16_0_0));
        acc1_0 = vaddw_s16(acc1_0, vget_high_s16(sum16_0_0));
        acc2_0 = vaddw_s16(acc2_0, vget_low_s16(sum16_1_0));
        acc3_0 = vaddw_s16(acc3_0, vget_high_s16(sum16_1_0));
        acc4_0 = vaddw_s16(acc4_0, vget_low_s16(sum16_2_0));
        acc5_0 = vaddw_s16(acc5_0, vget_high_s16(sum16_2_0));
        acc6_0 = vaddw_s16(acc6_0, vget_low_s16(sum16_3_0));
        acc7_0 = vaddw_s16(acc7_0, vget_high_s16(sum16_3_0));

        // Accumulate into 32-bit registers (Block 1)
        acc0_1 = vaddw_s16(acc0_1, vget_low_s16(sum16_0_1));
        acc1_1 = vaddw_s16(acc1_1, vget_high_s16(sum16_0_1));
        acc2_1 = vaddw_s16(acc2_1, vget_low_s16(sum16_1_1));
        acc3_1 = vaddw_s16(acc3_1, vget_high_s16(sum16_1_1));
        acc4_1 = vaddw_s16(acc4_1, vget_low_s16(sum16_2_1));
        acc5_1 = vaddw_s16(acc5_1, vget_high_s16(sum16_2_1));
        acc6_1 = vaddw_s16(acc6_1, vget_low_s16(sum16_3_1));
        acc7_1 = vaddw_s16(acc7_1, vget_high_s16(sum16_3_1));

        c += chunk_len;
    }

    // Write back Block 0 (rows 0..31)
    let mut buf = [0i32; 4];
    vst1q_s32(buf.as_mut_ptr(), acc0_0); for r in 0..4 { out_0[r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc1_0); for r in 0..4 { out_0[4 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc2_0); for r in 0..4 { out_0[8 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc3_0); for r in 0..4 { out_0[12 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc4_0); for r in 0..4 { out_0[16 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc5_0); for r in 0..4 { out_0[20 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc6_0); for r in 0..4 { out_0[24 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc7_0); for r in 0..4 { out_0[28 + r] = buf[r] as i64; }

    // Write back Block 1 (rows 32..63)
    vst1q_s32(buf.as_mut_ptr(), acc0_1); for r in 0..4 { out_1[r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc1_1); for r in 0..4 { out_1[4 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc2_1); for r in 0..4 { out_1[8 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc3_1); for r in 0..4 { out_1[12 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc4_1); for r in 0..4 { out_1[16 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc5_1); for r in 0..4 { out_1[20 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc6_1); for r in 0..4 { out_1[24 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc7_1); for r in 0..4 { out_1[28 + r] = buf[r] as i64; }

    if remainder_cols > 0 {
        let base_c = full_bytes;
        for r in 0..32 {
            let bw0 = *block_ptr_0.add(base_c * 32 + r);
            let bw1 = *block_ptr_1.add(base_c * 32 + r);
            for k in 0..remainder_cols {
                out_0[r] += remainder_product(qlut, base_c, k, (bw0 >> (k * 2)) & 0x03);
                out_1[r] += remainder_product(qlut, base_c, k, (bw1 >> (k * 2)) & 0x03);
            }
        }
    }
}

/// Product `weight * activation` of one trit in the trailing partial byte (`cols % 4 != 0`).
///
/// The pair table for activations `(x_a, x_b)` stores `w_a*x_a + w_b*x_b` at index `code_a | code_b << 2`.
/// Pairing the trit with a neutral partner (code 1, weight 0) therefore reads `w * x` directly.
#[inline(always)]
unsafe fn remainder_product(qlut: *const i8, full_bytes: usize, k: usize, code: u8) -> i64 {
    let pair = 2 * full_bytes + k / 2;
    let idx = if k % 2 == 0 { (code as usize) | (1 << 2) } else { 1 | ((code as usize) << 2) };
    *qlut.add(pair * 16 + idx) as i64
}

/// Helper kernel evaluating a single 32-row block on ARM NEON.
#[inline(always)]
unsafe fn gemv_lut_block32_neon(
    block_ptr: *const u8,
    qlut: *const i8,
    full_bytes: usize,
    remainder_cols: usize,
    mask_0f: uint8x16_t,
    out: &mut [i64],
) {
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    let mut acc4 = vdupq_n_s32(0);
    let mut acc5 = vdupq_n_s32(0);
    let mut acc6 = vdupq_n_s32(0);
    let mut acc7 = vdupq_n_s32(0);

    let mut c = 0;
    while c < full_bytes {
        let chunk_len = (full_bytes - c).min(16);

        let mut sum16_0 = vdupq_n_s16(0);
        let mut sum16_1 = vdupq_n_s16(0);
        let mut sum16_2 = vdupq_n_s16(0);
        let mut sum16_3 = vdupq_n_s16(0);

        for ci in 0..chunk_len {
            let cur_c = c + ci;
            let w_ptr = block_ptr.add(cur_c * 32);
            let w_lo = vld1q_u8(w_ptr);
            let w_hi = vld1q_u8(w_ptr.add(16));

            let bot_lo = vandq_u8(w_lo, mask_0f);
            let top_lo = vandq_u8(vshrq_n_u8(w_lo, 4), mask_0f);
            let bot_hi = vandq_u8(w_hi, mask_0f);
            let top_hi = vandq_u8(vshrq_n_u8(w_hi, 4), mask_0f);

            let tbl0 = vld1q_s8(qlut.add((2 * cur_c) * 16));
            let tbl1 = vld1q_s8(qlut.add((2 * cur_c + 1) * 16));

            let p0_lo = vqtbl1q_s8(tbl0, bot_lo);
            let p1_lo = vqtbl1q_s8(tbl1, top_lo);
            let p0_hi = vqtbl1q_s8(tbl0, bot_hi);
            let p1_hi = vqtbl1q_s8(tbl1, top_hi);

            sum16_0 = vaddq_s16(sum16_0, vaddl_s8(vget_low_s8(p0_lo), vget_low_s8(p1_lo)));
            sum16_1 = vaddq_s16(sum16_1, vaddl_s8(vget_high_s8(p0_lo), vget_high_s8(p1_lo)));
            sum16_2 = vaddq_s16(sum16_2, vaddl_s8(vget_low_s8(p0_hi), vget_low_s8(p1_hi)));
            sum16_3 = vaddq_s16(sum16_3, vaddl_s8(vget_high_s8(p0_hi), vget_high_s8(p1_hi)));
        }

        acc0 = vaddw_s16(acc0, vget_low_s16(sum16_0));
        acc1 = vaddw_s16(acc1, vget_high_s16(sum16_0));
        acc2 = vaddw_s16(acc2, vget_low_s16(sum16_1));
        acc3 = vaddw_s16(acc3, vget_high_s16(sum16_1));
        acc4 = vaddw_s16(acc4, vget_low_s16(sum16_2));
        acc5 = vaddw_s16(acc5, vget_high_s16(sum16_2));
        acc6 = vaddw_s16(acc6, vget_low_s16(sum16_3));
        acc7 = vaddw_s16(acc7, vget_high_s16(sum16_3));

        c += chunk_len;
    }

    let mut buf = [0i32; 4];
    vst1q_s32(buf.as_mut_ptr(), acc0); for r in 0..4 { out[r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc1); for r in 0..4 { out[4 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc2); for r in 0..4 { out[8 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc3); for r in 0..4 { out[12 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc4); for r in 0..4 { out[16 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc5); for r in 0..4 { out[20 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc6); for r in 0..4 { out[24 + r] = buf[r] as i64; }
    vst1q_s32(buf.as_mut_ptr(), acc7); for r in 0..4 { out[28 + r] = buf[r] as i64; }

    if remainder_cols > 0 {
        let base_c = full_bytes;
        for r in 0..32 {
            let bw = *block_ptr.add(base_c * 32 + r);
            for k in 0..remainder_cols {
                out[r] += remainder_product(qlut, base_c, k, (bw >> (k * 2)) & 0x03);
            }
        }
    }
}

/// Multi-threaded GEMV matrix-vector multiplication using 32-row tiled LUT kernel on ARM NEON.
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
            gemv_lut_tiled32_neon(tiled_weights, qlut, rows, cols, out);
        }
        return;
    }

    let bytes_per_row = cols.div_ceil(4);
    let blocks_per_thread = (num_blocks + num_threads - 1) / num_threads;
    let rows_per_thread = blocks_per_thread * 32;

    let weights_per_thread = rows_per_thread * bytes_per_row;
    let main_rows = num_blocks * 32;

    use rayon::prelude::*;
    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..main_rows]
            .par_chunks_mut(rows_per_thread)
            .zip(tiled_weights[..main_rows * bytes_per_row].par_chunks(weights_per_thread))
            .for_each(|(out_chunk, weights_chunk)| {
                let chunk_rows = out_chunk.len();
                unsafe {
                    gemv_lut_tiled32_neon(weights_chunk, qlut, chunk_rows, cols, out_chunk);
                }
            });
    });

    let processed_rows = num_blocks * 32;
    if processed_rows < rows {
        crate::scalar::gemv_lut_tiled32_scalar(
            &tiled_weights[num_blocks * (32 * bytes_per_row)..],
            qlut,
            rows - processed_rows,
            cols,
            &mut out[processed_rows..],
        );
    }
}
