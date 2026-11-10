#![allow(unsafe_op_in_unsafe_fn)]

//! # 32-Row & 64-Row Tiled 2-Table Activation-LUT GEMV NEON Kernels
//!
//! Evaluates 32-row and 64-row blocks of weights against precomputed QLUT
//! on AArch64 architectures (ARM Cortex-A76 / Raspberry Pi 5) using ARM NEON 128-bit SIMD intrinsics.
//!
//! Features the 2-Table `vtbl2` Architecture (Phase 2 Evolution):
//! - Upper trits (t3, t4) resolved via single `vqtbl1q_s8(tbl_34, h34)`
//! - Lower trio (t0, t1, t2) resolved in a single 32-byte table lookup via `vqtbl2q_s8(tbl_012, r3)`
//!   where remainder `r3 = w - 28 * q` is computed in 1 cycle via `vmlsq_u8(w, h34, c_28)`
//! - Completely eliminates table 2 and intermediate middle-trit arithmetic
//! - Multi-Block Register Reuse (BM = 64) where activation tables (`tbl_34`, `tbl_012`)
//!   are loaded once per column and shared across two 32-row blocks (64 rows total).

use core::arch::aarch64::*;

#[inline(always)]
unsafe fn prefetch_l1(ptr: *const u8) {
    core::arch::asm!(
        "prfm pldl1keep, [{ptr}]",
        ptr = in(reg) ptr,
        options(nostack, readonly, preserves_flags)
    );
}

/// ARMv8.6-A Matrix Multiply Int8 (I8MM) instruction primitive.
///
/// Computes 2x2 matrix multiply of 8-bit integers accumulating into 32-bit integers:
/// `acc += a * b` ($2 \times 8 \times 2 = 32$ MAC operations in a single cycle).
#[inline(always)]
pub unsafe fn smmla_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    acc
}


#[inline(always)]
unsafe fn eval_column_terms_32rows(
    w_lo: uint8x16_t,
    w_hi: uint8x16_t,
    tbl_34: int8x16_t,
    tbl_012: int8x16x2_t,
    c_28: uint8x16_t,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    let v11_lo = vshrq_n_u8(w_lo, 2);
    let v11_hi = vshrq_n_u8(w_hi, 2);

    let v15_lo = vshrq_n_u8(w_lo, 5);
    let v15_hi = vshrq_n_u8(w_hi, 5);

    let v17_lo = vrhaddq_u8(v11_lo, v15_lo);
    let v17_hi = vrhaddq_u8(v11_hi, v15_hi);

    let h34_lo = vshrq_n_u8(v17_lo, 2);
    let h34_hi = vshrq_n_u8(v17_hi, 2);

    let p34_lo = vqtbl1q_s8(tbl_34, h34_lo);
    let p34_hi = vqtbl1q_s8(tbl_34, h34_hi);

    let r3_lo = vmlsq_u8(w_lo, h34_lo, c_28);
    let r3_hi = vmlsq_u8(w_hi, h34_hi, c_28);

    let p012_lo = vqtbl2q_s8(tbl_012, r3_lo);
    let p012_hi = vqtbl2q_s8(tbl_012, r3_hi);

    (p34_lo, p012_lo, p34_hi, p012_hi)
}

#[inline(always)]
unsafe fn eval_column_32rows(
    w_lo: uint8x16_t,
    w_hi: uint8x16_t,
    tbl_34: int8x16_t,
    tbl_012: int8x16x2_t,
    c_28: uint8x16_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let (p34_lo, p012_lo, p34_hi, p012_hi) =
        eval_column_terms_32rows(w_lo, w_hi, tbl_34, tbl_012, c_28);

    let sum_0_7 = vaddl_s8(vget_low_s8(p34_lo), vget_low_s8(p012_lo));
    let sum_8_15 = vaddl_high_s8(p34_lo, p012_lo);
    let sum_16_23 = vaddl_s8(vget_low_s8(p34_hi), vget_low_s8(p012_hi));
    let sum_24_31 = vaddl_high_s8(p34_hi, p012_hi);

    (sum_0_7, sum_8_15, sum_16_23, sum_24_31)
}

#[inline(always)]
unsafe fn eval_column_terms_64rows_interleaved(
    w0_lo: uint8x16_t,
    w0_hi: uint8x16_t,
    w1_lo: uint8x16_t,
    w1_hi: uint8x16_t,
    tbl_34: int8x16_t,
    tbl_012: int8x16x2_t,
    c_28: uint8x16_t,
) -> (
    (int8x16_t, int8x16_t, int8x16_t, int8x16_t),
    (int8x16_t, int8x16_t, int8x16_t, int8x16_t),
) {
    let v11_0_lo = vshrq_n_u8(w0_lo, 2);
    let v11_1_lo = vshrq_n_u8(w1_lo, 2);
    let v11_0_hi = vshrq_n_u8(w0_hi, 2);
    let v11_1_hi = vshrq_n_u8(w1_hi, 2);

    let v15_0_lo = vshrq_n_u8(w0_lo, 5);
    let v15_1_lo = vshrq_n_u8(w1_lo, 5);
    let v15_0_hi = vshrq_n_u8(w0_hi, 5);
    let v15_1_hi = vshrq_n_u8(w1_hi, 5);

    let v17_0_lo = vrhaddq_u8(v11_0_lo, v15_0_lo);
    let v17_1_lo = vrhaddq_u8(v11_1_lo, v15_1_lo);
    let v17_0_hi = vrhaddq_u8(v11_0_hi, v15_0_hi);
    let v17_1_hi = vrhaddq_u8(v11_1_hi, v15_1_hi);

    let h34_0_lo = vshrq_n_u8(v17_0_lo, 2);
    let h34_1_lo = vshrq_n_u8(v17_1_lo, 2);
    let h34_0_hi = vshrq_n_u8(v17_0_hi, 2);
    let h34_1_hi = vshrq_n_u8(v17_1_hi, 2);

    let p34_0_lo = vqtbl1q_s8(tbl_34, h34_0_lo);
    let r3_0_lo = vmlsq_u8(w0_lo, h34_0_lo, c_28);

    let p34_1_lo = vqtbl1q_s8(tbl_34, h34_1_lo);
    let r3_1_lo = vmlsq_u8(w1_lo, h34_1_lo, c_28);

    let p34_0_hi = vqtbl1q_s8(tbl_34, h34_0_hi);
    let r3_0_hi = vmlsq_u8(w0_hi, h34_0_hi, c_28);

    let p34_1_hi = vqtbl1q_s8(tbl_34, h34_1_hi);
    let r3_1_hi = vmlsq_u8(w1_hi, h34_1_hi, c_28);

    let p012_0_lo = vqtbl2q_s8(tbl_012, r3_0_lo);
    let p012_1_lo = vqtbl2q_s8(tbl_012, r3_1_lo);
    let p012_0_hi = vqtbl2q_s8(tbl_012, r3_0_hi);
    let p012_1_hi = vqtbl2q_s8(tbl_012, r3_1_hi);

    (
        (p34_0_lo, p012_0_lo, p34_0_hi, p012_0_hi),
        (p34_1_lo, p012_1_lo, p34_1_hi, p012_1_hi),
    )
}

#[inline(always)]
unsafe fn eval_column_64rows_interleaved(
    w0_lo: uint8x16_t,
    w0_hi: uint8x16_t,
    w1_lo: uint8x16_t,
    w1_hi: uint8x16_t,
    tbl_34: int8x16_t,
    tbl_012: int8x16x2_t,
    c_28: uint8x16_t,
    mut s0_0: int16x8_t,
    mut s0_1: int16x8_t,
    mut s0_2: int16x8_t,
    mut s0_3: int16x8_t,
    mut s1_0: int16x8_t,
    mut s1_1: int16x8_t,
    mut s1_2: int16x8_t,
    mut s1_3: int16x8_t,
) -> (
    int16x8_t, int16x8_t, int16x8_t, int16x8_t,
    int16x8_t, int16x8_t, int16x8_t, int16x8_t,
) {
    let (
        (p34_0_lo, p012_0_lo, p34_0_hi, p012_0_hi),
        (p34_1_lo, p012_1_lo, p34_1_hi, p012_1_hi),
    ) = eval_column_terms_64rows_interleaved(
        w0_lo, w0_hi, w1_lo, w1_hi, tbl_34, tbl_012, c_28,
    );

    // Sum p34 and p012 terms with 1-cycle latency 16-bit widening add (zero RAW dependencies)
    let sum0_0 = vaddl_s8(vget_low_s8(p34_0_lo), vget_low_s8(p012_0_lo));
    let sum0_1 = vaddl_high_s8(p34_0_lo, p012_0_lo);
    let sum0_2 = vaddl_s8(vget_low_s8(p34_0_hi), vget_low_s8(p012_0_hi));
    let sum0_3 = vaddl_high_s8(p34_0_hi, p012_0_hi);

    let sum1_0 = vaddl_s8(vget_low_s8(p34_1_lo), vget_low_s8(p012_1_lo));
    let sum1_1 = vaddl_high_s8(p34_1_lo, p012_1_lo);
    let sum1_2 = vaddl_s8(vget_low_s8(p34_1_hi), vget_low_s8(p012_1_hi));
    let sum1_3 = vaddl_high_s8(p34_1_hi, p012_1_hi);

    // Accumulate across columns via 1-cycle dual-issue vaddq_s16
    s0_0 = vaddq_s16(s0_0, sum0_0);
    s0_1 = vaddq_s16(s0_1, sum0_1);
    s0_2 = vaddq_s16(s0_2, sum0_2);
    s0_3 = vaddq_s16(s0_3, sum0_3);

    s1_0 = vaddq_s16(s1_0, sum1_0);
    s1_1 = vaddq_s16(s1_1, sum1_1);
    s1_2 = vaddq_s16(s1_2, sum1_2);
    s1_3 = vaddq_s16(s1_3, sum1_3);

    (s0_0, s0_1, s0_2, s0_3, s1_0, s1_1, s1_2, s1_3)
}

/// Evaluates a 32-row block of weights against the precomputed QLUT
/// using ARM NEON 128-bit SIMD intrinsics with 2-table lookup.
pub unsafe fn gemv_lut_block32_neon(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);

    // 32-bit accumulators for all 32 rows
    let mut acc0 = vdupq_n_s32(0); // Rows 0..3
    let mut acc1 = vdupq_n_s32(0); // Rows 4..7
    let mut acc2 = vdupq_n_s32(0); // Rows 8..11
    let mut acc3 = vdupq_n_s32(0); // Rows 12..15
    let mut acc4 = vdupq_n_s32(0); // Rows 16..19
    let mut acc5 = vdupq_n_s32(0); // Rows 20..23
    let mut acc6 = vdupq_n_s32(0); // Rows 24..27
    let mut acc7 = vdupq_n_s32(0); // Rows 28..31

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(32);

        let mut sum_r0_r7 = vdupq_n_s16(0);
        let mut sum_r8_r15 = vdupq_n_s16(0);
        let mut sum_r16_r23 = vdupq_n_s16(0);
        let mut sum_r24_r31 = vdupq_n_s16(0);

        let mut qlut_ptr = qlut.add(c * 64);
        let mut w_ptr = block_weights.add(c * 32);

        for _ in 0..chunk_len {
            prefetch_l1(w_ptr.add(128));

            let tbl_34 = vld1q_s8(qlut_ptr);
            let tbl_012 = vld1q_s8_x2(qlut_ptr.add(16));
            qlut_ptr = qlut_ptr.add(64);

            let w_lo = vld1q_u8(w_ptr);
            let w_hi = vld1q_u8(w_ptr.add(16));
            w_ptr = w_ptr.add(32);

            let (r0, r1, r2, r3) = eval_column_32rows(
                w_lo, w_hi, tbl_34, tbl_012, c_28,
            );

            sum_r0_r7 = vaddq_s16(sum_r0_r7, r0);
            sum_r8_r15 = vaddq_s16(sum_r8_r15, r1);
            sum_r16_r23 = vaddq_s16(sum_r16_r23, r2);
            sum_r24_r31 = vaddq_s16(sum_r24_r31, r3);
        }

        acc0 = vaddw_s16(acc0, vget_low_s16(sum_r0_r7));
        acc1 = vaddw_s16(acc1, vget_high_s16(sum_r0_r7));
        acc2 = vaddw_s16(acc2, vget_low_s16(sum_r8_r15));
        acc3 = vaddw_s16(acc3, vget_high_s16(sum_r8_r15));

        acc4 = vaddw_s16(acc4, vget_low_s16(sum_r16_r23));
        acc5 = vaddw_s16(acc5, vget_high_s16(sum_r16_r23));
        acc6 = vaddw_s16(acc6, vget_low_s16(sum_r24_r31));
        acc7 = vaddw_s16(acc7, vget_high_s16(sum_r24_r31));

        c += chunk_len;
    }

    vst1q_f32(out_32, vcvtq_f32_s32(acc0));
    vst1q_f32(out_32.add(4), vcvtq_f32_s32(acc1));
    vst1q_f32(out_32.add(8), vcvtq_f32_s32(acc2));
    vst1q_f32(out_32.add(12), vcvtq_f32_s32(acc3));
    vst1q_f32(out_32.add(16), vcvtq_f32_s32(acc4));
    vst1q_f32(out_32.add(20), vcvtq_f32_s32(acc5));
    vst1q_f32(out_32.add(24), vcvtq_f32_s32(acc6));
    vst1q_f32(out_32.add(28), vcvtq_f32_s32(acc7));
}

/// Evaluates a pair of 32-row blocks (BM = 64, rows 0..63) with shared activation table reuse.
pub unsafe fn gemv_lut_block64_neon(
    block_weights_0: *const u8,
    block_weights_1: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_0: *mut f32,
    out_1: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);

    // 32-bit accumulators for Block 0 (Rows 0..31)
    let mut acc0_0 = vdupq_n_s32(0);
    let mut acc1_0 = vdupq_n_s32(0);
    let mut acc2_0 = vdupq_n_s32(0);
    let mut acc3_0 = vdupq_n_s32(0);
    let mut acc4_0 = vdupq_n_s32(0);
    let mut acc5_0 = vdupq_n_s32(0);
    let mut acc6_0 = vdupq_n_s32(0);
    let mut acc7_0 = vdupq_n_s32(0);

    // 32-bit accumulators for Block 1 (Rows 32..63)
    let mut acc0_1 = vdupq_n_s32(0);
    let mut acc1_1 = vdupq_n_s32(0);
    let mut acc2_1 = vdupq_n_s32(0);
    let mut acc3_1 = vdupq_n_s32(0);
    let mut acc4_1 = vdupq_n_s32(0);
    let mut acc5_1 = vdupq_n_s32(0);
    let mut acc6_1 = vdupq_n_s32(0);
    let mut acc7_1 = vdupq_n_s32(0);

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(32);

        let mut sum_r0_r7_0 = vdupq_n_s16(0);
        let mut sum_r8_r15_0 = vdupq_n_s16(0);
        let mut sum_r16_r23_0 = vdupq_n_s16(0);
        let mut sum_r24_r31_0 = vdupq_n_s16(0);

        let mut sum_r0_r7_1 = vdupq_n_s16(0);
        let mut sum_r8_r15_1 = vdupq_n_s16(0);
        let mut sum_r16_r23_1 = vdupq_n_s16(0);
        let mut sum_r24_r31_1 = vdupq_n_s16(0);

        let mut qlut_ptr = qlut.add(c * 64);
        let mut w_ptr_0 = block_weights_0.add(c * 32);
        let mut w_ptr_1 = block_weights_1.add(c * 32);

        let mut k = 0;
        while k + 1 < chunk_len {
            prefetch_l1(w_ptr_0.add(128));
            prefetch_l1(w_ptr_1.add(128));

            // Column 0
            let tbl_34_0 = vld1q_s8(qlut_ptr);
            let tbl_012_0 = vld1q_s8_x2(qlut_ptr.add(16));
            let w0_lo_0 = vld1q_u8(w_ptr_0);
            let w0_hi_0 = vld1q_u8(w_ptr_0.add(16));
            let w1_lo_0 = vld1q_u8(w_ptr_1);
            let w1_hi_0 = vld1q_u8(w_ptr_1.add(16));

            // Column 1
            let tbl_34_1 = vld1q_s8(qlut_ptr.add(64));
            let tbl_012_1 = vld1q_s8_x2(qlut_ptr.add(80));
            let w0_lo_1 = vld1q_u8(w_ptr_0.add(32));
            let w0_hi_1 = vld1q_u8(w_ptr_0.add(48));
            let w1_lo_1 = vld1q_u8(w_ptr_1.add(32));
            let w1_hi_1 = vld1q_u8(w_ptr_1.add(48));

            qlut_ptr = qlut_ptr.add(128);
            w_ptr_0 = w_ptr_0.add(64);
            w_ptr_1 = w_ptr_1.add(64);

            let (
                s0_0_a, s0_1_a, s0_2_a, s0_3_a,
                s1_0_a, s1_1_a, s1_2_a, s1_3_a,
            ) = eval_column_64rows_interleaved(
                w0_lo_0, w0_hi_0, w1_lo_0, w1_hi_0,
                tbl_34_0, tbl_012_0, c_28,
                sum_r0_r7_0, sum_r8_r15_0, sum_r16_r23_0, sum_r24_r31_0,
                sum_r0_r7_1, sum_r8_r15_1, sum_r16_r23_1, sum_r24_r31_1,
            );

            let (
                s0_0_b, s0_1_b, s0_2_b, s0_3_b,
                s1_0_b, s1_1_b, s1_2_b, s1_3_b,
            ) = eval_column_64rows_interleaved(
                w0_lo_1, w0_hi_1, w1_lo_1, w1_hi_1,
                tbl_34_1, tbl_012_1, c_28,
                s0_0_a, s0_1_a, s0_2_a, s0_3_a,
                s1_0_a, s1_1_a, s1_2_a, s1_3_a,
            );

            sum_r0_r7_0 = s0_0_b;
            sum_r8_r15_0 = s0_1_b;
            sum_r16_r23_0 = s0_2_b;
            sum_r24_r31_0 = s0_3_b;

            sum_r0_r7_1 = s1_0_b;
            sum_r8_r15_1 = s1_1_b;
            sum_r16_r23_1 = s1_2_b;
            sum_r24_r31_1 = s1_3_b;

            k += 2;
        }

        if k < chunk_len {
            prefetch_l1(w_ptr_0.add(128));
            prefetch_l1(w_ptr_1.add(128));

            let tbl_34 = vld1q_s8(qlut_ptr);
            let tbl_012 = vld1q_s8_x2(qlut_ptr.add(16));
            qlut_ptr = qlut_ptr.add(64);

            let w0_lo = vld1q_u8(w_ptr_0);
            let w0_hi = vld1q_u8(w_ptr_0.add(16));
            w_ptr_0 = w_ptr_0.add(32);

            let w1_lo = vld1q_u8(w_ptr_1);
            let w1_hi = vld1q_u8(w_ptr_1.add(16));
            w_ptr_1 = w_ptr_1.add(32);

            let (
                new_s0_0, new_s0_1, new_s0_2, new_s0_3,
                new_s1_0, new_s1_1, new_s1_2, new_s1_3,
            ) = eval_column_64rows_interleaved(
                w0_lo, w0_hi, w1_lo, w1_hi,
                tbl_34, tbl_012, c_28,
                sum_r0_r7_0, sum_r8_r15_0, sum_r16_r23_0, sum_r24_r31_0,
                sum_r0_r7_1, sum_r8_r15_1, sum_r16_r23_1, sum_r24_r31_1,
            );

            sum_r0_r7_0 = new_s0_0;
            sum_r8_r15_0 = new_s0_1;
            sum_r16_r23_0 = new_s0_2;
            sum_r24_r31_0 = new_s0_3;

            sum_r0_r7_1 = new_s1_0;
            sum_r8_r15_1 = new_s1_1;
            sum_r16_r23_1 = new_s1_2;
            sum_r24_r31_1 = new_s1_3;
        }


        // Chunk Boundary: Widen and accumulate into Block 0 32-bit accumulators
        acc0_0 = vaddw_s16(acc0_0, vget_low_s16(sum_r0_r7_0));
        acc1_0 = vaddw_s16(acc1_0, vget_high_s16(sum_r0_r7_0));
        acc2_0 = vaddw_s16(acc2_0, vget_low_s16(sum_r8_r15_0));
        acc3_0 = vaddw_s16(acc3_0, vget_high_s16(sum_r8_r15_0));

        acc4_0 = vaddw_s16(acc4_0, vget_low_s16(sum_r16_r23_0));
        acc5_0 = vaddw_s16(acc5_0, vget_high_s16(sum_r16_r23_0));
        acc6_0 = vaddw_s16(acc6_0, vget_low_s16(sum_r24_r31_0));
        acc7_0 = vaddw_s16(acc7_0, vget_high_s16(sum_r24_r31_0));

        // Chunk Boundary: Widen and accumulate into Block 1 32-bit accumulators
        acc0_1 = vaddw_s16(acc0_1, vget_low_s16(sum_r0_r7_1));
        acc1_1 = vaddw_s16(acc1_1, vget_high_s16(sum_r0_r7_1));
        acc2_1 = vaddw_s16(acc2_1, vget_low_s16(sum_r8_r15_1));
        acc3_1 = vaddw_s16(acc3_1, vget_high_s16(sum_r8_r15_1));

        acc4_1 = vaddw_s16(acc4_1, vget_low_s16(sum_r16_r23_1));
        acc5_1 = vaddw_s16(acc5_1, vget_high_s16(sum_r16_r23_1));
        acc6_1 = vaddw_s16(acc6_1, vget_low_s16(sum_r24_r31_1));
        acc7_1 = vaddw_s16(acc7_1, vget_high_s16(sum_r24_r31_1));

        c += chunk_len;
    }

    // Convert directly to f32 and store to output buffers
    vst1q_f32(out_0, vcvtq_f32_s32(acc0_0));
    vst1q_f32(out_0.add(4), vcvtq_f32_s32(acc1_0));
    vst1q_f32(out_0.add(8), vcvtq_f32_s32(acc2_0));
    vst1q_f32(out_0.add(12), vcvtq_f32_s32(acc3_0));
    vst1q_f32(out_0.add(16), vcvtq_f32_s32(acc4_0));
    vst1q_f32(out_0.add(20), vcvtq_f32_s32(acc5_0));
    vst1q_f32(out_0.add(24), vcvtq_f32_s32(acc6_0));
    vst1q_f32(out_0.add(28), vcvtq_f32_s32(acc7_0));

    vst1q_f32(out_1, vcvtq_f32_s32(acc0_1));
    vst1q_f32(out_1.add(4), vcvtq_f32_s32(acc1_1));
    vst1q_f32(out_1.add(8), vcvtq_f32_s32(acc2_1));
    vst1q_f32(out_1.add(12), vcvtq_f32_s32(acc3_1));
    vst1q_f32(out_1.add(16), vcvtq_f32_s32(acc4_1));
    vst1q_f32(out_1.add(20), vcvtq_f32_s32(acc5_1));
    vst1q_f32(out_1.add(24), vcvtq_f32_s32(acc6_1));
    vst1q_f32(out_1.add(28), vcvtq_f32_s32(acc7_1));
}

#[inline(always)]
unsafe fn accumulate_pair_dotprod_16rows(
    p34_c0: int8x16_t,
    p012_c0: int8x16_t,
    p34_c1: int8x16_t,
    p012_c1: int8x16_t,
    ones: int8x16_t,
    acc_lo0: int32x4_t,
    acc_lo1: int32x4_t,
    acc_hi0: int32x4_t,
    acc_hi1: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let p_lo = vzip1q_s8(p34_c0, p012_c0);
    let p_hi = vzip2q_s8(p34_c0, p012_c0);

    let q_lo = vzip1q_s8(p34_c1, p012_c1);
    let q_hi = vzip2q_s8(p34_c1, p012_c1);

    let quad0 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(p_lo), vreinterpretq_s16_s8(q_lo)));
    let quad1 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(p_lo), vreinterpretq_s16_s8(q_lo)));
    let quad2 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(p_hi), vreinterpretq_s16_s8(q_hi)));
    let quad3 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(p_hi), vreinterpretq_s16_s8(q_hi)));

    (
        vdotq_s32(acc_lo0, quad0, ones),
        vdotq_s32(acc_lo1, quad1, ones),
        vdotq_s32(acc_hi0, quad2, ones),
        vdotq_s32(acc_hi1, quad3, ones),
    )
}

#[inline(always)]
unsafe fn accumulate_single_dotprod_16rows(
    p34: int8x16_t,
    p012: int8x16_t,
    ones: int8x16_t,
    acc_lo0: int32x4_t,
    acc_lo1: int32x4_t,
    acc_hi0: int32x4_t,
    acc_hi1: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let zero = vdupq_n_s8(0);
    let p_lo = vzip1q_s8(p34, p012);
    let p_hi = vzip2q_s8(p34, p012);

    let quad0 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(p_lo), vreinterpretq_s16_s8(zero)));
    let quad1 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(p_lo), vreinterpretq_s16_s8(zero)));
    let quad2 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(p_hi), vreinterpretq_s16_s8(zero)));
    let quad3 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(p_hi), vreinterpretq_s16_s8(zero)));

    (
        vdotq_s32(acc_lo0, quad0, ones),
        vdotq_s32(acc_lo1, quad1, ones),
        vdotq_s32(acc_hi0, quad2, ones),
        vdotq_s32(acc_hi1, quad3, ones),
    )
}

/// Evaluates a 32-row block of weights against precomputed QLUT using ARMv8.2-A `FEAT_DotProd`
/// (`vdotq_s32`) with direct 32-bit hardware accumulation.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv_lut_block32_neon_dotprod(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);
    let ones = vdupq_n_s8(1);

    let mut acc0 = vdupq_n_s32(0); // Rows 0..3
    let mut acc1 = vdupq_n_s32(0); // Rows 4..7
    let mut acc2 = vdupq_n_s32(0); // Rows 8..11
    let mut acc3 = vdupq_n_s32(0); // Rows 12..15
    let mut acc4 = vdupq_n_s32(0); // Rows 16..19
    let mut acc5 = vdupq_n_s32(0); // Rows 20..23
    let mut acc6 = vdupq_n_s32(0); // Rows 24..27
    let mut acc7 = vdupq_n_s32(0); // Rows 28..31

    let mut c = 0;
    while c + 1 < bytes_per_row {
        let cur_c0 = c;
        let cur_c1 = c + 1;

        let qlut_ptr0 = qlut.add(cur_c0 * 64);
        let qlut_ptr1 = qlut.add(cur_c1 * 64);

        let tbl_34_0 = vld1q_s8(qlut_ptr0);
        let tbl_012_0 = vld1q_s8_x2(qlut_ptr0.add(16));

        let tbl_34_1 = vld1q_s8(qlut_ptr1);
        let tbl_012_1 = vld1q_s8_x2(qlut_ptr1.add(16));

        let w_ptr0 = block_weights.add(cur_c0 * 32);
        let w_ptr1 = block_weights.add(cur_c1 * 32);

        let w0_lo = vld1q_u8(w_ptr0);
        let w0_hi = vld1q_u8(w_ptr0.add(16));

        let w1_lo = vld1q_u8(w_ptr1);
        let w1_hi = vld1q_u8(w_ptr1.add(16));

        let (p34_0_lo, p012_0_lo, p34_0_hi, p012_0_hi) =
            eval_column_terms_32rows(w0_lo, w0_hi, tbl_34_0, tbl_012_0, c_28);
        let (p34_1_lo, p012_1_lo, p34_1_hi, p012_1_hi) =
            eval_column_terms_32rows(w1_lo, w1_hi, tbl_34_1, tbl_012_1, c_28);

        let (new_a0, new_a1, new_a2, new_a3) = accumulate_pair_dotprod_16rows(
            p34_0_lo, p012_0_lo, p34_1_lo, p012_1_lo, ones, acc0, acc1, acc2, acc3,
        );
        acc0 = new_a0;
        acc1 = new_a1;
        acc2 = new_a2;
        acc3 = new_a3;

        let (new_a4, new_a5, new_a6, new_a7) = accumulate_pair_dotprod_16rows(
            p34_0_hi, p012_0_hi, p34_1_hi, p012_1_hi, ones, acc4, acc5, acc6, acc7,
        );
        acc4 = new_a4;
        acc5 = new_a5;
        acc6 = new_a6;
        acc7 = new_a7;

        c += 2;
    }

    if c < bytes_per_row {
        let cur_c0 = c;
        let qlut_ptr0 = qlut.add(cur_c0 * 64);
        let tbl_34_0 = vld1q_s8(qlut_ptr0);
        let tbl_012_0 = vld1q_s8_x2(qlut_ptr0.add(16));

        let w_ptr0 = block_weights.add(cur_c0 * 32);
        let w0_lo = vld1q_u8(w_ptr0);
        let w0_hi = vld1q_u8(w_ptr0.add(16));

        let (p34_0_lo, p012_0_lo, p34_0_hi, p012_0_hi) =
            eval_column_terms_32rows(w0_lo, w0_hi, tbl_34_0, tbl_012_0, c_28);

        let (new_a0, new_a1, new_a2, new_a3) = accumulate_single_dotprod_16rows(
            p34_0_lo, p012_0_lo, ones, acc0, acc1, acc2, acc3,
        );
        acc0 = new_a0;
        acc1 = new_a1;
        acc2 = new_a2;
        acc3 = new_a3;

        let (new_a4, new_a5, new_a6, new_a7) = accumulate_single_dotprod_16rows(
            p34_0_hi, p012_0_hi, ones, acc4, acc5, acc6, acc7,
        );
        acc4 = new_a4;
        acc5 = new_a5;
        acc6 = new_a6;
        acc7 = new_a7;
    }

    vst1q_f32(out_32, vcvtq_f32_s32(acc0));
    vst1q_f32(out_32.add(4), vcvtq_f32_s32(acc1));
    vst1q_f32(out_32.add(8), vcvtq_f32_s32(acc2));
    vst1q_f32(out_32.add(12), vcvtq_f32_s32(acc3));
    vst1q_f32(out_32.add(16), vcvtq_f32_s32(acc4));
    vst1q_f32(out_32.add(20), vcvtq_f32_s32(acc5));
    vst1q_f32(out_32.add(24), vcvtq_f32_s32(acc6));
    vst1q_f32(out_32.add(28), vcvtq_f32_s32(acc7));
}

/// Evaluates a pair of 32-row blocks (BM = 64, rows 0..63) with shared activation table reuse
/// using ARMv8.2-A `FEAT_DotProd` with direct 32-bit hardware accumulation.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv_lut_block64_neon_dotprod(
    block_weights_0: *const u8,
    block_weights_1: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_0: *mut f32,
    out_1: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);
    let ones = vdupq_n_s8(1);

    // 32-bit accumulators for Block 0 (Rows 0..31)
    let mut acc0_0 = vdupq_n_s32(0);
    let mut acc1_0 = vdupq_n_s32(0);
    let mut acc2_0 = vdupq_n_s32(0);
    let mut acc3_0 = vdupq_n_s32(0);
    let mut acc4_0 = vdupq_n_s32(0);
    let mut acc5_0 = vdupq_n_s32(0);
    let mut acc6_0 = vdupq_n_s32(0);
    let mut acc7_0 = vdupq_n_s32(0);

    // 32-bit accumulators for Block 1 (Rows 32..63)
    let mut acc0_1 = vdupq_n_s32(0);
    let mut acc1_1 = vdupq_n_s32(0);
    let mut acc2_1 = vdupq_n_s32(0);
    let mut acc3_1 = vdupq_n_s32(0);
    let mut acc4_1 = vdupq_n_s32(0);
    let mut acc5_1 = vdupq_n_s32(0);
    let mut acc6_1 = vdupq_n_s32(0);
    let mut acc7_1 = vdupq_n_s32(0);

    let mut c = 0;
    while c + 1 < bytes_per_row {
        let cur_c0 = c;
        let cur_c1 = c + 1;

        let qlut_ptr0 = qlut.add(cur_c0 * 64);
        let qlut_ptr1 = qlut.add(cur_c1 * 64);

        let tbl_34_0 = vld1q_s8(qlut_ptr0);
        let tbl_012_0 = vld1q_s8_x2(qlut_ptr0.add(16));

        let tbl_34_1 = vld1q_s8(qlut_ptr1);
        let tbl_012_1 = vld1q_s8_x2(qlut_ptr1.add(16));

        // Interleaved evaluation for Column 0 across Block 0 and Block 1
        let w_ptr_0_0 = block_weights_0.add(cur_c0 * 32);
        let w_ptr_1_0 = block_weights_1.add(cur_c0 * 32);
        let w0_0_lo = vld1q_u8(w_ptr_0_0);
        let w0_0_hi = vld1q_u8(w_ptr_0_0.add(16));
        let w1_0_lo = vld1q_u8(w_ptr_1_0);
        let w1_0_hi = vld1q_u8(w_ptr_1_0.add(16));

        let (
            (p34_0_0_lo, p012_0_0_lo, p34_0_0_hi, p012_0_0_hi),
            (p34_1_0_lo, p012_1_0_lo, p34_1_0_hi, p012_1_0_hi),
        ) = eval_column_terms_64rows_interleaved(
            w0_0_lo, w0_0_hi, w1_0_lo, w1_0_hi, tbl_34_0, tbl_012_0, c_28,
        );

        // Interleaved evaluation for Column 1 across Block 0 and Block 1
        let w_ptr_0_1 = block_weights_0.add(cur_c1 * 32);
        let w_ptr_1_1 = block_weights_1.add(cur_c1 * 32);
        let w0_1_lo = vld1q_u8(w_ptr_0_1);
        let w0_1_hi = vld1q_u8(w_ptr_0_1.add(16));
        let w1_1_lo = vld1q_u8(w_ptr_1_1);
        let w1_1_hi = vld1q_u8(w_ptr_1_1.add(16));

        let (
            (p34_0_1_lo, p012_0_1_lo, p34_0_1_hi, p012_0_1_hi),
            (p34_1_1_lo, p012_1_1_lo, p34_1_1_hi, p012_1_1_hi),
        ) = eval_column_terms_64rows_interleaved(
            w0_1_lo, w0_1_hi, w1_1_lo, w1_1_hi, tbl_34_1, tbl_012_1, c_28,
        );

        // Accumulate Block 0
        let (new_a0_0, new_a1_0, new_a2_0, new_a3_0) = accumulate_pair_dotprod_16rows(
            p34_0_0_lo, p012_0_0_lo, p34_0_1_lo, p012_0_1_lo, ones, acc0_0, acc1_0, acc2_0, acc3_0,
        );
        let (new_a4_0, new_a5_0, new_a6_0, new_a7_0) = accumulate_pair_dotprod_16rows(
            p34_0_0_hi, p012_0_0_hi, p34_0_1_hi, p012_0_1_hi, ones, acc4_0, acc5_0, acc6_0, acc7_0,
        );

        // Accumulate Block 1 (alternating with Block 0 to exploit late forwarding across register pairs)
        let (new_a0_1, new_a1_1, new_a2_1, new_a3_1) = accumulate_pair_dotprod_16rows(
            p34_1_0_lo, p012_1_0_lo, p34_1_1_lo, p012_1_1_lo, ones, acc0_1, acc1_1, acc2_1, acc3_1,
        );
        let (new_a4_1, new_a5_1, new_a6_1, new_a7_1) = accumulate_pair_dotprod_16rows(
            p34_1_0_hi, p012_1_0_hi, p34_1_1_hi, p012_1_1_hi, ones, acc4_1, acc5_1, acc6_1, acc7_1,
        );

        acc0_0 = new_a0_0;
        acc1_0 = new_a1_0;
        acc2_0 = new_a2_0;
        acc3_0 = new_a3_0;
        acc4_0 = new_a4_0;
        acc5_0 = new_a5_0;
        acc6_0 = new_a6_0;
        acc7_0 = new_a7_0;

        acc0_1 = new_a0_1;
        acc1_1 = new_a1_1;
        acc2_1 = new_a2_1;
        acc3_1 = new_a3_1;
        acc4_1 = new_a4_1;
        acc5_1 = new_a5_1;
        acc6_1 = new_a6_1;
        acc7_1 = new_a7_1;

        c += 2;
    }

    if c < bytes_per_row {
        let cur_c0 = c;
        let qlut_ptr0 = qlut.add(cur_c0 * 64);
        let tbl_34_0 = vld1q_s8(qlut_ptr0);
        let tbl_012_0 = vld1q_s8_x2(qlut_ptr0.add(16));

        let w_ptr_0_0 = block_weights_0.add(cur_c0 * 32);
        let w_ptr_1_0 = block_weights_1.add(cur_c0 * 32);
        let w0_0_lo = vld1q_u8(w_ptr_0_0);
        let w0_0_hi = vld1q_u8(w_ptr_0_0.add(16));
        let w1_0_lo = vld1q_u8(w_ptr_1_0);
        let w1_0_hi = vld1q_u8(w_ptr_1_0.add(16));

        let (
            (p34_0_0_lo, p012_0_0_lo, p34_0_0_hi, p012_0_0_hi),
            (p34_1_0_lo, p012_1_0_lo, p34_1_0_hi, p012_1_0_hi),
        ) = eval_column_terms_64rows_interleaved(
            w0_0_lo, w0_0_hi, w1_0_lo, w1_0_hi, tbl_34_0, tbl_012_0, c_28,
        );

        let (new_a0_0, new_a1_0, new_a2_0, new_a3_0) = accumulate_single_dotprod_16rows(
            p34_0_0_lo, p012_0_0_lo, ones, acc0_0, acc1_0, acc2_0, acc3_0,
        );
        let (new_a4_0, new_a5_0, new_a6_0, new_a7_0) = accumulate_single_dotprod_16rows(
            p34_0_0_hi, p012_0_0_hi, ones, acc4_0, acc5_0, acc6_0, acc7_0,
        );

        let (new_a0_1, new_a1_1, new_a2_1, new_a3_1) = accumulate_single_dotprod_16rows(
            p34_1_0_lo, p012_1_0_lo, ones, acc0_1, acc1_1, acc2_1, acc3_1,
        );
        let (new_a4_1, new_a5_1, new_a6_1, new_a7_1) = accumulate_single_dotprod_16rows(
            p34_1_0_hi, p012_1_0_hi, ones, acc4_1, acc5_1, acc6_1, acc7_1,
        );

        acc0_0 = new_a0_0;
        acc1_0 = new_a1_0;
        acc2_0 = new_a2_0;
        acc3_0 = new_a3_0;
        acc4_0 = new_a4_0;
        acc5_0 = new_a5_0;
        acc6_0 = new_a6_0;
        acc7_0 = new_a7_0;

        acc0_1 = new_a0_1;
        acc1_1 = new_a1_1;
        acc2_1 = new_a2_1;
        acc3_1 = new_a3_1;
        acc4_1 = new_a4_1;
        acc5_1 = new_a5_1;
        acc6_1 = new_a6_1;
        acc7_1 = new_a7_1;
    }

    vst1q_f32(out_0, vcvtq_f32_s32(acc0_0));
    vst1q_f32(out_0.add(4), vcvtq_f32_s32(acc1_0));
    vst1q_f32(out_0.add(8), vcvtq_f32_s32(acc2_0));
    vst1q_f32(out_0.add(12), vcvtq_f32_s32(acc3_0));
    vst1q_f32(out_0.add(16), vcvtq_f32_s32(acc4_0));
    vst1q_f32(out_0.add(20), vcvtq_f32_s32(acc5_0));
    vst1q_f32(out_0.add(24), vcvtq_f32_s32(acc6_0));
    vst1q_f32(out_0.add(28), vcvtq_f32_s32(acc7_0));

    vst1q_f32(out_1, vcvtq_f32_s32(acc0_1));
    vst1q_f32(out_1.add(4), vcvtq_f32_s32(acc1_1));
    vst1q_f32(out_1.add(8), vcvtq_f32_s32(acc2_1));
    vst1q_f32(out_1.add(12), vcvtq_f32_s32(acc3_1));
    vst1q_f32(out_1.add(16), vcvtq_f32_s32(acc4_1));
    vst1q_f32(out_1.add(20), vcvtq_f32_s32(acc5_1));
    vst1q_f32(out_1.add(24), vcvtq_f32_s32(acc6_1));
    vst1q_f32(out_1.add(28), vcvtq_f32_s32(acc7_1));
}


// ============================================================================
// Fused hi/lo ("wide") kernel
// ============================================================================

/// One column for 32 rows against the `lo` and `hi` activation tables.
///
/// The weight decoding (`h34`, `r3`) is shared; only the table lookups are done twice. The returned
/// int16 sums are `lo + 16 * hi` per row group (rows 0..7, 8..15, 16..23, 24..31).
#[inline(always)]
unsafe fn eval_column_32rows_wide(
    w_lo: uint8x16_t,
    w_hi: uint8x16_t,
    tbl_34_lo: int8x16_t,
    tbl_012_lo: int8x16x2_t,
    tbl_34_hi: int8x16_t,
    tbl_012_hi: int8x16x2_t,
    c_28: uint8x16_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let v11_lo = vshrq_n_u8(w_lo, 2);
    let v11_hi = vshrq_n_u8(w_hi, 2);
    let v15_lo = vshrq_n_u8(w_lo, 5);
    let v15_hi = vshrq_n_u8(w_hi, 5);
    let v17_lo = vrhaddq_u8(v11_lo, v15_lo);
    let v17_hi = vrhaddq_u8(v11_hi, v15_hi);
    let h34_lo = vshrq_n_u8(v17_lo, 2);
    let h34_hi = vshrq_n_u8(v17_hi, 2);
    let r3_lo = vmlsq_u8(w_lo, h34_lo, c_28);
    let r3_hi = vmlsq_u8(w_hi, h34_hi, c_28);

    // lo activation table
    let a34_lo = vqtbl1q_s8(tbl_34_lo, h34_lo);
    let a34_hi = vqtbl1q_s8(tbl_34_lo, h34_hi);
    let a012_lo = vqtbl2q_s8(tbl_012_lo, r3_lo);
    let a012_hi = vqtbl2q_s8(tbl_012_lo, r3_hi);
    // hi activation table
    let b34_lo = vqtbl1q_s8(tbl_34_hi, h34_lo);
    let b34_hi = vqtbl1q_s8(tbl_34_hi, h34_hi);
    let b012_lo = vqtbl2q_s8(tbl_012_hi, r3_lo);
    let b012_hi = vqtbl2q_s8(tbl_012_hi, r3_hi);

    let combine = |a: int16x8_t, b: int16x8_t| vmlaq_n_s16(a, b, 16);
    (
        combine(
            vaddl_s8(vget_low_s8(a34_lo), vget_low_s8(a012_lo)),
            vaddl_s8(vget_low_s8(b34_lo), vget_low_s8(b012_lo)),
        ),
        combine(vaddl_high_s8(a34_lo, a012_lo), vaddl_high_s8(b34_lo, b012_lo)),
        combine(
            vaddl_s8(vget_low_s8(a34_hi), vget_low_s8(a012_hi)),
            vaddl_s8(vget_low_s8(b34_hi), vget_low_s8(b012_hi)),
        ),
        combine(vaddl_high_s8(a34_hi, a012_hi), vaddl_high_s8(b34_hi, b012_hi)),
    )
}

/// Fused variant of [`gemv_lut_block32_neon`] for a wide activation table (`x = 16 * hi + lo`):
/// the weights are decoded once and both tables are looked up, producing `W lo + 16 * (W hi)`.
/// Per column the combined magnitude is at most 40 + 16 * 40; 32 columns fit into int16.
pub unsafe fn gemv_lut_block32_wide_neon(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    let mut acc4 = vdupq_n_s32(0);
    let mut acc5 = vdupq_n_s32(0);
    let mut acc6 = vdupq_n_s32(0);
    let mut acc7 = vdupq_n_s32(0);

    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(32);

        let mut sum_r0_r7 = vdupq_n_s16(0);
        let mut sum_r8_r15 = vdupq_n_s16(0);
        let mut sum_r16_r23 = vdupq_n_s16(0);
        let mut sum_r24_r31 = vdupq_n_s16(0);

        let mut lo_ptr = qlut_lo.add(c * 64);
        let mut hi_ptr = qlut_hi.add(c * 64);
        let mut w_ptr = block_weights.add(c * 32);

        for _ in 0..chunk_len {
            prefetch_l1(w_ptr.add(128));

            let tbl_34_lo = vld1q_s8(lo_ptr);
            let tbl_012_lo = vld1q_s8_x2(lo_ptr.add(16));
            let tbl_34_hi = vld1q_s8(hi_ptr);
            let tbl_012_hi = vld1q_s8_x2(hi_ptr.add(16));
            lo_ptr = lo_ptr.add(64);
            hi_ptr = hi_ptr.add(64);

            let w_lo = vld1q_u8(w_ptr);
            let w_hi = vld1q_u8(w_ptr.add(16));
            w_ptr = w_ptr.add(32);

            let (r0, r1, r2, r3) =
                eval_column_32rows_wide(w_lo, w_hi, tbl_34_lo, tbl_012_lo, tbl_34_hi, tbl_012_hi, c_28);

            sum_r0_r7 = vaddq_s16(sum_r0_r7, r0);
            sum_r8_r15 = vaddq_s16(sum_r8_r15, r1);
            sum_r16_r23 = vaddq_s16(sum_r16_r23, r2);
            sum_r24_r31 = vaddq_s16(sum_r24_r31, r3);
        }

        acc0 = vaddw_s16(acc0, vget_low_s16(sum_r0_r7));
        acc1 = vaddw_s16(acc1, vget_high_s16(sum_r0_r7));
        acc2 = vaddw_s16(acc2, vget_low_s16(sum_r8_r15));
        acc3 = vaddw_s16(acc3, vget_high_s16(sum_r8_r15));
        acc4 = vaddw_s16(acc4, vget_low_s16(sum_r16_r23));
        acc5 = vaddw_s16(acc5, vget_high_s16(sum_r16_r23));
        acc6 = vaddw_s16(acc6, vget_low_s16(sum_r24_r31));
        acc7 = vaddw_s16(acc7, vget_high_s16(sum_r24_r31));

        c += chunk_len;
    }

    vst1q_f32(out_32, vcvtq_f32_s32(acc0));
    vst1q_f32(out_32.add(4), vcvtq_f32_s32(acc1));
    vst1q_f32(out_32.add(8), vcvtq_f32_s32(acc2));
    vst1q_f32(out_32.add(12), vcvtq_f32_s32(acc3));
    vst1q_f32(out_32.add(16), vcvtq_f32_s32(acc4));
    vst1q_f32(out_32.add(20), vcvtq_f32_s32(acc5));
    vst1q_f32(out_32.add(24), vcvtq_f32_s32(acc6));
    vst1q_f32(out_32.add(28), vcvtq_f32_s32(acc7));
}


// ============================================================================
// Experimental variant of the fused hi/lo kernel
// ============================================================================

/// One column for 32 rows against both tables, products summed per table in int8 (`|sum| <= 40`):
/// returns `(lo rows 0..15, hi rows 0..15, lo rows 16..31, hi rows 16..31)`.
#[inline(always)]
unsafe fn column_sums_i8_neon(
    w_lo: uint8x16_t,
    w_hi: uint8x16_t,
    tbl_34_lo: int8x16_t,
    tbl_012_lo: int8x16x2_t,
    tbl_34_hi: int8x16_t,
    tbl_012_hi: int8x16x2_t,
    c_28: uint8x16_t,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    let h34_lo = vshrq_n_u8(vrhaddq_u8(vshrq_n_u8(w_lo, 2), vshrq_n_u8(w_lo, 5)), 2);
    let h34_hi = vshrq_n_u8(vrhaddq_u8(vshrq_n_u8(w_hi, 2), vshrq_n_u8(w_hi, 5)), 2);
    let r3_lo = vmlsq_u8(w_lo, h34_lo, c_28);
    let r3_hi = vmlsq_u8(w_hi, h34_hi, c_28);
    (
        vaddq_s8(vqtbl1q_s8(tbl_34_lo, h34_lo), vqtbl2q_s8(tbl_012_lo, r3_lo)),
        vaddq_s8(vqtbl1q_s8(tbl_34_hi, h34_lo), vqtbl2q_s8(tbl_012_hi, r3_lo)),
        vaddq_s8(vqtbl1q_s8(tbl_34_lo, h34_hi), vqtbl2q_s8(tbl_012_lo, r3_hi)),
        vaddq_s8(vqtbl1q_s8(tbl_34_hi, h34_hi), vqtbl2q_s8(tbl_012_hi, r3_hi)),
    )
}

/// Variant `i8red` of [`gemv_lut_block32_wide_neon`]: the per-table sums of up to three columns are added in
/// int8 (<= 120) and widened once per group; `lo + 16 * hi` is formed with `vaddw` / `vmlal`. The result is identical.
pub unsafe fn gemv_lut_block32_wide_neon_i8red(
    block_weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);
    let c_16 = vdupq_n_s8(16);

    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);
    let mut acc4 = vdupq_n_s32(0);
    let mut acc5 = vdupq_n_s32(0);
    let mut acc6 = vdupq_n_s32(0);
    let mut acc7 = vdupq_n_s32(0);

    // 15 groups of three columns: |lo + 16 hi| <= 2040 per group and 15 * 2040 < 32768.
    const CHUNK: usize = 45;
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut s0 = vdupq_n_s16(0); // rows 0..7
        let mut s1 = vdupq_n_s16(0); // rows 8..15
        let mut s2 = vdupq_n_s16(0); // rows 16..23
        let mut s3 = vdupq_n_s16(0); // rows 24..31

        let mut ci = 0;
        while ci < chunk_len {
            let n = (chunk_len - ci).min(3);
            let mut la = vdupq_n_s8(0);
            let mut ha = vdupq_n_s8(0);
            let mut lb = vdupq_n_s8(0);
            let mut hb = vdupq_n_s8(0);
            for k in 0..n {
                let col = c + ci + k;
                let lo_ptr = qlut_lo.add(col * 64);
                let hi_ptr = qlut_hi.add(col * 64);
                let w_ptr = block_weights.add(col * 32);
                prefetch_l1(w_ptr.add(128));
                let (a, b, d, e) = column_sums_i8_neon(
                    vld1q_u8(w_ptr),
                    vld1q_u8(w_ptr.add(16)),
                    vld1q_s8(lo_ptr),
                    vld1q_s8_x2(lo_ptr.add(16)),
                    vld1q_s8(hi_ptr),
                    vld1q_s8_x2(hi_ptr.add(16)),
                    c_28,
                );
                la = vaddq_s8(la, a);
                ha = vaddq_s8(ha, b);
                lb = vaddq_s8(lb, d);
                hb = vaddq_s8(hb, e);
            }
            s0 = vmlal_s8(vaddw_s8(s0, vget_low_s8(la)), vget_low_s8(ha), vget_low_s8(c_16));
            s1 = vmlal_high_s8(vaddw_high_s8(s1, la), ha, c_16);
            s2 = vmlal_s8(vaddw_s8(s2, vget_low_s8(lb)), vget_low_s8(hb), vget_low_s8(c_16));
            s3 = vmlal_high_s8(vaddw_high_s8(s3, lb), hb, c_16);
            ci += 3;
        }

        acc0 = vaddw_s16(acc0, vget_low_s16(s0));
        acc1 = vaddw_s16(acc1, vget_high_s16(s0));
        acc2 = vaddw_s16(acc2, vget_low_s16(s1));
        acc3 = vaddw_s16(acc3, vget_high_s16(s1));
        acc4 = vaddw_s16(acc4, vget_low_s16(s2));
        acc5 = vaddw_s16(acc5, vget_high_s16(s2));
        acc6 = vaddw_s16(acc6, vget_low_s16(s3));
        acc7 = vaddw_s16(acc7, vget_high_s16(s3));

        c += chunk_len;
    }

    vst1q_f32(out_32, vcvtq_f32_s32(acc0));
    vst1q_f32(out_32.add(4), vcvtq_f32_s32(acc1));
    vst1q_f32(out_32.add(8), vcvtq_f32_s32(acc2));
    vst1q_f32(out_32.add(12), vcvtq_f32_s32(acc3));
    vst1q_f32(out_32.add(16), vcvtq_f32_s32(acc4));
    vst1q_f32(out_32.add(20), vcvtq_f32_s32(acc5));
    vst1q_f32(out_32.add(24), vcvtq_f32_s32(acc6));
    vst1q_f32(out_32.add(28), vcvtq_f32_s32(acc7));
}


// ============================================================================
// Second campaign: several row blocks per pass over the tables (experimental variants)
// ============================================================================

/// Constant tables of the table-based weight decodes.
struct DecodeTables {
    /// `28 * h` for `h` in `0..=8` (16 entries).
    m28: uint8x16_t,
    /// `floor(i / 7)` for `i` in `0..64` (indexed by `w >> 2`).
    div7: uint8x16x4_t,
}

static DECODE_M28_16: [u8; 16] = [0, 28, 56, 84, 112, 140, 168, 196, 224, 0, 0, 0, 0, 0, 0, 0];

static DECODE_DIV7: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = (i / 7) as u8;
        i += 1;
    }
    t
};

/// Weight decode of one 16-byte vector into `(h34, r3)`.
/// `DEC = 0`: shifts and rounding halving add for `h34`, `vmls` for `r3` (the production decode);
/// `DEC = 1`: same `h34`, `r3 = w - tbl(28 * h34)` (no multiply-accumulate);
/// `DEC = 2`: `h34 = tbl4(w >> 2)` from a 64-entry table, `r3` by `vmls`.
#[inline(always)]
unsafe fn decode_neon<const DEC: u8>(w: uint8x16_t, t: &DecodeTables, c_28: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    if DEC == 2 {
        let h = vqtbl4q_u8(t.div7, vshrq_n_u8(w, 2));
        (h, vmlsq_u8(w, h, c_28))
    } else {
        let h = vshrq_n_u8(vrhaddq_u8(vshrq_n_u8(w, 2), vshrq_n_u8(w, 5)), 2);
        if DEC == 1 {
            (h, vsubq_u8(w, vqtbl1q_u8(t.m28, h)))
        } else {
            (h, vmlsq_u8(w, h, c_28))
        }
    }
}

/// [`column_sums_i8_neon`] with a selectable weight decode (see [`decode_neon`]).
#[inline(always)]
unsafe fn column_sums_i8_neon_dec<const DEC: u8>(
    t: &DecodeTables,
    w_lo: uint8x16_t,
    w_hi: uint8x16_t,
    tbl_34_lo: int8x16_t,
    tbl_012_lo: int8x16x2_t,
    tbl_34_hi: int8x16_t,
    tbl_012_hi: int8x16x2_t,
    c_28: uint8x16_t,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    let (h34_lo, r3_lo) = decode_neon::<DEC>(w_lo, t, c_28);
    let (h34_hi, r3_hi) = decode_neon::<DEC>(w_hi, t, c_28);
    (
        vaddq_s8(vqtbl1q_s8(tbl_34_lo, h34_lo), vqtbl2q_s8(tbl_012_lo, r3_lo)),
        vaddq_s8(vqtbl1q_s8(tbl_34_hi, h34_lo), vqtbl2q_s8(tbl_012_hi, r3_lo)),
        vaddq_s8(vqtbl1q_s8(tbl_34_lo, h34_hi), vqtbl2q_s8(tbl_012_lo, r3_hi)),
        vaddq_s8(vqtbl1q_s8(tbl_34_hi, h34_hi), vqtbl2q_s8(tbl_012_hi, r3_hi)),
    )
}

/// `i8red` kernel for `NB` consecutive 32-row blocks (`block_stride` bytes apart) at once. The activation tables of a
/// column are loaded once and applied to the weights of all `NB` blocks. With `SPLIT` the `lo` and `hi` sums of a group
/// are kept in separate int16 accumulators (`vaddw`) and combined as `lo + 16 hi` once per chunk instead of using
/// `vmlal` per group. `PF` is the prefetch distance in bytes ahead of every weight stream (0 = none).
/// The result is bit-identical to [`gemv_lut_block32_wide_neon_i8red`].
pub unsafe fn gemv_lut_blocks_wide_neon_multi<const NB: usize, const SPLIT: bool, const PF: usize, const DEC: u8>(
    weights: *const u8,
    block_stride: usize,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let c_28 = vdupq_n_u8(28);
    let c_16 = vdupq_n_s8(16);
    let dec_tables = DecodeTables { m28: vld1q_u8(DECODE_M28_16.as_ptr()), div7: vld1q_u8_x4(DECODE_DIV7.as_ptr()) };
    let z32 = vdupq_n_s32(0);
    let z16 = vdupq_n_s16(0);
    let z8 = vdupq_n_s8(0);
    let mut acc = [[z32; 8]; NB];

    const CHUNK: usize = 45;
    let mut c = 0;
    while c < bytes_per_row {
        let chunk_len = (bytes_per_row - c).min(CHUNK);
        let mut s = [[z16; 4]; NB]; // lo + 16 hi (not SPLIT) or lo (SPLIT)
        let mut sh = [[z16; 4]; NB]; // hi (SPLIT only)
        let mut ci = 0;
        while ci < chunk_len {
            let n = (chunk_len - ci).min(3);
            let mut la = [z8; NB];
            let mut ha = [z8; NB];
            let mut lb = [z8; NB];
            let mut hb = [z8; NB];
            for k in 0..n {
                let col = c + ci + k;
                let lo_ptr = qlut_lo.add(col * 64);
                let hi_ptr = qlut_hi.add(col * 64);
                let t34_lo = vld1q_s8(lo_ptr);
                let t012_lo = vld1q_s8_x2(lo_ptr.add(16));
                let t34_hi = vld1q_s8(hi_ptr);
                let t012_hi = vld1q_s8_x2(hi_ptr.add(16));
                for blk in 0..NB {
                    let w_ptr = weights.add(blk * block_stride + col * 32);
                    if PF != 0 {
                        prefetch_l1(w_ptr.add(PF));
                    }
                    let (a, b, d, e) = column_sums_i8_neon_dec::<DEC>(
                        &dec_tables,
                        vld1q_u8(w_ptr),
                        vld1q_u8(w_ptr.add(16)),
                        t34_lo,
                        t012_lo,
                        t34_hi,
                        t012_hi,
                        c_28,
                    );
                    la[blk] = vaddq_s8(la[blk], a);
                    ha[blk] = vaddq_s8(ha[blk], b);
                    lb[blk] = vaddq_s8(lb[blk], d);
                    hb[blk] = vaddq_s8(hb[blk], e);
                }
            }
            for blk in 0..NB {
                if SPLIT {
                    s[blk][0] = vaddw_s8(s[blk][0], vget_low_s8(la[blk]));
                    s[blk][1] = vaddw_high_s8(s[blk][1], la[blk]);
                    s[blk][2] = vaddw_s8(s[blk][2], vget_low_s8(lb[blk]));
                    s[blk][3] = vaddw_high_s8(s[blk][3], lb[blk]);
                    sh[blk][0] = vaddw_s8(sh[blk][0], vget_low_s8(ha[blk]));
                    sh[blk][1] = vaddw_high_s8(sh[blk][1], ha[blk]);
                    sh[blk][2] = vaddw_s8(sh[blk][2], vget_low_s8(hb[blk]));
                    sh[blk][3] = vaddw_high_s8(sh[blk][3], hb[blk]);
                } else {
                    s[blk][0] = vmlal_s8(vaddw_s8(s[blk][0], vget_low_s8(la[blk])), vget_low_s8(ha[blk]), vget_low_s8(c_16));
                    s[blk][1] = vmlal_high_s8(vaddw_high_s8(s[blk][1], la[blk]), ha[blk], c_16);
                    s[blk][2] = vmlal_s8(vaddw_s8(s[blk][2], vget_low_s8(lb[blk])), vget_low_s8(hb[blk]), vget_low_s8(c_16));
                    s[blk][3] = vmlal_high_s8(vaddw_high_s8(s[blk][3], lb[blk]), hb[blk], c_16);
                }
            }
            ci += n;
        }
        for blk in 0..NB {
            for q in 0..4 {
                let lo16 = s[blk][q];
                let (mut a_lo, mut a_hi) = (
                    vaddw_s16(acc[blk][2 * q], vget_low_s16(lo16)),
                    vaddw_s16(acc[blk][2 * q + 1], vget_high_s16(lo16)),
                );
                if SPLIT {
                    a_lo = vmlal_n_s16(a_lo, vget_low_s16(sh[blk][q]), 16);
                    a_hi = vmlal_n_s16(a_hi, vget_high_s16(sh[blk][q]), 16);
                }
                acc[blk][2 * q] = a_lo;
                acc[blk][2 * q + 1] = a_hi;
            }
        }
        c += chunk_len;
    }

    for blk in 0..NB {
        let o = out.add(blk * 32);
        for q in 0..8 {
            vst1q_f32(o.add(4 * q), vcvtq_f32_s32(acc[blk][q]));
        }
    }
}

/// Runs [`gemv_lut_blocks_wide_neon_multi`] over `nblocks` 32-row blocks in groups of `NB`; a trailing group uses `NB = 1`.
pub unsafe fn gemv_lut_wide_neon_multi_driver<const NB: usize, const SPLIT: bool, const PF: usize, const DEC: u8>(
    weights: *const u8,
    qlut_lo: *const i8,
    qlut_hi: *const i8,
    nblocks: usize,
    bytes_per_row: usize,
    out: *mut f32,
) {
    let stride = 32 * bytes_per_row;
    let mut b = 0;
    while b + NB <= nblocks {
        gemv_lut_blocks_wide_neon_multi::<NB, SPLIT, PF, DEC>(weights.add(b * stride), stride, qlut_lo, qlut_hi, bytes_per_row, out.add(b * 32));
        b += NB;
    }
    while b < nblocks {
        gemv_lut_blocks_wide_neon_multi::<1, SPLIT, PF, DEC>(weights.add(b * stride), stride, qlut_lo, qlut_hi, bytes_per_row, out.add(b * 32));
        b += 1;
    }
}
