#![allow(unsafe_op_in_unsafe_fn)]

//! 32-Row Tiled Fused Dual-Hash Activation-LUT GEMV Architecture.
//!
//! Provides activation lookup table (QLUT) precomputation, 32-row tiled matrix
//! weight transformations, and single/multi-threaded GEMV execution.

use crate::pack::{tile_byte, tile_row_bytes, untile_row, BLOCK_BYTES_5, QK_5};
use crate::simd::scalar::gemv_lut_block32_scalar;

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2::gemv_lut_block32_avx2;

#[cfg(target_arch = "x86_64")]
pub use crate::simd::avx512::{
    activation_lut_len_avx512, avx512_supported, build_activation_lut_avx512, build_activation_lut_wide_avx512, build_activation_lut_wide_avx512_into,
    build_quintrit_lut_avx512,
    gemv_lut_64_1xn_avx512, gemv_lut_64_parallel_avx512,
    gemv_lut_64_wide_1xn_avx512, gemv_lut_64_wide_parallel_avx512, wide_activation_lut_len_avx512,
    gemv_quintrit_lut_64_1xn, gemv_quintrit_lut_64_parallel,
    transform_weights_lut_64, transform_weights_quintrit_lut_64,
    untransform_weights_lut_64, untransform_weights_quintrit_lut_64,
};

#[cfg(target_arch = "aarch64")]
use crate::simd::neon::{gemv_lut_block32_neon, gemv_lut_block64_neon};

/// Deterministic slot mapping for Hash h34 in [0..15] -> (t3, t4) in {-1, 0, +1}
pub const MAP_34: [(i8, i8); 16] = [
    (-1, -1), // Slot 0:  d3=0, d4=0
    ( 0, -1), // Slot 1:  d3=1, d4=0
    ( 1, -1), // Slot 2:  d3=2, d4=0
    (-1,  0), // Slot 3:  d3=0, d4=1
    ( 0,  0), // Slot 4:  d3=1, d4=1
    ( 1,  0), // Slot 5:  d3=2, d4=1
    (-1,  1), // Slot 6:  d3=0, d4=2
    ( 0,  1), // Slot 7:  d3=1, d4=2
    ( 1,  1), // Slot 8:  d3=2, d4=2
    ( 0,  0), // Slot 9:  Unused
    ( 0,  0), // Slot 10: Unused
    ( 0,  0), // Slot 11: Unused
    ( 0,  0), // Slot 12: Unused
    ( 0,  0), // Slot 13: Unused
    ( 0,  0), // Slot 14: Unused
    ( 0,  0), // Slot 15: Unused
];

/// Deterministic slot mapping for Hash h01 in [0..15] -> (t0, t1) in {-1, 0, +1}
pub const MAP_01: [(i8, i8); 16] = [
    (-1, -1), // Slot 0:  d0=0, d1=0
    ( 0,  0), // Slot 1:  Unused
    ( 0, -1), // Slot 2:  d0=1, d1=0
    ( 1, -1), // Slot 3:  d0=2, d1=0
    ( 1, -1), // Slot 4:  d0=2, d1=0
    (-1,  0), // Slot 5:  d0=0, d1=1
    (-1,  0), // Slot 6:  d0=0, d1=1
    ( 0,  0), // Slot 7:  d0=1, d1=1
    ( 0,  0), // Slot 8:  Unused
    ( 1,  0), // Slot 9:  d0=2, d1=1
    (-1,  1), // Slot 10: d0=0, d1=2
    (-1,  1), // Slot 11: d0=0, d1=2
    ( 0,  1), // Slot 12: d0=1, d1=2
    ( 0,  1), // Slot 13: d0=1, d1=2
    ( 1,  1), // Slot 14: d0=2, d1=2
    ( 0,  0), // Slot 15: Unused
];

/// Deterministic slot mapping for Hash u in [0..15] -> t2 in {-1, 0, +1}
pub const MAP_2: [i8; 16] = [
    -1, // Slot 0:  d2=0 (t2=-1)
    -1, // Slot 1:  d2=0 (t2=-1)
    -1, // Slot 2:  d2=0 (t2=-1)
    -1, // Slot 3:  d2=0 (t2=-1)
    -1, // Slot 4:  d2=0 (t2=-1)
     0, // Slot 5:  d2=1 (t2= 0)
     0, // Slot 6:  d2=1 (t2= 0)
     0, // Slot 7:  d2=1 (t2= 0)
     0, // Slot 8:  d2=1 (t2= 0)
     0, // Slot 9:  d2=1 (t2= 0)
     1, // Slot 10: d2=2 (t2=+1)
     1, // Slot 11: d2=2 (t2=+1)
     1, // Slot 12: d2=2 (t2=+1)
     1, // Slot 13: d2=2 (t2=+1)
     1, // Slot 14: d2=2 (t2=+1)
     0, // Slot 15: Unused
];

/// Deterministic slot mapping for Remainder r3 in [0..31] -> (t0, t1, t2) in {-1, 0, +1}
/// Evaluated by r3 = w - 28 * h34 in [0..26]
pub const MAP_012: [(i8, i8, i8); 32] = [
    (-1, -1, -1), // Slot 0:  d0=0, d1=0, d2=0
    ( 0, -1, -1), // Slot 1:  d0=1, d1=0, d2=0
    ( 1, -1, -1), // Slot 2:  d0=2, d1=0, d2=0
    (-1,  0, -1), // Slot 3:  d0=0, d1=1, d2=0
    ( 0,  0, -1), // Slot 4:  d0=1, d1=1, d2=0
    ( 1,  0, -1), // Slot 5:  d0=2, d1=1, d2=0
    (-1,  1, -1), // Slot 6:  d0=0, d1=2, d2=0
    ( 0,  1, -1), // Slot 7:  d0=1, d1=2, d2=0
    ( 1,  1, -1), // Slot 8:  d0=2, d1=2, d2=0
    (-1, -1,  0), // Slot 9:  d0=0, d1=0, d2=1
    ( 0, -1,  0), // Slot 10: d0=1, d1=0, d2=1
    ( 1, -1,  0), // Slot 11: d0=2, d1=0, d2=1
    (-1,  0,  0), // Slot 12: d0=0, d1=1, d2=1
    ( 0,  0,  0), // Slot 13: d0=1, d1=1, d2=1
    ( 1,  0,  0), // Slot 14: d0=2, d1=1, d2=1
    (-1,  1,  0), // Slot 15: d0=0, d1=2, d2=1
    ( 0,  1,  0), // Slot 16: d0=1, d1=2, d2=1
    ( 1,  1,  0), // Slot 17: d0=2, d1=2, d2=1
    (-1, -1,  1), // Slot 18: d0=0, d1=0, d2=2
    ( 0, -1,  1), // Slot 19: d0=1, d1=0, d2=2
    ( 1, -1,  1), // Slot 20: d0=2, d1=0, d2=2
    (-1,  0,  1), // Slot 21: d0=0, d1=1, d2=2
    ( 0,  0,  1), // Slot 22: d0=1, d1=1, d2=2
    ( 1,  0,  1), // Slot 23: d0=2, d1=1, d2=2
    (-1,  1,  1), // Slot 24: d0=0, d1=2, d2=2
    ( 0,  1,  1), // Slot 25: d0=1, d1=2, d2=2
    ( 1,  1,  1), // Slot 26: d0=2, d1=2, d2=2
    ( 0,  0,  0), // Slot 27: Unused
    ( 0,  0,  0), // Slot 28: Unused
    ( 0,  0,  0), // Slot 29: Unused
    ( 0,  0,  0), // Slot 30: Unused
    ( 0,  0,  0), // Slot 31: Unused
];

/// Transforms weights from standard 5-planar layout into 32-row tiled format for LUT GEMV.
///
/// Each block of 32 rows is stored consecutively. For each column byte `c`,
/// the 32 bytes for rows `0..31` are laid out contiguously in memory.
pub fn transform_weights_lut(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(rows % 32, 0, "rows must be a multiple of 32");
    let src_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 32;

    let mut dst = vec![0u8; num_blocks * 32 * bytes_per_row];

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_dst = &mut dst[b * (32 * bytes_per_row)..(b + 1) * (32 * bytes_per_row)];

        for c in 0..bytes_per_row {
            let col_dst = &mut block_dst[c * 32..(c + 1) * 32];
            for r in 0..32 {
                let row = &src[(r_base + r) * src_row_bytes..(r_base + r + 1) * src_row_bytes];
                col_dst[r] = tile_byte(row, cols, c);
            }
        }
    }

    dst
}

/// Restores standard 5-planar layout from 32-row tiled format.
pub fn untransform_weights_lut(tiled: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(rows % 32, 0, "rows must be a multiple of 32");
    let dst_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 32;

    let mut dst = vec![0u8; rows * dst_row_bytes];
    let mut tile_row = vec![0u8; bytes_per_row];

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_src = &tiled[b * (32 * bytes_per_row)..(b + 1) * (32 * bytes_per_row)];

        for r in 0..32 {
            for c in 0..bytes_per_row {
                tile_row[c] = block_src[c * 32 + r];
            }
            let row_dst = &mut dst[(r_base + r) * dst_row_bytes..(r_base + r + 1) * dst_row_bytes];
            untile_row(&tile_row, cols, row_dst);
        }
    }

    dst
}

#[cfg(target_arch = "aarch64")]
unsafe fn build_activation_lut_neon(
    activations: &[i8],
    cols: usize,
    bytes_per_row: usize,
    a2_offset: usize,
    qlut: &mut [i8],
) {
    use core::arch::aarch64::*;

    let t3_coeff = vld1q_s8([-1,  0,  1, -1,  0,  1, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0].as_ptr());
    let t4_coeff = vld1q_s8([-1, -1, -1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0].as_ptr());

    let t0_coeff_lo = vld1q_s8([-1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1].as_ptr());
    let t0_coeff_hi = vld1q_s8([ 0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1,  0,  0,  0,  0,  0].as_ptr());

    let t1_coeff_lo = vld1q_s8([-1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1].as_ptr());
    let t1_coeff_hi = vld1q_s8([ 1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0].as_ptr());

    let t2_coeff_lo = vld1q_s8([-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0].as_ptr());
    let t2_coeff_hi = vld1q_s8([ 0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0].as_ptr());

    for c in 0..bytes_per_row {

        let get_act = |p: usize| -> i8 {
            let idx = crate::pack::column_activation_index(cols, c, p);
            if idx < cols {
                activations[idx]
            } else {
                0
            }
        };

        let a0 = get_act(0);
        let a1 = get_act(1);
        let a2 = get_act(2);
        let a3 = get_act(3);
        let a4 = get_act(4);

        let base = c * 64;

        // Vectorized Table A (tbl_34)
        let a3_vec = vdupq_n_s8(a3);
        let a4_vec = vdupq_n_s8(a4);
        let p34_lo = vmlal_s8(
            vmull_s8(vget_low_s8(t3_coeff), vget_low_s8(a3_vec)),
            vget_low_s8(t4_coeff),
            vget_low_s8(a4_vec),
        );
        let p34_hi = vmlal_high_s8(
            vmull_high_s8(t3_coeff, a3_vec),
            t4_coeff,
            a4_vec,
        );
        let tbl_34 = vcombine_s8(vqmovn_s16(p34_lo), vqmovn_s16(p34_hi));
        vst1q_s8(qlut.as_mut_ptr().add(base), tbl_34);

        // Vectorized Table 012 Lower Half (tbl_012_lo, slots 0..15)
        let a0_vec = vdupq_n_s8(a0);
        let a1_vec = vdupq_n_s8(a1);
        let a2_vec = vdupq_n_s8(a2);

        let p012_lo_0 = vmlal_s8(
            vmlal_s8(
                vmull_s8(vget_low_s8(t0_coeff_lo), vget_low_s8(a0_vec)),
                vget_low_s8(t1_coeff_lo),
                vget_low_s8(a1_vec),
            ),
            vget_low_s8(t2_coeff_lo),
            vget_low_s8(a2_vec),
        );
        let p012_lo_1 = vmlal_high_s8(
            vmlal_high_s8(
                vmull_high_s8(t0_coeff_lo, a0_vec),
                t1_coeff_lo,
                a1_vec,
            ),
            t2_coeff_lo,
            a2_vec,
        );
        let tbl_012_lo = vcombine_s8(vqmovn_s16(p012_lo_0), vqmovn_s16(p012_lo_1));
        vst1q_s8(qlut.as_mut_ptr().add(base + 16), tbl_012_lo);

        // Vectorized Table 012 Upper Half (tbl_012_hi, slots 16..31)
        let p012_hi_0 = vmlal_s8(
            vmlal_s8(
                vmull_s8(vget_low_s8(t0_coeff_hi), vget_low_s8(a0_vec)),
                vget_low_s8(t1_coeff_hi),
                vget_low_s8(a1_vec),
            ),
            vget_low_s8(t2_coeff_hi),
            vget_low_s8(a2_vec),
        );
        let p012_hi_1 = vmlal_high_s8(
            vmlal_high_s8(
                vmull_high_s8(t0_coeff_hi, a0_vec),
                t1_coeff_hi,
                a1_vec,
            ),
            t2_coeff_hi,
            a2_vec,
        );
        let tbl_012_hi = vcombine_s8(vqmovn_s16(p012_hi_0), vqmovn_s16(p012_hi_1));
        vst1q_s8(qlut.as_mut_ptr().add(base + 32), tbl_012_hi);

        // Unused 16-byte tail in 64B column layout
        vst1q_s8(qlut.as_mut_ptr().add(base + 48), vdupq_n_s8(0));

        qlut[a2_offset + c] = a2;
    }
}

/// Precomputes the Dual-Hash Activation Lookup Table (QLUT) for all column groups.
/// Precomputes the Dual-Hash Activation Lookup Table (QLUT) into a caller-provided destination slice.
///
/// Returns the number of bytes written to `qlut`.
pub fn build_activation_lut_into(activations: &[i8], cols: usize, qlut: &mut [i8]) -> usize {
    let bytes_per_row = tile_row_bytes(cols);
    let a2_offset = bytes_per_row * 64;
    let total_len = a2_offset + bytes_per_row.next_multiple_of(32);
    assert!(qlut.len() >= total_len, "Destination buffer too small for QLUT");

    #[cfg(target_arch = "aarch64")]
    unsafe {
        build_activation_lut_neon(activations, cols, bytes_per_row, a2_offset, qlut);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
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

            let base = c * 64;

            // 1. Precompute Table A (Upper Trits d3, d4 via h34)
            let mut t34 = [0i8; 16];
            for s in 0..16usize {
                let (t3, t4) = MAP_34[s];
                let val = t3 as i16 * a3 + t4 as i16 * a4;
                t34[s] = val.clamp(-128, 127) as i8;
            }
            qlut[base..base + 16].copy_from_slice(&t34);
            qlut[base + 16..base + 32].copy_from_slice(&t34);

            // 2. Precompute Table B (Lower Trits d0, d1 via h01)
            let mut t01 = [0i8; 16];
            for s in 0..16usize {
                let (t0, t1) = MAP_01[s];
                let val = t0 as i16 * a0 + t1 as i16 * a1;
                t01[s] = val.clamp(-128, 127) as i8;
            }
            qlut[base + 32..base + 48].copy_from_slice(&t01);
            qlut[base + 48..base + 64].copy_from_slice(&t01);

            qlut[a2_offset + c] = a2 as i8;
        }
    }

    // With a packed tail the column count is not a multiple of 32: keep the padding of the `a2` array defined.
    qlut[a2_offset + bytes_per_row..total_len].fill(0);

    total_len
}

/// Precomputes the Dual-Hash Activation Lookup Table (QLUT) for all column groups.
///
/// Stores 64 bytes per column plus a contiguous tail array of a2 activations.
pub fn build_activation_lut(activations: &[i8], cols: usize) -> Vec<i8> {
    let bytes_per_row = tile_row_bytes(cols);
    let a2_offset = bytes_per_row * 64;
    let total_len = a2_offset + bytes_per_row.next_multiple_of(32);
    let mut qlut = vec![0i8; total_len];
    build_activation_lut_into(activations, cols, &mut qlut);
    qlut
}


/// Splits int8 activations into two small parts with `x = 16 * hi + lo`,
/// where `lo` lies in `[-8, 7]` and `hi` in `[-8, 8]`.
///
/// The int8 activation tables of the LUT kernels hold sums of two or three products, which only fit
/// into int8 for activations of roughly 7 bits. Running the kernel once on `hi` and once on `lo`
/// (and recombining the integer results as `16 * hi_result + lo_result`) keeps every table entry
/// below 25 in magnitude, so the full int8 activation range becomes exact.
pub fn split_activations_hi_lo(acts: &[i8], hi: &mut [i8], lo: &mut [i8]) {
    assert!(hi.len() >= acts.len() && lo.len() >= acts.len());
    for (i, &x) in acts.iter().enumerate() {
        let x = x as i16;
        let l = ((x + 8) & 15) - 8;
        lo[i] = l as i8;
        hi[i] = ((x - l) >> 4) as i8;
    }
}

/// Length in bytes of the table produced by [`build_activation_lut`] / [`build_activation_lut_into`] for `cols`
/// columns: 64 bytes per tile column plus the `a2` array padded to a multiple of 32.
pub fn activation_lut_len(cols: usize) -> usize {
    let bytes_per_row = tile_row_bytes(cols);
    bytes_per_row * 64 + bytes_per_row.next_multiple_of(32)
}

/// Length in bytes of the table produced by [`build_activation_lut_wide_into`] for `cols` columns:
/// two consecutive single tables, `[lo | hi]`.
pub fn wide_activation_lut_len(cols: usize) -> usize {
    2 * activation_lut_len(cols)
}

/// Checks the buffers of a GEMV over a `rows x cols` matrix stored in `tile_rows`-row tiles before any kernel reads
/// them through raw pointers: `rows` must be a multiple of the tile height, `weights` must hold `rows` tile rows,
/// `table` at least `table_len` entries and `out` at least `rows` values.
///
/// Every safe GEMV entry point calls this, so a wrong length is a panic and never an out-of-bounds access.
#[track_caller]
pub(crate) fn check_gemv_buffers(
    tile_rows: usize,
    weights: &[u8],
    table: &[i8],
    table_len: usize,
    rows: usize,
    cols: usize,
    out: &[f32],
) {
    assert_eq!(rows % tile_rows, 0, "rows must be a multiple of {tile_rows}");
    assert!(out.len() >= rows, "output buffer too small: {} values for {rows} rows", out.len());
    let weight_bytes = rows
        .checked_mul(tile_row_bytes(cols))
        .expect("rows * bytes per row overflows usize");
    assert!(
        weights.len() >= weight_bytes,
        "weight buffer too small: {} bytes, a {rows} x {cols} tiled matrix needs {weight_bytes}",
        weights.len()
    );
    assert!(
        table.len() >= table_len,
        "activation table too small: {} bytes, {cols} columns need {table_len}",
        table.len()
    );
}

/// Builds the pair of activation tables (`[lo | hi]`, see [`split_activations_hi_lo`]) that make the
/// LUT GEMV exact for the full int8 activation range.
///
/// `split_scratch` must hold at least `2 * cols` values and `qlut` at least
/// [`wide_activation_lut_len`] bytes. Returns the number of bytes written to `qlut`.
pub fn build_activation_lut_wide_into(
    activations: &[i8],
    cols: usize,
    split_scratch: &mut [i8],
    qlut: &mut [i8],
) -> usize {
    let total = wide_activation_lut_len(cols);
    let single = total / 2;
    assert!(split_scratch.len() >= 2 * cols, "split scratch too small");
    assert!(qlut.len() >= total, "Destination buffer too small for wide QLUT");
    let (hi, lo) = split_scratch[..2 * cols].split_at_mut(cols);
    split_activations_hi_lo(&activations[..cols], hi, lo);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // The parts have magnitude <= 8, inside the range of the vectorized builder.
            unsafe {
                crate::simd::avx2::build_activation_lut_small_avx2(lo, cols, &mut qlut[..single]);
                crate::simd::avx2::build_activation_lut_small_avx2(hi, cols, &mut qlut[single..total]);
            }
            return total;
        }
    }
    build_activation_lut_into(lo, cols, &mut qlut[..single]);
    build_activation_lut_into(hi, cols, &mut qlut[single..total]);
    total
}

/// Allocating convenience wrapper around [`build_activation_lut_wide_into`].
pub fn build_activation_lut_wide(activations: &[i8], cols: usize) -> Vec<i8> {
    let mut scratch = vec![0i8; 2 * cols];
    let mut qlut = vec![0i8; wide_activation_lut_len(cols)];
    build_activation_lut_wide_into(activations, cols, &mut scratch, &mut qlut);
    qlut
}

/// Target size in bytes of the weight block that both hi/lo passes of the wide GEMV process
/// back to back, so the second pass reads the weights from cache instead of DRAM.
const WIDE_BLOCK_BYTES: usize = 128 * 1024;

/// Rows per cache block for rows of `bytes_per_row` bytes: a multiple of 64 when possible
/// (the NEON kernel processes two 32-row tiles at once), never below 32 and never above 256.
fn wide_block_rows(bytes_per_row: usize) -> usize {
    let rows = (WIDE_BLOCK_BYTES / bytes_per_row.max(1)) / 64 * 64;
    if rows == 0 { 32 } else { rows.min(256) }
}

/// Single-threaded GEMV against a wide activation table (`[lo | hi]`), returning the exact integer
/// result `W lo + 16 * (W hi)` as f32.
///
/// The rows are processed in cache-sized blocks and both passes run on each block, so the weights
/// cross the memory bus once even though the kernel runs twice.
pub fn gemv_lut_wide_1xn(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(wide_qlut.len(), wide_activation_lut_len(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(32, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);

    // Fused kernels decode the weights once and look up both tables.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                for b in 0..rows / 32 {
                    let block_w = transformed_weights.as_ptr().add(b * (32 * bytes_per_row));
                    let block_out = out.as_mut_ptr().add(b * 32);
                    crate::simd::avx2::gemv_lut_block32_wide_avx2_i8red_impl::<2048>(block_w, lo.as_ptr(), hi.as_ptr(), bytes_per_row, block_out);
                }
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            if std::arch::is_aarch64_feature_detected!("i8mm") {
                // Cores of the Neoverse N2 class: two 32-row blocks share every pass over the activation tables
                // (also two weight streams per thread). On a Cortex-A76 class core the DRAM is saturated by the
                // single stream of four threads and the second stream makes the whole GEMV slower (measured).
                crate::simd::neon::gemv_lut_wide_neon_multi_driver::<2, false, 1024, 0>(
                    transformed_weights.as_ptr(),
                    lo.as_ptr(),
                    hi.as_ptr(),
                    rows / 32,
                    bytes_per_row,
                    out.as_mut_ptr(),
                );
                return;
            }
            for b in 0..rows / 32 {
                let block_w = transformed_weights.as_ptr().add(b * (32 * bytes_per_row));
                let block_out = out.as_mut_ptr().add(b * 32);
                crate::simd::neon::gemv_lut_block32_wide_neon_i8red(block_w, lo.as_ptr(), hi.as_ptr(), bytes_per_row, block_out);
            }
        }
        return;
    }

    // Portable fallback: two kernel passes over cache-sized row blocks.
    #[allow(unreachable_code)]
    {
    let step = wide_block_rows(bytes_per_row);

    let mut hi_out = [0.0f32; 256];
    let mut r = 0;
    while r < rows {
        let n = step.min(rows - r);
        let weights = &transformed_weights[r * bytes_per_row..(r + n) * bytes_per_row];
        gemv_lut_1xn(weights, lo, n, cols, &mut out[r..r + n]);
        gemv_lut_1xn(weights, hi, n, cols, &mut hi_out[..n]);
        for (o, h) in out[r..r + n].iter_mut().zip(&hi_out[..n]) {
            *o += 16.0 * *h;
        }
        r += n;
    }
    }
}

/// Multi-threaded variant of [`gemv_lut_wide_1xn`]; rows are split across threads in whole tiles.
pub fn gemv_lut_wide_parallel(
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    assert_eq!(wide_qlut.len(), wide_activation_lut_len(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(32, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let num_blocks = rows / 32;
    if num_threads <= 1 || num_blocks < num_threads {
        gemv_lut_wide_1xn(transformed_weights, wide_qlut, rows, cols, out);
        return;
    }
    let bytes_per_row = tile_row_bytes(cols);
    let blocks_per_thread = num_blocks.div_ceil(num_threads).next_multiple_of(2);
    let rows_per_thread = blocks_per_thread * 32;

    use rayon::prelude::*;
    crate::threading::get_thread_pool(num_threads).install(|| {
        out[..rows]
            .par_chunks_mut(rows_per_thread)
            .zip(transformed_weights[..rows * bytes_per_row].par_chunks(rows_per_thread * bytes_per_row))
            .for_each(|(out_chunk, weights)| {
                gemv_lut_wide_1xn(weights, wide_qlut, out_chunk.len(), cols, out_chunk);
            });
    });
}

/// Single-threaded GEMV matrix-vector multiplication using 32-row tiled Dual-Hash Activation-LUT kernel.
///
/// Automatically dispatches to hardware-accelerated AVX2 or NEON, with portable scalar fallback.
pub fn gemv_lut_1xn(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    check_gemv_buffers(32, transformed_weights, qlut, activation_lut_len(cols), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let num_blocks = rows / 32;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                for b in 0..num_blocks {
                    let block_w = transformed_weights.as_ptr().add(b * (32 * bytes_per_row));
                    let block_out = out.as_mut_ptr().add(b * 32);
                    gemv_lut_block32_avx2(block_w, qlut.as_ptr(), bytes_per_row, block_out);
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let num_pairs = num_blocks / 2;
            for p in 0..num_pairs {
                let b0 = p * 2;
                let b1 = p * 2 + 1;
                let ptr0 = transformed_weights.as_ptr().add(b0 * (32 * bytes_per_row));
                let ptr1 = transformed_weights.as_ptr().add(b1 * (32 * bytes_per_row));
                let out0 = out.as_mut_ptr().add(b0 * 32);
                let out1 = out.as_mut_ptr().add(b1 * 32);
                gemv_lut_block64_neon(ptr0, ptr1, qlut.as_ptr(), bytes_per_row, out0, out1);
            }
            if num_blocks % 2 != 0 {
                let b = num_pairs * 2;
                let block_w = transformed_weights.as_ptr().add(b * (32 * bytes_per_row));
                let block_out = out.as_mut_ptr().add(b * 32);
                gemv_lut_block32_neon(block_w, qlut.as_ptr(), bytes_per_row, block_out);
            }
        }
        return;
    }

    #[allow(unreachable_code)]
    {
        for b in 0..num_blocks {
            let block_w = unsafe { transformed_weights.as_ptr().add(b * (32 * bytes_per_row)) };
            let block_out = unsafe { out.as_mut_ptr().add(b * 32) };
            unsafe {
                gemv_lut_block32_scalar(block_w, qlut.as_ptr(), bytes_per_row, block_out);
            }
        }
    }
}

/// Multi-threaded GEMV matrix-vector multiplication using 32-row tiled Dual-Hash Activation-LUT kernel.
///
/// Uses standard library thread scoping with zero external dependencies.
pub fn gemv_lut_parallel(
    transformed_weights: &[u8],
    qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    num_threads: usize,
) {
    check_gemv_buffers(32, transformed_weights, qlut, activation_lut_len(cols), rows, cols, out);
    let num_blocks = rows / 32;
    if num_threads <= 1 || num_blocks < num_threads {
        gemv_lut_1xn(transformed_weights, qlut, rows, cols, out);
        return;
    }

    let bytes_per_row = tile_row_bytes(cols);
    let blocks_per_thread = ((num_blocks + num_threads - 1) / num_threads).next_multiple_of(2);
    let rows_per_thread = blocks_per_thread * 32;

    let weights_per_thread = rows_per_thread * bytes_per_row;
    let main_rows = num_blocks * 32;

    use rayon::prelude::*;
    let pool = crate::threading::get_thread_pool(num_threads);
    pool.install(|| {
        out[..main_rows]
            .par_chunks_mut(rows_per_thread)
            .zip(transformed_weights[..main_rows * bytes_per_row].par_chunks(weights_per_thread))
            .for_each(|(out_chunk, thread_w)| {
                let b_count = out_chunk.len() / 32;
                let qlut_ptr = qlut.as_ptr();

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            for b in 0..b_count {
                                let block_w = thread_w.as_ptr().add(b * (32 * bytes_per_row));
                                let block_out = out_chunk.as_mut_ptr().add(b * 32);
                                gemv_lut_block32_avx2(block_w, qlut_ptr, bytes_per_row, block_out);
                            }
                        }
                        return;
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    unsafe {
                        let num_pairs = b_count / 2;
                        for p in 0..num_pairs {
                            let b0 = p * 2;
                            let b1 = p * 2 + 1;
                            let ptr0 = thread_w.as_ptr().add(b0 * (32 * bytes_per_row));
                            let ptr1 = thread_w.as_ptr().add(b1 * (32 * bytes_per_row));
                            let out0 = out_chunk.as_mut_ptr().add(b0 * 32);
                            let out1 = out_chunk.as_mut_ptr().add(b1 * 32);
                            gemv_lut_block64_neon(ptr0, ptr1, qlut_ptr, bytes_per_row, out0, out1);
                        }
                        if b_count % 2 != 0 {
                            let b = num_pairs * 2;
                            let block_w = thread_w.as_ptr().add(b * (32 * bytes_per_row));
                            let block_out = out_chunk.as_mut_ptr().add(b * 32);
                            gemv_lut_block32_neon(block_w, qlut_ptr, bytes_per_row, block_out);
                        }
                    }
                    return;
                }

                #[allow(unreachable_code)]
                {
                    for b in 0..b_count {
                        let block_w = unsafe { thread_w.as_ptr().add(b * (32 * bytes_per_row)) };
                        let block_out = unsafe { out_chunk.as_mut_ptr().add(b * 32) };
                        unsafe {
                            gemv_lut_block32_scalar(block_w, qlut_ptr, bytes_per_row, block_out);
                        }
                    }
                }
            });
    });
}

/// Experimental driver: runs the single-threaded wide GEMV with a selectable AVX2 block kernel.
///
/// Variants: `0` = original fused kernel, `1` = `i8red` (int8 accumulation over three columns),
/// `2`..`4` = `i8red` with a software prefetch of 1 / 2 / 4 KiB.
/// Used by the correctness tests and the ladder benchmark while kernel variants are being compared.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub fn gemv_lut_wide_1xn_avx2_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(wide_qlut.len(), wide_activation_lut_len(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(32, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    assert!(is_x86_feature_detected!("avx2"));
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    type BlockFn = unsafe fn(*const u8, *const i8, *const i8, usize, *mut f32);
    let f: BlockFn = match variant {
        0 => crate::simd::avx2::gemv_lut_block32_wide_avx2,
        1 => crate::simd::avx2::gemv_lut_block32_wide_avx2_i8red,
        2 => crate::simd::avx2::gemv_lut_block32_wide_avx2_i8red_impl::<1024>,
        3 => crate::simd::avx2::gemv_lut_block32_wide_avx2_i8red_impl::<2048>,
        4 => crate::simd::avx2::gemv_lut_block32_wide_avx2_i8red_impl::<4096>,
        _ => panic!("unknown AVX2 variant {variant}"),
    };
    unsafe {
        for b in 0..rows / 32 {
            f(
                transformed_weights.as_ptr().add(b * (32 * bytes_per_row)),
                lo.as_ptr(),
                hi.as_ptr(),
                bytes_per_row,
                out.as_mut_ptr().add(b * 32),
            );
        }
    }
}

/// Experimental driver: runs the single-threaded wide GEMV with a selectable NEON block kernel.
/// Variants: `0` = original fused kernel, `1` = `i8red` (int8 accumulation over three columns), `2..=12` = multi-block
/// passes with selectable accumulation and prefetch distance (see the match).
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
pub fn gemv_lut_wide_1xn_neon_variant(
    variant: u32,
    transformed_weights: &[u8],
    wide_qlut: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    assert_eq!(wide_qlut.len(), wide_activation_lut_len(cols), "expected a wide (lo|hi) table");
    check_gemv_buffers(32, transformed_weights, wide_qlut, wide_qlut.len(), rows, cols, out);
    let bytes_per_row = tile_row_bytes(cols);
    let (lo, hi) = wide_qlut.split_at(wide_qlut.len() / 2);
    type BlockFn = unsafe fn(*const u8, *const i8, *const i8, usize, *mut f32);
    type MultiFn = unsafe fn(*const u8, *const i8, *const i8, usize, usize, *mut f32);
    use crate::simd::neon::gemv_lut_wide_neon_multi_driver as multi;
    let multi_f: Option<MultiFn> = match variant {
        2 => Some(multi::<1, true, 128, 0>),
        3 => Some(multi::<2, false, 128, 0>),
        4 => Some(multi::<2, true, 128, 0>),
        5 => Some(multi::<3, false, 128, 0>),
        6 => Some(multi::<4, false, 128, 0>),
        7 => Some(multi::<2, false, 512, 0>),
        8 => Some(multi::<2, false, 1024, 0>),
        9 => Some(multi::<2, false, 2048, 0>),
        10 => Some(multi::<1, false, 512, 0>),
        11 => Some(multi::<1, false, 2048, 0>),
        12 => Some(multi::<1, false, 0, 0>),
        13 => Some(multi::<2, false, 1024, 1>),
        14 => Some(multi::<2, false, 1024, 2>),
        15 => Some(multi::<2, false, 1024, 0>),
        16 => Some(multi::<2, false, 0, 0>),
        17 => Some(multi::<3, false, 0, 0>),
        _ => None,
    };
    if let Some(mf) = multi_f {
        unsafe { mf(transformed_weights.as_ptr(), lo.as_ptr(), hi.as_ptr(), rows / 32, bytes_per_row, out.as_mut_ptr()) };
        return;
    }
    let f: BlockFn = match variant {
        0 => crate::simd::neon::gemv_lut_block32_wide_neon,
        1 => crate::simd::neon::gemv_lut_block32_wide_neon_i8red,
        _ => panic!("unknown NEON variant {variant}"),
    };
    unsafe {
        for b in 0..rows / 32 {
            f(
                transformed_weights.as_ptr().add(b * (32 * bytes_per_row)),
                lo.as_ptr(),
                hi.as_ptr(),
                bytes_per_row,
                out.as_mut_ptr().add(b * 32),
            );
        }
    }
}
