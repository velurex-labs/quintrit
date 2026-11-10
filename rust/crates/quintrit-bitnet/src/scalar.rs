//! Reference portable scalar implementation for 4 trits per byte (2 bits / trit).
//!
//! Strict 1:1 Reference Port of Microsoft BitNet 2.00 bpw baselines.
//!
//! Includes:
//! - Packing and unpacking for 4 trits per byte (2.00 bpw / i2_s baseline).
//! - Lookup Table (LUT) construction for activation pairs (inspired by BitNet / T-MAC).
//! - Fast LUT-based ternary x int8 dot product and GEMV matrix multiplication.
//!
//! Encoding:
//! Each byte contains 4 trits t0, t1, t2, t3.
//! For weight w in {-1, 0, +1}, we use centered code t in {0, 1, 2}:
//!   t = w + 1  =>  0: -1,  1: 0,  2: +1,  3: 0 (neutral fallback).
//!
//! Byte layout:
//!   byte = (t0 & 3) | ((t1 & 3) << 2) | ((t2 & 3) << 4) | ((t3 & 3) << 6)
//!
//! Low nibble (bits 3:0):  (t0 & 3) | ((t1 & 3) << 2) -> 2-trit pair (w0, w1)
//! High nibble (bits 7:4): (t2 & 3) | ((t3 & 3) << 2) -> 2-trit pair (w2, w3)

/// Packs a slice of trits (values in {-1, 0, +1}) into 4-trits-per-byte format.
pub fn pack_trits_4_scalar(trits: &[i8]) -> Vec<u8> {
    let num_bytes = trits.len().div_ceil(4);
    let mut packed = Vec::with_capacity(num_bytes);

    for chunk in trits.chunks(4) {
        let mut byte = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            let code = match w {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                _ => 1u8, // fallback to neutral 0 for invalid values
            };
            byte |= code << (i * 2);
        }
        packed.push(byte);
    }
    packed
}

/// Unpacks the specified number of trits from a byte slice back to {-1, 0, +1}.
pub fn unpack_trits_4_scalar(packed: &[u8], trit_count: usize) -> Vec<i8> {
    let mut trits = Vec::with_capacity(trit_count);
    for i in 0..trit_count {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        let code = (packed[byte_idx] >> shift) & 0x03;
        let w = match code {
            0 => -1i8,
            1 => 0i8,
            2 => 1i8,
            _ => 0i8,
        };
        trits.push(w);
    }
    trits
}

/// Builds a 16-entry lookup table for a pair of activations `(x_a, x_b)`.
///
/// Each 4-bit nibble index `idx in 0..16` encodes:
/// - `t_a = idx & 3` (weight `w_a` in `{-1, 0, +1}`)
/// - `t_b = (idx >> 2) & 3` (weight `w_b` in `{-1, 0, +1}`)
///
/// Entry value is `w_a * x_a + w_b * x_b`, clamped to `[-128, 127]`.
#[inline]
pub fn build_pair_lut_entry(x_a: i8, x_b: i8) -> [i8; 16] {
    let mut table = [0i8; 16];
    let xa = x_a as i16;
    let xb = x_b as i16;

    for idx in 0..16 {
        let ta = idx & 3;
        let tb = (idx >> 2) & 3;

        let wa = match ta {
            0 => -1i16,
            2 => 1i16,
            _ => 0i16,
        };
        let wb = match tb {
            0 => -1i16,
            2 => 1i16,
            _ => 0i16,
        };

        let val = wa * xa + wb * xb;
        table[idx] = val.clamp(-128, 127) as i8;
    }

    table
}

/// Precomputes lookup tables for all consecutive activation pairs in `x`.
///
/// Returns a vector of 16-byte tables (total length = `num_pairs * 16`).
pub fn build_activation_lut_scalar(x: &[i8]) -> Vec<i8> {
    let num_pairs = x.len().div_ceil(2);
    let mut tables = Vec::with_capacity(num_pairs * 16);

    for p in 0..num_pairs {
        let xa = if 2 * p < x.len() { x[2 * p] } else { 0 };
        let xb = if 2 * p + 1 < x.len() { x[2 * p + 1] } else { 0 };
        let entry = build_pair_lut_entry(xa, xb);
        tables.extend_from_slice(&entry);
    }

    tables
}

/// Computes the dot product of ternary weights and int8 activations (Ternary x Int8)
/// using precomputed 16-entry lookup tables (inspired by BitNet / T-MAC).
///
/// `w` represents packed ternary weights (4 trits per byte).
/// `x` represents int8 / small-range integer activations.
pub fn dot_ternary_int8_scalar(w: &[u8], x: &[i8], trit_count: usize) -> i64 {
    let full_bytes = trit_count / 4;
    let mut sum: i64 = 0;

    let tables = build_activation_lut_scalar(x);

    for b in 0..full_bytes {
        let bw = w[b];
        let idx0 = (bw & 0x0F) as usize;
        let idx1 = ((bw >> 4) & 0x0F) as usize;

        let table0_offset = (2 * b) * 16;
        let table1_offset = (2 * b + 1) * 16;

        let p0 = tables[table0_offset + idx0] as i64;
        let p1 = tables[table1_offset + idx1] as i64;

        sum += p0 + p1;
    }

    // Process remainder trits (if trit_count is not a multiple of 4)
    let remainder = trit_count % 4;
    if remainder > 0 && full_bytes < w.len() {
        let bw = w[full_bytes];
        let x_offset = full_bytes * 4;
        for k in 0..remainder {
            let cw = (bw >> (k * 2)) & 0x03;
            let weight = match cw {
                0 => -1i64,
                2 => 1i64,
                _ => 0i64,
            };
            if x_offset + k < x.len() {
                sum += weight * (x[x_offset + k] as i64);
            }
        }
    }

    sum
}

/// Computes General Matrix-Vector Multiplication (GEMV: Y = W * X) using LUTs.
///
/// `w_matrix`: Row-major matrix of packed ternary weights (shape: `rows x cols.div_ceil(4)` bytes).
/// `x`: Input activation vector of length at least `cols`.
/// `rows`: Number of output rows.
/// `cols`: Number of input trits (columns).
/// `out`: Output buffer of length at least `rows`.
pub fn gemv_ternary_int8_scalar(
    w_matrix: &[u8],
    x: &[i8],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    let tables = build_activation_lut_scalar(&x[..cols]);
    let bytes_per_row = cols.div_ceil(4);
    let full_bytes = cols / 4;
    let remainder = cols % 4;

    for r in 0..rows {
        let row_w = &w_matrix[r * bytes_per_row..(r + 1) * bytes_per_row];
        let mut row_sum: i64 = 0;

        for b in 0..full_bytes {
            let bw = row_w[b];
            let idx0 = (bw & 0x0F) as usize;
            let idx1 = ((bw >> 4) & 0x0F) as usize;

            let p0 = tables[(2 * b) * 16 + idx0] as i64;
            let p1 = tables[(2 * b + 1) * 16 + idx1] as i64;

            row_sum += p0 + p1;
        }

        if remainder > 0 {
            let bw = row_w[full_bytes];
            let x_offset = full_bytes * 4;
            for k in 0..remainder {
                let cw = (bw >> (k * 2)) & 0x03;
                let weight = match cw {
                    0 => -1i64,
                    2 => 1i64,
                    _ => 0i64,
                };
                row_sum += weight * (x[x_offset + k] as i64);
            }
        }

        out[r] = row_sum;
    }
}

/// Transforms linear 4-trits-per-byte weights (`[rows x cols.div_ceil(4)]`)
/// into a 32-row block interleaved layout (tiled format, BM = 32).
///
/// In this layout:
/// - The matrix is divided into blocks of 32 rows (`rows / 32` blocks).
/// - Within each block, for each column byte `c`, the weights for all 32 rows
///   are stored contiguously in memory as a 32-byte (256-bit) chunk.
/// - This enables high-performance AVX2 SIMD loading (`_mm256_loadu_si256`)
///   with zero strided scatter/gather penalty.
///
/// Remainder rows (if `rows % 32 != 0`) are appended at the end in their original row-major linear order.
pub fn transform_weights_tiled_32(linear_weights: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let bytes_per_row = cols.div_ceil(4);
    assert!(
        linear_weights.len() >= rows * bytes_per_row,
        "Linear weights buffer is smaller than rows * bytes_per_row"
    );

    let num_blocks = rows / 32;
    let remainder_rows = rows % 32;
    let total_bytes = rows * bytes_per_row;
    let mut dst = vec![0u8; total_bytes];

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_offset = b * (32 * bytes_per_row);

        for c in 0..bytes_per_row {
            let col_offset = block_offset + c * 32;
            for r in 0..32 {
                dst[col_offset + r] = linear_weights[(r_base + r) * bytes_per_row + c];
            }
        }
    }

    if remainder_rows > 0 {
        let r_base = num_blocks * 32;
        let dst_offset = num_blocks * (32 * bytes_per_row);
        let src_offset = r_base * bytes_per_row;
        let rem_bytes = remainder_rows * bytes_per_row;
        dst[dst_offset..dst_offset + rem_bytes]
            .copy_from_slice(&linear_weights[src_offset..src_offset + rem_bytes]);
    }

    dst
}

/// Untransforms a 32-row block interleaved layout back into linear row-major format.
pub fn untransform_weights_tiled_32(tiled_weights: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let bytes_per_row = cols.div_ceil(4);
    assert!(
        tiled_weights.len() >= rows * bytes_per_row,
        "Tiled weights buffer is smaller than rows * bytes_per_row"
    );

    let num_blocks = rows / 32;
    let remainder_rows = rows % 32;
    let total_bytes = rows * bytes_per_row;
    let mut linear = vec![0u8; total_bytes];

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_offset = b * (32 * bytes_per_row);

        for c in 0..bytes_per_row {
            let col_offset = block_offset + c * 32;
            for r in 0..32 {
                linear[(r_base + r) * bytes_per_row + c] = tiled_weights[col_offset + r];
            }
        }
    }

    if remainder_rows > 0 {
        let r_base = num_blocks * 32;
        let src_offset = num_blocks * (32 * bytes_per_row);
        let dst_offset = r_base * bytes_per_row;
        let rem_bytes = remainder_rows * bytes_per_row;
        linear[dst_offset..dst_offset + rem_bytes]
            .copy_from_slice(&tiled_weights[src_offset..src_offset + rem_bytes]);
    }

    linear
}

/// Directly packs a matrix of ternary values in `{-1, 0, +1}` (row-major slice of length `rows * cols`)
/// into the 32-row block interleaved layout in a single pass.
pub fn pack_trits_tiled_32(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(trits.len(), rows * cols);
    let bytes_per_row = cols.div_ceil(4);
    let num_blocks = rows / 32;
    let remainder_rows = rows % 32;
    let total_bytes = rows * bytes_per_row;
    let mut dst = vec![0u8; total_bytes];

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_offset = b * (32 * bytes_per_row);

        for c in 0..bytes_per_row {
            let col_offset = block_offset + c * 32;
            let col_start_trit = c * 4;
            let trits_in_col = (cols - col_start_trit).min(4);

            for r in 0..32 {
                let row_idx = r_base + r;
                let row_trits = &trits[row_idx * cols + col_start_trit..row_idx * cols + col_start_trit + trits_in_col];
                let mut byte = 0u8;
                for (i, &w) in row_trits.iter().enumerate() {
                    let code = match w {
                        -1 => 0u8,
                        0 => 1u8,
                        1 => 2u8,
                        _ => 1u8,
                    };
                    byte |= code << (i * 2);
                }
                dst[col_offset + r] = byte;
            }
        }
    }

    if remainder_rows > 0 {
        let r_base = num_blocks * 32;
        let dst_offset = num_blocks * (32 * bytes_per_row);
        for r in 0..remainder_rows {
            let row_idx = r_base + r;
            let row_trits = &trits[row_idx * cols..(row_idx + 1) * cols];
            let packed_row = pack_trits_4_scalar(row_trits);
            dst[dst_offset + r * bytes_per_row..dst_offset + (r + 1) * bytes_per_row]
                .copy_from_slice(&packed_row);
        }
    }

    dst
}

/// Computes GEMV against 32-row tiled weights using portable scalar code.
pub fn gemv_lut_tiled32_scalar(
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

    for b in 0..num_blocks {
        let r_base = b * 32;
        let block_offset = b * (32 * bytes_per_row);

        for r in 0..32 {
            let mut sum: i64 = 0;
            for c in 0..full_bytes {
                let bw = tiled_weights[block_offset + c * 32 + r];
                let idx0 = (bw & 0x0F) as usize;
                let idx1 = ((bw >> 4) & 0x0F) as usize;
                let p0 = qlut[(2 * c) * 16 + idx0] as i64;
                let p1 = qlut[(2 * c + 1) * 16 + idx1] as i64;
                sum += p0 + p1;
            }
            if remainder_cols > 0 {
                let bw = tiled_weights[block_offset + full_bytes * 32 + r];
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
            out[r_base + r] = sum;
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
