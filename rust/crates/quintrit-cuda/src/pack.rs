//! Host-side layout helpers for the GPU kernels.
//!
//! The matrix layout the kernels read ("planar") is the layout of [`quintrit_core::pack::pack_matrix`]: a row is a sequence of
//! blocks of 160 trits = 32 bytes, and byte `j` of block `b` packs the trits of columns `b * 160 + p * 32 + j` for the planes
//! `p = 0..5` as `d0 + 3 d1 + 9 d2 + 28 d3 + 84 d4` with the digits `d = trit + 1`. The last block of a row is padded with zero
//! trits. The GPU weight representation is produced from this layout while a model is loaded; it is never written to a file.
//!
//! The activation vector is permuted into 8-byte entries, one per byte-column, so that the five plane activations of a weight byte sit next to
//! each other: bytes 0 to 3 of entry `b * 32 + j` are the activations of the planes 0 to 3 (one 32-bit operand of `dp4a`) and byte 4
//! is the activation of plane 4. The kernels that quantize activations on the device write this layout themselves;
//! [`permute_activations`] is the host reference.

use quintrit_core::pack::{pack_matrix_into, BLOCK_BYTES_5, QK_5};

/// Trits per block of a row.
pub const BLOCK_TRITS: usize = QK_5;
/// Bytes per block of a row.
pub const BLOCK_BYTES: usize = BLOCK_BYTES_5;
/// Extra activation entries after the last block that the kernels may read (they load four entries per step).
pub const ACT_PAD_ENTRIES: usize = 16;

/// Blocks per row for `cols` columns.
pub const fn blocks_per_row(cols: usize) -> usize {
    cols.div_ceil(BLOCK_TRITS)
}

/// Bytes of one packed row for `cols` columns.
pub const fn row_bytes(cols: usize) -> usize {
    blocks_per_row(cols) * BLOCK_BYTES
}

/// Entries (8 bytes each) of a permuted activation vector for `cols` columns, padding included.
pub const fn act_entries(cols: usize) -> usize {
    blocks_per_row(cols) * BLOCK_BYTES + ACT_PAD_ENTRIES
}

/// Packs `rows x cols` trits (`-1, 0, +1`) into the planar layout.
///
/// # Panics
/// If `trits.len() != rows * cols`.
pub fn pack_planar(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(trits.len(), rows * cols, "trits must hold rows * cols values");
    let mut out = vec![0u8; rows * row_bytes(cols)];
    pack_matrix_into(trits, rows, cols, &mut out);
    out
}

/// Concatenates packed matrices that have the same number of columns, one below the other (for example q, k and v into the
/// stacked projection of one launch).
pub fn stack_rows(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// Interleaves two packed matrices of the same shape row by row: row `2 i` is row `i` of `gate`, row `2 i + 1` is row `i` of `up`.
///
/// # Panics
/// If the matrices differ in length or the length is not a multiple of `row_bytes`.
pub fn interleave_rows(gate: &[u8], up: &[u8], row_bytes: usize) -> Vec<u8> {
    assert_eq!(gate.len(), up.len(), "gate and up must have the same shape");
    assert!(row_bytes > 0 && gate.len() % row_bytes == 0, "length must be a multiple of the row size");
    let mut out = Vec::with_capacity(gate.len() * 2);
    for (g, u) in gate.chunks_exact(row_bytes).zip(up.chunks_exact(row_bytes)) {
        out.extend_from_slice(g);
        out.extend_from_slice(u);
    }
    out
}

/// The permuted activation vector: `act_entries(cols)` entries of 8 bytes. Entry `b * 32 + j` holds the activations of the columns
/// `b * 160 + p * 32 + j` for `p = 0..5` in its first five bytes; padding columns and the remaining bytes are zero.
pub fn permute_activations(act: &[i8]) -> Vec<i8> {
    let cols = act.len();
    let mut out = vec![0i8; act_entries(cols) * 8];
    for (c, &a) in act.iter().enumerate() {
        let (b, rem) = (c / BLOCK_TRITS, c % BLOCK_TRITS);
        let (p, j) = (rem / BLOCK_BYTES, rem % BLOCK_BYTES);
        out[(b * BLOCK_BYTES + j) * 8 + p] = a;
    }
    out
}

/// Sum of the activations (the kernels subtract it: the digits are `trit + 1`).
pub fn activation_sum(act: &[i8]) -> i32 {
    act.iter().map(|&a| a as i32).sum()
}

/// Exact result of the projection: `sum_k trit[n, k] * act[k]` per row, as `i64` ground truth.
pub fn reference_gemv(trits: &[i8], rows: usize, cols: usize, act: &[i8]) -> Vec<i64> {
    assert_eq!(trits.len(), rows * cols);
    assert_eq!(act.len(), cols);
    (0..rows).map(|r| trits[r * cols..(r + 1) * cols].iter().zip(act).map(|(&t, &a)| t as i64 * a as i64).sum()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout stated in the module documentation, written out independently of `pack_matrix`.
    fn pack_by_definition(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let bpr = blocks_per_row(cols);
        let mut out = vec![0u8; rows * bpr * BLOCK_BYTES];
        let radix = [1u32, 3, 9, 28, 84];
        for r in 0..rows {
            for b in 0..bpr {
                for j in 0..BLOCK_BYTES {
                    let mut byte = 0u32;
                    for (p, &w) in radix.iter().enumerate() {
                        let c = b * BLOCK_TRITS + p * BLOCK_BYTES + j;
                        let t = if c < cols { trits[r * cols + c] } else { 0 };
                        byte += w * (t + 1) as u32;
                    }
                    assert!(byte <= 250);
                    out[(r * bpr + b) * BLOCK_BYTES + j] = byte as u8;
                }
            }
        }
        out
    }

    fn lcg_trits(n: usize, seed: u64) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 3) as i8 - 1
            })
            .collect()
    }

    #[test]
    fn planar_layout_matches_its_definition() {
        for &(rows, cols) in &[(1, 1), (3, 160), (4, 161), (5, 3072), (2, 2560), (7, 13312), (2, 319)] {
            let trits = lcg_trits(rows * cols, (rows * 31 + cols) as u64);
            assert_eq!(pack_planar(&trits, rows, cols), pack_by_definition(&trits, rows, cols), "{rows}x{cols}");
        }
    }

    #[test]
    fn row_sizes() {
        assert_eq!(row_bytes(3072), 640);
        assert_eq!(row_bytes(23040), 4608);
        assert_eq!(row_bytes(160), 32);
        assert_eq!(row_bytes(161), 64);
        assert_eq!(act_entries(3072), 20 * 32 + 16);
    }

    #[test]
    fn interleave_and_stack() {
        let a: Vec<u8> = (0..12).collect();
        let b: Vec<u8> = (100..112).collect();
        assert_eq!(interleave_rows(&a, &b, 3), vec![0, 1, 2, 100, 101, 102, 3, 4, 5, 103, 104, 105, 6, 7, 8, 106, 107, 108, 9, 10, 11, 109, 110, 111]);
        assert_eq!(stack_rows(&[&a[..4], &b[..2]]), vec![0, 1, 2, 3, 100, 101]);
    }

    #[test]
    fn permutation_places_plane_activations_side_by_side() {
        let cols = 200;
        let act: Vec<i8> = (0..cols).map(|c| (c as i32 % 200 - 100) as i8).collect();
        let perm = permute_activations(&act);
        assert_eq!(perm.len(), act_entries(cols) * 8);
        for c in 0..cols {
            let (b, rem) = (c / 160, c % 160);
            assert_eq!(perm[((b * 32) + rem % 32) * 8 + rem / 32], act[c]);
        }
        // Bytes 5..8 of every entry and the padding entries are zero.
        for e in 0..act_entries(cols) {
            assert!(perm[e * 8 + 5..e * 8 + 8].iter().all(|&v| v == 0));
        }
        assert!(perm[2 * 32 * 8..].iter().all(|&v| v == 0), "block 2 does not exist for 200 columns");
    }
}
