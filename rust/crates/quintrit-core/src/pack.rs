//! 5-trit packing, unpacking, and activation utilities.
//!
//! Packs five ternary weights in `{-1, 0, +1}` into a single byte `[0..=250]`
//! using the deterministic radix vector `(1, 3, 9, 28, 84)`:
//!
//! ```text
//! d_k = t_k + 1 in {0, 1, 2}
//! byte = d0 + 3 * d1 + 9 * d2 + 28 * d3 + 84 * d4
//! ```
//!
//! The maximum encodable value is `2 + 6 + 18 + 56 + 168 = 250`.
//! Byte values `251..=255` never appear in valid data.
//!
//! Matrices are laid out in 5-planar blocks: each block holds 160 trits and occupies
//! 32 packed bytes. Byte `j` of a block packs the trits at block offsets
//! `j`, `32 + j`, `64 + j`, `96 + j`, and `128 + j`.

/// Number of trits per 5-planar block (5 * 32 = 160).
pub const QK_5: usize = 160;

/// Number of packed bytes per block (160 / 5 = 32 bytes = one AVX2 register).
pub const BLOCK_BYTES_5: usize = 32;

/// Packs 5 ternary weights in `{-1, 0, +1}` into a single byte.
///
/// Returns an encoded byte in the range `0..=250`.
#[inline]
pub const fn pack_5trits(t: [i8; 5]) -> u8 {
    let d0 = (t[0] + 1) as u8;
    let d1 = (t[1] + 1) as u8;
    let d2 = (t[2] + 1) as u8;
    let d3 = (t[3] + 1) as u8;
    let d4 = (t[4] + 1) as u8;

    d0 + 3 * d1 + 9 * d2 + 28 * d3 + 84 * d4
}

/// Compiler-precomputed scalar decoder table (byte -> 5 trits).
///
/// Bytes outside the valid range (251..=255) decode to all zeros.
pub static BYTE_TO_TRITS: [[i8; 5]; 256] = init_byte_to_trits();

const fn init_byte_to_trits() -> [[i8; 5]; 256] {
    let mut table = [[0i8; 5]; 256];
    let mut s = 0usize;
    while s < 243 {
        let t = [
            (s % 3) as i8 - 1,
            ((s / 3) % 3) as i8 - 1,
            ((s / 9) % 3) as i8 - 1,
            ((s / 27) % 3) as i8 - 1,
            ((s / 81) % 3) as i8 - 1,
        ];
        table[pack_5trits(t) as usize] = t;
        s += 1;
    }
    table
}

/// Packs a matrix of ternary weights `{-1, 0, +1}` into the 5-planar layout.
///
/// Columns are zero-padded to a multiple of 160 trits with neutral weights.
pub fn pack_matrix(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(trits.len(), rows * cols, "trits length mismatch");
    let blocks_per_row = cols.div_ceil(QK_5);
    let row_bytes = blocks_per_row * BLOCK_BYTES_5;
    let mut packed = vec![0u8; rows * row_bytes];

    for r in 0..rows {
        let row_trits = &trits[r * cols..(r + 1) * cols];
        let row_dst = &mut packed[r * row_bytes..(r + 1) * row_bytes];

        for b in 0..blocks_per_row {
            let b_start = b * QK_5;
            let block_dst = &mut row_dst[b * BLOCK_BYTES_5..(b + 1) * BLOCK_BYTES_5];

            for (j, dst) in block_dst.iter_mut().enumerate() {
                let get_trit = |plane: usize| -> i8 {
                    let idx = b_start + plane * BLOCK_BYTES_5 + j;
                    if idx < cols {
                        row_trits[idx]
                    } else {
                        0 // Neutral zero weight
                    }
                };

                *dst = pack_5trits([
                    get_trit(0),
                    get_trit(1),
                    get_trit(2),
                    get_trit(3),
                    get_trit(4),
                ]);
            }
        }
    }

    packed
}

/// Unpacks a 5-planar matrix back into ternary weights `{-1, 0, +1}`.
pub fn unpack_matrix(packed: &[u8], rows: usize, cols: usize) -> Vec<i8> {
    let blocks_per_row = cols.div_ceil(QK_5);
    let row_bytes = blocks_per_row * BLOCK_BYTES_5;
    assert!(packed.len() >= rows * row_bytes, "packed buffer too small");

    let mut trits = vec![0i8; rows * cols];

    for r in 0..rows {
        let row_packed = &packed[r * row_bytes..(r + 1) * row_bytes];
        let row_trits = &mut trits[r * cols..(r + 1) * cols];

        for b in 0..blocks_per_row {
            let block = &row_packed[b * BLOCK_BYTES_5..(b + 1) * BLOCK_BYTES_5];
            let b_start = b * QK_5;

            for (j, &byte) in block.iter().enumerate() {
                let decoded = BYTE_TO_TRITS[byte as usize];
                for (plane, &t) in decoded.iter().enumerate() {
                    let idx = b_start + plane * BLOCK_BYTES_5 + j;
                    if idx < cols {
                        row_trits[idx] = t;
                    }
                }
            }
        }
    }

    trits
}

/// Splits an activation vector into five planar buffers matching the packed weight layout.
///
/// Activations are zero-padded up to a multiple of 160. Returns the planes
/// `[act0, act1, act2, act3, act4]`, each `blocks_per_row * 32` bytes long.
pub fn split_activation_planes(activations: &[i8], cols: usize) -> [Vec<i8>; 5] {
    let blocks_per_row = cols.div_ceil(QK_5);
    let plane_len = blocks_per_row * BLOCK_BYTES_5;

    let mut planes = [
        vec![0i8; plane_len],
        vec![0i8; plane_len],
        vec![0i8; plane_len],
        vec![0i8; plane_len],
        vec![0i8; plane_len],
    ];

    for b in 0..blocks_per_row {
        let b_start = b * QK_5;
        for p in 0..5 {
            let p_start = b_start + p * BLOCK_BYTES_5;
            for j in 0..BLOCK_BYTES_5 {
                let idx = p_start + j;
                if idx < cols {
                    planes[p][b * BLOCK_BYTES_5 + j] = activations[idx];
                }
            }
        }
    }

    planes
}

// ============================================================================
// Tile row length and the packed tail
// ============================================================================
//
// IMPORTANT FOR ANY FUTURE PRE-TILED OR MEMORY-MAPPED WEIGHT FORMAT: the layout of the in-memory LUT tiles defined below is
// fixed. The partial last 160-trit block of a row is stored packed (five trits per byte, rounded up to an even number of
// bytes), the rows of a tile are `tile_row_bytes(cols)` long, and the activation tables map the tail columns accordingly.
// A file that stores finished tiles (so that the kernels could read a memory-mapped file without a transcoding step) must
// store exactly this layout and record a layout version in its header; changing the layout later breaks such files.
// The packed (file) layout of `pack_matrix` is a different thing and is not affected.

/// Number of byte-columns of the full 160-trit blocks of a row of `cols` trits.
#[inline]
pub const fn tile_full_columns(cols: usize) -> usize {
    (cols / QK_5) * BLOCK_BYTES_5
}

/// Number of byte-columns a row occupies in a tile and in the activation tables: the full 160-trit blocks plus the partial
/// last block packed five trits to a byte and rounded up to an even count.
///
/// In the planar layout of [`pack_matrix`] a partial last block of `rem` trits occupies a whole 32-byte block (only
/// `ceil(rem / 32)` of its five planes carry data); for `cols = 3072` that is 32 bytes for 32 trits. The tile transform
/// re-encodes that tail into `ceil(rem / 5)` bytes (8 instead of 32 for 3072 columns, rows of 616 instead of 640 bytes), and
/// the table builders map the tail columns to consecutive activations, so every product and every sum stays the same and
/// the output is bit-identical; only fewer bytes are stored and read.
#[inline]
pub const fn tile_row_bytes(cols: usize) -> usize {
    let tail = (cols % QK_5).div_ceil(5);
    tile_full_columns(cols) + tail + (tail & 1)
}

/// Index of the activation that byte-column `c` of a tile row multiplies at position `p` (0..5) of its five trits;
/// the caller treats an index `>= cols` as a zero activation.
#[inline]
pub const fn column_activation_index(cols: usize, c: usize, p: usize) -> usize {
    let full = tile_full_columns(cols);
    if c < full {
        (c / BLOCK_BYTES_5) * QK_5 + p * BLOCK_BYTES_5 + c % BLOCK_BYTES_5
    } else {
        (cols / QK_5) * QK_5 + 5 * (c - full) + p
    }
}

/// Byte-column `c` of a tile row, taken from a row `src_row` in the planar layout of [`pack_matrix`].
#[inline]
pub fn tile_byte(src_row: &[u8], cols: usize, c: usize) -> u8 {
    let full = tile_full_columns(cols);
    if c < full {
        return src_row[c];
    }
    let rem = cols % QK_5;
    let mut t = [0i8; 5];
    for (k, v) in t.iter_mut().enumerate() {
        let q = 5 * (c - full) + k;
        if q < rem {
            *v = BYTE_TO_TRITS[src_row[full + q % BLOCK_BYTES_5] as usize][q / BLOCK_BYTES_5];
        }
    }
    pack_5trits(t)
}

/// Inverse of [`tile_byte`] for a whole row: rebuilds the planar row from a tile row.
pub fn untile_row(tile_row: &[u8], cols: usize, dst_row: &mut [u8]) {
    let full = tile_full_columns(cols);
    dst_row[..full].copy_from_slice(&tile_row[..full]);
    let rem = cols % QK_5;
    if rem == 0 {
        return;
    }
    let mut trits = [0i8; QK_5];
    for q in 0..rem {
        trits[q] = BYTE_TO_TRITS[tile_row[full + q / 5] as usize][q % 5];
    }
    for j in 0..BLOCK_BYTES_5 {
        dst_row[full + j] =
            pack_5trits([trits[j], trits[32 + j], trits[64 + j], trits[96 + j], trits[128 + j]]);
    }
}
