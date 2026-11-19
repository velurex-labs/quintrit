//! Microsoft BitNet planar format decoder (`GGML_TYPE_I2_S` = 36, 2.00 bpw).
//!
//! Encodes 4 trits per byte in 32-byte blocks (128 trits per block) with planar bit-slicing.

use super::TernaryDecoder;
use crate::gguf::GGML_TYPE_I2_S;
use rayon::prelude::*;
use quintrit_core::lut::transform_weights_lut;
use quintrit_core::pack::pack_matrix;
use quintrit_core::unpack::{decode_2bit_planes, PlaneOrder};

/// Canonical GGUF `GGML_TYPE_I2_S` block dimensions (128 trits in 32 bytes, identical on every CPU architecture).
pub const QK_GGUF_I2_S: usize = 128;
/// Bytes per `I2_S` block.
pub const BLOCK_BYTES_GGUF_I2_S: usize = 32;
/// Trits per bit plane inside one `I2_S` block.
pub const GROUP_SIZE_GGUF_I2_S: usize = 32;

/// Packs ternary weights in `{-1, 0, +1}` into the planar GGUF `I2_S` layout (codes 0, 1, 2 for -1, 0, +1).
pub fn pack_gguf_i2_s(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(trits.len(), rows * cols, "input length must match rows * cols");
    assert_eq!(cols % QK_GGUF_I2_S, 0, "cols must be a multiple of {QK_GGUF_I2_S}");
    let bytes_per_row = cols / 4;
    let mut dst = vec![0u8; rows * bytes_per_row];
    for r in 0..rows {
        for b in 0..cols / QK_GGUF_I2_S {
            let src = &trits[r * cols + b * QK_GGUF_I2_S..][..QK_GGUF_I2_S];
            let out = &mut dst[r * bytes_per_row + b * BLOCK_BYTES_GGUF_I2_S..][..BLOCK_BYTES_GGUF_I2_S];
            for (j, &w) in src.iter().enumerate() {
                let code = match w {
                    -1 => 0u8,
                    1 => 2u8,
                    _ => 1u8,
                };
                out[j % GROUP_SIZE_GGUF_I2_S] |= code << (6 - 2 * (j / GROUP_SIZE_GGUF_I2_S));
            }
        }
    }
    dst
}

/// Unpacks planar GGUF `I2_S` weights into trits in `{-1, 0, +1}` (code 3 decodes to 0).
pub fn unpack_gguf_i2_s(packed: &[u8], rows: usize, cols: usize) -> Vec<i8> {
    let mut trits = vec![0i8; rows * cols];
    unpack_gguf_i2_s_into(packed, rows, cols, &mut trits);
    trits
}

/// Like [`unpack_gguf_i2_s`], but writes into `trits` (`rows * cols` entries).
pub fn unpack_gguf_i2_s_into(packed: &[u8], rows: usize, cols: usize, trits: &mut [i8]) {
    assert_eq!(cols % QK_GGUF_I2_S, 0, "cols must be a multiple of {QK_GGUF_I2_S}");
    assert_eq!(packed.len(), rows * cols / 4, "packed length must be rows * cols / 4");
    assert_eq!(trits.len(), rows * cols, "output length must be rows * cols");
    let bytes_per_row = cols / 4;
    if rows == 0 || cols == 0 {
        return;
    }
    // A row is a whole number of 32-byte blocks; plane 0 sits in the high bits (SIMD on AVX2 and NEON, see `unpack`).
    let unpack_row = |src_row: &[u8], out_row: &mut [i8]| decode_2bit_planes(src_row, out_row, PlaneOrder::HighFirst);
    if rows * cols >= quintrit_core::pack::PARALLEL_MIN_WEIGHTS {
        trits.par_chunks_exact_mut(cols).zip(packed.par_chunks_exact(bytes_per_row)).for_each(|(out, src)| unpack_row(src, out));
    } else {
        for (out, src) in trits.chunks_exact_mut(cols).zip(packed.chunks_exact(bytes_per_row)) {
            unpack_row(src, out);
        }
    }
}


/// Decoder for Microsoft BitNet planar 2.00 bpw format (`GGML_TYPE_I2_S`).
pub struct I2sDecoder;

impl TernaryDecoder for I2sDecoder {
    #[inline]
    fn tensor_type(&self) -> u32 {
        GGML_TYPE_I2_S
    }

    #[inline]
    fn format_name(&self) -> &'static str {
        "GGML_TYPE_I2_S"
    }

    #[inline]
    fn nominal_bpw(&self) -> f32 {
        2.0
    }

    fn row_bytes(&self, cols: usize) -> Option<usize> {
        cols.is_multiple_of(QK_GGUF_I2_S).then_some(cols / 4)
    }

    fn unpack_trits_into(&self, data: &[u8], rows: usize, cols: usize, out: &mut [i8]) -> Result<(), String> {
        if cols % QK_GGUF_I2_S != 0 {
            return Err(format!("I2_S cols must be a multiple of {QK_GGUF_I2_S} (got cols={cols})"));
        }
        let expected_bytes = (rows * cols) / 4;
        if data.len() < expected_bytes {
            return Err(format!("I2_S data buffer too short: expected {} bytes, got {}", expected_bytes, data.len()));
        }
        if out.len() != rows * cols {
            return Err(format!("output buffer holds {} trits, expected {}", out.len(), rows * cols));
        }
        unpack_gguf_i2_s_into(&data[..expected_bytes], rows, cols, out);
        Ok(())
    }

    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String> {
        let expected_bytes = (rows * cols) / 4;
        if data.len() < expected_bytes {
            return Err(format!(
                "I2_S data buffer too short: expected {} bytes, got {}",
                expected_bytes, data.len()
            ));
        }
        Ok(unpack_gguf_i2_s(&data[..expected_bytes], rows, cols))
    }

    fn transcode_to_velurex_tiled(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        if cols % 128 != 0 {
            return Err(format!(
                "I2_S cols must be a multiple of 128 (got cols={})",
                cols
            ));
        }
        let expected_bytes = (rows * cols) / 4;
        if data.len() < expected_bytes {
            return Err(format!(
                "I2_S data buffer too short: expected {} bytes, got {}",
                expected_bytes, data.len()
            ));
        }

        if rows % 32 == 0 {
            return crate::transcoder::transcode_to_tiled(self, data, rows, cols, 32);
        }

        // A row count that is no multiple of 32: the last block is padded with neutral rows.
        let packed_planar = &data[..expected_bytes];
        let bytes_per_row_5 = quintrit_core::pack::tile_row_bytes(cols);
        let num_blocks = rows / 32;
        let remainder_rows = rows % 32;

        let total_tiled_bytes = if remainder_rows == 0 {
            num_blocks * (32 * bytes_per_row_5)
        } else {
            (num_blocks + 1) * (32 * bytes_per_row_5)
        };

        let mut out_tiled = vec![0u8; total_tiled_bytes];
        let planar_bytes_per_row = cols / 4;
        let planar_bytes_per_block = 32 * planar_bytes_per_row;
        let tiled_bytes_per_block = 32 * bytes_per_row_5;

        // Process complete 32-row blocks in parallel
        if num_blocks > 0 {
            out_tiled[..num_blocks * tiled_bytes_per_block]
                .par_chunks_exact_mut(tiled_bytes_per_block)
                .enumerate()
                .for_each(|(b, block_out)| {
                    let planar_start = b * planar_bytes_per_block;
                    let block_planar = &packed_planar[planar_start..planar_start + planar_bytes_per_block];

                    // 1. Unpack BitNet planar trits (32 rows x cols) with canonical 128-trit GGUF block size
                    let block_trits = unpack_gguf_i2_s(block_planar, 32, cols);

                    // 2. Pack into 1.60 bpw velurex encoding (32 rows x cols)
                    let block_packed = pack_matrix(&block_trits, 32, cols);

                    // 3. Transform into 32-row tiled SIMD layout
                    let tiled = transform_weights_lut(&block_packed, 32, cols);
                    block_out.copy_from_slice(&tiled);
                });
        }

        // Process remainder rows if rows is not a multiple of 32
        if remainder_rows > 0 {
            let b = num_blocks;
            let planar_start = b * planar_bytes_per_block;
            let rem_planar = &packed_planar[planar_start..];
            let rem_trits = unpack_gguf_i2_s(rem_planar, remainder_rows, cols);

            let mut padded_trits = vec![0i8; 32 * cols];
            padded_trits[..rem_trits.len()].copy_from_slice(&rem_trits);

            let rem_packed = pack_matrix(&padded_trits, 32, cols);
            let rem_tiled = transform_weights_lut(&rem_packed, 32, cols);

            let tiled_start = b * tiled_bytes_per_block;
            out_tiled[tiled_start..tiled_start + tiled_bytes_per_block]
                .copy_from_slice(&rem_tiled);
        }

        Ok(out_tiled)
    }
}
