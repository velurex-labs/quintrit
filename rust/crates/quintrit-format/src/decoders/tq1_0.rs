//! llama.cpp TQ1_0 format decoder (`GGML_TYPE_TQ1_0` = 34, 1.6875 bpw).
//!
//! # Storage Architecture (`block_tq1_0`)
//! - Block size: 256 weights in 54 bytes.
//! - `qs[48]`: 48 bytes encoding 240 weights using standard Base-3 ($B = t_0 + 3t_1 + 9t_2 + 27t_3 + 81t_4$).
//! - `qh[4]`: 4 bytes encoding 16 leftover weights packed at 2 bits each.
//! - `d`: 2-byte FP16 per-block scale factor.

use super::TernaryDecoder;
use crate::gguf::GGML_TYPE_TQ1_0;
use quintrit_core::lut::transform_weights_lut;
use quintrit_core::pack::pack_matrix;

/// Block size in number of weights.
pub const QK_TQ1_0: usize = 256;
/// Block size in bytes.
pub const BLOCK_BYTES_TQ1_0: usize = 54;

/// Decoder for llama.cpp 1.6875 bpw format (`GGML_TYPE_TQ1_0`).
pub struct Tq1_0Decoder;

impl TernaryDecoder for Tq1_0Decoder {
    #[inline]
    fn tensor_type(&self) -> u32 {
        GGML_TYPE_TQ1_0
    }

    #[inline]
    fn format_name(&self) -> &'static str {
        "GGML_TYPE_TQ1_0"
    }

    #[inline]
    fn nominal_bpw(&self) -> f32 {
        1.6875
    }

    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String> {
        let total_weights = rows * cols;
        if total_weights % QK_TQ1_0 != 0 {
            return Err(format!(
                "Total weights ({} x {} = {}) must be a multiple of {}",
                rows, cols, total_weights, QK_TQ1_0
            ));
        }

        let num_blocks = total_weights / QK_TQ1_0;
        let expected_bytes = num_blocks * BLOCK_BYTES_TQ1_0;
        if data.len() < expected_bytes {
            return Err(format!(
                "TQ1_0 data buffer too short: expected {} bytes, got {}",
                expected_bytes, data.len()
            ));
        }

        let mut out_trits = vec![0i8; total_weights];

        // ggml stores five trits per byte as a base-3 fraction with the first trit most significant;
        // trit `n` is recovered by multiplying with 3^n modulo 256 and scaling by 3 / 256
        // (`dequantize_row_tq1_0`). The element order is strided: within a group of bytes, all
        // bytes' first trits come first, then all second trits, and so on.
        const POW3: [u32; 5] = [1, 3, 9, 27, 81];
        let digit = |byte: u8, n: usize| -> i8 {
            let q = (byte as u32 * POW3[n]) as u8 as u32;
            (((q * 3) >> 8) as i16 - 1) as i8
        };

        for b in 0..num_blocks {
            let block_offset = b * BLOCK_BYTES_TQ1_0;
            let block = &data[block_offset..block_offset + BLOCK_BYTES_TQ1_0];
            let qs = &block[0..48];
            let qh = &block[48..52];
            let mut out = b * QK_TQ1_0;

            // 32 bytes -> 160 trits
            for n in 0..5 {
                for m in 0..32 {
                    out_trits[out] = digit(qs[m], n);
                    out += 1;
                }
            }
            // 16 bytes -> 80 trits
            for n in 0..5 {
                for m in 0..16 {
                    out_trits[out] = digit(qs[32 + m], n);
                    out += 1;
                }
            }
            // 4 bytes -> 16 trits (four trits per byte)
            for n in 0..4 {
                for j in 0..4 {
                    out_trits[out] = digit(qh[j], n);
                    out += 1;
                }
            }
        }

        Ok(out_trits)
    }

    fn transcode_to_velurex_tiled(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        let trits = self.unpack_trits(data, rows, cols)?;
        let packed_5 = pack_matrix(&trits, rows, cols);
        Ok(transform_weights_lut(&packed_5, rows, cols))
    }
}
