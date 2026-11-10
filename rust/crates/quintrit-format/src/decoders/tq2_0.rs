//! llama.cpp TQ2_0 format decoder (`GGML_TYPE_TQ2_0` = 35, 2.0625 bpw).
//!
//! # Storage Architecture (`block_tq2_0`)
//! - Block size: 256 weights in 66 bytes.
//! - `qs[64]`: 64 bytes encoding 256 weights (4 weights per byte at 2 bits each).
//! - `d`: 2-byte FP16 per-block scale factor.

use super::TernaryDecoder;
use crate::gguf::GGML_TYPE_TQ2_0;
use quintrit_core::lut::transform_weights_lut;
use quintrit_core::pack::pack_matrix;

/// Block size in number of weights.
pub const QK_TQ2_0: usize = 256;
/// Block size in bytes.
pub const BLOCK_BYTES_TQ2_0: usize = 66;

/// Decoder for llama.cpp 2.0625 bpw format (`GGML_TYPE_TQ2_0`).
pub struct Tq2_0Decoder;

impl TernaryDecoder for Tq2_0Decoder {
    #[inline]
    fn tensor_type(&self) -> u32 {
        GGML_TYPE_TQ2_0
    }

    #[inline]
    fn format_name(&self) -> &'static str {
        "GGML_TYPE_TQ2_0"
    }

    #[inline]
    fn nominal_bpw(&self) -> f32 {
        2.0625
    }

    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String> {
        let total_weights = rows * cols;
        if total_weights % QK_TQ2_0 != 0 {
            return Err(format!(
                "Total weights ({} x {} = {}) must be a multiple of {}",
                rows, cols, total_weights, QK_TQ2_0
            ));
        }

        let num_blocks = total_weights / QK_TQ2_0;
        let expected_bytes = num_blocks * BLOCK_BYTES_TQ2_0;
        if data.len() < expected_bytes {
            return Err(format!(
                "TQ2_0 data buffer too short: expected {} bytes, got {}",
                expected_bytes, data.len()
            ));
        }

        let mut out_trits = vec![0i8; total_weights];

        // Element order of ggml `dequantize_row_tq2_0`: for each group of 32 bytes, the four 2-bit
        // planes are emitted one after another, i.e. element `l * 32 + m` of a group comes from bits
        // `2l..2l+1` of byte `m`.
        for b in 0..num_blocks {
            let block_offset = b * BLOCK_BYTES_TQ2_0;
            let qs = &data[block_offset..block_offset + 64];
            let mut out = b * QK_TQ2_0;
            for j in (0..64).step_by(32) {
                for l in 0..4 {
                    for m in 0..32 {
                        let code = (qs[j + m] >> (l * 2)) & 0x03;
                        out_trits[out] = match code {
                            0 => -1,
                            2 => 1,
                            _ => 0, // 1 is zero; the unused code 3 decodes to zero
                        };
                        out += 1;
                    }
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
