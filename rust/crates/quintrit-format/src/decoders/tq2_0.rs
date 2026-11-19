//! llama.cpp TQ2_0 format decoder (`GGML_TYPE_TQ2_0` = 35, 2.0625 bpw).
//!
//! # Storage Architecture (`block_tq2_0`)
//! - Block size: 256 weights in 66 bytes.
//! - `qs[64]`: 64 bytes encoding 256 weights (4 weights per byte at 2 bits each).
//! - `d`: 2-byte FP16 per-block scale factor.

use super::TernaryDecoder;
use crate::gguf::GGML_TYPE_TQ2_0;
use quintrit_core::pack::PARALLEL_MIN_WEIGHTS;
use quintrit_core::unpack::{decode_2bit_planes, PlaneOrder};
use rayon::prelude::*;

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

    fn row_bytes(&self, cols: usize) -> Option<usize> {
        cols.is_multiple_of(QK_TQ2_0).then_some(cols / QK_TQ2_0 * BLOCK_BYTES_TQ2_0)
    }

    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String> {
        let mut trits = vec![0i8; rows * cols];
        self.unpack_trits_into(data, rows, cols, &mut trits)?;
        Ok(trits)
    }

    fn unpack_trits_into(&self, data: &[u8], rows: usize, cols: usize, out_trits: &mut [i8]) -> Result<(), String> {
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

        if out_trits.len() != total_weights {
            return Err(format!("output buffer holds {} trits, expected {}", out_trits.len(), total_weights));
        }

        // Element order of ggml `dequantize_row_tq2_0`: for each group of 32 bytes, the four 2-bit
        // planes are emitted one after another, i.e. element `l * 32 + m` of a group comes from bits
        // `2l..2l+1` of byte `m`.
        // Two groups of 32 bytes with plane 0 in the low bits; 1 is zero and the unused code 3 decodes to zero.
        let decode_block = |block: &[u8], out_block: &mut [i8]| decode_2bit_planes(&block[..64], out_block, PlaneOrder::LowFirst);

        if total_weights >= PARALLEL_MIN_WEIGHTS {
            out_trits
                .par_chunks_exact_mut(QK_TQ2_0)
                .zip(data[..expected_bytes].par_chunks_exact(BLOCK_BYTES_TQ2_0))
                .for_each(|(out_block, block)| decode_block(block, out_block));
        } else {
            for (out_block, block) in out_trits.chunks_exact_mut(QK_TQ2_0).zip(data[..expected_bytes].chunks_exact(BLOCK_BYTES_TQ2_0)) {
                decode_block(block, out_block);
            }
        }

        Ok(())
    }

    fn transcode_to_velurex_tiled(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        crate::transcoder::transcode_to_tiled(self, data, rows, cols, 32)
    }
}
