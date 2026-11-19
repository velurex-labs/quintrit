//! llama.cpp TQ1_0 format decoder (`GGML_TYPE_TQ1_0` = 34, 1.6875 bpw).
//!
//! # Storage Architecture (`block_tq1_0`)
//! - Block size: 256 weights in 54 bytes.
//! - `qs[48]`: 48 bytes encoding 240 weights, five per byte, as a base-3 fixed-point fraction
//!   $q = \lceil 256 v / 243 \rceil$ with $v = 81 d_0 + 27 d_1 + 9 d_2 + 3 d_3 + d_4$ and $d_k = t_k + 1$
//!   (first weight most significant).
//! - `qh[4]`: 4 bytes encoding the remaining 16 weights, four per byte, in the same way (the fifth digit is zero).
//! - Decoding needs no division: $d_n = \lfloor 3 (3^n q \bmod 256) / 256 \rfloor$.
//! - `d`: 2-byte FP16 per-block scale factor.

use super::TernaryDecoder;
use crate::gguf::GGML_TYPE_TQ1_0;
use quintrit_core::pack::PARALLEL_MIN_WEIGHTS;
use rayon::prelude::*;

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

    fn row_bytes(&self, cols: usize) -> Option<usize> {
        cols.is_multiple_of(QK_TQ1_0).then_some(cols / QK_TQ1_0 * BLOCK_BYTES_TQ1_0)
    }

    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String> {
        let mut trits = vec![0i8; rows * cols];
        self.unpack_trits_into(data, rows, cols, &mut trits)?;
        Ok(trits)
    }

    fn unpack_trits_into(&self, data: &[u8], rows: usize, cols: usize, out_trits: &mut [i8]) -> Result<(), String> {
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

        if out_trits.len() != total_weights {
            return Err(format!("output buffer holds {} trits, expected {}", out_trits.len(), total_weights));
        }

        // ggml stores five trits per byte as a base-3 fraction with the first trit most significant;
        // trit `n` is recovered by multiplying with 3^n modulo 256 and scaling by 3 / 256
        // (`dequantize_row_tq1_0`). The element order is strided: within a group of bytes, all
        // bytes' first trits come first, then all second trits, and so on.
        const POW3: [u32; 5] = [1, 3, 9, 27, 81];
        let digit = |byte: u8, n: usize| -> i8 {
            let q = (byte as u32 * POW3[n]) as u8 as u32;
            (((q * 3) >> 8) as i16 - 1) as i8
        };

        // Fixed trip counts and no branches, so that the compiler vectorizes the loops for the target.
        let decode_block = |block: &[u8], out_block: &mut [i8]| {
            let qs = &block[0..48];
            let qh = &block[48..52];
            let (out_32, rest) = out_block.split_at_mut(160);
            let (out_16, out_4) = rest.split_at_mut(80);

            // 32 bytes -> 160 trits
            for (n, plane) in out_32.chunks_exact_mut(32).enumerate() {
                for m in 0..32 {
                    plane[m] = digit(qs[m], n);
                }
            }
            // 16 bytes -> 80 trits
            for (n, plane) in out_16.chunks_exact_mut(16).enumerate() {
                for m in 0..16 {
                    plane[m] = digit(qs[32 + m], n);
                }
            }
            // 4 bytes -> 16 trits (four trits per byte)
            for (n, plane) in out_4.chunks_exact_mut(4).enumerate() {
                for j in 0..4 {
                    plane[j] = digit(qh[j], n);
                }
            }
        };

        if total_weights >= PARALLEL_MIN_WEIGHTS {
            out_trits
                .par_chunks_exact_mut(QK_TQ1_0)
                .zip(data[..expected_bytes].par_chunks_exact(BLOCK_BYTES_TQ1_0))
                .for_each(|(out_block, block)| decode_block(block, out_block));
        } else {
            for (out_block, block) in out_trits.chunks_exact_mut(QK_TQ1_0).zip(data[..expected_bytes].chunks_exact(BLOCK_BYTES_TQ1_0)) {
                decode_block(block, out_block);
            }
        }

        Ok(())
    }

    fn transcode_to_velurex_tiled(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
        crate::transcoder::transcode_to_tiled(self, data, rows, cols, 32)
    }
}
