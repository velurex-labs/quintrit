//! Dynamic in-memory weight transcoding from GGUF tensors into 1.60 bpw Velurex format.
//!
//! # Architecture & Rationale
//!
//! Rather than introducing a proprietary on-disk format prematurely,
//! Quintrit loads standard GGUF model files (such as Microsoft BitNet b1.58
//! `GGML_TYPE_I2_S` tensors or llama.cpp `GGML_TYPE_TQ1_0` / `GGML_TYPE_TQ2_0`) directly
//! and transcodes the ternary weights on-the-fly into RAM.
//!
//! ## Key Benefits:
//! 1. **Zero-friction community evaluation**: Users can run models using familiar,
//!    standard GGUF files from HuggingFace without downloading custom file types.
//! 2. **Prevents on-disk format churn**: Postpones finalizing an on-disk format until
//!    AVX-512 vector permutes, GPU/Compute Shader layouts, and alternative radix decompositions
//!    are fully researched and benchmarked.
//! 3. **Exact 20.0% DRAM reduction in RAM**: Once transcoded in memory, weights occupy
//!    canonical 1.60 bpw (5 trits/byte) 32-row tiled layouts, reducing memory traffic
//!    and executing on the SIMD GEMV microkernel.
//!
//! ## Trade-off:
//! - Because weights must be dynamically transformed into Velurex 32-row tiles at startup,
//!   the file cannot be directly zero-copy memory-mapped from disk into execution registers.
//!   A multi-threaded in-memory transcoding step is performed during model load.

use crate::decoders::{get_ternary_decoder, I2_S_DECODER, TernaryDecoder};
use crate::gguf::GgufFile;
use quintrit_core::lut::{transform_weights_lut, untransform_weights_lut};
use quintrit_core::pack::{pack_matrix, unpack_matrix, BLOCK_BYTES_5, QK_5};
use crate::decoders::i2_s::unpack_gguf_i2_s;

/// In-memory representation of a weight matrix transcoded into 1.60 bpw velurex encoding
/// and arranged in a 32-row tiled format for direct hardware-accelerated SIMD GEMV execution.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscodedTernaryMatrix {
    /// 32-row tiled, 1.60 bpw packed weight bytes.
    pub packed_tiled: Vec<u8>,
    /// Number of output rows (projection dimension).
    pub rows: usize,
    /// Number of input columns (activation dimension).
    pub cols: usize,
    /// Per-tensor floating-point scale factor.
    pub scale: f32,
    /// Number of bytes per individual row in the untiled packed representation.
    pub bytes_per_row: usize,
}

impl TranscodedTernaryMatrix {
    /// Returns the active DRAM footprint of the matrix in bytes.
    #[inline]
    pub fn dram_bytes(&self) -> usize {
        self.packed_tiled.len()
    }

    /// Returns the equivalent DRAM footprint of the BitNet 2.00 bpw baseline in bytes.
    #[inline]
    pub fn baseline_2bpw_bytes(&self) -> usize {
        (self.rows * self.cols) / 4
    }

    /// Returns the exact percentage DRAM bandwidth / memory saved over the 2.00 bpw baseline.
    #[inline]
    pub fn dram_savings_pct(&self) -> f64 {
        let base = self.baseline_2bpw_bytes() as f64;
        let actual = self.dram_bytes() as f64;
        ((base - actual) / base) * 100.0
    }
}

/// Transcodes a BitNet 2.00 bpw planar weight matrix (`GGML_TYPE_I2_S`) into
/// Quintrit 1.60 bpw velurex encoding 32-row tiled format in RAM.
///
/// Matrix dimensions: `rows x cols` (where `cols` must be a multiple of 128 for BitNet planar).
#[inline]
pub fn transcode_gguf_i2_s_to_velurex_tiled(
    packed_planar: &[u8],
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    I2_S_DECODER
        .transcode_to_velurex_tiled(packed_planar, rows, cols)
        .expect("Failed to transcode I2_S planar tensor")
}

/// Transcodes raw trits (`-1, 0, +1`) directly into 1.60 bpw velurex encoding 32-row tiled layout.
pub fn transcode_trits_to_velurex_tiled(
    raw_trits: &[i8],
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    assert_eq!(
        raw_trits.len(),
        rows * cols,
        "raw_trits length must match rows * cols"
    );

    let packed = pack_matrix(raw_trits, rows, cols);
    transform_weights_lut(&packed, rows, cols)
}

/// Loads a named tensor from a memory-mapped GGUF file and transcodes it into
/// a 1.60 bpw velurex encoding 32-row tiled matrix in RAM via the appropriate modular decoder.
pub fn load_and_transcode_gguf_tensor(
    gguf: &GgufFile,
    tensor_name: &str,
    rows: usize,
    cols: usize,
) -> Result<TranscodedTernaryMatrix, String> {
    let tensor_info = gguf
        .tensors
        .get(tensor_name)
        .ok_or_else(|| format!("Tensor '{}' not found in GGUF file", tensor_name))?;

    let decoder = get_ternary_decoder(tensor_info.tensor_type)?;

    let raw_data = gguf
        .get_tensor_data(tensor_name)
        .ok_or_else(|| format!("Failed to read raw data for tensor '{}'", tensor_name))?;

    let scale = match gguf.get_tensor_scale(tensor_name) {
        Some(scale) => scale,
        None if matches!(tensor_info.tensor_type, crate::GGML_TYPE_TQ1_0 | crate::GGML_TYPE_TQ2_0) => {
            return Err(format!(
                "Tensor '{}' has differing per-block scales, which a per-tensor weight scale cannot represent",
                tensor_name
            ));
        }
        None => 1.0,
    };

    let packed_tiled = decoder.transcode_to_velurex_tiled(raw_data, rows, cols)?;
    let bytes_per_row = (cols.div_ceil(QK_5)) * BLOCK_BYTES_5;

    Ok(TranscodedTernaryMatrix {
        packed_tiled,
        rows,
        cols,
        scale,
        bytes_per_row,
    })
}

/// Verifies bit-for-bit mathematical identity between the original BitNet planar
/// input tensor and the transcoded Velurex tiled matrix.
pub fn verify_transcoded_parity(
    original_planar: &[u8],
    transcoded_tiled: &[u8],
    rows: usize,
    cols: usize,
) -> bool {
    let expected_trits = unpack_gguf_i2_s(original_planar, rows, cols);
    let untiled = untransform_weights_lut(transcoded_tiled, rows, cols);
    let actual_trits = unpack_matrix(&untiled, rows, cols);

    expected_trits == actual_trits[..rows * cols]
}
