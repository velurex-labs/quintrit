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
//!   A multi-threaded in-memory transcoding step is performed during model load: [`transcode_to_tiled`] converts one
//!   tile (32 or 64 rows) per thread, so the intermediate trits and planar bytes stay in that thread's cache.

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

/// Transcodes a tensor of `rows x cols` weights of any [`TernaryDecoder`] format into velurex encoding tiles of
/// `tile_rows` rows (32 for the `quintrit` kernel, 64 for the AVX-512 kernel).
///
/// If the format stores its rows independently ([`TernaryDecoder::row_bytes`]), every tile is converted by one thread
/// from its own slice of the payload (decode, pack, transpose) through buffers that stay in the cache and are reused
/// for the next tile of that thread; the result does not depend on the number of threads. Other formats are converted
/// matrix-wide. The rows must be a multiple of `tile_rows`.
pub fn transcode_to_tiled(
    decoder: &dyn TernaryDecoder,
    data: &[u8],
    rows: usize,
    cols: usize,
    tile_rows: usize,
) -> Result<Vec<u8>, String> {
    match tile_rows {
        32 => transcode_tiles::<32>(decoder, data, rows, cols),
        64 => transcode_tiles::<64>(decoder, data, rows, cols),
        other => Err(format!("unsupported tile height {other} (32 or 64)")),
    }
}

fn transcode_tiles<const R: usize>(
    decoder: &dyn TernaryDecoder,
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, String> {
    use quintrit_core::pack::{pack_matrix_into, tile_row_bytes, tile_rows_into, PARALLEL_MIN_WEIGHTS};
    use rayon::prelude::*;

    if rows % R != 0 {
        return Err(format!("{rows} rows are not a multiple of the tile height {R}"));
    }
    let src_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let tile_bytes = R * tile_row_bytes(cols);
    let mut out = vec![0u8; rows / R * tile_bytes];
    if out.is_empty() {
        return Ok(out);
    }

    let Some(row_bytes) = decoder.row_bytes(cols) else {
        // The rows are not independent: decode the whole matrix, then tile it block by block.
        let trits = decoder.unpack_trits(data, rows, cols)?;
        let packed = pack_matrix(&trits, rows, cols);
        for (b, tile) in out.chunks_exact_mut(tile_bytes).enumerate() {
            tile_rows_into::<R>(&packed[b * R * src_row_bytes..(b + 1) * R * src_row_bytes], src_row_bytes, cols, tile);
        }
        return Ok(out);
    };
    if data.len() < rows * row_bytes {
        return Err(format!("{} data buffer too short: expected {} bytes, got {}", decoder.format_name(), rows * row_bytes, data.len()));
    }

    let convert = |(trits, packed): &mut (Vec<i8>, Vec<u8>), b: usize, tile: &mut [u8]| -> Result<(), String> {
        decoder.unpack_trits_into(&data[b * R * row_bytes..(b + 1) * R * row_bytes], R, cols, trits)?;
        pack_matrix_into(trits, R, cols, packed);
        tile_rows_into::<R>(packed, src_row_bytes, cols, tile);
        Ok(())
    };
    let buffers = || (vec![0i8; R * cols], vec![0u8; R * src_row_bytes]);
    if rows * cols >= PARALLEL_MIN_WEIGHTS {
        out.par_chunks_exact_mut(tile_bytes)
            .enumerate()
            .try_for_each_init(buffers, |bufs, (b, tile)| convert(bufs, b, tile))?;
    } else {
        let mut bufs = buffers();
        for (b, tile) in out.chunks_exact_mut(tile_bytes).enumerate() {
            convert(&mut bufs, b, tile)?;
        }
    }
    Ok(out)
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
