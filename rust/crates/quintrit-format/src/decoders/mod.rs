//! Modular GGUF tensor format decoding and transcoding subsystem.
//!
//! Rather than accumulating format conversions in a monolithic module, this subsystem
//! decomposes each GGML quantization format into an isolated, self-contained implementation
//! conforming to the [`TernaryDecoder`] trait.

pub mod i2_s;
pub mod tq1_0;
pub mod tq2_0;

pub use i2_s::I2sDecoder;
pub use tq1_0::Tq1_0Decoder;
pub use tq2_0::Tq2_0Decoder;

use crate::gguf::{GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0};

/// Common contract for all ternary quantization formats in GGUF.
pub trait TernaryDecoder: Send + Sync {
    /// Returns the unique GGML integer type constant (e.g., `36` for `I2_S`, `34` for `TQ1_0`).
    fn tensor_type(&self) -> u32;

    /// Canonical human-readable format identifier.
    fn format_name(&self) -> &'static str;

    /// Nominal storage density in bits per weight.
    fn nominal_bpw(&self) -> f32;

    /// Unpacks raw tensor bytes into a contiguous slice of trits in `{-1, 0, +1}`.
    fn unpack_trits(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>, String>;

    /// Transcodes raw tensor bytes directly into 1.60 bpw velurex encoding 32-row tiled format in RAM.
    fn transcode_to_velurex_tiled(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, String>;
}

/// Static decoder instance for Microsoft BitNet planar (`GGML_TYPE_I2_S` = 36).
pub static I2_S_DECODER: I2sDecoder = I2sDecoder;

/// Static decoder instance for llama.cpp TQ1_0 (`GGML_TYPE_TQ1_0` = 34).
pub static TQ1_0_DECODER: Tq1_0Decoder = Tq1_0Decoder;

/// Static decoder instance for llama.cpp TQ2_0 (`GGML_TYPE_TQ2_0` = 35).
pub static TQ2_0_DECODER: Tq2_0Decoder = Tq2_0Decoder;

/// Resolves a [`TernaryDecoder`] implementation based on the GGUF `tensor_type` ID.
pub fn get_ternary_decoder(tensor_type: u32) -> Result<&'static dyn TernaryDecoder, String> {
    match tensor_type {
        GGML_TYPE_I2_S => Ok(&I2_S_DECODER),
        GGML_TYPE_TQ1_0 => Ok(&TQ1_0_DECODER),
        GGML_TYPE_TQ2_0 => Ok(&TQ2_0_DECODER),
        other => Err(format!(
            "Unsupported GGML tensor type code {} (supported ternary formats: I2_S=36, TQ1_0=34, TQ2_0=35)",
            other
        )),
    }
}

/// Checks whether a given GGML `tensor_type` is a supported ternary format.
pub fn is_supported_ternary_type(tensor_type: u32) -> bool {
    matches!(tensor_type, GGML_TYPE_I2_S | GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0)
}

/// Returns a list of all currently supported ternary formats: `(format_name, tensor_type, bpw)`.
pub fn supported_ternary_formats() -> Vec<(&'static str, u32, f32)> {
    vec![
        (I2_S_DECODER.format_name(), I2_S_DECODER.tensor_type(), I2_S_DECODER.nominal_bpw()),
        (TQ1_0_DECODER.format_name(), TQ1_0_DECODER.tensor_type(), TQ1_0_DECODER.nominal_bpw()),
        (TQ2_0_DECODER.format_name(), TQ2_0_DECODER.tensor_type(), TQ2_0_DECODER.nominal_bpw()),
    ]
}

/// Converts an IEEE 754 half-precision value to f32 (used for the per-block scales of TQ formats).
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // Subnormal half: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Per-block scale `d` of a TQ1_0 / TQ2_0 tensor if all blocks share the same value.
///
/// The GEMV layers take a single weight scale per tensor. Ternary-trained models store the same
/// `d` in every block, so the scale is representable; tensors whose blocks differ return `None`
/// (they cannot be executed correctly with a per-tensor scale). Other formats return `None` too.
pub fn uniform_block_scale(tensor_type: u32, data: &[u8]) -> Option<f32> {
    let (block_bytes, scale_offset) = match tensor_type {
        GGML_TYPE_TQ1_0 => (tq1_0::BLOCK_BYTES_TQ1_0, 52usize),
        GGML_TYPE_TQ2_0 => (tq2_0::BLOCK_BYTES_TQ2_0, 64usize),
        _ => return None,
    };
    if data.is_empty() || data.len() % block_bytes != 0 {
        return None;
    }
    let read = |block: &[u8]| u16::from_le_bytes([block[scale_offset], block[scale_offset + 1]]);
    let mut blocks = data.chunks_exact(block_bytes);
    let first = read(blocks.next()?);
    if blocks.all(|b| read(b) == first) {
        Some(half_to_f32(first))
    } else {
        None
    }
}
