//! # Quintrit Format (`quintrit-format`)
//!
//! GGUF model parser and dynamic in-memory weight transcoder for **velurex encoding** (1.60 bpw).
//!
//! ## Architectural Overview
//!
//! Rather than inventing a bespoke on-disk format prematurely, Quintrit provides
//! native ingest for standard GGUF files (specifically Microsoft BitNet b1.58 models and community
//! ternary GGUFs with `GGML_TYPE_I2_S`, `GGML_TYPE_TQ1_0`, or `GGML_TYPE_TQ2_0` tensors).
//!
//! During model loading, weight tensors are parsed from memory-mapped GGUF files and dynamically
//! transcoded into 1.60 bpw velurex encoding 32-row tiled buffers in RAM:
//! - **Modular Format Decoders**: Each GGUF quantization format (`I2_S`, `TQ1_0`, `TQ2_0`) is isolated in [`decoders`].
//! - **Immediate compatibility**: Users and evaluators can verify benchmarks using familiar GGUF files from Hugging Face without downloading unknown file formats.
//! - **No on-disk format churn**: The final on-disk file specification is postponed until AVX-512 vector permute kernels, GPU/Compute Shader kernels, and candidate radix vectors are comprehensively evaluated.
//! - **Exact 20.0% DRAM reduction**: Transcoded matrices in RAM occupy 1.60 bpw (5 trits/byte) instead of 2.00 bpw (4 trits/byte), saving DRAM bandwidth during matrix-vector multiplication.

pub mod decoders;
pub mod gguf;
pub mod transcoder;

pub use decoders::{
    get_ternary_decoder, is_supported_ternary_type, supported_ternary_formats, uniform_block_scale,
    I2sDecoder,
    TernaryDecoder, Tq1_0Decoder, Tq2_0Decoder, I2_S_DECODER, TQ1_0_DECODER, TQ2_0_DECODER,
};

pub use gguf::{
    GgufFile, GgufMetadataValue, GgufTensorInfo, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32,
    GGML_TYPE_I2_S, GGML_TYPE_Q8_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
};

pub use transcoder::{
    load_and_transcode_gguf_tensor, transcode_gguf_i2_s_to_velurex_tiled,
    transcode_trits_to_velurex_tiled, verify_transcoded_parity, TranscodedTernaryMatrix,
};

/// Compiles the Rust examples of the crate README as doctests, so that they cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
