//! # Quintrit Standalone Inference Runtime (`quintrit-runtime`)
//!
//! Ultra-lean, high-performance local inference engine in pure Rust without
//! external C++ or GGML runtime dependencies.
//!
//! ## Submodules
//! - [`config`]: Model hyperparameters and architecture definitions (BitNet, Falcon3, LLaMA).
//! - [`ops`]: Mathematical primitives (RMSNorm, RoPE, Int8 activation quantization, SiLU, Relu2, SwiGLU).
//! - [`layer`]: Multi-paradigm ternary GEMV dispatchers (Quintrit 1.60 bpw).
//! - [`kv_cache`]: Contiguous preallocated Key-Value cache.
//! - [`attention`]: Grouped-Query Attention (GQA) with causal masking.
//! - [`model`]: Complete autoregressive transformer forward execution graph.
//! - [`tokenizer`]: HuggingFace BPE tokenizer integration from GGUF metadata.
//! - [`sampler`]: Autoregressive sampling strategies (greedy, temperature, top-k, top-p).
//! - [`engine`]: High-level inference and benchmark coordination engine.
//! - [`validate`]: Checks of a GGUF model (hyper-parameters, tensor sizes) before it is loaded.
//! - [`verbosity`]: Process-wide switch for load-time progress messages (off by default; see [`set_verbose`]).
//!
//! ## Quiet by default
//! Loading a model prints nothing unless the application calls [`set_verbose`]`(true)`; the `quintrit` CLI does.

pub mod attention;
pub mod config;
pub mod engine;
pub mod kv_cache;
pub mod layer;
pub mod model;
pub mod ops;
pub mod sampler;
pub mod scratchpad;
pub mod tokenizer;
pub mod validate;
pub mod verbosity;

pub use config::{ActivationType, ModelArchitecture, ModelConfig};
pub use engine::{format_chat_prompt, BenchmarkReport, ChatMessage, GenerationStats, InferenceEngine};
pub use kv_cache::KVCache;
pub use layer::{
    dispatch_ternary_layer, dispatch_weight_from_i2_s, LinearLayer, QuintritLayer, TernaryGemvLayer,
    TernaryKernelType,
};
pub use model::{TransformerBlock, QuintritModel};
pub use sampler::{Sampler, SamplingConfig};
pub use scratchpad::InferenceScratchpad;
pub use tokenizer::{QuintritTokenizer, StreamDecoder};
pub use validate::validate_gguf_model;
pub use verbosity::{is_verbose, set_verbose, VerboseScope};


/// Compiles the Rust examples of the crate README as doctests, so that they cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
