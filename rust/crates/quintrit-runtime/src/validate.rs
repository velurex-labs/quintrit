//! Validation of a GGUF model before it is loaded.
//!
//! The forward pass indexes the embedding table, the norms and the KV cache without bounds checks of its own, and the
//! hyper-parameters drive divisions and allocation sizes. A damaged or foreign file must therefore be rejected while it
//! is loaded, with an error and never with a panic, an abort on a failed allocation or a division by zero. Every
//! front end (the Rust engine, the C library and the Python package) loads models through [`validate_gguf_model`].

use crate::config::ModelConfig;
use quintrit_format::{GgufFile, GgufMetadataValue, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S};

/// Largest accepted size of the KV cache (keys and values, F16) for the full context of a model: 4 TiB. The cache is
/// allocated lazily (zeroed pages), so only the part that is used occupies memory; the bound rejects hyper-parameters
/// whose size computation would be absurd or overflow.
pub const MAX_KV_CACHE_BYTES: usize = 1 << 42;

/// Bytes of a KV cache for `layers` layers, `ctx` positions, `kv_heads` heads of `head_dim` values, keys and values in
/// F16; `None` if the product overflows `usize`.
pub fn kv_cache_bytes(layers: usize, ctx: usize, kv_heads: usize, head_dim: usize) -> Option<usize> {
    [layers, ctx, kv_heads, head_dim, 2 * size_of::<u16>()]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
}

/// Upper bounds (inclusive) of the hyper-parameters; all of them must also be at least 1.
const LIMITS: [(&str, usize); 7] = [
    ("vocab_size", 1 << 24),
    ("embedding_length", 1 << 20),
    ("feed_forward_length", 1 << 22),
    ("block_count", 4096),
    ("attention.head_count", 4096),
    ("attention.head_count_kv", 4096),
    ("context_length", 1 << 24),
];

/// Checks the hyper-parameters read from the metadata of a model whose architecture key prefix is `arch`.
///
/// `values` are, in the order of the limits: vocabulary size, embedding length, feed-forward length, block count, head
/// count, KV head count and context length.
pub(crate) fn check_hyperparameters(arch: &str, values: [usize; 7]) -> Result<(), String> {
    for ((name, max), v) in LIMITS.iter().zip(values) {
        if v == 0 || v > *max {
            return Err(format!("metadata '{arch}.{name}' = {v} is outside 1..={max}"));
        }
    }
    let [_, hidden, _, layers, heads, kv_heads, ctx] = values;
    if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(2) {
        return Err(format!(
            "embedding_length {hidden} must be divisible by head_count {heads} with an even head dimension"
        ));
    }
    if !heads.is_multiple_of(kv_heads) {
        return Err(format!("head_count {heads} must be a multiple of head_count_kv {kv_heads}"));
    }
    match kv_cache_bytes(layers, ctx, kv_heads, hidden / heads) {
        Some(bytes) if bytes <= MAX_KV_CACHE_BYTES => Ok(()),
        _ => Err(format!(
            "a context of {ctx} tokens with {layers} layers and {kv_heads} KV heads needs an unreasonably large KV cache"
        )),
    }
}

/// Reads and checks the configuration of a model and the tensors the loader and the forward pass index directly.
///
/// Checked: the hyper-parameters (see [`ModelConfig::from_gguf`]), the tokenizer vocabulary (present, not larger than
/// the vocabulary size, so every token id the tokenizer can produce indexes the embedding table), the type and the exact
/// element count of the embedding table, the output head and every norm, and the shape of every `I2_S` tensor.
/// The ternary projections themselves are checked by their decoders while they are transcoded.
pub fn validate_gguf_model(g: &GgufFile) -> Result<ModelConfig, String> {
    let config = ModelConfig::from_gguf(g)?;
    let vocab = config.vocab_size;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let q_dim = config.num_attention_heads * config.head_dim;

    match g.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufMetadataValue::Array(a)) if !a.is_empty() && a.len() <= vocab => {}
        Some(GgufMetadataValue::Array(a)) => {
            return Err(format!("tokenizer has {} tokens but vocab_size is {vocab}", a.len()));
        }
        _ => return Err("metadata 'tokenizer.ggml.tokens' is missing or not an array".to_string()),
    }

    let expect = |name: &str, ty: u32, elements: usize, at_least: bool| -> Result<(), String> {
        let Some(t) = g.tensors.get(name) else {
            return Err(format!("tensor '{name}' is missing"));
        };
        let n = t.num_elements();
        if t.tensor_type != ty || (if at_least { n < elements } else { n != elements }) {
            return Err(format!(
                "tensor '{name}' has type {} and {n} elements, expected type {ty} and {}{elements}",
                t.tensor_type,
                if at_least { "at least " } else { "" }
            ));
        }
        Ok(())
    };
    // `vocab * hidden` cannot overflow: both are bounded by `check_hyperparameters`.
    expect("token_embd.weight", GGML_TYPE_F16, vocab * hidden, false)?;
    expect("output_norm.weight", GGML_TYPE_F32, hidden, false)?;
    if g.tensors.contains_key("output.weight") {
        expect("output.weight", GGML_TYPE_F16, vocab * hidden, false)?;
    }
    for l in 0..config.num_hidden_layers {
        expect(&format!("blk.{l}.attn_norm.weight"), GGML_TYPE_F32, hidden, false)?;
        expect(&format!("blk.{l}.ffn_norm.weight"), GGML_TYPE_F32, hidden, false)?;
        if config.has_sub_norm {
            expect(&format!("blk.{l}.attn_sub_norm.weight"), GGML_TYPE_F32, q_dim, true)?;
            expect(&format!("blk.{l}.ffn_sub_norm.weight"), GGML_TYPE_F32, inter, true)?;
        }
    }
    // The I2_S decoders need whole 128-trit blocks per row.
    for t in g.tensors.values() {
        if t.tensor_type == GGML_TYPE_I2_S && (t.shape.len() != 2 || !t.shape[0].is_multiple_of(128)) {
            return Err(format!("I2_S tensor '{}' needs two dimensions with the first a multiple of 128", t.name));
        }
    }
    Ok(config)
}
