//! Validation of a GGUF model before it is loaded.
//!
//! The forward pass indexes the embedding table, the norms and the KV cache without bounds checks of its own, and the
//! hyper-parameters drive divisions and allocation sizes. A damaged or foreign file must therefore be rejected while it
//! is loaded, with an error and never with a panic, an abort on a failed allocation or a division by zero. Every
//! front end (the Rust engine, the C library and the Python package) loads models through [`validate_gguf_model`].

use crate::config::ModelConfig;
use quintrit_format::{GgufFile, GgufMetadataValue, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S, GGML_TYPE_Q2_0};

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

/// Hyper-parameter keys (after the `<architecture>.` prefix) that the loaders read and implement.
const ARCH_KEYS: [&str; 33] = [
    "vocab_size",
    "context_length",
    "embedding_length",
    "block_count",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    // Only the final layernorm of Nemotron-H's multi-token-prediction head uses it; that head is not run.
    "attention.layer_norm_epsilon",
    "rope.dimension_count",
    "rope.dimension_sections",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.inner_size",
    "ssm.time_step_rank",
    "full_attention_interval",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "expert_shared_count",
    "expert_weights_norm",
    "expert_weights_scale",
    "expert_group_count",
    "expert_group_used_count",
    // Informational in llama.cpp as well.
    "rope.scaling.finetuned",
];

/// Tokenizer keys the loader reads (or that carry no meaning for byte-level BPE, such as the SentencePiece scores).
const TOKENIZER_KEYS: [&str; 14] = [
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
    // The chat template is applied by the caller (`format_chat_prompt`); the runtime does not evaluate Jinja.
    "tokenizer.chat_template",
];

/// The PrismML Hadamard keys that `hybrid` implements.
const PRISM_KEYS: [&str; 11] = [
    "prism.hadamard.version",
    "prism.hadamard.tied_output",
    "prism.hadamard.block_size",
    "prism.hadamard.transform",
    "prism.hadamard.axis",
    "prism.hadamard.sign_mode",
    "prism.hadamard.sign_widths",
    "prism.hadamard.sign_values",
    "prism.hadamard.weight_names",
    "prism.hadamard.inverse_weight_names",
    "prism.hadamard.gdn_v_grouped",
];

/// Rejects a file with metadata the runtime would silently ignore: every key under the architecture prefix, `tokenizer.`
/// and `prism.` must be one the loaders implement, and the values the runtime does not implement (another tokenizer
/// model, an added end-of-sequence token, a space prefix, grouped expert routing, value heads of another width) are
/// errors. Keys under `general.` are descriptive (names, licence, sampling suggestions) and allowed.
pub fn check_metadata_keys(g: &GgufFile) -> Result<(), String> {
    let arch = g.metadata.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("bitnet-b1.58");
    let arch_prefix = format!("{arch}.");
    let mut unknown = Vec::new();
    for key in g.metadata.keys() {
        let known = if key.starts_with("general.") {
            true
        } else if let Some(rest) = key.strip_prefix(&arch_prefix) {
            ARCH_KEYS.contains(&rest)
        } else if key.starts_with("tokenizer.") {
            TOKENIZER_KEYS.contains(&key.as_str())
        } else if key.starts_with("prism.") {
            PRISM_KEYS.contains(&key.as_str())
        } else {
            false
        };
        if !known {
            unknown.push(key.clone());
        }
    }
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!("the file has metadata this runtime does not implement: {}", unknown.join(", ")));
    }
    let get = |k: &str| g.metadata.get(k);
    if let Some(model) = get("tokenizer.ggml.model").and_then(|v| v.as_str()) {
        if model != "gpt2" {
            return Err(format!("tokenizer model '{model}' is not supported (only byte-level BPE, 'gpt2')"));
        }
    }
    for flag in ["tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix"] {
        if get(flag).and_then(|v| v.as_bool()) == Some(true) {
            return Err(format!("metadata '{flag}' = true is not supported"));
        }
    }
    for k in ["expert_group_count", "expert_group_used_count"] {
        if get(&format!("{arch_prefix}{k}")).and_then(|v| v.as_u64()).is_some_and(|n| n > 1) {
            return Err(format!("grouped expert routing ('{arch_prefix}{k}' > 1) is not supported"));
        }
    }
    if let Some(n) = get(&format!("{arch_prefix}expert_shared_count")).and_then(|v| v.as_u64()) {
        if n > 1 {
            return Err(format!("{n} shared experts are not supported (at most one)"));
        }
    }
    let key_len = get(&format!("{arch_prefix}attention.key_length")).and_then(|v| v.as_u64());
    let value_len = get(&format!("{arch_prefix}attention.value_length")).and_then(|v| v.as_u64());
    if value_len.is_some() && value_len != key_len {
        return Err(format!("attention value length {value_len:?} differs from the key length {key_len:?}"));
    }
    Ok(())
}

/// Checks the hyper-parameters read from the metadata of a model whose architecture key prefix is `arch`.
///
/// `values` are, in the order of the limits: vocabulary size, embedding length, feed-forward length, block count, head
/// count, KV head count and context length. `head_dim` is the explicit head dimension (`attention.key_length`) when the
/// file gives one; otherwise the embedding length must split evenly into the heads.
pub(crate) fn check_hyperparameters(arch: &str, values: [usize; 7], head_dim: Option<usize>) -> Result<(), String> {
    for ((name, max), v) in LIMITS.iter().zip(values) {
        if v == 0 || v > *max {
            return Err(format!("metadata '{arch}.{name}' = {v} is outside 1..={max}"));
        }
    }
    let [_, hidden, _, layers, heads, kv_heads, ctx] = values;
    if head_dim.is_none() && (!hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(2)) {
        return Err(format!(
            "embedding_length {hidden} must be divisible by head_count {heads} with an even head dimension"
        ));
    }
    if !heads.is_multiple_of(kv_heads) {
        return Err(format!("head_count {heads} must be a multiple of head_count_kv {kv_heads}"));
    }
    match kv_cache_bytes(layers, ctx, kv_heads, head_dim.unwrap_or(hidden / heads)) {
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
    check_metadata_keys(g)?;
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
    // The embedding table and the output head are F16, or Q2_0 in the ternary-embedding models (Ternary Bonsai).
    let table_type = |name: &str| match g.tensors.get(name).map(|t| t.tensor_type) {
        Some(GGML_TYPE_Q2_0) => GGML_TYPE_Q2_0,
        Some(quintrit_format::GGML_TYPE_Q8_0) => quintrit_format::GGML_TYPE_Q8_0,
        _ => GGML_TYPE_F16,
    };
    expect("token_embd.weight", table_type("token_embd.weight"), vocab * hidden, false)?;
    expect("output_norm.weight", GGML_TYPE_F32, hidden, false)?;
    if g.tensors.contains_key("output.weight") {
        expect("output.weight", table_type("output.weight"), vocab * hidden, false)?;
    }
    // The hybrid architectures check their own layers while they are loaded.
    let dense = !matches!(
        config.architecture,
        crate::config::ModelArchitecture::Qwen35 | crate::config::ModelArchitecture::NemotronHMoe
    );
    for l in (0..config.num_hidden_layers).filter(|_| dense) {
        expect(&format!("blk.{l}.attn_norm.weight"), GGML_TYPE_F32, hidden, false)?;
        expect(&format!("blk.{l}.ffn_norm.weight"), GGML_TYPE_F32, hidden, false)?;
        if config.qk_norm {
            expect(&format!("blk.{l}.attn_q_norm.weight"), GGML_TYPE_F32, config.head_dim, false)?;
            expect(&format!("blk.{l}.attn_k_norm.weight"), GGML_TYPE_F32, config.head_dim, false)?;
        }
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
