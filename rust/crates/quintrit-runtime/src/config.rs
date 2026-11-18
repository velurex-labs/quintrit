//! Model configuration and architectural hyperparameters for LLaMA, Falcon3, and BitNet causal models.

use serde::{Deserialize, Serialize};
use quintrit_format::GgufFile;

/// Nonlinear activation function in the Feed-Forward Network (FFN).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    /// Squared ReLU: f(x) = (max(0, x))^2 (Microsoft BitNet b1.58 default).
    Relu2,
    /// Sigmoid Linear Unit / Swish: f(x) = x * sigmoid(x) (Falcon3 & LLaMA default).
    Silu,
}

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// Microsoft BitNet b1.58 family (Sub-LN RMSNorm, Squared ReLU).
    BitNet,
    /// TII Falcon3 1.58-bit family (SwiGLU, strict power-of-two).
    Falcon3,
    /// Standard LLaMA / LLaMA-3 architecture.
    Llama,
    /// Qwen3 transformer (`qwen3`): RMS-normalized query and key heads, NEOX RoPE.
    Qwen3,
    /// Qwen3.5 hybrid (`qwen35`): Gated DeltaNet linear-attention layers with a gated full-attention layer at a fixed
    /// interval.
    Qwen35,
    /// NVIDIA Nemotron-H with mixture-of-experts layers (`nemotron_h_moe`): Mamba-2, attention and MoE layers.
    NemotronHMoe,
}

/// Which reference runtime's arithmetic the forward pass reproduces where the two differ.
///
/// The `I2_S` format runs only in Microsoft's bitnet.cpp, the llama.cpp formats (`Q2_0`, `TQ1_0`, `TQ2_0`) only in
/// upstream llama.cpp, and the flash attention of the two accumulates the values in different precisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Numerics {
    /// bitnet.cpp: flash attention accumulates the values in FP16.
    #[default]
    BitnetCpp,
    /// upstream llama.cpp: flash attention accumulates the values in FP32.
    LlamaCpp,
}

/// How Rotary Positional Embeddings pair the coordinates of a head (ggml `GGML_ROPE_TYPE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeMode {
    /// Adjacent pairs `(x[2i], x[2i+1])` (`GGML_ROPE_TYPE_NORMAL`; LLaMA and Falcon3 GGUFs, whose Q/K weights the
    /// converter permutes).
    Normal,
    /// Pairs offset by half the rotated dimensions, `(x[i], x[i + n_dims/2])` (`GGML_ROPE_TYPE_NEOX`; BitNet, Qwen).
    Neox,
}

/// Rotary Positional Embedding parameters, named as in ggml's `ggml_rope_ext`.
///
/// Only the first `n_dims` coordinates of a head are rotated; the rest pass through unchanged. With `ext_factor != 0`
/// the frequencies are YaRN-interpolated exactly like ggml's `rope_yarn`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeConfig {
    /// Coordinate pairing.
    pub mode: RopeMode,
    /// Number of rotated coordinates per head (`rope.dimension_count`), even and at most the head dimension.
    pub n_dims: usize,
    /// Base frequency `theta_base`.
    pub freq_base: f32,
    /// Frequency scale (`1 / rope.scaling.factor` for linear and YaRN scaling, otherwise 1).
    pub freq_scale: f32,
    /// YaRN extrapolation mix factor (1 for YaRN scaling, otherwise 0).
    pub ext_factor: f32,
    /// Magnitude factor applied to `cos` and `sin` (llama.cpp's `yarn_attn_factor`).
    pub attn_factor: f32,
    /// YaRN low correction boundary (llama.cpp default 32).
    pub beta_fast: f32,
    /// YaRN high correction boundary (llama.cpp default 1).
    pub beta_slow: f32,
    /// Original training context of a YaRN-scaled model (`rope.scaling.original_context_length`).
    pub n_ctx_orig: usize,
}

impl RopeConfig {
    /// Plain RoPE without scaling, rotating `n_dims` coordinates.
    pub fn plain(mode: RopeMode, n_dims: usize, freq_base: f32, n_ctx_orig: usize) -> Self {
        Self {
            mode,
            n_dims,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig,
        }
    }
}

fn default_rope() -> RopeConfig {
    RopeConfig::plain(RopeMode::Normal, 128, 500000.0, 4096)
}

/// Complete configuration hyperparameters for a transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Underlying model architecture.
    pub architecture: ModelArchitecture,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension (embedding length, $d_{\text{model}}$).
    pub hidden_size: usize,
    /// Intermediate dimension for FFN projections.
    pub intermediate_size: usize,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (Grouped-Query Attention).
    pub num_key_value_heads: usize,
    /// Per-head dimension (typically `hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// Epsilon constant for RMSNorm stability.
    pub rms_norm_eps: f32,
    /// Base frequency constant for Rotary Positional Embeddings (RoPE).
    pub rope_theta: f32,
    /// Maximum context window sequence length.
    pub max_position_embeddings: usize,
    /// FFN activation function.
    pub hidden_act: ActivationType,
    /// Whether the model uses auxiliary sub-layer normalization (`attn_sub_norm`, `ffn_sub_norm`).
    pub has_sub_norm: bool,
    /// Whether language model head shares weights with token embeddings.
    pub tie_word_embeddings: bool,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID.
    pub eos_token_id: Option<u32>,
    /// Rotary positional embedding parameters (pairing, rotated dimensions, YaRN scaling).
    #[serde(default = "default_rope")]
    pub rope: RopeConfig,
    /// Whether every query and key head is RMS-normalized before RoPE (`attn_q_norm`, `attn_k_norm`; Qwen3).
    #[serde(default)]
    pub qk_norm: bool,
    /// Reference arithmetic where bitnet.cpp and llama.cpp differ.
    #[serde(default)]
    pub numerics: Numerics,
}

impl ModelConfig {
    /// Standard specification for Microsoft BitNet b1.58 2B-4T (`microsoft/bitnet-b1.58-2B-4T`).
    pub fn bitnet_2b_default() -> Self {
        Self {
            architecture: ModelArchitecture::BitNet,
            vocab_size: 128256,
            hidden_size: 2560,
            intermediate_size: 6912,
            num_hidden_layers: 30,
            num_attention_heads: 20,
            num_key_value_heads: 5,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 4096,
            hidden_act: ActivationType::Relu2,
            has_sub_norm: true,
            tie_word_embeddings: true,
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
            rope: RopeConfig::plain(RopeMode::Neox, 128, 500000.0, 4096),
            qk_norm: false,
            numerics: Numerics::BitnetCpp,
        }
    }

    /// Standard specification for TII Falcon3 1B Instruct 1.58-bit (`tiiuae/Falcon3-1B-Instruct-1.58bit`).
    pub fn falcon3_1b_default() -> Self {
        Self {
            architecture: ModelArchitecture::Falcon3,
            vocab_size: 131072,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10004.0,
            max_position_embeddings: 8192,
            hidden_act: ActivationType::Silu,
            has_sub_norm: false,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            rope: RopeConfig::plain(RopeMode::Normal, 256, 10004.0, 8192),
            qk_norm: false,
            numerics: Numerics::BitnetCpp,
        }
    }

    /// Standard specification for TII Falcon3 7B Instruct 1.58-bit (`tiiuae/Falcon3-7B-Instruct-1.58bit`).
    pub fn falcon3_7b_default() -> Self {
        Self {
            architecture: ModelArchitecture::Falcon3,
            vocab_size: 131072,
            hidden_size: 3072,
            intermediate_size: 23040,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 100000.0,
            max_position_embeddings: 32768,
            hidden_act: ActivationType::Silu,
            has_sub_norm: false,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            rope: RopeConfig::plain(RopeMode::Normal, 256, 100000.0, 32768),
            qk_norm: false,
            numerics: Numerics::BitnetCpp,
        }
    }

    /// Calculates the Grouped-Query Attention (GQA) sharing ratio ($H_Q / H_{KV}$).
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        assert_eq!(
            self.num_attention_heads % self.num_key_value_heads,
            0,
            "Query head count must be evenly divisible by KV head count"
        );
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Calculates total parameter count of one transformer block.
    pub fn block_parameter_count(&self) -> usize {
        let q = self.hidden_size * (self.num_attention_heads * self.head_dim);
        let k = self.hidden_size * (self.num_key_value_heads * self.head_dim);
        let v = self.hidden_size * (self.num_key_value_heads * self.head_dim);
        let o = (self.num_attention_heads * self.head_dim) * self.hidden_size;
        let gate = self.hidden_size * self.intermediate_size;
        let up = self.hidden_size * self.intermediate_size;
        let down = self.intermediate_size * self.hidden_size;
        q + k + v + o + gate + up + down
    }

    /// Extracts model configuration hyperparameters from GGUF metadata.
    ///
    /// Missing keys fall back to the values of BitNet b1.58 2B-4T. The values are checked (each at least 1 and below a
    /// generous upper bound, the embedding length divisible by the head count with an even head dimension, the head
    /// count a multiple of the KV head count, a KV cache of sane size); a file that violates this is an error.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch_str = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("bitnet-b1.58");

        let general_name = gguf
            .metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let general_basename = gguf
            .metadata
            .get("general.basename")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let architecture = if arch_str == "bitnet-b1.58" || arch_str == "bitnet" {
            ModelArchitecture::BitNet
        } else if arch_str == "qwen3" {
            ModelArchitecture::Qwen3
        } else if arch_str == "qwen35" {
            ModelArchitecture::Qwen35
        } else if arch_str == "nemotron_h_moe" {
            ModelArchitecture::NemotronHMoe
        } else if arch_str == "falcon3"
            || general_name.to_ascii_lowercase().contains("falcon")
            || general_basename.to_ascii_lowercase().contains("falcon")
        {
            ModelArchitecture::Falcon3
        } else {
            ModelArchitecture::Llama
        };

        // llama.cpp takes the vocabulary size from the tokenizer; files without `<arch>.vocab_size` (Qwen) rely on that.
        let vocab_size = gguf
            .metadata
            .get(&format!("{}.vocab_size", arch_str))
            .and_then(|v| v.as_usize())
            .or_else(|| gguf.tensors.get("token_embd.weight").and_then(|t| t.shape.get(1).copied()))
            .or_else(|| {
                gguf.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    quintrit_format::GgufMetadataValue::Array(a) => Some(a.len()),
                    _ => None,
                })
            })
            .unwrap_or(128256);

        let hidden_size = gguf
            .metadata
            .get(&format!("{}.embedding_length", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(2560);

        let intermediate_size = gguf
            .metadata
            .get(&format!("{}.feed_forward_length", arch_str))
            .and_then(scalar_or_max)
            .unwrap_or(6912);

        let num_hidden_layers = gguf
            .metadata
            .get(&format!("{}.block_count", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(30);

        let num_attention_heads = gguf
            .metadata
            .get(&format!("{}.attention.head_count", arch_str))
            .and_then(scalar_or_max)
            .unwrap_or(20);

        let num_key_value_heads = gguf
            .metadata
            .get(&format!("{}.attention.head_count_kv", arch_str))
            .and_then(scalar_or_max)
            .unwrap_or(5);

        let rms_norm_eps = gguf
            .metadata
            .get(&format!("{}.attention.layer_norm_rms_epsilon", arch_str))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        let rope_theta = gguf
            .metadata
            .get(&format!("{}.rope.freq_base", arch_str))
            .and_then(|v| v.as_f32())
            .unwrap_or(500000.0);

        let max_position_embeddings = gguf
            .metadata
            .get(&format!("{}.context_length", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);

        // Reject zero, absurd or inconsistent values before they are used as divisors or allocation sizes.
        crate::validate::check_hyperparameters(
            arch_str,
            [
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                max_position_embeddings,
            ],
            gguf.metadata.get(&format!("{}.attention.key_length", arch_str)).and_then(|v| v.as_usize()),
        )?;
        let key_length = gguf.metadata.get(&format!("{}.attention.key_length", arch_str)).and_then(|v| v.as_usize());
        let head_dim = key_length.unwrap_or(hidden_size / num_attention_heads);
        if head_dim == 0 || !head_dim.is_multiple_of(2) || head_dim > 4096 {
            return Err(format!("metadata '{arch_str}.attention.key_length' = {head_dim} must be even and in 2..=4096"));
        }
        let rope = rope_from_gguf(gguf, arch_str, head_dim, rope_theta, max_position_embeddings)?;

        // Resolve the BOS token exactly like the reference llama.cpp vocabulary loader:
        //  - `tokenizer.ggml.add_bos_token` overrides the default when present;
        //  - otherwise BOS is prepended for GPT-2 BPE vocabularies using a llama3-style
        //    pre-tokenizer (e.g. `falcon3`, `llama-bpe`, `tekken`);
        //  - GPT-2 BPE vocabularies default the BOS id to 11 when the file does not name one.
        let tokenizer_model = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str());
        let tokenizer_pre = gguf
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str());
        let add_bos = gguf
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| {
                matches!(
                    tokenizer_pre,
                    Some("llama3") | Some("llama-v3") | Some("llama-bpe") | Some("falcon3") | Some("falcon-h1")
                        | Some("pixtral") | Some("midm-2.0") | Some("lfm2") | Some("tekken")
                        | Some("jina-v5-nano")
                )
            });
        let bos_token_id = if add_bos {
            gguf.metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .or(if tokenizer_model == Some("gpt2") { Some(11) } else { None })
        } else {
            None
        };

        let eos_token_id = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let hidden_act = match architecture {
            ModelArchitecture::BitNet => ActivationType::Relu2,
            _ => ActivationType::Silu,
        };

        let has_sub_norm = architecture == ModelArchitecture::BitNet;
        let tie_word_embeddings = !gguf.tensors.contains_key("output.weight");
        let qk_norm = architecture == ModelArchitecture::Qwen3;
        let numerics = if gguf.tensors.values().any(|t| t.tensor_type == quintrit_format::GGML_TYPE_I2_S) {
            Numerics::BitnetCpp
        } else {
            Numerics::LlamaCpp
        };

        Ok(Self {
            architecture,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            hidden_act,
            has_sub_norm,
            tie_word_embeddings,
            bos_token_id,
            eos_token_id,
            rope,
            qk_norm,
            numerics,
        })
    }
}

/// Reads the RoPE parameters the way llama.cpp derives them (`llama_context` constructor and `llama_model_rope_type`).
fn rope_from_gguf(
    gguf: &GgufFile,
    arch: &str,
    head_dim: usize,
    freq_base: f32,
    n_ctx_train: usize,
) -> Result<RopeConfig, String> {
    let key = |k: &str| gguf.metadata.get(&format!("{arch}.{k}"));
    let mode = match arch {
        "bitnet" | "bitnet-b1.58" | "qwen2" | "qwen3" | "qwen35" => RopeMode::Neox,
        _ => RopeMode::Normal,
    };
    let n_dims = key("rope.dimension_count").and_then(|v| v.as_usize()).unwrap_or(head_dim);
    if n_dims == 0 || !n_dims.is_multiple_of(2) || n_dims > head_dim {
        return Err(format!("metadata '{arch}.rope.dimension_count' = {n_dims} must be even and at most the head dimension {head_dim}"));
    }
    let mut rope = RopeConfig::plain(mode, n_dims, freq_base, n_ctx_train);
    let scaling = key("rope.scaling.type").and_then(|v| v.as_str()).unwrap_or("none");
    let factor = key("rope.scaling.factor").and_then(|v| v.as_f32()).unwrap_or(0.0);
    if let Some(orig) = key("rope.scaling.original_context_length").and_then(|v| v.as_usize()) {
        rope.n_ctx_orig = orig;
    }
    match scaling {
        "none" => {}
        "linear" | "yarn" => {
            if factor != 0.0 {
                if !(factor.is_finite() && factor > 0.0) {
                    return Err(format!("metadata '{arch}.rope.scaling.factor' = {factor} is not a positive number"));
                }
                rope.freq_scale = 1.0 / factor;
            }
            if scaling == "yarn" {
                rope.ext_factor = 1.0;
                // llama.cpp: attn_factor = get_mscale(1/freq_scale, 1) / (1 + 0.1 ln(1/freq_scale)), both evaluated with
                // the multiply-add contracted into an FMA as clang does.
                let scale = 1.0 / rope.freq_scale;
                let get_mscale = |s: f32| if s <= 1.0 { 1.0 } else { (0.1f32 * 1.0f32).mul_add(s.ln(), 1.0) };
                rope.attn_factor = get_mscale(scale) * (1.0 / 0.1f32.mul_add(scale.ln(), 1.0));
            }
        }
        other => return Err(format!("RoPE scaling type '{other}' is not supported")),
    }
    Ok(rope)
}

/// An integer hyper-parameter that is either one value or (hybrid models) one value per layer; the largest is used.
fn scalar_or_max(v: &quintrit_format::GgufMetadataValue) -> Option<usize> {
    match v {
        quintrit_format::GgufMetadataValue::Array(a) => a.iter().filter_map(|x| x.as_usize()).max(),
        other => other.as_usize(),
    }
}
