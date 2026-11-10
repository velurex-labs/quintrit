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
        } else if arch_str == "falcon3"
            || general_name.to_ascii_lowercase().contains("falcon")
            || general_basename.to_ascii_lowercase().contains("falcon")
        {
            ModelArchitecture::Falcon3
        } else {
            ModelArchitecture::Llama
        };

        let vocab_size = gguf
            .metadata
            .get(&format!("{}.vocab_size", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(128256);

        let hidden_size = gguf
            .metadata
            .get(&format!("{}.embedding_length", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(2560);

        let intermediate_size = gguf
            .metadata
            .get(&format!("{}.feed_forward_length", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(6912);

        let num_hidden_layers = gguf
            .metadata
            .get(&format!("{}.block_count", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(30);

        let num_attention_heads = gguf
            .metadata
            .get(&format!("{}.attention.head_count", arch_str))
            .and_then(|v| v.as_usize())
            .unwrap_or(20);

        let num_key_value_heads = gguf
            .metadata
            .get(&format!("{}.attention.head_count_kv", arch_str))
            .and_then(|v| v.as_usize())
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
        )?;
        let head_dim = hidden_size / num_attention_heads;

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
        let tie_word_embeddings = architecture == ModelArchitecture::BitNet;

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
        })
    }
}
