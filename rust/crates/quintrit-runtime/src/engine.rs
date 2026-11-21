//! High-level inference engine coordinating model execution, tokenization, and benchmarking.

use crate::attention::AttentionBlock;
use crate::config::ModelArchitecture;
use crate::kv_cache::KVCache;
use crate::layer::{dispatch_ternary_layer, TernaryKernelType};
use crate::model::{TransformerBlock, QuintritModel};
use crate::sampler::{Sampler, SamplingConfig};
use crate::tokenizer::{QuintritTokenizer, StreamDecoder};
use crate::verbosity::progress;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use quintrit_format::{GgufFile, GgufMetadataValue};

/// Single conversation message for multi-turn chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

/// End-of-sequence token of Falcon3, which its chat template appends to every assistant turn.
pub const FALCON3_EOS_TOKEN: &str = "<|endoftext|>";

/// Formats a sequence of conversation messages into a prompt string based on model architecture.
///
/// The prompt always ends with the header of a new assistant turn. For Falcon3 it follows the chat template stored in
/// the model (`tokenizer.chat_template`, without tools): every assistant turn ends with [`FALCON3_EOS_TOKEN`], followed
/// by a newline unless it is the last message. The tokenizer of [`InferenceEngine`] encodes that text as the single
/// end-of-sequence id.
pub fn format_chat_prompt(messages: &[ChatMessage], arch: ModelArchitecture) -> String {
    match arch {
        ModelArchitecture::Falcon3 => {
            let mut prompt = String::new();
            for (i, msg) in messages.iter().enumerate() {
                let is_last = i + 1 == messages.len();
                let role = msg.role.trim().to_lowercase();
                match role.as_str() {
                    "system" => {
                        prompt.push_str("<|system|>\n");
                        prompt.push_str(&msg.content);
                        prompt.push('\n');
                    }
                    "user" => {
                        prompt.push_str("<|user|>\n");
                        prompt.push_str(&msg.content);
                        prompt.push('\n');
                    }
                    "assistant" => {
                        prompt.push_str("<|assistant|>\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str(FALCON3_EOS_TOKEN);
                        if !is_last {
                            prompt.push('\n');
                        }
                    }
                    _ => {
                        prompt.push_str("<|user|>\n");
                        prompt.push_str(&msg.content);
                        prompt.push('\n');
                    }
                }
            }
            prompt.push_str("<|assistant|>\n");
            prompt
        }
        ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 | ModelArchitecture::NemotronHMoe => {
            // ChatML, as in the chat template of the Qwen models; thinking is switched off with an empty block.
            let mut prompt = String::new();
            for msg in messages {
                let role = match msg.role.trim().to_lowercase().as_str() {
                    "system" => "system",
                    "assistant" => "assistant",
                    _ => "user",
                };
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            prompt
        }
        ModelArchitecture::BitNet | ModelArchitecture::Llama => {
            let mut prompt = String::new();
            for msg in messages {
                let role = msg.role.trim().to_lowercase();
                match role.as_str() {
                    "system" => {
                        prompt.push_str("### System:\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("\n\n");
                    }
                    "user" => {
                        prompt.push_str("### User:\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("\n\n");
                    }
                    "assistant" => {
                        prompt.push_str("### Assistant:\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("\n\n");
                    }
                    _ => {
                        prompt.push_str("### User:\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("\n\n");
                    }
                }
            }
            prompt.push_str("### Assistant:\n");
            prompt
        }
    }
}

/// Performance benchmark report for a token generation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Number of prompt tokens ingested.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
    /// Time to First Token (TTFT) in milliseconds.
    pub ttft_ms: f64,
    /// Generation time (excluding prompt ingestion) in milliseconds.
    pub generation_ms: f64,
    /// Generation throughput in tokens per second.
    pub tokens_per_sec: f64,
    /// Effective memory bandwidth utilization in GB/s.
    pub memory_bandwidth_gb_s: f64,
    /// Total DRAM weight traffic per token in MB (ternary projections plus the F16 `lm_head`).
    pub dram_traffic_per_token_mb: f64,
    /// Active kernel architecture used.
    pub kernel_type: TernaryKernelType,
    /// Text generated during the benchmark pass.
    pub generated_text: String,
    /// Count of layers dispatched by kernel type.
    pub kernel_counts: HashMap<String, usize>,
}

/// Standalone inference engine managing model execution and token generation.
pub struct InferenceEngine {
    pub model: QuintritModel,
    pub tokenizer: QuintritTokenizer,
    pub kv_cache: KVCache,
    pub sampler: Sampler,
    pub num_threads: usize,
    pub active_kernel: TernaryKernelType,
    pub scratchpad: crate::scratchpad::InferenceScratchpad,
    /// Tokens whose keys and values [`InferenceEngine::generate_streaming_with_stops`] left in `kv_cache`, position by
    /// position.
    cached_tokens: Vec<u32>,
    /// [`KVCache::reset_count`] at the time `cached_tokens` was recorded; a different value means that someone else
    /// has reset the cache since.
    cached_reset_count: u64,
    /// Statistics of the last [`InferenceEngine::generate_streaming_with_stops`] call.
    last_generation: GenerationStats,
    /// The decoder on the GPU, when the model was loaded with [`InferenceEngine::load_from_gguf_on_gpu`]; `kv_cache` is then only the bookkeeping of the
    /// positions (the keys and values live on the device).
    #[cfg(feature = "cuda")]
    gpu: Option<crate::gpu::GpuDecoder>,
    /// The weights on the GPU (shared by the sessions of a host that manages its own, such as the C API).
    #[cfg(feature = "cuda")]
    gpu_model: Option<std::sync::Arc<crate::gpu::GpuModel>>,
}

/// Token counts and timings of one [`InferenceEngine::generate_streaming_with_stops`] call.
///
/// The callback of that call receives pieces of text, not tokens: a token that ends inside a multi-byte character
/// yields no piece of its own, so counting callback calls does not count tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationStats {
    /// Tokens of the prompt, BOS included.
    pub prompt_tokens: usize,
    /// Prompt tokens taken from the KV cache of the previous call instead of being computed.
    pub reused_prompt_tokens: usize,
    /// Tokens sampled by the model, without the end-of-sequence token; the token that completed a stop sequence counts.
    pub generated_tokens: usize,
    /// Time spent on the prompt tokens that were computed, in milliseconds.
    pub prompt_ms: f64,
    /// Time from the end of the prompt to the end of the call (sampling, forward passes, decoding and the callback),
    /// in milliseconds.
    pub generation_ms: f64,
}

impl InferenceEngine {
    /// Loads a model from an official GGUF quantized file and dynamically transcodes
    /// ternary weights into 1.60 bpw Velurex format in RAM.
    pub fn load_from_gguf<P: AsRef<Path>>(
        path: P,
        num_threads: usize,
        force_kernel: Option<TernaryKernelType>,
    ) -> Result<Self, String> {
        let path_ref = path.as_ref();
        progress!("==> Loading GGUF model: {}", path_ref.display());
        let gguf = GgufFile::open(path_ref).map_err(|e| e.to_string())?;
        Self::load_from_parsed_gguf(&gguf, num_threads, force_kernel, Some(&path_ref.to_string_lossy()))
    }

    /// Loads a model from an already parsed/memory-mapped GGUF structure.
    pub fn load_from_parsed_gguf(
        gguf: &GgufFile,
        num_threads: usize,
        force_kernel: Option<TernaryKernelType>,
        model_name_hint: Option<&str>,
    ) -> Result<Self, String> {
        Self::load_impl(gguf, num_threads, force_kernel, model_name_hint, None)
    }

    /// Loads a model from a GGUF file onto NVIDIA GPU `ordinal` (feature `cuda`): the ternary weights are transcoded into the GPU layout while the file is
    /// read, the KV cache lives on the device, and every token is computed on the GPU with results bit-identical to the CPU path. `context` bounds the
    /// context window (default [`crate::gpu::DEFAULT_GPU_CONTEXT`], at most 7936 positions). Models the GPU decoder does not support return an error that
    /// says why.
    #[cfg(feature = "cuda")]
    pub fn load_from_gguf_on_gpu<P: AsRef<Path>>(path: P, ordinal: usize, context: Option<usize>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        progress!("==> Loading GGUF model for the GPU: {}", path_ref.display());
        let gguf = GgufFile::open(path_ref).map_err(|e| e.to_string())?;
        Self::load_impl(&gguf, 1, None, Some(&path_ref.to_string_lossy()), Some((ordinal, context, true)))
    }

    /// [`InferenceEngine::load_from_gguf_on_gpu`] for an already parsed GGUF structure.
    #[cfg(feature = "cuda")]
    pub fn load_from_parsed_gguf_on_gpu(gguf: &GgufFile, ordinal: usize, context: Option<usize>, model_name_hint: Option<&str>) -> Result<Self, String> {
        Self::load_impl(gguf, 1, None, model_name_hint, Some((ordinal, context, true)))
    }

    /// Loads only the weights onto the GPU: the engine has the tokenizer and the model description (and the embedding table) but no session, so
    /// [`InferenceEngine::forward_token`] fails; the sessions are made from [`InferenceEngine::gpu_model`] by the host (the C API does).
    #[cfg(feature = "cuda")]
    pub fn load_gpu_model_from_parsed_gguf(gguf: &GgufFile, ordinal: usize, context: Option<usize>, model_name_hint: Option<&str>) -> Result<Self, String> {
        Self::load_impl(gguf, 1, None, model_name_hint, Some((ordinal, context, false)))
    }

    /// The weights on the GPU, if the model was loaded there.
    #[cfg(feature = "cuda")]
    pub fn gpu_model(&self) -> Option<&std::sync::Arc<crate::gpu::GpuModel>> {
        self.gpu_model.as_ref()
    }

    /// Times the device work of one decode token on the GPU at a context of `context` tokens (feature `cuda`): the positions before it are filled with a
    /// fixed pseudo-random token sequence, then the token's CUDA graph is replayed `blocks` x `iterations` times. Returns the milliseconds per token of
    /// every block. Weights are far larger than the L2 cache, so every replay streams them from device memory, as in generation.
    #[cfg(feature = "cuda")]
    pub fn time_gpu_decode(&mut self, context: usize, blocks: usize, iterations: usize) -> Result<Vec<f64>, String> {
        if self.gpu.is_none() {
            return Err("the model is not on the GPU".into());
        }
        if context == 0 || context > self.kv_cache.max_seq_len() {
            return Err(format!("context must be 1..={}", self.kv_cache.max_seq_len()));
        }
        let vocab = self.model.config.vocab_size;
        let mut logits = vec![0f32; vocab];
        self.kv_cache.reset();
        for pos in 0..context - 1 {
            let token = ((pos * 7919 + 13) % vocab) as u32;
            self.forward_token(token, pos, &mut logits)?;
        }
        let hidden = self.model.config.hidden_size;
        let token = (((context - 1) * 7919 + 13) % vocab) as usize;
        let row = self.model.token_embeddings[token * hidden..(token + 1) * hidden].to_vec();
        let gpu = self.gpu.as_mut().expect("checked above");
        gpu.time_token(&row, context - 1, blocks, iterations)
    }

    /// Name of the GPU the model runs on, or `None` for a model on the CPU.
    pub fn gpu_device_name(&self) -> Option<String> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu_model {
            return Some(g.device_name().to_string());
        }
        None
    }

    /// One forward pass of `token_id` at position `pos` on the CPU or the GPU, writing the logits of the next token. The positions before `pos` must
    /// have been computed (in order) or still be in the cache; the generation methods do this themselves.
    pub fn forward_token(&mut self, token_id: u32, pos: usize, logits: &mut [f32]) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        if self.gpu.is_none() && self.gpu_model.is_some() {
            return Err("the model was loaded onto the GPU without a session; create a session from `gpu_model()`".to_string());
        }
        #[cfg(feature = "cuda")]
        if let Some(gpu) = self.gpu.as_mut() {
            let hidden = self.model.config.hidden_size;
            let off = token_id as usize * hidden;
            gpu.forward(&self.model.token_embeddings[off..off + hidden], pos, logits)?;
            self.kv_cache.mark_position(pos);
            return Ok(());
        }
        self.model.forward_token_with_scratchpad(token_id, pos, &mut self.kv_cache, &mut self.scratchpad, logits);
        Ok(())
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn load_impl(
        gguf: &GgufFile,
        num_threads: usize,
        force_kernel: Option<TernaryKernelType>,
        model_name_hint: Option<&str>,
        gpu_target: Option<(usize, Option<usize>, bool)>,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        #[cfg(feature = "cuda")]
        let mut gpu: Option<crate::gpu::GpuDecoder> = None;
        #[cfg(feature = "cuda")]
        let mut gpu_model: Option<std::sync::Arc<crate::gpu::GpuModel>> = None;

        // 1. Detect architecture & hyperparameters from GGUF metadata, and reject a damaged or foreign file (absurd
        //    hyper-parameters, missing or wrongly sized tensors) before anything is indexed or allocated.
        let mut config = crate::validate::validate_gguf_model(gguf)?;
        if let Some(hint) = model_name_hint {
            if hint.to_ascii_lowercase().contains("falcon") {
                config.architecture = ModelArchitecture::Falcon3;
            }
        }

        let target_kernel = force_kernel.unwrap_or(TernaryKernelType::QuintritLut);
        progress!("    [{:.2?}] configuration validated", t0.elapsed());

        progress!(
            "    Arch: {:?} | Layers: {} | Hidden: {} | FFN: {} | GQA: {}:{} | Kernel: {:?}",
            config.architecture,
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            target_kernel
        );

        // 2. Load Tokenizer directly from GGUF metadata tokens/merges
        progress!("==> Initializing HuggingFace BPE tokenizer from model metadata...");
        let tokens = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| "GGUF metadata missing tokenizer.ggml.tokens".to_string())?;

        let merges = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array());

        let token_types: Option<Vec<i32>> = match gguf.metadata.get("tokenizer.ggml.token_type") {
            Some(GgufMetadataValue::Array(items)) => items
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::Int32(t) => Some(*t),
                    other => other.as_u64().and_then(|t| i32::try_from(t).ok()),
                })
                .collect(),
            _ => None,
        };

        let tokenizer = QuintritTokenizer::from_gguf_metadata_with_pre(
            &tokens,
            merges.as_deref(),
            token_types.as_deref(),
            gguf.metadata.get("tokenizer.ggml.pre").and_then(|v| v.as_str()),
            config.bos_token_id,
            config.eos_token_id,
        )?;
        progress!("    [{:.2?}] tokenizer ready", t0.elapsed());

        let gpu_branch = gpu_target;
        let (model, kv_cache) = if let Some((ordinal, limit, with_session)) = gpu_branch {
            #[cfg(feature = "cuda")]
            {
                progress!("==> Loading the model onto the GPU...");
                let weights = std::sync::Arc::new(crate::gpu::GpuModel::load(gguf, &config, ordinal, limit)?);
                progress!("    [{:.2?}] weights on {}", t0.elapsed(), weights.device_name());
                let token_embeddings = gguf
                    .get_tensor_f16("token_embd.weight")
                    .ok_or_else(|| "Missing token_embd.weight tensor".to_string())?
                    .to_vec();
                let output_norm = gguf.get_tensor_f32("output_norm.weight").ok_or_else(|| "Missing output_norm.weight tensor".to_string())?;
                let kv = KVCache::try_new(1, weights.max_context(), 1, 2)?;
                let model = QuintritModel {
                    config: config.clone(),
                    token_embeddings,
                    blocks: Vec::new(),
                    output_norm,
                    lm_head: None,
                    lm_head_f16: None,
                    embedding_layer: None,
                    hybrid: None,
                };
                if with_session {
                    gpu = Some(crate::gpu::GpuDecoder::from_model(&weights)?);
                }
                gpu_model = Some(weights);
                (model, kv)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (ordinal, limit, with_session);
                return Err("this build has no GPU support (feature `cuda`)".to_string());
            }
        } else if matches!(config.architecture, ModelArchitecture::Qwen35 | ModelArchitecture::NemotronHMoe) {
            progress!("==> Loading hybrid layers ({:?})...", config.architecture);
            let output_norm = gguf
                .get_tensor_f32("output_norm.weight")
                .ok_or_else(|| "Missing output_norm.weight tensor".to_string())?;
            let (hybrid, kv) = if config.architecture == ModelArchitecture::Qwen35 {
                crate::hybrid::load_qwen35(gguf, &config, num_threads)?
            } else {
                crate::hybrid::load_nemotron_h_moe(gguf, &config, num_threads)?
            };
            let model = QuintritModel {
                config: config.clone(),
                token_embeddings: Vec::new(),
                blocks: Vec::new(),
                output_norm,
                lm_head: None,
                lm_head_f16: None,
                embedding_layer: None,
                hybrid: Some(hybrid),
            };
            (model, kv)
        } else {
            // 3. Load Token Embeddings (F16, or Q2_0 as a block-scaled layer whose rows are dequantized per token)
            progress!("==> Loading token embeddings...");
            let (token_embeddings, embedding_layer) = match gguf.tensors.get("token_embd.weight").map(|t| t.tensor_type) {
                Some(quintrit_format::GGML_TYPE_Q2_0) => (
                    Vec::new(),
                    Some(load_projection(gguf, "token_embd.weight", config.vocab_size, config.hidden_size, target_kernel, num_threads)?),
                ),
                _ => (
                    gguf.get_tensor_f16("token_embd.weight")
                        .ok_or_else(|| "Missing token_embd.weight tensor".to_string())?
                        .to_vec(),
                    None,
                ),
            };

            progress!("    [{:.2?}] token embeddings loaded", t0.elapsed());

            // 4. Load Output RMSNorm
            progress!("==> Loading output RMSNorm...");
            let output_norm = gguf
                .get_tensor_f32("output_norm.weight")
                .ok_or_else(|| "Missing output_norm.weight tensor".to_string())?;

            // 5. Load and Transcode Transformer Blocks into host RAM
            progress!(
                "==> Loading & Transcoding {} decoder layers into RAM (Kernel: {:?}, Threads: {})...",
                config.num_hidden_layers, target_kernel, num_threads
            );
            let mut blocks = Vec::with_capacity(config.num_hidden_layers);
            let q_dim = config.num_attention_heads * config.head_dim;
            let kv_dim = config.num_key_value_heads * config.head_dim;

            for l in 0..config.num_hidden_layers {
                let norm = |name: &str| -> Result<Vec<f32>, String> {
                    gguf.get_tensor_f32(&format!("blk.{l}.{name}.weight")).ok_or_else(|| format!("Missing blk.{l}.{name}.weight"))
                };
                let attn_norm = norm("attn_norm")?;
                let ffn_norm = norm("ffn_norm")?;
                let attn_sub_norm = if config.has_sub_norm { gguf.get_tensor_f32(&format!("blk.{l}.attn_sub_norm.weight")) } else { None };
                let ffn_sub_norm = if config.has_sub_norm { gguf.get_tensor_f32(&format!("blk.{l}.ffn_sub_norm.weight")) } else { None };

                let proj = |name: &str, rows: usize, cols: usize| {
                    load_projection(gguf, &format!("blk.{l}.{name}.weight"), rows, cols, target_kernel, num_threads)
                };
                let q_proj = proj("attn_q", q_dim, config.hidden_size)?;
                let k_proj = proj("attn_k", kv_dim, config.hidden_size)?;
                let v_proj = proj("attn_v", kv_dim, config.hidden_size)?;
                let o_proj = proj("attn_output", config.hidden_size, q_dim)?;
                let gate_proj = proj("ffn_gate", config.intermediate_size, config.hidden_size)?;
                let up_proj = proj("ffn_up", config.intermediate_size, config.hidden_size)?;
                let down_proj = proj("ffn_down", config.hidden_size, config.intermediate_size)?;

                let mut attn_block = AttentionBlock::new(q_proj, k_proj, v_proj, o_proj, attn_sub_norm, &config);
                if config.qk_norm {
                    attn_block = attn_block.with_qk_norm(norm("attn_q_norm")?, norm("attn_k_norm")?);
                }

                blocks.push(TransformerBlock {
                    attn_norm,
                    attn: attn_block,
                    ffn_norm,
                    gate_proj,
                    up_proj,
                    ffn_sub_norm,
                    down_proj,
                    hidden_act: config.hidden_act,
                    rms_norm_eps: config.rms_norm_eps,
                    hidden_size: config.hidden_size,
                    intermediate_size: config.intermediate_size,
                });
            }

            progress!("    [{:.2?}] decoder layers transcoded", t0.elapsed());

            // The output head: an F16 table, a Q2_0 matrix, or the token embedding (tied).
            let (lm_head, lm_head_f16) = match gguf.tensors.get("output.weight").map(|t| t.tensor_type) {
                Some(quintrit_format::GGML_TYPE_Q2_0) => (
                    Some(load_projection(gguf, "output.weight", config.vocab_size, config.hidden_size, target_kernel, num_threads)?),
                    None,
                ),
                Some(_) => (None, gguf.get_tensor_f16("output.weight").map(|s| s.to_vec())),
                None => (embedding_layer.clone(), None),
            };

            let model = QuintritModel {
                config: config.clone(),
                token_embeddings,
                blocks,
                output_norm,
                lm_head,
                lm_head_f16,
                embedding_layer,
                hybrid: None,
            };

            progress!("    [{:.2?}] output head loaded", t0.elapsed());

            // 6. Preallocate KV-Cache
            progress!("==> Preallocating KV cache for {} tokens...", config.max_position_embeddings);
            let kv_cache = KVCache::try_new(
                config.num_hidden_layers,
                config.max_position_embeddings,
                config.num_key_value_heads,
                config.head_dim,
            )?;
            (model, kv_cache)
        };

        progress!("    [{:.2?}] KV cache allocated", t0.elapsed());

        // Every tensor of the file must have been used: a tensor that no loader reads is a feature this runtime does not
        // implement, and running the model without it would give wrong results.
        let unread = gguf.unread_tensors();
        if !unread.is_empty() {
            let shown: Vec<&str> = unread.iter().take(8).map(String::as_str).collect();
            return Err(format!(
                "the file has {} tensor(s) this runtime does not use: {}{}",
                unread.len(),
                shown.join(", "),
                if unread.len() > shown.len() { ", ..." } else { "" }
            ));
        }

        let sampler = Sampler::new(SamplingConfig::default());
        let scratchpad = crate::scratchpad::InferenceScratchpad::new(&config, if gpu_branch.is_some() { 1 } else { config.max_position_embeddings });
        progress!("    [{:.2?}] scratchpad allocated", t0.elapsed());
        let load_duration = t0.elapsed();
        progress!("==> Model successfully loaded in {:.2?}", load_duration);

        Ok(Self {
            model,
            tokenizer,
            kv_cache,
            sampler,
            num_threads,
            active_kernel: target_kernel,
            scratchpad,
            cached_tokens: Vec::new(),
            cached_reset_count: 0,
            last_generation: GenerationStats::default(),
            #[cfg(feature = "cuda")]
            gpu,
            #[cfg(feature = "cuda")]
            gpu_model,
        })
    }

    /// Dynamically updates the thread pool parallelism across all model layers.
    pub fn set_num_threads(&mut self, threads: usize) {
        self.num_threads = threads.max(1);
        self.model.set_num_threads(threads);
    }

    /// Generates text autoregressively with real-time token streaming and stop sequence detection.
    ///
    /// Returns an error, before anything is computed, if the prompt does not fit into the context window (the KV cache)
    /// or contains a token outside the vocabulary. Generation ends at `max_tokens`, at an end-of-sequence token, at a
    /// stop sequence or when the context window is full. The returned text is exactly the text handed to `callback`;
    /// tokens are decoded with a [`StreamDecoder`], so a multi-byte UTF-8 character is never split.
    ///
    /// The keys and values of the previous call stay in the KV cache: if the new prompt starts with the tokens that call
    /// processed (a chat that re-sends its history, for example), only the tokens after the common prefix are computed.
    /// The result is the same as with an empty cache, because every position depends only on the tokens before it. The
    /// reuse is skipped when `kv_cache` has been reset since the previous call.
    pub fn generate_streaming_with_stops<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        stop_sequences: &[&str],
        mut callback: F,
    ) -> Result<String, String> {
        self.sampler = Sampler::new(sampling.clone());
        self.last_generation = GenerationStats::default();

        let mut prompt_tokens = self.tokenizer.encode(prompt);
        if let Some(bos) = self.tokenizer.bos_token_id() {
            if prompt_tokens.first() != Some(&bos) {
                prompt_tokens.insert(0, bos);
            }
        }

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }
        self.check_prompt(&prompt_tokens)?;
        let ctx = self.kv_cache.max_seq_len();

        let mut logits = vec![0.0f32; self.model.config.vocab_size];
        let mut full_output = String::new();
        let mut pending = String::new();
        let mut decoder = StreamDecoder::new();

        let active_stops: Vec<&str> = stop_sequences
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .collect();

        // 1. Ingest the prompt tokens that are not in the KV cache yet
        let t_prompt = Instant::now();
        let reused = self.reusable_prefix(&prompt_tokens);
        for (pos, &token_id) in prompt_tokens.iter().enumerate().skip(reused) {
            self.forward_token(token_id, pos, &mut logits)?;
            self.cached_tokens.push(token_id);
        }
        let prompt_ms = t_prompt.elapsed().as_secs_f64() * 1000.0;
        let t_generation = Instant::now();
        let mut generated_tokens = 0usize;

        // 2. Generate new tokens
        let mut current_pos = prompt_tokens.len();
        let eos_id = self.tokenizer.eos_token_id();
        let is_llama_or_bitnet = matches!(
            self.model.config.architecture,
            ModelArchitecture::BitNet | ModelArchitecture::Llama
        );

        let mut hit_stop = false;
        for _step in 0..max_tokens {
            let next_token = self.sampler.sample(&logits);

            // Halt immediately on model-designated end-of-sequence or end-of-turn token IDs
            if Some(next_token) == eos_id || (is_llama_or_bitnet && next_token == 128009) {
                break;
            }

            generated_tokens += 1;
            let piece = decoder.push(&self.tokenizer, next_token);
            if filter_stops(&piece, &active_stops, &mut pending, &mut full_output, &mut callback) {
                hit_stop = true;
                break;
            }

            if current_pos >= ctx {
                break;
            }

            self.forward_token(next_token, current_pos, &mut logits)?;
            self.cached_tokens.push(next_token);
            current_pos += 1;
        }

        if !hit_stop {
            let rest = decoder.finish(&self.tokenizer);
            hit_stop = filter_stops(&rest, &active_stops, &mut pending, &mut full_output, &mut callback);
        }
        if !hit_stop && !pending.is_empty() {
            full_output.push_str(&pending);
            callback(&pending);
        }

        self.last_generation = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            reused_prompt_tokens: reused,
            generated_tokens,
            prompt_ms,
            generation_ms: t_generation.elapsed().as_secs_f64() * 1000.0,
        };
        Ok(full_output)
    }

    /// Token counts and timings of the last [`InferenceEngine::generate_streaming_with_stops`] (or
    /// [`InferenceEngine::generate_streaming`]) call; all zero before the first one and after a call that returned an
    /// error.
    pub fn last_generation_stats(&self) -> GenerationStats {
        self.last_generation
    }

    /// Prepares the KV cache for `prompt_tokens` and returns how many of them it already holds.
    ///
    /// The cached positions are those of the previous [`InferenceEngine::generate_streaming_with_stops`] call, valid
    /// only while nobody has reset the cache or written further positions into it. At least the last prompt token is
    /// always recomputed, because its logits start the generation.
    fn reusable_prefix(&mut self, prompt_tokens: &[u32]) -> usize {
        let valid = self.kv_cache.reset_count() == self.cached_reset_count
            && self.kv_cache.len() == self.cached_tokens.len();
        // A recurrent state cannot be rolled back: reuse it only when the prompt continues the cached sequence and
        // at least one new token follows.
        let valid = valid
            && (!self.kv_cache.has_recurrent()
                || (prompt_tokens.len() > self.cached_tokens.len()
                    && prompt_tokens[..self.cached_tokens.len()] == self.cached_tokens[..]));
        let common = if valid {
            self.cached_tokens
                .iter()
                .zip(prompt_tokens)
                .take_while(|(a, b)| a == b)
                .count()
                .min(prompt_tokens.len().saturating_sub(1))
        } else {
            0
        };
        if common == 0 {
            self.kv_cache.reset();
            self.cached_reset_count = self.kv_cache.reset_count();
        } else {
            self.kv_cache.truncate(common);
        }
        self.cached_tokens.truncate(common);
        common
    }

    /// Checks that `prompt_tokens` fit into the context window and that every id indexes the vocabulary, so that the
    /// forward pass cannot run past the end of the KV cache or the embedding table.
    fn check_prompt(&self, prompt_tokens: &[u32]) -> Result<(), String> {
        let ctx = self.kv_cache.max_seq_len();
        if prompt_tokens.len() > ctx {
            return Err(format!(
                "the prompt has {} tokens but the context window holds {ctx}",
                prompt_tokens.len()
            ));
        }
        let vocab = self.model.config.vocab_size;
        if let Some(bad) = prompt_tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!("token {bad} is outside the vocabulary of {vocab} tokens"));
        }
        Ok(())
    }

    /// Generates text autoregressively with real-time token streaming.
    ///
    /// See [`InferenceEngine::generate_streaming_with_stops`] for the errors.
    pub fn generate_streaming<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        callback: F,
    ) -> Result<String, String> {
        self.generate_streaming_with_stops(prompt, max_tokens, sampling, &[], callback)
    }

    /// Evaluates generation performance and produces an empirical benchmark report.
    ///
    /// Generates `num_tokens` tokens greedily, or fewer if the context window fills up first
    /// ([`BenchmarkReport::generated_tokens`] tells how many). Returns an error if the prompt does not fit into the
    /// context window or contains a token outside the vocabulary.
    pub fn benchmark(
        &mut self,
        prompt: &str,
        num_tokens: usize,
    ) -> Result<BenchmarkReport, String> {
        self.kv_cache.reset();
        self.sampler = Sampler::new(SamplingConfig::greedy());

        let mut prompt_tokens = self.tokenizer.encode(prompt);
        if let Some(bos) = self.tokenizer.bos_token_id() {
            if prompt_tokens.first() != Some(&bos) {
                prompt_tokens.insert(0, bos);
            }
        }
        if prompt_tokens.is_empty() {
            prompt_tokens.push(1);
        }
        self.check_prompt(&prompt_tokens)?;
        let ctx = self.kv_cache.max_seq_len();

        let mut logits = vec![0.0f32; self.model.config.vocab_size];

        // 1. Measure Time to First Token (TTFT)
        let t_prompt_start = Instant::now();
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            self.forward_token(token_id, pos, &mut logits)?;
        }
        let ttft = t_prompt_start.elapsed();

        // 2. Measure subsequent token generation throughput
        let mut current_pos = prompt_tokens.len();
        let t_gen_start = Instant::now();
        let mut count = 0;
        let mut generated_ids = Vec::with_capacity(num_tokens);

        for _ in 0..num_tokens {
            if current_pos >= ctx {
                break;
            }
            let next_token = self.sampler.sample(&logits);
            generated_ids.push(next_token);
            self.forward_token(next_token, current_pos, &mut logits)?;
            current_pos += 1;
            count += 1;
        }
        let gen_time = t_gen_start.elapsed();

        let ttft_ms = ttft.as_secs_f64() * 1000.0;
        let gen_ms = gen_time.as_secs_f64() * 1000.0;
        let tokens_per_sec = if count > 0 && gen_ms > 0.0 { (count as f64) / (gen_ms / 1000.0) } else { 0.0 };

        // Calculate DRAM footprint per token and kernel counts
        let mut total_weight_bytes = 0;
        let mut kernel_counts = HashMap::new();
        for block in &self.model.blocks {
            total_weight_bytes += block.attn.memory_bytes();
            total_weight_bytes += block.gate_proj.memory_bytes();
            total_weight_bytes += block.up_proj.memory_bytes();
            total_weight_bytes += block.down_proj.memory_bytes();

            for k in [
                block.attn.q_proj().kernel_type(),
                block.attn.k_proj().kernel_type(),
                block.attn.v_proj().kernel_type(),
                block.attn.o_proj().kernel_type(),
                block.gate_proj.kernel_type(),
                block.up_proj.kernel_type(),
                block.down_proj.kernel_type(),
            ] {
                let name = match k {
                    TernaryKernelType::QuintritLut => "QuintritLut",
                    TernaryKernelType::QuintritLutAVX512 => "QuintritLutAVX512",
                    #[cfg(feature = "bitnet-baseline")]
                    TernaryKernelType::BitnetMad => "BitnetMad",
                    #[cfg(feature = "bitnet-baseline")]
                    TernaryKernelType::BitnetMad512 => "BitnetMad512",
                    TernaryKernelType::QuintritBlockScaled => "QuintritBlockScaled",
                };
                *kernel_counts.entry(name.to_string()).or_insert(0) += 1;
            }
        }

        // The F16 vocabulary projection (`lm_head`) is read in full for every token as well.
        total_weight_bytes += self.model.lm_head_f16.as_ref().unwrap_or(&self.model.token_embeddings).len() * std::mem::size_of::<u16>();
        let dram_mb_per_token = (total_weight_bytes as f64) / (1024.0 * 1024.0);
        let bandwidth_gb_s = (dram_mb_per_token * tokens_per_sec) / 1024.0;
        let generated_text = self.tokenizer.decode(&generated_ids);

        Ok(BenchmarkReport {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: count,
            ttft_ms,
            generation_ms: gen_ms,
            tokens_per_sec,
            memory_bandwidth_gb_s: bandwidth_gb_s,
            dram_traffic_per_token_mb: dram_mb_per_token,
            kernel_type: self.active_kernel,
            generated_text,
            kernel_counts,
        })
    }
}

/// Finds the smallest char index `i` in `pending` such that `&pending[i..]` is a prefix of at least one stop sequence.
#[inline]
pub fn find_safe_flush_index(pending: &str, stop_sequences: &[&str]) -> usize {
    for (idx, _) in pending.char_indices() {
        let suffix = &pending[idx..];
        for stop in stop_sequences {
            if stop.starts_with(suffix) {
                return idx;
            }
        }
    }
    pending.len()
}

/// Passes `piece` through the stop-sequence filter: text that cannot be the beginning of a stop sequence goes to
/// `callback` and to `output`, the rest waits in `pending`. Returns true when a stop sequence has been found; the text
/// before it has then been handed over and `pending` is empty.
fn filter_stops<F: FnMut(&str)>(
    piece: &str,
    stops: &[&str],
    pending: &mut String,
    output: &mut String,
    callback: &mut F,
) -> bool {
    if stops.is_empty() {
        if !piece.is_empty() {
            output.push_str(piece);
            callback(piece);
        }
        return false;
    }
    pending.push_str(piece);
    if let Some(idx) = stops.iter().filter_map(|s| pending.find(s)).min() {
        if idx > 0 {
            output.push_str(&pending[..idx]);
            callback(&pending[..idx]);
        }
        pending.clear();
        return true;
    }
    let safe = find_safe_flush_index(pending, stops);
    if safe > 0 {
        output.push_str(&pending[..safe]);
        callback(&pending[..safe]);
        pending.drain(..safe);
    }
    false
}

/// Loads the projection `name` (`rows x cols`) for the requested kernel: `Q2_0` tensors become block-scaled layers
/// (one scale per 64 weights), the per-tensor-scaled ternary formats (`I2_S`, `TQ1_0`, `TQ2_0`) are transcoded for
/// `target_kernel`.
pub(crate) fn load_projection(
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
    target_kernel: TernaryKernelType,
    num_threads: usize,
) -> Result<crate::layer::LinearLayer, String> {
    let info = gguf.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
    if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows {
        return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
    }
    let data = gguf.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
    if info.tensor_type == quintrit_format::GGML_TYPE_Q2_0 {
        let block = info.block_size.ok_or_else(|| format!("tensor '{name}': unknown Q2_0 block size"))?;
        let layer = crate::layer::BlockScaledLayer::from_q2_0(data, rows, cols, block, num_threads).map_err(|e| format!("{name}: {e}"))?;
        return Ok(crate::layer::LinearLayer::BlockScaled(layer));
    }
    let scale = match gguf.get_tensor_scale(name) {
        Some(scale) => scale,
        None => match info.tensor_type {
            quintrit_format::GGML_TYPE_TQ1_0 | quintrit_format::GGML_TYPE_TQ2_0 => {
                return Err(format!(
                    "Tensor '{name}' has differing per-block scales, which a per-tensor weight scale cannot represent"
                ));
            }
            _ => 1.0,
        },
    };
    dispatch_ternary_layer(info.tensor_type, data, rows, cols, scale, target_kernel, num_threads)
}
