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
        let t0 = Instant::now();

        // 1. Detect architecture & hyperparameters from GGUF metadata, and reject a damaged or foreign file (absurd
        //    hyper-parameters, missing or wrongly sized tensors) before anything is indexed or allocated.
        let mut config = crate::validate::validate_gguf_model(gguf)?;
        if let Some(hint) = model_name_hint {
            if hint.to_ascii_lowercase().contains("falcon") {
                config.architecture = ModelArchitecture::Falcon3;
            }
        }

        let target_kernel = force_kernel.unwrap_or(TernaryKernelType::QuintritLut);

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

        let tokenizer = QuintritTokenizer::from_gguf_metadata_with_types(
            &tokens,
            merges.as_deref(),
            token_types.as_deref(),
            config.bos_token_id,
            config.eos_token_id,
        )?;

        // 3. Load Token Embeddings
        progress!("==> Loading token embeddings (F16)...");
        let embd_f16 = gguf
            .get_tensor_f16("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight tensor".to_string())?;
        let token_embeddings = embd_f16.to_vec();

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

        for l in 0..config.num_hidden_layers {
            let attn_norm = gguf
                .get_tensor_f32(&format!("blk.{}.attn_norm.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", l))?;

            let ffn_norm = gguf
                .get_tensor_f32(&format!("blk.{}.ffn_norm.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.ffn_norm.weight", l))?;

            let attn_sub_norm = if config.has_sub_norm {
                gguf.get_tensor_f32(&format!("blk.{}.attn_sub_norm.weight", l))
            } else {
                None
            };

            let ffn_sub_norm = if config.has_sub_norm {
                gguf.get_tensor_f32(&format!("blk.{}.ffn_sub_norm.weight", l))
            } else {
                None
            };

            // Attention projections
            let q_data = gguf
                .get_tensor_data(&format!("blk.{}.attn_q.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.attn_q.weight", l))?;
            let k_data = gguf
                .get_tensor_data(&format!("blk.{}.attn_k.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.attn_k.weight", l))?;
            let v_data = gguf
                .get_tensor_data(&format!("blk.{}.attn_v.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.attn_v.weight", l))?;
            let o_data = gguf
                .get_tensor_data(&format!("blk.{}.attn_output.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.attn_output.weight", l))?;

            // FFN projections
            let gate_data = gguf
                .get_tensor_data(&format!("blk.{}.ffn_gate.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.ffn_gate.weight", l))?;
            let up_data = gguf
                .get_tensor_data(&format!("blk.{}.ffn_up.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.ffn_up.weight", l))?;
            let down_data = gguf
                .get_tensor_data(&format!("blk.{}.ffn_down.weight", l))
                .ok_or_else(|| format!("Missing blk.{}.ffn_down.weight", l))?;

            let q_dim = config.num_attention_heads * config.head_dim;
            let kv_dim = config.num_key_value_heads * config.head_dim;

            let q_name = format!("blk.{}.attn_q.weight", l);
            let k_name = format!("blk.{}.attn_k.weight", l);
            let v_name = format!("blk.{}.attn_v.weight", l);
            let o_name = format!("blk.{}.attn_output.weight", l);
            let gate_name = format!("blk.{}.ffn_gate.weight", l);
            let up_name = format!("blk.{}.ffn_up.weight", l);
            let down_name = format!("blk.{}.ffn_down.weight", l);

            let tensor_scale = |name: &str| -> Result<f32, String> {
                match gguf.get_tensor_scale(name) {
                    Some(scale) => Ok(scale),
                    None => match gguf.tensors.get(name).map(|t| t.tensor_type) {
                        Some(quintrit_format::GGML_TYPE_TQ1_0) | Some(quintrit_format::GGML_TYPE_TQ2_0) => Err(format!(
                            "Tensor '{}' has differing per-block scales, which a per-tensor weight scale cannot represent",
                            name
                        )),
                        _ => Ok(1.0),
                    },
                }
            };
            let q_scale = tensor_scale(&q_name)?;
            let k_scale = tensor_scale(&k_name)?;
            let v_scale = tensor_scale(&v_name)?;
            let o_scale = tensor_scale(&o_name)?;
            let gate_scale = tensor_scale(&gate_name)?;
            let up_scale = tensor_scale(&up_name)?;
            let down_scale = tensor_scale(&down_name)?;

            let q_type = gguf.tensors.get(&q_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let k_type = gguf.tensors.get(&k_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let v_type = gguf.tensors.get(&v_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let o_type = gguf.tensors.get(&o_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let gate_type = gguf.tensors.get(&gate_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let up_type = gguf.tensors.get(&up_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);
            let down_type = gguf.tensors.get(&down_name).map(|t| t.tensor_type).unwrap_or(quintrit_format::GGML_TYPE_I2_S);

            let q_proj = dispatch_ternary_layer(q_type, q_data, q_dim, config.hidden_size, q_scale, target_kernel, num_threads)?;
            let k_proj = dispatch_ternary_layer(k_type, k_data, kv_dim, config.hidden_size, k_scale, target_kernel, num_threads)?;
            let v_proj = dispatch_ternary_layer(v_type, v_data, kv_dim, config.hidden_size, v_scale, target_kernel, num_threads)?;
            let o_proj = dispatch_ternary_layer(o_type, o_data, config.hidden_size, q_dim, o_scale, target_kernel, num_threads)?;

            let gate_proj = dispatch_ternary_layer(gate_type, gate_data, config.intermediate_size, config.hidden_size, gate_scale, target_kernel, num_threads)?;
            let up_proj = dispatch_ternary_layer(up_type, up_data, config.intermediate_size, config.hidden_size, up_scale, target_kernel, num_threads)?;
            let down_proj = dispatch_ternary_layer(down_type, down_data, config.hidden_size, config.intermediate_size, down_scale, target_kernel, num_threads)?;

            let attn_block = AttentionBlock::new(q_proj, k_proj, v_proj, o_proj, attn_sub_norm, &config);

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

        let lm_head_f16 = gguf.get_tensor_f16("output.weight").map(|s| s.to_vec());

        let model = QuintritModel {
            config: config.clone(),
            token_embeddings,
            blocks,
            output_norm,
            lm_head: None,
            lm_head_f16,
        };

        // 6. Preallocate KV-Cache
        progress!("==> Preallocating KV cache for {} tokens...", config.max_position_embeddings);
        let kv_cache = KVCache::try_new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.head_dim,
        )?;

        let sampler = Sampler::new(SamplingConfig::default());
        let scratchpad = crate::scratchpad::InferenceScratchpad::new(&config, config.max_position_embeddings);
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
            self.model.forward_token_with_scratchpad(token_id, pos, &mut self.kv_cache, &mut self.scratchpad, &mut logits);
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

            self.model.forward_token_with_scratchpad(next_token, current_pos, &mut self.kv_cache, &mut self.scratchpad, &mut logits);
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
            self.model.forward_token_with_scratchpad(token_id, pos, &mut self.kv_cache, &mut self.scratchpad, &mut logits);
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
            self.model.forward_token_with_scratchpad(next_token, current_pos, &mut self.kv_cache, &mut self.scratchpad, &mut logits);
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
