//! Grouped-Query Attention (GQA) mechanism with causal KV cache support.

use crate::config::ModelConfig;
use crate::kv_cache::KVCache;
use crate::layer::LinearLayer;
use crate::ops::{
    apply_rotary_emb, f16_to_f32, f32_to_f16, ggml_vec_dot_f16, ggml_vec_mad_f16,
    ggml_vec_scale_f16, quantize_activations_int8_with_range, rms_norm,
};

/// Self-attention block implementing Grouped-Query Attention (GQA).
pub struct AttentionBlock {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    attn_sub_norm: Option<Vec<f32>>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    gqa_ratio: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    scale_factor: f32,
}

impl AttentionBlock {
    /// Constructs an AttentionBlock from projection layers and configuration.
    pub fn new(
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
        o_proj: LinearLayer,
        attn_sub_norm: Option<Vec<f32>>,
        config: &ModelConfig,
    ) -> Self {
        let gqa_ratio = config.gqa_ratio();
        let scale_factor = 1.0 / (config.head_dim as f32).sqrt();

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_sub_norm,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            gqa_ratio,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
            scale_factor,
        }
    }

    /// Updates execution thread count for all attention projection layers.
    pub fn set_num_threads(&mut self, threads: usize) {
        self.q_proj.set_num_threads(threads);
        self.k_proj.set_num_threads(threads);
        self.v_proj.set_num_threads(threads);
        self.o_proj.set_num_threads(threads);
    }

    /// Evaluates the complete self-attention forward pass for a single token at position `pos`.
    pub fn forward(
        &self,
        layer_idx: usize,
        x_norm: &[f32],
        pos: usize,
        kv_cache: &mut KVCache,
        out: &mut [f32],
    ) {
        let hidden_size = x_norm.len();
        let mut scratchpad = crate::scratchpad::InferenceScratchpad::from_dims(
            hidden_size,
            hidden_size * 4,
            1,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            pos + 1,
        );
        scratchpad.x_norm[..hidden_size].copy_from_slice(x_norm);
        self.forward_with_scratchpad(layer_idx, pos, kv_cache, &mut scratchpad);
        out.copy_from_slice(&scratchpad.attn_out[..hidden_size]);
    }

    /// Evaluates self-attention forward pass with zero heap allocations using pre-allocated scratchpad.
    pub fn forward_with_scratchpad(
        &self,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut KVCache,
        scratchpad: &mut crate::scratchpad::InferenceScratchpad,
    ) {
        let hidden_size = self.num_q_heads * self.head_dim;
        let q_dim = self.num_q_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // 1. Quantize activation input to 8-bit for ternary GEMV
        let act_scale = quantize_activations_int8_with_range(
            &scratchpad.x_norm[..hidden_size],
            &mut scratchpad.x_int8[..hidden_size],
            self.q_proj.kernel_type().activation_quant_levels(),
        );


        // 1b. Precompute QLUT once into scratchpad.qlut_buf if using 1.60 bpw velurex encoding
        #[cfg(target_arch = "x86_64")]
        let avx512_table = matches!(self.q_proj.kernel_type(), crate::layer::TernaryKernelType::QuintritLutAVX512);
        #[cfg(not(target_arch = "x86_64"))]
        let avx512_table = false;
        let qlut_len = if avx512_table {
            #[cfg(target_arch = "x86_64")]
            {
                quintrit_core::lut::build_activation_lut_wide_avx512_into(
                    &scratchpad.x_int8[..hidden_size],
                    hidden_size,
                    &mut scratchpad.qlut_buf,
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                0
            }
        } else if matches!(self.q_proj.kernel_type(), crate::layer::TernaryKernelType::QuintritLut) {
            quintrit_core::lut::build_activation_lut_wide_into(
                &scratchpad.x_int8[..hidden_size],
                hidden_size,
                &mut scratchpad.act_split,
                &mut scratchpad.qlut_buf,
            )
        } else {
            0
        };
        let qlut_ref = if qlut_len > 0 {
            Some(&scratchpad.qlut_buf[..qlut_len])
        } else {
            None
        };

        // 2. Project Q, K, V concurrently
        let num_threads = self.q_proj.num_threads();
        crate::layer::forward_linear_triple(
            &self.q_proj,
            &mut scratchpad.q[..q_dim],
            &self.k_proj,
            &mut scratchpad.k[..kv_dim],
            &self.v_proj,
            &mut scratchpad.v[..kv_dim],
            &scratchpad.x_int8[..hidden_size],
            act_scale,
            qlut_ref,
            num_threads,
        );

        // 3. Apply RoPE rotary positional embeddings to Q and K
        apply_rotary_emb(
            &mut scratchpad.q[..q_dim],
            &mut scratchpad.k[..kv_dim],
            pos,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
        );

        // 4. Append K and V into contiguous KV cache (FP16 precision)
        for i in 0..kv_dim {
            scratchpad.k_f16[i] = f32_to_f16(scratchpad.k[i]);
            scratchpad.v_f16[i] = f32_to_f16(scratchpad.v[i]);
        }
        kv_cache.append(layer_idx, pos, &scratchpad.k_f16[..kv_dim], &scratchpad.v_f16[..kv_dim]);

        // 5. Compute Grouped-Query Attention (GQA) matching BitNet ggml_flash_attn_ext
        let seq_len = pos + 1;
        let scale_factor = self.scale_factor;

        for h in 0..self.num_q_heads {
            let kv_h = h / self.gqa_ratio;
            let q_slice = &scratchpad.q[h * self.head_dim..(h + 1) * self.head_dim];

            for d in 0..self.head_dim {
                scratchpad.q_slice_f16[d] = f32_to_f16(q_slice[d]);
            }

            // Online softmax with FP16 VKQ accumulator, identical to the ggml flash-attention
            // kernel (ggml_compute_forward_flash_attn_ext_f16_one_chunk).
            let mut s_sum: f32 = 0.0;
            let mut max_score = f32::NEG_INFINITY;
            let vkq16 = &mut scratchpad.head_out_f16[..self.head_dim];
            vkq16.fill(0);
            for p in 0..seq_len {
                let k_past = kv_cache.get_key(layer_idx, p, kv_h);
                let dot = ggml_vec_dot_f16(&scratchpad.q_slice_f16[..self.head_dim], k_past);
                let score = dot * scale_factor;

                let m_old = max_score;
                let mut ms = 1.0f32;
                let mut vs = 1.0f32;
                if score > max_score {
                    max_score = score;
                    ms = (m_old - max_score).exp();
                    ggml_vec_scale_f16(vkq16, ms);
                } else {
                    vs = (score - max_score).exp();
                }
                let v_past = kv_cache.get_val(layer_idx, p, kv_h);
                ggml_vec_mad_f16(vkq16, v_past, vs);
                // `S = S*ms + vs` is contracted into an FMA in the reference build.
                s_sum = s_sum.mul_add(ms, vs);
            }

            let s_inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
            let head_out = &mut scratchpad.attn_out[h * self.head_dim..(h + 1) * self.head_dim];
            for d in 0..self.head_dim {
                head_out[d] = f16_to_f32(vkq16[d]) * s_inv;
            }
        }

        // 6. Sub-layer normalization (Sub-LN) if specified by architecture (e.g. BitNet)
        let o_act_scale = if let Some(sub_norm_weights) = &self.attn_sub_norm {
            rms_norm(
                &scratchpad.attn_out[..q_dim],
                sub_norm_weights,
                self.rms_norm_eps,
                &mut scratchpad.attn_sub_buf[..q_dim],
            );
            quantize_activations_int8_with_range(
                &scratchpad.attn_sub_buf[..q_dim],
                &mut scratchpad.o_int8[..q_dim],
                self.o_proj.kernel_type().activation_quant_levels(),
            )
        } else {
            quantize_activations_int8_with_range(
                &scratchpad.attn_out[..q_dim],
                &mut scratchpad.o_int8[..q_dim],
                self.o_proj.kernel_type().activation_quant_levels(),
            )
        };

        // 7. Output projection: W_o * attn_normed -> writes into scratchpad.attn_out
        self.o_proj.forward(
            &scratchpad.o_int8[..q_dim],
            o_act_scale,
            &mut scratchpad.attn_out[..hidden_size],
        );
    }


    /// Total memory footprint of all attention weight matrices in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.q_proj.memory_bytes()
            + self.k_proj.memory_bytes()
            + self.v_proj.memory_bytes()
            + self.o_proj.memory_bytes()
    }

    /// Number of query heads.
    pub fn num_q_heads(&self) -> usize {
        self.num_q_heads
    }

    /// Number of key/value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Accessor for Query projection layer.
    pub fn q_proj(&self) -> &LinearLayer {
        &self.q_proj
    }

    /// Accessor for Key projection layer.
    pub fn k_proj(&self) -> &LinearLayer {
        &self.k_proj
    }

    /// Accessor for Value projection layer.
    pub fn v_proj(&self) -> &LinearLayer {
        &self.v_proj
    }

    /// Accessor for Output projection layer.
    pub fn o_proj(&self) -> &LinearLayer {
        &self.o_proj
    }
}
