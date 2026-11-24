//! Grouped-Query Attention (GQA) mechanism with causal KV cache support.

use crate::config::{ModelConfig, Numerics, RopeConfig};
use crate::kv_cache::KVCache;
use crate::layer::LinearLayer;
use crate::dispatch::multiversion;
use crate::ops::{
    apply_rope, f32_to_f16_slice, ggml_vec_dot_f16, ggml_vec_dot_f16_f32, ggml_vec_mad_f16, ggml_vec_mad_f16_f32,
    ggml_vec_scale_f16, quantize_activations_int8_with_range, quantize_q8_0, rms_norm,
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
    /// Input (model) dimension of the Q, K and V projections.
    hidden_size: usize,
    rope: RopeConfig,
    rms_norm_eps: f32,
    scale_factor: f32,
    /// RMSNorm weights applied to every query head before RoPE (`attn_q_norm`, `head_dim` values).
    q_norm: Option<Vec<f32>>,
    /// RMSNorm weights applied to every key head before RoPE (`attn_k_norm`, `head_dim` values).
    k_norm: Option<Vec<f32>>,
    /// Accumulate the attention values in FP32 (llama.cpp) instead of FP16 (bitnet.cpp).
    f32_values: bool,
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
            hidden_size: config.hidden_size,
            // `rope_theta` and `head_dim` stay authoritative for the base frequency and the largest rotation width.
            rope: RopeConfig {
                freq_base: config.rope_theta,
                n_dims: config.rope.n_dims.min(config.head_dim),
                ..config.rope
            },
            rms_norm_eps: config.rms_norm_eps,
            scale_factor,
            q_norm: None,
            k_norm: None,
            f32_values: config.numerics == Numerics::LlamaCpp,
        }
    }

    /// Adds the per-head RMSNorm of the queries and keys (Qwen3), applied before RoPE.
    pub fn with_qk_norm(mut self, q_norm: Vec<f32>, k_norm: Vec<f32>) -> Self {
        assert_eq!(q_norm.len(), self.head_dim, "attn_q_norm length");
        assert_eq!(k_norm.len(), self.head_dim, "attn_k_norm length");
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
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
        if self.q_proj.is_block_scaled() {
            self.forward_block_scaled(layer_idx, pos, kv_cache, scratchpad);
            return;
        }
        let hidden_size = self.hidden_size;
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
        apply_rope(
            &mut scratchpad.q[..q_dim],
            &mut scratchpad.k[..kv_dim],
            pos,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &self.rope,
        );

        // 4. Append K and V into contiguous KV cache (FP16 precision)
        f32_to_f16_slice(&scratchpad.k[..kv_dim], &mut scratchpad.k_f16[..kv_dim]);
        f32_to_f16_slice(&scratchpad.v[..kv_dim], &mut scratchpad.v_f16[..kv_dim]);
        kv_cache.append(layer_idx, pos, &scratchpad.k_f16[..kv_dim], &scratchpad.v_f16[..kv_dim]);

        // 5. Compute Grouped-Query Attention (GQA) matching BitNet ggml_flash_attn_ext
        let seq_len = pos + 1;
        let scale_factor = self.scale_factor;

        self.attend_all_heads(layer_idx, seq_len, scale_factor, kv_cache, scratchpad);

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


    /// Attention of every query head over the first `seq_len` cached positions; writes the heads side by side into `scratchpad.attn_out`
    /// (see [`attend_heads`]).
    fn attend_all_heads(&self, layer_idx: usize, seq_len: usize, scale: f32, kv_cache: &KVCache, scratchpad: &mut crate::scratchpad::InferenceScratchpad) {
        let q_dim = self.num_q_heads * self.head_dim;
        attend_heads(
            &HeadsLayout { head_dim: self.head_dim, gqa_ratio: self.gqa_ratio, f32_values: self.f32_values, layer: layer_idx, seq_len, scale },
            &scratchpad.q[..q_dim],
            &mut scratchpad.q_slice_f16[..q_dim],
            &mut scratchpad.head_out_f16[..q_dim],
            &mut scratchpad.head_acc[..q_dim],
            kv_cache,
            &mut scratchpad.attn_out[..q_dim],
            self.q_proj.num_threads(),
        );
    }

    /// Forward pass for block-scaled (`Q2_0`) projections: the inputs of the projections are quantized to `Q8_0`, as
    /// llama.cpp does before its `Q2_0 x Q8_0` dot products.
    fn forward_block_scaled(
        &self,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut KVCache,
        scratchpad: &mut crate::scratchpad::InferenceScratchpad,
    ) {
        let hidden_size = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let nb = hidden_size / 32;

        quantize_q8_0(&scratchpad.x_norm[..hidden_size], &mut scratchpad.q8_qs[..hidden_size], &mut scratchpad.q8_d[..nb]);
        self.q_proj.forward_q8(&scratchpad.q8_qs[..hidden_size], &scratchpad.q8_d[..nb], &mut scratchpad.q[..q_dim]);
        self.k_proj.forward_q8(&scratchpad.q8_qs[..hidden_size], &scratchpad.q8_d[..nb], &mut scratchpad.k[..kv_dim]);
        self.v_proj.forward_q8(&scratchpad.q8_qs[..hidden_size], &scratchpad.q8_d[..nb], &mut scratchpad.v[..kv_dim]);

        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            let hd = self.head_dim;
            for h in 0..self.num_q_heads {
                let head = &mut scratchpad.q[h * hd..(h + 1) * hd];
                scratchpad.head_acc[..hd].copy_from_slice(head);
                rms_norm(&scratchpad.head_acc[..hd], qn, self.rms_norm_eps, head);
            }
            for h in 0..self.num_kv_heads {
                let head = &mut scratchpad.k[h * hd..(h + 1) * hd];
                scratchpad.head_acc[..hd].copy_from_slice(head);
                rms_norm(&scratchpad.head_acc[..hd], kn, self.rms_norm_eps, head);
            }
        }

        apply_rope(
            &mut scratchpad.q[..q_dim],
            &mut scratchpad.k[..kv_dim],
            pos,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &self.rope,
        );

        f32_to_f16_slice(&scratchpad.k[..kv_dim], &mut scratchpad.k_f16[..kv_dim]);
        f32_to_f16_slice(&scratchpad.v[..kv_dim], &mut scratchpad.v_f16[..kv_dim]);
        kv_cache.append(layer_idx, pos, &scratchpad.k_f16[..kv_dim], &scratchpad.v_f16[..kv_dim]);

        let seq_len = pos + 1;
        self.attend_all_heads(layer_idx, seq_len, self.scale_factor, kv_cache, scratchpad);

        let o_in = if let Some(sub_norm) = &self.attn_sub_norm {
            rms_norm(&scratchpad.attn_out[..q_dim], sub_norm, self.rms_norm_eps, &mut scratchpad.attn_sub_buf[..q_dim]);
            &scratchpad.attn_sub_buf[..q_dim]
        } else {
            &scratchpad.attn_out[..q_dim]
        };
        let nq = q_dim / 32;
        quantize_q8_0(o_in, &mut scratchpad.q8_qs[..q_dim], &mut scratchpad.q8_d[..nq]);
        self.o_proj.forward_q8(&scratchpad.q8_qs[..q_dim], &scratchpad.q8_d[..nq], &mut scratchpad.attn_out[..hidden_size]);
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

/// The attention heads of a token run on the worker pool once `heads x cached positions` reaches this value; below it the fork-join costs more than the
/// serial loop. The serial loop is about eight times slower per position on the Neoverse-N2 than on the Zen 4 EPYC, while the fork costs about the same
/// (15 to 20 us), so the break-even differs: measured with `benches/attention_threshold.rs` (both value numerics, 4 and 8 threads, 8 to 32 heads) it
/// is at 8 to 24 positions of a 20-head layer on the N2 and at 100 to 200 on the EPYC. The values are set a little above the lower end of those ranges.
#[cfg(target_arch = "aarch64")]
pub const PARALLEL_ATTENTION_MIN_WORK: usize = 256;
/// See the AArch64 value.
#[cfg(not(target_arch = "aarch64"))]
pub const PARALLEL_ATTENTION_MIN_WORK: usize = 2048;
/// What [`attend_heads`] needs to know about the heads of one attention layer and the position.
#[derive(Debug, Clone, Copy)]
pub struct HeadsLayout {
    /// Values per head.
    pub head_dim: usize,
    /// Query heads per key/value head.
    pub gqa_ratio: usize,
    /// llama.cpp arithmetic (F32 value accumulator) instead of bitnet.cpp's (F16).
    pub f32_values: bool,
    /// Layer slot in the KV cache.
    pub layer: usize,
    /// Number of cached positions to attend to (the position of the token plus one).
    pub seq_len: usize,
    /// Score scale, `1 / sqrt(head_dim)` unless the model says otherwise.
    pub scale: f32,
}

/// Attention of all query heads of one layer for the newest token: `q` holds the heads side by side (`n_heads * head_dim` values), `out` receives
/// the heads in the same layout. `q16`, `vkq16` and `acc` are scratch lanes of the same length (one `head_dim` lane per head).
///
/// The heads are independent (each reads the shared KV cache and writes only its own lanes), so on a pool of more than one thread they run in
/// parallel once `heads x positions` reaches [`PARALLEL_ATTENTION_MIN_WORK`]; the result does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
pub fn attend_heads(
    layout: &HeadsLayout,
    q: &[f32],
    q16: &mut [u16],
    vkq16: &mut [u16],
    acc: &mut [f32],
    kv_cache: &KVCache,
    out: &mut [f32],
    threads: usize,
) {
    let heads = q.len() / layout.head_dim;
    let parallel = threads > 1 && heads > 1 && heads * layout.seq_len >= PARALLEL_ATTENTION_MIN_WORK;
    attend_heads_with(layout, q, q16, vkq16, acc, kv_cache, out, threads, parallel);
}

/// [`attend_heads`] with the serial/parallel choice made by the caller (the benchmark that sets the threshold forces both).
#[allow(clippy::too_many_arguments)]
pub fn attend_heads_with(
    layout: &HeadsLayout,
    q: &[f32],
    q16: &mut [u16],
    vkq16: &mut [u16],
    acc: &mut [f32],
    kv_cache: &KVCache,
    out: &mut [f32],
    threads: usize,
    parallel: bool,
) {
    use rayon::prelude::*;
    let hd = layout.head_dim;
    f32_to_f16_slice(q, q16);
    let q16 = &*q16;
    let one_head = |h: usize, vkq16: &mut [u16], acc: &mut [f32], out: &mut [f32]| {
        attend_head(
            &q16[h * hd..(h + 1) * hd],
            &q[h * hd..(h + 1) * hd],
            kv_cache,
            layout.layer,
            h / layout.gqa_ratio,
            layout.seq_len,
            layout.scale,
            vkq16,
            acc,
            out,
            layout.f32_values,
        );
    };
    let vkq = vkq16.chunks_mut(hd);
    let acc = acc.chunks_mut(hd);
    let out = out.chunks_mut(hd);
    if parallel {
        let (vkq, acc, out): (Vec<_>, Vec<_>, Vec<_>) = (vkq.collect(), acc.collect(), out.collect());
        quintrit_core::threading::get_thread_pool(threads).install(|| {
            vkq.into_par_iter().zip(acc).zip(out).enumerate().for_each(|(h, ((vkq16, acc), out))| one_head(h, vkq16, acc, out));
        });
    } else {
        for (h, ((vkq16, acc), out)) in vkq.zip(acc).zip(out).enumerate() {
            one_head(h, vkq16, acc, out);
        }
    }
}

/// Number of worker threads of the reference runtime whose split-KV attention reduction [`attend_head`] reproduces
/// (the parity references are captured with `-t 4`).
pub const REFERENCE_ATTENTION_THREADS: usize = 4;
/// The reference runtime pads the number of KV cells it attends to (`n_kv`) to a multiple of this.
pub const KV_PADDING: usize = 256;
/// From this padded KV length on the reference splits the KV axis between its threads.
pub const SPLIT_KV_MIN_LEN: usize = 512;

/// Online softmax over the cached positions `range` of one head with an FP32 value accumulator, identical to
/// upstream llama.cpp's `ggml_compute_forward_flash_attn_ext_f16_one_chunk`: the scores are
/// `ggml_vec_dot_f16_f32(K, Q)` with the query in F32, the values are accumulated with `ggml_vec_scale_f32` and
/// `ggml_vec_mad_f16_f32`. Returns `(M, S)`; the accumulator is left in `vkq32`. Inlined into both versions of
/// [`attend_head`].
#[inline(always)]
fn attend_range_f32(
    q_f32: &[f32],
    kv_cache: &KVCache,
    layer: usize,
    kv_head: usize,
    range: std::ops::Range<usize>,
    scale: f32,
    vkq32: &mut [f32],
) -> (f32, f32) {
    let mut s_sum: f32 = 0.0;
    let mut max_score = f32::NEG_INFINITY;
    vkq32.fill(0.0);
    for p in range {
        let k_past = kv_cache.get_key(layer, p, kv_head);
        let score = ggml_vec_dot_f16_f32(k_past, q_f32) * scale;

        let m_old = max_score;
        let mut ms = 1.0f32;
        let mut vs = 1.0f32;
        if score > max_score {
            max_score = score;
            ms = (m_old - max_score).exp();
            for v in vkq32.iter_mut() {
                *v *= ms;
            }
        } else {
            vs = (score - max_score).exp();
        }
        ggml_vec_mad_f16_f32(vkq32, kv_cache.get_val(layer, p, kv_head), vs);
        s_sum = s_sum.mul_add(ms, vs);
    }
    (max_score, s_sum)
}

/// Online softmax over the cached positions `range` of one head with an FP16 value accumulator, identical to ggml's
/// `ggml_compute_forward_flash_attn_ext_f16_one_chunk`. Returns `(M, S)`; the accumulator is left in `vkq16`. Inlined
/// into both versions of [`attend_head`].
#[inline(always)]
fn attend_range(
    q_f16: &[u16],
    kv_cache: &KVCache,
    layer: usize,
    kv_head: usize,
    range: std::ops::Range<usize>,
    scale: f32,
    vkq16: &mut [u16],
) -> (f32, f32) {
    let mut s_sum: f32 = 0.0;
    let mut max_score = f32::NEG_INFINITY;
    vkq16.fill(0);
    for p in range {
        let k_past = kv_cache.get_key(layer, p, kv_head);
        let score = ggml_vec_dot_f16(q_f16, k_past) * scale;

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
        ggml_vec_mad_f16(vkq16, kv_cache.get_val(layer, p, kv_head), vs);
        // `S = S*ms + vs` is contracted into an FMA in the reference build.
        s_sum = s_sum.mul_add(ms, vs);
    }
    (max_score, s_sum)
}

multiversion! {
    /// Attention of one query head (already converted to FP16) over the first `seq_len` cached positions, written to
    /// `out` (`head_dim` values).
    ///
    /// It reproduces the reference runtime's flash attention bit for bit, including the KV padding: while the padded KV
    /// length is below [`SPLIT_KV_MIN_LEN`] one online softmax runs over all positions; from there on the padded KV axis
    /// is split into [`REFERENCE_ATTENTION_THREADS`] equal chunks with separate FP16 accumulators whose partial results
    /// are combined in chunk order (`ggml_flash_attn_ext_reduce_partials`). The result does not depend on this runtime's
    /// own thread count.
    ///
    /// `q_f16` is the query rounded to FP16 (bitnet.cpp converts it before the dot products), `q_f32` the query itself
    /// (llama.cpp keeps it in FP32); `f32_values` selects the llama.cpp arithmetic.
    ///
    /// The whole function is compiled twice (module `dispatch`): the scalar parts of the online softmax run with
    /// fused multiply-adds on x86-64 CPUs that have them, whatever flags the crate was built with.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_head(
        q_f16: &[u16],
        q_f32: &[f32],
        kv_cache: &KVCache,
        layer: usize,
        kv_head: usize,
        seq_len: usize,
        scale: f32,
        vkq16: &mut [u16],
        acc: &mut [f32],
        out: &mut [f32],
        f32_values: bool,
    ) {
        let n_kv = kv_cache.max_seq_len().min(seq_len.next_multiple_of(KV_PADDING).max(KV_PADDING));
        if f32_values {
            attend_head_f32(q_f32, kv_cache, layer, kv_head, seq_len, scale, n_kv, acc, out);
            return;
        }
        if n_kv < SPLIT_KV_MIN_LEN {
            let (_, s_sum) = attend_range(q_f16, kv_cache, layer, kv_head, 0..seq_len, scale, vkq16);
            let s_inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
            for (o, &v) in out.iter_mut().zip(vkq16.iter()) {
                *o = H::to_f32(v) * s_inv;
            }
            return;
        }

        let chunk = n_kv.div_ceil(REFERENCE_ATTENTION_THREADS);
        let mut m_final = f32::NEG_INFINITY;
        let mut s_final = 0.0f32;
        acc.fill(0.0);
        for c in 0..REFERENCE_ATTENTION_THREADS {
            let start = c * chunk;
            let end = (start + chunk).min(n_kv).min(seq_len);
            if start >= end {
                continue; // a chunk of masked cells only: S = 0, skipped by the reduction
            }
            let (m_chunk, s_chunk) = attend_range(q_f16, kv_cache, layer, kv_head, start..end, scale, vkq16);
            if s_chunk == 0.0 {
                continue;
            }
            let m_new = m_final.max(m_chunk);
            let scale_old = (m_final - m_new).exp();
            let scale_new = (m_chunk - m_new).exp();
            for (a, &v) in acc.iter_mut().zip(vkq16.iter()) {
                // `VKQ_final*scale_old + VKQ_chunk*scale_new`, contracted into one FMA.
                *a = a.mul_add(scale_old, H::to_f32(v) * scale_new);
            }
            s_final = s_final.mul_add(scale_old, s_chunk * scale_new);
            m_final = m_new;
        }
        if s_final != 0.0 {
            let s_inv = 1.0 / s_final;
            for (o, &a) in out.iter_mut().zip(acc.iter()) {
                *o = a * s_inv;
            }
        } else {
            out.copy_from_slice(acc);
        }
    }
}

/// [`attend_head`] with FP32 value accumulators (upstream llama.cpp). `out` doubles as the per-chunk accumulator.
/// Inlined into both versions of [`attend_head`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attend_head_f32(
    q_f32: &[f32],
    kv_cache: &KVCache,
    layer: usize,
    kv_head: usize,
    seq_len: usize,
    scale: f32,
    n_kv: usize,
    acc: &mut [f32],
    out: &mut [f32],
) {
    if n_kv < SPLIT_KV_MIN_LEN {
        let (_, s_sum) = attend_range_f32(q_f32, kv_cache, layer, kv_head, 0..seq_len, scale, out);
        let s_inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
        for o in out.iter_mut() {
            *o *= s_inv;
        }
        return;
    }
    let chunk = n_kv.div_ceil(REFERENCE_ATTENTION_THREADS);
    let mut m_final = f32::NEG_INFINITY;
    let mut s_final = 0.0f32;
    acc.fill(0.0);
    for c in 0..REFERENCE_ATTENTION_THREADS {
        let start = c * chunk;
        let end = (start + chunk).min(n_kv).min(seq_len);
        if start >= end {
            continue;
        }
        let (m_chunk, s_chunk) = attend_range_f32(q_f32, kv_cache, layer, kv_head, start..end, scale, out);
        if s_chunk == 0.0 {
            continue;
        }
        let m_new = m_final.max(m_chunk);
        let scale_old = (m_final - m_new).exp();
        let scale_new = (m_chunk - m_new).exp();
        for (a, &v) in acc.iter_mut().zip(out.iter()) {
            *a = a.mul_add(scale_old, v * scale_new);
        }
        s_final = s_final.mul_add(scale_old, s_chunk * scale_new);
        m_final = m_new;
    }
    if s_final != 0.0 {
        let s_inv = 1.0 / s_final;
        for (o, &a) in out.iter_mut().zip(acc.iter()) {
            *o = a * s_inv;
        }
    } else {
        out.copy_from_slice(acc);
    }
}
