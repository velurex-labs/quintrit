//! Hybrid decoders whose layers are not all attention layers: Qwen3.5 (`qwen35`), where most layers are Gated DeltaNet
//! linear-attention layers with a recurrent state and every `full_attention_interval`-th layer is a gated full-attention
//! layer.
//!
//! Every operation reproduces upstream llama.cpp's CPU arithmetic (AVX2 code paths) bit for bit, so that greedy
//! decoding gives the same tokens as llama.cpp: `Q2_0` projections with `Q8_0` inputs, `ggml_vec_dot_f32` for the F32
//! projections and the recurrence, `ggml_vec_dot_bf16` for BF16 projections, the normalized Walsh-Hadamard transform
//! of the PrismML Hadamard-folded weights (`prism.hadamard.*`), and flash attention with FP32 value accumulators.

use crate::attention::{attend_heads, HeadsLayout};
use crate::config::{ModelConfig, RopeConfig};
use crate::dispatch::multiversion;
use crate::kv_cache::KVCache;
use crate::layer::LinearLayer;
use crate::ops::{
    add_residual, apply_rope, f16_to_f32, f32_to_bf16, f32_to_f16_slice, fwht_blocks, ggml_vec_dot_bf16,
    ggml_vec_dot_f32, quantize_q8_0, rms_norm, sigmoid_f32, silu, softplus_f32, swiglu,
};
use quintrit_format::{GgufFile, GgufMetadataValue, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q2_0, GGML_TYPE_Q8_0};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Hadamard-folded inputs (PrismML `prism.hadamard.*`)
// ============================================================================

/// The input transform of a Hadamard-folded weight `W_f = W D H`: `W x = W_f (H (D x))`, with `D` a diagonal of
/// signs and `H` the normalized block Walsh-Hadamard transform (llama.cpp `build_hadamard_input`).
#[derive(Debug, Clone)]
pub struct Hadamard {
    /// Block size of the transform (a power of two dividing the input width).
    pub block: usize,
    /// The signs `D` (one per input value), or `None` for the identity.
    pub signs: Option<Arc<Vec<f32>>>,
    /// `(head_dim, k_heads, repeats)` of a Gated DeltaNet output projection whose value heads are grouped
    /// (`prism.hadamard.gdn_v_grouped`): the input is reordered from `[hd, nk, rep]` to `[hd, rep, nk]` first.
    pub v_perm: Option<(usize, usize, usize)>,
}

impl Hadamard {
    /// `out = H (D (P x))`.
    pub fn apply_input(&self, x: &[f32], out: &mut [f32]) {
        if let Some((hd, nk, rep)) = self.v_perm {
            for k in 0..nk {
                for r in 0..rep {
                    out[hd * (r + rep * k)..hd * (r + rep * k + 1)].copy_from_slice(&x[hd * (k + nk * r)..hd * (k + nk * r + 1)]);
                }
            }
        } else {
            out.copy_from_slice(x);
        }
        if let Some(signs) = &self.signs {
            for (v, s) in out.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
        }
        fwht_blocks(out, self.block);
    }

    /// The inverse after a lookup in a Hadamard-latent table: `x = D (H z)`, in place.
    pub fn apply_inverse(&self, x: &mut [f32]) {
        fwht_blocks(x, self.block);
        if let Some(signs) = &self.signs {
            for (v, s) in x.iter_mut().zip(signs.iter()) {
                *v *= s;
            }
        }
    }
}

/// The parsed `prism.hadamard.*` metadata.
pub(crate) struct HadamardSpec {
    pub(crate) block: usize,
    pub(crate) weights: HashMap<String, ()>,
    pub(crate) inverse: HashMap<String, ()>,
    pub(crate) signs: HashMap<usize, Arc<Vec<f32>>>,
    pub(crate) gdn_v_grouped: bool,
}

impl HadamardSpec {
    pub(crate) fn from_gguf(g: &GgufFile) -> Result<Option<Self>, String> {
        let get = |k: &str| g.metadata.get(&format!("prism.hadamard.{k}"));
        let Some(version) = get("version").and_then(|v| v.as_u64()) else {
            return Ok(None);
        };
        if version != 1 && version != 2 {
            return Err(format!("unsupported prism.hadamard.version {version}"));
        }
        let tied_output = get("tied_output").and_then(|v| v.as_bool()).unwrap_or(false);
        if (version == 2) != tied_output {
            return Err("prism.hadamard version 2 requires tied_output, version 1 forbids it".into());
        }
        let block = get("block_size").and_then(|v| v.as_usize()).ok_or("prism.hadamard.block_size missing")?;
        if block == 0 || !block.is_power_of_two() {
            return Err(format!("invalid prism.hadamard.block_size {block}"));
        }
        if get("transform").and_then(|v| v.as_str()) != Some("normalized-sylvester-walsh-hadamard") {
            return Err("unsupported prism.hadamard.transform".into());
        }
        if get("axis").and_then(|v| v.as_str()) != Some("input-last-dimension") {
            return Err("unsupported prism.hadamard.axis".into());
        }
        let names = |k: &str| -> HashMap<String, ()> {
            get(k).and_then(|v| v.as_string_array()).unwrap_or_default().into_iter().map(|n| (n, ())).collect()
        };
        let mut weights = names("weight_names");
        let inverse = names("inverse_weight_names");
        if weights.is_empty() {
            return Err("prism.hadamard.weight_names is empty".into());
        }
        if inverse.keys().any(|n| n != "token_embd.weight") {
            return Err("prism.hadamard: only token_embd.weight is supported as an inverse-after-lookup table".into());
        }
        if tied_output {
            if !inverse.contains_key("token_embd.weight") {
                return Err("prism.hadamard.tied_output requires a latent token embedding".into());
            }
            weights.insert("token_embd.weight".into(), ());
        }
        let ints = |v: Option<&GgufMetadataValue>| -> Vec<i64> {
            match v {
                Some(GgufMetadataValue::Array(a)) => a
                    .iter()
                    .filter_map(|x| match x {
                        GgufMetadataValue::Int32(i) => Some(*i as i64),
                        GgufMetadataValue::Int8(i) => Some(*i as i64),
                        other => other.as_u64().map(|u| u as i64),
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };
        let mut signs = HashMap::new();
        match get("sign_mode").and_then(|v| v.as_str()) {
            Some("identity") => {}
            Some("explicit") => {
                let widths = ints(get("sign_widths"));
                let values = ints(get("sign_values"));
                if widths.is_empty() {
                    return Err("prism.hadamard.sign_mode is explicit but sign_widths is empty".into());
                }
                let mut off = 0usize;
                for w in widths {
                    let w = usize::try_from(w).map_err(|_| "negative prism.hadamard sign width")?;
                    if w == 0 || !w.is_multiple_of(block) || off + w > values.len() {
                        return Err(format!("invalid prism.hadamard sign width {w}"));
                    }
                    let vec: Vec<f32> = values[off..off + w].iter().map(|&s| s as f32).collect();
                    if vec.iter().any(|&s| s != 1.0 && s != -1.0) {
                        return Err("prism.hadamard sign values must be +/-1".into());
                    }
                    signs.insert(w, Arc::new(vec));
                    off += w;
                }
                if off != values.len() {
                    return Err("prism.hadamard.sign_values length mismatch".into());
                }
            }
            other => return Err(format!("unsupported prism.hadamard.sign_mode {other:?}")),
        }
        let gdn_v_grouped = get("gdn_v_grouped").and_then(|v| v.as_bool()).unwrap_or(false);
        Ok(Some(Self { block, weights, inverse, signs, gdn_v_grouped }))
    }

    pub(crate) fn transform_for(&self, width: usize) -> Result<Hadamard, String> {
        if !width.is_multiple_of(self.block) {
            return Err(format!("prism.hadamard block size {} does not divide input width {width}", self.block));
        }
        let signs = if self.signs.is_empty() {
            None
        } else {
            Some(self.signs.get(&width).cloned().ok_or_else(|| format!("prism.hadamard has no sign vector for width {width}"))?)
        };
        Ok(Hadamard { block: self.block, signs, v_perm: None })
    }
}

// ============================================================================
// Projections
// ============================================================================

/// Weights of a projection, in the format of the file.
#[derive(Clone)]
pub enum Weights {
    /// Ternary weights with block scales (`Q2_0`), fed with `Q8_0` inputs.
    Ternary(LinearLayer),
    /// Plain F32 rows, `ggml_vec_dot_f32` with the F32 input.
    F32(Vec<f32>),
    /// BF16 rows, `ggml_vec_dot_bf16` with the input rounded to BF16.
    Bf16(Vec<u16>),
    /// `Q8_0` rows (int8 codes and one scale per 32 values), `ggml_vec_dot_q8_0_q8_0` with `Q8_0` inputs.
    Q8 { qs: Vec<i8>, d: Vec<f32>, threads: usize },
}

/// Dot product of a `Q8_0` weight row (codes `w`, block scales `wd`) with `Q8_0` activations (`a`, `ad`), bit-identical
/// to ggml's `ggml_vec_dot_q8_0_q8_0` (AVX2 code path): eight integer lane sums of four products per block of 32, one
/// accumulator `acc = fma(wd * ad, lane, acc)`, and the horizontal sum of `hsum_float_8`.
pub fn dot_q8_0(w: &[i8], wd: &[f32], a: &[i8], ad: &[f32]) -> f32 {
    assert!(w.len() == a.len() && w.len().is_multiple_of(32) && wd.len() >= w.len() / 32 && ad.len() >= a.len() / 32);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: the features are available; the lengths are checked above.
            return unsafe { dot_q8_0_avx512(w, wd, a, ad) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: the dot-product extension is available; the lengths are checked above.
            return unsafe { dot_q8_0_neon(w, wd, a, ad) };
        }
    }
    dot_q8_0_scalar(w, wd, a, ad)
}

/// NEON [`dot_q8_0`]: `vdotq_s32` yields the eight lane sums of four products of a block (lanes 0..3 from its first
/// 16 values, 4..7 from the second), then one fused multiply-add per lane, as in the definition.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q8_0_neon(w: &[i8], wd: &[f32], a: &[i8], ad: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_s32(0);
        let (mut lo, mut hi) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for b in 0..w.len() / 32 {
            let (pw, pa) = (w.as_ptr().add(32 * b), a.as_ptr().add(32 * b));
            let d = vdupq_n_f32(*wd.get_unchecked(b) * *ad.get_unchecked(b));
            let p_lo = vcvtq_f32_s32(vdotq_s32(zero, vld1q_s8(pw), vld1q_s8(pa)));
            let p_hi = vcvtq_f32_s32(vdotq_s32(zero, vld1q_s8(pw.add(16)), vld1q_s8(pa.add(16))));
            lo = vfmaq_f32(lo, p_lo, d);
            hi = vfmaq_f32(hi, p_hi, d);
        }
        let mut acc = [0.0f32; 8];
        vst1q_f32(acc.as_mut_ptr(), lo);
        vst1q_f32(acc.as_mut_ptr().add(4), hi);
        let r = [acc[4] + acc[0], acc[5] + acc[1], acc[6] + acc[2], acc[7] + acc[3]];
        (r[0] + r[2]) + (r[1] + r[3])
    }
}

/// AVX-512 [`dot_q8_0`]: `vpdpbusd` (VNNI) multiplies unsigned by signed bytes, so the weights are shifted to
/// `w + 128` and `128 * (a0 + a1 + a2 + a3)` is subtracted again; this gives the eight exact lane sums of four signed
/// products for every input. Then one fused multiply-add per lane, as in the definition.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512vl,avx512vnni")]
unsafe fn dot_q8_0_avx512(w: &[i8], wd: &[f32], a: &[i8], ad: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for b in 0..w.len() / 32 {
            let wv = _mm256_loadu_si256(w.as_ptr().add(32 * b) as *const __m256i);
            let av = _mm256_loadu_si256(a.as_ptr().add(32 * b) as *const __m256i);
            let zero = _mm256_setzero_si256();
            let biased = _mm256_dpbusd_epi32(zero, _mm256_xor_si256(wv, _mm256_set1_epi8(-128)), av);
            let lane = _mm256_sub_epi32(biased, _mm256_dpbusd_epi32(zero, _mm256_set1_epi8(-128), av));
            let d = _mm256_set1_ps(*wd.get_unchecked(b) * *ad.get_unchecked(b));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(lane), acc);
        }
        let mut l = [0.0f32; 8];
        _mm256_storeu_ps(l.as_mut_ptr(), acc);
        let r = [l[4] + l[0], l[5] + l[1], l[6] + l[2], l[7] + l[3]];
        (r[0] + r[2]) + (r[1] + r[3])
    }
}

multiversion! {
    /// The definition of [`dot_q8_0`] (portable code). On x86-64 it runs with fused multiply-adds when the CPU has
    /// them (same bits, module `dispatch`).
    pub fn dot_q8_0_scalar(w: &[i8], wd: &[f32], a: &[i8], ad: &[f32]) -> f32 {
        let mut acc = [0.0f32; 8];
        for (b, (wb, ab)) in w.chunks_exact(32).zip(a.chunks_exact(32)).enumerate() {
            let d = wd[b] * ad[b];
            for (j, lane) in acc.iter_mut().enumerate() {
                let p: i32 = (4 * j..4 * j + 4).map(|k| wb[k] as i32 * ab[k] as i32).sum();
                *lane = d.mul_add(p as f32, *lane);
            }
        }
        let r = [acc[4] + acc[0], acc[5] + acc[1], acc[6] + acc[2], acc[7] + acc[3]];
        (r[0] + r[2]) + (r[1] + r[3])
    }
}

multiversion! {
    /// One step of the causal depthwise convolution of every channel over `[state (kernel - 1 values), new value]`:
    /// `out[c] = sum_i v_i * w[c][i]` (accumulated with fused multiply-adds, plus the bias if there is one), then the
    /// state of the channel moves on by one value. `state` holds `kernel - 1` values per channel.
    fn causal_conv_step(
        state: &mut [f32],
        weights: &[f32],
        kernel: usize,
        input: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let km1 = kernel - 1;
        for c in 0..input.len() {
            let st = &mut state[c * km1..(c + 1) * km1];
            let w = &weights[c * kernel..(c + 1) * kernel];
            let mut sum = 0.0f32;
            for (i, &wi) in w.iter().enumerate() {
                let v = if i < km1 { st[i] } else { input[c] };
                sum = v.mul_add(wi, sum);
            }
            out[c] = match bias {
                Some(b) => sum + b[c],
                None => sum,
            };
            st.copy_within(1.., 0);
            st[km1 - 1] = input[c];
        }
    }
}

multiversion! {
    /// `state[j][e] = fma(k[e], delta[j], state[j][e])` for the `hd` rows `j` of a transposed recurrent state.
    fn rank_one_update(state: &mut [f32], k: &[f32], delta: &[f32], hd: usize) {
        for j in 0..hd {
            let dj = delta[j];
            for (e, &ki) in state[j * hd..(j + 1) * hd].iter_mut().zip(k) {
                *e = ki.mul_add(dj, *e);
            }
        }
    }
}

/// A projection `rows x cols` with an optional Hadamard input transform.
#[derive(Clone)]
pub struct Proj {
    pub weights: Weights,
    pub rows: usize,
    pub cols: usize,
    pub hadamard: Option<Hadamard>,
}

/// Reusable buffers of [`Proj::forward`].
#[derive(Debug, Clone, Default)]
pub struct ProjScratch {
    t: Vec<f32>,
    qs: Vec<i8>,
    d: Vec<f32>,
    b16: Vec<u16>,
}

impl Proj {
    /// `out = W x` (`out.len() == rows`, `x.len() == cols`).
    pub fn forward(&self, x: &[f32], out: &mut [f32], s: &mut ProjScratch) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        let x = if let Some(h) = &self.hadamard {
            s.t.resize(self.cols, 0.0);
            h.apply_input(x, &mut s.t);
            &s.t[..]
        } else {
            x
        };
        match &self.weights {
            Weights::Ternary(layer) => {
                s.qs.resize(self.cols, 0);
                s.d.resize(self.cols / 32, 0.0);
                quantize_q8_0(x, &mut s.qs, &mut s.d);
                layer.forward_q8(&s.qs, &s.d, out);
            }
            Weights::F32(w) => {
                for (r, o) in out.iter_mut().enumerate() {
                    *o = ggml_vec_dot_f32(&w[r * self.cols..(r + 1) * self.cols], x);
                }
            }
            Weights::Bf16(w) => {
                s.b16.clear();
                s.b16.extend(x.iter().map(|&v| f32_to_bf16(v)));
                for (r, o) in out.iter_mut().enumerate() {
                    *o = ggml_vec_dot_bf16(&w[r * self.cols..(r + 1) * self.cols], &s.b16);
                }
            }
            Weights::Q8 { qs, d, threads } => {
                s.qs.resize(self.cols, 0);
                s.d.resize(self.cols / 32, 0.0);
                quantize_q8_0(x, &mut s.qs, &mut s.d);
                let (aq, ad, cols) = (&s.qs[..], &s.d[..], self.cols);
                let row = |r: usize| dot_q8_0(&qs[r * cols..(r + 1) * cols], &d[r * cols / 32..(r + 1) * cols / 32], aq, ad);
                if *threads > 1 && self.rows >= 256 {
                    quintrit_core::threading::get_thread_pool(*threads).install(|| {
                        out.par_chunks_mut(256).enumerate().for_each(|(c, o)| {
                            for (i, v) in o.iter_mut().enumerate() {
                                *v = row(c * 256 + i);
                            }
                        });
                    });
                } else {
                    for (r, o) in out.iter_mut().enumerate() {
                        *o = row(r);
                    }
                }
            }
        }
    }

    /// Writes the dequantized row `r` into `out` (`cols` values): ggml's `get_rows` of the table.
    pub fn dequantize_row(&self, r: usize, out: &mut [f32]) {
        let c = self.cols;
        match &self.weights {
            Weights::Ternary(LinearLayer::BlockScaled(t)) => t.dequantize_row(r, out),
            Weights::Ternary(_) => panic!("only block-scaled ternary tables can be dequantized"),
            Weights::F32(w) => out.copy_from_slice(&w[r * c..(r + 1) * c]),
            Weights::Bf16(w) => {
                for (o, &v) in out.iter_mut().zip(&w[r * c..(r + 1) * c]) {
                    *o = crate::ops::bf16_to_f32(v);
                }
            }
            Weights::Q8 { qs, d, .. } => {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = qs[r * c + i] as f32 * d[(r * c + i) / 32];
                }
            }
        }
    }

    /// Worker threads the projection runs on (1 for the kinds that run on the calling thread).
    pub fn threads(&self) -> usize {
        match &self.weights {
            Weights::Ternary(l) => l.num_threads(),
            Weights::Q8 { threads, .. } => *threads,
            _ => 1,
        }
    }

    fn set_num_threads(&mut self, threads: usize) {
        match &mut self.weights {
            Weights::Ternary(l) => l.set_num_threads(threads),
            Weights::Q8 { threads: t, .. } => *t = threads.max(1),
            _ => {}
        }
    }

    /// Bytes of the weights.
    pub fn memory_bytes(&self) -> usize {
        match &self.weights {
            Weights::Ternary(l) => l.memory_bytes(),
            Weights::F32(w) => w.len() * 4,
            Weights::Bf16(w) => w.len() * 2,
            Weights::Q8 { qs, d, .. } => qs.len() + d.len() * 4,
        }
    }
}

// ============================================================================
// Layers
// ============================================================================

/// Gated DeltaNet linear-attention layer (Qwen3.5), with llama.cpp's `ggml_gated_delta_net` recurrence.
#[derive(Clone)]
pub struct GatedDeltaNet {
    qkv: Proj,
    z: Proj,
    beta: Proj,
    alpha: Proj,
    /// Depthwise causal convolution weights, `kernel` taps per channel.
    conv_w: Vec<f32>,
    dt_bias: Vec<f32>,
    /// `-exp(A_log)` per value head.
    a: Vec<f32>,
    norm: Vec<f32>,
    out: Proj,
    n_k_heads: usize,
    n_v_heads: usize,
    /// Head dimension of the keys and values (`ssm.state_size`).
    head_dim: usize,
    kernel: usize,
    eps: f32,
    /// Index of this layer among the recurrent layers (its slot in the recurrent state).
    rec_index: usize,
}

impl GatedDeltaNet {
    fn channels(&self) -> usize {
        2 * self.n_k_heads * self.head_dim + self.n_v_heads * self.head_dim
    }

    /// Length of the convolution state (`kernel - 1` values per channel).
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.channels()
    }

    /// Length of the recurrent state (`head_dim x head_dim` per value head).
    pub fn ssm_state_len(&self) -> usize {
        self.n_v_heads * self.head_dim * self.head_dim
    }

    fn forward(&self, x: &[f32], kv: &mut KVCache, out: &mut [f32], s: &mut HybridScratch) {
        let hd = self.head_dim;
        let ch = self.channels();
        let key_dim = self.n_k_heads * hd;
        let value_dim = self.n_v_heads * hd;
        let nv = self.n_v_heads;
        s.a.resize(ch, 0.0);
        s.b.resize(value_dim, 0.0);
        s.c.resize(ch, 0.0);
        s.small.resize(2 * nv, 0.0);

        self.qkv.forward(x, &mut s.a[..ch], &mut s.proj);
        self.z.forward(x, &mut s.b[..value_dim], &mut s.proj);
        let (beta, alpha) = s.small.split_at_mut(nv);
        self.beta.forward(x, beta, &mut s.proj);
        self.alpha.forward(x, alpha, &mut s.proj);
        for b in beta.iter_mut() {
            *b = sigmoid_f32(*b);
        }
        // gate = softplus(alpha + dt_bias) * A
        for (h, g) in alpha.iter_mut().enumerate() {
            *g = softplus_f32(*g + self.dt_bias[h]) * self.a[h];
        }

        // Causal depthwise convolution over [state (kernel - 1 values), new value] per channel; the state keeps the
        // last kernel - 1 inputs.
        let (conv_state, ssm_state) = kv.recurrent_state_mut(self.rec_index);
        causal_conv_step(conv_state, &self.conv_w, self.kernel, &s.a[..ch], None, &mut s.c[..ch]);
        silu(&s.c[..ch].to_vec(), &mut s.c[..ch]);

        // L2 norm of every query and key head: rms_norm(x, eps / n) * (1 / sqrt(n)).
        let n = hd as f32;
        let eps_l2 = self.eps / n;
        let inv_sqrt = 1.0 / n.sqrt();
        s.ones.resize(hd, 1.0);
        s.head.resize(hd, 0.0);
        for h in 0..2 * self.n_k_heads {
            let v = &mut s.c[h * hd..(h + 1) * hd];
            s.head.copy_from_slice(v);
            rms_norm(&s.head, &s.ones, eps_l2, v);
            for x in v.iter_mut() {
                *x *= inv_sqrt;
            }
        }

        // Recurrence (ggml_gated_delta_net, K = 1): the state of a value head is stored transposed, row j = column j.
        let scale = 1.0 / (hd as f32).sqrt();
        s.delta.resize(hd, 0.0);
        s.d.resize(value_dim, 0.0);
        for h in 0..nv {
            let kh = h % self.n_k_heads;
            let q = &s.c[kh * hd..(kh + 1) * hd];
            let k = &s.c[key_dim + kh * hd..key_dim + (kh + 1) * hd];
            let v = &s.c[2 * key_dim + h * hd..2 * key_dim + (h + 1) * hd];
            let st = &mut ssm_state[h * hd * hd..(h + 1) * hd * hd];
            let decay = alpha[h].exp();
            for e in st.iter_mut() {
                *e *= decay;
            }
            for j in 0..hd {
                let sum = ggml_vec_dot_f32(&st[j * hd..(j + 1) * hd], k);
                s.delta[j] = (v[j] - sum) * beta[h];
            }
            rank_one_update(st, k, &s.delta[..hd], hd);
            for j in 0..hd {
                s.d[h * hd + j] = ggml_vec_dot_f32(&st[j * hd..(j + 1) * hd], q) * scale;
            }
        }

        // Gated RMSNorm per value head: rms_norm(o, w) * silu(z).
        silu(&s.b[..value_dim].to_vec(), &mut s.b[..value_dim]);
        for h in 0..nv {
            s.head.copy_from_slice(&s.d[h * hd..(h + 1) * hd]);
            let o = &mut s.d[h * hd..(h + 1) * hd];
            rms_norm(&s.head, &self.norm, self.eps, o);
            for (x, &g) in o.iter_mut().zip(&s.b[h * hd..(h + 1) * hd]) {
                *x *= g;
            }
        }
        self.out.forward(&s.d[..value_dim], out, &mut s.proj);
    }
}

/// Gated full-attention layer (Qwen3.5): the query projection also yields a sigmoid gate per head and value.
#[derive(Clone)]
pub struct GatedAttention {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    rope: RopeConfig,
    eps: f32,
    /// Index of this layer among the attention layers (its slot in the KV cache).
    attn_index: usize,
}

impl GatedAttention {
    fn forward(&self, x: &[f32], pos: usize, kv: &mut KVCache, out: &mut [f32], s: &mut HybridScratch) {
        let hd = self.head_dim;
        let (nq, nkv) = (self.n_head, self.n_kv);
        s.a.resize(2 * nq * hd, 0.0);
        s.b.resize(nkv * hd, 0.0);
        s.c.resize(nkv * hd, 0.0);
        s.d.resize(nq * hd, 0.0);
        s.e.resize(nq * hd, 0.0);
        s.head.resize(hd, 0.0);
        self.q.forward(x, &mut s.a[..2 * nq * hd], &mut s.proj);
        self.k.forward(x, &mut s.b[..nkv * hd], &mut s.proj);
        self.v.forward(x, &mut s.c[..nkv * hd], &mut s.proj);

        // Split the joint projection: head h holds [query | gate].
        for h in 0..nq {
            s.head.copy_from_slice(&s.a[2 * h * hd..2 * h * hd + hd]);
            rms_norm(&s.head, &self.q_norm, self.eps, &mut s.d[h * hd..(h + 1) * hd]);
            s.e[h * hd..(h + 1) * hd].copy_from_slice(&s.a[2 * h * hd + hd..2 * (h + 1) * hd]);
        }
        for h in 0..nkv {
            s.head.copy_from_slice(&s.b[h * hd..(h + 1) * hd]);
            rms_norm(&s.head, &self.k_norm, self.eps, &mut s.b[h * hd..(h + 1) * hd]);
        }
        apply_rope(&mut s.d[..nq * hd], &mut s.b[..nkv * hd], pos, nq, nkv, hd, &self.rope);

        s.k16.resize(nkv * hd, 0);
        s.v16.resize(nkv * hd, 0);
        f32_to_f16_slice(&s.b[..nkv * hd], &mut s.k16[..nkv * hd]);
        f32_to_f16_slice(&s.c[..nkv * hd], &mut s.v16[..nkv * hd]);
        kv.append(self.attn_index, pos, &s.k16, &s.v16);

        s.q16.resize(nq * hd, 0);
        s.vkq16.resize(nq * hd, 0);
        s.acc.resize(nq * hd, 0.0);
        s.att.resize(nq * hd, 0.0);
        let layout = HeadsLayout { head_dim: hd, gqa_ratio: nq / nkv, f32_values: true, layer: self.attn_index, seq_len: pos + 1, scale: 1.0 / (hd as f32).sqrt() };
        attend_heads(&layout, &s.d[..nq * hd], &mut s.q16, &mut s.vkq16, &mut s.acc, kv, &mut s.att, self.q.threads());
        for (a, &g) in s.att.iter_mut().zip(&s.e) {
            *a *= sigmoid_f32(g);
        }
        let att = std::mem::take(&mut s.att);
        self.o.forward(&att, out, &mut s.proj);
        s.att = att;
    }
}

/// SwiGLU feed-forward block.
#[derive(Clone)]
pub struct Swiglu {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Swiglu {
    fn forward(&self, x: &[f32], out: &mut [f32], s: &mut HybridScratch) {
        let n = self.gate.rows;
        s.a.resize(n, 0.0);
        s.b.resize(n, 0.0);
        s.c.resize(n, 0.0);
        self.gate.forward(x, &mut s.a[..n], &mut s.proj);
        self.up.forward(x, &mut s.b[..n], &mut s.proj);
        swiglu(&s.a[..n], &s.b[..n], &mut s.c[..n]);
        let inter = std::mem::take(&mut s.c);
        self.down.forward(&inter[..n], out, &mut s.proj);
        s.c = inter;
    }
}


/// `ln(1 + e^x)` as ggml's `ggml_compute_softplus_f32` (used by the SSM scan; `logf(1 + expf(x))`, not `log1pf`).
#[inline]
fn softplus_scan(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Mamba-2 mixer (Nemotron-H), with llama.cpp's `ggml_ssm_conv` and `ggml_ssm_scan` arithmetic.
#[derive(Clone)]
pub struct Mamba2 {
    in_proj: Proj,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
    out: Proj,
    d_inner: usize,
    d_state: usize,
    n_group: usize,
    n_head: usize,
    kernel: usize,
    eps: f32,
    rec_index: usize,
}

impl Mamba2 {
    fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Length of the convolution state (`kernel - 1` values per channel).
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.conv_channels()
    }

    /// Length of the SSM state (`d_state` values per channel of `d_inner`).
    pub fn ssm_state_len(&self) -> usize {
        self.d_state * self.d_inner
    }

    fn forward(&self, x: &[f32], kv: &mut KVCache, out: &mut [f32], s: &mut HybridScratch) {
        let (di, ds, ng, nh) = (self.d_inner, self.d_state, self.n_group, self.n_head);
        let hd = di / nh;
        let ch = self.conv_channels();
        let n_in = self.in_proj.rows;
        s.a.resize(n_in, 0.0);
        s.c.resize(ch, 0.0);
        s.d.resize(di, 0.0);
        self.in_proj.forward(x, &mut s.a[..n_in], &mut s.proj);

        let (conv_state, ssm_state) = kv.recurrent_state_mut(self.rec_index);
        causal_conv_step(conv_state, &self.conv_w, self.kernel, &s.a[di..di + ch], Some(&self.conv_b), &mut s.c[..ch]);
        silu(&s.c[..ch].to_vec(), &mut s.c[..ch]);

        let (xs, rest) = s.c.split_at(di);
        let (bm, cm) = rest.split_at(ng * ds);
        let dt = &s.a[2 * di + 2 * ng * ds..2 * di + 2 * ng * ds + nh];
        for h in 0..nh {
            let sp = softplus_scan(dt[h] + self.dt_bias[h]);
            let da = (sp * self.a[h]).exp();
            let g = h / (nh / ng);
            let b = &bm[g * ds..(g + 1) * ds];
            let c = &cm[g * ds..(g + 1) * ds];
            for i1 in 0..hd {
                let ii = i1 + h * hd;
                let x_dt = xs[ii] * sp;
                let st = &mut ssm_state[ii * ds..(ii + 1) * ds];
                for (e, &bv) in st.iter_mut().zip(b) {
                    *e = *e * da + bv * x_dt;
                }
                s.d[ii] = ggml_vec_dot_f32(st, c);
            }
        }
        // y = y + x * D, then y = silu(z) * y, then the grouped RMSNorm.
        for (ii, y) in s.d[..di].iter_mut().enumerate() {
            *y += xs[ii] * self.d[ii / hd];
        }
        s.b.resize(di, 0.0);
        swiglu(&s.a[..di], &s.d[..di], &mut s.b[..di]);
        let gsz = di / ng;
        for g in 0..ng {
            rms_norm(&s.b[g * gsz..(g + 1) * gsz], &self.norm[g * gsz..(g + 1) * gsz], self.eps, &mut s.d[g * gsz..(g + 1) * gsz]);
        }
        self.out.forward(&s.d[..di], out, &mut s.proj);
    }
}

/// Plain multi-head attention without positional encoding (the attention layers of Nemotron-H).
#[derive(Clone)]
pub struct PlainAttention {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    attn_index: usize,
}

impl PlainAttention {
    fn forward(&self, x: &[f32], pos: usize, kv: &mut KVCache, out: &mut [f32], s: &mut HybridScratch) {
        let hd = self.head_dim;
        let (nq, nkv) = (self.n_head, self.n_kv);
        s.d.resize(nq * hd, 0.0);
        s.b.resize(nkv * hd, 0.0);
        s.c.resize(nkv * hd, 0.0);
        self.q.forward(x, &mut s.d[..nq * hd], &mut s.proj);
        self.k.forward(x, &mut s.b[..nkv * hd], &mut s.proj);
        self.v.forward(x, &mut s.c[..nkv * hd], &mut s.proj);
        s.k16.resize(nkv * hd, 0);
        s.v16.resize(nkv * hd, 0);
        f32_to_f16_slice(&s.b[..nkv * hd], &mut s.k16[..nkv * hd]);
        f32_to_f16_slice(&s.c[..nkv * hd], &mut s.v16[..nkv * hd]);
        kv.append(self.attn_index, pos, &s.k16, &s.v16);
        s.q16.resize(nq * hd, 0);
        s.vkq16.resize(nq * hd, 0);
        s.acc.resize(nq * hd, 0.0);
        s.att.resize(nq * hd, 0.0);
        let layout = HeadsLayout { head_dim: hd, gqa_ratio: nq / nkv, f32_values: true, layer: self.attn_index, seq_len: pos + 1, scale: 1.0 / (hd as f32).sqrt() };
        attend_heads(&layout, &s.d[..nq * hd], &mut s.q16, &mut s.vkq16, &mut s.acc, kv, &mut s.att, self.q.threads());
        let att = std::mem::take(&mut s.att);
        self.o.forward(&att, out, &mut s.proj);
        s.att = att;
    }
}

/// Mixture-of-experts feed-forward layer of Nemotron-H: sigmoid router with a selection bias, top-k experts of
/// squared-ReLU MLPs (no gate) weighted by the normalized and scaled router probabilities, plus a shared expert.
#[derive(Clone)]
pub struct Moe {
    router: Proj,
    probs_bias: Vec<f32>,
    up: Vec<Proj>,
    down: Vec<Proj>,
    shared_up: Proj,
    shared_down: Proj,
    n_used: usize,
    norm_weights: bool,
    weights_scale: f32,
}

fn relu_sqr(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = if *x > 0.0 { *x } else { 0.0 };
        *x = r * r;
    }
}

impl Moe {
    fn forward(&self, x: &[f32], out: &mut [f32], s: &mut HybridScratch) {
        let ne = self.up.len();
        let h = out.len();
        s.small.resize(ne, 0.0);
        self.router.forward(x, &mut s.small[..ne], &mut s.proj);
        let probs: Vec<f32> = s.small[..ne].iter().map(|&l| sigmoid_f32(l)).collect();
        let sel: Vec<f32> = probs.iter().zip(&self.probs_bias).map(|(p, b)| p + b).collect();
        // ggml_argsort (descending) runs std::sort; ties are broken exactly like the reference's standard library.
        let mut order: Vec<usize> = (0..ne).collect();
        crate::refsort::reference_sort(&mut order, &|a: usize, b: usize| sel[a] > sel[b]);
        let chosen = &order[..self.n_used];
        let mut w: Vec<f32> = chosen.iter().map(|&e| probs[e]).collect();
        if self.norm_weights {
            let sum = w.iter().map(|&v| v as f64).sum::<f64>() as f32;
            let sum = sum.max(6.103515625e-5);
            for v in w.iter_mut() {
                *v /= sum;
            }
        }
        if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
            for v in w.iter_mut() {
                *v *= self.weights_scale;
            }
        }
        s.e.resize(h, 0.0);
        s.head.resize(h, 0.0);
        for (i, &e) in chosen.iter().enumerate() {
            let nf = self.up[e].rows;
            s.a.resize(nf, 0.0);
            self.up[e].forward(x, &mut s.a[..nf], &mut s.proj);
            relu_sqr(&mut s.a[..nf]);
            let inter = s.a[..nf].to_vec();
            self.down[e].forward(&inter, &mut s.head[..h], &mut s.proj);
            for (o, &v) in s.e[..h].iter_mut().zip(&s.head[..h]) {
                let v = v * w[i];
                *o = if i == 0 { v } else { *o + v };
            }
        }
        let nf = self.shared_up.rows;
        s.a.resize(nf, 0.0);
        self.shared_up.forward(x, &mut s.a[..nf], &mut s.proj);
        relu_sqr(&mut s.a[..nf]);
        let inter = s.a[..nf].to_vec();
        self.shared_down.forward(&inter, out, &mut s.proj);
        for (o, &m) in out.iter_mut().zip(&s.e[..h]) {
            *o = m + *o;
        }
    }
}

/// Token mixer of a hybrid layer.
#[derive(Clone)]
pub enum Mixer {
    DeltaNet(GatedDeltaNet),
    Attention(GatedAttention),
    Mamba2(Mamba2),
    PlainAttention(PlainAttention),
    Moe(Moe),
}

/// One hybrid decoder layer: `x += mixer(norm(x))`, then (Qwen3.5) `x += ffn(post_norm(x))`.
#[derive(Clone)]
pub struct HybridLayer {
    attn_norm: Vec<f32>,
    mixer: Mixer,
    ffn: Option<(Vec<f32>, Swiglu)>,
}

/// Reusable buffers of the hybrid forward pass.
#[derive(Debug, Clone, Default)]
pub struct HybridScratch {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    e: Vec<f32>,
    small: Vec<f32>,
    head: Vec<f32>,
    ones: Vec<f32>,
    delta: Vec<f32>,
    acc: Vec<f32>,
    att: Vec<f32>,
    xn: Vec<f32>,
    mix: Vec<f32>,
    q16: Vec<u16>,
    k16: Vec<u16>,
    v16: Vec<u16>,
    vkq16: Vec<u16>,
    proj: ProjScratch,
}

impl HybridScratch {
    /// The buffers of the projections.
    pub fn proj_mut(&mut self) -> &mut ProjScratch {
        &mut self.proj
    }
}

/// A hybrid decoder stack with its embedding table and output head.
#[derive(Clone)]
pub struct HybridModel {
    pub layers: Vec<HybridLayer>,
    /// The token embedding table (rows dequantized per token).
    pub embedding: Proj,
    /// Inverse transform applied to a looked-up embedding row (Hadamard-latent token table).
    pub embedding_inverse: Option<Hadamard>,
    /// The vocabulary projection.
    pub output: Proj,
    /// Number of attention layers (slots of the KV cache).
    pub attn_layers: usize,
    /// `(conv, ssm)` state lengths of every recurrent layer, in order.
    pub recurrent_lens: Vec<(usize, usize)>,
}

impl HybridModel {
    fn projections_mut(&mut self) -> Vec<&mut Proj> {
        let mut v: Vec<&mut Proj> = vec![&mut self.output, &mut self.embedding];
        for l in &mut self.layers {
            match &mut l.mixer {
                Mixer::DeltaNet(m) => v.extend([&mut m.qkv, &mut m.z, &mut m.beta, &mut m.alpha, &mut m.out]),
                Mixer::Attention(m) => v.extend([&mut m.q, &mut m.k, &mut m.v, &mut m.o]),
                Mixer::Mamba2(m) => v.extend([&mut m.in_proj, &mut m.out]),
                Mixer::PlainAttention(m) => v.extend([&mut m.q, &mut m.k, &mut m.v, &mut m.o]),
                Mixer::Moe(m) => {
                    v.extend([&mut m.router, &mut m.shared_up, &mut m.shared_down]);
                    v.extend(m.up.iter_mut());
                    v.extend(m.down.iter_mut());
                }
            }
            if let Some((_, f)) = &mut l.ffn {
                v.extend([&mut f.gate, &mut f.up, &mut f.down]);
            }
        }
        v
    }

    pub fn set_num_threads(&mut self, threads: usize) {
        for p in self.projections_mut() {
            p.set_num_threads(threads);
        }
    }

    /// Bytes of all weights except the embedding table (for MoE layers including the experts that are not used for a
    /// given token).
    pub fn memory_bytes(&self) -> usize {
        let sum = |ps: &[&Proj]| ps.iter().map(|p| p.memory_bytes()).sum::<usize>();
        let mut total = self.output.memory_bytes();
        for l in &self.layers {
            total += match &l.mixer {
                Mixer::DeltaNet(m) => sum(&[&m.qkv, &m.z, &m.beta, &m.alpha, &m.out]),
                Mixer::Attention(m) => sum(&[&m.q, &m.k, &m.v, &m.o]),
                Mixer::Mamba2(m) => sum(&[&m.in_proj, &m.out]),
                Mixer::PlainAttention(m) => sum(&[&m.q, &m.k, &m.v, &m.o]),
                Mixer::Moe(m) => {
                    sum(&[&m.router, &m.shared_up, &m.shared_down])
                        + m.up.iter().chain(&m.down).map(|p| p.memory_bytes()).sum::<usize>()
                }
            };
            if let Some((_, f)) = &l.ffn {
                total += sum(&[&f.gate, &f.up, &f.down]);
            }
        }
        total
    }

    /// Embedding lookup: the dequantized row, with the inverse Hadamard transform when the table is stored rotated.
    pub fn embed(&self, token: usize, x: &mut [f32]) {
        self.embedding.dequantize_row(token, x);
        if let Some(inv) = &self.embedding_inverse {
            inv.apply_inverse(x);
        }
    }

    /// Runs the stack on `x` (the embedding of the token at `pos`, replaced by the final hidden state before the
    /// output norm).
    pub fn forward_layers(&self, x: &mut [f32], pos: usize, kv: &mut KVCache, eps: f32, s: &mut HybridScratch) {
        self.forward_layers_observed(x, pos, kv, eps, s, &mut |_, _| {});
    }

    /// [`HybridModel::forward_layers`] that hands the hidden state after every layer to `observer` (layer index, state),
    /// for layer-by-layer comparisons with a reference runtime.
    pub fn forward_layers_observed(
        &self,
        x: &mut [f32],
        pos: usize,
        kv: &mut KVCache,
        eps: f32,
        s: &mut HybridScratch,
        observer: &mut dyn FnMut(usize, &[f32]),
    ) {
        let h = x.len();
        let mut xn = std::mem::take(&mut s.xn);
        let mut mix = std::mem::take(&mut s.mix);
        xn.resize(h, 0.0);
        mix.resize(h, 0.0);
        for (li, layer) in self.layers.iter().enumerate() {
            rms_norm(x, &layer.attn_norm, eps, &mut xn);
            match &layer.mixer {
                Mixer::DeltaNet(m) => m.forward(&xn, kv, &mut mix, s),
                Mixer::Attention(m) => m.forward(&xn, pos, kv, &mut mix, s),
                Mixer::Mamba2(m) => m.forward(&xn, kv, &mut mix, s),
                Mixer::PlainAttention(m) => m.forward(&xn, pos, kv, &mut mix, s),
                Mixer::Moe(m) => m.forward(&xn, &mut mix, s),
            }
            // ggml_add(mixer_out, x): the residual is added to the mixer output.
            for (a, &m) in x.iter_mut().zip(&mix) {
                *a = m + *a;
            }
            if let Some((post_norm, ffn)) = &layer.ffn {
                rms_norm(x, post_norm, eps, &mut xn);
                ffn.forward(&xn, &mut mix, s);
                add_residual(x, &mix);
            }
            observer(li, x);
        }
        s.xn = xn;
        s.mix = mix;
    }
}

// ============================================================================
// Loading
// ============================================================================

struct Loader<'a> {
    g: &'a GgufFile,
    threads: usize,
    hadamard: Option<HadamardSpec>,
    hidden: usize,
}

impl Loader<'_> {
    fn f32(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let v = self.g.get_tensor_f32(name).ok_or_else(|| format!("Missing F32 tensor {name}"))?;
        if v.len() != len {
            return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
        }
        Ok(v)
    }

    fn proj(&self, name: &str, rows: usize, cols: usize) -> Result<Proj, String> {
        let info = self.g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
        if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows {
            return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
        }
        let data = self.g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
        let weights = match info.tensor_type {
            GGML_TYPE_Q2_0 => {
                let block = info.block_size.ok_or_else(|| format!("{name}: unknown Q2_0 block size"))?;
                let layer = crate::layer::BlockScaledLayer::from_q2_0(data, rows, cols, block, self.threads).map_err(|e| format!("{name}: {e}"))?;
                Weights::Ternary(LinearLayer::BlockScaled(layer))
            }
            GGML_TYPE_F32 => Weights::F32(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            GGML_TYPE_BF16 => Weights::Bf16(data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
            other => return Err(format!("tensor '{name}' has type {other}, which this architecture does not support")),
        };
        let hadamard = match &self.hadamard {
            Some(spec) if spec.weights.contains_key(name) => Some(spec.transform_for(cols)?),
            _ => None,
        };
        Ok(Proj { weights, rows, cols, hadamard })
    }

    /// An embedding table or output head: `Q2_0` or `Q8_0`.
    fn table(&self, name: &str, rows: usize, cols: usize) -> Result<Proj, String> {
        let info = self.g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
        if info.tensor_type != GGML_TYPE_Q8_0 {
            return self.proj(name, rows, cols);
        }
        if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows || !cols.is_multiple_of(32) {
            return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
        }
        let data = self.g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
        let mut qs = Vec::with_capacity(rows * cols);
        let mut d = Vec::with_capacity(rows * cols / 32);
        for b in data.chunks_exact(34) {
            d.push(f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
            qs.extend(b[2..].iter().map(|&v| v as i8));
        }
        let hadamard = match &self.hadamard {
            Some(spec) if spec.weights.contains_key(name) => Some(spec.transform_for(cols)?),
            _ => None,
        };
        Ok(Proj { weights: Weights::Q8 { qs, d, threads: self.threads }, rows, cols, hadamard })
    }

    /// The `n` experts of a 3-D expert tensor `[cols, rows, n]`, one projection each.
    fn experts(&self, name: &str, n: usize, rows: usize, cols: usize) -> Result<Vec<Proj>, String> {
        let info = self.g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
        if info.shape != [cols, rows, n] || info.tensor_type != GGML_TYPE_Q2_0 {
            return Err(format!("tensor '{name}' has shape {:?} and type {}, expected Q2_0 [{cols}, {rows}, {n}]", info.shape, info.tensor_type));
        }
        let block = info.block_size.ok_or_else(|| format!("{name}: unknown Q2_0 block size"))?;
        let data = self.g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
        let per = data.len() / n;
        (0..n)
            .map(|e| {
                let layer = crate::layer::BlockScaledLayer::from_q2_0(&data[e * per..(e + 1) * per], rows, cols, block, self.threads)
                    .map_err(|err| format!("{name}[{e}]: {err}"))?;
                Ok(Proj { weights: Weights::Ternary(LinearLayer::BlockScaled(layer)), rows, cols, hadamard: None })
            })
            .collect()
    }

    fn swiglu(&self, l: usize, n_ff: usize) -> Result<Swiglu, String> {
        let h = self.hidden;
        Ok(Swiglu {
            gate: self.proj(&format!("blk.{l}.ffn_gate.weight"), n_ff, h)?,
            up: self.proj(&format!("blk.{l}.ffn_up.weight"), n_ff, h)?,
            down: self.proj(&format!("blk.{l}.ffn_down.weight"), h, n_ff)?,
        })
    }
}

/// Loads a Qwen3.5 (`qwen35`) model. Returns the stack (with the output head) and the KV cache with its recurrent
/// state.
pub fn load_qwen35(
    g: &GgufFile,
    config: &ModelConfig,
    threads: usize,
) -> Result<(HybridModel, KVCache), String> {
    let key = |k: &str| g.metadata.get(&format!("qwen35.{k}"));
    let num = |k: &str| key(k).and_then(|v| v.as_usize()).ok_or_else(|| format!("metadata 'qwen35.{k}' missing"));
    let d_conv = num("ssm.conv_kernel")?;
    let d_state = num("ssm.state_size")?;
    let n_group = num("ssm.group_count")?;
    let dt_rank = num("ssm.time_step_rank")?;
    let d_inner = num("ssm.inner_size")?;
    let interval = key("full_attention_interval").and_then(|v| v.as_usize()).unwrap_or(4);
    if d_conv < 2 || d_state == 0 || n_group == 0 || dt_rank == 0 || !dt_rank.is_multiple_of(n_group) || d_inner != dt_rank * d_state {
        return Err("inconsistent qwen35.ssm.* hyper-parameters".into());
    }
    if interval == 0 {
        return Err("qwen35.full_attention_interval must be positive".into());
    }
    let h = config.hidden_size;
    let hd = config.head_dim;
    let (nq, nkv) = (config.num_attention_heads, config.num_key_value_heads);
    let eps = config.rms_norm_eps;
    let mut hadamard = HadamardSpec::from_gguf(g)?;
    let gdn_v_grouped = hadamard.as_ref().is_some_and(|s| s.gdn_v_grouped);
    let ld = Loader { g, threads, hadamard: hadamard.take(), hidden: h };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    let mut recurrent_lens = Vec::new();
    let mut attn_layers = 0;
    let key_dim = n_group * d_state;
    let value_dim = dt_rank * d_state;
    for l in 0..config.num_hidden_layers {
        let attn_norm = ld.f32(&format!("blk.{l}.attn_norm.weight"), h)?;
        let post_norm = ld.f32(&format!("blk.{l}.post_attention_norm.weight"), h)?;
        let recurrent = (l + 1) % interval != 0;
        let mixer = if recurrent {
            let mut out = ld.proj(&format!("blk.{l}.ssm_out.weight"), h, value_dim)?;
            if gdn_v_grouped {
                if let Some(t) = &mut out.hadamard {
                    t.v_perm = Some((value_dim / dt_rank, n_group, dt_rank / n_group));
                }
            }
            let m = GatedDeltaNet {
                qkv: ld.proj(&format!("blk.{l}.attn_qkv.weight"), 2 * key_dim + value_dim, h)?,
                z: ld.proj(&format!("blk.{l}.attn_gate.weight"), value_dim, h)?,
                beta: ld.proj(&format!("blk.{l}.ssm_beta.weight"), dt_rank, h)?,
                alpha: ld.proj(&format!("blk.{l}.ssm_alpha.weight"), dt_rank, h)?,
                conv_w: ld.f32(&format!("blk.{l}.ssm_conv1d.weight"), d_conv * (2 * key_dim + value_dim))?,
                dt_bias: ld.f32(&format!("blk.{l}.ssm_dt.bias"), dt_rank)?,
                a: ld.f32(&format!("blk.{l}.ssm_a"), dt_rank)?,
                norm: ld.f32(&format!("blk.{l}.ssm_norm.weight"), d_state)?,
                out,
                n_k_heads: n_group,
                n_v_heads: dt_rank,
                head_dim: d_state,
                kernel: d_conv,
                eps,
                rec_index: recurrent_lens.len(),
            };
            recurrent_lens.push((m.conv_state_len(), m.ssm_state_len()));
            Mixer::DeltaNet(m)
        } else {
            let m = GatedAttention {
                q: ld.proj(&format!("blk.{l}.attn_q.weight"), 2 * nq * hd, h)?,
                k: ld.proj(&format!("blk.{l}.attn_k.weight"), nkv * hd, h)?,
                v: ld.proj(&format!("blk.{l}.attn_v.weight"), nkv * hd, h)?,
                o: ld.proj(&format!("blk.{l}.attn_output.weight"), h, nq * hd)?,
                q_norm: ld.f32(&format!("blk.{l}.attn_q_norm.weight"), hd)?,
                k_norm: ld.f32(&format!("blk.{l}.attn_k_norm.weight"), hd)?,
                n_head: nq,
                n_kv: nkv,
                head_dim: hd,
                rope: config.rope,
                eps,
                attn_index: attn_layers,
            };
            attn_layers += 1;
            Mixer::Attention(m)
        };
        let ffn = ld.swiglu(l, config.intermediate_size)?;
        layers.push(HybridLayer { attn_norm, mixer, ffn: Some((post_norm, ffn)) });
    }

    let output_name = if g.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let output = ld.proj(output_name, config.vocab_size, h)?;
    let embedding_inverse = match &ld.hadamard {
        Some(spec) if spec.inverse.contains_key("token_embd.weight") => Some(spec.transform_for(h)?),
        _ => None,
    };

    let embedding = ld.table("token_embd.weight", config.vocab_size, h)?;
    let kv = KVCache::try_new_hybrid(attn_layers, context_len(config), nkv, hd, &recurrent_lens)?;
    Ok((HybridModel { layers, embedding, embedding_inverse, output, attn_layers, recurrent_lens }, kv))
}

/// Context window of the KV cache: the model's, at most [`MAX_HYBRID_CONTEXT`] positions.
fn context_len(config: &ModelConfig) -> usize {
    config.max_position_embeddings.min(MAX_HYBRID_CONTEXT)
}

/// Largest context window allocated for the hybrid models, whose nominal windows (256k and 1M positions) would
/// reserve tens of gigabytes of KV cache.
pub const MAX_HYBRID_CONTEXT: usize = 32768;

/// Per-layer integer metadata that is either one value or an array with one value per layer.
pub(crate) fn per_layer(g: &GgufFile, key: &str, layers: usize) -> Result<Vec<usize>, String> {
    match g.metadata.get(key) {
        Some(GgufMetadataValue::Array(a)) if a.len() == layers => {
            a.iter().map(|v| v.as_usize().ok_or_else(|| format!("metadata '{key}' has a non-integer entry"))).collect()
        }
        Some(GgufMetadataValue::Array(a)) => Err(format!("metadata '{key}' has {} entries for {layers} layers", a.len())),
        Some(v) => Ok(vec![v.as_usize().ok_or_else(|| format!("metadata '{key}' is not an integer"))?; layers]),
        None => Err(format!("metadata '{key}' missing")),
    }
}

/// Loads a Nemotron-H mixture-of-experts model (`nemotron_h_moe`): Mamba-2, attention and MoE layers.
pub fn load_nemotron_h_moe(g: &GgufFile, config: &ModelConfig, threads: usize) -> Result<(HybridModel, KVCache), String> {
    let arch = "nemotron_h_moe";
    let num = |k: &str| g.metadata.get(&format!("{arch}.{k}")).and_then(|v| v.as_usize()).ok_or_else(|| format!("metadata '{arch}.{k}' missing"));
    let n_layer = config.num_hidden_layers;
    let d_conv = num("ssm.conv_kernel")?;
    let d_state = num("ssm.state_size")?;
    let n_group = num("ssm.group_count")?;
    let n_head = num("ssm.time_step_rank")?;
    let d_inner = num("ssm.inner_size")?;
    let n_expert = num("expert_count")?;
    let n_used = num("expert_used_count")?;
    let n_ff_exp = num("expert_feed_forward_length")?;
    let n_ff_shexp = num("expert_shared_feed_forward_length")?;
    let norm_weights = g.metadata.get(&format!("{arch}.expert_weights_norm")).and_then(|v| v.as_bool()).unwrap_or(false);
    let weights_scale = g.metadata.get(&format!("{arch}.expert_weights_scale")).and_then(|v| v.as_f32()).unwrap_or(1.0);
    if d_conv < 2 || n_group == 0 || n_head == 0 || !n_head.is_multiple_of(n_group) || !d_inner.is_multiple_of(n_head)
        || !d_inner.is_multiple_of(n_group) || n_used == 0 || n_used > n_expert
    {
        return Err("inconsistent nemotron_h_moe hyper-parameters".into());
    }
    let kv_heads = per_layer(g, &format!("{arch}.attention.head_count_kv"), n_layer)?;
    let n_ff = per_layer(g, &format!("{arch}.feed_forward_length"), n_layer)?;
    let heads = per_layer(g, &format!("{arch}.attention.head_count"), n_layer)?;
    let h = config.hidden_size;
    let hd = config.head_dim;
    let eps = config.rms_norm_eps;
    let ld = Loader { g, threads, hadamard: None, hidden: h };

    let mut layers = Vec::with_capacity(n_layer);
    let mut recurrent_lens = Vec::new();
    let mut attn_layers = 0;
    let mut n_kv_attn = 0;
    for l in 0..n_layer {
        let attn_norm = ld.f32(&format!("blk.{l}.attn_norm.weight"), h)?;
        let mixer = if kv_heads[l] == 0 && n_ff[l] == 0 {
            let d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_head;
            let ch = d_inner + 2 * n_group * d_state;
            let m = Mamba2 {
                in_proj: ld.proj(&format!("blk.{l}.ssm_in.weight"), d_in_proj, h)?,
                conv_w: ld.f32(&format!("blk.{l}.ssm_conv1d.weight"), d_conv * ch)?,
                conv_b: ld.f32(&format!("blk.{l}.ssm_conv1d.bias"), ch)?,
                dt_bias: ld.f32(&format!("blk.{l}.ssm_dt.bias"), n_head)?,
                a: ld.f32(&format!("blk.{l}.ssm_a"), n_head)?,
                d: ld.f32(&format!("blk.{l}.ssm_d"), n_head)?,
                norm: ld.f32(&format!("blk.{l}.ssm_norm.weight"), d_inner)?,
                out: ld.proj(&format!("blk.{l}.ssm_out.weight"), h, d_inner)?,
                d_inner,
                d_state,
                n_group,
                n_head,
                kernel: d_conv,
                eps,
                rec_index: recurrent_lens.len(),
            };
            recurrent_lens.push((m.conv_state_len(), m.ssm_state_len()));
            Mixer::Mamba2(m)
        } else if n_ff[l] == 0 {
            let (nq, nkv) = (heads[l], kv_heads[l]);
            if nq == 0 || !nq.is_multiple_of(nkv) || (n_kv_attn != 0 && n_kv_attn != nkv) {
                return Err(format!("layer {l}: unsupported attention head counts {nq}/{nkv}"));
            }
            n_kv_attn = nkv;
            let m = PlainAttention {
                q: ld.proj(&format!("blk.{l}.attn_q.weight"), nq * hd, h)?,
                k: ld.proj(&format!("blk.{l}.attn_k.weight"), nkv * hd, h)?,
                v: ld.proj(&format!("blk.{l}.attn_v.weight"), nkv * hd, h)?,
                o: ld.proj(&format!("blk.{l}.attn_output.weight"), h, nq * hd)?,
                n_head: nq,
                n_kv: nkv,
                head_dim: hd,
                attn_index: attn_layers,
            };
            attn_layers += 1;
            Mixer::PlainAttention(m)
        } else {
            Mixer::Moe(Moe {
                router: ld.proj(&format!("blk.{l}.ffn_gate_inp.weight"), n_expert, h)?,
                probs_bias: ld.f32(&format!("blk.{l}.exp_probs_b.bias"), n_expert)?,
                up: ld.experts(&format!("blk.{l}.ffn_up_exps.weight"), n_expert, n_ff_exp, h)?,
                down: ld.experts(&format!("blk.{l}.ffn_down_exps.weight"), n_expert, h, n_ff_exp)?,
                shared_up: ld.proj(&format!("blk.{l}.ffn_up_shexp.weight"), n_ff_shexp, h)?,
                shared_down: ld.proj(&format!("blk.{l}.ffn_down_shexp.weight"), h, n_ff_shexp)?,
                n_used,
                norm_weights,
                weights_scale,
            })
        };
        layers.push(HybridLayer { attn_norm, mixer, ffn: None });
    }
    let output_name = if g.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let output = ld.table(output_name, config.vocab_size, h)?;
    let embedding = ld.table("token_embd.weight", config.vocab_size, h)?;
    let kv = KVCache::try_new_hybrid(attn_layers, context_len(config), n_kv_attn.max(1), hd, &recurrent_lens)?;
    Ok((HybridModel { layers, embedding, embedding_inverse: None, output, attn_layers, recurrent_lens }, kv))
}

/// The token embedding of a Qwen3.5 model on the host, for the GPU decoder: a row of the (block-scaled) table dequantized, and, when the table is stored in the
/// Hadamard latent space, transformed back. This is [`HybridModel::embed`] without the layers.
pub struct HostEmbedding {
    table: Proj,
    inverse: Option<Hadamard>,
}

impl HostEmbedding {
    /// Writes the embedding of `token` into `x` (`hidden_size` values).
    pub fn embed(&self, token: usize, x: &mut [f32]) {
        self.table.dequantize_row(token, x);
        if let Some(inv) = &self.inverse {
            inv.apply_inverse(x);
        }
    }
}

/// Loads the embedding table of a Qwen3.5 (`qwen35`) model for [`HostEmbedding`].
pub fn load_qwen35_embedding(g: &GgufFile, config: &ModelConfig, threads: usize) -> Result<HostEmbedding, String> {
    let hadamard = HadamardSpec::from_gguf(g)?;
    let h = config.hidden_size;
    let inverse = match &hadamard {
        Some(spec) if spec.inverse.contains_key("token_embd.weight") => Some(spec.transform_for(h)?),
        _ => None,
    };
    let ld = Loader { g, threads, hadamard, hidden: h };
    Ok(HostEmbedding { table: ld.table("token_embd.weight", config.vocab_size, h)?, inverse })
}

/// Loads the embedding table of a Nemotron-H (`nemotron_h_moe`) model for [`HostEmbedding`].
pub fn load_nemotron_embedding(g: &GgufFile, config: &ModelConfig, threads: usize) -> Result<HostEmbedding, String> {
    let h = config.hidden_size;
    let ld = Loader { g, threads, hadamard: None, hidden: h };
    Ok(HostEmbedding { table: ld.table("token_embd.weight", config.vocab_size, h)?, inverse: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed % 2_000_001) as f32 - 1_000_000.0) * 1e-6
    }

    /// `causal_conv_step` and `rank_one_update` against a plain model of their definition (with `mul_add` written out
    /// in test code), in both versions of the `multiversion!` functions.
    fn run_checks() {
        let mut seed = 0x5EED_C0DE_1234_5678u64;
        let (channels, kernel) = (37usize, 4usize);
        let km1 = kernel - 1;
        let weights: Vec<f32> = (0..channels * kernel).map(|_| rnd(&mut seed)).collect();
        let input: Vec<f32> = (0..channels).map(|_| rnd(&mut seed)).collect();
        let bias: Vec<f32> = (0..channels).map(|_| rnd(&mut seed)).collect();
        for with_bias in [false, true] {
            let state0: Vec<f32> = (0..channels * km1).map(|_| rnd(&mut seed)).collect();
            let (mut state, mut out) = (state0.clone(), vec![0.0f32; channels]);
            causal_conv_step(&mut state, &weights, kernel, &input, with_bias.then_some(&bias[..]), &mut out);
            for c in 0..channels {
                let mut sum = 0.0f32;
                for i in 0..kernel {
                    let v = if i < km1 { state0[c * km1 + i] } else { input[c] };
                    sum = v.mul_add(weights[c * kernel + i], sum);
                }
                let want = if with_bias { sum + bias[c] } else { sum };
                assert_eq!(out[c].to_bits(), want.to_bits(), "channel {c}");
                let mut moved: Vec<f32> = state0[c * km1 + 1..(c + 1) * km1].to_vec();
                moved.push(input[c]);
                assert_eq!(&state[c * km1..(c + 1) * km1], &moved[..], "state of channel {c}");
            }
        }

        let hd = 19usize;
        let k: Vec<f32> = (0..hd).map(|_| rnd(&mut seed)).collect();
        let delta: Vec<f32> = (0..hd).map(|_| rnd(&mut seed)).collect();
        let state0: Vec<f32> = (0..hd * hd).map(|_| rnd(&mut seed)).collect();
        let mut state = state0.clone();
        rank_one_update(&mut state, &k, &delta, hd);
        for j in 0..hd {
            for e in 0..hd {
                let want = k[e].mul_add(delta[j], state0[j * hd + e]);
                assert_eq!(state[j * hd + e].to_bits(), want.to_bits(), "row {j} element {e}");
            }
        }

        let w: Vec<i8> = (0..96).map(|_| (rnd(&mut seed) * 120.0) as i8).collect();
        let a: Vec<i8> = (0..96).map(|_| (rnd(&mut seed) * 120.0) as i8).collect();
        let (wd, ad) = ([0.5f32, 0.25, 0.125], [0.75f32, 1.5, 0.3]);
        let mut acc = [0.0f32; 8];
        for b in 0..3 {
            let d = wd[b] * ad[b];
            for j in 0..8 {
                let p: i32 = (4 * j..4 * j + 4).map(|k| w[32 * b + k] as i32 * a[32 * b + k] as i32).sum();
                acc[j] = d.mul_add(p as f32, acc[j]);
            }
        }
        let r = [acc[4] + acc[0], acc[5] + acc[1], acc[6] + acc[2], acc[7] + acc[3]];
        let want = (r[0] + r[2]) + (r[1] + r[3]);
        assert_eq!(dot_q8_0_scalar(&w, &wd, &a, &ad).to_bits(), want.to_bits());
    }

    #[test]
    fn recurrent_layer_helpers_follow_their_definition_in_every_version() {
        run_checks();
        #[cfg(target_arch = "x86_64")]
        crate::dispatch::with_baseline(run_checks);
    }
}
