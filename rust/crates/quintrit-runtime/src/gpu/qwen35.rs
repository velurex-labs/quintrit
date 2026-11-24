//! Hybrid Qwen3.5 (`qwen35`) models with block-scaled weights on the GPU: Gated DeltaNet layers with a recurrent state and every `full_attention_interval`-th layer a
//! gated full-attention layer, `Q2_0` / `F32` / `BF16` projections, and the Hadamard-folded weights of the Ternary Bonsai 2 models. Every kernel is bit-identical to
//! the CPU runtime (`hybrid.rs`; `quintrit-cuda`, `kernels/hybrid.cu`, `kernels/q2.cu` and the F32 attention of `kernels/layer.cu`).

use super::q2::q2_tensor;
use super::{err, GpuModel, GpuSession};
use crate::config::{ModelArchitecture, ModelConfig, Numerics, RopeMode};
use crate::hybrid::HadamardSpec;
use quintrit_cuda::{
    AttentionLayout, BlockScaledKernels, BlockScaledMatrix, DeltaNetLayout, Device, DeviceBuffer, HybridKernels, RopeKvLayout, RopeMode as GpuRopeMode,
    ValueHeadPermutation, MAX_HADAMARD_BLOCK,
};
use quintrit_format::{GgufFile, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q2_0};
use std::collections::HashMap;
use std::sync::Arc;

/// The input transform of a Hadamard-folded projection.
struct GHadamard {
    block: usize,
    signs: Option<Arc<DeviceBuffer<f32>>>,
    perm: Option<ValueHeadPermutation>,
}

enum GWeights {
    Ternary(BlockScaledMatrix),
    F32(DeviceBuffer<f32>),
    Bf16(DeviceBuffer<u16>),
}

/// A projection `rows x cols` with an optional Hadamard input transform.
struct GProj {
    w: GWeights,
    rows: usize,
    cols: usize,
    had: Option<GHadamard>,
}

/// Whether two projections read the same transformed input when they are given the same input vector: equal width and the same Hadamard transform.
fn same_input(a: &GProj, b: &GProj) -> bool {
    a.cols == b.cols
        && match (&a.had, &b.had) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                x.block == y.block
                    && x.perm == y.perm
                    && match (&x.signs, &y.signs) {
                        (None, None) => true,
                        (Some(s), Some(t)) => Arc::ptr_eq(s, t),
                        _ => false,
                    }
            }
            _ => false,
        }
}

struct GDeltaNet {
    qkv: GProj,
    z: GProj,
    beta: GProj,
    alpha: GProj,
    out: GProj,
    conv_w: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    layout: DeltaNetLayout,
    rec_index: usize,
}

struct GAttention {
    q: GProj,
    k: GProj,
    v: GProj,
    o: GProj,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    attn_index: usize,
}

enum GMixer {
    DeltaNet(GDeltaNet),
    Attention(GAttention),
}

struct GLayer {
    attn_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    mixer: GMixer,
    gate_up: GProj,
    down: GProj,
}

/// The weights of a hybrid model on the device.
pub(super) struct HybridWeights {
    layers: Vec<GLayer>,
    out_norm: DeviceBuffer<f32>,
    head: GProj,
    kernels: BlockScaledKernels,
    hy: HybridKernels,
    attn_layers: usize,
    /// Layout of every recurrent layer, in order.
    recurrent: Vec<DeltaNetLayout>,
    /// Widest input of a projection (values).
    widest: usize,
}

impl HybridWeights {
    /// Number of layers with a KV cache.
    pub(super) fn attn_layers(&self) -> usize {
        self.attn_layers
    }
}

/// The per-token buffers of a hybrid model that the common buffers of a session do not have.
pub(super) struct HybridBuffers {
    xn: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    t: DeviceBuffer<f32>,
    qs: DeviceBuffer<i8>,
    ad: DeviceBuffer<f32>,
    /// Output of the qkv projection of a Gated DeltaNet layer / the joint q and gate projection of a gated attention layer.
    pa: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    beta_raw: DeviceBuffer<f32>,
    alpha_raw: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    conv_out: DeviceBuffer<f32>,
    recur: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    /// q | k | v of a gated attention layer.
    qkv: DeviceBuffer<f32>,
    gu: DeviceBuffer<f32>,
    inter: DeviceBuffer<f32>,
    conv_states: Vec<DeviceBuffer<f32>>,
    ssm_states: Vec<DeviceBuffer<f32>>,
}

impl HybridBuffers {
    pub(super) fn new(device: &Device, config: &ModelConfig, w: &HybridWeights) -> Result<HybridBuffers, String> {
        let (hd, nq, nkv) = (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
        let (nv, ch, value_dim) = w
            .recurrent
            .iter()
            .fold((1, 1, 1), |(a, b, c), l| (a.max(l.v_heads), b.max(l.channels()), c.max(l.v_heads * l.head_dim)));
        let widest = w.widest;
        let alloc = |n: usize| device.alloc_zeroed::<f32>(n).map_err(err);
        let mut conv_states = Vec::new();
        let mut ssm_states = Vec::new();
        for l in &w.recurrent {
            conv_states.push(alloc(l.conv_state_len())?);
            ssm_states.push(alloc(l.ssm_state_len())?);
        }
        Ok(HybridBuffers {
            xn: alloc(config.hidden_size)?,
            mix: alloc(config.hidden_size)?,
            t: alloc(widest)?,
            qs: device.alloc_zeroed::<i8>(widest).map_err(err)?,
            ad: alloc(widest / 32)?,
            pa: alloc(ch.max(2 * nq * hd))?,
            z: alloc(value_dim)?,
            beta_raw: alloc(nv)?,
            alpha_raw: alloc(nv)?,
            beta: alloc(nv)?,
            gate: alloc(nv)?,
            conv_out: alloc(ch)?,
            recur: alloc(value_dim)?,
            gated: alloc(value_dim)?,
            qkv: alloc((nq + 2 * nkv) * hd)?,
            gu: alloc(2 * config.intermediate_size)?,
            inter: alloc(config.intermediate_size)?,
            conv_states,
            ssm_states,
        })
    }
}

/// Checks that the GPU decoder can run this Qwen3.5 model; the error says why not.
pub(super) fn check_supported(config: &ModelConfig, gguf: &GgufFile) -> Result<(), String> {
    if config.architecture != ModelArchitecture::Qwen35 {
        return Err("not a Qwen3.5 model".into());
    }
    if config.numerics != Numerics::LlamaCpp {
        return Err("block-scaled models follow the llama.cpp numerics".into());
    }
    if !matches!(config.head_dim, 32 | 64 | 128 | 256) {
        return Err(format!("attention head dimension {} is not supported on the GPU (32, 64, 128 or 256)", config.head_dim));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err("the number of query heads must be a multiple of the number of key-value heads".into());
    }
    if config.hidden_size % 64 != 0 || config.intermediate_size % 64 != 0 {
        return Err("the hidden and intermediate sizes must be multiples of 64".into());
    }
    match gguf.tensors.get("token_embd.weight").map(|t| t.tensor_type) {
        Some(GGML_TYPE_Q2_0) => {}
        _ => return Err("token_embd.weight is not Q2_0; the GPU decoder needs a block-scaled embedding".into()),
    }
    if let Some(t) = gguf.tensors.get("output.weight") {
        if t.tensor_type != GGML_TYPE_Q2_0 {
            return Err("output.weight is not Q2_0; the GPU decoder needs a block-scaled head".into());
        }
    }
    Ok(())
}

struct Ssm {
    d_conv: usize,
    d_state: usize,
    n_group: usize,
    dt_rank: usize,
    interval: usize,
}

fn ssm_parameters(g: &GgufFile) -> Result<Ssm, String> {
    let key = |k: &str| g.metadata.get(&format!("qwen35.{k}"));
    let num = |k: &str| key(k).and_then(|v| v.as_usize()).ok_or_else(|| format!("metadata 'qwen35.{k}' missing"));
    let s = Ssm {
        d_conv: num("ssm.conv_kernel")?,
        d_state: num("ssm.state_size")?,
        n_group: num("ssm.group_count")?,
        dt_rank: num("ssm.time_step_rank")?,
        interval: key("full_attention_interval").and_then(|v| v.as_usize()).unwrap_or(4),
    };
    let d_inner = num("ssm.inner_size")?;
    if s.d_conv < 2 || s.d_conv > 8 || s.d_state == 0 || s.n_group == 0 || s.dt_rank == 0 || s.dt_rank % s.n_group != 0 || d_inner != s.dt_rank * s.d_state || s.interval == 0 {
        return Err("inconsistent qwen35.ssm.* hyper-parameters".into());
    }
    if s.d_state % 32 != 0 || s.d_state > 256 {
        return Err(format!("the state size {} of the Gated DeltaNet must be a multiple of 32 up to 256", s.d_state));
    }
    Ok(s)
}

struct Loader<'a> {
    g: &'a GgufFile,
    device: &'a Device,
    spec: Option<HadamardSpec>,
    signs: std::cell::RefCell<HashMap<usize, Arc<DeviceBuffer<f32>>>>,
}

impl Loader<'_> {
    fn f32(&self, name: &str, len: usize) -> Result<DeviceBuffer<f32>, String> {
        let v = self.g.get_tensor_f32(name).ok_or_else(|| format!("Missing F32 tensor {name}"))?;
        if v.len() != len {
            return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
        }
        self.device.upload(&v).map_err(err)
    }

    fn hadamard(&self, name: &str, cols: usize) -> Result<Option<GHadamard>, String> {
        let Some(spec) = &self.spec else { return Ok(None) };
        if !spec.weights.contains_key(name) {
            return Ok(None);
        }
        let t = spec.transform_for(cols)?;
        if t.block > MAX_HADAMARD_BLOCK {
            return Err(format!("the Hadamard block of {} values is larger than the GPU kernel supports ({MAX_HADAMARD_BLOCK})", t.block));
        }
        let signs = match &t.signs {
            None => None,
            Some(host) => {
                let mut cache = self.signs.borrow_mut();
                if let Some(d) = cache.get(&cols) {
                    Some(d.clone())
                } else {
                    let d = Arc::new(self.device.upload(host.as_slice()).map_err(err)?);
                    cache.insert(cols, d.clone());
                    Some(d)
                }
            }
        };
        Ok(Some(GHadamard { block: t.block, signs, perm: None }))
    }

    /// One tensor as a projection.
    fn proj(&self, name: &str, rows: usize, cols: usize) -> Result<GProj, String> {
        let info = self.g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
        if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows {
            return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
        }
        let w = match info.tensor_type {
            GGML_TYPE_Q2_0 => {
                let (packed, scales) = q2_tensor(self.g, name, rows, cols)?;
                GWeights::Ternary(BlockScaledMatrix::new(self.device, &packed, &scales, rows, cols).map_err(err)?)
            }
            GGML_TYPE_F32 => {
                let data = self.g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
                let v: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                GWeights::F32(self.device.upload(&v).map_err(err)?)
            }
            GGML_TYPE_BF16 => {
                let data = self.g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
                let v: Vec<u16> = data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                GWeights::Bf16(self.device.upload(&v).map_err(err)?)
            }
            other => return Err(format!("tensor '{name}' has type {other}, which the GPU decoder does not support")),
        };
        if !matches!(w, GWeights::Ternary(_)) && cols % 32 != 0 {
            return Err(format!("tensor '{name}': an F32 or BF16 projection needs a column count that is a multiple of 32"));
        }
        Ok(GProj { w, rows, cols, had: self.hadamard(name, cols)? })
    }

    /// The gate and up projections of a feed-forward block as one stacked matrix (they read the same transformed input).
    fn gate_up(&self, l: usize, n_ff: usize, hidden: usize) -> Result<GProj, String> {
        let (gate_name, up_name) = (format!("blk.{l}.ffn_gate.weight"), format!("blk.{l}.ffn_up.weight"));
        let (hg, hu) = (self.hadamard(&gate_name, hidden)?, self.hadamard(&up_name, hidden)?);
        let same = match (&hg, &hu) {
            (None, None) => true,
            (Some(a), Some(b)) => a.block == b.block && a.perm == b.perm && a.signs.is_some() == b.signs.is_some(),
            _ => false,
        };
        if !same {
            return Err(format!("blk.{l}: the gate and up projections have different Hadamard transforms"));
        }
        let (mut packed, mut scales) = (Vec::new(), Vec::new());
        for name in [&gate_name, &up_name] {
            let (p, s) = q2_tensor(self.g, name, n_ff, hidden)?;
            packed.extend_from_slice(&p);
            scales.extend_from_slice(&s);
        }
        let m = BlockScaledMatrix::new(self.device, &packed, &scales, 2 * n_ff, hidden).map_err(err)?;
        Ok(GProj { w: GWeights::Ternary(m), rows: 2 * n_ff, cols: hidden, had: hg })
    }
}

/// Uploads the weights of a Qwen3.5 model.
pub(super) fn load(gguf: &GgufFile, config: &ModelConfig, device: &Device) -> Result<HybridWeights, String> {
    let ssm = ssm_parameters(gguf)?;
    let spec = HadamardSpec::from_gguf(gguf)?;
    let gdn_v_grouped = spec.as_ref().is_some_and(|s| s.gdn_v_grouped);
    let ld = Loader { g: gguf, device, spec, signs: Default::default() };
    let h = config.hidden_size;
    let hd = config.head_dim;
    let (nq, nkv) = (config.num_attention_heads, config.num_key_value_heads);
    let key_dim = ssm.n_group * ssm.d_state;
    let value_dim = ssm.dt_rank * ssm.d_state;
    let layout = DeltaNetLayout { k_heads: ssm.n_group, v_heads: ssm.dt_rank, head_dim: ssm.d_state, kernel: ssm.d_conv };
    let kernels = BlockScaledKernels::new(device).map_err(err)?;
    let hy = HybridKernels::new(device).map_err(err)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    let mut recurrent = Vec::new();
    let mut attn_layers = 0;
    let mut widest = h.max(config.intermediate_size);
    for l in 0..config.num_hidden_layers {
        let attn_norm = ld.f32(&format!("blk.{l}.attn_norm.weight"), h)?;
        let post_norm = ld.f32(&format!("blk.{l}.post_attention_norm.weight"), h)?;
        let mixer = if (l + 1) % ssm.interval != 0 {
            let mut out = ld.proj(&format!("blk.{l}.ssm_out.weight"), h, value_dim)?;
            if gdn_v_grouped {
                if let Some(t) = &mut out.had {
                    t.perm = Some(ValueHeadPermutation { head_dim: value_dim / ssm.dt_rank, k_heads: ssm.n_group, repeats: ssm.dt_rank / ssm.n_group });
                }
            }
            let m = GDeltaNet {
                qkv: ld.proj(&format!("blk.{l}.attn_qkv.weight"), 2 * key_dim + value_dim, h)?,
                z: ld.proj(&format!("blk.{l}.attn_gate.weight"), value_dim, h)?,
                beta: ld.proj(&format!("blk.{l}.ssm_beta.weight"), ssm.dt_rank, h)?,
                alpha: ld.proj(&format!("blk.{l}.ssm_alpha.weight"), ssm.dt_rank, h)?,
                conv_w: ld.f32(&format!("blk.{l}.ssm_conv1d.weight"), ssm.d_conv * (2 * key_dim + value_dim))?,
                dt_bias: ld.f32(&format!("blk.{l}.ssm_dt.bias"), ssm.dt_rank)?,
                a: ld.f32(&format!("blk.{l}.ssm_a"), ssm.dt_rank)?,
                norm: ld.f32(&format!("blk.{l}.ssm_norm.weight"), ssm.d_state)?,
                out,
                layout,
                rec_index: recurrent.len(),
            };
            recurrent.push(layout);
            widest = widest.max(value_dim);
            GMixer::DeltaNet(m)
        } else {
            let m = GAttention {
                q: ld.proj(&format!("blk.{l}.attn_q.weight"), 2 * nq * hd, h)?,
                k: ld.proj(&format!("blk.{l}.attn_k.weight"), nkv * hd, h)?,
                v: ld.proj(&format!("blk.{l}.attn_v.weight"), nkv * hd, h)?,
                o: ld.proj(&format!("blk.{l}.attn_output.weight"), h, nq * hd)?,
                q_norm: ld.f32(&format!("blk.{l}.attn_q_norm.weight"), hd)?,
                k_norm: ld.f32(&format!("blk.{l}.attn_k_norm.weight"), hd)?,
                attn_index: attn_layers,
            };
            attn_layers += 1;
            widest = widest.max(nq * hd);
            GMixer::Attention(m)
        };
        let gate_up = ld.gate_up(l, config.intermediate_size, h)?;
        let down = ld.proj(&format!("blk.{l}.ffn_down.weight"), h, config.intermediate_size)?;
        layers.push(GLayer { attn_norm, post_norm, mixer, gate_up, down });
    }
    let out_norm = ld.f32("output_norm.weight", h)?;
    let head_name = if gguf.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let head = ld.proj(head_name, config.vocab_size, h)?;
    if !matches!(head.w, GWeights::Ternary(_)) {
        return Err("the output head must be block-scaled".into());
    }
    Ok(HybridWeights { layers, out_norm, head, kernels, hy, attn_layers, recurrent, widest })
}

impl GpuSession {
    /// One projection: the Hadamard transform of the input if the weights are folded, then the product (a block-scaled matrix quantizes its input to `Q8_0` first).
    /// `prev` is the projection that was enqueued just before **with the same input vector `x`** (or `None`): when it has the same transform, its transformed and
    /// quantized input in the scratch buffers is reused instead of computing it again (the same operations on the same values, so the same bits).
    fn q35_proj(&self, p: &GProj, prev: Option<&GProj>, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, out_off: usize) -> quintrit_cuda::Result<()> {
        let w = self.model.hybrid.as_ref().expect("a hybrid model");
        let b = self.hy.as_ref().expect("the buffers of a hybrid model");
        let st = &self.stream;
        let shared = prev.filter(|q| same_input(q, p));
        let src = match &p.had {
            Some(h) => {
                if shared.is_none() {
                    w.hy.hadamard(st, x, &b.t, p.cols, h.block, h.signs.as_deref(), h.perm)?;
                }
                &b.t
            }
            None => x,
        };
        match &p.w {
            GWeights::Ternary(m) => {
                if !shared.is_some_and(|q| matches!(q.w, GWeights::Ternary(_))) {
                    w.kernels.quantize(st, src, &b.qs, &b.ad, p.cols)?;
                }
                w.kernels.gemv_into(st, m, &b.qs, &b.ad, out, out_off)
            }
            GWeights::F32(m) => w.hy.gemv_f32(st, m, src, out, out_off, p.rows, p.cols),
            GWeights::Bf16(m) => w.hy.gemv_bf16(st, m, src, out, out_off, p.rows, p.cols),
        }
    }

    /// Records the device work of one token of a hybrid model (the position comes from device memory).
    pub(super) fn enqueue_qwen35(&self, head: bool) -> quintrit_cuda::Result<()> {
        let model: &GpuModel = &self.model;
        let w = model.hybrid.as_ref().expect("a hybrid model");
        let q = self.hy.as_ref().expect("the buffers of a hybrid model");
        let k = &w.kernels;
        let c = &model.cfg;
        let st = &self.stream;
        let b = &self.buf;
        let (hidden, inter, hd) = (c.hidden_size, c.intermediate_size, c.head_dim);
        let (nq, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let (q_dim, kv_dim) = (nq * hd, nkv * hd);
        let eps = c.rms_norm_eps;
        let mode = match c.rope.mode {
            RopeMode::Normal => GpuRopeMode::Interleaved,
            RopeMode::Neox => GpuRopeMode::HalfSplit,
        };
        let attn_layout = AttentionLayout { q_heads: nq, kv_heads: nkv, head_dim: hd, cache_len: self.context, max_len: self.context };
        let rope_layout = RopeKvLayout { q_heads: nq, kv_heads: nkv, head_dim: hd, rope_dims: model.rope_dims, mode, cache_len: self.context };

        for layer in &w.layers {
            k.rms_norm_rows(st, &b.h, 0, &layer.attn_norm, &q.xn, 0, 1, hidden, eps)?;
            match &layer.mixer {
                GMixer::DeltaNet(m) => {
                    let l = m.layout;
                    self.q35_proj(&m.qkv, None, &q.xn, &q.pa, 0)?;
                    self.q35_proj(&m.z, Some(&m.qkv), &q.xn, &q.z, 0)?;
                    self.q35_proj(&m.beta, Some(&m.z), &q.xn, &q.beta_raw, 0)?;
                    self.q35_proj(&m.alpha, Some(&m.beta), &q.xn, &q.alpha_raw, 0)?;
                    w.hy.gates(st, &q.beta_raw, &q.alpha_raw, &m.dt_bias, &m.a, &q.beta, &q.gate, l.v_heads)?;
                    w.hy.conv(st, l, &q.pa, &q.conv_states[m.rec_index], &m.conv_w, &q.conv_out, &b.pos)?;
                    w.hy.l2_norm(st, l, &q.conv_out, eps)?;
                    w.hy.recurrence(st, l, &q.ssm_states[m.rec_index], &q.conv_out, &q.beta, &q.gate, &q.recur, &b.pos)?;
                    w.hy.gated_norm(st, l, &q.recur, &m.norm, &q.z, &q.gated, eps)?;
                    self.q35_proj(&m.out, None, &q.gated, &q.mix, 0)?;
                }
                GMixer::Attention(m) => {
                    self.q35_proj(&m.q, None, &q.xn, &q.pa, 0)?;
                    self.q35_proj(&m.k, Some(&m.q), &q.xn, &q.qkv, q_dim)?;
                    self.q35_proj(&m.v, Some(&m.k), &q.xn, &q.qkv, q_dim + kv_dim)?;
                    k.rms_norm_strided(st, &q.pa, 0, 2 * hd, &m.q_norm, &q.qkv, 0, hd, nq, hd, eps)?;
                    k.rms_norm_rows(st, &q.qkv, q_dim, &m.k_norm, &q.qkv, q_dim, nkv, hd, eps)?;
                    k.rope_kv(st, rope_layout, &q.qkv, &model.cos_sin, &b.pos, &b.q, &self.k_cache[m.attn_index], &self.v_cache[m.attn_index])?;
                    model.layer.attention_f32(
                        st,
                        attn_layout,
                        model.attn_scale,
                        &self.k_cache[m.attn_index],
                        &self.v_cache[m.attn_index],
                        &b.q,
                        &b.attn,
                        &b.pos,
                        &b.attn_ws,
                    )?;
                    w.hy.attention_gate(st, &b.attn, &q.pa, nq, hd)?;
                    self.q35_proj(&m.o, None, &b.attn, &q.mix, 0)?;
                }
            }
            k.add(st, &b.h, &q.mix, hidden)?;
            k.rms_norm_rows(st, &b.h, 0, &layer.post_norm, &q.xn, 0, 1, hidden, eps)?;
            self.q35_proj(&layer.gate_up, None, &q.xn, &q.gu, 0)?;
            k.swiglu(st, &q.gu, &q.inter, inter)?;
            self.q35_proj(&layer.down, None, &q.inter, &q.mix, 0)?;
            k.add(st, &b.h, &q.mix, hidden)?;
        }
        if !head {
            return Ok(());
        }
        k.rms_norm_rows(st, &b.h, 0, &w.out_norm, &q.xn, 0, 1, hidden, eps)?;
        self.q35_proj(&w.head, None, &q.xn, &b.logits, 0)
    }
}
