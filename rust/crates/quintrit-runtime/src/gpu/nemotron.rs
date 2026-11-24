//! Nemotron-H (`nemotron_h_moe`) models on the GPU: Mamba-2 layers with a recurrent state, attention layers without rotary embedding, mixture-of-experts layers with
//! block-scaled experts (the router and the selection of the experts run on the device, so the CUDA graph of a token stays one graph) and a `Q8_0` output head. Every
//! kernel is bit-identical to the CPU runtime (`hybrid.rs`; `quintrit-cuda`, `kernels/nemotron.cu`, `kernels/q2.cu`, `kernels/hybrid.cu` and the F32 attention of
//! `kernels/layer.cu`).

use super::q2::q2_tensor;
use super::{err, GpuModel, GpuSession};
use crate::config::{ModelArchitecture, ModelConfig, Numerics};
use crate::hybrid::per_layer;
use quintrit_cuda::{
    AttentionLayout, BlockScaledExperts, BlockScaledKernels, BlockScaledMatrix, Device, DeviceBuffer, HybridKernels, MambaLayout, NemotronKernels, Q8Matrix,
};
use quintrit_format::{decode_q2_0_packed, GgufFile, GGML_TYPE_F32, GGML_TYPE_Q2_0, GGML_TYPE_Q8_0};

struct NMamba {
    in_proj: BlockScaledMatrix,
    out: BlockScaledMatrix,
    conv_w: DeviceBuffer<f32>,
    conv_b: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    d: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    rec_index: usize,
}

struct NAttention {
    qkv: BlockScaledMatrix,
    o: BlockScaledMatrix,
    attn_index: usize,
}

struct NMoe {
    router: DeviceBuffer<f32>,
    probs_bias: DeviceBuffer<f32>,
    up: BlockScaledExperts,
    down: BlockScaledExperts,
    shared_up: BlockScaledMatrix,
    shared_down: BlockScaledMatrix,
}

enum NMixer {
    Mamba(NMamba),
    Attention(NAttention),
    Moe(NMoe),
}

struct NLayer {
    attn_norm: DeviceBuffer<f32>,
    mixer: NMixer,
}

enum NHead {
    Q8(Q8Matrix),
    Q2(BlockScaledMatrix),
}

/// The weights of a Nemotron-H model on the device.
pub(super) struct NemotronWeights {
    layers: Vec<NLayer>,
    out_norm: DeviceBuffer<f32>,
    head: NHead,
    kernels: BlockScaledKernels,
    hy: HybridKernels,
    nem: NemotronKernels,
    attn_layers: usize,
    mamba: MambaLayout,
    recurrent: usize,
    n_experts: usize,
    n_used: usize,
    norm_weights: bool,
    weights_scale: f32,
    n_ff_exp: usize,
    n_ff_shexp: usize,
    n_head: usize,
    n_kv: usize,
}

impl NemotronWeights {
    /// Number of layers with a KV cache.
    pub(super) fn attn_layers(&self) -> usize {
        self.attn_layers
    }

}

/// The per-token buffers of a Nemotron-H model that the common buffers of a session do not have.
pub(super) struct NemotronBuffers {
    xn: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    qs: DeviceBuffer<i8>,
    ad: DeviceBuffer<f32>,
    qs2: DeviceBuffer<i8>,
    ad2: DeviceBuffer<f32>,
    pa: DeviceBuffer<f32>,
    conv_out: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    a_up: DeviceBuffer<f32>,
    heads: DeviceBuffer<f32>,
    sh_a: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    chosen: DeviceBuffer<i32>,
    weights: DeviceBuffer<f32>,
    conv_states: Vec<DeviceBuffer<f32>>,
    ssm_states: Vec<DeviceBuffer<f32>>,
}

impl NemotronBuffers {
    pub(super) fn new(device: &Device, config: &ModelConfig, w: &NemotronWeights) -> Result<NemotronBuffers, String> {
        let hd = config.head_dim;
        let (nq, nkv) = (w.n_head, w.n_kv);
        let di = w.mamba.d_inner;
        let hidden = config.hidden_size;
        let moe_widest = (w.n_used * w.n_ff_exp).max(w.n_ff_shexp);
        let widest = hidden.max(di).max(nq * hd).max(moe_widest);
        let alloc = |n: usize| device.alloc_zeroed::<f32>(n.max(1)).map_err(err);
        let mut conv_states = Vec::new();
        let mut ssm_states = Vec::new();
        for _ in 0..w.recurrent {
            conv_states.push(alloc(w.mamba.conv_state_len())?);
            ssm_states.push(alloc(w.mamba.ssm_state_len())?);
        }
        Ok(NemotronBuffers {
            xn: alloc(hidden)?,
            mix: alloc(hidden)?,
            qs: device.alloc_zeroed::<i8>(widest).map_err(err)?,
            ad: alloc(widest / 32)?,
            qs2: device.alloc_zeroed::<i8>(widest).map_err(err)?,
            ad2: alloc(widest / 32)?,
            pa: alloc(w.mamba.in_proj_rows())?,
            conv_out: alloc(w.mamba.channels())?,
            y: alloc(di)?,
            gated: alloc(di)?,
            normed: alloc(di)?,
            qkv: alloc((nq + 2 * nkv) * hd)?,
            a_up: alloc(w.n_used * w.n_ff_exp)?,
            heads: alloc(w.n_used * hidden)?,
            sh_a: alloc(w.n_ff_shexp)?,
            shared: alloc(hidden)?,
            logits: alloc(w.n_experts)?,
            chosen: device.alloc_zeroed::<i32>(w.n_used.max(1)).map_err(err)?,
            weights: alloc(w.n_used)?,
            conv_states,
            ssm_states,
        })
    }
}

/// Checks that the GPU decoder can run this Nemotron-H model; the error says why not.
pub(super) fn check_supported(config: &ModelConfig, gguf: &GgufFile) -> Result<(), String> {
    if config.architecture != ModelArchitecture::NemotronHMoe {
        return Err("not a Nemotron-H model".into());
    }
    if config.numerics != Numerics::LlamaCpp {
        return Err("block-scaled models follow the llama.cpp numerics".into());
    }
    if !matches!(config.head_dim, 32 | 64 | 128 | 256) {
        return Err(format!("attention head dimension {} is not supported on the GPU (32, 64, 128 or 256)", config.head_dim));
    }
    if config.hidden_size % 64 != 0 {
        return Err("the hidden size must be a multiple of 64".into());
    }
    match gguf.tensors.get("token_embd.weight").map(|t| t.tensor_type) {
        Some(GGML_TYPE_Q8_0) | Some(GGML_TYPE_Q2_0) => {}
        _ => return Err("token_embd.weight is neither Q8_0 nor Q2_0".into()),
    }
    match gguf.tensors.get("output.weight").or_else(|| gguf.tensors.get("token_embd.weight")).map(|t| t.tensor_type) {
        Some(GGML_TYPE_Q8_0) | Some(GGML_TYPE_Q2_0) => {}
        _ => return Err("the output head is neither Q8_0 nor Q2_0".into()),
    }
    Ok(())
}

fn upload_f32(device: &Device, g: &GgufFile, name: &str, len: usize) -> Result<DeviceBuffer<f32>, String> {
    let v = g.get_tensor_f32(name).ok_or_else(|| format!("Missing F32 tensor {name}"))?;
    if v.len() != len {
        return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
    }
    device.upload(&v).map_err(err)
}

fn q2_matrix(device: &Device, g: &GgufFile, names: &[String], rows: &[usize], cols: usize) -> Result<BlockScaledMatrix, String> {
    let (mut packed, mut scales) = (Vec::new(), Vec::new());
    for (name, &r) in names.iter().zip(rows) {
        let (p, s) = q2_tensor(g, name, r, cols)?;
        packed.extend_from_slice(&p);
        scales.extend_from_slice(&s);
    }
    BlockScaledMatrix::new(device, &packed, &scales, rows.iter().sum(), cols).map_err(err)
}

/// The `n` experts of a 3-D tensor `[cols, rows, n]`.
fn experts(device: &Device, g: &GgufFile, name: &str, n: usize, rows: usize, cols: usize) -> Result<BlockScaledExperts, String> {
    let info = g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
    if info.shape != [cols, rows, n] || info.tensor_type != GGML_TYPE_Q2_0 {
        return Err(format!("tensor '{name}' has shape {:?} and type {}, expected Q2_0 [{cols}, {rows}, {n}]", info.shape, info.tensor_type));
    }
    let block = info.block_size.ok_or_else(|| format!("{name}: unknown Q2_0 block size"))?;
    let data = g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
    let per = data.len() / n;
    let (mut packed, mut scales) = (Vec::new(), Vec::new());
    for e in 0..n {
        let (p, s) = decode_q2_0_packed(&data[e * per..(e + 1) * per], rows, cols, block).map_err(|x| format!("{name}[{e}]: {x}"))?;
        packed.extend_from_slice(&p);
        for v in s {
            let h = crate::ops::f32_to_f16(v);
            if crate::ops::f16_to_f32(h).to_bits() != v.to_bits() {
                return Err(format!("tensor '{name}' has a scale that F16 cannot hold exactly"));
            }
            scales.push(h);
        }
    }
    BlockScaledExperts::new(device, &packed, &scales, n, rows, cols).map_err(err)
}

/// The output head: a `Q8_0` table or a block-scaled one.
fn head(device: &Device, g: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<NHead, String> {
    let info = g.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
    if info.shape != [cols, rows] {
        return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
    }
    match info.tensor_type {
        GGML_TYPE_Q8_0 => {
            let data = g.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
            let mut qs = Vec::with_capacity(rows * cols);
            let mut d = Vec::with_capacity(rows * cols / 32);
            for b in data.chunks_exact(34) {
                d.push(u16::from_le_bytes([b[0], b[1]]));
                qs.extend(b[2..].iter().map(|&v| v as i8));
            }
            Ok(NHead::Q8(Q8Matrix::new(device, &qs, &d, rows, cols).map_err(err)?))
        }
        _ => Ok(NHead::Q2(q2_matrix(device, g, &[name.to_string()], &[rows], cols)?)),
    }
}

/// Uploads the weights of a Nemotron-H model.
pub(super) fn load(gguf: &GgufFile, config: &ModelConfig, device: &Device) -> Result<NemotronWeights, String> {
    let arch = "nemotron_h_moe";
    let num = |k: &str| gguf.metadata.get(&format!("{arch}.{k}")).and_then(|v| v.as_usize()).ok_or_else(|| format!("metadata '{arch}.{k}' missing"));
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
    let norm_weights = gguf.metadata.get(&format!("{arch}.expert_weights_norm")).and_then(|v| v.as_bool()).unwrap_or(false);
    let weights_scale = gguf.metadata.get(&format!("{arch}.expert_weights_scale")).and_then(|v| v.as_f32()).unwrap_or(1.0);
    if d_conv < 2 || d_conv > 8 || n_group == 0 || n_head == 0 || n_head % n_group != 0 || d_inner % n_head != 0 || d_inner % n_group != 0 || n_used == 0 || n_used > n_expert {
        return Err("inconsistent nemotron_h_moe hyper-parameters".into());
    }
    if d_state % 32 != 0 || d_state > 256 || n_expert > 256 || n_used > 16 {
        return Err("the GPU decoder supports a Mamba-2 state size that is a multiple of 32 up to 256, at most 256 experts and 16 used experts".into());
    }
    if n_ff_exp % 64 != 0 || n_ff_shexp % 64 != 0 || d_inner % 64 != 0 || d_inner % 8 != 0 {
        return Err("the expert and Mamba-2 widths must be multiples of 64".into());
    }
    let kv_heads = per_layer(gguf, &format!("{arch}.attention.head_count_kv"), n_layer)?;
    let n_ff = per_layer(gguf, &format!("{arch}.feed_forward_length"), n_layer)?;
    let heads = per_layer(gguf, &format!("{arch}.attention.head_count"), n_layer)?;
    let h = config.hidden_size;
    let hd = config.head_dim;
    let kernels = BlockScaledKernels::new(device).map_err(err)?;
    let hy = HybridKernels::new(device).map_err(err)?;
    let nem = NemotronKernels::new(device).map_err(err)?;
    let mamba = MambaLayout { d_inner, d_state, n_group, n_head, kernel: d_conv };

    let mut layers = Vec::with_capacity(n_layer);
    let mut recurrent = 0;
    let mut attn_layers = 0;
    let (mut nq_attn, mut nkv_attn) = (0usize, 0usize);
    for l in 0..n_layer {
        let name = |n: &str| format!("blk.{l}.{n}.weight");
        let attn_norm = upload_f32(device, gguf, &name("attn_norm"), h)?;
        let mixer = if kv_heads[l] == 0 && n_ff[l] == 0 {
            let ch = mamba.channels();
            let m = NMamba {
                in_proj: q2_matrix(device, gguf, &[name("ssm_in")], &[mamba.in_proj_rows()], h)?,
                out: q2_matrix(device, gguf, &[name("ssm_out")], &[h], d_inner)?,
                conv_w: upload_f32(device, gguf, &name("ssm_conv1d"), d_conv * ch)?,
                conv_b: upload_f32(device, gguf, &format!("blk.{l}.ssm_conv1d.bias"), ch)?,
                dt_bias: upload_f32(device, gguf, &format!("blk.{l}.ssm_dt.bias"), n_head)?,
                a: upload_f32(device, gguf, &format!("blk.{l}.ssm_a"), n_head)?,
                d: upload_f32(device, gguf, &format!("blk.{l}.ssm_d"), n_head)?,
                norm: upload_f32(device, gguf, &name("ssm_norm"), d_inner)?,
                rec_index: recurrent,
            };
            recurrent += 1;
            NMixer::Mamba(m)
        } else if n_ff[l] == 0 {
            let (nq, nkv) = (heads[l], kv_heads[l]);
            if nq == 0 || nq % nkv != 0 || (nkv_attn != 0 && (nkv_attn != nkv || nq_attn != nq)) {
                return Err(format!("layer {l}: unsupported attention head counts {nq}/{nkv}"));
            }
            (nq_attn, nkv_attn) = (nq, nkv);
            let m = NAttention {
                qkv: q2_matrix(device, gguf, &[name("attn_q"), name("attn_k"), name("attn_v")], &[nq * hd, nkv * hd, nkv * hd], h)?,
                o: q2_matrix(device, gguf, &[name("attn_output")], &[h], nq * hd)?,
                attn_index: attn_layers,
            };
            attn_layers += 1;
            NMixer::Attention(m)
        } else {
            let router_name = name("ffn_gate_inp");
            let info = gguf.tensors.get(&router_name).ok_or_else(|| format!("Missing {router_name}"))?;
            if info.tensor_type != GGML_TYPE_F32 || info.shape != [h, n_expert] {
                return Err(format!("{router_name}: the router must be an F32 matrix [{h}, {n_expert}]"));
            }
            if h % 32 != 0 {
                return Err("the hidden size must be a multiple of 32".into());
            }
            let m = NMoe {
                router: upload_f32(device, gguf, &router_name, n_expert * h)?,
                probs_bias: upload_f32(device, gguf, &format!("blk.{l}.exp_probs_b.bias"), n_expert)?,
                up: experts(device, gguf, &format!("blk.{l}.ffn_up_exps.weight"), n_expert, n_ff_exp, h)?,
                down: experts(device, gguf, &format!("blk.{l}.ffn_down_exps.weight"), n_expert, h, n_ff_exp)?,
                shared_up: q2_matrix(device, gguf, &[name("ffn_up_shexp")], &[n_ff_shexp], h)?,
                shared_down: q2_matrix(device, gguf, &[name("ffn_down_shexp")], &[h], n_ff_shexp)?,
            };
            NMixer::Moe(m)
        };
        layers.push(NLayer { attn_norm, mixer });
    }
    let out_norm = upload_f32(device, gguf, "output_norm.weight", h)?;
    let head_name = if gguf.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let head = head(device, gguf, head_name, config.vocab_size, h)?;
    Ok(NemotronWeights {
        layers,
        out_norm,
        head,
        kernels,
        hy,
        nem,
        attn_layers,
        mamba,
        recurrent,
        n_experts: n_expert,
        n_used,
        norm_weights,
        weights_scale,
        n_ff_exp,
        n_ff_shexp,
        n_head: nq_attn.max(1),
        n_kv: nkv_attn.max(1),
    })
}

impl GpuSession {
    /// Records the device work of one token of a Nemotron-H model (the position comes from device memory).
    pub(super) fn enqueue_nemotron(&self, head: bool) -> quintrit_cuda::Result<()> {
        let model: &GpuModel = &self.model;
        let w = model.nemotron.as_ref().expect("a Nemotron-H model");
        let q = self.nm.as_ref().expect("the buffers of a Nemotron-H model");
        let (k, nem, hy) = (&w.kernels, &w.nem, &w.hy);
        let c = &model.cfg;
        let st = &self.stream;
        let b = &self.buf;
        let hidden = c.hidden_size;
        let hd = c.head_dim;
        let (nq, nkv) = (w.n_head, w.n_kv);
        let q_dim = nq * hd;
        let eps = c.rms_norm_eps;
        let ml = w.mamba;
        let di = ml.d_inner;
        let gsz = di / ml.n_group;
        let (nf, n_used) = (w.n_ff_exp, w.n_used);
        let attn_layout = AttentionLayout { q_heads: nq, kv_heads: nkv, head_dim: hd, cache_len: self.context, max_len: self.context };

        for layer in &w.layers {
            k.rms_norm_rows(st, &b.h, 0, &layer.attn_norm, &q.xn, 0, 1, hidden, eps)?;
            match &layer.mixer {
                NMixer::Mamba(m) => {
                    k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
                    k.gemv(st, &m.in_proj, &q.qs, &q.ad, &q.pa)?;
                    nem.conv(st, ml, &q.pa, &q.conv_states[m.rec_index], &m.conv_w, &m.conv_b, &q.conv_out, &b.pos)?;
                    nem.scan(st, ml, &q.ssm_states[m.rec_index], &q.conv_out, &q.pa, &m.dt_bias, &m.a, &m.d, &q.y, &b.pos)?;
                    nem.swiglu(st, &q.pa, &q.y, &q.gated, di)?;
                    k.rms_norm_grouped(st, &q.gated, 0, gsz, &m.norm, gsz, &q.normed, 0, gsz, ml.n_group, gsz, eps)?;
                    k.quantize(st, &q.normed, &q.qs, &q.ad, di)?;
                    k.gemv(st, &m.out, &q.qs, &q.ad, &q.mix)?;
                }
                NMixer::Attention(m) => {
                    k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
                    k.gemv(st, &m.qkv, &q.qs, &q.ad, &q.qkv)?;
                    nem.kv_append(st, &q.qkv, &b.pos, &self.k_cache[m.attn_index], &self.v_cache[m.attn_index], nkv, hd, self.context, q_dim)?;
                    model.layer.attention_f32(
                        st,
                        attn_layout,
                        model.attn_scale,
                        &self.k_cache[m.attn_index],
                        &self.v_cache[m.attn_index],
                        &q.qkv,
                        &b.attn,
                        &b.pos,
                        &b.attn_ws,
                    )?;
                    k.quantize(st, &b.attn, &q.qs, &q.ad, q_dim)?;
                    k.gemv(st, &m.o, &q.qs, &q.ad, &q.mix)?;
                }
                NMixer::Moe(m) => {
                    hy.gemv_f32(st, &m.router, &q.xn, &q.logits, 0, w.n_experts, hidden)?;
                    nem.route(st, &q.logits, &m.probs_bias, &q.chosen, &q.weights, w.n_experts, n_used, w.norm_weights, w.weights_scale)?;
                    k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
                    nem.expert_gemv(st, &m.up, &q.chosen, n_used, &q.qs, 0, &q.ad, 0, &q.a_up, nf)?;
                    nem.relu_sqr(st, &q.a_up, n_used * nf)?;
                    k.quantize(st, &q.a_up, &q.qs2, &q.ad2, n_used * nf)?;
                    nem.expert_gemv(st, &m.down, &q.chosen, n_used, &q.qs2, nf, &q.ad2, nf / 32, &q.heads, hidden)?;
                    // The shared expert reads the quantized input again (`qs` and `ad` are unchanged).
                    k.gemv(st, &m.shared_up, &q.qs, &q.ad, &q.sh_a)?;
                    nem.relu_sqr(st, &q.sh_a, w.n_ff_shexp)?;
                    k.quantize(st, &q.sh_a, &q.qs2, &q.ad2, w.n_ff_shexp)?;
                    k.gemv(st, &m.shared_down, &q.qs2, &q.ad2, &q.shared)?;
                    nem.combine(st, &q.heads, &q.weights, &q.shared, &q.mix, hidden, n_used)?;
                }
            }
            k.add(st, &b.h, &q.mix, hidden)?;
        }
        if !head {
            return Ok(());
        }
        k.rms_norm_rows(st, &b.h, 0, &w.out_norm, &q.xn, 0, 1, hidden, eps)?;
        k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
        match &w.head {
            NHead::Q8(m) => nem.gemv_q8(st, m, &q.qs, &q.ad, &b.logits),
            NHead::Q2(m) => k.gemv(st, m, &q.qs, &q.ad, &b.logits),
        }
    }
}
