//! Block-scaled (`Q2_0`) dense models on the GPU: the weights of every projection are the planar velurex layout with one F16 scale per 64 weights, the activations
//! of every projection are quantized in blocks of 32 (`Q8_0`) on the device, and the attention follows upstream llama.cpp (F32 query and value accumulator).
//! Every kernel is bit-identical to the CPU runtime (`quintrit-cuda`, `kernels/q2.cu` and the F32 variant of the attention in `kernels/layer.cu`).

use super::{err, GpuModel, GpuSession};
use crate::config::{ActivationType, ModelArchitecture, ModelConfig, Numerics};
use crate::ops::{f16_to_f32, f32_to_f16};
use quintrit_cuda::{AttentionLayout, BlockScaledKernels, BlockScaledMatrix, Device, DeviceBuffer, RopeKvLayout, RopeMode as GpuRopeMode};
use quintrit_format::{decode_q2_0_packed, GgufFile, GGML_TYPE_Q2_0};

use crate::config::RopeMode;

pub(super) struct Q2Layer {
    qkv: BlockScaledMatrix,
    o: BlockScaledMatrix,
    gate_up: BlockScaledMatrix,
    down: BlockScaledMatrix,
    attn_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    q_norm: Option<DeviceBuffer<f32>>,
    k_norm: Option<DeviceBuffer<f32>>,
}

/// The weights of a block-scaled model on the device.
pub(super) struct Q2Weights {
    layers: Vec<Q2Layer>,
    out_norm: DeviceBuffer<f32>,
    head: BlockScaledMatrix,
    kernels: BlockScaledKernels,
}

/// The per-token buffers of a block-scaled model that the common buffers of a session do not have.
pub(super) struct Q2Buffers {
    xn: DeviceBuffer<f32>,
    qs: DeviceBuffer<i8>,
    ad: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    o_out: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    down_out: DeviceBuffer<f32>,
}

impl Q2Buffers {
    pub(super) fn new(device: &Device, config: &ModelConfig) -> Result<Q2Buffers, String> {
        let (hidden, inter) = (config.hidden_size, config.intermediate_size);
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let widest = hidden.max(q_dim).max(inter);
        Ok(Q2Buffers {
            xn: device.alloc_zeroed(hidden).map_err(err)?,
            qs: device.alloc_zeroed(widest).map_err(err)?,
            ad: device.alloc_zeroed(widest / 32).map_err(err)?,
            qkv: device.alloc_zeroed(q_dim + 2 * kv_dim).map_err(err)?,
            o_out: device.alloc_zeroed(hidden).map_err(err)?,
            gate_up: device.alloc_zeroed(2 * inter).map_err(err)?,
            v: device.alloc_zeroed(inter).map_err(err)?,
            down_out: device.alloc_zeroed(hidden).map_err(err)?,
        })
    }
}

/// Whether the file holds block-scaled weights.
pub(super) fn is_block_scaled(gguf: &GgufFile) -> bool {
    gguf.tensors.values().any(|t| t.tensor_type == GGML_TYPE_Q2_0)
}

/// Checks that the GPU decoder can run this block-scaled model; the error says why not.
pub(super) fn check_supported(config: &ModelConfig, gguf: &GgufFile) -> Result<(), String> {
    if matches!(config.architecture, ModelArchitecture::Qwen35 | ModelArchitecture::NemotronHMoe) {
        return Err("hybrid and mixture-of-experts models run on the CPU only".into());
    }
    if config.numerics != Numerics::LlamaCpp {
        return Err("block-scaled models follow the llama.cpp numerics".into());
    }
    if config.has_sub_norm {
        return Err("block-scaled models with sub-layer normalization run on the CPU only".into());
    }
    if config.hidden_act != ActivationType::Silu {
        return Err("block-scaled models with another activation than SwiGLU run on the CPU only".into());
    }
    if !matches!(config.head_dim, 32 | 64 | 128 | 256) {
        return Err(format!("head dimension {} is not supported on the GPU (32, 64, 128 or 256)", config.head_dim));
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        return Err("the number of query heads must be a multiple of the number of key-value heads".into());
    }
    let q_dim = config.num_attention_heads * config.head_dim;
    for (what, n) in [("hidden size", config.hidden_size), ("attention width", q_dim), ("intermediate size", config.intermediate_size)] {
        if n % 64 != 0 {
            return Err(format!("the {what} ({n}) must be a multiple of 64 for the block-scaled projections"));
        }
    }
    if config.intermediate_size % 8 != 0 {
        return Err("the intermediate size must be a multiple of 8".into());
    }
    match gguf.tensors.get("token_embd.weight").map(|t| t.tensor_type) {
        Some(GGML_TYPE_Q2_0) => {}
        _ => return Err("token_embd.weight is not Q2_0; the GPU decoder needs a block-scaled embedding in a block-scaled model".into()),
    }
    if let Some(t) = gguf.tensors.get("output.weight") {
        if t.tensor_type != GGML_TYPE_Q2_0 {
            return Err("output.weight is not Q2_0; the GPU decoder needs a block-scaled head in a block-scaled model".into());
        }
    }
    Ok(())
}

/// The packed weights of a `Q2_0` tensor and its scales as F16 bit patterns (llama.cpp stores them as F16, so the conversion is exact).
pub(crate) fn q2_tensor(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<u16>), String> {
    let info = gguf.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
    if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows {
        return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
    }
    if info.tensor_type != GGML_TYPE_Q2_0 {
        return Err(format!("tensor '{name}' is not Q2_0 (a block-scaled model needs block-scaled projections)"));
    }
    let block = info.block_size.ok_or_else(|| format!("tensor '{name}': unknown Q2_0 block size"))?;
    let data = gguf.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
    let (packed, scales) = decode_q2_0_packed(data, rows, cols, block).map_err(|e| format!("{name}: {e}"))?;
    let mut f16 = Vec::with_capacity(scales.len());
    for s in scales {
        let h = f32_to_f16(s);
        if f16_to_f32(h).to_bits() != s.to_bits() {
            return Err(format!("tensor '{name}' has a scale that F16 cannot hold exactly"));
        }
        f16.push(h);
    }
    Ok((packed, f16))
}

fn matrix(device: &Device, gguf: &GgufFile, names: &[String], rows: &[usize], cols: usize) -> Result<BlockScaledMatrix, String> {
    let (mut packed, mut scales) = (Vec::new(), Vec::new());
    for (name, &r) in names.iter().zip(rows) {
        let (p, s) = q2_tensor(gguf, name, r, cols)?;
        packed.extend_from_slice(&p);
        scales.extend_from_slice(&s);
    }
    BlockScaledMatrix::new(device, &packed, &scales, rows.iter().sum(), cols).map_err(err)
}

/// Uploads the weights of a block-scaled model.
pub(super) fn load(gguf: &GgufFile, config: &ModelConfig, device: &Device) -> Result<Q2Weights, String> {
    let kernels = BlockScaledKernels::new(device).map_err(err)?;
    let (hidden, inter) = (config.hidden_size, config.intermediate_size);
    let (hd, qh, kvh) = (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
    let (q_dim, kv_dim) = (qh * hd, kvh * hd);
    let upload_f32 = |name: &str| -> Result<DeviceBuffer<f32>, String> {
        let v = gguf.get_tensor_f32(name).ok_or_else(|| format!("Missing {name}"))?;
        device.upload(&v).map_err(err)
    };
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for l in 0..config.num_hidden_layers {
        let name = |n: &str| format!("blk.{l}.{n}.weight");
        let qkv = matrix(device, gguf, &[name("attn_q"), name("attn_k"), name("attn_v")], &[q_dim, kv_dim, kv_dim], hidden)?;
        let o = matrix(device, gguf, &[name("attn_output")], &[hidden], q_dim)?;
        let gate_up = matrix(device, gguf, &[name("ffn_gate"), name("ffn_up")], &[inter, inter], hidden)?;
        let down = matrix(device, gguf, &[name("ffn_down")], &[hidden], inter)?;
        layers.push(Q2Layer {
            qkv,
            o,
            gate_up,
            down,
            attn_norm: upload_f32(&name("attn_norm"))?,
            ffn_norm: upload_f32(&name("ffn_norm"))?,
            q_norm: if config.qk_norm { Some(upload_f32(&name("attn_q_norm"))?) } else { None },
            k_norm: if config.qk_norm { Some(upload_f32(&name("attn_k_norm"))?) } else { None },
        });
    }
    let out_norm = upload_f32("output_norm.weight")?;
    let head_name = if gguf.tensors.contains_key("output.weight") { "output.weight" } else { "token_embd.weight" };
    let head = matrix(device, gguf, &[head_name.to_string()], &[config.vocab_size], hidden)?;
    Ok(Q2Weights { layers, out_norm, head, kernels })
}

impl GpuSession {
    /// Records the device work of one token of a block-scaled model (the position comes from device memory).
    pub(super) fn enqueue_q2(&self, head: bool) -> quintrit_cuda::Result<()> {
        let model: &GpuModel = &self.model;
        let w = model.q2.as_ref().expect("a block-scaled model");
        let q = self.q2.as_ref().expect("the buffers of a block-scaled model");
        let k = &w.kernels;
        let c = &model.cfg;
        let st = &self.stream;
        let b = &self.buf;
        let (hidden, inter, hd) = (c.hidden_size, c.intermediate_size, c.head_dim);
        let (qh, kvh) = (c.num_attention_heads, c.num_key_value_heads);
        let (q_dim, _kv_dim) = (qh * hd, kvh * hd);
        let eps = c.rms_norm_eps;
        let attn_layout = AttentionLayout { q_heads: qh, kv_heads: kvh, head_dim: hd, cache_len: self.context, max_len: self.context };
        let mode = match c.rope.mode {
            RopeMode::Normal => GpuRopeMode::Interleaved,
            RopeMode::Neox => GpuRopeMode::HalfSplit,
        };
        let rope_layout = RopeKvLayout { q_heads: qh, kv_heads: kvh, head_dim: hd, rope_dims: model.rope_dims, mode, cache_len: self.context };

        for (l, layer) in w.layers.iter().enumerate() {
            k.rms_norm_rows(st, &b.h, 0, &layer.attn_norm, &q.xn, 0, 1, hidden, eps)?;
            k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
            k.gemv(st, &layer.qkv, &q.qs, &q.ad, &q.qkv)?;
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                k.rms_norm_rows(st, &q.qkv, 0, qn, &q.qkv, 0, qh, hd, eps)?;
                k.rms_norm_rows(st, &q.qkv, q_dim, kn, &q.qkv, q_dim, kvh, hd, eps)?;
            }
            k.rope_kv(st, rope_layout, &q.qkv, &model.cos_sin, &b.pos, &b.q, &self.k_cache[l], &self.v_cache[l])?;
            model.layer.attention_f32(st, attn_layout, model.attn_scale, &self.k_cache[l], &self.v_cache[l], &b.q, &b.attn, &b.pos, &b.attn_ws)?;
            k.quantize(st, &b.attn, &q.qs, &q.ad, q_dim)?;
            k.gemv(st, &layer.o, &q.qs, &q.ad, &q.o_out)?;
            k.add(st, &b.h, &q.o_out, hidden)?;

            k.rms_norm_rows(st, &b.h, 0, &layer.ffn_norm, &q.xn, 0, 1, hidden, eps)?;
            k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
            k.gemv(st, &layer.gate_up, &q.qs, &q.ad, &q.gate_up)?;
            k.swiglu(st, &q.gate_up, &q.v, inter)?;
            k.quantize(st, &q.v, &q.qs, &q.ad, inter)?;
            k.gemv(st, &layer.down, &q.qs, &q.ad, &q.down_out)?;
            k.add(st, &b.h, &q.down_out, hidden)?;
        }
        if !head {
            return Ok(());
        }
        k.rms_norm_rows(st, &b.h, 0, &w.out_norm, &q.xn, 0, 1, hidden, eps)?;
        k.quantize(st, &q.xn, &q.qs, &q.ad, hidden)?;
        k.gemv(st, &w.head, &q.qs, &q.ad, &b.logits)
    }
}
