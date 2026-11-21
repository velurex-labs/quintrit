//! Decoder on an NVIDIA GPU (feature `cuda`).
//!
//! [`GpuDecoder`] runs the forward pass of one token on the GPU with results that are bit-identical to the CPU path: the ternary projections are
//! exact integer kernels, and the norms, the rotary embedding, the attention (including the CPU runtime's split of the key-value axis), the activation
//! functions and the output head follow the CPU arithmetic operation by operation (`quintrit-cuda`, `kernels/layer.cu`).
//!
//! The weights are transcoded into the GPU's planar layout while the model is loaded (a run-time representation, never written to a file); the
//! KV cache lives on the device. The device work of a token is recorded once as a CUDA graph and replayed for every token; the position is read from
//! device memory, so the graph does not change from token to token.
//!
//! Supported: models whose projections are per-tensor-scaled ternary tensors (`I2_S`, uniformly scaled `TQ1_0` / `TQ2_0`), an `F16` token embedding and
//! head (or a tied head), the bitnet.cpp attention numerics, head dimensions 32, 64, 128 and 256 and contexts of up to 7936 positions. Block-scaled
//! (`Q2_0`), mixture-of-experts and hybrid models run on the CPU only.

use crate::config::{ActivationType, ModelConfig, Numerics, RopeConfig, RopeMode};
use crate::ops::{f16_to_f32, rope_cache};
use quintrit_cuda::pack::{act_entries, interleave_rows, stack_rows};
use std::sync::Arc;
use quintrit_cuda::profile::Role;
use quintrit_cuda::layer::Prefetch;
use quintrit_cuda::{
    Activation, AttentionLayout, AttentionWorkspace, Device, DeviceBuffer, Gemv, Graph, GraphExec, LayerKernels, PinnedBuffer, RopeKvLayout, RopeLayout,
    RopeMode as GpuRopeMode, Stream,
};
use quintrit_format::decoders::get_ternary_decoder;
use quintrit_format::transcoder::transcode_to_planar;
use quintrit_format::{GgufFile, GGML_TYPE_Q2_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0};

/// Context window of the GPU decoder when the caller does not choose one.
pub const DEFAULT_GPU_CONTEXT: usize = 4096;

fn err<T: std::fmt::Display>(e: T) -> String {
    e.to_string()
}

struct Scales {
    q: f32,
    k: f32,
    v: f32,
    o: f32,
    gate: f32,
    up: f32,
    down: f32,
}

struct GpuLayer {
    qkv: DeviceBuffer<u8>,
    o: DeviceBuffer<u8>,
    gate_up: DeviceBuffer<u8>,
    down: DeviceBuffer<u8>,
    attn_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    attn_sub_norm: Option<DeviceBuffer<f32>>,
    ffn_sub_norm: Option<DeviceBuffer<f32>>,
    ws: Scales,
}

struct Buffers {
    h: DeviceBuffer<f32>,
    act: DeviceBuffer<i8>,
    act_sum: DeviceBuffer<i32>,
    act_scale: DeviceBuffer<f32>,
    qkv: DeviceBuffer<i32>,
    q: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    o_out: DeviceBuffer<i32>,
    v: DeviceBuffer<f32>,
    act_ff: DeviceBuffer<i8>,
    sum_ff: DeviceBuffer<i32>,
    cells: DeviceBuffer<i32>,
    down_out: DeviceBuffer<i32>,
    xn: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    pos: DeviceBuffer<i32>,
    attn_ws: AttentionWorkspace,
}

/// The weights of one model on one GPU: the ternary matrices in the GPU layout, the norms, the output head and the rotary table. Immutable once loaded,
/// shared by the sessions ([`GpuModel::new_session`]) and usable from several threads.
pub struct GpuModel {
    layers: Vec<GpuLayer>,
    out_norm: DeviceBuffer<f32>,
    head: DeviceBuffer<u16>,
    cos_sin: DeviceBuffer<f32>,
    gemv: Gemv,
    layer: LayerKernels,
    device: Device,
    cfg: ModelConfig,
    rope_dims: usize,
    context: usize,
    attn_scale: f32,
    use_graph: bool,
}

/// One sequence on the GPU: a KV cache, the buffers of a token and the CUDA graph that replays it. Used by one thread at a time.
pub struct GpuSession {
    // The graph is declared first so that it is destroyed before the buffers it refers to.
    graph: Option<GraphExec>,
    k_cache: Vec<DeviceBuffer<u16>>,
    v_cache: Vec<DeviceBuffer<u16>>,
    buf: Buffers,
    emb_pin: PinnedBuffer<f32>,
    pos_pin: PinnedBuffer<i32>,
    logits_pin: PinnedBuffer<f32>,
    stream: Stream,
    context: usize,
    model: Arc<GpuModel>,
}

/// A model with a single session: the convenience the engine uses.
pub struct GpuDecoder {
    session: GpuSession,
}

struct Planar {
    data: Vec<u8>,
    scale: f32,
}

fn planar(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<Planar, String> {
    let info = gguf.tensors.get(name).ok_or_else(|| format!("Missing {name}"))?;
    if info.shape.len() != 2 || info.shape[0] != cols || info.shape[1] != rows {
        return Err(format!("tensor '{name}' has shape {:?}, expected [{cols}, {rows}]", info.shape));
    }
    if info.tensor_type == GGML_TYPE_Q2_0 {
        return Err(format!("tensor '{name}' is block-scaled (Q2_0), which the GPU decoder does not support"));
    }
    let data = gguf.get_tensor_data(name).ok_or_else(|| format!("Failed to read tensor '{name}'"))?;
    let scale = match gguf.get_tensor_scale(name) {
        Some(s) => s,
        None if matches!(info.tensor_type, GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => {
            return Err(format!("Tensor '{name}' has differing per-block scales, which a per-tensor weight scale cannot represent"));
        }
        None => 1.0,
    };
    let decoder = get_ternary_decoder(info.tensor_type)?;
    Ok(Planar { data: transcode_to_planar(decoder, data, rows, cols)?, scale })
}

impl GpuModel {
    /// Context window the decoder uses for a model: the model's maximum, at most 7936 positions and `limit` (if given), rounded down to a multiple of
    /// 256. The multiple of 256 keeps the padded length of the attended axis, which the CPU runtime derives from its own cache length, identical.
    pub fn context_for(config: &ModelConfig, limit: Option<usize>) -> usize {
        let want = config.max_position_embeddings.min(limit.unwrap_or(DEFAULT_GPU_CONTEXT)).min(quintrit_cuda::layer::MAX_CONTEXT);
        if want >= 256 { want / 256 * 256 } else { want }
    }

    /// Checks that the GPU decoder can run the model; the error says why not.
    pub fn check_supported(config: &ModelConfig, gguf: &GgufFile) -> Result<(), String> {
        use crate::config::ModelArchitecture::*;
        if gguf.tensors.values().any(|t| t.tensor_type == GGML_TYPE_Q2_0) {
            return Err("block-scaled (Q2_0) models run on the CPU only; the GPU decoder supports I2_S models".into());
        }
        if matches!(config.architecture, Qwen35 | NemotronHMoe) {
            return Err("hybrid and mixture-of-experts models run on the CPU only".into());
        }
        if config.qk_norm {
            return Err("models with query/key normalization run on the CPU only".into());
        }
        if config.numerics != Numerics::BitnetCpp {
            return Err("the GPU decoder reproduces the bitnet.cpp attention numerics only (I2_S models)".into());
        }
        if !matches!(config.head_dim, 32 | 64 | 128 | 256) {
            return Err(format!("head dimension {} is not supported on the GPU (32, 64, 128 or 256)", config.head_dim));
        }
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err("the number of query heads must be a multiple of the number of key-value heads".into());
        }
        if config.hidden_size % 32 != 0 {
            return Err("the hidden size must be a multiple of 32".into());
        }
        for name in ["token_embd.weight", "output.weight"] {
            if let Some(t) = gguf.tensors.get(name) {
                if t.tensor_type != quintrit_format::GGML_TYPE_F16 {
                    return Err(format!("{name} is not F16; the GPU decoder needs an F16 embedding and head"));
                }
            }
        }
        Ok(())
    }

    /// Loads the model onto GPU `ordinal`. `limit` bounds the context window (see [`GpuModel::context_for`]).
    pub fn load(gguf: &GgufFile, config: &ModelConfig, ordinal: usize, limit: Option<usize>) -> Result<GpuModel, String> {
        Self::check_supported(config, gguf)?;
        let device = Device::open(ordinal).map_err(err)?;
        let gemv = Gemv::new(&device).map_err(err)?;
        let layer = LayerKernels::new(&device).map_err(err)?;
        let context = Self::context_for(config, limit);

        let (hidden, inter) = (config.hidden_size, config.intermediate_size);
        let (hd, qh, kvh) = (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
        let (q_dim, kv_dim) = (qh * hd, kvh * hd);
        let rope = RopeConfig { freq_base: config.rope_theta, n_dims: config.rope.n_dims.min(hd), ..config.rope };
        let rope_dims = rope.n_dims;

        let upload_f32 = |name: &str| -> Result<DeviceBuffer<f32>, String> {
            let v = gguf.get_tensor_f32(name).ok_or_else(|| format!("Missing {name}"))?;
            device.upload(&v).map_err(err)
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let name = |n: &str| format!("blk.{l}.{n}.weight");
            let (q, k, v) = (
                planar(gguf, &name("attn_q"), q_dim, hidden)?,
                planar(gguf, &name("attn_k"), kv_dim, hidden)?,
                planar(gguf, &name("attn_v"), kv_dim, hidden)?,
            );
            let o = planar(gguf, &name("attn_output"), hidden, q_dim)?;
            let gate = planar(gguf, &name("ffn_gate"), inter, hidden)?;
            let up = planar(gguf, &name("ffn_up"), inter, hidden)?;
            let down = planar(gguf, &name("ffn_down"), hidden, inter)?;
            let row_bytes = quintrit_cuda::pack::row_bytes(hidden);
            let qkv_buf = device.upload(&stack_rows(&[&q.data, &k.data, &v.data])).map_err(err)?;
            let gate_up_buf = device.upload(&interleave_rows(&gate.data, &up.data, row_bytes)).map_err(err)?;
            layers.push(GpuLayer {
                qkv: qkv_buf,
                o: device.upload(&o.data).map_err(err)?,
                gate_up: gate_up_buf,
                down: device.upload(&down.data).map_err(err)?,
                attn_norm: upload_f32(&name("attn_norm"))?,
                ffn_norm: upload_f32(&name("ffn_norm"))?,
                attn_sub_norm: if config.has_sub_norm { Some(upload_f32(&name("attn_sub_norm"))?) } else { None },
                ffn_sub_norm: if config.has_sub_norm { Some(upload_f32(&name("ffn_sub_norm"))?) } else { None },
                ws: Scales { q: q.scale, k: k.scale, v: v.scale, o: o.scale, gate: gate.scale, up: up.scale, down: down.scale },
            });
        }
        let out_norm = upload_f32("output_norm.weight")?;
        let head_host = match gguf.get_tensor_f16("output.weight") {
            Some(h) => h,
            None => gguf.get_tensor_f16("token_embd.weight").ok_or_else(|| "Missing token_embd.weight tensor".to_string())?,
        };
        if head_host.len() != config.vocab_size * hidden {
            return Err(format!("the output head has {} values, expected {}", head_host.len(), config.vocab_size * hidden));
        }
        let head = device.upload(head_host).map_err(err)?;

        // The cos / sin pairs of every position, built on the host with the CPU runtime's own routine (so with the platform's libm, as the CPU path does).
        let mut table = Vec::with_capacity(context * rope_dims);
        let mut c = Vec::new();
        for p in 0..context {
            rope_cache(p, &rope, &mut c);
            table.extend_from_slice(&c);
        }
        let cos_sin = device.upload(&table).map_err(err)?;

        Ok(GpuModel {
            layers,
            out_norm,
            head,
            cos_sin,
            gemv,
            layer,
            device,
            cfg: config.clone(),
            rope_dims,
            context,
            attn_scale: 1.0 / (hd as f32).sqrt(),
            use_graph: std::env::var("QUINTRIT_CUDA_GRAPH").map(|v| v != "0").unwrap_or(true),
        })
    }

    /// The largest context window a session of this model can have (a multiple of 256, or the model's maximum below 256).
    pub fn max_context(&self) -> usize {
        self.context
    }

    /// The configuration of the model.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Name of the GPU.
    pub fn device_name(&self) -> &str {
        &self.device.info().name
    }

    /// Index of the GPU in the driver's enumeration.
    pub fn device_ordinal(&self) -> usize {
        self.device.info().ordinal
    }

    /// Free and total memory of the GPU in bytes.
    pub fn memory_info(&self) -> Result<(u64, u64), String> {
        self.device.memory_info().map_err(err)
    }

    /// Creates a session with a context window of `context` positions (default and maximum: [`GpuModel::max_context`]; rounded down to a multiple of 256).
    /// A session holds the KV cache (`2 x layers x kv heads x context x head dim x 2` bytes) and the buffers of one token on the device.
    pub fn new_session(self: &Arc<Self>, context: Option<usize>) -> Result<GpuSession, String> {
        let config = &self.cfg;
        let device = &self.device;
        let want = context.unwrap_or(self.context).min(self.context);
        let context = if want >= 256 { want / 256 * 256 } else { want };
        if context == 0 {
            return Err("the context window of a session must hold at least one position".into());
        }
        let stream = device.create_stream().map_err(err)?;
        let (hidden, inter) = (config.hidden_size, config.intermediate_size);
        let (hd, qh, kvh) = (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
        let (q_dim, kv_dim) = (qh * hd, kvh * hd);
        let cache_elems = kvh * context * hd;
        let (mut k_cache, mut v_cache) = (Vec::new(), Vec::new());
        for _ in 0..config.num_hidden_layers {
            k_cache.push(device.alloc_zeroed::<u16>(cache_elems).map_err(err)?);
            v_cache.push(device.alloc_zeroed::<u16>(cache_elems).map_err(err)?);
        }
        let act_cols = hidden.max(q_dim);
        let buf = Buffers {
            h: device.alloc_zeroed(hidden).map_err(err)?,
            act: device.alloc_zeroed(act_entries(act_cols) * 8).map_err(err)?,
            act_sum: device.alloc_zeroed(1).map_err(err)?,
            act_scale: device.alloc_zeroed(1).map_err(err)?,
            qkv: device.alloc_zeroed(q_dim + 2 * kv_dim).map_err(err)?,
            q: device.alloc_zeroed(q_dim).map_err(err)?,
            attn: device.alloc_zeroed(q_dim).map_err(err)?,
            o_out: device.alloc_zeroed(hidden).map_err(err)?,
            v: device.alloc_zeroed(inter).map_err(err)?,
            act_ff: device.alloc_zeroed(act_entries(inter) * 8).map_err(err)?,
            sum_ff: device.alloc_zeroed(1).map_err(err)?,
            cells: device.alloc_zeroed(2).map_err(err)?,
            down_out: device.alloc_zeroed(hidden).map_err(err)?,
            xn: device.alloc_zeroed(hidden).map_err(err)?,
            logits: device.alloc_zeroed(config.vocab_size).map_err(err)?,
            pos: device.alloc_zeroed(1).map_err(err)?,
            attn_ws: AttentionWorkspace::new(device, config.num_attention_heads, config.head_dim).map_err(err)?,
        };
        Ok(GpuSession {
            graph: None,
            k_cache,
            v_cache,
            buf,
            emb_pin: device.alloc_pinned(hidden).map_err(err)?,
            pos_pin: device.alloc_pinned(1).map_err(err)?,
            logits_pin: device.alloc_pinned(config.vocab_size).map_err(err)?,
            stream,
            context,
            model: self.clone(),
        })
    }
}

impl GpuSession {
    /// Time of the device work of one token (the CUDA graph, without the host steps) at position `pos`, in milliseconds: `blocks` blocks of `iterations`
    /// replays each, the time of each block divided by `iterations`. The caller has computed the positions before `pos`; this replays the token at `pos`
    /// (it overwrites the cache entry of that position with the same values).
    pub fn time_token(&mut self, embedding: &[u16], pos: usize, blocks: usize, iterations: usize) -> Result<Vec<f64>, String> {
        let mut logits = vec![0f32; self.model.cfg.vocab_size];
        self.forward(embedding, pos, &mut logits)?;
        if !self.model.use_graph {
            return Err("timing needs the CUDA graph (QUINTRIT_CUDA_GRAPH is 0)".into());
        }
        let graph = self.graph.as_ref().ok_or("no graph")?;
        let (start, stop) = (self.model.device.create_event().map_err(err)?, self.model.device.create_event().map_err(err)?);
        // Warm-up: clocks and caches.
        for _ in 0..iterations.min(50) {
            graph.launch(&self.stream).map_err(err)?;
        }
        self.stream.synchronize().map_err(err)?;
        let mut out = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            self.stream.record(&start).map_err(err)?;
            for _ in 0..iterations {
                graph.launch(&self.stream).map_err(err)?;
            }
            self.stream.record(&stop).map_err(err)?;
            stop.synchronize().map_err(err)?;
            out.push(stop.elapsed_ms_since(&start).map_err(err)? as f64 / iterations as f64);
        }
        Ok(out)
    }

    /// Positions the KV cache holds.
    pub fn context(&self) -> usize {
        self.context
    }

    /// The model the session belongs to.
    pub fn model(&self) -> &Arc<GpuModel> {
        &self.model
    }

    /// The L2 prefetch of the start of the matrix `w` (`rows` x `cols`) for the quantization kernel that precedes its projection, as the device profile allows.
    fn prefetch(&self, w: &DeviceBuffer<u8>, rows: usize, cols: usize) -> Option<Prefetch> {
        let limit = self.model.gemv.profile().prefetch_rows as usize;
        (limit > 0).then(|| Prefetch { base: w.ptr(), row_bytes: quintrit_cuda::pack::row_bytes(cols), rows: rows.min(limit) })
    }

    /// Records the device work of one token (the position comes from device memory).
    fn enqueue(&self) -> quintrit_cuda::Result<()> {
        let c = &self.model.cfg;
        let st = &self.stream;
        let b = &self.buf;
        let (hidden, inter, hd) = (c.hidden_size, c.intermediate_size, c.head_dim);
        let (qh, kvh) = (c.num_attention_heads, c.num_key_value_heads);
        let (q_dim, kv_dim) = (qh * hd, kvh * hd);
        let eps = c.rms_norm_eps;
        let activation = match c.hidden_act {
            ActivationType::Silu => Activation::Silu,
            ActivationType::Relu2 => Activation::Relu2,
        };
        let rows = q_dim + 2 * kv_dim;
        let n = self.model.layers.len();
        let attn_layout = AttentionLayout { q_heads: qh, kv_heads: kvh, head_dim: hd, cache_len: self.context, max_len: self.context };
        let rope_cfg = {
            let rope = RopeConfig { freq_base: c.rope_theta, n_dims: c.rope.n_dims.min(hd), ..c.rope };
            rope.mode
        };

        let pf_qkv = |l: usize| self.prefetch(&self.model.layers[l].qkv, rows, hidden);
        self.model.layer.rms_quant(st, &b.h, Some(&self.model.layers[0].attn_norm), &b.act, &b.act_sum, &b.act_scale, hidden, eps, None, pf_qkv(0))?;
        for (l, w) in self.model.layers.iter().enumerate() {
            match rope_cfg {
                RopeMode::Normal => {
                    let layout = RopeLayout { q_rows: q_dim, kv_rows: kv_dim, head_dim: hd, rope_dims: self.model.rope_dims, cache_len: self.context };
                    self.model.gemv.qkv_rope(
                        st, layout, hidden, &w.qkv, &b.act, &b.act_sum, &b.act_scale, (w.ws.q, w.ws.k, w.ws.v), &self.model.cos_sin, &b.pos, &b.q, &self.k_cache[l],
                        &self.v_cache[l],
                    )?;
                }
                RopeMode::Neox => {
                    let cfg = self.model.gemv.profile().config(Role::QkvRope, rows as u32)?;
                    self.model.gemv.plain_with(st, &cfg, rows, hidden, &w.qkv, &b.act, &b.act_sum, &b.qkv)?;
                    let layout = RopeKvLayout {
                        q_heads: qh,
                        kv_heads: kvh,
                        head_dim: hd,
                        rope_dims: self.model.rope_dims,
                        mode: GpuRopeMode::HalfSplit,
                        cache_len: self.context,
                    };
                    self.model.layer.rope_kv(st, layout, &b.qkv, &b.act_scale, (w.ws.q, w.ws.k, w.ws.v), &self.model.cos_sin, &b.pos, &b.q, &self.k_cache[l], &self.v_cache[l])?;
                }
            }
            self.model.layer.attention(st, attn_layout, self.model.attn_scale, &self.k_cache[l], &self.v_cache[l], &b.q, &b.attn, &b.pos, &b.attn_ws)?;
            self.model.layer.rms_quant(st, &b.attn, w.attn_sub_norm.as_ref(), &b.act, &b.act_sum, &b.act_scale, q_dim, eps, None, self.prefetch(&w.o, hidden, q_dim))?;
            self.model.gemv.plain(st, Role::Output, hidden, q_dim, &w.o, &b.act, &b.act_sum, &b.o_out)?;
            self.model.layer.rms_quant(st, &b.h, Some(&w.ffn_norm), &b.act, &b.act_sum, &b.act_scale, hidden, eps, Some((&b.o_out, w.ws.o)), self.prefetch(&w.gate_up, 2 * inter, hidden))?;
            let par = l & 1;
            self.model.gemv.gated(st, 2 * inter, hidden, &w.gate_up, &b.act, &b.act_sum, &b.act_scale, (w.ws.gate, w.ws.up), activation, &b.v, &b.cells, par, &b.sum_ff)?;
            match &w.ffn_sub_norm {
                Some(sub) => self.model.layer.rms_quant(st, &b.v, Some(sub), &b.act_ff, &b.sum_ff, &b.act_scale, inter, eps, None, self.prefetch(&w.down, hidden, inter))?,
                None => self.model.layer.quant_cells(st, &b.v, &b.act_ff, &b.sum_ff, &b.act_scale, &b.cells, par, inter, self.prefetch(&w.down, hidden, inter))?,
            }
            self.model.gemv.plain(st, Role::Down, hidden, inter, &w.down, &b.act_ff, &b.sum_ff, &b.down_out)?;
            if l + 1 < n {
                self.model.layer.rms_quant(st, &b.h, Some(&self.model.layers[l + 1].attn_norm), &b.act, &b.act_sum, &b.act_scale, hidden, eps, Some((&b.down_out, w.ws.down)), pf_qkv(l + 1))?;
            } else {
                self.model.layer.residual_add(st, &b.h, &b.down_out, &b.act_scale, w.ws.down, hidden)?;
            }
        }
        self.model.layer.final_norm(st, &b.h, &self.model.out_norm, &b.xn, hidden, eps)?;
        self.model.layer.lm_head(st, &self.model.head, &b.xn, &b.logits, c.vocab_size, hidden)
    }

    /// Computes the logits of the token whose F16 embedding row is `embedding` at position `pos` and stores them in `logits`.
    ///
    /// The positions `0..pos` must have been computed before (in order, as the CPU path does); positions are overwritten when a sequence is
    /// restarted or continued from a shorter prefix.
    pub fn forward(&mut self, embedding: &[u16], pos: usize, logits: &mut [f32]) -> Result<(), String> {
        let hidden = self.model.cfg.hidden_size;
        if embedding.len() != hidden {
            return Err(format!("embedding row has {} values, expected {hidden}", embedding.len()));
        }
        if pos >= self.context {
            return Err(format!("position {pos} is outside the GPU context window of {} positions", self.context));
        }
        if logits.len() != self.model.cfg.vocab_size {
            return Err("the logits buffer must match the vocabulary size".into());
        }
        for (o, &e) in self.emb_pin.as_mut_slice().iter_mut().zip(embedding) {
            *o = f16_to_f32(e);
        }
        self.pos_pin.as_mut_slice()[0] = pos as i32;
        self.emb_pin.copy_to_device_async(&self.buf.h, hidden, &self.stream).map_err(err)?;
        self.pos_pin.copy_to_device_async(&self.buf.pos, 1, &self.stream).map_err(err)?;
        if self.model.use_graph {
            if self.graph.is_none() {
                self.stream.begin_capture().map_err(err)?;
                let recorded = self.enqueue();
                let captured: Result<Graph, _> = self.stream.end_capture();
                recorded.map_err(err)?;
                self.graph = Some(captured.map_err(err)?.instantiate().map_err(err)?);
            }
            self.graph.as_ref().expect("graph was just created").launch(&self.stream).map_err(err)?;
        } else {
            self.enqueue().map_err(err)?;
        }
        let vocab = self.model.cfg.vocab_size;
        self.logits_pin.copy_from_device_async(&self.buf.logits, vocab, &self.stream).map_err(err)?;
        self.stream.synchronize().map_err(err)?;
        logits.copy_from_slice(&self.logits_pin.as_slice()[..vocab]);
        Ok(())
    }
}

impl GpuDecoder {
    /// See [`GpuModel::context_for`].
    pub fn context_for(config: &ModelConfig, limit: Option<usize>) -> usize {
        GpuModel::context_for(config, limit)
    }

    /// See [`GpuModel::check_supported`].
    pub fn check_supported(config: &ModelConfig, gguf: &GgufFile) -> Result<(), String> {
        GpuModel::check_supported(config, gguf)
    }

    /// Loads the model onto GPU `ordinal` with one session. `limit` bounds the context window (see [`GpuModel::context_for`]).
    pub fn load(gguf: &GgufFile, config: &ModelConfig, ordinal: usize, limit: Option<usize>) -> Result<GpuDecoder, String> {
        Self::from_model(&Arc::new(GpuModel::load(gguf, config, ordinal, limit)?))
    }

    /// A decoder with a new session on an already loaded model.
    pub fn from_model(model: &Arc<GpuModel>) -> Result<GpuDecoder, String> {
        Ok(GpuDecoder { session: model.new_session(None)? })
    }

    /// Positions the KV cache holds.
    pub fn context(&self) -> usize {
        self.session.context()
    }

    /// Name of the GPU.
    pub fn device_name(&self) -> &str {
        self.session.model.device_name()
    }

    /// Free and total memory of the GPU in bytes.
    pub fn memory_info(&self) -> Result<(u64, u64), String> {
        self.session.model.memory_info()
    }

    /// See [`GpuSession::forward`].
    pub fn forward(&mut self, embedding: &[u16], pos: usize, logits: &mut [f32]) -> Result<(), String> {
        self.session.forward(embedding, pos, logits)
    }

    /// See [`GpuSession::time_token`].
    pub fn time_token(&mut self, embedding: &[u16], pos: usize, blocks: usize, iterations: usize) -> Result<Vec<f64>, String> {
        self.session.time_token(embedding, pos, blocks, iterations)
    }
}
