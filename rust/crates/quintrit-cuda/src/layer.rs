//! Launchers of the layer kernels (`kernels/layer.cu`): normalization with quantization, the residual add, the rotary embedding with the KV append, the
//! attention, the final norm and the output head. Their results are bit-identical to the CPU runtime (`quintrit-runtime`); see the kernel file for the
//! arithmetic each one follows.

use crate::device::{Args, Device, DeviceBuffer, DevicePtr, Function, Stream};
use crate::error::{Error, Result};
use crate::pack::{act_entries, blocks_per_row};

/// Pairing of the rotated coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// Coordinates `2 i` and `2 i + 1` form a pair.
    Interleaved,
    /// Coordinates `i` and `i + rope_dims / 2` form a pair.
    HalfSplit,
}

/// Head geometry of a standalone rotary embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeKvLayout {
    /// Number of query heads.
    pub q_heads: usize,
    /// Number of key-value heads.
    pub kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Rotated coordinates per head (even, at most `head_dim`).
    pub rope_dims: usize,
    /// Pairing.
    pub mode: RopeMode,
    /// Allocated length of the KV cache in positions.
    pub cache_len: usize,
}

/// Geometry of the attention of one token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionLayout {
    /// Number of query heads.
    pub q_heads: usize,
    /// Number of key-value heads (`q_heads` must be a multiple of it).
    pub kv_heads: usize,
    /// Head dimension: 32, 64, 128 or 256.
    pub head_dim: usize,
    /// Allocated length of the KV cache in positions (the CPU runtime's `max_seq_len`; it enters the padding of the attended axis).
    pub cache_len: usize,
    /// Largest context the launch must serve (at most [`MAX_CONTEXT`] and `cache_len`); the shared memory is sized for it, so one launch
    /// (or one graph) serves every position below it.
    pub max_len: usize,
}

/// The start of a weight matrix to prefetch into the L2 cache while a quantization kernel runs (it changes no result).
#[derive(Clone, Copy, Debug)]
pub struct Prefetch {
    /// First byte of the matrix.
    pub base: DevicePtr,
    /// Bytes per row of the matrix.
    pub row_bytes: usize,
    /// Rows whose first cache lines are requested.
    pub rows: usize,
}

const PREFETCH_LINES: i32 = 3;
const RMS_QUANT_PREFETCH_BLOCKS: u32 = 24;
const QUANT_CELLS_PREFETCH_BLOCKS: usize = 6;

/// Largest context the attention kernel supports.
pub const MAX_CONTEXT: usize = 7936;

const ATTENTION_THREADS: u32 = 384;
const SHARED_LIMIT: usize = 96 * 1024;

/// Number of chunks the padded KV axis is cut into from a context of 257 tokens on (the CPU runtime's `REFERENCE_ATTENTION_THREADS`); every chunk runs in a
/// block of its own.
const ATTENTION_CHUNKS: usize = 4;

/// Positions of the longest chunk a launch for contexts up to `max_len` can see: the unsplit softmax of a short context (up to 511 positions) or a quarter of
/// the padded axis.
fn attention_chunk_positions(cache_len: usize, max_len: usize) -> usize {
    let padded = cache_len.min(max_len.next_multiple_of(256).max(256));
    max_len.min(511).max(padded.div_ceil(ATTENTION_CHUNKS))
}

fn attention_shared_bytes(head_dim: usize, chunk_positions: usize) -> usize {
    ((2 * chunk_positions + 3) & !3) * 4 + 128 * head_dim
}

/// Device memory the attention needs next to its inputs: the partial results of the chunks of every head and the arrival counters that find the block
/// that combines them. One workspace serves every layer of a token (the layers run one after another on a stream); a launch leaves the counters at zero.
///
/// A workspace for `q_heads` heads of dimension `head_dim` takes `q_heads * 4 * (head_dim + 2) * 4` bytes for the partial results (the accumulator, the
/// softmax sum and the maximum of each of the four chunks) plus `q_heads * 4` bytes for the counters, for example 83,200 bytes for 40 heads of dimension 128. It can serve a
/// layout with fewer query heads, but not another head dimension. Launches that share a workspace must run in order on one stream; give every
/// sequence that runs concurrently a workspace of its own.
pub struct AttentionWorkspace {
    part: DeviceBuffer<f32>,
    arrivals: DeviceBuffer<u32>,
    q_heads: usize,
    head_dim: usize,
}

impl AttentionWorkspace {
    /// Allocates the workspace for `q_heads` heads of dimension `head_dim`.
    pub fn new(device: &Device, q_heads: usize, head_dim: usize) -> Result<AttentionWorkspace> {
        Ok(AttentionWorkspace {
            part: device.alloc_zeroed(q_heads * ATTENTION_CHUNKS * (head_dim + 2))?,
            arrivals: device.alloc_zeroed(q_heads)?,
            q_heads,
            head_dim,
        })
    }
}

/// The loaded layer kernels of one device.
pub struct LayerKernels {
    rms_quant: [Function; 3],
    quant_cells: Function,
    residual_add: Function,
    final_norm: Function,
    lm_head: Function,
    rope_kv: Function,
    attention: [Function; 4],
    shared_limit: usize,
}

const RMS_QUANT_MAX_COLS: [usize; 3] = [6 * 512, 8 * 512, 14 * 512];

fn check_len<T: Copy>(b: &DeviceBuffer<T>, need: usize, what: &str) -> Result<()> {
    if b.len() < need {
        return Err(Error::InvalidArgument(format!("{what}: buffer holds {} elements, {need} are needed", b.len())));
    }
    Ok(())
}

impl LayerKernels {
    /// Loads the kernels.
    pub fn new(device: &Device) -> Result<LayerKernels> {
        let module = device.module("layer")?;
        let shared_limit = SHARED_LIMIT.min(device.info().max_shared_memory_optin as usize);
        let with_shared = |name: &str| -> Result<Function> {
            let f = module.function(name)?;
            f.set_max_dynamic_shared(shared_limit as u32)?;
            Ok(f)
        };
        Ok(LayerKernels {
            rms_quant: [with_shared("rms_quant_p6")?, with_shared("rms_quant_p8")?, with_shared("rms_quant_p14")?],
            quant_cells: module.function("quant_cells")?,
            residual_add: module.function("residual_add")?,
            final_norm: module.function("final_norm")?,
            lm_head: module.function("lm_head")?,
            rope_kv: module.function("rope_kv")?,
            attention: [with_shared("attention_h32")?, with_shared("attention_h64")?, with_shared("attention_h128")?, with_shared("attention_h256")?],
            shared_limit,
        })
    }

    /// Normalization and quantization of a vector of `cols` floats.
    ///
    /// With `gamma`: `y = rmsnorm(x) * gamma` (at most 7168 columns), without: `y = x`. `y` is quantized to int8 with the scale `127 / max|y|` and written in
    /// the permuted layout of the projection kernel to `act` together with the sum of the codes and the scale. With `residual = (delta, weight_scale)` the
    /// residual add `x += delta * (weight_scale / previous act_scale)` is done first (it needs `gamma`) and `x` is updated in place; the previous scale is
    /// read from `act_scale` before it is overwritten.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_quant(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        gamma: Option<&DeviceBuffer<f32>>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        cols: usize,
        eps: f32,
        residual: Option<(&DeviceBuffer<i32>, f32)>,
        prefetch: Option<Prefetch>,
    ) -> Result<()> {
        check_len(x, cols, "rms_quant input")?;
        check_len(act, act_entries(cols) * 8, "rms_quant output")?;
        check_len(act_sum, 1, "act_sum")?;
        check_len(act_scale, 1, "act_scale")?;
        if let Some(g) = gamma {
            check_len(g, cols, "norm weights")?;
        }
        if residual.is_some() && gamma.is_none() {
            return Err(Error::InvalidArgument("the residual add is fused with a norm only".into()));
        }
        if cols * 4 > self.shared_limit {
            return Err(Error::InvalidArgument(format!("{cols} columns exceed the shared memory of the normalization kernel")));
        }
        let (idx, threads) = if gamma.is_some() {
            let i = RMS_QUANT_MAX_COLS
                .iter()
                .position(|&m| cols <= m)
                .ok_or_else(|| Error::InvalidArgument(format!("a norm over {cols} columns is not supported (at most {})", RMS_QUANT_MAX_COLS[2])))?;
            (i, 512)
        } else {
            (0, 1024)
        };
        let mut a = Args::new();
        a.buf(x);
        match gamma {
            Some(g) => a.buf(g),
            None => a.null(),
        };
        a.buf(act).buf(act_sum).buf(act_scale).push(cols as i32).push(eps).push(gamma.is_some() as i32);
        match residual {
            Some((d, ws)) => {
                check_len(d, cols, "residual delta")?;
                a.buf(d).push(ws)
            }
            None => a.null().push(0f32),
        };
        match prefetch {
            Some(p) => a.ptr(p.base).push(p.row_bytes).push(p.rows as i32).push(PREFETCH_LINES),
            None => a.null().push(0usize).push(0i32).push(0i32),
        };
        let blocks = if prefetch.is_some() { RMS_QUANT_PREFETCH_BLOCKS } else { 1 };
        // SAFETY: the arguments match `rms_quant_p<N>` and the buffers were checked above; the prefetch only touches the first bytes of the rows of a live matrix
        // (the caller passes a matrix that holds `rows` rows of `row_bytes`).
        unsafe { self.rms_quant[idx].launch(stream, (blocks, 1, 1), (threads, 1, 1), (cols * 4) as u32, &a) }
    }

    /// Quantization of `v` (`cols` floats) with the maximum collected in `cells[par]` by the gated epilogue of the projection kernel. `act_sum` must be zero
    /// (the epilogue clears it).
    #[allow(clippy::too_many_arguments)]
    pub fn quant_cells(
        &self,
        stream: &Stream,
        v: &DeviceBuffer<f32>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        cells: &DeviceBuffer<i32>,
        par: usize,
        cols: usize,
        prefetch: Option<Prefetch>,
    ) -> Result<()> {
        check_len(v, cols, "quant_cells input")?;
        check_len(act, act_entries(cols) * 8, "quant_cells output")?;
        check_len(act_sum, 1, "act_sum")?;
        check_len(act_scale, 1, "act_scale")?;
        check_len(cells, 2, "cells")?;
        if par > 1 {
            return Err(Error::InvalidArgument("par must be 0 or 1".into()));
        }
        let entries = blocks_per_row(cols) * 32;
        let mut a = Args::new();
        let quant_blocks = entries.div_ceil(256);
        a.buf(v).buf(act).buf(act_sum).buf(act_scale).buf(cells).push(par as i32).push(cols as i32);
        match prefetch {
            Some(p) => a.ptr(p.base).push(p.row_bytes).push(p.rows as i32).push(PREFETCH_LINES),
            None => a.null().push(0usize).push(0i32).push(0i32),
        };
        a.push(quant_blocks as i32);
        let blocks = quant_blocks + if prefetch.is_some() { QUANT_CELLS_PREFETCH_BLOCKS } else { 0 };
        // SAFETY: the arguments match `quant_cells`; see `rms_quant` for the prefetch.
        unsafe { self.quant_cells.launch(stream, (blocks as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `h += delta * (weight_scale / act_scale)` over `n` elements.
    pub fn residual_add(&self, stream: &Stream, h: &DeviceBuffer<f32>, delta: &DeviceBuffer<i32>, act_scale: &DeviceBuffer<f32>, ws: f32, n: usize) -> Result<()> {
        check_len(h, n, "residual target")?;
        check_len(delta, n, "residual delta")?;
        check_len(act_scale, 1, "act_scale")?;
        let mut a = Args::new();
        a.buf(h).buf(delta).buf(act_scale).push(ws).push(n as i32);
        // SAFETY: the arguments match `residual_add`.
        unsafe { self.residual_add.launch(stream, (1, 1, 1), (1024, 1, 1), 0, &a) }
    }

    /// Final norm: `out = f16-rounded rmsnorm(h) * w`.
    pub fn final_norm(&self, stream: &Stream, h: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, cols: usize, eps: f32) -> Result<()> {
        check_len(h, cols, "final norm input")?;
        check_len(w, cols, "final norm weights")?;
        check_len(out, cols, "final norm output")?;
        let mut a = Args::new();
        a.buf(h).buf(w).buf(out).push(cols as i32).push(eps);
        // SAFETY: the arguments match `final_norm`.
        unsafe { self.final_norm.launch(stream, (1, 1, 1), (1024, 1, 1), 0, &a) }
    }

    /// Output head: `out[v] = dot_f16(x, w[v])` for every vocabulary row; `w` holds F16 bit patterns `[vocab][k]`, `k` a multiple of 32.
    pub fn lm_head(&self, stream: &Stream, w: &DeviceBuffer<u16>, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, vocab: usize, k: usize) -> Result<()> {
        if k == 0 || k % 32 != 0 {
            return Err(Error::InvalidArgument("the output head needs a hidden size that is a multiple of 32".into()));
        }
        check_len(w, vocab * k, "head matrix")?;
        check_len(x, k, "head input")?;
        check_len(out, vocab, "logits")?;
        let mut a = Args::new();
        a.buf(w).buf(x).buf(out).push(vocab as i32).push(k as i32);
        // SAFETY: the arguments match `lm_head`.
        unsafe { self.lm_head.launch(stream, (vocab.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Standalone rotary embedding and KV append for the int32 result `qkv` of the stacked projection (`q | k | v`), as the plain projection returns it.
    /// `cos_sin` holds `rope_dims` floats per position for at least `*pos_dev + 1` positions.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv(
        &self,
        stream: &Stream,
        layout: RopeKvLayout,
        qkv: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        scales: (f32, f32, f32),
        cos_sin: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
        q_out: &DeviceBuffer<f32>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
    ) -> Result<()> {
        let hd = layout.head_dim;
        if hd == 0 || layout.rope_dims % 2 != 0 || layout.rope_dims > hd || layout.rope_dims == 0 {
            return Err(Error::InvalidArgument("rope: inconsistent head geometry".into()));
        }
        let cache_elems = layout.kv_heads * layout.cache_len * hd;
        check_len(qkv, (layout.q_heads + 2 * layout.kv_heads) * hd, "qkv")?;
        check_len(q_out, layout.q_heads * hd, "q output")?;
        check_len(k_cache, cache_elems, "key cache")?;
        check_len(v_cache, cache_elems, "value cache")?;
        check_len(cos_sin, layout.cache_len * layout.rope_dims, "cos/sin table")?;
        check_len(act_scale, 1, "act_scale")?;
        check_len(pos_dev, 1, "position")?;
        let mut a = Args::new();
        a.buf(qkv).buf(act_scale).push(scales.0).push(scales.1).push(scales.2).buf(cos_sin).buf(pos_dev).buf(q_out).buf(k_cache).buf(v_cache);
        a.push(layout.q_heads as i32).push(layout.kv_heads as i32).push(hd as i32).push(layout.rope_dims as i32);
        a.push((layout.mode == RopeMode::HalfSplit) as i32).push(layout.cache_len as i32);
        // SAFETY: the arguments match `rope_kv`; the position is read on the device and must be below `cache_len` (the caller's contract, as for the cache).
        unsafe { self.rope_kv.launch(stream, (1, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Attention of the new token (position `*pos_dev`) over the cached positions `0..=*pos_dev` for every query head, bit-identical to the CPU runtime's
    /// `attend_head` with the bitnet.cpp numerics (F16 accumulator). `q` holds the rotated query as F32; `scale` is `1 / sqrt(head_dim)`. `workspace` holds the
    /// partial results of the chunks of the split KV axis (see [`AttentionWorkspace`]); launches that share one must run in order on one stream.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        stream: &Stream,
        layout: AttentionLayout,
        scale: f32,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        q: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
        workspace: &AttentionWorkspace,
    ) -> Result<()> {
        let idx = match layout.head_dim {
            32 => 0,
            64 => 1,
            128 => 2,
            256 => 3,
            other => return Err(Error::InvalidArgument(format!("head dimension {other} is not supported (32, 64, 128 or 256)"))),
        };
        if layout.kv_heads == 0 || layout.q_heads % layout.kv_heads != 0 {
            return Err(Error::InvalidArgument("the number of query heads must be a multiple of the number of key-value heads".into()));
        }
        if layout.max_len == 0 || layout.max_len > MAX_CONTEXT || layout.max_len > layout.cache_len {
            return Err(Error::InvalidArgument(format!("the attended context must be 1..={} and within the cache", MAX_CONTEXT.min(layout.cache_len))));
        }
        if workspace.q_heads < layout.q_heads || workspace.head_dim != layout.head_dim {
            return Err(Error::InvalidArgument("the attention workspace was made for another head geometry".into()));
        }
        let chunk_positions = attention_chunk_positions(layout.cache_len, layout.max_len);
        let shared = attention_shared_bytes(layout.head_dim, chunk_positions);
        if shared > self.shared_limit {
            return Err(Error::InvalidArgument(format!("the attention needs {shared} bytes of shared memory, the limit is {}", self.shared_limit)));
        }
        let cache_elems = layout.kv_heads * layout.cache_len * layout.head_dim;
        check_len(k_cache, cache_elems, "key cache")?;
        check_len(v_cache, cache_elems, "value cache")?;
        check_len(q, layout.q_heads * layout.head_dim, "query")?;
        check_len(out, layout.q_heads * layout.head_dim, "attention output")?;
        check_len(pos_dev, 1, "position")?;
        let mut a = Args::new();
        a.buf(k_cache).buf(v_cache).buf(q).buf(out).buf(&workspace.part).buf(&workspace.arrivals);
        a.push((layout.q_heads / layout.kv_heads) as i32).push(layout.cache_len as i32).push(scale).buf(pos_dev);
        a.push(chunk_positions as i32);
        // SAFETY: the arguments match `attention_h<HD>`; the position read from `pos_dev` must be below `layout.max_len` (the caller's contract).
        unsafe { self.attention[idx].launch(stream, (layout.q_heads as u32, ATTENTION_CHUNKS as u32, 1), (ATTENTION_THREADS, 1, 1), shared as u32, &a) }
    }
}
