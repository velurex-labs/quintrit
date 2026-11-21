//! Launchers of the projection kernel (`kernels/gemv.cu`): exact int32 GEMV of a planar velurex-encoded matrix with an int8
//! activation vector, with optional epilogues fused into the same launch.
//!
//! Result of a plain projection: `out[n] = sum_k trit[n, k] * act[k]` as `i32`. The post-scaling to floating point is part of the
//! epilogues (`(raw - activation_sum) * (weight_scale / act_scale)`, the scale divided once per projection) or of the caller.

use crate::device::{Args, Device, DeviceBuffer, Function, Stream};
use crate::error::{Error, Result};
use crate::pack::{act_entries, blocks_per_row, row_bytes};
use crate::profile::{DeviceProfile, GemvConfig, Role, ALL_CONFIGS};
use std::mem::size_of;

/// The activation function of the gated epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate) * up` with ggml's polynomial exponential.
    Silu,
    /// Squared ReLU: `max(gate, 0)^2 * up`.
    Relu2,
}

/// Arguments of the epilogues; mirrors `EpArgs` of `kernels/gemv.cu` (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EpilogueArgs {
    act_scale: u64,
    ws_gate: f32,
    ws_up: f32,
    v: u64,
    cells: u64,
    par: i32,
    activation: i32,
    sum_out: u64,
    q_out: u64,
    kw: u64,
    vw: u64,
    cs_all: u64,
    pos_dev: u64,
    ws_q: f32,
    ws_k: f32,
    ws_v: f32,
    q_rows: i32,
    kv_rows: i32,
    head_dim: i32,
    rope_dims: i32,
    l_alloc: i32,
    pad: i32,
}

const _: () = assert!(size_of::<EpilogueArgs>() == 128);
const _: () = assert!(std::mem::offset_of!(EpilogueArgs, sum_out) == 40);
const _: () = assert!(std::mem::offset_of!(EpilogueArgs, cs_all) == 72);
const _: () = assert!(std::mem::offset_of!(EpilogueArgs, ws_q) == 88);
const _: () = assert!(std::mem::offset_of!(EpilogueArgs, l_alloc) == 116);

/// Geometry of the stacked q, k, v projection of the rotary epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeLayout {
    /// Rows of the query part (`query heads * head_dim`).
    pub q_rows: usize,
    /// Rows of each of the key and value parts (`key-value heads * head_dim`).
    pub kv_rows: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Rotated coordinates per head (even, at most `head_dim`); the rest of a head is passed through.
    pub rope_dims: usize,
    /// Allocated length of the KV cache in positions.
    pub cache_len: usize,
}

/// Which kernel family a launch uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Epilogue {
    Plain = 0,
    Gated = 1,
    Rope = 2,
}

const EPILOGUE_NAMES: [&str; 3] = ["plain", "gated", "rope"];
/// Shared memory per block the plain kernels may use (the epilogue kernels have static shared memory as well).
const SHARED_PLAIN: usize = 99 * 1024;
const SHARED_EPILOGUE: usize = 96 * 1024;

/// The loaded projection kernels of one device.
pub struct Gemv {
    functions: Vec<[Function; 3]>,
    profile: &'static DeviceProfile,
    shared_limit: [usize; 3],
}

impl Gemv {
    /// Loads the kernels and selects the device profile.
    pub fn new(device: &Device) -> Result<Gemv> {
        let module = device.module("gemv")?;
        let optin = device.info().max_shared_memory_optin as usize;
        let shared_limit = [SHARED_PLAIN.min(optin), SHARED_EPILOGUE.min(optin), SHARED_EPILOGUE.min(optin)];
        let mut functions = Vec::with_capacity(ALL_CONFIGS.len());
        for cfg in ALL_CONFIGS {
            let load = |ep: Epilogue| -> Result<Function> {
                let f = module.function(&format!("gemv_{}_{}", cfg.tag(), EPILOGUE_NAMES[ep as usize]))?;
                f.set_max_dynamic_shared(shared_limit[ep as usize] as u32)?;
                Ok(f)
            };
            functions.push([load(Epilogue::Plain)?, load(Epilogue::Gated)?, load(Epilogue::Rope)?]);
        }
        Ok(Gemv { functions, profile: DeviceProfile::select(device.info()), shared_limit })
    }

    /// The device profile in use.
    pub fn profile(&self) -> &'static DeviceProfile {
        self.profile
    }

    /// Dynamic shared memory of a launch with `cfg` for `cols` columns.
    fn shared_bytes(cfg: &GemvConfig, cols: usize) -> usize {
        let bpr = blocks_per_row(cols);
        let staged = if cfg.act_from_global { 0 } else { act_entries_swizzled(bpr * 32) + 1 };
        (staged + 256) * 8
    }

    fn check_inputs(&self, rows: usize, cols: usize, weights: &DeviceBuffer<u8>, act: &DeviceBuffer<i8>, sum: &DeviceBuffer<i32>) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Err(Error::InvalidArgument("a projection needs at least one row and one column".into()));
        }
        if weights.len() < rows * row_bytes(cols) {
            return Err(Error::InvalidArgument(format!("weights hold {} bytes, a {rows}x{cols} matrix needs {}", weights.len(), rows * row_bytes(cols))));
        }
        if act.len() < act_entries(cols) * 8 {
            return Err(Error::InvalidArgument(format!("activation buffer holds {} bytes, {} are needed", act.len(), act_entries(cols) * 8)));
        }
        if sum.is_empty() {
            return Err(Error::InvalidArgument("the activation sum buffer is empty".into()));
        }
        Ok(())
    }

    fn index_of(cfg: &GemvConfig) -> Result<usize> {
        ALL_CONFIGS.iter().position(|c| c == cfg).ok_or_else(|| Error::InvalidArgument(format!("configuration {} is not instantiated", cfg.tag())))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &Stream,
        cfg: &GemvConfig,
        ep: Epilogue,
        rows: usize,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        sum: &DeviceBuffer<i32>,
        out: Option<&DeviceBuffer<i32>>,
        ep_args: EpilogueArgs,
    ) -> Result<()> {
        if rows % cfg.rows_per_block() as usize != 0 {
            return Err(Error::InvalidArgument(format!("{rows} rows are not a multiple of {} (configuration {})", cfg.rows_per_block(), cfg.tag())));
        }
        if ep != Epilogue::Plain && cfg.rows_per_warp % 2 != 0 {
            return Err(Error::InvalidArgument("the epilogues work on row pairs and need an even number of rows per warp".into()));
        }
        let shared = Self::shared_bytes(cfg, cols);
        if shared > self.shared_limit[ep as usize] {
            return Err(Error::InvalidArgument(format!("{cols} columns need {shared} bytes of shared memory, the limit is {}", self.shared_limit[ep as usize])));
        }
        let f = &self.functions[Self::index_of(cfg)?][ep as usize];
        let mut args = Args::new();
        args.buf(weights).buf(act).buf(sum);
        match out {
            Some(o) => args.buf(o),
            None => args.null(),
        };
        args.push(blocks_per_row(cols) as i32).push(ep_args);
        // SAFETY: the argument list matches `gemv_<tag>_<ep>(w, act_perm, act_sum, out, blocks_per_row, EpArgs)`; the buffer sizes were
        // checked against `rows` and `cols` above and the epilogue buffers by the callers.
        unsafe { f.launch(stream, ((rows / cfg.rows_per_block() as usize) as u32, 1, 1), (cfg.threads(), 1, 1), shared as u32, &args) }
    }

    /// Plain projection with an explicit configuration (tests and sweeps). `out` receives `rows` `i32` values.
    #[allow(clippy::too_many_arguments)]
    pub fn plain_with(
        &self,
        stream: &Stream,
        cfg: &GemvConfig,
        rows: usize,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        out: &DeviceBuffer<i32>,
    ) -> Result<()> {
        self.check_inputs(rows, cols, weights, act, act_sum)?;
        if out.len() < rows {
            return Err(Error::InvalidArgument("output buffer too small".into()));
        }
        self.launch(stream, cfg, Epilogue::Plain, rows, cols, weights, act, act_sum, Some(out), EpilogueArgs::default())
    }

    /// Plain projection with the configuration of the device profile for `role` ([`Role::Output`] or [`Role::Down`]).
    #[allow(clippy::too_many_arguments)]
    pub fn plain(
        &self,
        stream: &Stream,
        role: Role,
        rows: usize,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        out: &DeviceBuffer<i32>,
    ) -> Result<()> {
        let cfg = self.profile.config(role, rows as u32)?;
        self.plain_with(stream, &cfg, rows, cols, weights, act, act_sum, out)
    }

    /// Stacked, row-interleaved gate and up projection (`rows = 2 * intermediate`) with the gated activation in the epilogue.
    ///
    /// Writes `v[i] = act(gate_i) * up_i` for `i < rows / 2`, takes the maximum of `|v|` into `cells[par]` (as float bits, initialized
    /// to zero by the previous launch with the other parity) and clears `cells[par ^ 1]` and `sum_out[0]` for the next stage.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_with(
        &self,
        stream: &Stream,
        cfg: &GemvConfig,
        rows: usize,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        scales: (f32, f32),
        activation: Activation,
        v: &DeviceBuffer<f32>,
        cells: &DeviceBuffer<i32>,
        par: usize,
        sum_out: &DeviceBuffer<i32>,
    ) -> Result<()> {
        self.check_inputs(rows, cols, weights, act, act_sum)?;
        if rows % 2 != 0 || v.len() < rows / 2 || cells.len() < 2 || par > 1 || act_scale.is_empty() || sum_out.is_empty() {
            return Err(Error::InvalidArgument("gated epilogue: invalid output buffers".into()));
        }
        let ep = EpilogueArgs {
            act_scale: act_scale.ptr().0,
            ws_gate: scales.0,
            ws_up: scales.1,
            v: v.ptr().0,
            cells: cells.ptr().0,
            par: par as i32,
            activation: match activation {
                Activation::Silu => 0,
                Activation::Relu2 => 1,
            },
            sum_out: sum_out.ptr().0,
            ..EpilogueArgs::default()
        };
        self.launch(stream, cfg, Epilogue::Gated, rows, cols, weights, act, act_sum, None, ep)
    }

    /// [`Gemv::gated_with`] with the configuration of the device profile.
    #[allow(clippy::too_many_arguments)]
    pub fn gated(
        &self,
        stream: &Stream,
        rows: usize,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        scales: (f32, f32),
        activation: Activation,
        v: &DeviceBuffer<f32>,
        cells: &DeviceBuffer<i32>,
        par: usize,
        sum_out: &DeviceBuffer<i32>,
    ) -> Result<()> {
        let cfg = self.profile.config(Role::GateUp, rows as u32)?;
        self.gated_with(stream, &cfg, rows, cols, weights, act, act_sum, act_scale, scales, activation, v, cells, par, sum_out)
    }

    /// Stacked q, k, v projection (`rows = q_rows + 2 * kv_rows`) with the rotary embedding and the KV-cache append in the epilogue.
    ///
    /// `q_out` receives the rotated query as `f32`; the rotated key and the value of the token at position `*pos_dev` are written as
    /// F16 bit patterns to `k_cache` and `v_cache`, laid out `[kv_head][cache_len][head_dim]`. `cos_sin` holds `rope_dims` floats per
    /// position (cosine at `2 i`, sine at `2 i + 1` for the pair `i`), for at least `cache_len` positions. The pairing is interleaved
    /// (`x[2 i]` with `x[2 i + 1]`).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope_with(
        &self,
        stream: &Stream,
        cfg: &GemvConfig,
        layout: RopeLayout,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        scales: (f32, f32, f32),
        cos_sin: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
        q_out: &DeviceBuffer<f32>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
    ) -> Result<()> {
        let rows = layout.q_rows + 2 * layout.kv_rows;
        self.check_inputs(rows, cols, weights, act, act_sum)?;
        let hd = layout.head_dim;
        if hd == 0 || hd % 2 != 0 || layout.rope_dims % 2 != 0 || layout.rope_dims > hd || layout.q_rows % hd != 0 || layout.kv_rows % hd != 0 {
            return Err(Error::InvalidArgument("rope epilogue: inconsistent head geometry".into()));
        }
        let cache_elems = layout.kv_rows / hd * layout.cache_len * hd;
        if q_out.len() < layout.q_rows
            || k_cache.len() < cache_elems
            || v_cache.len() < cache_elems
            || cos_sin.len() < layout.cache_len * layout.rope_dims
            || pos_dev.is_empty()
            || act_scale.is_empty()
        {
            return Err(Error::InvalidArgument("rope epilogue: a buffer is too small".into()));
        }
        let ep = EpilogueArgs {
            act_scale: act_scale.ptr().0,
            q_out: q_out.ptr().0,
            kw: k_cache.ptr().0,
            vw: v_cache.ptr().0,
            cs_all: cos_sin.ptr().0,
            pos_dev: pos_dev.ptr().0,
            ws_q: scales.0,
            ws_k: scales.1,
            ws_v: scales.2,
            q_rows: layout.q_rows as i32,
            kv_rows: layout.kv_rows as i32,
            head_dim: hd as i32,
            rope_dims: layout.rope_dims as i32,
            l_alloc: layout.cache_len as i32,
            ..EpilogueArgs::default()
        };
        self.launch(stream, cfg, Epilogue::Rope, rows, cols, weights, act, act_sum, None, ep)
    }

    /// [`Gemv::qkv_rope_with`] with the configuration of the device profile.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope(
        &self,
        stream: &Stream,
        layout: RopeLayout,
        cols: usize,
        weights: &DeviceBuffer<u8>,
        act: &DeviceBuffer<i8>,
        act_sum: &DeviceBuffer<i32>,
        act_scale: &DeviceBuffer<f32>,
        scales: (f32, f32, f32),
        cos_sin: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
        q_out: &DeviceBuffer<f32>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
    ) -> Result<()> {
        let rows = layout.q_rows + 2 * layout.kv_rows;
        let cfg = self.profile.config(Role::QkvRope, rows as u32)?;
        self.qkv_rope_with(stream, &cfg, layout, cols, weights, act, act_sum, act_scale, scales, cos_sin, pos_dev, q_out, k_cache, v_cache)
    }
}

/// Start of the digit table in the shared memory of a kernel that stages `n` activation entries: the swizzled copy
/// (entry `e` at `e + e / 4`) occupies `swz(n)` entries and one more is left free, exactly as the kernel computes it.
const fn act_entries_swizzled(n: usize) -> usize {
    n + (n >> 2)
}
