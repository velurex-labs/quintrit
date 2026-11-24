//! Launchers of the kernels of the Nemotron-H mixture-of-experts decoder (`kernels/nemotron.cu`): `Q8_0` projections, the Mamba-2 layer, the key-value append of an
//! attention layer without rotary embedding and the mixture-of-experts layer. Every kernel is bit-identical to the CPU runtime; the kernel file says which routine
//! each one follows.

use crate::device::{Args, Device, DeviceBuffer, Function, Stream};
use crate::error::{Error, Result};
use crate::pack::row_bytes;
use crate::q2::SCALE_BLOCK;

fn check_len<T: Copy>(b: &DeviceBuffer<T>, need: usize, what: &str) -> Result<()> {
    if b.len() < need {
        return Err(Error::InvalidArgument(format!("{what}: buffer holds {} elements, {need} are needed", b.len())));
    }
    Ok(())
}

/// A `Q8_0` matrix on the device: int8 codes `[rows][cols]` and one F16 scale per 32 values.
pub struct Q8Matrix {
    qs: DeviceBuffer<i8>,
    scales: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Q8Matrix {
    /// Uploads a matrix; `cols` must be a multiple of 32 and divisible by 4 times 8 for the kernel's loads (a multiple of 32 is enough).
    pub fn new(device: &Device, qs: &[i8], scales_f16: &[u16], rows: usize, cols: usize) -> Result<Q8Matrix> {
        if rows == 0 || cols == 0 || cols % 32 != 0 || qs.len() != rows * cols || scales_f16.len() != rows * cols / 32 {
            return Err(Error::InvalidArgument(format!("a {rows} x {cols} Q8_0 matrix needs {} codes and {} scales", rows * cols, rows * cols / 32)));
        }
        Ok(Q8Matrix { qs: device.upload(qs)?, scales: device.upload(scales_f16)?, rows, cols })
    }

    /// Rows of the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// The experts of a mixture-of-experts layer, stacked: `n` block-scaled matrices of `rows x cols` weights in the planar layout.
pub struct BlockScaledExperts {
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
    n: usize,
    rows: usize,
    cols: usize,
}

impl BlockScaledExperts {
    /// Uploads the experts: `packed` holds the matrices one after the other (`rows * row_bytes(cols)` bytes each), `scales_f16` their `rows * cols / 64` scales.
    pub fn new(device: &Device, packed: &[u8], scales_f16: &[u16], n: usize, rows: usize, cols: usize) -> Result<BlockScaledExperts> {
        if n == 0 || rows == 0 || cols == 0 || cols % SCALE_BLOCK != 0 || packed.len() != n * rows * row_bytes(cols) || scales_f16.len() != n * rows * (cols / SCALE_BLOCK) {
            return Err(Error::InvalidArgument(format!("{n} experts of {rows} x {cols} weights do not match the given packed weights and scales")));
        }
        // The weights are regrouped like those of a `BlockScaledMatrix` (the experts are `n * rows` rows one after the other) but the rows are not padded: the experts are
        // most of the device memory of the model.
        Ok(BlockScaledExperts { packed: device.upload(&crate::q2::group_blocks(packed, n * rows, cols, false))?, scales: device.upload(scales_f16)?, n, rows, cols })
    }

    /// Number of experts.
    pub fn experts(&self) -> usize {
        self.n
    }

    /// Rows of every expert.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of every expert.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Geometry of a Mamba-2 layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MambaLayout {
    /// Inner size (`d_inner`).
    pub d_inner: usize,
    /// State size per channel (`d_state`, a multiple of 32 up to 256).
    pub d_state: usize,
    /// Number of groups of B and C.
    pub n_group: usize,
    /// Number of heads (a multiple of `n_group`, dividing `d_inner`).
    pub n_head: usize,
    /// Taps of the convolution.
    pub kernel: usize,
}

impl MambaLayout {
    /// Channels of the convolution: x, B and C.
    pub fn channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Floats of the convolution state of the layer.
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.channels()
    }

    /// Floats of the state of the layer.
    pub fn ssm_state_len(&self) -> usize {
        self.d_state * self.d_inner
    }

    /// Values the input projection produces: z, the convolution channels and the time steps.
    pub fn in_proj_rows(&self) -> usize {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }
}

/// The loaded kernels of the Nemotron-H decoder.
pub struct NemotronKernels {
    probe_logf: Function,
    gemv_q8: Function,
    conv: Function,
    scan: Function,
    swiglu2: Function,
    kv_append: Function,
    route: Function,
    gemv_moe: Function,
    relu_sqr: Function,
    combine: Function,
}

impl NemotronKernels {
    /// Loads the kernels.
    pub fn new(device: &Device) -> Result<NemotronKernels> {
        let module = device.module("nemotron")?;
        Ok(NemotronKernels {
            probe_logf: module.function("probe_logf")?,
            gemv_q8: module.function("gemv_q8")?,
            conv: module.function("mamba_conv")?,
            scan: module.function("mamba_scan")?,
            swiglu2: module.function("swiglu2")?,
            kv_append: module.function("kv_append_f32")?,
            route: module.function("moe_route")?,
            gemv_moe: module.function("gemv_q2_moe")?,
            relu_sqr: module.function("relu_sqr")?,
            combine: module.function("moe_combine")?,
        })
    }

    /// `out[i] = logf(x[i])` as glibc computes it, for tests.
    pub fn probe_logf(&self, stream: &Stream, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize) -> Result<()> {
        check_len(x, n, "logf input")?;
        check_len(out, n, "logf output")?;
        let mut a = Args::new();
        a.buf(x).buf(out).push(n as i32);
        // SAFETY: the arguments match `probe_logf`.
        unsafe { self.probe_logf.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `out[r] = dot_q8_0(row r of m, activations)` with the activations `qs` / `d` of `BlockScaledKernels::quantize` (`m.cols()` values).
    pub fn gemv_q8(&self, stream: &Stream, m: &Q8Matrix, qs: &DeviceBuffer<i8>, d: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>) -> Result<()> {
        check_len(qs, m.cols, "quantized activations")?;
        check_len(d, m.cols / 32, "activation scales")?;
        check_len(out, m.rows, "projection output")?;
        let mut a = Args::new();
        a.buf(&m.qs).buf(&m.scales).buf(qs).buf(d).buf(out).push(m.rows as i32).push(m.cols as i32);
        // SAFETY: the arguments match `gemv_q8`; the matrix was validated when it was built and the vectors above.
        unsafe { self.gemv_q8.launch(stream, (m.rows.div_ceil(32) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Mamba-2: the convolution of the channels `a[a_off..a_off + channels]` (the output of the input projection after `z`) with the state of the layer, the bias and
    /// SiLU into `out`. The state of a sequence that starts (`*pos_dev == 0`) is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        stream: &Stream,
        layout: MambaLayout,
        a: &DeviceBuffer<f32>,
        conv_state: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
    ) -> Result<()> {
        let ch = layout.channels();
        if layout.kernel < 2 || layout.kernel > 8 {
            return Err(Error::InvalidArgument("the convolution kernel must have 2 to 8 taps".into()));
        }
        check_len(a, layout.d_inner + ch, "convolution input")?;
        check_len(conv_state, layout.conv_state_len(), "convolution state")?;
        check_len(weights, ch * layout.kernel, "convolution weights")?;
        check_len(bias, ch, "convolution bias")?;
        check_len(out, ch, "convolution output")?;
        check_len(pos_dev, 1, "position")?;
        let mut args = Args::new();
        args.buf(a).push(layout.d_inner as i32).buf(conv_state).buf(weights).buf(bias).buf(out).buf(pos_dev).push(ch as i32).push(layout.kernel as i32);
        // SAFETY: the arguments match `mamba_conv`.
        unsafe { self.conv.launch(stream, (ch.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &args) }
    }

    /// Mamba-2: the state update and the output `y` (`d_inner` values) of one token. `a` is the output of the input projection (its last `n_head` values are the time
    /// steps), `conv` the output of [`NemotronKernels::conv`]; the state of a sequence that starts (`*pos_dev == 0`) is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn scan(
        &self,
        stream: &Stream,
        layout: MambaLayout,
        state: &DeviceBuffer<f32>,
        conv: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<f32>,
        a_param: &DeviceBuffer<f32>,
        d_param: &DeviceBuffer<f32>,
        y: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
    ) -> Result<()> {
        if layout.d_state % 32 != 0 || layout.d_state > 256 || layout.n_group == 0 || layout.n_head % layout.n_group != 0 || layout.d_inner % layout.n_head != 0 {
            return Err(Error::InvalidArgument("inconsistent Mamba-2 geometry".into()));
        }
        check_len(state, layout.ssm_state_len(), "state")?;
        check_len(conv, layout.channels(), "convolved values")?;
        check_len(a, layout.in_proj_rows(), "input projection")?;
        for (b, what) in [(dt_bias, "dt_bias"), (a_param, "A"), (d_param, "D")] {
            check_len(b, layout.n_head, what)?;
        }
        check_len(y, layout.d_inner, "scan output")?;
        check_len(pos_dev, 1, "position")?;
        let dt_off = 2 * layout.d_inner + 2 * layout.n_group * layout.d_state;
        let mut args = Args::new();
        args.buf(state).buf(conv).ptr(a.ptr().offset_bytes(dt_off * 4)).buf(dt_bias).buf(a_param).buf(d_param).buf(y).buf(pos_dev);
        args.push(layout.d_inner as i32).push(layout.d_state as i32).push(layout.n_group as i32).push(layout.n_head as i32);
        // SAFETY: the arguments match `mamba_scan`.
        unsafe { self.scan.launch(stream, (layout.d_inner.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &args) }
    }

    /// `out[i] = silu(gate[i]) * up[i]` for `n` values (a multiple of 8).
    pub fn swiglu(&self, stream: &Stream, gate: &DeviceBuffer<f32>, up: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize) -> Result<()> {
        if n % 8 != 0 {
            return Err(Error::InvalidArgument("swiglu works on a multiple of 8 values".into()));
        }
        check_len(gate, n, "gate")?;
        check_len(up, n, "up")?;
        check_len(out, n, "swiglu output")?;
        let mut a = Args::new();
        a.buf(gate).buf(up).buf(out).push(n as i32);
        // SAFETY: the arguments match `swiglu2`.
        unsafe { self.swiglu2.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Appends the key and the value of the new token (`qkv` holds `q (q_dim) | k | v`) as F16 at position `*pos_dev` of the caches `[kv_head][cache_len][head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        stream: &Stream,
        qkv: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
        k_cache: &DeviceBuffer<u16>,
        v_cache: &DeviceBuffer<u16>,
        kv_heads: usize,
        head_dim: usize,
        cache_len: usize,
        q_dim: usize,
    ) -> Result<()> {
        let kv_dim = kv_heads * head_dim;
        check_len(qkv, q_dim + 2 * kv_dim, "qkv")?;
        check_len(k_cache, kv_heads * cache_len * head_dim, "key cache")?;
        check_len(v_cache, kv_heads * cache_len * head_dim, "value cache")?;
        check_len(pos_dev, 1, "position")?;
        let mut a = Args::new();
        a.buf(qkv).buf(pos_dev).buf(k_cache).buf(v_cache).push(kv_heads as i32).push(head_dim as i32).push(cache_len as i32).push(q_dim as i32);
        // SAFETY: the arguments match `kv_append_f32`; the position is read on the device and must be below `cache_len` (the caller's contract).
        unsafe { self.kv_append.launch(stream, (1, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Routing of one token: `chosen` receives the indices of the `n_used` selected experts, `weights` their weights.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        stream: &Stream,
        logits: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        chosen: &DeviceBuffer<i32>,
        weights: &DeviceBuffer<f32>,
        n_experts: usize,
        n_used: usize,
        norm_weights: bool,
        weights_scale: f32,
    ) -> Result<()> {
        if n_experts == 0 || n_experts > 256 || n_used == 0 || n_used > 16 || n_used > n_experts {
            return Err(Error::InvalidArgument("the router supports up to 256 experts and 16 used experts".into()));
        }
        check_len(logits, n_experts, "router logits")?;
        check_len(bias, n_experts, "selection bias")?;
        check_len(chosen, n_used, "chosen experts")?;
        check_len(weights, n_used, "expert weights")?;
        let mut a = Args::new();
        a.buf(logits).buf(bias).buf(chosen).buf(weights).push(n_experts as i32).push(n_used as i32).push(norm_weights as i32).push(weights_scale);
        // SAFETY: the arguments match `moe_route`.
        unsafe { self.route.launch(stream, (1, 1, 1), (32, 1, 1), 0, &a) }
    }

    /// The projection of the `n_used` selected experts: slot `s` multiplies the matrix of expert `chosen[s]` with the activations at `qs[s * qs_stride..]` /
    /// `d[s * d_stride..]` and writes `experts.rows()` values to `out[s * out_stride..]`.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gemv(
        &self,
        stream: &Stream,
        experts: &BlockScaledExperts,
        chosen: &DeviceBuffer<i32>,
        n_used: usize,
        qs: &DeviceBuffer<i8>,
        qs_stride: usize,
        d: &DeviceBuffer<f32>,
        d_stride: usize,
        out: &DeviceBuffer<f32>,
        out_stride: usize,
    ) -> Result<()> {
        check_len(chosen, n_used, "chosen experts")?;
        check_len(qs, (n_used - 1) * qs_stride + experts.cols, "quantized activations")?;
        check_len(d, (n_used - 1) * d_stride + experts.cols / 32, "activation scales")?;
        check_len(out, (n_used - 1) * out_stride + experts.rows, "expert output")?;
        let mut a = Args::new();
        a.buf(&experts.packed).buf(&experts.scales).buf(qs).buf(d).buf(out).buf(chosen);
        a.push(experts.rows as i32).push(experts.cols as i32).push(qs_stride as i32).push(d_stride as i32).push(out_stride as i32);
        // SAFETY: the arguments match `gemv_q2_moe`; the expert indices come from the router and are below `experts.experts()` (the caller's contract).
        unsafe { self.gemv_moe.launch(stream, (experts.rows.div_ceil(32) as u32, n_used as u32, 1), (256, 1, 1), 0, &a) }
    }

    /// `x[i] = max(x[i], 0)^2` for `n` values.
    pub fn relu_sqr(&self, stream: &Stream, x: &DeviceBuffer<f32>, n: usize) -> Result<()> {
        check_len(x, n, "relu input")?;
        let mut a = Args::new();
        a.buf(x).push(n as i32);
        // SAFETY: the arguments match `relu_sqr`.
        unsafe { self.relu_sqr.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// The combination of the selected experts: `out = (head_0 * w_0 + head_1 * w_1 + ...) + shared` with the sum in the order of the slots; `heads` holds `n_used` rows
    /// of `n` values.
    #[allow(clippy::too_many_arguments)]
    pub fn combine(
        &self,
        stream: &Stream,
        heads: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<f32>,
        shared: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        n: usize,
        n_used: usize,
    ) -> Result<()> {
        check_len(heads, n * n_used, "expert outputs")?;
        check_len(weights, n_used, "expert weights")?;
        check_len(shared, n, "shared expert output")?;
        check_len(out, n, "combined output")?;
        let mut a = Args::new();
        a.buf(heads).buf(weights).buf(shared).buf(out).push(n as i32).push(n_used as i32);
        // SAFETY: the arguments match `moe_combine`.
        unsafe { self.combine.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }
}
