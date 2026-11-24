//! Launchers of the kernels of the hybrid decoders of the Qwen3.5 family (`kernels/hybrid.cu`): the Hadamard transform of folded weights, F32 and BF16
//! projections, the Gated DeltaNet layer and the gate of the gated attention. Every kernel is bit-identical to the CPU runtime; the kernel file says which
//! routine each one follows.

use crate::device::{Args, Device, DeviceBuffer, Function, Stream};
use crate::error::{Error, Result};

fn check_len<T: Copy>(b: &DeviceBuffer<T>, need: usize, what: &str) -> Result<()> {
    if b.len() < need {
        return Err(Error::InvalidArgument(format!("{what}: buffer holds {} elements, {need} are needed", b.len())));
    }
    Ok(())
}

/// The input permutation of a Gated DeltaNet output projection whose value heads are grouped: the input `[head_dim, k_heads, repeats]` is reordered to
/// `[head_dim, repeats, k_heads]` before the transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueHeadPermutation {
    /// Values per head.
    pub head_dim: usize,
    /// Number of key heads.
    pub k_heads: usize,
    /// Value heads per key head.
    pub repeats: usize,
}

/// Geometry of the Gated DeltaNet kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaNetLayout {
    /// Number of key (and query) heads.
    pub k_heads: usize,
    /// Number of value heads (a multiple of `k_heads`).
    pub v_heads: usize,
    /// Head dimension of keys, queries and values (a multiple of 32, at most 256).
    pub head_dim: usize,
    /// Taps of the convolution.
    pub kernel: usize,
}

impl DeltaNetLayout {
    /// Channels of the convolution: queries, keys and values.
    pub fn channels(&self) -> usize {
        2 * self.k_heads * self.head_dim + self.v_heads * self.head_dim
    }

    /// Floats of the convolution state of the layer.
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.channels()
    }

    /// Floats of the recurrent state of the layer.
    pub fn ssm_state_len(&self) -> usize {
        self.v_heads * self.head_dim * self.head_dim
    }
}

/// The loaded kernels of the hybrid decoders.
pub struct HybridKernels {
    hadamard: Function,
    gemv_f32: Function,
    gemv_bf16: Function,
    gates: Function,
    conv: Function,
    l2: Function,
    recur: Function,
    out_norm: Function,
    attn_gate: Function,
    probe: Function,
}

impl HybridKernels {
    /// Loads the kernels.
    pub fn new(device: &Device) -> Result<HybridKernels> {
        let module = device.module("hybrid")?;
        let hadamard = module.function("hadamard_blocks")?;
        hadamard.set_max_dynamic_shared(MAX_HADAMARD_BLOCK as u32 * 4)?;
        Ok(HybridKernels {
            hadamard,
            gemv_f32: module.function("gemv_f32_dot")?,
            gemv_bf16: module.function("gemv_bf16_dot")?,
            gates: module.function("gdn_gates")?,
            conv: module.function("gdn_conv")?,
            l2: module.function("gdn_l2")?,
            recur: module.function("gdn_recur")?,
            out_norm: module.function("gdn_out_norm")?,
            attn_gate: module.function("attn_gate")?,
            probe: module.function("probe_unary")?,
        })
    }

    /// Hadamard transform of the `n` values of `x` into `out`, in blocks of `block` (a power of two dividing `n`, at most [`MAX_HADAMARD_BLOCK`]): the optional
    /// permutation of the value heads, the optional signs (`n` values of +-1), then the normalized Walsh-Hadamard transform of every block.
    #[allow(clippy::too_many_arguments)]
    pub fn hadamard(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        n: usize,
        block: usize,
        signs: Option<&DeviceBuffer<f32>>,
        perm: Option<ValueHeadPermutation>,
    ) -> Result<()> {
        if block == 0 || !block.is_power_of_two() || block > MAX_HADAMARD_BLOCK || n % block != 0 {
            return Err(Error::InvalidArgument(format!("a Hadamard block of {block} values does not divide {n} values (a power of two of at most {MAX_HADAMARD_BLOCK})")));
        }
        check_len(x, n, "Hadamard input")?;
        check_len(out, n, "Hadamard output")?;
        if let Some(s) = signs {
            check_len(s, n, "Hadamard signs")?;
        }
        let (phd, pnk, prep) = match perm {
            Some(p) => {
                if p.head_dim * p.k_heads * p.repeats != n {
                    return Err(Error::InvalidArgument("the value-head permutation does not cover the input".into()));
                }
                (p.head_dim, p.k_heads, p.repeats)
            }
            None => (0, 0, 0),
        };
        let scale = 1.0f32 / (block as f32).sqrt();
        let mut a = Args::new();
        a.buf(x).buf(out);
        match signs {
            Some(s) => a.buf(s),
            None => a.null(),
        };
        a.push(block as i32).push(scale).push(phd as i32).push(pnk as i32).push(prep as i32);
        let threads = (block / 2).clamp(32, 1024) as u32;
        // SAFETY: the arguments match `hadamard_blocks`; the buffers were checked above and the dynamic shared memory holds one block.
        unsafe { self.hadamard.launch(stream, ((n / block) as u32, 1, 1), (threads, 1, 1), (block * 4) as u32, &a) }
    }

    /// `out[out_off + r] = ggml_vec_dot_f32(w row r, x)` for `rows` rows of `cols` values (a multiple of 32).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f32(&self, stream: &Stream, w: &DeviceBuffer<f32>, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, out_off: usize, rows: usize, cols: usize) -> Result<()> {
        if rows == 0 || cols == 0 || cols % 32 != 0 {
            return Err(Error::InvalidArgument("an F32 projection needs a column count that is a multiple of 32".into()));
        }
        check_len(w, rows * cols, "F32 weights")?;
        check_len(x, cols, "F32 projection input")?;
        check_len(out, out_off + rows, "F32 projection output")?;
        let mut a = Args::new();
        a.buf(w).buf(x).ptr(out.ptr().offset_bytes(out_off * 4)).push(rows as i32).push(cols as i32);
        // SAFETY: the arguments match `gemv_f32_dot`.
        unsafe { self.gemv_f32.launch(stream, (rows.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `out[out_off + r] = ggml_vec_dot_bf16(w row r, bf16(x))` for `rows` rows of `cols` values (a multiple of 32); `w` holds BF16 bit patterns.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16(&self, stream: &Stream, w: &DeviceBuffer<u16>, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, out_off: usize, rows: usize, cols: usize) -> Result<()> {
        if rows == 0 || cols == 0 || cols % 32 != 0 {
            return Err(Error::InvalidArgument("a BF16 projection needs a column count that is a multiple of 32".into()));
        }
        check_len(w, rows * cols, "BF16 weights")?;
        check_len(x, cols, "BF16 projection input")?;
        check_len(out, out_off + rows, "BF16 projection output")?;
        let mut a = Args::new();
        a.buf(w).buf(x).ptr(out.ptr().offset_bytes(out_off * 4)).push(rows as i32).push(cols as i32);
        // SAFETY: the arguments match `gemv_bf16_dot`.
        unsafe { self.gemv_bf16.launch(stream, (rows.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Gated DeltaNet: `beta = sigmoid(beta_raw)` and `gate = softplus(alpha_raw + dt_bias) * a` for every value head.
    #[allow(clippy::too_many_arguments)]
    pub fn gates(
        &self,
        stream: &Stream,
        beta_raw: &DeviceBuffer<f32>,
        alpha_raw: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<f32>,
        a_param: &DeviceBuffer<f32>,
        beta: &DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        v_heads: usize,
    ) -> Result<()> {
        for (b, what) in [(beta_raw, "beta projection"), (alpha_raw, "alpha projection"), (dt_bias, "dt_bias"), (a_param, "A"), (beta, "beta"), (gate, "gate")] {
            check_len(b, v_heads, what)?;
        }
        let mut a = Args::new();
        a.buf(beta_raw).buf(alpha_raw).buf(dt_bias).buf(a_param).buf(beta).buf(gate).push(v_heads as i32);
        // SAFETY: the arguments match `gdn_gates`.
        unsafe { self.gates.launch(stream, (v_heads.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Gated DeltaNet: the causal convolution of the projected `[q | k | v]` (`a`) with the state of the layer, then SiLU; the state of a sequence that starts
    /// (`*pos_dev == 0`) is zero.
    pub fn conv(
        &self,
        stream: &Stream,
        layout: DeltaNetLayout,
        a: &DeviceBuffer<f32>,
        conv_state: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
    ) -> Result<()> {
        let ch = layout.channels();
        if layout.kernel < 2 || layout.kernel > 8 {
            return Err(Error::InvalidArgument("the convolution kernel must have 2 to 8 taps".into()));
        }
        check_len(a, ch, "convolution input")?;
        check_len(conv_state, layout.conv_state_len(), "convolution state")?;
        check_len(weights, ch * layout.kernel, "convolution weights")?;
        check_len(out, ch, "convolution output")?;
        check_len(pos_dev, 1, "position")?;
        let mut args = Args::new();
        args.buf(a).buf(conv_state).buf(weights).buf(out).buf(pos_dev).push(ch as i32).push(layout.kernel as i32);
        // SAFETY: the arguments match `gdn_conv`.
        unsafe { self.conv.launch(stream, (ch.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &args) }
    }

    /// Gated DeltaNet: L2 normalization of the query and key heads (the first `2 * k_heads` heads of `qkv`) in place; `eps` is the epsilon of the layer.
    pub fn l2_norm(&self, stream: &Stream, layout: DeltaNetLayout, qkv: &DeviceBuffer<f32>, eps: f32) -> Result<()> {
        let n = layout.head_dim as f32;
        check_len(qkv, layout.channels(), "query and key heads")?;
        let mut a = Args::new();
        a.buf(qkv).push(layout.head_dim as i32).push(eps / n).push(1.0f32 / n.sqrt());
        // SAFETY: the arguments match `gdn_l2`.
        unsafe { self.l2.launch(stream, ((2 * layout.k_heads) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Gated DeltaNet: the recurrence of one token. `conv` holds the convolved and normalized `[q | k | v]`, `ssm_state` the state of the layer (zero when
    /// `*pos_dev == 0`), `out` receives `v_heads * head_dim` values.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrence(
        &self,
        stream: &Stream,
        layout: DeltaNetLayout,
        ssm_state: &DeviceBuffer<f32>,
        conv: &DeviceBuffer<f32>,
        beta: &DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        pos_dev: &DeviceBuffer<i32>,
    ) -> Result<()> {
        let hd = layout.head_dim;
        if hd % 32 != 0 || hd > 256 || layout.k_heads == 0 || layout.v_heads % layout.k_heads != 0 {
            return Err(Error::InvalidArgument("the Gated DeltaNet head dimension must be a multiple of 32 up to 256 and the value heads a multiple of the key heads".into()));
        }
        check_len(ssm_state, layout.ssm_state_len(), "recurrent state")?;
        check_len(conv, layout.channels(), "convolved values")?;
        check_len(beta, layout.v_heads, "beta")?;
        check_len(gate, layout.v_heads, "gate")?;
        check_len(out, layout.v_heads * hd, "recurrence output")?;
        check_len(pos_dev, 1, "position")?;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut a = Args::new();
        a.buf(ssm_state).buf(conv).buf(beta).buf(gate).buf(out).buf(pos_dev);
        a.push(layout.k_heads as i32).push(layout.v_heads as i32).push(hd as i32).push(scale);
        let warps = layout.v_heads * hd;
        // SAFETY: the arguments match `gdn_recur`.
        unsafe { self.recur.launch(stream, (warps.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Gated DeltaNet: `out = ((x * scale) * w) * silu(z)` per value head (a gated RMSNorm); `x`, `z` and `out` hold `v_heads * head_dim` values, `w` `head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_norm(
        &self,
        stream: &Stream,
        layout: DeltaNetLayout,
        x: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        z: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        eps: f32,
    ) -> Result<()> {
        let n = layout.v_heads * layout.head_dim;
        check_len(x, n, "gated norm input")?;
        check_len(z, n, "gate")?;
        check_len(out, n, "gated norm output")?;
        check_len(w, layout.head_dim, "norm weights")?;
        let mut a = Args::new();
        a.buf(x).buf(w).buf(z).buf(out).push(layout.head_dim as i32).push(eps);
        // SAFETY: the arguments match `gdn_out_norm`.
        unsafe { self.out_norm.launch(stream, (layout.v_heads as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Gated attention: `att[h][e] *= sigmoid(joint[2 h hd + hd + e])`, the gate of every head being stored after its query in the joint projection.
    pub fn attention_gate(&self, stream: &Stream, att: &DeviceBuffer<f32>, joint: &DeviceBuffer<f32>, heads: usize, head_dim: usize) -> Result<()> {
        let n = heads * head_dim;
        check_len(att, n, "attention output")?;
        check_len(joint, 2 * n, "joint projection")?;
        let mut a = Args::new();
        a.buf(att).buf(joint).push(head_dim as i32).push(n as i32);
        // SAFETY: the arguments match `attn_gate`.
        unsafe { self.attn_gate.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Applies one scalar function to every value, for tests: `which` 0 = `expf` (glibc), 1 = sigmoid, 2 = softplus, 3 = `log1pf`, 4 = SiLU.
    pub fn probe(&self, stream: &Stream, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize, which: i32) -> Result<()> {
        check_len(x, n, "probe input")?;
        check_len(out, n, "probe output")?;
        let mut a = Args::new();
        a.buf(x).buf(out).push(n as i32).push(which);
        // SAFETY: the arguments match `probe_unary`.
        unsafe { self.probe.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }
}

/// Largest Hadamard block the kernel supports.
pub const MAX_HADAMARD_BLOCK: usize = 8192;
