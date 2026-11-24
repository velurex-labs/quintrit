//! Launchers of the kernels for block-scaled ternary models (`kernels/q2.cu`): the weights of llama.cpp's `Q2_0` format (one scale per 64 weights) multiplied
//! with activations quantized in blocks of 32 (`Q8_0`). Every kernel is bit-identical to the CPU runtime; the kernel file says which routine each one follows.
//!
//! The weights come in the planar layout of [`crate::pack`] (the same bytes as the CPU runtime's block-scaled layer) with one F16 scale per 64 weights of a row.
//! [`BlockScaledMatrix::new`] regroups the weight words on the way to the device (a run-time layout, not a file format); the arithmetic is unchanged.

use crate::device::{Args, Device, DeviceBuffer, Function, Stream};
use crate::error::{Error, Result};
use crate::layer::{RopeKvLayout, RopeMode};
use crate::pack::row_bytes;

/// Weights per scale (llama.cpp `QK2_0`).
pub const SCALE_BLOCK: usize = 64;
/// Activations per quantization block (llama.cpp `QK8_0`).
pub const ACT_BLOCK: usize = 32;

/// Blocks of 256 threads per multiprocessor below which `gemv_q2_split` is used.
const SPLIT_BLOCKS_PER_SM: usize = 6;

/// Weight blocks per group of the run-time layout of [`BlockScaledMatrix`].
const GROUP_BLOCKS: usize = 4;

/// Rearranges the planar rows (blocks of 32 bytes = 8 lane words) into the run-time layout of the projection kernels (`kernels/q2_words.cuh`): per row the blocks are
/// taken in groups of four; in a full group the words of the four blocks of lane `j` are adjacent (`[lane][block in group]`), so that a thread reads them with one
/// 16-byte load; the last group of one to three blocks is stored as `[lane][block]`. With `align_rows` a row is padded to whole groups of 128 bytes (the padding is
/// never read) so that every row starts on a 128-byte line, which the kernels read measurably faster; without it the size of a row is unchanged (for the experts of
/// a mixture-of-experts model, where device memory is the limit).
pub(crate) fn group_blocks(packed: &[u8], rows: usize, cols: usize, align_rows: bool) -> Vec<u8> {
    let rb = row_bytes(cols);
    let blocks = rb / 32;
    let tail = blocks % GROUP_BLOCKS;
    let full_groups = blocks / GROUP_BLOCKS;
    let new_rb = if align_rows { blocks.div_ceil(GROUP_BLOCKS) * GROUP_BLOCKS * 32 } else { rb };
    let mut out = vec![0u8; rows * new_rb];
    for r in 0..rows {
        let src = &packed[r * rb..(r + 1) * rb];
        let dst = &mut out[r * new_rb..(r + 1) * new_rb];
        for b in 0..blocks {
            let (g, u) = (b / GROUP_BLOCKS, b % GROUP_BLOCKS);
            for j in 0..8 {
                let from = b * 32 + j * 4;
                let word = if g < full_groups { j * GROUP_BLOCKS + u } else { j * tail + u };
                let to = g * GROUP_BLOCKS * 32 + word * 4;
                dst[to..to + 4].copy_from_slice(&src[from..from + 4]);
            }
        }
    }
    out
}

/// A block-scaled ternary matrix on the device: `rows` rows of `cols` weights.
pub struct BlockScaledMatrix {
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl BlockScaledMatrix {
    /// Uploads a matrix. `packed` holds the rows in the planar layout (`row_bytes(cols)` bytes each, see [`crate::pack`]), `scales_f16` the F16 bit patterns of the
    /// `cols / 64` scales of each row. `cols` must be a multiple of 64.
    pub fn new(device: &Device, packed: &[u8], scales_f16: &[u16], rows: usize, cols: usize) -> Result<BlockScaledMatrix> {
        if rows == 0 || cols == 0 || cols % SCALE_BLOCK != 0 {
            return Err(Error::InvalidArgument(format!("a {rows} x {cols} block-scaled matrix needs a column count that is a multiple of {SCALE_BLOCK}")));
        }
        if packed.len() != rows * row_bytes(cols) {
            return Err(Error::InvalidArgument(format!("packed weights hold {} bytes, {} are needed", packed.len(), rows * row_bytes(cols))));
        }
        if scales_f16.len() != rows * (cols / SCALE_BLOCK) {
            return Err(Error::InvalidArgument(format!("{} scales given, {} are needed", scales_f16.len(), rows * (cols / SCALE_BLOCK))));
        }
        Ok(BlockScaledMatrix { packed: device.upload(&group_blocks(packed, rows, cols, true))?, scales: device.upload(scales_f16)?, rows, cols })
    }

    /// Rows of the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Device memory of the weights and the scales in bytes.
    pub fn device_bytes(&self) -> usize {
        self.packed.byte_len() + self.scales.byte_len()
    }
}

/// The loaded kernels of block-scaled models.
pub struct BlockScaledKernels {
    quant: Function,
    gemv: Function,
    gemv_split: Function,
    sm_count: u32,
    rms_norm_rows: Function,
    rope_kv: Function,
    swiglu: Function,
    add: Function,
}

fn check_len<T: Copy>(b: &DeviceBuffer<T>, need: usize, what: &str) -> Result<()> {
    if b.len() < need {
        return Err(Error::InvalidArgument(format!("{what}: buffer holds {} elements, {need} are needed", b.len())));
    }
    Ok(())
}

impl BlockScaledKernels {
    /// Loads the kernels.
    pub fn new(device: &Device) -> Result<BlockScaledKernels> {
        let module = device.module("q2")?;
        Ok(BlockScaledKernels {
            quant: module.function("q8_quant")?,
            gemv: module.function("gemv_q2")?,
            gemv_split: module.function("gemv_q2_split")?,
            sm_count: device.info().sm_count,
            rms_norm_rows: module.function("rms_norm_rows")?,
            rope_kv: module.function("rope_kv_f32")?,
            swiglu: module.function("swiglu_f32")?,
            add: module.function("add_f32")?,
        })
    }

    /// `Q8_0` quantization of `cols` values (a multiple of 32): `qs` receives the int8 codes, `d` the F32 scale of each block of 32.
    pub fn quantize(&self, stream: &Stream, x: &DeviceBuffer<f32>, qs: &DeviceBuffer<i8>, d: &DeviceBuffer<f32>, cols: usize) -> Result<()> {
        if cols == 0 || cols % ACT_BLOCK != 0 {
            return Err(Error::InvalidArgument("Q8_0 quantization needs a multiple of 32 values".into()));
        }
        check_len(x, cols, "quantization input")?;
        check_len(qs, cols, "quantized values")?;
        check_len(d, cols / ACT_BLOCK, "quantization scales")?;
        let blocks = cols / ACT_BLOCK;
        let mut a = Args::new();
        a.buf(x).buf(qs).buf(d).push(blocks as i32);
        // SAFETY: the arguments match `q8_quant` and the buffers were checked above.
        unsafe { self.quant.launch(stream, (blocks.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `out[r] = dot(row r of m, activations)` for every row, with the activations `qs` / `d` of [`BlockScaledKernels::quantize`] (`m.cols()` values).
    pub fn gemv(&self, stream: &Stream, m: &BlockScaledMatrix, qs: &DeviceBuffer<i8>, d: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>) -> Result<()> {
        self.gemv_into(stream, m, qs, d, out, 0)
    }

    /// [`BlockScaledKernels::gemv`] writing the rows to `out[out_off..out_off + m.rows()]`.
    pub fn gemv_into(
        &self,
        stream: &Stream,
        m: &BlockScaledMatrix,
        qs: &DeviceBuffer<i8>,
        d: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        out_off: usize,
    ) -> Result<()> {
        check_len(qs, m.cols, "quantized activations")?;
        check_len(d, m.cols / ACT_BLOCK, "activation scales")?;
        check_len(out, out_off + m.rows, "projection output")?;
        let mut a = Args::new();
        a.buf(&m.packed).buf(&m.scales).buf(qs).buf(d).ptr(out.ptr().offset_bytes(out_off * 4)).push(m.rows as i32).push(m.cols as i32);
        // A matrix with so few rows that 32 rows per block leave fewer than six blocks per multiprocessor (what one multiprocessor holds at 256 threads) runs
        // with the two FMA chains of a lane in two warps: twice the threads for the same rows, the same results.
        let blocks = m.rows.div_ceil(32);
        // SAFETY: the arguments match `gemv_q2` and `gemv_q2_split`; the matrix was validated when it was built and the vectors above.
        unsafe {
            if blocks < SPLIT_BLOCKS_PER_SM * self.sm_count as usize {
                self.gemv_split.launch(stream, (m.rows.div_ceil(16) as u32, 1, 1), (256, 1, 1), 0, &a)
            } else {
                self.gemv.launch(stream, (blocks as u32, 1, 1), (256, 1, 1), 0, &a)
            }
        }
    }

    /// RMSNorm of `rows` consecutive rows of `cols` values starting at element `x_off` of `x`, written from element `out_off` of `out` (the same buffer and
    /// offsets are allowed). `w` holds the `cols` weights shared by the rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_rows(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        x_off: usize,
        w: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        out_off: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_strided(stream, x, x_off, cols, w, out, out_off, cols, rows, cols, eps)
    }

    /// [`BlockScaledKernels::rms_norm_rows`] with rows that start `in_stride` values apart in `x` and `out_stride` values apart in `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_strided(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        x_off: usize,
        in_stride: usize,
        w: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
        out_off: usize,
        out_stride: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        self.rms_norm_grouped(stream, x, x_off, in_stride, w, 0, out, out_off, out_stride, rows, cols, eps)
    }

    /// [`BlockScaledKernels::rms_norm_strided`] with a weight vector of its own for every row: row `r` uses `w[r * w_stride..r * w_stride + cols]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_grouped(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        x_off: usize,
        in_stride: usize,
        w: &DeviceBuffer<f32>,
        w_stride: usize,
        out: &DeviceBuffer<f32>,
        out_off: usize,
        out_stride: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        if in_stride < cols || out_stride < cols {
            return Err(Error::InvalidArgument("norm rows overlap".into()));
        }
        check_len(x, x_off + (rows - 1) * in_stride + cols, "norm input")?;
        check_len(out, out_off + (rows - 1) * out_stride + cols, "norm output")?;
        check_len(w, (rows - 1) * w_stride + cols, "norm weights")?;
        let mut a = Args::new();
        a.buf(x).buf(w).buf(out).push(cols as i32).push(eps).push(x_off as i32).push(out_off as i32).push(in_stride as i32).push(out_stride as i32).push(w_stride as i32);
        // SAFETY: the arguments match `rms_norm_rows` and the ranges were checked above.
        unsafe { self.rms_norm_rows.launch(stream, (rows as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// Rotary embedding and KV append for the F32 result `qkv` (`q | k | v`) of the stacked projection: rotated queries to `q_out`, rotated keys and the values
    /// as F16 into the caches at position `*pos_dev`. `cos_sin` holds `rope_dims` floats per position.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv(
        &self,
        stream: &Stream,
        layout: RopeKvLayout,
        qkv: &DeviceBuffer<f32>,
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
        check_len(pos_dev, 1, "position")?;
        let mut a = Args::new();
        a.buf(qkv).buf(cos_sin).buf(pos_dev).buf(q_out).buf(k_cache).buf(v_cache);
        a.push(layout.q_heads as i32).push(layout.kv_heads as i32).push(hd as i32).push(layout.rope_dims as i32);
        a.push((layout.mode == RopeMode::HalfSplit) as i32).push(layout.cache_len as i32);
        // SAFETY: the arguments match `rope_kv_f32`; the position is read on the device and must be below `cache_len` (the caller's contract).
        unsafe { self.rope_kv.launch(stream, (1, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `out[i] = silu(gate_up[i]) * gate_up[n + i]` for `i < n` (`n` a multiple of 8).
    pub fn swiglu(&self, stream: &Stream, gate_up: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize) -> Result<()> {
        if n % 8 != 0 {
            return Err(Error::InvalidArgument("swiglu works on a multiple of 8 values".into()));
        }
        check_len(gate_up, 2 * n, "gate and up")?;
        check_len(out, n, "swiglu output")?;
        let mut a = Args::new();
        a.buf(gate_up).buf(out).push(n as i32);
        // SAFETY: the arguments match `swiglu_f32`.
        unsafe { self.swiglu.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }

    /// `x[i] += r[i]` for `i < n`.
    pub fn add(&self, stream: &Stream, x: &DeviceBuffer<f32>, r: &DeviceBuffer<f32>, n: usize) -> Result<()> {
        check_len(x, n, "add target")?;
        check_len(r, n, "add operand")?;
        let mut a = Args::new();
        a.buf(x).buf(r).push(n as i32);
        // SAFETY: the arguments match `add_f32`.
        unsafe { self.add.launch(stream, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, &a) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct bytes per word, so that every word can be traced: the word of lane `j` of block `b` of row `r` is `1 + r * 1000 + b * 8 + j` (little endian).
    fn planar(rows: usize, cols: usize) -> Vec<u8> {
        let blocks = row_bytes(cols) / 32;
        let mut v = Vec::new();
        for r in 0..rows {
            for b in 0..blocks {
                for j in 0..8 {
                    v.extend_from_slice(&((1 + r * 1000 + b * 8 + j) as u32).to_le_bytes());
                }
            }
        }
        v
    }

    fn word(buf: &[u8], index: usize) -> u32 {
        u32::from_le_bytes(buf[index * 4..index * 4 + 4].try_into().unwrap())
    }

    #[test]
    fn grouped_layout_places_every_word_where_the_kernel_reads_it() {
        // 160 columns per block: 1, 3, 4, 5, 6, 7 and 26 blocks cover every tail length (0 to 3 blocks) and several full groups.
        for blocks in [1usize, 3, 4, 5, 6, 7, 26] {
            for align_rows in [false, true] {
                let cols = blocks * 160;
                let rows = 3;
                let src = planar(rows, cols);
                let out = group_blocks(&src, rows, cols, align_rows);
                let (full, tail) = (blocks / 4, blocks % 4);
                // the row stride the kernels use: the blocks, or whole groups of four blocks when the rows are aligned
                let row_words = if align_rows { blocks.div_ceil(4) * 32 } else { blocks * 8 };
                assert_eq!(out.len(), rows * row_words * 4, "{blocks} blocks, aligned {align_rows}: size");
                if !align_rows {
                    assert_eq!(out.len(), src.len(), "{blocks} blocks: without alignment the size of the matrix must not change");
                }
                for r in 0..rows {
                    for b in 0..blocks {
                        for j in 0..8 {
                            let (g, u) = (b / 4, b % 4);
                            // the index the kernel computes (`load_group` of q2_words.cuh)
                            let at = if g < full { g * 32 + 4 * j + u } else { g * 32 + tail * j + u };
                            assert_eq!(word(&out, r * row_words + at), (1 + r * 1000 + b * 8 + j) as u32, "{blocks} blocks, aligned {align_rows}, row {r}, block {b}, lane {j}");
                        }
                    }
                }
            }
        }
    }
}
