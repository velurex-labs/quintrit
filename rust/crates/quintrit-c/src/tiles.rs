//! Kernel level: tiled weight layout, pack / unpack and GEMV on int8 activations.

use crate::error::{
    fail, guard, Res, QUINTRIT_ERR_BUFFER_TOO_SMALL, QUINTRIT_ERR_INVALID_SHAPE, QUINTRIT_ERR_INVALID_VALUE,
};
use crate::kernel::{resolve, resolve_threads, Kernel};
use crate::util::{check_disjoint, check_out_struct, in_slice, out_slice};
use quintrit_core::pack::{pack_matrix, tile_byte, tile_row_bytes, unpack_matrix, untile_row, BLOCK_BYTES_5, QK_5};
use std::ffi::c_int;

/// Largest accepted row or column count.
pub const MAX_DIM: usize = 1 << 31;
/// Largest column count accepted by the GEMV functions (defined in `quintrit-core`: the accumulators are exact
/// integers only while `127 * cols < 2^24`).
pub use quintrit_core::MAX_GEMV_COLS;

/// Describes the tiled layout of a `rows x cols` matrix for one kernel (`quintrit_tiled_layout_t`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TiledLayout {
    pub struct_size: u32,
    /// Concrete kernel id the layout belongs to (`AUTO` is resolved).
    pub kernel: i32,
    /// Rows per tile: 32 or 64.
    pub tile_rows: u64,
    pub rows: u64,
    pub cols: u64,
    /// `rows` rounded up to a multiple of `tile_rows`; the padding rows hold zero weights.
    pub padded_rows: u64,
    /// Bytes of one row inside a tile (velurex encoding, 5 trits per byte; the partial last block is packed).
    pub bytes_per_row: u64,
    /// Total size of the tiled buffer: `padded_rows * bytes_per_row`.
    pub total_bytes: u64,
}

/// Computes the layout of a `rows x cols` matrix with checked arithmetic.
pub fn layout(kernel: Kernel, rows: usize, cols: usize) -> Res<TiledLayout> {
    if rows == 0 || cols == 0 {
        fail!(QUINTRIT_ERR_INVALID_SHAPE, "rows and cols must be at least 1 (got {rows} x {cols})");
    }
    if rows > MAX_DIM || cols > MAX_DIM {
        fail!(QUINTRIT_ERR_INVALID_SHAPE, "rows and cols must not exceed {MAX_DIM} (got {rows} x {cols})");
    }
    let tile = kernel.tile_rows();
    let padded_rows = rows.next_multiple_of(tile);
    let bytes_per_row = tile_row_bytes(cols);
    let total = padded_rows
        .checked_mul(bytes_per_row)
        .filter(|&t| t <= isize::MAX as usize)
        .ok_or_else(|| crate::error::Failure::new(QUINTRIT_ERR_INVALID_SHAPE, format!("{rows} x {cols} is too large")))?;
    // rows * cols must also be addressable (the trit matrix).
    if rows.checked_mul(cols).is_none_or(|n| n > isize::MAX as usize) {
        fail!(QUINTRIT_ERR_INVALID_SHAPE, "{rows} x {cols} is too large");
    }
    Ok(TiledLayout {
        struct_size: size_of::<TiledLayout>() as u32,
        kernel: kernel.id(),
        tile_rows: tile as u64,
        rows: rows as u64,
        cols: cols as u64,
        padded_rows: padded_rows as u64,
        bytes_per_row: bytes_per_row as u64,
        total_bytes: total as u64,
    })
}

/// Packs `rows x cols` trits into the tiled layout of `kernel`; padding rows get zero weights.
///
/// The caller has validated that every trit is in `{-1, 0, +1}` and that `out` holds `layout.total_bytes`.
pub fn pack_tiled(kernel: Kernel, trits: &[i8], rows: usize, cols: usize, out: &mut [u8]) {
    let tile = kernel.tile_rows();
    let bpr = tile_row_bytes(cols);
    let planar_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let mut block = vec![0i8; tile * cols];
    for b in 0..rows.div_ceil(tile) {
        let r0 = b * tile;
        let n = tile.min(rows - r0);
        block[..n * cols].copy_from_slice(&trits[r0 * cols..(r0 + n) * cols]);
        block[n * cols..].fill(0);
        let planar = pack_matrix(&block, tile, cols);
        let dst = &mut out[b * tile * bpr..(b + 1) * tile * bpr];
        for c in 0..bpr {
            for r in 0..tile {
                dst[c * tile + r] = tile_byte(&planar[r * planar_row_bytes..(r + 1) * planar_row_bytes], cols, c);
            }
        }
    }
}

/// Inverse of [`pack_tiled`]: decodes the first `rows` rows. The caller has validated every byte to be at most 250.
pub fn unpack_tiled(kernel: Kernel, tiled: &[u8], rows: usize, cols: usize, out: &mut [i8]) {
    let tile = kernel.tile_rows();
    let bpr = tile_row_bytes(cols);
    let planar_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let mut planar = vec![0u8; tile * planar_row_bytes];
    let mut tile_row = vec![0u8; bpr];
    for b in 0..rows.div_ceil(tile) {
        let r0 = b * tile;
        let n = tile.min(rows - r0);
        let src = &tiled[b * tile * bpr..(b + 1) * tile * bpr];
        for r in 0..n {
            for c in 0..bpr {
                tile_row[c] = src[c * tile + r];
            }
            untile_row(&tile_row, cols, &mut planar[r * planar_row_bytes..(r + 1) * planar_row_bytes]);
        }
        let trits = unpack_matrix(&planar[..n * planar_row_bytes], n, cols);
        out[r0 * cols..(r0 + n) * cols].copy_from_slice(&trits);
    }
}

/// Raw integer GEMV `W x` for a tiled matrix; returns `rows` values (exact integers stored as f32).
pub fn gemv_raw(
    kernel: Kernel,
    weights: &[u8],
    layout: &TiledLayout,
    acts: &[i8],
    threads: usize,
) -> Vec<f32> {
    let cols = layout.cols as usize;
    let rows = layout.rows as usize;
    let padded = layout.padded_rows as usize;
    let weights = &weights[..layout.total_bytes as usize];
    let mut out = vec![0.0f32; padded];
    match kernel {
        // On AArch64 the activation tables have a NEON-specific layout that the portable block kernel cannot read, and
        // NEON is part of the architecture, so the scalar kernel runs the NEON code there.
        Kernel::Scalar if !cfg!(target_arch = "aarch64") => gemv_scalar(weights, padded, cols, acts, &mut out),
        Kernel::Scalar | Kernel::Simd32 => {
            let wide = quintrit_core::lut::build_activation_lut_wide(acts, cols);
            if threads > 1 {
                quintrit_core::lut::gemv_lut_wide_parallel(weights, &wide, padded, cols, &mut out, threads);
            } else {
                quintrit_core::lut::gemv_lut_wide_1xn(weights, &wide, padded, cols, &mut out);
            }
        }
        Kernel::Avx512 => gemv_avx512(weights, padded, cols, acts, &mut out, threads),
    }
    out.truncate(rows);
    out
}

#[cfg(target_arch = "x86_64")]
fn gemv_avx512(weights: &[u8], padded: usize, cols: usize, acts: &[i8], out: &mut [f32], threads: usize) {
    let wide = quintrit_core::lut::build_activation_lut_wide_avx512(acts, cols);
    if threads > 1 {
        quintrit_core::lut::gemv_lut_64_wide_parallel_avx512(weights, &wide, padded, cols, out, threads);
    } else {
        quintrit_core::lut::gemv_lut_64_wide_1xn_avx512(weights, &wide, padded, cols, out);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn gemv_avx512(_: &[u8], _: usize, _: usize, _: &[i8], _: &mut [f32], _: usize) {
    unreachable!("the AVX-512 kernel is rejected as unsupported before the dispatch");
}

/// Portable kernel (x86-64 table layout): two passes of the scalar block kernel over the `lo` / `hi` activation tables (`x = 16 hi + lo`).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn gemv_scalar(weights: &[u8], padded: usize, cols: usize, acts: &[i8], out: &mut [f32]) {
    let wide = quintrit_core::lut::build_activation_lut_wide(acts, cols);
    let (lo, hi) = wide.split_at(wide.len() / 2);
    let bpr = tile_row_bytes(cols);
    let mut hi_out = [0.0f32; 32];
    for b in 0..padded / 32 {
        // SAFETY: `weights` holds `padded * bpr` bytes, each table `bpr * 65` bytes (rounded up), `out` `padded` floats.
        unsafe {
            let w = weights.as_ptr().add(b * 32 * bpr);
            quintrit_core::simd::scalar::gemv_lut_block32_scalar(w, lo.as_ptr(), bpr, out.as_mut_ptr().add(b * 32));
            quintrit_core::simd::scalar::gemv_lut_block32_scalar(w, hi.as_ptr(), bpr, hi_out.as_mut_ptr());
        }
        for (o, h) in out[b * 32..b * 32 + 32].iter_mut().zip(&hi_out) {
            *o += 16.0 * *h;
        }
    }
}

/// Validates the output pointer and capacity of a GEMV before the expensive part runs.
fn check_out<T>(out: *mut T, capacity: usize, rows: usize) -> Res<()> {
    if rows == 0 || rows > MAX_DIM {
        fail!(QUINTRIT_ERR_INVALID_SHAPE, "rows must be in 1..={MAX_DIM} (got {rows})");
    }
    crate::util::out_ptr(out, "out")?;
    if capacity < rows {
        fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "output holds {capacity} values but rows = {rows}");
    }
    Ok(())
}

/// Validates the arguments shared by the two GEMV entry points and runs the kernel.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_common(
    kernel: i32,
    weights: *const u8,
    weights_len: usize,
    rows: usize,
    cols: usize,
    acts: *const i8,
    acts_len: usize,
    threads: u32,
) -> Res<Vec<f32>> {
    let kernel = resolve(kernel)?;
    kernel.require_supported()?;
    let layout = layout(kernel, rows, cols)?;
    if cols > MAX_GEMV_COLS {
        fail!(
            QUINTRIT_ERR_INVALID_SHAPE,
            "cols = {cols} exceeds {MAX_GEMV_COLS}, the largest count for which the accumulators are exact"
        );
    }
    let threads = resolve_threads(threads)?;
    // SAFETY: validated by in_slice; lengths are the caller's declaration.
    let weights = unsafe { in_slice(weights, weights_len, "weights")? };
    if weights.len() < layout.total_bytes as usize {
        fail!(
            QUINTRIT_ERR_BUFFER_TOO_SMALL,
            "weights hold {} bytes but a {rows} x {cols} matrix needs {} for kernel '{}'",
            weights.len(),
            layout.total_bytes,
            kernel.name()
        );
    }
    let acts = unsafe { in_slice(acts, acts_len, "activations")? };
    if acts.len() < cols {
        fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "activations hold {} values but cols = {cols}", acts.len());
    }
    Ok(gemv_raw(kernel, weights, &layout, &acts[..cols], threads))
}

// ---------------------------------------------------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------------------------------------------------

/// Computes the tiled layout of a `rows x cols` matrix for a kernel.
///
/// # Safety
/// `out_layout` must point to a `quintrit_tiled_layout_t` whose `struct_size` is set to `sizeof` it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_tiled_layout(kernel: i32, rows: usize, cols: usize, out_layout: *mut TiledLayout) -> c_int {
    guard(|| {
        // SAFETY: validated by check_out_struct.
        let out = unsafe { check_out_struct(out_layout, "out_layout")? };
        let l = layout(resolve(kernel)?, rows, cols)?;
        unsafe { out.write(l) };
        Ok(())
    })
}

/// Returns the size in bytes of the tiled buffer for a `rows x cols` matrix (`padded_rows * bytes_per_row`).
///
/// # Safety
/// `out_bytes` must be a valid pointer to a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_tiled_size(kernel: i32, rows: usize, cols: usize, out_bytes: *mut usize) -> c_int {
    guard(|| {
        let l = layout(resolve(kernel)?, rows, cols)?;
        // SAFETY: validated by write_out.
        unsafe { crate::util::write_out(out_bytes, l.total_bytes as usize, "out_bytes") }
    })
}

/// Packs a row-major `rows x cols` matrix of trits (`-1`, `0`, `+1` as `int8_t`) into the tiled layout of `kernel`.
///
/// The buffer must hold at least `quintrit_tiled_size` bytes. Rows are padded to a multiple of the tile height with
/// zero weights; any number of rows and columns is accepted.
///
/// # Safety
/// `trits` must point to `trits_len` readable bytes, `out` to `out_capacity` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_pack_matrix(
    kernel: i32,
    trits: *const i8,
    trits_len: usize,
    rows: usize,
    cols: usize,
    out: *mut u8,
    out_capacity: usize,
) -> c_int {
    guard(|| {
        let kernel = resolve(kernel)?;
        let layout = layout(kernel, rows, cols)?;
        check_disjoint(trits, trits_len, out, out_capacity, "trits / out")?;
        // SAFETY: validated by in_slice / out_slice.
        let trits = unsafe { in_slice(trits, trits_len, "trits")? };
        let n = rows * cols;
        if trits.len() < n {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "trits hold {} values but a {rows} x {cols} matrix has {n}", trits.len());
        }
        let out = unsafe { out_slice(out, out_capacity, "out")? };
        if out.len() < layout.total_bytes as usize {
            fail!(
                QUINTRIT_ERR_BUFFER_TOO_SMALL,
                "output holds {} bytes but the tiled matrix needs {}",
                out.len(),
                layout.total_bytes
            );
        }
        let trits = &trits[..n];
        if let Some(pos) = trits.iter().position(|&t| !(-1..=1).contains(&t)) {
            fail!(
                QUINTRIT_ERR_INVALID_VALUE,
                "trit {} at index {pos} (row {}, column {}) is not -1, 0 or +1",
                trits[pos],
                pos / cols,
                pos % cols
            );
        }
        pack_tiled(kernel, trits, rows, cols, &mut out[..layout.total_bytes as usize]);
        Ok(())
    })
}

/// Unpacks a tiled matrix back into row-major trits (`rows * cols` values); the padding rows are dropped.
///
/// # Safety
/// `tiled` must point to `tiled_len` readable bytes, `out` to `out_capacity` writable `int8_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_unpack_matrix(
    kernel: i32,
    tiled: *const u8,
    tiled_len: usize,
    rows: usize,
    cols: usize,
    out: *mut i8,
    out_capacity: usize,
) -> c_int {
    guard(|| {
        let kernel = resolve(kernel)?;
        let layout = layout(kernel, rows, cols)?;
        check_disjoint(tiled, tiled_len, out, out_capacity, "tiled / out")?;
        // SAFETY: validated by in_slice / out_slice.
        let tiled = unsafe { in_slice(tiled, tiled_len, "tiled")? };
        if tiled.len() < layout.total_bytes as usize {
            fail!(
                QUINTRIT_ERR_BUFFER_TOO_SMALL,
                "tiled buffer holds {} bytes but a {rows} x {cols} matrix needs {}",
                tiled.len(),
                layout.total_bytes
            );
        }
        let n = rows * cols;
        let out = unsafe { out_slice(out, out_capacity, "out")? };
        if out.len() < n {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "output holds {} values but the matrix has {n}", out.len());
        }
        let tiled = &tiled[..layout.total_bytes as usize];
        if let Some(pos) = tiled.iter().position(|&b| b > 250) {
            fail!(QUINTRIT_ERR_INVALID_VALUE, "byte {} at offset {pos} is not a valid packed value (0..=250)", tiled[pos]);
        }
        unpack_tiled(kernel, tiled, rows, cols, &mut out[..n]);
        Ok(())
    })
}

/// Integer GEMV: `out[i] = sum_j W[i][j] * act[j]` for a tiled `rows x cols` ternary matrix and int8 activations.
///
/// The result is exact for the whole int8 range (`-128..=127`) and matches a brute-force `int64` evaluation for every
/// kernel. `cols` must not exceed 131072. `num_threads`: 0 = automatic, 1 = single thread (the scalar kernel ignores it).
///
/// # Safety
/// `weights` must point to `weights_len` readable bytes, `activations` to `activations_len` readable `int8_t`, `out` to
/// `out_capacity` writable `int32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gemv_i32(
    kernel: i32,
    weights: *const u8,
    weights_len: usize,
    rows: usize,
    cols: usize,
    activations: *const i8,
    activations_len: usize,
    out: *mut i32,
    out_capacity: usize,
    num_threads: u32,
) -> c_int {
    guard(|| {
        check_out(out, out_capacity, rows)?;
        // SAFETY: validated inside gemv_common; the input slices are dead when it returns, so `out` may be created
        // afterwards even if the caller made the buffers overlap.
        let result = unsafe { gemv_common(kernel, weights, weights_len, rows, cols, activations, activations_len, num_threads)? };
        let out = unsafe { out_slice(out, out_capacity, "out")? };
        for (dst, v) in out.iter_mut().zip(&result) {
            *dst = *v as i32;
        }
        Ok(())
    })
}

/// Float GEMV: `out[i] = scale * sum_j W[i][j] * act[j]`; the sum is computed exactly in integers first. Pass
/// `scale = weight_scale / activation_scale` (computed once) to dequantize, or 1.0 for the raw accumulators.
///
/// See `quintrit_gemv_i32` for the requirements.
///
/// # Safety
/// As `quintrit_gemv_i32`, with `out` pointing to `out_capacity` writable `float`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gemv_f32(
    kernel: i32,
    weights: *const u8,
    weights_len: usize,
    rows: usize,
    cols: usize,
    activations: *const i8,
    activations_len: usize,
    scale: f32,
    out: *mut f32,
    out_capacity: usize,
    num_threads: u32,
) -> c_int {
    guard(|| {
        check_out(out, out_capacity, rows)?;
        // SAFETY: see quintrit_gemv_i32.
        let result = unsafe { gemv_common(kernel, weights, weights_len, rows, cols, activations, activations_len, num_threads)? };
        let out = unsafe { out_slice(out, out_capacity, "out")? };
        for (dst, v) in out.iter_mut().zip(&result) {
            *dst = *v * scale;
        }
        Ok(())
    })
}
