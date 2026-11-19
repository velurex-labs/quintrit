//! llama.cpp `Q2_0` (GGML type 42): blocks of 64 weights, an f16 scale `d` followed by 16 bytes of 2-bit codes, four
//! codes per byte with the lowest bits first; weight `= (code - 1) * d`. A ternary model uses only the codes 0, 1
//! and 2. Some files (the first Ternary Bonsai generation) store blocks of 128 weights (f16 scale + 32 bytes) under the
//! same type number; the code layout is the same.
//!
//! The per-block scales are not uniform, so `Q2_0` tensors are not served by a [`super::TernaryDecoder`] (which yields
//! one scale per tensor). [`decode_q2_0`] returns the trits together with one scale per 64 weights.

use super::half_to_f32;
use quintrit_core::pack::PARALLEL_MIN_WEIGHTS;
use rayon::prelude::*;

/// Weights per block in llama.cpp's definition (`QK2_0`).
pub const QK2_0: usize = 64;

/// Checks the shape of a `Q2_0` tensor against its payload and returns the number of weights.
fn check_q2_0(data: &[u8], rows: usize, cols: usize, block_size: usize) -> Result<usize, String> {
    if block_size != 64 && block_size != 128 {
        return Err(format!("Q2_0 block size {block_size} is not supported (64 or 128)"));
    }
    if cols == 0 || !cols.is_multiple_of(block_size) {
        return Err(format!("a Q2_0 row of {cols} weights is not a multiple of the block size {block_size}"));
    }
    let n = rows.checked_mul(cols).ok_or("Q2_0 tensor too large")?;
    let block_bytes = 2 + block_size / 4;
    if data.len() != n / block_size * block_bytes {
        return Err(format!("Q2_0 data of {} bytes does not hold {rows} x {cols} weights", data.len()));
    }
    Ok(n)
}

/// Decodes one block (`block_size` weights) into `dst` and its scale(s) into `block_scales`. `first_weight` is the index
/// of the block's first weight in the tensor; an error names the first offending weight of the block.
fn decode_block(first_weight: usize, block: &[u8], dst: &mut [i8], block_scales: &mut [f32]) -> Result<(), String> {
    block_scales.fill(half_to_f32(u16::from_le_bytes([block[0], block[1]])));
    // Fixed trip counts and no branches in the loop, so that the compiler vectorizes it for the target; the code 3 is
    // only noted here and located afterwards.
    let mut invalid = 0u8;
    for (&byte, out) in block[2..].iter().zip(dst.chunks_exact_mut(4)) {
        for k in 0..4 {
            let code = (byte >> (2 * k)) & 3;
            invalid |= (code == 3) as u8;
            out[k] = code as i8 - 1;
        }
    }
    if invalid != 0 {
        let bad = block[2..]
            .iter()
            .enumerate()
            .flat_map(|(j, &byte)| (0..4).map(move |k| (4 * j + k, (byte >> (2 * k)) & 3)))
            .find(|&(_, code)| code == 3);
        let (offset, _) = bad.expect("a code 3 was seen");
        return Err(format!("Q2_0 weight {} uses code 3 (+2), which is not ternary", first_weight + offset));
    }
    Ok(())
}

/// Decodes a `Q2_0` tensor of `rows x cols` weights stored with `block_size` (64 or 128) weights per block.
///
/// Returns the trits (`rows * cols`, row-major) and the scales, one per 64 weights (`rows * cols / 64`, row-major; a
/// 128-weight block contributes its scale twice). Fails if the shape does not fit the data or a weight uses the code 3
/// (`+2`), which is not ternary. A tensor of at least [`PARALLEL_MIN_WEIGHTS`] weights is decoded by all threads of the
/// global rayon pool; the result and the error (the one of the first offending weight) do not depend on the scheduling.
pub fn decode_q2_0(data: &[u8], rows: usize, cols: usize, block_size: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
    let n = check_q2_0(data, rows, cols, block_size)?;
    let block_bytes = 2 + block_size / 4;
    let mut trits = vec![0i8; n];
    let mut scales = vec![0f32; n / QK2_0];
    let per_block = block_size / QK2_0;
    if n >= PARALLEL_MIN_WEIGHTS {
        // `find_first` returns the error of the lowest block.
        let failed = trits
            .par_chunks_exact_mut(block_size)
            .zip(scales.par_chunks_exact_mut(per_block))
            .zip(data.par_chunks_exact(block_bytes))
            .enumerate()
            .map(|(b, ((dst, block_scales), block))| decode_block(b * block_size, block, dst, block_scales))
            .find_first(|r| r.is_err());
        if let Some(Err(e)) = failed {
            return Err(e);
        }
    } else {
        for (b, ((dst, block_scales), block)) in
            trits.chunks_exact_mut(block_size).zip(scales.chunks_exact_mut(per_block)).zip(data.chunks_exact(block_bytes)).enumerate()
        {
            decode_block(b * block_size, block, dst, block_scales)?;
        }
    }
    Ok((trits, scales))
}

/// Rows decoded and packed together by [`decode_q2_0_packed`]: the trits of such a group (`8 * cols` bytes) stay in the
/// cache of the thread that converts it.
const PACKED_GROUP_ROWS: usize = 8;

/// Decodes a `Q2_0` tensor like [`decode_q2_0`], but returns the weights already packed into the 5-planar layout of
/// [`quintrit_core::pack::pack_matrix`] (`rows * cols.div_ceil(160) * 32` bytes) instead of one byte per trit.
///
/// The conversion runs in groups of rows on all threads of the global rayon pool, each through a small buffer that is
/// reused, so the large intermediate array of trits is never materialized. The packed bytes and the scales are exactly
/// those of `pack_matrix(decode_q2_0(..).0)` and `decode_q2_0(..).1`.
pub fn decode_q2_0_packed(data: &[u8], rows: usize, cols: usize, block_size: usize) -> Result<(Vec<u8>, Vec<f32>), String> {
    use quintrit_core::pack::{pack_matrix_into, BLOCK_BYTES_5, QK_5};
    let n = check_q2_0(data, rows, cols, block_size)?;
    let block_bytes = 2 + block_size / 4;
    let row_bytes = cols / block_size * block_bytes;
    let packed_row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let mut packed = vec![0u8; rows * packed_row_bytes];
    let mut scales = vec![0f32; n / QK2_0];
    if rows == 0 {
        return Ok((packed, scales));
    }
    let per_block = block_size / QK2_0;
    let scales_per_row = cols / QK2_0;

    let convert = |trits: &mut Vec<i8>, g: usize, data: &[u8], packed: &mut [u8], scales: &mut [f32]| -> Result<(), String> {
        let group_rows = data.len() / row_bytes;
        trits.resize(group_rows * cols, 0);
        let first_weight = g * PACKED_GROUP_ROWS * cols;
        for (b, ((dst, block_scales), block)) in trits
            .chunks_exact_mut(block_size)
            .zip(scales.chunks_exact_mut(per_block))
            .zip(data.chunks_exact(block_bytes))
            .enumerate()
        {
            decode_block(first_weight + b * block_size, block, dst, block_scales)?;
        }
        pack_matrix_into(trits, group_rows, cols, packed);
        Ok(())
    };

    if n >= PARALLEL_MIN_WEIGHTS {
        // `find_first` returns the error of the lowest group, so the message does not depend on the scheduling.
        let failed = data
            .par_chunks(PACKED_GROUP_ROWS * row_bytes)
            .zip(packed.par_chunks_mut(PACKED_GROUP_ROWS * packed_row_bytes))
            .zip(scales.par_chunks_mut(PACKED_GROUP_ROWS * scales_per_row))
            .enumerate()
            .map_init(Vec::new, |trits, (g, ((data, packed), scales))| convert(trits, g, data, packed, scales))
            .find_first(|r| r.is_err());
        if let Some(Err(e)) = failed {
            return Err(e);
        }
    } else {
        let mut trits = Vec::new();
        let groups = data
            .chunks(PACKED_GROUP_ROWS * row_bytes)
            .zip(packed.chunks_mut(PACKED_GROUP_ROWS * packed_row_bytes))
            .zip(scales.chunks_mut(PACKED_GROUP_ROWS * scales_per_row));
        for (g, ((data, packed), scales)) in groups.enumerate() {
            convert(&mut trits, g, data, packed, scales)?;
        }
    }
    Ok((packed, scales))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_both_block_sizes_identically() {
        // Two 64-blocks with scales 1.0 (0x3C00) and 0.5 (0x3800), then the same weights as one 128-block with 1.0.
        let codes: Vec<u8> = (0..128).map(|i| (i % 3) as u8).collect();
        let pack = |c: &[u8]| -> Vec<u8> { c.chunks(4).map(|q| q[0] | q[1] << 2 | q[2] << 4 | q[3] << 6).collect() };
        let mut b64 = vec![0x00, 0x3C];
        b64.extend(pack(&codes[..64]));
        b64.extend([0x00, 0x38]);
        b64.extend(pack(&codes[64..]));
        let (t64, s64) = decode_q2_0(&b64, 1, 128, 64).unwrap();
        assert_eq!(s64, vec![1.0, 0.5]);
        let mut b128 = vec![0x00, 0x3C];
        b128.extend(pack(&codes));
        let (t128, s128) = decode_q2_0(&b128, 1, 128, 128).unwrap();
        assert_eq!(s128, vec![1.0, 1.0]);
        assert_eq!(t64, t128);
        assert_eq!(&t64[..6], &[-1, 0, 1, -1, 0, 1]);
    }

    #[test]
    fn rejects_code_three_and_bad_shapes() {
        let mut b = vec![0x00, 0x3C];
        b.extend([0x03u8; 16]);
        assert!(decode_q2_0(&b, 1, 64, 64).is_err());
        assert!(decode_q2_0(&b, 1, 96, 64).is_err());
        assert!(decode_q2_0(&b, 2, 64, 64).is_err());
    }
}
