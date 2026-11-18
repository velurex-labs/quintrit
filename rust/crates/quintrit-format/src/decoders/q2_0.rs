//! llama.cpp `Q2_0` (GGML type 42): blocks of 64 weights, an f16 scale `d` followed by 16 bytes of 2-bit codes, four
//! codes per byte with the lowest bits first; weight `= (code - 1) * d`. A ternary model uses only the codes 0, 1
//! and 2. Some files (the first Ternary Bonsai generation) store blocks of 128 weights (f16 scale + 32 bytes) under the
//! same type number; the code layout is the same.
//!
//! The per-block scales are not uniform, so `Q2_0` tensors are not served by a [`super::TernaryDecoder`] (which yields
//! one scale per tensor). [`decode_q2_0`] returns the trits together with one scale per 64 weights.

use super::half_to_f32;

/// Weights per block in llama.cpp's definition (`QK2_0`).
pub const QK2_0: usize = 64;

/// Decodes a `Q2_0` tensor of `rows x cols` weights stored with `block_size` (64 or 128) weights per block.
///
/// Returns the trits (`rows * cols`, row-major) and the scales, one per 64 weights (`rows * cols / 64`, row-major; a
/// 128-weight block contributes its scale twice). Fails if the shape does not fit the data or a weight uses the code 3
/// (`+2`), which is not ternary.
pub fn decode_q2_0(data: &[u8], rows: usize, cols: usize, block_size: usize) -> Result<(Vec<i8>, Vec<f32>), String> {
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
    let mut trits = vec![0i8; n];
    let mut scales = Vec::with_capacity(n / QK2_0);
    for (b, block) in data.chunks_exact(block_bytes).enumerate() {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for _ in 0..block_size / QK2_0 {
            scales.push(d);
        }
        let dst = &mut trits[b * block_size..(b + 1) * block_size];
        for (j, &byte) in block[2..].iter().enumerate() {
            for k in 0..4 {
                let code = (byte >> (2 * k)) & 3;
                if code == 3 {
                    return Err(format!("Q2_0 weight {} uses code 3 (+2), which is not ternary", b * block_size + 4 * j + k));
                }
                dst[4 * j + k] = code as i8 - 1;
            }
        }
    }
    Ok((trits, scales))
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
