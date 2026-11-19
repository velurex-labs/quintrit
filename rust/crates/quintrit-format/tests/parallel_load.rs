//! The load-time conversions (unpacking, packing, tiling) spread large matrices over the rayon pool. These tests
//! check that the parallel path of every conversion gives exactly the bytes of the sequential path (small matrices
//! and single rows run sequentially) and of an independent element-by-element reference.

use quintrit_core::lut::{transform_weights_lut, untransform_weights_lut};
use quintrit_core::pack::{pack_5trits, pack_matrix, tile_row_bytes, BLOCK_BYTES_5, PARALLEL_MIN_WEIGHTS, QK_5};
use quintrit_format::decoders::i2_s::{pack_gguf_i2_s, unpack_gguf_i2_s};
use quintrit_format::{decode_q2_0, get_ternary_decoder, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0};

/// Deterministic pseudo-random trits.
fn trits(n: usize, seed: u64) -> Vec<i8> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 3) as i8 - 1
        })
        .collect()
}

/// A shape whose weight count is above the threshold of the parallel path while one 32-row block stays below it.
fn big_shape(cols: usize) -> (usize, usize) {
    let rows = (PARALLEL_MIN_WEIGHTS / cols / 64 + 2) * 64;
    assert!(rows * cols >= PARALLEL_MIN_WEIGHTS);
    (rows, cols)
}

/// Element-by-element reference of the 5-planar layout, written without any helper of the crate except `pack_5trits`.
fn reference_pack(t: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
    let mut out = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..cols.div_ceil(QK_5) {
            for j in 0..BLOCK_BYTES_5 {
                let at = |p: usize| {
                    let c = b * QK_5 + p * BLOCK_BYTES_5 + j;
                    if c < cols { t[r * cols + c] } else { 0 }
                };
                out[r * row_bytes + b * BLOCK_BYTES_5 + j] = pack_5trits([at(0), at(1), at(2), at(3), at(4)]);
            }
        }
    }
    out
}

#[test]
fn pack_matrix_parallel_matches_reference_and_rows() {
    for cols in [3072usize, 3100, 8192] {
        let (rows, cols) = big_shape(cols);
        let t = trits(rows * cols, cols as u64);
        let packed = pack_matrix(&t, rows, cols);
        assert_eq!(packed, reference_pack(&t, rows, cols), "cols = {cols}");
        let row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
        for r in [0, 1, rows / 2, rows - 1] {
            assert_eq!(&packed[r * row_bytes..(r + 1) * row_bytes], pack_matrix(&t[r * cols..(r + 1) * cols], 1, cols).as_slice());
        }
    }
}

#[test]
fn tiling_parallel_matches_blockwise_and_round_trips() {
    for cols in [3072usize, 3100] {
        let (rows, cols) = big_shape(cols);
        let t = trits(rows * cols, 7 + cols as u64);
        let packed = pack_matrix(&t, rows, cols);
        let row_bytes = cols.div_ceil(QK_5) * BLOCK_BYTES_5;
        let tile_bytes = 32 * tile_row_bytes(cols);

        let tiled = transform_weights_lut(&packed, rows, cols);
        assert_eq!(tiled.len(), rows / 32 * tile_bytes);
        for b in 0..rows / 32 {
            assert!(32 * cols < PARALLEL_MIN_WEIGHTS, "the single block must take the sequential path");
            let one = transform_weights_lut(&packed[b * 32 * row_bytes..(b + 1) * 32 * row_bytes], 32, cols);
            assert_eq!(&tiled[b * tile_bytes..(b + 1) * tile_bytes], one.as_slice(), "block {b}");
        }
        assert_eq!(untransform_weights_lut(&tiled, rows, cols), packed);

        #[cfg(target_arch = "x86_64")]
        {
            use quintrit_core::lut::{transform_weights_lut_64, untransform_weights_lut_64};
            let tiled64 = transform_weights_lut_64(&packed, rows, cols);
            let tile_bytes64 = 64 * tile_row_bytes(cols);
            for b in 0..rows / 64 {
                let one = transform_weights_lut_64(&packed[b * 64 * row_bytes..(b + 1) * 64 * row_bytes], 64, cols);
                assert_eq!(&tiled64[b * tile_bytes64..(b + 1) * tile_bytes64], one.as_slice(), "64-row block {b}");
            }
            assert_eq!(untransform_weights_lut_64(&tiled64, rows, cols), packed);
        }
    }
}

#[test]
fn i2_s_unpack_parallel_matches_rows() {
    let (rows, cols) = big_shape(3072);
    let t = trits(rows * cols, 11);
    let packed = pack_gguf_i2_s(&t, rows, cols);
    assert_eq!(unpack_gguf_i2_s(&packed, rows, cols), t);
    let decoder = get_ternary_decoder(quintrit_format::GGML_TYPE_I2_S).unwrap();
    // The fused transcoder (one block per thread) must give the bytes of the three separate steps.
    let tiled = decoder.transcode_to_velurex_tiled(&packed, rows, cols).unwrap();
    assert_eq!(tiled, transform_weights_lut(&pack_matrix(&t, rows, cols), rows, cols));
}

/// ggml's `Q2_0` encoding of `t` (codes 0, 1, 2 for -1, 0, +1) in blocks of `block` weights with the given f16 scale.
fn encode_q2_0(t: &[i8], block: usize, scale_bits: impl Fn(usize) -> u16) -> Vec<u8> {
    let mut out = Vec::new();
    for (b, w) in t.chunks(block).enumerate() {
        out.extend_from_slice(&scale_bits(b).to_le_bytes());
        for q in w.chunks(4) {
            out.push(q.iter().enumerate().fold(0u8, |acc, (k, &v)| acc | (((v + 1) as u8) << (2 * k))));
        }
    }
    out
}

#[test]
fn q2_0_decode_parallel_matches_blocks_and_reports_lowest_error() {
    for block in [64usize, 128] {
        let (rows, cols) = big_shape(4096);
        let t = trits(rows * cols, 13);
        let scale = |b: usize| 0x3C00 + (b % 7) as u16;
        let data = encode_q2_0(&t, block, scale);
        let (decoded, scales) = decode_q2_0(&data, rows, cols, block).unwrap();
        assert_eq!(decoded, t);
        let expected_scales: Vec<f32> = (0..rows * cols / 64)
            .map(|i| {
                let bits = scale(i / (block / 64));
                // 0x3C00 + n is 1.0 + n * 2^-10 in half precision.
                1.0 + (bits - 0x3C00) as f32 / 1024.0
            })
            .collect();
        assert_eq!(scales, expected_scales);

        // Two corrupt weights in different blocks: the error names the first one, whatever the scheduling.
        let block_bytes = 2 + block / 4;
        let mut bad = data.clone();
        bad[(rows * cols / block - 3) * block_bytes + 2 + 5] |= 0b11 << 4;
        bad[(rows * cols / block / 2) * block_bytes + 2 + 3] |= 0b11 << 2;
        let first = (rows * cols / block / 2) * block + 4 * 3 + 1;
        for _ in 0..4 {
            let err = decode_q2_0(&bad, rows, cols, block).unwrap_err();
            assert!(err.contains(&format!("weight {first} ")), "{err}");
        }
    }
}

#[test]
fn tq_unpack_parallel_matches_single_blocks() {
    for (ty, block_weights, block_bytes) in [(GGML_TYPE_TQ1_0, 256usize, 54usize), (GGML_TYPE_TQ2_0, 256, 66)] {
        let decoder = get_ternary_decoder(ty).unwrap();
        let (rows, cols) = big_shape(4096);
        let n_blocks = rows * cols / block_weights;
        let data: Vec<u8> = (0..n_blocks * block_bytes).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        let all = decoder.unpack_trits(&data, rows, cols).unwrap();
        assert_eq!(all.len(), rows * cols);
        for b in [0, 1, n_blocks / 2, n_blocks - 1] {
            // One block on its own is a 1 x 256 matrix and takes the sequential path.
            let one = decoder.unpack_trits(&data[b * block_bytes..(b + 1) * block_bytes], 1, block_weights).unwrap();
            assert_eq!(&all[b * block_weights..(b + 1) * block_weights], one.as_slice(), "type {ty} block {b}");
        }
    }
}

/// Payload of `rows x cols` weights in the given format, filled with arbitrary bytes (every byte is a valid code
/// except for the code 3 of `I2_S`, which decodes to zero, so the result is always well defined).
fn payload(ty: u32, rows: usize, cols: usize) -> Vec<u8> {
    let len = match ty {
        quintrit_format::GGML_TYPE_I2_S => rows * cols / 4,
        GGML_TYPE_TQ1_0 => rows * cols / 256 * 54,
        GGML_TYPE_TQ2_0 => rows * cols / 256 * 66,
        _ => unreachable!(),
    };
    (0..len).map(|i| ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 29) as u8).collect()
}

#[test]
fn fused_transcoder_matches_the_separate_steps() {
    use quintrit_format::transcode_to_tiled;
    for ty in [quintrit_format::GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let decoder = get_ternary_decoder(ty).unwrap();
        // 3072 columns end in a partial 160-trit block (32 trits); 4096 and 5120 too; 3200 is a whole number of blocks.
        for cols in [3072usize, 3200, 4096, 5120] {
            for rows in [64usize, 192, big_shape(cols).0] {
                let data = payload(ty, rows, cols);
                let trits = decoder.unpack_trits(&data, rows, cols).unwrap();
                let planar = pack_matrix(&trits, rows, cols);
                let expect32 = transform_weights_lut(&planar, rows, cols);
                assert_eq!(transcode_to_tiled(decoder, &data, rows, cols, 32).unwrap(), expect32, "type {ty} {rows}x{cols} tile 32");
                assert_eq!(decoder.transcode_to_velurex_tiled(&data, rows, cols).unwrap(), expect32, "type {ty} {rows}x{cols}");
                #[cfg(target_arch = "x86_64")]
                assert_eq!(
                    transcode_to_tiled(decoder, &data, rows, cols, 64).unwrap(),
                    quintrit_core::lut::transform_weights_lut_64(&planar, rows, cols),
                    "type {ty} {rows}x{cols} tile 64"
                );
            }
        }
    }
}

#[test]
fn fused_transcoder_rejects_bad_shapes() {
    use quintrit_format::transcode_to_tiled;
    let decoder = get_ternary_decoder(GGML_TYPE_TQ2_0).unwrap();
    let data = payload(GGML_TYPE_TQ2_0, 64, 256);
    assert!(transcode_to_tiled(decoder, &data, 48, 256, 32).is_err(), "48 rows are no whole number of 32-row tiles");
    assert!(transcode_to_tiled(decoder, &data, 64, 256, 16).is_err());
    assert!(transcode_to_tiled(decoder, &data[..data.len() - 1], 64, 256, 32).is_err(), "short payload");
}

#[test]
fn q2_0_packed_matches_decode_then_pack() {
    use quintrit_format::decode_q2_0_packed;
    for block in [64usize, 128] {
        // 4096 columns are a whole number of 160-trit blocks plus 96 trits; 3200 are exactly 20 blocks; 128 and 64 are tiny.
        for (rows, cols) in [(1usize, 128usize), (3, 640), (19, 3200), big_shape(4096)] {
            let t = trits(rows * cols, 17 + (rows * cols) as u64);
            let data = encode_q2_0(&t, block, |b| 0x3C00 + (b % 5) as u16);
            let (want_trits, want_scales) = decode_q2_0(&data, rows, cols, block).unwrap();
            let (packed, scales) = decode_q2_0_packed(&data, rows, cols, block).unwrap();
            assert_eq!(packed, pack_matrix(&want_trits, rows, cols), "{rows}x{cols} block {block}");
            assert_eq!(scales, want_scales);

            // The same error as the unfused decoder, whatever the scheduling.
            let block_bytes = 2 + block / 4;
            let mut bad = data.clone();
            let n_blocks = rows * cols / block;
            bad[(n_blocks - 1) * block_bytes + 2 + 1] |= 0b11 << 6;
            bad[(n_blocks / 2) * block_bytes + 2] |= 0b11;
            let want = decode_q2_0(&bad, rows, cols, block).unwrap_err();
            for _ in 0..3 {
                assert_eq!(decode_q2_0_packed(&bad, rows, cols, block).unwrap_err(), want);
            }
        }
    }
}
