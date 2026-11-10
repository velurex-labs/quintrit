//! Tests for 5-trit packing, unpacking, and matrix transforms.

use quintrit_core::pack::*;
#[allow(unused_imports)]
use quintrit_core::lut as _;

#[test]
fn test_all_243_states_roundtrip() {
    for d4 in 0..3usize {
        for d3 in 0..3usize {
            for d2 in 0..3usize {
                for d1 in 0..3usize {
                    for d0 in 0..3usize {
                        let original = [
                            d0 as i8 - 1,
                            d1 as i8 - 1,
                            d2 as i8 - 1,
                            d3 as i8 - 1,
                            d4 as i8 - 1,
                        ];

                        let byte = pack_5trits(original);
                        assert!(
                            byte <= 250,
                            "Byte value {} exceeded max valid range 250",
                            byte
                        );

                        let decoded = BYTE_TO_TRITS[byte as usize];
                        assert_eq!(
                            decoded, original,
                            "Decoded mismatch for byte {}: expected {:?}, got {:?}",
                            byte, original, decoded
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_unused_states_decode_to_zero() {
    for byte in 251..=255u8 {
        let decoded = BYTE_TO_TRITS[byte as usize];
        assert_eq!(
            decoded,
            [0, 0, 0, 0, 0],
            "Byte {} outside valid domain should decode to all zeros",
            byte
        );
    }
}

#[test]
fn test_matrix_roundtrip_shapes() {
    let test_shapes = [
        (32, 160),
        (32, 320),
        (64, 640),
        (128, 1600),
        (1, 160),
    ];

    for (rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 7 + 1) % 3) as i8 - 1)
            .collect();

        let packed = pack_matrix(&trits, rows, cols);
        let unpacked = unpack_matrix(&packed, rows, cols);

        assert_eq!(
            unpacked, trits,
            "Matrix roundtrip failed for shape ({}, {})",
            rows, cols
        );
    }
}

#[test]
fn test_split_activation_planes() {
    let cols = 320;
    let activations: Vec<i8> = (0..cols)
        .map(|i| ((i * 13 + 5) % 127) as i8)
        .collect();

    let planes = split_activation_planes(&activations, cols);

    for (p, plane) in planes.iter().enumerate() {
        assert_eq!(plane.len(), 64); // 2 blocks * 32 bytes
        for b in 0..2 {
            for j in 0..32 {
                let col_idx = b * QK_5 + p * BLOCK_BYTES_5 + j;
                assert_eq!(plane[b * 32 + j], activations[col_idx]);
            }
        }
    }
}

/// The tile transforms (32- and 64-row) must be lossless for every column count (the tail of the last block is packed in the tiles).
#[test]
fn tile_transforms_roundtrip_for_every_tail_length() {
    use quintrit_core::lut::{transform_weights_lut, untransform_weights_lut};
    let mut seed = 0x1357_9BDF_2468_ACE1u64;
    for cols in [1usize, 5, 31, 32, 33, 159, 160, 161, 192, 250, 320, 321, 479, 800, 3072, 3079] {
        let rows = 64;
        let trits: Vec<i8> = (0..rows * cols)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 3) as i8 - 1
            })
            .collect();
        let packed = pack_matrix(&trits, rows, cols);
        let tiled = transform_weights_lut(&packed, rows, cols);
        assert_eq!(tiled.len(), rows * quintrit_core::pack::tile_row_bytes(cols), "tile size, cols={cols}");
        assert_eq!(untransform_weights_lut(&tiled, rows, cols), packed, "32-row round trip, cols={cols}");
        #[cfg(target_arch = "x86_64")]
        {
            let tiled64 = quintrit_core::simd::avx512::transform_weights_lut_64(&packed, rows, cols);
            assert_eq!(
                quintrit_core::simd::avx512::untransform_weights_lut_64(&tiled64, rows, cols),
                packed,
                "64-row round trip, cols={cols}"
            );
        }
    }
}
