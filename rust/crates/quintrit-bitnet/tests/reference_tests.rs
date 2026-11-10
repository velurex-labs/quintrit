//! Tests for Microsoft BitNet strict 1:1 reference baselines (2.00 bpw).

use quintrit_bitnet::*;

#[test]
fn test_bitnet_planar_packing_roundtrip() {
    let rows = 4;
    let cols = 128; // Must be multiple of 128 (QK_I2_S)
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 13 + 7) % 3) as i8 - 1)
        .collect();

    let packed = pack_trits_bitnet_planar(&trits, rows, cols);
    let unpacked = unpack_trits_planar(&packed, rows, cols);

    assert_eq!(
        trits, unpacked,
        "BitNet planar packing/unpacking roundtrip must preserve trits exactly"
    );
}

#[test]
fn test_bitnet_mad_exact_match_against_ground_truth() {
    let test_shapes: &[(usize, usize)] = &[
        (4, 128),
        (8, 256),
        (7, 128),   // non-multiple of 4 rows (1x1 remainder test)
        (9, 256),   // 2 full tiles + 1 remainder row
        (4, 2048),  // full 32-block group on ARM
        (11, 2176), // 32 blocks + remainder blocks + remainder rows
    ];

    for &(rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 7 + 1) % 3) as i8 - 1)
            .collect();
        let activations: Vec<i8> = (0..cols)
            .map(|i| (((i * 17 + 3) % 51) as i16 - 25) as i8)
            .collect();

        // 1. Ground truth
        // Note: BitNet MAD kernel computes: \sum (code * activation) where code \in {0, 1, 2}
        // where code = trit + 1. The centered output is (raw - act_sum) * post_scale.
        // The raw output produced by gemv_mad is \sum (code * activation).
        let mut expected_raw = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0i32;
            for c in 0..cols {
                let code = (trits[r * cols + c] + 1) as i32;
                sum += code * (activations[c] as i32);
            }
            expected_raw[r] = sum as f32;
        }

        let packed = pack_trits_bitnet_planar(&trits, rows, cols);

        // Test 1xN if multiple of 4, otherwise general gemv_mad
        if rows % 4 == 0 {
            let mut actual_1t = vec![0.0f32; rows];
            gemv_bitnet_mad_1xn(&packed, &activations, rows, cols, &mut actual_1t);

            for r in 0..rows {
                assert_eq!(
                    actual_1t[r], expected_raw[r],
                    "BitNet MAD 1xn mismatch for shape ({}, {}) at row {}",
                    rows, cols, r
                );
            }
        }

        // Test general gemv_mad (handles arbitrary rows including 1x1 remainder)
        let mut actual_gemv = vec![0.0f32; rows];
        gemv_mad(&packed, &activations, rows, cols, &mut actual_gemv);
        for r in 0..rows {
            assert_eq!(
                actual_gemv[r], expected_raw[r],
                "BitNet MAD gemv mismatch for shape ({}, {}) at row {}",
                rows, cols, r
            );
        }

        // Test multi-threaded gemv_bitnet_mad_parallel
        let mut actual_mt = vec![0.0f32; rows];
        gemv_bitnet_mad_parallel(&packed, &activations, rows, cols, &mut actual_mt, 2);

        for r in 0..rows {
            assert_eq!(
                actual_mt[r], expected_raw[r],
                "BitNet MAD parallel mismatch for shape ({}, {}) at row {}",
                rows, cols, r
            );
        }

        // On AArch64, explicitly test individual NEON kernels (both dotprod and baseline)
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if rows % 4 == 0 {
                // Baseline NEON 1xN
                let mut actual_neon_1xn = vec![0.0f32; rows];
                gemv_mad_1xn_neon(&packed, &activations, rows, cols, &mut actual_neon_1xn);
                assert_eq!(
                    actual_neon_1xn, expected_raw,
                    "NEON 1xN mismatch for shape ({}, {})",
                    rows, cols
                );

                // DotProd NEON 1xN
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    let mut actual_dp_1xn = vec![0.0f32; rows];
                    gemv_mad_1xn_neon_dotprod(&packed, &activations, rows, cols, &mut actual_dp_1xn);
                    assert_eq!(
                        actual_dp_1xn, expected_raw,
                        "NEON DotProd 1xN mismatch for shape ({}, {})",
                        rows, cols
                    );
                }
            }

            // Baseline NEON 1x1
            let mut actual_neon_1x1 = vec![0.0f32; rows];
            gemv_mad_1x1_neon(&packed, &activations, rows, cols, &mut actual_neon_1x1);
            assert_eq!(
                actual_neon_1x1, expected_raw,
                "NEON 1x1 mismatch for shape ({}, {})",
                rows, cols
            );

            // DotProd NEON 1x1
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let mut actual_dp_1x1 = vec![0.0f32; rows];
                gemv_mad_1x1_neon_dotprod(&packed, &activations, rows, cols, &mut actual_dp_1x1);
                assert_eq!(
                    actual_dp_1x1, expected_raw,
                    "NEON DotProd 1x1 mismatch for shape ({}, {})",
                    rows, cols
                );
            }
        }
    }
}

#[test]
fn test_bitnet_lut_exact_match_against_ground_truth() {
    let rows = 32;
    let cols = 128;
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 11 + 5) % 3) as i8 - 1)
        .collect();
    let activations: Vec<i8> = (0..cols)
        .map(|i| (((i * 29 + 13) % 51) as i16 - 25) as i8)
        .collect();

    // Ground truth: \sum (trit * activation)
    let mut expected = vec![0i64; rows];
    for r in 0..rows {
        let mut sum = 0i64;
        for c in 0..cols {
            sum += (trits[r * cols + c] as i64) * (activations[c] as i64);
        }
        expected[r] = sum;
    }

    let mut trit4_linear = Vec::new();
    for r in 0..rows {
        trit4_linear.extend_from_slice(&pack_trits_4(&trits[r * cols..(r + 1) * cols]));
    }

    let tiled = transform_weights_bitnet_lut(&trit4_linear, rows, cols);
    let qlut = build_bitnet_lut(&activations);

    let mut actual_1t = vec![0i64; rows];
    gemv_bitnet_lut(&tiled, &qlut, rows, cols, &mut actual_1t);

    for r in 0..rows {
        assert_eq!(
            actual_1t[r], expected[r],
            "BitNet LUT 1t mismatch at row {}",
            r
        );
    }

    let mut actual_mt = vec![0i64; rows];
    gemv_bitnet_lut_parallel(&tiled, &qlut, rows, cols, &mut actual_mt, 4);

    for r in 0..rows {
        assert_eq!(
            actual_mt[r], expected[r],
            "BitNet LUT mt mismatch at row {}",
            r
        );
    }
}
