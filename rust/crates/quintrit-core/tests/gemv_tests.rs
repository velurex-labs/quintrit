//! Tests for 32-row tiled GEMV execution and weight transformations.

use quintrit_core::lut::*;
use quintrit_core::pack::*;
#[cfg(target_arch = "x86_64")]
use quintrit_core::simd::scalar::gemv_lut_block32_scalar;

#[cfg(target_arch = "x86_64")]
use quintrit_core::simd::avx2::gemv_lut_block32_avx2;

#[test]
fn test_weight_transform_untransform_roundtrip() {
    let test_shapes = [(32, 160), (64, 320), (128, 640)];

    for (rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 11 + 3) % 3) as i8 - 1)
            .collect();

        let packed = pack_matrix(&trits, rows, cols);
        let transformed = transform_weights_lut(&packed, rows, cols);
        let restored = untransform_weights_lut(&transformed, rows, cols);

        assert_eq!(
            restored, packed,
            "Tiled transform roundtrip failed for shape ({}, {})",
            rows, cols
        );
    }
}

#[test]
fn test_gemv_exact_match_against_ground_truth_dot_product() {
    let test_shapes = [(32, 160), (32, 320), (64, 640), (128, 960)];

    for (rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 17 + 5) % 3) as i8 - 1)
            .collect();

        let activations: Vec<i8> = (0..cols)
            .map(|i| (((i * 23 + 7) % 51) as i16 - 25) as i8)
            .collect();

        // 1. Compute naive reference dot product
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut dot: i32 = 0;
            for c in 0..cols {
                dot += trits[r * cols + c] as i32 * activations[c] as i32;
            }
            expected[r] = dot as f32;
        }

        // 2. Compute via LUT GEMV
        let packed = pack_matrix(&trits, rows, cols);
        let transformed = transform_weights_lut(&packed, rows, cols);
        let qlut = build_activation_lut(&activations, cols);

        let mut actual_1xn = vec![0.0f32; rows];
        gemv_lut_1xn(&transformed, &qlut, rows, cols, &mut actual_1xn);

        for r in 0..rows {
            assert_eq!(
                actual_1xn[r], expected[r],
                "GEMV 1xn mismatch at row {} for shape ({}, {}): got {}, expected {}",
                r, rows, cols, actual_1xn[r], expected[r]
            );
        }

        // 3. Compute via Parallel GEMV (4 threads)
        let mut actual_parallel = vec![0.0f32; rows];
        gemv_lut_parallel(&transformed, &qlut, rows, cols, &mut actual_parallel, 4);

        for r in 0..rows {
            assert_eq!(
                actual_parallel[r], expected[r],
                "GEMV parallel mismatch at row {} for shape ({}, {}): got {}, expected {}",
                r, rows, cols, actual_parallel[r], expected[r]
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_scalar_vs_avx2_block32_exact_equivalence() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let rows = 32;
    let cols = 640;
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 31 + 1) % 3) as i8 - 1)
        .collect();
    let activations: Vec<i8> = (0..cols)
        .map(|i| (((i * 19 + 11) % 51) as i16 - 25) as i8)
        .collect();

    let packed = pack_matrix(&trits, rows, cols);
    let transformed = transform_weights_lut(&packed, rows, cols);
    let qlut = build_activation_lut(&activations, cols);
    let bytes_per_row = quintrit_core::pack::tile_row_bytes(cols);

    let mut out_scalar = vec![0.0f32; 32];
    let mut out_avx2 = vec![0.0f32; 32];

    unsafe {
        gemv_lut_block32_scalar(
            transformed.as_ptr(),
            qlut.as_ptr(),
            bytes_per_row,
            out_scalar.as_mut_ptr(),
        );
    }

    unsafe {
        gemv_lut_block32_avx2(
            transformed.as_ptr(),
            qlut.as_ptr(),
            bytes_per_row,
            out_avx2.as_mut_ptr(),
        );
    }

    assert_eq!(
        out_scalar, out_avx2,
        "Scalar reference kernel output must match AVX2 kernel output bit-for-bit"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_dotprod_parity() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }

    use quintrit_core::simd::neon::{
        gemv_lut_block32_neon, gemv_lut_block32_neon_dotprod, gemv_lut_block64_neon,
        gemv_lut_block64_neon_dotprod,
    };

    let rows = 64;
    let cols = 640;
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 31 + 1) % 3) as i8 - 1)
        .collect();
    let activations: Vec<i8> = (0..cols)
        .map(|i| (((i * 19 + 11) % 51) as i16 - 25) as i8)
        .collect();

    let packed = pack_matrix(&trits, rows, cols);
    let transformed = transform_weights_lut(&packed, rows, cols);
    let qlut = build_activation_lut(&activations, cols);
    let bytes_per_row = quintrit_core::pack::tile_row_bytes(cols);

    // Test block32
    let mut out_neon_32 = vec![0.0f32; 32];
    let mut out_dotprod_32 = vec![0.0f32; 32];
    unsafe {
        gemv_lut_block32_neon(
            transformed.as_ptr(),
            qlut.as_ptr(),
            bytes_per_row,
            out_neon_32.as_mut_ptr(),
        );
        gemv_lut_block32_neon_dotprod(
            transformed.as_ptr(),
            qlut.as_ptr(),
            bytes_per_row,
            out_dotprod_32.as_mut_ptr(),
        );
    }
    assert_eq!(out_neon_32, out_dotprod_32, "block32 DotProd parity mismatch");

    // Test block64
    let mut out0_neon_64 = vec![0.0f32; 32];
    let mut out1_neon_64 = vec![0.0f32; 32];
    let mut out0_dotprod_64 = vec![0.0f32; 32];
    let mut out1_dotprod_64 = vec![0.0f32; 32];
    let ptr0 = transformed.as_ptr();
    let ptr1 = unsafe { transformed.as_ptr().add(32 * bytes_per_row) };

    unsafe {
        gemv_lut_block64_neon(
            ptr0,
            ptr1,
            qlut.as_ptr(),
            bytes_per_row,
            out0_neon_64.as_mut_ptr(),
            out1_neon_64.as_mut_ptr(),
        );
        gemv_lut_block64_neon_dotprod(
            ptr0,
            ptr1,
            qlut.as_ptr(),
            bytes_per_row,
            out0_dotprod_64.as_mut_ptr(),
            out1_dotprod_64.as_mut_ptr(),
        );
    }
    assert_eq!(out0_neon_64, out0_dotprod_64, "block64 (B0) DotProd parity mismatch");
    assert_eq!(out1_neon_64, out1_dotprod_64, "block64 (B1) DotProd parity mismatch");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_weight_transform_untransform_64_roundtrip() {
    let test_shapes = [(64, 160), (128, 320), (256, 640)];

    for (rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 13 + 7) % 3) as i8 - 1)
            .collect();

        let packed = pack_matrix(&trits, rows, cols);
        let transformed = transform_weights_lut_64(&packed, rows, cols);
        let restored = untransform_weights_lut_64(&transformed, rows, cols);

        assert_eq!(
            restored, packed,
            "64-row tiled transform roundtrip failed for shape ({}, {})",
            rows, cols
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemv_lut_64_avx512_exact_match() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512vbmi") {
        return;
    }

    let test_shapes = [(64, 160), (64, 320), (128, 640)];

    for (rows, cols) in test_shapes {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 17 + 5) % 3) as i8 - 1)
            .collect();

        let activations: Vec<i8> = (0..cols)
            .map(|i| (((i * 23 + 7) % 51) as i16 - 25) as i8)
            .collect();

        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut dot: i32 = 0;
            for c in 0..cols {
                dot += trits[r * cols + c] as i32 * activations[c] as i32;
            }
            expected[r] = dot as f32;
        }

        let packed = pack_matrix(&trits, rows, cols);
        let transformed_64 = transform_weights_lut_64(&packed, rows, cols);
        let qlut_512 = build_activation_lut_avx512(&activations, cols);

        let mut actual_1xn = vec![0.0f32; rows];
        gemv_lut_64_1xn_avx512(&transformed_64, &qlut_512, rows, cols, &mut actual_1xn);

        for r in 0..rows {
            assert_eq!(
                actual_1xn[r], expected[r],
                "AVX-512 GEMV 1xn mismatch at row {} for shape ({}, {}): got {}, expected {}",
                r, rows, cols, actual_1xn[r], expected[r]
            );
        }

        let mut actual_parallel = vec![0.0f32; rows];
        gemv_lut_64_parallel_avx512(&transformed_64, &qlut_512, rows, cols, &mut actual_parallel, 4);

        for r in 0..rows {
            assert_eq!(
                actual_parallel[r], expected[r],
                "AVX-512 GEMV parallel mismatch at row {} for shape ({}, {}): got {}, expected {}",
                r, rows, cols, actual_parallel[r], expected[r]
            );
        }
    }
}

