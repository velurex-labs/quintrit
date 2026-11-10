//! Contract tests of the safe GEMV entry points: a buffer that is too short must make the function panic before any
//! kernel reads or writes through a raw pointer. Before these checks existed, `gemv_lut_1xn` on an empty weight slice
//! read out of bounds and crashed with a segmentation fault.

use quintrit_core::lut::{
    activation_lut_len, build_activation_lut, build_activation_lut_wide, gemv_lut_1xn, gemv_lut_parallel,
    gemv_lut_wide_1xn, gemv_lut_wide_parallel, transform_weights_lut, wide_activation_lut_len,
};
use quintrit_core::pack::{pack_matrix, tile_row_bytes};
use std::panic::{catch_unwind, AssertUnwindSafe};

const ROWS: usize = 64;
const COLS: usize = 200; // one full 160-trit block and a packed tail

fn fixture() -> (Vec<u8>, Vec<i8>) {
    let trits: Vec<i8> = (0..ROWS * COLS).map(|i| (i % 3) as i8 - 1).collect();
    let tiled = transform_weights_lut(&pack_matrix(&trits, ROWS, COLS), ROWS, COLS);
    let acts: Vec<i8> = (0..COLS).map(|i| (i as i32 * 37 % 255 - 127) as i8).collect();
    (tiled, acts)
}

/// Runs `f` and asserts that it panics with a message containing `needle`.
fn assert_panics_with(needle: &str, f: impl FnOnce()) {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("the call must panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains(needle), "panic message {msg:?} does not mention {needle:?}");
}

#[test]
fn table_length_helpers_match_the_builders() {
    let (_, acts) = fixture();
    assert_eq!(build_activation_lut(&acts, COLS).len(), activation_lut_len(COLS));
    assert_eq!(build_activation_lut_wide(&acts, COLS).len(), wide_activation_lut_len(COLS));
}

#[test]
fn narrow_gemv_rejects_short_buffers() {
    let (tiled, acts) = fixture();
    let qlut = build_activation_lut(&acts, COLS);
    let short_w = &tiled[..tiled.len() - 1];
    let short_t = &qlut[..qlut.len() - 1];

    assert_panics_with("weight buffer too small", || gemv_lut_1xn(&[], &qlut, ROWS, COLS, &mut [0.0; ROWS]));
    assert_panics_with("weight buffer too small", || gemv_lut_1xn(short_w, &qlut, ROWS, COLS, &mut [0.0; ROWS]));
    assert_panics_with("activation table too small", || gemv_lut_1xn(&tiled, short_t, ROWS, COLS, &mut [0.0; ROWS]));
    assert_panics_with("output buffer too small", || gemv_lut_1xn(&tiled, &qlut, ROWS, COLS, &mut [0.0; ROWS - 1]));
    assert_panics_with("multiple of 32", || gemv_lut_1xn(&tiled, &qlut, ROWS - 1, COLS, &mut [0.0; ROWS]));

    assert_panics_with("weight buffer too small", || gemv_lut_parallel(short_w, &qlut, ROWS, COLS, &mut [0.0; ROWS], 2));
    assert_panics_with("activation table too small", || {
        gemv_lut_parallel(&tiled, short_t, ROWS, COLS, &mut [0.0; ROWS], 2)
    });
    assert_panics_with("output buffer too small", || {
        gemv_lut_parallel(&tiled, &qlut, ROWS, COLS, &mut [0.0; ROWS - 1], 2)
    });
}

#[test]
fn wide_gemv_rejects_short_buffers() {
    let (tiled, acts) = fixture();
    let wide = build_activation_lut_wide(&acts, COLS);
    let short_w = &tiled[..tiled.len() - 1];

    assert_panics_with("weight buffer too small", || gemv_lut_wide_1xn(&[], &wide, ROWS, COLS, &mut [0.0; ROWS]));
    assert_panics_with("weight buffer too small", || gemv_lut_wide_1xn(short_w, &wide, ROWS, COLS, &mut [0.0; ROWS]));
    assert_panics_with("wide (lo|hi) table", || {
        gemv_lut_wide_1xn(&tiled, &wide[..wide.len() - 2], ROWS, COLS, &mut [0.0; ROWS])
    });
    assert_panics_with("output buffer too small", || {
        gemv_lut_wide_1xn(&tiled, &wide, ROWS, COLS, &mut [0.0; ROWS - 1])
    });
    assert_panics_with("weight buffer too small", || {
        gemv_lut_wide_parallel(short_w, &wide, ROWS, COLS, &mut [0.0; ROWS], 2)
    });
    assert_panics_with("output buffer too small", || {
        gemv_lut_wide_parallel(&tiled, &wide, ROWS, COLS, &mut [0.0; ROWS - 1], 2)
    });
}

#[test]
fn checked_entry_points_still_compute_the_exact_product() {
    let (tiled, acts) = fixture();
    let trits: Vec<i8> = (0..ROWS * COLS).map(|i| (i % 3) as i8 - 1).collect();
    let expect: Vec<f32> = (0..ROWS)
        .map(|r| (0..COLS).map(|c| trits[r * COLS + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
        .collect();
    let wide = build_activation_lut_wide(&acts, COLS);
    for threads in [1, 2] {
        let mut out = vec![0.0f32; ROWS];
        gemv_lut_wide_parallel(&tiled, &wide, ROWS, COLS, &mut out, threads);
        assert_eq!(out, expect, "{threads} thread(s)");
    }
    // A longer weight buffer (for example a view into a larger allocation) is accepted.
    let mut longer = tiled.clone();
    longer.extend_from_slice(&[125; 64]);
    let mut out = vec![0.0f32; ROWS];
    gemv_lut_wide_1xn(&longer, &wide, ROWS, COLS, &mut out);
    assert_eq!(out, expect);
    assert_eq!(tiled.len(), ROWS * tile_row_bytes(COLS));
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use quintrit_core::lut::{
        activation_lut_len_avx512, avx512_supported, build_activation_lut_avx512, build_activation_lut_wide_avx512,
        gemv_lut_64_1xn_avx512, gemv_lut_64_parallel_avx512, gemv_lut_64_wide_1xn_avx512,
        gemv_lut_64_wide_parallel_avx512, transform_weights_lut_64,
    };

    fn fixture_64() -> (Vec<u8>, Vec<i8>) {
        let trits: Vec<i8> = (0..ROWS * COLS).map(|i| (i % 3) as i8 - 1).collect();
        let tiled = transform_weights_lut_64(&pack_matrix(&trits, ROWS, COLS), ROWS, COLS);
        let acts: Vec<i8> = (0..COLS).map(|i| (i as i32 * 37 % 255 - 127) as i8).collect();
        (tiled, acts)
    }

    /// Without AVX-512 every entry point must refuse to run (executing the kernels would be undefined behaviour);
    /// with AVX-512 a short buffer must be rejected.
    #[test]
    fn avx512_entry_points_check_cpu_and_buffers() {
        let (tiled, acts) = fixture_64();
        let narrow = build_activation_lut_avx512(&acts, COLS);
        assert_eq!(narrow.len(), activation_lut_len_avx512(COLS));
        let short_w = &tiled[..tiled.len() - 1];

        if !avx512_supported() {
            let needle = "need AVX-512";
            assert_panics_with(needle, || gemv_lut_64_1xn_avx512(&tiled, &narrow, ROWS, COLS, &mut [0.0; ROWS]));
            assert_panics_with(needle, || {
                gemv_lut_64_parallel_avx512(&tiled, &narrow, ROWS, COLS, &mut [0.0; ROWS], 2)
            });
            let wide = vec![0i8; quintrit_core::lut::wide_activation_lut_len_avx512(COLS)];
            assert_panics_with(needle, || gemv_lut_64_wide_1xn_avx512(&tiled, &wide, ROWS, COLS, &mut [0.0; ROWS]));
            assert_panics_with(needle, || {
                gemv_lut_64_wide_parallel_avx512(&tiled, &wide, ROWS, COLS, &mut [0.0; ROWS], 2)
            });
            return;
        }

        let wide = build_activation_lut_wide_avx512(&acts, COLS);
        assert_panics_with("weight buffer too small", || {
            gemv_lut_64_1xn_avx512(short_w, &narrow, ROWS, COLS, &mut [0.0; ROWS])
        });
        assert_panics_with("activation table too small", || {
            gemv_lut_64_1xn_avx512(&tiled, &narrow[..narrow.len() - 1], ROWS, COLS, &mut [0.0; ROWS])
        });
        assert_panics_with("output buffer too small", || {
            gemv_lut_64_1xn_avx512(&tiled, &narrow, ROWS, COLS, &mut [0.0; ROWS - 1])
        });
        assert_panics_with("output buffer too small", || {
            gemv_lut_64_parallel_avx512(&tiled, &narrow, ROWS, COLS, &mut [0.0; ROWS - 1], 2)
        });
        assert_panics_with("weight buffer too small", || {
            gemv_lut_64_wide_1xn_avx512(short_w, &wide, ROWS, COLS, &mut [0.0; ROWS])
        });
        assert_panics_with("output buffer too small", || {
            gemv_lut_64_wide_parallel_avx512(&tiled, &wide, ROWS, COLS, &mut [0.0; ROWS - 1], 2)
        });

        let trits: Vec<i8> = (0..ROWS * COLS).map(|i| (i % 3) as i8 - 1).collect();
        let expect: Vec<f32> = (0..ROWS)
            .map(|r| (0..COLS).map(|c| trits[r * COLS + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
            .collect();
        let mut out = vec![0.0f32; ROWS];
        gemv_lut_64_wide_parallel_avx512(&tiled, &wide, ROWS, COLS, &mut out, 2);
        assert_eq!(out, expect);
    }
}
