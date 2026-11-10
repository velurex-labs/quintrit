//! Per-algorithm correctness tests for every GEMV kernel, packer and table builder in
//! `quintrit-core`. Each SIMD variant is validated against an independent brute-force ground truth
//! (and therefore against the portable scalar implementation) on shapes that exercise full tiles,
//! remainder rows and remainder columns. Architecture-specific kernels are compiled and executed
//! only on the matching target and only when the CPU advertises the required features.
//!
//! The ground truth (`truth_i64`) is plain scalar code on standard CPU operations: a simple loop over the unpacked
//! trits and activations with `i64` accumulation. It uses no SIMD and none of the library's packing, tiling or lookup
//! tables, so the result is exact and a kernel must reproduce it bit for bit.

use quintrit_core::lut::{
    build_activation_lut, build_activation_lut_into, gemv_lut_1xn, gemv_lut_parallel,
    transform_weights_lut, untransform_weights_lut,
};
use quintrit_core::pack::{pack_matrix, unpack_matrix};
use quintrit_core::threading::get_thread_pool;

// ----------------------------------------------------------------------------
// Deterministic data helpers
// ----------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn trits(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.next() % 3) as i8 - 1).collect()
    }
    /// Small-range activations: pair-LUT entries are clamped to int8, so the LUT kernels are only
    /// exact for activations whose pair sums stay inside [-128, 127].
    fn acts(&mut self, n: usize, bound: i64) -> Vec<i8> {
        (0..n)
            .map(|_| ((self.next() % (2 * bound as u64 + 1)) as i64 - bound) as i8)
            .collect()
    }
}

/// Tells the user, once per test binary, what the ground truth is. It writes to the raw standard error stream because
/// the test harness swallows `println!` and `eprintln!` output of passing tests.
fn announce_ground_truth() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use std::io::Write;
        let _ = std::io::stderr().write_all(
            b"\nGround truth: brute-force product in plain scalar CPU code (standard integer operations, no SIMD).\n\
              Every SIMD kernel is compared with it and must return exactly the same integers.\n\n",
        );
    });
}

fn truth_i64(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<i64> {
    announce_ground_truth();
    (0..rows)
        .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum())
        .collect()
}

// ----------------------------------------------------------------------------
// 5-trit (1.60 bpw) packing, activation QLUT and LUT GEMV
// ----------------------------------------------------------------------------

#[test]
fn quintrit_matrix_pack_and_activation_lut_builders() {
    let mut rng = Rng::new(12);
    for &(rows, cols) in &[(2usize, 160usize), (5, 320), (3, 250)] {
        let trits = rng.trits(rows * cols);
        let packed = pack_matrix(&trits, rows, cols);
        assert_eq!(unpack_matrix(&packed, rows, cols), trits, "5-trit roundtrip ({rows}x{cols})");
    }

    // The in-place builder must produce exactly what the allocating builder returns, and must
    // not read or write outside of the documented length.
    for &cols in &[160usize, 250, 640] {
        let acts = rng.acts(cols, 60);
        let reference = build_activation_lut(&acts, cols);
        let bytes_per_row = quintrit_core::pack::tile_row_bytes(cols);
        assert_eq!(reference.len(), bytes_per_row * 64 + bytes_per_row.next_multiple_of(32));

        let mut buffer = vec![0x55i8; reference.len() + 64];
        let written = build_activation_lut_into(&acts, cols, &mut buffer);
        assert_eq!(written, reference.len());
        assert_eq!(&buffer[..written], &reference[..], "build_activation_lut_into (cols={cols})");
        assert!(buffer[written..].iter().all(|&b| b == 0x55), "wrote past the returned length");
    }
}

#[test]
fn quintrit_tiled_layout_gemv_and_threading_match_ground_truth() {
    let mut rng = Rng::new(13);
    for &(rows, cols) in &[(32usize, 160usize), (64, 250), (96, 320)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 40);
        let packed = pack_matrix(&trits, rows, cols);
        let tiled = transform_weights_lut(&packed, rows, cols);
        assert_eq!(untransform_weights_lut(&tiled, rows, cols), packed, "layout roundtrip ({rows}x{cols})");

        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();
        let qlut = build_activation_lut(&acts, cols);

        let mut out = vec![0.0f32; rows];
        gemv_lut_1xn(&tiled, &qlut, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_lut_1xn ({rows}x{cols})");

        for threads in [1usize, 2, 3, 8] {
            let mut par = vec![0.0f32; rows];
            gemv_lut_parallel(&tiled, &qlut, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "gemv_lut_parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn quintrit_avx512_layout_lut_and_gemv_match_ground_truth() {
    use quintrit_core::simd::avx512::*;

    let mut rng = Rng::new(14);

    // The 64-row layout and its aliases are plain data movement and run everywhere.
    for &(rows, cols) in &[(64usize, 160usize), (128, 250)] {
        let trits = rng.trits(rows * cols);
        let packed = pack_matrix(&trits, rows, cols);
        let tiled = transform_weights_quintrit_lut_64(&packed, rows, cols);
        assert_eq!(tiled, transform_weights_lut_64(&packed, rows, cols));
        assert_eq!(untransform_weights_quintrit_lut_64(&tiled, rows, cols), packed);
        assert_eq!(untransform_weights_lut_64(&tiled, rows, cols), packed);
    }

    if !(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi"))
    {
        return;
    }

    for &(rows, cols) in &[(64usize, 160usize), (128, 320)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 40);
        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();
        let packed = pack_matrix(&trits, rows, cols);
        let tiled = transform_weights_quintrit_lut_64(&packed, rows, cols);

        let qlut = build_activation_lut_avx512(&acts, cols);
        assert_eq!(build_quintrit_lut_avx512(&acts, cols), qlut, "alias builder differs");

        let mut out = vec![0.0f32; rows];
        gemv_quintrit_lut_64_1xn(&tiled, &qlut, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_quintrit_lut_64_1xn ({rows}x{cols})");

        let bytes_per_row = quintrit_core::pack::tile_row_bytes(cols);
        let mut vbmi = vec![0.0f32; 64];
        unsafe { gemv_quintrit_lut_block64_avx512_vbmi(tiled.as_ptr(), qlut.as_ptr(), bytes_per_row, vbmi.as_mut_ptr()) };
        assert_eq!(vbmi, expected[..64], "VBMI block kernel ({rows}x{cols})");
        if is_x86_feature_detected!("avx512vnni") {
            let mut vnni = vec![0.0f32; 64];
            unsafe { gemv_quintrit_lut_block64_avx512_vnni(tiled.as_ptr(), qlut.as_ptr(), bytes_per_row, vnni.as_mut_ptr()) };
            assert_eq!(vnni, vbmi, "VNNI and VBMI block kernels must agree ({rows}x{cols})");
        }

        for threads in [1usize, 3] {
            let mut par = vec![0.0f32; rows];
            gemv_quintrit_lut_64_parallel(&tiled, &qlut, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "gemv_quintrit_lut_64_parallel threads={threads} ({rows}x{cols})");
        }
    }
}

// `smmla` is emitted as inline assembly that needs i8mm at compile time (Neoverse-N2, not Cortex-A76).
#[cfg(all(target_arch = "aarch64", target_feature = "i8mm"))]
#[test]
fn neon_i8mm_smmla_matches_matrix_multiply() {
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    use core::arch::aarch64::*;
    use quintrit_core::simd::neon::smmla_s32;

    let a: [i8; 16] = [1, -2, 3, -4, 5, -6, 7, -8, 9, 10, -11, 12, -13, 14, -15, 16];
    let b: [i8; 16] = [2, 3, -4, 5, -6, 7, 8, -9, -1, 1, 2, -2, 3, -3, 4, -4];
    let mut acc = [100i32, 200, 300, 400];
    let mut expected = acc;
    for i in 0..2 {
        for j in 0..2 {
            expected[i * 2 + j] += (0..8).map(|k| a[i * 8 + k] as i32 * b[j * 8 + k] as i32).sum::<i32>();
        }
    }
    unsafe {
        let r = smmla_s32(vld1q_s32(acc.as_ptr()), vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr()));
        vst1q_s32(acc.as_mut_ptr(), r);
    }
    assert_eq!(acc, expected);
}

// ----------------------------------------------------------------------------
// Threading
// ----------------------------------------------------------------------------

#[test]
fn thread_pool_is_sized_and_cached() {
    for n in [1usize, 2, 3] {
        let pool = get_thread_pool(n);
        assert_eq!(pool.current_num_threads(), n);
        assert!(std::ptr::eq(pool, get_thread_pool(n)), "pool for {n} threads must be reused");
    }
    // Work submitted through the pool runs on that pool's workers.
    let seen = get_thread_pool(2).install(rayon::current_num_threads);
    assert_eq!(seen, 2);
}

/// Worst case for the NEON 5-trit LUT kernel inside its guaranteed range: a table entry holds three
/// products, so |entry| reaches 126 at |x| = 42 when weights and activations align fully.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_quintrit_lut_is_exact_for_worst_case_at_42_levels() {
    let (rows, cols) = (64usize, 640usize);
    let mut rng = Rng::new(98);
    for alternate in [false, true] {
        let acts: Vec<i8> = (0..cols).map(|c| if alternate && c % 2 == 1 { -42 } else { 42 }).collect();
        let mut trits = rng.trits(rows * cols);
        for c in 0..cols {
            trits[c] = 1; // row 0: every product is +42 (maximal positive entries)
            trits[cols + c] = -1; // row 1: every product is -42
            trits[2 * cols + c] = acts[c].signum(); // row 2: aligned with the activation signs
            trits[3 * cols + c] = -acts[c].signum();
        }
        let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);
        let qlut = build_activation_lut(&acts, cols);
        let mut out = vec![0.0f32; rows];
        gemv_lut_1xn(&tiled, &qlut, rows, cols, &mut out);
        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();
        assert_eq!(out, expected, "alternate={alternate}");
    }
}

/// Known limitation of the narrow kernel: the NEON 5-trit LUT kernel is inexact once |x| exceeds
/// roughly 45 because a three-product table entry saturates int8 (the AVX2 kernel is exact up to 63).
/// The engine no longer feeds it such values: it splits activations into hi/lo parts first (see
/// `wide_lut_is_exact_over_the_full_int8_range`). Run with `--ignored` to see the raw kernel fail.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "known issue: NEON Quintrit LUT kernel is inexact for |x| > ~45"]
fn neon_quintrit_lut_is_exact_at_engine_activation_range() {
    let (rows, cols) = (64usize, 640usize);
    let mut rng = Rng::new(99);
    let trits = rng.trits(rows * cols);
    let acts = rng.acts(cols, 63);
    let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);
    let qlut = build_activation_lut(&acts, cols);
    let mut out = vec![0.0f32; rows];
    gemv_lut_1xn(&tiled, &qlut, rows, cols, &mut out);
    let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();
    assert_eq!(out, expected);
}

#[test]
fn activation_split_is_lossless_and_small() {
    let all: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    let (mut hi, mut lo) = (vec![0i8; 256], vec![0i8; 256]);
    quintrit_core::lut::split_activations_hi_lo(&all, &mut hi, &mut lo);
    for i in 0..256 {
        assert_eq!(16 * hi[i] as i32 + lo[i] as i32, all[i] as i32, "x = {}", all[i]);
        assert!((-8..=7).contains(&lo[i]), "lo {} out of range for x = {}", lo[i], all[i]);
        assert!((-8..=8).contains(&hi[i]), "hi {} out of range for x = {}", hi[i], all[i]);
    }
}

#[test]
fn wide_lut_is_exact_over_the_full_int8_range() {
    use quintrit_core::lut::{build_activation_lut_wide, build_activation_lut_wide_into, wide_activation_lut_len};
    let mut rng = Rng::new(21);
    for &(rows, cols) in &[(32usize, 160usize), (64, 250), (96, 640)] {
        let trits = rng.trits(rows * cols);
        // Full int8 range including the extremes, which saturate the plain tables.
        let acts: Vec<i8> = (0..cols)
            .map(|c| match c % 7 { 0 => i8::MIN, 1 => i8::MAX, _ => (rng.next() % 256) as u8 as i8 })
            .collect();
        let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);

        let wide = build_activation_lut_wide(&acts, cols);
        assert_eq!(wide.len(), wide_activation_lut_len(cols));
        let half = wide.len() / 2;

        // Caller-provided buffers give the same table and do not write past the reported length.
        let mut scratch = vec![0i8; 2 * cols];
        let mut buffer = vec![0x33i8; wide.len() + 16];
        assert_eq!(build_activation_lut_wide_into(&acts, cols, &mut scratch, &mut buffer), wide.len());
        assert_eq!(&buffer[..wide.len()], &wide[..]);
        assert!(buffer[wide.len()..].iter().all(|&b| b == 0x33));

        let (mut lo, mut hi) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        gemv_lut_1xn(&tiled, &wide[..half], rows, cols, &mut lo);
        gemv_lut_1xn(&tiled, &wide[half..], rows, cols, &mut hi);
        let got: Vec<f32> = lo.iter().zip(&hi).map(|(l, h)| l + 16.0 * h).collect();
        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();
        assert_eq!(got, expected, "wide LUT ({rows}x{cols})");
    }
}

#[test]
fn fused_wide_gemv_matches_ground_truth_for_all_row_counts_and_threads() {
    use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn, gemv_lut_wide_parallel};
    let mut rng = Rng::new(31);
    // Row counts cross the cache-block boundary (it depends on the row width), the NEON tile pairing
    // (odd number of 32-row tiles) and the thread split; 6000 columns makes blocks of 128 rows.
    for &(rows, cols) in &[(32usize, 160usize), (96, 250), (320, 640), (544, 1000), (96, 6000), (160, 23040)] {
        let trits = rng.trits(rows * cols);
        let acts: Vec<i8> = (0..cols)
            .map(|c| match c % 5 { 0 => i8::MIN, 1 => i8::MAX, _ => (rng.next() % 256) as u8 as i8 })
            .collect();
        let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);
        let wide = build_activation_lut_wide(&acts, cols);
        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();

        let mut single = vec![0.0f32; rows];
        gemv_lut_wide_1xn(&tiled, &wide, rows, cols, &mut single);
        assert_eq!(single, expected, "gemv_lut_wide_1xn ({rows}x{cols})");

        for threads in [2usize, 3, 4, 7, 16] {
            let mut par = vec![0.0f32; rows];
            gemv_lut_wide_parallel(&tiled, &wide, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "gemv_lut_wide_parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[test]
fn fused_wide_block_kernel_matches_two_pass_composition_and_ground_truth() {
    use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_1xn, gemv_lut_wide_1xn};
    // Column counts around the chunk length of the kernels (16 on AVX2, 32 on NEON), odd and even
    // numbers of 5-trit column bytes, and activations that include both int8 extremes.
    let mut rng = Rng::new(41);
    for &cols in &[5usize, 160, 161, 320, 480, 800, 1280, 2560, 4000] {
        let rows = 64usize;
        let trits = rng.trits(rows * cols);
        for pattern in 0..3 {
            let acts: Vec<i8> = (0..cols)
                .map(|c| match pattern {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    _ => match c % 6 { 0 => i8::MIN, 1 => i8::MAX, _ => (rng.next() % 256) as u8 as i8 },
                })
                .collect();
            let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);
            let wide = build_activation_lut_wide(&acts, cols);
            let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();

            let mut fused = vec![0.0f32; rows];
            gemv_lut_wide_1xn(&tiled, &wide, rows, cols, &mut fused);
            assert_eq!(fused, expected, "fused kernel vs ground truth (cols={cols}, pattern={pattern})");

            // Independent two-pass composition through the narrow single-table kernel.
            let half = wide.len() / 2;
            let (mut lo, mut hi) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            gemv_lut_1xn(&tiled, &wide[..half], rows, cols, &mut lo);
            gemv_lut_1xn(&tiled, &wide[half..], rows, cols, &mut hi);
            let two_pass: Vec<f32> = lo.iter().zip(&hi).map(|(l, h)| l + 16.0 * h).collect();
            assert_eq!(fused, two_pass, "fused vs two-pass (cols={cols}, pattern={pattern})");
        }
    }
}


#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_wide_gemv_is_exact_over_the_full_int8_range() {
    use quintrit_core::simd::avx512::{
        build_activation_lut_wide_avx512, gemv_lut_64_wide_1xn_avx512, gemv_lut_64_wide_parallel_avx512,
        wide_activation_lut_len_avx512,
    };
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")) {
        return;
    }
    let mut rng = Rng::new(51);
    // Row counts cross the cache-block size (which depends on the row width) and the thread split.
    for &(rows, cols) in &[(64usize, 160usize), (128, 320), (320, 1000), (192, 6000), (128, 23040)] {
        let trits = rng.trits(rows * cols);
        let acts: Vec<i8> = (0..cols)
            .map(|c| match c % 5 { 0 => i8::MIN, 1 => i8::MAX, _ => (rng.next() % 256) as u8 as i8 })
            .collect();
        let tiled = transform_weights_quintrit_lut_64_for_test(&trits, rows, cols);
        let wide = build_activation_lut_wide_avx512(&acts, cols);
        assert_eq!(wide.len(), wide_activation_lut_len_avx512(cols));
        let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();

        let mut single = vec![0.0f32; rows];
        gemv_lut_64_wide_1xn_avx512(&tiled, &wide, rows, cols, &mut single);
        assert_eq!(single, expected, "avx512 wide 1xn ({rows}x{cols})");
        for threads in [2usize, 3, 7] {
            let mut par = vec![0.0f32; rows];
            gemv_lut_64_wide_parallel_avx512(&tiled, &wide, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "avx512 wide parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn transform_weights_quintrit_lut_64_for_test(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    quintrit_core::simd::avx512::transform_weights_lut_64(&pack_matrix(trits, rows, cols), rows, cols)
}


#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_fused_wide_block_kernels_match_two_pass_composition() {
    use quintrit_core::simd::avx512::*;
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")) {
        return;
    }
    let mut rng = Rng::new(61);
    let has_vnni = is_x86_feature_detected!("avx512vnni");
    // Column counts around the 16-column chunk of the VBMI kernel (bytes_per_row is a multiple of 32).
    for &cols in &[5usize, 160, 161, 320, 800, 1280, 4000, 6000] {
        let rows = 64usize;
        let trits = rng.trits(rows * cols);
        let bytes_per_row = quintrit_core::pack::tile_row_bytes(cols);
        for pattern in 0..4 {
            let acts: Vec<i8> = (0..cols)
                .map(|c| match pattern {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    2 => if c % 2 == 0 { i8::MIN } else { i8::MAX },
                    _ => (rng.next() % 256) as u8 as i8,
                })
                .collect();
            let tiled = transform_weights_lut_64(&pack_matrix(&trits, rows, cols), rows, cols);
            let wide = build_activation_lut_wide_avx512_full(&acts, cols);
            let (lo, hi) = wide.split_at(wide.len() / 2);
            let expected: Vec<f32> = truth_i64(&trits, &acts, rows, cols).iter().map(|&v| v as f32).collect();

            let mut vbmi = vec![0.0f32; rows];
            unsafe { gemv_quintrit_lut_block64_wide_avx512_vbmi(tiled.as_ptr(), lo.as_ptr(), hi.as_ptr(), bytes_per_row, vbmi.as_mut_ptr()) };
            assert_eq!(vbmi, expected, "fused VBMI vs ground truth (cols={cols}, pattern={pattern})");

            if has_vnni {
                let mut vnni = vec![0.0f32; rows];
                unsafe { gemv_quintrit_lut_block64_wide_avx512_vnni(tiled.as_ptr(), lo.as_ptr(), hi.as_ptr(), bytes_per_row, vnni.as_mut_ptr()) };
                assert_eq!(vnni, expected, "fused VNNI vs ground truth (cols={cols}, pattern={pattern})");
            }

            // Independent two-pass composition through the single-table kernel.
            let (mut l, mut h) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            gemv_lut_64_1xn_avx512(&tiled, lo, rows, cols, &mut l);
            gemv_lut_64_1xn_avx512(&tiled, hi, rows, cols, &mut h);
            let two_pass: Vec<f32> = l.iter().zip(&h).map(|(a, b)| a + 16.0 * b).collect();
            assert_eq!(vbmi, two_pass, "fused vs two-pass (cols={cols}, pattern={pattern})");
        }
    }
}
