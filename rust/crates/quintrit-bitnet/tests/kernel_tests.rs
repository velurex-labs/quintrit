//! Per-algorithm correctness tests for every GEMV kernel, packer and table builder in
//! `quintrit-core`. Each SIMD variant is validated against an independent brute-force ground truth
//! (and therefore against the portable scalar implementation) on shapes that exercise full tiles,
//! remainder rows and remainder columns. Architecture-specific kernels are compiled and executed
//! only on the matching target and only when the CPU advertises the required features.

use quintrit_bitnet::*;

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

fn truth_i64(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<i64> {
    (0..rows)
        .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum())
        .collect()
}

/// Raw MAD accumulator: sum of (trit + 1) * activation.
fn truth_mad_raw(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| (trits[r * cols + c] as i64 + 1) * acts[c] as i64)
                .sum::<i64>() as f32
        })
        .collect()
}

fn pack_rows_4(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    (0..rows)
        .flat_map(|r| pack_trits_4(&trits[r * cols..(r + 1) * cols]))
        .collect()
}

// ----------------------------------------------------------------------------
// 4-trit scalar format, pair LUT and scalar GEMV
// ----------------------------------------------------------------------------

#[test]
fn pack4_known_bytes_roundtrip_and_invalid_fallback() {
    // codes: -1 -> 0, 0 -> 1, +1 -> 2; byte = c0 | c1 << 2 | c2 << 4 | c3 << 6
    assert_eq!(pack_trits_4(&[-1, 0, 1, 0]), vec![0 | (1 << 2) | (2 << 4) | (1 << 6)]);
    assert_eq!(pack_trits_4(&[1, 1, 1, 1]), vec![0xAA]);
    assert_eq!(pack_trits_4(&[-1, -1, -1, -1]), vec![0x00]);

    // Partial trailing byte and out-of-range values (neutral fallback).
    assert_eq!(pack_trits_4(&[1, -1, 0]), vec![2 | (0 << 2) | (1 << 4)]);
    assert_eq!(unpack_trits_4(&pack_trits_4(&[7, -9]), 2), vec![0, 0]);

    let mut rng = Rng::new(1);
    for n in [1usize, 3, 4, 5, 16, 33, 130] {
        let t = rng.trits(n);
        assert_eq!(unpack_trits_4(&pack_trits_4(&t), n), t, "roundtrip length {n}");
    }
    // The unused code 3 decodes to a neutral weight.
    assert_eq!(unpack_trits_4(&[0xFF], 4), vec![0, 0, 0, 0]);
}

#[test]
fn pair_lut_entries_follow_definition() {
    let decode = |code: usize| match code {
        0 => -1i32,
        2 => 1,
        _ => 0,
    };
    for &(xa, xb) in &[(0i8, 0i8), (5, -7), (-128, -128), (127, 127), (127, -128), (-60, 33)] {
        let table = build_pair_lut_entry(xa, xb);
        for idx in 0..16usize {
            let wa = decode(idx & 3);
            let wb = decode((idx >> 2) & 3);
            let expected = (wa * xa as i32 + wb * xb as i32).clamp(-128, 127);
            assert_eq!(table[idx] as i32, expected, "pair ({xa},{xb}) idx {idx}");
        }
    }

    // Odd-length activation vectors are zero padded to a full pair.
    let x = [3i8, -4, 9];
    let lut = build_activation_lut_scalar(&x);
    assert_eq!(lut.len(), 2 * 16);
    assert_eq!(&lut[16..32], &build_pair_lut_entry(9, 0));
}

#[test]
fn scalar_dot_and_gemv_match_ground_truth() {
    let mut rng = Rng::new(2);
    for &(rows, cols) in &[(1usize, 4usize), (3, 64), (5, 130), (8, 257), (33, 96)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 60);
        let packed = pack_rows_4(&trits, rows, cols);
        let bytes_per_row = cols.div_ceil(4);
        let expected = truth_i64(&trits, &acts, rows, cols);

        let mut out = vec![0i64; rows];
        gemv_ternary_int8_scalar(&packed, &acts, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_ternary_int8_scalar ({rows}x{cols})");

        // The standalone dot product processes whole bytes only.
        let full = cols / 4 * 4;
        for r in 0..rows {
            let row_w = &packed[r * bytes_per_row..(r + 1) * bytes_per_row];
            let exp_dot: i64 = (0..full).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum();
            assert_eq!(dot_ternary_int8_scalar(row_w, &acts[..full], full), exp_dot, "dot row {r}");
        }
    }
}

// ----------------------------------------------------------------------------
// 32-row tiled layout and tiled LUT GEMV
// ----------------------------------------------------------------------------

#[test]
fn tiled32_layout_pack_transform_untransform_agree() {
    let mut rng = Rng::new(3);
    for &(rows, cols) in &[(32usize, 128usize), (64, 132), (70, 130), (31, 64), (96, 4)] {
        let trits = rng.trits(rows * cols);
        let linear = pack_rows_4(&trits, rows, cols);

        let tiled = transform_weights_tiled_32(&linear, rows, cols);
        assert_eq!(tiled.len(), linear.len());
        assert_eq!(pack_trits_tiled_32(&trits, rows, cols), tiled, "direct pack ({rows}x{cols})");
        assert_eq!(untransform_weights_tiled_32(&tiled, rows, cols), linear, "untransform ({rows}x{cols})");

        // Documented layout: within a 32-row block the 32 row bytes of one column byte are contiguous.
        if rows >= 32 {
            let bytes_per_row = cols.div_ceil(4);
            for c in [0, bytes_per_row - 1] {
                for r in 0..32 {
                    assert_eq!(tiled[c * 32 + r], linear[r * bytes_per_row + c]);
                }
            }
        }
    }
}

#[test]
fn tiled32_scalar_lut_gemv_matches_ground_truth() {
    let mut rng = Rng::new(4);
    for &(rows, cols) in &[(32usize, 128usize), (64, 130), (70, 132), (31, 64)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 60);
        let tiled = pack_trits_tiled_32(&trits, rows, cols);
        let qlut = build_activation_lut_scalar(&acts);
        let mut out = vec![0i64; rows];
        gemv_lut_tiled32_scalar(&tiled, &qlut, rows, cols, &mut out);
        assert_eq!(out, truth_i64(&trits, &acts, rows, cols), "tiled32 scalar ({rows}x{cols})");

        // The public dispatcher (best available SIMD) and its parallel form must agree.
        let mut dispatched = vec![0i64; rows];
        gemv_bitnet_lut(&tiled, &qlut, rows, cols, &mut dispatched);
        assert_eq!(dispatched, out, "gemv_bitnet_lut ({rows}x{cols})");
        for threads in [1usize, 3] {
            let mut par = vec![0i64; rows];
            gemv_bitnet_lut_parallel(&tiled, &qlut, rows, cols, &mut par, threads);
            assert_eq!(par, out, "gemv_bitnet_lut_parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_four_trit_kernels_match_ground_truth() {
    use quintrit_bitnet::avx2::*;
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = Rng::new(5);
    // `gemv_ternary_int8_avx2` adds the two pre-summed table entries in int8, i.e. it is exact only
    // while the sum of four products stays within [-128, 127] (|x| <= 31).
    const AVX2_LUT_BOUND: i64 = 31;

    // Horizontal sum helper.
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm256_setr_epi32(1, -2, 3, -4, 5, -6, 7, 100_000);
        assert_eq!(hsum_i32_avx2(v), 1 - 2 + 3 - 4 + 5 - 6 + 7 + 100_000);
    }

    for &(rows, cols) in &[(1usize, 128usize), (4, 256), (7, 130), (32, 128), (70, 132)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, AVX2_LUT_BOUND);
        let expected = truth_i64(&trits, &acts, rows, cols);
        let linear = pack_rows_4(&trits, rows, cols);
        let bytes_per_row = cols.div_ceil(4);

        let mut out = vec![0i64; rows];
        unsafe { gemv_ternary_int8_avx2(&linear, &acts, rows, cols, &mut out) };
        assert_eq!(out, expected, "gemv_ternary_int8_avx2 ({rows}x{cols})");

        // Row-wise dot product over whole bytes.
        let full = cols / 4 * 4;
        for r in 0..rows {
            let row_w = &linear[r * bytes_per_row..(r + 1) * bytes_per_row];
            let want: i64 = (0..full).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum();
            assert_eq!(unsafe { dot_ternary_int8_avx2(row_w, &acts[..full], full) }, want, "dot row {r}");
        }

        let tiled = pack_trits_tiled_32(&trits, rows, cols);
        let mut out_tiled_direct = vec![0i64; rows];
        unsafe { gemv_ternary_int8_tiled32_avx2(&tiled, &acts, rows, cols, &mut out_tiled_direct) };
        assert_eq!(out_tiled_direct, expected, "gemv_ternary_int8_tiled32_avx2 ({rows}x{cols})");

        let qlut = build_activation_lut_scalar(&acts);
        let mut out_lut = vec![0i64; rows];
        unsafe { gemv_lut_tiled32_avx2(&tiled, &qlut, rows, cols, &mut out_lut) };
        assert_eq!(out_lut, expected, "gemv_lut_tiled32_avx2 ({rows}x{cols})");

        for threads in [1usize, 2, 5] {
            let mut par = vec![0i64; rows];
            gemv_lut_tiled32_parallel(&tiled, &qlut, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "avx2 tiled32 parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_tiled32_lut_kernels_match_ground_truth() {
    use quintrit_bitnet::neon::{gemv_lut_tiled32_neon, gemv_lut_tiled32_parallel};
    let mut rng = Rng::new(6);
    for &(rows, cols) in &[(32usize, 128usize), (64, 132), (70, 130), (31, 64)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 60);
        let expected = truth_i64(&trits, &acts, rows, cols);
        let tiled = pack_trits_tiled_32(&trits, rows, cols);
        let qlut = build_activation_lut_scalar(&acts);

        let mut out = vec![0i64; rows];
        unsafe { gemv_lut_tiled32_neon(&tiled, &qlut, rows, cols, &mut out) };
        assert_eq!(out, expected, "gemv_lut_tiled32_neon ({rows}x{cols})");

        for threads in [1usize, 2, 4] {
            let mut par = vec![0i64; rows];
            gemv_lut_tiled32_parallel(&tiled, &qlut, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "neon tiled32 parallel threads={threads} ({rows}x{cols})");
        }
    }
}

// ----------------------------------------------------------------------------
// Planar MAD format (2.00 bpw) and MAD GEMV kernels
// ----------------------------------------------------------------------------

#[test]
fn planar_and_gguf_i2_s_packers_roundtrip() {
    let mut rng = Rng::new(7);
    for &(rows, cols) in &[(1usize, 128usize), (4, 256), (9, 384)] {
        let trits = rng.trits(rows * cols);

        let planar = pack_trits_planar(&trits, rows, cols);
        assert_eq!(planar.len(), rows * cols / 4);
        assert_eq!(unpack_trits_planar(&planar, rows, cols), trits, "planar ({rows}x{cols})");

        let gguf = pack_gguf_i2_s(&trits, rows, cols);
        assert!(gguf.len() >= rows * cols / 4);
        assert_eq!(unpack_gguf_i2_s(&gguf, rows, cols), trits, "gguf i2_s ({rows}x{cols})");
    }
}

#[test]
fn mad_scalar_and_dispatchers_match_ground_truth() {
    let mut rng = Rng::new(8);
    for &(rows, cols) in &[(1usize, 128usize), (4, 128), (7, 256), (9, 384), (12, 2048)] {
        let trits = rng.trits(rows * cols);
        // Full int8 range: MAD accumulates in int16 pairs followed by an int32 widening and is exact.
        let acts = rng.acts(cols, 127);
        let packed = pack_trits_planar(&trits, rows, cols);
        let expected = truth_mad_raw(&trits, &acts, rows, cols);

        let mut scalar = vec![0.0f32; rows];
        gemv_mad_scalar(&packed, &acts, rows, cols, &mut scalar);
        assert_eq!(scalar, expected, "gemv_mad_scalar ({rows}x{cols})");

        let mut best = vec![0.0f32; rows];
        gemv_mad(&packed, &acts, rows, cols, &mut best);
        assert_eq!(best, expected, "gemv_mad ({rows}x{cols})");

        for threads in [1usize, 2, 4, 7] {
            let mut par = vec![0.0f32; rows];
            gemv_mad_parallel(&packed, &acts, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "gemv_mad_parallel threads={threads} ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn mad_avx2_kernels_match_ground_truth() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = Rng::new(9);
    for &(rows, cols) in &[(4usize, 128usize), (8, 256), (7, 384), (9, 2048), (11, 2176)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 127);
        let packed = pack_trits_planar(&trits, rows, cols);
        let expected = truth_mad_raw(&trits, &acts, rows, cols);

        let mut out = vec![0.0f32; rows];
        gemv_mad_avx2(&packed, &acts, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_mad_avx2 ({rows}x{cols})");

        let mut out1 = vec![0.0f32; rows];
        unsafe { gemv_mad_1x1_avx2(&packed, &acts, rows, cols, &mut out1) };
        assert_eq!(out1, expected, "gemv_mad_1x1_avx2 ({rows}x{cols})");

        if rows % 4 == 0 {
            let mut outn = vec![0.0f32; rows];
            unsafe { gemv_mad_1xn_avx2(&packed, &acts, rows, cols, &mut outn) };
            assert_eq!(outn, expected, "gemv_mad_1xn_avx2 ({rows}x{cols})");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn mad_neon_dispatcher_matches_ground_truth() {
    let mut rng = Rng::new(10);
    for &(rows, cols) in &[(4usize, 128usize), (7, 256), (9, 2048), (11, 2176)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 127);
        let packed = pack_trits_planar(&trits, rows, cols);
        let expected = truth_mad_raw(&trits, &acts, rows, cols);

        let mut out = vec![0.0f32; rows];
        gemv_mad_neon(&packed, &acts, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_mad_neon ({rows}x{cols})");
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx512_bw() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
}

#[cfg(target_arch = "x86_64")]
#[test]
fn mad_avx512_kernels_match_ground_truth() {
    if !has_avx512_bw() {
        return;
    }
    let mut rng = Rng::new(11);
    for &(rows, cols) in &[(8usize, 128usize), (16, 256), (13, 384), (9, 2048)] {
        let trits = rng.trits(rows * cols);
        let acts = rng.acts(cols, 127);
        let packed = pack_trits_planar(&trits, rows, cols);
        let expected = truth_mad_raw(&trits, &acts, rows, cols);

        let mut out = vec![0.0f32; rows];
        gemv_mad_avx512(&packed, &acts, rows, cols, &mut out);
        assert_eq!(out, expected, "gemv_mad_avx512 ({rows}x{cols})");

        let mut out1 = vec![0.0f32; rows];
        unsafe { gemv_mad_1x1_avx512(&packed, &acts, rows, cols, &mut out1) };
        assert_eq!(out1, expected, "gemv_mad_1x1_avx512 ({rows}x{cols})");

        if rows % 8 == 0 {
            let mut outn = vec![0.0f32; rows];
            unsafe { gemv_mad_1xn_avx512(&packed, &acts, rows, cols, &mut outn) };
            assert_eq!(outn, expected, "gemv_mad_1xn_avx512 ({rows}x{cols})");
            let mut wrapper = vec![0.0f32; rows];
            gemv_bitnet_mad_avx512_1xn(&packed, &acts, rows, cols, &mut wrapper);
            assert_eq!(wrapper, expected, "gemv_bitnet_mad_avx512_1xn ({rows}x{cols})");
        }
        for threads in [1usize, 3] {
            let mut par = vec![0.0f32; rows];
            gemv_mad_parallel_avx512(&packed, &acts, rows, cols, &mut par, threads);
            assert_eq!(par, expected, "gemv_mad_parallel_avx512 threads={threads} ({rows}x{cols})");
            let mut wrapper = vec![0.0f32; rows];
            gemv_bitnet_mad_avx512_parallel(&packed, &acts, rows, cols, &mut wrapper, threads);
            assert_eq!(wrapper, expected, "gemv_bitnet_mad_avx512_parallel threads={threads}");
        }
    }
    unsafe {
        use core::arch::x86_64::*;
        assert_eq!(hsum_i32_avx512(_mm512_set1_epi32(3)), 48);
    }
}

