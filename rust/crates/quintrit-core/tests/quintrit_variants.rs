//! Correctness of the experimental kernel variants of the exact (hi/lo) velurex GEMV: every variant must give the
//! integer result of the brute-force dot product bit for bit, over the full int8 activation range and for the
//! worst-case magnitudes of the intermediate sums.
//!
//! The brute-force dot product is plain scalar code on standard CPU operations (a simple loop over the unpacked trits
//! and activations with `i64` accumulation), without SIMD and without the library's packing, tiling or lookup tables.

/// Tells the user, once per test binary, what the ground truth is. It writes to the raw standard error stream because
/// the test harness swallows `println!` and `eprintln!` output of passing tests.
#[allow(dead_code)]
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

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn_avx2_variant, transform_weights_lut};
    use quintrit_core::pack::pack_matrix;

    const VARIANTS: [u32; 4] = [0, 1, 2, 4];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn truth(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<f32> {
        super::announce_ground_truth();
        (0..rows)
            .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
            .collect()
    }

    fn check(variant: u32, trits: &[i8], acts: &[i8], rows: usize, cols: usize, what: &str) {
        let tiled = transform_weights_lut(&pack_matrix(trits, rows, cols), rows, cols);
        let wide = build_activation_lut_wide(acts, cols);
        let mut got = vec![0.0f32; rows];
        gemv_lut_wide_1xn_avx2_variant(variant, &tiled, &wide, rows, cols, &mut got);
        assert_eq!(got, truth(trits, acts, rows, cols), "variant {variant}, {what} ({rows}x{cols})");
    }

    #[test]
    fn variants_match_ground_truth_on_random_data_and_odd_shapes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for &(rows, cols) in &[(32usize, 5usize), (32, 160), (64, 161), (64, 320), (96, 480), (64, 800), (32, 1280), (64, 2560), (32, 4000), (64, 3072), (32, 23040)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            for v in VARIANTS {
                check(v, &trits, &acts, rows, cols, "random");
            }
        }
    }

    #[test]
    fn variants_match_ground_truth_for_worst_case_magnitudes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // 120 = 16 * 8 - 8 and -8 give |lo| = 8 and |hi| = 8 / 0; with equal-signed trits every product adds up.
        let extremes: [i8; 8] = [120, -120, -8, 7, 127, -128, 119, -127];
        for &cols in &[160usize, 161, 480, 3072, 4000, 23040] {
            let rows = 32;
            for &w in &[1i8, -1, 0] {
                let trits = vec![w; rows * cols];
                for &a in &extremes {
                    let acts = vec![a; cols];
                    for v in VARIANTS {
                        check(v, &trits, &acts, rows, cols, &format!("constant w={w} a={a}"));
                    }
                }
                // Alternating extremes in neighbouring columns.
                let acts: Vec<i8> = (0..cols).map(|c| extremes[c % 8]).collect();
                for v in VARIANTS {
                    check(v, &trits, &acts, rows, cols, &format!("alternating w={w}"));
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use quintrit_core::pack::pack_matrix;
    use quintrit_core::simd::avx512::{
        build_activation_lut_wide_avx512, build_activation_lut_wide_avx512_full,
        gemv_lut_64_wide_1xn_avx512_compact_variant, gemv_lut_64_wide_1xn_avx512_multi_variant,
        gemv_lut_64_wide_1xn_avx512_v2_variant,
        gemv_lut_64_wide_1xn_avx512_variant, transform_weights_lut_64,
    };

    const VARIANTS: [u32; 34] = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn supported() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")
    }

    fn truth(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<f32> {
        super::announce_ground_truth();
        (0..rows)
            .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
            .collect()
    }

    fn check(variant: u32, trits: &[i8], acts: &[i8], rows: usize, cols: usize, what: &str) {
        let tiled = transform_weights_lut_64(&pack_matrix(trits, rows, cols), rows, cols);
        let mut got = vec![0.0f32; rows];
        if variant >= 30 {
            let wide = build_activation_lut_wide_avx512(acts, cols);
            gemv_lut_64_wide_1xn_avx512_multi_variant(variant, &tiled, &wide, rows, cols, &mut got);
        } else if variant >= 8 {
            let wide = build_activation_lut_wide_avx512(acts, cols);
            gemv_lut_64_wide_1xn_avx512_v2_variant(variant, &tiled, &wide, rows, cols, &mut got);
        } else if variant >= 4 {
            let wide = build_activation_lut_wide_avx512(acts, cols);
            gemv_lut_64_wide_1xn_avx512_compact_variant(variant, &tiled, &wide, rows, cols, &mut got);
        } else {
            let wide = build_activation_lut_wide_avx512_full(acts, cols);
            gemv_lut_64_wide_1xn_avx512_variant(variant, &tiled, &wide, rows, cols, &mut got);
        }
        assert_eq!(got, truth(trits, acts, rows, cols), "avx512 variant {variant}, {what} ({rows}x{cols})");
    }

    #[test]
    fn variants_match_ground_truth_on_random_data_and_odd_shapes() {
        if !supported() {
            return;
        }
        let mut rng = Rng(0x0BAD_CAFE_1234_5679);
        for &(rows, cols) in &[(64usize, 5usize), (64, 160), (128, 161), (64, 320), (64, 480), (64, 800), (128, 1280), (64, 2560), (64, 4000), (128, 3072), (64, 23040)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            for v in VARIANTS {
                check(v, &trits, &acts, rows, cols, "random");
            }
        }
    }

    /// The single-table upper-bound kernels are exact for activations with |x| <= 25 only.
    #[test]
    fn single_table_variants_are_exact_for_small_activations() {
        if !supported() {
            return;
        }
        let mut rng = Rng(0x5151_7777_ABCD_0123);
        for &(rows, cols) in &[(64usize, 5usize), (64, 161), (128, 480), (192, 3072), (64, 4000), (64, 23040)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            for &limit in &[25u64, 1] {
                let acts: Vec<i8> = (0..cols).map(|_| ((rng.next() % (2 * limit + 1)) as i64 - limit as i64) as i8).collect();
                let tiled = transform_weights_lut_64(&pack_matrix(&trits, rows, cols), rows, cols);
                let wide = build_activation_lut_wide_avx512(&acts, cols);
                for v in [40u32, 41] {
                    let mut got = vec![0.0f32; rows];
                    quintrit_core::simd::avx512::gemv_lut_64_single_table_variant(v, &tiled, &wide, rows, cols, &mut got);
                    // The single table is the `lo` part of the split, which equals the activation for |x| <= 8 only.
                    let truth_lo: Vec<f32> = (0..rows)
                        .map(|r| {
                            (0..cols)
                                .map(|c| {
                                    let x = acts[c] as i16;
                                    let lo = ((x + 8) & 15) - 8;
                                    trits[r * cols + c] as i64 * lo as i64
                                })
                                .sum::<i64>() as f32
                        })
                        .collect();
                    assert_eq!(got, truth_lo, "single-table variant {v} ({rows}x{cols}, limit {limit})");
                }
            }
        }
    }

    #[test]
    fn variants_match_ground_truth_for_worst_case_magnitudes() {
        if !supported() {
            return;
        }
        let extremes: [i8; 8] = [120, -120, -8, 7, 127, -128, 119, -127];
        for &cols in &[160usize, 161, 480, 3072, 4000, 23040] {
            let rows = 64;
            for &w in &[1i8, -1, 0] {
                let trits = vec![w; rows * cols];
                for &a in &extremes {
                    let acts = vec![a; cols];
                    for v in VARIANTS {
                        check(v, &trits, &acts, rows, cols, &format!("constant w={w} a={a}"));
                    }
                }
                let acts: Vec<i8> = (0..cols).map(|c| extremes[c % 8]).collect();
                for v in VARIANTS {
                    check(v, &trits, &acts, rows, cols, &format!("alternating w={w}"));
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn_neon_variant, transform_weights_lut};
    use quintrit_core::pack::pack_matrix;

    const VARIANTS: [u32; 18] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn truth(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<f32> {
        super::announce_ground_truth();
        (0..rows)
            .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
            .collect()
    }

    fn check(variant: u32, trits: &[i8], acts: &[i8], rows: usize, cols: usize, what: &str) {
        let tiled = transform_weights_lut(&pack_matrix(trits, rows, cols), rows, cols);
        let wide = build_activation_lut_wide(acts, cols);
        let mut got = vec![0.0f32; rows];
        gemv_lut_wide_1xn_neon_variant(variant, &tiled, &wide, rows, cols, &mut got);
        assert_eq!(got, truth(trits, acts, rows, cols), "neon variant {variant}, {what} ({rows}x{cols})");
    }

    #[test]
    fn variants_match_ground_truth_on_random_data_and_odd_shapes() {
        let mut rng = Rng(0x2468_ACE0_1357_9BDF);
        for &(rows, cols) in &[(32usize, 5usize), (32, 160), (64, 161), (64, 320), (96, 480), (64, 800), (32, 1280), (64, 2560), (32, 4000), (64, 3072), (32, 23040)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            for v in VARIANTS {
                check(v, &trits, &acts, rows, cols, "random");
            }
        }
    }

    #[test]
    fn variants_match_ground_truth_for_worst_case_magnitudes() {
        let extremes: [i8; 8] = [120, -120, -8, 7, 127, -128, 119, -127];
        for &cols in &[160usize, 161, 480, 3072, 4000, 23040] {
            let rows = 32;
            for &w in &[1i8, -1, 0] {
                let trits = vec![w; rows * cols];
                for &a in &extremes {
                    let acts = vec![a; cols];
                    for v in VARIANTS {
                        check(v, &trits, &acts, rows, cols, &format!("constant w={w} a={a}"));
                    }
                }
                let acts: Vec<i8> = (0..cols).map(|c| extremes[c % 8]).collect();
                for v in VARIANTS {
                    check(v, &trits, &acts, rows, cols, &format!("alternating w={w}"));
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod builders {
    use quintrit_core::lut::{build_activation_lut_into, split_activations_hi_lo, wide_activation_lut_len};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn builders_match_scalar_reference() {
        let mut rng = Rng(0xFEED_FACE_0BAD_F00D);
        for &cols in &[5usize, 160, 161, 480, 1000, 3072, 4000, 23040] {
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            let (mut hi, mut lo) = (vec![0i8; cols], vec![0i8; cols]);
            split_activations_hi_lo(&acts, &mut hi, &mut lo);

            if is_x86_feature_detected!("avx2") {
                let single = wide_activation_lut_len(cols) / 2;
                for part in [&lo, &hi] {
                    let mut want = vec![0x55i8; single];
                    build_activation_lut_into(part, cols, &mut want);
                    let mut got = vec![0x55i8; single];
                    unsafe { quintrit_core::simd::avx2::build_activation_lut_small_avx2(part, cols, &mut got) };
                    assert_eq!(got, want, "AVX2 small-part builder, cols={cols}");
                }
            }

            if is_x86_feature_detected!("avx512bw") {
                use quintrit_core::simd::avx512::{
                    build_activation_lut_avx512_compact, build_activation_lut_wide_avx512,
                    wide_activation_lut_len_avx512,
                };
                let mut want = build_activation_lut_avx512_compact(&lo, cols);
                want.extend_from_slice(&build_activation_lut_avx512_compact(&hi, cols));
                let got = build_activation_lut_wide_avx512(&acts, cols);
                assert_eq!(got.len(), wide_activation_lut_len_avx512(cols));
                assert_eq!(got, want, "AVX-512 compact wide builder, cols={cols}");
            }
        }
    }
}

/// The production entry points (single- and multi-threaded, whatever block grouping they use internally) against ground
/// truth, for row counts that leave remainders for every possible grouping of blocks.
mod production {
    use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn, gemv_lut_wide_parallel, transform_weights_lut};
    use quintrit_core::pack::pack_matrix;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn truth(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<f32> {
        super::announce_ground_truth();
        (0..rows)
            .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum::<i64>() as f32)
            .collect()
    }

    #[test]
    fn generic_wide_gemv_matches_ground_truth_for_many_row_counts() {
        let mut rng = Rng(0xA5A5_1234_5678_9ABD);
        for &(rows, cols) in &[(32usize, 161usize), (96, 480), (160, 800), (224, 3072), (352, 1280), (480, 161)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            let tiled = transform_weights_lut(&pack_matrix(&trits, rows, cols), rows, cols);
            let wide = build_activation_lut_wide(&acts, cols);
            let want = truth(&trits, &acts, rows, cols);
            let mut got = vec![0.0f32; rows];
            gemv_lut_wide_1xn(&tiled, &wide, rows, cols, &mut got);
            assert_eq!(got, want, "single thread {rows}x{cols}");
            for threads in [2usize, 3, 4] {
                let mut got = vec![0.0f32; rows];
                gemv_lut_wide_parallel(&tiled, &wide, rows, cols, &mut got, threads);
                assert_eq!(got, want, "{threads} threads {rows}x{cols}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_wide_gemv_matches_ground_truth_for_many_row_counts() {
        use quintrit_core::simd::avx512::{
            build_activation_lut_wide_avx512, gemv_lut_64_wide_1xn_avx512, gemv_lut_64_wide_parallel_avx512, transform_weights_lut_64,
        };
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")) {
            return;
        }
        let mut rng = Rng(0x5A5A_4321_8765_DCBA);
        for &(rows, cols) in &[(64usize, 161usize), (192, 480), (256, 800), (320, 3072), (448, 1280), (640, 161), (704, 4000)] {
            let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
            let acts: Vec<i8> = (0..cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
            let tiled = transform_weights_lut_64(&pack_matrix(&trits, rows, cols), rows, cols);
            let wide = build_activation_lut_wide_avx512(&acts, cols);
            let want = truth(&trits, &acts, rows, cols);
            let mut got = vec![0.0f32; rows];
            gemv_lut_64_wide_1xn_avx512(&tiled, &wide, rows, cols, &mut got);
            assert_eq!(got, want, "single thread {rows}x{cols}");
            for threads in [2usize, 3, 4] {
                let mut got = vec![0.0f32; rows];
                gemv_lut_64_wide_parallel_avx512(&tiled, &wide, rows, cols, &mut got, threads);
                assert_eq!(got, want, "{threads} threads {rows}x{cols}");
            }
        }
    }
}
