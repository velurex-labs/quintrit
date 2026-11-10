//! # Activation-table builder timing
//!
//! Times the construction of the wide (`lo | hi`) activation tables of the velurex kernels for the activation
//! lengths of Falcon3-10B (K = 3072 and K = 23040). The engine builds one table per projection input, so the
//! builder cost is paid on top of every GEMV that does not share its input.
//!
//! ```bash
//! cargo bench -p quintrit-core --bench table_build
//! ```

use quintrit_core::lut::{build_activation_lut_wide_into, wide_activation_lut_len};
use std::time::Instant;

fn main() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &cols in &[3072usize, 23040] {
        let acts: Vec<i8> = (0..cols)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 255) as i64 as i8
            })
            .collect();
        let mut scratch = vec![0i8; 2 * cols];
        let mut qlut = vec![0i8; wide_activation_lut_len(cols)];
        let mut best = f64::MAX;
        for _ in 0..30 {
            let t0 = Instant::now();
            for _ in 0..20 {
                build_activation_lut_wide_into(&acts, cols, &mut scratch, &mut qlut);
                std::hint::black_box(&qlut);
            }
            best = best.min(t0.elapsed().as_secs_f64() / 20.0);
        }
        println!("generic wide table, K = {cols}: {:.1} us per build ({:.1} ns per column byte)", best * 1e6, best * 1e9 / (cols.div_ceil(160) * 32) as f64);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512bw") {
            let mut best = f64::MAX;
            for _ in 0..30 {
                let t0 = Instant::now();
                for _ in 0..20 {
                    let t = quintrit_core::simd::avx512::build_activation_lut_wide_avx512(&acts, cols);
                    std::hint::black_box(&t);
                }
                best = best.min(t0.elapsed().as_secs_f64() / 20.0);
            }
            println!("AVX-512 compact wide table, K = {cols}: {:.1} us per build", best * 1e6);
        }
    }
}
