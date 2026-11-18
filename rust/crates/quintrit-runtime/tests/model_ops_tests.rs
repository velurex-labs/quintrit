//! Ground-truth tests of the numerical primitives used by the llama.cpp-format models (`Q2_0`, Qwen3.5, Nemotron-H):
//! the dot products, BF16 rounding, `Q8_0` quantization, the Walsh-Hadamard transform, RoPE variants and the
//! reference sort. The bit-level agreement with llama.cpp itself is checked by parity suites outside the SDK; these
//! tests check the mathematics against `f64` or brute force.

use quintrit_runtime::config::{RopeConfig, RopeMode};
use quintrit_runtime::ops::*;
use quintrit_runtime::refsort::{libstdcxx, msvc};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn dot64(x: &[f32], y: &[f32]) -> f64 {
    x.iter().zip(y).map(|(&a, &b)| a as f64 * b as f64).sum()
}

#[test]
fn dot_products_agree_with_f64() {
    let mut r = Rng(1);
    for n in [1usize, 7, 8, 31, 32, 33, 64, 100, 128, 2048] {
        let x: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let y: Vec<f32> = (0..n).map(|_| r.next()).collect();
        let tol = 1e-5 * (n as f64).sqrt() + 1e-6;
        assert!((ggml_vec_dot_f32(&x, &y) as f64 - dot64(&x, &y)).abs() < tol, "f32 n={n}");
        let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
        let xr: Vec<f32> = xh.iter().map(|&h| f16_to_f32(h)).collect();
        assert!((ggml_vec_dot_f16_f32(&xh, &y) as f64 - dot64(&xr, &y)).abs() < tol, "f16 x f32 n={n}");
        let xb: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
        let yb: Vec<u16> = y.iter().map(|&v| f32_to_bf16(v)).collect();
        let xbr: Vec<f32> = xb.iter().map(|&h| bf16_to_f32(h)).collect();
        let ybr: Vec<f32> = yb.iter().map(|&h| bf16_to_f32(h)).collect();
        assert!((ggml_vec_dot_bf16(&xb, &yb) as f64 - dot64(&xbr, &ybr)).abs() < tol, "bf16 n={n}");
    }
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3F80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80, "tie to even (down)");
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82, "tie to even (up)");
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

#[test]
fn q8_0_quantization_is_within_half_a_step() {
    let mut r = Rng(2);
    let x: Vec<f32> = (0..256).map(|i| r.next() * (1 + i % 7) as f32).collect();
    let mut qs = vec![0i8; 256];
    let mut d = vec![0.0f32; 8];
    quantize_q8_0(&x, &mut qs, &mut d);
    for b in 0..8 {
        let amax = x[b * 32..(b + 1) * 32].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((d[b] - amax / 127.0).abs() <= amax / 127.0 * 1e-3, "scale of block {b}");
        assert!(qs[b * 32..(b + 1) * 32].iter().any(|&q| q.unsigned_abs() == 127), "the largest value maps to 127");
        for i in b * 32..(b + 1) * 32 {
            assert!((qs[i] as f32 * d[b] - x[i]).abs() <= 0.5 * d[b] + amax * 2e-3, "value {i}");
        }
    }
    let zeros = vec![0.0f32; 32];
    let (mut qz, mut dz) = (vec![1i8; 32], vec![1.0f32; 1]);
    quantize_q8_0(&zeros, &mut qz, &mut dz);
    assert!(qz.iter().all(|&q| q == 0) && dz[0] == 0.0);
}

#[test]
fn walsh_hadamard_matches_the_matrix_and_is_an_involution() {
    let mut r = Rng(3);
    for block in [1usize, 2, 8, 64, 512] {
        let x: Vec<f32> = (0..2 * block).map(|_| r.next()).collect();
        let mut y = x.clone();
        fwht_blocks(&mut y, block);
        let s = 1.0 / (block as f64).sqrt();
        for b in 0..2 {
            for row in 0..block {
                let want: f64 = (0..block)
                    .map(|col| {
                        let sign = if (row & col).count_ones() % 2 == 1 { -1.0 } else { 1.0 };
                        sign * s * x[b * block + col] as f64
                    })
                    .sum();
                assert!((y[b * block + row] as f64 - want).abs() < 1e-5, "block {block} row {row}");
            }
        }
        fwht_blocks(&mut y, block);
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-5, "H H x = x (block {block})");
        }
    }
}

#[test]
fn rope_variants_rotate_the_right_pairs() {
    let mut r = Rng(4);
    let hd = 16;
    for mode in [RopeMode::Normal, RopeMode::Neox] {
        for n_dims in [16usize, 8] {
            let rope = RopeConfig::plain(mode, n_dims, 10000.0, 4096);
            let q0: Vec<f32> = (0..hd).map(|_| r.next()).collect();
            let (mut q, mut k) = (q0.clone(), q0.clone());
            apply_rope(&mut q, &mut k, 0, 1, 1, hd, &rope);
            assert_eq!(q, q0, "position 0 is the identity");
            apply_rope(&mut q, &mut k, 5, 1, 1, hd, &rope);
            for i in 0..n_dims / 2 {
                let (a, b) = match mode {
                    RopeMode::Normal => (2 * i, 2 * i + 1),
                    RopeMode::Neox => (i, i + n_dims / 2),
                };
                let theta = 5.0f64 * 10000f64.powf(-2.0 * i as f64 / n_dims as f64);
                let (c, s) = (theta.cos(), theta.sin());
                let wa = q0[a] as f64 * c - q0[b] as f64 * s;
                let wb = q0[a] as f64 * s + q0[b] as f64 * c;
                assert!((q[a] as f64 - wa).abs() < 1e-4 && (q[b] as f64 - wb).abs() < 1e-4, "{mode:?} pair {i}");
            }
            assert_eq!(&q[n_dims..], &q0[n_dims..], "coordinates beyond n_dims pass through");
        }
    }
    // YaRN keeps a rotation, scaled by attn_factor * (1 + 0.1 ln(1 / freq_scale)).
    let mut yarn = RopeConfig::plain(RopeMode::Neox, hd, 1e6, 8192);
    yarn.freq_scale = 0.25;
    yarn.ext_factor = 1.0;
    let mut cache = Vec::new();
    rope_cache(7, &yarn, &mut cache);
    let m = 1.0 + 0.1 * 4f64.ln();
    for p in cache.chunks(2) {
        assert!(((p[0] as f64).hypot(p[1] as f64) - m).abs() < 1e-5);
    }
}

#[test]
fn activations_match_their_definitions() {
    for x in [-30.0f32, -2.0, -0.5, 0.0, 0.5, 3.0, 19.0, 25.0] {
        assert!((sigmoid_f32(x) as f64 - 1.0 / (1.0 + (-x as f64).exp())).abs() < 1e-6);
        let sp = if x > 20.0 { x as f64 } else { (1.0 + (x as f64).exp()).ln() };
        assert!((softplus_f32(x) as f64 - sp).abs() < 1e-5 * sp.max(1.0));
    }
}

#[test]
fn reference_sorts_sort() {
    let vals: Vec<f32> = (0..128).map(|i| ((i * 37) % 11) as f32).collect();
    let less = |a: usize, b: usize| vals[a] > vals[b];
    for f in [msvc::sort::<usize>, libstdcxx::sort::<usize>] {
        let mut idx: Vec<usize> = (0..128).collect();
        f(&mut idx, &less);
        assert!(idx.windows(2).all(|w| vals[w[0]] >= vals[w[1]]));
    }
}

/// The SIMD paths (NEON on AArch64) must give the bits of the portable definitions for every length, including the
/// tails after the last full vector group; on other targets the dispatching functions are the definitions themselves.
#[test]
fn simd_dot_products_equal_their_definitions() {
    let mut r = Rng(99);
    for n in [0usize, 1, 3, 4, 7, 8, 9, 15, 16, 31, 32, 33, 40, 63, 64, 65, 100, 127, 128, 129, 256, 1000, 2048, 2051] {
        for scale in [1.0f32, 37.0, 1e-3] {
            let x: Vec<f32> = (0..n).map(|_| r.next() * scale).collect();
            let y: Vec<f32> = (0..n).map(|_| r.next()).collect();
            let xh: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
            let xb: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
            let yb: Vec<u16> = y.iter().map(|&v| f32_to_bf16(v)).collect();
            assert_eq!(ggml_vec_dot_f32(&x, &y).to_bits(), ggml_vec_dot_f32_scalar(&x, &y).to_bits(), "f32 n={n}");
            assert_eq!(ggml_vec_dot_bf16(&xb, &yb).to_bits(), ggml_vec_dot_bf16_scalar(&xb, &yb).to_bits(), "bf16 n={n}");
            assert_eq!(
                ggml_vec_dot_f16_f32(&xh, &y).to_bits(),
                ggml_vec_dot_f16_f32_scalar(&xh, &y).to_bits(),
                "f16_f32 n={n}"
            );
            let s = r.next() * 3.0;
            let mut acc = y.clone();
            ggml_vec_mad_f16_f32(&mut acc, &xh, s);
            for k in 0..n {
                assert_eq!(acc[k].to_bits(), f16_to_f32(xh[k]).mul_add(s, y[k]).to_bits(), "mad n={n} k={k}");
            }
        }
    }
}

/// The SIMD paths of `Q8_0` quantization and of the Walsh-Hadamard transform (AVX-512 on x86-64) must give the bits
/// of their definitions, including blocks with zeros, NaN, infinities and subnormal maxima.
#[test]
fn simd_q8_0_quantization_and_fwht_equal_their_definitions() {
    let mut r = Rng(7);
    let blocks = 67;
    for scale in [1.0f32, 3.7e2, 1e-3, 1e-38, 1e-44] {
        let mut x: Vec<f32> = (0..32 * blocks).map(|_| r.next() * scale).collect();
        x[32..64].fill(0.0);
        x[64 + 5] = f32::NAN;
        x[96 + 1] = f32::INFINITY;
        x[128 + 30] = f32::NEG_INFINITY;
        x[160..192].fill(f32::NAN);
        x[192 + 3] = -0.0;
        // Exact ties of `x * id`: amax 127 gives id 1.
        for (k, v) in x[224..256].iter_mut().enumerate() {
            *v = if k == 0 { 127.0 } else { k as f32 - 15.5 };
        }
        let (mut q1, mut d1) = (vec![0i8; x.len()], vec![0.0f32; blocks]);
        let (mut q2, mut d2) = (vec![1i8; x.len()], vec![1.0f32; blocks]);
        quantize_q8_0(&x, &mut q1, &mut d1);
        quantize_q8_0_scalar(&x, &mut q2, &mut d2);
        assert_eq!(q1, q2, "codes, scale {scale}");
        assert!(d1.iter().zip(&d2).all(|(a, b)| a.to_bits() == b.to_bits()), "block scales, scale {scale}");
    }
    for block in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
        let x: Vec<f32> = (0..3 * block).map(|_| r.next() * 5.0).collect();
        let (mut a, mut b) = (x.clone(), x);
        fwht_blocks(&mut a, block);
        fwht_blocks_scalar(&mut b, block);
        assert!(a.iter().zip(&b).all(|(u, v)| u.to_bits() == v.to_bits()), "fwht block {block}");
    }
}

#[test]
fn simd_q8_0_dot_equals_its_definition() {
    use quintrit_runtime::hybrid::{dot_q8_0, dot_q8_0_scalar};
    let mut r = Rng(5);
    for blocks in [1usize, 2, 3, 7, 84, 129] {
        let n = 32 * blocks;
        let code = |r: &mut Rng| ((r.next() * 128.0) as i32).clamp(-128, 127) as i8;
        let w: Vec<i8> = (0..n).map(|_| code(&mut r)).collect();
        let a: Vec<i8> = (0..n).map(|_| code(&mut r)).collect();
        let wd: Vec<f32> = (0..blocks).map(|_| r.next() * 0.01).collect();
        let ad: Vec<f32> = (0..blocks).map(|_| r.next() * 0.1).collect();
        assert_eq!(dot_q8_0(&w, &wd, &a, &ad).to_bits(), dot_q8_0_scalar(&w, &wd, &a, &ad).to_bits(), "{blocks} blocks");
        // Extreme codes.
        let w: Vec<i8> = (0..n).map(|k| if k % 3 == 0 { -128 } else { 127 }).collect();
        assert_eq!(dot_q8_0(&w, &wd, &w, &ad).to_bits(), dot_q8_0_scalar(&w, &wd, &w, &ad).to_bits(), "extreme {blocks}");
    }
}
