//! Per-algorithm tests for `quintrit-runtime`: numerical primitives, projection layers, model
//! configuration, KV cache, attention and the full decoder stack on a small synthetic model.
//!
//! Every SIMD implementation is checked against an independent scalar / f64 reference.

use quintrit_format::decoders::i2_s::pack_gguf_i2_s;
use quintrit_runtime::attention::AttentionBlock;
use quintrit_runtime::config::{ActivationType, ModelArchitecture, ModelConfig};
use quintrit_runtime::engine::{find_safe_flush_index, format_chat_prompt, ChatMessage};
use quintrit_runtime::kv_cache::KVCache;
use quintrit_runtime::layer::{
    dispatch_weight_from_i2_s, forward_linear_pair, forward_linear_triple, LinearLayer,
    TernaryKernelType,
};
use quintrit_runtime::model::{QuintritModel, TransformerBlock};
use quintrit_runtime::ops::*;
use quintrit_runtime::scratchpad::InferenceScratchpad;

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
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    fn trits(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.next() % 3) as i8 - 1).collect()
    }
}

// ----------------------------------------------------------------------------
// Numerical primitives
// ----------------------------------------------------------------------------

#[test]
fn rms_norm_implementations_agree_and_match_f64_reference() {
    let mut rng = Rng::new(1);
    for &n in &[8usize, 64, 100, 3072] {
        let x = rng.floats(n, 3.0);
        let w = rng.floats(n, 1.0).iter().map(|v| v + 1.5).collect::<Vec<_>>();
        let eps = 1e-6f32;

        let mut dispatched = vec![0.0f32; n];
        let mut scalar = vec![0.0f32; n];
        rms_norm(&x, &w, eps, &mut dispatched);
        rms_norm_scalar(&x, &w, eps, &mut scalar);
        assert_eq!(dispatched, scalar, "SIMD and scalar RMSNorm must be bit-identical (n={n})");

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut avx2 = vec![0.0f32; n];
            unsafe { rms_norm_avx2(&x, &w, eps, &mut avx2) };
            assert_eq!(avx2, scalar, "rms_norm_avx2 (n={n})");
        }

        let mean = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        let inv = 1.0 / (mean + eps as f64).sqrt();
        for i in 0..n {
            let expected = x[i] as f64 * inv * w[i] as f64;
            assert!((scalar[i] as f64 - expected).abs() <= 1e-5 * expected.abs().max(1e-3), "n={n} i={i}");
        }
    }
}

#[test]
fn rotary_embedding_matches_f64_reference_and_preserves_norm() {
    let (q_heads, kv_heads, head_dim) = (6usize, 2usize, 64usize);
    let theta_base = 10_000.0f32;
    let mut rng = Rng::new(2);
    let q0 = rng.floats(q_heads * head_dim, 2.0);
    let k0 = rng.floats(kv_heads * head_dim, 2.0);

    // Position 0 is the identity rotation.
    let (mut q, mut k) = (q0.clone(), k0.clone());
    apply_rotary_emb(&mut q, &mut k, 0, q_heads, kv_heads, head_dim, theta_base);
    assert_eq!(q, q0);
    assert_eq!(k, k0);

    for &pos in &[1usize, 7, 100, 513] {
        let (mut q, mut k) = (q0.clone(), k0.clone());
        apply_rotary_emb(&mut q, &mut k, pos, q_heads, kv_heads, head_dim, theta_base);

        let check = |orig: &[f32], rotated: &[f32], heads: usize| {
            for h in 0..heads {
                for i in 0..head_dim / 2 {
                    let angle = pos as f64 * (theta_base as f64).powf(-2.0 * i as f64 / head_dim as f64);
                    let (s, c) = angle.sin_cos();
                    let x0 = orig[h * head_dim + 2 * i] as f64;
                    let x1 = orig[h * head_dim + 2 * i + 1] as f64;
                    let e0 = x0 * c - x1 * s;
                    let e1 = x0 * s + x1 * c;
                    let g0 = rotated[h * head_dim + 2 * i] as f64;
                    let g1 = rotated[h * head_dim + 2 * i + 1] as f64;
                    // The angle recurrence runs in f32, so allow a small absolute error growing with pos.
                    let tol = 2e-4 + pos as f64 * 2e-6;
                    assert!((g0 - e0).abs() < tol && (g1 - e1).abs() < tol, "pos={pos} head={h} pair={i}");
                    assert!(((g0 * g0 + g1 * g1) - (x0 * x0 + x1 * x1)).abs() < 1e-4, "rotation must preserve norm");
                }
            }
        };
        check(&q0, &q, q_heads);
        check(&k0, &k, kv_heads);
    }
}

#[test]
fn quantizer_round_trips_within_one_code() {
    let mut rng = Rng::new(3);
    let x = rng.floats(512, 5.0);
    for &levels in &[127.0f32, 63.0] {
        let mut q = vec![0i8; x.len()];
        let s = quantize_activations_int8_with_range(&x, &mut q, levels);
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert_eq!(s, levels / amax);
        assert!(q.iter().any(|&v| v as f32 == levels || v as f32 == -levels), "extreme value must hit full scale");
        assert!(q.iter().all(|&v| (v as f32) <= levels && (v as f32) >= -(levels + 1.0)));
        for (xi, qi) in x.iter().zip(&q) {
            assert!((xi * s - *qi as f32).abs() <= 0.5 + 1e-4, "round-to-nearest");
        }
    }
}

#[test]
fn softmax_and_elementwise_helpers() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    softmax(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(x.windows(2).all(|w| w[0] < w[1]));
    let expected: Vec<f64> = {
        let e: Vec<f64> = [1.0f64, 2.0, 3.0, 4.0].iter().map(|v| (v - 4.0).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    };
    for (g, e) in x.iter().zip(&expected) {
        assert!((*g as f64 - e).abs() < 1e-6);
    }

    // Large logits must not overflow.
    let mut big = vec![1000.0f32, 1001.0, 999.0];
    softmax(&mut big);
    assert!(big.iter().all(|v| v.is_finite()) && (big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let mut empty: Vec<f32> = vec![];
    softmax(&mut empty);

    let a = [1.0f32, -2.0, 3.0];
    let b = [4.0f32, 5.0, -6.0];
    let mut out = [0.0f32; 3];
    elementwise_mul(&a, &b, &mut out);
    assert_eq!(out, [4.0, -10.0, -18.0]);

    let mut acc = [1.0f32, 2.0, 3.0];
    add_residual(&mut acc, &[0.5, -2.0, 10.0]);
    assert_eq!(acc, [1.5, 0.0, 13.0]);

    let mut r = [0.0f32; 4];
    relu2(&[-1.0, 0.0, 2.0, 3.0], &mut r);
    assert_eq!(r, [0.0, 0.0, 4.0, 9.0]);
}

#[test]
fn silu_and_swiglu_match_f64_reference() {
    let mut rng = Rng::new(4);
    for &n in &[7usize, 8, 33, 256] {
        let g = rng.floats(n, 8.0);
        let u = rng.floats(n, 2.0);
        let mut s = vec![0.0f32; n];
        let mut sw = vec![0.0f32; n];
        silu(&g, &mut s);
        swiglu(&g, &u, &mut sw);
        for i in 0..n {
            let x = g[i] as f64;
            let want = x / (1.0 + (-x).exp());
            assert!((s[i] as f64 - want).abs() <= 2e-5 * want.abs().max(1.0), "silu n={n} i={i}: {} vs {}", s[i], want);
            let want_sw = want * u[i] as f64;
            assert!((sw[i] as f64 - want_sw).abs() <= 2e-5 * want_sw.abs().max(1.0), "swiglu n={n} i={i}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn vectorised_exp_and_silu_match_libm() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        return;
    }
    use core::arch::x86_64::*;
    let xs: Vec<f32> = (-80..80).map(|i| i as f32 * 0.5).chain([0.0, 1e-3, -1e-3, 87.0, -87.0]).collect();
    for chunk in xs.chunks(8) {
        let mut buf = [0.0f32; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let (mut e, mut s) = ([0.0f32; 8], [0.0f32; 8]);
        unsafe {
            let v = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(e.as_mut_ptr(), ggml_v_expf_avx2(v));
            _mm256_storeu_ps(s.as_mut_ptr(), ggml_v_silu_avx2(v));
        }
        for i in 0..chunk.len() {
            let x = buf[i] as f64;
            let want = x.exp();
            if want.is_finite() && want > 1e-30 && want < 1e30 {
                assert!((e[i] as f64 - want).abs() <= 3e-6 * want, "expf({x}) = {} vs {}", e[i], want);
            }
            let want_silu = x / (1.0 + (-x).exp());
            assert!((s[i] as f64 - want_silu).abs() <= 3e-5 * want_silu.abs().max(1e-3), "silu({x})");
        }
    }
}

#[test]
fn f16_conversion_edge_cases_and_rounding() {
    for &(f, bits) in &[(0.0f32, 0x0000u16), (-0.0, 0x8000), (1.0, 0x3C00), (-2.0, 0xC000), (65504.0, 0x7BFF)] {
        assert_eq!(f32_to_f16(f), bits);
        assert_eq!(f16_to_f32(bits).to_bits(), f.to_bits());
    }
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    assert!(f16_to_f32(0x7E00).is_nan());
    assert_eq!(f32_to_f16(1e9), 0x7C00, "overflow rounds to infinity");
    assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24), "smallest subnormal");
    assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
    // Round-to-nearest-even at the midpoint between two halves: 1 + 2^-11 is the tie below 1 + 2^-10.
    assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3C00);
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3C02);
    // round_f16 is the composition of both conversions.
    for &v in &[0.1f32, -3.14159, 1234.567, 1e-5] {
        assert_eq!(round_f16(v), f16_to_f32(f32_to_f16(v)));
    }
    // Exhaustive: every finite half survives a trip through f32.
    for bits in 0u16..=0xFFFF {
        let f = f16_to_f32(bits);
        if f.is_finite() {
            assert_eq!(f32_to_f16(f), bits, "0x{bits:04x}");
        }
    }
}

#[test]
fn f16_vector_kernels_match_f64_reference() {
    let mut rng = Rng::new(5);
    for &n in &[16usize, 32, 64, 100, 256] {
        let a32 = rng.floats(n, 2.0);
        let b32 = rng.floats(n, 2.0);
        let a16: Vec<u16> = a32.iter().map(|&v| f32_to_f16(v)).collect();
        let b16: Vec<u16> = b32.iter().map(|&v| f32_to_f16(v)).collect();

        // f16 . f16
        let want: f64 = a16.iter().zip(&b16).map(|(&x, &y)| f16_to_f32(x) as f64 * f16_to_f32(y) as f64).sum();
        assert!((ggml_vec_dot_f16(&a16, &b16) as f64 - want).abs() < 1e-3, "ggml_vec_dot_f16 n={n}");

        // f32 . f16
        let want: f64 = a32.iter().zip(&b16).map(|(&x, &y)| x as f64 * f16_to_f32(y) as f64).sum();
        assert!((vec_dot_f32_f16(&a32, &b16) as f64 - want).abs() < 1e-3, "vec_dot_f32_f16 n={n}");

        // out += f16 * s
        let mut out = a32.clone();
        vec_mad_f32_f16(&mut out, &b16, 0.75);
        for i in 0..n {
            let want = a32[i] as f64 + f16_to_f32(b16[i]) as f64 * 0.75;
            assert!((out[i] as f64 - want).abs() < 1e-5, "vec_mad_f32_f16 n={n} i={i}");
        }

        // y = round_f16(y + x * v)
        let mut y = a16.clone();
        ggml_vec_mad_f16(&mut y, &b16, 0.3);
        for i in 0..n {
            let exact = f16_to_f32(a16[i]) as f64 + f16_to_f32(b16[i]) as f64 * 0.3f32 as f64;
            let ulp = (exact.abs() * 2f64.powi(-10)).max(2f64.powi(-24));
            assert!((f16_to_f32(y[i]) as f64 - exact).abs() <= ulp, "ggml_vec_mad_f16 n={n} i={i}");
        }

        // y = round_f16(y * v)
        let mut y = a16.clone();
        ggml_vec_scale_f16(&mut y, 0.5);
        for i in 0..n {
            assert_eq!(f16_to_f32(y[i]), f16_to_f32(a16[i]) * 0.5, "ggml_vec_scale_f16 is exact for powers of two");
        }
    }
}

#[test]
fn ggml_soft_max_returns_sum_of_shifted_exponentials() {
    let mut rng = Rng::new(6);
    for &n in &[3usize, 8, 21, 128] {
        let x0 = rng.floats(n, 5.0);
        let max = x0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut x = x0.clone();
        let sum = ggml_vec_soft_max_f32(&mut x, max);
        let mut want_sum = 0.0f64;
        for i in 0..n {
            let e = (x0[i] as f64 - max as f64).exp();
            want_sum += e;
            assert!((x[i] as f64 - e).abs() <= 3e-6 * e.max(1e-6), "n={n} i={i}");
        }
        assert!((sum as f64 - want_sum).abs() <= 1e-5 * want_sum, "n={n}");
    }
}

// ----------------------------------------------------------------------------
// Projection layers
// ----------------------------------------------------------------------------

#[allow(unused_mut)]
fn kernels_under_test() -> Vec<TernaryKernelType> {
    let mut kinds = vec![TernaryKernelType::QuintritLut];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi") {
            kinds.push(TernaryKernelType::QuintritLutAVX512);
        }
    }
    // The BitNet kernels (4 trits per byte) must agree with the quintrit kernels on every projection.
    #[cfg(feature = "bitnet-baseline")]
    {
        kinds.push(TernaryKernelType::BitnetMad);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                kinds.push(TernaryKernelType::BitnetMad512);
            }
        }
    }
    kinds
}

#[test]
fn kernel_type_metadata() {
    use TernaryKernelType::*;
    for (k, name, bpw, levels) in [
        (QuintritLut, "QuintritLut", 1.6, 127.0),
        (QuintritLutAVX512, "QuintritLutAVX512", 1.6, 127.0),
    ] {
        assert_eq!(k.as_str(), name);
        assert_eq!(k.bpw(), bpw);
        // The Quintrit LUT layers split activations into hi/lo parts and take the full int8 range.
        assert_eq!(k.activation_quant_levels(), levels, "{name}");
    }
}

#[cfg(feature = "bitnet-baseline")]
#[test]
fn bitnet_kernel_metadata() {
    use TernaryKernelType::*;
    for (k, name) in [(BitnetMad, "BitnetMad"), (BitnetMad512, "BitnetMad512")] {
        assert_eq!(k.as_str(), name);
        assert_eq!(k.bpw(), 2.0);
        assert_eq!(k.activation_quant_levels(), 127.0, "{name}");
        let json = serde_json::to_string(&k).unwrap();
        assert_eq!(serde_json::from_str::<TernaryKernelType>(&json).unwrap(), k);
    }
}

#[test]
fn kernel_type_serializes_under_its_variant_name() {
    for k in [TernaryKernelType::QuintritLut, TernaryKernelType::QuintritLutAVX512] {
        let json = serde_json::to_string(&k).unwrap();
        assert_eq!(json, format!("\"{}\"", k.as_str()));
        assert_eq!(serde_json::from_str::<TernaryKernelType>(&json).unwrap(), k);
    }
}

#[test]
fn projection_layers_match_exact_integer_reference() {
    let mut rng = Rng::new(7);
    // 64 rows keeps every layout valid (32-row and 64-row tiles); cols is a multiple of 128 and 160.
    let (rows, cols) = (64usize, 640usize);
    let trits = rng.trits(rows * cols);
    let gguf_planar = pack_gguf_i2_s(&trits, rows, cols);
    let weight_scale = 0.0371f32;
    let x = rng.floats(cols, 4.0);

    for kind in kernels_under_test() {
        let mut acts = vec![0i8; cols];
        let levels = kind.activation_quant_levels();
        let act_scale = quantize_activations_int8_with_range(&x, &mut acts, levels);

        // Reference: exact integer dot product, scaled once by weight_scale / act_scale.
        let post_scale = weight_scale / act_scale;
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                let dot: i64 = (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum();
                dot as f32 * post_scale
            })
            .collect();

        for threads in [1usize, 3] {
            let layer = dispatch_weight_from_i2_s(&gguf_planar, rows, cols, weight_scale, kind, threads);
            assert_eq!(layer.kernel_type(), kind);
            assert_eq!(layer.shape(), (rows, cols));
            assert_eq!(layer.weight_scale(), weight_scale);
            assert_eq!(layer.num_threads(), threads);
            assert!(layer.memory_bytes() >= rows * cols / 5);

            let mut out = vec![0.0f32; rows];
            layer.forward(&acts, act_scale, &mut out);
            assert_eq!(out, expected, "{kind:?} threads={threads}");

            // A zero activation scale (all-zero input) yields a zero output.
            let mut zero = vec![1.0f32; rows];
            layer.forward(&acts, 0.0, &mut zero);
            assert!(zero.iter().all(|&v| v == 0.0), "{kind:?}");

            // forward_with_lut must agree with forward, with and without a precomputed table.
            let qlut = matches!(kind, TernaryKernelType::QuintritLut).then(|| quintrit_core::lut::build_activation_lut_wide(&acts, cols));
            let mut with_lut = vec![0.0f32; rows];
            layer.forward_with_lut(&acts, act_scale, qlut.as_deref(), &mut with_lut);
            assert_eq!(with_lut, expected, "{kind:?} forward_with_lut threads={threads}");
        }
    }
}

#[test]
fn concurrent_pair_and_triple_projection_match_sequential() {
    let mut rng = Rng::new(8);
    let cols = 256usize;
    let x = rng.floats(cols, 3.0);
    let mut acts = vec![0i8; cols];
    let s = quantize_activations_int8(&x, &mut acts);

    let make = |rows: usize, seed: u64, threads: usize| {
        let mut r = Rng::new(seed);
        let trits = r.trits(rows * cols);
        dispatch_weight_from_i2_s(&pack_gguf_i2_s(&trits, rows, cols), rows, cols, 0.05, TernaryKernelType::QuintritLut, threads)
    };

    for threads in [1usize, 4] {
        let (a, b, c) = (make(64, 1, threads), make(32, 2, threads), make(32, 3, threads));
        let (mut ea, mut eb, mut ec) = (vec![0.0f32; 64], vec![0.0f32; 32], vec![0.0f32; 32]);
        a.forward(&acts, s, &mut ea);
        b.forward(&acts, s, &mut eb);
        c.forward(&acts, s, &mut ec);

        let (mut pa, mut pb) = (vec![0.0f32; 64], vec![0.0f32; 32]);
        forward_linear_pair(&a, &mut pa, &b, &mut pb, &acts, s, None, threads);
        assert_eq!((pa, pb), (ea.clone(), eb.clone()), "pair threads={threads}");

        let (mut ta, mut tb, mut tc) = (vec![0.0f32; 64], vec![0.0f32; 32], vec![0.0f32; 32]);
        forward_linear_triple(&a, &mut ta, &b, &mut tb, &c, &mut tc, &acts, s, None, threads);
        assert_eq!((ta, tb, tc), (ea, eb, ec), "triple threads={threads}");
    }
}

// ----------------------------------------------------------------------------
// Configuration, chat templates, small engine helpers
// ----------------------------------------------------------------------------

#[test]
fn model_config_defaults_and_derived_quantities() {
    for cfg in [ModelConfig::bitnet_2b_default(), ModelConfig::falcon3_1b_default(), ModelConfig::falcon3_7b_default()] {
        assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);
        assert_eq!(cfg.gqa_ratio(), cfg.num_attention_heads / cfg.num_key_value_heads);
        assert!(cfg.head_dim > 0 && cfg.vocab_size > 0 && cfg.hidden_size > 0);
        let q = cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim;
        let kv = 2 * cfg.hidden_size * cfg.num_key_value_heads * cfg.head_dim;
        let o = cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size;
        let ffn = 3 * cfg.hidden_size * cfg.intermediate_size;
        assert_eq!(cfg.block_parameter_count(), q + kv + o + ffn);
    }
    assert_eq!(ModelConfig::bitnet_2b_default().architecture, ModelArchitecture::BitNet);
    assert_eq!(ModelConfig::falcon3_7b_default().architecture, ModelArchitecture::Falcon3);

    let mut bad = ModelConfig::bitnet_2b_default();
    bad.num_key_value_heads = 7;
    assert!(std::panic::catch_unwind(|| bad.gqa_ratio()).is_err(), "non-divisible head counts must be rejected");
}

#[test]
fn chat_templates_per_architecture() {
    let msgs = vec![
        ChatMessage { role: "System".into(), content: "be brief".into(), name: None },
        ChatMessage { role: "user".into(), content: "hi".into(), name: None },
        ChatMessage { role: "assistant".into(), content: "hello".into(), name: None },
        ChatMessage { role: "tool".into(), content: "x".into(), name: None },
    ];
    assert_eq!(
        format_chat_prompt(&msgs[1..2], ModelArchitecture::Falcon3),
        "<|user|>\nhi\n<|assistant|>\n",
        "the standard evaluation prompt shape"
    );
    assert_eq!(
        format_chat_prompt(&msgs, ModelArchitecture::Falcon3),
        "<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\nhello<|endoftext|>\n<|user|>\nx\n<|assistant|>\n",
        "an assistant turn ends with the end-of-sequence token, as in the chat template of the model"
    );
    assert_eq!(
        format_chat_prompt(&msgs[1..3], ModelArchitecture::Falcon3),
        "<|user|>\nhi\n<|assistant|>\nhello<|endoftext|><|assistant|>\n",
        "a final assistant turn has no newline after the end-of-sequence token"
    );
    assert_eq!(
        format_chat_prompt(&msgs[..2], ModelArchitecture::BitNet),
        "### System:\nbe brief\n\n### User:\nhi\n\n### Assistant:\n"
    );
    assert_eq!(format_chat_prompt(&[], ModelArchitecture::Llama), "### Assistant:\n");
}

#[test]
fn stop_sequence_flush_boundary() {
    assert_eq!(find_safe_flush_index("hello", &["STOP"]), 5);
    assert_eq!(find_safe_flush_index("hello S", &["STOP"]), 6, "a partial stop prefix must be held back");
    assert_eq!(find_safe_flush_index("hello ST", &["STOP"]), 6);
    assert_eq!(find_safe_flush_index("abc", &[]), 3);
    assert_eq!(find_safe_flush_index("", &["x"]), 0);
    assert_eq!(find_safe_flush_index("añ<", &["<|e|>"]), "añ".len(), "multi-byte chars stay on boundaries");
}

#[test]
fn kv_cache_geometry_and_memory() {
    let mut cache = KVCache::new(3, 16, 2, 8);
    assert_eq!(cache.max_seq_len(), 16);
    assert_eq!(cache.memory_bytes(), 2 * 3 * 16 * 2 * 8 * 2, "K and V, 2 bytes per element");
    let k: Vec<u16> = (0..16).collect();
    let v: Vec<u16> = (100..116).collect();
    cache.append(2, 5, &k, &v);
    assert_eq!(cache.get_key(2, 5, 1), &k[8..16]);
    assert_eq!(cache.get_val(2, 5, 0), &v[0..8]);
    assert_eq!(cache.len(), 6);
    // Other layers and positions are untouched.
    assert!(cache.get_key(1, 5, 0).iter().all(|&x| x == 0));
}

#[test]
fn scratchpad_buffers_follow_model_dimensions() {
    let cfg = {
        let mut c = ModelConfig::falcon3_1b_default();
        c.hidden_size = 256;
        c.intermediate_size = 512;
        c.num_attention_heads = 4;
        c.num_key_value_heads = 2;
        c.head_dim = 64;
        c.vocab_size = 100;
        c.max_position_embeddings = 128;
        c
    };
    let s = InferenceScratchpad::new(&cfg, 10);
    assert_eq!(s.x.len(), 256);
    assert_eq!(s.gate.len(), 512);
    assert_eq!(s.down_int8.len(), 512);
    assert_eq!(s.q.len(), 4 * 64);
    assert_eq!(s.k.len(), 2 * 64);
    assert!(s.scores.len() >= 128);
    let d = InferenceScratchpad::from_dims(256, 512, 100, 4, 2, 64, 10);
    assert_eq!(d.x.len(), s.x.len());
    assert!(d.scores.len() >= 2048, "from_dims enforces a minimum sequence capacity");
}

// ----------------------------------------------------------------------------
// Attention and decoder stack on a small synthetic model
// ----------------------------------------------------------------------------

fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::falcon3_1b_default();
    c.architecture = ModelArchitecture::Falcon3;
    c.hidden_size = 256;
    c.intermediate_size = 512;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 64;
    c.vocab_size = 64;
    c.max_position_embeddings = 64;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 10_000.0;
    c.hidden_act = ActivationType::Silu;
    c.has_sub_norm = false;
    c
}

fn tiny_layer(rows: usize, cols: usize, seed: u64, threads: usize) -> (LinearLayer, Vec<i8>) {
    tiny_layer_with(TernaryKernelType::QuintritLut, rows, cols, seed, threads)
}

fn tiny_layer_with(kind: TernaryKernelType, rows: usize, cols: usize, seed: u64, threads: usize) -> (LinearLayer, Vec<i8>) {
    let trits = Rng::new(seed).trits(rows * cols);
    let layer = dispatch_weight_from_i2_s(&pack_gguf_i2_s(&trits, rows, cols), rows, cols, 0.02, kind, threads);
    (layer, trits)
}

/// Dot product `W x` with W in {-1,0,1}, scaled like the projection layers do.
fn project(trits: &[i8], rows: usize, cols: usize, acts: &[i8], act_scale: f32, weight_scale: f32) -> Vec<f64> {
    let post = (weight_scale / act_scale) as f64;
    (0..rows)
        .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum::<i64>() as f64 * post)
        .collect()
}

#[test]
fn attention_matches_f64_reference_over_several_positions() {
    let cfg = tiny_config();
    let (h, qh, kvh, hd) = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
    let (qd, kvd) = (qh * hd, kvh * hd);
    let ws = 0.02f32;

    let (q_l, q_t) = tiny_layer(qd, h, 11, 1);
    let (k_l, k_t) = tiny_layer(kvd, h, 12, 1);
    let (v_l, v_t) = tiny_layer(kvd, h, 13, 1);
    let (o_l, o_t) = tiny_layer(h, qd, 14, 1);
    let attn = AttentionBlock::new(q_l, k_l, v_l, o_l, None, &cfg);
    assert!(attn.memory_bytes() > 0);
    assert_eq!(attn.q_proj().shape(), (qd, h));
    assert_eq!(attn.k_proj().shape(), (kvd, h));
    assert_eq!(attn.v_proj().shape(), (kvd, h));
    assert_eq!(attn.o_proj().shape(), (h, qd));

    let mut cache = KVCache::new(1, 32, kvh, hd);
    let mut scratch = InferenceScratchpad::new(&cfg, 32);
    let mut rng = Rng::new(15);

    // f64 reference state: rotated keys and values per position.
    let mut ref_k: Vec<Vec<f64>> = Vec::new();
    let mut ref_v: Vec<Vec<f64>> = Vec::new();

    for pos in 0..6usize {
        let x = rng.floats(h, 2.0);
        scratch.x_norm[..h].copy_from_slice(&x);
        attn.forward_with_scratchpad(0, pos, &mut cache, &mut scratch);
        let got: Vec<f32> = scratch.attn_out[..h].to_vec();

        // --- f64 reference ---
        let mut acts = vec![0i8; h];
        let s = quantize_activations_int8(&x, &mut acts);
        let q = project(&q_t, qd, h, &acts, s, ws);
        let k = project(&k_t, kvd, h, &acts, s, ws);
        let v = project(&v_t, kvd, h, &acts, s, ws);
        let rope = |vec: &[f64], heads: usize| -> Vec<f64> {
            let mut out = vec.to_vec();
            for hh in 0..heads {
                for i in 0..hd / 2 {
                    let angle = pos as f64 * (cfg.rope_theta as f64).powf(-2.0 * i as f64 / hd as f64);
                    let (sn, cs) = angle.sin_cos();
                    let (a, b) = (vec[hh * hd + 2 * i], vec[hh * hd + 2 * i + 1]);
                    out[hh * hd + 2 * i] = a * cs - b * sn;
                    out[hh * hd + 2 * i + 1] = a * sn + b * cs;
                }
            }
            out
        };
        ref_k.push(rope(&k, kvh));
        ref_v.push(v);
        let q = rope(&q, qh);

        let mut ctx = vec![0.0f64; qd];
        for head in 0..qh {
            let kv_head = head / cfg.gqa_ratio();
            let scores: Vec<f64> = (0..=pos)
                .map(|p| {
                    (0..hd).map(|d| q[head * hd + d] * ref_k[p][kv_head * hd + d]).sum::<f64>() / (hd as f64).sqrt()
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..hd {
                ctx[head * hd + d] = (0..=pos).map(|p| e[p] * ref_v[p][kv_head * hd + d]).sum::<f64>() / z;
            }
        }
        let ctx32: Vec<f32> = ctx.iter().map(|&v| v as f32).collect();
        let mut ctx_acts = vec![0i8; qd];
        let cs = quantize_activations_int8(&ctx32, &mut ctx_acts);
        let want = project(&o_t, h, qd, &ctx_acts, cs, ws);

        // Quantization of the context may flip individual int8 codes, so compare relative to the
        // output magnitude instead of per element exactness.
        let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-6);
        let max_err = got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
        assert!(max_err <= 0.03 * scale, "pos={pos}: max error {max_err} vs output scale {scale}");
    }
    assert_eq!(cache.len(), 6);
}

fn tiny_model(threads: usize) -> QuintritModel {
    tiny_model_with(TernaryKernelType::QuintritLut, threads)
}

fn tiny_model_with(kind: TernaryKernelType, threads: usize) -> QuintritModel {
    let cfg = tiny_config();
    let (h, i, qd, kvd) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim);
    let mut rng = Rng::new(21);
    let norm = |rng: &mut Rng, n: usize| rng.floats(n, 0.2).iter().map(|v| v + 1.0).collect::<Vec<f32>>();

    let blocks = (0..cfg.num_hidden_layers)
        .map(|l| {
            let s = 100 * (l as u64 + 1);
            let attn = AttentionBlock::new(
                tiny_layer_with(kind, qd, h, s + 1, threads).0,
                tiny_layer_with(kind, kvd, h, s + 2, threads).0,
                tiny_layer_with(kind, kvd, h, s + 3, threads).0,
                tiny_layer_with(kind, h, qd, s + 4, threads).0,
                None,
                &cfg,
            );
            TransformerBlock {
                attn_norm: norm(&mut rng, h),
                attn,
                ffn_norm: norm(&mut rng, h),
                gate_proj: tiny_layer_with(kind, i, h, s + 5, threads).0,
                up_proj: tiny_layer_with(kind, i, h, s + 6, threads).0,
                ffn_sub_norm: None,
                down_proj: tiny_layer_with(kind, h, i, s + 7, threads).0,
                hidden_act: ActivationType::Silu,
                rms_norm_eps: cfg.rms_norm_eps,
                hidden_size: h,
                intermediate_size: i,
            }
        })
        .collect();

    let embeddings: Vec<u16> = rng.floats(cfg.vocab_size * h, 0.5).iter().map(|&v| f32_to_f16(v)).collect();
    QuintritModel {
        output_norm: norm(&mut rng, h),
        lm_head: None,
        lm_head_f16: None,
        token_embeddings: embeddings,
        blocks,
        config: cfg,
    }
}

fn run_sequence(model: &QuintritModel, tokens: &[u32]) -> Vec<Vec<f32>> {
    let cfg = &model.config;
    let mut cache = KVCache::new(cfg.num_hidden_layers, 32, cfg.num_key_value_heads, cfg.head_dim);
    let mut scratch = InferenceScratchpad::new(cfg, 32);
    let mut all = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let mut logits = vec![0.0f32; cfg.vocab_size];
        model.forward_token_with_scratchpad(t, pos, &mut cache, &mut scratch, &mut logits);
        all.push(logits);
    }
    all
}

#[test]
fn decoder_stack_is_deterministic_thread_independent_and_input_sensitive() {
    let tokens = [3u32, 17, 42, 5, 9];
    let single = tiny_model(1);
    let reference = run_sequence(&single, &tokens);
    assert!(reference.iter().flatten().all(|v| v.is_finite()));

    // Identical on a second run and with a different thread count (and set_num_threads on a built model).
    assert_eq!(run_sequence(&single, &tokens), reference, "model must be deterministic");
    assert_eq!(run_sequence(&tiny_model(3), &tokens), reference, "thread count must not change the result");
    let mut resized = tiny_model(1);
    resized.set_num_threads(4);
    assert_eq!(run_sequence(&resized, &tokens), reference, "set_num_threads must not change the result");

    // The allocating convenience path matches the scratchpad path.
    let cfg = &single.config;
    let mut cache = KVCache::new(cfg.num_hidden_layers, 32, cfg.num_key_value_heads, cfg.head_dim);
    let mut logits = vec![0.0f32; cfg.vocab_size];
    single.forward_token(tokens[0], 0, &mut cache, &mut logits);
    assert_eq!(logits, reference[0]);

    // Different tokens and different histories give different logits.
    assert_ne!(reference[0], reference[1]);
    let other = run_sequence(&single, &[tokens[0], 11]);
    assert_eq!(other[0], reference[0], "the first position depends only on the first token");
    assert_ne!(other[1], reference[1]);
}

#[test]
fn transformer_block_forward_matches_scratchpad_path() {
    let model = tiny_model(1);
    let cfg = &model.config;
    let mut rng = Rng::new(31);
    let x0 = rng.floats(cfg.hidden_size, 1.0);

    let mut cache_a = KVCache::new(1, 16, cfg.num_key_value_heads, cfg.head_dim);
    let mut x_a = x0.clone();
    model.blocks[0].forward(0, &mut x_a, 0, &mut cache_a);

    let mut cache_b = KVCache::new(1, 16, cfg.num_key_value_heads, cfg.head_dim);
    let mut scratch = InferenceScratchpad::new(cfg, 16);
    scratch.x[..cfg.hidden_size].copy_from_slice(&x0);
    model.blocks[0].forward_with_scratchpad(0, 0, &mut cache_b, &mut scratch);

    assert_eq!(x_a, scratch.x[..cfg.hidden_size].to_vec());
    assert_ne!(x_a, x0, "the block must change the residual stream");
    assert!(x_a.iter().all(|v| v.is_finite()));
}

// ----------------------------------------------------------------------------
// ModelConfig::from_gguf (hyperparameters, architecture detection, BOS resolution)
// ----------------------------------------------------------------------------

enum MetaValue {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
}

fn gguf_with_metadata(entries: &[(&str, MetaValue)]) -> std::path::PathBuf {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, value) in entries {
        put_str(&mut out, key);
        match value {
            MetaValue::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            MetaValue::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            MetaValue::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(*v as u8);
            }
            MetaValue::Str(v) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, v);
            }
        }
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("quintrit_cfg_{}_{}.gguf", std::process::id(), id));
    std::fs::write(&path, out).unwrap();
    path
}

fn config_from(entries: &[(&str, MetaValue)]) -> ModelConfig {
    let path = gguf_with_metadata(entries);
    let gguf = quintrit_format::GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    ModelConfig::from_gguf(&gguf).unwrap()
}

/// Zero, absurd or inconsistent hyper-parameters are an error of `from_gguf`; they used to cause a division by zero,
/// a remainder by zero or an allocation that aborts the process.
#[test]
fn from_gguf_rejects_absurd_hyperparameters() {
    let base = |key: &'static str, value: u32| {
        let mut entries = vec![
            ("general.architecture", MetaValue::Str("llama")),
            ("llama.vocab_size", MetaValue::U32(1000)),
            ("llama.embedding_length", MetaValue::U32(256)),
            ("llama.feed_forward_length", MetaValue::U32(512)),
            ("llama.block_count", MetaValue::U32(2)),
            ("llama.attention.head_count", MetaValue::U32(4)),
            ("llama.attention.head_count_kv", MetaValue::U32(2)),
            ("llama.context_length", MetaValue::U32(64)),
        ];
        for e in entries.iter_mut() {
            if e.0 == key {
                e.1 = MetaValue::U32(value);
            }
        }
        let path = gguf_with_metadata(&entries);
        let gguf = quintrit_format::GgufFile::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        ModelConfig::from_gguf(&gguf)
    };
    assert!(base("llama.block_count", 2).is_ok());
    for (key, value, needle) in [
        ("llama.attention.head_count", 0, "head_count"),
        ("llama.attention.head_count_kv", 0, "head_count_kv"),
        ("llama.embedding_length", 0, "embedding_length"),
        ("llama.vocab_size", 0, "vocab_size"),
        ("llama.context_length", 0, "context_length"),
        ("llama.context_length", 1 << 30, "context_length"),
        ("llama.block_count", 1 << 20, "block_count"),
        ("llama.attention.head_count", 3, "divisible"),  // 256 / 3
        ("llama.attention.head_count", 256, "even head"), // head dimension 1
        ("llama.attention.head_count_kv", 3, "multiple of head_count_kv"),
    ] {
        let err = base(key, value).err().unwrap_or_else(|| panic!("{key} = {value} was accepted"));
        assert!(err.contains(needle), "{key} = {value}: {err:?} does not mention {needle:?}");
    }
    // Every value inside its range, but a KV cache of 4096 layers x 2^24 positions x 4096 heads is absurd.
    let path = gguf_with_metadata(&[
        ("general.architecture", MetaValue::Str("llama")),
        ("llama.embedding_length", MetaValue::U32(1 << 20)),
        ("llama.block_count", MetaValue::U32(4096)),
        ("llama.attention.head_count", MetaValue::U32(4096)),
        ("llama.attention.head_count_kv", MetaValue::U32(4096)),
        ("llama.context_length", MetaValue::U32(1 << 24)),
    ]);
    let gguf = quintrit_format::GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let err = ModelConfig::from_gguf(&gguf).unwrap_err();
    assert!(err.contains("KV cache"), "{err}");
}

#[test]
fn from_gguf_reads_hyperparameters_and_detects_architecture() {
    let cfg = config_from(&[
        ("general.architecture", MetaValue::Str("llama")),
        ("general.name", MetaValue::Str("Falcon3-10B-Instruct")),
        ("llama.vocab_size", MetaValue::U32(131072)),
        ("llama.embedding_length", MetaValue::U32(3072)),
        ("llama.feed_forward_length", MetaValue::U32(23040)),
        ("llama.block_count", MetaValue::U32(40)),
        ("llama.attention.head_count", MetaValue::U32(12)),
        ("llama.attention.head_count_kv", MetaValue::U32(4)),
        ("llama.attention.layer_norm_rms_epsilon", MetaValue::F32(1e-6)),
        ("llama.rope.freq_base", MetaValue::F32(1_000_042.0)),
        ("llama.context_length", MetaValue::U32(32768)),
    ]);
    assert_eq!(cfg.architecture, ModelArchitecture::Falcon3, "detected from general.name");
    assert_eq!((cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size), (131072, 3072, 23040));
    assert_eq!((cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads), (40, 12, 4));
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 1_000_042.0);
    assert_eq!(cfg.max_position_embeddings, 32768);
    assert_eq!(cfg.hidden_act, ActivationType::Silu);
    assert!(!cfg.has_sub_norm);

    let bitnet = config_from(&[("general.architecture", MetaValue::Str("bitnet-b1.58"))]);
    assert_eq!(bitnet.architecture, ModelArchitecture::BitNet);
    assert_eq!(bitnet.hidden_act, ActivationType::Relu2);
    assert!(bitnet.has_sub_norm && bitnet.tie_word_embeddings);

    let plain = config_from(&[("general.architecture", MetaValue::Str("llama"))]);
    assert_eq!(plain.architecture, ModelArchitecture::Llama);
}

#[test]
fn from_gguf_resolves_bos_like_the_reference_vocabulary_loader() {
    let bos = |entries: &[(&str, MetaValue)]| config_from(entries).bos_token_id;

    // Falcon3 style: GPT-2 BPE with a llama3-family pre-tokenizer and no explicit BOS id.
    // The reference loader prepends <|endoftext|> (id 11) here; omitting it shifts every position.
    assert_eq!(
        bos(&[
            ("general.architecture", MetaValue::Str("llama")),
            ("tokenizer.ggml.model", MetaValue::Str("gpt2")),
            ("tokenizer.ggml.pre", MetaValue::Str("falcon3")),
        ]),
        Some(11)
    );
    // An explicit id wins over the default.
    assert_eq!(
        bos(&[
            ("tokenizer.ggml.model", MetaValue::Str("gpt2")),
            ("tokenizer.ggml.pre", MetaValue::Str("llama-bpe")),
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(128000)),
        ]),
        Some(128000)
    );
    // `add_bos_token = false` disables it even when an id exists.
    assert_eq!(
        bos(&[
            ("tokenizer.ggml.model", MetaValue::Str("gpt2")),
            ("tokenizer.ggml.pre", MetaValue::Str("falcon3")),
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(5)),
            ("tokenizer.ggml.add_bos_token", MetaValue::Bool(false)),
        ]),
        None
    );
    // `add_bos_token = true` enables it for a pre-tokenizer that would not add one by default.
    assert_eq!(
        bos(&[
            ("tokenizer.ggml.model", MetaValue::Str("gpt2")),
            ("tokenizer.ggml.pre", MetaValue::Str("default")),
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(7)),
            ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
        ]),
        Some(7)
    );
    // Without a llama3-family pre-tokenizer and without an override there is no BOS.
    assert_eq!(
        bos(&[
            ("tokenizer.ggml.model", MetaValue::Str("gpt2")),
            ("tokenizer.ggml.pre", MetaValue::Str("default")),
            ("tokenizer.ggml.bos_token_id", MetaValue::U32(7)),
        ]),
        None
    );
}

#[test]
fn quintrit_lut_kernels_reproduce_each_other_bit_for_bit_at_full_activation_range() {
    // All kernels quantize with 127 levels and compute the exact integer dot product (verified against an i64
    // ground truth in `projection_layers_match_exact_integer_reference`), so the whole decoder stack must give
    // identical logits whichever of them executes the projections. With the feature `bitnet-baseline` this
    // includes the 4-trit BitNet kernels.
    let tokens = [3u32, 17, 42, 5, 9, 31];
    let reference = run_sequence(&tiny_model_with(TernaryKernelType::QuintritLut, 1), &tokens);
    let mut kinds = vec![TernaryKernelType::QuintritLut];
    kinds.extend(kernels_under_test().into_iter().filter(|k| *k != TernaryKernelType::QuintritLut));
    for kind in kinds {
        for threads in [1usize, 3] {
            assert_eq!(
                run_sequence(&tiny_model_with(kind, threads), &tokens),
                reference,
                "{kind:?} threads={threads} must match the 32-row Quintrit kernel exactly"
            );
        }
    }
}
