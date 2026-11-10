//! High-performance mathematical and tensor primitives for transformer inference.
//!
//! Includes AVX2 vector optimizations for RMSNorm, RoPE rotary embeddings,
//! Int8 activation quantization, and half-precision F16 operations.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ============================================================================
// RMSNorm (Root Mean Square Layer Normalization)
// ============================================================================

/// Evaluates Root Mean Square Layer Normalization:
/// $$y_i = \frac{x_i}{\sqrt{\frac{1}{N} \sum_{j=0}^{N-1} x_j^2 + \epsilon}} \times w_i$$
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                rms_norm_avx2(x, weight, eps, out);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { rms_norm_neon(x, weight, eps, out) };
        return;
    }
    #[allow(unreachable_code)]
    rms_norm_scalar(x, weight, eps, out);
}

/// Scalar fallback implementation of RMSNorm using double-precision accumulation.
pub fn rms_norm_scalar(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let n = x.len();
    let mut sum: f64 = 0.0;
    for i in 0..n {
        sum += (x[i] * x[i]) as f64;
    }
    let mean = (sum / (n as f64)) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();

    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// AVX2-accelerated RMSNorm with FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;

    let x_ptr = x.as_ptr();
    let w_ptr = weight.as_ptr();
    let out_ptr = out.as_mut_ptr();

    unsafe {
        // 1. Sum of squares exactly as the reference does it: each square is rounded to f32
        // first, then accumulated sequentially in f64. Any other order or an f64 square
        // changes the result by 1 ULP in a fraction of the norms.
        let mut sum_sq: f64 = 0.0;
        for i in 0..n {
            let v = *x_ptr.add(i);
            sum_sq += (v * v) as f64;
        }

        let mean = (sum_sq / (n as f64)) as f32;
        let inv_rms = 1.0 / (mean + eps).sqrt();
        let v_inv_rms = _mm256_set1_ps(inv_rms);

        // 2. Normalize and scale
        for i in 0..chunks {
            let vx = _mm256_loadu_ps(x_ptr.add(i * 8));
            let vw = _mm256_loadu_ps(w_ptr.add(i * 8));
            let v_scaled = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv_rms), vw);
            _mm256_storeu_ps(out_ptr.add(i * 8), v_scaled);
        }

        for i in (chunks * 8)..n {
            *out_ptr.add(i) = *x_ptr.add(i) * inv_rms * *w_ptr.add(i);
        }
    }
}

// ============================================================================
// RoPE (Rotary Position Embeddings)
// ============================================================================

/// Applies Rotary Positional Embeddings (RoPE) to query and key activation vectors.
///
/// For head dimension $d_{\text{head}}$, frequencies are $\theta_k = \theta_{\text{base}}^{-2k/d_{\text{head}}}$.
/// Rotates each consecutive coordinate pair $(x_{2k}, x_{2k+1})$ by angle $\text{pos} \cdot \theta_k$.
pub fn apply_rotary_emb(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
) {
    let half_dim = head_dim / 2;

    // The rotation angles only depend on the position, so build the cos/sin cache once and
    // share it between all Q and K heads (same recurrence as ggml_rope_cache_init).
    let n_dims = (half_dim * 2) as f32;
    let theta_scale = theta_base.powf(-2.0 / n_dims);
    let mut theta = pos as f32;
    let mut cache = Vec::with_capacity(half_dim * 2);
    for _ in 0..half_dim {
        cache.push(theta.cos());
        cache.push(theta.sin());
        theta *= theta_scale;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            unsafe { rotate_heads_fma(q, k, num_q_heads, num_kv_heads, head_dim, &cache) };
            return;
        }
    }
    rotate_heads(q, k, num_q_heads, num_kv_heads, head_dim, &cache);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn rotate_heads_fma(
    q: &mut [f32],
    k: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: &[f32],
) {
    rotate_heads(q, k, num_q_heads, num_kv_heads, head_dim, cache);
}

/// Rotates every Q and K head. The products are contracted into fused multiply-adds, which is
/// what the reference build emits for `x0*cos - x1*sin` and `x0*sin + x1*cos`
/// (`fma(x0, cos, -(x1*sin))` and `fma(x0, sin, x1*cos)`).
#[inline(always)]
fn rotate_heads(
    q: &mut [f32],
    k: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: &[f32],
) {
    for h in 0..num_q_heads {
        rotate_head(&mut q[h * head_dim..(h + 1) * head_dim], cache);
    }
    for h in 0..num_kv_heads {
        rotate_head(&mut k[h * head_dim..(h + 1) * head_dim], cache);
    }
}

#[inline(always)]
fn rotate_head(head: &mut [f32], cache: &[f32]) {
    for i in 0..cache.len() / 2 {
        let cos_val = cache[2 * i];
        let sin_val = cache[2 * i + 1];
        let x0 = head[2 * i];
        let x1 = head[2 * i + 1];

        head[2 * i] = x0.mul_add(cos_val, -(x1 * sin_val));
        head[2 * i + 1] = x0.mul_add(sin_val, x1 * cos_val);
    }
}

// ============================================================================
// Activation Quantization (f32 -> int8)
// ============================================================================

/// Quantizes a vector of single-precision floating-point activations into 8-bit signed integers.
///
/// Uses per-token symmetric absmax scaling matching Microsoft BitNet `quantize_row_i8_s`:
/// $$s = \frac{127.0}{\max_i |x_i|}, \quad q_i = \text{round}(x_i \cdot s)$$
/// Returns the scale factor $s = 127.0 / \text{amax}$ (to divide in GEMV post-processing).
pub fn quantize_activations_int8(x: &[f32], out_int8: &mut [i8]) -> f32 {
    quantize_activations_int8_with_range(x, out_int8, 127.0)
}

/// Same as [`quantize_activations_int8`] with a configurable largest code `levels`
/// (the scale is `levels / amax` and codes are clamped to `[-(levels + 1), levels]`).
///
/// `levels = 127` is the full int8 range that the Quintrit kernels use (they are exact at this range).
/// A smaller range is only needed by table kernels whose entries pre-sum several products into int8.
pub fn quantize_activations_int8_with_range(x: &[f32], out_int8: &mut [i8], levels: f32) -> f32 {
    assert_eq!(x.len(), out_int8.len());
    let n = x.len();

    let mut max_abs: f32 = 0.0;
    for &val in x {
        let abs_val = val.abs();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    if !(max_abs > 0.0) {
        out_int8.fill(0);
        return 0.0;
    }

    let s = levels / max_abs;
    let lo = -(levels + 1.0);

    for i in 0..n {
        let q = (x[i] * s).round();
        out_int8[i] = q.clamp(lo, levels) as i8;
    }

    s
}

// ============================================================================
// Softmax & Nonlinear Activations
// ============================================================================

/// In-place numerically stable softmax computation.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum: f64 = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v as f64;
    }

    if sum > 0.0 {
        let inv_sum = (1.0 / sum) as f32;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Evaluates Squared ReLU activation: $y_i = (\max(0, x_i))^2$.
pub fn relu2(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        let v = x[i];
        let pos = if v > 0.0 { v } else { 0.0 };
        out[i] = pos * pos;
    }
}

// ============================================================================
// SiLU & SwiGLU (matching ggml_v_silu and ggml_vec_swiglu_f32)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_v_expf_avx2(x: __m256) -> __m256 {
    let r = _mm256_set1_ps(f32::from_bits(0x4b400000));
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3fb8aa3b)), r);
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(f32::from_bits(0x35bfbe8e)),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3f317200)), x),
    );
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0))));
    let c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(126.0),
        _CMP_GT_OQ,
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3c072010)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3d2b9f17)),
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3e2aaf33)),
                b,
                _mm256_set1_ps(f32::from_bits(0x3efffedb)),
            ),
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3f7ffff6)), b),
    );
    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u32 as i32)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(192.0),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn ggml_v_silu_avx2(x: __m256) -> __m256 {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let neg_x = _mm256_sub_ps(zero, x);
        let exp_neg_x = ggml_v_expf_avx2(neg_x);
        let one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
        _mm256_div_ps(x, one_plus_exp_neg_x)
    }
}

/// Evaluates SiLU (Swish) activation: $y_i = x_i \cdot \frac{1}{1 + e^{-x_i}}$ matching ggml_vec_silu_f32.
pub fn silu(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    #[allow(unused_variables)]
    let n = x.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                let mut i = 0;
                while i + 8 <= n {
                    let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                    let vy = ggml_v_silu_avx2(vx);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), vy);
                    i += 8;
                }
                while i < n {
                    let v = x[i];
                    out[i] = v / (1.0 + (-v).exp());
                    i += 1;
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { silu_neon(x, out) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        let v = x[i];
        out[i] = v / (1.0 + (-v).exp());
    }
}

/// Evaluates SwiGLU gated activation: $out_i = \text{SiLU}(gate_i) \times up_i$ matching ggml_vec_swiglu_f32.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    #[allow(unused_variables)]
    let n = gate.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                let mut i = 0;
                while i + 8 <= n {
                    let vg = _mm256_loadu_ps(gate.as_ptr().add(i));
                    let vu = _mm256_loadu_ps(up.as_ptr().add(i));
                    let silu_g = ggml_v_silu_avx2(vg);
                    let res = _mm256_mul_ps(silu_g, vu);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
                    i += 8;
                }
                while i < n {
                    let v = gate[i];
                    out[i] = (v / (1.0 + (-v).exp())) * up[i];
                    i += 1;
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { swiglu_neon(gate, up, out) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        let v = gate[i];
        out[i] = (v / (1.0 + (-v).exp())) * up[i];
    }
}

/// Elementwise vector multiplication: $out_i = a_i \times b_i$.
pub fn elementwise_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Adds residual stream vector in-place: $x_i \mathrel{+}= r_i$.
pub fn add_residual(x: &mut [f32], residual: &[f32]) {
    assert_eq!(x.len(), residual.len());
    for i in 0..x.len() {
        x[i] += residual[i];
    }
}

// ============================================================================
// Half-Precision Float (F16) Primitives
// ============================================================================

/// Converts IEEE 754 half-precision float bit pattern (16-bit) to single-precision float (f32).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            unsafe {
                let v = std::arch::x86_64::_mm_cvtsi32_si128(h as i32);
                let f_vec = std::arch::x86_64::_mm_cvtph_ps(v);
                return std::arch::x86_64::_mm_cvtss_f32(f_vec);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return f16_to_f32_neon(h);
    }
    #[allow(unreachable_code)]
    f16_to_f32_soft(h)
}

/// Portable bit-manipulation conversion half -> single (exact); the reference for the hardware conversions.
pub fn f16_to_f32_soft(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut shift = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                shift += 1;
            }
            m &= 0x3FF;
            let new_exp = 127 - 15 - shift + 1;
            f32::from_bits((sign << 31) | (new_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
        }
    } else {
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

/// Converts a single-precision float (f32) to IEEE 754 half-precision float (f16) bit pattern.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            unsafe {
                let v = std::arch::x86_64::_mm_set_ss(f);
                let h = std::arch::x86_64::_mm_cvtps_ph::<0>(v);
                return std::arch::x86_64::_mm_cvtsi128_si32(h) as u16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return f32_to_f16_neon(f);
    }
    #[allow(unreachable_code)]
    f32_to_f16_soft(f)
}

/// Portable bit-manipulation conversion single -> half with round to nearest even; the reference for the hardware conversions.
pub fn f32_to_f16_soft(f: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x77800000);
    let scale_to_zero = f32::from_bits(0x08800000);
    let mut base = (f.abs() * scale_to_inf) * scale_to_zero;

    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x80000000;
    let mut bias = shl1_w & 0xFF000000;
    if bias < 0x71000000 {
        bias = 0x71000000;
    }

    base = f32::from_bits((bias >> 1) + 0x07800000) + base;
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x00007C00;
    let mantissa_bits = bits & 0x00000FFF;
    let nonsign = exp_bits + mantissa_bits;
    let res = (sign >> 16) | if shl1_w > 0xFF000000 { 0x7E00 } else { nonsign };
    res as u16
}

/// Rounds a single-precision float to IEEE 754 half-precision representation and back to f32.
#[inline]
pub fn round_f16(f: f32) -> f32 {
    f16_to_f32(f32_to_f16(f))
}

/// AVX2/FMA/F16C implementation of [`ggml_vec_dot_f16`].
///
/// The `target_feature` attribute is required so that the `_mm256_*` intrinsics are inlined into
/// this function body. Without it they are emitted as out-of-line calls that pass 256-bit values
/// through memory, which makes this hot loop several times slower.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn ggml_vec_dot_f16_avx2(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    unsafe {
        let np = n & !31;
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut i = 0;
        while i < np {
            let ax0 = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i) as *const __m128i));
            let ay0 = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(ax0, ay0, sum0);

            let ax1 = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i + 8) as *const __m128i));
            let ay1 = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(i + 8) as *const __m128i));
            sum1 = _mm256_fmadd_ps(ax1, ay1, sum1);

            let ax2 = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i + 16) as *const __m128i));
            let ay2 = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(i + 16) as *const __m128i));
            sum2 = _mm256_fmadd_ps(ax2, ay2, sum2);

            let ax3 = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i + 24) as *const __m128i));
            let ay3 = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(i + 24) as *const __m128i));
            sum3 = _mm256_fmadd_ps(ax3, ay3, sum3);

            i += 32;
        }

        let red0 = _mm256_add_ps(sum0, sum2);
        let red1 = _mm256_add_ps(sum1, sum3);
        let red = _mm256_add_ps(red0, red1);

        let t0 = _mm_add_ps(_mm256_castps256_ps128(red), _mm256_extractf128_ps(red, 1));
        let t1 = _mm_hadd_ps(t0, t0);
        let mut sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

        while i < n {
            sumf += f16_to_f32(x[i]) * f16_to_f32(y[i]);
            i += 1;
        }
        sumf
    }
}

/// Converts eight half-precision values to two `float32x4_t` (low and high four lanes). The conversion is exact.
///
/// The NEON half-precision intrinsics are not available on the stable toolchain, hence the inline assembly.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cvt_f16x8_to_f32x4x2(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let lo: float32x4_t;
    let hi: float32x4_t;
    core::arch::asm!(
        "fcvtl {lo:v}.4s, {h:v}.4h",
        "fcvtl2 {hi:v}.4s, {h:v}.8h",
        h = in(vreg) h,
        lo = out(vreg) lo,
        hi = out(vreg) hi,
        options(pure, nomem, nostack, preserves_flags)
    );
    (lo, hi)
}

/// NEON implementation of [`ggml_vec_dot_f16`] with exactly the arithmetic of the AVX2 kernel: four accumulators of eight
/// lanes (each held as two `float32x4_t`), one fused multiply-add per lane, the reduction `((s0 + s2) + (s1 + s3))`, the
/// fold of the upper four lanes onto the lower four, two pairwise adds, and the same scalar tail.
#[cfg(target_arch = "aarch64")]
unsafe fn ggml_vec_dot_f16_neon(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let np = n & !31;
    let zero = vdupq_n_f32(0.0);
    let mut lo = [zero; 4];
    let mut hi = [zero; 4];
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            let (xl, xh) = cvt_f16x8_to_f32x4x2(vld1q_u16(x.as_ptr().add(i + 8 * j)));
            let (yl, yh) = cvt_f16x8_to_f32x4x2(vld1q_u16(y.as_ptr().add(i + 8 * j)));
            lo[j] = vfmaq_f32(lo[j], xl, yl);
            hi[j] = vfmaq_f32(hi[j], xh, yh);
        }
        i += 32;
    }
    let red_lo = vaddq_f32(vaddq_f32(lo[0], lo[2]), vaddq_f32(lo[1], lo[3]));
    let red_hi = vaddq_f32(vaddq_f32(hi[0], hi[2]), vaddq_f32(hi[1], hi[3]));
    let t0 = vaddq_f32(red_lo, red_hi);
    let t1 = vpaddq_f32(t0, t0);
    let mut sumf = vgetq_lane_f32::<0>(t1) + vgetq_lane_f32::<1>(t1);
    while i < n {
        sumf += f16_to_f32(x[i]) * f16_to_f32(y[i]);
        i += 1;
    }
    sumf
}

/// Sequential scalar dot product: the portable fallback and the former AArch64 behaviour.
pub fn ggml_vec_dot_f16_scalar(x: &[u16], y: &[u16]) -> f32 {
    let mut sumf = 0.0f32;
    for i in 0..x.len() {
        sumf += f16_to_f32(x[i]) * f16_to_f32(y[i]);
    }
    sumf
}

/// Computes exact half-precision dot product matching ggml_vec_dot_f16 with 4-way FMA accumulation and pairwise
/// reduction: AVX2 on x86-64, NEON with the identical operation order on AArch64, sequential scalar elsewhere.
pub fn ggml_vec_dot_f16(x: &[u16], y: &[u16]) -> f32 {
    assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
            return unsafe { ggml_vec_dot_f16_avx2(x, y) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { ggml_vec_dot_f16_neon(x, y) };
    }
    #[allow(unreachable_code)]
    ggml_vec_dot_f16_scalar(x, y)
}

/// Computes dot product between an F32 query vector and an F16 key vector with AVX2 FMA and pairwise reduction,
/// matching ggml_flash_attn_ext Q dot K.
pub fn vec_dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
    assert_eq!(q.len(), k_f16.len());
    #[allow(unused_variables)]
    let n = q.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
            unsafe {
                let np = n & !31;
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();

                let mut i = 0;
                while i < np {
                    let k0 = _mm256_cvtph_ps(_mm_loadu_si128(k_f16.as_ptr().add(i) as *const __m128i));
                    let q0 = _mm256_loadu_ps(q.as_ptr().add(i));
                    sum0 = _mm256_fmadd_ps(q0, k0, sum0);

                    let k1 = _mm256_cvtph_ps(_mm_loadu_si128(k_f16.as_ptr().add(i + 8) as *const __m128i));
                    let q1 = _mm256_loadu_ps(q.as_ptr().add(i + 8));
                    sum1 = _mm256_fmadd_ps(q1, k1, sum1);

                    let k2 = _mm256_cvtph_ps(_mm_loadu_si128(k_f16.as_ptr().add(i + 16) as *const __m128i));
                    let q2 = _mm256_loadu_ps(q.as_ptr().add(i + 16));
                    sum2 = _mm256_fmadd_ps(q2, k2, sum2);

                    let k3 = _mm256_cvtph_ps(_mm_loadu_si128(k_f16.as_ptr().add(i + 24) as *const __m128i));
                    let q3 = _mm256_loadu_ps(q.as_ptr().add(i + 24));
                    sum3 = _mm256_fmadd_ps(q3, k3, sum3);

                    i += 32;
                }

                let red0 = _mm256_add_ps(sum0, sum2);
                let red1 = _mm256_add_ps(sum1, sum3);
                let red = _mm256_add_ps(red0, red1);

                let t0 = _mm_add_ps(_mm256_castps256_ps128(red), _mm256_extractf128_ps(red, 1));
                let t1 = _mm_hadd_ps(t0, t0);
                let mut sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

                while i < n {
                    sumf += q[i] * f16_to_f32(k_f16[i]);
                    i += 1;
                }
                return sumf;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { vec_dot_f32_f16_neon(q, k_f16) };
    }
    #[allow(unreachable_code)]
    {
        let mut sumf = 0.0f32;
        for i in 0..n {
            sumf += q[i] * f16_to_f32(k_f16[i]);
        }
        sumf
    }
}

/// Accumulates scaled F16 values into an F32 output vector: `out[i] += f16_to_f32(v_f16[i]) * scale`.
pub fn vec_mad_f32_f16(out: &mut [f32], v_f16: &[u16], scale: f32) {
    assert_eq!(out.len(), v_f16.len());
    #[allow(unused_variables)]
    let n = out.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
            unsafe {
                let np = n & !7;
                let v_scale = _mm256_set1_ps(scale);
                let mut i = 0;
                while i < np {
                    let v_val = _mm256_cvtph_ps(_mm_loadu_si128(v_f16.as_ptr().add(i) as *const __m128i));
                    let cur_out = _mm256_loadu_ps(out.as_ptr().add(i));
                    let res = _mm256_fmadd_ps(v_val, v_scale, cur_out);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), res);
                    i += 8;
                }
                while i < n {
                    out[i] += f16_to_f32(v_f16[i]) * scale;
                    i += 1;
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vec_mad_f32_f16_neon(out, v_f16, scale) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        out[i] += f16_to_f32(v_f16[i]) * scale;
    }
}

/// Evaluates in-place softmax matching ggml_vec_soft_max_f32: computes `x[i] = exp(x[i] - max)` and returns `sum(x)`.
pub fn ggml_vec_soft_max_f32(x: &mut [f32], max: f32) -> f32 {
    #[allow(unused_variables)]
    let n = x.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                let mut i = 0;
                let mut sum: f64 = 0.0;
                let v_max = _mm256_set1_ps(max);

                while i + 8 <= n {
                    let ptr = x.as_mut_ptr().add(i);
                    let vx = _mm256_loadu_ps(ptr);
                    let diff = _mm256_sub_ps(vx, v_max);
                    let val = ggml_v_expf_avx2(diff);
                    _mm256_storeu_ps(ptr, val);

                    let val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1), _mm256_castps256_ps128(val));
                    let val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
                    let val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
                    sum += _mm_cvtss_f32(val2) as f64;

                    i += 8;
                }

                while i < n {
                    let val = (x[i] - max).exp();
                    x[i] = val;
                    sum += val as f64;
                    i += 1;
                }

                return sum as f32;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { ggml_vec_soft_max_f32_neon(x, max) };
    }
    #[allow(unreachable_code)]
    {
        let mut sum: f64 = 0.0;
        for i in 0..n {
            let val = (x[i] - max).exp();
            x[i] = val;
            sum += val as f64;
        }
        sum as f32
    }
}

/// In-place vector multiply-add in FP16 precision: $y_i = \text{round}_{\text{f16}}(y_i + x_i \times v)$.
pub fn ggml_vec_mad_f16(y: &mut [u16], x: &[u16], v: f32) {
    assert_eq!(y.len(), x.len());
    #[allow(unused_variables)]
    let n = y.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
            unsafe {
                let np = n & !31;
                let vx = _mm256_set1_ps(v);

                let mut i = 0;
                while i < np {
                    for j in 0..4 {
                        let offset = i + j * 8;
                        let ax = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(offset) as *const __m128i));
                        let ay = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(offset) as *const __m128i));
                        let res = _mm256_fmadd_ps(ax, vx, ay);
                        _mm_storeu_si128(
                            y.as_mut_ptr().add(offset) as *mut __m128i,
                            _mm256_cvtps_ph::<0>(res),
                        );
                    }
                    i += 32;
                }

                while i < n {
                    let fy = f16_to_f32(y[i]);
                    let fx = f16_to_f32(x[i]);
                    y[i] = f32_to_f16(fy + fx * v);
                    i += 1;
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { ggml_vec_mad_f16_neon(y, x, v) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        let fy = f16_to_f32(y[i]);
        let fx = f16_to_f32(x[i]);
        y[i] = f32_to_f16(fy + fx * v);
    }
}

/// In-place vector scaling in FP16 precision: $y_i = \text{round}_{\text{f16}}(y_i \times v)$.
pub fn ggml_vec_scale_f16(y: &mut [u16], v: f32) {
    #[allow(unused_variables)]
    let n = y.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe {
                let np = n & !31;
                let vx = _mm256_set1_ps(v);

                let mut i = 0;
                while i < np {
                    for j in 0..4 {
                        let offset = i + j * 8;
                        let ay = _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(offset) as *const __m128i));
                        let res = _mm256_mul_ps(ay, vx);
                        _mm_storeu_si128(
                            y.as_mut_ptr().add(offset) as *mut __m128i,
                            _mm256_cvtps_ph::<0>(res),
                        );
                    }
                    i += 32;
                }

                while i < n {
                    let fy = f16_to_f32(y[i]);
                    y[i] = f32_to_f16(fy * v);
                    i += 1;
                }
                return;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { ggml_vec_scale_f16_neon(y, v) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        let fy = f16_to_f32(y[i]);
        y[i] = f32_to_f16(fy * v);
    }
}


// ============================================================================
// AArch64 NEON implementations of the vectorized primitives
//
// Every function below performs exactly the arithmetic of its AVX2 counterpart in this file (same lanes, same fused
// multiply-adds, same order of the reductions, same tails), so that both architectures produce identical bits. The
// half-precision conversions use inline assembly because the f16 intrinsics are unstable.
// ============================================================================

/// Converts four half-precision values (`uint16x4_t`) to `float32x4_t` (exact).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cvt_f16x4_to_f32x4(h: uint16x4_t) -> float32x4_t {
    let r: float32x4_t;
    core::arch::asm!("fcvtl {r:v}.4s, {h:v}.4h", h = in(vreg) h, r = out(vreg) r, options(pure, nomem, nostack, preserves_flags));
    r
}

/// Rounds eight f32 (two vectors) to eight half-precision values with the current rounding mode (nearest even).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cvt_f32x4x2_to_f16x8(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    let r: uint16x8_t;
    core::arch::asm!(
        "fcvtn {r:v}.4h, {lo:v}.4s",
        "fcvtn2 {r:v}.8h, {hi:v}.4s",
        lo = in(vreg) lo,
        hi = in(vreg) hi,
        r = out(vreg) r,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn f16_to_f32_neon(h: u16) -> f32 {
    unsafe { vgetq_lane_f32::<0>(cvt_f16x4_to_f32x4(vdup_n_u16(h))) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn f32_to_f16_neon(f: f32) -> u16 {
    unsafe {
        let v = vdupq_n_f32(f);
        vgetq_lane_u16::<0>(cvt_f32x4x2_to_f16x8(v, v))
    }
}

/// NEON `expf` of ggml (`ggml_v_expf`), the same polynomial and the same fused operations as [`ggml_v_expf_avx2`].
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn ggml_v_expf_neon(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(f32::from_bits(0x4b400000));
    let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3fb8aa3b)));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3f317200))),
        n,
        vdupq_n_f32(f32::from_bits(0x35bfbe8e)),
    );
    let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let inner_a = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3d2b9f17)), vdupq_n_f32(f32::from_bits(0x3c072010)), b);
    let inner_b = vfmaq_f32(vdupq_n_f32(f32::from_bits(0x3efffedb)), vdupq_n_f32(f32::from_bits(0x3e2aaf33)), b);
    let mid = vfmaq_f32(inner_b, inner_a, u);
    let j = vfmaq_f32(vmulq_f32(vdupq_n_f32(f32::from_bits(0x3f7ffff6)), b), mid, u);
    if vmaxvq_u32(c) == 0 {
        return vfmaq_f32(k, j, k);
    }
    let g = vandq_u32(vcleq_f32(n, vdupq_n_f32(0.0)), vdupq_n_u32(0x82000000));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(g, vdupq_n_u32(0x7f000000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, g));
    let d = vcagtq_f32(n, vdupq_n_f32(192.0));
    let in_range = vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j));
    vbslq_f32(d, vmulq_f32(s1, s1), in_range)
}

/// NEON SiLU of ggml (`ggml_v_silu`): `x / (1 + exp(0 - x))`.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn ggml_v_silu_neon(x: float32x4_t) -> float32x4_t {
    let exp_neg_x = ggml_v_expf_neon(vsubq_f32(vdupq_n_f32(0.0), x));
    vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp_neg_x))
}

#[cfg(target_arch = "aarch64")]
unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq: f64 = 0.0;
    for i in 0..n {
        let v = x[i];
        sum_sq += (v * v) as f64;
    }
    let mean = (sum_sq / (n as f64)) as f32;
    let inv_rms = 1.0 / (mean + eps).sqrt();
    let v_inv_rms = vdupq_n_f32(inv_rms);
    let chunks = n / 8;
    for i in 0..chunks {
        for h in 0..2 {
            let o = i * 8 + h * 4;
            let scaled = vmulq_f32(vmulq_f32(vld1q_f32(x.as_ptr().add(o)), v_inv_rms), vld1q_f32(weight.as_ptr().add(o)));
            vst1q_f32(out.as_mut_ptr().add(o), scaled);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn silu_neon(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        vst1q_f32(out.as_mut_ptr().add(i), ggml_v_silu_neon(vld1q_f32(x.as_ptr().add(i))));
        vst1q_f32(out.as_mut_ptr().add(i + 4), ggml_v_silu_neon(vld1q_f32(x.as_ptr().add(i + 4))));
        i += 8;
    }
    while i < n {
        let v = x[i];
        out[i] = v / (1.0 + (-v).exp());
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn swiglu_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let n = gate.len();
    let mut i = 0;
    while i + 8 <= n {
        for h in 0..2 {
            let o = i + 4 * h;
            let s = ggml_v_silu_neon(vld1q_f32(gate.as_ptr().add(o)));
            vst1q_f32(out.as_mut_ptr().add(o), vmulq_f32(s, vld1q_f32(up.as_ptr().add(o))));
        }
        i += 8;
    }
    while i < n {
        let v = gate[i];
        out[i] = (v / (1.0 + (-v).exp())) * up[i];
        i += 1;
    }
}

/// NEON dot product of an F32 vector and an F16 vector, same order as the AVX2 version.
#[cfg(target_arch = "aarch64")]
unsafe fn vec_dot_f32_f16_neon(q: &[f32], k_f16: &[u16]) -> f32 {
    let n = q.len();
    let np = n & !31;
    let zero = vdupq_n_f32(0.0);
    let mut lo = [zero; 4];
    let mut hi = [zero; 4];
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            let o = i + 8 * j;
            let (kl, kh) = cvt_f16x8_to_f32x4x2(vld1q_u16(k_f16.as_ptr().add(o)));
            lo[j] = vfmaq_f32(lo[j], vld1q_f32(q.as_ptr().add(o)), kl);
            hi[j] = vfmaq_f32(hi[j], vld1q_f32(q.as_ptr().add(o + 4)), kh);
        }
        i += 32;
    }
    let red_lo = vaddq_f32(vaddq_f32(lo[0], lo[2]), vaddq_f32(lo[1], lo[3]));
    let red_hi = vaddq_f32(vaddq_f32(hi[0], hi[2]), vaddq_f32(hi[1], hi[3]));
    let t0 = vaddq_f32(red_lo, red_hi);
    let t1 = vpaddq_f32(t0, t0);
    let mut sumf = vgetq_lane_f32::<0>(t1) + vgetq_lane_f32::<1>(t1);
    while i < n {
        sumf += q[i] * f16_to_f32(k_f16[i]);
        i += 1;
    }
    sumf
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_mad_f32_f16_neon(out: &mut [f32], v_f16: &[u16], scale: f32) {
    let n = out.len();
    let np = n & !7;
    let v_scale = vdupq_n_f32(scale);
    let mut i = 0;
    while i < np {
        let (vl, vh) = cvt_f16x8_to_f32x4x2(vld1q_u16(v_f16.as_ptr().add(i)));
        let p = out.as_mut_ptr().add(i);
        vst1q_f32(p, vfmaq_f32(vld1q_f32(p), vl, v_scale));
        vst1q_f32(p.add(4), vfmaq_f32(vld1q_f32(p.add(4)), vh, v_scale));
        i += 8;
    }
    while i < n {
        out[i] += f16_to_f32(v_f16[i]) * scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn ggml_vec_soft_max_f32_neon(x: &mut [f32], max: f32) -> f32 {
    let n = x.len();
    let mut i = 0;
    let mut sum: f64 = 0.0;
    let v_max = vdupq_n_f32(max);
    while i + 8 <= n {
        let ptr = x.as_mut_ptr().add(i);
        let lo = ggml_v_expf_neon(vsubq_f32(vld1q_f32(ptr), v_max));
        let hi = ggml_v_expf_neon(vsubq_f32(vld1q_f32(ptr.add(4)), v_max));
        vst1q_f32(ptr, lo);
        vst1q_f32(ptr.add(4), hi);
        // Same reduction as the AVX2 version: upper onto lower four lanes, then (v0 + v2) + (v1 + v3).
        let v = vaddq_f32(hi, lo);
        let a = vgetq_lane_f32::<0>(v) + vgetq_lane_f32::<2>(v);
        let b = vgetq_lane_f32::<1>(v) + vgetq_lane_f32::<3>(v);
        sum += (a + b) as f64;
        i += 8;
    }
    while i < n {
        let val = (x[i] - max).exp();
        x[i] = val;
        sum += val as f64;
        i += 1;
    }
    sum as f32
}

#[cfg(target_arch = "aarch64")]
unsafe fn ggml_vec_mad_f16_neon(y: &mut [u16], x: &[u16], v: f32) {
    let n = y.len();
    let np = n & !31;
    let vx = vdupq_n_f32(v);
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            let o = i + 8 * j;
            let (xl, xh) = cvt_f16x8_to_f32x4x2(vld1q_u16(x.as_ptr().add(o)));
            let (yl, yh) = cvt_f16x8_to_f32x4x2(vld1q_u16(y.as_ptr().add(o)));
            let res = cvt_f32x4x2_to_f16x8(vfmaq_f32(yl, xl, vx), vfmaq_f32(yh, xh, vx));
            vst1q_u16(y.as_mut_ptr().add(o), res);
        }
        i += 32;
    }
    while i < n {
        let fy = f16_to_f32(y[i]);
        let fx = f16_to_f32(x[i]);
        y[i] = f32_to_f16(fy + fx * v);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn ggml_vec_scale_f16_neon(y: &mut [u16], v: f32) {
    let n = y.len();
    let np = n & !31;
    let vx = vdupq_n_f32(v);
    let mut i = 0;
    while i < np {
        for j in 0..4 {
            let o = i + 8 * j;
            let (yl, yh) = cvt_f16x8_to_f32x4x2(vld1q_u16(y.as_ptr().add(o)));
            vst1q_u16(y.as_mut_ptr().add(o), cvt_f32x4x2_to_f16x8(vmulq_f32(yl, vx), vmulq_f32(yh, vx)));
        }
        i += 32;
    }
    while i < n {
        let fy = f16_to_f32(y[i]);
        y[i] = f32_to_f16(fy * v);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_matches_reference_scale_and_range() {
        let x = [0.5f32, -1.0, 0.25, 1.0, -0.001];
        let mut q = [0i8; 5];
        let s = quantize_activations_int8(&x, &mut q);
        assert_eq!(s, 127.0f32 / 1.0);
        assert_eq!(q, [64, -127, 32, 127, 0]);
    }

    #[test]
    fn quantize_with_range_uses_requested_levels() {
        let x = [0.5f32, -1.0, 0.25, 1.0];
        let mut q = [0i8; 4];
        let s = quantize_activations_int8_with_range(&x, &mut q, 63.0);
        assert_eq!(s, 63.0);
        assert_eq!(q, [32, -63, 16, 63]);
    }

    #[test]
    fn quantize_all_zero_returns_zero_scale() {
        let x = [0.0f32; 8];
        let mut q = [1i8; 8];
        assert_eq!(quantize_activations_int8(&x, &mut q), 0.0);
        assert_eq!(q, [0i8; 8]);
    }

    #[test]
    fn vec_dot_f16_close_to_scalar_reference() {
        let n = 256;
        let a: Vec<u16> = (0..n).map(|i| f32_to_f16(((i * 7 % 13) as f32 - 6.0) * 0.125)).collect();
        let b: Vec<u16> = (0..n).map(|i| f32_to_f16(((i * 5 % 11) as f32 - 5.0) * 0.25)).collect();
        let expected: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f16_to_f32(x) as f64 * f16_to_f32(y) as f64)
            .sum();
        let got = ggml_vec_dot_f16(&a, &b) as f64;
        assert!((got - expected).abs() < 1e-3, "got {got}, expected {expected}");
    }

    /// Portable model of the arithmetic of the vectorized dot product (4 accumulators of 8 lanes, fused multiply-add,
    /// pairwise reduction): the NEON and AVX2 kernels must reproduce it bit for bit.
    fn dot_f16_model(x: &[u16], y: &[u16]) -> f32 {
        let n = x.len();
        let np = n & !31;
        let mut acc = [[0.0f32; 8]; 4];
        let mut i = 0;
        while i < np {
            for j in 0..4 {
                for l in 0..8 {
                    acc[j][l] = f16_to_f32(x[i + 8 * j + l]).mul_add(f16_to_f32(y[i + 8 * j + l]), acc[j][l]);
                }
            }
            i += 32;
        }
        let mut red = [0.0f32; 8];
        for l in 0..8 {
            red[l] = (acc[0][l] + acc[2][l]) + (acc[1][l] + acc[3][l]);
        }
        let t0: Vec<f32> = (0..4).map(|l| red[l] + red[l + 4]).collect();
        let mut sumf = (t0[0] + t0[1]) + (t0[2] + t0[3]);
        while i < n {
            sumf += f16_to_f32(x[i]) * f16_to_f32(y[i]);
            i += 1;
        }
        sumf
    }

    fn test_rng(seed: u64) -> impl FnMut() -> u64 {
        let mut seed = seed;
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        }
    }

    #[test]
    fn vec_dot_f16_reproduces_the_modelled_operation_order_bit_for_bit() {
        let mut next = test_rng(0x1234_5678_9ABC_DEF1);
        for &n in &[1usize, 7, 31, 32, 33, 64, 100, 256, 3072, 3079] {
            for _ in 0..20 {
                let x: Vec<u16> = (0..n).map(|_| f32_to_f16(((next() % 20001) as f32 - 10000.0) * 1e-3)).collect();
                let y: Vec<u16> = (0..n).map(|_| f32_to_f16(((next() % 2001) as f32 - 1000.0) * 1e-3)).collect();
                assert_eq!(ggml_vec_dot_f16(&x, &y).to_bits(), dot_f16_model(&x, &y).to_bits(), "n={n}");
            }
        }
        // Half-precision special values convert exactly: subnormals, the largest finite value, signed zeros.
        let x: Vec<u16> = vec![0x0001, 0x03FF, 0x7BFF, 0x8000, 0x0000, 0xFBFF, 0x0400, 0x3C00];
        let y: Vec<u16> = vec![0x3C00, 0x3C00, 0x0400, 0x3C00, 0x3C00, 0x0400, 0x7800, 0xBC00];
        assert_eq!(ggml_vec_dot_f16(&x, &y).to_bits(), dot_f16_model(&x, &y).to_bits());
    }

    /// Prints a checksum of many dot products; identical output on two machines means bit-identical results.
    #[test]
    fn vec_dot_f16_checksum() {
        let mut next = test_rng(0x0F0F_1234_5678_9ABD);
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &n in &[256usize, 3072, 3079] {
            for _ in 0..2000 {
                let x: Vec<u16> = (0..n).map(|_| f32_to_f16(((next() % 20001) as f32 - 10000.0) * 1e-3)).collect();
                let y: Vec<u16> = (0..n).map(|_| f32_to_f16(((next() % 2001) as f32 - 1000.0) * 1e-3)).collect();
                h = (h ^ ggml_vec_dot_f16(&x, &y).to_bits() as u64).wrapping_mul(0x100_0000_01B3);
            }
        }
        println!("VEC_DOT_F16_CHECKSUM {h:016x}");
    }

    fn fnv(h: &mut u64, bits: u64) {
        *h = (*h ^ bits).wrapping_mul(0x100_0000_01B3);
    }

    fn rand_f32(next: &mut impl FnMut() -> u64, scale: f32) -> f32 {
        ((next() % 2_000_001) as f32 - 1_000_000.0) * 1e-6 * scale
    }

    /// The hardware conversions must agree with the portable ones for every half-precision value and for many floats.
    #[test]
    fn half_conversions_agree_with_the_portable_implementation() {
        for h in 0..=u16::MAX {
            let (a, b) = (f16_to_f32(h), f16_to_f32_soft(h));
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "f16 {h:#06x}: {a} vs {b}");
        }
        let mut next = test_rng(0xDEAD_BEEF_0BAD_F00D);
        let check = |f: f32| {
            let (a, b) = (f32_to_f16(f), f32_to_f16_soft(f));
            let nan = f.is_nan();
            assert!(a == b || (nan && (a & 0x7C00) == 0x7C00 && (b & 0x7C00) == 0x7C00), "f32 {:#010x}: {a:#06x} vs {b:#06x}", f.to_bits());
        };
        for _ in 0..4_000_000 {
            check(f32::from_bits(next() as u32));
        }
        for _ in 0..1_000_000 {
            // Values around the half-precision range, including the subnormal range and the rounding boundaries.
            let e = (next() % 40) as i32 - 28;
            check((rand_f32(&mut next, 1.0)) * 2f32.powi(e));
        }
        for &f in &[0.0f32, -0.0, 65504.0, 65519.9, 65520.0, 1e-8, 5.960_464_5e-8, 2.98e-8, 2.99e-8, f32::INFINITY, f32::NEG_INFINITY] {
            check(f);
        }
    }

    /// Prints a checksum per primitive; the lines must be identical on x86-64 (AVX2) and AArch64 (NEON).
    #[test]
    fn primitive_checksums() {
        let mut next = test_rng(0x7777_ABCD_1234_5679);

        // SiLU and SwiGLU (lengths that are multiples of eight: the scalar tail uses the platform libm).
        let extremes = [-200.0f32, -100.0, -88.0, -87.5, -1e-8, 0.0, 1e-8, 1.0, 88.0, 90.0, 100.0, 200.0];
        let n = 4096;
        let gate: Vec<f32> = (0..n)
            .map(|i| if i % 97 < extremes.len() { extremes[i % 97] } else { rand_f32(&mut next, 12.0) })
            .collect();
        let up: Vec<f32> = (0..n).map(|_| rand_f32(&mut next, 2.0)).collect();
        let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
        silu(&gate, &mut a);
        swiglu(&gate, &up, &mut b);
        let (mut h1, mut h2) = (0xCBF2_9CE4_8422_2325u64, 0xCBF2_9CE4_8422_2325u64);
        a.iter().for_each(|v| fnv(&mut h1, v.to_bits() as u64));
        b.iter().for_each(|v| fnv(&mut h2, v.to_bits() as u64));
        println!("OPS_CHECKSUM silu {h1:016x}");
        println!("OPS_CHECKSUM swiglu {h2:016x}");

        // Softmax.
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for _ in 0..200 {
            let mut x: Vec<f32> = (0..1000).map(|_| rand_f32(&mut next, 20.0)).collect();
            let max = x.iter().cloned().fold(f32::MIN, f32::max);
            let sum = ggml_vec_soft_max_f32(&mut x, max);
            fnv(&mut h, sum.to_bits() as u64);
            x.iter().for_each(|v| fnv(&mut h, v.to_bits() as u64));
        }
        println!("OPS_CHECKSUM soft_max {h:016x}");

        // RMSNorm.
        let mut h = 0xCBF2_9CE4_8422_2325u64;
        for &n in &[3072usize, 3079] {
            let x: Vec<f32> = (0..n).map(|_| rand_f32(&mut next, 3.0)).collect();
            let w: Vec<f32> = (0..n).map(|_| rand_f32(&mut next, 1.0)).collect();
            let mut out = vec![0.0f32; n];
            rms_norm(&x, &w, 1e-5, &mut out);
            out.iter().for_each(|v| fnv(&mut h, v.to_bits() as u64));
        }
        println!("OPS_CHECKSUM rms_norm {h:016x}");

        // F16 primitives of the attention.
        let (mut h_scale, mut h_mad, mut h_dot, mut h_madf32) = (0xCBF2_9CE4_8422_2325u64, 0xCBF2_9CE4_8422_2325u64, 0xCBF2_9CE4_8422_2325u64, 0xCBF2_9CE4_8422_2325u64);
        for &n in &[256usize, 293] {
            for _ in 0..500 {
                let x: Vec<u16> = (0..n).map(|_| f32_to_f16(rand_f32(&mut next, 4.0))).collect();
                let y: Vec<u16> = (0..n).map(|_| f32_to_f16(rand_f32(&mut next, 4.0))).collect();
                let v = rand_f32(&mut next, 3.0);
                let mut y1 = y.clone();
                ggml_vec_mad_f16(&mut y1, &x, v);
                y1.iter().for_each(|e| fnv(&mut h_mad, *e as u64));
                let mut y2 = y.clone();
                ggml_vec_scale_f16(&mut y2, v);
                y2.iter().for_each(|e| fnv(&mut h_scale, *e as u64));
                let q: Vec<f32> = (0..n).map(|_| rand_f32(&mut next, 2.0)).collect();
                fnv(&mut h_dot, vec_dot_f32_f16(&q, &x).to_bits() as u64);
                let mut o = q.clone();
                vec_mad_f32_f16(&mut o, &y, v);
                o.iter().for_each(|e| fnv(&mut h_madf32, e.to_bits() as u64));
            }
        }
        println!("OPS_CHECKSUM vec_mad_f16 {h_mad:016x}");
        println!("OPS_CHECKSUM vec_scale_f16 {h_scale:016x}");
        println!("OPS_CHECKSUM vec_dot_f32_f16 {h_dot:016x}");
        println!("OPS_CHECKSUM vec_mad_f32_f16 {h_madf32:016x}");
    }
}
