// Kernels for the hybrid decoders of the Qwen3.5 family (Gated DeltaNet layers, gated attention, Hadamard-folded weights, F32 and BF16 projections). They follow
// the CPU runtime (quintrit-runtime, `hybrid.rs` and `ops.rs`) operation by operation so that their results are bit-identical to it. Every floating-point
// operation of this file is an explicit round-to-nearest intrinsic or an `fmaf`, so the compiler cannot contract or reorder anything; the exponential is
// glibc's `expf` (glibc_expf.cuh), `log1pf` is a port of glibc's x86-64 function (`log1p_f32_glibc` of `ops.rs`).
#include "ggml_silu.cuh"
#include "glibc_expf.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// The reduction of ggml_vec_dot_f32 over the 32 lane accumulators of a warp (lane = 8 * accumulator + element of the 256-bit register):
// a[l] = (acc0[l] + acc2[l]) + (acc1[l] + acc3[l]), then ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7)). Every lane returns the result.
__device__ __forceinline__ float dot_reduce_f32(float acc, int lane) {
    const int l = lane & 7;
    const float a0 = __shfl_sync(0xffffffffu, acc, l), a1 = __shfl_sync(0xffffffffu, acc, l + 8);
    const float a2 = __shfl_sync(0xffffffffu, acc, l + 16), a3 = __shfl_sync(0xffffffffu, acc, l + 24);
    const float r = __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3));
    const float t0 = __fadd_rn(__shfl_sync(0xffffffffu, r, 0), __shfl_sync(0xffffffffu, r, 4));
    const float t1 = __fadd_rn(__shfl_sync(0xffffffffu, r, 1), __shfl_sync(0xffffffffu, r, 5));
    const float t2 = __fadd_rn(__shfl_sync(0xffffffffu, r, 2), __shfl_sync(0xffffffffu, r, 6));
    const float t3 = __fadd_rn(__shfl_sync(0xffffffffu, r, 3), __shfl_sync(0xffffffffu, r, 7));
    return __fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3));
}

// The reduction of ggml_vec_dot_bf16: a[l] = (acc0[l] + acc2[l]) + (acc1[l] + acc3[l]), g = [a0 + a4, a1 + a5, a2 + a6, a3 + a7], (g0 + g2) + (g1 + g3).
__device__ __forceinline__ float dot_reduce_bf16(float acc, int lane) {
    const int l = lane & 7;
    const float a0 = __shfl_sync(0xffffffffu, acc, l), a1 = __shfl_sync(0xffffffffu, acc, l + 8);
    const float a2 = __shfl_sync(0xffffffffu, acc, l + 16), a3 = __shfl_sync(0xffffffffu, acc, l + 24);
    const float r = __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3));
    const float g0 = __fadd_rn(__shfl_sync(0xffffffffu, r, 4), __shfl_sync(0xffffffffu, r, 0));
    const float g1 = __fadd_rn(__shfl_sync(0xffffffffu, r, 5), __shfl_sync(0xffffffffu, r, 1));
    const float g2 = __fadd_rn(__shfl_sync(0xffffffffu, r, 6), __shfl_sync(0xffffffffu, r, 2));
    const float g3 = __fadd_rn(__shfl_sync(0xffffffffu, r, 7), __shfl_sync(0xffffffffu, r, 3));
    return __fadd_rn(__fadd_rn(g0, g2), __fadd_rn(g1, g3));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------------------------------------------------------------------------

// `sigmoid_f32` of ops.rs: 1 / (1 + expf(-x)).
__device__ __forceinline__ float sigmoid_dev(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, glibc_expf(-x))); }

// `log1p_f32_glibc` of ops.rs (the fdlibm algorithm of glibc's s_log1pf.c, single precision, no fused multiply-adds).
__device__ float log1pf_glibc(float x) {
    const float LN2_HI = __uint_as_float(0x3f317180u), LN2_LO = __uint_as_float(0x3717f7d1u);
    const float LP0 = __uint_as_float(0x3F2AAAABu), LP1 = __uint_as_float(0x3ECCCCCDu), LP2 = __uint_as_float(0x3E924925u), LP3 = __uint_as_float(0x3E638E29u);
    const float LP4 = __uint_as_float(0x3E3A3325u), LP5 = __uint_as_float(0x3E1CD04Fu), LP6 = __uint_as_float(0x3E178897u);
    const int hx = __float_as_int(x);
    const int ax = hx & 0x7fffffff;
    int k = 1, hu = 0;
    float f = 0.0f, c = 0.0f;
    if (hx < 0x3ed413d7) {  // x < 0.41422
        if (ax >= 0x3f800000) return x == -1.0f ? -INFINITY : __uint_as_float(0x7fc00000u);  // x <= -1
        if (ax < 0x31000000) {                                                                  // |x| < 2^-29
            if (ax < 0x24800000) return x;
            return __fsub_rn(x, __fmul_rn(__fmul_rn(x, x), 0.5f));
        }
        if (hx > 0 || hx <= static_cast<int>(0xbe95f61fu)) {  // -0.2929 < x < 0.41422
            k = 0;
            f = x;
            hu = 1;
        }
    }
    if (hx >= 0x7f800000) return __fadd_rn(x, x);
    if (k != 0) {
        float u;
        if (hx < 0x5a000000) {
            u = __fadd_rn(1.0f, x);
            hu = __float_as_int(u);
            k = (hu >> 23) - 127;
            c = k > 0 ? __fsub_rn(1.0f, __fsub_rn(u, x)) : __fsub_rn(x, __fsub_rn(u, 1.0f));
            c = __fdiv_rn(c, u);
        } else {
            u = x;
            hu = __float_as_int(u);
            k = (hu >> 23) - 127;
            c = 0.0f;
        }
        hu &= 0x007fffff;
        if (hu < 0x003504f7) {
            u = __int_as_float(hu | 0x3f800000);
        } else {
            k += 1;
            u = __int_as_float(hu | 0x3f000000);
            hu = (0x00800000 - hu) >> 2;
        }
        f = __fsub_rn(u, 1.0f);
    }
    const float hfsq = __fmul_rn(__fmul_rn(0.5f, f), f);
    const float kf = static_cast<float>(k);
    if (hu == 0) {  // |f| < 2^-20
        if (f == 0.0f) {
            if (k == 0) return 0.0f;
            c = __fadd_rn(c, __fmul_rn(kf, LN2_LO));
            return __fadd_rn(__fmul_rn(kf, LN2_HI), c);
        }
        const float r = __fmul_rn(hfsq, __fsub_rn(1.0f, __fmul_rn(0.6666667f, f)));
        if (k == 0) return __fsub_rn(f, r);
        return __fsub_rn(__fmul_rn(kf, LN2_HI), __fsub_rn(__fsub_rn(r, __fadd_rn(__fmul_rn(kf, LN2_LO), c)), f));
    }
    const float s = __fdiv_rn(f, __fadd_rn(2.0f, f));
    const float z = __fmul_rn(s, s);
    float p = __fadd_rn(LP5, __fmul_rn(z, LP6));
    p = __fadd_rn(LP4, __fmul_rn(z, p));
    p = __fadd_rn(LP3, __fmul_rn(z, p));
    p = __fadd_rn(LP2, __fmul_rn(z, p));
    p = __fadd_rn(LP1, __fmul_rn(z, p));
    p = __fadd_rn(LP0, __fmul_rn(z, p));
    const float r = __fmul_rn(z, p);
    if (k == 0) return __fsub_rn(f, __fsub_rn(hfsq, __fmul_rn(s, __fadd_rn(hfsq, r))));
    return __fsub_rn(__fmul_rn(kf, LN2_HI),
                     __fsub_rn(__fsub_rn(hfsq, __fadd_rn(__fmul_rn(s, __fadd_rn(hfsq, r)), __fadd_rn(__fmul_rn(kf, LN2_LO), c))), f));
}

// `softplus_f32` of ops.rs: x above 20, log1pf(expf(x)) otherwise.
__device__ __forceinline__ float softplus_dev(float x) { return x > 20.0f ? x : log1pf_glibc(glibc_expf(x)); }

// Test probe: out[i] = f(x[i]) with f = 0: expf, 1: sigmoid, 2: softplus, 3: log1pf, 4: silu (ggml's polynomial).
extern "C" __global__ void __launch_bounds__(256) probe_unary(const float* __restrict__ x, float* __restrict__ out, int n, int which) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    float r;
    switch (which) {
        case 0: r = glibc_expf(v); break;
        case 1: r = sigmoid_dev(v); break;
        case 2: r = softplus_dev(v); break;
        case 3: r = log1pf_glibc(v); break;
        default: r = ggml_silu(v); break;
    }
    out[i] = r;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Hadamard transform of the input of a folded weight: out = H (D (P x)) per block of `block` values, as Hadamard::apply_input: optional permutation of the value
// heads (perm_hd > 0: input [hd, nk, rep] reordered to [hd, rep, nk]), the signs D, the normalized Walsh-Hadamard transform of every block (the values scaled
// by `scale` = 1 / sqrt(block) first, then the butterflies (u + v, u - v) with growing stride). One CUDA block per transform block, dynamic shared memory:
// `block` floats.
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(1024) hadamard_blocks(const float* __restrict__ x, float* __restrict__ out, const float* __restrict__ signs, int block,
                                                                   float scale, int perm_hd, int perm_nk, int perm_rep) {
    extern __shared__ float sh[];
    const int base = blockIdx.x * block;
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        const int g = base + i;
        int src = g;
        if (perm_hd > 0) {
            const int m = g / perm_hd, e = g % perm_hd;
            const int k = m / perm_rep, r = m % perm_rep;
            src = perm_hd * (k + perm_nk * r) + e;
        }
        float v = x[src];
        if (signs) v = __fmul_rn(v, signs[g]);
        sh[i] = __fmul_rn(v, scale);
    }
    __syncthreads();
    for (int lg = 0; (1 << lg) < block; ++lg) {
        const int len = 1 << lg;
        for (int p = threadIdx.x; p < block / 2; p += blockDim.x) {
            const int i0 = ((p >> lg) << (lg + 1)) | (p & (len - 1));  // = (p / len) * 2 * len + p % len for a power of two len
            const float u = sh[i0], v = sh[i0 + len];
            sh[i0] = __fadd_rn(u, v);
            sh[i0 + len] = __fsub_rn(u, v);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < block; i += blockDim.x) out[base + i] = sh[i];
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// F32 and BF16 projections: one warp per row, the 32 lane accumulators of ggml_vec_dot_f32 / ggml_vec_dot_bf16 (cols a multiple of 32).
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) gemv_f32_dot(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ out, int rows, int cols) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int rc = min(row, rows - 1);
    const float* wr = w + (size_t)rc * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols / 32; ++c) acc = __fmaf_rn(wr[32 * c + lane], x[32 * c + lane], acc);
    const float sum = dot_reduce_f32(acc, lane);
    if (lane == 0 && row < rows) out[row] = sum;
}

// f32_to_bf16 of ops.rs: round to nearest even, NaN made quiet.
__device__ __forceinline__ float bf16_round(float f) {
    const uint32_t u = __float_as_uint(f);
    uint32_t h;
    if ((u & 0x7fffffffu) > 0x7f800000u) h = (u >> 16) | 64u;
    else h = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
    return __uint_as_float((h & 0xffffu) << 16);
}

extern "C" __global__ void __launch_bounds__(256) gemv_bf16_dot(const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ out, int rows, int cols) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int rc = min(row, rows - 1);
    const uint16_t* wr = w + (size_t)rc * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols / 32; ++c) {
        const float wv = __uint_as_float(static_cast<uint32_t>(wr[32 * c + lane]) << 16);
        acc = __fadd_rn(acc, __fmul_rn(wv, bf16_round(x[32 * c + lane])));
    }
    const float sum = dot_reduce_bf16(acc, lane);
    if (lane == 0 && row < rows) out[row] = sum;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------------------------------------------------------------------------

// beta = sigmoid(beta_raw), gate = softplus(alpha_raw + dt_bias) * a, per value head.
extern "C" __global__ void __launch_bounds__(256) gdn_gates(const float* __restrict__ beta_raw, const float* __restrict__ alpha_raw, const float* __restrict__ dt_bias,
                                                            const float* __restrict__ a, float* __restrict__ beta, float* __restrict__ gate, int nv) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nv) return;
    beta[h] = sigmoid_dev(beta_raw[h]);
    gate[h] = __fmul_rn(softplus_dev(__fadd_rn(alpha_raw[h], dt_bias[h])), a[h]);
}

// Causal depthwise convolution over [state (kernel - 1 values), new value] per channel with the fused multiply-adds of the CPU, the state shifted by one, then
// SiLU. The state of a sequence that starts (position 0) is zero. conv_state: [ch][kernel - 1], w: [ch][kernel].
extern "C" __global__ void __launch_bounds__(256) gdn_conv(const float* __restrict__ a, float* __restrict__ conv_state, const float* __restrict__ w, float* __restrict__ out,
                                                           const int* __restrict__ pos_dev, int ch, int kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int km1 = kernel - 1;
    const bool fresh = pos_dev[0] == 0;
    float* st = conv_state + (size_t)c * km1;
    const float input = a[c];
    float prev[8];
    for (int i = 0; i < km1; ++i) prev[i] = fresh ? 0.0f : st[i];
    float sum = 0.0f;
    for (int i = 0; i < kernel; ++i) sum = __fmaf_rn(i < km1 ? prev[i] : input, w[(size_t)c * kernel + i], sum);
    out[c] = ggml_silu(sum);
    for (int i = 0; i + 1 < km1; ++i) st[i] = prev[i + 1];
    st[km1 - 1] = input;
}

// L2 normalization of `gridDim.x` heads of `hd` values in place: rms_norm(x, ones, eps / hd) * (1 / sqrt(hd)), eps_l2 and inv_sqrt computed by the host in f32.
extern "C" __global__ void __launch_bounds__(256) gdn_l2(float* __restrict__ x, int hd, float eps_l2, float inv_sqrt) {
    __shared__ double red[8];
    __shared__ float s_scale;
    float* row = x + (size_t)blockIdx.x * hd;
    double ss = 0.0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += static_cast<double>(__fmul_rn(row[i], row[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int k = 0; k < 8; ++k) t += red[k];
        const float mean = static_cast<float>(t / static_cast<double>(hd));
        s_scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps_l2)));
    }
    __syncthreads();
    const float scale = s_scale;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) row[i] = __fmul_rn(__fmul_rn(__fmul_rn(row[i], scale), 1.0f), inv_sqrt);
}

// The recurrence of ggml_gated_delta_net for one token (K = 1). One warp per (value head h, state row j): the state row is held in registers, lane t keeping the
// elements 32 i + t, which are the lane accumulators of ggml_vec_dot_f32. The state of a sequence that starts (position 0) is zero.
//   st *= exp(gate[h]); sum = dot(st_j, k); delta = (v_j - sum) * beta[h]; st_j[i] = fma(k[i], delta, st_j[i]); out[h][j] = dot(st_j, q) * scale.
// `conv` holds the convolved and normalized [q (key_dim) | k (key_dim) | v (value_dim)]; value head h uses key head h % nk.
extern "C" __global__ void __launch_bounds__(256) gdn_recur(float* __restrict__ state, const float* __restrict__ conv, const float* __restrict__ beta,
                                                            const float* __restrict__ gate, float* __restrict__ out, const int* __restrict__ pos_dev, int nk, int nv,
                                                            int hd, float scale) {
    const int warp = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int h = warp / hd, j = warp % hd;
    if (h >= nv) return;
    const int key_dim = nk * hd;
    const int kh = h % nk;
    const float* q = conv + kh * hd;
    const float* k = conv + key_dim + kh * hd;
    const float vj = conv[2 * key_dim + h * hd + j];
    float* st = state + ((size_t)h * hd + j) * hd;
    const bool fresh = pos_dev[0] == 0;
    const float decay = glibc_expf(gate[h]);
    const int chunks = hd / 32;
    float s[8];
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < chunks) {
            s[c] = __fmul_rn(fresh ? 0.0f : st[32 * c + lane], decay);
            acc = __fmaf_rn(s[c], k[32 * c + lane], acc);
        }
    }
    const float sum = dot_reduce_f32(acc, lane);
    const float delta = __fmul_rn(__fsub_rn(vj, sum), beta[h]);
    float acc2 = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < chunks) {
            s[c] = __fmaf_rn(k[32 * c + lane], delta, s[c]);
            acc2 = __fmaf_rn(s[c], q[32 * c + lane], acc2);
            st[32 * c + lane] = s[c];
        }
    }
    const float o = dot_reduce_f32(acc2, lane);
    if (lane == 0) out[h * hd + j] = __fmul_rn(o, scale);
}

// Gated RMSNorm per value head: out = ((x * scale) * w) * silu(z), `gridDim.x` heads of `hd` values; w holds hd weights.
extern "C" __global__ void __launch_bounds__(256) gdn_out_norm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ z,
                                                               float* __restrict__ out, int hd, float eps) {
    __shared__ double red[8];
    __shared__ float s_scale;
    const float* row = x + (size_t)blockIdx.x * hd;
    double ss = 0.0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += static_cast<double>(__fmul_rn(row[i], row[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int k = 0; k < 8; ++k) t += red[k];
        const float mean = static_cast<float>(t / static_cast<double>(hd));
        s_scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();
    const float scale = s_scale;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const size_t at = (size_t)blockIdx.x * hd + i;
        out[at] = __fmul_rn(__fmul_rn(__fmul_rn(row[i], scale), w[i]), ggml_silu(z[at]));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Gated attention: att[h][e] *= sigmoid(gate[h][e]), the gate of head h being stored after its query in the joint projection (q | gate per head, stride 2 hd).
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) attn_gate(float* __restrict__ att, const float* __restrict__ joint, int hd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i / hd, e = i % hd;
    att[i] = __fmul_rn(att[i], sigmoid_dev(joint[2 * h * hd + hd + e]));
}
