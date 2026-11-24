// Kernels for block-scaled ternary models (llama.cpp `Q2_0` weights, `Q8_0` activations). They follow the CPU runtime operation by operation, so that their
// results are bit-identical to it:
//  * q8_quant: `quantize_q8_0_scalar` of quintrit-runtime: per block of 32 values amax, scale d = f16(amax / 127) stored as F32, codes round_ties_even(v * (127 / amax)).
//  * gemv_q2: `dot_row_scalar` of quintrit_core::block_scaled. The weights are the planar velurex layout of gemv.cu (32-byte blocks of 160 trits; plane p of block
//    b is the activation block 5b + p), one F16 scale per 64 weights. For every activation block s there are eight integer lane sums (lane j = values 4j..4j+3) which
//    are accumulated with one fused multiply-add per lane into one of two chains, chain = (s / 2) & 1: acc[chain][j] = fma(w_scale[s / 2] * a_scale[s], lane_j, acc[chain][j]).
//    The chains of a lane are sequential in s, so one thread owns one lane j of one row (eight threads per row) and, in `gemv_q2_split`, one chain of it; the result is
//    (acc0 + acc1) per lane reduced as ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)).
//    The weight words are not in the file order but regrouped when the matrix is uploaded (`group_blocks` in q2.rs): per row and group of four blocks the four words of
//    lane j are adjacent, so that a thread reads them with one 16-byte load and a row is read in contiguous 128-byte pieces.
//  * rms_norm_rows: `rms_norm` (sum of the f32-rounded squares in double, scale = 1 / sqrt(mean + eps), y = (x * scale) * w).
//  * rope_kv_f32: `apply_rope_cached` on f32 input, key and value appended to the F16 cache.
//  * swiglu_f32, add_f32: `swiglu` (ggml's silu polynomial) and `add_residual`.
// Every floating-point operation is an explicit round-to-nearest intrinsic or an `fmaf`, so no contraction or reordering is possible.
#include "ggml_silu.cuh"
#include "q2_words.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------------------------------------------------------------------------
// Q8_0 quantization: one warp per block of 32 values.
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) q8_quant(const float* __restrict__ x, int8_t* __restrict__ qs, float* __restrict__ d, int blocks) {
    const int blk = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk >= blocks) return;
    const float v = x[blk * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // fmaxf ignores a NaN operand, like f32::max; a block of NaN only gives 0.
    if (lane == 0) d[blk] = __half2float(__float2half_rn(__fdiv_rn(amax, 127.0f)));
    const float id = amax != 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    const int q = __float2int_rn(__fmul_rn(v, id));
    qs[blk * 32 + lane] = static_cast<int8_t>(max(-128, min(127, q)));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Block-scaled GEMV: out[row] = dot(row, activations). 8 warps per block, a warp covers 4 rows with 8 lanes each.
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) gemv_q2(const uint32_t* __restrict__ w, const __half* __restrict__ wsc, const int8_t* __restrict__ qs,
                                                          const float* __restrict__ ad, float* __restrict__ out, int rows, int cols) {
    __shared__ uint32_t tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = digits_word_q2(i);
    __syncthreads();
    const int lane = threadIdx.x & 31, j = lane & 7;
    const int row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 4 + (lane >> 3);
    const int rc = min(row, rows - 1);  // every lane takes part in the shuffles; rows past the end are computed and discarded
    const int blocks_per_row = (cols + 159) / 160, nb32 = cols / 32, sb = cols / 64;
    // Run-time layout (see BlockScaledMatrix::new): the words of four consecutive blocks of one lane are adjacent, so that a thread reads 16 bytes per group
    // and the eight lanes of a row read 128 contiguous bytes. The last group of a row may be partial; its missing words are not read.
    const int groups = (blocks_per_row + 3) >> 2, full_groups = blocks_per_row >> 2, tail = blocks_per_row & 3;
    const uint32_t* wr = w + (size_t)rc * groups * 32;  // rows are padded to whole groups
    const __half* sc = wsc + (size_t)rc * sb;
    float acc0 = 0.f, acc1 = 0.f;
    for (int g = 0; g < groups; ++g) {
        uint32_t words[4];
        load_group(wr, g, full_groups, tail, j, words);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint32_t word = words[u];
            const int b = 4 * g + u;
            const uint32_t t0 = tab[word & 0xffu], t1 = tab[(word >> 8) & 0xffu], t2 = tab[(word >> 16) & 0xffu], t3 = tab[word >> 24];
#pragma unroll
            for (int p = 0; p < 5; ++p) {
                const int s = 5 * b + p;
                if (s < nb32) {
                    uint32_t dig;
                    if (p < 4) {
                        // byte p of t0..t3, transposed into one 32-bit operand
                        const uint32_t u01 = __byte_perm(t0, t1, p | ((p + 4) << 4));
                        const uint32_t u23 = __byte_perm(t2, t3, p | ((p + 4) << 4));
                        dig = __byte_perm(u01, u23, 0x5410);
                    } else {
                        dig = __vsub4(((((word & 0xffu) * 781u) >> 16)) | (((((word >> 8) & 0xffu) * 781u) >> 16) << 8) | (((((word >> 16) & 0xffu) * 781u) >> 16) << 16) |
                                      (((word >> 24) * 781u >> 16) << 24), 0x01010101u);
                    }
                    const int aw = *reinterpret_cast<const int*>(qs + s * 32 + 4 * j);
                    const int lane_sum = __dp4a(static_cast<int>(dig), aw, 0);
                    const float dd = __fmul_rn(__half2float(sc[s >> 1]), ad[s]);
                    if ((s >> 1) & 1) acc1 = __fmaf_rn(dd, __int2float_rn(lane_sum), acc1);
                    else acc0 = __fmaf_rn(dd, __int2float_rn(lane_sum), acc0);
                }
            }
        }
    }
    float l = __fadd_rn(acc0, acc1);
    const float r = __fadd_rn(l, __shfl_down_sync(0xffffffffu, l, 4, 8));      // lanes 0..3: l[j] + l[j + 4]
    const float y = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2, 8));      // lane 0: r0 + r2, lane 1: r1 + r3
    const float z = __fadd_rn(y, __shfl_down_sync(0xffffffffu, y, 1, 8));      // lane 0: (r0 + r2) + (r1 + r3)
    if (j == 0 && row < rows) out[row] = z;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The same projection with the two FMA chains of a lane in two warps (chain 0 and chain 1 of the same four rows), for matrices with so few rows that
// the kernel above cannot fill the device. In a group of four blocks the chain of plane s = 20 g + 5 u + p is ((5 u + p) >> 1) & 1, a compile-time
// constant of (u, p); each warp computes its own planes in the order of the original, the chains are added at the end exactly as in `gemv_q2`.
// ---------------------------------------------------------------------------------------------------------------------------------------------
template <int C>
__device__ __forceinline__ float q2_chain(const uint32_t* __restrict__ tab, const uint32_t* __restrict__ wr, const __half* __restrict__ sc,
                                          const int8_t* __restrict__ qs, const float* __restrict__ ad, int groups, int full_groups, int tail, int nb32, int j) {
    float acc = 0.f;
    for (int g = 0; g < groups; ++g) {
        uint32_t words[4];
        load_group(wr, g, full_groups, tail, j, words);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint32_t word = words[u];
            const uint32_t t0 = tab[word & 0xffu], t1 = tab[(word >> 8) & 0xffu], t2 = tab[(word >> 16) & 0xffu], t3 = tab[word >> 24];
#pragma unroll
            for (int p = 0; p < 5; ++p) {
                if ((((5 * u + p) >> 1) & 1) != C) continue;
                const int s = 5 * (4 * g + u) + p;
                if (s < nb32) {
                    uint32_t dig;
                    if (p < 4) {
                        const uint32_t u01 = __byte_perm(t0, t1, p | ((p + 4) << 4));
                        const uint32_t u23 = __byte_perm(t2, t3, p | ((p + 4) << 4));
                        dig = __byte_perm(u01, u23, 0x5410);
                    } else {
                        dig = __vsub4(((((word & 0xffu) * 781u) >> 16)) | (((((word >> 8) & 0xffu) * 781u) >> 16) << 8) | (((((word >> 16) & 0xffu) * 781u) >> 16) << 16) |
                                      (((word >> 24) * 781u >> 16) << 24), 0x01010101u);
                    }
                    const int aw = *reinterpret_cast<const int*>(qs + s * 32 + 4 * j);
                    const int lane_sum = __dp4a(static_cast<int>(dig), aw, 0);
                    const float dd = __fmul_rn(__half2float(sc[s >> 1]), ad[s]);
                    acc = __fmaf_rn(dd, __int2float_rn(lane_sum), acc);
                }
            }
        }
    }
    return acc;
}

// 8 warps per block: warp pairs (2 w, 2 w + 1) share four rows, the even warp computes chain 0, the odd one chain 1; a block covers 16 rows.
extern "C" __global__ void __launch_bounds__(256) gemv_q2_split(const uint32_t* __restrict__ w, const __half* __restrict__ wsc, const int8_t* __restrict__ qs,
                                                                const float* __restrict__ ad, float* __restrict__ out, int rows, int cols) {
    __shared__ uint32_t tab[256];
    __shared__ float chain1[4][32];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = digits_word_q2(i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = lane & 7, c = warp & 1, wg = warp >> 1;
    const int row = (blockIdx.x * 4 + wg) * 4 + (lane >> 3);
    const int rc = min(row, rows - 1);
    const int blocks_per_row = (cols + 159) / 160, nb32 = cols / 32, sb = cols / 64;
    const int groups = (blocks_per_row + 3) >> 2, full_groups = blocks_per_row >> 2, tail = blocks_per_row & 3;
    const uint32_t* wr = w + (size_t)rc * groups * 32;  // rows are padded to whole groups
    const __half* sc = wsc + (size_t)rc * sb;
    const float acc = c == 0 ? q2_chain<0>(tab, wr, sc, qs, ad, groups, full_groups, tail, nb32, j) : q2_chain<1>(tab, wr, sc, qs, ad, groups, full_groups, tail, nb32, j);
    if (c == 1) chain1[wg][lane] = acc;
    __syncthreads();
    if (c == 0) {
        const float l = __fadd_rn(acc, chain1[wg][lane]);
        const float r = __fadd_rn(l, __shfl_down_sync(0xffffffffu, l, 4, 8));
        const float y = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2, 8));
        const float z = __fadd_rn(y, __shfl_down_sync(0xffffffffu, y, 1, 8));
        if (j == 0 && row < rows) out[row] = z;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// RMSNorm of `gridDim.x` rows of `cols` values (one block per row, 256 threads); the rows start `in_stride` / `out_stride` values apart. In place is allowed
// (out == x with equal offsets and strides).
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) rms_norm_rows(const float* x, const float* __restrict__ w_all, float* out, int cols, float eps,
                                                                int x_off, int out_off, int in_stride, int out_stride, int w_stride) {
    const float* w = w_all + (size_t)blockIdx.x * w_stride;
    __shared__ double red[8];
    __shared__ float s_scale;
    const float* xr = x + x_off + (size_t)blockIdx.x * in_stride;
    float* orow = out + out_off + (size_t)blockIdx.x * out_stride;
    double ss = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += static_cast<double>(__fmul_rn(xr[i], xr[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int k = 0; k < 8; ++k) t += red[k];
        const float mean = static_cast<float>(t / static_cast<double>(cols));
        s_scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();
    const float scale = s_scale;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Rotary embedding and KV append for the f32 result of the stacked q, k, v projection (rows q | k | v), as rope_kv in layer.cu without the post-scaling.
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) rope_kv_f32(const float* __restrict__ qkv, const float* __restrict__ cs_all, const int* __restrict__ pos_dev,
                                                              float* __restrict__ q_out, __half* __restrict__ Kc, __half* __restrict__ Vc, int q_heads,
                                                              int kv_heads, int head_dim, int rope_dims, int neox, int cache_len) {
    const int pos = pos_dev[0];
    const float* cs = cs_all + (size_t)pos * rope_dims;
    const int q_rows = q_heads * head_dim, kv_rows = kv_heads * head_dim;
    const int off = neox ? rope_dims / 2 : 1, stride = neox ? 1 : 2;
    const int half_pairs = rope_dims / 2;
    for (int idx = threadIdx.x; idx < (q_heads + kv_heads) * half_pairs; idx += blockDim.x) {
        const int head = idx / half_pairs, i = idx % half_pairs;
        const bool is_q = head < q_heads;
        const int base = head * head_dim + i * stride;
        const float x0 = qkv[base], x1 = qkv[base + off];
        const float c = cs[2 * i], sn = cs[2 * i + 1];
        const float o0 = __fmaf_rn(x0, c, -__fmul_rn(x1, sn));
        const float o1 = __fmaf_rn(x0, sn, __fmul_rn(x1, c));
        if (is_q) {
            q_out[base] = o0;
            q_out[base + off] = o1;
        } else {
            const int kvh = head - q_heads, e = i * stride;
            const size_t at = ((size_t)kvh * cache_len + pos) * head_dim;
            Kc[at + e] = __float2half_rn(o0);
            Kc[at + e + off] = __float2half_rn(o1);
        }
    }
    const int tail = head_dim - rope_dims;
    for (int idx = threadIdx.x; idx < (q_heads + kv_heads) * tail; idx += blockDim.x) {
        const int head = idx / tail, e = rope_dims + idx % tail;
        const bool is_q = head < q_heads;
        const float v = qkv[head * head_dim + e];
        if (is_q) q_out[head * head_dim + e] = v;
        else Kc[((size_t)(head - q_heads) * cache_len + pos) * head_dim + e] = __float2half_rn(v);
    }
    for (int idx = threadIdx.x; idx < kv_rows; idx += blockDim.x) {
        const int kvh = idx / head_dim, e = idx % head_dim;
        Vc[((size_t)kvh * cache_len + pos) * head_dim + e] = __float2half_rn(qkv[q_rows + kv_rows + idx]);
    }
}

// out[i] = silu(gate[i]) * up[i] for gate_up = [gate (n) | up (n)] (n a multiple of 8, the vector part of the CPU routine).
extern "C" __global__ void __launch_bounds__(256) swiglu_f32(const float* __restrict__ gate_up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmul_rn(ggml_silu(gate_up[i]), gate_up[n + i]);
}

// x[i] = x[i] + r[i]
extern "C" __global__ void __launch_bounds__(256) add_f32(float* __restrict__ x, const float* __restrict__ r, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], r[i]);
}
