// Projection kernel: exact int32 GEMV of a velurex-encoded ternary matrix (five trits per byte) with an int8 activation vector.
//
// Weight layout ("planar", the layout of `quintrit_core::pack::pack_matrix`): a row consists of blocks of 160 trits = 32 bytes;
// byte j of block b holds the trits of columns b*160 + p*32 + j for the planes p = 0..4, as d0 + 3*d1 + 9*d2 + 28*d3 + 84*d4 with
// the digits d = trit + 1 in 0..2. A column count that is not a multiple of 160 is padded with zero trits. The result of a row is
// therefore sum(trit * act) = acc - sum(act), where acc is accumulated with the digits and `act_sum` is the sum of all activations.
//
// Activations come permuted into 8-byte entries, one per weight byte-column: bytes 0..3 are the activations of the planes 0..3
// (one 32-bit word for `dp4a`) and byte 4 is the activation of plane 4. `act_perm` holds blocks_per_row * 32 entries (plus padding).
//
// One warp handles R rows at once. Each lane reads one 32-bit word (four byte-columns) of every row per step, so consecutive lanes
// read consecutive words and a warp load is one contiguous 128-byte line. The digit table lives in shared memory (256 entries); with
// D4 it holds only d0..d3 as one 32-bit word and the fifth digit is computed as (b * 781) >> 16 (exact for every byte of the encoding).
//
// Template flags (the kernel instantiations are listed at the end of the file):
//  PF   the first weight word of every row is loaded before the prologue, so the DRAM latency overlaps with it;
//  ACTG activations are read through the L1/L2 caches instead of being staged in shared memory;
//  EP   0: plain result, 1: gated activation epilogue (gate and up rows interleaved), 2: rotary embedding and KV append epilogue.
//
// The row count must be a multiple of R * WARPS (the host chooses a configuration that tiles the matrix).
#include "ggml_silu.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ int dp4a_s(uint32_t a, uint32_t b, int c) { return __dp4a(static_cast<int>(a), static_cast<int>(b), c); }

// Shared-memory swizzle of the staged activations: entry e is stored at e + (e >> 2), so the 16 lanes of a half-warp fall on 16
// distinct bank pairs.
__host__ __device__ constexpr int swz(int e) { return e + (e >> 2); }

__device__ __forceinline__ uint32_t digits_word(int i) {
    if (i > 250) return 0;
    const int d4 = i / 84, r4 = i - 84 * d4;
    const int d3 = r4 / 28, r3 = r4 - 28 * d3;
    const int d2 = r3 / 9, r2 = r3 - 9 * d2;
    const int d1 = r2 / 3, d0 = r2 - 3 * d1;
    return d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);
}

// Arguments of the epilogues; one layout for all kernels (mirrored by `EpilogueArgs` in quintrit-cuda, 128 bytes).
struct EpArgs {
    const float* act_scale;  // scale of the quantized activations (one value)
    float ws_gate, ws_up;    // gated epilogue: weight scales of the gate and up matrices
    float* v;                // gated epilogue: output of the activation, one value per gate/up row pair
    int* cells;              // gated epilogue: block maximum of |v| goes to cells[par] (atomicMax on the float bits); cells[par ^ 1] is reset here
    int par, activation;     // activation: 0 = SiLU (SwiGLU), 1 = squared ReLU
    int* sum_out;            // gated epilogue: activation sum of the next quantization, reset here
    float* q_out;            // rope epilogue: rotated query, F32
    __half* Kw;              // rope epilogue: key cache [kv_head][L_alloc][head_dim], F16
    __half* Vw;              // rope epilogue: value cache, same layout
    const float* cs_all;     // rope epilogue: cos/sin pairs, rope_dims floats per position (cos at 2i, sin at 2i + 1)
    const int* pos_dev;      // rope epilogue: position of the token, read from device memory (position-independent CUDA graph)
    float ws_q, ws_k, ws_v;  // rope epilogue: weight scales of q, k, v
    int q_rows, kv_rows;     // rope epilogue: rows of the q part and of each of the k and v parts of the stacked matrix
    int head_dim, rope_dims; // rope epilogue: head dimension and the number of rotated coordinates per head (even, <= head_dim)
    int L_alloc;             // rope epilogue: allocated length of the KV cache (positions)
    int pad_;
};
static_assert(sizeof(EpArgs) == 128, "EpArgs must match quintrit-cuda's EpilogueArgs");
static_assert(offsetof(EpArgs, sum_out) == 40 && offsetof(EpArgs, cs_all) == 72 && offsetof(EpArgs, ws_q) == 88 && offsetof(EpArgs, L_alloc) == 116,
              "EpArgs layout");

template <int R, int WARPS, bool PF, bool ACTG, int EP, bool D4, int UNR>
__device__ __forceinline__ void gemv_body(const uint32_t* __restrict__ w, const uint2* __restrict__ act_perm, const int* __restrict__ act_sum,
                                          int* __restrict__ out, int blocks_per_row, const EpArgs& ep) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int act_entries = blocks_per_row * 32;
    uint2* act_s = reinterpret_cast<uint2*>(smem_raw);
    uint2* tab = ACTG ? act_s : act_s + swz(act_entries) + 1;
    uint32_t* tab4 = reinterpret_cast<uint32_t*>(tab);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WARPS + warp) * R;
    const int words_per_row = blocks_per_row * 8;
    const uint32_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = w + (size_t)(row0 + r) * words_per_row;

    uint32_t first[R];
    if (PF) {
#pragma unroll
        for (int r = 0; r < R; ++r) first[r] = __ldg(wr[r] + lane);
    }

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (D4) tab4[i] = digits_word(i);
        else tab[i] = make_uint2(digits_word(i), i <= 250 ? static_cast<uint32_t>(i / 84) : 0u);
    }
    if (!ACTG) {
        for (int i = threadIdx.x; i < act_entries; i += blockDim.x) act_s[swz(i)] = act_perm[i];
    }
    __syncthreads();

    int acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0;

#pragma unroll UNR
    for (int it = lane; it < words_per_row; it += 32) {
        uint32_t wd[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wd[r] = (PF && it == lane) ? first[r] : __ldg(wr[r] + it);
        const int e0 = it * 4;  // entry index of the first byte-column of this word (block * 32 + j)
        uint2 a[4];
        if (ACTG) {
            const uint4 lo = __ldg(reinterpret_cast<const uint4*>(act_perm + e0));
            const uint4 hi = __ldg(reinterpret_cast<const uint4*>(act_perm + e0) + 1);
            a[0] = make_uint2(lo.x, lo.y); a[1] = make_uint2(lo.z, lo.w);
            a[2] = make_uint2(hi.x, hi.y); a[3] = make_uint2(hi.z, hi.w);
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) a[k] = act_s[swz(e0 + k)];
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int a4 = static_cast<int>(static_cast<int8_t>(a[k].y & 0xffu));
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t b = (wd[r] >> (8 * k)) & 0xffu;
                if (D4) {
                    acc[r] = dp4a_s(tab4[b], a[k].x, acc[r]);
                    acc[r] += static_cast<int>((b * 781u) >> 16) * a4;
                } else {
                    const uint2 e = tab[b];
                    acc[r] = dp4a_s(e.x, a[k].x, acc[r]);
                    acc[r] += static_cast<int>(e.y) * a4;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        if (EP == 0 && lane == 0) out[row0 + r] = acc[r] - act_sum[0];
    }

    if (EP == 2) {
        // Rotary embedding and KV append. Rows row0 + 2*pr and row0 + 2*pr + 1 are one rotation pair (interleaved pairing; R is even and
        // row0 is a multiple of R, so a pair never straddles two warps).
        if (lane == 0) {
            const float s = ep.act_scale[0];
            const int pos = ep.pos_dev[0];
            const float* cs = ep.cs_all + static_cast<size_t>(pos) * ep.rope_dims;
#pragma unroll
            for (int pr = 0; pr < R / 2; ++pr) {
                const int row = row0 + 2 * pr;
                const float r0 = static_cast<float>(acc[2 * pr] - act_sum[0]), r1 = static_cast<float>(acc[2 * pr + 1] - act_sum[0]);
                if (row < ep.q_rows + ep.kv_rows) {
                    const bool is_q = row < ep.q_rows;
                    const int local = is_q ? row : row - ep.q_rows;
                    const int e = local % ep.head_dim;
                    const float ps = (is_q ? ep.ws_q : ep.ws_k) / s;
                    const float x0 = r0 * ps, x1 = r1 * ps;
                    float o0 = x0, o1 = x1;
                    if (e < ep.rope_dims) {
                        const float c = cs[e], sn = cs[e + 1];
                        o0 = fmaf(x0, c, -(x1 * sn));
                        o1 = fmaf(x0, sn, x1 * c);
                    }
                    if (is_q) {
                        ep.q_out[row] = o0;
                        ep.q_out[row + 1] = o1;
                    } else {
                        const size_t at = ((size_t)(local / ep.head_dim) * ep.L_alloc + pos) * ep.head_dim + e;
                        ep.Kw[at] = __float2half(o0);
                        ep.Kw[at + 1] = __float2half(o1);
                    }
                } else {
                    const int local = row - ep.q_rows - ep.kv_rows, e = local % ep.head_dim;
                    const float ps = ep.ws_v / s;
                    const size_t at = ((size_t)(local / ep.head_dim) * ep.L_alloc + pos) * ep.head_dim + e;
                    ep.Vw[at] = __float2half(r0 * ps);
                    ep.Vw[at + 1] = __float2half(r1 * ps);
                }
            }
        }
    }

    if (EP == 1) {
        // Gated activation: rows 2i and 2i + 1 are gate_i and up_i; v_i = act(gate_i) * up_i, and the maximum of |v| over the whole
        // matrix goes to a global cell with one atomicMax (the activation scale of the following quantization).
        __shared__ float red[WARPS];
        float m = 0.f;
        if (lane == 0) {
            const float s = ep.act_scale[0];
            const float psg = ep.ws_gate / s, psu = ep.ws_up / s;
#pragma unroll
            for (int pr = 0; pr < R / 2; ++pr) {
                const float g = static_cast<float>(acc[2 * pr] - act_sum[0]) * psg, u = static_cast<float>(acc[2 * pr + 1] - act_sum[0]) * psu;
                float v;
                if (ep.activation == 0) {
                    v = ggml_silu(g) * u;
                } else {
                    const float p = g > 0.0f ? g : 0.0f;
                    v = __fmul_rn(__fmul_rn(p, p), u);
                }
                ep.v[(row0 >> 1) + pr] = v;
                m = fmaxf(m, fabsf(v));
            }
            red[warp] = m;
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            ep.cells[ep.par ^ 1] = 0;
            ep.sum_out[0] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = red[0];
#pragma unroll
            for (int i = 1; i < WARPS; ++i) t = fmaxf(t, red[i]);
            atomicMax(ep.cells + ep.par, __float_as_int(t));
        }
    }
}

// One entry point per configuration and epilogue; the name is gemv_<tag>_<plain|gated|rope> (see GemvConfig::tag in quintrit-cuda).
#define GEMV_ENTRY(TAG, R, W, PF, G, D4, EPNAME, EP)                                                                                       \
    extern "C" __global__ void __launch_bounds__(32 * W) gemv_##TAG##_##EPNAME(                                                            \
        const uint32_t* __restrict__ w, const uint2* __restrict__ act_perm, const int* __restrict__ act_sum, int* __restrict__ out,        \
        int blocks_per_row, EpArgs ep) {                                                                                                   \
        gemv_body<R, W, PF, G, EP, D4, 2>(w, act_perm, act_sum, out, blocks_per_row, ep);                                                  \
    }
#define GEMV_CONFIG(TAG, R, W, PF, G, D4)         \
    GEMV_ENTRY(TAG, R, W, PF, G, D4, plain, 0)    \
    GEMV_ENTRY(TAG, R, W, PF, G, D4, gated, 1)    \
    GEMV_ENTRY(TAG, R, W, PF, G, D4, rope, 2)

// The configurations of quintrit-cuda's profile table (src/profile.rs, ALL_CONFIGS).
GEMV_CONFIG(r4_w8_pf, 4, 8, true, false, false)
GEMV_CONFIG(r4_w8_g, 4, 8, false, true, false)
GEMV_CONFIG(r2_w4_g_d4, 2, 4, false, true, true)
GEMV_CONFIG(r4_w4_g_d4, 4, 4, false, true, true)
GEMV_CONFIG(r2_w2_g_d4, 2, 2, false, true, true)
GEMV_CONFIG(r2_w1_g_d4, 2, 1, false, true, true)
