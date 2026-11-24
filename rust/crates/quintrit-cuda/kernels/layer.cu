// Kernels of a decoder layer and of the output head, written to follow the CPU runtime (quintrit-runtime) operation by operation so that their results
// are bit-identical to it:
//  * rms_quant: RMSNorm with the sum of the f32-rounded squares accumulated in double, scale = 1 / sqrt(mean + eps) in float, y = (x * scale) * gamma, then the
//    activation scale s = 127 / max|y| and q = round(y * s) (half away from zero) clamped to [-128, 127], as quantize_activations_int8_with_range. The
//    quantized values are written in the permuted 8-byte layout the projection kernel reads (see gemv.cu). The residual add of the previous projection
//    can be done first: h = h + float(delta) * (ws / s_prev), with the two roundings of the reference (no fused multiply-add).
//  * rope_kv: rotary embedding (interleaved or half-split pairing) with the fused multiply-adds of the reference, the key and value appended to the F16 cache.
//  * attention: the flash attention of the reference runtime (ggml_compute_forward_flash_attn_ext_f16_one_chunk as bitnet.cpp runs it), see attention_body.
//  * final_norm and lm_head: F16 output head in the operation order of ggml_vec_dot_f16.
#include "glibc_expf.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Norm and quantization
// ---------------------------------------------------------------------------------------------------------------------------------------------

// MAXPT is the number of vector elements one thread keeps in registers (cols <= MAXPT * blockDim.x when norm != 0); ys is shared memory for `cols` floats.
template <int MAXPT>
__device__ __forceinline__ void rq_body(float* __restrict__ x, const float* __restrict__ gamma, uint2* __restrict__ act_perm, int* __restrict__ act_sum,
                                        float* __restrict__ act_scale, int cols, float eps, int norm, const int* __restrict__ delta, float ws, float* ys) {
    // The scale of the previous quantization is read before this kernel overwrites it.
    const float ps_res = delta ? ws / act_scale[0] : 0.f;
    __shared__ double red_d[32];
    __shared__ float red_f[32];
    __shared__ int red_i[32];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
    float vreg[MAXPT], greg[MAXPT];
    float mx = 0.f;
    if (norm) {
        double ss = 0.0;
#pragma unroll
        for (int k = 0; k < MAXPT; ++k) {
            const int i = tid + k * blockDim.x;
            if (i < cols) {
                float v = x[i];
                greg[k] = gamma[i];
                if (delta) {
                    v = __fadd_rn(v, __fmul_rn(static_cast<float>(__ldcg(delta + i)), ps_res));
                    x[i] = v;
                }
                vreg[k] = v;
                ss += static_cast<double>(v * v);
            }
        }
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red_d[wid] = ss;
        __syncthreads();
        double t = lane < nw ? red_d[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        t = __shfl_sync(0xffffffffu, t, 0);
        const float mean = static_cast<float>(t / static_cast<double>(cols));
        const float scale = 1.0f / sqrtf(mean + eps);
#pragma unroll
        for (int k = 0; k < MAXPT; ++k) {
            const int i = tid + k * blockDim.x;
            if (i < cols) {
                const float y = (vreg[k] * scale) * greg[k];
                ys[i] = y;
                mx = fmaxf(mx, fabsf(y));
            }
        }
    } else {
        for (int i = tid; i < cols; i += blockDim.x) {
            const float y = x[i];
            ys[i] = y;
            mx = fmaxf(mx, fabsf(y));
        }
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
    if (lane == 0) red_f[wid] = mx;
    __syncthreads();
    float m = lane < nw ? red_f[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    m = __shfl_sync(0xffffffffu, m, 0);
    const float s = m > 0.f ? 127.0f / m : 0.0f;
    if (tid == 0) act_scale[0] = s;
    const int entries = ((cols + 159) / 160) * 32;
    int sum = 0;
    for (int e = tid; e < entries; e += blockDim.x) {
        int8_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int p = 0; p < 5; ++p) {
            const int i = (e >> 5) * 160 + p * 32 + (e & 31);
            if (i < cols) {
                const float r = roundf(ys[i] * s);
                q[p] = static_cast<int8_t>(max(-128, min(127, static_cast<int>(r))));
                sum += q[p];
            }
        }
        act_perm[e] = *reinterpret_cast<uint2*>(q);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) red_i[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        int t = lane < nw ? red_i[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) act_sum[0] = t;
    }
}

// L2 prefetch of the start of the weight matrix the next projection reads (the quantization runs on one block while the rest of the GPU would idle): the
// first `lines` cache lines of `rows` rows. It changes no result; it pays off only when the L2 cache can hold the lines (a field of the device profile).
__device__ __forceinline__ void prefetch_rows(const char* __restrict__ base, size_t row_bytes, int rows, int lines, int first_block, int blocks) {
    const int per_block = blockDim.x;
    for (int r = (blockIdx.x - first_block) * per_block + threadIdx.x; r < rows; r += blocks * per_block) {
        for (int l = 0; l < lines; ++l) asm volatile("prefetch.global.L2 [%0];" ::"l"(base + (size_t)r * row_bytes + 128 * l));
    }
}

// One quantizing block (block 0) and, with a prefetch, 23 more blocks that prefetch. norm != 0: y = rmsnorm(x) * gamma (512 threads, cols <= 512 * MAXPT);
// norm == 0: y = x (1024 threads). Dynamic shared memory: cols floats.
#define RMS_QUANT_ENTRY(MAXPT)                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(1024) rms_quant_p##MAXPT(                                                                         \
        float* __restrict__ x, const float* __restrict__ gamma, uint2* __restrict__ act_perm, int* __restrict__ act_sum,                           \
        float* __restrict__ act_scale, int cols, float eps, int norm, const int* __restrict__ delta, float ws, const char* __restrict__ pf_base,   \
        size_t pf_row_bytes, int pf_rows, int pf_lines) {                                                                                          \
        if (blockIdx.x != 0) {                                                                                                                     \
            prefetch_rows(pf_base, pf_row_bytes, pf_rows, pf_lines, 1, gridDim.x - 1);                                                             \
            return;                                                                                                                                \
        }                                                                                                                                          \
        extern __shared__ float ys[];                                                                                                              \
        rq_body<MAXPT>(x, gamma, act_perm, act_sum, act_scale, cols, eps, norm, delta, ws, ys);                                                    \
    }
RMS_QUANT_ENTRY(6)   // cols <= 3072
RMS_QUANT_ENTRY(8)   // cols <= 4096
RMS_QUANT_ENTRY(14)  // cols <= 7168

// Quantization of a vector whose maximum was collected by the gated epilogue of the projection kernel (cells[par], float bits): many blocks, one thread per
// permuted entry, the activation sum by atomicAdd (exact integers). Block 0 writes the scale. `act_sum` must have been zeroed by the epilogue.
extern "C" __global__ void __launch_bounds__(256) quant_cells(const float* __restrict__ v, uint2* __restrict__ act_perm, int* __restrict__ act_sum,
                                                              float* __restrict__ act_scale, const int* __restrict__ cells, int par, int cols,
                                                              const char* __restrict__ pf_base, size_t pf_row_bytes, int pf_rows, int pf_lines,
                                                              int n_quant_blocks) {
    if (blockIdx.x >= n_quant_blocks) {  // extra blocks: L2 prefetch of the next matrix
        prefetch_rows(pf_base, pf_row_bytes, pf_rows, pf_lines, n_quant_blocks, gridDim.x - n_quant_blocks);
        return;
    }
    __shared__ int red[8];
    const float m = __int_as_float(cells[par]);
    const float s = m > 0.f ? 127.0f / m : 0.0f;
    if (blockIdx.x == 0 && threadIdx.x == 0) act_scale[0] = s;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int entries = ((cols + 159) / 160) * 32;
    int sum = 0;
    if (e < entries) {
        int8_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int p = 0; p < 5; ++p) {
            const int i = (e >> 5) * 160 + p * 32 + (e & 31);
            if (i < cols) {
                const float r = roundf(v[i] * s);
                q[p] = static_cast<int8_t>(max(-128, min(127, static_cast<int>(r))));
                sum += q[p];
            }
        }
        act_perm[e] = *reinterpret_cast<uint2*>(q);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        int t = threadIdx.x < 8 ? red[threadIdx.x] : 0;
        for (int o = 4; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) atomicAdd(act_sum, t);
    }
}

// h += float(delta) * (ws / act_scale), the product rounded before the add (no fused multiply-add).
extern "C" __global__ void __launch_bounds__(1024) residual_add(float* __restrict__ h, const int* __restrict__ delta, const float* __restrict__ act_scale,
                                                                float ws, int n) {
    const float ps = ws / act_scale[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[i] = __fadd_rn(h[i], __fmul_rn(static_cast<float>(delta[i]), ps));
}

// Final norm: out = f16-rounded rmsnorm(h) * w, the F16-exact input of the output head. One block of 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) final_norm(const float* __restrict__ h, const float* __restrict__ w, float* __restrict__ out, int cols, float eps) {
    __shared__ double red[32];
    __shared__ float s_scale;
    double ss = 0.0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += static_cast<double>(h[i] * h[i]);
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int k = 0; k < (blockDim.x >> 5); ++k) t += red[k];
        s_scale = 1.0f / sqrtf(static_cast<float>(t / static_cast<double>(cols)) + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[i] = __half2float(__float2half((h[i] * s_scale) * w[i]));
}

// The reduction of ggml_vec_dot_f16_avx2 over the 32 lane accumulators of a warp (lane = 8 * accumulator + element of the 256-bit register).
__device__ __forceinline__ float ggml_dot_reduce(float acc, int lane) {
    const int l = lane & 7;
    const float a0 = __shfl_sync(0xffffffffu, acc, l), a1 = __shfl_sync(0xffffffffu, acc, l + 8);
    const float a2 = __shfl_sync(0xffffffffu, acc, l + 16), a3 = __shfl_sync(0xffffffffu, acc, l + 24);
    const float r = (a0 + a2) + (a1 + a3);
    const float t0 = __shfl_sync(0xffffffffu, r, 0) + __shfl_sync(0xffffffffu, r, 4);
    const float t1 = __shfl_sync(0xffffffffu, r, 1) + __shfl_sync(0xffffffffu, r, 5);
    const float t2 = __shfl_sync(0xffffffffu, r, 2) + __shfl_sync(0xffffffffu, r, 6);
    const float t3 = __shfl_sync(0xffffffffu, r, 3) + __shfl_sync(0xffffffffu, r, 7);
    return (t0 + t1) + (t2 + t3);
}

// Output head: one warp per vocabulary row, x (F16-exact floats) against the F16 matrix W [vocab][K], K a multiple of 32.
extern "C" __global__ void __launch_bounds__(256) lm_head(const __half* __restrict__ W, const float* __restrict__ x, float* __restrict__ out, int vocab, int K) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= vocab) return;
    const __half* wr = W + (size_t)row * K;
    float acc = 0.f;
    for (int b = 0; b < K / 32; ++b) acc = fmaf(__half2float(__float2half(x[32 * b + lane])), __half2float(wr[32 * b + lane]), acc);
    const float dot = ggml_dot_reduce(acc, lane);
    if (lane == 0) out[row] = dot;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Rotary embedding and KV append (standalone variant; the projection kernel has the interleaved variant as an epilogue)
// ---------------------------------------------------------------------------------------------------------------------------------------------

// qkv: the int32 result of the stacked q, k, v projection (rows q | k | v). Post-scale (weight scale / act scale), rotation of the first rope_dims coordinates of
// every q and k head (neox == 0: pairs (2i, 2i + 1); neox != 0: pairs (i, i + rope_dims / 2)), key and value appended as F16 at position *pos_dev of
// [kv_head][cache_len][head_dim]. cs_all holds rope_dims floats per position (cos at 2i, sin at 2i + 1). One block.
extern "C" __global__ void __launch_bounds__(256) rope_kv(const int* __restrict__ qkv, const float* __restrict__ act_scale,
                                                          float ws_q, float ws_k, float ws_v, const float* __restrict__ cs_all, const int* __restrict__ pos_dev,
                                                          float* __restrict__ q_out, __half* __restrict__ Kc, __half* __restrict__ Vc, int q_heads, int kv_heads,
                                                          int head_dim, int rope_dims, int neox, int cache_len) {
    const float s = act_scale[0];
    const float ps_q = ws_q / s, ps_k = ws_k / s, ps_v = ws_v / s;
    const int pos = pos_dev[0];
    const float* cs = cs_all + (size_t)pos * rope_dims;
    const int q_rows = q_heads * head_dim, kv_rows = kv_heads * head_dim;
    const int off = neox ? rope_dims / 2 : 1, stride = neox ? 1 : 2;
    const int half_pairs = rope_dims / 2;
    for (int idx = threadIdx.x; idx < (q_heads + kv_heads) * half_pairs; idx += blockDim.x) {
        const int head = idx / half_pairs, i = idx % half_pairs;
        const bool is_q = head < q_heads;
        const int base = head * head_dim + i * stride;
        const float ps = is_q ? ps_q : ps_k;
        const float x0 = static_cast<float>(qkv[base]) * ps, x1 = static_cast<float>(qkv[base + off]) * ps;
        const float c = cs[2 * i], sn = cs[2 * i + 1];
        const float o0 = fmaf(x0, c, -(x1 * sn));
        const float o1 = fmaf(x0, sn, x1 * c);
        if (is_q) {
            q_out[base] = o0;
            q_out[base + off] = o1;
        } else {
            const int kvh = head - q_heads, e = i * stride;
            const size_t at = ((size_t)kvh * cache_len + pos) * head_dim;
            Kc[at + e] = __float2half(o0);
            Kc[at + e + off] = __float2half(o1);
        }
    }
    // Coordinates beyond the rotated width pass through (post-scaled only).
    const int tail = head_dim - rope_dims;
    for (int idx = threadIdx.x; idx < (q_heads + kv_heads) * tail; idx += blockDim.x) {
        const int head = idx / tail, e = rope_dims + idx % tail;
        const bool is_q = head < q_heads;
        const float v = static_cast<float>(qkv[head * head_dim + e]) * (is_q ? ps_q : ps_k);
        if (is_q) q_out[head * head_dim + e] = v;
        else Kc[((size_t)(head - q_heads) * cache_len + pos) * head_dim + e] = __float2half(v);
    }
    for (int idx = threadIdx.x; idx < kv_rows; idx += blockDim.x) {
        const int kvh = idx / head_dim, e = idx % head_dim;
        Vc[((size_t)kvh * cache_len + pos) * head_dim + e] = __float2half(static_cast<float>(qkv[q_rows + kv_rows + idx]) * ps_v);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------------------------------------------------------------------------
//
// One block (384 threads) per query head and chunk (grid heads x 4); q [heads][HD] F32 (rounded to F16 here), K and V [kv_head][cache_len][HD] F16, out
// [heads][HD] F32, L = *pos_dev + 1.
// The arithmetic is that of attend_head of quintrit-runtime, which reproduces the flash attention of the reference runtime:
//   score_p = dot_f16(q, k_p) * scale (ggml_vec_dot_f16 operation order); online softmax with the running maximum, ms / vs from glibc's expf;
//   S = fma(S, ms, vs); the value accumulator is F16: x = f16(x * ms) when the maximum grows, then x = f16(fma(v, vs, x)).
// The KV axis is padded: n_kv = min(cache_len, max(256, round up L to 256)). Below 512 one softmax runs over all positions; from there on the padded axis is cut
// into 4 chunks of ceil(n_kv / 4) positions, each with its own running maximum, S and F16 accumulator, and the chunk results are combined in chunk order:
//   m_new = max(m, m_c); acc = fma(acc, exp(m - m_new), float(x_c) * exp(m_c - m_new)); S = fma(S, exp(m - m_new), S_c * exp(m_c - m_new));
//   out = acc / S (or acc when S == 0); chunks with S_c == 0 are skipped.
// The chunks are independent, so each one runs in its own block: with an unsplit context only the block of chunk 0 works (the others return at once, so a
// launch serves every position); with a split axis every block writes its F16 accumulator, S and the maximum to a workspace and the block of a head that
// arrives last (an arrival counter, reset by that block so that a captured graph can be replayed) does the combination in chunk order.
// Parallel structure per chunk: the scores of its positions (independent, four lanes per position, no staging through shared memory), the prefix maximum (warp
// scan), ms / vs of every position (independent), and the chain over the positions per element of the head dimension (sequential per element, elements side by
// side, V tiles through cp.async).
// Shared memory: 2 * l_chunk floats (the longest chunk) plus one tile area. The head dimension HD is a multiple of 32 up to 256. The kernel is limited to 85
// registers per thread (two blocks of 384 threads per multiprocessor): with more registers the 48 blocks of a model with 12 heads would not fit next to each other on a
// device with 46 multiprocessors and the two that find no free multiprocessor would run after the others.

// Timing experiment (not for production): building with -DATTN_STOP=1, 2 or 3 (through QUINTRIT_CUDA_NVCC_FLAGS) leaves the kernel after the scores, after the
// prefix maximum and ms / vs, or after the chain; the results are wrong and only the time of the phases is meaningful.
// Threads per block: 384. The chain needs only HD / 2 of them (128 for the head dimension 256), the other warps work in the score phase; on the RTX 3070
// and the RTX 4060 384 are faster than 512 (-1.9 % / -3.2 % at 2048 / 3900 tokens on the 3070, -1.5 % / -2.6 % on the 4060), 256 and the sizes in between
// are slower. HD <= NT is required.
constexpr int NT = 384;
constexpr int ATTN_MAX_L = 7936;
constexpr int ATTN_SPLIT_MIN = 512;
constexpr int ATTN_PAD = 256;
constexpr int ATTN_CHUNKS = 4;

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;"); }

// Loads EPT consecutive F16 values (EPT * 2 bytes aligned) from shared memory as floats.
template <int EPT>
__device__ __forceinline__ void load_halves(const __half* p, float* v) {
    if constexpr (EPT == 2) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p));
        v[0] = f.x, v[1] = f.y;
    } else if constexpr (EPT == 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
    } else {
        const uint4 u = *reinterpret_cast<const uint4*>(p);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&u.z)), e = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
        v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y, v[4] = c.x, v[5] = c.y, v[6] = e.x, v[7] = e.y;
    }
}

// Elements of the head dimension per thread in the chain: two halve the shared-memory loads per element and give two independent chains per thread; four is as
// fast, eight and one are slower (RTX 3070).
constexpr int EPT = 2;

// F32 selects the arithmetic of upstream llama.cpp (query kept in F32, scores as ggml_vec_dot_f16_f32, F32 value accumulator) instead of the bitnet.cpp one.
template <int HD, bool F32>
__device__ __forceinline__ void attention_body(const __half* __restrict__ K, const __half* __restrict__ V, const float* __restrict__ q, float* __restrict__ out,
                                               float* __restrict__ part, unsigned* __restrict__ arrivals, int gqa, int cache_len, float scale,
                                               const int* __restrict__ pos_dev, int l_chunk) {
    constexpr int CPR = HD / 8;  // 16-byte chunks per row of K or V
    extern __shared__ __align__(16) float sm[];
    const int L = pos_dev[0] + 1;
    const int n_kv = min(cache_len, max(ATTN_PAD, (L + ATTN_PAD - 1) / ATTN_PAD * ATTN_PAD));
    const bool split = n_kv >= ATTN_SPLIT_MIN;
    const int c = blockIdx.y;
    if (!split && c > 0) return;  // one softmax over all positions: the block of chunk 0 does it
    const int chunk_len = split ? (n_kv + ATTN_CHUNKS - 1) / ATTN_CHUNKS : L;
    const int cs = c * chunk_len, ce = min(min(cs + chunk_len, n_kv), L);
    const int n = max(ce - cs, 0);
    float* vsa = sm;
    float* msa = sm + l_chunk;
    float2* mv = reinterpret_cast<float2*>(sm);  // after the ms / vs phase: {ms, vs} of every position of the chunk
    __shared__ __align__(16) float qs[HD];
    __shared__ float warp_max[NT / 32];
    __shared__ bool is_last;
    __shared__ int tile_rescale[ATTN_MAX_L / ATTN_CHUNKS / 32 + 2];  // per tile of 32 positions: does any position grow the running maximum (ms != 1)?
    const int h = blockIdx.x, kvh = h / gqa, d = threadIdx.x, warp = d >> 5, lane = d & 31;
    float* my_part = part + (size_t)(h * ATTN_CHUNKS + c) * (HD + 2);
    float s_sum = 0.f, m_chunk = -INFINITY;
    float xs[EPT];  // the F16 accumulators of the EPT elements of this thread (as floats)
#pragma unroll
    for (int e = 0; e < EPT; ++e) xs[e] = 0.f;
    if (n > 0) {  // block-uniform: a chunk of masked cells only has S = 0 and takes no part in the combination
        if (d < HD) qs[d] = F32 ? q[h * HD + d] : __half2float(__float2half(q[h * HD + d]));
        if (d < ATTN_MAX_L / ATTN_CHUNKS / 32 + 2) tile_rescale[d] = 0;
        __syncthreads();
        __half* tilebase = reinterpret_cast<__half*>(sm + ((2 * l_chunk + 3) & ~3));

        // Scores of the positions of the chunk (vsa[p - cs]).
        {
            // Four lanes per position: lane t of the quad holds elements 8t..8t+7 of every block of 32 elements of the row, i.e. the accumulators acc[8t + e]
            // of the CPU's dot product (lane = 8 * accumulator + element of the 256-bit register), and sums them over the blocks in order.
            constexpr int NB = HD / 32;
            const int t = lane & 3, qd = lane >> 2;
            for (int pb = cs + warp * 8; pb < ce; pb += (NT / 32) * 8) {
                const int p = min(pb + qd, ce - 1);
                const uint4* kr = reinterpret_cast<const uint4*>(K + ((size_t)kvh * cache_len + p) * HD) + t;
                uint4 raw[NB];
#pragma unroll
                for (int b = 0; b < NB; ++b) raw[b] = __ldg(kr + 4 * b);
                float acc[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    const unsigned w[4] = {raw[b].x, raw[b].y, raw[b].z, raw[b].w};
                    float qv[8];
                    {
                        const float4 a = *reinterpret_cast<const float4*>(qs + 32 * b + 8 * t), c4 = *reinterpret_cast<const float4*>(qs + 32 * b + 8 * t + 4);
                        qv[0] = a.x, qv[1] = a.y, qv[2] = a.z, qv[3] = a.w, qv[4] = c4.x, qv[5] = c4.y, qv[6] = c4.z, qv[7] = c4.w;
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[j]));
                        acc[2 * j] = fmaf(qv[2 * j], f.x, acc[2 * j]);
                        acc[2 * j + 1] = fmaf(qv[2 * j + 1], f.y, acc[2 * j + 1]);
                    }
                }
                float r[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if constexpr (F32) {
                        const float s = acc[e] + __shfl_xor_sync(0xffffffffu, acc[e], 1);  // (a0 + a1) in lanes 0 and 1, (a2 + a3) in lanes 2 and 3
                        r[e] = s + __shfl_xor_sync(0xffffffffu, s, 2);                     // (a0 + a1) + (a2 + a3)
                    } else {
                        const float s = acc[e] + __shfl_xor_sync(0xffffffffu, acc[e], 2);  // (a0 + a2) in lanes 0 and 2, (a1 + a3) in lanes 1 and 3
                        r[e] = s + __shfl_xor_sync(0xffffffffu, s, 1);                     // (a0 + a2) + (a1 + a3)
                    }
                }
                float dot;
                if constexpr (F32) dot = ((r[0] + r[4]) + (r[2] + r[6])) + ((r[1] + r[5]) + (r[3] + r[7]));  // ggml_vec_dot_f16_f32
                else dot = ((r[0] + r[4]) + (r[1] + r[5])) + ((r[2] + r[6]) + (r[3] + r[7]));
                if (t == 0 && pb + qd < ce) vsa[pb + qd - cs] = __fmul_rn(dot, scale);
            }
        }
        __syncthreads();
#if defined(ATTN_STOP) && ATTN_STOP == 1
        if (n > 0) return;
#endif

        // Prefix maximum of the scores of the chunk.
        const int per = (n + NT - 1) / NT;
        const int lo = min(d * per, n), hi = min(lo + per, n);
        float m = -INFINITY;
        for (int p = lo; p < hi; ++p) {
            const float sc = vsa[p];
            if (sc > m) m = sc;
            msa[p] = m;
        }
        float inc = m;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float other = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o && other > inc) inc = other;
        }
        float excl = __shfl_up_sync(0xffffffffu, inc, 1);
        if (lane == 0) excl = -INFINITY;
        if (lane == 31) warp_max[warp] = inc;
        __syncthreads();
        float carry = excl;
        for (int w = 0; w < warp; ++w) carry = fmaxf(carry, warp_max[w]);
        for (int p = lo; p < hi; ++p) msa[p] = fmaxf(msa[p], carry);
        __syncthreads();
        m_chunk = msa[n - 1];
        // ms / vs of every position of the chunk.
        float loc_ms[ATTN_MAX_L / ATTN_CHUNKS / NT + 1], loc_vs[ATTN_MAX_L / ATTN_CHUNKS / NT + 1];
        for (int p = d, i = 0; p < n; p += NT, ++i) {
            const float score = vsa[p], pm = msa[p];
            const float prev = p == 0 ? -INFINITY : msa[p - 1];
            float ms = 1.0f, vs = 1.0f;
            if (score > prev) ms = glibc_expf(prev - pm);
            else vs = glibc_expf(score - pm);
            loc_ms[i] = ms;
            loc_vs[i] = vs;
            if (ms != 1.0f) tile_rescale[p >> 5] = 1;
        }
        __syncthreads();
        for (int p = d, i = 0; p < n; p += NT, ++i) {
            mv[p] = make_float2(loc_ms[i], loc_vs[i]);
        }
        __syncthreads();
#if defined(ATTN_STOP) && ATTN_STOP == 2
        return;
#endif
        // The chain over the positions of the chunk, V tiles of 32 positions double-buffered through cp.async.
        const int ntiles = (n + 31) >> 5;
        constexpr int CTW = ((HD / EPT + 31) / 32) * 32;  // threads of the warps that carry elements of the head dimension (the chain loop runs on them only)
        auto issue = [&](int tile, int buf) {
#pragma unroll
            for (int j = 0; j < (32 * CPR + CTW - 1) / CTW; ++j) {
                const int cc = d + CTW * j;
                if (cc < 32 * CPR) {
                    const int r = cc / CPR, c16 = cc % CPR;
                    const int pos = min(cs + tile * 32 + r, ce - 1);
                    cp_async16(tilebase + buf * 32 * HD + r * HD + c16 * 8, V + ((size_t)kvh * cache_len + pos) * HD + c16 * 8);
                }
            }
            cp_async_commit();
        };
        if (d < CTW) issue(0, 0);
        if (d < CTW)
        for (int i = 0; i < ntiles; ++i) {
            cp_async_wait_all();
            asm volatile("bar.sync 1, %0;" ::"n"(CTW) : "memory");  // the warps of the chain only
            if (i + 1 < ntiles) issue(i + 1, (i + 1) & 1);
            if (d < HD / EPT) {
                const __half* vb = tilebase + (i & 1) * 32 * HD + EPT * d;
                const int nt = min(32, n - i * 32);
                const float2* mvt = mv + i * 32;
                if (nt == 32 && !tile_rescale[i]) {
                    // No position of the tile changes the maximum: ms == 1, so S = fma(S, 1, vs) = S + vs and the rescale f16(x * 1) is the identity.
#pragma unroll
                    for (int t = 0; t < 32; ++t) {
                        const float vs = mvt[t].y;
                        float v[EPT];
                        load_halves<EPT>(vb + t * HD, v);
                        s_sum = s_sum + vs;
#pragma unroll
                        for (int e = 0; e < EPT; ++e) {
                            if constexpr (F32) xs[e] = fmaf(v[e], vs, xs[e]);
                            else xs[e] = __half2float(__float2half(fmaf(v[e], vs, xs[e])));
                        }
                    }
                } else {
#pragma unroll 4
                    for (int t = 0; t < nt; ++t) {
                        const float ms = mvt[t].x, vs = mvt[t].y;
                        float v[EPT];
                        load_halves<EPT>(vb + t * HD, v);
                        s_sum = fmaf(s_sum, ms, vs);
#pragma unroll
                        for (int e = 0; e < EPT; ++e) {
                            if constexpr (F32) {
                                if (ms != 1.0f) xs[e] = __fmul_rn(xs[e], ms);
                                xs[e] = fmaf(v[e], vs, xs[e]);
                            } else {
                                if (ms != 1.0f) xs[e] = __half2float(__float2half(xs[e] * ms));
                                xs[e] = __half2float(__float2half(fmaf(v[e], vs, xs[e])));
                            }
                        }
                    }
                }
            }
        }
    }
#if defined(ATTN_STOP) && ATTN_STOP == 3
    if (d < HD / EPT) out[h * HD + EPT * d] = xs[0] + s_sum;
    return;
#endif
    if (!split) {
        if (d < HD / EPT) {
            const float f = s_sum == 0.f ? 0.f : 1.0f / s_sum;
#pragma unroll
            for (int e = 0; e < EPT; ++e) out[h * HD + EPT * d + e] = xs[e] * f;
        }
        return;
    }
    // Publish the chunk and let the last block of the head combine the four in chunk order.
    if (d < HD / EPT) {
#pragma unroll
        for (int e = 0; e < EPT; ++e) my_part[EPT * d + e] = xs[e];
    }
    if (d == 0) {
        my_part[HD] = s_sum;
        my_part[HD + 1] = m_chunk;
    }
    __threadfence();
    __syncthreads();
    if (d == 0) is_last = atomicAdd(arrivals + h, 1u) == ATTN_CHUNKS - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (d == 0) arrivals[h] = 0;  // ready for the next launch
    if (d < HD) {
        const float* parts = part + (size_t)h * ATTN_CHUNKS * (HD + 2);
        // All loads first, then the scale factors (they depend on the running maximum only), then the chain of fused multiply-adds in chunk order.
        float s_c[ATTN_CHUNKS], m_c[ATTN_CHUNKS], x_c[ATTN_CHUNKS];
#pragma unroll
        for (int k = 0; k < ATTN_CHUNKS; ++k) {
            const float* pk = parts + (size_t)k * (HD + 2);
            s_c[k] = __ldcg(pk + HD);
            m_c[k] = __ldcg(pk + HD + 1);
            x_c[k] = __ldcg(pk + d);
        }
        float m_fin = -INFINITY, scale_old[ATTN_CHUNKS], scale_new[ATTN_CHUNKS];
#pragma unroll
        for (int k = 0; k < ATTN_CHUNKS; ++k) {
            const float m_new = s_c[k] != 0.f ? fmaxf(m_fin, m_c[k]) : m_fin;  // an empty chunk (S is stored as 0) or a chunk without weight is skipped
            scale_old[k] = glibc_expf(m_fin - m_new);
            scale_new[k] = glibc_expf(m_c[k] - m_new);
            m_fin = m_new;
        }
        float s_fin = 0.f, acc_fin = 0.f;
#pragma unroll
        for (int k = 0; k < ATTN_CHUNKS; ++k) {
            if (s_c[k] != 0.f) {
                acc_fin = fmaf(acc_fin, scale_old[k], __fmul_rn(x_c[k], scale_new[k]));
                s_fin = fmaf(s_fin, scale_old[k], __fmul_rn(s_c[k], scale_new[k]));
            }
        }
        out[h * HD + d] = s_fin != 0.f ? acc_fin * (1.0f / s_fin) : acc_fin;
    }
}

#define ATTENTION_ENTRY(HD, SUFFIX, F32)                                                                                                            \
    extern "C" __global__ void __launch_bounds__(NT, 2) attention_h##HD##SUFFIX(const __half* __restrict__ K, const __half* __restrict__ V,         \
                                                                     const float* __restrict__ q, float* __restrict__ out,                       \
                                                                     float* __restrict__ part, unsigned* __restrict__ arrivals, int gqa,         \
                                                                     int cache_len, float scale, const int* __restrict__ pos_dev, int l_chunk) { \
        attention_body<HD, F32>(K, V, q, out, part, arrivals, gqa, cache_len, scale, pos_dev, l_chunk);                                             \
    }
ATTENTION_ENTRY(32, , false)
ATTENTION_ENTRY(64, , false)
ATTENTION_ENTRY(128, , false)
ATTENTION_ENTRY(256, , false)
ATTENTION_ENTRY(32, _f32, true)
ATTENTION_ENTRY(64, _f32, true)
ATTENTION_ENTRY(128, _f32, true)
ATTENTION_ENTRY(256, _f32, true)
