// Kernels for the Nemotron-H mixture-of-experts decoder (Mamba-2 layers, attention without rotary embedding, mixture-of-experts layers with block-scaled experts, a
// `Q8_0` output head). They follow the CPU runtime (`hybrid.rs`, `refsort.rs` of quintrit-runtime) operation by operation so that their results are bit-identical to
// it. Every floating-point operation is an explicit round-to-nearest intrinsic or an `fmaf`/`fma`; the exponential is glibc's `expf` (glibc_expf.cuh), `logf` is a port
// of glibc's function (the table and the coefficients are those of `__logf_data`).
#include "ggml_silu.cuh"
#include "glibc_expf.cuh"
#include "q2_words.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// The reduction of ggml_vec_dot_f32 over the 32 lane accumulators of a warp (see hybrid.cu).
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

__device__ __forceinline__ float sigmoid_dev(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, glibc_expf(-x))); }

// ---------------------------------------------------------------------------------------------------------------------------------------------
// logf of glibc (sysdeps/ieee754/flt-32/e_logf.c, the ARM optimized-routines algorithm: 16-entry table, cubic polynomial, double arithmetic). The reference CPU runs
// the FMA variant of glibc, in which the compiler contracts the products into fused multiply-adds; they are written out here.
// ---------------------------------------------------------------------------------------------------------------------------------------------
__device__ __constant__ uint64_t kLogfTab[32] = {
    0x3ff661ec79f8f3beull, 0xbfd57bf7808caadeull, 0x3ff571ed4aaf883dull, 0xbfd2bef0a7c06ddbull, 0x3ff49539f0f010b0ull, 0xbfd01eae7f513a67ull,
    0x3ff3c995b0b80385ull, 0xbfcb31d8a68224e9ull, 0x3ff30d190c8864a5ull, 0xbfc6574f0ac07758ull, 0x3ff25e227b0b8ea0ull, 0xbfc1aa2bc79c8100ull,
    0x3ff1bb4a4a1a343full, 0xbfba4e76ce8c0e5eull, 0x3ff12358f08ae5baull, 0xbfb1973c5a611cccull, 0x3ff0953f419900a7ull, 0xbfa252f438e10c1eull,
    0x3ff0000000000000ull, 0x0000000000000000ull, 0x3fee608cfd9a47acull, 0x3faaa5aa5df25984ull, 0x3feca4b31f026aa0ull, 0x3fbc5e53aa362eb4ull,
    0x3feb2036576afce6ull, 0x3fc526e57720db08ull, 0x3fe9c2d163a1aa2dull, 0x3fcbc2860d224770ull, 0x3fe886e6037841edull, 0x3fd1058bc8a07ee1ull,
    0x3fe767dcf5534862ull, 0x3fd4043057b6ee09ull};

__device__ float logf_glibc(float x) {
    const uint32_t ix0 = __float_as_uint(x);
    if (ix0 == 0x3f800000u) return 0.0f;
    uint32_t ix = ix0;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {  // x < 2^-126, inf or nan
        if (ix * 2 == 0) return -INFINITY;
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2 >= 0xff000000u) return __uint_as_float(0x7fc00000u);
        ix = __float_as_uint(__fmul_rn(x, 8388608.0f));  // subnormal: normalize
        ix -= 23u << 23;
    }
    const uint32_t tmp = ix - 0x3f330000u;
    const int i = (tmp >> (23 - 4)) % 16;
    const int k = static_cast<int32_t>(tmp) >> 23;
    const uint32_t iz = ix - (tmp & (0x1ffu << 23));
    const double invc = __longlong_as_double(static_cast<long long>(kLogfTab[2 * i]));
    const double logc = __longlong_as_double(static_cast<long long>(kLogfTab[2 * i + 1]));
    const double ln2 = __longlong_as_double(0x3fe62e42fefa39efll);
    const double a0 = __longlong_as_double(static_cast<long long>(0xbfd00ea348b88334ull)), a1 = __longlong_as_double(static_cast<long long>(0x3fd5575b0be00b6aull)),
                 a2 = __longlong_as_double(static_cast<long long>(0xbfdffffef20a4123ull));
    const double z = static_cast<double>(__uint_as_float(iz));
    const double r = __fma_rn(z, invc, -1.0);
    const double y0 = __fma_rn(static_cast<double>(k), ln2, logc);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(a1, r, a2);
    y = __fma_rn(a0, r2, y);
    y = __fma_rn(y, r2, __dadd_rn(y0, r));
    return static_cast<float>(y);
}

// Test probe: out[i] = logf(x[i]).
extern "C" __global__ void __launch_bounds__(256) probe_logf(const float* __restrict__ x, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf_glibc(x[i]);
}

// ggml_compute_softplus of the SSM scan: x above 20, logf(1 + expf(x)) otherwise.
__device__ __forceinline__ float softplus_scan(float x) { return x > 20.0f ? x : logf_glibc(__fadd_rn(1.0f, glibc_expf(x))); }

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Q8_0 projection: out[r] = dot_q8_0(row r, activations): eight lane sums of four products per block of 32, one accumulator per lane
// acc = fma(w_scale * a_scale, lane, acc) over all blocks, then ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)). Eight threads per row, four rows per warp.
// Weights: int8 codes [rows][cols] and F16 scales [rows][cols / 32]; the activations are the codes and F32 scales of q8_quant.
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) gemv_q8(const int8_t* __restrict__ w, const __half* __restrict__ wsc, const int8_t* __restrict__ qs,
                                                          const float* __restrict__ ad, float* __restrict__ out, int rows, int cols) {
    const int lane = threadIdx.x & 31, j = lane & 7;
    const int row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 4 + (lane >> 3);
    const int rc = min(row, rows - 1);
    const int nb = cols / 32;
    const int8_t* wr = w + (size_t)rc * cols + 4 * j;
    const __half* sc = wsc + (size_t)rc * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const int ww = *reinterpret_cast<const int*>(wr + 32 * b);
        const int aw = *reinterpret_cast<const int*>(qs + 32 * b + 4 * j);
        const float dd = __fmul_rn(__half2float(sc[b]), ad[b]);
        acc = __fmaf_rn(dd, __int2float_rn(__dp4a(ww, aw, 0)), acc);
    }
    const float r = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, 4, 8));
    const float y = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2, 8));
    const float z = __fadd_rn(y, __shfl_down_sync(0xffffffffu, y, 1, 8));
    if (j == 0 && row < rows) out[row] = z;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Mamba-2
// ---------------------------------------------------------------------------------------------------------------------------------------------

// Causal depthwise convolution of the channels a[a_off + c] with the state, the bias and SiLU (as ggml_ssm_conv): sum = fma chain over the taps, + bias. The state of
// a sequence that starts (position 0) is zero. conv_state: [ch][kernel - 1], w: [ch][kernel].
extern "C" __global__ void __launch_bounds__(256) mamba_conv(const float* __restrict__ a, int a_off, float* __restrict__ conv_state, const float* __restrict__ w,
                                                             const float* __restrict__ bias, float* __restrict__ out, const int* __restrict__ pos_dev, int ch,
                                                             int kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const int km1 = kernel - 1;
    const bool fresh = pos_dev[0] == 0;
    float* st = conv_state + (size_t)c * km1;
    const float input = a[a_off + c];
    float prev[8];
    for (int i = 0; i < km1; ++i) prev[i] = fresh ? 0.0f : st[i];
    float sum = 0.0f;
    for (int i = 0; i < kernel; ++i) sum = __fmaf_rn(i < km1 ? prev[i] : input, w[(size_t)c * kernel + i], sum);
    out[c] = ggml_silu(__fadd_rn(sum, bias[c]));
    for (int i = 0; i + 1 < km1; ++i) st[i] = prev[i + 1];
    st[km1 - 1] = input;
}

// The state update and output of one token of ggml_ssm_scan. One warp per (head h, channel i1 of the head): the state row of the channel (d_state values) is held in
// registers, lane t keeping the elements 32 c + t (the lane accumulators of ggml_vec_dot_f32).
//   sp = softplus(dt[h] + dt_bias[h]); da = exp(sp * A[h]); x_dt = x * sp; st = st * da + B * x_dt (two roundings); y = dot(st, C); y += x * D[h].
// conv holds the convolved [x (d_inner) | B (n_group * d_state) | C (n_group * d_state)]; the heads of a group share B and C. The state of a sequence that starts is zero.
extern "C" __global__ void __launch_bounds__(256) mamba_scan(float* __restrict__ state, const float* __restrict__ conv, const float* __restrict__ dt,
                                                             const float* __restrict__ dt_bias, const float* __restrict__ a_param, const float* __restrict__ d_param,
                                                             float* __restrict__ y, const int* __restrict__ pos_dev, int d_inner, int d_state, int n_group,
                                                             int n_head) {
    const int warp = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int hd = d_inner / n_head;
    const int h = warp / hd;
    if (h >= n_head) return;
    const int ii = warp;  // = h * hd + i1
    const float* bm = conv + d_inner;
    const float* cm = conv + d_inner + n_group * d_state;
    const int g = h / (n_head / n_group);
    const float* b = bm + g * d_state;
    const float* c = cm + g * d_state;
    const float sp = softplus_scan(__fadd_rn(dt[h], dt_bias[h]));
    const float da = glibc_expf(__fmul_rn(sp, a_param[h]));
    const float x = conv[ii];
    const float x_dt = __fmul_rn(x, sp);
    float* st = state + (size_t)ii * d_state;
    const bool fresh = pos_dev[0] == 0;
    const int chunks = d_state / 32;
    float acc = 0.0f;
    float s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < chunks) {
            const float e = fresh ? 0.0f : st[32 * k + lane];
            s[k] = __fadd_rn(__fmul_rn(e, da), __fmul_rn(b[32 * k + lane], x_dt));
            st[32 * k + lane] = s[k];
            acc = __fmaf_rn(s[k], c[32 * k + lane], acc);
        }
    }
    const float sum = dot_reduce_f32(acc, lane);
    if (lane == 0) y[ii] = __fadd_rn(sum, __fmul_rn(x, d_param[ii / hd]));
}

// out[i] = silu(gate[i]) * up[i] (n a multiple of 8).
extern "C" __global__ void __launch_bounds__(256) swiglu2(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmul_rn(ggml_silu(gate[i]), up[i]);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Attention without rotary embedding: appends the key and the value (F16) of the new token at position *pos_dev; the query stays in qkv[0 .. q_dim).
// ---------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) kv_append_f32(const float* __restrict__ qkv, const int* __restrict__ pos_dev, __half* __restrict__ Kc,
                                                                __half* __restrict__ Vc, int kv_heads, int head_dim, int cache_len, int q_dim) {
    const int pos = pos_dev[0];
    const int kv_dim = kv_heads * head_dim;
    for (int idx = threadIdx.x; idx < kv_dim; idx += blockDim.x) {
        const int kvh = idx / head_dim, e = idx % head_dim;
        const size_t at = ((size_t)kvh * cache_len + pos) * head_dim + e;
        Kc[at] = __float2half_rn(qkv[q_dim + idx]);
        Vc[at] = __float2half_rn(qkv[q_dim + kv_dim + idx]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------------------------------------------------------------------------

// std::sort of libstdc++ (introsort, final insertion sort) over expert indices in descending order of the selection score, element for element as refsort.rs does it.
struct ExpertSort {
    const float* sel;
    int* v;
    __device__ bool less(int a, int b) const { return sel[a] > sel[b]; }
    __device__ void unguarded_linear_insert(int last) {
        const int val = v[last];
        int next = last - 1;
        while (less(val, v[next])) {
            v[last] = v[next];
            last = next;
            --next;
        }
        v[last] = val;
    }
    __device__ void insertion_sort(int first, int last) {
        if (first == last) return;
        for (int i = first + 1; i < last; ++i) {
            if (less(v[i], v[first])) {
                const int val = v[i];
                for (int j = i; j > first; --j) v[j] = v[j - 1];
                v[first] = val;
            } else {
                unguarded_linear_insert(i);
            }
        }
    }
    __device__ void push_heap(int base, int hole, int top, int val) {
        int parent = static_cast<int>(static_cast<unsigned>(hole - 1) / 2u);
        while (hole > top && less(v[base + parent], val)) {
            v[base + hole] = v[base + parent];
            hole = parent;
            parent = static_cast<int>(static_cast<unsigned>(hole - 1) / 2u);
        }
        v[base + hole] = val;
    }
    __device__ void adjust_heap(int base, int hole, int len, int val) {
        const int top = hole;
        int second = hole;
        while (second < (len > 0 ? len - 1 : 0) / 2) {
            second = 2 * (second + 1);
            if (less(v[base + second], v[base + second - 1])) --second;
            v[base + hole] = v[base + second];
            hole = second;
        }
        if ((len & 1) == 0 && len >= 2 && second == (len - 2) / 2) {
            second = 2 * (second + 1);
            v[base + hole] = v[base + second - 1];
            hole = second - 1;
        }
        push_heap(base, hole, top, val);
    }
    __device__ void make_heap(int first, int last) {
        const int len = last - first;
        if (len < 2) return;
        int parent = (len - 2) / 2;
        while (true) {
            adjust_heap(first, parent, len, v[first + parent]);
            if (parent == 0) return;
            --parent;
        }
    }
    __device__ void sort_heap(int first, int last) {
        while (last - first > 1) {
            --last;
            const int val = v[last];
            v[last] = v[first];
            adjust_heap(first, 0, last - first, val);
        }
    }
    __device__ void move_median_to_first(int result, int a, int b, int c) {
        if (less(v[a], v[b])) {
            if (less(v[b], v[c])) swap(v[result], v[b]);
            else if (less(v[a], v[c])) swap(v[result], v[c]);
            else swap(v[result], v[a]);
        } else if (less(v[a], v[c])) {
            swap(v[result], v[a]);
        } else if (less(v[b], v[c])) {
            swap(v[result], v[c]);
        } else {
            swap(v[result], v[b]);
        }
    }
    __device__ static void swap(int& x, int& y) {
        const int t = x;
        x = y;
        y = t;
    }
    __device__ int unguarded_partition_pivot(int first, int last) {
        const int mid = first + (last - first) / 2;
        move_median_to_first(first, first + 1, mid, last - 1);
        const int pivot = first;
        int lo = first + 1, hi = last;
        while (true) {
            while (less(v[lo], v[pivot])) ++lo;
            --hi;
            while (less(v[pivot], v[hi])) --hi;
            if (lo >= hi) return lo;
            swap(v[lo], v[hi]);
            ++lo;
        }
    }
    // introsort_loop without recursion: the right part is pushed, the left part continues in the loop.
    __device__ void introsort(int n) {
        int depth = 0;
        for (int m = n; m > 1; m >>= 1) ++depth;
        depth *= 2;
        int stack_first[64], stack_last[64], stack_depth[64];
        int sp = 0;
        stack_first[0] = 0;
        stack_last[0] = n;
        stack_depth[0] = depth;
        sp = 1;
        while (sp > 0) {
            --sp;
            int first = stack_first[sp], last = stack_last[sp], d = stack_depth[sp];
            while (last - first > 16) {
                if (d == 0) {
                    make_heap(first, last);
                    sort_heap(first, last);
                    break;
                }
                --d;
                const int cut = unguarded_partition_pivot(first, last);
                stack_first[sp] = cut;
                stack_last[sp] = last;
                stack_depth[sp] = d;
                ++sp;
                last = cut;
            }
        }
    }
    __device__ void sort(int n) {
        if (n == 0) return;
        introsort(n);
        if (n > 16) {
            insertion_sort(0, 16);
            for (int i = 16; i < n; ++i) unguarded_linear_insert(i);
        } else {
            insertion_sort(0, n);
        }
    }
};

// Routing of one token (Moe::forward): probs = sigmoid(logits), sel = probs + bias, the experts in descending order of sel with the tie order of libstdc++'s
// std::sort, the first n_used with their probabilities, normalized (sum in double, at least 6.103515625e-5) and scaled. One thread; ne at most 256.
extern "C" __global__ void moe_route(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ chosen, float* __restrict__ weights, int ne,
                                     int n_used, int norm_weights, float weights_scale) {
    if (threadIdx.x != 0) return;
    __shared__ float probs[256], sel[256];
    __shared__ int order[256];
    for (int e = 0; e < ne; ++e) {
        probs[e] = sigmoid_dev(logits[e]);
        sel[e] = __fadd_rn(probs[e], bias[e]);
        order[e] = e;
    }
    ExpertSort s{sel, order};
    s.sort(ne);
    float w[16];
    for (int i = 0; i < n_used; ++i) {
        chosen[i] = order[i];
        w[i] = probs[order[i]];
    }
    if (norm_weights) {
        double sum = 0.0;
        for (int i = 0; i < n_used; ++i) sum += static_cast<double>(w[i]);
        float sf = static_cast<float>(sum);
        sf = fmaxf(sf, 6.103515625e-5f);
        for (int i = 0; i < n_used; ++i) w[i] = __fdiv_rn(w[i], sf);
    }
    if (weights_scale != 0.0f && weights_scale != 1.0f) {
        for (int i = 0; i < n_used; ++i) w[i] = __fmul_rn(w[i], weights_scale);
    }
    for (int i = 0; i < n_used; ++i) weights[i] = w[i];
}

// Block-scaled projection of the selected experts, one grid row per slot: the expert of slot y is chosen[y]; its matrix starts chosen[y] * (rows * row_words) words /
// chosen[y] * (rows * cols / 64) scales after the start of the stacked experts. The input of slot y starts at qs + y * qs_stride (codes) and ad + y * ad_stride
// (scales), the output at out + y * out_stride. Otherwise exactly gemv_q2 (q2.cu).

extern "C" __global__ void __launch_bounds__(256) gemv_q2_moe(const uint32_t* __restrict__ w, const __half* __restrict__ wsc, const int8_t* __restrict__ qs,
                                                              const float* __restrict__ ad, float* __restrict__ out, const int* __restrict__ chosen, int rows,
                                                              int cols, int qs_stride, int ad_stride, int out_stride) {
    __shared__ uint32_t tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tab[i] = digits_word_q2(i);
    __syncthreads();
    const int slot = blockIdx.y;
    const int lane = threadIdx.x & 31, j = lane & 7;
    const int row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 4 + (lane >> 3);
    const int rc = min(row, rows - 1);
    const int blocks_per_row = (cols + 159) / 160, nb32 = cols / 32, sb = cols / 64;
    const int groups = (blocks_per_row + 3) >> 2, full_groups = blocks_per_row >> 2, tail = blocks_per_row & 3;
    const size_t expert = static_cast<size_t>(chosen[slot]);
    const uint32_t* wr = w + (expert * rows + rc) * blocks_per_row * 8;  // the regrouped layout of q2_words.cuh
    const __half* sc = wsc + (expert * rows + rc) * sb;
    const int8_t* q = qs + (size_t)slot * qs_stride;
    const float* d = ad + (size_t)slot * ad_stride;
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
                        const uint32_t u01 = __byte_perm(t0, t1, p | ((p + 4) << 4));
                        const uint32_t u23 = __byte_perm(t2, t3, p | ((p + 4) << 4));
                        dig = __byte_perm(u01, u23, 0x5410);
                    } else {
                        dig = __vsub4((((word & 0xffu) * 781u) >> 16) | ((((word >> 8) & 0xffu) * 781u >> 16) << 8) | ((((word >> 16) & 0xffu) * 781u >> 16) << 16) |
                                      (((word >> 24) * 781u >> 16) << 24), 0x01010101u);
                    }
                    const int aw = *reinterpret_cast<const int*>(q + s * 32 + 4 * j);
                    const int lane_sum = __dp4a(static_cast<int>(dig), aw, 0);
                    const float dd = __fmul_rn(__half2float(sc[s >> 1]), d[s]);
                    if ((s >> 1) & 1) acc1 = __fmaf_rn(dd, __int2float_rn(lane_sum), acc1);
                    else acc0 = __fmaf_rn(dd, __int2float_rn(lane_sum), acc0);
                }
            }
        }
    }
    const float l = __fadd_rn(acc0, acc1);
    const float r = __fadd_rn(l, __shfl_down_sync(0xffffffffu, l, 4, 8));
    const float y = __fadd_rn(r, __shfl_down_sync(0xffffffffu, r, 2, 8));
    const float z = __fadd_rn(y, __shfl_down_sync(0xffffffffu, y, 1, 8));
    if (j == 0 && row < rows) out[(size_t)slot * out_stride + row] = z;
}

// x[i] = max(x[i], 0)^2 for the n values (relu_sqr of hybrid.rs).
extern "C" __global__ void __launch_bounds__(256) relu_sqr(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float r = v > 0.0f ? v : 0.0f;
    x[i] = __fmul_rn(r, r);
}

// The combination of the experts: routed = head_0 * w_0, routed += head_i * w_i in the order of the slots, out = routed + shared. `heads` holds n_used rows of n values.
extern "C" __global__ void __launch_bounds__(256) moe_combine(const float* __restrict__ heads, const float* __restrict__ weights, const float* __restrict__ shared,
                                                              float* __restrict__ out, int n, int n_used) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float routed = 0.0f;
    for (int i = 0; i < n_used; ++i) {
        const float v = __fmul_rn(heads[(size_t)i * n + o], weights[i]);
        routed = i == 0 ? v : __fadd_rn(routed, v);
    }
    out[o] = __fadd_rn(routed, shared[o]);
}
