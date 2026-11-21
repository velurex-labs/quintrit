// SiLU with the polynomial exponential of ggml (`ggml_v_expf`, `ggml_v_silu`), so that the device result is bit-identical to the
// CPU runtime. The operation order and the fused multiply-adds follow the AVX2 routine `ggml_v_expf_avx2` of quintrit-runtime
// (`ops.rs`). Arguments with |n| > 126 (results near the float range limits) use the accurate expf instead of the special-case
// blend of the CPU routine.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float ggml_expf(float x) {
    const float r = __uint_as_float(0x4b400000u);
    const float z = fmaf(x, __uint_as_float(0x3fb8aa3bu), r);
    const float n = z - r;
    const float b = fmaf(-n, __uint_as_float(0x35bfbe8eu), fmaf(-n, __uint_as_float(0x3f317200u), x));
    const uint32_t e = __float_as_uint(z) << 23;
    const float k = __uint_as_float(e + 0x3f800000u);
    if (fabsf(n) > 126.0f) return expf(x);
    const float u = b * b;
    const float j = fmaf(fmaf(fmaf(__uint_as_float(0x3c072010u), b, __uint_as_float(0x3d2b9f17u)), u,
                              fmaf(__uint_as_float(0x3e2aaf33u), b, __uint_as_float(0x3efffedbu))),
                         u, __uint_as_float(0x3f7ffff6u) * b);
    return fmaf(j, k, k);
}

__device__ __forceinline__ float ggml_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, ggml_expf(0.0f - x))); }
