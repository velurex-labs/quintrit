// glibc's expf (sysdeps/ieee754/flt-32/e_expf.c, the ARM optimized-routines algorithm: 32-entry table of 2^(i/32), cubic polynomial, double
// arithmetic) for the device. The reference CPU runs the FMA variant of glibc (e_expf-fma.c: the same C code compiled with -mfma), in which the compiler
// contracts `InvLn2N * xd + Shift` / `z - kd` and the polynomial into fused multiply-adds; the contraction used here is selected so that the results are
// bit-identical to the host libm over a million test values (tools/test_glibc_expf.sh).
#pragma once
#ifndef GLIBC_EXPF_FUSED
#define GLIBC_EXPF_FUSED 0
#endif
#include <cstdint>

__device__ __constant__ uint64_t kExp2fTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull
};

__device__ __forceinline__ float glibc_expf(float x) {
    const double InvLn2N = 0x1.71547652b82fep0 * 32.0;
    const double Shift = 0x1.8p52;
    const double C0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
    const double C1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
    const double C2 = 0x1.62e42ff0c52d6p-1 / 32.0;
    const uint32_t ix = __float_as_uint(x);
    const uint32_t abstop = (ix >> 20) & 0x7ff;
    if (abstop >= 0x42b) {  // |x| >= 88
        if (ix == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8) return x + x;
        if (x > 0x1.62e42ep6f) return __uint_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = static_cast<double>(x);
#if GLIBC_EXPF_FUSED
    double kd = __fma_rn(InvLn2N, xd, Shift);
    const uint64_t ki = static_cast<uint64_t>(__double_as_longlong(kd));
    kd -= Shift;
    const double r = __fma_rn(InvLn2N, xd, -kd);
#else
    const double z0 = InvLn2N * xd;
    double kd = z0 + Shift;
    const uint64_t ki = static_cast<uint64_t>(__double_as_longlong(kd));
    kd -= Shift;
    const double r = z0 - kd;
#endif
    const uint64_t t = kExp2fTab[ki % 32] + (ki << 47);
    const double s = __longlong_as_double(static_cast<long long>(t));
    const double z = __fma_rn(C0, r, C1);
    const double r2 = r * r;
    double y = __fma_rn(C2, r, 1.0);
    y = __fma_rn(z, r2, y);
    y = y * s;
    return static_cast<float>(y);
}
