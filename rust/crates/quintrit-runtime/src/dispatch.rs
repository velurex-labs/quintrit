//! Run-time selection of the instruction set for the portable code of the runtime.
//!
//! On x86-64 the SIMD kernels are `#[target_feature]` functions that are chosen by CPU detection, so they do not
//! depend on the flags the crate was compiled with. Plain Rust code (the scalar definitions of the primitives, the
//! attention loops, the recurrent layers) is different: without `-C target-cpu=...` the compiler has to assume the
//! x86-64 baseline for it. A fused multiply-add (`f32::mul_add`) then becomes a call of the library function `fmaf`,
//! the hardware half-precision conversion becomes a call of an out-of-line wrapper, and AVX2 intrinsics written inline
//! become calls that pass 256-bit values through memory. Such code ran 1.5 to 2.6 times slower in a default build.
//!
//! The macro [`multiversion!`] removes the dependency on the build flags: it compiles one function body twice, once
//! for the baseline and once with `avx2`, `fma` and `f16c` enabled, and picks the second version when the CPU has these
//! features. Both versions execute the same operations in the same order, so the results are bit-identical (a fused
//! multiply-add is exact in hardware and in the library function alike).

/// Whether the CPU has AVX2, FMA and F16C, the feature set the `*_avx2` versions of the primitives are compiled for.
///
/// The three feature tests are folded into one cached answer, so that a primitive pays one relaxed load per call.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma_f16c() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    #[cfg(test)]
    if FORCE_BASELINE.load(Ordering::Relaxed) {
        return false;
    }
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let yes = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
            STATE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
    }
}

/// Tests set this to run the baseline version of every [`multiversion!`] function on a CPU that has the extended one,
/// to compare the two. It may be flipped while other tests run: both versions give the same bits.
#[cfg(all(test, target_arch = "x86_64"))]
static FORCE_BASELINE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Runs `f` with the baseline version of every [`multiversion!`] function (tests only). Calls are serialized, so that
/// two tests that compare the versions cannot switch each other back to the extended version.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn with_baseline<R>(f: impl FnOnce() -> R) -> R {
    use std::sync::atomic::Ordering;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_BASELINE.store(false, Ordering::Relaxed);
        }
    }
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    FORCE_BASELINE.store(true, Ordering::Relaxed);
    let _reset = Reset;
    f()
}

/// Conversion between half and single precision for the bodies of functions made by [`multiversion!`].
///
/// [`Detect`] asks the CPU on every call, which is what the plain functions `ops::f16_to_f32` and `ops::f32_to_f16` do.
/// `F16c` (x86-64 only) uses the instruction unconditionally and may only be used in code that runs with the `f16c`
/// feature enabled, i.e. in the specialized version of a [`multiversion!`] function. Both conversions are exact (to
/// half: round to nearest even), so they give the same bits.
pub(crate) trait HalfConv {
    fn to_f32(h: u16) -> f32;
    fn to_f16(f: f32) -> u16;
}

/// The portable choice: detect the hardware conversion at run time, otherwise convert in software.
pub(crate) struct Detect;

impl HalfConv for Detect {
    #[inline(always)]
    fn to_f32(h: u16) -> f32 {
        crate::ops::f16_to_f32(h)
    }
    #[inline(always)]
    fn to_f16(f: f32) -> u16 {
        crate::ops::f32_to_f16(f)
    }
}

/// The hardware conversion without a check; see [`HalfConv`].
#[cfg(target_arch = "x86_64")]
pub(crate) struct F16c;

#[cfg(target_arch = "x86_64")]
impl HalfConv for F16c {
    #[inline(always)]
    fn to_f32(h: u16) -> f32 {
        use core::arch::x86_64::*;
        // SAFETY: only used in the code compiled with `f16c` enabled (see the type documentation).
        unsafe { _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32))) }
    }
    #[inline(always)]
    fn to_f16(f: f32) -> u16 {
        use core::arch::x86_64::*;
        // SAFETY: as above; the rounding control 0 is round to nearest even.
        unsafe { _mm_cvtsi128_si32(_mm_cvtps_ph::<0>(_mm_set_ss(f))) as u16 }
    }
}

/// Defines a free function whose body is compiled for the baseline and, on x86-64, once more with `avx2`, `fma` and
/// `f16c`; the call picks the faster version by CPU detection.
///
/// The body is written once, as the generic function `body<H: HalfConv>`: it converts half-precision values with
/// `H::to_f32` and `H::to_f16` instead of `f16_to_f32` and `f32_to_f16` (the specialized version then contains the
/// bare instruction instead of a run-time check), and may use `mul_add`, `round_ties_even` and other operations that
/// are single instructions only with the extended feature set. The signature accepts plain `name: Type` parameters.
macro_rules! multiversion {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    ) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[allow(unused_mut)]
            #[inline(always)]
            fn body<H: $crate::dispatch::HalfConv>($($arg: $ty),*) $(-> $ret)? $body

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma,f16c")]
            unsafe fn body_avx2($($arg: $ty),*) $(-> $ret)? {
                body::<$crate::dispatch::F16c>($($arg),*)
            }

            #[cfg(target_arch = "x86_64")]
            if $crate::dispatch::has_avx2_fma_f16c() {
                // SAFETY: AVX2, FMA and F16C are available.
                return unsafe { body_avx2($($arg),*) };
            }
            body::<$crate::dispatch::Detect>($($arg),*)
        }
    };
}
pub(crate) use multiversion;
