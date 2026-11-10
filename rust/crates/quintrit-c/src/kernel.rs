//! Version, CPU feature and kernel-selection queries.

use crate::error::{fail, guard, Res, QUINTRIT_ERR_BUFFER_TOO_SMALL, QUINTRIT_ERR_INVALID_ARGUMENT, QUINTRIT_ERR_UNSUPPORTED};
use crate::util::{out_slice, write_out};
use std::ffi::{c_char, c_int};

/// Version of the C ABI. It changes when a function or struct layout changes incompatibly.
pub const QUINTRIT_ABI_VERSION: u32 = 1;

/// Library version, taken from the Cargo package (NUL-terminated).
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// Selects the best usable kernel at run time.
pub const QUINTRIT_KERNEL_AUTO: i32 = 0;
/// Portable scalar kernel, 32-row tiles. Always available. Slow; meant as a reference and for unsupported CPUs. On AArch64 it runs the NEON code (the AArch64 activation tables use a NEON-specific layout).
pub const QUINTRIT_KERNEL_SCALAR: i32 = 1;
/// AVX2 (x86-64) or NEON (AArch64) activation-LUT kernel, 32-row tiles.
pub const QUINTRIT_KERNEL_SIMD32: i32 = 2;
/// AVX-512 (BW + VBMI) activation-LUT kernel, 64-row tiles. x86-64 only.
pub const QUINTRIT_KERNEL_AVX512: i32 = 3;

/// CPU feature bits reported by [`quintrit_cpu_features`].
pub const QUINTRIT_CPU_AVX2: u64 = 1 << 0;
pub const QUINTRIT_CPU_AVX512F: u64 = 1 << 1;
pub const QUINTRIT_CPU_AVX512BW: u64 = 1 << 2;
pub const QUINTRIT_CPU_AVX512VBMI: u64 = 1 << 3;
pub const QUINTRIT_CPU_AVX512VNNI: u64 = 1 << 4;
pub const QUINTRIT_CPU_NEON: u64 = 1 << 8;
pub const QUINTRIT_CPU_NEON_DOTPROD: u64 = 1 << 9;
pub const QUINTRIT_CPU_NEON_I8MM: u64 = 1 << 10;

/// Maximum thread count accepted by the API (every distinct count creates a persistent worker pool).
pub const QUINTRIT_MAX_THREADS: u32 = quintrit_core::MAX_THREADS as u32;

/// A concrete (resolved) kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Simd32,
    Avx512,
}

impl Kernel {
    pub fn id(self) -> i32 {
        match self {
            Kernel::Scalar => QUINTRIT_KERNEL_SCALAR,
            Kernel::Simd32 => QUINTRIT_KERNEL_SIMD32,
            Kernel::Avx512 => QUINTRIT_KERNEL_AVX512,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Simd32 => "simd32",
            Kernel::Avx512 => "avx512",
        }
    }

    /// Rows per weight tile of the layout this kernel reads.
    pub fn tile_rows(self) -> usize {
        match self {
            Kernel::Scalar | Kernel::Simd32 => 32,
            Kernel::Avx512 => 64,
        }
    }

    /// Whether this kernel can run on the current CPU (runtime detection).
    pub fn is_supported(self) -> bool {
        match self {
            Kernel::Scalar => true,
            Kernel::Simd32 => simd32_supported(),
            Kernel::Avx512 => avx512_supported(),
        }
    }

    /// Fails with `QUINTRIT_ERR_UNSUPPORTED` if the kernel cannot run here.
    pub fn require_supported(self) -> Res<()> {
        if !self.is_supported() {
            fail!(
                QUINTRIT_ERR_UNSUPPORTED,
                "kernel '{}' is not supported on this CPU (supported: {})",
                self.name(),
                supported_list().iter().map(|k| k.name()).collect::<Vec<_>>().join(", ")
            );
        }
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
fn simd32_supported() -> bool {
    is_x86_feature_detected!("avx2")
}
#[cfg(target_arch = "aarch64")]
fn simd32_supported() -> bool {
    true
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn simd32_supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx512_supported() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")
}
#[cfg(not(target_arch = "x86_64"))]
fn avx512_supported() -> bool {
    false
}

/// Kernels usable on this CPU, best first.
pub fn supported_list() -> Vec<Kernel> {
    [Kernel::Avx512, Kernel::Simd32, Kernel::Scalar].into_iter().filter(|k| k.is_supported()).collect()
}

/// Maps a kernel id of the C API to a concrete kernel; `AUTO` resolves to the best usable kernel.
pub fn resolve(id: i32) -> Res<Kernel> {
    match id {
        QUINTRIT_KERNEL_AUTO => Ok(supported_list()[0]),
        QUINTRIT_KERNEL_SCALAR => Ok(Kernel::Scalar),
        QUINTRIT_KERNEL_SIMD32 => Ok(Kernel::Simd32),
        QUINTRIT_KERNEL_AVX512 => Ok(Kernel::Avx512),
        other => fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "unknown kernel id {other}"),
    }
}

/// Maps a thread count of the C API (0 = automatic) to a concrete count.
pub fn resolve_threads(threads: u32) -> Res<usize> {
    if threads > QUINTRIT_MAX_THREADS {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "thread count {threads} exceeds the maximum of {QUINTRIT_MAX_THREADS}");
    }
    if threads == 0 {
        return Ok(std::thread::available_parallelism().map_or(1, |n| n.get()).min(QUINTRIT_MAX_THREADS as usize));
    }
    Ok(threads as usize)
}

// ---------------------------------------------------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------------------------------------------------

/// Returns the library version as a static NUL-terminated string, for example `"0.1.0"`.
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_version() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

/// Returns the ABI version of the library; compare it with `QUINTRIT_ABI_VERSION` of the header you compiled against.
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_abi_version() -> u32 {
    QUINTRIT_ABI_VERSION
}

/// Reports the CPU features the kernels use, as a bit mask of `QUINTRIT_CPU_*` (runtime detection).
///
/// # Safety
/// `out_mask` must be a valid pointer to a `uint64_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_cpu_features(out_mask: *mut u64) -> c_int {
    guard(|| {
        #[allow(unused_mut)]
        let mut mask = 0u64;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                mask |= QUINTRIT_CPU_AVX2;
            }
            if is_x86_feature_detected!("avx512f") {
                mask |= QUINTRIT_CPU_AVX512F;
            }
            if is_x86_feature_detected!("avx512bw") {
                mask |= QUINTRIT_CPU_AVX512BW;
            }
            if is_x86_feature_detected!("avx512vbmi") {
                mask |= QUINTRIT_CPU_AVX512VBMI;
            }
            if is_x86_feature_detected!("avx512vnni") {
                mask |= QUINTRIT_CPU_AVX512VNNI;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            mask |= QUINTRIT_CPU_NEON;
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                mask |= QUINTRIT_CPU_NEON_DOTPROD;
            }
            if std::arch::is_aarch64_feature_detected!("i8mm") {
                mask |= QUINTRIT_CPU_NEON_I8MM;
            }
        }
        // SAFETY: validated by write_out.
        unsafe { write_out(out_mask, mask, "out_mask") }
    })
}

/// Tells whether a kernel can run on this CPU. `QUINTRIT_KERNEL_AUTO` and `QUINTRIT_KERNEL_SCALAR` always can.
///
/// # Safety
/// `out_supported` must be a valid pointer to an `int32_t`; it receives 1 or 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_kernel_supported(kernel: i32, out_supported: *mut i32) -> c_int {
    guard(|| {
        let supported = if kernel == QUINTRIT_KERNEL_AUTO { true } else { resolve(kernel)?.is_supported() };
        // SAFETY: validated by write_out.
        unsafe { write_out(out_supported, supported as i32, "out_supported") }
    })
}

/// Resolves a kernel id to the concrete kernel that would run: `QUINTRIT_KERNEL_AUTO` becomes the best kernel usable on
/// this CPU, any other id is returned unchanged (even if unsupported; see `quintrit_kernel_supported`).
///
/// # Safety
/// `out_kernel` must be a valid pointer to an `int32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_kernel_resolve(kernel: i32, out_kernel: *mut i32) -> c_int {
    guard(|| {
        let k = resolve(kernel)?;
        // SAFETY: validated by write_out.
        unsafe { write_out(out_kernel, k.id(), "out_kernel") }
    })
}

/// Returns the name of a kernel (`"auto"`, `"scalar"`, `"simd32"`, `"avx512"`), or NULL for an unknown id.
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_kernel_name(kernel: i32) -> *const c_char {
    match kernel {
        QUINTRIT_KERNEL_AUTO => c"auto".as_ptr(),
        QUINTRIT_KERNEL_SCALAR => c"scalar".as_ptr(),
        QUINTRIT_KERNEL_SIMD32 => c"simd32".as_ptr(),
        QUINTRIT_KERNEL_AVX512 => c"avx512".as_ptr(),
        _ => std::ptr::null(),
    }
}

/// Returns the number of rows per weight tile of the layout of a kernel (`AUTO` is resolved first).
///
/// # Safety
/// `out_tile_rows` must be a valid pointer to a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_kernel_tile_rows(kernel: i32, out_tile_rows: *mut usize) -> c_int {
    guard(|| {
        let k = resolve(kernel)?;
        // SAFETY: validated by write_out.
        unsafe { write_out(out_tile_rows, k.tile_rows(), "out_tile_rows") }
    })
}

/// Lists the kernels usable on this CPU, best first (`QUINTRIT_KERNEL_SCALAR` is always last).
///
/// `out` may be NULL with `capacity == 0` to query the count. `*out_count` always receives the number of usable
/// kernels; `QUINTRIT_ERR_BUFFER_TOO_SMALL` is returned if `capacity` is smaller.
///
/// # Safety
/// `out` must point to `capacity` writable `int32_t`, `out_count` to a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_supported_kernels(out: *mut i32, capacity: usize, out_count: *mut usize) -> c_int {
    guard(|| {
        let list = supported_list();
        // SAFETY: validated by write_out / out_slice.
        unsafe { write_out(out_count, list.len(), "out_count")? };
        let out = unsafe { out_slice(out, capacity, "out")? };
        if capacity < list.len() {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "{} kernels are usable but the buffer holds {capacity}", list.len());
        }
        for (dst, k) in out.iter_mut().zip(&list) {
            *dst = k.id();
        }
        Ok(())
    })
}

/// Returns the number of worker threads that a thread count of 0 ("automatic") stands for.
///
/// # Safety
/// `out_threads` must be a valid pointer to a `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_default_threads(out_threads: *mut u32) -> c_int {
    guard(|| {
        let n = resolve_threads(0)? as u32;
        // SAFETY: validated by write_out.
        unsafe { write_out(out_threads, n, "out_threads") }
    })
}
