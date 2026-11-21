//! GPU queries: `quintrit_cuda_device_count` and `quintrit_cuda_device_info`.
//!
//! The functions exist in every build of the library. Without the cargo feature `cuda` they return `QUINTRIT_ERR_UNSUPPORTED`, so a host can use the same
//! code against both builds and fall back to the CPU. With the feature, a machine without an NVIDIA driver reports zero devices.

use crate::error::{fail, guard};
#[cfg(not(feature = "cuda"))]
use crate::error::QUINTRIT_ERR_UNSUPPORTED;
use crate::util::{check_out_struct, out_ptr};
use std::ffi::{c_char, c_int};

#[cfg(feature = "cuda")]
use crate::util::write_out;
#[cfg(feature = "cuda")]
use crate::error::{Failure, QUINTRIT_ERR_DEVICE, QUINTRIT_ERR_INVALID_ARGUMENT};

/// `quintrit_cuda_device_info_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CudaDeviceInfo {
    pub struct_size: u32,
    /// Index of the GPU in the driver's enumeration (what `quintrit_gpu_options_t::device_index` takes).
    pub index: u32,
    /// Compute capability.
    pub compute_major: i32,
    pub compute_minor: i32,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// 1 if the GPU can run the kernels (compute capability 8.0 or newer), 0 otherwise.
    pub supported: u32,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// L2 cache in bytes.
    pub l2_cache_bytes: u64,
    /// Marketing name (NUL-terminated, truncated to 63 bytes).
    pub name: [c_char; 64],
}

/// Number of NVIDIA GPUs the driver reports (0 without a driver). `QUINTRIT_ERR_UNSUPPORTED` if this build of the library has no GPU support.
///
/// # Safety
/// `out_count` points to a `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_cuda_device_count(out_count: *mut u32) -> c_int {
    guard(|| {
        out_ptr(out_count, "out_count")?;
        #[cfg(feature = "cuda")]
        {
            let n = match quintrit_cuda::device_count() {
                Ok(n) => n,
                Err(quintrit_cuda::Error::NoDriver(_)) => 0,
                Err(e) => fail!(QUINTRIT_ERR_DEVICE, "{e}"),
            };
            unsafe { write_out(out_count, n as u32, "out_count")? };
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            fail!(QUINTRIT_ERR_UNSUPPORTED, "this build of the library has no GPU support (build quintrit-c with the feature `cuda`)")
        }
    })
}

/// Describes GPU `index` (below the count of `quintrit_cuda_device_count`).
///
/// # Safety
/// `out_info` points to a `quintrit_cuda_device_info_t` with `struct_size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_cuda_device_info(index: u32, out_info: *mut CudaDeviceInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_out_struct.
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        #[cfg(feature = "cuda")]
        {
            let info = quintrit_cuda::DeviceInfo::query(index as usize).map_err(|e| match e {
                quintrit_cuda::Error::InvalidArgument(m) => Failure::new(QUINTRIT_ERR_INVALID_ARGUMENT, m),
                quintrit_cuda::Error::NoDriver(m) => Failure::new(QUINTRIT_ERR_INVALID_ARGUMENT, format!("no GPU {index}: {m}")),
                other => Failure::new(QUINTRIT_ERR_DEVICE, other.to_string()),
            })?;
            let mut name = [0 as c_char; 64];
            for (dst, b) in name.iter_mut().zip(info.name.bytes().take(63)) {
                *dst = b as c_char;
            }
            let value = CudaDeviceInfo {
                struct_size: size_of::<CudaDeviceInfo>() as u32,
                index,
                compute_major: info.compute_capability.0 as i32,
                compute_minor: info.compute_capability.1 as i32,
                sm_count: info.sm_count,
                supported: info.check_supported().is_ok() as u32,
                total_memory: info.total_memory,
                l2_cache_bytes: info.l2_cache_bytes,
                name,
            };
            unsafe { out.write(value) };
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (index, out);
            fail!(QUINTRIT_ERR_UNSUPPORTED, "this build of the library has no GPU support (build quintrit-c with the feature `cuda`)")
        }
    })
}
