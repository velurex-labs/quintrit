//! # Quintrit CUDA (`quintrit-cuda`)
//!
//! Optional NVIDIA GPU backend of Quintrit: CUDA kernels for velurex-encoded ternary matrices (five trits per byte)
//! and a small safe layer over the CUDA driver API.
//!
//! - The driver library (`libcuda.so.1`, `nvcuda.dll`) is loaded at run time. Nothing of NVIDIA is linked at build
//!   time, so the crate builds and its CPU-side tests pass on machines without CUDA.
//! - The kernels in `kernels/` are compiled by `build.rs` with `nvcc` when the CUDA toolkit is found, for `sm_80`,
//!   `sm_86` and `sm_89` (and PTX for newer GPUs). Without the toolkit the crate contains no kernels and loading one
//!   returns [`Error::NoKernels`].
//! - The minimum supported GPU is compute capability 8.0 (Ampere: A100, RTX 30xx and newer). [`Device::open`] rejects
//!   older devices with [`Error::Unsupported`].
//! - Kernel configurations are chosen by device resources (multiprocessor count and L2 size), see [`profile`].
//!
//! ## Example
//!
//! ```no_run
//! use quintrit_cuda::Device;
//!
//! # fn main() -> Result<(), quintrit_cuda::Error> {
//! let device = Device::open(0)?;
//! let info = device.info();
//! println!("{}: compute capability {}.{}, {} SMs", info.name, info.compute_capability.0, info.compute_capability.1, info.sm_count);
//! # Ok(())
//! # }
//! ```

pub mod device;
pub mod driver;
pub mod error;
pub mod gemv;
pub mod kernels;
pub mod layer;
pub mod pack;
pub mod profile;

pub use device::{
    device_count, Args, Device, DeviceBuffer, DeviceInfo, DevicePtr, Event, Function, Graph, GraphExec, Module, PinnedBuffer, Stream,
    MIN_COMPUTE_CAPABILITY, MIN_SHARED_MEMORY_OPTIN,
};
pub use error::{Error, Result};
pub use gemv::{Activation, Gemv, RopeLayout};
pub use layer::{AttentionLayout, AttentionWorkspace, LayerKernels, RopeKvLayout, RopeMode};
pub use profile::{DeviceProfile, GemvConfig, Role};

/// Compiles the Rust examples of the crate README as doctests, so that they cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
