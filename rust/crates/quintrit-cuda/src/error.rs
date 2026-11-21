//! Errors of the GPU backend.

use std::fmt;

/// An error of the GPU backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The CUDA driver library could not be loaded (no NVIDIA driver installed) or lacks a required function.
    NoDriver(String),
    /// The crate was built without CUDA kernels (no `nvcc` was found at build time).
    NoKernels,
    /// A CUDA driver call failed.
    Driver {
        /// Name of the driver function.
        call: &'static str,
        /// The `CUresult` code.
        code: i32,
        /// Name and description reported by the driver.
        message: String,
    },
    /// The device cannot run the kernels (compute capability below 8.0 or too little shared memory).
    Unsupported(String),
    /// An argument of a call is invalid (a shape the kernels cannot handle, a buffer that is too small, ...).
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoDriver(m) => write!(f, "the CUDA driver is not available: {m}"),
            Error::NoKernels => f.write_str(
                "this build of quintrit-cuda contains no GPU kernels (the CUDA toolkit, nvcc, was not found at build time)",
            ),
            Error::Driver { call, code, message } => write!(f, "CUDA driver call {call} failed with code {code}: {message}"),
            Error::Unsupported(m) => write!(f, "unsupported GPU: {m}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type of the GPU backend.
pub type Result<T> = std::result::Result<T, Error>;
