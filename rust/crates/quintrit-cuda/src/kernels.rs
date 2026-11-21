//! The GPU kernel images embedded at build time (one fat binary per `.cu` file in `kernels/`).

use crate::error::{Error, Result};

include!(concat!(env!("OUT_DIR"), "/images.rs"));

/// True when this build contains GPU kernels (the CUDA toolkit was available when the crate was compiled).
pub fn available() -> bool {
    !IMAGES.is_empty()
}

/// The fat binary compiled from `kernels/<name>.cu`.
pub fn image(name: &str) -> Result<&'static [u8]> {
    if IMAGES.is_empty() {
        return Err(Error::NoKernels);
    }
    IMAGES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, bytes)| *bytes)
        .ok_or_else(|| Error::InvalidArgument(format!("no kernel image named {name}")))
}
