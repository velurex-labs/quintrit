//! Python bindings for the Quintrit SDK (`quintrit._native`).
//!
//! Three levels are exposed:
//! * kernel level ([`packed`]): `pack`, `unpack`, `gemv` and the `PackedMatrix` type on numpy arrays;
//! * format level ([`format`]): `GgufFile`, tensor listing and ternary tensor extraction;
//! * runtime level ([`runtime`]): `Model`, `Tokenizer` and token streaming.
//!
//! The pure-Python package `quintrit` (directory `python/quintrit`) re-exports this module and defines the exception
//! hierarchy in `quintrit._errors`.

mod arrays;
mod error;
mod format;
mod kernels;
mod packed;
mod runtime;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Names of the kernels this CPU can run, fastest last.
#[pyfunction]
fn available_kernels() -> Vec<&'static str> {
    kernels::available_names()
}

/// The kernel `"auto"` resolves to on this CPU.
#[pyfunction]
fn best_kernel() -> &'static str {
    kernels::Kernel::best().name()
}

/// Detected CPU features that matter for kernel selection, as a `{name: bool}` dict.
#[pyfunction]
fn cpu_features(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for (name, present) in kernels::cpu_features() {
        d.set_item(name, present)?;
    }
    Ok(d)
}

/// Default number of worker threads (logical CPUs, at most 256).
#[pyfunction]
fn default_threads() -> usize {
    kernels::default_threads()
}

/// The NVIDIA GPUs the driver reports, as a list of dicts (`index`, `name`, `compute_capability`, `multiprocessors`, `memory_bytes`, `supported`).
/// Empty without a driver or in a build without GPU support (see `cuda_built()`).
#[pyfunction]
fn cuda_devices(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    {
        let count = quintrit_cuda::device_count().unwrap_or(0);
        for i in 0..count {
            let Ok(info) = quintrit_cuda::DeviceInfo::query(i) else { continue };
            let d = PyDict::new(py);
            d.set_item("index", i)?;
            d.set_item("name", &info.name)?;
            d.set_item("compute_capability", info.compute_capability)?;
            d.set_item("multiprocessors", info.sm_count)?;
            d.set_item("memory_bytes", info.total_memory)?;
            d.set_item("supported", info.check_supported().is_ok())?;
            out.push(d);
        }
    }
    let _ = py;
    Ok(out)
}

/// True if this build of the extension has GPU support (it was compiled with the cargo feature `cuda`).
#[pyfunction]
fn cuda_built() -> bool {
    cfg!(feature = "cuda")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(available_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(best_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(default_threads, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_devices, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_built, m)?)?;
    m.add_function(wrap_pyfunction!(packed::pack, m)?)?;
    m.add_function(wrap_pyfunction!(packed::unpack, m)?)?;
    m.add_function(wrap_pyfunction!(packed::gemv, m)?)?;
    m.add_class::<packed::PackedMatrix>()?;
    format::register(m)?;
    runtime::register(m)?;
    Ok(())
}
