//! Validation of numpy inputs and construction of numpy outputs.

use crate::error::{Kind, QError};
use numpy::{PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// An int8 array copied out of Python: C-order data plus its shape.
///
/// The data is always copied so that the GIL can be released afterwards without another Python thread being able to
/// modify the buffer under the running kernel.
pub struct Int8Array {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "object".to_string())
}

/// Extracts a numpy array of dtype int8 with one of the `allowed_ndim` dimension counts and no zero-length axis.
///
/// Arrays of any memory layout (C, Fortran, strided views) are accepted; other dtypes are rejected rather than
/// silently converted.
pub fn read_int8(obj: &Bound<'_, PyAny>, what: &str, allowed_ndim: &[usize]) -> Result<Int8Array, QError> {
    let py = obj.py();
    let untyped = obj.cast::<PyUntypedArray>().map_err(|_| {
        QError::new(
            Kind::DType,
            format!("{what} must be a numpy.ndarray with dtype int8, got {}", type_name(obj)),
        )
    })?;
    if !untyped.dtype().is_equiv_to(&numpy::dtype::<i8>(py)) {
        return Err(QError::new(
            Kind::DType,
            format!(
                "{what} must have dtype int8, got {}; convert it explicitly with .astype(numpy.int8)",
                untyped.dtype()
            ),
        ));
    }
    let ndim = untyped.ndim();
    if !allowed_ndim.contains(&ndim) {
        let expected: Vec<String> = allowed_ndim.iter().map(|d| format!("{d}-D")).collect();
        return Err(QError::new(
            Kind::Shape,
            format!("{what} must be {}, got a {ndim}-D array", expected.join(" or ")),
        ));
    }
    let shape: Vec<usize> = untyped.shape().to_vec();
    if shape.contains(&0) {
        return Err(QError::new(Kind::Shape, format!("{what} must not be empty, got shape {shape:?}")));
    }
    let typed = obj
        .cast::<PyArrayDyn<i8>>()
        .map_err(|e| QError::new(Kind::DType, format!("{what}: {e}")))?;
    let readonly = typed.readonly();
    let view = readonly.as_array();
    let data = match view.as_slice() {
        Some(s) => s.to_vec(),
        None => view.iter().copied().collect(),
    };
    Ok(Int8Array { data, shape })
}

/// Builds a 2-D int8 numpy array from row-major data.
pub fn int8_matrix<'py>(py: Python<'py>, data: Vec<i8>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
    let flat = PyArray1::from_vec(py, data);
    Ok(flat.reshape([rows, cols])?.into_any())
}

/// Builds a numpy array of the given shape (1-D or 2-D) from row-major data.
pub fn shaped<'py, T: numpy::Element>(py: Python<'py>, data: Vec<T>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let flat = PyArray1::from_vec(py, data);
    if shape.len() == 1 {
        return Ok(flat.into_any());
    }
    Ok(flat.reshape(shape.to_vec())?.into_any())
}
