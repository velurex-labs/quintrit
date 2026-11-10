//! Format level: GGUF files, tensor listing, ternary tensor extraction and transcoding.

use crate::arrays::int8_matrix;
use crate::error::{guard, Kind, QError, QResult};
use crate::kernels::Kernel;
use crate::packed::PackedMatrix;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use quintrit_format::decoders::get_ternary_decoder;
use quintrit_format::{
    GgufFile, GgufMetadataValue, GgufTensorInfo, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S,
    GGML_TYPE_Q8_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

fn type_name(t: u32) -> String {
    match t {
        GGML_TYPE_F32 => "F32".to_string(),
        GGML_TYPE_F16 => "F16".to_string(),
        GGML_TYPE_Q8_0 => "Q8_0".to_string(),
        GGML_TYPE_BF16 => "BF16".to_string(),
        GGML_TYPE_TQ1_0 => "TQ1_0".to_string(),
        GGML_TYPE_TQ2_0 => "TQ2_0".to_string(),
        GGML_TYPE_I2_S => "I2_S".to_string(),
        other => format!("UNKNOWN({other})"),
    }
}

fn is_ternary(t: u32) -> bool {
    quintrit_format::is_supported_ternary_type(t)
}

/// Byte length of a tensor's data as `GgufFile::get_tensor_data` computes it (`None` for types it cannot read).
fn tensor_nbytes(t: &GgufTensorInfo) -> Option<usize> {
    let n = t.num_elements();
    match t.tensor_type {
        GGML_TYPE_F32 => Some(n * 4),
        GGML_TYPE_F16 => Some(n * 2),
        GGML_TYPE_I2_S => Some(n.div_ceil(4)),
        GGML_TYPE_TQ1_0 if n % 256 == 0 => Some(n / 256 * 54),
        GGML_TYPE_TQ2_0 if n % 256 == 0 => Some(n / 256 * 66),
        _ => None,
    }
}

/// Description of one tensor of a GGUF file.
#[pyclass(frozen, name = "TensorInfo", module = "quintrit")]
pub struct TensorInfo {
    name: String,
    ggml_shape: Vec<usize>,
    ggml_type: u32,
    offset: u64,
    nbytes: Option<usize>,
}

impl TensorInfo {
    fn from_info(t: &GgufTensorInfo) -> Self {
        Self {
            name: t.name.clone(),
            ggml_shape: t.shape.clone(),
            ggml_type: t.tensor_type,
            offset: t.offset,
            nbytes: tensor_nbytes(t),
        }
    }
}

#[pymethods]
impl TensorInfo {
    /// Tensor name, for example `blk.0.attn_q.weight`.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Shape in numpy order (`(rows, cols)` for a matrix); GGUF stores the dimensions innermost first.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.ggml_shape.iter().rev().copied().collect()
    }

    /// Shape as stored in the file (innermost dimension first).
    #[getter]
    fn ggml_shape(&self) -> Vec<usize> {
        self.ggml_shape.clone()
    }

    /// Numeric GGML type id.
    #[getter]
    fn dtype_id(&self) -> u32 {
        self.ggml_type
    }

    /// GGML type name, for example `"I2_S"`, `"F16"` or `"UNKNOWN(12)"`.
    #[getter]
    fn dtype(&self) -> String {
        type_name(self.ggml_type)
    }

    /// True for the ternary types the SDK can read (`I2_S`, `TQ1_0`, `TQ2_0`).
    #[getter]
    fn is_ternary(&self) -> bool {
        is_ternary(self.ggml_type)
    }

    /// Number of elements.
    #[getter]
    fn n_elements(&self) -> usize {
        self.ggml_shape.iter().product()
    }

    /// Byte offset relative to the start of the tensor data section.
    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    /// Size of the tensor data in bytes, or `None` for types whose size is not known to the SDK.
    #[getter]
    fn nbytes(&self) -> Option<usize> {
        self.nbytes
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorInfo(name={:?}, shape={:?}, dtype={:?})",
            self.name,
            self.ggml_shape.iter().rev().collect::<Vec<_>>(),
            type_name(self.ggml_type)
        )
    }
}

fn io_error(path: &Path, e: io::Error) -> QError {
    match e.kind() {
        io::ErrorKind::NotFound => QError::new(Kind::FileNotFound, format!("{}: no such file", path.display())),
        io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => {
            QError::new(Kind::Format, format!("{}: not a valid GGUF file ({e})", path.display()))
        }
        _ => QError::new(Kind::Io, format!("{}: {e}", path.display())),
    }
}

fn metadata_to_py<'py>(py: Python<'py>, v: &GgufMetadataValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        GgufMetadataValue::Uint8(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Int8(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Uint16(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Int16(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Uint32(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Int32(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Uint64(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Int64(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Float32(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Float64(x) => x.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Bool(x) => x.into_pyobject(py)?.to_owned().into_any(),
        GgufMetadataValue::String(s) => s.into_pyobject(py)?.into_any(),
        GgufMetadataValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(metadata_to_py(py, item)?)?;
            }
            list.into_any()
        }
    })
}

/// A memory-mapped GGUF model file (read only).
#[pyclass(frozen, name = "GgufFile", module = "quintrit")]
pub struct PyGguf {
    path: PathBuf,
    inner: RwLock<Option<GgufFile>>,
}

impl PyGguf {
    fn with<R>(&self, f: impl FnOnce(&GgufFile) -> QResult<R>) -> QResult<R> {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        match guard.as_ref() {
            Some(file) => f(file),
            None => Err(QError::new(Kind::Io, "the GgufFile is closed")),
        }
    }

    fn info(file: &GgufFile, name: &str) -> QResult<GgufTensorInfo> {
        file.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| QError::new(Kind::TensorNotFound, format!("tensor {name:?} not found")))
    }

    /// Decodes a ternary tensor to row-major trits.
    fn trits(file: &GgufFile, name: &str) -> QResult<(Vec<i8>, usize, usize)> {
        let info = Self::info(file, name)?;
        if !is_ternary(info.tensor_type) {
            return Err(QError::new(
                Kind::Format,
                format!(
                    "tensor {name:?} has type {} and is not a ternary type (I2_S, TQ1_0 or TQ2_0)",
                    type_name(info.tensor_type)
                ),
            ));
        }
        if info.shape.len() != 2 {
            return Err(QError::new(
                Kind::Shape,
                format!("tensor {name:?} must be 2-D to be read as a matrix, got shape {:?}", info.shape),
            ));
        }
        let (cols, rows) = (info.shape[0], info.shape[1]);
        if rows == 0 || cols == 0 {
            return Err(QError::new(Kind::Shape, format!("tensor {name:?} is empty")));
        }
        if info.tensor_type == GGML_TYPE_I2_S && cols % 128 != 0 {
            return Err(QError::new(Kind::Format, format!("I2_S tensor {name:?}: cols must be a multiple of 128, got {cols}")));
        }
        let decoder = get_ternary_decoder(info.tensor_type).map_err(|e| QError::new(Kind::Format, e))?;
        let data = file
            .get_tensor_data(name)
            .ok_or_else(|| QError::new(Kind::Format, format!("data of tensor {name:?} lies outside the file")))?;
        let trits = guard("unpack_trits", || decoder.unpack_trits(data, rows, cols))?
            .map_err(|e| QError::new(Kind::Format, format!("tensor {name:?}: {e}")))?;
        Ok((trits, rows, cols))
    }
}

#[pymethods]
impl PyGguf {
    /// Opens and memory-maps a GGUF file.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("quintrit")) {
            return Err(QError::new(
                Kind::Format,
                format!(
                    "{}: .quintrit files are not supported by this version of the SDK; open the GGUF file instead",
                    path.display()
                ),
            )
            .into());
        }
        let file = py
            .detach(|| guard("GgufFile::open", || GgufFile::open(&path)))
            .map_err(PyErr::from)?
            .map_err(|e| io_error(&path, e))?;
        Ok(Self { path, inner: RwLock::new(Some(file)) })
    }

    /// Path the file was opened from.
    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// GGUF container version.
    #[getter]
    fn version(&self) -> PyResult<u32> {
        Ok(self.with(|f| Ok(f.version))?)
    }

    /// Value of `general.architecture`, if present.
    #[getter]
    fn architecture(&self) -> PyResult<Option<String>> {
        Ok(self.with(|f| Ok(f.architecture().map(str::to_string)))?)
    }

    /// Byte offset of the tensor data section.
    #[getter]
    fn data_start(&self) -> PyResult<usize> {
        Ok(self.with(|f| Ok(f.data_start()))?)
    }

    /// Number of tensors.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.with(|f| Ok(f.tensors.len()))?)
    }

    fn __contains__(&self, name: &str) -> PyResult<bool> {
        Ok(self.with(|f| Ok(f.tensors.contains_key(name)))?)
    }

    /// All tensors, ordered by their position in the file.
    fn tensors(&self) -> PyResult<Vec<TensorInfo>> {
        Ok(self.with(|f| {
            let mut all: Vec<&GgufTensorInfo> = f.tensors.values().collect();
            all.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
            Ok(all.into_iter().map(TensorInfo::from_info).collect())
        })?)
    }

    /// Description of one tensor; raises `TensorNotFoundError` if the name is unknown.
    fn tensor(&self, name: &str) -> PyResult<TensorInfo> {
        Ok(self.with(|f| Ok(TensorInfo::from_info(&Self::info(f, name)?)))?)
    }

    /// Metadata keys in sorted order.
    fn metadata_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.with(|f| {
            let mut keys: Vec<String> = f.metadata.keys().cloned().collect();
            keys.sort();
            Ok(keys)
        })?)
    }

    /// One metadata value (`default` if the key is missing).
    #[pyo3(signature = (key, default = None))]
    fn get_metadata<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let file = guard.as_ref().ok_or_else(|| QError::new(Kind::Io, "the GgufFile is closed"))?;
        match file.metadata.get(key) {
            Some(v) => Ok(Some(metadata_to_py(py, v)?)),
            None => Ok(default),
        }
    }

    /// All metadata as a dict (large token tables included).
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let file = guard.as_ref().ok_or_else(|| QError::new(Kind::Io, "the GgufFile is closed"))?;
        let d = PyDict::new(py);
        for (k, v) in &file.metadata {
            d.set_item(k, metadata_to_py(py, v)?)?;
        }
        Ok(d)
    }

    /// The per-tensor scale stored with a ternary tensor (`None` if the file does not carry one).
    fn tensor_scale(&self, name: &str) -> PyResult<Option<f32>> {
        Ok(self.with(|f| {
            Self::info(f, name)?;
            Ok(f.get_tensor_scale(name))
        })?)
    }

    /// The raw bytes of a tensor as stored in the file (F32, F16, I2_S, TQ1_0 and TQ2_0 only).
    fn read_raw<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let file = guard.as_ref().ok_or_else(|| QError::new(Kind::Io, "the GgufFile is closed"))?;
        let info = Self::info(file, name)?;
        let data = file.get_tensor_data(name).ok_or_else(|| {
            QError::new(
                Kind::Format,
                format!(
                    "cannot read tensor {name:?}: type {} is not readable or its data lies outside the file",
                    type_name(info.tensor_type)
                ),
            )
        })?;
        Ok(PyBytes::new(py, data))
    }

    /// Reads a ternary tensor (`I2_S`, `TQ1_0`, `TQ2_0`) as an int8 array of trits, shape `(rows, cols)`.
    fn read_trits<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let (trits, rows, cols) = py.detach(|| self.with(|f| Self::trits(f, name))).map_err(PyErr::from)?;
        int8_matrix(py, trits, rows, cols)
    }

    /// Transcodes a ternary tensor into a `PackedMatrix` ready for `gemv`.
    #[pyo3(signature = (name, *, kernel = None))]
    fn transcode(&self, py: Python<'_>, name: &str, kernel: Option<&str>) -> PyResult<PackedMatrix> {
        let kernel = Kernel::parse(kernel)?;
        let m = py
            .detach(|| {
                self.with(|f| {
                    let (trits, rows, cols) = Self::trits(f, name)?;
                    PackedMatrix::from_trits(&trits, rows, cols, kernel)
                })
            })
            .map_err(PyErr::from)?;
        Ok(m)
    }

    /// Unmaps the file. Further access raises `QuintritIOError`. Also done when the object is garbage collected.
    fn close(&self) {
        let mut guard = self.inner.write().unwrap_or_else(|p| p.into_inner());
        *guard = None;
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close();
    }

    fn __repr__(&self) -> String {
        format!("GgufFile({:?})", self.path.display().to_string())
    }
}

/// Opens a GGUF file; shorthand for `GgufFile(path)`.
#[pyfunction]
fn open(py: Python<'_>, path: PathBuf) -> PyResult<PyGguf> {
    PyGguf::new(py, path)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGguf>()?;
    m.add_class::<TensorInfo>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}
