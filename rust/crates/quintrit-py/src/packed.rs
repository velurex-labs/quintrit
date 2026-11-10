//! `PackedMatrix`, `pack`, `unpack` and `gemv`: the kernel level of the bindings.

use crate::arrays::{int8_matrix, read_int8, shaped};
use crate::error::{guard, Fail, Kind, QError, QResult};
use crate::kernels::{resolve_threads, Kernel};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use quintrit_core::{gemv_cols_supported, MAX_GEMV_COLS};
use quintrit_core::pack::{pack_matrix, unpack_matrix, BLOCK_BYTES_5, QK_5};
use std::sync::OnceLock;

/// Header of `PackedMatrix.to_bytes()`: magic, format version (u32), rows (u64), cols (u64), all little endian.
const MAGIC: &[u8; 4] = b"QPKM";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 4 + 4 + 8 + 8;

/// Packed byte of five zero trits (`1 + 3 + 9 + 28 + 84`); used to pad the row count up to the tile height.
const ZERO_BYTE: u8 = 125;
/// Largest valid packed byte; `251..=255` never occur in valid data.
const MAX_BYTE: u8 = 250;

/// Bytes of one packed row (planar layout: 160 trits per 32-byte block).
fn planar_row_bytes(cols: usize) -> usize {
    cols.div_ceil(QK_5) * BLOCK_BYTES_5
}

/// Builds the kernel's tiled layout from the planar bytes; rows are padded with zero rows up to a whole tile.
fn build_tile(planar: &[u8], rows: usize, cols: usize, kernel: Kernel) -> Vec<u8> {
    let padded_rows = rows.next_multiple_of(kernel.tile_rows());
    let row_bytes = planar_row_bytes(cols);
    let padded;
    let src: &[u8] = if padded_rows == rows {
        planar
    } else {
        let mut v = Vec::with_capacity(padded_rows * row_bytes);
        v.extend_from_slice(planar);
        v.resize(padded_rows * row_bytes, ZERO_BYTE);
        padded = v;
        &padded
    };
    match kernel {
        Kernel::Quintrit => quintrit_core::lut::transform_weights_lut(src, padded_rows, cols),
        Kernel::QuintritAvx512 => {
            #[cfg(target_arch = "x86_64")]
            {
                quintrit_core::lut::transform_weights_lut_64(src, padded_rows, cols)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("the 64-row AVX-512 layout is only built on x86-64")
            }
        }
    }
}

/// A ternary matrix packed at 5 trits per byte (1.60 bits per weight, velurex encoding).
///
/// The canonical storage is the planar packed form (`nbytes` bytes, what `to_bytes` serializes). The tiled layout
/// that the kernels read is built from it on demand and cached per kernel; the layout of the kernel selected at
/// pack time is built eagerly.
#[pyclass(frozen, name = "PackedMatrix", module = "quintrit")]
pub struct PackedMatrix {
    rows: usize,
    cols: usize,
    kernel: Kernel,
    planar: Vec<u8>,
    tiles: [OnceLock<Vec<u8>>; 2],
}

impl PackedMatrix {
    fn new(rows: usize, cols: usize, kernel: Kernel, planar: Vec<u8>) -> Self {
        Self { rows, cols, kernel, planar, tiles: [OnceLock::new(), OnceLock::new()] }
    }

    fn tiled(&self, kernel: Kernel) -> &[u8] {
        self.tiles[kernel.slot()].get_or_init(|| build_tile(&self.planar, self.rows, self.cols, kernel))
    }

    /// Packs row-major trits (validated here) for `kernel`.
    pub fn from_trits(trits: &[i8], rows: usize, cols: usize, kernel: Kernel) -> QResult<Self> {
        if let Some(pos) = trits.iter().position(|&t| !(-1..=1).contains(&t)) {
            return Err(QError::new(
                Kind::TritValue,
                format!(
                    "trits must be -1, 0 or +1, found {} at row {}, column {}",
                    trits[pos],
                    pos / cols,
                    pos % cols
                ),
            ));
        }
        let planar = guard("pack", || pack_matrix(trits, rows, cols))?;
        let m = Self::new(rows, cols, kernel, planar);
        guard("tile", || {
            m.tiled(kernel);
        })?;
        Ok(m)
    }
}

#[pymethods]
impl PackedMatrix {
    /// Number of weight rows (outputs of a GEMV).
    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    /// Number of weight columns (the length of the activation vector).
    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Name of the kernel the matrix was packed for (the default of `gemv`).
    #[getter]
    fn kernel(&self) -> &'static str {
        self.kernel.name()
    }

    /// Size of the packed weights in bytes (what `to_bytes` stores after its header).
    #[getter]
    fn nbytes(&self) -> usize {
        self.planar.len()
    }

    /// Bytes currently held in memory: the packed weights plus every tiled layout built so far.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.planar.len() + self.tiles.iter().filter_map(|t| t.get()).map(|t| t.len()).sum::<usize>()
    }

    /// Storage cost in bits per weight, including block padding (1.60 for column counts that are multiples of 160).
    #[getter]
    fn bits_per_weight(&self) -> f64 {
        self.planar.len() as f64 * 8.0 / (self.rows as f64 * self.cols as f64)
    }

    /// Unpacks to an int8 array of shape `(rows, cols)` with values in {-1, 0, +1}.
    fn unpack<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let trits = py
            .detach(|| guard("unpack", || unpack_matrix(&self.planar, self.rows, self.cols)))
            .map_err(PyErr::from)?;
        int8_matrix(py, trits, self.rows, self.cols)
    }

    /// Serializes to bytes: a 24 byte header (magic, version, rows, cols) followed by the packed weights.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.planar.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.rows as u64).to_le_bytes());
        out.extend_from_slice(&(self.cols as u64).to_le_bytes());
        out.extend_from_slice(&self.planar);
        PyBytes::new(py, &out)
    }

    /// Restores a matrix serialized by `to_bytes`. Accepts bytes, bytearray, memoryview or a uint8 array.
    #[staticmethod]
    #[pyo3(signature = (data, *, kernel = None))]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyAny>, kernel: Option<&str>) -> PyResult<PackedMatrix> {
        // bytes is the fast path; bytearray, memoryview and uint8 arrays go through memoryview(...).tobytes()
        // (which, unlike bytes(...), rejects integers and strings instead of converting them).
        let owned: Bound<'_, PyBytes> = match data.cast::<PyBytes>() {
            Ok(b) => b.clone(),
            Err(_) => py
                .import("builtins")
                .and_then(|m| m.getattr("memoryview"))
                .and_then(|mv| mv.call1((data,)))
                .and_then(|view| view.call_method0("tobytes"))
                .and_then(|o| o.cast_into::<PyBytes>().map_err(PyErr::from))
                .map_err(|_| QError::new(Kind::DType, format!("data must be a bytes-like object, got {}", data.get_type())))?,
        };
        let bytes = owned.as_bytes().to_vec();
        let kernel = Kernel::parse(kernel)?;
        let m = py.detach(|| parse_serialized(&bytes, kernel)).map_err(PyErr::from)?;
        py.detach(|| guard("tile", || m.tiled(kernel))).map_err(PyErr::from)?;
        Ok(m)
    }

    /// Computes `W @ x` for an int8 activation vector `x` (or a batch of vectors, shape `(batch, cols)`).
    ///
    /// See `quintrit.gemv`.
    #[pyo3(signature = (activations, *, kernel = None, threads = None, dtype = None, scale = None))]
    fn gemv<'py>(
        &self,
        py: Python<'py>,
        activations: &Bound<'py, PyAny>,
        kernel: Option<&str>,
        threads: Option<usize>,
        dtype: Option<&Bound<'py, PyAny>>,
        scale: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        gemv_impl(py, activations, self, kernel, threads, dtype, scale)
    }

    fn __eq__(&self, other: PyRef<'_, PackedMatrix>) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.planar == other.planar
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let ctor = py.get_type::<PackedMatrix>().getattr("from_bytes")?;
        Ok((ctor, PyTuple::new(py, [self.to_bytes(py)])?))
    }

    fn __repr__(&self) -> String {
        format!(
            "PackedMatrix(rows={}, cols={}, kernel={:?}, nbytes={}, bits_per_weight={:.3})",
            self.rows,
            self.cols,
            self.kernel.name(),
            self.planar.len(),
            self.bits_per_weight()
        )
    }
}

fn parse_serialized(bytes: &[u8], kernel: Kernel) -> Result<PackedMatrix, Fail> {
    let bad = |msg: String| Fail::Q(QError::new(Kind::Serialization, msg));
    if bytes.len() < HEADER_LEN {
        return Err(bad(format!("data is too short for a PackedMatrix header ({} bytes)", bytes.len())));
    }
    if &bytes[..4] != MAGIC {
        return Err(bad("bad magic number: this is not a serialized PackedMatrix".to_string()));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(bad(format!("unsupported PackedMatrix format version {version} (expected {FORMAT_VERSION})")));
    }
    let rows = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let cols = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let (rows, cols) = match (usize::try_from(rows), usize::try_from(cols)) {
        (Ok(r), Ok(c)) if r > 0 && c > 0 => (r, c),
        _ => return Err(bad(format!("invalid dimensions {rows} x {cols}"))),
    };
    let expected = cols
        .div_ceil(QK_5)
        .checked_mul(BLOCK_BYTES_5)
        .and_then(|row_bytes| row_bytes.checked_mul(rows))
        .ok_or_else(|| bad(format!("dimensions {rows} x {cols} are too large")))?;
    let payload = &bytes[HEADER_LEN..];
    if payload.len() != expected {
        return Err(bad(format!(
            "payload has {} bytes but a {rows} x {cols} matrix needs {expected}",
            payload.len()
        )));
    }
    if let Some(pos) = payload.iter().position(|&b| b > MAX_BYTE) {
        return Err(bad(format!("invalid packed byte {} at payload offset {pos} (valid range is 0..=250)", payload[pos])));
    }
    Ok(PackedMatrix::new(rows, cols, kernel, payload.to_vec()))
}

/// Packs a 2-D int8 array of trits (values -1, 0, +1) into a `PackedMatrix`.
///
/// `kernel` selects the tiled layout that is built eagerly (`"auto"`, `"quintrit"` or `"quintrit_512"`); any
/// kernel can be used with the matrix later, its layout is then built on first use.
#[pyfunction]
#[pyo3(signature = (trits, *, kernel = None))]
pub fn pack(py: Python<'_>, trits: &Bound<'_, PyAny>, kernel: Option<&str>) -> PyResult<PackedMatrix> {
    let arr = read_int8(trits, "trits", &[2])?;
    let kernel = Kernel::parse(kernel)?;
    let (rows, cols) = (arr.shape[0], arr.shape[1]);
    let matrix = py
        .detach(|| PackedMatrix::from_trits(&arr.data, rows, cols, kernel))
        .map_err(PyErr::from)?;
    Ok(matrix)
}

/// Unpacks a `PackedMatrix` to an int8 array of shape `(rows, cols)`.
#[pyfunction]
pub fn unpack<'py>(py: Python<'py>, packed: &Bound<'py, PackedMatrix>) -> PyResult<Bound<'py, PyAny>> {
    packed.get().unpack(py)
}

/// Computes `W @ x` for the packed ternary matrix `W` and an int8 activation vector `x`.
///
/// `activations` is a 1-D int8 array of length `cols`, or a 2-D array `(batch, cols)` whose rows are multiplied one
/// after the other. The result has shape `(rows,)` or `(batch, rows)`. The integer results are exact for the full
/// int8 range (-128..127). The GIL is released while the kernel runs.
///
/// * `kernel`: `None` uses the kernel the matrix was packed for; `"auto"`, `"quintrit"` or `"quintrit_512"` select one.
/// * `threads`: worker threads (default: number of logical CPUs).
/// * `dtype`: `"int32"` (default) or `"float32"`. The kernels accumulate in float32, so int32 results are exact only
///   while the magnitude stays below 2**24 (`cols <= 131072` for full-range activations).
/// * `scale`: optional factor applied to the result; implies `dtype="float32"`.
#[pyfunction]
#[pyo3(signature = (activations, packed, *, kernel = None, threads = None, dtype = None, scale = None))]
pub fn gemv<'py>(
    py: Python<'py>,
    activations: &Bound<'py, PyAny>,
    packed: &Bound<'py, PackedMatrix>,
    kernel: Option<&str>,
    threads: Option<usize>,
    dtype: Option<&Bound<'py, PyAny>>,
    scale: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    gemv_impl(py, activations, packed.get(), kernel, threads, dtype, scale)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutDType {
    Int32,
    Float32,
}

fn parse_dtype(dtype: Option<&Bound<'_, PyAny>>, scale: Option<f64>) -> Result<OutDType, QError> {
    let Some(obj) = dtype else {
        return Ok(if scale.is_some() { OutDType::Float32 } else { OutDType::Int32 });
    };
    if obj.is_none() {
        return Ok(if scale.is_some() { OutDType::Float32 } else { OutDType::Int32 });
    }
    // Accept the names as strings, numpy scalar types (numpy.int32) and numpy.dtype instances.
    let name: String = if let Ok(s) = obj.extract::<String>() {
        s
    } else if let Ok(n) = obj.getattr("name").and_then(|n| n.extract::<String>()) {
        n
    } else if let Ok(n) = obj.getattr("__name__").and_then(|n| n.extract::<String>()) {
        n
    } else {
        return Err(QError::new(Kind::DType, "dtype must be 'int32' or 'float32'"));
    };
    match name.as_str() {
        "int32" => {
            if scale.is_some() {
                Err(QError::new(Kind::DType, "a scale factor needs dtype='float32'"))
            } else {
                Ok(OutDType::Int32)
            }
        }
        "float32" => Ok(OutDType::Float32),
        other => Err(QError::new(Kind::DType, format!("unsupported output dtype {other:?}; use 'int32' or 'float32'"))),
    }
}

/// Raw kernel result for one activation vector: float32 values holding the exact integer sums.
fn run_kernel(kernel: Kernel, packed: &PackedMatrix, acts: &[i8], threads: usize) -> Vec<f32> {
    let cols = packed.cols;
    let padded_rows = packed.rows.next_multiple_of(kernel.tile_rows());
    let weights = packed.tiled(kernel);
    let mut out = vec![0.0f32; padded_rows];
    match kernel {
        Kernel::Quintrit => {
            let qlut = quintrit_core::lut::build_activation_lut_wide(acts, cols);
            quintrit_core::lut::gemv_lut_wide_parallel(weights, &qlut, padded_rows, cols, &mut out, threads);
        }
        Kernel::QuintritAvx512 => {
            #[cfg(target_arch = "x86_64")]
            {
                let qlut = quintrit_core::lut::build_activation_lut_wide_avx512(acts, cols);
                quintrit_core::lut::gemv_lut_64_wide_parallel_avx512(weights, &qlut, padded_rows, cols, &mut out, threads);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("the AVX-512 kernel is only available on x86-64")
            }
        }
    }
    out.truncate(packed.rows);
    out
}

fn gemv_impl<'py>(
    py: Python<'py>,
    activations: &Bound<'py, PyAny>,
    packed: &PackedMatrix,
    kernel: Option<&str>,
    threads: Option<usize>,
    dtype: Option<&Bound<'py, PyAny>>,
    scale: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let acts = read_int8(activations, "activations", &[1, 2])?;
    let (batch, len) = if acts.shape.len() == 2 { (acts.shape[0], acts.shape[1]) } else { (1, acts.shape[0]) };
    if len != packed.cols {
        return Err(QError::new(
            Kind::Shape,
            format!("activations have length {len} but the packed matrix has {} columns", packed.cols),
        )
        .into());
    }
    if !gemv_cols_supported(packed.cols) {
        return Err(QError::new(
            Kind::Shape,
            format!(
                "the packed matrix has {} columns, but gemv supports at most {MAX_GEMV_COLS}, the largest count for which the accumulators are exact",
                packed.cols
            ),
        )
        .into());
    }
    let kernel = match kernel {
        Some(name) => Kernel::parse(Some(name))?,
        None => packed.kernel,
    };
    let threads = resolve_threads(threads)?;
    let out_dtype = parse_dtype(dtype, scale)?;
    if scale.is_some_and(|s| !s.is_finite()) {
        return Err(QError::new(Kind::InvalidArgument, "scale must be finite").into());
    }

    let raw: Vec<f32> = py
        .detach(|| {
            guard("gemv", || {
                let mut all = Vec::with_capacity(batch * packed.rows);
                for b in 0..batch {
                    all.extend(run_kernel(kernel, packed, &acts.data[b * len..(b + 1) * len], threads));
                }
                all
            })
        })
        .map_err(PyErr::from)?;

    let shape: Vec<usize> = if acts.shape.len() == 2 { vec![batch, packed.rows] } else { vec![packed.rows] };
    match out_dtype {
        OutDType::Int32 => shaped(py, raw.iter().map(|&v| v as i32).collect::<Vec<i32>>(), &shape),
        OutDType::Float32 => {
            let data = match scale {
                Some(s) => {
                    let s = s as f32;
                    raw.into_iter().map(|v| v * s).collect()
                }
                None => raw,
            };
            shaped(py, data, &shape)
        }
    }
}
