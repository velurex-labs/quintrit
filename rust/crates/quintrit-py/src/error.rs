//! Error type of the bindings and its mapping onto the Python exception hierarchy.
//!
//! The exception classes themselves live in the pure-Python module `quintrit._errors` (so that they can inherit from
//! several builtin exceptions, for example `ShapeError(QuintritError, ValueError)`). Rust code builds a [`QError`]
//! that carries the class name and the message and converts it to the Python exception on the way out.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The exception class a [`QError`] is raised as (a class of `quintrit._errors`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Wrong array shape or dimension.
    Shape,
    /// Wrong array dtype or object type.
    DType,
    /// Invalid scalar argument (thread count, sampling parameter, ...).
    InvalidArgument,
    /// A trit outside `{-1, 0, +1}`.
    TritValue,
    /// Unknown kernel name.
    Kernel,
    /// Known kernel that this CPU cannot run.
    UnsupportedKernel,
    /// Malformed serialized `PackedMatrix`.
    Serialization,
    /// Malformed or unsupported file content.
    Format,
    /// Tensor name missing from a GGUF file.
    TensorNotFound,
    /// File access failure.
    Io,
    /// File does not exist.
    FileNotFound,
    /// The model could not be loaded.
    ModelLoad,
    /// Tokenizer failure.
    Tokenizer,
    /// Generation failure.
    Generation,
    /// A token stream was invalidated by a newer generation or `reset()`.
    StaleStream,
    /// The prompt does not fit into the context window.
    ContextOverflow,
    /// A bug or an unexpected panic inside the native code.
    Internal,
}

impl Kind {
    fn class_name(self) -> &'static str {
        match self {
            Kind::Shape => "ShapeError",
            Kind::DType => "DTypeError",
            Kind::InvalidArgument => "InvalidArgumentError",
            Kind::TritValue => "TritValueError",
            Kind::Kernel => "KernelError",
            Kind::UnsupportedKernel => "UnsupportedKernelError",
            Kind::Serialization => "SerializationError",
            Kind::Format => "FormatError",
            Kind::TensorNotFound => "TensorNotFoundError",
            Kind::Io => "QuintritIOError",
            Kind::FileNotFound => "QuintritFileNotFoundError",
            Kind::ModelLoad => "ModelLoadError",
            Kind::Tokenizer => "TokenizerError",
            Kind::Generation => "GenerationError",
            Kind::StaleStream => "StaleStreamError",
            Kind::ContextOverflow => "ContextOverflowError",
            Kind::Internal => "InternalError",
        }
    }
}

/// An error that does not need the GIL to be created (so it can be produced while the GIL is released).
#[derive(Debug, Clone)]
pub struct QError {
    pub kind: Kind,
    pub msg: String,
}

impl QError {
    pub fn new(kind: Kind, msg: impl Into<String>) -> Self {
        Self { kind, msg: msg.into() }
    }
}

pub type QResult<T> = Result<T, QError>;

impl From<QError> for PyErr {
    fn from(e: QError) -> PyErr {
        Python::attach(|py| {
            let built = py
                .import("quintrit._errors")
                .and_then(|m| m.getattr(e.kind.class_name()))
                .and_then(|c| c.call1((e.msg.clone(),)));
            match built {
                Ok(instance) => PyErr::from_value(instance),
                Err(_) => PyRuntimeError::new_err(format!("{}: {}", e.kind.class_name(), e.msg)),
            }
        })
    }
}

/// Error type of code that runs with the GIL released: either a [`QError`] or an already formed Python error
/// (for example `KeyboardInterrupt`). Converted to a `PyErr` after the GIL has been re-acquired.
#[derive(Debug)]
pub enum Fail {
    Q(QError),
    Py(PyErr),
}

impl From<QError> for Fail {
    fn from(e: QError) -> Self {
        Fail::Q(e)
    }
}

impl From<PyErr> for Fail {
    fn from(e: PyErr) -> Self {
        Fail::Py(e)
    }
}

impl From<Fail> for PyErr {
    fn from(f: Fail) -> PyErr {
        match f {
            Fail::Q(q) => q.into(),
            Fail::Py(p) => p,
        }
    }
}

/// Text of a panic payload.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Runs `f` and turns a panic of the underlying Rust code into an `InternalError`.
///
/// All inputs are validated before the native kernels run, so this is a safety net and not a control-flow
/// mechanism; it relies on the crate being built with `panic = "unwind"`.
pub fn guard<T>(what: &str, f: impl FnOnce() -> T) -> QResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|p| {
        QError::new(
            Kind::Internal,
            format!("internal error in {what}: {} (please report this as a bug)", panic_message(&*p)),
        )
    })
}
