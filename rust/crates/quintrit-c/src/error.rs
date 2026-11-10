//! Status codes, the thread-local last-error slot and the panic guard shared by every entry point.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Success.
pub const QUINTRIT_OK: c_int = 0;
/// A required pointer argument is NULL.
pub const QUINTRIT_ERR_NULL_POINTER: c_int = 1;
/// An argument has an invalid value (unknown enumeration, bad flag, misaligned pointer, overlapping buffers, ...).
pub const QUINTRIT_ERR_INVALID_ARGUMENT: c_int = 2;
/// A matrix shape is invalid (zero or too large dimension, arithmetic overflow).
pub const QUINTRIT_ERR_INVALID_SHAPE: c_int = 3;
/// A caller-provided buffer is too small; the required size is reported where the function has an `out_*` size.
pub const QUINTRIT_ERR_BUFFER_TOO_SMALL: c_int = 4;
/// Data holds an invalid value (a trit outside `{-1, 0, +1}`, a packed byte above 250).
pub const QUINTRIT_ERR_INVALID_VALUE: c_int = 5;
/// The requested kernel, tensor type or feature is not available on this CPU or in this build.
pub const QUINTRIT_ERR_UNSUPPORTED: c_int = 6;
/// A file could not be opened or read.
pub const QUINTRIT_ERR_IO: c_int = 7;
/// A file is not a valid GGUF file, or a tensor cannot be decoded.
pub const QUINTRIT_ERR_FORMAT: c_int = 8;
/// A named tensor or metadata key does not exist.
pub const QUINTRIT_ERR_NOT_FOUND: c_int = 9;
/// A model file is well formed but cannot be turned into a runnable model.
pub const QUINTRIT_ERR_MODEL: c_int = 10;
/// The prompt does not fit into the context window of the session.
pub const QUINTRIT_ERR_CONTEXT_OVERFLOW: c_int = 11;
/// The handle is in use by another thread, or has sessions that forbid the operation.
pub const QUINTRIT_ERR_BUSY: c_int = 12;
/// An internal error (a Rust panic that was caught).
pub const QUINTRIT_ERR_INTERNAL: c_int = 13;
/// The handle is NULL, of the wrong type, or already freed.
pub const QUINTRIT_ERR_INVALID_HANDLE: c_int = 14;

/// An error that is reported to the caller as a status code plus a message.
#[derive(Debug)]
pub struct Failure {
    pub code: c_int,
    pub msg: String,
}

impl Failure {
    pub fn new(code: c_int, msg: impl Into<String>) -> Self {
        Self { code, msg: msg.into() }
    }
}

/// Result type of every internal operation.
pub type Res<T> = Result<T, Failure>;

/// Returns early with a [`Failure`].
macro_rules! fail {
    ($code:expr, $($arg:tt)*) => {
        return Err($crate::error::Failure::new($code, format!($($arg)*)))
    };
}
pub(crate) use fail;

thread_local! {
    static LAST_ERROR: RefCell<(c_int, CString)> = RefCell::new((0, CString::default()));
}

fn set_last(code: c_int, msg: &str) {
    let text = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    // The slot may already be gone while a thread is shutting down; the status code is returned anyway.
    let _ = LAST_ERROR.try_with(|slot| *slot.borrow_mut() = (code, text));
}

fn clear_last() {
    let _ = LAST_ERROR.try_with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.0 != 0 {
            *slot = (0, CString::default());
        }
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Runs the body of an entry point: clears the last error, converts a [`Failure`] into a status code plus message, and
/// converts a panic into [`QUINTRIT_ERR_INTERNAL`] (only possible when panics unwind, see the crate documentation).
pub fn guard<F: FnOnce() -> Res<()>>(body: F) -> c_int {
    clear_last();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => QUINTRIT_OK,
        Ok(Err(f)) => {
            set_last(f.code, &f.msg);
            f.code
        }
        Err(payload) => {
            set_last(QUINTRIT_ERR_INTERNAL, &format!("internal error (panic): {}", panic_message(payload.as_ref())));
            QUINTRIT_ERR_INTERNAL
        }
    }
}

/// Returns a static, NUL-terminated description of a status code.
pub fn status_text(code: c_int) -> &'static [u8] {
    match code {
        QUINTRIT_OK => b"success\0",
        QUINTRIT_ERR_NULL_POINTER => b"required pointer argument is NULL\0",
        QUINTRIT_ERR_INVALID_ARGUMENT => b"invalid argument\0",
        QUINTRIT_ERR_INVALID_SHAPE => b"invalid matrix shape\0",
        QUINTRIT_ERR_BUFFER_TOO_SMALL => b"buffer too small\0",
        QUINTRIT_ERR_INVALID_VALUE => b"invalid data value\0",
        QUINTRIT_ERR_UNSUPPORTED => b"unsupported on this CPU or in this build\0",
        QUINTRIT_ERR_IO => b"file I/O error\0",
        QUINTRIT_ERR_FORMAT => b"invalid or unsupported file format\0",
        QUINTRIT_ERR_NOT_FOUND => b"not found\0",
        QUINTRIT_ERR_MODEL => b"model cannot be loaded\0",
        QUINTRIT_ERR_CONTEXT_OVERFLOW => b"prompt does not fit into the context window\0",
        QUINTRIT_ERR_BUSY => b"handle is busy\0",
        QUINTRIT_ERR_INTERNAL => b"internal error\0",
        QUINTRIT_ERR_INVALID_HANDLE => b"invalid handle\0",
        _ => b"unknown status code\0",
    }
}

/// Returns a short description of a status code. The returned string is static and never NULL.
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_status_string(status: c_int) -> *const c_char {
    status_text(status).as_ptr() as *const c_char
}

/// Returns the status code of the last failed call on the calling thread (0 if the last call succeeded).
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_last_error_code() -> c_int {
    LAST_ERROR.try_with(|slot| slot.borrow().0).unwrap_or(0)
}

/// Returns the message of the last failed call on the calling thread; an empty string if there is none.
///
/// The pointer stays valid until the next `quintrit_*` call on the same thread. It is never NULL.
#[unsafe(no_mangle)]
pub extern "C" fn quintrit_last_error_message() -> *const c_char {
    static EMPTY: &[u8] = b"\0";
    LAST_ERROR
        .try_with(|slot| {
            let slot = slot.borrow();
            if slot.0 == 0 { EMPTY.as_ptr() as *const c_char } else { slot.1.as_ptr() }
        })
        .unwrap_or(EMPTY.as_ptr() as *const c_char)
}
