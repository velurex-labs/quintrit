//! # Quintrit C API (`quintrit-c`)
//!
//! A stable, un-mangled `extern "C"` interface to the Quintrit ternary kernels, the GGUF reader and the inference
//! runtime. The shared library is `quintrit_c.dll` / `libquintrit_c.so` / `libquintrit_c.dylib`, the static library
//! `quintrit_c.lib` / `libquintrit_c.a`, and the header is `include/quintrit.h` (plus the optional C++ wrapper
//! `include/quintrit.hpp`). See the crate README for usage, error handling, threading and lifetime rules.
//!
//! ## Layers
//! - **Kernel level** ([`kernel`], [`tiles`]): feature and kernel queries, `quintrit_pack_matrix`,
//!   `quintrit_unpack_matrix`, `quintrit_gemv_i32` / `quintrit_gemv_f32`.
//! - **Format level** ([`mod@format`]): `quintrit_gguf_*` to open a GGUF file, list tensors and convert ternary tensors.
//! - **Runtime level** ([`runtime`]): `quintrit_model_*` and `quintrit_session_*` for tokenization and generation.
//!
//! ## Rules every entry point follows
//! - It returns an `int` status (`0` = `QUINTRIT_OK`); results go through out-parameters. The few functions that cannot
//!   fail (`quintrit_version`, `quintrit_abi_version`, `quintrit_status_string`, `quintrit_kernel_name`,
//!   `quintrit_last_error_*`) return their value directly.
//! - On failure it stores a message in a thread-local slot, readable with `quintrit_last_error_message`.
//! - It validates every pointer, length, alignment, shape and enumeration it receives and returns an error instead of
//!   panicking. As a second line of defence the body runs inside `catch_unwind`.
//!
//! ## Panics and `panic = "abort"`
//! The workspace `release` profile sets `panic = "abort"`. With it `catch_unwind` cannot catch anything: a panic that
//! slips through the argument validation (a bug, or a malformed model that the validation of `quintrit_model_open`
//! did not foresee) terminates the host process. The default `dev` and `test` profiles unwind, and so does the
//! `release-unwind` profile of the workspace (`cargo build --profile release-unwind -p quintrit-c`), which reports a
//! panic as `QUINTRIT_ERR_INTERNAL` instead. Use that profile for host applications that must not crash.

pub mod error;
pub mod format;
mod gguf_check;
pub mod kernel;
pub mod runtime;
pub mod tiles;
mod util;

pub use error::*;
pub use format::*;
pub use kernel::*;
pub use runtime::*;
pub use tiles::*;
