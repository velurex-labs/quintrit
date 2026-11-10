//! Opens a GGUF file for the C API and maps parser errors to status codes.
//!
//! The bounds checking of hostile or damaged files (counts, lengths, offsets, overlaps, alignment) lives in the GGUF
//! reader of `quintrit-format`, which every user of the SDK gets. What is left here is what only the C API needs: a
//! path that is not a regular file is an I/O error (the memory map of a directory or an empty file fails with a
//! confusing message), and a malformed file is a format error while any other failure is an I/O error.

use crate::error::{fail, Failure, Res, QUINTRIT_ERR_FORMAT, QUINTRIT_ERR_IO};
use quintrit_format::GgufFile;
use std::io::ErrorKind;
use std::path::Path;

/// Opens and parses the GGUF file at `path`.
pub fn open_checked(path: &Path) -> Res<GgufFile> {
    let meta = std::fs::metadata(path).map_err(|e| Failure::new(QUINTRIT_ERR_IO, format!("cannot open '{}': {e}", path.display())))?;
    if !meta.is_file() {
        fail!(QUINTRIT_ERR_IO, "'{}' is not a regular file", path.display());
    }
    if meta.len() < 4 {
        fail!(QUINTRIT_ERR_FORMAT, "'{}' is too short to be a GGUF file", path.display());
    }
    GgufFile::open(path).map_err(|e| {
        let code = match e.kind() {
            ErrorKind::InvalidData | ErrorKind::UnexpectedEof => QUINTRIT_ERR_FORMAT,
            _ => QUINTRIT_ERR_IO,
        };
        Failure::new(code, format!("cannot parse '{}': {e}", path.display()))
    })
}
