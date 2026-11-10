//! Format level: open a GGUF file, list its tensors, read metadata, convert a ternary tensor to the tiled layout.

use crate::error::{
    fail, guard, Res, QUINTRIT_ERR_BUFFER_TOO_SMALL, QUINTRIT_ERR_FORMAT, QUINTRIT_ERR_NOT_FOUND, QUINTRIT_ERR_UNSUPPORTED,
};
use crate::kernel::resolve;
use crate::tiles::{layout, pack_tiled, TiledLayout};
use crate::util::{
    c_str, check_handle, check_out_struct, out_slice, write_out, HasMagic, MAGIC_GGUF,
};
use quintrit_format::{GgufFile, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S, GGML_TYPE_Q8_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CString};
use std::path::Path;

/// Maximum number of tensor dimensions reported in `quintrit_tensor_info_t`.
pub const QUINTRIT_MAX_DIMS: usize = 8;

/// GGML type ids known by name (mirrors the `QUINTRIT_GGML_*` macros of the header).
pub const QUINTRIT_GGML_F32: i32 = GGML_TYPE_F32 as i32;
pub const QUINTRIT_GGML_F16: i32 = GGML_TYPE_F16 as i32;
pub const QUINTRIT_GGML_Q8_0: i32 = GGML_TYPE_Q8_0 as i32;
pub const QUINTRIT_GGML_BF16: i32 = GGML_TYPE_BF16 as i32;
pub const QUINTRIT_GGML_TQ1_0: i32 = GGML_TYPE_TQ1_0 as i32;
pub const QUINTRIT_GGML_TQ2_0: i32 = GGML_TYPE_TQ2_0 as i32;
pub const QUINTRIT_GGML_I2_S: i32 = GGML_TYPE_I2_S as i32;

/// Description of one tensor of a GGUF file (`quintrit_tensor_info_t`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TensorInfo {
    pub struct_size: u32,
    /// GGML type id (0 = F32, 1 = F16, 8 = Q8_0, 30 = BF16, 34 = TQ1_0, 35 = TQ2_0, 36 = I2_S).
    pub ggml_type: i32,
    pub n_dims: u32,
    /// 1 if the tensor holds ternary weights this library can convert (I2_S, TQ1_0, TQ2_0).
    pub is_ternary: i32,
    /// Dimensions as stored in the file: `shape[0]` is the innermost (fastest) one, so a weight matrix is
    /// `[cols, rows]`. Only the first `min(n_dims, 8)` entries are meaningful.
    pub shape: [u64; QUINTRIT_MAX_DIMS],
    pub num_elements: u64,
    /// Offset of the tensor inside the data section of the file.
    pub offset: u64,
    /// Size of the raw payload in bytes (0 for tensor types this library cannot read).
    pub data_bytes: u64,
    /// Tensor name, NUL-terminated; valid until the GGUF handle is closed.
    pub name: *const c_char,
    /// GGML type name, NUL-terminated and static.
    pub type_name: *const c_char,
}

/// File-level summary (`quintrit_gguf_info_t`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GgufInfo {
    pub struct_size: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    /// Absolute file offset of the tensor data section.
    pub data_offset: u64,
}

struct Entry {
    name: CString,
    info: TensorInfo,
}

/// Opaque handle: an open, memory-mapped GGUF file (`quintrit_gguf_t`).
#[repr(C)]
pub struct GgufHandle {
    magic: u32,
    file: GgufFile,
    entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

impl HasMagic for GgufHandle {
    const MAGIC: u32 = MAGIC_GGUF;
    fn magic(&self) -> u32 {
        self.magic
    }
}

fn type_name(t: u32) -> &'static [u8] {
    match t {
        GGML_TYPE_F32 => b"F32\0",
        GGML_TYPE_F16 => b"F16\0",
        GGML_TYPE_Q8_0 => b"Q8_0\0",
        GGML_TYPE_BF16 => b"BF16\0",
        GGML_TYPE_TQ1_0 => b"TQ1_0\0",
        GGML_TYPE_TQ2_0 => b"TQ2_0\0",
        GGML_TYPE_I2_S => b"I2_S\0",
        _ => b"UNKNOWN\0",
    }
}

fn open_gguf(path: &Path) -> Res<GgufHandle> {
    let file = crate::gguf_check::open_checked(path)?;
    let mut entries: Vec<Entry> = file
        .tensors
        .values()
        .map(|t| {
            let mut shape = [0u64; QUINTRIT_MAX_DIMS];
            for (dst, &d) in shape.iter_mut().zip(&t.shape) {
                *dst = d as u64;
            }
            let name = CString::new(t.name.replace('\0', " ")).unwrap_or_default();
            let info = TensorInfo {
                struct_size: size_of::<TensorInfo>() as u32,
                ggml_type: t.tensor_type as i32,
                n_dims: t.shape.len() as u32,
                is_ternary: quintrit_format::is_supported_ternary_type(t.tensor_type) as i32,
                shape,
                num_elements: t.num_elements() as u64,
                offset: t.offset,
                data_bytes: file.get_tensor_data(&t.name).map_or(0, |d| d.len() as u64),
                name: name.as_ptr(),
                type_name: type_name(t.tensor_type).as_ptr() as *const c_char,
            };
            Entry { name, info }
        })
        .collect();
    entries.sort_by(|a, b| (a.info.offset, &a.name).cmp(&(b.info.offset, &b.name)));
    let index = entries.iter().enumerate().map(|(i, e)| (e.name.to_string_lossy().into_owned(), i)).collect();
    Ok(GgufHandle { magic: MAGIC_GGUF, file, entries, index })
}

/// Rows and columns of a 2-D weight tensor (`shape = [cols, rows]`).
fn matrix_shape(e: &Entry) -> Res<(usize, usize)> {
    let name = e.name.to_string_lossy();
    if e.info.n_dims != 2 {
        fail!(QUINTRIT_ERR_FORMAT, "tensor '{name}' has {} dimensions; a weight matrix needs exactly 2", e.info.n_dims);
    }
    let (cols, rows) = (e.info.shape[0], e.info.shape[1]);
    let (cols, rows) = (usize::try_from(cols).unwrap_or(usize::MAX), usize::try_from(rows).unwrap_or(usize::MAX));
    Ok((rows, cols))
}

fn find<'a>(h: &'a GgufHandle, name: &str) -> Res<&'a Entry> {
    match h.index.get(name) {
        Some(&i) => Ok(&h.entries[i]),
        None => fail!(QUINTRIT_ERR_NOT_FOUND, "tensor '{name}' not found"),
    }
}

/// The per-tensor weight scale stored with the tensor.
fn tensor_scale(h: &GgufHandle, name: &str, ggml_type: u32) -> Res<f32> {
    Ok(match h.file.get_tensor_scale(name) {
        Some(s) => s,
        None if matches!(ggml_type, GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => fail!(
            QUINTRIT_ERR_FORMAT,
            "tensor '{name}' has differing per-block scales, which one per-tensor scale cannot represent"
        ),
        None => 1.0,
    })
}

/// Decodes a ternary tensor into trits and returns the layout plus the scale.
fn load_trits(h: &GgufHandle, e: &Entry) -> Res<(Vec<i8>, usize, usize, f32)> {
    let name = e.name.to_string_lossy().into_owned();
    let ggml_type = e.info.ggml_type as u32;
    if !quintrit_format::is_supported_ternary_type(ggml_type) {
        fail!(
            QUINTRIT_ERR_UNSUPPORTED,
            "tensor '{name}' has type {} which is not ternary (supported: I2_S, TQ1_0, TQ2_0)",
            String::from_utf8_lossy(&type_name(ggml_type)[..type_name(ggml_type).len() - 1])
        );
    }
    let (rows, cols) = matrix_shape(e)?;
    if rows == 0 || cols == 0 {
        fail!(QUINTRIT_ERR_FORMAT, "tensor '{name}' has an empty dimension");
    }
    if ggml_type == GGML_TYPE_I2_S && cols % 128 != 0 {
        fail!(QUINTRIT_ERR_FORMAT, "I2_S tensor '{name}': cols = {cols} is not a multiple of 128");
    }
    let raw = h
        .file
        .get_tensor_data(&name)
        .ok_or_else(|| crate::error::Failure::new(QUINTRIT_ERR_FORMAT, format!("tensor '{name}' lies outside the file")))?;
    let scale = tensor_scale(h, &name, ggml_type)?;
    let decoder = quintrit_format::get_ternary_decoder(ggml_type).map_err(|m| crate::error::Failure::new(QUINTRIT_ERR_UNSUPPORTED, m))?;
    let trits = decoder.unpack_trits(raw, rows, cols).map_err(|m| crate::error::Failure::new(QUINTRIT_ERR_FORMAT, format!("tensor '{name}': {m}")))?;
    Ok((trits, rows, cols, scale))
}

// ---------------------------------------------------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------------------------------------------------

/// Opens and memory-maps a GGUF file (version 2 or 3) and indexes its tensors.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 string; `out_gguf` must point to a handle pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_open(path: *const c_char, out_gguf: *mut *mut GgufHandle) -> c_int {
    guard(|| {
        // SAFETY: validated by c_str / write_out.
        let path = unsafe { c_str(path, "path")? };
        crate::util::out_ptr(out_gguf, "out_gguf")?;
        unsafe { out_gguf.write(std::ptr::null_mut()) };
        let handle = Box::new(open_gguf(Path::new(path))?);
        unsafe { out_gguf.write(Box::into_raw(handle)) };
        Ok(())
    })
}

/// Closes a GGUF file. Passing NULL is allowed and does nothing. All pointers obtained from the handle (tensor names,
/// raw tensor data) become invalid.
///
/// # Safety
/// `gguf` must be NULL or a handle returned by `quintrit_gguf_open` that is not used by another thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_close(gguf: *mut GgufHandle) -> c_int {
    guard(|| {
        if gguf.is_null() {
            return Ok(());
        }
        // SAFETY: check_handle validates the tag before the Box is reconstructed.
        unsafe { check_handle(gguf, "gguf")? };
        unsafe {
            (*gguf).magic = 0;
            drop(Box::from_raw(gguf));
        }
        Ok(())
    })
}

/// Returns the file-level summary.
///
/// # Safety
/// `gguf` must be a live handle; `out_info` must point to a `quintrit_gguf_info_t` with `struct_size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_info(gguf: *const GgufHandle, out_info: *mut GgufInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / check_out_struct.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        let info = GgufInfo {
            struct_size: size_of::<GgufInfo>() as u32,
            version: h.file.version,
            tensor_count: h.entries.len() as u64,
            metadata_count: h.file.metadata.len() as u64,
            data_offset: h.file.data_start() as u64,
        };
        unsafe { out.write(info) };
        Ok(())
    })
}

/// Returns the description of the tensor at `index` (0-based, in file order). `index` must be below
/// `quintrit_gguf_info().tensor_count`.
///
/// # Safety
/// As `quintrit_gguf_info`, with a `quintrit_tensor_info_t` output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_tensor_at(gguf: *const GgufHandle, index: usize, out_info: *mut TensorInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / check_out_struct.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        match h.entries.get(index) {
            Some(e) => unsafe { out.write(e.info) },
            None => fail!(
                crate::error::QUINTRIT_ERR_INVALID_ARGUMENT,
                "tensor index {index} is out of range ({} tensors)",
                h.entries.len()
            ),
        }
        Ok(())
    })
}

/// Looks a tensor up by name; `QUINTRIT_ERR_NOT_FOUND` if there is none.
///
/// # Safety
/// As `quintrit_gguf_tensor_at`, with `name` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_tensor_find(gguf: *const GgufHandle, name: *const c_char, out_info: *mut TensorInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / check_out_struct.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let name = unsafe { c_str(name, "name")? };
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        let e = find(h, name)?;
        unsafe { out.write(e.info) };
        Ok(())
    })
}

/// Reads a string metadata value into `buffer` (NUL-terminated). `*out_len` receives the length without the NUL; if
/// `capacity <= *out_len` the call fails with `QUINTRIT_ERR_BUFFER_TOO_SMALL` and nothing is written, so call it first
/// with `buffer = NULL, capacity = 0` to learn the size.
///
/// # Safety
/// `buffer` must point to `capacity` writable bytes; `key` is NUL-terminated; `out_len` a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_metadata_string(
    gguf: *const GgufHandle,
    key: *const c_char,
    buffer: *mut c_char,
    capacity: usize,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / out_slice.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let key = unsafe { c_str(key, "key")? };
        let value = match h.file.metadata.get(key) {
            Some(v) => v,
            None => fail!(QUINTRIT_ERR_NOT_FOUND, "metadata key '{key}' not found"),
        };
        let Some(s) = value.as_str() else {
            fail!(QUINTRIT_ERR_FORMAT, "metadata key '{key}' is not a string");
        };
        unsafe { write_out(out_len, s.len(), "out_len")? };
        let buf = unsafe { out_slice(buffer as *mut u8, capacity, "buffer")? };
        if buf.len() <= s.len() {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "metadata value needs {} bytes including the NUL, buffer holds {capacity}", s.len() + 1);
        }
        buf[..s.len()].copy_from_slice(s.as_bytes());
        buf[s.len()] = 0;
        Ok(())
    })
}

/// Reads an unsigned integer metadata value (also accepts non-negative signed integers).
///
/// # Safety
/// `key` is a NUL-terminated string; `out_value` points to a `uint64_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_metadata_u64(gguf: *const GgufHandle, key: *const c_char, out_value: *mut u64) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / write_out.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let key = unsafe { c_str(key, "key")? };
        let value = match h.file.metadata.get(key) {
            Some(v) => v,
            None => fail!(QUINTRIT_ERR_NOT_FOUND, "metadata key '{key}' not found"),
        };
        let Some(v) = value.as_u64() else {
            fail!(QUINTRIT_ERR_FORMAT, "metadata key '{key}' is not an unsigned integer");
        };
        unsafe { write_out(out_value, v, "out_value") }
    })
}

/// Returns a pointer to the raw payload of a tensor inside the memory-mapped file. The pointer stays valid until
/// `quintrit_gguf_close`. Only readable tensor types (F32, F16, I2_S, TQ1_0, TQ2_0) have a payload.
///
/// # Safety
/// `out_data` and `out_len` must point to a `const uint8_t*` and a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_tensor_data(
    gguf: *const GgufHandle,
    name: *const c_char,
    out_data: *mut *const u8,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / write_out.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let name = unsafe { c_str(name, "name")? };
        let e = find(h, name)?;
        let Some(data) = h.file.get_tensor_data(name) else {
            fail!(QUINTRIT_ERR_UNSUPPORTED, "tensor '{name}' has type {} whose payload cannot be read", e.info.ggml_type);
        };
        unsafe {
            write_out(out_data, data.as_ptr(), "out_data")?;
            write_out(out_len, data.len(), "out_len")
        }
    })
}

/// Computes the tiled layout (size, tile height, padding) and the weight scale of a ternary tensor for a kernel,
/// without converting it. `out_layout` and `out_scale` may be NULL if not needed.
///
/// # Safety
/// `out_layout`, if not NULL, must point to a `quintrit_tiled_layout_t` with `struct_size` set; `out_scale` to a float.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_tensor_layout(
    gguf: *const GgufHandle,
    name: *const c_char,
    kernel: i32,
    out_layout: *mut TiledLayout,
    out_scale: *mut f32,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / check_out_struct.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let name = unsafe { c_str(name, "name")? };
        let kernel = resolve(kernel)?;
        let e = find(h, name)?;
        let ggml_type = e.info.ggml_type as u32;
        if !quintrit_format::is_supported_ternary_type(ggml_type) {
            fail!(QUINTRIT_ERR_UNSUPPORTED, "tensor '{name}' is not a ternary tensor");
        }
        let (rows, cols) = matrix_shape(e)?;
        let l = layout(kernel, rows, cols)?;
        let scale = tensor_scale(h, name, ggml_type)?;
        if !out_layout.is_null() {
            let out = unsafe { check_out_struct(out_layout, "out_layout")? };
            unsafe { out.write(l) };
        }
        if !out_scale.is_null() {
            unsafe { write_out(out_scale, scale, "out_scale")? };
        }
        Ok(())
    })
}

/// Converts a ternary tensor (I2_S, TQ1_0 or TQ2_0) into the tiled quintrit layout of `kernel`, writing it into the
/// caller's buffer. Use `quintrit_gguf_tensor_layout` to size the buffer. The result is the same layout that
/// `quintrit_pack_matrix` produces and that `quintrit_gemv_*` reads; `out_scale` receives the tensor's weight scale.
///
/// # Safety
/// `out` must point to `out_capacity` writable bytes; `out_layout` / `out_scale` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_gguf_convert_tensor(
    gguf: *const GgufHandle,
    name: *const c_char,
    kernel: i32,
    out: *mut u8,
    out_capacity: usize,
    out_layout: *mut TiledLayout,
    out_scale: *mut f32,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / c_str / out_slice.
        let h = unsafe { check_handle(gguf, "gguf")? };
        let name = unsafe { c_str(name, "name")? };
        let kernel = resolve(kernel)?;
        let e = find(h, name)?;
        let out_layout_ptr = if out_layout.is_null() { None } else { Some(unsafe { check_out_struct(out_layout, "out_layout")? }) };
        let (rows, cols) = matrix_shape(e)?;
        let l = layout(kernel, rows, cols).map_err(|f| {
            crate::error::Failure::new(f.code, format!("tensor '{name}': {}", f.msg))
        })?;
        let out = unsafe { out_slice(out, out_capacity, "out")? };
        if (out.len() as u64) < l.total_bytes {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "tensor '{name}' needs {} bytes, buffer holds {}", l.total_bytes, out.len());
        }
        let (trits, rows, cols, scale) = load_trits(h, e)?;
        pack_tiled(kernel, &trits, rows, cols, &mut out[..l.total_bytes as usize]);
        if let Some(p) = out_layout_ptr {
            unsafe { p.write(l) };
        }
        if !out_scale.is_null() {
            unsafe { write_out(out_scale, scale, "out_scale")? };
        }
        Ok(())
    })
}
