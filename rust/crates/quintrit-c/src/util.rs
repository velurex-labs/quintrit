//! Validation of raw pointers, lengths and strings received from C.
//!
//! No entry point dereferences a caller pointer without going through one of these helpers.

use crate::error::{fail, Res, QUINTRIT_ERR_INVALID_ARGUMENT, QUINTRIT_ERR_INVALID_HANDLE, QUINTRIT_ERR_NULL_POINTER};
use std::ffi::{c_char, CStr};
use std::mem::{align_of, size_of};

/// Checks the alignment of a non-NULL pointer.
fn check_aligned<T>(ptr: *const T, what: &str) -> Res<()> {
    if !(ptr as usize).is_multiple_of(align_of::<T>()) {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "{what} is not aligned to {} bytes", align_of::<T>());
    }
    Ok(())
}

/// Checks that `len` elements of `T` fit into the address space limits of a Rust slice.
fn check_len<T>(len: usize, what: &str) -> Res<()> {
    if len.checked_mul(size_of::<T>()).is_none_or(|bytes| bytes > isize::MAX as usize) {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "{what}: length {len} is too large");
    }
    Ok(())
}

/// Borrows an input buffer of `len` elements. A NULL pointer is accepted only for `len == 0`.
///
/// # Safety
/// `ptr` must point to `len` readable elements for as long as the returned slice is used.
pub unsafe fn in_slice<'a, T>(ptr: *const T, len: usize, what: &str) -> Res<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        fail!(QUINTRIT_ERR_NULL_POINTER, "{what} is NULL");
    }
    check_aligned(ptr, what)?;
    check_len::<T>(len, what)?;
    // SAFETY: non-NULL, aligned, length-checked; readability is the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Borrows an output buffer of `len` elements. A NULL pointer is accepted only for `len == 0`.
///
/// # Safety
/// `ptr` must point to `len` writable elements that no other reference aliases while the slice is used.
pub unsafe fn out_slice<'a, T>(ptr: *mut T, len: usize, what: &str) -> Res<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        fail!(QUINTRIT_ERR_NULL_POINTER, "{what} is NULL");
    }
    check_aligned(ptr, what)?;
    check_len::<T>(len, what)?;
    // SAFETY: non-NULL, aligned, length-checked; writability is the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

/// Validates a pointer to a single output value.
pub fn out_ptr<T>(ptr: *mut T, what: &str) -> Res<*mut T> {
    if ptr.is_null() {
        fail!(QUINTRIT_ERR_NULL_POINTER, "{what} is NULL");
    }
    check_aligned(ptr, what)?;
    Ok(ptr)
}

/// Writes a single output value through a validated pointer.
///
/// # Safety
/// `ptr` must be valid for a write of `T`.
pub unsafe fn write_out<T>(ptr: *mut T, value: T, what: &str) -> Res<()> {
    let ptr = out_ptr(ptr, what)?;
    // SAFETY: non-NULL and aligned; validity is the caller's contract.
    unsafe { ptr.write(value) };
    Ok(())
}

/// Borrows a NUL-terminated UTF-8 string.
///
/// # Safety
/// `ptr` must be NULL or point to a NUL-terminated string.
pub unsafe fn c_str<'a>(ptr: *const c_char, what: &str) -> Res<&'a str> {
    if ptr.is_null() {
        fail!(QUINTRIT_ERR_NULL_POINTER, "{what} is NULL");
    }
    // SAFETY: non-NULL; NUL termination is the caller's contract.
    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes();
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(e) => fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "{what} is not valid UTF-8 ({e})"),
    }
}

/// Borrows a UTF-8 text of explicit length (not necessarily NUL-terminated).
///
/// # Safety
/// `ptr` must point to `len` readable bytes (or be NULL with `len == 0`).
pub unsafe fn text<'a>(ptr: *const c_char, len: usize, what: &str) -> Res<&'a str> {
    // SAFETY: forwarded contract.
    let bytes = unsafe { in_slice(ptr as *const u8, len, what)? };
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(e) => fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "{what} is not valid UTF-8 ({e})"),
    }
}

/// Fails if the byte ranges of two caller buffers overlap. Call it before the slices are created.
pub fn check_disjoint<A, B>(a: *const A, a_len: usize, b: *const B, b_len: usize, what: &str) -> Res<()> {
    let a0 = a as usize;
    let a1 = a0.saturating_add(a_len.saturating_mul(size_of::<A>()));
    let b0 = b as usize;
    let b1 = b0.saturating_add(b_len.saturating_mul(size_of::<B>()));
    if a_len > 0 && b_len > 0 && a0 < b1 && b0 < a1 {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "{what}: the buffers overlap");
    }
    Ok(())
}

/// Reads and validates a versioned input struct: the caller's `struct_size` must cover at least the fields known to
/// this library version; fields beyond that are ignored. Returns `None` for a NULL pointer.
///
/// # Safety
/// `ptr` must be NULL or point to a struct whose first four bytes are the caller's `struct_size` and that is readable
/// for that many bytes.
pub unsafe fn read_versioned<T: Copy>(ptr: *const T, what: &str) -> Res<Option<T>> {
    if ptr.is_null() {
        return Ok(None);
    }
    check_aligned(ptr, what)?;
    // SAFETY: the first member of every versioned struct is a u32 `struct_size`.
    let declared = unsafe { (ptr as *const u32).read() } as usize;
    if declared < size_of::<T>() {
        fail!(
            QUINTRIT_ERR_INVALID_ARGUMENT,
            "{what}: struct_size is {declared}, expected at least {} (set struct_size = sizeof(struct) and use the matching header)",
            size_of::<T>()
        );
    }
    // SAFETY: `declared >= size_of::<T>()` bytes are readable by the caller's contract.
    Ok(Some(unsafe { ptr.read() }))
}

/// Validates an output struct: `struct_size` must be at least what this library writes.
///
/// # Safety
/// `ptr` must be NULL or point to a writable struct whose first four bytes are the caller's `struct_size`.
pub unsafe fn check_out_struct<T>(ptr: *mut T, what: &str) -> Res<*mut T> {
    let ptr = out_ptr(ptr, what)?;
    // SAFETY: the first member of every versioned struct is a u32 `struct_size`.
    let declared = unsafe { (ptr as *const u32).read() } as usize;
    if declared < size_of::<T>() {
        fail!(
            QUINTRIT_ERR_INVALID_ARGUMENT,
            "{what}: struct_size is {declared}, expected at least {} (set struct_size = sizeof(struct) before the call)",
            size_of::<T>()
        );
    }
    Ok(ptr)
}

/// Magic numbers that tag the handle types; zeroed when a handle is freed.
pub const MAGIC_GGUF: u32 = 0x5154_4747; // "QTGG"
pub const MAGIC_MODEL: u32 = 0x5154_4d44; // "QTMD"
pub const MAGIC_SESSION: u32 = 0x5154_5353; // "QTSS"

/// Checks the type tag of a handle (best effort: it detects NULL and handles of another type, and most use after
/// free, but cannot make a dangling pointer safe).
///
/// # Safety
/// `ptr` must be NULL or point to memory that is readable for a `u32` (every handle starts with its magic number).
pub unsafe fn check_handle<'a, T: HasMagic>(ptr: *const T, what: &str) -> Res<&'a T> {
    if ptr.is_null() {
        fail!(QUINTRIT_ERR_INVALID_HANDLE, "{what} handle is NULL");
    }
    check_aligned(ptr, what)?;
    // SAFETY: handles start with a u32 magic; readability is the caller's contract.
    let r = unsafe { &*ptr };
    if r.magic() != T::MAGIC {
        fail!(QUINTRIT_ERR_INVALID_HANDLE, "{what} handle has the wrong type or was already freed");
    }
    Ok(r)
}

/// Implemented by the handle structs.
pub trait HasMagic {
    const MAGIC: u32;
    fn magic(&self) -> u32;
}
