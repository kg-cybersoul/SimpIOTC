//! # C Foreign Function Interface
//!
//! Provides `extern "C"` functions for calling the IoT compressor from C/C++.
//! Gated behind the `ffi` Cargo feature.
//!
//! ## Error Handling
//!
//! Functions return negative error codes on failure. Call `iotc_last_error` to
//! retrieve a human-readable error message (thread-local, valid until the next
//! FFI call on the same thread).
//!
//! ## Thread Safety
//!
//! All stateless functions (`iotc_compress`, `iotc_decompress`, `iotc_free`) are
//! thread-safe. `IotcSeekableReader` handles are NOT thread-safe — each handle
//! must be used from a single thread at a time.

use std::cell::RefCell;
use std::ffi::CString;
use std::panic::catch_unwind;
use std::slice;

use crate::parallel::{compress, decompress};
use crate::seekable::SeekableReader;
use crate::{CompressionConfig, DataType};

// ═══════════════════════════════════════════════════════════════════════════════
// Error Codes
// ═══════════════════════════════════════════════════════════════════════════════

/// Success.
pub const IOTC_OK: i64 = 0;
/// A null pointer was passed where a valid pointer was required.
pub const IOTC_ERR_NULL_PTR: i64 = -1;
/// The provided output buffer is too small.
pub const IOTC_ERR_BUFFER_TOO_SMALL: i64 = -2;
/// The input data is invalid or corrupted.
pub const IOTC_ERR_INVALID_DATA: i64 = -3;
/// An internal error occurred (panic caught at FFI boundary).
pub const IOTC_ERR_INTERNAL: i64 = -4;
/// Invalid argument (e.g., unknown data type, invalid stride).
pub const IOTC_ERR_INVALID_ARG: i64 = -5;

// ═══════════════════════════════════════════════════════════════════════════════
// Thread-Local Error Message
// ═══════════════════════════════════════════════════════════════════════════════

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|cell| {
        // Error messages should never contain interior nulls; truncate if they do
        let c = CString::new(msg).unwrap_or_else(|e| {
            let pos = e.nul_position();
            CString::new(&e.into_vec()[..pos]).unwrap()
        });
        *cell.borrow_mut() = c;
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::default();
    });
}

/// Retrieve the last error message. Returns a pointer to a null-terminated
/// UTF-8 string. The pointer is valid until the next FFI call on the same thread.
///
/// Returns a pointer to an empty string if no error has occurred.
#[no_mangle]
pub extern "C" fn iotc_last_error() -> *const u8 {
    LAST_ERROR.with(|cell| {
        let s = cell.borrow();
        // The CString lives in the RefCell — its pointer is stable until
        // the next set_last_error call on this thread.
        s.as_ptr() as *const u8
    })
}

/// Returns the byte length of the last error message (excluding null terminator).
#[no_mangle]
pub extern "C" fn iotc_last_error_len() -> usize {
    LAST_ERROR.with(|cell| cell.borrow().to_bytes().len())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compression / Decompression
// ═══════════════════════════════════════════════════════════════════════════════

/// Compress `input_len` bytes from `input_ptr` into `output_ptr`.
///
/// # Parameters
/// - `input_ptr`: Pointer to input data.
/// - `input_len`: Length of input data in bytes.
/// - `output_ptr`: Pointer to output buffer.
/// - `output_capacity`: Size of output buffer in bytes.
/// - `stride`: Struct stride for columnar transposition (0 = disabled).
/// - `data_type`: Data type hint (0=auto, 1=raw, 2=i32, 3=i64, 4=f32, 5=f64).
///
/// # Returns
/// On success: the number of compressed bytes written (positive).
/// On failure: a negative error code. Call `iotc_last_error()` for details.
///
/// # Safety
///
/// - `input_ptr` must be valid for reads of `input_len` bytes.
/// - `output_ptr` must be valid for writes of `output_capacity` bytes.
/// - Both pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_compress(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
    output_capacity: usize,
    stride: u16,
    data_type: u8,
) -> i64 {
    clear_last_error();

    if input_ptr.is_null() || output_ptr.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }
    if input_len == 0 {
        set_last_error("empty input".into());
        return IOTC_ERR_INVALID_DATA;
    }

    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };

        let dt = match data_type_from_u8(data_type) {
            Some(dt) => dt,
            None => {
                set_last_error(format!("unknown data_type: {}", data_type));
                return IOTC_ERR_INVALID_ARG;
            }
        };

        let config = CompressionConfig {
            data_type: dt,
            stride: if stride > 0 { Some(stride) } else { None },
            ..CompressionConfig::balanced()
        };

        match compress(input, &config) {
            Ok(compressed) => {
                if compressed.len() > output_capacity {
                    set_last_error(format!(
                        "output buffer too small: need {} bytes, have {}",
                        compressed.len(),
                        output_capacity
                    ));
                    return IOTC_ERR_BUFFER_TOO_SMALL;
                }
                let output = unsafe { slice::from_raw_parts_mut(output_ptr, output_capacity) };
                output[..compressed.len()].copy_from_slice(&compressed);
                compressed.len() as i64
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during compression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

/// Compress with allocation — the library allocates the output buffer.
///
/// # Parameters
/// - `input_ptr`: Pointer to input data.
/// - `input_len`: Length of input data in bytes.
/// - `output_ptr_out`: Pointer to a `*mut u8` that will receive the allocated buffer.
/// - `output_len_out`: Pointer to a `usize` that will receive the compressed length.
/// - `stride`: Struct stride (0 = disabled).
/// - `data_type`: Data type hint (0=auto, 1=raw, 2=i32, 3=i64, 4=f32, 5=f64).
///
/// # Returns
/// `IOTC_OK` on success. The caller MUST free the buffer via `iotc_free`.
///
/// # Safety
///
/// - `input_ptr` must be valid for reads of `input_len` bytes.
/// - `output_ptr_out` must be a valid, writable pointer to a `*mut u8`.
/// - `output_len_out` must be a valid, writable pointer to a `usize`.
/// - All pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_compress_alloc(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr_out: *mut *mut u8,
    output_len_out: *mut usize,
    stride: u16,
    data_type: u8,
) -> i64 {
    clear_last_error();

    if input_ptr.is_null() || output_ptr_out.is_null() || output_len_out.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }
    if input_len == 0 {
        set_last_error("empty input".into());
        return IOTC_ERR_INVALID_DATA;
    }

    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };

        let dt = match data_type_from_u8(data_type) {
            Some(dt) => dt,
            None => {
                set_last_error(format!("unknown data_type: {}", data_type));
                return IOTC_ERR_INVALID_ARG;
            }
        };

        let config = CompressionConfig {
            data_type: dt,
            stride: if stride > 0 { Some(stride) } else { None },
            ..CompressionConfig::balanced()
        };

        match compress(input, &config) {
            Ok(compressed) => {
                let len = compressed.len();
                let mut boxed = compressed.into_boxed_slice();
                let ptr = boxed.as_mut_ptr();
                std::mem::forget(boxed);
                unsafe {
                    *output_ptr_out = ptr;
                    *output_len_out = len;
                }
                IOTC_OK
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during compression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

/// Decompress `input_len` bytes from `input_ptr` into `output_ptr`.
///
/// # Returns
/// On success: the number of decompressed bytes written (positive).
/// On failure: a negative error code.
///
/// # Safety
///
/// - `input_ptr` must be valid for reads of `input_len` bytes.
/// - `output_ptr` must be valid for writes of `output_capacity` bytes.
/// - Both pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_decompress(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
    output_capacity: usize,
) -> i64 {
    clear_last_error();

    if input_ptr.is_null() || output_ptr.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }

    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };

        match decompress(input) {
            Ok(decompressed) => {
                if decompressed.len() > output_capacity {
                    set_last_error(format!(
                        "output buffer too small: need {} bytes, have {}",
                        decompressed.len(),
                        output_capacity
                    ));
                    return IOTC_ERR_BUFFER_TOO_SMALL;
                }
                let output = unsafe { slice::from_raw_parts_mut(output_ptr, output_capacity) };
                output[..decompressed.len()].copy_from_slice(&decompressed);
                decompressed.len() as i64
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during decompression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

/// Decompress with allocation — the library allocates the output buffer.
///
/// # Returns
/// `IOTC_OK` on success. The caller MUST free the buffer via `iotc_free`.
///
/// # Safety
///
/// - `input_ptr` must be valid for reads of `input_len` bytes.
/// - `output_ptr_out` must be a valid, writable pointer to a `*mut u8`.
/// - `output_len_out` must be a valid, writable pointer to a `usize`.
/// - All pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_decompress_alloc(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr_out: *mut *mut u8,
    output_len_out: *mut usize,
) -> i64 {
    clear_last_error();

    if input_ptr.is_null() || output_ptr_out.is_null() || output_len_out.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }

    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };

        match decompress(input) {
            Ok(decompressed) => {
                let len = decompressed.len();
                let mut boxed = decompressed.into_boxed_slice();
                let ptr = boxed.as_mut_ptr();
                std::mem::forget(boxed);
                unsafe {
                    *output_ptr_out = ptr;
                    *output_len_out = len;
                }
                IOTC_OK
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    });

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during decompression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

/// Free a buffer previously allocated by `iotc_compress_alloc` or
/// `iotc_decompress_alloc`.
///
/// Passing a null pointer is a safe no-op.
///
/// # Safety
///
/// - `ptr` must have been returned by `iotc_compress_alloc` or
///   `iotc_decompress_alloc`, and `len` must be the exact length that was
///   written to the corresponding `output_len_out`.
/// - Each `(ptr, len)` pair must be freed at most once.
/// - Passing a null `ptr` is permitted and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn iotc_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // Reconstruct the Box<[u8]> and drop it.
    unsafe {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Seekable Reader
// ═══════════════════════════════════════════════════════════════════════════════

/// Opaque handle to a seekable reader.
///
/// Owns a copy of the compressed data so the caller can free their buffer
/// after calling `iotc_seekable_open`. This avoids dangling-pointer bugs
/// at the cost of one memcpy on open.
pub struct IotcSeekableReader {
    /// Owned copy of the compressed frame data. The SeekableReader borrows from this.
    _data: Vec<u8>,
    /// The actual reader. Borrows from `_data` via a lifetime-erased reference.
    /// SAFETY: `_data` is heap-allocated, never reallocated after construction,
    /// and is dropped AFTER `reader` because struct fields drop in declaration order.
    reader: SeekableReader<'static>,
}

/// Open a seekable reader over compressed data.
///
/// The input data is copied internally — the caller may free `input_ptr` after this call.
///
/// # Returns
/// On success: an opaque pointer to `IotcSeekableReader`.
/// On failure: null. Call `iotc_last_error()` for details.
///
/// # Safety
///
/// - `input_ptr` must be valid for reads of `input_len` bytes.
/// - `input_ptr` must be non-null (null is detected and returns null).
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_open(
    input_ptr: *const u8,
    input_len: usize,
) -> *mut IotcSeekableReader {
    clear_last_error();

    if input_ptr.is_null() {
        set_last_error("null pointer argument".into());
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let input = unsafe { slice::from_raw_parts(input_ptr, input_len) };
        let data = input.to_vec();

        // SAFETY: We create a SeekableReader that borrows from `data`. The data
        // lives in the same struct and is never moved or mutated after this point.
        // The lifetime is erased via transmute because the struct is self-referential.
        // This is sound because:
        // 1. `data` is heap-allocated (Vec) — its buffer address is stable.
        // 2. We never push/resize `data` after construction.
        // 3. Struct fields drop in order: `reader` drops before `_data`.
        let data_slice: &[u8] = &data;
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(data_slice) };

        match SeekableReader::new(static_slice) {
            Ok(reader) => {
                let handle = Box::new(IotcSeekableReader {
                    _data: data,
                    reader,
                });
                Box::into_raw(handle)
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                std::ptr::null_mut()
            }
        }
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("internal panic opening seekable reader".into());
            std::ptr::null_mut()
        }
    }
}

/// Close and free a seekable reader handle.
///
/// Passing a null pointer is a safe no-op.
///
/// # Safety
///
/// - `handle` must have been returned by [`iotc_seekable_open`], or be null.
/// - Each handle must be closed at most once.
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_close(handle: *mut IotcSeekableReader) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(handle);
    }
}

/// Get the number of blocks in the frame.
///
/// # Safety
///
/// - `handle` must be a valid pointer returned by [`iotc_seekable_open`] that
///   has not been closed, or null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_block_count(handle: *const IotcSeekableReader) -> i64 {
    if handle.is_null() {
        set_last_error("null handle".into());
        return IOTC_ERR_NULL_PTR;
    }
    let reader = unsafe { &*handle };
    reader.reader.block_count() as i64
}

/// Get the original (uncompressed) size of the frame.
///
/// # Safety
///
/// - `handle` must be a valid pointer returned by [`iotc_seekable_open`] that
///   has not been closed, or null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_original_size(handle: *const IotcSeekableReader) -> i64 {
    if handle.is_null() {
        set_last_error("null handle".into());
        return IOTC_ERR_NULL_PTR;
    }
    let reader = unsafe { &*handle };
    reader.reader.original_size() as i64
}

/// Decompress a single block by index, writing to the provided output buffer.
///
/// # Returns
/// On success: the number of decompressed bytes written.
/// On failure: a negative error code.
///
/// # Safety
///
/// - `handle` must be a valid pointer returned by [`iotc_seekable_open`] that
///   has not been closed.
/// - `output_ptr` must be valid for writes of `output_capacity` bytes.
/// - Both pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_decompress_block(
    handle: *mut IotcSeekableReader,
    block_index: u32,
    output_ptr: *mut u8,
    output_capacity: usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() || output_ptr.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reader = unsafe { &mut *handle };

        match reader.reader.decompress_block(block_index as usize) {
            Ok(block) => {
                if block.len() > output_capacity {
                    set_last_error(format!(
                        "output buffer too small: need {} bytes, have {}",
                        block.len(),
                        output_capacity
                    ));
                    return IOTC_ERR_BUFFER_TOO_SMALL;
                }
                let output = unsafe { slice::from_raw_parts_mut(output_ptr, output_capacity) };
                output[..block.len()].copy_from_slice(&block);
                block.len() as i64
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during block decompression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

/// Decompress a byte range `[byte_start, byte_end)` with allocation.
///
/// # Returns
/// `IOTC_OK` on success. The caller MUST free the buffer via `iotc_free`.
///
/// # Safety
///
/// - `handle` must be a valid pointer returned by [`iotc_seekable_open`] that
///   has not been closed.
/// - `output_ptr_out` must be a valid, writable pointer to a `*mut u8`.
/// - `output_len_out` must be a valid, writable pointer to a `usize`.
/// - All pointers must be non-null (null is detected and returns an error code).
#[no_mangle]
pub unsafe extern "C" fn iotc_seekable_decompress_range(
    handle: *mut IotcSeekableReader,
    byte_start: u64,
    byte_end: u64,
    output_ptr_out: *mut *mut u8,
    output_len_out: *mut usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() || output_ptr_out.is_null() || output_len_out.is_null() {
        set_last_error("null pointer argument".into());
        return IOTC_ERR_NULL_PTR;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let reader = unsafe { &mut *handle };

        match reader.reader.decompress_byte_range(byte_start, byte_end) {
            Ok(data) => {
                let len = data.len();
                let mut boxed = data.into_boxed_slice();
                let ptr = boxed.as_mut_ptr();
                std::mem::forget(boxed);
                unsafe {
                    *output_ptr_out = ptr;
                    *output_len_out = len;
                }
                IOTC_OK
            }
            Err(e) => {
                set_last_error(format!("{}", e));
                IOTC_ERR_INVALID_DATA
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("internal panic during range decompression".into());
            IOTC_ERR_INTERNAL
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a C data type enum to the Rust `DataType`. Returns `None` for
/// the auto-detect sentinel (0).
fn data_type_from_u8(v: u8) -> Option<Option<DataType>> {
    match v {
        0 => Some(None), // auto-detect
        1 => Some(Some(DataType::Raw)),
        2 => Some(Some(DataType::IntegerI32)),
        3 => Some(Some(DataType::IntegerI64)),
        4 => Some(Some(DataType::Float32)),
        5 => Some(Some(DataType::Float64)),
        6 => Some(Some(DataType::IntegerU32)),
        7 => Some(Some(DataType::IntegerU64)),
        8 => Some(Some(DataType::Float32Shuffle)),
        9 => Some(Some(DataType::Float64Shuffle)),
        10 => Some(Some(DataType::Float32ShuffleDelta)),
        11 => Some(Some(DataType::Float64ShuffleDelta)),
        _ => None, // unknown
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_compress_decompress_roundtrip() {
        unsafe {
            let input = b"Hello FFI world! This is a test. Hello FFI world! This is a test.";
            let mut output = vec![0u8; input.len() + 512];
            let compressed_len = iotc_compress(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                0, // no stride
                1, // Raw
            );
            assert!(compressed_len > 0, "compress failed: {}", compressed_len);

            let mut decompressed = vec![0u8; input.len() + 512];
            let decompressed_len = iotc_decompress(
                output.as_ptr(),
                compressed_len as usize,
                decompressed.as_mut_ptr(),
                decompressed.len(),
            );
            assert!(
                decompressed_len > 0,
                "decompress failed: {}",
                decompressed_len
            );
            assert_eq!(&decompressed[..decompressed_len as usize], &input[..]);
        }
    }

    #[test]
    fn ffi_compress_alloc_roundtrip() {
        unsafe {
            let input = b"Alloc roundtrip test data! Alloc roundtrip test data!";
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: usize = 0;

            let rc = iotc_compress_alloc(
                input.as_ptr(),
                input.len(),
                &mut out_ptr,
                &mut out_len,
                0,
                1,
            );
            assert_eq!(rc, IOTC_OK);
            assert!(!out_ptr.is_null());
            assert!(out_len > 0);

            let mut dec_ptr: *mut u8 = std::ptr::null_mut();
            let mut dec_len: usize = 0;
            let rc = iotc_decompress_alloc(out_ptr, out_len, &mut dec_ptr, &mut dec_len);
            assert_eq!(rc, IOTC_OK);
            let decompressed = slice::from_raw_parts(dec_ptr, dec_len);
            assert_eq!(decompressed, &input[..]);

            iotc_free(out_ptr, out_len);
            iotc_free(dec_ptr, dec_len);
        }
    }

    #[test]
    fn ffi_null_pointer_handling() {
        unsafe {
            let rc = iotc_compress(std::ptr::null(), 10, std::ptr::null_mut(), 10, 0, 1);
            assert_eq!(rc, IOTC_ERR_NULL_PTR);
            assert!(iotc_last_error_len() > 0);
        }
    }

    #[test]
    fn ffi_buffer_too_small() {
        unsafe {
            let input = b"Buffer size test data. Buffer size test data. Repeat for compression.";
            let mut tiny_output = [0u8; 4]; // way too small
            let rc = iotc_compress(
                input.as_ptr(),
                input.len(),
                tiny_output.as_mut_ptr(),
                tiny_output.len(),
                0,
                1,
            );
            assert_eq!(rc, IOTC_ERR_BUFFER_TOO_SMALL);
        }
    }

    #[test]
    fn ffi_invalid_data_type() {
        unsafe {
            let input = b"test";
            let mut output = [0u8; 256];
            let rc = iotc_compress(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                0,
                99, // invalid
            );
            assert_eq!(rc, IOTC_ERR_INVALID_ARG);
        }
    }

    #[test]
    fn ffi_free_null_is_safe() {
        unsafe {
            iotc_free(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn ffi_seekable_lifecycle() {
        // Compress with seek table
        let input = b"Seekable FFI test! Seekable FFI test! Seekable FFI test! More data here.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 32,
            store_checksum: false,
            store_seek_table: true,
            ..CompressionConfig::fast()
        };
        let compressed = compress(input, &config).unwrap();

        unsafe {
            // Open seekable reader via FFI
            let handle = iotc_seekable_open(compressed.as_ptr(), compressed.len());
            assert!(!handle.is_null(), "seekable_open failed");

            let block_count = iotc_seekable_block_count(handle);
            assert!(block_count > 0);

            let original_size = iotc_seekable_original_size(handle);
            assert_eq!(original_size, input.len() as i64);

            // Decompress block 0
            let mut block_buf = vec![0u8; 1024];
            let block_len =
                iotc_seekable_decompress_block(handle, 0, block_buf.as_mut_ptr(), block_buf.len());
            assert!(block_len > 0, "decompress_block failed: {}", block_len);
            assert_eq!(
                &block_buf[..block_len as usize],
                &input[..block_len as usize]
            );

            // Decompress byte range via alloc
            let mut range_ptr: *mut u8 = std::ptr::null_mut();
            let mut range_len: usize = 0;
            let rc = iotc_seekable_decompress_range(
                handle,
                0,
                input.len() as u64,
                &mut range_ptr,
                &mut range_len,
            );
            assert_eq!(rc, IOTC_OK);
            let range_data = slice::from_raw_parts(range_ptr, range_len);
            assert_eq!(range_data, &input[..]);
            iotc_free(range_ptr, range_len);

            // Close
            iotc_seekable_close(handle);
        }
    }

    #[test]
    fn ffi_seekable_null_handle() {
        unsafe {
            assert_eq!(
                iotc_seekable_block_count(std::ptr::null()),
                IOTC_ERR_NULL_PTR
            );
            assert_eq!(
                iotc_seekable_original_size(std::ptr::null()),
                IOTC_ERR_NULL_PTR
            );
            iotc_seekable_close(std::ptr::null_mut()); // should not crash
        }
    }

    #[test]
    fn ffi_auto_detect_data_type() {
        unsafe {
            // data_type=0 means auto-detect
            let input = b"Auto detect test data! Auto detect test data!";
            let mut output = vec![0u8; input.len() + 512];
            let compressed_len = iotc_compress(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                0,
                0, // auto-detect
            );
            assert!(compressed_len > 0);

            let mut decompressed = vec![0u8; input.len() + 512];
            let decompressed_len = iotc_decompress(
                output.as_ptr(),
                compressed_len as usize,
                decompressed.as_mut_ptr(),
                decompressed.len(),
            );
            assert_eq!(&decompressed[..decompressed_len as usize], &input[..]);
        }
    }

    #[test]
    fn ffi_stride_roundtrip() {
        // 12-byte struct × 100 records
        let mut input = Vec::with_capacity(1200);
        for i in 0u32..100 {
            input.extend_from_slice(&i.to_le_bytes());
            input.extend_from_slice(&(20.0f32 + 0.01 * i as f32).to_le_bytes());
            input.extend_from_slice(&(50.0f32 + 0.005 * i as f32).to_le_bytes());
        }

        unsafe {
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: usize = 0;
            let rc = iotc_compress_alloc(
                input.as_ptr(),
                input.len(),
                &mut out_ptr,
                &mut out_len,
                12, // stride = 12 bytes per struct
                1,  // Raw
            );
            assert_eq!(rc, IOTC_OK);

            let mut dec_ptr: *mut u8 = std::ptr::null_mut();
            let mut dec_len: usize = 0;
            let rc = iotc_decompress_alloc(out_ptr, out_len, &mut dec_ptr, &mut dec_len);
            assert_eq!(rc, IOTC_OK);

            let decompressed = slice::from_raw_parts(dec_ptr, dec_len);
            assert_eq!(decompressed, &input[..]);

            iotc_free(out_ptr, out_len);
            iotc_free(dec_ptr, dec_len);
        }
    }
}
