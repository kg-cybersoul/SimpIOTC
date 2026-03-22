/**
 * @file iotc.h
 * @brief IoT Time-Series Compressor — C API
 *
 * High-speed, high-ratio LZ77/ANS compression for sensor and IoT time-series
 * data. This header exposes the Rust library via a stable C ABI.
 *
 * Build the static library:
 *   cargo build --release --features ffi
 *
 * Link against: -liot_compressor -ldl -lpthread -lm (Linux)
 *               iot_compressor.lib ws2_32.lib userenv.lib bcrypt.lib (Windows)
 */

#ifndef IOTC_H
#define IOTC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Error Codes
 * ═══════════════════════════════════════════════════════════════════════════ */

#define IOTC_OK               0   /**< Success. */
#define IOTC_ERR_NULL_PTR    -1   /**< A null pointer was passed. */
#define IOTC_ERR_BUFFER_SMALL -2  /**< Output buffer is too small. */
#define IOTC_ERR_INVALID_DATA -3  /**< Input data is corrupted or invalid. */
#define IOTC_ERR_INTERNAL    -4   /**< Internal error (Rust panic caught). */
#define IOTC_ERR_INVALID_ARG -5   /**< Invalid argument value. */

/* ═══════════════════════════════════════════════════════════════════════════
 * Data Type Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define IOTC_TYPE_AUTO   0  /**< Auto-detect data type. */
#define IOTC_TYPE_RAW    1  /**< Raw bytes, no preprocessing. */
#define IOTC_TYPE_I32    2  /**< 32-bit signed integers (delta-of-delta). */
#define IOTC_TYPE_I64    3  /**< 64-bit signed integers (delta-of-delta). */
#define IOTC_TYPE_F32    4  /**< 32-bit floats (Gorilla XOR). */
#define IOTC_TYPE_F64    5  /**< 64-bit floats (Gorilla XOR). */
#define IOTC_TYPE_U32    6  /**< 32-bit unsigned integers (delta-of-delta). */
#define IOTC_TYPE_U64    7  /**< 64-bit unsigned integers (delta-of-delta). */
#define IOTC_TYPE_F32S   8  /**< 32-bit floats (byte shuffle). */
#define IOTC_TYPE_F64S   9  /**< 64-bit floats (byte shuffle). */
#define IOTC_TYPE_F32SD  10 /**< 32-bit floats (byte shuffle + byte-delta). */
#define IOTC_TYPE_F64SD  11 /**< 64-bit floats (byte shuffle + byte-delta). */

/* ═══════════════════════════════════════════════════════════════════════════
 * Error Reporting
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Retrieve the last error message (UTF-8, not null-terminated).
 * Valid until the next IOTC call on the same thread.
 * Use iotc_last_error_len() for the byte length.
 */
const uint8_t* iotc_last_error(void);

/** Byte length of the last error message. */
size_t iotc_last_error_len(void);

/* ═══════════════════════════════════════════════════════════════════════════
 * Compression / Decompression (caller-provided buffer)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compress data into a caller-provided output buffer.
 *
 * @param input_ptr     Pointer to input data.
 * @param input_len     Length of input in bytes.
 * @param output_ptr    Pointer to output buffer.
 * @param output_capacity  Size of output buffer in bytes.
 * @param stride        Struct stride for columnar transposition (0 = disabled).
 *                      Set to the byte-size of your telemetry struct for best
 *                      results on structured data.
 * @param data_type     Data type hint (IOTC_TYPE_*). Use IOTC_TYPE_AUTO to
 *                      let the library detect the optimal preprocessing.
 *
 * @return On success: number of compressed bytes written (>0).
 *         On failure: negative error code.
 */
int64_t iotc_compress(
    const uint8_t* input_ptr,
    size_t input_len,
    uint8_t* output_ptr,
    size_t output_capacity,
    uint16_t stride,
    uint8_t data_type
);

/**
 * Decompress data into a caller-provided output buffer.
 *
 * @return On success: number of decompressed bytes written (>0).
 *         On failure: negative error code.
 */
int64_t iotc_decompress(
    const uint8_t* input_ptr,
    size_t input_len,
    uint8_t* output_ptr,
    size_t output_capacity
);

/* ═══════════════════════════════════════════════════════════════════════════
 * Compression / Decompression (library-allocated buffer)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compress data — the library allocates the output buffer.
 * Caller MUST free the buffer with iotc_free().
 *
 * @param[out] output_ptr_out  Receives the allocated buffer pointer.
 * @param[out] output_len_out  Receives the compressed byte count.
 * @return IOTC_OK on success, negative error code on failure.
 */
int64_t iotc_compress_alloc(
    const uint8_t* input_ptr,
    size_t input_len,
    uint8_t** output_ptr_out,
    size_t* output_len_out,
    uint16_t stride,
    uint8_t data_type
);

/**
 * Decompress data — the library allocates the output buffer.
 * Caller MUST free the buffer with iotc_free().
 */
int64_t iotc_decompress_alloc(
    const uint8_t* input_ptr,
    size_t input_len,
    uint8_t** output_ptr_out,
    size_t* output_len_out
);

/** Free a buffer allocated by iotc_compress_alloc / iotc_decompress_alloc. */
void iotc_free(uint8_t* ptr, size_t len);

/* ═══════════════════════════════════════════════════════════════════════════
 * Seekable Reader — O(1) Random Block Access
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Opaque handle to a seekable reader. NOT thread-safe. */
typedef struct IotcSeekableReader IotcSeekableReader;

/**
 * Open a seekable reader over compressed data.
 * The data is copied internally — the caller may free input_ptr after this call.
 * The frame must have been compressed with seek tables enabled (default).
 *
 * @return Handle on success, NULL on failure.
 */
IotcSeekableReader* iotc_seekable_open(
    const uint8_t* input_ptr,
    size_t input_len
);

/** Close and free a seekable reader. Passing NULL is a safe no-op. */
void iotc_seekable_close(IotcSeekableReader* handle);

/** Get the number of blocks in the frame. */
int64_t iotc_seekable_block_count(const IotcSeekableReader* handle);

/** Get the original (uncompressed) size in bytes. */
int64_t iotc_seekable_original_size(const IotcSeekableReader* handle);

/**
 * Decompress a single block by index into a caller-provided buffer.
 *
 * @return On success: bytes written. On failure: negative error code.
 */
int64_t iotc_seekable_decompress_block(
    IotcSeekableReader* handle,
    uint32_t block_index,
    uint8_t* output_ptr,
    size_t output_capacity
);

/**
 * Decompress a byte range [byte_start, byte_end) with library allocation.
 * Caller MUST free the buffer with iotc_free().
 */
int64_t iotc_seekable_decompress_range(
    IotcSeekableReader* handle,
    uint64_t byte_start,
    uint64_t byte_end,
    uint8_t** output_ptr_out,
    size_t* output_len_out
);

#ifdef __cplusplus
}
#endif

#endif /* IOTC_H */
