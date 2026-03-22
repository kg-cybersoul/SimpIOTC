//! Fuzz target for `parallel::decompress()`.
//!
//! Feeds arbitrary bytes into the decompressor and asserts it either
//! returns Ok (valid data) or Err (graceful rejection). Must never panic,
//! stack overflow, or OOM on any input.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The decompressor must handle any byte sequence without panicking.
    // Valid compressed data returns Ok, garbage returns Err — both are fine.
    let _ = iot_compressor::parallel::decompress(data);
});
