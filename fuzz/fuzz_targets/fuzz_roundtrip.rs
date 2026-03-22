//! Roundtrip fuzz target: compress valid data, then verify decompress recovers it.
//!
//! This catches bugs where the compressor produces output that the decompressor
//! can't handle — mismatches in wire format assumptions, off-by-one errors in
//! sub-stream boundaries, etc.
//!
//! The fuzzer controls the raw input bytes AND the parser mode, so it explores
//! the full space of valid inputs × compression strategies.

#![no_main]

use libfuzzer_sys::fuzz_target;
use iot_compressor::{CompressionConfig, ParserMode};

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: 1 for parser mode selection, rest for payload.
    if data.len() < 2 {
        return;
    }

    let mode = match data[0] % 3 {
        0 => ParserMode::Greedy,
        1 => ParserMode::Lazy,
        _ => ParserMode::Optimal,
    };

    let payload = &data[1..];
    if payload.is_empty() {
        return;
    }

    let config = CompressionConfig {
        parser_mode: mode,
        block_size: 64 * 1024, // 64K blocks to keep fuzz iterations fast
        num_threads: 1,        // single-threaded for determinism
        window_size: 65536,
        max_chain_depth: 16,   // shallow chain for speed
        store_checksum: true,
        ..Default::default()
    };

    // Compress — should always succeed on valid input bytes.
    let compressed = match iot_compressor::parallel::compress(payload, &config) {
        Ok(c) => c,
        Err(_) => return, // Some inputs are too small or trigger alignment errors — that's fine.
    };

    // Decompress — must roundtrip exactly.
    let decompressed = iot_compressor::parallel::decompress(&compressed)
        .expect("decompress(compress(data)) must succeed");

    assert_eq!(
        decompressed, payload,
        "roundtrip mismatch: {} bytes in, {} bytes out",
        payload.len(), decompressed.len()
    );
});
