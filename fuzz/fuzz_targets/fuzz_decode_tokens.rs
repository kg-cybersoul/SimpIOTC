//! Fuzz target for `entropy::decode_tokens()`.
//!
//! Feeds arbitrary bytes into the FSE/ANS token decoder. This is the most
//! complex parser in the codebase — it reads frequency tables, FSE state
//! machines, and variable-width extra bits. Any of these can be malformed.
//!
//! Must return Err(CompressorError) on invalid input, never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = iot_compressor::entropy::decode_tokens(data);
});
