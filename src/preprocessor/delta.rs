//! # Delta-of-Delta Encoding for Integer Time-Series
//!
//! This module implements a two-stage differencing transform followed by
//! ZigZag + variable-length integer encoding. It is the core preprocessing
//! step for integer-typed IoT data (timestamps, counters, monotonic IDs).
//!
//! ## Algorithm
//!
//! Given a sequence of integers `V = [v₀, v₁, v₂, …]`:
//!
//! 1. **First delta**: `D[i] = V[i] - V[i-1]` (with `D[0] = V[0]`).
//!    For a series incrementing by ~1000 each step, deltas become `[v₀, 1000, 1000, 1001, …]`.
//!
//! 2. **Second delta** (optional, on by default): `DD[i] = D[i] - D[i-1]`.
//!    For the same series, double-deltas become `[v₀, d₀, 0, 1, …]` — mostly zeros.
//!
//! 3. **ZigZag encoding**: Maps signed integers to unsigned without sign-extension waste.
//!    `zigzag(n) = (n << 1) ^ (n >> 63)` — maps 0→0, -1→1, 1→2, -2→3, etc.
//!
//! 4. **Varint encoding**: Encodes unsigned integers in 1–10 bytes using 7 bits per byte
//!    with continuation bit. Small values (common after delta-of-delta) use 1 byte.
//!
//! ## Wire Format
//!
//! ```text
//! [0]        flags: u8 (bit 0 = double_delta)
//! [1..9]     first_value: i64 (raw, little-endian)
//! [9..17]    first_delta: i64 (raw, little-endian) — only present if double_delta
//! [17..]     varint-encoded zigzag values
//! ```

use crate::{CompressorError, Result};

// ═══════════════════════════════════════════════════════════════════════════════
// ZigZag Encoding
// ═══════════════════════════════════════════════════════════════════════════════

/// ZigZag-encode a signed 64-bit integer to unsigned.
/// Maps: 0→0, -1→1, 1→2, -2→3, 2→4, …
///
/// This is a branchless operation that compiles to two instructions (shift + xor).
#[inline(always)]
pub fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// ZigZag-decode an unsigned 64-bit integer back to signed.
#[inline(always)]
pub fn zigzag_decode_i64(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

/// ZigZag-encode a signed 32-bit integer to unsigned.
#[inline(always)]
pub fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// ZigZag-decode an unsigned 32-bit integer back to signed.
#[inline(always)]
pub fn zigzag_decode_i32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Variable-Length Integer Encoding (LEB128-style)
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum bytes a varint-encoded u64 can occupy (ceil(64/7) = 10).
pub const MAX_VARINT_LEN: usize = 10;

/// Encode a u64 as a variable-length integer into `buf`.
/// Returns the number of bytes written (1..=10).
///
/// Format: each byte stores 7 data bits in bits [6:0], with bit 7 as a
/// continuation flag (1 = more bytes follow, 0 = final byte).
#[inline]
pub fn encode_varint_u64(mut value: u64, buf: &mut [u8]) -> usize {
    let mut i = 0;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            return i + 1;
        }
        buf[i] = byte | 0x80;
        i += 1;
    }
}

/// Decode a varint-encoded u64 from `data` starting at `pos`.
/// Returns `(value, bytes_consumed)`.
#[inline]
pub fn decode_varint_u64(data: &[u8], pos: usize) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = pos;

    loop {
        if i >= data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 1,
                available: 0,
            });
        }
        let byte = data[i];
        let payload = (byte & 0x7F) as u64;

        // Check for overflow: if we're on the 10th byte, only 1 bit should be used.
        if shift >= 63 && payload > 1 {
            return Err(CompressorError::VarintOverflow(i));
        }

        value |= payload << shift;
        i += 1;

        if byte & 0x80 == 0 {
            return Ok((value, i - pos));
        }

        shift += 7;
        if shift > 63 {
            return Err(CompressorError::VarintOverflow(i));
        }
    }
}

/// Encode a u32 as a variable-length integer. Returns bytes written (1..=5).
#[inline]
pub fn encode_varint_u32(mut value: u32, buf: &mut [u8]) -> usize {
    let mut i = 0;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            return i + 1;
        }
        buf[i] = byte | 0x80;
        i += 1;
    }
}

/// Decode a varint-encoded u32. Returns `(value, bytes_consumed)`.
#[inline]
pub fn decode_varint_u32(data: &[u8], pos: usize) -> Result<(u32, usize)> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut i = pos;

    loop {
        if i >= data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 1,
                available: 0,
            });
        }
        let byte = data[i];
        let payload = (byte & 0x7F) as u32;

        if shift >= 28 && payload > 0x0F {
            return Err(CompressorError::VarintOverflow(i));
        }

        value |= payload << shift;
        i += 1;

        if byte & 0x80 == 0 {
            return Ok((value, i - pos));
        }

        shift += 7;
        if shift > 35 {
            return Err(CompressorError::VarintOverflow(i));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Delta-of-Delta Encoding — i64
// ═══════════════════════════════════════════════════════════════════════════════

/// Wire format flag: double-delta enabled.
const FLAG_DOUBLE_DELTA: u8 = 0x01;

/// Encode a slice of i64 values using delta-of-delta + zigzag + varint.
///
/// The output is a self-contained byte buffer that `decode_i64` can invert.
#[allow(clippy::needless_range_loop)]
pub fn encode_i64(values: &[i64], double_delta: bool) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    // Worst case: 1 flag + 8 first_value + 8 first_delta + 10 bytes per value.
    let mut out = Vec::with_capacity(17 + values.len() * MAX_VARINT_LEN);
    let mut varint_buf = [0u8; MAX_VARINT_LEN];

    // --- Header ---
    let flags: u8 = if double_delta { FLAG_DOUBLE_DELTA } else { 0 };
    out.push(flags);
    out.extend_from_slice(&values[0].to_le_bytes());

    if values.len() == 1 {
        return Ok(out);
    }

    // --- First delta pass ---
    // Compute deltas: d[i] = v[i] - v[i-1], using wrapping subtraction for safety.
    let mut deltas: Vec<i64> = Vec::with_capacity(values.len() - 1);
    for i in 1..values.len() {
        deltas.push(values[i].wrapping_sub(values[i - 1]));
    }

    if double_delta {
        // Store the first delta raw so we can reconstruct.
        out.extend_from_slice(&deltas[0].to_le_bytes());

        if deltas.len() > 1 {
            // --- Second delta pass ---
            // dd[i] = d[i] - d[i-1]
            let mut prev = deltas[0];
            for i in 1..deltas.len() {
                let dd = deltas[i].wrapping_sub(prev);
                prev = deltas[i];
                let zz = zigzag_encode_i64(dd);
                let n = encode_varint_u64(zz, &mut varint_buf);
                out.extend_from_slice(&varint_buf[..n]);
            }
        }
    } else {
        // Single delta: zigzag + varint encode each delta directly.
        for &d in &deltas {
            let zz = zigzag_encode_i64(d);
            let n = encode_varint_u64(zz, &mut varint_buf);
            out.extend_from_slice(&varint_buf[..n]);
        }
    }

    Ok(out)
}

/// Decode a delta-of-delta encoded i64 stream back to original values.
///
/// `expected_count` is used to pre-allocate and validate output length.
pub fn decode_i64(data: &[u8], expected_count: usize) -> Result<Vec<i64>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::DeltaDecode("expected_count is 0".into()));
    }

    let mut pos: usize = 0;
    let mut values: Vec<i64> = Vec::with_capacity(expected_count);

    // --- Read flags ---
    let flags = data[pos];
    pos += 1;
    if flags & !FLAG_DOUBLE_DELTA != 0 {
        return Err(CompressorError::DeltaDecode(format!(
            "unknown flag bits: 0x{:02X}",
            flags & !FLAG_DOUBLE_DELTA
        )));
    }
    let double_delta = (flags & FLAG_DOUBLE_DELTA) != 0;

    // --- Read first value ---
    if pos + 8 > data.len() {
        return Err(CompressorError::BufferUnderflow {
            needed: 8,
            available: data.len() - pos,
        });
    }
    let first_value = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    values.push(first_value);

    if expected_count == 1 {
        return Ok(values);
    }

    if double_delta {
        // --- Read first delta ---
        if pos + 8 > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 8,
                available: data.len() - pos,
            });
        }
        let first_delta = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        // Reconstruct: v[1] = v[0] + d[0]
        values.push(first_value.wrapping_add(first_delta));

        // Read double-deltas and reconstruct.
        let mut prev_delta = first_delta;
        for _ in 2..expected_count {
            let (zz, consumed) = decode_varint_u64(data, pos)?;
            pos += consumed;
            let dd = zigzag_decode_i64(zz);
            let delta = prev_delta.wrapping_add(dd);
            let value = values.last().unwrap().wrapping_add(delta);
            values.push(value);
            prev_delta = delta;
        }
    } else {
        // Single delta.
        let mut prev = first_value;
        for _ in 1..expected_count {
            let (zz, consumed) = decode_varint_u64(data, pos)?;
            pos += consumed;
            let d = zigzag_decode_i64(zz);
            let value = prev.wrapping_add(d);
            values.push(value);
            prev = value;
        }
    }

    if values.len() != expected_count {
        return Err(CompressorError::DeltaDecode(format!(
            "decoded {} values, expected {}",
            values.len(),
            expected_count
        )));
    }

    if pos < data.len() {
        return Err(CompressorError::DeltaDecode(format!(
            "{} trailing bytes after decoding {} i64 values",
            data.len() - pos,
            expected_count
        )));
    }

    Ok(values)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Delta-of-Delta Encoding — u64
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode u64 values. Internally casts to i64 for delta computation (wrapping).
pub fn encode_u64(values: &[u64], double_delta: bool) -> Result<Vec<u8>> {
    // Reinterpret as i64 — wrapping arithmetic makes this lossless.
    let i64_values: Vec<i64> = values.iter().map(|&v| v as i64).collect();
    encode_i64(&i64_values, double_delta)
}

/// Decode u64 values.
pub fn decode_u64(data: &[u8], expected_count: usize) -> Result<Vec<u64>> {
    let i64_values = decode_i64(data, expected_count)?;
    Ok(i64_values.iter().map(|&v| v as u64).collect())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Delta-of-Delta Encoding — i32
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a slice of i32 values using delta-of-delta + zigzag + varint.
///
/// Uses 32-bit zigzag and varint for tighter packing than promoting to i64.
#[allow(clippy::needless_range_loop)]
pub fn encode_i32(values: &[i32], double_delta: bool) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let mut out = Vec::with_capacity(9 + values.len() * 5);
    let mut varint_buf = [0u8; 5]; // max varint32 = 5 bytes

    // --- Header ---
    let flags: u8 = if double_delta { FLAG_DOUBLE_DELTA } else { 0 };
    out.push(flags);
    out.extend_from_slice(&values[0].to_le_bytes());

    if values.len() == 1 {
        return Ok(out);
    }

    // --- First delta pass ---
    let mut deltas: Vec<i32> = Vec::with_capacity(values.len() - 1);
    for i in 1..values.len() {
        deltas.push(values[i].wrapping_sub(values[i - 1]));
    }

    if double_delta {
        out.extend_from_slice(&deltas[0].to_le_bytes());

        if deltas.len() > 1 {
            let mut prev = deltas[0];
            for i in 1..deltas.len() {
                let dd = deltas[i].wrapping_sub(prev);
                prev = deltas[i];
                let zz = zigzag_encode_i32(dd);
                let n = encode_varint_u32(zz, &mut varint_buf);
                out.extend_from_slice(&varint_buf[..n]);
            }
        }
    } else {
        for &d in &deltas {
            let zz = zigzag_encode_i32(d);
            let n = encode_varint_u32(zz, &mut varint_buf);
            out.extend_from_slice(&varint_buf[..n]);
        }
    }

    Ok(out)
}

/// Decode a delta-of-delta encoded i32 stream.
pub fn decode_i32(data: &[u8], expected_count: usize) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::DeltaDecode("expected_count is 0".into()));
    }

    let mut pos: usize = 0;
    let mut values: Vec<i32> = Vec::with_capacity(expected_count);

    let flags = data[pos];
    pos += 1;
    if flags & !FLAG_DOUBLE_DELTA != 0 {
        return Err(CompressorError::DeltaDecode(format!(
            "unknown flag bits: 0x{:02X}",
            flags & !FLAG_DOUBLE_DELTA
        )));
    }
    let double_delta = (flags & FLAG_DOUBLE_DELTA) != 0;

    if pos + 4 > data.len() {
        return Err(CompressorError::BufferUnderflow {
            needed: 4,
            available: data.len() - pos,
        });
    }
    let first_value = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    values.push(first_value);

    if expected_count == 1 {
        return Ok(values);
    }

    if double_delta {
        if pos + 4 > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 4,
                available: data.len() - pos,
            });
        }
        let first_delta = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        values.push(first_value.wrapping_add(first_delta));

        let mut prev_delta = first_delta;
        for _ in 2..expected_count {
            let (zz, consumed) = decode_varint_u32(data, pos)?;
            pos += consumed;
            let dd = zigzag_decode_i32(zz);
            let delta = prev_delta.wrapping_add(dd);
            let value = values.last().unwrap().wrapping_add(delta);
            values.push(value);
            prev_delta = delta;
        }
    } else {
        let mut prev = first_value;
        for _ in 1..expected_count {
            let (zz, consumed) = decode_varint_u32(data, pos)?;
            pos += consumed;
            let d = zigzag_decode_i32(zz);
            let value = prev.wrapping_add(d);
            values.push(value);
            prev = value;
        }
    }

    if values.len() != expected_count {
        return Err(CompressorError::DeltaDecode(format!(
            "decoded {} values, expected {}",
            values.len(),
            expected_count
        )));
    }

    if pos < data.len() {
        return Err(CompressorError::DeltaDecode(format!(
            "{} trailing bytes after decoding {} i32 values",
            data.len() - pos,
            expected_count
        )));
    }

    Ok(values)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Delta-of-Delta Encoding — u32
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode u32 values. Internally reinterprets as i32 (wrapping).
pub fn encode_u32(values: &[u32], double_delta: bool) -> Result<Vec<u8>> {
    let i32_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();
    encode_i32(&i32_values, double_delta)
}

/// Decode u32 values.
pub fn decode_u32(data: &[u8], expected_count: usize) -> Result<Vec<u32>> {
    let i32_values = decode_i32(data, expected_count)?;
    Ok(i32_values.iter().map(|&v| v as u32).collect())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace-Aware Decode — Write LE Bytes Directly
// ═══════════════════════════════════════════════════════════════════════════════

/// Workspace-aware i64 decoder. Writes LE bytes directly to `out` instead of
/// building an intermediate `Vec<i64>`. `out` is cleared and reused.
pub fn decode_i64_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::DeltaDecode("expected_count is 0".into()));
    }

    out.reserve(expected_count * 8);
    let mut pos: usize = 0;

    let flags = data[pos];
    pos += 1;
    if flags & !FLAG_DOUBLE_DELTA != 0 {
        return Err(CompressorError::DeltaDecode(format!(
            "unknown flag bits: 0x{:02X}",
            flags & !FLAG_DOUBLE_DELTA
        )));
    }
    let double_delta = (flags & FLAG_DOUBLE_DELTA) != 0;

    if pos + 8 > data.len() {
        return Err(CompressorError::BufferUnderflow {
            needed: 8,
            available: data.len() - pos,
        });
    }
    let first_value = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
    pos += 8;
    out.extend_from_slice(&first_value.to_le_bytes());

    if expected_count == 1 {
        if pos < data.len() {
            return Err(CompressorError::DeltaDecode(format!(
                "{} trailing bytes after decoding 1 i64 value",
                data.len() - pos
            )));
        }
        return Ok(());
    }

    let mut decoded = 1usize;

    if double_delta {
        if pos + 8 > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 8,
                available: data.len() - pos,
            });
        }
        let first_delta = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let second = first_value.wrapping_add(first_delta);
        out.extend_from_slice(&second.to_le_bytes());
        decoded += 1;

        let mut prev_value = second;
        let mut prev_delta = first_delta;
        for _ in 2..expected_count {
            let (zz, consumed) = decode_varint_u64(data, pos)?;
            pos += consumed;
            let dd = zigzag_decode_i64(zz);
            let delta = prev_delta.wrapping_add(dd);
            let value = prev_value.wrapping_add(delta);
            out.extend_from_slice(&value.to_le_bytes());
            prev_value = value;
            prev_delta = delta;
            decoded += 1;
        }
    } else {
        let mut prev = first_value;
        for _ in 1..expected_count {
            let (zz, consumed) = decode_varint_u64(data, pos)?;
            pos += consumed;
            let d = zigzag_decode_i64(zz);
            let value = prev.wrapping_add(d);
            out.extend_from_slice(&value.to_le_bytes());
            prev = value;
            decoded += 1;
        }
    }

    if decoded != expected_count {
        return Err(CompressorError::DeltaDecode(format!(
            "decoded {} values, expected {}",
            decoded, expected_count
        )));
    }
    if pos < data.len() {
        return Err(CompressorError::DeltaDecode(format!(
            "{} trailing bytes after decoding {} i64 values",
            data.len() - pos,
            expected_count
        )));
    }
    Ok(())
}

/// Workspace-aware u64 decoder. Writes LE bytes directly to `out`.
pub fn decode_u64_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    // u64 delegates to i64 via wrapping reinterpret — the LE bytes are identical.
    decode_i64_into(data, expected_count, out)
}

/// Workspace-aware i32 decoder. Writes LE bytes directly to `out`.
pub fn decode_i32_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::DeltaDecode("expected_count is 0".into()));
    }

    out.reserve(expected_count * 4);
    let mut pos: usize = 0;

    let flags = data[pos];
    pos += 1;
    if flags & !FLAG_DOUBLE_DELTA != 0 {
        return Err(CompressorError::DeltaDecode(format!(
            "unknown flag bits: 0x{:02X}",
            flags & !FLAG_DOUBLE_DELTA
        )));
    }
    let double_delta = (flags & FLAG_DOUBLE_DELTA) != 0;

    if pos + 4 > data.len() {
        return Err(CompressorError::BufferUnderflow {
            needed: 4,
            available: data.len() - pos,
        });
    }
    let first_value = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    out.extend_from_slice(&first_value.to_le_bytes());

    if expected_count == 1 {
        if pos < data.len() {
            return Err(CompressorError::DeltaDecode(format!(
                "{} trailing bytes after decoding 1 i32 value",
                data.len() - pos
            )));
        }
        return Ok(());
    }

    let mut decoded = 1usize;

    if double_delta {
        if pos + 4 > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 4,
                available: data.len() - pos,
            });
        }
        let first_delta = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let second = first_value.wrapping_add(first_delta);
        out.extend_from_slice(&second.to_le_bytes());
        decoded += 1;

        let mut prev_value = second;
        let mut prev_delta = first_delta;
        for _ in 2..expected_count {
            let (zz, consumed) = decode_varint_u32(data, pos)?;
            pos += consumed;
            let dd = zigzag_decode_i32(zz);
            let delta = prev_delta.wrapping_add(dd);
            let value = prev_value.wrapping_add(delta);
            out.extend_from_slice(&value.to_le_bytes());
            prev_value = value;
            prev_delta = delta;
            decoded += 1;
        }
    } else {
        let mut prev = first_value;
        for _ in 1..expected_count {
            let (zz, consumed) = decode_varint_u32(data, pos)?;
            pos += consumed;
            let d = zigzag_decode_i32(zz);
            let value = prev.wrapping_add(d);
            out.extend_from_slice(&value.to_le_bytes());
            prev = value;
            decoded += 1;
        }
    }

    if decoded != expected_count {
        return Err(CompressorError::DeltaDecode(format!(
            "decoded {} values, expected {}",
            decoded, expected_count
        )));
    }
    if pos < data.len() {
        return Err(CompressorError::DeltaDecode(format!(
            "{} trailing bytes after decoding {} i32 values",
            data.len() - pos,
            expected_count
        )));
    }
    Ok(())
}

/// Workspace-aware u32 decoder. Writes LE bytes directly to `out`.
pub fn decode_u32_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    // u32 delegates to i32 via wrapping reinterpret — the LE bytes are identical.
    decode_i32_into(data, expected_count, out)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ZigZag ──────────────────────────────────────────────────────────

    #[test]
    fn zigzag_i64_known_values() {
        assert_eq!(zigzag_encode_i64(0), 0);
        assert_eq!(zigzag_encode_i64(-1), 1);
        assert_eq!(zigzag_encode_i64(1), 2);
        assert_eq!(zigzag_encode_i64(-2), 3);
        assert_eq!(zigzag_encode_i64(2), 4);
        assert_eq!(zigzag_encode_i64(i64::MAX), u64::MAX - 1);
        assert_eq!(zigzag_encode_i64(i64::MIN), u64::MAX);
    }

    #[test]
    fn zigzag_i64_roundtrip_exhaustive_small() {
        for n in -10000..=10000i64 {
            let encoded = zigzag_encode_i64(n);
            let decoded = zigzag_decode_i64(encoded);
            assert_eq!(decoded, n, "ZigZag roundtrip failed for {}", n);
        }
    }

    #[test]
    fn zigzag_i64_roundtrip_extremes() {
        for &n in &[i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX] {
            assert_eq!(zigzag_decode_i64(zigzag_encode_i64(n)), n);
        }
    }

    #[test]
    fn zigzag_i32_known_values() {
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(i32::MAX), u32::MAX - 1);
        assert_eq!(zigzag_encode_i32(i32::MIN), u32::MAX);
    }

    #[test]
    fn zigzag_i32_roundtrip() {
        for &n in &[i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX] {
            assert_eq!(zigzag_decode_i32(zigzag_encode_i32(n)), n);
        }
    }

    // ── Varint ──────────────────────────────────────────────────────────

    #[test]
    fn varint_u64_single_byte() {
        let mut buf = [0u8; MAX_VARINT_LEN];
        for v in 0..128u64 {
            let n = encode_varint_u64(v, &mut buf);
            assert_eq!(n, 1, "value {} should encode in 1 byte", v);
            assert_eq!(buf[0], v as u8);
            let (decoded, consumed) = decode_varint_u64(&buf, 0).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(consumed, 1);
        }
    }

    #[test]
    fn varint_u64_multi_byte() {
        let mut buf = [0u8; MAX_VARINT_LEN];
        let test_values: Vec<u64> = vec![
            128,
            255,
            256,
            16383,
            16384,
            2097151,
            2097152,
            u32::MAX as u64,
            u64::MAX / 2,
            u64::MAX,
        ];
        for &v in &test_values {
            let n = encode_varint_u64(v, &mut buf);
            let (decoded, consumed) = decode_varint_u64(&buf[..n], 0).unwrap();
            assert_eq!(decoded, v, "Varint roundtrip failed for {}", v);
            assert_eq!(consumed, n);
        }
    }

    #[test]
    fn varint_u64_max_value() {
        let mut buf = [0u8; MAX_VARINT_LEN];
        let n = encode_varint_u64(u64::MAX, &mut buf);
        assert_eq!(n, 10); // u64::MAX requires 10 bytes
        let (decoded, consumed) = decode_varint_u64(&buf, 0).unwrap();
        assert_eq!(decoded, u64::MAX);
        assert_eq!(consumed, 10);
    }

    #[test]
    fn varint_u64_decode_from_offset() {
        let mut buf = [0u8; 20];
        // Write a padding byte, then a varint at offset 5.
        buf[0..5].fill(0xFF);
        let n = encode_varint_u64(42, &mut buf[5..]);
        let (decoded, consumed) = decode_varint_u64(&buf, 5).unwrap();
        assert_eq!(decoded, 42);
        assert_eq!(consumed, n);
    }

    #[test]
    fn varint_u32_roundtrip() {
        let mut buf = [0u8; 5];
        let test_values: Vec<u32> = vec![0, 1, 127, 128, 16383, 16384, u32::MAX / 2, u32::MAX];
        for &v in &test_values {
            let n = encode_varint_u32(v, &mut buf);
            let (decoded, consumed) = decode_varint_u32(&buf[..n], 0).unwrap();
            assert_eq!(decoded, v, "Varint32 roundtrip failed for {}", v);
            assert_eq!(consumed, n);
        }
    }

    #[test]
    fn varint_decode_buffer_underflow() {
        // Empty buffer.
        assert!(matches!(
            decode_varint_u64(&[], 0),
            Err(CompressorError::BufferUnderflow { .. })
        ));
        // Continuation bit set but no next byte.
        assert!(matches!(
            decode_varint_u64(&[0x80], 0),
            Err(CompressorError::BufferUnderflow { .. })
        ));
    }

    // ── Delta-of-Delta i64 ─────────────────────────────────────────────

    #[test]
    fn delta_i64_single_value() {
        let values = vec![42i64];
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 1).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_two_values() {
        let values = vec![100i64, 200];
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 2).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_constant_sequence() {
        // All same value → deltas are 0, double-deltas are 0. Maximum compression.
        let values: Vec<i64> = vec![42; 1000];
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 1000).unwrap();
        assert_eq!(decoded, values);

        // Should be very compact: 1 flag + 8 first_value + 8 first_delta + 998 * 1 byte (varint 0)
        assert!(
            encoded.len() <= 17 + 998,
            "Constant sequence should compress well, got {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn delta_i64_linear_sequence() {
        // Linear sequence: v[i] = 1000 + i * 100
        // Deltas: [100, 100, 100, …] → Double deltas: [0, 0, 0, …]
        let values: Vec<i64> = (0..1000).map(|i| 1000 + i * 100).collect();
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 1000).unwrap();
        assert_eq!(decoded, values);

        // Double-deltas are all zero → 1 byte each.
        let header_size = 1 + 8 + 8; // flag + first_value + first_delta
        let expected_max = header_size + 998; // 998 varint(0)
        assert!(
            encoded.len() <= expected_max,
            "Linear sequence should compress to ~{} bytes, got {}",
            expected_max,
            encoded.len()
        );
    }

    #[test]
    fn delta_i64_quadratic_sequence() {
        // v[i] = i² → deltas = [1, 3, 5, 7, …] → double deltas = [2, 2, 2, …]
        let values: Vec<i64> = (0..500).map(|i| i * i).collect();
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 500).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_negative_values() {
        let values: Vec<i64> = (-500..500).collect();
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 1000).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_extreme_values() {
        let values = vec![i64::MIN, 0, i64::MAX, i64::MIN, i64::MAX];
        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 5).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_timestamps_realistic() {
        // Simulate real Unix timestamps in nanoseconds with jitter.
        let base: i64 = 1_700_000_000_000_000_000; // ~2023 in nanos
        let step: i64 = 1_000_000; // 1ms intervals
        let values: Vec<i64> = (0..10_000)
            .map(|i| {
                let jitter = ((i as i64 * 7 + 13) % 100) - 50; // deterministic pseudo-jitter
                base + i as i64 * step + jitter
            })
            .collect();

        let encoded = encode_i64(&values, true).unwrap();
        let decoded = decode_i64(&encoded, 10_000).unwrap();
        assert_eq!(decoded, values);

        // Check compression ratio.
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 3.0,
            "Expected >3x compression on timestamps, got {:.1}x ({} → {} bytes)",
            ratio,
            raw_size,
            encoded.len()
        );
    }

    #[test]
    fn delta_i64_single_delta_mode() {
        // Test without double-delta (single delta only).
        let values: Vec<i64> = (0..100).map(|i| i * 1000).collect();
        let encoded = encode_i64(&values, false).unwrap();
        let decoded = decode_i64(&encoded, 100).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i64_empty_fails() {
        assert!(matches!(
            encode_i64(&[], true),
            Err(CompressorError::EmptyInput)
        ));
    }

    // ── Delta-of-Delta u64 ─────────────────────────────────────────────

    #[test]
    fn delta_u64_roundtrip() {
        let values: Vec<u64> = (0..1000).map(|i| i * 12345).collect();
        let encoded = encode_u64(&values, true).unwrap();
        let decoded = decode_u64(&encoded, 1000).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_u64_large_values() {
        let values: Vec<u64> = (0..100).map(|i| u64::MAX - i * 1000).collect();
        let encoded = encode_u64(&values, true).unwrap();
        let decoded = decode_u64(&encoded, 100).unwrap();
        assert_eq!(decoded, values);
    }

    // ── Delta-of-Delta i32 ─────────────────────────────────────────────

    #[test]
    fn delta_i32_linear() {
        let values: Vec<i32> = (0..500).map(|i| 100 + i * 10).collect();
        let encoded = encode_i32(&values, true).unwrap();
        let decoded = decode_i32(&encoded, 500).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i32_negative() {
        let values: Vec<i32> = (-200..200).collect();
        let encoded = encode_i32(&values, true).unwrap();
        let decoded = decode_i32(&encoded, 400).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i32_extremes() {
        let values = vec![i32::MIN, 0, i32::MAX, i32::MIN];
        let encoded = encode_i32(&values, true).unwrap();
        let decoded = decode_i32(&encoded, 4).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_i32_single_delta() {
        let values: Vec<i32> = (0..100).map(|i| i * 50).collect();
        let encoded = encode_i32(&values, false).unwrap();
        let decoded = decode_i32(&encoded, 100).unwrap();
        assert_eq!(decoded, values);
    }

    // ── Delta-of-Delta u32 ─────────────────────────────────────────────

    #[test]
    fn delta_u32_roundtrip() {
        let values: Vec<u32> = (0..1000).map(|i| i * 999).collect();
        let encoded = encode_u32(&values, true).unwrap();
        let decoded = decode_u32(&encoded, 1000).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn delta_u32_max_values() {
        let values = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2, u32::MAX - 3];
        let encoded = encode_u32(&values, true).unwrap();
        let decoded = decode_u32(&encoded, 4).unwrap();
        assert_eq!(decoded, values);
    }

    // ── Compression Effectiveness ──────────────────────────────────────

    #[test]
    fn compression_ratio_constant_i64() {
        let values: Vec<i64> = vec![999_999; 10_000];
        let encoded = encode_i64(&values, true).unwrap();
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        // Constant series: all double-deltas are 0.
        // Should achieve massive compression.
        assert!(
            ratio > 5.0,
            "Constant i64 series: expected >5x, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn compression_ratio_linear_i64() {
        let values: Vec<i64> = (0..10_000).map(|i| 1_000_000 + i * 1000).collect();
        let encoded = encode_i64(&values, true).unwrap();
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 5.0,
            "Linear i64 series: expected >5x, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn compression_ratio_quadratic_i32() {
        let values: Vec<i32> = (0..5_000).map(|i| i * i / 100).collect();
        let encoded = encode_i32(&values, true).unwrap();
        let raw_size = values.len() * 4;
        let ratio = raw_size as f64 / encoded.len() as f64;
        // Quadratic: double-deltas are constant small values.
        assert!(
            ratio > 2.0,
            "Quadratic i32 series: expected >2x, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn single_delta_vs_double_delta() {
        // For a sequence with slowly changing deltas, double-delta should win.
        // Pure linear has constant deltas (both encode well), but the 8-byte
        // first_delta header makes double-delta slightly bigger for that case.
        // Use quadratic data where deltas themselves change linearly.
        let values: Vec<i64> = (0..10_000).map(|i| i * i + 1000 * i).collect();
        let single = encode_i64(&values, false).unwrap();
        let double = encode_i64(&values, true).unwrap();
        assert!(
            double.len() < single.len(),
            "Double-delta ({}) should be < single-delta ({}) for quadratic data",
            double.len(),
            single.len()
        );
    }
}
