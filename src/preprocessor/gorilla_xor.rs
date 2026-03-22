//! # Gorilla XOR Compression for Floating-Point Time-Series
//!
//! Implements the XOR-based floating-point compression scheme described in
//! Facebook's "Gorilla: A Fast, Scalable, In-Memory Time Series Database"
//! (Pelkonen et al., VLDB 2015).
//!
//! ## Algorithm
//!
//! Consecutive sensor readings (temperature, voltage, pressure) often share
//! most of their IEEE 754 bit representation — same sign, same exponent,
//! and similar high mantissa bits. XORing adjacent values produces results
//! dominated by zeros, which we pack tightly at the bit level.
//!
//! ### Encoding scheme (per value after the first):
//!
//! 1. Compute `xor = current_bits ^ previous_bits` (as u64/u32).
//!
//! 2. If `xor == 0` (identical values):
//!    - Write a single `0` bit.
//!
//! 3. If `xor != 0`:
//!    - Write a `1` bit.
//!    - Count `leading_zeros` and `trailing_zeros` of the XOR result.
//!    - The "meaningful bits" are in the middle: `meaningful_len = 64 - leading - trailing`.
//!    - If the meaningful window fits within the previous value's window
//!      (i.e., `leading >= prev_leading` and `trailing >= prev_trailing`):
//!      - Write `0` (control bit): reuse previous window.
//!      - Write just the `prev_meaningful_len` bits from the same position.
//!    - Otherwise:
//!      - Write `1` (control bit): new window.
//!      - Write 6 bits: `leading_zeros` (0–63).
//!      - Write 6 bits: `meaningful_len - 1` (0–63, representing 1–64).
//!      - Write `meaningful_len` bits: the meaningful portion of `xor`.
//!
//! ### Wire format:
//!
//! ```text
//! [0..8]  first value raw (f64 little-endian) or [0..4] for f32
//! [8..]   packed bitstream as described above, byte-aligned with trailing padding
//! ```
//!
//! ## Bit I/O
//!
//! We implement a `BitWriter` and `BitReader` that pack/unpack individual
//! bits into byte arrays, MSB-first within each byte (matching the original
//! Gorilla paper convention).

use crate::{CompressorError, Result};

// ═══════════════════════════════════════════════════════════════════════════════
// Bit Writer
// ═══════════════════════════════════════════════════════════════════════════════

/// High-throughput bit writer using a u64 accumulator.
///
/// Bits are shifted into a 64-bit register and flushed as complete bytes
/// when the accumulator fills. This eliminates per-bit branches and lets
/// `write_bits` compile down to a shift + OR + conditional memcpy — no loops.
struct BitWriter {
    buffer: Vec<u8>,
    /// Accumulator holding bits MSB-first (bit 63 is the next bit to be flushed).
    accum: u64,
    /// Number of valid bits in `accum` (counted from the MSB side).
    bits_in_accum: u8,
}

impl BitWriter {
    #[cfg(test)]
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            accum: 0,
            bits_in_accum: 0,
        }
    }

    fn with_capacity(byte_capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(byte_capacity),
            accum: 0,
            bits_in_accum: 0,
        }
    }

    /// Flush complete bytes from the accumulator to the buffer.
    #[inline]
    fn flush_bytes(&mut self) {
        while self.bits_in_accum >= 8 {
            // Top 8 bits of accum are the next byte.
            let byte = (self.accum >> 56) as u8;
            self.buffer.push(byte);
            self.accum <<= 8;
            self.bits_in_accum -= 8;
        }
    }

    /// Write a single bit (0 or 1).
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        self.accum |= (bit as u64) << (63 - self.bits_in_accum);
        self.bits_in_accum += 1;
        if self.bits_in_accum >= 8 {
            self.flush_bytes();
        }
    }

    /// Write `count` bits from the lowest bits of `value`.
    /// `count` must be in 1..=64.
    ///
    /// Fast path: a single shift + OR when bits fit in the accumulator.
    /// Split path: write what fits, flush, write the remainder.
    /// After flush_bytes(), bits_in_accum is 0–7, so the split remainder
    /// is at most 7 bits and always fits.
    #[inline]
    fn write_bits(&mut self, value: u64, count: u8) {
        debug_assert!(count > 0 && count <= 64);

        // Mask to the relevant bits.
        let masked = if count == 64 {
            value
        } else {
            value & ((1u64 << count) - 1)
        };

        let avail = 64 - self.bits_in_accum; // space remaining (57..=64)

        if count <= avail {
            // Fast path: all bits fit.
            let shift = avail - count;
            self.accum |= masked << shift;
            self.bits_in_accum += count;
            self.flush_bytes();
        } else {
            // Split path: write top `avail` bits, flush, write remaining.
            let overflow = count - avail;
            // Top `avail` bits of the value (the high portion).
            self.accum |= masked >> overflow;
            self.bits_in_accum = 64;
            self.flush_bytes();
            // After flush: bits_in_accum == 0, accumulator is empty.
            // Write the remaining `overflow` bits (always <= 7).
            let remaining = masked & ((1u64 << overflow) - 1);
            let shift = 64 - overflow;
            self.accum = remaining << shift;
            self.bits_in_accum = overflow;
            // No flush needed — overflow <= 7, so < 8 bits.
        }
    }

    /// Write `count` bits from the lowest bits of a u32 value.
    /// `count` must be in 1..=32.
    #[inline]
    fn write_bits_u32(&mut self, value: u32, count: u8) {
        self.write_bits(value as u64, count);
    }

    /// Flush any remaining bits (zero-padded) and return the byte buffer.
    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_accum > 0 {
            // The remaining bits are already MSB-aligned in accum.
            // Push the top byte (which contains the remaining bits, zero-padded).
            let byte = (self.accum >> 56) as u8;
            self.buffer.push(byte);
        }
        self.buffer
    }

    /// Number of complete bytes written so far (not counting partial accumulator).
    #[allow(dead_code)]
    fn bytes_written(&self) -> usize {
        self.buffer.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bit Reader
// ═══════════════════════════════════════════════════════════════════════════════

/// High-throughput bit reader using a u64 accumulator.
///
/// Refills from the byte slice in bulk (up to 8 bytes at a time) and
/// serves bit requests from the register via shifts — no per-bit branches.
struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to consume from `data`.
    byte_pos: usize,
    /// Accumulator holding bits MSB-first (bit 63 is the next bit to read).
    accum: u64,
    /// Number of valid bits remaining in `accum`.
    bits_in_accum: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            byte_pos: 0,
            accum: 0,
            bits_in_accum: 0,
        };
        reader.refill();
        reader
    }

    /// Bulk-refill the accumulator from the byte stream.
    /// Loads as many bytes as will fit (up to 8).
    #[inline]
    fn refill(&mut self) {
        while self.bits_in_accum <= 56 && self.byte_pos < self.data.len() {
            self.accum |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_accum);
            self.bits_in_accum += 8;
            self.byte_pos += 1;
        }
    }

    /// Read a single bit. Returns `true` for 1, `false` for 0.
    #[inline]
    fn read_bit(&mut self) -> Result<bool> {
        if self.bits_in_accum == 0 {
            self.refill();
            if self.bits_in_accum == 0 {
                return Err(CompressorError::GorillaDecode(
                    "unexpected end of bitstream".into(),
                ));
            }
        }
        let bit = (self.accum >> 63) & 1;
        self.accum <<= 1;
        self.bits_in_accum -= 1;
        // Opportunistically refill when running low.
        if self.bits_in_accum <= 56 {
            self.refill();
        }
        Ok(bit == 1)
    }

    /// Read `count` bits and return them as the lowest bits of a u64.
    /// `count` must be in 1..=64. Fast path: single shift extraction.
    /// Split path for count > bits_in_accum: extract what we have, refill, extract rest.
    #[inline]
    fn read_bits(&mut self, count: u8) -> Result<u64> {
        debug_assert!(count > 0 && count <= 64);

        if count <= self.bits_in_accum {
            // Fast path: all requested bits are already in the accumulator.
            let value = self.accum >> (64 - count);
            if count == 64 {
                self.accum = 0;
            } else {
                self.accum <<= count;
            }
            self.bits_in_accum -= count;
            if self.bits_in_accum <= 56 {
                self.refill();
            }
            let masked = if count == 64 {
                value
            } else {
                value & ((1u64 << count) - 1)
            };
            Ok(masked)
        } else {
            // Split path: drain the accumulator, refill, read the remainder.
            let have = self.bits_in_accum;
            if have == 0 {
                self.refill();
                if self.bits_in_accum < count {
                    return Err(CompressorError::GorillaDecode(
                        "unexpected end of bitstream".into(),
                    ));
                }
                return self.read_bits(count);
            }
            // Read the `have` bits we already hold.
            let high = self.accum >> (64 - have);
            self.accum = 0;
            self.bits_in_accum = 0;
            self.refill();
            let need = count - have;
            if self.bits_in_accum < need {
                return Err(CompressorError::GorillaDecode(
                    "unexpected end of bitstream".into(),
                ));
            }
            // Read the remaining `need` bits.
            let low = self.accum >> (64 - need);
            if need == 64 {
                self.accum = 0;
            } else {
                self.accum <<= need;
            }
            self.bits_in_accum -= need;
            if self.bits_in_accum <= 56 {
                self.refill();
            }
            let combined = (high << need) | (low & ((1u64 << need) - 1));
            Ok(combined)
        }
    }

    /// Read `count` bits as u32. `count` must be in 1..=32.
    #[inline]
    fn read_bits_u32(&mut self, count: u8) -> Result<u32> {
        let v = self.read_bits(count)?;
        Ok(v as u32)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gorilla XOR — f64
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a slice of f64 values using Gorilla XOR compression.
///
/// Returns a byte buffer containing the first value raw followed by a
/// packed bitstream of XOR-encoded differences.
pub fn encode_f64(values: &[f64]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    // Estimate: worst case is ~12 bits overhead per value (1 + 1 + 6 + 6 + 64 = 78 bits ≈ 10 bytes).
    // Typical case for smooth data is ~2–3 bits per value.
    let estimated_bytes = 8 + (values.len() * 10);
    let mut writer = BitWriter::with_capacity(estimated_bytes);

    // Write first value raw into the bitstream (64 bits).
    let first_bits = values[0].to_bits();
    writer.write_bits(first_bits, 64);

    if values.len() == 1 {
        return Ok(writer.finish());
    }

    let mut prev_bits = first_bits;
    let mut prev_leading: u8 = 64; // sentinel: forces first XOR through "new window"
    let mut prev_trailing: u8 = 64;
    let mut prev_meaningful_len: u8 = 0;

    for &value in &values[1..] {
        let current_bits = value.to_bits();
        let xor = current_bits ^ prev_bits;

        if xor == 0 {
            // Identical to previous: single 0 bit.
            writer.write_bit(false);
        } else {
            // Different: write 1 bit, then encode the XOR.
            writer.write_bit(true);

            let leading = xor.leading_zeros() as u8;
            let trailing = xor.trailing_zeros() as u8;
            let meaningful_len = 64 - leading - trailing;

            // Can we reuse the previous window?
            if prev_meaningful_len > 0 && leading >= prev_leading && trailing >= prev_trailing {
                // Reuse previous window: control bit 0.
                writer.write_bit(false);
                // Extract bits at the previous window position.
                let shifted = xor >> prev_trailing;
                writer.write_bits(shifted, prev_meaningful_len);
            } else {
                // New window: control bit 1.
                writer.write_bit(true);
                // 6 bits for leading zeros (0–63).
                writer.write_bits(leading as u64, 6);
                // 6 bits for meaningful_len - 1 (0–63 representing 1–64).
                writer.write_bits((meaningful_len - 1) as u64, 6);
                // Write the meaningful bits.
                let shifted = xor >> trailing;
                writer.write_bits(shifted, meaningful_len);

                prev_leading = leading;
                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }
        }

        prev_bits = current_bits;
    }

    Ok(writer.finish())
}

/// Decode a Gorilla XOR encoded f64 stream.
///
/// `expected_count` is the number of f64 values to decode.
pub fn decode_f64(data: &[u8], expected_count: usize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::GorillaDecode("expected_count is 0".into()));
    }

    let mut reader = BitReader::new(data);
    let mut values: Vec<f64> = Vec::with_capacity(expected_count);

    // Read first value raw (64 bits).
    let first_bits = reader.read_bits(64)?;
    values.push(f64::from_bits(first_bits));

    if expected_count == 1 {
        return Ok(values);
    }

    let mut prev_bits = first_bits;
    let mut prev_trailing: u8 = 0;
    let mut prev_meaningful_len: u8 = 64;

    for _ in 1..expected_count {
        let is_nonzero = reader.read_bit()?;

        if !is_nonzero {
            // XOR is 0: same value as previous.
            values.push(f64::from_bits(prev_bits));
        } else {
            let new_window = reader.read_bit()?;

            let xor;
            if !new_window {
                // Reuse previous window.
                let meaningful_bits = reader.read_bits(prev_meaningful_len)?;
                xor = meaningful_bits << prev_trailing;
            } else {
                // New window.
                let leading = reader.read_bits(6)? as u8;
                let meaningful_len = reader.read_bits(6)? as u8 + 1;
                if leading as u16 + meaningful_len as u16 > 64 {
                    return Err(CompressorError::GorillaDecode(format!(
                        "corrupted f64 window: leading({}) + meaningful_len({}) > 64",
                        leading, meaningful_len
                    )));
                }
                let trailing = 64 - leading - meaningful_len;
                let meaningful_bits = reader.read_bits(meaningful_len)?;
                xor = meaningful_bits << trailing;

                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }

            let current_bits = prev_bits ^ xor;
            values.push(f64::from_bits(current_bits));
            prev_bits = current_bits;
        }
    }

    Ok(values)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gorilla XOR — f32
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a slice of f32 values using Gorilla XOR compression.
///
/// Same algorithm as f64 but operates on 32-bit representations.
/// Uses 5 bits for leading zeros (0–31) and 5 bits for meaningful_len - 1 (0–31).
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let estimated_bytes = 4 + (values.len() * 6);
    let mut writer = BitWriter::with_capacity(estimated_bytes);

    // Write first value raw (32 bits).
    let first_bits = values[0].to_bits();
    writer.write_bits_u32(first_bits, 32);

    if values.len() == 1 {
        return Ok(writer.finish());
    }

    let mut prev_bits = first_bits;
    let mut prev_leading: u8 = 32; // sentinel: forces first XOR through "new window"
    let mut prev_trailing: u8 = 32;
    let mut prev_meaningful_len: u8 = 0;

    for &value in &values[1..] {
        let current_bits = value.to_bits();
        let xor = current_bits ^ prev_bits;

        if xor == 0 {
            writer.write_bit(false);
        } else {
            writer.write_bit(true);

            let leading = xor.leading_zeros() as u8;
            let trailing = xor.trailing_zeros() as u8;
            let meaningful_len = 32 - leading - trailing;

            if prev_meaningful_len > 0 && leading >= prev_leading && trailing >= prev_trailing {
                writer.write_bit(false);
                let shifted = xor >> prev_trailing;
                writer.write_bits_u32(shifted, prev_meaningful_len);
            } else {
                writer.write_bit(true);
                // 5 bits for leading zeros (0–31).
                writer.write_bits_u32(leading as u32, 5);
                // 5 bits for meaningful_len - 1 (0–31 representing 1–32).
                writer.write_bits_u32((meaningful_len - 1) as u32, 5);
                // Write meaningful bits.
                let shifted = xor >> trailing;
                writer.write_bits_u32(shifted, meaningful_len);

                prev_leading = leading;
                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }
        }

        prev_bits = current_bits;
    }

    Ok(writer.finish())
}

/// Decode a Gorilla XOR encoded f32 stream.
pub fn decode_f32(data: &[u8], expected_count: usize) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::GorillaDecode("expected_count is 0".into()));
    }

    let mut reader = BitReader::new(data);
    let mut values: Vec<f32> = Vec::with_capacity(expected_count);

    let first_bits = reader.read_bits_u32(32)?;
    values.push(f32::from_bits(first_bits));

    if expected_count == 1 {
        return Ok(values);
    }

    let mut prev_bits = first_bits;
    let mut prev_trailing: u8 = 0;
    let mut prev_meaningful_len: u8 = 32;

    for _ in 1..expected_count {
        let is_nonzero = reader.read_bit()?;

        if !is_nonzero {
            values.push(f32::from_bits(prev_bits));
        } else {
            let new_window = reader.read_bit()?;

            let xor;
            if !new_window {
                let meaningful_bits = reader.read_bits_u32(prev_meaningful_len)?;
                xor = meaningful_bits << prev_trailing;
            } else {
                let leading = reader.read_bits_u32(5)? as u8;
                let meaningful_len = reader.read_bits_u32(5)? as u8 + 1;
                if leading as u16 + meaningful_len as u16 > 32 {
                    return Err(CompressorError::GorillaDecode(format!(
                        "corrupted f32 window: leading({}) + meaningful_len({}) > 32",
                        leading, meaningful_len
                    )));
                }
                let trailing = 32 - leading - meaningful_len;
                let meaningful_bits = reader.read_bits_u32(meaningful_len)?;
                xor = meaningful_bits << trailing;

                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }

            let current_bits = prev_bits ^ xor;
            values.push(f32::from_bits(current_bits));
            prev_bits = current_bits;
        }
    }

    Ok(values)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workspace-Aware Decode — Write LE Bytes Directly
// ═══════════════════════════════════════════════════════════════════════════════

/// Workspace-aware f64 decoder. Writes LE bytes directly to `out` instead of
/// building an intermediate `Vec<f64>`. `out` is cleared and reused.
pub fn decode_f64_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::GorillaDecode("expected_count is 0".into()));
    }

    out.reserve(expected_count * 8);
    let mut reader = BitReader::new(data);

    let first_bits = reader.read_bits(64)?;
    out.extend_from_slice(&f64::from_bits(first_bits).to_le_bytes());

    if expected_count == 1 {
        return Ok(());
    }

    let mut prev_bits = first_bits;
    let mut prev_trailing: u8 = 0;
    let mut prev_meaningful_len: u8 = 64;

    for _ in 1..expected_count {
        let is_nonzero = reader.read_bit()?;

        if !is_nonzero {
            out.extend_from_slice(&f64::from_bits(prev_bits).to_le_bytes());
        } else {
            let new_window = reader.read_bit()?;

            let xor;
            if !new_window {
                let meaningful_bits = reader.read_bits(prev_meaningful_len)?;
                xor = meaningful_bits << prev_trailing;
            } else {
                let leading = reader.read_bits(6)? as u8;
                let meaningful_len = reader.read_bits(6)? as u8 + 1;
                if leading as u16 + meaningful_len as u16 > 64 {
                    return Err(CompressorError::GorillaDecode(format!(
                        "corrupted f64 window: leading({}) + meaningful_len({}) > 64",
                        leading, meaningful_len
                    )));
                }
                let trailing = 64 - leading - meaningful_len;
                let meaningful_bits = reader.read_bits(meaningful_len)?;
                xor = meaningful_bits << trailing;

                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }

            let current_bits = prev_bits ^ xor;
            out.extend_from_slice(&f64::from_bits(current_bits).to_le_bytes());
            prev_bits = current_bits;
        }
    }

    Ok(())
}

/// Workspace-aware f32 decoder. Writes LE bytes directly to `out`.
pub fn decode_f32_into(data: &[u8], expected_count: usize, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    if expected_count == 0 {
        return Err(CompressorError::GorillaDecode("expected_count is 0".into()));
    }

    out.reserve(expected_count * 4);
    let mut reader = BitReader::new(data);

    let first_bits = reader.read_bits_u32(32)?;
    out.extend_from_slice(&f32::from_bits(first_bits).to_le_bytes());

    if expected_count == 1 {
        return Ok(());
    }

    let mut prev_bits = first_bits;
    let mut prev_trailing: u8 = 0;
    let mut prev_meaningful_len: u8 = 32;

    for _ in 1..expected_count {
        let is_nonzero = reader.read_bit()?;

        if !is_nonzero {
            out.extend_from_slice(&f32::from_bits(prev_bits).to_le_bytes());
        } else {
            let new_window = reader.read_bit()?;

            let xor;
            if !new_window {
                let meaningful_bits = reader.read_bits_u32(prev_meaningful_len)?;
                xor = meaningful_bits << prev_trailing;
            } else {
                let leading = reader.read_bits_u32(5)? as u8;
                let meaningful_len = reader.read_bits_u32(5)? as u8 + 1;
                if leading as u16 + meaningful_len as u16 > 32 {
                    return Err(CompressorError::GorillaDecode(format!(
                        "corrupted f32 window: leading({}) + meaningful_len({}) > 32",
                        leading, meaningful_len
                    )));
                }
                let trailing = 32 - leading - meaningful_len;
                let meaningful_bits = reader.read_bits_u32(meaningful_len)?;
                xor = meaningful_bits << trailing;

                prev_trailing = trailing;
                prev_meaningful_len = meaningful_len;
            }

            let current_bits = prev_bits ^ xor;
            out.extend_from_slice(&f32::from_bits(current_bits).to_le_bytes());
            prev_bits = current_bits;
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitWriter / BitReader ──────────────────────────────────────────

    #[test]
    fn bit_writer_single_bits() {
        let mut w = BitWriter::new();
        // Write 10110001 = 0xB1
        for &b in &[true, false, true, true, false, false, false, true] {
            w.write_bit(b);
        }
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xB1]);
    }

    #[test]
    fn bit_writer_partial_byte() {
        let mut w = BitWriter::new();
        // Write 101 → should be padded to 10100000 = 0xA0
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xA0]);
    }

    #[test]
    fn bit_writer_multi_byte() {
        let mut w = BitWriter::new();
        // Write 12 bits: 1111 0000 1010 → 0xF0 and 0xA0 (padded)
        for &b in &[
            true, true, true, true, false, false, false, false, true, false, true, false,
        ] {
            w.write_bit(b);
        }
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xF0, 0xA0]);
    }

    #[test]
    fn bit_writer_write_bits() {
        let mut w = BitWriter::new();
        // Write 0b1010 (4 bits) = 10
        w.write_bits(0b1010, 4);
        // Write 0b1111 (4 bits) = 15
        w.write_bits(0b1111, 4);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b1010_1111]);
    }

    #[test]
    fn bit_reader_single_bits() {
        let data = vec![0xB1]; // 10110001
        let mut r = BitReader::new(&data);
        assert!(r.read_bit().unwrap()); // 1
        assert!(!r.read_bit().unwrap()); // 0
        assert!(r.read_bit().unwrap()); // 1
        assert!(r.read_bit().unwrap()); // 1
        assert!(!r.read_bit().unwrap()); // 0
        assert!(!r.read_bit().unwrap()); // 0
        assert!(!r.read_bit().unwrap()); // 0
        assert!(r.read_bit().unwrap()); // 1
    }

    #[test]
    fn bit_reader_read_bits() {
        let data = vec![0b1010_1111];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b1111);
    }

    #[test]
    fn bit_roundtrip_arbitrary_values() {
        let mut w = BitWriter::new();
        w.write_bits(42, 6); // 101010
        w.write_bits(0, 1); // 0
        w.write_bits(u64::MAX, 64); // all ones
        w.write_bits(1, 1); // 1
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(6).unwrap(), 42);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.read_bits(64).unwrap(), u64::MAX);
        assert_eq!(r.read_bits(1).unwrap(), 1);
    }

    #[test]
    fn bit_reader_underflow() {
        let data = vec![0xFF];
        let mut r = BitReader::new(&data);
        for _ in 0..8 {
            r.read_bit().unwrap();
        }
        // 9th read should fail
        assert!(r.read_bit().is_err());
    }

    // ── Gorilla XOR f64 ───────────────────────────────────────────────

    #[test]
    fn gorilla_f64_single_value() {
        let values = vec![3.14159265];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 1).unwrap();
        assert_eq!(decoded[0].to_bits(), values[0].to_bits());
    }

    #[test]
    fn gorilla_f64_identical_values() {
        // All identical → each subsequent value is 1 bit (the zero-XOR flag).
        let values = vec![22.5; 1000];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 1000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }

        // Size should be: 8 bytes (first value) + ~125 bytes (999 bits, rounded up).
        let expected_max = 8 + (999 / 8) + 2; // some padding
        assert!(
            encoded.len() <= expected_max,
            "Identical values should be ~{} bytes, got {}",
            expected_max,
            encoded.len()
        );
    }

    #[test]
    fn gorilla_f64_slowly_varying_temperature() {
        // Simulate temperature readings: 22.500, 22.501, 22.499, 22.502, ...
        let values: Vec<f64> = (0..10_000)
            .map(|i| 22.5 + 0.001 * (i as f64 * 0.1).sin())
            .collect();

        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 10_000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "bit-exact mismatch at index {}",
                i
            );
        }

        // Check compression ratio.
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        // Gorilla XOR alone achieves ~1.3x on sin-wave data (mantissa still changes
        // significantly). The big win comes when LZ77 operates on the XOR-compressed
        // output. For truly constant or near-constant data, ratios are much higher.
        assert!(
            ratio > 1.2,
            "Expected >1.2x compression on slowly varying temps, got {:.1}x ({} → {} bytes)",
            ratio,
            raw_size,
            encoded.len()
        );
    }

    #[test]
    fn gorilla_f64_monotonic_sensor() {
        // Monotonically increasing pressure readings.
        let values: Vec<f64> = (0..5000).map(|i| 101.325 + i as f64 * 0.0001).collect();

        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 5000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn gorilla_f64_zero_and_negatives() {
        let values = vec![0.0, -0.0, 1.0, -1.0, f64::MIN_POSITIVE, -f64::MIN_POSITIVE];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 6).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn gorilla_f64_special_values() {
        // NaN, infinity — these are valid IEEE 754 and must round-trip.
        let values = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 1.0];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 5).unwrap();

        // NaN is special: NaN != NaN, so compare bits.
        assert!(decoded[0].is_nan());
        assert_eq!(decoded[1], f64::INFINITY);
        assert_eq!(decoded[2], f64::NEG_INFINITY);
        assert_eq!(decoded[3].to_bits(), 0.0f64.to_bits());
        assert_eq!(decoded[4], 1.0);
    }

    #[test]
    fn gorilla_f64_large_jumps() {
        // Worst case: values that are maximally different.
        let values = vec![0.0, f64::MAX, f64::MIN_POSITIVE, -f64::MAX, 1e-300, 1e300];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 6).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn gorilla_f64_two_values() {
        let values = vec![1.0, 2.0];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 2).unwrap();
        assert_eq!(decoded[0].to_bits(), 1.0f64.to_bits());
        assert_eq!(decoded[1].to_bits(), 2.0f64.to_bits());
    }

    #[test]
    fn gorilla_f64_empty_fails() {
        assert!(matches!(encode_f64(&[]), Err(CompressorError::EmptyInput)));
    }

    #[test]
    fn gorilla_f64_decode_zero_count_fails() {
        let encoded = encode_f64(&[1.0]).unwrap();
        assert!(decode_f64(&encoded, 0).is_err());
    }

    // ── Gorilla XOR f32 ───────────────────────────────────────────────

    #[test]
    fn gorilla_f32_single_value() {
        let values = vec![3.14f32];
        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 1).unwrap();
        assert_eq!(decoded[0].to_bits(), values[0].to_bits());
    }

    #[test]
    fn gorilla_f32_identical_values() {
        let values = vec![100.0f32; 500];
        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 500).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn gorilla_f32_slowly_varying() {
        let values: Vec<f32> = (0..5000)
            .map(|i| 25.0 + 0.01 * (i as f32 * 0.05).sin())
            .collect();

        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 5000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }

        let raw_size = values.len() * 4;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 1.5,
            "Expected >1.5x compression on slowly varying f32, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn gorilla_f32_special_values() {
        let values = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0f32, -0.0f32];
        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 5).unwrap();

        assert!(decoded[0].is_nan());
        assert_eq!(decoded[1], f32::INFINITY);
        assert_eq!(decoded[2], f32::NEG_INFINITY);
        assert_eq!(decoded[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(decoded[4].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn gorilla_f32_large_jumps() {
        let values = vec![0.0f32, f32::MAX, f32::MIN_POSITIVE, -f32::MAX];
        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 4).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn gorilla_f32_empty_fails() {
        assert!(matches!(encode_f32(&[]), Err(CompressorError::EmptyInput)));
    }

    // ── Compression Effectiveness ──────────────────────────────────────

    #[test]
    fn gorilla_f64_compression_ratio_constant() {
        let values = vec![42.0f64; 10_000];
        let encoded = encode_f64(&values).unwrap();
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        // Constant series: 8 bytes + ~1250 bytes = ~1258 bytes for 80000 raw.
        assert!(
            ratio > 50.0,
            "Constant f64 series: expected >50x, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn gorilla_f64_compression_ratio_smooth() {
        // Realistic IoT temperature data: very smooth, small variations.
        let values: Vec<f64> = (0..10_000)
            .map(|i| {
                22.5 + 0.001 * (i as f64 * 0.01).sin() + 0.0001 * ((i * 7) as f64 * 0.03).cos()
            })
            .collect();
        let encoded = encode_f64(&values).unwrap();
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 1.2,
            "Smooth f64 temperature: expected >1.2x, got {:.1}x ({} → {} bytes)",
            ratio,
            raw_size,
            encoded.len()
        );
    }

    #[test]
    fn gorilla_f32_compression_ratio_constant() {
        let values = vec![100.0f32; 5000];
        let encoded = encode_f32(&values).unwrap();
        let raw_size = values.len() * 4;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 25.0,
            "Constant f32 series: expected >25x, got {:.1}x",
            ratio
        );
    }

    // ── Window Reuse Logic ─────────────────────────────────────────────

    #[test]
    fn gorilla_f64_window_reuse_pattern() {
        // Create data where the XOR meaningful window stays the same.
        // Incrementing the last few mantissa bits should produce XOR results
        // that fit in the same window repeatedly.
        let base = 100.0f64;
        let base_bits = base.to_bits();
        let values: Vec<f64> = (0..100).map(|i| f64::from_bits(base_bits + i)).collect();

        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 100).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "mismatch at index {}", i);
        }

        // With window reuse, encoding should be very compact (no 6+6 bit headers
        // after the first non-zero XOR).
        let raw_size = values.len() * 8;
        let ratio = raw_size as f64 / encoded.len() as f64;
        assert!(
            ratio > 3.0,
            "Window-reuse pattern: expected >3x, got {:.1}x",
            ratio
        );
    }

    // ── Stress Test ────────────────────────────────────────────────────

    #[test]
    fn gorilla_f64_large_dataset_roundtrip() {
        // 100K values — mix of patterns.
        let mut values: Vec<f64> = Vec::with_capacity(100_000);

        // Block 1: constant
        values.extend(std::iter::repeat(42.0).take(10_000));
        // Block 2: slowly varying
        values.extend((0..30_000).map(|i| 20.0 + 0.001 * (i as f64).sin()));
        // Block 3: monotonic ramp
        values.extend((0..30_000).map(|i| 1000.0 + i as f64 * 0.1));
        // Block 4: random-ish (but deterministic)
        values.extend((0..30_000).map(|i| {
            let x = (i as f64 * 1.61803398875).fract();
            x * 1000.0 - 500.0
        }));

        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded, 100_000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "bit-exact mismatch at index {} (orig={}, dec={})",
                i,
                orig,
                dec
            );
        }
    }

    #[test]
    fn gorilla_f32_large_dataset_roundtrip() {
        let mut values: Vec<f32> = Vec::with_capacity(50_000);
        values.extend(std::iter::repeat(99.9f32).take(5_000));
        values.extend((0..20_000).map(|i| 30.0 + 0.01 * (i as f32).sin()));
        values.extend((0..25_000).map(|i| {
            let x = (i as f32 * 1.41421356).fract();
            x * 100.0
        }));

        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded, 50_000).unwrap();

        for (i, (&orig, &dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.to_bits(), dec.to_bits(), "f32 mismatch at index {}", i);
        }
    }
}
