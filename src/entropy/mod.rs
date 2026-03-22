//! # FSE/ANS Entropy Coding for LZ77 Token Streams
//!
//! This module implements **Finite State Entropy (FSE)**, a table-based variant of
//! **Asymmetric Numeral Systems (ANS)**, to pack LZ77 token streams into near-optimal
//! bitstreams. FSE achieves compression approaching the Shannon entropy limit while
//! maintaining O(1) per-symbol encode/decode — no arithmetic division at runtime.
//!
//! ## Architecture
//!
//! The LZ77 token stream (`&[LzToken]`) contains two kinds of tokens:
//! - **Literals**: raw bytes (alphabet size 256)
//! - **Matches**: (length, offset) pairs
//!
//! We split these into **three independent sub-streams**, each with its own FSE table:
//!
//! 1. **Literal stream** — the raw byte values (0–255)
//! 2. **Match-length stream** — encoded as logarithmic band codes (similar to zstd)
//! 3. **Match-offset stream** — encoded as logarithmic band codes
//!
//! Each sub-stream gets its own frequency distribution and FSE table because their
//! statistical properties are very different (uniform-ish literals vs. geometric-ish
//! lengths and offsets).
//!
//! ## Symbol Coding Scheme
//!
//! Match lengths and offsets use a **logarithmic banding** scheme inspired by zstd:
//!
//! - Small values are coded directly (1:1 mapping to a code)
//! - Larger values are split into a *base code* (encoded by FSE) plus *extra bits*
//!   (written raw to the bitstream)
//!
//! This keeps the FSE alphabet small (dozens of codes, not thousands) while still
//! representing the full u32 range.
//!
//! ## Wire Format
//!
//! ```text
//! [header]
//!   literal_count: u32          — number of literal symbols
//!   match_count: u32            — number of match tokens
//!   literal_table_log: u8       — log2 of literal FSE table size
//!   length_table_log: u8        — log2 of length FSE table size
//!   offset_table_log: u8        — log2 of offset FSE table size
//!   reserved: u8                — padding / future flags
//! [literal frequency table]     — compressed frequency data for literal FSE
//! [length frequency table]      — compressed frequency data for length FSE
//! [offset frequency table]      — compressed frequency data for offset FSE
//! [literal bitstream]           — FSE-encoded literal symbols (reversed)
//! [length bitstream]            — FSE-encoded length codes (reversed)
//! [offset bitstream]            — FSE-encoded offset codes (reversed)
//! [extra bits stream]           — raw extra bits for length/offset bands
//! ```

pub mod fse;

use crate::{CompressorError, LzToken, Result};

// ═══════════════════════════════════════════════════════════════════════════════
// Symbol Coding — Match Length Bands
// ═══════════════════════════════════════════════════════════════════════════════

/// A match-length code: an FSE symbol index plus extra bits to recover the exact length.
///
/// The banding scheme maps match lengths (4..=65535) to codes 0..=MAX_LENGTH_CODE.
/// Small lengths get 1:1 codes (no extra bits). Larger lengths use logarithmic bands
/// where the code identifies the magnitude and extra bits encode the remainder.
///
/// | Length Range | Code | Extra Bits |
/// |-------------|------|------------|
/// | 4           | 0    | 0          |
/// | 5           | 1    | 0          |
/// | ...         | ...  | 0          |
/// | 35          | 31   | 0          |
/// | 36–37       | 32   | 1          |
/// | 38–39       | 33   | 1          |
/// | 40–43       | 34   | 2          |
/// | 44–47       | 35   | 2          |
/// | 48–55       | 36   | 3          |
/// | 56–63       | 37   | 3          |
/// | ...         | ...  | ...        |
/// | (exponential bands continue)     |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthCode {
    /// The FSE symbol to encode.
    pub code: u16,
    /// Number of extra bits (0–16).
    pub extra_bits: u8,
    /// The extra bits value.
    pub extra_value: u16,
}

/// A match-offset code: analogous to LengthCode but for offsets.
///
/// Codes 0–2 are **repcode** slots (repeat offsets 0/1/2, zero extra bits).
/// Codes 3–46 encode real offsets via logarithmic banding, identical to the
/// old codes 0–43 but shifted by `NUM_REPCODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetCode {
    /// The FSE symbol to encode.
    pub code: u16,
    /// Number of extra bits (0–22 for 4MiB window).
    pub extra_bits: u8,
    /// The extra bits value.
    pub extra_value: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Repcodes — Recently-Used Offset Tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of repcode slots (recently-used match offsets).
pub const NUM_REPCODE: usize = 3;

/// Returns true if an offset code represents a repcode (code < NUM_REPCODE).
#[inline]
pub fn is_repcode(code: u16) -> bool {
    (code as usize) < NUM_REPCODE
}

/// Encode a repcode index (0, 1, or 2) as an OffsetCode.
#[inline]
pub fn encode_rep_offset(rep_index: u8) -> OffsetCode {
    debug_assert!((rep_index as usize) < NUM_REPCODE);
    OffsetCode {
        code: rep_index as u16,
        extra_bits: 0,
        extra_value: 0,
    }
}

/// Tracks the 3 most recently used match offsets for repcode encoding.
///
/// Repcode state is maintained identically in encoder and decoder so they
/// stay synchronized. Initial offsets `[1, 4, 8]` cover the most common
/// small distances in structured IoT data.
///
/// Update rules (zstd-standard):
/// - New raw offset: push to front, shift others back
/// - Rep0 match: no change (most recent already at front)
/// - Rep1 match: swap positions 0 and 1
/// - Rep2 match: rotate position 2 to front
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepcodeState {
    pub offsets: [u32; NUM_REPCODE],
}

impl Default for RepcodeState {
    fn default() -> Self {
        Self::new()
    }
}

impl RepcodeState {
    /// Initialize with default offsets.
    #[inline]
    pub fn new() -> Self {
        Self { offsets: [1, 4, 8] }
    }

    /// Check if the given offset matches any repcode. Returns the index (0, 1, or 2).
    #[inline]
    pub fn find(&self, offset: u32) -> Option<u8> {
        if offset == self.offsets[0] {
            return Some(0);
        }
        if offset == self.offsets[1] {
            return Some(1);
        }
        if offset == self.offsets[2] {
            return Some(2);
        }
        None
    }

    /// Update state after a match with a new (non-repcode) offset.
    #[inline]
    pub fn update_raw(&mut self, offset: u32) {
        if offset != self.offsets[0] {
            self.offsets[2] = self.offsets[1];
            self.offsets[1] = self.offsets[0];
            self.offsets[0] = offset;
        }
    }

    /// Update state after a repcode match.
    #[inline]
    pub fn update_rep(&mut self, rep_index: u8) {
        match rep_index {
            0 => {} // rep0: most recent offset already at front
            1 => self.offsets.swap(0, 1),
            2 => {
                let tmp = self.offsets[2];
                self.offsets[2] = self.offsets[1];
                self.offsets[1] = self.offsets[0];
                self.offsets[0] = tmp;
            }
            _ => debug_assert!(false, "invalid repcode index"),
        }
    }
}

// ── Length coding tables ────────────────────────────────────────────────────

/// Base values for each length code. `LENGTH_BASE[c]` is the minimum match length
/// that maps to code `c`.
pub(crate) const LENGTH_BASE: [u32; 52] = [
    // Codes 0–31: direct mapping (length = code + MIN_MATCH_LEN)
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, // Codes 32+: exponential bands
    36, 38, 40, 44, 48, 56, 64, 80, // +1,+1,+2,+2,+3,+3,+4,+4 extra bits
    96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

/// Extra bits for each length code.
pub(crate) const LENGTH_EXTRA: [u8; 52] = [
    // Codes 0–31: no extra bits
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Codes 32+: logarithmic extra bits
    1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    16, // code 51: catch-all for lengths up to 65535
];

/// Number of length codes in our alphabet.
pub const NUM_LENGTH_CODES: usize = LENGTH_BASE.len(); // 52

/// Maximum length code index.
pub const MAX_LENGTH_CODE: u16 = (NUM_LENGTH_CODES - 1) as u16;

/// Encode a match length (MIN_MATCH_LEN..=MAX_MATCH_LEN) into a length code.
pub fn encode_match_length(length: u32) -> LengthCode {
    debug_assert!((crate::MIN_MATCH_LEN..=crate::MAX_MATCH_LEN).contains(&length));

    // Binary search for the right band: find the last code whose base <= length.
    let mut lo = 0usize;
    let mut hi = NUM_LENGTH_CODES;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if LENGTH_BASE[mid] <= length {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let code = lo as u16;
    let extra_bits = LENGTH_EXTRA[lo];
    let extra_value = (length - LENGTH_BASE[lo]) as u16;

    debug_assert!(
        (extra_value as u32) < (1u32 << extra_bits),
        "length {} -> code {}: extra_value {} doesn't fit in {} bits",
        length,
        code,
        extra_value,
        extra_bits
    );

    LengthCode {
        code,
        extra_bits,
        extra_value,
    }
}

/// Decode a length code back to the original match length.
pub fn decode_match_length(code: &LengthCode) -> u32 {
    LENGTH_BASE[code.code as usize] + code.extra_value as u32
}

// ── Offset coding tables ────────────────────────────────────────────────────

/// Base values for each offset code.
///
/// Codes 0–2 are **repcode** sentinels (base 0, 0 extra bits). They are never
/// used by `encode_match_offset`; the encoder emits them explicitly when a
/// match reuses a recent offset. Real offset codes start at index `NUM_REPCODE`
/// (3) and are identical to the old codes 0–43 shifted by 3.
///
/// | Offset Range  | Code | Extra Bits |
/// |--------------|------|------------|
/// | rep0          | 0    | 0          |
/// | rep1          | 1    | 0          |
/// | rep2          | 2    | 0          |
/// | 1             | 3    | 0          |
/// | 2             | 4    | 0          |
/// | 3             | 5    | 0          |
/// | 4–5           | 6    | 1          |
/// | 6–7           | 7    | 1          |
/// | 8–11          | 8    | 2          |
/// | ...           | ...  | ...        |
pub(crate) const OFFSET_BASE: [u32; 47] = [
    0, 0, 0, // codes 0-2: repcode sentinels
    1, 2, 3, // codes 3-5: direct (old 0-2)
    4, 6, // codes 6-7: 1 extra bit (old 3-4)
    8, 12, // codes 8-9: 2 extra bits (old 5-6)
    16, 24, // codes 10-11: 3 extra bits (old 7-8)
    32, 48, // codes 12-13: 4 extra bits (old 9-10)
    64, 96, // codes 14-15: 5 extra bits (old 11-12)
    128, 192, // codes 16-17: 6 extra bits (old 13-14)
    256, 384, // codes 18-19: 7 extra bits (old 15-16)
    512, 768, // codes 20-21: 8 extra bits (old 17-18)
    1024, 1536, // codes 22-23: 9 extra bits (old 19-20)
    2048, 3072, // codes 24-25: 10 extra bits (old 21-22)
    4096, 6144, // codes 26-27: 11 extra bits (old 23-24)
    8192, 12288, // codes 28-29: 12 extra bits (old 25-26)
    16384, 24576, // codes 30-31: 13 extra bits (old 27-28)
    32768, 49152, // codes 32-33: 14 extra bits (old 29-30)
    65536, 98304, // codes 34-35: 15 extra bits (old 31-32)
    131072, 196608, // codes 36-37: 16 extra bits (old 33-34)
    262144, 393216, // codes 38-39: 17 extra bits (old 35-36)
    524288, 786432, // codes 40-41: 18 extra bits (old 37-38)
    1048576, 1572864, // codes 42-43: 19 extra bits (old 39-40)
    2097152, 3145728, // codes 44-45: 20 extra bits (old 41-42)
    4194304, // code 46: catch-all up to MAX_MATCH_OFFSET
];

pub(crate) const OFFSET_EXTRA: [u8; 47] = [
    0, 0, 0, // codes 0-2: repcodes (no extra bits)
    0, 0, 0, // codes 3-5 (old 0-2)
    1, 1, // codes 6-7 (old 3-4)
    2, 2, // codes 8-9 (old 5-6)
    3, 3, // codes 10-11 (old 7-8)
    4, 4, // codes 12-13 (old 9-10)
    5, 5, // codes 14-15 (old 11-12)
    6, 6, // codes 16-17 (old 13-14)
    7, 7, // codes 18-19 (old 15-16)
    8, 8, // codes 20-21 (old 17-18)
    9, 9, // codes 22-23 (old 19-20)
    10, 10, // codes 24-25 (old 21-22)
    11, 11, // codes 26-27 (old 23-24)
    12, 12, // codes 28-29 (old 25-26)
    13, 13, // codes 30-31 (old 27-28)
    14, 14, // codes 32-33 (old 29-30)
    15, 15, // codes 34-35 (old 31-32)
    16, 16, // codes 36-37 (old 33-34)
    17, 17, // codes 38-39 (old 35-36)
    18, 18, // codes 40-41 (old 37-38)
    19, 19, // codes 42-43 (old 39-40)
    20, 20, // codes 44-45 (old 41-42)
    0,  // code 46: exact match for MAX_MATCH_OFFSET
];

/// Number of offset codes in our alphabet (including NUM_REPCODE repcode slots).
pub const NUM_OFFSET_CODES: usize = OFFSET_BASE.len(); // 47

/// Maximum offset code index.
pub const MAX_OFFSET_CODE: u16 = (NUM_OFFSET_CODES - 1) as u16;

/// Number of *real* (non-repcode) offset codes.
pub const NUM_RAW_OFFSET_CODES: usize = NUM_OFFSET_CODES - NUM_REPCODE; // 44

/// Encode a match offset (1..=MAX_MATCH_OFFSET) into an offset code.
///
/// Returns codes in the range `NUM_REPCODE..NUM_OFFSET_CODES` (3..46).
/// Repcode offset codes (0–2) are never returned here — use `encode_rep_offset`.
pub fn encode_match_offset(offset: u32) -> OffsetCode {
    debug_assert!((1..=crate::MAX_MATCH_OFFSET).contains(&offset));

    // Binary search starts at NUM_REPCODE to skip repcode sentinel entries.
    let mut lo = NUM_REPCODE;
    let mut hi = NUM_OFFSET_CODES;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if OFFSET_BASE[mid] <= offset {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let code = lo as u16;
    let extra_bits = OFFSET_EXTRA[lo];
    let extra_value = offset - OFFSET_BASE[lo];

    debug_assert!(
        extra_bits == 0 || extra_value < (1u32 << extra_bits),
        "offset {} -> code {}: extra_value {} doesn't fit in {} bits",
        offset,
        code,
        extra_value,
        extra_bits
    );

    OffsetCode {
        code,
        extra_bits,
        extra_value,
    }
}

/// Decode an offset code back to the original match offset.
///
/// For real offset codes (>= NUM_REPCODE), returns the actual offset.
/// For repcode codes (0–2), returns 0 — caller must resolve via RepcodeState.
pub fn decode_match_offset(code: &OffsetCode) -> u32 {
    OFFSET_BASE[code.code as usize] + code.extra_value
}

// ═══════════════════════════════════════════════════════════════════════════════
// Number of literal codes (full byte alphabet)
// ═══════════════════════════════════════════════════════════════════════════════

/// Literals use the full byte range: 256 symbols.
pub const NUM_LITERAL_CODES: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Cost Model API
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-computed bit-cost tables for the optimal parser.
///
/// The optimal parser (Phase 4) needs to evaluate the cost of encoding each
/// possible parse decision. This struct caches the per-symbol costs so the
/// parser can query them in O(1).
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost in fractional bits for each literal byte (0..=255).
    pub literal_costs: [f32; NUM_LITERAL_CODES],
    /// Cost in fractional bits for each length code (0..=MAX_LENGTH_CODE).
    /// Does NOT include extra bits — caller adds those.
    pub length_code_costs: [f32; NUM_LENGTH_CODES],
    /// Cost in fractional bits for each offset code (0..=MAX_OFFSET_CODE).
    /// Does NOT include extra bits — caller adds those.
    pub offset_code_costs: [f32; NUM_OFFSET_CODES],
}

impl CostModel {
    /// Estimate the total bit cost of encoding a literal byte.
    #[inline]
    pub fn literal_cost(&self, byte: u8) -> f32 {
        self.literal_costs[byte as usize]
    }

    /// Estimate the total bit cost of encoding a match with the given length and offset.
    ///
    /// Includes the FSE symbol cost plus extra bits for both length and offset bands.
    #[inline]
    pub fn match_cost(&self, length: u32, offset: u32) -> f32 {
        let lc = encode_match_length(length);
        let oc = encode_match_offset(offset);

        let length_cost = self.length_code_costs[lc.code as usize] + lc.extra_bits as f32;
        let offset_cost = self.offset_code_costs[oc.code as usize] + oc.extra_bits as f32;

        length_cost + offset_cost
    }

    /// Estimate the total bit cost of encoding a match using a repcode offset.
    ///
    /// Repcode offset codes (0–2) have zero extra bits, so the offset cost is
    /// just the FSE symbol cost for that repcode slot.
    #[inline]
    pub fn rep_match_cost(&self, length: u32, rep_index: u8) -> f32 {
        let lc = encode_match_length(length);
        let length_cost = self.length_code_costs[lc.code as usize] + lc.extra_bits as f32;
        let offset_cost = self.offset_code_costs[rep_index as usize]; // codes 0,1,2 — 0 extra bits
        length_cost + offset_cost
    }

    /// Build a cost model from normalized FSE frequency tables.
    ///
    /// The cost of a symbol with normalized frequency `f` in a table of size `1 << table_log`
    /// is approximately `-log2(f / table_size) = table_log - log2(f)`.
    /// Symbols with frequency 0 get a penalty cost (effectively infinite).
    pub fn from_normalized_frequencies(
        literal_freqs: &[u16],
        literal_table_log: u8,
        length_freqs: &[u16],
        length_table_log: u8,
        offset_freqs: &[u16],
        offset_table_log: u8,
    ) -> Self {
        let mut literal_costs = [0.0f32; NUM_LITERAL_CODES];
        let mut length_code_costs = [0.0f32; NUM_LENGTH_CODES];
        let mut offset_code_costs = [0.0f32; NUM_OFFSET_CODES];

        let penalty = 36.0f32; // extremely expensive — discourages using absent symbols

        for (i, cost) in literal_costs.iter_mut().enumerate() {
            if i < literal_freqs.len() && literal_freqs[i] > 0 {
                *cost = literal_table_log as f32 - (literal_freqs[i] as f32).log2();
            } else {
                *cost = penalty;
            }
        }

        for (i, cost) in length_code_costs.iter_mut().enumerate() {
            if i < length_freqs.len() && length_freqs[i] > 0 {
                *cost = length_table_log as f32 - (length_freqs[i] as f32).log2();
            } else {
                *cost = penalty;
            }
        }

        for (i, cost) in offset_code_costs.iter_mut().enumerate() {
            if i < offset_freqs.len() && offset_freqs[i] > 0 {
                *cost = offset_table_log as f32 - (offset_freqs[i] as f32).log2();
            } else {
                *cost = penalty;
            }
        }

        Self {
            literal_costs,
            length_code_costs,
            offset_code_costs,
        }
    }

    /// Build a flat (uniform) cost model. Used when no statistics are available
    /// (e.g., for the first block or when no FSE tables have been built yet).
    pub fn uniform() -> Self {
        // Uniform distribution: each literal costs 8 bits, each length/offset code
        // costs log2(alphabet_size) bits.
        let lit_cost = 8.0f32;
        let len_cost = (NUM_LENGTH_CODES as f32).log2();
        let off_cost = (NUM_OFFSET_CODES as f32).log2();

        Self {
            literal_costs: [lit_cost; NUM_LITERAL_CODES],
            length_code_costs: [len_cost; NUM_LENGTH_CODES],
            offset_code_costs: [off_cost; NUM_OFFSET_CODES],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public Encode / Decode API
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode an LZ77 token stream into a compressed byte buffer using FSE.
///
/// Returns the compressed buffer and a cost model derived from the actual statistics
/// (useful for subsequent blocks or for the optimal parser to refine).
pub fn encode_tokens(tokens: &[LzToken]) -> Result<(Vec<u8>, CostModel)> {
    if tokens.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    fse::encode_token_stream(tokens)
}

/// Workspace-aware variant of `encode_tokens`. Reuses all scratch buffers.
/// Encoded tokens are returned as `(Vec<u8>, CostModel)`.
pub fn encode_tokens_into(
    tokens: &[LzToken],
    scratch: &mut crate::workspace::EntropyEncodeScratch,
) -> Result<(Vec<u8>, CostModel)> {
    if tokens.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    fse::encode_token_stream_into(tokens, scratch)
}

/// Decode a compressed byte buffer back into an LZ77 token stream.
pub fn decode_tokens(data: &[u8]) -> Result<Vec<LzToken>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    fse::decode_token_stream(data)
}

/// Workspace-aware variant of `decode_tokens`. Uses scratch buffers from
/// `EntropyDecodeScratch` instead of allocating. Decoded tokens are left
/// in `scratch.tokens`.
pub fn decode_tokens_into(
    data: &[u8],
    scratch: &mut crate::workspace::EntropyDecodeScratch,
) -> Result<()> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }
    fse::decode_token_stream_into(data, scratch)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Length coding ──────────────────────────────────────────────────

    #[test]
    fn length_code_direct_range() {
        // Lengths 4..=35 should map to codes 0..=31 with 0 extra bits.
        for len in 4..=35u32 {
            let lc = encode_match_length(len);
            assert_eq!(lc.code, (len - 4) as u16, "length {}", len);
            assert_eq!(lc.extra_bits, 0, "length {}", len);
            assert_eq!(lc.extra_value, 0, "length {}", len);
            assert_eq!(decode_match_length(&lc), len, "roundtrip length {}", len);
        }
    }

    #[test]
    fn length_code_banded_range() {
        // Check a few specific banded values.
        let lc = encode_match_length(36);
        assert_eq!(lc.code, 32);
        assert_eq!(lc.extra_bits, 1);
        assert_eq!(lc.extra_value, 0);
        assert_eq!(decode_match_length(&lc), 36);

        let lc = encode_match_length(37);
        assert_eq!(lc.code, 32);
        assert_eq!(lc.extra_bits, 1);
        assert_eq!(lc.extra_value, 1);
        assert_eq!(decode_match_length(&lc), 37);

        let lc = encode_match_length(38);
        assert_eq!(lc.code, 33);
        assert_eq!(lc.extra_bits, 1);
        assert_eq!(lc.extra_value, 0);
        assert_eq!(decode_match_length(&lc), 38);
    }

    #[test]
    fn length_code_roundtrip_exhaustive_small() {
        // Test all lengths from MIN_MATCH_LEN to 1000.
        for len in crate::MIN_MATCH_LEN..=1000 {
            let lc = encode_match_length(len);
            let decoded = decode_match_length(&lc);
            assert_eq!(decoded, len, "roundtrip failed for length {}", len);
        }
    }

    #[test]
    fn length_code_roundtrip_large() {
        // Test powers of 2 and nearby values up to MAX_MATCH_LEN.
        let test_values = [
            1024, 2048, 4096, 8192, 16384, 32768, 65535, 1025, 2049, 4097, 10000, 50000, 65534,
        ];
        for &len in &test_values {
            let lc = encode_match_length(len);
            let decoded = decode_match_length(&lc);
            assert_eq!(decoded, len, "roundtrip failed for length {}", len);
        }
    }

    #[test]
    fn length_code_max() {
        let lc = encode_match_length(crate::MAX_MATCH_LEN);
        assert_eq!(decode_match_length(&lc), crate::MAX_MATCH_LEN);
    }

    #[test]
    fn length_code_min() {
        let lc = encode_match_length(crate::MIN_MATCH_LEN);
        assert_eq!(lc.code, 0);
        assert_eq!(lc.extra_bits, 0);
        assert_eq!(decode_match_length(&lc), crate::MIN_MATCH_LEN);
    }

    // ── Offset coding ──────────────────────────────────────────────────

    #[test]
    fn offset_code_direct_range() {
        // Offsets 1, 2, 3 now map to codes 3, 4, 5 (shifted by NUM_REPCODE).
        for off in 1..=3u32 {
            let oc = encode_match_offset(off);
            assert_eq!(
                oc.code,
                (off - 1 + NUM_REPCODE as u32) as u16,
                "offset {}",
                off
            );
            assert_eq!(oc.extra_bits, 0, "offset {}", off);
            assert_eq!(decode_match_offset(&oc), off, "roundtrip offset {}", off);
        }
    }

    #[test]
    fn offset_code_banded_range() {
        // Codes shifted by NUM_REPCODE (3) from old values.
        let oc = encode_match_offset(4);
        assert_eq!(oc.code, 6); // old code 3 + 3
        assert_eq!(oc.extra_bits, 1);
        assert_eq!(oc.extra_value, 0);
        assert_eq!(decode_match_offset(&oc), 4);

        let oc = encode_match_offset(5);
        assert_eq!(oc.code, 6); // old code 3 + 3
        assert_eq!(oc.extra_bits, 1);
        assert_eq!(oc.extra_value, 1);
        assert_eq!(decode_match_offset(&oc), 5);

        let oc = encode_match_offset(6);
        assert_eq!(oc.code, 7); // old code 4 + 3
        assert_eq!(oc.extra_bits, 1);
        assert_eq!(oc.extra_value, 0);
        assert_eq!(decode_match_offset(&oc), 6);
    }

    #[test]
    fn offset_code_roundtrip_exhaustive_small() {
        for off in 1..=5000u32 {
            let oc = encode_match_offset(off);
            let decoded = decode_match_offset(&oc);
            assert_eq!(decoded, off, "roundtrip failed for offset {}", off);
        }
    }

    #[test]
    fn offset_code_roundtrip_large() {
        let test_values = [
            1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 1025, 4097, 100000, 1000000,
            3000000, 4194303,
        ];
        for &off in &test_values {
            let oc = encode_match_offset(off);
            let decoded = decode_match_offset(&oc);
            assert_eq!(decoded, off, "roundtrip failed for offset {}", off);
        }
    }

    #[test]
    fn offset_code_max() {
        let oc = encode_match_offset(crate::MAX_MATCH_OFFSET);
        assert_eq!(decode_match_offset(&oc), crate::MAX_MATCH_OFFSET);
    }

    // ── Cost model ──────────────────────────────────────────────────────

    #[test]
    fn cost_model_uniform_reasonable() {
        let cm = CostModel::uniform();
        // All literals should cost 8 bits in uniform model.
        assert!((cm.literal_cost(0) - 8.0).abs() < 0.001);
        assert!((cm.literal_cost(255) - 8.0).abs() < 0.001);
        // Match costs should be positive.
        let mc = cm.match_cost(4, 1);
        assert!(mc > 0.0, "match cost should be positive, got {}", mc);
    }

    #[test]
    fn cost_model_from_frequencies() {
        // Create a simple frequency distribution: symbol 0 is very frequent.
        let mut lit_freqs = vec![0u16; NUM_LITERAL_CODES];
        lit_freqs[0] = 512; // half the table
        lit_freqs[1] = 256;
        lit_freqs[2] = 128;
        lit_freqs[3] = 128;

        let mut len_freqs = vec![0u16; NUM_LENGTH_CODES];
        len_freqs[0] = 512;
        len_freqs[1] = 256;
        len_freqs[2] = 128;
        len_freqs[3] = 128;

        let mut off_freqs = vec![0u16; NUM_OFFSET_CODES];
        off_freqs[0] = 512;
        off_freqs[1] = 256;
        off_freqs[2] = 128;
        off_freqs[3] = 128;

        let cm =
            CostModel::from_normalized_frequencies(&lit_freqs, 10, &len_freqs, 10, &off_freqs, 10);

        // Symbol 0 (freq 512 out of 1024) should cost ~1 bit.
        assert!((cm.literal_cost(0) - 1.0).abs() < 0.01);
        // Symbol 1 (freq 256 out of 1024) should cost ~2 bits.
        assert!((cm.literal_cost(1) - 2.0).abs() < 0.01);
        // Symbol 4 (freq 0) should get the penalty cost.
        assert!(cm.literal_cost(4) > 30.0);
    }

    // ── Table consistency ───────────────────────────────────────────────

    #[test]
    fn length_table_monotonic_bases() {
        for i in 1..LENGTH_BASE.len() {
            assert!(
                LENGTH_BASE[i] > LENGTH_BASE[i - 1],
                "LENGTH_BASE not strictly increasing at index {}: {} <= {}",
                i,
                LENGTH_BASE[i],
                LENGTH_BASE[i - 1]
            );
        }
    }

    #[test]
    fn length_table_coverage() {
        // Verify that the bands cover the full range without gaps.
        for i in 0..LENGTH_BASE.len() - 1 {
            let next_base = LENGTH_BASE[i + 1];
            let this_end = LENGTH_BASE[i] + (1u32 << LENGTH_EXTRA[i]);
            assert_eq!(
                this_end,
                next_base,
                "gap between length codes {} and {}: band ends at {}, next starts at {}",
                i,
                i + 1,
                this_end,
                next_base
            );
        }
    }

    #[test]
    fn offset_table_monotonic_bases() {
        // Skip repcode sentinels (indices 0..NUM_REPCODE). Real offsets start at NUM_REPCODE.
        for i in (NUM_REPCODE + 1)..OFFSET_BASE.len() {
            assert!(
                OFFSET_BASE[i] > OFFSET_BASE[i - 1],
                "OFFSET_BASE not strictly increasing at index {}: {} <= {}",
                i,
                OFFSET_BASE[i],
                OFFSET_BASE[i - 1]
            );
        }
    }

    #[test]
    fn offset_table_coverage() {
        // Verify real offset bands (NUM_REPCODE..) cover without gaps.
        for i in NUM_REPCODE..OFFSET_BASE.len() - 1 {
            let next_base = OFFSET_BASE[i + 1];
            let this_end = OFFSET_BASE[i]
                + if OFFSET_EXTRA[i] == 0 {
                    1
                } else {
                    1u32 << OFFSET_EXTRA[i]
                };
            assert_eq!(
                this_end,
                next_base,
                "gap between offset codes {} and {}: band ends at {}, next starts at {}",
                i,
                i + 1,
                this_end,
                next_base
            );
        }
    }

    #[test]
    fn offset_table_covers_max() {
        // Last band must reach MAX_MATCH_OFFSET.
        let last = OFFSET_BASE.len() - 1;
        assert!(OFFSET_BASE[last] <= crate::MAX_MATCH_OFFSET);
        assert_eq!(
            encode_match_offset(crate::MAX_MATCH_OFFSET).code,
            last as u16
        );
    }

    // ── Repcode state ────────────────────────────────────────────────

    #[test]
    fn repcode_state_find() {
        let state = RepcodeState::new(); // [1, 4, 8]
        assert_eq!(state.find(1), Some(0));
        assert_eq!(state.find(4), Some(1));
        assert_eq!(state.find(8), Some(2));
        assert_eq!(state.find(99), None);
    }

    #[test]
    fn repcode_state_update_raw() {
        let mut state = RepcodeState::new(); // [1, 4, 8]
        state.update_raw(100);
        assert_eq!(state.offsets, [100, 1, 4]);
        state.update_raw(200);
        assert_eq!(state.offsets, [200, 100, 1]);
        // Updating with the same offset as rep0 is a no-op.
        state.update_raw(200);
        assert_eq!(state.offsets, [200, 100, 1]);
    }

    #[test]
    fn repcode_state_update_rep() {
        let mut state = RepcodeState {
            offsets: [10, 20, 30],
        };
        // Rep0: no change.
        state.update_rep(0);
        assert_eq!(state.offsets, [10, 20, 30]);
        // Rep1: swap 0 and 1.
        state.update_rep(1);
        assert_eq!(state.offsets, [20, 10, 30]);
        // Rep2: rotate 2 → front.
        state.update_rep(2);
        assert_eq!(state.offsets, [30, 20, 10]);
    }

    #[test]
    fn repcode_encode_decode_roundtrip() {
        for rep_idx in 0..NUM_REPCODE as u8 {
            let oc = encode_rep_offset(rep_idx);
            assert_eq!(oc.code, rep_idx as u16);
            assert_eq!(oc.extra_bits, 0);
            assert!(is_repcode(oc.code));
        }
        // Real offset codes should NOT be repcodes.
        assert!(!is_repcode(encode_match_offset(1).code));
    }
}
