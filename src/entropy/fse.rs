//! # Finite State Entropy (FSE) — Table-Based ANS Implementation
//!
//! Implements tANS (table-based Asymmetric Numeral Systems) following Yann Collet's
//! design from zstd. O(1) per-symbol encode and decode with no division.
//!
//! ## Encoding (ANS is LIFO)
//!
//! Symbols are encoded in **reverse** order so the decoder reads them forward.
//! We use a two-pass approach:
//! 1. Process symbols in reverse, collecting (nb_bits, bits_value) pairs
//! 2. Write final state first, then bit pairs in forward-decode order
//!
//! This avoids the need for a backward-reading bitstream decoder.
//!
//! ## Bitstream
//!
//! Uses **LSB-first** packing (standard for ANS).

use super::{
    decode_match_length, decode_match_offset, encode_match_length, encode_match_offset,
    encode_rep_offset, is_repcode, CostModel, LengthCode, OffsetCode, RepcodeState, LENGTH_EXTRA,
    NUM_LENGTH_CODES, NUM_LITERAL_CODES, NUM_OFFSET_CODES, OFFSET_EXTRA,
};
use crate::workspace::{
    EntropyDecodeScratch, EntropyEncodeScratch, FseEncodeTableScratch, FseTableScratch,
};
use crate::{CompressorError, LzToken, Result};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_LITERAL_TABLE_LOG: u8 = 11;
const DEFAULT_LENGTH_TABLE_LOG: u8 = 9;
const DEFAULT_OFFSET_TABLE_LOG: u8 = 9;
const MIN_TABLE_LOG: u8 = 5;
const MAX_TABLE_LOG: u8 = 13;

// Header flags for RLE sub-streams.
const FLAG_LITERAL_RLE: u8 = 0x01;
const FLAG_LENGTH_RLE: u8 = 0x02;
const FLAG_OFFSET_RLE: u8 = 0x04;
const FLAG_WIDE_TYPE_BITS: u8 = 0x08; // type_bits_len stored as u32 instead of u16

// ═══════════════════════════════════════════════════════════════════════════════
// LSB-First Bitstream Writer
// ═══════════════════════════════════════════════════════════════════════════════

struct BitWriter {
    buffer: Vec<u8>,
    accum: u64,
    bits_in_accum: u8,
}

impl BitWriter {
    fn with_capacity(byte_cap: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(byte_cap),
            accum: 0,
            bits_in_accum: 0,
        }
    }

    #[inline]
    fn flush_bytes(&mut self) {
        while self.bits_in_accum >= 8 {
            self.buffer.push(self.accum as u8);
            self.accum >>= 8;
            self.bits_in_accum -= 8;
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u64, count: u8) {
        if count == 0 {
            return;
        }
        debug_assert!(count <= 57);
        let masked = if count >= 64 {
            value
        } else {
            value & ((1u64 << count) - 1)
        };
        self.accum |= masked << self.bits_in_accum;
        self.bits_in_accum += count;
        self.flush_bytes();
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.write_bits(v as u64, 8);
    }
    #[inline]
    fn write_u16(&mut self, v: u16) {
        self.write_bits(v as u64, 16);
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write_bits(v as u64, 32);
    }

    fn align_to_byte(&mut self) {
        if self.bits_in_accum % 8 != 0 {
            let pad = 8 - self.bits_in_accum % 8;
            self.bits_in_accum += pad;
            // Zero-padding is implicit — unused bits in accum are already 0.
        }
        self.flush_bytes();
        // Now bits_in_accum == 0 and all bytes are in buffer.
    }

    fn write_aligned_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(
            self.bits_in_accum, 0,
            "must call align_to_byte() before write_aligned_bytes()"
        );
        self.buffer.extend_from_slice(bytes);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_accum > 0 {
            // Push all remaining bits as complete bytes.
            while self.bits_in_accum > 0 {
                self.buffer.push(self.accum as u8);
                self.accum >>= 8;
                self.bits_in_accum = self.bits_in_accum.saturating_sub(8);
            }
        }
        self.buffer
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LSB-First Bitstream Reader
// ═══════════════════════════════════════════════════════════════════════════════

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    accum: u64,
    bits_in_accum: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            byte_pos: 0,
            accum: 0,
            bits_in_accum: 0,
        };
        r.refill();
        r
    }

    #[inline]
    fn refill(&mut self) {
        let available = self.data.len() - self.byte_pos;
        // Fast path: bulk-read up to 7 bytes via a single u64 LE load.
        if self.bits_in_accum <= 56 && available >= 8 {
            // SAFETY: We verified available >= 8, so byte_pos + 8 <= data.len().
            // Using read_unaligned avoids the try_into().unwrap() panic branch
            // that LLVM cannot elide in this hot loop (~millions of calls/sec).
            let chunk = u64::from_le(unsafe {
                self.data
                    .as_ptr()
                    .add(self.byte_pos)
                    .cast::<u64>()
                    .read_unaligned()
            });
            let bytes_to_read = ((64 - self.bits_in_accum) / 8) as usize;
            let mask = if bytes_to_read == 8 {
                u64::MAX
            } else {
                (1u64 << (bytes_to_read * 8)) - 1
            };
            self.accum |= (chunk & mask) << self.bits_in_accum;
            self.bits_in_accum += (bytes_to_read as u8) * 8;
            self.byte_pos += bytes_to_read;
        } else {
            // Slow path for the tail of the buffer.
            while self.bits_in_accum <= 56 && self.byte_pos < self.data.len() {
                self.accum |= (self.data[self.byte_pos] as u64) << self.bits_in_accum;
                self.bits_in_accum += 8;
                self.byte_pos += 1;
            }
        }
    }

    #[inline]
    fn read_bits(&mut self, count: u8) -> Result<u64> {
        if count == 0 {
            return Ok(0);
        }
        debug_assert!(count <= 57);
        if self.bits_in_accum < count {
            self.refill();
            if self.bits_in_accum < count {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: "FSE bitstream underflow".into(),
                });
            }
        }
        let mask = (1u64 << count) - 1;
        let value = self.accum & mask;
        self.accum >>= count;
        self.bits_in_accum -= count;
        Ok(value)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bits(8)? as u8)
    }
    #[inline]
    fn read_u16(&mut self) -> Result<u16> {
        Ok(self.read_bits(16)? as u16)
    }
    #[inline]
    fn read_u32(&mut self) -> Result<u32> {
        Ok(self.read_bits(32)? as u32)
    }

    /// Align to the next byte boundary, discarding any partial-byte bits
    /// and rewinding pre-fetched bytes back to the logical stream position.
    fn align_to_byte(&mut self) {
        let discard = self.bits_in_accum % 8;
        if discard > 0 {
            self.accum >>= discard;
            self.bits_in_accum -= discard;
        }
        // Un-refill: push unconsumed complete bytes back to the byte stream.
        let unconsumed_bytes = (self.bits_in_accum / 8) as usize;
        self.byte_pos -= unconsumed_bytes;
        self.accum = 0;
        self.bits_in_accum = 0;
    }

    /// Zero-copy read of `n` bytes directly from the underlying data.
    /// Must call `align_to_byte()` first.
    fn read_aligned_slice(&mut self, n: usize) -> Result<&'a [u8]> {
        debug_assert_eq!(
            self.bits_in_accum, 0,
            "must call align_to_byte() before read_aligned_slice()"
        );
        if self.byte_pos + n > self.data.len() {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!(
                    "aligned read: need {} bytes at pos {}, have {}",
                    n,
                    self.byte_pos,
                    self.data.len()
                ),
            });
        }
        let slice = &self.data[self.byte_pos..self.byte_pos + n];
        self.byte_pos += n;
        Ok(slice)
    }

    /// Read a u32 LE directly from byte-aligned position.
    fn read_aligned_u32(&mut self) -> Result<u32> {
        let bytes = self.read_aligned_slice(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Frequency Counting & Normalization
// ═══════════════════════════════════════════════════════════════════════════════

fn count_frequencies(symbols: &[u16], alphabet_size: usize) -> Vec<u32> {
    let mut freqs = vec![0u32; alphabet_size];
    for &sym in symbols {
        freqs[sym as usize] += 1;
    }
    freqs
}

/// Workspace-aware frequency counting. Clears and reuses the output Vec.
fn count_frequencies_into(symbols: &[u16], alphabet_size: usize, out: &mut Vec<u32>) {
    out.clear();
    out.resize(alphabet_size, 0u32);
    for &sym in symbols {
        out[sym as usize] += 1;
    }
}

/// Normalize raw frequencies to sum to exactly `1 << table_log`.
/// Every non-zero symbol gets at least frequency 1.
///
/// Uses a two-phase approach:
/// 1. Floor-proportional scaling with minimum-1 guarantee
/// 2. Fix-up: if under-budget add to largest, if over-budget take from largest first
fn normalize_frequencies(raw_freqs: &[u32], table_log: u8) -> Vec<u16> {
    let table_size = 1u32 << table_log;
    let total_count: u64 = raw_freqs.iter().map(|&f| f as u64).sum();
    if total_count == 0 {
        return vec![0u16; raw_freqs.len()];
    }

    let mut normalized = vec![0u16; raw_freqs.len()];
    let mut largest_idx: usize = 0;
    let mut largest_freq: u32 = 0;

    // Phase 1: floor-proportional, guarantee >= 1 for every non-zero symbol.
    for (i, &freq) in raw_freqs.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        if freq > largest_freq {
            largest_freq = freq;
            largest_idx = i;
        }
        let proportional = (freq as u64 * table_size as u64) / total_count;
        normalized[i] = proportional.max(1) as u16;
    }

    // Phase 2: fix-up to hit exactly table_size.
    let sum: u32 = normalized.iter().map(|&n| n as u32).sum();

    if sum < table_size {
        // Under-budget (common with floor): give the rest to the most frequent symbol.
        normalized[largest_idx] += (table_size - sum) as u16;
    } else if sum > table_size {
        // Over-budget: happens when many rare symbols each got the minimum-1 override.
        // Take from symbols with the highest allocations first, never below 1.
        let mut excess = sum - table_size;

        // Try the largest first (fast path).
        let can_take = (normalized[largest_idx] as u32).saturating_sub(1);
        let take = can_take.min(excess);
        normalized[largest_idx] -= take as u16;
        excess -= take;

        if excess > 0 {
            // Still over: build sorted list and take from the next largest.
            let mut indices: Vec<usize> = (0..normalized.len())
                .filter(|&i| i != largest_idx && normalized[i] > 1)
                .collect();
            indices.sort_by(|&a, &b| normalized[b].cmp(&normalized[a]));
            for &idx in &indices {
                if excess == 0 {
                    break;
                }
                let can_take = (normalized[idx] as u32).saturating_sub(1);
                let take = can_take.min(excess);
                normalized[idx] -= take as u16;
                excess -= take;
            }
        }
    }

    debug_assert_eq!(
        normalized.iter().map(|&n| n as u32).sum::<u32>(),
        table_size,
    );
    normalized
}

/// Workspace-aware normalization. Clears and reuses the output Vec.
fn normalize_frequencies_into(raw_freqs: &[u32], table_log: u8, out: &mut Vec<u16>) {
    let table_size = 1u32 << table_log;
    let total_count: u64 = raw_freqs.iter().map(|&f| f as u64).sum();

    out.clear();
    out.resize(raw_freqs.len(), 0u16);

    if total_count == 0 {
        return;
    }

    let mut largest_idx: usize = 0;
    let mut largest_freq: u32 = 0;

    for (i, &freq) in raw_freqs.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        if freq > largest_freq {
            largest_freq = freq;
            largest_idx = i;
        }
        let proportional = (freq as u64 * table_size as u64) / total_count;
        out[i] = proportional.max(1) as u16;
    }

    let sum: u32 = out.iter().map(|&n| n as u32).sum();

    if sum < table_size {
        out[largest_idx] += (table_size - sum) as u16;
    } else if sum > table_size {
        let mut excess = sum - table_size;

        let can_take = (out[largest_idx] as u32).saturating_sub(1);
        let take = can_take.min(excess);
        out[largest_idx] -= take as u16;
        excess -= take;

        if excess > 0 {
            // Rare path: over-budget even after taking from largest.
            // Small temporary allocation here is acceptable (rare path, tiny Vec).
            let mut indices: Vec<usize> = (0..out.len())
                .filter(|&i| i != largest_idx && out[i] > 1)
                .collect();
            indices.sort_by(|&a, &b| out[b].cmp(&out[a]));
            for &idx in &indices {
                if excess == 0 {
                    break;
                }
                let can_take = (out[idx] as u32).saturating_sub(1);
                let take = can_take.min(excess);
                out[idx] -= take as u16;
                excess -= take;
            }
        }
    }

    debug_assert_eq!(out.iter().map(|&n| n as u32).sum::<u32>(), table_size,);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FSE Table — Core Data Structure
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode table entry: state → (symbol, num_bits_to_read, baseline).
/// `next_state = baseline + read_bits(num_bits)`
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeEntry {
    pub(crate) symbol: u16,
    pub(crate) num_bits: u8,
    pub(crate) baseline: u16,
}

/// Per-symbol encoding parameters (from zstd's symbolTT).
#[derive(Debug, Clone, Copy)]
struct SymbolTT {
    /// Packed threshold: `(max_bits_out << 16) - min_state_plus`.
    /// `nb_bits_out = (state + delta_nb_bits) >> 16`
    delta_nb_bits: u32,
    /// Offset into state_table: `new_state = state_table[delta_find_state + renormalized]`
    delta_find_state: i32,
}

/// A complete FSE table for encoding and decoding one symbol stream.
#[derive(Debug, Clone)]
pub struct FseTable {
    table_log: u8,
    table_size: usize,
    norm_freqs: Vec<u16>,
    /// Decode table, indexed by state (0..table_size).
    decode_table: Vec<DecodeEntry>,
    /// Encode state table: for symbol s, the k-th occurrence maps to
    /// `state_table[cum_freq[s] + k]`. Values are spread positions.
    state_table: Vec<u16>,
    /// Per-symbol encoding parameters.
    symbol_tt: Vec<SymbolTT>,
    alphabet_size: usize,
}

/// Floor of log2(x). x must be > 0.
#[inline]
fn highbit(x: u32) -> u8 {
    debug_assert!(x > 0);
    (31 - x.leading_zeros()) as u8
}

impl FseTable {
    /// Create an empty table suitable for use with `rebuild_decode_only`.
    pub fn empty() -> Self {
        Self {
            table_log: 0,
            table_size: 0,
            norm_freqs: Vec::new(),
            decode_table: Vec::new(),
            state_table: Vec::new(),
            symbol_tt: Vec::new(),
            alphabet_size: 0,
        }
    }

    /// Build an FSE table from normalized frequencies.
    #[allow(clippy::needless_range_loop)]
    pub fn from_normalized(norm_freqs: &[u16], table_log: u8) -> Result<Self> {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&table_log) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE table_log {} out of range", table_log),
            });
        }

        let table_size = 1usize << table_log;
        let alphabet_size = norm_freqs.len();

        let freq_sum: u32 = norm_freqs.iter().map(|&f| f as u32).sum();
        if freq_sum != table_size as u32 {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE freq sum {} != table_size {}", freq_sum, table_size),
            });
        }

        // ── Symbol spreading (identical to zstd) ────────────────────
        // Stride coprime to table_size (which is a power of 2).
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;

        let mut spread = vec![0u16; table_size];
        let mut pos = 0usize;
        for (symbol, &freq) in norm_freqs.iter().enumerate() {
            for _ in 0..freq {
                spread[pos] = symbol as u16;
                pos = (pos + step) & mask;
            }
        }

        // ── Build decode table (zstd's FSE_buildDTable approach) ────
        // For each symbol, a counter starts at its frequency and increments.
        let mut symbol_next = vec![0u32; alphabet_size];
        for (s, &f) in norm_freqs.iter().enumerate() {
            symbol_next[s] = f as u32;
        }

        let mut decode_table = vec![
            DecodeEntry {
                symbol: 0,
                num_bits: 0,
                baseline: 0
            };
            table_size
        ];

        for u in 0..table_size {
            let symbol = spread[u] as usize;
            let next_state = symbol_next[symbol];
            symbol_next[symbol] += 1;

            // highbit of next_state gives the magnitude.
            let nb = table_log - highbit(next_state);
            let baseline = (next_state << nb).wrapping_sub(table_size as u32);

            decode_table[u] = DecodeEntry {
                symbol: symbol as u16,
                num_bits: nb,
                baseline: baseline as u16,
            };
        }

        // ── Build cumulative frequencies ────────────────────────────
        let mut cum_freq = vec![0u32; alphabet_size + 1];
        for i in 0..alphabet_size {
            cum_freq[i + 1] = cum_freq[i] + norm_freqs[i] as u32;
        }

        // ── Build encode state table ────────────────────────────────
        // state_table[cum_freq[s] + k] = spread position of k-th occurrence of symbol s
        let mut state_table = vec![0u16; table_size];
        let mut sym_count = vec![0u32; alphabet_size];
        for u in 0..table_size {
            let s = spread[u] as usize;
            let idx = cum_freq[s] + sym_count[s];
            state_table[idx as usize] = u as u16;
            sym_count[s] += 1;
        }

        // ── Build symbolTT (per-symbol encode parameters) ───────────
        let mut symbol_tt = vec![
            SymbolTT {
                delta_nb_bits: 0,
                delta_find_state: 0
            };
            alphabet_size
        ];
        let mut total: u32 = 0;
        for s in 0..alphabet_size {
            let f = norm_freqs[s] as u32;
            if f == 0 {
                continue;
            }

            if f == 1 {
                symbol_tt[s].delta_nb_bits = (table_log as u32) << 16;
                // For freq=1: we subtract (1 << table_log) so that any state
                // in [table_size, 2*table_size) yields nb_bits = table_log.
                // Actually: (state + deltaNbBits) >> 16 for state in [ts, 2ts):
                // state >= ts, deltaNbBits = tl << 16, so (state + tl<<16) >> 16 >= tl.
                // But we want exactly tl bits, not tl+1.
                // zstd uses: deltaNbBits = (tl << 16) - (1 << tl) for freq=1.
                symbol_tt[s].delta_nb_bits =
                    ((table_log as u32) << 16).wrapping_sub(1u32 << table_log);
                symbol_tt[s].delta_find_state = total as i32 - 1;
                total += 1;
            } else {
                let max_bits_out = table_log - highbit(f - 1);
                let min_state_plus = f << max_bits_out;
                symbol_tt[s].delta_nb_bits =
                    ((max_bits_out as u32) << 16).wrapping_sub(min_state_plus);
                symbol_tt[s].delta_find_state = total as i32 - f as i32;
                total += f;
            }
        }

        Ok(Self {
            table_log,
            table_size,
            norm_freqs: norm_freqs.to_vec(),
            decode_table,
            state_table,
            symbol_tt,
            alphabet_size,
        })
    }

    /// Build an FSE table with only the decode path populated.
    ///
    /// Skips construction of encode-only structures (state_table, symbol_tt,
    /// cum_freq). Used by the decompressor where encode capability is never needed.
    pub fn from_normalized_decode_only(norm_freqs: &[u16], table_log: u8) -> Result<Self> {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&table_log) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE table_log {} out of range", table_log),
            });
        }

        let table_size = 1usize << table_log;
        let alphabet_size = norm_freqs.len();

        let freq_sum: u32 = norm_freqs.iter().map(|&f| f as u32).sum();
        if freq_sum != table_size as u32 {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE freq sum {} != table_size {}", freq_sum, table_size),
            });
        }

        // Symbol spreading (identical to full build)
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;

        let mut spread = vec![0u16; table_size];
        let mut pos = 0usize;
        for (symbol, &freq) in norm_freqs.iter().enumerate() {
            for _ in 0..freq {
                spread[pos] = symbol as u16;
                pos = (pos + step) & mask;
            }
        }

        // Build decode table only — skip state_table, symbol_tt, cum_freq
        let mut symbol_next = vec![0u32; alphabet_size];
        for (s, &f) in norm_freqs.iter().enumerate() {
            symbol_next[s] = f as u32;
        }

        let mut decode_table = vec![
            DecodeEntry {
                symbol: 0,
                num_bits: 0,
                baseline: 0
            };
            table_size
        ];

        for u in 0..table_size {
            let symbol = spread[u] as usize;
            let next_state = symbol_next[symbol];
            symbol_next[symbol] += 1;

            let nb = table_log - highbit(next_state);
            let baseline = (next_state << nb).wrapping_sub(table_size as u32);

            decode_table[u] = DecodeEntry {
                symbol: symbol as u16,
                num_bits: nb,
                baseline: baseline as u16,
            };
        }

        Ok(Self {
            table_log,
            table_size,
            norm_freqs: Vec::new(),
            decode_table,
            state_table: Vec::new(),
            symbol_tt: Vec::new(),
            alphabet_size,
        })
    }

    /// Rebuild this table's decode path in-place, reusing scratch buffers.
    ///
    /// Equivalent to `from_normalized_decode_only` but makes zero allocations
    /// after the first call — all internal vecs are cleared and reused via the
    /// `FseTableScratch`.
    pub fn rebuild_decode_only(
        &mut self,
        norm_freqs: &[u16],
        table_log: u8,
        scratch: &mut FseTableScratch,
    ) -> Result<()> {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&table_log) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE table_log {} out of range", table_log),
            });
        }

        let table_size = 1usize << table_log;
        let alphabet_size = norm_freqs.len();

        let freq_sum: u32 = norm_freqs.iter().map(|&f| f as u32).sum();
        if freq_sum != table_size as u32 {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE freq sum {} != table_size {}", freq_sum, table_size),
            });
        }

        // Symbol spreading — reuse scratch.spread
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;

        scratch.spread.clear();
        scratch.spread.resize(table_size, 0u16);
        let mut pos = 0usize;
        for (symbol, &freq) in norm_freqs.iter().enumerate() {
            for _ in 0..freq {
                scratch.spread[pos] = symbol as u16;
                pos = (pos + step) & mask;
            }
        }

        // Build decode table — reuse scratch.symbol_next and self.decode_table
        scratch.symbol_next.clear();
        scratch.symbol_next.resize(alphabet_size, 0u32);
        for (s, &f) in norm_freqs.iter().enumerate() {
            scratch.symbol_next[s] = f as u32;
        }

        self.decode_table.clear();
        self.decode_table.resize(
            table_size,
            DecodeEntry {
                symbol: 0,
                num_bits: 0,
                baseline: 0,
            },
        );

        for u in 0..table_size {
            let symbol = scratch.spread[u] as usize;
            let next_state = scratch.symbol_next[symbol];
            scratch.symbol_next[symbol] += 1;

            let nb = table_log - highbit(next_state);
            let baseline = (next_state << nb).wrapping_sub(table_size as u32);

            self.decode_table[u] = DecodeEntry {
                symbol: symbol as u16,
                num_bits: nb,
                baseline: baseline as u16,
            };
        }

        self.table_log = table_log;
        self.table_size = table_size;
        self.alphabet_size = alphabet_size;

        Ok(())
    }

    /// Rebuild the full table (encode + decode) in-place, reusing all internal Vecs.
    ///
    /// Equivalent to `from_normalized` but makes zero allocations after warmup.
    /// The `FseEncodeTableScratch` holds temporaries; `self`'s own Vecs
    /// (decode_table, state_table, symbol_tt, norm_freqs) are cleared and reused.
    #[allow(clippy::needless_range_loop)]
    pub fn rebuild_full(
        &mut self,
        norm_freqs: &[u16],
        table_log: u8,
        scratch: &mut FseEncodeTableScratch,
    ) -> Result<()> {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&table_log) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE table_log {} out of range", table_log),
            });
        }

        let table_size = 1usize << table_log;
        let alphabet_size = norm_freqs.len();

        let freq_sum: u32 = norm_freqs.iter().map(|&f| f as u32).sum();
        if freq_sum != table_size as u32 {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("FSE freq sum {} != table_size {}", freq_sum, table_size),
            });
        }

        // ── Symbol spreading — reuse scratch.spread ──────────────────
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;

        scratch.spread.clear();
        scratch.spread.resize(table_size, 0u16);
        let mut pos = 0usize;
        for (symbol, &freq) in norm_freqs.iter().enumerate() {
            for _ in 0..freq {
                scratch.spread[pos] = symbol as u16;
                pos = (pos + step) & mask;
            }
        }

        // ── Build decode table — reuse scratch.symbol_next + self.decode_table
        scratch.symbol_next.clear();
        scratch.symbol_next.resize(alphabet_size, 0u32);
        for (s, &f) in norm_freqs.iter().enumerate() {
            scratch.symbol_next[s] = f as u32;
        }

        self.decode_table.clear();
        self.decode_table.resize(
            table_size,
            DecodeEntry {
                symbol: 0,
                num_bits: 0,
                baseline: 0,
            },
        );

        for u in 0..table_size {
            let symbol = scratch.spread[u] as usize;
            let next_state = scratch.symbol_next[symbol];
            scratch.symbol_next[symbol] += 1;

            let nb = table_log - highbit(next_state);
            let baseline = (next_state << nb).wrapping_sub(table_size as u32);

            self.decode_table[u] = DecodeEntry {
                symbol: symbol as u16,
                num_bits: nb,
                baseline: baseline as u16,
            };
        }

        // ── Build cumulative frequencies — reuse scratch.cum_freq ────
        scratch.cum_freq.clear();
        scratch.cum_freq.resize(alphabet_size + 1, 0u32);
        for i in 0..alphabet_size {
            scratch.cum_freq[i + 1] = scratch.cum_freq[i] + norm_freqs[i] as u32;
        }

        // ── Build encode state table — reuse self.state_table + scratch.sym_count
        self.state_table.clear();
        self.state_table.resize(table_size, 0u16);
        scratch.sym_count.clear();
        scratch.sym_count.resize(alphabet_size, 0u32);
        for u in 0..table_size {
            let s = scratch.spread[u] as usize;
            let idx = scratch.cum_freq[s] + scratch.sym_count[s];
            self.state_table[idx as usize] = u as u16;
            scratch.sym_count[s] += 1;
        }

        // ── Build symbolTT — reuse self.symbol_tt ────────────────────
        self.symbol_tt.clear();
        self.symbol_tt.resize(
            alphabet_size,
            SymbolTT {
                delta_nb_bits: 0,
                delta_find_state: 0,
            },
        );
        let mut total: u32 = 0;
        for s in 0..alphabet_size {
            let f = norm_freqs[s] as u32;
            if f == 0 {
                continue;
            }

            if f == 1 {
                self.symbol_tt[s].delta_nb_bits =
                    ((table_log as u32) << 16).wrapping_sub(1u32 << table_log);
                self.symbol_tt[s].delta_find_state = total as i32 - 1;
                total += 1;
            } else {
                let max_bits_out = table_log - highbit(f - 1);
                let min_state_plus = f << max_bits_out;
                self.symbol_tt[s].delta_nb_bits =
                    ((max_bits_out as u32) << 16).wrapping_sub(min_state_plus);
                self.symbol_tt[s].delta_find_state = total as i32 - f as i32;
                total += f;
            }
        }

        // ── Update metadata — reuse self.norm_freqs ──────────────────
        self.norm_freqs.clear();
        self.norm_freqs.extend_from_slice(norm_freqs);
        self.table_log = table_log;
        self.table_size = table_size;
        self.alphabet_size = alphabet_size;

        Ok(())
    }

    /// Decode `count` symbols from a bitstream into a caller-provided Vec.
    ///
    /// Equivalent to `decode()` but writes into `out` instead of allocating.
    /// `out` is cleared and reused.
    pub fn decode_into(&self, data: &[u8], count: usize, out: &mut Vec<u16>) -> Result<()> {
        out.clear();
        if count == 0 {
            return Ok(());
        }
        if data.is_empty() {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: "FSE decode: empty stream".into(),
            });
        }

        out.reserve(count);
        let mut reader = BitReader::new(data);

        // Read initial state.
        let mut state = reader.read_bits(self.table_log)? as usize;

        for _ in 0..count {
            if state >= self.table_size {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!(
                        "FSE decode: state {} >= table_size {}",
                        state, self.table_size
                    ),
                });
            }

            let entry = &self.decode_table[state];
            out.push(entry.symbol);

            // Read bits and compute next state.
            let rest = reader.read_bits(entry.num_bits)? as u16;
            state = (entry.baseline + rest) as usize;
        }

        Ok(())
    }

    /// Encode symbols using a two-pass approach.
    ///
    /// Pass 1: process symbols in reverse, collecting (nb_bits, bit_value) pairs.
    /// Pass 2: write final_state, then pairs in forward-decode order.
    pub fn encode(&self, symbols: &[u16]) -> Result<Vec<u8>> {
        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        let ts = self.table_size as u32;

        // Validate all symbols are in the table.
        for &sym in symbols {
            let s = sym as usize;
            if s >= self.alphabet_size || self.norm_freqs[s] == 0 {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!("FSE encode: symbol {} not in table", sym),
                });
            }
        }

        // Pass 1: encode in reverse, collect bit emissions.
        let mut state: u32 = ts; // initial state = table_size
        let mut emissions: Vec<(u8, u64)> = Vec::with_capacity(symbols.len());

        for &sym in symbols.iter().rev() {
            let s = sym as usize;
            let tt = &self.symbol_tt[s];

            // Compute number of bits to emit.
            let nb_bits_out = ((state + tt.delta_nb_bits) >> 16) as u8;

            // Emit low nb_bits_out bits of state.
            let bits_value = if nb_bits_out == 0 {
                0u64
            } else {
                state as u64 & ((1u64 << nb_bits_out) - 1)
            };
            emissions.push((nb_bits_out, bits_value));

            // Renormalize: shift state right.
            state >>= nb_bits_out;

            // Transition: look up new state from state_table.
            let idx = (tt.delta_find_state + state as i32) as usize;
            if idx >= self.state_table.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!(
                        "FSE encode: state_table index {} out of range (len {}), sym={}, state={}, dfs={}",
                        idx, self.state_table.len(), s, state, tt.delta_find_state
                    ),
                });
            }
            state = self.state_table[idx] as u32 + ts;
        }

        // Pass 2: write to bitstream in decoder-friendly order.
        let mut writer = BitWriter::with_capacity(symbols.len() + 16);

        // Write final state (the decoder's initial state).
        // State is in [table_size, 2*table_size), write the low table_log bits.
        writer.write_bits((state - ts) as u64, self.table_log);

        // Write bit emissions in REVERSE of the order we collected them.
        // We collected: [emission for sym[N-1], sym[N-2], ..., sym[0]]
        // We reverse to: [emission for sym[0], sym[1], ..., sym[N-1]]
        // So the decoder reads bits for sym[0] first, then sym[1], etc.
        for &(nb, val) in emissions.iter().rev() {
            writer.write_bits(val, nb);
        }

        Ok(writer.finish())
    }

    /// Workspace-aware encode. Reuses the `emissions` buffer from the caller.
    /// Returns the encoded bitstream bytes (irreducible allocation per sub-stream).
    pub fn encode_into(&self, symbols: &[u16], emissions: &mut Vec<(u8, u64)>) -> Result<Vec<u8>> {
        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        let ts = self.table_size as u32;

        for &sym in symbols {
            let s = sym as usize;
            if s >= self.alphabet_size || self.norm_freqs[s] == 0 {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!("FSE encode: symbol {} not in table", sym),
                });
            }
        }

        // Pass 1: encode in reverse, collect bit emissions.
        let mut state: u32 = ts;
        emissions.clear();
        emissions.reserve(symbols.len());

        for &sym in symbols.iter().rev() {
            let s = sym as usize;
            let tt = &self.symbol_tt[s];

            let nb_bits_out = ((state + tt.delta_nb_bits) >> 16) as u8;
            let bits_value = if nb_bits_out == 0 {
                0u64
            } else {
                state as u64 & ((1u64 << nb_bits_out) - 1)
            };
            emissions.push((nb_bits_out, bits_value));

            state >>= nb_bits_out;

            let idx = (tt.delta_find_state + state as i32) as usize;
            if idx >= self.state_table.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!(
                        "FSE encode: state_table index {} out of range (len {}), sym={}, state={}, dfs={}",
                        idx, self.state_table.len(), s, state, tt.delta_find_state
                    ),
                });
            }
            state = self.state_table[idx] as u32 + ts;
        }

        // Pass 2: write to bitstream.
        let mut writer = BitWriter::with_capacity(symbols.len() + 16);
        writer.write_bits((state - ts) as u64, self.table_log);

        for &(nb, val) in emissions.iter().rev() {
            writer.write_bits(val, nb);
        }

        Ok(writer.finish())
    }

    /// Decode `count` symbols from a bitstream.
    pub fn decode(&self, data: &[u8], count: usize) -> Result<Vec<u16>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if data.is_empty() {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: "FSE decode: empty stream".into(),
            });
        }

        let mut reader = BitReader::new(data);
        let mut symbols = Vec::with_capacity(count);

        // Read initial state.
        let mut state = reader.read_bits(self.table_log)? as usize;

        for _ in 0..count {
            if state >= self.table_size {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!(
                        "FSE decode: state {} >= table_size {}",
                        state, self.table_size
                    ),
                });
            }

            let entry = &self.decode_table[state];
            symbols.push(entry.symbol);

            // Read bits and compute next state.
            let rest = reader.read_bits(entry.num_bits)? as u16;
            state = (entry.baseline + rest) as usize;
        }

        Ok(symbols)
    }

    /// Bit cost of encoding a symbol with this table.
    pub fn symbol_cost(&self, symbol: usize) -> f32 {
        if symbol >= self.norm_freqs.len() || self.norm_freqs[symbol] == 0 {
            return 36.0;
        }
        self.table_log as f32 - (self.norm_freqs[symbol] as f32).log2()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Frequency Table Serialization
// ═══════════════════════════════════════════════════════════════════════════════

fn serialize_freq_table(freqs: &[u16], table_log: u8, writer: &mut BitWriter) {
    let max_symbol = freqs
        .iter()
        .rposition(|&f| f > 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    writer.write_u16(max_symbol as u16);
    for &freq in &freqs[..max_symbol] {
        if freq == 0 {
            writer.write_bits(0, 1);
        } else {
            writer.write_bits(1, 1);
            writer.write_bits((freq - 1) as u64, table_log);
        }
    }
}

fn deserialize_freq_table(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
    table_log: u8,
) -> Result<Vec<u16>> {
    let max_symbol = reader.read_u16()? as usize;
    if max_symbol > alphabet_size {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "freq table: max_symbol {} > alphabet {}",
                max_symbol, alphabet_size
            ),
        });
    }

    let mut freqs = vec![0u16; alphabet_size];
    let table_size = 1u32 << table_log;
    let mut total: u32 = 0;

    for freq in freqs[..max_symbol].iter_mut() {
        let present = reader.read_bits(1)? as u8;
        if present == 1 {
            let f = reader.read_bits(table_log)? as u16 + 1;
            *freq = f;
            total += f as u32;
        }
    }

    if total != table_size {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!("freq table: sum {} != table_size {}", total, table_size),
        });
    }

    Ok(freqs)
}

/// Workspace-aware variant of `deserialize_freq_table`. Clears and reuses
/// the provided `out` Vec instead of allocating a new one.
fn deserialize_freq_table_into(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
    table_log: u8,
    out: &mut Vec<u16>,
) -> Result<()> {
    let max_symbol = reader.read_u16()? as usize;
    if max_symbol > alphabet_size {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "freq table: max_symbol {} > alphabet {}",
                max_symbol, alphabet_size
            ),
        });
    }

    out.clear();
    out.resize(alphabet_size, 0u16);
    let table_size = 1u32 << table_log;
    let mut total: u32 = 0;

    for freq in out[..max_symbol].iter_mut() {
        let present = reader.read_bits(1)? as u8;
        if present == 1 {
            let f = reader.read_bits(table_log)? as u16 + 1;
            *freq = f;
            total += f as u32;
        }
    }

    if total != table_size {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!("freq table: sum {} != table_size {}", total, table_size),
        });
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Table Log Selection
// ═══════════════════════════════════════════════════════════════════════════════

fn choose_table_log(symbols: &[u16], default_log: u8, alphabet_size: usize) -> u8 {
    if symbols.is_empty() {
        return MIN_TABLE_LOG;
    }

    let mut seen = vec![false; alphabet_size];
    let mut distinct = 0usize;
    for &s in symbols {
        let idx = s as usize;
        if idx < alphabet_size && !seen[idx] {
            seen[idx] = true;
            distinct += 1;
        }
    }

    let mut log = default_log;

    // Don't use table bigger than 2x symbol count.
    while (1usize << log) > symbols.len() * 2 && log > MIN_TABLE_LOG {
        log -= 1;
    }

    // Ensure enough slots for all distinct symbols.
    while (1usize << log) < distinct && log < MAX_TABLE_LOG {
        log += 1;
    }

    log.clamp(MIN_TABLE_LOG, MAX_TABLE_LOG)
}

// ═══════════════════════════════════════════════════════════════════════════════
// RLE Detection
// ═══════════════════════════════════════════════════════════════════════════════

fn is_single_symbol(symbols: &[u16]) -> Option<u16> {
    if symbols.is_empty() {
        return None;
    }
    let first = symbols[0];
    if symbols.iter().all(|&s| s == first) {
        Some(first)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-Level LzToken Encode / Decode
// ═══════════════════════════════════════════════════════════════════════════════
//
// Wire format:
//   header (12 bytes):
//     literal_count: u32
//     match_count: u32
//     lit_tl: u8, len_tl: u8, off_tl: u8, flags: u8
//   token_type_bits: ceil(total_tokens / 8) bytes — 0=literal, 1=match
//   [frequency tables — skipped for RLE streams]
//   [FSE compressed streams with u32 byte-length prefix — skipped for RLE]
//   [extra bits: length extras then offset extras]

/// Encode an LZ77 token stream into a compressed byte buffer.
pub fn encode_token_stream(tokens: &[LzToken]) -> Result<(Vec<u8>, CostModel)> {
    if tokens.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    // ── Split into sub-streams, preserving order via type bits ───────
    let total_tokens = tokens.len();
    let mut type_bits: Vec<u8> = Vec::with_capacity(total_tokens.div_ceil(8));
    let mut literal_symbols: Vec<u16> = Vec::new();
    let mut length_symbols: Vec<u16> = Vec::new();
    let mut offset_symbols: Vec<u16> = Vec::new();
    let mut length_extras: Vec<(u8, u16)> = Vec::new();
    let mut offset_extras: Vec<(u8, u32)> = Vec::new();

    let mut literal_count: u32 = 0;
    let mut match_count: u32 = 0;
    let mut repcode_state = RepcodeState::new();

    // Pack token types into bytes (8 tokens per byte, LSB first).
    let mut current_byte: u8 = 0;
    let mut bit_idx: u8 = 0;

    for token in tokens {
        match *token {
            LzToken::Literal(byte) => {
                // bit = 0 (already zero)
                literal_symbols.push(byte as u16);
                literal_count += 1;
            }
            LzToken::Match { offset, length } => {
                current_byte |= 1 << bit_idx;
                let lc = encode_match_length(length);

                // Check if offset matches a repcode — encode as repcode if so.
                let oc = if let Some(rep_idx) = repcode_state.find(offset) {
                    repcode_state.update_rep(rep_idx);
                    encode_rep_offset(rep_idx)
                } else {
                    repcode_state.update_raw(offset);
                    encode_match_offset(offset)
                };

                length_symbols.push(lc.code);
                offset_symbols.push(oc.code);
                length_extras.push((lc.extra_bits, lc.extra_value));
                offset_extras.push((oc.extra_bits, oc.extra_value));
                match_count += 1;
            }
        }
        bit_idx += 1;
        if bit_idx == 8 {
            type_bits.push(current_byte);
            current_byte = 0;
            bit_idx = 0;
        }
    }
    if bit_idx > 0 {
        type_bits.push(current_byte);
    }

    // ── Choose table logs ───────────────────────────────────────────
    let lit_tl = if literal_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(
            &literal_symbols,
            DEFAULT_LITERAL_TABLE_LOG,
            NUM_LITERAL_CODES,
        )
    };
    let len_tl = if length_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(&length_symbols, DEFAULT_LENGTH_TABLE_LOG, NUM_LENGTH_CODES)
    };
    let off_tl = if offset_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(&offset_symbols, DEFAULT_OFFSET_TABLE_LOG, NUM_OFFSET_CODES)
    };

    // ── Normalize frequencies ───────────────────────────────────────
    let lit_raw = count_frequencies(&literal_symbols, NUM_LITERAL_CODES);
    let len_raw = count_frequencies(&length_symbols, NUM_LENGTH_CODES);
    let off_raw = count_frequencies(&offset_symbols, NUM_OFFSET_CODES);

    let lit_norm = if literal_symbols.is_empty() {
        vec![0u16; NUM_LITERAL_CODES]
    } else {
        normalize_frequencies(&lit_raw, lit_tl)
    };
    let len_norm = if length_symbols.is_empty() {
        vec![0u16; NUM_LENGTH_CODES]
    } else {
        normalize_frequencies(&len_raw, len_tl)
    };
    let off_norm = if offset_symbols.is_empty() {
        vec![0u16; NUM_OFFSET_CODES]
    } else {
        normalize_frequencies(&off_raw, off_tl)
    };

    // ── Cost model ──────────────────────────────────────────────────
    let cost_model = if literal_symbols.is_empty() && length_symbols.is_empty() {
        CostModel::uniform()
    } else {
        CostModel::from_normalized_frequencies(
            &lit_norm, lit_tl, &len_norm, len_tl, &off_norm, off_tl,
        )
    };

    // ── RLE detection ───────────────────────────────────────────────
    let lit_rle = is_single_symbol(&literal_symbols);
    let len_rle = is_single_symbol(&length_symbols);
    let off_rle = is_single_symbol(&offset_symbols);

    let mut flags: u8 = 0;
    if lit_rle.is_some() {
        flags |= FLAG_LITERAL_RLE;
    }
    if len_rle.is_some() {
        flags |= FLAG_LENGTH_RLE;
    }
    if off_rle.is_some() {
        flags |= FLAG_OFFSET_RLE;
    }
    if type_bits.len() > u16::MAX as usize {
        flags |= FLAG_WIDE_TYPE_BITS;
    }

    // ── Write output ────────────────────────────────────────────────
    let est = 64 + type_bits.len() + literal_symbols.len() + length_symbols.len() * 4;
    let mut writer = BitWriter::with_capacity(est);

    // Header.
    writer.write_u32(literal_count);
    writer.write_u32(match_count);
    writer.write_u8(lit_tl);
    writer.write_u8(len_tl);
    writer.write_u8(off_tl);
    writer.write_u8(flags);

    // Token type bits (raw bytes, not bit-packed through the writer — already packed).
    if flags & FLAG_WIDE_TYPE_BITS != 0 {
        writer.write_u32(type_bits.len() as u32);
    } else {
        writer.write_u16(type_bits.len() as u16);
    }
    for &b in &type_bits {
        writer.write_u8(b);
    }

    // Frequency tables (skip for RLE — just store the symbol).
    if let Some(sym) = lit_rle {
        writer.write_u16(sym);
    } else if !literal_symbols.is_empty() {
        serialize_freq_table(&lit_norm, lit_tl, &mut writer);
    }

    if let Some(sym) = len_rle {
        writer.write_u16(sym);
    } else if !length_symbols.is_empty() {
        serialize_freq_table(&len_norm, len_tl, &mut writer);
    }

    if let Some(sym) = off_rle {
        writer.write_u16(sym);
    } else if !offset_symbols.is_empty() {
        serialize_freq_table(&off_norm, off_tl, &mut writer);
    }

    // Align to byte boundary before sub-stream data (enables zero-copy decode).
    writer.align_to_byte();

    // FSE compressed sub-streams (byte-aligned, with u32 LE length prefix).
    if lit_rle.is_none() && !literal_symbols.is_empty() {
        let table = FseTable::from_normalized(&lit_norm, lit_tl)?;
        let compressed = table.encode(&literal_symbols)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    if len_rle.is_none() && !length_symbols.is_empty() {
        let table = FseTable::from_normalized(&len_norm, len_tl)?;
        let compressed = table.encode(&length_symbols)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    if off_rle.is_none() && !offset_symbols.is_empty() {
        let table = FseTable::from_normalized(&off_norm, off_tl)?;
        let compressed = table.encode(&offset_symbols)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    // Extra bits (raw).
    for &(extra_bits, extra_value) in &length_extras {
        if extra_bits > 0 {
            writer.write_bits(extra_value as u64, extra_bits);
        }
    }
    for &(extra_bits, extra_value) in &offset_extras {
        if extra_bits > 0 {
            writer.write_bits(extra_value as u64, extra_bits);
        }
    }

    Ok((writer.finish(), cost_model))
}

/// Workspace-aware entropy encode. Reuses all scratch Vecs from `EntropyEncodeScratch`.
/// Returns the encoded payload and cost model. The payload `Vec<u8>` is the only
/// irreducible allocation per call.
pub fn encode_token_stream_into(
    tokens: &[LzToken],
    scratch: &mut EntropyEncodeScratch,
) -> Result<(Vec<u8>, CostModel)> {
    if tokens.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    // ── Split into sub-streams — reuse scratch Vecs ──────────────────
    let total_tokens = tokens.len();
    scratch.type_bits.clear();
    scratch.type_bits.reserve(total_tokens.div_ceil(8));
    scratch.literal_symbols.clear();
    scratch.length_symbols.clear();
    scratch.offset_symbols.clear();
    scratch.length_extras.clear();
    scratch.offset_extras.clear();

    let mut literal_count: u32 = 0;
    let mut match_count: u32 = 0;
    let mut repcode_state = RepcodeState::new();

    let mut current_byte: u8 = 0;
    let mut bit_idx: u8 = 0;

    for token in tokens {
        match *token {
            LzToken::Literal(byte) => {
                scratch.literal_symbols.push(byte as u16);
                literal_count += 1;
            }
            LzToken::Match { offset, length } => {
                current_byte |= 1 << bit_idx;
                let lc = encode_match_length(length);

                let oc = if let Some(rep_idx) = repcode_state.find(offset) {
                    repcode_state.update_rep(rep_idx);
                    encode_rep_offset(rep_idx)
                } else {
                    repcode_state.update_raw(offset);
                    encode_match_offset(offset)
                };

                scratch.length_symbols.push(lc.code);
                scratch.offset_symbols.push(oc.code);
                scratch.length_extras.push((lc.extra_bits, lc.extra_value));
                scratch.offset_extras.push((oc.extra_bits, oc.extra_value));
                match_count += 1;
            }
        }
        bit_idx += 1;
        if bit_idx == 8 {
            scratch.type_bits.push(current_byte);
            current_byte = 0;
            bit_idx = 0;
        }
    }
    if bit_idx > 0 {
        scratch.type_bits.push(current_byte);
    }

    // ── Choose table logs ────────────────────────────────────────────
    let lit_tl = if scratch.literal_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(
            &scratch.literal_symbols,
            DEFAULT_LITERAL_TABLE_LOG,
            NUM_LITERAL_CODES,
        )
    };
    let len_tl = if scratch.length_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(
            &scratch.length_symbols,
            DEFAULT_LENGTH_TABLE_LOG,
            NUM_LENGTH_CODES,
        )
    };
    let off_tl = if scratch.offset_symbols.is_empty() {
        MIN_TABLE_LOG
    } else {
        choose_table_log(
            &scratch.offset_symbols,
            DEFAULT_OFFSET_TABLE_LOG,
            NUM_OFFSET_CODES,
        )
    };

    // ── Normalize frequencies — reuse scratch Vecs ───────────────────
    count_frequencies_into(
        &scratch.literal_symbols,
        NUM_LITERAL_CODES,
        &mut scratch.lit_raw_freqs,
    );
    count_frequencies_into(
        &scratch.length_symbols,
        NUM_LENGTH_CODES,
        &mut scratch.len_raw_freqs,
    );
    count_frequencies_into(
        &scratch.offset_symbols,
        NUM_OFFSET_CODES,
        &mut scratch.off_raw_freqs,
    );

    if scratch.literal_symbols.is_empty() {
        scratch.lit_norm_freqs.clear();
        scratch.lit_norm_freqs.resize(NUM_LITERAL_CODES, 0u16);
    } else {
        normalize_frequencies_into(&scratch.lit_raw_freqs, lit_tl, &mut scratch.lit_norm_freqs);
    }
    if scratch.length_symbols.is_empty() {
        scratch.len_norm_freqs.clear();
        scratch.len_norm_freqs.resize(NUM_LENGTH_CODES, 0u16);
    } else {
        normalize_frequencies_into(&scratch.len_raw_freqs, len_tl, &mut scratch.len_norm_freqs);
    }
    if scratch.offset_symbols.is_empty() {
        scratch.off_norm_freqs.clear();
        scratch.off_norm_freqs.resize(NUM_OFFSET_CODES, 0u16);
    } else {
        normalize_frequencies_into(&scratch.off_raw_freqs, off_tl, &mut scratch.off_norm_freqs);
    }

    // ── Cost model ───────────────────────────────────────────────────
    let cost_model = if scratch.literal_symbols.is_empty() && scratch.length_symbols.is_empty() {
        CostModel::uniform()
    } else {
        CostModel::from_normalized_frequencies(
            &scratch.lit_norm_freqs,
            lit_tl,
            &scratch.len_norm_freqs,
            len_tl,
            &scratch.off_norm_freqs,
            off_tl,
        )
    };

    // ── RLE detection ────────────────────────────────────────────────
    let lit_rle = is_single_symbol(&scratch.literal_symbols);
    let len_rle = is_single_symbol(&scratch.length_symbols);
    let off_rle = is_single_symbol(&scratch.offset_symbols);

    let mut flags: u8 = 0;
    if lit_rle.is_some() {
        flags |= FLAG_LITERAL_RLE;
    }
    if len_rle.is_some() {
        flags |= FLAG_LENGTH_RLE;
    }
    if off_rle.is_some() {
        flags |= FLAG_OFFSET_RLE;
    }
    if scratch.type_bits.len() > u16::MAX as usize {
        flags |= FLAG_WIDE_TYPE_BITS;
    }

    // ── Write output ─────────────────────────────────────────────────
    let est = 64
        + scratch.type_bits.len()
        + scratch.literal_symbols.len()
        + scratch.length_symbols.len() * 4;
    let mut writer = BitWriter::with_capacity(est);

    writer.write_u32(literal_count);
    writer.write_u32(match_count);
    writer.write_u8(lit_tl);
    writer.write_u8(len_tl);
    writer.write_u8(off_tl);
    writer.write_u8(flags);

    if flags & FLAG_WIDE_TYPE_BITS != 0 {
        writer.write_u32(scratch.type_bits.len() as u32);
    } else {
        writer.write_u16(scratch.type_bits.len() as u16);
    }
    for &b in &scratch.type_bits {
        writer.write_u8(b);
    }

    // Frequency tables
    if let Some(sym) = lit_rle {
        writer.write_u16(sym);
    } else if !scratch.literal_symbols.is_empty() {
        serialize_freq_table(&scratch.lit_norm_freqs, lit_tl, &mut writer);
    }

    if let Some(sym) = len_rle {
        writer.write_u16(sym);
    } else if !scratch.length_symbols.is_empty() {
        serialize_freq_table(&scratch.len_norm_freqs, len_tl, &mut writer);
    }

    if let Some(sym) = off_rle {
        writer.write_u16(sym);
    } else if !scratch.offset_symbols.is_empty() {
        serialize_freq_table(&scratch.off_norm_freqs, off_tl, &mut writer);
    }

    writer.align_to_byte();

    // FSE compressed sub-streams — rebuild single table 3×
    if lit_rle.is_none() && !scratch.literal_symbols.is_empty() {
        scratch.fse_table.rebuild_full(
            &scratch.lit_norm_freqs,
            lit_tl,
            &mut scratch.table_scratch,
        )?;
        let compressed = scratch
            .fse_table
            .encode_into(&scratch.literal_symbols, &mut scratch.emissions)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    if len_rle.is_none() && !scratch.length_symbols.is_empty() {
        scratch.fse_table.rebuild_full(
            &scratch.len_norm_freqs,
            len_tl,
            &mut scratch.table_scratch,
        )?;
        let compressed = scratch
            .fse_table
            .encode_into(&scratch.length_symbols, &mut scratch.emissions)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    if off_rle.is_none() && !scratch.offset_symbols.is_empty() {
        scratch.fse_table.rebuild_full(
            &scratch.off_norm_freqs,
            off_tl,
            &mut scratch.table_scratch,
        )?;
        let compressed = scratch
            .fse_table
            .encode_into(&scratch.offset_symbols, &mut scratch.emissions)?;
        writer.write_aligned_bytes(&(compressed.len() as u32).to_le_bytes());
        writer.write_aligned_bytes(&compressed);
    }

    // Extra bits
    for &(extra_bits, extra_value) in &scratch.length_extras {
        if extra_bits > 0 {
            writer.write_bits(extra_value as u64, extra_bits);
        }
    }
    for &(extra_bits, extra_value) in &scratch.offset_extras {
        if extra_bits > 0 {
            writer.write_bits(extra_value as u64, extra_bits);
        }
    }

    Ok((writer.finish(), cost_model))
}

/// Decode a compressed byte buffer back into an LZ77 token stream.
pub fn decode_token_stream(data: &[u8]) -> Result<Vec<LzToken>> {
    let mut reader = BitReader::new(data);

    // ── Header ──────────────────────────────────────────────────────
    let literal_count = reader.read_u32()? as usize;
    let match_count = reader.read_u32()? as usize;
    let lit_tl = reader.read_u8()?;
    let len_tl = reader.read_u8()?;
    let off_tl = reader.read_u8()?;
    let flags = reader.read_u8()?;

    // Validate table_log values before they reach 1u32 << table_log in deserialize_freq_table.
    for &(name, tl) in &[("literal", lit_tl), ("length", len_tl), ("offset", off_tl)] {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&tl) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!(
                    "FSE header: {} table_log {} out of range [{}, {}]",
                    name, tl, MIN_TABLE_LOG, MAX_TABLE_LOG
                ),
            });
        }
    }

    // Validate token counts are plausible relative to compressed data size.
    // Each compressed byte can expand to at most ~256 tokens in any realistic scheme.
    let max_tokens = data.len().saturating_mul(256);
    if literal_count > max_tokens || match_count > max_tokens {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "FSE header: token counts ({} lit, {} match) implausible for {} bytes of data",
                literal_count,
                match_count,
                data.len()
            ),
        });
    }

    if flags & !(FLAG_LITERAL_RLE | FLAG_LENGTH_RLE | FLAG_OFFSET_RLE | FLAG_WIDE_TYPE_BITS) != 0 {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!("FSE header: unknown flags 0x{:02X}", flags),
        });
    }

    let lit_is_rle = (flags & FLAG_LITERAL_RLE) != 0;
    let len_is_rle = (flags & FLAG_LENGTH_RLE) != 0;
    let off_is_rle = (flags & FLAG_OFFSET_RLE) != 0;

    let total_tokens = literal_count + match_count;

    // ── Token type bits ─────────────────────────────────────────────
    let type_bits_len = if flags & FLAG_WIDE_TYPE_BITS != 0 {
        reader.read_u32()? as usize
    } else {
        reader.read_u16()? as usize
    };
    let mut type_bits = Vec::with_capacity(type_bits_len);
    for _ in 0..type_bits_len {
        type_bits.push(reader.read_u8()?);
    }

    // ── Frequency tables ────────────────────────────────────────────
    let lit_rle_sym = if lit_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    let lit_norm = if !lit_is_rle && literal_count > 0 {
        deserialize_freq_table(&mut reader, NUM_LITERAL_CODES, lit_tl)?
    } else {
        Vec::new()
    };

    let len_rle_sym = if len_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    let len_norm = if !len_is_rle && match_count > 0 {
        deserialize_freq_table(&mut reader, NUM_LENGTH_CODES, len_tl)?
    } else {
        Vec::new()
    };

    let off_rle_sym = if off_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    let off_norm = if !off_is_rle && match_count > 0 {
        deserialize_freq_table(&mut reader, NUM_OFFSET_CODES, off_tl)?
    } else {
        Vec::new()
    };

    // ── Decode FSE sub-streams (zero-copy) ───────────────────────────
    // Align to byte boundary to match the encoder's alignment point.
    reader.align_to_byte();

    let literal_symbols = if let Some(sym) = lit_rle_sym {
        vec![sym; literal_count]
    } else if literal_count > 0 {
        let table = FseTable::from_normalized_decode_only(&lit_norm, lit_tl)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        table.decode(stream_data, literal_count)?
    } else {
        Vec::new()
    };

    let length_symbols = if let Some(sym) = len_rle_sym {
        vec![sym; match_count]
    } else if match_count > 0 {
        let table = FseTable::from_normalized_decode_only(&len_norm, len_tl)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        table.decode(stream_data, match_count)?
    } else {
        Vec::new()
    };

    let offset_symbols = if let Some(sym) = off_rle_sym {
        vec![sym; match_count]
    } else if match_count > 0 {
        let table = FseTable::from_normalized_decode_only(&off_norm, off_tl)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        table.decode(stream_data, match_count)?
    } else {
        Vec::new()
    };

    // ── Read extra bits (with symbol bounds validation) ─────────────
    let mut length_codes: Vec<LengthCode> = Vec::with_capacity(match_count);
    for &sym in &length_symbols {
        if sym as usize >= NUM_LENGTH_CODES {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("length symbol {} >= {}", sym, NUM_LENGTH_CODES),
            });
        }
        let eb = LENGTH_EXTRA[sym as usize];
        let ev = if eb > 0 {
            reader.read_bits(eb)? as u16
        } else {
            0
        };
        length_codes.push(LengthCode {
            code: sym,
            extra_bits: eb,
            extra_value: ev,
        });
    }

    let mut offset_codes: Vec<OffsetCode> = Vec::with_capacity(match_count);
    for &sym in &offset_symbols {
        if sym as usize >= NUM_OFFSET_CODES {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("offset symbol {} >= {}", sym, NUM_OFFSET_CODES),
            });
        }
        let eb = OFFSET_EXTRA[sym as usize];
        let ev = if eb > 0 {
            reader.read_bits(eb)? as u32
        } else {
            0
        };
        offset_codes.push(OffsetCode {
            code: sym,
            extra_bits: eb,
            extra_value: ev,
        });
    }

    // ── Reconstruct LzToken stream using type bits, resolving repcodes ──
    let mut tokens = Vec::with_capacity(total_tokens);
    let mut lit_idx: usize = 0;
    let mut match_idx: usize = 0;
    let mut repcode_state = RepcodeState::new();

    for i in 0..total_tokens {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_match = if byte_idx < type_bits.len() {
            (type_bits[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        };

        if is_match {
            if match_idx >= length_codes.len() || match_idx >= offset_codes.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: "FSE decode: match index overflow".into(),
                });
            }
            let length = decode_match_length(&length_codes[match_idx]);
            let offset_code = &offset_codes[match_idx];

            // Resolve repcodes: codes 0–2 are repcode slots.
            let offset = if is_repcode(offset_code.code) {
                let rep_idx = offset_code.code as u8;
                let real_offset = repcode_state.offsets[rep_idx as usize];
                repcode_state.update_rep(rep_idx);
                real_offset
            } else {
                let real_offset = decode_match_offset(offset_code);
                repcode_state.update_raw(real_offset);
                real_offset
            };

            tokens.push(LzToken::Match { offset, length });
            match_idx += 1;
        } else {
            if lit_idx >= literal_symbols.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: "FSE decode: literal index overflow".into(),
                });
            }
            tokens.push(LzToken::Literal(literal_symbols[lit_idx] as u8));
            lit_idx += 1;
        }
    }

    if lit_idx != literal_count || match_idx != match_count {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "FSE decode: consumed {} lits (expected {}), {} matches (expected {})",
                lit_idx, literal_count, match_idx, match_count
            ),
        });
    }

    Ok(tokens)
}

/// Workspace-aware variant of `decode_token_stream`. Uses scratch buffers from
/// `EntropyDecodeScratch` instead of allocating. After the first block warms up
/// the scratch, all subsequent blocks make zero heap allocations.
///
/// The decoded tokens are left in `scratch.tokens` for the caller to consume.
pub fn decode_token_stream_into(data: &[u8], scratch: &mut EntropyDecodeScratch) -> Result<()> {
    let mut reader = BitReader::new(data);

    // ── Header ──────────────────────────────────────────────────────
    let literal_count = reader.read_u32()? as usize;
    let match_count = reader.read_u32()? as usize;
    let lit_tl = reader.read_u8()?;
    let len_tl = reader.read_u8()?;
    let off_tl = reader.read_u8()?;
    let flags = reader.read_u8()?;

    // Validate table_log values.
    for &(name, tl) in &[("literal", lit_tl), ("length", len_tl), ("offset", off_tl)] {
        if !(MIN_TABLE_LOG..=MAX_TABLE_LOG).contains(&tl) {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!(
                    "FSE header: {} table_log {} out of range [{}, {}]",
                    name, tl, MIN_TABLE_LOG, MAX_TABLE_LOG
                ),
            });
        }
    }

    // Validate token counts.
    let max_tokens = data.len().saturating_mul(256);
    if literal_count > max_tokens || match_count > max_tokens {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "FSE header: token counts ({} lit, {} match) implausible for {} bytes of data",
                literal_count,
                match_count,
                data.len()
            ),
        });
    }

    if flags & !(FLAG_LITERAL_RLE | FLAG_LENGTH_RLE | FLAG_OFFSET_RLE | FLAG_WIDE_TYPE_BITS) != 0 {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!("FSE header: unknown flags 0x{:02X}", flags),
        });
    }

    let lit_is_rle = (flags & FLAG_LITERAL_RLE) != 0;
    let len_is_rle = (flags & FLAG_LENGTH_RLE) != 0;
    let off_is_rle = (flags & FLAG_OFFSET_RLE) != 0;

    let total_tokens = literal_count + match_count;

    // ── Token type bits (reuse scratch.type_bits) ────────────────────
    let type_bits_len = if flags & FLAG_WIDE_TYPE_BITS != 0 {
        reader.read_u32()? as usize
    } else {
        reader.read_u16()? as usize
    };
    scratch.type_bits.clear();
    scratch.type_bits.reserve(type_bits_len);
    for _ in 0..type_bits_len {
        scratch.type_bits.push(reader.read_u8()?);
    }

    // ── Frequency tables ──────────────────────────────────────────────
    // Wire format reads RLE sym + freq table per sub-stream (interleaved),
    // all before align_to_byte. All 3 freq tables must be held simultaneously
    // because the sub-stream data follows after alignment. Each deserializes
    // into its own scratch vec (lit_freqs/len_freqs/off_freqs).

    let lit_rle_sym = if lit_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    if !lit_is_rle && literal_count > 0 {
        deserialize_freq_table_into(
            &mut reader,
            NUM_LITERAL_CODES,
            lit_tl,
            &mut scratch.lit_freqs,
        )?;
    }

    let len_rle_sym = if len_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    if !len_is_rle && match_count > 0 {
        deserialize_freq_table_into(
            &mut reader,
            NUM_LENGTH_CODES,
            len_tl,
            &mut scratch.len_freqs,
        )?;
    }

    let off_rle_sym = if off_is_rle {
        Some(reader.read_u16()?)
    } else {
        None
    };
    if !off_is_rle && match_count > 0 {
        deserialize_freq_table_into(
            &mut reader,
            NUM_OFFSET_CODES,
            off_tl,
            &mut scratch.off_freqs,
        )?;
    }

    // Single FseTable rebuilt 3x (lit/len/off are sequential, never concurrent).
    // rebuild_decode_only reuses scratch for spread/symbol_next temporaries;
    // the FseTable's own decode_table Vec retains capacity across rebuilds.
    let table = &mut scratch.table_scratch;
    let mut fse_table = FseTable::empty();

    // ── Decode FSE sub-streams (zero-copy) ─────────────────────────
    reader.align_to_byte();

    // Literals
    scratch.literal_symbols.clear();
    if let Some(sym) = lit_rle_sym {
        scratch.literal_symbols.resize(literal_count, sym);
    } else if literal_count > 0 {
        fse_table.rebuild_decode_only(&scratch.lit_freqs, lit_tl, table)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        fse_table.decode_into(stream_data, literal_count, &mut scratch.literal_symbols)?;
    }

    // Lengths
    scratch.length_symbols.clear();
    if let Some(sym) = len_rle_sym {
        scratch.length_symbols.resize(match_count, sym);
    } else if match_count > 0 {
        fse_table.rebuild_decode_only(&scratch.len_freqs, len_tl, table)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        fse_table.decode_into(stream_data, match_count, &mut scratch.length_symbols)?;
    }

    // Offsets
    scratch.offset_symbols.clear();
    if let Some(sym) = off_rle_sym {
        scratch.offset_symbols.resize(match_count, sym);
    } else if match_count > 0 {
        fse_table.rebuild_decode_only(&scratch.off_freqs, off_tl, table)?;
        let stream_len = reader.read_aligned_u32()? as usize;
        let stream_data = reader.read_aligned_slice(stream_len)?;
        fse_table.decode_into(stream_data, match_count, &mut scratch.offset_symbols)?;
    }

    // ── Read extra bits (with symbol bounds validation) ─────────────
    scratch.length_codes.clear();
    scratch.length_codes.reserve(match_count);
    for &sym in &scratch.length_symbols {
        if sym as usize >= NUM_LENGTH_CODES {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("length symbol {} >= {}", sym, NUM_LENGTH_CODES),
            });
        }
        let eb = LENGTH_EXTRA[sym as usize];
        let ev = if eb > 0 {
            reader.read_bits(eb)? as u16
        } else {
            0
        };
        scratch.length_codes.push(LengthCode {
            code: sym,
            extra_bits: eb,
            extra_value: ev,
        });
    }

    scratch.offset_codes.clear();
    scratch.offset_codes.reserve(match_count);
    for &sym in &scratch.offset_symbols {
        if sym as usize >= NUM_OFFSET_CODES {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("offset symbol {} >= {}", sym, NUM_OFFSET_CODES),
            });
        }
        let eb = OFFSET_EXTRA[sym as usize];
        let ev = if eb > 0 {
            reader.read_bits(eb)? as u32
        } else {
            0
        };
        scratch.offset_codes.push(OffsetCode {
            code: sym,
            extra_bits: eb,
            extra_value: ev,
        });
    }

    // ── Reconstruct LzToken stream using type bits, resolving repcodes ──
    scratch.tokens.clear();
    scratch.tokens.reserve(total_tokens);
    let mut lit_idx: usize = 0;
    let mut match_idx: usize = 0;
    let mut repcode_state = RepcodeState::new();

    for i in 0..total_tokens {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let is_match = if byte_idx < scratch.type_bits.len() {
            (scratch.type_bits[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        };

        if is_match {
            if match_idx >= scratch.length_codes.len() || match_idx >= scratch.offset_codes.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: "FSE decode: match index overflow".into(),
                });
            }
            let length = decode_match_length(&scratch.length_codes[match_idx]);
            let offset_code = &scratch.offset_codes[match_idx];

            let offset = if is_repcode(offset_code.code) {
                let rep_idx = offset_code.code as u8;
                let real_offset = repcode_state.offsets[rep_idx as usize];
                repcode_state.update_rep(rep_idx);
                real_offset
            } else {
                let real_offset = decode_match_offset(offset_code);
                repcode_state.update_raw(real_offset);
                real_offset
            };

            scratch.tokens.push(LzToken::Match { offset, length });
            match_idx += 1;
        } else {
            if lit_idx >= scratch.literal_symbols.len() {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: "FSE decode: literal index overflow".into(),
                });
            }
            scratch
                .tokens
                .push(LzToken::Literal(scratch.literal_symbols[lit_idx] as u8));
            lit_idx += 1;
        }
    }

    if lit_idx != literal_count || match_idx != match_count {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "FSE decode: consumed {} lits (expected {}), {} matches (expected {})",
                lit_idx, literal_count, match_idx, match_count
            ),
        });
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bit I/O ─────────────────────────────────────────────────────

    #[test]
    fn bitwriter_reader_roundtrip() {
        let mut w = BitWriter::with_capacity(64);
        w.write_bits(0b101, 3);
        w.write_bits(0xFF, 8);
        w.write_bits(42, 16);
        w.write_u32(0xDEADBEEF);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(8).unwrap(), 0xFF);
        assert_eq!(r.read_bits(16).unwrap(), 42);
        assert_eq!(r.read_u32().unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn bitwriter_reader_aligned_roundtrip() {
        // Write some variable-width bits (simulating freq tables), then align
        // and write raw bytes (simulating sub-streams).
        let mut w = BitWriter::with_capacity(64);
        w.write_bits(0b101, 3); // 3 bits — not byte-aligned
        w.write_bits(0xAB, 8); // now 11 bits total
        w.align_to_byte(); // pads to 16 bits (2 bytes)
        let raw_data = b"HELLO";
        w.write_aligned_bytes(&(raw_data.len() as u32).to_le_bytes());
        w.write_aligned_bytes(raw_data);
        // Back to bit-level for extra bits.
        w.write_bits(0b11, 2);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
        r.align_to_byte();
        let len = r.read_aligned_u32().unwrap() as usize;
        assert_eq!(len, 5);
        let slice = r.read_aligned_slice(len).unwrap();
        assert_eq!(slice, b"HELLO");
        // Transition back to bit-level.
        assert_eq!(r.read_bits(2).unwrap(), 0b11);
    }

    // ── Frequency normalization ─────────────────────────────────────

    #[test]
    fn normalize_uniform() {
        let raw = vec![100u32; 4];
        let norm = normalize_frequencies(&raw, 8);
        assert_eq!(norm.iter().map(|&n| n as u32).sum::<u32>(), 256);
        for &n in &norm {
            assert!(n >= 63 && n <= 65);
        }
    }

    #[test]
    fn normalize_skewed() {
        let raw = vec![1000, 100, 10, 1];
        let norm = normalize_frequencies(&raw, 10);
        assert_eq!(norm.iter().map(|&n| n as u32).sum::<u32>(), 1024);
        for (i, &n) in norm.iter().enumerate() {
            if raw[i] > 0 {
                assert!(n >= 1, "symbol {} got 0", i);
            }
        }
        assert!(norm[0] > norm[1]);
    }

    #[test]
    fn normalize_single_symbol() {
        let raw = vec![0, 0, 500, 0];
        let norm = normalize_frequencies(&raw, 9);
        assert_eq!(norm[2], 512);
    }

    #[test]
    fn normalize_many_rare() {
        let raw = vec![1u32; 100];
        let norm = normalize_frequencies(&raw, 10);
        assert_eq!(norm.iter().map(|&n| n as u32).sum::<u32>(), 1024);
        for &n in &norm[..100] {
            assert!(n >= 1);
        }
    }

    #[test]
    fn normalize_preserves_nonzero() {
        let mut raw = vec![0u32; 256];
        for i in 0..200 {
            raw[i] = 1;
        }
        raw[0] = 10000;
        let norm = normalize_frequencies(&raw, 11);
        assert_eq!(norm.iter().map(|&n| n as u32).sum::<u32>(), 2048);
        for (i, &n) in norm.iter().enumerate() {
            if raw[i] > 0 {
                assert!(n >= 1, "symbol {} with raw {} got 0", i, raw[i]);
            }
        }
    }

    // ── FSE table construction ──────────────────────────────────────

    #[test]
    fn fse_table_build_uniform() {
        let freqs = vec![8u16, 8, 8, 8];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        assert_eq!(table.table_size, 32);
    }

    // ── FSE encode/decode roundtrip ─────────────────────────────────

    #[test]
    fn fse_roundtrip_uniform_4sym() {
        let freqs = vec![8u16, 8, 8, 8];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        let symbols: Vec<u16> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3];
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, symbols.len()).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn fse_roundtrip_skewed_4sym() {
        let freqs = vec![16u16, 8, 4, 4];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        let symbols: Vec<u16> = vec![0, 0, 0, 0, 1, 1, 0, 0, 2, 3, 0, 1, 0, 0, 0, 0];
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, symbols.len()).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn fse_roundtrip_single_symbol() {
        let freqs = vec![0u16, 0, 32, 0];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        let symbols: Vec<u16> = vec![2; 100];
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, 100).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn fse_roundtrip_two_symbols() {
        let freqs = vec![16u16, 16];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        let symbols: Vec<u16> = vec![0, 1, 0, 1, 1, 0, 0, 0, 1, 1];
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, symbols.len()).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn fse_roundtrip_longer_stream() {
        let freqs = vec![8u16, 8, 8, 8];
        let table = FseTable::from_normalized(&freqs, 5).unwrap();
        let symbols: Vec<u16> = (0..200).map(|i| (i % 4) as u16).collect();
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, 200).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn fse_roundtrip_large_alphabet() {
        // Simulate 256-symbol literal FSE.
        let table_log: u8 = 11;
        let table_size = 1u32 << table_log;
        let mut freqs = vec![1u16; 256]; // Start with 1 each = 256 total
                                         // Distribute remaining table_size - 256 = 1792 slots.
        freqs[0] += (table_size - 256) as u16; // Give the rest to symbol 0.
        assert_eq!(freqs.iter().map(|&f| f as u32).sum::<u32>(), table_size);

        let table = FseTable::from_normalized(&freqs, table_log).unwrap();
        let symbols: Vec<u16> = (0..500).map(|i| ((i * 7 + 3) % 256) as u16).collect();
        let encoded = table.encode(&symbols).unwrap();
        let decoded = table.decode(&encoded, 500).unwrap();
        assert_eq!(decoded, symbols);
    }

    // ── Freq table serialization ────────────────────────────────────

    #[test]
    fn freq_table_serialize_roundtrip() {
        let mut freqs = vec![0u16; 256];
        freqs[0] = 512;
        freqs[10] = 256;
        freqs[65] = 128;
        freqs[200] = 128;

        let mut writer = BitWriter::with_capacity(256);
        serialize_freq_table(&freqs, 10, &mut writer);
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        let decoded = deserialize_freq_table(&mut reader, 256, 10).unwrap();
        assert_eq!(decoded, freqs);
    }

    // ── Full token stream roundtrip ─────────────────────────────────

    #[test]
    fn token_stream_all_literals() {
        let tokens: Vec<LzToken> = (0..100u8).map(|b| LzToken::Literal(b)).collect();
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_all_matches() {
        let tokens: Vec<LzToken> = (0..50)
            .map(|i| LzToken::Match {
                offset: (i % 100) + 1,
                length: (i % 32) + 4,
            })
            .collect();
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_mixed() {
        let tokens = vec![
            LzToken::Literal(b'H'),
            LzToken::Literal(b'e'),
            LzToken::Literal(b'l'),
            LzToken::Literal(b'l'),
            LzToken::Literal(b'o'),
            LzToken::Match {
                offset: 5,
                length: 5,
            }, // "Hello" repeated
            LzToken::Literal(b'!'),
        ];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_single_literal() {
        let tokens = vec![LzToken::Literal(42)];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_single_match() {
        let tokens = vec![LzToken::Match {
            offset: 100,
            length: 10,
        }];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_rle_literals() {
        // All same literal byte → RLE mode.
        let tokens: Vec<LzToken> = vec![LzToken::Literal(0xAA); 1000];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_rle_matches() {
        // All same match → RLE for both length and offset codes.
        let tokens: Vec<LzToken> = vec![
            LzToken::Match {
                offset: 1,
                length: 4
            };
            500
        ];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_large_offsets() {
        let tokens: Vec<LzToken> = vec![
            LzToken::Match {
                offset: 1,
                length: 4,
            },
            LzToken::Match {
                offset: 1000,
                length: 100,
            },
            LzToken::Match {
                offset: 100000,
                length: 1000,
            },
            LzToken::Match {
                offset: crate::MAX_MATCH_OFFSET,
                length: crate::MAX_MATCH_LEN,
            },
            LzToken::Literal(0xFF),
        ];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_stress_mixed() {
        // 5000 tokens: alternating literals and matches with varying params.
        let mut tokens = Vec::with_capacity(5000);
        for i in 0..5000u32 {
            if i % 3 == 0 {
                tokens.push(LzToken::Match {
                    offset: (i % 1000) + 1,
                    length: (i % 60) + 4,
                });
            } else {
                tokens.push(LzToken::Literal((i % 256) as u8));
            }
        }
        let (encoded, cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);

        // Verify cost model returns reasonable values.
        assert!(cost.literal_cost(0) > 0.0);
        assert!(cost.match_cost(4, 1) > 0.0);
    }

    #[test]
    fn token_stream_empty_fails() {
        assert!(encode_token_stream(&[]).is_err());
        assert!(decode_token_stream(&[]).is_err());
    }

    #[test]
    fn cost_model_from_token_stream() {
        // Use a smaller literal range so all symbols are well-represented.
        let tokens: Vec<LzToken> = (0..2000)
            .map(|i| {
                if i % 2 == 0 {
                    LzToken::Literal((i % 16) as u8)
                } else {
                    LzToken::Match {
                        offset: (i % 50) + 1,
                        length: (i % 20) + 4,
                    }
                }
            })
            .collect();

        let (_, cost) = encode_token_stream(&tokens).unwrap();

        // With 16 distinct literals each appearing ~62 times, costs should
        // be close to log2(16) = 4 bits.
        let c0 = cost.literal_cost(0);
        let c2 = cost.literal_cost(2); // both 0 and 2 are present (even indices)
        assert!(c0 > 0.0 && c0 < 10.0, "literal 0 cost {} unreasonable", c0);
        assert!(
            (c0 - c2).abs() < 1.0,
            "costs {} and {} too different",
            c0,
            c2
        );

        // Match costs should be positive.
        assert!(cost.match_cost(4, 1) > 0.0);
    }

    // ── Repcode encode/decode ─────────────────────────────────────────

    #[test]
    fn token_stream_repcode_roundtrip_repeated_offset() {
        // Many matches at the same offset — repcodes should fire for all after the first.
        let tokens: Vec<LzToken> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    LzToken::Literal((i % 256) as u8)
                } else {
                    LzToken::Match {
                        offset: 42,
                        length: 4,
                    }
                }
            })
            .collect();
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_repcode_alternating_offsets() {
        // Two offsets alternating — rep0 and rep1 should both fire.
        let mut tokens = Vec::new();
        for i in 0..200 {
            tokens.push(LzToken::Literal((i % 256) as u8));
            tokens.push(LzToken::Match {
                offset: if i % 2 == 0 { 10 } else { 20 },
                length: 4,
            });
        }
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_repcode_three_offsets_rotating() {
        // Three offsets cycling — all three repcode slots exercised.
        let offsets = [100, 200, 300];
        let mut tokens = Vec::new();
        for i in 0..300 {
            tokens.push(LzToken::Literal((i % 256) as u8));
            tokens.push(LzToken::Match {
                offset: offsets[i % 3],
                length: 5,
            });
        }
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_repcode_initial_offsets() {
        // Matches at offsets 1, 4, 8 (initial repcode values) — should be repcodes
        // from the very first match.
        let tokens = vec![
            LzToken::Match {
                offset: 1,
                length: 4,
            },
            LzToken::Match {
                offset: 4,
                length: 4,
            },
            LzToken::Match {
                offset: 8,
                length: 4,
            },
            LzToken::Match {
                offset: 1,
                length: 4,
            }, // back to rep0/1/2
            LzToken::Match {
                offset: 4,
                length: 4,
            },
            LzToken::Match {
                offset: 8,
                length: 4,
            },
        ];
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_repcode_consecutive_matches() {
        // Consecutive matches with no intervening literals — stress test
        // for repcode state tracking staying in sync.
        let mut tokens = Vec::new();
        for _ in 0..50 {
            tokens.push(LzToken::Match {
                offset: 7,
                length: 4,
            });
            tokens.push(LzToken::Match {
                offset: 13,
                length: 6,
            });
            tokens.push(LzToken::Match {
                offset: 7,
                length: 8,
            }); // rep1
            tokens.push(LzToken::Match {
                offset: 13,
                length: 4,
            }); // rep1 again
        }
        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_repcode_improves_compression() {
        // Same data encoded with repeated offsets should compress smaller than
        // the same number of tokens with all-different offsets.
        let repeated: Vec<LzToken> = (0..500)
            .map(|i| {
                if i % 2 == 0 {
                    LzToken::Literal((i % 256) as u8)
                } else {
                    LzToken::Match {
                        offset: 42,
                        length: 4,
                    }
                }
            })
            .collect();

        let diverse: Vec<LzToken> = (0..500)
            .map(|i| {
                if i % 2 == 0 {
                    LzToken::Literal((i % 256) as u8)
                } else {
                    LzToken::Match {
                        offset: (i as u32 * 7 % 1000) + 1,
                        length: 4,
                    }
                }
            })
            .collect();

        let (enc_rep, _) = encode_token_stream(&repeated).unwrap();
        let (enc_div, _) = encode_token_stream(&diverse).unwrap();

        // Repeated offsets should compress significantly smaller because repcodes
        // eliminate offset entropy.
        assert!(
            enc_rep.len() < enc_div.len(),
            "repeated offsets ({} bytes) should compress smaller than diverse ({} bytes)",
            enc_rep.len(),
            enc_div.len()
        );
    }

    #[test]
    fn wide_type_bits_roundtrip_large_token_stream() {
        // Regression test: >65535 type_bits bytes (>524280 tokens) caused u16
        // truncation of type_bits_len, corrupting the FSE bitstream.
        // Generate 600K literal tokens — well past the u16 boundary.
        let n = 600_000;
        let tokens: Vec<LzToken> = (0..n).map(|i| LzToken::Literal((i & 0xFF) as u8)).collect();

        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded.len(), n, "token count mismatch");
        for (i, tok) in decoded.iter().enumerate() {
            match tok {
                LzToken::Literal(v) => assert_eq!(*v, (i & 0xFF) as u8),
                _ => panic!("expected literal at index {}, got match", i),
            }
        }
    }

    #[test]
    fn wide_type_bits_boundary_just_under() {
        // 524,280 tokens = 65,535 type_bits bytes — exact u16 max.
        // Should NOT set FLAG_WIDE_TYPE_BITS (u16 path).
        let n = 524_280;
        let tokens: Vec<LzToken> = (0..n).map(|i| LzToken::Literal((i & 0xFF) as u8)).collect();

        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        // Verify FLAG_WIDE_TYPE_BITS is NOT set: the flags byte is at offset 12
        // (4 bytes literal_count + 4 bytes match_count + 1 lit_tl + 1 len_tl + 1 off_tl + 1 flags).
        assert_eq!(
            encoded[11] & FLAG_WIDE_TYPE_BITS,
            0,
            "should use u16 at boundary"
        );
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded.len(), n);
        for (i, tok) in decoded.iter().enumerate() {
            match tok {
                LzToken::Literal(v) => assert_eq!(*v, (i & 0xFF) as u8),
                _ => panic!("expected literal at index {}", i),
            }
        }
    }

    #[test]
    fn wide_type_bits_boundary_just_over() {
        // 524,288 tokens = 65,536 type_bits bytes — one byte over u16 max.
        // MUST set FLAG_WIDE_TYPE_BITS (u32 path).
        let n = 524_288;
        let tokens: Vec<LzToken> = (0..n).map(|i| LzToken::Literal((i & 0xFF) as u8)).collect();

        let (encoded, _cost) = encode_token_stream(&tokens).unwrap();
        assert_ne!(
            encoded[11] & FLAG_WIDE_TYPE_BITS,
            0,
            "should use u32 past boundary"
        );
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded.len(), n);
        for (i, tok) in decoded.iter().enumerate() {
            match tok {
                LzToken::Literal(v) => assert_eq!(*v, (i & 0xFF) as u8),
                _ => panic!("expected literal at index {}", i),
            }
        }
    }

    #[test]
    fn wide_type_bits_roundtrip_into_variant() {
        // Exercises encode_tokens_into / decode_tokens_into (workspace path).
        // Verifies both count AND content.
        use crate::workspace::EntropyEncodeScratch;
        let n = 600_000;
        let tokens: Vec<LzToken> = (0..n).map(|i| LzToken::Literal((i & 0xFF) as u8)).collect();

        let mut scratch = EntropyEncodeScratch::new();
        let (encoded, _cost) = encode_token_stream_into(&tokens, &mut scratch).unwrap();
        let decoded = decode_token_stream(&encoded).unwrap();
        assert_eq!(decoded.len(), n, "token count mismatch");
        for (i, tok) in decoded.iter().enumerate() {
            match tok {
                LzToken::Literal(v) => assert_eq!(*v, (i & 0xFF) as u8),
                _ => panic!("expected literal at index {}, got match", i),
            }
        }
    }
}
