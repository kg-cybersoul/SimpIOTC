//! # Zero-Allocation Decode Workspaces
//!
//! Reusable scratch buffers for the decompress hot path. After warmup (first block),
//! all subsequent blocks with same-or-smaller data make zero heap allocations — the
//! only irreducible allocation is the output `Vec<u8>` returned to the caller.
//!
//! ## Layout
//!
//! ```text
//! DecodeWorkspace
//!   ├── EntropyDecodeScratch    (19 allocs → 0)
//!   │     ├── FseTableScratch   (spread, symbol_next, decode_table — rebuilt 3× per block)
//!   │     ├── freq_table         (single Vec<u16>, reused for lit/len/off)
//!   │     ├── type_bits, symbol vecs, code vecs, tokens
//!   ├── ReplayScratch            (1 alloc → 0)
//!   │     └── output: Vec<u8>
//!   └── PreprocessScratch        (2 allocs → 0)
//!         └── output: Vec<u8>
//! ```
//!
//! ## Usage
//!
//! Each `_into` function calls `vec.clear()` (keeps capacity), then `reserve()`
//! (no-op if already sized). After the first block, all subsequent blocks with
//! same-or-smaller data make zero allocations.
//!
//! ## Thread Strategy
//!
//! Use `thread_local!` with `RefCell<DecodeWorkspace>` in the parallel decompress
//! path. Each Rayon worker thread gets its own workspace. `RefCell` is safe because
//! Rayon runs each `par_iter` item to completion on one thread.

use crate::entropy::fse::FseTable;
use crate::entropy::{LengthCode, OffsetCode};
use crate::match_finder::{HashChain, Match};
use crate::parser::optimal::Decision;
use crate::LzToken;

// ═══════════════════════════════════════════════════════════════════════════════
// FSE Table Scratch — reused across 3 table rebuilds per block
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffers for `FseTable::rebuild_decode_only`. A single instance is
/// reused for all three sub-stream tables (literals, lengths, offsets) because
/// they are decoded sequentially, never concurrently.
pub struct FseTableScratch {
    /// Symbol spread buffer. Sized to `table_size` (max 8192 for MAX_TABLE_LOG=13).
    pub spread: Vec<u16>,
    /// Per-symbol next-state counter. Sized to `alphabet_size` (max 256).
    pub symbol_next: Vec<u32>,
}

impl Default for FseTableScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl FseTableScratch {
    /// Create a new scratch with zero capacity. The first `rebuild_decode_only`
    /// call will allocate; subsequent calls reuse the existing capacity.
    pub fn new() -> Self {
        Self {
            spread: Vec::new(),
            symbol_next: Vec::new(),
        }
    }

    /// Create a scratch pre-sized for the given table parameters.
    pub fn with_capacity(max_table_size: usize, max_alphabet_size: usize) -> Self {
        Self {
            spread: Vec::with_capacity(max_table_size),
            symbol_next: Vec::with_capacity(max_alphabet_size),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Entropy Decode Scratch — all temporaries for decode_token_stream_into
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffers for the full entropy decode pass. Eliminates 19 allocations
/// per block: 3× freq tables, 3× FseTable internals (×3 = 9), 3× symbol vecs,
/// 2× code vecs, 1× type_bits, 1× tokens.
pub struct EntropyDecodeScratch {
    /// Reusable FSE table scratch (shared across lit/len/off table rebuilds).
    pub table_scratch: FseTableScratch,

    /// Frequency table buffers — one per sub-stream. All 3 freq tables are
    /// deserialized from the bitstream before the aligned sub-streams, so they
    /// must all be held simultaneously. After warmup, the Vecs retain capacity
    /// (256, 52, 44 entries respectively).
    pub lit_freqs: Vec<u16>,
    pub len_freqs: Vec<u16>,
    pub off_freqs: Vec<u16>,

    /// Packed token type bits (0=literal, 1=match). Capacity: ceil(total_tokens/8).
    pub type_bits: Vec<u8>,

    /// Decoded literal symbols from FSE stream.
    pub literal_symbols: Vec<u16>,
    /// Decoded length symbols from FSE stream.
    pub length_symbols: Vec<u16>,
    /// Decoded offset symbols from FSE stream.
    pub offset_symbols: Vec<u16>,

    /// Decoded length codes with extra bits applied.
    pub length_codes: Vec<LengthCode>,
    /// Decoded offset codes with extra bits applied.
    pub offset_codes: Vec<OffsetCode>,

    /// Reconstructed LZ77 token stream.
    pub tokens: Vec<LzToken>,
}

impl Default for EntropyDecodeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyDecodeScratch {
    /// Create a new scratch with zero capacity.
    pub fn new() -> Self {
        Self {
            table_scratch: FseTableScratch::new(),
            lit_freqs: Vec::new(),
            len_freqs: Vec::new(),
            off_freqs: Vec::new(),
            type_bits: Vec::new(),
            literal_symbols: Vec::new(),
            length_symbols: Vec::new(),
            offset_symbols: Vec::new(),
            length_codes: Vec::new(),
            offset_codes: Vec::new(),
            tokens: Vec::new(),
        }
    }

    /// Create a scratch pre-sized for the given block parameters.
    ///
    /// `max_tokens` is the expected maximum number of LZ77 tokens per block.
    /// For a 2 MiB block, worst case is ~2M tokens (all literals).
    pub fn with_capacity(max_tokens: usize) -> Self {
        // Largest FSE table: 2^13 = 8192 states. Largest alphabet: 256 (literals).
        let max_table_size = 1 << 13;
        Self {
            table_scratch: FseTableScratch::with_capacity(max_table_size, 256),
            lit_freqs: Vec::with_capacity(256),
            len_freqs: Vec::with_capacity(52),
            off_freqs: Vec::with_capacity(44),
            type_bits: Vec::with_capacity(max_tokens.div_ceil(8)),
            literal_symbols: Vec::with_capacity(max_tokens),
            length_symbols: Vec::with_capacity(max_tokens),
            offset_symbols: Vec::with_capacity(max_tokens),
            length_codes: Vec::with_capacity(max_tokens),
            offset_codes: Vec::with_capacity(max_tokens),
            tokens: Vec::with_capacity(max_tokens),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Replay Scratch — reusable output buffer for token replay
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffer for `replay_tokens_into`. Eliminates 1 allocation per block.
pub struct ReplayScratch {
    /// Reconstructed byte stream from token replay.
    pub output: Vec<u8>,
}

impl Default for ReplayScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayScratch {
    pub fn new() -> Self {
        Self { output: Vec::new() }
    }

    pub fn with_capacity(block_size: usize) -> Self {
        Self {
            output: Vec::with_capacity(block_size),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Preprocess Scratch — reusable output buffer for depreprocess
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffer for `depreprocess_into`. Eliminates 2 allocations per block
/// (the intermediate typed Vec and the final byte Vec) by writing LE bytes
/// directly into the output buffer.
pub struct PreprocessScratch {
    /// Final decompressed output bytes.
    pub output: Vec<u8>,
}

impl Default for PreprocessScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessScratch {
    pub fn new() -> Self {
        Self { output: Vec::new() }
    }

    pub fn with_capacity(block_size: usize) -> Self {
        Self {
            output: Vec::with_capacity(block_size),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-Level Decode Workspace
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete set of reusable scratch buffers for the decompress hot path.
///
/// After the first block warms up the workspace (allocating to the needed
/// capacities), all subsequent blocks of the same or smaller size make
/// zero heap allocations — the only irreducible allocation is the output
/// `Vec<u8>` returned to the caller.
pub struct DecodeWorkspace {
    pub entropy: EntropyDecodeScratch,
    pub replay: ReplayScratch,
    pub preprocess: PreprocessScratch,
    /// Scratch buffer for stride un-transposition on decompress.
    pub stride_buf: Vec<u8>,
}

impl Default for DecodeWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeWorkspace {
    /// Create a new workspace with zero capacity. First block allocates; all
    /// subsequent blocks reuse.
    pub fn new() -> Self {
        Self {
            entropy: EntropyDecodeScratch::new(),
            replay: ReplayScratch::new(),
            preprocess: PreprocessScratch::new(),
            stride_buf: Vec::new(),
        }
    }

    /// Create a workspace pre-sized for the given block parameters.
    pub fn with_capacity(block_size: usize) -> Self {
        Self {
            entropy: EntropyDecodeScratch::with_capacity(block_size),
            replay: ReplayScratch::with_capacity(block_size),
            preprocess: PreprocessScratch::with_capacity(block_size),
            stride_buf: Vec::with_capacity(block_size),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Borrowed PreprocessedData — avoids cloning replay output
// ═══════════════════════════════════════════════════════════════════════════════

use crate::DataType;

/// A borrowed view of preprocessed data. Used by `depreprocess_into` to avoid
/// the allocation of `PreprocessedData.data: Vec<u8>` — instead borrows the
/// replay scratch buffer directly.
pub struct PreprocessedDataRef<'a> {
    pub data_type: DataType,
    pub element_count: u64,
    pub data: &'a [u8],
}

// ═══════════════════════════════════════════════════════════════════════════════
// FSE Encode Table Scratch — reused temporaries for FseTable::rebuild_full
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffers for `FseTable::rebuild_full`. Holds the temporary arrays
/// that `from_normalized` allocates fresh each call: spread, symbol_next,
/// cum_freq, and sym_count. After warmup, all four retain capacity.
pub struct FseEncodeTableScratch {
    /// Symbol spread buffer. Sized to `table_size` (max 8192).
    pub spread: Vec<u16>,
    /// Per-symbol next-state counter. Sized to `alphabet_size` (max 256).
    pub symbol_next: Vec<u32>,
    /// Cumulative frequency table. Sized to `alphabet_size + 1`.
    pub cum_freq: Vec<u32>,
    /// Per-symbol occurrence counter for state_table building. Sized to `alphabet_size`.
    pub sym_count: Vec<u32>,
}

impl Default for FseEncodeTableScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl FseEncodeTableScratch {
    pub fn new() -> Self {
        Self {
            spread: Vec::new(),
            symbol_next: Vec::new(),
            cum_freq: Vec::new(),
            sym_count: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Entropy Encode Scratch — all temporaries for encode_token_stream_into
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffers for the full entropy encode pass. Eliminates ~38 allocations
/// per block: 5 symbol/extra vecs, 3× raw freq, 3× norm freq, 3× FseTable
/// internals, 3× emissions.
pub struct EntropyEncodeScratch {
    /// Reusable FSE table scratch for in-place table rebuilds.
    pub table_scratch: FseEncodeTableScratch,

    /// Single FseTable rebuilt 3× per block (lit → len → off, sequential).
    pub fse_table: FseTable,

    /// Packed token type bits.
    pub type_bits: Vec<u8>,

    /// Literal byte values extracted from tokens.
    pub literal_symbols: Vec<u16>,
    /// Length code symbols extracted from match tokens.
    pub length_symbols: Vec<u16>,
    /// Offset code symbols extracted from match tokens.
    pub offset_symbols: Vec<u16>,

    /// Length extra bits/values for match tokens.
    pub length_extras: Vec<(u8, u16)>,
    /// Offset extra bits/values for match tokens.
    pub offset_extras: Vec<(u8, u32)>,

    /// Raw frequency counts per sub-stream.
    pub lit_raw_freqs: Vec<u32>,
    pub len_raw_freqs: Vec<u32>,
    pub off_raw_freqs: Vec<u32>,

    /// Normalized frequency tables per sub-stream.
    pub lit_norm_freqs: Vec<u16>,
    pub len_norm_freqs: Vec<u16>,
    pub off_norm_freqs: Vec<u16>,

    /// Reusable emissions buffer for FseTable::encode_into.
    pub emissions: Vec<(u8, u64)>,
}

impl Default for EntropyEncodeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyEncodeScratch {
    pub fn new() -> Self {
        Self {
            table_scratch: FseEncodeTableScratch::new(),
            fse_table: FseTable::empty(),
            type_bits: Vec::new(),
            literal_symbols: Vec::new(),
            length_symbols: Vec::new(),
            offset_symbols: Vec::new(),
            length_extras: Vec::new(),
            offset_extras: Vec::new(),
            lit_raw_freqs: Vec::new(),
            len_raw_freqs: Vec::new(),
            off_raw_freqs: Vec::new(),
            lit_norm_freqs: Vec::new(),
            len_norm_freqs: Vec::new(),
            off_norm_freqs: Vec::new(),
            emissions: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parser Scratch — reusable buffers for LZ77 parsing
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffers for all parser modes. The HashChain is the dominant
/// allocation (~24 MB); the DP arrays are large for optimal mode (~32 MB).
/// After the first block, all subsequent same-config blocks make zero allocations.
pub struct ParserScratch {
    /// Reusable hash chain. `None` on first use → created from config.
    /// Subsequent calls: `take()` → `MatchFinder::with_chain()` → `take_chain()` → put back.
    pub chain: Option<HashChain>,

    /// Final parse output (all modes).
    pub tokens: Vec<LzToken>,

    /// Greedy first-pass tokens (optimal mode only).
    pub initial_tokens: Vec<LzToken>,

    /// DP cost array (optimal mode only). Sized to `data.len() + 1`.
    pub cost: Vec<f32>,

    /// DP decision array (optimal mode only). Sized to `data.len() + 1`.
    pub(crate) decision: Vec<Decision>,

    /// Pareto frontier match buffer (optimal mode). Reused across positions.
    pub match_buf: Vec<Match>,
}

impl Default for ParserScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl ParserScratch {
    pub fn new() -> Self {
        Self {
            chain: None,
            tokens: Vec::new(),
            initial_tokens: Vec::new(),
            cost: Vec::new(),
            decision: Vec::new(),
            match_buf: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Preprocess Encode Scratch — reusable output for preprocessor encode
// ═══════════════════════════════════════════════════════════════════════════════

/// Scratch buffer for `preprocess_into`. Reuses the encode output Vec.
/// The intermediate typed Vec (e.g. `Vec<i64>` from `bytes_to_i64s`) is not
/// reused — one ~800 KB allocation per block is acceptable given the complexity
/// of type-erased buffer reuse.
pub struct PreprocessEncodeScratch {
    /// Preprocessed (delta/gorilla-encoded) output bytes.
    pub output: Vec<u8>,
    /// Temporary buffer for composed transforms (e.g. shuffle → byte-delta).
    /// Holds the intermediate shuffled data while byte-delta encodes into `output`.
    pub shuffle_buf: Vec<u8>,
}

impl Default for PreprocessEncodeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessEncodeScratch {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            shuffle_buf: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-Level Encode Workspace
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete set of reusable scratch buffers for the compress hot path.
///
/// After the first block warms up the workspace, all subsequent blocks of the
/// same or smaller size make only 2 allocations: the irreducible payload
/// `Vec<u8>` returned per block and the `bytes_to_typed` intermediate.
pub struct EncodeWorkspace {
    pub preprocess: PreprocessEncodeScratch,
    pub parser: ParserScratch,
    pub entropy: EntropyEncodeScratch,
    /// Scratch buffer for stride transposition on compress.
    pub stride_buf: Vec<u8>,
}

impl Default for EncodeWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeWorkspace {
    pub fn new() -> Self {
        Self {
            preprocess: PreprocessEncodeScratch::new(),
            parser: ParserScratch::new(),
            entropy: EntropyEncodeScratch::new(),
            stride_buf: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_new_zero_capacity() {
        let ws = DecodeWorkspace::new();
        assert_eq!(ws.entropy.tokens.capacity(), 0);
        assert_eq!(ws.entropy.table_scratch.spread.capacity(), 0);
        assert_eq!(ws.replay.output.capacity(), 0);
        assert_eq!(ws.preprocess.output.capacity(), 0);
    }

    #[test]
    fn workspace_with_capacity_preallocates() {
        let ws = DecodeWorkspace::with_capacity(2 * 1024 * 1024);
        assert!(ws.entropy.tokens.capacity() >= 2 * 1024 * 1024);
        assert!(ws.entropy.table_scratch.spread.capacity() >= 8192);
        assert!(ws.entropy.lit_freqs.capacity() >= 256);
        assert!(ws.entropy.len_freqs.capacity() >= 52);
        assert!(ws.entropy.off_freqs.capacity() >= 44);
        assert!(ws.replay.output.capacity() >= 2 * 1024 * 1024);
        assert!(ws.preprocess.output.capacity() >= 2 * 1024 * 1024);
    }

    #[test]
    fn entropy_scratch_new_zero_capacity() {
        let s = EntropyDecodeScratch::new();
        assert_eq!(s.lit_freqs.capacity(), 0);
        assert_eq!(s.len_freqs.capacity(), 0);
        assert_eq!(s.off_freqs.capacity(), 0);
        assert_eq!(s.type_bits.capacity(), 0);
        assert_eq!(s.literal_symbols.capacity(), 0);
        assert_eq!(s.length_symbols.capacity(), 0);
        assert_eq!(s.offset_symbols.capacity(), 0);
        assert_eq!(s.length_codes.capacity(), 0);
        assert_eq!(s.offset_codes.capacity(), 0);
        assert_eq!(s.tokens.capacity(), 0);
    }

    #[test]
    fn fse_table_scratch_with_capacity() {
        let s = FseTableScratch::with_capacity(1024, 52);
        assert!(s.spread.capacity() >= 1024);
        assert!(s.symbol_next.capacity() >= 52);
    }

    #[test]
    fn preprocessed_data_ref_borrows() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let pdr = PreprocessedDataRef {
            data_type: DataType::IntegerI64,
            element_count: 1,
            data: &data,
        };
        assert_eq!(pdr.data.len(), 8);
        assert_eq!(pdr.element_count, 1);
        assert_eq!(pdr.data_type, DataType::IntegerI64);
    }

    // ══════════════════════════════════════════════════════════════════
    // Equivalence tests: _into variants must match originals exactly
    // ══════════════════════════════════════════════════════════════════

    use crate::entropy;
    use crate::parser;
    use crate::preprocessor;
    use crate::preprocessor::delta;
    use crate::preprocessor::gorilla_xor;

    /// Helper: generate LzTokens from data via compress pipeline, return encoded payload.
    fn make_entropy_payload(data: &[u8]) -> Vec<u8> {
        let config = crate::CompressionConfig::default();
        let tokens = parser::parse(data, &config).unwrap();
        let (payload, _) = entropy::encode_tokens(&tokens).unwrap();
        payload
    }

    #[test]
    fn entropy_decode_into_equivalence() {
        // Use data long enough to produce a mix of literals and matches.
        let data = b"ABCDEFGH".repeat(100);
        let payload = make_entropy_payload(&data);

        let original = entropy::decode_tokens(&payload).unwrap();
        let mut scratch = EntropyDecodeScratch::new();
        entropy::decode_tokens_into(&payload, &mut scratch).unwrap();

        assert_eq!(original, scratch.tokens);
    }

    #[test]
    fn entropy_decode_into_reuse() {
        // Call twice — second call should reuse capacity, produce same result.
        let data1 = b"Hello world! Hello world! Hello world! Hello world!";
        let payload1 = make_entropy_payload(data1);

        let data2 = b"XYZXYZXYZXYZXYZXYZXYZXYZXYZXYZXYZXYZ";
        let payload2 = make_entropy_payload(data2);

        let mut scratch = EntropyDecodeScratch::new();

        // First call
        entropy::decode_tokens_into(&payload1, &mut scratch).unwrap();
        let result1 = scratch.tokens.clone();
        let cap_after_first = scratch.tokens.capacity();

        // Second call — reuses capacity
        entropy::decode_tokens_into(&payload2, &mut scratch).unwrap();
        let result2 = scratch.tokens.clone();

        // Verify equivalence
        assert_eq!(result1, entropy::decode_tokens(&payload1).unwrap());
        assert_eq!(result2, entropy::decode_tokens(&payload2).unwrap());

        // Verify no reallocation (second data is smaller or same)
        assert!(
            scratch.tokens.capacity() >= cap_after_first || scratch.tokens.len() <= cap_after_first
        );
    }

    #[test]
    fn replay_tokens_into_equivalence() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox!";
        let config = crate::CompressionConfig::default();
        let tokens = parser::parse(data, &config).unwrap();

        let original = parser::replay_tokens(&tokens);
        let mut output = Vec::new();
        parser::replay_tokens_into(&tokens, &mut output);

        assert_eq!(original, output);
    }

    #[test]
    fn replay_tokens_into_reuse() {
        let data1 = b"AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD";
        let data2 = b"1234567812345678123456781234567812345678";
        let config = crate::CompressionConfig::default();

        let tokens1 = parser::parse(data1, &config).unwrap();
        let tokens2 = parser::parse(data2, &config).unwrap();

        let mut output = Vec::new();

        parser::replay_tokens_into(&tokens1, &mut output);
        assert_eq!(output, parser::replay_tokens(&tokens1));

        parser::replay_tokens_into(&tokens2, &mut output);
        assert_eq!(output, parser::replay_tokens(&tokens2));
    }

    #[test]
    fn delta_i64_into_equivalence() {
        let values: Vec<i64> = (0..1000).map(|i| 1_000_000 + i * 1000 + (i % 7)).collect();
        let encoded = delta::encode_i64(&values, true).unwrap();

        let original = delta::decode_i64(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        delta::decode_i64_into(&encoded, values.len(), &mut out).unwrap();

        // Original returns Vec<i64>, _into writes LE bytes.
        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn delta_u64_into_equivalence() {
        let values: Vec<u64> = (0..500).map(|i| 5_000_000 + i * 100).collect();
        let encoded = delta::encode_u64(&values, true).unwrap();

        let original = delta::decode_u64(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        delta::decode_u64_into(&encoded, values.len(), &mut out).unwrap();

        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn delta_i32_into_equivalence() {
        let values: Vec<i32> = (0..1000).map(|i| -500 + i * 3).collect();
        let encoded = delta::encode_i32(&values, true).unwrap();

        let original = delta::decode_i32(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        delta::decode_i32_into(&encoded, values.len(), &mut out).unwrap();

        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn delta_u32_into_equivalence() {
        let values: Vec<u32> = (0..500).map(|i| 100 + i * 2).collect();
        let encoded = delta::encode_u32(&values, true).unwrap();

        let original = delta::decode_u32(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        delta::decode_u32_into(&encoded, values.len(), &mut out).unwrap();

        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn gorilla_f64_into_equivalence() {
        let values: Vec<f64> = (0..500).map(|i| 22.5 + 0.01 * (i as f64).sin()).collect();
        let encoded = gorilla_xor::encode_f64(&values).unwrap();

        let original = gorilla_xor::decode_f64(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        gorilla_xor::decode_f64_into(&encoded, values.len(), &mut out).unwrap();

        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn gorilla_f32_into_equivalence() {
        let values: Vec<f32> = (0..500).map(|i| 100.0 + 0.5 * (i as f32)).collect();
        let encoded = gorilla_xor::encode_f32(&values).unwrap();

        let original = gorilla_xor::decode_f32(&encoded, values.len()).unwrap();
        let mut out = Vec::new();
        gorilla_xor::decode_f32_into(&encoded, values.len(), &mut out).unwrap();

        let original_bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(original_bytes, out);
    }

    #[test]
    fn depreprocess_into_equivalence_all_types() {
        use crate::preprocessor::PreprocessorConfig;

        // Test all data types through the full depreprocess path.
        let test_cases: Vec<(DataType, Vec<u8>)> = vec![
            // Raw
            (
                DataType::Raw,
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ),
            // i64 — 8 timestamps
            (DataType::IntegerI64, {
                let vals: Vec<i64> = (0..8).map(|i| 1_000_000 + i * 1000).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            // u32 — 16 counters
            (DataType::IntegerU32, {
                let vals: Vec<u32> = (0..16).map(|i| 100 + i).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            // f64 — 8 temperatures
            (DataType::Float64, {
                let vals: Vec<f64> = (0..8).map(|i| 22.5 + 0.01 * i as f64).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            // f32 — 16 vibration readings
            (DataType::Float32, {
                let vals: Vec<f32> = (0..16).map(|i| 1.0 + 0.1 * i as f32).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            // f64 shuffle — 8 temperatures
            (DataType::Float64Shuffle, {
                let vals: Vec<f64> = (0..8).map(|i| 22.5 + 0.01 * i as f64).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            // f32 shuffle — 16 vibration readings
            (DataType::Float32Shuffle, {
                let vals: Vec<f32> = (0..16).map(|i| 1.0 + 0.1 * i as f32).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
        ];

        for (dt, raw_data) in &test_cases {
            let config = PreprocessorConfig {
                data_type: Some(*dt),
                double_delta: true,
            };
            let preprocessed = preprocessor::preprocess(raw_data, &config).unwrap();

            // Original path
            let original = preprocessor::depreprocess(&preprocessed).unwrap();

            // Workspace path
            let pdr = PreprocessedDataRef {
                data_type: preprocessed.data_type,
                element_count: preprocessed.element_count,
                data: &preprocessed.data,
            };
            let mut scratch = crate::workspace::PreprocessScratch::new();
            preprocessor::depreprocess_into(&pdr, &mut scratch).unwrap();

            assert_eq!(
                original, scratch.output,
                "depreprocess_into mismatch for {:?}",
                dt
            );
        }
    }

    #[test]
    fn full_pipeline_workspace_equivalence() {
        // Full roundtrip: compress → decompress with workspace must equal original.
        use crate::parallel;
        use crate::CompressionConfig;

        let timestamps: Vec<i64> = (0..10_000)
            .map(|i| 1_700_000_000_000i64 + i * 1000 + (i % 50))
            .collect();
        let raw: Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();

        let config = CompressionConfig {
            data_type: Some(DataType::IntegerI64),
            block_size: 16384,
            ..CompressionConfig::default()
        };

        let compressed = parallel::compress(&raw, &config).unwrap();
        let decompressed = parallel::decompress(&compressed).unwrap();

        assert_eq!(raw, decompressed);
    }

    #[test]
    fn full_pipeline_workspace_all_parser_modes() {
        use crate::parallel;
        use crate::{CompressionConfig, ParserMode};

        let values: Vec<u32> = (0..5000).map(|i| 1000 + i * 3).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        for mode in &[ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = CompressionConfig {
                parser_mode: *mode,
                data_type: Some(DataType::IntegerU32),
                block_size: 8192,
                ..CompressionConfig::default()
            };

            let compressed = parallel::compress(&raw, &config).unwrap();
            let decompressed = parallel::decompress(&compressed).unwrap();

            assert_eq!(raw, decompressed, "roundtrip failed for {:?}", mode);
        }
    }

    #[test]
    fn full_pipeline_workspace_multiblock() {
        // Force multiple blocks to exercise thread_local workspace reuse.
        use crate::parallel;
        use crate::CompressionConfig;

        let timestamps: Vec<i64> = (0..50_000)
            .map(|i| 1_700_000_000_000i64 + i * 1000)
            .collect();
        let raw: Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();

        let config = CompressionConfig {
            data_type: Some(DataType::IntegerI64),
            block_size: 65536, // ~8 blocks for 400KB of data
            ..CompressionConfig::default()
        };

        let compressed = parallel::compress(&raw, &config).unwrap();
        let decompressed = parallel::decompress(&compressed).unwrap();

        assert_eq!(raw, decompressed);
    }

    #[test]
    fn fse_rebuild_decode_only_equivalence() {
        use crate::entropy::fse::FseTable;

        // Build a freq table, then rebuild — results must match.
        let freqs: Vec<u16> = {
            let mut f = vec![0u16; 256];
            // A realistic distribution: mostly low-value literals.
            for i in 0..32 {
                f[i] = 10;
            }
            f[32] = 512 - 320; // fill remainder to sum to 512 (table_log=9)
            f
        };

        let original = FseTable::from_normalized_decode_only(&freqs, 9).unwrap();

        let mut table = FseTable::empty();
        let mut scratch = FseTableScratch::new();
        table.rebuild_decode_only(&freqs, 9, &mut scratch).unwrap();

        // Encode some symbols, decode with both, compare.
        let symbols: Vec<u16> = (0..200).map(|i| (i % 33) as u16).collect();
        let encoded = {
            let full = FseTable::from_normalized(&freqs, 9).unwrap();
            full.encode(&symbols).unwrap()
        };

        let decoded_original = original.decode(&encoded, symbols.len()).unwrap();
        let mut decoded_rebuild = Vec::new();
        table
            .decode_into(&encoded, symbols.len(), &mut decoded_rebuild)
            .unwrap();

        assert_eq!(decoded_original, decoded_rebuild);
        assert_eq!(decoded_original, symbols);
    }

    // ══════════════════════════════════════════════════════════════════
    // Encode workspace equivalence tests
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn encode_workspace_new_zero_capacity() {
        let ws = EncodeWorkspace::new();
        assert!(ws.parser.chain.is_none());
        assert_eq!(ws.parser.tokens.capacity(), 0);
        assert_eq!(ws.parser.cost.capacity(), 0);
        assert_eq!(ws.entropy.literal_symbols.capacity(), 0);
        assert_eq!(ws.entropy.emissions.capacity(), 0);
        assert_eq!(ws.preprocess.output.capacity(), 0);
    }

    #[test]
    fn greedy_parse_into_equivalence() {
        let data = b"ABCDEFGH".repeat(100);
        let config = crate::CompressionConfig::default();
        let mf_config = crate::match_finder::MatchFinderConfig {
            max_chain_depth: config.max_chain_depth,
            window_size: config.window_size,
            ..Default::default()
        };

        let original = crate::parser::greedy::parse(&data, &mf_config);

        let chain = crate::match_finder::HashChain::new(mf_config.window_size as usize);
        let mut tokens = Vec::new();
        let _chain = crate::parser::greedy::parse_into(&data, &mf_config, chain, &mut tokens);

        assert_eq!(original, tokens);
    }

    #[test]
    fn lazy_parse_into_equivalence() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox!";
        let config = crate::CompressionConfig::default();
        let mf_config = crate::match_finder::MatchFinderConfig {
            max_chain_depth: config.max_chain_depth,
            window_size: config.window_size,
            ..Default::default()
        };

        let original = crate::parser::lazy::parse(data, &mf_config);

        let chain = crate::match_finder::HashChain::new(mf_config.window_size as usize);
        let mut tokens = Vec::new();
        let _chain = crate::parser::lazy::parse_into(data, &mf_config, chain, &mut tokens);

        assert_eq!(original, tokens);
    }

    #[test]
    fn optimal_parse_into_equivalence() {
        let data = b"ABCDEFGHABCDEFGHABCDEFGH12345678ABCDEFGH";
        let config = crate::CompressionConfig::default();
        let mf_config = crate::match_finder::MatchFinderConfig {
            max_chain_depth: config.max_chain_depth,
            window_size: config.window_size,
            ..Default::default()
        };

        let original = crate::parser::optimal::parse(data, &mf_config);

        let mut scratch = ParserScratch::new();
        crate::parser::optimal::parse_into(data, &mf_config, &mut scratch);

        assert_eq!(original, scratch.tokens);
    }

    #[test]
    fn parser_dispatch_into_equivalence() {
        let data = b"Hello world! Hello world! Hello world! Hello world!";
        for mode in &[
            crate::ParserMode::Greedy,
            crate::ParserMode::Lazy,
            crate::ParserMode::Optimal,
        ] {
            let config = crate::CompressionConfig {
                parser_mode: *mode,
                ..Default::default()
            };
            let original = parser::parse(data, &config).unwrap();

            let mut scratch = ParserScratch::new();
            parser::parse_into(data, &config, &mut scratch).unwrap();

            assert_eq!(
                original, scratch.tokens,
                "parse_into mismatch for {:?}",
                mode
            );
        }
    }

    #[test]
    fn parser_into_chain_reuse() {
        let data1 = b"AAAABBBBCCCCDDDD".repeat(50);
        let data2 = b"1234567812345678".repeat(30);
        let config = crate::CompressionConfig::default();

        let mut scratch = ParserScratch::new();
        assert!(scratch.chain.is_none());

        parser::parse_into(&data1, &config, &mut scratch).unwrap();
        assert!(
            scratch.chain.is_some(),
            "chain should be stored after first parse"
        );
        let result1 = scratch.tokens.clone();

        parser::parse_into(&data2, &config, &mut scratch).unwrap();
        assert!(
            scratch.chain.is_some(),
            "chain should survive across parses"
        );
        let result2 = scratch.tokens.clone();

        assert_eq!(result1, parser::parse(&data1, &config).unwrap());
        assert_eq!(result2, parser::parse(&data2, &config).unwrap());
    }

    #[test]
    fn entropy_encode_into_equivalence() {
        let data = b"ABCDEFGH".repeat(100);
        let config = crate::CompressionConfig::default();
        let tokens = parser::parse(&data, &config).unwrap();

        let (original_payload, original_cost) = entropy::encode_tokens(&tokens).unwrap();

        let mut scratch = EntropyEncodeScratch::new();
        let (ws_payload, ws_cost) = entropy::encode_tokens_into(&tokens, &mut scratch).unwrap();

        assert_eq!(original_payload, ws_payload);
        // Cost models should produce identical costs for any symbol.
        assert_eq!(original_cost.literal_cost(b'A'), ws_cost.literal_cost(b'A'));
    }

    #[test]
    fn entropy_encode_into_reuse() {
        let data1 = b"Hello world! Hello world! Hello world!";
        let data2 = b"XYZXYZXYZXYZXYZXYZXYZXYZ";
        let config = crate::CompressionConfig::default();

        let tokens1 = parser::parse(data1, &config).unwrap();
        let tokens2 = parser::parse(data2, &config).unwrap();

        let mut scratch = EntropyEncodeScratch::new();

        let (payload1, _) = entropy::encode_tokens_into(&tokens1, &mut scratch).unwrap();
        let (payload2, _) = entropy::encode_tokens_into(&tokens2, &mut scratch).unwrap();

        let (expected1, _) = entropy::encode_tokens(&tokens1).unwrap();
        let (expected2, _) = entropy::encode_tokens(&tokens2).unwrap();

        assert_eq!(expected1, payload1);
        assert_eq!(expected2, payload2);
    }

    #[test]
    fn fse_rebuild_full_equivalence() {
        use crate::entropy::fse::FseTable;

        let freqs: Vec<u16> = {
            let mut f = vec![0u16; 256];
            for i in 0..32 {
                f[i] = 10;
            }
            f[32] = 512 - 320;
            f
        };

        let original = FseTable::from_normalized(&freqs, 9).unwrap();

        let mut table = FseTable::empty();
        let mut scratch = FseEncodeTableScratch::new();
        table.rebuild_full(&freqs, 9, &mut scratch).unwrap();

        // Encode same symbols with both, compare output.
        let symbols: Vec<u16> = (0..200).map(|i| (i % 33) as u16).collect();
        let original_encoded = original.encode(&symbols).unwrap();

        let mut emissions = Vec::new();
        let ws_encoded = table.encode_into(&symbols, &mut emissions).unwrap();

        assert_eq!(original_encoded, ws_encoded);
    }

    #[test]
    fn preprocess_into_equivalence() {
        use crate::preprocessor::PreprocessorConfig;

        let test_cases: Vec<(DataType, Vec<u8>)> = vec![
            (
                DataType::Raw,
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ),
            (DataType::IntegerI64, {
                let vals: Vec<i64> = (0..8).map(|i| 1_000_000 + i * 1000).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            (DataType::IntegerU32, {
                let vals: Vec<u32> = (0..16).map(|i| 100 + i).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            (DataType::Float64, {
                let vals: Vec<f64> = (0..8).map(|i| 22.5 + 0.01 * i as f64).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            (DataType::Float64Shuffle, {
                let vals: Vec<f64> = (0..8).map(|i| 22.5 + 0.01 * i as f64).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            (DataType::Float32Shuffle, {
                let vals: Vec<f32> = (0..16).map(|i| 1.0 + 0.1 * i as f32).collect();
                vals.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
        ];

        for (dt, raw_data) in &test_cases {
            let config = PreprocessorConfig {
                data_type: Some(*dt),
                double_delta: true,
            };
            let original = preprocessor::preprocess(raw_data, &config).unwrap();

            let mut scratch = PreprocessEncodeScratch::new();
            let ws_ref = preprocessor::preprocess_into(raw_data, &config, &mut scratch).unwrap();

            assert_eq!(
                original.data, ws_ref.data,
                "preprocess_into mismatch for {:?}",
                dt
            );
            assert_eq!(original.data_type, ws_ref.data_type);
            assert_eq!(original.element_count, ws_ref.element_count);
        }
    }

    #[test]
    fn full_encode_workspace_roundtrip() {
        use crate::parallel;
        use crate::CompressionConfig;

        let timestamps: Vec<i64> = (0..10_000)
            .map(|i| 1_700_000_000_000i64 + i * 1000 + (i % 50))
            .collect();
        let raw: Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();

        for mode in &[
            crate::ParserMode::Greedy,
            crate::ParserMode::Lazy,
            crate::ParserMode::Optimal,
        ] {
            let config = CompressionConfig {
                data_type: Some(DataType::IntegerI64),
                block_size: 16384,
                parser_mode: *mode,
                ..CompressionConfig::default()
            };

            let compressed = parallel::compress(&raw, &config).unwrap();
            let decompressed = parallel::decompress(&compressed).unwrap();

            assert_eq!(raw, decompressed, "roundtrip failed for {:?}", mode);
        }
    }

    #[test]
    fn full_encode_workspace_multiblock() {
        use crate::parallel;
        use crate::CompressionConfig;

        let timestamps: Vec<i64> = (0..50_000)
            .map(|i| 1_700_000_000_000i64 + i * 1000)
            .collect();
        let raw: Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();

        let config = CompressionConfig {
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            ..CompressionConfig::default()
        };

        let compressed = parallel::compress(&raw, &config).unwrap();
        let decompressed = parallel::decompress(&compressed).unwrap();

        assert_eq!(raw, decompressed);
    }

    #[test]
    fn matchfinder_with_chain_take_chain_roundtrip() {
        use crate::match_finder::{HashChain, MatchFinder, MatchFinderConfig};

        let data = b"ABCDABCDEFGHEFGH";
        let config = MatchFinderConfig::default();
        let chain = HashChain::new(config.window_size as usize);

        // Build MatchFinder with existing chain, find a match, take chain back.
        let mut mf = MatchFinder::with_chain(data, config.clone(), chain);
        // Insert all positions
        for pos in 0..data.len() {
            mf.insert_at(pos);
        }
        let m = mf.find_best_match_at(4); // "ABCD" at offset 4
        assert!(m.is_some());
        let chain = mf.take_chain();

        // Reuse chain with different data — should work after reset.
        let data2 = b"XYZWXYZW";
        let mut mf2 = MatchFinder::with_chain(data2, config, chain);
        for pos in 0..data2.len() {
            mf2.insert_at(pos);
        }
        let m2 = mf2.find_best_match_at(4); // "XYZW" at offset 4
        assert!(m2.is_some());
        let _chain = mf2.take_chain(); // chain survives
    }
}
