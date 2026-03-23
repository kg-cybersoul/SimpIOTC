//! # IoT Time-Series Compressor
//!
//! A high-speed, high-ratio LZ77/ANS compression library purpose-built for
//! sensor and IoT time-series data. Achieves XZ-class compression ratios on
//! structured numeric streams while maintaining zstd-class throughput via
//! SIMD-accelerated match finding and parallel block processing.

pub mod entropy;
pub mod match_finder;
pub mod preprocessor;

pub mod parallel;
pub mod parser;

#[doc(hidden)]
pub mod harness;

pub mod schema;
pub mod seekable;
pub mod workspace;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "python")]
pub mod python;

use std::fmt;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════════════
// Magic & Version Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Magic bytes identifying an IoT compressor frame: "IOTC" in ASCII.
pub const FRAME_MAGIC: [u8; 4] = [0x49, 0x4F, 0x54, 0x43];

/// Current frame format version.
pub const FRAME_VERSION: u8 = 1;

/// Default block size for parallel compression (2 MiB).
pub const DEFAULT_BLOCK_SIZE: usize = 2 * 1024 * 1024;

/// Minimum match length for LZ77 (below this, a literal is cheaper).
pub const MIN_MATCH_LEN: u32 = 4;

/// Maximum match length representable in our token format.
pub const MAX_MATCH_LEN: u32 = 65535;

/// Maximum match offset (window size). 4 MiB sliding window.
pub const MAX_MATCH_OFFSET: u32 = 4 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════════
// Error Types
// ═══════════════════════════════════════════════════════════════════════════════

/// All errors that can occur during compression or decompression.
#[derive(Error, Debug)]
pub enum CompressorError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid magic bytes: expected {expected:?}, got {got:?}")]
    InvalidMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("Unsupported frame version: {0} (max supported: {FRAME_VERSION})")]
    UnsupportedVersion(u8),

    #[error("Checksum mismatch: expected 0x{expected:08X}, computed 0x{computed:08X}")]
    ChecksumMismatch { expected: u32, computed: u32 },

    #[error("SHA-256 integrity check failed on decompressed output")]
    IntegrityCheckFailed,

    #[error("Preprocessor error: {0}")]
    Preprocessor(String),

    #[error("Corrupted block at offset {offset}: {detail}")]
    CorruptedBlock { offset: u64, detail: String },

    #[error("Buffer underflow: needed {needed} bytes, only {available} remain")]
    BufferUnderflow { needed: usize, available: usize },

    #[error("Invalid delta-of-delta stream: {0}")]
    DeltaDecode(String),

    #[error("Invalid Gorilla XOR stream: {0}")]
    GorillaDecode(String),

    #[error("Varint decode overflow at byte position {0}")]
    VarintOverflow(usize),

    #[error("Empty input")]
    EmptyInput,

    #[error("Data type mismatch: element size {element_size} does not divide buffer length {buffer_len}")]
    DataTypeMismatch {
        element_size: usize,
        buffer_len: usize,
    },

    #[error("Stride mismatch: stride {stride} does not divide buffer length {buffer_len}")]
    StrideMismatch { stride: usize, buffer_len: usize },

    #[error("Seek table CRC mismatch: expected 0x{expected:08X}, computed 0x{computed:08X}")]
    SeekTableCrcMismatch { expected: u32, computed: u32 },

    #[error("Seek table offset out of bounds: entry[{index}] = {offset} exceeds frame size {frame_size}")]
    SeekTableOffsetOutOfBounds {
        index: usize,
        offset: u64,
        frame_size: u64,
    },

    #[error("Block index {index} out of range (block_count = {block_count})")]
    BlockIndexOutOfRange { index: usize, block_count: u32 },

    #[error("Byte range [{start}, {end}) out of bounds (original_size = {original_size})")]
    ByteRangeOutOfBounds {
        start: u64,
        end: u64,
        original_size: u64,
    },

    #[error("Invalid block size: {size} — {reason}")]
    InvalidBlockSize { size: usize, reason: &'static str },
}

pub type Result<T> = std::result::Result<T, CompressorError>;

// ═══════════════════════════════════════════════════════════════════════════════
// LZ77 Token Representation
// ═══════════════════════════════════════════════════════════════════════════════

/// A single LZ77 token — either a raw literal byte or a back-reference match.
///
/// The parser produces a stream of these tokens, which the entropy coder then
/// packs into the final bitstream. The `Match` variant encodes a (length, offset)
/// pair referring to previously seen data within the sliding window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LzToken {
    /// A single literal byte that could not be matched.
    Literal(u8),
    /// A back-reference: copy `length` bytes from `offset` bytes behind
    /// the current position.
    Match {
        /// Distance backwards in the output stream. Must be >= 1.
        offset: u32,
        /// Number of bytes to copy. Must be >= MIN_MATCH_LEN.
        length: u32,
    },
}

impl LzToken {
    /// Returns the number of raw bytes this token represents in the
    /// uncompressed stream.
    #[inline]
    pub fn uncompressed_size(&self) -> u32 {
        match self {
            LzToken::Literal(_) => 1,
            LzToken::Match { length, .. } => *length,
        }
    }

    /// Returns true if this token is a literal.
    #[inline]
    pub fn is_literal(&self) -> bool {
        matches!(self, LzToken::Literal(_))
    }

    /// Returns true if this token is a match.
    #[inline]
    pub fn is_match(&self) -> bool {
        matches!(self, LzToken::Match { .. })
    }
}

impl fmt::Display for LzToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LzToken::Literal(b) => write!(f, "Lit(0x{:02X})", b),
            LzToken::Match { offset, length } => {
                write!(f, "Match(off={}, len={})", offset, length)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compression Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsing strategy for the LZ77 engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParserMode {
    /// Emit the first match found at each position. Fastest, lowest ratio.
    Greedy,
    /// Check one position ahead before committing to a match. Balanced.
    #[default]
    Lazy,
    /// Use FSE bit-cost models to find the optimal parse. Slowest, highest ratio.
    Optimal,
}

impl fmt::Display for ParserMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserMode::Greedy => write!(f, "greedy"),
            ParserMode::Lazy => write!(f, "lazy"),
            ParserMode::Optimal => write!(f, "optimal"),
        }
    }
}

/// What kind of time-series data we expect. Determines which preprocessor
/// is applied before LZ77.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Raw bytes — no preprocessing. Fallback for non-numeric data.
    Raw,
    /// Stream of signed 64-bit integers (timestamps, counters, etc.).
    /// Delta-of-delta + zigzag varint encoding is applied.
    IntegerI64,
    /// Stream of unsigned 64-bit integers.
    /// Delta-of-delta + zigzag varint encoding is applied.
    IntegerU64,
    /// Stream of signed 32-bit integers.
    IntegerI32,
    /// Stream of unsigned 32-bit integers.
    IntegerU32,
    /// Stream of IEEE 754 double-precision floats (sensor readings).
    /// Gorilla XOR encoding is applied.
    Float64,
    /// Stream of IEEE 754 single-precision floats.
    /// Gorilla XOR encoding is applied.
    Float32,
    /// Stream of IEEE 754 double-precision floats.
    /// Byte-level shuffle (bit-shuffle) encoding is applied — groups all
    /// byte-0s, byte-1s, ..., byte-7s together for superior LZ77 matching
    /// on noisy sensor data where Gorilla XOR degrades.
    Float64Shuffle,
    /// Stream of IEEE 754 single-precision floats.
    /// Byte-level shuffle encoding is applied.
    Float32Shuffle,
    /// Stream of IEEE 754 double-precision floats.
    /// Byte shuffle + byte-delta composition: shuffles byte lanes then applies
    /// wrapping delta within each lane. Exponent bytes become near-constant
    /// delta streams (many zeros), giving LZ77 long matches that plain shuffle
    /// or Gorilla XOR cannot achieve on noisy sensor data.
    Float64ShuffleDelta,
    /// Stream of IEEE 754 single-precision floats.
    /// Byte shuffle + byte-delta composition.
    Float32ShuffleDelta,
}

impl DataType {
    /// Returns the byte width of a single element for this data type.
    /// Returns `None` for `Raw` since element size is 1 byte (passthrough).
    pub fn element_size(&self) -> usize {
        match self {
            DataType::Raw => 1,
            DataType::IntegerI64
            | DataType::IntegerU64
            | DataType::Float64
            | DataType::Float64Shuffle
            | DataType::Float64ShuffleDelta => 8,
            DataType::IntegerI32
            | DataType::IntegerU32
            | DataType::Float32
            | DataType::Float32Shuffle
            | DataType::Float32ShuffleDelta => 4,
        }
    }

    /// Returns true if this type uses delta-of-delta preprocessing.
    pub fn uses_delta(&self) -> bool {
        matches!(
            self,
            DataType::IntegerI64
                | DataType::IntegerU64
                | DataType::IntegerI32
                | DataType::IntegerU32
        )
    }

    /// Returns true if this type uses Gorilla XOR preprocessing.
    pub fn uses_gorilla(&self) -> bool {
        matches!(self, DataType::Float64 | DataType::Float32)
    }

    /// Returns true if this type uses byte-level shuffle preprocessing.
    pub fn uses_shuffle(&self) -> bool {
        matches!(
            self,
            DataType::Float64Shuffle
                | DataType::Float32Shuffle
                | DataType::Float64ShuffleDelta
                | DataType::Float32ShuffleDelta
        )
    }

    /// Returns true if this type uses the composed shuffle+byte-delta transform.
    pub fn uses_shuffle_delta(&self) -> bool {
        matches!(
            self,
            DataType::Float64ShuffleDelta | DataType::Float32ShuffleDelta
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Raw => write!(f, "raw"),
            DataType::IntegerI64 => write!(f, "i64"),
            DataType::IntegerU64 => write!(f, "u64"),
            DataType::IntegerI32 => write!(f, "i32"),
            DataType::IntegerU32 => write!(f, "u32"),
            DataType::Float64 => write!(f, "f64"),
            DataType::Float32 => write!(f, "f32"),
            DataType::Float64Shuffle => write!(f, "f64s"),
            DataType::Float32Shuffle => write!(f, "f32s"),
            DataType::Float64ShuffleDelta => write!(f, "f64sd"),
            DataType::Float32ShuffleDelta => write!(f, "f32sd"),
        }
    }
}

/// Full compression configuration.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// LZ77 parser strategy.
    pub parser_mode: ParserMode,
    /// Block size for parallel compression. Each block is compressed independently.
    pub block_size: usize,
    /// Number of worker threads. 0 = use all available cores.
    pub num_threads: usize,
    /// Data type hint for the preprocessor. If `None`, auto-detection is attempted.
    pub data_type: Option<DataType>,
    /// LZ77 sliding window size in bytes. Capped at MAX_MATCH_OFFSET.
    pub window_size: u32,
    /// Maximum hash chain depth for the match finder. Higher = better ratio, slower.
    pub max_chain_depth: u32,
    /// Whether to compute and store a SHA-256 digest for integrity verification.
    pub store_checksum: bool,
    /// Optional stride for columnar transposition. When set, input data is treated
    /// as an array of fixed-size records of `stride` bytes each. Before LZ77, the
    /// data is transposed column-wise (byte 0 of all records grouped, byte 1, etc.)
    /// to improve match quality and repcode utilization on structured data.
    /// Must divide the input data length evenly.
    pub stride: Option<u16>,
    /// Whether to write a seek table after the frame header for O(1) block lookup.
    pub store_seek_table: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            parser_mode: ParserMode::Lazy,
            block_size: DEFAULT_BLOCK_SIZE,
            num_threads: 0,
            data_type: None,
            window_size: MAX_MATCH_OFFSET,
            max_chain_depth: 64,
            store_checksum: true,
            stride: None,
            store_seek_table: true,
        }
    }
}

impl CompressionConfig {
    /// Preset for maximum speed, lowest ratio.
    pub fn fast() -> Self {
        Self {
            parser_mode: ParserMode::Greedy,
            max_chain_depth: 8,
            ..Default::default()
        }
    }

    /// Preset for balanced speed/ratio (default).
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Preset for maximum compression ratio, slowest.
    pub fn max_compression() -> Self {
        Self {
            parser_mode: ParserMode::Optimal,
            max_chain_depth: 256,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Frame & Block Headers (Wire Format)
// ═══════════════════════════════════════════════════════════════════════════════

/// Flags packed into the frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameFlags {
    /// The preprocessor applied to data before LZ77.
    pub data_type: DataType,
    /// The parser mode used during compression.
    pub parser_mode: ParserMode,
    /// Whether a SHA-256 content checksum follows the last block.
    pub has_content_checksum: bool,
    /// Whether the offset alphabet includes repcode slots (codes 0–2).
    /// When true, the offset FSE alphabet is 47 codes (3 repcodes + 44 real).
    /// When false, the legacy 44-code alphabet is used (unsupported by this decoder).
    pub has_repcodes: bool,
    /// Whether a seek table follows the frame header (before block data).
    pub has_seek_table: bool,
}

impl FrameFlags {
    /// Serialize flags into a single u16 for the wire format.
    pub fn to_u16(&self) -> u16 {
        let dt: u16 = match self.data_type {
            DataType::Raw => 0,
            DataType::IntegerI64 => 1,
            DataType::IntegerU64 => 2,
            DataType::IntegerI32 => 3,
            DataType::IntegerU32 => 4,
            DataType::Float64 => 5,
            DataType::Float32 => 6,
            DataType::Float64Shuffle => 7,
            DataType::Float32Shuffle => 8,
            DataType::Float64ShuffleDelta => 9,
            DataType::Float32ShuffleDelta => 10,
        };
        let pm: u16 = match self.parser_mode {
            ParserMode::Greedy => 0,
            ParserMode::Lazy => 1,
            ParserMode::Optimal => 2,
        };
        let ck: u16 = if self.has_content_checksum { 1 } else { 0 };
        let rc: u16 = if self.has_repcodes { 1 } else { 0 };
        let st: u16 = if self.has_seek_table { 1 } else { 0 };
        dt | (pm << 4) | (ck << 8) | (rc << 9) | (st << 10)
    }

    /// Deserialize flags from a u16.
    pub fn from_u16(bits: u16) -> Result<Self> {
        // Reject unknown high bits — a future format version would set these,
        // and silently ignoring them risks misinterpreting the frame.
        if bits & 0xF800 != 0 {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: format!("unknown flags in high bits: 0x{:04X}", bits & 0xF800),
            });
        }
        let dt = match bits & 0x0F {
            0 => DataType::Raw,
            1 => DataType::IntegerI64,
            2 => DataType::IntegerU64,
            3 => DataType::IntegerI32,
            4 => DataType::IntegerU32,
            5 => DataType::Float64,
            6 => DataType::Float32,
            7 => DataType::Float64Shuffle,
            8 => DataType::Float32Shuffle,
            9 => DataType::Float64ShuffleDelta,
            10 => DataType::Float32ShuffleDelta,
            other => {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!("unknown data type flag: {}", other),
                })
            }
        };
        let pm = match (bits >> 4) & 0x0F {
            0 => ParserMode::Greedy,
            1 => ParserMode::Lazy,
            2 => ParserMode::Optimal,
            other => {
                return Err(CompressorError::CorruptedBlock {
                    offset: 0,
                    detail: format!("unknown parser mode flag: {}", other),
                })
            }
        };
        let ck = ((bits >> 8) & 1) == 1;
        let rc = ((bits >> 9) & 1) == 1;
        let st = ((bits >> 10) & 1) == 1;
        Ok(Self {
            data_type: dt,
            parser_mode: pm,
            has_content_checksum: ck,
            has_repcodes: rc,
            has_seek_table: st,
        })
    }
}

/// The frame header written at the start of every compressed stream.
///
/// Wire layout (little-endian):
/// ```text
/// [0..4]   magic: "IOTC"
/// [4]      version: u8
/// [5..7]   flags: u16  (data type, parser mode, checksum flag, repcodes)
/// [7..11]  block_size: u32
/// [11..19] original_size: u64  (total uncompressed bytes)
/// [19..23] block_count: u32
/// [23..25] stride: u16  (0 = no stride transposition)
/// ```
/// Total: 25 bytes fixed header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: u8,
    pub flags: FrameFlags,
    pub block_size: u32,
    pub original_size: u64,
    pub block_count: u32,
    /// Stride for columnar transposition. 0 means no transposition.
    pub stride: u16,
}

impl FrameHeader {
    pub const SERIALIZED_SIZE: usize = 25;

    /// Serialize the frame header to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..4].copy_from_slice(&FRAME_MAGIC);
        buf[4] = self.version;
        let flags = self.flags.to_u16();
        buf[5..7].copy_from_slice(&flags.to_le_bytes());
        buf[7..11].copy_from_slice(&self.block_size.to_le_bytes());
        buf[11..19].copy_from_slice(&self.original_size.to_le_bytes());
        buf[19..23].copy_from_slice(&self.block_count.to_le_bytes());
        buf[23..25].copy_from_slice(&self.stride.to_le_bytes());
        buf
    }

    /// Deserialize a frame header from bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SERIALIZED_SIZE {
            return Err(CompressorError::BufferUnderflow {
                needed: Self::SERIALIZED_SIZE,
                available: buf.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != FRAME_MAGIC {
            return Err(CompressorError::InvalidMagic {
                expected: FRAME_MAGIC,
                got: magic,
            });
        }
        let version = buf[4];
        if version > FRAME_VERSION {
            return Err(CompressorError::UnsupportedVersion(version));
        }
        let flags_bits = u16::from_le_bytes([buf[5], buf[6]]);
        let flags = FrameFlags::from_u16(flags_bits)?;
        let block_size = u32::from_le_bytes([buf[7], buf[8], buf[9], buf[10]]);
        let original_size = u64::from_le_bytes([
            buf[11], buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18],
        ]);
        let block_count = u32::from_le_bytes([buf[19], buf[20], buf[21], buf[22]]);
        let stride = u16::from_le_bytes([buf[23], buf[24]]);

        Ok(Self {
            version,
            flags,
            block_size,
            original_size,
            block_count,
            stride,
        })
    }
}

/// Per-block header preceding each compressed block in the stream.
///
/// Wire layout (little-endian):
/// ```text
/// [0..4]   compressed_size: u32   (size of compressed payload)
/// [4..8]   original_size: u32     (uncompressed size of this block)
/// [8..12]  crc32: u32             (CRC-32 of the compressed payload)
/// ```
/// Total: 12 bytes per block header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHeader {
    pub compressed_size: u32,
    pub original_size: u32,
    pub crc32: u32,
}

impl BlockHeader {
    pub const SERIALIZED_SIZE: usize = 12;

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..4].copy_from_slice(&self.compressed_size.to_le_bytes());
        buf[4..8].copy_from_slice(&self.original_size.to_le_bytes());
        buf[8..12].copy_from_slice(&self.crc32.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SERIALIZED_SIZE {
            return Err(CompressorError::BufferUnderflow {
                needed: Self::SERIALIZED_SIZE,
                available: buf.len(),
            });
        }
        Ok(Self {
            compressed_size: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            original_size: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            crc32: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Seek Table
// ═══════════════════════════════════════════════════════════════════════════════

/// A seek table mapping block indices to their byte offsets within the frame.
///
/// Stored immediately after the FrameHeader (when `has_seek_table` flag is set),
/// before block data. Enables O(1) random access to any compressed block.
///
/// Wire layout (little-endian):
/// ```text
/// [u64 LE; block_count]   absolute byte offsets from frame start to each BlockHeader
/// [u32 LE]                CRC32 over the entries array
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekTable {
    /// Absolute byte offsets from frame start to each BlockHeader.
    pub entries: Vec<u64>,
    /// CRC32 of the serialized entries array.
    pub crc32: u32,
}

impl SeekTable {
    /// Compute the serialized size of a seek table for the given block count.
    pub fn serialized_size(block_count: u32) -> usize {
        8 * block_count as usize + 4
    }

    /// Serialize the seek table to bytes (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 * self.entries.len() + 4);
        for &offset in &self.entries {
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        // CRC32 over entries
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&buf);
        let crc = hasher.finalize();
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserialize a seek table from bytes, verifying CRC32.
    pub fn from_bytes(buf: &[u8], block_count: u32) -> Result<Self> {
        let expected_size = Self::serialized_size(block_count);
        if buf.len() < expected_size {
            return Err(CompressorError::BufferUnderflow {
                needed: expected_size,
                available: buf.len(),
            });
        }
        let entries_bytes = 8 * block_count as usize;
        let mut entries = Vec::with_capacity(block_count as usize);
        for i in 0..block_count as usize {
            let off = i * 8;
            entries.push(u64::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
                buf[off + 4],
                buf[off + 5],
                buf[off + 6],
                buf[off + 7],
            ]));
        }
        let stored_crc = u32::from_le_bytes([
            buf[entries_bytes],
            buf[entries_bytes + 1],
            buf[entries_bytes + 2],
            buf[entries_bytes + 3],
        ]);
        // Verify CRC32 over entries
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&buf[..entries_bytes]);
        let computed_crc = hasher.finalize();
        if stored_crc != computed_crc {
            return Err(CompressorError::SeekTableCrcMismatch {
                expected: stored_crc,
                computed: computed_crc,
            });
        }
        Ok(Self {
            entries,
            crc32: stored_crc,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz_token_sizes() {
        assert_eq!(LzToken::Literal(0x42).uncompressed_size(), 1);
        assert_eq!(
            LzToken::Match {
                offset: 100,
                length: 50
            }
            .uncompressed_size(),
            50
        );
    }

    #[test]
    fn lz_token_classification() {
        assert!(LzToken::Literal(0).is_literal());
        assert!(!LzToken::Literal(0).is_match());
        assert!(LzToken::Match {
            offset: 1,
            length: 4
        }
        .is_match());
        assert!(!LzToken::Match {
            offset: 1,
            length: 4
        }
        .is_literal());
    }

    #[test]
    fn frame_flags_roundtrip() {
        let flags = FrameFlags {
            data_type: DataType::Float64,
            parser_mode: ParserMode::Optimal,
            has_content_checksum: true,
            has_repcodes: true,
            has_seek_table: true,
        };
        let bits = flags.to_u16();
        let decoded = FrameFlags::from_u16(bits).unwrap();
        assert_eq!(decoded.data_type, DataType::Float64);
        assert_eq!(decoded.parser_mode, ParserMode::Optimal);
        assert!(decoded.has_content_checksum);
        assert!(decoded.has_repcodes);
        assert!(decoded.has_seek_table);
    }

    #[test]
    fn frame_flags_all_variants() {
        // Exhaustively test all data type / parser mode combinations
        let data_types = [
            DataType::Raw,
            DataType::IntegerI64,
            DataType::IntegerU64,
            DataType::IntegerI32,
            DataType::IntegerU32,
            DataType::Float64,
            DataType::Float32,
            DataType::Float64Shuffle,
            DataType::Float32Shuffle,
            DataType::Float64ShuffleDelta,
            DataType::Float32ShuffleDelta,
        ];
        let parser_modes = [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal];

        for &dt in &data_types {
            for &pm in &parser_modes {
                for &ck in &[false, true] {
                    for &rc in &[false, true] {
                        for &st in &[false, true] {
                            let flags = FrameFlags {
                                data_type: dt,
                                parser_mode: pm,
                                has_content_checksum: ck,
                                has_repcodes: rc,
                                has_seek_table: st,
                            };
                            let bits = flags.to_u16();
                            let decoded = FrameFlags::from_u16(bits).unwrap();
                            assert_eq!(decoded.data_type, dt);
                            assert_eq!(decoded.parser_mode, pm);
                            assert_eq!(decoded.has_content_checksum, ck);
                            assert_eq!(decoded.has_repcodes, rc);
                            assert_eq!(decoded.has_seek_table, st);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn frame_flags_rejects_unknown_high_bits() {
        // Valid flags with bit 11 set — should be rejected
        let valid_base = FrameFlags {
            data_type: DataType::Raw,
            parser_mode: ParserMode::Greedy,
            has_content_checksum: false,
            has_repcodes: false,
            has_seek_table: false,
        };
        let bits = valid_base.to_u16();
        assert!(FrameFlags::from_u16(bits).is_ok());

        // Set each reserved bit (11-15) and verify rejection
        for bit in 11..=15 {
            let bad_bits = bits | (1 << bit);
            assert!(
                FrameFlags::from_u16(bad_bits).is_err(),
                "bit {} should be rejected",
                bit
            );
        }
    }

    #[test]
    fn frame_header_roundtrip() {
        let header = FrameHeader {
            version: FRAME_VERSION,
            flags: FrameFlags {
                data_type: DataType::IntegerI64,
                parser_mode: ParserMode::Lazy,
                has_content_checksum: true,
                has_repcodes: true,
                has_seek_table: true,
            },
            block_size: DEFAULT_BLOCK_SIZE as u32,
            original_size: 123_456_789,
            block_count: 60,
            stride: 0,
        };
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), FrameHeader::SERIALIZED_SIZE);
        let decoded = FrameHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn frame_header_bad_magic() {
        let mut bytes = FrameHeader {
            version: FRAME_VERSION,
            flags: FrameFlags {
                data_type: DataType::Raw,
                parser_mode: ParserMode::Greedy,
                has_content_checksum: false,
                has_repcodes: true,
                has_seek_table: false,
            },
            block_size: 1024,
            original_size: 0,
            block_count: 0,
            stride: 0,
        }
        .to_bytes();
        bytes[0] = 0xFF; // corrupt magic
        assert!(matches!(
            FrameHeader::from_bytes(&bytes),
            Err(CompressorError::InvalidMagic { .. })
        ));
    }

    #[test]
    fn frame_header_buffer_underflow() {
        let short = [0u8; 10];
        assert!(matches!(
            FrameHeader::from_bytes(&short),
            Err(CompressorError::BufferUnderflow { .. })
        ));
    }

    #[test]
    fn block_header_roundtrip() {
        let header = BlockHeader {
            compressed_size: 8192,
            original_size: 65536,
            crc32: 0xDEADBEEF,
        };
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), BlockHeader::SERIALIZED_SIZE);
        let decoded = BlockHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn compression_config_presets() {
        let fast = CompressionConfig::fast();
        assert_eq!(fast.parser_mode, ParserMode::Greedy);
        assert_eq!(fast.max_chain_depth, 8);

        let balanced = CompressionConfig::balanced();
        assert_eq!(balanced.parser_mode, ParserMode::Lazy);
        assert_eq!(balanced.max_chain_depth, 64);

        let max = CompressionConfig::max_compression();
        assert_eq!(max.parser_mode, ParserMode::Optimal);
        assert_eq!(max.max_chain_depth, 256);
    }

    #[test]
    fn data_type_properties() {
        assert!(DataType::IntegerI64.uses_delta());
        assert!(DataType::IntegerU32.uses_delta());
        assert!(!DataType::Float64.uses_delta());
        assert!(!DataType::Raw.uses_delta());
        assert!(!DataType::Float64Shuffle.uses_delta());

        assert!(DataType::Float64.uses_gorilla());
        assert!(DataType::Float32.uses_gorilla());
        assert!(!DataType::IntegerI64.uses_gorilla());
        assert!(!DataType::Raw.uses_gorilla());
        assert!(!DataType::Float64Shuffle.uses_gorilla());

        assert!(DataType::Float64Shuffle.uses_shuffle());
        assert!(DataType::Float32Shuffle.uses_shuffle());
        assert!(!DataType::Float64.uses_shuffle());
        assert!(!DataType::Raw.uses_shuffle());

        assert_eq!(DataType::IntegerI64.element_size(), 8);
        assert_eq!(DataType::IntegerI32.element_size(), 4);
        assert_eq!(DataType::Float64.element_size(), 8);
        assert_eq!(DataType::Float32.element_size(), 4);
        assert_eq!(DataType::Float64Shuffle.element_size(), 8);
        assert_eq!(DataType::Float32Shuffle.element_size(), 4);
        assert_eq!(DataType::Raw.element_size(), 1);
    }

    #[test]
    fn seek_table_roundtrip() {
        let entries = vec![25u64, 150, 312, 500, 720];
        let st = SeekTable {
            entries: entries.clone(),
            crc32: 0,
        };
        let bytes = st.to_bytes();
        assert_eq!(bytes.len(), SeekTable::serialized_size(5));
        let decoded = SeekTable::from_bytes(&bytes, 5).unwrap();
        assert_eq!(decoded.entries, entries);
    }

    #[test]
    fn seek_table_crc_mismatch() {
        let st = SeekTable {
            entries: vec![100, 200, 300],
            crc32: 0,
        };
        let mut bytes = st.to_bytes();
        // Flip a byte in the entries area
        bytes[0] ^= 0xFF;
        let result = SeekTable::from_bytes(&bytes, 3);
        assert!(matches!(
            result,
            Err(CompressorError::SeekTableCrcMismatch { .. })
        ));
    }

    #[test]
    fn frame_flags_bit10_seek_table() {
        let flags_on = FrameFlags {
            data_type: DataType::Raw,
            parser_mode: ParserMode::Greedy,
            has_content_checksum: false,
            has_repcodes: false,
            has_seek_table: true,
        };
        let bits = flags_on.to_u16();
        assert_eq!((bits >> 10) & 1, 1);

        let flags_off = FrameFlags {
            has_seek_table: false,
            ..flags_on
        };
        let bits = flags_off.to_u16();
        assert_eq!((bits >> 10) & 1, 0);

        // Roundtrip
        let decoded = FrameFlags::from_u16(flags_on.to_u16()).unwrap();
        assert!(decoded.has_seek_table);
        let decoded = FrameFlags::from_u16(flags_off.to_u16()).unwrap();
        assert!(!decoded.has_seek_table);
    }
}
