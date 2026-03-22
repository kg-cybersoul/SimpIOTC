# iotc Architecture

Technical design reference for contributors. Covers the compression pipeline, key algorithms, and the reasoning behind non-obvious design choices.

## Compression Pipeline

Data flows through six stages. Each stage is optional and independently testable.

```
Input bytes
  → Stride Transpose (AoS → SoA, if stride > 0)
  → Preprocessor (delta-of-delta, Gorilla XOR, or shuffle+delta)
  → Match Finder (hash chain + SIMD extension)
  → Parser (greedy, lazy, or optimal DP)
  → Entropy Coder (FSE/ANS, 3 sub-streams)
  → Frame Assembly (headers, seek table, checksums)
```

Decompression reverses the pipeline: FSE decode → LZ77 replay → inverse preprocess → inverse transpose.

## Frame Format

```
[FrameHeader — 25 bytes]
  magic:          "IOTC" (4 bytes)
  version:        u8
  flags:          u16 (data type 4 bits, parser 2 bits, checksum/repcode/seek bits)
  block_size:     u32
  original_size:  u64
  block_count:    u32
  stride:         u16 (0 = no transposition)

[SeekTable — 8*N+4 bytes, optional]
  entries:        u64 x block_count (absolute byte offsets to each BlockHeader)
  crc32:          u32

[Block 0..N]
  BlockHeader:    12 bytes (compressed_size: u32, original_size: u32, crc32: u32)
  Payload:        variable (FSE-encoded LZ77 token stream)

[SHA-256 — 32 bytes, optional]
```

Each block is independently compressed and checksummed. This enables:
- Parallel compression/decompression via rayon
- O(1) random access via SeekableReader
- Corruption isolation to single blocks

## Stride Transposition

For structured records (arrays of fixed-size C structs), stride transposition converts Array-of-Structs (AoS) to Structure-of-Arrays (SoA) at the byte level.

Given records `[R0, R1, R2, ...]` where each `Ri` is `stride` bytes, the transposition groups byte position 0 from all records, then byte position 1, etc. This creates columns where:
- Timestamp bytes form a smooth monotonic sequence
- Temperature exponent bytes are nearly constant
- Status flag bytes are highly repetitive

The LZ77 match finder sees much longer matches in SoA layout than in interleaved AoS.

Inverse transposition on decompression is a simple scatter operation. Both directions are O(n) with a single pass.

## Preprocessor Strategies

### Delta-of-Delta (integers)

For i32/i64/u32/u64 elements:
1. Compute first differences: `d[i] = x[i] - x[i-1]`
2. Compute second differences: `dd[i] = d[i] - d[i-1]`
3. Zigzag encode (maps signed to unsigned, small values → small codes)
4. Varint encode (1-10 bytes per value, small values → fewer bytes)

On monotonic sequences (timestamps, counters), the second derivative is nearly zero. Varint encodes these as single bytes, giving the entropy coder a stream of near-constant symbols.

### Gorilla XOR (floats)

Facebook's Gorilla encoding (VLDB 2015) for IEEE 754 floats:
1. XOR consecutive values: `x[i] ^ x[i-1]`
2. When consecutive values are similar, XOR produces long runs of zero bits
3. Run-length encode the leading/trailing zeros

Best for smooth, slowly-changing float series (temperature, pressure). Expands noisy data where consecutive values have unrelated mantissa bits.

### Byte Shuffle + Byte Delta (floats)

The adaptive float strategy, selected by auto-detect when it outperforms Gorilla:
1. **Byte shuffle**: Separate the 4/8 bytes of each float into lanes (all byte-0s together, all byte-1s, etc.)
2. **Byte delta**: Apply delta encoding to each lane

Exponent bytes (which change slowly even in noisy data) form a near-constant lane that compresses well. Mantissa bytes remain chaotic but the exponent lane savings outweigh the overhead.

The shuffle operation is **L1-cache tiled**: large blocks are processed in tiles that fit in L1 to avoid cache thrashing.

## Match Finder

Hash chain with SIMD-accelerated match extension:
- Hash table maps 4-byte windows to positions in a circular buffer
- Chain depth is configurable (8/64/256 for greedy/balanced/max presets)
- Match extension uses AVX2 `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` to compare 32 bytes per cycle (portable fallback for non-x86)
- Minimum match length: 4 bytes
- Maximum match length: 65,535 bytes

### Pareto Frontier

The match finder maintains up to 4 candidate matches of increasing length. The parser selects from this frontier — short matches for nearby offsets, long matches for distant ones.

## Parser Modes

### Greedy
Takes the first match at each position. O(n) with low constant. Suitable for real-time embedded use.

### Lazy
At each match, checks one position ahead. If the next position has a longer/better match, emits a literal and takes the better match. O(n) with ~1.5x the work of greedy. Default mode.

### Optimal
Two-pass forward dynamic programming:
1. **Forward pass**: For each position, compute the bit cost of every reachable (literal, match) choice using FSE bit-cost models. Build a cost array.
2. **Backward pass**: Trace back the minimum-cost path.

The FSE cost models are bootstrapped from a preliminary greedy parse (first-pass statistics). This is O(n * max_match_len) per block.

## Entropy Coding (FSE/ANS)

The LZ77 token stream is split into three independent sub-streams:

1. **Literals** (256-symbol alphabet): Raw unmatched bytes
2. **Match lengths** (52 logarithmic band codes): How long each match is
3. **Match offsets** (47 codes): 3 repcodes + 44 real offset bands with extra bits

Each sub-stream has its own FSE table (frequency-normalized). Tables are serialized in the block payload header.

### Repcodes

Three most-recently-used match offsets are tracked (`[1, 4, 8]` initial). When a match reuses a recent offset, it encodes as a 0-extra-bit repcode symbol. This is the mechanism behind structured data compression — when stride transposition creates columns, matches at offset=stride are extremely common and encode for free via repcode 0.

### BitWriter / BitReader

Bit-level I/O uses 64-bit accumulators with bulk flush/refill:
- **BitWriter**: Accumulates up to 56 bits, flushes 8 bytes at a time
- **BitReader**: Refills 8 bytes via unaligned `u64` read when bits available drops below threshold

The unaligned read in `BitReader::refill()` is safe because the guard `available >= 8` ensures the read stays within the buffer.

## Parallel Processing

When the `parallel` feature is enabled (default), compression and decompression use rayon:
- Input is divided into 2 MiB blocks (configurable via `block_size`)
- Each block is independently compressed/decompressed
- Block payloads are assembled sequentially into the output frame

When `num_threads = 0` (default), the global rayon thread pool is reused. Setting `num_threads > 0` creates a new thread pool per `compress()` call — suitable for one-shot CLI usage but not recommended for server loops.

### Decode Workspaces

Decompression uses thread-local reusable workspaces (`thread_local! { RefCell<DecodeWorkspace> }`) to avoid per-block heap allocation. Each rayon worker gets its own workspace with zero contention.

## Schema Bridge

`schema.rs` provides `Schema` and `extract_columns()` for typed column extraction from compressed structured data:

1. Decompress the requested block range
2. Reverse stride transposition (SoA → AoS)
3. Split row-major records into per-column contiguous byte buffers

The output buffers are directly castable to Arrow arrays or NumPy `frombuffer()` — no parsing, no copies beyond the decompression itself.

Supported column types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`.

## FFI Layer

`ffi.rs` (feature-gated) exposes `extern "C"` functions for C/C++ integration:
- Thread-safe stateless functions (`iotc_compress`, `iotc_decompress`)
- `IotcSeekableReader` handles for random access (NOT thread-safe per handle)
- Thread-local error messages via `CString` in `RefCell` (null-terminated, valid until next FFI call on same thread)
- Panic boundary: all FFI entry points wrap in `catch_unwind` to prevent panics from crossing the C ABI

## Python Bindings

`python.rs` (feature-gated) provides PyO3 bindings:
- `iotc.compress()` / `iotc.decompress()` — stateless functions
- `iotc.SeekableReader` — wraps the Rust `SeekableReader` with Python-friendly API
- `reader.extract_columns(schema_dict)` — returns `dict[str, bytes]` for NumPy/Polars ingest

## Key Design Decisions

### Zero-dependency decompression path
The decompressor uses only `sha2` and `crc32fast` — no allocator-heavy crates. This keeps the embedded/no-std path viable.

### Byte-level transforms over typed transforms
iotc's preprocessor operates at the byte level (byte shuffle, byte delta) rather than requiring typed arrays. This means:
- The wire format doesn't need to know the source type
- Mixed-type structs work naturally via stride transposition
- The tradeoff is ~5x slower float decompress vs Pcodec's typed approach

### FSE over Huffman
FSE (Finite State Entropy) achieves ~0.1 bits/symbol better compression than Huffman on typical LZ77 token distributions, at comparable decode speed. The 3-stream split (literals, lengths, offsets) allows each stream to have its own optimal frequency table.

### Self-referential structs in FFI/Python
`IotcSeekableReader` (FFI) and `PySeekableReader` (Python) hold both the data (`Vec<u8>`) and a reader referencing that data (`SeekableReader<'static>`). The lifetime is erased via `transmute`. This is sound because:
- Fields drop in declaration order (data outlives reader)
- The data is never moved after construction
- The alternative (adding a crate like `ouroboros`) was rejected to keep the dependency count minimal
