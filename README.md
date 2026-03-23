# iotc — IoT Time-Series Compressor

**483x on timestamps. 4.7x on real GPS trajectories. 1.4 GiB/s decompress.**

A high-speed LZ77/ANS compression engine in Rust, purpose-built for sensor and IoT time-series data. On integer sequences (timestamps, counters, monotonic IDs), iotc is not an incremental improvement over general-purpose compressors — it is **categorically different**: 483x vs zstd's 2.5x vs LZ4's 1.3x.

On structured data, stride transposition + domain-aware preprocessing achieves ratios that generic byte-level compressors cannot reach. On general-purpose data (Silesia corpus), the LZ77+FSE engine is still competitive with zstd (3.23x vs 3.19x).

Ships with **C/C++ bindings** for systems integration (Linux/RTOS IoT devices, ground stations), **Python bindings** for data workflows, and a **schema bridge** for direct Arrow/Polars DataFrame ingest.

## Key Numbers

| Metric | Integer Data | Float Data | General (Silesia) |
|--------|-------------|------------|-------------------|
| Compression ratio | **483x** (timestamps) | 1.3x (shuffle+delta) | 3.23x |
| Compress throughput | 55 MiB/s | 23-30 MiB/s | 22-125 MiB/s |
| Decompress throughput | **1.4 GiB/s** | 147-278 MiB/s | 413 MiB/s - 1.1 GiB/s |
| vs zstd -3 ratio | **190x better** | 1.2x better | 1.01x better |
| vs LZ4 ratio | **371x better** | 1.15x better | 1.54x better |

Decompress throughput on integer data (1.4 GiB/s) saturates DDR4 memory bandwidth. Float decompress is slower (147-278 MiB/s) due to the un-shuffle + un-delta pipeline.

**Real-world GPS validation**: 33M+ points across T-Drive, GeoLife, and NOAA AIS datasets — **4.5-4.7x** with fixed-point encoding vs zstd's 2.1-2.9x. See [BENCHMARKS.md](BENCHMARKS.md).

## Architecture

```
Raw IoT Data
    |
    v
+-----------------+
| Stride Transpose|  Array-of-Structs -> Structure-of-Arrays (optional)
+--------+--------+
         |
         v
+-----------------+
|  Preprocessor   |  Delta-of-delta (integers), Gorilla XOR, or shuffle+delta (floats)
+--------+--------+
         |
         v
+-----------------+
|  Match Finder   |  Hash chain + SIMD extension (AVX2 / portable)
+--------+--------+
         |
         v
+-----------------+
|     Parser      |  Greedy / Lazy / Optimal (forward DP)
+--------+--------+
         |
         v
+-----------------+
|  Entropy Coder  |  FSE/ANS with repcodes (3 independent sub-streams)
+--------+--------+
         |
         v
+-----------------+
|    Parallel     |  Optional rayon — 2 MiB blocks, independent compress/decompress
+--------+--------+
         |
         v
   Compressed Frame
   [Header | SeekTable | Blocks | SHA-256]
```

Each block is independently compressed. Per-block CRC32, optional whole-frame SHA-256, and optional seek table for O(1) random block access.

## Feature Flags

```toml
[features]
default = ["parallel"]
parallel = ["rayon"]    # Multi-threaded block processing
ffi = []                # C/C++ extern "C" bindings
python = ["pyo3/extension-module"]  # Python module via PyO3
```

| Build Target | Command |
|-------------|---------|
| Library (default, multi-threaded) | `cargo build --release` |
| Library (single-threaded, no rayon) | `cargo build --release --no-default-features` |
| C static library | `cargo build --release --features ffi` |
| Python wheel | `maturin build --release --features python` |

## Quick Start — CLI

```bash
cargo build --release

# Auto-detect data type, lazy parser (default)
iotc sensor_log.bin compressed.iotc

# Explicit: 64-bit integer timestamps, optimal parser
iotc -t i64 -p optimal sensor_log.bin compressed.iotc

# Structured data: 12-byte records (u32 + f32 + f32)
iotc --stride 12 sensor_log.bin compressed.iotc

# Decompress
iotc -d compressed.iotc restored.bin

# Inspect a compressed frame
iotc --info compressed.iotc
```

## Quick Start — Rust Library

```rust
use iot_compressor::{
    parallel::{compress, decompress},
    seekable::SeekableReader,
    schema::{Schema, ColumnType},
    CompressionConfig, DataType,
};

// Compress 12-byte structs: { u32 timestamp, f32 temp, f32 humidity }
let config = CompressionConfig {
    data_type: Some(DataType::Raw),  // Raw — let stride handle struct layout
    stride: Some(12),                // 12 bytes per record
    ..CompressionConfig::balanced()
};
let compressed = compress(&data, &config)?;

// Decompress
let original = decompress(&compressed)?;

// Random access (O(1) per block)
let mut reader = SeekableReader::new(&compressed)?;
let block_5 = reader.decompress_block(5)?;
let range = reader.decompress_byte_range(1000, 2000)?;

// Schema-based column extraction (Arrow/Polars bridge)
let schema = Schema::new(vec![
    ("timestamp".into(), ColumnType::U32),
    ("temperature".into(), ColumnType::F32),
    ("humidity".into(), ColumnType::F32),
]);
let columns = reader.extract_all_columns(&schema)?;
// columns[0].data → contiguous u32 bytes, zero-copy into Arrow/Polars
```

### Configuration Presets

```rust
CompressionConfig::fast()            // Greedy, chain depth 8
CompressionConfig::balanced()        // Lazy, chain depth 64 (default)
CompressionConfig::max_compression() // Optimal, chain depth 256
```

## Quick Start — Python

Build and install:
```bash
pip install maturin
maturin build --release --features python
pip install target/wheels/iotc-*.whl
```

```python
import iotc
import struct

# Compress telemetry structs
data = b""
for i in range(10000):
    data += struct.pack("<Iff", i, 20.0 + i*0.01, 50.0 + i*0.005)

compressed = iotc.compress(data, data_type="raw", stride=12)
assert iotc.decompress(compressed) == data

# O(1) random access — jump to any block without decompressing the whole file
reader = iotc.SeekableReader(compressed)
block_50 = reader.decompress_block(50)
first_1kb = reader.decompress_byte_range(0, 1024)

# Extract typed columns directly into NumPy/Polars
columns = reader.extract_columns(
    {"timestamp": "u32", "temperature": "f32", "humidity": "f32"},
)
import numpy as np
ts = np.frombuffer(columns["timestamp"], dtype=np.uint32)

# Or go straight to a Polars DataFrame
import polars as pl
df = pl.DataFrame({
    name: pl.Series(name, np.frombuffer(col, dtype=dtype))
    for (name, dtype), col in zip(
        [("timestamp", np.uint32), ("temperature", np.float32), ("humidity", np.float32)],
        [columns["timestamp"], columns["temperature"], columns["humidity"]],
    )
})
```

## Quick Start — C/C++

Build the static library:
```bash
cargo build --release --features ffi
```

Link against `target/release/libiot_compressor.a` (Linux) or `iot_compressor.lib` (Windows).

```c
#include "iotc.h"
#include <stdio.h>

// Compress telemetry
uint8_t* compressed;
size_t compressed_len;
int64_t rc = iotc_compress_alloc(
    telemetry_buf, telemetry_len,
    &compressed, &compressed_len,
    12,             // stride = sizeof(TelemetryStruct)
    IOTC_TYPE_RAW   // let stride handle the structure
);
if (rc != IOTC_OK) {
    printf("Error: %.*s\n", (int)iotc_last_error_len(), iotc_last_error());
}

// Random access on ground station — blocks default to 2 MiB
IotcSeekableReader* reader = iotc_seekable_open(compressed, compressed_len);
uint8_t* block_buf = malloc(2 * 1024 * 1024);
int64_t block_len = iotc_seekable_decompress_block(reader, 50, block_buf, 2 * 1024 * 1024);
iotc_seekable_close(reader);
free(block_buf);

// Free allocated buffers
iotc_free(compressed, compressed_len);
```

See `include/iotc.h` for the full API reference.

## Data Types & Preprocessing

| Type | Flag | Preprocessing | Best For |
|------|------|--------------|----------|
| `Raw` | `raw` | None (passthrough) | Generic / unknown data |
| `IntegerI64` | `i64` | Delta-of-delta + zigzag varint | Timestamps, counters |
| `IntegerU64` | `u64` | Delta-of-delta + zigzag varint | Unsigned counters |
| `IntegerI32` | `i32` | Delta-of-delta + zigzag varint | 32-bit sensors, GPS fixed-point |
| `IntegerU32` | `u32` | Delta-of-delta + zigzag varint | Status codes, enums |
| `Float64` | `f64` | Gorilla XOR encoding | Temperature, pressure |
| `Float32` | `f32` | Gorilla XOR encoding | Accelerometer |
| `Float64Shuffle` | `f64s` | Byte shuffle | Noisy f64 sensors |
| `Float32Shuffle` | `f32s` | Byte shuffle | Noisy f32 sensors |
| `Float64ShuffleDelta` | `f64sd` | Byte shuffle + byte delta | Smooth f64 (best auto-detect choice) |
| `Float32ShuffleDelta` | `f32sd` | Byte shuffle + byte delta | Smooth f32 (best auto-detect choice) |

Auto-detect selects the best strategy per block via Shannon entropy estimation. In most cases, `auto` picks the right transform — explicit type hints give marginal improvement.

## Stride Transposition

For structured records (arrays of fixed-size structs), `--stride N` transposes the data from Array-of-Structs to Structure-of-Arrays byte layout before compression. This groups identical byte positions across records, dramatically improving match quality.

Example: a 12-byte struct `{u32 timestamp, f32 temp, f32 humidity}` with `--stride 12`:
- **Without stride**: 5.5x compression ratio
- **With stride**: 22x compression ratio (4x improvement)

## Wire Format

```
[FrameHeader - 25 bytes]
  magic:          4 bytes  "IOTC"
  version:        1 byte
  flags:          2 bytes  (data type, parser mode, checksum, repcodes, seek_table)
  block_size:     4 bytes
  original_size:  8 bytes
  block_count:    4 bytes
  stride:         2 bytes  (0 = no transposition)

[SeekTable - 8*N+4 bytes, optional]
  entries:        8 bytes x block_count  (absolute byte offsets)
  crc32:          4 bytes

[Block 0]
  BlockHeader:   12 bytes  (compressed_size, original_size, crc32)
  Payload:       variable  (FSE-encoded LZ77 tokens)

[Block 1]
  ...

[SHA-256 digest - 32 bytes, optional]
```

## Parser Modes

| Mode | Speed | Ratio | Algorithm |
|------|-------|-------|-----------|
| **Greedy** | Fastest | Good | First match found at each position |
| **Lazy** | Balanced | Better | Checks one position ahead before committing |
| **Optimal** | Slowest | Best | Forward DP with FSE bit-cost models |

## Entropy Coding

The entropy coder implements **FSE (Finite State Entropy)**, a table-based ANS variant. The LZ77 token stream is split into three independent sub-streams, each with its own FSE table:

1. **Literals** (256-symbol alphabet)
2. **Match lengths** (52 logarithmic band codes)
3. **Match offsets** (47 codes: 3 repcodes + 44 real offset bands)

**Repcodes** track the 3 most recently used match offsets (`[1, 4, 8]` initial). When a match reuses a recent offset, it encodes as a 0-extra-bit repcode symbol instead of a full offset.

## Project Structure

```
src/
  lib.rs                  Core types, frame format, config
  main.rs                 CLI (clap)
  workspace.rs            Zero-allocation encode/decode workspaces
  schema.rs               Column extraction, Arrow/Polars bridge
  seekable.rs             SeekableReader: O(1) random block access
  ffi.rs                  C/C++ FFI bindings (feature-gated)
  python.rs               Python/PyO3 bindings (feature-gated)
  preprocessor/
    mod.rs                Dispatcher, auto-detect
    delta.rs              Delta-of-delta + zigzag + varint
    gorilla_xor.rs        Gorilla XOR float encoding
    bitshuffle.rs         Byte-level shuffle (L1-cache tiled)
    stride.rs             Struct transposition (AoS -> SoA)
  match_finder/
    mod.rs                MatchFinder API, Pareto frontier
    hash_chain.rs         Hash table + circular prev buffer
    simd_x86.rs           AVX2 / portable match extension
  entropy/
    mod.rs                Symbol coding, cost model
    fse.rs                FSE/ANS encoder/decoder
  parser/
    mod.rs                Dispatcher, replay, validation
    greedy.rs             Greedy parser
    lazy.rs               Lazy evaluation parser
    optimal.rs            Two-pass forward DP
  parallel/
    mod.rs                Block compression/decompression (parallel or sequential)
  harness/
    mod.rs                Test data generators
    verify.rs             Roundtrip verification suite
include/
  iotc.h                  C header for FFI
benches/
  compression.rs          Criterion benchmarks
```

~18,000 lines of Rust. 439 tests.

## Dependencies

| Crate | Purpose | Optional |
|-------|---------|----------|
| `rayon` | Parallel block processing | Yes (`parallel` feature) |
| `pyo3` | Python bindings | Yes (`python` feature) |
| `sha2` | SHA-256 frame integrity | No |
| `crc32fast` | Per-block CRC32 | No |
| `clap` | CLI argument parsing | No |
| `thiserror` | Error type derivation | No |
| `anyhow` | CLI error handling | No |
| `criterion` | Benchmarks (dev) | Dev only |

## Demo

The `demo/` directory contains a one-click Windows GUI demo (303 KB). Double-click the exe, pick any file, see compression results with ratio, throughput, and S3 cost savings. No CLI knowledge required.

```bash
cd demo && cargo build --release
```

## License

MIT
