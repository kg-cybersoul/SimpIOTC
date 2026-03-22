# iotc — Compression Guide

You have a compressor. It eats IoT sensor data and makes it tiny. Here's everything you need to know to use it.

## What Is This?

A compression engine for the kind of data that sensors, drones, and IoT devices produce: timestamps, temperatures, GPS coordinates, accelerometer readings, struct-packed telemetry.

Generic compressors treat all data the same. This one understands that sensor data has **patterns**: timestamps increment, temperatures drift slowly, struct fields repeat at fixed intervals. On integer time-series, the difference is not incremental — it's **483x vs zstd's 2.5x**.

## The 30-Second Version

```python
import iotc

compressed = iotc.compress(my_data)         # squish it
original   = iotc.decompress(compressed)    # unsquish it
assert original == my_data                  # lossless, always
```

That's it. Everything below is about making it compress **better** and **faster**.

## The Three Knobs That Actually Matter

### 1. Data Type (`data_type`)

Tells the compressor what kind of numbers your data contains so it can pick the right preprocessing.

| You have... | Use this | What it does |
|-------------|----------|-------------|
| No idea / mixed binary | `"auto"` or `"raw"` | Nothing special, just compresses |
| Timestamps (epoch millis, counters) | `"i64"` or `"u64"` | Delta-of-delta: turns `[1000, 1001, 1002]` into `[0, 0, 0]` |
| 32-bit IDs, sequence numbers | `"i32"` or `"u32"` | Same delta trick, 32-bit |
| Any floats (temperature, vibration, GPS) | `"auto"` | Auto-detect picks the best float strategy per block |

**If you only remember one thing:** for integers, explicitly set the data type. For floats, `auto` is fine — it selects the best strategy (Gorilla XOR, byte shuffle, or shuffle+delta) per block via entropy estimation.

```python
# Timestamps — always specify explicitly
compressed = iotc.compress(ts_bytes, data_type="i64")

# Floats — auto handles it
compressed = iotc.compress(temp_bytes)  # auto picks shuffle+delta for smooth, shuffle for noisy
```

### 2. Stride (`stride`)

This is the magic knob for **structured data** (arrays of C-structs / fixed-width records).

Say you have a telemetry struct: `{ uint32_t seq; float temp; float humidity; }` — that's 12 bytes per record. Your data is thousands of these packed end-to-end.

Without stride, the compressor sees interleaved garbage: seq-byte, temp-byte, humidity-byte, seq-byte...

With `stride=12`, it **transposes** first: all seq bytes together, all temp bytes together, all humidity bytes together. Now each column is a smooth series that compresses beautifully.

```python
# 12-byte structs: u32 + f32 + f32
compressed = iotc.compress(data, stride=12)   # 22x ratio
compressed = iotc.compress(data)              # 5.5x ratio without stride
```

**Rule of thumb:** if your data is an array of fixed-size records, set `stride` to the record size in bytes.

### 3. Parser Mode (`parser`)

How hard the compressor works to find matches.

| Mode | Speed | Ratio | When to use |
|------|-------|-------|------------|
| `"greedy"` | Fastest | Good enough | Real-time on the drone, bandwidth isn't tight |
| `"lazy"` | Default | Better | Most situations. Good tradeoff. |
| `"optimal"` | 5-10x slower | Best possible | Archival, cold storage, every byte counts |

```python
# Fast — for real-time
compressed = iotc.compress(data, parser="greedy")

# Best ratio — for storage
compressed = iotc.compress(data, parser="optimal")
```

## Column Extraction — The Arrow/Polars Bridge

The killer feature for data workflows: go from compressed binary directly to a DataFrame in 3 lines. No manual struct parsing.

```python
import iotc
import numpy as np
import polars as pl

# Open compressed telemetry (stride=12: u32 seq + f32 temp + f32 humidity)
reader = iotc.SeekableReader(compressed_bytes)

# Define your struct layout
columns = reader.extract_columns({
    "seq": "u32",
    "temperature": "f32",
    "humidity": "f32",
})

# Straight to Polars
df = pl.DataFrame({
    "seq": pl.Series(np.frombuffer(columns["seq"], dtype=np.uint32)),
    "temperature": pl.Series(np.frombuffer(columns["temperature"], dtype=np.float32)),
    "humidity": pl.Series(np.frombuffer(columns["humidity"], dtype=np.float32)),
})
```

`extract_columns` decompresses the requested block range, reverses the stride transposition, and splits the row-major data into per-column contiguous byte buffers. The buffers are directly castable to NumPy arrays — no copies, no parsing.

Supported column types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`.

### Block-Range Extraction

Don't need the whole file? Extract columns from a specific block range:

```python
# Only decompress blocks 10-20 (of 500)
columns = reader.extract_columns(
    {"seq": "u32", "temp": "f32", "humidity": "f32"},
    start_block=10,
    end_block=20,
)
```

## Python API — Full Reference

### `iotc.compress(data, **kwargs) -> bytes`

| Parameter | Type | Default | What it does |
|-----------|------|---------|-------------|
| `data` | `bytes` | required | Raw input bytes |
| `data_type` | `str` | `"auto"` | Preprocessing hint (see table above) |
| `stride` | `int` | `0` | Struct size in bytes, 0 = disabled |
| `parser` | `str` | `"lazy"` | `"greedy"`, `"lazy"`, or `"optimal"` |
| `threads` | `int` | `0` | Thread count, 0 = auto (all cores) |
| `block_size` | `int` | `2097152` | Block size in bytes (2 MiB default) |
| `checksum` | `bool` | `True` | SHA-256 integrity check |
| `seek_table` | `bool` | `True` | Enables random access |

### `iotc.decompress(data) -> bytes`

Decompresses. No options needed — everything is stored in the frame header.

### `iotc.SeekableReader(data)`

Random access into compressed data without decompressing the whole thing. The compressed data must have been created with `seek_table=True` (the default).

```python
reader = iotc.SeekableReader(compressed)

reader.block_count      # number of independently compressed blocks
reader.original_size    # total uncompressed size in bytes
reader.block_size       # bytes per block (before compression)

# Decompress just one block
block = reader.decompress_block(5)

# Decompress a byte range (figures out which blocks to touch)
chunk = reader.decompress_byte_range(1000, 2000)

# Decompress everything
all_data = reader.decompress_all()

# Extract typed columns (see Column Extraction section)
columns = reader.extract_columns({"col": "f64"})

len(reader)             # same as reader.original_size
```

**Why this matters:** if you have a 10 GB flight log and need 5 seconds of data, `decompress_byte_range` decompresses only the blocks that overlap your range. O(1) seek + O(block_size) decompress instead of O(10 GB).

## C/C++ API — Quick Reference

Build: `cargo build --release --features ffi`

Link: `iot_compressor.lib` (Windows) or `libiot_compressor.a` (Linux)

Header: `include/iotc.h`

```c
#include "iotc.h"

// Compress (library allocates output)
uint8_t* out; size_t out_len;
int64_t rc = iotc_compress_alloc(buf, buf_len, &out, &out_len, stride, IOTC_TYPE_RAW);
if (rc != IOTC_OK) {
    fprintf(stderr, "Error: %.*s\n", (int)iotc_last_error_len(), iotc_last_error());
}

// Decompress
uint8_t* dec; size_t dec_len;
iotc_decompress_alloc(out, out_len, &dec, &dec_len);

// Random access
IotcSeekableReader* rdr = iotc_seekable_open(out, out_len);
int64_t n = iotc_seekable_decompress_block(rdr, 0, block_buf, sizeof(block_buf));
iotc_seekable_close(rdr);

// Always free what the library allocated
iotc_free(out, out_len);
iotc_free(dec, dec_len);
```

**Data type constants:** `IOTC_TYPE_AUTO` (0), `IOTC_TYPE_RAW` (1), `IOTC_TYPE_I32` (2), `IOTC_TYPE_I64` (3), `IOTC_TYPE_F32` (4), `IOTC_TYPE_F64` (5), `IOTC_TYPE_U32` (6), `IOTC_TYPE_U64` (7), `IOTC_TYPE_F32S` (8), `IOTC_TYPE_F64S` (9)

**Error pattern:** every function returns `IOTC_OK` (0) or a negative code. Call `iotc_last_error()` for the human-readable message.

**Memory rule:** if the function name ends in `_alloc`, you own the buffer and must call `iotc_free(ptr, len)`. If you provide the buffer (non-alloc variants), you manage it yourself.

## Embedded / Drone Build

For ARM micro-controllers with no OS thread support:

```bash
cargo build --release --no-default-features --features ffi \
    --target thumbv7em-none-eabihf
```

`--no-default-features` disables rayon (pthreads). The compressor runs single-threaded with stack-local workspaces — zero heap allocations after first call.

## GPS / Trajectory Data

iotc achieves **4.5-4.7x** on real GPS trajectories (T-Drive, GeoLife, NOAA AIS) using fixed-point i32 encoding with stride transposition. This makes it a strong fit for fleet telemetry, logistics tracking, and trajectory archival.

### Requirements

> **Pre-sort by entity ID.** iotc's delta-of-delta preprocessing needs spatially correlated adjacent records to work. If you feed it a raw AIS broadcast (positions from thousands of vessels interleaved by time), adjacent records jump between locations worldwide and iotc achieves only 1.7x — worse than zstd's 2.4x. Pre-sorting by vehicle/vessel ID (`GROUP BY vessel_id ORDER BY timestamp`) restores the spatial correlation and unlocks the 4.5x+ ratios.

### Recommended encoding

Convert float64 lat/lon to **fixed-point i32** before compression:

```python
lat_i32 = int(lat * 1e7)   # degrees x 10^7 → ±1.1 cm accuracy
lon_i32 = int(lon * 1e7)   # exceeds consumer GPS resolution
```

Pack as stride structs (`{i64 timestamp, i32 lat, i32 lon}` = 16 bytes), compress with `stride=16`. The delta-of-delta in transposed coordinate lanes collapses slowly-drifting positions to near-constant residuals.

See [BENCHMARKS.md](BENCHMARKS.md) for full GIS benchmark results.

## Common Mistakes

**"My compression ratio sucks"**
- Did you set the data type? For integers, explicitly setting `data_type="i64"` makes a huge difference. For floats, `auto` is fine.
- Is your data structured? Set `stride` to the struct size.
- Is your trajectory data sorted by entity? Unsorted spatial data loses to zstd. See the GPS section above.
- Are you compressing tiny inputs? The 25-byte frame header + 12-byte block header is fixed overhead. Below ~100 bytes, the output can be larger than the input.

**"Decompression fails"**
- The compressed data is a self-contained frame. You can't split it arbitrarily — you need the complete frame (header + all blocks + optional trailer).
- If you truncated the file, it's gone. The per-block CRC32 will catch corruption.

**"SeekableReader says 'no seek table'"**
- The data was compressed with `seek_table=False`. Re-compress with the default (`seek_table=True`).

**"Why is `optimal` so slow?"**
- It runs a forward dynamic programming pass over the entire block. That's O(n * max_match_len) per block. Use `lazy` unless you're archiving and every byte matters.

## Building From Source

Prerequisites: Rust 1.80+

```bash
# Library + CLI
cargo build --release

# Run tests
cargo test

# Run tests including FFI
cargo test --features ffi

# Build Python wheel
pip install maturin
maturin build --release --features python
pip install target/wheels/iotc-*.whl

# Benchmarks
cargo bench
```
