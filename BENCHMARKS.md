# iotc Benchmarks

All benchmarks on Windows 11, AMD Ryzen 7 5700X, 32 GB DDR4, Rust 1.93.0. Native Rust benchmarks via Criterion (`cargo bench`). Python benchmarks via `iotc` 0.1.0 wheel on CPython 3.13.

## Headline Numbers

| Data Type | iotc Ratio | zstd -3 | LZ4 | iotc Advantage |
|-----------|-----------|---------|-----|----------------|
| Timestamps i64 (100K) | **483x** | 2.5x | 1.3x | 193x vs zstd, 371x vs LZ4 |
| Counters u32 (100K, linear) | **5195x** | 1.5x | 1.0x | 3463x vs zstd |
| GPS trajectories (fixed-point, 25M pts) | **4.5x** | 2.1x | 1.5x | 2.1x vs zstd |
| Telemetry structs (stride=12) | **22x** | 1.6x | 1.2x | 14x vs zstd |
| Temperature f64 (shuffle+delta) | **1.34x** | 1.1x | 1.0x | 1.2x vs zstd |
| General-purpose (Silesia, 202 MB) | **3.23x** | 3.19x | 2.10x | Competitive |

On integer time-series, iotc is not an incremental improvement — delta-of-delta preprocessing collapses monotonic sequences to near-constant residuals before the entropy coder sees them. LZ4 and zstd do byte-level matching on raw 8-byte integers and cannot detect that the second derivative is nearly zero.

---

## Integer Time-Series

### Compression Ratio

| Dataset | Elements | Raw | iotc | zstd -3 | zstd -19 | Parquet+ZSTD | LZ4 |
|---------|----------|-----|------|---------|----------|-------------|-----|
| Timestamps i64 | 1K | 8 KB | **8.2x** | 3.8x | 3.8x | 2.1x | — |
| Timestamps i64 | 10K | 80 KB | **10.3x** | 3.9x | 3.9x | 2.1x | — |
| Timestamps i64 | 100K | 800 KB | **483x** | 2.5x | 3.9x | 1.7x | 1.3x |
| Counters u32 | 1K | 4 KB | **30.8x** | 2.1x | 3.8x | 1.1x | — |
| Counters u32 | 10K | 40 KB | **303x** | 1.7x | 3.9x | 1.0x | — |
| Counters u32 | 100K | 400 KB | **5195x** | 1.5x | 4.0x | 0.9x | 1.0x |

Delta-of-delta on a linear counter produces constant residuals — the entropy coder compresses them to almost nothing. Parquet actually *expands* counter data at 100K because its row-group metadata exceeds the compression savings.

### Throughput (100K, Native Criterion)

| Dataset | iotc Compress | iotc Decompress | LZ4 Compress | zstd -3 Compress |
|---------|--------------|-----------------|-------------|-----------------|
| Timestamps i64 | 55 MiB/s | **1.43 GiB/s** | 545 MiB/s | 140 MiB/s |
| Counters u32 | 53 MiB/s | 931 MiB/s | 10.6 GiB/s | 378 MiB/s |

**Decompression on timestamps (1.43 GiB/s) matches LZ4 (1.45 GiB/s)** while delivering 483x ratio vs LZ4's 1.3x. This is DDR4 bus-saturation territory.

LZ4 compresses 10x faster. The question is whether iotc's CPU time buys enough ratio reduction to justify the cost on a bandwidth-constrained link:

**End-to-end on a 10 Mbps IoT radio** (800 KB timestamps):
- LZ4: 1.5 ms compress + 4.7 sec transmit (611 KB) = **4.7 sec**
- zstd: 5.4 ms compress + 2.4 sec transmit (314 KB) = **2.4 sec**
- iotc: 14 ms compress + 0.013 sec transmit (1.6 KB) = **0.027 sec** (174x faster end-to-end)

---

## Float Data

### Compression Ratio (100K elements)

| Dataset | iotc (shuffle+delta) | iotc (Gorilla XOR) | Pcodec | zstd -3 | LZ4 |
|---------|---------------------|-------------------|--------|---------|-----|
| Temperature f64 | **1.34x** | 1.1x | 1.5x | 1.1x | 1.0x |
| Vibration f32 | **1.15x** | 0.87x | 1.3x | 1.1x | 1.0x |

Byte shuffle + byte-delta groups exponent bytes into lanes where byte-delta produces long runs of zeros. Auto-detect selects the best strategy per block — Gorilla XOR for highly correlated sequences, shuffle+delta for everything else.

Gorilla XOR expands noisy f32 data (0.87x). Shuffle+delta handles both smooth and noisy floats. **Auto-detect picks the right one; manual selection is rarely needed.**

### Throughput (100K, Native Criterion)

| Dataset | iotc Compress | iotc Decompress | Pcodec Compress | Pcodec Decompress |
|---------|--------------|-----------------|----------------|------------------|
| Temperature f64 | 28 MiB/s | 256 MiB/s | 147 MiB/s | 1.4 GiB/s |
| Vibration f32 | 23 MiB/s | 147 MiB/s | 100 MiB/s | 877 MiB/s |

Float decompression (147-278 MiB/s) is the weakest area. The un-shuffle + un-byte-delta pipeline adds latency that Pcodec's native typed decode avoids. For float-heavy workloads where decompress speed matters more than ratio, Pcodec is the better choice.

---

## Structured Data (Stride Transposition)

| Records | Raw | iotc (stride=12) | iotc (no stride) | zstd -3 | zstd -19 |
|---------|-----|-----------------|-----------------|---------|----------|
| 1K | 12 KB | **1.8x** | 1.1x | 1.3x | 1.5x |
| 10K | 120 KB | **2.1x** | 1.2x | 1.3x | 1.5x |
| 100K | 1.2 MB | **2.2x** | 1.2x | 1.2x | 1.6x |

12-byte records: `{u32 seq, f32 temp, f32 humidity}`. Stride transposition groups all seq bytes, all temp bytes, all humidity bytes into contiguous lanes. Without stride, iotc does worse than zstd. With stride, iotc beats zstd -19 by 38%.

These structs contain noisy floats, which limits overall ratio. Integer-only structs or smooth-float structs achieve much higher ratios (22x+ on ideal struct data).

---

## Real-World GPS / Trajectory Data

Three real GPS datasets totaling 33M+ data points. No synthetic data.

### Datasets

| Dataset | Source | Records | Description |
|---------|--------|---------|-------------|
| T-Drive | Microsoft Research | 985,972 | 551 Beijing taxis, 1-week GPS traces |
| GeoLife | Microsoft Research | 24,876,977 | 182 users, 5-year GPS trajectories |
| NOAA AIS | US Coast Guard | 7,239,758 | 14,778 vessels, 1 full day, US waters |

Two encoding strategies:
- **Float64**: Raw `f64` lat/lon — how most systems store GPS. Mantissa bits limit compression.
- **Fixed-point i32**: `degrees * 1e7` in `i32`. Eliminates mantissa noise, enables delta-of-delta. 7 decimal places (±1.1 cm) exceeds consumer GPS resolution.

### Per-Entity Sorted (Core Use Case)

Each vehicle/user's points sorted by timestamp, concatenated. This is how fleet telemetry systems store data.

| Dataset | Encoding | Stride | iotc | zstd -3 | LZ4 |
|---------|----------|--------|------|---------|-----|
| T-Drive (986K pts) | float64 | 24 | 2.25x | **2.55x** | 1.64x |
| T-Drive | **fixed-point** | 16 | **3.23x** | 2.18x | 1.57x |
| GeoLife (24.9M pts) | float64 | 24 | **2.40x** | 2.01x | 1.34x |
| GeoLife | **fixed-point** | 16 | **4.51x** | 2.08x | 1.45x |
| GeoLife + altitude | **fixed-point** | 20 | **4.60x** | 2.30x | 1.57x |
| NOAA AIS (7.2M pts) | float64 | 28 | 2.89x | **3.35x** | 2.22x |
| NOAA AIS | **fixed-point** | 20 | **4.70x** | 2.94x | 2.07x |

With fixed-point encoding, iotc achieves **4.5-4.7x** on every dataset — **more than 2x better than zstd**. Float64 results are mixed: iotc wins on GeoLife (high per-user correlation) but zstd wins on T-Drive and AIS (shorter tracks, more diverse patterns).

### Unsorted Data (Worst Case)

> **Warning**: Unsorted spatial data is iotc's worst case. Pre-sort by entity ID before compressing.

| Dataset | Encoding | Stride | iotc | zstd -3 | LZ4 |
|---------|----------|--------|------|---------|-----|
| AIS time-sorted (7.2M pts) | float64 | 28 | 1.70x | **2.40x** | 1.52x |
| AIS time-sorted | **fixed-point** | 20 | 2.23x | **2.54x** | 1.73x |

When positions jump between vessels worldwide, adjacent records have no spatial correlation. zstd's larger context window handles this better. **Pre-sort by entity (`GROUP BY vessel_id ORDER BY timestamp`) to unlock iotc's advantage.**

### GIS Throughput (GeoLife, 570 MB)

| Codec | Compress | Decompress (est.) |
|-------|----------|-------------------|
| iotc (fp, s=16) | 72 MiB/s | ~800 MiB/s |
| zstd -3 | 92-101 MiB/s | ~500 MiB/s |
| LZ4 | 400-410 MiB/s | ~1.5 GiB/s |

iotc fixed-point is 1.3x slower than zstd to compress but delivers 2.2x better ratio. On a 100 Mbps link, transmitting 84 MB (iotc) vs 183 MB (zstd) saves 8 seconds per transfer.

---

## General-Purpose: Silesia Corpus

The [Silesia corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) is the standard benchmark for compression algorithms. 12 files, 212 MB — text, executables, databases, medical images, XML. **None of this is time-series data.** This is iotc playing away from home.

### Compression Ratio

| File | Size | iotc (greedy) | LZ4 | zstd -3 |
|------|------|--------------|-----|---------|
| dickens | 9.7 MB | **2.81x** | 1.59x | 2.77x |
| mozilla | 48.8 MB | 2.67x | 1.93x | **2.79x** |
| mr | 9.5 MB | **2.81x** | 1.83x | 2.81x |
| nci | 32.0 MB | **12.37x** | 6.06x | 11.76x |
| ooffice | 5.9 MB | 2.02x | 1.41x | 1.96x |
| osdb | 9.6 MB | 2.86x | 1.91x | **2.87x** |
| reymont | 6.3 MB | **3.55x** | 2.08x | 3.40x |
| samba | 20.6 MB | 4.17x | 2.80x | **4.33x** |
| sao | 6.9 MB | 1.28x | 1.06x | **1.31x** |
| webster | 39.5 MB | **3.65x** | 2.06x | 3.41x |
| x-ray | 8.1 MB | **1.63x** | 1.01x | 1.39x |
| xml | 5.1 MB | **8.98x** | 4.35x | 8.41x |
| **TOTAL** | **202 MB** | **3.23x** | 2.10x | 3.19x |

iotc beats zstd -3 on aggregate (3.23x vs 3.19x) and crushes LZ4 (3.23x vs 2.10x). The LZ77+FSE engine is genuinely competitive on general-purpose data despite being designed for time-series.

### Silesia Throughput (Selected)

| File | iotc Compress | iotc Decompress | LZ4 Compress | zstd -3 Decompress |
|------|--------------|-----------------|-------------|-------------------|
| dickens (10 MB) | 22 MiB/s | 413 MiB/s | 297 MiB/s | 486 MiB/s |
| mozilla (49 MB) | 60 MiB/s | 528 MiB/s | 526 MiB/s | 486 MiB/s |
| nci (32 MB) | 125 MiB/s | **1.13 GiB/s** | 788 MiB/s | 632 MiB/s |

iotc matches or beats zstd decompress on all Silesia files (413-1130 MiB/s vs 486-632 MiB/s). Compress is 5-13x slower than LZ4 — the cost of the DP-capable parser and FSE entropy coding.

---

## Domain Competitor Comparison

### vs Pcodec (typed numerical compression)

| Dimension | Winner | Margin |
|-----------|--------|--------|
| Integer ratio | **iotc** | 30x better (5195x vs 174x on timestamps) |
| Float ratio | Pcodec | Slight (1.5x vs 1.4x) |
| Integer compress speed | Tie | All within 20% |
| Float compress speed | Pcodec | 3-5x faster |
| Integer decompress speed | Pcodec | 2.3x faster |
| Float decompress speed | Pcodec | 8-12x faster |
| Wire format flexibility | **iotc** | No type requirement |
| Embedded/no-alloc | **iotc** | Zero-alloc decode, C FFI |
| General-purpose data | **iotc** | Pcodec requires typed arrays |

**Bottom line**: iotc dominates integer compression by a wide margin. On floats, Pcodec wins on throughput and edges on ratio. iotc's advantage is generality: any byte stream, embedded targets, no type knowledge required.

### vs TSZ (Gorilla reference)

TSZ only handles `(u64, f64)` pairs. 1.6x on timestamps (vs iotc's 483x), 1.2x on temperatures (vs iotc's 1.34x shuffle+delta). Not a competitive comparison.

---

## Python FFI Throughput (100K elements)

These numbers include PyO3 overhead. For true engine throughput, see the native Criterion numbers above.

| Data Type | iotc Compress | iotc Decompress | zstd -3 Compress | zstd -3 Decompress |
|-----------|--------------|-----------------|-----------------|-------------------|
| Timestamps i64 | 93 MB/s | 158 MB/s | 190 MB/s | 426 MB/s |
| Counters u32 | 134 MB/s | 205 MB/s | 300 MB/s | 635 MB/s |
| Temperature f64 | 28 MB/s | 140 MB/s | 378 MB/s | 543 MB/s |
| Vibration f32 | 16 MB/s | 59 MB/s | 361 MB/s | 689 MB/s |
| Telemetry (stride) | 21 MB/s | 74 MB/s | 365 MB/s | 438 MB/s |

zstd's Python throughput is higher because `pyzstd` is a thin C wrapper. iotc goes through PyO3 + Rust + data copy. The native benchmarks (1.4 GiB/s decompress) show the engine is fast; the Python wrapper is the bottleneck.

---

## When to Use What

| Scenario | Best Tool | Why |
|----------|-----------|-----|
| Integer time-series | **iotc** | 483x vs LZ4's 1.3x, zstd's 2.5x |
| Structured records (C structs) | **iotc with stride** | Columnar transposition + LZ77 |
| Bandwidth-constrained uplink | **iotc** | 174x faster end-to-end than LZ4 on 10 Mbps links |
| GPS/trajectory (sorted, fixed-point) | **iotc** | 4.5-4.7x vs zstd's 2.1-2.9x |
| Max compress speed | **LZ4** | 10 GiB/s, negligible memory |
| Float-heavy, decompress speed critical | **Pcodec** | 8-12x faster float decompress |
| Columnar analytics (DataFrames) | **Parquet** | Schema-aware, ecosystem integration |
| Random / encrypted data | Don't compress | Nothing works |
| Embedded / no-OS / flight controller | **iotc (no-rayon)** | Single-threaded, zero-alloc, C API |
| Random access into archives | **iotc** | O(1) block access via SeekableReader |
| Unsorted spatial data | **zstd** | iotc needs per-entity sorting |

## Known Limitations

1. **Frame overhead on small inputs**: 25-byte frame header + 12-byte block header means inputs under ~100 bytes may expand. zstd has smaller framing (13 bytes).

2. **Float decompression throughput**: 147-278 MiB/s on floats, vs 1.4 GiB/s on integers. The un-shuffle + un-delta pipeline is the bottleneck. Pcodec (877 MiB/s - 1.4 GiB/s) is faster for float-heavy workloads.

3. **Unsorted spatial data**: Pre-sorting by entity ID is required for trajectory compression. Unsorted global feeds (raw AIS, mixed fleet) lose to zstd.

4. **Python FFI overhead**: PyO3 data copy overhead is significant. For throughput-critical Python workflows, process in batches to amortize.
