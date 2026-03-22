# iotc-demo

One-click compression demo for non-technical audiences. Double-click the exe, pick a file, see results.

## Build

```bash
cd demo
cargo build --release
```

The binary is ~300 KB after LTO + strip.

## What It Does

1. Welcome dialog
2. File picker (CSV, binary, telemetry, JSON, Parquet, etc.)
3. Compresses with iotc (greedy mode for speed)
4. Verifies lossless roundtrip
5. Shows results:
   - Original vs compressed size
   - Compression ratio and savings %
   - Compress/decompress throughput
   - AWS S3 cost savings per copy
   - Auto-detected struct stride (for binary files)
6. Optionally saves the `.iotc` file

## Stride Auto-Detection

For binary files, the demo tries common struct sizes (8, 12, 16, 20, 24, 28, 32, 40, 48, 64 bytes) and picks the first that cleanly divides the file size with at least 10 records. Text formats (CSV, JSON, TSV, log) skip stride detection.
