use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use iot_compressor::{
    entropy,
    harness::{
        generate_counters, generate_random, generate_temperatures, generate_timestamps,
        generate_vibration,
    },
    parallel::{compress, decompress},
    parser,
    preprocessor::{self, PreprocessorConfig},
    seekable::SeekableReader,
    CompressionConfig, DataType, ParserMode,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Preprocessor Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_preprocess(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    let mut group = c.benchmark_group("preprocess/delta_i64");
    for &size in &sizes {
        let data = generate_timestamps(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| preprocessor::preprocess(black_box(data), &config))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("preprocess/gorilla_f64");
    for &size in &sizes {
        let data = generate_temperatures(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float64),
            double_delta: false,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| preprocessor::preprocess(black_box(data), &config))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("preprocess/gorilla_f32");
    for &size in &sizes {
        let data = generate_vibration(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float32),
            double_delta: false,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| preprocessor::preprocess(black_box(data), &config))
        });
    }
    group.finish();
}

fn bench_depreprocess(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    let mut group = c.benchmark_group("depreprocess/delta_i64");
    for &size in &sizes {
        let data = generate_timestamps(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocessor::preprocess(&data, &config).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &preprocessed,
            |b, preprocessed| b.iter(|| preprocessor::depreprocess(black_box(preprocessed))),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("depreprocess/gorilla_f64");
    for &size in &sizes {
        let data = generate_temperatures(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float64),
            double_delta: false,
        };
        let preprocessed = preprocessor::preprocess(&data, &config).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &preprocessed,
            |b, preprocessed| b.iter(|| preprocessor::depreprocess(black_box(preprocessed))),
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parser Benchmarks — All three modes on the same preprocessed data
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_parser(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000];

    // Parser on preprocessed timestamp data
    let mut group = c.benchmark_group("parser/timestamps");
    for &size in &sizes {
        let raw = generate_timestamps(size);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocessor::preprocess(&raw, &config).unwrap();
        group.throughput(Throughput::Bytes(preprocessed.data.len() as u64));

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let parse_config = CompressionConfig {
                parser_mode: mode,
                window_size: 65536,
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new(format!("{}", mode), size),
                &preprocessed.data,
                |b, data| b.iter(|| parser::parse(black_box(data), &parse_config)),
            );
        }
    }
    group.finish();

    // Parser on preprocessed temperature data
    let mut group = c.benchmark_group("parser/temperatures");
    for &size in &sizes {
        let raw = generate_temperatures(size);
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float64),
            double_delta: false,
        };
        let preprocessed = preprocessor::preprocess(&raw, &config).unwrap();
        group.throughput(Throughput::Bytes(preprocessed.data.len() as u64));

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let parse_config = CompressionConfig {
                parser_mode: mode,
                window_size: 65536,
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new(format!("{}", mode), size),
                &preprocessed.data,
                |b, data| b.iter(|| parser::parse(black_box(data), &parse_config)),
            );
        }
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Entropy Coder Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_entropy(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000];

    let mut group = c.benchmark_group("entropy/encode");
    for &size in &sizes {
        let raw = generate_timestamps(size);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocessor::preprocess(&raw, &config).unwrap();
        let parse_config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            window_size: 65536,
            ..Default::default()
        };
        let tokens = parser::parse(&preprocessed.data, &parse_config).unwrap();
        group.throughput(Throughput::Elements(tokens.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &tokens, |b, tokens| {
            b.iter(|| entropy::encode_tokens(black_box(tokens)))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("entropy/decode");
    for &size in &sizes {
        let raw = generate_timestamps(size);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocessor::preprocess(&raw, &config).unwrap();
        let parse_config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            window_size: 65536,
            ..Default::default()
        };
        let tokens = parser::parse(&preprocessed.data, &parse_config).unwrap();
        let (encoded, _) = entropy::encode_tokens(&tokens).unwrap();
        group.throughput(Throughput::Bytes(encoded.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &encoded, |b, encoded| {
            b.iter(|| entropy::decode_tokens(black_box(encoded)))
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full Pipeline Benchmarks — compress + decompress end-to-end
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_compress(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    // Timestamps with all parser modes
    let mut group = c.benchmark_group("compress/timestamps");
    for &size in &sizes {
        let data = generate_timestamps(size);
        group.throughput(Throughput::Bytes(data.len() as u64));

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = CompressionConfig {
                parser_mode: mode,
                data_type: Some(DataType::IntegerI64),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new(format!("{}", mode), size),
                &data,
                |b, data| b.iter(|| compress(black_box(data), &config)),
            );
        }
    }
    group.finish();

    // Temperatures (f64, lazy only to keep bench time reasonable)
    let mut group = c.benchmark_group("compress/temperatures");
    for &size in &sizes {
        let data = generate_temperatures(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float64),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Counters (u32)
    let mut group = c.benchmark_group("compress/counters");
    for &size in &sizes {
        let data = generate_counters(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerU32),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Vibration (f32)
    let mut group = c.benchmark_group("compress/vibration");
    for &size in &sizes {
        let data = generate_vibration(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float32),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Temperatures with bit-shuffle (f64)
    let mut group = c.benchmark_group("compress/temperatures_shuffle");
    for &size in &sizes {
        let data = generate_temperatures(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float64Shuffle),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Vibration with bit-shuffle (f32)
    let mut group = c.benchmark_group("compress/vibration_shuffle");
    for &size in &sizes {
        let data = generate_vibration(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float32Shuffle),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Random (incompressible baseline)
    let mut group = c.benchmark_group("compress/random");
    for &size in &sizes {
        let data = generate_random(size * 8); // match byte count of i64 generators
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: None,
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    // Timestamps
    let mut group = c.benchmark_group("decompress/timestamps");
    for &size in &sizes {
        let data = generate_timestamps(size);
        group.throughput(Throughput::Bytes(data.len() as u64));

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = CompressionConfig {
                parser_mode: mode,
                data_type: Some(DataType::IntegerI64),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            };
            let compressed = compress(&data, &config).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("{}", mode), size),
                &compressed,
                |b, compressed| b.iter(|| decompress(black_box(compressed))),
            );
        }
    }
    group.finish();

    // Temperatures
    let mut group = c.benchmark_group("decompress/temperatures");
    for &size in &sizes {
        let data = generate_temperatures(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float64),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &compressed,
            |b, compressed| b.iter(|| decompress(black_box(compressed))),
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stride Transposition Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a struct array: N records of {u32 id, f32 temp, f32 humidity} = 12 bytes each.
fn generate_struct_array_12(n: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(n * 12);
    for i in 0..n as u32 {
        data.extend_from_slice(&(1000 + i / 100).to_le_bytes());
        let temp = 22.5f32 + 0.001 * (i as f32).sin();
        data.extend_from_slice(&temp.to_le_bytes());
        let hum = 55.0f32 + 0.001 * (i as f32).cos();
        data.extend_from_slice(&hum.to_le_bytes());
    }
    data
}

/// Generate a struct array: N records of {f64 x, f64 y, f64 z} = 24 bytes each.
fn generate_struct_array_24(n: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(n * 24);
    for i in 0..n as u32 {
        let t = i as f64 * 0.001;
        data.extend_from_slice(&(t.cos() * 10.0f64).to_le_bytes());
        data.extend_from_slice(&(t.sin() * 10.0f64).to_le_bytes());
        data.extend_from_slice(&(t * 0.5f64).to_le_bytes());
    }
    data
}

fn bench_stride_transpose(c: &mut Criterion) {
    use iot_compressor::preprocessor::{bitshuffle, stride as stride_mod};

    // 600 KB (fits in L2 — tests small-data path)
    let data_12_small = generate_struct_array_12(50_000); // 600 KB

    let mut group = c.benchmark_group("transpose/stride12_600KB");
    group.throughput(Throughput::Bytes(data_12_small.len() as u64));
    group.bench_function("cache_aware", |b| {
        b.iter(|| stride_mod::transpose(black_box(&data_12_small), 12))
    });
    group.bench_function("naive_bitshuffle", |b| {
        b.iter(|| bitshuffle::shuffle(black_box(&data_12_small), 12))
    });
    group.finish();

    // 2 MiB (exceeds L2 on most CPUs — tests tiling benefit)
    let data_12_large = generate_struct_array_12(174_762); // ~2 MiB

    let mut group = c.benchmark_group("transpose/stride12_2MB");
    group.throughput(Throughput::Bytes(data_12_large.len() as u64));
    group.bench_function("cache_aware", |b| {
        b.iter(|| stride_mod::transpose(black_box(&data_12_large), 12))
    });
    group.bench_function("naive_bitshuffle", |b| {
        b.iter(|| bitshuffle::shuffle(black_box(&data_12_large), 12))
    });
    group.finish();

    // stride=24, 2 MiB
    let data_24_large = generate_struct_array_24(87_381); // ~2 MiB

    let mut group = c.benchmark_group("transpose/stride24_2MB");
    group.throughput(Throughput::Bytes(data_24_large.len() as u64));
    group.bench_function("cache_aware", |b| {
        b.iter(|| stride_mod::transpose(black_box(&data_24_large), 24))
    });
    group.bench_function("naive_bitshuffle", |b| {
        b.iter(|| bitshuffle::shuffle(black_box(&data_24_large), 24))
    });
    group.finish();
}

fn bench_compress_stride(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    // Struct array stride=12 with stride transposition
    let mut group = c.benchmark_group("compress/struct12_stride");
    for &size in &sizes {
        let data = generate_struct_array_12(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            stride: Some(12),
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Same struct array WITHOUT stride (baseline comparison)
    let mut group = c.benchmark_group("compress/struct12_raw");
    for &size in &sizes {
        let data = generate_struct_array_12(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            stride: None,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();

    // Struct array stride=24 with stride transposition
    let mut group = c.benchmark_group("compress/struct24_stride");
    for &size in &sizes {
        let data = generate_struct_array_24(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            stride: Some(24),
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| compress(black_box(data), &config))
        });
    }
    group.finish();
}

fn bench_decompress_stride(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000];

    let mut group = c.benchmark_group("decompress/struct12_stride");
    for &size in &sizes {
        let data = generate_struct_array_12(size);
        group.throughput(Throughput::Bytes(data.len() as u64));
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            stride: Some(12),
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &compressed,
            |b, compressed| b.iter(|| decompress(black_box(compressed))),
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Seek Table Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_seek_table(c: &mut Criterion) {
    // Build a 12-block frame (~768 KB of timestamp data)
    let data = generate_timestamps(12_000); // 96 KB
    let config_st = CompressionConfig {
        parser_mode: ParserMode::Lazy,
        data_type: Some(DataType::IntegerI64),
        block_size: 8192,
        store_checksum: false,
        store_seek_table: true,
        ..Default::default()
    };
    let compressed_st = compress(&data, &config_st).unwrap();
    let block_count = {
        let fh = iot_compressor::FrameHeader::from_bytes(&compressed_st).unwrap();
        fh.block_count
    };

    // Random block access — single block from middle of frame
    let mut group = c.benchmark_group("seek_table/random_block_access");
    group.throughput(Throughput::Elements(1));
    let mid = (block_count / 2) as usize;
    group.bench_function(
        BenchmarkId::from_parameter(format!("block_{}_of_{}", mid, block_count)),
        |b| {
            b.iter(|| {
                let mut reader = SeekableReader::new(black_box(&compressed_st)).unwrap();
                reader.decompress_block(mid).unwrap()
            })
        },
    );
    group.finish();

    // Byte range — 1 KB from middle of frame
    let mut group = c.benchmark_group("seek_table/byte_range_1KB");
    group.throughput(Throughput::Bytes(1024));
    let mid_byte = data.len() as u64 / 2;
    group.bench_function("1KB_mid", |b| {
        b.iter(|| {
            let mut reader = SeekableReader::new(black_box(&compressed_st)).unwrap();
            reader
                .decompress_byte_range(mid_byte, mid_byte + 1024)
                .unwrap()
        })
    });
    group.finish();

    // Seekable decompress_all vs parallel decompress
    let mut group = c.benchmark_group("seek_table/seekable_vs_parallel");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("seekable_all", |b| {
        b.iter(|| {
            let mut reader = SeekableReader::new(black_box(&compressed_st)).unwrap();
            reader.decompress_all().unwrap()
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| decompress(black_box(&compressed_st)).unwrap())
    });
    group.finish();

    // Overhead — compressed size with vs without seek table
    let config_no_st = CompressionConfig {
        store_seek_table: false,
        ..config_st.clone()
    };
    let compressed_no_st = compress(&data, &config_no_st).unwrap();
    let mut group = c.benchmark_group("seek_table/overhead");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("with_seek_table", |b| {
        b.iter(|| compress(black_box(&data), &config_st).unwrap())
    });
    group.bench_function("without_seek_table", |b| {
        b.iter(|| compress(black_box(&data), &config_no_st).unwrap())
    });
    eprintln!(
        "  Seek table overhead: {} bytes ({:.4}% of compressed)",
        compressed_st.len() - compressed_no_st.len(),
        (compressed_st.len() - compressed_no_st.len()) as f64 / compressed_no_st.len() as f64
            * 100.0
    );
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-Compressor Comparison: iotc vs LZ4 vs Zstd
// ═══════════════════════════════════════════════════════════════════════════════

/// Run compress benchmarks for iotc, LZ4, and zstd on the same data.
/// Also prints compression ratios to stderr for ratio table collection.
fn bench_vs_lz4_zstd(c: &mut Criterion) {
    // ── Data generators ──────────────────────────────────────────────
    struct Dataset {
        name: &'static str,
        data: Vec<u8>,
        iotc_config: CompressionConfig,
    }

    let datasets = vec![
        Dataset {
            name: "timestamps_i64",
            data: generate_timestamps(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::IntegerI64),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "counters_u32",
            data: generate_counters(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::IntegerU32),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "temperatures_f64",
            data: generate_temperatures(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::Float64),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "vibration_f32",
            data: generate_vibration(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::Float32),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "temperatures_f64sd",
            data: generate_temperatures(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::Float64ShuffleDelta),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "vibration_f32sd",
            data: generate_vibration(100_000),
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: Some(DataType::Float32ShuffleDelta),
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
        Dataset {
            name: "random",
            data: generate_random(800_000), // 800 KB to match i64/f64 byte count
            iotc_config: CompressionConfig {
                parser_mode: ParserMode::Greedy,
                data_type: None,
                block_size: 65536,
                store_checksum: false,
                ..Default::default()
            },
        },
    ];

    // ── Print ratio table ────────────────────────────────────────────
    eprintln!("\n=== COMPRESSION RATIO COMPARISON (100K elements) ===");
    eprintln!(
        "{:<20} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Dataset", "Raw", "iotc", "ratio", "lz4", "ratio", "zstd-3"
    );
    for ds in &datasets {
        let raw = ds.data.len();
        let iotc_c = compress(&ds.data, &ds.iotc_config).unwrap();
        let lz4_c = lz4_flex::compress_prepend_size(&ds.data);
        let zstd_c = zstd::encode_all(ds.data.as_slice(), 3).unwrap();
        eprintln!(
            "{:<20} {:>8} {:>10} {:>10.2} {:>10} {:>10.2} {:>10}",
            ds.name,
            raw,
            iotc_c.len(),
            raw as f64 / iotc_c.len() as f64,
            lz4_c.len(),
            raw as f64 / lz4_c.len() as f64,
            zstd_c.len(),
        );
    }
    eprintln!();

    // ── Compress throughput benchmarks ────────────────────────────────
    for ds in &datasets {
        let mut group = c.benchmark_group(format!("vs/compress/{}", ds.name));
        group.throughput(Throughput::Bytes(ds.data.len() as u64));

        group.bench_function("iotc", |b| {
            b.iter(|| compress(black_box(&ds.data), &ds.iotc_config))
        });

        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::compress_prepend_size(black_box(&ds.data)))
        });

        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::encode_all(black_box(ds.data.as_slice()), 3))
        });

        group.finish();
    }

    // ── Decompress throughput benchmarks ──────────────────────────────
    for ds in &datasets {
        let iotc_c = compress(&ds.data, &ds.iotc_config).unwrap();
        let lz4_c = lz4_flex::compress_prepend_size(&ds.data);
        let zstd_c = zstd::encode_all(ds.data.as_slice(), 3).unwrap();

        let mut group = c.benchmark_group(format!("vs/decompress/{}", ds.name));
        // Throughput measured against ORIGINAL data size (decompressed output)
        group.throughput(Throughput::Bytes(ds.data.len() as u64));

        group.bench_function("iotc", |b| b.iter(|| decompress(black_box(&iotc_c))));

        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::decompress_size_prepended(black_box(&lz4_c)))
        });

        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::decode_all(black_box(zstd_c.as_slice())))
        });

        group.finish();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Silesia Corpus — Real-World Compression Benchmark
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_silesia(c: &mut Criterion) {
    // Standard Silesia corpus location — skip gracefully if not present.
    // Set SILESIA_DIR env var to your corpus path, or place it in ./silesia/
    let corpus_dir = std::env::var("SILESIA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("silesia"));
    let corpus_dir = corpus_dir.as_path();
    if !corpus_dir.exists() {
        eprintln!(
            "Silesia corpus not found at {:?}, skipping (set SILESIA_DIR env var)",
            corpus_dir
        );
        return;
    }

    let files = [
        "dickens", "mozilla", "mr", "nci", "ooffice", "osdb", "reymont", "samba", "sao", "webster",
        "x-ray", "xml",
    ];

    struct CorpusFile {
        name: &'static str,
        data: Vec<u8>,
    }

    let corpus: Vec<CorpusFile> = files
        .iter()
        .filter_map(|&name| {
            let path = corpus_dir.join(name);
            std::fs::read(&path)
                .ok()
                .map(|data| CorpusFile { name, data })
        })
        .collect();

    if corpus.is_empty() {
        eprintln!("No Silesia files found, skipping");
        return;
    }

    // ── Ratio + throughput summary table ──
    let iotc_config_greedy = CompressionConfig {
        parser_mode: ParserMode::Greedy,
        data_type: None, // auto-detect (will select Raw for general data)
        block_size: 2 * 1024 * 1024,
        store_checksum: false,
        ..Default::default()
    };
    let iotc_config_lazy = CompressionConfig {
        parser_mode: ParserMode::Lazy,
        ..iotc_config_greedy.clone()
    };

    eprintln!("\n=== SILESIA CORPUS BENCHMARK ===");
    eprintln!(
        "{:<12} {:>10} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7}",
        "File", "Raw", "iotc-g", "ratio", "iotc-l", "ratio", "lz4", "ratio", "zstd-3", "ratio"
    );
    eprintln!("{}", "=".repeat(100));

    let mut total_raw: u64 = 0;
    let mut total_iotc_g: u64 = 0;
    let mut total_iotc_l: u64 = 0;
    let mut total_lz4: u64 = 0;
    let mut total_zstd: u64 = 0;

    for cf in &corpus {
        let raw = cf.data.len();

        let iotc_g = compress(&cf.data, &iotc_config_greedy).unwrap();
        let iotc_l = compress(&cf.data, &iotc_config_lazy).unwrap();
        let lz4_c = lz4_flex::compress_prepend_size(&cf.data);
        let zstd_c = zstd::encode_all(cf.data.as_slice(), 3).unwrap();

        // Verify iotc roundtrip
        assert_eq!(
            cf.data,
            decompress(&iotc_g).unwrap(),
            "{} greedy roundtrip failed",
            cf.name
        );
        assert_eq!(
            cf.data,
            decompress(&iotc_l).unwrap(),
            "{} lazy roundtrip failed",
            cf.name
        );

        total_raw += raw as u64;
        total_iotc_g += iotc_g.len() as u64;
        total_iotc_l += iotc_l.len() as u64;
        total_lz4 += lz4_c.len() as u64;
        total_zstd += zstd_c.len() as u64;

        eprintln!(
            "{:<12} {:>10} {:>10} {:>6.2}x {:>10} {:>6.2}x {:>10} {:>6.2}x {:>10} {:>6.2}x",
            cf.name,
            raw,
            iotc_g.len(),
            raw as f64 / iotc_g.len() as f64,
            iotc_l.len(),
            raw as f64 / iotc_l.len() as f64,
            lz4_c.len(),
            raw as f64 / lz4_c.len() as f64,
            zstd_c.len(),
            raw as f64 / zstd_c.len() as f64,
        );
    }

    eprintln!("{}", "-".repeat(100));
    eprintln!(
        "{:<12} {:>10} {:>10} {:>6.2}x {:>10} {:>6.2}x {:>10} {:>6.2}x {:>10} {:>6.2}x",
        "TOTAL",
        total_raw,
        total_iotc_g,
        total_raw as f64 / total_iotc_g as f64,
        total_iotc_l,
        total_raw as f64 / total_iotc_l as f64,
        total_lz4,
        total_raw as f64 / total_lz4 as f64,
        total_zstd,
        total_raw as f64 / total_zstd as f64,
    );
    eprintln!();

    // ── Criterion benchmarks: compress + decompress per file ──
    for cf in &corpus {
        // Only bench files > 1 MB to keep runtime reasonable
        if cf.data.len() < 1_000_000 {
            continue;
        }

        let mut group = c.benchmark_group(format!("silesia/compress/{}", cf.name));
        group.throughput(Throughput::Bytes(cf.data.len() as u64));
        group.sample_size(10); // Large files, fewer samples

        group.bench_function("iotc_greedy", |b| {
            b.iter(|| compress(black_box(&cf.data), &iotc_config_greedy))
        });
        group.bench_function("iotc_lazy", |b| {
            b.iter(|| compress(black_box(&cf.data), &iotc_config_lazy))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::compress_prepend_size(black_box(&cf.data)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::encode_all(black_box(cf.data.as_slice()), 3))
        });
        group.finish();

        let iotc_g = compress(&cf.data, &iotc_config_greedy).unwrap();
        let iotc_l = compress(&cf.data, &iotc_config_lazy).unwrap();
        let lz4_c = lz4_flex::compress_prepend_size(&cf.data);
        let zstd_c = zstd::encode_all(cf.data.as_slice(), 3).unwrap();

        let mut group = c.benchmark_group(format!("silesia/decompress/{}", cf.name));
        group.throughput(Throughput::Bytes(cf.data.len() as u64));
        group.sample_size(10);

        group.bench_function("iotc_greedy", |b| b.iter(|| decompress(black_box(&iotc_g))));
        group.bench_function("iotc_lazy", |b| b.iter(|| decompress(black_box(&iotc_l))));
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::decompress_size_prepended(black_box(&lz4_c)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::decode_all(black_box(zstd_c.as_slice())))
        });
        group.finish();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Domain Competitor Benchmark: iotc vs Pcodec vs TSZ (Gorilla) vs LZ4 vs zstd
// ═══════════════════════════════════════════════════════════════════════════════
//
// Blosc2 is excluded — requires cmake to build the C library from source.
// Will add when cmake is available on the build machine.

fn bench_competitors(c: &mut Criterion) {
    use pco::standalone::{simple_compress as pco_compress, simple_decompress as pco_decompress};
    use pco::ChunkConfig;
    use tsz::stream::BufferedWriter;
    use tsz::{DataPoint, Encode, StdEncoder};

    // ── Helper: extract typed arrays from byte generators ──
    fn bytes_to_i64(data: &[u8]) -> Vec<i64> {
        data.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    // ── Helper: TSZ encode ──
    fn tsz_encode(timestamps: &[u64], values: &[f64]) -> Vec<u8> {
        let w = BufferedWriter::new();
        let mut enc = StdEncoder::new(timestamps[0], w);
        for (&t, &v) in timestamps.iter().zip(values.iter()) {
            enc.encode(DataPoint::new(t, v));
        }
        enc.close().into_vec()
    }

    let n = 100_000usize;

    // ── Dataset 1: Timestamps (i64) ──
    let ts_bytes = generate_timestamps(n);
    let ts_i64 = bytes_to_i64(&ts_bytes);
    let ts_u64: Vec<u64> = ts_i64.iter().map(|&v| v as u64).collect();
    // For TSZ: need timestamps + values. Use timestamps as both.
    let ts_values: Vec<f64> = ts_i64.iter().map(|&v| v as f64).collect();

    // ── Dataset 2: Temperatures (f64) ──
    let temp_bytes = generate_temperatures(n);
    let temp_f64 = bytes_to_f64(&temp_bytes);
    // TSZ timestamps for temperatures: synthetic 1-second intervals
    let temp_timestamps: Vec<u64> = (0..n as u64).map(|i| 1_700_000_000 + i).collect();

    // ── Dataset 3: Vibration (f32) ──
    let vib_bytes = generate_vibration(n);
    let vib_f32 = bytes_to_f32(&vib_bytes);

    // ══════════════════════════════════════════════════════════════════════
    // Ratio comparison table
    // ══════════════════════════════════════════════════════════════════════

    let pco_config = ChunkConfig::default();

    // Timestamps
    let iotc_ts = compress(
        &ts_bytes,
        &CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::IntegerI64),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        },
    )
    .unwrap();
    let pco_ts = pco_compress(&ts_i64, &pco_config).unwrap();
    let tsz_ts = tsz_encode(&ts_u64, &ts_values);
    let lz4_ts = lz4_flex::compress_prepend_size(&ts_bytes);
    let zstd_ts = zstd::encode_all(ts_bytes.as_slice(), 3).unwrap();

    // Temperatures
    let iotc_temp_g = compress(
        &temp_bytes,
        &CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::Float64),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        },
    )
    .unwrap();
    let iotc_temp_sd = compress(
        &temp_bytes,
        &CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::Float64ShuffleDelta),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        },
    )
    .unwrap();
    let pco_temp = pco_compress(&temp_f64, &pco_config).unwrap();
    let tsz_temp = tsz_encode(&temp_timestamps, &temp_f64);
    let lz4_temp = lz4_flex::compress_prepend_size(&temp_bytes);
    let zstd_temp = zstd::encode_all(temp_bytes.as_slice(), 3).unwrap();

    // Vibration (f32 — no TSZ, it's f64-only)
    let iotc_vib_sd = compress(
        &vib_bytes,
        &CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::Float32ShuffleDelta),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        },
    )
    .unwrap();
    let pco_vib = pco_compress(&vib_f32, &pco_config).unwrap();
    let lz4_vib = lz4_flex::compress_prepend_size(&vib_bytes);
    let zstd_vib = zstd::encode_all(vib_bytes.as_slice(), 3).unwrap();

    // Verify roundtrips
    assert_eq!(ts_bytes, decompress(&iotc_ts).unwrap());
    assert_eq!(temp_bytes, decompress(&iotc_temp_g).unwrap());
    assert_eq!(temp_bytes, decompress(&iotc_temp_sd).unwrap());
    assert_eq!(vib_bytes, decompress(&iotc_vib_sd).unwrap());
    assert_eq!(ts_i64, pco_decompress::<i64>(&pco_ts).unwrap());
    assert_eq!(temp_f64, pco_decompress::<f64>(&pco_temp).unwrap());
    assert_eq!(vib_f32, pco_decompress::<f32>(&pco_vib).unwrap());

    let raw_ts = ts_bytes.len();
    let raw_temp = temp_bytes.len();
    let raw_vib = vib_bytes.len();

    eprintln!("\n=== DOMAIN COMPETITOR COMPARISON (100K elements) ===");
    eprintln!(
        "{:<24} {:>8} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7}",
        "Dataset",
        "Raw",
        "iotc",
        "ratio",
        "pcodec",
        "ratio",
        "tsz",
        "ratio",
        "lz4",
        "ratio",
        "zstd-3",
        "ratio"
    );
    eprintln!("{}", "=".repeat(110));

    eprintln!("{:<24} {:>8} {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x",
        "timestamps_i64", raw_ts,
        iotc_ts.len(), raw_ts as f64 / iotc_ts.len() as f64,
        pco_ts.len(), raw_ts as f64 / pco_ts.len() as f64,
        tsz_ts.len(), raw_ts as f64 / tsz_ts.len() as f64,
        lz4_ts.len(), raw_ts as f64 / lz4_ts.len() as f64,
        zstd_ts.len(), raw_ts as f64 / zstd_ts.len() as f64,
    );

    eprintln!("{:<24} {:>8} {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>6.1}x",
        "temperatures_f64 (gor)", raw_temp,
        iotc_temp_g.len(), raw_temp as f64 / iotc_temp_g.len() as f64,
        pco_temp.len(), raw_temp as f64 / pco_temp.len() as f64,
        tsz_temp.len(), raw_temp as f64 / tsz_temp.len() as f64,
        lz4_temp.len(), raw_temp as f64 / lz4_temp.len() as f64,
        zstd_temp.len(), raw_temp as f64 / zstd_temp.len() as f64,
    );
    eprintln!(
        "{:<24} {:>8} {:>10} {:>6.1}x",
        "  iotc (shuffle+delta)",
        raw_temp,
        iotc_temp_sd.len(),
        raw_temp as f64 / iotc_temp_sd.len() as f64,
    );

    eprintln!(
        "{:<24} {:>8} {:>10} {:>6.1}x {:>10} {:>6.1}x {:>10} {:>7} {:>10} {:>6.1}x {:>10} {:>6.1}x",
        "vibration_f32 (sd)",
        raw_vib,
        iotc_vib_sd.len(),
        raw_vib as f64 / iotc_vib_sd.len() as f64,
        pco_vib.len(),
        raw_vib as f64 / pco_vib.len() as f64,
        "—",
        "",
        lz4_vib.len(),
        raw_vib as f64 / lz4_vib.len() as f64,
        zstd_vib.len(),
        raw_vib as f64 / zstd_vib.len() as f64,
    );
    eprintln!();

    // ══════════════════════════════════════════════════════════════════════
    // Criterion throughput benchmarks
    // ══════════════════════════════════════════════════════════════════════

    // -- Timestamps compress --
    {
        let mut group = c.benchmark_group("competitors/compress/timestamps_i64");
        group.throughput(Throughput::Bytes(raw_ts as u64));

        let iotc_cfg = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::IntegerI64),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_function("iotc", |b| {
            b.iter(|| compress(black_box(&ts_bytes), &iotc_cfg))
        });
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_compress(black_box(&ts_i64), &pco_config))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::compress_prepend_size(black_box(&ts_bytes)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::encode_all(black_box(ts_bytes.as_slice()), 3))
        });
        group.finish();
    }

    // -- Timestamps decompress --
    {
        let mut group = c.benchmark_group("competitors/decompress/timestamps_i64");
        group.throughput(Throughput::Bytes(raw_ts as u64));

        group.bench_function("iotc", |b| b.iter(|| decompress(black_box(&iotc_ts))));
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_decompress::<i64>(black_box(&pco_ts)))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::decompress_size_prepended(black_box(&lz4_ts)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::decode_all(black_box(zstd_ts.as_slice())))
        });
        group.finish();
    }

    // -- Temperatures compress --
    {
        let mut group = c.benchmark_group("competitors/compress/temperatures_f64");
        group.throughput(Throughput::Bytes(raw_temp as u64));

        let iotc_cfg_sd = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::Float64ShuffleDelta),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_function("iotc_sd", |b| {
            b.iter(|| compress(black_box(&temp_bytes), &iotc_cfg_sd))
        });
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_compress(black_box(&temp_f64), &pco_config))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::compress_prepend_size(black_box(&temp_bytes)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::encode_all(black_box(temp_bytes.as_slice()), 3))
        });
        group.finish();
    }

    // -- Temperatures decompress --
    {
        let mut group = c.benchmark_group("competitors/decompress/temperatures_f64");
        group.throughput(Throughput::Bytes(raw_temp as u64));

        group.bench_function("iotc_sd", |b| {
            b.iter(|| decompress(black_box(&iotc_temp_sd)))
        });
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_decompress::<f64>(black_box(&pco_temp)))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::decompress_size_prepended(black_box(&lz4_temp)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::decode_all(black_box(zstd_temp.as_slice())))
        });
        group.finish();
    }

    // -- Vibration compress --
    {
        let mut group = c.benchmark_group("competitors/compress/vibration_f32");
        group.throughput(Throughput::Bytes(raw_vib as u64));

        let iotc_cfg_sd = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::Float32ShuffleDelta),
            block_size: 2 * 1024 * 1024,
            store_checksum: false,
            ..Default::default()
        };
        group.bench_function("iotc_sd", |b| {
            b.iter(|| compress(black_box(&vib_bytes), &iotc_cfg_sd))
        });
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_compress(black_box(&vib_f32), &pco_config))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::compress_prepend_size(black_box(&vib_bytes)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::encode_all(black_box(vib_bytes.as_slice()), 3))
        });
        group.finish();
    }

    // -- Vibration decompress --
    {
        let mut group = c.benchmark_group("competitors/decompress/vibration_f32");
        group.throughput(Throughput::Bytes(raw_vib as u64));

        group.bench_function("iotc_sd", |b| {
            b.iter(|| decompress(black_box(&iotc_vib_sd)))
        });
        group.bench_function("pcodec", |b| {
            b.iter(|| pco_decompress::<f32>(black_box(&pco_vib)))
        });
        group.bench_function("lz4", |b| {
            b.iter(|| lz4_flex::decompress_size_prepended(black_box(&lz4_vib)))
        });
        group.bench_function("zstd_3", |b| {
            b.iter(|| zstd::decode_all(black_box(zstd_vib.as_slice())))
        });
        group.finish();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_preprocess,
    bench_depreprocess,
    bench_parser,
    bench_entropy,
    bench_compress,
    bench_decompress,
    bench_stride_transpose,
    bench_compress_stride,
    bench_decompress_stride,
    bench_seek_table,
    bench_vs_lz4_zstd,
    bench_silesia,
    bench_competitors,
);
criterion_main!(benches);
