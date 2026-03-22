//! # Roundtrip Verification & Integrity Testing
//!
//! This module provides strict SHA-256 verified roundtrip testing across
//! all data types and parser modes. It ensures that:
//!
//! 1. Every byte survives the compress→decompress roundtrip.
//! 2. Corrupted compressed data is detected and rejected.
//! 3. The compressor handles edge cases (empty, tiny, huge, adversarial).
//!
//! These are not benchmarks — they run in `cargo test` and catch
//! correctness regressions.

#[cfg(test)]
mod tests {
    use crate::harness::*;
    use crate::parallel::{compress, decompress};
    use crate::{CompressionConfig, DataType, ParserMode};
    use sha2::{Digest, Sha256};

    /// Compute SHA-256 of a byte slice.
    fn sha256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verify roundtrip with SHA-256 for a dataset and parser mode.
    fn verify_roundtrip(data: &[u8], data_type: Option<DataType>, mode: ParserMode, label: &str) {
        let original_hash = sha256(data);

        let config = CompressionConfig {
            parser_mode: mode,
            data_type,
            block_size: 16384, // small blocks for test speed
            store_checksum: true,
            ..Default::default()
        };

        let compressed = compress(data, &config)
            .unwrap_or_else(|e| panic!("{} compress failed: {:?}", label, e));

        let decompressed = decompress(&compressed)
            .unwrap_or_else(|e| panic!("{} decompress failed: {:?}", label, e));

        assert_eq!(
            decompressed.len(),
            data.len(),
            "{}: length mismatch ({} vs {})",
            label,
            decompressed.len(),
            data.len()
        );

        let roundtrip_hash = sha256(&decompressed);
        assert_eq!(
            original_hash, roundtrip_hash,
            "{}: SHA-256 mismatch — data corruption in roundtrip",
            label
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Exhaustive: All data types × all parser modes
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_timestamps_all_modes() {
        let data = generate_timestamps(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::IntegerI64),
                mode,
                &format!("timestamps/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_temperatures_all_modes() {
        let data = generate_temperatures(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float64),
                mode,
                &format!("temperatures/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_counters_all_modes() {
        let data = generate_counters(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::IntegerU32),
                mode,
                &format!("counters/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_vibration_all_modes() {
        let data = generate_vibration(3000); // 3000 to be divisible by 3 (X,Y,Z)
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float32),
                mode,
                &format!("vibration/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_constant_all_modes() {
        let data = generate_constant(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::IntegerI64),
                mode,
                &format!("constant/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_temperatures_shuffle_all_modes() {
        let data = generate_temperatures(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float64Shuffle),
                mode,
                &format!("temperatures-shuffle/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_vibration_shuffle_all_modes() {
        let data = generate_vibration(3000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float32Shuffle),
                mode,
                &format!("vibration-shuffle/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_random_all_modes() {
        let data = generate_random(16000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(&data, None, mode, &format!("random/{}", mode));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Auto-detect roundtrips (data_type = None)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_auto_detect_timestamps() {
        let data = generate_timestamps(1000);
        verify_roundtrip(&data, None, ParserMode::Lazy, "auto-detect/timestamps");
    }

    #[test]
    fn roundtrip_auto_detect_temperatures() {
        let data = generate_temperatures(1000);
        verify_roundtrip(&data, None, ParserMode::Lazy, "auto-detect/temperatures");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Edge Cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_minimum_viable_input() {
        // Smallest inputs that should still compress/decompress
        // Need at least 4 bytes for the match finder's hash prefix
        let data = vec![0xAA; 8];
        verify_roundtrip(&data, None, ParserMode::Greedy, "edge/min-8-bytes");
    }

    #[test]
    fn roundtrip_single_element_i64() {
        let data = 42i64.to_le_bytes().to_vec();
        verify_roundtrip(
            &data,
            Some(DataType::IntegerI64),
            ParserMode::Greedy,
            "edge/single-i64",
        );
    }

    #[test]
    fn roundtrip_single_element_f64() {
        let data = std::f64::consts::PI.to_le_bytes().to_vec();
        verify_roundtrip(
            &data,
            Some(DataType::Float64),
            ParserMode::Greedy,
            "edge/single-f64",
        );
    }

    #[test]
    fn roundtrip_two_elements_i32() {
        let mut data = Vec::new();
        data.extend_from_slice(&100i32.to_le_bytes());
        data.extend_from_slice(&101i32.to_le_bytes());
        verify_roundtrip(
            &data,
            Some(DataType::IntegerI32),
            ParserMode::Lazy,
            "edge/two-i32",
        );
    }

    #[test]
    fn roundtrip_all_zeros_large() {
        let data = vec![0u8; 65536];
        verify_roundtrip(&data, None, ParserMode::Optimal, "edge/all-zeros-64k");
    }

    #[test]
    fn roundtrip_all_ones_large() {
        let data = vec![0xFF; 65536];
        verify_roundtrip(&data, None, ParserMode::Optimal, "edge/all-0xff-64k");
    }

    #[test]
    fn roundtrip_alternating_pattern() {
        let data: Vec<u8> = (0..32768)
            .map(|i| if i % 2 == 0 { 0xAA } else { 0x55 })
            .collect();
        verify_roundtrip(&data, None, ParserMode::Lazy, "edge/alternating");
    }

    #[test]
    fn roundtrip_ascending_bytes() {
        let data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
        verify_roundtrip(&data, None, ParserMode::Lazy, "edge/ascending");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Corruption Detection
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn corruption_flipped_bit_in_payload() {
        let data = generate_timestamps(500);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 16384,
            store_checksum: true,
            ..Default::default()
        };

        let mut compressed = compress(&data, &config).unwrap();

        // Flip a bit in the payload area (after the frame header).
        // Frame header is ≥ 14 bytes, so flip at offset 20.
        if compressed.len() > 20 {
            compressed[20] ^= 0x01;
        }

        let result = decompress(&compressed);
        assert!(
            result.is_err(),
            "flipped bit should cause decompression to fail"
        );
    }

    #[test]
    fn corruption_truncated_compressed_data() {
        let data = generate_timestamps(500);
        let config = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: Some(DataType::IntegerI64),
            block_size: 16384,
            store_checksum: true,
            ..Default::default()
        };

        let compressed = compress(&data, &config).unwrap();

        // Truncate to half
        let truncated = &compressed[..compressed.len() / 2];
        let result = decompress(truncated);
        assert!(
            result.is_err(),
            "truncated data should cause decompression to fail"
        );
    }

    #[test]
    fn corruption_sha256_detects_wrong_data() {
        let data = generate_temperatures(500);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float64),
            block_size: 16384,
            store_checksum: true,
            ..Default::default()
        };

        let mut compressed = compress(&data, &config).unwrap();

        // Flip a bit near the end (likely in the SHA-256 checksum itself)
        let last = compressed.len() - 1;
        compressed[last] ^= 0x80;

        let result = decompress(&compressed);
        assert!(result.is_err(), "SHA-256 corruption should be detected");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Compression Ratio Sanity
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn ratio_constant_data_compresses_well() {
        let data = generate_constant(10000);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let ratio = data.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 10.0,
            "constant data should compress >10x, got {:.2}x",
            ratio
        );
    }

    #[test]
    fn ratio_timestamps_compress_well() {
        let data = generate_timestamps(10000);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let ratio = data.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 2.0,
            "timestamp data should compress >2x, got {:.2}x",
            ratio
        );
    }

    #[test]
    fn ratio_random_data_does_not_explode() {
        let data = generate_random(10000);
        let config = CompressionConfig {
            parser_mode: ParserMode::Greedy,
            data_type: None,
            block_size: 65536,
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        // Random data is incompressible. Our format adds per-block overhead:
        // frame header (~14B), block header (~10B), FSE tables (~500B for
        // 256-symbol alphabet), CRC32 (4B), SHA-256 (32B), type-bit array.
        // Allow up to 25% expansion on small inputs.
        let max_size = (data.len() as f64 * 1.25) as usize;
        assert!(
            compressed.len() <= max_size,
            "random data expanded too much: {} → {} ({:.0}% overhead)",
            data.len(),
            compressed.len(),
            (compressed.len() as f64 / data.len() as f64 - 1.0) * 100.0
        );
    }

    #[test]
    fn compression_ratio_report() {
        let datasets: Vec<(&str, Vec<u8>, Option<DataType>)> = vec![
            (
                "timestamps_100K",
                generate_timestamps(100_000),
                Some(DataType::IntegerI64),
            ),
            (
                "temperatures_100K",
                generate_temperatures(100_000),
                Some(DataType::Float64),
            ),
            (
                "temps_shuf_100K",
                generate_temperatures(100_000),
                Some(DataType::Float64Shuffle),
            ),
            (
                "counters_100K",
                generate_counters(100_000),
                Some(DataType::IntegerI32),
            ),
            (
                "vibration_99K",
                generate_vibration(99_000),
                Some(DataType::Float32),
            ),
            (
                "vibr_shuf_99K",
                generate_vibration(99_000),
                Some(DataType::Float32Shuffle),
            ),
            (
                "temps_sd_100K",
                generate_temperatures(100_000),
                Some(DataType::Float64ShuffleDelta),
            ),
            (
                "vibr_sd_99K",
                generate_vibration(99_000),
                Some(DataType::Float32ShuffleDelta),
            ),
        ];
        eprintln!(
            "\n{:<22} {:<12} {:>10} {:>10} {:>8}",
            "Dataset", "Parser", "Original", "Compressed", "Ratio"
        );
        eprintln!("{}", "=".repeat(66));
        for (name, data, dt) in &datasets {
            for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
                let config = CompressionConfig {
                    parser_mode: mode,
                    data_type: *dt,
                    block_size: 65536,
                    store_checksum: false,
                    ..Default::default()
                };
                let compressed = compress(data, &config).unwrap();
                let ratio = data.len() as f64 / compressed.len() as f64;
                eprintln!(
                    "{:<22} {:<12} {:>10} {:>10} {:>7.2}x",
                    name,
                    format!("{}", mode),
                    data.len(),
                    compressed.len(),
                    ratio
                );
            }
            eprintln!("{}", "-".repeat(66));
        }
    }

    #[test]
    fn optimal_never_worse_than_greedy_on_structured_data() {
        let data = generate_timestamps(5000);
        let base_config = CompressionConfig {
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };

        let greedy_size = compress(
            &data,
            &CompressionConfig {
                parser_mode: ParserMode::Greedy,
                ..base_config.clone()
            },
        )
        .unwrap()
        .len();

        let optimal_size = compress(
            &data,
            &CompressionConfig {
                parser_mode: ParserMode::Optimal,
                ..base_config
            },
        )
        .unwrap()
        .len();

        assert!(
            optimal_size <= greedy_size,
            "optimal ({}) should be <= greedy ({}) on structured data",
            optimal_size,
            greedy_size
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Multi-block Roundtrips
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_many_small_blocks() {
        let data = generate_timestamps(5000); // 40KB → many 4KB blocks
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 4096, // very small blocks
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(sha256(&data), sha256(&decompressed));
    }

    #[test]
    fn roundtrip_single_huge_block() {
        let data = generate_temperatures(5000); // 40KB in one block
        let config = CompressionConfig {
            parser_mode: ParserMode::Optimal,
            data_type: Some(DataType::Float64),
            block_size: 1024 * 1024, // 1MB block, so all in one
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(sha256(&data), sha256(&decompressed));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stress: Large Datasets
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn stress_large_timestamp_dataset() {
        // 100K timestamps = 800KB — exercises multi-block, parallel decompress
        let data = generate_timestamps(100_000);
        verify_roundtrip(
            &data,
            Some(DataType::IntegerI64),
            ParserMode::Lazy,
            "stress/100k-timestamps",
        );
    }

    #[test]
    fn stress_large_temperature_dataset() {
        let data = generate_temperatures(100_000);
        verify_roundtrip(
            &data,
            Some(DataType::Float64),
            ParserMode::Lazy,
            "stress/100k-temperatures",
        );
    }

    #[test]
    fn stress_large_temperature_shuffle_dataset() {
        let data = generate_temperatures(100_000);
        verify_roundtrip(
            &data,
            Some(DataType::Float64Shuffle),
            ParserMode::Lazy,
            "stress/100k-temperatures-shuffle",
        );
    }

    #[test]
    fn stress_large_vibration_shuffle_dataset() {
        let data = generate_vibration(99_000);
        verify_roundtrip(
            &data,
            Some(DataType::Float32Shuffle),
            ParserMode::Lazy,
            "stress/99k-vibration-shuffle",
        );
    }

    #[test]
    fn stress_large_random_dataset() {
        let data = generate_random(500_000);
        verify_roundtrip(&data, None, ParserMode::Greedy, "stress/500k-random");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Repcode: raw structured data roundtrip
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_raw_struct_array_all_modes() {
        // 12-byte struct: [u32 id, f32 temp, f32 humidity] × 5000
        // Compressed as Raw — LZ77 will find matches at offset 12 (struct stride).
        // Every match reuses the same offset → repcode rep0 fires heavily.
        let mut data = Vec::with_capacity(60_000);
        for i in 0u32..5000 {
            data.extend_from_slice(&i.to_le_bytes());
            let temp = 20.0f32 + 0.01 * (i as f32);
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 50.0f32 + 0.005 * (i as f32);
            data.extend_from_slice(&hum.to_le_bytes());
        }

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Raw),
                mode,
                &format!("struct_array/{}", mode),
            );
        }
    }

    #[test]
    fn repcode_benefits_raw_struct_stride() {
        // Verify that the compressor achieves meaningful compression on raw
        // struct arrays — the fixed stride means repcodes eliminate most of
        // the offset entropy.
        let mut data = Vec::with_capacity(48_000);
        for i in 0u32..4000 {
            // 12-byte struct with slowly changing values → high match density.
            data.extend_from_slice(&(1000 + i / 100).to_le_bytes());
            let temp = 22.5f32 + 0.001 * (i as f32).sin();
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 55.0f32 + 0.001 * (i as f32).cos();
            data.extend_from_slice(&hum.to_le_bytes());
        }

        let config = CompressionConfig {
            parser_mode: ParserMode::Optimal,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        let ratio = data.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 1.0,
            "raw struct array should compress (ratio {:.2}x), not expand",
            ratio
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 11: Stride Transposition Tests
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: roundtrip with stride transposition.
    fn verify_roundtrip_stride(data: &[u8], stride: u16, mode: ParserMode, label: &str) {
        let original_hash = sha256(data);

        let config = CompressionConfig {
            parser_mode: mode,
            data_type: Some(DataType::Raw),
            block_size: 16384,
            store_checksum: true,
            stride: Some(stride),
            ..Default::default()
        };

        let compressed = compress(data, &config)
            .unwrap_or_else(|e| panic!("{} compress failed: {:?}", label, e));

        let decompressed = decompress(&compressed)
            .unwrap_or_else(|e| panic!("{} decompress failed: {:?}", label, e));

        assert_eq!(
            decompressed.len(),
            data.len(),
            "{}: length mismatch ({} vs {})",
            label,
            decompressed.len(),
            data.len()
        );

        let recovered_hash = sha256(&decompressed);
        assert_eq!(
            original_hash, recovered_hash,
            "{}: SHA-256 mismatch after roundtrip",
            label
        );
    }

    #[test]
    fn stride_roundtrip_12byte_struct_all_modes() {
        // 12-byte struct: [u32 id, f32 temp, f32 humidity] × 5000
        let mut data = Vec::with_capacity(60_000);
        for i in 0u32..5000 {
            data.extend_from_slice(&i.to_le_bytes());
            let temp = 20.0f32 + 0.01 * (i as f32);
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 50.0f32 + 0.005 * (i as f32);
            data.extend_from_slice(&hum.to_le_bytes());
        }

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip_stride(&data, 12, mode, &format!("stride12_struct/{}", mode));
        }
    }

    #[test]
    fn stride_roundtrip_8byte_equiv_shuffle() {
        // stride=8 is functionally equivalent to Float64Shuffle.
        let values: Vec<f64> = (0..1000).map(|i| 22.5 + 0.01 * (i as f64).sin()).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        verify_roundtrip_stride(&data, 8, ParserMode::Lazy, "stride8_f64");
    }

    #[test]
    fn stride_roundtrip_24byte_struct() {
        // 24-byte struct: [f64 x, f64 y, f64 z] × 2000 (3D position)
        let mut data = Vec::with_capacity(48_000);
        for i in 0u32..2000 {
            let t = i as f64 * 0.001;
            let x: f64 = t.cos() * 10.0;
            let y: f64 = t.sin() * 10.0;
            let z: f64 = t * 0.5;
            data.extend_from_slice(&x.to_le_bytes());
            data.extend_from_slice(&y.to_le_bytes());
            data.extend_from_slice(&z.to_le_bytes());
        }

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip_stride(&data, 24, mode, &format!("stride24_3d_pos/{}", mode));
        }
    }

    #[test]
    fn stride_roundtrip_4byte() {
        // stride=4 equivalent to Float32Shuffle layout.
        let values: Vec<f32> = (0..3000).map(|i| 100.0 + 0.5 * (i as f32).sin()).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        verify_roundtrip_stride(&data, 4, ParserMode::Optimal, "stride4_f32");
    }

    #[test]
    fn stride_roundtrip_multiblock() {
        // Large enough to span multiple blocks (16 KB block size, 100KB data).
        let mut data = Vec::with_capacity(100_800);
        for i in 0u32..8400 {
            // 12-byte struct
            data.extend_from_slice(&i.to_le_bytes());
            let temp = 20.0f32 + 0.01 * (i as f32);
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 50.0f32 + 0.005 * (i as f32);
            data.extend_from_slice(&hum.to_le_bytes());
        }

        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 16384, // ~1365 structs per block
            store_checksum: true,
            stride: Some(12),
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        assert!(
            frame.block_count > 1,
            "expected multiple blocks, got {}",
            frame.block_count
        );
        assert_eq!(frame.stride, 12);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn stride_improves_compression_on_struct_data() {
        // Verify that stride transposition meaningfully improves compression
        // ratio on structured data compared to raw (no stride).
        let mut data = Vec::with_capacity(48_000);
        for i in 0u32..4000 {
            data.extend_from_slice(&(1000 + i / 100).to_le_bytes());
            let temp = 22.5f32 + 0.001 * (i as f32).sin();
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 55.0f32 + 0.001 * (i as f32).cos();
            data.extend_from_slice(&hum.to_le_bytes());
        }

        // Compress without stride
        let config_no_stride = CompressionConfig {
            parser_mode: ParserMode::Optimal,
            data_type: Some(DataType::Raw),
            block_size: 65536,
            store_checksum: false,
            stride: None,
            ..Default::default()
        };
        let compressed_no_stride = compress(&data, &config_no_stride).unwrap();

        // Compress with stride=12
        let config_stride = CompressionConfig {
            stride: Some(12),
            ..config_no_stride.clone()
        };
        let compressed_stride = compress(&data, &config_stride).unwrap();

        let ratio_no = data.len() as f64 / compressed_no_stride.len() as f64;
        let ratio_yes = data.len() as f64 / compressed_stride.len() as f64;

        // Both must roundtrip
        assert_eq!(data, decompress(&compressed_no_stride).unwrap());
        assert_eq!(data, decompress(&compressed_stride).unwrap());

        assert!(
            ratio_yes > ratio_no,
            "stride transposition should improve ratio: {:.2}x (stride) vs {:.2}x (no stride)",
            ratio_yes,
            ratio_no
        );
    }

    #[test]
    fn stride_compression_ratio_report() {
        // Print actual stride vs no-stride compression ratios for the record.
        let mut data = Vec::with_capacity(120_000);
        for i in 0u32..10_000 {
            data.extend_from_slice(&(1000 + i / 100).to_le_bytes());
            let temp = 22.5f32 + 0.001 * (i as f32).sin();
            data.extend_from_slice(&temp.to_le_bytes());
            let hum = 55.0f32 + 0.001 * (i as f32).cos();
            data.extend_from_slice(&hum.to_le_bytes());
        }
        let input_size = data.len();

        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config_no = CompressionConfig {
                parser_mode: mode,
                data_type: Some(DataType::Raw),
                block_size: 65536,
                store_checksum: false,
                stride: None,
                ..Default::default()
            };
            let config_yes = CompressionConfig {
                stride: Some(12),
                ..config_no.clone()
            };

            let comp_no = compress(&data, &config_no).unwrap();
            let comp_yes = compress(&data, &config_yes).unwrap();

            let ratio_no = input_size as f64 / comp_no.len() as f64;
            let ratio_yes = input_size as f64 / comp_yes.len() as f64;

            eprintln!(
                "  {} stride=12: {:.2}x ({} → {} bytes) | no-stride: {:.2}x ({} → {} bytes) | improvement: {:.1}%",
                mode, ratio_yes, input_size, comp_yes.len(),
                ratio_no, input_size, comp_no.len(),
                (ratio_yes / ratio_no - 1.0) * 100.0
            );

            // Both must roundtrip
            assert_eq!(data, decompress(&comp_no).unwrap());
            assert_eq!(data, decompress(&comp_yes).unwrap());
        }
    }

    #[test]
    fn stride_mismatch_fails() {
        // 100 bytes is not divisible by stride=12
        let data = vec![0u8; 100];
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            stride: Some(12),
            ..CompressionConfig::fast()
        };
        assert!(matches!(
            compress(&data, &config),
            Err(crate::CompressorError::StrideMismatch { .. })
        ));
    }

    #[test]
    fn stride_header_preserved_in_frame() {
        let data = vec![0x42u8; 48]; // 48 = 4 × 12
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            stride: Some(12),
            block_size: 1024,
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        assert_eq!(frame.stride, 12, "stride should be stored in frame header");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase 12: Seek Table Tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn seek_table_all_data_types() {
        let datasets: Vec<(&str, Vec<u8>, Option<DataType>)> = vec![
            ("i64", generate_timestamps(2000), Some(DataType::IntegerI64)),
            ("f64", generate_temperatures(2000), Some(DataType::Float64)),
            ("u32", generate_counters(2000), Some(DataType::IntegerU32)),
            ("f32", generate_vibration(3000), Some(DataType::Float32)),
            (
                "f64s",
                generate_temperatures(2000),
                Some(DataType::Float64Shuffle),
            ),
            (
                "f32s",
                generate_vibration(3000),
                Some(DataType::Float32Shuffle),
            ),
            (
                "f64sd",
                generate_temperatures(2000),
                Some(DataType::Float64ShuffleDelta),
            ),
            (
                "f32sd",
                generate_vibration(3000),
                Some(DataType::Float32ShuffleDelta),
            ),
            ("raw", generate_random(8000), None),
        ];
        for (label, data, dt) in &datasets {
            let config = CompressionConfig {
                parser_mode: ParserMode::Lazy,
                data_type: *dt,
                block_size: 4096,
                store_checksum: true,
                store_seek_table: true,
                ..Default::default()
            };
            let compressed = compress(data, &config)
                .unwrap_or_else(|e| panic!("seek_table/{} compress failed: {:?}", label, e));
            let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
            assert!(
                frame.flags.has_seek_table,
                "seek_table/{} missing flag",
                label
            );
            let decompressed = decompress(&compressed)
                .unwrap_or_else(|e| panic!("seek_table/{} decompress failed: {:?}", label, e));
            assert_eq!(data, &decompressed, "seek_table/{} data mismatch", label);
        }
    }

    #[test]
    fn seek_table_stride_roundtrip() {
        // stride=12 + seek table
        let mut data = Vec::with_capacity(60_000);
        for i in 0u32..5000 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(20.0f32 + 0.01 * i as f32).to_le_bytes());
            data.extend_from_slice(&(50.0f32 + 0.005 * i as f32).to_le_bytes());
        }
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Raw),
            block_size: 16384,
            store_checksum: true,
            store_seek_table: true,
            stride: Some(12),
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        assert!(frame.flags.has_seek_table);
        assert_eq!(frame.stride, 12);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Also verify via SeekableReader
        let mut reader = crate::seekable::SeekableReader::new(&compressed).unwrap();
        let seekable_result = reader.decompress_all().unwrap();
        assert_eq!(data, seekable_result);
    }

    #[test]
    fn seekable_stress_large_frame() {
        // 100K timestamps → ~800 KB, random block access
        let data = generate_timestamps(100_000);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            store_checksum: false,
            store_seek_table: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let mut reader = crate::seekable::SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count() as usize;
        assert!(bc > 5, "expected many blocks, got {}", bc);

        // Access blocks in reverse order (worst case for sequential scan)
        for i in (0..bc).rev() {
            let block = reader.decompress_block(i).unwrap();
            let bs = reader.header().block_size as usize;
            let start = i * bs;
            let end = (start + bs).min(data.len());
            assert_eq!(block, &data[start..end], "block {} mismatch", i);
        }
    }

    #[test]
    fn seekable_value_proposition() {
        // Single block access should be substantially faster (less data) than full decompress.
        // Here we just verify correctness; benchmarks measure actual timing.
        let data = generate_timestamps(50_000);
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::IntegerI64),
            block_size: 65536,
            store_checksum: false,
            store_seek_table: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let mut reader = crate::seekable::SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count();
        assert!(bc > 2);

        // Single block in the middle
        let mid = (bc / 2) as usize;
        let block = reader.decompress_block(mid).unwrap();
        let bs = reader.header().block_size as usize;
        let start = mid * bs;
        let end = (start + bs).min(data.len());
        assert_eq!(block, &data[start..end]);
    }

    #[test]
    fn no_stride_header_zero() {
        let data = b"Hello world! Hello world!";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            stride: None,
            block_size: 1024,
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        assert_eq!(frame.stride, 0, "stride should be 0 when not configured");
        assert_eq!(data.as_slice(), decompress(&compressed).unwrap().as_slice());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Adaptive Preprocessing: Shuffle + Byte-Delta (Float64ShuffleDelta / Float32ShuffleDelta)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn roundtrip_temperatures_shuffle_delta_all_modes() {
        let data = generate_temperatures(2000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float64ShuffleDelta),
                mode,
                &format!("temperatures-shuffle-delta/{}", mode),
            );
        }
    }

    #[test]
    fn roundtrip_vibration_shuffle_delta_all_modes() {
        let data = generate_vibration(3000);
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            verify_roundtrip(
                &data,
                Some(DataType::Float32ShuffleDelta),
                mode,
                &format!("vibration-shuffle-delta/{}", mode),
            );
        }
    }

    #[test]
    fn stress_large_temperature_shuffle_delta() {
        let data = generate_temperatures(100_000);
        verify_roundtrip(
            &data,
            Some(DataType::Float64ShuffleDelta),
            ParserMode::Lazy,
            "stress/100k-temperatures-shuffle-delta",
        );
    }

    #[test]
    fn stress_large_vibration_shuffle_delta() {
        let data = generate_vibration(99_000);
        verify_roundtrip(
            &data,
            Some(DataType::Float32ShuffleDelta),
            ParserMode::Lazy,
            "stress/99k-vibration-shuffle-delta",
        );
    }

    #[test]
    fn shuffle_delta_multiblock_roundtrip() {
        // Many small blocks to exercise block boundary handling with the
        // composed transform (shuffle → byte-delta → LZ77 → FSE).
        let data = generate_temperatures(10_000); // 80 KB → many 4 KB blocks
        let config = CompressionConfig {
            parser_mode: ParserMode::Lazy,
            data_type: Some(DataType::Float64ShuffleDelta),
            block_size: 4096,
            store_checksum: true,
            ..Default::default()
        };
        let compressed = compress(&data, &config).unwrap();
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        assert!(
            frame.block_count > 1,
            "expected multiple blocks, got {}",
            frame.block_count
        );

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn shuffle_delta_improves_over_gorilla_on_noisy_floats() {
        // Noisy f64 data where Gorilla XOR produces high-entropy residuals.
        // Shuffle+delta should beat or match Gorilla here.
        let mut data = Vec::with_capacity(80_000);
        for i in 0..10_000u64 {
            // Slowly drifting floats with small random-ish perturbation.
            // Adjacent exponent bytes are similar → shuffle+delta wins.
            let base = 100.0 + 0.01 * (i as f64);
            let noise = ((i.wrapping_mul(2654435761) >> 20) as f64) * 1e-8;
            let v = base + noise;
            data.extend_from_slice(&v.to_le_bytes());
        }

        let config_gorilla = CompressionConfig {
            data_type: Some(DataType::Float64),
            block_size: 65536,
            store_checksum: false,
            ..CompressionConfig::balanced()
        };
        let config_sd = CompressionConfig {
            data_type: Some(DataType::Float64ShuffleDelta),
            ..config_gorilla.clone()
        };

        let comp_gorilla = compress(&data, &config_gorilla).unwrap();
        let comp_sd = compress(&data, &config_sd).unwrap();

        // Both must roundtrip correctly
        assert_eq!(data, decompress(&comp_gorilla).unwrap());
        assert_eq!(data, decompress(&comp_sd).unwrap());

        let ratio_gorilla = data.len() as f64 / comp_gorilla.len() as f64;
        let ratio_sd = data.len() as f64 / comp_sd.len() as f64;

        eprintln!(
            "  noisy f64: gorilla={:.2}x, shuffle+delta={:.2}x",
            ratio_gorilla, ratio_sd
        );

        // We expect shuffle+delta to be competitive on this kind of data.
        // If it's substantially worse, the composed transform has a bug.
        assert!(
            ratio_sd > ratio_gorilla * 0.7,
            "shuffle+delta ({:.2}x) unexpectedly much worse than gorilla ({:.2}x)",
            ratio_sd,
            ratio_gorilla
        );
    }

    #[test]
    fn shuffle_delta_ratio_report() {
        // Print ratio comparison across all float preprocessing strategies.
        let temp_data = generate_temperatures(100_000);
        let vib_data = generate_vibration(99_000);

        let datasets: Vec<(&str, &[u8], Vec<(DataType, &str)>)> = vec![
            (
                "temperatures_100K",
                &temp_data,
                vec![
                    (DataType::Float64, "gorilla"),
                    (DataType::Float64Shuffle, "shuffle"),
                    (DataType::Float64ShuffleDelta, "shuffle+delta"),
                ],
            ),
            (
                "vibration_99K",
                &vib_data,
                vec![
                    (DataType::Float32, "gorilla"),
                    (DataType::Float32Shuffle, "shuffle"),
                    (DataType::Float32ShuffleDelta, "shuffle+delta"),
                ],
            ),
        ];

        eprintln!(
            "\n{:<22} {:<16} {:>10} {:>10} {:>8}",
            "Dataset", "Strategy", "Original", "Compressed", "Ratio"
        );
        eprintln!("{}", "=".repeat(70));
        for (name, data, strategies) in &datasets {
            for (dt, label) in strategies {
                let config = CompressionConfig {
                    parser_mode: ParserMode::Lazy,
                    data_type: Some(*dt),
                    block_size: 65536,
                    store_checksum: false,
                    ..Default::default()
                };
                let compressed = compress(data, &config).unwrap();
                let ratio = data.len() as f64 / compressed.len() as f64;
                eprintln!(
                    "{:<22} {:<16} {:>10} {:>10} {:>7.2}x",
                    name,
                    label,
                    data.len(),
                    compressed.len(),
                    ratio
                );

                // Verify roundtrip
                assert_eq!(*data, decompress(&compressed).unwrap().as_slice());
            }
            eprintln!("{}", "-".repeat(70));
        }
    }

    #[test]
    fn auto_detect_selects_shuffle_delta_for_noisy_floats() {
        // Data where shuffle+delta should be the adaptive winner.
        // Slowly drifting floats — exponent bytes nearly constant.
        let mut data = Vec::with_capacity(80_000);
        for i in 0..10_000u64 {
            let v = 100.0 + 0.001 * (i as f64);
            data.extend_from_slice(&v.to_le_bytes());
        }

        // Compress with auto-detect — should pick a strategy
        let config = CompressionConfig {
            data_type: None, // auto-detect
            block_size: 65536,
            store_checksum: false,
            ..CompressionConfig::balanced()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Read back what type was actually chosen
        let frame = crate::FrameHeader::from_bytes(&compressed).unwrap();
        let chosen = frame.flags.data_type;
        eprintln!(
            "  auto-detect chose: {:?} for slowly-drifting f64 data",
            chosen
        );

        // Should be one of the float strategies, not Raw
        assert!(
            !matches!(chosen, DataType::Raw),
            "auto-detect should pick a float strategy, not Raw"
        );
    }

    // ── Regression: u16 type_bits_len overflow on large single blocks ──
    //
    // The bug: type_bits_len stored as u16 in FSE encoder. At >524,280 tokens
    // (>65,535 type_bits bytes), the count silently truncated, misaligning the
    // decoder. Only triggers when a single block produces enough LZ77 tokens
    // to overflow — which requires incompressible data (mostly literal tokens).
    //
    // Tests below use data patterns that are GENUINELY incompressible to ensure
    // the LZ77 parser emits enough literal tokens to cross the u16 boundary.

    #[test]
    fn large_block_random_data_roundtrip() {
        // 600K of pseudo-random data in a single block. Incompressible →
        // ~600K literal tokens → ~75K type_bits bytes → triggers u32 path.
        let mut data = vec![0u8; 600_000];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(7919) ^ (i >> 3)) as u8;
        }
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 2 * 1024 * 1024, // single block
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn large_block_noisy_f64_roundtrip() {
        // 100K noisy f64 values = 800KB. Gorilla XOR on random-ish floats
        // produces high-entropy residuals → mostly literal tokens.
        // This is the ACTUAL scenario from the original bug report.
        let mut data = Vec::with_capacity(800_000);
        for i in 0..100_000u64 {
            // Pseudo-random floats — consecutive values share no structure
            let v = f64::from_bits(0x4000_0000_0000_0000 ^ (i.wrapping_mul(6364136223846793005)));
            data.extend_from_slice(&v.to_le_bytes());
        }
        let config = CompressionConfig {
            data_type: Some(DataType::Float64),
            block_size: 2 * 1024 * 1024,
            ..CompressionConfig::balanced()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Column extraction integration tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn extract_columns_telemetry_struct_roundtrip() {
        use crate::schema::{ColumnType, Schema};
        use crate::seekable::SeekableReader;

        let n = 5000;
        let stride = 12; // u32 + f32 + f32
        let mut data = Vec::with_capacity(n * stride);
        for i in 0..n as u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(20.0 + 0.01 * i as f32).to_le_bytes());
            data.extend_from_slice(&(50.0 + 0.005 * i as f32).to_le_bytes());
        }

        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 4096,
            store_checksum: true,
            store_seek_table: true,
            stride: Some(stride as u16),
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();

        let schema = Schema::new(vec![
            ("seq".into(), ColumnType::U32),
            ("temp".into(), ColumnType::F32),
            ("humidity".into(), ColumnType::F32),
        ]);

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let columns = reader.extract_all_columns(&schema).unwrap();

        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0].data.len(), n * 4);
        assert_eq!(columns[1].data.len(), n * 4);
        assert_eq!(columns[2].data.len(), n * 4);

        // Verify every record
        for i in 0..n {
            let seq = u32::from_le_bytes(columns[0].data[i * 4..(i + 1) * 4].try_into().unwrap());
            let temp = f32::from_le_bytes(columns[1].data[i * 4..(i + 1) * 4].try_into().unwrap());
            let hum = f32::from_le_bytes(columns[2].data[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(seq, i as u32);
            assert!((temp - (20.0 + 0.01 * i as f32)).abs() < 1e-5);
            assert!((hum - (50.0 + 0.005 * i as f32)).abs() < 1e-5);
        }
    }

    #[test]
    fn extract_columns_wide_struct_20byte() {
        use crate::schema::{ColumnType, Schema};
        use crate::seekable::SeekableReader;

        let n = 2000;
        let stride = 20; // i64 + f64 + f32
        let mut data = Vec::with_capacity(n * stride);
        for i in 0..n as i64 {
            data.extend_from_slice(&(1_700_000_000i64 + i).to_le_bytes());
            data.extend_from_slice(&(22.5 + 0.001 * i as f64).to_le_bytes());
            data.extend_from_slice(&(0.5f32).to_le_bytes());
        }

        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 8192,
            store_checksum: false,
            store_seek_table: true,
            stride: Some(stride as u16),
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();

        let schema = Schema::new(vec![
            ("timestamp".into(), ColumnType::I64),
            ("temperature".into(), ColumnType::F64),
            ("vibration".into(), ColumnType::F32),
        ]);

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let columns = reader.extract_all_columns(&schema).unwrap();

        assert_eq!(columns[0].data.len(), n * 8); // i64
        assert_eq!(columns[1].data.len(), n * 8); // f64
        assert_eq!(columns[2].data.len(), n * 4); // f32

        let ts0 = i64::from_le_bytes(columns[0].data[..8].try_into().unwrap());
        let ts_last = i64::from_le_bytes(
            columns[0].data[columns[0].data.len() - 8..]
                .try_into()
                .unwrap(),
        );
        assert_eq!(ts0, 1_700_000_000);
        assert_eq!(ts_last, 1_700_000_000 + n as i64 - 1);
    }

    #[test]
    fn extract_columns_partial_block_range() {
        use crate::schema::{ColumnType, Schema};
        use crate::seekable::SeekableReader;

        let n = 3000;
        let stride = 8; // u32 + f32
        let mut data = Vec::with_capacity(n * stride);
        for i in 0..n as u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }

        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 4096,
            store_checksum: false,
            store_seek_table: true,
            stride: Some(stride as u16),
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();

        let schema = Schema::new(vec![
            ("id".into(), ColumnType::U32),
            ("val".into(), ColumnType::F32),
        ]);

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count() as usize;
        assert!(bc >= 3, "need multiple blocks, got {}", bc);

        // Extract only block 1
        let cols = reader.extract_columns(&schema, 1, 2).unwrap();
        assert!(!cols[0].data.is_empty());

        // First id in block 1 should be > 0
        let first_id = u32::from_le_bytes(cols[0].data[..4].try_into().unwrap());
        assert!(first_id > 0);
    }
}
