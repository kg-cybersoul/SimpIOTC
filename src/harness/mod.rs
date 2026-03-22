//! # Harness — Benchmark Data Generators & Comparison Runner
//!
//! This module provides:
//!
//! 1. **Synthetic IoT data generators** — Realistic time-series patterns
//!    (timestamps, temperatures, counters, vibration, etc.) for benchmarks
//!    and stress tests.
//!
//! 2. **Comparison runner** — Measures compression ratio, compress throughput,
//!    and decompress throughput for our compressor, with optional comparison
//!    against external tools (xz, zstd) when available.
//!
//! ## Data Generators
//!
//! Each generator produces raw `Vec<u8>` in little-endian format, ready to
//! feed directly into the compression pipeline. The generators model real
//! IoT sensor patterns:
//!
//! - **Timestamps**: Monotonic i64 millisecond timestamps with sub-ms jitter.
//! - **Temperatures**: Slowly varying f64 with diurnal drift and noise.
//! - **Counters**: Monotonic u32 with occasional resets/jumps.
//! - **Vibration**: Fast-oscillating f32 (accelerometer-like).
//! - **Constant**: Perfectly uniform values (best-case for delta encoding).
//! - **Random**: Uniform random bytes (worst-case baseline).

pub mod verify;

use crate::parallel::{compress, decompress};
use crate::{CompressionConfig, DataType, ParserMode};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// Data Generators
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate synthetic i64 timestamps: monotonic millisecond timestamps
/// with a ~1s base interval and small sub-ms jitter.
///
/// Models a sensor reporting every ~1000ms with clock jitter of ±50µs.
/// This produces data where delta-of-delta encoding excels: first deltas
/// are ~1_000_000 (µs), second deltas are ±50.
pub fn generate_timestamps(count: usize) -> Vec<u8> {
    let base: i64 = 1_700_000_000_000; // ~2023-11-14 epoch ms
    let interval: i64 = 1_000_000; // 1 second in µs
    let mut bytes = Vec::with_capacity(count * 8);
    for i in 0..count as i64 {
        // Jitter: deterministic pseudo-random in [-50, +50] range
        let jitter = ((i * 7 + 13) % 101) - 50;
        let ts = base + i * interval + jitter;
        bytes.extend_from_slice(&ts.to_le_bytes());
    }
    bytes
}

/// Generate synthetic f64 temperature readings: slowly varying sinusoidal
/// pattern with small Gaussian-like noise.
///
/// Models an indoor temperature sensor reading ~22.5°C with ±0.5°C diurnal
/// drift and ±0.01°C measurement noise. Gorilla XOR encoding excels here
/// because consecutive readings share most mantissa bits.
pub fn generate_temperatures(count: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 8);
    for i in 0..count {
        let t = i as f64;
        // Slow sinusoidal drift (period ~6283 samples ≈ 1.7 hours at 1Hz)
        let drift = 0.5 * (t * 0.001).sin();
        // Tiny noise: deterministic, bounded ±0.01
        let noise = 0.01 * ((t * 7.3 + 2.1).sin() * (t * 13.7 + 0.3).cos());
        let temp = 22.5 + drift + noise;
        bytes.extend_from_slice(&temp.to_le_bytes());
    }
    bytes
}

/// Generate synthetic u32 monotonic counters: strictly increasing with
/// small steps and occasional larger jumps.
///
/// Models a packet counter or event counter that increments by 1-3 per
/// sample with rare bursts of 100+.
pub fn generate_counters(count: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 4);
    let mut value: u32 = 1000;
    for i in 0..count {
        bytes.extend_from_slice(&value.to_le_bytes());
        // Normal: increment 1-3. Every 500th sample: burst of 100-200.
        if i % 500 == 499 {
            value = value.wrapping_add(100 + (i as u32 % 101));
        } else {
            value = value.wrapping_add(1 + (i as u32 % 3));
        }
    }
    bytes
}

/// Generate synthetic f32 vibration data: fast-oscillating accelerometer
/// readings with multiple frequency components.
///
/// Models a 3-axis vibration sensor (interleaved X,Y,Z) at high sample
/// rates. Less compressible than slowly-varying data, but Gorilla XOR
/// still captures shared exponent bits.
pub fn generate_vibration(count: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 4);
    for i in 0..count {
        let t = i as f32;
        let axis = i % 3; // interleaved X, Y, Z
        let base_freq = match axis {
            0 => 0.1_f32,
            1 => 0.07,
            _ => 0.13,
        };
        // Multi-frequency signal with deterministic noise
        let signal = 2.0 * (t * base_freq).sin()
            + 0.5 * (t * base_freq * 3.0 + 0.7).sin()
            + 0.1 * (t * base_freq * 7.0 + 1.3).sin();
        bytes.extend_from_slice(&signal.to_le_bytes());
    }
    bytes
}

/// Generate perfectly constant i64 values.
///
/// Best-case scenario for delta encoding: all deltas are zero, second
/// deltas are zero. The entire stream compresses to essentially nothing.
pub fn generate_constant(count: usize) -> Vec<u8> {
    let value: i64 = 42;
    let mut bytes = Vec::with_capacity(count * 8);
    for _ in 0..count {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Generate uniformly random bytes.
///
/// Worst-case baseline: incompressible data. The compressor should not
/// expand this by more than the frame header overhead.
pub fn generate_random(size: usize) -> Vec<u8> {
    // Simple LCG for deterministic "random" bytes — we don't need
    // cryptographic quality, just byte-level entropy.
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut bytes = Vec::with_capacity(size);
    for _ in 0..size {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        bytes.push((state >> 33) as u8);
    }
    bytes
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dataset Description
// ═══════════════════════════════════════════════════════════════════════════════

/// A named dataset with its data, data type hint, and element count.
pub struct Dataset {
    /// Human-readable name (e.g., "timestamps_10k").
    pub name: String,
    /// Raw byte data in little-endian format.
    pub data: Vec<u8>,
    /// Data type hint for the preprocessor.
    pub data_type: Option<DataType>,
    /// Number of logical elements (data.len() / element_size).
    pub element_count: usize,
}

/// Generate a standard suite of datasets at a given element count.
///
/// Returns datasets covering the full range of IoT data patterns,
/// from best-case (constant) to worst-case (random).
pub fn standard_datasets(element_count: usize) -> Vec<Dataset> {
    vec![
        Dataset {
            name: format!("timestamps_{}", element_count),
            data: generate_timestamps(element_count),
            data_type: Some(DataType::IntegerI64),
            element_count,
        },
        Dataset {
            name: format!("temperatures_{}", element_count),
            data: generate_temperatures(element_count),
            data_type: Some(DataType::Float64),
            element_count,
        },
        Dataset {
            name: format!("counters_{}", element_count),
            data: generate_counters(element_count),
            data_type: Some(DataType::IntegerU32),
            element_count,
        },
        Dataset {
            name: format!("vibration_{}", element_count),
            data: generate_vibration(element_count),
            data_type: Some(DataType::Float32),
            element_count,
        },
        Dataset {
            name: format!("constant_{}", element_count),
            data: generate_constant(element_count),
            data_type: Some(DataType::IntegerI64),
            element_count,
        },
        Dataset {
            name: format!("random_{}", element_count * 8),
            data: generate_random(element_count * 8),
            data_type: None, // raw
            element_count: element_count * 8,
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Comparison Runner
// ═══════════════════════════════════════════════════════════════════════════════

/// Results from a single compression benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the compressor (e.g., "iotc-lazy", "zstd -19").
    pub compressor: String,
    /// Name of the dataset.
    pub dataset: String,
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Compression ratio (original / compressed). Higher = better.
    pub ratio: f64,
    /// Compression throughput in MB/s.
    pub compress_mb_s: f64,
    /// Decompression throughput in MB/s.
    pub decompress_mb_s: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<16} {:<24} {:>10} → {:>10}  {:>6.2}x  {:>8.1} MB/s  {:>8.1} MB/s",
            self.compressor,
            self.dataset,
            self.original_size,
            self.compressed_size,
            self.ratio,
            self.compress_mb_s,
            self.decompress_mb_s,
        )
    }
}

/// Run our compressor on a dataset with a given parser mode.
/// Returns the benchmark result with throughput measurements.
///
/// Performs multiple iterations (minimum 3) to get stable timing.
pub fn benchmark_iotc(
    dataset: &Dataset,
    parser_mode: ParserMode,
    block_size: usize,
) -> BenchmarkResult {
    let config = CompressionConfig {
        parser_mode,
        data_type: dataset.data_type,
        block_size,
        store_checksum: true,
        ..Default::default()
    };

    let compressor_name = format!("iotc-{}", parser_mode);

    // Warm-up run
    let compressed = compress(&dataset.data, &config).expect("compression failed");

    // Measure compression: run enough iterations for stable timing
    let (compress_time, _) = measure_min(3, || compress(&dataset.data, &config).unwrap());

    // Measure decompression
    let (decompress_time, decompressed) = measure_min(3, || decompress(&compressed).unwrap());

    // Verify roundtrip correctness
    assert_eq!(
        decompressed.len(),
        dataset.data.len(),
        "roundtrip length mismatch for {}",
        dataset.name
    );
    assert_eq!(
        decompressed, dataset.data,
        "roundtrip data mismatch for {}",
        dataset.name
    );

    let original_size = dataset.data.len();
    let compressed_size = compressed.len();

    BenchmarkResult {
        compressor: compressor_name,
        dataset: dataset.name.clone(),
        original_size,
        compressed_size,
        ratio: original_size as f64 / compressed_size as f64,
        compress_mb_s: original_size as f64 / 1_000_000.0 / compress_time.as_secs_f64(),
        decompress_mb_s: original_size as f64 / 1_000_000.0 / decompress_time.as_secs_f64(),
    }
}

/// Run a full benchmark suite: all parser modes × all datasets.
///
/// Returns results sorted by dataset then parser mode.
pub fn run_full_suite(element_count: usize, block_size: usize) -> Vec<BenchmarkResult> {
    let datasets = standard_datasets(element_count);
    let modes = [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal];

    let mut results = Vec::new();
    for dataset in &datasets {
        for &mode in &modes {
            results.push(benchmark_iotc(dataset, mode, block_size));
        }
    }
    results
}

/// Print a formatted comparison table to stderr.
pub fn print_results(results: &[BenchmarkResult]) {
    eprintln!(
        "{:<16} {:<24} {:>10}    {:>10}  {:>6}   {:>12}  {:>12}",
        "Compressor", "Dataset", "Original", "Compressed", "Ratio", "Compress", "Decompress"
    );
    eprintln!("{}", "-".repeat(100));
    for r in results {
        eprintln!("{}", r);
    }
}

/// Run a function multiple times and return the minimum elapsed time
/// and the result from the last run.
fn measure_min<T, F: FnMut() -> T>(iterations: usize, mut f: F) -> (std::time::Duration, T) {
    let mut best_time = std::time::Duration::MAX;
    let mut result = f(); // warm-up
    for _ in 0..iterations {
        let start = Instant::now();
        result = f();
        let elapsed = start.elapsed();
        if elapsed < best_time {
            best_time = elapsed;
        }
    }
    (best_time, result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// External Tool Comparison (only available in test/bench — uses tempfile dev-dep)
// ═══════════════════════════════════════════════════════════════════════════════

/// Check if an external compression tool is available on PATH.
#[cfg(test)]
pub fn tool_available(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Benchmark an external compression tool by shelling out.
/// Returns None if the tool is not available.
///
/// `compress_cmd` should be a closure that takes (input_path, output_path)
/// and returns the Command to run.
#[cfg(test)]
pub fn benchmark_external<F, G>(
    name: &str,
    dataset: &Dataset,
    compress_cmd: F,
    decompress_cmd: G,
) -> Option<BenchmarkResult>
where
    F: Fn(&std::path::Path, &std::path::Path) -> std::process::Command,
    G: Fn(&std::path::Path, &std::path::Path) -> std::process::Command,
{
    use std::io::Write;

    let dir = tempfile::tempdir().ok()?;
    let input_path = dir.path().join("input.bin");
    let compressed_path = dir.path().join("compressed.bin");
    let decompressed_path = dir.path().join("decompressed.bin");

    // Write input
    std::fs::File::create(&input_path)
        .ok()?
        .write_all(&dataset.data)
        .ok()?;

    // Compress
    let start = Instant::now();
    let status = compress_cmd(&input_path, &compressed_path).status().ok()?;
    let compress_time = start.elapsed();
    if !status.success() {
        return None;
    }

    let compressed_size = std::fs::metadata(&compressed_path).ok()?.len() as usize;

    // Decompress
    let start = Instant::now();
    let status = decompress_cmd(&compressed_path, &decompressed_path)
        .status()
        .ok()?;
    let decompress_time = start.elapsed();
    if !status.success() {
        return None;
    }

    // Verify
    let decompressed = std::fs::read(&decompressed_path).ok()?;
    if decompressed != dataset.data {
        return None;
    }

    let original_size = dataset.data.len();
    Some(BenchmarkResult {
        compressor: name.to_string(),
        dataset: dataset.name.clone(),
        original_size,
        compressed_size,
        ratio: original_size as f64 / compressed_size as f64,
        compress_mb_s: original_size as f64 / 1_000_000.0 / compress_time.as_secs_f64(),
        decompress_mb_s: original_size as f64 / 1_000_000.0 / decompress_time.as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_produce_correct_sizes() {
        let count = 1000;

        let ts = generate_timestamps(count);
        assert_eq!(ts.len(), count * 8, "timestamps: i64 = 8 bytes each");

        let temps = generate_temperatures(count);
        assert_eq!(temps.len(), count * 8, "temperatures: f64 = 8 bytes each");

        let counters = generate_counters(count);
        assert_eq!(counters.len(), count * 4, "counters: u32 = 4 bytes each");

        let vib = generate_vibration(count);
        assert_eq!(vib.len(), count * 4, "vibration: f32 = 4 bytes each");

        let constant = generate_constant(count);
        assert_eq!(constant.len(), count * 8, "constant: i64 = 8 bytes each");

        let random = generate_random(count);
        assert_eq!(random.len(), count, "random: 1 byte each");
    }

    #[test]
    fn timestamps_are_monotonic() {
        let data = generate_timestamps(1000);
        let values: Vec<i64> = data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for w in values.windows(2) {
            assert!(w[1] > w[0], "timestamps must be strictly increasing");
        }
    }

    #[test]
    fn temperatures_are_reasonable() {
        let data = generate_temperatures(10000);
        let values: Vec<f64> = data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for &v in &values {
            assert!(
                (20.0..25.0).contains(&v),
                "temperature {} out of expected range [20, 25]",
                v
            );
        }
    }

    #[test]
    fn counters_are_monotonic() {
        let data = generate_counters(2000);
        let values: Vec<u32> = data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for w in values.windows(2) {
            assert!(w[1] > w[0], "counters must be strictly increasing");
        }
    }

    #[test]
    fn vibration_values_bounded() {
        let data = generate_vibration(10000);
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for &v in &values {
            assert!(
                v.abs() < 10.0,
                "vibration {} out of expected range [-10, 10]",
                v
            );
        }
    }

    #[test]
    fn random_has_byte_entropy() {
        let data = generate_random(100_000);
        // All 256 byte values should appear in 100K random bytes.
        let mut seen = [false; 256];
        for &b in &data {
            seen[b as usize] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        assert!(
            distinct >= 200,
            "random data should cover most byte values, got {}",
            distinct
        );
    }

    #[test]
    fn generators_are_deterministic() {
        let a = generate_timestamps(100);
        let b = generate_timestamps(100);
        assert_eq!(a, b, "generators must be deterministic");

        let a = generate_temperatures(100);
        let b = generate_temperatures(100);
        assert_eq!(a, b);

        let a = generate_random(100);
        let b = generate_random(100);
        assert_eq!(a, b);
    }

    #[test]
    fn standard_datasets_correct_count() {
        let datasets = standard_datasets(100);
        assert_eq!(datasets.len(), 6, "should produce 6 dataset types");

        // Each should have a name and non-empty data
        for ds in &datasets {
            assert!(!ds.name.is_empty());
            assert!(!ds.data.is_empty());
            assert!(ds.element_count > 0);
        }
    }

    #[test]
    fn benchmark_iotc_roundtrips_correctly() {
        // Small dataset to keep test fast
        let dataset = Dataset {
            name: "test_timestamps".to_string(),
            data: generate_timestamps(500),
            data_type: Some(DataType::IntegerI64),
            element_count: 500,
        };

        let result = benchmark_iotc(&dataset, ParserMode::Greedy, 16384);
        assert!(result.ratio > 1.0, "timestamps should compress");
        assert!(result.compress_mb_s > 0.0);
        assert!(result.decompress_mb_s > 0.0);
    }

    #[test]
    fn benchmark_result_display() {
        let r = BenchmarkResult {
            compressor: "iotc-lazy".into(),
            dataset: "test_data".into(),
            original_size: 80000,
            compressed_size: 10000,
            ratio: 8.0,
            compress_mb_s: 150.0,
            decompress_mb_s: 400.0,
        };
        let s = format!("{}", r);
        assert!(s.contains("iotc-lazy"));
        assert!(s.contains("test_data"));
        assert!(s.contains("8.00x"));
    }
}
