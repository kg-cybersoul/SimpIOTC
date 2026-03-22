//! # Preprocessor Module
//!
//! The critical differentiator for IoT time-series compression. This module
//! losslessly transforms structured numeric streams into highly compressible
//! byte sequences before they reach the LZ77 match finder.
//!
//! Two preprocessing strategies are implemented:
//!
//! - **Delta-of-delta encoding** for integer streams (timestamps, counters).
//!   Converts slowly changing sequences into streams of near-zero values.
//!
//! - **Gorilla XOR encoding** for floating-point streams (sensor readings).
//!   XORs adjacent values, producing streams dominated by zero bits that
//!   LZ77 handles trivially.
//!
//! Both transforms are fully reversible and add a small header so the
//! decompressor knows how to invert them.

pub mod bitshuffle;
pub mod byte_delta;
pub mod delta;
pub mod entropy_probe;
pub mod gorilla_xor;
pub mod stride;

use crate::{CompressorError, DataType, Result};

// ═══════════════════════════════════════════════════════════════════════════════
// Preprocessed Data Container
// ═══════════════════════════════════════════════════════════════════════════════

/// The output of a preprocessing pass. Contains the transformed data plus
/// metadata the decompressor needs to invert the transform.
#[derive(Debug, Clone)]
pub struct PreprocessedData {
    /// Which preprocessor was applied.
    pub data_type: DataType,
    /// Number of elements in the original stream (not byte count).
    pub element_count: u64,
    /// The transformed byte stream, ready for LZ77.
    pub data: Vec<u8>,
}

/// Configuration for the preprocessor stage.
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    /// The data type to use. If `None`, auto-detection is attempted.
    pub data_type: Option<DataType>,
    /// For delta encoding: whether to apply a second delta pass.
    /// Double-delta is almost always beneficial for timestamps.
    pub double_delta: bool,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            data_type: None,
            double_delta: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply the appropriate preprocessing transform to raw input data.
///
/// If `config.data_type` is `None`, auto-detection is attempted by examining
/// the data for patterns characteristic of integer or floating-point series.
/// When in doubt, falls back to `DataType::Raw` (passthrough).
pub fn preprocess(data: &[u8], config: &PreprocessorConfig) -> Result<PreprocessedData> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let data_type = config.data_type.unwrap_or_else(|| auto_detect(data));

    match data_type {
        DataType::Raw => Ok(PreprocessedData {
            data_type: DataType::Raw,
            element_count: data.len() as u64,
            data: data.to_vec(),
        }),

        DataType::IntegerI64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_i64s(data);
            let encoded = delta::encode_i64(&values, config.double_delta)?;
            Ok(PreprocessedData {
                data_type: DataType::IntegerI64,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::IntegerU64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_u64s(data);
            let encoded = delta::encode_u64(&values, config.double_delta)?;
            Ok(PreprocessedData {
                data_type: DataType::IntegerU64,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::IntegerI32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_i32s(data);
            let encoded = delta::encode_i32(&values, config.double_delta)?;
            Ok(PreprocessedData {
                data_type: DataType::IntegerI32,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::IntegerU32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_u32s(data);
            let encoded = delta::encode_u32(&values, config.double_delta)?;
            Ok(PreprocessedData {
                data_type: DataType::IntegerU32,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::Float64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_f64s(data);
            let encoded = gorilla_xor::encode_f64(&values)?;
            Ok(PreprocessedData {
                data_type: DataType::Float64,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::Float32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_f32s(data);
            let encoded = gorilla_xor::encode_f32(&values)?;
            Ok(PreprocessedData {
                data_type: DataType::Float32,
                element_count: values.len() as u64,
                data: encoded,
            })
        }

        DataType::Float64Shuffle => {
            validate_alignment(data, 8)?;
            let element_count = (data.len() / 8) as u64;
            let encoded = bitshuffle::shuffle(data, 8);
            Ok(PreprocessedData {
                data_type: DataType::Float64Shuffle,
                element_count,
                data: encoded,
            })
        }

        DataType::Float32Shuffle => {
            validate_alignment(data, 4)?;
            let element_count = (data.len() / 4) as u64;
            let encoded = bitshuffle::shuffle(data, 4);
            Ok(PreprocessedData {
                data_type: DataType::Float32Shuffle,
                element_count,
                data: encoded,
            })
        }

        DataType::Float64ShuffleDelta => {
            validate_alignment(data, 8)?;
            let element_count = (data.len() / 8) as u64;
            let shuffled = bitshuffle::shuffle(data, 8);
            let encoded = byte_delta::encode(&shuffled);
            Ok(PreprocessedData {
                data_type: DataType::Float64ShuffleDelta,
                element_count,
                data: encoded,
            })
        }

        DataType::Float32ShuffleDelta => {
            validate_alignment(data, 4)?;
            let element_count = (data.len() / 4) as u64;
            let shuffled = bitshuffle::shuffle(data, 4);
            let encoded = byte_delta::encode(&shuffled);
            Ok(PreprocessedData {
                data_type: DataType::Float32ShuffleDelta,
                element_count,
                data: encoded,
            })
        }
    }
}

/// Workspace-aware preprocess. Reuses the output Vec in `PreprocessEncodeScratch`.
/// Returns a `PreprocessedDataRef` borrowing the scratch output.
///
/// The `bytes_to_typed` intermediate (e.g. `Vec<i64>`) still allocates per block —
/// one ~800 KB allocation is acceptable given type-erasure complexity.
pub fn preprocess_into<'a>(
    data: &[u8],
    config: &PreprocessorConfig,
    scratch: &'a mut crate::workspace::PreprocessEncodeScratch,
) -> Result<crate::workspace::PreprocessedDataRef<'a>> {
    use crate::workspace::PreprocessedDataRef;

    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let data_type = config.data_type.unwrap_or_else(|| auto_detect(data));

    // For Raw, just copy into scratch output.
    // For typed, encode into scratch output (swap the encoded Vec in).
    match data_type {
        DataType::Raw => {
            scratch.output.clear();
            scratch.output.extend_from_slice(data);
            Ok(PreprocessedDataRef {
                data_type: DataType::Raw,
                element_count: data.len() as u64,
                data: &scratch.output,
            })
        }

        DataType::IntegerI64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_i64s(data);
            let element_count = values.len() as u64;
            let encoded = delta::encode_i64(&values, config.double_delta)?;
            scratch.output.clear();
            scratch.output = encoded; // take ownership of fresh Vec (replaces old capacity)
            Ok(PreprocessedDataRef {
                data_type: DataType::IntegerI64,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::IntegerU64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_u64s(data);
            let element_count = values.len() as u64;
            let encoded = delta::encode_u64(&values, config.double_delta)?;
            scratch.output.clear();
            scratch.output = encoded;
            Ok(PreprocessedDataRef {
                data_type: DataType::IntegerU64,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::IntegerI32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_i32s(data);
            let element_count = values.len() as u64;
            let encoded = delta::encode_i32(&values, config.double_delta)?;
            scratch.output.clear();
            scratch.output = encoded;
            Ok(PreprocessedDataRef {
                data_type: DataType::IntegerI32,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::IntegerU32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_u32s(data);
            let element_count = values.len() as u64;
            let encoded = delta::encode_u32(&values, config.double_delta)?;
            scratch.output.clear();
            scratch.output = encoded;
            Ok(PreprocessedDataRef {
                data_type: DataType::IntegerU32,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float64 => {
            validate_alignment(data, 8)?;
            let values = bytes_to_f64s(data);
            let element_count = values.len() as u64;
            let encoded = gorilla_xor::encode_f64(&values)?;
            scratch.output.clear();
            scratch.output = encoded;
            Ok(PreprocessedDataRef {
                data_type: DataType::Float64,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float32 => {
            validate_alignment(data, 4)?;
            let values = bytes_to_f32s(data);
            let element_count = values.len() as u64;
            let encoded = gorilla_xor::encode_f32(&values)?;
            scratch.output.clear();
            scratch.output = encoded;
            Ok(PreprocessedDataRef {
                data_type: DataType::Float32,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float64Shuffle => {
            validate_alignment(data, 8)?;
            let element_count = (data.len() / 8) as u64;
            bitshuffle::shuffle_into(data, 8, &mut scratch.output);
            Ok(PreprocessedDataRef {
                data_type: DataType::Float64Shuffle,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float32Shuffle => {
            validate_alignment(data, 4)?;
            let element_count = (data.len() / 4) as u64;
            bitshuffle::shuffle_into(data, 4, &mut scratch.output);
            Ok(PreprocessedDataRef {
                data_type: DataType::Float32Shuffle,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float64ShuffleDelta => {
            validate_alignment(data, 8)?;
            let element_count = (data.len() / 8) as u64;
            // Shuffle into scratch.output, then byte-delta in-place.
            bitshuffle::shuffle_into(data, 8, &mut scratch.output);
            let shuffled = std::mem::take(&mut scratch.output);
            byte_delta::encode_into(&shuffled, &mut scratch.output);
            scratch.shuffle_buf = shuffled; // return capacity for reuse
            Ok(PreprocessedDataRef {
                data_type: DataType::Float64ShuffleDelta,
                element_count,
                data: &scratch.output,
            })
        }

        DataType::Float32ShuffleDelta => {
            validate_alignment(data, 4)?;
            let element_count = (data.len() / 4) as u64;
            bitshuffle::shuffle_into(data, 4, &mut scratch.output);
            let shuffled = std::mem::take(&mut scratch.output);
            byte_delta::encode_into(&shuffled, &mut scratch.output);
            scratch.shuffle_buf = shuffled;
            Ok(PreprocessedDataRef {
                data_type: DataType::Float32ShuffleDelta,
                element_count,
                data: &scratch.output,
            })
        }
    }
}

/// Invert a preprocessing transform, recovering the original byte stream.
pub fn depreprocess(preprocessed: &PreprocessedData) -> Result<Vec<u8>> {
    match preprocessed.data_type {
        DataType::Raw => Ok(preprocessed.data.clone()),

        DataType::IntegerI64 => {
            let values =
                delta::decode_i64(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(i64s_to_bytes(&values))
        }

        DataType::IntegerU64 => {
            let values =
                delta::decode_u64(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(u64s_to_bytes(&values))
        }

        DataType::IntegerI32 => {
            let values =
                delta::decode_i32(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(i32s_to_bytes(&values))
        }

        DataType::IntegerU32 => {
            let values =
                delta::decode_u32(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(u32s_to_bytes(&values))
        }

        DataType::Float64 => {
            let values =
                gorilla_xor::decode_f64(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(f64s_to_bytes(&values))
        }

        DataType::Float32 => {
            let values =
                gorilla_xor::decode_f32(&preprocessed.data, preprocessed.element_count as usize)?;
            Ok(f32s_to_bytes(&values))
        }

        DataType::Float64Shuffle => Ok(bitshuffle::unshuffle(&preprocessed.data, 8)),

        DataType::Float32Shuffle => Ok(bitshuffle::unshuffle(&preprocessed.data, 4)),

        DataType::Float64ShuffleDelta => {
            let undelta = byte_delta::decode(&preprocessed.data);
            Ok(bitshuffle::unshuffle(&undelta, 8))
        }

        DataType::Float32ShuffleDelta => {
            let undelta = byte_delta::decode(&preprocessed.data);
            Ok(bitshuffle::unshuffle(&undelta, 4))
        }
    }
}

/// Workspace-aware variant of `depreprocess`. Uses `PreprocessedDataRef` to
/// borrow data instead of cloning, and writes directly into `scratch.output`
/// instead of allocating intermediate typed Vecs.
pub fn depreprocess_into(
    preprocessed: &crate::workspace::PreprocessedDataRef<'_>,
    scratch: &mut crate::workspace::PreprocessScratch,
) -> Result<()> {
    let count = preprocessed.element_count as usize;
    match preprocessed.data_type {
        DataType::Raw => {
            scratch.output.clear();
            scratch.output.extend_from_slice(preprocessed.data);
            Ok(())
        }
        DataType::IntegerI64 => {
            delta::decode_i64_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::IntegerU64 => {
            delta::decode_u64_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::IntegerI32 => {
            delta::decode_i32_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::IntegerU32 => {
            delta::decode_u32_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::Float64 => {
            gorilla_xor::decode_f64_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::Float32 => {
            gorilla_xor::decode_f32_into(preprocessed.data, count, &mut scratch.output)
        }
        DataType::Float64Shuffle => {
            bitshuffle::unshuffle_decode_into(preprocessed.data, count, 8, &mut scratch.output)
        }
        DataType::Float32Shuffle => {
            bitshuffle::unshuffle_decode_into(preprocessed.data, count, 4, &mut scratch.output)
        }

        DataType::Float64ShuffleDelta => {
            // Reverse: byte-delta decode → unshuffle.
            let undelta = byte_delta::decode(preprocessed.data);
            scratch.output.clear();
            scratch.output = bitshuffle::unshuffle(&undelta, 8);
            Ok(())
        }

        DataType::Float32ShuffleDelta => {
            let undelta = byte_delta::decode(preprocessed.data);
            scratch.output.clear();
            scratch.output = bitshuffle::unshuffle(&undelta, 4);
            Ok(())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Auto-Detection Heuristics
// ═══════════════════════════════════════════════════════════════════════════════

/// Attempt to detect the data type by analyzing byte patterns.
///
/// Heuristics applied (in order):
/// 1. If the length is divisible by 8, check for f64 patterns (NaN/Inf absence,
///    exponent distribution consistent with real sensor data).
/// 2. If divisible by 8, check for i64 patterns (monotonic, small deltas).
/// 3. If divisible by 4, repeat for f32 / i32.
/// 4. Fall back to Raw.
///
/// All heuristic functions operate directly on `chunks_exact()` iterators
/// over the raw byte slice — zero heap allocations.
pub(crate) fn auto_detect(data: &[u8]) -> DataType {
    // Need at least a few elements to make a meaningful decision.
    if data.len() < 32 {
        return DataType::Raw;
    }

    // Try 64-bit types first (more specific).
    if data.len() % 8 == 0 && data.len() >= 64 {
        if looks_like_f64_series(data) {
            return select_best_float_strategy(data, 8);
        }
        if looks_like_i64_series(data) {
            return DataType::IntegerI64;
        }
    }

    // Try 32-bit types.
    if data.len() % 4 == 0 && data.len() >= 32 {
        if looks_like_f32_series(data) {
            return select_best_float_strategy(data, 4);
        }
        if looks_like_i32_series(data) {
            return DataType::IntegerI32;
        }
    }

    DataType::Raw
}

/// Evaluate candidate float preprocessing strategies on a sample of the data
/// and return the one with the lowest estimated entropy (best compressibility).
///
/// Candidates for f64 (elem_size=8): Gorilla XOR, shuffle, shuffle+delta, raw.
/// Candidates for f32 (elem_size=4): Gorilla XOR, shuffle, shuffle+delta, raw.
///
/// Uses order-0 Shannon entropy on the first 8 KB of preprocessed output
/// to estimate compressibility without running the full LZ77+FSE pipeline.
fn select_best_float_strategy(data: &[u8], elem_size: usize) -> DataType {
    // Sample size: use at most 8 KB for entropy estimation (fast).
    let sample_len = data.len().min(8192);
    // Round down to element boundary.
    let sample_len = (sample_len / elem_size) * elem_size;
    if sample_len < elem_size * 4 {
        // Too small to meaningfully compare — fall back to default.
        return if elem_size == 8 {
            DataType::Float64
        } else {
            DataType::Float32
        };
    }
    let sample = &data[..sample_len];

    // Candidate 1: Raw (baseline — entropy of the original data).
    let entropy_raw = entropy_probe::entropy_bits_per_byte(sample);

    // Candidate 2: Gorilla XOR.
    let entropy_gorilla = if elem_size == 8 {
        let values = bytes_to_f64s(sample);
        match gorilla_xor::encode_f64(&values) {
            Ok(encoded) => entropy_probe::entropy_bits_per_byte(&encoded),
            Err(_) => f64::MAX,
        }
    } else {
        let values = bytes_to_f32s(sample);
        match gorilla_xor::encode_f32(&values) {
            Ok(encoded) => entropy_probe::entropy_bits_per_byte(&encoded),
            Err(_) => f64::MAX,
        }
    };

    // Candidate 3: Byte shuffle only.
    let shuffled = bitshuffle::shuffle(sample, elem_size);
    let entropy_shuffle = entropy_probe::entropy_bits_per_byte(&shuffled);

    // Candidate 4: Byte shuffle + byte-delta.
    let shuffle_delta = byte_delta::encode(&shuffled);
    let entropy_shuffle_delta = entropy_probe::entropy_bits_per_byte(&shuffle_delta);

    // Pick the strategy with the lowest entropy.
    let mut best_entropy = entropy_raw;
    let mut best = DataType::Raw;

    if entropy_gorilla < best_entropy {
        best_entropy = entropy_gorilla;
        best = if elem_size == 8 {
            DataType::Float64
        } else {
            DataType::Float32
        };
    }
    if entropy_shuffle < best_entropy {
        best_entropy = entropy_shuffle;
        best = if elem_size == 8 {
            DataType::Float64Shuffle
        } else {
            DataType::Float32Shuffle
        };
    }
    if entropy_shuffle_delta < best_entropy {
        best_entropy = entropy_shuffle_delta;
        best = if elem_size == 8 {
            DataType::Float64ShuffleDelta
        } else {
            DataType::Float32ShuffleDelta
        };
    }

    // If best entropy is within 5% of raw, don't bother preprocessing —
    // the ratio gain won't justify the CPU cost plus frame overhead.
    if best_entropy > entropy_raw * 0.95 && !matches!(best, DataType::Raw) {
        // None of the strategies meaningfully helped.
        return DataType::Raw;
    }

    let _ = best_entropy; // suppress unused warning
    best
}

/// Check if raw bytes, interpreted as little-endian f64, look like sensor data.
/// Criteria: no NaN/Inf, bounded range, mostly smooth adjacent differences.
/// Operates on chunks_exact(8) — zero allocations.
fn looks_like_f64_series(data: &[u8]) -> bool {
    let n_elements = data.len() / 8;
    if n_elements < 4 {
        return false;
    }

    // First pass: find min/max and reject NaN/Inf.
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    for chunk in data.chunks_exact(8) {
        let v = f64::from_le_bytes(chunk.try_into().unwrap());
        if v.is_nan() || v.is_infinite() {
            return false;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let range = max - min;
    if range == 0.0 {
        return true; // constant series
    }

    // Second pass: check smoothness via adjacent pairs.
    let threshold = range * 0.25;
    let mut smooth_count = 0usize;
    let mut prev = f64::from_le_bytes(data[..8].try_into().unwrap());
    for chunk in data[8..].chunks_exact(8) {
        let cur = f64::from_le_bytes(chunk.try_into().unwrap());
        if (cur - prev).abs() <= threshold {
            smooth_count += 1;
        }
        prev = cur;
    }
    let smoothness = smooth_count as f64 / (n_elements - 1) as f64;
    smoothness > 0.7
}

/// Check if raw bytes, interpreted as little-endian i64, look like timestamps/counters.
/// Criteria: mostly monotonic, small deltas. Zero allocations.
fn looks_like_i64_series(data: &[u8]) -> bool {
    let n_elements = data.len() / 8;
    if n_elements < 4 {
        return false;
    }
    let mut monotonic_count = 0usize;
    let mut small_delta_count = 0usize;
    let mut prev = i64::from_le_bytes(data[..8].try_into().unwrap());
    for chunk in data[8..].chunks_exact(8) {
        let cur = i64::from_le_bytes(chunk.try_into().unwrap());
        let delta = cur.wrapping_sub(prev);
        if delta >= 0 {
            monotonic_count += 1;
        }
        if delta.unsigned_abs() < 1_000_000 {
            small_delta_count += 1;
        }
        prev = cur;
    }
    let n = (n_elements - 1) as f64;
    let monotonic_ratio = monotonic_count as f64 / n;
    let small_delta_ratio = small_delta_count as f64 / n;
    monotonic_ratio > 0.8 || small_delta_ratio > 0.9
}

/// Check if raw bytes, interpreted as little-endian f32, look like sensor data.
/// Zero allocations.
fn looks_like_f32_series(data: &[u8]) -> bool {
    let n_elements = data.len() / 4;
    if n_elements < 4 {
        return false;
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for chunk in data.chunks_exact(4) {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        if v.is_nan() || v.is_infinite() {
            return false;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let range = max - min;
    if range == 0.0 {
        return true;
    }
    let threshold = range * 0.25;
    let mut smooth_count = 0usize;
    let mut prev = f32::from_le_bytes(data[..4].try_into().unwrap());
    for chunk in data[4..].chunks_exact(4) {
        let cur = f32::from_le_bytes(chunk.try_into().unwrap());
        if (cur - prev).abs() <= threshold {
            smooth_count += 1;
        }
        prev = cur;
    }
    let smoothness = smooth_count as f64 / (n_elements - 1) as f64;
    smoothness > 0.7
}

/// Check if raw bytes, interpreted as little-endian i32, look like counters.
/// Zero allocations.
fn looks_like_i32_series(data: &[u8]) -> bool {
    let n_elements = data.len() / 4;
    if n_elements < 4 {
        return false;
    }
    let mut monotonic_count = 0usize;
    let mut small_delta_count = 0usize;
    let mut prev = i32::from_le_bytes(data[..4].try_into().unwrap());
    for chunk in data[4..].chunks_exact(4) {
        let cur = i32::from_le_bytes(chunk.try_into().unwrap());
        let delta = cur as i64 - prev as i64;
        if delta >= 0 {
            monotonic_count += 1;
        }
        if delta.unsigned_abs() < 100_000 {
            small_delta_count += 1;
        }
        prev = cur;
    }
    let n = (n_elements - 1) as f64;
    let monotonic_ratio = monotonic_count as f64 / n;
    let small_delta_ratio = small_delta_count as f64 / n;
    monotonic_ratio > 0.8 || small_delta_ratio > 0.9
}

// ═══════════════════════════════════════════════════════════════════════════════
// Byte ↔ Typed Value Conversion Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn validate_alignment(data: &[u8], element_size: usize) -> Result<()> {
    if data.len() % element_size != 0 {
        return Err(CompressorError::DataTypeMismatch {
            element_size,
            buffer_len: data.len(),
        });
    }
    Ok(())
}

fn bytes_to_i64s(data: &[u8]) -> Vec<i64> {
    data.chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_u64s(data: &[u8]) -> Vec<u64> {
    data.chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_i32s(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_u32s(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_f64s(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_f32s(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn i64s_to_bytes(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn u64s_to_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn i32s_to_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f64s_to_bytes(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_conversion_roundtrip_i64() {
        let values: Vec<i64> = vec![-1000, 0, 1, i64::MAX, i64::MIN, 42];
        let bytes = i64s_to_bytes(&values);
        let recovered = bytes_to_i64s(&bytes);
        assert_eq!(values, recovered);
    }

    #[test]
    fn byte_conversion_roundtrip_u64() {
        let values: Vec<u64> = vec![0, 1, u64::MAX, 999999999999];
        let bytes = u64s_to_bytes(&values);
        let recovered = bytes_to_u64s(&bytes);
        assert_eq!(values, recovered);
    }

    #[test]
    fn byte_conversion_roundtrip_f64() {
        let values: Vec<f64> = vec![0.0, -1.5, 3.14159265, f64::MAX, f64::MIN_POSITIVE];
        let bytes = f64s_to_bytes(&values);
        let recovered = bytes_to_f64s(&bytes);
        assert_eq!(values, recovered);
    }

    #[test]
    fn byte_conversion_roundtrip_i32() {
        let values: Vec<i32> = vec![-1, 0, i32::MAX, i32::MIN];
        let bytes = i32s_to_bytes(&values);
        let recovered = bytes_to_i32s(&bytes);
        assert_eq!(values, recovered);
    }

    #[test]
    fn byte_conversion_roundtrip_f32() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, f32::MAX];
        let bytes = f32s_to_bytes(&values);
        let recovered = bytes_to_f32s(&bytes);
        assert_eq!(values, recovered);
    }

    #[test]
    fn alignment_validation() {
        assert!(validate_alignment(&[0u8; 16], 8).is_ok());
        assert!(validate_alignment(&[0u8; 15], 8).is_err());
        assert!(validate_alignment(&[0u8; 12], 4).is_ok());
        assert!(validate_alignment(&[0u8; 13], 4).is_err());
    }

    #[test]
    fn auto_detect_monotonic_i64() {
        // Monotonically increasing timestamps (nanoseconds)
        let timestamps: Vec<i64> = (0..100)
            .map(|i| 1_700_000_000_000 + i * 1_000_000)
            .collect();
        let bytes = i64s_to_bytes(&timestamps);
        let detected = auto_detect(&bytes);
        // Should detect as some form of integer (might be i64 or f64 depending on heuristics,
        // but should NOT be Raw)
        assert!(
            matches!(
                detected,
                DataType::IntegerI64
                    | DataType::Float64
                    | DataType::Float64Shuffle
                    | DataType::Float64ShuffleDelta
            ),
            "Expected integer or float detection for monotonic series, got {:?}",
            detected
        );
    }

    #[test]
    fn auto_detect_smooth_f64() {
        // Slowly varying temperature readings
        let mut temps: Vec<f64> = Vec::new();
        let mut t = 22.5;
        for _ in 0..100 {
            temps.push(t);
            t += 0.01 * ((t * 0.1).sin()); // tiny variation
        }
        let bytes = f64s_to_bytes(&temps);
        let detected = auto_detect(&bytes);
        // Adaptive selection may pick Gorilla, shuffle, or shuffle+delta
        // depending on entropy. All are valid float strategies.
        assert!(
            matches!(
                detected,
                DataType::Float64 | DataType::Float64Shuffle | DataType::Float64ShuffleDelta
            ),
            "Expected a float strategy for smooth f64, got {:?}",
            detected
        );
    }

    #[test]
    fn auto_detect_random_fallback() {
        // Non-structured bytes (text, binary blobs) typically aren't aligned to
        // 4- or 8-byte boundaries. Use a 253-byte buffer (prime length) which
        // can't be interpreted as any typed series.
        let data: Vec<u8> = (0u16..253)
            .map(|i| (i.wrapping_mul(151) ^ 0xAB) as u8)
            .collect();
        assert_eq!(data.len(), 253); // not divisible by 4 or 8
        let detected = auto_detect(&data);
        assert_eq!(
            detected,
            DataType::Raw,
            "Non-aligned data should auto-detect as Raw"
        );
    }

    #[test]
    fn preprocess_raw_passthrough() {
        let data = vec![1, 2, 3, 4, 5];
        let config = PreprocessorConfig {
            data_type: Some(DataType::Raw),
            ..Default::default()
        };
        let result = preprocess(&data, &config).unwrap();
        assert_eq!(result.data_type, DataType::Raw);
        assert_eq!(result.data, data);
        assert_eq!(result.element_count, 5);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_i64() {
        let values: Vec<i64> = (0..500).map(|i| 1000 + i * 100 + i * i).collect();
        let bytes = i64s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        assert_eq!(preprocessed.data_type, DataType::IntegerI64);
        assert_eq!(preprocessed.element_count, 500);

        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_f64() {
        let values: Vec<f64> = (0..200).map(|i| 22.5 + 0.01 * (i as f64).sin()).collect();
        let bytes = f64s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float64),
            ..Default::default()
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_u32() {
        let values: Vec<u32> = (0..300).map(|i| i * 1000).collect();
        let bytes = u32s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerU32),
            double_delta: true,
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_f32() {
        let values: Vec<f32> = (0..200).map(|i| 100.0 + 0.5 * (i as f32)).collect();
        let bytes = f32s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float32),
            ..Default::default()
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_f64_shuffle() {
        let values: Vec<f64> = (0..200).map(|i| 22.5 + 0.01 * (i as f64).sin()).collect();
        let bytes = f64s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float64Shuffle),
            ..Default::default()
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        assert_eq!(preprocessed.data_type, DataType::Float64Shuffle);
        assert_eq!(preprocessed.element_count, 200);
        // Shuffled data is same size as input
        assert_eq!(preprocessed.data.len(), bytes.len());
        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_depreprocess_roundtrip_f32_shuffle() {
        let values: Vec<f32> = (0..300)
            .map(|i| {
                let t = i as f32 * 0.001;
                (t * 50.0).sin() * 2.0 + (t * 120.0).cos() * 0.5
            })
            .collect();
        let bytes = f32s_to_bytes(&values);
        let config = PreprocessorConfig {
            data_type: Some(DataType::Float32Shuffle),
            ..Default::default()
        };
        let preprocessed = preprocess(&bytes, &config).unwrap();
        assert_eq!(preprocessed.data_type, DataType::Float32Shuffle);
        let recovered = depreprocess(&preprocessed).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn preprocess_empty_input_fails() {
        let config = PreprocessorConfig::default();
        assert!(matches!(
            preprocess(&[], &config),
            Err(CompressorError::EmptyInput)
        ));
    }

    #[test]
    fn preprocess_misaligned_fails() {
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            ..Default::default()
        };
        // 7 bytes is not divisible by 8
        assert!(matches!(
            preprocess(&[0u8; 7], &config),
            Err(CompressorError::DataTypeMismatch { .. })
        ));
    }

    #[test]
    fn preprocessor_compression_ratio_benefit() {
        // Demonstrate that preprocessing actually produces smaller output
        // for structured data (compared to raw passthrough).
        let timestamps: Vec<i64> = (0..1000).map(|i| 1_700_000_000 + i * 1000).collect();
        let raw_bytes = i64s_to_bytes(&timestamps);
        let config = PreprocessorConfig {
            data_type: Some(DataType::IntegerI64),
            double_delta: true,
        };
        let preprocessed = preprocess(&raw_bytes, &config).unwrap();

        // The delta-of-delta encoded output should be significantly smaller
        // because the double deltas are all 0 (constant step of 1000).
        assert!(
            preprocessed.data.len() < raw_bytes.len() / 2,
            "Expected at least 2x compression from delta-of-delta on linear sequence. \
             Raw: {} bytes, preprocessed: {} bytes",
            raw_bytes.len(),
            preprocessed.data.len()
        );
    }
}
