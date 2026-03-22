//! # Byte-Level Shuffle (Bit-Shuffle) Preprocessor
//!
//! Transposes N elements of S bytes from row-major (natural) layout to
//! column-major (byte-grouped) layout. This dramatically improves LZ77
//! compression on noisy float data by grouping structurally similar bytes:
//!
//! - **Byte 0** (exponent + sign for floats): near-identical across samples → long LZ77 matches
//! - **Bytes 1–2** (high mantissa): moderate entropy, gradual changes → good delta/LZ77
//! - **Bytes 5–7** (low mantissa): high entropy but grouped → better than interleaved
//!
//! ## Layout
//!
//! Input (N elements, S bytes each):
//! ```text
//! [e0_b0, e0_b1, ..., e0_bS-1, e1_b0, e1_b1, ..., e1_bS-1, ...]
//! ```
//!
//! Shuffled output:
//! ```text
//! [e0_b0, e1_b0, ..., eN-1_b0, e0_b1, e1_b1, ..., eN-1_b1, ..., e0_bS-1, ..., eN-1_bS-1]
//! ```
//!
//! The transform is its own inverse only for S=1. For S>1, `unshuffle` is the
//! distinct inverse operation.
//!
//! ## Complexity
//!
//! Both shuffle and unshuffle are O(N×S) with a single pass over the data.
//! Output is exactly the same size as input — no headers, no overhead.

use crate::Result;

/// Shuffle N elements of `element_size` bytes from row-major to column-major byte layout.
///
/// # Panics
/// Panics if `data.len()` is not divisible by `element_size` or `element_size` is 0.
pub fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    assert!(element_size > 0, "element_size must be > 0");
    assert!(
        data.len() % element_size == 0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let n = data.len() / element_size;
    let mut output = vec![0u8; data.len()];

    for byte_idx in 0..element_size {
        for elem_idx in 0..n {
            output[byte_idx * n + elem_idx] = data[elem_idx * element_size + byte_idx];
        }
    }

    output
}

/// Unshuffle (inverse of `shuffle`): column-major byte layout back to row-major.
///
/// # Panics
/// Panics if `data.len()` is not divisible by `element_size` or `element_size` is 0.
pub fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    assert!(element_size > 0, "element_size must be > 0");
    assert!(
        data.len() % element_size == 0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let n = data.len() / element_size;
    let mut output = vec![0u8; data.len()];

    for byte_idx in 0..element_size {
        for elem_idx in 0..n {
            output[elem_idx * element_size + byte_idx] = data[byte_idx * n + elem_idx];
        }
    }

    output
}

/// Workspace-aware shuffle. Clears and reuses the output Vec.
pub fn shuffle_into(data: &[u8], element_size: usize, output: &mut Vec<u8>) {
    assert!(element_size > 0, "element_size must be > 0");
    assert!(
        data.len() % element_size == 0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let n = data.len() / element_size;
    output.clear();
    output.resize(data.len(), 0);

    for byte_idx in 0..element_size {
        for elem_idx in 0..n {
            output[byte_idx * n + elem_idx] = data[elem_idx * element_size + byte_idx];
        }
    }
}

/// Workspace-aware unshuffle. Clears and reuses the output Vec.
pub fn unshuffle_into(data: &[u8], element_size: usize, output: &mut Vec<u8>) {
    assert!(element_size > 0, "element_size must be > 0");
    assert!(
        data.len() % element_size == 0,
        "data length {} not divisible by element_size {}",
        data.len(),
        element_size
    );

    let n = data.len() / element_size;
    output.clear();
    output.resize(data.len(), 0);

    for byte_idx in 0..element_size {
        for elem_idx in 0..n {
            output[elem_idx * element_size + byte_idx] = data[byte_idx * n + elem_idx];
        }
    }
}

/// Workspace-aware unshuffle that writes LE bytes directly into output.
/// Used by the decode pipeline to match the `_into` pattern of other preprocessors.
pub fn unshuffle_decode_into(
    data: &[u8],
    element_count: usize,
    element_size: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    let expected_len = element_count * element_size;
    if data.len() < expected_len {
        return Err(crate::CompressorError::Preprocessor(format!(
            "bitshuffle: expected {} bytes ({} elements × {} bytes), got {}",
            expected_len,
            element_count,
            element_size,
            data.len()
        )));
    }

    output.clear();
    output.resize(expected_len, 0);

    let n = element_count;
    for byte_idx in 0..element_size {
        for elem_idx in 0..n {
            output[elem_idx * element_size + byte_idx] = data[byte_idx * n + elem_idx];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Roundtrip Tests ──────────────────────────────────────────────────

    #[test]
    fn shuffle_unshuffle_roundtrip_f64() {
        // 10 f64 values = 80 bytes, element_size = 8
        let values: Vec<f64> = (0..10).map(|i| 22.5 + 0.01 * i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 8);
        assert_eq!(shuffled.len(), data.len());

        let recovered = unshuffle(&shuffled, 8);
        assert_eq!(recovered, data);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_f32() {
        let values: Vec<f32> = (0..20).map(|i| 100.0 + 0.5 * i as f32).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 4);
        let recovered = unshuffle(&shuffled, 4);
        assert_eq!(recovered, data);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_single_element() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        let shuffled = shuffle(&data, 8);
        // Single element: shuffle is identity
        assert_eq!(shuffled, data);
        let recovered = unshuffle(&shuffled, 8);
        assert_eq!(recovered, data);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip_two_elements() {
        // Two f32 values: [A0, A1, A2, A3, B0, B1, B2, B3]
        // Shuffled:       [A0, B0, A1, B1, A2, B2, A3, B3]
        let data = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        let shuffled = shuffle(&data, 4);
        assert_eq!(
            shuffled,
            vec![0x10, 0x50, 0x20, 0x60, 0x30, 0x70, 0x40, 0x80]
        );
        let recovered = unshuffle(&shuffled, 4);
        assert_eq!(recovered, data);
    }

    // ── Layout Verification ──────────────────────────────────────────────

    #[test]
    fn shuffle_layout_f64_groups_exponent_bytes() {
        // 3 f64 values with same exponent but different mantissas
        let v1: f64 = 22.51;
        let v2: f64 = 22.53;
        let v3: f64 = 22.49;
        let data: Vec<u8> = [v1, v2, v3].iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 8);

        // After shuffle: first 3 bytes are byte-0 of each element,
        // next 3 are byte-1, etc.
        // Byte 7 (MSB) of f64 contains sign + high exponent bits.
        // For values near 22.5, byte 7 should be identical (0x40).
        let n = 3;
        let byte7_column = &shuffled[7 * n..8 * n];
        assert_eq!(byte7_column[0], byte7_column[1]);
        assert_eq!(byte7_column[1], byte7_column[2]);
    }

    #[test]
    fn shuffle_layout_f32_groups_exponent_bytes() {
        let v1: f32 = 100.01;
        let v2: f32 = 100.03;
        let v3: f32 = 100.05;
        let v4: f32 = 99.99;
        let data: Vec<u8> = [v1, v2, v3, v4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let shuffled = shuffle(&data, 4);
        let n = 4;

        // Byte 3 (MSB for f32) should be identical for all values near 100.0
        let byte3_column = &shuffled[3 * n..4 * n];
        assert_eq!(byte3_column[0], byte3_column[1]);
        assert_eq!(byte3_column[1], byte3_column[2]);
        assert_eq!(byte3_column[2], byte3_column[3]);
    }

    // ── Workspace-Aware Variants ─────────────────────────────────────────

    #[test]
    fn shuffle_into_matches_shuffle() {
        let values: Vec<f64> = (0..50).map(|i| 3.14 * i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let expected = shuffle(&data, 8);

        let mut output = Vec::new();
        shuffle_into(&data, 8, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn unshuffle_into_matches_unshuffle() {
        let values: Vec<f64> = (0..50).map(|i| 3.14 * i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shuffled = shuffle(&data, 8);

        let expected = unshuffle(&shuffled, 8);

        let mut output = Vec::new();
        unshuffle_into(&shuffled, 8, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn workspace_variants_reuse_capacity() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut output = Vec::with_capacity(1024);
        shuffle_into(&data, 8, &mut output);
        let cap_after_first = output.capacity();

        // Second call with smaller data should reuse capacity
        let small_data: Vec<u8> = (0..10u64).flat_map(|v| v.to_le_bytes()).collect();
        shuffle_into(&small_data, 8, &mut output);
        assert!(output.capacity() >= cap_after_first - 1); // may shrink slightly due to resize
    }

    #[test]
    fn unshuffle_decode_into_roundtrip() {
        let values: Vec<f32> = (0..30).map(|i| -50.0 + 0.1 * i as f32).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 4);

        let mut output = Vec::new();
        unshuffle_decode_into(&shuffled, 30, 4, &mut output).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn unshuffle_decode_into_rejects_short_data() {
        let result = unshuffle_decode_into(&[0u8; 10], 4, 4, &mut Vec::new());
        assert!(result.is_err());
    }

    // ── Edge Cases ───────────────────────────────────────────────────────

    #[test]
    fn shuffle_empty_data() {
        let data: Vec<u8> = vec![];
        let shuffled = shuffle(&data, 4);
        assert!(shuffled.is_empty());
        let recovered = unshuffle(&shuffled, 4);
        assert!(recovered.is_empty());
    }

    #[test]
    fn shuffle_element_size_1_is_identity() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shuffled = shuffle(&data, 1);
        assert_eq!(shuffled, data);
        let recovered = unshuffle(&shuffled, 1);
        assert_eq!(recovered, data);
    }

    #[test]
    fn shuffle_large_dataset() {
        // 10000 f64 values = 80KB — exercises the hot loop
        let values: Vec<f64> = (0..10000)
            .map(|i| 22.5 + 0.001 * (i as f64).sin())
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 8);
        let recovered = unshuffle(&shuffled, 8);
        assert_eq!(recovered, data);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn shuffle_misaligned_panics() {
        shuffle(&[0u8; 7], 4);
    }

    #[test]
    #[should_panic(expected = "must be > 0")]
    fn shuffle_zero_element_size_panics() {
        shuffle(&[0u8; 8], 0);
    }

    // ── Compression Quality Verification ─────────────────────────────────

    #[test]
    fn shuffled_floats_have_lower_byte_entropy_than_raw() {
        // Noisy temperature data: 22.5°C ± 0.01 noise
        let values: Vec<f64> = (0..1000)
            .map(|i| 22.5 + 0.01 * ((i as f64 * 0.1).sin()))
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 8);
        let n = 1000;

        // Count unique bytes in the exponent column (byte 7 of f64 LE)
        // vs a full stride of original data
        let byte7_unique: std::collections::HashSet<u8> =
            shuffled[7 * n..8 * n].iter().copied().collect();
        let raw_unique: std::collections::HashSet<u8> = data[..n].iter().copied().collect();

        // Exponent column should have very few unique values (1-2 for values near 22.5)
        assert!(
            byte7_unique.len() <= 3,
            "exponent byte column should have ≤3 unique values, got {}",
            byte7_unique.len()
        );
        // Raw first-N-bytes spans across all byte positions — much higher entropy
        assert!(
            raw_unique.len() > byte7_unique.len(),
            "raw bytes should have higher entropy than exponent column"
        );
    }

    #[test]
    fn shuffled_f32_vibration_exponent_column_low_entropy() {
        // Simulate noisy vibration data (the worst case for Gorilla).
        // Shuffle groups exponent bytes together — that column should have
        // very few unique values even when the data is noisy.
        let values: Vec<f32> = (0..500)
            .map(|i| {
                let t = i as f32 * 0.001;
                (t * 50.0).sin() * 2.0 + (t * 120.0).cos() * 0.5
            })
            .collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&data, 4);
        let n = 500;

        // Byte 3 (MSB of f32 LE) contains sign bit + top 7 exponent bits.
        // For values in the range ~[-2.5, 2.5], the exponent is tightly bounded,
        // so only a handful of unique byte-3 values exist.
        let byte3_col = &shuffled[3 * n..4 * n];
        let byte3_unique: std::collections::HashSet<u8> = byte3_col.iter().copied().collect();

        // Values oscillate between roughly -2.5 and +2.5. Sign bit × several
        // exponent levels gives ~10-15 unique byte-3 values — dramatically fewer
        // than the 256 possible byte values, which is the whole point of shuffle.
        assert!(
            byte3_unique.len() <= 16,
            "exponent column should have ≤16 unique values for bounded vibration data, got {}",
            byte3_unique.len()
        );
    }
}
