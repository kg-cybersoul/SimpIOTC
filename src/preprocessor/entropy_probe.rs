// ═══════════════════════════════════════════════════════════════════════════════
// Fast Entropy Estimator
// ═══════════════════════════════════════════════════════════════════════════════
//
// Order-0 Shannon entropy estimate on a byte slice. Single pass, no allocation
// beyond a 256-entry stack array. Used by the adaptive preprocessor to evaluate
// candidate strategies without running the full LZ77+FSE pipeline.
//
// Returns bits-per-byte in [0.0, 8.0]. Lower is better (more compressible).
// A value near 8.0 means the data looks random to an order-0 model.
//
// Accuracy: order-0 entropy is a LOWER BOUND on the true compressibility.
// LZ77 can exploit higher-order patterns (matches, repeats) that order-0
// cannot see. But for COMPARING strategies on the same data, the ranking
// is reliable — if strategy A has lower order-0 entropy than strategy B,
// it will almost always compress smaller.

/// Estimate order-0 Shannon entropy of `data` in bits per byte.
///
/// Returns 0.0 for empty input. On non-empty input, returns a value in [0.0, 8.0].
pub fn entropy_bits_per_byte(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut freq = [0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }

    let n = data.len() as f64;
    let mut entropy = 0.0f64;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Estimate the compression ratio achievable by an order-0 entropy coder.
///
/// Returns `original_size / estimated_compressed_size`. Higher is better.
/// A value of 1.0 means no compression expected. Values below 1.0 are not
/// possible (entropy coding can't expand beyond 8 bits/byte), but frame
/// overhead can cause real-world expansion.
pub fn estimated_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 1.0;
    }
    let bpb = entropy_bits_per_byte(data);
    if bpb < 0.001 {
        // Near-zero entropy — data is essentially constant.
        // Avoid division by zero; return a large but finite ratio.
        return data.len() as f64;
    }
    8.0 / bpb
}

/// Estimate entropy on a prefix of the data (for speed on large inputs).
/// Uses at most `max_sample` bytes. If data is shorter, uses all of it.
pub fn entropy_bits_per_byte_sampled(data: &[u8], max_sample: usize) -> f64 {
    let sample = if data.len() > max_sample {
        &data[..max_sample]
    } else {
        data
    };
    entropy_bits_per_byte(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert_eq!(entropy_bits_per_byte(&[]), 0.0);
        assert_eq!(estimated_ratio(&[]), 1.0);
    }

    #[test]
    fn constant_data() {
        let data = vec![42u8; 10_000];
        let e = entropy_bits_per_byte(&data);
        assert!(e < 0.001, "constant data should have ~0 entropy, got {e}");
        let r = estimated_ratio(&data);
        assert!(r > 1000.0, "constant data should have huge ratio, got {r}");
    }

    #[test]
    fn two_symbols_equal() {
        // Equal mix of two symbols → 1.0 bit/byte.
        let mut data = vec![0u8; 10_000];
        for i in 0..5000 {
            data[i] = 1;
        }
        let e = entropy_bits_per_byte(&data);
        assert!((e - 1.0).abs() < 0.01, "expected ~1.0 bpb, got {e}");
    }

    #[test]
    fn uniform_random() {
        // All 256 symbols equally likely → ~8.0 bits/byte.
        let data: Vec<u8> = (0..256 * 100).map(|i| (i % 256) as u8).collect();
        let e = entropy_bits_per_byte(&data);
        assert!((e - 8.0).abs() < 0.01, "expected ~8.0 bpb, got {e}");
    }

    #[test]
    fn sampled_matches_full_on_uniform() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let full = entropy_bits_per_byte(&data);
        let sampled = entropy_bits_per_byte_sampled(&data, 8192);
        assert!(
            (full - sampled).abs() < 0.1,
            "sampled ({sampled}) should be close to full ({full})"
        );
    }

    #[test]
    fn ratio_sanity() {
        // Data with ~4 bpb → estimated ratio ~2x.
        // 16 equally frequent symbols = log2(16) = 4 bpb.
        let data: Vec<u8> = (0..16 * 1000).map(|i| (i % 16) as u8).collect();
        let r = estimated_ratio(&data);
        assert!(
            (r - 2.0).abs() < 0.1,
            "expected ~2.0x ratio for 16-symbol data, got {r}"
        );
    }

    #[test]
    fn delta_encoded_constant_is_low_entropy() {
        // Simulate byte-delta of constant lane: first byte + all zeros.
        let mut data = vec![0u8; 10_000];
        data[0] = 42;
        let e = entropy_bits_per_byte(&data);
        assert!(
            e < 0.05,
            "delta of constant should be near-zero entropy, got {e}"
        );
    }
}
