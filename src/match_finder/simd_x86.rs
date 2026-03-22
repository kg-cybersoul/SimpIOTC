//! # SIMD-Accelerated Match Extension
//!
//! Once the hash chain identifies a candidate position, we need to determine
//! how many bytes actually match. This module does that comparison as fast as
//! possible.
//!
//! ## Strategy
//!
//! Match extension is the innermost hot loop of LZ77 — it runs for every
//! candidate at every input position. We provide two implementations:
//!
//! 1. **Portable (always available)**: Compares 8 bytes at a time using u64
//!    XOR + count-trailing-zeros to find the first differing byte. This is
//!    already much faster than byte-by-byte comparison.
//!
//! 2. **AVX2 (x86_64 with runtime feature detection)**: Compares 32 bytes
//!    at a time using `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`. A single
//!    clock cycle finds mismatches across an entire cache line.
//!
//! The public `match_length()` function dispatches to the fastest available
//! implementation at runtime.

/// Returns a function pointer to the fastest available match-length
/// implementation. On x86_64 with AVX2, returns the SIMD path;
/// otherwise returns the portable u64 path.
///
/// Call this once (e.g. at MatchFinder construction) and store the result
/// to avoid per-call feature-detection overhead.
pub fn get_match_length_fn() -> fn(&[u8], usize, usize, usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return match_length_avx2_wrapper;
        }
    }
    match_length_portable_wrapper
}

fn match_length_portable_wrapper(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let len1 = data.len() - pos1;
    let len2 = data.len() - pos2;
    let limit = max_len.min(len1).min(len2);
    if limit == 0 {
        return 0;
    }
    match_length_portable(&data[pos1..pos1 + limit], &data[pos2..pos2 + limit])
}

#[cfg(target_arch = "x86_64")]
fn match_length_avx2_wrapper(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let len1 = data.len() - pos1;
    let len2 = data.len() - pos2;
    let limit = max_len.min(len1).min(len2);
    if limit == 0 {
        return 0;
    }
    unsafe { match_length_avx2(&data[pos1..pos1 + limit], &data[pos2..pos2 + limit]) }
}

/// Compute the length of the matching prefix between `data[pos1..]` and
/// `data[pos2..]`, up to `max_len` bytes.
///
/// Convenience entry point that dispatches on every call. For hot loops,
/// prefer `get_match_length_fn()` to resolve once and call the pointer.
#[inline]
pub fn match_length(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    get_match_length_fn()(data, pos1, pos2, max_len)
}

/// Portable match length using u64 XOR comparison.
///
/// Compares 8 bytes at a time. On the first XOR that produces a non-zero
/// result, uses trailing-zero count to find the exact mismatch position.
/// This is ~8x faster than byte-by-byte on 64-bit platforms.
#[inline]
fn match_length_portable(s1: &[u8], s2: &[u8]) -> usize {
    let len = s1.len().min(s2.len());
    let mut i = 0;

    // Compare 8 bytes at a time.
    while i + 8 <= len {
        let a = u64::from_le_bytes(s1[i..i + 8].try_into().unwrap());
        let b = u64::from_le_bytes(s2[i..i + 8].try_into().unwrap());
        let xor = a ^ b;
        if xor != 0 {
            // First differing bit → first differing byte.
            return i + (xor.trailing_zeros() as usize / 8);
        }
        i += 8;
    }

    // Compare remaining bytes one at a time.
    while i < len {
        if s1[i] != s2[i] {
            return i;
        }
        i += 1;
    }

    len
}

/// AVX2-accelerated match length. Compares 32 bytes per iteration.
///
/// Uses `_mm256_cmpeq_epi8` to compare 32 bytes, then `_mm256_movemask_epi8`
/// to extract a 32-bit bitmask of equality results. A single `ctz` on the
/// inverted mask finds the first mismatch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn match_length_avx2(s1: &[u8], s2: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = s1.len().min(s2.len());
    let mut i = 0;

    // Compare 32 bytes at a time using AVX2.
    while i + 32 <= len {
        let a = _mm256_loadu_si256(s1[i..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(s2[i..].as_ptr() as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(a, b);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0xFFFFFFFF {
            // At least one byte differs. Find the first differing byte.
            let diff_mask = !mask;
            return i + diff_mask.trailing_zeros() as usize;
        }
        i += 32;
    }

    // Handle remaining bytes with the portable path.
    if i < len {
        i += match_length_portable(&s1[i..], &s2[i..]);
    }

    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_length_identical() {
        let data = vec![0xAA; 1024];
        let len = match_length(&data, 0, 512, 512);
        assert_eq!(len, 512);
    }

    #[test]
    fn match_length_no_match() {
        let data = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC];
        let len = match_length(&data, 0, 4, 4);
        assert_eq!(len, 0); // first bytes differ
    }

    #[test]
    fn match_length_partial() {
        let mut data = vec![0xAA; 100];
        data[50] = 0xBB; // mismatch at position 50 from start of s2
                         // s1 starts at 0, s2 starts at 10. Compare up to 50 bytes.
                         // Bytes 0..9 of s1 = 0xAA, bytes 10..19 of s2 = 0xAA → match.
                         // At absolute position 50, s2[50-10=40] = 0xBB but s1[40] = 0xAA → differ.
        let len = match_length(&data, 0, 10, 50);
        assert_eq!(len, 40);
    }

    #[test]
    fn match_length_one_byte_match() {
        let data = vec![0xAA, 0xAA, 0xBB, 0xAA, 0xCC];
        // pos1=0 [AA,AA,BB,...], pos2=3 [AA,CC,...], max=2
        let len = match_length(&data, 0, 3, 2);
        assert_eq!(len, 1); // first byte matches, second differs
    }

    #[test]
    fn match_length_zero_max() {
        let data = vec![0xAA; 100];
        let len = match_length(&data, 0, 50, 0);
        assert_eq!(len, 0);
    }

    #[test]
    fn match_length_exact_boundary() {
        // Match exactly at the data boundary.
        let data = vec![0xAA; 16];
        let len = match_length(&data, 0, 8, 100);
        assert_eq!(len, 8); // limited by data.len() - pos2
    }

    #[test]
    fn portable_match_length_crosses_u64_boundary() {
        // Match that extends past the first 8-byte chunk.
        let mut data = Vec::new();
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 12]); // diff at byte 10
        let len = match_length_portable(&data[..12], &data[12..24]);
        assert_eq!(len, 10);
    }

    #[test]
    fn portable_match_length_all_equal() {
        let s1 = vec![0x42; 1000];
        let s2 = vec![0x42; 1000];
        assert_eq!(match_length_portable(&s1, &s2), 1000);
    }

    #[test]
    fn portable_match_length_first_byte_differs() {
        let s1 = [0x00, 0x01];
        let s2 = [0xFF, 0x01];
        assert_eq!(match_length_portable(&s1, &s2), 0);
    }

    #[test]
    fn portable_match_length_single_byte_slices() {
        assert_eq!(match_length_portable(&[0xAA], &[0xAA]), 1);
        assert_eq!(match_length_portable(&[0xAA], &[0xBB]), 0);
    }

    #[test]
    fn portable_match_length_unequal_lengths() {
        let s1 = vec![0xAA; 100];
        let s2 = vec![0xAA; 50];
        assert_eq!(match_length_portable(&s1, &s2), 50);
    }

    // Only run AVX2 tests on x86_64 where it might be available.
    #[cfg(target_arch = "x86_64")]
    mod avx2_tests {
        use super::*;

        fn has_avx2() -> bool {
            is_x86_feature_detected!("avx2")
        }

        #[test]
        fn avx2_match_length_identical() {
            if !has_avx2() {
                return;
            }
            let data = vec![0xAA; 2048];
            let len = match_length(&data, 0, 1024, 1024);
            assert_eq!(len, 1024);
        }

        #[test]
        fn avx2_match_length_mismatch_in_first_32() {
            if !has_avx2() {
                return;
            }
            let s1 = vec![0xAA; 64];
            let mut s2 = vec![0xAA; 64];
            s2[15] = 0xBB; // mismatch at byte 15
            let len = match_length_portable(&s1, &s2);
            assert_eq!(len, 15);
            // Also test via the main entry point.
            let mut data = Vec::new();
            data.extend_from_slice(&s1);
            data.extend_from_slice(&s2);
            let len = match_length(&data, 0, 64, 64);
            assert_eq!(len, 15);
        }

        #[test]
        fn avx2_match_length_mismatch_after_32() {
            if !has_avx2() {
                return;
            }
            let mut data = vec![0xAA; 128];
            // Place mismatch at offset 40 from start of s2 region.
            data[64 + 40] = 0xBB;
            let len = match_length(&data, 0, 64, 64);
            assert_eq!(len, 40);
        }

        #[test]
        fn avx2_match_length_remainder_bytes() {
            if !has_avx2() {
                return;
            }
            // 37 bytes — 32 matched by AVX2, 5 by portable fallback.
            let mut data = vec![0xAA; 74];
            data[74 - 1] = 0xBB; // no mismatch within first 37
                                 // But the second slice starts at 37, so s2 = data[37..74].
                                 // s1[36] = 0xAA, s2[36] = data[73] = 0xBB → mismatch at 36.
            let len = match_length(&data, 0, 37, 37);
            assert_eq!(len, 36);
        }

        #[test]
        fn avx2_match_length_large_identical() {
            if !has_avx2() {
                return;
            }
            // 10 KiB of identical data — exercises multiple AVX2 iterations.
            let data = vec![0x55; 20480];
            let len = match_length(&data, 0, 10240, 10240);
            assert_eq!(len, 10240);
        }
    }

    // Integration test: match_length dispatches correctly.
    #[test]
    fn match_length_dispatch_consistency() {
        // Verify that the dispatched match_length agrees with portable.
        let mut data = vec![0xCC; 512];
        data[300] = 0xDD; // mismatch in the middle

        let portable_len = match_length_portable(&data[..256], &data[256..]);
        let dispatch_len = match_length(&data, 0, 256, 256);

        assert_eq!(portable_len, dispatch_len);
    }
}
