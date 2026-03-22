//! # LZ77 Match Finder
//!
//! The pattern-matching engine at the heart of the compressor. Given a byte
//! stream, it discovers repeated sequences (matches) that can be encoded as
//! back-references (offset, length) instead of raw literals.
//!
//! ## Architecture
//!
//! The match finder is built in two layers:
//!
//! 1. **Hash Chain** (`hash_chain.rs`): A hash table mapping 4-byte prefixes
//!    to chains of positions where that prefix occurred. This narrows the
//!    search space from the entire window to a small set of candidates.
//!
//! 2. **Match Extension** (`simd_x86.rs`): Once a candidate position is found
//!    via the hash chain, we extend the match forward byte-by-byte (or, on
//!    x86_64 with AVX2, 32 bytes at a time using `_mm256_cmpeq_epi8`).
//!
//! The public interface is the `MatchFinder` struct, which combines both
//! layers behind a simple `find_matches_at()` API.

pub mod hash_chain;
pub mod simd_x86;

pub use hash_chain::HashChain;

/// A candidate match found by the match finder.
///
/// This is a raw match — the parser decides whether to accept it based on
/// cost modeling. The match finder just reports what it sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Byte offset backward from the current position (distance). Always >= 1.
    pub offset: u32,
    /// Length of the match in bytes. Always >= MIN_MATCH_LEN.
    pub length: u32,
}

impl Match {
    #[inline]
    pub fn new(offset: u32, length: u32) -> Self {
        Self { offset, length }
    }
}

/// Configuration for the match finder.
#[derive(Debug, Clone)]
pub struct MatchFinderConfig {
    /// Maximum number of hash chain entries to walk before giving up.
    /// Higher = better matches found, slower search.
    pub max_chain_depth: u32,
    /// Minimum match length to report. Must be >= 4.
    pub min_match_len: u32,
    /// Maximum match length to report.
    pub max_match_len: u32,
    /// Sliding window size in bytes. Matches beyond this distance are not reported.
    pub window_size: u32,
}

impl Default for MatchFinderConfig {
    fn default() -> Self {
        Self {
            max_chain_depth: 64,
            min_match_len: crate::MIN_MATCH_LEN,
            max_match_len: crate::MAX_MATCH_LEN,
            window_size: crate::MAX_MATCH_OFFSET,
        }
    }
}

/// Maximum number of Pareto-optimal match candidates collected per position
/// by `find_matches_at()`. Beyond 8, diminishing quality returns vs. DP
/// inner-loop cost. Each candidate is strictly longer than all candidates at
/// shorter offsets, so the set is small in practice.
pub const MAX_MATCH_CANDIDATES: usize = 8;

/// The main match finder. Wraps a hash chain and provides match queries
/// against a sliding window over the input data.
///
/// Usage pattern:
/// 1. Create with `new()` and the input data.
/// 2. For each position, call `find_best_match_at()` to get the best match,
///    or `find_matches_at()` for the full Pareto frontier (optimal parser).
/// 3. Call `advance()` or `advance_by()` to move the window forward,
///    inserting skipped positions into the hash chain.
pub struct MatchFinder<'a> {
    /// The full input data.
    data: &'a [u8],
    /// Hash chain for 4-byte prefix lookups.
    chain: hash_chain::HashChain,
    /// Configuration.
    config: MatchFinderConfig,
    /// Current position in the input (the "cursor").
    pos: usize,
    /// Resolved function pointer for SIMD match extension. Feature detection
    /// runs once at construction instead of on every match_length call.
    match_len_fn: fn(&[u8], usize, usize, usize) -> usize,
}

impl<'a> MatchFinder<'a> {
    /// Create a new match finder over the given data.
    pub fn new(data: &'a [u8], config: MatchFinderConfig) -> Self {
        let chain = hash_chain::HashChain::new(config.window_size as usize);
        let match_len_fn = simd_x86::get_match_length_fn();
        Self {
            data,
            chain,
            config,
            pos: 0,
            match_len_fn,
        }
    }

    /// Create a match finder reusing an existing `HashChain`.
    ///
    /// The chain is `reset()` before use. This eliminates the dominant
    /// allocation in the compress path (4 MB hash table + ~20 MB prev buffer)
    /// by reusing the same backing memory across blocks.
    pub fn with_chain(data: &'a [u8], config: MatchFinderConfig, mut chain: HashChain) -> Self {
        chain.reset();
        let match_len_fn = simd_x86::get_match_length_fn();
        Self {
            data,
            chain,
            config,
            pos: 0,
            match_len_fn,
        }
    }

    /// Consume the match finder and return the owned `HashChain`.
    ///
    /// Paired with `with_chain()` for workspace-based reuse: take the chain
    /// out of the workspace, build a MatchFinder, parse, then return the chain.
    pub fn take_chain(self) -> HashChain {
        self.chain
    }

    /// Current cursor position in the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Whether the cursor has reached the end of the data.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Bytes remaining from the cursor to the end of the data.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Find the best match at the current position.
    ///
    /// Returns `None` if no match of at least `min_match_len` bytes is found
    /// within the window. The returned match has the longest length among all
    /// candidates (ties broken by shortest offset).
    ///
    /// This does NOT advance the cursor — call `advance()` separately.
    pub fn find_best_match_at(&self, pos: usize) -> Option<Match> {
        // Need at least min_match_len bytes remaining.
        if pos + self.config.min_match_len as usize > self.data.len() {
            return None;
        }

        // Need 4 bytes for the hash prefix lookup.
        if pos + 4 > self.data.len() {
            return None;
        }

        let prefix = read_u32_le(&self.data[pos..]);
        let max_len =
            std::cmp::min(self.config.max_match_len as usize, self.data.len() - pos) as u32;

        let mut best: Option<Match> = None;
        let mut best_len = self.config.min_match_len - 1; // must beat this

        let min_valid_pos = pos.saturating_sub(self.config.window_size as usize);

        let mut steps = 0u32;
        self.chain.walk(prefix, |candidate_pos| {
            // Stop if we've exceeded chain depth.
            if steps >= self.config.max_chain_depth {
                return false; // stop walking
            }
            steps += 1;

            // Stop if candidate is outside the window.
            if candidate_pos < min_valid_pos || candidate_pos >= pos {
                return true; // continue to next — might be in range
            }

            let offset = (pos - candidate_pos) as u32;

            // Quick reject: check the byte at best_len position.
            // If it doesn't match, this candidate can't beat our current best.
            if best_len > 0 {
                let check_pos = best_len as usize;
                if pos + check_pos < self.data.len()
                    && candidate_pos + check_pos < self.data.len()
                    && self.data[pos + check_pos] != self.data[candidate_pos + check_pos]
                {
                    return true; // continue
                }
            }

            // Extend the match to find its full length.
            let match_len =
                (self.match_len_fn)(self.data, pos, candidate_pos, max_len as usize) as u32;

            if match_len > best_len {
                best_len = match_len;
                best = Some(Match::new(offset, match_len));

                // Early exit if we've hit the maximum possible length.
                if match_len >= max_len {
                    return false; // stop
                }
            } else if match_len == best_len {
                // Prefer shorter offset on tie (better for entropy coding).
                if let Some(ref b) = best {
                    if offset < b.offset {
                        best = Some(Match::new(offset, match_len));
                    }
                }
            }

            true // continue walking
        });

        best
    }

    /// Find multiple matches at `pos` representing the Pareto frontier of
    /// (offset, length) tradeoffs for the optimal parser's DP.
    ///
    /// The hash chain visits candidates in order of increasing offset (most
    /// recently inserted positions first). A candidate is kept iff its match
    /// length strictly exceeds the best length seen so far from any shorter
    /// offset. This produces matches where each one provides lengths that no
    /// cheaper (shorter-offset) match can reach.
    ///
    /// Results are ordered by ascending offset and strictly ascending length.
    /// The caller reuses the `matches` buffer across positions to avoid
    /// per-position allocation.
    pub fn find_matches_at(&self, pos: usize, matches: &mut Vec<Match>) {
        matches.clear();

        if pos + self.config.min_match_len as usize > self.data.len() {
            return;
        }
        if pos + 4 > self.data.len() {
            return;
        }

        let prefix = read_u32_le(&self.data[pos..]);
        let max_len =
            std::cmp::min(self.config.max_match_len as usize, self.data.len() - pos) as u32;

        let min_valid_pos = pos.saturating_sub(self.config.window_size as usize);

        // Tracks the longest match found so far. A new candidate must beat
        // this to be Pareto-optimal (it's at a longer offset, so it must
        // compensate by being strictly longer).
        let mut best_len = self.config.min_match_len - 1;
        let mut steps = 0u32;

        self.chain.walk(prefix, |candidate_pos| {
            if steps >= self.config.max_chain_depth {
                return false;
            }
            steps += 1;

            if candidate_pos < min_valid_pos || candidate_pos >= pos {
                return true;
            }

            if matches.len() >= MAX_MATCH_CANDIDATES {
                return false;
            }

            let offset = (pos - candidate_pos) as u32;

            // Quick reject: if the byte at best_len doesn't match, this
            // candidate can't produce a longer match than what we already have.
            if best_len > 0 {
                let check_pos = best_len as usize;
                if pos + check_pos < self.data.len()
                    && candidate_pos + check_pos < self.data.len()
                    && self.data[pos + check_pos] != self.data[candidate_pos + check_pos]
                {
                    return true;
                }
            }

            let match_len =
                (self.match_len_fn)(self.data, pos, candidate_pos, max_len as usize) as u32;

            if match_len < self.config.min_match_len {
                return true;
            }

            if match_len > best_len {
                // Strictly longer than everything at shorter offsets → Pareto-optimal.
                best_len = match_len;
                matches.push(Match::new(offset, match_len));

                if match_len >= max_len {
                    return false; // hit maximum possible length, stop
                }
            }
            // If match_len <= best_len: dominated (equal or shorter length at a
            // longer offset). The DP already has a cheaper option for these lengths.

            true
        });
    }

    /// Insert the current position into the hash chain and advance by 1.
    pub fn advance(&mut self) {
        if self.pos + 4 <= self.data.len() {
            let prefix = read_u32_le(&self.data[self.pos..]);
            self.chain.insert(prefix, self.pos);
        }
        self.pos += 1;
    }

    /// Insert positions and advance the cursor by `n` bytes.
    /// Every position is inserted into the hash chain (important for
    /// maintaining match quality after literal/match decisions).
    pub fn advance_by(&mut self, n: usize) {
        for _ in 0..n {
            if self.pos >= self.data.len() {
                break;
            }
            self.advance();
        }
    }

    /// Insert a position into the hash chain without advancing the cursor.
    /// Used by the lazy/optimal parsers when they need to insert positions
    /// they've decided to skip.
    pub fn insert_at(&mut self, pos: usize) {
        if pos + 4 <= self.data.len() {
            let prefix = read_u32_le(&self.data[pos..]);
            self.chain.insert(prefix, pos);
        }
    }

    /// Get the underlying data slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

/// Read a little-endian u32 from a byte slice (unaligned).
#[inline(always)]
fn read_u32_le(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-friendly config with a small window to avoid allocating
    /// the full 4 MiB hash chain per test.
    fn test_config() -> MatchFinderConfig {
        MatchFinderConfig {
            window_size: 65536, // 64 KiB window — plenty for tests
            ..Default::default()
        }
    }

    #[test]
    fn match_finder_no_match_in_unique_data() {
        // Every 4-byte window is unique → no matches.
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let config = test_config();
        let mut mf = MatchFinder::new(&data, config);

        // Advance through the first 128 bytes, then check for matches.
        mf.advance_by(128);
        // At position 128, the 4-byte prefix [128,129,130,131] hasn't been seen
        // in the same order before (since we've only stored 0-127 prefixes).
        let m = mf.find_best_match_at(128);
        assert!(m.is_none());
    }

    #[test]
    fn match_finder_finds_repeat() {
        let data = b"ABCDABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(4);
        let m = mf.find_best_match_at(4).unwrap();
        assert_eq!(m.offset, 4);
        assert_eq!(m.length, 4);
    }

    #[test]
    fn match_finder_longer_match_preferred() {
        let data = b"ABCDEFABCDEF";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(6);
        let m = mf.find_best_match_at(6).unwrap();
        assert_eq!(m.offset, 6);
        assert_eq!(m.length, 6);
    }

    #[test]
    fn match_finder_respects_window_size() {
        let mut data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        for i in 0..100 {
            data.push(i);
        }
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        let config = MatchFinderConfig {
            window_size: 50,
            min_match_len: 4,
            ..test_config()
        };
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(104);
        let m = mf.find_best_match_at(104);
        assert!(m.is_none(), "match outside window should not be found");
    }

    #[test]
    fn match_finder_respects_max_chain_depth() {
        let mut data = Vec::new();
        for i in 0u8..100 {
            data.extend_from_slice(&[0x58, 0x58, 0x58, 0x58, i, i, i, i]);
        }
        data.extend_from_slice(&[0x58, 0x58, 0x58, 0x58, 0xFF, 0xFF, 0xFF, 0xFF]);

        let config = MatchFinderConfig {
            max_chain_depth: 2,
            min_match_len: 4,
            ..test_config()
        };
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(800);
        let m = mf.find_best_match_at(800);
        assert!(m.is_some());
        assert!(m.unwrap().length >= 4);
    }

    #[test]
    fn match_finder_prefers_shorter_offset_on_tie() {
        let data = b"ABCDXXXXABCDYYYYABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(16);
        let m = mf.find_best_match_at(16).unwrap();
        assert_eq!(m.length, 4);
        assert_eq!(m.offset, 8);
    }

    #[test]
    fn match_finder_insert_at_independent() {
        let data = b"ABCDABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        assert_eq!(mf.pos(), 0);
        mf.insert_at(0);
        assert_eq!(mf.pos(), 0);
    }

    #[test]
    fn match_finder_large_repeated_block() {
        let pattern: Vec<u8> = (0..=255).collect();
        let mut data = Vec::with_capacity(4096);
        for _ in 0..16 {
            data.extend_from_slice(&pattern);
        }

        let config = test_config();
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(256);

        let m = mf.find_best_match_at(256).unwrap();
        assert_eq!(m.offset, 256);
        assert!(
            m.length >= 256,
            "Expected at least 256-byte match, got {}",
            m.length
        );
    }

    #[test]
    fn match_finder_stress_all_zeros() {
        let data = vec![0u8; 10240];
        let config = MatchFinderConfig {
            max_chain_depth: 16,
            ..test_config()
        };
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(1000);

        let m = mf.find_best_match_at(1000).unwrap();
        assert!(m.length >= 100, "Expected long match in all-zeros data");
    }

    // ── find_matches_at (Pareto frontier) ─────────────────────────

    #[test]
    fn find_matches_at_pareto_frontier() {
        // Construct data where the prefix "ABCD" occurs at three positions
        // with different amounts of matching data beyond the prefix.
        //
        // pos  0: ABCDEFGHIJKLMNOP (16 matching bytes from pos 80)
        // pos 20: fill
        // pos 40: ABCDEFGH + junk  (8 matching bytes from pos 80)
        // pos 48: fill
        // pos 60: ABCDEF + junk    (6 matching bytes from pos 80)
        // pos 66: fill
        // pos 80: ABCDEFGHIJKLMNOP (search position)
        let mut data = vec![0xFFu8; 96];

        // pos 0: full 16-byte match source
        data[0..16].copy_from_slice(b"ABCDEFGHIJKLMNOP");
        // pos 40: 8-byte match source, then different bytes
        data[40..48].copy_from_slice(b"ABCDEFGH");
        data[48..50].copy_from_slice(&[0xEE, 0xEE]);
        // pos 60: 6-byte match source, then different bytes
        data[60..66].copy_from_slice(b"ABCDEF");
        data[66..68].copy_from_slice(&[0xDD, 0xDD]);
        // pos 80: the search position — 16 matching bytes
        data[80..96].copy_from_slice(b"ABCDEFGHIJKLMNOP");

        let config = test_config();
        let mut mf = MatchFinder::new(&data, config);
        // Insert all positions up to (but not including) 80 into the hash chain.
        mf.advance_by(80);

        let mut matches = Vec::new();
        mf.find_matches_at(80, &mut matches);

        // Expect 3 Pareto matches (in ascending offset order):
        //   offset=20  (from pos 60): length=6
        //   offset=40  (from pos 40): length=8
        //   offset=80  (from pos 0):  length=16
        assert!(
            matches.len() >= 3,
            "expected at least 3 Pareto matches, got {}: {:?}",
            matches.len(),
            matches
        );

        // Verify strictly ascending lengths (Pareto property).
        for i in 1..matches.len() {
            assert!(
                matches[i].length > matches[i - 1].length,
                "Pareto violation: match[{}]={:?} not longer than match[{}]={:?}",
                i,
                matches[i],
                i - 1,
                matches[i - 1]
            );
        }

        // Verify ascending offsets.
        for i in 1..matches.len() {
            assert!(
                matches[i].offset > matches[i - 1].offset,
                "offset order violation: match[{}]={:?} not farther than match[{}]={:?}",
                i,
                matches[i],
                i - 1,
                matches[i - 1]
            );
        }

        // The shortest-offset match should have the shortest length.
        assert_eq!(matches[0].length, 6);
        // The longest match should be 16.
        assert_eq!(matches.last().unwrap().length, 16);
    }

    #[test]
    fn find_matches_at_single_match_is_pareto() {
        // When only one match exists, it should be the sole Pareto candidate.
        let data = b"ABCDABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(4);

        let mut matches = Vec::new();
        mf.find_matches_at(4, &mut matches);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].offset, 4);
        assert_eq!(matches[0].length, 4);
    }

    #[test]
    fn find_matches_at_no_match() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let config = test_config();
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(128);

        let mut matches = Vec::new();
        mf.find_matches_at(128, &mut matches);
        assert!(matches.is_empty());
    }

    #[test]
    fn find_matches_at_dominated_candidates_filtered() {
        // Two occurrences of "ABCD" at different distances, but both
        // only match 4 bytes. The farther one should be filtered out
        // (dominated: same length, worse offset).
        let data = b"ABCDXXXXABCDYYYYABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(16);

        let mut matches = Vec::new();
        mf.find_matches_at(16, &mut matches);

        // Only 1 match should survive — the one at shorter offset.
        assert_eq!(
            matches.len(),
            1,
            "dominated match should be filtered: {:?}",
            matches
        );
        assert_eq!(matches[0].offset, 8); // nearer occurrence
        assert_eq!(matches[0].length, 4);
    }

    #[test]
    fn find_matches_at_buffer_reuse() {
        let data = b"ABCDABCDABCD";
        let config = test_config();
        let mut mf = MatchFinder::new(data, config);
        mf.advance_by(8);

        let mut matches = Vec::new();

        // First call.
        mf.find_matches_at(8, &mut matches);
        assert!(!matches.is_empty());
        let first_result = matches.clone();

        // Second call with same buffer — should clear and repopulate.
        mf.find_matches_at(8, &mut matches);
        assert_eq!(matches, first_result);
    }

    #[test]
    fn find_matches_at_agrees_with_find_best() {
        // find_matches_at's longest match should equal find_best_match_at's result.
        let pattern: Vec<u8> = (0..=255).collect();
        let mut data = Vec::with_capacity(4096);
        for _ in 0..16 {
            data.extend_from_slice(&pattern);
        }

        let config = test_config();
        let mut mf = MatchFinder::new(&data, config);
        mf.advance_by(256);

        let best = mf.find_best_match_at(256).unwrap();
        let mut matches = Vec::new();
        mf.find_matches_at(256, &mut matches);

        // The last Pareto match should be the longest overall.
        let pareto_best = matches.last().unwrap();
        assert_eq!(
            pareto_best.length, best.length,
            "Pareto longest ({}) != find_best_match_at ({})",
            pareto_best.length, best.length
        );
    }
}
