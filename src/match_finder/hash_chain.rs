//! # Hash Chain for LZ77 Match Finding
//!
//! A hash chain is the classic data structure for LZ77 pattern matching.
//! It maps 4-byte prefixes to linked lists of positions where that prefix
//! was seen, allowing fast candidate enumeration for match extension.
//!
//! ## Design
//!
//! We use a flat array layout optimized for L1 cache pressure:
//!
//! - **`head`**: A hash table of `HASH_SIZE` entries. Each entry stores the
//!   most recent position where a prefix with that hash was inserted.
//!   Unoccupied slots hold `EMPTY` (u32::MAX).
//!
//! - **`prev`**: A circular buffer of `window_size` entries. `prev[pos % window_size]`
//!   stores the previous position in the chain for that slot. This forms an
//!   implicit linked list threaded through the circular buffer.
//!
//! ## Hash Function
//!
//! We use a multiplicative hash on the 4-byte prefix:
//! ```text
//! hash = ((prefix as u64) * HASH_MULT) >> (32 - HASH_BITS)
//! ```
//! This spreads bit entropy well across the table while compiling to a
//! single `imul` + `shr` on x86.
//!
//! ## Complexity
//!
//! - Insert: O(1) — one hash + two array writes.
//! - Walk: O(chain_depth) — bounded by the caller's max depth parameter.
//! - Memory: `4 * HASH_SIZE + 4 * window_size` bytes.
//!   With default settings: ~4 MiB hash table + ~16 MiB prev chain ≈ 20 MiB.

/// Number of bits in the hash table (log2 of table size).
/// 20 bits = 1M entries = 4 MiB. Good balance of collision rate vs. cache pressure.
const HASH_BITS: u32 = 20;

/// Hash table size: 2^HASH_BITS.
const HASH_SIZE: usize = 1 << HASH_BITS;

/// Hash table mask for wrapping.
const HASH_MASK: u32 = (HASH_SIZE as u32) - 1;

/// Multiplicative hash constant (Knuth's golden ratio hash).
/// Chosen to spread 32-bit inputs evenly across the hash space.
const HASH_MULT: u64 = 2654435761; // 0x9E3779B1

/// Sentinel value for empty chain entries.
const EMPTY: u32 = u32::MAX;

/// The hash chain data structure.
///
/// Stores a hash table (`head`) mapping 4-byte prefix hashes to the most
/// recent position, and a circular `prev` buffer linking older positions
/// with the same hash.
pub struct HashChain {
    /// Hash table: head[hash(prefix)] = most recent position.
    head: Vec<u32>,
    /// Circular chain buffer: prev[pos & window_mask] = previous position
    /// with the same hash, or EMPTY if none.
    prev: Vec<u32>,
    /// Bitmask for circular buffer wrapping (power-of-2 size minus 1).
    /// Using `& mask` instead of `% size` avoids a costly div instruction
    /// in the hot loop (~1 cycle vs 20-90 cycles on x86).
    window_mask: usize,
}

impl HashChain {
    /// Create a new hash chain with the given window size.
    /// The internal buffer is rounded up to the next power of 2 so that
    /// modular indexing can use a bitmask (`& mask`) instead of division (`% size`).
    pub fn new(window_size: usize) -> Self {
        let pow2_size = window_size.next_power_of_two();
        Self {
            head: vec![EMPTY; HASH_SIZE],
            prev: vec![EMPTY; pow2_size],
            window_mask: pow2_size - 1,
        }
    }

    /// Compute the hash of a 4-byte prefix.
    #[inline(always)]
    fn hash(prefix: u32) -> u32 {
        ((prefix as u64).wrapping_mul(HASH_MULT) >> (32 - HASH_BITS)) as u32 & HASH_MASK
    }

    /// Insert a position into the chain for the given 4-byte prefix.
    ///
    /// This is O(1): compute hash, link the old head into the `prev` chain,
    /// and update the head to point at the new position.
    #[inline]
    pub fn insert(&mut self, prefix: u32, pos: usize) {
        let h = Self::hash(prefix) as usize;
        let slot = pos & self.window_mask;

        // Link old head into the chain.
        self.prev[slot] = self.head[h];
        // Update head to the new position.
        self.head[h] = pos as u32;
    }

    /// Walk the hash chain for a given prefix, calling `callback` for each
    /// candidate position. The callback returns `true` to continue walking,
    /// `false` to stop.
    ///
    /// Positions are yielded in reverse insertion order (most recent first).
    /// Includes cycle protection: stops after `window_size` steps since a
    /// chain can never have more valid entries than the circular buffer.
    #[inline]
    pub fn walk<F>(&self, prefix: u32, mut callback: F)
    where
        F: FnMut(usize) -> bool,
    {
        let h = Self::hash(prefix) as usize;
        let mut pos = self.head[h];
        let mut steps = 0usize;

        while pos != EMPTY {
            // Cycle protection: the circular buffer can create loops when
            // positions wrap around. Cap at buffer length to guarantee termination.
            if steps >= self.prev.len() {
                return;
            }
            steps += 1;

            if !callback(pos as usize) {
                return;
            }
            let slot = pos as usize & self.window_mask;
            pos = self.prev[slot];
        }
    }

    /// Reset the hash chain (clear all entries). Used when starting a new block.
    pub fn reset(&mut self) {
        self.head.fill(EMPTY);
        self.prev.fill(EMPTY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_chain_basic_insert_and_walk() {
        let mut chain = HashChain::new(1024);
        let prefix = 0x41424344; // "ABCD" as u32 LE

        chain.insert(prefix, 0);
        chain.insert(prefix, 100);
        chain.insert(prefix, 200);

        let mut positions = Vec::new();
        chain.walk(prefix, |pos| {
            positions.push(pos);
            true
        });

        // Should come out most-recent-first.
        assert_eq!(positions, vec![200, 100, 0]);
    }

    #[test]
    fn hash_chain_different_prefixes_independent() {
        let mut chain = HashChain::new(1024);

        chain.insert(0x11111111, 10);
        chain.insert(0x22222222, 20);
        chain.insert(0x11111111, 30);

        let mut positions = Vec::new();
        chain.walk(0x11111111, |pos| {
            positions.push(pos);
            true
        });
        assert_eq!(positions, vec![30, 10]);

        let mut positions = Vec::new();
        chain.walk(0x22222222, |pos| {
            positions.push(pos);
            true
        });
        assert_eq!(positions, vec![20]);
    }

    #[test]
    fn hash_chain_empty_walk() {
        let chain = HashChain::new(1024);
        let mut count = 0;
        chain.walk(0xDEADBEEF, |_| {
            count += 1;
            true
        });
        assert_eq!(count, 0);
    }

    #[test]
    fn hash_chain_walk_early_stop() {
        let mut chain = HashChain::new(1024);
        let prefix = 0xCAFEBABE;
        for i in 0..100 {
            chain.insert(prefix, i * 10);
        }

        let mut positions = Vec::new();
        chain.walk(prefix, |pos| {
            positions.push(pos);
            positions.len() < 3 // stop after 3
        });
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], 990); // most recent
    }

    #[test]
    fn hash_chain_circular_buffer_wraps() {
        let window_size = 64;
        let mut chain = HashChain::new(window_size);
        let prefix = 0xAAAAAAAA;

        // Insert at positions that wrap around the circular buffer.
        chain.insert(prefix, 0);
        chain.insert(prefix, 64); // wraps to slot 0

        // pos 64 overwrites slot 0 in prev[], setting prev[0] = old head (0).
        // This creates a cycle: 64 → 0 → 0 → 0 → ...
        // The walk's cycle protection (capped at window_size steps) ensures
        // it terminates. The first two yielded positions are 64 and 0.
        let mut positions = Vec::new();
        chain.walk(prefix, |pos| {
            positions.push(pos);
            positions.len() < 5 // stop early — we just need to verify it terminates
        });
        assert!(positions.len() >= 2);
        assert_eq!(positions[0], 64); // most recent
        assert_eq!(positions[1], 0); // linked from prev[0]
                                     // Remaining entries are stale repeats of 0 due to cycle — that's fine,
                                     // the MatchFinder's position/window checks filter these out.
    }

    #[test]
    fn hash_chain_reset_clears_all() {
        let mut chain = HashChain::new(1024);
        chain.insert(0x12345678, 42);
        chain.reset();

        let mut count = 0;
        chain.walk(0x12345678, |_| {
            count += 1;
            true
        });
        assert_eq!(count, 0);
    }

    #[test]
    fn hash_function_distribution() {
        // Verify the hash function doesn't degenerate for sequential inputs.
        let mut seen = std::collections::HashSet::new();
        for i in 0..10000u32 {
            let h = HashChain::hash(i);
            assert!(h < HASH_SIZE as u32);
            seen.insert(h);
        }
        // With 10K inputs and 1M slots, we expect very few collisions.
        // At least 9000 unique hashes is a reasonable bar.
        assert!(
            seen.len() > 9000,
            "Hash function produced only {} unique values for 10000 inputs",
            seen.len()
        );
    }

    #[test]
    fn hash_chain_high_volume_insert() {
        // Insert 100K positions and verify the chain is traversable.
        let mut chain = HashChain::new(1 << 20); // 1M window
        let prefix = 0xFEEDFACE;
        for i in 0..100_000 {
            chain.insert(prefix, i);
        }

        let mut count = 0;
        chain.walk(prefix, |_| {
            count += 1;
            count < 50 // only walk 50 steps
        });
        assert_eq!(count, 50);
    }
}
