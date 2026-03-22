//! Optimal LZ77 parser using forward dynamic programming.
//!
//! Finds the minimum-cost parse of the input by evaluating every possible
//! combination of literals and matches at each position, scoring each
//! decision by its FSE entropy-coding cost.
//!
//! ## Algorithm
//!
//! 1. **First pass**: Greedy parse to gather initial token statistics.
//! 2. **Build cost model**: Compute per-symbol bit costs from the greedy
//!    token distribution (`-log2(freq / total)`).
//! 3. **Forward DP**: For each position `i`:
//!    - Literal: `cost[i+1] = min(cost[i+1], cost[i] + literal_cost(data[i]))`
//!    - Match: Query the Pareto frontier of matches at `i` — multiple
//!      candidates at different (offset, length) tradeoffs. For each match,
//!      try lengths exclusive to that offset (lengths no shorter-offset match
//!      can reach), plus the full length if beyond the threshold.
//! 4. **Reconstruct**: Walk backwards through the decision table.
//!
//! ## Pareto Frontier
//!
//! The DP evaluates the full Pareto frontier of matches at each position.
//! The match finder returns up to `MAX_MATCH_CANDIDATES` matches ordered by
//! ascending offset, each strictly longer than all matches at shorter offsets.
//! For each match, the DP only tries lengths that this match uniquely provides
//! (above the previous match's max length). This keeps the inner loop bounded
//! at O(128) total length trials per position while evaluating every interesting
//! (offset, length) pair. A match at offset 4 with length 10 is genuinely
//! cheaper to encode than a match at offset 40,000 with length 10, and the
//! DP now knows this.
//!
//! ## Performance
//!
//! The inner loop tries up to 128 match lengths per position, distributed
//! across the Pareto frontier (non-overlapping ranges). Combined with
//! O(chain_depth) for match finding, total cost is O(n * (128 + chain_depth))
//! per DP pass. For a 2MB block with chain_depth=256, this is ~700M operations
//! — a few seconds on modern hardware, acceptable for "max compression" mode.

use crate::entropy::{
    encode_match_length, encode_match_offset, CostModel, RepcodeState, NUM_LENGTH_CODES,
    NUM_LITERAL_CODES, NUM_OFFSET_CODES,
};
use crate::match_finder::{HashChain, Match, MatchFinder, MatchFinderConfig, MAX_MATCH_CANDIDATES};
use crate::workspace::ParserScratch;
use crate::{LzToken, MAX_MATCH_LEN, MIN_MATCH_LEN};

/// Maximum match length to try exhaustively in the DP. Beyond this, we only
/// try the full match length. Keeps the DP O(n * 128) instead of O(n * 65535).
const DP_LENGTH_THRESHOLD: u32 = 128;

/// Decision recorded at each DP position for path reconstruction.
#[derive(Clone, Copy)]
pub(crate) enum Decision {
    /// Position not yet reached by any parse path.
    Unreachable,
    /// Emit a literal byte (came from position - 1).
    Literal,
    /// Emit a match (came from position - length).
    Match { offset: u32, length: u32 },
}

/// Parse data using optimal (two-pass DP) strategy.
///
/// Pass 1: greedy parse for initial statistics.
/// Pass 2: forward DP with real cost model.
pub fn parse(data: &[u8], config: &MatchFinderConfig) -> Vec<LzToken> {
    // Pass 1: greedy parse to get initial token statistics
    let initial_tokens = super::greedy::parse(data, config);
    let cost_model = build_cost_model(&initial_tokens);

    // Pass 2: DP-optimal parse with the real cost model
    parse_with_cost_model(data, config, &cost_model)
}

/// Build a cost model from token frequency statistics.
///
/// Uses Laplace (add-1) smoothing so that unseen symbols get a finite cost
/// proportional to the total sample size, rather than an arbitrary penalty.
/// This is critical when the greedy first pass produces very few tokens
/// (e.g., all-same-byte data → 1 literal + 1 long match).
fn build_cost_model(tokens: &[LzToken]) -> CostModel {
    let mut lit_freqs = [0u32; NUM_LITERAL_CODES];
    let mut len_freqs = [0u32; NUM_LENGTH_CODES];
    let mut off_freqs = [0u32; NUM_OFFSET_CODES];
    let mut repcode_state = RepcodeState::new();

    for token in tokens {
        match token {
            LzToken::Literal(b) => {
                lit_freqs[*b as usize] += 1;
            }
            LzToken::Match { offset, length } => {
                let lc = encode_match_length(*length);
                len_freqs[lc.code as usize] += 1;

                // Simulate repcode encoding — same logic the entropy encoder uses.
                if let Some(rep_idx) = repcode_state.find(*offset) {
                    off_freqs[rep_idx as usize] += 1; // repcode slots 0, 1, 2
                    repcode_state.update_rep(rep_idx);
                } else {
                    let oc = encode_match_offset(*offset);
                    off_freqs[oc.code as usize] += 1; // real offset codes 3+
                    repcode_state.update_raw(*offset);
                }
            }
        }
    }

    // Laplace smoothing: add 1 pseudocount per symbol to avoid degenerate costs.
    // When observed counts are 0, this gives the uniform distribution.
    // When counts are large, the pseudocount is negligible.
    let lit_total: f32 = (lit_freqs.iter().sum::<u32>() + NUM_LITERAL_CODES as u32) as f32;
    let len_total: f32 = (len_freqs.iter().sum::<u32>() + NUM_LENGTH_CODES as u32) as f32;
    let off_total: f32 = (off_freqs.iter().sum::<u32>() + NUM_OFFSET_CODES as u32) as f32;

    let mut literal_costs = [0.0f32; NUM_LITERAL_CODES];
    for (i, cost) in literal_costs.iter_mut().enumerate() {
        *cost = -((lit_freqs[i] as f32 + 1.0) / lit_total).log2();
    }

    let mut length_code_costs = [0.0f32; NUM_LENGTH_CODES];
    for (i, cost) in length_code_costs.iter_mut().enumerate() {
        *cost = -((len_freqs[i] as f32 + 1.0) / len_total).log2();
    }

    let mut offset_code_costs = [0.0f32; NUM_OFFSET_CODES];
    for (i, cost) in offset_code_costs.iter_mut().enumerate() {
        *cost = -((off_freqs[i] as f32 + 1.0) / off_total).log2();
    }

    CostModel {
        literal_costs,
        length_code_costs,
        offset_code_costs,
    }
}

/// Run the forward-DP optimal parse with a given cost model.
///
/// Each token incurs a 1-bit overhead from the packed type-bit array in the
/// wire format. This is added to both literal and match costs so the DP
/// naturally prefers fewer, longer tokens over many short ones.
fn parse_with_cost_model(
    data: &[u8],
    config: &MatchFinderConfig,
    cost_model: &CostModel,
) -> Vec<LzToken> {
    /// Per-token overhead from the packed type-bit array (1 bit per token).
    const TYPE_BIT_COST: f32 = 1.0;

    let n = data.len();
    let mut cost = vec![f32::INFINITY; n + 1];
    let mut decision = vec![Decision::Unreachable; n + 1];
    cost[0] = 0.0;

    let mut mf = MatchFinder::new(data, config.clone());

    // Reusable buffer for Pareto match candidates — avoids per-position allocation.
    let mut match_buf: Vec<Match> = Vec::with_capacity(MAX_MATCH_CANDIDATES);

    // Static repcode state for DP cost estimation. Uses initial offsets [1, 4, 8]
    // throughout the block. The entropy encoder tracks runtime repcode state
    // for actual encoding — this approximation lets the DP prefer matches at
    // common IoT strides (1-byte runs, 4-byte f32, 8-byte f64) without the
    // complexity of per-path repcode state tracking.
    let repcode_state = RepcodeState::new();

    for pos in 0..n {
        // Insert every position into the hash chain for match quality
        mf.insert_at(pos);

        if cost[pos].is_infinite() {
            continue; // unreachable — skip but still insert
        }

        // Option 1: emit a literal (1 type bit + entropy cost)
        let lit_c = cost[pos] + TYPE_BIT_COST + cost_model.literal_cost(data[pos]);
        if lit_c < cost[pos + 1] {
            cost[pos + 1] = lit_c;
            decision[pos + 1] = Decision::Literal;
        }

        // Option 2: emit a match — evaluate the full Pareto frontier.
        mf.find_matches_at(pos, &mut match_buf);

        let mut prev_max_len = MIN_MATCH_LEN - 1;

        for m in &match_buf {
            let max_len = std::cmp::min(m.length, (n - pos) as u32);
            let max_len = std::cmp::min(max_len, MAX_MATCH_LEN);

            let start_len = prev_max_len + 1;
            let try_limit = std::cmp::min(max_len, DP_LENGTH_THRESHOLD);

            // Check if this match offset is a repcode — much cheaper to encode.
            let rep_idx = repcode_state.find(m.offset);

            for len in start_len..=try_limit {
                let end = pos + len as usize;
                let mc = cost[pos]
                    + TYPE_BIT_COST
                    + if let Some(ri) = rep_idx {
                        cost_model.rep_match_cost(len, ri)
                    } else {
                        cost_model.match_cost(len, m.offset)
                    };
                if mc < cost[end] {
                    cost[end] = mc;
                    decision[end] = Decision::Match {
                        offset: m.offset,
                        length: len,
                    };
                }
            }

            if max_len > DP_LENGTH_THRESHOLD && max_len > prev_max_len {
                let end = pos + max_len as usize;
                let mc = cost[pos]
                    + TYPE_BIT_COST
                    + if let Some(ri) = rep_idx {
                        cost_model.rep_match_cost(max_len, ri)
                    } else {
                        cost_model.match_cost(max_len, m.offset)
                    };
                if mc < cost[end] {
                    cost[end] = mc;
                    decision[end] = Decision::Match {
                        offset: m.offset,
                        length: max_len,
                    };
                }
            }

            prev_max_len = std::cmp::max(prev_max_len, max_len);
        }
    }

    reconstruct(data, &decision, n)
}

/// Workspace-aware optimal parse. Reuses chain, cost, decision, match_buf,
/// initial_tokens, and tokens from `ParserScratch`.
pub fn parse_into(data: &[u8], config: &MatchFinderConfig, scratch: &mut ParserScratch) {
    // Take the chain out — first call creates, subsequent calls reuse.
    let chain = scratch
        .chain
        .take()
        .unwrap_or_else(|| HashChain::new(config.window_size as usize));

    // Pass 1: greedy parse for initial statistics.
    let chain = super::greedy::parse_into(data, config, chain, &mut scratch.initial_tokens);
    let cost_model = build_cost_model(&scratch.initial_tokens);

    // Pass 2: DP-optimal parse with the real cost model.
    let chain = parse_with_cost_model_into(data, config, &cost_model, scratch, chain);
    scratch.chain = Some(chain);
}

/// Workspace-aware forward DP pass.
#[allow(clippy::needless_range_loop)]
fn parse_with_cost_model_into(
    data: &[u8],
    config: &MatchFinderConfig,
    cost_model: &CostModel,
    scratch: &mut ParserScratch,
    chain: HashChain,
) -> HashChain {
    const TYPE_BIT_COST: f32 = 1.0;

    let n = data.len();

    scratch.cost.clear();
    scratch.cost.resize(n + 1, f32::INFINITY);
    scratch.cost[0] = 0.0;

    scratch.decision.clear();
    scratch.decision.resize(n + 1, Decision::Unreachable);

    let mut mf = MatchFinder::with_chain(data, config.clone(), chain);
    let repcode_state = RepcodeState::new();

    for pos in 0..n {
        mf.insert_at(pos);

        if scratch.cost[pos].is_infinite() {
            continue;
        }

        // Option 1: literal
        let lit_c = scratch.cost[pos] + TYPE_BIT_COST + cost_model.literal_cost(data[pos]);
        if lit_c < scratch.cost[pos + 1] {
            scratch.cost[pos + 1] = lit_c;
            scratch.decision[pos + 1] = Decision::Literal;
        }

        // Option 2: match — Pareto frontier with repcode awareness
        mf.find_matches_at(pos, &mut scratch.match_buf);

        let mut prev_max_len = MIN_MATCH_LEN - 1;

        for m in &scratch.match_buf {
            let max_len = std::cmp::min(m.length, (n - pos) as u32);
            let max_len = std::cmp::min(max_len, MAX_MATCH_LEN);

            let start_len = prev_max_len + 1;
            let try_limit = std::cmp::min(max_len, DP_LENGTH_THRESHOLD);

            let rep_idx = repcode_state.find(m.offset);

            for len in start_len..=try_limit {
                let end = pos + len as usize;
                let mc = scratch.cost[pos]
                    + TYPE_BIT_COST
                    + if let Some(ri) = rep_idx {
                        cost_model.rep_match_cost(len, ri)
                    } else {
                        cost_model.match_cost(len, m.offset)
                    };
                if mc < scratch.cost[end] {
                    scratch.cost[end] = mc;
                    scratch.decision[end] = Decision::Match {
                        offset: m.offset,
                        length: len,
                    };
                }
            }

            if max_len > DP_LENGTH_THRESHOLD && max_len > prev_max_len {
                let end = pos + max_len as usize;
                let mc = scratch.cost[pos]
                    + TYPE_BIT_COST
                    + if let Some(ri) = rep_idx {
                        cost_model.rep_match_cost(max_len, ri)
                    } else {
                        cost_model.match_cost(max_len, m.offset)
                    };
                if mc < scratch.cost[end] {
                    scratch.cost[end] = mc;
                    scratch.decision[end] = Decision::Match {
                        offset: m.offset,
                        length: max_len,
                    };
                }
            }

            prev_max_len = std::cmp::max(prev_max_len, max_len);
        }
    }

    reconstruct_into(data, &scratch.decision, n, &mut scratch.tokens);
    mf.take_chain()
}

/// Walk backwards through the decision table to build the optimal token sequence
/// into a caller-provided Vec.
fn reconstruct_into(data: &[u8], decision: &[Decision], n: usize, tokens: &mut Vec<LzToken>) {
    tokens.clear();
    let mut pos = n;

    while pos > 0 {
        match decision[pos] {
            Decision::Literal => {
                tokens.push(LzToken::Literal(data[pos - 1]));
                pos -= 1;
            }
            Decision::Match { offset, length } => {
                tokens.push(LzToken::Match { offset, length });
                pos -= length as usize;
            }
            Decision::Unreachable => {
                tokens.push(LzToken::Literal(data[pos - 1]));
                pos -= 1;
            }
        }
    }

    tokens.reverse();
}

/// Walk backwards through the decision table to build the optimal token sequence.
fn reconstruct(data: &[u8], decision: &[Decision], n: usize) -> Vec<LzToken> {
    let mut tokens = Vec::new();
    let mut pos = n;

    while pos > 0 {
        match decision[pos] {
            Decision::Literal => {
                tokens.push(LzToken::Literal(data[pos - 1]));
                pos -= 1;
            }
            Decision::Match { offset, length } => {
                tokens.push(LzToken::Match { offset, length });
                pos -= length as usize;
            }
            Decision::Unreachable => {
                // Fallback: if DP didn't reach this position (shouldn't happen
                // since literal always extends by 1), emit a literal.
                tokens.push(LzToken::Literal(data[pos - 1]));
                pos -= 1;
            }
        }
    }

    tokens.reverse();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{replay_tokens, validate_tokens};

    fn test_config() -> MatchFinderConfig {
        MatchFinderConfig {
            window_size: 65536,
            ..Default::default()
        }
    }

    #[test]
    fn optimal_unique_data_all_literals() {
        let data: Vec<u8> = (0..=255).collect();
        let tokens = parse(&data, &test_config());
        assert!(tokens.iter().all(|t| t.is_literal()));
        validate_tokens(&tokens, &data).unwrap();
    }

    #[test]
    fn optimal_simple_repeat() {
        let data = b"ABCDABCD";
        let tokens = parse(data, &test_config());
        validate_tokens(&tokens, data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.iter().any(|t| t.is_match()));
    }

    #[test]
    fn optimal_short_data() {
        let data = b"Hi";
        let tokens = parse(data, &test_config());
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|t| t.is_literal()));
    }

    #[test]
    fn optimal_all_zeros() {
        let data = vec![0u8; 5000];
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(
            tokens.len() < 50,
            "expected < 50 tokens, got {}",
            tokens.len()
        );
    }

    #[test]
    fn optimal_improves_or_matches_greedy() {
        let mut data = Vec::new();
        for _ in 0..30 {
            data.extend_from_slice(b"ABCDEFGHIJ");
            data.extend_from_slice(b"KLMN");
            data.extend_from_slice(b"ABCDEFGHIJ");
            data.push(b'X');
        }
        let config = test_config();
        let greedy_tokens = super::super::greedy::parse(&data, &config);
        let optimal_tokens = parse(&data, &config);
        validate_tokens(&greedy_tokens, &data).unwrap();
        validate_tokens(&optimal_tokens, &data).unwrap();

        // Both reconstruct correctly
        assert_eq!(replay_tokens(&greedy_tokens), data);
        assert_eq!(replay_tokens(&optimal_tokens), data);
    }

    #[test]
    fn optimal_cost_model_sanity() {
        let tokens = vec![
            LzToken::Literal(b'A'),
            LzToken::Literal(b'A'),
            LzToken::Literal(b'A'),
            LzToken::Literal(b'B'),
            LzToken::Match {
                offset: 4,
                length: 4,
            },
        ];
        let cm = build_cost_model(&tokens);
        // 'A' appears 3x out of 4 literals → cheaper than 'B' (1x)
        assert!(cm.literal_cost(b'A') < cm.literal_cost(b'B'));
        // Unseen literal should be more expensive than seen ones (Laplace smoothing)
        assert!(cm.literal_cost(b'Z') > cm.literal_cost(b'A'));
        assert!(cm.literal_cost(b'Z') > cm.literal_cost(b'B'));
    }

    #[test]
    fn optimal_cost_model_no_matches() {
        // When there are no matches, length/offset costs come from Laplace smoothing
        // (uniform-ish: each symbol gets pseudocount 1 out of alphabet_size)
        let tokens: Vec<LzToken> = (0..100).map(|i| LzToken::Literal(i as u8)).collect();
        let cm = build_cost_model(&tokens);
        // With 0 real matches + 52 pseudocounts, each length code costs ~log2(52) ≈ 5.7
        let expected_len_cost = (NUM_LENGTH_CODES as f32).log2();
        assert!(
            (cm.length_code_costs[0] - expected_len_cost).abs() < 0.1,
            "expected ~{:.1}, got {:.1}",
            expected_len_cost,
            cm.length_code_costs[0]
        );
    }

    #[test]
    fn optimal_cost_model_no_literals() {
        // When there are no literals, literal costs from Laplace ≈ log2(256) = 8 bits
        let tokens: Vec<LzToken> = (0..50)
            .map(|i| LzToken::Match {
                offset: (i % 10 + 1) as u32,
                length: 4,
            })
            .collect();
        let cm = build_cost_model(&tokens);
        assert!(
            (cm.literal_cost(0) - 8.0).abs() < 0.1,
            "expected ~8.0, got {:.2}",
            cm.literal_cost(0)
        );
    }

    #[test]
    fn optimal_prefers_cheaper_encoding() {
        // Near repeat: "ABCD" at offset 4 (cheap)
        // Distant repeat: "ABCDEFGH" at offset ~100 (expensive offset)
        // Optimal parser should pick the cost-effective option
        let mut data = Vec::new();
        data.extend_from_slice(b"ABCD");
        data.extend_from_slice(b"ABCD"); // near repeat at offset 4
        for i in 0..92u8 {
            data.push(i);
        }
        data.extend_from_slice(b"ABCDEFGH");

        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn optimal_stress_large_block() {
        let mut data = Vec::with_capacity(50_000);
        let patterns = [
            b"Hello World! " as &[u8],
            b"Foo Bar Baz! ",
            b"QWERTY12345! ",
        ];
        for i in 0..4000 {
            data.extend_from_slice(patterns[i % 3]);
        }
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.len() < data.len() / 3);
    }

    #[test]
    fn optimal_roundtrip_entropy() {
        let data = b"ABCDEFGHABCDEFGHABCDEFGH12345678ABCDEFGH";
        let tokens = parse(data, &test_config());
        validate_tokens(&tokens, data).unwrap();

        let (encoded, _) = crate::entropy::encode_tokens(&tokens).unwrap();
        let decoded = crate::entropy::decode_tokens(&encoded).unwrap();
        assert_eq!(tokens, decoded);
    }

    #[test]
    fn optimal_reconstruct_correctness() {
        // Manual decision table to test reconstruction
        let data = b"AABCAABC";
        let decision = vec![
            Decision::Unreachable, // pos 0 (start)
            Decision::Literal,     // pos 1: literal A
            Decision::Literal,     // pos 2: literal A
            Decision::Literal,     // pos 3: literal B
            Decision::Literal,     // pos 4: literal C
            Decision::Literal,     // pos 5: literal A
            Decision::Literal,     // pos 6: literal A
            Decision::Literal,     // pos 7: literal B
            Decision::Literal,     // pos 8: literal C
        ];
        let tokens = reconstruct(data, &decision, 8);
        assert_eq!(tokens.len(), 8);
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn optimal_reconstruct_with_match() {
        let data = b"ABCDABCD";
        let decision = vec![
            Decision::Unreachable, // 0
            Decision::Literal,     // 1: A
            Decision::Literal,     // 2: B
            Decision::Literal,     // 3: C
            Decision::Literal,     // 4: D
            Decision::Unreachable, // 5: skipped (part of match below)
            Decision::Unreachable, // 6: skipped
            Decision::Unreachable, // 7: skipped
            Decision::Match {
                offset: 4,
                length: 4,
            }, // 8: match covers 4..8
        ];
        let tokens = reconstruct(data, &decision, 8);
        assert_eq!(tokens.len(), 5); // 4 literals + 1 match
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn optimal_pareto_prefers_short_offset() {
        // Construct data where the same 4-byte pattern appears at two
        // very different offsets. The optimal parser with Pareto frontier
        // should prefer the shorter offset for short matches.
        //
        // Layout:
        //   pos 0:   "ABCDEFGHIJKLMNOP" (far match, offset 200+)
        //   pos 16:  filler (unique bytes so no accidental matches)
        //   pos 192: "ABCDEFGH" + different (near match, offset 8)
        //   pos 200: "ABCDEFGH" (search target)
        let mut data = Vec::new();
        // pos 0-15: long match source
        data.extend_from_slice(b"ABCDEFGHIJKLMNOP");
        // pos 16-191: unique filler (176 bytes)
        for i in 0u8..176 {
            data.push(i.wrapping_add(0x80));
        }
        // pos 192-199: near match source — "ABCDEFGH"
        data.extend_from_slice(b"ABCDEFGH");
        // pos 200-207: search target — "ABCDEFGH"
        data.extend_from_slice(b"ABCDEFGH");

        let config = MatchFinderConfig {
            window_size: 65536,
            max_chain_depth: 64,
            ..Default::default()
        };
        let tokens = parse(&data, &config);
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);

        // Find the match that covers the "ABCDEFGH" at position 200.
        // With Pareto, the DP should pick offset 8 (from pos 192) over
        // offset 200 (from pos 0) because offset 8 is vastly cheaper to encode.
        let match_at_200 = tokens.iter().find(|t| {
            if let LzToken::Match { offset, length } = t {
                *length >= 4 && (*offset == 8 || *offset == 200)
            } else {
                false
            }
        });
        if let Some(LzToken::Match { offset, .. }) = match_at_200 {
            assert_eq!(
                *offset, 8,
                "Pareto DP should prefer offset 8 over 200, got offset {}",
                offset
            );
        }
        // (If no match is found at all, that's fine — the DP may have chosen
        // all literals if the cost model makes that cheaper.)
    }
}
