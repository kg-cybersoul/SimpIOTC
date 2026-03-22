//! Greedy LZ77 parser.
//!
//! At each position, takes the longest match found by the match finder.
//! If no match meets the minimum length, emits a literal byte. This is
//! the simplest and fastest parsing strategy — O(n * chain_depth).

use crate::match_finder::{HashChain, MatchFinder, MatchFinderConfig};
use crate::LzToken;

/// Parse data greedily: always take the longest available match.
pub fn parse(data: &[u8], config: &MatchFinderConfig) -> Vec<LzToken> {
    let mut mf = MatchFinder::new(data, config.clone());
    let mut tokens = Vec::with_capacity(data.len() / 2);

    while !mf.is_done() {
        let pos = mf.pos();

        match mf.find_best_match_at(pos) {
            Some(m) => {
                tokens.push(LzToken::Match {
                    offset: m.offset,
                    length: m.length,
                });
                mf.advance_by(m.length as usize);
            }
            None => {
                tokens.push(LzToken::Literal(data[pos]));
                mf.advance();
            }
        }
    }

    tokens
}

/// Workspace-aware greedy parse. Reuses the hash chain and tokens Vec.
/// Returns the hash chain for the caller to store back in the workspace.
pub fn parse_into(
    data: &[u8],
    config: &MatchFinderConfig,
    chain: HashChain,
    tokens: &mut Vec<LzToken>,
) -> HashChain {
    let mut mf = MatchFinder::with_chain(data, config.clone(), chain);
    tokens.clear();
    tokens.reserve(data.len() / 2);

    while !mf.is_done() {
        let pos = mf.pos();

        match mf.find_best_match_at(pos) {
            Some(m) => {
                tokens.push(LzToken::Match {
                    offset: m.offset,
                    length: m.length,
                });
                mf.advance_by(m.length as usize);
            }
            None => {
                tokens.push(LzToken::Literal(data[pos]));
                mf.advance();
            }
        }
    }

    mf.take_chain()
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
    fn greedy_unique_data_all_literals() {
        let data: Vec<u8> = (0..=255).collect();
        let tokens = parse(&data, &test_config());
        assert_eq!(tokens.len(), 256);
        assert!(tokens.iter().all(|t| t.is_literal()));
        validate_tokens(&tokens, &data).unwrap();
    }

    #[test]
    fn greedy_simple_repeat() {
        let data = b"ABCDABCD";
        let tokens = parse(data, &test_config());
        validate_tokens(&tokens, data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.iter().any(|t| t.is_match()));
    }

    #[test]
    fn greedy_long_repeated_pattern() {
        let pattern = b"Hello, World! ";
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(pattern);
        }
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.len() < data.len() / 2);
    }

    #[test]
    fn greedy_short_data() {
        let data = b"AB";
        let tokens = parse(data, &test_config());
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|t| t.is_literal()));
    }

    #[test]
    fn greedy_all_zeros() {
        let data = vec![0u8; 5000];
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        let match_bytes: u32 = tokens
            .iter()
            .filter_map(|t| {
                if let LzToken::Match { length, .. } = t {
                    Some(*length)
                } else {
                    None
                }
            })
            .sum();
        assert!(
            match_bytes as usize > data.len() / 2,
            "most bytes should be covered by matches"
        );
    }

    #[test]
    fn greedy_overlapping_match() {
        let data = vec![b'A'; 100];
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn greedy_stress_mixed_data() {
        let mut data = Vec::new();
        for i in 0u8..20 {
            for _ in 0..10 {
                data.extend_from_slice(b"PATTERN");
            }
            for j in 0..50u8 {
                data.push(i.wrapping_mul(50).wrapping_add(j));
            }
        }
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn greedy_single_byte() {
        let data = b"X";
        let tokens = parse(data, &test_config());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], LzToken::Literal(b'X'));
    }

    #[test]
    fn greedy_exactly_min_match() {
        let data = b"ABCDABCD";
        let tokens = parse(data, &test_config());
        validate_tokens(&tokens, data).unwrap();
        // Should find the 4-byte match (exactly MIN_MATCH_LEN)
        let has_match = tokens
            .iter()
            .any(|t| matches!(t, LzToken::Match { length, .. } if *length >= 4));
        assert!(has_match, "should find 4-byte match");
    }
}
