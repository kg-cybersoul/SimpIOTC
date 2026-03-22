//! Lazy LZ77 parser.
//!
//! Before committing to a match, checks one position ahead. If the next
//! position yields a strictly longer match, emits a literal at the current
//! position and takes the better match instead. This simple one-step
//! look-ahead often improves compression ratio with minimal speed cost.
//!
//! This is the default parser mode for balanced compression.

use crate::match_finder::{HashChain, Match, MatchFinder, MatchFinderConfig};
use crate::LzToken;

/// Returns true if `m2` is strictly better than `m1`.
///
/// "Better" means longer (covers more data per token).
/// On equal length, prefer shorter offset (cheaper to entropy-code).
#[inline]
fn is_better(m2: &Match, m1: &Match) -> bool {
    m2.length > m1.length || (m2.length == m1.length && m2.offset < m1.offset)
}

/// Parse data with lazy evaluation: look one position ahead before committing.
pub fn parse(data: &[u8], config: &MatchFinderConfig) -> Vec<LzToken> {
    let mut mf = MatchFinder::new(data, config.clone());
    let mut tokens = Vec::with_capacity(data.len() / 2);

    while !mf.is_done() {
        let pos = mf.pos();

        let m1 = match mf.find_best_match_at(pos) {
            Some(m) => m,
            None => {
                tokens.push(LzToken::Literal(data[pos]));
                mf.advance();
                continue;
            }
        };

        // Insert current position so pos+1 can reference it in the hash chain
        mf.insert_at(pos);

        // Look ahead one position
        let took_lookahead = if let Some(m2) = mf.find_best_match_at(pos + 1) {
            if is_better(&m2, &m1) {
                // Emit literal for current position, take the better match
                tokens.push(LzToken::Literal(data[pos]));
                tokens.push(LzToken::Match {
                    offset: m2.offset,
                    length: m2.length,
                });
                // advance() re-inserts pos (harmless duplicate) and moves to pos+1
                mf.advance();
                // advance_by inserts pos+1..pos+1+len and moves past the match
                mf.advance_by(m2.length as usize);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !took_lookahead {
            tokens.push(LzToken::Match {
                offset: m1.offset,
                length: m1.length,
            });
            mf.advance_by(m1.length as usize);
        }
    }

    tokens
}

/// Workspace-aware lazy parse. Reuses the hash chain and tokens Vec.
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

        let m1 = match mf.find_best_match_at(pos) {
            Some(m) => m,
            None => {
                tokens.push(LzToken::Literal(data[pos]));
                mf.advance();
                continue;
            }
        };

        mf.insert_at(pos);

        let took_lookahead = if let Some(m2) = mf.find_best_match_at(pos + 1) {
            if is_better(&m2, &m1) {
                tokens.push(LzToken::Literal(data[pos]));
                tokens.push(LzToken::Match {
                    offset: m2.offset,
                    length: m2.length,
                });
                mf.advance();
                mf.advance_by(m2.length as usize);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !took_lookahead {
            tokens.push(LzToken::Match {
                offset: m1.offset,
                length: m1.length,
            });
            mf.advance_by(m1.length as usize);
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
    fn lazy_unique_data_all_literals() {
        let data: Vec<u8> = (0..=255).collect();
        let tokens = parse(&data, &test_config());
        assert!(tokens.iter().all(|t| t.is_literal()));
        validate_tokens(&tokens, &data).unwrap();
    }

    #[test]
    fn lazy_simple_repeat() {
        let data = b"ABCDABCD";
        let tokens = parse(data, &test_config());
        validate_tokens(&tokens, data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.iter().any(|t| t.is_match()));
    }

    #[test]
    fn lazy_finds_better_match_ahead() {
        // At position where "XYZW" repeats (4-byte match), but pos+1 starts
        // "ABCDEFGHIJKL" repeat (12-byte match). Lazy should skip the short
        // match and take the longer one.
        let mut data = Vec::new();
        data.extend_from_slice(b"XYZW"); // 0..4
        data.extend_from_slice(b"ABCDEFGHIJKL"); // 4..16
        for i in 0..50u8 {
            data.push(i);
        }
        data.extend_from_slice(b"XABCDEFGHIJKL"); // 66..79
                                                  // At pos 66: "XYZW" matches offset 66 len 4
                                                  // At pos 67: "ABCDEFGHIJKL" matches offset 63 len 12 — strictly better
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn lazy_long_repeated_pattern() {
        let pattern = b"The quick brown fox ";
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
    fn lazy_improves_or_matches_greedy() {
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(b"ABCDEFABCDEF");
            data.extend_from_slice(b"XXXXYZ");
        }
        let config = test_config();
        let greedy_tokens = super::super::greedy::parse(&data, &config);
        let lazy_tokens = parse(&data, &config);
        validate_tokens(&greedy_tokens, &data).unwrap();
        validate_tokens(&lazy_tokens, &data).unwrap();

        // Both must reconstruct correctly
        assert_eq!(replay_tokens(&greedy_tokens), data);
        assert_eq!(replay_tokens(&lazy_tokens), data);
    }

    #[test]
    fn lazy_all_zeros() {
        let data = vec![0u8; 5000];
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
    }

    #[test]
    fn lazy_short_data() {
        let data = b"AB";
        let tokens = parse(data, &test_config());
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().all(|t| t.is_literal()));
    }

    #[test]
    fn lazy_stress_large() {
        let mut data = Vec::with_capacity(50_000);
        let segments = [b"Hello! " as &[u8], b"World! ", b"FooBar ", b"BazQux "];
        for i in 0..7000 {
            data.extend_from_slice(segments[i % 4]);
        }
        let tokens = parse(&data, &test_config());
        validate_tokens(&tokens, &data).unwrap();
        assert_eq!(replay_tokens(&tokens), data);
        assert!(tokens.len() < data.len() / 3);
    }

    #[test]
    fn lazy_is_better_logic() {
        let m1 = Match::new(10, 5);
        let m2_longer = Match::new(10, 6);
        let m2_shorter_offset = Match::new(5, 5);
        let m2_worse = Match::new(10, 4);

        assert!(is_better(&m2_longer, &m1));
        assert!(is_better(&m2_shorter_offset, &m1));
        assert!(!is_better(&m2_worse, &m1));
        assert!(!is_better(&m1, &m1)); // not strictly better
    }
}
