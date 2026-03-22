//! # LZ77 Parser
//!
//! Converts raw match candidates from the match finder into an optimal sequence
//! of LZ77 tokens (literals and matches). Three strategies are available:
//!
//! - **Greedy**: Always takes the best match at the current position. Fastest.
//! - **Lazy**: Looks one position ahead before committing. Balanced.
//! - **Optimal**: Dynamic programming over all possible parse decisions using
//!   FSE bit costs. Slowest but highest compression ratio.

pub mod greedy;
pub mod lazy;
pub mod optimal;

use crate::match_finder::{HashChain, MatchFinderConfig};
use crate::workspace::ParserScratch;
use crate::{CompressionConfig, CompressorError, LzToken, ParserMode, Result, MIN_MATCH_LEN};

/// Parse input data into an LZ77 token stream using the configured strategy.
///
/// Creates a match finder over the data, scans through it deciding at each
/// position whether to emit a literal or a match, and returns the token stream
/// that feeds directly into the entropy coder.
pub fn parse(data: &[u8], config: &CompressionConfig) -> Result<Vec<LzToken>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let mf_config = MatchFinderConfig {
        max_chain_depth: config.max_chain_depth,
        window_size: config.window_size,
        ..Default::default()
    };

    let tokens = match config.parser_mode {
        ParserMode::Greedy => greedy::parse(data, &mf_config),
        ParserMode::Lazy => lazy::parse(data, &mf_config),
        ParserMode::Optimal => optimal::parse(data, &mf_config),
    };

    Ok(tokens)
}

/// Workspace-aware parse. Reuses hash chain, tokens, and DP arrays from scratch.
/// Results are left in `scratch.tokens`.
pub fn parse_into(
    data: &[u8],
    config: &CompressionConfig,
    scratch: &mut ParserScratch,
) -> Result<()> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    let mf_config = MatchFinderConfig {
        max_chain_depth: config.max_chain_depth,
        window_size: config.window_size,
        ..Default::default()
    };

    // Take chain from scratch — first call creates, subsequent calls reuse.
    let chain = scratch
        .chain
        .take()
        .unwrap_or_else(|| HashChain::new(mf_config.window_size as usize));

    let chain = match config.parser_mode {
        ParserMode::Greedy => greedy::parse_into(data, &mf_config, chain, &mut scratch.tokens),
        ParserMode::Lazy => lazy::parse_into(data, &mf_config, chain, &mut scratch.tokens),
        ParserMode::Optimal => {
            // Optimal takes the full scratch (uses cost, decision, etc.)
            // But we already took the chain out — put it back before calling.
            scratch.chain = Some(chain);
            optimal::parse_into(data, &mf_config, scratch);
            // Chain is back in scratch.chain after optimal::parse_into.
            return Ok(());
        }
    };

    scratch.chain = Some(chain);
    Ok(())
}

/// Replay a token stream to reconstruct the original data.
///
/// This is the inverse of parsing: it "decodes" the token stream back into
/// a byte buffer by copying literals and resolving back-references.
pub fn replay_tokens(tokens: &[LzToken]) -> Vec<u8> {
    // Pre-compute output size to avoid reallocations.
    let output_size: usize = tokens
        .iter()
        .map(|t| match t {
            LzToken::Literal(_) => 1,
            LzToken::Match { length, .. } => *length as usize,
        })
        .sum();

    let mut output = Vec::with_capacity(output_size);
    for token in tokens {
        match token {
            LzToken::Literal(b) => output.push(*b),
            LzToken::Match { offset, length } => {
                let start = output.len() - *offset as usize;
                let len = *length as usize;
                if *offset as usize >= len {
                    // Non-overlapping: bulk memcpy
                    output.extend_from_within(start..start + len);
                } else {
                    // Overlapping (offset < length): LZ77 repeat semantics
                    for i in 0..len {
                        output.push(output[start + i]);
                    }
                }
            }
        }
    }
    output
}

/// Workspace-aware variant of `replay_tokens`. Clears and reuses the provided
/// output buffer instead of allocating. After the first block, this makes zero
/// heap allocations for same-or-smaller token streams.
pub fn replay_tokens_into(tokens: &[LzToken], output: &mut Vec<u8>) {
    output.clear();

    // Pre-compute output size to avoid reallocations.
    let output_size: usize = tokens
        .iter()
        .map(|t| match t {
            LzToken::Literal(_) => 1,
            LzToken::Match { length, .. } => *length as usize,
        })
        .sum();

    output.reserve(output_size);
    for token in tokens {
        match token {
            LzToken::Literal(b) => output.push(*b),
            LzToken::Match { offset, length } => {
                let start = output.len() - *offset as usize;
                let len = *length as usize;
                if *offset as usize >= len {
                    // Non-overlapping: bulk memcpy
                    output.extend_from_within(start..start + len);
                } else {
                    // Overlapping (offset < length): LZ77 repeat semantics
                    for i in 0..len {
                        output.push(output[start + i]);
                    }
                }
            }
        }
    }
}

/// Validate that a token stream correctly represents the source data.
///
/// Checks every literal against the original byte, verifies match offsets
/// point within already-decoded data, match lengths meet the minimum, match
/// content matches, and total decoded length equals data length.
pub fn validate_tokens(tokens: &[LzToken], data: &[u8]) -> Result<()> {
    let mut pos = 0usize;

    for (i, token) in tokens.iter().enumerate() {
        match token {
            LzToken::Literal(b) => {
                if pos >= data.len() {
                    return Err(CompressorError::Preprocessor(format!(
                        "token {} (literal) overflows data at pos {}",
                        i, pos
                    )));
                }
                if *b != data[pos] {
                    return Err(CompressorError::Preprocessor(format!(
                        "literal mismatch at pos {}: expected 0x{:02X}, got 0x{:02X}",
                        pos, data[pos], b
                    )));
                }
                pos += 1;
            }
            LzToken::Match { offset, length } => {
                if *offset == 0 {
                    return Err(CompressorError::Preprocessor(format!(
                        "zero offset in match at token {}, pos {}",
                        i, pos
                    )));
                }
                if (*offset as usize) > pos {
                    return Err(CompressorError::Preprocessor(format!(
                        "match offset {} exceeds decoded length {} at token {}",
                        offset, pos, i
                    )));
                }
                if *length < MIN_MATCH_LEN {
                    return Err(CompressorError::Preprocessor(format!(
                        "match length {} below minimum {} at token {}, pos {}",
                        length, MIN_MATCH_LEN, i, pos
                    )));
                }
                let len = *length as usize;
                if pos + len > data.len() {
                    return Err(CompressorError::Preprocessor(format!(
                        "match at pos {} (len {}) overflows data (len {})",
                        pos,
                        len,
                        data.len()
                    )));
                }
                // Verify match content: data at source must equal data at dest
                let src_start = pos - *offset as usize;
                for j in 0..len {
                    if data[src_start + j] != data[pos + j] {
                        return Err(CompressorError::Preprocessor(format!(
                            "match content mismatch at pos {} offset {} byte {}",
                            pos, offset, j
                        )));
                    }
                }
                pos += len;
            }
        }
    }

    if pos != data.len() {
        return Err(CompressorError::Preprocessor(format!(
            "tokens cover {} bytes but data is {} bytes",
            pos,
            data.len()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompressionConfig;

    fn test_config(mode: ParserMode) -> CompressionConfig {
        CompressionConfig {
            parser_mode: mode,
            window_size: 65536,
            max_chain_depth: 64,
            ..Default::default()
        }
    }

    #[test]
    fn parse_empty_fails() {
        let config = test_config(ParserMode::Greedy);
        assert!(matches!(
            parse(&[], &config),
            Err(CompressorError::EmptyInput)
        ));
    }

    #[test]
    fn parse_all_modes_valid_tokens() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = test_config(mode);
            let tokens = parse(data, &config).unwrap();
            validate_tokens(&tokens, data).unwrap_or_else(|e| {
                panic!("validate_tokens failed for {:?}: {}", mode, e);
            });
            let replayed = replay_tokens(&tokens);
            assert_eq!(&replayed[..], &data[..], "replay mismatch for {:?}", mode);
        }
    }

    #[test]
    fn parse_all_modes_short_data() {
        let data = b"AB";
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = test_config(mode);
            let tokens = parse(data, &config).unwrap();
            assert_eq!(tokens.len(), 2);
            assert!(tokens.iter().all(|t| t.is_literal()));
            validate_tokens(&tokens, data).unwrap();
        }
    }

    #[test]
    fn parse_all_modes_repeated_pattern() {
        let pattern = b"ABCDEFGH";
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(pattern);
        }
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = test_config(mode);
            let tokens = parse(&data, &config).unwrap();
            validate_tokens(&tokens, &data).unwrap();
            assert_eq!(replay_tokens(&tokens), data);
            let match_count = tokens.iter().filter(|t| t.is_match()).count();
            assert!(
                match_count > 0,
                "{:?} should find matches in repeated data",
                mode
            );
        }
    }

    #[test]
    fn parse_roundtrip_through_entropy() {
        let data = b"Hello world! Hello world! Hello world! Goodbye world!";
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = test_config(mode);
            let tokens = parse(data, &config).unwrap();
            validate_tokens(&tokens, data).unwrap();

            let (encoded, _cost) = crate::entropy::encode_tokens(&tokens).unwrap();
            let decoded_tokens = crate::entropy::decode_tokens(&encoded).unwrap();
            assert_eq!(
                tokens, decoded_tokens,
                "entropy roundtrip failed for {:?}",
                mode
            );
        }
    }

    #[test]
    fn validate_tokens_catches_bad_literal() {
        let data = b"ABCD";
        let tokens = vec![
            LzToken::Literal(b'X'),
            LzToken::Literal(b'B'),
            LzToken::Literal(b'C'),
            LzToken::Literal(b'D'),
        ];
        assert!(validate_tokens(&tokens, data).is_err());
    }

    #[test]
    fn validate_tokens_catches_bad_offset() {
        let data = b"ABCDABCD";
        let tokens = vec![
            LzToken::Literal(b'A'),
            LzToken::Literal(b'B'),
            LzToken::Literal(b'C'),
            LzToken::Literal(b'D'),
            LzToken::Match {
                offset: 5,
                length: 4,
            },
        ];
        assert!(validate_tokens(&tokens, data).is_err());
    }

    #[test]
    fn validate_tokens_catches_length_mismatch() {
        let data = b"ABCD";
        let tokens = vec![
            LzToken::Literal(b'A'),
            LzToken::Literal(b'B'),
            LzToken::Literal(b'C'),
        ];
        assert!(validate_tokens(&tokens, data).is_err());
    }

    #[test]
    fn replay_tokens_overlapping_match() {
        let tokens = vec![
            LzToken::Literal(b'A'),
            LzToken::Match {
                offset: 1,
                length: 5,
            },
        ];
        let output = replay_tokens(&tokens);
        assert_eq!(output, b"AAAAAA");
    }

    #[test]
    fn all_modes_compress_repetitive_data() {
        let data = vec![0xABu8; 10000];
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = test_config(mode);
            let tokens = parse(&data, &config).unwrap();
            assert!(
                tokens.len() < 100,
                "{:?}: expected < 100 tokens for all-same-byte data, got {}",
                mode,
                tokens.len()
            );
        }
    }
}
