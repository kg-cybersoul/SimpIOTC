//! # Parallel Block Compression
//!
//! Splits input data into independent blocks and compresses/decompresses them
//! in parallel using rayon. Each block goes through the full pipeline:
//!
//! **Compress**: preprocess → parse → entropy encode → CRC32
//! **Decompress**: CRC32 verify → entropy decode → replay tokens → depreprocess
//!
//! The output is a self-contained frame:
//! ```text
//! [FrameHeader 25 bytes]
//! [BlockHeader₀ 12 bytes] [payload₀]
//! [BlockHeader₁ 12 bytes] [payload₁]
//! ...
//! [SHA-256 digest 32 bytes]   (optional)
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::cell::RefCell;
#[cfg(feature = "parallel")]
use std::sync::Arc;

use crate::entropy;
use crate::parser;
use crate::preprocessor::{self, stride as stride_transform, PreprocessorConfig};
use crate::workspace::{DecodeWorkspace, EncodeWorkspace, PreprocessedDataRef};
use crate::{
    BlockHeader, CompressionConfig, CompressorError, DataType, FrameFlags, FrameHeader, Result,
    SeekTable, FRAME_VERSION,
};

use crc32fast::Hasher as CrcHasher;
use sha2::{Digest, Sha256};

#[cfg(feature = "parallel")]
thread_local! {
    static DECODE_WS: RefCell<DecodeWorkspace> = RefCell::new(DecodeWorkspace::new());
    static ENCODE_WS: RefCell<EncodeWorkspace> = RefCell::new(EncodeWorkspace::new());
}

/// A single compressed block, ready to be serialized.
struct CompressedBlock {
    header: BlockHeader,
    payload: Vec<u8>,
}

/// Compress a single chunk through the full pipeline using an explicit workspace.
///
/// Pipeline: stride transpose → preprocess → LZ77 parse → entropy encode → CRC32.
fn compress_one_block(
    chunk: &[u8],
    config: &CompressionConfig,
    data_type: DataType,
    stride_val: usize,
    ws: &mut EncodeWorkspace,
) -> Result<CompressedBlock> {
    let EncodeWorkspace {
        ref mut preprocess,
        ref mut parser,
        ref mut entropy,
        ref mut stride_buf,
    } = *ws;

    // 0. Stride transposition (if configured) — cache-aware tiled transpose
    let preprocess_input = if stride_val > 0 {
        stride_transform::transpose_into(chunk, stride_val, stride_buf);
        &stride_buf[..]
    } else {
        chunk
    };

    // 1. Preprocess — data lands in preprocess.output
    let pp_config = PreprocessorConfig {
        data_type: Some(data_type),
        double_delta: true,
    };
    preprocessor::preprocess_into(preprocess_input, &pp_config, preprocess)?;

    // 2. LZ77 Parse — tokens land in parser.tokens
    parser::parse_into(&preprocess.output, config, parser)?;

    // 3. Entropy Encode — only the payload Vec<u8> is a fresh allocation
    let (payload, _cost_model) = entropy::encode_tokens_into(&parser.tokens, entropy)?;

    // 4. CRC32 of compressed payload
    let mut hasher = CrcHasher::new();
    hasher.update(&payload);
    let crc32 = hasher.finalize();

    Ok(CompressedBlock {
        header: BlockHeader {
            compressed_size: payload.len() as u32,
            original_size: chunk.len() as u32,
            crc32,
        },
        payload,
    })
}

/// Compress an entire input byte slice into a framed, block-parallel stream.
///
/// Each block is independently preprocessed, parsed, and entropy-coded.
/// With the `parallel` feature (default), blocks are processed in parallel
/// via rayon's work-stealing pool. Without it, blocks are processed sequentially
/// using a single reusable workspace.
pub fn compress(data: &[u8], config: &CompressionConfig) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Err(CompressorError::EmptyInput);
    }

    // Determine data type: explicit → auto-detect → Raw
    let data_type = match config.data_type {
        Some(dt) => dt,
        None => preprocessor::auto_detect(data),
    };

    let elem_size = data_type.element_size();
    let stride_val = config.stride.map(|s| s as usize).unwrap_or(0);

    // Validate total alignment
    if data.len() % elem_size != 0 {
        return Err(CompressorError::DataTypeMismatch {
            element_size: elem_size,
            buffer_len: data.len(),
        });
    }

    // Validate stride alignment
    if stride_val > 0 && data.len() % stride_val != 0 {
        return Err(CompressorError::StrideMismatch {
            stride: stride_val,
            buffer_len: data.len(),
        });
    }

    // Align block size to element boundaries AND stride
    let align = if stride_val > 1 {
        stride_val.max(elem_size)
    } else {
        elem_size
    };
    let block_size = if align > 1 {
        (config.block_size / align) * align
    } else {
        config.block_size
    };
    let block_size = block_size.max(align);
    let block_count = data.len().div_ceil(block_size);

    // Guard against silent truncation — FrameHeader stores these as u32
    if block_size > u32::MAX as usize {
        return Err(CompressorError::InvalidBlockSize {
            size: block_size,
            reason: "block_size exceeds u32::MAX",
        });
    }
    if block_count > u32::MAX as usize {
        return Err(CompressorError::InvalidBlockSize {
            size: block_count,
            reason: "block_count exceeds u32::MAX (input too large for block size)",
        });
    }

    let chunks: Vec<&[u8]> = data.chunks(block_size).collect();

    // ── Block compression dispatch ───────────────────────────────────
    #[cfg(feature = "parallel")]
    let results: Vec<Result<CompressedBlock>> = {
        // NOTE: num_threads > 0 creates a NEW rayon ThreadPool per compress() call.
        // For server/loop usage, prefer num_threads = 0 to reuse the global pool.
        let pool = if config.num_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build()
                    .map_err(|e| CompressorError::Io(std::io::Error::other(e.to_string())))?,
            )
        } else {
            None
        };
        let config_arc = Arc::new(config.clone());
        let compress_block = |chunk: &[u8]| -> Result<CompressedBlock> {
            ENCODE_WS.with(|cell| {
                let mut ws = cell.borrow_mut();
                compress_one_block(chunk, &config_arc, data_type, stride_val, &mut ws)
            })
        };
        if let Some(pool) = pool {
            pool.install(|| chunks.par_iter().map(|c| compress_block(c)).collect())
        } else {
            chunks.par_iter().map(|c| compress_block(c)).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<Result<CompressedBlock>> = {
        let mut ws = EncodeWorkspace::new();
        chunks
            .iter()
            .map(|chunk| compress_one_block(chunk, config, data_type, stride_val, &mut ws))
            .collect()
    };

    // Propagate errors upfront and collect blocks
    let blocks: Vec<CompressedBlock> = {
        let mut v = Vec::with_capacity(block_count);
        for result in results {
            v.push(result?);
        }
        v
    };

    // Assemble output frame
    let flags = FrameFlags {
        data_type,
        parser_mode: config.parser_mode,
        has_content_checksum: config.store_checksum,
        has_repcodes: true,
        has_seek_table: config.store_seek_table,
    };
    let frame_header = FrameHeader {
        version: FRAME_VERSION,
        flags,
        block_size: block_size as u32,
        original_size: data.len() as u64,
        block_count: block_count as u32,
        stride: config.stride.unwrap_or(0),
    };

    let mut output = Vec::new();
    output.extend_from_slice(&frame_header.to_bytes());

    // Write seek table if configured
    if config.store_seek_table {
        let st_size = SeekTable::serialized_size(block_count as u32);
        let mut cursor = (FrameHeader::SERIALIZED_SIZE + st_size) as u64;
        let mut entries = Vec::with_capacity(block_count);
        for block in &blocks {
            entries.push(cursor);
            cursor += (BlockHeader::SERIALIZED_SIZE + block.payload.len()) as u64;
        }
        // crc32 field unused on serialization — to_bytes() computes it fresh
        let seek_table = SeekTable { entries, crc32: 0 };
        output.extend_from_slice(&seek_table.to_bytes());
    }

    for block in &blocks {
        output.extend_from_slice(&block.header.to_bytes());
        output.extend_from_slice(&block.payload);
    }

    // Optional SHA-256 content checksum over original data
    if config.store_checksum {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let digest = hasher.finalize();
        output.extend_from_slice(&digest);
    }

    Ok(output)
}

/// Decode a single compressed block payload into its original bytes.
///
/// Pipeline: CRC32 verify -> entropy decode -> replay tokens -> depreprocess -> unstride.
/// Used by both `decompress()` (via thread-local workspace) and `SeekableReader` (owned workspace).
pub(crate) fn decode_block_payload(
    payload: &[u8],
    bh: &BlockHeader,
    data_type: DataType,
    stride_val: usize,
    ws: &mut DecodeWorkspace,
) -> Result<Vec<u8>> {
    // Verify CRC32
    let mut hasher = CrcHasher::new();
    hasher.update(payload);
    let actual_crc32 = hasher.finalize();
    if actual_crc32 != bh.crc32 {
        return Err(CompressorError::ChecksumMismatch {
            expected: bh.crc32,
            computed: actual_crc32,
        });
    }

    // Destructure workspace for split borrows
    let DecodeWorkspace {
        entropy: ref mut entropy_scratch,
        ref mut replay,
        ref mut preprocess,
        ref mut stride_buf,
    } = *ws;

    // Entropy decode -> tokens left in entropy_scratch.tokens
    entropy::decode_tokens_into(payload, entropy_scratch)?;

    // Replay tokens -> preprocessed bytes left in replay.output
    parser::replay_tokens_into(&entropy_scratch.tokens, &mut replay.output);

    // Inverse preprocess -> original bytes left in preprocess.output
    let elem_size = data_type.element_size();
    let element_count = bh.original_size as u64 / elem_size as u64;

    // Validate element_count is plausible
    let max_original = (replay.output.len() as u64).saturating_mul(256);
    if element_count.saturating_mul(elem_size as u64) > max_original {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "block claims {} elements ({}B original) but preprocessed data is only {}B",
                element_count,
                element_count.saturating_mul(elem_size as u64),
                replay.output.len()
            ),
        });
    }

    let preprocessed_ref = PreprocessedDataRef {
        data_type,
        element_count,
        data: &replay.output,
    };
    preprocessor::depreprocess_into(&preprocessed_ref, preprocess)?;

    // Stride un-transposition (if present)
    let final_data = if stride_val > 0 {
        stride_transform::untranspose_into(&preprocess.output, stride_val, stride_buf);
        &stride_buf[..]
    } else {
        &preprocess.output[..]
    };

    // The only irreducible allocation: copy output to return to caller.
    let mut result = Vec::with_capacity(final_data.len());
    result.extend_from_slice(final_data);
    Ok(result)
}

/// Decompress a framed stream back into the original data.
///
/// Reads the frame header, locates all blocks, decompresses them in parallel,
/// and verifies CRC32 per block and SHA-256 over the full output.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < FrameHeader::SERIALIZED_SIZE {
        return Err(CompressorError::BufferUnderflow {
            needed: FrameHeader::SERIALIZED_SIZE,
            available: data.len(),
        });
    }

    let frame_header = FrameHeader::from_bytes(&data[..FrameHeader::SERIALIZED_SIZE])?;
    let data_type = frame_header.flags.data_type;
    let stride_val = frame_header.stride as usize;
    let mut pos = FrameHeader::SERIALIZED_SIZE;

    // Skip and verify seek table if present
    if frame_header.flags.has_seek_table {
        let st_size = SeekTable::serialized_size(frame_header.block_count);
        if pos + st_size > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: st_size,
                available: data.len() - pos,
            });
        }
        let _seek_table =
            SeekTable::from_bytes(&data[pos..pos + st_size], frame_header.block_count)?;
        pos += st_size;
    }

    // Validate block_count is plausible for the data we have — each block needs at
    // least its header, so block_count can't exceed the remaining bytes / header size.
    let remaining = data.len() - pos;
    let max_blocks = remaining / BlockHeader::SERIALIZED_SIZE;
    if frame_header.block_count as usize > max_blocks {
        return Err(CompressorError::CorruptedBlock {
            offset: 0,
            detail: format!(
                "frame claims {} blocks but only {} bytes remain (max {} blocks)",
                frame_header.block_count, remaining, max_blocks
            ),
        });
    }

    // Phase 1: Sequential scan to locate all block boundaries
    let mut block_infos: Vec<(BlockHeader, usize)> =
        Vec::with_capacity(frame_header.block_count as usize);

    for _ in 0..frame_header.block_count {
        if pos + BlockHeader::SERIALIZED_SIZE > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: BlockHeader::SERIALIZED_SIZE,
                available: data.len() - pos,
            });
        }
        let bh = BlockHeader::from_bytes(&data[pos..pos + BlockHeader::SERIALIZED_SIZE])?;
        pos += BlockHeader::SERIALIZED_SIZE;

        let payload_start = pos;
        if pos + bh.compressed_size as usize > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: bh.compressed_size as usize,
                available: data.len() - pos,
            });
        }
        pos += bh.compressed_size as usize;

        block_infos.push((bh, payload_start));
    }

    // Phase 2: Decode all blocks (workspace-aware, zero-alloc after warmup)
    // Skip parallel dispatch for single-block frames — overhead exceeds benefit.
    #[cfg(feature = "parallel")]
    let results: Vec<Result<Vec<u8>>> = {
        let decode_block = |(bh, payload_start): &(BlockHeader, usize)| -> Result<Vec<u8>> {
            DECODE_WS.with(|cell| {
                let mut ws = cell.borrow_mut();
                let payload = &data[*payload_start..*payload_start + bh.compressed_size as usize];
                decode_block_payload(payload, bh, data_type, stride_val, &mut ws)
            })
        };
        if block_infos.len() > 1 {
            block_infos.par_iter().map(decode_block).collect()
        } else {
            block_infos.iter().map(decode_block).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<Result<Vec<u8>>> = {
        let mut ws = DecodeWorkspace::new();
        block_infos
            .iter()
            .map(|(bh, payload_start)| {
                let payload = &data[*payload_start..*payload_start + bh.compressed_size as usize];
                decode_block_payload(payload, bh, data_type, stride_val, &mut ws)
            })
            .collect()
    };

    // Assemble output in block order.
    // Cap pre-allocation to prevent OOM from crafted headers claiming huge original_size.
    // Valid data will still produce correct output — the Vec just grows incrementally.
    let alloc_hint = std::cmp::min(
        frame_header.original_size as usize,
        data.len().saturating_mul(64),
    );
    let mut output = Vec::with_capacity(alloc_hint);
    for result in results {
        output.extend_from_slice(&result?);
    }

    // Verify SHA-256 if present
    if frame_header.flags.has_content_checksum {
        if pos + 32 > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: 32,
                available: data.len() - pos,
            });
        }
        let expected_digest = &data[pos..pos + 32];
        let mut hasher = Sha256::new();
        hasher.update(&output);
        let actual_digest = hasher.finalize();
        if expected_digest != actual_digest.as_slice() {
            return Err(CompressorError::IntegrityCheckFailed);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompressionConfig, DataType, ParserMode};

    #[test]
    fn roundtrip_raw() {
        let data = b"Hello, parallel world! This is a test. Hello, parallel world! This is a test.";
        let config = CompressionConfig {
            block_size: 32,
            store_checksum: true,
            data_type: Some(DataType::Raw),
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn roundtrip_i32_delta() {
        let mut data = Vec::new();
        let mut val = 1000i32;
        for _ in 0..200 {
            data.extend_from_slice(&val.to_le_bytes());
            val += 15;
        }

        let config = CompressionConfig {
            data_type: Some(DataType::IntegerI32),
            block_size: 256,
            ..CompressionConfig::balanced()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_f64_gorilla() {
        let values: Vec<f64> = (0..100).map(|i| 20.0 + 0.1 * (i as f64).sin()).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let config = CompressionConfig {
            data_type: Some(DataType::Float64),
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_multiblock() {
        let pattern = b"ABCDEFGHIJKLMNOP";
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(pattern);
        }

        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 512,
            store_checksum: true,
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);

        // Should have multiple blocks
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        assert!(
            frame.block_count > 1,
            "expected multiple blocks, got {}",
            frame.block_count
        );
    }

    #[test]
    fn roundtrip_all_parser_modes() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        for mode in [ParserMode::Greedy, ParserMode::Lazy, ParserMode::Optimal] {
            let config = CompressionConfig {
                parser_mode: mode,
                data_type: Some(DataType::Raw),
                block_size: 256,
                ..Default::default()
            };
            let compressed = compress(data, &config).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(&decompressed[..], &data[..], "failed for {:?}", mode);
        }
    }

    #[test]
    fn roundtrip_no_checksum() {
        let data = b"No checksum test data here. No checksum test data here.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            store_checksum: false,
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn roundtrip_with_checksum() {
        let data = b"SHA-256 verified data. SHA-256 verified data. SHA-256 verified data.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            store_checksum: true,
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn compress_empty_fails() {
        let config = CompressionConfig::fast();
        assert!(matches!(
            compress(&[], &config),
            Err(CompressorError::EmptyInput)
        ));
    }

    #[test]
    fn decompress_corrupted_crc_fails() {
        let data = b"Test data for CRC corruption. Test data for CRC corruption.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            store_checksum: false,
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let mut compressed = compress(data, &config).unwrap();

        // Corrupt a byte in the payload (after frame header + block header)
        let payload_offset = FrameHeader::SERIALIZED_SIZE + BlockHeader::SERIALIZED_SIZE + 1;
        if payload_offset < compressed.len() {
            compressed[payload_offset] ^= 0xFF;
        }

        let result = decompress(&compressed);
        assert!(result.is_err(), "corrupted CRC should fail");
    }

    #[test]
    fn decompress_corrupted_sha256_fails() {
        let data = b"SHA-256 integrity test data. SHA-256 integrity test data.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            store_checksum: true,
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let mut compressed = compress(data, &config).unwrap();

        // Corrupt the SHA-256 digest (last 32 bytes)
        let digest_start = compressed.len() - 32;
        compressed[digest_start] ^= 0xFF;

        let result = decompress(&compressed);
        assert!(
            matches!(result, Err(CompressorError::IntegrityCheckFailed)),
            "corrupted SHA-256 should fail with IntegrityCheckFailed"
        );
    }

    #[test]
    fn roundtrip_single_block() {
        let data = b"Small data that fits in one block";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 1024 * 1024, // 1MB — larger than data
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        assert_eq!(frame.block_count, 1);

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn roundtrip_custom_threads() {
        let data = b"Custom thread count test. Custom thread count test.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            num_threads: 2,
            block_size: 16,
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn decompress_truncated_header_fails() {
        let short = [0u8; 10];
        assert!(matches!(
            decompress(&short),
            Err(CompressorError::BufferUnderflow { .. })
        ));
    }

    #[test]
    fn alignment_validation() {
        // 7 bytes of "i64" data — not aligned to 8
        let data = vec![0u8; 7];
        let config = CompressionConfig {
            data_type: Some(DataType::IntegerI64),
            ..CompressionConfig::fast()
        };
        assert!(matches!(
            compress(&data, &config),
            Err(CompressorError::DataTypeMismatch { .. })
        ));
    }

    #[test]
    fn compression_actually_compresses() {
        let mut data = Vec::new();
        for _ in 0..1000 {
            data.extend_from_slice(b"REPEATED PATTERN ");
        }
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 2048,
            ..CompressionConfig::balanced()
        };
        let compressed = compress(&data, &config).unwrap();
        assert!(
            compressed.len() < data.len() / 2,
            "expected significant compression: {} -> {}",
            data.len(),
            compressed.len()
        );

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_stride_12() {
        // 12-byte struct × 500 records = 6000 bytes
        let mut data = Vec::with_capacity(6000);
        for i in 0u32..500 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(20.0f32 + 0.01 * i as f32).to_le_bytes());
            data.extend_from_slice(&(50.0f32 + 0.005 * i as f32).to_le_bytes());
        }
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            stride: Some(12),
            block_size: 2048,
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_stride_no_stride_same_data() {
        // Ensure data without stride still works after header format change.
        let data = b"No stride here. No stride here. Repeat for matches.";
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            stride: None,
            block_size: 256,
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn roundtrip_with_seek_table() {
        let data = b"Hello, seek table world! This is a test. Hello, seek table world!";
        let config = CompressionConfig {
            block_size: 32,
            store_checksum: true,
            store_seek_table: true,
            data_type: Some(DataType::Raw),
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        assert!(frame.flags.has_seek_table);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn roundtrip_without_seek_table() {
        let data = b"No seek table here. No seek table here. Repeat for matches.";
        let config = CompressionConfig {
            block_size: 32,
            store_checksum: true,
            store_seek_table: false,
            data_type: Some(DataType::Raw),
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        assert!(!frame.flags.has_seek_table);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn seek_table_offsets_correct() {
        let pattern = b"ABCDEFGHIJKLMNOP";
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(pattern);
        }
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 512,
            store_checksum: false,
            store_seek_table: true,
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        assert!(frame.flags.has_seek_table);

        // Parse the seek table and verify each entry points to a valid BlockHeader
        let st_size = SeekTable::serialized_size(frame.block_count);
        let st = SeekTable::from_bytes(
            &compressed[FrameHeader::SERIALIZED_SIZE..FrameHeader::SERIALIZED_SIZE + st_size],
            frame.block_count,
        )
        .unwrap();

        assert_eq!(st.entries.len(), frame.block_count as usize);
        for (i, &offset) in st.entries.iter().enumerate() {
            let off = offset as usize;
            assert!(
                off + BlockHeader::SERIALIZED_SIZE <= compressed.len(),
                "block {} offset {} out of bounds",
                i,
                off
            );
            let bh = BlockHeader::from_bytes(&compressed[off..off + BlockHeader::SERIALIZED_SIZE])
                .unwrap();
            assert!(
                bh.compressed_size > 0,
                "block {} has zero compressed size",
                i
            );
            assert!(bh.original_size > 0, "block {} has zero original size", i);
        }
    }

    #[test]
    fn decode_block_payload_equivalence() {
        // Verify that the factored-out decode_block_payload produces the same
        // result as the full parallel decompress pipeline.
        let data = b"Test data for equivalence check. Test data for equivalence check.";
        let config = CompressionConfig {
            block_size: 256,
            store_checksum: false,
            store_seek_table: true,
            data_type: Some(DataType::Raw),
            ..CompressionConfig::fast()
        };
        let compressed = compress(data, &config).unwrap();

        // Full pipeline decompress
        let full_result = decompress(&compressed).unwrap();
        assert_eq!(&full_result[..], &data[..]);

        // Manual decode via decode_block_payload
        let frame = FrameHeader::from_bytes(&compressed).unwrap();
        let st_size = SeekTable::serialized_size(frame.block_count);
        let st = SeekTable::from_bytes(
            &compressed[FrameHeader::SERIALIZED_SIZE..FrameHeader::SERIALIZED_SIZE + st_size],
            frame.block_count,
        )
        .unwrap();

        let mut ws = crate::workspace::DecodeWorkspace::new();
        let mut manual_result = Vec::new();
        for &offset in &st.entries {
            let off = offset as usize;
            let bh = BlockHeader::from_bytes(&compressed[off..off + BlockHeader::SERIALIZED_SIZE])
                .unwrap();
            let payload_start = off + BlockHeader::SERIALIZED_SIZE;
            let payload = &compressed[payload_start..payload_start + bh.compressed_size as usize];
            let block = decode_block_payload(
                payload,
                &bh,
                frame.flags.data_type,
                frame.stride as usize,
                &mut ws,
            )
            .unwrap();
            manual_result.extend_from_slice(&block);
        }
        assert_eq!(full_result, manual_result);
    }
}
