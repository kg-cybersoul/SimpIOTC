//! # Seekable Reader — O(1) Random Block Access
//!
//! Provides `SeekableReader`, which parses a compressed frame's seek table
//! and enables random access to individual blocks, block ranges, or arbitrary
//! byte ranges without decompressing the entire frame.
//!
//! Requires the frame to have been compressed with `store_seek_table = true`.

use crate::parallel::decode_block_payload;
use crate::schema::{NamedColumn, Schema};
use crate::workspace::DecodeWorkspace;
use crate::{BlockHeader, CompressorError, FrameHeader, Result, SeekTable};

/// A reader that provides O(1) random access to compressed blocks via the seek table.
///
/// Borrows the compressed frame data and owns a single-threaded `DecodeWorkspace`.
/// Use `&mut self` methods because the workspace is mutated during decompression.
pub struct SeekableReader<'a> {
    data: &'a [u8],
    header: FrameHeader,
    seek_table: SeekTable,
    workspace: DecodeWorkspace,
}

impl<'a> SeekableReader<'a> {
    /// Create a new `SeekableReader` from compressed frame data.
    ///
    /// Returns an error if the frame doesn't contain a seek table or is malformed.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < FrameHeader::SERIALIZED_SIZE {
            return Err(CompressorError::BufferUnderflow {
                needed: FrameHeader::SERIALIZED_SIZE,
                available: data.len(),
            });
        }

        let header = FrameHeader::from_bytes(&data[..FrameHeader::SERIALIZED_SIZE])?;

        if !header.flags.has_seek_table {
            return Err(CompressorError::CorruptedBlock {
                offset: 0,
                detail: "frame does not contain a seek table".into(),
            });
        }

        let st_size = SeekTable::serialized_size(header.block_count);
        let st_start = FrameHeader::SERIALIZED_SIZE;
        if st_start + st_size > data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: st_size,
                available: data.len() - st_start,
            });
        }

        let seek_table =
            SeekTable::from_bytes(&data[st_start..st_start + st_size], header.block_count)?;

        Ok(Self {
            data,
            header,
            seek_table,
            workspace: DecodeWorkspace::new(),
        })
    }

    /// Returns a reference to the frame header.
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns the number of blocks in the frame.
    pub fn block_count(&self) -> u32 {
        self.header.block_count
    }

    /// Returns the original (uncompressed) size of the frame.
    pub fn original_size(&self) -> u64 {
        self.header.original_size
    }

    /// Decompress a single block by index.
    pub fn decompress_block(&mut self, index: usize) -> Result<Vec<u8>> {
        if index >= self.header.block_count as usize {
            return Err(CompressorError::BlockIndexOutOfRange {
                index,
                block_count: self.header.block_count,
            });
        }

        let offset = self.seek_table.entries[index] as usize;
        if offset + BlockHeader::SERIALIZED_SIZE > self.data.len() {
            return Err(CompressorError::SeekTableOffsetOutOfBounds {
                index,
                offset: offset as u64,
                frame_size: self.data.len() as u64,
            });
        }

        let bh =
            BlockHeader::from_bytes(&self.data[offset..offset + BlockHeader::SERIALIZED_SIZE])?;
        let payload_start = offset + BlockHeader::SERIALIZED_SIZE;
        if payload_start + bh.compressed_size as usize > self.data.len() {
            return Err(CompressorError::BufferUnderflow {
                needed: bh.compressed_size as usize,
                available: self.data.len() - payload_start,
            });
        }

        let payload = &self.data[payload_start..payload_start + bh.compressed_size as usize];
        let data_type = self.header.flags.data_type;
        let stride_val = self.header.stride as usize;
        decode_block_payload(payload, &bh, data_type, stride_val, &mut self.workspace)
    }

    /// Decompress a contiguous range of blocks `[start, end)`.
    pub fn decompress_range(&mut self, start: usize, end: usize) -> Result<Vec<u8>> {
        if end > self.header.block_count as usize {
            return Err(CompressorError::BlockIndexOutOfRange {
                index: end.saturating_sub(1),
                block_count: self.header.block_count,
            });
        }
        if start >= end {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();
        for i in start..end {
            let block = self.decompress_block(i)?;
            output.extend_from_slice(&block);
        }
        Ok(output)
    }

    /// Decompress an arbitrary byte range `[byte_start, byte_end)` from the original data.
    ///
    /// Only decompresses the blocks that overlap the requested range, then trims.
    pub fn decompress_byte_range(&mut self, byte_start: u64, byte_end: u64) -> Result<Vec<u8>> {
        if byte_start >= byte_end {
            return Ok(Vec::new());
        }
        if byte_end > self.header.original_size {
            return Err(CompressorError::ByteRangeOutOfBounds {
                start: byte_start,
                end: byte_end,
                original_size: self.header.original_size,
            });
        }

        let block_size = self.header.block_size as u64;
        let start_block = (byte_start / block_size) as usize;
        let end_block = byte_end.div_ceil(block_size) as usize;
        let end_block = end_block.min(self.header.block_count as usize);

        let raw = self.decompress_range(start_block, end_block)?;

        // Trim to the requested byte range
        let range_base = start_block as u64 * block_size;
        let trim_start = (byte_start - range_base) as usize;
        let trim_len = (byte_end - byte_start) as usize;
        Ok(raw[trim_start..trim_start + trim_len].to_vec())
    }

    /// Decompress the entire frame (all blocks sequentially).
    pub fn decompress_all(&mut self) -> Result<Vec<u8>> {
        self.decompress_range(0, self.header.block_count as usize)
    }

    /// Decompress and extract typed columns from structured record data.
    ///
    /// The schema describes the field layout within each fixed-width record.
    /// The schema's total stride must match the frame's stride (if stride > 0),
    /// or the frame's stride must be 0 (user-asserted schema).
    ///
    /// Returns one `NamedColumn` per schema field, each containing contiguous
    /// typed values ready for zero-copy Arrow/Polars ingest via `np.frombuffer`.
    pub fn extract_columns(
        &mut self,
        schema: &Schema,
        start_block: usize,
        end_block: usize,
    ) -> Result<Vec<NamedColumn>> {
        schema.validate_frame_stride(self.header.stride)?;
        let data = self.decompress_range(start_block, end_block)?;
        schema.extract_columns(&data)
    }

    /// Extract typed columns from the entire frame.
    pub fn extract_all_columns(&mut self, schema: &Schema) -> Result<Vec<NamedColumn>> {
        self.extract_columns(schema, 0, self.header.block_count as usize)
    }

    /// Returns the stride value from the frame header.
    pub fn stride(&self) -> u16 {
        self.header.stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::{compress, decompress};
    use crate::schema::{ColumnType, Schema};
    use crate::{CompressionConfig, DataType};

    fn make_test_data() -> Vec<u8> {
        // 5 blocks of 1024 bytes each = 5120 bytes
        let mut data = Vec::with_capacity(5120);
        for i in 0u32..1280 {
            data.extend_from_slice(&(1000u32.wrapping_add(i)).to_le_bytes());
        }
        data
    }

    fn compress_with_seek_table(data: &[u8]) -> Vec<u8> {
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 1024,
            store_checksum: false,
            store_seek_table: true,
            ..CompressionConfig::fast()
        };
        compress(data, &config).unwrap()
    }

    fn compress_without_seek_table(data: &[u8]) -> Vec<u8> {
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 1024,
            store_checksum: false,
            store_seek_table: false,
            ..CompressionConfig::fast()
        };
        compress(data, &config).unwrap()
    }

    #[test]
    fn seekable_reader_requires_seek_table() {
        let data = make_test_data();
        let compressed = compress_without_seek_table(&data);
        let result = SeekableReader::new(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn seekable_decompress_each_block() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count() as usize;
        assert!(bc > 1, "expected multi-block frame");

        let mut concat = Vec::new();
        for i in 0..bc {
            let block = reader.decompress_block(i).unwrap();
            concat.extend_from_slice(&block);
        }
        assert_eq!(concat, data);
    }

    #[test]
    fn seekable_decompress_range() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count() as usize;

        // Decompress blocks 1..3 should match the equivalent slice
        if bc >= 3 {
            let range = reader.decompress_range(1, 3).unwrap();
            let block_size = reader.header().block_size as usize;
            assert_eq!(range, data[block_size..block_size * 3]);
        }
    }

    #[test]
    fn seekable_byte_range_exact_block() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bs = reader.header().block_size as u64;

        // Read exactly block 1
        let result = reader.decompress_byte_range(bs, bs * 2).unwrap();
        assert_eq!(result, &data[bs as usize..(bs * 2) as usize]);
    }

    #[test]
    fn seekable_byte_range_cross_block() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bs = reader.header().block_size as u64;

        // Read across block boundary: last 100 bytes of block 0 + first 100 of block 1
        let start = bs - 100;
        let end = bs + 100;
        let result = reader.decompress_byte_range(start, end).unwrap();
        assert_eq!(result, &data[start as usize..end as usize]);
    }

    #[test]
    fn seekable_byte_range_partial() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bs = reader.header().block_size as u64;

        // Mid-block-0 to mid-block-2
        let start = bs / 2;
        let end = bs * 2 + bs / 2;
        let end = end.min(reader.original_size());
        let result = reader.decompress_byte_range(start, end).unwrap();
        assert_eq!(result, &data[start as usize..end as usize]);
    }

    #[test]
    fn seekable_decompress_all_matches() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);

        // SeekableReader.decompress_all() should match parallel::decompress()
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let seekable_result = reader.decompress_all().unwrap();
        let parallel_result = decompress(&compressed).unwrap();
        assert_eq!(seekable_result, parallel_result);
        assert_eq!(seekable_result, data);
    }

    #[test]
    fn seekable_oob_error() {
        let data = make_test_data();
        let compressed = compress_with_seek_table(&data);
        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count();

        let result = reader.decompress_block(bc as usize);
        assert!(matches!(
            result,
            Err(CompressorError::BlockIndexOutOfRange { .. })
        ));

        let result = reader.decompress_block(bc as usize + 100);
        assert!(matches!(
            result,
            Err(CompressorError::BlockIndexOutOfRange { .. })
        ));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Column extraction tests
    // ═══════════════════════════════════════════════════════════════════════

    /// Build test struct data: N records of {seq: u32, temp: f32, humidity: f32} = stride 12.
    fn make_struct_data(n: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * 12);
        for i in 0..n as u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(20.0 + 0.01 * i as f32).to_le_bytes());
            data.extend_from_slice(&(50.0 + 0.005 * i as f32).to_le_bytes());
        }
        data
    }

    fn struct_schema() -> Schema {
        Schema::new(vec![
            ("seq".into(), ColumnType::U32),
            ("temp".into(), ColumnType::F32),
            ("humidity".into(), ColumnType::F32),
        ])
    }

    fn compress_structs(data: &[u8]) -> Vec<u8> {
        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 4096,
            store_checksum: false,
            store_seek_table: true,
            stride: Some(12),
            ..CompressionConfig::fast()
        };
        compress(data, &config).unwrap()
    }

    #[test]
    fn extract_columns_roundtrip_basic() {
        let data = make_struct_data(100);
        let compressed = compress_structs(&data);
        let schema = struct_schema();

        let mut reader = SeekableReader::new(&compressed).unwrap();
        assert_eq!(reader.stride(), 12);

        let columns = reader.extract_all_columns(&schema).unwrap();
        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0].name, "seq");
        assert_eq!(columns[1].name, "temp");
        assert_eq!(columns[2].name, "humidity");

        // Verify first and last seq values
        let first = u32::from_le_bytes(columns[0].data[..4].try_into().unwrap());
        let last = u32::from_le_bytes(
            columns[0].data[columns[0].data.len() - 4..]
                .try_into()
                .unwrap(),
        );
        assert_eq!(first, 0);
        assert_eq!(last, 99);

        // Verify first temp value
        let t0 = f32::from_le_bytes(columns[1].data[..4].try_into().unwrap());
        assert!((t0 - 20.0).abs() < 0.01);
    }

    #[test]
    fn extract_columns_matches_manual_parse() {
        let n = 500;
        let data = make_struct_data(n);
        let compressed = compress_structs(&data);
        let schema = struct_schema();

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let columns = reader.extract_all_columns(&schema).unwrap();

        // Parse manually from original data and compare
        for i in 0..n {
            let expected_seq = i as u32;
            let expected_temp = 20.0 + 0.01 * i as f32;
            let expected_hum = 50.0 + 0.005 * i as f32;

            let got_seq =
                u32::from_le_bytes(columns[0].data[i * 4..(i + 1) * 4].try_into().unwrap());
            let got_temp =
                f32::from_le_bytes(columns[1].data[i * 4..(i + 1) * 4].try_into().unwrap());
            let got_hum =
                f32::from_le_bytes(columns[2].data[i * 4..(i + 1) * 4].try_into().unwrap());

            assert_eq!(got_seq, expected_seq, "seq mismatch at record {}", i);
            assert!(
                (got_temp - expected_temp).abs() < 1e-5,
                "temp mismatch at record {}: got {}, expected {}",
                i,
                got_temp,
                expected_temp,
            );
            assert!(
                (got_hum - expected_hum).abs() < 1e-5,
                "humidity mismatch at record {}: got {}, expected {}",
                i,
                got_hum,
                expected_hum,
            );
        }
    }

    #[test]
    fn extract_columns_block_range() {
        // Large enough for multiple blocks (4096 byte blocks, 12 byte stride)
        let n = 2000; // 24000 bytes → ~5-6 blocks at 4096
        let data = make_struct_data(n);
        let compressed = compress_structs(&data);
        let schema = struct_schema();

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let bc = reader.block_count() as usize;
        assert!(bc >= 2, "need multi-block for this test, got {}", bc);

        // Extract from block 1 only
        let partial_cols = reader.extract_columns(&schema, 1, 2).unwrap();
        assert!(!partial_cols[0].data.is_empty());

        // The seq values should NOT start at 0 (they're from a later block)
        let first_seq = u32::from_le_bytes(partial_cols[0].data[..4].try_into().unwrap());
        assert!(first_seq > 0, "block 1 should not start at seq 0");
    }

    #[test]
    fn extract_columns_stride_mismatch_errors() {
        let data = make_struct_data(100);
        let compressed = compress_structs(&data);

        // Wrong schema: stride 16 != frame stride 12
        let bad_schema = Schema::new(vec![
            ("a".into(), ColumnType::I64),
            ("b".into(), ColumnType::I64),
        ]);

        let mut reader = SeekableReader::new(&compressed).unwrap();
        let result = reader.extract_all_columns(&bad_schema);
        assert!(result.is_err());
    }

    #[test]
    fn extract_columns_no_stride_frame() {
        // Compress without stride — extract_columns still works if schema is user-asserted
        let n = 100;
        let mut data = Vec::with_capacity(n * 8);
        for i in 0..n as u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(i as f32 * 2.0).to_le_bytes());
        }

        let config = CompressionConfig {
            data_type: Some(DataType::Raw),
            block_size: 4096,
            store_checksum: false,
            store_seek_table: true,
            stride: None, // no stride compression
            ..CompressionConfig::fast()
        };
        let compressed = compress(&data, &config).unwrap();

        let schema = Schema::new(vec![
            ("id".into(), ColumnType::U32),
            ("val".into(), ColumnType::F32),
        ]);

        let mut reader = SeekableReader::new(&compressed).unwrap();
        assert_eq!(reader.stride(), 0);

        // Should work — frame stride 0 means "user-asserted schema"
        let columns = reader.extract_all_columns(&schema).unwrap();
        let first_id = u32::from_le_bytes(columns[0].data[..4].try_into().unwrap());
        assert_eq!(first_id, 0);
    }
}
