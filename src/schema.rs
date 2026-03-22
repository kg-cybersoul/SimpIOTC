//! # Schema — Column Types and Extraction
//!
//! Provides a record schema for structured data: column names, types, and offsets.
//! `Schema::extract_columns()` de-interleaves row-major (AoS) byte streams into
//! per-column contiguous buffers suitable for Arrow/Polars zero-copy ingest.

use crate::{CompressorError, Result};

/// Primitive numeric type for a single column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl ColumnType {
    /// Byte width of this type.
    pub fn size(self) -> usize {
        match self {
            ColumnType::I8 | ColumnType::U8 => 1,
            ColumnType::I16 | ColumnType::U16 => 2,
            ColumnType::I32 | ColumnType::U32 | ColumnType::F32 => 4,
            ColumnType::I64 | ColumnType::U64 | ColumnType::F64 => 8,
        }
    }

    /// Parse from string (matching DataType display conventions).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "i8" => Some(ColumnType::I8),
            "i16" => Some(ColumnType::I16),
            "i32" => Some(ColumnType::I32),
            "i64" => Some(ColumnType::I64),
            "u8" => Some(ColumnType::U8),
            "u16" => Some(ColumnType::U16),
            "u32" => Some(ColumnType::U32),
            "u64" => Some(ColumnType::U64),
            "f32" => Some(ColumnType::F32),
            "f64" => Some(ColumnType::F64),
            _ => None,
        }
    }
}

impl std::fmt::Display for ColumnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnType::I8 => write!(f, "i8"),
            ColumnType::I16 => write!(f, "i16"),
            ColumnType::I32 => write!(f, "i32"),
            ColumnType::I64 => write!(f, "i64"),
            ColumnType::U8 => write!(f, "u8"),
            ColumnType::U16 => write!(f, "u16"),
            ColumnType::U32 => write!(f, "u32"),
            ColumnType::U64 => write!(f, "u64"),
            ColumnType::F32 => write!(f, "f32"),
            ColumnType::F64 => write!(f, "f64"),
        }
    }
}

/// A single column definition within a record schema.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Column name (used as dict key in Python API).
    pub name: String,
    /// Primitive type.
    pub col_type: ColumnType,
    /// Byte offset of this field within a single record.
    pub offset: usize,
}

/// A fixed-width record schema: ordered list of typed columns.
///
/// The schema maps a flat byte stream of interleaved records into per-column
/// contiguous arrays. The total column widths define the record stride.
#[derive(Debug, Clone)]
pub struct Schema {
    columns: Vec<ColumnDef>,
    stride: usize,
}

impl Schema {
    /// Create a schema from an ordered list of (name, type) pairs.
    ///
    /// Fields are packed sequentially with no padding. The stride equals the
    /// sum of all column sizes.
    pub fn new(columns: Vec<(String, ColumnType)>) -> Self {
        let mut offset = 0;
        let defs: Vec<ColumnDef> = columns
            .into_iter()
            .map(|(name, ct)| {
                let def = ColumnDef {
                    name,
                    col_type: ct,
                    offset,
                };
                offset += ct.size();
                def
            })
            .collect();
        Self {
            columns: defs,
            stride: offset,
        }
    }

    /// Record stride (total bytes per record).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Column definitions.
    pub fn columns(&self) -> &[ColumnDef] {
        &self.columns
    }

    /// Validate this schema against a frame's stride value.
    ///
    /// Returns `Ok(())` if the schema stride matches, or if the frame stride is 0
    /// (meaning no stride compression was used — schema is user-asserted).
    pub fn validate_frame_stride(&self, frame_stride: u16) -> Result<()> {
        if frame_stride > 0 && frame_stride as usize != self.stride {
            return Err(CompressorError::StrideMismatch {
                stride: frame_stride as usize,
                buffer_len: self.stride,
            });
        }
        Ok(())
    }

    /// Extract columns from row-major (AoS) byte data.
    ///
    /// Input: `[rec0_f0, rec0_f1, ..., rec1_f0, rec1_f1, ...]`
    /// Output: one `Vec<u8>` per column, each containing contiguous typed values.
    ///
    /// Processes one column at a time for write-side cache locality.
    pub fn extract_columns(&self, data: &[u8]) -> Result<Vec<NamedColumn>> {
        if data.is_empty() {
            return Ok(self
                .columns
                .iter()
                .map(|c| NamedColumn {
                    name: c.name.clone(),
                    col_type: c.col_type,
                    data: Vec::new(),
                })
                .collect());
        }

        if self.stride == 0 || data.len() % self.stride != 0 {
            return Err(CompressorError::StrideMismatch {
                stride: self.stride,
                buffer_len: data.len(),
            });
        }

        let num_records = data.len() / self.stride;

        let mut result: Vec<NamedColumn> = self
            .columns
            .iter()
            .map(|c| NamedColumn {
                name: c.name.clone(),
                col_type: c.col_type,
                data: vec![0u8; num_records * c.col_type.size()],
            })
            .collect();

        // Process one column at a time — sequential writes to each output buffer.
        for (col_idx, col_def) in self.columns.iter().enumerate() {
            let elem_size = col_def.col_type.size();
            let output = &mut result[col_idx].data;
            for record in 0..num_records {
                let src = record * self.stride + col_def.offset;
                let dst = record * elem_size;
                output[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
            }
        }

        Ok(result)
    }
}

/// A named column extracted from a record stream.
#[derive(Debug, Clone)]
pub struct NamedColumn {
    /// Column name.
    pub name: String,
    /// Primitive type.
    pub col_type: ColumnType,
    /// Contiguous column bytes (little-endian typed values).
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_stride_calculation() {
        let schema = Schema::new(vec![
            ("ts".into(), ColumnType::I64),
            ("temp".into(), ColumnType::F64),
            ("vib".into(), ColumnType::F32),
        ]);
        assert_eq!(schema.stride(), 20); // 8 + 8 + 4
        assert_eq!(schema.num_columns(), 3);
        assert_eq!(schema.columns()[0].offset, 0);
        assert_eq!(schema.columns()[1].offset, 8);
        assert_eq!(schema.columns()[2].offset, 16);
    }

    #[test]
    fn schema_stride_single_column() {
        let schema = Schema::new(vec![("x".into(), ColumnType::F32)]);
        assert_eq!(schema.stride(), 4);
        assert_eq!(schema.columns()[0].offset, 0);
    }

    #[test]
    fn extract_columns_basic() {
        // Schema: {a: u32, b: f32} = stride 8
        let schema = Schema::new(vec![
            ("a".into(), ColumnType::U32),
            ("b".into(), ColumnType::F32),
        ]);

        // 3 records: (1, 10.0), (2, 20.0), (3, 30.0)
        let mut data = Vec::new();
        for i in 1..=3u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(i as f32 * 10.0).to_le_bytes());
        }

        let columns = schema.extract_columns(&data).unwrap();
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].name, "a");
        assert_eq!(columns[1].name, "b");

        // Verify column a: [1, 2, 3] as u32 LE
        let a_vals: Vec<u32> = columns[0]
            .data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(a_vals, vec![1, 2, 3]);

        // Verify column b: [10.0, 20.0, 30.0] as f32 LE
        let b_vals: Vec<f32> = columns[1]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(b_vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn extract_columns_mixed_widths() {
        // Schema: {id: u8, value: i64, flag: u16} = stride 11
        let schema = Schema::new(vec![
            ("id".into(), ColumnType::U8),
            ("value".into(), ColumnType::I64),
            ("flag".into(), ColumnType::U16),
        ]);
        assert_eq!(schema.stride(), 11);

        let mut data = Vec::new();
        for i in 0..5u8 {
            data.push(i);
            data.extend_from_slice(&(i as i64 * 1000).to_le_bytes());
            data.extend_from_slice(&(i as u16 * 100).to_le_bytes());
        }

        let columns = schema.extract_columns(&data).unwrap();
        assert_eq!(columns.len(), 3);

        let ids: Vec<u8> = columns[0].data.clone();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);

        let values: Vec<i64> = columns[1]
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0, 1000, 2000, 3000, 4000]);

        let flags: Vec<u16> = columns[2]
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(flags, vec![0, 100, 200, 300, 400]);
    }

    #[test]
    fn extract_columns_empty_data() {
        let schema = Schema::new(vec![("x".into(), ColumnType::F64)]);
        let columns = schema.extract_columns(&[]).unwrap();
        assert_eq!(columns.len(), 1);
        assert!(columns[0].data.is_empty());
    }

    #[test]
    fn extract_columns_misaligned_data() {
        let schema = Schema::new(vec![
            ("a".into(), ColumnType::U32),
            ("b".into(), ColumnType::F32),
        ]);
        // 7 bytes is not divisible by stride 8
        let data = vec![0u8; 7];
        let result = schema.extract_columns(&data);
        assert!(result.is_err());
    }

    #[test]
    fn validate_frame_stride_match() {
        let schema = Schema::new(vec![
            ("ts".into(), ColumnType::I64),
            ("val".into(), ColumnType::F32),
        ]);
        assert!(schema.validate_frame_stride(12).is_ok());
    }

    #[test]
    fn validate_frame_stride_mismatch() {
        let schema = Schema::new(vec![
            ("ts".into(), ColumnType::I64),
            ("val".into(), ColumnType::F32),
        ]);
        assert!(schema.validate_frame_stride(16).is_err());
    }

    #[test]
    fn validate_frame_stride_zero_always_ok() {
        let schema = Schema::new(vec![("x".into(), ColumnType::F64)]);
        assert!(schema.validate_frame_stride(0).is_ok());
    }

    #[test]
    fn column_type_from_str() {
        assert_eq!(ColumnType::from_str("i64"), Some(ColumnType::I64));
        assert_eq!(ColumnType::from_str("f32"), Some(ColumnType::F32));
        assert_eq!(ColumnType::from_str("u8"), Some(ColumnType::U8));
        assert_eq!(ColumnType::from_str("nope"), None);
    }

    #[test]
    fn extract_columns_large_record_count() {
        // Stress: 100K records of {u32, f64} = stride 12
        let schema = Schema::new(vec![
            ("seq".into(), ColumnType::U32),
            ("val".into(), ColumnType::F64),
        ]);
        let n = 100_000usize;
        let mut data = Vec::with_capacity(n * 12);
        for i in 0..n as u32 {
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(&(i as f64 * 0.1).to_le_bytes());
        }

        let columns = schema.extract_columns(&data).unwrap();

        // Spot-check first and last
        let first_seq = u32::from_le_bytes(columns[0].data[..4].try_into().unwrap());
        let last_seq = u32::from_le_bytes(
            columns[0].data[columns[0].data.len() - 4..]
                .try_into()
                .unwrap(),
        );
        assert_eq!(first_seq, 0);
        assert_eq!(last_seq, 99_999);

        let first_val = f64::from_le_bytes(columns[1].data[..8].try_into().unwrap());
        let last_val = f64::from_le_bytes(
            columns[1].data[columns[1].data.len() - 8..]
                .try_into()
                .unwrap(),
        );
        assert!((first_val - 0.0).abs() < 1e-10);
        assert!((last_val - 9999.9).abs() < 0.1);
    }
}
