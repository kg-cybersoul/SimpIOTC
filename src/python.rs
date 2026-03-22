//! # Python Bindings via PyO3
//!
//! Provides a `iotc` Python module with `compress`, `decompress`, and a
//! `SeekableReader` class for O(1) random block access.
//!
//! Build with: `maturin build --features python`
//! Install with: `pip install target/wheels/iotc-*.whl`

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::parallel;
use crate::schema::{ColumnType, Schema};
use crate::seekable;
use crate::{CompressionConfig, CompressorError, DataType, ParserMode};

// ═══════════════════════════════════════════════════════════════════════════════
// Error Conversion
// ═══════════════════════════════════════════════════════════════════════════════

fn to_py_err(e: CompressorError) -> PyErr {
    match e {
        CompressorError::Io(io_err) => PyIOError::new_err(format!("{}", io_err)),
        CompressorError::EmptyInput => PyValueError::new_err("empty input"),
        CompressorError::DataTypeMismatch {
            element_size,
            buffer_len,
        } => PyValueError::new_err(format!(
            "buffer length {} not aligned to element size {}",
            buffer_len, element_size
        )),
        CompressorError::StrideMismatch { stride, buffer_len } => PyValueError::new_err(format!(
            "buffer length {} not aligned to stride {}",
            buffer_len, stride
        )),
        other => PyValueError::new_err(format!("{}", other)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Config Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn parse_data_type(s: &str) -> PyResult<Option<DataType>> {
    match s {
        "auto" => Ok(None),
        "raw" => Ok(Some(DataType::Raw)),
        "i32" => Ok(Some(DataType::IntegerI32)),
        "u32" => Ok(Some(DataType::IntegerU32)),
        "i64" => Ok(Some(DataType::IntegerI64)),
        "u64" => Ok(Some(DataType::IntegerU64)),
        "f32" => Ok(Some(DataType::Float32)),
        "f64" => Ok(Some(DataType::Float64)),
        "f32s" => Ok(Some(DataType::Float32Shuffle)),
        "f64s" => Ok(Some(DataType::Float64Shuffle)),
        "f32sd" => Ok(Some(DataType::Float32ShuffleDelta)),
        "f64sd" => Ok(Some(DataType::Float64ShuffleDelta)),
        _ => Err(PyValueError::new_err(format!(
            "unknown data_type '{}': expected auto, raw, i32, u32, i64, u64, f32, f64, f32s, f64s, f32sd, f64sd", s
        ))),
    }
}

fn parse_parser_mode(s: &str) -> PyResult<ParserMode> {
    match s {
        "greedy" => Ok(ParserMode::Greedy),
        "lazy" => Ok(ParserMode::Lazy),
        "optimal" => Ok(ParserMode::Optimal),
        _ => Err(PyValueError::new_err(format!(
            "unknown parser '{}': expected greedy, lazy, optimal",
            s
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Schema Parsing
// ═══════════════════════════════════════════════════════════════════════════════

fn parse_schema(dict: &Bound<'_, PyDict>) -> PyResult<Schema> {
    let mut columns = Vec::new();
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let type_str: String = value.extract()?;
        let col_type = ColumnType::from_str(&type_str).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown column type '{}': expected i8, i16, i32, i64, u8, u16, u32, u64, f32, f64",
                type_str
            ))
        })?;
        columns.push((name, col_type));
    }
    if columns.is_empty() {
        return Err(PyValueError::new_err(
            "schema must have at least one column",
        ));
    }
    Ok(Schema::new(columns))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compress data.
///
/// Args:
///     data (bytes): Input data to compress.
///     parser (str): Parser mode — "greedy", "lazy" (default), or "optimal".
///     data_type (str): Data type — "auto" (default), "raw", "i32", "u32", "i64", "u64", "f32", "f64", "f32s", "f64s".
///     stride (int): Struct stride in bytes for columnar transposition (0 = disabled).
///     threads (int): Number of threads (0 = auto). Requires the 'parallel' feature.
///     block_size (int): Block size in bytes (default 2 MiB).
///     checksum (bool): Whether to include a SHA-256 content checksum (default True).
///     seek_table (bool): Whether to include a seek table (default True).
///
/// Returns:
///     bytes: Compressed data.
#[pyfunction]
#[pyo3(signature = (data, *, parser="lazy", data_type="auto", stride=0, threads=0, block_size=2097152, checksum=true, seek_table=true))]
#[allow(clippy::too_many_arguments)]
fn compress<'py>(
    py: Python<'py>,
    data: &[u8],
    parser: &str,
    data_type: &str,
    stride: u16,
    threads: usize,
    block_size: usize,
    checksum: bool,
    seek_table: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mode = parse_parser_mode(parser)?;
    let base = match mode {
        ParserMode::Greedy => CompressionConfig::fast(),
        ParserMode::Lazy => CompressionConfig::balanced(),
        ParserMode::Optimal => CompressionConfig::max_compression(),
    };
    let config = CompressionConfig {
        parser_mode: mode,
        data_type: parse_data_type(data_type)?,
        stride: if stride > 0 { Some(stride) } else { None },
        num_threads: threads,
        block_size,
        store_checksum: checksum,
        store_seek_table: seek_table,
        window_size: base.window_size,
        max_chain_depth: base.max_chain_depth,
    };

    let result = py.allow_threads(|| parallel::compress(data, &config));

    match result {
        Ok(compressed) => Ok(PyBytes::new(py, &compressed)),
        Err(e) => Err(to_py_err(e)),
    }
}

/// Decompress data.
///
/// Args:
///     data (bytes): Compressed data.
///
/// Returns:
///     bytes: Decompressed data.
#[pyfunction]
fn decompress<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let result = py.allow_threads(|| parallel::decompress(data));

    match result {
        Ok(decompressed) => Ok(PyBytes::new(py, &decompressed)),
        Err(e) => Err(to_py_err(e)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SeekableReader Class
// ═══════════════════════════════════════════════════════════════════════════════

/// A seekable reader for O(1) random block access to compressed data.
///
/// The reader copies the compressed data internally, so the original bytes
/// can be freed after construction.
///
/// Example:
///     reader = iotc.SeekableReader(compressed_bytes)
///     block0 = reader.decompress_block(0)
///     all_data = reader.decompress_all()
#[pyclass]
struct SeekableReader {
    /// Owned copy of the compressed data. Kept alive for the reader's borrow.
    #[allow(dead_code)]
    data: Vec<u8>,
    /// The actual reader. Borrows from `data` via a lifetime-erased reference.
    /// SAFETY: `data` is heap-allocated, never reallocated after construction,
    /// and is dropped AFTER `reader` because struct fields drop in declaration order.
    reader: seekable::SeekableReader<'static>,
}

#[pymethods]
impl SeekableReader {
    /// Create a new SeekableReader from compressed data.
    ///
    /// The data must have been compressed with seek_table=True (the default).
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        let owned = data.to_vec();

        // SAFETY: Same self-referential pattern as the FFI module.
        // The Vec is heap-allocated and never reallocated. The SeekableReader
        // borrows from the stable heap buffer. Fields drop in order.
        let slice: &[u8] = &owned;
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };

        let reader = seekable::SeekableReader::new(static_slice).map_err(to_py_err)?;

        Ok(Self {
            data: owned,
            reader,
        })
    }

    /// Number of blocks in the compressed frame.
    #[getter]
    fn block_count(&self) -> u32 {
        self.reader.block_count()
    }

    /// Original (uncompressed) size in bytes.
    #[getter]
    fn original_size(&self) -> u64 {
        self.reader.original_size()
    }

    /// Block size in bytes (as configured during compression).
    #[getter]
    fn block_size(&self) -> u32 {
        self.reader.header().block_size
    }

    /// Decompress a single block by index.
    fn decompress_block<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let block = self.reader.decompress_block(index).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &block))
    }

    /// Decompress a contiguous range of blocks [start, end).
    fn decompress_range<'py>(
        &mut self,
        py: Python<'py>,
        start: usize,
        end: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = self
            .reader
            .decompress_range(start, end)
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &data))
    }

    /// Decompress an arbitrary byte range [byte_start, byte_end).
    fn decompress_byte_range<'py>(
        &mut self,
        py: Python<'py>,
        byte_start: u64,
        byte_end: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = self
            .reader
            .decompress_byte_range(byte_start, byte_end)
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &data))
    }

    /// Decompress all blocks (the entire frame).
    fn decompress_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.reader.decompress_all().map_err(to_py_err)?;
        Ok(PyBytes::new(py, &data))
    }

    /// Stride value from the frame header (0 if not stride-compressed).
    #[getter]
    fn stride(&self) -> u16 {
        self.reader.stride()
    }

    /// Extract typed columns from structured record data.
    ///
    /// Decompresses the specified block range, then splits the row-major byte
    /// stream into per-column contiguous buffers using the provided schema.
    ///
    /// Args:
    ///     schema (dict): Ordered mapping of {column_name: type_str}.
    ///         Type strings: "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64".
    ///     start_block (int): First block to decompress (default 0).
    ///     end_block (int): One past last block (default: all blocks).
    ///
    /// Returns:
    ///     dict: {column_name: bytes} — each value is contiguous little-endian
    ///     typed data suitable for `numpy.frombuffer(data, dtype=...)`.
    ///
    /// Example:
    ///     columns = reader.extract_columns(
    ///         {"timestamp": "i64", "temperature": "f64", "vibration": "f32"},
    ///     )
    ///     import numpy as np
    ///     ts = np.frombuffer(columns["timestamp"], dtype=np.int64)
    #[pyo3(signature = (schema, *, start_block=None, end_block=None))]
    fn extract_columns<'py>(
        &mut self,
        py: Python<'py>,
        schema: &Bound<'py, PyDict>,
        start_block: Option<usize>,
        end_block: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rust_schema = parse_schema(schema)?;
        let start = start_block.unwrap_or(0);
        let end = end_block.unwrap_or(self.reader.block_count() as usize);

        let columns = self
            .reader
            .extract_columns(&rust_schema, start, end)
            .map_err(to_py_err)?;

        let result = PyDict::new(py);
        for col in &columns {
            result.set_item(&col.name, PyBytes::new(py, &col.data))?;
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "SeekableReader(blocks={}, original_size={}, block_size={}, stride={})",
            self.block_count(),
            self.original_size(),
            self.block_size(),
            self.stride(),
        )
    }

    fn __len__(&self) -> usize {
        self.original_size() as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module Definition
// ═══════════════════════════════════════════════════════════════════════════════

/// IoT Time-Series Compressor — Python API
///
/// High-speed, high-ratio LZ77/ANS compression for sensor and IoT time-series data.
///
/// Functions:
///     compress(data, **kwargs) -> bytes
///     decompress(data) -> bytes
///
/// Classes:
///     SeekableReader(data) — O(1) random block access
#[pymodule]
fn iotc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_class::<SeekableReader>()?;
    Ok(())
}
