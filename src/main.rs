use clap::{Parser, ValueEnum};
use iot_compressor::{
    parallel::{compress, decompress},
    BlockHeader, CompressionConfig, DataType, FrameHeader, ParserMode, SeekTable,
};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "iotc",
    about = "IoT Time-Series Compressor — LZ77/ANS for sensor data",
    version
)]
struct Args {
    /// Input file path
    input: PathBuf,

    /// Output file path (not required with --info)
    output: Option<PathBuf>,

    /// Decompress the input file
    #[arg(short, long)]
    decompress: bool,

    /// Parser mode
    #[arg(short, long, value_enum, default_value_t = CliParserMode::Lazy)]
    parser: CliParserMode,

    /// Data type hint for preprocessing
    #[arg(short = 't', long, value_enum)]
    data_type: Option<CliDataType>,

    /// Number of threads (0 = auto)
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,

    /// Block size in bytes
    #[arg(short, long, default_value_t = 2 * 1024 * 1024)]
    block_size: usize,

    /// Disable SHA-256 content checksum
    #[arg(long)]
    no_checksum: bool,

    /// Struct stride for columnar transposition (bytes per record).
    /// Transposes fixed-size records column-wise before LZ77 for better
    /// match quality on structured data. E.g., --stride 12 for a struct
    /// with 3 × i32 fields.
    #[arg(short, long)]
    stride: Option<u16>,

    /// Disable seek table generation
    #[arg(long)]
    no_seek_table: bool,

    /// Print frame info and exit (no output file needed)
    #[arg(long)]
    info: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum CliParserMode {
    Greedy,
    Lazy,
    Optimal,
}

impl From<CliParserMode> for ParserMode {
    fn from(m: CliParserMode) -> Self {
        match m {
            CliParserMode::Greedy => ParserMode::Greedy,
            CliParserMode::Lazy => ParserMode::Lazy,
            CliParserMode::Optimal => ParserMode::Optimal,
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
enum CliDataType {
    Raw,
    I64,
    U64,
    I32,
    U32,
    F64,
    F32,
    /// f64 with byte-level shuffle (better for noisy sensor data)
    F64s,
    /// f32 with byte-level shuffle (better for noisy sensor data)
    F32s,
    /// f64 with byte shuffle + byte delta (best auto-detect choice for smooth floats)
    F64sd,
    /// f32 with byte shuffle + byte delta (best auto-detect choice for smooth floats)
    F32sd,
}

impl From<CliDataType> for DataType {
    fn from(t: CliDataType) -> Self {
        match t {
            CliDataType::Raw => DataType::Raw,
            CliDataType::I64 => DataType::IntegerI64,
            CliDataType::U64 => DataType::IntegerU64,
            CliDataType::I32 => DataType::IntegerI32,
            CliDataType::U32 => DataType::IntegerU32,
            CliDataType::F64 => DataType::Float64,
            CliDataType::F32 => DataType::Float32,
            CliDataType::F64s => DataType::Float64Shuffle,
            CliDataType::F32s => DataType::Float32Shuffle,
            CliDataType::F64sd => DataType::Float64ShuffleDelta,
            CliDataType::F32sd => DataType::Float32ShuffleDelta,
        }
    }
}

fn print_frame_info(data: &[u8]) -> anyhow::Result<()> {
    let header = FrameHeader::from_bytes(data)
        .map_err(|e| anyhow::anyhow!("Failed to parse frame header: {:?}", e))?;

    let flags = &header.flags;
    println!(
        "Frame: IOTC v{}, {} bytes original, {} blocks, block_size={}",
        header.version, header.original_size, header.block_count, header.block_size
    );
    println!(
        "Flags: data_type={}, parser={}, checksum={}, repcodes={}, seek_table={}",
        flags.data_type,
        flags.parser_mode,
        if flags.has_content_checksum {
            "yes"
        } else {
            "no"
        },
        if flags.has_repcodes { "yes" } else { "no" },
        if flags.has_seek_table { "yes" } else { "no" },
    );
    if header.stride > 0 {
        println!("Stride: {}", header.stride);
    }

    if flags.has_seek_table {
        let st_size = SeekTable::serialized_size(header.block_count);
        let st_start = FrameHeader::SERIALIZED_SIZE;
        if st_start + st_size > data.len() {
            println!(
                "Seek Table: TRUNCATED (need {} bytes, have {})",
                st_size,
                data.len() - st_start
            );
            return Ok(());
        }
        let st = SeekTable::from_bytes(&data[st_start..st_start + st_size], header.block_count)
            .map_err(|e| anyhow::anyhow!("Seek table error: {:?}", e))?;

        println!(
            "Seek Table ({} entries, CRC: 0x{:08X}):",
            st.entries.len(),
            st.crc32
        );
        for (i, &offset) in st.entries.iter().enumerate() {
            // Parse block header at offset to get compressed/original sizes
            let off = offset as usize;
            if off + BlockHeader::SERIALIZED_SIZE <= data.len() {
                let bh = BlockHeader::from_bytes(&data[off..off + BlockHeader::SERIALIZED_SIZE])
                    .map_err(|e| anyhow::anyhow!("Block header error: {:?}", e))?;
                println!(
                    "  Block {}: offset={}, compressed={}, original={}",
                    i, offset, bh.compressed_size, bh.original_size
                );
            } else {
                println!("  Block {}: offset={} (header truncated)", i, offset);
            }
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut input_file = File::open(&args.input)?;
    let mut data = Vec::new();
    input_file.read_to_end(&mut data)?;

    if args.info {
        return print_frame_info(&data);
    }

    let output_path = args.output.ok_or_else(|| {
        anyhow::anyhow!("Output file path is required (use --info to inspect without output)")
    })?;

    let start = Instant::now();

    if args.decompress {
        eprintln!("Decompressing {} bytes...", data.len());

        let result =
            decompress(&data).map_err(|e| anyhow::anyhow!("Decompression failed: {:?}", e))?;

        let mut output_file = File::create(&output_path)?;
        output_file.write_all(&result)?;

        let elapsed = start.elapsed();
        let throughput = result.len() as f64 / 1_000_000.0 / elapsed.as_secs_f64();
        eprintln!(
            "Decompressed to {} bytes in {:.2?} ({:.1} MB/s)",
            result.len(),
            elapsed,
            throughput
        );
    } else {
        let config = CompressionConfig {
            parser_mode: args.parser.into(),
            data_type: args.data_type.map(|t| t.into()),
            num_threads: args.threads,
            block_size: args.block_size,
            store_checksum: !args.no_checksum,
            stride: args.stride,
            store_seek_table: !args.no_seek_table,
            ..Default::default()
        };

        let stride_info = match config.stride {
            Some(s) => format!(", stride: {}", s),
            None => String::new(),
        };
        eprintln!(
            "Compressing {} bytes (parser: {}, block_size: {}{})",
            data.len(),
            config.parser_mode,
            config.block_size,
            stride_info
        );

        let result =
            compress(&data, &config).map_err(|e| anyhow::anyhow!("Compression failed: {:?}", e))?;

        let mut output_file = File::create(&output_path)?;
        output_file.write_all(&result)?;

        let elapsed = start.elapsed();
        let ratio = data.len() as f64 / result.len() as f64;
        let throughput = data.len() as f64 / 1_000_000.0 / elapsed.as_secs_f64();
        eprintln!(
            "Compressed to {} bytes in {:.2?} ({:.1} MB/s, {:.2}x ratio)",
            result.len(),
            elapsed,
            throughput,
            ratio
        );
    }

    Ok(())
}
