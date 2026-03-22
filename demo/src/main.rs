#![windows_subsystem = "windows"]

// iotc-demo: One-click compression demo for C-suite / non-technical audiences.
//
// Double-click the .exe → pick a file → see compression results → save.
// No installer, no dependencies, no CLI knowledge required.

use iot_compressor::{CompressionConfig, DataType, ParserMode};
use iot_compressor::parallel::{compress, decompress};
use rfd::{FileDialog, MessageButtons, MessageDialog, MessageDialogResult, MessageLevel};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    // ── Welcome ──
    MessageDialog::new()
        .set_level(MessageLevel::Info)
        .set_title("iotc — IoT Compressor Demo")
        .set_description(
            "Welcome to the iotc compression demo.\n\n\
             Select any file to see how iotc compresses it.\n\n\
             For best results, use sensor data, telemetry logs,\n\
             or CSV files with numeric columns.",
        )
        .set_buttons(MessageButtons::Ok)
        .show();

    // ── Pick file ──
    let path = match pick_input_file() {
        Some(p) => p,
        None => return,
    };

    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".into());

    // ── Read ──
    let raw = match fs::read(&path) {
        Ok(data) => data,
        Err(e) => {
            show_error(&format!("Failed to read file:\n{}", e));
            return;
        }
    };

    if raw.is_empty() {
        show_error("File is empty.");
        return;
    }

    let raw_size = raw.len();

    // ── Try stride detection from filename ──
    let stride = detect_stride_hint(&filename, &raw);

    // ── Compress (greedy for speed) ──
    let config = CompressionConfig {
        parser_mode: ParserMode::Greedy,
        data_type: Some(DataType::Raw),
        store_checksum: true,
        store_seek_table: true,
        stride,
        ..CompressionConfig::fast()
    };

    let start = Instant::now();
    let compressed = match compress(&raw, &config) {
        Ok(c) => c,
        Err(e) => {
            show_error(&format!("Compression failed:\n{}", e));
            return;
        }
    };
    let compress_time = start.elapsed();

    // ── Verify roundtrip ──
    let decompress_start = Instant::now();
    let decompressed = match decompress(&compressed) {
        Ok(d) => d,
        Err(e) => {
            show_error(&format!("Decompression verification failed:\n{}", e));
            return;
        }
    };
    let decompress_time = decompress_start.elapsed();

    if decompressed != raw {
        show_error("Roundtrip verification failed — data mismatch!");
        return;
    }

    let compressed_size = compressed.len();
    let ratio = raw_size as f64 / compressed_size as f64;
    let savings_pct = (1.0 - compressed_size as f64 / raw_size as f64) * 100.0;

    let compress_speed = raw_size as f64 / compress_time.as_secs_f64() / (1024.0 * 1024.0);
    let decompress_speed = raw_size as f64 / decompress_time.as_secs_f64() / (1024.0 * 1024.0);

    // AWS S3 Standard: $0.023/GB/month
    let monthly_savings_per_copy =
        (raw_size as f64 - compressed_size as f64) / (1024.0 * 1024.0 * 1024.0) * 0.023;

    // ── Results ──
    let results = format!(
        "File: {}\n\n\
         Original:     {}\n\
         Compressed:   {}\n\
         Ratio:        {:.1}x\n\
         Savings:      {:.1}%\n\n\
         Compress:     {:.1} MiB/s  ({:.2}s)\n\
         Decompress:   {:.1} MiB/s  ({:.2}s)\n\
         Roundtrip:    Verified OK\n\n\
         ─── Cost Impact ───\n\
         AWS S3 savings: ${:.4}/month per copy\n\
         Network:        {:.1}% less bandwidth\n\
         {}\n\n\
         Save the compressed file?",
        filename,
        format_size(raw_size),
        format_size(compressed_size),
        ratio,
        savings_pct,
        compress_speed,
        compress_time.as_secs_f64(),
        decompress_speed,
        decompress_time.as_secs_f64(),
        monthly_savings_per_copy,
        savings_pct,
        if stride.is_some() {
            format!("\nStride: {} bytes (auto-detected)", stride.unwrap())
        } else {
            String::new()
        },
    );

    let save = MessageDialog::new()
        .set_level(MessageLevel::Info)
        .set_title("iotc — Compression Results")
        .set_description(&results)
        .set_buttons(MessageButtons::YesNo)
        .show();

    // ── Save ──
    if save == MessageDialogResult::Yes {
        let default_name = format!("{}.iotc", filename);
        if let Some(save_path) = FileDialog::new()
            .set_title("Save compressed file")
            .set_file_name(&default_name)
            .add_filter("iotc compressed", &["iotc"])
            .add_filter("All files", &["*"])
            .save_file()
        {
            match fs::write(&save_path, &compressed) {
                Ok(()) => {
                    MessageDialog::new()
                        .set_level(MessageLevel::Info)
                        .set_title("iotc — Saved")
                        .set_description(&format!(
                            "Compressed file saved to:\n{}\n\n{}",
                            save_path.display(),
                            format_size(compressed_size),
                        ))
                        .set_buttons(MessageButtons::Ok)
                        .show();
                }
                Err(e) => show_error(&format!("Failed to save:\n{}", e)),
            }
        }
    }
}

fn pick_input_file() -> Option<PathBuf> {
    FileDialog::new()
        .set_title("iotc — Select a file to compress")
        .add_filter("Data files", &["csv", "bin", "dat", "raw", "log", "tsv", "json", "parquet"])
        .add_filter("All files", &["*"])
        .pick_file()
}

fn show_error(msg: &str) {
    MessageDialog::new()
        .set_level(MessageLevel::Error)
        .set_title("iotc — Error")
        .set_description(msg)
        .set_buttons(MessageButtons::Ok)
        .show();
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Try to detect struct stride from file extension or content heuristics.
///
/// Returns `Some(stride)` if the file looks like fixed-width records,
/// `None` for text/variable-width data.
fn detect_stride_hint(filename: &str, data: &[u8]) -> Option<u16> {
    let lower = filename.to_lowercase();

    // Skip stride for text formats
    if lower.ends_with(".csv")
        || lower.ends_with(".json")
        || lower.ends_with(".tsv")
        || lower.ends_with(".log")
        || lower.ends_with(".txt")
        || lower.ends_with(".xml")
    {
        return None;
    }

    // For binary files, check common struct sizes by alignment
    // Only apply if the file size is cleanly divisible
    for &candidate in &[20u16, 24, 16, 12, 32, 28, 8, 40, 48, 64] {
        if data.len() >= 1024
            && data.len() % candidate as usize == 0
            && data.len() / candidate as usize >= 10
        {
            return Some(candidate);
        }
    }

    None
}
