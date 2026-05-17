//! PolarQuant lossy preprocessor (HD3 + uniform scalar quantization).

use crate::workspace::{PolarQuantDecodeScratch, PolarQuantEncodeScratch};
use crate::{CompressorError, Result};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Default vector dimension for PolarQuant. Must be a power of 2.
pub const POLAR_QUANT_DEFAULT_DIM: u16 = 256;
/// Default bits per quantized coordinate.
pub const POLAR_QUANT_DEFAULT_BITS: u8 = 8;
/// Default deterministic seed when no seed is configured.
pub const POLAR_QUANT_DEFAULT_SEED: u64 = 0xC0FFEE_FEEDFACE;
/// Minimum bits per coordinate.
pub const POLAR_QUANT_MIN_BITS: u8 = 2;
/// Maximum bits per coordinate.
pub const POLAR_QUANT_MAX_BITS: u8 = 8;
/// Number of HD3 rotation rounds.
pub const HD3_ROUNDS: usize = 3;
/// Quantization range in units of sigma.
pub const POLAR_QUANT_SIGMA_RANGE: f64 = 6.0;
const POLAR_QUANT_MAX_DIM: u16 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolarQuantConfig {
    pub vector_dim: u16,
    pub bits_per_coord: u8,
    pub seed: u64,
}

impl PolarQuantConfig {
    pub fn new(vector_dim: u16, bits_per_coord: u8, seed: u64) -> Result<Self> {
        if !(2..=POLAR_QUANT_MAX_DIM).contains(&vector_dim) || !vector_dim.is_power_of_two() {
            return Err(CompressorError::PolarQuantDimNotPow2(vector_dim));
        }
        if !(POLAR_QUANT_MIN_BITS..=POLAR_QUANT_MAX_BITS).contains(&bits_per_coord) {
            return Err(CompressorError::PolarQuantBitsOutOfRange {
                got: bits_per_coord,
                min: POLAR_QUANT_MIN_BITS,
                max: POLAR_QUANT_MAX_BITS,
            });
        }
        let bits_per_vec = vector_dim as usize * bits_per_coord as usize;
        if bits_per_vec % 8 != 0 {
            return Err(CompressorError::InvalidPolarQuantConfig(format!(
                "vector_dim*bits_per_coord must be byte-aligned, got {}*{}={} bits",
                vector_dim, bits_per_coord, bits_per_vec
            )));
        }
        Ok(Self {
            vector_dim,
            bits_per_coord,
            seed,
        })
    }

    pub fn with_bits(bits_per_coord: u8) -> Result<Self> {
        Self::new(
            POLAR_QUANT_DEFAULT_DIM,
            bits_per_coord,
            POLAR_QUANT_DEFAULT_SEED,
        )
    }
}

pub struct PolarQuantEncoded {
    pub codes: Vec<u8>,
    pub means: Vec<u8>,
    pub scales: Vec<u8>,
    pub vector_count: u32,
}

pub struct PolarQuantEncodedRef<'a> {
    pub codes: &'a [u8],
    pub means: &'a [u8],
    pub scales: &'a [u8],
    pub vector_count: u32,
}

impl<'a> From<&'a PolarQuantEncoded> for PolarQuantEncodedRef<'a> {
    fn from(value: &'a PolarQuantEncoded) -> Self {
        Self {
            codes: &value.codes,
            means: &value.means,
            scales: &value.scales,
            vector_count: value.vector_count,
        }
    }
}

pub struct Hd3 {
    signs: [Vec<i8>; HD3_ROUNDS],
    dim: usize,
    inv_sqrt_d: f64,
}

impl Hd3 {
    pub fn from_seed(seed: u64, dim: usize) -> Self {
        let mut seed32 = [0u8; 32];
        for i in 0..4 {
            let s = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            seed32[i * 8..(i + 1) * 8].copy_from_slice(&s.to_le_bytes());
        }

        let mut rng = ChaCha8Rng::from_seed(seed32);
        let total_bits = HD3_ROUNDS * dim;
        let mut random = vec![0u8; total_bits.div_ceil(8)];
        rng.fill_bytes(&mut random);

        let signs = std::array::from_fn(|round| {
            let mut v = Vec::with_capacity(dim);
            for i in 0..dim {
                let bit_idx = round * dim + i;
                let bit = (random[bit_idx / 8] >> (bit_idx & 7)) & 1;
                v.push(if bit == 0 { -1 } else { 1 });
            }
            v
        });

        Self {
            signs,
            dim,
            inv_sqrt_d: 1.0 / (dim as f64).sqrt(),
        }
    }

    #[inline]
    fn hadamard_in_place(&self, v: &mut [f64]) {
        let mut stride = 1usize;
        while stride < self.dim {
            let step = stride * 2;
            for base in (0..self.dim).step_by(step) {
                for i in 0..stride {
                    let a = v[base + i];
                    let b = v[base + i + stride];
                    v[base + i] = a + b;
                    v[base + i + stride] = a - b;
                }
            }
            stride = step;
        }
    }

    #[inline]
    fn hadamard_in_place_f32(&self, v: &mut [f32]) {
        let mut stride = 1usize;
        while stride < self.dim {
            let step = stride * 2;
            for base in (0..self.dim).step_by(step) {
                for i in 0..stride {
                    let a = v[base + i] as f64;
                    let b = v[base + i + stride] as f64;
                    v[base + i] = (a + b) as f32;
                    v[base + i + stride] = (a - b) as f32;
                }
            }
            stride = step;
        }
    }

    pub fn apply(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), self.dim);
        for round in 0..HD3_ROUNDS {
            for (x, sign) in v.iter_mut().zip(&self.signs[round]) {
                *x *= *sign as f64;
            }
            self.hadamard_in_place(v);
            for x in v.iter_mut() {
                *x *= self.inv_sqrt_d;
            }
        }
    }

    pub fn apply_inverse(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), self.dim);
        for round in (0..HD3_ROUNDS).rev() {
            self.hadamard_in_place(v);
            for x in v.iter_mut() {
                *x *= self.inv_sqrt_d;
            }
            for (x, sign) in v.iter_mut().zip(&self.signs[round]) {
                *x *= *sign as f64;
            }
        }
    }

    pub fn apply_f32(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        for round in 0..HD3_ROUNDS {
            for (x, sign) in v.iter_mut().zip(&self.signs[round]) {
                *x *= *sign as f32;
            }
            self.hadamard_in_place_f32(v);
            for x in v.iter_mut() {
                *x = (*x as f64 * self.inv_sqrt_d) as f32;
            }
        }
    }

    pub fn apply_inverse_f32(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        for round in (0..HD3_ROUNDS).rev() {
            self.hadamard_in_place_f32(v);
            for x in v.iter_mut() {
                *x = (*x as f64 * self.inv_sqrt_d) as f32;
            }
            for (x, sign) in v.iter_mut().zip(&self.signs[round]) {
                *x *= *sign as f32;
            }
        }
    }
}

#[inline]
fn quantize_code(x: f64, sigma: f64, bits: u8) -> u16 {
    let max_code = ((1u16 << bits) - 1) as f64;
    if sigma <= 0.0 || !sigma.is_finite() {
        return (max_code / 2.0).round() as u16;
    }
    let lo = -POLAR_QUANT_SIGMA_RANGE * sigma;
    let span = 2.0 * POLAR_QUANT_SIGMA_RANGE * sigma;
    let q = ((x - lo) / span * max_code).round();
    q.clamp(0.0, max_code) as u16
}

#[inline]
fn dequantize_code(code: u16, sigma: f64, bits: u8) -> f64 {
    if sigma <= 0.0 || !sigma.is_finite() {
        return 0.0;
    }
    let max_code = ((1u16 << bits) - 1) as f64;
    let q = code as f64 / max_code;
    let span = 2.0 * POLAR_QUANT_SIGMA_RANGE * sigma;
    q * span - POLAR_QUANT_SIGMA_RANGE * sigma
}

#[inline]
fn write_packed_symbol(buf: &mut [u8], bit_pos: usize, bits: u8, symbol: u16) {
    let byte = bit_pos >> 3;
    let shift = bit_pos & 7;
    let word = (symbol as u32) << shift;
    buf[byte] |= word as u8;
    if shift + bits as usize > 8 {
        buf[byte + 1] |= (word >> 8) as u8;
    }
}

#[inline]
fn read_packed_symbol(buf: &[u8], bit_pos: usize, bits: u8) -> u16 {
    let byte = bit_pos >> 3;
    let shift = bit_pos & 7;
    let mut word = buf[byte] as u16;
    if byte + 1 < buf.len() {
        word |= (buf[byte + 1] as u16) << 8;
    }
    let mask = (1u16 << bits) - 1;
    (word >> shift) & mask
}

fn validate_input_len(len: usize, dim: usize) -> Result<()> {
    if len == 0 {
        return Err(CompressorError::EmptyInput);
    }
    if len % dim != 0 {
        return Err(CompressorError::DataTypeMismatch {
            element_size: dim,
            buffer_len: len,
        });
    }
    Ok(())
}

fn validate_encoded(enc: &PolarQuantEncodedRef<'_>, cfg: PolarQuantConfig) -> Result<()> {
    let dim = cfg.vector_dim as usize;
    let vecs = enc.vector_count as usize;
    let expected_codes_bits = vecs * dim * cfg.bits_per_coord as usize;
    let expected_codes_bytes = expected_codes_bits.div_ceil(8);
    let expected_param_bytes = vecs * 4;

    if enc.codes.len() != expected_codes_bytes {
        return Err(CompressorError::InvalidPolarQuantConfig(format!(
            "codes length mismatch: expected {}, got {}",
            expected_codes_bytes,
            enc.codes.len()
        )));
    }
    if enc.means.len() != expected_param_bytes {
        return Err(CompressorError::InvalidPolarQuantConfig(format!(
            "means length mismatch: expected {}, got {}",
            expected_param_bytes,
            enc.means.len()
        )));
    }
    if enc.scales.len() != expected_param_bytes {
        return Err(CompressorError::InvalidPolarQuantConfig(format!(
            "scales length mismatch: expected {}, got {}",
            expected_param_bytes,
            enc.scales.len()
        )));
    }
    Ok(())
}

pub fn encode_f64(values: &[f64], cfg: PolarQuantConfig) -> Result<PolarQuantEncoded> {
    let mut scratch = PolarQuantEncodeScratch::new();
    encode_f64_into(values, cfg, &mut scratch)?;
    Ok(PolarQuantEncoded {
        vector_count: (values.len() / cfg.vector_dim as usize) as u32,
        codes: std::mem::take(&mut scratch.codes),
        means: std::mem::take(&mut scratch.means),
        scales: std::mem::take(&mut scratch.scales),
    })
}

pub fn encode_f32(values: &[f32], cfg: PolarQuantConfig) -> Result<PolarQuantEncoded> {
    let mut scratch = PolarQuantEncodeScratch::new();
    encode_f32_into(values, cfg, &mut scratch)?;
    Ok(PolarQuantEncoded {
        vector_count: (values.len() / cfg.vector_dim as usize) as u32,
        codes: std::mem::take(&mut scratch.codes),
        means: std::mem::take(&mut scratch.means),
        scales: std::mem::take(&mut scratch.scales),
    })
}

pub fn decode_f64(enc: &PolarQuantEncoded, cfg: PolarQuantConfig) -> Result<Vec<f64>> {
    let enc_ref = PolarQuantEncodedRef::from(enc);
    let mut out = Vec::new();
    let mut scratch = PolarQuantDecodeScratch::new();
    decode_f64_into(&enc_ref, cfg, &mut out, &mut scratch)?;
    let mut values = Vec::with_capacity(out.len() / 8);
    for chunk in out.chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

pub fn decode_f32(enc: &PolarQuantEncoded, cfg: PolarQuantConfig) -> Result<Vec<f32>> {
    let enc_ref = PolarQuantEncodedRef::from(enc);
    let mut out = Vec::new();
    let mut scratch = PolarQuantDecodeScratch::new();
    decode_f32_into(&enc_ref, cfg, &mut out, &mut scratch)?;
    let mut values = Vec::with_capacity(out.len() / 4);
    for chunk in out.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

pub fn encode_f64_into(
    values: &[f64],
    cfg: PolarQuantConfig,
    scratch: &mut PolarQuantEncodeScratch,
) -> Result<()> {
    validate_input_len(values.len(), cfg.vector_dim as usize)?;
    let dim = cfg.vector_dim as usize;
    let vector_count = values.len() / dim;
    let hd3 = Hd3::from_seed(cfg.seed, dim);

    scratch.working.resize(dim, 0.0);
    scratch.means.clear();
    scratch.scales.clear();
    scratch.codes.clear();
    scratch.means.reserve(vector_count * 4);
    scratch.scales.reserve(vector_count * 4);
    scratch.codes.resize(
        (vector_count * dim * cfg.bits_per_coord as usize).div_ceil(8),
        0,
    );

    let mut bit_pos = 0usize;
    for chunk in values.chunks_exact(dim) {
        let mean = chunk.iter().copied().sum::<f64>() / dim as f64;
        let mean_f32 = mean as f32;
        scratch.means.extend_from_slice(&mean_f32.to_le_bytes());

        let mut sumsq = 0.0;
        for (dst, &v) in scratch.working.iter_mut().zip(chunk) {
            let centered = v - mean;
            *dst = centered;
            sumsq += centered * centered;
        }
        let sigma = (sumsq / dim as f64).sqrt();
        let sigma_f32 = sigma as f32;
        scratch.scales.extend_from_slice(&sigma_f32.to_le_bytes());

        hd3.apply(&mut scratch.working);
        for &coord in &scratch.working {
            let code = quantize_code(coord, sigma, cfg.bits_per_coord);
            write_packed_symbol(&mut scratch.codes, bit_pos, cfg.bits_per_coord, code);
            bit_pos += cfg.bits_per_coord as usize;
        }
    }

    Ok(())
}

pub fn encode_f32_into(
    values: &[f32],
    cfg: PolarQuantConfig,
    scratch: &mut PolarQuantEncodeScratch,
) -> Result<()> {
    validate_input_len(values.len(), cfg.vector_dim as usize)?;
    let dim = cfg.vector_dim as usize;
    let vector_count = values.len() / dim;
    let hd3 = Hd3::from_seed(cfg.seed, dim);

    scratch.working.resize(dim, 0.0);
    scratch.means.clear();
    scratch.scales.clear();
    scratch.codes.clear();
    scratch.means.reserve(vector_count * 4);
    scratch.scales.reserve(vector_count * 4);
    scratch.codes.resize(
        (vector_count * dim * cfg.bits_per_coord as usize).div_ceil(8),
        0,
    );

    let mut bit_pos = 0usize;
    for chunk in values.chunks_exact(dim) {
        let mean = chunk.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
        let mean_f32 = mean as f32;
        scratch.means.extend_from_slice(&mean_f32.to_le_bytes());

        let mut sumsq = 0.0;
        for (dst, &v) in scratch.working.iter_mut().zip(chunk) {
            let centered = v as f64 - mean;
            *dst = centered;
            sumsq += centered * centered;
        }
        let sigma = (sumsq / dim as f64).sqrt();
        let sigma_f32 = sigma as f32;
        scratch.scales.extend_from_slice(&sigma_f32.to_le_bytes());

        hd3.apply(&mut scratch.working);
        for &coord in &scratch.working {
            let code = quantize_code(coord, sigma, cfg.bits_per_coord);
            write_packed_symbol(&mut scratch.codes, bit_pos, cfg.bits_per_coord, code);
            bit_pos += cfg.bits_per_coord as usize;
        }
    }

    Ok(())
}

pub fn decode_f64_into(
    enc: &PolarQuantEncodedRef<'_>,
    cfg: PolarQuantConfig,
    out: &mut Vec<u8>,
    scratch: &mut PolarQuantDecodeScratch,
) -> Result<()> {
    validate_encoded(enc, cfg)?;
    let dim = cfg.vector_dim as usize;
    let vec_count = enc.vector_count as usize;
    let hd3 = Hd3::from_seed(cfg.seed, dim);

    out.clear();
    out.reserve(vec_count * dim * 8);
    scratch.working.resize(dim, 0.0);
    scratch.dequant_buf.resize(dim, 0.0);

    let mut bit_pos = 0usize;
    for v in 0..vec_count {
        let moff = v * 4;
        let mean = f32::from_le_bytes(enc.means[moff..moff + 4].try_into().unwrap()) as f64;
        let sigma = f32::from_le_bytes(enc.scales[moff..moff + 4].try_into().unwrap()) as f64;

        for i in 0..dim {
            let q = read_packed_symbol(enc.codes, bit_pos, cfg.bits_per_coord);
            bit_pos += cfg.bits_per_coord as usize;
            scratch.dequant_buf[i] = dequantize_code(q, sigma, cfg.bits_per_coord);
        }

        hd3.apply_inverse(&mut scratch.dequant_buf);
        for &x in &scratch.dequant_buf {
            out.extend_from_slice(&(x + mean).to_le_bytes());
        }
    }

    Ok(())
}

pub fn decode_f32_into(
    enc: &PolarQuantEncodedRef<'_>,
    cfg: PolarQuantConfig,
    out: &mut Vec<u8>,
    scratch: &mut PolarQuantDecodeScratch,
) -> Result<()> {
    validate_encoded(enc, cfg)?;
    let dim = cfg.vector_dim as usize;
    let vec_count = enc.vector_count as usize;
    let hd3 = Hd3::from_seed(cfg.seed, dim);

    out.clear();
    out.reserve(vec_count * dim * 4);
    scratch.working.resize(dim, 0.0);
    scratch.dequant_buf.resize(dim, 0.0);

    let mut bit_pos = 0usize;
    for v in 0..vec_count {
        let moff = v * 4;
        let mean = f32::from_le_bytes(enc.means[moff..moff + 4].try_into().unwrap()) as f64;
        let sigma = f32::from_le_bytes(enc.scales[moff..moff + 4].try_into().unwrap()) as f64;

        for i in 0..dim {
            let q = read_packed_symbol(enc.codes, bit_pos, cfg.bits_per_coord);
            bit_pos += cfg.bits_per_coord as usize;
            scratch.dequant_buf[i] = dequantize_code(q, sigma, cfg.bits_per_coord);
        }

        hd3.apply_inverse(&mut scratch.dequant_buf);
        for &x in &scratch.dequant_buf {
            out.extend_from_slice(&((x + mean) as f32).to_le_bytes());
        }
    }

    Ok(())
}
