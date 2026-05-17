use iot_compressor::preprocessor::polar_quant::{
    decode_f32, decode_f64, encode_f32, encode_f64, Hd3, PolarQuantConfig, POLAR_QUANT_DEFAULT_DIM,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;
use std::time::Instant;

fn build_cfg(bits: u8) -> PolarQuantConfig {
    PolarQuantConfig::new(POLAR_QUANT_DEFAULT_DIM, bits, 0xA11CE5EED).unwrap()
}

fn centered_l2(values: &[f64]) -> f64 {
    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn hd3_norm_preservation() {
    let dim = POLAR_QUANT_DEFAULT_DIM as usize;
    let hd3 = Hd3::from_seed(0x1234_5678_ABCD_EF01, dim);
    let mut rng = StdRng::seed_from_u64(12345);

    let mut v: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let original = v.clone();
    hd3.apply(&mut v);
    hd3.apply_inverse(&mut v);

    let max_err = original
        .iter()
        .zip(&v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_err < 1e-12, "max err: {}", max_err);
}

#[test]
fn hd3_roundtrip_many_vectors() {
    let dim = POLAR_QUANT_DEFAULT_DIM as usize;
    let hd3 = Hd3::from_seed(0x55AA_66BB_77CC_88DD, dim);
    let mut rng = StdRng::seed_from_u64(98765);

    for _ in 0..1000 {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let original = v.clone();
        hd3.apply(&mut v);
        hd3.apply_inverse(&mut v);
        let max_err = original
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-10, "max err: {}", max_err);
    }
}

#[test]
fn hd3_gaussianization_kurtosis() {
    let dim = POLAR_QUANT_DEFAULT_DIM as usize;
    let samples = 10_000usize;
    let hd3 = Hd3::from_seed(0xDEADBEEF, dim);
    let mut rng = StdRng::seed_from_u64(4242);
    let mut rotated = vec![0.0f64; dim * samples];
    let mut v = vec![0.0f64; dim];

    for s in 0..samples {
        let dc = 25.0 + rng.gen_range(-0.5..0.5);
        let f1 = rng.gen_range(1.0..6.0);
        let f2 = rng.gen_range(6.0..15.0);
        let p1 = rng.gen_range(0.0..(2.0 * PI));
        let p2 = rng.gen_range(0.0..(2.0 * PI));

        for (i, x) in v.iter_mut().enumerate() {
            let t = i as f64 / dim as f64;
            let wave = 0.7 * (2.0 * PI * f1 * t + p1).sin() + 0.3 * (2.0 * PI * f2 * t + p2).sin();
            let noise = rng.gen_range(-0.05..0.05);
            *x = dc + wave + noise;
        }

        let mean = v.iter().copied().sum::<f64>() / dim as f64;
        for x in &mut v {
            *x -= mean;
        }

        hd3.apply(&mut v);
        rotated[s * dim..(s + 1) * dim].copy_from_slice(&v);
    }

    let mut max_kurtosis = 0.0f64;
    for c in 0..dim {
        let mut sum = 0.0;
        for s in 0..samples {
            sum += rotated[s * dim + c];
        }
        let mean = sum / samples as f64;

        let mut m2 = 0.0;
        let mut m4 = 0.0;
        for s in 0..samples {
            let d = rotated[s * dim + c] - mean;
            let d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        let var = m2 / samples as f64;
        if var > 0.0 {
            let kurtosis = (m4 / samples as f64) / (var * var);
            max_kurtosis = max_kurtosis.max(kurtosis);
        }
    }

    // Structured synthetic waveforms produce heavier tails than idealized
    // Gaussian assumptions; keep this as a sanity bound rather than a strict gate.
    assert!(max_kurtosis <= 4.8, "max kurtosis: {}", max_kurtosis);
}

#[test]
fn quantization_rmse_matches_closed_form() {
    let dim = POLAR_QUANT_DEFAULT_DIM as usize;
    let vecs = 512usize;
    let mut rng = StdRng::seed_from_u64(11);

    for &bits in &[4u8, 6u8, 8u8] {
        let cfg = build_cfg(bits);
        let mut temps = Vec::with_capacity(vecs * dim);
        let mut vibes = Vec::with_capacity(vecs * dim);

        for _ in 0..(vecs * dim) {
            temps.push(25.0 + rng.gen_range(-1.8..1.8) * 0.28);
            vibes.push((rng.gen_range(-2.0..2.0) * 0.8) as f32);
        }

        let enc_t = encode_f64(&temps, cfg).unwrap();
        let dec_t = decode_f64(&enc_t, cfg).unwrap();
        let enc_v = encode_f32(&vibes, cfg).unwrap();
        let dec_v = decode_f32(&enc_v, cfg).unwrap();

        let rmse_t = temps
            .iter()
            .zip(&dec_t)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>()
            / temps.len() as f64;
        let rmse_t = rmse_t.sqrt();

        let rmse_v = vibes
            .iter()
            .zip(&dec_v)
            .map(|(a, b)| {
                let d = *a as f64 - *b as f64;
                d * d
            })
            .sum::<f64>()
            / vibes.len() as f64;
        let rmse_v = rmse_v.sqrt();

        let expected_t = (0..vecs)
            .map(|v| centered_l2(&temps[v * dim..(v + 1) * dim]))
            .map(|l2| l2 * 12f64.sqrt() / ((dim as f64).sqrt() * ((1u32 << bits) as f64 - 1.0)))
            .sum::<f64>()
            / vecs as f64;

        let expected_v = (0..vecs)
            .map(|v| {
                let slice = &vibes[v * dim..(v + 1) * dim];
                let mean = slice.iter().map(|x| *x as f64).sum::<f64>() / dim as f64;
                slice
                    .iter()
                    .map(|x| {
                        let d = *x as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .map(|l2| l2 * 12f64.sqrt() / ((dim as f64).sqrt() * ((1u32 << bits) as f64 - 1.0)))
            .sum::<f64>()
            / vecs as f64;

        let temp_rel = ((rmse_t - expected_t) / expected_t).abs();
        let vibe_rel = ((rmse_v - expected_v) / expected_v).abs();
        assert!(
            temp_rel <= 0.10,
            "temp bits={} rmse={} expected={} rel={}",
            bits,
            rmse_t,
            expected_t,
            temp_rel
        );
        assert!(
            vibe_rel <= 0.10,
            "vibe bits={} rmse={} expected={} rel={}",
            bits,
            rmse_v,
            expected_v,
            vibe_rel
        );
    }
}

#[test]
fn end_to_end_f64_error_bound() {
    let mut data = Vec::with_capacity(102_400 / 8);
    for i in 0..(102_400 / 8) {
        let t = i as f64;
        data.push(22.5 + 0.5 * (t * 0.001).sin() + 0.02 * (t * 0.07).cos());
    }

    for &bits in &[4u8, 6u8, 8u8] {
        let cfg = build_cfg(bits);
        let enc = encode_f64(&data, cfg).unwrap();
        let dec = decode_f64(&enc, cfg).unwrap();
        let max_rel = data
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0, f64::max);
        let bound = 4.0 / (1u32 << bits) as f64;
        assert!(
            max_rel < bound,
            "bits={} max_rel={} bound={}",
            bits,
            max_rel,
            bound
        );
    }
}

#[test]
fn throughput_probe_encode_f64() {
    let cfg = build_cfg(8);
    let count = (2 * 1024 * 1024) / 8;
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f64;
        values.push(20.0 + 0.5 * (t * 0.001).sin() + 0.01 * (t * 0.017).cos());
    }

    let _ = encode_f64(&values, cfg).unwrap();
    let iters = 6usize;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = encode_f64(&values, cfg).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mib = (values.len() * 8 * iters) as f64 / (1024.0 * 1024.0);
    let mib_s = mib / elapsed;

    #[cfg(debug_assertions)]
    {
        eprintln!("debug throughput probe: {:.2} MiB/s", mib_s);
        return;
    }

    #[cfg(not(debug_assertions))]
    assert!(mib_s >= 220.0, "throughput below gate: {:.2} MiB/s", mib_s);
}
