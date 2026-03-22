// ═══════════════════════════════════════════════════════════════════════════════
// Byte-Level Delta Encoding
// ═══════════════════════════════════════════════════════════════════════════════
//
// Simple wrapping delta on raw bytes: out[i] = in[i].wrapping_sub(in[i-1]).
// Designed for post-shuffle byte lanes where consecutive values in the same
// lane change slowly (e.g. exponent bytes of float sequences). Produces long
// runs of zeros and near-zeros that compress well with LZ77+FSE.
//
// Unlike the typed delta in `delta.rs` (zigzag + varint for i32/i64), this
// operates directly on bytes with zero overhead — same input/output size,
// no headers, no alignment requirements.

/// Byte-delta encode in-place: `out[0] = in[0]`, `out[i] = in[i] - in[i-1]`.
/// Output has the same length as input.
#[inline]
pub fn encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(data[i].wrapping_sub(data[i - 1]));
    }
    out
}

/// Byte-delta decode (cumulative sum): `out[0] = in[0]`, `out[i] = out[i-1] + in[i]`.
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn decode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        let prev = *out.last().unwrap();
        out.push(prev.wrapping_add(data[i]));
    }
    out
}

/// Byte-delta encode into an existing buffer, returning the slice written.
/// Caller must ensure `output` has enough capacity.
#[inline]
pub fn encode_into(data: &[u8], output: &mut Vec<u8>) {
    output.clear();
    if data.is_empty() {
        return;
    }
    output.reserve(data.len());
    output.push(data[0]);
    for i in 1..data.len() {
        output.push(data[i].wrapping_sub(data[i - 1]));
    }
}

/// Byte-delta decode into an existing buffer.
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn decode_into(data: &[u8], output: &mut Vec<u8>) {
    output.clear();
    if data.is_empty() {
        return;
    }
    output.reserve(data.len());
    output.push(data[0]);
    for i in 1..data.len() {
        let prev = *output.last().unwrap();
        output.push(prev.wrapping_add(data[i]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(&[])), Vec::<u8>::new());
    }

    #[test]
    fn roundtrip_single() {
        let data = vec![42u8];
        assert_eq!(decode(&encode(&data)), data);
    }

    #[test]
    fn roundtrip_constant() {
        // Constant sequence → first byte + all zeros.
        let data = vec![7u8; 1000];
        let encoded = encode(&data);
        assert_eq!(encoded[0], 7);
        assert!(encoded[1..].iter().all(|&b| b == 0));
        assert_eq!(decode(&encoded), data);
    }

    #[test]
    fn roundtrip_incrementing() {
        // 0,1,2,...,255,0,1,2,...  → first byte + all 1s (wrapping).
        let data: Vec<u8> = (0..1000).map(|i| (i & 0xFF) as u8).collect();
        let encoded = encode(&data);
        assert_eq!(encoded[0], 0);
        assert!(encoded[1..].iter().all(|&b| b == 1));
        assert_eq!(decode(&encoded), data);
    }

    #[test]
    fn roundtrip_random_ish() {
        let data: Vec<u8> = (0..10_000u64)
            .map(|i| (i.wrapping_mul(7919) ^ (i >> 3)) as u8)
            .collect();
        assert_eq!(decode(&encode(&data)), data);
    }

    #[test]
    fn wrapping_behavior() {
        // 255 → 0 should produce delta of 1, not underflow.
        let data = vec![255u8, 0, 1, 2];
        let encoded = encode(&data);
        assert_eq!(encoded, vec![255, 1, 1, 1]);
        assert_eq!(decode(&encoded), data);
    }

    #[test]
    fn roundtrip_into_variants() {
        let data: Vec<u8> = (0..5000).map(|i| (i * 37 + 11) as u8).collect();
        let mut enc_buf = Vec::new();
        let mut dec_buf = Vec::new();
        encode_into(&data, &mut enc_buf);
        decode_into(&enc_buf, &mut dec_buf);
        assert_eq!(dec_buf, data);
    }

    #[test]
    fn exponent_byte_lane_simulation() {
        // Simulate what exponent bytes of f64 look like after shuffle:
        // slowly changing values (e.g. 0x40 repeated with occasional 0x41).
        let mut data = vec![0x40u8; 1000];
        for i in (100..200).step_by(3) {
            data[i] = 0x41;
        }
        let encoded = encode(&data);
        // Count zeros — should be very high for slowly-changing lane.
        let zero_count = encoded.iter().filter(|&&b| b == 0).count();
        assert!(
            zero_count > 900,
            "expected >900 zeros in delta of slow lane, got {}",
            zero_count
        );
        assert_eq!(decode(&encoded), data);
    }
}
