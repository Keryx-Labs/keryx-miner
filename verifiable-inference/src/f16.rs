//! IEEE half precision conversions.

/// Half bits to f32 (exact).
pub fn to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 to half bits, round to nearest even.
pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut out = (m >> shift) as u16;
        if rest > half || (rest == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out;
    }
    let mut out = ((e as u32) << 10 | (man >> 13)) as u16;
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (out & 1) == 1) {
        out += 1;
    }
    sign | out
}

/// Round an f32 through half precision.
pub fn round(x: f32) -> f32 {
    to_f32(from_f32(x))
}

pub fn read(b: &[u8], off: usize) -> f32 {
    to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}
