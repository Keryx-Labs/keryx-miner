//! Recomputation of one matmul output element.

use crate::{q8, quant};

/// Recomputed output and its error scale.
#[derive(Debug, Clone, Copy)]
pub struct Recomputed {
    pub y_hat: f64,
    pub scale: f64,
}

/// Output element of weight row `row` applied to activation row `x_leaf`.
pub fn canonical_dot(wtype: u32, row: &[u8], k: usize, x_leaf: &[u8], layout: u8) -> Option<Recomputed> {
    let w = quant::dequant_row(wtype, row, k)?;
    let (x, s) = q8::decode(x_leaf, layout, k)?;
    let scale: f64 = w.iter().zip(&x).map(|(&a, &b)| (a as f64 * b as f64).abs()).sum();

    if layout == q8::MMQ_DS4 && (wtype == quant::Q4_K || wtype == quant::Q5_K) {
        let (w_eff, b_h) = quant::k_tile_form(wtype, row, k)?;
        let s = s?;
        let main: f64 = w_eff.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
        let min: f64 = s.iter().zip(&b_h).map(|(&a, &b)| a as f64 * b as f64).sum();
        return Some(Recomputed { y_hat: main - min, scale });
    }

    if layout == q8::MMQ_D2S6 && wtype == quant::Q2_K {
        let (q, a, m) = quant::q2k_tile_form(row, k)?;
        let sums = q8::d2s6_sums(x_leaf, k)?;
        let mut y_hat = 0f64;
        for g in 0..k / 16 {
            let xs = &x[16 * g..16 * g + 16];
            let dot: f64 = q[16 * g..16 * g + 16].iter().zip(xs).map(|(&qv, &xv)| qv as f64 * xv as f64).sum();
            let s = if g % 8 < 6 { sums[g / 8 * 6 + g % 8] as f64 } else { xs.iter().map(|&v| v as f64).sum() };
            y_hat += a[g] as f64 * dot - m[g] as f64 * s;
        }
        return Some(Recomputed { y_hat, scale });
    }

    let mut y_hat: f64 = w.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
    if layout == q8::MMVQ
        && let Some((dw, off)) = quant::offset_block_scales(wtype, row, k)
    {
        let s = s?;
        for b in 0..k / 32 {
            let sq: f64 = x[32 * b..32 * b + 32].iter().map(|&v| v as f64).sum();
            y_hat += off * dw[b] as f64 * (sq - s[b] as f64);
        }
    }
    Some(Recomputed { y_hat, scale })
}

pub fn accept(y: f32, r: &Recomputed, eps: f64) -> bool {
    (y as f64 - r.y_hat).abs() <= eps * r.scale
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case {
        wtype: u32,
        layout: u8,
        k: usize,
        row: Vec<u8>,
        x: Vec<u8>,
        y: f32,
        wrong: Vec<u8>,
    }

    fn cases() -> Vec<Case> {
        let b: &[u8] = include_bytes!("../testdata/v4-products.bin");
        let mut out = Vec::new();
        let mut o = 0usize;
        let mut u32_at = |o: &mut usize| {
            let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
            *o += 4;
            v
        };
        while o < b.len() {
            let wtype = u32_at(&mut o);
            let layout = b[o];
            o += 1;
            let k = u32_at(&mut o) as usize;
            let n = u32_at(&mut o) as usize;
            let row = b[o..o + n].to_vec();
            o += n;
            let n = u32_at(&mut o) as usize;
            let x = b[o..o + n].to_vec();
            o += n;
            let y = f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            o += 4;
            let n = u32_at(&mut o) as usize;
            let wrong = b[o..o + n].to_vec();
            o += n;
            out.push(Case { wtype, layout, k, row, x, y, wrong });
        }
        out
    }

    #[test]
    fn real_gpu_products_of_every_network_model_form() {
        let cases = cases();
        let mut forms: Vec<(u32, u8)> = cases.iter().map(|c| (c.wtype, c.layout)).collect();
        forms.sort_unstable();
        forms.dedup();
        assert_eq!(forms.len(), 7);
        for c in cases.iter() {
            let eps = crate::verify::eps_for(c.wtype);
            let honest = canonical_dot(c.wtype, &c.row, c.k, &c.x, c.layout).unwrap();
            assert!(accept(c.y, &honest, eps), "wtype {} layout {}: {} vs {}", c.wtype, c.layout, c.y, honest.y_hat);
            let wrong = canonical_dot(c.wtype, &c.wrong, c.k, &c.x, c.layout).unwrap();
            assert!(!accept(c.y, &wrong, eps), "wtype {} layout {} accepts other weights", c.wtype, c.layout);
        }
    }
}
