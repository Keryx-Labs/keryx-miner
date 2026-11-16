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
