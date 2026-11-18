//! 2D non-uniform FFT (type 1 / type 2) with an "exponential of semicircle" kernel,
//! following the FINUFFT algorithm (2× upsampling, kernel width 7 → ~1e-6 accuracy).
//!
//! Conventions match `mrinufft`'s finufft operator as used by SpinSight:
//! * modes are centred (array index i ↔ frequency i − ⌊N/2⌋, i.e. fftshifted order),
//! * sample coordinates are in radians, in [−π, π),
//! * type 2 (`forward`): c_j = Σ_k f_k e^{−i k·x_j},
//! * type 1 (`adjoint`): f_k = Σ_j c_j e^{+i k·x_j}.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::array::{fft2_inplace, Array2, C64};

const NSPREAD: usize = 7;
const BETA_PER_WIDTH: f64 = 2.30;

pub struct Nufft2d {
    pub shape: [usize; 2],
    nf: [usize; 2],
    beta: f64,
    /// 1 / Φ(k) for each mode (array index order) per dimension
    corr: [Vec<f64>; 2],
    /// sample coordinates in fine-grid units
    s: Vec<[f64; 2]>,
}

/// smallest even number ≥ n whose only prime factors are 2, 3 and 5
fn next235even(n: usize) -> usize {
    let mut m = n.max(2);
    if m % 2 == 1 {
        m += 1;
    }
    loop {
        let mut r = m;
        for p in [2, 3, 5] {
            while r % p == 0 {
                r /= p;
            }
        }
        if r == 1 {
            return m;
        }
        m += 2;
    }
}

#[inline]
fn kernel(z: f64, beta: f64) -> f64 {
    let x = 2.0 * z / NSPREAD as f64;
    let t = 1.0 - x * x;
    if t < 0.0 {
        0.0
    } else {
        (beta * (t.sqrt() - 1.0)).exp()
    }
}

/// Gauss–Legendre nodes and weights on [-1, 1]
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

impl Nufft2d {
    /// `points` in radians ([−π, π)), `shape` = number of modes per dimension
    pub fn new(points: &[[f64; 2]], shape: [usize; 2]) -> Self {
        let beta = BETA_PER_WIDTH * NSPREAD as f64;
        let nf = [next235even((2 * shape[0]).max(2 * NSPREAD)), next235even((2 * shape[1]).max(2 * NSPREAD))];
        // Fourier transform of the kernel by quadrature over its support [-w/2, w/2]
        let (gx, gw) = gauss_legendre(4 * NSPREAD + 40);
        let half = NSPREAD as f64 / 2.0;
        let phi_hat = |k: f64, n: usize| -> f64 {
            let mut s = 0.0;
            for (x, w) in gx.iter().zip(gw.iter()) {
                let u = x * half;
                s += w * half * kernel(u, beta) * (2.0 * PI * k * u / n as f64).cos();
            }
            s
        };
        let corr = [0, 1].map(|d| (0..shape[d]).map(|i| 1.0 / phi_hat(i as f64 - (shape[d] / 2) as f64, nf[d])).collect::<Vec<f64>>());
        let s = points.iter().map(|p| [p[0] * nf[0] as f64 / (2.0 * PI), p[1] * nf[1] as f64 / (2.0 * PI)]).collect();
        Nufft2d { shape, nf, beta, corr, s }
    }

    pub fn num_points(&self) -> usize {
        self.s.len()
    }

    #[inline]
    fn weights(&self, s: f64, n: usize) -> ([f64; NSPREAD], [usize; NSPREAD]) {
        let l0 = (s - NSPREAD as f64 / 2.0).ceil();
        let mut w = [0.0; NSPREAD];
        let mut idx = [0usize; NSPREAD];
        for t in 0..NSPREAD {
            let l = l0 + t as f64;
            w[t] = kernel(s - l, self.beta);
            idx[t] = (l as i64).rem_euclid(n as i64) as usize;
        }
        (w, idx)
    }

    fn mode_to_grid(&self, i: usize, d: usize) -> usize {
        let k = i as i64 - (self.shape[d] / 2) as i64;
        k.rem_euclid(self.nf[d] as i64) as usize
    }

    /// Type 2 for a batch of images (all with `shape`): returns one sample vector per image
    pub fn forward_batch(&self, images: &[&Array2<C64>]) -> Vec<Vec<C64>> {
        let nb = images.len();
        let grids: Vec<Array2<C64>> = images
            .par_iter()
            .map(|img| {
                assert_eq!(img.shape(), self.shape);
                let mut g: Array2<C64> = Array2::zeros(self.nf[0], self.nf[1]);
                for i in 0..self.shape[0] {
                    let gi = self.mode_to_grid(i, 0);
                    for j in 0..self.shape[1] {
                        let gj = self.mode_to_grid(j, 1);
                        g.data[gi * self.nf[1] + gj] = *img.get(i, j) * (self.corr[0][i] * self.corr[1][j]);
                    }
                }
                fft2_inplace(&mut g, false);
                g
            })
            .collect();
        let m = self.s.len();
        let mut out = vec![C64::new(0.0, 0.0); m * nb];
        out.par_chunks_mut(nb).enumerate().for_each(|(p, o)| {
            let (w0, i0) = self.weights(self.s[p][0], self.nf[0]);
            let (w1, i1) = self.weights(self.s[p][1], self.nf[1]);
            for (b, g) in grids.iter().enumerate() {
                let mut acc = C64::new(0.0, 0.0);
                for t0 in 0..NSPREAD {
                    let row = &g.data[i0[t0] * self.nf[1]..(i0[t0] + 1) * self.nf[1]];
                    let mut racc = C64::new(0.0, 0.0);
                    for t1 in 0..NSPREAD {
                        racc += row[i1[t1]] * w1[t1];
                    }
                    acc += racc * w0[t0];
                }
                o[b] = acc;
            }
        });
        (0..nb).map(|b| (0..m).map(|p| out[p * nb + b]).collect()).collect()
    }

    /// Type 1 (adjoint): f_k = Σ_j c_j e^{+i k·x_j}
    pub fn adjoint(&self, c: &[C64]) -> Array2<C64> {
        assert_eq!(c.len(), self.s.len());
        let nf1 = self.nf[1];
        // spread in parallel chunks into private grids, then sum
        let chunks = rayon::current_num_threads().clamp(1, 8);
        let chunk_len = c.len().div_ceil(chunks).max(1);
        let grids: Vec<Vec<C64>> = (0..chunks)
            .into_par_iter()
            .map(|ch| {
                let mut g = vec![C64::new(0.0, 0.0); self.nf[0] * nf1];
                let start = ch * chunk_len;
                let end = ((ch + 1) * chunk_len).min(c.len());
                for p in start..end {
                    let (w0, i0) = self.weights(self.s[p][0], self.nf[0]);
                    let (w1, i1) = self.weights(self.s[p][1], nf1);
                    for t0 in 0..NSPREAD {
                        let v = c[p] * w0[t0];
                        let row = &mut g[i0[t0] * nf1..(i0[t0] + 1) * nf1];
                        for t1 in 0..NSPREAD {
                            row[i1[t1]] += v * w1[t1];
                        }
                    }
                }
                g
            })
            .collect();
        let mut g = Array2::from_vec(self.nf[0], nf1, vec![C64::new(0.0, 0.0); self.nf[0] * nf1]);
        for part in &grids {
            g.data.par_iter_mut().zip(part.par_iter()).for_each(|(a, b)| *a += *b);
        }
        fft2_inplace(&mut g, true);
        let mut f = Array2::zeros(self.shape[0], self.shape[1]);
        for i in 0..self.shape[0] {
            let gi = self.mode_to_grid(i, 0);
            for j in 0..self.shape[1] {
                let gj = self.mode_to_grid(j, 1);
                f.data[i * self.shape[1] + j] = g.data[gi * nf1 + gj] * (self.corr[0][i] * self.corr[1][j]);
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_type2(img: &Array2<C64>, x: [f64; 2]) -> C64 {
        let mut s = C64::new(0.0, 0.0);
        for i in 0..img.rows {
            for j in 0..img.cols {
                let k0 = i as f64 - (img.rows / 2) as f64;
                let k1 = j as f64 - (img.cols / 2) as f64;
                s += img.get(i, j) * C64::from_polar(1.0, -(k0 * x[0] + k1 * x[1]));
            }
        }
        s
    }

    #[test]
    fn type2_matches_direct_sum() {
        let shape = [13, 10];
        let img = Array2::from_vec(13, 10, (0..130).map(|i| C64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos())).collect());
        let pts: Vec<[f64; 2]> = (0..50).map(|i| [-PI + 0.123 * i as f64, PI - 0.1 - 0.117 * i as f64]).collect();
        let op = Nufft2d::new(&pts, shape);
        let out = &op.forward_batch(&[&img])[0];
        let norm: f64 = img.data.iter().map(|v| v.norm()).sum();
        for (p, x) in pts.iter().enumerate() {
            let d = direct_type2(&img, *x);
            assert!((out[p] - d).norm() / norm < 1e-6, "{} vs {}", out[p], d);
        }
    }

    #[test]
    fn type1_is_adjoint_of_type2() {
        let shape = [8, 11];
        let pts: Vec<[f64; 2]> = (0..40).map(|i| [((i * 7) % 40) as f64 / 40.0 * 2.0 * PI - PI, (i as f64 * 0.157) - PI]).collect();
        let op = Nufft2d::new(&pts, shape);
        let c: Vec<C64> = (0..40).map(|i| C64::new((i as f64).cos(), (i as f64 * 0.3).sin())).collect();
        let f = op.adjoint(&c);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let k0 = i as f64 - (shape[0] / 2) as f64;
                let k1 = j as f64 - (shape[1] / 2) as f64;
                let mut d = C64::new(0.0, 0.0);
                for (p, x) in pts.iter().enumerate() {
                    d += c[p] * C64::from_polar(1.0, k0 * x[0] + k1 * x[1]);
                }
                assert!((f.get(i, j) - d).norm() < 1e-5 * 40.0, "{} vs {}", f.get(i, j), d);
            }
        }
    }
}
