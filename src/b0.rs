//! Inhomogeneity of the main magnetic field B0: off-resonance maps for the simulation.
//!
//! For the 3D brain phantom the field is computed from the magnetic susceptibility of the air
//! in the paranasal sinuses, mastoid cells and ear canals (Δχ ≈ 9.4 ppm relative to tissue):
//! ΔB/B0 = F⁻¹{(1/3 − kz²/k²) F{Δχ}}, with B0 along the head–feet axis, as in a cylindrical
//! scanner bore. Optionally a second-order shim fitted over the brain removes the smooth part,
//! as the scanner's automatic shim does. 2D phantoms get a generic field (an air-cavity-like
//! hot spot and, unshimmed, smooth gradients).

use std::sync::OnceLock;

use rayon::prelude::*;
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::array::{fft2_inplace, Array2, C64};
use crate::constants::GYRO;
use crate::geometry::V3;
use crate::phantom::Phantom;
use crate::protocol::Inputs;
use crate::volume::Volume;

/// Susceptibility difference air – tissue [ppm]
pub const AIR_SUSCEPTIBILITY: f64 = 9.4;
/// Grid spacing of the field computation [mm] (the field of the cavities is smooth)
const GRID: f64 = 2.0;
/// Zero padding around the head [grid points]
const PAD: usize = 20;

/// Field of the 3D phantom in ppm of B0 (without and with shim)
pub struct FieldModel {
    dims: [usize; 3],
    /// world position of grid point (0, 0, 0) [mm]
    origin: V3,
    raw: Vec<f32>,
    shimmed: Vec<f32>,
}

impl FieldModel {
    /// ΔB/B0 [ppm] at world point p (trilinear; 0 outside the grid)
    pub fn ppm(&self, p: V3, shimmed: bool) -> f64 {
        let data = if shimmed { &self.shimmed } else { &self.raw };
        let [nx, ny, nz] = self.dims;
        let g = [0, 1, 2].map(|d| (p[d] - self.origin[d]) / GRID);
        let i0 = g.map(|x| x.floor());
        let t = [g[0] - i0[0], g[1] - i0[1], g[2] - i0[2]];
        let (x0, y0, z0) = (i0[0] as i64, i0[1] as i64, i0[2] as i64);
        let mut s = 0.0;
        for c in 0..8 {
            let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let (x, y, z) = (x0 + dx, y0 + dy, z0 + dz);
            if x < 0 || y < 0 || z < 0 || x >= nx as i64 || y >= ny as i64 || z >= nz as i64 {
                continue;
            }
            let w = (if dx == 1 { t[0] } else { 1.0 - t[0] }) * (if dy == 1 { t[1] } else { 1.0 - t[1] }) * (if dz == 1 { t[2] } else { 1.0 - t[2] });
            s += w * data[x as usize + nx * (y as usize + ny * z as usize)] as f64;
        }
        s
    }

    /// Compute the field of the air cavities of `vol`
    pub fn from_volume(vol: &Volume) -> FieldModel {
        let idx = |name: &str| vol.tissues.iter().position(|t| t == name);
        let bone = idx("Cortical bone");
        let brain: Vec<usize> = ["Gray matter", "White matter", "CSF"].iter().filter_map(|n| idx(n)).collect();
        // coarse grid (2 mm): fractions of bone/air and brain
        let step = [0, 1, 2].map(|d| ((GRID / vol.voxel[d]).round() as usize).max(1));
        let dims = [0, 1, 2].map(|d| vol.dims[d].div_ceil(step[d]));
        let [nx, ny, nz] = dims;
        let origin = [0, 1, 2].map(|d| vol.origin[d] + 0.5 * (step[d] as f64 - 1.0) * vol.voxel[d]);
        let n = nx * ny * nz;
        let mut dark = vec![0f32; n];
        let mut brain_frac = vec![0f32; n];
        dark.par_chunks_mut(nx).zip(brain_frac.par_chunks_mut(nx)).enumerate().for_each(|(row, (dk, br))| {
            let (y, z) = (row % ny, row / ny);
            for x in 0..nx {
                let (mut sd, mut sb, mut cnt) = (0f32, 0f32, 0f32);
                for fz in z * step[2]..((z + 1) * step[2]).min(vol.dims[2]) {
                    for fy in y * step[1]..((y + 1) * step[1]).min(vol.dims[1]) {
                        for fx in x * step[0]..((x + 1) * step[0]).min(vol.dims[0]) {
                            let i = fx + vol.dims[0] * (fy + vol.dims[1] * fz);
                            if let Some(b) = bone {
                                sd += vol.voxel_fraction(i, b);
                            }
                            for &t in &brain {
                                sb += vol.voxel_fraction(i, t);
                            }
                            cnt += 1.0;
                        }
                    }
                }
                dk[x] = sd / cnt.max(1.0);
                br[x] = sb / cnt.max(1.0);
            }
        });
        // air cavities: large dark regions (morphological opening removes the thin skull) in
        // the lower head (sinuses, mastoid cells, ear canals; thick bone of the upper skull is
        // dark in MR images as well, but contains no air)
        let mask: Vec<bool> = dark.iter().map(|&d| d > 0.5).collect();
        let mut opened = mask.clone();
        for _ in 0..2 {
            opened = morph(&opened, dims, false);
        }
        for _ in 0..2 {
            opened = morph(&opened, dims, true);
        }
        let opened: Vec<bool> = opened.iter().zip(&mask).map(|(&o, &m)| o && m).collect();
        let keep = components(&opened, dims, |members| {
            let zc = members.iter().map(|&i| (i / (nx * ny)) as f64).sum::<f64>() / members.len() as f64;
            members.len() >= 50 && origin[2] + zc * GRID < -20.0
        });
        let chi: Vec<f32> = (0..n).map(|i| if keep[i] { (AIR_SUSCEPTIBILITY as f32) * dark[i] } else { 0.0 }).collect();
        let mut raw = dipole_field(&chi, dims);
        // smooth field of the head–air boundary and the magnet (not computed here): what the
        // scanner's shim mainly corrects
        raw.par_iter_mut().enumerate().for_each(|(i, v)| {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let p = [origin[0] + x as f64 * GRID, origin[1] + y as f64 * GRID, origin[2] + z as f64 * GRID].map(|c| c / 100.0);
            let bg = 0.45 * p[2] + 0.3 * p[1] - 0.15 * p[0] + 0.5 * (p[0] * p[0] + p[1] * p[1] - 2.0 * p[2] * p[2]) + 0.25 * p[1] * p[2];
            *v += bg as f32;
        });
        // second-order shim over the brain (least squares fit of 1, x, y, z, x², y², z², xy, xz, yz)
        let basis = |i: usize| -> [f64; 10] {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let p = [origin[0] + x as f64 * GRID, origin[1] + y as f64 * GRID, origin[2] + z as f64 * GRID].map(|v| v / 100.0);
            [1.0, p[0], p[1], p[2], p[0] * p[0], p[1] * p[1], p[2] * p[2], p[0] * p[1], p[0] * p[2], p[1] * p[2]]
        };
        let mut ata = [[0f64; 10]; 10];
        let mut atb = [0f64; 10];
        for i in 0..n {
            if brain_frac[i] > 0.5 {
                let b = basis(i);
                for r in 0..10 {
                    atb[r] += b[r] * raw[i] as f64;
                    for c in 0..10 {
                        ata[r][c] += b[r] * b[c];
                    }
                }
            }
        }
        let coef = solve(ata, atb).unwrap_or([0.0; 10]);
        // without shim, the scanner still adjusts its frequency to the mean over the brain
        let (sum, cnt) = (0..n).filter(|&i| brain_frac[i] > 0.5).fold((0.0, 0usize), |(s, c), i| (s + raw[i] as f64, c + 1));
        let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        let shimmed: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let b = basis(i);
                raw[i] - (0..10).map(|k| coef[k] * b[k]).sum::<f64>() as f32
            })
            .collect();
        let raw = raw.iter().map(|&v| v - mean as f32).collect();
        FieldModel { dims, origin, raw, shimmed }
    }
}

/// Field model of the built-in 3D phantom (computed once, when first needed)
pub fn field_model(vol: &Volume) -> &'static FieldModel {
    static MODEL: OnceLock<FieldModel> = OnceLock::new();
    MODEL.get_or_init(|| FieldModel::from_volume(vol))
}

/// Erosion (`dilate = false`) or dilation with the 3×3×3 neighbourhood
fn morph(m: &[bool], dims: [usize; 3], dilate: bool) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    (0..m.len())
        .into_par_iter()
        .map(|i| {
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            let mut any = false;
            let mut all = true;
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        let v = a >= 0
                            && b >= 0
                            && c >= 0
                            && a < nx as i64
                            && b < ny as i64
                            && c < nz as i64
                            && m[a as usize + nx * (b as usize + ny * c as usize)];
                        any |= v;
                        all &= v;
                    }
                }
            }
            if dilate {
                any
            } else {
                all
            }
        })
        .collect()
}

/// Connected components (26-neighbourhood) of `m`; returns the union of those accepted by `keep`
fn components(m: &[bool], dims: [usize; 3], keep: impl Fn(&[usize]) -> bool) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let mut seen = vec![false; m.len()];
    let mut out = vec![false; m.len()];
    for start in 0..m.len() {
        if !m[start] || seen[start] {
            continue;
        }
        let mut members = vec![start];
        seen[start] = true;
        let mut k = 0;
        while k < members.len() {
            let i = members[k];
            k += 1;
            let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        if a < 0 || b < 0 || c < 0 || a >= nx as i64 || b >= ny as i64 || c >= nz as i64 {
                            continue;
                        }
                        let j = a as usize + nx * (b as usize + ny * c as usize);
                        if m[j] && !seen[j] {
                            seen[j] = true;
                            members.push(j);
                        }
                    }
                }
            }
        }
        if keep(&members) {
            for i in members {
                out[i] = true;
            }
        }
    }
    out
}

/// Field [ppm] of the susceptibility distribution `chi` [ppm] on the grid: dipole convolution
/// in k-space with zero padding, B0 along z
fn dipole_field(chi: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let p = dims.map(|n| n + 2 * PAD);
    let [px, py, pz] = p;
    let mut data = vec![Complex32::new(0.0, 0.0); px * py * pz];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                data[(x + PAD) + px * ((y + PAD) + py * (z + PAD))].re = chi[x + dims[0] * (y + dims[1] * z)];
            }
        }
    }
    fft3(&mut data, p, false);
    let freq = |i: usize, n: usize| -> f32 { (if i <= n / 2 { i as f32 } else { i as f32 - n as f32 }) / n as f32 };
    data.par_chunks_mut(px * py).enumerate().for_each(|(z, plane)| {
        let kz = freq(z, pz);
        for y in 0..py {
            let ky = freq(y, py);
            for x in 0..px {
                let kx = freq(x, px);
                let k2 = kx * kx + ky * ky + kz * kz;
                let d = if k2 > 0.0 { 1.0 / 3.0 - kz * kz / k2 } else { 0.0 };
                plane[x + px * y] *= d;
            }
        }
    });
    fft3(&mut data, p, true);
    let norm = 1.0 / (px * py * pz) as f32;
    let mut out = vec![0f32; chi.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                out[x + dims[0] * (y + dims[1] * z)] = data[(x + PAD) + px * ((y + PAD) + py * (z + PAD))].re * norm;
            }
        }
    }
    out
}

/// Unnormalised 3D FFT (x fastest)
fn fft3(data: &mut [Complex32], dims: [usize; 3], inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    for axis in 0..3 {
        let n = dims[axis];
        let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let stride = match axis {
            0 => 1,
            1 => dims[0],
            _ => dims[0] * dims[1],
        };
        let lines = data.len() / n;
        // gather the lines of this axis, transform them and scatter them back
        let start = |l: usize| -> usize {
            match axis {
                0 => l * dims[0],
                1 => (l % dims[0]) + (l / dims[0]) * dims[0] * dims[1],
                _ => l,
            }
        };
        let mut buf = vec![Complex32::new(0.0, 0.0); lines * n];
        buf.par_chunks_mut(n).enumerate().for_each(|(l, b)| {
            let s = start(l);
            for (k, v) in b.iter_mut().enumerate() {
                *v = data[s + k * stride];
            }
            fft.process(b);
        });
        for (l, b) in buf.chunks(n).enumerate() {
            let s = start(l);
            for (k, v) in b.iter().enumerate() {
                data[s + k * stride] = *v;
            }
        }
    }
}

/// Gaussian elimination with partial pivoting
fn solve(mut a: [[f64; 10]; 10], mut b: [f64; 10]) -> Option<[f64; 10]> {
    for c in 0..10 {
        let p = (c..10).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..10 {
            let f = a[r][c] / a[c][c];
            for k in c..10 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0f64; 10];
    for r in (0..10).rev() {
        x[r] = (b[r] - (r + 1..10).map(|k| a[r][k] * x[k]).sum::<f64>()) / a[r][r];
    }
    Some(x)
}

/// Off-resonance [Hz] at in-plane position (x along the image columns, y along the rows) [mm]
/// of the phantom, and its gradient through the slice [Hz/mm]
pub struct FieldSampler<'a> {
    ph: &'a Phantom,
    model: Option<&'static FieldModel>,
    hz_per_ppm: f64,
    shimmed: bool,
}

impl<'a> FieldSampler<'a> {
    pub fn new(ph: &'a Phantom, inp: &Inputs) -> FieldSampler<'a> {
        let model = ph.volume.as_ref().map(|v| field_model(v));
        FieldSampler { ph, model, hz_per_ppm: GYRO * inp.field_strength * inp.b0_strength, shimmed: inp.b0_shim }
    }

    pub fn hz(&self, x: f64, y: f64) -> f64 {
        match (self.model, &self.ph.slice) {
            (Some(m), Some(g)) => m.ppm(g.point(x, y, 0.0), self.shimmed) * self.hz_per_ppm,
            _ => generic_ppm(self.ph, x, y, self.shimmed) * self.hz_per_ppm,
        }
    }

    /// Gradient along the slice normal [Hz/mm] (0 for 2D phantoms)
    pub fn through_plane(&self, x: f64, y: f64) -> f64 {
        match (self.model, &self.ph.slice) {
            (Some(m), Some(g)) => {
                let h = 1.0;
                let (a, b) = (g.point(x, y, h), g.point(x, y, -h));
                (m.ppm(a, self.shimmed) - m.ppm(b, self.shimmed)) / (2.0 * h) * self.hz_per_ppm
            }
            _ => 0.0,
        }
    }
}

/// Generic field of 2D phantoms [ppm]: the field of an air cavity in front of the object (the
/// slice is perpendicular to B0) and, without shim, smooth gradients
fn generic_ppm(ph: &Phantom, x: f64, y: f64, shimmed: bool) -> f64 {
    let y0 = -0.42 * ph.support[0];
    let sigma = 18.0;
    let d2 = x * x + (y - y0) * (y - y0);
    let hot = -0.5 * AIR_SUSCEPTIBILITY / 3.0 * (sigma * sigma / (sigma * sigma + d2)).powf(1.5);
    if shimmed {
        hot
    } else {
        let (u, v) = (x / 100.0, y / 100.0);
        hot + 0.5 * u + 0.3 * v + 0.4 * (u * u + v * v)
    }
}

/// Field map [Hz] on a grid of `shape` pixels covering `fov` [mm] (rows, columns), centred;
/// NaN outside the head (3D phantom)
pub fn field_map(ph: &Phantom, inp: &Inputs, shape: [usize; 2], fov: [f64; 2]) -> Array2<f64> {
    let s = FieldSampler::new(ph, inp);
    let [rows, cols] = shape;
    let (c0, c1) = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
    let (d0, d1) = (fov[0] / rows as f64, fov[1] / cols as f64);
    let mut out = Array2::from_vec(rows, cols, vec![0.0; rows * cols]);
    out.data.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        for (c, v) in row.iter_mut().enumerate() {
            let (x, y) = ((c as f64 - c1) * d1, (r as f64 - c0) * d0);
            let inside = match (&ph.volume, &ph.slice) {
                (Some(vol), Some(g)) => vol.tissue_fraction(g.point(x, y, 0.0)) > 0.3,
                _ => true,
            };
            *v = if inside { s.hz(x, y) } else { f64::NAN };
        }
    });
    out
}

/// Phantom k-space split into off-resonance bins: index `t * nbins + l` is tissue `t` at the
/// frequency `freqs[l]` (linear interpolation between the bin frequencies), on the phantom k-space
/// grid cropped to `k_axes`. The phase accumulated until `t_center` (time from the echo at the
/// k-space centre, e.g. TE in gradient echoes) is contained exactly; the bins describe the phase
/// evolution relative to it, e^{2πi f (t − t_center)}.
pub struct Binned {
    /// image per tissue on the cropped grid, with the phase until t_center and the dephasing
    images: Vec<Array2<C64>>,
    /// interpolation coefficients (bin, weight) per pixel
    coefs: Vec<[(usize, f64); INTERP_POINTS]>,
    pub freqs: Vec<f64>,
    pub k_axes: [Vec<f64>; 2],
    pub t_center: f64,
}

impl Binned {
    pub fn num_tissues(&self) -> usize {
        self.images.len()
    }

    /// k-space of tissue `t` in frequency bin `l` on the cropped phantom grid
    pub fn kspace(&self, t: usize, l: usize) -> Array2<C64> {
        let mut a = self.images[t].clone();
        for (v, c) in a.data.iter_mut().zip(&self.coefs) {
            let w: f64 = c.iter().filter(|(j, _)| *j == l).map(|(_, x)| x).sum();
            *v *= w;
        }
        centered_dft(&a)
    }
}

/// Split the phantom k-space `pk` (per tissue, on the phantom grid) by the local off-resonance.
/// `k_max` [1/mm]: largest sampled |k| per dimension (the grid is cropped accordingly);
/// `t_span` [ms]: largest |t − t_center| during the readouts (determines the bin spacing);
/// `t_center` [ms]: time from the echo at the k-space centre.
/// `max_bins` limits the number of bins (memory of the resampled k-space).
pub fn bin_phantom(ph: &Phantom, inp: &Inputs, pk: &[Array2<C64>], k_max: [f64; 2], t_span: f64, t_center: f64, max_bins: usize) -> Binned {
    // crop the k-space grid to the sampled region (keeping the parity, so that the k values stay)
    let full = [pk[0].rows, pk[0].cols];
    let keep: [usize; 2] = [0, 1].map(|d| {
        let r = (k_max[d] * 1.2 * ph.support[d]).ceil() as usize + 4;
        let n = if full[d] % 2 == 1 { 2 * r + 1 } else { 2 * r + 2 };
        n.min(full[d])
    });
    let off = [0, 1].map(|d| (full[d] - keep[d]) / 2);
    let [rows, cols] = keep;
    let cropped: Vec<Array2<C64>> = pk
        .par_iter()
        .map(|a| {
            let mut b = Array2::from_vec(rows, cols, vec![C64::new(0.0, 0.0); rows * cols]);
            for r in 0..rows {
                b.data[r * cols..(r + 1) * cols].copy_from_slice(&a.data[(r + off[0]) * a.cols + off[1]..(r + off[0]) * a.cols + off[1] + cols]);
            }
            b
        })
        .collect();
    let k_axes = [0, 1].map(|d| ph.k_axes[d][off[d]..off[d] + keep[d]].to_vec());

    let pixel = [ph.support[0] / rows as f64, ph.support[1] / cols as f64];
    let s = FieldSampler::new(ph, inp);
    let (c0, c1) = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
    let n = rows * cols;
    let mut hz = vec![0f64; n];
    // image-domain factor: phase until t_center and dephasing across the slice (3D phantoms)
    let mut factor = vec![C64::new(1.0, 0.0); n];
    let thickness = ph.slice.map_or(0.0, |g| g.thickness);
    hz.par_chunks_mut(cols).zip(factor.par_chunks_mut(cols)).enumerate().for_each(|(r, (h, f))| {
        let y = (r as f64 - c0) * pixel[0];
        for c in 0..cols {
            let x = (c as f64 - c1) * pixel[1];
            h[c] = s.hz(x, y);
            let mut v = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * h[c] * t_center * 1e-3);
            if t_center != 0.0 {
                // in the image plane, a field gradient g shifts the local echo in k-space by
                // g·t: beyond the sampled k-space the signal is lost (dephased); the phase
                // itself is applied exactly, which also stops aliasing on the cropped grid
                let t = t_center * 1e-3;
                let gx = (s.hz(x + 0.5 * pixel[1], y) - s.hz(x - 0.5 * pixel[1], y)) / pixel[1];
                let gy = (s.hz(x, y + 0.5 * pixel[0]) - s.hz(x, y - 0.5 * pixel[0])) / pixel[0];
                v *= echo_window(gx * t, k_max[1]) * echo_window(gy * t, k_max[0]);
                // across the slice profile (3D phantoms): ∫ e^{2πi g z t} dz / d = sinc(π g d t)
                if thickness > 0.0 {
                    let a = std::f64::consts::PI * s.through_plane(x, y) * thickness * t;
                    v *= if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                }
            }
            f[c] = v;
        }
    });
    let mut images: Vec<Array2<C64>> = cropped.par_iter().map(centered_idft).collect();
    // frequency range where there is signal (without the extreme values right at air cavities,
    // where the signal is dephased anyway; they are assigned to the outermost bins)
    let mag: Vec<f64> = (0..n).map(|i| images.iter().map(|im| im.data[i].norm()).sum()).collect();
    let max = mag.iter().cloned().fold(0.0, f64::max);
    let mut sig: Vec<f64> = (0..n).filter(|&i| mag[i] > 0.02 * max).map(|i| hz[i]).collect();
    let (lo, hi) = if sig.is_empty() {
        (0.0, 0.0)
    } else {
        sig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| sig[((sig.len() - 1) as f64 * f).round() as usize];
        (q(0.003), q(0.997))
    };
    // bins a quarter cycle apart over the readout time span
    let t_s = t_span * 1e-3;
    let nbins = if t_s > 0.0 { (((hi - lo) * 4.0 * t_s).ceil() as usize + 1).clamp(1, max_bins.clamp(2, MAX_BINS)) } else { 1 };
    let freqs: Vec<f64> = if nbins == 1 { vec![(lo + hi) / 2.0] } else { (0..nbins).map(|l| lo + (hi - lo) * l as f64 / (nbins - 1) as f64).collect() };
    // interpolation of e^{2πi f τ} between the bins: least-squares coefficients of the nearest
    // bins over τ ∈ [−t_span, t_span] (time-segmentation interpolators)
    let coefs: Vec<[(usize, f64); INTERP_POINTS]> = hz.par_iter().map(|&f| interpolator(&freqs, f, t_s)).collect();
    images.par_iter_mut().for_each(|im| {
        for (v, f) in im.data.iter_mut().zip(&factor) {
            *v *= f;
        }
    });
    Binned { images, coefs, freqs, k_axes, t_center }
}

/// Fraction of the signal of a voxel whose local echo is shifted by `shift` [1/mm] in k-space,
/// with the sampled k-space extending to ±`k_max` (smooth edge)
fn echo_window(shift: f64, k_max: f64) -> f64 {
    let u = shift.abs() / k_max.max(1e-9);
    if u <= 1.0 {
        1.0
    } else if u >= 1.15 {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * (u - 1.0) / 0.15).cos())
    }
}

/// Upper limit of the number of off-resonance bins
const MAX_BINS: usize = 64;
/// Number of bins used for the interpolation of one frequency
const INTERP_POINTS: usize = 6;

/// Coefficients c_l (for up to INTERP_POINTS bins l, unused entries have weight 0) such that
/// e^{2πi f τ} ≈ Σ c_l e^{2πi f_l τ} in the least-squares sense over τ ∈ [−t, t]:
/// G c = b with G_jk = sinc(2π (f_j − f_k) t), b_j = sinc(2π (f − f_j) t)
fn interpolator(freqs: &[f64], f: f64, t: f64) -> [(usize, f64); INTERP_POINTS] {
    const P: usize = INTERP_POINTS;
    let nb = freqs.len();
    let mut out = [(0usize, 0.0f64); P];
    if nb == 1 || t <= 0.0 {
        out[0] = (0, 1.0);
        return out;
    }
    let step = freqs[1] - freqs[0];
    let pos = ((f - freqs[0]) / step).clamp(0.0, (nb - 1) as f64);
    let m = nb.min(P);
    let j0 = ((pos.floor() as i64) - (m as i64 / 2 - 1)).clamp(0, (nb - m) as i64) as usize;
    let fc = f.clamp(freqs[0], freqs[nb - 1]);
    let sinc = |x: f64| {
        let a = 2.0 * std::f64::consts::PI * x * t;
        if a.abs() < 1e-9 {
            1.0
        } else {
            a.sin() / a
        }
    };
    let mut g = [[0f64; P]; P];
    let mut b = [0f64; P];
    for j in 0..m {
        for k in 0..m {
            g[j][k] = sinc(freqs[j0 + j] - freqs[j0 + k]) + if j == k { 1e-9 } else { 0.0 };
        }
        b[j] = sinc(fc - freqs[j0 + j]);
    }
    // Gaussian elimination (symmetric positive definite)
    for c in 0..m {
        for r in c + 1..m {
            let q = g[r][c] / g[c][c];
            for k in c..m {
                g[r][k] -= q * g[c][k];
            }
            b[r] -= q * b[c];
        }
    }
    let mut x = [0f64; P];
    for r in (0..m).rev() {
        x[r] = (b[r] - (r + 1..m).map(|k| g[r][k] * x[k]).sum::<f64>()) / g[r][r];
    }
    for j in 0..m {
        out[j] = (j0 + j, x[j]);
    }
    out
}

/// Twiddle factors of the centred DFT: K_p = Σ_m x_m e^{-2πi (p-c)(m-c)/N}, c = (N-1)/2
fn twiddles(n: usize, inverse: bool) -> (Vec<C64>, Vec<C64>) {
    let c = (n as f64 - 1.0) / 2.0;
    let s = if inverse { -1.0 } else { 1.0 };
    let tau = 2.0 * std::f64::consts::PI;
    let pre: Vec<C64> = (0..n).map(|m| C64::from_polar(1.0, s * tau * c * m as f64 / n as f64)).collect();
    let post: Vec<C64> = (0..n).map(|p| C64::from_polar(1.0, s * (tau * c * p as f64 / n as f64 - tau * c * c / n as f64))).collect();
    (pre, post)
}

fn centered_transform(a: &Array2<C64>, inverse: bool) -> Array2<C64> {
    let (pre0, post0) = twiddles(a.rows, inverse);
    let (pre1, post1) = twiddles(a.cols, inverse);
    let mut b = a.clone();
    for r in 0..a.rows {
        for c in 0..a.cols {
            b.data[r * a.cols + c] *= pre0[r] * pre1[c];
        }
    }
    fft2_inplace(&mut b, inverse);
    let scale = if inverse { 1.0 / (a.rows * a.cols) as f64 } else { 1.0 };
    for r in 0..a.rows {
        for c in 0..a.cols {
            b.data[r * a.cols + c] *= post0[r] * post1[c] * scale;
        }
    }
    b
}

/// Centred DFT (image → k-space on the phantom grid)
pub fn centered_dft(a: &Array2<C64>) -> Array2<C64> {
    centered_transform(a, false)
}

/// Inverse of `centered_dft`
pub fn centered_idft(a: &Array2<C64>) -> Array2<C64> {
    centered_transform(a, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_dft_roundtrip() {
        for (r, c) in [(6, 5), (7, 8)] {
            let a = Array2::from_vec(r, c, (0..r * c).map(|i| C64::new((i as f64).sin(), (i as f64 * 0.3).cos())).collect());
            let b = centered_idft(&centered_dft(&a));
            let err: f64 = a.data.iter().zip(&b.data).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
            assert!(err < 1e-12, "{err}");
        }
    }

    #[test]
    fn interpolators_reproduce_the_phase_evolution() {
        let t = 0.02; // ±20 ms
                      // bins as chosen by `bin_phantom`: a quarter cycle apart over the time span
        let freqs: Vec<f64> = (0..33).map(|l| -200.0 + 12.5 * l as f64).collect();
        let mut worst: f64 = 0.0;
        for k in 0..=80 {
            let f = -200.0 + 5.0 * k as f64;
            let c = interpolator(&freqs, f, t);
            for s in 0..=40 {
                let tau = -t + 2.0 * t * s as f64 / 40.0;
                let exact = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * f * tau);
                let approx: C64 = c.iter().map(|&(l, w)| C64::from_polar(w, 2.0 * std::f64::consts::PI * freqs[l] * tau)).sum();
                worst = worst.max((exact - approx).norm());
            }
        }
        assert!(worst < 0.03, "worst interpolation error {worst}");
    }

    #[test]
    fn air_cavity_fields() {
        let vol = crate::volume::builtin(crate::volume::BRAIN3D).unwrap();
        let m = field_model(&vol);
        // inferior frontal lobe above the sinuses vs. the centre of the brain
        let orbitofrontal = m.ppm([0.0, 30.0, -20.0], true).abs();
        let centre = m.ppm([0.0, -15.0, 20.0], true).abs();
        assert!(orbitofrontal > 0.3, "orbitofrontal field {orbitofrontal} ppm");
        assert!(orbitofrontal > 3.0 * centre, "{orbitofrontal} vs {centre}");
        // the shim reduces the field in the brain
        let mut raw = 0.0;
        let mut shim = 0.0;
        for p in [[0.0, -15.0, 20.0], [40.0, -20.0, 30.0], [-40.0, 10.0, 0.0], [0.0, -60.0, 10.0]] {
            raw += m.ppm(p, false).powi(2);
            shim += m.ppm(p, true).powi(2);
        }
        assert!(shim < raw, "shimmed {shim} vs raw {raw}");
    }
}
