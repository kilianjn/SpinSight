//! k-space simulation and image reconstruction (port of `nodes/kspace_simulation.py`,
//! `kspace_processing.py`, `image_reconstruction.py`, `kspace_plot.py`, `signal_curves`).
//!
//! Expensive intermediate results (phantom k-space resampled onto the sampling trajectory, the
//! radial gridding operator with its Voronoi density compensation, the noise realisation) are
//! cached and only recomputed when the parameters they depend on change.

use std::collections::hash_map::DefaultHasher;
use std::f64::consts::PI;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::array::{Array2, C64};
use crate::constants::*;
use crate::nufft::Nufft2d;
use crate::phantom::{Phantom, PhantomKspace};
use crate::protocol::{component_pd_t1w, pyround, Inputs, Protocol};
use crate::recon::{crop_or_pad, fft_shifted, get_k_axis, homodyne_weights, ifft_shifted, radial_tukey, sinc_interpolate, sinc_kernel};
use crate::sequence::{get_signal, Waveform};

/// Result of one simulation run. Display operators (image/k-space mode, exponents) and the
/// signal curves of the displayed shot are applied afterwards, see [`display_image`],
/// [`display_kspace`] and [`signal_curves`], so they can change without re-simulating.
#[derive(Debug, Clone, Default)]
pub struct SimOutput {
    /// reconstructed complex image, shape recon_matrix
    pub image_array: Array2<C64>,
    /// FOV [mm] per dimension (dim 0 = vertical)
    pub fov: [f64; 2],
    /// gridded (measured) k-space and its axes [1/mm]
    pub gridded: Array2<C64>,
    pub grid_axes: [Vec<f64>; 2],
    /// processed (homodyne, apodized, zero-filled) k-space and its axes
    pub zerofilled: Array2<C64>,
    pub processed_axes: [Vec<f64>; 2],
    /// measured k-space samples (n0, n1, na)
    pub measured: Vec<C64>,
    pub measured_shape: [usize; 3],
    /// time after excitation per (i, j) sample position
    pub tae: Vec<f64>,
    pub timings: Vec<(&'static str, f64)>,
}

/// Image values for display (`OPERATORS[image_mode](image_array)`)
pub fn display_image(out: &SimOutput, image_mode: &str) -> Array2<f64> {
    let op = Operator::from_name(image_mode);
    out.image_array.map(|z| op.apply(*z))
}

/// k-space values and axes for display (`kspace` node)
pub fn display_kspace(out: &SimOutput, kspace_mode: &str, exponent: f64, processed: bool) -> (Array2<f64>, [Vec<f64>; 2]) {
    let (src, axes) = if processed { (&out.zerofilled, &out.processed_axes) } else { (&out.gridded, &out.grid_axes) };
    let op = Operator::from_name(kspace_mode);
    let e = if op == Operator::Phase { 1.0 } else { exponent };
    let k = src.map(|z| {
        let v = op.apply(*z);
        if v == 0.0 {
            0.0
        } else {
            v.signum() * v.abs().powf(e)
        }
    });
    (k, axes.clone())
}

/// Signal curves of the displayed shot for the sequence diagram (`signal_curves`)
pub fn signal_curves(p: &Protocol, out: &SimOutput, signal_exponent: f64) -> Vec<Waveform> {
    let [n0, n1, na] = out.measured_shape;
    if out.measured.is_empty() || n0 * n1 * na != out.measured.len() || out.tae.len() != n0 * n1 {
        return vec![];
    }
    let n_read = if p.freq_dir == 0 { n0 } else { n1 };
    let n_pe = if p.freq_dir == 0 { n1 } else { n0 };
    let max_abs = out.measured.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    let scale = if max_abs > 0.0 { 1.0 / max_abs } else { 0.0 };
    let spoke = if p.is_radial { p.shot.min(na - 1) } else { 0 };
    let mut curves = Vec::new();
    for rf in 0..p.turbo_factor {
        for gr in 0..p.epi_factor {
            let ky = p.pe(p.shot, rf, gr);
            if ky >= n_pe {
                continue;
            }
            let mut waveform = Vec::with_capacity(n_read);
            let mut t = Vec::with_capacity(n_read);
            for r in 0..n_read {
                let (i, j) = if p.freq_dir == 0 { (r, ky) } else { (ky, r) };
                waveform.push(out.measured[(i * n1 + j) * na + spoke].re);
                t.push(out.tae[i * n1 + j]);
            }
            curves.push(get_signal(&waveform, &t, scale, signal_exponent, "sampling"));
        }
    }
    curves
}

/// Simulation key: the inputs that influence the simulated data (display-only inputs reset)
pub fn simulation_key(inp: &Inputs) -> Inputs {
    let d = Inputs::default();
    Inputs {
        show_fov: d.show_fov,
        show_te_span: false,
        show_tr_span: false,
        show_ti_span: false,
        reference_tissue: d.reference_tissue.clone(),
        image_mode: d.image_mode.clone(),
        show_processed_kspace: d.show_processed_kspace,
        kspace_exponent: d.kspace_exponent,
        kspace_mode: d.kspace_mode.clone(),
        shot: 0,
        signal_exponent: d.signal_exponent,
        ..inp.clone()
    }
}

struct CompsCache {
    key: u64,
    /// per tissue, flattened (n0, n1, na)
    comps: Arc<Vec<Vec<C64>>>,
}

struct GridderCache {
    key: u64,
    op: Arc<Nufft2d>,
    density: Arc<Vec<f64>>,
    shape: [usize; 2],
}

/// Cached simulation state
pub struct Simulator {
    comps: Option<CompsCache>,
    gridder: Option<GridderCache>,
    noise: Option<(usize, u64, Arc<Vec<C64>>)>,
    rng: Rng,
    noise_generation: u64,
}

impl Default for Simulator {
    fn default() -> Self {
        Simulator { comps: None, gridder: None, noise: None, rng: Rng::new(0x5EED_5EED_1234_5678), noise_generation: 0 }
    }
}

/// xoshiro256** PRNG with Box–Muller normal deviates
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = x ^ (x >> 31);
        }
        Rng { s }
    }
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
    fn normal_pair(&mut self) -> (f64, f64) {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let th = 2.0 * PI * self.uniform();
        (r * th.cos(), r * th.sin())
    }
}

fn hash_f64s(h: &mut DefaultHasher, v: &[f64]) {
    v.len().hash(h);
    for x in v {
        x.to_bits().hash(h);
    }
}

/// Sampled k-space axes (`k_axes`)
pub fn k_axes(p: &Protocol) -> [Vec<f64>; 2] {
    let mut ax: [Vec<f64>; 2] = [vec![], vec![]];
    ax[p.freq_dir] = p.k_read_axis.clone();
    ax[p.phase_dir] = p.k_phase_axis[..p.num_sampled_phase_encodes].to_vec();
    ax
}

/// Axes of the Cartesian grid the data is (re)gridded to (`k_grid_axes`)
pub fn k_grid_axes(p: &Protocol, ph: &Phantom) -> [Vec<f64>; 2] {
    if !p.is_radial {
        return k_axes(p);
    }
    [0, 1].map(|d| {
        let voxel = p.fov[d] / p.matrix[d] as f64;
        let m = (p.fov[d].max(ph.support[d]) / voxel).ceil() as usize;
        get_k_axis(m, voxel)
    })
}

/// k-space sample coordinates (n0, n1, na) flattened, (`k_samples`)
fn k_samples(axes: &[Vec<f64>; 2], angles: &[f64]) -> Vec<[f64; 2]> {
    let na = angles.len();
    let rot: Vec<(f64, f64)> = angles.iter().map(|a| a.sin_cos()).collect();
    let mut out = Vec::with_capacity(axes[0].len() * axes[1].len() * na);
    for &s0 in &axes[0] {
        for &s1 in &axes[1] {
            for &(s, c) in &rot {
                out.push([s0 * c + s1 * s, -s0 * s + s1 * c]);
            }
        }
    }
    out
}

/// `get_gridder`: NUFFT operator for samples `k` with field of view `fov`
fn make_gridder(k: &[[f64; 2]], fov: [f64; 2], with_density: bool) -> (Nufft2d, Option<Vec<f64>>, [usize; 2]) {
    let shape = [0, 1].map(|d| {
        let m = k.iter().fold(0.0f64, |m, s| m.max(s[d].abs()));
        ((m * fov[d] * 2.0).ceil() as usize).max(1)
    });
    let samples: Vec<[f64; 2]> = k.iter().map(|s| [(s[0] as f32) as f64 * fov[0] / shape[0] as f64, (s[1] as f32) as f64 * fov[1] / shape[1] as f64]).collect();
    let density = with_density.then(|| crate::voronoi::voronoi_density(&samples));
    let pts: Vec<[f64; 2]> = samples.iter().map(|s| [2.0 * PI * s[0], 2.0 * PI * s[1]]).collect();
    (Nufft2d::new(&pts, shape), density, shape)
}

impl Simulator {
    /// Draw a new noise realisation on the next run
    pub fn renew_noise(&mut self) {
        self.noise_generation += 1;
    }

    fn noise(&mut self, n: usize) -> Arc<Vec<C64>> {
        if let Some((len, generation, v)) = &self.noise {
            if *len == n && *generation == self.noise_generation {
                return v.clone();
            }
        }
        let v: Vec<C64> = (0..n)
            .map(|_| {
                let (a, b) = self.rng.normal_pair();
                C64::new(a, b)
            })
            .collect();
        let v = Arc::new(v);
        self.noise = Some((n, self.noise_generation, v.clone()));
        v
    }

    /// Phantom k-space resampled at the sample positions, per tissue (`kspace_comps`)
    fn kspace_comps(&mut self, p: &Protocol, ph: &Phantom, pk: &PhantomKspace, axes: &[Vec<f64>; 2], grid_axes: &[Vec<f64>; 2]) -> Arc<Vec<Vec<C64>>> {
        let mut h = DefaultHasher::new();
        ph.hash.hash(&mut h);
        p.is_radial.hash(&mut h);
        hash_f64s(&mut h, &axes[0]);
        hash_f64s(&mut h, &axes[1]);
        hash_f64s(&mut h, &p.k_angles);
        hash_f64s(&mut h, &grid_axes[0]);
        hash_f64s(&mut h, &grid_axes[1]);
        let key = h.finish();
        if let Some(c) = &self.comps {
            if c.key == key {
                return c.comps.clone();
            }
        }
        let comps: Vec<Vec<C64>> = if !p.is_radial {
            let k0 = sinc_kernel(&grid_axes[0], &ph.k_axes[0], ph.support[0]);
            let k1 = sinc_kernel(&grid_axes[1], &ph.k_axes[1], ph.support[1]);
            pk.kspace.par_iter().map(|ksp| sinc_interpolate(ksp, &k0, &k1).data).collect()
        } else {
            let k = k_samples(axes, &p.k_angles);
            let (op, _, shape) = make_gridder(&k, ph.support, false);
            let images: Vec<Array2<C64>> = pk
                .kspace
                .par_iter()
                .map(|g| {
                    let sample_shifts = [0, 1].map(|d| if g.shape()[d] % 2 == 1 { 0.0 } else { 0.5 });
                    ifft_shifted(&crop_or_pad(g, shape), [0.0, 0.0], sample_shifts)
                })
                .collect();
            let refs: Vec<&Array2<C64>> = images.iter().collect();
            op.forward_batch(&refs)
        };
        let comps = Arc::new(comps);
        self.comps = Some(CompsCache { key, comps: comps.clone() });
        comps
    }

    /// Run the simulation and reconstruction
    pub fn run(&mut self, p: &Protocol, inp: &Inputs, ph: &Phantom, pk: &PhantomKspace) -> SimOutput {
        let mut timings = Vec::new();
        let t0 = Instant::now();
        let axes = k_axes(p);
        let grid_axes = k_grid_axes(p, ph);
        let [n0, n1] = [axes[0].len(), axes[1].len()];
        let na = p.k_angles.len();
        let comps = self.kspace_comps(p, ph, pk, &axes, &grid_axes);
        timings.push(("k-space resampling", t0.elapsed().as_secs_f64()));

        // ---- contrast weights per (i, j) and tissue: T2/T2' decay, fat dephasing, PD/T1, slice profile
        let t1 = Instant::now();
        let fi = field_index(inp.field_strength);
        let n_read = p.k_read_axis.len();
        let fat_ratio = |r: &FatResonance| if inp.fatsat { r.ratio_with_fatsat } else { r.ratio };
        let pd_t1w_fat: Vec<f64> = FAT_RESONANCES.iter().map(|r| component_pd_t1w(r.name, &inp.sequence_type, p.tr, inp.ti, inp.fa, inp.field_strength)).collect();
        struct TissueW {
            t2: f64,
            water: f64,
            ff: f64,
        }
        let tw: Vec<TissueW> = ph
            .tissues
            .iter()
            .map(|t| {
                let ts = tissue(t).copied().unwrap_or(Tissue { name: "", pd: 0.0, ff: 0.0, t1: [1.0; 2], t2: [1.0; 2] });
                let pdt1 = component_pd_t1w(t, &inp.sequence_type, p.tr, inp.ti, inp.fa, inp.field_strength);
                TissueW { t2: ts.t2[fi], water: pdt1 * if ts.ff == 0.0 { 1.0 } else { 1.0 - ts.ff }, ff: ts.ff }
            })
            .collect();
        let nt = tw.len();
        // tae[i, j]: time after excitation
        let mut weights = vec![C64::new(0.0, 0.0); n0 * n1 * nt];
        let mut tae_grid = vec![0.0; n0 * n1];
        weights.par_chunks_mut(nt).zip(tae_grid.par_iter_mut()).enumerate().for_each(|(ij, (w, tae_out))| {
            let (i, j) = (ij / n1, ij % n1);
            let (i_read, i_pe) = if p.freq_dir == 0 { (i, j) } else { (j, i) };
            let (rf, gr) = p.pe_echo[i_pe];
            let ro = &p.readouts[rf][gr];
            let st = if ro.area < 0.0 { p.sampling_time[n_read - 1 - i_read] } else { p.sampling_time[i_read] };
            let tae = ro.center + st;
            *tae_out = tae;
            let t_rel = if p.is_gradient_echo { tae } else { tae - p.spin_echoes[i_pe] };
            let e2p = (-t_rel.abs() / T2_PRIM).exp();
            let kabs = (axes[0][i] * axes[0][i] + axes[1][j] * axes[1][j]).sqrt();
            let thick = inp.slice_thickness * (-0.5 * inp.slice_thickness * kabs).exp();
            // fat: Σ_r ratio_r PD/T1w_r e^{-|t|/T2_r} e^{2πi γ B0 shift_r t_rel}
            let mut fat = C64::new(0.0, 0.0);
            for (r, res) in FAT_RESONANCES.iter().enumerate() {
                let decay = (-tae.abs() / res.t2[fi]).exp() * e2p;
                let deph = C64::from_polar(1.0, 2.0 * PI * GYRO * inp.field_strength * res.shift * t_rel * 1e-3);
                fat += deph * (decay * pd_t1w_fat[r] * fat_ratio(res));
            }
            for (t, x) in tw.iter().enumerate() {
                let water = (-tae.abs() / x.t2).exp() * e2p * x.water;
                w[t] = (C64::new(water, 0.0) + if x.ff > 0.0 { fat * x.ff } else { C64::new(0.0, 0.0) }) * thick;
            }
        });
        let n = n0 * n1 * na;
        let noise = self.noise(n);
        let mut measured = vec![C64::new(0.0, 0.0); n];
        let noise_std = p.noise_std;
        measured.par_chunks_mut(na).enumerate().for_each(|(ij, m)| {
            let w = &weights[ij * nt..(ij + 1) * nt];
            for (a, v) in m.iter_mut().enumerate() {
                let s = ij * na + a;
                let mut acc = noise[s] * noise_std;
                for t in 0..nt {
                    acc += comps[t][s] * w[t];
                }
                *v = acc;
            }
        });
        timings.push(("contrast & noise", t1.elapsed().as_secs_f64()));

        // ---- gridding
        let t2 = Instant::now();
        let grid_shape = [grid_axes[0].len(), grid_axes[1].len()];
        let gridded: Array2<C64> = if !p.is_radial {
            Array2::from_vec(grid_shape[0], grid_shape[1], measured.clone())
        } else {
            let mut h = DefaultHasher::new();
            hash_f64s(&mut h, &axes[0]);
            hash_f64s(&mut h, &axes[1]);
            hash_f64s(&mut h, &p.k_angles);
            hash_f64s(&mut h, &p.fov);
            let key = h.finish();
            if self.gridder.as_ref().is_none_or(|g| g.key != key) {
                let k = k_samples(&axes, &p.k_angles);
                let (op, density, shape) = make_gridder(&k, p.fov, true);
                self.gridder = Some(GridderCache { key, op: Arc::new(op), density: Arc::new(density.unwrap()), shape });
            }
            let g = self.gridder.as_ref().unwrap();
            let shape = g.shape;
            let inv_norm = 1.0 / ((shape[0] * shape[1]) as f64 * 4.0).sqrt();
            let c: Vec<C64> = measured.par_iter().zip(g.density.par_iter()).map(|(m, d)| m * (d * inv_norm)).collect();
            let img = g.op.adjoint(&c);
            let sample_shifts = [0, 1].map(|d| if shape[d] % 2 == 1 { 0.0 } else { 0.5 });
            crop_or_pad(&fft_shifted(&img, [0.0, 0.0], sample_shifts), grid_shape)
        };
        timings.push(("gridding", t2.elapsed().as_secs_f64()));

        // ---- k-space processing
        let t3 = Instant::now();
        let full_kspace: Array2<C64> = if p.num_blank_lines == 0 || p.is_radial {
            gridded.clone()
        } else {
            let mut shape = gridded.shape();
            shape[p.phase_dir] += p.num_blank_lines;
            let mut full: Array2<C64> = Array2::zeros(shape[0], shape[1]);
            for i in 0..gridded.rows {
                full.data[i * shape[1]..i * shape[1] + gridded.cols].copy_from_slice(gridded.row(i));
            }
            if inp.homodyne {
                let w = homodyne_weights(p.num_phase_encodes_full, p.num_blank_lines);
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        let k = if p.phase_dir == 0 { i } else { j };
                        full.data[i * shape[1] + j] *= w.get(k).copied().unwrap_or(1.0);
                    }
                }
                let conj_flip: Vec<C64> = full.data.iter().rev().map(|z| z.conj()).collect();
                for (a, b) in full.data.iter_mut().zip(conj_flip.iter()) {
                    *a += *b;
                }
            }
            full
        };
        let full_k_matrix = full_kspace.shape();
        let mut apodized = full_kspace;
        if inp.do_apodize {
            let w = radial_tukey(inp.apodization_alpha, full_k_matrix);
            apodized.data.par_iter_mut().zip(w.data.par_iter()).for_each(|(a, b)| *a *= *b);
        }
        let recon_matrix = p.recon_matrix;
        let oversampled = [0, 1].map(|d| (pyround(recon_matrix[d] as f64 * full_k_matrix[d] as f64 / p.matrix[d] as f64) as usize).max(1));
        let zerofilled = crop_or_pad(&apodized, oversampled);
        let mut pixel_shifts = [0.0; 2];
        let mut sample_shifts = [0.0; 2];
        for d in 0..2 {
            if oversampled[d] % 2 == 0 {
                pixel_shifts[d] += 0.5; // half pixel shift for even matrixsize due to fft
            }
            if (oversampled[d] as i64 - recon_matrix[d] as i64).rem_euclid(2) == 1 {
                pixel_shifts[d] += 0.5; // half pixel shift due to cropping an odd number of pixels in image space
            }
            if full_k_matrix[d] % 2 == 0 {
                sample_shifts[d] += 0.5; // half sample shift for even matrixsize due to fft
                if (oversampled[d] as i64 - full_k_matrix[d] as i64).rem_euclid(2) == 1 {
                    sample_shifts[d] -= 1.0; // sample shift for odd number of zeroes added
                }
            }
        }
        let oversampled_image = ifft_shifted(&zerofilled, pixel_shifts, sample_shifts);
        let image_array = crop_or_pad(&oversampled_image, recon_matrix);
        timings.push(("reconstruction", t3.elapsed().as_secs_f64()));

        let processed_axes = [0, 1].map(|d| {
            let mut ax = get_k_axis(oversampled[d], p.fov[d] / recon_matrix[d] as f64);
            if (oversampled[d] as i64 - full_k_matrix[d] as i64).rem_euclid(2) == 1 {
                // half-sample shift axis when odd number of zeroes
                let shift = recon_matrix[d] as f64 / (2.0 * oversampled[d] as f64 * p.fov[d]);
                for k in ax.iter_mut() {
                    *k -= shift;
                }
            }
            ax
        });
        timings.push(("total", t0.elapsed().as_secs_f64()));
        SimOutput {
            image_array,
            fov: p.fov,
            gridded,
            grid_axes,
            zerofilled,
            processed_axes,
            measured,
            measured_shape: [n0, n1, na],
            tae: tae_grid,
            timings,
        }
    }
}
