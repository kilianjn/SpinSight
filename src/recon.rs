//! k-space helpers: axes, shifted FFTs, cropping/zero-filling, sinc interpolation, filters
//! (port of `recon.py`; the non-Cartesian gridding lives in `nufft.rs`).

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::array::{fft2, fftshift_2d, ifft2, ifftshift_2d, Array2, C64};

/// numpy `fftfreq(n)` (d = 1)
pub fn fftfreq(n: usize) -> Vec<f64> {
    let val = 1.0 / n as f64;
    let half = n.div_ceil(2);
    (0..n).map(|i| if i < half { i as f64 } else { i as f64 - n as f64 } * val).collect()
}

/// numpy `fftshift` of a 1D vector
pub fn fftshift_1d<T: Clone>(v: &[T]) -> Vec<T> {
    let n = v.len();
    let s = n / 2;
    (0..n).map(|j| v[(j + n - s) % n].clone()).collect()
}

/// Symmetric, fftshifted k-space axis [1/mm] for `matrix` samples with pixel size [mm] (`get_k_axis`)
pub fn get_k_axis(matrix: usize, pixel_size: f64) -> Vec<f64> {
    let mut kax = fftfreq(matrix);
    if matrix % 2 == 0 {
        // half-pixel to make even matrix symmetric
        for k in kax.iter_mut() {
            *k += 1.0 / (2 * matrix) as f64;
        }
    }
    for k in kax.iter_mut() {
        *k /= pixel_size;
    }
    fftshift_1d(&kax)
}

/// Separable factors of `get_pixel_shift_matrix(shape, shift)`: M[i, j] = e0[i] * e1[j]
fn pixel_shift_factors(shape: [usize; 2], shift: [f64; 2]) -> [Vec<C64>; 2] {
    let f = |d: usize| -> Vec<C64> { fftfreq(shape[d]).into_iter().map(|q| C64::from_polar(1.0, q * shift[d] * 2.0 * PI)).collect() };
    [f(0), f(1)]
}

fn apply_factors(a: &mut Array2<C64>, f: &[Vec<C64>; 2], divide: bool) {
    let cols = a.cols;
    a.data.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        for (c, v) in row.iter_mut().enumerate() {
            let m = f[0][r] * f[1][c];
            if divide {
                *v /= m;
            } else {
                *v *= m;
            }
        }
    });
}

/// `recon.FFT`
pub fn fft_shifted(img: &Array2<C64>, pixel_shifts: [f64; 2], sample_shifts: [f64; 2]) -> Array2<C64> {
    let mut x = ifftshift_2d(img);
    apply_factors(&mut x, &pixel_shift_factors(img.shape(), sample_shifts), true);
    let mut k = fft2(&x);
    apply_factors(&mut k, &pixel_shift_factors(img.shape(), pixel_shifts), true);
    fftshift_2d(&k)
}

/// `recon.IFFT`
pub fn ifft_shifted(ksp: &Array2<C64>, pixel_shifts: [f64; 2], sample_shifts: [f64; 2]) -> Array2<C64> {
    let mut k = ifftshift_2d(ksp);
    apply_factors(&mut k, &pixel_shift_factors(ksp.shape(), pixel_shifts), false);
    let mut img = ifft2(&k);
    apply_factors(&mut img, &pixel_shift_factors(ksp.shape(), sample_shifts), false);
    fftshift_2d(&img)
}

/// Crop (from center) or zero-pad each dimension to `shape` (`crop_or_pad`)
pub fn crop_or_pad<T: Clone + Default>(arr: &Array2<T>, shape: [usize; 2]) -> Array2<T> {
    let mut src_start = [0usize; 2];
    let mut dst_start = [0usize; 2];
    let mut len = [0usize; 2];
    for (d, &n) in arr.shape().iter().enumerate() {
        if shape[d] < n {
            src_start[d] = (n - shape[d]) / 2;
            len[d] = shape[d];
        } else {
            let after = (shape[d] - n) / 2;
            dst_start[d] = shape[d] - n - after;
            len[d] = n;
        }
    }
    let mut out = Array2::zeros(shape[0], shape[1]);
    for r in 0..len[0] {
        let s = (src_start[0] + r) * arr.cols + src_start[1];
        let d = (dst_start[0] + r) * shape[1] + dst_start[1];
        out.data[d..d + len[1]].clone_from_slice(&arr.data[s..s + len[1]]);
    }
    out
}

#[inline]
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Sinc interpolation kernel K[a, b] = sinc((target[b] - source[a]) * support), shape (source × target)
pub fn sinc_kernel(target: &[f64], source: &[f64], support: f64) -> Array2<f64> {
    let mut k = Array2::zeros(source.len(), target.len());
    k.data.par_chunks_mut(target.len().max(1)).enumerate().for_each(|(a, row)| {
        for (b, v) in row.iter_mut().enumerate() {
            *v = sinc((target[b] - source[a]) * support);
        }
    });
    k
}

/// out = K0ᵀ · ksp · K1 with real kernels (`sinc_interpolate`)
pub fn sinc_interpolate(ksp: &Array2<C64>, k0: &Array2<f64>, k1: &Array2<f64>) -> Array2<C64> {
    assert_eq!(ksp.rows, k0.rows);
    assert_eq!(ksp.cols, k1.rows);
    let (a, b) = (ksp.rows, ksp.cols);
    let (n0, n1) = (k0.cols, k1.cols);
    let mut tmp: Array2<C64> = Array2::zeros(a, n1);
    let mut out: Array2<C64> = Array2::zeros(n0, n1);
    if a == 0 || b == 0 || n0 == 0 || n1 == 0 {
        return out;
    }
    for part in 0..2 {
        unsafe {
            // tmp = ksp (a×b) · K1 (b×n1)
            matrixmultiply::dgemm(
                a,
                b,
                n1,
                1.0,
                (ksp.data.as_ptr() as *const f64).add(part),
                2 * b as isize,
                2,
                k1.data.as_ptr(),
                n1 as isize,
                1,
                0.0,
                (tmp.data.as_mut_ptr() as *mut f64).add(part),
                2 * n1 as isize,
                2,
            );
            // out = K0ᵀ (n0×a) · tmp (a×n1)
            matrixmultiply::dgemm(
                n0,
                a,
                n1,
                1.0,
                k0.data.as_ptr(),
                1,
                n0 as isize,
                (tmp.data.as_ptr() as *const f64).add(part),
                2 * n1 as isize,
                2,
                0.0,
                (out.data.as_mut_ptr() as *mut f64).add(part),
                2 * n1 as isize,
                2,
            );
        }
    }
    out
}

/// Homodyne ramp filter of length n with `num_blank` unsampled lines (`homodyne_weights`)
pub fn homodyne_weights(n: usize, num_blank: usize) -> Vec<f64> {
    let mut w = vec![1.0; n];
    if num_blank == 0 || 2 * num_blank > n {
        return w;
    }
    let ramp = crate::params::linspace(1.0, 0.0, n + 2 - 2 * num_blank);
    w[num_blank..n - num_blank].copy_from_slice(&ramp[1..ramp.len() - 1]);
    for v in w[n - num_blank..].iter_mut() {
        *v = 0.0;
    }
    w
}

/// Radial Tukey apodization window (`radial_Tukey`)
pub fn radial_tukey(alpha: f64, shape: [usize; 2]) -> Array2<f64> {
    let kx: Vec<f64> = crate::params::linspace(-1.0, 1.0, shape[0] + 2)[1..shape[0] + 1].to_vec();
    let ky: Vec<f64> = crate::params::linspace(-1.0, 1.0, shape[1] + 2)[1..shape[1] + 1].to_vec();
    let mut w = Array2::zeros(shape[0], shape[1]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let k = ((1.0 - (kx[i] * kx[i] + ky[j] * ky[j]).sqrt()) / alpha).clamp(0.0, 1.0);
            let s = (PI * k / 2.0).sin();
            w.data[i * shape[1] + j] = s * s;
        }
    }
    w
}

/// Rotation matrix [[cos, -sin], [sin, cos]]
pub fn rotation_matrix(theta: f64) -> [[f64; 2]; 2] {
    let (s, c) = theta.sin_cos();
    [[c, -s], [s, c]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_axis() {
        let k = get_k_axis(4, 1.0);
        assert_eq!(k, vec![-0.375, -0.125, 0.125, 0.375]);
        let k = get_k_axis(5, 2.0);
        assert_eq!(k, vec![-0.2, -0.1, 0.0, 0.1, 0.2]);
    }

    #[test]
    fn fft_roundtrip() {
        let a = Array2::from_vec(3, 4, (0..12).map(|i| C64::new(i as f64, (i * i) as f64 * 0.1)).collect());
        let k = fft_shifted(&a, [0.5, 0.0], [0.5, 0.5]);
        let b = ifft_shifted(&k, [0.5, 0.0], [0.5, 0.5]);
        for (x, y) in a.data.iter().zip(b.data.iter()) {
            assert!((x - y).norm() < 1e-12);
        }
    }

    #[test]
    fn crop_pad() {
        let a = Array2::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let p = crop_or_pad(&a, [4, 4]);
        // pad: after = (4-2)//2 = 1 → before = 1 rows; cols: after = 0, before = 1
        assert_eq!(p.data, vec![0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0]);
        let c = crop_or_pad(&a, [1, 2]);
        assert_eq!(c.data, vec![1, 2]);
    }
}
