//! Minimal row-major 2D array and FFT helpers.

use std::cell::RefCell;

use rayon::prelude::*;
use rustfft::{num_complex::Complex64, FftPlanner};

pub type C64 = Complex64;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Array2<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Clone + Default> Array2<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Array2 { rows, cols, data: vec![T::default(); rows * cols] }
    }
}

impl<T> Array2<T> {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len());
        Array2 { rows, cols, data }
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    #[inline]
    pub fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> &T {
        &self.data[r * self.cols + c]
    }
    #[inline]
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        let i = r * self.cols + c;
        &mut self.data[i]
    }
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Array2<U> {
        Array2 { rows: self.rows, cols: self.cols, data: self.data.iter().map(f).collect() }
    }
    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

impl<T: Clone + Default + Send + Sync> Array2<T> {
    pub fn transpose(&self) -> Array2<T> {
        let mut out = Array2::zeros(self.cols, self.rows);
        out.data.par_chunks_mut(self.rows).enumerate().for_each(|(c, col)| {
            for (r, v) in col.iter_mut().enumerate() {
                *v = self.data[r * self.cols + c].clone();
            }
        });
        out
    }
}

/// numpy `fftshift` index map: element at `i` moves to `(i + n/2) % n`
pub fn fftshift_2d<T: Clone + Default>(a: &Array2<T>) -> Array2<T> {
    roll_2d(a, a.rows / 2, a.cols / 2)
}

/// numpy `ifftshift`
pub fn ifftshift_2d<T: Clone + Default>(a: &Array2<T>) -> Array2<T> {
    roll_2d(a, a.rows - a.rows / 2, a.cols - a.cols / 2)
}

/// `np.roll` by (s0, s1) (non-negative shifts)
pub fn roll_2d<T: Clone + Default>(a: &Array2<T>, s0: usize, s1: usize) -> Array2<T> {
    let mut out = Array2::zeros(a.rows, a.cols);
    if a.rows == 0 || a.cols == 0 {
        return out;
    }
    for r in 0..a.rows {
        let ro = (r + s0) % a.rows;
        for c in 0..a.cols {
            let co = (c + s1) % a.cols;
            out.data[ro * a.cols + co] = a.data[r * a.cols + c].clone();
        }
    }
    out
}

thread_local! {
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

fn fft_rows(data: &mut [C64], cols: usize, inverse: bool) {
    if cols == 0 || data.is_empty() {
        return;
    }
    let fft = PLANNER.with(|p| {
        let mut p = p.borrow_mut();
        if inverse {
            p.plan_fft_inverse(cols)
        } else {
            p.plan_fft_forward(cols)
        }
    });
    data.par_chunks_mut(cols).for_each(|row| fft.process(row));
}

/// Unnormalised 2D DFT (forward: e^{-i..}, inverse: e^{+i..})
pub fn fft2_inplace(a: &mut Array2<C64>, inverse: bool) {
    fft_rows(&mut a.data, a.cols, inverse);
    let mut t = a.transpose();
    fft_rows(&mut t.data, t.cols, inverse);
    *a = t.transpose();
}

/// numpy `fft2`
pub fn fft2(a: &Array2<C64>) -> Array2<C64> {
    let mut b = a.clone();
    fft2_inplace(&mut b, false);
    b
}

/// numpy `ifft2` (normalised by 1/N)
pub fn ifft2(a: &Array2<C64>) -> Array2<C64> {
    let mut b = a.clone();
    fft2_inplace(&mut b, true);
    let n = (a.rows * a.cols) as f64;
    b.data.par_iter_mut().for_each(|v| *v /= n);
    b
}
