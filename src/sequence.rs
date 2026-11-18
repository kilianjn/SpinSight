//! Pulse sequence objects: gradients, RF pulses, ADC windows and signals (port of `sequence.py`).

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::constants::GYRO;

/// A sequence object placed on one of the sequence boards (gradient, RF pulse, ADC window or signal).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Waveform {
    pub name: String,
    /// amplitude samples (gradient [mT/m], RF [μT] or signal [a.u.]); empty for ADC windows
    pub amp: Vec<f64>,
    /// sample times [ms]
    pub time: Vec<f64>,
    pub center: f64,
    pub dur: f64,
    pub flat_dur: f64,
    pub risetime: f64,
    /// gradient area [μTs/m]
    pub area: f64,
    /// flip angle [°] (RF pulses)
    pub flip_angle: f64,
    /// spectral full width at half maximum [Hz] (RF pulses)
    pub fwhm: f64,
}

impl Waveform {
    pub fn t_start(&self) -> f64 {
        self.time[0]
    }
    pub fn t_end(&self) -> f64 {
        *self.time.last().unwrap()
    }
}

/// How the gradient trapezoid is specified (exactly one of the Python keyword arguments)
#[derive(Debug, Clone, Copy)]
pub enum GradSpec {
    TotalArea(f64),
    FlatArea(f64),
    FlatDur(f64),
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub fn get_gradient(time: f64, max_amp: f64, max_slew: f64, spec: GradSpec, name: &str) -> Waveform {
    let mut max_amp = max_amp;
    let mut flat_area = None;
    let mut flat_dur = 0.0;
    match spec {
        GradSpec::TotalArea(total_area) => {
            let slew_area = max_amp * max_amp / max_slew;
            if total_area.abs() < slew_area {
                max_amp = (total_area.abs() * max_slew).sqrt() * sign(total_area);
                flat_dur = 0.0;
            } else {
                flat_area = Some((total_area.abs() - slew_area) * sign(total_area));
            }
        }
        GradSpec::FlatArea(a) => flat_area = Some(a),
        GradSpec::FlatDur(d) => flat_dur = d,
    }
    if let Some(fa) = flat_area {
        flat_dur = (fa / max_amp).abs();
        max_amp = max_amp.abs() * sign(fa);
    }
    let risetime = max_amp.abs() / max_slew;
    let mut t = vec![0.0, risetime, risetime + flat_dur, risetime + flat_dur + risetime];
    let amp = vec![0.0, max_amp, max_amp, 0.0];
    let dur = t[3] - t[0];
    for x in t.iter_mut() {
        *x += time - dur / 2.0;
    }
    let area = gradient_area(&amp, &t);
    Waveform { name: name.to_string(), amp, time: t, center: time, dur, flat_dur, risetime, area, ..Default::default() }
}

pub fn gradient_area(g: &[f64], t: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..g.len().saturating_sub(1) {
        s += (t[i + 1] - t[i]) * (g[i] + g[i + 1]);
    }
    s / 2.0
}

pub fn move_waveform(wf: &mut Waveform, time: f64) {
    let shift = time - wf.center;
    for t in wf.time.iter_mut() {
        *t += shift;
    }
    wf.center = time;
}

pub fn place(wf: &Waveform, time: f64) -> Waveform {
    let mut w = wf.clone();
    move_waveform(&mut w, time);
    w
}

pub fn rescale_gradient(g: &mut Waveform, scale: f64) {
    for a in g.amp.iter_mut() {
        *a *= scale;
    }
    g.area *= scale;
}

/// Pad waveform with zeros and assign times (`prepare_waveform`)
fn prepare_waveform(waveform: &[f64], t0: f64, t1: f64, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let n = waveform.len();
    let mut wf = Vec::with_capacity(n + 2);
    wf.push(0.0);
    wf.extend(waveform.iter().map(|w| w * scale));
    wf.push(0.0);
    let mut t = Vec::with_capacity(n + 2);
    t.push(t0);
    t.extend(crate::params::linspace(t0, t1, n));
    t.push(t1);
    if t0 < t1 {
        (wf, t)
    } else {
        wf.reverse();
        t.reverse();
        (wf, t)
    }
}

/// Hamming-windowed sinc RF pulse shape (51 samples)
fn hamming_sinc() -> &'static [f64] {
    static CELL: OnceLock<Vec<f64>> = OnceLock::new();
    CELL.get_or_init(|| {
        let n = 51usize;
        (0..n)
            .map(|i| {
                let x = (i as f64 - n as f64 / 2.0) / n as f64 * 5.0;
                let sinc = if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) };
                let hamming = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                sinc * hamming
            })
            .collect()
    })
}

pub fn get_rf(flip_angle: f64, dur: f64, name: &str, time: f64) -> Waveform {
    let waveform = hamming_sinc();
    let (t0, t1) = (time - dur / 2.0, time + dur / 2.0);
    let mean = waveform.iter().sum::<f64>() / waveform.len() as f64;
    let scale = flip_angle / (mean * dur / 1e3 * GYRO * 360.0);
    let (am, t) = prepare_waveform(waveform, t0, t1, scale);
    let fwhm = get_fwhm(&am[1..am.len() - 1], &t[1..t.len() - 1]);
    Waveform { name: name.to_string(), amp: am, time: t, center: time, dur, flip_angle, fwhm, ..Default::default() }
}

/// ADC (sampling) window
pub fn get_adc(dur: f64, name: &str, time: f64) -> Waveform {
    Waveform { name: name.to_string(), time: vec![-dur / 2.0 + time, dur / 2.0 + time], center: time, dur, ..Default::default() }
}

/// Signal curve for the signal board
pub fn get_signal(signal: &[f64], time: &[f64], scale: f64, exponent: f64, name: &str) -> Waveform {
    let (t0, t1) = (time[0], *time.last().unwrap());
    let (mut am, t) = prepare_waveform(signal, t0, t1, scale);
    for a in am.iter_mut() {
        *a = sign(*a) * a.abs().powf(exponent);
    }
    Waveform { name: name.to_string(), amp: am, time: t, center: (t0 + t1) / 2.0, dur: (t1 - t0).abs(), ..Default::default() }
}

/// Spectral FWHM [Hz] of the pulse `s` sampled at times `t` [ms] (`get_FWHM`).
///
/// Interpolates the magnitude spectrum minus half its maximum with a not-a-knot cubic
/// spline (identical to FITPACK's interpolating spline with s=0) and returns the distance
/// between its two roots.
pub fn get_fwhm(s: &[f64], t: &[f64]) -> f64 {
    let n = s.len();
    let dt = t[1] - t[0];
    // |fftshift(fft(s))| and fftshift(fftfreq(n)) / dt * 1e3
    let half = n / 2; // fftshift moves index `half` of the unshifted array... computed directly below
    let freqs: Vec<i64> = (0..n as i64).map(|i| i - half as i64).collect();
    let mut mag = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for &k in &freqs {
        let (mut re, mut im) = (0.0, 0.0);
        for (j, &x) in s.iter().enumerate() {
            let phase = -2.0 * PI * (k as f64) * (j as f64) / n as f64;
            re += x * phase.cos();
            im += x * phase.sin();
        }
        mag.push((re * re + im * im).sqrt());
        f.push(k as f64 * (1.0 / n as f64) / dt * 1e3);
    }
    let max = mag.iter().cloned().fold(f64::MIN, f64::max);
    let y: Vec<f64> = mag.iter().map(|m| m - max / 2.0).collect();
    let roots = spline_roots(&f, &y);
    if roots.len() < 2 {
        return f64::NAN;
    }
    (roots[roots.len() - 1] - roots[0]).abs()
}

/// Roots of the not-a-knot cubic spline interpolating (x, y) (x strictly increasing)
fn spline_roots(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = not_a_knot_second_derivatives(x, y);
    let eval = |i: usize, xv: f64| -> f64 {
        let h = x[i + 1] - x[i];
        let a = x[i + 1] - xv;
        let b = xv - x[i];
        m[i] * a * a * a / (6.0 * h) + m[i + 1] * b * b * b / (6.0 * h) + (y[i] - m[i] * h * h / 6.0) * a / h + (y[i + 1] - m[i + 1] * h * h / 6.0) * b / h
    };
    let mut roots = Vec::new();
    for i in 0..n - 1 {
        // subdivide each interval to also catch multiple roots per interval
        const SUB: usize = 8;
        let h = x[i + 1] - x[i];
        for s in 0..SUB {
            let mut lo = x[i] + h * s as f64 / SUB as f64;
            let mut hi = x[i] + h * (s + 1) as f64 / SUB as f64;
            let (mut flo, fhi) = (eval(i, lo), eval(i, hi));
            if flo == 0.0 {
                if roots.last().is_none_or(|&r: &f64| (r - lo).abs() > 1e-12 * h.max(1.0)) {
                    roots.push(lo);
                }
                continue;
            }
            if flo * fhi >= 0.0 {
                continue;
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                let fm = eval(i, mid);
                if fm == 0.0 || (hi - lo) < 1e-15 * mid.abs().max(1.0) {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if flo * fm < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            roots.push(0.5 * (lo + hi));
        }
    }
    roots
}

fn not_a_knot_second_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut a = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for i in 1..n - 1 {
        let h0 = x[i] - x[i - 1];
        let h1 = x[i + 1] - x[i];
        a[i][i - 1] = h0 / 6.0;
        a[i][i] = (h0 + h1) / 3.0;
        a[i][i + 1] = h1 / 6.0;
        rhs[i] = (y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0;
    }
    // continuous third derivative at x[1] and x[n-2]
    let (h0, h1) = (x[1] - x[0], x[2] - x[1]);
    a[0][0] = 1.0 / h0;
    a[0][1] = -1.0 / h0 - 1.0 / h1;
    a[0][2] = 1.0 / h1;
    let (h0, h1) = (x[n - 2] - x[n - 3], x[n - 1] - x[n - 2]);
    a[n - 1][n - 3] = 1.0 / h0;
    a[n - 1][n - 2] = -1.0 / h0 - 1.0 / h1;
    a[n - 1][n - 1] = 1.0 / h1;
    solve_dense(a, rhs)
}

/// Gaussian elimination with partial pivoting
pub fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d == 0.0 {
            continue;
        }
        for row in col + 1..n {
            let f = a[row][col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row][k] * x[k];
        }
        x[row] = if a[row][row] != 0.0 { s / a[row][row] } else { 0.0 };
    }
    x
}

/// Sum of the waveforms on a board, as (time, amplitude) vertices (`accumulate_waveforms`)
pub fn accumulate_waveforms<'a>(waveforms: impl IntoIterator<Item = &'a Waveform>) -> Vec<(f64, f64)> {
    let mut slopes = Vec::new();
    for w in waveforms {
        let n = w.time.len();
        for i in 0..n {
            let d_amp = if i + 1 < n { w.amp[i + 1] - w.amp[i] } else { -w.amp[i] };
            let dt = if i + 1 < n { w.time[i + 1] - w.time[i] } else { 0.0 };
            slopes.push((w.time[i], d_amp, dt));
        }
    }
    accumulate_slopes(slopes)
}

/// `slopes` is a list of (t, dG, dt) where t is the start time and dG is the change over duration dt
pub fn accumulate_slopes(mut slopes: Vec<(f64, f64, f64)>) -> Vec<(f64, f64)> {
    slopes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut waveform: Vec<(f64, f64)> = Vec::new();
    let mut slewing: Vec<(f64, f64)> = Vec::new();
    let mut g = 0.0;
    for (t, dg, dt) in slopes {
        let mut keep = Vec::new();
        for &(slew, stoptime) in &slewing {
            g += slew * (stoptime.min(t) - waveform.last().unwrap().0);
            if stoptime > t {
                keep.push((slew, stoptime));
            }
        }
        slewing = keep;
        if waveform.len() > 2 && (t - waveform[waveform.len() - 2].0) < 1e-6 {
            waveform.pop(); // remove intermediate gradient amplitudes at an instant
        }
        waveform.push((t, g));
        if dg != 0.0 {
            if dt > 0.0 {
                slewing.push((dg / dt, t + dt));
            } else {
                g += dg; // infinite slope
            }
        }
    }
    waveform
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid() {
        let g = get_gradient(0.0, 25.0, 80.0, GradSpec::TotalArea(30.0), "spoiler");
        assert!((g.area - 30.0).abs() < 1e-9);
        let g = get_gradient(0.0, 25.0, 80.0, GradSpec::TotalArea(-2.0), "triangle");
        assert!((g.area + 2.0).abs() < 1e-9);
        assert_eq!(g.flat_dur, 0.0);
    }

    #[test]
    fn spline_roots_of_parabola() {
        let x: Vec<f64> = (0..21).map(|i| i as f64 * 0.5 - 5.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 4.0 - v * v).collect();
        let r = spline_roots(&x, &y);
        assert_eq!(r.len(), 2);
        assert!((r[0] + 2.0).abs() < 1e-9 && (r[1] - 2.0).abs() < 1e-9);
    }
}
