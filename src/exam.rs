//! Examinations with 3D phantoms: the planned slice stack is acquired slice by slice ("scan")
//! into a series of images, as on a scanner.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::array::Array2;
use crate::geometry::{Orientation, SliceGeometry};
use crate::imaging::{self, Simulator};
use crate::phantom;
use crate::protocol::{self, Inputs};
use crate::volume::Volume;

/// An acquired series
#[derive(Clone)]
pub struct Series {
    /// series number within the examination (1, 2, …)
    pub number: usize,
    /// protocol name with orientation, e.g. "T2w TSE tra"
    pub name: String,
    /// all parameters of the acquisition (for "copy parameters")
    pub inputs: Inputs,
    /// slice positions
    pub geometries: Vec<SliceGeometry>,
    /// reconstructed FOV [mm] (rows, columns)
    pub fov: [f64; 2],
    /// one image per slice (image mode of the protocol)
    pub images: Vec<Array2<f32>>,
    /// display window (low, high), common to all slices
    pub window: [f32; 2],
    /// acquisition time [ms]
    pub scantime: f64,
    /// relative SNR at the time of the acquisition, as displayed
    pub relative_snr: String,
}

/// Progress of a running scan
#[derive(Default)]
pub struct ScanProgress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: AtomicBool,
}

impl ScanProgress {
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed).max(1);
        self.done.load(Ordering::Relaxed) as f32 / t as f32
    }
}

/// Short scanner-like name of the slice orientation: "tra", "sag>cor", …
pub fn orientation_label(inp: &Inputs) -> String {
    let o = Orientation::from_name(&inp.slice_orientation);
    let (t1, t2) = o.tilt_targets();
    // the dominant tilt determines the label, as on scanners ("tra>cor")
    let (a, t) = if inp.slice_angle_1.abs() >= inp.slice_angle_2.abs() { (inp.slice_angle_1, t1) } else { (inp.slice_angle_2, t2) };
    let base = o.short().to_lowercase();
    if a.abs() >= 0.5 {
        format!("{base}>{}", t.short().to_lowercase())
    } else {
        base
    }
}

/// Acquire all slices of the planned stack. `None` if cancelled.
pub fn acquire(inp: &Inputs, vol: &Arc<Volume>, name: &str, number: usize, seed: u64, progress: Option<&ScanProgress>) -> Option<Series> {
    let stack = inp.stack();
    let geometries = stack.slices();
    if let Some(p) = progress {
        p.total.store(geometries.len(), Ordering::Relaxed);
        p.done.store(0, Ordering::Relaxed);
    }
    let mut images = Vec::with_capacity(geometries.len());
    let mut fov = [inp.fov_p, inp.fov_f];
    let mut scantime = 0.0;
    for (i, g) in geometries.iter().enumerate() {
        if progress.is_some_and(|p| p.cancel.load(Ordering::Relaxed)) {
            return None;
        }
        let ph = phantom::slice_phantom(vol, *g, inp.min_voxel_size);
        let p = protocol::compute(inp, &ph);
        let pk = phantom::compute_kspace(&ph, None);
        // independent noise in every slice
        let mut sim = Simulator::with_seed(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64 + 1));
        let out = sim.run(&p, inp, &ph, &pk);
        let img = imaging::display_image(&out, &inp.image_mode);
        images.push(Array2 { rows: img.rows, cols: img.cols, data: img.data.iter().map(|&v| v as f32).collect() });
        fov = out.fov;
        scantime = p.scantime;
        if let Some(pr) = progress {
            pr.done.fetch_add(1, Ordering::Relaxed);
        }
    }
    let window = display_window(&images, inp.image_mode.eq_ignore_ascii_case("magnitude"));
    Some(Series {
        number,
        name: format!("{name} {}", orientation_label(inp)),
        inputs: inp.clone(),
        geometries,
        fov,
        images,
        window,
        scantime,
        relative_snr: String::new(),
    })
}

/// Common display window of all images: robust minimum/maximum (0.1 % / 99.9 % quantiles)
pub fn display_window(images: &[Array2<f32>], from_zero: bool) -> [f32; 2] {
    let mut v: Vec<f32> = images.par_iter().flat_map_iter(|a| a.data.iter().cloned().filter(|x| x.is_finite())).collect();
    if v.is_empty() {
        return [0.0, 1.0];
    }
    let q = |v: &mut Vec<f32>, f: f64| -> f32 {
        let k = ((v.len() - 1) as f64 * f).round() as usize;
        *v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
    };
    let hi = q(&mut v, 0.999) * 1.05;
    let lo = if from_zero { 0.0 } else { q(&mut v, 0.001) };
    if hi > lo {
        [lo, hi]
    } else {
        [lo, lo + 1.0]
    }
}

/// 8-bit grey values of an image with the given window
pub fn to_gray(img: &Array2<f32>, window: [f32; 2]) -> Vec<u8> {
    let scale = 255.0 / (window[1] - window[0]).max(1e-12);
    img.data.iter().map(|&v| if v.is_finite() { ((v - window[0]) * scale).round().clamp(0.0, 255.0) as u8 } else { 0 }).collect()
}

/// Write the images of a series as PNG files `<dir>/<number>_<name>/slice_NN.png`; returns the folder
pub fn save_series(series: &Series, dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let safe: String = series
        .name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else if c == '>' {
                '-'
            } else {
                '_'
            }
        })
        .collect();
    let folder = dir.join(format!("{:02}_{safe}", series.number));
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    for (i, img) in series.images.iter().enumerate() {
        let bytes = to_gray(img, series.window);
        let png = image::GrayImage::from_raw(img.cols as u32, img.rows as u32, bytes).ok_or("invalid image size")?;
        let path = folder.join(format!("slice_{:02}.png", i + 1));
        png.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(folder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_labels() {
        let mut inp = Inputs { slice_orientation: "Transversal".into(), ..Default::default() };
        assert_eq!(orientation_label(&inp), "tra");
        inp.slice_angle_1 = -12.0;
        assert_eq!(orientation_label(&inp), "tra>cor");
        inp.slice_orientation = "Sagittal".into();
        inp.slice_angle_1 = 0.0;
        inp.slice_angle_2 = 20.0;
        assert_eq!(orientation_label(&inp), "sag>cor");
    }

    #[test]
    fn multislice_acquisition() {
        let vol = crate::volume::builtin(crate::volume::BRAIN3D).unwrap();
        let inp = Inputs { num_slices: 3, slice_gap: 1.0, matrix_f: 64, matrix_p: 64, ..Default::default() };
        let progress = ScanProgress::default();
        let s = acquire(&inp, &vol, "test", 1, 7, Some(&progress)).unwrap();
        assert_eq!(s.images.len(), 3);
        assert_eq!(progress.done.load(Ordering::Relaxed), 3);
        // different slices give different images, all with signal
        let sum = |a: &Array2<f32>| a.data.iter().map(|v| *v as f64).sum::<f64>();
        assert!(s.images.iter().all(|a| sum(a) > 0.0));
        assert!((sum(&s.images[0]) - sum(&s.images[2])).abs() > 1e-3 * sum(&s.images[1]));
        assert!(s.window[1] > s.window[0]);
    }
}
