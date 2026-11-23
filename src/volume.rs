//! 3D tissue volumes (the "brain 3D" phantom): decoding, trilinear sampling, slice extraction
//! with slice-thickness integration, and localizer images.
//!
//! The volume stores per voxel the two strongest tissues and their fractions
//! (`[label a, label b, fraction a, fraction b]`, label 255 = none), see `tools/make_brain3d.py`.

use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use crate::geometry::{SliceGeometry, V3};

pub struct Volume {
    pub name: String,
    /// voxels along x (R), y (A), z (S)
    pub dims: [usize; 3],
    pub voxel: [f64; 3],
    /// RAS position of voxel (0, 0, 0) [mm]
    pub origin: [f64; 3],
    pub tissues: Vec<String>,
    data: Vec<[u8; 4]>,
    pub hash: u64,
}

impl std::fmt::Debug for Volume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Volume({}, {:?})", self.name, self.dims)
    }
}

const BRAIN3D_PNG: &[u8] = include_bytes!("../assets/volumes/brain3d.png");
const BRAIN3D_TOML: &str = include_str!("../assets/volumes/brain3d.toml");

pub const BRAIN3D: &str = "brain 3D";

pub fn builtin_names() -> &'static [&'static str] {
    &[BRAIN3D]
}

pub fn is_volume(name: &str) -> bool {
    builtin_names().contains(&name)
}

/// The built-in volume `name` (decoded on first use)
pub fn builtin(name: &str) -> Option<Arc<Volume>> {
    static BRAIN: OnceLock<Arc<Volume>> = OnceLock::new();
    match name {
        BRAIN3D => Some(BRAIN.get_or_init(|| Arc::new(Volume::from_mosaic(BRAIN3D, BRAIN3D_PNG, BRAIN3D_TOML).expect("built-in 3D phantom"))).clone()),
        _ => None,
    }
}

impl Volume {
    /// Decode a volume stored as a PNG mosaic of axial slices plus TOML metadata
    pub fn from_mosaic(name: &str, png: &[u8], toml_text: &str) -> Result<Volume, String> {
        let meta: toml::Table = toml::from_str(toml_text).map_err(|e| format!("invalid volume metadata: {e}"))?;
        let num = |v: &toml::Value| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)).unwrap_or(0.0);
        let triple = |k: &str| -> Result<[f64; 3], String> {
            let a = meta.get(k).and_then(|v| v.as_array()).ok_or(format!("{k} missing"))?;
            if a.len() != 3 {
                return Err(format!("{k} must have three entries"));
            }
            Ok([num(&a[0]), num(&a[1]), num(&a[2])])
        };
        let d = triple("dims")?;
        let dims = [d[0] as usize, d[1] as usize, d[2] as usize];
        let voxel = triple("voxel_size")?;
        let origin = triple("origin")?;
        let cols = meta.get("tile_columns").map(num).unwrap_or(1.0) as usize;
        let tissues: Vec<String> =
            meta.get("tissues").and_then(|v| v.as_array()).ok_or("tissues missing")?.iter().filter_map(|t| t.as_str().map(String::from)).collect();
        let img = image::load_from_memory(png).map_err(|e| format!("invalid volume image: {e}"))?.into_rgba8();
        let [nx, ny, nz] = dims;
        let mut data = vec![[255u8, 255, 0, 0]; nx * ny * nz];
        let w = img.width() as usize;
        let raw = img.as_raw();
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| {
            let (r, c) = (k / cols, k % cols);
            for y in 0..ny {
                for x in 0..nx {
                    let (py, px) = (r * ny + y, c * nx + x);
                    let o = (py * w + px) * 4;
                    if o + 3 < raw.len() {
                        slab[y * nx + x] = [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]];
                    }
                }
            }
        });
        let mut hash = 0xcbf29ce484222325u64;
        for b in toml_text.bytes().chain(name.bytes()).chain((png.len() as u64).to_le_bytes()) {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(Volume { name: name.to_string(), dims, voxel, origin, tissues, data, hash })
    }

    pub fn world_bounds(&self) -> (V3, V3) {
        let lo = [0, 1, 2].map(|d| self.origin[d] - 0.5 * self.voxel[d]);
        let hi = [0, 1, 2].map(|d| self.origin[d] + (self.dims[d] as f64 - 0.5) * self.voxel[d]);
        (lo, hi)
    }

    pub fn center(&self) -> V3 {
        let (lo, hi) = self.world_bounds();
        [0, 1, 2].map(|d| (lo[d] + hi[d]) / 2.0)
    }

    /// Trilinear corners of world point p: (flat index, weight)
    #[inline]
    fn corners(&self, p: V3, mut f: impl FnMut(usize, f32)) {
        let mut i0 = [0i64; 3];
        let mut t = [0f64; 3];
        for d in 0..3 {
            let g = (p[d] - self.origin[d]) / self.voxel[d];
            let fl = g.floor();
            i0[d] = fl as i64;
            t[d] = g - fl;
        }
        let [nx, ny, nz] = self.dims.map(|n| n as i64);
        for corner in 0..8 {
            let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let (x, y, z) = (i0[0] + dx, i0[1] + dy, i0[2] + dz);
            if x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz {
                continue;
            }
            let w = (if dx == 1 { t[0] } else { 1.0 - t[0] }) * (if dy == 1 { t[1] } else { 1.0 - t[1] }) * (if dz == 1 { t[2] } else { 1.0 - t[2] });
            if w > 0.0 {
                f((x + nx * (y + ny * z)) as usize, w as f32);
            }
        }
    }

    /// Add `weight` × tissue fractions at world point p to `acc` (one entry per tissue)
    #[inline]
    pub fn accumulate(&self, p: V3, weight: f32, acc: &mut [f32]) {
        self.corners(p, |i, w| {
            let [la, lb, fa, fb] = self.data[i];
            let s = w * weight / 255.0;
            if la != 255 {
                acc[la as usize] += s * fa as f32;
            }
            if lb != 255 {
                acc[lb as usize] += s * fb as f32;
            }
        });
    }

    /// Fraction of tissue `tissue` in voxel `index` (x fastest, then y, then z)
    #[inline]
    pub fn voxel_fraction(&self, index: usize, tissue: usize) -> f32 {
        let [la, lb, fa, fb] = self.data[index];
        let mut f = 0.0;
        if la as usize == tissue {
            f += fa as f32;
        }
        if lb as usize == tissue {
            f += fb as f32;
        }
        f / 255.0
    }

    /// Fraction of tissue (vs. air) at world point p
    #[inline]
    pub fn tissue_fraction(&self, p: V3) -> f32 {
        let mut s = 0.0;
        self.corners(p, |i, w| {
            let [la, lb, fa, fb] = self.data[i];
            let f = if la != 255 { fa as f32 } else { 0.0 } + if lb != 255 { fb as f32 } else { 0.0 };
            s += w * f / 255.0;
        });
        s
    }

    /// Weighted sum Σ_t w_t · fraction_t at p (for localizer contrast)
    #[inline]
    pub fn contrast(&self, p: V3, weights: &[f32]) -> f32 {
        let mut s = 0.0;
        self.corners(p, |i, w| {
            let [la, lb, fa, fb] = self.data[i];
            if la != 255 {
                s += w * weights[la as usize] * fa as f32 / 255.0;
            }
            if lb != 255 {
                s += w * weights[lb as usize] * fb as f32 / 255.0;
            }
        });
        s
    }

    /// Tissue density maps of a slice: `rows × cols` pixels of size `pixel` centred on the slice
    /// centre; each value is the integral of the tissue fraction over the slice thickness [mm].
    /// Returns one map per tissue (row-major).
    pub fn slice_maps(&self, g: &SliceGeometry, rows: usize, cols: usize, pixel: [f64; 2]) -> Vec<Vec<f32>> {
        let nt = self.tissues.len();
        let nsub = ((g.thickness / 0.5).ceil() as usize).clamp(3, 24);
        let dz = g.thickness / nsub as f64;
        let (c0, c1) = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
        let mut inter = vec![0f32; rows * cols * nt];
        inter.par_chunks_mut(cols * nt).enumerate().for_each(|(r, row)| {
            let y = (r as f64 - c0) * pixel[0];
            for c in 0..cols {
                let x = (c as f64 - c1) * pixel[1];
                let acc = &mut row[c * nt..(c + 1) * nt];
                for s in 0..nsub {
                    let z = -g.thickness / 2.0 + (s as f64 + 0.5) * dz;
                    self.accumulate(g.point(x, y, z), dz as f32, acc);
                }
            }
        });
        (0..nt).map(|t| inter.iter().skip(t).step_by(nt).copied().collect()).collect()
    }

    /// Extent of tissue in a slice plane: max |y| and max |x| [mm] (None if the slice misses the head)
    pub fn slice_extent(&self, g: &SliceGeometry) -> Option<[f64; 2]> {
        let step = 2.0;
        let (lo, hi) = self.world_bounds();
        let half_diag = 0.5 * crate::geometry::norm(crate::geometry::sub(hi, lo));
        let reach = crate::geometry::norm(crate::geometry::sub(g.center, self.center())) + half_diag;
        let n = (reach / step).ceil() as i64;
        let zs = [-0.5 * g.thickness, 0.0, 0.5 * g.thickness];
        let ext = (-n..=n)
            .into_par_iter()
            .map(|iy| {
                let y = iy as f64 * step;
                let mut e = [f64::NEG_INFINITY; 2];
                for ix in -n..=n {
                    let x = ix as f64 * step;
                    if zs.iter().any(|&z| self.tissue_fraction(g.point(x, y, z)) > 0.02) {
                        e[0] = e[0].max(y.abs());
                        e[1] = e[1].max(x.abs());
                    }
                }
                e
            })
            .reduce(|| [f64::NEG_INFINITY; 2], |a, b| [a[0].max(b[0]), a[1].max(b[1])]);
        (ext[0].is_finite()).then_some(ext)
    }

    /// Localizer image of an arbitrary plane: values of the weighted tissue contrast on a grid of
    /// `pixel` [mm] over [u0, u1] × [v0, v1] (in-plane coordinates relative to `origin`)
    #[allow(clippy::too_many_arguments)]
    pub fn render_plane(&self, origin: V3, u: V3, v: V3, urange: (f64, f64), vrange: (f64, f64), pixel: f64, weights: &[f32]) -> (usize, usize, Vec<f32>) {
        let w = (((urange.1 - urange.0) / pixel).ceil() as usize).max(1);
        let h = (((vrange.1 - vrange.0) / pixel).ceil() as usize).max(1);
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
            let b = vrange.0 + (r as f64 + 0.5) * pixel;
            for (c, o) in row.iter_mut().enumerate() {
                let a = urange.0 + (c as f64 + 0.5) * pixel;
                let p = crate::geometry::add(crate::geometry::add(origin, crate::geometry::scale(u, a)), crate::geometry::scale(v, b));
                *o = self.contrast(p, weights);
            }
        });
        (w, h, out)
    }

    /// T1-weighted-like localizer contrast per tissue
    pub fn localizer_weights(&self) -> Vec<f32> {
        self.tissues
            .iter()
            .map(|t| match t.as_str() {
                "Gray matter" => 0.55,
                "White matter" => 0.8,
                "CSF" => 0.12,
                "Adipose tissue" => 1.0,
                "Muscle" => 0.35,
                "Cortical bone" => 0.02,
                "Bone marrow" => 0.75,
                _ => 0.5,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Orientation, StackSpec};

    #[test]
    fn builtin_brain_volume() {
        let v = builtin(BRAIN3D).unwrap();
        assert_eq!(v.dims, [189, 220, 184]);
        assert_eq!(v.tissues.len(), 6);
        // centre of the brain: mostly white/gray matter, fractions sum to about one
        let mut acc = vec![0f32; 6];
        v.accumulate([0.0, -18.0, 20.0], 1.0, &mut acc);
        let s: f32 = acc.iter().sum();
        assert!((s - 1.0).abs() < 0.05, "{acc:?}");
        assert!(acc[0] + acc[1] > 0.5);
        // outside the head: nothing
        assert_eq!(v.tissue_fraction([0.0, 0.0, 200.0]), 0.0);
        // an axial slice through the brain covers roughly the head size
        let g =
            StackSpec { orientation: Orientation::Transversal, angle1: 0.0, angle2: 0.0, center: [0.0, -15.0, 18.0], thickness: 3.0, num_slices: 1, gap: 0.0 }
                .slice(0);
        let e = v.slice_extent(&g).unwrap();
        assert!(e[0] > 80.0 && e[0] < 130.0 && e[1] > 60.0 && e[1] < 100.0, "{e:?}");
    }
}
