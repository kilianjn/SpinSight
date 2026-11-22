//! Slice geometry in patient coordinates for 3D phantoms.
//!
//! World coordinates are RAS [mm] as in NIfTI/MNI: +x = patient right, +y = anterior,
//! +z = superior (head). Images are displayed in radiological convention (patient right on the
//! image left, anterior / head at the top).
//!
//! A slice is described by its centre `c`, the in-plane unit vectors `u` (image right) and `v`
//! (image down) and the normal `n`; phantom coordinates (dim 0, dim 1) = (y along v, x along u).

pub type V3 = [f64; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}

/// Rotate `a` about the unit axis `k` by `theta` [rad] (Rodrigues)
pub fn rotate(a: V3, k: V3, theta: f64) -> V3 {
    let (s, c) = theta.sin_cos();
    add(add(scale(a, c), scale(cross(k, a), s)), scale(k, dot(k, a) * (1.0 - c)))
}

pub const RIGHT: V3 = [1.0, 0.0, 0.0];
pub const LEFT: V3 = [-1.0, 0.0, 0.0];
pub const ANTERIOR: V3 = [0.0, 1.0, 0.0];
pub const POSTERIOR: V3 = [0.0, -1.0, 0.0];
pub const SUPERIOR: V3 = [0.0, 0.0, 1.0];
pub const INFERIOR: V3 = [0.0, 0.0, -1.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Transversal,
    Sagittal,
    Coronal,
}

impl Orientation {
    pub const NAMES: [&'static str; 3] = ["Transversal", "Sagittal", "Coronal"];

    pub fn from_name(s: &str) -> Orientation {
        match s {
            "Sagittal" => Orientation::Sagittal,
            "Coronal" => Orientation::Coronal,
            _ => Orientation::Transversal,
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Orientation::Transversal => "Tra",
            Orientation::Sagittal => "Sag",
            Orientation::Coronal => "Cor",
        }
    }

    /// (u = image right, v = image down) in radiological display convention
    pub fn base_axes(self) -> (V3, V3) {
        match self {
            Orientation::Transversal => (LEFT, POSTERIOR),
            Orientation::Coronal => (LEFT, INFERIOR),
            Orientation::Sagittal => (POSTERIOR, INFERIOR),
        }
    }

    /// Orientations the two angulations tilt towards (rotation about u, about v)
    pub fn tilt_targets(self) -> (Orientation, Orientation) {
        match self {
            Orientation::Transversal => (Orientation::Coronal, Orientation::Sagittal),
            Orientation::Sagittal => (Orientation::Transversal, Orientation::Coronal),
            Orientation::Coronal => (Orientation::Transversal, Orientation::Sagittal),
        }
    }
}

/// Position and orientation of one slice
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceGeometry {
    pub center: V3,
    /// image right
    pub u: V3,
    /// image down
    pub v: V3,
    pub n: V3,
    pub thickness: f64,
}

impl SliceGeometry {
    /// World position of in-plane coordinates (x along u, y along v) and offset z along n
    pub fn point(&self, x: f64, y: f64, z: f64) -> V3 {
        add(add(add(self.center, scale(self.u, x)), scale(self.v, y)), scale(self.n, z))
    }
}

/// Slice stack parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackSpec {
    pub orientation: Orientation,
    /// angulations [°]: about the base image-right axis and about the base image-down axis
    pub angle1: f64,
    pub angle2: f64,
    /// stack centre (RAS) [mm]
    pub center: V3,
    pub thickness: f64,
    pub num_slices: usize,
    /// distance factor (gap / thickness)
    pub gap: f64,
}

impl StackSpec {
    /// Rotated (u, v, n)
    pub fn axes(&self) -> (V3, V3, V3) {
        let (u0, v0) = self.orientation.base_axes();
        let n0 = cross(u0, v0);
        let r = |a: V3| rotate(rotate(a, u0, self.angle1.to_radians()), v0, self.angle2.to_radians());
        (r(u0), r(v0), r(n0))
    }

    /// Distance between slice centres [mm]
    pub fn spacing(&self) -> f64 {
        self.thickness * (1.0 + self.gap)
    }

    pub fn slice(&self, index: usize) -> SliceGeometry {
        let (u, v, n) = self.axes();
        let offset = (index as f64 - (self.num_slices.max(1) as f64 - 1.0) / 2.0) * self.spacing();
        SliceGeometry { center: add(self.center, scale(n, offset)), u, v, n, thickness: self.thickness }
    }

    pub fn slices(&self) -> Vec<SliceGeometry> {
        (0..self.num_slices.max(1)).map(|i| self.slice(i)).collect()
    }
}

/// Orientation letters at the (top, bottom, left, right) edges of an image with axes (u, v)
pub fn edge_letters(u: V3, v: V3) -> [String; 4] {
    // dominant patient direction(s) of a display direction, e.g. "A" or "AH" for oblique slices
    let letter = |d: V3| -> String {
        let mut parts = [
            (d[0].abs(), if d[0] > 0.0 { 'R' } else { 'L' }),
            (d[1].abs(), if d[1] > 0.0 { 'A' } else { 'P' }),
            (d[2].abs(), if d[2] > 0.0 { 'H' } else { 'F' }),
        ];
        parts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut s = parts[0].1.to_string();
        if parts[1].0 > 0.25 {
            s.push(parts[1].1);
        }
        s
    };
    [letter(scale(v, -1.0)), letter(v), letter(scale(u, -1.0)), letter(u)]
}

/// "R 12.5" / "L 3.0" style position label for a coordinate along `axis` (0 = x, 1 = y, 2 = z)
pub fn position_label(axis: usize, value: f64) -> String {
    let (pos, neg) = match axis {
        0 => ('R', 'L'),
        1 => ('A', 'P'),
        _ => ('H', 'F'),
    };
    format!("{}{:.1}", if value >= 0.0 { pos } else { neg }, value.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_axes_are_right_handed_and_orthonormal() {
        for o in [Orientation::Transversal, Orientation::Sagittal, Orientation::Coronal] {
            let s = StackSpec { orientation: o, angle1: 17.0, angle2: -9.0, center: [0.0; 3], thickness: 3.0, num_slices: 5, gap: 0.2 };
            let (u, v, n) = s.axes();
            for (a, b) in [(u, v), (u, n), (v, n)] {
                assert!(dot(a, b).abs() < 1e-12);
            }
            for a in [u, v, n] {
                assert!((norm(a) - 1.0).abs() < 1e-12);
            }
            assert!(norm(sub(cross(u, v), n)) < 1e-12);
            let g0 = s.slice(0);
            let g4 = s.slice(4);
            assert!((norm(sub(g4.center, g0.center)) - 4.0 * 3.6).abs() < 1e-9);
        }
        let tra = StackSpec { orientation: Orientation::Transversal, angle1: 0.0, angle2: 0.0, center: [0.0; 3], thickness: 3.0, num_slices: 1, gap: 0.0 };
        assert_eq!(tra.axes().2, SUPERIOR);
        assert_eq!(edge_letters(LEFT, POSTERIOR), ["A".to_string(), "P".into(), "R".into(), "L".into()]);
    }
}
