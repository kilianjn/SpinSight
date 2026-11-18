//! Computational phantoms: shape loading (SVG polygons, TOML ellipses or PNG label images)
//! and analytic k-space synthesis (port of `phantom.py`, extended with label-image phantoms).
//!
//! The k-space of each shape is evaluated analytically on a sparse grid matched to the shape's
//! support, sinc-interpolated to the phantom grid and summed per tissue — exactly the Python
//! algorithm, but the polygon transform uses a separable vertex formulation and all shapes are
//! processed in parallel. Results are cached on disk.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::array::{Array2, C64};
use crate::constants;
use crate::recon::{get_k_axis, rotation_matrix, sinc_interpolate, sinc_kernel};
use crate::svg;

/// Geometry of a shape. Coordinates are (dim 0, dim 1) = (y, x) [mm].
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Polygon {
        vertices: Vec<[f64; 2]>,
    },
    Ellipse {
        pos: [f64; 2],
        radius: [f64; 2],
        angle: f64,
        negative: bool,
    },
    /// Pixelated region of a label image: `mask` (1 = tissue) with its top-left corner at
    /// `origin` (y, x) [mm] and pixel size `pixel` (dy, dx) [mm]
    Raster {
        mask: Arc<Array2<u8>>,
        origin: [f64; 2],
        pixel: [f64; 2],
    },
}

#[derive(Debug, Clone)]
pub enum PhantomSource {
    Svg {
        svg: String,
        colormap: String,
    },
    Toml {
        toml: String,
    },
    /// PNG label image + colormap (`'#rrggbb' = 'Tissue'` and `pixel_size = <mm>`)
    Raster {
        png: Vec<u8>,
        colormap: String,
    },
}

impl PhantomSource {
    fn hash_bytes(&self) -> Vec<u8> {
        match self {
            PhantomSource::Svg { svg, colormap } => [svg.as_bytes(), b"\0", colormap.as_bytes()].concat(),
            PhantomSource::Toml { toml } => toml.as_bytes().to_vec(),
            PhantomSource::Raster { png, colormap } => [png.as_slice(), b"\0", colormap.as_bytes()].concat(),
        }
    }
}

/// Phantom geometry and k-space grid (cheap to compute; the k-space itself is in [`PhantomKspace`])
#[derive(Debug, Clone)]
pub struct Phantom {
    pub name: String,
    pub tissues: Vec<String>,
    pub shapes: Vec<Vec<Shape>>,
    pub support: [f64; 2],
    pub center: [f64; 2],
    pub matrix: [usize; 2],
    pub k_axes: [Vec<f64>; 2],
    pub min_voxel_size: f64,
    /// content hash used as cache key
    pub hash: u64,
}

/// Phantom k-space per tissue (same order as `Phantom::tissues`)
#[derive(Debug, Clone)]
pub struct PhantomKspace {
    pub kspace: Vec<Array2<C64>>,
}

// ---------------------------------------------------------------------------------------------
// built-in and user phantoms

const BUILTIN: &[(&str, &str, &str)] = &[
    ("brain", include_str!("../spinsight/phantoms/brain/brain.svg"), include_str!("../spinsight/phantoms/brain/brain.toml")),
    ("abdomen", include_str!("../spinsight/phantoms/abdomen/abdomen.svg"), include_str!("../spinsight/phantoms/abdomen/abdomen.toml")),
];
const BUILTIN_TOML: &[(&str, &str)] = &[("Shepp-Logan", include_str!("../spinsight/phantoms/Shepp-Logan/Shepp-Logan.toml"))];

pub fn builtin_phantom_names() -> Vec<&'static str> {
    let mut v: Vec<&str> = BUILTIN.iter().map(|b| b.0).collect();
    v.extend(BUILTIN_TOML.iter().map(|b| b.0));
    v
}

/// Directories searched for user phantoms (`<dir>/<name>/<name>.svg|png|toml`)
pub fn user_phantom_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = directories::ProjectDirs::from("se", "uu", "SpinSight") {
        dirs.push(d.data_dir().join("phantoms"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("phantoms"));
        }
    }
    if let Ok(extra) = std::env::var("SPINSIGHT_PHANTOMS") {
        dirs.push(PathBuf::from(extra));
    }
    dirs
}

/// Directory where imported and self-made phantoms are stored
pub fn user_phantom_write_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("se", "uu", "SpinSight").map(|d| d.data_dir().join("phantoms"))
}

fn user_phantom(dir: &Path, name: &str) -> Option<PhantomSource> {
    let base = dir.join(name);
    let toml = base.join(format!("{name}.toml"));
    let svg = base.join(format!("{name}.svg"));
    let png = base.join(format!("{name}.png"));
    if svg.is_file() {
        let svg_text = std::fs::read_to_string(&svg).ok()?;
        let colormap = std::fs::read_to_string(&toml).ok()?;
        Some(PhantomSource::Svg { svg: svg_text, colormap })
    } else if png.is_file() {
        Some(PhantomSource::Raster { png: std::fs::read(&png).ok()?, colormap: std::fs::read_to_string(&toml).ok()? })
    } else if toml.is_file() {
        Some(PhantomSource::Toml { toml: std::fs::read_to_string(&toml).ok()? })
    } else {
        None
    }
}

/// Names of all available phantoms (built-in first, then user phantoms)
pub fn phantom_names() -> Vec<String> {
    let mut names: Vec<String> = builtin_phantom_names().iter().map(|s| s.to_string()).collect();
    for dir in user_phantom_dirs() {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut found: Vec<String> = rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !names.contains(n) && user_phantom(&dir, n).is_some())
                .collect();
            found.sort();
            names.extend(found);
        }
    }
    names
}

/// Whether `name` is a built-in phantom (built-in phantoms cannot be deleted)
pub fn is_builtin(name: &str) -> bool {
    builtin_phantom_names().contains(&name)
}

pub fn phantom_source(name: &str) -> Option<PhantomSource> {
    if let Some(b) = BUILTIN.iter().find(|b| b.0 == name) {
        return Some(PhantomSource::Svg { svg: b.1.to_string(), colormap: b.2.to_string() });
    }
    if let Some(b) = BUILTIN_TOML.iter().find(|b| b.0 == name) {
        return Some(PhantomSource::Toml { toml: b.1.to_string() });
    }
    user_phantom_dirs().iter().find_map(|d| user_phantom(d, name))
}

// ---------------------------------------------------------------------------------------------
// shape loading

fn polygon_area(v: &[[f64; 2]]) -> f64 {
    // Python: sum((y - roll(y,1)) * (x + roll(x,1))) / 2 with coords = (y, x)
    let n = v.len();
    let mut s = 0.0;
    for i in 0..n {
        let p = v[(i + n - 1) % n];
        s += (v[i][0] - p[0]) * (v[i][1] + p[1]);
    }
    s / 2.0
}

fn load_shapes_svg(svg_text: &str, colormap: &str) -> Result<Vec<(String, Vec<Shape>)>, String> {
    let paths = svg::read_paths(svg_text)?;
    // shapes per hexcolor in order of first appearance
    let mut by_color: Vec<(Option<String>, Vec<Shape>)> = Vec::new();
    for path in &paths {
        let mut polys: Vec<Vec<[f64; 2]>> = Vec::new();
        for sub in svg::continuous_subpaths(&path.segments) {
            if sub.first().map(|s| s.start) != sub.last().map(|s| s.end) {
                return Err("All paths in SVG file must be closed".into());
            }
            if sub.iter().any(|s| s.kind != svg::SegKind::Line) {
                eprintln!("warning: only SVG line segments are supported (curved subpath ignored)");
                continue;
            }
            let verts: Vec<[f64; 2]> = sub
                .iter()
                .map(|s| {
                    let p = svg::apply_transform(&path.transform, s.start);
                    [p[1], p[0]] // (y, x)
                })
                .collect();
            if !verts.is_empty() {
                polys.push(verts);
            }
        }
        if polys.iter().map(|p| polygon_area(p)).sum::<f64>() < 0.0 {
            // invert polygons to make total area positive
            for p in polys.iter_mut() {
                p.reverse();
            }
        }
        let shapes = polys.into_iter().map(|vertices| Shape::Polygon { vertices });
        match by_color.iter_mut().find(|(c, _)| *c == path.fill) {
            Some((_, v)) => v.extend(shapes),
            None => by_color.push((path.fill.clone(), shapes.collect())),
        }
    }
    let cmap: HashMap<String, String> = toml::from_str::<toml::Table>(colormap)
        .map_err(|e| format!("invalid colormap: {e}"))?
        .into_iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
        .collect();
    let mut tissues: Vec<(String, Vec<Shape>)> = Vec::new();
    for (color, shapes) in by_color {
        let tissue = color.as_ref().and_then(|c| cmap.get(c));
        match tissue {
            Some(t) if constants::tissue(t).is_some() => match tissues.iter_mut().find(|(n, _)| n == t) {
                Some(entry) => entry.1 = shapes,
                None => tissues.push((t.clone(), shapes)),
            },
            Some(t) => eprintln!("warning: tissue \"{t}\" from colormap not defined"),
            None => eprintln!("warning: tissue color {color:?} in phantom not defined in colormap"),
        }
    }
    Ok(tissues)
}

fn load_shapes_toml(text: &str) -> Result<Vec<(String, Vec<Shape>)>, String> {
    // keep the table order of the file: parse the keys in order of appearance
    let table: toml::Table = toml::from_str(text).map_err(|e| format!("invalid phantom TOML: {e}"))?;
    let mut order: Vec<String> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("[[") {
            if let Some(name) = rest.strip_suffix("]]") {
                let name = name.trim().to_string();
                if !order.contains(&name) {
                    order.push(name);
                }
            }
        }
    }
    for k in table.keys() {
        if !order.contains(k) {
            order.push(k.clone());
        }
    }
    let mut out = Vec::new();
    for key in order {
        let Some(list) = table.get(&key).and_then(|v| v.as_array()) else { continue };
        let mut shapes = Vec::new();
        for item in list {
            let Some(t) = item.as_table() else { continue };
            let num = |v: &toml::Value| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)).unwrap_or(0.0);
            let pair = |k: &str| -> [f64; 2] {
                let a = t.get(k).and_then(|v| v.as_array()).cloned().unwrap_or_default();
                // reversed: [x, y] in the file → (y, x)
                [a.get(1).map(num).unwrap_or(0.0), a.first().map(num).unwrap_or(0.0)]
            };
            match t.get("type").and_then(|v| v.as_str()) {
                Some("ellipse") => shapes.push(Shape::Ellipse {
                    pos: pair("pos"),
                    radius: pair("radius"),
                    angle: t.get("angle").map(num).unwrap_or(0.0),
                    negative: t.get("negative").and_then(|v| v.as_bool()).unwrap_or(false),
                }),
                other => eprintln!("warning: unsupported shape type {other:?} in phantom TOML"),
            }
        }
        out.push((key.replace('_', " "), shapes));
    }
    Ok(out)
}

/// Colour → tissue entries (`'#rrggbb' = 'Tissue'`) of a colormap, in file order
pub fn parse_colormap(text: &str) -> Result<Vec<(String, String)>, String> {
    let table: toml::Table = toml::from_str(text).map_err(|e| format!("invalid colormap: {e}"))?;
    let mut entries: Vec<(String, String)> = table.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.to_lowercase(), s.to_string()))).collect();
    // file order (toml::Table is sorted)
    let pos = |k: &str| text.to_lowercase().find(&k.to_lowercase()).unwrap_or(usize::MAX);
    entries.sort_by_key(|(k, _)| pos(k));
    Ok(entries)
}

/// Pixel size [mm] of a raster phantom colormap (`pixel_size = 0.5` or `[dy, dx]`)
fn colormap_pixel_size(text: &str) -> Result<[f64; 2], String> {
    let table: toml::Table = toml::from_str(text).map_err(|e| format!("invalid colormap: {e}"))?;
    let num = |v: &toml::Value| v.as_float().or_else(|| v.as_integer().map(|i| i as f64));
    match table.get("pixel_size") {
        Some(toml::Value::Array(a)) if a.len() == 2 => Ok([num(&a[0]).unwrap_or(1.0), num(&a[1]).unwrap_or(1.0)]),
        Some(v) => num(v).map(|p| [p, p]).ok_or_else(|| "pixel_size must be a number".to_string()),
        None => Ok([1.0, 1.0]),
    }
}

pub fn hexcolor(r: u8, g: u8, b: u8) -> String {
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn load_shapes_raster(png: &[u8], colormap: &str) -> Result<Vec<(String, Vec<Shape>)>, String> {
    let img = image::load_from_memory(png).map_err(|e| format!("invalid label image: {e}"))?.into_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pixel = colormap_pixel_size(colormap)?;
    let cmap = parse_colormap(colormap)?;
    // tissues in colormap order (several colours may map to the same tissue)
    let mut tissues: Vec<String> = Vec::new();
    let mut color_to_tissue: HashMap<[u8; 3], usize> = HashMap::new();
    for (c, t) in &cmap {
        if constants::tissue(t).is_none() {
            eprintln!("warning: tissue \"{t}\" from colormap not defined");
            continue;
        }
        let Some(rgb) = parse_hex(c) else { continue };
        let ti = match tissues.iter().position(|x| x == t) {
            Some(i) => i,
            None => {
                tissues.push(t.clone());
                tissues.len() - 1
            }
        };
        color_to_tissue.insert(rgb, ti);
    }
    let mut labels = vec![u8::MAX; w * h];
    for (i, p) in img.pixels().enumerate() {
        if p[3] >= 128 {
            if let Some(&t) = color_to_tissue.get(&[p[0], p[1], p[2]]) {
                labels[i] = t as u8;
            }
        }
    }
    Ok(raster_shapes_from_labels(&labels, w, h, pixel, [0.0, 0.0], &tissues))
}

/// One raster shape per tissue (cropped to its bounding box) from a label map (`u8::MAX` = background)
pub fn raster_shapes_from_labels(labels: &[u8], w: usize, h: usize, pixel: [f64; 2], origin: [f64; 2], tissues: &[String]) -> Vec<(String, Vec<Shape>)> {
    let mut out = Vec::new();
    for (ti, t) in tissues.iter().enumerate() {
        let (mut r0, mut r1, mut c0, mut c1) = (usize::MAX, 0usize, usize::MAX, 0usize);
        for r in 0..h {
            for c in 0..w {
                if labels[r * w + c] as usize == ti {
                    r0 = r0.min(r);
                    r1 = r1.max(r);
                    c0 = c0.min(c);
                    c1 = c1.max(c);
                }
            }
        }
        if r0 == usize::MAX {
            continue;
        }
        let (hh, ww) = (r1 - r0 + 1, c1 - c0 + 1);
        let mut mask = Array2::<u8>::zeros(hh, ww);
        for r in 0..hh {
            for c in 0..ww {
                mask.data[r * ww + c] = u8::from(labels[(r0 + r) * w + c0 + c] as usize == ti);
            }
        }
        let shape = Shape::Raster { mask: Arc::new(mask), origin: [origin[0] + r0 as f64 * pixel[0], origin[1] + c0 as f64 * pixel[1]], pixel };
        out.push((t.clone(), vec![shape]));
    }
    out
}

fn parse_hex(c: &str) -> Option<[u8; 3]> {
    let c = c.trim().trim_start_matches('#');
    if c.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(c, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

fn shape_bounds(shape: &Shape) -> ([f64; 2], [f64; 2]) {
    match shape {
        Shape::Polygon { vertices } => {
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for v in vertices {
                for d in 0..2 {
                    lo[d] = lo[d].min(v[d]);
                    hi[d] = hi[d].max(v[d]);
                }
            }
            (lo, hi)
        }
        Shape::Ellipse { pos, radius, angle, .. } => {
            let r = rotation_matrix(angle.to_radians());
            let half = [r[0][0].abs() * radius[0] + r[0][1].abs() * radius[1], r[1][0].abs() * radius[0] + r[1][1].abs() * radius[1]];
            ([pos[0] - half[0], pos[1] - half[1]], [pos[0] + half[0], pos[1] + half[1]])
        }
        Shape::Raster { mask, origin, pixel } => (*origin, [origin[0] + mask.rows as f64 * pixel[0], origin[1] + mask.cols as f64 * pixel[1]]),
    }
}

fn support_and_center<'a>(shapes: impl IntoIterator<Item = &'a Shape>) -> ([f64; 2], [f64; 2]) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for s in shapes {
        let (a, b) = shape_bounds(s);
        for d in 0..2 {
            lo[d] = lo[d].min(a[d]);
            hi[d] = hi[d].max(b[d]);
        }
    }
    ([hi[0] - lo[0], hi[1] - lo[1]], [(hi[0] + lo[0]) / 2.0, (hi[1] + lo[1]) / 2.0])
}

fn k_axes_for(matrix: [usize; 2], support: [f64; 2]) -> [Vec<f64>; 2] {
    [get_k_axis(matrix[0], support[0] / matrix[0] as f64), get_k_axis(matrix[1], support[1] / matrix[1] as f64)]
}

fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Load the geometry of phantom `name` (fast; no k-space computation)
pub fn load(name: &str, min_voxel_size: f64) -> Result<Phantom, String> {
    let source = phantom_source(name).ok_or_else(|| format!("Phantom {name}.svg/png/toml not found"))?;
    let tissue_shapes = match &source {
        PhantomSource::Svg { svg, colormap } => load_shapes_svg(svg, colormap)?,
        PhantomSource::Toml { toml } => load_shapes_toml(toml)?,
        PhantomSource::Raster { png, colormap } => load_shapes_raster(png, colormap)?,
    };
    if tissue_shapes.iter().all(|(_, s)| s.is_empty()) {
        return Err(format!("Phantom {name} contains no shapes"));
    }
    let (support, center) = support_and_center(tissue_shapes.iter().flat_map(|(_, s)| s.iter()));
    let matrix = [(support[0] / min_voxel_size).ceil() as usize, (support[1] / min_voxel_size).ceil() as usize];
    let k_axes = k_axes_for(matrix, support);
    let mut hash = fnv1a(&source.hash_bytes(), 0xcbf29ce484222325);
    hash = fnv1a(&min_voxel_size.to_le_bytes(), hash);
    hash = fnv1a(CACHE_VERSION.as_bytes(), hash);
    let (tissues, shapes) = tissue_shapes.into_iter().unzip();
    Ok(Phantom { name: name.to_string(), tissues, shapes, support, center, matrix, k_axes, min_voxel_size, hash })
}

// ---------------------------------------------------------------------------------------------
// analytic k-space

/// k-space of a polygon on the grid k_axes[0] × k_axes[1] (`kspace_for_polygon`).
///
/// Uses Σ_e L(k·n) sinc(k·L) e^{-2πi k·r_c} = Σ_e (k×L)/(k·L) (E(r_e) − E(r_{e+1})) / (2i π)... with
/// E(r) = e^{-2πi k·r}, which is separable in k₀, k₁ so that no transcendental function has to
/// be evaluated per grid point and edge.
fn kspace_polygon(vertices: &[[f64; 2]], a0: &[f64], a1: &[f64]) -> Array2<C64> {
    let nv = vertices.len();
    let (n0, n1) = (a0.len(), a1.len());
    let mut out: Array2<C64> = Array2::zeros(n0, n1);
    if nv < 2 {
        return out;
    }
    let area = polygon_area(vertices);
    // per-vertex separable phase factors along dim 1
    let e1: Vec<Vec<C64>> = vertices.iter().map(|v| a1.iter().map(|&k| C64::from_polar(1.0, -2.0 * PI * k * v[1])).collect()).collect();
    let edges: Vec<[f64; 2]> = (0..nv).map(|e| [vertices[(e + 1) % nv][0] - vertices[e][0], vertices[(e + 1) % nv][1] - vertices[e][1]]).collect();
    out.data.par_chunks_mut(n1).enumerate().for_each(|(p, row)| {
        let k0 = a0[p];
        let mut acc = vec![C64::new(0.0, 0.0); n1];
        let mut e_cur = vec![C64::new(0.0, 0.0); n1];
        let mut e_next = vec![C64::new(0.0, 0.0); n1];
        let e0: Vec<C64> = vertices.iter().map(|v| C64::from_polar(1.0, -2.0 * PI * k0 * v[0])).collect();
        for q in 0..n1 {
            e_cur[q] = e0[0] * e1[0][q];
        }
        for e in 0..nv {
            let nxt = (e + 1) % nv;
            let lv = edges[e];
            let f0 = e0[nxt];
            let e1n = &e1[nxt];
            let a_k0 = k0 * lv[0];
            let c_k0 = -k0 * lv[1];
            for q in 0..n1 {
                e_next[q] = f0 * e1n[q];
                let k1 = a1[q];
                let a = a_k0 + k1 * lv[1]; // k·L
                let cross = c_k0 + k1 * lv[0]; // k×L = L (k·n)
                let term = if a.abs() > 1e-7 {
                    (e_cur[q] - e_next[q]) * (cross / a)
                } else {
                    // limit k·L → 0: (E_e − E_{e+1})/(k·L) → 2πi E(r_c)
                    let half = C64::from_polar(1.0, -PI * a);
                    e_cur[q] * half * C64::new(0.0, 2.0 * PI * cross) * crate::recon::sinc(a)
                };
                acc[q] += term;
            }
            std::mem::swap(&mut e_cur, &mut e_next);
        }
        for q in 0..n1 {
            let k1 = a1[q];
            let k2 = k0 * k0 + k1 * k1;
            row[q] = if k2 == 0.0 { C64::new(area, 0.0) } else { acc[q] / (4.0 * PI * PI * k2) };
        }
    });
    out
}

/// k-space of an ellipse (`kspace_for_ellipse`)
fn kspace_ellipse(pos: [f64; 2], radius: [f64; 2], angle: f64, negative: bool, a0: &[f64], a1: &[f64]) -> Array2<C64> {
    let r = rotation_matrix(angle.to_radians());
    let (n0, n1) = (a0.len(), a1.len());
    let mut out: Array2<C64> = Array2::zeros(n0, n1);
    let scale = radius[0] * radius[1] * if negative { -1.0 } else { 1.0 };
    out.data.par_chunks_mut(n1).enumerate().for_each(|(p, row)| {
        let k0 = a0[p];
        for (q, v) in row.iter_mut().enumerate() {
            let k1 = a1[q];
            // k @ R @ diag(radius)
            let u0 = (k0 * r[0][0] + k1 * r[1][0]) * radius[0];
            let u1 = (k0 * r[0][1] + k1 * r[1][1]) * radius[1];
            let rr = (u0 * u0 + u1 * u1).sqrt();
            let mag = if rr != 0.0 { libm::j1(2.0 * PI * rr) / rr } else { PI };
            *v = C64::from_polar(mag * scale, -2.0 * PI * (k0 * pos[0] + k1 * pos[1]));
        }
    });
    out
}

/// Exact k-space of a pixelated region: FT(pixel) × Σ_pixels e^{-2πi k·r}, evaluated as
/// E₀ · M · E₁ with E_d[p, m] = e^{-2πi k_p r_m} (separable, two complex matrix products).
/// Coordinates are taken relative to `center`.
fn kspace_raster(mask: &Array2<u8>, origin: [f64; 2], pixel: [f64; 2], k_axes: &[Vec<f64>; 2], center: [f64; 2]) -> Array2<C64> {
    let (h, w) = (mask.rows, mask.cols);
    let (n0, n1) = (k_axes[0].len(), k_axes[1].len());
    let mut out: Array2<C64> = Array2::zeros(n0, n1);
    if h == 0 || w == 0 || n0 == 0 || n1 == 0 {
        return out;
    }
    let y: Vec<f64> = (0..h).map(|m| origin[0] + (m as f64 + 0.5) * pixel[0] - center[0]).collect();
    let x: Vec<f64> = (0..w).map(|n| origin[1] + (n as f64 + 0.5) * pixel[1] - center[1]).collect();
    // E1 (w × n1), E0 (n0 × h)
    let mut e1 = vec![C64::new(0.0, 0.0); w * n1];
    e1.par_chunks_mut(n1).enumerate().for_each(|(n, row)| {
        for (q, v) in row.iter_mut().enumerate() {
            *v = C64::from_polar(1.0, -2.0 * PI * k_axes[1][q] * x[n]);
        }
    });
    let mut e0 = vec![C64::new(0.0, 0.0); n0 * h];
    e0.par_chunks_mut(h).enumerate().for_each(|(p, row)| {
        for (m, v) in row.iter_mut().enumerate() {
            *v = C64::from_polar(1.0, -2.0 * PI * k_axes[0][p] * y[m]);
        }
    });
    let mf: Vec<f64> = mask.data.iter().map(|&b| f64::from(b)).collect();
    // T = M (h × w) · E1 (w × n1), complex (h × n1)
    let mut t = vec![C64::new(0.0, 0.0); h * n1];
    unsafe {
        for part in 0..2 {
            matrixmultiply::dgemm(
                h,
                w,
                n1,
                1.0,
                mf.as_ptr(),
                w as isize,
                1,
                (e1.as_ptr() as *const f64).add(part),
                2 * n1 as isize,
                2,
                0.0,
                (t.as_mut_ptr() as *mut f64).add(part),
                2 * n1 as isize,
                2,
            );
        }
        // out = E0 (n0 × h) · T (h × n1): re = E0r Tr − E0i Ti, im = E0r Ti + E0i Tr
        let e0p = e0.as_ptr() as *const f64;
        let tp = t.as_ptr() as *const f64;
        let op = out.data.as_mut_ptr() as *mut f64;
        let mm = |alpha: f64, a_off: usize, b_off: usize, c_off: usize, beta: f64| {
            matrixmultiply::dgemm(
                n0,
                h,
                n1,
                alpha,
                e0p.add(a_off),
                2 * h as isize,
                2,
                tp.add(b_off),
                2 * n1 as isize,
                2,
                beta,
                op.add(c_off),
                2 * n1 as isize,
                2,
            );
        };
        mm(1.0, 0, 0, 0, 0.0);
        mm(-1.0, 1, 1, 0, 1.0);
        mm(1.0, 0, 1, 1, 0.0);
        mm(1.0, 1, 0, 1, 1.0);
    }
    let area = pixel[0] * pixel[1];
    let s1: Vec<f64> = k_axes[1].iter().map(|&k| crate::recon::sinc(k * pixel[1])).collect();
    out.data.par_chunks_mut(n1).enumerate().for_each(|(p, row)| {
        let s0 = area * crate::recon::sinc(k_axes[0][p] * pixel[0]);
        for (q, v) in row.iter_mut().enumerate() {
            *v *= s0 * s1[q];
        }
    });
    out
}

/// Multiply by e^{2πi k·offset} on the separable grid (`offset_FOV`)
fn offset_fov(k: &mut Array2<C64>, a0: &[f64], a1: &[f64], offset: [f64; 2]) {
    let f1: Vec<C64> = a1.iter().map(|&k1| C64::from_polar(1.0, 2.0 * PI * k1 * offset[1])).collect();
    let cols = k.cols;
    k.data.par_chunks_mut(cols).enumerate().for_each(|(p, row)| {
        let f0 = C64::from_polar(1.0, 2.0 * PI * a0[p] * offset[0]);
        for (q, v) in row.iter_mut().enumerate() {
            *v *= f0 * f1[q];
        }
    });
}

/// k-space of one shape on the phantom grid (`kspace_for_shape`)
fn kspace_for_shape(shape: &Shape, k_axes: &[Vec<f64>; 2], axes_support: [f64; 2], axes_center: [f64; 2]) -> Option<Array2<C64>> {
    if let Shape::Raster { mask, origin, pixel } = shape {
        // exact on the full grid; no sparse grid / interpolation needed
        return Some(kspace_raster(mask, *origin, *pixel, k_axes, axes_center));
    }
    let (shape_support, shape_center) = support_and_center(std::iter::once(shape));
    let matrix = [k_axes[0].len(), k_axes[1].len()];
    let sparse_matrix = [0, 1].map(|d| (matrix[d] as f64 * shape_support[d] / axes_support[d]).ceil() as usize);
    if sparse_matrix.contains(&0) {
        return None;
    }
    let sparse_support = [0, 1].map(|d| axes_support[d] * sparse_matrix[d] as f64 / matrix[d] as f64);
    let sparse_axes = k_axes_for(sparse_matrix, sparse_support);
    let mut k = match shape {
        Shape::Polygon { vertices } => kspace_polygon(vertices, &sparse_axes[0], &sparse_axes[1]),
        Shape::Ellipse { pos, radius, angle, negative } => kspace_ellipse(*pos, *radius, *angle, *negative, &sparse_axes[0], &sparse_axes[1]),
        Shape::Raster { .. } => unreachable!(),
    };
    offset_fov(&mut k, &sparse_axes[0], &sparse_axes[1], shape_center);
    let k0 = sinc_kernel(&k_axes[0], &sparse_axes[0], sparse_support[0]);
    let k1 = sinc_kernel(&k_axes[1], &sparse_axes[1], sparse_support[1]);
    let mut out = sinc_interpolate(&k, &k0, &k1);
    offset_fov(&mut out, &k_axes[0], &k_axes[1], [axes_center[0] - shape_center[0], axes_center[1] - shape_center[1]]);
    Some(out)
}

pub const CACHE_VERSION: &str = "spinsight-phantom-kspace-v2";

pub fn cache_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("se", "uu", "SpinSight").map(|d| d.cache_dir().join("phantoms"))
}

fn safe_name(name: &str) -> String {
    name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn cache_file(ph: &Phantom) -> Option<PathBuf> {
    cache_dir().map(|d| d.join(format!("{}-{:016x}.bin", safe_name(&ph.name), ph.hash)))
}

/// Total size [bytes] and number of files in the phantom k-space cache
pub fn cache_size() -> (u64, usize) {
    let Some(dir) = cache_dir() else { return (0, 0) };
    let Ok(rd) = std::fs::read_dir(dir) else { return (0, 0) };
    rd.filter_map(|e| e.ok()).filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).fold((0, 0), |(b, n), m| (b + m.len(), n + 1))
}

/// Remove cache files written by older versions (different file format)
pub fn purge_outdated_cache() {
    let Some(dir) = cache_dir() else { return };
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let mut magic = [0u8; 8];
        let ok = std::fs::File::open(&p).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic)).is_ok();
        if !ok || &magic != b"SSPKSP02" {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Delete all cached phantom k-space files
pub fn clear_cache() -> Result<(), String> {
    let Some(dir) = cache_dir() else { return Ok(()) };
    if dir.is_dir() {
        for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
            if e.path().is_file() {
                std::fs::remove_file(e.path()).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

/// Remove cache files of phantom `name` other than `keep` (outdated versions)
fn remove_stale_cache(name: &str, keep: &Path) {
    let Some(dir) = cache_dir() else { return };
    let prefix = format!("{}-", safe_name(name));
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let fname = e.file_name().to_string_lossy().to_string();
            // "<name>-<16 hex digits>.bin": the hash part must not contain further dashes
            let is_this =
                fname.strip_prefix(&prefix).and_then(|r| r.strip_suffix(".bin")).is_some_and(|h| h.len() == 16 && h.chars().all(|c| c.is_ascii_hexdigit()));
            if is_this && p != keep {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}

fn read_cache(ph: &Phantom) -> Option<PhantomKspace> {
    let bytes = std::fs::read(cache_file(ph)?).ok()?;
    let mut pos = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = bytes.get(pos..pos + n)?;
        pos += n;
        Some(s)
    };
    if take(8)? != b"SSPKSP02" {
        return None;
    }
    let u = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap()) as usize;
    let nt = u(take(8)?);
    let rows = u(take(8)?);
    let cols = u(take(8)?);
    if nt != ph.tissues.len() || [rows, cols] != ph.matrix {
        return None;
    }
    let mut kspace = Vec::with_capacity(nt);
    for _ in 0..nt {
        let raw = take(rows * cols * 8)?;
        let data = raw
            .chunks_exact(8)
            .map(|c| C64::new(f32::from_le_bytes(c[..4].try_into().unwrap()) as f64, f32::from_le_bytes(c[4..].try_into().unwrap()) as f64))
            .collect();
        kspace.push(Array2::from_vec(rows, cols, data));
    }
    Some(PhantomKspace { kspace })
}

fn write_cache(ph: &Phantom, k: &PhantomKspace) {
    let Some(file) = cache_file(ph) else { return };
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // single precision halves the file size; the relative error (~1e-7) is far below the noise
    let mut bytes = Vec::with_capacity(32 + k.kspace.iter().map(|a| a.data.len() * 8).sum::<usize>());
    bytes.extend_from_slice(b"SSPKSP02");
    for v in [k.kspace.len(), ph.matrix[0], ph.matrix[1]] {
        bytes.extend_from_slice(&(v as u64).to_le_bytes());
    }
    for a in &k.kspace {
        for z in &a.data {
            bytes.extend_from_slice(&(z.re as f32).to_le_bytes());
            bytes.extend_from_slice(&(z.im as f32).to_le_bytes());
        }
    }
    let tmp = file.with_extension("tmp");
    if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &file).is_ok() {
        remove_stale_cache(&ph.name, &file);
    }
}

/// Progress of a phantom k-space computation (shapes done / total)
#[derive(Debug, Default)]
pub struct Progress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

/// Compute the phantom k-space (no cache)
pub fn compute_kspace(ph: &Phantom, progress: Option<&Arc<Progress>>) -> PhantomKspace {
    let jobs: Vec<(usize, &Shape)> = ph.shapes.iter().enumerate().flat_map(|(t, shapes)| shapes.iter().map(move |s| (t, s))).collect();
    if let Some(p) = progress {
        p.total.store(jobs.len(), Ordering::Relaxed);
        p.done.store(0, Ordering::Relaxed);
    }
    // the largest shapes first for better load balancing
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    let size = |s: &Shape| {
        let (sup, _) = support_and_center(std::iter::once(s));
        sup[0]
            * sup[1]
            * match s {
                Shape::Polygon { vertices } => vertices.len() as f64,
                Shape::Ellipse { .. } => 4.0,
                Shape::Raster { mask, .. } => mask.rows.max(mask.cols) as f64,
            }
    };
    order.sort_by(|&a, &b| size(jobs[b].1).partial_cmp(&size(jobs[a].1)).unwrap_or(std::cmp::Ordering::Equal));
    let nt = ph.tissues.len();
    let [r, c] = ph.matrix;
    let zero = || (0..nt).map(|_| Array2::<C64>::zeros(r, c)).collect::<Vec<_>>();
    let kspace = order
        .par_iter()
        .fold(zero, |mut acc, &j| {
            let (t, shape) = jobs[j];
            if let Some(k) = kspace_for_shape(shape, &ph.k_axes, ph.support, ph.center) {
                for (a, b) in acc[t].data.iter_mut().zip(k.data.iter()) {
                    *a += *b;
                }
            }
            if let Some(p) = progress {
                p.done.fetch_add(1, Ordering::Relaxed);
            }
            acc
        })
        .reduce(zero, |mut a, b| {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                for (u, v) in x.data.iter_mut().zip(y.data.iter()) {
                    *u += *v;
                }
            }
            a
        });
    PhantomKspace { kspace }
}

/// Phantom k-space from the disk cache, or computed (and cached)
pub fn load_kspace(ph: &Phantom, progress: Option<&Arc<Progress>>) -> PhantomKspace {
    if let Some(k) = read_cache(ph) {
        return k;
    }
    let k = compute_kspace(ph, progress);
    write_cache(ph, &k);
    k
}

pub fn is_cached(ph: &Phantom) -> bool {
    cache_file(ph).is_some_and(|f| f.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// direct evaluation of the Python polygon formula
    fn polygon_reference(v: &[[f64; 2]], k: [f64; 2]) -> C64 {
        let n = v.len();
        if k == [0.0, 0.0] {
            return C64::new(polygon_area(v), 0.0);
        }
        let mut s = C64::new(0.0, 0.0);
        for e in 0..n {
            let r = v[e];
            let lv = [v[(e + 1) % n][0] - r[0], v[(e + 1) % n][1] - r[1]];
            let l = (lv[0] * lv[0] + lv[1] * lv[1]).sqrt();
            let t = [lv[0] / l, lv[1] / l];
            let nn = [-t[1], t[0]];
            let rc = [r[0] + lv[0] / 2.0, r[1] + lv[1] / 2.0];
            let kn = k[0] * nn[0] + k[1] * nn[1];
            let kl = k[0] * lv[0] + k[1] * lv[1];
            s += C64::from_polar(l * kn * crate::recon::sinc(kl), -2.0 * PI * (k[0] * rc[0] + k[1] * rc[1]));
        }
        s * C64::new(0.0, 1.0) / (2.0 * PI * (k[0] * k[0] + k[1] * k[1]))
    }

    #[test]
    fn polygon_transform_matches_reference() {
        let v = [[0.0, 0.0], [1.0, 3.0], [4.0, 2.5], [5.0, -1.0], [2.0, -2.0], [2.0, -2.0 + 1e-9]];
        let a0: Vec<f64> = (-6..6).map(|i| i as f64 * 0.07).collect();
        let a1: Vec<f64> = (-5..7).map(|i| i as f64 * 0.05).collect();
        let k = kspace_polygon(&v[..5], &a0, &a1);
        for (p, &k0) in a0.iter().enumerate() {
            for (q, &k1) in a1.iter().enumerate() {
                let r = polygon_reference(&v[..5], [k0, k1]);
                assert!((k.get(p, q) - r).norm() < 1e-9 * (1.0 + r.norm()), "{:?} {} {}", (k0, k1), k.get(p, q), r);
            }
        }
        // edge parallel to some k (k·L = 0) handled by the limit
        let sq = vec![[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]];
        let k = kspace_polygon(&sq, &[0.0, 0.1], &[0.0, 0.2]);
        let r = polygon_reference(&sq, [0.1, 0.0]);
        assert!((k.get(1, 0) - r).norm() < 1e-9, "{} {}", k.get(1, 0), r);
    }

    #[test]
    fn raster_transform_equals_polygon_transform() {
        // a rectangle of whole pixels is exactly the polygon of its outline
        let mut mask = Array2::<u8>::zeros(10, 10);
        for r in 2..8 {
            for c in 3..7 {
                mask.data[r * 10 + c] = 1;
            }
        }
        let (origin, pixel, center) = ([1.0, -2.0], [0.5, 0.5], [0.3, 0.2]);
        let a0: Vec<f64> = (-7..7).map(|i| i as f64 * 0.11).collect();
        let a1: Vec<f64> = (-6..8).map(|i| i as f64 * 0.09).collect();
        let raster = kspace_raster(&mask, origin, pixel, &[a0.clone(), a1.clone()], center);
        let (y0, y1, x0, x1) = (2.0 - center[0], 5.0 - center[0], -0.5 - center[1], 1.5 - center[1]);
        let poly = kspace_polygon(&[[y0, x0], [y0, x1], [y1, x1], [y1, x0]], &a0, &a1);
        for (a, b) in raster.data.iter().zip(poly.data.iter()) {
            assert!((a - b).norm() < 1e-10 * (1.0 + b.norm()), "{a} vs {b}");
        }
    }

    #[test]
    fn label_image_phantom_from_png() {
        let mut img = image::RgbImage::new(8, 6);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if x < 4 {
                image::Rgb([0, 255, 0])
            } else if y < 3 {
                image::Rgb([0, 255, 255])
            } else {
                image::Rgb([0, 0, 0])
            };
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let shapes = load_shapes_raster(&png, "pixel_size = 2.0\n'#00ffff' = 'CSF'\n'#00FF00' = 'Gray matter'\n").unwrap();
        let names: Vec<&str> = shapes.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(names, vec!["CSF", "Gray matter"]);
        let (sup, _) = support_and_center(shapes.iter().flat_map(|(_, s)| s.iter()));
        assert_eq!(sup, [12.0, 16.0]);
        match &shapes[0].1[0] {
            Shape::Raster { mask, origin, .. } => {
                assert_eq!((mask.rows, mask.cols), (3, 4));
                assert_eq!(*origin, [0.0, 8.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn builtin_phantoms_load() {
        let b = load("brain", 0.5).unwrap();
        assert_eq!(b.matrix, [369, 302]);
        assert_eq!(b.tissues, ["Gray matter", "CSF", "White matter", "Muscle", "Blood", "Adipose tissue", "Cortical bone", "Bone marrow"]);
        assert!((b.support[0] - 184.0177085).abs() < 1e-6 && (b.support[1] - 150.8124959).abs() < 1e-6);
        assert!((b.center[0] - 92.14114575).abs() < 1e-6 && (b.center[1] - 75.53861205).abs() < 1e-6);
        let s = load("Shepp-Logan", 0.5).unwrap();
        assert_eq!(s.matrix, [380, 288]);
        assert_eq!(s.tissues[0], "Shepp-Logan Scalp");
        assert!(load("abdomen", 0.5).is_ok());
    }
}
