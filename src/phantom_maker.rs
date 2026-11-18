//! Making and importing phantoms: rudimentary image segmentation, simple drawn shapes and
//! import of SVG / TOML / PNG phantom files into the user phantom directory.
//!
//! Self-made phantoms are stored as a PNG label image plus a TOML colormap
//! (`pixel_size = <mm>` and `'#rrggbb' = 'Tissue'` entries), see `phantom.rs`.

use std::path::{Path, PathBuf};

use crate::constants::{tissue, TISSUES};
use crate::phantom;

/// Display/label colour of each tissue (the built-in phantoms' colours where they exist)
pub fn tissue_color(name: &str) -> [u8; 3] {
    match name {
        "Gray matter" => [0x00, 0xff, 0x00],
        "White matter" => [0xd4, 0x00, 0x00],
        "CSF" => [0x00, 0xff, 0xff],
        "Adipose tissue" => [0xff, 0xe6, 0x80],
        "Bone marrow" => [0xff, 0xff, 0x44],
        "Liver" => [0x80, 0x00, 0x00],
        "Spleen" => [0xff, 0x00, 0x00],
        "Muscle" => [0x00, 0x80, 0x00],
        "Kidney medulla" => [0xaa, 0x44, 0x00],
        "Kidney cortex" => [0x55, 0x22, 0x00],
        "Spinal cord" => [0xff, 0xff, 0x00],
        "Cortical bone" => [0x80, 0x80, 0x00],
        "Blood" => [0xff, 0xee, 0xee],
        "Stomach" => [0x1a, 0x20, 0x20],
        "Perotineum" => [0xff, 0x80, 0x80],
        "Shepp-Logan Scalp" => [0x99, 0x66, 0xcc],
        "Shepp-Logan Bone and Marrow" => [0xcc, 0xcc, 0x99],
        "Shepp-Logan CSF" => [0x33, 0x66, 0xff],
        "Shepp-Logan Gray matter" => [0x66, 0xcc, 0x66],
        "Shepp-Logan White matter" => [0xcc, 0x66, 0x66],
        "Shepp-Logan Tumor" => [0xff, 0x33, 0x99],
        _ => [0x80, 0x80, 0xff],
    }
}

pub fn tissue_names() -> Vec<&'static str> {
    TISSUES.iter().map(|t| t.name).collect()
}

// ---------------------------------------------------------------------------------------------
// segmentation

/// Grayscale image with values in [0, 1]
#[derive(Debug, Clone, PartialEq)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Maximum image size used for segmentation (larger images are scaled down)
pub const MAX_SEGMENTATION_SIZE: u32 = 512;

pub fn load_gray(path: &Path) -> Result<GrayImage, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(gray_from_image(img))
}

pub fn gray_from_image(img: image::DynamicImage) -> GrayImage {
    let img = if img.width().max(img.height()) > MAX_SEGMENTATION_SIZE {
        img.resize(MAX_SEGMENTATION_SIZE, MAX_SEGMENTATION_SIZE, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let g = img.into_luma16();
    let (w, h) = (g.width() as usize, g.height() as usize);
    GrayImage { w, h, data: g.pixels().map(|p| p[0] as f32 / 65535.0).collect() }
}

/// k-means clustering of the intensities into `k` classes; labels sorted by intensity (0 = darkest)
pub fn kmeans(gray: &[f32], k: usize) -> (Vec<u8>, Vec<f32>) {
    let k = k.clamp(1, 16);
    if gray.is_empty() {
        return (vec![], vec![0.0; k]);
    }
    let mut sorted = gray.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut centers: Vec<f32> = (0..k).map(|i| sorted[((i as f32 + 0.5) / k as f32 * (sorted.len() - 1) as f32) as usize]).collect();
    // 1D k-means on a 256-bin histogram (fast and exact enough for display images)
    let mut hist = [0u64; 256];
    for &v in gray {
        hist[(v.clamp(0.0, 1.0) * 255.0).round() as usize] += 1;
    }
    for _ in 0..50 {
        let mut sum = vec![0f64; k];
        let mut cnt = vec![0u64; k];
        for (b, &n) in hist.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let v = b as f32 / 255.0;
            let c = nearest(&centers, v);
            sum[c] += v as f64 * n as f64;
            cnt[c] += n;
        }
        let mut changed = false;
        for c in 0..k {
            if cnt[c] > 0 {
                let nc = (sum[c] / cnt[c] as f64) as f32;
                if (nc - centers[c]).abs() > 1e-5 {
                    changed = true;
                }
                centers[c] = nc;
            }
        }
        if !changed {
            break;
        }
    }
    centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let labels = gray.iter().map(|&v| nearest(&centers, v) as u8).collect();
    (labels, centers)
}

fn nearest(centers: &[f32], v: f32) -> usize {
    let mut best = 0;
    for (i, c) in centers.iter().enumerate() {
        if (v - c).abs() < (v - centers[best]).abs() {
            best = i;
        }
    }
    best
}

/// 3×3 majority filter (removes isolated pixels / speckle)
pub fn majority_filter(labels: &[u8], w: usize, h: usize, k: usize, iterations: usize) -> Vec<u8> {
    let mut cur = labels.to_vec();
    for _ in 0..iterations {
        let mut next = cur.clone();
        let mut counts = vec![0u8; k.max(1)];
        for r in 0..h {
            for c in 0..w {
                counts.iter_mut().for_each(|x| *x = 0);
                for dr in -1i64..=1 {
                    for dc in -1i64..=1 {
                        let (rr, cc) = (r as i64 + dr, c as i64 + dc);
                        if rr >= 0 && cc >= 0 && (rr as usize) < h && (cc as usize) < w {
                            let l = cur[rr as usize * w + cc as usize] as usize;
                            if l < counts.len() {
                                counts[l] += 1;
                            }
                        }
                    }
                }
                let (best, &n) = counts.iter().enumerate().max_by_key(|(_, n)| **n).unwrap();
                if n >= 5 {
                    next[r * w + c] = best as u8;
                }
            }
        }
        cur = next;
    }
    cur
}

/// Default tissue for each intensity class (darkest = background), e.g. T1-like brain contrast
pub fn default_class_tissues(k: usize) -> Vec<Option<String>> {
    let order = ["CSF", "Gray matter", "White matter", "Adipose tissue", "Muscle", "Blood", "Cortical bone", "Bone marrow", "Liver", "Spleen"];
    (0..k).map(|i| if i == 0 { None } else { order.get(i - 1).map(|s| s.to_string()) }).collect()
}

// ---------------------------------------------------------------------------------------------
// drawing

/// Shape geometry in canvas coordinates (x right, y down) [mm]
#[derive(Debug, Clone, PartialEq)]
pub enum Geom {
    Ellipse { center: [f64; 2], radius: [f64; 2] },
    Rect { a: [f64; 2], b: [f64; 2] },
    Polygon { points: Vec<[f64; 2]> },
}

impl Geom {
    pub fn contains(&self, p: [f64; 2]) -> bool {
        match self {
            Geom::Ellipse { center, radius } => {
                if radius[0] <= 0.0 || radius[1] <= 0.0 {
                    return false;
                }
                let dx = (p[0] - center[0]) / radius[0];
                let dy = (p[1] - center[1]) / radius[1];
                dx * dx + dy * dy <= 1.0
            }
            Geom::Rect { a, b } => p[0] >= a[0].min(b[0]) && p[0] <= a[0].max(b[0]) && p[1] >= a[1].min(b[1]) && p[1] <= a[1].max(b[1]),
            Geom::Polygon { points } => {
                let n = points.len();
                if n < 3 {
                    return false;
                }
                let mut inside = false;
                let mut j = n - 1;
                for i in 0..n {
                    let (xi, yi) = (points[i][0], points[i][1]);
                    let (xj, yj) = (points[j][0], points[j][1]);
                    if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
                        inside = !inside;
                    }
                    j = i;
                }
                inside
            }
        }
    }

    /// Outline for display
    pub fn outline(&self) -> Vec<[f64; 2]> {
        match self {
            Geom::Ellipse { center, radius } => (0..72)
                .map(|i| {
                    let a = i as f64 / 72.0 * std::f64::consts::TAU;
                    [center[0] + radius[0] * a.cos(), center[1] + radius[1] * a.sin()]
                })
                .collect(),
            Geom::Rect { a, b } => vec![[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]],
            Geom::Polygon { points } => points.clone(),
        }
    }

    pub fn translate(&mut self, d: [f64; 2]) {
        match self {
            Geom::Ellipse { center, .. } => {
                center[0] += d[0];
                center[1] += d[1];
            }
            Geom::Rect { a, b } => {
                for p in [a, b] {
                    p[0] += d[0];
                    p[1] += d[1];
                }
            }
            Geom::Polygon { points } => {
                for p in points {
                    p[0] += d[0];
                    p[1] += d[1];
                }
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Geom::Ellipse { .. } => "Ellipse",
            Geom::Rect { .. } => "Rectangle",
            Geom::Polygon { .. } => "Polygon",
        }
    }
}

/// A drawn shape; `tissue = None` erases (paints background)
#[derive(Debug, Clone, PartialEq)]
pub struct DrawShape {
    pub geom: Geom,
    pub tissue: Option<String>,
}

/// Segmented image used as the bottom layer
#[derive(Debug, Clone)]
pub struct BaseLayer {
    pub image: GrayImage,
    pub classes: usize,
    pub smoothing: usize,
    pub labels: Vec<u8>,
    pub centers: Vec<f32>,
    pub class_tissue: Vec<Option<String>>,
    pub name: String,
}

impl BaseLayer {
    pub fn new(image: GrayImage, name: String, classes: usize) -> BaseLayer {
        let mut b = BaseLayer { image, classes, smoothing: 1, labels: vec![], centers: vec![], class_tissue: default_class_tissues(classes), name };
        b.segment();
        b
    }

    /// (Re)run the segmentation after changing the number of classes or the smoothing
    pub fn segment(&mut self) {
        let (labels, centers) = kmeans(&self.image.data, self.classes);
        self.labels = majority_filter(&labels, self.image.w, self.image.h, self.classes, self.smoothing);
        self.centers = centers;
        if self.class_tissue.len() != self.classes {
            self.class_tissue = default_class_tissues(self.classes);
        }
    }
}

/// The phantom maker document: an optional segmented image below, drawn shapes on top
#[derive(Debug, Clone)]
pub struct MakerDoc {
    /// canvas size [mm]
    pub width: f64,
    pub height: f64,
    /// pixel size of the resulting label image [mm]
    pub pixel: f64,
    pub base: Option<BaseLayer>,
    pub shapes: Vec<DrawShape>,
}

impl Default for MakerDoc {
    fn default() -> Self {
        MakerDoc { width: 200.0, height: 240.0, pixel: 0.5, base: None, shapes: Vec::new() }
    }
}

/// Label image: tissue name per pixel (`None` = background)
pub struct LabelMap {
    pub w: usize,
    pub h: usize,
    pub tissues: Vec<String>,
    /// index into `tissues`, `u8::MAX` = background
    pub labels: Vec<u8>,
}

impl MakerDoc {
    /// Set an image as bottom layer; the canvas takes the image's aspect ratio
    pub fn set_image(&mut self, image: GrayImage, name: String, classes: usize) {
        self.height = self.width * image.h as f64 / image.w.max(1) as f64;
        self.base = Some(BaseLayer::new(image, name, classes));
    }

    pub fn grid_size(&self) -> (usize, usize) {
        (((self.width / self.pixel).ceil() as usize).max(1), ((self.height / self.pixel).ceil() as usize).max(1))
    }

    /// Rasterize the document (base layer, then shapes in drawing order)
    pub fn label_map(&self) -> LabelMap {
        let (w, h) = self.grid_size();
        let mut tissues: Vec<String> = Vec::new();
        let index = |t: &str, tissues: &mut Vec<String>| -> u8 {
            match tissues.iter().position(|x| x == t) {
                Some(i) => i as u8,
                None => {
                    tissues.push(t.to_string());
                    (tissues.len() - 1) as u8
                }
            }
        };
        let mut labels = vec![u8::MAX; w * h];
        if let Some(b) = &self.base {
            let class_idx: Vec<u8> = b.class_tissue.iter().map(|t| t.as_ref().map_or(u8::MAX, |t| index(t, &mut tissues))).collect();
            for r in 0..h {
                let iy = (((r as f64 + 0.5) / h as f64) * b.image.h as f64) as usize;
                for c in 0..w {
                    let ix = (((c as f64 + 0.5) / w as f64) * b.image.w as f64) as usize;
                    let l = b.labels[iy.min(b.image.h - 1) * b.image.w + ix.min(b.image.w - 1)] as usize;
                    labels[r * w + c] = class_idx.get(l).copied().unwrap_or(u8::MAX);
                }
            }
        }
        for s in &self.shapes {
            let li = s.tissue.as_ref().map_or(u8::MAX, |t| index(t, &mut tissues));
            for r in 0..h {
                let y = (r as f64 + 0.5) * self.pixel;
                for c in 0..w {
                    let x = (c as f64 + 0.5) * self.pixel;
                    if s.geom.contains([x, y]) {
                        labels[r * w + c] = li;
                    }
                }
            }
        }
        // drop tissues that ended up completely covered
        let used: Vec<bool> = (0..tissues.len()).map(|i| labels.contains(&(i as u8))).collect();
        let mut remap = vec![u8::MAX; tissues.len()];
        let mut kept = Vec::new();
        for (i, t) in tissues.iter().enumerate() {
            if used[i] {
                remap[i] = kept.len() as u8;
                kept.push(t.clone());
            }
        }
        for l in labels.iter_mut() {
            if *l != u8::MAX {
                *l = remap[*l as usize];
            }
        }
        LabelMap { w, h, tissues: kept, labels }
    }

    /// Save as a phantom named `name` in the user phantom directory; returns the phantom name
    pub fn save(&self, name: &str) -> Result<String, String> {
        let map = self.label_map();
        if map.tissues.is_empty() {
            return Err("The phantom is empty: assign tissues to the image classes or draw shapes".into());
        }
        save_label_map(name, &map, self.pixel)
    }
}

/// Write a label map as `<name>.png` + `<name>.toml` into the user phantom directory
pub fn save_label_map(name: &str, map: &LabelMap, pixel: f64) -> Result<String, String> {
    let name = sanitize_name(name)?;
    let mut img = image::RgbImage::new(map.w as u32, map.h as u32);
    for r in 0..map.h {
        for c in 0..map.w {
            let l = map.labels[r * map.w + c];
            let rgb = if l == u8::MAX { [0, 0, 0] } else { tissue_color(&map.tissues[l as usize]) };
            img.put_pixel(c as u32, r as u32, image::Rgb(rgb));
        }
    }
    let mut toml = format!("# SpinSight label-image phantom: pixel size [mm] and colour → tissue\npixel_size = {pixel}\n");
    for t in &map.tissues {
        let [r, g, b] = tissue_color(t);
        toml.push_str(&format!("'{}' = '{}'\n", phantom::hexcolor(r, g, b), t));
    }
    let dir = phantom_dir(&name)?;
    clear_phantom_files(&dir, &name);
    img.save(dir.join(format!("{name}.png"))).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.toml")), toml).map_err(|e| e.to_string())?;
    verify(&name, &dir)
}

// ---------------------------------------------------------------------------------------------
// import

/// Result of importing a file
#[derive(Debug, Clone)]
pub enum Import {
    /// imported as phantom with this name
    Done(String),
    /// an SVG without (complete) colormap: the fill colours need tissues
    NeedsColormap { name: String, svg: String, colors: Vec<(String, Option<String>)> },
    /// a regular image: open in the phantom maker for segmentation
    Image(PathBuf),
}

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif", "webp"];

/// Allowed phantom name (letters, digits, space, - and _), not a built-in phantom
pub fn sanitize_name(name: &str) -> Result<String, String> {
    let n: String = name.trim().chars().filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == ' ').collect();
    let n = n.trim().to_string();
    if n.is_empty() {
        return Err("Please enter a name for the phantom".into());
    }
    if phantom::is_builtin(&n) {
        return Err(format!("\"{n}\" is the name of a built-in phantom"));
    }
    Ok(n)
}

/// A free phantom name based on `base` (does not clash with existing phantoms)
pub fn unique_name(base: &str) -> String {
    let base = sanitize_name(base).unwrap_or_else(|_| "phantom".into());
    let names = phantom::phantom_names();
    if !names.contains(&base) {
        return base;
    }
    (2..).map(|i| format!("{base}-{i}")).find(|n| !names.contains(n)).unwrap()
}

fn phantom_dir(name: &str) -> Result<PathBuf, String> {
    let dir = phantom::user_phantom_write_dir().ok_or("no user data directory available")?.join(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

fn clear_phantom_files(dir: &Path, name: &str) {
    for ext in ["svg", "png", "toml"] {
        let _ = std::fs::remove_file(dir.join(format!("{name}.{ext}")));
    }
}

/// Check that the written phantom loads; remove it otherwise
fn verify(name: &str, dir: &Path) -> Result<String, String> {
    match phantom::load(name, 0.5) {
        Ok(_) => Ok(name.to_string()),
        Err(e) => {
            let _ = std::fs::remove_dir_all(dir);
            Err(format!("The phantom could not be loaded: {e}"))
        }
    }
}

/// Distinct fill colours of the paths of an SVG document (lower-case hex), in order of appearance
pub fn svg_fill_colors(svg: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for p in crate::svg::read_paths(svg)? {
        if let Some(f) = p.fill {
            let f = f.to_lowercase();
            if !out.contains(&f) {
                out.push(f);
            }
        }
    }
    Ok(out)
}

/// Save an SVG phantom with the given colour → tissue mapping
pub fn save_svg_phantom(name: &str, svg: &str, mapping: &[(String, Option<String>)]) -> Result<String, String> {
    let name = sanitize_name(name)?;
    let mut toml = String::from("# SpinSight phantom colormap: SVG fill colour → tissue\n");
    let mut any = false;
    for (c, t) in mapping {
        if let Some(t) = t {
            toml.push_str(&format!("'{c}' = '{t}'\n"));
            any = true;
        }
    }
    if !any {
        return Err("Assign a tissue to at least one colour".into());
    }
    let dir = phantom_dir(&name)?;
    clear_phantom_files(&dir, &name);
    std::fs::write(dir.join(format!("{name}.svg")), svg).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.toml")), toml).map_err(|e| e.to_string())?;
    verify(&name, &dir)
}

/// Import a phantom file (SVG with colormap, ellipse TOML, PNG label image) or recognise an
/// image for segmentation
pub fn import_file(path: &Path) -> Result<Import, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("phantom").to_string();
    let sibling = |e: &str| path.with_extension(e);
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    match ext.as_str() {
        "svg" => {
            let svg = read(path)?;
            let colors = svg_fill_colors(&svg)?;
            if colors.is_empty() {
                return Err("The SVG file contains no filled paths".into());
            }
            let existing = if sibling("toml").is_file() { phantom::parse_colormap(&read(&sibling("toml"))?).unwrap_or_default() } else { vec![] };
            let mapping: Vec<(String, Option<String>)> =
                colors.iter().map(|c| (c.clone(), existing.iter().find(|(k, _)| k == c).map(|(_, t)| t.clone()).filter(|t| tissue(t).is_some()))).collect();
            let name = unique_name(&stem);
            if mapping.iter().all(|(_, t)| t.is_some()) {
                return save_svg_phantom(&name, &svg, &mapping).map(Import::Done);
            }
            Ok(Import::NeedsColormap { name, svg, colors: mapping })
        }
        "toml" => {
            let text = read(path)?;
            for e in ["svg", "png"] {
                if sibling(e).is_file() {
                    return import_file(&sibling(e));
                }
            }
            let name = unique_name(&stem);
            let dir = phantom_dir(&name)?;
            clear_phantom_files(&dir, &name);
            std::fs::write(dir.join(format!("{name}.toml")), text).map_err(|e| e.to_string())?;
            verify(&name, &dir).map(Import::Done)
        }
        e if IMAGE_EXTENSIONS.contains(&e) => {
            // a label image with its colormap next to it is a phantom, anything else is segmented
            if e == "png" && sibling("toml").is_file() {
                let cmap = read(&sibling("toml"))?;
                if phantom::parse_colormap(&cmap).is_ok_and(|m| !m.is_empty()) {
                    let name = unique_name(&stem);
                    let dir = phantom_dir(&name)?;
                    clear_phantom_files(&dir, &name);
                    std::fs::copy(path, dir.join(format!("{name}.png"))).map_err(|e| e.to_string())?;
                    std::fs::write(dir.join(format!("{name}.toml")), cmap).map_err(|e| e.to_string())?;
                    return verify(&name, &dir).map(Import::Done);
                }
            }
            Ok(Import::Image(path.to_path_buf()))
        }
        _ => Err(format!("Unsupported file type .{ext} (use .svg, .toml, .png, .jpg, …)")),
    }
}

/// Delete a user phantom (built-in phantoms cannot be deleted)
pub fn delete_user_phantom(name: &str) -> Result<(), String> {
    if phantom::is_builtin(name) {
        return Err("Built-in phantoms cannot be deleted".into());
    }
    let dir = phantom::user_phantom_write_dir().ok_or("no user data directory")?.join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))
    } else {
        Err(format!("{name} is not in {}", dir.parent().map(|p| p.display().to_string()).unwrap_or_default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_separates_levels() {
        let mut g = vec![0.1f32; 100];
        g.extend(vec![0.5f32; 100]);
        g.extend(vec![0.9f32; 100]);
        let (labels, centers) = kmeans(&g, 3);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[150], 1);
        assert_eq!(labels[299], 2);
        assert!((centers[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn shapes_paint_in_order() {
        let mut doc = MakerDoc { width: 10.0, height: 10.0, pixel: 1.0, base: None, shapes: vec![] };
        doc.shapes.push(DrawShape { geom: Geom::Rect { a: [0.0, 0.0], b: [10.0, 10.0] }, tissue: Some("Muscle".into()) });
        doc.shapes.push(DrawShape { geom: Geom::Ellipse { center: [5.0, 5.0], radius: [2.0, 2.0] }, tissue: Some("CSF".into()) });
        doc.shapes.push(DrawShape { geom: Geom::Polygon { points: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]] }, tissue: None });
        let m = doc.label_map();
        assert_eq!(m.tissues, vec!["Muscle", "CSF"]);
        assert_eq!(m.labels[5 * 10 + 5], 1); // ellipse centre: CSF
        assert_eq!(m.labels[9 * 10 + 9], 0); // corner: muscle
        assert_eq!(m.labels[0], u8::MAX); // erased
    }
}
