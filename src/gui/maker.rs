//! Phantom maker window (segment an image and/or draw simple shapes) and the dialog that maps
//! SVG fill colours to tissues when importing an SVG phantom without colormap.

use std::path::Path;

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2};

use crate::phantom_maker::{self, DrawShape, Geom, MakerDoc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Ellipse,
    Rect,
    Polygon,
}

pub struct MakerWindow {
    pub open: bool,
    doc: MakerDoc,
    tool: Tool,
    /// tissue of new shapes (`None` = eraser)
    tissue: Option<String>,
    drag: Option<([f64; 2], [f64; 2])>,
    moving: Option<(usize, [f64; 2])>,
    poly: Vec<[f64; 2]>,
    selected: Option<usize>,
    name: String,
    tex: Option<TextureHandle>,
    dirty: bool,
    show_image: bool,
    opacity: f32,
    message: Option<String>,
}

impl Default for MakerWindow {
    fn default() -> Self {
        MakerWindow {
            open: false,
            doc: MakerDoc::default(),
            tool: Tool::Ellipse,
            tissue: Some("Gray matter".into()),
            drag: None,
            moving: None,
            poly: Vec::new(),
            selected: None,
            name: phantom_maker::unique_name("my phantom"),
            tex: None,
            dirty: true,
            show_image: true,
            opacity: 0.55,
            message: None,
        }
    }
}

pub fn color32(rgb: [u8; 3]) -> Color32 {
    Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn swatch(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(14.0, 14.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.painter().rect_stroke(rect, 2.0, Stroke::new(1.0, Color32::GRAY), egui::StrokeKind::Inside);
}

/// Tissue selector; `none_label` is the entry for `None` (background / eraser)
fn tissue_combo(ui: &mut Ui, id: impl std::hash::Hash + std::fmt::Debug, value: &mut Option<String>, none_label: &str, width: f32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        swatch(ui, value.as_ref().map_or(Color32::from_gray(25), |t| color32(phantom_maker::tissue_color(t))));
        egui::ComboBox::from_id_salt(id).selected_text(value.clone().unwrap_or_else(|| none_label.to_string())).width(width).show_ui(ui, |ui| {
            if ui.selectable_label(value.is_none(), none_label).clicked() {
                *value = None;
                changed = true;
            }
            for t in phantom_maker::tissue_names() {
                let sel = value.as_deref() == Some(t);
                let r = ui.horizontal(|ui| {
                    swatch(ui, color32(phantom_maker::tissue_color(t)));
                    ui.selectable_label(sel, t)
                });
                if r.inner.clicked() {
                    *value = Some(t.to_string());
                    changed = true;
                }
            }
        });
    });
    changed
}

impl MakerWindow {
    pub fn load_image(&mut self, path: &Path) {
        match phantom_maker::load_gray(path) {
            Ok(img) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image").to_string();
                let classes = self.doc.base.as_ref().map_or(4, |b| b.classes);
                self.doc.set_image(img, stem.clone(), classes);
                self.name = phantom_maker::unique_name(&stem);
                self.message = Some("Image segmented: assign a tissue to each intensity class (darkest = background).".into());
                self.dirty = true;
            }
            Err(e) => self.message = Some(e),
        }
    }

    /// Show the window; returns (phantom name, use it now) after saving
    pub fn show(&mut self, ctx: &egui::Context) -> Option<(String, bool)> {
        let mut open = self.open;
        let mut saved = None;
        egui::Window::new("Phantom maker").open(&mut open).default_size([1150.0, 760.0]).min_size([800.0, 500.0]).resizable(true).show(ctx, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(330.0);
                    egui::ScrollArea::vertical().id_salt("maker_controls").show(ui, |ui| {
                        saved = self.controls(ui);
                    });
                });
                ui.separator();
                ui.vertical(|ui| self.canvas(ui));
            });
        });
        self.open = open;
        saved
    }

    fn controls(&mut self, ui: &mut Ui) -> Option<(String, bool)> {
        let heading = |ui: &mut Ui, t: &str| {
            ui.add_space(4.0);
            ui.label(RichText::new(t).strong().size(15.0));
        };
        heading(ui, "1  Image (optional)");
        ui.label(RichText::new("An image (e.g. an MR image or a drawing) is segmented into intensity classes.").small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            if ui.button("Load image…").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif", "webp"]).pick_file() {
                    self.load_image(&p);
                }
            }
            if self.doc.base.is_some() && ui.button("Remove image").clicked() {
                self.doc.base = None;
                self.dirty = true;
            }
        });
        let mut resegment = false;
        if let Some(b) = &mut self.doc.base {
            ui.label(format!("{} ({}×{} px)", b.name, b.image.w, b.image.h));
            let mut classes = b.classes;
            if ui.add(egui::Slider::new(&mut classes, 2..=8).text("intensity classes")).changed() {
                b.classes = classes;
                resegment = true;
            }
            let mut smooth = b.smoothing;
            if ui.add(egui::Slider::new(&mut smooth, 0..=4).text("smoothing")).changed() {
                b.smoothing = smooth;
                resegment = true;
            }
        }
        if resegment {
            if let Some(b) = &mut self.doc.base {
                b.segment();
            }
            self.dirty = true;
        }
        if self.doc.base.is_some() {
            let mut width = self.doc.width;
            if ui.add(egui::Slider::new(&mut width, 40.0..=500.0).text("image width [mm]")).changed() {
                let aspect = self.doc.height / self.doc.width;
                self.doc.width = width;
                self.doc.height = width * aspect;
                self.dirty = true;
            }
            ui.label(RichText::new("Tissue of each class (dark → bright):").color(Color32::GRAY));
            let mut changed = false;
            if let Some(b) = &mut self.doc.base {
                for i in 0..b.classes {
                    ui.horizontal(|ui| {
                        let g = (b.centers.get(i).copied().unwrap_or(0.0) * 255.0) as u8;
                        swatch(ui, Color32::from_gray(g));
                        ui.label(format!("class {}", i + 1));
                        changed |= tissue_combo(ui, ("class", i), &mut b.class_tissue[i], "— background —", 170.0);
                    });
                }
            }
            if changed {
                self.dirty = true;
            }
            if ui.checkbox(&mut self.show_image, "show image underneath").changed() {
                self.dirty = true;
            }
            if self.show_image && ui.add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).text("label opacity")).changed() {
                self.dirty = true;
            }
        } else {
            ui.horizontal(|ui| {
                ui.label("Canvas");
                let w = ui.add(egui::DragValue::new(&mut self.doc.width).range(40.0..=500.0).suffix(" mm"));
                ui.label("×");
                let h = ui.add(egui::DragValue::new(&mut self.doc.height).range(40.0..=500.0).suffix(" mm"));
                if w.changed() || h.changed() {
                    self.dirty = true;
                }
            });
        }

        ui.add_space(6.0);
        ui.separator();
        heading(ui, "2  Draw shapes");
        ui.horizontal(|ui| {
            for (t, label) in [(Tool::Select, "Select/move"), (Tool::Ellipse, "Ellipse"), (Tool::Rect, "Rectangle"), (Tool::Polygon, "Polygon")] {
                // (Enter re-activates the focused button: only a real tool change resets the polygon)
                let r = ui.selectable_label(self.tool == t, label);
                if r.clicked() {
                    // keep the keyboard free for the canvas (Enter would re-click a focused button)
                    ui.memory_mut(|m| m.surrender_focus(r.id));
                    if self.tool != t {
                        self.tool = t;
                        self.poly.clear();
                    }
                }
            }
        });
        let hint = match self.tool {
            Tool::Select => "Click a shape to select it, drag to move it, Delete to remove it.",
            Tool::Ellipse | Tool::Rect => "Drag on the canvas to draw.",
            Tool::Polygon => "Click to add corners; double-click, Enter or click the first corner to close; Esc cancels.",
        };
        ui.label(RichText::new(hint).small().color(Color32::GRAY));
        ui.horizontal(|ui| {
            ui.label("Tissue");
            tissue_combo(ui, "new_tissue", &mut self.tissue, "Eraser (background)", 190.0);
        });
        ui.label(RichText::new("Later shapes are painted on top of earlier ones.").small().color(Color32::GRAY));
        let mut remove = None;
        let mut swap = None;
        let n = self.doc.shapes.len();
        egui::ScrollArea::vertical().id_salt("shape_list").max_height(220.0).show(ui, |ui| {
            for i in 0..n {
                ui.horizontal(|ui| {
                    let sel = self.selected == Some(i);
                    if ui.selectable_label(sel, format!("{} {}", i + 1, self.doc.shapes[i].geom.kind())).clicked() {
                        self.selected = Some(i);
                        self.tool = Tool::Select;
                    }
                    if tissue_combo(ui, ("shape", i), &mut self.doc.shapes[i].tissue, "Eraser", 120.0) {
                        self.dirty = true;
                    }
                    if ui.small_button("↑").on_hover_text("towards the front").clicked() && i + 1 < n {
                        swap = Some((i, i + 1));
                    }
                    if ui.small_button("↓").on_hover_text("towards the back").clicked() && i > 0 {
                        swap = Some((i, i - 1));
                    }
                    if ui.small_button("✕").clicked() {
                        remove = Some(i);
                    }
                });
            }
        });
        if let Some((a, b)) = swap {
            self.doc.shapes.swap(a, b);
            self.selected = Some(b);
            self.dirty = true;
        }
        if let Some(i) = remove {
            self.doc.shapes.remove(i);
            self.selected = None;
            self.dirty = true;
        }
        if n > 0 && ui.button("Remove all shapes").clicked() {
            self.doc.shapes.clear();
            self.selected = None;
            self.dirty = true;
        }

        ui.add_space(6.0);
        ui.separator();
        heading(ui, "3  Save");
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(200.0));
        });
        let (w, h) = self.doc.grid_size();
        ui.label(
            RichText::new(format!("{:.0} × {:.0} mm, {} × {} pixels of {} mm", self.doc.width, self.doc.height, w, h, self.doc.pixel))
                .small()
                .color(Color32::GRAY),
        );
        let mut result = None;
        ui.horizontal(|ui| {
            let mut save = |use_it: bool, this: &mut Self| match this.doc.save(&this.name) {
                Ok(name) => {
                    this.message = Some(format!("Saved \"{name}\" in the phantom folder."));
                    result = Some((name, use_it));
                }
                Err(e) => this.message = Some(e),
            };
            if ui.button("Save").clicked() {
                save(false, self);
            }
            if ui.button(RichText::new("Save and use").strong()).clicked() {
                save(true, self);
            }
        });
        if let Some(m) = &self.message {
            ui.label(RichText::new(m).color(Color32::from_rgb(230, 180, 80)));
        }
        result
    }

    fn rebuild_texture(&mut self, ctx: &egui::Context) {
        let map = self.doc.label_map();
        let colors: Vec<[u8; 3]> = map.tissues.iter().map(|t| phantom_maker::tissue_color(t)).collect();
        let base = self.doc.base.as_ref().filter(|_| self.show_image);
        let mut rgba = Vec::with_capacity(map.w * map.h * 4);
        for r in 0..map.h {
            for c in 0..map.w {
                let gray = base.map(|b| {
                    let iy = (((r as f64 + 0.5) / map.h as f64) * b.image.h as f64) as usize;
                    let ix = (((c as f64 + 0.5) / map.w as f64) * b.image.w as f64) as usize;
                    b.image.data[iy.min(b.image.h - 1) * b.image.w + ix.min(b.image.w - 1)]
                });
                let l = map.labels[r * map.w + c];
                let px = match (l, gray) {
                    (u8::MAX, Some(g)) => {
                        let v = (g * 255.0 * 0.7) as u8;
                        [v, v, v]
                    }
                    (u8::MAX, None) => [18, 18, 22],
                    (l, Some(g)) => {
                        let col = colors[l as usize];
                        let a = self.opacity;
                        let v = g * 255.0;
                        [0, 1, 2].map(|i| (col[i] as f32 * a + v * (1.0 - a)) as u8)
                    }
                    (l, None) => colors[l as usize],
                };
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        let img = egui::ColorImage::from_rgba_unmultiplied([map.w, map.h], &rgba);
        self.tex = Some(ctx.load_texture("maker_canvas", img, TextureOptions::NEAREST));
        self.dirty = false;
    }

    fn canvas(&mut self, ui: &mut Ui) {
        if self.dirty || self.tex.is_none() {
            self.rebuild_texture(ui.ctx());
        }
        let avail = ui.available_size().max(Vec2::new(300.0, 300.0));
        let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
        let scale = (rect.width() / self.doc.width as f32).min(rect.height() / self.doc.height as f32);
        let size = Vec2::new(self.doc.width as f32 * scale, self.doc.height as f32 * scale);
        let canvas = Rect::from_center_size(rect.center(), size);
        let to_mm = |p: Pos2| -> [f64; 2] { [((p.x - canvas.min.x) / scale) as f64, ((p.y - canvas.min.y) / scale) as f64] };
        let to_screen = |p: [f64; 2]| -> Pos2 { Pos2::new(canvas.min.x + p[0] as f32 * scale, canvas.min.y + p[1] as f32 * scale) };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(10));
        if let Some(t) = &self.tex {
            painter.image(t.id(), canvas, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        painter.rect_stroke(canvas, 0.0, Stroke::new(1.0, Color32::GRAY), egui::StrokeKind::Outside);
        // scale bar (10 mm)
        let bar = [canvas.left() + 10.0, canvas.bottom() - 12.0];
        painter.line_segment([Pos2::new(bar[0], bar[1]), Pos2::new(bar[0] + 10.0 * scale, bar[1])], Stroke::new(3.0, Color32::WHITE));
        painter.text(Pos2::new(bar[0], bar[1] - 4.0), egui::Align2::LEFT_BOTTOM, "10 mm", egui::FontId::proportional(11.0), Color32::WHITE);

        // shape outlines
        for (i, s) in self.doc.shapes.iter().enumerate() {
            let pts: Vec<Pos2> = s.geom.outline().into_iter().map(to_screen).collect();
            let sel = self.selected == Some(i);
            let stroke = if sel { Stroke::new(2.5, Color32::YELLOW) } else { Stroke::new(1.0, Color32::from_white_alpha(160)) };
            painter.add(egui::Shape::closed_line(pts, stroke));
        }

        let pointer = resp.hover_pos().map(to_mm);
        let (dw, dh) = (self.doc.width, self.doc.height);
        let clamp = move |p: [f64; 2]| [p[0].clamp(0.0, dw), p[1].clamp(0.0, dh)];
        let new_shape = |geom: Geom, tissue: &Option<String>| DrawShape { geom, tissue: tissue.clone() };
        match self.tool {
            Tool::Ellipse | Tool::Rect => {
                if resp.drag_started() {
                    if let Some(p) = pointer {
                        self.drag = Some((clamp(p), clamp(p)));
                    }
                }
                if let (Some(d), Some(p)) = (&mut self.drag, pointer) {
                    d.1 = clamp(p);
                }
                if let Some((a, b)) = self.drag {
                    let geom = drag_geom(self.tool, a, b);
                    let pts: Vec<Pos2> = geom.outline().into_iter().map(to_screen).collect();
                    painter.add(egui::Shape::closed_line(pts, Stroke::new(2.0, Color32::WHITE)));
                    if resp.drag_stopped() {
                        if (a[0] - b[0]).abs() > 0.5 && (a[1] - b[1]).abs() > 0.5 {
                            self.doc.shapes.push(new_shape(geom, &self.tissue));
                            self.selected = Some(self.doc.shapes.len() - 1);
                            self.dirty = true;
                        }
                        self.drag = None;
                    }
                }
            }
            Tool::Polygon => {
                let close = |this: &mut Self| {
                    if this.poly.len() >= 3 {
                        let points = std::mem::take(&mut this.poly);
                        this.doc.shapes.push(new_shape(Geom::Polygon { points }, &this.tissue));
                        this.selected = Some(this.doc.shapes.len() - 1);
                        this.dirty = true;
                    }
                    this.poly.clear();
                };
                if resp.clicked() || resp.double_clicked() {
                    if let Some(p) = pointer.map(clamp) {
                        // a double click closes only when it is on the last corner (fast clicks on
                        // different corners must not count as a double click)
                        let near_first = self.poly.first().is_some_and(|f| to_screen(*f).distance(to_screen(p)) < 8.0);
                        let near_last = self.poly.last().is_some_and(|l| to_screen(*l).distance(to_screen(p)) < 6.0);
                        if self.poly.len() >= 3 && (near_first || (near_last && resp.double_clicked())) {
                            close(self);
                        } else if !near_last {
                            self.poly.push(p);
                        }
                    }
                }
                if resp.secondary_clicked() || key(ui, egui::Key::Escape) {
                    self.poly.clear();
                }
                if key(ui, egui::Key::Enter) {
                    close(self);
                }
                if !self.poly.is_empty() {
                    let mut pts: Vec<Pos2> = self.poly.iter().map(|p| to_screen(*p)).collect();
                    if let Some(p) = pointer {
                        pts.push(to_screen(clamp(p)));
                    }
                    painter.add(egui::Shape::line(pts, Stroke::new(2.0, Color32::WHITE)));
                    for p in &self.poly {
                        painter.circle_filled(to_screen(*p), 3.0, Color32::WHITE);
                    }
                }
            }
            Tool::Select => {
                let hit = |doc: &MakerDoc, p: [f64; 2]| (0..doc.shapes.len()).rev().find(|&i| doc.shapes[i].geom.contains(p));
                if resp.clicked() {
                    self.selected = pointer.and_then(|p| hit(&self.doc, p));
                }
                if resp.drag_started() {
                    if let Some(p) = pointer {
                        if let Some(i) = hit(&self.doc, p) {
                            self.selected = Some(i);
                            self.moving = Some((i, p));
                        }
                    }
                }
                if let (Some((i, last)), Some(p)) = (self.moving, pointer) {
                    let d = [p[0] - last[0], p[1] - last[1]];
                    if d != [0.0, 0.0] {
                        self.doc.shapes[i].geom.translate(d);
                        self.moving = Some((i, p));
                        self.dirty = true;
                    }
                }
                if resp.drag_stopped() {
                    self.moving = None;
                }
                if let Some(i) = self.selected {
                    if i < self.doc.shapes.len() && (key(ui, egui::Key::Delete) || key(ui, egui::Key::Backspace)) && !ui.ctx().egui_wants_keyboard_input() {
                        self.doc.shapes.remove(i);
                        self.selected = None;
                        self.dirty = true;
                    }
                }
            }
        }
        if let Some(p) = pointer {
            painter.text(
                rect.right_top() + Vec2::new(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
                format!("x {:.1} mm, y {:.1} mm", p[0], p[1]),
                egui::FontId::proportional(12.0),
                Color32::LIGHT_GRAY,
            );
        }
    }
}

/// Key press (the tool buttons give up keyboard focus, so Enter/Esc reach the canvas)
fn key(ui: &Ui, key: egui::Key) -> bool {
    ui.input(|i| i.key_pressed(key))
}

fn drag_geom(tool: Tool, a: [f64; 2], b: [f64; 2]) -> Geom {
    match tool {
        Tool::Ellipse => Geom::Ellipse { center: [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0], radius: [(a[0] - b[0]).abs() / 2.0, (a[1] - b[1]).abs() / 2.0] },
        _ => Geom::Rect { a: [a[0].min(b[0]), a[1].min(b[1])], b: [a[0].max(b[0]), a[1].max(b[1])] },
    }
}

// ---------------------------------------------------------------------------------------------

pub enum DialogResult {
    Open,
    Cancel,
    Saved(String),
}

/// Assign tissues to the fill colours of an imported SVG drawing
pub struct SvgImportDialog {
    name: String,
    svg: String,
    colors: Vec<(String, Option<String>)>,
    message: Option<String>,
}

fn parse_hex(c: &str) -> Color32 {
    let h = c.trim_start_matches('#');
    if h.len() == 6 {
        if let Ok(v) = u32::from_str_radix(h, 16) {
            return Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8);
        }
    }
    Color32::GRAY
}

impl SvgImportDialog {
    pub fn new(name: String, svg: String, colors: Vec<(String, Option<String>)>) -> Self {
        SvgImportDialog { name, svg, colors, message: None }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult {
        let mut result = DialogResult::Open;
        egui::Modal::new(egui::Id::unique("svg_import")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.heading("Import SVG phantom");
            ui.label("Assign a tissue to each fill colour of the drawing (unassigned colours are ignored).");
            ui.add_space(6.0);
            egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                for (i, (c, t)) in self.colors.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        swatch(ui, parse_hex(c));
                        ui.label(RichText::new(c.as_str()).monospace());
                        tissue_combo(ui, ("svg_color", i), t, "— ignore —", 200.0);
                    });
                }
            });
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(220.0));
            });
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Import").strong()).clicked() {
                    match phantom_maker::save_svg_phantom(&self.name, &self.svg, &self.colors) {
                        Ok(name) => result = DialogResult::Saved(name),
                        Err(e) => self.message = Some(e),
                    }
                }
                if ui.button("Cancel").clicked() {
                    result = DialogResult::Cancel;
                }
            });
            if let Some(m) = &self.message {
                ui.label(RichText::new(m).color(Color32::from_rgb(230, 120, 80)));
            }
        });
        result
    }
}
