//! Slice planning on three localizer views (transversal, sagittal, coronal) of a 3D phantom,
//! like on a scanner console: the slice stack is shown in every view and can be moved by
//! dragging and angulated with the rotation handle; the localizer planes are chosen by clicking
//! (crosshair) or with the mouse wheel.

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2};

use crate::geometry::{self, dot, edge_letters, position_label, Orientation, SliceGeometry, StackSpec, V3};
use crate::params::Value;
use crate::protocol::Inputs;
use crate::volume::Volume;

struct ViewDef {
    orientation: Orientation,
    /// screen right / down
    u: V3,
    v: V3,
    /// plane normal and the RAS axis it is aligned with
    normal: V3,
    axis: usize,
}

fn views() -> [ViewDef; 3] {
    [Orientation::Transversal, Orientation::Sagittal, Orientation::Coronal].map(|o| {
        let (u, v) = o.base_axes();
        let normal = geometry::cross(u, v);
        let axis = (0..3).max_by(|&a, &b| normal[a].abs().partial_cmp(&normal[b].abs()).unwrap()).unwrap();
        ViewDef { orientation: o, u, v, normal, axis }
    })
}

#[derive(Debug, Clone, Copy)]
enum Drag {
    Move { last: V3 },
    Rotate { view: usize, last_angle: f64 },
}

pub struct Planning {
    /// point where the three localizer planes cross (RAS) [mm]
    pub loc: V3,
    textures: [Option<(TextureHandle, i64)>; 3],
    drag: Option<Drag>,
    initialized: bool,
}

impl Default for Planning {
    fn default() -> Self {
        Planning { loc: [0.0, -15.0, 18.0], textures: [None, None, None], drag: None, initialized: false }
    }
}

const SLICE_COLOR: Color32 = Color32::from_rgb(255, 210, 0);
pub(super) const SERIES_COLOR: Color32 = Color32::from_rgb(0, 210, 255);

/// Convex hull (monotone chain) of 2D points
fn convex_hull(mut pts: Vec<Pos2>) -> Vec<Pos2> {
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: Pos2, a: Pos2, b: Pos2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lower: Vec<Pos2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Pos2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn point_in_polygon(p: Pos2, poly: &[Pos2]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Outline of a projected rectangle; degenerate (edge-on) rectangles are drawn as a segment
fn draw_outline(painter: &egui::Painter, pts: &[Pos2], stroke: Stroke) {
    let hull = convex_hull(pts.to_vec());
    let area: f32 = (0..hull.len())
        .map(|i| {
            let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f32>()
        .abs()
        / 2.0;
    if hull.len() >= 3 && area > 2.0 {
        painter.add(egui::Shape::closed_line(hull, stroke));
    } else {
        // the two points farthest apart
        let mut best = (0usize, 0usize, -1.0f32);
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let d = pts[i].distance(pts[j]);
                if d > best.2 {
                    best = (i, j, d);
                }
            }
        }
        if best.2 > 0.0 {
            painter.line_segment([pts[best.0], pts[best.1]], stroke);
        }
    }
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t).distance(p)
}

/// FOV rectangle corners of a slice (in-plane extent `fov` = [vertical, horizontal]) at offset z
fn fov_corners(g: &SliceGeometry, fov: [f64; 2], z: f64) -> [V3; 4] {
    let (h, w) = (fov[0] / 2.0, fov[1] / 2.0);
    [g.point(-w, -h, z), g.point(w, -h, z), g.point(w, h, z), g.point(-w, h, z)]
}

/// Section of the slab of slice `g` (FOV × thickness box) with the plane `dot(x, normal) = depth`
fn slab_section(g: &SliceGeometry, fov: [f64; 2], normal: V3, depth: f64) -> Vec<V3> {
    let (h, w, t) = (fov[0] / 2.0, fov[1] / 2.0, g.thickness / 2.0);
    let corner = |i: usize| g.point(if i & 1 == 0 { -w } else { w }, if i & 2 == 0 { -h } else { h }, if i & 4 == 0 { -t } else { t });
    let mut pts = Vec::new();
    for a in 0..8usize {
        for bit in [1usize, 2, 4] {
            if a & bit != 0 {
                continue;
            }
            let (p, q) = (corner(a), corner(a | bit));
            let (dp, dq) = (dot(p, normal) - depth, dot(q, normal) - depth);
            if dp.abs() < 1e-9 && dq.abs() < 1e-9 {
                pts.push(p);
                pts.push(q);
            } else if dp * dq <= 0.0 && (dp - dq).abs() > 1e-12 {
                let f = dp / (dp - dq);
                pts.push(geometry::add(p, geometry::scale(geometry::sub(q, p), f)));
            }
        }
    }
    pts
}

impl Planning {
    /// Show the three localizer views; returns parameter changes
    #[allow(clippy::too_many_arguments)]
    /// `series`: slices of a reviewed series (cyan), `highlight` the one shown in the viewer
    pub fn show(
        &mut self,
        ui: &mut Ui,
        vol: &Volume,
        inp: &Inputs,
        fov: [f64; 2],
        series: &[SliceGeometry],
        highlight: Option<usize>,
        size: f32,
        help: bool,
    ) -> Vec<(&'static str, Value)> {
        let mut changes = Vec::new();
        let stack = inp.stack();
        if !self.initialized {
            self.loc = stack.center;
            self.initialized = true;
        }
        let views = views();
        for (i, view) in views.iter().enumerate() {
            ui.vertical(|ui| {
                self.view(ui, i, view, vol, inp, &stack, fov, series, highlight, size, &mut changes);
            });
        }
        let _ = help;
        changes
    }

    #[allow(clippy::too_many_arguments)]
    fn view(
        &mut self,
        ui: &mut Ui,
        index: usize,
        view: &ViewDef,
        vol: &Volume,
        inp: &Inputs,
        stack: &StackSpec,
        fov: [f64; 2],
        series: &[SliceGeometry],
        highlight: Option<usize>,
        size: f32,
        changes: &mut Vec<(&'static str, Value)>,
    ) {
        // fixed localizer field of view, centred on the volume
        const LOCALIZER_FOV: f64 = 270.0;
        let vc = vol.center();
        let (ac, bc) = (dot(vc, view.u), dot(vc, view.v));
        let (amin, amax) = (ac - LOCALIZER_FOV / 2.0, ac + LOCALIZER_FOV / 2.0);
        let (bmin, bmax) = (bc - LOCALIZER_FOV / 2.0, bc + LOCALIZER_FOV / 2.0);
        let depth = dot(self.loc, view.normal);

        // localizer image
        let key = (depth * 2.0).round() as i64;
        if self.textures[index].as_ref().is_none_or(|(_, k)| *k != key) {
            let origin = geometry::scale(view.normal, depth);
            let (w, h, data) = vol.render_plane(origin, view.u, view.v, (amin, amax), (bmin, bmax), 1.0, &vol.localizer_weights());
            let max = data.iter().cloned().fold(1e-6f32, f32::max);
            let bytes: Vec<u8> = data.iter().map(|v| (v / max * 255.0).clamp(0.0, 255.0) as u8).collect();
            let img = egui::ColorImage::from_gray([w, h], &bytes);
            self.textures[index] = Some((ui.ctx().load_texture(format!("localizer{index}"), img, TextureOptions::LINEAR), key));
        }

        // title
        ui.horizontal(|ui| {
            ui.label(RichText::new(view.orientation.short()).strong());
            ui.label(RichText::new(position_label(view.axis, self.loc[view.axis])).color(Color32::GRAY));
        });
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
        let scale = (rect.width() as f64 / (amax - amin)).min(rect.height() as f64 / (bmax - bmin));
        let (amid, bmid) = ((amin + amax) / 2.0, (bmin + bmax) / 2.0);
        let to_screen = |p: V3| -> Pos2 {
            Pos2::new(rect.center().x + ((dot(p, view.u) - amid) * scale) as f32, rect.center().y + ((dot(p, view.v) - bmid) * scale) as f32)
        };
        let to_world = |s: Pos2| -> V3 {
            let a = (s.x - rect.center().x) as f64 / scale + amid;
            let b = (s.y - rect.center().y) as f64 / scale + bmid;
            geometry::add(geometry::add(geometry::scale(view.u, a), geometry::scale(view.v, b)), geometry::scale(view.normal, depth))
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        if let Some((t, _)) = &self.textures[index] {
            let img_rect = Rect::from_min_max(
                Pos2::new(rect.center().x + ((amin - amid) * scale) as f32, rect.center().y + ((bmin - bmid) * scale) as f32),
                Pos2::new(rect.center().x + ((amax - amid) * scale) as f32, rect.center().y + ((bmax - bmid) * scale) as f32),
            );
            painter.image(t.id(), img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        // crosshair: the other two localizer planes
        let cross = to_screen(self.loc);
        let dashed = Stroke::new(1.0, Color32::from_rgba_unmultiplied(160, 160, 160, 140));
        painter.add(egui::Shape::dashed_line(&[Pos2::new(cross.x, rect.top()), Pos2::new(cross.x, rect.bottom())], dashed, 4.0, 4.0));
        painter.add(egui::Shape::dashed_line(&[Pos2::new(rect.left(), cross.y), Pos2::new(rect.right(), cross.y)], dashed, 4.0, 4.0));

        // slices of reviewed series (cyan)
        for (i, g) in series.iter().enumerate() {
            let pts: Vec<Pos2> = fov_corners(g, fov, 0.0).iter().map(|p| to_screen(*p)).collect();
            let hi = highlight == Some(i);
            draw_outline(&painter, &pts, Stroke::new(if hi { 2.0 } else { 1.0 }, SERIES_COLOR.gamma_multiply(if hi { 1.0 } else { 0.55 })));
            if hi {
                let cut = convex_hull(slab_section(g, fov, view.normal, depth).iter().map(|p| to_screen(*p)).collect());
                let in_plane = dot(g.n, view.normal).abs() > 0.9995;
                if cut.len() >= 3 {
                    painter.add(egui::Shape::convex_polygon(cut, SERIES_COLOR.gamma_multiply(if in_plane { 0.06 } else { 0.3 }), Stroke::NONE));
                }
            }
        }

        // planned stack
        let slices = stack.slices();
        let shown = inp.displayed_slice.min(slices.len() - 1);
        for (i, g) in slices.iter().enumerate() {
            let pts: Vec<Pos2> = fov_corners(g, fov, 0.0).iter().map(|p| to_screen(*p)).collect();
            let w = if i == shown { 2.0 } else { 1.0 };
            draw_outline(&painter, &pts, Stroke::new(w, SLICE_COLOR.gamma_multiply(if i == shown { 1.0 } else { 0.7 })));
        }
        // exact cut of the displayed slice (its slab) with this localizer plane, and the outline
        // of the whole stack (for grabbing)
        let g = &slices[shown];
        let section: Vec<Pos2> = slab_section(g, fov, view.normal, depth).iter().map(|p| to_screen(*p)).collect();
        let slab_hull = convex_hull(section);
        let (_, _, n) = stack.axes();
        let parallel = dot(n, view.normal).abs() > 0.98;
        // (a slice in the plane of the localizer covers it completely: only a faint tint)
        let in_plane = dot(n, view.normal).abs() > 0.9995;
        if slab_hull.len() >= 3 {
            painter.add(egui::Shape::convex_polygon(slab_hull.clone(), SLICE_COLOR.gamma_multiply(if in_plane { 0.06 } else { 0.35 }), Stroke::NONE));
        }
        let stack_pts: Vec<Pos2> = slices.iter().flat_map(|g| fov_corners(g, fov, 0.0).iter().map(|p| to_screen(*p)).collect::<Vec<_>>()).collect();
        let stack_hull = convex_hull(stack_pts);

        // rotation handle (not in the view parallel to the slices)
        let center_s = to_screen(stack.center);
        let rotation = rotation_param(stack, view, &to_screen);
        let handle = if parallel || rotation.is_none() || dot(n, view.normal).abs() > 0.9 {
            None
        } else {
            // direction of the slice line in this view
            let line_dir = {
                let a = to_screen(geometry::add(stack.center, geometry::scale(geometry::cross(n, view.normal), 10.0)));
                (a - center_s).normalized()
            };
            let reach = fov_corners(&stack.slice(0), fov, 0.0).iter().map(|p| (to_screen(*p) - center_s).dot(line_dir).abs()).fold(0.0f32, f32::max);
            // beyond the end of the slice line, but inside the view
            let inner = rect.shrink(12.0);
            let limit = |c: f32, d: f32, lo: f32, hi: f32| -> f32 {
                if d > 1e-6 {
                    (hi - c) / d
                } else if d < -1e-6 {
                    (lo - c) / d
                } else {
                    f32::INFINITY
                }
            };
            let t_edge = limit(center_s.x, line_dir.x, inner.left(), inner.right()).min(limit(center_s.y, line_dir.y, inner.top(), inner.bottom()));
            Some(center_s + line_dir * (reach + 14.0).min(t_edge.max(0.0)))
        };
        if let Some(h) = handle {
            painter.circle(h, 6.0, Color32::from_rgba_unmultiplied(255, 210, 0, 90), Stroke::new(1.5, SLICE_COLOR));
            painter.text(h + Vec2::new(9.0, -9.0), egui::Align2::LEFT_BOTTOM, "⟲", egui::FontId::proportional(13.0), SLICE_COLOR);
        }

        // orientation letters
        let letters = edge_letters(view.u, view.v);
        let font = egui::FontId::proportional(13.0);
        let lc = Color32::from_gray(200);
        painter.text(rect.center_top() + Vec2::new(0.0, 4.0), egui::Align2::CENTER_TOP, &letters[0], font.clone(), lc);
        painter.text(rect.center_bottom() - Vec2::new(0.0, 4.0), egui::Align2::CENTER_BOTTOM, &letters[1], font.clone(), lc);
        painter.text(rect.left_center() + Vec2::new(4.0, 0.0), egui::Align2::LEFT_CENTER, &letters[2], font.clone(), lc);
        painter.text(rect.right_center() - Vec2::new(4.0, 0.0), egui::Align2::RIGHT_CENTER, &letters[3], font, lc);

        // ---- interaction
        let pointer = resp.hover_pos();
        if resp.drag_started() {
            // where the button went down (the drag is only recognised after some movement)
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            if let Some(p) = origin {
                if handle.is_some_and(|h| h.distance(p) < 11.0) {
                    let d = p - center_s;
                    self.drag = Some(Drag::Rotate { view: index, last_angle: (d.y as f64).atan2(d.x as f64) });
                } else if point_in_polygon(p, &stack_hull) || point_in_polygon(p, &slab_hull) || dist_to_segment_hull(p, &stack_hull) < 8.0 {
                    self.drag = Some(Drag::Move { last: to_world(p) });
                }
            }
        }
        if let (Some(drag), Some(p)) = (self.drag, pointer) {
            if resp.dragged() {
                match drag {
                    Drag::Move { last } => {
                        let w = to_world(p);
                        let d = geometry::sub(w, last);
                        if geometry::norm(d) > 0.0 {
                            for (axis, name) in [(0, "slice_pos_LR"), (1, "slice_pos_AP"), (2, "slice_pos_HF")] {
                                if d[axis].abs() > 1e-9 {
                                    let spec = crate::params::param(name);
                                    let (lo, hi) = spec.bounds.unwrap_or((-200.0, 200.0));
                                    let v = ((inp.slice_pos[axis] + d[axis]) * 2.0).round() / 2.0;
                                    changes.push((name, Value::Num(v.clamp(lo, hi))));
                                }
                            }
                            self.drag = Some(Drag::Move { last: w });
                        }
                    }
                    Drag::Rotate { view: vi, last_angle } if vi == index => {
                        let d = p - center_s;
                        let angle = (d.y as f64).atan2(d.x as f64);
                        let mut dphi = angle - last_angle;
                        if dphi > std::f64::consts::PI {
                            dphi -= std::f64::consts::TAU;
                        }
                        if dphi < -std::f64::consts::PI {
                            dphi += std::f64::consts::TAU;
                        }
                        if let Some((name, sign)) = rotation {
                            let current = if name == "slice_angle_1" { inp.slice_angle_1 } else { inp.slice_angle_2 };
                            let v = (current + sign * dphi.to_degrees()).clamp(-45.0, 45.0);
                            if (v - current).abs() >= 0.5 {
                                changes.push((name, Value::Num(v.round())));
                                self.drag = Some(Drag::Rotate { view: vi, last_angle: angle });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        // click: move the crosshair (the other localizers) to this point
        if resp.clicked() {
            if let Some(p) = pointer {
                let w = to_world(p);
                for axis in 0..3 {
                    if axis != view.axis {
                        self.loc[axis] = w[axis];
                    }
                }
            }
        }
        // mouse wheel: scroll through the localizer planes of this view
        if resp.hovered() {
            let scroll = ui.ctx().input_mut(|i| {
                let d = i.smooth_scroll_delta.y;
                i.smooth_scroll_delta = Vec2::ZERO;
                d
            });
            if scroll != 0.0 {
                let (lo, hi) = vol.world_bounds();
                self.loc[view.axis] = (self.loc[view.axis] + scroll as f64 / 20.0).clamp(lo[view.axis], hi[view.axis]);
            }
            let tip = if handle.is_some_and(|h| pointer.is_some_and(|p| h.distance(p) < 11.0)) {
                "Drag to angulate the slices"
            } else if pointer.is_some_and(|p| point_in_polygon(p, &stack_hull) || point_in_polygon(p, &slab_hull)) {
                "Drag to move the slices"
            } else {
                "Click: show this point in the other views · wheel: scroll through the localizer"
            };
            resp.on_hover_text_at_pointer(tip);
        }
    }

    /// Move the localizer crosshair to the centre of the displayed slice
    pub fn center_on(&mut self, g: &SliceGeometry) {
        self.loc = g.center;
    }
}

fn dist_to_segment_hull(p: Pos2, hull: &[Pos2]) -> f32 {
    let n = hull.len();
    if n == 0 {
        return f32::INFINITY;
    }
    if n == 1 {
        return hull[0].distance(p);
    }
    (0..n).map(|i| dist_to_segment(p, hull[i], hull[(i + 1) % n])).fold(f32::INFINITY, f32::min)
}

/// Which angulation parameter rotates the slices about this view's normal, and with which sign
/// (so that the slice line follows the mouse)
fn rotation_param(stack: &StackSpec, view: &ViewDef, to_screen: &dyn Fn(V3) -> Pos2) -> Option<(&'static str, f64)> {
    let (u0, v0) = stack.orientation.base_axes();
    let (cu, cv) = (dot(u0, view.normal).abs(), dot(v0, view.normal).abs());
    if cu.max(cv) < 0.7 {
        return None;
    }
    let name = if cu >= cv { "slice_angle_1" } else { "slice_angle_2" };
    // screen angle of the projected slice normal as a function of the angulation
    let screen_angle = |s: &StackSpec| -> f64 {
        let (_, _, n) = s.axes();
        let a = to_screen(s.center);
        let b = to_screen(geometry::add(s.center, geometry::scale(n, 10.0)));
        ((b.y - a.y) as f64).atan2((b.x - a.x) as f64)
    };
    let mut s2 = *stack;
    if name == "slice_angle_1" {
        s2.angle1 += 1.0;
    } else {
        s2.angle2 += 1.0;
    }
    let mut d = screen_angle(&s2) - screen_angle(stack);
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    if d.abs() < 1e-6 {
        return None;
    }
    Some((name, d.to_degrees().signum() / d.to_degrees().abs().max(1e-3)))
}
