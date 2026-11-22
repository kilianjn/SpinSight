//! Image, k-space and pulse sequence plots.

use egui::{Align2, Color32, Id, Rect, Stroke, TextureHandle, TextureOptions, Ui, Vec2};
use egui_plot::{Arrows, FilledArea, HLine, Line, Plot, PlotBounds, PlotImage, PlotPoint, PlotPoints, Polygon, Span, Text, VLine};

use crate::array::Array2;
use crate::constants::MAX_AMP;
use crate::formatting;
use crate::protocol::{self, Protocol};
use crate::sequence::Waveform;

pub const CADETBLUE: Color32 = Color32::from_rgb(95, 158, 160);
pub const RED: Color32 = Color32::from_rgb(255, 0, 0);
pub const ORANGE: Color32 = Color32::from_rgb(255, 165, 0);
pub const PERU: Color32 = Color32::from_rgb(205, 133, 63);
pub const LIGHTBLUE: Color32 = Color32::from_rgb(173, 216, 230);
pub const YELLOW: Color32 = Color32::from_rgb(255, 255, 0);
pub const GRAY: Color32 = Color32::from_rgb(128, 128, 128);

const TIME_BOUNDS: (f64, f64) = (-19000.0, 19000.0);
const TIME_INITIAL_RANGE: (f64, f64) = (-2.0, 40.0);
const G_RANGE: f64 = MAX_AMP * 1.05;
const BOARD_HEIGHT: f32 = 140.0;
const BOARD_HEIGHT_LAST: f32 = 190.0;
const Y_AXIS_WIDTH: f32 = 48.0;

/// Annotation colour (yellow on dark backgrounds, darker gold on light ones)
pub fn annotation_color(ui: &Ui) -> Color32 {
    if ui.visuals().dark_mode {
        YELLOW
    } else {
        Color32::from_rgb(200, 150, 0)
    }
}

fn alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a * 255.0) as u8)
}

/// Grayscale texture (linear min..max like Bokeh's default colour mapper)
pub fn gray_texture(ctx: &egui::Context, name: &str, a: &Array2<f64>) -> Option<TextureHandle> {
    if a.rows == 0 || a.cols == 0 {
        return None;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &a.data {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let scale = if hi > lo { 255.0 / (hi - lo) } else { 0.0 };
    let bytes: Vec<u8> = a.data.iter().map(|&v| if v.is_finite() { ((v - lo) * scale).round().clamp(0.0, 255.0) as u8 } else { 0 }).collect();
    let img = egui::ColorImage::from_gray([a.cols, a.rows], &bytes);
    Some(ctx.load_texture(name, img, TextureOptions::NEAREST))
}

/// Swallow mouse-wheel input so that the surrounding scroll area does not scroll as well
fn take_scroll(ctx: &egui::Context) -> f32 {
    ctx.input_mut(|i| {
        let d = i.smooth_scroll_delta.y + i.smooth_scroll_delta.x;
        i.smooth_scroll_delta = Vec2::ZERO;
        d
    })
}

fn wheel_zoom(plot_ui: &mut egui_plot::PlotUi, xy: [bool; 2]) {
    if !plot_ui.response().hovered() {
        return;
    }
    let d = take_scroll(plot_ui.ctx());
    if d != 0.0 {
        let f = (d / 200.0).exp();
        plot_ui.zoom_bounds_around_hovered(Vec2::new(if xy[0] { f } else { 1.0 }, if xy[1] { f } else { 1.0 }));
    }
}

fn rect_line(w: f64, h: f64) -> Vec<[f64; 2]> {
    vec![[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0], [-w / 2.0, -h / 2.0]]
}

/// FOV overlays on the image (`FOV_box`)
pub struct FovBox {
    pub reconstructed: [f64; 2],
    pub acquired_box: Option<[f64; 2]>,
    pub acquired_circle: Option<f64>,
}

pub fn fov_box(p: &Protocol) -> FovBox {
    if p.is_radial {
        let d = p.fov[p.freq_dir] * p.k_read_axis.len() as f64 / p.matrix[p.freq_dir] as f64;
        FovBox { reconstructed: p.fov, acquired_box: None, acquired_circle: Some(d) }
    } else {
        let mut acq = p.fov;
        acq[p.phase_dir] *= p.k_phase_axis.len() as f64 / p.matrix[p.phase_dir] as f64;
        FovBox { reconstructed: p.fov, acquired_box: Some(acq), acquired_circle: None }
    }
}

/// MR image plot; `fov` = [height, width] in mm
/// `letters`: patient orientation at the (top, bottom, left, right) image edges (3D phantoms)
#[allow(clippy::too_many_arguments)]
pub fn image_plot(ui: &mut Ui, tex: Option<&TextureHandle>, fov: [f64; 2], fov_box: Option<&FovBox>, letters: Option<&[String; 4]>, size: f32, reset: bool) {
    let annotation = annotation_color(ui);
    let mut plot = Plot::new("image_plot")
        .show_x(false)
        .show_y(false)
        .width(size)
        .height(size)
        .data_aspect(1.0)
        .x_axis_label("x [mm]")
        .y_axis_label("y [mm]")
        .allow_scroll(false)
        .allow_zoom(false)
        .allow_boxed_zoom(true)
        .show_grid(false)
        .default_x_bounds(-fov[1] / 2.0, fov[1] / 2.0)
        .default_y_bounds(-fov[0] / 2.0, fov[0] / 2.0);
    if reset {
        plot = plot.reset();
    }
    plot.show(ui, |plot_ui| {
        wheel_zoom(plot_ui, [true, true]);
        if let Some(t) = tex {
            plot_ui.image(PlotImage::new("image", t.id(), PlotPoint::new(0.0, 0.0), Vec2::new(fov[1] as f32, fov[0] as f32)));
        }
        if let Some(b) = fov_box {
            if let Some(a) = b.acquired_box {
                plot_ui.line(Line::new("acquired FOV", PlotPoints::from(rect_line(a[1], a[0]))).color(LIGHTBLUE).width(3.0));
            }
            if let Some(d) = b.acquired_circle {
                let pts: Vec<[f64; 2]> = (0..=180)
                    .map(|i| {
                        let a = i as f64 / 180.0 * std::f64::consts::TAU;
                        [d / 2.0 * a.cos(), d / 2.0 * a.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new("acquired FOV", PlotPoints::from(pts)).color(LIGHTBLUE).width(3.0));
            }
            plot_ui.line(Line::new("reconstructed FOV", PlotPoints::from(rect_line(b.reconstructed[1], b.reconstructed[0]))).color(annotation).width(3.0));
        }
        if let Some(l) = letters {
            let (h, w) = (fov[0] / 2.0, fov[1] / 2.0);
            let col = Color32::from_gray(210);
            let txt = |s: &str| egui::RichText::new(s).size(15.0).color(col).strong();
            plot_ui.text(Text::new("top", PlotPoint::new(0.0, h), txt(&l[0])).anchor(Align2::CENTER_TOP));
            plot_ui.text(Text::new("bottom", PlotPoint::new(0.0, -h), txt(&l[1])).anchor(Align2::CENTER_BOTTOM));
            plot_ui.text(Text::new("left", PlotPoint::new(-w, 0.0), txt(&l[2])).anchor(Align2::LEFT_CENTER));
            plot_ui.text(Text::new("right", PlotPoint::new(w, 0.0), txt(&l[3])).anchor(Align2::RIGHT_CENTER));
        }
    });
}

/// k-space plot; rows of the texture are dim 0 (ky, upwards), columns dim 1 (kx)
pub fn kspace_plot(ui: &mut Ui, tex: Option<&TextureHandle>, axes: &[Vec<f64>; 2], k_line: Option<&[[f64; 2]]>, size: f32, reset: bool) {
    let extent = |ax: &[f64]| -> (f64, f64) {
        if ax.len() < 2 {
            return (-0.5, 0.5);
        }
        let d = (ax[ax.len() - 1] - ax[0]) / (ax.len() - 1) as f64;
        (ax[0] - d / 2.0, ax[ax.len() - 1] + d / 2.0)
    };
    let (x0, x1) = extent(&axes[1]);
    let (y0, y1) = extent(&axes[0]);
    let lim = 1.12 * axes[1].iter().cloned().fold(0.0f64, f64::max);
    let mut plot = Plot::new("kspace_plot")
        .show_x(false)
        .show_y(false)
        .width(size)
        .height(size)
        .data_aspect(1.0)
        .x_axis_label("kx [1/mm]")
        .y_axis_label("ky [1/mm]")
        .allow_scroll(false)
        .allow_zoom(false)
        .allow_boxed_zoom(true)
        .show_grid(false)
        .default_x_bounds(-lim, lim)
        .default_y_bounds(-lim, lim);
    if reset {
        plot = plot.reset();
    }
    plot.show(ui, |plot_ui| {
        wheel_zoom(plot_ui, [true, true]);
        if let Some(t) = tex {
            plot_ui.image(
                PlotImage::new("k-space", t.id(), PlotPoint::new((x0 + x1) / 2.0, (y0 + y1) / 2.0), Vec2::new((x1 - x0) as f32, (y1 - y0) as f32))
                    .uv(Rect::from_min_max(egui::pos2(0.0, 1.0), egui::pos2(1.0, 0.0))),
            );
        }
        if let Some(line) = k_line {
            if line.len() > 1 {
                plot_ui.line(Line::new("k trajectory", PlotPoints::from(line.to_vec())).color(PERU).width(4.0));
                let n = line.len();
                plot_ui.arrows(Arrows::new("k", PlotPoints::from(vec![line[n - 2]]), PlotPoints::from(vec![line[n - 1]])).color(PERU).tip_length(18.0));
            }
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Board {
    Frequency,
    Phase,
    Slice,
    Rf,
    Signal,
}

impl Board {
    pub const ALL: [Board; 5] = [Board::Frequency, Board::Phase, Board::Slice, Board::Rf, Board::Signal];
    fn label(self) -> &'static str {
        match self {
            Board::Frequency => "G read [mT/m]",
            Board::Phase => "G phase [mT/m]",
            Board::Slice => "G slice [mT/m]",
            Board::Rf => "RF [μT]",
            Board::Signal => "signal [a.u.]",
        }
    }
    fn range(self) -> (f64, f64) {
        match self {
            Board::Rf => (-3.0, 22.0),
            Board::Signal => (-1.0, 1.0),
            _ => (-G_RANGE, G_RANGE),
        }
    }
    fn color(self) -> Color32 {
        match self {
            Board::Rf => RED,
            Board::Signal => ORANGE,
            _ => CADETBLUE,
        }
    }
}

/// A hovered sequence object
#[derive(Debug, Clone)]
pub struct Hovered {
    pub board: Board,
    pub t0: f64,
    pub t1: f64,
}

/// Which annotation spans to show
#[derive(Debug, Clone, Copy, Default)]
pub struct Spans {
    pub te: bool,
    pub tr: bool,
    pub ti: bool,
}

/// Fill between zero and the curve, with zero crossings inserted
fn zero_area(name: &str, pts: &[(f64, f64)], fill: Color32, stroke: Option<Stroke>) -> FilledArea {
    let mut xs = Vec::with_capacity(pts.len() * 2);
    let mut ys = Vec::with_capacity(pts.len() * 2);
    for (i, &(x, y)) in pts.iter().enumerate() {
        if i > 0 {
            let (px, py) = pts[i - 1];
            if py * y < 0.0 {
                let xc = px + (x - px) * py / (py - y);
                xs.push(xc);
                ys.push(0.0);
            }
        }
        xs.push(x);
        ys.push(y);
    }
    let lo: Vec<f64> = ys.iter().map(|y| y.min(0.0)).collect();
    let hi: Vec<f64> = ys.iter().map(|y| y.max(0.0)).collect();
    let mut a = FilledArea::new(name, &xs, &lo, &hi).fill_color(fill);
    if let Some(s) = stroke {
        a = a.stroke(s);
    }
    a
}

fn point_in_polygon(x: f64, y: f64, poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn waveform_points(w: &Waveform) -> Vec<(f64, f64)> {
    w.time.iter().copied().zip(w.amp.iter().copied()).collect()
}

fn tooltip_lines(board: Board, w: &Waveform) -> Vec<(&'static str, String)> {
    let mut v = vec![("name", w.name.clone()), ("center", formatting::seq_time(w.center)), ("duration", formatting::seq_time(w.dur))];
    match board {
        Board::Rf => v.push(("flip_angle", formatting::flip_angle(w.flip_angle))),
        Board::Signal => {}
        _ => v.push(("area", formatting::grad_area(w.area))),
    }
    v
}

/// The pulse sequence diagram; returns the hovered object (if any)
pub fn sequence_plot(ui: &mut Ui, p: &Protocol, signal_curves: &[Waveform], spans: Spans, reset: bool) -> Option<Hovered> {
    let annotation = annotation_color(ui);
    let link = Id::unique("sequence_time_axis");
    let width = ui.available_width().max(400.0);
    let mut hovered = None;
    let ti = p.boards.rf.iter().find(|w| w.name == "inversion").map(|w| -w.center);
    for board in Board::ALL {
        let last = board == Board::Signal;
        let objects: &[Waveform] = match board {
            Board::Frequency => &p.boards.frequency,
            Board::Phase => &p.boards.phase,
            Board::Slice => &p.boards.slice,
            Board::Rf => &p.boards.rf,
            Board::Signal => &p.boards.adc,
        };
        let (ylo, yhi) = board.range();
        let mut plot = Plot::new(("sequence_board", board as u8))
            .width(width)
            .height(if last { BOARD_HEIGHT_LAST } else { BOARD_HEIGHT })
            .link_axis(link, [true, false])
            .link_cursor(link, [true, false])
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_drag([true, false])
            .show_grid([false, false])
            .y_axis_label(egui::RichText::new(board.label()).size(11.0))
            .y_axis_min_width(Y_AXIS_WIDTH)
            .default_x_bounds(TIME_INITIAL_RANGE.0, TIME_INITIAL_RANGE.1)
            .default_y_bounds(ylo, yhi)
            .show_x(false)
            .show_y(false)
            .show_axes([last, true]);
        if last {
            plot = plot.x_axis_label("time [ms]");
        }
        if board == Board::Signal {
            plot = plot.y_axis_formatter(|_, _| String::new());
        }
        if reset {
            plot = plot.reset();
        }
        let resp = plot.show(ui, |plot_ui| {
            wheel_zoom(plot_ui, [true, false]);
            // keep y fixed and x within bounds
            let b = plot_ui.plot_bounds();
            let (bx0, bx1) = (b.min()[0], b.max()[0]);
            let (mut x0, mut x1) = (bx0, bx1);
            if x1 - x0 > TIME_BOUNDS.1 - TIME_BOUNDS.0 {
                (x0, x1) = TIME_BOUNDS;
            }
            if x0 < TIME_BOUNDS.0 {
                x1 += TIME_BOUNDS.0 - x0;
                x0 = TIME_BOUNDS.0;
            }
            if x1 > TIME_BOUNDS.1 {
                x0 -= x1 - TIME_BOUNDS.1;
                x1 = TIME_BOUNDS.1;
            }
            // (only when needed: with linked axes, setting the bounds overrides the other boards)
            if x0 != bx0 || x1 != bx1 || (b.min()[1] - ylo).abs() > 1e-9 || (b.max()[1] - yhi).abs() > 1e-9 {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([x0, ylo], [x1, yhi]));
            }

            // outside TR
            plot_ui.span(Span::new("", -20000.0..=p.sequence_start).fill(alpha(GRAY, 0.3)).border_width(0.0));
            plot_ui.span(Span::new("", (p.sequence_start + p.tr)..=20000.0).fill(alpha(GRAY, 0.3)).border_width(0.0));
            plot_ui.hline(HLine::new("zero", 0.0).color(GRAY).width(1.5));

            // net waveform / signal
            if board == Board::Signal {
                for (i, s) in signal_curves.iter().enumerate() {
                    plot_ui.add(zero_area(&format!("signal {i}"), &waveform_points(s), alpha(ORANGE, 0.5), Some(Stroke::new(1.0, GRAY))));
                }
            } else {
                let net = protocol::net_waveform(objects);
                plot_ui.add(zero_area("net", &net, alpha(board.color(), 0.5), Some(Stroke::new(1.5, GRAY))));
            }

            // hover detection
            let pointer = plot_ui.pointer_coordinate();
            let mut hit: Option<usize> = None;
            if let Some(pt) = pointer {
                if plot_ui.response().hovered() {
                    for (i, w) in objects.iter().enumerate() {
                        let inside =
                            if board == Board::Signal { pt.x >= w.t_start() && pt.x <= w.t_end() } else { point_in_polygon(pt.x, pt.y, &waveform_points(w)) };
                        if inside {
                            hit = Some(i);
                            break;
                        }
                    }
                }
            }
            if board == Board::Signal {
                for (i, w) in objects.iter().enumerate() {
                    let h = hit == Some(i);
                    let rect = vec![[w.t_start(), -2.0], [w.t_end(), -2.0], [w.t_end(), 2.0], [w.t_start(), 2.0]];
                    plot_ui.polygon(
                        Polygon::new("ADC", PlotPoints::from(rect))
                            .fill_color(alpha(PERU, if h { 0.8 } else { 0.1 }))
                            .stroke(Stroke::new(1.0, alpha(PERU, 0.3))),
                    );
                }
            } else if let Some(i) = hit {
                plot_ui.add(zero_area("hovered", &waveform_points(&objects[i]), alpha(board.color(), 0.8), Some(Stroke::new(1.5, GRAY))));
            }

            // annotation spans
            let interval = |plot_ui: &mut egui_plot::PlotUi, t0: f64, t1: f64| {
                plot_ui.span(Span::new("", t0..=t1).fill(alpha(annotation, 0.1)).border_width(0.0));
                plot_ui.vline(VLine::new("t0", t0).color(annotation).width(1.5));
                plot_ui.vline(VLine::new("t1", t1).color(annotation).width(1.5));
            };
            let labelled = |plot_ui: &mut egui_plot::PlotUi, t0: f64, t1: f64, tl: f64, y: f64, label: &str, align: Align2| {
                plot_ui.line(Line::new(label, PlotPoints::from(vec![[t0, y], [t1, y]])).color(annotation).width(3.0));
                plot_ui.text(Text::new(label, PlotPoint::new(tl, y), egui::RichText::new(label).size(20.0).color(annotation)).anchor(align));
            };
            if spans.te {
                interval(plot_ui, 0.0, p.te);
                if board == Board::Rf {
                    labelled(plot_ui, 0.0, p.te, p.te / 4.0, 12.0, if p.num_segm == 1 { "TE" } else { "Effective TE" }, Align2::LEFT_BOTTOM);
                }
            }
            if spans.tr {
                interval(plot_ui, p.sequence_start, p.sequence_start + p.tr);
                if board == Board::Slice {
                    let pos = p.readouts.last().and_then(|r| r.last()).map(|r| r.center).unwrap_or(0.0);
                    labelled(plot_ui, p.sequence_start, p.sequence_start + p.tr, pos, -20.0, "TR", Align2::RIGHT_BOTTOM);
                }
            }
            if spans.ti {
                if let Some(ti) = ti {
                    interval(plot_ui, -ti, 0.0);
                    if board == Board::Rf {
                        labelled(plot_ui, -ti, 0.0, -ti / 8.0, 12.0, "TI", Align2::RIGHT_BOTTOM);
                    }
                }
            }
            hit
        });
        if let Some(i) = resp.inner {
            let w = &objects[i];
            if board != Board::Slice {
                hovered = Some(Hovered { board, t0: w.t_start(), t1: w.t_end() });
            }
            let lines = tooltip_lines(board, w);
            resp.response.on_hover_ui_at_pointer(|ui| {
                egui::Grid::new("seq_tooltip").num_columns(2).show(ui, |ui| {
                    for (k, v) in lines {
                        ui.label(egui::RichText::new(k).weak());
                        ui.label(egui::RichText::new(v).strong());
                        ui.end_row();
                    }
                });
            });
        }
    }
    hovered
}
