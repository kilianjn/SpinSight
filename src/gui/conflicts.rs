//! Parameter conflicts, as on MR scanners: sliders show in colour which values change other
//! parameters (and are extended by such values), and a dialog asks before such a change is made.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use egui::{Color32, Pos2, Rect, RichText, Ui};

use crate::controller::{Adjustment, Controller, SliderRange};
use crate::params::Value;

/// Accent colour of conflicts (dialog title, pending values)
pub const ADJUST_COLOR: Color32 = Color32::from_rgb(235, 150, 40);

/// Computes the slider ranges in a background thread (they need many protocol evaluations)
pub struct RangeAnalyzer {
    tx: Sender<(Controller, Vec<&'static str>)>,
    rx: Receiver<SliderRange>,
    ranges: HashMap<&'static str, SliderRange>,
    requested: Option<(u64, Vec<&'static str>)>,
}

impl RangeAnalyzer {
    pub fn spawn(notify: impl Fn() + Send + 'static) -> RangeAnalyzer {
        let (tx, job_rx) = channel::<(Controller, Vec<&'static str>)>();
        let (res_tx, rx) = channel::<SliderRange>();
        let _ = std::thread::Builder::new().name("slider ranges".into()).spawn(move || {
            'jobs: while let Ok(mut job) = job_rx.recv() {
                loop {
                    // only the newest request matters
                    loop {
                        match job_rx.try_recv() {
                            Ok(j) => job = j,
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    let (ctrl, names) = &job;
                    let mut interrupted = false;
                    for name in names {
                        if let Some(r) = ctrl.slider_range(name) {
                            if res_tx.send(r).is_err() {
                                return;
                            }
                            notify();
                        }
                        // a newer request: start over with it
                        match job_rx.try_recv() {
                            Ok(j) => {
                                job = j;
                                interrupted = true;
                                break;
                            }
                            Err(TryRecvError::Empty) => {}
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    if !interrupted {
                        continue 'jobs;
                    }
                }
            }
        });
        RangeAnalyzer { tx, rx, ranges: HashMap::new(), requested: None }
    }

    /// Ask for the ranges of the sliders `names` of the current controller state
    pub fn request(&mut self, ctrl: &Controller, names: Vec<&'static str>) {
        let key = (ctrl.generation, names);
        if self.requested.as_ref() != Some(&key) {
            let _ = self.tx.send((ctrl.clone(), key.1.clone()));
            self.requested = Some(key);
        }
    }

    pub fn poll(&mut self) {
        while let Ok(r) = self.rx.try_recv() {
            let newer = self.ranges.get(r.name).is_none_or(|old| old.generation <= r.generation);
            if newer {
                self.ranges.insert(r.name, r);
            }
        }
    }

    /// The range of slider `name` if it fits the current state of the parameter (also a slightly
    /// outdated one while a new one is computed, so that sliders do not jump while dragging)
    pub fn range(&self, ctrl: &Controller, name: &str) -> Option<&SliderRange> {
        let r = self.ranges.get(name)?;
        let st = ctrl.state(name);
        let (first, last) = (st.options.first()?.value.num(), st.options.last()?.value.num());
        let (rf, rl) = (r.options.first()?.value.num(), r.options.last()?.value.num());
        let contains_value = r.options.iter().any(|o| o.value == st.value);
        (contains_value && rf <= first && rl >= last).then_some(r)
    }

    pub fn clear(&mut self) {
        self.ranges.clear();
        self.requested = None;
    }
}

/// Colours of the slider rail (values that change no other parameter, values that adjust others)
pub fn rail_colors(dark: bool) -> (Color32, Color32) {
    if dark {
        (Color32::from_rgb(52, 122, 72), Color32::from_rgb(190, 122, 38))
    } else {
        (Color32::from_rgb(128, 192, 140), Color32::from_rgb(236, 170, 84))
    }
}

/// Colour the rail of the discrete slider `r` with `n = adjusts.len()` positions: green where no
/// other parameter changes, orange where other parameters would be adjusted; the handle (at
/// position `idx`) is drawn on top again
pub fn paint_rail(ui: &Ui, r: &egui::Response, adjusts: &[bool], idx: usize) {
    let n = adjusts.len();
    if n < 2 {
        return;
    }
    let rect = r.rect;
    let rail_r = (ui.spacing().slider_rail_height / 2.0).max(1.0);
    let rail = Rect::from_min_max(Pos2::new(rect.left(), rect.center().y - rail_r), Pos2::new(rect.right(), rect.center().y + rail_r));
    // positions of the handle centre (as egui: the rect shrunk by the handle radius)
    let hr = rect.height() / 2.5;
    let (x0, x1) = (rect.left() + hr, rect.right() - hr);
    let x_of = |i: f32| x0 + (x1 - x0) * i / (n - 1) as f32;
    let (free, adjust) = rail_colors(ui.visuals().dark_mode);
    let painter = ui.painter();
    let cr = rail_r.round() as u8;
    let mut start = 0usize;
    for i in 1..=n {
        if i == n || adjusts[i] != adjusts[start] {
            let a = if start == 0 { rail.left() } else { x_of(start as f32 - 0.5) };
            let b = if i == n { rail.right() } else { x_of(i as f32 - 0.5) };
            let (left, right) = (if start == 0 { cr } else { 0 }, if i == n { cr } else { 0 });
            let corners = egui::CornerRadius { nw: left, sw: left, ne: right, se: right };
            painter.rect_filled(Rect::from_min_max(Pos2::new(a, rail.top()), Pos2::new(b, rail.bottom())), corners, if adjusts[start] { adjust } else { free });
            start = i;
        }
    }
    let visuals = ui.style().interact(r);
    painter.add(egui::epaint::CircleShape {
        center: Pos2::new(x_of(idx.min(n - 1) as f32), rect.center().y),
        radius: hr + visuals.expansion,
        fill: visuals.bg_fill,
        stroke: visuals.fg_stroke,
    });
}

/// Option index under the pointer of a hovered slider
pub fn hovered_index(ui: &Ui, slider_rect: Rect, n: usize) -> Option<usize> {
    let p = ui.input(|i| i.pointer.hover_pos())?;
    if n < 2 || !slider_rect.expand(4.0).contains(p) {
        return None;
    }
    let r = slider_rect.height() / 2.5;
    let (x0, x1) = (slider_rect.left() + r, slider_rect.right() - r);
    let f = ((p.x - x0) / (x1 - x0)).clamp(0.0, 1.0);
    Some((f * (n - 1) as f32).round() as usize)
}

/// Text listing adjustments: "TR 600 ms → 620 ms, TE …"
pub fn describe(adj: &[Adjustment]) -> String {
    adj.iter().map(|a| format!("{} {} → {}", a.label, a.old, a.new)).collect::<Vec<_>>().join("\n")
}

/// A change that waits for confirmation
pub struct PendingConflict {
    pub name: &'static str,
    pub label: String,
    pub old: String,
    pub value: Value,
    pub new: String,
    pub adjustments: Vec<Adjustment>,
}

pub enum Decision {
    Accept,
    Cancel,
}

/// The confirmation dialog; `warn` is the option to keep asking
pub fn dialog(ctx: &egui::Context, pending: &PendingConflict, warn: &mut bool) -> Option<Decision> {
    let mut decision = None;
    egui::Modal::new(egui::Id::unique("parameter_conflict")).show(ctx, |ui| {
        ui.set_max_width(460.0);
        ui.heading(RichText::new("⚠ Parameter conflict").color(ADJUST_COLOR));
        ui.add_space(4.0);
        ui.label(RichText::new(format!("{}: {} → {}", pending.label, pending.old, pending.new)).strong());
        ui.label("This change also adjusts the following parameters:");
        ui.add_space(4.0);
        egui::Grid::new("conflict_grid").striped(true).spacing([18.0, 4.0]).show(ui, |ui| {
            for a in &pending.adjustments {
                ui.label(&a.label);
                ui.label(&a.old);
                ui.label("→");
                ui.label(RichText::new(&a.new).strong());
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.checkbox(warn, "Warn before other parameters are changed");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Accept").strong()).on_hover_text("Enter").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                decision = Some(Decision::Accept);
            }
            if ui.button("Cancel").on_hover_text("Esc").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                decision = Some(Decision::Cancel);
            }
        });
    });
    decision
}
