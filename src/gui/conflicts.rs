//! Parameter conflicts, as on MR scanners: sliders show in colour which values change other
//! parameters (and are extended by such values), and a dialog asks before such a change is made.

use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use egui::{Color32, Pos2, Rect, RichText, Stroke, Ui};

use crate::controller::{Adjustment, Controller, SliderRange};
use crate::params::Value;

/// Values that change no other parameter
pub const FREE_COLOR: Color32 = Color32::from_rgb(70, 170, 90);
/// Values for which other parameters are adjusted
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

/// Paint the colour band of a slider (below its rail): green where no other parameter changes,
/// orange where other parameters would be adjusted
pub fn paint_band(ui: &Ui, slider_rect: Rect, adjusts: &[bool]) {
    let n = adjusts.len();
    if n < 2 {
        return;
    }
    // positions of the handle centre (egui: the rect shrunk by the handle radius)
    let r = slider_rect.height() / 2.5;
    let (x0, x1) = (slider_rect.left() + r, slider_rect.right() - r);
    let y = slider_rect.center().y + 6.0;
    let x_of = |i: f32| x0 + (x1 - x0) * i / (n - 1) as f32;
    let painter = ui.painter();
    let mut start = 0usize;
    for i in 1..=n {
        if i == n || adjusts[i] != adjusts[start] {
            let a = if start == 0 { x0 } else { x_of(start as f32 - 0.5) };
            let b = if i == n { x1 } else { x_of(i as f32 - 0.5) };
            let color = if adjusts[start] { ADJUST_COLOR } else { FREE_COLOR };
            painter.line_segment([Pos2::new(a, y), Pos2::new(b, y)], Stroke::new(3.0, color));
            start = i;
        }
    }
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
