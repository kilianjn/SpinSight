//! The SpinSight desktop application (egui), laid out like an MR scanner console:
//! exam bar (phantom, field strength, parameter style) on top, protocol and phantom browser on
//! the left, viewing segments (image, k-space, pulse sequence) in the centre and the parameter
//! card with its tabs and the acquisition status line at the bottom.

mod conflicts;
mod exam_ui;
mod maker;
mod markdown;
mod planning;
pub mod plots;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use egui::{Color32, RichText, TextureHandle, Ui};

use crate::controller::{Controller, ParamState, SliderRange, UNCHECKED_CHANGES};
use crate::imaging::{self, simulation_key};
use crate::params::{Value, Widget, CUSTOM_PRESET};
use crate::phantom;
use crate::phantom_maker;
use crate::protocol::{self, Inputs, KTrajectory};
use crate::sequence::Waveform;
use crate::tooltips;
use crate::worker::{Job, JobResult, Worker};
use plots::Spans;

const PARAM_WIDTH: f32 = 270.0;
const MIN_IMAGE_SIZE: f32 = 280.0;

pub struct Options {
    /// `None` = follow the system theme
    pub dark_mode: Option<bool>,
    pub settings_file: Option<PathBuf>,
    pub sliders_with_text: bool,
    pub start_time: String,
    pub show_sequence: bool,
    pub show_kspace: bool,
    pub help_mode: bool,
    /// parameter changes applied at startup (after the settings file)
    pub sets: Vec<(String, Value)>,
    /// open the phantom maker at startup (optionally with an image to segment)
    pub phantom_maker: Option<Option<PathBuf>>,
}

/// Tabs of the parameter card (as on a scanner console)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CardTab {
    Routine,
    Contrast,
    Resolution,
    Geometry,
    Sequence,
    Recon,
}

impl CardTab {
    const ALL: [CardTab; 6] = [CardTab::Routine, CardTab::Contrast, CardTab::Resolution, CardTab::Geometry, CardTab::Sequence, CardTab::Recon];

    fn label(self) -> &'static str {
        match self {
            CardTab::Routine => "Routine",
            CardTab::Contrast => "Contrast",
            CardTab::Resolution => "Resolution",
            CardTab::Geometry => "Geometry",
            CardTab::Sequence => "Sequence",
            CardTab::Recon => "Recon",
        }
    }

    /// Parameters on the tab (hidden ones are skipped automatically). Like on scanners, the
    /// most important parameters appear on the Routine tab as well as on their own tab.
    fn params(self) -> &'static [&'static str] {
        match self {
            CardTab::Routine => &[
                "slice_orientation",
                "num_slices",
                "sequence_type",
                "slice_thickness",
                "FOV_F",
                "FOV_P",
                "phase_oversampling",
                "TR",
                "TE",
                "TI",
                "FA",
                "NSA",
                "frequency_direction",
            ],
            CardTab::Contrast => &["TR", "TE", "TI", "FA", "FatSat", "echo_order"],
            CardTab::Resolution => &[
                "matrix_F",
                "matrix_P",
                "voxel_F",
                "voxel_P",
                "recon_matrix_F",
                "recon_matrix_P",
                "recon_voxel_F",
                "recon_voxel_P",
                "partial_Fourier_factor",
                "radial_sampling_factor",
            ],
            CardTab::Geometry => &[
                "slice_orientation",
                "slice_pos_LR",
                "slice_pos_AP",
                "slice_pos_HF",
                "slice_angle_1",
                "slice_angle_2",
                "num_slices",
                "slice_gap",
                "slice_thickness",
                "frequency_direction",
                "FOV_F",
                "FOV_P",
                "phase_oversampling",
                "show_FOV",
            ],
            CardTab::Sequence => &["sequence_type", "trajectory", "pixel_bandwidth", "FOV_bandwidth", "FW_shift", "turbo_factor", "EPI_factor", "echo_order"],
            CardTab::Recon => &["image_mode", "homodyne", "do_apodize", "apodization_alpha", "do_zerofill"],
        }
    }
}

pub struct SpinSightApp {
    ctrl: Controller,
    worker: Worker,
    next_job_id: u64,
    submitted_key: Option<Inputs>,
    renew_noise: bool,
    result: Option<JobResult>,
    image_tex: Option<(TextureHandle, (u64, String))>,
    kspace_tex: Option<(TextureHandle, (u64, String, u64, bool))>,
    signal_cache: Option<((u64, u64, u64), Vec<Waveform>)>,
    k_traj: Option<(u64, KTrajectory)>,
    hover_k_line: Option<Vec<[f64; 2]>>,
    hover_key: Option<(u64, u64)>,
    spans: Spans,
    show_help: bool,
    show_image: bool,
    show_kspace: bool,
    show_sequence: bool,
    show_browser: bool,
    show_planning: bool,
    planning: planning::Planning,
    exam: exam_ui::ExamState,
    /// slider ranges (values that adjust other parameters) and conflict confirmation
    ranges: conflicts::RangeAnalyzer,
    range_names: Vec<&'static str>,
    confirm_conflicts: bool,
    show_ranges: bool,
    pending_conflict: Option<conflicts::PendingConflict>,
    /// slider value that would adjust other parameters, held (not applied) while the slider is
    /// dragged; the conflict dialog follows when it is released there
    slider_hold: Option<(&'static str, Value)>,
    confirm_reset: bool,
    show_field_map: bool,
    field_tex: Option<(TextureHandle, f64, u64)>,
    tab: CardTab,
    sliders_with_text: bool,
    text_buf: HashMap<&'static str, String>,
    settings_file: Option<PathBuf>,
    status: Option<String>,
    start_time: String,
    reset_plots: bool,
    maker: Option<maker::MakerWindow>,
    svg_import: Option<maker::SvgImportDialog>,
    cache_info: Option<((u64, usize), Instant)>,
    confirm_delete: Option<String>,
}

impl SpinSightApp {
    pub fn new(cc: &eframe::CreationContext, opts: Options) -> Result<Self, String> {
        let ctx = cc.egui_ctx.clone();
        let worker = Worker::spawn(move || ctx.request_repaint());
        let ctx = cc.egui_ctx.clone();
        let ranges = conflicts::RangeAnalyzer::spawn(move || ctx.request_repaint());
        let mut ctrl = Controller::new()?;
        ctrl.reset_to_standard()?;
        let mut status = None;
        if let Some(f) = &opts.settings_file {
            if f.is_file() {
                if let Err(e) = ctrl.load_settings_file(f) {
                    status = Some(e);
                }
            }
        }
        for (k, v) in &opts.sets {
            match crate::params::try_param(k) {
                Some(spec) => ctrl.on_change(spec.name, v.clone()),
                None => status = Some(format!("unknown parameter {k}")),
            }
        }
        match opts.dark_mode {
            Some(true) => cc.egui_ctx.set_theme(egui::Theme::Dark),
            Some(false) => cc.egui_ctx.set_theme(egui::Theme::Light),
            None => cc.egui_ctx.set_theme(egui::ThemePreference::System),
        }
        let mut app = SpinSightApp {
            ctrl,
            worker,
            next_job_id: 0,
            submitted_key: None,
            renew_noise: false,
            result: None,
            image_tex: None,
            kspace_tex: None,
            signal_cache: None,
            k_traj: None,
            hover_k_line: None,
            hover_key: None,
            spans: Spans::default(),
            show_help: opts.help_mode,
            show_image: true,
            show_kspace: opts.show_kspace,
            show_sequence: opts.show_sequence,
            show_browser: true,
            show_planning: true,
            planning: planning::Planning::default(),
            exam: exam_ui::ExamState::default(),
            ranges,
            range_names: Vec::new(),
            confirm_conflicts: true,
            show_ranges: true,
            pending_conflict: None,
            slider_hold: None,
            confirm_reset: false,
            show_field_map: false,
            field_tex: None,
            tab: CardTab::Routine,
            sliders_with_text: opts.sliders_with_text,
            text_buf: HashMap::new(),
            settings_file: opts.settings_file,
            status,
            start_time: opts.start_time,
            reset_plots: false,
            maker: None,
            svg_import: None,
            cache_info: None,
            confirm_delete: None,
        };
        if let Some(img) = opts.phantom_maker {
            app.open_maker(img);
        }
        app.submit_if_needed();
        Ok(app)
    }

    /// Start a new simulation if anything influencing the simulated data changed
    fn submit_if_needed(&mut self) {
        let key = simulation_key(&self.ctrl.inputs);
        if self.submitted_key.as_ref() == Some(&key) && !self.renew_noise {
            return;
        }
        self.next_job_id += 1;
        self.worker.submit(Job {
            id: self.next_job_id,
            inputs: self.ctrl.inputs.clone(),
            protocol: self.ctrl.protocol.clone(),
            phantom: self.ctrl.phantom.clone(),
            renew_noise: self.renew_noise,
        });
        self.renew_noise = false;
        self.submitted_key = Some(key);
    }

    fn apply_changes(&mut self, changes: Vec<(&'static str, Value)>) {
        if self.pending_conflict.is_some() {
            return; // a change waits for confirmation
        }
        let object_before = self.ctrl.inputs.object.clone();
        for (name, value) in changes {
            if !UNCHECKED_CHANGES.contains(&name) {
                let adjustments = self.ctrl.preview(name, &value);
                if !adjustments.is_empty() && !self.confirm_conflicts {
                    self.ctrl.change_with_adjustments(name, value, &adjustments);
                    continue;
                }
                if !adjustments.is_empty() {
                    self.pending_conflict = Some(self.conflict_for(name, value, adjustments));
                    break;
                }
            }
            self.ctrl.on_change(name, value);
        }
        self.after_controller_change(&object_before);
    }

    /// Description of a change of `name` to `value` that adjusts other parameters
    fn conflict_for(&self, name: &'static str, value: Value, adjustments: Vec<crate::controller::Adjustment>) -> conflicts::PendingConflict {
        let st = self.ctrl.state(name);
        let on_off = |b: bool| if b { "on".to_string() } else { "off".to_string() };
        let label_of = |v: &Value| match v {
            Value::Bool(b) => on_off(*b),
            Value::Num(x) => {
                st.spec.options.iter().flatten().chain(st.options.iter()).find(|o| &o.value == v).map_or_else(|| format!("{x}"), |o| o.label.clone())
            }
            Value::Str(t) => st.options.iter().find(|o| &o.value == v).map_or_else(|| t.clone(), |o| o.label.clone()),
        };
        let old = match &st.value {
            Value::Bool(b) => on_off(*b),
            _ => st.value_label(),
        };
        let new = label_of(&value);
        conflicts::PendingConflict { name, label: st.label.trim_end_matches(" [°]").to_string(), old, value, new, adjustments }
    }

    /// A held slider value (see `slider_hold`) once the mouse button is released: ask
    pub(crate) fn release_slider_hold(&mut self) {
        if let Some((name, value)) = self.slider_hold.take() {
            let adjustments = self.ctrl.preview(name, &value);
            if adjustments.is_empty() {
                self.apply_changes(vec![(name, value)]);
            } else if self.pending_conflict.is_none() {
                self.pending_conflict = Some(self.conflict_for(name, value, adjustments));
            }
        }
    }

    /// Reset all parameters to the standard protocol (the phantom and the examination are kept)
    fn reset_all(&mut self) {
        if let Err(e) = self.ctrl.reset_to_standard() {
            self.status = Some(e);
            return;
        }
        self.ranges.clear();
        self.planning = planning::Planning::default();
        self.exam.review = None;
        self.text_buf.clear();
        self.pending_conflict = None;
        self.submitted_key = None;
        self.reset_plots = true;
        self.status = Some("All parameters were reset to the standard protocol (T1w SE)".into());
        self.submit_if_needed();
    }

    fn reset_dialog(&mut self, ctx: &egui::Context) {
        if !self.confirm_reset {
            return;
        }
        let mut close = false;
        egui::Modal::new(egui::Id::unique("confirm_reset")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading("Reset to standard");
            ui.label("Reset all parameters to the standard protocol, a T1-weighted spin echo (TR 500 ms, minimum TE)? Field of view, slice position, noise and display settings return to their defaults as well. The phantom and the acquired series are kept.");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Reset").strong()).clicked() {
                    self.reset_all();
                    close = true;
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            });
        });
        if close {
            self.confirm_reset = false;
        }
    }

    /// The confirmation dialog of a parameter conflict
    fn conflict_dialog(&mut self, ctx: &egui::Context) {
        let Some(pending) = &self.pending_conflict else { return };
        let mut warn = self.confirm_conflicts;
        let decision = conflicts::dialog(ctx, pending, &mut warn);
        self.confirm_conflicts = warn;
        match decision {
            Some(conflicts::Decision::Accept) => {
                let p = self.pending_conflict.take().unwrap();
                let before = self.ctrl.inputs.object.clone();
                self.ctrl.change_with_adjustments(p.name, p.value, &p.adjustments);
                self.after_controller_change(&before);
            }
            Some(conflicts::Decision::Cancel) => {
                // the parameter keeps its value (the slider returns to it)
                self.pending_conflict = None;
            }
            None => {}
        }
    }

    fn after_controller_change(&mut self, object_before: &str) {
        for m in self.ctrl.messages.drain(..) {
            eprintln!("{m}");
            self.status = Some(m);
        }
        if self.ctrl.inputs.object != object_before {
            self.reset_plots = true;
        }
        self.submit_if_needed();
    }

    /// Switch to a (new or changed) phantom
    fn use_phantom(&mut self, name: &str, reload: bool) {
        let before = self.ctrl.inputs.object.clone();
        self.ctrl.select_phantom(name, reload);
        if reload {
            self.submitted_key = None; // same parameters, new phantom content
        }
        self.reset_plots = true;
        self.after_controller_change(&before);
        self.cache_info = None;
    }

    // -------------------------------------------------------------------------------------------
    // parameter widgets

    fn param_widget(&mut self, ui: &mut Ui, name: &'static str, changes: &mut Vec<(&'static str, Value)>) {
        if !self.ctrl.state(name).visible() {
            return;
        }
        if self.ctrl.widget(name) == Widget::DiscreteSlider && !self.range_names.contains(&name) {
            self.range_names.push(name);
        }
        let st = self.ctrl.state(name);
        let range = self.ranges.range(&self.ctrl, name);
        let ctrl = &self.ctrl;
        let tip = |v: &Value| -> Option<String> {
            let adj = ctrl.preview(name, v);
            (!adj.is_empty()).then(|| conflicts::describe(&adj))
        };
        let style = RowStyle { help: self.show_help, with_text: self.sliders_with_text, width: PARAM_WIDTH, show_ranges: self.show_ranges };
        // value shown but not applied: held while dragging, or waiting for confirmation
        let held = self
            .slider_hold
            .as_ref()
            .filter(|(n, _)| *n == name)
            .map(|(_, v)| v.clone())
            .or_else(|| self.pending_conflict.as_ref().filter(|p| p.name == name).map(|p| p.value.clone()));
        let row = param_row(ui, st, &style, &mut self.text_buf, range, held.as_ref(), &tip);
        let (change, hovered) = (row.change, row.hovered);
        // sliders move freely; a value that would adjust other parameters is held (the image is
        // not updated) and only proposed in the conflict dialog when the slider is released there
        let change = match change {
            Some(v) if row.dragging && self.confirm_conflicts && !UNCHECKED_CHANGES.contains(&name) => {
                if self.ctrl.preview(name, &v).is_empty() {
                    self.slider_hold = None;
                    Some(v)
                } else {
                    self.slider_hold = Some((name, v));
                    None
                }
            }
            other => other,
        };
        if hovered {
            match name {
                "TE" => self.spans.te = true,
                "TR" => self.spans.tr = true,
                "TI" => self.spans.ti = true,
                _ => {}
            }
        }
        if let Some(v) = change {
            changes.push((name, v));
        }
    }

    fn compact_select(&mut self, ui: &mut Ui, name: &'static str, title: &str, width: f32, changes: &mut Vec<(&'static str, Value)>) {
        let st = self.ctrl.state(name);
        ui.label(RichText::new(title).color(Color32::GRAY));
        let current = st.value_label();
        let mut change = None;
        let r = egui::ComboBox::from_id_salt(("compact", name)).selected_text(RichText::new(current).strong()).width(width).show_ui(ui, |ui| {
            for o in &st.options {
                if ui.selectable_label(o.value == st.value, o.label.as_str()).clicked() && o.value != st.value {
                    change = Some(o.value.clone());
                }
            }
        });
        if self.show_help {
            if let Some(t) = tooltips::tooltip(name) {
                r.response.on_hover_ui(|ui| markdown::show(ui, t, 420.0));
            }
        }
        if let Some(v) = change {
            changes.push((name, v));
        }
    }

    // -------------------------------------------------------------------------------------------
    // display data derived from the latest simulation result

    fn update_textures(&mut self, ctx: &egui::Context) {
        let Some(res) = &self.result else { return };
        let ikey = (res.id, self.ctrl.inputs.image_mode.clone());
        if self.image_tex.as_ref().is_none_or(|(_, k)| *k != ikey) {
            let img = imaging::display_image(&res.output, &ikey.1);
            self.image_tex = plots::gray_texture(ctx, "mr_image", &img).map(|t| (t, ikey));
        }
        let i = &self.ctrl.inputs;
        let kkey = (res.id, i.kspace_mode.clone(), i.kspace_exponent.to_bits(), i.show_processed_kspace);
        if self.show_kspace && self.kspace_tex.as_ref().is_none_or(|(_, k)| *k != kkey) {
            let (k, _) = imaging::display_kspace(&res.output, &kkey.1, i.kspace_exponent, kkey.3);
            self.kspace_tex = plots::gray_texture(ctx, "kspace", &k).map(|t| (t, kkey));
        }
    }

    fn signal_curves(&mut self) -> Vec<Waveform> {
        let Some(res) = &self.result else { return vec![] };
        // the measured data matches the current protocol if the simulation key is unchanged
        let current = simulation_key(&self.ctrl.inputs) == simulation_key(&res.inputs);
        let p = if current { &self.ctrl.protocol } else { &res.protocol };
        let key = (res.id, if current { self.ctrl.generation } else { 0 }, self.ctrl.inputs.signal_exponent.to_bits());
        if let Some((k, v)) = &self.signal_cache {
            if *k == key {
                return v.clone();
            }
        }
        let curves = imaging::signal_curves(p, &res.output, self.ctrl.inputs.signal_exponent);
        self.signal_cache = Some((key, curves.clone()));
        curves
    }

    fn k_line_for(&mut self, h: &plots::Hovered) -> Option<Vec<[f64; 2]>> {
        let gen = self.ctrl.generation;
        if self.k_traj.as_ref().is_none_or(|(g, _)| *g != gen) {
            self.k_traj = Some((gen, protocol::k_trajectory(&self.ctrl.protocol)));
        }
        let tr = &self.k_traj.as_ref().unwrap().1;
        Some(protocol::k_on_interval(tr, h.t0, h.t1))
    }

    // -------------------------------------------------------------------------------------------
    // exam bar

    fn exam_bar(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        egui::Panel::top("exam_bar").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                if ui.selectable_label(self.show_browser, RichText::new("☰").size(18.0)).on_hover_text("Protocol and phantom browser").clicked() {
                    self.show_browser = !self.show_browser;
                }
                ui.label(RichText::new("SpinSight").strong().size(20.0));
                ui.separator();
                self.compact_select(ui, "object", "Phantom", 150.0, changes);
                ui.add_space(8.0);
                self.compact_select(ui, "field_strength", "B₀ [T]", 50.0, changes);
                ui.add_space(8.0);
                self.compact_select(ui, "parameter_style", "Parameter style", 230.0, changes);
                ui.add_space(8.0);
                let preset = self.ctrl.str("load_presets").to_string();
                ui.label(RichText::new("Protocol").color(Color32::GRAY));
                ui.label(RichText::new(if preset == CUSTOM_PRESET { "(custom)".to_string() } else { preset_title(&preset) }).strong());
                if ui
                    .button("↺ Reset")
                    .on_hover_text("Reset all parameters (protocol, FOV, slice position, noise) to the standard: a T1-weighted spin echo")
                    .clicked()
                {
                    self.confirm_reset = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dark = ui.visuals().dark_mode;
                    if ui.button(if dark { "☀" } else { "☾" }).on_hover_text("Toggle light/dark theme").clicked() {
                        ui.ctx().set_theme(if dark { egui::Theme::Light } else { egui::Theme::Dark });
                    }
                    if ui
                        .selectable_label(self.show_help, RichText::new("? Help").size(14.0))
                        .on_hover_text("Help mode: explanations of each parameter")
                        .clicked()
                    {
                        self.show_help = !self.show_help;
                    }
                    if self.worker.status.busy.load(Ordering::Relaxed) {
                        ui.add(egui::Spinner::new());
                    }
                });
            });
            ui.add_space(2.0);
        });
    }

    // -------------------------------------------------------------------------------------------
    // browser: protocols, phantoms, cache

    fn browser(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        egui::Panel::left("browser").resizable(true).default_size(250.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                self.exam_list(ui);
                section_title(ui, "PROTOCOLS");
                let current = self.ctrl.str("load_presets").to_string();
                let presets: Vec<String> = self.ctrl.state("load_presets").options.iter().map(|o| o.label.clone()).filter(|p| p != CUSTOM_PRESET).collect();
                egui::CollapsingHeader::new("SpinSight presets").default_open(true).show(ui, |ui| {
                    for p in presets {
                        let label = format!("▸ {}", preset_title(&p));
                        let r = ui.selectable_label(current == p, label);
                        if r.clicked() {
                            changes.push(("load_presets", Value::Str(p.clone())));
                        }
                        if self.show_help {
                            r.on_hover_text(format!("Load preset {p}"));
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Open…").on_hover_text("Load protocol (settings file, .toml)").clicked() {
                        self.open_protocol();
                    }
                    if ui.button("Save…").on_hover_text("Save protocol (settings file, .toml)").clicked() {
                        self.save_protocol();
                    }
                });

                ui.add_space(10.0);
                section_title(ui, "PHANTOMS");
                let current_object = self.ctrl.inputs.object.clone();
                let names: Vec<String> = self.ctrl.state("object").options.iter().map(|o| o.label.clone()).collect();
                for n in &names {
                    let builtin = phantom::is_builtin(n);
                    let label = if builtin { n.clone() } else { format!("{n}  ✎") };
                    let r = ui.selectable_label(*n == current_object, label);
                    if r.clicked() && *n != current_object {
                        changes.push(("object", Value::Str(n.clone())));
                    }
                    if !builtin {
                        r.on_hover_text("own phantom (imported or made with the phantom maker)");
                    }
                }
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Import…")
                        .on_hover_text("Import an SVG/TOML phantom, a PNG label image or an image to segment (or drop a file onto the window)")
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Phantoms and images", &["svg", "toml", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif", "webp"])
                            .pick_file()
                        {
                            self.import(&path);
                        }
                    }
                    if ui.button("Phantom maker…").on_hover_text("Make a phantom from an image or from drawn shapes").clicked() {
                        self.open_maker(None);
                    }
                    let deletable = !phantom::is_builtin(&current_object);
                    if ui.add_enabled(deletable, egui::Button::new("Delete")).on_hover_text("Delete the selected own phantom").clicked() {
                        self.confirm_delete = Some(current_object.clone());
                    }
                });
                if let Some(dir) = phantom::user_phantom_write_dir() {
                    if ui.small_button("Show phantom folder").clicked() {
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = open::that(&dir);
                    }
                }

                ui.add_space(10.0);
                section_title(ui, "K-SPACE CACHE");
                let refresh = self.cache_info.as_ref().is_none_or(|(_, t)| t.elapsed().as_secs_f32() > 2.0);
                if refresh {
                    self.cache_info = Some((phantom::cache_size(), Instant::now()));
                }
                let (bytes, files) = self.cache_info.as_ref().map(|c| c.0).unwrap_or((0, 0));
                ui.label(format!("{:.1} MB in {files} file{}", bytes as f64 / 1e6, if files == 1 { "" } else { "s" }));
                if let Some(dir) = phantom::cache_dir() {
                    ui.label(RichText::new(dir.display().to_string()).small().color(Color32::GRAY));
                    ui.horizontal(|ui| {
                        if ui.small_button("Show folder").clicked() {
                            let _ = std::fs::create_dir_all(&dir);
                            let _ = open::that(&dir);
                        }
                        if ui.small_button("Clear").on_hover_text("Delete the cached phantom k-space (recomputed when needed)").clicked() {
                            if let Err(e) = phantom::clear_cache() {
                                self.status = Some(e);
                            }
                            self.cache_info = None;
                        }
                    });
                }
                if let Some((name, secs, cached)) = self.worker.status.last_phantom_secs.lock().unwrap().clone() {
                    ui.label(
                        RichText::new(format!("{name}: k-space {} in {:.2} s", if cached { "loaded from cache" } else { "computed" }, secs))
                            .small()
                            .color(Color32::GRAY),
                    );
                }

                ui.add_space(10.0);
                self.noise_menu(ui, changes);
                ui.add_space(10.0);
                self.b0_menu(ui, changes);

                ui.add_space(10.0);
                section_title(ui, "OPTIONS");
                ui.checkbox(&mut self.sliders_with_text, "Text boxes next to sliders");
                ui.checkbox(&mut self.confirm_conflicts, "Warn before other parameters change")
                    .on_hover_text("Ask before a change that also adjusts other parameters (e.g. a longer TE that needs a longer TR)");
                ui.checkbox(&mut self.show_ranges, "Colour-coded slider ranges")
                    .on_hover_text("Green: values that change no other parameter; orange: values for which other parameters are adjusted");
            });
        });
    }

    /// Noise section of the browser: kind and strength of the noise
    fn noise_menu(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        section_title(ui, "NOISE");
        let help = self.show_help;
        let tip = |r: egui::Response, name: &str| {
            if help {
                if let Some(t) = tooltips::tooltip(name) {
                    r.on_hover_ui(|ui| markdown::show(ui, t, 380.0));
                }
            }
        };
        let current = self.ctrl.inputs.noise_type.clone();
        ui.label(RichText::new("Type").color(Color32::GRAY));
        let r = egui::ComboBox::from_id_salt("noise_type").selected_text(current.as_str()).width(ui.available_width() - 8.0).show_ui(ui, |ui| {
            for &t in crate::params::NOISE_TYPES {
                if ui.selectable_label(current == t, t).clicked() && current != t {
                    changes.push(("noise_type", Value::s(t)));
                }
            }
        });
        tip(r.response, "noise_type");
        ui.spacing_mut().slider_width = (ui.available_width() - 70.0).max(80.0);
        let mut level = self.ctrl.inputs.noise_gain / crate::params::STANDARD_NOISE_GAIN * 100.0;
        ui.label(RichText::new("Thermal noise level").color(Color32::GRAY));
        let r = ui.add(egui::Slider::new(&mut level, 0.0..=400.0).step_by(5.0).suffix(" %"));
        if r.changed() {
            changes.push(("noise_gain", Value::Num(level / 100.0 * crate::params::STANDARD_NOISE_GAIN)));
        }
        tip(r, "noise_gain");
        if current != crate::params::NOISE_TYPES[0] {
            let mut strength = self.ctrl.inputs.interference_level * 100.0;
            let what = match (current.contains("spikes"), current.contains("RF")) {
                (true, true) => "Spike / RF strength",
                (true, false) => "Spike strength",
                _ => "RF interference strength",
            };
            ui.label(RichText::new(what).color(Color32::GRAY));
            let r = ui.add(egui::Slider::new(&mut strength, 0.0..=400.0).step_by(5.0).suffix(" %"));
            if r.changed() {
                changes.push(("interference_level", Value::Num(strength / 100.0)));
            }
            tip(r, "interference_level");
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("New realisation").on_hover_text("Draw new random noise (and new spike positions / interference frequency)").clicked() {
                self.renew_noise = true;
                self.submit_if_needed();
            }
            let standard = current == crate::params::NOISE_TYPES[0]
                && self.ctrl.inputs.noise_gain == crate::params::STANDARD_NOISE_GAIN
                && self.ctrl.inputs.interference_level == 1.0;
            if ui.add_enabled(!standard, egui::Button::new("Standard")).on_hover_text("Thermal noise at 100 %").clicked() {
                changes.push(("noise_type", Value::s(crate::params::NOISE_TYPES[0])));
                changes.push(("noise_gain", Value::Num(crate::params::STANDARD_NOISE_GAIN)));
                changes.push(("interference_level", Value::Num(1.0)));
            }
        });
    }

    /// B0 inhomogeneity section of the browser
    fn b0_menu(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        section_title(ui, "B₀ FIELD");
        let help = self.show_help;
        let tip = |r: egui::Response, name: &str| {
            if help {
                if let Some(t) = tooltips::tooltip(name) {
                    r.on_hover_ui(|ui| markdown::show(ui, t, 380.0));
                }
            }
        };
        let mut on = self.ctrl.inputs.b0_inhomogeneity;
        let r = ui.checkbox(&mut on, "Simulate B0 inhomogeneities");
        if r.changed() {
            changes.push(("b0_inhomogeneity", Value::Bool(on)));
        }
        tip(r, "b0_inhomogeneity");
        if !on {
            return;
        }
        ui.spacing_mut().slider_width = (ui.available_width() - 70.0).max(80.0);
        let mut strength = self.ctrl.inputs.b0_strength * 100.0;
        ui.label(RichText::new("Strength").color(Color32::GRAY));
        let r = ui.add(egui::Slider::new(&mut strength, 0.0..=300.0).step_by(5.0).suffix(" %"));
        if r.changed() {
            changes.push(("b0_strength", Value::Num(strength / 100.0)));
        }
        tip(r, "b0_strength");
        let mut shim = self.ctrl.inputs.b0_shim;
        let r = ui.checkbox(&mut shim, "Shimmed (2nd order)");
        if r.changed() {
            changes.push(("b0_shim", Value::Bool(shim)));
        }
        tip(r, "b0_shim");
        ui.checkbox(&mut self.show_field_map, "Show field map").on_hover_text("Show the off-resonance [Hz] instead of the MR image");
        let source = if self.ctrl.phantom.is_volume_slice() {
            "Field of the air in the paranasal sinuses, mastoid cells and ear canals"
        } else {
            "Generic field (air cavity in front of the object)"
        };
        ui.label(RichText::new(source).small().color(Color32::GRAY));
    }

    fn open_protocol(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("SpinSight protocol", &["toml"]);
        if let Some(dir) = self.settings_file.as_ref().and_then(|f| f.parent()) {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.pick_file() {
            match self.ctrl.load_settings_file(&path) {
                Ok(()) => self.status = Some(format!("Loaded protocol {}", path.display())),
                Err(e) => self.status = Some(e),
            }
            self.settings_file = Some(path);
            self.submit_if_needed();
        }
    }

    fn save_protocol(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("SpinSight protocol", &["toml"]).set_file_name("protocol.toml");
        if let Some(dir) = self.settings_file.as_ref().and_then(|f| f.parent()) {
            dlg = dlg.set_directory(dir);
        }
        if let Some(path) = dlg.save_file() {
            match self.ctrl.save_settings_file(&path) {
                Ok(()) => self.status = Some(format!("Saved protocol {}", path.display())),
                Err(e) => self.status = Some(e),
            }
            self.settings_file = Some(path);
        }
    }

    fn open_maker(&mut self, image: Option<PathBuf>) {
        let mut m = self.maker.take().unwrap_or_default();
        m.open = true;
        if let Some(p) = image {
            m.load_image(&p);
        }
        self.maker = Some(m);
    }

    fn import(&mut self, path: &std::path::Path) {
        match phantom_maker::import_file(path) {
            Ok(phantom_maker::Import::Done(name)) => {
                self.status = Some(format!("Imported phantom \"{name}\""));
                self.use_phantom(&name, true);
            }
            Ok(phantom_maker::Import::NeedsColormap { name, svg, colors }) => {
                self.svg_import = Some(maker::SvgImportDialog::new(name, svg, colors));
            }
            Ok(phantom_maker::Import::Image(p)) => self.open_maker(Some(p)),
            Err(e) => self.status = Some(e),
        }
    }

    // -------------------------------------------------------------------------------------------
    // parameter card and status line

    fn parameter_card(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        egui::Panel::bottom("parameter_card").resizable(true).default_size(300.0).min_size(160.0).show(ui, |ui| {
            ui.add_space(4.0);
            self.status_line(ui, changes);
            ui.separator();
            ui.horizontal(|ui| {
                for t in CardTab::ALL {
                    let text = RichText::new(t.label()).size(15.0);
                    let text = if self.tab == t { text.strong() } else { text };
                    if ui.selectable_label(self.tab == t, text).clicked() {
                        self.tab = t;
                    }
                }
                if self.show_ranges {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let legend = |ui: &mut Ui, color: Color32, text: &str| {
                            ui.label(RichText::new(text).small().color(Color32::GRAY));
                            let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 8.0), egui::Sense::hover());
                            ui.painter().rect_filled(r, 4.0, color);
                        };
                        let (free, adjust) = conflicts::rail_colors(ui.visuals().dark_mode);
                        legend(ui, adjust, "adjusts other parameters");
                        legend(ui, free, "no other changes");
                    });
                }
            });
            ui.separator();
            self.range_names.clear();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let names: Vec<&'static str> = self.tab.params().iter().copied().filter(|n| self.ctrl.state(n).visible()).collect();
                let col_w = PARAM_WIDTH + if self.show_help { 52.0 } else { 30.0 };
                let ncols = ((ui.available_width() / col_w).floor() as usize).clamp(1, names.len().max(1));
                let nrows = names.len().div_ceil(ncols);
                egui::Grid::new(("card", self.tab.label())).num_columns(ncols).min_col_width(col_w).spacing([4.0, 6.0]).show(ui, |ui| {
                    for r in 0..nrows {
                        for c in 0..ncols {
                            // fill column by column, like a scanner card
                            match names.get(c * nrows + r) {
                                Some(&name) => self.param_widget(ui, name, changes),
                                None => {
                                    ui.label("");
                                }
                            }
                        }
                        ui.end_row();
                    }
                });
            });
            let names = std::mem::take(&mut self.range_names);
            self.ranges.request(&self.ctrl, names);
        });
    }

    fn status_line(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        let p = self.ctrl.protocol.clone();
        let info = self.ctrl.info.clone();
        let voxel = format!(
            "{} × {} × {} mm",
            crate::formatting::format_float(self.ctrl.inputs.fov_f / p.matrix_f as f64, 3),
            crate::formatting::format_float(self.ctrl.inputs.fov_p / p.matrix_p as f64, 3),
            crate::formatting::format_float(self.ctrl.inputs.slice_thickness, 2)
        );
        ui.horizontal_wrapped(|ui| {
            self.scan_controls(ui);
            status_item(ui, "TA", &crate::formatting::ta(p.scantime), true);
            if self.ctrl.phantom.is_volume_slice() {
                status_item(ui, "Slices", &self.ctrl.inputs.num_slices.to_string(), false);
            }
            if p.concatenations > 1 {
                status_item(ui, "Concat.", &p.concatenations.to_string(), false)
            }
            status_item(ui, "Voxel", &voxel, false);
            status_item(ui, "Rel. SNR", &info.relative_snr, true);
            if ui.small_button("Set ref.").on_hover_text("Use the current SNR as reference (100 %)").clicked() {
                self.ctrl.set_reference_snr();
            }
            ui.separator();
            self.compact_select(ui, "reference_tissue", "Ref. tissue", 140.0, changes);
            ui.separator();
            status_item(ui, "BW", &info.pixel_bandwidth, false);
            status_item(ui, "Fat/water shift", &info.fw_shift, false);
            status_item(ui, &format!("# {}s", self.ctrl.shot_label), &info.num_shots, false);
            if let Some(t) = self.result.as_ref().and_then(|r| r.output.timings.iter().find(|t| t.0 == "total").map(|t| t.1)) {
                ui.label(RichText::new(format!("sim. {:.0} ms", t * 1e3)).small().color(Color32::GRAY));
            }
        });
    }

    // -------------------------------------------------------------------------------------------
    // viewing segments

    fn viewer(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        let ctx = ui.ctx().clone();
        self.update_textures(&ctx);
        let reset = std::mem::take(&mut self.reset_plots);
        // toolbar
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("View").color(Color32::GRAY));
            let is_3d = self.ctrl.phantom.is_volume_slice();
            if is_3d {
                ui.toggle_value(&mut self.show_planning, "Planning").on_hover_text("Localizers (transversal, sagittal, coronal) with the planned slices");
            }
            ui.toggle_value(&mut self.show_image, "Image");
            ui.toggle_value(&mut self.show_kspace, "k-space");
            ui.toggle_value(&mut self.show_sequence, "Sequence");
            ui.separator();
            if self.show_image {
                self.compact_select(ui, "image_mode", "Image", 100.0, changes);
                let mut fov = self.ctrl.bool("show_FOV");
                if ui.checkbox(&mut fov, "FOV").on_hover_text("Show the prescribed (yellow) and acquired (light blue) FOV").changed() {
                    changes.push(("show_FOV", Value::Bool(fov)));
                }
                ui.separator();
            }
            if self.show_kspace {
                self.compact_select(ui, "kspace_mode", "k-space", 100.0, changes);
                if self.ctrl.state("kspace_exponent").visible() {
                    let mut e = self.ctrl.num("kspace_exponent");
                    ui.label(RichText::new("exponent").color(Color32::GRAY));
                    if ui.add(egui::Slider::new(&mut e, 0.1..=1.0).step_by(0.01).fixed_decimals(2)).changed() {
                        changes.push(("kspace_exponent", Value::Num(e)));
                    }
                }
                let mut pk = self.ctrl.bool("show_processed_kspace");
                if ui.checkbox(&mut pk, "processed").on_hover_text("Show the processed (zero-filled, apodized) k-space").changed() {
                    changes.push(("show_processed_kspace", Value::Bool(pk)));
                }
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let n_img = usize::from(self.show_image) + usize::from(self.show_kspace);
            let planning = self.ctrl.phantom.is_volume_slice() && self.show_planning;
            if planning {
                ui.horizontal(|ui| {
                    if ui.button("Localizers → slice").on_hover_text("Show the centre of the displayed slice in the localizers").clicked() {
                        let g = self.ctrl.inputs.displayed_slice_geometry();
                        self.planning.center_on(&g);
                    }
                    ui.label(
                        RichText::new("Drag the slices to move them, the ⟲ handle to angulate; click to navigate, wheel to scroll.")
                            .small()
                            .color(Color32::GRAY),
                    );
                    self.param_widget(ui, "displayed_slice", changes);
                });
            }
            let n_cells = n_img + if planning { 3 } else { 0 };
            if n_cells > 0 {
                let avail_w = ui.available_width();
                let avail_h = ui.available_height();
                let height_share = if self.show_sequence { 0.62 } else { 1.0 };
                let min = if planning { 220.0 } else { MIN_IMAGE_SIZE };
                let size = ((avail_w - 22.0 * n_cells as f32) / n_cells as f32).min(avail_h * height_share - 60.0).clamp(min, 760.0);
                ui.horizontal_wrapped(|ui| {
                    if planning {
                        if let Some(vol) = self.ctrl.phantom.volume.clone() {
                            let fov = self.ctrl.protocol.fov;
                            let inputs = self.ctrl.inputs.clone();
                            let (series, highlight) = self.exam.review_geometries();
                            let c = self.planning.show(ui, &vol, &inputs, fov, &series, highlight, size, self.show_help);
                            changes.extend(c);
                        }
                    }
                    let field_map = self.result.as_ref().and_then(|r| r.output.field_map.as_ref().map(|m| (r.id, m.clone())));
                    let field_map = field_map.filter(|_| self.show_image && self.exam.reviewed().is_none() && self.show_field_map);
                    if let Some((id, map)) = field_map {
                        if self.field_tex.as_ref().is_none_or(|t| t.2 != id) {
                            self.field_tex = plots::field_texture(&ctx, "field_map", &map).map(|(t, s)| (t, s, id));
                        }
                        ui.vertical(|ui| {
                            let (lo, hi) =
                                map.data.iter().filter(|v| v.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                            let scale = self.field_tex.as_ref().map_or(0.0, |t| t.1);
                            ui.label(RichText::new(format!("B₀ field map {lo:.0} … {hi:.0} Hz")).strong())
                                .on_hover_text(format!("Off-resonance in the slice: blue below, grey on, red above resonance; full colour at ±{scale:.0} Hz"));
                            let fov = self.result.as_ref().map(|r| r.output.fov).unwrap_or(self.ctrl.protocol.fov);
                            let letters = self.ctrl.phantom.slice.map(|g| crate::geometry::edge_letters(g.u, g.v));
                            plots::image_plot(ui, self.field_tex.as_ref().map(|t| &t.0), fov, None, letters.as_ref(), size, reset);
                        });
                    } else if self.show_image && self.exam.reviewed().is_some() {
                        ui.vertical(|ui| self.review_image(ui, size, reset));
                    } else if self.show_image {
                        ui.vertical(|ui| {
                            let fov = self.result.as_ref().map(|r| r.output.fov).unwrap_or(self.ctrl.protocol.fov);
                            let fov_box = self.ctrl.inputs.show_fov.then(|| plots::fov_box(&self.ctrl.protocol));
                            if self.result.is_none() {
                                placeholder(ui, "Preparing image", size);
                            } else {
                                let letters = self.ctrl.phantom.slice.map(|g| crate::geometry::edge_letters(g.u, g.v));
                                if planning {
                                    ui.label(RichText::new("MR image").strong());
                                }
                                plots::image_plot(ui, self.image_tex.as_ref().map(|t| &t.0), fov, fov_box.as_ref(), letters.as_ref(), size, reset);
                            }
                        });
                    }
                    if self.show_kspace {
                        ui.vertical(|ui| match &self.result {
                            None => placeholder(ui, "Preparing k-space", size),
                            Some(res) => {
                                let axes = if self.ctrl.inputs.show_processed_kspace { &res.output.processed_axes } else { &res.output.grid_axes };
                                let axes = axes.clone();
                                plots::kspace_plot(ui, self.kspace_tex.as_ref().map(|t| &t.0), &axes, self.hover_k_line.as_deref(), size, reset);
                            }
                        });
                    }
                });
            }
            if self.show_sequence {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    self.param_widget(ui, "shot", changes);
                    ui.add_space(12.0);
                    let info = self.ctrl.info.clone();
                    if info.spoke_angle_visible {
                        status_item(ui, "Angle", &info.spoke_angle, false);
                    }
                    status_item(ui, &format!("# {}s", self.ctrl.shot_label), &info.num_shots, false);
                    ui.add_space(12.0);
                    self.param_widget(ui, "signal_exponent", changes);
                });
                let curves = self.signal_curves();
                let mut hovered = None;
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        hovered = plots::sequence_plot(ui, &self.ctrl.protocol, &curves, self.spans, reset);
                    });
                });
                let hkey = hovered.as_ref().map(|h| (h.t0.to_bits(), h.t1.to_bits()));
                if hkey != self.hover_key {
                    self.hover_k_line = hovered.as_ref().and_then(|h| self.k_line_for(h));
                    self.hover_key = hkey;
                    ui.ctx().request_repaint();
                }
            } else if self.hover_k_line.is_some() {
                self.hover_k_line = None;
                self.hover_key = None;
            }
            if n_img == 0 && !self.show_sequence {
                ui.add_space(40.0);
                ui.label(RichText::new("Select Image, k-space or Sequence above.").color(Color32::GRAY));
            }
            ui.add_space(12.0);
            let gray = Color32::GRAY;
            ui.label(
                RichText::new(format!(
                    "SpinSight v {} – original Python version by Johan Berglund, Ph.D. (johan.berglund@akademiska.se); Rust port. Started {}.",
                    env!("CARGO_PKG_VERSION"),
                    self.start_time
                ))
                .small()
                .italics()
                .color(gray),
            );
            if let Some(s) = &self.status {
                ui.label(RichText::new(s).color(gray));
            }
        });
    }

    // -------------------------------------------------------------------------------------------
    // dialogs

    fn dialogs(&mut self, ctx: &egui::Context) {
        let preparing = self.worker.status.preparing_phantom.lock().unwrap().clone();
        if let Some(name) = preparing {
            let done = self.worker.status.progress.done.load(Ordering::Relaxed);
            let total = self.worker.status.progress.total.load(Ordering::Relaxed).max(1);
            egui::Modal::new(egui::Id::unique("progress")).show(ctx, |ui| {
                ui.set_width(460.0);
                ui.heading(format!("Preparing k-space for \"{name}\" phantom…"));
                ui.label(RichText::new("(first-time loading; the result is cached)").italics());
                ui.add_space(8.0);
                ui.add(egui::ProgressBar::new(done as f32 / total as f32).text(format!("{done} / {total} shapes")).animate(true));
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if let Some(name) = self.confirm_delete.clone() {
            let mut close = false;
            egui::Modal::new(egui::Id::unique("confirm_delete")).show(ctx, |ui| {
                ui.heading(format!("Delete phantom \"{name}\"?"));
                ui.label("Its files are removed from the phantom folder.");
                ui.horizontal(|ui| {
                    if ui.button("Delete").clicked() {
                        match phantom_maker::delete_user_phantom(&name) {
                            Ok(()) => {
                                self.status = Some(format!("Deleted phantom \"{name}\""));
                                self.use_phantom("brain", false);
                            }
                            Err(e) => self.status = Some(e),
                        }
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
            if close {
                self.confirm_delete = None;
            }
        }
        if let Some(dlg) = &mut self.svg_import {
            match dlg.show(ctx) {
                maker::DialogResult::Open => {}
                maker::DialogResult::Cancel => self.svg_import = None,
                maker::DialogResult::Saved(name) => {
                    self.svg_import = None;
                    self.status = Some(format!("Imported phantom \"{name}\""));
                    self.use_phantom(&name, true);
                }
            }
        }
        let mut saved = None;
        if let Some(m) = &mut self.maker {
            if m.open {
                saved = m.show(ctx);
            }
        }
        if let Some((name, use_it)) = saved {
            self.status = Some(format!("Saved phantom \"{name}\""));
            self.cache_info = None;
            if use_it || self.ctrl.inputs.object == name {
                self.use_phantom(&name, true);
            } else {
                self.ctrl.refresh_phantoms();
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        if let Some(path) = dropped.first() {
            self.import(path);
        }
    }
}

fn section_title(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).small().strong().color(Color32::from_rgb(120, 160, 220)));
}

fn preset_title(p: &str) -> String {
    match p {
        "T2w-TSE" => "T2w TSE".into(),
        "T1w-TSE" => "T1w TSE".into(),
        "T1w-GRE" => "T1w GRE".into(),
        "T1w-SE" => "T1w SE".into(),
        "T2-FLAIR" => "T2 FLAIR".into(),
        "GRE-EPI" => "GRE EPI".into(),
        other => other.to_string(),
    }
}

fn status_item(ui: &mut Ui, title: &str, value: &str, emphasize: bool) {
    ui.label(RichText::new(title).color(Color32::GRAY));
    let v = RichText::new(value).size(15.0);
    ui.label(if emphasize { v.strong() } else { v });
    ui.add_space(10.0);
}

fn placeholder(ui: &mut Ui, text: &str, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(egui::Layout::top_down(egui::Align::Center)));
    child.add_space(size / 2.0 - 30.0);
    child.add(egui::Spinner::new().size(40.0));
    child.label(text);
}

fn help_icon(ui: &mut Ui, name: &str) -> bool {
    let Some(text) = tooltips::tooltip(name) else {
        ui.add_space(18.0);
        return false;
    };
    let (rect, r) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
    let color = if r.hovered() { Color32::from_rgb(140, 185, 255) } else { Color32::from_rgb(100, 150, 230) };
    ui.painter().circle_stroke(rect.center(), 7.0, egui::Stroke::new(1.5, color));
    ui.painter().text(rect.center() + egui::vec2(0.0, 0.5), egui::Align2::CENTER_CENTER, "?", egui::FontId::proportional(11.0), color);
    let hovered = r.hovered();
    r.on_hover_ui(|ui| {
        egui::Frame::new().inner_margin(egui::Margin::symmetric(15, 4)).show(ui, |ui| markdown::show(ui, text, 420.0));
    });
    hovered
}

/// Display options of parameter widgets
struct RowStyle {
    help: bool,
    with_text: bool,
    width: f32,
    show_ranges: bool,
}

/// Result of a parameter widget
struct RowResult {
    /// new value if changed
    change: Option<Value>,
    hovered: bool,
    /// a slider is being dragged
    dragging: bool,
}

/// One parameter widget. `range`: the selectable values of a discrete slider with the
/// information which of them adjust other parameters; `held`: a slider value shown but not yet
/// applied; `tip` describes the adjustments of a value.
fn param_row(
    ui: &mut Ui,
    st: &ParamState,
    style: &RowStyle,
    text_buf: &mut HashMap<&'static str, String>,
    range: Option<&SliderRange>,
    held: Option<&Value>,
    tip: &dyn Fn(&Value) -> Option<String>,
) -> RowResult {
    let (help, with_text, width) = (style.help, style.with_text, style.width);
    let name = st.spec.name;
    let mut change = None;
    let mut dragging = false;
    let mut help_hovered = false;
    let resp = ui.horizontal(|ui| {
        if help {
            help_hovered = help_icon(ui, name);
        }
        ui.vertical(|ui| match st.spec.widget {
            Widget::Checkbox => {
                let mut v = st.value.bool();
                if ui.checkbox(&mut v, st.label.as_str()).changed() {
                    change = Some(Value::Bool(v));
                }
            }
            Widget::Select => {
                ui.label(RichText::new(st.label.as_str()).color(Color32::GRAY));
                let current = st.value_label();
                egui::ComboBox::from_id_salt(("select", name)).selected_text(current).width(width - 20.0).show_ui(ui, |ui| {
                    for o in &st.options {
                        if ui.selectable_label(o.value == st.value, o.label.as_str()).clicked() && o.value != st.value {
                            change = Some(o.value.clone());
                        }
                    }
                });
            }
            Widget::FloatSlider => {
                let (lo, hi) = st.spec.bounds.unwrap_or((0.0, 1.0));
                let mut v = st.value.num();
                let shown = match name {
                    "slice_pos_LR" => format!("{} mm", crate::geometry::position_label(0, v)),
                    "slice_pos_AP" => format!("{} mm", crate::geometry::position_label(1, v)),
                    "slice_pos_HF" => format!("{} mm", crate::geometry::position_label(2, v)),
                    "slice_angle_1" | "slice_angle_2" => format!("{v:.0}°"),
                    _ => format!("{v:.2}"),
                };
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}:", st.label.trim_end_matches(" [°]"))).color(Color32::GRAY));
                    ui.label(RichText::new(shown).strong());
                });
                ui.spacing_mut().slider_width = width - 20.0;
                let r = ui.add(egui::Slider::new(&mut v, lo..=hi).step_by(st.spec.step).show_value(false));
                if r.changed() {
                    change = Some(Value::Num(v));
                }
            }
            Widget::DiscreteSlider => {
                // the extended range (if known) includes values that adjust other parameters
                let opts: &[crate::params::Opt] = range.map_or(st.options.as_slice(), |r| r.options.as_slice());
                let shown = held.unwrap_or(&st.value);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}:", st.label)).color(Color32::GRAY));
                    match held.and_then(|h| opts.iter().find(|o| &o.value == h)) {
                        Some(o) => {
                            ui.label(RichText::new(format!("{} ⚠", o.label)).strong().color(conflicts::ADJUST_COLOR))
                                .on_hover_text("Adjusts other parameters: release to confirm");
                        }
                        None => {
                            ui.label(RichText::new(st.value_label()).strong());
                        }
                    }
                });
                let n = opts.len();
                ui.horizontal(|ui| {
                    let text_w = if with_text { 64.0 } else { 0.0 };
                    ui.spacing_mut().slider_width = width - 20.0 - text_w;
                    if n == 0 {
                        let mut dummy = 0usize;
                        ui.add_enabled(false, egui::Slider::new(&mut dummy, 0..=1).show_value(false));
                    } else {
                        let current = opts.iter().position(|o| &o.value == shown).or_else(|| {
                            let vals: Vec<f64> = opts.iter().map(|o| o.value.num()).collect();
                            crate::params::snap(shown.num(), &vals, crate::params::SnapMode::Nearest).and_then(|s| vals.iter().position(|&x| x == s))
                        });
                        let mut idx = current.unwrap_or(0);
                        let r = ui.add_enabled(n > 1, egui::Slider::new(&mut idx, 0..=n - 1).show_value(false));
                        dragging = r.dragged();
                        if r.changed() && &opts[idx].value != shown {
                            change = Some(opts[idx].value.clone());
                        }
                        if let Some(rg) = range {
                            if style.show_ranges {
                                conflicts::paint_rail(ui, &r, &rg.adjusts, idx);
                            }
                            if r.hovered() && !r.dragged() {
                                if let Some(k) = conflicts::hovered_index(ui, r.rect, n) {
                                    if rg.adjusts[k] {
                                        let what = tip(&opts[k].value).unwrap_or_else(|| "other parameters are adjusted".into());
                                        r.on_hover_text(format!("{}: {}", opts[k].label, what));
                                    }
                                }
                            }
                        }
                    }
                    if with_text {
                        let is_pct = st.options.first().is_some_and(|o| o.label.ends_with('%'));
                        let shown = if is_pct { crate::formatting::percentage(st.value.num()) } else { st.value.num().to_string() };
                        let buf = text_buf.entry(name).or_insert_with(|| shown.clone());
                        let r = ui.add(egui::TextEdit::singleline(buf).desired_width(text_w - 8.0).align(egui::Align2::CENTER_CENTER));
                        if r.lost_focus() {
                            let txt = buf.trim().trim_end_matches('%').to_string();
                            if let Ok(x) = txt.parse::<f64>() {
                                let x = if is_pct { x / 100.0 } else { x };
                                let vals: Vec<f64> = opts.iter().map(|o| o.value.num()).collect();
                                if let Some(s) = crate::params::snap(x, &vals, crate::params::SnapMode::Nearest) {
                                    if s != st.value.num() {
                                        change = Some(Value::Num(s));
                                    }
                                }
                            }
                            text_buf.remove(name);
                        } else if !r.has_focus() {
                            *buf = shown;
                        }
                    }
                });
            }
        });
    });
    let hovered = help_hovered || resp.response.contains_pointer();
    RowResult { change, hovered, dragging }
}

impl eframe::App for SpinSightApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.poll_scan();
        self.ranges.poll();
        if let Some(r) = self.worker.poll() {
            if r.id >= self.result.as_ref().map_or(0, |x| x.id) {
                self.result = Some(r);
            }
        }
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        let prev_spans = self.spans;
        self.spans = Spans::default();
        let mut changes = Vec::new();
        self.exam_bar(ui, &mut changes);
        if self.show_browser {
            self.browser(ui, &mut changes);
        }
        self.parameter_card(ui, &mut changes);
        egui::CentralPanel::default_margins().show(ui, |ui| {
            self.viewer(ui, &mut changes);
        });
        self.dialogs(&ctx);
        self.exam_dialogs(&ctx);
        self.conflict_dialog(&ctx);
        self.reset_dialog(&ctx);
        if self.spans.te != prev_spans.te || self.spans.tr != prev_spans.tr || self.spans.ti != prev_spans.ti {
            ctx.request_repaint();
        }
        if !changes.is_empty() {
            self.apply_changes(changes);
            ctx.request_repaint();
        }
        // a slider held in a range that adjusts other parameters was released
        if self.slider_hold.is_some() && !ctx.input(|i| i.pointer.any_down()) {
            self.release_slider_hold();
            ctx.request_repaint();
        }
        if self.worker.status.busy.load(Ordering::Relaxed) {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }
}
