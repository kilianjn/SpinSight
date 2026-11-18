//! The SpinSight desktop application (egui), mirroring the layout of the Python dashboard.

mod markdown;
pub mod plots;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use egui::{Color32, RichText, TextureHandle, Ui};

use crate::controller::{Controller, ParamState};
use crate::imaging::{self, simulation_key};
use crate::params::{Value, Widget};
use crate::protocol::{self, Inputs, KTrajectory};
use crate::sequence::Waveform;
use crate::tooltips;
use crate::worker::{Job, JobResult, Worker};
use plots::Spans;

const PARAM_WIDTH: f32 = 290.0;
const IMAGE_SIZE: f32 = 500.0;
const MIN_IMAGE_SIZE: f32 = 320.0;

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
    show_sequence: bool,
    show_kspace: bool,
    show_sidebar: bool,
    sliders_with_text: bool,
    text_buf: HashMap<&'static str, String>,
    settings_file: Option<PathBuf>,
    status: Option<String>,
    start_time: String,
    reset_plots: bool,
    dark_mode: Option<bool>,
}

impl SpinSightApp {
    pub fn new(cc: &eframe::CreationContext, opts: Options) -> Result<Self, String> {
        let ctx = cc.egui_ctx.clone();
        let worker = Worker::spawn(move || ctx.request_repaint());
        let mut ctrl = Controller::new()?;
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
            show_sequence: opts.show_sequence,
            show_kspace: opts.show_kspace,
            show_sidebar: false,
            sliders_with_text: opts.sliders_with_text,
            text_buf: HashMap::new(),
            settings_file: opts.settings_file,
            status,
            start_time: opts.start_time,
            reset_plots: false,
            dark_mode: opts.dark_mode,
        };
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
        let object_before = self.ctrl.inputs.object.clone();
        for (name, value) in changes {
            self.ctrl.on_change(name, value);
        }
        for m in self.ctrl.messages.drain(..) {
            eprintln!("{m}");
            self.status = Some(m);
        }
        if self.ctrl.inputs.object != object_before {
            self.reset_plots = true;
        }
        self.submit_if_needed();
    }

    // -------------------------------------------------------------------------------------------
    // parameter widgets

    fn param_widget(&mut self, ui: &mut Ui, name: &'static str, changes: &mut Vec<(&'static str, Value)>) {
        let st = self.ctrl.state(name);
        if !st.visible() {
            return;
        }
        let (change, hovered) = param_row(ui, st, self.show_help, self.sliders_with_text, &mut self.text_buf, PARAM_WIDTH);
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

    fn group_panel(&mut self, ui: &mut Ui, group: &str, changes: &mut Vec<(&'static str, Value)>) {
        ui.vertical(|ui| {
            ui.set_width(PARAM_WIDTH + 30.0);
            ui.label(RichText::new(group).strong().size(15.0));
            ui.add_space(2.0);
            for name in self.ctrl.group_params(group) {
                self.param_widget(ui, name, changes);
            }
        });
    }

    fn indicator(ui: &mut Ui, title: &str, value: &str) {
        ui.vertical(|ui| {
            ui.label(RichText::new(title).color(Color32::GRAY).size(13.0));
            ui.label(RichText::new(value).color(Color32::GRAY).size(16.0));
        });
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
    // panels

    fn top_bar(&mut self, ui: &mut Ui) {
        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.show_sidebar, RichText::new("☰").size(18.0)).on_hover_text("Settings panel").clicked() {
                    self.show_sidebar = !self.show_sidebar;
                }
                ui.label(RichText::new("SpinSight MRI simulator").strong().size(20.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dark = ui.visuals().dark_mode;
                    if ui.button(if dark { "☀" } else { "☾" }).on_hover_text("Toggle light/dark theme").clicked() {
                        let d = !dark;
                        self.dark_mode = Some(d);
                        ui.ctx().set_theme(if d { egui::Theme::Dark } else { egui::Theme::Light });
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
        });
    }

    fn sidebar(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        egui::Panel::left("sidebar").resizable(false).default_size(PARAM_WIDTH + 40.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Load settings…").clicked() {
                        let mut dlg = rfd::FileDialog::new().add_filter("SpinSight settings", &["toml"]);
                        if let Some(dir) = self.settings_file.as_ref().and_then(|f| f.parent()) {
                            dlg = dlg.set_directory(dir);
                        }
                        if let Some(path) = dlg.pick_file() {
                            match self.ctrl.load_settings_file(&path) {
                                Ok(()) => self.status = Some(format!("Loaded settings from {}", path.display())),
                                Err(e) => self.status = Some(e),
                            }
                            self.settings_file = Some(path);
                            self.submit_if_needed();
                        }
                    }
                    if ui.button("Save settings…").clicked() {
                        let mut dlg = rfd::FileDialog::new().add_filter("SpinSight settings", &["toml"]).set_file_name("settings.toml");
                        if let Some(dir) = self.settings_file.as_ref().and_then(|f| f.parent()) {
                            dlg = dlg.set_directory(dir);
                        }
                        if let Some(path) = dlg.save_file() {
                            match self.ctrl.save_settings_file(&path) {
                                Ok(()) => self.status = Some(format!("Saved settings to {}", path.display())),
                                Err(e) => self.status = Some(e),
                            }
                            self.settings_file = Some(path);
                        }
                    }
                });
                ui.add_space(8.0);
                self.group_panel(ui, "Settings", changes);
                ui.add_space(8.0);
                ui.separator();
                ui.checkbox(&mut self.sliders_with_text, "Text boxes next to sliders");
                if ui.button("New noise realisation").clicked() {
                    self.renew_noise = true;
                    self.submit_if_needed();
                }
                if let Some((name, secs, cached)) = self.worker.status.last_phantom_secs.lock().unwrap().clone() {
                    ui.label(
                        RichText::new(format!("{name} phantom k-space {} in {:.2} s", if cached { "loaded from cache" } else { "computed" }, secs))
                            .small()
                            .color(Color32::GRAY),
                    );
                }
            });
        });
    }

    fn main_page(&mut self, ui: &mut Ui, changes: &mut Vec<(&'static str, Value)>) {
        let ctx = ui.ctx().clone();
        self.update_textures(&ctx);
        let reset = std::mem::take(&mut self.reset_plots);
        // fit the image (and k-space) plots into the window width
        let avail = ui.available_width();
        let params_w = 2.0 * (PARAM_WIDTH + 30.0) + 4.0 * ui.spacing().item_spacing.x + 20.0;
        let size = if self.show_kspace { (avail - params_w) / 2.0 - 20.0 } else { avail - params_w - (PARAM_WIDTH + 30.0) - 20.0 };
        let size = size.clamp(MIN_IMAGE_SIZE, IMAGE_SIZE);
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal_top(|ui| {
                // column 1: buttons, sequence and contrast parameters
                ui.vertical(|ui| {
                    ui.set_width(PARAM_WIDTH + 30.0);
                    ui.horizontal(|ui| {
                        let seq_label = if self.show_sequence { "Hide sequence" } else { "Show sequence" };
                        if ui.button(seq_label).clicked() {
                            self.show_sequence = !self.show_sequence;
                        }
                        let k_label = if self.show_kspace { "Hide k-space" } else { "Show k-space" };
                        if ui.button(k_label).clicked() {
                            self.show_kspace = !self.show_kspace;
                        }
                    });
                    ui.add_space(6.0);
                    self.group_panel(ui, "Sequence", changes);
                    ui.add_space(8.0);
                    self.group_panel(ui, "Contrast", changes);
                });
                // column 2: geometry
                ui.vertical(|ui| {
                    ui.add_space(30.0);
                    self.group_panel(ui, "Geometry", changes);
                });
                // column 3: image and info
                ui.vertical(|ui| {
                    ui.set_width(size + 10.0);
                    let fov = self.result.as_ref().map(|r| r.output.fov).unwrap_or(self.ctrl.protocol.fov);
                    let fov_box = self.ctrl.inputs.show_fov.then(|| plots::fov_box(&self.ctrl.protocol));
                    if self.result.is_none() {
                        placeholder(ui, "Preparing image", size);
                    } else {
                        plots::image_plot(ui, self.image_tex.as_ref().map(|t| &t.0), fov, fov_box.as_ref(), size, reset);
                    }
                    ui.horizontal(|ui| {
                        self.param_widget(ui, "image_mode", changes);
                        ui.add_space(12.0);
                        self.param_widget(ui, "show_FOV", changes);
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        self.param_widget(ui, "reference_tissue", changes);
                        if ui.button("Set reference SNR").clicked() {
                            self.ctrl.set_reference_snr();
                        }
                    });
                    ui.add_space(4.0);
                    let info = self.ctrl.info.clone();
                    ui.horizontal(|ui| {
                        Self::indicator(ui, "Relative SNR", &info.relative_snr);
                        ui.add_space(14.0);
                        Self::indicator(ui, "Scan time", &info.scantime);
                        ui.add_space(14.0);
                        Self::indicator(ui, "Fat/water shift", &info.fw_shift);
                        ui.add_space(14.0);
                        Self::indicator(ui, "Bandwidth", &info.pixel_bandwidth);
                    });
                });
                // column 4: k-space and k-space processing
                ui.vertical(|ui| {
                    if self.show_kspace {
                        ui.set_width(size + 10.0);
                        match &self.result {
                            None => placeholder(ui, "Preparing k-space", size),
                            Some(res) => {
                                let axes = if self.ctrl.inputs.show_processed_kspace { &res.output.processed_axes } else { &res.output.grid_axes };
                                let axes = axes.clone();
                                plots::kspace_plot(ui, self.kspace_tex.as_ref().map(|t| &t.0), &axes, self.hover_k_line.as_deref(), size, reset);
                            }
                        }
                        ui.horizontal(|ui| {
                            self.param_widget(ui, "kspace_mode", changes);
                            self.param_widget(ui, "kspace_exponent", changes);
                        });
                        self.param_widget(ui, "show_processed_kspace", changes);
                        ui.add_space(8.0);
                    } else {
                        ui.add_space(30.0);
                    }
                    self.group_panel(ui, "k-space processing", changes);
                });
            });

            if self.show_sequence {
                ui.add_space(8.0);
                ui.separator();
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
                ui.horizontal(|ui| {
                    self.param_widget(ui, "shot", changes);
                    ui.add_space(12.0);
                    let info = self.ctrl.info.clone();
                    if info.spoke_angle_visible {
                        Self::indicator(ui, "Angle", &info.spoke_angle);
                        ui.add_space(12.0);
                    }
                    Self::indicator(ui, &format!("# {}s", self.ctrl.shot_label), &info.num_shots);
                    ui.add_space(12.0);
                    self.param_widget(ui, "signal_exponent", changes);
                });
            } else if self.hover_k_line.is_some() {
                self.hover_k_line = None;
                self.hover_key = None;
            }

            ui.add_space(16.0);
            self.footer(ui);
        });
    }

    fn footer(&mut self, ui: &mut Ui) {
        let gray = Color32::GRAY;
        let timing = self.result.as_ref().and_then(|r| r.output.timings.iter().find(|t| t.0 == "total").map(|t| t.1));
        ui.label(
            RichText::new("SpinSight – original Python version written by Johan Berglund, Ph.D. (johan.berglund@akademiska.se); Rust port.")
                .italics()
                .color(gray),
        );
        let mut line = format!("(started {}, v {})", self.start_time, env!("CARGO_PKG_VERSION"));
        if let Some(t) = timing {
            line.push_str(&format!(" · last simulation {:.0} ms", t * 1e3));
        }
        ui.label(RichText::new(line).italics().color(gray));
        if let Some(s) = &self.status {
            ui.label(RichText::new(s).color(gray));
        }
    }

    fn progress_modal(&mut self, ui: &mut Ui) {
        let preparing = self.worker.status.preparing_phantom.lock().unwrap().clone();
        if let Some(name) = preparing {
            let done = self.worker.status.progress.done.load(Ordering::Relaxed);
            let total = self.worker.status.progress.total.load(Ordering::Relaxed).max(1);
            egui::Modal::new(egui::Id::unique("progress")).show(ui.ctx(), |ui| {
                ui.set_width(460.0);
                ui.heading(format!("Preparing k-space for \"{name}\" phantom…"));
                ui.label(RichText::new("(first-time loading; the result is cached)").italics());
                ui.add_space(8.0);
                ui.add(egui::ProgressBar::new(done as f32 / total as f32).text(format!("{done} / {total} shapes")).animate(true));
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
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

/// One parameter widget: returns (new value if changed, hovered)
fn param_row(ui: &mut Ui, st: &ParamState, help: bool, with_text: bool, text_buf: &mut HashMap<&'static str, String>, width: f32) -> (Option<Value>, bool) {
    let name = st.spec.name;
    let mut change = None;
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
                ui.label(st.label.as_str());
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
                ui.label(format!("{}: {:.2}", st.label, v));
                ui.spacing_mut().slider_width = width - 20.0;
                let r = ui.add(egui::Slider::new(&mut v, lo..=hi).step_by(st.spec.step).show_value(false));
                if r.changed() {
                    change = Some(Value::Num(v));
                }
            }
            Widget::DiscreteSlider => {
                ui.horizontal(|ui| {
                    ui.label(format!("{}:", st.label));
                    ui.label(RichText::new(st.value_label()).strong());
                });
                let n = st.options.len();
                ui.horizontal(|ui| {
                    let text_w = if with_text { 64.0 } else { 0.0 };
                    ui.spacing_mut().slider_width = width - 20.0 - text_w;
                    if n == 0 {
                        let mut dummy = 0usize;
                        ui.add_enabled(false, egui::Slider::new(&mut dummy, 0..=1).show_value(false));
                    } else {
                        let mut idx = st.option_index().unwrap_or(0);
                        let r = ui.add_enabled(n > 1, egui::Slider::new(&mut idx, 0..=n - 1).show_value(false));
                        if r.changed() && st.options[idx].value != st.value {
                            change = Some(st.options[idx].value.clone());
                        }
                    }
                    if with_text {
                        let is_pct = st.options.first().is_some_and(|o| o.label.ends_with('%'));
                        let shown =
                            if is_pct { crate::formatting::percentage(st.value.num()) } else { crate::params::Value::Num(st.value.num()).num().to_string() };
                        let buf = text_buf.entry(name).or_insert_with(|| shown.clone());
                        let r = ui.add(egui::TextEdit::singleline(buf).desired_width(text_w - 8.0).align(egui::Align2::CENTER_CENTER));
                        if r.lost_focus() {
                            let txt = buf.trim().trim_end_matches('%').to_string();
                            if let Ok(x) = txt.parse::<f64>() {
                                let x = if is_pct { x / 100.0 } else { x };
                                let vals = st.option_values();
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
    ui.add_space(2.0);
    (change, hovered)
}

impl eframe::App for SpinSightApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        if let Some(r) = self.worker.poll() {
            if r.id >= self.result.as_ref().map_or(0, |x| x.id) {
                self.result = Some(r);
            }
        }
        let prev_spans = self.spans;
        self.spans = Spans::default();
        let mut changes = Vec::new();
        self.top_bar(ui);
        if self.show_sidebar {
            self.sidebar(ui, &mut changes);
        }
        egui::CentralPanel::default_margins().show(ui, |ui| {
            self.main_page(ui, &mut changes);
        });
        self.progress_modal(ui);
        if self.spans.te != prev_spans.te || self.spans.tr != prev_spans.tr || self.spans.ti != prev_spans.ti {
            ui.ctx().request_repaint();
        }
        if !changes.is_empty() {
            self.apply_changes(changes);
            ui.ctx().request_repaint();
        }
        if self.worker.status.busy.load(Ordering::Relaxed) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(30));
        }
    }
}
