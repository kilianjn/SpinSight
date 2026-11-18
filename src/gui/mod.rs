//! The SpinSight desktop application (egui), laid out like an MR scanner console:
//! exam bar (phantom, field strength, parameter style) on top, protocol and phantom browser on
//! the left, viewing segments (image, k-space, pulse sequence) in the centre and the parameter
//! card with its tabs and the acquisition status line at the bottom.

mod maker;
mod markdown;
pub mod plots;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use egui::{Color32, RichText, TextureHandle, Ui};

use crate::controller::{Controller, ParamState};
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
            CardTab::Routine => {
                &["sequence_type", "slice_thickness", "FOV_F", "FOV_P", "phase_oversampling", "TR", "TE", "TI", "FA", "NSA", "frequency_direction"]
            }
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
            CardTab::Geometry => &["frequency_direction", "FOV_F", "FOV_P", "phase_oversampling", "slice_thickness", "show_FOV"],
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
            show_image: true,
            show_kspace: opts.show_kspace,
            show_sequence: opts.show_sequence,
            show_browser: true,
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
        let object_before = self.ctrl.inputs.object.clone();
        for (name, value) in changes {
            self.ctrl.on_change(name, value);
        }
        self.after_controller_change(&object_before);
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
                ui.label(RichText::new(if preset == CUSTOM_PRESET { "(custom)".to_string() } else { preset }).strong());
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
                section_title(ui, "OPTIONS");
                ui.checkbox(&mut self.sliders_with_text, "Text boxes next to sliders");
                if ui.button("New noise realisation").clicked() {
                    self.renew_noise = true;
                    self.submit_if_needed();
                }
            });
        });
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
            });
            ui.separator();
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
            status_item(ui, "TA", &crate::formatting::ta(p.scantime), true);
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
            if n_img > 0 {
                let avail_w = ui.available_width();
                let avail_h = ui.available_height();
                let height_share = if self.show_sequence { 0.62 } else { 1.0 };
                let size = ((avail_w - 30.0 * n_img as f32) / n_img as f32).min(avail_h * height_share - 50.0).clamp(MIN_IMAGE_SIZE, 760.0);
                ui.horizontal_top(|ui| {
                    if self.show_image {
                        ui.vertical(|ui| {
                            let fov = self.result.as_ref().map(|r| r.output.fov).unwrap_or(self.ctrl.protocol.fov);
                            let fov_box = self.ctrl.inputs.show_fov.then(|| plots::fov_box(&self.ctrl.protocol));
                            if self.result.is_none() {
                                placeholder(ui, "Preparing image", size);
                            } else {
                                plots::image_plot(ui, self.image_tex.as_ref().map(|t| &t.0), fov, fov_box.as_ref(), size, reset);
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
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}:", st.label)).color(Color32::GRAY));
                    ui.label(RichText::new(format!("{v:.2}")).strong());
                });
                ui.spacing_mut().slider_width = width - 20.0;
                let r = ui.add(egui::Slider::new(&mut v, lo..=hi).step_by(st.spec.step).show_value(false));
                if r.changed() {
                    change = Some(Value::Num(v));
                }
            }
            Widget::DiscreteSlider => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}:", st.label)).color(Color32::GRAY));
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
                        let shown = if is_pct { crate::formatting::percentage(st.value.num()) } else { st.value.num().to_string() };
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
    (change, hovered)
}

impl eframe::App for SpinSightApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
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
        if self.spans.te != prev_spans.te || self.spans.tr != prev_spans.tr || self.spans.ti != prev_spans.ti {
            ctx.request_repaint();
        }
        if !changes.is_empty() {
            self.apply_changes(changes);
            ctx.request_repaint();
        }
        if self.worker.status.busy.load(Ordering::Relaxed) {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }
}
