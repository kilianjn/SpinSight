//! Examination: scanning the planned slice stack, the list of acquired series and their review

use std::sync::atomic::Ordering;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;

use egui::{Color32, RichText, TextureHandle, TextureOptions, Ui};

use super::{preset_title, section_title, SpinSightApp};
use crate::exam::{self, ScanProgress, Series};
use crate::geometry::SliceGeometry;
use crate::params::{Value, CUSTOM_PRESET};

pub struct RunningScan {
    pub progress: Arc<ScanProgress>,
    rx: Receiver<Option<Series>>,
    name: String,
    settings: Vec<(String, Value)>,
    preset: String,
}

pub struct ExamEntry {
    pub series: Arc<Series>,
    /// protocol parameters of the acquisition (for "copy parameters") and the preset they are based on
    settings: Vec<(String, Value)>,
    preset: String,
}

#[derive(Default)]
pub struct ExamState {
    pub entries: Vec<ExamEntry>,
    pub scan: Option<RunningScan>,
    /// series shown in the image viewer (`None`: live image of the current protocol)
    pub review: Option<usize>,
    pub review_slice: usize,
    texture: Option<(TextureHandle, (usize, usize))>,
    next_number: usize,
    confirm_new: bool,
}

impl ExamState {
    pub fn reviewed(&self) -> Option<&Arc<Series>> {
        self.review.and_then(|i| self.entries.get(i)).map(|e| &e.series)
    }

    /// Slices of the reviewed series and the index of the displayed one (for the localizers)
    pub fn review_geometries(&self) -> (Vec<SliceGeometry>, Option<usize>) {
        match self.reviewed() {
            Some(s) => (s.geometries.clone(), Some(self.review_slice.min(s.geometries.len().saturating_sub(1)))),
            None => (Vec::new(), None),
        }
    }
}

impl SpinSightApp {
    /// Name of the current protocol for the series list
    fn protocol_name(&self) -> String {
        let label = self.ctrl.str("load_presets");
        if label != CUSTOM_PRESET {
            preset_title(label)
        } else if self.ctrl.loaded_preset_name != CUSTOM_PRESET {
            format!("{} (mod.)", preset_title(&self.ctrl.loaded_preset_name))
        } else {
            let seq = &self.ctrl.inputs.sequence_type;
            let short = if seq.contains("Inversion") {
                "IR"
            } else if seq.contains("Gradient") {
                "GRE"
            } else {
                "SE"
            };
            format!("{short} custom")
        }
    }

    /// Acquire the planned slice stack in the background
    pub(super) fn start_scan(&mut self, ctx: &egui::Context) {
        if self.exam.scan.is_some() {
            return;
        }
        let Some(vol) = self.ctrl.phantom.volume.clone() else { return };
        self.exam.next_number = self.exam.next_number.max(self.exam.entries.len()) + 1;
        let number = self.exam.next_number;
        let name = self.protocol_name();
        let inputs = self.ctrl.inputs.clone();
        let snr = self.ctrl.info.relative_snr.clone();
        let progress = Arc::new(ScanProgress::default());
        progress.total.store(inputs.num_slices.max(1), Ordering::Relaxed);
        let (tx, rx) = channel();
        let (p, c, n) = (progress.clone(), ctx.clone(), name.clone());
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0) ^ number as u64;
        let spawned = std::thread::Builder::new().name("scan".into()).spawn(move || {
            let ticker = {
                // repaint while scanning (progress bar)
                let (p, c) = (p.clone(), c.clone());
                std::thread::spawn(move || {
                    while p.done.load(Ordering::Relaxed) < p.total.load(Ordering::Relaxed) && !p.cancel.load(Ordering::Relaxed) {
                        c.request_repaint();
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                })
            };
            let series = exam::acquire(&inputs, &vol, &n, number, seed, Some(&p)).map(|mut s| {
                s.relative_snr = snr;
                s
            });
            let _ = tx.send(series);
            c.request_repaint();
            let _ = ticker.join();
        });
        if spawned.is_ok() {
            let settings = self.ctrl.settings().into_iter().map(|(k, v, _)| (k, v)).collect();
            let preset = self.ctrl.loaded_preset_name.clone();
            self.exam.scan = Some(RunningScan { progress, rx, name, settings, preset });
        }
    }

    /// Collect a finished scan
    pub(super) fn poll_scan(&mut self) {
        let Some(scan) = &self.exam.scan else { return };
        match scan.rx.try_recv() {
            Ok(result) => {
                let scan = self.exam.scan.take().unwrap();
                match result {
                    Some(series) => {
                        let mid = series.images.len() / 2;
                        self.status = Some(format!("Series {} \u{201c}{}\u{201d} acquired ({} slices)", series.number, series.name, series.images.len()));
                        self.exam.entries.push(ExamEntry { series: Arc::new(series), settings: scan.settings, preset: scan.preset });
                        self.exam.review = Some(self.exam.entries.len() - 1);
                        self.exam.review_slice = mid;
                        self.show_image = true;
                        self.reset_plots = true;
                    }
                    None => self.status = Some(format!("Scan of \u{201c}{}\u{201d} stopped", scan.name)),
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.exam.scan = None;
                self.status = Some("The scan failed".into());
            }
        }
    }

    /// "Scan" button or progress of the running scan (status line)
    pub(super) fn scan_controls(&mut self, ui: &mut Ui) {
        if !self.ctrl.phantom.is_volume_slice() {
            return;
        }
        if let Some(scan) = &self.exam.scan {
            let p = &scan.progress;
            let (done, total) = (p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed));
            ui.add(egui::ProgressBar::new(p.fraction()).desired_width(150.0).text(format!("Scanning {}/{}", (done + 1).min(total), total)));
            if ui.button("■ Stop").on_hover_text("Stop the acquisition").clicked() {
                p.cancel.store(true, Ordering::Relaxed);
            }
        } else {
            let scan = egui::Button::new(RichText::new("▶ Scan").strong().color(Color32::WHITE)).fill(Color32::from_rgb(30, 130, 70));
            if ui.add(scan).on_hover_text("Acquire all planned slices as a new series of the examination (shown in the EXAM list)").clicked() {
                let ctx = ui.ctx().clone();
                self.start_scan(&ctx);
            }
        }
        ui.separator();
    }

    /// Examination section of the browser: live image and acquired series
    pub(super) fn exam_list(&mut self, ui: &mut Ui) {
        if !self.ctrl.phantom.is_volume_slice() && self.exam.entries.is_empty() {
            return;
        }
        section_title(ui, "EXAM");
        if ui.selectable_label(self.exam.review.is_none(), "● Live (planning)").on_hover_text("Live image of the current protocol and slice position").clicked()
        {
            self.exam.review = None;
            self.reset_plots = true;
        }
        let mut select = None;
        for (i, e) in self.exam.entries.iter().enumerate() {
            let s = &e.series;
            let r = ui.selectable_label(self.exam.review == Some(i), format!("{:>2}  {}", s.number, s.name));
            if r.clicked() {
                select = Some(i);
            }
            let first = &s.inputs;
            let te = first.te.map(|t| format!("{t}")).unwrap_or_else(|| "min.".into());
            r.on_hover_text(format!(
                "Series {}: {}\n{} slices, TA {}\nTR {} ms, TE {te} ms, slice thickness {} mm, FOV {}×{} mm\nmatrix {}×{}, rel. SNR {}",
                s.number,
                s.name,
                s.images.len(),
                crate::formatting::ta(s.scantime),
                first.tr,
                first.slice_thickness,
                first.fov_f,
                first.fov_p,
                first.matrix_f,
                first.matrix_p,
                s.relative_snr
            ));
        }
        if let Some(i) = select {
            if self.exam.review != Some(i) {
                let n = self.exam.entries[i].series.images.len();
                self.exam.review = Some(i);
                self.exam.review_slice = n / 2;
                self.reset_plots = true;
            }
        }
        if let Some(scan) = &self.exam.scan {
            ui.label(RichText::new(format!("    {}  (scanning…)", scan.name)).italics().color(Color32::GRAY));
        }
        if self.exam.entries.is_empty() && self.exam.scan.is_none() {
            ui.label(RichText::new("Plan the slices and press ▶ Scan").small().color(Color32::GRAY));
        }
        ui.horizontal_wrapped(|ui| {
            let selected = self.exam.review;
            if ui
                .add_enabled(selected.is_some(), egui::Button::new("Copy parameters"))
                .on_hover_text("Load the protocol and slice position of the selected series")
                .clicked()
            {
                if let Some(i) = selected {
                    let (settings, preset) = (self.exam.entries[i].settings.clone(), self.exam.entries[i].preset.clone());
                    let before = self.ctrl.inputs.object.clone();
                    self.ctrl.restore_settings(&preset, settings);
                    self.exam.review = None;
                    self.after_controller_change(&before);
                }
            }
            if ui
                .add_enabled(selected.is_some(), egui::Button::new("Save images…"))
                .on_hover_text("Save the images of the selected series as PNG files")
                .clicked()
            {
                if let (Some(i), Some(dir)) = (selected, rfd::FileDialog::new().set_title("Folder for the series images").pick_folder()) {
                    match exam::save_series(&self.exam.entries[i].series, &dir) {
                        Ok(folder) => self.status = Some(format!("Images saved to {}", folder.display())),
                        Err(e) => self.status = Some(e),
                    }
                }
            }
            if ui.add_enabled(!self.exam.entries.is_empty(), egui::Button::new("New exam")).on_hover_text("Remove all series").clicked() {
                self.exam.confirm_new = true;
            }
        });
        ui.add_space(10.0);
    }

    /// Confirmation dialog for "New exam"
    pub(super) fn exam_dialogs(&mut self, ctx: &egui::Context) {
        if !self.exam.confirm_new {
            return;
        }
        let mut close = false;
        egui::Modal::new(egui::Id::unique("confirm_new_exam")).show(ctx, |ui| {
            ui.heading("New examination");
            ui.label(format!("Remove all {} series of the current examination?", self.exam.entries.len()));
            ui.horizontal(|ui| {
                if ui.button("Remove").clicked() {
                    self.exam.entries.clear();
                    self.exam.review = None;
                    self.exam.texture = None;
                    self.exam.next_number = 0;
                    close = true;
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.exam.confirm_new = false;
        }
    }

    /// Image of the reviewed series with a slice selector
    pub(super) fn review_image(&mut self, ui: &mut Ui, size: f32, reset: bool) {
        let Some(series) = self.exam.reviewed().cloned() else { return };
        let n = series.images.len();
        if n == 0 {
            return;
        }
        let mut slice = self.exam.review_slice.min(n - 1);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Series {}: {}", series.number, series.name)).strong().color(super::planning::SERIES_COLOR));
            if n > 1 {
                ui.add(egui::Slider::new(&mut slice, 0..=n - 1).custom_formatter(move |v, _| format!("{}/{n}", v as usize + 1)).text("slice"));
            }
        });
        // page through the slices with the arrow keys while the pointer is over the image area
        self.exam.review_slice = slice;
        let key = (series.number, slice);
        if self.exam.texture.as_ref().is_none_or(|(_, k)| *k != key) {
            let img = &series.images[slice];
            let gray = exam::to_gray(img, series.window);
            let tex = ui.ctx().load_texture("series_image", egui::ColorImage::from_gray([img.cols, img.rows], &gray), TextureOptions::NEAREST);
            self.exam.texture = Some((tex, key));
        }
        let g = series.geometries[slice];
        let letters = crate::geometry::edge_letters(g.u, g.v);
        let tex = self.exam.texture.as_ref().map(|t| &t.0);
        let r = ui.scope(|ui| super::plots::image_plot(ui, tex, series.fov, None, Some(&letters), size, reset)).response;
        if r.contains_pointer() && n > 1 {
            let (up, down) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::PageUp),
                    i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::PageDown),
                )
            });
            if up && self.exam.review_slice + 1 < n {
                self.exam.review_slice += 1;
            }
            if down && self.exam.review_slice > 0 {
                self.exam.review_slice -= 1;
            }
        }
        ui.label(
            RichText::new(format!("TA {} · rel. SNR {} · ↑/↓: page through the slices", crate::formatting::ta(series.scantime), series.relative_snr))
                .small()
                .color(Color32::GRAY),
        );
    }
}
