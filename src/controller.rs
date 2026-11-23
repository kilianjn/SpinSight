//! Parameter state handling between the GUI and the simulation (port of `controller.py`):
//! parameter style conversions, discrete option filtering by the computed bounds, visibility
//! and labels, presets and settings files.

use std::path::Path;
use std::sync::Arc;

use crate::convert;
use crate::formatting;
use crate::params::{self, snap, Opt, ParamSpec, SnapMode, Value, Widget};
use crate::phantom::{self, Phantom};
use crate::presets;
use crate::protocol::{self, pyround, Inputs, MinMax, Protocol, INPUT_NAMES};
use crate::volume;

/// Dynamic state of one parameter widget
#[derive(Debug, Clone)]
pub struct ParamState {
    pub spec: &'static ParamSpec,
    pub value: Value,
    /// current (bounded) options of selector parameters
    pub options: Vec<Opt>,
    pub label: String,
    pub precedence: i32,
}

impl ParamState {
    pub fn visible(&self) -> bool {
        self.precedence > 0
    }
    pub fn option_values(&self) -> Vec<f64> {
        self.options.iter().map(|o| o.value.num()).collect()
    }
    /// index of the current value among the options
    pub fn option_index(&self) -> Option<usize> {
        self.options.iter().position(|o| o.value == self.value).or_else(|| {
            let v = self.value.num();
            if v.is_nan() {
                return None;
            }
            let vals = self.option_values();
            snap(v, &vals, SnapMode::Nearest).and_then(|s| vals.iter().position(|&x| x == s))
        })
    }
    pub fn value_label(&self) -> String {
        match self.option_index() {
            Some(i) if self.options[i].value == self.value => self.options[i].label.clone(),
            _ => match &self.value {
                Value::Num(v) => format!("{v}"),
                Value::Str(s) => s.clone(),
                Value::Bool(b) => b.to_string(),
            },
        }
    }
}

/// Text indicators below the image and sequence diagram
#[derive(Debug, Clone, Default)]
pub struct Info {
    pub relative_snr: String,
    pub scantime: String,
    pub fw_shift: String,
    pub pixel_bandwidth: String,
    pub spoke_angle: String,
    pub num_shots: String,
    pub spoke_angle_visible: bool,
}

/// A parameter that a change of another parameter would adjust
#[derive(Debug, Clone, PartialEq)]
pub struct Adjustment {
    pub name: &'static str,
    pub label: String,
    pub old: String,
    pub new: String,
    pub new_value: Value,
}

/// Parameters that are never reported as adjusted (display settings, presets, phantom, style)
const NOT_ADJUSTMENTS: &[&str] = &[
    "load_presets",
    "object",
    "parameter_style",
    "field_strength",
    "displayed_slice",
    "shot",
    "reference_tissue",
    "image_mode",
    "kspace_mode",
    "kspace_exponent",
    "show_processed_kspace",
    "signal_exponent",
    "show_FOV",
    "show_TE_span",
    "show_TR_span",
    "show_TI_span",
];

/// Parameters whose changes are never checked for conflicts
pub const UNCHECKED_CHANGES: &[&str] = &[
    "load_presets",
    "object",
    "parameter_style",
    "noise_gain",
    "noise_type",
    "interference_level",
    "displayed_slice",
    "shot",
    "reference_tissue",
    "image_mode",
    "kspace_mode",
    "kspace_exponent",
    "show_processed_kspace",
    "signal_exponent",
    "show_FOV",
    "show_TE_span",
    "show_TR_span",
    "show_TI_span",
    "slice_pos_LR",
    "slice_pos_AP",
    "slice_pos_HF",
    "slice_angle_1",
    "slice_angle_2",
    "slice_orientation",
];

/// Result of a quick check of a parameter change, ordered by severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RangeState {
    Free,
    Adjusts,
    Impossible,
}

/// Options of a slider with the information which of them adjust other parameters
#[derive(Debug, Clone)]
pub struct SliderRange {
    pub name: &'static str,
    /// controller generation the range was computed for
    pub generation: u64,
    pub options: Vec<Opt>,
    /// per option: other parameters would be adjusted
    pub adjusts: Vec<bool>,
}

#[derive(Clone)]
pub struct Controller {
    pub params: Vec<ParamState>,
    pub inputs: Inputs,
    pub protocol: Arc<Protocol>,
    pub phantom: Arc<Phantom>,
    pub loaded_preset: Vec<(String, Value)>,
    pub loaded_preset_name: String,
    pub rec_acq_ratio_f: f64,
    pub rec_acq_ratio_p: f64,
    pub reference_snr: f64,
    pub shot_label: String,
    pub info: Info,
    /// incremented whenever the protocol was recomputed
    pub generation: u64,
    pub messages: Vec<String>,
    /// only previewing a change: the slice of a 3D phantom is not recomputed
    preview_only: bool,
}

/// Value of the graph node of parameter `name` for the given inputs and protocol
fn graph_value_of(i: &Inputs, p: &Protocol, name: &str) -> Option<Value> {
    let n = |x: f64| Some(Value::Num(x));
    let s = |x: &str| Some(Value::Str(x.to_string()));
    let b = |x: bool| Some(Value::Bool(x));
    match name {
        "object" => s(&i.object),
        "field_strength" => n(i.field_strength),
        "min_voxel_size" => n(i.min_voxel_size),
        "noise_gain" => n(i.noise_gain),
        "noise_type" => s(&i.noise_type),
        "interference_level" => n(i.interference_level),
        "sequence_type" => s(&i.sequence_type),
        "pixel_bandwidth" => n(i.pixel_bandwidth),
        "FOV_bandwidth" => n(43.2),
        "NSA" => n(i.nsa),
        "partial_Fourier_factor" => n(i.partial_fourier_factor),
        "turbo_factor" => n(p.turbo_factor as f64),
        "EPI_factor" => n(p.epi_factor as f64),
        "echo_order" => s(&p.echo_order),
        "FatSat" => b(i.fatsat),
        "TR" => n(p.tr),
        "TE" => n(p.te),
        "TI" => n(i.ti),
        "FA" => n(i.fa),
        "trajectory" => s(&i.trajectory),
        "frequency_direction" => s(&i.frequency_direction),
        "FOV_P" => n(i.fov_p),
        "FOV_F" => n(i.fov_f),
        "phase_oversampling" => n(i.phase_oversampling),
        "radial_sampling_factor" => n(i.radial_sampling_factor),
        "matrix_P" => n(p.matrix_p as f64),
        "matrix_F" => n(p.matrix_f as f64),
        "recon_matrix_P" => n(p.recon_matrix_p as f64),
        "recon_matrix_F" => n(p.recon_matrix_f as f64),
        "slice_thickness" => n(i.slice_thickness),
        "radial_FOV_oversampling" => n(i.radial_fov_oversampling),
        "show_FOV" => b(i.show_fov),
        "show_TE_span" => b(i.show_te_span),
        "show_TR_span" => b(i.show_tr_span),
        "show_TI_span" => b(i.show_ti_span),
        "reference_tissue" => s(&p.reference_tissue),
        "image_mode" => s(&i.image_mode),
        "show_processed_kspace" => b(i.show_processed_kspace),
        "kspace_exponent" => n(i.kspace_exponent),
        "kspace_mode" => s(&i.kspace_mode),
        "homodyne" => b(i.homodyne),
        "do_apodize" => b(i.do_apodize),
        "apodization_alpha" => n(i.apodization_alpha),
        "do_zerofill" => b(i.do_zerofill),
        "shot" => n(p.shot as f64),
        "signal_exponent" => n(i.signal_exponent),
        "slice_orientation" => s(&i.slice_orientation),
        "slice_angle_1" => n(i.slice_angle_1),
        "slice_angle_2" => n(i.slice_angle_2),
        "slice_pos_LR" => n(i.slice_pos[0]),
        "slice_pos_AP" => n(i.slice_pos[1]),
        "slice_pos_HF" => n(i.slice_pos[2]),
        "num_slices" => n(i.num_slices as f64),
        "slice_gap" => n(i.slice_gap),
        "displayed_slice" => n(i.displayed_slice.min(i.num_slices.max(1) - 1) as f64),
        _ => None,
    }
}

/// Bounds of parameter `name` for the given inputs and protocol (`bounds` graph nodes)
fn bounds_of(i: &Inputs, p: &Protocol, name: &str) -> Option<MinMax> {
    match name {
        "shot" => Some(p.shot_bounds),
        "turbo_factor" => Some(p.turbo_factor_bounds),
        "matrix_F" => Some(p.matrix_f_bounds),
        "matrix_P" => Some(p.matrix_p_bounds),
        "recon_matrix_F" => Some(p.recon_matrix_f_bounds),
        "recon_matrix_P" => Some(p.recon_matrix_p_bounds),
        "FOV_F" => Some(p.fov_f_bounds),
        "FOV_P" => Some(p.fov_p_bounds),
        "TR" => Some(p.tr_bounds),
        "TI" => p.ti_bounds,
        "TE" => Some(p.te_bounds),
        "pixel_bandwidth" => Some(p.pixel_bandwidth_bounds),
        "slice_thickness" => Some(p.slice_thickness_bounds),
        "displayed_slice" => Some(MinMax::new(0.0, (i.num_slices.max(1) - 1) as f64)),
        _ => None,
    }
}

fn upper_contains(s: &str, pat: &str) -> bool {
    s.to_uppercase().contains(pat)
}

impl Controller {
    pub fn new() -> Result<Self, String> {
        let params: Vec<ParamState> = params::params()
            .iter()
            .map(|spec| ParamState {
                spec,
                value: spec.default.clone(),
                options: spec.options.clone().unwrap_or_default(),
                label: spec.label.to_string(),
                precedence: spec.precedence,
            })
            .collect();
        let inputs = Inputs::default();
        let phantom = Arc::new(phantom::load(&inputs.object, inputs.min_voxel_size)?);
        let protocol = Arc::new(protocol::compute(&inputs, &phantom));
        let mut c = Controller {
            params,
            inputs,
            protocol,
            phantom,
            loaded_preset: Vec::new(),
            loaded_preset_name: params::CUSTOM_PRESET.to_string(),
            rec_acq_ratio_f: 0.0,
            rec_acq_ratio_p: 0.0,
            reference_snr: 0.0,
            shot_label: "shot".into(),
            info: Info::default(),
            generation: 0,
            messages: Vec::new(),
            preview_only: false,
        };
        c.refresh_phantoms();
        c.sync_with_graph();
        Ok(c)
    }

    pub fn state(&self, name: &str) -> &ParamState {
        self.params.iter().find(|p| p.spec.name == name).unwrap_or_else(|| panic!("unknown parameter {name}"))
    }
    fn state_mut(&mut self, name: &str) -> &mut ParamState {
        self.params.iter_mut().find(|p| p.spec.name == name).unwrap_or_else(|| panic!("unknown parameter {name}"))
    }
    pub fn value(&self, name: &str) -> &Value {
        &self.state(name).value
    }
    pub fn num(&self, name: &str) -> f64 {
        self.value(name).num()
    }
    pub fn str(&self, name: &str) -> &str {
        self.value(name).str()
    }
    pub fn bool(&self, name: &str) -> bool {
        self.value(name).bool()
    }

    fn parameter_style(&self) -> String {
        self.str("parameter_style").to_string()
    }
    pub fn pixel_bw_is_input(&self) -> bool {
        upper_contains(&self.parameter_style(), "PIXEL BW")
    }
    pub fn fov_bw_is_input(&self) -> bool {
        upper_contains(&self.parameter_style(), "FOV BW")
    }
    pub fn fw_shift_is_input(&self) -> bool {
        upper_contains(&self.parameter_style(), "FAT/WATER SHIFT")
    }
    pub fn matrix_is_input(&self) -> bool {
        upper_contains(&self.parameter_style(), "MATRIX")
    }
    pub fn voxel_size_is_input(&self) -> bool {
        upper_contains(&self.parameter_style(), "VOXEL SIZE")
    }
    fn isotropic_voxel_size(&self) -> bool {
        self.str("trajectory").eq_ignore_ascii_case("radial")
    }

    /// Value of a graph node that corresponds to a parameter
    fn graph_value(&self, name: &str) -> Option<Value> {
        graph_value_of(&self.inputs, &self.protocol, name)
    }

    fn bounds_node(&self, name: &str) -> Option<MinMax> {
        bounds_of(&self.inputs, &self.protocol, name)
    }

    // ---------------------------------------------------------------------------------------
    // user interaction

    /// The other parameters that setting `name` to `value` would adjust (parameter conflicts),
    /// without changing anything
    pub fn preview(&self, name: &str, value: &Value) -> Vec<Adjustment> {
        if UNCHECKED_CHANGES.contains(&name) || self.value(name) == value {
            return Vec::new();
        }
        let mut c = self.clone();
        c.preview_only = true;
        c.on_change(name, value.clone());
        self.adjustments(&c, name)
    }

    /// Set `name` to `value` together with the adjustments it causes (see `preview`), which are
    /// kept as the new values of those parameters (e.g. a TR prolonged for a longer TE stays
    /// prolonged when TE is shortened again), as on scanners
    pub fn change_with_adjustments(&mut self, name: &str, value: Value, adjustments: &[Adjustment]) {
        self.on_change(name, value);
        for a in adjustments {
            // only parameters that took the previewed value and are inputs themselves
            if INPUT_NAMES.contains(&a.name) && self.value(a.name) == &a.new_value {
                let st = self.state_mut(a.name);
                st.value = a.new_value.clone();
                if let Err(e) = self.inputs.set(a.name, &a.new_value) {
                    self.messages.push(e);
                }
            }
        }
        self.update_inputs(&[]);
        self.sync_with_graph();
    }

    /// Visible parameters that differ between `self` and `other`, apart from `changed` and the
    /// parameters that follow it by design
    fn adjustments(&self, other: &Controller, changed: &str) -> Vec<Adjustment> {
        let companions: &[&str] = match changed {
            "matrix_F" => &["recon_matrix_F"],
            "matrix_P" => &["recon_matrix_P"],
            "voxel_F" => &["recon_voxel_F"],
            "voxel_P" => &["recon_voxel_P"],
            _ => &[],
        };
        let mut out = Vec::new();
        for (a, b) in self.params.iter().zip(other.params.iter()) {
            let n = a.spec.name;
            if n == changed || !a.visible() || a.spec.group.is_none() || NOT_ADJUSTMENTS.contains(&n) || companions.contains(&n) {
                continue;
            }
            let differs = match (&a.value, &b.value) {
                (Value::Num(x), Value::Num(y)) => (x - y).abs() > 1e-9 * x.abs().max(y.abs()).max(1.0),
                (x, y) => x != y,
            };
            if differs {
                out.push(Adjustment {
                    name: n,
                    label: a.label.trim_end_matches(" [°]").to_string(),
                    old: a.value_label(),
                    new: b.value_label(),
                    new_value: b.value.clone(),
                });
            }
        }
        out
    }

    /// Displayed value of parameter `name` (incl. the conversions of the parameter styles) for
    /// the given inputs and protocol
    fn raw_value(&self, i: &Inputs, p: &Protocol, name: &str) -> Option<Value> {
        match name {
            "FOV_bandwidth" => Some(Value::Num(convert::pixel_bw_to_fov_bw(i.pixel_bandwidth, p.matrix_f as f64))),
            "FW_shift" => Some(Value::Num(convert::pixel_bw_to_shift(i.pixel_bandwidth, i.field_strength))),
            "voxel_F" => Some(Value::Num(i.fov_f / p.matrix_f as f64)),
            "voxel_P" => Some(Value::Num(i.fov_p / p.matrix_p as f64)),
            "recon_voxel_F" => Some(Value::Num(i.fov_f / p.recon_matrix_f as f64)),
            "recon_voxel_P" => Some(Value::Num(i.fov_p / p.recon_matrix_p as f64)),
            _ => graph_value_of(i, p, name),
        }
    }

    /// Whether `v` lies within the bounds of parameter `name` for the given inputs and protocol
    /// (as `update_bounds` restricts the options)
    fn within_bounds(&self, i: &Inputs, p: &Protocol, name: &str, v: f64) -> bool {
        let tol = 1e-9 * v.abs().max(1.0);
        let inside = |b: MinMax| b.min.is_none_or(|m| v >= m - tol) && b.max.is_none_or(|m| v <= m + tol);
        let pbw = p.pixel_bandwidth_bounds;
        let (pmin, pmax) = (pbw.min.unwrap_or(f64::INFINITY), pbw.max.unwrap_or(f64::NEG_INFINITY));
        match name {
            "FOV_bandwidth" => {
                let m = p.matrix_f as f64;
                inside(MinMax::new(convert::pixel_bw_to_fov_bw(pmin, m), convert::pixel_bw_to_fov_bw(pmax, m)))
            }
            "FW_shift" => inside(MinMax::new(convert::pixel_bw_to_shift(pmax, i.field_strength), convert::pixel_bw_to_shift(pmin, i.field_strength))),
            "voxel_F" | "voxel_P" | "recon_voxel_F" | "recon_voxel_P" => {
                let dir = &name[name.len() - 1..];
                let fov = if dir == "F" { i.fov_f } else { i.fov_p };
                let prefix = if name.starts_with("recon_") { "recon_" } else { "" };
                let b = bounds_of(i, p, &format!("{prefix}matrix_{dir}")).unwrap_or_default();
                inside(MinMax { min: b.max.map(|m| fov / m), max: b.min.map(|m| fov / m) })
            }
            "EPI_factor" => v <= p.max_epi_factor as f64 + tol && (p.turbo_factor <= 1 || (v.round() as i64) % 2 == 1),
            _ => bounds_of(i, p, name).is_none_or(inside),
        }
    }

    /// Quick evaluation of a change (inputs and protocol only, no parameter state): whether
    /// `name` takes `value` and whether other visible parameters change
    fn quick_check(&self, name: &str, value: &Value) -> RangeState {
        let mut inp = self.inputs.clone();
        for (k, v) in self.inputs_to_update(name, value.clone()) {
            if inp.set(&k, &v).is_err() {
                return RangeState::Impossible;
            }
        }
        let p = protocol::compute(&inp, &self.phantom);
        let same = |a: &Option<Value>, b: &Option<Value>| match (a, b) {
            (Some(Value::Num(x)), Some(Value::Num(y))) => (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0),
            (x, y) => x == y,
        };
        if !same(&self.raw_value(&inp, &p, name), &Some(value.clone())) || !self.within_bounds(&inp, &p, name, value.num()) {
            return RangeState::Impossible;
        }
        let companions: &[&str] = match name {
            "matrix_F" => &["recon_matrix_F"],
            "matrix_P" => &["recon_matrix_P"],
            "voxel_F" => &["recon_voxel_F"],
            "voxel_P" => &["recon_voxel_P"],
            _ => &[],
        };
        for st in &self.params {
            let n = st.spec.name;
            if n == name || !st.visible() || st.spec.group.is_none() || NOT_ADJUSTMENTS.contains(&n) || companions.contains(&n) {
                continue;
            }
            if !same(&self.raw_value(&self.inputs, &self.protocol, n), &self.raw_value(&inp, &p, n)) {
                return RangeState::Adjusts;
            }
        }
        RangeState::Free
    }

    /// Selectable range of the discrete slider `name`, Siemens style: the values that change no
    /// other parameter (the normal range), extended by the values for which other parameters are
    /// adjusted (e.g. TR is prolonged for a longer TE or a higher turbo factor). `None` if the
    /// parameter has no such range.
    pub fn slider_range(&self, name: &str) -> Option<SliderRange> {
        let st = self.state(name);
        // (EPI factor: only odd values in turbo spin echo, not a contiguous range)
        if st.spec.widget != Widget::DiscreteSlider || UNCHECKED_CHANGES.contains(&name) || name == "EPI_factor" || st.options.is_empty() {
            return None;
        }
        let all = st.spec.options.as_ref()?;
        let vals: Vec<f64> = all.iter().map(|o| o.value.num()).collect();
        if vals.windows(2).any(|w| w[1] <= w[0]) {
            return None;
        }
        let pos = |v: f64| vals.iter().position(|&x| x == v);
        let current = pos(st.value.num()).or_else(|| snap(st.value.num(), &vals, SnapMode::Nearest).and_then(pos))?;
        let lo_b = pos(st.options[0].value.num())?;
        let hi_b = pos(st.options[st.options.len() - 1].value.num())?;
        let check = |k: usize| if k == current { RangeState::Free } else { self.quick_check(name, &all[k].value) };
        // first index (going away from the current value) whose state is at least `level`,
        // assuming the state only gets worse with the distance (Free → Adjusts → Impossible)
        let boundary = |up: bool, level: RangeState| -> usize {
            let n = if up { vals.len() - 1 - current } else { current };
            let at = |d: usize| if up { current + d } else { current - d };
            let (mut a, mut b) = (0usize, n + 1); // invariant: state(a) < level, state(b) >= level (b = n+1: none)
            while b - a > 1 {
                let m = (a + b) / 2;
                if check(at(m)) >= level {
                    b = m;
                } else {
                    a = m;
                }
            }
            b
        };
        let (free_up, ok_up) = (boundary(true, RangeState::Adjusts), boundary(true, RangeState::Impossible));
        let (free_dn, ok_dn) = (boundary(false, RangeState::Adjusts), boundary(false, RangeState::Impossible));
        // selectable: at least the normal range, plus the extension in which the change succeeds
        let lo = (current - (ok_dn - 1)).min(lo_b);
        let hi = (current + (ok_up - 1)).max(hi_b);
        let options: Vec<Opt> = all[lo..=hi].to_vec();
        let adjusts: Vec<bool> = (lo..=hi)
            .map(|k| {
                if k > current {
                    k - current >= free_up
                } else if k < current {
                    current - k >= free_dn
                } else {
                    false
                }
            })
            .collect();
        Some(SliderRange { name: st.spec.name, generation: self.generation, options, adjusts })
    }

    /// The user changed parameter `name` to `value` (`on_param_change`)
    pub fn on_change(&mut self, name: &str, value: Value) {
        self.state_mut(name).value = value.clone();
        if name == "load_presets" {
            self.loaded_preset_name = value.str().to_string();
            let preset = if self.loaded_preset_name == params::CUSTOM_PRESET { None } else { presets::preset(&self.loaded_preset_name) };
            match preset {
                Some(p) => self.apply_settings(p, true),
                None => self.loaded_preset.clear(),
            }
            return;
        }
        if self.loaded_preset.iter().any(|(k, _)| k == name) {
            self.update_preset_label();
        }
        let updates = self.inputs_to_update(name, value);
        self.update_inputs(&updates);
        if name == "object" {
            // adapt FOV to phantom
            let fd = self.protocol.freq_dir;
            let pd = self.protocol.phase_dir;
            let sup = self.phantom.support;
            let fov_f = sup[fd].max(self.inputs.fov_f);
            let fov_p = sup[pd].max(self.inputs.fov_p);
            self.update_inputs(&[("FOV_F".into(), Value::Num(fov_f)), ("FOV_P".into(), Value::Num(fov_p))]);
        }
        self.sync_with_graph();
    }

    fn update_preset_label(&mut self) {
        let matches = self.loaded_preset.iter().all(|(k, v)| match params::try_param(k) {
            Some(_) => {
                let cur = self.value(k);
                match (cur, v) {
                    (Value::Num(a), Value::Num(b)) => a == b,
                    _ => cur == v,
                }
            }
            None => true,
        });
        let label = if matches { self.loaded_preset_name.clone() } else { params::CUSTOM_PRESET.to_string() };
        self.state_mut("load_presets").value = Value::Str(label);
    }

    /// Convert the triggering change into graph input updates (`inputs_to_update`)
    fn inputs_to_update(&self, triggered: &str, value: Value) -> Vec<(String, Value)> {
        let mut inputs: Vec<(String, Value)> = vec![(triggered.to_string(), value)];
        let has = |inputs: &Vec<(String, Value)>, k: &str| inputs.iter().any(|(n, _)| n == k);
        let get = |inputs: &Vec<(String, Value)>, k: &str| inputs.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        let remove = |inputs: &mut Vec<(String, Value)>, k: &str| inputs.retain(|(n, _)| n != k);
        let set = |inputs: &mut Vec<(String, Value)>, k: &str, v: Value| {
            if let Some(e) = inputs.iter_mut().find(|(n, _)| n == k) {
                e.1 = v;
            } else {
                inputs.push((k.to_string(), v));
            }
        };
        if has(&inputs, "parameter_style") {
            set(&mut inputs, "constant_voxel_bounds", Value::Bool(self.voxel_size_is_input()));
            set(&mut inputs, "constant_FOV_BW_bounds", Value::Bool(self.fov_bw_is_input()));
            remove(&mut inputs, "parameter_style");
        }
        if let Some(v) = get(&inputs, "FOV_bandwidth") {
            set(&mut inputs, "pixel_bandwidth", Value::Num(convert::fov_bw_to_pixel_bw(v.num(), self.protocol.matrix_f as f64)));
            remove(&mut inputs, "FOV_bandwidth");
        }
        if let Some(v) = get(&inputs, "FW_shift") {
            set(&mut inputs, "pixel_bandwidth", Value::Num(convert::shift_to_pixel_bw(v.num(), self.inputs.field_strength)));
            remove(&mut inputs, "FW_shift");
        }
        for dir in ["F", "P"] {
            let fov = if dir == "F" { self.inputs.fov_f } else { self.inputs.fov_p };
            for prefix in ["", "recon_"] {
                let key = format!("{prefix}voxel_{dir}");
                if let Some(v) = get(&inputs, &key) {
                    set(&mut inputs, &format!("{prefix}matrix_{dir}"), Value::Num(pyround(fov / v.num())));
                    remove(&mut inputs, &key);
                }
            }
        }
        if self.isotropic_voxel_size() {
            if let Some(m) = get(&inputs, "matrix_F") {
                let voxel = self.num("FOV_F") / m.num();
                set(&mut inputs, "matrix_P", Value::Num(pyround(self.num("FOV_P") / voxel)));
            }
        }
        if self.fov_bw_is_input() && has(&inputs, "matrix_F") && !has(&inputs, "pixel_bandwidth") {
            // maintain constant FOV bandwidth
            let m = get(&inputs, "matrix_F").unwrap().num();
            set(&mut inputs, "pixel_bandwidth", Value::Num(convert::fov_bw_to_pixel_bw(self.num("FOV_bandwidth"), m)));
        }
        for dir in ["F", "P"] {
            // maintain constant voxel size
            let fov_key = format!("FOV_{dir}");
            let matrix_key = format!("matrix_{dir}");
            if self.voxel_size_is_input() && has(&inputs, &fov_key) && !has(&inputs, &matrix_key) {
                let fov = get(&inputs, &fov_key).unwrap().num();
                set(&mut inputs, &matrix_key, Value::Num(pyround(fov / self.num(&format!("voxel_{dir}")))));
            }
            // maintain constant rec/acq ratio when acq matrix is changed
            if let Some(m) = get(&inputs, &matrix_key) {
                let ratio = if dir == "F" { self.rec_acq_ratio_f } else { self.rec_acq_ratio_p };
                set(&mut inputs, &format!("recon_matrix_{dir}"), Value::Num(pyround(m.num() * ratio)));
            }
        }
        // a new number of slices shows the middle slice of the stack
        if let Some(n) = get(&inputs, "num_slices") {
            if !has(&inputs, "displayed_slice") {
                set(&mut inputs, "displayed_slice", Value::Num(((n.num().max(1.0) - 1.0) / 2.0).floor()));
            }
        }
        if get(&inputs, "echo_order").is_some_and(|v| v.str() == "Centric") && !has(&inputs, "TE") {
            set(&mut inputs, "TE", Value::s("min"));
        }
        inputs
    }

    /// Set graph inputs (`Graph.update_inputs`); loads the phantom geometry when the object changes
    fn update_inputs(&mut self, updates: &[(String, Value)]) {
        for (k, v) in updates {
            if let Err(e) = self.inputs.set(k, v) {
                self.messages.push(e);
            }
        }
        self.inputs.displayed_slice = self.inputs.displayed_slice.min(self.inputs.num_slices.max(1) - 1);
        if self.preview_only && self.phantom.is_volume_slice() && self.phantom.name == self.inputs.object {
            // (previews only need the protocol, which hardly depends on the slice position)
        } else if let Some(vol) = volume::builtin(&self.inputs.object) {
            // slice through the 3D phantom at the planned position
            let geom = self.inputs.displayed_slice_geometry();
            if self.phantom.slice != Some(geom) || self.phantom.name != self.inputs.object || self.phantom.min_voxel_size != self.inputs.min_voxel_size {
                self.phantom = Arc::new(phantom::slice_phantom(&vol, geom, self.inputs.min_voxel_size));
            }
        } else if self.phantom.name != self.inputs.object || self.phantom.min_voxel_size != self.inputs.min_voxel_size {
            match phantom::load(&self.inputs.object, self.inputs.min_voxel_size) {
                Ok(ph) => self.phantom = Arc::new(ph),
                Err(e) => {
                    self.messages.push(e);
                    self.inputs.object = self.phantom.name.clone();
                }
            }
        }
        self.protocol = Arc::new(protocol::compute(&self.inputs, &self.phantom));
        self.generation += 1;
    }

    /// Apply settings (from a preset or a settings file) directly to the graph inputs
    pub fn apply_settings(&mut self, settings: Vec<(String, Value)>, is_preset: bool) {
        self.loaded_preset = if is_preset { settings.clone() } else { Vec::new() };
        let valid: Vec<(String, Value)> = settings.into_iter().filter(|(k, _)| INPUT_NAMES.contains(&k.as_str()) || k.starts_with("constant_")).collect();
        self.update_inputs(&valid);
        self.sync_with_graph();
    }

    /// Restore the parameters of an earlier acquisition (`preset`: the preset it was based on)
    pub fn restore_settings(&mut self, preset: &str, settings: Vec<(String, Value)>) {
        match presets::preset(preset).filter(|_| preset != params::CUSTOM_PRESET) {
            Some(p) => {
                self.loaded_preset_name = preset.to_string();
                self.apply_settings(p, true);
            }
            None => self.forget_preset(),
        }
        let preset_settings = std::mem::take(&mut self.loaded_preset);
        self.apply_settings(settings, false);
        self.loaded_preset = preset_settings;
        if self.loaded_preset.is_empty() {
            self.forget_preset();
        } else {
            self.update_preset_label();
        }
    }

    /// Reset all parameters (protocol, FOV, slice planning, noise, display) to the standard
    /// protocol, a T1-weighted spin echo; the phantom is kept
    pub fn reset_to_standard(&mut self) -> Result<(), String> {
        let object = self.inputs.object.clone();
        let mut c = Controller::new()?;
        c.generation = self.generation + 1;
        if object != c.inputs.object && c.state("object").options.iter().any(|o| o.value.str() == object) {
            c.on_change("object", Value::Str(object));
        }
        c.on_change("load_presets", Value::s(presets::STANDARD_PRESET));
        c.set_reference_snr();
        *self = c;
        Ok(())
    }

    pub fn forget_preset(&mut self) {
        self.state_mut("load_presets").value = Value::s(params::CUSTOM_PRESET);
        self.loaded_preset_name = params::CUSTOM_PRESET.to_string();
        self.loaded_preset.clear();
    }

    pub fn load_settings_file(&mut self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let settings = presets::parse_settings(&text)?;
        self.forget_preset();
        self.apply_settings(settings, false);
        Ok(())
    }

    /// Visible input parameters and their values (`get_input_params`)
    pub fn settings(&self) -> Vec<(String, Value, bool)> {
        self.params
            .iter()
            .filter(|p| INPUT_NAMES.contains(&p.spec.name) && p.spec.name != "FOV_bandwidth" && p.precedence >= 0)
            .map(|p| (p.spec.name.to_string(), p.value.clone(), p.spec.integer))
            .collect()
    }

    pub fn save_settings_file(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, presets::write_settings(&self.settings())).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Re-read the list of available phantoms (after importing, making or deleting one)
    pub fn refresh_phantoms(&mut self) {
        let names = phantom::phantom_names();
        self.state_mut("object").options = names.iter().map(|n| Opt { label: n.clone(), value: Value::Str(n.clone()) }).collect();
    }

    /// Switch to phantom `name`; `reload` forces re-reading its files (e.g. after editing)
    pub fn select_phantom(&mut self, name: &str, reload: bool) {
        self.refresh_phantoms();
        if reload && self.phantom.name == name {
            match phantom::load(name, self.inputs.min_voxel_size) {
                Ok(ph) => {
                    self.phantom = Arc::new(ph);
                    self.protocol = Arc::new(protocol::compute(&self.inputs, &self.phantom));
                    self.generation += 1;
                    self.sync_with_graph();
                }
                Err(e) => self.messages.push(e),
            }
        } else {
            self.on_change("object", Value::Str(name.to_string()));
        }
    }

    pub fn set_reference_snr(&mut self) {
        self.reference_snr = self.protocol.snr;
        self.update_relative_snr();
    }

    // ---------------------------------------------------------------------------------------
    // synchronisation with the computed protocol

    /// `sync_with_graph`
    pub fn sync_with_graph(&mut self) {
        self.update_bounds();
        self.update_params();
        self.update_visibility();
        self.update_info_params();
        self.update_rec_acq_ratio();
    }

    fn set_visibility(&mut self, name: &str, visible: bool) {
        let st = self.state_mut(name);
        let p = st.precedence.abs().max(1);
        st.precedence = if visible { p } else { -p };
    }

    /// `set_param`: snap to the current options and set
    fn set_param(&mut self, name: &str, value: Value) {
        let st = self.state_mut(name);
        let new = match (&value, st.options.is_empty()) {
            (Value::Num(v), false) => {
                let vals = st.option_values();
                Value::Num(snap(*v, &vals, SnapMode::Nearest).unwrap_or(*v))
            }
            (Value::Str(s), false) => match st.options.iter().find(|o| o.value.str() == s) {
                Some(o) => o.value.clone(),
                None => value.clone(),
            },
            _ => value.clone(),
        };
        if st.value != new {
            st.value = new;
        }
    }

    /// `set_param_bounds` for discrete parameters: filter the default options
    fn set_param_bounds(&mut self, name: &str, minval: Option<f64>, maxval: Option<f64>) {
        let st = self.state_mut(name);
        let Some(defaults) = st.spec.options.as_ref() else { return };
        if defaults.is_empty() {
            return;
        }
        let first = defaults[0].value.num();
        let last = defaults[defaults.len() - 1].value.num();
        let lo = minval.map_or(first, |m| m.max(first));
        let hi = maxval.map_or(last, |m| m.min(last));
        st.options = defaults.iter().filter(|o| o.value.num() >= lo && o.value.num() <= hi).cloned().collect();
    }

    fn update_bounds(&mut self) {
        for &name in INPUT_NAMES {
            if let Some(b) = self.bounds_node(name) {
                self.set_param_bounds(name, b.min, b.max);
            }
        }
        let pbw = self.protocol.pixel_bandwidth_bounds;
        let (pmin, pmax) = (pbw.min.unwrap_or(f64::INFINITY), pbw.max.unwrap_or(f64::NEG_INFINITY));
        if self.fov_bw_is_input() {
            let m = self.protocol.matrix_f as f64;
            self.set_param_bounds("FOV_bandwidth", Some(convert::pixel_bw_to_fov_bw(pmin, m)), Some(convert::pixel_bw_to_fov_bw(pmax, m)));
        }
        if self.fw_shift_is_input() {
            let b0 = self.inputs.field_strength;
            self.set_param_bounds("FW_shift", Some(convert::pixel_bw_to_shift(pmax, b0)), Some(convert::pixel_bw_to_shift(pmin, b0)));
        }
        if self.voxel_size_is_input() {
            for dir in ["F", "P"] {
                let fov = if dir == "F" { self.inputs.fov_f } else { self.inputs.fov_p };
                for prefix in ["", "recon_"] {
                    let b = self.bounds_node(&format!("{prefix}matrix_{dir}")).unwrap_or_default();
                    let lo = b.max.map(|m| fov / m);
                    let hi = b.min.map(|m| fov / m);
                    self.set_param_bounds(&format!("{prefix}voxel_{dir}"), lo, hi);
                }
            }
        }
        // EPI factor
        let max_epi = self.protocol.max_epi_factor as f64;
        self.set_param_bounds("EPI_factor", None, Some(max_epi));
        if self.protocol.turbo_factor > 1 {
            // EPI_factor must be odd for turbo spin echo (GRASE)
            self.state_mut("EPI_factor").options.retain(|o| (o.value.num() as i64) % 2 == 1);
        }
        let echo_opts = self.protocol.echo_order_options.iter().map(|s| Opt { label: s.clone(), value: Value::Str(s.clone()) }).collect();
        self.state_mut("echo_order").options = echo_opts;
        let tissue_opts = self.protocol.tissues.iter().map(|s| Opt { label: s.clone(), value: Value::Str(s.clone()) }).collect();
        self.state_mut("reference_tissue").options = tissue_opts;
        self.update_x_y_labels();
        self.update_shot_label();
        self.update_slice_labels();
        let te_label = if self.protocol.num_segm == 1 { "TE" } else { "TE (effective)" };
        self.state_mut("TE").label = te_label.into();
    }

    /// Labels depending on the slice orientation of 3D phantoms
    fn update_slice_labels(&mut self) {
        use crate::geometry::Orientation;
        let o = Orientation::from_name(&self.inputs.slice_orientation);
        let (t1, t2) = o.tilt_targets();
        let l1 = format!("Angulation {}>{} [°]", o.short(), t1.short());
        let l2 = format!("Angulation {}>{} [°]", o.short(), t2.short());
        self.state_mut("slice_angle_1").label = l1;
        self.state_mut("slice_angle_2").label = l2;
        // frequency direction: the stored values mean "vertical" / "horizontal" in the image
        let (vertical, horizontal) = if !self.phantom.is_volume_slice() {
            ("anterior-posterior", "left-right")
        } else {
            match o {
                Orientation::Transversal => ("anterior-posterior", "left-right"),
                Orientation::Sagittal => ("head-feet", "anterior-posterior"),
                Orientation::Coronal => ("head-feet", "left-right"),
            }
        };
        self.state_mut("frequency_direction").options =
            vec![Opt { label: vertical.into(), value: Value::s("anterior-posterior") }, Opt { label: horizontal.into(), value: Value::s("left-right") }];
    }

    fn update_x_y_labels(&mut self) {
        let lr = self.inputs.frequency_direction == "left-right";
        for name in ["FOV_F", "FOV_P", "matrix_F", "matrix_P", "recon_matrix_F", "recon_matrix_P", "voxel_F", "voxel_P", "recon_voxel_F", "recon_voxel_P"] {
            let is_f = name.ends_with("_F");
            let st = self.state_mut(name);
            let axis = if is_f == lr { " x" } else { " y" };
            let base = st.spec.label.trim_end_matches(" x").trim_end_matches(" y");
            st.label = format!("{base}{axis}");
        }
    }

    fn update_shot_label(&mut self) {
        self.shot_label = if !self.protocol.is_radial {
            "shot"
        } else if self.protocol.num_segm == 1 {
            "spoke"
        } else {
            "blade"
        }
        .to_string();
        let l = format!("Displayed {}", self.shot_label);
        self.state_mut("shot").label = l;
    }

    fn update_params(&mut self) {
        for &name in INPUT_NAMES {
            if let Some(v) = self.graph_value(name) {
                self.set_param(name, v);
            }
        }
        let pbw = self.inputs.pixel_bandwidth;
        let m_f = self.protocol.matrix_f as f64;
        self.set_param("FOV_bandwidth", Value::Num(convert::pixel_bw_to_fov_bw(pbw, m_f)));
        self.set_param("FW_shift", Value::Num(convert::pixel_bw_to_shift(pbw, self.inputs.field_strength)));
        for dir in ["F", "P"] {
            let fov = if dir == "F" { self.inputs.fov_f } else { self.inputs.fov_p };
            let (m, rm) =
                if dir == "F" { (self.protocol.matrix_f, self.protocol.recon_matrix_f) } else { (self.protocol.matrix_p, self.protocol.recon_matrix_p) };
            self.set_param(&format!("voxel_{dir}"), Value::Num(fov / m as f64));
            self.set_param(&format!("recon_voxel_{dir}"), Value::Num(fov / rm as f64));
        }
        // label radial trajectory 'Radial' or 'PROPELLER' depending on lines per shot
        let radial_label = if self.protocol.num_segm == 1 { "Radial" } else { "PROPELLER" };
        self.state_mut("trajectory").options =
            vec![Opt { label: "Cartesian".into(), value: Value::s("Cartesian") }, Opt { label: radial_label.into(), value: Value::s("Radial") }];
    }

    fn zerofill_possible(&self) -> bool {
        if self.voxel_size_is_input() {
            return self.num("recon_voxel_F") < self.num("voxel_F") || self.num("recon_voxel_P") < self.num("voxel_P");
        }
        self.num("recon_matrix_F") > self.num("matrix_F") || self.num("recon_matrix_P") > self.num("matrix_P")
    }

    fn update_visibility(&mut self) {
        let (pbw, fovbw, fws, vox, mat) =
            (self.pixel_bw_is_input(), self.fov_bw_is_input(), self.fw_shift_is_input(), self.voxel_size_is_input(), self.matrix_is_input());
        let zf = self.bool("do_zerofill");
        self.set_visibility("pixel_bandwidth", pbw);
        self.set_visibility("FOV_bandwidth", fovbw);
        self.set_visibility("FW_shift", fws);
        for suffix in ["_F", "_P"] {
            self.set_visibility(&format!("voxel{suffix}"), vox);
            self.set_visibility(&format!("recon_voxel{suffix}"), vox && zf);
            self.set_visibility(&format!("matrix{suffix}"), mat);
            self.set_visibility(&format!("recon_matrix{suffix}"), mat && zf);
        }
        let p = self.protocol.clone();
        for name in ["partial_Fourier_factor", "frequency_direction", "phase_oversampling"] {
            self.set_visibility(name, !p.is_radial);
        }
        self.set_visibility("radial_sampling_factor", p.is_radial);
        self.set_visibility("TI", self.inputs.sequence_type == "Inversion Recovery");
        self.set_visibility("FA", self.inputs.sequence_type == "Spoiled Gradient Echo");
        self.set_visibility("turbo_factor", !p.is_gradient_echo);
        self.set_visibility("echo_order", p.echo_order_options.len() > 1);
        self.set_visibility("homodyne", p.num_blank_lines > 0 && !p.is_radial);
        self.set_visibility("apodization_alpha", self.inputs.do_apodize);
        self.set_visibility("kspace_exponent", self.inputs.kspace_mode != "Phase");
        let zfp = self.zerofill_possible();
        self.set_visibility("do_zerofill", zfp);
        self.info.spoke_angle_visible = p.is_radial;
        let is_3d = self.phantom.is_volume_slice();
        for name in ["slice_orientation", "slice_angle_1", "slice_angle_2", "slice_pos_LR", "slice_pos_AP", "slice_pos_HF", "num_slices", "slice_gap"] {
            self.set_visibility(name, is_3d);
        }
        self.set_visibility("displayed_slice", is_3d && self.inputs.num_slices > 1);
    }

    fn update_info_params(&mut self) {
        self.info.fw_shift = formatting::fw_shift(self.num("FW_shift"));
        if self.reference_snr == 0.0 {
            self.set_reference_snr();
        }
        self.update_relative_snr();
        let p = &self.protocol;
        self.info.spoke_angle = formatting::spoke_angle(p.spoke_angle);
        self.info.num_shots = formatting::num_shots(p.num_shots);
        self.info.scantime = formatting::scantime(p.scantime);
        self.info.pixel_bandwidth = formatting::pixel_bandwidth(self.inputs.pixel_bandwidth);
    }

    fn update_relative_snr(&mut self) {
        self.info.relative_snr = formatting::relative_snr(self.protocol.snr / self.reference_snr);
    }

    fn update_rec_acq_ratio(&mut self) {
        self.rec_acq_ratio_f = self.protocol.recon_matrix_f as f64 / self.protocol.matrix_f as f64;
        self.rec_acq_ratio_p = self.protocol.recon_matrix_p as f64 / self.protocol.matrix_p as f64;
    }

    /// Parameters of a group panel, in display order
    pub fn group_params(&self, group: &str) -> Vec<&'static str> {
        self.params.iter().filter(|p| p.spec.group == Some(group)).map(|p| p.spec.name).collect()
    }

    /// Widget type of a parameter
    pub fn widget(&self, name: &str) -> Widget {
        self.state(name).spec.widget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_sync() {
        let c = Controller::new().unwrap();
        assert_eq!(c.num("TE"), 12.0);
        assert_eq!(c.num("TR"), 9000.0);
        assert_eq!(c.str("echo_order"), "Linear");
        assert!(!c.state("TI").visible());
        assert!(c.state("pixel_bandwidth").visible());
        assert_eq!(c.info.scantime, "27 min 0 sec");
        assert_eq!(c.info.relative_snr, "100%");
        assert_eq!(c.rec_acq_ratio_f, 2.0);
        // bounds restrict the TE options to ≥ min TE
        assert!(c.state("TE").options.first().unwrap().value.num() >= c.protocol.min_te);
    }

    #[test]
    fn preset_and_conversions() {
        let mut c = Controller::new().unwrap();
        c.on_change("load_presets", Value::s("T2-FLAIR"));
        assert_eq!(c.str("load_presets"), "T2-FLAIR");
        assert_eq!(c.str("sequence_type"), "Inversion Recovery");
        assert!(c.state("TI").visible());
        assert_eq!(c.num("turbo_factor"), 16.0);
        // changing a preset parameter marks the preset as custom
        c.on_change("NSA", Value::Num(2.0));
        assert_eq!(c.str("load_presets"), params::CUSTOM_PRESET);
        // acquisition matrix change keeps the rec/acq ratio
        let mut c = Controller::new().unwrap();
        c.on_change("matrix_P", Value::Num(100.0));
        assert_eq!(c.num("recon_matrix_P"), 200.0);
        // voxel size style
        c.on_change("parameter_style", Value::s("Voxel size and Fat/water shift"));
        assert!(c.state("voxel_F").visible() && !c.state("matrix_F").visible());
        assert!(c.state("FW_shift").visible());
        // centric echo order sets TE to its minimum
        let mut c = Controller::new().unwrap();
        c.on_change("turbo_factor", Value::Num(8.0));
        c.on_change("echo_order", Value::s("Centric"));
        assert_eq!(c.num("TE"), snap(c.protocol.min_te, params::te_vals(), SnapMode::Ceil).unwrap());
    }

    #[test]
    fn settings_roundtrip() {
        let mut c = Controller::new().unwrap();
        c.on_change("sequence_type", Value::s("Spoiled Gradient Echo"));
        c.on_change("FA", Value::Num(30.0));
        let text = presets::write_settings(&c.settings());
        assert!(text.contains("FA = 30\n"));
        assert!(text.contains("sequence_type = \"Spoiled Gradient Echo\""));
        let mut d = Controller::new().unwrap();
        d.apply_settings(presets::parse_settings(&text).unwrap(), false);
        assert_eq!(d.num("FA"), 30.0);
        assert_eq!(d.str("sequence_type"), "Spoiled Gradient Echo");
    }

    #[test]
    fn restore_settings_keeps_preset_and_slice_position() {
        let mut c = Controller::new().unwrap();
        c.on_change("load_presets", Value::s("T2w-TSE"));
        c.on_change("slice_orientation", Value::s("Sagittal"));
        c.on_change("slice_pos_LR", Value::Num(12.5));
        c.on_change("num_slices", Value::Num(9.0));
        let saved: Vec<(String, Value)> = c.settings().into_iter().map(|(k, v, _)| (k, v)).collect();
        let preset = c.loaded_preset_name.clone();
        let mut d = Controller::new().unwrap();
        d.on_change("load_presets", Value::s("T1w-GRE"));
        d.restore_settings(&preset, saved);
        assert_eq!(d.str("load_presets"), "T2w-TSE");
        assert_eq!(d.str("slice_orientation"), "Sagittal");
        assert_eq!(d.num("slice_pos_LR"), 12.5);
        assert_eq!(d.inputs.num_slices, 9);
        assert_eq!(d.num("TR"), c.num("TR"));
    }

    #[test]
    fn conflicts_preview_and_keep_adjustments() {
        let mut c = Controller::new().unwrap();
        c.reset_to_standard().unwrap();
        assert_eq!(c.str("load_presets"), "T1w-SE");
        assert_eq!(c.num("TR"), 500.0);
        // a TE within the normal range changes nothing else
        assert!(c.preview("TE", &Value::Num(20.0)).is_empty());
        // a long TE needs a longer TR
        let adj = c.preview("TE", &Value::Num(750.0));
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0].name, "TR");
        let new_tr = adj[0].new_value.num();
        assert!(new_tr > 750.0);
        c.change_with_adjustments("TE", Value::Num(750.0), &adj);
        assert_eq!(c.num("TE"), 750.0);
        assert_eq!(c.num("TR"), new_tr);
        // the prolonged TR is kept when TE is shortened again
        c.on_change("TE", Value::Num(20.0));
        assert_eq!(c.num("TR"), new_tr);
        // nothing is changed by previews
        assert_eq!(c.num("TE"), 20.0);
    }

    #[test]
    fn slider_ranges_extend_into_adjusting_values() {
        let mut c = Controller::new().unwrap();
        c.reset_to_standard().unwrap();
        let r = c.slider_range("TE").unwrap();
        let normal_max = c.state("TE").options.last().unwrap().value.num();
        assert!(r.options.last().unwrap().value.num() > normal_max, "TE range is extended");
        assert!(!r.adjusts[0] && *r.adjusts.last().unwrap(), "short TEs are free, the longest adjust TR");
        // the extension starts beyond the normal range
        let first_adj = r.adjusts.iter().position(|&a| a).unwrap();
        assert!(r.options[first_adj].value.num() > normal_max - 1e-9);
        // no extended range for values that would be impossible: TE below the minimum
        assert_eq!(r.options[0].value, c.state("TE").options[0].value);
        assert!(c.slider_range("slice_pos_LR").is_none());
    }
}
