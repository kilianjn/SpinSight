//! Parameter specifications: labels, discrete option lists, defaults and widget types
//! (port of `params.py`).

use std::sync::OnceLock;

use crate::{convert, formatting};

pub const CUSTOM_PRESET: &str = "(custom)";

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
}

impl Value {
    pub fn num(&self) -> f64 {
        match self {
            Value::Num(v) => *v,
            Value::Bool(b) => f64::from(u8::from(*b)),
            Value::Str(s) => s.parse().unwrap_or(f64::NAN),
        }
    }
    pub fn str(&self) -> &str {
        match self {
            Value::Str(s) => s,
            _ => "",
        }
    }
    pub fn bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Num(v) => *v != 0.0,
            Value::Str(_) => false,
        }
    }
    pub fn s(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Opt {
    pub label: String,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Widget {
    Select,
    DiscreteSlider,
    Checkbox,
    FloatSlider,
}

#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: &'static str,
    pub label: &'static str,
    pub widget: Widget,
    pub options: Option<Vec<Opt>>,
    /// bounds of continuous (FloatSlider) parameters
    pub bounds: Option<(f64, f64)>,
    pub step: f64,
    pub default: Value,
    pub precedence: i32,
    pub group: Option<&'static str>,
    /// numeric values are integers (saved as such in settings files)
    pub integer: bool,
}

impl ParamSpec {
    pub fn option_values(&self) -> Vec<f64> {
        self.options.as_ref().map(|o| o.iter().map(|o| o.value.num()).collect()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------
// numpy helpers used to build the discrete value lists exactly like the Python version

/// `np.linspace(start, stop, num)`
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return vec![];
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// `np.geomspace(start, stop, num)`
pub fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let mut v: Vec<f64> = linspace(start.log10(), stop.log10(), num).into_iter().map(|e| 10f64.powf(e)).collect();
    if num > 0 {
        v[0] = start;
    }
    if num > 1 {
        v[num - 1] = stop;
    }
    v
}

/// `sorted({float(f'{v:.2g}') for v in values})`
fn unique_rounded(values: &[f64], sigfigs: usize) -> Vec<f64> {
    let mut v: Vec<f64> = values.iter().map(|&x| formatting::round_sig(x, sigfigs)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// Build options like a Python dict comprehension `{label(v): v for v in values}`:
/// duplicate labels keep their first position but take the last value.
fn dict_options(values: impl IntoIterator<Item = f64>, label: impl Fn(f64) -> String) -> Vec<Opt> {
    let mut opts: Vec<Opt> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for v in values {
        let l = label(v);
        match index.get(&l) {
            Some(&i) => {
                let o: &mut Opt = &mut opts[i];
                o.value = Value::Num(v);
            }
            None => {
                index.insert(l.clone(), opts.len());
                opts.push(Opt { label: l, value: Value::Num(v) });
            }
        }
    }
    opts
}

fn list_options(values: impl IntoIterator<Item = f64>) -> Vec<Opt> {
    values.into_iter().map(|v| Opt { label: format_list_value(v), value: Value::Num(v) }).collect()
}

fn format_list_value(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// options from a list of Python floats (labelled like `str(float)`, e.g. "3.0")
fn float_list_options(values: impl IntoIterator<Item = f64>) -> Vec<Opt> {
    values.into_iter().map(|v| Opt { label: if v.fract() == 0.0 { format!("{v:.1}") } else { format!("{v}") }, value: Value::Num(v) }).collect()
}

fn str_options(values: &[&str]) -> Vec<Opt> {
    values.iter().map(|s| Opt { label: s.to_string(), value: Value::s(s) }).collect()
}

// ---------------------------------------------------------------------------------------------
// value lists

macro_rules! cached {
    ($name:ident, $body:expr) => {
        pub fn $name() -> &'static [f64] {
            static CELL: OnceLock<Vec<f64>> = OnceLock::new();
            CELL.get_or_init(|| $body)
        }
    };
}

cached!(pbw_vals, unique_rounded(&geomspace(125.0, 2e3, 1000), 2));
cached!(tr_vals, unique_rounded(&geomspace(1.0, 1e4, 1000), 2));
cached!(te_vals, unique_rounded(&geomspace(1.0, 1e3, 1000), 2));
cached!(ti_vals, unique_rounded(&geomspace(1e2, 4e3, 1000), 2));
cached!(fov_bw_vals, pbw_vals().iter().map(|&bw| convert::pixel_bw_to_fov_bw(bw, 180.0)).collect());
cached!(fw_shift_vals, pbw_vals().iter().rev().map(|&bw| convert::pixel_bw_to_shift(bw, 1.5)).collect());
cached!(voxel_vals, (16..=600).rev().map(|m| 240.0 / m as f64).collect());
cached!(recon_voxel_vals, (16..=1200).rev().map(|m| 240.0 / m as f64).collect());
cached!(partial_fourier_vals, linspace(0.6, 1.0, 41));
cached!(phase_oversampling_vals, linspace(0.0, 1.0, 101));
cached!(radial_sampling_vals, linspace(0.1, 4.0, 391));
cached!(slice_thickness_vals, linspace(0.5, 10.0, 96));

/// Values of the slice thickness options (as used by the slice thickness bounds)
pub fn slice_thickness_option_values() -> Vec<f64> {
    param("slice_thickness").option_values()
}

pub fn pixel_bandwidth_option_values() -> Vec<f64> {
    param("pixel_bandwidth").option_values()
}

/// The largest phase oversampling factor (`MAX_PHASE_OVERSAMPLING_FACTOR`)
pub fn max_phase_oversampling_factor() -> f64 {
    *phase_oversampling_vals().last().unwrap()
}

// ---------------------------------------------------------------------------------------------
// snapping

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapMode {
    Nearest,
    Ceil,
    Floor,
}

/// Snap `value` to one of `values` (`param_utils.snap`)
pub fn snap(value: f64, values: &[f64], mode: SnapMode) -> Option<f64> {
    if let Some(v) = values.iter().find(|&&v| v == value) {
        return Some(*v);
    }
    match mode {
        SnapMode::Nearest => {
            let mut best: Option<f64> = None;
            for &v in values {
                if best.is_none_or(|b| (v - value).abs() < (b - value).abs()) {
                    best = Some(v);
                }
            }
            best
        }
        SnapMode::Ceil => values.iter().copied().filter(|&v| v >= value).fold(None, |m, v| Some(m.map_or(v, |m: f64| m.min(v)))),
        SnapMode::Floor => values.iter().copied().filter(|&v| v <= value).fold(None, |m, v| Some(m.map_or(v, |m: f64| m.max(v)))),
    }
}

// ---------------------------------------------------------------------------------------------
// the parameter table

static PARAMS: OnceLock<Vec<ParamSpec>> = OnceLock::new();

/// Must be called once at startup (before `params()`), with the available phantoms and presets.
pub fn init(phantom_names: &[String], preset_names: &[String]) {
    let _ = PARAMS.set(build(phantom_names, preset_names));
}

pub fn params() -> &'static [ParamSpec] {
    PARAMS.get_or_init(|| {
        let phantoms: Vec<String> = crate::phantom::builtin_phantom_names().iter().map(|s| s.to_string()).collect();
        let presets: Vec<String> = crate::presets::builtin_preset_names().iter().map(|s| s.to_string()).collect();
        build(&phantoms, &presets)
    })
}

pub fn param(name: &str) -> &'static ParamSpec {
    params().iter().find(|p| p.name == name).unwrap_or_else(|| panic!("unknown parameter {name}"))
}

pub fn try_param(name: &str) -> Option<&'static ParamSpec> {
    params().iter().find(|p| p.name == name)
}

#[allow(clippy::too_many_arguments)]
fn spec(
    name: &'static str,
    label: &'static str,
    widget: Widget,
    options: Option<Vec<Opt>>,
    default: Value,
    precedence: i32,
    group: Option<&'static str>,
    integer: bool,
) -> ParamSpec {
    let mut default = default;
    if let (Some(opts), Value::Num(d)) = (&options, &default) {
        // assert default is in objects
        let values: Vec<f64> = opts.iter().map(|o| o.value.num()).collect();
        if let Some(v) = snap(*d, &values, SnapMode::Nearest) {
            default = Value::Num(v);
        }
    }
    ParamSpec { name, label, widget, options, bounds: None, step: 0.0, default, precedence, group, integer }
}

fn number(
    name: &'static str,
    label: &'static str,
    bounds: Option<(f64, f64)>,
    step: f64,
    default: f64,
    precedence: i32,
    group: Option<&'static str>,
) -> ParamSpec {
    ParamSpec { name, label, widget: Widget::FloatSlider, options: None, bounds, step, default: Value::Num(default), precedence, group, integer: false }
}

fn boolean(name: &'static str, label: &'static str, default: bool, precedence: i32, group: Option<&'static str>) -> ParamSpec {
    ParamSpec {
        name,
        label,
        widget: Widget::Checkbox,
        options: None,
        bounds: None,
        step: 0.0,
        default: Value::Bool(default),
        precedence,
        group,
        integer: false,
    }
}

fn build(phantom_names: &[String], preset_names: &[String]) -> Vec<ParamSpec> {
    use Widget::*;
    const SETTINGS: Option<&str> = Some("Settings");
    const SEQUENCE: Option<&str> = Some("Sequence");
    const CONTRAST: Option<&str> = Some("Contrast");
    const GEOMETRY: Option<&str> = Some("Geometry");
    const KSPACE: Option<&str> = Some("k-space processing");

    let mut presets = vec![CUSTOM_PRESET.to_string()];
    presets.extend(preset_names.iter().cloned());
    let preset_opts = presets.iter().map(|s| Opt { label: s.clone(), value: Value::Str(s.clone()) }).collect();
    let phantom_opts = phantom_names.iter().map(|s| Opt { label: s.clone(), value: Value::Str(s.clone()) }).collect();

    let fov_opts = || dict_options((100..=600).map(f64::from), formatting::fov);
    let voxel_opts = || dict_options(voxel_vals().iter().copied(), formatting::voxel_size);
    let recon_voxel_opts = || dict_options(recon_voxel_vals().iter().copied(), formatting::voxel_size);
    let matrix_opts = || list_options((16..=600).map(f64::from));
    let recon_matrix_opts = || list_options((16..=1200).map(f64::from));

    vec![
        spec("load_presets", "Load preset parameters", Select, Some(preset_opts), Value::s(CUSTOM_PRESET), 1, SETTINGS, false),
        spec("object", "Phantom object", Select, Some(phantom_opts), Value::s("brain"), 2, SETTINGS, false),
        spec("field_strength", "B0 field strength [T]", Select, Some(float_list_options([1.5, 3.0])), Value::Num(1.5), 3, SETTINGS, false),
        spec(
            "parameter_style",
            "Parameter style",
            Select,
            Some(str_options(&["Matrix and Pixel BW", "Voxel size and Fat/water shift", "Matrix and FOV BW"])),
            Value::s("Matrix and Pixel BW"),
            4,
            SETTINGS,
            false,
        ),
        number("min_voxel_size", "Phantom resolution limit (to bound computation time) [mm]", None, 0.0, 0.5, -1, None),
        number("noise_gain", "Scaling factor for noise standard deviation", None, 0.0, 3.0, -1, None),
        spec(
            "sequence_type",
            "Pulse sequence",
            Select,
            Some(str_options(&["Spin Echo", "Spoiled Gradient Echo", "Inversion Recovery"])),
            Value::s("Spin Echo"),
            1,
            SEQUENCE,
            false,
        ),
        spec(
            "pixel_bandwidth",
            "Bandwidth",
            DiscreteSlider,
            Some(dict_options(pbw_vals().iter().copied(), formatting::pixel_bandwidth)),
            Value::Num(480.0),
            2,
            SEQUENCE,
            false,
        ),
        spec(
            "FOV_bandwidth",
            "FOV bandwidth",
            DiscreteSlider,
            Some(dict_options(fov_bw_vals().iter().copied(), formatting::fov_bandwidth)),
            Value::Num(43.2),
            -2,
            SEQUENCE,
            false,
        ),
        spec(
            "FW_shift",
            "Fat/water shift",
            DiscreteSlider,
            Some(dict_options(fw_shift_vals().iter().copied(), formatting::fw_shift)),
            Value::Num(0.45),
            -2,
            SEQUENCE,
            false,
        ),
        spec("NSA", "NSA", DiscreteSlider, Some(list_options((1..=16).map(f64::from))), Value::Num(1.0), 3, SEQUENCE, true),
        spec(
            "partial_Fourier_factor",
            "Partial Fourier factor",
            DiscreteSlider,
            Some(dict_options(partial_fourier_vals().iter().copied(), formatting::percentage)),
            Value::Num(1.0),
            5,
            SEQUENCE,
            false,
        ),
        spec("turbo_factor", "Turbo factor", DiscreteSlider, Some(list_options((1..=64).map(f64::from))), Value::Num(1.0), 6, SEQUENCE, true),
        spec("EPI_factor", "EPI factor", DiscreteSlider, Some(list_options((1..=64).map(f64::from))), Value::Num(1.0), 7, SEQUENCE, true),
        spec("echo_order", "Echo order", Select, Some(str_options(&["Linear", "Centric", "Flexible"])), Value::s("Flexible"), -8, SEQUENCE, false),
        boolean("FatSat", "Fat saturation", false, 1, CONTRAST),
        spec("TR", "TR", DiscreteSlider, Some(dict_options(tr_vals().iter().copied(), formatting::ts)), Value::Num(9000.0), 2, CONTRAST, false),
        spec("TE", "TE", DiscreteSlider, Some(dict_options(te_vals().iter().copied(), formatting::ts)), Value::Num(12.0), 3, CONTRAST, false),
        spec("TI", "TI", DiscreteSlider, Some(dict_options(ti_vals().iter().copied(), formatting::ts)), Value::Num(40.0), -4, CONTRAST, false),
        spec("FA", "Flip angle", DiscreteSlider, Some(dict_options((1..=90).map(f64::from), formatting::flip_angle)), Value::Num(90.0), -5, CONTRAST, true),
        spec("trajectory", "k-space trajectory", Select, Some(str_options(&["Cartesian", "Radial"])), Value::s("Cartesian"), 1, GEOMETRY, false),
        spec(
            "frequency_direction",
            "Frequency encoding direction",
            Select,
            Some(str_options(&["anterior-posterior", "left-right"])),
            Value::s("anterior-posterior"),
            1,
            GEOMETRY,
            false,
        ),
        spec("FOV_P", "FOV x", DiscreteSlider, Some(fov_opts()), Value::Num(240.0), 2, GEOMETRY, true),
        spec("FOV_F", "FOV y", DiscreteSlider, Some(fov_opts()), Value::Num(240.0), 2, GEOMETRY, true),
        spec(
            "phase_oversampling",
            "Phase oversampling",
            DiscreteSlider,
            Some(dict_options(phase_oversampling_vals().iter().copied(), formatting::percentage)),
            Value::Num(0.0),
            3,
            GEOMETRY,
            false,
        ),
        spec(
            "radial_sampling_factor",
            "Radial sampling factor",
            DiscreteSlider,
            Some(dict_options(radial_sampling_vals().iter().copied(), formatting::percentage)),
            Value::Num(1.0),
            -3,
            GEOMETRY,
            false,
        ),
        spec("matrix_P", "Acquisition matrix x", DiscreteSlider, Some(matrix_opts()), Value::Num(180.0), 4, GEOMETRY, true),
        spec("matrix_F", "Acquisition matrix y", DiscreteSlider, Some(matrix_opts()), Value::Num(180.0), 4, GEOMETRY, true),
        spec("voxel_P", "Voxel size x", DiscreteSlider, Some(voxel_opts()), Value::Num(1.33), -4, GEOMETRY, false),
        spec("voxel_F", "Voxel size y", DiscreteSlider, Some(voxel_opts()), Value::Num(1.33), -4, GEOMETRY, false),
        spec("recon_matrix_P", "Reconstruction matrix x", DiscreteSlider, Some(recon_matrix_opts()), Value::Num(360.0), 5, GEOMETRY, true),
        spec("recon_matrix_F", "Reconstruction matrix y", DiscreteSlider, Some(recon_matrix_opts()), Value::Num(360.0), 5, GEOMETRY, true),
        spec("recon_voxel_P", "Reconstructed voxel size x", DiscreteSlider, Some(recon_voxel_opts()), Value::Num(0.667), -5, GEOMETRY, false),
        spec("recon_voxel_F", "Reconstructed voxel size y", DiscreteSlider, Some(recon_voxel_opts()), Value::Num(0.667), -5, GEOMETRY, false),
        spec(
            "slice_thickness",
            "Slice thickness",
            DiscreteSlider,
            Some(dict_options(slice_thickness_vals().iter().copied(), formatting::slice_thickness)),
            Value::Num(3.0),
            6,
            GEOMETRY,
            false,
        ),
        number("radial_FOV_oversampling", "Radial FOV oversampling factor", Some((1.0, 2.0)), 0.01, 2.0, -1, None),
        boolean("show_FOV", "Show FOV", false, 1, None),
        boolean("show_TE_span", "Show TE interval", false, -1, None),
        boolean("show_TR_span", "Show TR interval", false, -1, None),
        boolean("show_TI_span", "Show TI interval", false, -1, None),
        spec("reference_tissue", "Reference tissue", Select, Some(vec![]), Value::s("Gray matter"), 2, None, false),
        spec("image_mode", "Image mode", Select, Some(str_options(&crate::constants::Operator::ALL)), Value::s("Magnitude"), 3, None, false),
        boolean("show_processed_kspace", "Show processed k-space", false, 1, None),
        number("kspace_exponent", "k-space exponent", Some((0.1, 1.0)), 0.01, 0.20, 2, None),
        spec("kspace_mode", "k-space mode", Select, Some(str_options(&crate::constants::Operator::ALL)), Value::s("Magnitude"), 3, None, false),
        boolean("homodyne", "Homodyne", true, -1, KSPACE),
        boolean("do_apodize", "Apodization", true, 2, KSPACE),
        number("apodization_alpha", "Apodization alpha", Some((0.01, 1.0)), 0.01, 0.25, 3, KSPACE),
        boolean("do_zerofill", "Zerofill", true, 4, KSPACE),
        spec(
            "shot",
            "Displayed shot",
            DiscreteSlider,
            Some((0..1000).map(|i| Opt { label: format!("{}", i + 1), value: Value::Num(f64::from(i)) }).collect()),
            Value::Num(0.0),
            1,
            None,
            true,
        ),
        number("signal_exponent", "Signal exponent", Some((0.1, 1.0)), 0.01, 0.50, 2, None),
    ]
}

/// The names of the parameter groups shown as panels, in display order
pub const GROUPS: [&str; 5] = ["Settings", "Sequence", "Contrast", "Geometry", "k-space processing"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_lists() {
        assert_eq!(pbw_vals().len(), 99);
        assert_eq!(pbw_vals()[0], 120.0);
        assert_eq!(*pbw_vals().last().unwrap(), 2000.0);
        assert_eq!(te_vals().len(), 271);
        assert_eq!(tr_vals().len(), 361);
        assert_eq!(ti_vals().len(), 121);
        assert_eq!(param("TI").default, Value::Num(100.0));
        assert_eq!(param("voxel_P").options.as_ref().unwrap().len(), 535);
        assert_eq!(param("recon_voxel_P").options.as_ref().unwrap().len(), 735);
    }

    #[test]
    fn snapping() {
        let v = [1.0, 2.0, 4.0];
        assert_eq!(snap(2.9, &v, SnapMode::Nearest), Some(2.0));
        assert_eq!(snap(2.1, &v, SnapMode::Ceil), Some(4.0));
        assert_eq!(snap(5.0, &v, SnapMode::Ceil), None);
        assert_eq!(snap(1.5, &v, SnapMode::Floor), Some(1.0));
    }
}
