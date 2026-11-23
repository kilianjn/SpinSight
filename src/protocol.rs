//! Derived sequence parameters, pulse sequence timing and parameter bounds.
//!
//! This is a port of the scalar part of the Python computation graph
//! (`nodes/internal_input_params.py`, `helpers.py`, `phase_encoding_order.py`,
//! `setup_sequence_objects.py`, `sequence_timing.py`, `place_sequence_objects.py`,
//! `param_bounds.py`, `SNR_and_scantime.py`). It is cheap to evaluate (well below a
//! millisecond), so it runs synchronously on every parameter change; the heavy k-space and
//! image computations are done by `imaging.rs` in a background thread.

use std::collections::BTreeSet;
use std::f64::consts::PI;

use crate::constants::*;
use crate::params::{self, snap, SnapMode, Value};
use crate::phantom::Phantom;
use crate::recon::get_k_axis;
use crate::sequence::{self, get_adc, get_gradient, get_rf, place, rescale_gradient, GradSpec, Waveform};

/// Python's `round()` (round half to even)
pub fn pyround(x: f64) -> f64 {
    x.round_ties_even()
}

/// Graph input nodes (the "prescribed" parameter values)
#[derive(Debug, Clone, PartialEq)]
pub struct Inputs {
    pub object: String,
    pub field_strength: f64,
    pub min_voxel_size: f64,
    pub noise_gain: f64,
    /// kind of noise: thermal (white Gaussian) with optional spikes / RF interference
    pub noise_type: String,
    /// strength of spikes / RF interference (1 = standard)
    pub interference_level: f64,
    pub sequence_type: String,
    pub pixel_bandwidth: f64,
    pub nsa: f64,
    pub partial_fourier_factor: f64,
    pub turbo_factor: usize,
    pub epi_factor: usize,
    pub echo_order: String,
    pub fatsat: bool,
    pub tr: f64,
    /// `None` = 'min'
    pub te: Option<f64>,
    pub ti: f64,
    pub fa: f64,
    pub trajectory: String,
    pub frequency_direction: String,
    pub fov_p: f64,
    pub fov_f: f64,
    pub phase_oversampling: f64,
    pub radial_sampling_factor: f64,
    pub matrix_p: usize,
    pub matrix_f: usize,
    pub recon_matrix_p: usize,
    pub recon_matrix_f: usize,
    pub slice_thickness: f64,
    pub radial_fov_oversampling: f64,
    pub show_fov: bool,
    pub show_te_span: bool,
    pub show_tr_span: bool,
    pub show_ti_span: bool,
    pub reference_tissue: String,
    pub image_mode: String,
    pub show_processed_kspace: bool,
    pub kspace_exponent: f64,
    pub kspace_mode: String,
    pub homodyne: bool,
    pub do_apodize: bool,
    pub apodization_alpha: f64,
    pub do_zerofill: bool,
    pub shot: usize,
    pub signal_exponent: f64,
    pub slice_orientation: String,
    pub slice_angle_1: f64,
    pub slice_angle_2: f64,
    /// stack centre (RAS) [mm]
    pub slice_pos: [f64; 3],
    pub num_slices: usize,
    pub slice_gap: f64,
    pub displayed_slice: usize,
    pub constant_voxel_bounds: bool,
    pub constant_fov_bw_bounds: bool,
}

/// Names of all graph input nodes, i.e. parameters that can be set directly
pub const INPUT_NAMES: &[&str] = &[
    "object",
    "field_strength",
    "min_voxel_size",
    "noise_gain",
    "noise_type",
    "interference_level",
    "sequence_type",
    "pixel_bandwidth",
    "FOV_bandwidth",
    "NSA",
    "partial_Fourier_factor",
    "turbo_factor",
    "EPI_factor",
    "echo_order",
    "FatSat",
    "TR",
    "TE",
    "TI",
    "FA",
    "trajectory",
    "frequency_direction",
    "FOV_P",
    "FOV_F",
    "phase_oversampling",
    "radial_sampling_factor",
    "matrix_P",
    "matrix_F",
    "recon_matrix_P",
    "recon_matrix_F",
    "slice_thickness",
    "radial_FOV_oversampling",
    "show_FOV",
    "show_TE_span",
    "show_TR_span",
    "show_TI_span",
    "reference_tissue",
    "image_mode",
    "show_processed_kspace",
    "kspace_exponent",
    "kspace_mode",
    "homodyne",
    "do_apodize",
    "apodization_alpha",
    "do_zerofill",
    "shot",
    "signal_exponent",
    "slice_orientation",
    "slice_angle_1",
    "slice_angle_2",
    "slice_pos_LR",
    "slice_pos_AP",
    "slice_pos_HF",
    "num_slices",
    "slice_gap",
    "displayed_slice",
];

impl Default for Inputs {
    fn default() -> Self {
        let d = |n: &str| params::param(n).default.clone();
        let num = |n: &str| d(n).num();
        let s = |n: &str| d(n).str().to_string();
        let b = |n: &str| d(n).bool();
        let u = |n: &str| num(n).round() as usize;
        Inputs {
            object: s("object"),
            field_strength: num("field_strength"),
            min_voxel_size: num("min_voxel_size"),
            noise_gain: num("noise_gain"),
            noise_type: s("noise_type"),
            interference_level: num("interference_level"),
            sequence_type: s("sequence_type"),
            pixel_bandwidth: num("pixel_bandwidth"),
            nsa: num("NSA"),
            partial_fourier_factor: num("partial_Fourier_factor"),
            turbo_factor: u("turbo_factor"),
            epi_factor: u("EPI_factor"),
            echo_order: s("echo_order"),
            fatsat: b("FatSat"),
            tr: num("TR"),
            te: Some(num("TE")),
            ti: num("TI"),
            fa: num("FA"),
            trajectory: s("trajectory"),
            frequency_direction: s("frequency_direction"),
            fov_p: num("FOV_P"),
            fov_f: num("FOV_F"),
            phase_oversampling: num("phase_oversampling"),
            radial_sampling_factor: num("radial_sampling_factor"),
            matrix_p: u("matrix_P"),
            matrix_f: u("matrix_F"),
            recon_matrix_p: u("recon_matrix_P"),
            recon_matrix_f: u("recon_matrix_F"),
            slice_thickness: num("slice_thickness"),
            radial_fov_oversampling: num("radial_FOV_oversampling"),
            show_fov: b("show_FOV"),
            show_te_span: false,
            show_tr_span: false,
            show_ti_span: false,
            reference_tissue: s("reference_tissue"),
            image_mode: s("image_mode"),
            show_processed_kspace: b("show_processed_kspace"),
            kspace_exponent: num("kspace_exponent"),
            kspace_mode: s("kspace_mode"),
            homodyne: b("homodyne"),
            do_apodize: b("do_apodize"),
            apodization_alpha: num("apodization_alpha"),
            do_zerofill: b("do_zerofill"),
            shot: u("shot"),
            signal_exponent: num("signal_exponent"),
            slice_orientation: s("slice_orientation"),
            slice_angle_1: num("slice_angle_1"),
            slice_angle_2: num("slice_angle_2"),
            slice_pos: [num("slice_pos_LR"), num("slice_pos_AP"), num("slice_pos_HF")],
            num_slices: u("num_slices"),
            slice_gap: num("slice_gap"),
            displayed_slice: u("displayed_slice"),
            constant_voxel_bounds: false,
            constant_fov_bw_bounds: false,
        }
    }
}

impl Inputs {
    /// Set an input node (`Graph.update_inputs`). Unknown/derived names are ignored and reported.
    pub fn set(&mut self, name: &str, v: &Value) -> Result<(), String> {
        let num = || -> Result<f64, String> {
            match v {
                Value::Num(x) => Ok(*x),
                Value::Bool(b) => Ok(f64::from(u8::from(*b))),
                Value::Str(s) => s.parse().map_err(|_| format!("{name}: expected a number, got {s:?}")),
            }
        };
        let uint = || -> Result<usize, String> { num().map(|x| x.round().max(0.0) as usize) };
        let s = || v.str().to_string();
        let b = || v.bool();
        match name {
            "object" => self.object = s(),
            "field_strength" => self.field_strength = num()?,
            "min_voxel_size" => self.min_voxel_size = num()?,
            "noise_gain" => self.noise_gain = num()?,
            "noise_type" => self.noise_type = s(),
            "interference_level" => self.interference_level = num()?,
            "sequence_type" => self.sequence_type = s(),
            "pixel_bandwidth" => self.pixel_bandwidth = num()?,
            "FOV_bandwidth" => {} // unused input node
            "NSA" => self.nsa = num()?,
            "partial_Fourier_factor" => self.partial_fourier_factor = num()?,
            "turbo_factor" => self.turbo_factor = uint()?.max(1),
            "EPI_factor" => self.epi_factor = uint()?.max(1),
            "echo_order" => self.echo_order = s(),
            "FatSat" => self.fatsat = b(),
            "TR" => self.tr = num()?,
            "TE" => {
                self.te = match v {
                    Value::Str(x) if x == "min" => None,
                    _ => Some(num()?),
                }
            }
            "TI" => self.ti = num()?,
            "FA" => self.fa = num()?,
            "trajectory" => self.trajectory = s(),
            "frequency_direction" => self.frequency_direction = s(),
            "FOV_P" => self.fov_p = num()?,
            "FOV_F" => self.fov_f = num()?,
            "phase_oversampling" => self.phase_oversampling = num()?,
            "radial_sampling_factor" => self.radial_sampling_factor = num()?,
            "matrix_P" => self.matrix_p = uint()?.max(1),
            "matrix_F" => self.matrix_f = uint()?.max(1),
            "recon_matrix_P" => self.recon_matrix_p = uint()?.max(1),
            "recon_matrix_F" => self.recon_matrix_f = uint()?.max(1),
            "slice_thickness" => self.slice_thickness = num()?,
            "radial_FOV_oversampling" => self.radial_fov_oversampling = num()?,
            "show_FOV" => self.show_fov = b(),
            "show_TE_span" => self.show_te_span = b(),
            "show_TR_span" => self.show_tr_span = b(),
            "show_TI_span" => self.show_ti_span = b(),
            "reference_tissue" => self.reference_tissue = s(),
            "image_mode" => self.image_mode = s(),
            "show_processed_kspace" => self.show_processed_kspace = b(),
            "kspace_exponent" => self.kspace_exponent = num()?,
            "kspace_mode" => self.kspace_mode = s(),
            "homodyne" => self.homodyne = b(),
            "do_apodize" => self.do_apodize = b(),
            "apodization_alpha" => self.apodization_alpha = num()?,
            "do_zerofill" => self.do_zerofill = b(),
            "shot" => self.shot = uint()?,
            "signal_exponent" => self.signal_exponent = num()?,
            "slice_orientation" => self.slice_orientation = s(),
            "slice_angle_1" => self.slice_angle_1 = num()?,
            "slice_angle_2" => self.slice_angle_2 = num()?,
            "slice_pos_LR" => self.slice_pos[0] = num()?,
            "slice_pos_AP" => self.slice_pos[1] = num()?,
            "slice_pos_HF" => self.slice_pos[2] = num()?,
            "num_slices" => self.num_slices = uint()?.max(1),
            "slice_gap" => self.slice_gap = num()?,
            "displayed_slice" => self.displayed_slice = uint()?,
            "constant_voxel_bounds" => self.constant_voxel_bounds = b(),
            "constant_FOV_BW_bounds" => self.constant_fov_bw_bounds = b(),
            _ => return Err(format!("Input node {name} not found in graph")),
        }
        Ok(())
    }
}

impl Inputs {
    /// The planned slice stack (for 3D phantoms)
    pub fn stack(&self) -> crate::geometry::StackSpec {
        crate::geometry::StackSpec {
            orientation: crate::geometry::Orientation::from_name(&self.slice_orientation),
            angle1: self.slice_angle_1,
            angle2: self.slice_angle_2,
            center: self.slice_pos,
            thickness: self.slice_thickness,
            num_slices: self.num_slices.max(1),
            gap: self.slice_gap,
        }
    }

    /// Geometry of the displayed slice of the stack
    pub fn displayed_slice_geometry(&self) -> crate::geometry::SliceGeometry {
        self.stack().slice(self.displayed_slice.min(self.num_slices.max(1) - 1))
    }
}

/// Parameter bounds (`MinMax`); `None` = unbounded
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MinMax {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl MinMax {
    pub fn new(min: f64, max: f64) -> Self {
        MinMax { min: Some(min), max: Some(max) }
    }
    pub fn min_only(min: f64) -> Self {
        MinMax { min: Some(min), max: None }
    }
}

/// All sequence objects of one board, for plotting
#[derive(Debug, Clone, Default)]
pub struct Boards {
    pub frequency: Vec<Waveform>,
    pub phase: Vec<Waveform>,
    pub slice: Vec<Waveform>,
    pub rf: Vec<Waveform>,
    pub adc: Vec<Waveform>,
}

/// The evaluated protocol: derived parameters, sequence objects, bounds and SNR/scan time
#[derive(Debug, Clone)]
pub struct Protocol {
    pub is_radial: bool,
    pub is_gradient_echo: bool,
    pub freq_dir: usize,
    pub phase_dir: usize,
    pub fov: [f64; 2],
    pub matrix: [usize; 2],
    pub matrix_f: usize,
    pub matrix_p: usize,
    pub recon_matrix_f: usize,
    pub recon_matrix_p: usize,
    pub recon_matrix: [usize; 2],
    pub turbo_factor: usize,
    pub epi_factor: usize,
    pub echo_order_options: Vec<String>,
    pub echo_order: String,
    pub num_segm: usize,
    pub num_shots: usize,
    pub num_blades: usize,
    pub num_sampled_phase_encodes: usize,
    pub num_phase_encodes_full: usize,
    pub num_blank_lines: usize,
    pub num_sym_segm: usize,
    pub k0_index: (usize, usize, bool),
    pub k_phase_axis: Vec<f64>,
    pub k_read_axis: Vec<f64>,
    /// pe_table[(shot * turbo + rf) * epi + gr]
    pub pe_table: Vec<usize>,
    pub shot: usize,
    pub tr: f64,
    pub te: f64,
    pub min_te: f64,
    pub max_te: f64,
    pub min_tr: f64,
    pub readtrain_spacing: f64,
    pub readtrain_center_time: Vec<f64>,
    pub sequence_start: f64,
    pub boards: Boards,
    /// readouts[rf][gr] (placed)
    pub readouts: Vec<Vec<Waveform>>,
    pub rf_refocusing: Option<Vec<Waveform>>,
    // bounds
    pub shot_bounds: MinMax,
    pub turbo_factor_bounds: MinMax,
    pub max_epi_factor: usize,
    pub matrix_f_bounds: MinMax,
    pub matrix_p_bounds: MinMax,
    pub recon_matrix_f_bounds: MinMax,
    pub recon_matrix_p_bounds: MinMax,
    pub fov_f_bounds: MinMax,
    pub fov_p_bounds: MinMax,
    pub tr_bounds: MinMax,
    pub ti_bounds: Option<MinMax>,
    pub te_bounds: MinMax,
    pub pixel_bandwidth_bounds: MinMax,
    pub slice_thickness_bounds: MinMax,
    pub min_slice_thickness: Option<f64>,
    pub max_readout_area: f64,
    pub max_phaser_area: f64,
    // contrast / SNR
    pub tissues: Vec<String>,
    pub reference_tissue: String,
    pub noise_std: f64,
    pub snr: f64,
    pub scantime: f64,
    /// slices per TR and number of concatenations of a multi-slice acquisition
    pub slices_per_tr: usize,
    pub concatenations: usize,
    pub spoke_angle: f64,
    pub k_angles: Vec<f64>,
    pub sampling_time: Vec<f64>,
    /// (rf_echo, gr_echo) of the first occurrence of each sampled phase encode in the PE table
    pub pe_echo: Vec<(usize, usize)>,
    pub spin_echoes: Vec<f64>,
}

impl Protocol {
    pub fn pe(&self, shot: usize, rf: usize, gr: usize) -> usize {
        self.pe_table[(shot * self.turbo_factor + rf) * self.epi_factor + gr]
    }
}

fn readtrain_shift(gr_echo_spacing: f64, k0_gr_echo_index: f64, num_gr_echoes: usize) -> f64 {
    gr_echo_spacing * (k0_gr_echo_index - (num_gr_echoes as f64 - 1.0) / 2.0)
}

fn get_readtrain_spacing(te: f64, num_gr_echoes: usize, gr_echo_spacing: f64, k0_gr: usize, k0_rf: usize) -> f64 {
    let readtrain_center = te - readtrain_shift(gr_echo_spacing, k0_gr as f64, num_gr_echoes);
    readtrain_center / (1.0 + k0_rf as f64)
}

#[allow(clippy::too_many_arguments)]
fn min_readtrain_spacing(
    gr_echo_spacing: f64,
    k0_gr: usize,
    num_gr_echoes: usize,
    is_gradient_echo: bool,
    min_rf_to_readtrain_center: f64,
    num_rf_echoes: usize,
    min_refocusing_time: f64,
) -> f64 {
    if is_gradient_echo {
        return min_rf_to_readtrain_center;
    }
    let mut min_rf_to_spin_echo = min_rf_to_readtrain_center;
    let te_shift = readtrain_shift(gr_echo_spacing, k0_gr as f64, num_gr_echoes);
    if num_rf_echoes == 1 {
        min_rf_to_spin_echo += te_shift;
    }
    let min_first_spin_echo = min_refocusing_time.max(min_rf_to_spin_echo) * 2.0;
    let mut min_first_readtrain_center = min_first_spin_echo;
    if num_rf_echoes == 1 {
        min_first_readtrain_center -= te_shift;
    }
    min_first_readtrain_center
}

/// Temporal order of k-space segments with flexible k-space center (`flexible_segment_order`)
pub fn flexible_segment_order(num_segm: i64, num_sym: i64, k0_index: i64) -> Vec<i64> {
    if k0_index > (num_segm - 1).div_euclid(2) {
        let mut v = flexible_segment_order(num_segm, num_sym, num_segm - 1 - k0_index);
        v.reverse();
        return v;
    }
    let num_linear = (2 * k0_index + 2 - num_sym.rem_euclid(2)).min(k0_index + (num_sym + 1).div_euclid(2));
    let linear_start = num_segm - (k0_index + (num_sym + 1).div_euclid(2));
    let linear_end = linear_start + num_linear;
    let mut linear: Vec<i64> = (linear_start..linear_end).collect();
    let num_asym = num_segm - num_sym;
    let mut left_pivot: Vec<i64> = (num_asym..linear_start).rev().collect();
    let mut right_pivot: Vec<i64> = (linear_end..num_segm).collect();
    if linear_start >= num_asym {
        linear.reverse();
        std::mem::swap(&mut left_pivot, &mut right_pivot);
    }
    let pivoting: Vec<i64> = left_pivot.iter().zip(right_pivot.iter()).flat_map(|(a, b)| [*a, *b]).collect();
    let mut out = linear.clone();
    out.extend(&pivoting);
    if let Some(&m) = linear.iter().chain(pivoting.iter()).min() {
        out.extend((0..m).rev());
    }
    out
}

/// `np.roots([a, b, c])` as (re, im) pairs
fn quadratic_roots(a: f64, b: f64, c: f64) -> Vec<(f64, f64)> {
    if a == 0.0 {
        if b == 0.0 {
            return vec![];
        }
        return vec![(-c / b, 0.0)];
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        // numerically stable form
        let q = -0.5 * (b + b.signum().max(0.0).mul_add(2.0, -1.0) * sq);
        let q = if q == 0.0 { -0.5 * (b - sq) } else { q };
        let r1 = q / a;
        let r2 = if q != 0.0 { c / q } else { -b / a - r1 };
        vec![(r1.max(r2), 0.0), (r1.min(r2), 0.0)]
    } else {
        let re = -b / (2.0 * a);
        let im = (-disc).sqrt() / (2.0 * a);
        vec![(re, im.abs()), (re, -im.abs())]
    }
}

/// numpy's lexicographic complex comparison `z > 0`
fn gt0(z: (f64, f64)) -> bool {
    z.0 > 0.0 || (z.0 == 0.0 && z.1 > 0.0)
}

fn pd_and_t1w(pd: f64, t1: f64, sequence_type: &str, tr: f64, ti: f64, fa: f64) -> f64 {
    let e1 = (-tr / t1).exp();
    match sequence_type {
        // Bernstein eq. 14.57 (assuming (TE_1)/2 << TR)
        "Spin Echo" => pd * (1.0 - e1),
        // Bernstein eq. 14.8
        "Spoiled Gradient Echo" => pd * fa.to_radians().sin() * (1.0 - e1) / (1.0 - fa.to_radians().cos() * e1),
        // Bernstein eq. 14.32
        "Inversion Recovery" => pd * (1.0 - 2.0 * (-ti / t1).exp() + e1),
        _ => pd * (1.0 - e1),
    }
}

/// PD- and T1-weighting of a tissue or fat resonance component (`get_PD_and_T1w`)
pub fn component_pd_t1w(component: &str, sequence_type: &str, tr: f64, ti: f64, fa: f64, b0: f64) -> f64 {
    let fi = field_index(b0);
    if let Some(r) = FAT_RESONANCES.iter().find(|r| r.name == component) {
        pd_and_t1w(r.pd, r.t1[fi], sequence_type, tr, ti, fa)
    } else if let Some(t) = tissue(component) {
        pd_and_t1w(t.pd, t.t1[fi], sequence_type, tr, ti, fa)
    } else {
        0.0
    }
}

fn suffix_rf(turbo: usize, rf: usize) -> String {
    if turbo > 1 {
        format!(" {}", rf + 1)
    } else {
        String::new()
    }
}

fn suffix_rf_gr(turbo: usize, epi: usize, rf: usize, gr: usize) -> String {
    let mut s = String::new();
    if turbo > 1 || epi > 1 {
        s.push(' ');
    }
    if turbo > 1 {
        s.push_str(&(rf + 1).to_string());
    }
    if turbo > 1 && epi > 1 {
        s.push('.');
    }
    if epi > 1 {
        s.push_str(&(gr + 1).to_string());
    }
    s
}

fn grad(spec: GradSpec, name: &str) -> Waveform {
    get_gradient(0.0, MAX_AMP, MAX_SLEW, spec, name)
}

pub fn crusher_area(slice_thickness: f64) -> f64 {
    CRUSHER_DEPHASING / GYRO / (1e-3 * slice_thickness) // µTs / m
}

/// Evaluate the protocol for the given inputs and phantom geometry
pub fn compute(inp: &Inputs, ph: &Phantom) -> Protocol {
    // ---- helpers / internal input params
    let is_radial = inp.trajectory.eq_ignore_ascii_case("radial");
    let is_ge = inp.sequence_type.contains("Gradient Echo");
    let freq_dir = if is_radial { 1 } else { usize::from(inp.frequency_direction == "left-right") };
    let phase_dir = 1 - freq_dir;
    let fov = if freq_dir == 1 { [inp.fov_p, inp.fov_f] } else { [inp.fov_f, inp.fov_p] };
    let matrix_p = inp.matrix_p;
    let matrix_f = if is_radial {
        let voxel = inp.fov_p / matrix_p as f64;
        (pyround(inp.fov_f / voxel) as usize).max(1)
    } else {
        inp.matrix_f
    };
    let matrix = if freq_dir == 1 { [matrix_p, matrix_f] } else { [matrix_f, matrix_p] };
    let recon_matrix_f = inp.recon_matrix_f.max(matrix_f);
    let recon_matrix_p = inp.recon_matrix_p.max(matrix_p);
    let recon_matrix = if inp.do_zerofill {
        if freq_dir == 1 {
            [recon_matrix_p, recon_matrix_f]
        } else {
            [recon_matrix_f, recon_matrix_p]
        }
    } else {
        matrix
    };
    let turbo = if is_ge { 1 } else { inp.turbo_factor.max(1) };
    let epi = if turbo > 1 && inp.epi_factor % 2 == 0 { inp.epi_factor + 1 } else { inp.epi_factor.max(1) };
    let mut echo_order_options: Vec<String> = ["Linear", "Centric", "Flexible"].iter().map(|s| s.to_string()).collect();
    if epi > 1 || turbo <= 1 {
        echo_order_options.retain(|o| o != "Flexible" && o != "Centric");
    }
    let echo_order = if echo_order_options.contains(&inp.echo_order) { inp.echo_order.clone() } else { "Linear".to_string() };

    // ---- phase encoding order
    let num_segm = turbo * epi;
    let num_shots = {
        let mut sampling_factor = if is_radial { inp.radial_sampling_factor } else { 1.0 + inp.phase_oversampling };
        if !is_radial {
            sampling_factor *= inp.partial_fourier_factor;
        }
        let nyquist = if is_radial { *matrix.iter().max().unwrap() as f64 * PI / 2.0 } else { matrix_p as f64 };
        ((nyquist * sampling_factor / num_segm as f64).ceil() as usize).max(1)
    };
    let num_blades = if is_radial { num_shots } else { 1 };
    let num_sampled = num_segm * if is_radial { 1 } else { num_shots };
    let num_full = if is_radial { num_sampled } else { num_sampled.max((matrix_p as f64 * (1.0 + inp.phase_oversampling)).ceil() as usize) };
    let num_sym_lines = 2 * num_sampled as i64 - num_full as i64;
    let num_blank_lines = num_full - num_sampled;
    let num_sym_segm = {
        let x = num_segm as f64 * (num_sym_lines as f64 / num_sampled as f64);
        if x % 2.0 == 0.0 {
            x as i64
        } else {
            (pyround((x - 1.0) / 2.0) * 2.0 + 1.0) as i64
        }
    };
    let k0_segment: Vec<i64> = {
        let k = num_segm as i64 - num_sym_segm.div_euclid(2) - 1;
        if num_sym_segm % 2 == 0 {
            vec![k, k + 1]
        } else {
            vec![k]
        }
    };
    let t_i = turbo as i64;
    let k0_linear: BTreeSet<(i64, i64)> = k0_segment.iter().map(|&s| (s.div_euclid(t_i), s.rem_euclid(t_i))).collect();
    let k0_reverse: BTreeSet<(i64, i64)> = k0_linear.iter().map(|&(g, r)| (epi as i64 - 1 - g, t_i - 1 - r)).collect();

    // ---- k-space axes
    let k_phase_axis = {
        let voxel = if is_radial { inp.fov_f.max(inp.fov_p) / num_full as f64 } else { inp.fov_p / matrix_p as f64 };
        get_k_axis(num_full, voxel)
    };
    let k_read_axis = {
        let voxel = fov[freq_dir] / matrix[freq_dir] as f64;
        let n = if !is_radial {
            let mut n = matrix[freq_dir];
            if fov[freq_dir] < ph.support[freq_dir] {
                n = (ph.support[freq_dir] / voxel).ceil() as usize;
            }
            n
        } else {
            let max_fov = ph.support[0].max(ph.support[1]).max(fov[0].max(fov[1]));
            (max_fov / voxel * inp.radial_fov_oversampling).ceil() as usize
        };
        get_k_axis(n.max(1), voxel)
    };
    let largest_phaser_area = k_phase_axis.iter().cloned().fold(f64::INFINITY, f64::min) * 1e3 / GYRO;
    let phase_step_area = if k_phase_axis.len() == 1 {
        0.0
    } else {
        let d: f64 = k_phase_axis.windows(2).map(|w| w[1] - w[0]).sum();
        d / (k_phase_axis.len() - 1) as f64 * 1e3 / GYRO
    };

    // ---- sequence objects (floating)
    let readouts_floating: Vec<Vec<Waveform>> = {
        let n = k_read_axis.len() as f64;
        let kmax = k_read_axis.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let kmin = k_read_axis.iter().cloned().fold(f64::INFINITY, f64::min);
        let pixel_size = (n - 1.0) / n / (kmax - kmin);
        let flat_area = 1e3 / pixel_size / GYRO;
        let amp = inp.pixel_bandwidth * matrix_f as f64 / (inp.fov_f * GYRO);
        (0..turbo)
            .map(|rf| {
                (0..epi)
                    .map(|gr| {
                        let mut r = get_gradient(0.0, amp, MAX_SLEW, GradSpec::FlatArea(flat_area), &format!("readout{}", suffix_rf_gr(turbo, epi, rf, gr)));
                        if gr % 2 == 1 {
                            rescale_gradient(&mut r, -1.0);
                        }
                        r
                    })
                    .collect()
            })
            .collect()
    };
    let r00 = readouts_floating[0][0].clone();
    let rf_excitation = get_rf(if is_ge { inp.fa } else { 90.0 }, 3.0, "excitation", 0.0);
    let slice_select_excitation =
        get_gradient(0.0, rf_excitation.fwhm / (inp.slice_thickness * GYRO), MAX_SLEW, GradSpec::FlatDur(rf_excitation.dur), "slice select excitation");
    let slice_select_rephaser_floating = grad(GradSpec::TotalArea(-slice_select_excitation.area / 2.0), "slice select rephaser");
    let read_prephaser_floating = {
        let mut g = get_gradient(0.0, MAX_AMP, MAX_SLEW, GradSpec::TotalArea(r00.area / 2.0), "read prephaser");
        if is_ge {
            rescale_gradient(&mut g, -1.0);
        }
        g
    };
    let rf_refocusing_floating: Option<Vec<Waveform>> =
        if is_ge { None } else { Some((0..turbo).map(|i| get_rf(180.0, 3.0, &format!("refocusing{}", suffix_rf(turbo, i)), 0.0)).collect()) };
    let slice_select_refocusing_floating: Option<Vec<Waveform>> = rf_refocusing_floating.as_ref().map(|rfs| {
        let amp = rfs[0].fwhm / (inp.slice_thickness * GYRO);
        (0..turbo)
            .map(|i| get_gradient(0.0, amp, MAX_SLEW, GradSpec::FlatDur(rfs[0].dur), &format!("slice select refocusing{}", suffix_rf(turbo, i))))
            .collect()
    });
    let c_area = crusher_area(inp.slice_thickness);
    let left_crusher_floating: Option<Vec<Waveform>> = rf_refocusing_floating
        .as_ref()
        .map(|_| (0..turbo).map(|i| grad(GradSpec::TotalArea(c_area), &format!("left crusher{}", suffix_rf(turbo, i)))).collect());
    let right_crusher_floating: Option<Vec<Waveform>> = rf_refocusing_floating
        .as_ref()
        .map(|_| (0..turbo).map(|i| grad(GradSpec::TotalArea(c_area), &format!("right crusher{}", suffix_rf(turbo, i)))).collect());
    let is_ir = inp.sequence_type == "Inversion Recovery";
    let rf_inversion_floating = is_ir.then(|| get_rf(180.0, 3.0, "inversion", 0.0));
    let slice_select_inversion_floating = rf_inversion_floating.as_ref().map(|rf| {
        let amp = rf.fwhm / (INVERSION_THK_FACTOR * inp.slice_thickness * GYRO);
        get_gradient(0.0, amp, MAX_SLEW, GradSpec::FlatDur(rf.dur), "slice select inversion")
    });
    let inversion_spoiler_floating = is_ir.then(|| grad(GradSpec::TotalArea(SPOILER_AREA), "inversion spoiler"));
    let rf_fatsat_floating = inp.fatsat.then(|| get_rf(90.0, 30.0 / inp.field_strength, "FatSat", 0.0));
    let fatsat_spoiler_floating = inp.fatsat.then(|| grad(GradSpec::TotalArea(SPOILER_AREA), "FatSat spoiler"));
    let spoiler_floating = grad(GradSpec::TotalArea(SPOILER_AREA), "spoiler");

    // ---- timing
    let phaser_duration = grad(GradSpec::TotalArea(largest_phaser_area), "").dur;
    let max_blip_dur = if epi <= 1 { 0.0 } else { grad(GradSpec::TotalArea(phase_step_area * num_shots as f64 * turbo as f64), "").dur };
    let readout_gap = (max_blip_dur - 2.0 * r00.risetime).max(0.0);
    let gr_echo_spacing = r00.dur + readout_gap;
    let gre_echo_train_dur = epi as f64 * gr_echo_spacing - readout_gap;
    let readout_risetime = r00.risetime;

    let min_refocusing_time = if is_ge {
        0.0
    } else {
        let rfr = &rf_refocusing_floating.as_ref().unwrap()[0];
        rf_excitation.dur / 2.0
            + rfr.dur / 2.0
            + read_prephaser_floating.dur.max(
                slice_select_excitation.risetime
                    + slice_select_rephaser_floating.dur
                    + left_crusher_floating.as_ref().unwrap()[0].dur.max(slice_select_refocusing_floating.as_ref().unwrap()[0].risetime),
            )
    };
    let (rf_dur_for_read, read_prephaser_dur, slice_rewind_dur) = if is_ge {
        (rf_excitation.dur, read_prephaser_floating.dur, slice_select_excitation.risetime + slice_select_rephaser_floating.dur)
    } else {
        (
            rf_refocusing_floating.as_ref().unwrap()[0].dur,
            0.0,
            right_crusher_floating.as_ref().unwrap()[0].dur.max(slice_select_refocusing_floating.as_ref().unwrap()[0].risetime),
        )
    };
    let min_rf_to_readtrain_center = rf_dur_for_read / 2.0 + gre_echo_train_dur / 2.0 - readout_risetime
        + (read_prephaser_dur + readout_risetime).max(phaser_duration).max(slice_rewind_dur);

    let min_te_from_indices = |gr: usize, rf: usize| -> f64 {
        let spacing = min_readtrain_spacing(gr_echo_spacing, gr, epi, is_ge, min_rf_to_readtrain_center, turbo, min_refocusing_time);
        let min_spin_echo_time = spacing * (1.0 + rf as f64);
        min_spin_echo_time + readtrain_shift(gr_echo_spacing, gr as f64, epi)
    };
    let min_te = if echo_order == "Centric" || echo_order == "Flexible" {
        min_te_from_indices(0, 0)
    } else {
        k0_linear.iter().chain(k0_reverse.iter()).map(|&(g, r)| min_te_from_indices(g as usize, r as usize)).fold(f64::INFINITY, f64::min)
    };
    let te = {
        let te_vals = params::te_vals();
        let min_te_snapped = snap(min_te, te_vals, SnapMode::Ceil).unwrap_or(min_te);
        match inp.te {
            None => min_te_snapped,
            Some(t) => t.max(min_te_snapped),
        }
    };

    // ---- k-space center echo indices
    let k0_index: (usize, usize, bool) = match echo_order.as_str() {
        "Centric" => (0, 0, false),
        "Flexible" => {
            let spacing = min_readtrain_spacing(gr_echo_spacing, 0, epi, is_ge, min_rf_to_readtrain_center, turbo, min_refocusing_time);
            let rf_index = (te / spacing).floor() as i64 - 1;
            (rf_index.min(turbo as i64 - 1).max(0) as usize, 0, false)
        }
        _ => {
            let mut best: Option<(f64, (usize, usize, bool))> = None;
            for (reverse, indices) in [(true, &k0_reverse), (false, &k0_linear)] {
                for &(g, r) in indices.iter() {
                    let (g, r) = (g as usize, r as usize);
                    let first_refocusing_time = if is_ge {
                        0.0
                    } else if turbo == 1 {
                        te / 2.0
                    } else {
                        get_readtrain_spacing(te, epi, gr_echo_spacing, g, r) / 2.0
                    };
                    let rf_to_readtrain_center = get_readtrain_spacing(te, epi, gr_echo_spacing, g, r) - first_refocusing_time;
                    if first_refocusing_time >= min_refocusing_time && rf_to_readtrain_center >= min_rf_to_readtrain_center {
                        let spacing = first_refocusing_time + rf_to_readtrain_center;
                        if best.is_none_or(|(b, _)| spacing < b) {
                            best = Some((spacing, (r, g, reverse)));
                        }
                    }
                }
            }
            best.map(|b| b.1).unwrap_or((0, 0, false))
        }
    };
    let (k0_rf, k0_gr, reverse_linear_order) = k0_index;

    let readtrain_spacing = get_readtrain_spacing(te, epi, gr_echo_spacing, k0_gr, k0_rf);
    let readtrain_center_time: Vec<f64> = (0..turbo).map(|rf| readtrain_spacing * (rf + 1) as f64).collect();
    let refocusing_time: Vec<f64> =
        if readtrain_center_time.len() == 1 { vec![te / 2.0] } else { readtrain_center_time.iter().map(|t| t - readtrain_spacing / 2.0).collect() };
    let readout_center_time: Vec<Vec<f64>> =
        readtrain_center_time.iter().map(|c| (0..epi).map(|g| c + (g as f64 - (epi as f64 - 1.0) / 2.0) * gr_echo_spacing).collect()).collect();

    // ---- PE table
    let shot = inp.shot.min(num_shots - 1);
    let pe_table: Vec<usize> = {
        let segment_order: Vec<i64> =
            if echo_order == "Linear" { (0..turbo as i64).collect() } else { flexible_segment_order(turbo as i64, num_sym_segm, k0_rf as i64) };
        let mut epi_idx: Vec<i64> = (0..epi as i64).collect();
        let mut seg = segment_order.clone();
        if reverse_linear_order {
            epi_idx.reverse();
            seg.reverse();
        }
        let nlps = (num_sampled as f64 / num_segm as f64) as i64;
        let mut t = Vec::with_capacity(num_shots * turbo * epi);
        for s in 0..num_shots as i64 {
            let line = if nlps > 0 { s.rem_euclid(nlps) } else { 0 };
            for r in 0..turbo {
                for g in 0..epi {
                    let view = epi_idx[g] * turbo as i64 + seg.get(r).copied().unwrap_or(r as i64);
                    t.push((nlps * view + line).max(0) as usize);
                }
            }
        }
        t
    };
    let pe = |s: usize, r: usize, g: usize| pe_table[(s * turbo + r) * epi + g];

    let phasers_floating: Vec<Waveform> = (0..turbo)
        .map(|rf| {
            let phaser_area = largest_phaser_area + pe(shot, rf, 0) as f64 * phase_step_area;
            let mut p = grad(GradSpec::TotalArea(largest_phaser_area), &format!("phase encode{}", suffix_rf(turbo, rf)));
            if largest_phaser_area.abs() > 1e-5 {
                rescale_gradient(&mut p, phaser_area / largest_phaser_area);
            }
            p
        })
        .collect();
    let blips_floating: Vec<Vec<Waveform>> = (0..turbo)
        .map(|rf| {
            (1..epi)
                .map(|gr| {
                    let area = phase_step_area * (pe(shot, rf, gr) as f64 - pe(shot, rf, gr - 1) as f64);
                    grad(GradSpec::TotalArea(area), "blip")
                })
                .collect()
        })
        .collect();
    let rephasers_floating: Vec<Waveform> = (0..turbo)
        .map(|rf| {
            let mut area = -phasers_floating[rf].area;
            for b in &blips_floating[rf] {
                area -= b.area;
            }
            let mut r = grad(GradSpec::TotalArea(largest_phaser_area), &format!("rephaser{}", suffix_rf(turbo, rf)));
            if largest_phaser_area.abs() > 1e-5 {
                rescale_gradient(&mut r, area / largest_phaser_area);
            }
            r
        })
        .collect();
    let sampling_windows_floating: Vec<Vec<Waveform>> =
        (0..turbo).map(|rf| (0..epi).map(|gr| get_adc(r00.flat_dur, &format!("sampling{}", suffix_rf_gr(turbo, epi, rf, gr)), 0.0)).collect()).collect();

    // ---- place sequence objects
    let rf_inversion = rf_inversion_floating.as_ref().map(|w| place(w, -inp.ti));
    let slice_select_inversion = slice_select_inversion_floating.as_ref().map(|w| place(w, -inp.ti));
    let inversion_spoiler = inversion_spoiler_floating.as_ref().map(|w| place(w, rf_inversion.as_ref().unwrap().t_end() + w.dur / 2.0));
    let fatsat_spoiler = fatsat_spoiler_floating.as_ref().map(|w| place(w, slice_select_excitation.t_start() - w.dur / 2.0));
    let rf_fatsat = rf_fatsat_floating.as_ref().map(|w| place(w, fatsat_spoiler.as_ref().unwrap().t_start() - w.dur / 2.0));
    let slice_select_rephaser = place(&slice_select_rephaser_floating, (slice_select_excitation.dur + slice_select_rephaser_floating.dur) / 2.0);
    let readouts: Vec<Vec<Waveform>> =
        readouts_floating.iter().zip(readout_center_time.iter()).map(|(rs, ts)| rs.iter().zip(ts.iter()).map(|(r, t)| place(r, *t)).collect()).collect();
    let read_prephaser = if is_ge {
        let first = &readouts[0][0];
        place(&read_prephaser_floating, first.center - (read_prephaser_floating.dur + first.dur) / 2.0)
    } else {
        place(&read_prephaser_floating, (rf_excitation.dur + read_prephaser_floating.dur) / 2.0)
    };
    let rf_refocusing: Option<Vec<Waveform>> =
        rf_refocusing_floating.as_ref().map(|v| v.iter().zip(refocusing_time.iter()).map(|(w, t)| place(w, *t)).collect());
    let slice_select_refocusing: Option<Vec<Waveform>> =
        slice_select_refocusing_floating.as_ref().map(|v| v.iter().zip(refocusing_time.iter()).map(|(w, t)| place(w, *t)).collect());
    let crusher_offset = left_crusher_floating.as_ref().map(|l| (rf_refocusing_floating.as_ref().unwrap()[0].dur + l[0].dur) / 2.0);
    let left_crusher: Option<Vec<Waveform>> =
        left_crusher_floating.as_ref().map(|v| v.iter().zip(refocusing_time.iter()).map(|(w, t)| place(w, t - crusher_offset.unwrap())).collect());
    let right_crusher: Option<Vec<Waveform>> =
        right_crusher_floating.as_ref().map(|v| v.iter().zip(refocusing_time.iter()).map(|(w, t)| place(w, t + crusher_offset.unwrap())).collect());
    let phasers: Vec<Waveform> =
        phasers_floating.iter().zip(readtrain_center_time.iter()).map(|(p, c)| place(p, c - (gre_echo_train_dur + p.dur) / 2.0 + readout_risetime)).collect();
    let sampling_windows: Vec<Vec<Waveform>> = sampling_windows_floating
        .iter()
        .zip(readout_center_time.iter())
        .map(|(ss, ts)| ss.iter().zip(ts.iter()).map(|(s, t)| place(s, *t)).collect())
        .collect();
    let blips: Vec<Vec<Waveform>> = readtrain_center_time
        .iter()
        .zip(blips_floating.iter())
        .map(|(c, bs)| bs.iter().enumerate().map(|(g, b)| place(b, c + gr_echo_spacing * (g as f64 - epi as f64 / 2.0 + 1.0))).collect())
        .collect();
    let rephasers: Vec<Waveform> =
        rephasers_floating.iter().zip(readtrain_center_time.iter()).map(|(r, c)| place(r, c + (gre_echo_train_dur + r.dur) / 2.0 - readout_risetime)).collect();
    let last_readout = readouts.last().unwrap().last().unwrap().clone();
    let spoiler = place(&spoiler_floating, last_readout.center + (last_readout.flat_dur + spoiler_floating.dur) / 2.0);

    let sequence_start = if let Some(s) = &slice_select_inversion {
        s.t_start()
    } else if let Some(r) = &rf_fatsat {
        r.t_start()
    } else {
        slice_select_excitation.t_start()
    };
    let min_tr = spoiler.t_end() - sequence_start;
    let tr = {
        let v = inp.tr.max(min_tr);
        snap(v, params::tr_vals(), SnapMode::Ceil).unwrap_or(v)
    };

    // ---- bounds
    let ti_bounds = slice_select_inversion_floating.as_ref().map(|ssi| MinMax { min: None, max: Some(tr - spoiler.t_end() - ssi.dur / 2.0) });
    let max_te = {
        let max_last_readtrain_center = tr + sequence_start - spoiler_floating.dur + readout_risetime - gre_echo_train_dur / 2.0;
        let max_readtrain_spacing = max_last_readtrain_center / turbo as f64;
        match echo_order.as_str() {
            "Flexible" => max_last_readtrain_center,
            "Centric" => max_readtrain_spacing,
            _ => k0_linear
                .iter()
                .chain(k0_reverse.iter())
                .map(|&(g, r)| max_readtrain_spacing * (r as f64 + 1.0) + readtrain_shift(gr_echo_spacing, g as f64, epi))
                .fold(f64::NEG_INFINITY, f64::max),
        }
    };
    let te_bounds = MinMax::new(min_te, max_te);
    let tr_bounds = MinMax::min_only(min_tr);

    let pixel_bandwidth_bounds = {
        let readout_area = 1e3 * matrix_f as f64 / (inp.fov_f * GYRO);
        let min_read_duration = readout_area / MAX_AMP;
        let first_readtrain_ref = if turbo == 1 { te } else { readtrain_spacing };
        let last_readtrain_ref = if turbo == 1 { te } else { readtrain_spacing * turbo as f64 };
        let last_read_end = tr + sequence_start - spoiler_floating.dur;
        let max_dur_ref_to_read_end = last_read_end - last_readtrain_ref;
        let rf_time = if is_ge { 0.0 } else { refocusing_time[0] };
        let k0_all: BTreeSet<(i64, i64)> = k0_linear.union(&k0_reverse).cloned().collect();
        let mut values: Vec<f64> = Vec::new();
        for pbw in params::pixel_bandwidth_option_values() {
            let read_duration = 1e3 / pbw;
            if read_duration < min_read_duration {
                continue;
            }
            let readout_risetime = readout_area / read_duration / MAX_SLEW;
            let read_gap = max_blip_dur.max(2.0 * readout_risetime);
            let first_read_start = rf_time + rf_dur_for_read / 2.0 + (read_prephaser_dur + readout_risetime).max(phaser_duration).max(slice_rewind_dur);
            let max_dur_read_start_to_ref = first_readtrain_ref - first_read_start;
            for &(gr_echo, _) in &k0_all {
                let num_blips_before_ref = if turbo == 1 { gr_echo as f64 } else { (epi as f64 - 1.0) / 2.0 };
                let num_blips_after_ref = epi as f64 - 1.0 - num_blips_before_ref;
                let dur_ref_to_read_end = read_duration * (num_blips_after_ref + 0.5) + read_gap * num_blips_after_ref;
                if dur_ref_to_read_end > max_dur_ref_to_read_end {
                    continue;
                }
                let dur_read_start_to_ref = read_duration * (num_blips_before_ref + 0.5) + read_gap * num_blips_before_ref;
                if dur_read_start_to_ref > max_dur_read_start_to_ref {
                    continue;
                }
                values.push(pbw);
            }
        }
        MinMax::new(values.iter().cloned().fold(f64::INFINITY, f64::min), values.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    };

    let min_slice_thickness = {
        let mut min_thks = vec![rf_excitation.fwhm / (MAX_AMP * GYRO)];
        if let Some(r) = &rf_refocusing {
            min_thks.push(r[0].fwhm / (MAX_AMP * GYRO));
        }
        if let Some(r) = &rf_inversion {
            min_thks.push(r.fwhm / (MAX_AMP * GYRO) * INVERSION_THK_FACTOR);
        }
        let first = rf_inversion.as_ref().unwrap_or(&rf_excitation);
        let mut max_risetime = tr - (spoiler.t_end() - first.t_start());
        if let (Some(rf), Some(sp)) = (&rf_fatsat_floating, &fatsat_spoiler_floating) {
            max_risetime -= rf.dur + sp.dur;
        }
        let max_amp = MAX_SLEW * max_risetime;
        min_thks.push(first.fwhm / (max_amp * GYRO));
        let min_thk = min_thks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let available_time = if is_ge { sampling_windows[0][0].t_start() } else { rf_refocusing.as_ref().unwrap()[0].t_start() };
        let mut crusher_duration = 0.0;
        let mut found = None;
        for thk in params::slice_thickness_option_values() {
            if thk < min_thk {
                continue;
            }
            if !is_ge {
                crusher_duration = grad(GradSpec::TotalArea(crusher_area(thk)), "").dur;
            }
            let slice_amp = rf_excitation.fwhm / (thk * GYRO);
            let slice_rise_time = slice_amp / MAX_SLEW;
            let slice_rephaser_area = slice_amp * (rf_excitation.dur + slice_rise_time) / 2.0;
            let slice_rephaser = grad(GradSpec::TotalArea(slice_rephaser_area), "");
            if available_time < rf_excitation.dur / 2.0 + slice_rise_time + slice_rephaser.dur + crusher_duration {
                continue;
            }
            found = Some(thk);
            break;
        }
        found
    };
    let slice_thickness_bounds = MinMax { min: min_slice_thickness, max: None };

    let max_phaser_area = {
        let max_phaser_duration = if is_ge {
            readtrain_spacing - rf_excitation.dur / 2.0 - gre_echo_train_dur / 2.0 + readout_risetime
        } else {
            readtrain_spacing - refocusing_time[0] - rf_refocusing_floating.as_ref().unwrap()[0].dur / 2.0 - gre_echo_train_dur / 2.0 + readout_risetime
        };
        let max_risetime = MAX_AMP / MAX_SLEW;
        if max_phaser_duration > 2.0 * max_risetime {
            (max_phaser_duration - max_risetime) * MAX_AMP
        } else {
            (max_phaser_duration / 2.0).powi(2) * MAX_SLEW
        }
    };

    let max_readout_area = 'mra: {
        // see paramBounds.tex for formulae
        let mut areas: Vec<f64> = Vec::new();
        let d = 1e3 / inp.pixel_bandwidth;
        let s = MAX_SLEW;
        if is_ge {
            let n = k0_gr as f64 + 0.5;
            let m = k0_gr as f64 * 2.0 + 1.0;
            let t = te - rf_excitation.dur / 2.0;
            let mut v = 0.0;
            for _ in 0..2 {
                if m > 1.0 {
                    let q = t - phaser_duration.max(slice_select_excitation.risetime + slice_select_rephaser_floating.dur);
                    areas.push(d * s * (q - n * (d + v) + v / 2.0) / (m - 1.0));
                    // eq. 15
                }
                // max wrt G read (eq. 12)
                let h_roots = quadratic_roots(
                    8.0 * (3.0 - 2.0 * m),
                    4.0 * s * (t * (2.0 * m - 6.0) + d * (2.0 * n - m) + v * (2.0 * n - 1.0)),
                    s * s * (4.0 * t * t - d * d),
                );
                let h = h_roots.iter().filter(|z| gt0(**z)).map(|z| z.0).fold(MAX_AMP, f64::min);
                // eq. 13
                let a_roots = quadratic_roots(1.0, d * (d * s + 2.0 * m * h), d * d * h * (2.0 * h - s * (2.0 * t - 2.0 * n * (d + v) + v)));
                if a_roots.iter().all(|z| z.0 < 0.0 || (z.0 == 0.0 && z.1 < 0.0)) {
                    break 'mra 0.0;
                }
                let a = a_roots.iter().filter(|z| gt0(**z)).map(|z| z.0).fold(f64::INFINITY, f64::min);
                areas.push(a);
                let read_risetime = areas.iter().cloned().fold(f64::INFINITY, f64::min) / (d * s);
                v = (max_blip_dur - 2.0 * read_risetime).max(0.0);
            }
        } else {
            let rfr = &rf_refocusing_floating.as_ref().unwrap()[0];
            // limit by half readout duration tr:
            let trd = (readtrain_spacing - refocusing_time[0] - rfr.dur / 2.0) / epi as f64;
            areas.push(d * s * trd - d * d * s / 2.0);
            // limit by prephaser duration tp:
            let tp = refocusing_time[0] - rfr.dur / 2.0 - rf_excitation.dur / 2.0;
            let h = (s * tp / 2.0).min(MAX_AMP);
            areas.push(d * (((d * s).powi(2) - 8.0 * h * (h - s * tp)).sqrt() - d * s) / 2.0);
        }
        areas.push(MAX_AMP * 1e3 / inp.pixel_bandwidth); // max wrt max_amp
        areas.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let min_voxel_f_aniso = 1e3 / (max_readout_area * GYRO);
    let min_voxel_p_aniso = 1e3 / (2.0 * max_phaser_area * GYRO);
    let (min_voxel_f, min_voxel_p) =
        if is_radial { (min_voxel_f_aniso.max(min_voxel_p_aniso), min_voxel_f_aniso.max(min_voxel_p_aniso)) } else { (min_voxel_f_aniso, min_voxel_p_aniso) };
    let matrix_f_bounds = {
        let mut lo = vec![1.0];
        let mut hi = vec![inp.fov_f / min_voxel_f];
        if inp.constant_fov_bw_bounds {
            lo.push(matrix_f as f64 * inp.pixel_bandwidth / pixel_bandwidth_bounds.max.unwrap());
            hi.push(matrix_f as f64 * inp.pixel_bandwidth / pixel_bandwidth_bounds.min.unwrap());
        }
        MinMax::new(lo.iter().cloned().fold(f64::NEG_INFINITY, f64::max), hi.iter().cloned().fold(f64::INFINITY, f64::min))
    };
    let matrix_p_bounds = MinMax::new(1.0, inp.fov_p / min_voxel_p);
    let recon_matrix_f_bounds = MinMax::min_only(matrix_f as f64);
    let recon_matrix_p_bounds = MinMax::min_only(matrix_p as f64);
    let fov_bounds = |fov_d: f64, matrix_d: usize, min_voxel: f64, mb: MinMax| -> MinMax {
        if inp.constant_voxel_bounds {
            let voxel = fov_d / matrix_d as f64;
            MinMax { min: mb.min.map(|m| voxel * m), max: mb.max.map(|m| voxel * m) }
        } else {
            MinMax::min_only(min_voxel * matrix_d as f64)
        }
    };
    let fov_f_bounds = fov_bounds(inp.fov_f, matrix_f, min_voxel_f, matrix_f_bounds);
    let fov_p_bounds = fov_bounds(inp.fov_p, matrix_p, min_voxel_p, matrix_p_bounds);
    let turbo_factor_bounds = if epi % 2 == 1 {
        let undersampling = if is_radial { 1.0 } else { inp.partial_fourier_factor };
        MinMax::new(1.0, (matrix[phase_dir] as f64 * undersampling / epi as f64 * (1.0 + params::max_phase_oversampling_factor())).floor())
    } else {
        MinMax::new(1.0, 1.0)
    };
    let max_epi_factor = {
        let undersampling = if is_radial { 1.0 } else { inp.partial_fourier_factor };
        (matrix[phase_dir] as f64 * undersampling / turbo as f64 * (1.0 + params::max_phase_oversampling_factor())).floor() as usize
    };
    let shot_bounds = MinMax::new(0.0, (num_shots - 1) as f64);

    // ---- sampling times, PE echoes, spin echoes
    let sampling_time = {
        let half = 0.5e3 / inp.pixel_bandwidth;
        params::linspace(-half, half, k_read_axis.len())
    };
    let pe_echo: Vec<(usize, usize)> = {
        let mut first = vec![None; num_sampled];
        for s in 0..num_shots {
            for r in 0..turbo {
                for g in 0..epi {
                    let ky = pe(s, r, g);
                    if ky < num_sampled && first[ky].is_none() {
                        first[ky] = Some((r, g));
                    }
                }
            }
        }
        first.into_iter().map(|f| f.unwrap_or((0, 0))).collect()
    };
    let spin_echoes: Vec<f64> = if turbo == 1 { vec![te; num_sampled] } else { pe_echo.iter().map(|&(r, _)| (r + 1) as f64 * readtrain_spacing).collect() };

    // ---- SNR and scan time
    let tissues = ph.tissues.clone();
    let reference_tissue = if tissues.contains(&inp.reference_tissue) { inp.reference_tissue.clone() } else { tissues.first().cloned().unwrap_or_default() };
    let dwell_time = if sampling_time.len() > 1 { sampling_time[1] - sampling_time[0] } else { f64::NAN };
    let noise_std = inp.noise_gain / (dwell_time * inp.nsa).sqrt() / inp.field_strength;
    let signal_level =
        ((k_read_axis.len() * num_sampled * num_blades) as f64).sqrt() * inp.slice_thickness * (fov[0] * fov[1]) / (matrix[0] * matrix[1]) as f64;
    let decayed_signal = {
        let argmin = |v: &[f64]| v.iter().enumerate().fold((0usize, f64::INFINITY), |(bi, bv), (i, x)| if x.abs() < bv { (i, x.abs()) } else { (bi, bv) }).0;
        let i_read = argmin(&k_read_axis);
        let i_pe = argmin(&k_phase_axis).min(num_sampled - 1);
        let (rf, gr) = pe_echo[i_pe];
        let ro = &readouts[rf][gr];
        let n = sampling_time.len();
        let st = if ro.area < 0.0 { sampling_time[n - 1 - i_read] } else { sampling_time[i_read] };
        let tae = ro.center + st;
        let t_rel = if is_ge { tae } else { tae - spin_echoes[i_pe] };
        let t2 = tissue(&reference_tissue).map(|t| t.t2[field_index(inp.field_strength)]).unwrap_or(f64::INFINITY);
        signal_level * (-tae.abs() / t2).exp() * (-t_rel.abs() / T2_PRIM).exp()
    };
    let reference_signal = decayed_signal * component_pd_t1w(&reference_tissue, &inp.sequence_type, tr, inp.ti, inp.fa, inp.field_strength).abs();
    let snr = reference_signal / noise_std;
    // multi-slice: slices are interleaved within TR; more slices than fit need concatenations
    // (with inversion recovery, other slices are inverted and read out during the inversion time
    // of a slice – interleaved inversion – so a slice occupies its inversion pulse plus its readout)
    let slice_time = match &slice_select_inversion {
        Some(inv) => {
            let readout_start = rf_fatsat.as_ref().map_or(slice_select_excitation.t_start(), |r| r.t_start());
            (spoiler.t_end() - readout_start) + (inv.t_end() - inv.t_start())
        }
        None => min_tr,
    };
    let slices_per_tr = ((tr / slice_time.max(1e-3)).floor() as usize).max(1);
    // (only 3D phantoms have a slice stack; 2D phantoms are a single slice)
    let num_slices = if ph.is_volume_slice() { inp.num_slices.max(1) } else { 1 };
    let concatenations = num_slices.div_ceil(slices_per_tr);
    let scantime = num_shots as f64 * inp.nsa * tr * concatenations as f64;
    let k_angles: Vec<f64> = (0..num_blades).map(|i| i as f64 * (PI / num_blades as f64)).collect();
    let spoke_angle = k_angles[shot.min(k_angles.len() - 1)].to_degrees();

    // ---- boards
    let mut boards = Boards::default();
    boards.frequency.push(read_prephaser);
    boards.frequency.extend(readouts.iter().flatten().cloned());
    boards.phase.extend(phasers.iter().cloned());
    boards.phase.extend(rephasers.iter().cloned());
    boards.phase.extend(blips.iter().flatten().cloned());
    boards.slice.extend(slice_select_inversion.iter().cloned());
    boards.slice.extend(inversion_spoiler.iter().cloned());
    boards.slice.extend(fatsat_spoiler.iter().cloned());
    boards.slice.push(slice_select_excitation.clone());
    boards.slice.push(slice_select_rephaser);
    boards.slice.extend(slice_select_refocusing.iter().flatten().cloned());
    boards.slice.extend(left_crusher.iter().flatten().cloned());
    boards.slice.extend(right_crusher.iter().flatten().cloned());
    boards.slice.push(spoiler);
    boards.rf.extend(rf_inversion.iter().cloned());
    boards.rf.extend(rf_fatsat.iter().cloned());
    boards.rf.push(rf_excitation);
    boards.rf.extend(rf_refocusing.iter().flatten().cloned());
    boards.adc.extend(sampling_windows.iter().flatten().cloned());

    Protocol {
        is_radial,
        is_gradient_echo: is_ge,
        freq_dir,
        phase_dir,
        fov,
        matrix,
        matrix_f,
        matrix_p,
        recon_matrix_f,
        recon_matrix_p,
        recon_matrix,
        turbo_factor: turbo,
        epi_factor: epi,
        echo_order_options,
        echo_order,
        num_segm,
        num_shots,
        num_blades,
        num_sampled_phase_encodes: num_sampled,
        num_phase_encodes_full: num_full,
        num_blank_lines,
        num_sym_segm: num_sym_segm.max(0) as usize,
        k0_index,
        k_phase_axis,
        k_read_axis,
        pe_table,
        shot,
        tr,
        te,
        min_te,
        max_te,
        min_tr,
        readtrain_spacing,
        readtrain_center_time,
        sequence_start,
        boards,
        readouts,
        rf_refocusing,
        shot_bounds,
        turbo_factor_bounds,
        max_epi_factor,
        matrix_f_bounds,
        matrix_p_bounds,
        recon_matrix_f_bounds,
        recon_matrix_p_bounds,
        fov_f_bounds,
        fov_p_bounds,
        tr_bounds,
        ti_bounds,
        te_bounds,
        pixel_bandwidth_bounds,
        slice_thickness_bounds,
        min_slice_thickness,
        max_readout_area,
        max_phaser_area,
        tissues,
        reference_tissue,
        noise_std,
        snr,
        scantime,
        slices_per_tr,
        concatenations,
        spoke_angle,
        k_angles,
        sampling_time,
        pe_echo,
        spin_echoes,
    }
}

/// Net waveform of a board (`accumulate_waveforms`)
pub fn net_waveform(objects: &[Waveform]) -> Vec<(f64, f64)> {
    sequence::accumulate_waveforms(objects.iter())
}

/// k-space trajectory (kx, ky over time) from the net read/phase gradients (`k_trajectory`)
#[derive(Debug, Clone, Default)]
pub struct KTrajectory {
    pub t: Vec<f64>,
    pub kx: Vec<f64>,
    pub ky: Vec<f64>,
    pub dt: f64,
}

/// `np.interp`
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.is_empty() {
        return 0.0;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    // binary search for the right interval (first index with xp[i] > x)
    let i = xp.partition_point(|&v| v <= x);
    let (x0, x1, f0, f1) = (xp[i - 1], xp[i], fp[i - 1], fp[i]);
    if x1 == x0 {
        return f1;
    }
    let slope = (f1 - f0) / (x1 - x0);
    slope * (x - x0) + f0
}

fn get_k_coords(t: &[f64], gp: &[f64], tp: &[f64], refocus_intervals: &[(f64, f64)]) -> Vec<f64> {
    let g: Vec<f64> = t.iter().map(|&x| interp(x, tp, gp)).collect();
    let mut k = vec![0.0; t.len()];
    for i in 1..t.len() {
        let dk = (t[i] - t[i - 1]) * (g[i - 1] + (g[i] - g[i - 1]) / 2.0) * GYRO * 1e-3;
        k[i] = k[i - 1] + dk;
    }
    for &(start, stop) in refocus_intervals {
        // k inversion of refocusing pulse corresponds to negative shift of 2k
        let k_before = t.iter().zip(k.iter()).rev().find(|(tt, _)| **tt <= start).map(|(_, kk)| *kk).unwrap_or(0.0);
        for i in 0..t.len() {
            if t[i] > start && t[i] < stop {
                k[i] -= 2.0 * k_before * (t[i] - start) / (stop - start);
            } else if t[i] >= stop {
                k[i] -= 2.0 * k_before;
            }
        }
    }
    k
}

pub fn k_trajectory(p: &Protocol) -> KTrajectory {
    let dt = 0.01;
    let freq = net_waveform(&p.boards.frequency);
    let phase = net_waveform(&p.boards.phase);
    let refocus: Vec<(f64, f64)> = p.rf_refocusing.as_ref().map(|v| v.iter().map(|rf| (rf.t_start(), rf.t_end())).collect()).unwrap_or_default();
    let mut t: Vec<f64> = freq.iter().map(|x| x.0).chain(phase.iter().map(|x| x.0)).chain(refocus.iter().flat_map(|r| [r.0, r.1])).collect();
    let tmax = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut x = 0.0;
    let mut i = 0usize;
    while x < tmax {
        t.push(x);
        i += 1;
        x = i as f64 * dt;
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.dedup();
    let (ft, fg): (Vec<f64>, Vec<f64>) = freq.into_iter().unzip();
    let (pt, pg): (Vec<f64>, Vec<f64>) = phase.into_iter().unzip();
    let mut kx = get_k_coords(&t, &fg, &ft, &refocus);
    let mut ky = get_k_coords(&t, &pg, &pt, &refocus);
    if !p.is_radial {
        if p.phase_dir == 1 {
            std::mem::swap(&mut kx, &mut ky);
        }
    } else {
        let (s, c) = p.spoke_angle.to_radians().sin_cos();
        for i in 0..kx.len() {
            let (a, b) = (kx[i], ky[i]);
            kx[i] = c * a - s * b;
            ky[i] = s * a + c * b;
        }
    }
    KTrajectory { t, kx, ky, dt }
}

/// k-space coordinates sampled during [t0, t1] (`get_k_on_interval`)
pub fn k_on_interval(tr: &KTrajectory, t0: f64, t1: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let mut i = 0usize;
    loop {
        let t = t0 + i as f64 * tr.dt;
        if t >= t1 {
            break;
        }
        out.push([interp(t, &tr.t, &tr.kx), interp(t, &tr.t, &tr.ky)]);
        i += 1;
        if i > 1_000_000 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_order() {
        assert_eq!(flexible_segment_order(5, 3, 0), vec![3, 4, 2, 1, 0]);
        assert_eq!(flexible_segment_order(5, 3, 1), vec![4, 3, 2, 1, 0]);
        assert_eq!(flexible_segment_order(5, 3, 2), vec![1, 2, 3, 4, 0]);
        assert_eq!(flexible_segment_order(5, 3, 3), vec![0, 1, 2, 3, 4]);
        assert_eq!(flexible_segment_order(5, 3, 4), vec![0, 1, 2, 4, 3]);
        assert_eq!(flexible_segment_order(5, 2, 0), vec![4, 3, 2, 1, 0]);
        assert_eq!(flexible_segment_order(5, 2, 1), vec![4, 3, 2, 1, 0]);
        assert_eq!(flexible_segment_order(5, 2, 2), vec![2, 3, 4, 1, 0]);
        assert_eq!(flexible_segment_order(5, 2, 3), vec![0, 1, 2, 3, 4]);
        assert_eq!(flexible_segment_order(5, 2, 4), vec![0, 1, 2, 3, 4]);
    }
}
