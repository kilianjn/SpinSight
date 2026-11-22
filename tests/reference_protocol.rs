//! Cross-validation of the protocol computation against values computed by the original Python
//! implementation (fixtures in `tests/reference/*.json`).

use serde_json::Value as J;
use spinsight::params::{self, Value};
use spinsight::phantom;
use spinsight::protocol::{self, Inputs, MinMax};
use spinsight::sequence::Waveform;

fn load(name: &str) -> J {
    let path = format!("{}/tests/reference/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

struct Checker {
    config: String,
    failures: Vec<String>,
    rtol: f64,
}

impl Checker {
    fn close(&self, a: f64, b: f64) -> bool {
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        (a - b).abs() <= self.rtol * (1.0 + a.abs().max(b.abs()))
    }
    fn num(&mut self, node: &str, ours: f64, theirs: &J) {
        let t = match theirs {
            J::Number(n) => n.as_f64().unwrap(),
            J::String(s) if s == "inf" => f64::INFINITY,
            J::String(s) if s == "-inf" => f64::NEG_INFINITY,
            J::Bool(b) => f64::from(u8::from(*b)),
            _ => {
                self.failures.push(format!("{}: {node}: expected number, reference is {theirs}", self.config));
                return;
            }
        };
        if !self.close(ours, t) {
            self.failures.push(format!("{}: {node}: ours {ours} vs python {t}", self.config));
        }
    }
    fn opt(&mut self, node: &str, ours: Option<f64>, theirs: &J) {
        match (ours, theirs) {
            (None, J::Null) => {}
            (Some(o), t) if !t.is_null() => self.num(node, o, t),
            _ => self.failures.push(format!("{}: {node}: ours {ours:?} vs python {theirs}", self.config)),
        }
    }
    fn vec(&mut self, node: &str, ours: &[f64], theirs: &J) {
        let Some(t) = theirs.as_array() else {
            self.failures.push(format!("{}: {node}: reference is not an array: {theirs}", self.config));
            return;
        };
        if t.len() != ours.len() {
            self.failures.push(format!("{}: {node}: length {} vs python {}", self.config, ours.len(), t.len()));
            return;
        }
        for (i, (o, tv)) in ours.iter().zip(t.iter()).enumerate() {
            let before = self.failures.len();
            self.num(&format!("{node}[{i}]"), *o, tv);
            if self.failures.len() > before {
                return; // report first mismatch only
            }
        }
    }
    fn minmax(&mut self, node: &str, ours: Option<MinMax>, theirs: &J) {
        match (ours, theirs) {
            (None, J::Null) => {}
            (Some(m), J::Array(a)) if a.len() == 2 => {
                self.opt(&format!("{node}.min"), m.min, &a[0]);
                self.opt(&format!("{node}.max"), m.max, &a[1]);
            }
            _ => self.failures.push(format!("{}: {node}: ours {ours:?} vs python {theirs}", self.config)),
        }
    }
    fn waveforms(&mut self, node: &str, ours: &[Waveform], theirs: &J) {
        let t = theirs.as_array().unwrap();
        let mut ours: Vec<&Waveform> = ours.iter().collect();
        let mut t: Vec<&J> = t.iter().collect();
        if ours.len() != t.len() {
            let on: Vec<&str> = ours.iter().map(|w| w.name.as_str()).collect();
            let tn: Vec<&str> = t.iter().map(|w| w["name"].as_str().unwrap()).collect();
            self.failures.push(format!("{}: {node}: objects {on:?} vs python {tn:?}", self.config));
            return;
        }
        // same order as python
        ours.truncate(t.len());
        t.truncate(ours.len());
        for (w, tw) in ours.iter().zip(t.iter()) {
            let name = format!("{node}/{}", w.name);
            if tw["name"].as_str() != Some(&w.name) {
                self.failures.push(format!("{}: {node}: name {} vs python {}", self.config, w.name, tw["name"]));
                continue;
            }
            self.vec(&format!("{name}.time"), &w.time, &tw["time"]);
            if !tw["amp"].is_null() {
                self.vec(&format!("{name}.amp"), &w.amp, &tw["amp"]);
            }
            if !tw["area"].is_null() {
                self.num(&format!("{name}.area"), w.area, &tw["area"]);
            }
            if !tw["fwhm"].is_null() {
                self.num(&format!("{name}.fwhm"), w.fwhm, &tw["fwhm"]);
            }
        }
    }
}

fn json_to_value(v: &J) -> Value {
    match v {
        J::Number(n) => Value::Num(n.as_f64().unwrap()),
        J::Bool(b) => Value::Bool(*b),
        J::String(s) => Value::Str(s.clone()),
        _ => panic!("unsupported input {v}"),
    }
}

#[test]
fn option_lists_match_python() {
    let reference = load("options.json");
    let mut failures = Vec::new();
    for (name, opts) in reference["options"].as_object().unwrap() {
        if name == "load_presets" || name == "object" {
            continue; // directory listing order in Python
        }
        let ours = params::param(name).options.as_ref().unwrap();
        let theirs = opts.as_array().unwrap();
        if ours.len() != theirs.len() {
            failures.push(format!("{name}: {} options vs python {}", ours.len(), theirs.len()));
            continue;
        }
        for (o, t) in ours.iter().zip(theirs.iter()) {
            let (tl, tv) = (t[0].as_str().unwrap(), &t[1]);
            let value_ok = match (&o.value, tv) {
                (Value::Num(a), J::Number(b)) => (a - b.as_f64().unwrap()).abs() <= 1e-12 * a.abs().max(1.0),
                (Value::Str(a), J::String(b)) => a == b,
                _ => false,
            };
            if o.label != tl || !value_ok {
                failures.push(format!("{name}: option ({}, {:?}) vs python ({tl}, {tv})", o.label, o.value));
                break;
            }
        }
    }
    for (name, d) in reference["defaults"].as_object().unwrap() {
        let Some(p) = params::try_param(name) else { continue };
        // the default phantom of the desktop version is the 3D brain (Python: 2D brain)
        if d.is_null() || name == "load_presets" || name == "object" {
            continue;
        }
        let ours = &p.default;
        let ok = match (ours, d) {
            (Value::Num(a), J::Number(b)) => (a - b.as_f64().unwrap()).abs() < 1e-12,
            (Value::Str(a), J::String(b)) => a == b,
            (Value::Bool(a), J::Bool(b)) => a == b,
            _ => false,
        };
        if !ok {
            failures.push(format!("default {name}: {ours:?} vs python {d}"));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn protocols_match_python() {
    let reference = load("protocols.json");
    let mut all_failures = Vec::new();
    for (config, r) in reference.as_object().unwrap() {
        // the Python default phantom
        let mut inp = Inputs { object: "brain".into(), ..Default::default() };
        for (k, v) in r["inputs"].as_object().unwrap() {
            inp.set(k, &json_to_value(v)).unwrap();
        }
        let ph = phantom::load(&inp.object, inp.min_voxel_size).unwrap();
        let p = protocol::compute(&inp, &ph);
        let mut c = Checker { config: config.clone(), failures: Vec::new(), rtol: 1e-9 };
        c.num("TE", p.te, &r["TE"]);
        c.num("TR", p.tr, &r["TR"]);
        c.num("min_TE", p.min_te, &r["min_TE"]);
        c.num("max_TE", p.max_te, &r["max_TE"]);
        c.num("min_TR", p.min_tr, &r["min_TR"]);
        c.num("sequence_start", p.sequence_start, &r["sequence_start"]);
        c.num("num_shots", p.num_shots as f64, &r["num_shots"]);
        c.num("num_segm", p.num_segm as f64, &r["num_segm"]);
        c.num("num_blades", p.num_blades as f64, &r["num_blades"]);
        c.num("num_sampled_phase_encodes", p.num_sampled_phase_encodes as f64, &r["num_sampled_phase_encodes"]);
        c.num("num_phase_encodes_full", p.num_phase_encodes_full as f64, &r["num_phase_encodes_full"]);
        c.num("num_blank_lines", p.num_blank_lines as f64, &r["num_blank_lines"]);
        c.num("num_sym_segm", p.num_sym_segm as f64, &r["num_sym_segm"]);
        let k0 = &r["k0_index"];
        c.num("k0_rf", p.k0_index.0 as f64, &k0[0]);
        c.num("k0_gr", p.k0_index.1 as f64, &k0[1]);
        c.num("reverse_linear_order", f64::from(u8::from(p.k0_index.2)), &k0[2]);
        c.num("readtrain_spacing", p.readtrain_spacing, &r["readtrain_spacing"]);
        c.vec("readtrain_center_time", &p.readtrain_center_time, &r["readtrain_center_time"]);
        c.vec("matrix", &p.matrix.map(|m| m as f64), &r["matrix"]);
        c.vec("recon_matrix", &p.recon_matrix.map(|m| m as f64), &r["recon_matrix"]);
        c.vec("FOV", &p.fov, &r["FOV"]);
        c.num("matrix_F", p.matrix_f as f64, &r["matrix_F"]);
        c.num("recon_matrix_F", p.recon_matrix_f as f64, &r["recon_matrix_F"]);
        c.num("recon_matrix_P", p.recon_matrix_p as f64, &r["recon_matrix_P"]);
        c.num("turbo_factor", p.turbo_factor as f64, &r["turbo_factor"]);
        c.num("EPI_factor", p.epi_factor as f64, &r["EPI_factor"]);
        if r["echo_order"].as_str() != Some(&p.echo_order) {
            c.failures.push(format!("{config}: echo_order {} vs {}", p.echo_order, r["echo_order"]));
        }
        let eo: Vec<String> = r["echo_order_options"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        if eo != p.echo_order_options {
            c.failures.push(format!("{config}: echo_order_options {:?} vs {eo:?}", p.echo_order_options));
        }
        let pe_flat: Vec<f64> = r["pe_table"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|a| a.as_array().unwrap().iter().flat_map(|b| b.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())))
            .collect();
        c.vec("pe_table", &p.pe_table.iter().map(|&v| v as f64).collect::<Vec<_>>(), &J::from(pe_flat));
        c.num("shot", p.shot as f64, &r["shot"]);
        c.vec("k_read_axis", &p.k_read_axis, &r["k_read_axis"]);
        c.vec("k_phase_axis", &p.k_phase_axis, &r["k_phase_axis"]);
        c.minmax("TE_bounds", Some(p.te_bounds), &r["TE_bounds"]);
        c.minmax("TR_bounds", Some(p.tr_bounds), &r["TR_bounds"]);
        c.minmax("TI_bounds", p.ti_bounds, &r["TI_bounds"]);
        c.minmax("pixel_bandwidth_bounds", Some(p.pixel_bandwidth_bounds), &r["pixel_bandwidth_bounds"]);
        c.minmax("slice_thickness_bounds", Some(p.slice_thickness_bounds), &r["slice_thickness_bounds"]);
        c.opt("min_slice_thickness", p.min_slice_thickness, &r["min_slice_thickness"]);
        c.minmax("matrix_F_bounds", Some(p.matrix_f_bounds), &r["matrix_F_bounds"]);
        c.minmax("matrix_P_bounds", Some(p.matrix_p_bounds), &r["matrix_P_bounds"]);
        c.minmax("FOV_F_bounds", Some(p.fov_f_bounds), &r["FOV_F_bounds"]);
        c.minmax("FOV_P_bounds", Some(p.fov_p_bounds), &r["FOV_P_bounds"]);
        c.minmax("recon_matrix_F_bounds", Some(p.recon_matrix_f_bounds), &r["recon_matrix_F_bounds"]);
        c.minmax("recon_matrix_P_bounds", Some(p.recon_matrix_p_bounds), &r["recon_matrix_P_bounds"]);
        c.minmax("turbo_factor_bounds", Some(p.turbo_factor_bounds), &r["turbo_factor_bounds"]);
        c.num("max_EPI_factor", p.max_epi_factor as f64, &r["max_EPI_factor"]);
        c.minmax("shot_bounds", Some(p.shot_bounds), &r["shot_bounds"]);
        c.num("max_readout_area", p.max_readout_area, &r["max_readout_area"]);
        c.num("max_phaser_area", p.max_phaser_area, &r["max_phaser_area"]);
        c.num("SNR", p.snr, &r["SNR"]);
        c.num("scantime", p.scantime, &r["scantime"]);
        c.num("spoke_angle", p.spoke_angle, &r["spoke_angle"]);
        c.num("noise_std", p.noise_std, &r["noise_std"]);
        if r["reference_tissue"].as_str() != Some(&p.reference_tissue) {
            c.failures.push(format!("{config}: reference_tissue {} vs {}", p.reference_tissue, r["reference_tissue"]));
        }
        c.vec("spin_echoes", &p.spin_echoes, &r["spin_echoes"]);
        c.vec("sampling_time", &p.sampling_time, &r["sampling_time"]);
        c.waveforms("frequency", &p.boards.frequency, &r["frequency_objects"]);
        c.waveforms("phase", &p.boards.phase, &r["phase_objects"]);
        c.waveforms("slice", &p.boards.slice, &r["slice_objects"]);
        c.waveforms("RF", &p.boards.rf, &r["RF_objects"]);
        c.waveforms("ADC", &p.boards.adc, &r["ADC_objects"]);
        all_failures.extend(c.failures);
    }
    assert!(all_failures.is_empty(), "{} mismatches:\n{}", all_failures.len(), all_failures.join("\n"));
}
