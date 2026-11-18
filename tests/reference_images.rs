//! Cross-validation of phantom k-space and reconstructed images against raw arrays dumped from
//! the original Python implementation. The arrays are large and therefore not committed: set
//! `SPINSIGHT_PYREF` to a directory containing `phref/` and `imgref/` to run these tests.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value as J;
use spinsight::array::C64;
use spinsight::imaging::Simulator;
use spinsight::params::Value;
use spinsight::phantom;
use spinsight::protocol::{self, Inputs};

fn pyref() -> Option<PathBuf> {
    std::env::var("SPINSIGHT_PYREF").ok().map(PathBuf::from)
}

fn read_c64(path: &PathBuf) -> Vec<C64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    b.chunks_exact(16).map(|c| C64::new(f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
}

fn rel_err(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
    let den: f64 = b.iter().map(|y| y.norm_sqr()).sum();
    (num / den.max(1e-300)).sqrt()
}

#[test]
fn phantom_kspace_matches_python() {
    let Some(dir) = pyref() else {
        eprintln!("SPINSIGHT_PYREF not set; skipping");
        return;
    };
    let meta: J = serde_json::from_str(&std::fs::read_to_string(dir.join("phref/meta.json")).unwrap()).unwrap();
    let mut worst: f64 = 0.0;
    for name in ["Shepp-Logan", "brain", "abdomen"] {
        if meta.get(name).is_none() {
            continue;
        }
        let ph = phantom::load(name, 0.5).unwrap();
        let t = Instant::now();
        let k = phantom::compute_kspace(&ph, None);
        let secs = t.elapsed().as_secs_f64();
        for (i, tissue) in ph.tissues.iter().enumerate() {
            let reference = read_c64(&dir.join(format!("phref/{name}/{tissue}.bin")));
            let e = rel_err(&k.kspace[i].data, &reference);
            worst = worst.max(e);
            eprintln!("{name:12} {tissue:28} rel. error {e:.2e}");
        }
        eprintln!("{name}: phantom k-space computed in {secs:.2} s");
    }
    assert!(worst < 1e-9, "worst relative error {worst:e}");
}

#[test]
fn images_match_python() {
    let Some(dir) = pyref() else {
        eprintln!("SPINSIGHT_PYREF not set; skipping");
        return;
    };
    let meta: J = serde_json::from_str(&std::fs::read_to_string(dir.join("imgref/meta.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut phantoms = std::collections::HashMap::new();
    for (config, m) in meta.as_object().unwrap() {
        let mut inp = Inputs::default();
        for (k, v) in m["inputs"].as_object().unwrap() {
            let v = match v {
                J::Number(n) => Value::Num(n.as_f64().unwrap()),
                J::Bool(b) => Value::Bool(*b),
                J::String(s) => Value::Str(s.clone()),
                _ => continue,
            };
            inp.set(k, &v).unwrap();
        }
        let ph = phantom::load(&inp.object, inp.min_voxel_size).unwrap();
        let pk = phantoms.entry(inp.object.clone()).or_insert_with(|| phantom::load_kspace(&ph, None)).clone();
        let p = protocol::compute(&inp, &ph);
        let mut sim = Simulator::default();
        let t = Instant::now();
        let im = sim.run(&p, &inp, &ph, &pk);
        let secs = t.elapsed().as_secs_f64();
        let measured = read_c64(&dir.join(format!("imgref/{config}.measured_kspace.bin")));
        let gridded = read_c64(&dir.join(format!("imgref/{config}.gridded_kspace.bin")));
        let image = read_c64(&dir.join(format!("imgref/{config}.image_array.bin")));
        let shape_ok = im.gridded.data.len() == gridded.len() && im.image_array.data.len() == image.len() && im.measured.len() == measured.len();
        if !shape_ok {
            failures.push(format!("{config}: shape mismatch"));
            continue;
        }
        let (em, eg, ei) = (rel_err(&im.measured, &measured), rel_err(&im.gridded.data, &gridded), rel_err(&im.image_array.data, &image));
        eprintln!(
            "{config:14} measured {em:.2e}  gridded {eg:.2e}  image {ei:.2e}   (rust {:.0} ms, python {:.0} ms)",
            secs * 1e3,
            m["time"].as_f64().unwrap() * 1e3
        );
        let tol = if p.is_radial { 1e-3 } else { 1e-9 };
        if em > tol || eg > tol || ei > tol {
            failures.push(format!("{config}: errors {em:e} {eg:e} {ei:e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
