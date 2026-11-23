//! Preset parameter sets typical for clinical sequences (port of `presets.py`).

use crate::params::Value;

/// The standard protocol (start-up and reset): a T1-weighted spin echo
pub const STANDARD_PRESET: &str = "T1w-SE";

const STANDARD_PROTOCOL: &str = r#"EPI_factor = 1
FatSat = false
NSA = 1
TE = 12.0
TR = 500.0
echo_order = "Linear"
frequency_direction = "anterior-posterior"
matrix_F = 256
matrix_P = 224
partial_Fourier_factor = 1.0
phase_oversampling = 0
pixel_bandwidth = 160.0
sequence_type = "Spin Echo"
slice_thickness = 4.0
trajectory = "Cartesian"
turbo_factor = 1
"#;

const BUILTIN: &[(&str, &str)] = &[
    ("GRASE", include_str!("../spinsight/presets/GRASE.toml")),
    ("GRE-EPI", include_str!("../spinsight/presets/GRE-EPI.toml")),
    ("T1w-GRE", include_str!("../spinsight/presets/T1w-GRE.toml")),
    (STANDARD_PRESET, STANDARD_PROTOCOL),
    ("T1w-TSE", include_str!("../spinsight/presets/T1w-TSE.toml")),
    ("T2-FLAIR", include_str!("../spinsight/presets/T2-FLAIR.toml")),
    ("T2w-TSE", include_str!("../spinsight/presets/T2w-TSE.toml")),
];

pub fn builtin_preset_names() -> Vec<&'static str> {
    BUILTIN.iter().map(|b| b.0).collect()
}

/// Parse a settings/preset TOML document into (parameter, value) pairs (in file order)
pub fn parse_settings(text: &str) -> Result<Vec<(String, Value)>, String> {
    let table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (k, v) in table {
        let value = match v {
            toml::Value::Float(f) => Value::Num(f),
            toml::Value::Integer(i) => Value::Num(i as f64),
            toml::Value::Boolean(b) => Value::Bool(b),
            toml::Value::String(s) => Value::Str(s),
            _ => continue,
        };
        out.push((k, value));
    }
    Ok(out)
}

pub fn preset(name: &str) -> Option<Vec<(String, Value)>> {
    BUILTIN.iter().find(|b| b.0 == name).and_then(|b| parse_settings(b.1).ok())
}

/// Serialize settings like Python's `toml.dump(dict(sorted(settings.items())))`
pub fn write_settings(settings: &[(String, Value, bool)]) -> String {
    let mut sorted: Vec<&(String, Value, bool)> = settings.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut s = String::new();
    for (k, v, integer) in sorted {
        let val = match v {
            Value::Num(x) if *integer => format!("{}", x.round() as i64),
            Value::Num(x) => {
                if x.fract() == 0.0 && x.abs() < 1e15 {
                    format!("{x:.1}")
                } else {
                    format!("{x}")
                }
            }
            Value::Bool(b) => format!("{b}"),
            Value::Str(t) => format!("\"{}\"", t.replace('\\', "\\\\").replace('"', "\\\"")),
        };
        s.push_str(&format!("{k} = {val}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse() {
        for name in builtin_preset_names() {
            let p = preset(name).unwrap();
            assert!(p.iter().any(|(k, _)| k == "TR"));
        }
        let p = preset("T2-FLAIR").unwrap();
        assert!(p.contains(&("TI".to_string(), Value::Num(2100.0))));
        assert!(p.contains(&("sequence_type".to_string(), Value::s("Inversion Recovery"))));
    }
}
