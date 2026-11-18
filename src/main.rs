//! SpinSight MRI simulator – desktop application entry point.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::time::Instant;

use spinsight::controller::Controller;
use spinsight::gui::{Options, SpinSightApp};
use spinsight::imaging::{self, Simulator};
use spinsight::params::Value;
use spinsight::{params, phantom, presets};

const HELP: &str = "\
SpinSight MRI simulator (Rust port)

USAGE:
    spinsight [OPTIONS]

OPTIONS:
    -m, --mode <dark|light>       GUI theme (default: follow the system)
    -s, --settingsFile <FILE>     settings file (.toml) to load at startup
    -t, --slidersWithText         add text boxes next to sliders
        --show-sequence           show the pulse sequence diagram at startup
        --show-kspace             show k-space at startup
        --help-mode               start in help mode
        --set <name=value>        set a parameter at startup (repeatable)
        --export <FILE.png>       render the MR image to a PNG file and exit (no GUI)
        --export-kspace <FILE>    also write the k-space display to a PNG file
        --benchmark               print computation times and exit
    -V, --version                 print the version
    -h, --help                    print this help

Extra phantoms are read from <data dir>/SpinSight/phantoms/<name>/<name>.svg|toml,
from a 'phantoms' folder next to the executable, or from $SPINSIGHT_PHANTOMS.
";

struct Cli {
    dark_mode: Option<bool>,
    settings: Option<PathBuf>,
    sliders_with_text: bool,
    export: Option<PathBuf>,
    export_kspace: Option<PathBuf>,
    sets: Vec<(String, String)>,
    benchmark: bool,
    show_sequence: bool,
    show_kspace: bool,
    help_mode: bool,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli { dark_mode: None, settings: None, sliders_with_text: false, export: None, export_kspace: None, sets: vec![], benchmark: false, show_sequence: false, show_kspace: false, help_mode: false };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("spinsight {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-m" | "--mode" => {
                cli.dark_mode = match value("--mode")?.as_str() {
                    "dark" => Some(true),
                    "light" => Some(false),
                    _ => return Err("GUI mode must be either \"dark\" or \"light\"".into()),
                }
            }
            "-s" | "--settingsFile" | "--settings" => {
                let mut p = PathBuf::from(value("--settingsFile")?);
                if p.extension().is_none() {
                    p.set_extension("toml");
                }
                cli.settings = Some(p);
            }
            "-t" | "--slidersWithText" => cli.sliders_with_text = true,
            "--export" => cli.export = Some(PathBuf::from(value("--export")?)),
            "--export-kspace" => cli.export_kspace = Some(PathBuf::from(value("--export-kspace")?)),
            "--set" => {
                let s = value("--set")?;
                let (k, v) = s.split_once('=').ok_or("--set expects name=value")?;
                cli.sets.push((k.trim().to_string(), v.trim().to_string()));
            }
            "--benchmark" => cli.benchmark = true,
            "--show-sequence" => cli.show_sequence = true,
            "--show-kspace" => cli.show_kspace = true,
            "--help-mode" => cli.help_mode = true,
            // options of the Python web server version
            "-b" | "--noBrowser" | "-n" | "--network" | "-l" | "--liveSliders" => eprintln!("note: option {a} only applies to the Python web version and is ignored"),
            "-p" | "--port" | "-u" | "--url" | "-i" | "--instances" => {
                let _ = value(&a);
                eprintln!("note: option {a} only applies to the Python web version and is ignored");
            }
            other => return Err(format!("unknown option {other} (see --help)")),
        }
    }
    Ok(cli)
}

fn save_png(path: &PathBuf, a: &spinsight::array::Array2<f64>, flip_vertical: bool) -> Result<(), String> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &a.data {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let scale = if hi > lo { 255.0 / (hi - lo) } else { 0.0 };
    let mut img = image::GrayImage::new(a.cols as u32, a.rows as u32);
    for r in 0..a.rows {
        for c in 0..a.cols {
            let v = *a.get(r, c);
            let y = if flip_vertical { a.rows - 1 - r } else { r };
            img.put_pixel(c as u32, y as u32, image::Luma([((v - lo) * scale).round().clamp(0.0, 255.0) as u8]));
        }
    }
    img.save(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Interpret a `--set name=value` string for parameter `name`
fn parse_set(k: &str, v: &str) -> Result<(String, Value), String> {
    let spec = params::try_param(k).ok_or_else(|| format!("unknown parameter {k}"))?;
    let value = match (&spec.widget, &spec.default) {
        (params::Widget::Checkbox, _) => Value::Bool(v == "true" || v == "1"),
        (_, Value::Str(_)) => Value::Str(v.to_string()),
        _ => Value::Num(v.parse::<f64>().map_err(|_| format!("{k}: expected a number, got {v:?}"))?),
    };
    Ok((spec.name.to_string(), value))
}

fn headless(cli: &Cli) -> Result<(), String> {
    let mut ctrl = Controller::new()?;
    if let Some(f) = &cli.settings {
        ctrl.load_settings_file(f)?;
    }
    for (k, v) in &cli.sets {
        let (name, value) = parse_set(k, v)?;
        ctrl.on_change(&name, value);
    }
    for m in ctrl.messages.drain(..) {
        eprintln!("{m}");
    }
    let t = Instant::now();
    let pk = phantom::load_kspace(&ctrl.phantom, None);
    let t_phantom = t.elapsed().as_secs_f64();
    let mut sim = Simulator::default();
    let out = sim.run(&ctrl.protocol, &ctrl.inputs, &ctrl.phantom, &pk);
    let img = imaging::display_image(&out, &ctrl.inputs.image_mode);
    if let Some(p) = &cli.export {
        save_png(p, &img, false)?;
        println!("wrote {}", p.display());
    }
    if let Some(p) = &cli.export_kspace {
        let (k, _) = imaging::display_kspace(&out, &ctrl.inputs.kspace_mode, ctrl.inputs.kspace_exponent, ctrl.inputs.show_processed_kspace);
        save_png(p, &k, true)?;
        println!("wrote {}", p.display());
    }
    println!("phantom k-space: {t_phantom:.3} s; simulation: {}", out.timings.iter().map(|(k, v)| format!("{k} {:.1} ms", v * 1e3)).collect::<Vec<_>>().join(", "));
    println!("relative SNR {}, scan time {}", ctrl.info.relative_snr, ctrl.info.scantime);
    Ok(())
}

fn benchmark() -> Result<(), String> {
    println!("SpinSight {} benchmark ({} threads)", env!("CARGO_PKG_VERSION"), rayon::current_num_threads());
    for name in phantom::builtin_phantom_names() {
        let ph = phantom::load(name, 0.5)?;
        let t = Instant::now();
        let _ = phantom::compute_kspace(&ph, None);
        println!("phantom k-space {name:12} {:>7.0} ms  (matrix {}x{}, {} shapes)", t.elapsed().as_secs_f64() * 1e3, ph.matrix[0], ph.matrix[1], ph.shapes.iter().map(|s| s.len()).sum::<usize>());
    }
    let ph = phantom::load("brain", 0.5)?;
    let pk = phantom::load_kspace(&ph, None);
    let cases: Vec<(&str, Vec<(&str, Value)>)> = vec![
        ("default spin echo", vec![]),
        ("T2w-TSE preset", presets::preset("T2w-TSE").unwrap().into_iter().map(|(k, v)| (params::param(&k).name, v)).collect()),
        ("radial", vec![("trajectory", Value::s("Radial"))]),
        ("PROPELLER", vec![("trajectory", Value::s("Radial")), ("turbo_factor", Value::Num(8.0))]),
    ];
    for (label, settings) in cases {
        let mut ctrl = Controller::new()?;
        ctrl.apply_settings(settings.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), false);
        let t = Instant::now();
        for _ in 0..100 {
            let _ = spinsight::protocol::compute(&ctrl.inputs, &ctrl.phantom);
        }
        let t_protocol = t.elapsed().as_secs_f64() / 100.0;
        let mut sim = Simulator::default();
        let t = Instant::now();
        let _ = sim.run(&ctrl.protocol, &ctrl.inputs, &ctrl.phantom, &pk);
        let cold = t.elapsed().as_secs_f64();
        let mut inp = ctrl.inputs.clone();
        inp.te = Some(inp.te.unwrap_or(20.0) + 1.0);
        let p = spinsight::protocol::compute(&inp, &ctrl.phantom);
        let t = Instant::now();
        let _ = sim.run(&p, &inp, &ctrl.phantom, &pk);
        let warm = t.elapsed().as_secs_f64();
        println!("{label:20} protocol {:>6.2} ms | image (new geometry) {:>6.0} ms | image (contrast change) {:>5.0} ms", t_protocol * 1e3, cold * 1e3, warm * 1e3);
    }
    Ok(())
}

fn load_icon() -> Option<egui::IconData> {
    let img = image::load_from_memory(include_bytes!("../assets/icon/icon_256.png")).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::IconData { rgba: img.into_raw(), width: w, height: h })
}

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("dejavu".into(), std::sync::Arc::new(egui::FontData::from_static(include_bytes!("../assets/fonts/DejaVuSans.ttf"))));
    // fallback for subscripts and symbols (T₁, T₂, B₀, μ, ±) missing in the default font
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("dejavu".into());
    }
    ctx.set_fonts(fonts);
}

fn main() {
    let cli = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    params::init(&phantom::phantom_names(), &presets::builtin_preset_names().iter().map(|s| s.to_string()).collect::<Vec<_>>());
    if cli.benchmark {
        if let Err(e) = benchmark() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if cli.export.is_some() || cli.export_kspace.is_some() {
        if let Err(e) = headless(&cli) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    let start_time = chrono_like_now();
    let mut viewport = egui::ViewportBuilder::default().with_title("SpinSight").with_inner_size([1680.0, 1000.0]).with_min_inner_size([800.0, 500.0]);
    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    let mut sets = Vec::new();
    for (k, v) in &cli.sets {
        match parse_set(k, v) {
            Ok(s) => sets.push(s),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        }
    }
    let opts = Options {
        dark_mode: cli.dark_mode,
        settings_file: cli.settings.clone(),
        sliders_with_text: cli.sliders_with_text,
        start_time,
        show_sequence: cli.show_sequence,
        show_kspace: cli.show_kspace,
        help_mode: cli.help_mode,
        sets,
    };
    let result = eframe::run_native(
        "SpinSight",
        options,
        Box::new(move |cc| {
            setup_fonts(&cc.egui_ctx);
            match SpinSightApp::new(cc, opts) {
                Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
                Err(e) => Err(e.into()),
            }
        }),
    );
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Local date and time "YYYY-MM-DD HH:MM:SS" (UTC offset not applied; no extra dependency)
fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", rem / 3600, (rem % 3600) / 60, rem % 60)
}
