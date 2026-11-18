//! SpinSight MRI simulator – Rust port.
//!
//! The numerical core (sequence design, phantom k-space synthesis, MR signal simulation and
//! image reconstruction) is GUI-independent; the egui desktop application lives in `gui`.

pub mod array;
pub mod constants;
pub mod convert;
pub mod formatting;
pub mod imaging;
pub mod nufft;
pub mod params;
pub mod phantom;
pub mod presets;
pub mod protocol;
pub mod recon;
pub mod sequence;
pub mod svg;
pub mod voronoi;
