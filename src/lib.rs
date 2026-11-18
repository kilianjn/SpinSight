//! SpinSight MRI simulator – Rust port.
//!
//! The numerical core (sequence design, phantom k-space synthesis, MR signal simulation and
//! image reconstruction) is GUI-independent; the egui desktop application lives in `gui`.

// index loops mirror the numpy formulations of the Python version
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

pub mod array;
pub mod constants;
pub mod controller;
pub mod convert;
pub mod formatting;
pub mod gui;
pub mod imaging;
pub mod nufft;
pub mod params;
pub mod phantom;
pub mod phantom_maker;
pub mod presets;
pub mod protocol;
pub mod recon;
pub mod sequence;
pub mod svg;
pub mod tooltips;
pub mod voronoi;
pub mod worker;
