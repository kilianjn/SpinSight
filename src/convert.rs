//! Conversions between bandwidth parameterizations (port of `convert.py`).

use crate::constants::{FAT_RESONANCES, GYRO};

fn fat2_shift() -> f64 {
    FAT_RESONANCES[1].shift
}

/// Fat/water chemical shift [pixels] from pixel bandwidth [Hz/pixel] and B0 [T]
pub fn pixel_bw_to_shift(pixel_bw: f64, b0: f64) -> f64 {
    (fat2_shift() * GYRO * b0 / pixel_bw).abs()
}

/// Pixel bandwidth [Hz/pixel] from fat/water chemical shift [pixels] and B0 [T]
pub fn shift_to_pixel_bw(shift: f64, b0: f64) -> f64 {
    (fat2_shift() * GYRO * b0 / shift).abs()
}

/// FOV bandwidth [±kHz] from pixel bandwidth [Hz/pixel] and read direction matrix
pub fn pixel_bw_to_fov_bw(pixel_bw: f64, matrix_f: f64) -> f64 {
    pixel_bw * matrix_f / 2e3
}

/// Pixel bandwidth [Hz/pixel] from FOV bandwidth [±kHz] and read direction matrix
pub fn fov_bw_to_pixel_bw(fov_bw: f64, matrix_f: f64) -> f64 {
    fov_bw / matrix_f * 2e3
}
