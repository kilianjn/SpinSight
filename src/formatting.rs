//! Value formatting for labels and indicators (port of `formatting.py`).

/// Round `value` to `sigfigs` significant figures (like Python's `float(f'{value:.{sigfigs}g}')`).
pub fn round_sig(value: f64, sigfigs: usize) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    format!("{:.*e}", sigfigs.saturating_sub(1), value).parse().unwrap_or(value)
}

pub fn format_float(value: f64, sigfigs: usize) -> String {
    let rounded = round_sig(value, sigfigs);
    if rounded == 0.0 || !rounded.is_finite() {
        return format!("{rounded}");
    }
    let s = format!("{rounded}");
    let (integer, decimal) = match s.split_once('.') {
        Some((i, d)) => (i.to_string(), d.to_string()),
        None => (s.clone(), "0".to_string()),
    };
    let exponent = rounded.abs().log10().floor() as i64;
    let num_decimals = sigfigs as i64 - exponent - 1;
    if num_decimals <= 0 {
        return integer;
    }
    let mut decimal = decimal;
    while (decimal.len() as i64) < num_decimals {
        decimal.push('0');
    }
    format!("{integer}.{decimal}")
}

pub fn scantime(milliseconds: f64) -> String {
    let total_seconds = milliseconds / 1000.0;
    let minutes = (total_seconds / 60.0).floor() as i64;
    let seconds = total_seconds.rem_euclid(60.0).floor() as i64;
    if minutes > 0 {
        format!("{minutes} min {seconds} sec")
    } else if seconds >= 10 {
        format!("{seconds} sec")
    } else if seconds > 0 {
        format!("{total_seconds:.1} sec")
    } else {
        format!("{milliseconds:.0} ms")
    }
}

/// Acquisition time like on a scanner console: "m:ss" (or "h:mm:ss")
pub fn ta(milliseconds: f64) -> String {
    let total = (milliseconds / 1000.0).round().max(0.0) as i64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

pub fn pixel_bandwidth(bw: f64) -> String {
    format!("{} Hz / pixel", format_float(bw, 3))
}

pub fn fov_bandwidth(fov_bw: f64) -> String {
    format!("±{} kHz", format_float(fov_bw, 3))
}

pub fn fw_shift(shift: f64) -> String {
    format!("{} pixels", format_float(shift, 3))
}

pub fn ts(t: f64) -> String {
    format!("{} ms", format_float(t, 2))
}

pub fn flip_angle(fa: f64) -> String {
    format!("{fa:.0}°")
}

pub fn spoke_angle(angle: f64) -> String {
    format!("{angle:.0}°")
}

pub fn num_shots(num: usize) -> String {
    format!("{num}")
}

pub fn fov(fov: f64) -> String {
    format!("{fov:.0} mm")
}

pub fn relative_snr(snr: f64) -> String {
    format!("{:.0}%", snr * 100.0)
}

pub fn percentage(factor: f64) -> String {
    format!("{:.0}%", factor * 100.0)
}

pub fn voxel_size(voxel: f64) -> String {
    format!("{} mm", format_float(voxel, 3))
}

pub fn slice_thickness(thk: f64) -> String {
    format!("{} mm", format_float(thk, 2))
}

pub fn seq_time(time: f64) -> String {
    format!("{time:.1} ms")
}

pub fn grad_area(area: f64) -> String {
    format!("{area:.1} μTs/m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_float_matches_python() {
        assert_eq!(format_float(480.0, 3), "480");
        assert_eq!(format_float(48.0, 3), "48.0");
        assert_eq!(format_float(0.45, 3), "0.450");
        assert_eq!(format_float(1.3333, 3), "1.33");
        assert_eq!(format_float(12.0, 2), "12");
        assert_eq!(format_float(4.6, 2), "4.6");
        assert_eq!(format_float(1250.0, 3), "1250");
        assert_eq!(scantime(1_620_000.0), "27 min 0 sec");
        assert_eq!(scantime(5500.0), "5.5 sec");
        assert_eq!(scantime(500.0), "500 ms");
        assert_eq!(ta(100_000.0), "1:40");
        assert_eq!(ta(3_725_000.0), "1:02:05");
    }
}
