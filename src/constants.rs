//! Physical constants, tissue parameters and hardware limits (port of `constants.py`).

/// 1H gyromagnetic ratio [MHz/T]
pub const GYRO: f64 = 42.577;

/// Ad hoc T2' value [msec]
pub const T2_PRIM: f64 = 35.0;

/// Maximum gradient amplitude [mT/m]
pub const MAX_AMP: f64 = 25.0;
/// Maximum gradient slew rate [T/m/s]
pub const MAX_SLEW: f64 = 80.0;
/// Make inversion slice 10% thicker
pub const INVERSION_THK_FACTOR: f64 = 1.1;
/// Spoiler gradient area [uTs/m]
pub const SPOILER_AREA: f64 = 30.0;
/// Crusher dephasing [cycles]
pub const CRUSHER_DEPHASING: f64 = 2.0;

/// The supported B0 field strengths [T]
pub const FIELD_STRENGTHS: [f64; 2] = [1.5, 3.0];

/// Index into the `[1.5 T, 3.0 T]` relaxation time tables
pub fn field_index(b0: f64) -> usize {
    if (b0 - 3.0).abs() < (b0 - 1.5).abs() {
        1
    } else {
        0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tissue {
    pub name: &'static str,
    /// proton density
    pub pd: f64,
    /// fat fraction
    pub ff: f64,
    /// T1 [msec] at 1.5 T and 3.0 T
    pub t1: [f64; 2],
    /// T2 [msec] at 1.5 T and 3.0 T
    pub t2: [f64; 2],
}

const fn t(name: &'static str, pd: f64, ff: f64, t1: [f64; 2], t2: [f64; 2]) -> Tissue {
    Tissue { name, pd, ff, t1, t2 }
}

// Tissue relaxation times from:
// Stanisz GJ, Odrobina EE, Pun J, Escaravage M, Graham SJ, Bronskill MJ, et al. "T1, T2 relaxation and magnetization transfer in tissue at 3T." Magn Reson Med. 2005 Sep;54(3):507–12.
// Steinhoff S, Zaitsev M, Zilles K, Shah N j. "Fast T1 mapping with volume coverage." Magn Reson Med. 2001;46(1):131–40.
// Wansapura JP, Holland SK, Dunn RS, Ball WS. "NMR relaxation times in the human brain at 3.0 tesla." J Magn Reson Imaging. 1999 Apr;9(4):531–8.
// Steen RG, Gronemeyer SA, Kingsley PB, Reddick WE, Langston JS, Taylor JS. "Precise and accurate measurement of proton T1 in human brain in vivo: Validation and preliminary clinical application." J Magn Reson Imaging. 1994;4(5):681–91.
// Gold GE, Han E, Stainsby J, Wright G, Brittain J, Beaulieu C. "Musculoskeletal MRI at 3.0 T: Relaxation Times and Image Contrast." American Journal of Roentgenology. 2004 Aug;183(2):343–51.
// de Bazelaire CMJ, Duhamel GD, Rofsky NM, Alsop DC. "MR imaging relaxation times of abdominal and pelvic tissues measured in vivo at 3.0 T: preliminary results." Radiology. 2004 Mar;230(3):652–9.
// Lin C, Bernstein M, Huston J, Fain S. "Measurements of T1 Relaxation times at 3.0T: Implications for clinical MRA." In: Proceedings of the 9th Annual Meeting of ISMRM. Glasgow, Scotland, United Kingdom; 2001. p. 1391.
pub const TISSUES: &[Tissue] = &[
    t("Gray matter", 1.0, 0.00, [1100.0, 1330.0], [95.0, 99.0]),
    t("White matter", 0.9, 0.00, [560.0, 830.0], [72.0, 69.0]),
    t("CSF", 1.0, 0.00, [4280.0, 4160.0], [2030.0, 2100.0]),
    t("Adipose tissue", 1.0, 0.95, [290.0, 370.0], [165.0, 130.0]),
    t("Bone marrow", 1.0, 0.50, [856.0, 898.0], [46.0, 34.0]), // relaxation times for water component
    t("Liver", 1.0, 0.01, [586.0, 809.0], [46.0, 34.0]),
    t("Spleen", 1.0, 0.00, [1057.0, 1328.0], [79.0, 61.0]),
    t("Muscle", 1.0, 0.00, [856.0, 898.0], [27.0, 29.0]),
    t("Kidney medulla", 1.0, 0.00, [1412.0, 1545.0], [85.0, 81.0]),
    t("Kidney cortex", 1.0, 0.00, [966.0, 1142.0], [87.0, 76.0]),
    t("Spinal cord", 1.0, 0.00, [745.0, 993.0], [74.0, 78.0]),
    t("Cortical bone", 0.05, 0.00, [1000.0, 1000.0], [3.0, 1.0]),
    t("Blood", 1.0, 0.00, [1441.0, 1932.0], [290.0, 275.0]),
    t("Stomach", 1.0, 0.00, [3000.0, 3000.0], [800.0, 800.0]),
    t("Perotineum", 1.0, 0.00, [1500.0, 1500.0], [30.0, 30.0]),
    // Shepp-Logan phantom tissues as defined in:
    // Gach, H., Costin Tanase, and Fernando E. Boada. 2008. "2D & 3D Shepp-Logan Phantom Standards for MRI." In: Int Conf Syst Eng, 521–26.
    t("Shepp-Logan Scalp", 0.8, 0.00, [343.0, 377.0], [70.0, 70.0]),
    t("Shepp-Logan Bone and Marrow", 0.12, 0.00, [552.0, 587.0], [50.0, 50.0]),
    t("Shepp-Logan CSF", 0.98, 0.00, [4200.0, 4200.0], [1990.0, 1990.0]),
    t("Shepp-Logan Gray matter", 0.745, 0.00, [998.0, 1295.0], [100.0, 100.0]),
    t("Shepp-Logan White matter", 0.617, 0.00, [681.0, 887.0], [80.0, 80.0]),
    t("Shepp-Logan Tumor", 0.95, 0.00, [1011.0, 1175.0], [100.0, 100.0]),
];

pub fn tissue(name: &str) -> Option<&'static Tissue> {
    TISSUES.iter().find(|t| t.name == name)
}

#[derive(Debug, Clone, Copy)]
pub struct FatResonance {
    pub name: &'static str,
    /// chemical shift relative to water [ppm]
    pub shift: f64,
    pub ratio: f64,
    pub ratio_with_fatsat: f64,
    pub pd: f64,
    pub t1: [f64; 2],
    pub t2: [f64; 2],
}

const ADIPOSE_T1: [f64; 2] = [290.0, 370.0];
const ADIPOSE_T2: [f64; 2] = [165.0, 130.0];

const fn fat(name: &'static str, shift: f64, ratio: f64, ratio_with_fatsat: f64) -> FatResonance {
    FatResonance { name, shift, ratio, ratio_with_fatsat, pd: 1.0, t1: ADIPOSE_T1, t2: ADIPOSE_T2 }
}

// Fat spectral model defined in:
// Cui C, Wu X, Newell JD, Jacob M. "Fat water decomposition using globally optimal surface estimation (GOOSE) algorithm." Magn Reson Med. 2015 Mar;73(3):1289–99.
pub const FAT_RESONANCES: [FatResonance; 6] = [
    fat("Fat1", 0.9 - 4.7, 0.087, 0.010),
    fat("Fat2", 1.3 - 4.7, 0.694, 0.033),
    fat("Fat3", 2.1 - 4.7, 0.129, 0.038),
    fat("Fat4", 2.8 - 4.7, 0.004, 0.003),
    fat("Fat5", 4.3 - 4.7, 0.039, 0.037),
    fat("Fat6", 5.3 - 4.7, 0.047, 0.045),
];

/// Complex-to-real display operators (`OPERATORS` in the Python version)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Magnitude,
    Phase,
    Real,
    Imaginary,
}

impl Operator {
    pub const ALL: [&'static str; 4] = ["Magnitude", "Phase", "Real", "Imaginary"];

    pub fn from_name(name: &str) -> Operator {
        match name {
            "Phase" => Operator::Phase,
            "Real" => Operator::Real,
            "Imaginary" => Operator::Imaginary,
            _ => Operator::Magnitude,
        }
    }

    pub fn apply(self, z: rustfft::num_complex::Complex64) -> f64 {
        match self {
            Operator::Magnitude => z.norm(),
            Operator::Phase => z.im.atan2(z.re),
            Operator::Real => z.re,
            Operator::Imaginary => z.im,
        }
    }
}
