//! Help texts shown in help mode (port of `param_tooltips.py`). Generated from the Python source.

pub const TOOLTIPS: &[(&str, &str)] = &[
    (
        "load_presets",
        r#"# Load preset parameters
Loads preset parameters typical for clinical sequences."#,
    ),
    (
        "object",
        r#"# Phantom object
Selects the numerical phantom to be imaged. The phantom object determines the spatial distribution and MR properties of the simulated tissues."#,
    ),
    (
        "field_strength",
        r#"# B0 field strength
Main magnetic field strength, *B₀*, measured in tesla (T). Field strength affects signal level, resonance frequencies, and tissue relaxation properties."#,
    ),
    (
        "parameter_style",
        r#"# Parameter style
Determines which acquisition parameters are available, reflecting the styles of different vendors. Specifically if the receiver bandwidth is set in terms of the pixel bandwidth [Hz/pixel], the field of view (FOV) bandwidth [±kHz/FOV], or fat/water shift [pixels]; and if the spatial resolution is set in terms of matrix or voxel size."#,
    ),
    (
        "min_voxel_size",
        r#"# Minimum voxel size
Minimum voxel size used internally to represent the simulated object. Smaller values provide greater spatial accuracy but require more memory and computation time when loading a phantom for the first time."#,
    ),
    (
        "noise_gain",
        r#"# Noise gain
Controls the amount of simulated image noise. Increasing the noise gain reduces the signal-to-noise ratio (SNR)."#,
    ),
    (
        "sequence_type",
        r#"# Pulse sequence
Selects the type of MR pulse sequence. The sequence type is reflected in the pulse sequence diagram and determines the MR signal and which acquisition parameters are available."#,
    ),
    (
        "pixel_bandwidth",
        r#"# Pixel bandwidth
Receiver bandwidth per acquired pixel in the frequency-encoding direction. Corresponds to the frequency range across a pixel when the readout gradient is switched on. A higher bandwidth shortens the signal readout, which reduces both the chemical shift displacement and the signal-to-noise ratio (SNR)."#,
    ),
    (
        "FOV_bandwidth",
        r#"# FOV bandwidth
Total receiver bandwidth across the field of view (FOV) in the frequency-encoding direction. Corresponds to the frequency range over the FOV when the readout gradient is switched on. A higher bandwidth shortens the signal readout, which reduces both the chemical shift displacement and the signal-to-noise ratio (SNR)."#,
    ),
    (
        "FW_shift",
        r#"# Fat/water shift
Displacement between fat and water in the frequency-encoding direction due to chemical shift. Fat/water whift is determined by the receiver bandwidth. Therefore, reducing the shift corresponds to increasing the bandwidth which results in shorter signal readouts and lower signal-to-noise ratio (SNR)."#,
    ),
    (
        "NSA",
        r#"# Number of signal averages (NSA)
Increasing the NSA improves the signal-to-noise ratio (SNR) by averaging repeated measurements, but increases scan time."#,
    ),
    (
        "partial_Fourier_factor",
        r#"# Partial Fourier factor
Partiel Fourier imaging implies only a fraction of the *k*-space is acquired, such that one half is sampled fully, but the other only partially. The unmeasured *k*-space lines are synthesized from the measured based on the assumption that the *k*-space data is *conjugate symmetric*. Reducing the Partial Fourier factor enables shortening the scan time and the effective echo time if an EPI- or turbo spin echo train is used."#,
    ),
    (
        "turbo_factor",
        r#"# Turbo factor
The spin echo train length, equal to the number of refocusing pulses per shot, which results in the same number of spin echoes. A turbo factor > 1 results in a turbo/fast spin echo sequence. Since multiple echoes are acquired, the number of shots required to sample *k*-space decreases, which reduces the scan time."#,
    ),
    (
        "EPI_factor",
        r#"# EPI factor
The gradient echo train length, equal to the number of *k*-space lines acquired per gradient echo train. Since multiple echoes are acquired, the number of shots required to sample *k*-space decreases, which reduces the scan time. Since spin dephasing proceeds throughout the echo train, increasing the gradient echo train length results in more severe artifacts.

Gradient echo trains are used in EPI sequences (Echo Planar Imaging)."#,
    ),
    (
        "echo_order",
        r#"# Echo order
Determines the order in which the *k*-space lines are acquired during a spin echo train.

**Linear**: the *k*-space lines are acquired according to their position in *k*-space. This implies that central *k*-space is acquired in the middle of the echo train, which makes the shortest achievable effective echo time increase with echo train length. This scheme minimizes T₂-blurring.

**Centric**: the *k*-space lines are acquired according to their distance from the *k*-space center. This implies that central *k*-space is acquired first, which minimizes the shortest achievable effective echo time.

**Flexible**: the *k*-space lines are acquired to achieve the prescribed effective echo time while minimizing the echo train duration and avoiding T₂-blurring."#,
    ),
    (
        "FatSat",
        r#"# Fat saturation
"FatSat" suppresses the signal from fat before the excitation pulse using a chemical-shift-selective RF pulse that tips the fat magnetization into the transverse plane, where it is dephased by a spoiler gradient, effectively destroying the fat signal."#,
    ),
    (
        "TR",
        r#"# Repetition time (TR)
The time after which the core sequence (a shot) is repeated. TR determines how much time is available for T₁ relaxation (recovery of longitudinal magnetization) and therefore determines the degree of T₁-weighting. It also defines the duration of each shot and therefore determines the scan time."#,
    ),
    (
        "TE",
        r#"# Echo time (TE)
The time between the excitation pulse and the acquired echo. During this time, T₂ relaxation (decay of transverse magnetization) occurs. Therefore, TE determines the degree of T₂-weighting (T₂*-weighting for spoiled gradient echo sequences).

If multiple echoes are sampled, the *effective TE* corresponds to TE for the echo closest to the *k*-space center."#,
    ),
    (
        "TI",
        r#"# Inversion time (TI)
The time between the opening 180° inversion pulse and the subsequent excitation pulse. After the inversion pulse, the now negative longitudinal magnetization relaxes through zero towards its positive equilibrium state, according to each tissue's T₁ relaxation time. Therefore, together with TR, TI determines the distinctive form of T₁ weighting characteristic of Inversion Recovery sequences. This includes the possibility of suppressing particular tissues on the basis of their T₁ relaxation time."#,
    ),
    (
        "FA",
        r#"# Flip angle
Angle through which the net magnetization is rotated by the excitation RF-pulse. Together with TR, it influences signal strength and degree of T₁ weighting."#,
    ),
    (
        "trajectory",
        r#"# *k*-space trajectory
Selects the path through *k*-space used to acquire the MR data.

**Cartesian**: A rectangular *k*-space grid is sampled line by line.

**Radial**: Lines passing through central *k*-space are measured at different angles to sample a star-shaped disc.

**PROPELLER**: Paralell *k*-space lines are sampled in each shot, but the angle varies between shots so that each shot forms a "stripe" or "blade" through central *k*-space. (*PROPELLER = Periodically Rotated Overlapping ParallEL Lines with Enhanced Reconstruction*)

"#,
    ),
    (
        "frequency_direction",
        r#"# Frequency encoding direction
Selects the spatial direction used for frequency encoding (i.e. in which the frequency/readout gradient is applied). The perpendicular in-plane direction is used for phase encoding."#,
    ),
    (
        "FOV_P",
        r#"# Field of view 
## (phase-encoding direction)
Field of view (FOV) in the phase-encoding direction. It determines the spatial extent of the reconstructed image in this direction."#,
    ),
    (
        "FOV_F",
        r#"# Field of view 
## (frequency-encoding direction)
Field of view (FOV) in the frequency-encoding direction. It determines the spatial extent of the reconstructed image in this direction."#,
    ),
    (
        "phase_oversampling",
        r#"# Phase oversampling
Adds additional sampling in the phase-encoding direction beyond the prescribed field of view (FOV), so that the actually acquired FOV becomes larger than the prescribed / reconstructed FOV. This reduces wrap-around (aliasing) from anatomy outside the FOV, and increases the signal-to-noise ratio (SNR) at the cost of longer acquisition time."#,
    ),
    (
        "radial_sampling_factor",
        r#"# Radial sampling factor
Number of acquired radial spokes or blades relative to full *k*-space coverage (Nyquist sampling limit). Increasing the radial sampling factor reduces streaking artifacts and increases signal-to-noise ratio (SNR) at the cost of longer acquisition times."#,
    ),
    (
        "matrix_P",
        r#"# Acquisition matrix
## (phase-encoding direction)
Nominal size of the acquisition matrix in the phase encoding direction, i.e. the number of phase-encoding steps before any phase over- or undersampling. Determines the spatial resolution in this direction together with the field of view (FOV)."#,
    ),
    (
        "matrix_F",
        r#"# Acquisition matrix
## (frequency-encoding direction)
Number of acquired samples along the frequency-encoding direction. Determines the spatial resolution in this direction together with the field of view (FOV)."#,
    ),
    (
        "voxel_P",
        r#"# Voxel size
## (phase-encoding direction)
Acquired voxel size in the phase-encoding direction, defining the spatial resolution in this direction. It also determines the acquisition matrix together with the field of view (FOV)."#,
    ),
    (
        "voxel_F",
        r#"# Voxel size
## (frequency-encoding direction)
Acquired voxel size in the frequency-encoding direction, defining the spatial resolution in this direction. It also determines the acquisition matrix together with the field of view (FOV)."#,
    ),
    (
        "recon_matrix_P",
        r#"# Reconstruction matrix
## (phase-encoding direction)
Number of pixels in the reconstructed image in the phase-encoding direction. If larger than the acquisition matrix, *zero filling* of *k*-space is used."#,
    ),
    (
        "recon_matrix_F",
        r#"# Reconstruction matrix
## (frequency-encoding direction)
Number of pixels in the reconstructed image in the frequency-encoding direction. If larger than the acquisition matrix, *zero filling* of *k*-space is used."#,
    ),
    (
        "recon_voxel_P",
        r#"# Reconstructed voxel size
## (phase-encoding direction)
Pixel size in the reconstructed image in the phase-encoding direction. If smaller than the acquired voxel size, *zero filling* of *k*-space is used."#,
    ),
    (
        "recon_voxel_F",
        r#"# Reconstructed voxel size
## (frequency-encoding direction)
Pixel size in the reconstructed image in the frequency-encoding direction. If smaller than the acquired voxel size, *zero filling* of *k*-space is used."#,
    ),
    (
        "slice_thickness",
        r#"# Slice thickness
Prescribed thickness of the excited slice, defining the strength of the slice encoding gradient. Increasing the slice thickness improves the signal strength but reduces spatial resolution in the direction through the slice."#,
    ),
    (
        "radial_FOV_oversampling",
        r#"# Radial FOV oversampling factor
Extends the sampled field of view (FOV) for radial imaging in the readout direction to reduce aliasing from signal outside the prescribed FOV."#,
    ),
    (
        "show_FOV",
        r#"# Show FOV
Indicate the prescribed (yellow) and oversampled (light blue) field of view (FOV) on the image display."#,
    ),
    (
        "reference_tissue",
        r#"# Reference tissue
Selects the tissue used as reference when calculating the relative signal-to-noise ratio (SNR)."#,
    ),
    (
        "image_mode",
        r#"# Image mode
Selects which component of the reconstructed complex MR image is displayed.

**Magntiude**: The size of the transverse magnetization during sampling.

**Phase**: The direction of the magnetization in the transverse plane during sampling.

**Real**: The component of the transverse magnetization along one axis of the transverse plane.

**Imaginary**: The component along the perpendicular axis.

"#,
    ),
    (
        "show_processed_kspace",
        r#"# Show processed *k*-space
Displays *k*-space after processing rather than the originally measured *k*-space data."#,
    ),
    (
        "kspace_exponent",
        r#"# *k*-space exponent
Applies nonlinear intensity scaling to the *k*-space display to make low-signal values more visible. Affects visualization only."#,
    ),
    (
        "kspace_mode",
        r#"# *k*-space mode
Selects which component of the *k*-space data is displayed.

**Magntiude**: The size of the sampled signal.

**Phase**: The direction of the sampled signal in the transverse plane.

**Real**: The component of the sampled signal along one axis of the transverse plane.

**Imaginary**: The component along the perpendicular axis.

"#,
    ),
    (
        "homodyne",
        r#"# Homodyne
Applies so-called *homodyne reconstruction* to synthesize missing *k*-space data in partial Fourier acquisitions."#,
    ),
    (
        "do_apodize",
        r#"# Apodization
Applies a so-called *apodization filter* to *k*-space before image reconstruction to dampen the edge of the data matrix. This reduces Gibbs ringing artifact at the expense of some spatial resolution."#,
    ),
    (
        "apodization_alpha",
        r#"# Apodization alpha
Controls the strength of the *k*-space apodization filter."#,
    ),
    (
        "do_zerofill",
        r#"# Zerofill
Pad the *k*-space matrix with zeroes before reconstruction, according to the reconstructed matrix size. This results in smaller reconstructed pixels and reduces pixelization artifacts, but does not increase the spatial resolution."#,
    ),
    (
        "shot",
        r#"# Displayed shot
Selects which acquisition shot is displayed in the pulse sequence diagram."#,
    ),
    (
        "signal_exponent",
        r#"# Signal exponent
Applies nonlinear intensity scaling to the displayed signal to make low-signal values more visible. Affects visualization only."#,
    ),
    (
        "slice_orientation",
        r#"# Slice orientation
Main orientation of the slices in the 3D phantom: transversal (axial), sagittal or coronal. The slices can additionally be angulated (tilted) towards the other orientations. In the planning view, the slices are shown in yellow in the three localizer images."#,
    ),
    (
        "slice_angle_1",
        r#"# Angulation
Tilts the slices about the first in-plane axis towards the orientation given in the label (e.g. *Tra>Cor*: a transversal slice tilted towards coronal). Can also be set by dragging the ⟲ handle in the localizer images. Oblique slices are used e.g. to align brain slices with the AC–PC line."#,
    ),
    (
        "slice_angle_2",
        r#"# Angulation
Tilts the slices about the second in-plane axis towards the orientation given in the label (e.g. *Tra>Sag*). Can also be set by dragging the ⟲ handle in the localizer images."#,
    ),
    (
        "slice_pos_LR",
        r#"# Slice position R/L
Position of the centre of the slice stack in the right–left direction of the patient (R = right, L = left), in mm from the centre of the phantom. Can also be set by dragging the slices in the localizer images."#,
    ),
    (
        "slice_pos_AP",
        r#"# Slice position A/P
Position of the centre of the slice stack in the anterior–posterior direction of the patient (A = anterior, P = posterior), in mm. Can also be set by dragging the slices in the localizer images."#,
    ),
    (
        "slice_pos_HF",
        r#"# Slice position H/F
Position of the centre of the slice stack in the head–feet direction of the patient (H = head, F = feet), in mm. Can also be set by dragging the slices in the localizer images."#,
    ),
    (
        "num_slices",
        r#"# Number of slices
Number of parallel slices in the stack. In multi-slice imaging, the other slices are excited during the waiting time of each repetition (TR), so additional slices are free as long as they fit into TR. If they do not fit, the acquisition is split into several *concatenations*, which multiplies the scan time."#,
    ),
    (
        "slice_gap",
        r#"# Distance factor
Gap between neighbouring slices in percent of the slice thickness. A gap reduces cross-talk between slices (imperfect slice profiles) and lets a stack with fewer slices cover more of the anatomy, but tissue in the gap is not imaged."#,
    ),
    (
        "displayed_slice",
        r#"# Displayed slice
Selects which slice of the stack is simulated live and shown as MR image. Press *▶ Scan* to acquire all slices as a series of the examination."#,
    ),
];

pub fn tooltip(name: &str) -> Option<&'static str> {
    TOOLTIPS.iter().find(|t| t.0 == name).map(|t| t.1)
}
