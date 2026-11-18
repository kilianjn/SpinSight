[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![](docs/screenshot.png)

# SpinSight MRI simulator
SpinSight is an MRI simulator created for educational puposes. It jointly visualizes the imaging parameters, the MRI pulse sequence, the k-space data matrix, and the MR image. These are updated in real-time when the user changes parameters. The data is simulated from computational 2D phantoms in vector graphics format (SVG).

This repository contains two implementations:

* **SpinSight for desktop (Rust)** – a native application shipped as a single standalone executable for Windows, macOS and Linux. It is a port of the Python simulator and computes the same images, but much faster (see [Performance](#performance)).
* **SpinSight for the web browser (Python)** – the original implementation in [spinsight/](./spinsight), run as a browser dashboard (see [Python version](#python-version)).

## Download (desktop version)
Get the file for your computer from the **[latest release](../../releases/latest)** – nothing to install:

| Computer | File | How to start |
|---|---|---|
| Windows 10/11 (64-bit) | `SpinSight-windows-x64.exe` | Double-click. SmartScreen may warn about an unknown publisher: *More info → Run anyway*. |
| macOS 11+ (Apple Silicon and Intel) | `SpinSight-macos-universal.zip` | Unzip and move *SpinSight.app* to *Applications*. The app is not notarised: the first time, right-click → *Open*. |
| Linux x86-64 | `SpinSight-linux-x86_64.tar.gz` | `tar xzf SpinSight-linux-x86_64.tar.gz && ./SpinSight-linux-x86_64/spinsight` |

Help mode (**? Help** in the top bar) provides explanations of each parameter; hovering the help icon of TE, TR or TI highlights the interval in the pulse sequence diagram. The **☰** button opens the settings panel with phantom, field strength, parameter style, presets and loading/saving of settings files (TOML, compatible with the Python version).

### Command line options
```
spinsight [OPTIONS]
    -m, --mode <dark|light>       GUI theme (default: follow the system)
    -s, --settingsFile <FILE>     settings file (.toml) to load at startup
    -t, --slidersWithText         add text boxes next to sliders
        --show-sequence           show the pulse sequence diagram at startup
        --show-kspace             show k-space at startup
        --help-mode               start in help mode
        --set <name=value>        set a parameter at startup (repeatable), e.g. --set TE=80
        --export <FILE.png>       render the MR image to a PNG file and exit (no GUI)
        --export-kspace <FILE>    also write the k-space display to a PNG file
        --benchmark               print computation times and exit
```
Example: `spinsight --set load_presets=T2-FLAIR --set object=abdomen --export flair.png` renders an image without opening a window.

## Performance
The desktop version is a faithful port of the Python computation graph: the same sequence timing and parameter bounds, the same analytic phantom k-space, and the same reconstruction. The test suite cross-validates it against values computed with the Python version (protocol values agree to 10⁻⁹, Cartesian images to 10⁻¹², radial images to about 10⁻⁵, limited by the NUFFT accuracy). Measured on the same 4-core machine:

| Computation | Python | Rust |
|---|---:|---:|
| Phantom k-space, brain (first-time loading) | 40.9 s | 0.6 s |
| Phantom k-space, abdomen (first-time loading) | 30.7 s | 1.8 s |
| Image, default spin echo | 0.27 s | 0.02 s |
| Image, abdomen turbo spin echo | 0.75 s | 0.08 s |
| Image, radial | 3.8 s | 0.11 s |
| Image, PROPELLER | 4.7 s | 0.14 s |

The Python times exclude the time needed to render the browser dashboard. In the desktop version, changing contrast parameters (TE, TR, flip angle, …) re-uses the cached k-space and takes 6–20 ms, so images update continuously while a slider is dragged. Phantom k-space is cached on disk after the first computation.

How the speed-up was achieved:
* polygon Fourier transforms are evaluated with a separable vertex formulation (no trigonometric functions per grid point and edge), in parallel over all shapes;
* sinc interpolation uses optimized matrix multiplication; FFTs use [RustFFT](https://crates.io/crates/rustfft);
* radial/PROPELLER gridding uses a built-in 2D non-uniform FFT (exponential-of-semicircle kernel, as in FINUFFT) with Voronoi density compensation computed from a Delaunay triangulation;
* intermediate results (resampled phantom k-space, gridding operator, noise) are cached and the simulation runs in a background thread.

## Building the desktop version from source
Install [Rust](https://rustup.rs) (1.95 or newer), then
```
cargo run --release
```
No other libraries are needed for building. On Linux the program needs X11 or Wayland with OpenGL at runtime. Run the tests with `cargo test --release`; with the environment variable `SPINSIGHT_PYREF` pointing to raw arrays dumped from the Python version, `tests/reference_images.rs` additionally compares phantom k-space and images.

The release workflow ([.github/workflows/release.yml](./.github/workflows/release.yml)) builds the executables for all platforms; pushing a tag `v*` (or to `main`) publishes them as a GitHub release.

### Source overview
| Module | Python counterpart | Content |
|---|---|---|
| `src/protocol.rs` | `nodes/*` (scalar part), `sequence.py` | derived parameters, pulse sequence timing, parameter bounds, SNR, scan time |
| `src/imaging.rs` | `nodes/kspace_simulation.py`, `kspace_processing.py`, `image_reconstruction.py` | k-space sampling, contrast, noise, homodyne, apodization, zero filling, reconstruction |
| `src/phantom.rs`, `src/svg.rs` | `phantom.py` | phantom shapes and analytic k-space |
| `src/nufft.rs`, `src/voronoi.rs` | `recon.py` (finufft / mri-nufft) | non-Cartesian gridding |
| `src/controller.rs` | `controller.py` | parameter styles, bounds, visibility, presets, settings files |
| `src/gui/` | `gui.py`, `nodes/*_plot.py` | the egui user interface |

## Phantom construction
To create a new phantom, add a directory with the phantom name containing files with the same name as the directory. For the Python version, the directory goes under [spinsight/phantoms](./spinsight/phantoms). The desktop version has the built-in phantoms compiled in and reads additional phantoms from a `phantoms` folder next to the executable, from the folder given by the environment variable `SPINSIGHT_PHANTOMS`, or from `phantoms` in the SpinSight data directory (`~/.local/share/spinsight` on Linux, `~/Library/Application Support/se.uu.SpinSight` on macOS, `%APPDATA%\uu\SpinSight\data` on Windows). Two formats are supported:

### A single `.toml` file
The file shall consist of a list of shapes (see [Shepp-Logan_shapes.toml](./spinsight/phantoms/Shepp-Logan/Shepp-Logan.toml) for reference). Each shape is assigned a tissue type which must be defined in the `TISSUES` dict in [constants.py](./spinsight/constants.py) (`TISSUES` in [src/constants.rs](./src/constants.rs) for the desktop version).

### A pair of `.toml` and `.svg` files
The `.toml` file specifies a mapping from hexcolors to tissues (see [brain.toml](./spinsight/phantoms/brain/brain.toml) for reference). The tissues must be defined in the `TISSUES` dict in [constants.py](./spinsight/constants.py) (`TISSUES` in [src/constants.rs](./src/constants.rs) for the desktop version).

The specified `.svg` file must meet the following specifications:
* All paths must be closed
* All paths must have a fill color matching a hexcolor defined in the `.toml` file (this defines the tissue).
* Only polygons are supported (not Bézier curves etc.)

## Python version
Install using pip:
```
pip install spinsight
```
Then run as a command line tool
```
spinsight
```
This serves SpinSight on the local host, so that the simulator can be run by navigating to [localhost](http://localhost) in the web browser. The same command line tool can be used to deploy the simulator on a local network, or on a web server (run `spinsight -h` for help). Be aware that several minutes are required upon loading a phantom for the first time.

Help mode can be enabled in the sidebar menu, to provide explanations of each parameter.

### Dependencies
See [pyproject.toml](./pyproject.toml), *dependencies* under heading **[project]**. The desktop version's dependencies are listed in [Cargo.toml](./Cargo.toml).

## How to cite
If you use this software in your research, please cite:

> Berglund J, Jain K, Sousa JM, Hedman K, Fahlström M. “SpinSight – An educational open-source MRI simulator with joint visualization of pulse sequence, k-space, and MR image. In: *Proceedings of the Annual Meeting of the International Society of Magnetic Resonance in Medicine*, Honolulu 2025, pp. 1968.

## License
SpinSight is distributed under the terms of the GNU General Public License. See [LICENSE.md](./LICENSE.md). The desktop version bundles the DejaVu Sans font ([license](./assets/fonts/LICENSE-DejaVu.txt)).

## Contact Information
Johan Berglund, Ph.D.  
Uppsala University Hospital,  
Uppsala, Sweden  
johan.berglund@akademiska.se

---
Copyright © 2021–2026 Johan Berglund.
