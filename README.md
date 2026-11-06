# detector_stats

Calibration-free estimation of detector response statistics—absorbed quantum gain ($G$), cascade factor ($\gamma$), and pixel noise factor ($\psi$)—directly from diffraction rotation-scan datasets.

## Overview

Following the detector response model of Waterman & Evans (2010), *J. Appl. Cryst.* **43**, 1356–1371, the signal variance on an integrating or charge-sharing detector is governed by:

$$\sigma_p^2 = \bar{p} \, G \gamma + N \psi$$

Local single-frame dispersion estimators (such as `dials.estimate_gain`) underestimate the true response product $G\gamma$ whenever a point-spread function (PSF) or charge-sharing kernel $K(\mathbf{u})$ introduces spatial correlations between neighboring pixels, damping single-pixel variance by the zero-lag PSF autocorrelation factor $S_2 = \alpha(\mathbf{0}) = \sum_{\mathbf{v}} K(\mathbf{v})^2 < 1$.

`detector_stats` treats a rotation scan as an ensemble of coupled pixel time series to separate smooth spatiotemporal background trends and transient outliers (Bragg reflections, cosmic-ray zingers) from stochastic detector fluctuations, recovering $(G, \gamma, \psi)$ and the spatial autocorrelation kernel $\alpha(\mathbf{u})$ via lag-summed spatial cross-covariance across intensity regimes.

---

## Environment & Installation

### Standard Python vs. DIALS Python

- **Standard Python ($\ge 3.10$):** The core `detector_stats` package (including the synthetic rotation-scan simulator and statistical estimators) is pure Python and depends **only on `numpy` and `scipy`**. You can run it in any standard Python virtual environment or conda environment without DIALS installed.
- **DIALS Python (`dials.python` / DIALS `conda_base`):** Recommended if you want to run the DIALS comparison tests (e.g., comparing against `dials.algorithms.image.threshold.DispersionThresholdDebug` used by `dials.estimate_gain`) or load experimental diffraction datasets via `dxtbx`. If DIALS is not present in your environment, DIALS-specific tests are automatically skipped by `pytest`.

### Installing into a Standard Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

### Installing into an Existing DIALS Environment

If you have a DIALS installation (for example, a DIALS `conda_base` environment or developer build):

```bash
# Using the pip inside DIALS's conda_base environment:
/path/to/dials/conda_base/bin/pip install -e .

# Or, with a DIALS environment sourced (source /path/to/dials/dials_env.sh):
libtbx.pip install -e .
```

---

## Simulator Usage (`detector_stats.simulator`)

The `detector_stats.simulator` module generates synthetic diffraction rotation-scan image stacks $\mathbf{Y}(\mathbf{x}, t)$ with exact ground-truth detector response statistics $(G, \gamma, \psi)$, 2D point-spread kernels $K(\mathbf{u})$, non-stationary incident beam flux $I_0(t)$, radially varying and rotating background scatter, transient 3D Bragg reflections, single-frame cosmic-ray zingers, and static bad/shadowed pixels.

### 1. Simulating a Rotation Scan with Built-in Detector Presets

Three benchmark detector presets are provided out of the box:

- `DetectorConfig.ideal_pad()`: Ideal photon-counting Pixel Array Detector ($G = 1.0$, $\gamma = 1.0$, $\psi = 0.0$, $\delta$-PSF with $S_2 = 1.0$).
- `DetectorConfig.pt_ccd_waterman_evans()`: Phosphor-taper-CCD parameterized after Waterman & Evans (2010) ($G = 1.44\text{ ADU}$, $\gamma = 1.54$, $\psi = 9.08\text{ ADU}^2$, Lorentzian PSF with $\text{FWHM} = 1.3\text{ px}$ and $S_2 \approx 0.177$).
- `DetectorConfig.ed_pad_timepix()`: Hybrid PAD operated in 200 keV 3D electron diffraction mode after Vypritskaia et al. (2025) ($G = 2.9$, $\gamma = 1.35$, $\psi = 0.0$, discrete multi-pixel charge-sharing cluster PSF with $S_2 \approx 0.455$).

```python
from detector_stats.simulator import (
    DetectorConfig,
    SceneConfig,
    simulate_rotation_scan,
)

# 1. Choose a detector preset (or customize parameters)
detector = DetectorConfig.pt_ccd_waterman_evans(
    gain=1.44,
    cascade_factor=1.54,
    pedestal=50.0,
)

# 2. Configure the rotation-scan scene
scene = SceneConfig(
    n_frames=60,
    height=128,
    width=128,
    bg_min_quanta=2.0,          # Minimum background (absorbed quanta / pixel / frame)
    bg_max_quanta=60.0,         # Maximum background near beam center / solvent ring
    beam_drift_amplitude=0.05,  # 5% slow sinusoidal drift across scan
    beam_topup_period=25,       # Synchrotron top-up sawtooth period (frames)
    beam_topup_jump=0.04,       # 4% jump at top-up injection
    beam_jitter_sigma=0.01,     # 1% frame-to-frame random source jitter
    n_bragg_spots=80,           # Number of 3D Gaussian Bragg reflections
    n_zingers=15,               # Single-frame cosmic-ray spikes
    n_hot_pixels=10,
    n_dead_pixels=10,
    shadow_margin_px=4,
)

# 3. Run the simulation
result = simulate_rotation_scan(detector, scene, seed=42)

print("Image stack shape (T, H, W):", result.images.shape)
print("True G * gamma:", detector.g_gamma)
print("Zero-lag PSF factor S_2:", detector.s2_factor)
print("Apparent single-pixel dispersion slope:", detector.apparent_single_pixel_gain)
```

### 2. Inspecting the `SimulationResult`

`simulate_rotation_scan` returns a `SimulationResult` dataclass containing both the observed image stack and the exact ground-truth components for benchmarking estimators:

| Attribute | Shape / Type | Description |
| :--- | :--- | :--- |
| `result.images` | `(n_frames, height, width)` `float64` | Observed pixel intensities $Y(y, x, t)$ in ADU (including pedestal, noise, spots, zingers, and bad pixels). |
| `result.expected_background_adu` | `(n_frames, height, width)` `float64` | True noiseless expected background signal $\mu(y, x, t)$ in ADU above pedestal (after PSF blurring). |
| `result.incident_flux_scale` | `(n_frames,)` `float64` | True frame-by-frame global incident beam scale factor $I_0(t)$ (normalized to unit mean). |
| `result.spot_mask` | `(n_frames, height, width)` `bool` | Ground-truth 3D mask marking pixels contaminated by Bragg reflections or cosmic-ray zingers. |
| `result.static_bad_mask` | `(height, width)` `bool` | Ground-truth 2D mask marking static hot pixels, dead pixels, and shadowed detector borders. |

### 3. Customizing Detector Response & PSF Kernels

You can construct custom detectors with analytical or empirical 2D PSF kernels (`delta_psf`, `gaussian_psf`, `lorentzian_psf`, `timepix_cluster_psf`, or any normalized 2D NumPy array with odd dimensions) and query their theoretical spatial autocovariance:

```python
from detector_stats.simulator import DetectorConfig, gaussian_psf

custom_detector = DetectorConfig(
    name="custom_cmos",
    gain=2.2,
    cascade_factor=1.25,
    pixel_factor=3.5,
    pedestal=100.0,
    psf_kernel=gaussian_psf(sigma_px=0.85, radius=3),
    psf_mode="continuous",  # or "discrete_cluster" for integer cluster splitting
)

# Normalized 2D PSF autocorrelation alpha(u) = (K * K)(u)
alpha = custom_detector.psf_autocorrelation

# Exact theoretical 2D autocovariance matrix C(u; mu) at expected signal mu = 50 ADU
cov_matrix = custom_detector.theoretical_covariance_kernel(mu_adu=50.0)
```

---

## Estimating Detector Response Statistics (`detector_stats.pipeline`)

Use `estimate_detector_statistics` to run the complete four-stage pipeline (anomaly masking $\rightarrow$ spatiotemporal detrending $\rightarrow$ lagged cross-covariance regression $\rightarrow$ $G$/$\gamma$ separation) on any 3D rotation-scan image stack `(n_frames, height, width)`:

```python
from detector_stats import (
    DetectorConfig,
    SceneConfig,
    estimate_detector_statistics,
    plot_detector_stats_report,
    simulate_rotation_scan,
)

# Simulate a 3D ED Timepix rotation scan with a low-flux corner
det = DetectorConfig.ed_pad_timepix(gain=2.9, cascade_factor=1.35)
scene = SceneConfig(n_frames=60, height=128, width=128, include_sparse_corner=True)
sim = simulate_rotation_scan(det, scene, seed=42)

# Estimate (G, gamma, psi) directly from the image stack
report = estimate_detector_statistics(
    sim.images,
    pedestal=det.pedestal,
    trend_method="hollow_stencil_kriging",  # or "low_rank_spline", "temporal_difference"
)

print(f"Estimated G * gamma : {report.g_gamma:.3f} +- {report.covariance.g_gamma_se:.3f}")
print(f"Estimated gain G    : {report.gain:.3f} (method: {report.separation.separation_method})")
print(f"Estimated cascade γ : {report.cascade_factor:.3f}")
print(f"Estimated noise ψ   : {report.psi:.3f} +- {report.covariance.psi_se:.3f}")
print(f"Detected PSF radius : {report.psf_radius} px (S_2 = {report.s2_factor:.3f})")

# Save an 8-panel diagnostic summary figure
plot_detector_stats_report(
    report,
    output_path="diagnostic_report.png",
    title="3D ED Timepix Rotation Scan",
    true_detector=det,
    true_flux=sim.incident_flux_scale,
)
```

---

## Running Tests

Run the test suite with `pytest`:

```bash
# Using a standard Python environment:
pytest

# Or using DIALS's conda_base pytest (includes the DIALS DispersionThresholdDebug comparison test):
/path/to/dials/conda_base/bin/pytest -v
```
