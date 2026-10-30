"""Unit and statistical verification tests for the synthetic diffraction rotation-scan simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from detector_stats.simulator import (
    DetectorConfig,
    SceneConfig,
    _sample_cascade_and_psf,
    delta_psf,
    gaussian_psf,
    lorentzian_psf,
    simulate_rotation_scan,
    timepix_cluster_psf,
)


def test_psf_kernels_and_autocorrelation_identities() -> None:
    """Verify normalization and autocorrelation sum identities for all PSF kernels."""
    for kernel in [
        delta_psf(0),
        delta_psf(2),
        gaussian_psf(1.1, radius=4),
        lorentzian_psf(1.3, radius=5),
        timepix_cluster_psf(0.66, radius=2),
    ]:
        assert kernel.ndim == 2
        assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
        assert np.isclose(np.sum(kernel), 1.0, atol=1e-12)

        det = DetectorConfig(gain=2.0, cascade_factor=1.4, psf_kernel=kernel)
        alpha = det.psf_autocorrelation
        # Sum of autocorrelation over all lags must equal (sum K)^2 == 1.0
        assert np.isclose(np.sum(alpha), 1.0, atol=1e-12)
        cy, cx = alpha.shape[0] // 2, alpha.shape[1] // 2
        assert np.isclose(alpha[cy, cx], det.s2_factor, atol=1e-12)
        assert 0.0 < det.s2_factor <= 1.0


@pytest.mark.parametrize(
    "det",
    [
        DetectorConfig.ideal_pad(),
        DetectorConfig.pt_ccd_waterman_evans(psf_radius=3),
        DetectorConfig.ed_pad_timepix(psf_mode="continuous"),
        DetectorConfig.ed_pad_timepix(psf_mode="discrete_cluster"),
    ],
)
def test_stationary_ensemble_moments_and_lagged_covariance_recovery(
    det: DetectorConfig,
) -> None:
    """Verify that simulated flat-field ensembles match theoretical mean, single-pixel variance,
    and 2D spatial autocovariance, and that summing over lags recovers G*gamma + psi.
    """
    rng = np.random.default_rng(12345)
    lambda_quanta = 25.0
    n_frames, height, width = 80, 96, 96
    lambda_field = np.full((n_frames, height, width), lambda_quanta, dtype=np.float64)

    signal = _sample_cascade_and_psf(lambda_field, det, rng)
    if det.pixel_factor > 0:
        signal += rng.normal(0.0, np.sqrt(det.pixel_factor), size=signal.shape)

    # Exclude border of width equal to PSF radius to avoid boundary reflection artifacts
    r = det.psf_kernel.shape[0] // 2
    interior = signal[:, r + 1 : height - r - 1, r + 1 : width - r - 1]

    expected_mean = lambda_quanta * det.gain
    sample_mean = float(np.mean(interior))
    assert np.isclose(sample_mean, expected_mean, rtol=0.015)

    # Single-pixel variance (lag 0)
    expected_var_lag0 = (
        expected_mean * det.apparent_single_pixel_gain + det.pixel_factor
    )
    sample_var_lag0 = float(np.var(interior))
    assert np.isclose(sample_var_lag0, expected_var_lag0, rtol=0.04)

    # Compute empirical 2D spatial autocovariance over lags [-2*r, 2*r]
    max_lag = 2 * r
    residuals = signal - expected_mean
    core = residuals[
        :,
        max_lag : height - max_lag,
        max_lag : width - max_lag,
    ]
    h_c, w_c = core.shape[1], core.shape[2]
    emp_cov = np.zeros((2 * max_lag + 1, 2 * max_lag + 1), dtype=np.float64)
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            shifted = residuals[
                :,
                max_lag + dy : max_lag + dy + h_c,
                max_lag + dx : max_lag + dx + w_c,
            ]
            emp_cov[dy + max_lag, dx + max_lag] = float(np.mean(core * shifted))

    theo_cov = det.theoretical_covariance_kernel(expected_mean)
    assert emp_cov.shape == theo_cov.shape
    # Check center and 1st-neighbor lags
    assert np.allclose(emp_cov, theo_cov, atol=0.05 * expected_var_lag0)

    # Crucial property: summing empirical covariance over all lags recovers G*gamma*mu + psi
    total_var_summed = float(np.sum(emp_cov))
    expected_total_var = expected_mean * det.g_gamma + det.pixel_factor
    assert np.isclose(total_var_summed, expected_total_var, rtol=0.04)


def test_dials_dispersion_threshold_underestimates_gain_under_psf() -> None:
    """Demonstrate that DIALS's local spatial index-of-dispersion underestimates G*gamma
    by the zero-lag PSF factor S_2 = sum K(u)^2 when point-spread is present.
    """
    from dials.algorithms.image.threshold import DispersionThresholdDebug
    from scitbx.array_family import flex

    rng = np.random.default_rng(99)
    lambda_field = np.full((1, 200, 200), 40.0, dtype=np.float64)

    # Case 1: Ideal PAD (G=1, gamma=1, psi=0, S_2=1)
    pad = DetectorConfig.ideal_pad()
    img_pad = _sample_cascade_and_psf(lambda_field, pad, rng)[0]
    flex_img_pad = flex.double(np.ascontiguousarray(img_pad))
    flex_mask = flex.bool(flex.grid(img_pad.shape), True)
    flex_gain = flex.double(flex.grid(img_pad.shape), 1.0)
    debug_pad = DispersionThresholdDebug(
        flex_img_pad, flex_mask, flex_gain, (10, 10), 6.0, 3.0, 0.0, 0
    )
    disp_pad = float(np.median(debug_pad.index_of_dispersion().as_numpy_array()))
    assert np.isclose(disp_pad, 1.0, rtol=0.05)

    # Case 2: p-t-CCD with PSF (no readout noise to isolate the PSF effect)
    ccd = DetectorConfig.pt_ccd_waterman_evans(pixel_factor=0.0, psf_fwhm_px=1.5, psf_radius=4)
    img_ccd = _sample_cascade_and_psf(lambda_field, ccd, rng)[0]
    flex_img_ccd = flex.double(np.ascontiguousarray(img_ccd))
    debug_ccd = DispersionThresholdDebug(
        flex_img_ccd, flex_mask, flex_gain, (10, 10), 6.0, 3.0, 0.0, 0
    )
    disp_ccd = float(np.median(debug_ccd.index_of_dispersion().as_numpy_array()))

    # True G*gamma is 1.44 * 1.54 = 2.2176, but DispersionThresholdDebug measures ~ G*gamma*S_2
    assert disp_ccd < 0.5 * ccd.g_gamma
    assert np.isclose(disp_ccd, ccd.apparent_single_pixel_gain, rtol=0.08)


def test_simulate_rotation_scan_full_scene() -> None:
    """Verify full rotation-scan simulation including beam drift, Bragg spots, and bad masks."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0)
    scene = SceneConfig(
        n_frames=30,
        height=64,
        width=64,
        n_bragg_spots=25,
        n_zingers=5,
        n_hot_pixels=4,
        n_dead_pixels=4,
        shadow_margin_px=3,
        include_sparse_corner=True,
    )
    sim = simulate_rotation_scan(det, scene, seed=7)

    assert sim.images.shape == (30, 64, 64)
    assert sim.expected_background_adu.shape == (30, 64, 64)
    assert sim.incident_flux_scale.shape == (30,)
    assert sim.spot_mask.shape == (30, 64, 64)
    assert sim.static_bad_mask.shape == (64, 64)

    # Check that some spots and static bad pixels were generated
    assert np.count_nonzero(sim.spot_mask) > 50
    assert np.count_nonzero(sim.static_bad_mask) > 4 * 64
    # Dead pixels must be exactly 0.0
    assert np.min(sim.images[:, sim.static_bad_mask]) == 0.0
    # Unmasked background pixels should have mean close to expected_background_adu + pedestal
    valid_3d = (~sim.spot_mask) & (~sim.static_bad_mask[None, :, :])
    obs_bg_mean = float(np.mean(sim.images[valid_3d]))
    exp_bg_mean = float(np.mean(sim.expected_background_adu[valid_3d]) + det.pedestal)
    assert np.isclose(obs_bg_mean, exp_bg_mean, rtol=0.02)
