"""Unit and benchmark tests for Component 2 (masking, trend, covariance, and G/gamma separation)."""

from __future__ import annotations

import numpy as np
import pytest

from detector_stats import (
    DetectorConfig,
    SceneConfig,
    detect_static_bad_pixels,
    detect_transient_outliers,
    estimate_detector_statistics,
    estimate_incident_flux_scale,
    simulate_rotation_scan,
)


def test_masking_static_and_transient_anomalies() -> None:
    """Verify static bad-pixel/shadow detection and transient Bragg/zinger masking."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(
        n_frames=40,
        height=96,
        width=96,
        n_bragg_spots=40,
        n_zingers=10,
        n_hot_pixels=6,
        n_dead_pixels=6,
        shadow_margin_px=4,
    )
    sim = simulate_rotation_scan(det, scene, seed=101)

    static_mask = detect_static_bad_pixels(sim.images, pedestal=det.pedestal)
    # Recall on true static bad mask (shadowed border + hot + dead pixels) should be > 99%
    true_bad = sim.static_bad_mask
    recall_static = float(np.count_nonzero(static_mask & true_bad)) / float(
        np.count_nonzero(true_bad)
    )
    assert recall_static > 0.99
    # False positive rate on valid interior pixels should be tiny (< 0.5%)
    false_pos_static = float(np.count_nonzero(static_mask & (~true_bad))) / float(
        np.count_nonzero(~true_bad)
    )
    assert false_pos_static < 0.005

    mask_res = detect_transient_outliers(
        sim.images, pedestal=det.pedestal, static_bad_mask=static_mask
    )
    # Check that most true spot/zinger core voxels are masked while >80% of background remains valid
    interior_spots = sim.spot_mask & (~true_bad[None, :, :])
    recall_spots = float(
        np.count_nonzero(mask_res.transient_outlier_mask & interior_spots)
    ) / float(max(1, np.count_nonzero(interior_spots)))
    assert recall_spots > 0.80
    valid_bg_frac = float(
        np.count_nonzero(mask_res.valid_mask[:, ~true_bad])
    ) / float(sim.images[:, ~true_bad].size)
    assert valid_bg_frac > 0.75


def test_incident_flux_scale_recovery() -> None:
    """Verify missing-data-unbiased recovery of non-stationary incident beam flux I_0(t)."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=20.0, psf_radius=3)
    scene = SceneConfig(
        n_frames=50,
        height=96,
        width=96,
        beam_drift_amplitude=0.08,
        beam_topup_period=20,
        beam_topup_jump=0.06,
        beam_jitter_sigma=0.02,
        n_bragg_spots=50,
    )
    sim = simulate_rotation_scan(det, scene, seed=17)
    mask_res = detect_transient_outliers(sim.images, pedestal=det.pedestal)
    y_sub = sim.images - det.pedestal
    flux_est, _ = estimate_incident_flux_scale(y_sub, mask_res.valid_mask)

    rel_rms_err = float(np.sqrt(np.mean(((flux_est - sim.incident_flux_scale) / sim.incident_flux_scale) ** 2)))
    assert rel_rms_err < 0.01


@pytest.mark.parametrize(
    "trend_method",
    ["hollow_stencil_kriging", "low_rank_spline", "temporal_difference"],
)
def test_ideal_pad_parameter_recovery(trend_method: str) -> None:
    """Test 1 (Ideal PAD): Verify |G*gamma - 1.0| < 0.02, |psi| < 0.05, r_PSF == 0."""
    det = DetectorConfig.ideal_pad()
    scene = SceneConfig(
        n_frames=60,
        height=128,
        width=128,
        n_bragg_spots=80,
        n_zingers=15,
        n_hot_pixels=10,
        n_dead_pixels=10,
        shadow_margin_px=4,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(
        sim.images, pedestal=det.pedestal, trend_method=trend_method  # type: ignore[arg-type]
    )

    assert abs(rep.g_gamma - 1.0) < 0.02
    assert abs(rep.psi - 0.0) < 0.05
    assert rep.psf_radius == 0
    assert np.isclose(rep.s2_factor, 1.0, atol=0.02)
    assert rep.separation.separated is True
    assert rep.separation.separation_method == "photon_counting_pad"
    assert np.isclose(rep.gain, 1.0, atol=0.01)
    assert abs(rep.cascade_factor - 1.0) < 0.02


@pytest.mark.parametrize(
    "trend_method",
    ["hollow_stencil_kriging", "low_rank_spline", "temporal_difference"],
)
def test_pt_ccd_waterman_evans_parameter_recovery(trend_method: str) -> None:
    """Test 2 (p-t-CCD with PSF & Beam Drift): Verify recovery of G*gamma = 2.22 and psi = 9.08."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(
        n_frames=60,
        height=128,
        width=128,
        n_bragg_spots=80,
        n_zingers=15,
        n_hot_pixels=10,
        n_dead_pixels=10,
        shadow_margin_px=4,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(
        sim.images, pedestal=det.pedestal, trend_method=trend_method  # type: ignore[arg-type]
    )

    # Zero-lag single-pixel slope is damped by S_2 ~= 0.19 (reporting ~0.42 instead of 2.22)
    assert rep.covariance.apparent_single_pixel_gain < 0.55
    assert np.isclose(
        rep.covariance.apparent_single_pixel_gain,
        det.apparent_single_pixel_gain,
        rtol=0.08,
    )

    # Lag-summed spatial cross-covariance recovers the full G*gamma = 2.218 and psi = 9.082
    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.05)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.25)
    assert rep.psf_radius >= 3

    # Without a sparse single-photon region, fallback rule sets G = 1.0 and gamma = G*gamma
    assert rep.separation.separated is False
    assert rep.separation.separation_method == "fallback_g_one"
    assert rep.gain == 1.0
    assert np.isclose(rep.cascade_factor, rep.g_gamma, rtol=1e-12)


@pytest.mark.parametrize(
    "trend_method",
    ["hollow_stencil_kriging", "low_rank_spline", "temporal_difference"],
)
def test_ed_pad_timepix_sparse_cluster_separation(trend_method: str) -> None:
    """Test 3 (3D ED Timepix PAD): Verify separate recovery of G = 2.9 and gamma = 1.35."""
    det = DetectorConfig.ed_pad_timepix()
    scene = SceneConfig(
        n_frames=60,
        height=128,
        width=128,
        include_sparse_corner=True,
        sparse_corner_quanta=0.008,
        n_bragg_spots=80,
        n_zingers=15,
        n_hot_pixels=10,
        n_dead_pixels=10,
        shadow_margin_px=4,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(
        sim.images, pedestal=det.pedestal, trend_method=trend_method  # type: ignore[arg-type]
    )

    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.03)
    assert abs(rep.psi - 0.0) < 0.05
    assert rep.psf_radius == 2
    assert rep.separation.separated is True
    assert rep.separation.separation_method == "sparse_cluster_multiplicity"
    assert rep.separation.n_isolated_clusters >= 150
    assert abs(rep.gain - det.gain) < 0.05
    assert abs(rep.cascade_factor - det.cascade_factor) < 0.05
    assert np.isclose(rep.gain * rep.cascade_factor, rep.g_gamma, rtol=1e-12)
