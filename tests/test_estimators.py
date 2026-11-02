"""Unit and benchmark tests for Component 2 (masking, trend, covariance, and G/gamma separation)."""

from __future__ import annotations

import numpy as np
import pytest

from detector_stats import (
    DetectorConfig,
    SceneConfig,
    delta_psf,
    detect_static_bad_pixels,
    detect_transient_outliers,
    estimate_detector_statistics,
    estimate_incident_flux_scale,
    gaussian_psf,
    lorentzian_psf,
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


def test_multi_seed_monte_carlo_ensemble() -> None:
    """Verify unbiasedness and absence of outlier failures across a multi-seed Monte Carlo ensemble."""
    scene_ccd = SceneConfig(n_frames=45, height=104, width=104, n_bragg_spots=55, n_zingers=12)
    det_ccd = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)

    gg_ccd: list[float] = []
    psi_ccd: list[float] = []
    for seed in range(8):
        sim = simulate_rotation_scan(det_ccd, scene_ccd, seed=seed)
        rep = estimate_detector_statistics(sim.images, pedestal=det_ccd.pedestal)
        gg_ccd.append(rep.g_gamma)
        psi_ccd.append(rep.psi)
        # Every individual seed must stay within 10% of true G*gamma and 0.6 ADU^2 of true psi
        assert np.isclose(rep.g_gamma, det_ccd.g_gamma, rtol=0.10)
        assert np.isclose(rep.psi, det_ccd.pixel_factor, atol=0.60)

    # Ensemble mean across seeds must be within 3.5% of true G*gamma and 0.25 ADU^2 of true psi
    assert np.isclose(float(np.mean(gg_ccd)), det_ccd.g_gamma, rtol=0.035)
    assert np.isclose(float(np.mean(psi_ccd)), det_ccd.pixel_factor, atol=0.25)

    # Also test multi-seed ensemble on 3D ED Timepix with sparse single-event cluster separation
    det_ed = DetectorConfig.ed_pad_timepix()
    scene_ed = SceneConfig(
        n_frames=45,
        height=112,
        width=112,
        include_sparse_corner=True,
        sparse_corner_quanta=0.008,
        n_bragg_spots=50,
    )
    g_ed: list[float] = []
    gamma_ed: list[float] = []
    for seed in range(5):
        sim = simulate_rotation_scan(det_ed, scene_ed, seed=100 + seed)
        rep = estimate_detector_statistics(sim.images, pedestal=det_ed.pedestal)
        assert rep.separation.separated is True
        g_ed.append(rep.gain)
        gamma_ed.append(rep.cascade_factor)

    assert abs(float(np.mean(g_ed)) - det_ed.gain) < 0.06
    assert abs(float(np.mean(gamma_ed)) - det_ed.cascade_factor) < 0.05


@pytest.mark.parametrize(
    "det_cfg,has_sparse",
    [
        # 1. Sub-unity gain detector with Gaussian PSF
        (
            DetectorConfig(
                name="low_gain_cmos",
                gain=0.45,
                cascade_factor=1.60,
                pixel_factor=2.50,
                pedestal=100.0,
                psf_kernel=gaussian_psf(sigma_px=0.8, radius=3),
                psf_mode="continuous",
            ),
            False,
        ),
        # 2. High-gain, heavy-cascade phosphor detector with Lorentzian PSF
        (
            DetectorConfig(
                name="high_gain_phosphor",
                gain=4.20,
                cascade_factor=2.10,
                pixel_factor=18.0,
                pedestal=25.0,
                psf_kernel=lorentzian_psf(fwhm_px=1.4, radius=3),
                psf_mode="continuous",
            ),
            False,
        ),
        # 3. Zero-PSF integrating detector with readout noise
        (
            DetectorConfig(
                name="delta_psf_integrating",
                gain=1.80,
                cascade_factor=1.30,
                pixel_factor=6.00,
                pedestal=10.0,
                psf_kernel=delta_psf(0),
                psf_mode="continuous",
            ),
            False,
        ),
        # 4. High-multiplicity discrete cluster detector with sparse corner
        (
            DetectorConfig.ed_pad_timepix(gain=3.80, cascade_factor=1.45),
            True,
        ),
    ],
)
def test_parameter_sweep_across_detector_configurations(
    det_cfg: DetectorConfig, has_sparse: bool
) -> None:
    """Verify recovery across diverse (G, gamma, psi, PSF) parameter combinations."""
    scene = SceneConfig(
        n_frames=50,
        height=112,
        width=112,
        include_sparse_corner=has_sparse,
        sparse_corner_quanta=0.008,
        n_bragg_spots=50,
        n_zingers=10,
    )
    sim = simulate_rotation_scan(det_cfg, scene, seed=29)
    rep = estimate_detector_statistics(sim.images, pedestal=det_cfg.pedestal)

    assert np.isclose(rep.g_gamma, det_cfg.g_gamma, rtol=0.06)
    assert np.isclose(rep.psi, det_cfg.pixel_factor, atol=max(0.35, 0.06 * det_cfg.pixel_factor))
    if has_sparse:
        assert rep.separation.separated is True
        assert np.isclose(rep.gain, det_cfg.gain, rtol=0.05)
        assert np.isclose(rep.cascade_factor, det_cfg.cascade_factor, rtol=0.06)


def test_crowded_bragg_spots_and_zingers() -> None:
    """Verify parameter recovery under high Bragg spot and cosmic-ray zinger crowding."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(
        n_frames=50,
        height=112,
        width=112,
        n_bragg_spots=350,
        n_zingers=40,
        n_hot_pixels=12,
        n_dead_pixels=12,
        shadow_margin_px=4,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(sim.images, pedestal=det.pedestal)

    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.06)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.50)
    assert rep.psf_radius >= 3


def test_narrow_background_dynamic_range() -> None:
    """Verify parameter recovery when the spatial background dynamic range is narrow."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(
        n_frames=50,
        height=112,
        width=112,
        bg_min_quanta=20.0,
        bg_max_quanta=28.0,
        n_bragg_spots=50,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(sim.images, pedestal=det.pedestal)

    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.06)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.60)


def test_adc_integer_quantization() -> None:
    """Verify that integer ADC quantization does not bias (G*gamma, psi) recovery."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    det.quantize = True
    scene = SceneConfig(n_frames=50, height=112, width=112, n_bragg_spots=60)
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(sim.images, pedestal=det.pedestal)

    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.05)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.45)


@pytest.mark.parametrize("pedestal_error_adu", [-5.0, +5.0])
def test_pedestal_offset_self_calibration(pedestal_error_adu: float) -> None:
    """Verify that off-diagonal PSF covariance self-calibrates additive pedestal errors."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(n_frames=50, height=112, width=112, n_bragg_spots=60)
    sim = simulate_rotation_scan(det, scene, seed=42)

    rep = estimate_detector_statistics(
        sim.images, pedestal=det.pedestal + pedestal_error_adu
    )
    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.05)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.70)

