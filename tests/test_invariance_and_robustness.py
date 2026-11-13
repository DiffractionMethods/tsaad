"""Invariance, robustness, and edge-case tests for the tsaad hardening pass."""

from __future__ import annotations

import numpy as np
import pytest

from tsaad import (
    CovarianceConfig,
    DetectorConfig,
    HollowStencilKrigingTrend,
    MaskingConfig,
    SceneConfig,
    SeparationConfig,
    TrendResult,
    detect_transient_outliers,
    estimate_detector_statistics,
    gaussian_psf,
    simulate_rotation_scan,
)


def test_exact_multiplicative_scale_invariance() -> None:
    """Scaling a continuous-PSF image stack by c > 0 should preserve all dimensionless
    masks, radii, and kernels while scaling (G*gamma) by c and psi by c^2.
    """
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=0.0, psf_radius=3)
    det.quantize = False
    scene = SceneConfig(n_frames=24, height=80, width=80, n_bragg_spots=25, n_zingers=5)
    sim = simulate_rotation_scan(det, scene, seed=404)
    y_base = sim.images.astype(np.float64)

    res_base = estimate_detector_statistics(
        y_base,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=4,
        auto_calibrate_pedestal=False,
    )

    for scale in (0.02, 50.0):
        y_scaled = scale * y_base
        res_scaled = estimate_detector_statistics(
            y_scaled,
            pedestal=0.0,
            trend_method="hollow_stencil_kriging",
            max_lag=4,
            auto_calibrate_pedestal=False,
        )

        # 1. Dimensionless masks must be identical
        np.testing.assert_array_equal(
            res_scaled.masking.static_bad_mask,
            res_base.masking.static_bad_mask,
        )
        np.testing.assert_array_equal(
            res_scaled.masking.transient_outlier_mask,
            res_base.masking.transient_outlier_mask,
        )

        # 2. Dimensionless PSF radius, S2 factor, and normalized kernel alpha(u) must match
        assert res_scaled.psf_radius == res_base.psf_radius
        assert res_scaled.s2_factor == pytest.approx(res_base.s2_factor, rel=1e-5, abs=1e-5)
        np.testing.assert_allclose(
            res_scaled.covariance.alpha_kernel,
            res_base.covariance.alpha_kernel,
            rtol=1e-5,
            atol=1e-5,
        )

        # 3. Dimensionful quantities must scale equivariantly: G*gamma ~ c, psi ~ c^2
        assert res_scaled.g_gamma == pytest.approx(scale * res_base.g_gamma, rel=1e-4)
        assert res_scaled.covariance.g_gamma_se == pytest.approx(
            scale * res_base.covariance.g_gamma_se, rel=1e-4
        )
        assert res_scaled.psi == pytest.approx((scale**2) * res_base.psi, rel=1e-4, abs=1e-6)
        assert res_scaled.covariance.psi_se == pytest.approx(
            (scale**2) * res_base.covariance.psi_se, rel=5e-4, abs=1e-6
        )
        assert res_base.quality is not None and res_scaled.quality is not None
        assert res_scaled.quality.standardized_residual_std == pytest.approx(
            res_base.quality.standardized_residual_std, rel=1e-4
        )
        assert res_scaled.quality.is_reliable == res_base.quality.is_reliable


def test_psf_guard_band_auto_expansion() -> None:
    """When fixed_psf_radius is None and r_PSF reaches max_lag, estimate_lagged_covariance
    should automatically expand max_lag so the baseline ring lies outside the PSF support.
    """
    det = DetectorConfig(
        name="wide_psf_detector",
        gain=3.0,
        cascade_factor=1.2,
        pixel_factor=1.5,
        pedestal=0.0,
        psf_kernel=gaussian_psf(sigma_px=1.35, radius=3),
        psf_mode="continuous",
        quantize=False,
    )
    scene = SceneConfig(n_frames=30, height=88, width=88, n_bragg_spots=20)
    sim = simulate_rotation_scan(det, scene, seed=505)

    # Request a deliberately small initial max_lag=2 so r_PSF hits the boundary
    res = estimate_detector_statistics(
        sim.images,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=2,
    )
    # Auto-expansion should have increased the lag window beyond 2 (kernel shape > 5x5)
    assert res.covariance.alpha_kernel.shape[0] > 5
    assert res.psf_radius >= 3
    assert res.covariance.psf_converged is True
    assert abs(res.g_gamma - det.g_gamma) / det.g_gamma < 0.12


def test_pedestal_offset_diagnostic_and_opt_out() -> None:
    """Verify estimated_pedestal_offset is reported and auto_calibrate_pedestal=False
    preserves unshifted bin means.
    """
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(n_frames=36, height=96, width=96, n_bragg_spots=30)
    sim = simulate_rotation_scan(det, scene, seed=606)

    # Pass an intentionally under-subtracted pedestal (45.0 vs 50.0 -> bin_means +5 ADU too high,
    # so the required additive correction delta_p0 to true intensity is -5.0 ADU)
    res_auto = estimate_detector_statistics(
        sim.images,
        pedestal=45.0,
        trend_method="hollow_stencil_kriging",
        max_lag=4,
        auto_calibrate_pedestal=True,
    )
    res_no_auto = estimate_detector_statistics(
        sim.images,
        pedestal=45.0,
        trend_method="hollow_stencil_kriging",
        max_lag=4,
        auto_calibrate_pedestal=False,
    )

    # Both should estimate the required correction delta_p0 (~ -5.0 ADU)
    assert res_auto.covariance.estimated_pedestal_offset == pytest.approx(-5.0, abs=2.5)
    assert res_no_auto.covariance.estimated_pedestal_offset == pytest.approx(-5.0, abs=2.5)

    # With auto_calibrate_pedestal=True, bin_means are shifted by +estimated_pedestal_offset
    np.testing.assert_allclose(
        res_auto.covariance.bin_means - res_no_auto.covariance.bin_means,
        res_auto.covariance.estimated_pedestal_offset,
        rtol=1e-6,
    )


def test_non_integer_sparse_cluster_wing_integration() -> None:
    """Verify that continuous (non-integer) sparse-cluster multiplicity integrates
    1-px dilated cluster support so sub-threshold PSF wings are not clipped.
    """
    det = DetectorConfig(
        name="continuous_sparse_detector",
        gain=5.0,
        cascade_factor=1.35,
        pixel_factor=0.02,
        pedestal=0.0,
        psf_kernel=gaussian_psf(sigma_px=0.65, radius=2),
        psf_mode="discrete_cluster",
        quantize=False,
    )
    scene = SceneConfig(
        n_frames=40,
        height=96,
        width=96,
        include_sparse_corner=True,
        sparse_corner_quanta=0.008,
        n_bragg_spots=20,
    )
    sim = simulate_rotation_scan(det, scene, seed=707)
    # Add small continuous Gaussian readout noise so values are non-integer floats
    rng = np.random.default_rng(708)
    y_cont = sim.images.astype(np.float64) + rng.normal(0.0, np.sqrt(0.02), size=sim.images.shape)

    res = estimate_detector_statistics(
        y_cont,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=3,
    )
    assert res.separation.separated is True
    assert res.separation.separation_method == "sparse_cluster_multiplicity"
    assert abs(res.gain - det.gain) / det.gain < 0.08


def test_hysteresis_wing_propagation_masks_extended_bragg_skirts() -> None:
    """Verify that detect_transient_outliers propagates seed detections into contiguous
    moderate-sigma Bragg reflection skirts while leaving isolated background noise unmasked.
    """
    rng = np.random.default_rng(808)
    n_frames, ny, nx = 24, 64, 64
    # Smooth Poisson background around 20 quanta/px (G=1)
    y = rng.poisson(20.0, size=(n_frames, ny, nx)).astype(np.float64)

    # Inject a bright Bragg spot on frame 12 with a wide 2.5-sigma outer skirt at radius 3..4 px
    yy, xx = np.meshgrid(np.arange(ny) - 32.0, np.arange(nx) - 32.0, indexing="ij")
    r2 = yy**2 + xx**2
    spot_profile = 150.0 * np.exp(-0.5 * r2 / (1.0**2)) + 15.0 * np.exp(-0.5 * r2 / (2.5**2))
    y[12] += spot_profile

    mask_res = detect_transient_outliers(
        y,
        pedestal=0.0,
        config=MaskingConfig(
            transient_seed_sigma=5.0,
            transient_wing_sigma=2.0,
            dilation_xy=2,
            dilation_t=0,
        ),
    )

    # Pixels at radius 3..4 px from (32, 32) on frame 12 have ~2.5-3.0 sigma skirts
    # and should be captured by hysteresis propagation into transient_wing_sigma
    skirt_ring = (r2 >= 9.0) & (r2 <= 16.0)
    skirt_masked_fraction = float(np.mean(mask_res.transient_outlier_mask[12][skirt_ring]))
    assert skirt_masked_fraction > 0.85

    # Unaffected frames should have near-zero false-positive masking
    other_frames_masked = float(np.mean(mask_res.transient_outlier_mask[:8]))
    assert other_frames_masked < 0.005


def test_flat_roi_collinearity_enforces_psd_lag_zero_bound() -> None:
    """On a nearly flat-background CCD ROI where narrow dynamic range causes unconstrained
    2-parameter lag-0 regression to trade slope g(0) into intercept psi, the Cauchy-Schwarz /
    PSF-monotonicity bound must preserve a positive peak at alpha(0, 0), recover G*gamma and psi,
    and remain strictly scale-equivariant.
    """
    from scipy.ndimage import convolve

    rng = np.random.default_rng(909)
    n_t, ny, nx = 32, 72, 72
    psf = gaussian_psf(sigma_px=1.05, radius=3)
    true_gain = 2.4
    true_psi = 36.0

    # Very narrow spatial background gradient (~8% span around 190 ADU, mimicking idp52003_7sgw / 4y0e)
    yy, xx = np.mgrid[0:ny, 0:nx]
    mu_quanta_2d = (190.0 + 14.0 * (yy / float(ny) + xx / float(nx)) * 0.5) / true_gain
    quanta = rng.poisson(mu_quanta_2d[None, :, :], size=(n_t, ny, nx)).astype(np.float64)
    signal = np.empty_like(quanta)
    for t in range(n_t):
        signal[t] = true_gain * convolve(quanta[t], psf, mode="nearest")
    read_noise = rng.normal(0.0, np.sqrt(true_psi), size=(n_t, ny, nx))
    y = signal + read_noise

    res = estimate_detector_statistics(
        y,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=5,
        auto_calibrate_pedestal=False,
    )
    L = res.covariance.alpha_kernel.shape[0] // 2
    assert res.covariance.apparent_single_pixel_gain > 0.0
    assert res.covariance.alpha_kernel[L, L] >= float(np.max(res.covariance.alpha_kernel)) - 1e-12
    assert res.s2_factor > 0.05
    assert abs(res.g_gamma - true_gain) / true_gain < 0.18
    assert abs(res.psi - true_psi) < 15.0

    # Verify strict multiplicative scale equivariance even when the bound is active
    scale = 25.0
    res_scaled = estimate_detector_statistics(
        scale * y,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=5,
        auto_calibrate_pedestal=False,
    )
    assert res_scaled.psf_radius == res.psf_radius
    assert res_scaled.g_gamma == pytest.approx(scale * res.g_gamma, rel=1e-4)
    assert res_scaled.psi == pytest.approx((scale**2) * res.psi, rel=1e-4, abs=1e-6)


def test_high_mu_bragg_halo_bins_rejected_by_tukey_biweight() -> None:
    """When the highest-intensity bins sit inside unmasked spatially extended Bragg/powder
    halos (mimicking 3rsi / 53bp1_6vip), Tukey-biweight IRLS across lags and shells must
    reject the contaminated top bins and prevent false r_PSF and G*gamma inflation.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(1010)
    n_t, ny, nx = 36, 80, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Smooth radial background spanning 8..40 counts/px on an ideal G=1, psi=0 PAD
    r_norm = np.hypot(yy, xx) / np.hypot(ny, nx)
    mu_2d = 8.0 + 32.0 * (1.0 - r_norm)
    y = rng.poisson(mu_2d[None, :, :], size=(n_t, ny, nx)).astype(np.float64)

    # Inject strong spatially correlated (sigma=2.2 px) frame-varying Bragg/powder halo noise
    # strictly in the brightest 14% corner (top ~2 intensity bins)
    bright_corner = mu_2d >= np.percentile(mu_2d, 86.0)
    for t in range(n_t):
        halo = gaussian_filter(rng.normal(0.0, 18.0, size=(ny, nx)), sigma=2.2, mode="nearest")
        y[t] = np.maximum(0.0, y[t] + np.where(bright_corner, halo, 0.0))

    res = estimate_detector_statistics(
        y,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=5,
        auto_calibrate_pedestal=False,
    )
    assert res.psf_radius == 0
    assert res.s2_factor == pytest.approx(1.0, abs=1e-6)
    assert res.g_gamma == pytest.approx(1.0, abs=0.08)


def test_janesick_quadratic_halo_excess_variance_guard() -> None:
    """When non-separable diffuse background curvature injects spatially correlated
    multiplicative fluctuations `eta(x, t) * mu(x, t)` across ALL intensity bins (producing
    Janesick quadratic excess variance `C(u; mu) = s * mu + beta * mu^2`, mimicking
    `dmoa_6ak1` / `53bp1_6vip`), `_irls_offdiag_fit` must separate out `beta * mu^2`
    and recover the true linear shot-noise gain `G*gamma = 1.0` and `r_PSF = 0`.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2020)
    n_t, ny, nx = 40, 88, 88
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Smooth radial gradient spanning 15..90 ADU (std/mean ~ 0.42)
    r_norm = np.hypot(yy, xx) / np.hypot(ny, nx)
    mu_2d = 15.0 + 75.0 * ((1.0 - r_norm) ** 1.5)

    # Multiplicative frame-to-frame medium-scale (sigma=2.5 px) background modulation eta(x, t)
    eta = np.empty((n_t, ny, nx), dtype=np.float64)
    for t in range(n_t):
        raw_mod = gaussian_filter(rng.normal(0.0, 0.35, size=(ny, nx)), sigma=2.5, mode="nearest")
        eta[t] = np.clip(raw_mod, -0.25, 0.25)

    lam = np.maximum(1.0, mu_2d[None, :, :] * (1.0 + eta))
    y = rng.poisson(lam).astype(np.float64)

    res = estimate_detector_statistics(
        y,
        pedestal=0.0,
        trend_method="hollow_stencil_kriging",
        max_lag=5,
        auto_calibrate_pedestal=False,
    )
    assert res.psf_radius == 0
    assert res.s2_factor == pytest.approx(1.0, abs=1e-6)
    assert res.g_gamma == pytest.approx(1.0, abs=0.08)


def test_nan_inf_and_invalid_shape_handling() -> None:
    """Verify input validation errors and automatic masking of isolated/static NaN/Inf pixels."""
    with pytest.raises(ValueError, match="3D image stack"):
        estimate_detector_statistics(np.ones((64, 64)))
    with pytest.raises(ValueError, match="at least 2 frames"):
        estimate_detector_statistics(np.ones((1, 32, 32)))
    with pytest.raises(ValueError, match="at least \\(8, 8\\)"):
        estimate_detector_statistics(np.ones((10, 6, 32)))
    with pytest.raises(ValueError, match="pedestal must be finite"):
        estimate_detector_statistics(np.ones((10, 32, 32)), pedestal=float("nan"))

    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=40.0, psf_radius=3)
    scene = SceneConfig(n_frames=32, height=88, width=88, n_bragg_spots=20)
    sim = simulate_rotation_scan(det, scene, seed=303)

    corrupted = sim.images.copy()
    # Static all-frame NaN pixel + isolated single-frame NaN and +Inf voxels
    corrupted[:, 20, 20] = np.nan
    corrupted[5, 40, 45] = np.nan
    corrupted[12, 55, 35] = np.inf

    rep = estimate_detector_statistics(
        corrupted, pedestal=det.pedestal, max_lag=4, trusted_range=(0.0, 65535.0)
    )
    assert bool(rep.masking.static_bad_mask[20, 20]) is True
    assert bool(rep.masking.transient_outlier_mask[5, 40, 45]) is True
    assert bool(rep.masking.transient_outlier_mask[12, 55, 35]) is True
    assert np.all(np.isfinite(rep.trend.residuals))
    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.08)
    assert np.isclose(rep.psi, det.pixel_factor, atol=0.80)


def test_quality_diagnostics_flags_contaminated_scan() -> None:
    """Clean scans report is_reliable=True, whereas severe unmasked overdispersion is flagged."""
    det = DetectorConfig.ideal_pad()
    scene = SceneConfig(n_frames=30, height=80, width=80, n_bragg_spots=20)
    sim = simulate_rotation_scan(det, scene, seed=414)
    clean_rep = estimate_detector_statistics(sim.images, pedestal=0.0, max_lag=4)
    assert clean_rep.quality is not None
    assert clean_rep.quality.is_reliable is True
    assert abs(clean_rep.quality.standardized_residual_std - 1.0) < 0.10

    # Inject spatially localized heavy Bragg/diffuse clutter on 35% of pixels (mimicking 3rsi)
    rng = np.random.default_rng(415)
    noisy = sim.images.copy()
    noisy[:, :30, :40] += rng.choice([0.0, 45.0], p=[0.65, 0.35], size=(30, 30, 40))
    bad_rep = estimate_detector_statistics(
        noisy,
        pedestal=0.0,
        max_lag=4,
        masking_config=MaskingConfig(transient_seed_sigma=50.0, transient_cluster_sigma=50.0),
    )
    assert bad_rep.quality is not None
    assert bad_rep.quality.is_reliable is False
    assert any("non_gaussian_residuals" in w for w in bad_rep.quality.warnings)


def test_boundary_bragg_spot_and_scan_step_no_wrap_artifact() -> None:
    """Endpoint-median temporal padding must mask a Bragg spot at t=0 without wrapping t=T-1."""
    rng = np.random.default_rng(515)
    n_t, ny, nx = 30, 64, 64
    t_norm = np.linspace(0.0, 1.0, n_t)[:, None, None]
    # Anisotropic rotation: top half goes 45 -> 10 ADU while bottom half goes 10 -> 45 ADU
    # (global flux stays constant, so mode='wrap' would pull t=0 top-half median down and
    # falsely mask >50% of the top half at t=0!)
    yy, xx = np.mgrid[0:ny, 0:nx]
    top_half = yy < (ny // 2)
    lam = np.where(
        top_half[None, :, :],
        45.0 - 35.0 * t_norm,
        10.0 + 35.0 * t_norm,
    )
    y = rng.poisson(lam).astype(np.float64)

    # Place a bright Bragg spot directly on frame t=0 at (48, 32)
    spot_r2 = (yy - 48.0) ** 2 + (xx - 32.0) ** 2
    y[0] += 220.0 * np.exp(-0.5 * spot_r2 / (1.2**2))

    mask_res = detect_transient_outliers(y, pedestal=0.0)
    # Spot core at t=0 must be masked
    assert bool(mask_res.transient_outlier_mask[0, 48, 32]) is True
    # Top half at t=0 must NOT suffer false masking from wrapping t=T-1 (10 ADU) onto t=0 (45 ADU)
    assert float(np.mean(mask_res.transient_outlier_mask[0, top_half])) < 0.05


def test_custom_trend_estimator_and_config_dataclasses() -> None:
    """Verify passing custom TrendEstimator instances and Covariance/Separation configs."""
    det = DetectorConfig.ideal_pad()
    scene = SceneConfig(n_frames=28, height=72, width=72, n_bragg_spots=15)
    sim = simulate_rotation_scan(det, scene, seed=616)

    custom_kriging = HollowStencilKrigingTrend(temporal_sigma_frames=2.0)
    rep1 = estimate_detector_statistics(
        sim.images,
        pedestal=0.0,
        trend_method=custom_kriging,
        covariance_config=CovarianceConfig(temporal_chunk_frames=8),
        separation_config=SeparationConfig(pad_g_gamma_tol=0.15),
    )
    assert abs(rep1.g_gamma - 1.0) < 0.04

    class _CustomWrapperTrend:
        def fit_predict(
            self, images: np.ndarray, valid_mask: np.ndarray, pedestal: float = 0.0
        ) -> TrendResult:
            res = HollowStencilKrigingTrend(temporal_sigma_frames=1.8).fit_predict(
                images, valid_mask=valid_mask, pedestal=pedestal
            )
            res.method_name = "custom_wrapper"
            return res

    rep2 = estimate_detector_statistics(sim.images, pedestal=0.0, trend_method=_CustomWrapperTrend())
    assert rep2.trend.method_name == "custom_wrapper"
    assert abs(rep2.g_gamma - 1.0) < 0.04


def test_sparse_pad_with_blanked_initial_frames() -> None:
    """Sparse 3DED PAD scan (<20% nonzero) with shutter-blanked/grid-bar-eclipsed early frames."""
    det = DetectorConfig.ed_pad_timepix(gain=2.6, cascade_factor=1.15)
    scene = SceneConfig(
        n_frames=60,
        height=96,
        width=96,
        bg_min_quanta=0.04,
        bg_max_quanta=0.22,
        n_bragg_spots=15,
        n_zingers=2,
        n_hot_pixels=0,
        n_dead_pixels=0,
        shadow_margin_px=0,
    )
    sim = simulate_rotation_scan(det, scene, seed=909)
    imgs = sim.images.copy()
    # Eclipse the first 28 frames (simulating TEM grid bar occlusion at high starting tilt)
    rng = np.random.default_rng(910)
    imgs[:28] = rng.binomial(1, 0.002, size=imgs[:28].shape).astype(np.float64)

    rep = estimate_detector_statistics(imgs, pedestal=0.0)
    assert np.isclose(rep.g_gamma, det.g_gamma, rtol=0.18)
    assert rep.psi < 0.25


def test_flat_integer_pad_narrow_range_zero_intercept_guard() -> None:
    """On a flat-plateau integer photon-counting PAD tile (CV < 0.12) where crowded spot-wing
    censoring slightly depresses variance in the highest intensity bins, enforcing the physical
    zero-intercept PAD constraint avoids lever-arm pivoting of (g0, psi).
    """
    rng = np.random.default_rng(717)
    n_t, ny, nx = 48, 80, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Narrow flat plateau spanning ~12.4..14.2 ADU (CV ~ 0.04)
    mu_2d = 12.4 + 1.8 * (yy / float(ny))
    y = rng.poisson(mu_2d[None, :, :], size=(n_t, ny, nx)).astype(np.float64)
    # Mild upper-tail censoring on the brightest 20% band (mimicking crowded spot-wing hysteresis)
    bright_band = mu_2d > 13.8
    high_tail = bright_band[None, :, :] & (y > 21.0)
    y[high_tail] = 21.0

    rep = estimate_detector_statistics(y, pedestal=0.0, max_lag=4)
    assert rep.psi == pytest.approx(0.0, abs=1e-9)
    assert rep.g_gamma == pytest.approx(1.0, abs=0.07)


def test_diffuse_mosaicity_streak_pedestal_does_not_inflate_r_psf() -> None:
    """Faint long-range diffuse streaks on a photon-counting PAD must not cause r_PSF to
    march outward across growing ring circumferences (mimicking 5vog / pfplrxdata_9jq9).
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(818)
    n_t, ny, nx = 48, 96, 96
    yy, xx = np.mgrid[0:ny, 0:nx]
    mu_2d = 35.0 + 95.0 * (1.0 - np.hypot(yy, xx) / np.hypot(ny, nx))
    lam = np.empty((n_t, ny, nx), dtype=np.float64)
    for t in range(n_t):
        # Broad (sigma=4.8 px) faint frame-varying diffuse background (~0.8% per-pixel pedestal)
        diffuse = gaussian_filter(rng.normal(0.0, 4.2, size=(ny, nx)), sigma=4.8, mode="nearest")
        lam[t] = np.maximum(2.0, mu_2d + np.sqrt(mu_2d) * 0.28 * diffuse)
    y = rng.poisson(lam).astype(np.float64)

    rep = estimate_detector_statistics(y, pedestal=0.0, max_lag=7)
    assert rep.psf_radius <= 1
    assert rep.s2_factor > 0.82
    assert rep.g_gamma == pytest.approx(1.0, abs=0.14)
