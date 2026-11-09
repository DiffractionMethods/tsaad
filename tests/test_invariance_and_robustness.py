"""Invariance, robustness, and edge-case tests for the detector_stats hardening pass."""

from __future__ import annotations

import numpy as np
import pytest

from detector_stats import (
    DetectorConfig,
    MaskingConfig,
    SceneConfig,
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

    # Pass an intentionally under-subtracted pedestal (45.0 instead of 50.0 -> bin_means are +5 ADU too high,
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
        psf_mode="discrete_multinomial",
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
