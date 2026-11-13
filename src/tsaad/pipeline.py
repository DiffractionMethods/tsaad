"""End-to-end estimation pipeline for detector response statistics (G, gamma, psi)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tsaad._utils import is_effectively_integer_adu, validate_and_sanitize_stack
from tsaad.covariance import (
    CovarianceConfig,
    CovarianceResult,
    estimate_lagged_covariance,
)
from tsaad.masking import (
    MaskingConfig,
    MaskingResult,
    detect_static_bad_pixels,
    detect_transient_outliers,
)
from tsaad.separation import (
    SeparationConfig,
    SeparationResult,
    separate_gain_and_cascade,
)
from tsaad.trend import (
    HollowStencilKrigingTrend,
    LowRankSplineTrend,
    TemporalDifferenceEstimator,
    TrendEstimator,
    TrendMethodName,
    TrendResult,
)


@dataclass(frozen=True)
class QualityDiagnostics:
    """Self-diagnostic data-quality and model-fit metrics for a DetectorStatsReport."""

    standardized_residual_std: float  # Target ~1.0 under clean Gaussian/Poisson noise model
    valid_voxel_fraction: float  # Fraction of (T, H, W) voxels retained for covariance
    static_bad_fraction: float  # Fraction of 2D pixels flagged as static defects/shadows
    transient_masked_fraction: float  # Fraction of active-pixel voxels masked as Bragg/zingers
    dynamic_range_cv: float  # Coefficient of variation std(mu_b)/mean(mu_b) across bins
    psf_converged: bool  # True if r_PSF < max_lag (baseline ring outside active PSF)
    warnings: tuple[str, ...] = ()
    is_reliable: bool = True


@dataclass
class DetectorStatsReport:
    """Complete result from the detector response estimation pipeline."""

    separation: SeparationResult
    covariance: CovarianceResult
    trend: TrendResult
    masking: MaskingResult
    representative_frame_idx: int = 0
    representative_frame_adu: np.ndarray | None = None  # (H, W) pedestal-subtracted raw frame t_0
    scan_peak_projection_adu: np.ndarray | None = None  # (H, W) temporal max over active pixels
    quality: QualityDiagnostics | None = None

    @property
    def gain(self) -> float:
        return self.separation.gain

    @property
    def cascade_factor(self) -> float:
        return self.separation.cascade_factor

    @property
    def g_gamma(self) -> float:
        return self.covariance.g_gamma

    @property
    def psi(self) -> float:
        return self.covariance.psi

    @property
    def s2_factor(self) -> float:
        return self.covariance.s2_factor

    @property
    def psf_radius(self) -> int:
        return self.covariance.psf_radius


def compute_quality_diagnostics(
    masking: MaskingResult,
    trend: TrendResult,
    cov: CovarianceResult,
) -> QualityDiagnostics:
    """Compute scale-invariant model-fit and data-quality diagnostics."""
    static_bad_frac = float(np.mean(masking.static_bad_mask))
    active_2d = ~masking.static_bad_mask
    if np.any(active_2d):
        transient_frac = float(np.mean(masking.transient_outlier_mask[:, active_2d]))
    else:
        transient_frac = 0.0
    valid_frac = float(np.mean(trend.valid_mask))

    mean_mu = float(np.mean(cov.bin_means)) if cov.bin_means.size > 0 else 0.0
    dyn_cv = (
        float(np.std(cov.bin_means) / max(abs(mean_mu), 1e-30))
        if cov.bin_means.size > 1
        else 0.0
    )

    # Scale-covariant active-variance floor so standardized_residual_std is c^0 invariant:
    # excludes exact zero-variance dark PAD voxels (mu == 0, psi == 0) while retaining all
    # positive-variance readout-noise voxels when psi > 0.
    ped_offset = float(cov.estimated_pedestal_offset)
    g0 = max(float(cov.apparent_single_pixel_gain), 0.0)
    psi_val = max(float(cov.psi), 0.0)
    scale_unit = max(g0, float(np.sqrt(psi_val)), 1e-12)
    min_active_var = (0.25 * max(g0, 1e-12)) ** 2

    mu_shifted = np.maximum(trend.expected_intensity + ped_offset, 0.0)
    pred_var = g0 * mu_shifted + psi_val
    eval_mask = trend.valid_mask & (pred_var >= min_active_var)
    if np.count_nonzero(eval_mask) < 64:
        eval_mask = trend.valid_mask

    if np.any(eval_mask):
        res_eval = trend.residuals[eval_mask]
        var_floor = (1e-3 * scale_unit) ** 2
        sigma_px = np.sqrt(np.maximum(pred_var[eval_mask], var_floor))
        z = res_eval / sigma_px
        tail_frac = float(np.mean(np.abs(z) > 3.5))
        z_clip = z[(z >= -5.0) & (z <= 5.0)]
        z_std = float(np.std(z_clip)) if z_clip.size >= 16 else float(np.std(z))
    else:
        z_std = float("nan")
        tail_frac = 0.0

    warn_list: list[str] = []
    critical_fail = False

    if not np.isfinite(z_std) or z_std > 1.12 or tail_frac > 0.015:
        warn_list.append(
            f"non_gaussian_residuals: standardized residual SD={z_std:.3f}, "
            f"|z|>3.5 tail={100.0 * tail_frac:.2f}% "
            "(possible unmasked overlapping Bragg or diffuse scattering contamination)"
        )
        critical_fail = True
    elif z_std < 0.85:
        warn_list.append(
            f"underdispersed_residuals: standardized residual SD={z_std:.3f} < 0.85 "
            "(possible readout noise overestimation or temporal smoothing)"
        )
        critical_fail = True

    if valid_frac < 0.25:
        warn_list.append(
            f"low_valid_voxel_fraction: only {100.0 * valid_frac:.1f}% of voxels are unmasked"
        )
        critical_fail = True

    if not cov.psf_converged:
        warn_list.append(
            f"psf_window_truncated: r_PSF={cov.psf_radius} reached max_lag ceiling without plateau"
        )
        critical_fail = True

    if cov.g_gamma <= 0.0 or cov.g_gamma_se > 0.25 * max(cov.g_gamma, 1e-12):
        warn_list.append(
            f"high_g_gamma_uncertainty: G*gamma={cov.g_gamma:.3f} +/- {cov.g_gamma_se:.3f}"
        )
        critical_fail = True

    if dyn_cv < 0.05:
        warn_list.append(
            f"narrow_dynamic_range: bin intensity CV={dyn_cv:.3f} < 0.05 "
            "(elevated lag-0 slope/intercept collinearity)"
        )

    return QualityDiagnostics(
        standardized_residual_std=z_std,
        valid_voxel_fraction=valid_frac,
        static_bad_fraction=static_bad_frac,
        transient_masked_fraction=transient_frac,
        dynamic_range_cv=dyn_cv,
        psf_converged=bool(cov.psf_converged),
        warnings=tuple(warn_list),
        is_reliable=not critical_fail,
    )


def estimate_detector_statistics(
    images: np.ndarray,
    pedestal: float = 0.0,
    trend_method: TrendMethodName | TrendEstimator = "hollow_stencil_kriging",
    max_lag: int = 7,
    n_bins: int = 12,
    masking_config: MaskingConfig | None = None,
    covariance_config: CovarianceConfig | None = None,
    separation_config: SeparationConfig | None = None,
    external_static_bad_mask: np.ndarray | None = None,
    external_dynamic_bad_mask: np.ndarray | None = None,
    trusted_range: tuple[float, float] | None = None,
    auto_calibrate_pedestal: bool = True,
) -> DetectorStatsReport:
    """Estimate detector response statistics (G, gamma, psi) from a rotation-scan image stack.

    Parameters
    ----------
    images : np.ndarray
        3D rotation-scan image stack of shape (T, H, W) in ADU.
    pedestal : float, default 0.0
        Detector baseline pedestal in ADU.
    trend_method : {'hollow_stencil_kriging', 'low_rank_spline', 'temporal_difference'} or TrendEstimator
        Spatiotemporal background detrending method name or custom `TrendEstimator` instance.
    max_lag : int, default 7
        Maximum spatial lag radius L in pixels.
    n_bins : int, default 12
        Number of intensity quantile bins B.
    masking_config : MaskingConfig, optional
        Custom anomaly masking thresholds.
    covariance_config : CovarianceConfig, optional
        Custom lagged covariance and IRLS tuning parameters.
    separation_config : SeparationConfig, optional
        Custom gain/cascade separation thresholds.
    external_static_bad_mask : np.ndarray, optional
        Additional 2D boolean mask (H, W) of static untrusted pixels (e.g. from dxtbx).
    external_dynamic_bad_mask : np.ndarray, optional
        Additional 3D boolean mask (T, H, W) of per-frame overloaded/untrusted voxels.
    trusted_range : tuple of (min_trusted, max_trusted), optional
        Raw ADU trusted range bounds; underflow pixels join the static bad mask while
        saturated pixels (> max_trusted) are dynamically masked per frame.
    auto_calibrate_pedestal : bool, default True
        Whether to automatically adjust intensity bin means when off-diagonal ring-1
        covariance detects a significant (>5 sigma) additive pedestal offset.

    Returns
    -------
    DetectorStatsReport
        Full report containing masking, trend, lagged covariance, (G, gamma, psi) estimates,
        and `QualityDiagnostics`.
    """
    y_sub, nonfinite_mask = validate_and_sanitize_stack(images, pedestal=pedestal)

    use_mask_cfg = masking_config if masking_config is not None else MaskingConfig()
    if trusted_range is not None and use_mask_cfg.trusted_range is None:
        use_mask_cfg = replace(use_mask_cfg, trusted_range=trusted_range)

    static_mask = detect_static_bad_pixels(
        images,
        pedestal=pedestal,
        config=use_mask_cfg,
        _precomputed_y_sub=y_sub,
        _nonfinite_mask=nonfinite_mask,
    )
    if external_static_bad_mask is not None:
        ext_static = np.asarray(external_static_bad_mask, dtype=bool)
        if ext_static.shape != static_mask.shape:
            raise ValueError(
                f"external_static_bad_mask shape {ext_static.shape} != {static_mask.shape}."
            )
        static_mask = static_mask | ext_static

    mask_res = detect_transient_outliers(
        images,
        pedestal=pedestal,
        static_bad_mask=static_mask,
        config=use_mask_cfg,
        external_dynamic_bad_mask=external_dynamic_bad_mask,
        _precomputed_y_sub=y_sub,
        _nonfinite_mask=nonfinite_mask,
    )

    if isinstance(trend_method, str):
        if trend_method == "hollow_stencil_kriging":
            detrender: TrendEstimator = HollowStencilKrigingTrend()
        elif trend_method == "low_rank_spline":
            detrender = LowRankSplineTrend()
        elif trend_method == "temporal_difference":
            detrender = TemporalDifferenceEstimator()
        else:
            raise ValueError(f"Unknown trend_method: {trend_method!r}")
    elif isinstance(trend_method, TrendEstimator):
        detrender = trend_method
    else:
        raise ValueError(
            f"trend_method must be a method name or TrendEstimator, got {trend_method!r}"
        )

    if isinstance(
        detrender,
        (HollowStencilKrigingTrend, LowRankSplineTrend, TemporalDifferenceEstimator),
    ):
        trend_res = detrender.fit_predict(
            images,
            valid_mask=mask_res.valid_mask,
            pedestal=pedestal,
            _precomputed_y_sub=y_sub,
        )
    else:
        trend_res = detrender.fit_predict(
            images,
            valid_mask=mask_res.valid_mask,
            pedestal=pedestal,
        )

    is_nonneg_int = False
    if np.any(mask_res.valid_mask):
        valid_vals = y_sub[mask_res.valid_mask]
        is_nonneg_int = bool(
            float(np.min(valid_vals)) >= -1e-6 and is_effectively_integer_adu(valid_vals)
        )

    cov_res = estimate_lagged_covariance(
        trend_res,
        max_lag=max_lag,
        n_bins=n_bins,
        auto_calibrate_pedestal=auto_calibrate_pedestal,
        config=covariance_config,
        _is_nonneg_integer=is_nonneg_int,
    )

    sep_res = separate_gain_and_cascade(
        images=images,
        trend_result=trend_res,
        cov_result=cov_res,
        pedestal=pedestal,
        static_bad_mask=mask_res.static_bad_mask,
        config=separation_config,
        _precomputed_y_sub=y_sub,
    )

    t0 = int(np.argmax(np.sum(mask_res.transient_outlier_mask, axis=(1, 2))))
    rep_frame = y_sub[t0].copy()
    peak_proj = np.max(y_sub, axis=0)
    if np.any(mask_res.static_bad_mask):
        peak_proj[mask_res.static_bad_mask] = 0.0

    quality = compute_quality_diagnostics(mask_res, trend_res, cov_res)

    return DetectorStatsReport(
        separation=sep_res,
        covariance=cov_res,
        trend=trend_res,
        masking=mask_res,
        representative_frame_idx=t0,
        representative_frame_adu=rep_frame,
        scan_peak_projection_adu=peak_proj,
        quality=quality,
    )
