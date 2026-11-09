"""End-to-end estimation pipeline for detector response statistics (G, gamma, psi)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from detector_stats.covariance import CovarianceResult, estimate_lagged_covariance
from detector_stats.masking import (
    MaskingConfig,
    MaskingResult,
    detect_static_bad_pixels,
    detect_transient_outliers,
)
from detector_stats.separation import SeparationResult, separate_gain_and_cascade
from detector_stats.trend import (
    HollowStencilKrigingTrend,
    LowRankSplineTrend,
    TemporalDifferenceEstimator,
    TrendResult,
)


@dataclass
class DetectorStatsReport:
    """Complete result from the detector response estimation pipeline."""

    separation: SeparationResult
    covariance: CovarianceResult
    trend: TrendResult
    masking: MaskingResult
    representative_frame_idx: int = 0
    representative_frame_adu: np.ndarray | None = None  # (H, W) pedestal-subtracted raw frame t_0
    scan_peak_projection_adu: np.ndarray | None = None  # (H, W) temporal max projection over unmasked static pixels

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


def estimate_detector_statistics(
    images: np.ndarray,
    pedestal: float = 0.0,
    trend_method: Literal[
        "hollow_stencil_kriging", "low_rank_spline", "temporal_difference"
    ] = "hollow_stencil_kriging",
    max_lag: int = 7,
    n_bins: int = 12,
    masking_config: MaskingConfig | None = None,
    external_static_bad_mask: np.ndarray | None = None,
    auto_calibrate_pedestal: bool = True,
) -> DetectorStatsReport:
    """Estimate detector response statistics (G, gamma, psi) from a rotation-scan image stack.

    Parameters
    ----------
    images : np.ndarray
        3D rotation-scan image stack of shape (T, H, W) in ADU.
    pedestal : float, default 0.0
        Detector baseline pedestal in ADU.
    trend_method : {'hollow_stencil_kriging', 'low_rank_spline', 'temporal_difference'}
        Spatiotemporal background detrending method to use.
    max_lag : int, default 7
        Maximum spatial lag radius L in pixels.
    n_bins : int, default 12
        Number of intensity quantile bins B.
    masking_config : MaskingConfig, optional
        Custom anomaly masking thresholds.
    external_static_bad_mask : np.ndarray, optional
        Additional 2D boolean mask (H, W) of untrusted/masked pixels (e.g. from dxtbx).

    Returns
    -------
    DetectorStatsReport
        Full report containing masking, trend, lagged covariance, and (G, gamma, psi) estimates.
    """
    static_mask = detect_static_bad_pixels(
        images, pedestal=pedestal, config=masking_config
    )
    if external_static_bad_mask is not None:
        static_mask = static_mask | np.asarray(external_static_bad_mask, dtype=bool)

    mask_res = detect_transient_outliers(
        images,
        pedestal=pedestal,
        static_bad_mask=static_mask,
        config=masking_config,
    )

    if trend_method == "hollow_stencil_kriging":
        detrender = HollowStencilKrigingTrend()
    elif trend_method == "low_rank_spline":
        detrender = LowRankSplineTrend()
    elif trend_method == "temporal_difference":
        detrender = TemporalDifferenceEstimator()
    else:
        raise ValueError(f"Unknown trend_method: {trend_method!r}")

    trend_res = detrender.fit_predict(
        images, valid_mask=mask_res.valid_mask, pedestal=pedestal
    )
    cov_res = estimate_lagged_covariance(
        trend_res,
        max_lag=max_lag,
        n_bins=n_bins,
        auto_calibrate_pedestal=auto_calibrate_pedestal,
    )

    sep_res = separate_gain_and_cascade(
        images=images,
        trend_result=trend_res,
        cov_result=cov_res,
        pedestal=pedestal,
        static_bad_mask=mask_res.static_bad_mask,
    )

    y_sub = np.asarray(images, dtype=np.float64) - float(pedestal)
    t0 = int(np.argmax(np.sum(mask_res.transient_outlier_mask, axis=(1, 2))))
    rep_frame = y_sub[t0].copy()
    y_no_static = np.where(mask_res.static_bad_mask[None, :, :], 0.0, y_sub)
    peak_proj = np.max(y_no_static, axis=0)

    return DetectorStatsReport(
        separation=sep_res,
        covariance=cov_res,
        trend=trend_res,
        masking=mask_res,
        representative_frame_idx=t0,
        representative_frame_adu=rep_frame,
        scan_peak_projection_adu=peak_proj,
    )
