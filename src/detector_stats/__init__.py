"""Spatiotemporal estimation of detector response statistics (G, gamma, psi) from diffraction scans."""

from .covariance import CovarianceResult, estimate_lagged_covariance
from .masking import (
    MaskingConfig,
    MaskingResult,
    detect_static_bad_pixels,
    detect_transient_outliers,
)
from .pipeline import DetectorStatsReport, estimate_detector_statistics
from .separation import (
    SeparationResult,
    estimate_sparse_cluster_multiplicity,
    separate_gain_and_cascade,
)
from .simulator import (
    DetectorConfig,
    SceneConfig,
    SimulationResult,
    delta_psf,
    gaussian_psf,
    lorentzian_psf,
    simulate_rotation_scan,
    timepix_cluster_psf,
)
from .trend import (
    HollowStencilKrigingTrend,
    LowRankSplineTrend,
    TemporalDifferenceEstimator,
    TrendResult,
    estimate_incident_flux_scale,
)

__all__ = [
    "CovarianceResult",
    "DetectorConfig",
    "DetectorStatsReport",
    "HollowStencilKrigingTrend",
    "LowRankSplineTrend",
    "MaskingConfig",
    "MaskingResult",
    "SceneConfig",
    "SeparationResult",
    "SimulationResult",
    "TemporalDifferenceEstimator",
    "TrendResult",
    "delta_psf",
    "detect_static_bad_pixels",
    "detect_transient_outliers",
    "estimate_detector_statistics",
    "estimate_incident_flux_scale",
    "estimate_lagged_covariance",
    "estimate_sparse_cluster_multiplicity",
    "gaussian_psf",
    "lorentzian_psf",
    "separate_gain_and_cascade",
    "simulate_rotation_scan",
    "timepix_cluster_psf",
]
