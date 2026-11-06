"""Spatiotemporal estimation of detector response statistics (G, gamma, psi) from diffraction scans."""

from .covariance import CovarianceResult, estimate_lagged_covariance
from .dxtbx_loader import (
    DxtbxTileData,
    compute_dials_dispersion_gain_on_tile,
    estimate_from_dxtbx,
    extract_dxtbx_tile,
    load_imageset,
    select_default_roi_bbox,
)
from .masking import (
    MaskingConfig,
    MaskingResult,
    detect_static_bad_pixels,
    detect_transient_outliers,
)
from .pipeline import DetectorStatsReport, estimate_detector_statistics
from .plotting import (
    plot_detector_stats_report,
    plot_incident_flux_and_outliers,
    plot_multi_lag_ptc,
    plot_psf_autocorrelation_kernel,
    plot_radial_convergence,
    plot_representative_frame_and_masks,
    plot_scan_peak_projection_and_masks,
    plot_separation_or_residuals,
    plot_spatial_background_and_masks,
)
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
    "DxtbxTileData",
    "HollowStencilKrigingTrend",
    "LowRankSplineTrend",
    "MaskingConfig",
    "MaskingResult",
    "SceneConfig",
    "SeparationResult",
    "SimulationResult",
    "TemporalDifferenceEstimator",
    "TrendResult",
    "compute_dials_dispersion_gain_on_tile",
    "delta_psf",
    "detect_static_bad_pixels",
    "detect_transient_outliers",
    "estimate_detector_statistics",
    "estimate_from_dxtbx",
    "estimate_incident_flux_scale",
    "estimate_lagged_covariance",
    "estimate_sparse_cluster_multiplicity",
    "extract_dxtbx_tile",
    "gaussian_psf",
    "load_imageset",
    "lorentzian_psf",
    "plot_detector_stats_report",
    "plot_incident_flux_and_outliers",
    "plot_multi_lag_ptc",
    "plot_psf_autocorrelation_kernel",
    "plot_radial_convergence",
    "plot_representative_frame_and_masks",
    "plot_scan_peak_projection_and_masks",
    "plot_separation_or_residuals",
    "plot_spatial_background_and_masks",
    "select_default_roi_bbox",
    "separate_gain_and_cascade",
    "simulate_rotation_scan",
    "timepix_cluster_psf",
]
