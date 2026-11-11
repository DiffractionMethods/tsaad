"""tsaad: Time Series Analysis of Area Detectors — spatiotemporal estimation of (G, gamma, psi)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .covariance import CovarianceConfig, CovarianceResult, estimate_lagged_covariance
from .masking import (
    MaskingConfig,
    MaskingResult,
    detect_static_bad_pixels,
    detect_transient_outliers,
)
from .pipeline import (
    DetectorStatsReport,
    QualityDiagnostics,
    compute_quality_diagnostics,
    estimate_detector_statistics,
)
from .separation import (
    SeparationConfig,
    SeparationResult,
    estimate_sparse_cluster_multiplicity,
    separate_gain_and_cascade,
)
from .trend import (
    HollowStencilKrigingTrend,
    LowRankSplineTrend,
    TemporalDifferenceEstimator,
    TrendEstimator,
    TrendMethodName,
    TrendResult,
    estimate_incident_flux_scale,
)

if TYPE_CHECKING:
    from .dxtbx_loader import (
        DxtbxTileData,
        compute_dials_dispersion_gain_on_tile,
        estimate_from_dxtbx,
        extract_dxtbx_tile,
        load_imageset,
        select_default_roi_bbox,
    )
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

_LAZY_EXPORTS: dict[str, str] = {
    # dxtbx_loader
    "DxtbxTileData": ".dxtbx_loader",
    "compute_dials_dispersion_gain_on_tile": ".dxtbx_loader",
    "estimate_from_dxtbx": ".dxtbx_loader",
    "extract_dxtbx_tile": ".dxtbx_loader",
    "load_imageset": ".dxtbx_loader",
    "select_default_roi_bbox": ".dxtbx_loader",
    # plotting
    "plot_detector_stats_report": ".plotting",
    "plot_incident_flux_and_outliers": ".plotting",
    "plot_multi_lag_ptc": ".plotting",
    "plot_psf_autocorrelation_kernel": ".plotting",
    "plot_radial_convergence": ".plotting",
    "plot_representative_frame_and_masks": ".plotting",
    "plot_scan_peak_projection_and_masks": ".plotting",
    "plot_separation_or_residuals": ".plotting",
    "plot_spatial_background_and_masks": ".plotting",
    # simulator
    "DetectorConfig": ".simulator",
    "SceneConfig": ".simulator",
    "SimulationResult": ".simulator",
    "delta_psf": ".simulator",
    "gaussian_psf": ".simulator",
    "lorentzian_psf": ".simulator",
    "simulate_rotation_scan": ".simulator",
    "timepix_cluster_psf": ".simulator",
}

__all__ = [
    "CovarianceConfig",
    "CovarianceResult",
    "DetectorConfig",
    "DetectorStatsReport",
    "DxtbxTileData",
    "HollowStencilKrigingTrend",
    "LowRankSplineTrend",
    "MaskingConfig",
    "MaskingResult",
    "QualityDiagnostics",
    "SceneConfig",
    "SeparationConfig",
    "SeparationResult",
    "SimulationResult",
    "TemporalDifferenceEstimator",
    "TrendEstimator",
    "TrendMethodName",
    "TrendResult",
    "compute_dials_dispersion_gain_on_tile",
    "compute_quality_diagnostics",
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


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
