"""Separation of absorbed quantum gain G and cascade factor gamma.

Implements:
1. Sparse single-event cluster multiplicity analysis (generalizing Vypritskaia et al., 2025)
   for low-background detector regions with low readout noise.
2. Exploratory 3rd-order spatial cumulant (skewness) analysis.
3. The agreed fallback assignment (G = 1.0, gamma = G*gamma) when G and gamma cannot be
   decoupled from continuous background statistics alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label, median_filter

from detector_stats.covariance import CovarianceResult
from detector_stats.trend import TrendResult


@dataclass
class SeparationResult:
    """Final separated detector response statistics (G, gamma, psi)."""

    gain: float  # Estimated absorbed quantum gain G (ADU per primary quantum)
    gain_se: float  # Standard error on G (0.0 if fallback G=1.0 is used)
    cascade_factor: float  # Estimated cascade factor gamma = 1 + Var(M)/E[M]^2
    cascade_factor_se: float  # Standard error on gamma
    g_gamma: float  # Estimated product G * gamma from lagged spatial covariance
    g_gamma_se: float  # Standard error on G * gamma
    psi: float  # Estimated pixel noise variance factor psi (ADU^2)
    psi_se: float  # Standard error on psi
    s2_factor: float  # Zero-lag PSF variance damping factor S_2 = alpha(0)
    psf_radius: int  # Detected PSF autocorrelation support radius r_PSF
    separated: bool  # True if G and gamma were independently separated from data
    separation_method: str  # 'sparse_cluster_multiplicity', 'photon_counting_pad', or 'fallback_g_one'
    n_isolated_clusters: int = 0
    sparse_gamma_direct: float | None = None
    cumulant_g_estimate: float | None = None
    cluster_sums: np.ndarray | None = None
    cluster_areas: np.ndarray | None = None


def estimate_sparse_cluster_multiplicity(
    images: np.ndarray,
    static_bad_mask: np.ndarray,
    cov_result: CovarianceResult,
    pedestal: float = 0.0,
    max_sparse_quanta_rate: float = 0.03,
    min_clusters: int = 40,
) -> SeparationResult | None:
    """Attempt to separate G and gamma using isolated single-event clusters in sparse regions.

    Follows and generalizes the single-event cluster multiplicity method of
    Vypritskaia et al. (2025), Acta Cryst. C81, 1-13.

    Parameters
    ----------
    images : np.ndarray
        Observed image stack (T, H, W) in ADU.
    static_bad_mask : np.ndarray
        Boolean mask (H, W) of static bad or shadowed pixels.
    cov_result : CovarianceResult
        Lagged spatial covariance regression result providing G*gamma and psi.
    pedestal : float, default 0.0
        Detector baseline pedestal in ADU.
    max_sparse_quanta_rate : float, default 0.03
        Maximum incident primary rate lambda (events/pixel/frame) qualifying as sparse.
    min_clusters : int, default 40
        Minimum number of isolated single-event clusters required for a reliable estimate.

    Returns
    -------
    SeparationResult or None
        Separated (G, gamma, psi) if sufficient isolated clusters are found, else None.
    """
    # Require low readout noise so individual quantum hits stand clearly above baseline noise
    gg_scale = max(float(cov_result.g_gamma), 1e-12)
    if np.sqrt(max(cov_result.psi, 0.0)) > 0.25 * gg_scale:
        return None

    y_sub = np.asarray(images, dtype=np.float64) - float(pedestal)
    n_frames, height, width = y_sub.shape
    valid_2d = ~static_bad_mask

    # Compute 2D temporally averaged and edge-preserving spatially smoothed background rate
    p95 = np.percentile(y_sub, 95.0, axis=0)
    y_clipped = np.minimum(y_sub, np.maximum(p95[None, :, :], 20.0 * gg_scale))
    mean_2d = np.where(valid_2d, np.mean(y_clipped, axis=0), 1e9 * gg_scale)
    smooth_mean_2d = median_filter(mean_2d, size=5, mode="nearest")

    approx_rate_2d = smooth_mean_2d / gg_scale
    sparse_2d = valid_2d & (approx_rate_2d >= 0.0) & (approx_rate_2d <= max_sparse_quanta_rate)

    # Exclude a 2-pixel border around the image perimeter so truncated edge clusters are ignored
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[2 : height - 2, 2 : width - 2] = True
    sparse_2d &= border_mask

    n_sparse_px = int(np.count_nonzero(sparse_2d))
    if n_sparse_px * n_frames < 1500:
        return None

    valid_px = y_sub[:, valid_2d] if np.any(valid_2d) else y_sub
    is_integer = bool(np.percentile(np.abs(valid_px - np.rint(valid_px)), 99.0) < 1e-6)
    if is_integer and cov_result.psi < 1.0:
        hit_threshold = 0.5
    else:
        g0_scale = max(float(cov_result.apparent_single_pixel_gain), 0.1 * gg_scale)
        hit_threshold = max(0.35 * g0_scale, 4.0 * np.sqrt(max(cov_result.psi, 0.0)))
    max_span = max(3, 2 * min(cov_result.psf_radius, 2) + 1)
    max_single_event_adu = 12.0 * gg_scale

    cluster_sums: list[float] = []
    cluster_areas: list[int] = []
    n_raw_components = 0

    bridge_struct = np.ones((3, 3), dtype=bool)
    label_struct = np.ones((3, 3), dtype=int)

    for t in range(n_frames):
        frame = y_sub[t]
        hits = (frame >= hit_threshold) & sparse_2d
        if not np.any(hits):
            continue

        # Bridge 1-pixel gaps within a 3x3 single-event PSF footprint before labeling
        bridged = binary_dilation(hits, structure=bridge_struct)
        labeled, n_comp = label(bridged, structure=label_struct)
        if n_comp == 0:
            continue
        n_raw_components += n_comp
        slices = find_objects(labeled)

        for comp_idx, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            ys, xs = sl
            comp_hits = (labeled[ys, xs] == comp_idx) & hits[ys, xs]
            if not np.any(comp_hits):
                continue

            hy, hx = np.nonzero(comp_hits)
            # Reject clusters touching the boundary of the 2D sparse region
            abs_y = hy + ys.start
            abs_x = hx + xs.start
            y0, y1 = max(0, int(np.min(abs_y)) - 1), min(height, int(np.max(abs_y)) + 2)
            x0, x1 = max(0, int(np.min(abs_x)) - 1), min(width, int(np.max(abs_x)) + 2)
            if not np.all(sparse_2d[y0:y1, x0:x1]):
                continue

            # Reject multi-event overlaps or Bragg reflections by bounding box span and signal cap
            span_y = int(np.max(hy) - np.min(hy)) + 1
            span_x = int(np.max(hx) - np.min(hx)) + 1
            if span_y > max_span or span_x > max_span:
                continue

            if is_integer:
                comp_support = comp_hits
            else:
                # For continuous PSF detectors with readout noise, integrate over the 1-px dilated
                # cluster support inside the bounding slice so sub-threshold PSF wings are not clipped
                comp_support = (labeled[ys, xs] == comp_idx) & sparse_2d[ys, xs]

            total_adu = float(np.sum(frame[ys, xs][comp_support]))
            if total_adu > max_single_event_adu or total_adu <= 0.0:
                continue

            area = int(np.count_nonzero(comp_support))
            cluster_sums.append(total_adu)
            cluster_areas.append(area)

    n_clusters = len(cluster_sums)
    if n_clusters < min_clusters:
        return None

    sums_arr = np.asarray(cluster_sums, dtype=np.float64)
    areas_arr = np.asarray(cluster_areas, dtype=np.float64)

    # Empirical primary event rate lambda_obs in the sparse region
    total_sparse_voxels = float(n_sparse_px * n_frames)
    lambda_obs = float(n_raw_components) / total_sparse_voxels
    # First-order Poisson coalescence correction for compact 3x3 boxes (effective area ~9 px):
    overlap_factor = 1.0 + 4.5 * lambda_obs

    raw_mean_s = float(np.mean(sums_arr))
    raw_var_s = float(np.var(sums_arr, ddof=1))
    mean_area = float(np.mean(areas_arr))

    gain_est = raw_mean_s / overlap_factor
    gain_se = float(np.std(sums_arr, ddof=1) / np.sqrt(n_clusters)) / overlap_factor

    # Direct sparse cascade factor estimate: gamma_sparse = 1 + (Var(S) - N_c * psi) / G^2
    p_double = max(0.0, 1.0 - 1.0 / overlap_factor)
    var_single = max(0.0, raw_var_s - p_double * (gain_est**2) - mean_area * cov_result.psi)
    gamma_sparse = 1.0 + var_single / max(gain_est**2, 1e-12)

    # Derive cascade factor gamma = (G * gamma) / G so the macroscopic product G * gamma
    # from the full detector is exactly preserved, while reporting gamma_sparse as a diagnostic.
    gamma_from_ratio = max(1.0, float(cov_result.g_gamma / max(gain_est, 1e-12)))
    ratio_se = float(
        gamma_from_ratio
        * np.hypot(
            cov_result.g_gamma_se / max(cov_result.g_gamma, 1e-12),
            gain_se / max(gain_est, 1e-12),
        )
    )

    return SeparationResult(
        gain=float(gain_est),
        gain_se=float(gain_se),
        cascade_factor=gamma_from_ratio,
        cascade_factor_se=ratio_se,
        g_gamma=cov_result.g_gamma,
        g_gamma_se=cov_result.g_gamma_se,
        psi=cov_result.psi,
        psi_se=cov_result.psi_se,
        s2_factor=cov_result.s2_factor,
        psf_radius=cov_result.psf_radius,
        separated=True,
        separation_method="sparse_cluster_multiplicity",
        n_isolated_clusters=n_clusters,
        sparse_gamma_direct=float(gamma_sparse),
        cluster_sums=sums_arr,
        cluster_areas=areas_arr,
    )


def estimate_cumulant_g_exploratory(
    trend_result: TrendResult,
    cov_result: CovarianceResult,
) -> float | None:
    """Exploratory 3rd-order central moment (skewness) estimator for G in continuous regimes."""
    res = trend_result.residuals
    exp = trend_result.expected_intensity
    val = trend_result.valid_mask
    gg_scale = max(float(cov_result.g_gamma), 1e-12)
    active = val & (exp >= gg_scale)
    if np.count_nonzero(active) < 5000:
        return None
    r_act = res[active]
    k2 = float(np.mean(r_act**2)) - cov_result.psi
    k3 = float(np.mean(r_act**3))
    if k2 <= 0 or k3 <= 0:
        return None
    s3 = float(np.sum(cov_result.alpha_kernel**1.5))
    return float((k3 / max(s3, 0.05)) / (k2 / max(cov_result.s2_factor, 0.05)))


def separate_gain_and_cascade(
    images: np.ndarray,
    trend_result: TrendResult,
    cov_result: CovarianceResult,
    pedestal: float = 0.0,
    static_bad_mask: np.ndarray | None = None,
) -> SeparationResult:
    """Decouple G and gamma when supported by the data, or apply the G=1.0 fallback rule.

    Priority order:
    1. Sparse single-event cluster multiplicity (when a sparse low-flux region with low
       readout noise is present, e.g. 3D ED on Timepix or sparse corners on PADs).
    2. Unit-gain photon-counting PAD identification (when data are non-negative integers,
       r_PSF == 0, psi ~= 0, and G*gamma ~= 1.0).
    3. Agreed fallback rule: set G = 1.0 and put the entire estimated G*gamma product into
       gamma = G*gamma so that downstream error estimates sigma_p^2 = p_bar*G*gamma + N*psi
       remain exact.
    """
    cumulant_g = estimate_cumulant_g_exploratory(trend_result, cov_result)

    if static_bad_mask is None:
        static_bad_mask = ~np.any(trend_result.valid_mask, axis=0)

    # 1. Try sparse single-event cluster multiplicity
    sparse_res = estimate_sparse_cluster_multiplicity(
        images=images,
        static_bad_mask=static_bad_mask,
        cov_result=cov_result,
        pedestal=pedestal,
    )
    if sparse_res is not None:
        sparse_res.cumulant_g_estimate = cumulant_g
        return sparse_res

    # 2. Check if detector is a unit-gain photon-counting PAD (integer counts, G*gamma ~= 1, psi ~= 0)
    y_sub = np.asarray(images, dtype=np.float64) - float(pedestal)
    valid_px = y_sub[:, ~static_bad_mask] if np.any(~static_bad_mask) else y_sub
    is_integer = bool(np.percentile(np.abs(valid_px - np.rint(valid_px)), 99.0) < 1e-6)
    if (
        is_integer
        and cov_result.psf_radius == 0
        and cov_result.psi < 0.15
        and abs(cov_result.g_gamma - 1.0) < 0.12
    ):
        return SeparationResult(
            gain=1.0,
            gain_se=0.0,
            cascade_factor=cov_result.g_gamma,
            cascade_factor_se=cov_result.g_gamma_se,
            g_gamma=cov_result.g_gamma,
            g_gamma_se=cov_result.g_gamma_se,
            psi=cov_result.psi,
            psi_se=cov_result.psi_se,
            s2_factor=cov_result.s2_factor,
            psf_radius=cov_result.psf_radius,
            separated=True,
            separation_method="photon_counting_pad",
            cumulant_g_estimate=cumulant_g,
        )

    # 3. Fallback rule: set G = 1.0 and assign gamma = G * gamma
    return SeparationResult(
        gain=1.0,
        gain_se=0.0,
        cascade_factor=cov_result.g_gamma,
        cascade_factor_se=cov_result.g_gamma_se,
        g_gamma=cov_result.g_gamma,
        g_gamma_se=cov_result.g_gamma_se,
        psi=cov_result.psi,
        psi_se=cov_result.psi_se,
        s2_factor=cov_result.s2_factor,
        psf_radius=cov_result.psf_radius,
        separated=False,
        separation_method="fallback_g_one",
        cumulant_g_estimate=cumulant_g,
    )
