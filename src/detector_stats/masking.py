"""Spatiotemporal anomaly masking for diffraction rotation-scan time series.

Separates static detector defects (dead pixels, hot/flickering pixels, shadowed regions)
and transient outliers (3D Bragg reflections, single-frame cosmic-ray zingers) from
the smooth background time series without biasing single-pixel or lagged covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    median_filter,
    uniform_filter,
)


@dataclass
class MaskingConfig:
    """Configuration parameters for static and transient anomaly masking."""

    # Static bad-pixel and shadow detection
    hot_pixel_spatial_sigma: float = 8.0
    dead_pixel_rel_threshold: float = 0.08
    shadow_rel_threshold: float = 0.05
    # Sliding-window temporal outlier detection (Bragg spots & zingers)
    temporal_window: int = 15
    transient_seed_sigma: float = 4.5
    transient_cluster_sigma: float = 2.8
    transient_wing_sigma: float = 1.4
    dilation_t: int = 2
    dilation_xy: int = 3


@dataclass
class MaskingResult:
    """Container for static and transient outlier masks."""

    static_bad_mask: np.ndarray  # (H, W) bool: True where pixel is dead, hot, or shadowed
    transient_outlier_mask: np.ndarray  # (T, H, W) bool: True where Bragg spot or zinger is present
    valid_mask: np.ndarray  # (T, H, W) bool: True where pixel-frame observation is valid background
    rough_flux_scale: np.ndarray  # (T,) float64: initial global frame-flux estimate I_0(t)


def estimate_rough_frame_flux(
    y_sub: np.ndarray, static_bad_mask: np.ndarray | None = None
) -> np.ndarray:
    """Estimate rough frame-to-frame incident flux scale I_0(t) from trimmed spatial means.

    Parameters
    ----------
    y_sub : np.ndarray
        Pedestal-subtracted image stack of shape (T, H, W).
    static_bad_mask : np.ndarray, optional
        Boolean mask of shape (H, W) where True indicates bad/shadowed pixels.

    Returns
    -------
    flux_scale : np.ndarray
        Normalized frame flux scale factor of shape (T,) with mean 1.0.
    """
    n_frames = y_sub.shape[0]
    flux = np.ones(n_frames, dtype=np.float64)
    valid_2d = ~static_bad_mask if static_bad_mask is not None else np.ones(y_sub.shape[1:], dtype=bool)
    if not np.any(valid_2d):
        return flux

    for t_idx in range(n_frames):
        vals = y_sub[t_idx, valid_2d]
        # Use interquartile / 10-80% trimmed mean to ignore Bragg reflections and zingers
        q10, q80 = np.percentile(vals, [10.0, 80.0])
        in_range = vals[(vals >= q10) & (vals <= q80)]
        flux[t_idx] = float(np.mean(in_range)) if in_range.size > 0 else float(np.mean(vals))

    flux = np.clip(flux, 1e-6, None)
    mean_flux = float(np.mean(flux))
    if mean_flux > 0:
        flux /= mean_flux
    return flux


def detect_static_bad_pixels(
    images: np.ndarray,
    pedestal: float = 0.0,
    config: MaskingConfig | None = None,
) -> np.ndarray:
    """Detect dead pixels, hot/flickering pixels, and shadowed detector borders.

    Parameters
    ----------
    images : np.ndarray
        Observed image stack of shape (T, H, W) in ADU.
    pedestal : float, default 0.0
        Detector baseline pedestal in ADU.
    config : MaskingConfig, optional
        Threshold configuration.

    Returns
    -------
    static_bad_mask : np.ndarray
        2D boolean mask of shape (H, W) where True marks defective or shadowed pixels.
    """
    if config is None:
        config = MaskingConfig()

    y_sub = np.asarray(images, dtype=np.float64) - float(pedestal)
    n_frames, height, width = y_sub.shape

    # Per-pixel temporal median and temporal variance across the scan
    temp_median = np.median(y_sub, axis=0)
    temp_mean = np.mean(y_sub, axis=0)
    temp_var = np.var(y_sub, axis=0)

    # Local 5x5 spatial median and local spatial MAD of the temporal median image
    local_med_5 = median_filter(temp_median, size=5, mode="nearest")
    local_mad_5 = 1.4826 * median_filter(
        np.abs(temp_median - local_med_5), size=7, mode="nearest"
    )
    local_mean_9 = uniform_filter(temp_mean, size=9, mode="nearest")

    # Estimate baseline readout noise floor from temporal differences in lowest-signal pixels
    if n_frames > 1:
        diff_var = 0.5 * np.median((y_sub[1:] - y_sub[:-1]) ** 2, axis=0) / 0.4549364
        read_sigma_est = float(np.sqrt(max(np.percentile(diff_var, 5.0), 0.0)))
    else:
        read_sigma_est = 0.0

    # 1. Hot / flickering pixels: temporal median or temporal variance sticks out sharply
    # above the immediate 5x5 spatial neighborhood
    scale_floor = np.maximum(local_mad_5, np.sqrt(np.maximum(local_med_5, 1.0)))
    hot_mask = (temp_median - local_med_5) > (config.hot_pixel_spatial_sigma * scale_floor + 10.0)

    local_var_med = median_filter(temp_var, size=5, mode="nearest")
    local_var_mad = 1.4826 * median_filter(
        np.abs(temp_var - local_var_med), size=7, mode="nearest"
    )
    hot_var_mask = (
        (temp_var - local_var_med)
        > config.hot_pixel_spatial_sigma * np.maximum(local_var_mad, local_var_med + 1.0)
    ) & (temp_median > local_med_5 + 3.0 * scale_floor)
    hot_mask |= hot_var_mask

    # 2. Isolated dead pixels: zero temporal variance or signal collapsed relative to active neighborhood
    # Require that the immediate 5x5 median is active (> 0.5 ADU) and cumulative 9x9 flux (T * local_mean > 20)
    # so genuine zero-count pixels in sparse low-flux regions are never flagged as dead.
    active_neighborhood = ((n_frames * local_mean_9) > 20.0) & (local_med_5 > 0.5)
    dead_mask = active_neighborhood & (
        ((temp_var <= 1e-10) & ((np.abs(temp_mean) <= max(read_sigma_est, 1e-6)) | np.all(images == 0.0, axis=0)))
        | (temp_mean < config.dead_pixel_rel_threshold * local_med_5)
    )
    # Also catch pixels pinned at raw 0.0 when pedestal > 0 or local neighborhood is active
    pinned_zero = np.all(np.abs(images) <= 1e-10, axis=0) & active_neighborhood
    dead_mask |= pinned_zero

    # 3. Shadowed border strips around the perimeter (where an entire edge row/col receives zero flux,
    # including its central/80th-percentile segment, distinguishing a full border shadow from sparse corners)
    global_ref = float(np.percentile(temp_mean, 60.0))
    shadow_thresh = max(
        3.5 * read_sigma_est / np.sqrt(max(n_frames, 1)),
        0.02 * global_ref,
    )
    shadow_mask = np.zeros((height, width), dtype=bool)
    if global_ref > 0.5:
        max_margin = min(height, width) // 4
        # Top border rows
        for r in range(max_margin):
            if float(np.percentile(temp_mean[r, :], 80.0)) <= shadow_thresh:
                shadow_mask[r, :] = True
            else:
                break
        # Bottom border rows
        for r in range(height - 1, height - 1 - max_margin, -1):
            if float(np.percentile(temp_mean[r, :], 80.0)) <= shadow_thresh:
                shadow_mask[r, :] = True
            else:
                break
        # Left border cols
        for c in range(max_margin):
            if float(np.percentile(temp_mean[:, c], 80.0)) <= shadow_thresh:
                shadow_mask[:, c] = True
            else:
                break
        # Right border cols
        for c in range(width - 1, width - 1 - max_margin, -1):
            if float(np.percentile(temp_mean[:, c], 80.0)) <= shadow_thresh:
                shadow_mask[:, c] = True
            else:
                break

    return hot_mask | dead_mask | shadow_mask


def detect_transient_outliers(
    images: np.ndarray,
    pedestal: float = 0.0,
    static_bad_mask: np.ndarray | None = None,
    config: MaskingConfig | None = None,
) -> MaskingResult:
    """Detect transient 3D Bragg reflections and cosmic-ray zingers along pixel time series.

    Uses a flux-normalized sliding-window temporal median and spatially regularized MAD scale
    so that point-spread correlations do not bias the threshold and pure background tails are
    not truncated.

    Parameters
    ----------
    images : np.ndarray
        Observed image stack of shape (T, H, W) in ADU.
    pedestal : float, default 0.0
        Detector baseline pedestal in ADU.
    static_bad_mask : np.ndarray, optional
        2D boolean mask of shape (H, W). Computed automatically if None.
    config : MaskingConfig, optional
        Masking parameters.

    Returns
    -------
    MaskingResult
        Static bad mask, transient 3D outlier mask, combined valid mask, and rough flux scale.
    """
    if config is None:
        config = MaskingConfig()
    if static_bad_mask is None:
        static_bad_mask = detect_static_bad_pixels(images, pedestal=pedestal, config=config)

    y_sub = np.asarray(images, dtype=np.float64) - float(pedestal)
    n_frames, height, width = y_sub.shape

    rough_flux = estimate_rough_frame_flux(y_sub, static_bad_mask=static_bad_mask)
    y_norm = y_sub / rough_flux[:, None, None]

    # Sliding-window temporal median along axis 0 (time)
    win_t = min(max(5, int(config.temporal_window) | 1), n_frames)
    if n_frames <= 15:
        temp_med = np.median(y_norm, axis=0, keepdims=True)
        abs_dev = np.abs(y_norm - temp_med)
        mad_t = 1.4826 * np.median(abs_dev, axis=0, keepdims=True)
    else:
        temp_med = median_filter(y_norm, size=(win_t, 1, 1), mode="reflect")
        abs_dev = np.abs(y_norm - temp_med)
        mad_t = 1.4826 * median_filter(abs_dev, size=(win_t, 1, 1), mode="reflect")

    # Regularize the temporal MAD scale spatially (over 7x7 neighborhoods) so single pixels
    # with quantized/low-count draws never get an artificially tiny denominator.
    mean_mad_2d = np.median(mad_t, axis=0)
    if np.any(static_bad_mask):
        fill_val = float(np.median(mean_mad_2d[~static_bad_mask])) if np.any(~static_bad_mask) else 1.0
        mean_mad_2d_clean = np.where(static_bad_mask, fill_val, mean_mad_2d)
    else:
        mean_mad_2d_clean = mean_mad_2d
    spatial_scale_2d = median_filter(mean_mad_2d_clean, size=7, mode="nearest")

    # Poisson/discrete floor prevents false-positive clipping in ultra-low-flux sparse corners
    poisson_floor = np.sqrt(np.maximum(temp_med, 1.5))
    sigma_eff = np.maximum(mad_t, np.maximum(spatial_scale_2d[None, :, :], poisson_floor))

    # Positive excess z-score (Bragg spots and cosmic-ray zingers are positive spikes)
    z_pos = (y_norm - temp_med) / (sigma_eff + 1e-6)

    # High-confidence single-voxel seeds (catches zingers and Bragg peak cores)
    # Require actual excess above a minimum discrete step (> 4 ADU) to never trigger on single photons
    raw_excess = y_norm - temp_med
    seed_mask = (z_pos > config.transient_seed_sigma) & (raw_excess > 4.0)

    # Also detect locally correlated 2x2 spatial excesses for broad, moderate-intensity Bragg spots
    smooth_z = uniform_filter(z_pos, size=(1, 3, 3), mode="nearest")
    cluster_seeds = (smooth_z > config.transient_cluster_sigma) & (z_pos > 2.2) & (raw_excess > 2.5)
    seed_mask |= cluster_seeds

    # Do not seed from static bad pixels
    seed_mask &= ~static_bad_mask[None, :, :]

    # Propagate seeds into contiguous reflection halos via hysteresis thresholding,
    # then dilate in 3D (t, y, x) to mask out weak Bragg reflection wings and PSF tails
    if np.any(seed_mask):
        wing_candidates = ((smooth_z > config.transient_wing_sigma) | (z_pos > 2.0)) & (
            ~static_bad_mask[None, :, :]
        )
        grown_mask = binary_propagation(seed_mask, mask=wing_candidates | seed_mask)
        if config.dilation_t > 0 or config.dilation_xy > 0:
            st_t = 2 * max(0, config.dilation_t) + 1
            st_xy = 2 * max(0, config.dilation_xy) + 1
            struct = np.ones((st_t, st_xy, st_xy), dtype=bool)
            transient_mask = binary_dilation(grown_mask, structure=struct)
        else:
            transient_mask = grown_mask
    else:
        transient_mask = seed_mask

    valid_mask = (~static_bad_mask[None, :, :]) & (~transient_mask)
    return MaskingResult(
        static_bad_mask=static_bad_mask,
        transient_outlier_mask=transient_mask,
        valid_mask=valid_mask,
        rough_flux_scale=rough_flux,
    )
