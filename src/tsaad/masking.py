"""Spatiotemporal anomaly masking for diffraction rotation-scan time series.

Separates static detector defects (dead pixels, hot/flickering pixels, shadowed regions)
and transient outliers (3D Bragg reflections, single-frame cosmic-ray zingers) from
the smooth background time series without biasing single-pixel or lagged covariance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    label,
    median_filter,
    uniform_filter,
)

from tsaad._utils import (
    estimate_quantum_scale_adu,
    validate_and_sanitize_stack,
)


@dataclass
class MaskingConfig:
    """Configuration parameters for static and transient anomaly masking."""

    # Static bad-pixel and shadow detection
    hot_pixel_spatial_sigma: float = 8.0
    dead_pixel_rel_threshold: float = 0.08
    shadow_rel_threshold: float = 0.05
    shadow_penumbra_dilation_px: int = 8
    shadow_penumbra_rel_level: float = 0.72
    # Sliding-window temporal outlier detection (Bragg spots & zingers)
    temporal_window: int = 21
    transient_seed_sigma: float = 5.0
    transient_cluster_sigma: float = 4.0
    transient_wing_sigma: float = 1.8
    dilation_t: int = 1
    dilation_xy: int = 2
    # Optional raw ADU (min_trusted, max_trusted) bounds for static underflow / dynamic saturation
    trusted_range: tuple[float, float] | None = None


@dataclass
class MaskingResult:
    """Container for static and transient outlier masks."""

    static_bad_mask: np.ndarray  # (H, W) bool: True where pixel is dead, hot, or shadowed
    transient_outlier_mask: np.ndarray  # (T, H, W) bool: True where Bragg spot or zinger is present
    valid_mask: np.ndarray  # (T, H, W) bool: True where pixel-frame observation is valid background
    rough_flux_scale: np.ndarray  # (T,) float64: initial global frame-flux estimate I_0(t)


def estimate_rough_frame_flux(
    y_sub: np.ndarray,
    static_bad_mask: np.ndarray | None = None,
    dynamic_bad_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate rough frame-to-frame incident flux scale I_0(t) from trimmed spatial means.

    Parameters
    ----------
    y_sub : np.ndarray
        Pedestal-subtracted image stack of shape (T, H, W).
    static_bad_mask : np.ndarray, optional
        Boolean mask of shape (H, W) where True indicates bad/shadowed pixels.
    dynamic_bad_mask : np.ndarray, optional
        Boolean mask of shape (T, H, W) where True indicates per-frame invalid voxels.

    Returns
    -------
    flux_scale : np.ndarray
        Normalized frame flux scale factor of shape (T,) with mean 1.0.
    """
    n_frames = y_sub.shape[0]
    flux = np.ones(n_frames, dtype=np.float64)
    valid_2d = (
        ~static_bad_mask
        if static_bad_mask is not None
        else np.ones(y_sub.shape[1:], dtype=bool)
    )
    if not np.any(valid_2d):
        return flux

    # Choose one consistent rank-trimming window across all frames of the stack:
    # - Standard scans (>=40% positive background pixels on upper-quartile frames): 10%..80%
    #   rank trim, immune to up to 20% Bragg-spot pixel coverage.
    # - Sparse photon-counting scans (<40% positive pixels, where the 80th percentile can hit 0):
    #   0%..95% rank trim across all frames so there is no quantization cliff at 20% occupancy.
    pos_fracs = np.mean((y_sub > 0.0) & valid_2d[None, :, :], axis=(1, 2)) / max(
        float(np.mean(valid_2d)), 1e-12
    )
    use_sparse_trim = float(np.percentile(pos_fracs, 75.0)) < 0.40
    lo_frac, hi_frac = (0.0, 0.95) if use_sparse_trim else (0.10, 0.80)

    for t_idx in range(n_frames):
        if dynamic_bad_mask is not None:
            frame_valid = valid_2d & (~dynamic_bad_mask[t_idx])
            vals = y_sub[t_idx, frame_valid] if np.any(frame_valid) else y_sub[t_idx, valid_2d]
        else:
            vals = y_sub[t_idx, valid_2d]
        n_v = vals.size
        if n_v >= 20:
            lo = int(lo_frac * n_v)
            hi = max(lo + 1, int(hi_frac * n_v))
            if lo == 0:
                part = np.partition(vals, hi - 1)
                flux[t_idx] = float(np.mean(part[:hi]))
            else:
                part = np.partition(vals, (lo, hi - 1))
                flux[t_idx] = float(np.mean(part[lo:hi]))
        else:
            flux[t_idx] = float(np.mean(vals))

    ref_flux = max(float(np.mean(np.abs(flux))), 1e-30)
    flux = np.clip(flux / ref_flux, 1e-6, None)
    mean_flux = float(np.mean(flux))
    if mean_flux > 0:
        flux /= mean_flux
    return flux


def _sliding_temporal_median(y_norm: np.ndarray, win_t: int) -> np.ndarray:
    """Compute cache-friendly 1D temporal sliding median with two-window robust linear padding.

    Extrapolates `pad = win_t // 2` ghost frames at `t = 0` and `t = T - 1` from the medians of
    two overlapping 15-frame boundary blocks (`breakdown = 8` frames each). This avoids:
    1. `mode="wrap"` scan-boundary jumps (coupling `phi_end` onto `phi_start`),
    2. `mode="nearest"` self-reflection of boundary Bragg reflections, and
    3. flat-pad slope bias on steep background rotation ramps.
    Transposing into contiguous `(H*W, T + 2*pad)` order keeps each pixel's time series in L1.
    """
    n_frames, height, width = y_norm.shape
    pad = win_t // 2
    n_pix = height * width

    y_2d = np.ascontiguousarray(y_norm.reshape(n_frames, n_pix).T)
    w = min(15, n_frames)
    step = min(7, max(1, n_frames - w))

    padded = np.empty((n_pix, n_frames + 2 * pad), dtype=np.float64)
    if n_frames > w:
        m_a0 = np.median(y_2d[:, :w], axis=1, keepdims=True)
        m_b0 = np.median(y_2d[:, step : step + w], axis=1, keepdims=True)
        slope0 = (m_b0 - m_a0) / float(step)
        offs0 = np.arange(-pad, 0, dtype=np.float64)[None, :] - 0.5 * (w - 1)
        padded[:, :pad] = m_a0 + slope0 * offs0

        m_a1 = np.median(y_2d[:, -w:], axis=1, keepdims=True)
        m_b1 = np.median(y_2d[:, -(w + step) : -step], axis=1, keepdims=True)
        slope1 = (m_a1 - m_b1) / float(step)
        t_a1 = (n_frames - 1) - 0.5 * (w - 1)
        offs1 = np.arange(n_frames, n_frames + pad, dtype=np.float64)[None, :] - t_a1
        padded[:, pad + n_frames :] = m_a1 + slope1 * offs1
    else:
        med_all = np.median(y_2d, axis=1, keepdims=True)
        padded[:, :pad] = med_all
        padded[:, pad + n_frames :] = med_all

    padded[:, pad : pad + n_frames] = y_2d
    del y_2d

    med_2d = median_filter(padded, size=(1, win_t), mode="nearest")[:, pad : pad + n_frames]
    del padded
    return np.ascontiguousarray(med_2d.T).reshape(n_frames, height, width)


def detect_static_bad_pixels(
    images: np.ndarray,
    pedestal: float = 0.0,
    config: MaskingConfig | None = None,
    *,
    _precomputed_y_sub: np.ndarray | None = None,
    _nonfinite_mask: np.ndarray | None = None,
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

    if _precomputed_y_sub is not None:
        y_sub = _precomputed_y_sub
        nonfinite_mask = _nonfinite_mask
    else:
        y_sub, nonfinite_mask = validate_and_sanitize_stack(images, pedestal=pedestal)

    n_frames, height, width = y_sub.shape
    has_nonfinite = nonfinite_mask is not None and bool(np.any(nonfinite_mask))
    all_nonfinite_2d = (
        np.all(nonfinite_mask, axis=0)
        if (has_nonfinite and nonfinite_mask is not None)
        else np.zeros((height, width), dtype=bool)
    )

    # Per-pixel temporal median, mean, and variance across the scan
    if has_nonfinite and nonfinite_mask is not None:
        y_nan = np.where(nonfinite_mask, np.nan, y_sub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp_median = np.nan_to_num(np.nanmedian(y_nan, axis=0), nan=0.0)
            temp_mean = np.nan_to_num(np.nanmean(y_nan, axis=0), nan=0.0)
            temp_var = np.nan_to_num(np.nanvar(y_nan, axis=0), nan=0.0)
        del y_nan
    else:
        temp_median = np.median(y_sub, axis=0)
        temp_mean = np.mean(y_sub, axis=0)
        temp_var = np.var(y_sub, axis=0)

    # Local 5x5 spatial median and local spatial MAD of the temporal median image
    local_med_5 = median_filter(temp_median, size=5, mode="nearest")
    local_mad_5 = 1.4826 * median_filter(
        np.abs(temp_median - local_med_5), size=7, mode="nearest"
    )
    local_mean_9 = uniform_filter(temp_mean, size=9, mode="nearest")

    # Estimate baseline readout noise floor and scale-covariant quantum unit q_scale
    if n_frames > 1:
        diff_sq = np.square(y_sub[1:] - y_sub[:-1])
        diff_var = 0.5 * np.median(diff_sq, axis=0) / 0.4549364
        del diff_sq
        read_sigma_est = float(np.sqrt(max(np.percentile(diff_var, 5.0), 0.0)))
    else:
        diff_var = temp_var
        read_sigma_est = 0.0

    q_scale = estimate_quantum_scale_adu(y_sub, temp_median, diff_var)

    # 1. Hot / flickering pixels: temporal median or temporal variance sticks out sharply
    # above the immediate 5x5 spatial neighborhood (all thresholds homogeneous in q_scale)
    scale_floor = np.maximum(
        local_mad_5, np.sqrt(np.maximum(q_scale * local_med_5, q_scale**2))
    )
    hot_mask = (temp_median - local_med_5) > (
        config.hot_pixel_spatial_sigma * scale_floor + 10.0 * q_scale
    )

    local_var_med = median_filter(temp_var, size=5, mode="nearest")
    local_var_mad = 1.4826 * median_filter(
        np.abs(temp_var - local_var_med), size=7, mode="nearest"
    )
    hot_var_mask = (
        (temp_var - local_var_med)
        > config.hot_pixel_spatial_sigma * np.maximum(local_var_mad, local_var_med + q_scale**2)
    ) & (temp_median > local_med_5 + 3.0 * scale_floor)
    hot_mask |= hot_var_mask

    # 2. Isolated dead pixels: zero temporal variance or signal collapsed relative to active
    # neighborhood. Require immediate 5x5 median > 0.5 * q_scale and cumulative 9x9 flux
    # T * local_mean > 20 * q_scale so sparse zero-count pixels are never flagged as dead.
    active_neighborhood = ((n_frames * local_mean_9) > 20.0 * q_scale) & (
        local_med_5 > 0.5 * q_scale
    )
    zero_var_tol = (1e-6 * q_scale) ** 2
    raw_zero_tol = 1e-6 * q_scale
    ped_val = float(pedestal)
    pinned_raw_zero = (
        (np.max(y_sub, axis=0) <= -ped_val + raw_zero_tol)
        & (np.min(y_sub, axis=0) >= -ped_val - raw_zero_tol)
        & (~all_nonfinite_2d)
    )

    dead_mask = active_neighborhood & (
        (
            (temp_var <= zero_var_tol)
            & (
                (np.abs(temp_mean) <= max(read_sigma_est, raw_zero_tol))
                | pinned_raw_zero
            )
        )
        | (
            (local_med_5 > max(6.0 * read_sigma_est, 2.0 * q_scale))
            & (temp_mean < config.dead_pixel_rel_threshold * local_med_5)
            & (temp_var < 0.5 * local_var_med)
        )
        | pinned_raw_zero
    )
    cold_mask = (temp_median < -max(4.0 * read_sigma_est, 4.0 * q_scale)) & (
        (local_med_5 - temp_median)
        > (config.hot_pixel_spatial_sigma * scale_floor + 6.0 * q_scale)
    )
    dead_mask |= all_nonfinite_2d | cold_mask
    dead_physical_mask = dead_mask.copy()

    underflow_2d = np.zeros((height, width), dtype=bool)
    if config.trusted_range is not None:
        min_sub = float(config.trusted_range[0]) - ped_val
        max_sub = float(config.trusted_range[1]) - ped_val
        if has_nonfinite and nonfinite_mask is not None:
            finite_3d = ~nonfinite_mask
            underflow_2d = np.any((y_sub < min_sub) & finite_3d, axis=0)
            always_saturated_2d = np.all((y_sub > max_sub) | nonfinite_mask, axis=0)
        else:
            underflow_2d = np.any(y_sub < min_sub, axis=0)
            always_saturated_2d = np.all(y_sub > max_sub, axis=0)
        dead_mask |= underflow_2d
        hot_mask |= always_saturated_2d

    # 3. Shadowed border strips around the perimeter
    non_underflow = ~underflow_2d
    global_ref = (
        float(np.percentile(temp_mean[non_underflow], 60.0))
        if np.any(non_underflow)
        else float(np.percentile(temp_mean, 60.0))
    )
    shadow_thresh = max(
        3.5 * read_sigma_est / np.sqrt(max(n_frames, 1)),
        0.02 * global_ref,
    )
    shadow_mask = np.zeros((height, width), dtype=bool)
    if global_ref > 0.5 * q_scale:
        max_margin = min(height, width) // 4
        for r in range(max_margin):
            if float(np.percentile(temp_mean[r, :], 80.0)) <= shadow_thresh:
                shadow_mask[r, :] = True
            else:
                break
        for r in range(height - 1, height - 1 - max_margin, -1):
            if float(np.percentile(temp_mean[r, :], 80.0)) <= shadow_thresh:
                shadow_mask[r, :] = True
            else:
                break
        for c in range(max_margin):
            if float(np.percentile(temp_mean[:, c], 80.0)) <= shadow_thresh:
                shadow_mask[:, c] = True
            else:
                break
        for c in range(width - 1, width - 1 - max_margin, -1):
            if float(np.percentile(temp_mean[:, c], 80.0)) <= shadow_thresh:
                shadow_mask[:, c] = True
            else:
                break

    # 4. Interior beamstop / arm shadow and soft penumbra dilation.
    # Digital `< min_trusted` hardware sentinels (`underflow_2d`, e.g. -1 module gaps) have crisp
    # pixel boundaries and no optical penumbra, so exclude them from penumbra dilation; evaluate
    # the penumbra profile on `temp_median` so transient Bragg spots cannot punch holes in it.
    interior_eligible = (~shadow_mask) & non_underflow
    low_5_pct = (
        float(np.percentile(temp_median[interior_eligible], 5.0))
        if np.any(interior_eligible)
        else 0.0
    )
    has_high_background_floor = low_5_pct > 2.0 * q_scale
    raw_shadow_candidates = (dead_physical_mask & interior_eligible) | (
        has_high_background_floor
        & (global_ref > 15.0 * q_scale)
        & (temp_median <= config.shadow_rel_threshold * global_ref)
        & interior_eligible
    )
    if np.any(raw_shadow_candidates) and config.shadow_penumbra_dilation_px > 0:
        labeled_sh, n_sh = label(raw_shadow_candidates)
        if n_sh > 0:
            comp_sizes = np.bincount(labeled_sh.ravel())
            comp_sizes[0] = 0
            interior_shadow_core = comp_sizes[labeled_sh] >= 25
        else:
            interior_shadow_core = np.zeros_like(shadow_mask)

        if np.any(interior_shadow_core):
            r_pen = int(config.shadow_penumbra_dilation_px)
            yy, xx = np.mgrid[-r_pen : r_pen + 1, -r_pen : r_pen + 1]
            disk_pen = (yy * yy + xx * xx) <= (r_pen + 0.35) ** 2
            dilated_zone = binary_dilation(interior_shadow_core, structure=disk_pen) & non_underflow
            valid_outside = ~(shadow_mask | interior_shadow_core | underflow_2d)
            ref_level = (
                float(np.percentile(temp_median[valid_outside], 65.0))
                if np.any(valid_outside)
                else global_ref
            )
            med_for_grad = np.where(underflow_2d, ref_level, temp_median)
            gx = uniform_filter(np.abs(np.gradient(med_for_grad, axis=1)), size=3, mode="nearest")
            gy = uniform_filter(np.abs(np.gradient(med_for_grad, axis=0)), size=3, mode="nearest")
            grad_mag = np.hypot(gx, gy)
            penumbra_mask = dilated_zone & (
                (temp_median < config.shadow_penumbra_rel_level * ref_level)
                | (grad_mag > 0.04 * max(ref_level, q_scale))
            )
            shadow_mask |= interior_shadow_core | penumbra_mask

    return hot_mask | dead_mask | shadow_mask


def detect_transient_outliers(
    images: np.ndarray,
    pedestal: float = 0.0,
    static_bad_mask: np.ndarray | None = None,
    config: MaskingConfig | None = None,
    external_dynamic_bad_mask: np.ndarray | None = None,
    *,
    _precomputed_y_sub: np.ndarray | None = None,
    _nonfinite_mask: np.ndarray | None = None,
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
    external_dynamic_bad_mask : np.ndarray, optional
        3D boolean mask of shape (T, H, W) marking known per-frame untrusted/saturated voxels.

    Returns
    -------
    MaskingResult
        Static bad mask, transient 3D outlier mask, combined valid mask, and rough flux scale.
    """
    if config is None:
        config = MaskingConfig()

    if _precomputed_y_sub is not None:
        y_sub = _precomputed_y_sub
        nonfinite_mask = _nonfinite_mask
    else:
        y_sub, nonfinite_mask = validate_and_sanitize_stack(images, pedestal=pedestal)

    n_frames, height, width = y_sub.shape

    if static_bad_mask is None:
        static_bad_mask = detect_static_bad_pixels(
            images,
            pedestal=pedestal,
            config=config,
            _precomputed_y_sub=y_sub,
            _nonfinite_mask=nonfinite_mask,
        )
    else:
        static_bad_mask = np.asarray(static_bad_mask, dtype=bool).copy()
        if static_bad_mask.shape != (height, width):
            raise ValueError(
                f"static_bad_mask shape {static_bad_mask.shape} != {(height, width)}."
            )
        if nonfinite_mask is not None and np.any(nonfinite_mask):
            static_bad_mask |= np.all(nonfinite_mask, axis=0)

    # Assemble any external / non-finite / saturated dynamic per-frame mask
    dyn_invalid: np.ndarray | None = None
    if nonfinite_mask is not None and np.any(nonfinite_mask):
        dyn_invalid = nonfinite_mask & (~static_bad_mask[None, :, :])

    if external_dynamic_bad_mask is not None:
        ext_dyn = np.asarray(external_dynamic_bad_mask, dtype=bool)
        if ext_dyn.shape != y_sub.shape:
            raise ValueError(
                f"external_dynamic_bad_mask shape {ext_dyn.shape} != {y_sub.shape}."
            )
        if np.any(ext_dyn):
            ext_clean = ext_dyn & (~static_bad_mask[None, :, :])
            dyn_invalid = ext_clean if dyn_invalid is None else (dyn_invalid | ext_clean)

    saturated_halo: np.ndarray | None = None
    if config.trusted_range is not None:
        max_sub = float(config.trusted_range[1]) - float(pedestal)
        sat_raw = (y_sub > max_sub) & (~static_bad_mask[None, :, :])
        if np.any(sat_raw):
            saturated_halo = binary_dilation(sat_raw, structure=np.ones((1, 3, 3), dtype=bool))
            dyn_invalid = sat_raw if dyn_invalid is None else (dyn_invalid | sat_raw)

    # If all frames of a pixel were marked dynamically invalid, promote to static_bad_mask
    if dyn_invalid is not None and np.any(dyn_invalid):
        all_dyn_bad = np.all(dyn_invalid, axis=0)
        if np.any(all_dyn_bad):
            static_bad_mask = static_bad_mask | all_dyn_bad
            dyn_invalid = dyn_invalid & (~static_bad_mask[None, :, :])

    rough_flux = estimate_rough_frame_flux(
        y_sub, static_bad_mask=static_bad_mask, dynamic_bad_mask=dyn_invalid
    )
    # Detect shutter-closed / beam-blanked / totally eclipsed frames (< 5% of upper-quartile
    # illuminated flux) so dark frames do not get amplified by 1/rough_flux or corrupt medians.
    q75_flux = float(np.percentile(rough_flux, 75.0))
    if q75_flux > 0.0:
        blanked_frames = rough_flux < (0.05 * q75_flux)
        if np.any(blanked_frames) and int(np.count_nonzero(~blanked_frames)) >= 4:
            blanked_3d = blanked_frames[:, None, None] & (~static_bad_mask[None, :, :])
            dyn_invalid = blanked_3d if dyn_invalid is None else (dyn_invalid | blanked_3d)
            rough_flux = np.where(blanked_frames, q75_flux, rough_flux)
            rough_flux /= float(np.mean(rough_flux))
    y_norm = y_sub / rough_flux[:, None, None]

    # Replace isolated non-finite/dynamic-masked voxels in y_norm with finite per-pixel median
    # so dark dynamic masks or zeroed NaNs cannot pull temporal medians downward.
    if dyn_invalid is not None and np.any(dyn_invalid):
        affected_px = np.any(dyn_invalid, axis=0)
        y_aff = np.where(dyn_invalid[:, affected_px], np.nan, y_norm[:, affected_px])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med_aff = np.nan_to_num(np.nanmedian(y_aff, axis=0), nan=0.0)
        t_idx, px_idx = np.nonzero(dyn_invalid[:, affected_px])
        ay, ax = np.nonzero(affected_px)
        y_norm[t_idx, ay[px_idx], ax[px_idx]] = med_aff[px_idx]

    win_t = min(max(21, int(config.temporal_window) | 1), n_frames)
    if n_frames <= 19:
        temp_med = np.broadcast_to(np.median(y_norm, axis=0, keepdims=True), y_norm.shape).copy()
    else:
        temp_med = _sliding_temporal_median(y_norm, win_t=win_t)

    # Per-pixel temporal MAD of deviations, regularized spatially over 7x7 neighborhoods
    abs_dev = np.abs(y_norm - temp_med)
    mean_mad_2d = 1.4826 * np.median(abs_dev, axis=0)
    del abs_dev

    if np.any(static_bad_mask):
        fill_val = (
            float(np.median(mean_mad_2d[~static_bad_mask]))
            if np.any(~static_bad_mask)
            else 1.0
        )
        mad_2d_clean = np.where(static_bad_mask, fill_val, mean_mad_2d)
    else:
        mad_2d_clean = mean_mad_2d
    spatial_scale_2d = median_filter(mad_2d_clean, size=7, mode="nearest")

    # Estimate scale-covariant single-quantum ADU unit q_scale on y_norm
    temp_med_2d = np.median(y_norm, axis=0)
    if n_frames > 1:
        diff_sq_norm = np.square(y_norm[1:] - y_norm[:-1])
        diff_var_norm = 0.5 * np.median(diff_sq_norm, axis=0) / 0.4549364
        del diff_sq_norm
    else:
        diff_var_norm = np.var(y_norm, axis=0)
    q_scale = estimate_quantum_scale_adu(y_norm, temp_med_2d, diff_var_norm)

    # Scale-covariant Poisson floor prevents false-positive clipping in ultra-low-flux corners
    raw_excess = y_norm - temp_med
    del y_norm
    poisson_floor = np.sqrt(np.maximum(q_scale * temp_med, 1.5 * (q_scale**2)))
    del temp_med
    sigma_eff = np.maximum(spatial_scale_2d[None, :, :], poisson_floor)
    del poisson_floor

    # Positive excess z-score (Bragg spots and cosmic-ray zingers are positive spikes)
    z_pos = raw_excess / np.maximum(sigma_eff, 1e-6 * q_scale)
    del sigma_eff

    # High-confidence single-voxel seeds + locally correlated 3x3 Bragg cluster seeds
    seed_mask = (z_pos > config.transient_seed_sigma) & (raw_excess > 4.0 * q_scale)
    smooth_z = uniform_filter(z_pos, size=(1, 3, 3), mode="nearest")
    cluster_seeds = (
        (smooth_z > config.transient_cluster_sigma)
        & (z_pos > 2.2)
        & (raw_excess > 2.5 * q_scale)
    )
    del z_pos
    seed_mask |= cluster_seeds
    seed_mask &= ~static_bad_mask[None, :, :]

    # Bounded hysteresis propagation into contiguous Bragg spot wings, followed by 3D dilation
    if np.any(seed_mask):
        if config.dilation_t > 0 or config.dilation_xy > 0:
            st_t = 2 * max(0, config.dilation_t) + 1
            st_xy = 2 * max(0, config.dilation_xy) + 1
            struct = np.ones((st_t, st_xy, st_xy), dtype=bool)
            dilated_seeds = binary_dilation(seed_mask, structure=struct)
        else:
            dilated_seeds = seed_mask

        if config.transient_wing_sigma > 0.0:
            max_wing_reach = 2 * (max(1, int(config.dilation_xy)) + 2) + 1
            wing_envelope = binary_dilation(
                seed_mask,
                structure=np.ones((1, max_wing_reach, max_wing_reach), dtype=bool),
            )
            wing_candidates = (
                wing_envelope
                & (smooth_z > config.transient_wing_sigma)
                & (raw_excess > 1.5 * q_scale)
                & (~static_bad_mask[None, :, :])
            )
            wing_propagated = binary_propagation(
                seed_mask,
                mask=wing_candidates | seed_mask,
                structure=np.ones((1, 3, 3), dtype=bool),
            )
            wing_rim = binary_dilation(
                wing_propagated ^ seed_mask,
                structure=np.ones((1, 3, 3), dtype=bool),
            )
            transient_mask = dilated_seeds | wing_propagated | wing_rim
        else:
            transient_mask = dilated_seeds
    else:
        transient_mask = seed_mask

    del smooth_z, raw_excess

    if dyn_invalid is not None and np.any(dyn_invalid):
        transient_mask = transient_mask | dyn_invalid
    if saturated_halo is not None:
        transient_mask = transient_mask | (saturated_halo & (~static_bad_mask[None, :, :]))

    valid_mask = (~static_bad_mask[None, :, :]) & (~transient_mask)
    return MaskingResult(
        static_bad_mask=static_bad_mask,
        transient_outlier_mask=transient_mask,
        valid_mask=valid_mask,
        rough_flux_scale=rough_flux,
    )
