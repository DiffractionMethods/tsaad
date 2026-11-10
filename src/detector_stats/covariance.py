"""Lagged spatial cross-covariance accumulation and IRLS parameter regression.

Estimates the global product G*gamma, pixel noise factor psi, PSF autocorrelation
convergence radius r_PSF, and normalized 2D spatial correlation kernel alpha(u)
by summing empirical spatial cross-covariances across intensity bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

from detector_stats.trend import TrendResult


@dataclass
class CovarianceResult:
    """Estimated detector response parameters from lagged spatial cross-covariance."""

    g_gamma: float  # Estimated product G * gamma (ADU per absorbed primary quantum * cascade factor)
    g_gamma_se: float  # Standard error on G * gamma
    psi: float  # Estimated pixel noise variance factor psi (ADU^2)
    psi_se: float  # Standard error on psi
    apparent_single_pixel_gain: float  # Zero-lag slope g(0) = G * gamma * S_2 (what local dispersion sees)
    s2_factor: float  # Estimated zero-lag PSF factor S_2 = g(0) / (G * gamma)
    psf_radius: int  # Automatically detected PSF autocorrelation convergence radius r_PSF
    alpha_kernel: np.ndarray  # (2*L+1, 2*L+1) estimated normalized 2D correlation kernel alpha(u)
    lag_slope_map: np.ndarray  # (2*L+1, 2*L+1) estimated slope g(u) at each lag u
    cumulative_g_gamma_by_radius: np.ndarray  # (L+1,) cumulative slope S(R) for R = 0..L
    bin_means: np.ndarray  # (B,) mean expected intensity mu_b in each bin
    bin_counts: np.ndarray  # (B,) number of valid observations in each bin
    bin_covariances: np.ndarray  # (B, 2*L+1, 2*L+1) empirical covariance C(u; mu_b)
    cumulative_g_gamma_se_by_radius: np.ndarray | None = None  # (L+1,) SE of cumulative slope S(R)
    bg_baseline_per_lag: float = 0.0  # subtracted outer-ring baseline slope per lag pixel
    estimated_pedestal_offset: float = 0.0  # off-diagonal estimated additive pedestal error delta_p0 (ADU)
    psf_converged: bool = True  # True if r_PSF < max_lag (converged inside lag window)


def _flat_top_redescending_weights(std_res: np.ndarray, min_inliers: int) -> np.ndarray:
    """Compute C1-smooth Hampel/Tukey flat-top redescending robustness weights in [0, 1].

    Preserves exact unit weight (100% Gauss-Markov efficiency and zero sensitivity to small
    perturbations) for normal inliers `|z| <= 3.5`, tapers smoothly over `3.5 < |z| < 7.0`, and
    completely zeroes out (`w = 0`) extreme high-leverage outliers `|z| >= 7.0`.
    `std_res` must already be dimensionless (standardized by physical sampling SD).
    """
    med_sr = float(np.median(std_res))
    mad = 1.4826 * float(np.median(np.abs(std_res - med_sr)))
    mad_eff = max(mad, 1.0)
    z = np.abs(std_res - med_sr) / mad_eff
    t = np.clip((z - 3.5) / 3.5, 0.0, 1.0)
    rob = np.where(z <= 3.5, 1.0, np.where(z < 7.0, (1.0 - t * t) ** 2, 0.0))
    if np.count_nonzero(rob > 0.05) < min_inliers:
        uh = z / 3.5
        rob = np.where(uh <= 1.0, 1.0, 1.0 / uh)
    return rob


def _irls_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    n_iter: int = 8,
    clamp_nonnegative_intercept: bool = True,
) -> tuple[float, float, float, float, np.ndarray]:
    """Fit y = slope * x + intercept via Iteratively Reweighted Least Squares (IRLS).

    Initialized via pairwise Theil-Sen median slope and reweighted with flat-top redescending
    weights so normal statistical fluctuations keep unit weight while extreme high-leverage
    Bragg-halo or clipped bins are cleanly rejected. Strictly scale-equivariant under
    `(x, y) -> (c * x, c^2 * y)`. Returns (slope, intercept, slope_se, intercept_se, weights).
    """
    y_abs = np.abs(y)
    x_abs = np.abs(x)
    y_ref = max(float(np.median(y_abs)), 1e-6 * float(np.mean(y_abs)), 1e-300)
    x_ref = max(float(np.median(x_abs)), 1e-6 * float(np.mean(x_abs)), 1e-300)

    dx_span = float(np.max(x) - np.min(x))
    dx_mat = x[:, None] - x[None, :]
    dy_mat = y[:, None] - y[None, :]
    valid_pairs = dx_mat > max(0.05 * dx_span, 1e-12 * x_ref)
    if np.any(valid_pairs):
        slope = float(np.median(dy_mat[valid_pairs] / dx_mat[valid_pairs]))
        intercept = float(np.median(y - slope * x))
        if clamp_nonnegative_intercept and intercept < 0.0:
            intercept = 0.0
            x_pos = x > 0.01 * x_ref
            if np.any(x_pos):
                slope = float(np.median(y[x_pos] / x[x_pos]))
    else:
        slope = y_ref / x_ref
        intercept = 0.0

    min_inliers = min(max(4, (len(x) + 1) // 2), len(x))
    rob = np.ones_like(x, dtype=np.float64)
    # Zero-lag sample variance of r^2 has theoretical SD ~ sqrt(2) * E[r^2] / sqrt(N_b)
    scale0 = np.sqrt(2.0) * np.maximum(y_abs, 0.05 * y_ref) / np.sqrt(np.maximum(counts, 1.0))
    w = 1.0 / (scale0**2)
    w /= float(np.mean(w))

    for _ in range(max(1, n_iter)):
        pred_raw = slope * x + intercept
        pred_ref = max(float(np.median(np.abs(pred_raw))), y_ref)
        scale = (
            np.sqrt(2.0)
            * np.maximum(pred_raw, 0.05 * pred_ref)
            / np.sqrt(np.maximum(counts, 1.0))
        )
        std_res = (y - pred_raw) / scale
        rob = _flat_top_redescending_weights(std_res, min_inliers=min_inliers)

        inv_var = 1.0 / (scale**2)
        inv_var /= float(np.mean(inv_var))
        w = rob * inv_var
        w_mean = float(np.mean(w))
        if w_mean <= 1e-30:
            w = inv_var.copy()
        else:
            w /= w_mean

        sw = float(np.sum(w))
        sx = float(np.sum(w * x))
        sy = float(np.sum(w * y))
        sxx = float(np.sum(w * x * x))
        sxy = float(np.sum(w * x * y))
        denom = sw * sxx - sx * sx
        if abs(denom) <= 1e-24 * (sxx * sw + 1e-300):
            break
        slope = (sw * sxy - sx * sy) / denom
        intercept = (sxx * sy - sx * sxy) / denom
        if clamp_nonnegative_intercept and intercept < 0.0:
            intercept = 0.0
            slope = float(np.sum(w * x * y) / max(sxx, 1e-300))

    # Compute standard errors of slope and intercept
    dof = max(1.0, float(np.count_nonzero(rob > 0.05)) - 2.0)
    res = y - (slope * x + intercept)
    s2 = float(np.sum(w * res * res)) / dof
    sw = float(np.sum(w))
    sx = float(np.sum(w * x))
    sxx = float(np.sum(w * x * x))
    denom = max(sw * sxx - sx * sx, 1e-300)
    slope_se = float(np.sqrt(s2 * sw / denom))
    intercept_se = float(np.sqrt(s2 * sxx / denom))
    return float(slope), float(intercept), slope_se, intercept_se, w


def _irls_zero_intercept_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: np.ndarray,
    n_iter: int = 6,
) -> tuple[float, float, np.ndarray]:
    """Fit zero-intercept linear model `y = slope * x` via flat-top redescending GLS/IRLS.

    `sigma_y` is the physical ADU^2 sampling standard deviation of each bin's covariance
    estimate (scaling as `c^2`). Initialized at the precision-weighted median of `y / x`
    so high-intensity bins contaminated by Bragg/powder halos cannot pull slopes upward,
    while normal `<= 3 sigma` bins retain exact unit robustness weight.
    """
    x_ref = max(float(np.median(np.abs(x))), 1e-6 * float(np.mean(np.abs(x))), 1e-300)
    x_pos = np.maximum(x, 1e-12 * x_ref)
    ratios = y / x_pos

    inv_var = 1.0 / np.maximum(sigma_y, 1e-300) ** 2
    base_weights = inv_var / max(float(np.mean(inv_var)), 1e-300)

    v_raw = base_weights * ((x_pos / x_ref) ** 2)
    v_weights = np.maximum(v_raw, 1e-12 * max(float(np.mean(v_raw)), 1e-300))

    order = np.argsort(ratios)
    cum_v = np.cumsum(v_weights[order])
    med_idx = int(np.searchsorted(cum_v, 0.5 * cum_v[-1]))
    med_idx = min(max(med_idx, 0), len(ratios) - 1)
    slope = float(ratios[order[med_idx]])

    min_inliers = min(max(4, (len(x) + 1) // 2), len(x))
    rob = np.ones_like(x_pos)
    w = base_weights.copy()
    x_norm = x / x_ref

    for _ in range(max(1, n_iter)):
        std_res = (y - slope * x) / np.maximum(sigma_y, 1e-300)
        rob = _flat_top_redescending_weights(std_res, min_inliers=min_inliers)
        w = base_weights * rob
        if float(np.sum(w * x_norm * x_norm)) <= 1e-30:
            rob = np.ones_like(x_pos)
            w = base_weights.copy()
            break
        slope = float(np.sum(w * x * y)) / max(float(np.sum(w * x * x)), 1e-300)

    dof = max(1.0, float(np.count_nonzero(rob > 0.05)) - 1.0)
    res = y - slope * x
    denom = max(float(np.sum(w * x * x)), 1e-300)
    s2 = float(np.sum(w * res * res)) / dof
    slope_se = float(np.sqrt(s2 / denom))
    return float(slope), slope_se, w


def estimate_lagged_covariance(
    trend_result: TrendResult,
    max_lag: int = 7,
    n_bins: int = 12,
    fixed_psf_radius: int | None = None,
    auto_calibrate_pedestal: bool = True,
    _allow_auto_expand: bool = True,
) -> CovarianceResult:
    """Accumulate lagged spatial cross-covariances across intensity bins and fit (G*gamma, psi).

    Parameters
    ----------
    trend_result : TrendResult
        Detrended residuals r(x, t), expected intensities mu(x, t), and valid mask.
    max_lag : int, default 7
        Maximum spatial lag radius L in pixels (evaluates lags in [-L, L]^2).
    n_bins : int, default 12
        Number of intensity quantile bins B.
    fixed_psf_radius : int, optional
        If provided, uses this exact PSF integration radius instead of automatic detection.
    auto_calibrate_pedestal : bool, default True
        Whether to shift `bin_means` when off-diagonal ring-1 covariance detects a highly
        significant (>5 sigma) additive pedestal error. Regardless of this flag, any detected
        offset is recorded in `CovarianceResult.estimated_pedestal_offset`.

    Returns
    -------
    CovarianceResult
        Estimated G*gamma, psi, S_2, convergence radius r_PSF, and 2D correlation kernel alpha(u).
    """
    residuals = trend_result.residuals
    expected = trend_result.expected_intensity
    valid = trend_result.valid_mask.copy()
    n_t, height, width = residuals.shape

    # Estimate scale-covariant single-pixel quantum ADU unit q_scale (scales strictly as c^1)
    pos_exp = valid & (expected > 0.0)
    if np.count_nonzero(pos_exp) >= 16:
        med_r2 = float(np.median(residuals[pos_exp] ** 2))
        med_mu = float(np.median(expected[pos_exp]))
        q_scale = max(med_r2 / max(med_mu, 1e-12), 1e-12) if med_r2 > 0.0 else 1.0
    elif np.any(valid):
        q_scale = max(float(np.median(np.abs(residuals[valid]))), 1e-12)
    else:
        q_scale = 1.0

    # Post-detrending outlier safety gate: excise any surviving >6.0*sigma residual spike
    # (computing MAD strictly over valid unmasked frames so masked zeros never bias the scale)
    abs_res_nan = np.where(valid, np.abs(residuals), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res_mad_2d = 1.4826 * np.nanmedian(abs_res_nan, axis=0)
        fallback_mad = (
            float(np.nanmedian(res_mad_2d)) if np.any(np.isfinite(res_mad_2d)) else q_scale
        )
    res_mad_2d = np.nan_to_num(res_mad_2d, nan=fallback_mad)
    res_scale_3d = np.maximum(
        median_filter(res_mad_2d, size=7, mode="nearest")[None, :, :],
        np.sqrt(np.maximum(q_scale * expected, 1.5 * (q_scale**2))),
    )
    spike_mask = valid & (np.abs(residuals) > 6.0 * res_scale_3d) & (np.abs(residuals) > 6.0 * q_scale)
    if np.any(spike_mask):
        spike_dilated = binary_dilation(spike_mask, structure=np.ones((1, 5, 5), dtype=bool))
        valid &= ~spike_dilated

    max_allowed_lag = max(1, min(height, width) // 4 - 1)
    L = min(max(1, int(max_lag)), max_allowed_lag)
    # Build a spatially smoothed copy of expected intensity via normalized 2D convolution
    # strictly for conditioning-free quantile bin assignment, while keeping unblurred
    # expected intensity for unbiased bin means.
    valid_2d = np.any(valid, axis=0).astype(np.float64)
    mean_exp_2d = np.where(
        valid_2d > 0,
        np.sum(np.where(valid, expected, 0.0), axis=0) / np.maximum(np.sum(valid, axis=0), 1),
        0.0,
    )
    smooth_num = gaussian_filter(mean_exp_2d * valid_2d, sigma=1.2, mode="nearest")
    smooth_den = gaussian_filter(valid_2d, sigma=1.2, mode="nearest")
    smooth_2d = np.where(smooth_den > 1e-6, smooth_num / np.maximum(smooth_den, 1e-6), mean_exp_2d)
    exp_tol = 1e-6 * q_scale
    frame_scale = np.where(
        mean_exp_2d[None, :, :] > exp_tol,
        expected / np.maximum(mean_exp_2d[None, :, :], exp_tol),
        1.0,
    )
    bin_field = frame_scale * smooth_2d[None, :, :]

    # Restrict center pixels to interior [L : H-L, L : W-L] so all shifted lags stay in-bounds
    core_valid = valid[:, L : height - L, L : width - L]
    core_exp = expected[:, L : height - L, L : width - L]
    core_bin_field = bin_field[:, L : height - L, L : width - L]
    core_res = residuals[:, L : height - L, L : width - L]

    # Filter out ultra-sparse pixels (mu < 0.5 * q_scale) when forming intensity bins for continuous regression
    active_core = core_valid & (core_bin_field >= 0.5 * q_scale)
    if np.count_nonzero(active_core) < 500:
        active_core = core_valid & (core_bin_field > 0.0)

    valid_mu = core_bin_field[active_core]
    if valid_mu.size == 0:
        raise ValueError("No valid background pixels available for covariance estimation.")

    # Define quantile bin edges across the dimensionless normalized intensity distribution
    norm_bin_field = core_bin_field / max(q_scale, 1e-300)
    norm_valid_mu = norm_bin_field[active_core]
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(norm_valid_mu, quantiles)
    edge_pad = 1e-6 * max(float(edges[-1] - edges[0]), 1.0)
    edges[0] -= edge_pad
    edges[-1] += edge_pad

    # Assign each core voxel to a bin 0..B-1 (-1 if invalid) using smoothed bin_field,
    # while accumulating unblurred core_exp for unbiased bin_means.
    bin_idx = np.digitize(norm_bin_field, edges) - 1
    bin_idx = np.where(active_core & (bin_idx >= 0) & (bin_idx < n_bins), bin_idx, -1)
    flat_bins = bin_idx.ravel()
    valid_flat_mask = flat_bins >= 0
    flat_bins_valid = flat_bins[valid_flat_mask]

    bin_counts = np.bincount(flat_bins_valid, minlength=n_bins).astype(np.int64)
    bin_sum_mu = np.bincount(
        flat_bins_valid, weights=core_exp.ravel()[valid_flat_mask], minlength=n_bins
    )

    # Keep only non-empty bins
    non_empty = bin_counts >= 50
    if np.count_nonzero(non_empty) < 3:
        non_empty = bin_counts > 0

    k_size = 2 * L + 1
    raw_bin_cov = np.zeros((n_bins, k_size, k_size), dtype=np.float64)

    h_c, w_c = height - 2 * L, width - 2 * L
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            # Exploit symmetry C(-dy, -dx) == C(dy, dx): compute upper half and mirror
            if (dy < 0) or (dy == 0 and dx < 0):
                continue
            shifted_valid = valid[:, L + dy : L + dy + h_c, L + dx : L + dx + w_c]
            shifted_res = residuals[:, L + dy : L + dy + h_c, L + dx : L + dx + w_c]

            pair_ok = active_core & shifted_valid
            pair_bins = np.where(pair_ok, bin_idx, -1).ravel()
            ok_flat = pair_bins >= 0
            if not np.any(ok_flat):
                continue

            prod_flat = (core_res * shifted_res).ravel()[ok_flat]
            bins_ok = pair_bins[ok_flat]
            p_counts = np.bincount(bins_ok, minlength=n_bins)
            p_sums = np.bincount(bins_ok, weights=prod_flat, minlength=n_bins)
            cov_b = np.where(p_counts > 0, p_sums / np.maximum(p_counts, 1), 0.0)

            raw_bin_cov[:, L + dy, L + dx] = cov_b
            raw_bin_cov[:, L - dy, L - dx] = cov_b

    # Filter arrays to non-empty bins
    bin_means = bin_sum_mu[non_empty] / np.maximum(bin_counts[non_empty], 1)
    bin_counts_use = bin_counts[non_empty]
    bin_cov_use = raw_bin_cov[non_empty]

    yy, xx = np.mgrid[-L : L + 1, -L : L + 1]
    r_inf_grid = np.maximum(np.abs(yy), np.abs(xx))
    r_euc_grid = np.hypot(yy, xx)

    # Self-calibrate any additive pedestal error delta_p0 using off-diagonal ring 1 (r == 1),
    # where uncorrelated electronic readout noise psi is strictly zero: C_ring1(mu) = S_1 * (mu + delta_p0).
    low_q_res = core_res[core_valid & (core_exp <= np.percentile(valid_mu, 10.0))]
    has_zero_noise_floor = (
        low_q_res.size > 0 and float(np.median(low_q_res**2)) < 0.05 * (q_scale**2)
    )
    estimated_pedestal_offset = 0.0
    if (
        not has_zero_noise_floor
        and float(np.std(bin_means)) > 0.32 * float(np.mean(bin_means))
        and float(np.min(bin_means)) > 1.5 * q_scale
    ):
        ring1_mask = (r_inf_grid == 1) & (r_euc_grid <= 1.45)
        outer_mask = r_inf_grid == L
        n_r1 = int(np.count_nonzero(ring1_mask))
        n_out = max(1, int(np.count_nonzero(outer_mask)))
        c_ring1 = np.sum(bin_cov_use[:, ring1_mask], axis=1) - (n_r1 / n_out) * np.sum(
            bin_cov_use[:, outer_mask], axis=1
        )
        s1, i1, s1_se, i1_se, _ = _irls_linear_fit(
            bin_means, c_ring1, bin_counts_use, clamp_nonnegative_intercept=False
        )
        if (
            s1 > max(5.0 * s1_se, 0.15 * q_scale)
            and abs(i1) > 5.0 * i1_se
            and abs(i1 / s1) > 2.0 * q_scale
        ):
            delta_p0 = float(
                np.clip(i1 / s1, -0.98 * float(np.min(bin_means)), 50.0 * q_scale)
            )
            if float(np.min(bin_means)) + delta_p0 > 0.25 * q_scale:
                estimated_pedestal_offset = delta_p0
                if auto_calibrate_pedestal:
                    bin_means = np.maximum(bin_means + delta_p0, 0.05 * q_scale)

    # 1. Fit zero-lag Photon Transfer Curve: C(0; mu_b) = g(0) * mu_b + psi
    # Electronic readout noise psi is localized at lag (0, 0), so fitting psi at lag (0, 0)
    # achieves (2R+1)^2 times lower variance than fitting a free intercept to summed lags.
    c0_bins = bin_cov_use[:, L, L]
    g0, psi, g0_se, psi_se, _ = _irls_linear_fit(
        bin_means, c0_bins, bin_counts_use, clamp_nonnegative_intercept=True
    )

    c0_abs = np.abs(c0_bins)
    c0_ref = max(float(np.median(c0_abs)), 1e-6 * float(np.mean(c0_abs)), 1e-300)
    pred_c0 = np.maximum(max(g0, 0.0) * bin_means + psi, 0.05 * c0_ref)
    scale_c0 = pred_c0 / np.sqrt(np.maximum(bin_counts_use, 1.0))
    base_weights = 1.0 / (scale_c0**2)
    base_weights /= float(np.mean(base_weights))

    # If the fitted intercept is statistically indistinguishable from 0 (within 2.0*SE)
    # and the lowest-intensity pixels show zero baseline noise floor, set psi = 0.0.
    if psi < 2.0 * psi_se and has_zero_noise_floor:
        psi = 0.0
        g0, g0_se, _ = _irls_zero_intercept_fit(bin_means, c0_bins, np.sqrt(2.0) * scale_c0)
        pred_c0 = np.maximum(max(g0, 0.0) * bin_means, 0.05 * c0_ref)
        scale_c0 = pred_c0 / np.sqrt(np.maximum(bin_counts_use, 1.0))
        base_weights = 1.0 / (scale_c0**2)
        base_weights /= float(np.mean(base_weights))

    # 2. Estimate slope g(u) at every spatial lag u = (dy, dx)
    # At u != (0, 0), theoretical intercept at mu = 0 is zero: C(u; mu_b) = g(u) * mu_b.
    # Use flat-top redescending IRLS per lag so high-mu Bragg/powder halos cannot pull g(u) up.
    lag_slope_map = np.zeros((k_size, k_size), dtype=np.float64)
    lag_slope_se_map = np.zeros((k_size, k_size), dtype=np.float64)
    lag_slope_map[L, L] = g0
    lag_slope_se_map[L, L] = g0_se

    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            if (dy < 0) or (dy == 0 and dx <= 0):
                continue
            cu = bin_cov_use[:, L + dy, L + dx]
            gu, se_u, _ = _irls_zero_intercept_fit(bin_means, cu, scale_c0)
            lag_slope_map[L + dy, L + dx] = gu
            lag_slope_map[L - dy, L - dx] = gu
            lag_slope_se_map[L + dy, L + dx] = se_u
            lag_slope_se_map[L - dy, L - dx] = se_u

    # 3. Evaluate cumulative slope S(R) across lag shells R = 0..L and detect r_PSF
    yy, xx = np.mgrid[-L : L + 1, -L : L + 1]
    r_inf_grid = np.maximum(np.abs(yy), np.abs(xx))
    r_euc_grid = np.hypot(yy, xx)

    # Estimate and subtract any residual long-range background baseline from the outermost ring (r == L)
    outer_ring_mask = r_inf_grid == L
    bg_baseline_per_lag = max(0.0, float(np.median(lag_slope_map[outer_ring_mask])))

    # Enforce Cauchy-Schwarz / PSF peak-monotonicity lower bound on g(0) when narrow-mu ROI
    # slope-intercept collinearity trades lag-0 slope g(0) into readout noise intercept psi.
    cardinal_mask = (r_inf_grid == 1) & (r_euc_grid <= 1.05)
    diag_mask = (np.abs(yy) == 1) & (np.abs(xx) == 1)
    off_diag_mask = r_inf_grid >= 1
    if np.any(cardinal_mask):
        cardinal_vals = lag_slope_map[cardinal_mask] - bg_baseline_per_lag
        mean_g1 = float(np.mean(cardinal_vals))
        se_mean_g1 = 0.5 * float(np.sqrt(np.mean(lag_slope_se_map[cardinal_mask] ** 2)))
        max_off_diag = float(np.max(lag_slope_map[off_diag_mask] - bg_baseline_per_lag))
        if mean_g1 > max(3.0 * se_mean_g1, 0.02 * q_scale):
            mean_g_diag = (
                float(np.mean(lag_slope_map[diag_mask] - bg_baseline_per_lag))
                if np.any(diag_mask)
                else 0.0
            )
            if mean_g_diag > 0.15 * mean_g1:
                step_ratio = float(np.clip(0.95 * (mean_g1 / mean_g_diag), 1.02, 2.5))
            else:
                step_ratio = 1.25
            g0_min_bound = max(max_off_diag, step_ratio * mean_g1)
            if (g0 - bg_baseline_per_lag) < g0_min_bound:
                g0 = g0_min_bound + bg_baseline_per_lag
                lag_slope_map[L, L] = g0
                res_psi = c0_bins - g0 * bin_means
                min_inliers = min(max(4, (len(bin_means) + 1) // 2), len(bin_means))
                sigma_psi = np.sqrt(2.0) * scale_c0
                w_psi = base_weights.copy()
                psi = max(0.0, float(np.sum(w_psi * res_psi)) / max(float(np.sum(w_psi)), 1e-300))
                for _ in range(3):
                    std_r = (res_psi - psi) / np.maximum(sigma_psi, 1e-300)
                    rob_r = _flat_top_redescending_weights(std_r, min_inliers=min_inliers)
                    w_psi = base_weights * rob_r
                    psi = max(
                        0.0, float(np.sum(w_psi * res_psi)) / max(float(np.sum(w_psi)), 1e-300)
                    )
                w_sum = max(float(np.sum(w_psi)), 1e-300)
                dof_psi = max(1.0, float(np.sum(w_psi / np.maximum(base_weights, 1e-300))) - 1.0)
                s2_psi = float(np.sum(w_psi * ((res_psi - psi) ** 2))) / dof_psi
                mean_mu_w = float(np.sum(w_psi * bin_means)) / w_sum
                g0_se = max(g0_se, step_ratio * se_mean_g1)
                lag_slope_se_map[L, L] = g0_se
                psi_se = float(np.sqrt(s2_psi / w_sum + (mean_mu_w * step_ratio * se_mean_g1) ** 2))
                pred_c0 = np.maximum(max(g0, 0.0) * bin_means + psi, 0.05 * c0_ref)
                scale_c0 = pred_c0 / np.sqrt(np.maximum(bin_counts_use, 1.0))
                base_weights = 1.0 / (scale_c0**2)
                base_weights /= float(np.mean(base_weights))

    lag_slope_corr = lag_slope_map - bg_baseline_per_lag

    cumulative_slopes = np.zeros(L + 1, dtype=np.float64)
    cumulative_ses = np.zeros(L + 1, dtype=np.float64)
    shell_slopes = np.zeros(L + 1, dtype=np.float64)
    shell_ses = np.zeros(L + 1, dtype=np.float64)
    shell_mean_per_px = np.zeros(L + 1, dtype=np.float64)

    g0_corr = float(lag_slope_corr[L, L])
    cumulative_slopes[0] = g0_corr
    cumulative_ses[0] = g0_se
    shell_slopes[0] = g0_corr
    shell_ses[0] = g0_se
    shell_mean_per_px[0] = g0_corr

    for r in range(1, L + 1):
        # Use Euclidean disk ||u||_2 <= r + 0.45 inside box [-r, r]^2 to avoid corner noise
        disk_mask = (r_inf_grid <= r) & (r_euc_grid <= r + 0.45)
        prev_disk = (r_inf_grid <= r - 1) & (r_euc_grid <= (r - 1) + 0.45)
        ring_mask = disk_mask & (~prev_disk)
        n_ring_px = max(1, int(np.count_nonzero(ring_mask)))

        # Sum covariance in shell r across bins (with outer-ring baseline subtracted)
        ring_cov_bins = (
            np.sum(bin_cov_use[:, ring_mask], axis=1)
            - n_ring_px * bg_baseline_per_lag * bin_means
        )
        s_ring, se_ring, _ = _irls_zero_intercept_fit(
            bin_means, ring_cov_bins, np.sqrt(float(n_ring_px)) * scale_c0
        )

        shell_slopes[r] = s_ring
        shell_ses[r] = se_ring
        shell_mean_per_px[r] = s_ring / float(n_ring_px)
        cumulative_slopes[r] = cumulative_slopes[r - 1] + s_ring

        # Standard error on cumulative disk slope S(r)
        n_disk_px = max(1, int(np.count_nonzero(disk_mask)) )
        disk_cov_minus_psi = (
            np.sum(bin_cov_use[:, disk_mask], axis=1)
            - psi
            - n_disk_px * bg_baseline_per_lag * bin_means
        )
        _, se_disk, _ = _irls_zero_intercept_fit(
            bin_means, disk_cov_minus_psi, np.sqrt(float(n_disk_px)) * scale_c0
        )
        cumulative_ses[r] = se_disk

    if fixed_psf_radius is not None:
        r_psf = int(np.clip(fixed_psf_radius, 0, L))
    else:
        # Automatic convergence detection: increment r_PSF while shell r adds significant
        # positive correlation above noise (> 2.0 * SE, > 0.75% of cumulative slope, and
        # per-pixel mean correlation in shell r is at least 0.25% of g(0))
        r_psf = 0
        for r in range(1, L + 1):
            sig_thresh = max(2.0 * shell_ses[r], 0.0075 * max(cumulative_slopes[r - 1], g0_corr))
            per_px_ok = shell_mean_per_px[r] > 0.0025 * g0_corr
            if shell_slopes[r] > sig_thresh and per_px_ok:
                r_psf = r
            else:
                break

        # Guard band: if r_PSF reached the outer ceiling L (where bg_baseline_per_lag is measured),
        # automatically expand max_lag by up to +4 (if image geometry allows) so the baseline ring sits
        # outside the active PSF support.
        if _allow_auto_expand and r_psf == L and L < min(11, max_allowed_lag):
            expanded_lag = min(L + 2, 11, max_allowed_lag)
            if expanded_lag > L:
                return estimate_lagged_covariance(
                    trend_result=trend_result,
                    max_lag=expanded_lag,
                    n_bins=n_bins,
                    fixed_psf_radius=None,
                    auto_calibrate_pedestal=auto_calibrate_pedestal,
                    _allow_auto_expand=(expanded_lag < min(L + 4, 11, max_allowed_lag)),
                )

    psf_converged = bool(r_psf < L) or (fixed_psf_radius is not None)

    # Final estimate of G*gamma and its standard error over the converged support disk
    support_mask = (r_inf_grid <= r_psf) & (r_euc_grid <= r_psf + 0.45)
    g_gamma = float(cumulative_slopes[r_psf])
    g_gamma_se = float(cumulative_ses[r_psf])

    # Construct normalized 2D spatial autocorrelation kernel alpha(u)
    alpha_kernel = np.zeros_like(lag_slope_corr)
    if g_gamma > 0:
        alpha_support = np.where(support_mask, np.maximum(lag_slope_corr, 0.0), 0.0)
        alpha_sum = float(np.sum(alpha_support))
        if alpha_sum > 0:
            alpha_kernel = alpha_support / alpha_sum
        else:
            alpha_kernel[L, L] = 1.0
    else:
        alpha_kernel[L, L] = 1.0

    s2_factor = float(g0_corr / g_gamma) if g_gamma > 0 else 1.0

    return CovarianceResult(
        g_gamma=g_gamma,
        g_gamma_se=g_gamma_se,
        psi=psi,
        psi_se=psi_se,
        apparent_single_pixel_gain=g0_corr,
        s2_factor=s2_factor,
        psf_radius=r_psf,
        alpha_kernel=alpha_kernel,
        lag_slope_map=lag_slope_corr,
        cumulative_g_gamma_by_radius=cumulative_slopes,
        bin_means=bin_means,
        bin_counts=bin_counts_use,
        bin_covariances=bin_cov_use,
        cumulative_g_gamma_se_by_radius=cumulative_ses,
        bg_baseline_per_lag=bg_baseline_per_lag,
        estimated_pedestal_offset=estimated_pedestal_offset,
        psf_converged=psf_converged,
    )
