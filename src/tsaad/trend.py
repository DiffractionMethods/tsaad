"""Spatiotemporal background trend and detrending estimators for rotation-scan time series.

Implements three candidate detrending methods that remove global incident beam variations
I_0(t) and smoothly evolving 2D background scattering without leaking local PSF-correlated
noise into the residuals:
1. HollowStencilKrigingTrend (Global flux scaling + leave-one-frame-out local temporal kernel
   with hollow-annulus spatial fallback for missing pixel series)
2. LowRankSplineTrend (Global flux scaling + orthonormal temporal spline/polynomial basis
   with analytical leverage correction)
3. TemporalDifferenceEstimator (Flux-balanced lag-1 temporal differencing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import convolve

from tsaad._utils import validate_and_sanitize_stack

TrendMethodName = Literal[
    "hollow_stencil_kriging",
    "low_rank_spline",
    "temporal_difference",
]


@dataclass
class TrendResult:
    """Container for detrended residuals and expected intensity fields."""

    residuals: np.ndarray  # (N_t, H, W) float64: leverage-corrected zero-mean residuals r(x, t)
    expected_intensity: np.ndarray  # (N_t, H, W) float64: expected signal mu(x, t) above pedestal
    valid_mask: np.ndarray  # (N_t, H, W) bool: valid observations for covariance accumulation
    incident_flux_scale: np.ndarray  # (T,) float64: estimated global frame flux series I_0(t)
    method_name: str


@runtime_checkable
class TrendEstimator(Protocol):
    """Protocol implemented by spatiotemporal background detrending estimators."""

    def fit_predict(
        self,
        images: np.ndarray,
        valid_mask: np.ndarray,
        pedestal: float = 0.0,
    ) -> TrendResult:
        """Fit background trend on valid voxels and return leverage-corrected residuals."""
        ...


def estimate_incident_flux_scale(
    y_sub: np.ndarray,
    valid_mask: np.ndarray,
    n_iter: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate global frame-to-frame incident flux I_0(t) and static template mu_0(x).

    Uses an alternating missing-data-unbiased Poisson/ratio estimator so that frames with
    different transient Bragg spot masks do not suffer from spatial selection bias.
    """
    n_frames, height, width = y_sub.shape
    n_pix = height * width
    v_float = valid_mask.astype(np.float64)
    v_2d = v_float.reshape(n_frames, n_pix)
    y_masked = np.where(valid_mask, y_sub, 0.0)
    y_abs_ref = max(float(np.mean(np.abs(y_masked))), 1e-30)

    counts_per_frame = np.maximum(np.sum(v_2d, axis=1), 1.0)
    sum_y_t = np.sum(y_masked, axis=(1, 2))
    num_x = np.sum(y_masked, axis=0)
    del y_masked

    flux = sum_y_t / counts_per_frame
    flux /= max(float(np.mean(flux)), 1e-30 * y_abs_ref)
    flux = np.clip(flux, 1e-6, None)
    flux /= float(np.mean(flux))

    template_2d = np.zeros((height, width), dtype=np.float64)
    for _ in range(max(1, n_iter)):
        denom_x = (flux @ v_2d).reshape(height, width)
        template_2d = np.where(denom_x > 0, num_x / np.maximum(denom_x, 1e-30), 0.0)

        pos_template = np.maximum(template_2d, 0.0)
        denom_t = v_2d @ pos_template.ravel()
        flux_new = np.where(
            denom_t > 0,
            sum_y_t / np.maximum(denom_t, 1e-30 * y_abs_ref),
            flux,
        )
        flux = np.clip(flux_new, 1e-6, None)
        flux /= float(np.mean(flux))

    return flux, np.maximum(template_2d, 0.0)


def _build_hollow_annulus_kernel(r_inner: int, r_outer: int, sigma: float) -> np.ndarray:
    """Construct a normalized 2D hollow-annulus spatial kernel with zero weight inside r_inner."""
    y, x = np.mgrid[-r_outer : r_outer + 1, -r_outer : r_outer + 1].astype(np.float64)
    r_inf = np.maximum(np.abs(y), np.abs(x))
    r_euc = np.hypot(y, x)
    mask = (r_inf >= r_inner) & (r_euc <= r_outer + 0.5)
    kernel = np.where(mask, np.exp(-0.5 * (r_euc / sigma) ** 2), 0.0)
    total = float(np.sum(kernel))
    if total > 0:
        kernel /= total
    return kernel


class HollowStencilKrigingTrend:
    """Method A: Global flux scaling + leave-one-frame-out temporal & hollow-stencil kriging.

    Predicts the background at (x, t) using a leave-one-frame-out local temporal kernel
    across unmasked adjacent frames t' != t at the same pixel x (with exact analytical
    variance normalization 1 / sqrt(1 + sum_{t'!=t} w_{t,t'}^2)). Because frames t' != t
    have strictly zero noise correlation with frame t, this preserves the entire 2D spatial
    autocovariance C(u; mu) at all spatial lags u without any (H * H) spatial filter distortion.
    A hollow spatial annulus (r_inner > 2 * r_PSF) provides fallback interpolation for any
    pixel lacking unmasked temporal neighbors.
    """

    def __init__(
        self,
        temporal_sigma_frames: float = 1.6,
        r_inner_px: int = 6,
        r_outer_px: int = 11,
    ) -> None:
        self.temporal_sigma_frames = float(temporal_sigma_frames)
        self.r_inner_px = int(r_inner_px)
        self.r_outer_px = int(r_outer_px)

    def fit_predict(
        self,
        images: np.ndarray,
        valid_mask: np.ndarray,
        pedestal: float = 0.0,
        *,
        _precomputed_y_sub: np.ndarray | None = None,
    ) -> TrendResult:
        """Fit background trend and return leverage-corrected residuals."""
        if _precomputed_y_sub is not None:
            y_sub = _precomputed_y_sub
        else:
            y_sub, _ = validate_and_sanitize_stack(images, pedestal=pedestal)

        n_frames, height, width = y_sub.shape
        n_pix = height * width
        flux, template_2d = estimate_incident_flux_scale(y_sub, valid_mask)

        v_float = valid_mask.astype(np.float64)
        y_norm_masked = np.where(valid_mask, y_sub / flux[:, None, None], 0.0)

        # Leave-one-frame-out banded temporal kernel W[t, s] with W[t, t] == 0
        t_idx = np.arange(n_frames, dtype=np.float64)
        dt = t_idx[:, None] - t_idx[None, :]
        band_limit = max(10.0, float(np.ceil(5.5 * self.temporal_sigma_frames)))
        w_raw = np.where(
            np.abs(dt) <= band_limit,
            np.exp(-0.5 * (dt / self.temporal_sigma_frames) ** 2),
            0.0,
        )
        np.fill_diagonal(w_raw, 0.0)

        # Account for shot-noise variance scaling Var(I_0(t)/I_0(s) * Y_s) ~ (I_0(t)/I_0(s)) * Var(Y_t)
        flux_ratio = flux[:, None] / np.maximum(flux[None, :], 1e-12)
        w_var_kernel = (w_raw**2) * flux_ratio

        v_2d = v_float.reshape(n_frames, n_pix)
        num_t = (w_raw @ y_norm_masked.reshape(n_frames, n_pix)).reshape(n_frames, height, width)
        del y_norm_masked
        den_t = (w_raw @ v_2d).reshape(n_frames, height, width)
        sum_w2 = (w_var_kernel @ v_2d).reshape(n_frames, height, width)

        # Hollow-annulus spatial fallback template for pixels with no valid temporal neighbors
        hollow_kernel = _build_hollow_annulus_kernel(
            self.r_inner_px, self.r_outer_px, sigma=float(self.r_outer_px) * 0.65
        )
        valid_2d = (np.sum(v_float, axis=0) >= 2).astype(np.float64)
        del v_float
        h_num = convolve(template_2d * valid_2d, hollow_kernel, mode="nearest")
        h_den = convolve(valid_2d, hollow_kernel, mode="nearest")
        fallback_2d = np.where(h_den > 0, h_num / np.maximum(h_den, 1e-12), template_2d)

        has_neighbors = den_t > 1e-6
        pred_norm = np.where(
            has_neighbors,
            num_t / np.maximum(den_t, 1e-12),
            fallback_2d[None, :, :],
        )
        del num_t
        np.maximum(pred_norm, 0.0, out=pred_norm)

        # Exact variance inflation from leave-one-frame-out temporal weights:
        # Var(y_t - sum_{s!=t} omega_{t,s} y_s) = sigma^2 * (1 + sum_{s!=t} omega_{t,s}^2)
        omega_sq_sum = np.where(
            has_neighbors,
            sum_w2 / np.maximum(den_t**2, 1e-12),
            0.0,
        )
        del den_t, sum_w2

        expected_adu = flux[:, None, None] * pred_norm
        del pred_norm
        out_valid = valid_mask & has_neighbors
        residuals = np.where(
            out_valid,
            (y_sub - expected_adu) / np.sqrt(1.0 + omega_sq_sum),
            0.0,
        )
        del expected_adu, omega_sq_sum

        binning_intensity = flux[:, None, None] * np.maximum(template_2d[None, :, :], 0.0)

        return TrendResult(
            residuals=residuals,
            expected_intensity=binning_intensity,
            valid_mask=out_valid,
            incident_flux_scale=flux,
            method_name="hollow_stencil_kriging",
        )


class LowRankSplineTrend:
    """Method B: Global flux scaling + low-rank temporal polynomial basis with leverage correction.

    Projects each pixel's flux-normalized time series onto an orthonormal temporal basis of
    dimension K_t << T over unmasked frames, and normalizes each residual r(x, t) by
    1 / sqrt(1 - h(x, t)) where h(x, t) is the exact projection leverage.
    """

    def __init__(self, temporal_rank: int = 8) -> None:
        self.temporal_rank = max(1, int(temporal_rank))

    def fit_predict(
        self,
        images: np.ndarray,
        valid_mask: np.ndarray,
        pedestal: float = 0.0,
        *,
        _precomputed_y_sub: np.ndarray | None = None,
    ) -> TrendResult:
        """Fit low-rank temporal basis per pixel and return leverage-corrected residuals."""
        if _precomputed_y_sub is not None:
            y_sub = _precomputed_y_sub
        else:
            y_sub, _ = validate_and_sanitize_stack(images, pedestal=pedestal)

        n_frames, height, width = y_sub.shape
        flux, template_2d = estimate_incident_flux_scale(y_sub, valid_mask)

        rank = min(self.temporal_rank, max(1, n_frames // 3))
        t_grid = np.linspace(-1.0, 1.0, n_frames, dtype=np.float64)
        raw_basis = np.vstack([np.cos(k * np.arccos(t_grid)) for k in range(rank)]).T
        q_basis, _ = np.linalg.qr(raw_basis)

        y_norm = y_sub / flux[:, None, None]
        n_pix = height * width
        y_flat = y_norm.reshape(n_frames, n_pix)
        v_flat = valid_mask.reshape(n_frames, n_pix)

        pred_norm_flat = np.zeros_like(y_flat)
        leverage_flat = np.zeros_like(y_flat)

        all_valid = np.all(v_flat, axis=0)
        if np.any(all_valid):
            y_all = y_flat[:, all_valid]
            coeffs = q_basis.T @ y_all
            pred_norm_flat[:, all_valid] = q_basis @ coeffs
            h_all = np.sum(q_basis**2, axis=1)[:, None]
            leverage_flat[:, all_valid] = h_all

        partial_indices = np.flatnonzero(~all_valid)
        if partial_indices.size > 0:
            n_vals = np.count_nonzero(v_flat[:, partial_indices], axis=0)

            # Constant-mean fallback for pixels with very few unmasked frames (2 <= n_val <= rank + 1)
            low_dof = (n_vals >= 2) & (n_vals <= rank + 1)
            if np.any(low_dof):
                low_cols = partial_indices[low_dof]
                v_low = v_flat[:, low_cols]
                n_low = n_vals[low_dof].astype(np.float64)
                sums_low = np.sum(np.where(v_low, y_flat[:, low_cols], 0.0), axis=0)
                means_low = sums_low / n_low
                pred_norm_flat[:, low_cols] = means_low[None, :]
                leverage_flat[:, low_cols] = np.where(v_low, (1.0 / n_low)[None, :], 0.0)

            # Batched normal-equation projection for pixels with n_val > rank + 1
            full_cols = partial_indices[n_vals > rank + 1]
            batch_size = 4096
            for start in range(0, full_cols.size, batch_size):
                cols = full_cols[start : start + batch_size]
                v_batch = v_flat[:, cols].T.astype(np.float64)  # (B, T)
                y_batch = y_flat[:, cols].T * v_batch  # (B, T)
                vb = v_batch[:, :, None] * q_basis[None, :, :]  # (B, T, K)
                vbt = vb.transpose(0, 2, 1)  # (B, K, T)
                gram = vbt @ vb  # (B, K, K)

                cond = np.linalg.cond(gram)
                good = np.isfinite(cond) & (cond < 1e10)
                if np.any(good):
                    g_good = gram[good]
                    vb_good = vb[good]
                    vbt_good = vbt[good]
                    rhs_good = (vbt_good @ y_batch[good, :, None])[:, :, 0]  # (B_g, K)
                    c_good = np.linalg.solve(g_good, rhs_good[:, :, None])  # (B_g, K, 1)
                    pred_good = (vb_good @ c_good)[:, :, 0]  # (B_g, T)
                    ginv_vbt = np.linalg.solve(g_good, vbt_good)  # (B_g, K, T)
                    lev_good = np.sum(vbt_good * ginv_vbt, axis=1)  # (B_g, T)

                    good_cols = cols[good]
                    pred_norm_flat[:, good_cols] = pred_good.T
                    leverage_flat[:, good_cols] = lev_good.T

                if not np.all(good):
                    for idx in cols[~good]:
                        v_col = v_flat[:, idx]
                        b_sub = q_basis[v_col, :]
                        q_sub, _ = np.linalg.qr(b_sub)
                        y_sub_col = y_flat[v_col, idx]
                        pred_norm_flat[v_col, idx] = q_sub @ (q_sub.T @ y_sub_col)
                        leverage_flat[v_col, idx] = np.sum(q_sub**2, axis=1)

        del y_norm
        pred_norm = np.maximum(pred_norm_flat.reshape(n_frames, height, width), 0.0)
        leverage = np.clip(leverage_flat.reshape(n_frames, height, width), 0.0, 0.9)
        expected_adu = flux[:, None, None] * pred_norm
        del pred_norm

        dof_scale = np.sqrt(1.0 - leverage)
        del leverage
        residuals = np.where(
            valid_mask & (dof_scale > 0.1),
            (y_sub - expected_adu) / dof_scale,
            0.0,
        )
        del expected_adu, dof_scale

        binning_intensity = flux[:, None, None] * np.maximum(template_2d[None, :, :], 0.0)

        return TrendResult(
            residuals=residuals,
            expected_intensity=binning_intensity,
            valid_mask=valid_mask.copy(),
            incident_flux_scale=flux,
            method_name="low_rank_spline",
        )


class TemporalDifferenceEstimator:
    """Method C: Flux-balanced lag-1 temporal differencing.

    Computes scaled differences between consecutive valid frames (t-1, t):
        d(x, t) = [ I_0(t-1)*Y(x, t) - I_0(t)*Y(x, t-1) ] / sqrt(I_0(t-1)^2 + I_0(t)^2)
    which cancels slowly varying spatial-temporal background trends without any spatial
    interpolation while preserving both the exact covariance slope G*gamma*alpha(u) (with
    respect to the paired effective intensity mu_eff) and the exact zero-lag intercept psi.
    """

    def __init__(self, lag_frames: int = 1) -> None:
        self.lag_frames = max(1, int(lag_frames))

    def fit_predict(
        self,
        images: np.ndarray,
        valid_mask: np.ndarray,
        pedestal: float = 0.0,
        *,
        _precomputed_y_sub: np.ndarray | None = None,
    ) -> TrendResult:
        """Compute flux-balanced temporal differences and paired effective intensities."""
        if _precomputed_y_sub is not None:
            y_sub = _precomputed_y_sub
        else:
            y_sub, _ = validate_and_sanitize_stack(images, pedestal=pedestal)

        lag = self.lag_frames
        if y_sub.shape[0] <= lag:
            raise ValueError(f"Need more than {lag} frames for temporal differencing.")

        flux, template_2d = estimate_incident_flux_scale(y_sub, valid_mask)

        i1 = flux[:-lag, None, None]
        i2 = flux[lag:, None, None]
        y1 = y_sub[:-lag]
        y2 = y_sub[lag:]
        pair_valid = valid_mask[:-lag] & valid_mask[lag:]

        norm_denom = np.sqrt(i1**2 + i2**2)
        diffs = np.where(pair_valid, (i1 * y2 - i2 * y1) / norm_denom, 0.0)

        eff_scale = (i1 * i2 * (i1 + i2)) / (i1**2 + i2**2)
        expected_eff = eff_scale * np.maximum(template_2d[None, :, :], 0.0)

        return TrendResult(
            residuals=diffs,
            expected_intensity=expected_eff,
            valid_mask=pair_valid,
            incident_flux_scale=flux,
            method_name="temporal_difference",
        )
