"""Internal shared numerical, geometric, and validation helpers for tsaad."""

from __future__ import annotations

import numpy as np


def validate_and_sanitize_stack(
    images: np.ndarray,
    pedestal: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 3D `(T, H, W)` image stack dimensions, subtract `pedestal`, and sanitize `NaN`/`Inf`.

    Parameters
    ----------
    images : np.ndarray
        Input 3D array of shape `(T, H, W)`.
    pedestal : float, default 0.0
        Baseline pedestal in ADU to subtract.

    Returns
    -------
    y_sub : np.ndarray
        Contiguous `float64` array `images - pedestal` with any non-finite voxels replaced by `0.0`.
    nonfinite_mask : np.ndarray
        Boolean 3D mask `(T, H, W)` where `True` marks input voxels that were `NaN` or `Inf`.
    """
    arr = np.asarray(images)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D image stack of shape (T, H, W), got {arr.ndim}D shape {arr.shape}."
        )
    n_frames, height, width = arr.shape
    if n_frames < 2:
        raise ValueError(f"Need at least 2 frames along axis 0, got shape {arr.shape}.")
    if height < 8 or width < 8:
        raise ValueError(
            f"Spatial dimensions (H, W) must be at least (8, 8), got shape {arr.shape}."
        )
    if not np.isfinite(pedestal):
        raise ValueError(f"pedestal must be finite, got {pedestal!r}.")

    y_sub = np.array(arr, dtype=np.float64, copy=True)
    nonfinite_mask = ~np.isfinite(y_sub)
    if np.any(nonfinite_mask):
        y_sub[nonfinite_mask] = 0.0
        if pedestal != 0.0:
            y_sub[~nonfinite_mask] -= float(pedestal)
    elif pedestal != 0.0:
        y_sub -= float(pedestal)

    return y_sub, nonfinite_mask


def make_lag_disk_mask(max_lag: int, radius: int, euc_pad: float = 0.45) -> np.ndarray:
    """Construct a `(2*L+1, 2*L+1)` Euclidean-bounded Chebyshev lag disk mask for radius `R`."""
    L = max(0, int(max_lag))
    r = int(np.clip(radius, 0, L))
    coords = np.arange(-L, L + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    r_inf = np.maximum(np.abs(yy), np.abs(xx))
    r_euc = np.hypot(yy, xx)
    return (r_inf <= r) & (r_euc <= r + euc_pad)


def estimate_quantum_scale_adu(
    y_sub: np.ndarray,
    temp_median: np.ndarray,
    diff_var: np.ndarray,
) -> float:
    """Estimate a scale-covariant single-pixel quantum ADU unit `q_scale` (homogeneous of degree 1).

    Under any multiplicative rescaling `y_sub -> c * y_sub` (`c > 0`), `diff_var` scales as `c^2`
    and `temp_median` scales as `c`, so `q_scale` scales strictly as `c^1`. For standard unit-gain
    detectors, `q_scale ~= 1.0 ADU`.
    """
    pos_thresh = float(np.percentile(temp_median, 20.0))
    pos_mask = (temp_median > max(pos_thresh, 0.0)) & np.isfinite(diff_var)
    if np.count_nonzero(pos_mask) >= 16:
        med_mu = float(np.median(temp_median[pos_mask]))
        med_var = float(np.median(diff_var[pos_mask]))
        if med_mu > 0.0 and med_var > 0.0:
            return max(med_var / med_mu, 1e-12)

    pos_all = temp_median > 0.0
    if np.any(pos_all) and np.any(diff_var > 0.0):
        med_mu = float(np.median(temp_median[pos_all]))
        med_var = float(np.median(diff_var[pos_all]))
        if med_mu > 0.0 and med_var > 0.0:
            return max(med_var / med_mu, 1e-12)

    fallback = float(np.sqrt(max(float(np.median(diff_var)), 0.0)))
    if fallback > 0.0:
        return fallback
    abs_y = np.abs(y_sub)
    pos_vals = abs_y[abs_y > 0.0]
    if pos_vals.size > 0:
        return max(float(np.percentile(pos_vals, 25.0)), 1e-12)
    return 1e-12


def is_effectively_integer_adu(
    values: np.ndarray,
    tol: float = 1e-6,
    max_samples: int = 250_000,
) -> bool:
    """Check whether observed ADU values are integer-quantized within `tol` at the 99th percentile."""
    if values.size == 0:
        return False
    flat = values.ravel()
    if flat.size > max_samples:
        stride = max(1, flat.size // max_samples)
        sample = flat[::stride]
    else:
        sample = flat
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    return bool(np.percentile(np.abs(finite - np.rint(finite)), 99.0) < tol)
