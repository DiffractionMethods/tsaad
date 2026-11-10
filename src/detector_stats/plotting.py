"""Diagnostic plotting utilities for detector response statistics (G, gamma, psi, alpha(u)).

Provides individual plotting functions and an 8-panel summary figure (`plot_detector_stats_report`)
arranged in a 2x4 grid that visualizes:
1. Multi-Lag Photon Transfer Curves (summed spatial covariance V(mu; R) vs expected intensity mu).
2. Cumulative PSF lag-radius convergence S(R) vs lag radius R (with r_PSF plateau).
3. Recovered 2D spatial autocorrelation kernel alpha(u).
4. Incident beam flux scale I_0(t) and frame-wise transient outlier (Bragg/zinger) fraction.
5. Representative single raw frame Y(x, t_0) with single-frame Bragg/zinger mask contours.
6. Scan-wide peak projection max_t Y(x, t) with cumulative transient mask contours.
7. Detrended mean background mu(x) with median-inpainted static bad pixels and Bragg spot footprints.
8. Sparse single-event cluster sum histogram (when G and gamma are separated via cluster
   multiplicity) or standardized detrended residual distribution vs N(0, 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import label, median_filter

from detector_stats.covariance import CovarianceResult
from detector_stats.masking import MaskingResult
from detector_stats.pipeline import DetectorStatsReport
from detector_stats.separation import SeparationResult
from detector_stats.simulator import DetectorConfig
from detector_stats.trend import TrendResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _get_pyplot():
    """Import matplotlib.pyplot lazily so core detector_stats only requires numpy/scipy."""
    import matplotlib.pyplot as plt

    return plt


def _disk_summed_covariance(cov: CovarianceResult, radius: int) -> np.ndarray:
    """Compute disk-summed covariance V(mu_b; R) across intensity bins for radius R."""
    k_size = cov.bin_covariances.shape[1]
    L = k_size // 2
    r = int(np.clip(radius, 0, L))
    yy, xx = np.mgrid[-L : L + 1, -L : L + 1]
    r_inf = np.maximum(np.abs(yy), np.abs(xx))
    r_euc = np.hypot(yy, xx)
    mask = (r_inf <= r) & (r_euc <= r + 0.45)
    n_px = int(np.count_nonzero(mask))
    return (
        np.sum(cov.bin_covariances[:, mask], axis=1)
        - n_px * cov.bg_baseline_per_lag * cov.bin_means
    )


def _split_static_bad_mask(
    static_bad_mask: np.ndarray, max_isolated_area: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Split static_bad_mask into (large_region_mask, isolated_pixel_mask) for visualization."""
    labeled, n_comp = label(static_bad_mask)
    large_mask = np.zeros_like(static_bad_mask, dtype=bool)
    isolated_mask = np.zeros_like(static_bad_mask, dtype=bool)
    for idx in range(1, n_comp + 1):
        comp = labeled == idx
        if np.count_nonzero(comp) > max_isolated_area:
            large_mask |= comp
        else:
            isolated_mask |= comp
    return large_mask, isolated_mask


def plot_multi_lag_ptc(
    cov: CovarianceResult,
    ax: Axes | None = None,
    true_detector: DetectorConfig | None = None,
) -> Axes:
    """Plot Multi-Lag Photon Transfer Curves V_total(mu; R) vs expected signal mu."""
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))

    mu = cov.bin_means
    mu_line = np.linspace(0.0, float(np.max(mu)) * 1.06, 100)

    # Zero-lag (single-pixel) variance C(0; mu)
    v_r0 = _disk_summed_covariance(cov, 0)
    ax.scatter(
        mu,
        v_r0,
        color="#1f77b4",
        s=36,
        alpha=0.85,
        zorder=3,
        label=r"Single-pixel ($R=0$) bins",
    )
    ax.plot(
        mu_line,
        cov.apparent_single_pixel_gain * mu_line + cov.psi,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.6,
        label=(
            rf"Fit $R=0$: slope $\hat{{g}}(0)={cov.apparent_single_pixel_gain:.2f}$"
            rf" ($S_2={cov.s2_factor:.2f}$)"
        ),
    )

    # Intermediate shell R=1 if r_PSF >= 2
    if cov.psf_radius >= 2:
        v_r1 = _disk_summed_covariance(cov, 1)
        s_r1 = float(cov.cumulative_g_gamma_by_radius[1])
        ax.scatter(
            mu,
            v_r1,
            color="#ff7f0e",
            s=28,
            alpha=0.7,
            marker="^",
            zorder=3,
            label=r"Lag disk $R=1$ bins",
        )
        ax.plot(
            mu_line,
            s_r1 * mu_line + cov.psi,
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.4,
            label=rf"Fit $R=1$: slope $\hat{{S}}(1)={s_r1:.2f}$",
        )

    # Converged lag disk R = r_PSF
    if cov.psf_radius > 0:
        v_rpsf = _disk_summed_covariance(cov, cov.psf_radius)
        ax.scatter(
            mu,
            v_rpsf,
            color="#d62728",
            s=42,
            alpha=0.9,
            marker="s",
            zorder=4,
            label=rf"Converged ($R=r_{{\mathrm{{PSF}}}}={cov.psf_radius}$) bins",
        )
    ax.plot(
        mu_line,
        cov.g_gamma * mu_line + cov.psi,
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        zorder=4,
        label=(
            rf"Fit $R={cov.psf_radius}$: $\widehat{{G\gamma}}={cov.g_gamma:.2f}\pm{cov.g_gamma_se:.2f}$,"
            rf" $\hat{{\psi}}={cov.psi:.2f}$"
        ),
    )

    if true_detector is not None:
        ax.plot(
            mu_line,
            true_detector.g_gamma * mu_line + true_detector.pixel_factor,
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.85,
            label=(
                rf"Truth: $G\gamma={true_detector.g_gamma:.2f}$,"
                rf" $\psi={true_detector.pixel_factor:.2f}$"
            ),
        )

    ax.set_xlabel(r"Expected background signal $\mu$ (ADU)", fontsize=9.5)
    ax.set_ylabel(r"Lag-summed covariance $V_{\mathrm{total}}(\mu; R)$ ($\mathrm{ADU}^2$)", fontsize=9.5)
    ax.set_title("Multi-Lag Photon Transfer Curve (PTC)", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
    return ax


def plot_radial_convergence(
    cov: CovarianceResult,
    ax: Axes | None = None,
    true_detector: DetectorConfig | None = None,
) -> Axes:
    """Plot cumulative covariance slope S(R) vs lag radius R = 0..L."""
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))

    radii = np.arange(len(cov.cumulative_g_gamma_by_radius))
    slopes = cov.cumulative_g_gamma_by_radius
    ses = (
        cov.cumulative_g_gamma_se_by_radius
        if cov.cumulative_g_gamma_se_by_radius is not None
        else np.zeros_like(slopes)
    )

    ax.errorbar(
        radii,
        slopes,
        yerr=ses,
        fmt="-o",
        color="#1f77b4",
        ecolor="#6baed6",
        elinewidth=1.5,
        capsize=3.5,
        markersize=5.5,
        linewidth=1.8,
        label=r"Estimated cumulative slope $\hat{S}(R)$",
        zorder=3,
    )

    ax.axvline(
        cov.psf_radius,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=rf"Detected $r_{{\mathrm{{PSF}}}} = {cov.psf_radius}$ px",
    )
    ax.axhline(
        cov.g_gamma,
        color="#d62728",
        linestyle=":",
        linewidth=1.4,
        label=rf"$\widehat{{G\gamma}} = {cov.g_gamma:.3f} \pm {cov.g_gamma_se:.3f}$",
    )
    ax.axhline(
        cov.apparent_single_pixel_gain,
        color="#7f7f7f",
        linestyle="-.",
        linewidth=1.2,
        label=rf"Single-pixel $\hat{{g}}(0) = {cov.apparent_single_pixel_gain:.3f}$",
    )

    if true_detector is not None:
        cov_true_unit = true_detector.theoretical_covariance_kernel(1.0).copy()
        lt = cov_true_unit.shape[0] // 2
        cov_true_unit[lt, lt] -= true_detector.pixel_factor
        yy, xx = np.mgrid[-lt : lt + 1, -lt : lt + 1]
        r_inf_t = np.maximum(np.abs(yy), np.abs(xx))
        r_euc_t = np.hypot(yy, xx)
        true_cum = []
        for r in radii:
            if r >= lt:
                true_cum.append(true_detector.g_gamma)
            else:
                m = (r_inf_t <= r) & (r_euc_t <= r + 0.45)
                true_cum.append(float(np.sum(cov_true_unit[m])))
        ax.plot(
            radii,
            true_cum,
            "--s",
            color="#2ca02c",
            markersize=4.5,
            linewidth=1.5,
            alpha=0.85,
            label=rf"Truth $S(R)$ ($G\gamma={true_detector.g_gamma:.3f}$)",
        )

    ax.set_xticks(radii)
    ax.set_xlabel(r"Integration lag radius $R$ (pixels)", fontsize=9.5)
    ax.set_ylabel(r"Cumulative slope $\hat{S}(R) = \sum_{\|\mathbf{u}\|\leq R} \hat{g}(\mathbf{u})$", fontsize=9.5)
    ax.set_title("Spatial Lag Convergence Profile", fontsize=10.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.6, loc="lower right", framealpha=0.9)
    return ax


def plot_psf_autocorrelation_kernel(
    cov: CovarianceResult,
    ax: Axes | None = None,
    max_display_lag: int | None = None,
) -> Axes:
    """Plot 2D heatmap of the estimated normalized PSF autocorrelation kernel alpha(u)."""
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    alpha = cov.alpha_kernel
    L = alpha.shape[0] // 2
    if max_display_lag is None:
        disp_L = min(L, max(3, cov.psf_radius + 1))
    else:
        disp_L = min(L, max(1, int(max_display_lag)))

    sub = alpha[L - disp_L : L + disp_L + 1, L - disp_L : L + disp_L + 1]
    extent = [-disp_L - 0.5, disp_L + 0.5, -disp_L - 0.5, disp_L + 0.5]

    im = ax.imshow(
        sub,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(np.max(sub)), 0.1),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\hat{\alpha}(\Delta x, \Delta y)$")

    if disp_L <= 4:
        vmax = float(np.max(sub))
        for dy in range(-disp_L, disp_L + 1):
            for dx in range(-disp_L, disp_L + 1):
                val = float(sub[disp_L + dy, disp_L + dx])
                if val >= 0.005:
                    text_color = "black" if val > 0.55 * vmax else "white"
                    ax.text(
                        dx,
                        dy,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7.0 if disp_L >= 3 else 8.0,
                    )

    ax.set_xticks(np.arange(-disp_L, disp_L + 1))
    ax.set_yticks(np.arange(-disp_L, disp_L + 1))
    ax.set_xlabel(r"Spatial lag $\Delta x$ (pixels)", fontsize=9.5)
    ax.set_ylabel(r"Spatial lag $\Delta y$ (pixels)", fontsize=9.5)
    ax.set_title(
        rf"2D PSF Autocorrelation $\hat{{\alpha}}(\mathbf{{u}})$ ($\hat{{S}}_2={cov.s2_factor:.3f}$)",
        fontsize=10.5,
        fontweight="bold",
    )
    return ax


def plot_incident_flux_and_outliers(
    trend: TrendResult,
    masking: MaskingResult,
    ax: Axes | None = None,
    true_flux: np.ndarray | None = None,
) -> Axes:
    """Plot frame-to-frame incident beam flux scale I_0(t) and transient outlier percentage."""
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))

    frames = np.arange(len(trend.incident_flux_scale))
    ax.plot(
        frames,
        trend.incident_flux_scale,
        color="#1f77b4",
        linewidth=1.8,
        label=r"Estimated flux $\hat{I}_0(t)$",
    )
    if true_flux is not None and len(true_flux) == len(frames):
        true_norm = true_flux / max(float(np.mean(true_flux)), 1e-12)
        ax.plot(
            frames,
            true_norm,
            "--",
            color="#2ca02c",
            linewidth=1.5,
            alpha=0.85,
            label=r"True flux $I_0(t)$",
        )

    ax.set_xlabel("Frame index $t$", fontsize=9.5)
    ax.set_ylabel(r"Relative incident beam flux $\hat{I}_0(t)$", color="#1f77b4", fontsize=9.5)
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax.twinx()
    valid_static = ~masking.static_bad_mask
    n_active_px = max(1, int(np.count_nonzero(valid_static)))
    transient_pct = (
        100.0
        * np.sum(masking.transient_outlier_mask[:, valid_static], axis=1)
        / float(n_active_px)
    )
    ax2.fill_between(
        frames,
        0.0,
        transient_pct,
        color="#ff7f0e",
        alpha=0.22,
        step="mid",
    )
    ax2.plot(
        frames,
        transient_pct,
        color="#ff7f0e",
        linewidth=1.2,
        linestyle="-",
        alpha=0.85,
        label="Transient mask % (Bragg/zingers)",
    )
    ax2.set_ylabel("Transient masked pixels (%)", color="#d95f02", fontsize=9.0)
    ax2.tick_params(axis="y", labelcolor="#d95f02")
    ax2.set_ylim(0.0, max(float(np.max(transient_pct)) * 1.35, 2.0))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.6, loc="upper right", framealpha=0.9)
    ax.set_title("Incident Beam Flux & Transient Masking", fontsize=10.5, fontweight="bold")
    return ax


def plot_representative_frame_and_masks(
    report: DetectorStatsReport,
    ax: Axes | None = None,
) -> Axes:
    """Plot a single representative raw frame Y(x, t0) with active frame mask contours."""
    plt = _get_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    masking = report.masking
    t0 = report.representative_frame_idx
    if report.representative_frame_adu is not None:
        frame_t0 = report.representative_frame_adu
    else:
        t0 = min(t0, report.trend.expected_intensity.shape[0] - 1)
        frame_t0 = report.trend.expected_intensity[t0] + report.trend.residuals[t0]

    border_mask, isolated_mask = _split_static_bad_mask(masking.static_bad_mask)
    valid_static = ~masking.static_bad_mask
    vmax = (
        float(np.percentile(frame_t0[valid_static], 99.2))
        if np.any(valid_static)
        else float(np.max(frame_t0))
    )
    masked_t0 = np.ma.masked_where(border_mask, frame_t0)
    cmap = plt.get_cmap("magma").with_extremes(bad="#3a3a4a")
    im = ax.imshow(masked_t0, origin="lower", cmap=cmap, vmin=0.0, vmax=max(vmax, 1.0))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Frame $t={t0}$ signal (ADU)")

    if np.any(masking.transient_outlier_mask[t0]):
        ax.contour(
            masking.transient_outlier_mask[t0],
            levels=[0.5],
            colors=["#00ffff"],
            linewidths=1.3,
        )

    iy, ix = np.nonzero(isolated_mask)
    if iy.size > 0:
        ax.scatter(
            ix,
            iy,
            marker="s",
            s=26,
            facecolors="none",
            edgecolors="#00ff66",
            linewidths=1.3,
            zorder=5,
        )

    handles = [
        Line2D([0], [0], color="#00ffff", lw=1.4, label=f"Bragg/zinger mask ($t={t0}$)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor="#00ff66",
            markeredgewidth=1.3,
            markersize=5.5,
            label="Hot/dead px (static)",
        ),
        Patch(facecolor="#3a3a4a", edgecolor="white", linewidth=0.5, label="Static shadow/corner"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.2, framealpha=0.88)
    ax.set_xlabel("Pixel $x$", fontsize=9.5)
    ax.set_ylabel("Pixel $y$", fontsize=9.5)
    ax.set_title(f"Single Raw Frame (t={t0}) & Frame Mask", fontsize=10.5, fontweight="bold")
    return ax


def plot_scan_peak_projection_and_masks(
    report: DetectorStatsReport,
    ax: Axes | None = None,
) -> Axes:
    """Plot scan-wide peak projection max_t Y(x, t) with cumulative transient mask contours."""
    plt = _get_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    masking = report.masking
    if report.scan_peak_projection_adu is not None:
        peak_proj = report.scan_peak_projection_adu
    else:
        peak_proj = np.max(report.trend.expected_intensity, axis=0)

    border_mask, isolated_mask = _split_static_bad_mask(masking.static_bad_mask)
    valid_static = ~masking.static_bad_mask
    vmax = (
        float(np.percentile(peak_proj[valid_static], 98.5))
        if np.any(valid_static)
        else float(np.max(peak_proj))
    )
    masked_peak = np.ma.masked_where(border_mask, peak_proj)
    cmap = plt.get_cmap("magma").with_extremes(bad="#3a3a4a")
    im = ax.imshow(masked_peak, origin="lower", cmap=cmap, vmin=0.0, vmax=max(vmax, 1.0))
    plt.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label=r"Scan peak $\max_t Y(\mathbf{x}, t)$ (ADU)"
    )

    trans_any = np.any(masking.transient_outlier_mask, axis=0)
    if np.any(trans_any):
        ax.contour(
            trans_any,
            levels=[0.5],
            colors=["#00ffff"],
            linewidths=0.9,
            alpha=0.85,
        )

    iy, ix = np.nonzero(isolated_mask)
    if iy.size > 0:
        ax.scatter(
            ix,
            iy,
            marker="s",
            s=26,
            facecolors="none",
            edgecolors="#00ff66",
            linewidths=1.3,
            zorder=5,
        )

    handles = [
        Line2D([0], [0], color="#00ffff", lw=1.2, label="Cumulative mask (all $t$)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor="#00ff66",
            markeredgewidth=1.3,
            markersize=5.5,
            label="Hot/dead px (static)",
        ),
        Patch(facecolor="#3a3a4a", edgecolor="white", linewidth=0.5, label="Shadowed border"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.2, framealpha=0.88)
    ax.set_xlabel("Pixel $x$", fontsize=9.5)
    ax.set_ylabel("Pixel $y$", fontsize=9.5)
    ax.set_title("Scan Peak Projection & Cumulative Mask", fontsize=10.5, fontweight="bold")
    return ax


def plot_spatial_background_and_masks(
    trend: TrendResult,
    masking: MaskingResult,
    ax: Axes | None = None,
) -> Axes:
    """Plot detrended mean background mu(x) with inpainted static bad pixels and spot footprints."""
    plt = _get_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))

    border_mask, isolated_mask = _split_static_bad_mask(masking.static_bad_mask)
    mean_bg = np.mean(trend.expected_intensity, axis=0).copy()
    if np.any(isolated_mask):
        mean_bg_smooth = median_filter(mean_bg, size=3)
        mean_bg[isolated_mask] = mean_bg_smooth[isolated_mask]

    masked_bg = np.ma.masked_where(border_mask, mean_bg)
    cmap = plt.get_cmap("magma").with_extremes(bad="#3a3a4a")
    im = ax.imshow(masked_bg, origin="lower", cmap=cmap, vmin=0.0)
    plt.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label=r"Detrended mean $\hat{\mu}(\mathbf{x})$ (ADU)"
    )

    trans_frac = 100.0 * np.mean(masking.transient_outlier_mask, axis=0)
    if np.any(trans_frac >= 5.0):
        ax.contour(
            trans_frac,
            levels=[5.0],
            colors=["#00ffff"],
            linewidths=1.0,
            alpha=0.85,
        )

    iy, ix = np.nonzero(isolated_mask)
    if iy.size > 0:
        ax.scatter(
            ix,
            iy,
            marker="s",
            s=26,
            facecolors="none",
            edgecolors="#00ff66",
            linewidths=1.3,
            zorder=5,
        )

    handles = [
        Line2D([0], [0], color="#00ffff", lw=1.2, label=r"Spot footprints ($\geq 5\%$ frames)"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markeredgecolor="#00ff66",
            markeredgewidth=1.3,
            markersize=5.5,
            label="Hot/dead px (inpainted)",
        ),
        Patch(facecolor="#3a3a4a", edgecolor="white", linewidth=0.5, label="Shadowed border"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.2, framealpha=0.88)

    static_pct = 100.0 * float(np.mean(masking.static_bad_mask))
    trans_pct = 100.0 * float(np.mean(masking.transient_outlier_mask))
    ax.set_xlabel("Pixel $x$", fontsize=9.5)
    ax.set_ylabel("Pixel $y$", fontsize=9.5)
    ax.set_title(
        f"Detrended Mean μ(x) (Static {static_pct:.1f}%, Trans {trans_pct:.1f}%)",
        fontsize=10.2,
        fontweight="bold",
    )
    return ax


def plot_separation_or_residuals(
    separation: SeparationResult,
    trend: TrendResult,
    cov: CovarianceResult,
    ax: Axes | None = None,
    true_detector: DetectorConfig | None = None,
) -> Axes:
    """Plot single-event cluster sum histogram (if separated) or standardized residuals."""
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))

    if (
        separation.separation_method == "sparse_cluster_multiplicity"
        and separation.cluster_sums is not None
        and separation.cluster_sums.size >= 10
    ):
        sums = separation.cluster_sums
        p99 = float(np.percentile(sums, 99.5))
        bins = min(35, max(12, int(np.sqrt(sums.size))))
        ax.hist(
            sums[sums <= p99],
            bins=bins,
            density=True,
            color="#4daf4a",
            alpha=0.65,
            edgecolor="black",
            linewidth=0.6,
            label=f"Isolated clusters ($N={separation.n_isolated_clusters}$)",
        )
        ax.axvline(
            separation.gain,
            color="#d62728",
            linestyle="-",
            linewidth=2.0,
            label=rf"Estimated $\hat{{G}} = {separation.gain:.2f} \pm {separation.gain_se:.2f}$ ADU",
        )
        if true_detector is not None:
            ax.axvline(
                true_detector.gain,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.6,
                label=rf"True $G = {true_detector.gain:.2f}$ ADU",
            )
        ax.set_xlabel(r"Isolated cluster integrated signal $S_c$ (ADU)", fontsize=9.5)
        ax.set_ylabel("Probability density", fontsize=9.5)
        ax.set_title(
            rf"Sparse Single-Event Multiplicity ($\hat{{\gamma}}={separation.cascade_factor:.2f}$)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
        return ax

    ped_offset = float(getattr(cov, "estimated_pedestal_offset", 0.0))
    mu_shifted_all = trend.expected_intensity + ped_offset
    valid = trend.valid_mask & (mu_shifted_all >= 1.0)
    if not np.any(valid):
        valid = trend.valid_mask
    res = trend.residuals[valid]
    mu = np.maximum(mu_shifted_all[valid], 1e-3)
    sigma_px = np.sqrt(np.maximum(cov.apparent_single_pixel_gain * mu + cov.psi, 1e-3))
    z = res / sigma_px
    z_clip = z[(z >= -5.0) & (z <= 5.0)]

    ax.hist(
        z_clip,
        bins=60,
        density=True,
        color="#6baed6",
        alpha=0.7,
        edgecolor="none",
        label=rf"Standardized residuals ($\mathrm{{std}}={float(np.std(z_clip)):.2f}$)",
    )
    x_grid = np.linspace(-4.5, 4.5, 200)
    gauss_pdf = np.exp(-0.5 * x_grid * x_grid) / np.sqrt(2.0 * np.pi)
    ax.plot(
        x_grid,
        gauss_pdf,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=r"Standard normal $\mathcal{N}(0, 1)$",
    )
    ax.set_xlabel(
        r"Normalized residual $r(\mathbf{x},t) / \sqrt{\hat{g}(0)\hat{\mu}(\mathbf{x},t) + \hat{\psi}}$",
        fontsize=9.5,
    )
    ax.set_ylabel("Probability density", fontsize=9.5)
    ax.set_title(
        f"Detrended Residual Distribution ({trend.method_name})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
    return ax


def plot_detector_stats_report(
    report: DetectorStatsReport,
    output_path: str | Path | None = None,
    title: str | None = None,
    true_detector: DetectorConfig | None = None,
    true_flux: np.ndarray | None = None,
    dpi: int = 160,
) -> Figure:
    """Create a comprehensive 8-panel (2x4) diagnostic figure from a DetectorStatsReport.

    Parameters
    ----------
    report : DetectorStatsReport
        Output from `estimate_detector_statistics(...)`.
    output_path : str or Path, optional
        If provided, saves the figure to this path (e.g. PNG or PDF).
    title : str, optional
        Custom figure suptitle.
    true_detector : DetectorConfig, optional
        Ground-truth detector configuration (for synthetic benchmark overlays).
    true_flux : np.ndarray, optional
        Ground-truth incident beam flux series I_0(t) (for synthetic benchmark overlays).
    dpi : int, default 160
        Resolution when saving to `output_path`.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed 8-panel matplotlib Figure.
    """
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(21.5, 9.8), constrained_layout=True)

    # Top row: statistical & temporal models
    plot_multi_lag_ptc(report.covariance, ax=axes[0, 0], true_detector=true_detector)
    plot_radial_convergence(report.covariance, ax=axes[0, 1], true_detector=true_detector)
    plot_psf_autocorrelation_kernel(report.covariance, ax=axes[0, 2])
    plot_incident_flux_and_outliers(
        report.trend, report.masking, ax=axes[0, 3], true_flux=true_flux
    )

    # Bottom row: 3 spatial background/mask views + separation/residual distribution
    plot_representative_frame_and_masks(report, ax=axes[1, 0])
    plot_scan_peak_projection_and_masks(report, ax=axes[1, 1])
    plot_spatial_background_and_masks(report.trend, report.masking, ax=axes[1, 2])
    plot_separation_or_residuals(
        report.separation,
        report.trend,
        report.covariance,
        ax=axes[1, 3],
        true_detector=true_detector,
    )

    sep = report.separation
    summary_banner = (
        f"Estimated: G·γ = {report.g_gamma:.3f} ± {report.covariance.g_gamma_se:.3f}  |  "
        f"G = {report.gain:.3f}  |  γ = {report.cascade_factor:.3f} ({sep.separation_method})  |  "
        f"ψ = {report.psi:.2f} ± {report.covariance.psi_se:.2f} ADU²  |  "
        f"g(0) = {report.covariance.apparent_single_pixel_gain:.3f} (S₂ = {report.s2_factor:.3f}, r_PSF = {report.psf_radius} px)"
    )
    if true_detector is not None:
        true_s0 = true_detector.apparent_single_pixel_gain / max(true_detector.g_gamma, 1e-12)
        summary_banner += (
            f"\nGround Truth: G·γ = {true_detector.g_gamma:.3f}  |  "
            f"G = {true_detector.gain:.3f}  |  γ = {true_detector.cascade_factor:.3f}  |  "
            f"ψ = {true_detector.pixel_factor:.2f} ADU²  |  "
            f"g(0) = {true_detector.apparent_single_pixel_gain:.3f} (g(0)/(G·γ) = {true_s0:.3f})"
        )

    full_title = f"{title}\n{summary_banner}" if title else summary_banner
    fig.suptitle(full_title, fontsize=11.5, fontweight="bold")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig
