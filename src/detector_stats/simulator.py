"""Synthetic diffraction rotation-scan time-series simulator with cascade, PSF, and pixel noise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import correlate2d


def delta_psf(radius: int = 0) -> np.ndarray:
    """Return a 2D Kronecker delta PSF kernel of shape (2*radius+1, 2*radius+1)."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    kernel[radius, radius] = 1.0
    return kernel


def gaussian_psf(sigma_px: float, radius: int | None = None) -> np.ndarray:
    """Return a normalized 2D isotropic Gaussian PSF kernel."""
    if sigma_px <= 0:
        return delta_psf(radius=radius or 0)
    if radius is None:
        radius = max(1, int(np.ceil(4.0 * sigma_px)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma_px * sigma_px))
    kernel /= np.sum(kernel)
    return kernel


def lorentzian_psf(fwhm_px: float = 1.2, radius: int = 5) -> np.ndarray:
    """Return a normalized 2D Lorentzian/exponential-tail PSF kernel typical of phosphor screens."""
    if fwhm_px <= 0:
        return delta_psf(radius=radius)
    gamma = fwhm_px / 2.0
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r2 = x * x + y * y
    # 2D Poisson/Lorentzian point-spread profile (Holton 2008 / Waterman & Evans 2010)
    kernel = (1.0 + r2 / (gamma * gamma)) ** (-1.5)
    kernel /= np.sum(kernel)
    return kernel


def timepix_cluster_psf(center_weight: float = 0.66, radius: int = 2) -> np.ndarray:
    """Return a 2D charge-sharing PSF kernel representative of 200 keV electrons in a hybrid PAD.

    With default center_weight=0.66 on a 5x5 grid, the zero-lag autocorrelation factor is
    S_2 = sum(K^2) ~= 0.48, matching the ~0.5 spot-finding dispersion damping observed
    in Vypritskaia et al. (2025) for Timepix 3D ED data.
    """
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r = np.hypot(x, y)
    kernel = np.zeros_like(r, dtype=np.float64)
    kernel[r == 0] = center_weight
    ring1 = (r > 0) & (r <= 1.05)
    ring2 = (r > 1.05) & (r <= 1.5)
    rem = 1.0 - center_weight
    kernel[ring1] = 0.78 * rem / np.count_nonzero(ring1)
    kernel[ring2] = 0.22 * rem / np.count_nonzero(ring2)
    kernel /= np.sum(kernel)
    return kernel


@dataclass
class DetectorConfig:
    """Ground-truth detector cascade, point-spread, and readout noise parameters.

    Follows the cascade model of Waterman & Evans (2010), Eq. (10):
        Var(p) = p_bar * G * gamma + N * psi
    with spatial point-spread kernel K(u) satisfying sum_u K(u) = 1.
    """

    name: str = "custom_detector"
    gain: float = 1.0  # G: ADU per absorbed primary quantum
    cascade_factor: float = 1.0  # gamma >= 1.0: excess variance factor from cascade
    pixel_factor: float = 0.0  # psi >= 0.0: readout + quantization variance (ADU^2 / pixel)
    pedestal: float = 0.0  # p_0: constant baseline offset (ADU)
    psf_kernel: np.ndarray = field(default_factory=lambda: delta_psf(0))
    psf_mode: Literal["continuous", "discrete_cluster"] = "continuous"
    quantize: bool = False  # Whether to round final signal to integer ADUs

    def __post_init__(self) -> None:
        if self.gain <= 0:
            raise ValueError(f"gain G must be > 0, got {self.gain}")
        if self.cascade_factor < 1.0 - 1e-12:
            raise ValueError(
                f"cascade_factor gamma must be >= 1.0, got {self.cascade_factor}"
            )
        if self.pixel_factor < 0:
            raise ValueError(
                f"pixel_factor psi must be >= 0, got {self.pixel_factor}"
            )
        self.psf_kernel = np.asarray(self.psf_kernel, dtype=np.float64)
        if (
            self.psf_kernel.ndim != 2
            or self.psf_kernel.shape[0] % 2 == 0
            or self.psf_kernel.shape[1] % 2 == 0
        ):
            raise ValueError(
                f"psf_kernel must be 2D with odd dimensions, got shape {self.psf_kernel.shape}"
            )
        k_sum = float(np.sum(self.psf_kernel))
        if k_sum <= 0:
            raise ValueError("psf_kernel must have positive sum")
        self.psf_kernel = self.psf_kernel / k_sum

    @property
    def g_gamma(self) -> float:
        """Return the effective cascade-gain variance-to-mean slope G * gamma."""
        return float(self.gain * self.cascade_factor)

    @property
    def psf_autocorrelation(self) -> np.ndarray:
        """Return the 2D spatial autocorrelation alpha(u) = (K * K)(u) of the PSF kernel.

        Satisfies sum_u alpha(u) == 1.0.
        """
        return correlate2d(self.psf_kernel, self.psf_kernel, mode="full")

    @property
    def s2_factor(self) -> float:
        """Return the zero-lag PSF autocorrelation factor S_2 = alpha(0) = sum_v K(v)^2 <= 1."""
        return float(np.sum(self.psf_kernel * self.psf_kernel))

    @property
    def apparent_single_pixel_gain(self) -> float:
        """Return the apparent single-pixel variance-to-mean slope in the high-signal limit."""
        if self.psf_mode == "continuous":
            return float(self.gain * self.cascade_factor * self.s2_factor)
        # In discrete_cluster mode where each primary event splits into integer sub-counts
        # of unit weight 1 ADU (so E[M] = G and Var(M) = G^2*(gamma - 1)):
        # Var(p_i) = mu_i * [ G * (gamma - 1/G) * S_2 + 1 ]
        return float(
            self.gain * (self.cascade_factor - 1.0 / self.gain) * self.s2_factor + 1.0
        )

    def theoretical_covariance_kernel(self, mu_adu: float) -> np.ndarray:
        """Return the exact 2D spatial autocovariance C(u; mu) for uniform expected signal mu_adu."""
        alpha = self.psf_autocorrelation
        cy, cx = alpha.shape[0] // 2, alpha.shape[1] // 2
        if self.psf_mode == "continuous":
            cov = float(mu_adu) * self.g_gamma * alpha
        else:
            # Discrete multinomial partitioning of M secondary counts per primary event:
            # Off-diagonal (and continuous part of diagonal) scales with E[M(M-1)] = G^2*(gamma - 1/G)
            # while lag (0,0) gets an extra +mu_adu * 1.0 from multinomial variance K(v)(1 - K(v)).
            cov = (
                float(mu_adu)
                * self.gain
                * (self.cascade_factor - 1.0 / self.gain)
                * alpha
            )
            cov[cy, cx] += float(mu_adu) * 1.0
        cov[cy, cx] += self.pixel_factor
        return cov

    @classmethod
    def ideal_pad(cls) -> DetectorConfig:
        """Ideal photon-counting Pixel Array Detector (G=1, gamma=1, psi=0, delta PSF)."""
        return cls(
            name="ideal_pad",
            gain=1.0,
            cascade_factor=1.0,
            pixel_factor=0.0,
            pedestal=0.0,
            psf_kernel=delta_psf(0),
            psf_mode="continuous",
        )

    @classmethod
    def pt_ccd_waterman_evans(
        cls,
        gain: float = 1.44,
        cascade_factor: float = 1.54,
        pixel_factor: float | None = None,
        pedestal: float = 0.0,
        psf_fwhm_px: float = 1.3,
        psf_radius: int = 4,
    ) -> DetectorConfig:
        """Phosphor-taper-CCD detector parameterized after Waterman & Evans (2010).

        Defaults use G = 1.44 ADU/photon (measured for the Rayonix MX300 at Diamond I24),
        gamma = 1.54 (derived from the Gd2O2S:Tb phosphor cascade in Eq. 13), and
        psi = 4.38 * G^2 ~= 9.08 ADU^2/pixel.
        """
        if pixel_factor is None:
            pixel_factor = 4.38 * (gain**2)
        return cls(
            name="pt_ccd_waterman_evans",
            gain=gain,
            cascade_factor=cascade_factor,
            pixel_factor=pixel_factor,
            pedestal=pedestal,
            psf_kernel=lorentzian_psf(fwhm_px=psf_fwhm_px, radius=psf_radius),
            psf_mode="continuous",
        )

    @classmethod
    def ed_pad_timepix(
        cls,
        gain: float = 2.9,
        cascade_factor: float = 1.35,
        center_weight: float = 0.66,
        psf_mode: Literal["continuous", "discrete_cluster"] = "discrete_cluster",
    ) -> DetectorConfig:
        """Hybrid PAD in 200 keV 3D electron diffraction mode (Vypritskaia et al., 2025).

        Each incident 200 keV electron produces a cluster of counts with mean event
        multiplicity G = 2.9 and cascade factor gamma = 1.35 (due to cluster size variation).
        """
        return cls(
            name="ed_pad_timepix",
            gain=gain,
            cascade_factor=cascade_factor,
            pixel_factor=0.0,
            pedestal=0.0,
            psf_kernel=timepix_cluster_psf(center_weight=center_weight, radius=2),
            psf_mode=psf_mode,
        )


@dataclass
class SceneConfig:
    """Configuration for the spatiotemporal diffraction rotation-scan scene."""

    n_frames: int = 60
    height: int = 128
    width: int = 128
    # Background flux (in absorbed primary quanta per pixel per frame)
    bg_min_quanta: float = 2.0
    bg_max_quanta: float = 60.0
    include_sparse_corner: bool = False
    sparse_corner_quanta: float = 0.008
    # Global incident beam flux variation I_0(t)
    beam_drift_amplitude: float = 0.05  # Slow fractional drift across scan
    beam_topup_period: int = 25  # Frames between synchrotron top-up injections (0 to disable)
    beam_topup_jump: float = 0.04  # Fractional sawtooth jump at top-up
    beam_jitter_sigma: float = 0.01  # Frame-to-frame random fractional jitter
    # Smoothly rotating sample background anisotropy
    bg_rotation_anisotropy: float = 0.15
    # Transient Bragg spots & cosmic ray zingers
    n_bragg_spots: int = 80
    spot_peak_quanta_range: tuple[float, float] = (50.0, 1500.0)
    spot_sigma_xy: float = 1.2
    spot_sigma_t: float = 1.1
    n_zingers: int = 15
    zinger_adu_range: tuple[float, float] = (200.0, 2000.0)
    # Static bad pixels and shadowed border
    n_hot_pixels: int = 10
    n_dead_pixels: int = 10
    shadow_margin_px: int = 4


@dataclass
class SimulationResult:
    """Container for simulated rotation-scan images and ground-truth fields."""

    images: np.ndarray  # (n_frames, height, width) observed ADU time series Y(y, x, t)
    expected_background_adu: np.ndarray  # (n_frames, height, width) true background mu(y, x, t) above pedestal
    incident_flux_scale: np.ndarray  # (n_frames,) true global beam scale factor I_0(t)
    spot_mask: np.ndarray  # (n_frames, height, width) bool mask where Bragg spots / zingers are present
    static_bad_mask: np.ndarray  # (height, width) bool mask where hot/dead/shadowed pixels are located
    detector_config: DetectorConfig
    scene_config: SceneConfig


def _generate_beam_flux_series(
    cfg: SceneConfig, rng: np.random.Generator
) -> np.ndarray:
    """Generate a realistic non-stationary incident beam flux series I_0(t) with mean ~1.0."""
    t = np.arange(cfg.n_frames, dtype=np.float64)
    # Smooth slow drift
    phase = rng.uniform(0, 2.0 * np.pi)
    flux = 1.0 + cfg.beam_drift_amplitude * np.sin(
        2.0 * np.pi * t / max(cfg.n_frames, 1) + phase
    )
    # Synchrotron top-up sawtooth decay and injection jump
    if cfg.beam_topup_period > 0 and cfg.beam_topup_jump > 0:
        cycle_pos = (t % cfg.beam_topup_period) / float(cfg.beam_topup_period)
        flux += cfg.beam_topup_jump * (0.5 - cycle_pos)
    # Frame-to-frame random shutter/source jitter
    if cfg.beam_jitter_sigma > 0:
        flux += rng.normal(0.0, cfg.beam_jitter_sigma, size=cfg.n_frames)
    flux = np.clip(flux, 0.2, None)
    flux /= np.mean(flux)
    return flux


def _generate_background_quanta_field(
    cfg: SceneConfig, beam_flux: np.ndarray
) -> np.ndarray:
    """Generate a smooth 3D spatiotemporal background field Lambda_bg(t, y, x) in quanta."""
    y, x = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float64)
    cy, cx = (cfg.height - 1) / 2.0, (cfg.width - 1) / 2.0
    dy, dx = (y - cy) / max(cy, 1.0), (x - cx) / max(cx, 1.0)
    r = np.hypot(dy, dx)  # 0 at center, ~1.41 at corners
    theta = np.arctan2(dy, dx)

    # Radial profile: central air/inelastic scatter peak + mid-radius solvent ring + outer decay
    radial_profile = (
        0.65 * np.exp(-2.2 * r * r)
        + 0.45 * np.exp(-((r - 0.52) ** 2) / (2.0 * 0.16**2))
    )
    # Normalize radial_profile to [0, 1]
    r_min, r_max = float(np.min(radial_profile)), float(np.max(radial_profile))
    radial_norm = (radial_profile - r_min) / max(r_max - r_min, 1e-12)
    base_2d = cfg.bg_min_quanta + (cfg.bg_max_quanta - cfg.bg_min_quanta) * radial_norm

    if cfg.include_sparse_corner:
        # Make outer corners (r > 1.05) ultra-low flux for testing single-event cluster isolation
        corner_weight = np.clip((r - 0.95) / 0.25, 0.0, 1.0)
        base_2d = (1.0 - corner_weight) * base_2d + corner_weight * cfg.sparse_corner_quanta

    # Evolve smoothly with rotation angle phi(t)
    t_norm = np.linspace(0.0, 1.0, cfg.n_frames, dtype=np.float64)
    phi_t = np.pi * t_norm  # 180 deg rotation scan
    bg_3d = np.empty((cfg.n_frames, cfg.height, cfg.width), dtype=np.float64)
    for t_idx in range(cfg.n_frames):
        anisotropy = 1.0 + cfg.bg_rotation_anisotropy * np.cos(
            2.0 * (theta - phi_t[t_idx])
        ) * (0.3 + 0.5 * r)
        bg_3d[t_idx] = beam_flux[t_idx] * base_2d * anisotropy

    if cfg.include_sparse_corner:
        # Keep the sparse corners strictly near sparse_corner_quanta
        sparse_mask = r > 1.08
        for t_idx in range(cfg.n_frames):
            bg_3d[t_idx, sparse_mask] = beam_flux[t_idx] * cfg.sparse_corner_quanta

    return np.clip(bg_3d, 1e-6, None)


def _sample_cascade_and_psf(
    lambda_quanta: np.ndarray,
    det: DetectorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Poisson primary quanta, stochastic cascade amplification, and PSF redistribution.

    Parameters
    ----------
    lambda_quanta : np.ndarray
        Expected number of absorbed primary quanta per pixel, shape (n_frames, height, width).
    det : DetectorConfig
        Detector response configuration.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    signal_adu : np.ndarray
        Simulated detector signal in ADU (prior to readout noise and pedestal), shape (n_frames, height, width).
    """
    n_abs = rng.poisson(lambda_quanta)

    if det.psf_mode == "continuous":
        # Mechanism A: Each absorbed quantum generates a continuous amplified signal A_k
        # with E[A_k] = G and Var(A_k) = G^2 * (gamma - 1), then spread by K(u).
        excess = det.cascade_factor - 1.0
        if excess <= 1e-12:
            q_source = n_abs.astype(np.float64) * det.gain
        else:
            q_source = np.zeros(n_abs.shape, dtype=np.float64)
            nonzero = n_abs > 0
            shape_param = n_abs[nonzero].astype(np.float64) / excess
            scale_param = det.gain * excess
            q_source[nonzero] = rng.gamma(shape=shape_param, scale=scale_param)

        # Convolve each frame with the 2D PSF kernel (reflect boundary to preserve local mean at edges)
        if det.psf_kernel.shape == (1, 1):
            return q_source
        out = np.empty_like(q_source)
        for t_idx in range(q_source.shape[0]):
            out[t_idx] = convolve(q_source[t_idx], det.psf_kernel, mode="reflect")
        return out

    # Mechanism B ("discrete_cluster"): Each absorbed primary event generates M discrete
    # sub-counts with E[M] = G and Var(M) = G^2 * (gamma - 1), partitioned across the
    # PSF footprint via a Multinomial draw (modeling discrete secondary quanta or Timepix clusters).
    out = np.zeros(n_abs.shape, dtype=np.float64)
    ky, kx = det.psf_kernel.shape
    ry, rx = ky // 2, kx // 2
    probs = det.psf_kernel.ravel()
    excess = det.cascade_factor - 1.0

    # To handle boundaries cleanly with reflection, pad each frame by (ry, rx)
    n_frames, height, width = n_abs.shape
    for t_idx in range(n_frames):
        frame_n = n_abs[t_idx]
        if not np.any(frame_n):
            continue
        # Sample total secondary counts M for each pixel with frame_n > 0
        ys, xs = np.nonzero(frame_n)
        counts_primary = frame_n[ys, xs]
        if excess <= 1e-12:
            m_float = counts_primary.astype(np.float64) * det.gain
        else:
            # Gamma-distributed continuous multiplicity, then rounded/Poisson-sampled so
            # Var(M) matches G^2 * (gamma - 1)
            # Note: If M | Lambda_M ~ Poisson(Lambda_M), Var(M) = E[Lambda_M] + Var(Lambda_M).
            # For Var(M) = n * G^2 * (gamma - 1), we need Var(Lambda_M) = n * (G^2*(gamma - 1) - G).
            target_var_per_event = (det.gain**2) * excess
            if target_var_per_event > det.gain:
                var_lambda = target_var_per_event - det.gain
                shape_l = counts_primary * (det.gain**2) / var_lambda
                scale_l = var_lambda / det.gain
                lambda_m = rng.gamma(shape=shape_l, scale=scale_l)
                m_total = rng.poisson(lambda_m)
            else:
                # Sub-Poisson cluster multiplicity (e.g. fixed cluster size around G):
                # use normal approximation rounded to >= 1
                sd = np.sqrt(counts_primary * target_var_per_event)
                m_total = np.maximum(
                    1,
                    np.rint(
                        rng.normal(counts_primary * det.gain, sd)
                    ).astype(np.int64),
                )
            m_float = m_total

        m_int = np.rint(m_float).astype(np.int64)
        # Partition m_int across the ky*kx kernel offsets via multinomial
        splits = rng.multinomial(m_int, probs)  # shape (n_active, ky*kx)
        padded = np.zeros((height + 2 * ry, width + 2 * rx), dtype=np.float64)
        idx = 0
        for dy in range(ky):
            for dx in range(kx):
                np.add.at(padded, (ys + dy, xs + dx), splits[:, idx])
                idx += 1
        # Fold reflected padding back into the interior so edge intensity is conserved
        if ry > 0:
            padded[ry : 2 * ry, :] += padded[ry - 1 :: -1, :]
            padded[-2 * ry : -ry, :] += padded[-1 : -ry - 1 : -1, :]
        if rx > 0:
            padded[:, rx : 2 * rx] += padded[:, rx - 1 :: -1]
            padded[:, -2 * rx : -rx] += padded[:, -1 : -rx - 1 : -1]
        out[t_idx] = padded[ry : ry + height, rx : rx + width]

    return out


def simulate_rotation_scan(
    detector_config: DetectorConfig,
    scene_config: SceneConfig | None = None,
    seed: int = 42,
) -> SimulationResult:
    """Simulate a complete 3D diffraction rotation scan with known ground-truth statistics.

    Parameters
    ----------
    detector_config : DetectorConfig
        Ground-truth detector response configuration (G, gamma, psi, pedestal, PSF).
    scene_config : SceneConfig, optional
        Configuration for the incident background, beam drift, Bragg spots, and defects.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    SimulationResult
        Simulated image stack Y(t, y, x) and ground-truth masks and background fields.
    """
    if scene_config is None:
        scene_config = SceneConfig()
    rng = np.random.default_rng(seed)

    # 1. Global beam flux time series I_0(t) and smooth background field Lambda_bg(t, y, x)
    beam_flux = _generate_beam_flux_series(scene_config, rng)
    lambda_bg = _generate_background_quanta_field(scene_config, beam_flux)

    # 2. Compute true expected background ADU after PSF convolution (above pedestal)
    expected_bg_adu = np.empty_like(lambda_bg)
    if detector_config.psf_kernel.shape == (1, 1):
        expected_bg_adu[:] = lambda_bg * detector_config.gain
    else:
        for t_idx in range(scene_config.n_frames):
            expected_bg_adu[t_idx] = (
                convolve(lambda_bg[t_idx], detector_config.psf_kernel, mode="reflect")
                * detector_config.gain
            )

    # 3. Inject transient 3D Gaussian Bragg reflections into the incident quanta field
    lambda_total = lambda_bg.copy()
    spot_mask = np.zeros(lambda_bg.shape, dtype=bool)
    margin = max(6, scene_config.shadow_margin_px + 3)

    if scene_config.n_bragg_spots > 0:
        spot_t = rng.uniform(2, scene_config.n_frames - 3, size=scene_config.n_bragg_spots)
        spot_y = rng.uniform(margin, scene_config.height - margin, size=scene_config.n_bragg_spots)
        spot_x = rng.uniform(margin, scene_config.width - margin, size=scene_config.n_bragg_spots)
        spot_peaks = rng.uniform(
            scene_config.spot_peak_quanta_range[0],
            scene_config.spot_peak_quanta_range[1],
            size=scene_config.n_bragg_spots,
        )
        rt = int(np.ceil(3.5 * scene_config.spot_sigma_t))
        rxy = int(np.ceil(3.5 * scene_config.spot_sigma_xy) + detector_config.psf_kernel.shape[0] // 2)

        for i in range(scene_config.n_bragg_spots):
            t0 = int(np.rint(spot_t[i]))
            y0 = int(np.rint(spot_y[i]))
            x0 = int(np.rint(spot_x[i]))
            t_slice = slice(max(0, t0 - rt), min(scene_config.n_frames, t0 + rt + 1))
            y_slice = slice(max(0, y0 - rxy), min(scene_config.height, y0 + rxy + 1))
            x_slice = slice(max(0, x0 - rxy), min(scene_config.width, x0 + rxy + 1))

            tt, yy, xx = np.mgrid[t_slice, y_slice, x_slice]
            profile = spot_peaks[i] * np.exp(
                -((tt - spot_t[i]) ** 2) / (2.0 * scene_config.spot_sigma_t**2)
                - ((yy - spot_y[i]) ** 2 + (xx - spot_x[i]) ** 2)
                / (2.0 * scene_config.spot_sigma_xy**2)
            )
            lambda_total[t_slice, y_slice, x_slice] += profile
            spot_mask[t_slice, y_slice, x_slice] |= profile > 0.15

    # 4. Sample stochastic detection cascade + PSF redistribution
    images = _sample_cascade_and_psf(lambda_total, detector_config, rng)

    # 5. Add pixel readout noise and pedestal offset
    if detector_config.pixel_factor > 0:
        if detector_config.quantize and detector_config.pixel_factor >= 1.0 / 12.0:
            read_var = detector_config.pixel_factor - 1.0 / 12.0
        else:
            read_var = detector_config.pixel_factor
        if read_var > 0:
            images += rng.normal(0.0, np.sqrt(read_var), size=images.shape)

    images += detector_config.pedestal

    # 6. Inject single-frame cosmic ray zingers (direct sensor hits bypassing phosphor PSF)
    if scene_config.n_zingers > 0:
        zt = rng.integers(0, scene_config.n_frames, size=scene_config.n_zingers)
        zy = rng.integers(margin, scene_config.height - margin, size=scene_config.n_zingers)
        zx = rng.integers(margin, scene_config.width - margin, size=scene_config.n_zingers)
        z_amp = rng.uniform(
            scene_config.zinger_adu_range[0],
            scene_config.zinger_adu_range[1],
            size=scene_config.n_zingers,
        )
        for i in range(scene_config.n_zingers):
            images[zt[i], zy[i], zx[i]] += z_amp[i]
            spot_mask[zt[i], zy[i], zx[i]] = True

    # 7. Apply static bad pixels (dead pixels, hot pixels, shadowed border)
    static_bad_mask = np.zeros((scene_config.height, scene_config.width), dtype=bool)
    sm = scene_config.shadow_margin_px
    if sm > 0:
        static_bad_mask[:sm, :] = True
        static_bad_mask[-sm:, :] = True
        static_bad_mask[:, :sm] = True
        static_bad_mask[:, -sm:] = True
        # Shadowed pixels receive zero X-ray/electron flux (only pedestal + readout noise)
        shadow_noise = (
            rng.normal(0.0, np.sqrt(detector_config.pixel_factor), size=images.shape)
            if detector_config.pixel_factor > 0
            else 0.0
        )
        shadow_vals = detector_config.pedestal + shadow_noise
        for t_idx in range(scene_config.n_frames):
            frame = images[t_idx]
            s_frame = shadow_vals[t_idx] if isinstance(shadow_vals, np.ndarray) else shadow_vals
            if isinstance(s_frame, np.ndarray):
                frame[static_bad_mask] = s_frame[static_bad_mask]
            else:
                frame[static_bad_mask] = s_frame

    if scene_config.n_dead_pixels > 0:
        dy_px = rng.integers(sm + 1, scene_config.height - sm - 1, size=scene_config.n_dead_pixels)
        dx_px = rng.integers(sm + 1, scene_config.width - sm - 1, size=scene_config.n_dead_pixels)
        static_bad_mask[dy_px, dx_px] = True
        images[:, dy_px, dx_px] = 0.0

    if scene_config.n_hot_pixels > 0:
        hy_px = rng.integers(sm + 1, scene_config.height - sm - 1, size=scene_config.n_hot_pixels)
        hx_px = rng.integers(sm + 1, scene_config.width - sm - 1, size=scene_config.n_hot_pixels)
        static_bad_mask[hy_px, hx_px] = True
        hot_level = detector_config.pedestal + 500.0 * detector_config.gain
        images[:, hy_px, hx_px] = hot_level + rng.normal(
            0.0, 20.0 * detector_config.gain, size=(scene_config.n_frames, scene_config.n_hot_pixels)
        )

    if detector_config.quantize:
        images = np.rint(images)

    return SimulationResult(
        images=images,
        expected_background_adu=expected_bg_adu,
        incident_flux_scale=beam_flux,
        spot_mask=spot_mask,
        static_bad_mask=static_bad_mask,
        detector_config=detector_config,
        scene_config=scene_config,
    )
