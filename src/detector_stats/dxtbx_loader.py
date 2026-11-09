"""Bridge between dxtbx ExperimentList / ImageSet objects and detector_stats.

Provides memory-efficient extraction of 3D rotation-scan ROI tiles (T, H, W) from
any dxtbx ExperimentList, ImageSet, `.expt` file, or image sequence, along with
panel metadata, static untrusted-pixel masks, and direct comparison against
DIALS's single-frame spatial dispersion estimator (`DispersionThresholdDebug` /
`dials.estimate_gain`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from detector_stats.masking import MaskingConfig, detect_static_bad_pixels
from detector_stats.pipeline import DetectorStatsReport, estimate_detector_statistics


@dataclass
class DxtbxTileData:
    """Extracted 3D rotation-scan ROI tile and metadata from a dxtbx ImageSet."""

    images: np.ndarray  # (T, H, W) float64 raw pixel values in ADU
    static_untrusted_mask: np.ndarray  # (H, W) bool: True where dxtbx mask is untrusted in any frame
    frame_indices: np.ndarray  # (T,) int64: 0-based image indices inside the ImageSet
    panel_id: int  # 0-based panel index
    panel_name: str  # dxtbx Panel name
    detector_gain: float  # nominal gain stored in dxtbx Panel
    detector_pedestal: float  # nominal pedestal stored in dxtbx Panel
    trusted_range: tuple[float, float]  # dxtbx Panel (min_trusted, max_trusted)
    bbox_yx: tuple[int, int, int, int]  # (y0, y1, x0, x1) pixel bounding box on the panel
    panel_shape_yx: tuple[int, int]  # (full_height, full_width) of the panel
    dials_dispersion_gains: np.ndarray | None = None  # (N_disp,) per-frame dispersion medians
    sensor_material: str = ""  # e.g. 'Si', 'CdTe', 'Phosphor', etc. from dxtbx Panel
    sensor_thickness_mm: float = 0.0  # sensor thickness in mm from dxtbx Panel
    pixel_size_mm: tuple[float, float] = (0.0, 0.0)  # (px_x, px_y) in mm from dxtbx Panel

    @property
    def dials_dispersion_gain_mean(self) -> float | None:
        if self.dials_dispersion_gains is None or self.dials_dispersion_gains.size == 0:
            return None
        return float(np.mean(self.dials_dispersion_gains))

    @property
    def dials_dispersion_gain_std(self) -> float | None:
        if self.dials_dispersion_gains is None or self.dials_dispersion_gains.size <= 1:
            return 0.0 if self.dials_dispersion_gains is not None else None
        return float(np.std(self.dials_dispersion_gains, ddof=1))


def load_imageset(source: object):
    """Resolve a path, ExperimentList, Experiment, or ImageSet into a dxtbx ImageSet.

    Parameters
    ----------
    source : str, Path, Sequence[str | Path], ImageSet, Experiment, or ExperimentList
        Path to an `.expt` file, one or more image files, or an already-loaded
        `dxtbx` object.

    Returns
    -------
    dxtbx.imageset.ImageSet
    """
    from dxtbx.imageset import ImageSet
    from dxtbx.model.experiment_list import Experiment, ExperimentList, ExperimentListFactory

    if isinstance(source, ImageSet):
        return source
    if isinstance(source, Experiment):
        return source.imageset
    if isinstance(source, ExperimentList):
        imagesets = source.imagesets()
        if not imagesets:
            raise ValueError("ExperimentList contains no imagesets.")
        return imagesets[0]

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() in {".expt", ".json"}:
            expts = ExperimentListFactory.from_json_file(str(p), check_format=True)
        else:
            expts = ExperimentListFactory.from_filenames([str(p)])
        imagesets = expts.imagesets()
        if not imagesets:
            raise ValueError(f"No imagesets found in {source!r}.")
        return imagesets[0]

    if isinstance(source, Sequence):
        paths = [str(Path(item)) for item in source]
        expts = ExperimentListFactory.from_filenames(paths)
        imagesets = expts.imagesets()
        if not imagesets:
            raise ValueError(f"No imagesets found in {source!r}.")
        return imagesets[0]

    raise TypeError(f"Unsupported dxtbx source type: {type(source)!r}")


def select_default_roi_bbox(
    panel_height: int,
    panel_width: int,
    tile_size: tuple[int, int] | None = (256, 256),
    beam_center_yx: tuple[float, float] | None = None,
) -> tuple[int, int, int, int]:
    """Select a default 2D ROI bounding box `(y0, y1, x0, x1)` on a detector panel.

    If the panel is smaller than or equal to `tile_size` (or `tile_size is None`),
    returns the full panel `(0, panel_height, 0, panel_width)`. Otherwise selects a
    tile spanning from near the beam center (high-flux solvent ring) toward a detector
    corner (lower-flux background) so that a wide dynamic range of intensities is sampled.
    """
    if tile_size is None:
        return (0, panel_height, 0, panel_width)

    th = min(panel_height, max(32, int(tile_size[0])))
    tw = min(panel_width, max(32, int(tile_size[1])))
    if th == panel_height and tw == panel_width:
        return (0, panel_height, 0, panel_width)

    if beam_center_yx is not None:
        by, bx = beam_center_yx
        cy = float(np.clip(by, 0.0, panel_height - 1.0))
        cx = float(np.clip(bx, 0.0, panel_width - 1.0))
    else:
        cy, cx = 0.5 * panel_height, 0.5 * panel_width

    # Offset the tile from the beam center into the largest quadrant so the tile starts
    # outside the central beamstop shadow (~0.06 * panel_dim from the beam center) and
    # extends radially across the solvent ring toward the outer low-flux region.
    dir_y = 1.0 if (panel_height - cy) >= cy else -1.0
    dir_x = 1.0 if (panel_width - cx) >= cx else -1.0
    clearance_y = max(32.0, 0.06 * panel_height)
    clearance_x = max(32.0, 0.06 * panel_width)
    target_cy = cy + dir_y * (clearance_y + 0.5 * th)
    target_cx = cx + dir_x * (clearance_x + 0.5 * tw)

    y0 = int(np.clip(np.rint(target_cy - 0.5 * th), 0, panel_height - th))
    x0 = int(np.clip(np.rint(target_cx - 0.5 * tw), 0, panel_width - tw))
    return (y0, y0 + th, x0, x0 + tw)


def _box_sum_2d(arr: np.ndarray, ky: int, kx: int) -> np.ndarray:
    """Fast 2D summed-area-table box sum over `[r - ky, r + ky] x [c - kx, c + kx]`."""
    h, w = arr.shape
    sat = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    rows = np.arange(h)
    cols = np.arange(w)
    r0 = np.maximum(0, rows - ky)[:, None]
    r1 = np.minimum(h, rows + ky + 1)[:, None]
    c0 = np.maximum(0, cols - kx)[None, :]
    c1 = np.minimum(w, cols + kx + 1)[None, :]
    return sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]


def compute_dials_dispersion_gain_on_tile(
    images: np.ndarray,
    untrusted_mask: np.ndarray,
    kernel_size: tuple[int, int] = (10, 10),
    max_frames: int = 10,
) -> np.ndarray:
    """Compute the single-frame Kabsch local index-of-dispersion gain on up to `max_frames`.

    Replicates the exact summed-area-table `IndexOfDispersionFilterMasked` and IQR-filtered
    median calculation used by `dials.estimate_gain` in pure NumPy (without requiring
    `dials` to be installed).
    """
    n_frames, height, width = images.shape
    ky, kx = max(1, int(kernel_size[0])), max(1, int(kernel_size[1]))
    min_count = (2 * ky + 1) * (2 * kx + 1)

    step = max(1, n_frames // max(1, int(max_frames)))
    sample_indices = range(0, n_frames, step)[: max(1, int(max_frames))]

    valid_base = ~np.asarray(untrusted_mask, dtype=bool)
    big_limit = float(1 << 24)

    gains: list[float] = []
    for idx in sample_indices:
        frame = np.asarray(images[idx], dtype=np.float64)
        mask = valid_base & (frame < big_limit)
        m = _box_sum_2d(mask.astype(np.float64), ky, kx)
        x = _box_sum_2d(np.where(mask, frame, 0.0), ky, kx)
        y = _box_sum_2d(np.where(mask, frame * frame, 0.0), ky, kx)

        disp_map = np.ones((height, width), dtype=np.float64)
        valid_px = mask & (m >= min_count) & (x > 0.0)
        disp_map[valid_px] = (m[valid_px] * y[valid_px] - x[valid_px] ** 2) / (
            (m[valid_px] - 1.0) * x[valid_px]
        )

        disp = disp_map.ravel()
        disp = disp[np.isfinite(disp) & (disp > 0.0)]
        if disp.size < 10:
            continue
        q1, q3 = np.percentile(disp, [25.0, 75.0])
        iqr = q3 - q1
        if iqr <= 0.0:
            gains.append(float(np.median(disp)))
            continue
        inliers = disp[(disp > q1 - 1.5 * iqr) & (disp < q3 + 1.5 * iqr)]
        if inliers.size > 0:
            gains.append(float(np.median(inliers)))

    return np.asarray(gains, dtype=np.float64)


def extract_dxtbx_tile(
    source: object,
    panel_id: int = 0,
    bbox_yx: tuple[int, int, int, int] | None = None,
    tile_size: tuple[int, int] | None = (256, 256),
    frame_range: tuple[int, int] | None = None,
    max_frames: int = 80,
    frame_stride: int = 1,
    compute_dials_gain: bool = True,
) -> DxtbxTileData:
    """Extract a 3D rotation-scan ROI tile `(T, H, W)` from a dxtbx source.

    Parameters
    ----------
    source : str, Path, ImageSet, Experiment, or ExperimentList
        Input dxtbx dataset or `.expt` / image file path(s).
    panel_id : int, default 0
        Detector panel index to extract from.
    bbox_yx : tuple of (y0, y1, x0, x1), optional
        Explicit pixel bounding box on the panel. If None, chosen automatically via
        `select_default_roi_bbox` using `tile_size`.
    tile_size : tuple of (height, width) or None, default (256, 256)
        Default tile dimensions when `bbox_yx` is None. Pass `None` to load the full panel.
    frame_range : tuple of (start_frame, stop_frame), optional
        0-based half-open frame interval `[start_frame, stop_frame)` within the ImageSet.
    max_frames : int, default 80
        Maximum number of frames to extract.
    frame_stride : int, default 1
        Frame step between consecutive extracted frames (typically 1 so adjacent frames
        remain temporally contiguous for Bragg spot masking and trend estimation).
    compute_dials_gain : bool, default True
        Whether to also evaluate DIALS's `DispersionThresholdDebug` on the extracted tile.

    Returns
    -------
    DxtbxTileData
        Extracted 3D image stack, static untrusted mask, panel metadata, and DIALS gain comparison.
    """
    imageset = load_imageset(source)
    detector = imageset.get_detector()
    if panel_id < 0 or panel_id >= len(detector):
        raise ValueError(f"panel_id {panel_id} out of range for detector with {len(detector)} panels.")

    panel = detector[panel_id]
    # Note: dxtbx panel.get_image_size() returns (fast_width, slow_height)
    panel_width, panel_height = int(panel.get_image_size()[0]), int(panel.get_image_size()[1])

    beam_yx: tuple[float, float] | None = None
    try:
        beam = imageset.get_beam(0)
        if beam is not None and panel.is_coord_valid_mm(panel.get_beam_centre(beam.get_s0())):
            bx, by = panel.get_beam_centre_px(beam.get_s0())
            beam_yx = (float(by), float(bx))
    except Exception:
        beam_yx = None

    if bbox_yx is None:
        y0, y1, x0, x1 = select_default_roi_bbox(
            panel_height, panel_width, tile_size=tile_size, beam_center_yx=beam_yx
        )
    else:
        y0, y1, x0, x1 = (
            max(0, int(bbox_yx[0])),
            min(panel_height, int(bbox_yx[1])),
            max(0, int(bbox_yx[2])),
            min(panel_width, int(bbox_yx[3])),
        )
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"Invalid bbox_yx {bbox_yx} for panel shape {(panel_height, panel_width)}.")

    n_total = len(imageset)
    if frame_range is not None:
        f_start = max(0, int(frame_range[0]))
        f_stop = min(n_total, int(frame_range[1]))
    else:
        f_start = 0
        f_stop = min(n_total, f_start + max(1, int(max_frames)) * max(1, int(frame_stride)))

    stride = max(1, int(frame_stride))
    indices = np.arange(f_start, f_stop, stride, dtype=np.int64)[: max(1, int(max_frames))]
    if indices.size < 4:
        raise ValueError(
            f"Need at least 4 frames for spatiotemporal estimation, got {indices.size} from imageset of length {n_total}."
        )

    th, tw = y1 - y0, x1 - x0
    images = np.empty((indices.size, th, tw), dtype=np.float64)
    untrusted_mask = np.zeros((th, tw), dtype=bool)

    for out_idx, img_idx in enumerate(indices):
        raw_tuple = imageset.get_raw_data(int(img_idx))
        raw_panel = raw_tuple[panel_id].as_numpy_array()
        images[out_idx] = raw_panel[y0:y1, x0:x1].astype(np.float64, copy=False)

        mask_tuple = imageset.get_mask(int(img_idx))
        mask_panel = mask_tuple[panel_id].as_numpy_array()
        # In dxtbx, mask_panel is True for valid/trusted pixels and False for untrusted pixels
        untrusted_mask |= ~mask_panel[y0:y1, x0:x1]

    dials_gains: np.ndarray | None = None
    if compute_dials_gain:
        try:
            dials_gains = compute_dials_dispersion_gain_on_tile(images, untrusted_mask)
        except Exception:
            dials_gains = None

    trusted = panel.get_trusted_range()
    px_size = panel.get_pixel_size()
    return DxtbxTileData(
        images=images,
        static_untrusted_mask=untrusted_mask,
        frame_indices=indices,
        panel_id=panel_id,
        panel_name=str(panel.get_name()),
        detector_gain=float(panel.get_gain()),
        detector_pedestal=float(panel.get_pedestal()),
        trusted_range=(float(trusted[0]), float(trusted[1])),
        bbox_yx=(y0, y1, x0, x1),
        panel_shape_yx=(panel_height, panel_width),
        dials_dispersion_gains=dials_gains,
        sensor_material=str(panel.get_material()) if hasattr(panel, "get_material") else "",
        sensor_thickness_mm=float(panel.get_thickness()) if hasattr(panel, "get_thickness") else 0.0,
        pixel_size_mm=(float(px_size[0]), float(px_size[1])),
    )


def estimate_from_dxtbx(
    source: object,
    panel_id: int = 0,
    bbox_yx: tuple[int, int, int, int] | None = None,
    tile_size: tuple[int, int] | None = (256, 256),
    frame_range: tuple[int, int] | None = None,
    max_frames: int = 80,
    frame_stride: int = 1,
    pedestal: float | None = None,
    trend_method: Literal[
        "hollow_stencil_kriging", "low_rank_spline", "temporal_difference"
    ] = "hollow_stencil_kriging",
    max_lag: int = 7,
    n_bins: int = 12,
    masking_config: MaskingConfig | None = None,
) -> tuple[DetectorStatsReport, DxtbxTileData]:
    """Extract a 3D tile from a dxtbx dataset and estimate `(G, gamma, psi, alpha(u))`.

    Automatically merges the dxtbx static untrusted-pixel mask (panel gaps, beamstop
    mask, negative overloaded pixels) with `detector_stats` static anomaly detection.

    Returns
    -------
    (DetectorStatsReport, DxtbxTileData)
    """
    tile = extract_dxtbx_tile(
        source=source,
        panel_id=panel_id,
        bbox_yx=bbox_yx,
        tile_size=tile_size,
        frame_range=frame_range,
        max_frames=max_frames,
        frame_stride=frame_stride,
        compute_dials_gain=True,
    )
    use_pedestal = tile.detector_pedestal if pedestal is None else float(pedestal)

    # Zero out dxtbx untrusted pixels to pedestal so static bad-pixel detection also flags them,
    # and pass the combined static mask into the pipeline.
    images_clean = tile.images.copy()
    if np.any(tile.static_untrusted_mask):
        images_clean[:, tile.static_untrusted_mask] = use_pedestal

    report = estimate_detector_statistics(
        images=images_clean,
        pedestal=use_pedestal,
        trend_method=trend_method,
        max_lag=max_lag,
        n_bins=n_bins,
        masking_config=masking_config,
        external_static_bad_mask=tile.static_untrusted_mask,
    )
    return report, tile
