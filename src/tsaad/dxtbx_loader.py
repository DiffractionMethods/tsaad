"""Bridge between dxtbx ExperimentList / ImageSet objects and tsaad.

Provides memory-efficient extraction of 3D rotation-scan ROI tiles (T, H, W) from
any dxtbx ExperimentList, ImageSet, `.expt` file, or image sequence, along with
panel metadata, static and dynamic untrusted-pixel masks, and direct comparison
against the single-frame Kabsch spatial dispersion estimator (`dials.estimate_gain`)
implemented in pure NumPy.
"""

from __future__ import annotations

import atexit
import inspect
import logging
import os
import re
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tsaad.masking import MaskingConfig
from tsaad.pipeline import DetectorStatsReport, estimate_detector_statistics
from tsaad.trend import TrendEstimator, TrendMethodName

logger = logging.getLogger(__name__)


@dataclass
class DxtbxTileData:
    """Extracted 3D rotation-scan ROI tile and metadata from a dxtbx ImageSet."""

    images: np.ndarray  # (T, H, W) float64 raw pixel values in ADU
    static_untrusted_mask: np.ndarray  # (H, W) bool: True where pixel is statically untrusted
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
    dynamic_untrusted_mask: np.ndarray | None = None  # (T, H, W) bool: per-frame overloads/masks
    format_class: str = ""  # dxtbx Format class name (e.g. 'FormatCBFMiniPilatus')
    total_frames: int = 0  # total frames in full sweep before sub-window slicing

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


_NON_IMAGE_EXTS: frozenset[str] = frozenset(
    {
        ".expt",
        ".json",
        ".refl",
        ".log",
        ".txt",
        ".xml",
        ".ini",
        ".par",
        ".par3d",
        ".run",
        ".phil",
        ".sh",
        ".csh",
        ".com",
        ".py",
        ".md",
        ".csv",
        ".jpg",
        ".jpeg",
        ".png",
        ".pdf",
        ".html",
        ".js",
        ".lp",
        ".inp",
        ".xds",
        ".hkl",
        ".mtz",
        ".sca",
        ".cif",
        ".mmcif",
        ".pdb",
        ".dat",
        ".out",
        ".in",
        ".mat",
        ".bpm",
        ".gen",
        ".sum",
        ".p4p",
        ".x",
        ".gnu",
        ".pickle",
        ".npy",
    }
)

_XDS_TABLE_CBF_RE = re.compile(
    r"^(bkginit|blank|gain|abs|absorp|decay|modpix|bkgpix|"
    r"d?x-corrections|d?y-corrections|frame)(_\d+)?\.cbf$",
    re.IGNORECASE,
)

_SKIP_SUBDIR_NAMES: frozenset[str] = frozenset(
    {
        "other",
        "proc",
        "process",
        "processing",
        "xds",
        "xia2",
        "dials",
        "autoproc",
        "fast_dp",
        "hkl2000",
        "mosflm",
        "dimple",
        "ccp4",
        "logfiles",
    }
)

_PREFERRED_DATA_SUBDIRS: frozenset[str] = frozenset(
    {"data", "frames", "images", "series", "raw"}
)


def _list_candidate_image_files(directory: Path) -> list[Path]:
    """Return candidate diffraction image/master files directly inside `directory`."""
    candidates = [
        f
        for f in directory.iterdir()
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix.lower() not in _NON_IMAGE_EXTS
        and not _XDS_TABLE_CBF_RE.match(f.name)
    ]
    if not candidates:
        return []

    nxs_files = [f for f in candidates if f.suffix.lower() == ".nxs"]
    if nxs_files:
        return nxs_files

    master_h5 = [f for f in candidates if f.name.lower().endswith("_master.h5")]
    if master_h5:
        return master_h5

    # Group numeric frame extensions (.001 .. .360) under one bucket so they are never split
    ext_counts: dict[str, int] = {}
    for f in candidates:
        ext = f.suffix.lower()
        key = "<numeric>" if re.fullmatch(r"\.\d+", ext) else ext
        ext_counts[key] = ext_counts.get(key, 0) + 1
    dom_key = max(ext_counts, key=lambda k: ext_counts[k])
    return [
        f
        for f in candidates
        if ("<numeric>" if re.fullmatch(r"\.\d+", f.suffix.lower()) else f.suffix.lower())
        == dom_key
    ]


def _resolve_image_directory(root_dir: Path, max_depth: int = 4) -> tuple[Path, list[Path]]:
    """Locate `root_dir` or its primary raw-data subdirectory via bounded BFS."""
    direct = _list_candidate_image_files(root_dir)
    if direct:
        return root_dir, direct

    frontier = [root_dir]
    for _ in range(max_depth):
        next_frontier: list[Path] = []
        matches: list[tuple[Path, list[Path]]] = []
        for cur in frontier:
            try:
                subdirs = sorted(
                    d
                    for d in cur.iterdir()
                    if d.is_dir()
                    and not d.name.startswith(".")
                    and d.name.lower() not in _SKIP_SUBDIR_NAMES
                )
            except OSError:
                continue
            for sub in subdirs:
                cands = _list_candidate_image_files(sub)
                if cands:
                    matches.append((sub, cands))
                else:
                    next_frontier.append(sub)
        if matches:
            matches.sort(
                key=lambda item: (
                    item[0].name.lower() in _PREFERRED_DATA_SUBDIRS,
                    len(item[1]),
                ),
                reverse=True,
            )
            return matches[0]
        if not next_frontier:
            break
        frontier = next_frontier

    return root_dir, []


def _register_tempdir_cleanup(owner: object, link_dir: Path) -> None:
    """Clean up a temporary symlink directory on object GC and interpreter exit."""
    dir_str = str(link_dir)
    atexit.register(shutil.rmtree, dir_str, True)
    try:
        weakref.finalize(owner, shutil.rmtree, dir_str, True)
    except TypeError:
        # Boost.Python extension objects without weakref support still clean up via atexit
        pass


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
            imagesets = expts.imagesets()
            if not imagesets:
                raise ValueError(f"No imagesets found in {source!r}.")
            return imagesets[0]
        if p.is_dir():
            target_dir, candidates = _resolve_image_directory(p)
            dir_files = sorted(
                candidates,
                key=lambda f: [
                    int(tok) if tok.isdigit() else tok.lower()
                    for tok in re.split(r"(\d+)", f.name)
                ],
            )
            if dir_files:
                return load_imageset(dir_files)

            expts = ExperimentListFactory.from_filenames([str(target_dir)])
            imagesets = expts.imagesets()
            if not imagesets:
                raise ValueError(f"No image files found in directory {source!r}.")
            return sorted(imagesets, key=lambda ims: len(ims), reverse=True)[0]

        expts = ExperimentListFactory.from_filenames([str(p)])
        imagesets = expts.imagesets()
        if not imagesets:
            raise ValueError(f"No imagesets found in {source!r}.")
        return imagesets[0]

    if isinstance(source, Sequence):
        paths = [str(Path(item)) for item in source]
        link_dir: Path | None = None
        try:
            expts = ExperimentListFactory.from_filenames(paths)
        except ValueError as exc:
            if "Could not determine filename template" not in str(exc):
                raise
            # Workaround for Windows-style parenthesized filenames like '9P59 (100).sfrm'
            # where dxtbx's template_regex cannot parse the parenthesized frame index.
            link_dir = Path(tempfile.mkdtemp(prefix="tsaad_dxtbx_"))
            try:
                sorted_paths = sorted(
                    (Path(p) for p in paths),
                    key=lambda p: [
                        int(tok) if tok.isdigit() else tok.lower()
                        for tok in re.split(r"(\d+)", p.name)
                    ],
                )
                linked_paths: list[str] = []
                for idx, orig in enumerate(sorted_paths, start=1):
                    m = re.match(r"^(.*?)\s*\((\d+)\)(\..+)$", orig.name)
                    if m:
                        prefix = re.sub(r"\W+", "_", m.group(1)).strip("_") or "image"
                        frame_num = int(m.group(2))
                        ext = m.group(3)
                        clean_name = f"{prefix}_{frame_num:04d}{ext}"
                    else:
                        stem = re.sub(r"\W+", "_", orig.stem).strip("_") or "image"
                        clean_name = f"{stem}_{idx:04d}{orig.suffix}"
                    dst = link_dir / clean_name
                    if not dst.exists():
                        os.symlink(orig.resolve(), dst)
                    linked_paths.append(str(dst))
                expts = ExperimentListFactory.from_filenames(linked_paths)
            except Exception:
                shutil.rmtree(link_dir, ignore_errors=True)
                raise

        imagesets = expts.imagesets()
        if not imagesets:
            if link_dir is not None:
                shutil.rmtree(link_dir, ignore_errors=True)
            raise ValueError(f"No imagesets found in {source!r}.")

        best = sorted(imagesets, key=lambda ims: len(ims), reverse=True)[0]
        if len(paths) >= 4 and len(best) < 4:
            try:
                from dxtbx.imageset import ImageSetFactory

                fallback_ims = ImageSetFactory.make_imageset(paths)
                if len(fallback_ims) > len(best):
                    best = fallback_ims
            except (AttributeError, OSError, RuntimeError, ValueError) as exc:
                logger.debug("ImageSetFactory fallback failed: %s", exc)

        if link_dir is not None:
            _register_tempdir_cleanup(best, link_dir)
        return best

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


def _select_best_quadrant_roi_bbox(
    raw_panel0: np.ndarray,
    static_panel_untrusted: np.ndarray | None,
    trusted_range: tuple[float, float],
    pedestal: float,
    tile_size: tuple[int, int] | None,
    beam_center_yx: tuple[float, float] | None,
) -> tuple[int, int, int, int]:
    """Score multi-radius quadrant candidate ROIs on frame 0 for health and spatial gradient."""
    panel_height, panel_width = raw_panel0.shape
    primary = select_default_roi_bbox(
        panel_height, panel_width, tile_size=tile_size, beam_center_yx=beam_center_yx
    )
    if tile_size is None or primary == (0, panel_height, 0, panel_width):
        return primary

    th, tw = primary[1] - primary[0], primary[3] - primary[2]
    if beam_center_yx is not None:
        cy = float(np.clip(beam_center_yx[0], 0.0, panel_height - 1.0))
        cx = float(np.clip(beam_center_yx[1], 0.0, panel_width - 1.0))
    else:
        cy, cx = 0.5 * panel_height, 0.5 * panel_width

    dir_y0 = 1.0 if (panel_height - cy) >= cy else -1.0
    dir_x0 = 1.0 if (panel_width - cx) >= cx else -1.0
    candidates: list[tuple[int, int, int, int]] = [primary]
    for frac in (0.06, 0.16, 0.28):
        cl_y = max(32.0, frac * panel_height)
        cl_x = max(32.0, frac * panel_width)
        for dy_sign in (dir_y0, -dir_y0):
            for dx_sign in (dir_x0, -dir_x0):
                t_cy = cy + dy_sign * (cl_y + 0.5 * th)
                t_cx = cx + dx_sign * (cl_x + 0.5 * tw)
                y0 = int(np.clip(np.rint(t_cy - 0.5 * th), 0, panel_height - th))
                x0 = int(np.clip(np.rint(t_cx - 0.5 * tw), 0, panel_width - tw))
                cand = (y0, y0 + th, x0, x0 + tw)
                if cand not in candidates:
                    candidates.append(cand)

    min_t, max_t = trusted_range
    best_bbox = primary
    best_score = -1.0

    for idx, (y0, y1, x0, x1) in enumerate(candidates):
        sub = raw_panel0[y0:y1, x0:x1]
        valid = np.isfinite(sub) & (sub >= min_t) & (sub <= max_t)
        if static_panel_untrusted is not None:
            valid &= ~static_panel_untrusted[y0:y1, x0:x1]
        valid_frac = float(np.mean(valid))
        if valid_frac < 0.25:
            continue
        vals = sub[valid] - pedestal
        active_frac = float(np.mean(vals > 0.0))
        # Evaluate spatial background span on 17x17 block-averaged intensities (after clamping
        # top-5% Bragg spikes) so Poisson shot noise on a flat solvent-ring plateau is not
        # mistaken for a genuine macroscopic background gradient.
        if vals.size >= 64 and active_frac > 0.02:
            hi_clip = float(np.percentile(vals, 95.0))
            clipped = np.where(valid, np.minimum(sub - pedestal, hi_clip), 0.0)
            cnt_box = _box_sum_2d(valid.astype(np.float64), 8, 8)
            sum_box = _box_sum_2d(clipped, 8, 8)
            good_box = valid & (cnt_box >= 64.0)
            smooth_vals = (
                sum_box[good_box] / cnt_box[good_box]
                if np.count_nonzero(good_box) >= 32
                else vals
            )
        else:
            smooth_vals = vals
        q10, q50, q90 = np.percentile(smooth_vals, [10.0, 50.0, 90.0])
        denom = max(abs(float(q50)), 0.25 * abs(float(q90)), 1e-6)
        rel_span = float((q90 - q10) / denom)
        span_factor = min(1.0, max(0.20, rel_span / 0.45))
        # Small preference (+0.03) for primary quadrant when candidates are equally healthy
        tie_bonus = 0.03 if idx == 0 else 0.0
        score = valid_frac * (0.4 + 0.6 * active_frac) * span_factor + tie_bonus
        if score > best_score:
            best_score = score
            best_bbox = (y0, y1, x0, x1)

    return best_bbox


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
        mask = valid_base & np.isfinite(frame) & (frame < big_limit)
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


def _needs_legacy_bruker_format100_workaround(fmt_class: type | None) -> bool:
    """Return True only when running against an older dxtbx lacking the #905 fixes.

    Once dxtbx includes `fix-bruker-modern-905` (replacing underflow values instead of
    double-counting the baseline and populating `panel.get_pedestal()` from `NEXP[2]`),
    `extract_dxtbx_tile` delegates directly to `imageset.get_raw_data()` and
    `panel.get_pedestal()`.
    """
    if fmt_class is None or fmt_class.__name__ != "FormatBrukerModern":
        return False
    try:
        src_raw = inspect.getsource(fmt_class.get_raw_data)  # type: ignore[attr-defined]
        src_det = inspect.getsource(fmt_class._detector)  # type: ignore[attr-defined]
        has_underflow_fix = "underflow_sel" in src_raw
        has_pedestal_fix = "pedestal=" in src_det
        return not (has_underflow_fix and has_pedestal_fix)
    except (AttributeError, OSError, TypeError):
        return False


def _read_bruker_format100_panel(image_path: str) -> tuple[np.ndarray, float] | None:
    """Legacy fallback reader for Bruker FORMAT 100 `.sfrm` files on unpatched dxtbx (< #905).

    Uses `FormatBruker.open_file` (supporting `.bz2`/`.gz` streams), explicit little-endian
    unsigned integer types (`<u1`, `<u2`, `<u4`), and header/table bounds validation.
    Returns `(image_2d, baseline_pedestal)` or `None` if not a Bruker FORMAT 100 file.
    """
    dtype_by_bytes: dict[int, str] = {1: "<u1", 2: "<u2", 4: "<u4"}
    try:
        from dxtbx.format.FormatBruker import FormatBruker

        hdr = FormatBruker.parse_header(FormatBruker.read_header_lines(image_path))
        if hdr.get("FORMAT") != "100":
            return None
        nrows = int(hdr["NROWS"].split()[0])
        ncols = int(hdr["NCOLS"].split()[0])
        if nrows <= 0 or ncols <= 0:
            return None
        n_u, n_2o, n_4o = (int(e) for e in hdr["NOVERFL"].split()[:3])
        npixelb = [int(e) for e in hdr["NPIXELB"].split()]
        if not npixelb or npixelb[0] not in dtype_by_bytes:
            logger.debug("Unsupported Bruker NPIXELB=%s in %s", npixelb, image_path)
            return None

        main_b = npixelb[0]
        under_b = npixelb[1] if len(npixelb) > 1 else 1
        if under_b not in dtype_by_bytes:
            return None

        nexp_parts = [int(e) for e in hdr["NEXP"].split()]
        header_baseline = nexp_parts[2] if len(nexp_parts) > 2 else 0
        n_px = nrows * ncols
        hdr_bytes = int(hdr["HDRBLKS"]) * 512

        with FormatBruker.open_file(image_path, "rb") as f:
            hdr_buf = f.read(hdr_bytes)
            if len(hdr_buf) != hdr_bytes:
                return None
            raw_buf = f.read(n_px * main_b)
            if len(raw_buf) != n_px * main_b:
                return None
            raw = np.frombuffer(raw_buf, dtype=dtype_by_bytes[main_b]).astype(np.float64)

            u_vals: np.ndarray | None = None
            if n_u > 0:
                u_pad = (n_u * under_b + 15) & ~15
                u_buf = f.read(u_pad)
                if len(u_buf) < n_u * under_b:
                    return None
                u_vals = np.frombuffer(u_buf, dtype=dtype_by_bytes[under_b])[:n_u].astype(
                    np.float64
                )

            o2_vals: np.ndarray | None = None
            if n_2o > 0:
                o2_pad = (n_2o * 2 + 15) & ~15
                o2_buf = f.read(o2_pad)
                if len(o2_buf) < n_2o * 2:
                    return None
                o2_vals = np.frombuffer(o2_buf, dtype=np.dtype("<u2"))[:n_2o].astype(np.float64)

            o4_vals: np.ndarray | None = None
            if n_4o > 0:
                o4_pad = (n_4o * 4 + 15) & ~15
                o4_buf = f.read(o4_pad)
                if len(o4_buf) < n_4o * 4:
                    return None
                o4_vals = np.frombuffer(o4_buf, dtype=np.dtype("<u4"))[:n_4o].astype(np.float64)

        zero_mask = raw == 0
        if o2_vals is not None and main_b == 1:
            m255 = raw == 255
            if int(np.count_nonzero(m255)) != o2_vals.size:
                return None
            raw[m255] = o2_vals
        if o4_vals is not None and main_b in (1, 2):
            m65535 = raw == 65535
            if int(np.count_nonzero(m65535)) != o4_vals.size:
                return None
            raw[m65535] = o4_vals
        if n_u != -1:
            raw += float(header_baseline)
        if u_vals is not None:
            if int(np.count_nonzero(zero_mask)) != u_vals.size:
                return None
            raw[zero_mask] = u_vals
        return raw.reshape((nrows, ncols)), float(header_baseline)
    except (ImportError, OSError, KeyError, ValueError, IndexError) as exc:
        logger.debug("Bruker FORMAT 100 direct read fallback for %s: %s", image_path, exc)
        return None


def _read_panel_frame(
    imageset: object,
    img_idx: int,
    panel_id: int,
    use_legacy_bruker_reader: bool,
) -> tuple[np.ndarray, float | None]:
    """Read a single 2D panel image from `imageset` (with legacy Bruker fallback if needed)."""
    if use_legacy_bruker_reader:
        try:
            img_path = imageset.get_path(img_idx)  # type: ignore[attr-defined]
            bruker_res = _read_bruker_format100_panel(img_path)
            if bruker_res is not None:
                return bruker_res[0], bruker_res[1]
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.debug("Bruker path lookup failed on frame %d: %s", img_idx, exc)

    raw_tuple = imageset.get_raw_data(img_idx)  # type: ignore[attr-defined]
    raw_panel = raw_tuple[panel_id].as_numpy_array()
    return raw_panel, None


def _try_load_per_frame_subwindow(
    source: object,
    frame_range: tuple[int, int] | None,
    max_frames: int,
    frame_stride: int,
) -> tuple[Any, int, int] | None:
    """Fast path for directories with hundreds/thousands of per-frame files.

    Groups candidate filenames into contiguous sweeps in pure Python and opens only the
    requested sub-window via `dxtbx`, avoiding O(N_files) header reads over NFS.
    Returns `(sub_imageset, frame_index_offset)` or `None` when not applicable.
    """
    if not isinstance(source, (str, Path)):
        return None
    p = Path(source)
    if not p.is_dir():
        return None

    stride = max(1, int(frame_stride))
    req_span = max(1, int(max_frames)) * stride
    _, candidates = _resolve_image_directory(p)
    if len(candidates) <= max(160, 2 * req_span):
        return None
    if candidates[0].suffix.lower() in {".nxs", ".h5", ".hdf5", ".mrc"}:
        return None

    groups: dict[tuple[str, str, str], list[tuple[int, Path]]] = {}
    for f in candidates:
        name_low = f.name.lower()
        comp_ext = ""
        for c_ext in (".bz2", ".gz", ".xz"):
            if name_low.endswith(c_ext):
                comp_ext = c_ext
                name_low = name_low[: -len(c_ext)]
                break
        base_stem, base_ext = os.path.splitext(name_low)
        if re.fullmatch(r"\.\d+", base_ext):
            key = (base_stem, "", "<numeric>" + comp_ext)
            num = int(base_ext[1:])
        else:
            m = re.match(r"^(.*?)(\d+)(\D*)$", base_stem)
            if not m:
                continue
            key = (m.group(1), m.group(3), base_ext + comp_ext)
            num = int(m.group(2))
        groups.setdefault(key, []).append((num, f))

    if not groups:
        return None
    best_items = max(groups.values(), key=len)
    best_items.sort(key=lambda item: (item[0], item[1].name))
    n_sweep = len(best_items)
    if n_sweep <= max(160, 2 * req_span):
        return None

    nums = [item[0] for item in best_items]
    if len(set(nums)) != n_sweep or (nums[-1] - nums[0]) != (n_sweep - 1):
        return None

    if frame_range is not None:
        f_start = max(0, int(frame_range[0]))
        f_stop = min(n_sweep, int(frame_range[1]))
    else:
        span = min(n_sweep, req_span)
        f_start = max(0, (n_sweep - span) // 2)
        f_stop = f_start + span

    window_files = [item[1] for item in best_items[f_start:f_stop]]
    if len(window_files) < 4:
        return None

    try:
        sub_ims = load_imageset(window_files)
    except Exception as exc:
        logger.debug("Sub-window directory fast path fell back to full load: %s", exc)
        return None

    if len(sub_ims) != len(window_files):
        return None
    return sub_ims, f_start, n_sweep


def extract_dxtbx_tile(
    source: object,
    panel_id: int = 0,
    bbox_yx: tuple[int, int, int, int] | None = None,
    tile_size: tuple[int, int] | None = (256, 256),
    frame_range: tuple[int, int] | None = None,
    max_frames: int = 80,
    frame_stride: int = 1,
    compute_dials_gain: bool = True,
    auto_select_roi: bool = True,
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
        `select_default_roi_bbox` (with multi-quadrant health scoring when `auto_select_roi=True`).
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
        Whether to also evaluate the pure-NumPy single-frame Kabsch dispersion gain baseline.
    auto_select_roi : bool, default True
        When `bbox_yx` is None and the panel is larger than `tile_size`, score the 4 quadrants
        around the beam center on the first frame to avoid shadowed or dead detector regions.

    Returns
    -------
    DxtbxTileData
        Extracted 3D image stack, static and dynamic untrusted masks, panel metadata,
        and single-frame dispersion baseline.
    """
    sub_win = _try_load_per_frame_subwindow(
        source=source,
        frame_range=frame_range,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    if sub_win is not None:
        imageset, index_offset, total_sweep_frames = sub_win
    else:
        imageset = load_imageset(source)
        index_offset = 0
        total_sweep_frames = len(imageset)

    detector = imageset.get_detector()
    if panel_id < 0 or panel_id >= len(detector):
        raise ValueError(
            f"panel_id {panel_id} out of range for detector with {len(detector)} panels."
        )

    panel = detector[panel_id]
    # Note: dxtbx panel.get_image_size() returns (fast_width, slow_height)
    panel_width, panel_height = int(panel.get_image_size()[0]), int(panel.get_image_size()[1])

    beam_yx: tuple[float, float] | None = None
    try:
        beam = imageset.get_beam(0)
        if beam is not None and panel.is_coord_valid_mm(panel.get_beam_centre(beam.get_s0())):
            bx, by = panel.get_beam_centre_px(beam.get_s0())
            beam_yx = (float(by), float(bx))
    except (AttributeError, RuntimeError, ValueError) as exc:
        logger.debug("Beam center projection unavailable: %s", exc)
        beam_yx = None

    n_total = len(imageset)
    stride = max(1, int(frame_stride))
    if sub_win is not None:
        f_start, f_stop = 0, n_total
    elif frame_range is not None:
        f_start = max(0, int(frame_range[0]))
        f_stop = min(n_total, int(frame_range[1]))
    else:
        # Center the contiguous window around mid-scan when n_total > max_frames * stride,
        # avoiding high-tilt TEM grid-bar eclipses in 3DED and frame-0 shutter lag.
        span = min(n_total, max(1, int(max_frames)) * stride)
        f_start = max(0, (n_total - span) // 2)
        f_stop = f_start + span

    indices = np.arange(f_start, f_stop, stride, dtype=np.int64)[: max(1, int(max_frames))]
    if indices.size < 4:
        raise ValueError(
            f"Need at least 4 frames for spatiotemporal estimation, got {indices.size} "
            f"from imageset of length {n_total}."
        )

    trusted = panel.get_trusted_range()
    min_trusted, max_trusted = float(trusted[0]), float(trusted[1])
    nominal_pedestal = float(panel.get_pedestal())

    has_dyn_mask = (
        bool(imageset.has_dynamic_mask()) if hasattr(imageset, "has_dynamic_mask") else False
    )

    # Query static geometry/pixel mask once outside the frame loop (avoiding full-panel C++
    # copies on every frame while preventing single-frame overloads from polluting static mask).
    static_panel_untrusted: np.ndarray | None = None
    if hasattr(imageset, "get_static_mask"):
        try:
            sm_tuple = imageset.get_static_mask()
            if sm_tuple is not None and len(sm_tuple) > panel_id:
                static_panel_untrusted = ~sm_tuple[panel_id].as_numpy_array()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("get_static_mask() failed: %s", exc)

    fmt_class = imageset.get_format_class()
    use_legacy_bruker = panel_id == 0 and _needs_legacy_bruker_format100_workaround(fmt_class)
    bruker_pedestal: float | None = None

    # Read first frame once so it can both score candidate ROIs (when auto_select_roi=True)
    # and populate images[0] without re-reading from disk.
    panel0_raw, b_ped0 = _read_panel_frame(
        imageset, int(indices[0]), panel_id, use_legacy_bruker_reader=use_legacy_bruker
    )
    if b_ped0 is not None:
        bruker_pedestal = b_ped0
    effective_pedestal_init = (
        float(bruker_pedestal)
        if (bruker_pedestal is not None and nominal_pedestal == 0.0)
        else nominal_pedestal
    )

    if static_panel_untrusted is None and not has_dyn_mask and hasattr(imageset, "get_mask"):
        try:
            m0_tuple = imageset.get_mask(int(indices[0]))
            if m0_tuple is not None and len(m0_tuple) > panel_id:
                m0_untrusted = ~m0_tuple[panel_id].as_numpy_array()
                # Exclude pixels that only failed get_mask(0) due to single-frame overload
                static_panel_untrusted = m0_untrusted & (~(panel0_raw > max_trusted))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("get_mask(0) static extraction failed: %s", exc)

    if bbox_yx is None:
        if auto_select_roi:
            y0, y1, x0, x1 = _select_best_quadrant_roi_bbox(
                raw_panel0=np.asarray(panel0_raw, dtype=np.float64),
                static_panel_untrusted=static_panel_untrusted,
                trusted_range=(min_trusted, max_trusted),
                pedestal=effective_pedestal_init,
                tile_size=tile_size,
                beam_center_yx=beam_yx,
            )
        else:
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
            raise ValueError(
                f"Invalid bbox_yx {bbox_yx} for panel shape {(panel_height, panel_width)}."
            )

    th, tw = y1 - y0, x1 - x0
    images = np.empty((indices.size, th, tw), dtype=np.float64)
    images[0] = panel0_raw[y0:y1, x0:x1].astype(np.float64, copy=False)
    del panel0_raw

    untrusted_mask = np.zeros((th, tw), dtype=bool)
    if static_panel_untrusted is not None:
        untrusted_mask |= static_panel_untrusted[y0:y1, x0:x1]
        del static_panel_untrusted

    dynamic_untrusted = np.zeros((indices.size, th, tw), dtype=bool)
    if has_dyn_mask:
        try:
            dm0 = imageset.get_dynamic_mask(int(indices[0]))
            if dm0 is not None and len(dm0) > panel_id:
                dynamic_untrusted[0] |= ~dm0[panel_id].as_numpy_array()[y0:y1, x0:x1]
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.debug("get_dynamic_mask(0) failed: %s", exc)
            has_dyn_mask = False

    for out_idx in range(1, indices.size):
        img_idx = int(indices[out_idx])
        raw_panel, b_ped = _read_panel_frame(
            imageset, img_idx, panel_id, use_legacy_bruker_reader=use_legacy_bruker
        )
        images[out_idx] = raw_panel[y0:y1, x0:x1].astype(np.float64, copy=False)
        if bruker_pedestal is None and b_ped is not None:
            bruker_pedestal = b_ped

        if has_dyn_mask:
            try:
                dm = imageset.get_dynamic_mask(img_idx)
                if dm is not None and len(dm) > panel_id:
                    dynamic_untrusted[out_idx] |= ~dm[panel_id].as_numpy_array()[y0:y1, x0:x1]
            except (AttributeError, RuntimeError, ValueError) as exc:
                logger.debug("get_dynamic_mask(%d) failed: %s", img_idx, exc)
                has_dyn_mask = False

    # Evaluate trusted_range directly on the sliced (T, th, tw) tile:
    # - Underflow / negative sentinel pixels (< min_trusted in any frame) or pixels saturated
    #   in ALL frames belong to the 2D static_untrusted_mask.
    # - Per-frame saturated pixels (> max_trusted on a subset of frames) belong to the 3D
    #   dynamic_untrusted_mask so non-saturated frames at those pixels remain valid!
    nonfinite_3d = ~np.isfinite(images)
    underflow_2d = np.any((images < min_trusted) & (~nonfinite_3d), axis=0)
    overload_3d = (images > max_trusted) | nonfinite_3d
    always_overloaded_2d = np.all(overload_3d | dynamic_untrusted, axis=0)
    untrusted_mask |= underflow_2d | always_overloaded_2d

    dynamic_untrusted |= overload_3d
    dynamic_untrusted &= ~untrusted_mask[None, :, :]

    dials_gains: np.ndarray | None = None
    if compute_dials_gain:
        try:
            dials_gains = compute_dials_dispersion_gain_on_tile(images, untrusted_mask)
        except (ValueError, FloatingPointError) as exc:
            logger.debug("Dispersion gain baseline calculation failed: %s", exc)
            dials_gains = None

    px_size = panel.get_pixel_size()
    effective_pedestal = (
        float(bruker_pedestal)
        if (bruker_pedestal is not None and nominal_pedestal == 0.0)
        else nominal_pedestal
    )
    # Legacy FormatBrukerModern (< dxtbx #906) hardcoded Pilatus-copied "Si" / 0.45 mm on all
    # Bruker detectors; ignore those bogus sensor attributes when the legacy fallback is active.
    sensor_material = (
        ""
        if use_legacy_bruker
        else (str(panel.get_material()) if hasattr(panel, "get_material") else "")
    )
    sensor_thickness_mm = (
        0.0
        if use_legacy_bruker
        else (float(panel.get_thickness()) if hasattr(panel, "get_thickness") else 0.0)
    )
    fmt_name = (
        fmt_class.__name__ if fmt_class is not None else type(imageset).__name__
    )
    return DxtbxTileData(
        images=images,
        static_untrusted_mask=untrusted_mask,
        frame_indices=indices + index_offset,
        panel_id=panel_id,
        panel_name=str(panel.get_name()),
        detector_gain=float(panel.get_gain()),
        detector_pedestal=effective_pedestal,
        trusted_range=(min_trusted, max_trusted),
        bbox_yx=(y0, y1, x0, x1),
        panel_shape_yx=(panel_height, panel_width),
        dials_dispersion_gains=dials_gains,
        sensor_material=sensor_material,
        sensor_thickness_mm=sensor_thickness_mm,
        pixel_size_mm=(float(px_size[0]), float(px_size[1])),
        dynamic_untrusted_mask=dynamic_untrusted,
        format_class=fmt_name,
        total_frames=int(total_sweep_frames),
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
    trend_method: TrendMethodName | TrendEstimator = "hollow_stencil_kriging",
    max_lag: int = 7,
    n_bins: int = 12,
    masking_config: MaskingConfig | None = None,
    compute_dials_gain: bool = True,
    auto_select_roi: bool = True,
) -> tuple[DetectorStatsReport, DxtbxTileData]:
    """Extract a 3D tile from a dxtbx dataset and estimate `(G, gamma, psi, alpha(u))`.

    Automatically merges the dxtbx static untrusted-pixel mask (panel gaps, beamstop
    mask, negative sentinel pixels) with `tsaad` static anomaly detection,
    while forwarding per-frame saturated pixels into the dynamic transient mask.

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
        compute_dials_gain=compute_dials_gain,
        auto_select_roi=auto_select_roi,
    )
    use_pedestal = tile.detector_pedestal if pedestal is None else float(pedestal)

    if np.any(tile.static_untrusted_mask):
        images_clean = tile.images.copy()
        images_clean[:, tile.static_untrusted_mask] = use_pedestal
    else:
        images_clean = tile.images

    report = estimate_detector_statistics(
        images=images_clean,
        pedestal=use_pedestal,
        trend_method=trend_method,
        max_lag=max_lag,
        n_bins=n_bins,
        masking_config=masking_config,
        external_static_bad_mask=tile.static_untrusted_mask,
        external_dynamic_bad_mask=tile.dynamic_untrusted_mask,
        trusted_range=tile.trusted_range,
    )
    return report, tile
