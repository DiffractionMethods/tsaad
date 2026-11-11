"""Command-line entry point (`tsaad`) for Time Series Analysis of Area Detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tsaad.pipeline import DetectorStatsReport, estimate_detector_statistics


def report_to_dict(report: DetectorStatsReport, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a `DetectorStatsReport` into a JSON-friendly dictionary."""
    cov = report.covariance
    sep = report.separation
    qual = report.quality
    payload: dict[str, Any] = {
        "gain": float(report.gain),
        "gain_se": float(sep.gain_se),
        "cascade_factor": float(report.cascade_factor),
        "cascade_factor_se": float(sep.cascade_factor_se),
        "g_gamma": float(report.g_gamma),
        "g_gamma_se": float(cov.g_gamma_se),
        "psi": float(report.psi),
        "psi_se": float(cov.psi_se),
        "apparent_single_pixel_gain": float(cov.apparent_single_pixel_gain),
        "s2_factor": float(report.s2_factor),
        "psf_radius": int(report.psf_radius),
        "psf_converged": bool(cov.psf_converged),
        "estimated_pedestal_offset": float(cov.estimated_pedestal_offset),
        "separated": bool(sep.separated),
        "separation_method": sep.separation_method,
        "n_isolated_clusters": int(sep.n_isolated_clusters),
        "trend_method": report.trend.method_name,
    }
    if qual is not None:
        payload["quality"] = {
            "is_reliable": bool(qual.is_reliable),
            "standardized_residual_std": float(qual.standardized_residual_std),
            "valid_voxel_fraction": float(qual.valid_voxel_fraction),
            "static_bad_fraction": float(qual.static_bad_fraction),
            "transient_masked_fraction": float(qual.transient_masked_fraction),
            "dynamic_range_cv": float(qual.dynamic_range_cv),
            "psf_converged": bool(qual.psf_converged),
            "warnings": list(qual.warnings),
        }
    if extra:
        payload.update(extra)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for `tsaad`."""
    parser = argparse.ArgumentParser(
        prog="tsaad",
        description=(
            "Time Series Analysis of Area Detectors (tsaad): estimate detector response "
            "parameters (G, gamma, psi, alpha(u)) from a diffraction rotation scan."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path to a .npy/.npz 3D stack, a dxtbx .expt file, or diffraction image file(s).",
    )
    parser.add_argument(
        "--pedestal",
        type=float,
        default=None,
        help="Detector baseline pedestal in ADU (defaults to 0.0 for .npy or panel pedestal for dxtbx).",
    )
    parser.add_argument(
        "--panel",
        type=int,
        default=0,
        help="0-based detector panel index when reading via dxtbx (default: 0).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help="ROI tile height and width for dxtbx extraction (default: 256 256; 0 0 for full panel).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Maximum number of rotation frames to analyze (default: 80).",
    )
    parser.add_argument(
        "--trend-method",
        choices=("hollow_stencil_kriging", "low_rank_spline", "temporal_difference"),
        default="hollow_stencil_kriging",
        help="Spatiotemporal background detrending method (default: hollow_stencil_kriging).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=7,
        help="Maximum spatial lag radius L in pixels (default: 7).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of intensity quantile bins (default: 12).",
    )
    parser.add_argument(
        "--no-pedestal-calibration",
        action="store_true",
        help="Disable automatic off-diagonal ring-1 pedestal offset calibration.",
    )
    parser.add_argument(
        "--json",
        "--json-out",
        dest="json",
        type=Path,
        default=None,
        help="Optional output path for JSON summary report.",
    )
    parser.add_argument(
        "--plot",
        "--plot-out",
        dest="plot",
        type=Path,
        default=None,
        help="Optional output path for 8-panel diagnostic figure (e.g. report.png).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `tsaad` command-line interface."""
    args = build_parser().parse_args(argv)
    inputs = [Path(p) for p in args.inputs]
    extra_meta: dict[str, Any] = {}

    if len(inputs) == 1 and inputs[0].suffix.lower() in {".npy", ".npz"}:
        path0 = inputs[0]
        if path0.suffix.lower() == ".npy":
            images = np.load(path0)
        else:
            with np.load(path0) as archive:
                key = "images" if "images" in archive else archive.files[0]
                images = archive[key]
        if args.max_frames > 0 and images.shape[0] > args.max_frames:
            images = images[: args.max_frames]
        use_pedestal = 0.0 if args.pedestal is None else float(args.pedestal)
        report = estimate_detector_statistics(
            images=images,
            pedestal=use_pedestal,
            trend_method=args.trend_method,
            max_lag=args.max_lag,
            n_bins=args.n_bins,
            auto_calibrate_pedestal=not args.no_pedestal_calibration,
        )
        title = path0.name
    else:
        from tsaad.dxtbx_loader import estimate_from_dxtbx

        th, tw = int(args.tile_size[0]), int(args.tile_size[1])
        tile_sz = None if (th <= 0 or tw <= 0) else (th, tw)
        source: object = inputs[0] if len(inputs) == 1 else inputs
        report, tile = estimate_from_dxtbx(
            source=source,
            panel_id=args.panel,
            tile_size=tile_sz,
            max_frames=args.max_frames,
            pedestal=args.pedestal,
            trend_method=args.trend_method,
            max_lag=args.max_lag,
            n_bins=args.n_bins,
        )
        extra_meta = {
            "panel_id": tile.panel_id,
            "panel_name": tile.panel_name,
            "bbox_yx": list(tile.bbox_yx),
            "nominal_gain": tile.detector_gain,
            "nominal_pedestal": tile.detector_pedestal,
            "dials_dispersion_gain_mean": tile.dials_dispersion_gain_mean,
        }
        title = f"{inputs[0].name} ({tile.panel_name})"

    summary = report_to_dict(report, extra=extra_meta)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tsaad.plotting import plot_detector_stats_report

        fig = plot_detector_stats_report(report, output_path=args.plot, title=title)
        plt.close(fig)

    print(
        f"G*gamma = {report.g_gamma:.4f} +/- {report.covariance.g_gamma_se:.4f} | "
        f"G = {report.gain:.4f} | gamma = {report.cascade_factor:.4f} "
        f"({report.separation.separation_method}) | "
        f"psi = {report.psi:.3f} +/- {report.covariance.psi_se:.3f} ADU^2 | "
        f"S2 = {report.s2_factor:.3f} | r_PSF = {report.psf_radius} px"
    )
    if report.quality is not None and report.quality.warnings:
        for warning_msg in report.quality.warnings:
            print(f"WARNING: {warning_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
