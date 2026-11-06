"""Tests for diagnostic plotting utilities in detector_stats.plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector_stats import (
    DetectorConfig,
    SceneConfig,
    estimate_detector_statistics,
    plot_detector_stats_report,
    simulate_rotation_scan,
)


def test_plot_detector_stats_report_pt_ccd(tmp_path: Path) -> None:
    """Verify 6-panel figure generation and saving for a p-t-CCD simulation."""
    det = DetectorConfig.pt_ccd_waterman_evans(pedestal=50.0, psf_radius=3)
    scene = SceneConfig(n_frames=30, height=72, width=72, n_bragg_spots=30, n_zingers=5)
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(sim.images, pedestal=det.pedestal)

    out_png = tmp_path / "diagnostic_pt_ccd.png"
    fig = plot_detector_stats_report(
        rep,
        output_path=out_png,
        title="Test p-t-CCD Diagnostic Report",
        true_detector=det,
        true_flux=sim.incident_flux_scale,
    )
    try:
        assert len(fig.axes) >= 8
        assert out_png.exists()
        assert out_png.stat().st_size > 10_000
    finally:
        plt.close(fig)


def test_plot_detector_stats_report_timepix_sparse_clusters(tmp_path: Path) -> None:
    """Verify 6-panel figure generation when sparse single-event cluster histogram is active."""
    det = DetectorConfig.ed_pad_timepix(gain=2.9, cascade_factor=1.35)
    scene = SceneConfig(
        n_frames=40,
        height=88,
        width=88,
        include_sparse_corner=True,
        sparse_corner_quanta=0.012,
    )
    sim = simulate_rotation_scan(det, scene, seed=42)
    rep = estimate_detector_statistics(sim.images, pedestal=det.pedestal)

    assert rep.separation.separation_method == "sparse_cluster_multiplicity"
    assert rep.separation.cluster_sums is not None
    assert rep.separation.cluster_sums.size >= 40

    out_png = tmp_path / "diagnostic_timepix.png"
    fig = plot_detector_stats_report(
        rep,
        output_path=out_png,
        title="Test 3D ED Timepix Diagnostic Report",
        true_detector=det,
        true_flux=sim.incident_flux_scale,
    )
    try:
        assert len(fig.axes) >= 8
        assert out_png.exists()
        assert out_png.stat().st_size > 10_000
    finally:
        plt.close(fig)
