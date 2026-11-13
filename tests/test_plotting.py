"""Tests for diagnostic plotting utilities and CLI output in tsaad."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsaad import (  # noqa: E402
    DetectorConfig,
    SceneConfig,
    estimate_detector_statistics,
    plot_detector_stats_report,
    plot_multi_lag_ptc,
    plot_multi_lag_variance_mean,
    simulate_rotation_scan,
)
from tsaad.cli import main as cli_main  # noqa: E402


def test_plot_detector_stats_report_pt_ccd(tmp_path: Path) -> None:
    """Verify 8-panel figure generation and saving for a p-t-CCD simulation."""
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
        assert fig.axes[0].get_title() == "Multi-Lag Variance\u2013Mean Curves"
        assert plot_multi_lag_ptc is plot_multi_lag_variance_mean
        # Detrended Mean panel (bottom row, 3rd subplot) must not show spot-footprint contours
        detrended_ax = [ax for ax in fig.axes if ax.get_title().startswith("Detrended Mean")][0]
        leg = detrended_ax.get_legend()
        if leg is not None:
            labels = [t.get_text() for t in leg.get_texts()]
            assert all("Spot footprints" not in lbl for lbl in labels)
            assert any("Static mask" in lbl for lbl in labels)
        assert out_png.exists()
        assert out_png.stat().st_size > 10_000
    finally:
        plt.close(fig)


def test_plot_detector_stats_report_timepix_sparse_clusters(tmp_path: Path) -> None:
    """Verify 8-panel figure generation when sparse single-event cluster histogram is active."""
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
        assert fig.axes[0].get_title() == "Multi-Lag Variance\u2013Mean Curves"
        assert plot_multi_lag_ptc is plot_multi_lag_variance_mean
        # Detrended Mean panel (bottom row, 3rd subplot) must not show spot-footprint contours
        detrended_ax = [ax for ax in fig.axes if ax.get_title().startswith("Detrended Mean")][0]
        leg = detrended_ax.get_legend()
        if leg is not None:
            labels = [t.get_text() for t in leg.get_texts()]
            assert all("Spot footprints" not in lbl for lbl in labels)
            assert any("Static mask" in lbl for lbl in labels)
        assert out_png.exists()
        assert out_png.stat().st_size > 10_000
    finally:
        plt.close(fig)


def test_cli_npy_json_and_plot(tmp_path: Path) -> None:
    """Verify the `tsaad` CLI entrypoint on a 3D `.npy` stack."""
    det = DetectorConfig.ideal_pad()
    scene = SceneConfig(n_frames=24, height=64, width=64, n_bragg_spots=15)
    sim = simulate_rotation_scan(det, scene, seed=11)

    npy_path = tmp_path / "scan.npy"
    json_path = tmp_path / "report.json"
    png_path = tmp_path / "report.png"
    np.save(npy_path, sim.images.astype(np.float32))

    rc = cli_main(
        [
            str(npy_path),
            "--pedestal",
            "0.0",
            "--max-lag",
            "4",
            "--json",
            str(json_path),
            "--plot",
            str(png_path),
        ]
    )
    assert rc == 0
    assert json_path.exists()
    assert png_path.exists() and png_path.stat().st_size > 10_000
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert abs(payload["g_gamma"] - 1.0) < 0.05
    assert payload["quality"]["is_reliable"] is True
