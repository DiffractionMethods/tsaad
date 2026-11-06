"""Tests for the dxtbx bridge and tile extractor in detector_stats.dxtbx_loader."""

from __future__ import annotations

import numpy as np
import pytest

from detector_stats import (
    DetectorConfig,
    SceneConfig,
    estimate_from_dxtbx,
    extract_dxtbx_tile,
    select_default_roi_bbox,
    simulate_rotation_scan,
)


def test_select_default_roi_bbox() -> None:
    """Verify ROI bounding-box selection for small and large panels."""
    # Small panel <= tile_size returns full panel
    assert select_default_roi_bbox(100, 120, tile_size=(256, 256)) == (0, 100, 0, 120)
    assert select_default_roi_bbox(500, 600, tile_size=None) == (0, 500, 0, 600)

    # Large panel returns in-bounds tile of requested size
    y0, y1, x0, x1 = select_default_roi_bbox(
        1024, 1024, tile_size=(256, 256), beam_center_yx=(512.0, 512.0)
    )
    assert (y1 - y0, x1 - x0) == (256, 256)
    assert 0 <= y0 < y1 <= 1024
    assert 0 <= x0 < x1 <= 1024


def test_estimate_from_dxtbx_memreader_imageset() -> None:
    """Verify end-to-end extraction and estimation from a dxtbx ImageSet."""
    pytest.importorskip("dxtbx")
    from dxtbx.imageset import ImageSet, ImageSetData, MemReader
    from dxtbx.model import Beam, Detector, Panel
    from scitbx.array_family import flex

    det_cfg = DetectorConfig.pt_ccd_waterman_evans(pedestal=40.0, psf_radius=3)
    scene_cfg = SceneConfig(n_frames=36, height=96, width=96, n_bragg_spots=35, n_zingers=5)
    sim = simulate_rotation_scan(det_cfg, scene_cfg, seed=19)

    class _SimImage:
        def __init__(self, arr_2d: np.ndarray) -> None:
            self._raw = (flex.double(np.ascontiguousarray(arr_2d, dtype=np.float64)),)

        def get_raw_data(self):
            return self._raw

    # Inject a synthetic untrusted 6x6 beamstop patch into the raw images (value -10 < trusted_range[0])
    raw_images = sim.images.copy()
    raw_images[:, 45:51, 45:51] = -10.0

    items = [_SimImage(raw_images[t]) for t in range(scene_cfg.n_frames)]
    reader = MemReader(items)
    iset_data = ImageSetData(reader, None)

    detector = Detector()
    panel = Panel()
    panel.set_name("Panel0")
    panel.set_image_size((scene_cfg.width, scene_cfg.height))
    panel.set_pixel_size((0.1, 0.1))
    panel.set_trusted_range((0.0, 65535.0))
    panel.set_gain(det_cfg.gain)
    panel.set_pedestal(det_cfg.pedestal)
    detector.add_panel(panel)

    beam = Beam((0.0, 0.0, -1.0), 1.0)
    for t in range(scene_cfg.n_frames):
        iset_data.set_detector(detector, t)
        iset_data.set_beam(beam, t)

    imageset = ImageSet(iset_data)

    tile = extract_dxtbx_tile(imageset, tile_size=None, max_frames=36)
    assert tile.images.shape == (36, 96, 96)
    assert tile.panel_name == "Panel0"
    assert np.isclose(tile.detector_gain, det_cfg.gain)
    assert np.isclose(tile.detector_pedestal, det_cfg.pedestal)
    assert np.all(tile.static_untrusted_mask[45:51, 45:51])
    assert tile.dials_dispersion_gain_mean is not None
    # DIALS single-frame dispersion underestimates G*gamma under the phosphor PSF
    assert tile.dials_dispersion_gain_mean < 0.6 * det_cfg.g_gamma

    rep, tile2 = estimate_from_dxtbx(imageset, tile_size=None, max_frames=36)
    assert np.all(rep.masking.static_bad_mask[45:51, 45:51])
    assert np.isclose(rep.g_gamma, det_cfg.g_gamma, rtol=0.08)
    assert np.isclose(rep.psi, det_cfg.pixel_factor, atol=0.80)
    assert rep.psf_radius >= 2
