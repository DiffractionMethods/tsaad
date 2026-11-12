"""Tests for the dxtbx bridge and tile extractor in tsaad.dxtbx_loader."""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import pytest

from tsaad import (
    DetectorConfig,
    SceneConfig,
    estimate_from_dxtbx,
    extract_dxtbx_tile,
    select_default_roi_bbox,
    simulate_rotation_scan,
)
from tsaad.dxtbx_loader import (
    _needs_legacy_bruker_format100_workaround,
    _read_bruker_format100_panel,
    load_imageset,
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


def test_beamstop_shadow_penumbra_dilation() -> None:
    """Verify that an interior circular beamstop shadow with a soft penumbra is dilated."""
    from tsaad import detect_static_bad_pixels

    det_cfg = DetectorConfig.pt_ccd_waterman_evans(pedestal=40.0, psf_radius=3)
    scene_cfg = SceneConfig(n_frames=30, height=96, width=96, n_bragg_spots=20, shadow_margin_px=0)
    sim = simulate_rotation_scan(det_cfg, scene_cfg, seed=7)

    # Inject a synthetic circular beamstop shadow (radius 10 px) with a 4-px soft penumbra
    y, x = np.mgrid[0:96, 0:96]
    r_bs = np.hypot(y - 48.0, x - 48.0)
    atten = np.clip((r_bs - 10.0) / 4.0, 0.0, 1.0)
    imgs = det_cfg.pedestal + (sim.images - det_cfg.pedestal) * atten[None, :, :]

    static_mask = detect_static_bad_pixels(imgs, pedestal=det_cfg.pedestal)
    # Both the core (r <= 10) and the inner penumbra (10 < r <= 12.5) must be masked
    assert np.all(static_mask[r_bs <= 10.0])
    assert np.mean(static_mask[(r_bs > 10.0) & (r_bs <= 12.5)]) > 0.85


def test_dxtbx_saturated_bragg_spot_stays_dynamic_not_static() -> None:
    """A single-frame saturated Bragg peak (> max_trusted) must populate dynamic_untrusted_mask
    rather than wiping those pixels across all frames via static_untrusted_mask.
    """
    pytest.importorskip("dxtbx")
    from dxtbx.imageset import ImageSet, ImageSetData, MemReader
    from dxtbx.model import Beam, Detector, Panel
    from scitbx.array_family import flex

    det_cfg = DetectorConfig.ideal_pad()
    scene_cfg = SceneConfig(
        n_frames=28, height=72, width=72, n_bragg_spots=15, n_hot_pixels=0, n_dead_pixels=0,
        shadow_margin_px=0,
    )
    sim = simulate_rotation_scan(det_cfg, scene_cfg, seed=21)

    class _SimImage:
        def __init__(self, arr_2d: np.ndarray) -> None:
            self._raw = (flex.double(np.ascontiguousarray(arr_2d, dtype=np.float64)),)

        def get_raw_data(self):
            return self._raw

    raw_images = sim.images.copy()
    # Static negative sentinel patch (-2 < min_trusted=0) in all frames
    raw_images[:, 55:60, 55:60] = -2.0
    # Single-frame saturated Bragg spot (> max_trusted=65535) ONLY on frame 10
    raw_images[10, 22:26, 22:26] = 95_000.0

    items = [_SimImage(raw_images[t]) for t in range(scene_cfg.n_frames)]
    iset_data = ImageSetData(MemReader(items), None)

    detector = Detector()
    panel = Panel()
    panel.set_name("Panel0")
    panel.set_image_size((scene_cfg.width, scene_cfg.height))
    panel.set_pixel_size((0.1, 0.1))
    panel.set_trusted_range((0.0, 65535.0))
    panel.set_gain(1.0)
    panel.set_pedestal(0.0)
    detector.add_panel(panel)

    beam = Beam((0.0, 0.0, -1.0), 1.0)
    for t in range(scene_cfg.n_frames):
        iset_data.set_detector(detector, t)
        iset_data.set_beam(beam, t)

    imageset = ImageSet(iset_data)
    rep, tile = estimate_from_dxtbx(
        imageset, tile_size=None, max_frames=28, compute_dials_gain=False
    )

    assert tile.dials_dispersion_gains is None
    assert np.all(tile.static_untrusted_mask[55:60, 55:60])
    assert not np.any(tile.static_untrusted_mask[22:26, 22:26])
    assert tile.dynamic_untrusted_mask is not None
    assert np.all(tile.dynamic_untrusted_mask[10, 22:26, 22:26])
    assert not np.any(tile.dynamic_untrusted_mask[0, 22:26, 22:26])
    assert not np.any(rep.masking.static_bad_mask[22:26, 22:26])
    assert np.all(rep.masking.transient_outlier_mask[10, 22:26, 22:26])
    # Non-saturated frames remain predominantly valid at those pixels (not wiped across the scan)
    assert float(np.mean(rep.masking.valid_mask[:, 22:26, 22:26])) > 0.65


def test_auto_select_roi_avoids_dead_quadrant() -> None:
    """When the primary quadrant is mostly untrusted/dead, auto_select_roi selects a clean quadrant."""
    pytest.importorskip("dxtbx")
    from dxtbx.imageset import ImageSet, ImageSetData, MemReader
    from dxtbx.model import Beam, Detector, Panel
    from scitbx.array_family import flex

    rng = np.random.default_rng(77)
    n_t, ny, nx = 8, 512, 512
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - 256.0, xx - 256.0) / 360.0
    mu = 5.0 + 40.0 * np.exp(-2.0 * r * r)
    imgs = rng.poisson(mu[None, :, :], size=(n_t, ny, nx)).astype(np.float64)
    # Corrupt bottom-right (+1, +1) quadrant with -1 untrusted sentinel
    imgs[:, 256:, 256:] = -1.0

    class _SimImage:
        def __init__(self, arr_2d: np.ndarray) -> None:
            self._raw = (flex.double(np.ascontiguousarray(arr_2d, dtype=np.float64)),)

        def get_raw_data(self):
            return self._raw

    iset_data = ImageSetData(MemReader([_SimImage(imgs[t]) for t in range(n_t)]), None)
    detector = Detector()
    panel = Panel()
    panel.set_name("Panel0")
    panel.set_image_size((nx, ny))
    panel.set_pixel_size((0.1, 0.1))
    panel.set_trusted_range((0.0, 65535.0))
    detector.add_panel(panel)
    beam = Beam((0.0, 0.0, -1.0), 1.0)
    for t in range(n_t):
        iset_data.set_detector(detector, t)
        iset_data.set_beam(beam, t)

    tile = extract_dxtbx_tile(
        ImageSet(iset_data),
        tile_size=(160, 160),
        max_frames=8,
        compute_dials_gain=False,
        auto_select_roi=True,
    )
    # Selected ROI must have < 5% untrusted pixels (avoiding the dead bottom-right quadrant)
    assert float(np.mean(tile.static_untrusted_mask)) < 0.05


def _write_bruker_format100_sfrm(
    path: Path,
    image_u32: np.ndarray,
    *,
    baseline: int = 64,
    subtract_baseline: bool = True,
    dettype: str = "CMOS-PHOTONII        37.037037   1.004000    0 0.425000    0.035000    1",
    wavelength: float = 1.54184,
    osc_start: float = 0.0,
    osc_inc: float = 0.5,
) -> None:
    """Write a valid Bruker FORMAT 100 `.sfrm` file with 1-byte data and u1/u2/u4 tables."""
    nrows, ncols = image_u32.shape
    flat = np.asarray(image_u32, dtype=np.uint32).ravel()

    if subtract_baseline:
        under_mask = flat <= baseline
        u_vals = flat[under_mask].astype(np.dtype("<u1"))
        shifted = np.where(under_mask, 0, flat - baseline)
        n_u = int(u_vals.size)
    else:
        u_vals = np.empty(0, dtype=np.dtype("<u1"))
        shifted = flat.copy()
        n_u = -1

    o2_mask = shifted >= 255
    o2_raw = shifted[o2_mask]
    o4_mask = shifted >= 65535
    o4_vals = shifted[o4_mask].astype(np.dtype("<u4"))
    o2_vals = np.minimum(o2_raw, 65535).astype(np.dtype("<u2"))

    pixel_u8 = np.where(o2_mask, 255, shifted).astype(np.dtype("<u1"))

    def _pad16(raw_bytes: bytes) -> bytes:
        rem = len(raw_bytes) % 16
        return raw_bytes if rem == 0 else raw_bytes + b"\x00" * (16 - rem)

    cards = [
        ("FORMAT ", "100"),
        ("VERSION", "18"),
        ("HDRBLKS", "5"),
        ("TYPE   ", "Simulated Bruker Frame"),
        ("SITE   ", "TestSite"),
        ("MODEL  ", "TestDiffractometer"),
        ("USER   ", "tester"),
        ("SAMPLE ", "crystal"),
        ("SETNAME", "scan"),
        ("RUN    ", "1"),
        ("SAMPNUM", "1"),
        ("TITLE  ", "test"),
        ("NCOUNTS", f"{int(flat.sum())} 0"),
        ("NOVERFL", f"{n_u} {int(o2_vals.size)} {int(o4_vals.size)}"),
        ("MINIMUM", f"{int(flat.min())}"),
        ("MAXIMUM", f"{int(flat.max())}"),
        ("NONTIME", "0"),
        ("NLATE  ", "0"),
        ("FILENAM", path.name),
        ("CREATED", "2026-11-11 12:00:00"),
        ("CUMULAT", "1.0"),
        ("ELAPSDR", "1.0 1.0"),
        ("ELAPSDA", "1.0 1.0"),
        ("OSCILLA", "1"),
        ("NSTEPS ", "1"),
        ("RANGE  ", f"{osc_inc:.4f}"),
        ("START  ", f"{osc_start:.4f}"),
        ("INCREME", f"{osc_inc:.4f}"),
        ("NUMBER ", "1"),
        ("NFRAMES", "1"),
        ("ANGLES ", f"0.0000 {osc_start:.4f} 0.0000 0.0000"),
        ("NOVER64", "0 0 0"),
        ("NPIXELB", "1 1"),
        ("NROWS  ", f"{nrows} 1"),
        ("NCOLS  ", f"{ncols} 1"),
        ("WORDORD", "0"),
        ("LONGORD", "0"),
        ("TARGET ", "Cu"),
        ("SOURCEK", "45.0"),
        ("SOURCEM", "1.0"),
        ("WAVELEN", f"{wavelength:.6f} {wavelength:.6f} {wavelength:.6f} {wavelength:.6f}"),
        ("DISTANC", "6.0000 6.0000"),
        ("CENTER ", f"{ncols / 2.0:.2f} {nrows / 2.0:.2f} {ncols / 2.0:.2f} {nrows / 2.0:.2f}"),
        ("DISPERSION", "0.0"),
        ("AXIS   ", "2"),
        ("DETTYPE", dettype),
        ("CCDPARM", "1.20000 36.60000 119.81560 0.00000 163810.00000"),
        ("NEXP   ", f"1 0 {baseline} 0 1"),
    ]
    # HDRBLKS = 10 -> 5120 bytes = 64 cards of 80 ASCII chars (divisible by 5)
    cards[2] = ("HDRBLKS", "10")
    while len(cards) < 63:
        cards.append(("CFR    ", ""))
    cards.append(("END    ", ""))

    header_bytes = b"".join(f"{k:<7s}:{v:<72s}"[:80].encode("ascii") for k, v in cards)
    assert len(header_bytes) == 10 * 512

    payload = bytearray(header_bytes)
    payload.extend(pixel_u8.tobytes())
    if n_u > 0:
        payload.extend(_pad16(u_vals.tobytes()))
    if o2_vals.size > 0:
        payload.extend(_pad16(o2_vals.tobytes()))
    if o4_vals.size > 0:
        payload.extend(_pad16(o4_vals.tobytes()))
    path.write_bytes(bytes(payload))


def test_legacy_bruker_format100_reader_unit(tmp_path: Path) -> None:
    """Verify legacy Bruker FORMAT 100 reader on both uncompressed .sfrm and .sfrm.bz2."""
    pytest.importorskip("dxtbx")

    expected = np.full((32, 40), 75, dtype=np.uint32)
    expected[0, 0] = 0
    expected[0, 1] = 19
    expected[0, 2] = 64
    expected[1, 0] = 65
    expected[2, 3] = 512
    expected[4, 5] = 92_000

    p_p2 = tmp_path / "photon2_0001.sfrm"
    _write_bruker_format100_sfrm(p_p2, expected, baseline=64, subtract_baseline=True)
    p_bz2 = tmp_path / "photon2_0001.sfrm.bz2"
    p_bz2.write_bytes(bz2.compress(p_p2.read_bytes()))

    for p_chk in (p_p2, p_bz2):
        leg = _read_bruker_format100_panel(str(p_chk))
        assert leg is not None
        assert np.array_equal(leg[0], expected.astype(np.float64))
        assert np.isclose(leg[1], 64.0)


def test_format_bruker_modern_905_fixes_unit(tmp_path: Path) -> None:
    """Verify all four upstream FormatBrukerModern fixes (dxtbx #906)."""
    pytest.importorskip("dxtbx")
    from dxtbx.format.FormatBrukerModern import FormatBrukerModern
    from dxtbx.model.beam import Probe

    if _needs_legacy_bruker_format100_workaround(FormatBrukerModern):
        pytest.skip("Upstream FormatBrukerModern fixes (#906) require dxtbx >= 3.32")

    # 1. Photon II X-ray with baseline=64, underflows (0, 19, 64), 2b overflow (512), 4b (92000)
    expected = np.full((32, 40), 75, dtype=np.uint32)
    expected[0, 0] = 0
    expected[0, 1] = 19
    expected[0, 2] = 64
    expected[1, 0] = 65
    expected[2, 3] = 512
    expected[4, 5] = 92_000

    p_p2 = tmp_path / "photon2_0001.sfrm"
    _write_bruker_format100_sfrm(p_p2, expected, baseline=64, subtract_baseline=True)

    iset_p2 = load_imageset(p_p2)
    panel_p2 = iset_p2.get_detector(0)[0]
    raw_p2 = iset_p2.get_raw_data(0)[0].as_numpy_array()
    assert np.array_equal(raw_p2, expected)
    assert np.isclose(panel_p2.get_pedestal(), 64.0)
    assert panel_p2.get_material() == ""
    assert np.isclose(panel_p2.get_thickness(), 0.0)
    assert np.isclose(panel_p2.get_mu(), 0.0)
    assert iset_p2.get_beam(0).get_probe() == Probe.xray

    # 2. Photon IV X-ray: DETTYPE starting with CMOS-PHOTONIV must take the Photon branch (not ED-1)
    p_p4 = tmp_path / "photon4_0001.sfrm"
    _write_bruker_format100_sfrm(
        p_p4,
        expected,
        baseline=32,
        dettype="CMOS-PHOTONIV        37.037037   1.004000    0 0.425000    0.035000    1",
        wavelength=0.71073,
    )
    assert FormatBrukerModern.understand(str(p_p4)) is True
    iset_p4 = load_imageset(p_p4)
    panel_p4 = iset_p4.get_detector(0)[0]
    assert panel_p4.get_type() == "SENSOR_CCD"
    assert np.isclose(panel_p4.get_pixel_size()[0], 5.0 / 37.037037, rtol=1e-5)
    assert np.isclose(panel_p4.get_pedestal(), 32.0)

    # 3. Electron diffraction (< 0.1 Angstrom) on QUADRO: mu = 1e10 and Probe.electron
    p_ed = tmp_path / "ed_quadro_0001.sfrm"
    _write_bruker_format100_sfrm(
        p_ed,
        expected,
        baseline=0,
        subtract_baseline=False,
        dettype="QUADRO               66.666667   1.000000    0 0.000000    0.000000    1",
        wavelength=0.0251,
    )
    iset_ed = load_imageset(p_ed)
    panel_ed = iset_ed.get_detector(0)[0]
    assert panel_ed.get_type() == "SENSOR_PAD"
    assert np.isclose(panel_ed.get_mu(), 1e10)
    assert iset_ed.get_beam(0).get_probe() == Probe.electron


def test_estimate_from_dxtbx_bruker_format100_scan_with_parenthesized_dir(tmp_path: Path) -> None:
    """End-to-end Bruker FORMAT 100 scan recovery via FormatBrukerModern with parenthesized names."""
    pytest.importorskip("dxtbx")

    det_cfg = DetectorConfig.pt_ccd_waterman_evans(
        gain=2.0, cascade_factor=1.4, pixel_factor=6.0, pedestal=64.0, psf_radius=2
    )
    det_cfg.quantize = True
    scene_cfg = SceneConfig(
        n_frames=28,
        height=72,
        width=72,
        bg_min_quanta=6.0,
        bg_max_quanta=45.0,
        n_bragg_spots=16,
        n_hot_pixels=0,
        n_dead_pixels=0,
        shadow_margin_px=0,
    )
    sim = simulate_rotation_scan(det_cfg, scene_cfg, seed=314)
    raw_u32 = np.clip(np.rint(sim.images), 0, 160_000).astype(np.uint32)
    # Inject a cold flickering sub-pedestal defective pixel (like [858, 737] in real 9p59)
    raw_u32[:, 12, 14] = np.where(np.arange(scene_cfg.n_frames) % 3 == 0, 18, 0).astype(
        np.uint32
    )

    scan_dir = tmp_path / "bruker_parenthesized"
    scan_dir.mkdir()
    for t in range(scene_cfg.n_frames):
        _write_bruker_format100_sfrm(
            scan_dir / f"9P59 ({t + 1}).sfrm",
            raw_u32[t],
            baseline=64,
            subtract_baseline=True,
            osc_start=0.5 * t,
            osc_inc=0.5,
        )

    rep, tile = estimate_from_dxtbx(scan_dir, tile_size=None, max_frames=28)
    assert np.isclose(tile.detector_pedestal, 64.0)
    assert tile.sensor_material == ""
    assert bool(rep.masking.static_bad_mask[12, 14]) is True
    assert np.isclose(rep.g_gamma, det_cfg.g_gamma, rtol=0.15)
    assert np.isclose(rep.psi, det_cfg.pixel_factor, atol=2.5)
