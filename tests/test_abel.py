"""Tests for the abel module."""
import numpy as np
import pytest

from laser_beam_analysis.abel import (
    inverse_abel_basex,
    inverse_abel_hansenlaw,
    basex_regularization_sweep,
)


def _symmetric_projection(h=121, w=121, xc=None, sigma=20.0, seed=0):
    """Symmetric Gaussian beam — a valid inverse-Abel input."""
    rng = np.random.default_rng(seed)
    if xc is None:
        xc = (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w]
    beam = np.exp(-((x - xc) ** 2 + (y - h / 2.0) ** 2) / (2 * sigma ** 2))
    img = beam + rng.normal(0, 0.002, size=(h, w))
    return img, xc


def test_basex_returns_2d_array():
    img, xc = _symmetric_projection()
    recon, transform = inverse_abel_basex(img, origin_x=xc, regularization=50)
    assert recon.ndim == 2
    assert transform is not None


def test_basex_max_is_positive_on_symmetric_input():
    img, xc = _symmetric_projection()
    recon, _ = inverse_abel_basex(img, origin_x=xc)
    assert np.max(recon) > 0


def test_hansenlaw_returns_2d_array():
    img, xc = _symmetric_projection()
    recon, transform = inverse_abel_hansenlaw(img, origin_x=xc)
    assert recon.ndim == 2
    assert transform is not None


def test_basex_and_hansenlaw_agree_on_peak_location():
    img, xc = _symmetric_projection()
    recon_bx, _ = inverse_abel_basex(img, origin_x=xc, regularization=50)
    recon_hl, _ = inverse_abel_hansenlaw(img, origin_x=xc)

    peak_bx = np.unravel_index(np.argmax(recon_bx), recon_bx.shape)
    peak_hl = np.unravel_index(np.argmax(recon_hl), recon_hl.shape)
    # Both methods should locate the peak within a few pixels of each other.
    dy = abs(peak_bx[0] - peak_hl[0])
    dx = abs(peak_bx[1] - peak_hl[1])
    assert dy <= 5 and dx <= 5, f"peaks differ: {peak_bx} vs {peak_hl}"


def test_basex_regularization_sweep_returns_all_keys():
    img, xc = _symmetric_projection()
    regs = [0, 10, 50]
    results = basex_regularization_sweep(img, origin_x=xc, regularization_values=regs)
    assert set(results.keys()) == {float(r) for r in regs}
    for reg, res in results.items():
        for key in (
            "transform",
            "image",
            "max",
            "min",
            "negative_fraction",
            "relative_negative_amplitude",
            "roughness",
        ):
            assert key in res, f"missing key {key} for reg={reg}"


def test_basex_regularization_sweep_default_values():
    img, xc = _symmetric_projection()
    results = basex_regularization_sweep(img, origin_x=xc)
    # Default list from the notebook.
    assert set(results.keys()) == {0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0}


def test_basex_rejects_3d():
    with pytest.raises(ValueError):
        inverse_abel_basex(np.zeros((10, 10, 3)), origin_x=5.0)


def test_basex_rejects_nan():
    img = np.zeros((101, 101))
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        inverse_abel_basex(img, origin_x=50.0)


def test_basex_rejects_negative_regularization():
    img, xc = _symmetric_projection()
    with pytest.raises(ValueError):
        inverse_abel_basex(img, origin_x=xc, regularization=-1.0)


def test_hansenlaw_rejects_3d():
    with pytest.raises(ValueError):
        inverse_abel_hansenlaw(np.zeros((10, 10, 3)), origin_x=5.0)