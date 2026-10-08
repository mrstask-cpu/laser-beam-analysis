"""Tests for the abel_center module."""
import numpy as np
import pytest

from laser_beam_analysis.abel_center import (
    find_abel_center,
    compare_abel_center_methods,
)


def _symmetric_beam(h=200, w=200, xc=None, sigma=25.0, noise=0.01, seed=0):
    """Symmetric Gaussian beam centered at xc on a uniform background."""
    rng = np.random.default_rng(seed)
    if xc is None:
        xc = w / 2.0
    y, x = np.mgrid[0:h, 0:w]
    beam = np.exp(-((x - xc) ** 2 + (y - h / 2.0) ** 2) / (2 * sigma ** 2))
    img = beam + rng.normal(0, noise, size=(h, w))
    return img, xc


def test_find_abel_center_recovers_known_x():
    img, xc_true = _symmetric_beam(xc=100.0)
    _, center_x = find_abel_center(img, slice_width=10)
    assert abs(center_x - xc_true) < 2.0, f"got {center_x}, expected ~{xc_true}"


def test_find_abel_center_off_center():
    img, xc_true = _symmetric_beam(xc=120.0)
    _, center_x = find_abel_center(img, slice_width=10)
    assert abs(center_x - xc_true) < 2.0


def test_find_abel_center_returns_tuple_pair():
    img, _ = _symmetric_beam()
    center_xy, center_x = find_abel_center(img)
    assert isinstance(center_xy, tuple)
    assert len(center_xy) == 2
    assert center_xy[1] == pytest.approx(center_x)


def test_find_abel_center_rejects_3d():
    with pytest.raises(ValueError):
        find_abel_center(np.zeros((10, 10, 3)))


def test_find_abel_center_rejects_nan():
    img = np.zeros((100, 100))
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        find_abel_center(img)


def test_find_abel_center_rejects_bad_slice_width():
    img, _ = _symmetric_beam(h=100, w=100)
    with pytest.raises(ValueError):
        find_abel_center(img, slice_width=0)


def test_compare_abel_center_methods_returns_all_methods():
    img, _ = _symmetric_beam()
    results = compare_abel_center_methods(img)
    for method in ("image_center", "com", "convolution", "gaussian", "slice"):
        assert method in results, f"missing method: {method}"
        assert "x" in results[method]
        assert "y" in results[method]


def test_compare_abel_center_methods_agree_on_symmetric_input():
    img, xc_true = _symmetric_beam(xc=100.0, noise=0.005)
    results = compare_abel_center_methods(img)

    finite_x = [
        r["x"] for r in results.values()
        if np.isfinite(r["x"])
    ]
    assert len(finite_x) >= 3, "too few methods produced a finite result"
    # All methods should be within ~3 px of the true center on clean symmetric data.
    for x in finite_x:
        assert abs(x - xc_true) < 3.0, f"method returned {x}, expected ~{xc_true}"