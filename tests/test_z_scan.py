"""Tests for the z_scan module."""
import numpy as np
import pytest

from laser_beam_analysis.z_scan import (
    make_z_sampling,
    get_z_positions,
    analyze_all_z_profiles,
)


def _gaussian_abel_image(h=81, w=81, sigma=12.0, seed=0):
    """Synthetic 2D image that mimics an Abel reconstruction."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    img = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma ** 2))
    img = img + rng.normal(0, 0.001, size=(h, w))
    return img


# ---------------------------------------------------------- z-sampling

def test_make_z_sampling_basic():
    pos = make_z_sampling(n_z=100, n_points=11, margin=10)
    assert pos.dtype.kind == "i"
    assert len(pos) == 11
    assert pos[0] == 10
    assert pos[-1] == 89
    assert np.all(np.diff(pos) > 0)


def test_make_z_sampling_rejects_small_n_z():
    with pytest.raises(ValueError):
        make_z_sampling(n_z=2)


def test_make_z_sampling_rejects_small_n_points():
    with pytest.raises(ValueError):
        make_z_sampling(n_z=100, n_points=1)


def test_make_z_sampling_rejects_negative_margin():
    with pytest.raises(ValueError):
        make_z_sampling(n_z=100, margin=-1)


def test_make_z_sampling_rejects_too_large_margin():
    with pytest.raises(ValueError):
        make_z_sampling(n_z=10, margin=10)


def test_make_z_sampling_rejects_too_many_points():
    with pytest.raises(ValueError):
        make_z_sampling(n_z=10, n_points=50, margin=2)


def test_get_z_positions_diagnostic():
    pos = get_z_positions(n_z=100, mode="diagnostic", margin=10)
    assert len(pos) == 15


def test_get_z_positions_dense():
    pos = get_z_positions(n_z=100, mode="dense", n_points=21, margin=10)
    assert len(pos) == 21


def test_get_z_positions_full():
    pos = get_z_positions(n_z=50, mode="full", margin=5)
    assert pos[0] == 5
    assert pos[-1] == 44
    assert len(pos) == 40


def test_get_z_positions_rejects_unknown_mode():
    with pytest.raises(ValueError):
        get_z_positions(n_z=100, mode="superdense")


# ---------------------------------------------------- batch analysis

def test_analyze_all_z_profiles_basic():
    img = _gaussian_abel_image()
    z = np.array([20, 40, 60])
    results = analyze_all_z_profiles(img, z)
    assert len(results) == 3
    for item in results:
        assert item["success"] is True
        assert "sigma" in item
        assert "z_index" in item


def test_analyze_all_z_profiles_keys_present_on_failure():
    """Even when a fit fails, all numeric fields must be present (NaN)."""
    img = _gaussian_abel_image()
    # z_index out of range -> extract_radial_profile raises IndexError,
    # which the batch must catch.
    z = np.array([40, 10000])
    results = analyze_all_z_profiles(img, z)
    assert len(results) == 2
    failed = results[1]
    assert failed["success"] is False
    assert failed["error"] is not None
    assert np.isnan(failed["sigma"])
    assert np.isnan(failed["beam_radius_w"])


def test_analyze_all_z_profiles_raises_if_all_fail():
    img = _gaussian_abel_image()
    z = np.array([10000, 20000])
    with pytest.raises(RuntimeError):
        analyze_all_z_profiles(img, z)


def test_analyze_all_z_profiles_rejects_3d_input():
    with pytest.raises(ValueError):
        analyze_all_z_profiles(np.zeros((10, 10, 3)), np.array([5]))


def test_analyze_all_z_profiles_rejects_nan():
    img = _gaussian_abel_image()
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyze_all_z_profiles(img, np.array([40]))


def test_analyze_recovers_sigma_across_slices():
    sigma_true = 12.0
    img = _gaussian_abel_image(sigma=sigma_true)
    z = np.array([20, 40, 60])
    results = analyze_all_z_profiles(img, z)
    for item in results:
        assert item["success"] is True
        # The profile at these z-slices is a clean Gaussian of the same width.
        assert abs(item["sigma"] - sigma_true) < 1.5