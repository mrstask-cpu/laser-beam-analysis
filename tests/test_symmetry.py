"""Tests for the symmetry module."""
import numpy as np
import pytest

from laser_beam_analysis.symmetry import calculate_left_right_symmetry


def _symmetric_beam(h=200, w=200, xc=None, sigma=25.0, seed=0):
    rng = np.random.default_rng(seed)
    if xc is None:
        xc = w / 2.0
    y, x = np.mgrid[0:h, 0:w]
    beam = np.exp(-((x - xc) ** 2 + (y - h / 2.0) ** 2) / (2 * sigma ** 2))
    return beam + rng.normal(0, 0.001, size=(h, w)), xc


def test_symmetric_input_gives_low_normalized_rms():
    img, xc = _symmetric_beam()
    m = calculate_left_right_symmetry(img, center_x=xc)
    assert m["normalized_rms"] < 0.05


def test_symmetric_input_gives_high_correlation():
    img, xc = _symmetric_beam()
    m = calculate_left_right_symmetry(img, center_x=xc)
    assert m["correlation"] > 0.99


def test_asymmetric_input_lowers_correlation():
    img, xc = _symmetric_beam()
    # Shift a chunk of intensity to one side to break the symmetry.
    img[:, : int(xc)] *= 0.5
    m = calculate_left_right_symmetry(img, center_x=xc)
    assert m["correlation"] < 0.95


def test_wrong_center_increases_rms():
    img, xc = _symmetric_beam()
    m_true = calculate_left_right_symmetry(img, center_x=xc)
    m_off = calculate_left_right_symmetry(img, center_x=xc + 10.0)
    assert m_off["normalized_rms"] > m_true["normalized_rms"]


def test_radial_limit_restricts_points():
    img, xc = _symmetric_beam()
    m_full = calculate_left_right_symmetry(img, center_x=xc)
    m_limited = calculate_left_right_symmetry(img, center_x=xc, radial_limit=20.0)
    assert m_limited["number_of_points"] < m_full["number_of_points"]


def test_noise_floor_reduces_points():
    img, xc = _symmetric_beam()
    m_all = calculate_left_right_symmetry(img, center_x=xc)
    m_thresh = calculate_left_right_symmetry(img, center_x=xc, noise_floor=0.5)
    assert m_thresh["number_of_points"] < m_all["number_of_points"]


def test_noise_floor_too_high_raises():
    img, xc = _symmetric_beam()
    with pytest.raises(ValueError):
        calculate_left_right_symmetry(img, center_x=xc, noise_floor=1e6)


def test_rejects_3d_input():
    with pytest.raises(ValueError):
        calculate_left_right_symmetry(np.zeros((10, 10, 3)), center_x=5.0)


def test_rejects_nan():
    img = np.zeros((100, 100))
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        calculate_left_right_symmetry(img, center_x=50.0)


def test_rejects_bad_radial_limit():
    img, xc = _symmetric_beam()
    with pytest.raises(ValueError):
        calculate_left_right_symmetry(img, center_x=xc, radial_limit=0)


def test_metrics_keys_complete():
    img, xc = _symmetric_beam()
    m = calculate_left_right_symmetry(img, center_x=xc)
    for key in (
        "normalized_rms",
        "correlation",
        "median_relative_difference",
        "p95_relative_difference",
        "number_of_points",
    ):
        assert key in m, f"missing metric: {key}"