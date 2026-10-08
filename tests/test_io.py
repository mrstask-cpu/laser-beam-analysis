"""Smoke tests for the io module."""
import numpy as np
import pytest

from laser_beam_analysis.io import load_image, _require_finite_2d


def test_require_finite_2d_ok():
    arr = np.zeros((10, 10))
    out = _require_finite_2d(arr)
    assert out.shape == (10, 10)
    assert out.dtype == np.float64


def test_require_finite_2d_rejects_3d():
    with pytest.raises(ValueError):
        _require_finite_2d(np.zeros((10, 10, 3)))


def test_require_finite_2d_rejects_nan():
    arr = np.zeros((10, 10))
    arr[0, 0] = np.nan
    with pytest.raises(ValueError):
        _require_finite_2d(arr)


def test_load_image_from_array_roundtrip():
    arr = np.arange(100, dtype=np.float64).reshape(10, 10)
    out = load_image(arr)
    assert out.shape == (10, 10)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, arr)


def test_load_image_rejects_uniform():
    with pytest.raises(ValueError):
        load_image(np.ones((10, 10)))


def test_load_image_rejects_bad_type():
    with pytest.raises(RuntimeError):
        load_image(12345)