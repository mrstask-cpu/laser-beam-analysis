"""Tests for the camera module."""
import numpy as np
import pytest

from laser_beam_analysis.camera import (
    apply_camera_corrections,
    acquisition_saturation_diagnostics,
)


def test_apply_camera_corrections_disabled():
    img = np.arange(100, dtype=np.float64).reshape(10, 10) + 1.0
    out, info = apply_camera_corrections(img)
    np.testing.assert_array_equal(out, img)
    assert info["dark_enabled"] is False
    assert info["flat_enabled"] is False


def test_apply_camera_corrections_dark():
    img = np.full((5, 5), 10.0)
    dark = np.full((5, 5), 3.0)
    out, info = apply_camera_corrections(img, dark_frame=dark)
    np.testing.assert_allclose(out, 7.0)
    assert info["dark_enabled"] is True


def test_apply_camera_corrections_flat():
    img = np.full((5, 5), 10.0)
    flat = np.full((5, 5), 2.0)
    out, info = apply_camera_corrections(img, flat_field=flat)
    np.testing.assert_allclose(out, 10.0)
    assert info["flat_enabled"] is True


def test_apply_camera_corrections_shape_mismatch():
    img = np.zeros((5, 5))
    dark = np.zeros((3, 3))
    with pytest.raises(ValueError):
        apply_camera_corrections(img, dark_frame=dark)


def test_apply_camera_corrections_bad_flat_produces_nan():
    img = np.ones((5, 5))
    flat = np.zeros((5, 5))
    with pytest.raises(ValueError):
        apply_camera_corrections(img, flat_field=flat)


def test_saturation_diagnostics_integer_source():
    raw = np.full((10, 10), 255, dtype=np.uint8)
    img = raw.astype(np.float64)
    diag = acquisition_saturation_diagnostics(raw, img)
    assert diag["available"] is True
    assert diag["bit_depth"] == 8
    assert diag["saturated_fraction"] == 1.0
    assert diag["status"] == "WARNING_SATURATION"


def test_saturation_diagnostics_no_saturation():
    raw = np.zeros((10, 10), dtype=np.uint8)
    img = np.ones((10, 10))  # avoid uniform-float rejection elsewhere
    diag = acquisition_saturation_diagnostics(raw, img)
    assert diag["available"] is True
    assert diag["saturated_fraction"] == 0.0
    assert diag["status"] == "OK"


def test_saturation_diagnostics_float_source_not_available():
    raw = np.zeros((10, 10), dtype=np.float64)
    img = np.ones((10, 10))
    diag = acquisition_saturation_diagnostics(raw, img)
    assert diag["available"] is False
    assert diag["status"] == "not_available_for_noninteger_source"