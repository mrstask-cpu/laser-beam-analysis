"""Tests for the calibration module."""
import numpy as np
import pytest

from laser_beam_analysis.calibration import (
    length_to_um,
    um_to_unit,
    resolve_pixel_scale_um,
    calibration_summary,
    validate_calibration_config,
    convert_px_to_display,
)


def test_length_to_um_known_units():
    assert length_to_um(1, "um") == 1.0
    assert length_to_um(1, "nm") == pytest.approx(1e-3)
    assert length_to_um(1, "pm") == pytest.approx(1e-6)
    assert length_to_um(1, "A") == pytest.approx(1e-4)


def test_length_to_um_rejects_px():
    with pytest.raises(ValueError):
        length_to_um(1, "px")


def test_um_to_unit_roundtrip():
    assert um_to_unit(1.0, "nm") == pytest.approx(1000.0)
    assert um_to_unit(1.0, "um") == pytest.approx(1.0)


def test_resolve_pixel_scale_direct():
    cfg = {"enabled": True, "mode": "direct", "pixel_size": 2.5, "pixel_unit": "um"}
    assert resolve_pixel_scale_um(cfg) == pytest.approx(2.5)


def test_resolve_pixel_scale_magnification():
    cfg = {
        "enabled": True,
        "mode": "magnification",
        "camera_pixel_size_um": 5.0,
        "magnification": 2.0,
    }
    assert resolve_pixel_scale_um(cfg) == pytest.approx(2.5)


def test_resolve_pixel_scale_disabled():
    assert resolve_pixel_scale_um(None) is None
    assert resolve_pixel_scale_um({"enabled": False}) is None


def test_resolve_pixel_scale_rejects_bad_direct():
    cfg = {"enabled": True, "mode": "direct", "pixel_size": 0, "pixel_unit": "um"}
    with pytest.raises(ValueError):
        resolve_pixel_scale_um(cfg)


def test_calibration_summary_disabled():
    s = calibration_summary(None)
    assert s["enabled"] is False
    assert s["pixel_size_um"] is None
    assert s["display_unit"] == "px"


def test_calibration_summary_enabled():
    cfg = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 3.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 5.0,
        "method": "target",
    }
    s = calibration_summary(cfg)
    assert s["enabled"] is True
    assert s["pixel_size_um"] == pytest.approx(3.0)
    assert s["relative_uncertainty"] == pytest.approx(0.05)
    assert s["method"] == "target"


def test_validate_calibration_config_ok():
    cfg = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 3.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 2.0,
        "method": "target",
    }
    out = validate_calibration_config(cfg)
    assert out["valid"] is True
    assert out["pixel_size_um"] == pytest.approx(3.0)


def test_validate_calibration_config_rejects_missing_method():
    cfg = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 3.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 2.0,
    }
    with pytest.raises(ValueError):
        validate_calibration_config(cfg)


def test_validate_calibration_config_rejects_bad_uncertainty():
    cfg = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 3.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 150.0,
        "method": "target",
    }
    with pytest.raises(ValueError):
        validate_calibration_config(cfg)


def test_convert_px_to_display_disabled_returns_px():
    values = np.array([1.0, 2.0, 3.0])
    out = convert_px_to_display(values, None)
    np.testing.assert_array_equal(out, values)


def test_convert_px_to_display_enabled():
    cfg = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 2.0,
        "pixel_unit": "um",
        "display_unit": "um",
    }
    values = np.array([1.0, 2.0, 3.0])
    out = convert_px_to_display(values, cfg)
    np.testing.assert_allclose(out, [2.0, 4.0, 6.0])