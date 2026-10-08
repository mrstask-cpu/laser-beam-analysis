"""Tests for the metrology module."""
import numpy as np
import pytest

from laser_beam_analysis.metrology import (
    radiometric_convert_intensity,
    gaussian_psf_correct_width,
    apply_psf_metadata,
    build_sensitivity_uncertainty_budget,
)


# --------------------------------------------------------- radiometry

def test_radiometric_disabled_returns_input():
    v = np.array([1.0, 2.0, 3.0])
    out = radiometric_convert_intensity(v, None)
    np.testing.assert_array_equal(out, v)
    out = radiometric_convert_intensity(v, {"enabled": False})
    np.testing.assert_array_equal(out, v)


def test_radiometric_applies_factor():
    v = np.array([1.0, 2.0, 3.0])
    out = radiometric_convert_intensity(v, {"enabled": True, "factor_to_W_cm2": 0.5})
    np.testing.assert_allclose(out, [0.5, 1.0, 1.5])


def test_radiometric_rejects_missing_factor():
    with pytest.raises(ValueError):
        radiometric_convert_intensity(np.array([1.0]), {"enabled": True})


def test_radiometric_rejects_bad_factor():
    with pytest.raises(ValueError):
        radiometric_convert_intensity(
            np.array([1.0]), {"enabled": True, "factor_to_W_cm2": -0.1}
        )


# ---------------------------------------------------------------- PSF

def test_psf_correct_width_normal():
    # w_obs = 5, w_psf = 3 -> w_true = 4
    assert gaussian_psf_correct_width(5.0, 3.0) == pytest.approx(4.0)


def test_psf_correct_width_returns_nan_when_psf_too_large():
    assert np.isnan(gaussian_psf_correct_width(2.0, 3.0))


def test_psf_correct_width_rejects_negative():
    with pytest.raises(ValueError):
        gaussian_psf_correct_width(-1.0, 3.0)
    with pytest.raises(ValueError):
        gaussian_psf_correct_width(5.0, -3.0)


def test_apply_psf_metadata_disabled():
    results = [
        {"success": True, "beam_radius_w": 5.0},
        {"success": False},
    ]
    out = apply_psf_metadata(results, {"enabled": False})
    for item in out:
        assert item["psf_correction_enabled"] is False


def test_apply_psf_metadata_enabled_adds_corrected_values():
    results = [
        {"success": True, "beam_radius_w": 5.0},
        {"success": False},
    ]
    out = apply_psf_metadata(results, {"enabled": True, "w_psf_px": 3.0})
    assert out[0]["psf_correction_enabled"] is True
    assert out[0]["beam_radius_w_psf_corrected"] == pytest.approx(4.0)
    assert out[0]["diameter_1e2_psf_corrected"] == pytest.approx(8.0)
    assert np.isnan(out[1]["beam_radius_w_psf_corrected"])


def test_apply_psf_metadata_does_not_mutate_input():
    results = [{"success": True, "beam_radius_w": 5.0}]
    _ = apply_psf_metadata(results, {"enabled": True, "w_psf_px": 3.0})
    assert "beam_radius_w_psf_corrected" not in results[0]


def test_apply_psf_metadata_rejects_missing_w_psf():
    results = [{"success": True, "beam_radius_w": 5.0}]
    with pytest.raises(ValueError):
        apply_psf_metadata(results, {"enabled": True})


# ---------------------------------------------- uncertainty budget

def test_budget_not_available_for_invalid_reference():
    out = build_sensitivity_uncertainty_budget(0.0)
    assert out["status"] == "not_available"
    out = build_sensitivity_uncertainty_budget(float("nan"))
    assert out["status"] == "not_available"


def test_budget_basic_no_components():
    out = build_sensitivity_uncertainty_budget(10.0)
    assert out["status"] == "available"
    assert out["components"]["method_difference"] == 0.0
    assert out["components"]["calibration"] == 0.0
    assert out["relative_sensitivity_envelope_rss"] == 0.0
    assert out["is_traceable_gum_uncertainty"] is False


def test_budget_includes_method_and_calibration():
    out = build_sensitivity_uncertainty_budget(
        10.0, method_relative_difference=0.04, calibration_relative_uncertainty=0.03
    )
    # method_difference = 0.02, calibration = 0.03
    expected_rss = float(np.sqrt(0.02 ** 2 + 0.03 ** 2))
    assert out["relative_sensitivity_envelope_rss"] == pytest.approx(expected_rss)


def test_budget_with_sigma_results():
    sigma_results = [
        {"diameter_1e2": 10.0},
        {"diameter_1e2": 11.0},
    ]
    out = build_sensitivity_uncertainty_budget(10.0, sigma_results=sigma_results)
    # half-range / |ref| = (11 - 10) / (2 * 10) = 0.05
    assert out["components"]["preprocessing_sigma"] == pytest.approx(0.05)


def test_budget_skips_failed_center_results():
    center_results = [
        {"diameter_1e2": 10.0, "success": True},
        {"diameter_1e2": 15.0, "success": False},  # must be ignored
    ]
    out = build_sensitivity_uncertainty_budget(10.0, center_results=center_results)
    # single valid point -> min == max -> component is 0.0
    assert out["components"]["abel_center"] == pytest.approx(0.0)


def test_budget_absolute_envelope_k2():
    out = build_sensitivity_uncertainty_budget(
        10.0, method_relative_difference=0.04
    )
    # RSS = 0.02, k2 = 0.04, absolute = 0.04 * 10 = 0.4
    assert out["absolute_sensitivity_envelope_k2"] == pytest.approx(0.4)