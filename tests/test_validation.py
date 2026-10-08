"""Tests for the validation module."""
import numpy as np

from laser_beam_analysis.validation import scientific_audit_report


def _payload():
    alignment_info = {"axis_rms_error": 0.5}
    symmetry_metrics = {"correlation": 0.998, "normalized_rms": 0.01}
    abel_comparison = {
        "mean_difference_percent": 2.0,
        "max_difference_percent": 5.0,
    }
    propagation_bx = {"r_squared": 0.95, "delta_bic": 20.0, "z0_inside_data": True}
    propagation_hl = {"r_squared": 0.94, "delta_bic": 18.0, "z0_inside_data": True}
    waist_bx = {"z_min_observed": 50.0}
    waist_hl = {"z_min_observed": 51.0}
    profiles_bx = [{"success": True}, {"success": False}, {"success": True}]
    profiles_hl = [{"success": True}, {"success": True}, {"success": True}]
    return dict(
        alignment_info=alignment_info,
        symmetry_metrics=symmetry_metrics,
        abel_method_comparison=abel_comparison,
        propagation_basex=propagation_bx,
        propagation_hl=propagation_hl,
        waist_basex=waist_bx,
        waist_hl=waist_hl,
        profile_results_basex=profiles_bx,
        profile_results_hl=profiles_hl,
    )


def test_report_basic_numbers():
    out = scientific_audit_report(**_payload())
    assert out["axis_rms_px"] == 0.5
    assert out["symmetry_correlation"] == 0.998
    assert out["method_mean_relative_difference"] == 0.02
    assert out["method_max_relative_difference"] == 0.05
    assert out["waist_z_difference_px"] == 1.0


def test_report_fit_counts():
    out = scientific_audit_report(**_payload())
    assert out["basex_successful_fits"] == 2
    assert out["basex_total_fits"] == 3
    assert out["hl_successful_fits"] == 3
    assert out["hl_total_fits"] == 3


def test_report_propagation_fields():
    out = scientific_audit_report(**_payload())
    assert out["propagation_basex"]["r_squared"] == 0.95
    assert out["propagation_basex"]["z0_inside_data"] is True
    assert out["propagation_hl"]["delta_bic"] == 18.0


def test_report_handles_missing_keys():
    empty = {}
    results_empty = []
    out = scientific_audit_report(
        alignment_info=empty,
        symmetry_metrics=empty,
        abel_method_comparison=empty,
        propagation_basex=empty,
        propagation_hl=empty,
        waist_basex={"z_min_observed": 0.0},
        waist_hl={"z_min_observed": 0.0},
        profile_results_basex=results_empty,
        profile_results_hl=results_empty,
    )
    assert np.isnan(out["axis_rms_px"])
    assert np.isnan(out["symmetry_correlation"])
    assert np.isnan(out["method_mean_relative_difference"])
    assert out["basex_successful_fits"] == 0


def test_report_passes_optional_metadata_through():
    out = scientific_audit_report(
        **_payload(),
        calibration_info={"enabled": True, "pixel_size_um": 2.0},
        saturation_info={"status": "OK"},
        uncertainty_budget={"status": "available"},
        side_consistency_3d={"normalized_rms_difference": 0.01, "correlation": 0.99},
    )
    assert out["calibration"]["enabled"] is True
    assert out["saturation"]["status"] == "OK"
    assert out["uncertainty_budget"]["status"] == "available"
    assert out["side_consistency_3d"]["correlation"] == 0.99