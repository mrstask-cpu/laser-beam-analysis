"""Tests for the validation module."""
import numpy as np
import pytest

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

# ------------------------------------- compare_basex_hansenlaw

from laser_beam_analysis.validation import compare_basex_hansenlaw


def _pair(basex_diams, hl_diams, zs=None):
    """Build two lists of per-slice dicts for the comparison."""
    if zs is None:
        zs = list(range(len(basex_diams)))
    basex = [
        {"z_index": z, "success": True, "diameter_1e2": d}
        for z, d in zip(zs, basex_diams)
    ]
    hl = [
        {"z_index": z, "success": True, "diameter_1e2": d}
        for z, d in zip(zs, hl_diams)
    ]
    return basex, hl


def test_compare_identical_methods_zero_difference():
    basex, hl = _pair([10.0, 12.0, 14.0], [10.0, 12.0, 14.0])
    out = compare_basex_hansenlaw(basex, hl)
    assert out["mean_difference_percent"] == 0.0
    assert out["max_difference_percent"] == 0.0
    assert len(out["z"]) == 3


def test_compare_computes_relative_difference():
    # 10 vs 11 -> |10-11| / 10.5 * 100 ≈ 9.5238...
    basex, hl = _pair([10.0, 20.0], [11.0, 20.0])
    out = compare_basex_hansenlaw(basex, hl)
    assert out["max_difference_percent"] > 0
    assert out["z"].tolist() == [0, 1]


def test_compare_uses_only_common_successful_slices():
    basex = [
        {"z_index": 0, "success": True, "diameter_1e2": 10.0},
        {"z_index": 1, "success": True, "diameter_1e2": 12.0},
        {"z_index": 2, "success": True, "diameter_1e2": 14.0},
    ]
    hl = [
        {"z_index": 0, "success": True, "diameter_1e2": 10.0},
        {"z_index": 1, "success": False},  # excluded
        {"z_index": 2, "success": True, "diameter_1e2": 14.0},
    ]
    out = compare_basex_hansenlaw(basex, hl)
    assert out["z"].tolist() == [0, 2]


def test_compare_raises_when_too_few_common():
    basex, hl = _pair([10.0], [10.0])
    with pytest.raises(RuntimeError):
        compare_basex_hansenlaw(basex, hl)


def test_compare_result_keys_complete():
    basex, hl = _pair([10.0, 12.0], [10.5, 12.2])
    out = compare_basex_hansenlaw(basex, hl)
    for key in (
        "z",
        "diameter_basex",
        "diameter_hansenlaw",
        "relative_difference_percent",
        "mean_difference_percent",
        "max_difference_percent",
    ):
        assert key in out, f"missing key: {key}"