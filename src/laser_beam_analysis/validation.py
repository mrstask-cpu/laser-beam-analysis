"""Scientific QA/QC report assembled from pipeline results.

Public API:
    scientific_audit_report(
        alignment_info, symmetry_metrics, abel_method_comparison,
        propagation_basex, propagation_hl,
        waist_basex, waist_hl,
        profile_results_basex, profile_results_hl,
        calibration_info=None, saturation_info=None,
        uncertainty_budget=None, side_consistency_3d=None,
    ) -> dict

The report keeps two physically distinct concepts separate:
    - "observed waist"   — the discrete minimum of w(z) (find_beam_waist)
    - "model-fitted z0"  — the propagation-model waist position
These must not be conflated in downstream reporting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def scientific_audit_report(
    alignment_info,
    symmetry_metrics,
    abel_method_comparison,
    propagation_basex,
    propagation_hl,
    waist_basex,
    waist_hl,
    profile_results_basex,
    profile_results_hl,
    calibration_info: Optional[dict] = None,
    saturation_info: Optional[dict] = None,
    uncertainty_budget: Optional[dict] = None,
    side_consistency_3d: Optional[dict] = None,
) -> dict:
    """Build the QA/QC report dict.

    All numeric fields are given as plain floats. Where an input dict does
    not carry the required key, the corresponding value falls back to NaN.
    """
    axis_rms = float(alignment_info.get("axis_rms_error", np.nan))
    corr = float(symmetry_metrics.get("correlation", np.nan))
    asym = float(symmetry_metrics.get("normalized_rms", np.nan))

    mean_diff = float(
        abel_method_comparison.get("mean_difference_percent", np.nan) / 100.0
    )
    max_diff = float(
        abel_method_comparison.get("max_difference_percent", np.nan) / 100.0
    )

    dz = abs(
        float(waist_basex["z_min_observed"]) - float(waist_hl["z_min_observed"])
    )

    basex_success = sum(
        x.get("success", False) for x in profile_results_basex
    )
    hl_success = sum(x.get("success", False) for x in profile_results_hl)

    report = {
        "axis_rms_px": axis_rms,
        "symmetry_correlation": corr,
        "symmetry_rms_asymmetry": asym,
        "method_mean_relative_difference": mean_diff,
        "method_max_relative_difference": max_diff,
        "waist_z_difference_px": dz,
        "basex_successful_fits": int(basex_success),
        "basex_total_fits": int(len(profile_results_basex)),
        "hl_successful_fits": int(hl_success),
        "hl_total_fits": int(len(profile_results_hl)),
        "propagation_basex": {
            "r_squared": float(propagation_basex.get("r_squared", np.nan)),
            "delta_bic": float(propagation_basex.get("delta_bic", np.nan)),
            "z0_inside_data": bool(propagation_basex.get("z0_inside_data", False)),
        },
        "propagation_hl": {
            "r_squared": float(propagation_hl.get("r_squared", np.nan)),
            "delta_bic": float(propagation_hl.get("delta_bic", np.nan)),
            "z0_inside_data": bool(propagation_hl.get("z0_inside_data", False)),
        },
        "calibration": calibration_info,
        "saturation": saturation_info,
        "uncertainty_budget": uncertainty_budget,
        "side_consistency_3d": side_consistency_3d,
    }
    return report

def compare_basex_hansenlaw(
    basex_results,
    hansenlaw_results,
) -> dict:
    """Compare BASEX and Hansen–Law reconstructions via Gaussian diameters.

    Only z-slices that succeeded under BOTH methods are compared. Failure
    of an individual slice excludes it from the comparison but does not abort
    the rest of the analysis.

    Args:
        basex_results: list of per-slice dicts from analyze_all_z_profiles
            (BASEX branch).
        hansenlaw_results: same, Hansen–Law branch.

    Returns:
        dict with:
            z                              – common z indices (numpy array)
            diameter_basex, diameter_hansenlaw – D(1/e²) arrays over common z
            relative_difference_percent    – per-slice |Δ|/mean * 100
            mean_difference_percent
            max_difference_percent

    Raises:
        RuntimeError: if fewer than 2 common successful slices exist.
    """
    basex_map = {
        item["z_index"]: item
        for item in basex_results
        if item.get("success", False)
    }
    hl_map = {
        item["z_index"]: item
        for item in hansenlaw_results
        if item.get("success", False)
    }

    common_z = sorted(set(basex_map) & set(hl_map))
    if len(common_z) < 2:
        raise RuntimeError(
            "Недостаточно общих успешных сечений для сравнения."
        )

    d_basex = np.array(
        [basex_map[z]["diameter_1e2"] for z in common_z], dtype=float
    )
    d_hl = np.array(
        [hl_map[z]["diameter_1e2"] for z in common_z], dtype=float
    )

    diff_percent = (
        100.0
        * np.abs(d_basex - d_hl)
        / np.maximum(
            0.5 * (np.abs(d_basex) + np.abs(d_hl)),
            np.finfo(float).eps,
        )
    )

    mean_difference = float(np.mean(diff_percent))
    max_difference = float(np.max(diff_percent))

    return {
        "z": np.asarray(common_z),
        "diameter_basex": d_basex,
        "diameter_hansenlaw": d_hl,
        "relative_difference_percent": diff_percent,
        "mean_difference_percent": mean_difference,
        "max_difference_percent": max_difference,
    }