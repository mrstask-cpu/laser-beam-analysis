"""Radiometry, PSF correction, and sensitivity-based uncertainty budget.

Public API:
    radiometric_convert_intensity(values, radiometry_config) -> np.ndarray
    gaussian_psf_correct_width(width_px, psf_width_px) -> float
    apply_psf_metadata(profile_results, psf_config) -> list
    build_sensitivity_uncertainty_budget(reference_value, ...) -> dict

Design notes:
    - Radiometric conversion requires an externally determined factor; the
      module does not assume it automatically.
    - PSF correction uses the standard Gaussian model w_obs² ≈ w_true² + w_psf².
      If w_psf ≥ w_obs, w_true is not identifiable and NaN is returned.
    - The uncertainty budget is an engineering sensitivity envelope assembled
      from observed parameter ranges. It is NOT a metrologically traceable
      ISO/GUM uncertainty budget.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np


def radiometric_convert_intensity(values, radiometry_config) -> np.ndarray:
    """Optionally convert relative / calibrated signal into W/cm².

    The conversion factor must be obtained by an independent radiometric
    calibration. If radiometry_config is None or disabled, the input is
    returned unchanged.

    Args:
        values: intensity array.
        radiometry_config: dict with keys:
            enabled         – bool
            factor_to_W_cm2 – float > 0 (required if enabled)

    Returns:
        Converted array, or the input cast to float if disabled.
    """
    values = np.asarray(values, dtype=float)
    if radiometry_config is None or not radiometry_config.get("enabled", False):
        return values
    factor = radiometry_config.get("factor_to_W_cm2", None)
    if factor is None or float(factor) <= 0:
        raise ValueError(
            "Для radiometry.enabled=True нужен factor_to_W_cm2 > 0."
        )
    return values * float(factor)


def gaussian_psf_correct_width(width_px: float, psf_width_px: float) -> float:
    """Apply a Gaussian-PSF deconvolution to a beam width.

        w_obs² ≈ w_true² + w_psf²

    Args:
        width_px: observed beam width (in pixels).
        psf_width_px: Gaussian PSF width (in pixels).

    Returns:
        Corrected width, or NaN if the observation is not wider than the PSF.
    """
    width_px = float(width_px)
    psf_width_px = float(psf_width_px)
    if width_px < 0 or psf_width_px < 0:
        raise ValueError("Ширины должны быть неотрицательными.")
    corrected_sq = width_px ** 2 - psf_width_px ** 2
    if corrected_sq <= 0:
        return float("nan")
    return float(np.sqrt(corrected_sq))


def apply_psf_metadata(profile_results, psf_config) -> list:
    """Return a deep copy of profile results with PSF-corrected widths added.

    Original measurements are preserved; corrected values are added under
    beam_radius_w_psf_corrected and diameter_1e2_psf_corrected.

    Args:
        profile_results: list of per-slice dicts.
        psf_config: dict with keys:
            enabled  – bool
            w_psf_px – float > 0 (required if enabled)

    Returns:
        New list of dicts (original input is not modified).
    """
    results = deepcopy(profile_results)

    if not psf_config.get("enabled", False):
        for item in results:
            item["psf_correction_enabled"] = False
        return results

    psf_w_px = psf_config.get("w_psf_px", None)
    if psf_w_px is None or float(psf_w_px) <= 0:
        raise ValueError(
            "При включённой PSF correction требуется w_psf_px > 0."
        )

    for item in results:
        if item.get("success", False):
            w_obs = float(item.get("beam_radius_w", np.nan))
            w_corr = gaussian_psf_correct_width(w_obs, psf_w_px)
            item["psf_correction_enabled"] = True
            item["w_psf_px"] = float(psf_w_px)
            item["beam_radius_w_psf_corrected"] = w_corr
            item["diameter_1e2_psf_corrected"] = (
                2.0 * w_corr if np.isfinite(w_corr) else float("nan")
            )
        else:
            item["psf_correction_enabled"] = True
            item["beam_radius_w_psf_corrected"] = float("nan")
            item["diameter_1e2_psf_corrected"] = float("nan")
    return results


def build_sensitivity_uncertainty_budget(
    reference_value: float,
    sigma_results: Optional[list] = None,
    center_results: Optional[list] = None,
    regularization_results: Optional[list] = None,
    method_relative_difference: float = 0.0,
    calibration_relative_uncertainty: float = 0.0,
) -> dict:
    """Engineering sensitivity envelope assembled from observed parameter ranges.

    Each sensitivity family contributes one component, defined as
    (max - min) / (2 * |reference|), which corresponds to a half-range relative
    uncertainty. Components are combined in quadrature (RSS). The result also
    reports an expanded (k=2) envelope.

    This is NOT a metrologically traceable ISO/GUM uncertainty budget.

    Args:
        reference_value: physical quantity used as the denominator (e.g. the
            beam diameter at waist).
        sigma_results, center_results, regularization_results: lists of dicts
            each containing a "diameter_1e2" field (center_results may also
            include a "success" flag).
        method_relative_difference: relative difference between BASEX and
            Hansen–Law diameters; half of it is used as a component.
        calibration_relative_uncertainty: relative calibration uncertainty.

    Returns:
        dict with status, reference_value, relative envelope, absolute
        envelope at k=2, components breakdown, and a traceability flag.
    """
    ref = float(reference_value)
    if not np.isfinite(ref) or ref == 0:
        return {"status": "not_available"}

    components: dict = {}

    if sigma_results:
        vals = np.array(
            [x.get("diameter_1e2", np.nan) for x in sigma_results], dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            components["preprocessing_sigma"] = float(
                (np.max(vals) - np.min(vals)) / (2 * abs(ref))
            )

    if center_results:
        vals = np.array(
            [
                x.get("diameter_1e2", np.nan)
                for x in center_results
                if x.get("success", False)
            ],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            components["abel_center"] = float(
                (np.max(vals) - np.min(vals)) / (2 * abs(ref))
            )

    if regularization_results:
        vals = np.array(
            [x.get("diameter_1e2", np.nan) for x in regularization_results],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            components["basex_regularization"] = float(
                (np.max(vals) - np.min(vals)) / (2 * abs(ref))
            )

    components["method_difference"] = abs(float(method_relative_difference)) / 2.0
    components["calibration"] = abs(float(calibration_relative_uncertainty))

    u_combined_rel = float(np.sqrt(np.sum(np.square(list(components.values())))))

    return {
        "status": "available",
        "reference_value": ref,
        "relative_sensitivity_envelope_rss": u_combined_rel,
        "sensitivity_envelope_k2": 2.0 * u_combined_rel,
        "absolute_sensitivity_envelope_k2": 2.0 * u_combined_rel * abs(ref),
        "components": components,
        "is_traceable_gum_uncertainty": False,
        "note": (
            "Engineering sensitivity envelope assembled from observed "
            "parameter ranges; not a metrologically traceable ISO/GUM "
            "uncertainty budget."
        ),
    }