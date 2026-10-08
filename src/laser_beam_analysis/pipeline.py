"""End-to-end analysis pipeline for a single beam image.

Public API:
    analyze_beam(source, config=None) -> BeamAnalysisResult

The pipeline runs the following stages in order:
    load -> camera corrections -> calibration -> alignment
         -> preprocessing -> Abel center -> symmetry check
         -> BASEX + Hansen-Law -> z-scan Gaussian fits
         -> observed waist + propagation fits
         -> cross-method comparison -> QA/QC report

Each stage is optional to a degree: nothing is forced beyond what the
config enables. If a stage produces an error and is marked optional, the
corresponding field of the result is left empty.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .abel import inverse_abel_basex, inverse_abel_hansenlaw
from .abel_center import find_abel_center
from .alignment import align_laser_beam
from .calibration import (
    calibration_summary,
    resolve_pixel_scale_um,
    validate_calibration_config,
)
from .camera import (
    acquisition_saturation_diagnostics,
    apply_camera_corrections,
)
from .io import load_image
from .metrology import apply_psf_metadata, build_sensitivity_uncertainty_budget
from .preprocessing import preprocess_background_and_noise
from .propagation import find_beam_waist, fit_gaussian_beam_propagation
from .results import BeamAnalysisResult
from .symmetry import calculate_left_right_symmetry
from .validation import compare_basex_hansenlaw, scientific_audit_report
from .z_scan import analyze_all_z_profiles, get_z_positions


_DEFAULT_CONFIG: dict = {
    "camera": {
        "dark_frame": None,
        "flat_field": None,
        "flat_floor": 1e-6,
    },
    "calibration": None,
    "radiometry": None,
    "psf": {"enabled": False},
    "alignment": {
        "threshold_ratio": 0.15,
        "min_snr": 3.0,
        "median_size": 3,
    },
    "preprocessing": {
        "bg_corner_size": 40,
        "sigma": 1.5,
    },
    "abel_center": {
        "slice_width": 10,
    },
    "symmetry": {
        "radial_limit": None,
        "noise_floor": None,
    },
    "abel": {
        "regularization": 50,
        "symmetry_axis": 0,
    },
    "z_scan": {
        "mode": "dense",
        "n_points": 61,
        "margin": 10,
    },
    "profiles": {
        "fit_fraction": 0.05,
        "robust": True,
        "r0_max": 3.0,
    },
    "propagation": {
        "robust": True,
    },
}


def _merge_config(config: Optional[dict]) -> dict:
    """Merge the user config over the defaults (shallow per top-level key)."""
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in _DEFAULT_CONFIG.items()}
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _optional_call(fn, *args, **kwargs):
    """Run fn; return None if it raises. Used for stages that are optional."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def analyze_beam(source, config: Optional[dict] = None) -> BeamAnalysisResult:
    """Run the full beam-analysis pipeline on a single image.

    Args:
        source: path, URL, or 2D numpy array.
        config: optional dict overriding any default. See _DEFAULT_CONFIG for
            the accepted structure. All keys are optional.

    Returns:
        BeamAnalysisResult populated with the stages that succeeded.
    """
    cfg = _merge_config(config)
    result = BeamAnalysisResult(config=cfg)

    # 1. Load
    raw = load_image(source)
    result.raw_image = raw
    result.image_shape = tuple(raw.shape)

    # 2. Camera corrections
    corrected, cam_info = apply_camera_corrections(
        raw,
        dark_frame=cfg["camera"].get("dark_frame"),
        flat_field=cfg["camera"].get("flat_field"),
        flat_floor=cfg["camera"].get("flat_floor", 1e-6),
    )
    result.camera = dict(cam_info)
    result.camera["saturation"] = acquisition_saturation_diagnostics(source, raw)

    # 3. Calibration validation
    calibration = cfg.get("calibration")
    if calibration is not None and calibration.get("enabled", False):
        validate_calibration_config(calibration)
    result.metrology = {
        "calibration": calibration_summary(calibration),
        "pixel_size_um": resolve_pixel_scale_um(calibration),
    }

    # 4. Alignment
    aligned, angle_deg, alignment_info = align_laser_beam(
        corrected,
        threshold_ratio=cfg["alignment"]["threshold_ratio"],
        min_snr=cfg["alignment"]["min_snr"],
        median_size=cfg["alignment"]["median_size"],
    )
    result.aligned_image = aligned
    alignment_info["angle_deg_applied"] = float(angle_deg)
    result.alignment = alignment_info

    # 5. Preprocessing
    processed, preprocess_info = preprocess_background_and_noise(
        aligned,
        bg_corner_size=cfg["preprocessing"]["bg_corner_size"],
        sigma=cfg["preprocessing"]["sigma"],
    )
    result.processed_image = processed
    result.preprocessing = preprocess_info

    # 6. Abel center
    center_xy, center_x = find_abel_center(
        processed,
        slice_width=cfg["abel_center"]["slice_width"],
    )
    result.center = {
        "center_xy": (float(center_xy[0]), float(center_xy[1])),
        "center_x": float(center_x),
    }

    # 7. Symmetry check
    symmetry_metrics = calculate_left_right_symmetry(
        processed,
        center_x=center_x,
        radial_limit=cfg["symmetry"].get("radial_limit"),
        noise_floor=cfg["symmetry"].get("noise_floor"),
    )
    result.symmetry = symmetry_metrics

    # 8. Abel reconstructions
    basex_image, basex_transform = inverse_abel_basex(
        processed,
        origin_x=center_x,
        regularization=cfg["abel"]["regularization"],
        symmetry_axis=cfg["abel"]["symmetry_axis"],
    )
    hl_image, hl_transform = inverse_abel_hansenlaw(
        processed,
        origin_x=center_x,
        symmetry_axis=cfg["abel"]["symmetry_axis"],
    )
    result.basex_image = basex_image
    result.hansenlaw_image = hl_image
    result.abel = {
        "basex_transform": basex_transform,
        "hansenlaw_transform": hl_transform,
    }

    # 9. Z-scan Gaussian analysis on both reconstructions
    basex_positions = get_z_positions(
        n_z=basex_image.shape[0],
        mode=cfg["z_scan"]["mode"],
        n_points=cfg["z_scan"]["n_points"],
        margin=cfg["z_scan"]["margin"],
    )
    hl_positions = get_z_positions(
        n_z=hl_image.shape[0],
        mode=cfg["z_scan"]["mode"],
        n_points=cfg["z_scan"]["n_points"],
        margin=cfg["z_scan"]["margin"],
    )

    basex_profiles = analyze_all_z_profiles(
        basex_image,
        basex_positions,
        fit_fraction=cfg["profiles"]["fit_fraction"],
        robust=cfg["profiles"]["robust"],
        r0_max=cfg["profiles"]["r0_max"],
    )
    hl_profiles = analyze_all_z_profiles(
        hl_image,
        hl_positions,
        fit_fraction=cfg["profiles"]["fit_fraction"],
        robust=cfg["profiles"]["robust"],
        r0_max=cfg["profiles"]["r0_max"],
    )
    result.profiles_basex = basex_profiles
    result.profiles_hansenlaw = hl_profiles

    # 10. Optional PSF correction
    psf_cfg = cfg.get("psf") or {"enabled": False}
    if psf_cfg.get("enabled", False):
        result.psf_profiles_basex = apply_psf_metadata(basex_profiles, psf_cfg)
        result.psf_profiles_hansenlaw = apply_psf_metadata(hl_profiles, psf_cfg)

    # 11. Observed waist (per method)
    waist_bx = _optional_call(find_beam_waist, basex_profiles, calibration)
    waist_hl = _optional_call(find_beam_waist, hl_profiles, calibration)
    result.waist = {"basex": waist_bx, "hansenlaw": waist_hl}

    # 12. Propagation fits (per method)
    pixel_size = resolve_pixel_scale_um(calibration)
    propagation_bx = _optional_call(
        fit_gaussian_beam_propagation,
        basex_profiles,
        pixel_size=pixel_size,
        robust=cfg["propagation"]["robust"],
    )
    propagation_hl = _optional_call(
        fit_gaussian_beam_propagation,
        hl_profiles,
        pixel_size=pixel_size,
        robust=cfg["propagation"]["robust"],
    )
    result.propagation = {"basex": propagation_bx, "hansenlaw": propagation_hl}

    # 13. Cross-method comparison
    comparison = _optional_call(compare_basex_hansenlaw, basex_profiles, hl_profiles)
    result.validation = {"method_comparison": comparison}

    # 14. Sensitivity-based uncertainty budget (optional)
    if waist_bx and comparison and propagation_bx:
        result.validation["uncertainty_budget"] = build_sensitivity_uncertainty_budget(
            reference_value=float(waist_bx["w0"]),
            method_relative_difference=float(comparison["mean_difference_percent"]) / 100.0,
            calibration_relative_uncertainty=(
                float(calibration["uncertainty_percent"]) / 100.0
                if calibration and calibration.get("enabled", False)
                else 0.0
            ),
        )

    # 15. QA/QC audit report
    if propagation_bx and propagation_hl and waist_bx and waist_hl and comparison:
        result.audit = scientific_audit_report(
            alignment_info=alignment_info,
            symmetry_metrics=symmetry_metrics,
            abel_method_comparison=comparison,
            propagation_basex=propagation_bx,
            propagation_hl=propagation_hl,
            waist_basex=waist_bx,
            waist_hl=waist_hl,
            profile_results_basex=basex_profiles,
            profile_results_hl=hl_profiles,
            calibration_info=result.metrology["calibration"],
            saturation_info=result.camera["saturation"],
            uncertainty_budget=result.validation.get("uncertainty_budget"),
        )

    return result