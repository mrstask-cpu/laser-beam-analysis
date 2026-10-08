"""Camera-level corrections and acquisition diagnostics.

Public API:
    apply_camera_corrections(image, dark_frame=None, flat_field=None, flat_floor=1e-6)
    acquisition_saturation_diagnostics(source, image)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .io import _load_raw, _require_finite_2d


def _load_optional_frame(frame_source, name: str) -> Optional[np.ndarray]:
    """Load an optional correction frame (dark / flat) as a finite 2D array.

    Uses _load_raw because calibration frames are not required to have
    non-zero dynamic range.
    """
    if frame_source is None:
        return None
    arr = _load_raw(frame_source)
    return _require_finite_2d(arr, name)


def apply_camera_corrections(
    image,
    dark_frame: Any = None,
    flat_field: Any = None,
    flat_floor: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Apply optional dark-frame and flat-field corrections.

    By default both corrections are disabled. If a frame is provided,
    it is loaded via load_image and validated against the input shape.

    Args:
        image: 2D input image.
        dark_frame: optional dark frame (path/URL/array) subtracted from the image.
        flat_field: optional flat-field frame (path/URL/array); normalized to its
            own median before division.
        flat_floor: minimum absolute normalized flat value; smaller values are
            replaced with NaN and later trigger a finite-check failure.

    Returns:
        (corrected_image, info_dict)
    """
    image = _require_finite_2d(image, "image")
    corrected = image.copy()

    if dark_frame is not None:
        dark = _load_optional_frame(dark_frame, "dark_frame")
        if dark.shape != corrected.shape:
            raise ValueError(
                f"dark_frame имеет форму {dark.shape}, а изображение {corrected.shape}."
            )
        corrected = corrected - dark

    if flat_field is not None:
        flat = _load_optional_frame(flat_field, "flat_field")
        if flat.shape != corrected.shape:
            raise ValueError(
                f"flat_field имеет форму {flat.shape}, а изображение {corrected.shape}."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            flat_normalized = flat / np.median(flat)
        flat_normalized = np.where(
            np.abs(flat_normalized) < flat_floor, np.nan, flat_normalized
        )
        corrected = corrected / flat_normalized

    if not np.all(np.isfinite(corrected)):
        raise ValueError("После dark/flat correction появились NaN/inf.")

    return corrected, {
        "dark_enabled": dark_frame is not None,
        "flat_enabled": flat_field is not None,
    }


def acquisition_saturation_diagnostics(source, image) -> dict:
    """Check for sensor saturation from the raw integer source.

    Detection only works if the original source is an integer image whose
    maximum code value equals the dtype's max (i.e. the sensor's full scale).
    For non-integer / unknown sources the diagnostics report unavailability.
    """
    diagnostics = {
        "available": False,
        "dtype": None,
        "bit_depth": None,
        "saturated_pixels": 0,
        "saturated_fraction": 0.0,
        "status": "not_available",
    }

    raw = None
    try:
        if isinstance(source, np.ndarray):
            raw = np.asarray(source)
        elif isinstance(source, (str, Path)):
            with Image.open(source) as pil_image:
                raw = np.asarray(pil_image)
        else:
            raw = None
    except Exception as exc:
        diagnostics["status"] = f"unavailable: {exc}"
        return diagnostics

    if raw is None or not np.issubdtype(raw.dtype, np.integer):
        diagnostics["status"] = "not_available_for_noninteger_source"
        return diagnostics

    max_code = np.iinfo(raw.dtype).max
    sat = raw >= max_code
    sat_pixels = np.any(sat[..., :3], axis=-1) if raw.ndim == 3 else sat

    diagnostics["available"] = True
    diagnostics["dtype"] = str(raw.dtype)
    diagnostics["bit_depth"] = int(np.iinfo(raw.dtype).bits)
    diagnostics["saturated_pixels"] = int(np.count_nonzero(sat_pixels))
    diagnostics["saturated_fraction"] = float(np.mean(sat_pixels))
    diagnostics["status"] = (
        "WARNING_SATURATION" if diagnostics["saturated_fraction"] > 0 else "OK"
    )
    return diagnostics