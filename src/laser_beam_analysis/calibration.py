"""Physical unit handling and pixel-to-physical calibration.

Public API:
    LENGTH_UNITS_UM
    length_to_um(value, unit)
    um_to_unit(value_um, unit)
    resolve_pixel_scale_um(calibration)
    calibration_summary(calibration)
    validate_calibration_config(calibration)
    convert_px_to_display(values_px, calibration, output_unit=None)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


LENGTH_UNITS_UM = {
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 1e-3,
    "pm": 1e-6,
    "A": 1e-4,
    "Å": 1e-4,
    "px": None,
}


def length_to_um(value: float, unit: str) -> float:
    """Convert a length in the given unit to micrometers."""
    unit_key = str(unit).strip()
    if unit_key not in LENGTH_UNITS_UM or LENGTH_UNITS_UM[unit_key] is None:
        raise ValueError(
            f"Неизвестная единица длины для физической калибровки: {unit}"
        )
    return float(value) * LENGTH_UNITS_UM[unit_key]


def um_to_unit(value_um: float, unit: str) -> float:
    """Convert micrometers to the given unit."""
    unit_key = str(unit).strip()
    if unit_key not in LENGTH_UNITS_UM or LENGTH_UNITS_UM[unit_key] is None:
        raise ValueError(f"Неизвестная единица отображения: {unit}")
    return float(value_um) / LENGTH_UNITS_UM[unit_key]


def resolve_pixel_scale_um(calibration: Optional[dict]) -> Optional[float]:
    """Return physical pixel size in µm/px, or None if calibration is disabled."""
    if calibration is None or not calibration.get("enabled", False):
        return None

    mode = calibration.get("mode", "direct")
    if mode == "direct":
        value = calibration.get("pixel_size", None)
        unit = calibration.get("pixel_unit", "um")
        if value is None or float(value) <= 0:
            raise ValueError("Для direct-калибровки нужен pixel_size > 0.")
        return length_to_um(value, unit)

    if mode == "magnification":
        camera_pixel_um = calibration.get("camera_pixel_size_um", None)
        magnification = calibration.get("magnification", None)
        if camera_pixel_um is None or float(camera_pixel_um) <= 0:
            raise ValueError(
                "Для magnification-калибровки нужен camera_pixel_size_um > 0."
            )
        if magnification is None or float(magnification) <= 0:
            raise ValueError("Для magnification-калибровки нужно magnification > 0.")
        return float(camera_pixel_um) / float(magnification)

    raise ValueError("calibration['mode'] должен быть 'direct' или 'magnification'.")


def calibration_summary(calibration: Optional[dict]) -> dict:
    """Return a compact summary of the calibration state."""
    scale_um = resolve_pixel_scale_um(calibration)
    if scale_um is None:
        return {
            "enabled": False,
            "pixel_size_um": None,
            "display_unit": "px",
            "relative_uncertainty": None,
            "method": calibration.get("mode", "disabled") if calibration else "disabled",
        }
    rel = float(calibration.get("uncertainty_percent", 0.0)) / 100.0
    return {
        "enabled": True,
        "pixel_size_um": scale_um,
        "display_unit": calibration.get("display_unit", "um"),
        "relative_uncertainty": rel,
        "method": calibration.get("method", calibration.get("mode", "direct")),
    }


def validate_calibration_config(calibration: Optional[dict]) -> dict:
    """Strict validation of the calibration config prior to running an analysis."""
    if calibration is None or not calibration.get("enabled", False):
        return {"enabled": False, "valid": True}

    display_unit = calibration.get("display_unit", "um")
    if display_unit not in LENGTH_UNITS_UM or LENGTH_UNITS_UM[display_unit] is None:
        raise ValueError(f"Неподдерживаемая display_unit: {display_unit}")

    scale_um = resolve_pixel_scale_um(calibration)
    unc = float(calibration.get("uncertainty_percent", 0.0))
    if not 0 <= unc < 100:
        raise ValueError("uncertainty_percent должна быть в диапазоне [0, 100).")

    method = str(calibration.get("method", "")).strip()
    if not method:
        raise ValueError(
            "Для включённой калибровки необходимо указать calibration['method']."
        )

    return {
        "enabled": True,
        "valid": True,
        "pixel_size_um": scale_um,
        "display_unit": display_unit,
        "relative_uncertainty": unc / 100.0,
        "method": method,
    }


def convert_px_to_display(
    values_px: Any,
    calibration: Optional[dict],
    output_unit: Optional[str] = None,
) -> np.ndarray:
    """Convert pixel values to physical units; returns pixels unchanged if disabled."""
    values = np.asarray(values_px, dtype=float)
    scale_um = resolve_pixel_scale_um(calibration)
    if scale_um is None:
        return values
    unit = output_unit or calibration.get("display_unit", "um")
    return values * scale_um / LENGTH_UNITS_UM[unit]