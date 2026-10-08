"""Convert per-slice fit results into a pandas DataFrame.

Public API:
    profiles_to_dataframe(profile_results, calibration=None,
                         radiometry_config=None) -> pd.DataFrame

The DataFrame contains:
    - pixel-space columns (z_px, w_px, ...),
    - physical-unit columns (<unit> suffixed) when calibration is enabled,
    - optionally radiometric intensity in W/cm^2,
    - optional PSF-corrected widths if present in the input items.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .calibration import LENGTH_UNITS_UM, resolve_pixel_scale_um
from .metrology import radiometric_convert_intensity


def profiles_to_dataframe(
    profile_results,
    calibration: Optional[dict] = None,
    radiometry_config: Optional[dict] = None,
) -> pd.DataFrame:
    """Convert per-slice fit results into a tabular DataFrame.

    Args:
        profile_results: list of per-slice dicts from analyze_all_z_profiles
            (optionally augmented by apply_psf_metadata).
        calibration: optional calibration dict (see calibration module).
        radiometry_config: optional radiometry dict (see metrology module).
            If enabled, a column peak_intensity_W_cm2 is added.

    Returns:
        pandas.DataFrame with one row per input slice.
    """
    rows = []

    scale_um = resolve_pixel_scale_um(calibration) if calibration else None
    unit = (calibration or {}).get("display_unit", "px") if calibration else "px"

    radiometry_enabled = bool(
        radiometry_config and radiometry_config.get("enabled", False)
    )

    for item in profile_results:
        peak = item.get("peak_intensity", np.nan)

        row = {
            "z_px": item.get("z_index", np.nan),
            "success": item.get("success", False),
            "amplitude": item.get("amplitude", np.nan),
            "peak_intensity": peak,
            "baseline": item.get("baseline", np.nan),
            "r0_px": item.get("r0", np.nan),
            "sigma_px": item.get("sigma", np.nan),
            "w_px": item.get("beam_radius_w", np.nan),
            "D_1e_px": item.get("diameter_1e", np.nan),
            "D_1e2_px": item.get("diameter_1e2", np.nan),
            "FWHM_px": item.get("fwhm", np.nan),
            "R2_signal": item.get("r_squared_signal", np.nan),
            "R2_full": item.get("r_squared_full", np.nan),
            "NRMSE_full": item.get("nrmse_full", np.nan),
            "max_abs_residual": item.get("max_abs_residual_full", np.nan),
            "peak_intensity_W_cm2": (
                float(radiometric_convert_intensity(peak, radiometry_config))
                if radiometry_enabled and np.isfinite(peak)
                else np.nan
            ),
        }

        if scale_um is not None:
            factor = scale_um / LENGTH_UNITS_UM[unit]
            row[f"z_{unit}"] = row["z_px"] * factor
            row[f"r0_{unit}"] = row["r0_px"] * factor
            row[f"sigma_{unit}"] = row["sigma_px"] * factor
            row[f"w_{unit}"] = row["w_px"] * factor
            row[f"D_1e_{unit}"] = row["D_1e_px"] * factor
            row[f"D_1e2_{unit}"] = row["D_1e2_px"] * factor
            row[f"FWHM_{unit}"] = row["FWHM_px"] * factor

            w_psf = item.get("beam_radius_w_psf_corrected", np.nan)
            d_psf = item.get("diameter_1e2_psf_corrected", np.nan)
            row[f"w_psf_corrected_{unit}"] = (
                w_psf * factor if np.isfinite(w_psf) else np.nan
            )
            row[f"D_1e2_psf_corrected_{unit}"] = (
                d_psf * factor if np.isfinite(d_psf) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)