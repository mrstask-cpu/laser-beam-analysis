"""Tests for the reporting module."""
import numpy as np
import pandas as pd

from laser_beam_analysis.reporting import profiles_to_dataframe


def _sample_results(n=3):
    return [
        {
            "z_index": i,
            "success": True,
            "amplitude": 1.0,
            "peak_intensity": 1.1,
            "baseline": 0.0,
            "r0": 0.5,
            "sigma": 2.0 + 0.1 * i,
            "beam_radius_w": 4.0 + 0.2 * i,
            "diameter_1e": 5.6,
            "diameter_1e2": 8.0 + 0.4 * i,
            "fwhm": 4.7,
            "r_squared_signal": 0.99,
            "r_squared_full": 0.98,
            "nrmse_full": 0.02,
            "max_abs_residual_full": 0.005,
        }
        for i in range(n)
    ]


def test_dataframe_pixel_only_when_no_calibration():
    df = profiles_to_dataframe(_sample_results())
    assert isinstance(df, pd.DataFrame)
    assert "z_px" in df.columns
    assert "w_px" in df.columns
    # No physical-unit columns when calibration is off.
    assert "w_um" not in df.columns
    assert len(df) == 3


def test_dataframe_adds_physical_columns_when_calibration_enabled():
    calibration = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 2.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 5.0,
        "method": "target",
    }
    df = profiles_to_dataframe(_sample_results(), calibration=calibration)
    assert "w_um" in df.columns
    np.testing.assert_allclose(df["w_um"], df["w_px"] * 2.0)


def test_dataframe_adds_radiometry_column():
    radiometry = {"enabled": True, "factor_to_W_cm2": 0.5}
    df = profiles_to_dataframe(_sample_results(), radiometry_config=radiometry)
    assert "peak_intensity_W_cm2" in df.columns
    np.testing.assert_allclose(df["peak_intensity_W_cm2"], df["peak_intensity"] * 0.5)


def test_dataframe_no_radiometry_column_when_disabled():
    df = profiles_to_dataframe(_sample_results())
    assert (df["peak_intensity_W_cm2"].isna()).all()


def test_dataframe_psf_corrected_columns_when_present():
    results = _sample_results()
    for item in results:
        item["beam_radius_w_psf_corrected"] = 3.5
        item["diameter_1e2_psf_corrected"] = 7.0
    calibration = {
        "enabled": True,
        "mode": "direct",
        "pixel_size": 2.0,
        "pixel_unit": "um",
        "display_unit": "um",
        "uncertainty_percent": 5.0,
        "method": "target",
    }
    df = profiles_to_dataframe(results, calibration=calibration)
    assert "w_psf_corrected_um" in df.columns
    np.testing.assert_allclose(df["w_psf_corrected_um"], 7.0)


def test_dataframe_handles_failed_slices():
    results = [
        {"z_index": 0, "success": True, "beam_radius_w": 4.0},
        {"z_index": 1, "success": False},
    ]
    df = profiles_to_dataframe(results)
    assert len(df) == 2
    assert df.loc[1, "success"] == False  # noqa: E712
    assert np.isnan(df.loc[1, "w_px"])