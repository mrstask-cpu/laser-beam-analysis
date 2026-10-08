"""Tests for the profiles module."""
import numpy as np
import pytest

from laser_beam_analysis.profiles import (
    extract_radial_profile,
    extract_symmetric_radial_profile,
    fit_gaussian_radial_profile,
)


def _gaussian_2d(h=121, w=121, sigma=15.0, xc=None, seed=0):
    """Symmetric Gaussian image; suitable as a stand-in for an Abel reconstruction."""
    rng = np.random.default_rng(seed)
    if xc is None:
        xc = (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w]
    img = np.exp(-((x - xc) ** 2 + (y - h / 2.0) ** 2) / (2 * sigma ** 2))
    img = img + rng.normal(0, 0.001, size=(h, w))
    return img, xc


def _gaussian_1d(r, sigma=15.0):
    return np.exp(-(r ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------- extraction

def test_extract_symmetric_returns_three_items():
    img, _ = _gaussian_2d()
    out = extract_radial_profile(img, z_index=60, symmetric=True)
    assert len(out) == 3
    r, profile, diag = out
    assert r.ndim == 1
    assert profile.ndim == 1
    assert r.size == profile.size


def test_extract_asymmetric_returns_two_items():
    img, _ = _gaussian_2d(w=121)
    out = extract_radial_profile(img, z_index=60, symmetric=False)
    assert len(out) == 2
    r, profile = out
    assert r.size == profile.size


def test_extract_asymmetric_rejects_even_width():
    img, _ = _gaussian_2d(w=120)
    with pytest.raises(ValueError):
        extract_radial_profile(img, z_index=60, symmetric=False)


def test_extract_radial_profile_z_index_out_of_range():
    img, _ = _gaussian_2d()
    with pytest.raises(IndexError):
        extract_radial_profile(img, z_index=10000)


def test_extract_max_radius_limits_points():
    img, _ = _gaussian_2d()
    _, p_all, _ = extract_symmetric_radial_profile(img, z_index=60)
    _, p_lim, _ = extract_symmetric_radial_profile(img, z_index=60, max_radius=10.0)
    assert p_lim.size < p_all.size


def test_extract_diagnostics_keys():
    img, _ = _gaussian_2d()
    _, _, diag = extract_symmetric_radial_profile(img, z_index=60)
    for key in (
        "left_profile",
        "right_profile",
        "normalized_rms_difference",
        "correlation",
        "radius_max",
    ):
        assert key in diag, f"missing diagnostic key: {key}"


# ------------------------------------------------------------------ fitting

def test_fit_gaussian_recovers_sigma():
    r = np.arange(0.0, 60.0, 1.0)
    sigma_true = 12.0
    profile = _gaussian_1d(r, sigma=sigma_true)
    res = fit_gaussian_radial_profile(r, profile, r0_max=5.0)
    assert res["success"] is True
    assert abs(res["sigma"] - sigma_true) < 0.5


def test_fit_gaussian_r_squared_high_on_clean_data():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r, sigma=12.0)
    res = fit_gaussian_radial_profile(r, profile, r0_max=5.0)
    assert res["r_squared_signal"] > 0.99
    assert res["r_squared_full"] > 0.99


def test_fit_gaussian_beam_radius_is_2_sigma():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r, sigma=12.0)
    res = fit_gaussian_radial_profile(r, profile, r0_max=5.0)
    assert abs(res["beam_radius_w"] - 2.0 * res["sigma"]) < 1e-9


def test_fit_gaussian_fwhm_formula():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r, sigma=12.0)
    res = fit_gaussian_radial_profile(r, profile, r0_max=5.0)
    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * res["sigma"]
    assert abs(res["fwhm"] - expected_fwhm) < 1e-9


def test_fit_gaussian_rejects_mismatched_sizes():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r)[:-5]
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile)


def test_fit_gaussian_rejects_too_few_points():
    r = np.arange(0.0, 5.0, 1.0)
    profile = _gaussian_1d(r)
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile)


def test_fit_gaussian_rejects_nan():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r)
    profile[0] = np.nan
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile)


def test_fit_gaussian_rejects_bad_fit_fraction():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r)
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile, fit_fraction=0.0)
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile, fit_fraction=1.0)


def test_fit_gaussian_rejects_bad_r0_max():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r)
    with pytest.raises(ValueError):
        fit_gaussian_radial_profile(r, profile, r0_max=0.0)


def test_fit_gaussian_result_keys_complete():
    r = np.arange(0.0, 60.0, 1.0)
    profile = _gaussian_1d(r)
    res = fit_gaussian_radial_profile(r, profile, r0_max=5.0)
    for key in (
        "success",
        "amplitude",
        "peak_intensity",
        "r0",
        "sigma",
        "baseline",
        "radius_1e",
        "radius_1e2",
        "diameter_1e",
        "diameter_1e2",
        "beam_radius_w",
        "beam_diameter_2w",
        "fwhm",
        "r_squared",
        "r_squared_signal",
        "r_squared_full",
        "rmse",
        "rmse_signal",
        "rmse_full",
        "nrmse_signal",
        "nrmse_full",
        "max_abs_residual_signal",
        "max_abs_residual_full",
        "fit_mask",
        "r",
        "profile",
        "fitted_profile",
        "residual",
        "optimizer_cost",
        "optimizer_optimality",
    ):
        assert key in res, f"missing key: {key}"