"""Tests for the propagation module."""
import numpy as np
import pytest

from laser_beam_analysis.propagation import (
    find_beam_waist,
    gaussian_beam_width_model,
    fit_gaussian_beam_propagation,
)


def _ideal_width(z, w0=0.5, z0=0.0, zR=20.0):
    return gaussian_beam_width_model(z, w0, z0, zR)


def _make_results(z, w, noise=0.0, seed=0):
    """Convert (z, w) arrays into the profile_results format used by the module."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float).copy()
    if noise > 0:
        w = w + rng.normal(0, noise, size=w.shape)
    return [
        {"z_index": int(zi), "success": True, "beam_radius_w": float(wi)}
        for zi, wi in zip(z, w)
    ]


# ---------------------------------------------------- find_beam_waist

def test_find_beam_waist_discrete_minimum():
    z = np.arange(0, 41, 1)
    w = _ideal_width(z, w0=0.5, z0=20.0, zR=20.0)
    results = _make_results(z, w)
    res = find_beam_waist(results)
    assert res["z0"] == 20.0
    assert res["w0"] == pytest.approx(0.5)
    assert res["boundary_limited"] is False


def test_find_beam_waist_subpixel_available():
    z = np.arange(0, 41, 1)
    w = _ideal_width(z, w0=0.5, z0=20.5, zR=20.0)
    results = _make_results(z, w)
    res = find_beam_waist(results)
    assert res["subpixel_available"] is True


def test_find_beam_waist_boundary_flagged():
    z = np.arange(0, 41, 1)
    # Monotonic increase -> minimum at the first sample.
    w = z.astype(float) + 1.0
    results = _make_results(z, w)
    res = find_beam_waist(results)
    assert res["boundary_limited"] is True
    assert res["subpixel_available"] is False


def test_find_beam_waist_rejects_too_few():
    results = [
        {"z_index": 0, "success": True, "beam_radius_w": 1.0},
        {"z_index": 1, "success": True, "beam_radius_w": 0.9},
    ]
    with pytest.raises(RuntimeError):
        find_beam_waist(results)


def test_find_beam_waist_ignores_failed_slices():
    z = np.arange(0, 41, 1)
    w = _ideal_width(z, w0=0.5, z0=20.0, zR=20.0)
    results = _make_results(z, w)
    # Mark a handful of slices as failed; they must be ignored.
    for r in results[::3]:
        r["success"] = False
    res = find_beam_waist(results)
    assert res["z0"] == 20.0


# ------------------------------------------ gaussian_beam_width_model

def test_gaussian_beam_width_model_at_waist():
    assert gaussian_beam_width_model(0.0, 1.0, 0.0, 10.0) == pytest.approx(1.0)


def test_gaussian_beam_width_model_far_field():
    z = np.array([100.0])
    w = gaussian_beam_width_model(z, 1.0, 0.0, 10.0)
    assert w[0] == pytest.approx(1.0 * np.sqrt(1 + 100.0))


# --------------------------------------- fit_gaussian_beam_propagation

def test_fit_recovers_known_parameters():
    z = np.arange(0, 61, 1)
    w = _ideal_width(z, w0=1.0, z0=30.0, zR=15.0)
    results = _make_results(z, w)
    out = fit_gaussian_beam_propagation(results, robust=False)
    assert out["success"] is True
    assert out["w0"] == pytest.approx(1.0, abs=1e-3)
    assert out["z0"] == pytest.approx(30.0, abs=1e-2)
    assert out["zR"] == pytest.approx(15.0, abs=1e-2)


def test_fit_high_r_squared_on_clean_data():
    z = np.arange(0, 61, 1)
    w = _ideal_width(z, w0=1.0, z0=30.0, zR=15.0)
    results = _make_results(z, w)
    out = fit_gaussian_beam_propagation(results, robust=False)
    assert out["r_squared"] > 0.999


def test_fit_warns_on_extrapolated_waist():
    z = np.arange(0, 61, 1)
    # True waist at z=200, well outside the sampled range 0..60.
    w = _ideal_width(z, w0=0.5, z0=200.0, zR=50.0)
    results = _make_results(z, w)
    out = fit_gaussian_beam_propagation(results, robust=False)
    assert out["z0_inside_data"] is False
    assert any("экстраполяция" in msg for msg in out["warnings"])


def test_fit_warns_on_constant_width():
    z = np.arange(0, 61, 1)
    w = np.full_like(z, 1.0, dtype=float)
    results = _make_results(z, w)
    out = fit_gaussian_beam_propagation(results, robust=False)
    # Constant width -> propagation model has no advantage.
    assert out["delta_bic"] < 6
    assert any("constant-width" in msg for msg in out["warnings"])


def test_fit_rejects_too_few_points():
    z = np.arange(0, 5, 1)
    w = _ideal_width(z)
    results = _make_results(z, w)
    with pytest.raises(RuntimeError):
        fit_gaussian_beam_propagation(results)


def test_fit_rejects_bad_pixel_size():
    z = np.arange(0, 21, 1)
    w = _ideal_width(z)
    results = _make_results(z, w)
    with pytest.raises(ValueError):
        fit_gaussian_beam_propagation(results, pixel_size=-1.0)


def test_fit_result_keys_complete():
    z = np.arange(0, 41, 1)
    w = _ideal_width(z, w0=1.0, z0=20.0, zR=15.0)
    results = _make_results(z, w)
    out = fit_gaussian_beam_propagation(results)
    for key in (
        "success",
        "w0",
        "z0",
        "zR",
        "parameter_std",
        "covariance",
        "correlation_matrix",
        "r_squared",
        "rmse",
        "sse_gaussian",
        "sse_constant",
        "bic_gaussian",
        "bic_constant",
        "delta_bic",
        "relative_variation",
        "zR_relative_error",
        "zR_to_span",
        "z0_inside_data",
        "warnings",
        "z",
        "w",
        "fitted_w",
        "residual",
        "definition",
    ):
        assert key in out, f"missing key: {key}"