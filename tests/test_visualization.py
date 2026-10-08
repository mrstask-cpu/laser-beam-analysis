"""Tests for the visualization module."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from laser_beam_analysis.visualization import (
    IPYWIDGETS_AVAILABLE,
    interactive_3d_cross_section,
    plot_3d_beam_cross_section,
    plot_beam_parameters,
    radial_profile_to_xy_surface,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _symmetric_image(h=121, w=121, sigma=15.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    img = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma ** 2))
    return img + rng.normal(0, 0.001, size=(h, w))


def _beam_results(n=5):
    return [
        {
            "z_index": i,
            "success": True,
            "peak_intensity": 1.0 + i * 0.01,
            "sigma": 2.0 + i * 0.01,
            "beam_radius_w": 4.0 + i * 0.02,
            "diameter_1e": 5.6,
            "diameter_1e2": 8.0 + i * 0.04,
            "fwhm": 4.7,
            "r_squared_signal": 0.99,
            "r_squared_full": 0.98,
            "nrmse_full": 0.02,
        }
        for i in range(n)
    ]


# ------------------------------------------- radial_profile_to_xy_surface

def test_surface_basic_shape():
    r = np.arange(0.0, 20.0, 1.0)
    profile = np.exp(-(r ** 2) / 50.0)
    X, Y, I_xy = radial_profile_to_xy_surface(r, profile, xy_step=1.0)
    assert X.shape == Y.shape == I_xy.shape
    assert X.shape[0] == X.shape[1]


def test_surface_nan_outside_radius():
    r = np.arange(0.0, 5.0, 1.0)
    profile = np.ones_like(r)
    _, _, I_xy = radial_profile_to_xy_surface(r, profile, xy_step=1.0, radius_max=10.0)
    # The corners lie beyond r[-1] = 4 -> must be NaN.
    assert np.isnan(I_xy[0, 0])


def test_surface_rejects_2d_r():
    with pytest.raises(ValueError):
        radial_profile_to_xy_surface(np.zeros((5, 5)), np.zeros(5))


def test_surface_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        radial_profile_to_xy_surface(np.arange(5), np.arange(6))


def test_surface_rejects_bad_xy_step():
    r = np.arange(5.0)
    with pytest.raises(ValueError):
        radial_profile_to_xy_surface(r, r, xy_step=0)


def test_surface_rejects_bad_radius_max():
    r = np.arange(5.0)
    with pytest.raises(ValueError):
        radial_profile_to_xy_surface(r, r, radius_max=-1.0)


# ------------------------------------------------- plot_beam_parameters

def test_plot_beam_parameters_returns_expected_keys():
    out = plot_beam_parameters(_beam_results())
    for key in (
        "z", "valid", "peak_intensity", "sigma", "beam_radius_w",
        "diameter_1e", "diameter_1e2", "fwhm",
        "r_squared_signal", "r_squared_full", "nrmse_full",
        "z_plot", "unit",
    ):
        assert key in out, f"missing key: {key}"
    assert out["unit"] == "px"


def test_plot_beam_parameters_handles_failed_slices():
    results = _beam_results(3) + [{"z_index": 99, "success": False}]
    out = plot_beam_parameters(results)
    assert np.isnan(out["sigma"][-1])
    assert out["valid"][-1] == False


def test_plot_beam_parameters_with_pixel_size():
    out = plot_beam_parameters(_beam_results(), pixel_size=2.0, unit="um")
    assert out["unit"] == "um"
    # z_plot = z * 2.0
    np.testing.assert_allclose(out["z_plot"], out["z"] * 2.0)


# ------------------------------------------- plot_3d_beam_cross_section

def test_plot_3d_basic():
    img = _symmetric_image()
    X, Y, I_xy, sc = plot_3d_beam_cross_section(img, z_index=60, radius_max=20.0)
    assert X.shape == Y.shape == I_xy.shape
    assert "normalized_rms_difference" in sc
    assert "correlation" in sc


def test_plot_3d_rejects_bad_color_mode():
    img = _symmetric_image()
    with pytest.raises(ValueError):
        plot_3d_beam_cross_section(img, z_index=60, radius_max=20.0, color_mode="log")


def test_plot_3d_rejects_bad_gamma():
    img = _symmetric_image()
    with pytest.raises(ValueError):
        plot_3d_beam_cross_section(
            img, z_index=60, radius_max=20.0, color_mode="power", color_gamma=0.0
        )


def test_plot_3d_rejects_zero_signal():
    img = np.zeros((81, 81))
    with pytest.raises(ValueError):
        plot_3d_beam_cross_section(img, z_index=40, radius_max=20.0)


# ----------------------------------------- interactive_3d_cross_section

def test_interactive_raises_when_widgets_missing(monkeypatch):
    monkeypatch.setattr(
        "laser_beam_analysis.visualization.IPYWIDGETS_AVAILABLE", False
    )
    with pytest.raises(RuntimeError):
        interactive_3d_cross_section(np.zeros((20, 20)))


def test_interactive_rejects_3d_input(monkeypatch):
    monkeypatch.setattr(
        "laser_beam_analysis.visualization.IPYWIDGETS_AVAILABLE", True
    )
    with pytest.raises(ValueError):
        interactive_3d_cross_section(np.zeros((10, 10, 3)))