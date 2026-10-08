"""Tests for the alignment module."""
import numpy as np
import pytest

from laser_beam_analysis.alignment import align_laser_beam
from tests._helpers import create_dummy_beam


def test_align_recovers_known_angle():
    img = create_dummy_beam(shape=(400, 400), angle_deg=5.0, noise_level=0.02)
    _, angle_deg, diag = align_laser_beam(img)
    # The applied correction should be roughly -5° (the input was tilted by +5°).
    assert abs(angle_deg + 5.0) < 1.0, f"expected ~-5°, got {angle_deg}"


def test_align_returns_same_shape():
    img = create_dummy_beam(shape=(300, 300), angle_deg=3.0, noise_level=0.02)
    out, _, _ = align_laser_beam(img)
    assert out.shape == img.shape


def test_align_centers_beam_near_geometric_center():
    img = create_dummy_beam(shape=(400, 400), angle_deg=4.0, noise_level=0.02)
    out, _, diag = align_laser_beam(img)
    # After centering, the beam's column of maximum intensity should be
    # near the geometric center.
    col_profile = np.sum(np.clip(out - diag["background"], 0, None), axis=0)
    peak_x = int(np.argmax(col_profile))
    center = (out.shape[1] - 1) / 2.0
    assert abs(peak_x - center) < 5, f"peak at {peak_x}, center at {center}"


def test_align_rejects_3d_input():
    with pytest.raises(ValueError):
        align_laser_beam(np.zeros((10, 10, 3)))


def test_align_rejects_nan():
    img = np.zeros((100, 100))
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        align_laser_beam(img)


def test_align_rejects_bad_threshold():
    img = create_dummy_beam(shape=(200, 200), angle_deg=2.0, noise_level=0.02)
    with pytest.raises(ValueError):
        align_laser_beam(img, threshold_ratio=0.0)
    with pytest.raises(ValueError):
        align_laser_beam(img, threshold_ratio=1.0)


def test_align_diagnostics_keys():
    img = create_dummy_beam(shape=(300, 300), angle_deg=3.0, noise_level=0.02)
    _, _, diag = align_laser_beam(img)
    for key in (
        "background",
        "noise_std",
        "axis_slope",
        "axis_intercept",
        "angle_deg",
        "axis_rms_error",
        "center_before",
        "center_target",
        "shift_x",
    ):
        assert key in diag, f"missing diagnostic key: {key}"