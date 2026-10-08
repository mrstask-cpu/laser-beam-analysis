"""Tests for the pipeline module."""
import numpy as np
import pytest

from laser_beam_analysis.pipeline import analyze_beam
from laser_beam_analysis.results import BeamAnalysisResult


def _synthetic_beam(h=201, w=201, sigma=12.0, angle_deg=0.0, seed=0):
    """Tilted, noisy Gaussian beam for end-to-end testing."""
    from scipy.ndimage import rotate as ndrotate

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    # Elongated vertically so the alignment step has something to work with.
    beam = np.exp(
        -((x - xc) ** 2) / (2 * sigma ** 2)
        - ((y - yc) ** 2) / (2 * (h / 3.0) ** 2)
    )
    if angle_deg:
        beam = ndrotate(beam, angle_deg, reshape=False, order=3, mode="constant", cval=0.0)
    img = beam + rng.normal(0, 0.005, size=(h, w))
    # Background offset to exercise the preprocessing stage.
    return img * 100.0 + 20.0


def test_pipeline_returns_beam_analysis_result():
    img = _synthetic_beam()
    result = analyze_beam(img)
    assert isinstance(result, BeamAnalysisResult)


def test_pipeline_fills_core_stages():
    img = _synthetic_beam()
    result = analyze_beam(img)
    assert result.raw_image is not None
    assert result.aligned_image is not None
    assert result.processed_image is not None
    assert result.basex_image is not None
    assert result.hansenlaw_image is not None
    assert result.alignment  # not empty
    assert result.preprocessing
    assert result.center
    assert result.symmetry
    assert result.profiles_basex
    assert result.profiles_hansenlaw


def test_pipeline_fills_audit_on_clean_input():
    img = _synthetic_beam()
    result = analyze_beam(img)
    # On a clean symmetric beam everything upstream should succeed,
    # so audit must be populated.
    assert result.audit
    assert "axis_rms_px" in result.audit
    assert "method_mean_relative_difference" in result.audit


def test_pipeline_respects_config_overrides():
    img = _synthetic_beam()
    cfg = {
        "preprocessing": {"sigma": 0.0},
        "abel_center": {"slice_width": 5},
        "z_scan": {"mode": "diagnostic", "margin": 5},
    }
    result = analyze_beam(img, config=cfg)
    assert result.preprocessing["sigma"] == 0.0
    # diagnostic mode returns 15 z positions, so profiles count must equal 15.
    assert len(result.profiles_basex) == 15


def test_pipeline_rejects_uniform_input():
    with pytest.raises(ValueError):
        analyze_beam(np.ones((50, 50)))


def test_pipeline_image_shape_recorded():
    img = _synthetic_beam(h=150, w=150)
    result = analyze_beam(img)
    assert result.image_shape == (150, 150)