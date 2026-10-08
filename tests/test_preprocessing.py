"""Tests for the preprocessing module."""
import numpy as np
import pytest

from laser_beam_analysis.preprocessing import preprocess_background_and_noise


def _make_synthetic(h=200, w=200, bg=100.0, peak=200.0, noise=2.0, seed=0):
    """Uniform background + central Gaussian beam + Gaussian noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    xc, yc = w / 2.0, h / 2.0
    beam = peak * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * 20.0 ** 2))
    img = bg + beam + rng.normal(0.0, noise, size=(h, w))
    return img


def test_preprocess_returns_same_shape():
    img = _make_synthetic()
    out, info = preprocess_background_and_noise(img)
    assert out.shape == img.shape


def test_background_estimated_close_to_true():
    img = _make_synthetic(bg=100.0, noise=1.0)
    _, info = preprocess_background_and_noise(img, bg_corner_size=30)
    assert abs(info["background"] - 100.0) < 1.0


def test_negatives_are_not_clipped():
    img = _make_synthetic(bg=100.0, noise=5.0)
    out, _ = preprocess_background_and_noise(img, sigma=0.0)
    # Some pixels in the corners must be below zero after background subtraction.
    assert np.min(out) < 0.0


def test_sigma_zero_disables_smoothing():
    img = _make_synthetic()
    out_smoothed, _ = preprocess_background_and_noise(img, sigma=1.5)
    out_raw, _ = preprocess_background_and_noise(img, sigma=0.0)
    # After smoothing, the max should be <= raw max (blur reduces peak).
    assert np.max(out_smoothed) <= np.max(out_raw) + 1e-9


def test_noise_reduction_factor_reported():
    img = _make_synthetic(noise=3.0)
    _, info = preprocess_background_and_noise(img, sigma=1.5)
    assert "noise_reduction_factor" in info
    assert info["noise_reduction_factor"] > 1.0


def test_rejects_3d():
    with pytest.raises(ValueError):
        preprocess_background_and_noise(np.zeros((10, 10, 3)))


def test_rejects_nan():
    img = np.zeros((100, 100))
    img[0, 0] = np.nan
    with pytest.raises(ValueError):
        preprocess_background_and_noise(img)


def test_rejects_bad_corner_size():
    img = np.zeros((100, 100))
    with pytest.raises(ValueError):
        preprocess_background_and_noise(img, bg_corner_size=0)
    with pytest.raises(ValueError):
        preprocess_background_and_noise(img, bg_corner_size=60)  # 2*60 > 100


def test_rejects_bad_sigma():
    img = np.zeros((100, 100))
    with pytest.raises(ValueError):
        preprocess_background_and_noise(img, sigma=-1.0)


def test_info_keys_complete():
    img = _make_synthetic()
    _, info = preprocess_background_and_noise(img)
    for key in (
        "background",
        "noise_std_before",
        "noise_std_after",
        "conventional_std_before",
        "noise_reduction_factor",
        "max_before",
        "max_after",
        "sigma",
    ):
        assert key in info, f"missing info key: {key}"