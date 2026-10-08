"""Background estimation, background subtraction and Gaussian smoothing.

Public API:
    preprocess_background_and_noise(img_array, bg_corner_size=40, sigma=1.5)
        -> (filtered_image, info)

Design notes:
    Negative values after background subtraction are NOT clipped. They are a
    legitimate representation of noise deviation and are needed downstream
    (Abel inversion, Gaussian fitting). Clipping is a visualization concern.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _corner_pixels(img: np.ndarray, corner_size: int) -> np.ndarray:
    """Concatenate 4 corner blocks of the image into a 1D array."""
    return np.concatenate([
        img[:corner_size, :corner_size].ravel(),
        img[:corner_size, -corner_size:].ravel(),
        img[-corner_size:, :corner_size].ravel(),
        img[-corner_size:, -corner_size:].ravel(),
    ])


def preprocess_background_and_noise(
    img_array,
    bg_corner_size: int = 40,
    sigma: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Estimate background level, subtract it, and apply Gaussian smoothing.

    Steps:
        1. Estimate background from four corner blocks (median).
        2. Estimate noise from corners via MAD -> 1.4826 * MAD.
        3. Subtract background (do NOT clip negatives).
        4. Apply Gaussian smoothing if sigma > 0.
        5. Re-estimate residual noise on the processed corners.

    Args:
        img_array: 2D input image (typically already aligned).
        bg_corner_size: size (px) of each corner block used for background.
        sigma: Gaussian filter sigma in pixels. Set to 0 to disable smoothing.

    Returns:
        filtered_image: background-subtracted (and smoothed) image.
        info: dict with background, noise before/after, noise reduction factor,
            conventional std before, max before/after, and sigma.
    """
    img_array = np.asarray(img_array, dtype=np.float64)

    if img_array.ndim != 2:
        raise ValueError("img_array должен быть двумерным.")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("img_array содержит NaN или inf.")
    if bg_corner_size <= 0:
        raise ValueError("bg_corner_size должен быть > 0.")
    if sigma < 0:
        raise ValueError("sigma не может быть отрицательной.")

    h, w = img_array.shape
    if 2 * bg_corner_size > min(h, w):
        raise ValueError("bg_corner_size слишком велик.")

    corners = _corner_pixels(img_array, bg_corner_size)

    bg_level = float(np.median(corners))
    mad_before = float(np.median(np.abs(corners - bg_level)))
    noise_std_before = 1.4826 * mad_before
    conventional_std_before = float(np.std(corners))

    img_subtracted = img_array - bg_level

    if sigma > 0:
        img_filtered = gaussian_filter(img_subtracted, sigma=sigma)
    else:
        img_filtered = img_subtracted.copy()

    corners_filtered = _corner_pixels(img_filtered, bg_corner_size)
    filtered_bg_median = float(np.median(corners_filtered))
    mad_after = float(np.median(np.abs(corners_filtered - filtered_bg_median)))
    noise_std_after = 1.4826 * mad_after

    if noise_std_after > 0:
        noise_reduction_factor = noise_std_before / noise_std_after
    else:
        noise_reduction_factor = float("inf")

    info = {
        "background": bg_level,
        "noise_std_before": float(noise_std_before),
        "noise_std_after": float(noise_std_after),
        "conventional_std_before": conventional_std_before,
        "noise_reduction_factor": float(noise_reduction_factor),
        "max_before": float(np.max(img_array)),
        "max_after": float(np.max(img_filtered)),
        "sigma": float(sigma),
    }

    return img_filtered, info