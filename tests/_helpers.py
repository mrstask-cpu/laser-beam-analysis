"""Test-only helpers. Not part of the package public API."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, rotate


def create_dummy_beam(shape=(500, 500), angle_deg=5, noise_level=0.1):
    """Generate a synthetic tilted, noisy Gaussian beam for testing."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]

    x0 = w / 2.0
    y0 = h / 2.0
    sigma_x = w / 30.0
    sigma_y = h / 3.0

    beam = np.exp(-((x - x0) ** 2) / (2 * sigma_x ** 2) -
                  ((y - y0) ** 2) / (2 * sigma_y ** 2))

    beam = rotate(beam, angle_deg, reshape=False, order=3, mode="constant", cval=0.0)

    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=noise_level, size=shape)
    noisy = beam + noise
    noisy = gaussian_filter(noisy, sigma=1.0)

    return noisy.astype(np.float64)