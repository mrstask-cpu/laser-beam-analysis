"""Beam-axis detection and geometric alignment.

Public API:
    align_laser_beam(img_array, threshold_ratio=0.15, min_snr=3.0, median_size=3)
        -> (centered_img, angle_deg, diagnostics)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, rotate, shift


def align_laser_beam(
    img_array,
    threshold_ratio: float = 0.15,
    min_snr: float = 3.0,
    median_size: int = 3,
) -> tuple[np.ndarray, float, dict]:
    """Detect the beam's longitudinal axis and rotate the image so the axis is vertical.

    The axis is estimated from intensity-weighted row centers of energy using a
    soft median filter (only for detection, not for the output), robust
    background / noise estimation, and a weighted least-squares line fit.

    The original image is rotated without prior filtering, preserving the beam
    profile. After rotation the beam is centered horizontally on the geometric
    image center, as required for subsequent Abel inversion.

    Args:
        img_array: 2D input image.
        threshold_ratio: relative threshold (0..1) of the row's peak signal.
        min_snr: minimum signal-to-noise ratio for a row to be considered.
        median_size: median filter size used only for axis detection.

    Returns:
        centered_img: rotated and horizontally centered image.
        angle_deg: applied rotation angle in degrees.
        diagnostics: dict with background, noise, slope, intercept, RMS error,
            per-row centers, fitted line, and centering shift.
    """
    img_array = np.asarray(img_array, dtype=np.float64)

    if img_array.ndim != 2:
        raise ValueError("img_array должен быть двумерным.")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("img_array содержит NaN или inf.")
    if threshold_ratio <= 0 or threshold_ratio >= 1:
        raise ValueError("threshold_ratio должен находиться в диапазоне (0, 1).")
    if min_snr <= 0:
        raise ValueError("min_snr должен быть > 0.")

    h, w = img_array.shape
    y_indices = np.arange(h, dtype=np.float64)
    x_indices = np.arange(w, dtype=np.float64)

    # Axis detection uses a median-filtered copy; the output image is NOT filtered.
    analysis_img = median_filter(img_array, size=median_size)

    # Robust background / noise estimation from image borders.
    border = max(5, min(h, w) // 20)
    border_pixels = np.concatenate([
        analysis_img[:border, :].ravel(),
        analysis_img[-border:, :].ravel(),
        analysis_img[:, :border].ravel(),
        analysis_img[:, -border:].ravel(),
    ])
    bg_level = np.median(border_pixels)
    mad = np.median(np.abs(border_pixels - bg_level))
    noise_std = 1.4826 * mad
    noise_std = max(noise_std, np.finfo(float).eps)

    # Row-wise centers of energy.
    x_centers = []
    valid_y = []
    row_signal = []

    for y in range(h):
        row = analysis_img[y, :]
        row_excess = np.clip(row - bg_level, 0, None)
        peak = np.max(row_excess)

        if peak < min_snr * noise_std:
            continue

        threshold = max(min_snr * noise_std, threshold_ratio * peak)
        mask = row_excess >= threshold

        if np.count_nonzero(mask) < 3:
            continue

        signal = row_excess[mask]
        if np.sum(signal) <= 0:
            continue

        x_values = x_indices[mask]
        x_center = np.sum(x_values * signal) / np.sum(signal)

        x_centers.append(x_center)
        valid_y.append(y)
        row_signal.append(np.sum(signal))

    if len(valid_y) < 10:
        raise ValueError("Недостаточно информативных строк для определения оси.")

    x_centers = np.asarray(x_centers)
    valid_y = np.asarray(valid_y, dtype=np.float64)
    row_signal = np.asarray(row_signal)

    # Weighted least-squares fit of x = m*y + c.
    weights = row_signal / np.max(row_signal)
    sqrt_weights = np.sqrt(weights)
    A = np.column_stack([valid_y, np.ones_like(valid_y)])
    A_weighted = A * sqrt_weights[:, None]
    b_weighted = x_centers * sqrt_weights

    coeffs, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
    m, c = coeffs

    fitted_centers = m * valid_y + c
    residuals = x_centers - fitted_centers
    rms_axis_error = np.sqrt(np.average(residuals ** 2, weights=weights))

    angle_rad = -np.arctan(m)
    angle_deg = float(np.degrees(angle_rad))

    aligned_img = rotate(
        img_array,
        angle_deg,
        reshape=False,
        order=3,
        mode="nearest",
        prefilter=True,
    )

    # Coarse horizontal centering on the geometric image center.
    profile_x = np.sum(np.clip(aligned_img - bg_level, 0, None), axis=0)
    profile_threshold = max(
        min_snr * noise_std * h,
        threshold_ratio * np.max(profile_x),
    )
    profile_mask = profile_x >= profile_threshold

    if np.count_nonzero(profile_mask) < 3:
        raise ValueError("Не удалось определить поперечный центр после вращения.")

    current_center_x = np.sum(
        x_indices[profile_mask] * profile_x[profile_mask]
    ) / np.sum(profile_x[profile_mask])

    target_center_x = (w - 1) / 2.0
    shift_x = target_center_x - current_center_x

    centered_img = shift(
        aligned_img,
        shift=(0, shift_x),
        order=3,
        mode="nearest",
        prefilter=True,
    )

    diagnostics = {
        "background": float(bg_level),
        "noise_std": float(noise_std),
        "axis_slope": float(m),
        "axis_intercept": float(c),
        "angle_deg": angle_deg,
        "axis_rms_error": float(rms_axis_error),
        "center_before": float(current_center_x),
        "center_target": float(target_center_x),
        "shift_x": float(shift_x),
        "x_centers": x_centers,
        "valid_y": valid_y,
        "fitted_centers": fitted_centers,
        "weights": weights,
    }

    return centered_img, angle_deg, diagnostics