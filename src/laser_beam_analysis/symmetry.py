"""Left/right symmetry metrics around a vertical axis.

Public API:
    calculate_left_right_symmetry(img_array, center_x,
                                  radial_limit=None, noise_floor=None) -> dict

The function mirrors the image about center_x using linear interpolation,
restricts comparison to pixels whose mirrored coordinate lies inside the
image (and optionally inside a radial limit), and reports robust symmetry
metrics. Regions below noise_floor are excluded from the statistics.
"""
from __future__ import annotations

import numpy as np


def calculate_left_right_symmetry(
    img_array,
    center_x: float,
    radial_limit: float | None = None,
    noise_floor: float | None = None,
) -> dict:
    """Compute left/right symmetry metrics about a vertical axis.

    Args:
        img_array: 2D preprocessed image.
        center_x: sub-pixel x-coordinate of the symmetry axis.
        radial_limit: optional maximum radius (px) to include. If None, the
            full overlapping range is used.
        noise_floor: optional noise threshold. Regions where the average of
            |left| and |right| falls below this value are excluded.

    Returns:
        dict with:
            normalized_rms            – RMS(diff) / RMS(scale)
            correlation               – Pearson correlation of mirrored pairs
            median_relative_difference
            p95_relative_difference
            number_of_points          – number of informative points used

    Raises:
        ValueError: if input is not 2D, contains NaN/inf, radial_limit <= 0,
            or no informative points remain after filtering.
    """
    img_array = np.asarray(img_array, dtype=np.float64)

    if img_array.ndim != 2:
        raise ValueError("Ожидалось 2D изображение.")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("Изображение содержит NaN или inf.")

    h, w = img_array.shape
    x = np.arange(w, dtype=np.float64)

    radius = np.abs(x - center_x)
    mirrored_x = 2.0 * center_x - x

    valid_geometry = (mirrored_x >= 0) & (mirrored_x <= w - 1)

    if radial_limit is not None:
        if radial_limit <= 0:
            raise ValueError("radial_limit должен быть > 0.")
        valid_geometry &= radius <= radial_limit

    x_valid = x[valid_geometry]
    mirrored_x_valid = mirrored_x[valid_geometry]

    # Interpolate the mirrored image row by row.
    mirrored = np.empty_like(img_array)
    for y in range(h):
        mirrored[y, valid_geometry] = np.interp(
            mirrored_x_valid, x, img_array[y]
        )

    original = img_array[:, valid_geometry]
    mirrored_valid = mirrored[:, valid_geometry]

    # Optional noise mask.
    if noise_floor is not None:
        signal_mask = 0.5 * (np.abs(original) + np.abs(mirrored_valid)) > noise_floor
    else:
        signal_mask = np.ones_like(original, dtype=bool)

    difference = original - mirrored_valid
    signal_scale = 0.5 * (np.abs(original) + np.abs(mirrored_valid))

    diff_used = difference[signal_mask]
    signal_used = signal_scale[signal_mask]

    if diff_used.size == 0:
        raise ValueError("После фильтрации не осталось информативных точек.")

    rms_difference = float(np.sqrt(np.mean(diff_used ** 2)))
    rms_signal = float(np.sqrt(np.mean(signal_used ** 2)))
    normalized_rms = rms_difference / rms_signal if rms_signal > 0 else float("inf")

    original_flat = original[signal_mask]
    mirrored_flat = mirrored_valid[signal_mask]

    if np.std(original_flat) > 0 and np.std(mirrored_flat) > 0:
        correlation = float(np.corrcoef(original_flat, mirrored_flat)[0, 1])
    else:
        correlation = float("nan")

    relative_difference = np.abs(difference) / np.maximum(
        signal_scale, np.finfo(float).eps
    )
    relative_difference_used = relative_difference[signal_mask]

    metrics = {
        "normalized_rms": normalized_rms,
        "correlation": correlation,
        "median_relative_difference": float(np.median(relative_difference_used)),
        "p95_relative_difference": float(np.percentile(relative_difference_used, 95)),
        "number_of_points": int(diff_used.size),
    }

    return metrics