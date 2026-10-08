"""Abel-origin detection: locating the axis of symmetry for inversion.

Public API:
    find_abel_center(img_array, slice_width=10) -> (center_xy, center_x)
    compare_abel_center_methods(img_array, slice_width=10) -> dict

Both functions operate on the horizontal axis (axis=1), because upstream
alignment has already made the beam's longitudinal axis vertical.
"""
from __future__ import annotations

import abel
import numpy as np


def find_abel_center(
    img_array,
    slice_width: int = 10,
) -> tuple[tuple[float, float], float]:
    """Find the vertical axis of symmetry using PyAbel's slice method.

    Args:
        img_array: 2D preprocessed and geometrically aligned image.
        slice_width: width of the central summation band used by the method.

    Returns:
        center_xy: (center_y, center_x) as returned by PyAbel.
        center_x: the x-coordinate of the Abel axis.
    """
    img_array = np.asarray(img_array, dtype=np.float64)

    if img_array.ndim != 2:
        raise ValueError("Для поиска центра требуется 2D-массив.")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("Изображение содержит NaN или inf.")
    if slice_width <= 0:
        raise ValueError("slice_width должен быть > 0.")

    center_xy = abel.tools.center.find_origin(
        img_array,
        method="slice",
        axes=1,
        slice_width=slice_width,
    )
    center_y, center_x = center_xy
    return center_xy, float(center_x)


def compare_abel_center_methods(
    img_array,
    slice_width: int = 10,
) -> dict:
    """Compare several independent methods for Abel-origin detection.

    Only the X-coordinate is meaningful here (the longitudinal axis is
    already vertical after alignment). Methods that fail are recorded with
    NaN values and an "error" message.

    Args:
        img_array: 2D preprocessed image.
        slice_width: passed through to PyAbel's find_origin.

    Returns:
        dict mapping method name -> {"y": float, "x": float [, "error": str]}.
    """
    img_array = np.asarray(img_array, dtype=np.float64)

    methods = [
        "image_center",
        "com",
        "convolution",
        "gaussian",
        "slice",
    ]

    results: dict = {}

    for method in methods:
        try:
            origin = abel.tools.center.find_origin(
                img_array,
                method=method,
                axes=1,
                slice_width=slice_width,
            )
            center_y, center_x = origin
            results[method] = {
                "y": float(center_y),
                "x": float(center_x),
            }
        except Exception as exc:
            results[method] = {
                "y": float("nan"),
                "x": float("nan"),
                "error": str(exc),
            }

    return results