"""Inverse Abel reconstruction: BASEX, Hansen–Law, regularization sweep.

Public API:
    inverse_abel_basex(img_array, origin_x, regularization=50, symmetry_axis=0)
        -> (reconstructed, transform)
    inverse_abel_hansenlaw(img_array, origin_x, symmetry_axis=0)
        -> (reconstructed, transform)
    basex_regularization_sweep(img_array, origin_x, regularization_values=None)
        -> dict[reg -> {"transform", "image", "max", "min",
                        "negative_fraction", "relative_negative_amplitude",
                        "roughness"}]

BASEX and Hansen–Law are independent numerical methods for the same
inverse problem; running both provides a cross-method sanity check on the
reconstruction. The regularization sweep is a research tool, not an
automatic selector.
"""
from __future__ import annotations

from typing import Optional

import abel
import numpy as np


_BASEX_CENTER_OPTIONS = {
    "odd_size": True,
    "axes": 1,
    "crop": "valid_region",
}


def _validate_abel_input(img_array) -> np.ndarray:
    img_array = np.asarray(img_array, dtype=np.float64)
    if img_array.ndim != 2:
        raise ValueError("Для Abel inversion нужен 2D-массив.")
    if not np.all(np.isfinite(img_array)):
        raise ValueError("Входное изображение содержит NaN или inf.")
    return img_array


def inverse_abel_basex(
    img_array,
    origin_x: float,
    regularization: float = 50,
    symmetry_axis: int = 0,
) -> tuple[np.ndarray, "abel.Transform"]:
    """Inverse Abel transform using BASEX.

    Args:
        img_array: 2D preprocessed projection image.
        origin_x: x-coordinate of the vertical axis of symmetry.
        regularization: Tikhonov regularization parameter.
        symmetry_axis: 0 means left/right symmetry about the vertical axis.

    Returns:
        reconstructed: 2D reconstructed distribution I(r, z).
        transform: the underlying PyAbel Transform object.
    """
    img_array = _validate_abel_input(img_array)

    if regularization < 0:
        raise ValueError("regularization не может быть отрицательным.")

    transform = abel.Transform(
        img_array,
        direction="inverse",
        method="basex",
        origin=(None, float(origin_x)),
        symmetry_axis=symmetry_axis,
        symmetrize_method="average",
        center_options=_BASEX_CENTER_OPTIONS,
        transform_options={"reg": float(regularization)},
        recast_as_float64=True,
        verbose=False,
    )

    return transform.transform, transform


def inverse_abel_hansenlaw(
    img_array,
    origin_x: float,
    symmetry_axis: int = 0,
) -> tuple[np.ndarray, "abel.Transform"]:
    """Inverse Abel transform using the Hansen–Law method.

    Used as an independent numerical cross-check against BASEX.
    """
    img_array = _validate_abel_input(img_array)

    transform = abel.Transform(
        img_array,
        direction="inverse",
        method="hansenlaw",
        origin=(None, float(origin_x)),
        symmetry_axis=symmetry_axis,
        symmetrize_method="average",
        center_options=_BASEX_CENTER_OPTIONS,
        recast_as_float64=True,
        verbose=False,
    )

    return transform.transform, transform


def basex_regularization_sweep(
    img_array,
    origin_x: float,
    regularization_values: Optional[list] = None,
) -> dict:
    """Run BASEX with several regularization values and summarize each result.

    For each value the following are recorded:
        transform, image, max, min,
        negative_fraction (fraction of negative pixels),
        relative_negative_amplitude (|min| / max),
        roughness (RMS of the discrete second derivative along the radial axis).

    Args:
        img_array: 2D preprocessed projection image.
        origin_x: x-coordinate of the vertical axis of symmetry.
        regularization_values: list of reg values to try. Defaults to
            [0, 1, 5, 10, 25, 50, 100, 200].

    Returns:
        dict mapping reg value -> summary dict.
    """
    if regularization_values is None:
        regularization_values = [0, 1, 5, 10, 25, 50, 100, 200]

    img_array = _validate_abel_input(img_array)

    results: dict = {}

    for reg in regularization_values:
        reg_key = float(reg)

        transform = abel.Transform(
            img_array,
            direction="inverse",
            method="basex",
            origin=(None, float(origin_x)),
            symmetry_axis=0,
            symmetrize_method="average",
            center_options=_BASEX_CENTER_OPTIONS,
            transform_options={"reg": reg_key},
            recast_as_float64=True,
            verbose=False,
        )

        recon = transform.transform

        maximum = float(np.max(recon))
        minimum = float(np.min(recon))
        negative_fraction = float(np.mean(recon < 0))
        relative_negative_amplitude = (
            abs(minimum) / maximum if maximum > 0 else float("nan")
        )

        second_derivative = np.diff(recon, n=2, axis=1)
        roughness = float(np.sqrt(np.mean(second_derivative ** 2)))

        results[reg_key] = {
            "transform": transform,
            "image": recon,
            "max": maximum,
            "min": minimum,
            "negative_fraction": negative_fraction,
            "relative_negative_amplitude": relative_negative_amplitude,
            "roughness": roughness,
        }

    return results