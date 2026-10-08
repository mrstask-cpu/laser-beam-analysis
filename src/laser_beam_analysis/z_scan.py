"""Z-axis sampling and batch Gaussian analysis across z-slices.

Public API:
    make_z_sampling(n_z, n_points=61, margin=10) -> np.ndarray
    get_z_positions(n_z, mode="dense", n_points=61, margin=10) -> np.ndarray
    analyze_all_z_profiles(abel_image, z_positions,
                           fit_fraction=0.05, robust=True, r0_max=3.0)
        -> list[dict]

analyze_all_z_profiles is tolerant to individual fit failures: failed
slices are recorded with success=False, error=<message>, and NaN for all
numeric fields, so downstream plotting stays robust.
"""
from __future__ import annotations

import numpy as np

from .io import _require_finite_2d
from .profiles import extract_radial_profile, fit_gaussian_radial_profile


# Numeric fields that are pre-filled with NaN and later overwritten by the
# fit result when the fit succeeds.
_FIT_METRIC_NAMES = (
    "amplitude",
    "peak_intensity",
    "r0",
    "sigma",
    "baseline",
    "radius_1e",
    "radius_1e2",
    "diameter_1e",
    "diameter_1e2",
    "beam_radius_w",
    "beam_diameter_2w",
    "fwhm",
    "r_squared",
    "r_squared_signal",
    "r_squared_full",
    "rmse",
    "rmse_signal",
    "rmse_full",
    "nrmse_signal",
    "nrmse_full",
    "max_abs_residual_signal",
    "max_abs_residual_full",
)


def make_z_sampling(
    n_z: int,
    n_points: int = 61,
    margin: int = 10,
) -> np.ndarray:
    """Create a uniform set of longitudinal z-indices.

    Args:
        n_z: total number of available rows.
        n_points: number of z-slices to pick.
        margin: number of rows to exclude at each end.

    Returns:
        Sorted unique integer z-indices.
    """
    n_z = int(n_z)
    n_points = int(n_points)
    margin = int(margin)

    if n_z < 3:
        raise ValueError("n_z должен быть >= 3.")
    if n_points < 2:
        raise ValueError("n_points должен быть >= 2.")
    if margin < 0:
        raise ValueError("margin не может быть отрицательным.")

    first = margin
    last = n_z - 1 - margin

    if first > last:
        raise ValueError(
            f"Слишком большой margin={margin}. "
            f"Для изображения с n_z={n_z} "
            f"допустимый диапазон margin: 0...{(n_z - 1) // 2}."
        )

    available_positions = last - first + 1
    if n_points > available_positions:
        raise ValueError(
            f"Нельзя выбрать {n_points} уникальных "
            f"z-позиций из {available_positions} доступных. "
            f"Уменьшите n_points или margin."
        )

    positions = np.rint(np.linspace(first, last, n_points)).astype(int)
    positions = np.unique(positions)

    if len(positions) != n_points:
        raise RuntimeError(
            "При округлении координат появились дубли. Уменьшите n_points."
        )
    return positions


def get_z_positions(
    n_z: int,
    mode: str = "dense",
    n_points: int = 61,
    margin: int = 10,
) -> np.ndarray:
    """Unified selection of longitudinal z-indices.

    Modes:
        "diagnostic" -> up to 15 points
        "dense"      -> n_points
        "full"       -> all admissible rows
    """
    mode = str(mode).lower()

    if mode == "diagnostic":
        return make_z_sampling(
            n_z=n_z,
            n_points=min(15, n_z - 2 * margin),
            margin=margin,
        )
    if mode == "dense":
        return make_z_sampling(n_z=n_z, n_points=n_points, margin=margin)
    if mode == "full":
        first = int(margin)
        last = int(n_z - 1 - margin)
        if first > last:
            raise ValueError(f"margin={margin} слишком велик для n_z={n_z}.")
        return np.arange(first, last + 1, dtype=int)

    raise ValueError(
        "Неизвестный режим z sampling. "
        "Используйте: 'diagnostic', 'dense' или 'full'."
    )


def analyze_all_z_profiles(
    abel_image,
    z_positions,
    fit_fraction: float = 0.05,
    robust: bool = True,
    r0_max: float = 3.0,
) -> list:
    """Fit a Gaussian profile at each of the given z-positions.

    Failed fits do not abort the batch. For a failed slice the entry has
    success=False, error=<message>, and NaN for all numeric fields.

    Args:
        abel_image: 2D Abel reconstruction.
        z_positions: iterable of row indices.
        fit_fraction, robust, r0_max: forwarded to fit_gaussian_radial_profile.

    Returns:
        list of per-slice result dicts.

    Raises:
        RuntimeError: if none of the slices produced a successful fit.
    """
    abel_image = _require_finite_2d(abel_image, "abel_image")
    z_positions = np.asarray(z_positions, dtype=int)

    results: list = []

    for z_index in z_positions:
        result = {
            "z_index": int(z_index),
            "success": False,
            "error": None,
        }

        # Pre-fill numeric fields with NaN so plotting is robust on failure.
        for name in _FIT_METRIC_NAMES:
            result[name] = float("nan")

        try:
            r, profile = extract_radial_profile(abel_image, z_index=z_index)
            fit_result = fit_gaussian_radial_profile(
                r,
                profile,
                fit_fraction=fit_fraction,
                robust=robust,
                r0_max=r0_max,
            )
            result.update(fit_result)
            result["success"] = True
        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

    valid_count = sum(item["success"] for item in results)
    if valid_count == 0:
        raise RuntimeError("Ни одного успешного Gaussian fit.")

    return results