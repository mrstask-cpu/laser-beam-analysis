"""Radial profile extraction and Gaussian fitting.

Public API:
    extract_radial_profile(abel_image, z_index,
                           center_column=None, max_radius=None,
                           symmetric=True)
        -> (r, profile) or (r, profile, diagnostics)
    extract_symmetric_radial_profile(abel_image, z_index,
                                     center_column=None, max_radius=None)
        -> (r, profile, diagnostics)
    fit_gaussian_radial_profile(r, profile,
                                fit_fraction=0.05,
                                robust=True,
                                r0_max=3.0)
        -> dict

Design notes:
    - extract_radial_profile with symmetric=True averages left and right
      halves of the reconstruction, removing any dependence on the choice
      of side. symmetric=False is a diagnostic mode only.
    - fit_gaussian_radial_profile returns both signal-only and full-profile
      quality metrics: R²_signal reports fit quality in the region used by
      the optimizer, R²_full reports how the same Gaussian behaves across
      the entire available radial profile.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .io import _require_finite_2d


def extract_radial_profile(
    abel_image,
    z_index: int,
    center_column: Optional[float] = None,
    max_radius: Optional[float] = None,
    symmetric: bool = True,
):
    """Extract I(r) from an Abel reconstruction at a given z-slice.

    Args:
        abel_image: 2D reconstruction.
        z_index: row index along the propagation axis.
        center_column: sub-pixel x-position of the symmetry axis. If None:
            - symmetric=True uses the geometric center (w - 1) / 2.
            - symmetric=False requires an odd image width and uses w // 2.
        max_radius: optional cap on the radial range.
        symmetric: if True, average left and right halves. If False, take a
            one-sided profile (diagnostic mode).

    Returns:
        Always (r, profile). If you need the left/right side diagnostics,
        call extract_symmetric_radial_profile directly.
    """
    abel_image = _require_finite_2d(abel_image, "abel_image")
    h, w = abel_image.shape
    z_index = int(z_index)
    if not 0 <= z_index < h:
        raise IndexError(f"z_index должен быть в диапазоне 0...{h-1}.")

    if symmetric:
        return extract_symmetric_radial_profile(
            abel_image,
            z_index=z_index,
            center_column=center_column,
            max_radius=max_radius,
        )[:2]

    if center_column is None:
        if w % 2 == 0:
            raise ValueError(
                "Для одностороннего Abel-профиля ожидается нечётная ширина результата."
            )
        center_column = w // 2
    else:
        center_column = int(center_column)
    if not 0 <= center_column < w:
        raise ValueError("Некорректная центральная колонка.")

    profile = abel_image[z_index, center_column:].copy()
    r = np.arange(profile.size, dtype=np.float64)

    if max_radius is not None:
        max_radius = float(max_radius)
        if not np.isfinite(max_radius) or max_radius <= 0:
            raise ValueError("max_radius должен быть положительным.")
        keep = r <= max_radius
        r, profile = r[keep], profile[keep]

    if profile.size < 8:
        raise ValueError(
            "После ограничения радиуса осталось слишком мало точек для анализа."
        )
    return r, profile


def extract_symmetric_radial_profile(
    abel_image,
    z_index: int,
    center_column: Optional[float] = None,
    max_radius: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build I(r) as the average of the left and right halves of the reconstruction.

    Returns:
        r, profile, diagnostics where diagnostics contains the two raw sides
        (left_profile, right_profile), their normalized RMS difference, their
        correlation, and the effective radius_max used.
    """
    abel_image = _require_finite_2d(abel_image, "abel_image")
    h, w = abel_image.shape
    z_index = int(z_index)
    if not 0 <= z_index < h:
        raise IndexError(f"z_index должен быть в диапазоне 0...{h-1}.")

    center = (w - 1) / 2.0 if center_column is None else float(center_column)
    if not (0 <= center <= w - 1):
        raise ValueError("Некорректный center_column.")

    available_radius = min(center, (w - 1) - center)
    if max_radius is None:
        max_radius = available_radius
    max_radius = float(max_radius)
    if max_radius <= 0:
        raise ValueError("max_radius должен быть > 0.")
    max_radius = min(max_radius, available_radius)

    r = np.arange(0.0, np.floor(max_radius) + 1.0, 1.0)
    x_left = center - r
    x_right = center + r

    row = abel_image[z_index]
    left = np.interp(x_left, np.arange(w, dtype=float), row)
    right = np.interp(x_right, np.arange(w, dtype=float), row)

    profile = 0.5 * (left + right)

    denom = max(float(np.max(np.abs(profile))), np.finfo(float).eps)
    side_diff = left - right
    normalized_rms = float(np.sqrt(np.mean(side_diff ** 2)) / denom)
    corr = (
        float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else float("nan")
    )

    diagnostics = {
        "left_profile": left,
        "right_profile": right,
        "normalized_rms_difference": normalized_rms,
        "correlation": corr,
        "radius_max": float(max_radius),
    }
    return r, profile, diagnostics


def fit_gaussian_radial_profile(
    r,
    profile,
    fit_fraction: float = 0.05,
    robust: bool = True,
    r0_max: float = 3.0,
) -> dict:
    """Fit a Gaussian to a radial profile:

        I(r) = B + A * exp(-(r - r0)^2 / (2 * sigma^2))

    Derived quantities:
        R_1/e, R_1/e², D_1/e, D_1/e²
        beam radius w = R_1/e² = 2 sigma, beam diameter 2w
        FWHM = 2 sqrt(2 ln 2) sigma
        R²_signal, R²_full
        RMSE_signal, RMSE_full, NRMSE_full, max_abs_residual_full

    Args:
        r: 1D radius axis.
        profile: 1D radial intensity.
        fit_fraction: threshold (relative to the estimated amplitude) below
            which points are excluded from the optimizer.
        robust: use a soft-L1 loss instead of plain least squares.
        r0_max: upper bound for r0 in pixels.

    Returns:
        dict with fitted parameters, derived widths, quality metrics and the
        full-resolution fitted profile for downstream plotting.
    """
    r = np.asarray(r, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)

    if r.ndim != 1 or profile.ndim != 1:
        raise ValueError("r и profile должны быть 1D.")
    if r.size != profile.size:
        raise ValueError("r и profile должны иметь одинаковый размер.")
    if r.size < 8:
        raise ValueError("Слишком мало точек для Gaussian fit.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r содержит NaN/inf.")
    if not np.all(np.isfinite(profile)):
        raise ValueError("profile содержит NaN/inf.")
    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction должен быть между 0 и 1.")
    if r0_max <= 0:
        raise ValueError("r0_max должен быть > 0.")

    peak_index = int(np.argmax(profile))
    peak_value = float(profile[peak_index])
    if peak_value <= 0:
        raise ValueError("Профиль не содержит положительного сигнала.")

    # Baseline from the far tail.
    tail_start = max(int(0.8 * len(profile)), peak_index + 5)
    tail = profile[tail_start:]
    baseline_guess = float(np.median(tail)) if tail.size > 0 else 0.0

    signal = profile - baseline_guess
    amplitude_guess = max(float(np.max(signal)), np.finfo(float).eps)

    # Initial sigma from the 1/e level.
    target = baseline_guess + amplitude_guess / np.e
    above = np.where(profile >= target)[0]
    if above.size >= 2:
        radius_guess = float(r[above[-1]])
    else:
        radius_guess = max(1.0, float(r[min(peak_index + 3, len(r) - 1)]))
    sigma_guess = max(radius_guess / np.sqrt(2.0), 0.5)

    r0_lower = 0.0
    r0_upper = min(float(r[-1]), float(r0_max))
    if r0_upper <= r0_lower:
        raise ValueError("Недостаточный радиальный диапазон для оценки r0.")

    eps = min(1e-6, 0.1 * max(r0_upper, 1.0))
    r0_guess = float(np.clip(float(r[peak_index]), r0_lower + eps, r0_upper - eps))

    threshold = baseline_guess + fit_fraction * amplitude_guess
    fit_mask = profile >= threshold
    if np.count_nonzero(fit_mask) < 8:
        raise ValueError("Недостаточно точек для Gaussian fit.")

    fit_r = r[fit_mask]
    fit_y = profile[fit_mask]

    def gaussian_model(params, x):
        amplitude, r0, sigma, baseline = params
        return baseline + amplitude * np.exp(
            -((x - r0) ** 2) / (2.0 * sigma ** 2)
        )

    def residuals(params):
        return gaussian_model(params, fit_r) - fit_y

    initial = np.array([amplitude_guess, r0_guess, sigma_guess, baseline_guess])

    sigma_upper = max(2.0 * float(r[-1]), 1.0)
    lower = np.array([0.0, r0_lower, 0.05, -np.inf])
    upper = np.array([np.inf, r0_upper, sigma_upper, np.inf])

    if robust:
        fit = least_squares(
            residuals,
            x0=initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(np.std(fit_y), 1e-6),
        )
    else:
        fit = least_squares(
            residuals,
            x0=initial,
            bounds=(lower, upper),
        )

    if not fit.success:
        raise RuntimeError(f"Gaussian fit не сошёлся: {fit.message}")

    amplitude, r0, sigma_r, baseline = fit.x

    radius_1e = float(np.sqrt(2.0) * sigma_r)
    radius_1e2 = float(2.0 * sigma_r)
    diameter_1e = float(2.0 * radius_1e)
    diameter_1e2 = float(2.0 * radius_1e2)

    beam_radius_w = float(2.0 * sigma_r)
    beam_diameter_2w = float(2.0 * beam_radius_w)
    fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_r)
    peak_intensity = float(baseline + amplitude)

    fitted = gaussian_model(fit.x, r)
    residual = profile - fitted
    fitted_used = gaussian_model(fit.x, fit_r)

    def calculate_metrics(y, y_fit):
        residual_local = y - y_fit
        ss_res = float(np.sum(residual_local ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(residual_local ** 2)))
        data_range = float(np.max(y) - np.min(y))
        nrmse = rmse / data_range if data_range > 0 else float("nan")
        max_abs_residual = float(np.max(np.abs(residual_local)))
        return r_squared, rmse, nrmse, max_abs_residual

    r2_signal, rmse_signal, nrmse_signal, max_residual_signal = calculate_metrics(
        fit_y, fitted_used
    )
    r2_full, rmse_full, nrmse_full, max_residual_full = calculate_metrics(
        profile, fitted
    )

    return {
        "success": bool(fit.success),
        "message": fit.message,

        # Gaussian parameters
        "amplitude": float(amplitude),
        "peak_intensity": peak_intensity,
        "r0": float(r0),
        "sigma": float(sigma_r),
        "baseline": float(baseline),

        # Widths
        "radius_1e": radius_1e,
        "radius_1e2": radius_1e2,
        "diameter_1e": diameter_1e,
        "diameter_1e2": diameter_1e2,
        "beam_radius_w": beam_radius_w,
        "beam_diameter_2w": beam_diameter_2w,
        "fwhm": fwhm,

        # Quality
        "r_squared": r2_signal,
        "r_squared_signal": r2_signal,
        "r_squared_full": r2_full,
        "rmse": rmse_signal,
        "rmse_signal": rmse_signal,
        "rmse_full": rmse_full,
        "nrmse_signal": nrmse_signal,
        "nrmse_full": nrmse_full,
        "max_abs_residual_signal": max_residual_signal,
        "max_abs_residual_full": max_residual_full,

        # Fit internals
        "fit_mask": fit_mask,
        "r": r,
        "profile": profile,
        "fitted_profile": fitted,
        "residual": residual,
        "optimizer_cost": float(fit.cost),
        "optimizer_optimality": float(fit.optimality),
    }