"""Gaussian beam propagation analysis: observed waist and model fit.

Public API:
    find_beam_waist(profile_results, calibration=None) -> dict
    gaussian_beam_width_model(z, w0, z0, zR) -> np.ndarray
    fit_gaussian_beam_propagation(profile_results, pixel_size=None, robust=True)
        -> dict

Design notes:
    - find_beam_waist reports the *observed* minimum of w(z) with an optional
      sub-pixel quadratic refinement. This is distinct from the model-fitted
      waist position z0 returned by fit_gaussian_beam_propagation.
    - fit_gaussian_beam_propagation emits explicit identifiability warnings:
      z0 may lie outside the experimental range (extrapolation, not direct
      observation), zR may be much larger than the measured span, or the
      propagation model may not beat a constant-width baseline (ΔBIC).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares

from .calibration import resolve_pixel_scale_um, um_to_unit


def find_beam_waist(
    profile_results,
    calibration: dict | None = None,
) -> dict:
    """Locate the observed minimum of w(z) across successful slices.

    Returns a dict with:
        z0, w0                       – discrete minimum (also exposed as
                                       z_min_observed / w_min_observed)
        z_min_observed_subpixel,
        w_min_observed_subpixel      – local quadratic refinement (NaN if not
                                       available or the minimum is on a boundary)
        boundary_limited             – True if the minimum is at an endpoint
        subpixel_available           – True if the sub-pixel fit succeeded
        definition                   – string, describing the quantity
        z, w                         – the sampled arrays used

    Physical units:
        If calibration is enabled, z0/w0/subpixel values are returned in the
        calibration's display unit. Otherwise they stay in pixels.
    """
    valid = sorted(
        [
            x
            for x in profile_results
            if x.get("success", False)
            and np.isfinite(x.get("beam_radius_w", np.nan))
        ],
        key=lambda x: x["z_index"],
    )
    if len(valid) < 3:
        raise RuntimeError("Недостаточно успешных сечений для поиска waist.")

    z = np.array([x["z_index"] for x in valid], dtype=float)
    w = np.array([x["beam_radius_w"] for x in valid], dtype=float)

    i = int(np.argmin(w))
    z0 = float(z[i])
    w0 = float(w[i])
    boundary = i in (0, len(w) - 1)

    zsub = float("nan")
    wsub = float("nan")
    sub = False

    if not boundary:
        zl = z[i - 1:i + 2]
        wl = w[i - 1:i + 2]
        try:
            a, b, c = np.polyfit(zl, wl, 2)
            if np.isfinite(a) and a > 0:
                zv = -b / (2 * a)
                if zl[0] <= zv <= zl[-1]:
                    zsub = float(zv)
                    wsub = float(a * zv * zv + b * zv + c)
                    sub = bool(np.isfinite(wsub))
        except Exception:
            pass

    scale_um = resolve_pixel_scale_um(calibration) if calibration else None
    unit = (calibration or {}).get("display_unit", "px")

    if scale_um is None:
        zd, wd, zsd, wsd = z0, w0, zsub, wsub
    else:
        zd = um_to_unit(z0 * scale_um, unit)
        wd = um_to_unit(w0 * scale_um, unit)
        zsd = um_to_unit(zsub * scale_um, unit) if np.isfinite(zsub) else float("nan")
        wsd = um_to_unit(wsub * scale_um, unit) if np.isfinite(wsub) else float("nan")

    return {
        "z0": z0,
        "w0": w0,
        "z_min_observed": z0,
        "w_min_observed": w0,
        "z_min_observed_subpixel": zsub,
        "w_min_observed_subpixel": wsub,
        "boundary_limited": bool(boundary),
        "subpixel_available": bool(sub),
        "definition": (
            "discrete observed minimum of w(z); "
            "optional local quadratic refinement"
        ),
        "z": z,
        "w": w,
        # Display-unit versions, useful for reporting.
        "z0_display": float(zd),
        "w0_display": float(wd),
        "z_subpixel_display": float(zsd),
        "w_subpixel_display": float(wsd),
        "display_unit": unit,
    }


def gaussian_beam_width_model(z, w0, z0, zR):
    """Standard Gaussian-beam width model:

        w(z) = w0 * sqrt(1 + ((z - z0) / zR)^2)
    """
    z = np.asarray(z, dtype=float)
    return w0 * np.sqrt(1.0 + ((z - z0) / zR) ** 2)


def fit_gaussian_beam_propagation(
    profile_results,
    pixel_size: float | None = None,
    robust: bool = True,
) -> dict:
    """Fit Gaussian-beam propagation to reconstructed w(z).

    The model parameters are:
        w0  – beam radius at waist (in pixel or physical units)
        z0  – model-estimated waist position (may be outside the data range)
        zR  – Rayleigh range

    Identifiability diagnostics:
        R², RMSE, ΔBIC against a constant-width baseline, parameter standard
        deviations, correlation matrix, ratio zR / span, and an explicit check
        of whether z0 lies inside the measured range. The result also carries
        a list of human-readable warnings when the propagation model is not
        well-constrained.

    Args:
        profile_results: list of per-slice dicts (from analyze_all_z_profiles).
        pixel_size: optional px-to-physical unit factor. If provided, z and w
            are scaled before fitting (and returned scaled).
        robust: use a soft-L1 loss instead of plain least squares.

    Returns:
        dict with fit parameters, quality metrics, covariance, warnings, and
        the arrays used for fitting.
    """
    valid = [
        x
        for x in profile_results
        if x.get("success", False)
        and np.isfinite(x.get("beam_radius_w", np.nan))
    ]
    if len(valid) < 10:
        raise RuntimeError("Недостаточно точек для Gaussian-beam fit.")

    z = np.array([x["z_index"] for x in valid], dtype=float)
    w = np.array([x["beam_radius_w"] for x in valid], dtype=float)

    if pixel_size is not None:
        if pixel_size <= 0:
            raise ValueError("pixel_size должен быть > 0.")
        z_fit = z * pixel_size
        w_fit = w * pixel_size
    else:
        z_fit, w_fit = z, w

    z_min, z_max = float(z_fit.min()), float(z_fit.max())
    z_span = z_max - z_min
    if z_span <= 0:
        raise RuntimeError("Нет продольного диапазона для propagation fit.")

    k = int(np.argmin(w_fit))
    p0 = [
        max(float(w_fit[k]), np.finfo(float).eps),
        float(z_fit[k]),
        max(z_span, 1.0),
    ]
    lower = [np.finfo(float).eps, z_min - 20.0 * z_span, np.finfo(float).eps]
    upper = [np.inf, z_max + 20.0 * z_span, 1e6 * max(z_span, 1.0)]

    if robust:
        def residual_func(p):
            return gaussian_beam_width_model(z_fit, *p) - w_fit

        lsq = least_squares(
            residual_func,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(np.std(w_fit), 1e-6),
            max_nfev=100000,
        )
        popt = lsq.x
        fitted_w = gaussian_beam_width_model(z_fit, *popt)
        try:
            J = lsq.jac
            dof = max(len(w_fit) - 3, 1)
            s2 = max(2.0 * float(lsq.cost) / dof, np.finfo(float).eps)
            pcov = s2 * np.linalg.pinv(J.T @ J)
        except Exception:
            pcov = np.full((3, 3), np.nan)
    else:
        popt, pcov = curve_fit(
            gaussian_beam_width_model,
            z_fit,
            w_fit,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
        fitted_w = gaussian_beam_width_model(z_fit, *popt)

    w0, z0, zR = map(float, popt)
    residual = w_fit - fitted_w
    sse_g = float(np.sum(residual ** 2))
    sst = float(np.sum((w_fit - w_fit.mean()) ** 2))
    r2 = 1.0 - sse_g / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    w_const = np.full_like(w_fit, w_fit.mean())
    sse_c = float(np.sum((w_fit - w_const) ** 2))
    n = len(w_fit)
    bic_g = n * np.log(max(sse_g / n, np.finfo(float).eps)) + 3 * np.log(n)
    bic_c = n * np.log(max(sse_c / n, np.finfo(float).eps)) + np.log(n)
    delta_bic = float(bic_c - bic_g)
    relative_variation = float((w_fit.max() - w_fit.min()) / w_fit.mean())

    warnings_list: list = []
    if relative_variation < 1e-3:
        warnings_list.append(
            "Ширина практически постоянна (<0.1%); параметры z0 и zR "
            "практически неидентифицируемы."
        )

    if np.all(np.isfinite(pcov)):
        parameter_std = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_matrix = pcov / np.outer(parameter_std, parameter_std)
    else:
        parameter_std = np.full(3, np.nan)
        correlation_matrix = np.full((3, 3), np.nan)

    zR_relative_error = (
        parameter_std[2] / zR
        if np.isfinite(parameter_std[2]) and zR > 0
        else float("nan")
    )
    zR_to_span = zR / z_span if z_span > 0 else float("inf")
    z0_inside = bool(z_min <= z0 <= z_max)

    if not np.isfinite(r2) or r2 < 0.8:
        warnings_list.append(
            "Gaussian propagation model объясняет менее 80% вариации w(z)."
        )
    if not z0_inside:
        warnings_list.append(
            "Оцененный waist z0 находится за пределами исследованного "
            "диапазона: это экстраполяция, а не прямое наблюдение."
        )
    if delta_bic < 6:
        warnings_list.append(
            "Gaussian propagation model не даёт убедимого преимущества "
            "над constant-width."
        )
    if zR_to_span > 20:
        warnings_list.append(
            "Rayleigh range значительно больше исследованного диапазона; "
            "zR плохо идентифицируется."
        )
    if np.isfinite(zR_relative_error) and zR_relative_error > 0.5:
        warnings_list.append(
            "zR определяется с высокой относительной неопределённостью."
        )
    if relative_variation < 0.02:
        warnings_list.append(
            "Ширина пучка меняется менее чем на 2% по исследованному диапазону."
        )

    return {
        "success": True,
        "w0": w0,
        "z0": z0,
        "zR": zR,
        "parameter_std": parameter_std,
        "covariance": pcov,
        "correlation_matrix": correlation_matrix,
        "r_squared": float(r2),
        "rmse": rmse,
        "sse_gaussian": sse_g,
        "sse_constant": sse_c,
        "bic_gaussian": float(bic_g),
        "bic_constant": float(bic_c),
        "delta_bic": delta_bic,
        "relative_variation": relative_variation,
        "zR_relative_error": float(zR_relative_error),
        "zR_to_span": float(zR_to_span),
        "z0_inside_data": z0_inside,
        "warnings": warnings_list,
        "z": z_fit,
        "w": w_fit,
        "fitted_w": fitted_w,
        "residual": residual,
        "definition": "model-fitted Gaussian propagation waist",
    }