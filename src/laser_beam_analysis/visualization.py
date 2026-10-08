"""Visualization: beam parameters, 2D cross-sections and 3D surfaces.

Public API:
    radial_profile_to_xy_surface(r, profile, xy_step=1.0, radius_max=None)
        -> (X, Y, I_xy)
    plot_beam_parameters(profile_results, pixel_size=None, unit="px",
                         radiometry=None, figsize=(12, 6), dpi=150) -> dict
    plot_3d_beam_cross_section(abel_image, z_index, ...) -> (X, Y, I_xy, diagnostics)
    interactive_3d_cross_section(abel_image, ...) -> None

The interactive viewer requires ipywidgets and an IPython kernel. If
ipywidgets is not available the function raises RuntimeError; all other
functions work without it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

from .calibration import LENGTH_UNITS_UM, resolve_pixel_scale_um
from .metrology import radiometric_convert_intensity
from .profiles import extract_symmetric_radial_profile


try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def radial_profile_to_xy_surface(
    r,
    profile,
    xy_step: float = 1.0,
    radius_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an axisymmetric I(r) into a 2D I(x, y) surface.

        I(x, y) = I(sqrt(x^2 + y^2))

    Values outside the reconstructed radius are set to NaN.
    """
    r = np.asarray(r, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)

    if r.ndim != 1 or profile.ndim != 1:
        raise ValueError("r и profile должны быть 1D.")
    if len(r) != len(profile):
        raise ValueError("r и profile должны иметь одинаковую длину.")
    if xy_step <= 0:
        raise ValueError("xy_step должен быть > 0.")

    if radius_max is None:
        radius_max = float(r[-1])
    if radius_max <= 0:
        raise ValueError("radius_max должен быть > 0.")

    xy = np.arange(-radius_max, radius_max + xy_step, xy_step)
    X, Y = np.meshgrid(xy, xy)
    R = np.sqrt(X ** 2 + Y ** 2)

    intensity_xy = np.interp(
        R.ravel(), r, profile, left=profile[0], right=0.0
    ).reshape(R.shape)

    intensity_xy[R > r[-1]] = np.nan
    return X, Y, intensity_xy


def plot_beam_parameters(
    profile_results,
    pixel_size: Optional[float] = None,
    unit: str = "px",
    radiometry: Optional[dict] = None,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> dict:
    """Plot longitudinal beam parameters across z-slices.

    Returns a dict with the raw arrays (in pixels) and the plotted arrays
    (converted to physical units if pixel_size is provided).
    """
    valid = np.array(
        [item.get("success", False) for item in profile_results], dtype=bool
    )
    z = np.array([item["z_index"] for item in profile_results], dtype=float)

    def get_array(name):
        return np.array(
            [item.get(name, np.nan) for item in profile_results], dtype=float
        )

    peak = get_array("peak_intensity")
    sigma = get_array("sigma")
    w = get_array("beam_radius_w")
    d1e = get_array("diameter_1e")
    d1e2 = get_array("diameter_1e2")
    fwhm = get_array("fwhm")
    r2_signal = get_array("r_squared_signal")
    r2_full = get_array("r_squared_full")
    nrmse_full = get_array("nrmse_full")

    if pixel_size is not None:
        z_plot = z * pixel_size
        sigma_plot = sigma * pixel_size
        w_plot = w * pixel_size
        d1e_plot = d1e * pixel_size
        d1e2_plot = d1e2 * pixel_size
        fwhm_plot = fwhm * pixel_size
        unit_text = unit
        x_label = f"Продольная координата, {unit}"
        width_label = f"Диаметр, {unit}"
        radius_label = f"Gaussian beam radius w, {unit}"
        sigma_label = f"Gaussian sigma, {unit}"
    else:
        z_plot, sigma_plot = z, sigma
        w_plot, d1e_plot, d1e2_plot, fwhm_plot = w, d1e, d1e2, fwhm
        unit_text = "px"
        x_label = "Продольная координата, px"
        width_label = "Диаметр, px"
        radius_label = "Gaussian beam radius w, px"
        sigma_label = "Gaussian sigma, px"

    peak_plot = radiometric_convert_intensity(peak, radiometry)
    intensity_label = (
        "Peak intensity, W/cm²"
        if radiometry and radiometry.get("enabled", False)
        else "Fitted peak intensity / relative signal"
    )

    plot_specs = [
        (peak_plot, intensity_label, "Продольная эволюция максимальной интенсивности", {}),
        (None, width_label, "Продольная эволюция ширины пучка", {"widths": True}),
        (w_plot, radius_label, "Gaussian beam radius w(z)", {}),
        (sigma_plot, sigma_label, "Gaussian sigma(z)", {}),
        (r2_signal, r"$R^2$", "Качество Gaussian approximation", {"quality": True}),
        (nrmse_full, "NRMSE", "Нормированная ошибка Gaussian fit", {}),
    ]

    for yvals, ylabel, title, extra in plot_specs:
        plt.figure(figsize=figsize, dpi=dpi)
        if extra.get("widths"):
            plt.plot(z_plot, d1e_plot, marker="o", markersize=3, label="D 1/e")
            plt.plot(z_plot, d1e2_plot, marker="o", markersize=3, label="D 1/e² = 2w")
            plt.plot(z_plot, fwhm_plot, marker="o", markersize=3, label="FWHM")
            plt.legend()
        elif extra.get("quality"):
            plt.plot(z_plot, r2_signal, marker="o", markersize=3, label=r"$R^2_{signal}$")
            plt.plot(z_plot, r2_full, marker="o", markersize=3, label=r"$R^2_{full}$")
            finite_r2 = np.concatenate(
                [r2_signal[np.isfinite(r2_signal)], r2_full[np.isfinite(r2_full)]]
            )
            if finite_r2.size:
                lo, hi = float(np.min(finite_r2)), float(np.max(finite_r2))
                span = max(hi - lo, 0.01)
                plt.ylim(lo - 0.1 * span, min(1.01, hi + 0.1 * span))
            plt.legend()
        else:
            plt.plot(z_plot, yvals, marker="o", markersize=3)
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "z": z,
        "valid": valid,
        "peak_intensity": peak,
        "sigma": sigma,
        "beam_radius_w": w,
        "diameter_1e": d1e,
        "diameter_1e2": d1e2,
        "fwhm": fwhm,
        "r_squared_signal": r2_signal,
        "r_squared_full": r2_full,
        "nrmse_full": nrmse_full,
        "z_plot": z_plot,
        "unit": unit_text,
    }


def plot_3d_beam_cross_section(
    abel_image,
    z_index: int,
    center_column: Optional[float] = None,
    xy_step: float = 0.5,
    radius_max: float = 40.0,
    elev: float = 35,
    azim: float = -60,
    clip_negative_for_display: bool = True,
    color_mode: str = "linear",
    color_gamma: float = 0.7,
    major_tick: float = 10.0,
    minor_tick: float = 5.0,
    calibration: Optional[dict] = None,
    display_unit: Optional[str] = None,
    figsize: tuple = (11, 8),
    dpi: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """3D cross-section of the beam with left/right symmetrization.

    Uses I(r) = (I_left(r) + I_right(r)) / 2 (Abel requires axial symmetry).
    Returns (X, Y, intensity_xy, side_consistency_diagnostics).

    xy_step applies only to the display grid interpolation and does not
    increase the physical resolution of the reconstruction.
    """
    r, profile, side_consistency = extract_symmetric_radial_profile(
        abel_image,
        z_index=z_index,
        center_column=center_column,
        max_radius=radius_max,
    )
    radius_max = float(r[-1])

    X, Y, intensity_xy = radial_profile_to_xy_surface(
        r, profile, xy_step=xy_step, radius_max=radius_max
    )

    intensity_display = (
        np.clip(intensity_xy, 0, None)
        if clip_negative_for_display
        else intensity_xy.copy()
    )

    finite_mask = np.isfinite(intensity_display)
    finite_values = intensity_display[finite_mask]
    if finite_values.size == 0:
        raise ValueError("В 3D-срезе нет конечных значений.")
    vmax = float(np.max(finite_values))
    if vmax <= 0:
        raise ValueError("Максимальная интенсивность <= 0.")

    cmap = plt.get_cmap("viridis")
    if color_mode == "linear":
        norm = colors.Normalize(vmin=0.0, vmax=vmax)
    elif color_mode == "power":
        if color_gamma <= 0:
            raise ValueError("color_gamma должен быть > 0.")
        norm = colors.PowerNorm(gamma=color_gamma, vmin=0.0, vmax=vmax)
    else:
        raise ValueError("color_mode должен быть 'linear' или 'power'.")

    safe_intensity = np.nan_to_num(intensity_display, nan=0.0)
    facecolors = cmap(norm(safe_intensity))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        intensity_display,
        facecolors=facecolors,
        rcount=X.shape[0],
        ccount=X.shape[1],
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    scale_um = resolve_pixel_scale_um(calibration) if calibration else None
    unit = display_unit or (calibration or {}).get("display_unit", "px")

    if scale_um is not None:
        unit_factor = scale_um / LENGTH_UNITS_UM[unit]
        ax.set_xlabel(f"x, {unit}")
        ax.set_ylabel(f"y, {unit}")
        lim_display = radius_max * unit_factor
        ax.set_xlim(-lim_display, lim_display)
        ax.set_ylim(-lim_display, lim_display)
        ax.xaxis.set_major_locator(MultipleLocator(major_tick * unit_factor))
        ax.yaxis.set_major_locator(MultipleLocator(major_tick * unit_factor))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick * unit_factor))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick * unit_factor))
        z_label = str(z_index)
    else:
        ax.set_xlabel("x, px")
        ax.set_ylabel("y, px")
        ax.set_xlim(-radius_max, radius_max)
        ax.set_ylim(-radius_max, radius_max)
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.yaxis.set_major_locator(MultipleLocator(major_tick))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))
        z_label = f"{z_index} px"

    ax.set_zlabel("Intensity / relative signal")
    ax.grid(True)
    ax.set_title(
        f"3D поперечный профиль, z = {z_label}\n"
        f"Post-symmetrization check: "
        f"RMS={side_consistency['normalized_rms_difference']:.3%}, "
        f"corr={side_consistency['correlation']:.5f}"
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.65))

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(finite_values)
    fig.colorbar(scalar_mappable, ax=ax, shrink=0.65, pad=0.10, label="Intensity")
    plt.tight_layout()
    plt.show()

    return X, Y, intensity_xy, side_consistency


def interactive_3d_cross_section(
    abel_image,
    center_column: Optional[float] = None,
    profile_results: Optional[list] = None,
    xy_step: float = 0.5,
    radius_max: float = 40.0,
    color_mode: str = "linear",
    color_gamma: float = 0.7,
    calibration: Optional[dict] = None,
    display_unit: Optional[str] = None,
) -> None:
    """Interactive z-slider for the 3D cross-section. Requires ipywidgets."""
    if not IPYWIDGETS_AVAILABLE:
        raise RuntimeError(
            "Интерактивный 3D-слайдер требует ipywidgets. "
            "Основной научный расчёт и статический 3D-график доступны без него."
        )

    abel_image = np.asarray(abel_image, dtype=np.float64)
    if abel_image.ndim != 2:
        raise ValueError("abel_image должен быть 2D.")

    h = abel_image.shape[0]
    slider = widgets.IntSlider(
        value=h // 2,
        min=0,
        max=h - 1,
        step=1,
        description="z:",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    info_label = widgets.HTML()
    output = widgets.Output()
    result_map = (
        {int(item["z_index"]): item for item in profile_results}
        if profile_results is not None
        else {}
    )

    def update(change=None):
        z_index = int(slider.value)
        try:
            r, profile, side_consistency = extract_symmetric_radial_profile(
                abel_image,
                z_index=z_index,
                center_column=center_column,
                max_radius=radius_max,
            )
            positive_profile = np.clip(profile, 0, None)
            text = (
                f"<b>z = {z_index} px</b> | "
                f"I<sub>max</sub> = {float(np.max(positive_profile)):.5g}"
            )
            if z_index in result_map and result_map[z_index].get("success", False):
                item = result_map[z_index]
                text += (
                    f" | w = {item['beam_radius_w']:.3f} px | "
                    f"2w = {item['beam_diameter_2w']:.3f} px | "
                    f"R²<sub>full</sub> = {item['r_squared_full']:.5f}"
                )
            elif z_index in result_map:
                text += " | Gaussian fit: неуспешен"
            text += (
                f" | left/right RMS = "
                f"{side_consistency['normalized_rms_difference']:.2%} | "
                f"corr = {side_consistency['correlation']:.5f}"
            )
            info_label.value = text
            with output:
                clear_output(wait=True)
                plot_3d_beam_cross_section(
                    abel_image,
                    z_index=z_index,
                    center_column=center_column,
                    xy_step=xy_step,
                    radius_max=radius_max,
                    color_mode=color_mode,
                    color_gamma=color_gamma,
                    major_tick=10.0,
                    minor_tick=5.0,
                    calibration=calibration,
                    display_unit=display_unit,
                )
        except Exception as exc:
            info_label.value = f"<b>Ошибка:</b> {exc}"
            with output:
                clear_output(wait=True)
                print(repr(exc))

    slider.observe(update, names="value")
    display(slider, info_label, output)
    update()