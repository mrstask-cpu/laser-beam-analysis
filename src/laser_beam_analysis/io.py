"""Image loading and validation for laser beam analysis.

Public API:
    load_image(source) -> np.ndarray
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageSource = Union[str, Path, np.ndarray]


def _require_finite_2d(image, name: str = "image") -> np.ndarray:
    """Validate a scientific 2D array.

    Raises ValueError if the input is not a finite, non-empty 2D array.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} должен быть двумерным; получена форма {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} пуст.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} содержит NaN или бесконечные значения.")
    return arr


def _load_raw(source: ImageSource) -> np.ndarray:
    """Load an image as 2D float64 without semantic validation.

    Unlike load_image(), this does NOT enforce a non-zero dynamic range.
    Used for calibration frames (dark / flat) which are legitimately
    allowed to be uniform.
    """
    try:
        if isinstance(source, np.ndarray):
            image = np.asarray(source).copy()
        elif isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.startswith(("http://", "https://")):
                with urllib.request.urlopen(source_str, timeout=15) as response:
                    image_bytes = response.read()
                with Image.open(io.BytesIO(image_bytes)) as pil_image:
                    image = np.asarray(pil_image).copy()
            else:
                source_path = Path(source_str)
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Файл изображения не найден: {source_path}"
                    )
                with Image.open(source_path) as pil_image:
                    image = np.asarray(pil_image).copy()
        else:
            raise TypeError("source должен быть путём/URL или numpy.ndarray.")
    except Exception as exc:
        raise RuntimeError(f"Не удалось загрузить изображение: {exc}") from exc

    if image.ndim == 3:
        if image.shape[-1] < 3:
            raise ValueError(
                f"Неподдерживаемая многоканальная форма: {image.shape}."
            )
        rgb = np.asarray(image[..., :3], dtype=np.float64)
        image = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
    elif image.ndim != 2:
        raise ValueError(
            f"Ожидалось 2D grayscale или RGB/RGBA изображение; форма {image.shape}."
        )

    image = np.asarray(image, dtype=np.float64)
    if image.size == 0:
        raise ValueError("Изображение пустое.")
    if not np.all(np.isfinite(image)):
        raise ValueError("Изображение содержит NaN или бесконечные значения.")
    return image


def load_image(source: ImageSource) -> np.ndarray:
    """Load a beam image as 2D float64.

    Accepts a path, an http(s) URL, or a numpy array. RGB/RGBA inputs are
    converted to grayscale using Rec.709 luma coefficients. Original
    dynamic range is preserved (no normalization).

    Unlike _load_raw(), this enforces that the image has a non-zero
    dynamic range, which is a meaningful requirement for beam images but
    not for calibration frames.
    """
    image = _load_raw(source)
    if not np.ptp(image) > 0:
        raise ValueError(
            "Изображение не имеет динамического диапазона: все пиксели одинаковы."
        )
    return image