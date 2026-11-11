"""Portable route-A input preparation matching the pinned Cellpose runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizedPlane:
    """One prepared channel and the statistics of its complete inference halo."""

    values: np.ndarray
    image_low: float
    image_high: float
    low: float
    high: float
    bounds_dtype: str
    stride: tuple[int, int, int]


def normalize_model_plane(
    plane: np.ndarray, *, image_bounds: tuple[float, float] | None = None
) -> NormalizedPlane:
    """Prepare one halo channel with the production uint8 and Cellpose steps.

    Percentiles are measured on the complete halo. Cropping the returned plane
    therefore preserves the same input as production inference.
    """
    image = np.asarray(plane, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Model input must be a finite two-dimensional image.")
    p2, p98 = (
        np.percentile(image, (2, 98))
        if image_bounds is None
        else (np.float64(image_bounds[0]), np.float64(image_bounds[1]))
    )
    quantized = (np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1) * 255).astype(np.uint8)
    del image
    channel = quantized[np.newaxis].astype(np.float32)
    stride = (
        (1, max(1, channel.shape[1] // 224), max(1, channel.shape[2] // 224))
        if channel.size > 224**3
        else (1, 1, 1)
    )
    sampled = channel[tuple(slice(None, None, step) for step in stride)]
    low, high = np.percentile(sampled, 1.0), np.percentile(sampled, 99.0)
    if high - low > 1e-3:
        channel -= low
        channel /= high - low
    else:
        channel.fill(0)
    return NormalizedPlane(
        channel[0],
        float(p2),
        float(p98),
        float(low),
        float(high),
        str(np.asarray(low).dtype),
        stride,
    )
