#!/usr/bin/env python3
"""Utility helpers for the Xenium/MERSCOPE Napari comparison viewer."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from hashlib import blake2s
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import zarr
from shapely.geometry import box

try:
    from skimage.draw import polygon as draw_polygon
except Exception:  # pragma: no cover - import error depends on runtime env
    draw_polygon = None

log = logging.getLogger("napari_compare_utils")

DERIVED_CACHE_PREFIX = "_napari_compare_"
DERIVED_CACHE_ATTR = "napari_compare_derived_cache"
CELLPOSE_SHAPE_KEY = "MOSAIK_cellpose"
CELLPOSE_LABEL_KEY = "MOSAIK_cellpose_labels"
CELLPOSE_QUANTIFICATION_TABLE_KEY = "table_MOSAIK_cellpose_image_quantification"
CELLPOSE_VALUE_BINS = 256
CELLPOSE_VALUE_CLIP_PERCENTILES = (1.0, 99.0)
TRANSPARENT_RGBA = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CellposeQuantificationFeature:
    """One image-statistic column in the Cellpose quantification table."""

    feature: str
    image_key: str
    channel: str
    statistic: str
    column_index: int


@dataclass(frozen=True)
class CellposeQuantificationValues:
    """Label ids and selected per-cell values from the Cellpose quantification table."""

    feature: CellposeQuantificationFeature
    label_ids: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class BinnedLabelColorMapping:
    """Direct label colors plus numeric scaling metadata for a value overlay."""

    color_dict: dict[int | None, tuple[float, float, float, float]]
    clip_low: float
    clip_high: float
    finite_count: int
    label_count: int
    bin_count: int
    unique_color_count: int
    lower_percentile: float
    upper_percentile: float


def _safe_cache_token(value: str, max_len: int = 96) -> str:
    """Return a zarr-element-safe token that stays readable for common keys."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not token:
        token = "cache"
    if token == str(value) and len(token) <= max_len:
        return token

    digest = blake2s(str(value).encode("utf-8"), digest_size=5).hexdigest()
    token = token[: max(1, max_len - len(digest) - 3)].strip("_")
    return f"{token}__h{digest}" if token else f"h{digest}"


def is_derived_cache_key(key: str) -> bool:
    """Return True for private viewer-derived zarr elements."""
    return str(key).startswith(DERIVED_CACHE_PREFIX)


def derived_outline_cache_key(label_key: str, width: int) -> str:
    """Build the private labels cache key for a precomputed outline pyramid."""
    return f"{DERIVED_CACHE_PREFIX}outline__{_safe_cache_token(label_key)}__w{int(width)}"


def _format_float_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def derived_transcript_density_cache_key(points_key: str, bin_um: float) -> str:
    """Build the private images cache key for transcript density."""
    return f"{DERIVED_CACHE_PREFIX}tx_density__{_safe_cache_token(points_key)}__bin{_format_float_token(bin_um)}"


def open_zarr_group_unconsolidated(zarr_path: str | Path):
    """Open a zarr group without consolidated metadata.

    Some SpatialData stores can have stale consolidated metadata after new
    arrays are written. Reading direct metadata avoids false missing-key errors.
    """
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _zarr_path_exists(root, path: str) -> bool:
    try:
        root[path]
        return True
    except Exception:
        return False


def _read_zarr_string_list(array_like) -> list[str]:
    values = array_like[:]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [str(value) for value in values]


def _cellpose_quantification_table(root):
    return root[f"tables/{CELLPOSE_QUANTIFICATION_TABLE_KEY}"]


def _cellpose_quantification_features_from_table(table) -> list[CellposeQuantificationFeature]:
    image_keys = _read_zarr_string_list(table["var/image_key"])
    channels = _read_zarr_string_list(table["var/channel"])
    statistics = _read_zarr_string_list(table["var/statistic"])
    if not (len(image_keys) == len(channels) == len(statistics)):
        raise ValueError("Cellpose quantification var columns have inconsistent lengths.")

    try:
        features = _read_zarr_string_list(table["var/feature"])
    except Exception:
        features = [
            f"{image_key}__{channel}__{statistic}"
            for image_key, channel, statistic in zip(image_keys, channels, statistics, strict=True)
        ]
    if len(features) != len(channels):
        raise ValueError("Cellpose quantification feature column length does not match var metadata.")

    return [
        CellposeQuantificationFeature(
            feature=str(feature),
            image_key=str(image_key),
            channel=str(channel),
            statistic=str(statistic),
            column_index=int(idx),
        )
        for idx, (feature, image_key, channel, statistic) in enumerate(
            zip(features, image_keys, channels, statistics, strict=True)
        )
    ]


def cellpose_quantification_table_available(zarr_path: str | Path) -> bool:
    """Return whether the hard-coded Cellpose value-overlay inputs are present."""
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        table = _cellpose_quantification_table(root)
        required = (
            "obs/label_id",
            "var/image_key",
            "var/channel",
            "var/statistic",
            "X",
        )
        has_required_table_parts = all(_zarr_path_exists(table, key) for key in required)
        has_region = _zarr_path_exists(root, f"labels/{CELLPOSE_LABEL_KEY}") or _zarr_path_exists(
            root,
            f"shapes/{CELLPOSE_SHAPE_KEY}",
        )
        return bool(has_required_table_parts and has_region)
    except Exception:
        return False


def cellpose_quantification_features(zarr_path: str | Path) -> list[CellposeQuantificationFeature]:
    """List available Cellpose image-statistic quantification columns."""
    root = open_zarr_group_unconsolidated(zarr_path)
    return _cellpose_quantification_features_from_table(_cellpose_quantification_table(root))


def resolve_cellpose_quantification_feature(
    zarr_path: str | Path,
    channel: str,
    statistic: str,
) -> CellposeQuantificationFeature:
    """Resolve a selected channel/statistic to a quantification table column."""
    channel_key = str(channel).casefold()
    statistic_key = str(statistic).casefold()
    for feature in cellpose_quantification_features(zarr_path):
        if feature.channel.casefold() == channel_key and feature.statistic.casefold() == statistic_key:
            return feature
    raise KeyError(f"No Cellpose quantification feature for channel={channel!r}, statistic={statistic!r}.")


def load_cellpose_quantification_values(
    zarr_path: str | Path,
    channel: str,
    statistic: str,
) -> CellposeQuantificationValues:
    """Load label ids and one selected Cellpose quantification column."""
    root = open_zarr_group_unconsolidated(zarr_path)
    table = _cellpose_quantification_table(root)
    feature = None
    channel_key = str(channel).casefold()
    statistic_key = str(statistic).casefold()
    for candidate in _cellpose_quantification_features_from_table(table):
        if candidate.channel.casefold() == channel_key and candidate.statistic.casefold() == statistic_key:
            feature = candidate
            break
    if feature is None:
        raise KeyError(f"No Cellpose quantification feature for channel={channel!r}, statistic={statistic!r}.")

    label_ids = np.asarray(table["obs/label_id"][:], dtype=np.uint32)
    values = np.asarray(table["X"][:, feature.column_index], dtype=np.float32)
    if label_ids.shape[0] != values.shape[0]:
        raise ValueError(
            "Cellpose quantification label/value length mismatch: "
            f"{label_ids.shape[0]} labels vs {values.shape[0]} values."
        )
    return CellposeQuantificationValues(feature=feature, label_ids=label_ids, values=values)


def cellpose_value_clip_range(
    values,
    lower_percentile: float = CELLPOSE_VALUE_CLIP_PERCENTILES[0],
    upper_percentile: float = CELLPOSE_VALUE_CLIP_PERCENTILES[1],
) -> tuple[float, float]:
    """Return finite percentile clip limits for per-cell value coloring."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite Cellpose quantification values are available.")

    low, high = np.percentile(
        finite,
        [float(lower_percentile), float(upper_percentile)],
    )
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Could not compute finite Cellpose quantification percentiles.")
    return float(low), float(high)


def build_binned_label_color_dict(
    label_ids,
    values,
    colors_rgba,
    lower_percentile: float = CELLPOSE_VALUE_CLIP_PERCENTILES[0],
    upper_percentile: float = CELLPOSE_VALUE_CLIP_PERCENTILES[1],
) -> BinnedLabelColorMapping:
    """Map label ids to a bounded set of RGBA colors from scalar values."""
    labels = np.asarray(label_ids)
    values_arr = np.asarray(values, dtype=np.float64)
    if labels.shape[0] != values_arr.shape[0]:
        raise ValueError(f"Expected one value per label id, got {labels.shape[0]} labels and {values_arr.shape[0]} values.")

    color_lut = np.asarray(colors_rgba, dtype=np.float32)
    if color_lut.ndim != 2 or color_lut.shape[0] == 0 or color_lut.shape[1] not in (3, 4):
        raise ValueError("colors_rgba must be an array with shape (N, 3) or (N, 4).")
    if color_lut.shape[1] == 3:
        color_lut = np.column_stack([color_lut, np.ones(color_lut.shape[0], dtype=np.float32)])

    clip_low, clip_high = cellpose_value_clip_range(
        values_arr,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    finite = np.isfinite(values_arr) & (labels > 0)
    finite_count = int(np.count_nonzero(finite))
    color_dict: dict[int | None, tuple[float, float, float, float]] = {
        0: TRANSPARENT_RGBA,
        None: TRANSPARENT_RGBA,
    }

    if finite_count > 0:
        if clip_high <= clip_low:
            bin_indices = np.full(finite_count, min(color_lut.shape[0] // 2, color_lut.shape[0] - 1), dtype=np.int64)
        else:
            normalized = (values_arr[finite] - clip_low) / (clip_high - clip_low)
            normalized = np.clip(normalized, 0.0, 1.0)
            bin_indices = np.floor(normalized * (color_lut.shape[0] - 1)).astype(np.int64, copy=False)

        for label, bin_index in zip(labels[finite], bin_indices, strict=True):
            rgba = color_lut[int(bin_index), :4]
            color_dict[int(label)] = (
                float(rgba[0]),
                float(rgba[1]),
                float(rgba[2]),
                float(rgba[3]),
            )

    unique_color_count = len({tuple(color) for color in color_dict.values()})
    return BinnedLabelColorMapping(
        color_dict=color_dict,
        clip_low=float(clip_low),
        clip_high=float(clip_high),
        finite_count=finite_count,
        label_count=int(labels.shape[0]),
        bin_count=int(color_lut.shape[0]),
        unique_color_count=int(unique_color_count),
        lower_percentile=float(lower_percentile),
        upper_percentile=float(upper_percentile),
    )


def first_existing_col(df_like, candidates: Iterable[str]) -> str | None:
    """Return the first existing column name from candidates."""
    cols = set(map(str, list(df_like.columns)))
    for col in candidates:
        if col in cols:
            return col
    return None


def pick_default_shape_key(shape_keys: Iterable[str]) -> str | None:
    """Pick the default startup shape key using a stable preference order."""
    keys = sorted(str(k) for k in shape_keys)
    if not keys:
        return None

    for preferred in (
        "table_MOSAIK_proseg",
        "MOSAIK_proseg",
        "cell_boundaries",
        "table_original",
    ):
        if preferred in keys:
            return preferred
        for key in keys:
            if key.startswith(preferred + "_"):
                return key
    return keys[0]


def make_layer_name(dataset: str, layer_type: str, key: str, channel: str | None = None) -> str:
    """Build a standardized layer name for this viewer."""
    parts = [str(dataset).upper(), str(layer_type), str(key)]
    if channel is not None:
        parts.append(str(channel))
    return " | ".join(parts)


def layer_name_prefix(dataset: str, layer_type: str, key: str | None = None) -> str:
    """Build a prefix used to find/remove related layers."""
    parts = [str(dataset).upper(), str(layer_type)]
    if key is not None:
        parts.append(str(key))
    return " | ".join(parts)


def matching_layer_names(layer_names: Iterable[str], prefix: str) -> list[str]:
    """Return layer names matching a standardized prefix."""
    out: list[str] = []
    for name in layer_names:
        s = str(name)
        if s == prefix or s.startswith(prefix + " | "):
            out.append(s)
    return out


def to_pandas(df_like) -> pd.DataFrame:
    """Convert a pandas/dask-like dataframe to an in-memory pandas DataFrame."""
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if hasattr(df_like, "compute"):
        return df_like.compute()
    return pd.DataFrame(df_like).copy()


def assignment_mask(series_like) -> pd.Series:
    """Infer assigned/unassigned transcript status from a column."""
    series = pd.Series(series_like)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0

    as_str = series.astype("string")
    bad_values = {"", "0", "-1", "nan", "None", "<NA>"}
    return as_str.notna() & ~as_str.isin(bad_values)


def _dataarray_from_image_node(node):
    """Return the underlying xarray DataArray from a SpatialData image node."""
    if hasattr(node, "ds"):
        if "image" in node.ds:
            return node.ds["image"]
        if len(node.ds.data_vars) > 0:
            return next(iter(node.ds.data_vars.values()))

    return node


def _scale_sort_key(scale_name: str) -> tuple[int, int | str]:
    """Sort scale0, scale1, ... numerically before non-standard keys."""
    match = re.search(r"(\d+)$", str(scale_name))
    if match is not None:
        return (0, int(match.group(1)))
    return (1, str(scale_name))


def image_scale_dataarrays(image_elem) -> list[tuple[str, object]]:
    """Return all available image scale DataArrays from a SpatialData image element."""
    if hasattr(image_elem, "keys"):
        scales: list[tuple[str, object]] = []
        for key in image_elem.keys():
            key_str = str(key)
            if not key_str.startswith("scale"):
                continue
            scales.append((key_str, _dataarray_from_image_node(image_elem[key])))
        if len(scales) > 0:
            return sorted(scales, key=lambda item: _scale_sort_key(item[0]))

    return [("scale0", _dataarray_from_image_node(image_elem))]


def get_scale0_dataarray(image_elem):
    """Return the finest-resolution DataArray from a SpatialData image element."""
    scales = image_scale_dataarrays(image_elem)
    for name, dataarray in scales:
        if name == "scale0":
            return dataarray
    return scales[0][1]


def ensure_cyx(image_da):
    """Normalize a DataArray to (c, y, x) dimensions."""
    for dim in ("z", "Z"):
        if dim in image_da.dims:
            if int(image_da.sizes[dim]) != 1:
                raise ValueError(
                    f"Unsupported image dims with non-singleton {dim}: "
                    f"{tuple(str(d) for d in image_da.dims)}"
                )
            image_da = image_da.isel({dim: 0}, drop=True)

    dims = tuple(str(d) for d in image_da.dims)

    if all(d in dims for d in ("c", "y", "x")):
        return image_da.transpose("c", "y", "x")
    if all(d in dims for d in ("y", "x", "c")):
        return image_da.transpose("c", "y", "x")
    if all(d in dims for d in ("y", "x")):
        return image_da.expand_dims(c=["c0"]).transpose("c", "y", "x")

    raise ValueError(f"Unsupported image dims for channel extraction: {dims}")


def channel_labels(image_cyx) -> list[str]:
    """Get channel labels from a (c, y, x) image DataArray."""
    if "c" in image_cyx.coords:
        return [str(c) for c in image_cyx.coords["c"].values]
    return [f"c{i}" for i in range(int(image_cyx.sizes.get("c", 1)))]


def _coords_origin_step(coords) -> tuple[float, float]:
    """Infer origin and step for a monotonic coordinate array."""
    if coords is None:
        return 0.0, 1.0

    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0
    if arr.size == 1:
        return float(arr[0]), 1.0

    diffs = np.diff(arr)
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=1e-3, atol=1e-6):
        step = float(diffs[0])

    return float(arr[0]), float(step)


def build_napari_affine_from_px_to_um(
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    x_coords=None,
    y_coords=None,
) -> np.ndarray:
    """Build a 3x3 affine for napari (row/col -> y/x) from px->um transforms."""
    a, b, c = map(float, x_transform)
    d, e, f = map(float, y_transform)

    x_origin, x_step = _coords_origin_step(x_coords)
    y_origin, y_step = _coords_origin_step(y_coords)

    # Napari uses (row=y_idx, col=x_idx) order.
    # Convert idx -> pixel coords -> micron coords in one matrix.
    return np.array(
        [
            [e * y_step, d * x_step, d * x_origin + e * y_origin + f],  # y_um
            [b * y_step, a * x_step, a * x_origin + b * y_origin + c],  # x_um
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def resolve_dataset_mask_affine(
    dataset_name: str,
    merscope_transform_path: str | Path | None = None,
    xenium_spec_path: str | Path | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve pixel->micron affine transform for a dataset."""
    ds = str(dataset_name).upper()

    if ds == "MERSCOPE":
        path = Path(merscope_transform_path) if merscope_transform_path else None
        if path is not None and path.exists():
            matrix = np.loadtxt(path)
            inv = np.linalg.inv(matrix)
            return (
                (float(inv[0, 0]), float(inv[0, 1]), float(inv[0, 2])),
                (float(inv[1, 0]), float(inv[1, 1]), float(inv[1, 2])),
            )

        log.warning(
            "[MERSCOPE] Transform file missing; using fallback 0.108 um/px isotropic scale."
        )
        return (0.108, 0.0, 0.0), (0.0, 0.108, 0.0)

    if ds == "XENIUM":
        mpp = None
        path = Path(xenium_spec_path) if xenium_spec_path else None
        if path is not None and path.exists():
            try:
                spec = json.loads(path.read_text())
                if "pixel_size" in spec:
                    mpp = float(spec["pixel_size"])
            except Exception as exc:
                log.warning("[XENIUM] Failed to parse spec file %s (%s)", path, exc)

        if mpp is None:
            mpp = 0.2125
            log.warning(
                "[XENIUM] Spec file missing/unreadable; using fallback pixel_size=%s um.",
                mpp,
            )

        return (float(mpp), 0.0, 0.0), (0.0, float(mpp), 0.0)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_points_dataframe(
    points_obj,
    x_col: str,
    y_col: str,
    assignment_col: str | None = None,
    max_points: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load selected points columns into pandas with optional unbiased sampling."""
    cols = [x_col, y_col] + ([assignment_col] if assignment_col is not None else [])
    work = points_obj[cols]

    if max_points is None:
        return to_pandas(work)

    if hasattr(work, "npartitions") and hasattr(work, "compute"):
        total = int(work.map_partitions(len, meta=("n", "i8")).sum().compute())
        if total <= max_points:
            pdf = work.compute()
        else:
            frac = float(max_points) / float(total)
            pdf = work.sample(frac=frac, random_state=random_state).compute()
        if len(pdf) > max_points:
            pdf = pdf.sample(n=max_points, random_state=random_state)
        return pdf

    pdf = to_pandas(work)
    if len(pdf) > max_points:
        pdf = pdf.sample(n=max_points, random_state=random_state)
    return pdf


def load_viewport_points_dataframe(
    points_obj,
    x_col: str,
    y_col: str,
    assignment_col: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    sample_percent: float | None = None,
    max_points: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int, float]:
    """Load a sampled point table for the current viewport.

    ``bounds`` are global (min_x, min_y, max_x, max_y). When ``sample_percent``
    is ``None`` this loads as many viewport points as possible up to
    ``max_points``. Otherwise it loads that percentage, still respecting
    ``max_points`` as a hard display cap.
    """
    cols = [x_col, y_col] + ([assignment_col] if assignment_col is not None else [])
    work = points_obj[cols]

    if bounds is not None:
        min_x, min_y, max_x, max_y = (float(v) for v in bounds)
        work = work[
            (work[x_col] >= min_x)
            & (work[x_col] <= max_x)
            & (work[y_col] >= min_y)
            & (work[y_col] <= max_y)
        ]

    if max_points is not None:
        max_points = max(0, int(max_points))
    if sample_percent is not None:
        sample_percent = min(100.0, max(0.0, float(sample_percent)))

    if hasattr(work, "npartitions") and hasattr(work, "compute"):
        total = int(work.map_partitions(len, meta=("n", "i8")).sum().compute())
        if total == 0 or max_points == 0 or sample_percent == 0:
            return pd.DataFrame(columns=cols), total, 0.0

        if sample_percent is None:
            frac = 1.0 if max_points is None else min(1.0, float(max_points) / float(total))
        else:
            frac = sample_percent / 100.0
            if max_points is not None:
                frac = min(frac, float(max_points) / float(total))
            frac = min(1.0, max(0.0, frac))

        if frac < 1.0:
            pdf = work.sample(frac=frac, random_state=random_state).compute()
        else:
            pdf = work.compute()

        if len(pdf) == 0 and total > 0 and frac > 0:
            try:
                pdf = work.head(1, npartitions=-1, compute=True)
            except Exception:
                pass

        if max_points is not None and len(pdf) > max_points:
            pdf = pdf.sample(n=max_points, random_state=random_state)
        loaded_fraction = 0.0 if total == 0 else float(len(pdf)) / float(total)
        return pdf, total, loaded_fraction

    pdf = to_pandas(work)
    total = int(len(pdf))
    if total == 0 or max_points == 0 or sample_percent == 0:
        return pdf.iloc[:0].copy(), total, 0.0

    if sample_percent is None:
        n = total if max_points is None else min(total, max_points)
    else:
        n = int(np.ceil(total * (sample_percent / 100.0)))
        if max_points is not None:
            n = min(n, max_points)
        n = min(total, max(1, n))

    if n < total:
        pdf = pdf.sample(n=n, random_state=random_state)
    loaded_fraction = 0.0 if total == 0 else float(len(pdf)) / float(total)
    return pdf, total, loaded_fraction


def _point_columns(x_col: str, y_col: str, assignment_col: str | None = None) -> list[str]:
    return [x_col, y_col] + ([assignment_col] if assignment_col is not None else [])


def _iter_point_partitions(points_obj, cols: list[str]):
    work = points_obj[cols]
    if hasattr(work, "to_delayed"):
        for part in work.to_delayed():
            pdf = part.compute()
            if len(pdf) > 0:
                yield pdf
        return

    pdf = to_pandas(work)
    if len(pdf) > 0:
        yield pdf


def _point_bounds(points_obj, x_col: str, y_col: str) -> tuple[float, float, float, float]:
    work = points_obj[[x_col, y_col]]
    if hasattr(work, "compute") and hasattr(work, "npartitions"):
        min_x = float(work[x_col].min().compute())
        min_y = float(work[y_col].min().compute())
        max_x = float(work[x_col].max().compute())
        max_y = float(work[y_col].max().compute())
    else:
        pdf = to_pandas(work)
        min_x = float(pdf[x_col].min())
        min_y = float(pdf[y_col].min())
        max_x = float(pdf[x_col].max())
        max_y = float(pdf[y_col].max())

    if not np.all(np.isfinite([min_x, min_y, max_x, max_y])):
        raise ValueError("Transcript coordinates do not contain finite bounds.")
    return min_x, min_y, max_x, max_y


def adjusted_density_bin_um(
    bounds: tuple[float, float, float, float],
    requested_bin_um: float,
    max_pixels: int,
) -> tuple[float, int, int]:
    """Return a bin size and image shape that obey the requested pixel cap."""
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    bin_um = max(float(requested_bin_um), np.finfo(float).eps)
    max_pixels = max(1, int(max_pixels))

    width = max(max_x - min_x, bin_um)
    height = max(max_y - min_y, bin_um)
    nx = max(1, int(np.ceil(width / bin_um)))
    ny = max(1, int(np.ceil(height / bin_um)))
    if nx * ny > max_pixels:
        bin_um = max(bin_um, float(np.sqrt((width * height) / float(max_pixels))))
        nx = max(1, int(np.ceil(width / bin_um)))
        ny = max(1, int(np.ceil(height / bin_um)))

    return float(bin_um), int(ny), int(nx)


def compute_transcript_density_array(
    points_obj,
    x_col: str,
    y_col: str,
    assignment_col: str | None = None,
    bin_um: float = 4.0,
    max_pixels: int = 25_000_000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rasterize transcript point counts into assigned/unassigned density channels."""
    cols = _point_columns(x_col, y_col, assignment_col)
    bounds = _point_bounds(points_obj, x_col, y_col)
    actual_bin_um, ny, nx = adjusted_density_bin_um(bounds, bin_um, max_pixels)
    min_x, min_y, max_x, max_y = bounds

    density = np.zeros((2, ny, nx), dtype=np.uint32)
    total = 0
    assigned_total = 0
    unassigned_total = 0

    for pdf in _iter_point_partitions(points_obj, cols):
        x_vals = pdf[x_col].to_numpy(dtype=np.float64, copy=False)
        y_vals = pdf[y_col].to_numpy(dtype=np.float64, copy=False)
        good = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(good):
            continue
        x_vals = x_vals[good]
        y_vals = y_vals[good]

        ix = np.floor((x_vals - min_x) / actual_bin_um).astype(np.int64, copy=False)
        iy = np.floor((y_vals - min_y) / actual_bin_um).astype(np.int64, copy=False)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        if assignment_col is not None and assignment_col in pdf.columns:
            assigned = assignment_mask(pdf.loc[good, assignment_col]).to_numpy(dtype=bool, copy=False)
        else:
            assigned = np.ones(ix.shape[0], dtype=bool)

        if np.any(assigned):
            np.add.at(density[0], (iy[assigned], ix[assigned]), 1)
        if np.any(~assigned):
            np.add.at(density[1], (iy[~assigned], ix[~assigned]), 1)

        total += int(ix.shape[0])
        assigned_total += int(np.count_nonzero(assigned))
        unassigned_total += int(np.count_nonzero(~assigned))

    meta = {
        "bounds": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "requested_bin_um": float(bin_um),
        "actual_bin_um": float(actual_bin_um),
        "shape": [int(ny), int(nx)],
        "total": int(total),
        "assigned": int(assigned_total),
        "unassigned": int(unassigned_total),
        "max_count": int(density.max(initial=0)),
    }
    return density, meta


@dataclass
class TranscriptSpatialIndex:
    """Compact tiled point index for viewport transcript detail overlays."""

    x: np.ndarray
    y: np.ndarray
    assigned: np.ndarray
    tile_ids: np.ndarray
    tile_starts: np.ndarray
    tile_counts: np.ndarray
    min_x: float
    min_y: float
    tile_um: float
    nx_tiles: int
    total_rows: int
    indexed_rows: int
    sampled: bool


def build_transcript_spatial_index(
    points_obj,
    x_col: str,
    y_col: str,
    assignment_col: str | None = None,
    max_points: int = 25_000_000,
    tile_um: float = 250.0,
    random_state: int = 42,
) -> TranscriptSpatialIndex | None:
    """Build a sampled tiled point index for fast small-viewport queries."""
    max_points = int(max_points)
    if max_points <= 0:
        return None

    cols = _point_columns(x_col, y_col, assignment_col)
    work = points_obj[cols]
    if hasattr(work, "npartitions") and hasattr(work, "compute"):
        source_total = int(work.map_partitions(len, meta=("n", "i8")).sum().compute())
    else:
        source_total = int(len(work))

    pdf = load_points_dataframe(
        points_obj,
        x_col=x_col,
        y_col=y_col,
        assignment_col=assignment_col,
        max_points=max_points,
        random_state=random_state,
    )
    total_rows = int(len(pdf))
    if total_rows == 0:
        return None

    x_vals = pdf[x_col].to_numpy(dtype=np.float32, copy=False)
    y_vals = pdf[y_col].to_numpy(dtype=np.float32, copy=False)
    good = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(good):
        return None

    x_vals = x_vals[good]
    y_vals = y_vals[good]
    if assignment_col is not None and assignment_col in pdf.columns:
        assigned = assignment_mask(pdf.loc[good, assignment_col]).to_numpy(dtype=bool, copy=False)
    else:
        assigned = np.ones(x_vals.shape[0], dtype=bool)

    min_x = float(np.min(x_vals))
    min_y = float(np.min(y_vals))
    tile_um = max(float(tile_um), np.finfo(float).eps)
    tile_x = np.floor((x_vals.astype(np.float64) - min_x) / tile_um).astype(np.int64)
    tile_y = np.floor((y_vals.astype(np.float64) - min_y) / tile_um).astype(np.int64)
    nx_tiles = int(tile_x.max(initial=0)) + 1
    tile_ids = tile_y * np.int64(nx_tiles) + tile_x

    order = np.argsort(tile_ids, kind="stable")
    tile_ids = tile_ids[order]
    x_vals = np.ascontiguousarray(x_vals[order], dtype=np.float32)
    y_vals = np.ascontiguousarray(y_vals[order], dtype=np.float32)
    assigned = np.ascontiguousarray(assigned[order], dtype=bool)

    unique_ids, starts, counts = np.unique(tile_ids, return_index=True, return_counts=True)
    return TranscriptSpatialIndex(
        x=x_vals,
        y=y_vals,
        assigned=assigned,
        tile_ids=unique_ids.astype(np.int64, copy=False),
        tile_starts=starts.astype(np.int64, copy=False),
        tile_counts=counts.astype(np.int64, copy=False),
        min_x=min_x,
        min_y=min_y,
        tile_um=tile_um,
        nx_tiles=nx_tiles,
        total_rows=source_total,
        indexed_rows=int(x_vals.shape[0]),
        sampled=source_total > int(x_vals.shape[0]),
    )


def query_transcript_spatial_index(
    index: TranscriptSpatialIndex,
    bounds: tuple[float, float, float, float],
    max_points: int,
    sample_percent: float | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    """Query the transcript index and return capped assigned/unassigned napari coords."""
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    tile_x0 = int(np.floor((min_x - index.min_x) / index.tile_um))
    tile_x1 = int(np.floor((max_x - index.min_x) / index.tile_um))
    tile_y0 = int(np.floor((min_y - index.min_y) / index.tile_um))
    tile_y1 = int(np.floor((max_y - index.min_y) / index.tile_um))

    chunks: list[tuple[int, int]] = []
    for ty in range(tile_y0, tile_y1 + 1):
        for tx in range(tile_x0, tile_x1 + 1):
            tile_id = np.int64(ty) * np.int64(index.nx_tiles) + np.int64(tx)
            pos = int(np.searchsorted(index.tile_ids, tile_id))
            if pos < len(index.tile_ids) and index.tile_ids[pos] == tile_id:
                start = int(index.tile_starts[pos])
                chunks.append((start, start + int(index.tile_counts[pos])))

    empty = np.empty((0, 2), dtype=np.float32)
    if len(chunks) == 0:
        return {
            "assigned_coords": empty,
            "unassigned_coords": empty,
            "total_in_view": 0,
            "loaded": 0,
            "loaded_fraction": 0.0,
        }

    x_vals = np.concatenate([index.x[start:end] for start, end in chunks])
    y_vals = np.concatenate([index.y[start:end] for start, end in chunks])
    assigned = np.concatenate([index.assigned[start:end] for start, end in chunks])
    in_view = (x_vals >= min_x) & (x_vals <= max_x) & (y_vals >= min_y) & (y_vals <= max_y)
    if not np.any(in_view):
        return {
            "assigned_coords": empty,
            "unassigned_coords": empty,
            "total_in_view": 0,
            "loaded": 0,
            "loaded_fraction": 0.0,
        }

    x_vals = x_vals[in_view]
    y_vals = y_vals[in_view]
    assigned = assigned[in_view]
    total = int(x_vals.shape[0])

    max_points = max(0, int(max_points))
    if max_points == 0:
        n = 0
    elif sample_percent is None:
        n = min(total, max_points)
    else:
        n = int(np.ceil(total * (min(100.0, max(0.0, float(sample_percent))) / 100.0)))
        n = min(total, max_points, max(1, n))

    if n < total:
        rng = np.random.default_rng(int(random_state))
        keep = rng.choice(total, size=n, replace=False)
        x_vals = x_vals[keep]
        y_vals = y_vals[keep]
        assigned = assigned[keep]

    assigned_coords = np.column_stack([y_vals[assigned], x_vals[assigned]]).astype(np.float32, copy=False)
    unassigned_coords = np.column_stack([y_vals[~assigned], x_vals[~assigned]]).astype(np.float32, copy=False)
    loaded = int(assigned_coords.shape[0] + unassigned_coords.shape[0])
    return {
        "assigned_coords": assigned_coords,
        "unassigned_coords": unassigned_coords,
        "total_in_view": total,
        "loaded": loaded,
        "loaded_fraction": 0.0 if total == 0 else float(loaded) / float(total),
    }


def geometry_to_napari_polygons(
    geometries,
    max_shapes: int | None = None,
    simplify_tolerance: float | None = None,
    max_vertices_per_polygon: int | None = None,
) -> list[np.ndarray]:
    """Convert shapely Polygon/MultiPolygon geometries to napari polygon arrays."""
    out: list[np.ndarray] = []
    n_added = 0

    for geom in geometries:
        if max_shapes is not None and n_added >= max_shapes:
            break
        if geom is None or geom.is_empty:
            continue

        if simplify_tolerance is not None and simplify_tolerance > 0:
            geom = geom.simplify(float(simplify_tolerance), preserve_topology=True)
            if geom is None or geom.is_empty:
                continue

        if geom.geom_type == "Polygon":
            arr = _limit_closed_ring_vertices(
                np.asarray(geom.exterior.coords, dtype=np.float32),
                max_vertices_per_polygon,
            )
            if arr.shape[0] >= 3:
                out.append(arr[:, [1, 0]])  # y, x order for napari
                n_added += 1
            continue

        if geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                if max_shapes is not None and n_added >= max_shapes:
                    break
                arr = _limit_closed_ring_vertices(
                    np.asarray(part.exterior.coords, dtype=np.float32),
                    max_vertices_per_polygon,
                )
                if arr.shape[0] >= 3:
                    out.append(arr[:, [1, 0]])  # y, x order
                    n_added += 1

    return out


def _limit_closed_ring_vertices(arr: np.ndarray, max_vertices: int | None) -> np.ndarray:
    """Return a stride-decimated closed ring with at most max_vertices rows."""
    if max_vertices is None or max_vertices <= 0 or arr.shape[0] <= max_vertices:
        return arr

    if max_vertices < 4:
        max_vertices = 4

    is_closed = bool(arr.shape[0] > 1 and np.allclose(arr[0], arr[-1]))
    body = arr[:-1] if is_closed else arr
    target = max_vertices - 1 if is_closed else max_vertices
    if body.shape[0] <= target:
        return arr

    idx = np.linspace(0, body.shape[0] - 1, num=target, dtype=np.int64)
    idx = np.unique(idx)
    decimated = body[idx]
    if is_closed:
        decimated = np.vstack([decimated, decimated[0]])
    return decimated.astype(np.float32, copy=False)


def geometry_to_napari_bounding_boxes(
    geometries,
    max_shapes: int | None = None,
) -> np.ndarray:
    """Convert geometries to napari rectangle coordinates in one compact array."""
    if max_shapes is not None:
        geometries = geometries.iloc[:max_shapes] if hasattr(geometries, "iloc") else list(geometries)[:max_shapes]

    if hasattr(geometries, "bounds"):
        bounds = geometries.bounds
        minx = bounds["minx"].to_numpy(dtype=np.float32, copy=False)
        miny = bounds["miny"].to_numpy(dtype=np.float32, copy=False)
        maxx = bounds["maxx"].to_numpy(dtype=np.float32, copy=False)
        maxy = bounds["maxy"].to_numpy(dtype=np.float32, copy=False)
        good = np.isfinite(minx) & np.isfinite(miny) & np.isfinite(maxx) & np.isfinite(maxy)
        minx, miny, maxx, maxy = minx[good], miny[good], maxx[good], maxy[good]
    else:
        rows = []
        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            rows.append(geom.bounds)
        if len(rows) == 0:
            return np.empty((0, 4, 2), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        minx, miny, maxx, maxy = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    rectangles = np.empty((len(minx), 4, 2), dtype=np.float32)
    rectangles[:, 0, 0] = miny
    rectangles[:, 0, 1] = minx
    rectangles[:, 1, 0] = miny
    rectangles[:, 1, 1] = maxx
    rectangles[:, 2, 0] = maxy
    rectangles[:, 2, 1] = maxx
    rectangles[:, 3, 0] = maxy
    rectangles[:, 3, 1] = minx
    return rectangles


def geometry_to_napari_centroids(
    geometries,
    max_shapes: int | None = None,
) -> np.ndarray:
    """Convert geometries to representative points in napari y/x order."""
    if max_shapes is not None:
        geometries = geometries.iloc[:max_shapes] if hasattr(geometries, "iloc") else list(geometries)[:max_shapes]

    if hasattr(geometries, "representative_point"):
        points = geometries.representative_point()
        x = points.x.to_numpy(dtype=np.float32, copy=False)
        y = points.y.to_numpy(dtype=np.float32, copy=False)
        good = np.isfinite(x) & np.isfinite(y)
        return np.column_stack([y[good], x[good]]).astype(np.float32, copy=False)

    rows = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        point = geom.representative_point()
        rows.append((float(point.y), float(point.x)))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def affine_matrix_from_px_to_um(
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    x_coords=None,
    y_coords=None,
) -> np.ndarray:
    """Alias for building a row/col -> global y/x affine matrix."""
    return build_napari_affine_from_px_to_um(
        x_transform=x_transform,
        y_transform=y_transform,
        x_coords=x_coords,
        y_coords=y_coords,
    )


def pixel_window_global_bounds(
    affine: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[float, float, float, float]:
    """Return global x/y bounds covered by a row/col pixel window."""
    corners = np.array(
        [
            [float(y0), float(x0), 1.0],
            [float(y0), float(x1), 1.0],
            [float(y1), float(x0), 1.0],
            [float(y1), float(x1), 1.0],
        ],
        dtype=float,
    )
    yx = corners @ np.asarray(affine, dtype=float).T
    y_vals = yx[:, 0]
    x_vals = yx[:, 1]
    return (
        float(np.nanmin(x_vals)),
        float(np.nanmin(y_vals)),
        float(np.nanmax(x_vals)),
        float(np.nanmax(y_vals)),
    )


def query_geometries_for_bounds(gdf, bounds: tuple[float, float, float, float]):
    """Return geometries whose bounds intersect a global x/y bounding box."""
    query_box = box(*bounds)
    try:
        idx = gdf.sindex.query(query_box, predicate="intersects")
        return gdf.iloc[np.asarray(idx, dtype=np.int64)]
    except Exception:
        intersects = gdf.geometry.intersects(query_box)
        return gdf.loc[intersects]


def rasterize_geometries_chunk(
    geometries,
    labels,
    shape: tuple[int, int],
    inv_affine: np.ndarray,
    y0: int = 0,
    x0: int = 0,
    dtype=np.uint32,
) -> np.ndarray:
    """Rasterize global x/y geometries into one local label chunk."""
    if draw_polygon is None:
        raise RuntimeError("scikit-image is required to rasterize polygon labels.")

    out = np.zeros(tuple(int(v) for v in shape), dtype=dtype)
    inv = np.asarray(inv_affine, dtype=float)

    def draw_ring(coords, value):
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 3:
            return
        xy1 = np.column_stack([coords[:, 1], coords[:, 0], np.ones(coords.shape[0])])
        rc = xy1 @ inv.T
        rows = rc[:, 0] - float(y0)
        cols = rc[:, 1] - float(x0)
        rr, cc = draw_polygon(rows, cols, shape=out.shape)
        if rr.size:
            out[rr, cc] = value

    for geom, label in zip(geometries, labels, strict=False):
        if geom is None or geom.is_empty:
            continue
        value = np.asarray(label, dtype=dtype).item()
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else (geom,)
        for part in parts:
            if part.is_empty:
                continue
            draw_ring(part.exterior.coords, value)
            for interior in part.interiors:
                draw_ring(interior.coords, 0)

    return out


def label_outline_mask_chunk(labels, width: int = 1) -> np.ndarray:
    """Return a uint8 outline mask for a 2D label tile."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label tile, got shape {arr.shape}")

    fg = arr != 0
    outline = np.zeros(arr.shape, dtype=bool)
    outline[1:, :] |= fg[1:, :] & (arr[1:, :] != arr[:-1, :])
    outline[:-1, :] |= fg[:-1, :] & (arr[:-1, :] != arr[1:, :])
    outline[:, 1:] |= fg[:, 1:] & (arr[:, 1:] != arr[:, :-1])
    outline[:, :-1] |= fg[:, :-1] & (arr[:, :-1] != arr[:, 1:])

    width = int(width)
    if width > 1 and np.any(outline):
        try:
            from scipy.ndimage import binary_dilation

            outline = binary_dilation(outline, iterations=width - 1)
        except Exception:
            for _ in range(width - 1):
                expanded = outline.copy()
                expanded[1:, :] |= outline[:-1, :]
                expanded[:-1, :] |= outline[1:, :]
                expanded[:, 1:] |= outline[:, :-1]
                expanded[:, :-1] |= outline[:, 1:]
                outline = expanded

    return outline.astype(np.uint8, copy=False)
