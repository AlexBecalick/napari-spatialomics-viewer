#!/usr/bin/env python3
"""Utility helpers for the Xenium/MERSCOPE Napari comparison viewer."""

from __future__ import annotations

import colorsys
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import zarr
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

from .platforms import (
    ATERA_ROOT_ATTR,
    GENE_SELECTION_SLOT_CAP,
    PIXEL_TRANSFORM_MERSCOPE_AFFINE,
    PIXEL_TRANSFORM_SPEC_FILE,
    find_platform_spec,
    find_spec_path,
    platform_spec,
)

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
#: zarr attribute marking a viewer label cache (or a MerXen vendor-label alias).
LABEL_CACHE_ATTR = "napari_compare_label_cache"
#: ``source`` of a MerXen-managed alias hardlinked to vendor label payloads.
VENDOR_LABELS_SOURCE = "vendor_labels"
CELLPOSE_VALUE_BINS = 256
CELLPOSE_VALUE_CLIP_PERCENTILES = (1.0, 99.0)
TRANSPARENT_RGBA = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CorticalDepthRoleSpec:
    """One supported cortical-depth annotation role."""

    key: str
    geojson_role: str
    geometry_kind: str
    layer_label: str


CORTICAL_DEPTH_ROLE_SPECS: dict[str, CorticalDepthRoleSpec] = {
    "pia": CorticalDepthRoleSpec(
        key="pia",
        geojson_role="pial_boundary",
        geometry_kind="line",
        layer_label="Pial boundary",
    ),
    "wm": CorticalDepthRoleSpec(
        key="wm",
        geojson_role="gray_white_boundary",
        geometry_kind="line",
        layer_label="Gray/white boundary",
    ),
    "side": CorticalDepthRoleSpec(
        key="side",
        geojson_role="side_boundary",
        geometry_kind="line",
        layer_label="Tissue edge boundary",
    ),
    "exclusion": CorticalDepthRoleSpec(
        key="exclusion",
        geojson_role="exclusion",
        geometry_kind="polygon",
        layer_label="Exclusion polygons",
    ),
    "ribbon": CorticalDepthRoleSpec(
        key="ribbon",
        geojson_role="cortical_ribbon",
        geometry_kind="polygon",
        layer_label="Cortical ribbon",
    ),
}
CORTICAL_DEPTH_ROLE_ORDER = ("pia", "wm", "side", "exclusion", "ribbon")
CORTICAL_DEPTH_PIECE_ID_PROPERTY = "tissue_piece_id"
CORTICAL_DEPTH_PIECE_MODE_PROPERTY = "piece_mode"
CORTICAL_DEPTH_DEFAULT_PIECE_ID = "piece_1"
CORTICAL_DEPTH_DEPTH_MODE = "depth"
CORTICAL_DEPTH_MASK_QC_ONLY_MODE = "mask_qc_only"
CORTICAL_DEPTH_SEPARATE_FILE_STEMS = {
    "pia": "pial_boundary",
    "wm": "wm_boundary",
    "side": "side_boundaries",
    "exclusion": "exclusion_masks",
    "ribbon": "cortical_ribbon",
}


@dataclass(frozen=True)
class CorticalDepthAnnotationExport:
    """Combined GeoJSON payload plus validation messages."""

    geojson: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass(frozen=True)
class CorticalDepthShapeInput:
    """One napari shape plus its optional cortical-depth tissue piece id."""

    data: np.ndarray
    tissue_piece_id: str | None = None


OBJECT_ANNOTATION_SCHEMA = "merxen.distance_from_object/v1"
OBJECT_ANNOTATION_ROLE = "analysis_object"
OBJECT_ID_PROPERTY = "object_id"
OBJECT_TYPE_PROPERTY = "object_type"


@dataclass(frozen=True)
class ObjectAnnotationExport:
    """Generic polygon-object GeoJSON payload plus validation messages."""

    geojson: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether the object annotations passed validation."""
        return len(self.errors) == 0


@dataclass(frozen=True)
class ObjectAnnotationShapeInput:
    """One napari polygon and its optional persistent object identifier."""

    data: np.ndarray
    object_id: str | None = None


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


def derived_image_pyramid_cache_key(image_key: str, downsample: int) -> str:
    """Build the private images cache key for a materialized coarse-level pyramid."""
    return f"{DERIVED_CACHE_PREFIX}imgpyr__{_safe_cache_token(image_key)}__ds{int(downsample)}"


def derived_label_pyramid_cache_key(label_key: str, downsample: int) -> str:
    """Build the private labels cache key for a materialized coarse-level label pyramid."""
    return f"{DERIVED_CACHE_PREFIX}labelpyr__{_safe_cache_token(label_key)}__ds{int(downsample)}"


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


def _cellpose_quantification_table(root, table_key: str | None = None):
    return root[f"tables/{table_key or CELLPOSE_QUANTIFICATION_TABLE_KEY}"]


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


_QUANTIFICATION_REQUIRED_PARTS = (
    "obs/label_id",
    "var/image_key",
    "var/channel",
    "var/statistic",
    "X",
)


def cellpose_quantification_table_available(zarr_path: str | Path) -> bool:
    """Return whether the hard-coded Cellpose value-overlay inputs are present."""
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        table = _cellpose_quantification_table(root)
        has_required_table_parts = all(
            _zarr_path_exists(table, key) for key in _QUANTIFICATION_REQUIRED_PARTS
        )
        has_region = _zarr_path_exists(root, f"labels/{CELLPOSE_LABEL_KEY}") or _zarr_path_exists(
            root,
            f"shapes/{CELLPOSE_SHAPE_KEY}",
        )
        return bool(has_required_table_parts and has_region)
    except Exception:
        return False


@dataclass(frozen=True)
class MaskQuantification:
    """A per-cell image-statistics table and the mask it was measured on.

    ``label_key`` is the label element whose pixel values are the table's
    ``obs/label_id`` values. For a MerXen vendor-label store it is the managed
    alias (``atera_cell_boundaries_labels``), never the vendor element named by
    the table's ``mask_label_key``.
    """

    table_key: str
    shape_key: str
    label_key: str

    @property
    def is_cellpose(self) -> bool:
        return self.table_key == CELLPOSE_QUANTIFICATION_TABLE_KEY

    @property
    def display_name(self) -> str:
        """User-facing name: ``Cellpose`` or ``Per-cell image statistics (<mask>)``."""
        if self.is_cellpose:
            return "Cellpose"
        return f"Per-cell image statistics ({self.shape_key})"


CELLPOSE_MASK_QUANTIFICATION = MaskQuantification(
    table_key=CELLPOSE_QUANTIFICATION_TABLE_KEY,
    shape_key=CELLPOSE_SHAPE_KEY,
    label_key=CELLPOSE_LABEL_KEY,
)


def _read_group_attributes(path: Path) -> dict:
    """Attributes of a zarr v3 (``zarr.json``) or v2 (``.zattrs``) group, or ``{}``."""
    for name, nested in (("zarr.json", True), (".zattrs", False)):
        payload = _read_json_mapping(Path(path) / name)
        if payload is None:
            continue
        attrs = payload.get("attributes") if nested else payload
        if isinstance(attrs, dict):
            return attrs
    return {}


def vendor_label_alias_markers(zarr_path: str | Path) -> dict[str, dict]:
    """``{label_key: marker}`` for every label carrying a vendor-labels alias marker.

    MerXen links vendor label rasters (for example Atera ``cell_labels``) into
    ``<shape>_labels`` aliases and marks them with the viewer's label-cache
    attribute plus ``source: vendor_labels``. Such aliases are hardlinked
    payloads: the viewer must never delete, rewrite or re-mark them. Reads JSON
    metadata only.
    """
    labels_dir = Path(zarr_path) / "labels"
    out: dict[str, dict] = {}
    try:
        children = sorted(p for p in labels_dir.iterdir() if p.is_dir())
    except OSError:
        return out
    for child in children:
        if is_derived_cache_key(child.name):
            continue
        marker = _read_group_attributes(child).get(LABEL_CACHE_ATTR)
        if isinstance(marker, dict) and marker.get("source") == VENDOR_LABELS_SOURCE:
            out[child.name] = dict(marker)
    return out


def is_vendor_label_marker(marker: Any) -> bool:
    """Whether a label-cache marker describes a MerXen-managed vendor-label alias."""
    return isinstance(marker, Mapping) and marker.get("source") == VENDOR_LABELS_SOURCE


def hidden_vendor_label_keys(markers: Mapping[str, Mapping]) -> set[str]:
    """Vendor label elements already shown through a complete alias (hide them)."""
    return {
        str(marker["source_label_key"])
        for key, marker in markers.items()
        if marker.get("complete") and marker.get("source_label_key")
        and str(marker.get("source_label_key")) != str(key)
    }


def _quantification_label_key(zarr_path: Path, shape_key: str) -> str:
    """The label element whose pixels carry ``shape_key``'s instance ids.

    A complete vendor-label alias marked for ``shape_key`` wins; otherwise the
    viewer's own ``<shape>_labels`` name (rasterized on demand).
    """
    for key, marker in vendor_label_alias_markers(zarr_path).items():
        if marker.get("complete") and marker.get("source_shape_key") == str(shape_key):
            return key
    return f"{shape_key}_labels"


def discover_mask_quantification(zarr_path: str | Path | None) -> MaskQuantification | None:
    """Find the per-cell image-statistics table of a store.

    The MOSAIK Cellpose table is tried first (identical to
    :func:`cellpose_quantification_table_available`). Otherwise the first table,
    in key order, whose ``uns['mask_image_quantification']`` names a shape that
    is present in the store and which has the label-id, var metadata and ``X``
    parts. Only metadata is read.
    """
    if zarr_path is None:
        return None
    if cellpose_quantification_table_available(zarr_path):
        return CELLPOSE_MASK_QUANTIFICATION
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        if "tables" not in root:
            return None
        tables = root["tables"]
        shapes = set(root["shapes"].group_keys()) if "shapes" in root else set()
        from anndata.io import read_elem

        for table_key in sorted(tables.group_keys()):
            table = tables[table_key]
            try:
                if "uns" not in table or "mask_image_quantification" not in table["uns"]:
                    continue
                meta = read_elem(table["uns"]["mask_image_quantification"])
            except Exception:
                continue
            shape_key = meta.get("shape_key") if isinstance(meta, Mapping) else None
            if not isinstance(shape_key, str) or shape_key not in shapes:
                continue
            if not all(_zarr_path_exists(table, part) for part in _QUANTIFICATION_REQUIRED_PARTS):
                continue
            return MaskQuantification(
                table_key=str(table_key),
                shape_key=shape_key,
                label_key=_quantification_label_key(Path(zarr_path), shape_key),
            )
    except Exception:
        log.debug("Cannot discover a mask quantification table in %s", zarr_path, exc_info=True)
    return None


def cellpose_quantification_features(
    zarr_path: str | Path, table_key: str | None = None
) -> list[CellposeQuantificationFeature]:
    """List the image-statistic quantification columns (Cellpose by default)."""
    root = open_zarr_group_unconsolidated(zarr_path)
    return _cellpose_quantification_features_from_table(
        _cellpose_quantification_table(root, table_key)
    )


def cellpose_quantification_channel_labels(
    features: list[CellposeQuantificationFeature],
) -> list[str]:
    """Qualify channels shared by multiple images so selections stay unambiguous."""
    sources: dict[str, set[str]] = {}
    for feature in features:
        sources.setdefault(feature.channel.casefold(), set()).add(feature.image_key)
    return [
        f"{feature.image_key}: {feature.channel}"
        if len(sources[feature.channel.casefold()]) > 1 else feature.channel
        for feature in features
    ]


def _resolve_cellpose_feature(
    features: list[CellposeQuantificationFeature], channel: str, statistic: str
) -> CellposeQuantificationFeature:
    labels = cellpose_quantification_channel_labels(features)
    matches = [
        feature for feature, label in zip(features, labels, strict=True)
        if (label.casefold() == str(channel).casefold()
            or feature.channel.casefold() == str(channel).casefold())
        and feature.statistic.casefold() == str(statistic).casefold()
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous quantification channel {channel!r}; select an image-qualified channel.")
    if not matches:
        raise KeyError(f"No Cellpose quantification feature for channel={channel!r}, statistic={statistic!r}.")
    return matches[0]


def resolve_cellpose_quantification_feature(
    zarr_path: str | Path,
    channel: str,
    statistic: str,
    table_key: str | None = None,
) -> CellposeQuantificationFeature:
    """Resolve a channel (image-qualified when shared) and statistic."""
    return _resolve_cellpose_feature(
        cellpose_quantification_features(zarr_path, table_key), channel, statistic
    )


def load_cellpose_quantification_values(
    zarr_path: str | Path,
    channel: str,
    statistic: str,
    table_key: str | None = None,
) -> CellposeQuantificationValues:
    """Load label ids and one selected quantification column (Cellpose by default)."""
    root = open_zarr_group_unconsolidated(zarr_path)
    table = _cellpose_quantification_table(root, table_key)
    feature = _resolve_cellpose_feature(
        _cellpose_quantification_features_from_table(table), channel, statistic
    )

    label_ids = np.asarray(table["obs/label_id"][:], dtype=np.uint32)
    values = np.asarray(table["X"][:, feature.column_index], dtype=np.float32)
    if label_ids.shape[0] != values.shape[0]:
        raise ValueError(
            "Cellpose quantification label/value length mismatch: "
            f"{label_ids.shape[0]} labels vs {values.shape[0]} values."
        )
    return CellposeQuantificationValues(feature=feature, label_ids=label_ids, values=values)


def cellpose_cell_mean_intensities(
    zarr_path, label_id, table_key: str | None = None
) -> list[tuple[str, float | None]]:
    """Read only one quantification row, retaining image-qualified channel names."""
    root = open_zarr_group_unconsolidated(zarr_path)
    table = _cellpose_quantification_table(root, table_key)
    features = _cellpose_quantification_features_from_table(table)
    labels = cellpose_quantification_channel_labels(features)
    matches = np.flatnonzero(np.asarray(table["obs/label_id"][:]).astype(str) == str(label_id))
    row = np.asarray(table["X"][int(matches[0]), :]) if matches.size else None
    return [(label, float(row[feature.column_index])
             if row is not None and np.isfinite(row[feature.column_index]) else None)
            for label, feature in zip(labels, features) if feature.statistic == "mean"]


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


#: User-facing display names for segmentation + cell-type overlay layers. These
#: read as "Segmentation | Proseg" / "Cell types | Cellpose" in the napari layer
#: list and, like "Image"/"Genes", omit the dataset (only one dataset is active
#: at a time). All name building goes through the helpers below, so the friendly
#: form stays consistent for find/remove logic.
_FRIENDLY_LAYER_TYPE_LABELS = {"labels": "Segmentation", "cell_types": "Cell types"}


#: Display names for platform segmentation keys the generic shortening reads badly.
_SEGMENTATION_DISPLAY_NAMES = {"atera_nucleus": "Nucleus"}


def _pretty_segmentation_key(key: str) -> str:
    """Shorten a segmentation element key for display (MOSAIK_proseg -> Proseg)."""
    text = str(key)
    if text.endswith("_labels"):
        text = text[: -len("_labels")]
    if text in _SEGMENTATION_DISPLAY_NAMES:
        return _SEGMENTATION_DISPLAY_NAMES[text]
    if text.startswith("MOSAIK_"):
        text = text[len("MOSAIK_"):]
    text = text.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def make_layer_name(dataset: str, layer_type: str, key: str, channel: str | None = None) -> str:
    """Build a standardized layer name for this viewer."""
    if str(layer_type) == "image":
        label = str(channel if channel is not None else key)
        spec = find_platform_spec(dataset)
        if spec is not None:
            label = spec.channel_alias(str(key), label) or label
        return " | ".join(["Image", label])
    friendly = _FRIENDLY_LAYER_TYPE_LABELS.get(str(layer_type))
    if friendly is not None:
        parts = [friendly, _pretty_segmentation_key(key)]
        if channel is not None:
            parts.append(str(channel))
        return " | ".join(parts)
    parts = [str(dataset).upper(), str(layer_type), str(key)]
    if channel is not None:
        parts.append(str(channel))
    return " | ".join(parts)


def layer_name_prefix(dataset: str, layer_type: str, key: str | None = None) -> str:
    """Build a prefix used to find/remove related layers."""
    if str(layer_type) == "image":
        parts = ["Image"]
        if key is not None:
            parts.append(str(key))
        return " | ".join(parts)
    if str(layer_type) == "genes":
        return "Genes"
    friendly = _FRIENDLY_LAYER_TYPE_LABELS.get(str(layer_type))
    if friendly is not None:
        parts = [friendly]
        if key is not None:
            parts.append(_pretty_segmentation_key(key))
        return " | ".join(parts)
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


def gene_marker_symbol_label(symbol: str) -> str:
    """Return a concise display label for a napari Points marker symbol."""
    text = str(symbol)
    return GENE_MARKER_SYMBOL_LABELS.get(text, text.replace("_", " ").capitalize())


def build_cortical_depth_annotation_geojson(
    layers_by_role: Mapping[str, Iterable[Any]],
    *,
    layer_names: Mapping[str, str] | None = None,
    dataset: str | None = None,
) -> CorticalDepthAnnotationExport:
    """Build a validated MerXen cortical-depth GeoJSON FeatureCollection.

    Napari shape coordinates are stored in display order ``(y, x)``. GeoJSON and
    MerXen expect ``[x, y]``. This function only swaps those two axes; it does
    not scale, normalize, rotate, or otherwise transform coordinate values.
    """
    layer_names = {} if layer_names is None else dict(layer_names)
    errors: list[str] = []
    warnings_out: list[str] = []

    line_inputs: dict[str, list[tuple[LineString, str | None]]] = {}
    polygon_inputs: dict[str, list[tuple[Polygon, str | None]]] = {}
    for role in CORTICAL_DEPTH_ROLE_ORDER:
        spec = CORTICAL_DEPTH_ROLE_SPECS[role]
        role_shapes = layers_by_role.get(role, ())
        raw_shapes = _coerce_annotation_shape_inputs([] if role_shapes is None else list(role_shapes))
        source_name = layer_names.get(role, spec.layer_label)
        if spec.geometry_kind == "line":
            line_inputs[role] = _napari_shape_inputs_to_lines(
                raw_shapes,
                label=source_name,
                allow_closed=role == "side",
                errors=errors,
            )
        else:
            polygon_inputs[role] = _napari_shape_inputs_to_polygons(
                raw_shapes,
                label=source_name,
                errors=errors,
            )

    side_lines = tuple(line for line, _piece_id in line_inputs.get("side", ()))
    if not side_lines:
        errors.append("Missing tissue-edge boundary line.")
    elif len(side_lines) > 1:
        errors.append("Draw exactly one tissue-edge boundary line; multiple edge lines are not supported.")
    edge_line = side_lines[0] if len(side_lines) == 1 else None

    pial_by_piece = _group_lines_by_piece(line_inputs.get("pia", ()))
    wm_by_piece = _group_lines_by_piece(line_inputs.get("wm", ()))
    exclusions_by_piece = _group_polygons_by_piece(polygon_inputs.get("exclusion", ()))
    ribbons_by_piece = _group_polygons_by_piece(polygon_inputs.get("ribbon", ()))
    piece_ids = sorted(set(pial_by_piece) | set(wm_by_piece) | set(exclusions_by_piece) | set(ribbons_by_piece))
    if not piece_ids:
        errors.append("Missing pial boundary line.")

    pieces: dict[str, dict[str, Any]] = {}
    overlap_polygons: dict[str, Polygon | MultiPolygon] = {}
    for piece_id in piece_ids:
        pial = _merge_single_required_line(
            pial_by_piece.get(piece_id, []),
            role_label=f"pial boundary for {piece_id}",
            errors=errors,
            warnings_out=warnings_out,
        )
        wm = _merge_single_required_line(
            wm_by_piece.get(piece_id, []),
            role_label=f"gray/white boundary for {piece_id}",
            errors=errors,
            warnings_out=warnings_out,
        )
        exclusions = tuple(exclusions_by_piece.get(piece_id, ()))
        ribbons = tuple(ribbons_by_piece.get(piece_id, ()))
        if pial is None:
            if wm is not None:
                errors.append(f"{piece_id} has a gray/white boundary but no pial boundary.")
            elif exclusions or ribbons:
                errors.append(f"{piece_id} has piece-specific polygons but no pial boundary.")
            continue

        piece_mode = CORTICAL_DEPTH_DEPTH_MODE if wm is not None else CORTICAL_DEPTH_MASK_QC_ONLY_MODE
        pieces[piece_id] = {
            "pial": pial,
            "wm": wm,
            "exclusions": exclusions,
            "ribbons": ribbons,
            "piece_mode": piece_mode,
        }
        polygon = _validate_piece_relationships(
            piece_id=piece_id,
            pial=pial,
            wm=wm,
            edge_line=edge_line,
            exclusions=exclusions,
            ribbons=ribbons,
            errors=errors,
            warnings_out=warnings_out,
        )
        if polygon is not None:
            overlap_polygons[piece_id] = polygon

    _warn_for_overlapping_piece_polygons(overlap_polygons, warnings_out=warnings_out)

    features: list[dict[str, Any]] = []
    if edge_line is not None:
        features.append(
            _feature_for_geometry(
                edge_line,
                role="side",
                layer_name=layer_names.get("side"),
                dataset=dataset,
            )
        )
    for piece_id in piece_ids:
        piece = pieces.get(piece_id)
        if piece is None:
            continue
        piece_mode = str(piece["piece_mode"])
        features.append(
            _feature_for_geometry(
                piece["pial"],
                role="pia",
                layer_name=layer_names.get("pia"),
                dataset=dataset,
                tissue_piece_id=piece_id,
                piece_mode=piece_mode,
            )
        )
        if piece["wm"] is not None:
            features.append(
                _feature_for_geometry(
                    piece["wm"],
                    role="wm",
                    layer_name=layer_names.get("wm"),
                    dataset=dataset,
                    tissue_piece_id=piece_id,
                    piece_mode=piece_mode,
                )
            )
        exclusions = tuple(piece["exclusions"])
        if exclusions:
            exclusion_geom = exclusions[0] if len(exclusions) == 1 else MultiPolygon(exclusions)
            features.append(
                _feature_for_geometry(
                    exclusion_geom,
                    role="exclusion",
                    layer_name=layer_names.get("exclusion"),
                    dataset=dataset,
                    tissue_piece_id=piece_id,
                    piece_mode=piece_mode,
                )
            )
        ribbons = tuple(piece["ribbons"])
        if ribbons:
            ribbon_geom = ribbons[0] if len(ribbons) == 1 else MultiPolygon(ribbons)
            features.append(
                _feature_for_geometry(
                    ribbon_geom,
                    role="ribbon",
                    layer_name=layer_names.get("ribbon"),
                    dataset=dataset,
                    tissue_piece_id=piece_id,
                    piece_mode=piece_mode,
                )
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    return CorticalDepthAnnotationExport(
        geojson=_jsonable(geojson),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings_out)),
    )


def write_cortical_depth_annotation_geojson(
    path: str | Path,
    export: CorticalDepthAnnotationExport,
    *,
    allow_invalid: bool = False,
    include_validation_report: bool = False,
) -> None:
    """Write a cortical-depth annotation export, failing on validation errors."""
    if not export.ok and not allow_invalid:
        joined = "\n".join(export.errors)
        raise ValueError(f"Cortical-depth annotations are invalid:\n{joined}")
    payload = dict(export.geojson)
    if include_validation_report:
        payload["napari_compare_validation"] = {
            "ok": bool(export.ok),
            "errors": list(export.errors),
            "warnings": list(export.warnings),
        }
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def split_cortical_depth_annotation_geojson(
    export: CorticalDepthAnnotationExport,
) -> dict[str, dict[str, Any]]:
    """Split a combined cortical-depth export into per-role FeatureCollections."""
    if not export.ok:
        joined = "\n".join(export.errors)
        raise ValueError(f"Cortical-depth annotations are invalid:\n{joined}")

    out: dict[str, dict[str, Any]] = {}
    for feature in export.geojson.get("features", []):
        properties = feature.get("properties", {})
        role = str(properties.get("annotation_role", ""))
        if role not in CORTICAL_DEPTH_SEPARATE_FILE_STEMS:
            continue
        out.setdefault(role, {"type": "FeatureCollection", "features": []})["features"].append(feature)
    return out


def write_cortical_depth_separate_geojsons(
    output_dir: str | Path,
    export: CorticalDepthAnnotationExport,
    *,
    stem: str = "cortical_depth",
) -> dict[str, Path]:
    """Write per-role cortical-depth GeoJSON files and return role->path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = split_cortical_depth_annotation_geojson(export)
    written: dict[str, Path] = {}
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem)).strip("_") or "cortical_depth"
    for role in CORTICAL_DEPTH_ROLE_ORDER:
        payload = payloads.get(role)
        if not payload:
            continue
        path = output_dir / f"{safe_stem}_{CORTICAL_DEPTH_SEPARATE_FILE_STEMS[role]}.geojson"
        path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
        written[role] = path
    return written


def build_object_annotation_geojson(
    objects_by_type: Mapping[str, Iterable[Any]],
    *,
    dataset: str | None = None,
) -> ObjectAnnotationExport:
    """Build a validated GeoJSON file for named polygon object sets.

    Args:
        objects_by_type: Mapping from a user-provided object type/name to napari
            polygon arrays. Arrays use napari ``(y, x)`` coordinate order.
        dataset: Optional dataset label stored on every feature.

    Returns:
        Validated object annotation payload. Each polygon is emitted as an
        independent feature with a persistent or generated ``object_id``.
    """
    errors: list[str] = []
    warnings_out: list[str] = []
    features: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_object_type, raw_shapes in objects_by_type.items():
        object_type = str(raw_object_type).strip()
        if not object_type:
            errors.append("Object annotation names must not be blank.")
            continue
        shapes = _coerce_object_annotation_shape_inputs(
            [] if raw_shapes is None else list(raw_shapes)
        )
        if not shapes:
            warnings_out.append(f"Object set {object_type!r} contains no polygons.")
            continue
        safe_type = _safe_object_identifier(object_type)
        for index, shape_input in enumerate(shapes, start=1):
            polygon = _napari_shape_to_polygon(
                shape_input.data,
                label=f"{object_type} polygon {index}",
                errors=errors,
            )
            if polygon is None:
                continue
            object_id = str(shape_input.object_id or "").strip()
            if not object_id:
                generated_index = index
                object_id = f"{safe_type}_{generated_index:04d}"
                while object_id in seen_ids:
                    generated_index += 1
                    object_id = f"{safe_type}_{generated_index:04d}"
            if object_id in seen_ids:
                errors.append(f"Duplicate object_id {object_id!r}.")
                continue
            seen_ids.add(object_id)
            properties: dict[str, Any] = {
                "role": OBJECT_ANNOTATION_ROLE,
                "annotation_role": "object",
                OBJECT_ID_PROPERTY: object_id,
                OBJECT_TYPE_PROPERTY: object_type,
                "name": object_type,
            }
            if dataset:
                properties["dataset"] = str(dataset)
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": mapping(polygon),
                }
            )

    if not features:
        errors.append("Draw at least one object polygon before exporting.")
    payload = {
        "type": "FeatureCollection",
        "object_annotation_schema": OBJECT_ANNOTATION_SCHEMA,
        "features": features,
    }
    return ObjectAnnotationExport(
        geojson=_jsonable(payload),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings_out)),
    )


def write_object_annotation_geojson(
    path: str | Path,
    export: ObjectAnnotationExport,
) -> None:
    """Write validated polygon object annotations to GeoJSON."""
    if not export.ok:
        joined = "\n".join(export.errors)
        raise ValueError(f"Object annotations are invalid:\n{joined}")
    Path(path).write_text(json.dumps(_jsonable(export.geojson), indent=2) + "\n")


def read_object_annotation_geojson(
    path: str | Path,
) -> dict[str, list[ObjectAnnotationShapeInput]]:
    """Read object GeoJSON into named napari ``(y, x)`` polygon inputs.

    Args:
        path: Object annotation GeoJSON written by this viewer or another tool
            using the MerXen object properties.

    Returns:
        Mapping of object type/name to polygon shapes and persistent IDs.

    Raises:
        ValueError: If features are not polygonal, IDs are duplicated, or a
            polygon contains holes that napari Shapes cannot faithfully reload.
    """
    data = json.loads(Path(path).read_text())
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": data}]

    objects: dict[str, list[ObjectAnnotationShapeInput]] = {}
    seen_ids: set[str] = set()
    counters: dict[str, int] = {}
    for feature_index, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            raise ValueError(
                f"Object feature {feature_index} must be a Polygon or MultiPolygon."
            )
        object_type = str(
            properties.get(OBJECT_TYPE_PROPERTY)
            or properties.get("name")
            or "objects"
        ).strip()
        if not object_type:
            raise ValueError(f"Object feature {feature_index} has a blank object type.")
        base_id = str(properties.get(OBJECT_ID_PROPERTY) or "").strip()
        for polygon_index, polygon in enumerate(polygons, start=1):
            if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
                raise ValueError(f"Object feature {feature_index} has invalid geometry.")
            if polygon.interiors:
                raise ValueError(
                    f"Object feature {feature_index} contains polygon holes, which "
                    "cannot be reloaded faithfully into a napari Shapes layer."
                )
            counters[object_type] = counters.get(object_type, 0) + 1
            object_id = base_id
            if len(polygons) > 1 and object_id:
                object_id = f"{object_id}_{polygon_index}"
            if not object_id:
                object_id = (
                    f"{_safe_object_identifier(object_type)}_"
                    f"{counters[object_type]:04d}"
                )
            if object_id in seen_ids:
                raise ValueError(f"Duplicate object_id {object_id!r} in {path}.")
            seen_ids.add(object_id)
            xy = np.asarray(polygon.exterior.coords[:-1], dtype=float)
            objects.setdefault(object_type, []).append(
                ObjectAnnotationShapeInput(
                    data=xy[:, [1, 0]].astype(np.float32, copy=False),
                    object_id=object_id,
                )
            )
    if not objects:
        raise ValueError(f"No polygon object annotations found in {path}.")
    return objects


def _coerce_object_annotation_shape_inputs(
    shapes: Iterable[Any],
) -> list[ObjectAnnotationShapeInput]:
    out: list[ObjectAnnotationShapeInput] = []
    for shape_input in shapes:
        if isinstance(shape_input, ObjectAnnotationShapeInput):
            out.append(shape_input)
            continue
        if isinstance(shape_input, Mapping):
            data = shape_input.get("data")
            if data is None:
                data = shape_input.get("shape")
            out.append(
                ObjectAnnotationShapeInput(
                    data=np.asarray(data, dtype=float),
                    object_id=str(shape_input.get(OBJECT_ID_PROPERTY) or "").strip()
                    or None,
                )
            )
            continue
        if isinstance(shape_input, tuple) and len(shape_input) == 2:
            out.append(
                ObjectAnnotationShapeInput(
                    data=np.asarray(shape_input[0], dtype=float),
                    object_id=str(shape_input[1] or "").strip() or None,
                )
            )
            continue
        out.append(ObjectAnnotationShapeInput(data=np.asarray(shape_input, dtype=float)))
    return out


def _safe_object_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()
    return safe or "object"


def snap_cortical_depth_boundaries_to_edge(
    pial_shapes: Iterable[np.ndarray],
    wm_shapes: Iterable[np.ndarray] | None = None,
    edge_shapes: Iterable[np.ndarray] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Snap pia/WM path endpoints to the nearest point on the one tissue edge.

    Returned arrays remain in napari ``(y, x)`` order.
    """
    errors: list[str] = []
    pial_lines = _napari_shapes_to_lines(
        list(pial_shapes if pial_shapes is not None else ()),
        label="pial boundary",
        errors=errors,
    )
    wm_lines = _napari_shapes_to_lines(
        list(wm_shapes if wm_shapes is not None else ()),
        label="gray/white boundary",
        errors=errors,
    )
    edge_lines = _napari_shapes_to_lines(
        list(edge_shapes or ()),
        label="tissue edge boundary",
        allow_closed=True,
        errors=errors,
    )
    if not edge_lines:
        errors.append("One tissue-edge boundary line is required for snapping.")
    elif len(edge_lines) > 1:
        errors.append("Draw exactly one tissue-edge boundary line before snapping.")
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))

    edge = edge_lines[0]
    return {
        "pia": [_snap_line_endpoints_to_edge(line, edge) for line in pial_lines],
        "wm": [_snap_line_endpoints_to_edge(line, edge) for line in wm_lines],
    }


def snap_side_boundaries_to_pia_wm(
    pial_shapes: Iterable[np.ndarray],
    wm_shapes: Iterable[np.ndarray],
    side_shapes: Iterable[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Deprecated compatibility wrapper for the old two-side-edge snapping API."""
    snapped = snap_cortical_depth_boundaries_to_edge(pial_shapes, wm_shapes, side_shapes)
    return snapped["pia"] + snapped["wm"]


def _coerce_annotation_shape_inputs(shapes: Iterable[Any]) -> list[CorticalDepthShapeInput]:
    out: list[CorticalDepthShapeInput] = []
    for shape in shapes:
        if isinstance(shape, CorticalDepthShapeInput):
            out.append(shape)
            continue
        if isinstance(shape, Mapping):
            data = shape.get("data")
            if data is None:
                data = shape.get("shape")
            out.append(
                CorticalDepthShapeInput(
                    data=np.asarray(data, dtype=float),
                    tissue_piece_id=_clean_piece_id(shape.get(CORTICAL_DEPTH_PIECE_ID_PROPERTY)),
                )
            )
            continue
        if isinstance(shape, tuple) and len(shape) == 2:
            out.append(
                CorticalDepthShapeInput(
                    data=np.asarray(shape[0], dtype=float),
                    tissue_piece_id=_clean_piece_id(shape[1]),
                )
            )
            continue
        out.append(CorticalDepthShapeInput(data=np.asarray(shape, dtype=float)))
    return out


def _clean_piece_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _piece_id_for_shape(shape: CorticalDepthShapeInput) -> str:
    return _clean_piece_id(shape.tissue_piece_id) or CORTICAL_DEPTH_DEFAULT_PIECE_ID


def _napari_shape_inputs_to_lines(
    shapes: Iterable[CorticalDepthShapeInput],
    *,
    label: str,
    errors: list[str],
    allow_closed: bool = False,
) -> list[tuple[LineString, str | None]]:
    lines: list[tuple[LineString, str | None]] = []
    for idx, shape in enumerate(shapes, start=1):
        parsed = _napari_shape_to_line(
            shape.data,
            label=f"{label} shape {idx}",
            errors=errors,
            allow_closed=allow_closed,
        )
        if parsed is not None:
            lines.append((parsed, _piece_id_for_shape(shape)))
    return lines


def _napari_shape_inputs_to_polygons(
    shapes: Iterable[CorticalDepthShapeInput],
    *,
    label: str,
    errors: list[str],
) -> list[tuple[Polygon, str | None]]:
    polygons: list[tuple[Polygon, str | None]] = []
    for idx, shape in enumerate(shapes, start=1):
        parsed = _napari_shape_to_polygon(shape.data, label=f"{label} shape {idx}", errors=errors)
        if parsed is not None:
            polygons.append((parsed, _piece_id_for_shape(shape)))
    return polygons


def _napari_shapes_to_lines(
    shapes: Iterable[np.ndarray],
    *,
    label: str,
    errors: list[str],
    allow_closed: bool = False,
) -> list[LineString]:
    lines: list[LineString] = []
    for idx, shape_data in enumerate(shapes, start=1):
        line = _napari_shape_to_line(
            shape_data,
            label=f"{label} shape {idx}",
            errors=errors,
            allow_closed=allow_closed,
        )
        if line is not None:
            lines.append(line)
    return lines


def _napari_shape_to_line(
    shape_data,
    *,
    label: str,
    errors: list[str],
    allow_closed: bool = False,
) -> LineString | None:
    xy = _napari_yx_to_xy(shape_data, label=label, errors=errors)
    if xy is None:
        return None
    xy = _drop_consecutive_duplicate_points(xy)
    if xy.shape[0] < 2:
        errors.append(f"{label} must contain at least two distinct points.")
        return None
    if np.allclose(xy[0], xy[-1]) and not allow_closed:
        errors.append(f"{label} is closed; draw pia/WM boundaries as open polylines.")
        return None
    line = LineString(xy)
    if line.is_empty or line.length <= 0:
        errors.append(f"{label} has zero line length.")
        return None
    return line


def _napari_shapes_to_polygons(
    shapes: Iterable[np.ndarray],
    *,
    label: str,
    errors: list[str],
) -> list[Polygon]:
    polygons: list[Polygon] = []
    for idx, shape_data in enumerate(shapes, start=1):
        polygon = _napari_shape_to_polygon(shape_data, label=f"{label} shape {idx}", errors=errors)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def _napari_shape_to_polygon(shape_data, *, label: str, errors: list[str]) -> Polygon | None:
    xy = _napari_yx_to_xy(shape_data, label=label, errors=errors)
    if xy is None:
        return None
    xy = _drop_consecutive_duplicate_points(xy)
    if xy.shape[0] < 3:
        errors.append(f"{label} must contain at least three distinct points.")
        return None
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    polygon = Polygon(xy)
    if polygon.is_empty or polygon.area <= 0:
        errors.append(f"{label} has zero polygon area.")
        return None
    if not polygon.is_valid:
        errors.append(f"{label} is invalid or self-intersecting; redraw it as a simple polygon.")
        return None
    return polygon


def _napari_yx_to_xy(
    shape_data,
    *,
    label: str,
    errors: list[str],
) -> np.ndarray | None:
    arr = np.asarray(shape_data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        errors.append(f"{label} must be a 2D napari shape with y/x coordinates.")
        return None
    yx = arr[:, :2]
    if not np.isfinite(yx).all():
        errors.append(f"{label} contains non-finite coordinates.")
        return None
    return yx[:, [1, 0]].astype(float, copy=False)


def _drop_consecutive_duplicate_points(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy
    keep = np.ones(xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0
    return xy[keep]


def _merge_single_required_line(
    lines: list[LineString],
    *,
    role_label: str,
    errors: list[str],
    warnings_out: list[str],
) -> LineString | None:
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    merged = linemerge(MultiLineString(lines))
    if isinstance(merged, LineString):
        warnings_out.append(f"Merged {len(lines)} {role_label} path segments into one LineString.")
        return merged
    errors.append(f"Multiple {role_label} path segments could not be merged into one continuous line.")
    return None


def _group_lines_by_piece(items: Iterable[tuple[LineString, str | None]]) -> dict[str, list[LineString]]:
    grouped: dict[str, list[LineString]] = {}
    for line, piece_id in items:
        grouped.setdefault(piece_id or CORTICAL_DEPTH_DEFAULT_PIECE_ID, []).append(line)
    return grouped


def _group_polygons_by_piece(items: Iterable[tuple[Polygon, str | None]]) -> dict[str, list[Polygon]]:
    grouped: dict[str, list[Polygon]] = {}
    for polygon, piece_id in items:
        grouped.setdefault(piece_id or CORTICAL_DEPTH_DEFAULT_PIECE_ID, []).append(polygon)
    return grouped


def _validate_piece_relationships(
    *,
    piece_id: str,
    pial: LineString,
    wm: LineString | None,
    edge_line: LineString | None,
    exclusions: tuple[Polygon, ...],
    ribbons: tuple[Polygon, ...],
    errors: list[str],
    warnings_out: list[str],
) -> Polygon | MultiPolygon | None:
    if wm is not None:
        intersection = pial.intersection(wm)
        if not intersection.is_empty:
            warnings_out.append(
                f"{piece_id}: pial and gray/white boundary lines intersect; avoid crossing or touching these boundaries."
            )

    for idx, polygon in enumerate(exclusions, start=1):
        if polygon.intersects(pial) or (wm is not None and polygon.intersects(wm)):
            warnings_out.append(
                f"{piece_id}: exclusion polygon {idx} touches or overlaps the pial/gray-white boundary; "
                "this can remove Dirichlet boundary pixels and degrade the Laplace solve."
            )

    explicit_ribbon = _union_piece_polygons(ribbons)
    if explicit_ribbon is not None:
        return explicit_ribbon

    if edge_line is None:
        return None

    tolerance = max(1e-6, 0.02 * max(float(pial.length), float(wm.length) if wm is not None else 0.0))
    for label, line in (("pial", pial), ("gray/white", wm)):
        if line is None:
            continue
        coords = np.asarray(line.coords, dtype=float)[:, :2]
        for endpoint_label, xy in (("start", coords[0]), ("end", coords[-1])):
            distance = float(edge_line.distance(Point(float(xy[0]), float(xy[1]))))
            if distance > tolerance:
                warnings_out.append(
                    f"{piece_id}: {label} boundary {endpoint_label} point is not on the tissue edge "
                    f"(distance {distance:.3g}); use Snap To Edge or draw an explicit ribbon polygon."
                )

    candidates = _candidate_piece_polygons(edge_line=edge_line, pial=pial, wm=wm)
    if wm is None:
        if len(candidates) != 1:
            errors.append(
                f"{piece_id}: pial-only pieces need exactly one polygon from tissue edge + pia, "
                "or an explicit cortical-ribbon polygon."
            )
            return None
    elif len(candidates) == 0:
        errors.append(
            f"{piece_id}: tissue edge, pial boundary, and gray/white boundary do not form a polygon; "
            "snap endpoints to the edge or draw an explicit cortical-ribbon polygon."
        )
        return None
    elif len(candidates) > 1:
        warnings_out.append(
            f"{piece_id}: multiple candidate ribbon polygons were found; draw an explicit cortical-ribbon "
            "polygon if MerXen chooses the wrong tissue region."
        )
    return min(candidates, key=lambda geom: float(geom.area)) if candidates else None


def _candidate_piece_polygons(
    *,
    edge_line: LineString,
    pial: LineString,
    wm: LineString | None,
) -> list[Polygon]:
    candidates = _edge_subchain_candidate_piece_polygons(edge_line=edge_line, pial=pial, wm=wm)

    lines = [edge_line, pial]
    if wm is not None:
        lines.append(wm)
    merged = unary_union(lines)
    polygonized = polygonize(merged)
    polygon_geoms = getattr(polygonized, "geoms", polygonized)
    polygons = [poly for poly in polygon_geoms if isinstance(poly, Polygon) and poly.area > 0]
    for polygon in polygons:
        boundary = polygon.boundary
        if not boundary.intersects(pial):
            continue
        if wm is not None and not boundary.intersects(wm):
            continue
        candidates.append(polygon)
    return [
        polygon
        for polygon in _unique_valid_polygons(candidates)
        if _polygon_boundary_line_coverage(polygon, pial) >= 0.95
        and (wm is None or _polygon_boundary_line_coverage(polygon, wm) >= 0.95)
    ]


def _edge_subchain_candidate_piece_polygons(
    *,
    edge_line: LineString,
    pial: LineString,
    wm: LineString | None,
) -> list[Polygon]:
    pial_coords = _line_xy_array(pial)
    if pial_coords.shape[0] < 2:
        return []

    pial_distances = _line_endpoint_edge_distances(edge_line, pial)
    if wm is None:
        candidates: list[Polygon] = []
        for edge_path in _edge_paths_between(edge_line, pial_distances[1], pial_distances[0]):
            ring = _join_coordinate_parts([pial_coords, edge_path[1:]])
            candidates.extend(_valid_polygons_from_ring_coordinates(ring))
        return _unique_valid_polygons(candidates)

    wm_coords = _line_xy_array(wm)
    if wm_coords.shape[0] < 2:
        return []
    wm_distances = _line_endpoint_edge_distances(edge_line, wm)

    pairings = (
        # pial start -> WM start, pial end -> WM end.
        (wm_coords[::-1], wm_distances[1], wm_distances[0]),
        # pial start -> WM end, pial end -> WM start.
        (wm_coords, wm_distances[0], wm_distances[1]),
    )
    candidates: list[Polygon] = []
    for wm_path_coords, pial_end_wm_distance, wm_start_pial_distance in pairings:
        for edge_to_wm in _edge_paths_between(edge_line, pial_distances[1], pial_end_wm_distance):
            for edge_to_pia in _edge_paths_between(edge_line, wm_start_pial_distance, pial_distances[0]):
                ring = _join_coordinate_parts(
                    [
                        pial_coords,
                        edge_to_wm[1:],
                        wm_path_coords[1:],
                        edge_to_pia[1:],
                    ]
                )
                candidates.extend(_valid_polygons_from_ring_coordinates(ring))
    return _unique_valid_polygons(candidates)


def _line_xy_array(line: LineString) -> np.ndarray:
    return np.asarray(line.coords, dtype=float)[:, :2]


def _line_endpoint_edge_distances(edge_line: LineString, line: LineString) -> tuple[float, float]:
    coords = _line_xy_array(line)
    start = Point(float(coords[0, 0]), float(coords[0, 1]))
    end = Point(float(coords[-1, 0]), float(coords[-1, 1]))
    return float(edge_line.project(start)), float(edge_line.project(end))


def _edge_paths_between(edge_line: LineString, start_distance: float, end_distance: float) -> list[np.ndarray]:
    if _is_closed_line(edge_line):
        if np.isclose(start_distance, end_distance, rtol=0.0, atol=1e-9):
            return [_edge_path_forward(edge_line, start_distance, end_distance)]
        forward = _edge_path_forward(edge_line, start_distance, end_distance)
        backward = _edge_path_forward(edge_line, end_distance, start_distance)[::-1]
        return _unique_coordinate_paths([forward, backward])
    return [_edge_path_no_wrap(edge_line, start_distance, end_distance)]


def _edge_path_forward(edge_line: LineString, start_distance: float, end_distance: float) -> np.ndarray:
    length = float(edge_line.length)
    start = float(np.clip(start_distance, 0.0, length))
    end = float(np.clip(end_distance, 0.0, length))
    if start <= end:
        return _edge_path_no_wrap(edge_line, start, end)
    first = _edge_path_no_wrap(edge_line, start, length)
    second = _edge_path_no_wrap(edge_line, 0.0, end)
    return _drop_consecutive_duplicate_points(np.vstack([first, second[1:]]))


def _edge_path_no_wrap(edge_line: LineString, start_distance: float, end_distance: float) -> np.ndarray:
    length = float(edge_line.length)
    start = float(np.clip(start_distance, 0.0, length))
    end = float(np.clip(end_distance, 0.0, length))
    if start > end:
        return _edge_path_no_wrap(edge_line, end, start)[::-1]

    edge_coords = _line_xy_array(edge_line)
    points: list[np.ndarray] = [_point_on_line(edge_line, start)]
    cumulative = 0.0
    for idx in range(edge_coords.shape[0] - 1):
        segment_length = float(np.linalg.norm(edge_coords[idx + 1] - edge_coords[idx]))
        next_cumulative = cumulative + segment_length
        if segment_length > 0 and start < next_cumulative < end:
            points.append(edge_coords[idx + 1].astype(float, copy=True))
        cumulative = next_cumulative
    points.append(_point_on_line(edge_line, end))
    return _drop_consecutive_duplicate_points(np.vstack(points))


def _point_on_line(line: LineString, distance: float) -> np.ndarray:
    point = line.interpolate(float(distance))
    return np.asarray(point.coords[0][:2], dtype=float)


def _is_closed_line(line: LineString) -> bool:
    coords = _line_xy_array(line)
    return coords.shape[0] > 2 and bool(np.allclose(coords[0], coords[-1], rtol=0.0, atol=1e-9))


def _join_coordinate_parts(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(part, dtype=float)[:, :2] for part in parts if np.asarray(part).size]
    if not arrays:
        return np.empty((0, 2), dtype=float)
    return _drop_consecutive_duplicate_points(np.vstack(arrays))


def _valid_polygons_from_ring_coordinates(coords: np.ndarray) -> list[Polygon]:
    coords = _drop_consecutive_duplicate_points(np.asarray(coords, dtype=float)[:, :2])
    if coords.shape[0] < 3:
        return []
    if not np.allclose(coords[0], coords[-1], rtol=0.0, atol=1e-9):
        coords = np.vstack([coords, coords[0]])
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, Polygon):
        return [polygon] if not polygon.is_empty and polygon.area > 0 else []
    if isinstance(polygon, MultiPolygon):
        return [part for part in polygon.geoms if not part.is_empty and part.area > 0]
    if isinstance(polygon, GeometryCollection):
        return [part for part in polygon.geoms if isinstance(part, Polygon) and not part.is_empty and part.area > 0]
    return []


def _unique_valid_polygons(polygons: Iterable[Polygon]) -> list[Polygon]:
    out: list[Polygon] = []
    seen: set[tuple[float, float, float]] = set()
    for polygon in polygons:
        if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0:
            continue
        candidate = polygon if polygon.is_valid else polygon.buffer(0)
        for part in _geometry_polygon_parts(candidate):
            if any(_polygons_are_near_duplicates(part, existing) for existing in out):
                continue
            key = (round(float(part.area), 6), round(float(part.centroid.x), 6), round(float(part.centroid.y), 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(part)
    return out


def _polygons_are_near_duplicates(left: Polygon, right: Polygon) -> bool:
    area_scale = max(float(left.area), float(right.area), 1.0)
    return float(left.symmetric_difference(right).area) <= 1e-6 * area_scale


def _polygon_boundary_line_coverage(polygon: Polygon, line: LineString) -> float:
    tolerance = max(1e-6, 1e-9 * max(float(polygon.length), float(line.length)))
    missing = line.difference(polygon.boundary.buffer(tolerance))
    missing_length = 0.0 if missing.is_empty else float(getattr(missing, "length", 0.0))
    line_length = float(line.length)
    if line_length <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - missing_length / line_length))


def _geometry_polygon_parts(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty and geom.area > 0 else []
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty and part.area > 0]
    if isinstance(geom, GeometryCollection):
        return [part for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty and part.area > 0]
    return []


def _unique_coordinate_paths(paths: Iterable[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for path in paths:
        path = _drop_consecutive_duplicate_points(np.asarray(path, dtype=float)[:, :2])
        key = tuple((round(float(x), 6), round(float(y), 6)) for x, y in path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _union_piece_polygons(polygons: tuple[Polygon, ...]) -> Polygon | MultiPolygon | None:
    if not polygons:
        return None
    merged = unary_union(polygons)
    if isinstance(merged, Polygon | MultiPolygon):
        return merged
    if isinstance(merged, GeometryCollection):
        parts = [geom for geom in merged.geoms if isinstance(geom, Polygon) and geom.area > 0]
        if parts:
            unioned = unary_union(parts)
            if isinstance(unioned, Polygon | MultiPolygon):
                return unioned
    return None


def _warn_for_overlapping_piece_polygons(
    polygons: Mapping[str, Polygon | MultiPolygon],
    *,
    warnings_out: list[str],
) -> None:
    items = list(polygons.items())
    for idx, (left_id, left) in enumerate(items):
        for right_id, right in items[idx + 1 :]:
            try:
                overlap = left.intersection(right)
            except Exception:
                continue
            if not overlap.is_empty and float(overlap.area) > 0:
                warnings_out.append(
                    f"{left_id} and {right_id} candidate ribbon polygons overlap; check tissue_piece_id grouping."
                )


def _snap_line_endpoints_to_edge(line: LineString, edge: LineString) -> np.ndarray:
    coords = np.asarray(line.coords, dtype=float)[:, :2].copy()
    for idx in (0, coords.shape[0] - 1):
        point = Point(float(coords[idx, 0]), float(coords[idx, 1]))
        snapped = edge.interpolate(edge.project(point))
        coords[idx] = np.asarray(snapped.coords[0][:2], dtype=float)
    return coords[:, [1, 0]].astype(np.float32, copy=False)


def _feature_for_geometry(
    geom,
    *,
    role: str,
    layer_name: str | None,
    dataset: str | None,
    tissue_piece_id: str | None = None,
    piece_mode: str | None = None,
) -> dict[str, Any]:
    spec = CORTICAL_DEPTH_ROLE_SPECS[role]
    properties: dict[str, Any] = {
        "role": spec.geojson_role,
        "annotation_role": role,
    }
    if tissue_piece_id is not None:
        properties[CORTICAL_DEPTH_PIECE_ID_PROPERTY] = str(tissue_piece_id)
    if piece_mode is not None:
        properties[CORTICAL_DEPTH_PIECE_MODE_PROPERTY] = str(piece_mode)
    if layer_name:
        properties["name"] = str(layer_name)
    if dataset:
        properties["dataset"] = str(dataset)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(geom),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return float(value)
    return value


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


#: MerXen's name for the paired Xenium frame. MerXen's aligned rasters
#: (``*_aligned_nonrigid`` images and labels in a MERSCOPE store) are resampled
#: onto the Xenium pixel grid and map only to this frame, with the Xenium pixel
#: size; they carry no ``global`` (MERSCOPE mosaic) transformation.
MERXEN_ALIGNMENT_COORDINATE_SYSTEM = "merxen_xenium"


def _element_transformations(element) -> dict[str, Any]:
    from spatialdata.transformations import get_transformation

    try:
        transformations = get_transformation(element, get_all=True)
    except Exception:
        return {}
    return dict(transformations) if isinstance(transformations, Mapping) else {}


def raster_display_transformation(element) -> tuple[str, Any]:
    """Return the (coordinate system, transformation) a raster is placed through.

    Native rasters are placed through ``global``. A MerXen aligned raster has no
    ``global`` entry, only its own transformation to the alignment frame, so it
    is placed through that one.

    Raises:
        ValueError: If the element maps to neither frame.
    """
    transformations = _element_transformations(element)
    for coordinate_system in ("global", MERXEN_ALIGNMENT_COORDINATE_SYSTEM):
        if coordinate_system in transformations:
            return coordinate_system, transformations[coordinate_system]
    raise ValueError(
        "Raster has no 'global' or "
        f"{MERXEN_ALIGNMENT_COORDINATE_SYSTEM!r} transformation "
        f"(found: {sorted(transformations)})."
    )


def aligned_raster_transformation(element):
    """Return the raster's own alignment-frame transformation, if it lives only there.

    ``None`` for every raster that maps to ``global``: those are placed with the
    dataset's resolved pixel->micron transform.
    """
    transformations = _element_transformations(element)
    if "global" in transformations:
        return None
    return transformations.get(MERXEN_ALIGNMENT_COORDINATE_SYSTEM)


def napari_affine_from_transformation(transformation) -> np.ndarray:
    """napari (row, col) -> (y, x) affine of a SpatialData raster transformation."""
    m = np.asarray(
        transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        dtype=float,
    )
    return np.array(
        [
            [float(m[1, 1]), float(m[1, 0]), float(m[1, 2])],
            [float(m[0, 1]), float(m[0, 0]), float(m[0, 2])],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


MERSCOPE_TRANSFORM_CSV_NAME = "micron_to_mosaic_pixel_transform.csv"
# The SpatialData MERSCOPE reader stores cell boundaries in microns with the
# micron->mosaic-pixel affine as their ``global`` transformation, so the store
# carries the same matrix as the CSV even when the CSV itself was not copied.
MERSCOPE_TRANSFORM_SHAPES_KEY = "merscope_cell_boundaries"
MERSCOPE_FALLBACK_UM_PER_PX = 0.108
# Plausible MERSCOPE mosaic sampling in pixels per micron (nominal ~9.26, i.e.
# 0.108 um/px). The range rejects an identity (1 px/um: boundaries already in
# the micron frame) and nonsense matrices while tolerating other optics.
MERSCOPE_PX_PER_UM_RANGE = (2.0, 50.0)
# Mosaic pixels are square; a larger x/y scale ratio is not a micron->pixel map.
MERSCOPE_MAX_PIXEL_ANISOTROPY = 1.05


def merscope_transform_csv_candidates(zarr_path: str | Path) -> list[Path]:
    """Places a micron_to_mosaic_pixel_transform.csv may sit for a MERSCOPE store."""
    zarr_path = Path(zarr_path)
    return [zarr_path / MERSCOPE_TRANSFORM_CSV_NAME, zarr_path.parent / MERSCOPE_TRANSFORM_CSV_NAME]


def validate_micron_to_mosaic_matrix(matrix, *, source: str) -> np.ndarray:
    """Return ``matrix`` as a 3x3 float affine, or raise ``ValueError``.

    Accepts any orientation-preserving 2D affine (rotation and translation are
    handled exactly by the inverse) whose scale is a plausible, square MERSCOPE
    mosaic sampling.
    """
    try:
        m = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: transform is not numeric ({exc})") from exc
    if m.shape == (2, 3):
        m = np.vstack([m, (0.0, 0.0, 1.0)])
    if m.shape != (3, 3):
        raise ValueError(f"{source}: expected a 3x3 2D affine, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError(f"{source}: transform has non-finite entries")
    if not np.allclose(m[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise ValueError(f"{source}: last row {m[2].tolist()} is not [0, 0, 1]; not a 2D affine")
    linear = m[:2, :2]
    det = float(np.linalg.det(linear))
    if not det > 0.0:
        raise ValueError(f"{source}: transform is singular or mirrored (det={det:.6g})")
    scales = np.linalg.svd(linear, compute_uv=False)
    lo, hi = MERSCOPE_PX_PER_UM_RANGE
    if float(scales.min()) < lo or float(scales.max()) > hi:
        raise ValueError(
            f"{source}: scale {np.round(scales, 6).tolist()} px/um is outside the plausible "
            f"MERSCOPE mosaic range [{lo}, {hi}] px/um (nominal ~9.26)"
        )
    if float(scales.max() / scales.min()) > MERSCOPE_MAX_PIXEL_ANISOTROPY:
        raise ValueError(
            f"{source}: x/y scales {np.round(scales, 6).tolist()} px/um are not square mosaic pixels"
        )
    return m


def _read_json_mapping(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _zarr_element_attributes(zarr_path: Path, group: str, key: str) -> dict | None:
    """Return one element's zarr attributes from metadata only (no data reads).

    Prefers the element's own ``zarr.json`` (v3) or ``.zattrs`` (v2) and falls
    back to the root consolidated metadata.
    """
    element = zarr_path / group / key
    own_v3 = _read_json_mapping(element / "zarr.json")
    if own_v3 is not None and isinstance(own_v3.get("attributes"), dict):
        return own_v3["attributes"]
    own_v2 = _read_json_mapping(element / ".zattrs")
    if own_v2 is not None:
        return own_v2

    root = _read_json_mapping(zarr_path / "zarr.json")
    consolidated = root.get("consolidated_metadata") if root is not None else None
    metadata = consolidated.get("metadata") if isinstance(consolidated, dict) else None
    entry = metadata.get(f"{group}/{key}") if isinstance(metadata, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("attributes"), dict):
        return entry["attributes"]

    zmetadata = _read_json_mapping(zarr_path / ".zmetadata")
    metadata = zmetadata.get("metadata") if zmetadata is not None else None
    entry = metadata.get(f"{group}/{key}/.zattrs") if isinstance(metadata, dict) else None
    return entry if isinstance(entry, dict) else None


def _global_xy_affine_from_ngff(transformations, *, source: str) -> np.ndarray | None:
    """Return the x/y affine of the NGFF transformation into ``global``, if any."""
    if not isinstance(transformations, list):
        return None
    for entry in transformations:
        if not isinstance(entry, dict):
            continue
        output = entry.get("output")
        name = output.get("name") if isinstance(output, dict) else output
        if name != "global":
            continue
        try:
            from spatialdata.transformations.ngff.ngff_transformations import NgffBaseTransformation
            from spatialdata.transformations.transformations import BaseTransformation

            transformation = BaseTransformation.from_ngff(NgffBaseTransformation.from_dict(dict(entry)))
            matrix = transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        except Exception as exc:
            raise ValueError(f"{source}: could not read its 'global' transformation ({exc})") from exc
        return np.asarray(matrix, dtype=float)
    return None


def read_merscope_micron_to_mosaic_from_zarr(
    zarr_path: str | Path,
    shapes_key: str = MERSCOPE_TRANSFORM_SHAPES_KEY,
) -> np.ndarray | None:
    """Return the store's micron->mosaic-pixel matrix from metadata alone.

    Reads the ``global`` transformation of ``shapes/<shapes_key>`` without
    loading geometries. Returns ``None`` when the element or its ``global``
    transformation is absent; raises ``ValueError`` when it is present but is
    not a plausible MERSCOPE micron->mosaic-pixel affine.
    """
    zarr_path = Path(zarr_path)
    source = f"shapes/{shapes_key} in {zarr_path}"
    attrs = _zarr_element_attributes(zarr_path, "shapes", shapes_key)
    if attrs is None:
        return None
    matrix = _global_xy_affine_from_ngff(attrs.get("coordinateTransformations"), source=source)
    if matrix is None:
        return None
    return validate_micron_to_mosaic_matrix(matrix, source=f"{source} 'global' transformation")


def _resolve_merscope_micron_to_mosaic(
    merscope_transform_path: str | Path | None,
    zarr_path: str | Path | None,
) -> np.ndarray | None:
    """Pick the micron->mosaic matrix: CSV first, then the store's own transform."""
    missing: list[str] = []
    path = Path(merscope_transform_path) if merscope_transform_path else None
    if path is not None and path.exists():
        try:
            matrix = validate_micron_to_mosaic_matrix(np.loadtxt(path), source=str(path))
        except (OSError, ValueError) as exc:
            log.warning("[MERSCOPE] Ignoring transform file %s (%s)", path, exc)
            missing.append(f"{path} is not a usable micron->mosaic transform")
        else:
            log.info("[MERSCOPE] Placing images with the micron->mosaic transform from %s", path)
            return matrix
    elif path is not None:
        missing.append(f"{path} does not exist")
    elif zarr_path is not None:
        missing.append(f"no {MERSCOPE_TRANSFORM_CSV_NAME} in or beside {zarr_path}")
    else:
        missing.append(f"no {MERSCOPE_TRANSFORM_CSV_NAME} given")

    if zarr_path is not None:
        source = f"shapes/{MERSCOPE_TRANSFORM_SHAPES_KEY} 'global' transformation in {zarr_path}"
        try:
            matrix = read_merscope_micron_to_mosaic_from_zarr(zarr_path)
        except ValueError as exc:
            missing.append(f"{source} is unusable ({exc})")
        else:
            if matrix is not None:
                log.info(
                    "[MERSCOPE] Placing images with the micron->mosaic transform from the %s",
                    source,
                )
                return matrix
            missing.append(f"no {source}")

    log.warning(
        "[MERSCOPE] No micron->mosaic transform found (%s); using fallback %s um/px isotropic "
        "scale with no translation. Images will be offset from masks and transcripts unless "
        "the mosaic origin is (0, 0) um.",
        "; ".join(missing),
        MERSCOPE_FALLBACK_UM_PER_PX,
    )
    return None


SPEC_FILE_PIXEL_SIZE_KEYS = ("pixel_size", "microns_per_pixel")


def read_spec_file_pixel_size(path: str | Path, *, platform: str = "XENIUM") -> float | None:
    """Return ``pixel_size`` (or ``microns_per_pixel``) in um from a JSON spec file.

    Returns ``None`` (with a warning when the file exists) for a missing,
    unreadable or implausible value.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        spec = json.loads(path.read_text())
    except Exception as exc:
        log.warning("[%s] Failed to parse spec file %s (%s)", platform, path, exc)
        return None
    if not isinstance(spec, dict):
        log.warning("[%s] Spec file %s is not a JSON object", platform, path)
        return None
    for key in SPEC_FILE_PIXEL_SIZE_KEYS:
        if key in spec:
            value = _positive_float(spec[key])
            if value is None:
                log.warning("[%s] Spec file %s has an unusable %s=%r", platform, path, key, spec[key])
                return None
            return value
    return None


def _positive_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _atera_root_attr_pixel_size(zarr_path: Path) -> float | None:
    """``merxen_atera.experiment.pixel_size`` from the root attributes, if usable."""
    root = _read_json_mapping(zarr_path / "zarr.json")
    attrs = root.get("attributes") if isinstance(root, dict) else None
    if not isinstance(attrs, dict):
        attrs = _read_json_mapping(zarr_path / ".zattrs") or {}
    atera = attrs.get(ATERA_ROOT_ATTR)
    experiment = atera.get("experiment") if isinstance(atera, dict) else None
    if not isinstance(experiment, dict):
        return None
    return _positive_float(experiment.get("pixel_size"))


def _shapes_global_um_per_px(zarr_path: Path, shapes_key: str) -> float | None:
    """1 / scale of ``shapes/<key>``'s isotropic ``global`` transform (metadata only)."""
    attrs = _zarr_element_attributes(zarr_path, "shapes", shapes_key)
    if attrs is None:
        return None
    try:
        matrix = _global_xy_affine_from_ngff(
            attrs.get("coordinateTransformations"), source=f"shapes/{shapes_key} in {zarr_path}"
        )
    except ValueError:
        return None
    if matrix is None:
        return None
    sx, sy = float(matrix[0, 0]), float(matrix[1, 1])
    if abs(matrix[0, 1]) > 1e-9 or abs(matrix[1, 0]) > 1e-9 or sx <= 0 or sy <= 0:
        return None
    if abs(sx - sy) > 1e-6 * max(sx, sy):
        return None
    return _positive_float(1.0 / sx)


def _resolve_spec_file_um_per_px(
    ds: str,
    spec_path: str | Path | None,
    zarr_path: str | Path | None,
) -> float:
    spec = platform_spec(ds)
    path = Path(spec_path) if spec_path else None
    mpp = read_spec_file_pixel_size(path, platform=spec.name) if path is not None else None
    if mpp is not None:
        log.info("[%s] Pixel size %s um from %s", spec.name, mpp, path)
        return mpp
    if spec.name == "ATERA" and path is not None:
        log.warning(
            "[ATERA] Explicit spec file %s is %s; looking for the store's own sidecar.",
            path,
            "unreadable or has no usable pixel_size" if path.is_file() else "missing",
        )
    if spec.name == "ATERA" and zarr_path is not None:
        found = find_spec_path(zarr_path, spec.name)
        if found is not None and found != path:
            mpp = read_spec_file_pixel_size(found, platform=spec.name)
            if mpp is not None:
                log.info("[%s] Pixel size %s um from %s", spec.name, mpp, found)
                return mpp

    if spec.name != "ATERA":
        mpp = float(spec.fallback_um_per_px)
        log.warning(
            "[%s] Spec file missing/unreadable; using fallback pixel_size=%s um.",
            spec.name,
            mpp,
        )
        return mpp

    # ATERA: never fall back silently to a hard-coded value.
    if zarr_path is not None:
        store = Path(zarr_path)
        mpp = _atera_root_attr_pixel_size(store)
        if mpp is not None:
            log.warning(
                "[ATERA] No experiment.spatial found; using pixel_size=%s um from the root "
                "%s attributes of %s.",
                mpp,
                ATERA_ROOT_ATTR,
                store,
            )
            return mpp
        shapes_key = spec.original_cell_shape_key
        mpp = _shapes_global_um_per_px(store, shapes_key) if shapes_key else None
        if mpp is not None:
            log.warning(
                "[ATERA] No experiment.spatial or %s pixel_size found; using %s um/px from the "
                "'global' scale of shapes/%s in %s.",
                ATERA_ROOT_ATTR,
                mpp,
                shapes_key,
                store,
            )
            return mpp
    raise ValueError(
        "ATERA pixel size unknown: no experiment.spatial sidecar, no "
        f"{ATERA_ROOT_ATTR}.experiment.pixel_size attribute and no usable 'global' scale on "
        f"shapes/{spec.original_cell_shape_key}"
        + (f" in {zarr_path}" if zarr_path is not None else " (no store path given)")
    )


def resolve_dataset_mask_affine(
    dataset_name: str,
    merscope_transform_path: str | Path | None = None,
    xenium_spec_path: str | Path | None = None,
    *,
    zarr_path: str | Path | None = None,
    spec_path: str | Path | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve pixel->micron affine transform for a dataset.

    For MERSCOPE the micron->mosaic-pixel matrix comes from
    ``merscope_transform_path`` when it exists, otherwise from the ``global``
    transformation of ``shapes/merscope_cell_boundaries`` in ``zarr_path``;
    its inverse (translation included) is returned. Only when neither is usable
    does it fall back to a 0.108 um/px scale with no translation.

    For spec-file platforms (XENIUM, ATERA) the isotropic pixel size comes from
    ``spec_path`` (``xenium_spec_path`` is a compatibility alias), reading
    ``pixel_size`` or ``microns_per_pixel``. XENIUM falls back to 0.2125 um
    with a warning. ATERA looks for its ``experiment.spatial`` sidecar in or
    beside ``zarr_path``, then the root ``merxen_atera`` attributes, then the
    ``global`` scale of ``shapes/atera_cell_boundaries`` (each with a warning),
    and raises ``ValueError`` when none is available.
    """
    spec = find_platform_spec(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if spec.pixel_transform == PIXEL_TRANSFORM_MERSCOPE_AFFINE:
        matrix = _resolve_merscope_micron_to_mosaic(merscope_transform_path, zarr_path)
        if matrix is not None:
            inv = np.linalg.inv(matrix)
            return (
                (float(inv[0, 0]), float(inv[0, 1]), float(inv[0, 2])),
                (float(inv[1, 0]), float(inv[1, 1]), float(inv[1, 2])),
            )
        mpp = float(MERSCOPE_FALLBACK_UM_PER_PX)
        return (mpp, 0.0, 0.0), (0.0, mpp, 0.0)

    if spec.pixel_transform == PIXEL_TRANSFORM_SPEC_FILE:
        path = spec_path if spec_path is not None else xenium_spec_path
        mpp = _resolve_spec_file_um_per_px(spec.name, path, zarr_path)
        return (float(mpp), 0.0, 0.0), (0.0, float(mpp), 0.0)

    raise ValueError(f"Unknown dataset: {dataset_name}")


#: Partitions read ahead of the one being processed by the eager build.
_POINT_PARTITION_READ_AHEAD = 1


def _iter_point_partitions(
    points_obj,
    cols: list[str],
    *,
    cancel_check: Callable[[], bool] | None = None,
    read_ahead: int | None = None,
):
    """Yield the non-empty pandas partitions of ``points_obj[cols]`` in order.

    A dask frame's next ``read_ahead`` partitions are computed on one helper
    thread while the caller processes the current one, so reading overlaps the
    (vectorised) processing; at most ``read_ahead + 1`` partitions are held.
    ``cancel_check`` is polled before each partition is handed out, and
    closing the generator cancels the queued reads.
    """
    work = points_obj[cols]
    if hasattr(work, "to_delayed"):
        parts = list(work.to_delayed())
        depth = _POINT_PARTITION_READ_AHEAD if read_ahead is None else max(0, int(read_ahead))
        if depth == 0 or len(parts) <= 1:
            for part in parts:
                _check_transcript_cancelled(cancel_check)
                pdf = part.compute()
                if len(pdf) > 0:
                    yield pdf
            return
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="points-read-ahead")
        pending: deque = deque()
        next_part = 0
        try:
            while next_part < len(parts) and len(pending) <= depth:
                pending.append(executor.submit(parts[next_part].compute))
                next_part += 1
            while pending:
                _check_transcript_cancelled(cancel_check)
                pdf = pending.popleft().result()
                if next_part < len(parts) and len(pending) < depth:
                    pending.append(executor.submit(parts[next_part].compute))
                    next_part += 1
                if len(pdf) > 0:
                    yield pdf
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return

    pdf = to_pandas(work)
    if len(pdf) > 0:
        yield pdf


# ---------------------------------------------------------------------------
# Per-gene transcript inspection
# ---------------------------------------------------------------------------

GENE_COLUMN_CANDIDATES = ("gene", "feature_name", "target", "feature_id", "gene_id")

# Napari's point symbols, ordered so the most visually distinct shapes come
# first. Genes cycle through this list; combined with a distinct colour per
# gene, every gene gets a unique (colour, shape) pair.
GENE_MARKER_SYMBOLS = (
    "disc",
    "ring",
    "cross",
    "x",
    "square",
    "diamond",
    "triangle_up",
    "triangle_down",
    "star",
    "arrow",
    "hbar",
    "vbar",
    "clobber",
    "tailed_arrow",
)

GENE_MARKER_SYMBOL_LABELS = {
    "disc": "Disc",
    "ring": "Ring",
    "cross": "Cross",
    "x": "X",
    "square": "Square",
    "diamond": "Diamond",
    "triangle_up": "Triangle up",
    "triangle_down": "Triangle down",
    "star": "Star",
    "arrow": "Arrow",
    "hbar": "Horizontal bar",
    "vbar": "Vertical bar",
    "clobber": "Clobber",
    "tailed_arrow": "Tailed arrow",
}

_CONTROL_GENE_PATTERNS = (
    "blank",
    "negcontrol",
    "neg_control",
    "negprobe",
    "neg_probe",
    "antisense",
    "intergenic",
    "deprecated",
    "unassigned_codeword",
    "unassignedcodeword",
    "genomic_control",
)


def resolve_gene_column(df_like) -> str | None:
    """Return the transcript gene/feature-name column, if present."""
    return first_existing_col(df_like, GENE_COLUMN_CANDIDATES)


def is_control_gene(name: object) -> bool:
    """True for negative-control / blank codeword names (not real genes)."""
    text = str(name).strip().lower()
    if not text:
        return False
    return any(pattern in text for pattern in _CONTROL_GENE_PATTERNS)


#: Atera transcript column flagging a real gene (False for control probes and
#: codewords).  It is the authority for the control set when present; the name
#: rule (:func:`is_control_gene`) is only cross-checked against it.
IS_GENE_COLUMN = "is_gene"


def accumulate_is_gene_flags(flags: dict[str, bool], names, is_gene) -> None:
    """Fold one partition's ``(feature name, is_gene)`` rows into ``flags``.

    A feature counts as a real gene only if every one of its rows says so, so a
    feature with any ``is_gene == False`` row (or a missing flag) is a control.
    """
    names = np.asarray(names, dtype=object)
    values = pd.Series(is_gene).astype("boolean").fillna(False).to_numpy(dtype=bool)
    if names.size == 0:
        return
    keep = pd.notna(names)
    names = names[keep].astype(str)
    values = values[keep]
    if names.size == 0:
        return
    unique, inverse = np.unique(names, return_inverse=True)
    # A per-feature all(): one False row makes the feature a control.
    gene_all = np.ones(len(unique), dtype=bool)
    np.logical_and.at(gene_all, inverse, values)
    for name, flag in zip(unique.tolist(), gene_all.tolist(), strict=True):
        text = str(name)
        flags[text] = bool(flags.get(text, True) and flag)


def control_genes_from_is_gene(
    names: Iterable[str],
    is_gene_flags: Mapping[str, bool] | None,
    *,
    source: str = "",
) -> set[str]:
    """Return the control features of ``names``, preferring an ``is_gene`` flag.

    With ``is_gene_flags`` (Atera points carry ``is_gene``) a feature is a control
    exactly when it is flagged ``False``; a name the flags do not cover falls back
    to :func:`is_control_gene`.  Every disagreement between the flag and the name
    rule is logged once as a warning so a new control family, or a gene whose name
    merely looks like a control, is visible rather than silently misfiled.
    Without flags this is the name rule alone (Xenium/MERSCOPE behaviour).
    """
    names = [str(name) for name in names]
    if not is_gene_flags:
        return {name for name in names if is_control_gene(name)}
    controls: set[str] = set()
    flag_only: list[str] = []
    name_only: list[str] = []
    for name in names:
        by_name = is_control_gene(name)
        flag = is_gene_flags.get(name)
        if flag is None:
            if by_name:
                controls.add(name)
            continue
        if not flag:
            controls.add(name)
            if not by_name:
                flag_only.append(name)
        elif by_name:
            name_only.append(name)
    if flag_only or name_only:
        where = f" in {source}" if source else ""
        log.warning(
            "is_gene disagrees with the control-name rule%s: %d feature(s) flagged "
            "is_gene=False with a gene-like name (treated as controls, e.g. %s); "
            "%d flagged is_gene=True with a control-like name (treated as genes, e.g. %s)",
            where,
            len(flag_only),
            sorted(flag_only)[:5],
            len(name_only),
            sorted(name_only)[:5],
        )
    return controls


def _hex_to_rgba(value: str, alpha: float) -> tuple[float, float, float, float]:
    text = str(value).lstrip("#")
    r = int(text[0:2], 16) / 255.0
    g = int(text[2:4], 16) / 255.0
    b = int(text[4:6], 16) / 255.0
    return (float(r), float(g), float(b), float(alpha))


def _distinct_palette(n: int) -> list[tuple[float, float, float]]:
    """Fallback max-distinct RGB palette when colorcet is unavailable."""
    colors: list[tuple[float, float, float]] = []
    golden = 0.61803398875
    for i in range(max(0, int(n))):
        hue = (i * golden) % 1.0
        sat = 0.65 + 0.30 * ((i // 2) % 2)
        val = 0.95 - 0.30 * (i % 3) / 2.0
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    return colors


def gene_palette_rgba(n: int, alpha: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Return ``n`` deterministic, maximally distinct RGBA colours."""
    try:
        import colorcet as cc

        base = list(cc.glasbey)
        return [_hex_to_rgba(base[i % len(base)], alpha) for i in range(max(0, int(n)))]
    except Exception:  # pragma: no cover - depends on runtime env
        return [(r, g, b, float(alpha)) for (r, g, b) in _distinct_palette(n)]


@dataclass(frozen=True)
class GeneVisual:
    """The stable colour + marker shape assigned to one gene."""

    rgba: tuple[float, float, float, float]
    symbol: str


def assign_gene_visuals(
    gene_names: Iterable[str],
    alpha: float = 1.0,
) -> dict[str, GeneVisual]:
    """Assign each gene a deterministic (colour, marker shape) pair.

    Genes are sorted alphabetically and indexed ``i``. The shape cycles through
    :data:`GENE_MARKER_SYMBOLS` (period 14) and the colour cycles through a
    max-distinct palette (period 256). The two periods only realign after
    ``lcm(14, 256) = 1792`` genes, so every real Xenium/MERSCOPE panel gets a
    unique (colour, shape) pair per gene with no collisions.
    """
    names = sorted({str(name) for name in gene_names})
    palette = gene_palette_rgba(len(names), alpha=alpha)
    visuals: dict[str, GeneVisual] = {}
    for i, name in enumerate(names):
        symbol = GENE_MARKER_SYMBOLS[i % len(GENE_MARKER_SYMBOLS)]
        visuals[name] = GeneVisual(rgba=palette[i], symbol=symbol)
    return visuals


class GeneSlotAllocator:
    """Colour slots for the genes shown in selection mode.

    Selection mode draws at most ``cap`` genes at once, each as a ``disc`` in
    one of ``cap`` maximally distinct (glasbey) colours. A newly selected gene
    takes the lowest free slot, so a freed colour is reused before a new one is
    handed out and the palette never grows past ``cap``. ``allocate`` returns
    ``None`` once every slot is taken; callers show a status instead.
    """

    SYMBOL = "disc"

    def __init__(self, cap: int = GENE_SELECTION_SLOT_CAP, alpha: float = 1.0):
        self.cap = max(1, int(cap))
        self._palette = gene_palette_rgba(self.cap, alpha=alpha)
        self._slot_of: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._slot_of)

    def __contains__(self, gene: object) -> bool:
        return str(gene) in self._slot_of

    @property
    def full(self) -> bool:
        return len(self._slot_of) >= self.cap

    @property
    def genes(self) -> list[str]:
        """Allocated genes in slot order."""
        return sorted(self._slot_of, key=self._slot_of.__getitem__)

    def slot_of(self, gene: object) -> int | None:
        return self._slot_of.get(str(gene))

    def allocate(self, gene: object) -> int | None:
        """Give ``gene`` the lowest free slot (idempotent); ``None`` when full."""
        gene = str(gene)
        if gene in self._slot_of:
            return self._slot_of[gene]
        if self.full:
            return None
        used = set(self._slot_of.values())
        slot = next(i for i in range(self.cap) if i not in used)
        self._slot_of[gene] = slot
        return slot

    def release(self, gene: object) -> int | None:
        """Free ``gene``'s slot and return it (``None`` if it had none)."""
        return self._slot_of.pop(str(gene), None)

    def clear(self) -> None:
        self._slot_of.clear()

    def slot_rgba(self, slot: int) -> tuple[float, float, float, float]:
        return self._palette[int(slot) % self.cap]

    def visual(self, gene: object) -> GeneVisual | None:
        """The slot colour + disc for an allocated ``gene``, else ``None``."""
        slot = self.slot_of(gene)
        if slot is None:
            return None
        return GeneVisual(rgba=self.slot_rgba(slot), symbol=self.SYMBOL)


# -- Cell-type marker reference ------------------------------------------------
# A curated map from marker gene -> (broad, fine) cell type, optionally stored in
# a SpatialData object so the gene inspector can group genes by the cell type
# they mark. The broad set is fixed; each broad hue is deliberately far from the
# others so a group reads as one colour family, and (in the fine scheme) the fine
# subtypes of one broad type stay close in hue while different broad types don't.
CELL_TYPE_BROAD_COL = "broad_cell_type"
CELL_TYPE_FINE_COL = "fine_cell_type"
CELL_TYPE_MARKER_UNS_KEY = "cell_type_marker_reference"
CONTROL_GROUP_TITLE = "Control / blank probes"
UNCLASSIFIED_GROUP_TITLE = "Unclassified"

COARSE_CELL_TYPE_ORDER = (
    "Astrocyte",
    "Fibroblast",
    "Microglia",
    "Neuron",
    "Oligodendrocyte",
    "Oligodendrocyte precursor",
    "Vascular",
)
#: One clearly-distinct base hue (0..1) per broad cell type. Oligodendrocyte and
#: its precursor sit next to each other (violet/magenta) so the lineage reads as
#: related; every other pair is well separated around the wheel.
COARSE_CELL_TYPE_HUES = {
    "Astrocyte": 0.33,
    "Fibroblast": 0.13,
    "Microglia": 0.00,
    "Neuron": 0.62,
    "Oligodendrocyte": 0.78,
    "Oligodendrocyte precursor": 0.88,
    "Vascular": 0.50,
}
#: Functional (non-cell-type) broad groups for panel genes that don't mark a cell
#: type -- Alzheimer's, autophagy, cell-cycle, stress. Hues sit in the gaps
#: between the cell-type hues so they stay distinguishable.
FUNCTIONAL_GROUP_HUES = {
    "Alzheimer's disease": 0.95,
    "Autophagy & lysosomal": 0.22,
    "Cell cycle": 0.43,
    "Stress & chaperones": 0.70,
}
#: Broad groups rendered as neutral greys rather than a colour family (controls
#: and the catch-all functional bucket).
GREY_GROUP_TITLES = {CONTROL_GROUP_TITLE, UNCLASSIFIED_GROUP_TITLE, "Other / signalling"}


@dataclass(frozen=True)
class MarkerReferenceResolution:
    """A panel-matched marker reference plus user-facing provenance."""

    reference: dict[str, dict[str, str]] | None
    source: str = ""
    panel_fingerprint: str = ""
    panel_gene_count: int = 0
    matched_gene_count: int = 0
    broad_gene_count: int = 0
    fine_gene_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def broad_available(self) -> bool:
        return self.broad_gene_count > 0

    @property
    def fine_available(self) -> bool:
        return self.fine_gene_count > 0

    @property
    def summary(self) -> str:
        if self.reference is None:
            return "No cell-type marker reference matched this gene panel."
        return (
            f"Cell-type reference: {self.source} "
            f"({self.matched_gene_count}/{self.panel_gene_count} panel genes classified)."
        )


@dataclass
class GeneVisualScheme:
    """A full (colour, shape) assignment plus the grouped display order.

    ``visuals`` maps every gene (including controls) to its :class:`GeneVisual`.
    ``groups`` is the ordered list of ``(title, genes)`` sections the gene
    inspector renders under headings. ``kind`` is ``"coarse"`` or ``"fine"``.
    Symbols are identical across the coarse and fine schemes (they depend only on
    the broad grouping), so switching schemes only recolours -- it never changes
    which napari Points layer a transcript belongs to.
    """

    kind: str
    visuals: dict[str, GeneVisual]
    groups: list[tuple[str, list[str]]]


def _broad_hue(broad: str) -> float:
    """Return the base hue for a broad group (hashed fallback if unknown)."""
    if broad in COARSE_CELL_TYPE_HUES:
        return float(COARSE_CELL_TYPE_HUES[broad])
    if broad in FUNCTIONAL_GROUP_HUES:
        return float(FUNCTIONAL_GROUP_HUES[broad])
    digest = blake2s(str(broad).encode("utf-8"), digest_size=4).hexdigest()
    return (int(digest, 16) / 0xFFFFFFFF) % 1.0


def _shade_rgba(
    hue: float,
    i: int,
    n: int,
    alpha: float = 1.0,
    sat_range: tuple[float, float] = (0.55, 0.95),
    val_range: tuple[float, float] = (0.62, 0.98),
) -> tuple[float, float, float, float]:
    """Return the ``i``-th of ``n`` distinguishable shades of one hue.

    Saturation and value are walked in opposite directions along a golden-ratio
    sequence, so consecutive genes -- and genes 14 apart that share a marker
    symbol -- land on visibly different shades while staying the same colour.
    """
    if n <= 1:
        s, v = 0.85, 0.92
    else:
        t = (int(i) * 0.61803398875) % 1.0
        v = val_range[0] + (val_range[1] - val_range[0]) * t
        s = sat_range[1] - (sat_range[1] - sat_range[0]) * t
    r, g, b = colorsys.hsv_to_rgb(float(hue) % 1.0, s, v)
    return (float(r), float(g), float(b), float(alpha))


def _control_shade_rgba(i: int, n: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Neutral grey shades for control / blank probes (no colour family)."""
    t = (int(i) * 0.61803398875) % 1.0 if n > 1 else 0.4
    v = 0.45 + 0.35 * t
    return (float(v), float(v), float(v), float(alpha))


def _normalize_reference(reference: Any) -> dict[str, dict[str, str]]:
    """Coerce a marker reference into ``{gene: {"broad": .., "fine": ..}}``.

    Accepts a mapping, a JSON string (as stored in a table's ``uns``), or ``None``.
    """
    out: dict[str, dict[str, str]] = {}
    if reference is None:
        return out
    if isinstance(reference, (str, bytes)):
        try:
            reference = json.loads(reference)
        except Exception:
            return out
    if not reference:
        return out
    items = reference.get("genes", reference) if isinstance(reference, Mapping) else {}
    if not isinstance(items, Mapping):
        return out
    for gene, info in items.items():
        if isinstance(info, Mapping):
            broad = info.get("broad") or info.get(CELL_TYPE_BROAD_COL)
            fine = info.get("fine") or info.get(CELL_TYPE_FINE_COL)
        elif isinstance(info, (list, tuple)) and len(info) >= 2:
            broad, fine = info[0], info[1]
        else:
            continue
        if broad is None:
            continue
        out[str(gene)] = {"broad": str(broad), "fine": str(fine) if fine is not None else ""}
    return out


def _symbols_by_broad(genes: Iterable[str], reference: Mapping[str, dict]) -> dict[str, str]:
    """Assign each gene a marker symbol from its within-broad-group index.

    Grouping by broad type (controls form their own group) and cycling the symbol
    list per group keeps symbols stable whether the inspector is grouped by broad
    or fine, and maximises symbol diversity inside each broad group.
    """
    grouped: dict[str, list[str]] = {}
    for gene in genes:
        info = reference.get(str(gene))
        broad = (
            info["broad"]
            if info
            else CONTROL_GROUP_TITLE if is_control_gene(gene) else UNCLASSIFIED_GROUP_TITLE
        )
        grouped.setdefault(broad, []).append(str(gene))
    symbols: dict[str, str] = {}
    for names in grouped.values():
        for i, name in enumerate(sorted(names)):
            symbols[name] = GENE_MARKER_SYMBOLS[i % len(GENE_MARKER_SYMBOLS)]
    return symbols


def _coarse_groups(genes: Iterable[str], reference: Mapping[str, dict]) -> list[tuple[str, list[str]]]:
    """Ordered ``(broad, genes)`` sections, with unclassified/control tails."""
    by_broad: dict[str, list[str]] = {}
    unclassified: list[str] = []
    controls: list[str] = []
    for gene in genes:
        info = reference.get(str(gene))
        if info:
            by_broad.setdefault(info["broad"], []).append(str(gene))
        elif is_control_gene(gene):
            controls.append(str(gene))
        else:
            unclassified.append(str(gene))
    groups = [(broad, sorted(by_broad[broad])) for broad in sorted(by_broad)]
    if unclassified:
        groups.append((UNCLASSIFIED_GROUP_TITLE, sorted(unclassified)))
    if controls:
        groups.append((CONTROL_GROUP_TITLE, sorted(controls)))
    return groups


def _fine_groups(
    genes: Iterable[str], reference: Mapping[str, dict]
) -> list[tuple[str, str, str, list[str]]]:
    """Ordered ``(title, broad, fine, genes)`` sections by (broad, fine) alphabetical."""
    keyed: dict[tuple[str, str], list[str]] = {}
    unclassified: list[str] = []
    controls: list[str] = []
    for gene in genes:
        info = reference.get(str(gene))
        if info:
            keyed.setdefault((info["broad"], info["fine"]), []).append(str(gene))
        elif is_control_gene(gene):
            controls.append(str(gene))
        else:
            unclassified.append(str(gene))
    groups: list[tuple[str, str, str, list[str]]] = []
    for broad, fine in sorted(keyed):
        title = f"{broad} — {fine}" if fine else broad
        groups.append((title, broad, fine, sorted(keyed[(broad, fine)])))
    if unclassified:
        groups.append(
            (UNCLASSIFIED_GROUP_TITLE, UNCLASSIFIED_GROUP_TITLE, "", sorted(unclassified))
        )
    if controls:
        groups.append((CONTROL_GROUP_TITLE, CONTROL_GROUP_TITLE, "", sorted(controls)))
    return groups


def _fine_hue_map(fine_groups: list[tuple[str, str, str, list[str]]]) -> dict[tuple[str, str], float]:
    """Hue per ``(broad, fine)`` -- fine subtypes cluster around their broad hue."""
    fines_by_broad: dict[str, list[str]] = {}
    for _title, broad, fine, _names in fine_groups:
        if broad in GREY_GROUP_TITLES:
            continue
        fines_by_broad.setdefault(broad, []).append(fine)
    hue_map: dict[tuple[str, str], float] = {}
    for broad, fines in fines_by_broad.items():
        base = _broad_hue(broad)
        ordered = sorted(set(fines))
        k = len(ordered)
        # Wider arc when a broad type has many subtypes, capped so it never bleeds
        # into a neighbouring broad hue.
        band = min(0.13, 0.028 * k)
        for j, fine in enumerate(ordered):
            offset = 0.0 if k <= 1 else band * (j / (k - 1) - 0.5)
            hue_map[(broad, fine)] = base + offset
    return hue_map


def build_cell_type_gene_visuals(
    genes: Iterable[str],
    reference: Mapping[str, Any] | None,
    kind: str = "coarse",
    alpha: float = 1.0,
) -> GeneVisualScheme:
    """Build a :class:`GeneVisualScheme` grouping genes by broad or fine cell type.

    ``kind="coarse"`` gives each broad type one hue (shaded per gene); ``"fine"``
    gives each fine subtype its own hue near its broad type's hue. Symbols are the
    same in both. Unmatched biological genes and control/codeword probes form
    separate trailing grey groups. Falls back to :func:`assign_gene_visuals` when
    no reference genes match, so callers get a usable scheme regardless.
    """
    names = sorted({str(g) for g in genes})
    ref = _normalize_reference(reference)
    if not any(name in ref for name in names):
        base = assign_gene_visuals(names, alpha=alpha)
        return GeneVisualScheme(kind=kind, visuals=base, groups=[("Genes", names)])

    symbols = _symbols_by_broad(names, ref)
    visuals: dict[str, GeneVisual] = {}
    display_groups: list[tuple[str, list[str]]] = []

    if str(kind) == "fine":
        fine_groups = _fine_groups(names, ref)
        hue_map = _fine_hue_map(fine_groups)
        for title, broad, fine, group_names in fine_groups:
            for i, name in enumerate(group_names):
                if broad in GREY_GROUP_TITLES:
                    rgba = _control_shade_rgba(i, len(group_names), alpha)
                else:
                    rgba = _shade_rgba(
                        hue_map[(broad, fine)], i, len(group_names), alpha,
                        sat_range=(0.6, 0.98), val_range=(0.66, 0.99),
                    )
                visuals[name] = GeneVisual(rgba=rgba, symbol=symbols.get(name, "disc"))
            display_groups.append((title, group_names))
    else:
        for broad, group_names in _coarse_groups(names, ref):
            hue = _broad_hue(broad)
            for i, name in enumerate(group_names):
                if broad in GREY_GROUP_TITLES:
                    rgba = _control_shade_rgba(i, len(group_names), alpha)
                else:
                    rgba = _shade_rgba(hue, i, len(group_names), alpha)
                visuals[name] = GeneVisual(rgba=rgba, symbol=symbols.get(name, "disc"))
            display_groups.append((broad, group_names))

    # Any gene the reference/grouping missed still needs a visual.
    for i, name in enumerate(names):
        visuals.setdefault(name, GeneVisual(rgba=_control_shade_rgba(i, len(names), alpha), symbol="disc"))
    return GeneVisualScheme(kind=str(kind), visuals=visuals, groups=display_groups)


def _reference_from_frame(frame: Any) -> dict[str, dict[str, str]]:
    """Read marker columns from an AnnData ``obs`` or ``var`` frame."""
    if frame is None or CELL_TYPE_BROAD_COL not in getattr(frame, "columns", []):
        return {}
    out: dict[str, dict[str, str]] = {}
    fine_present = CELL_TYPE_FINE_COL in frame.columns
    for gene, broad in frame[CELL_TYPE_BROAD_COL].items():
        if broad is None or (isinstance(broad, float) and np.isnan(broad)):
            continue
        text = str(broad).strip()
        if not text or text.lower() in _MISSING_LABEL_TEXTS:
            continue
        fine = str(frame[CELL_TYPE_FINE_COL].get(gene, "")).strip() if fine_present else ""
        out[str(gene)] = {
            "broad": text,
            "fine": "" if fine.lower() in _MISSING_LABEL_TEXTS else fine,
        }
    return out


def load_cell_type_marker_reference(source: Any) -> dict[str, dict[str, str]] | None:
    """Load an embedded marker reference from SpatialData/AnnData or a store path.

    Looks (in order) for a ``cell_type_marker_reference`` entry in the table's
    ``uns``, then for ``broad_cell_type`` / ``fine_cell_type`` columns in the
    table's ``obs`` or ``var``. Returns ``{gene: {"broad", "fine"}}`` or ``None``
    if absent. Read failures are logged and treated as an absent reference.
    """
    try:
        table = _resolve_marker_table(source)
    except Exception as exc:
        log.warning("Could not inspect cell-type marker reference in %s: %s", source, exc)
        return None
    if table is None:
        return None

    uns = getattr(table, "uns", None)
    if isinstance(uns, Mapping) and CELL_TYPE_MARKER_UNS_KEY in uns:
        ref = _normalize_reference(uns[CELL_TYPE_MARKER_UNS_KEY])
        if ref:
            return ref

    # Dedicated marker tables conventionally put one gene per obs row. Older
    # stores used var rows, so retain both encodings.
    for frame_name in ("obs", "var"):
        out = _reference_from_frame(getattr(table, frame_name, None))
        if out:
            return out
    return None


def gene_panel_fingerprint(genes: Iterable[Any]) -> str:
    """Stable identity for a biological panel, excluding controls/codewords."""
    names = sorted({str(gene).strip() for gene in genes if not is_control_gene(gene)})
    return blake2s("\n".join(names).encode("utf-8"), digest_size=16).hexdigest()


def _load_marker_reference_file(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Read a standalone JSON/CSV marker reference and its metadata."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        return _normalize_reference(raw), raw if isinstance(raw, Mapping) else {}
    frame = pd.read_csv(path)
    gene_col = next((name for name in ("gene", "feature_name", "target") if name in frame), None)
    if gene_col is None or CELL_TYPE_BROAD_COL not in frame:
        raise ValueError("expected gene and broad_cell_type columns")
    frame = frame.set_index(gene_col)
    return _reference_from_frame(frame), {}


def _reference_resolution(
    reference: Mapping[str, Any] | None,
    genes: Iterable[Any],
    *,
    source: str,
    warnings: Iterable[str] = (),
) -> MarkerReferenceResolution | None:
    """Align one candidate reference to actual panel spelling and measure coverage."""
    normalized = _normalize_reference(reference)
    panel_genes = sorted({str(g).strip() for g in genes if not is_control_gene(g)})
    folded: dict[str, dict[str, str] | None] = {}
    for name, info in normalized.items():
        key = name.casefold()
        folded[key] = info if key not in folded else None
    matched: dict[str, dict[str, str]] = {}
    for gene in panel_genes:
        info = normalized.get(gene)
        if info is None:
            info = folded.get(gene.casefold())
        if info:
            matched[gene] = dict(info)
    if not matched:
        return None
    return MarkerReferenceResolution(
        reference=matched,
        source=str(source),
        panel_fingerprint=gene_panel_fingerprint(panel_genes),
        panel_gene_count=len(panel_genes),
        matched_gene_count=len(matched),
        broad_gene_count=sum(bool(info.get("broad")) for info in matched.values()),
        fine_gene_count=sum(bool(info.get("fine")) for info in matched.values()),
        warnings=tuple(str(item) for item in warnings),
    )


def _marker_reference_sidecars(zarr_path: Path) -> list[Path]:
    directories = [zarr_path.parent, zarr_path.parent.parent]
    paths: list[Path] = []
    for directory in directories:
        for suffix in ("json", "csv"):
            candidate = directory / f"{CELL_TYPE_MARKER_UNS_KEY}.{suffix}"
            if candidate.is_file() and candidate not in paths:
                paths.append(candidate)
    return paths


def _bundled_marker_reference_files() -> list[Path]:
    directory = Path(__file__).parent / "resources" / "cell_type_marker_references"
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    configured = os.environ.get("NAPARI_COMPARE_MARKER_REFERENCE_DIR", "")
    for value in configured.split(os.pathsep):
        external = Path(value).expanduser() if value else None
        if external is not None and external.is_dir():
            paths.extend(path for path in sorted(external.glob("*.json")) if path not in paths)
    return paths


def resolve_cell_type_marker_reference(
    zarr_path: str | Path | None,
    genes: Iterable[Any],
) -> MarkerReferenceResolution:
    """Resolve embedded, sidecar, or bundled metadata for an exact gene panel.

    Store-local metadata wins. Bundled catalogue entries are selected only by
    exact panel fingerprint, so similarly named samples cannot silently receive
    an incompatible reference.
    """
    panel_genes = tuple(str(g).strip() for g in genes if not is_control_gene(g))
    fingerprint = gene_panel_fingerprint(panel_genes)
    problems: list[str] = []
    path = Path(zarr_path) if zarr_path is not None else None

    if path is not None:
        embedded = load_cell_type_marker_reference(path)
        resolution = _reference_resolution(embedded, panel_genes, source="embedded SpatialData table")
        if resolution is not None:
            return resolution

        for candidate in _marker_reference_sidecars(path):
            try:
                reference, _metadata = _load_marker_reference_file(candidate)
                resolution = _reference_resolution(reference, panel_genes, source=str(candidate))
            except Exception as exc:
                problems.append(f"Could not read {candidate}: {exc}")
                log.warning(problems[-1])
                continue
            if resolution is not None:
                return resolution

    for candidate in _bundled_marker_reference_files():
        try:
            reference, metadata = _load_marker_reference_file(candidate)
        except Exception as exc:
            problems.append(f"Could not read bundled reference {candidate.name}: {exc}")
            log.warning(problems[-1])
            continue
        if str(metadata.get("panel_fingerprint", "")) != fingerprint:
            continue
        resolution = _reference_resolution(reference, panel_genes, source=f"bundled {candidate.name}")
        if resolution is not None:
            return resolution

    return MarkerReferenceResolution(
        reference=None,
        panel_fingerprint=fingerprint,
        panel_gene_count=len(set(panel_genes)),
        warnings=tuple(problems),
    )


def _read_marker_table_parts(group) -> Any:
    """Read only the marker-bearing parts of one AnnData zarr group.

    Never reads ``X``, ``obsm`` or ``layers``: an Atera ``table_original`` holds
    millions of cells by ~18k genes, so a full ``ad.read_zarr`` would load
    gigabytes to look for a few hundred marker rows. ``uns`` contributes only
    the marker entry, and ``obs`` / ``var`` are read only when their column order
    lists ``broad_cell_type``. Returns an object with ``uns`` / ``obs`` / ``var``
    (``None`` for parts not read), or ``None`` when the table has no marker data.
    """
    from types import SimpleNamespace

    from anndata.io import read_elem

    uns: dict[str, Any] = {}
    try:
        uns_group = group["uns"]
    except KeyError:
        uns_group = None
    if uns_group is not None:
        try:
            marker = uns_group[CELL_TYPE_MARKER_UNS_KEY]
        except KeyError:
            marker = None
        if marker is not None:
            uns[CELL_TYPE_MARKER_UNS_KEY] = read_elem(marker)
    frames: dict[str, Any] = {"obs": None, "var": None}
    for name in frames:
        try:
            frame_group = group[name]
        except KeyError:
            continue
        columns = frame_group.attrs.get("column-order", [])
        if isinstance(columns, str):
            columns = [columns]
        if CELL_TYPE_BROAD_COL in list(columns):
            frames[name] = read_elem(frame_group)
    if not uns and frames["obs"] is None and frames["var"] is None:
        return None
    return SimpleNamespace(uns=uns, obs=frames["obs"], var=frames["var"])


def _resolve_marker_table(source: Any):
    """Return the AnnData table carrying the marker reference, or ``None``."""
    if source is None:
        return None
    # Read AnnData tables directly instead of constructing a partial SpatialData
    # object.  ``SpatialData.read_zarr(..., selection=("tables",))`` validates
    # every table against its annotated region; because shapes/labels were
    # intentionally omitted by that selection it emits misleading "region is
    # not present" warnings for otherwise valid stores.  Direct reads also let
    # us stop as soon as the small dedicated marker-reference table is found.
    if isinstance(source, (str, Path)):
        path = Path(source)
        root = open_zarr_group_unconsolidated(path)
        if "tables" not in root:
            return None
        tables_group = root["tables"]
        table_keys = list(tables_group.group_keys())
        preferred = [CELL_TYPE_MARKER_UNS_KEY, "table"]
        ordered_keys = list(dict.fromkeys([*preferred, *table_keys]))
        for key in ordered_keys:
            if key not in table_keys:
                continue
            try:
                table = _read_marker_table_parts(tables_group[key])
            except Exception as exc:
                log.warning("Could not read marker candidate table %s in %s: %s", key, path, exc)
                continue
            if table is not None:
                return table
        return None
    tables = getattr(source, "tables", None)
    if tables is not None:  # a SpatialData object
        for key in ("table", *tables.keys()):
            table = tables.get(key)
            if table is None:
                continue
            uns = getattr(table, "uns", {})
            obs = getattr(table, "obs", None)
            var = getattr(table, "var", None)
            has_uns = isinstance(uns, Mapping) and CELL_TYPE_MARKER_UNS_KEY in uns
            has_obs = obs is not None and CELL_TYPE_BROAD_COL in getattr(obs, "columns", [])
            has_var = var is not None and CELL_TYPE_BROAD_COL in getattr(var, "columns", [])
            if has_uns or has_obs or has_var:
                return table
        return tables.get("table")
    return source  # assume already an AnnData-like table


# -- Per-cell cell-type assignments (from MerXen clustering) -------------------
# MerXen's clustering-squidpy stage writes a per-cell annotation table back into
# the store as ``<source_table>_clustering_squidpy``. Each cell row carries a
# broad class (``broad_class``, human-readable, e.g. "Neurons") and a fine label
# (``hierarchical_cluster``, formatted ``"{broad}:{sub}"`` or
# ``"{broad}/{split}:{sub}"`` for neurons). The row joins to a segmentation label
# mask through the SpatialData instance key: the mask's integer pixel ids equal
# that key's per-row values. No colours are stored -- they are generated here to
# mirror the transcript scheme (a max-distinct hue per broad type, shaded per
# fine subtype). See [[celltype-marker-reference]] for the gene-level analogue.
CELL_TYPE_BROAD_CLASS_COL = "broad_class"
CELL_TYPE_FINE_CLUSTER_COL = "hierarchical_cluster"
CELL_TYPE_NEURON_SPLIT_COL = "neuron_split_label"
CLUSTERING_TABLE_SUFFIX = "_clustering_squidpy"
#: MerXen collapses all neurons into a single ``broad_class``; split it back into
#: excitatory/inhibitory broad categories using ``neuron_split_label``.
NEURON_BROAD_CLASS = "Neurons"
_NEURON_SPLIT_LABELS = {"excitatory": "Excitatory neurons", "inhibitory": "Inhibitory neurons"}

#: Instance-key candidates when a table omits ``uns['spatialdata_attrs']``.
_INSTANCE_KEY_CANDIDATES = ("instance_id", "cell", "cell_id", "cells", "cell_ID", "EntityID", "label_id")
_MISSING_LABEL_TEXTS = {"", "nan", "none", "na", "<na>"}


@dataclass(frozen=True)
class CellTypeSource:
    """One selectable cell-type layer: a clustering table joined to a mask.

    A single set of segmentation masks can be coloured by more than one
    clustering: ProSeg masks by the *reseg* (probabilistic-assignment) clustering
    or by a *proseg-mask* (in-mask transcript) clustering; the original instrument
    masks only by the *original* clustering. Each source pins the clustering
    ``clustering_tables`` (including their named map_first variants) to the
    ``mask_shape_keys`` it annotates (first present wins), so a request never joins a clustering's
    instance ids against the wrong mask.
    """

    key: str
    label: str
    clustering_tables: tuple[str, ...]
    mask_shape_keys: tuple[str, ...]
    method: str = "clustering"
    segmentation: str | None = None


#: Every cell-type source the viewer knows about, in display order. ``key`` is the
#: stable identifier stored on the overlay state and passed through the widget;
#: ``label`` is the button text. ``"proseg"`` is kept as the reseg key for
#: backward compatibility (the historical name for this clustering).
CELL_TYPE_SOURCES: tuple[CellTypeSource, ...] = (
    CellTypeSource(
        key="proseg_hybrid",
        label="ProSeg hybrid",
        clustering_tables=("table_MOSAIK_proseg_hybrid_clustering_squidpy",),
        mask_shape_keys=("MOSAIK_proseg_hybrid",),
    ),
    CellTypeSource(
        key="proseg",
        label="ProSeg · reseg (probabilistic)",
        clustering_tables=("table_MOSAIK_proseg_clustering_squidpy",),
        mask_shape_keys=("MOSAIK_proseg",),
    ),
    CellTypeSource(
        key="proseg_mask",
        label="ProSeg · in-mask counts",
        clustering_tables=(
            "table_MOSAIK_proseg_mask_clustering_squidpy",
            "table_MOSAIK_proseg_mask_counts_clustering_squidpy",
        ),
        mask_shape_keys=("MOSAIK_proseg",),
    ),
    CellTypeSource(
        key="cellpose",
        label="Cellpose",
        clustering_tables=("table_MOSAIK_cellpose_clustering_squidpy",),
        mask_shape_keys=("MOSAIK_cellpose",),
    ),
    CellTypeSource(
        key="original",
        label="Original segmentation",
        clustering_tables=("table_original_clustering_squidpy",),
        mask_shape_keys=("merscope_cell_boundaries", "cell_boundaries", "atera_cell_boundaries"),
    ),
    *(
        CellTypeSource(
            key=f"pciseq_{branch}",
            label=f"pciSeq · {label}",
            clustering_tables=(),
            mask_shape_keys=shapes,
            method="pciseq",
            segmentation=branch,
        )
        for branch, label, shapes in (
            ("proseg_hybrid", "ProSeg hybrid", ("MOSAIK_proseg_hybrid",)),
            ("proseg", "ProSeg", ("MOSAIK_proseg",)),
            ("cellpose", "Cellpose", ("MOSAIK_cellpose",)),
            ("original", "Original segmentation", (
                "merscope_cell_boundaries", "xenium_cell_boundaries", "cell_boundaries",
            )),
        )
    ),
)
_CELL_TYPE_SOURCE_BY_KEY = {source.key: source for source in CELL_TYPE_SOURCES}


def cell_type_source(key: Any) -> CellTypeSource | None:
    """Return the :class:`CellTypeSource` for ``key`` (``None`` if unknown)."""
    return _CELL_TYPE_SOURCE_BY_KEY.get(str(key))


def _table_region_matches(table, shape_key: str, instance_key: str) -> bool:
    """Validate a table's declared shape/instance link without expression reads."""
    from anndata.io import read_elem

    attrs = read_elem(table["uns"]["spatialdata_attrs"])
    region = attrs.get("region")
    regions = [region] if isinstance(region, str) else list(region) if region is not None else []
    return (
        regions == [shape_key]
        and attrs.get("instance_key") == instance_key
        and instance_key in table["obs"]
    )


def _analysis_marker_matches(element, analysis_id: str, *, table: bool = False) -> bool:
    """Reject a mixed generation while a registered source is being refreshed."""
    marker_key = "merxen_pciseq_viewer"
    if table:
        from anndata.io import read_elem

        marker = read_elem(element["uns"][marker_key])
    else:
        # SpatialData only restores standard feature/instance points attrs.
        # Publication markers therefore live on the standard Zarr group itself.
        marker = element.attrs.get(marker_key)
    return isinstance(marker, Mapping) and marker.get("analysis_id") == analysis_id


def analysis_source_metadata(zarr_path: Any, source_key: str) -> dict[str, Any] | None:
    """Read a registered pciSeq source, failing closed on an unsafe mask join.

    The registry must identify a native segmentation, existing points and a
    table linked to exactly that segmentation. Only metadata is read; points,
    polygons, expression and embeddings remain lazy.
    """
    source = cell_type_source(source_key)
    if source is None or source.method != "pciseq":
        return None
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        schema = root.attrs.get("merxen_schema", {})
        if not isinstance(schema, Mapping):
            return None
        registry = schema.get("analysis_sources", {})
        segmentations = schema.get("segmentations", {})
        if not isinstance(registry, Mapping) or not isinstance(segmentations, Mapping):
            return None
        entry = registry.get(str(source_key))
        native = segmentations.get(source.segmentation)
        if not isinstance(entry, Mapping) or not isinstance(native, Mapping):
            return None
        required = (
            "shape", "table", "embedding_table", "points", "source_points",
            "assignment_column", "background_column", "status_column",
            "instance_key", "analysis_id",
        )
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in required):
            return None
        if (
            entry.get("method") != "pciseq"
            or entry.get("segmentation") != source.segmentation
            or entry["shape"] != native.get("shape")
            or entry["source_points"] != native.get("points", schema.get("primary_points"))
            or entry["points"] == entry["source_points"]
            or entry["table"] == entry["embedding_table"]
            or re.fullmatch(r"[0-9a-f]{64}", entry["analysis_id"]) is None
        ):
            return None
        if (
            entry["shape"] not in root["shapes"]
            or entry["points"] not in root["points"]
            or entry["source_points"] not in root["points"]
            or entry["table"] not in root["tables"]
        ):
            return None
        table = root["tables"][entry["table"]]
        if (
            "pciseq_cell_type" not in table["obs"]
            or not _table_region_matches(table, entry["shape"], entry["instance_key"])
            or not _analysis_marker_matches(table, entry["analysis_id"], table=True)
            or not _analysis_marker_matches(root["points"][entry["points"]], entry["analysis_id"])
        ):
            return None
        embedding_key = entry["embedding_table"]
        if embedding_key in root["tables"]:
            embedding_shape = entry.get("embedding_shape", entry["shape"])
            registered_variant = any(
                isinstance(variant, Mapping)
                and variant.get("shape") == embedding_shape
                and variant.get("coordinate_variant_of") == source.segmentation
                for variant in segmentations.values()
            )
            legacy_variant = (
                "embedding_shape" in entry
                and embedding_shape == f"{entry['shape']}_aligned_nonrigid"
            )
            if (
                not isinstance(embedding_shape, str)
                or embedding_shape not in root["shapes"]
                or not (embedding_shape == entry["shape"] or registered_variant or legacy_variant)
            ):
                return None
            embedding = root["tables"][embedding_key]
            if not (
                _table_region_matches(embedding, embedding_shape, entry["instance_key"])
                and _analysis_marker_matches(embedding, entry["analysis_id"], table=True)
            ):
                return None
        return dict(entry)
    except Exception:
        log.debug("Cannot validate analysis source %s", source_key, exc_info=True)
        return None


def assignment_source_points(zarr_path: Any, source_key: str) -> dict[str, Any] | None:
    """Return the validated points/assignment metadata for a pciSeq source."""
    return analysis_source_metadata(zarr_path, source_key)


def cell_type_shape_key_for_source(zarr_path: Any, source_key: str) -> str | None:
    """Resolve a source's exact native shape key from its registry or candidates."""
    source = cell_type_source(source_key)
    if source is None:
        return None
    if source.method == "pciseq":
        metadata = analysis_source_metadata(zarr_path, source_key)
        return metadata["shape"] if metadata else None
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        shapes = root["shapes"]
        # Exact candidates only: a merxen_schema fallback would resolve
        # 'original' to xenium_cell_boundaries on MerXen-native Xenium stores.
        return next((key for key in source.mask_shape_keys if key in shapes), None)
    except Exception:
        return None


def cell_type_table_key_for_source(
    zarr_path: Any, source_key: str, *, embedding: bool = False,
) -> str | None:
    """Resolve annotations or the separate QC-retained embedding for a source."""
    source = cell_type_source(source_key)
    if source is not None and source.method == "pciseq":
        metadata = analysis_source_metadata(zarr_path, source_key)
        if metadata is None:
            return None
        key = metadata["embedding_table" if embedding else "table"]
        root = open_zarr_group_unconsolidated(zarr_path)
        return key if key in root["tables"] else None
    return clustering_table_key_for_segmentation(zarr_path, source_key)


def assignment_points_in_display_frame(
    assignment_points,
    display_points,
    *,
    assignment_gene_col: str | None = None,
    display_gene_col: str | None = None,
    transcript_id_col: str = "transcript_id",
    assignment_col: str = "assignment",
    background_col: str = "background",
    status_col: str = "pciseq_assignment_status",
    coordinate_cols: Iterable[str] | None = None,
):
    """Join pciSeq assignments onto exact-ID aligned display coordinates.

    Both inputs must contain the same unique uint64 transcript ids and
    gene names. Coordinates and gene spelling come only from ``display_points``;
    assignments, background and model status come only from ``assignment_points``.
    Native measured coordinates are never modified or approximately transformed.

    Dask inputs remain lazy. Matching hash shuffles colocate every transcript id
    before a partition-wise one-to-one outer join, so duplicate ids spanning
    partitions, missing ids and conflicting genes fail when the renderer reads
    the data. No full transcript table is collected into pandas.
    """
    assignment_gene_col = assignment_gene_col or resolve_gene_column(assignment_points)
    display_gene_col = display_gene_col or resolve_gene_column(display_points)
    coordinates = list(coordinate_cols) if coordinate_cols is not None else [
        axis for axis in ("x", "y", "z") if axis in display_points.columns
    ]
    if not coordinates or assignment_gene_col is None or display_gene_col is None:
        raise ValueError("Assignment/display points require coordinates and gene columns")
    display_columns = [*coordinates, display_gene_col, transcript_id_col]
    assignment_columns = [
        transcript_id_col, assignment_gene_col, assignment_col, background_col, status_col,
    ]
    if not set(display_columns).issubset(display_points.columns):
        raise ValueError("Display points lack coordinates, genes or transcript_id")
    if not set(assignment_columns).issubset(assignment_points.columns):
        raise ValueError("Assignment points lack exact transcript IDs or pciSeq fields")
    internal_gene = "__viewer_assignment_gene"
    internal_fields = {
        assignment_col: "__viewer_assignment",
        background_col: "__viewer_background",
        status_col: "__viewer_status",
    }
    output_columns = [*display_columns, assignment_col, background_col, status_col]
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("Display coordinate/gene columns overlap assignment columns")

    def project(part, *, assignment):
        columns = assignment_columns if assignment else display_columns
        projected = part[columns].copy()
        ids, valid = cell_ids_as_int64(projected[transcript_id_col])
        if not valid.all() or (ids < 0).any():
            raise ValueError("Display/assignment transcript_id must be exact uint64 integers")
        projected[transcript_id_col] = ids.astype(np.uint64, copy=False)
        gene_col = assignment_gene_col if assignment else display_gene_col
        if projected[gene_col].isna().any():
            raise ValueError("Display/assignment transcript genes cannot be missing")
        if assignment:
            projected = projected.rename(columns={assignment_gene_col: internal_gene, **internal_fields})
        return projected

    def join(display, assignments):
        try:
            joined = display.merge(
                assignments, on=transcript_id_col, how="outer",
                validate="one_to_one", indicator="__viewer_match",
            )
        except pd.errors.MergeError as error:
            raise ValueError("Display/assignment points contain duplicate transcript IDs") from error
        if not joined["__viewer_match"].eq("both").all():
            raise ValueError("Display and assignment transcript IDs differ: missing transcripts")
        if not np.array_equal(
            joined[display_gene_col].astype(str).to_numpy(),
            joined[internal_gene].astype(str).to_numpy(),
        ):
            raise ValueError("Display and assignment transcript IDs resolve to different genes")
        return joined.rename(columns={value: key for key, value in internal_fields.items()})[output_columns]

    if isinstance(assignment_points, pd.DataFrame) and isinstance(display_points, pd.DataFrame):
        result = join(project(display_points, assignment=False), project(assignment_points, assignment=True))
    else:
        import dask.dataframe as dd

        assignments = (
            dd.from_pandas(assignment_points, npartitions=1, sort=False)
            if isinstance(assignment_points, pd.DataFrame) else assignment_points
        )
        display = (
            dd.from_pandas(display_points, npartitions=1, sort=False)
            if isinstance(display_points, pd.DataFrame) else display_points
        )
        assignment_meta = project(assignments._meta, assignment=True)
        display_meta = project(display._meta, assignment=False)
        assignments = assignments.map_partitions(project, assignment=True, meta=assignment_meta)
        display = display.map_partitions(project, assignment=False, meta=display_meta)
        partitions = max(assignments.npartitions, display.npartitions)
        shuffle = dict(
            on=transcript_id_col, npartitions=partitions, shuffle_method="disk",
            ignore_index=True, force=True,
        )
        assignments, display = assignments.shuffle(**shuffle), display.shuffle(**shuffle)
        result = dd.map_partitions(
            join, display, assignments, meta=join(display_meta, assignment_meta),
            align_dataframes=False, clear_divisions=True,
        )
    # Plain Dask has no attrs; SpatialData installs a special accessor which
    # must be updated in place rather than replaced by an ordinary dictionary.
    result_attrs = getattr(result, "attrs", None)
    if result_attrs is not None:
        result_attrs.update(dict(getattr(display_points, "attrs", {})))
    return result


@dataclass
class CellTypeAssignments:
    """Per-cell broad/fine cell-type labels joined to a segmentation mask.

    ``cell_ids`` are SpatialData instance-key values matching mask pixel IDs.
    ``levels`` holds the available, row-aligned legacy and Allen labels; missing
    labels remain None. ``broad``/``fine`` retain the cell inspector's summary.
    Cells missing every label are dropped; unannotated mask IDs stay transparent.
    """

    segmentation: str
    table_key: str
    instance_key: str
    cell_ids: np.ndarray            # exact signed/unsigned integers
    broad: np.ndarray               # str (object)
    fine: np.ndarray                # str (object)

    levels: dict[str, np.ndarray] = field(default_factory=dict)
    probabilities: np.ndarray | None = None
    statuses: np.ndarray | None = None
    shape_key: str | None = None
    analysis_id: str | None = None

    def labels_for(self, kind: str) -> np.ndarray:
        """Return labels at a stored level; absent levels remain unannotated."""
        if self.levels:
            return self.levels.get(str(kind), np.full(len(self.cell_ids), None, dtype=object))
        if kind == "broad":
            return self.broad
        if kind == "fine":
            return self.fine
        return np.full(len(self.cell_ids), None, dtype=object)


def clustering_table_key_for_segmentation(zarr_path: Any, segmentation: str) -> str | None:
    """Resolve one source's annotation table, including named map_first runs.

    MerXen can retain legacy tables beside ``_mapfirst`` or user-named runs
    such as ``_m8``. Prefer an unsuffixed reference-annotated table, then the
    conventional ``_mapfirst`` run, then other reference-annotated suffixes in
    lexical order. Fall back to the source's original unsuffixed table when
    no reference-annotated run exists. Both labels and UMAP use this resolver
    so their rows always come from the same run.

    Candidates must match a complete source-specific clustering-table prefix;
    ProSeg, ProSeg hybrid and ProSeg in-mask tables never cross-match.
    """
    source = cell_type_source(segmentation)
    if source is None:
        return None
    if source.method == "pciseq":
        return cell_type_table_key_for_source(zarr_path, segmentation)
    try:
        root = open_zarr_group_unconsolidated(zarr_path)
        if "tables" not in root:
            return None
        tables = root["tables"]
        present = set(tables.group_keys())
        annotated = []
        for base_index, base in enumerate(source.clustering_tables):
            for key in present:
                if key != base and not key.startswith(base + "_"):
                    continue
                try:
                    obs = tables[key]["obs"]
                    columns = set(obs.attrs.get("column-order", []))
                    # Check the stored fields, not just a suffix that could name
                    # a legacy or incomplete run. Do not read expression data.
                    has_labels = any(
                        f"{level}_name" in columns and f"{level}_name" in obs
                        for level in CELL_TYPE_LEVEL_LABELS
                        if level.startswith("ct_")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if has_labels:
                    rank = 0 if key == base else 1 if key == base + "_mapfirst" else 2
                    annotated.append((rank, base_index, key))
        if annotated:
            return min(annotated)[2]
        return next((key for key in source.clustering_tables if key in present), None)
    except Exception:
        log.debug("Cannot resolve clustering table for %s", segmentation, exc_info=True)
        return None


def _resolve_instance_key(table) -> str:
    """Pick the obs column whose values are the segmentation-mask label ids."""
    uns = getattr(table, "uns", None)
    obs = getattr(table, "obs", None)
    columns = list(getattr(obs, "columns", []))
    if isinstance(uns, Mapping):
        attrs = uns.get("spatialdata_attrs")
        key = attrs.get("instance_key") if isinstance(attrs, Mapping) else None
        if isinstance(key, str) and key in columns:
            return key
    for candidate in _INSTANCE_KEY_CANDIDATES:
        if candidate in columns:
            return candidate
    return ""


# MerXen annotation/schema.py and clustering/map_first.py define these columns.
# Keep resolved assignments distinct from the legacy branch/leaf clustering.
CELL_TYPE_LEVEL_LABELS = {
    "broad": "Broad cell type",
    "fine": "Fine cell type",
    "ct_final": "Allen · final assignment",
    "ct_lineage": "Allen · lineage",
    "ct_broad": "Allen · broad type",
    "ct_class": "Allen · class",
    "ct_nt": "Allen · neurotransmitter",
    "ct_supercluster": "Allen · supercluster",
    "ct_subclass": "Allen · subclass",
    "ct_seaad_subclass": "SEA-AD · subclass",
    "ct_cluster": "Allen · cluster (report only)",
    "ct_supertype": "Allen · supertype (report only)",
    "pciseq": "pciSeq · cell type",
}


def cell_id_to_int(value: Any) -> int | None:
    """Return an exact integer cell id, or ``None`` when ``value`` is not one.

    Integer and decimal strings never pass through ``float``. Already-rounded
    floating point ids beyond its exact integer range are rejected.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return int(as_float) if (
            np.isfinite(as_float) and as_float.is_integer() and abs(as_float) <= 2**53
        ) else None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    return int(decimal) if decimal.is_finite() and decimal == decimal.to_integral_value() else None


def cell_ids_as_int64(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return exact integer ids and validity, preserving the full uint64 range.

    The historical name is retained for callers. Small ids use int64; unsigned
    ids above its maximum use uint64. A mixture of negative and large unsigned
    ids uses Python integers rather than overflowing either numpy dtype.

    The cell-type and UMAP joins that go through this function are only
    exercised with ids below ``2**63`` (Atera ids are 1..91,431); the full
    uint64 exactness guarantee for transcript assignments is scoped to
    :class:`CellTranscriptIndex` and the lazy Atera engine.
    """
    arr = np.asarray(values, dtype=object)
    parsed = [0] * len(arr)
    valid = np.zeros(len(arr), dtype=bool)
    low, high = np.iinfo(np.int64).min, np.iinfo(np.uint64).max
    for i, value in enumerate(arr):
        as_int = cell_id_to_int(value)
        if as_int is not None and low <= as_int <= high:
            parsed[i] = as_int
            valid[i] = True
    minimum, maximum = min(parsed, default=0), max(parsed, default=0)
    dtype = np.int64 if maximum <= np.iinfo(np.int64).max else np.uint64 if minimum >= 0 else object
    return np.asarray(parsed, dtype=dtype), valid


def _clean_cell_labels(values) -> np.ndarray:
    text = pd.Series(np.asarray(values), dtype=object).astype("string").str.strip()
    text = text.mask(text.str.lower().isin(_MISSING_LABEL_TEXTS))
    return text.to_numpy(dtype=object, na_value=None)


def _read_cell_type_obs(zarr_path: Any, segmentation: str):
    """Read annotation metadata only, without expression or neighbour matrices."""
    source = cell_type_source(segmentation)
    metadata = None
    if source is not None and source.method == "pciseq":
        metadata = analysis_source_metadata(zarr_path, segmentation)
        if metadata is None:
            return None
    table_key = metadata["table"] if metadata else clustering_table_key_for_segmentation(zarr_path, segmentation)
    if table_key is None:
        return None
    from anndata.io import read_elem
    from types import SimpleNamespace

    table = open_zarr_group_unconsolidated(zarr_path)[f"tables/{table_key}"]
    obs_group = table["obs"]
    columns = list(obs_group.attrs.get("column-order", []))
    attrs = {}
    if "uns" in table and "spatialdata_attrs" in table["uns"]:
        attrs = dict(read_elem(table["uns"]["spatialdata_attrs"]))
    instance_key = _resolve_instance_key(SimpleNamespace(
        obs=SimpleNamespace(columns=columns), uns={"spatialdata_attrs": attrs}
    ))
    wanted = {CELL_TYPE_BROAD_CLASS_COL, CELL_TYPE_FINE_CLUSTER_COL,
              CELL_TYPE_NEURON_SPLIT_COL, instance_key}
    wanted.update(f"{key}_name" for key in CELL_TYPE_LEVEL_LABELS if key.startswith("ct_"))
    if source is not None and source.method == "pciseq":
        wanted = {instance_key, "pciseq_cell_type", "pciseq_probability", "pciseq_status"}
    index_key = str(obs_group.attrs.get("_index", "_index"))
    obs = pd.DataFrame(index=np.asarray(read_elem(obs_group[index_key])))
    for column in columns:
        if column in wanted:
            obs[column] = read_elem(obs_group[column])
    if metadata is not None:
        current = analysis_source_metadata(zarr_path, segmentation)
        if current is None or current["analysis_id"] != metadata["analysis_id"]:
            return None
    return table_key, obs, instance_key, metadata


def _cell_label_levels(obs) -> dict[str, np.ndarray]:
    levels = {}
    if "pciseq_cell_type" in obs:
        # Reference class names are identities: strings such as 'NA' or 'None'
        # can be legitimate custom classes, unlike legacy missing-label text.
        labels = obs["pciseq_cell_type"].to_numpy(dtype=object, copy=True)
        labels[pd.isna(labels) | (labels == "")] = None
        if "pciseq_status" in obs:
            labels[obs["pciseq_status"].eq("excluded_overlap").to_numpy()] = None
        if any(value is not None for value in labels):
            levels["pciseq"] = labels
        return levels
    if CELL_TYPE_BROAD_CLASS_COL in obs:
        broad = _clean_cell_labels(obs[CELL_TYPE_BROAD_CLASS_COL])
        if CELL_TYPE_NEURON_SPLIT_COL in obs:
            split = obs[CELL_TYPE_NEURON_SPLIT_COL].astype(str).str.lower().map(_NEURON_SPLIT_LABELS)
            for i, label in enumerate(broad):
                if label and label.lower() == NEURON_BROAD_CLASS.lower() and pd.notna(split.iloc[i]):
                    broad[i] = split.iloc[i]
        if any(value is not None for value in broad):
            levels["broad"] = broad
            fine = (_clean_cell_labels(obs[CELL_TYPE_FINE_CLUSTER_COL])
                    if CELL_TYPE_FINE_CLUSTER_COL in obs else broad.copy())
            levels["fine"] = np.array([
                (f if f is not None else b) if b is not None else None
                for b, f in zip(broad, fine)
            ], dtype=object)
    for kind in CELL_TYPE_LEVEL_LABELS:
        column = f"{kind}_name"
        if kind.startswith("ct_") and column in obs:
            labels = _clean_cell_labels(obs[column])
            if any(value is not None for value in labels):
                levels[kind] = labels
    return levels


def cell_type_levels(zarr_path: Any, segmentation: str) -> dict[str, str]:
    """List usable stored levels; missing, empty or unreadable data offers none."""
    try:
        result = _read_cell_type_obs(zarr_path, segmentation)
        if result is None:
            return {}
        levels = _cell_label_levels(result[1])
        return {key: CELL_TYPE_LEVEL_LABELS[key] for key in levels}
    except Exception:
        log.debug("Cannot inspect cell-type levels for %s", segmentation, exc_info=True)
        return {}


def load_cell_type_assignments(zarr_path: Any, segmentation: str = "proseg") -> CellTypeAssignments | None:
    """Load legacy and resolved Allen hierarchy labels joined by exact mask IDs.

    Nullable levels stay missing at those cells; no finer assignment is inferred
    from a coarser level. Stored candidate names retain their upstream meaning.
    """
    try:
        result = _read_cell_type_obs(zarr_path, segmentation)
        if result is None:
            return None
        table_key, obs, instance_key, metadata = result
        levels = _cell_label_levels(obs)
        if not levels:
            return None
        # Share the exact-integer join used by UMAP, including 19-digit IDs.
        ids, valid = cell_ids_as_int64(obs[instance_key] if instance_key else obs.index)
        valid &= np.logical_or.reduce([pd.notna(values) for values in levels.values()])
        if not valid.any():
            return None
        levels = {key: values[valid] for key, values in levels.items()}
        broad = levels.get("broad", levels.get("ct_broad", next(iter(levels.values()))))
        fine = levels.get("fine", levels.get("ct_final", broad))
        return CellTypeAssignments(
            segmentation=str(segmentation), table_key=table_key,
            instance_key=instance_key, cell_ids=ids[valid],
            broad=broad, fine=fine, levels=levels,
            probabilities=(
                obs["pciseq_probability"].to_numpy(dtype=float)[valid]
                if "pciseq_probability" in obs else None
            ),
            statuses=(obs["pciseq_status"].to_numpy()[valid] if "pciseq_status" in obs else None),
            shape_key=metadata["shape"] if metadata else cell_type_shape_key_for_source(zarr_path, segmentation),
            analysis_id=metadata["analysis_id"] if metadata else None,
        )
    except Exception:
        log.debug("Cannot load cell-type assignments for %s", segmentation, exc_info=True)
        return None


@dataclass
class CellTypeColorScheme:
    """Colours + tick grouping for one cell-type level (broad or fine).

    ``colors`` maps every label to its RGBA. ``order`` is the flat, ordered list
    of tickable labels. ``groups`` is the ordered ``(header, [members])`` the
    panel renders: for the fine level each broad class is a header over its fine
    subtypes; for the broad level a single header holds all broad classes.
    """

    kind: str
    colors: dict[str, tuple[float, float, float, float]]
    order: list[str]
    groups: list[tuple[str, list[str]]]


def build_cell_type_color_schemes(
    broad_per_cell: Iterable[str],
    fine_per_cell: Iterable[str],
    alpha: float = 1.0,
) -> dict[str, CellTypeColorScheme]:
    """Build the ``"broad"`` and ``"fine"`` colour schemes from per-cell labels.

    Both levels draw from the same maximally-distinct glasbey palette the
    transcripts use: each broad class gets its own colour, and each fine subtype
    gets its own distinct colour too (not a shade of its broad class). Fine
    subtypes are still *grouped* under their broad class in the panel; the
    fine->broad grouping is taken from the aligned per-cell pairing (no string
    parsing).
    """
    broad_arr = [str(b).strip() for b in broad_per_cell]
    fine_arr = [str(f).strip() for f in fine_per_cell]

    fines_by_broad: dict[str, list[str]] = {}
    for broad, fine in zip(broad_arr, fine_arr):
        if broad.lower() in _MISSING_LABEL_TEXTS:
            continue
        members = fines_by_broad.setdefault(broad, [])
        if fine not in members:
            members.append(fine)

    broads = sorted(fines_by_broad)
    broad_palette = gene_palette_rgba(len(broads), alpha=alpha)
    broad_colors = {broad: broad_palette[i] for i, broad in enumerate(broads)}

    # Every fine subtype gets its own maximally-distinct colour. Order them by
    # their broad group (so the palette walk matches the panel order) but colour
    # each independently.
    fine_order: list[str] = [fine for broad in broads for fine in sorted(fines_by_broad[broad])]
    fine_palette = gene_palette_rgba(len(fine_order), alpha=alpha)
    fine_colors = {fine: fine_palette[i] for i, fine in enumerate(fine_order)}
    fine_groups = [(broad, sorted(fines_by_broad[broad])) for broad in broads]

    broad_scheme = CellTypeColorScheme(
        kind="broad",
        colors=broad_colors,
        order=broads,
        groups=[("Broad cell types", broads)] if broads else [],
    )
    fine_scheme = CellTypeColorScheme(
        kind="fine",
        colors=fine_colors,
        order=fine_order,
        groups=fine_groups,
    )
    return {"broad": broad_scheme, "fine": fine_scheme}


def build_assignment_color_schemes(
    assignments: Iterable[CellTypeAssignments],
) -> dict[str, CellTypeColorScheme]:
    """Build matching palettes over one or multiple datasets' stored levels."""
    items = list(assignments)
    if not items:
        return {}
    schemes = build_cell_type_color_schemes(
        np.concatenate([item.broad for item in items]),
        np.concatenate([item.fine for item in items]),
    )
    keys = set().union(*(
        item.levels or {"broad": item.broad, "fine": item.fine} for item in items
    ))
    schemes = {key: scheme for key, scheme in schemes.items() if key in keys}
    for key in CELL_TYPE_LEVEL_LABELS:
        if key not in keys or key in ("broad", "fine"):
            continue
        labels = sorted({
            str(label) for item in items for label in item.labels_for(key)
            if label is not None and (
                str(label) != "" if key == "pciseq"
                else str(label).lower() not in _MISSING_LABEL_TEXTS
            )
        })
        palette = gene_palette_rgba(len(labels), alpha=1.0)
        schemes[key] = CellTypeColorScheme(
            kind=key, colors=dict(zip(labels, palette)), order=labels,
            groups=[(CELL_TYPE_LEVEL_LABELS[key], labels)] if labels else [],
        )
    return schemes


def build_cell_type_color_dict(
    cell_ids: np.ndarray,
    labels: Iterable[str],
    scheme: CellTypeColorScheme,
    enabled: Iterable[str] | None = None,
) -> dict[int | None, tuple[float, float, float, float]]:
    """Map mask label ids to RGBA for a :class:`DirectLabelColormap`.

    ``cell_ids`` and ``labels`` are aligned per-cell arrays. Background (``0``)
    and unmapped ids stay transparent; a cell whose type is not in ``enabled``
    (when given) is omitted, so unticking a type hides those masks.
    """
    enabled_set = None if enabled is None else {str(e) for e in enabled}
    color_dict: dict[int | None, tuple[float, float, float, float]] = {
        0: TRANSPARENT_RGBA,
        None: TRANSPARENT_RGBA,
    }
    colors = scheme.colors
    for cid, label in zip(np.asarray(cell_ids).tolist(), labels):
        text = str(label)
        if enabled_set is not None and text not in enabled_set:
            continue
        rgba = colors.get(text)
        if rgba is None:
            continue
        color_dict[int(cid)] = rgba
    return color_dict


@dataclass
class CellTranscriptIndex:
    """Compact assigned-transcript index grouped by normalized cell id.

    Gene names are dictionary encoded once in ``gene_names`` and represented by
    the smallest practical unsigned integer dtype in ``gene_codes``.  This keeps
    the always-resident index bounded to roughly ten bytes per assigned
    transcript (two float32 coordinates plus a compact code), rather than an
    object-string array per transcript.
    """

    coords_yx: np.ndarray
    gene_codes: np.ndarray
    gene_names: tuple[str, ...]
    slices: dict[str, tuple[int, int]]
    #: ``gene_names`` as one object array, built on first use and reused by
    #: every :meth:`transcripts_for` call (never serialised: the transcript
    #: cache stores the fields above as ``.npy``).
    _gene_name_array: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def empty(cls) -> "CellTranscriptIndex":
        return cls(
            coords_yx=np.empty((0, 2), dtype=np.float32),
            gene_codes=np.empty((0,), dtype=np.uint8),
            gene_names=(),
            slices={},
        )

    def transcripts_for(self, cell_id):
        """Return ``(coords_yx, genes)`` for one cell (empty arrays if absent)."""
        span = self.slices.get(normalize_cell_key(cell_id))
        if span is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=object),
            )
        start, end = span
        names = self._gene_name_array
        if names is None or len(names) != len(self.gene_names):
            names = np.asarray(self.gene_names, dtype=object)
            self._gene_name_array = names
        return self.coords_yx[start:end], names[self.gene_codes[start:end]]


class TranscriptBuildCancelled(RuntimeError):
    """Raised when a cooperative transcript build cancellation is requested."""


@dataclass
class GenePointStore:
    """Backing store for the per-gene transcript renderer.

    Points are grouped by marker symbol into ``group_coords`` / ``group_colors``
    (one entry per napari Points layer). Within a group, each gene's points form
    one contiguous run, foreground (in-cell) points first then background points,
    so toggling a gene or hiding background is a cheap contiguous-range gather.
    """

    group_symbols: list[str]
    group_coords: list[np.ndarray]           # each (M, 2) float32, napari (y, x)
    group_colors: list[np.ndarray]           # each (M, 4) float32 RGBA
    # gene -> (group_index, fg_start, fg_end, bg_end) relative to that group.
    gene_offsets: dict[str, tuple[int, int, int, int]]
    gene_counts: dict[str, int]              # rendered total (fg + bg) points per gene
    genes: list[str]                         # alphabetical (real genes + controls)
    control_genes: set[str]
    total_points: int                        # rendered point count
    sampled: bool = False
    source_gene_counts: dict[str, int] | None = None
    source_total_points: int | None = None
    #: Exact pre-sampling assignment status counts, including unmodelled spots.
    source_status_counts: dict[str, int] | None = None
    #: The (colour, symbol) scheme the points were coloured with. Symbols are
    #: fixed at build time (they decide the layer grouping); only colours change
    #: when the inspector switches between the broad and fine cell-type schemes.
    gene_visuals: dict[str, GeneVisual] | None = None
    #: Compact all-assigned-transcript index built during the same streaming
    #: passes as the renderer.  It is intentionally independent of render
    #: sampling so cell summaries remain exact when the point view is capped.
    cell_transcript_index: CellTranscriptIndex | None = None

    def gene_symbol(self, gene: str) -> str | None:
        entry = self.gene_offsets.get(str(gene))
        return None if entry is None else self.group_symbols[entry[0]]

    def recolor(self, gene_visuals: Mapping[str, GeneVisual]) -> None:
        """Rewrite per-point colours in place from a new visual scheme.

        Each gene's points are one contiguous run within its group array, so this
        is a cheap per-gene fill -- no re-read and no regrouping. The new scheme
        must assign the SAME symbol per gene as the current one (else the layer
        grouping would be stale); callers guarantee this by deriving symbols only
        from the broad grouping.
        """
        for gene, (gi, fg_start, _fg_end, bg_end) in self.gene_offsets.items():
            visual = gene_visuals.get(gene)
            if visual is None or gi >= len(self.group_colors):
                continue
            self.group_colors[gi][fg_start:bg_end] = np.asarray(visual.rgba, dtype=np.float32)
        self.gene_visuals = dict(gene_visuals)

    def full_gene_count(self, gene: str) -> int:
        """Return the source-dataset count for ``gene`` when available."""
        counts = self.source_gene_counts if self.source_gene_counts is not None else self.gene_counts
        return int(counts.get(str(gene), 0))


def _empty_gene_point_store() -> GenePointStore:
    return GenePointStore(
        group_symbols=[],
        group_coords=[],
        group_colors=[],
        gene_offsets={},
        gene_counts={},
        genes=[],
        control_genes=set(),
        total_points=0,
        sampled=False,
        source_gene_counts={},
        source_total_points=0,
        cell_transcript_index=None,
    )


def _unsigned_code_dtype(category_count: int):
    """Smallest unsigned dtype capable of representing ``category_count`` ids."""
    maximum = max(0, int(category_count) - 1)
    if maximum <= np.iinfo(np.uint8).max:
        return np.uint8
    if maximum <= np.iinfo(np.uint16).max:
        return np.uint16
    if maximum <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _check_transcript_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise TranscriptBuildCancelled("Transcript build cancelled")


#: numpy's ``Generator.hypergeometric`` refuses ``ngood`` or ``nbad`` at or
#: above this population size.
_HYPERGEOMETRIC_LIMIT = 10**9


def _exact_factorize(series: pd.Series) -> tuple[np.ndarray, pd.Series] | None:
    """Dense codes for ``series`` whose uniques map exactly like its rows.

    Returns ``(codes, uniques)``: ``codes`` is int64 with ``-1`` for missing
    values and ``uniques`` is a Series of the column's own dtype, so any
    elementwise conversion (``astype("string")``, :func:`normalize_cell_key`)
    applied to ``uniques`` and gathered by ``codes`` equals that conversion
    applied row by row.  That holds only where values that hash equal also
    convert equal, so mixed object columns (``1`` vs ``True``), decimals and
    other unusual dtypes return ``None`` and the caller converts per row.
    Floats are factorised on their bit patterns, keeping ``0.0`` and ``-0.0``
    apart.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        codes = series.cat.codes.to_numpy(dtype=np.int64)
        return codes, pd.Series(dtype.categories)
    if isinstance(dtype, np.dtype) and dtype.kind == "f":
        values = series.to_numpy()
        missing = np.isnan(values)
        bits = values.view(np.int64 if values.dtype.itemsize == 8 else np.int32 if values.dtype.itemsize == 4 else np.int16)
        codes = np.full(len(values), -1, dtype=np.int64)
        present = np.flatnonzero(~missing)
        if len(present):
            sub_codes, first = _first_occurrence_codes(bits[present])
            codes[present] = sub_codes
            uniques = values[present[first]]
        else:
            uniques = values[:0]
        return codes, pd.Series(uniques, dtype=dtype)
    exact = (
        isinstance(dtype, pd.StringDtype)
        or (isinstance(dtype, np.dtype) and dtype.kind in "iub")
        or (isinstance(dtype, pd.api.extensions.ExtensionDtype)
            and getattr(dtype, "kind", "") in "iub" and not isinstance(dtype, np.dtype))
    )
    if not exact and isinstance(dtype, pd.ArrowDtype):
        exact = pd.api.types.is_string_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
    if not exact and dtype == object:
        exact = pd.api.types.infer_dtype(series, skipna=True) in {"string", "integer", "empty"}
    if not exact:
        return None
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int64), pd.Series(uniques, dtype=dtype)


def _first_occurrence_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Codes in first-occurrence order and each unique's first row."""
    codes, uniques = pd.factorize(values)
    first = np.full(len(uniques), len(values), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(values), dtype=np.int64))
    return np.asarray(codes, dtype=np.int64), first


@dataclass
class _PartitionArrays:
    """One partition's valid rows, with genes and cells as small local vocabularies."""

    x: np.ndarray                    # float32, valid rows
    y: np.ndarray                    # float32, valid rows
    gene_codes: np.ndarray           # int64 into ``gene_vocab``
    gene_vocab: np.ndarray           # sorted unique ``str`` gene names (numpy U)
    background: np.ndarray           # bool, valid rows
    #: int64 into ``cell_vocab`` for assigned rows with a non-empty key, else
    #: -1; ``None`` without an assignment column.
    cell_codes: np.ndarray | None
    cell_vocab: np.ndarray | None    # object array of normalised cell keys


def _sorted_string_vocab(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(codes, vocab)``: rows coded into the sorted ``str`` names (-1 = NA).

    ``vocab`` equals ``np.unique`` of ``series.astype("string")`` rows cast to
    ``str``, without converting every row.
    """
    factorized = _exact_factorize(series)
    if factorized is None:
        raw_codes, uniques = pd.factorize(series.astype("string"), use_na_sentinel=True)
        raw_codes = np.asarray(raw_codes, dtype=np.int64)
        uniques = pd.Series(uniques, dtype="string")
    else:
        raw_codes, uniques = factorized
    text = uniques.astype("string").to_numpy(dtype=object)
    if len(text) == 0:
        return raw_codes, np.zeros(0, dtype=str)
    vocab, lut = np.unique(text.astype(str), return_inverse=True)
    codes = np.where(raw_codes >= 0, lut.reshape(-1)[np.maximum(raw_codes, 0)], -1)
    return codes.astype(np.int64, copy=False), vocab


def _assignment_codes(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(codes, vocab, assigned)`` for an assignment column.

    ``assigned`` is :func:`assignment_mask` per row; ``codes`` index the
    normalised cell keys ``vocab`` for assigned rows whose
    :func:`normalize_cell_key` is non-empty, else -1.  Both are computed on the
    column's distinct values.
    """
    numeric = pd.api.types.is_numeric_dtype(series)
    factorized = _exact_factorize(series)
    if factorized is None:
        assigned = assignment_mask(series).to_numpy(dtype=bool, copy=False)
        values = series.to_numpy(dtype=object)
        keys = np.full(len(series), "", dtype=object)
        positions = np.flatnonzero(assigned)
        keys[positions] = [normalize_cell_key(value) for value in values[positions]]
        raw_codes, uniques = pd.factorize(keys)
        vocab = np.asarray(uniques, dtype=object)
        codes = np.asarray(raw_codes, dtype=np.int64)
        empty = np.flatnonzero(vocab == "")
        if len(empty):
            codes[codes == int(empty[0])] = -1
        return codes, vocab, assigned
    raw_codes, uniques = factorized
    if numeric:
        assigned_rows = assignment_mask(series).to_numpy(dtype=bool, copy=False)
        assigned_unique = None
    else:
        as_str = uniques.astype("string")
        bad_values = {"", "0", "-1", "nan", "None", "<NA>"}
        assigned_unique = (as_str.notna() & ~as_str.isin(bad_values)).to_numpy(dtype=bool)
        assigned_rows = None
    unique_values = uniques.to_numpy(dtype=object)
    keys = np.asarray([normalize_cell_key(value) for value in unique_values], dtype=object)
    if len(keys):
        key_codes, vocab = pd.factorize(keys)
        key_codes = np.asarray(key_codes, dtype=np.int64)
        vocab = np.asarray(vocab, dtype=object)
        empty = np.flatnonzero(vocab == "")
        if len(empty):
            key_codes[key_codes == int(empty[0])] = -1
        if assigned_unique is not None:
            key_codes[~assigned_unique] = -1
        codes = np.where(raw_codes >= 0, key_codes[np.maximum(raw_codes, 0)], -1)
    else:
        vocab = np.zeros(0, dtype=object)
        codes = np.full(len(series), -1, dtype=np.int64)
    if assigned_rows is not None:
        codes = np.where(assigned_rows, codes, -1)
    else:
        assigned_rows = np.zeros(len(raw_codes), dtype=bool)
        present = raw_codes >= 0
        assigned_rows[present] = assigned_unique[raw_codes[present]]
    return codes.astype(np.int64, copy=False), vocab, assigned_rows


def _status_unassigned(series: pd.Series) -> np.ndarray:
    """``series.astype("string").fillna("unknown") != "assigned"`` per row."""
    factorized = _exact_factorize(series)
    if factorized is None:
        return (series.astype("string").fillna("unknown") != "assigned").to_numpy(dtype=bool)
    codes, uniques = factorized
    other = (uniques.astype("string").fillna("unknown") != "assigned").to_numpy(dtype=bool)
    out = np.ones(len(codes), dtype=bool)  # missing -> "unknown" -> not assigned
    present = codes >= 0
    out[present] = other[codes[present]]
    return out


def _point_partition_arrays(
    pdf: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None,
    background_col: str | None,
    status_col: str | None = None,
    gene_coding: tuple[np.ndarray, np.ndarray] | None = None,
) -> _PartitionArrays | None:
    """Return compact valid arrays for one transcript partition.

    Genes and cell ids are converted once per distinct value, never per row.
    ``gene_coding`` reuses a ``_sorted_string_vocab`` of ``pdf[gene_col]``.
    """
    x_all = pdf[x_col].to_numpy(dtype=np.float32, copy=False)
    y_all = pdf[y_col].to_numpy(dtype=np.float32, copy=False)
    gene_codes_all, gene_vocab = (
        gene_coding if gene_coding is not None else _sorted_string_vocab(pdf[gene_col])
    )
    good = np.isfinite(x_all) & np.isfinite(y_all) & (gene_codes_all >= 0)
    if not np.any(good):
        return None

    has_assignment = assignment_col is not None and assignment_col in pdf.columns
    cell_codes_all = cell_vocab = assigned_all = None
    if has_assignment:
        cell_codes_all, cell_vocab, assigned_all = _assignment_codes(pdf[assignment_col])

    if background_col is not None and background_col in pdf.columns:
        background_all = pdf[background_col].to_numpy(dtype=bool, copy=False)
    elif has_assignment:
        background_all = ~assigned_all
    else:
        background_all = np.zeros(len(pdf), dtype=bool)
    if status_col is not None:
        # Rendering's unassigned group includes excluded transcripts. Their
        # scientific background flag remains false in the original points.
        background_all = background_all | _status_unassigned(pdf[status_col])

    return _PartitionArrays(
        x=np.ascontiguousarray(x_all[good]),
        y=np.ascontiguousarray(y_all[good]),
        gene_codes=gene_codes_all[good],
        gene_vocab=gene_vocab,
        background=np.ascontiguousarray(background_all[good]),
        cell_codes=None if cell_codes_all is None else cell_codes_all[good],
        cell_vocab=cell_vocab,
    )


class _StreamingSampler:
    """Exact uniform ``render_count``-of-``total`` sample, partition by partition.

    While numpy accepts every sequential hypergeometric draw (each partition's
    ``ngood`` and ``nbad`` below 1e9) this replays the historical sampler draw
    for draw, so the output is unchanged.  Otherwise (``total`` minus the first
    partition at or above 1e9, which made numpy raise) positions are drawn once
    over the whole stream and split by partition offsets.
    """

    def __init__(self, partition_counts: Sequence[int], render_count: int, random_state: int):
        counts = [int(c) for c in partition_counts]
        self.total = int(sum(counts))
        self.render_count = int(render_count)
        self.random_state = int(random_state)
        self.uses_hypergeometric = self.hypergeometric_accepts(counts)
        self._rng = np.random.default_rng(self.random_state)
        self._remaining_source = self.total
        self._remaining_sample = self.render_count
        self._offset = 0
        self._positions: np.ndarray | None = None
        if not self.uses_hypergeometric and 0 < self.render_count < self.total:
            from .atera_transcripts import sample_global_positions

            self._positions = sample_global_positions(self.total, self.render_count, self.random_state)

    @staticmethod
    def hypergeometric_accepts(partition_counts: Sequence[int]) -> bool:
        """Whether every sequential draw stays inside numpy's limits."""
        remaining = int(sum(int(c) for c in partition_counts))
        for count in partition_counts:
            count = int(count)
            if count >= _HYPERGEOMETRIC_LIMIT or remaining - count >= _HYPERGEOMETRIC_LIMIT:
                return False
            remaining -= count
        return True

    def take(self, part_count: int) -> np.ndarray:
        """Sorted kept row positions within the next partition of ``part_count`` rows."""
        part_count = int(part_count)
        if self._remaining_sample == self._remaining_source:
            keep = np.arange(part_count, dtype=np.int64)
        elif self._remaining_sample <= 0:
            keep = np.empty((0,), dtype=np.int64)
        elif self._positions is not None:
            lo = int(np.searchsorted(self._positions, self._offset, side="left"))
            hi = int(np.searchsorted(self._positions, self._offset + part_count, side="left"))
            keep = self._positions[lo:hi] - self._offset
        else:
            rng = self._rng
            take = int(
                rng.hypergeometric(
                    ngood=part_count,
                    nbad=max(0, self._remaining_source - part_count),
                    nsample=self._remaining_sample,
                )
            )
            keep = (
                np.sort(rng.choice(part_count, size=take, replace=False))
                if take > 0
                else np.empty((0,), dtype=np.int64)
            )
        self._offset += part_count
        self._remaining_source -= part_count
        self._remaining_sample -= int(len(keep))
        return keep


def _scatter_cell_rows(
    codes: np.ndarray,
    positions: np.ndarray,
    cell_write: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Destination slots for rows grouped by cell code, stable in row order.

    Advances ``cell_write`` (each cell's next free slot) in place and returns
    ``(dest, src)`` index arrays.
    """
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    src = positions[order]
    n = len(codes_sorted)
    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    dest = cell_write[codes_sorted] + rank
    cell_write[codes_sorted[starts]] += sizes
    return dest, src


def build_gene_point_groups(
    points_obj,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None = None,
    background_col: str | None = None,
    gene_visuals: Mapping[str, GeneVisual] | None = None,
    reference: Mapping[str, Any] | None = None,
    reference_resolver: Callable[[Iterable[str]], Mapping[str, Any] | None] | None = None,
    max_points: int | None = None,
    random_state: int = 42,
    alpha: float = 1.0,
    build_cell_index: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    status_col: str | None = None,
    is_gene_col: str | None = None,
) -> GenePointStore:
    """Read all transcripts into a symbol-grouped, gene-sorted point store.

    ``background`` is taken from ``background_col`` (bool) if present, else from
    ``assignment_col`` (unassigned == not :func:`assignment_mask`), else all
    foreground. If the total exceeds ``max_points`` a uniform random subsample is
    taken (this preserves per-gene proportions in expectation).

    ``is_gene_col`` (Atera's ``is_gene``) makes the store's ``control_genes`` the
    features flagged not-a-gene, cross-checked against the name rule by
    :func:`control_genes_from_is_gene`.  Without it controls are name-only.

    Both streaming passes are vectorised: genes and cell ids are converted
    once per distinct value per partition and mapped to global codes through
    small lookup tables, so no Python runs per transcript.
    """
    cols = [x_col, y_col, gene_col]
    if is_gene_col is not None and is_gene_col not in cols:
        cols.append(is_gene_col)
    is_gene_flags: dict[str, bool] | None = {} if is_gene_col is not None else None
    if background_col is not None:
        cols.append(background_col)
    if assignment_col is not None and assignment_col not in cols:
        cols.append(assignment_col)
    if status_col is not None and status_col not in cols:
        cols.append(status_col)

    def partition_arrays(pdf, gene_coding=None):
        return _point_partition_arrays(
            pdf,
            x_col=x_col,
            y_col=y_col,
            gene_col=gene_col,
            assignment_col=assignment_col,
            background_col=background_col,
            status_col=status_col,
            gene_coding=gene_coding,
        )

    # Pass 1 retains only small aggregate dictionaries.  In particular, it does
    # not keep a Python string object for every transcript and it lets us size
    # all output arrays exactly before the second streaming pass.
    source_gene_counts: dict[str, int] = {}
    source_status_counts: dict[str, int] | None = {} if status_col else None
    cell_counts: dict[str, int] = {}
    partition_counts: list[int] = []
    total_source = 0
    for pdf in _iter_point_partitions(points_obj, cols, cancel_check=cancel_check):
        _check_transcript_cancelled(cancel_check)
        gene_coding = _sorted_string_vocab(pdf[gene_col])
        if is_gene_flags is not None:
            # Raw rows, before coordinate filtering: the flag is per feature.
            codes_all, vocab = gene_coding
            flags = pd.Series(pdf[is_gene_col]).astype("boolean").fillna(False).to_numpy(dtype=bool)
            present = codes_all >= 0
            if np.any(present):
                gene_all = np.ones(len(vocab), dtype=bool)
                np.logical_and.at(gene_all, codes_all[present], flags[present])
                seen = np.bincount(codes_all[present], minlength=len(vocab)) > 0
                for name, flag in zip(vocab[seen].tolist(), gene_all[seen].tolist(), strict=True):
                    text = str(name)
                    is_gene_flags[text] = bool(is_gene_flags.get(text, True) and flag)
        if status_col is not None:
            # Status totals describe the complete source, independent of the
            # render cap and coordinate filtering used for the point layers.
            for status, count in pdf[status_col].value_counts(dropna=False).items():
                key = str(status) if pd.notna(status) else "unknown"
                source_status_counts[key] = source_status_counts.get(key, 0) + int(count)
        arrays = partition_arrays(pdf, gene_coding)
        if arrays is None:
            continue
        part_count = int(len(arrays.gene_codes))
        partition_counts.append(part_count)
        total_source += part_count
        counts = np.bincount(arrays.gene_codes, minlength=len(arrays.gene_vocab))
        present = np.flatnonzero(counts)
        for name, count in zip(arrays.gene_vocab[present].tolist(), counts[present].tolist(), strict=True):
            text = str(name)
            source_gene_counts[text] = source_gene_counts.get(text, 0) + int(count)

        if build_cell_index and arrays.cell_codes is not None:
            cell_local = arrays.cell_codes[arrays.cell_codes >= 0]
            if len(cell_local):
                local_counts = np.bincount(cell_local, minlength=len(arrays.cell_vocab))
                used = np.flatnonzero(local_counts)
                keys = arrays.cell_vocab[used]
                order = np.argsort(keys.astype(str), kind="stable")
                for key, count in zip(keys[order].tolist(), local_counts[used][order].tolist(), strict=True):
                    text = str(key)
                    cell_counts[text] = cell_counts.get(text, 0) + int(count)

    if total_source == 0:
        return _empty_gene_point_store()

    source_genes = sorted(source_gene_counts)
    if reference is None and reference_resolver is not None:
        reference = reference_resolver(source_genes)
    gene_to_code = {name: index for index, name in enumerate(source_genes)}
    gene_code_dtype = _unsigned_code_dtype(len(source_genes))

    render_count = total_source
    if max_points is not None and 0 < int(max_points) < total_source:
        render_count = int(max_points)
    sampled = render_count < total_source

    render_x = np.empty(render_count, dtype=np.float32)
    render_y = np.empty(render_count, dtype=np.float32)
    render_gene_codes = np.empty(render_count, dtype=gene_code_dtype)
    render_background = np.empty(render_count, dtype=bool)

    cell_index = CellTranscriptIndex.empty() if build_cell_index else None
    cell_keys = sorted(cell_counts)
    cell_code_by_key = {key: index for index, key in enumerate(cell_keys)}
    cell_offsets = np.zeros(len(cell_keys) + 1, dtype=np.int64)
    if build_cell_index and cell_keys:
        cell_offsets[1:] = np.cumsum(
            np.asarray([cell_counts[key] for key in cell_keys], dtype=np.int64)
        )
        cell_coords = np.empty((int(cell_offsets[-1]), 2), dtype=np.float32)
        cell_gene_codes = np.empty(int(cell_offsets[-1]), dtype=gene_code_dtype)
        cell_write = cell_offsets[:-1].copy()
    else:
        cell_coords = np.empty((0, 2), dtype=np.float32)
        cell_gene_codes = np.empty((0,), dtype=gene_code_dtype)
        cell_write = np.empty((0,), dtype=np.int64)

    # Pass 2 fills fixed-size outputs.  Sequential hypergeometric allocation
    # selects an exact uniform sample without ever materialising the uncapped
    # coordinate set in memory (a global position draw above numpy's limit).
    sampler = _StreamingSampler(partition_counts, render_count, random_state)
    render_cursor = 0
    for pdf in _iter_point_partitions(points_obj, cols, cancel_check=cancel_check):
        _check_transcript_cancelled(cancel_check)
        arrays = partition_arrays(pdf)
        if arrays is None:
            continue
        part_count = int(len(arrays.gene_codes))
        gene_lut = np.fromiter(
            (gene_to_code.get(str(name), 0) for name in arrays.gene_vocab.tolist()),
            dtype=np.int64,
            count=len(arrays.gene_vocab),
        )
        part_gene_codes = gene_lut[arrays.gene_codes].astype(gene_code_dtype, copy=False)

        keep = sampler.take(part_count)
        take = int(len(keep))
        if take:
            end = render_cursor + take
            if take == part_count:
                render_x[render_cursor:end] = arrays.x
                render_y[render_cursor:end] = arrays.y
                render_gene_codes[render_cursor:end] = part_gene_codes
                render_background[render_cursor:end] = arrays.background
            else:
                render_x[render_cursor:end] = arrays.x[keep]
                render_y[render_cursor:end] = arrays.y[keep]
                render_gene_codes[render_cursor:end] = part_gene_codes[keep]
                render_background[render_cursor:end] = arrays.background[keep]
            render_cursor = end

        if build_cell_index and arrays.cell_codes is not None and cell_keys:
            assigned_positions = np.flatnonzero(arrays.cell_codes >= 0)
            if len(assigned_positions):
                cell_lut = np.fromiter(
                    (cell_code_by_key.get(str(key), -1) for key in arrays.cell_vocab.tolist()),
                    dtype=np.int64,
                    count=len(arrays.cell_vocab),
                )
                codes = cell_lut[arrays.cell_codes[assigned_positions]]
                dest, src = _scatter_cell_rows(codes, assigned_positions, cell_write)
                cell_coords[dest, 0] = arrays.y[src]
                cell_coords[dest, 1] = arrays.x[src]
                cell_gene_codes[dest] = part_gene_codes[src]

    if render_cursor != render_count:
        raise RuntimeError(
            f"Transcript streaming sample filled {render_cursor} points; expected {render_count}"
        )

    if cell_keys:
        slices = {
            key: (int(cell_offsets[index]), int(cell_offsets[index + 1]))
            for index, key in enumerate(cell_keys)
        }
        cell_index = CellTranscriptIndex(
            coords_yx=cell_coords,
            gene_codes=cell_gene_codes,
            gene_names=tuple(source_genes),
            slices=slices,
        )

    control_genes = control_genes_from_is_gene(
        source_genes, is_gene_flags, source=f"column {is_gene_col!r}" if is_gene_col else ""
    )
    return assemble_gene_point_store(
        list(source_genes),
        render_x,
        render_y,
        render_gene_codes,
        render_background,
        gene_visuals=gene_visuals,
        reference=reference,
        alpha=alpha,
        control_genes=control_genes,
        sampled=sampled,
        source_gene_counts=source_gene_counts,
        source_total_points=total_source,
        source_status_counts=source_status_counts,
        cell_transcript_index=cell_index,
    )


def assemble_gene_point_store(
    genes: Sequence[str],
    render_x: np.ndarray,
    render_y: np.ndarray,
    render_gene_codes: np.ndarray,
    render_background: np.ndarray,
    *,
    gene_visuals: Mapping[str, GeneVisual] | None = None,
    reference: Mapping[str, Any] | None = None,
    alpha: float = 1.0,
    control_genes: set[str] | None = None,
    sampled: bool = False,
    source_gene_counts: dict[str, int] | None = None,
    source_total_points: int | None = None,
    source_status_counts: dict[str, int] | None = None,
    cell_transcript_index: CellTranscriptIndex | None = None,
) -> GenePointStore:
    """Group compact per-point arrays into a :class:`GenePointStore` (vectorised).

    ``genes`` is the alphabetical panel; ``render_gene_codes`` index it.
    ``render_background`` marks unassigned points (drawn after the assigned
    ones of each gene).  Used by :func:`build_gene_point_groups` and by the
    Atera whole-slide load, which streams its arrays itself.
    """
    unique_names = [str(g) for g in genes]
    render_count = int(len(render_gene_codes))
    if gene_visuals is None:
        if reference is not None:
            gene_visuals = build_cell_type_gene_visuals(
                unique_names, reference, kind="coarse", alpha=alpha
            ).visuals
        else:
            gene_visuals = assign_gene_visuals(unique_names, alpha=alpha)
    # Which of the 14 marker symbols each gene falls under, mapped to compact
    # group indices for only the symbols actually used. The symbol is taken from
    # the assigned visual so cell-type grouping (not alphabetical position) drives
    # the layer split; genes with no visual fall back to alphabetical index % 14.
    symbol_to_idx = {sym: i for i, sym in enumerate(GENE_MARKER_SYMBOLS)}
    symbol_idx_by_code = np.array(
        [
            symbol_to_idx.get(getattr(gene_visuals.get(name), "symbol", None), i % len(GENE_MARKER_SYMBOLS))
            for i, name in enumerate(unique_names)
        ],
        dtype=np.int64,
    )
    n_names = len(unique_names)
    # Symbols are per gene, so group, used groups and sort keys are computed per
    # gene code and gathered once per point.
    if render_count:
        code_counts = np.bincount(render_gene_codes, minlength=n_names)
    else:
        code_counts = np.zeros(n_names, dtype=np.int64)
    used_symbol_idx = np.unique(symbol_idx_by_code[code_counts > 0]) if n_names else np.zeros(0, np.int64)
    sidx_to_group = np.full(len(GENE_MARKER_SYMBOLS), -1, dtype=np.int64)
    for group_index, sidx in enumerate(used_symbol_idx):
        sidx_to_group[int(sidx)] = group_index
    group_by_code = sidx_to_group[symbol_idx_by_code] if n_names else np.zeros(0, np.int64)
    group_symbols = [GENE_MARKER_SYMBOLS[int(sidx)] for sidx in used_symbol_idx]

    # Sort primary by group, then gene code, then background (foreground
    # first).  The (group, code) order is a per-code rank, so the sort key is
    # rank * 2 + background; under 2**16 it is a uint16 radix sort.
    rank_order = np.lexsort((np.arange(n_names), group_by_code)) if n_names else np.zeros(0, np.int64)
    rank_by_code = np.empty(n_names, dtype=np.int64)
    rank_by_code[rank_order] = np.arange(n_names)
    key_dtype = np.uint16 if 2 * max(1, n_names) <= np.iinfo(np.uint16).max + 1 else np.uint32
    if render_count:
        sort_key = rank_by_code.astype(key_dtype)[render_gene_codes]
        sort_key *= key_dtype(2)
        sort_key += np.asarray(render_background, dtype=bool).astype(key_dtype)
        order = np.argsort(sort_key, kind="stable")
        del sort_key
    else:
        order = np.zeros(0, dtype=np.int64)
    coords = np.empty((render_count, 2), dtype=np.float32)
    coords[:, 0] = render_y[order]
    coords[:, 1] = render_x[order]
    codes_sorted = render_gene_codes[order]
    bg_sorted = np.asarray(render_background, dtype=bool)[order]
    del order

    code_to_rgba = np.array(
        [gene_visuals[name].rgba for name in unique_names], dtype=np.float32
    ).reshape(-1, 4)
    colors_all = code_to_rgba[codes_sorted]

    n = coords.shape[0]
    # Per-group coord/colour slices (views into the sorted arrays). Groups are
    # contiguous in rank order, so each group's start is a cumulative count.
    group_sizes = np.zeros(len(group_symbols), dtype=np.int64)
    if n_names and len(group_symbols):
        np.add.at(group_sizes, group_by_code[code_counts > 0], code_counts[code_counts > 0])
    group_starts = np.concatenate(([0], np.cumsum(group_sizes)[:-1])) if len(group_sizes) else np.zeros(0, np.int64)
    group_coords = []
    group_colors = []
    for gi in range(len(group_symbols)):
        gs = int(group_starts[gi])
        ge = gs + int(group_sizes[gi])
        group_coords.append(coords[gs:ge])
        group_colors.append(colors_all[gs:ge])

    # Per-gene contiguous ranges (relative to their group array), in rank order.
    fg_counts = np.zeros(n_names, dtype=np.int64)
    if n:
        fg_counts = np.bincount(codes_sorted[~bg_sorted], minlength=n_names)
    gene_offsets: dict[str, tuple[int, int, int, int]] = {}
    gene_counts: dict[str, int] = {}
    cursor = 0
    for code in rank_order.tolist():
        count = int(code_counts[code])
        if count == 0:
            continue
        g = int(group_by_code[code])
        gstart = int(group_starts[g])
        name = unique_names[code]
        cs = cursor
        ce = cursor + count
        gene_offsets[name] = (g, cs - gstart, cs + int(fg_counts[code]) - gstart, ce - gstart)
        gene_counts[name] = count
        cursor = ce

    return GenePointStore(
        group_symbols=group_symbols,
        group_coords=group_coords,
        group_colors=group_colors,
        gene_offsets=gene_offsets,
        gene_counts=gene_counts,
        genes=list(unique_names),
        control_genes=set(control_genes or ()),
        total_points=n,
        sampled=sampled,
        source_gene_counts=source_gene_counts,
        source_total_points=source_total_points,
        source_status_counts=source_status_counts,
        gene_visuals=dict(gene_visuals),
        cell_transcript_index=cell_transcript_index,
    )


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
    if inv.shape != (3, 3) or not np.isfinite(inv).all():
        raise ValueError("inv_affine must be a finite 3x3 matrix.")

    def draw_ring(coords, value):
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
            return
        if not np.isfinite(coords[:, :2]).all():
            return
        # Spell out the affine transform instead of dispatching thousands of
        # tiny (N, 3) @ (3, 3) operations to BLAS.  Accelerate-backed NumPy on
        # Apple Silicon can leak spurious divide/overflow floating-point status
        # from these tiny matmuls even when every operand and result is finite.
        # Label rasterisation only needs the first two affine output rows.
        xs = coords[:, 0]
        ys = coords[:, 1]
        rows = ys * inv[0, 0] + xs * inv[0, 1] + inv[0, 2] - float(y0)
        cols = ys * inv[1, 0] + xs * inv[1, 1] + inv[1, 2] - float(x0)
        if not np.isfinite(rows).all() or not np.isfinite(cols).all():
            return
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


# ---------------------------------------------------------------------------
# Cell inspection: click-to-select a segmentation mask and summarise its cell
# ---------------------------------------------------------------------------


def normalize_cell_key(value) -> str:
    """Return a canonical string key for a cell id.

    The transcript ``assignment`` column and the segmentation GeoDataFrame index
    both identify a cell, but one may store the id as an integer and the other as
    a string (``5`` vs ``"5"``). Normalising both through this function so an
    integer-valued float/str collapses to its plain integer text lets the two
    line up regardless of dtype. Integer and string ids keep every digit;
    floating-point ids outside their exact integer range are rejected.
    """
    if value is None or value is pd.NA:
        return ""
    integer = cell_id_to_int(value)
    if integer is not None:
        return str(integer)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return "" if not np.isfinite(value) or abs(value) > 2**53 else repr(float(value))
    text = str(value).strip()
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text
    if not decimal.is_finite():
        return ""
    return format(decimal.normalize(), "f")


def pick_cell_at_point(gdf, x_um: float, y_um: float):
    """Return ``(cell_id, geometry)`` for the polygon containing a point.

    ``gdf`` is a segmentation GeoDataFrame in global micron coordinates; the
    returned ``cell_id`` is the GeoDataFrame index label (the same value stored
    in the transcript ``assignment`` column). When several polygons contain the
    point (overlapping masks) the smallest-area one wins, so nested cells resolve
    to the most specific mask. Returns ``None`` when no polygon contains it.
    """
    if gdf is None or len(gdf) == 0:
        return None
    point = Point(float(x_um), float(y_um))
    try:
        positions = np.asarray(gdf.sindex.query(point, predicate="intersects"), dtype=np.int64)
        subset = gdf.iloc[positions]
    except Exception:
        subset = gdf
    if len(subset) == 0:
        return None
    try:
        contains = subset.geometry.contains(point)
    except Exception:
        return None
    containing = subset[contains.to_numpy(dtype=bool, copy=False)]
    if len(containing) == 0:
        return None
    areas = containing.geometry.area
    best_label = areas.idxmin()
    geometry = containing.geometry.loc[best_label]
    return best_label, geometry


def build_cell_transcript_index(
    points_obj,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None,
) -> CellTranscriptIndex:
    """Build a compact assigned-transcript-per-cell index.

    This fallback uses the same bounded-memory two-pass streamer as the gene
    renderer but retains only a one-point render sample.  The normal viewer path
    builds the index together with the full transcript renderer and therefore
    does not perform this extra read.
    """
    if assignment_col is None:
        return CellTranscriptIndex.empty()
    store = build_gene_point_groups(
        points_obj,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
        assignment_col=assignment_col,
        max_points=1,
        build_cell_index=True,
    )
    return store.cell_transcript_index or CellTranscriptIndex.empty()


def cell_transcript_index_from_arrays(
    cell_ids,
    x,
    y,
    codes,
    names: Sequence[str],
) -> CellTranscriptIndex:
    """A :class:`CellTranscriptIndex` from compact per-row arrays.

    ``codes`` index ``names`` (negative codes are dropped); only the names
    that occur are kept, re-coded densely.  Rows are grouped by cell id
    (stable, so each cell keeps its read order).  Cell ids keep every digit of
    the uint64 range.
    """
    cell_ids = np.asarray(cell_ids)
    codes = np.asarray(codes, dtype=np.int64)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if not (len(cell_ids) == len(codes) == len(x) == len(y)):
        raise ValueError("cell_ids, x, y and codes must have one entry per row")
    valid = codes >= 0
    if not valid.all():
        cell_ids, codes, x, y = cell_ids[valid], codes[valid], x[valid], y[valid]
    if len(codes) == 0:
        return CellTranscriptIndex.empty()
    used, dense = np.unique(codes, return_inverse=True)
    gene_names = tuple(str(names[int(i)]) for i in used)
    n_names = len(gene_names)
    dtype = np.uint8 if n_names <= 2**8 else np.uint16 if n_names <= 2**16 else np.uint32
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    ends = np.r_[starts[1:], len(sorted_ids)]
    slices = {
        normalize_cell_key(int(sorted_ids[s]) if np.issubdtype(sorted_ids.dtype, np.integer) else sorted_ids[s]):
            (int(s), int(e))
        for s, e in zip(starts, ends)
    }
    return CellTranscriptIndex(
        coords_yx=np.column_stack([y[order], x[order]]).astype(np.float32, copy=False),
        gene_codes=dense[order].astype(dtype),
        gene_names=gene_names,
        slices=slices,
    )


@dataclass(frozen=True)
class TableCellCounts:
    """One cell's row of a table's count matrix (genes only, highest first)."""

    instance_id: int
    genes: tuple[str, ...]
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        return int(sum(self.counts))

    def ranked(self) -> list[tuple[str, int]]:
        return list(zip(self.genes, self.counts))


@dataclass
class _TableRowLookup:
    stamp: tuple
    var_names: np.ndarray  # object
    sorted_ids: np.ndarray
    sorted_rows: np.ndarray
    sparse: bool
    #: ``X`` (dense) or ``(indptr, indices, data)`` zarr arrays, opened once.
    matrix: Any = None


_TABLE_LOOKUPS: dict[tuple[str, str], _TableRowLookup] = {}
_TABLE_LOOKUPS_LOCK = threading.Lock()


def _table_stamp(table_dir: Path) -> tuple:
    stamp = []
    for part in ("zarr.json", "obs/zarr.json", "var/zarr.json", "X/zarr.json"):
        try:
            st = os.stat(table_dir / part)
            stamp.append((part, st.st_size, st.st_mtime_ns))
        except OSError:
            stamp.append((part, None, None))
    return tuple(stamp)


def _table_row_lookup(zarr_path: str | Path, table_key: str) -> _TableRowLookup:
    key = (str(Path(zarr_path).resolve()), str(table_key))
    stamp = _table_stamp(Path(zarr_path) / "tables" / str(table_key))
    with _TABLE_LOOKUPS_LOCK:
        cached = _TABLE_LOOKUPS.get(key)
    if cached is not None and cached.stamp == stamp:
        return cached
    table = open_zarr_group_unconsolidated(zarr_path)["tables"][str(table_key)]
    attrs = dict(table.attrs)
    spatial = attrs.get("spatialdata_attrs") if isinstance(attrs.get("spatialdata_attrs"), Mapping) else {}
    instance_key = str(attrs.get("instance_key") or spatial.get("instance_key") or "instance_id")
    obs = table["obs"]
    if instance_key in obs:
        raw_ids = obs[instance_key]
        if isinstance(raw_ids, zarr.Group):  # categorical
            categories = np.asarray(raw_ids["categories"][:])
            codes = np.asarray(raw_ids["codes"][:], dtype=np.int64)
            raw = np.where(codes >= 0, categories[np.clip(codes, 0, None)], None)
        else:
            raw = np.asarray(raw_ids[:])
    else:
        index_name = str(dict(obs.attrs).get("_index", "_index"))
        raw = np.asarray(obs[index_name][:])
    ids, valid = cell_ids_as_int64(raw) if raw.dtype.kind not in "iu" else (raw, np.ones(len(raw), dtype=bool))
    rows = np.flatnonzero(valid)
    ids = np.asarray(ids)[rows]
    order = np.argsort(ids, kind="stable")
    var = table["var"]
    var_index = str(dict(var.attrs).get("_index", "_index"))
    var_names = np.asarray(_read_zarr_string_list(var[var_index]), dtype=object)
    x_node = table["X"]
    encoding = str(dict(x_node.attrs).get("encoding-type", "array"))
    sparse = encoding == "csr_matrix"
    lookup = _TableRowLookup(
        stamp=stamp,
        var_names=var_names,
        sorted_ids=ids[order],
        sorted_rows=rows[order],
        sparse=sparse,
        matrix=(x_node["indptr"], x_node["indices"], x_node["data"]) if sparse else x_node,
    )
    with _TABLE_LOOKUPS_LOCK:
        _TABLE_LOOKUPS[key] = lookup
    return lookup


def table_cell_gene_counts(
    zarr_path: str | Path, table_key: str, instance_id: Any
) -> TableCellCounts | None:
    """One cell's gene counts from ``tables/<table_key>/X``, reading one row.

    The var names and an ``instance_id -> row`` lookup are cached per table
    (invalidated when its metadata changes); a CSR matrix is read through
    ``indptr`` so only that row's slice of ``data`` / ``indices`` is decoded.
    Control features (by the name rule) are dropped; Atera's ``X`` holds
    genes only, so these are gene transcripts.  ``None`` when the cell is not
    in the table.
    """
    target = cell_id_to_int(instance_id)
    if target is None:
        return None
    lookup = _table_row_lookup(zarr_path, table_key)
    ids = lookup.sorted_ids
    if ids.dtype.kind == "u" and target < 0:
        return None
    try:
        probe = np.asarray([target], dtype=ids.dtype)[0] if ids.dtype != object else target
    except OverflowError:
        return None
    pos = int(np.searchsorted(ids, probe))
    if pos >= len(ids) or ids[pos] != probe:
        return None
    row = int(lookup.sorted_rows[pos])
    if lookup.sparse:
        indptr, indices, data = lookup.matrix
        start, end = (int(v) for v in np.asarray(indptr[row : row + 2]))
        columns = np.asarray(indices[start:end], dtype=np.int64)
        values = np.asarray(data[start:end])
    else:
        dense = np.asarray(lookup.matrix[row, :]).ravel()
        columns = np.flatnonzero(dense)
        values = dense[columns]
    counts = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    keep = counts > 0
    columns, counts = columns[keep], counts[keep]
    names = lookup.var_names[columns]
    pairs = [(str(n), int(c)) for n, c in zip(names, counts) if not is_control_gene(str(n))]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return TableCellCounts(
        instance_id=int(target),
        genes=tuple(p[0] for p in pairs),
        counts=tuple(p[1] for p in pairs),
    )


def ranked_gene_counts(genes) -> list[tuple[str, int]]:
    """Return ``(gene, count)`` pairs sorted by count desc, then name asc."""
    genes = np.asarray(genes, dtype=object)
    if genes.size == 0:
        return []
    names, counts = np.unique(genes.astype(str), return_counts=True)
    pairs = list(zip(names.tolist(), counts.tolist(), strict=True))
    pairs.sort(key=lambda item: (-int(item[1]), str(item[0])))
    return [(str(name), int(count)) for name, count in pairs]


def darken_rgba(rgba, factor: float = 0.6) -> tuple[float, float, float, float]:
    """Return ``rgba`` with its RGB channels scaled toward black by ``factor``."""
    factor = float(factor)
    r, g, b = (max(0.0, min(1.0, float(c) * factor)) for c in tuple(rgba)[:3])
    alpha = float(rgba[3]) if len(tuple(rgba)) >= 4 else 1.0
    return (r, g, b, alpha)


def mean_intensity_in_polygon(image_2d, napari_affine, polygon) -> float | None:
    """Mean pixel intensity inside ``polygon`` for one 2D image channel.

    ``image_2d`` is a ``(y, x)`` pixel array (numpy or lazy/dask, sliced then
    materialised over just the polygon's bounding box). ``napari_affine`` is the
    3x3 matrix mapping pixel ``(row, col, 1)`` to global ``(y_um, x_um, 1)`` used
    by the layer; ``polygon`` is a shapely geometry in global microns. Returns
    ``None`` when the polygon falls outside the image or covers no pixels.
    """
    if polygon is None or getattr(polygon, "is_empty", True):
        return None
    affine = np.asarray(napari_affine, dtype=float)
    if affine.shape != (3, 3):
        return None
    try:
        inv_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError:
        return None

    shape = tuple(int(s) for s in getattr(image_2d, "shape", ()))
    if len(shape) < 2:
        return None
    height, width = shape[-2], shape[-1]

    parts = polygon.geoms if getattr(polygon, "geom_type", "") == "MultiPolygon" else (polygon,)
    all_coords = []
    for part in parts:
        if part.is_empty:
            continue
        all_coords.append(np.asarray(part.exterior.coords, dtype=float))
    if not all_coords:
        return None
    coords = np.concatenate(all_coords, axis=0)  # (N, 2) as (x_um, y_um)
    yx1 = np.column_stack([coords[:, 1], coords[:, 0], np.ones(len(coords))])
    rc = yx1 @ inv_affine.T
    rows, cols = rc[:, 0], rc[:, 1]
    r0 = max(0, int(np.floor(rows.min())))
    r1 = min(height, int(np.ceil(rows.max())) + 1)
    c0 = max(0, int(np.floor(cols.min())))
    c1 = min(width, int(np.ceil(cols.max())) + 1)
    if r1 <= r0 or c1 <= c0:
        return None

    mask = rasterize_geometries_chunk(
        parts,
        [1] * len(parts),
        shape=(r1 - r0, c1 - c0),
        inv_affine=inv_affine,
        y0=r0,
        x0=c0,
        dtype=np.uint8,
    ).astype(bool)
    if not mask.any():
        return None

    window = np.asarray(image_2d[..., r0:r1, c0:c1])
    while window.ndim > 2:
        window = window[0]
    values = window[mask]
    if values.size == 0:
        return None
    return float(np.nanmean(values))


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
