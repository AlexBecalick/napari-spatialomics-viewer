#!/usr/bin/env python3
"""Utility helpers for the Xenium/MERSCOPE Napari comparison viewer."""

from __future__ import annotations

import colorsys
import json
import logging
import re
from dataclasses import dataclass
from hashlib import blake2s
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import zarr
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import linemerge, polygonize, unary_union

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
    n_bins = int(ny) * int(nx)
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
        np.clip(ix, 0, nx - 1, out=ix)
        np.clip(iy, 0, ny - 1, out=iy)
        flat = iy * np.int64(nx) + ix

        if assignment_col is not None and assignment_col in pdf.columns:
            assigned = assignment_mask(pdf.loc[good, assignment_col]).to_numpy(dtype=bool, copy=False)
        else:
            assigned = np.ones(flat.shape[0], dtype=bool)

        # np.bincount is vectorized and ~10-50x faster than np.add.at, which is
        # an unbuffered scatter. The per-channel counts are reshaped (a view) and
        # accumulated into the running density grid.
        n_points = int(flat.shape[0])
        assigned_count = int(np.count_nonzero(assigned))
        if assigned_count > 0:
            counts = np.bincount(flat[assigned], minlength=n_bins)
            density[0] += counts[:n_bins].reshape(ny, nx).astype(np.uint32, copy=False)
        if assigned_count < n_points:
            counts = np.bincount(flat[~assigned], minlength=n_bins)
            density[1] += counts[:n_bins].reshape(ny, nx).astype(np.uint32, copy=False)

        total += n_points
        assigned_total += assigned_count
        unassigned_total += n_points - assigned_count

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


# ---------------------------------------------------------------------------
# Per-gene transcript inspection ("Inspect Genes")
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
    gene_counts: dict[str, int]              # total (fg + bg) points per gene
    genes: list[str]                         # alphabetical (real genes + controls)
    control_genes: set[str]
    total_points: int
    sampled: bool = False

    def gene_symbol(self, gene: str) -> str | None:
        entry = self.gene_offsets.get(str(gene))
        return None if entry is None else self.group_symbols[entry[0]]


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
    )


def build_gene_point_groups(
    points_obj,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None = None,
    background_col: str | None = None,
    gene_visuals: Mapping[str, GeneVisual] | None = None,
    max_points: int | None = None,
    random_state: int = 42,
    alpha: float = 1.0,
) -> GenePointStore:
    """Read all transcripts into a symbol-grouped, gene-sorted point store.

    ``background`` is taken from ``background_col`` (bool) if present, else from
    ``assignment_col`` (unassigned == not :func:`assignment_mask`), else all
    foreground. If the total exceeds ``max_points`` a uniform random subsample is
    taken (this preserves per-gene proportions in expectation).
    """
    cols = [x_col, y_col, gene_col]
    if background_col is not None:
        cols.append(background_col)
    if assignment_col is not None and assignment_col not in cols:
        cols.append(assignment_col)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gene_parts: list[np.ndarray] = []
    bg_parts: list[np.ndarray] = []

    for pdf in _iter_point_partitions(points_obj, cols):
        x_vals = pdf[x_col].to_numpy(dtype=np.float32, copy=False)
        y_vals = pdf[y_col].to_numpy(dtype=np.float32, copy=False)
        gene_vals = pdf[gene_col].astype("string").to_numpy(dtype=object)
        good = np.isfinite(x_vals) & np.isfinite(y_vals) & pd.notna(gene_vals)
        if not np.any(good):
            continue
        if background_col is not None and background_col in pdf.columns:
            bg = pdf[background_col].to_numpy(dtype=bool, copy=False)
        elif assignment_col is not None and assignment_col in pdf.columns:
            bg = ~assignment_mask(pdf[assignment_col]).to_numpy(dtype=bool, copy=False)
        else:
            bg = np.zeros(len(pdf), dtype=bool)
        xs.append(np.ascontiguousarray(x_vals[good]))
        ys.append(np.ascontiguousarray(y_vals[good]))
        gene_parts.append(gene_vals[good].astype(str))
        bg_parts.append(np.ascontiguousarray(bg[good]))

    if not xs:
        return _empty_gene_point_store()

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    gene = np.concatenate(gene_parts)
    bg = np.concatenate(bg_parts)
    del xs, ys, gene_parts, bg_parts
    total_source = int(x.shape[0])

    sampled = False
    if max_points is not None and 0 < int(max_points) < total_source:
        rng = np.random.default_rng(int(random_state))
        keep = np.sort(rng.choice(total_source, size=int(max_points), replace=False))
        x, y, gene, bg = x[keep], y[keep], gene[keep], bg[keep]
        sampled = True

    # Stable gene visuals + integer codes over the alphabetical gene panel.
    unique_names = sorted(set(gene.tolist()))
    if gene_visuals is None:
        gene_visuals = assign_gene_visuals(unique_names, alpha=alpha)
    codes = pd.Categorical(gene, categories=unique_names).codes.astype(np.int64)
    del gene

    # Which of the 14 marker symbols each gene falls under, mapped to compact
    # group indices for only the symbols actually used.
    symbol_idx_per_point = codes % len(GENE_MARKER_SYMBOLS)
    used_symbol_idx = np.unique(symbol_idx_per_point)
    sidx_to_group = np.full(len(GENE_MARKER_SYMBOLS), -1, dtype=np.int64)
    for group_index, sidx in enumerate(used_symbol_idx):
        sidx_to_group[int(sidx)] = group_index
    group_per_point = sidx_to_group[symbol_idx_per_point]
    group_symbols = [GENE_MARKER_SYMBOLS[int(sidx)] for sidx in used_symbol_idx]

    # Sort primary by group, then gene code, then background (foreground first).
    order = np.lexsort((bg, codes, group_per_point))
    coords = np.column_stack([y[order], x[order]]).astype(np.float32, copy=False)
    codes_sorted = codes[order]
    bg_sorted = bg[order]
    group_sorted = group_per_point[order]
    del order, x, y, bg, codes, group_per_point, symbol_idx_per_point

    code_to_rgba = np.array(
        [gene_visuals[name].rgba for name in unique_names], dtype=np.float32
    )
    colors_all = code_to_rgba[codes_sorted]

    n = coords.shape[0]
    # Per-group coord/colour slices (views into the sorted arrays). group_sorted
    # is non-decreasing, so np.unique gives each group's start in order.
    _uniq_groups, group_starts = np.unique(group_sorted, return_index=True)
    group_ends = list(group_starts[1:]) + [n]
    group_coords = []
    group_colors = []
    for gi in range(len(group_symbols)):
        gs = int(group_starts[gi])
        ge = int(group_ends[gi])
        group_coords.append(coords[gs:ge])
        group_colors.append(colors_all[gs:ge])

    # Per-gene contiguous ranges (relative to their group array). Each gene code
    # is one contiguous run in ``codes_sorted`` (each code maps to one group and
    # is code-sorted within it), but the runs are ordered by (group, code) rather
    # than by code value, so run boundaries must come from actual value changes
    # -- NOT from a code-sorted np.unique index.
    if n > 0:
        change = np.flatnonzero(np.diff(codes_sorted) != 0) + 1
        block_starts = np.concatenate(([0], change))
        block_ends = np.concatenate((change, [n]))
    else:
        block_starts = np.empty(0, dtype=np.int64)
        block_ends = np.empty(0, dtype=np.int64)
    gene_offsets: dict[str, tuple[int, int, int, int]] = {}
    gene_counts: dict[str, int] = {}
    for bi in range(len(block_starts)):
        cs = int(block_starts[bi])
        ce = int(block_ends[bi])
        g = int(group_sorted[cs])
        gstart = int(group_starts[g])
        fg_count = int(np.count_nonzero(~bg_sorted[cs:ce]))
        name = unique_names[int(codes_sorted[cs])]
        gene_offsets[name] = (g, cs - gstart, cs + fg_count - gstart, ce - gstart)
        gene_counts[name] = ce - cs

    control_genes = {name for name in unique_names if is_control_gene(name)}
    return GenePointStore(
        group_symbols=group_symbols,
        group_coords=group_coords,
        group_colors=group_colors,
        gene_offsets=gene_offsets,
        gene_counts=gene_counts,
        genes=list(unique_names),
        control_genes=control_genes,
        total_points=n,
        sampled=sampled,
    )


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
