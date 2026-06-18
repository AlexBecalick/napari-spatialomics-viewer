#!/usr/bin/env python3
"""Utility helpers for the Xenium/MERSCOPE Napari comparison viewer."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import box

try:
    from skimage.draw import polygon as draw_polygon
except Exception:  # pragma: no cover - import error depends on runtime env
    draw_polygon = None

log = logging.getLogger("napari_compare_utils")


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
