#!/usr/bin/env python3
"""
Interactive Napari viewer for comparing MERSCOPE and Xenium SpatialData outputs.

Features:
- Single viewer with dataset switcher (MERSCOPE/XENIUM).
- All image channels as separate image layers in micron coordinates.
- All shape keys as separate polygon layers with deterministic colors.
- Assigned vs unassigned transcript layers from points['assignment']-style columns.
"""

from __future__ import annotations

import argparse
import colorsys
import gc
import hashlib
import html
import json
import logging
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import dask
    import dask.array as da

    dask.config.set({"dataframe.query-planning": True})
except Exception:
    dask = None
    da = None

warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated and will be removed in a future version.*",
    category=FutureWarning,
    module="dask\\.dataframe",
)

import spatialdata as sd
import zarr
from napari.utils.colormaps import Colormap, DirectLabelColormap, ensure_colormap
from packaging.version import InvalidVersion, Version
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations import Affine
from spatialdata.transformations import get_transformation
from spatialdata.transformations import set_transformation

try:
    import napari
except ImportError:
    print(
        "ERROR: napari is not installed in this environment.\n"
        "Install it with: pip install 'napari[all]' or conda install napari",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from qtpy.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
    from qtpy.QtGui import (
        QBrush,
        QColor,
        QFont,
        QFontMetricsF,
        QIcon,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QPolygonF,
    )
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print(
        "ERROR: qtpy is required for the Napari dock widget.\n"
        "Install it with: pip install qtpy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import psutil
except Exception:
    psutil = None

try:
    from napari.qt.threading import thread_worker
except Exception:  # pragma: no cover - depends on the installed napari runtime
    thread_worker = None

from .dask_cache import install_thread_safe_napari_dask_cache
from .utils import (
    CELLPOSE_LABEL_KEY,
    CELLPOSE_QUANTIFICATION_TABLE_KEY,
    CELLPOSE_SHAPE_KEY,
    CELLPOSE_VALUE_BINS,
    CORTICAL_DEPTH_DEFAULT_PIECE_ID,
    CORTICAL_DEPTH_PIECE_ID_PROPERTY,
    CORTICAL_DEPTH_ROLE_ORDER,
    CORTICAL_DEPTH_ROLE_SPECS,
    OBJECT_ID_PROPERTY,
    ObjectAnnotationShapeInput,
    GeneVisual,
    CellTranscriptIndex,
    CorticalDepthShapeInput,
    DERIVED_CACHE_ATTR,
    assign_gene_visuals,
    build_cell_type_gene_visuals,
    build_cell_type_color_dict,
    build_cell_type_color_schemes,
    clustering_table_key_for_segmentation,
    load_cell_type_assignments,
    load_cell_type_marker_reference,
    build_cell_transcript_index,
    darken_rgba,
    mean_intensity_in_polygon,
    normalize_cell_key,
    pick_cell_at_point,
    ranked_gene_counts,
    affine_matrix_from_px_to_um,
    build_binned_label_color_dict,
    build_cortical_depth_annotation_geojson,
    build_object_annotation_geojson,
    build_gene_point_groups,
    build_napari_affine_from_px_to_um,
    cellpose_quantification_features,
    cellpose_quantification_table_available,
    channel_labels,
    derived_image_pyramid_cache_key,
    derived_label_pyramid_cache_key,
    derived_outline_cache_key,
    ensure_cyx,
    first_existing_col,
    get_scale0_dataarray,
    gene_marker_symbol_label,
    image_scale_dataarrays,
    is_derived_cache_key,
    label_outline_mask_chunk,
    layer_name_prefix,
    load_cellpose_quantification_values,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    query_geometries_for_bounds,
    read_object_annotation_geojson,
    rasterize_geometries_chunk,
    resolve_dataset_mask_affine,
    resolve_gene_column,
    snap_cortical_depth_boundaries_to_edge,
    write_cortical_depth_annotation_geojson,
    write_cortical_depth_separate_geojsons,
    write_object_annotation_geojson,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("napari_compare")
logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.io").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.scale").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.format").setLevel(logging.WARNING)


class _OmeZarrLabelParentWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.name == "ome_zarr.reader"
            and message.startswith("no parent found for")
            and "ome_zarr.reader.Label" in message
        )


logging.getLogger("ome_zarr.reader").addFilter(_OmeZarrLabelParentWarningFilter())


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    zarr_path: Path
    merscope_transform_path: Path | None = None
    xenium_spec_path: Path | None = None



@dataclass
class GeneInspectorState:
    """Live state for the per-gene transcript renderer of one dataset."""

    dataset: str
    points_key: str
    store: object                      # utils.GenePointStore
    gene_visuals: dict                 # gene -> utils.GeneVisual
    layer_names: list[str]             # one napari Points layer per symbol group
    enabled_genes: set                 # genes currently shown
    spot_size: float
    hide_assigned: bool = False
    hide_background: bool = False
    show_controls: bool = False
    rebuild_timer: QTimer | None = None
    pending_groups: set | None = None
    highlighted_genes: list | None = None
    group_display_ranges: list[list[tuple[int, int, str]]] | None = None
    #: Cell-type marker metadata + the two precomputed visual schemes. When
    #: ``reference`` is None the panel has no metadata and only flat A–Z ordering
    #: (the legacy rainbow) is offered. ``ordering`` is the list layout the user
    #: picked (coarse/fine/alphabetical); ``color_kind`` is which scheme's colours
    #: are currently painted on the points (A–Z keeps whichever was last applied).
    reference: dict | None = None
    coarse_scheme: object | None = None       # utils.GeneVisualScheme
    fine_scheme: object | None = None          # utils.GeneVisualScheme
    ordering: str = "coarse"
    color_kind: str = "coarse"
    colour_by_assignment: bool = False


@dataclass
class CellTypeOverlayState:
    """Live state for the cell-type mask-fill overlay of one dataset.

    One dataset can colour one segmentation at a time. ``assignments`` and the
    two colour ``schemes`` are cached per segmentation so switching broad/fine or
    toggling a type recolours without re-reading the store. ``enabled`` holds the
    labels currently shown for the active ``kind`` (broad/fine tracked
    separately, so switching levels preserves each level's tick state).
    """

    dataset: str
    segmentation: str = "proseg"
    kind: str = "broad"
    opacity: float = 0.95
    layer_name: str | None = None
    label_key: str | None = None
    #: segmentation -> utils.CellTypeAssignments (None means "looked up, absent").
    assignments: dict = field(default_factory=dict)
    #: segmentation -> {"broad": CellTypeColorScheme, "fine": CellTypeColorScheme}.
    schemes: dict = field(default_factory=dict)
    #: segmentation -> {"broad": set(enabled labels), "fine": set(enabled labels)}.
    enabled: dict = field(default_factory=dict)


DEFAULT_GENE_SPOT_SIZE = 0.5
GENE_HIGHLIGHT_SPOT_SIZE = 2.0
#: When the (smaller) visible field of view is at or below this many microns, the
#: label outlines switch to crisp nearest-neighbour interpolation; above it they
#: use anti-aliased linear interpolation. Only active with --label-interpolation linear.
LABEL_CRISP_ZOOM_UM = 150.0
GENE_SPOT_SIZE_MIN = 0.1
GENE_SPOT_SIZE_MAX = 5.0
GENE_SPOT_SIZE_STEP = 0.1
DEFAULT_GENE_MAX_RENDER_POINTS = 40_000_000
GENE_REBUILD_DEBOUNCE_MS = 60
GENE_ASSIGNED_RGBA = np.array((1.0, 1.0, 0.0, 1.0), dtype=np.float32)
GENE_UNASSIGNED_RGBA = np.array((1.0, 0.0, 0.0, 1.0), dtype=np.float32)
GENE_STATUS_SYMBOL_GLYPHS = {
    "disc": "●",
    "ring": "○",
    "square": "■",
    "diamond": "◆",
    "cross": "+",
    "x": "×",
    "triangle_up": "▲",
    "triangle_down": "▼",
    "star": "★",
    "arrow": "▲",
    "tailed_arrow": "▲",
    "hbar": "▬",
    "vbar": "▌",
    "clobber": "✣",
}

SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE = 4096
SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS = 10
LABEL_CACHE_ATTR = "napari_compare_label_cache"
# Raster layers use napari's bounded opportunistic RAM cache. At startup we
# replace its thread-unsafe callback bookkeeping with ThreadSafeDaskCache so
# concurrent layers retain hot chunks without Cache._posttask races.
NAPARI_DASK_CACHE_ENABLED = True
LABEL_OUTLINE_PYRAMID_MIN_SIZE = 4096
LABEL_OUTLINE_PYRAMID_MAX_LEVELS = 10
DERIVED_CACHE_VERSION = 1
CORTICAL_DEPTH_LAYER_TYPE = "cortical_depth"
CORTICAL_DEPTH_LAYER_COLORS = {
    "pia": "#00d5ff",
    "wm": "#ffb000",
    "side": "#ff5da2",
    "exclusion": "#ff3333",
    "ribbon": "#00d084",
}
CORTICAL_DEPTH_FILL_COLORS = {
    "pia": "transparent",
    "wm": "transparent",
    "side": "transparent",
    "exclusion": [1.0, 0.2, 0.2, 0.20],
    "ribbon": [0.0, 0.8, 0.5, 0.12],
}
CORTICAL_DEPTH_PIECE_ROLES = ("pia", "wm", "exclusion", "ribbon")
DISTANCE_OBJECT_LAYER_TYPE = "distance_object"
DISTANCE_OBJECT_EDGE_COLOR = "#ff4fd8"
DISTANCE_OBJECT_FILL_COLOR = "#ff4fd833"

# -- Cell inspector (click a segmentation mask to summarise its cell) --------
CELL_INSPECTOR_BOUNDARY_LAYER = "Cell inspector | selected boundary"
CELL_INSPECTOR_LINKS_LAYER = "Cell inspector | transcript links"
#: Distinct border colours cycled across simultaneously-highlighted cells; each
#: cell's boundary and its Cell ID text in the bottom bar share the same colour.
CELL_HIGHLIGHT_COLORS = (
    "#ffe14d",  # yellow
    "#4dd2ff",  # cyan
    "#ff6ec7",  # pink
    "#7cff4d",  # green
    "#ff9f40",  # orange
    "#b48cff",  # violet
    "#ff5d5d",  # red
    "#4dffd0",  # teal
)
#: The selected-cell outline is drawn in micron (data) units so it scales with
#: zoom exactly like the rasterised mask outlines. Its width is the mask outline
#: width scaled by this factor, so it always reads as marginally thicker than the
#: surrounding cells' outlines. The fallback (in microns) is only used when no
#: mask outline layer is loaded to measure against.
CELL_BOUNDARY_WIDTH_FACTOR = 1.6
CELL_BOUNDARY_FALLBACK_WIDTH_UM = 0.3
CELL_LINK_COLOR = (1.0, 1.0, 1.0, 0.35)
CELL_LINK_WIDTH = 0.25
CELL_PIE_SLICE_DARKEN = 0.6
#: Cell masks whose GeoDataFrames are preferred targets for a mask click, most
#: specific first. ProSeg is the primary target named in the feature request.
CELL_INSPECTOR_SHAPE_PREFERENCE = ("proseg", "cellpose", "cell")


def startup_selection(segmentation_source: str) -> tuple[str, ...]:
    """Return the SpatialData element types to read when a dataset opens.

    Points and segmentations are read (lazily) so their keys can populate the tab
    lists; images are read separately (and defensively) so an image-less store
    still opens. The ``--skip-*`` flags gate whether layers are *rendered* at
    startup, not whether the metadata is read.
    """
    seg = "labels" if str(segmentation_source).lower() == "labels" else "shapes"
    return ("points", seg)


def _parse_version_or_none(value: str | None) -> Version | None:
    """Best-effort semantic version parsing."""
    if not value:
        return None
    try:
        return Version(str(value))
    except InvalidVersion:
        return None


def read_root_zarr_metadata(zarr_path: Path) -> dict | None:
    """Read root zarr metadata when present."""
    meta_path = Path(zarr_path) / "zarr.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def validate_spatialdata_store_compatibility(zarr_path: Path):
    """Fail early with an actionable message for incompatible SpatialData stores."""
    meta = read_root_zarr_metadata(zarr_path)
    if meta is None:
        return

    if meta.get("zarr_format") != 3:
        return

    attrs = meta.get("attributes", {})
    spatial_attrs = attrs.get("spatialdata_attrs", {})
    writer_version = spatial_attrs.get("spatialdata_software_version")

    installed_spatialdata = getattr(sd, "__version__", "unknown")
    installed_zarr = getattr(zarr, "__version__", "unknown")

    problems: list[str] = []
    zarr_version = _parse_version_or_none(installed_zarr)
    if zarr_version is not None and zarr_version.major < 3:
        problems.append(f"installed zarr is {installed_zarr} but this store is Zarr v3")

    writer_semver = _parse_version_or_none(writer_version)
    installed_semver = _parse_version_or_none(installed_spatialdata)
    if writer_semver is not None and installed_semver is not None and installed_semver < writer_semver:
        problems.append(
            f"store was written by spatialdata {writer_version} but this env has spatialdata {installed_spatialdata}"
        )

    if not problems:
        return

    details = "; ".join(problems)
    required_spatialdata = writer_version or "a version compatible with the store writer"
    raise RuntimeError(
        "Incompatible SpatialData reader environment for "
        f"{zarr_path}. {details}. "
        f"Update this viewer environment to spatialdata>={required_spatialdata} with zarr>=3."
    )


def memory_snapshot_gb() -> dict[str, float]:
    """Return current process/system memory snapshot in GB."""
    if psutil is None:
        return {"rss_gb": float("nan"), "sys_used_gb": float("nan")}

    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024**3)
    used = psutil.virtual_memory().used / (1024**3)
    return {"rss_gb": float(rss), "sys_used_gb": float(used)}


def stable_layer_color(key: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Generate deterministic RGBA color for a layer key."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    sat = 0.75
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (float(r), float(g), float(b), float(alpha))


def image_colormap_for_channel(channel_name: str, fallback_index: int = 0) -> str:
    """Select a napari colormap based on channel name."""
    name = str(channel_name).lower()
    if "dapi" in name:
        return "blue"
    if ("polyt" in name) or ("18s" in name) or ("rna" in name):
        return "green"

    fallback = ["gray", "magenta", "cyan", "yellow", "red", "orange"]
    return fallback[fallback_index % len(fallback)]


def contrast_limits_from_dtype(data) -> tuple[float, float] | None:
    """Return cheap contrast limits that avoid full-image min/max scans."""
    try:
        dtype = np.dtype(getattr(data, "dtype"))
    except Exception:
        return None

    if dtype.kind == "b":
        return (0.0, 1.0)
    if dtype.kind in ("u", "i"):
        info = np.iinfo(dtype)
        return (float(info.min), float(info.max))
    return None


def lazy_subsampled_pyramid(
    base_data,
    min_size: int = SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE,
    max_levels: int = SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS,
) -> list[object]:
    """Build a lazy 2D pyramid by stride-subsampling a single-scale image."""
    levels = [base_data]
    data = base_data

    while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
        if len(data.shape) != 2:
            break
        data = data[::2, ::2]
        if data.shape == levels[-1].shape:
            break
        levels.append(data)

    return levels


def lazy_coarsened_pyramid(
    base_data,
    step: int,
    reducer=None,
    min_size: int = SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE,
    max_levels: int = SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS,
    tile: int = 1024,
) -> list[object]:
    """Build materialized-ready coarse levels for an image or label array.

    Each level downsamples the trailing (y, x) axes by ``step`` using ``reducer``
    (``np.mean`` for intensity images; ``np.max`` for label ids so ids are
    preserved rather than averaged). Unlike stride subsampling, every returned
    level is a genuine reduced-resolution array, so reading a coarse tile does
    NOT force a full-resolution read once the level is persisted. Level 0 (the
    base) is intentionally excluded; callers reuse the existing lazy base array
    so the multi-gigapixel scale0 is never duplicated.
    """
    if da is None:
        return []
    if reducer is None:
        reducer = np.mean
    step = max(2, int(step))
    data = da.asarray(base_data)
    ndim = data.ndim
    if ndim not in (2, 3):
        return []
    y_axis, x_axis = ndim - 2, ndim - 1
    dtype = data.dtype
    is_integer = np.issubdtype(dtype, np.integer)

    levels: list[object] = []
    current = data
    prev_shape = tuple(int(s) for s in data.shape)
    while len(levels) < max_levels and max(int(current.shape[y_axis]), int(current.shape[x_axis])) > min_size:
        if int(current.shape[y_axis]) < step or int(current.shape[x_axis]) < step:
            break
        coarsened = da.coarsen(reducer, current, axes={y_axis: step, x_axis: step}, trim_excess=True)
        # Averaging promotes to float; round back for integer sources. Max/other
        # reducers preserve the integer dtype, so avoid a needless float round-trip.
        if reducer is np.mean and is_integer:
            coarsened = da.rint(coarsened).astype(dtype)
        else:
            coarsened = coarsened.astype(dtype)
        new_shape = tuple(int(s) for s in coarsened.shape)
        if new_shape[-2:] == prev_shape[-2:]:
            break
        if ndim == 3:
            chunks = (new_shape[0], min(tile, new_shape[1]), min(tile, new_shape[2]))
        else:
            chunks = (min(tile, new_shape[0]), min(tile, new_shape[1]))
        coarsened = coarsened.rechunk(chunks)
        levels.append(coarsened)
        current = coarsened
        prev_shape = new_shape
    return levels


def lazy_outline_pyramid(
    label_data,
    width: int,
    min_size: int = LABEL_OUTLINE_PYRAMID_MIN_SIZE,
    max_levels: int = LABEL_OUTLINE_PYRAMID_MAX_LEVELS,
) -> list[object]:
    """Build a lazy multiscale uint8 outline pyramid from a 2D label image."""
    width = max(1, int(width))
    label_levels = lazy_label_pyramid(label_data, min_size=min_size, max_levels=max_levels)
    return lazy_outline_pyramid_from_label_levels(label_levels, width=width)


def lazy_label_pyramid(
    label_data,
    min_size: int = LABEL_OUTLINE_PYRAMID_MIN_SIZE,
    max_levels: int = LABEL_OUTLINE_PYRAMID_MAX_LEVELS,
) -> list[object]:
    """Build a lazy 2D label pyramid by max-pooling label ids."""
    if da is not None:
        data = da.asarray(label_data)
        levels: list[object] = [data]
        while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
            data = da.coarsen(np.max, data, axes={0: 2, 1: 2}, trim_excess=True)
            if data.shape == levels[-1].shape:
                break
            levels.append(data)
        return levels

    data = np.asarray(label_data)
    levels = [data]
    while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
        y = (data.shape[0] // 2) * 2
        x = (data.shape[1] // 2) * 2
        if y < 2 or x < 2:
            break
        data = data[:y, :x].reshape(y // 2, 2, x // 2, 2).max(axis=(1, 3))
        if data.shape == levels[-1].shape:
            break
        levels.append(data)
    return levels


def lazy_outline_mask(label_data, width: int) -> object:
    """Build one lazy uint8 outline mask from one 2D label level."""
    width = max(1, int(width))

    if da is not None:
        labels = da.asarray(label_data)
        return labels.map_overlap(
            label_outline_mask_chunk,
            depth=max(1, width),
            boundary=0,
            trim=True,
            dtype=np.uint8,
            width=width,
        )

    return label_outline_mask_chunk(label_data, width=width)


def _outline_width_for_level(width: int, base_shape: tuple[int, int], level_shape: tuple[int, int]) -> int:
    """Scale outline width down for coarser pyramid levels."""
    if width <= 1:
        return 1
    y_factor = float(base_shape[0]) / max(1.0, float(level_shape[0]))
    x_factor = float(base_shape[1]) / max(1.0, float(level_shape[1]))
    scale_factor = max(1.0, y_factor, x_factor)
    return max(1, int(np.ceil(float(width) / scale_factor)))


def lazy_outline_pyramid_from_label_levels(label_levels: list[object], width: int) -> list[object]:
    """Build outline masks independently from existing or synthetic label levels."""
    if len(label_levels) == 0:
        return []
    base_shape = tuple(int(axis) for axis in getattr(label_levels[0], "shape"))
    outlines: list[object] = []
    for level in label_levels:
        level_shape = tuple(int(axis) for axis in getattr(level, "shape"))
        level_width = _outline_width_for_level(int(width), base_shape, level_shape)
        outlines.append(lazy_outline_mask(level, width=level_width))
    return outlines


def lazy_density_pyramid(
    density_cyx,
    min_size: int = SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE,
    max_levels: int = SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS,
) -> list[object]:
    """Build a display pyramid with coarse levels normalized to base-bin density."""
    if da is None:
        data = np.asarray(density_cyx, dtype=np.float32)
        levels: list[object] = [data]
        while len(levels) < max_levels and max(int(axis) for axis in data.shape[-2:]) > min_size:
            y = (int(data.shape[1]) // 2) * 2
            x = (int(data.shape[2]) // 2) * 2
            if y < 2 or x < 2:
                break
            data = data[:, :y, :x].reshape(data.shape[0], y // 2, 2, x // 2, 2).mean(axis=(2, 4)).astype(
                np.float32,
                copy=False,
            )
            if data.shape == levels[-1].shape:
                break
            levels.append(data)
        return levels

    data = da.asarray(density_cyx).astype(np.float32)
    levels: list[object] = [data]
    while len(levels) < max_levels and max(int(axis) for axis in data.shape[-2:]) > min_size:
        axes = {axis: 2 for axis in (1, 2) if int(data.shape[axis]) >= 2}
        if len(axes) == 0:
            break
        data = da.coarsen(np.mean, data, axes=axes, trim_excess=True).astype(np.float32)
        if data.shape == levels[-1].shape:
            break
        levels.append(data)
    return levels


def rgba_array(color, alpha: float = 1.0) -> np.ndarray:
    """Convert common napari color inputs to an RGBA float array."""
    if isinstance(color, (list, tuple, np.ndarray)):
        arr = np.asarray(color, dtype=np.float32)
        if arr.size == 3:
            arr = np.concatenate([arr, np.asarray([1.0], dtype=np.float32)])
        if arr.size >= 4:
            arr = arr[:4].astype(np.float32, copy=False)
            arr[3] *= float(alpha)
            return arr

    text = str(color).strip().lower()
    named = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "orange": (1.0, 0.55, 0.0),
    }
    if text in named:
        rgb = named[text]
        return np.asarray([rgb[0], rgb[1], rgb[2], float(alpha)], dtype=np.float32)

    if text.startswith("#"):
        raw = text[1:]
        if len(raw) == 3:
            raw = "".join(ch * 2 for ch in raw)
        if len(raw) == 6:
            rgb = tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
            return np.asarray([rgb[0], rgb[1], rgb[2], float(alpha)], dtype=np.float32)

    return np.asarray([1.0, 1.0, 1.0, float(alpha)], dtype=np.float32)


def transparent_colormap(name: str, color, alpha: float = 1.0) -> Colormap:
    """Return a transparent-to-color colormap for binary/density overlays."""
    rgba = rgba_array(color, alpha=alpha)
    return Colormap(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                rgba,
            ],
            dtype=np.float32,
        ),
        name=name,
        controls=np.asarray([0.0, 1.0], dtype=np.float32),
    )


class FlowLayout(QLayout):
    """A layout that lays widgets left-to-right and wraps to new rows as needed.

    Used for the tab buttons so every button keeps its full-text width (nothing is
    elided) and the row simply wraps onto additional rows when the dock is narrow.
    Adapted from the canonical Qt FlowLayout example.
    """

    def __init__(self, parent=None, margin: int = 0, hspacing: int = 4, vspacing: int = 4):
        super().__init__(parent)
        self._items: list = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):  # noqa: N802 (Qt override)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only: bool) -> int:
        margins = self.contentsMargins()
        x = rect.x() + margins.left()
        y = rect.y() + margins.top()
        right = rect.right() - margins.right()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            w, h = hint.width(), hint.height()
            next_x = x + w + self._hspace
            if next_x - self._hspace > right and line_height > 0:
                x = rect.x() + margins.left()
                y = y + line_height + self._vspace
                next_x = x + w + self._hspace
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), QSize(w, h)))
            x = next_x
            line_height = max(line_height, h)
        return y + line_height - rect.y() + margins.bottom()


class ViewerControlPanel(QWidget):
    """Right-dock control panel: tabbed controls with a shared progress bar.

    Tabs (a wrapping button bar across the top): Gene inspector, Cell segmentation, Cell type labels,
    Per cell statistics, Draw tissue annotations, Images, Dataset. A busy progress bar plus a stage
    label sit below the tabs and stay visible whatever tab is selected.
    """

    # Thread-safe status/progress updates: heavy work runs in napari
    # thread_workers that call these from a background thread. Routing every
    # update through a Qt signal marshals the actual widget mutation back onto
    # the GUI thread.
    status_message = Signal(str)
    progress_message = Signal(str, str, bool)  # (stage key, text, active)

    def __init__(
        self,
        datasets: list[str],
        gene_inspector_widget,
        cell_type_widget,
        load_callback,
        load_selected_labels_callback,
        unload_selected_shapes_callback,
        load_transcripts_callback,
        unload_transcripts_callback,
        load_selected_image_callback,
        load_all_images_callback,
        unload_selected_image_callback,
        load_cellpose_values_callback,
        remove_cellpose_values_callback,
        create_annotation_layers_callback,
        set_annotation_piece_callback,
        apply_annotation_piece_callback,
        snap_annotation_side_edges_callback,
        validate_annotation_callback,
        export_annotation_callback,
        create_object_annotation_callback,
        validate_object_annotations_callback,
        export_object_annotations_callback,
        load_object_annotations_callback,
        load_paired_callback,
        load_standalone_callback,
        initial_dataset: str | None = None,
    ):
        super().__init__()
        self._gene_inspector_widget = gene_inspector_widget
        self._cell_type_widget = cell_type_widget
        self._load_callback = load_callback
        self._load_selected_labels_callback = load_selected_labels_callback
        self._unload_selected_shapes_callback = unload_selected_shapes_callback
        self._load_transcripts_callback = load_transcripts_callback
        self._unload_transcripts_callback = unload_transcripts_callback
        self._load_selected_image_callback = load_selected_image_callback
        self._load_all_images_callback = load_all_images_callback
        self._unload_selected_image_callback = unload_selected_image_callback
        self._load_paired_callback = load_paired_callback
        self._load_standalone_callback = load_standalone_callback
        self._load_cellpose_values_callback = load_cellpose_values_callback
        self._remove_cellpose_values_callback = remove_cellpose_values_callback
        self._create_annotation_layers_callback = create_annotation_layers_callback
        self._set_annotation_piece_callback = set_annotation_piece_callback
        self._apply_annotation_piece_callback = apply_annotation_piece_callback
        self._snap_annotation_side_edges_callback = snap_annotation_side_edges_callback
        self._validate_annotation_callback = validate_annotation_callback
        self._export_annotation_callback = export_annotation_callback
        self._create_object_annotation_callback = create_object_annotation_callback
        self._validate_object_annotations_callback = validate_object_annotations_callback
        self._export_object_annotations_callback = export_object_annotations_callback
        self._load_object_annotations_callback = load_object_annotations_callback

        self._active_stages: dict[str, str] = {}
        self._loaded_shape_keys: set[str] = set()
        self._loaded_image_entries: set[tuple[str, str]] = set()
        self._LOADED_TEXT_COLOR = QColor(0x3E, 0xCF, 0x6B)  # green for loaded rows

        # -- Dataset loader (its own tab) -----------------------------------
        self._dataset_combo = QComboBox()
        self._dataset_combo.addItems(datasets)
        if initial_dataset in datasets:
            self._dataset_combo.setCurrentText(initial_dataset)
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        self._reload_button = QPushButton("Reload Dataset")
        self._reload_button.setEnabled(bool(datasets))
        self._reload_button.clicked.connect(self._on_reload_clicked)
        self._load_paired_button = QPushButton("Load new paired dataset")
        self._load_paired_button.clicked.connect(self._on_load_paired)
        self._load_standalone_merscope_button = QPushButton("Load new standalone MERSCOPE dataset")
        self._load_standalone_merscope_button.clicked.connect(self._on_load_standalone_merscope)
        self._load_standalone_xenium_button = QPushButton("Load new standalone Xenium dataset")
        self._load_standalone_xenium_button.clicked.connect(self._on_load_standalone_xenium)

        # -- Cell segmentation ----------------------------------------------
        self._shape_list = QListWidget()
        self._shape_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._shape_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._load_selected_labels_button = QPushButton("Load selected cell segmentation")
        self._load_selected_labels_button.clicked.connect(self._on_load_selected_labels)
        self._unload_selected_shapes_button = QPushButton("Unload selected cell segmentation")
        self._unload_selected_shapes_button.clicked.connect(self._on_unload_selected_shapes)

        # -- Per cell statistics --------------------------------------------
        self._cellpose_channel_combo = QComboBox()
        self._cellpose_statistic_combo = QComboBox()
        self._cellpose_colormap_combo = QComboBox()
        self._cellpose_colormap_combo.addItems(["viridis", "magma", "inferno", "plasma", "turbo", "gray"])
        self._load_cellpose_values_button = QPushButton("Load per-cell statistic overlay")
        self._load_cellpose_values_button.clicked.connect(self._on_load_cellpose_values)
        self._remove_cellpose_values_button = QPushButton("Unload per-cell statistic overlay")
        self._remove_cellpose_values_button.clicked.connect(self._on_remove_cellpose_values)
        self.set_cellpose_value_options([], [], enabled=False)

        # -- Transcripts (Gene inspector tab) -------------------------------
        self._load_transcripts_button = QPushButton("Load / reload transcripts")
        self._load_transcripts_button.clicked.connect(self._on_load_transcripts)
        self._unload_transcripts_button = QPushButton("Unload transcripts")
        self._unload_transcripts_button.clicked.connect(self._on_unload_transcripts)

        # -- Images ----------------------------------------------------------
        self._image_list = QListWidget()
        self._image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._image_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._load_selected_image_button = QPushButton("Load selected image(s)")
        self._load_selected_image_button.clicked.connect(self._on_load_selected_image)
        self._load_all_images_button = QPushButton("Load all images")
        self._load_all_images_button.clicked.connect(self._on_load_all_images)
        self._unload_selected_image_button = QPushButton("Unload selected image(s)")
        self._unload_selected_image_button.clicked.connect(self._on_unload_selected_image)

        # -- Draw tissue annotations (cortical depth) -----------------------
        self._create_annotations_button = QPushButton("Create Drawing Layers")
        self._create_annotations_button.clicked.connect(self._on_create_annotations)
        self._piece_combo = QComboBox()
        self._piece_combo.setEditable(True)
        self._piece_combo.addItem(CORTICAL_DEPTH_DEFAULT_PIECE_ID)
        self._piece_combo.currentTextChanged.connect(self._on_piece_changed)
        self._new_piece_button = QPushButton("New Piece")
        self._new_piece_button.clicked.connect(self._on_new_piece)
        self._apply_piece_button = QPushButton("Apply Piece To Selection")
        self._apply_piece_button.clicked.connect(self._on_apply_piece)
        self._snap_side_edges_button = QPushButton("Snap Boundaries To Edge")
        self._snap_side_edges_button.clicked.connect(self._on_snap_side_edges)
        self._validate_annotations_button = QPushButton("Validate Annotations")
        self._validate_annotations_button.clicked.connect(self._on_validate_annotations)
        self._export_annotations_button = QPushButton("Export Combined GeoJSON")
        self._export_annotations_button.clicked.connect(self._on_export_annotations)
        self._object_name_input = QLineEdit()
        self._object_name_input.setPlaceholderText("e.g. Amyloid plaques")
        self._create_object_annotations_button = QPushButton("Create Named Object Layer")
        self._create_object_annotations_button.clicked.connect(
            self._on_create_object_annotations
        )
        self._validate_object_annotations_button = QPushButton(
            "Validate Object Annotations"
        )
        self._validate_object_annotations_button.clicked.connect(
            self._on_validate_object_annotations
        )
        self._export_object_annotations_button = QPushButton("Export Object GeoJSON")
        self._export_object_annotations_button.clicked.connect(
            self._on_export_object_annotations
        )
        self._load_object_annotations_button = QPushButton("Load Object GeoJSON")
        self._load_object_annotations_button.clicked.connect(
            self._on_load_object_annotations
        )

        # -- Progress + status (shared, below the tabs) ---------------------
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate "busy" animation
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setWordWrap(True)
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        self.status_message.connect(self._on_status_message)
        self.progress_message.connect(self._on_progress_message)

        # -- Assemble tabs ---------------------------------------------------
        # A wrapping button bar (each button keeps its full-text width and wraps
        # onto extra rows on narrow docks) drives a QStackedWidget, so tab names
        # are never truncated regardless of dock width.
        self._tab_group = QButtonGroup(self)
        self._tab_group.setExclusive(True)
        self._tab_bar = QWidget()
        self._tab_bar_layout = FlowLayout(self._tab_bar, margin=0, hspacing=4, vspacing=4)
        _tab_bar_policy = self._tab_bar.sizePolicy()
        _tab_bar_policy.setHeightForWidth(True)
        _tab_bar_policy.setVerticalPolicy(QSizePolicy.Minimum)
        self._tab_bar.setSizePolicy(_tab_bar_policy)
        self._tab_stack = QStackedWidget()
        for title, builder in (
            ("Gene inspector", self._build_genes_tab),
            ("Cell segmentation", self._build_segmentation_tab),
            ("Cell type labels", self._build_cell_type_tab),
            ("Per cell statistics", self._build_statistics_tab),
            ("Draw tissue annotations", self._build_annotations_tab),
            ("Images", self._build_images_tab),
            ("Dataset loader", self._build_dataset_tab),
        ):
            self._add_tab(title, builder())
        initial_tab_index = 0 if datasets else 6
        initial_tab_button = self._tab_group.button(initial_tab_index)
        if initial_tab_button is not None:
            initial_tab_button.setChecked(True)
            self._tab_stack.setCurrentIndex(initial_tab_index)

        outer = QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(self._tab_bar)
        outer.addWidget(self._tab_stack, stretch=1)
        outer.addWidget(self._progress_bar)
        outer.addWidget(self._progress_label)
        outer.addWidget(self._status_label)
        self.setLayout(outer)
        self.setMinimumWidth(240)

    #: White background + black bold text so the tabs read as the main controls
    #: on napari's dark theme; the checked tab gets a blue border/underline.
    _TAB_BUTTON_STYLE = (
        "QPushButton {"
        " background: #ffffff; color: #000000; font-weight: bold;"
        " border: 1px solid #b0b0b0; border-radius: 4px; padding: 4px 10px; }"
        "QPushButton:hover { background: #eef2f7; }"
        "QPushButton:checked {"
        " background: #ffffff; color: #000000;"
        " border: 2px solid #1a73e8; border-bottom: 3px solid #1a73e8; }"
    )

    def _add_tab(self, title: str, widget: QWidget):
        """Add a wrapping tab button + its page to the stacked widget."""
        index = self._tab_stack.addWidget(widget)
        button = QPushButton(title)
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(self._TAB_BUTTON_STYLE)
        button.clicked.connect(lambda _checked=False, idx=index: self._tab_stack.setCurrentIndex(idx))
        self._tab_group.addButton(button, index)
        self._tab_bar_layout.addWidget(button)

    # -- tab builders -------------------------------------------------------
    def _scrollable(self, layout) -> QScrollArea:
        content = QWidget()
        content.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_genes_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(self._load_transcripts_button)
        row.addWidget(self._unload_transcripts_button)
        layout.addLayout(row)
        # The gene inspector widget provides the per-gene list + spot controls.
        if self._gene_inspector_widget is not None:
            layout.addWidget(self._gene_inspector_widget, stretch=1)
        else:
            layout.addStretch(1)
        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _build_cell_type_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        # The cell-type widget provides the segmentation/level controls, the
        # opacity slider, and the per-type tick list.
        if self._cell_type_widget is not None:
            layout.addWidget(self._cell_type_widget, stretch=1)
        else:
            layout.addStretch(1)
        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _build_segmentation_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Cell segmentations"))
        layout.addWidget(self._shape_list, stretch=1)
        layout.addWidget(self._load_selected_labels_button)
        layout.addWidget(self._unload_selected_shapes_button)
        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _build_statistics_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Per-cell statistics overlay"))
        layout.addWidget(QLabel("Channel"))
        layout.addWidget(self._cellpose_channel_combo)
        layout.addWidget(QLabel("Statistic"))
        layout.addWidget(self._cellpose_statistic_combo)
        layout.addWidget(QLabel("Colormap"))
        layout.addWidget(self._cellpose_colormap_combo)
        layout.addWidget(self._load_cellpose_values_button)
        layout.addWidget(self._remove_cellpose_values_button)
        layout.addStretch(1)
        return self._scrollable(layout)

    def _build_annotations_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Cortical Depth Annotations"))
        layout.addWidget(self._create_annotations_button)
        layout.addWidget(QLabel("Current Tissue Piece"))
        layout.addWidget(self._piece_combo)
        piece_row = QHBoxLayout()
        piece_row.addWidget(self._new_piece_button)
        piece_row.addWidget(self._apply_piece_button)
        layout.addLayout(piece_row)
        layout.addWidget(self._snap_side_edges_button)
        layout.addWidget(self._validate_annotations_button)
        layout.addWidget(self._export_annotations_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Distance-from-object Annotations"))
        layout.addWidget(QLabel("Object Set Name"))
        layout.addWidget(self._object_name_input)
        layout.addWidget(self._create_object_annotations_button)
        layout.addWidget(self._validate_object_annotations_button)
        layout.addWidget(self._export_object_annotations_button)
        layout.addWidget(self._load_object_annotations_button)
        layout.addStretch(1)
        return self._scrollable(layout)

    def _build_images_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Image channels"))
        layout.addWidget(self._image_list, stretch=1)
        layout.addWidget(self._load_selected_image_button)
        layout.addWidget(self._load_all_images_button)
        layout.addWidget(self._unload_selected_image_button)
        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _build_dataset_tab(self) -> QWidget:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Current dataset"))
        layout.addWidget(self._dataset_combo)
        layout.addWidget(self._reload_button)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        layout.addWidget(QLabel("Open a different dataset"))
        layout.addWidget(self._load_paired_button)
        layout.addWidget(self._load_standalone_merscope_button)
        layout.addWidget(self._load_standalone_xenium_button)
        layout.addStretch(1)
        return self._scrollable(layout)

    # -- public API ---------------------------------------------------------
    @property
    def current_dataset(self) -> str:
        return str(self._dataset_combo.currentText())

    def set_status(self, text: str):
        # Safe to call from any thread; the queued signal hops to the GUI thread.
        self.status_message.emit(str(text))

    def set_progress(self, key: str, text: str, active: bool):
        # Safe to call from any thread; the queued signal hops to the GUI thread.
        self.progress_message.emit(str(key), str(text), bool(active))

    def _on_status_message(self, text: str):
        self._status_label.setText(str(text))

    def _on_progress_message(self, key: str, text: str, active: bool):
        if active:
            self._active_stages[key] = text
        else:
            self._active_stages.pop(key, None)
        if self._active_stages:
            self._progress_bar.setVisible(True)
            self._progress_label.setText(" | ".join(v for v in self._active_stages.values() if v))
        else:
            self._progress_bar.setVisible(False)
            self._progress_label.setText("")

    def set_shape_keys(self, keys: list[str]):
        self._shape_list.clear()
        for key in keys:
            self._shape_list.addItem(QListWidgetItem(str(key)))
        self._apply_shape_key_colors()

    def set_loaded_shape_keys(self, keys: list[str]):
        """Recolor the segmentations that currently have a loaded layer (green)."""
        self._loaded_shape_keys = {str(k) for k in keys}
        self._apply_shape_key_colors()

    def _apply_shape_key_colors(self):
        for i in range(self._shape_list.count()):
            item = self._shape_list.item(i)
            if str(item.text()) in self._loaded_shape_keys:
                item.setForeground(self._LOADED_TEXT_COLOR)
            else:
                item.setData(Qt.ForegroundRole, None)  # reset to the theme default

    def set_image_entries(self, entries: list[tuple[str, str, str]]):
        """Populate the image list, one row per (image, channel).

        ``entries`` is a list of ``(display_label, image_key, channel)``. The
        image_key/channel pair is stashed on the item so selections map back to
        the exact channel layer.
        """
        self._image_list.clear()
        for label, image_key, channel in entries:
            item = QListWidgetItem(str(label))
            item.setData(Qt.UserRole, (str(image_key), str(channel)))
            self._image_list.addItem(item)
        self._apply_image_entry_colors()

    def set_loaded_image_entries(self, entries):
        """Recolor the image channels that currently have a loaded layer (green)."""
        self._loaded_image_entries = {(str(k), str(c)) for k, c in entries}
        self._apply_image_entry_colors()

    def _apply_image_entry_colors(self):
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            data = item.data(Qt.UserRole)
            key = (str(data[0]), str(data[1])) if data else None
            if key in self._loaded_image_entries:
                item.setForeground(self._LOADED_TEXT_COLOR)
            else:
                item.setData(Qt.ForegroundRole, None)  # reset to the theme default

    def set_cellpose_value_options(self, channels: list[str], statistics: list[str], enabled: bool):
        channel_current = str(self._cellpose_channel_combo.currentText())
        statistic_current = str(self._cellpose_statistic_combo.currentText())

        self._cellpose_channel_combo.clear()
        self._cellpose_statistic_combo.clear()
        self._cellpose_channel_combo.addItems([str(value) for value in channels])
        self._cellpose_statistic_combo.addItems([str(value) for value in statistics])

        if channel_current in channels:
            self._cellpose_channel_combo.setCurrentText(channel_current)
        elif "DAPI" in channels:
            self._cellpose_channel_combo.setCurrentText("DAPI")

        if statistic_current in statistics:
            self._cellpose_statistic_combo.setCurrentText(statistic_current)
        elif "median" in statistics:
            self._cellpose_statistic_combo.setCurrentText("median")

        enabled = bool(enabled) and len(channels) > 0 and len(statistics) > 0
        for widget in (
            self._cellpose_channel_combo,
            self._cellpose_statistic_combo,
            self._cellpose_colormap_combo,
            self._load_cellpose_values_button,
            self._remove_cellpose_values_button,
        ):
            widget.setEnabled(enabled)

    def selected_shape_keys(self) -> list[str]:
        return [str(item.text()) for item in self._shape_list.selectedItems()]

    def selected_image_entries(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for item in self._image_list.selectedItems():
            data = item.data(Qt.UserRole)
            if data:
                entries.append((str(data[0]), str(data[1])))
        return entries

    def current_annotation_piece(self) -> str:
        text = str(self._piece_combo.currentText()).strip()
        return text or CORTICAL_DEPTH_DEFAULT_PIECE_ID

    def set_datasets(self, names: list[str], initial: str | None = None):
        """Repopulate the current-dataset dropdown after loading new stores."""
        self._dataset_combo.blockSignals(True)
        try:
            self._dataset_combo.clear()
            self._dataset_combo.addItems([str(n) for n in names])
            if initial and initial in names:
                self._dataset_combo.setCurrentText(str(initial))
        finally:
            self._dataset_combo.blockSignals(False)
        self._reload_button.setEnabled(bool(names))

    # -- event handlers -----------------------------------------------------
    def _on_dataset_changed(self, text: str):
        if not text:
            return
        self._load_callback(str(text), False)

    def _on_reload_clicked(self):
        if self.current_dataset:
            self._load_callback(self.current_dataset, True)

    def _browse_zarr(self, title: str) -> str | None:
        path = QFileDialog.getExistingDirectory(self, title, "")
        return path or None

    def _on_load_paired(self):
        merscope_path = self._browse_zarr("Select the MERSCOPE .zarr store")
        if not merscope_path:
            return
        xenium_path = self._browse_zarr("Select the Xenium .zarr store")
        if not xenium_path:
            return
        self._load_paired_callback(merscope_path, xenium_path)

    def _on_load_standalone_merscope(self):
        path = self._browse_zarr("Select the MERSCOPE .zarr store")
        if not path:
            return
        self._load_standalone_callback("MERSCOPE", path)

    def _on_load_standalone_xenium(self):
        path = self._browse_zarr("Select the Xenium .zarr store")
        if not path:
            return
        self._load_standalone_callback("XENIUM", path)

    def _on_load_selected_labels(self):
        keys = self.selected_shape_keys()
        if not keys:
            self.set_status("No segmentation selected.")
            return
        self._load_selected_labels_callback(self.current_dataset, keys)

    def _on_unload_selected_shapes(self):
        keys = self.selected_shape_keys()
        if not keys:
            self.set_status("No segmentation selected.")
            return
        self._unload_selected_shapes_callback(self.current_dataset, keys)

    def _on_load_transcripts(self):
        self._load_transcripts_callback(self.current_dataset)

    def _on_unload_transcripts(self):
        self._unload_transcripts_callback(self.current_dataset)

    def _on_load_selected_image(self):
        entries = self.selected_image_entries()
        if not entries:
            self.set_status("No image channel selected.")
            return
        self._load_selected_image_callback(self.current_dataset, entries)

    def _on_load_all_images(self):
        self._load_all_images_callback(self.current_dataset)

    def _on_unload_selected_image(self):
        entries = self.selected_image_entries()
        if not entries:
            self.set_status("No image channel selected.")
            return
        self._unload_selected_image_callback(self.current_dataset, entries)

    def _on_load_cellpose_values(self):
        if not self._load_cellpose_values_button.isEnabled():
            self.set_status("Per-cell statistic overlay is not available for this dataset.")
            return
        self._load_cellpose_values_callback(
            self.current_dataset,
            str(self._cellpose_channel_combo.currentText()),
            str(self._cellpose_statistic_combo.currentText()),
            str(self._cellpose_colormap_combo.currentText()),
        )

    def _on_remove_cellpose_values(self):
        self._remove_cellpose_values_callback(self.current_dataset)

    def _on_create_annotations(self):
        self._create_annotation_layers_callback(self.current_dataset)
        self._set_annotation_piece_callback(self.current_dataset, self.current_annotation_piece())

    def _on_piece_changed(self, text: str):
        piece_id = str(text).strip()
        if piece_id:
            self._set_annotation_piece_callback(self.current_dataset, piece_id)

    def _on_new_piece(self):
        existing = {str(self._piece_combo.itemText(idx)) for idx in range(self._piece_combo.count())}
        next_idx = 1
        while f"piece_{next_idx}" in existing:
            next_idx += 1
        piece_id = f"piece_{next_idx}"
        self._piece_combo.addItem(piece_id)
        self._piece_combo.setCurrentText(piece_id)

    def _on_apply_piece(self):
        piece_id = self.current_annotation_piece()
        try:
            changed = self._apply_annotation_piece_callback(self.current_dataset, piece_id)
        except Exception as exc:
            self.set_status(f"Apply piece failed: {exc}")
            QMessageBox.warning(self, "Apply Piece To Selection", str(exc))
            return
        self.set_status(f"Applied {piece_id} to {changed} selected annotation shape(s).")

    def _on_snap_side_edges(self):
        try:
            snapped = self._snap_annotation_side_edges_callback(self.current_dataset)
        except Exception as exc:
            self.set_status(f"Boundary snapping failed: {exc}")
            QMessageBox.warning(self, "Snap Boundaries To Edge", str(exc))
            return
        if snapped:
            self.set_status("Pial/WM endpoints snapped to the tissue edge.")

    def _on_validate_annotations(self):
        try:
            result = self._validate_annotation_callback(self.current_dataset)
        except Exception as exc:
            self.set_status(f"Annotation validation failed: {exc}")
            QMessageBox.warning(self, "Validate Annotations", str(exc))
            return
        self._show_annotation_validation_result("Validate Annotations", result)

    def _on_export_annotations(self):
        default_name = f"{self.current_dataset.lower()}_cortical_depth_annotations.geojson"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Cortical Depth Annotations",
            default_name,
            "GeoJSON (*.geojson *.json);;All files (*)",
        )
        if not path:
            return
        try:
            result = self._export_annotation_callback(self.current_dataset, Path(path))
        except Exception as exc:
            self.set_status(f"Annotation export failed: {exc}")
            QMessageBox.warning(self, "Export Combined GeoJSON", str(exc))
            return
        if not getattr(result, "ok", False):
            errors = "\n".join(f"- {message}" for message in getattr(result, "errors", ()))
            choice = QMessageBox.question(
                self,
                "Export Combined GeoJSON",
                (
                    "Validation failed, so this file may not run in MerXen.\n\n"
                    f"{errors}\n\n"
                    "Save the current annotations anyway for debugging?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice == QMessageBox.Yes:
                try:
                    result = self._export_annotation_callback(self.current_dataset, Path(path), allow_invalid=True)
                except Exception as exc:
                    self.set_status(f"Debug annotation export failed: {exc}")
                    QMessageBox.warning(self, "Export Combined GeoJSON", str(exc))
                    return
                self.set_status(f"Saved validation-failed annotations for debugging: {path}")
                QMessageBox.information(
                    self,
                    "Export Combined GeoJSON",
                    (
                        "Saved annotations for debugging.\n\n"
                        f"{path}\n\n"
                        "The file includes napari_compare_validation with the validation errors."
                    ),
                )
                return
        self._show_annotation_validation_result("Export Combined GeoJSON", result, export_path=Path(path))

    def _show_annotation_validation_result(self, title: str, result, export_path: Path | None = None):
        errors = list(getattr(result, "errors", ()))
        warnings_ = list(getattr(result, "warnings", ()))
        if errors:
            text = "Export blocked:\n\n" + "\n".join(f"- {message}" for message in errors)
            if warnings_:
                text += "\n\nWarnings:\n" + "\n".join(f"- {message}" for message in warnings_)
            QMessageBox.warning(self, title, text)
            return

        text = "Annotations are valid."
        if export_path is not None:
            text = f"Exported:\n{export_path}"
        if warnings_:
            text += "\n\nWarnings:\n" + "\n".join(f"- {message}" for message in warnings_)
            QMessageBox.information(self, title, text)
            return
        QMessageBox.information(self, title, text)

    def _on_create_object_annotations(self):
        object_name = self._object_name_input.text().strip()
        if not object_name:
            QMessageBox.warning(
                self,
                "Create Named Object Layer",
                "Enter an object set name, such as 'Amyloid plaques'.",
            )
            return
        try:
            self._create_object_annotation_callback(
                self.current_dataset,
                object_name,
            )
        except Exception as exc:
            self.set_status(f"Object-layer creation failed: {exc}")
            QMessageBox.warning(self, "Create Named Object Layer", str(exc))

    def _on_validate_object_annotations(self):
        try:
            result = self._validate_object_annotations_callback(self.current_dataset)
        except Exception as exc:
            self.set_status(f"Object annotation validation failed: {exc}")
            QMessageBox.warning(self, "Validate Object Annotations", str(exc))
            return
        self._show_object_annotation_result("Validate Object Annotations", result)

    def _on_export_object_annotations(self):
        default_name = (
            f"{self.current_dataset.lower()}_distance_object_annotations.geojson"
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Object Annotations",
            default_name,
            "GeoJSON (*.geojson *.json);;All files (*)",
        )
        if not path:
            return
        try:
            result = self._export_object_annotations_callback(
                self.current_dataset,
                Path(path),
            )
        except Exception as exc:
            self.set_status(f"Object annotation export failed: {exc}")
            QMessageBox.warning(self, "Export Object GeoJSON", str(exc))
            return
        self._show_object_annotation_result(
            "Export Object GeoJSON",
            result,
            export_path=Path(path),
        )

    def _on_load_object_annotations(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Object Annotations",
            "",
            "GeoJSON (*.geojson *.json);;All files (*)",
        )
        if not path:
            return
        try:
            self._load_object_annotations_callback(
                self.current_dataset,
                Path(path),
            )
        except Exception as exc:
            self.set_status(f"Object annotation load failed: {exc}")
            QMessageBox.warning(self, "Load Object GeoJSON", str(exc))

    def _show_object_annotation_result(
        self,
        title: str,
        result,
        export_path: Path | None = None,
    ):
        errors = list(getattr(result, "errors", ()))
        warnings_ = list(getattr(result, "warnings", ()))
        if errors:
            text = "Object annotations are invalid:\n\n" + "\n".join(
                f"- {message}" for message in errors
            )
            QMessageBox.warning(self, title, text)
            return
        text = "Object annotations are valid."
        if export_path is not None:
            text = f"Exported:\n{export_path}"
        if warnings_:
            text += "\n\nWarnings:\n" + "\n".join(
                f"- {message}" for message in warnings_
            )
        QMessageBox.information(self, title, text)


def _rgba_to_qcolor(rgba) -> QColor:
    r, g, b, a = (float(v) for v in rgba)
    color = QColor()
    color.setRgbF(max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)), max(0.0, min(1.0, a)))
    return color


def make_gene_marker_pixmap(rgba, symbol: str, px: int = 20) -> QPixmap:
    """Draw a napari point ``symbol`` in ``rgba`` as a small QPixmap legend icon."""
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        color = _rgba_to_qcolor(rgba)
        m = 2.0
        lo, hi = m, px - m
        cx = cy = px / 2.0
        r = (px - 2.0 * m) / 2.0
        fill = QBrush(color)
        line = QPen(color)
        line.setWidthF(max(1.6, px * 0.14))
        line.setCapStyle(Qt.RoundCap)
        no_pen = QPen(Qt.NoPen)
        sym = str(symbol)

        def poly(points):
            return QPolygonF([QPointF(x, y) for x, y in points])

        if sym == "ring":
            painter.setBrush(Qt.NoBrush)
            ring_pen = QPen(color)
            ring_pen.setWidthF(max(1.6, px * 0.14))
            painter.setPen(ring_pen)
            painter.drawEllipse(QPointF(cx, cy), r * 0.9, r * 0.9)
        elif sym == "square":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawRect(QRectF(lo, lo, hi - lo, hi - lo))
        elif sym == "diamond":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawPolygon(poly([(cx, lo), (hi, cy), (cx, hi), (lo, cy)]))
        elif sym == "cross":
            painter.setPen(line)
            painter.drawLine(QPointF(cx, lo), QPointF(cx, hi))
            painter.drawLine(QPointF(lo, cy), QPointF(hi, cy))
        elif sym == "x":
            painter.setPen(line)
            painter.drawLine(QPointF(lo, lo), QPointF(hi, hi))
            painter.drawLine(QPointF(lo, hi), QPointF(hi, lo))
        elif sym == "triangle_up":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawPolygon(poly([(cx, lo), (hi, hi), (lo, hi)]))
        elif sym == "triangle_down":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawPolygon(poly([(lo, lo), (hi, lo), (cx, hi)]))
        elif sym == "star":
            import math

            pts = []
            for k in range(10):
                ang = -math.pi / 2 + k * math.pi / 5
                rad = r if k % 2 == 0 else r * 0.42
                pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawPolygon(poly(pts))
        elif sym in ("arrow", "tailed_arrow"):
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawPolygon(poly([(cx, lo), (hi, cy), (cx + r * 0.35, cy), (cx + r * 0.35, hi), (cx - r * 0.35, hi), (cx - r * 0.35, cy), (lo, cy)]))
        elif sym == "hbar":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawRect(QRectF(lo, cy - r * 0.35, hi - lo, r * 0.7))
        elif sym == "vbar":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawRect(QRectF(cx - r * 0.35, lo, r * 0.7, hi - lo))
        elif sym == "clobber":
            painter.setPen(no_pen)
            painter.setBrush(fill)
            cr = r * 0.5
            painter.drawEllipse(QPointF(cx, lo + cr), cr, cr)
            painter.drawEllipse(QPointF(lo + cr, hi - cr), cr, cr)
            painter.drawEllipse(QPointF(hi - cr, hi - cr), cr, cr)
        else:  # disc (default)
            painter.setPen(no_pen)
            painter.setBrush(fill)
            painter.drawEllipse(QPointF(cx, cy), r, r)
    finally:
        painter.end()
    return pm


class _GroupHeaderLabel(QLabel):
    """A gene-group heading that emits ``clicked`` when pressed."""

    clicked = Signal()

    def mousePressEvent(self, event):  # noqa: N802 (Qt override)
        self.clicked.emit()
        super().mousePressEvent(event)


class GeneInspectorWidget(QWidget):
    """Right-dock panel listing every gene with a colour+shape marker and toggle."""

    status_message = Signal(str)

    # Slider integer <-> world-micron spot size at GENE_SPOT_SIZE_STEP (0.1 um)
    # granularity, over [GENE_SPOT_SIZE_MIN, GENE_SPOT_SIZE_MAX].
    _SPOT_SLIDER_SCALE = 1.0 / GENE_SPOT_SIZE_STEP
    _SPOT_SLIDER_MIN = int(round(GENE_SPOT_SIZE_MIN / GENE_SPOT_SIZE_STEP))
    _SPOT_SLIDER_MAX = int(round(GENE_SPOT_SIZE_MAX / GENE_SPOT_SIZE_STEP))

    def __init__(
        self,
        close_callback,
        set_gene_visible_callback,
        set_all_genes_callback,
        set_spot_size_callback,
        set_hide_background_callback,
        set_show_controls_callback,
        set_hide_assigned_callback=None,
        set_colour_by_assignment_callback=None,
        set_ordering_callback=None,
        set_genes_visible_callback=None,
        spot_size: float = DEFAULT_GENE_SPOT_SIZE,
        hide_assigned: bool = False,
        hide_background: bool = False,
        show_controls: bool = False,
        colour_by_assignment: bool = False,
    ):
        super().__init__()
        self._close_callback = close_callback
        self._set_gene_visible_callback = set_gene_visible_callback
        self._set_all_genes_callback = set_all_genes_callback
        self._set_spot_size_callback = set_spot_size_callback
        self._set_hide_background_callback = set_hide_background_callback
        self._set_hide_assigned_callback = set_hide_assigned_callback
        self._set_colour_by_assignment_callback = set_colour_by_assignment_callback
        self._set_show_controls_callback = set_show_controls_callback
        self._set_ordering_callback = set_ordering_callback
        self._set_genes_visible_callback = set_genes_visible_callback

        self._dataset: str | None = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._items: dict[str, QListWidgetItem] = {}
        #: (header item, group title, genes-in-group) so filtering can hide a
        #: header once all of its genes are filtered/hidden.
        self._headers: list[tuple[QListWidgetItem, str, list[str]]] = []
        self._control_genes: set[str] = set()
        self._ordering = "coarse"
        self._suppress = False

        self._title = QLabel("Gene Inspector")
        self._status_label = QLabel(
            "Transcripts load automatically. Use “Load / reload transcripts” to rebuild the gene panel."
        )
        self._status_label.setWordWrap(True)
        self.status_message.connect(self._status_label.setText)

        self._close_button = QPushButton("Unload transcripts")
        self._close_button.clicked.connect(self._on_close)

        self._spot_slider = QSlider(Qt.Horizontal)
        self._spot_slider.setRange(self._SPOT_SLIDER_MIN, self._SPOT_SLIDER_MAX)
        self._spot_slider.setValue(self._spot_size_to_slider(spot_size))
        # Only emit (and rebuild) on release; the spin box mirrors the value live
        # while dragging. This avoids an O(points) size re-upload per drag step.
        self._spot_slider.setTracking(False)
        self._spot_slider.valueChanged.connect(self._on_spot_slider_changed)
        self._spot_slider.sliderMoved.connect(self._on_spot_slider_moved)
        # Editable numeric entry: clamps to [MIN, MAX] and rounds to 0.1 um,
        # rejecting out-of-range / non-numeric input.
        self._spot_spin = QDoubleSpinBox()
        self._spot_spin.setRange(GENE_SPOT_SIZE_MIN, GENE_SPOT_SIZE_MAX)
        self._spot_spin.setSingleStep(GENE_SPOT_SIZE_STEP)
        self._spot_spin.setDecimals(1)
        self._spot_spin.setSuffix(" µm")
        self._spot_spin.setKeyboardTracking(False)  # emit only on Enter/focus-out
        self._spot_spin.setValue(float(spot_size))
        self._spot_spin.valueChanged.connect(self._on_spot_spin_changed)

        self._show_all_check = QCheckBox("Show all genes")
        self._show_all_check.setTristate(True)
        self._show_all_check.clicked.connect(self._on_show_all_clicked)
        self._hide_assigned_check = QCheckBox("Hide assigned spots")
        self._hide_assigned_check.setChecked(bool(hide_assigned))
        self._hide_assigned_check.toggled.connect(self._on_hide_assigned_toggled)
        self._hide_bg_check = QCheckBox("Hide unassigned spots")
        self._hide_bg_check.setChecked(bool(hide_background))
        self._hide_bg_check.toggled.connect(self._on_hide_background_toggled)
        self._colour_by_assignment_check = QCheckBox("Colour transcripts by assigned/unassigned")
        self._colour_by_assignment_check.setChecked(bool(colour_by_assignment))
        self._colour_by_assignment_check.toggled.connect(self._on_colour_by_assignment_toggled)
        self._show_controls_check = QCheckBox("Show control / blank probes")
        self._show_controls_check.setChecked(bool(show_controls))
        self._show_controls_check.toggled.connect(self._on_show_controls_toggled)

        # -- Ordering of the gene list (grouped by cell type, or flat A–Z) ----
        self._order_label = QLabel("Order genes by")
        self._order_group = QButtonGroup(self)
        self._order_group.setExclusive(True)
        self._order_coarse_btn = QPushButton("Broad cell type")
        self._order_fine_btn = QPushButton("Fine cell type")
        self._order_alpha_btn = QPushButton("A–Z")
        self._order_buttons = {
            "coarse": self._order_coarse_btn,
            "fine": self._order_fine_btn,
            "alphabetical": self._order_alpha_btn,
        }
        for kind, button in self._order_buttons.items():
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, k=kind: self._on_ordering_clicked(k))
            self._order_group.addButton(button)
        self._order_coarse_btn.setChecked(True)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter genes…")
        self._filter_edit.textChanged.connect(self._apply_filter)

        self._gene_list = QListWidget()
        self._gene_list.setSelectionMode(QAbstractItemView.NoSelection)
        # Rows are not a uniform height: the group headings are taller than the
        # gene rows, so each item must be sized from its own widget.
        self._gene_list.setUniformItemSizes(False)
        self._gene_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._title)
        root.addWidget(self._close_button)
        spot_row = QHBoxLayout()
        spot_row.addWidget(QLabel("Spot size"))
        spot_row.addWidget(self._spot_slider)
        spot_row.addWidget(self._spot_spin)
        root.addLayout(spot_row)
        root.addWidget(self._show_all_check)
        root.addWidget(self._hide_assigned_check)
        root.addWidget(self._hide_bg_check)
        root.addWidget(self._colour_by_assignment_check)
        root.addWidget(self._show_controls_check)
        root.addWidget(self._order_label)
        order_row = QHBoxLayout()
        order_row.addWidget(self._order_coarse_btn)
        order_row.addWidget(self._order_fine_btn)
        order_row.addWidget(self._order_alpha_btn)
        root.addLayout(order_row)
        root.addWidget(self._filter_edit)
        root.addWidget(self._gene_list, stretch=1)
        root.addWidget(self._status_label)
        self.setLayout(root)
        self.setMinimumWidth(240)

    # -- public API ---------------------------------------------------------
    def set_status(self, text: str):
        self.status_message.emit(str(text))

    @property
    def dataset(self) -> str | None:
        return self._dataset

    def clear(self):
        self._suppress = True
        try:
            self._gene_list.clear()
            self._checkboxes.clear()
            self._items.clear()
            self._headers = []
            self._control_genes = set()
            self._dataset = None
            self._show_all_check.setCheckState(Qt.Unchecked)
        finally:
            self._suppress = False

    def spot_size(self) -> float:
        return self._slider_to_spot_size(self._spot_slider.value())

    def set_ordering_available(self, available: bool, ordering: str = "coarse"):
        """Enable/disable the broad/fine ordering buttons (off with no reference)."""
        self._ordering = str(ordering)
        for kind, button in self._order_buttons.items():
            button.blockSignals(True)
            # A–Z always works; broad/fine only when a marker reference exists.
            button.setEnabled(bool(available) or kind == "alphabetical")
            button.setChecked(kind == self._ordering)
            button.blockSignals(False)
        self._order_label.setVisible(True)

    def populate(
        self,
        dataset: str,
        layout: list[tuple],
        gene_visuals: dict,
        gene_counts: dict,
        control_genes: set,
        enabled_genes: set,
        labels: dict,
        ordering: str,
        hide_assigned: bool,
        hide_background: bool,
        show_controls: bool,
        spot_size: float,
        colour_by_assignment: bool = False,
    ):
        """Build the gene rows for ``dataset`` (called on the GUI thread).

        ``layout`` is an ordered list of ``("header", title, rgba)`` and
        ``("gene", name)`` entries. In A–Z (flat) mode there are no headers and
        each row is annotated with its ``labels[name]`` broad/fine text.
        """
        self._suppress = True
        try:
            self._gene_list.clear()
            self._checkboxes.clear()
            self._items.clear()
            self._headers = []
            self._dataset = str(dataset)
            self._ordering = str(ordering)
            self._control_genes = set(control_genes)
            self._hide_assigned_check.setChecked(bool(hide_assigned))
            self._hide_bg_check.setChecked(bool(hide_background))
            self._colour_by_assignment_check.setChecked(bool(colour_by_assignment))
            self._show_controls_check.setChecked(bool(show_controls))
            self._set_spot_widgets(float(spot_size))

            pending_header: tuple | None = None
            header_genes: list[str] = []

            def _finalize_header():
                if pending_header is None:
                    return
                item, widget, title = pending_header
                genes = list(header_genes)
                self._headers.append((item, title, genes))
                # Clicking a heading selects/deselects every gene beneath it.
                widget.clicked.connect(lambda gs=genes: self._on_header_clicked(gs))

            for entry in layout:
                if entry[0] == "header":
                    _finalize_header()
                    _, title, rgba = entry
                    item = QListWidgetItem()
                    item.setFlags(Qt.NoItemFlags)
                    widget = self._make_header_widget(str(title), rgba)
                    # Reserve extra height so the heading isn't crowded by the gene
                    # rows above/below (the stylesheet top margin isn't counted in
                    # the label's own sizeHint).
                    hint = widget.sizeHint()
                    item.setSizeHint(QSize(hint.width(), hint.height() + 14))
                    self._gene_list.addItem(item)
                    self._gene_list.setItemWidget(item, widget)
                    pending_header = (item, widget, str(title))
                    header_genes = []
                    continue

                gene = str(entry[1])
                visual = gene_visuals.get(gene)
                symbol = getattr(visual, "symbol", "disc")
                rgba = getattr(visual, "rgba", (1.0, 1.0, 1.0, 1.0))
                count = int(gene_counts.get(gene, 0))
                text = f"{gene}    ({count:,})"
                if str(ordering) == "alphabetical" and labels.get(gene):
                    text = f"{text}    ·  {labels[gene]}"
                check = QCheckBox(text)
                check.setIcon(QIcon(make_gene_marker_pixmap(rgba, symbol, px=22)))
                check.setChecked(gene in enabled_genes)
                check.toggled.connect(lambda on, g=gene: self._on_gene_toggled(g, on))
                item = QListWidgetItem()
                item.setSizeHint(check.sizeHint())
                self._gene_list.addItem(item)
                self._gene_list.setItemWidget(item, check)
                self._checkboxes[gene] = check
                self._items[gene] = item
                header_genes.append(gene)

            _finalize_header()

            self._refresh_row_visibility()
            self._refresh_show_all_state()
        finally:
            self._suppress = False

    def _make_header_widget(self, title: str, rgba) -> QLabel:
        """A clickable, bold group heading with a colour cue from the group's shade.

        Clicking the heading toggles every (currently visible) gene under it.
        """
        label = _GroupHeaderLabel(title)
        label.setCursor(Qt.PointingHandCursor)
        label.setToolTip("Click to show / hide all genes in this group")
        r, g, b = (int(round(255 * float(c))) for c in (rgba[:3] if rgba else (0.6, 0.6, 0.6)))
        label.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 13pt; color: #f0f0f0; margin-top: 6px;"
            f" padding: 4px 4px 4px 8px; border-left: 7px solid rgb({r}, {g}, {b}); }}"
            "QLabel:hover { color: #ffffff; background: rgba(255, 255, 255, 20); }"
        )
        return label

    # -- helpers ------------------------------------------------------------
    def _spot_size_to_slider(self, um: float) -> int:
        raw = int(round(float(um) * self._SPOT_SLIDER_SCALE))
        return max(self._SPOT_SLIDER_MIN, min(self._SPOT_SLIDER_MAX, raw))

    def _slider_to_spot_size(self, value: int) -> float:
        return round(float(value) / self._SPOT_SLIDER_SCALE, 1)

    def _set_spot_widgets(self, um: float):
        """Set slider + spin box to ``um`` without emitting callbacks."""
        um = round(min(GENE_SPOT_SIZE_MAX, max(GENE_SPOT_SIZE_MIN, float(um))), 1)
        self._spot_slider.blockSignals(True)
        self._spot_spin.blockSignals(True)
        self._spot_slider.setValue(self._spot_size_to_slider(um))
        self._spot_spin.setValue(um)
        self._spot_slider.blockSignals(False)
        self._spot_spin.blockSignals(False)

    def _eligible_genes(self) -> list[str]:
        show_controls = self._show_controls_check.isChecked()
        return [g for g in self._checkboxes if show_controls or g not in self._control_genes]

    def _refresh_row_visibility(self):
        show_controls = self._show_controls_check.isChecked()
        needle = self._filter_edit.text().strip().lower()
        for gene, item in self._items.items():
            hidden = (gene in self._control_genes and not show_controls) or (
                bool(needle) and needle not in gene.lower()
            )
            item.setHidden(hidden)
        # A group heading disappears once every gene beneath it is hidden.
        for header_item, _title, genes in self._headers:
            any_visible = any(
                g in self._items and not self._items[g].isHidden() for g in genes
            )
            header_item.setHidden(not any_visible)

    def _refresh_show_all_state(self):
        eligible = self._eligible_genes()
        checked = sum(1 for g in eligible if self._checkboxes[g].isChecked())
        self._show_all_check.blockSignals(True)
        if not eligible or checked == 0:
            self._show_all_check.setCheckState(Qt.Unchecked)
        elif checked == len(eligible):
            self._show_all_check.setCheckState(Qt.Checked)
        else:
            self._show_all_check.setCheckState(Qt.PartiallyChecked)
        self._show_all_check.blockSignals(False)

    # -- event handlers -----------------------------------------------------
    def _on_close(self):
        if self._dataset is not None:
            self._close_callback(self._dataset)

    def _on_gene_toggled(self, gene: str, on: bool):
        if self._suppress or self._dataset is None:
            return
        self._set_gene_visible_callback(self._dataset, gene, bool(on))
        self._refresh_show_all_state()

    def _on_show_all_clicked(self, _checked=None):
        if self._suppress or self._dataset is None:
            return
        # A click on a tri-state box cycles to fully on unless already fully on.
        turn_on = self._show_all_check.checkState() != Qt.Unchecked
        self._suppress = True
        try:
            for gene in self._eligible_genes():
                self._checkboxes[gene].setChecked(turn_on)
        finally:
            self._suppress = False
        self._refresh_show_all_state()
        self._set_all_genes_callback(self._dataset, turn_on)

    def _on_hide_background_toggled(self, on: bool):
        if self._suppress or self._dataset is None:
            return
        self._set_hide_background_callback(self._dataset, bool(on))

    def _on_hide_assigned_toggled(self, on: bool):
        if self._suppress or self._dataset is None:
            return
        if self._set_hide_assigned_callback is not None:
            self._set_hide_assigned_callback(self._dataset, bool(on))

    def _on_colour_by_assignment_toggled(self, on: bool):
        if self._suppress or self._dataset is None:
            return
        if self._set_colour_by_assignment_callback is not None:
            self._set_colour_by_assignment_callback(self._dataset, bool(on))

    def _on_show_controls_toggled(self, on: bool):
        self._refresh_row_visibility()
        self._refresh_show_all_state()
        if self._suppress or self._dataset is None:
            return
        self._set_show_controls_callback(self._dataset, bool(on))

    def _on_ordering_clicked(self, kind: str):
        if kind == self._ordering:
            return
        self._ordering = str(kind)
        if self._suppress or self._dataset is None or self._set_ordering_callback is None:
            return
        self._set_ordering_callback(self._dataset, str(kind))

    def _on_header_clicked(self, genes: list[str]):
        """Toggle every currently-visible gene under a group heading at once."""
        if self._suppress or self._dataset is None or self._set_genes_visible_callback is None:
            return
        visible = [
            g for g in genes
            if g in self._checkboxes and g in self._items and not self._items[g].isHidden()
        ]
        if not visible:
            return
        # If any visible gene is off, turn the group on; otherwise turn it all off.
        turn_on = not all(self._checkboxes[g].isChecked() for g in visible)
        self._suppress = True
        try:
            for g in visible:
                self._checkboxes[g].setChecked(turn_on)
        finally:
            self._suppress = False
        self._refresh_show_all_state()
        self._set_genes_visible_callback(self._dataset, visible, turn_on)

    def _emit_spot_size(self, um: float):
        self._set_spot_widgets(um)
        if self._suppress or self._dataset is None:
            return
        self._set_spot_size_callback(self._dataset, round(float(um), 1))

    def _on_spot_slider_changed(self, value: int):
        # Fires on release (tracking off) or programmatic setValue.
        self._emit_spot_size(self._slider_to_spot_size(value))

    def _on_spot_slider_moved(self, value: int):
        # Live mirror into the spin box while dragging (no callback).
        self._spot_spin.blockSignals(True)
        self._spot_spin.setValue(self._slider_to_spot_size(value))
        self._spot_spin.blockSignals(False)

    def _on_spot_spin_changed(self, um: float):
        self._emit_spot_size(float(um))

    def _apply_filter(self, _text: str):
        self._refresh_row_visibility()


class CellTypeWidget(QWidget):
    """Right-dock panel that fills segmentation masks by broad/fine cell type.

    Mirrors :class:`GeneInspectorWidget`: a grouped, ticked list (each cell type
    with a colour swatch, fine subtypes grouped under their broad class) plus a
    Broad/Fine level toggle, a ProSeg/Cellpose segmentation selector, and a mask
    opacity slider. Toggling a type shows/hides those cells; the controller
    recolours the labels layer.
    """

    status_message = Signal(str)

    def __init__(
        self,
        close_callback,
        set_segmentation_callback,
        set_kind_callback,
        set_type_visible_callback,
        set_types_visible_callback,
        set_all_types_callback,
        set_opacity_callback,
        opacity: float = 0.95,
    ):
        super().__init__()
        self._close_callback = close_callback
        self._set_segmentation_callback = set_segmentation_callback
        self._set_kind_callback = set_kind_callback
        self._set_type_visible_callback = set_type_visible_callback
        self._set_types_visible_callback = set_types_visible_callback
        self._set_all_types_callback = set_all_types_callback
        self._set_opacity_callback = set_opacity_callback

        self._dataset: str | None = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._headers: list[tuple[QListWidgetItem, str, list[str]]] = []
        # No level is active until the user clicks Broad or Fine (those buttons
        # are the "colour the masks" trigger), so start with none checked.
        self._kind = ""
        self._segmentation = "proseg"
        self._suppress = False

        self._title = QLabel("Cell type labels")
        self._status_label = QLabel(
            "Fills cell masks by their stored cell-type annotation. "
            "Choose a segmentation and level below."
        )
        self._status_label.setWordWrap(True)
        self.status_message.connect(self._status_label.setText)

        self._close_button = QPushButton("Remove cell-type colouring")
        self._close_button.clicked.connect(self._on_close)

        # -- Segmentation whose masks are filled ------------------------------
        self._seg_label = QLabel("Fill segmentation")
        self._seg_group = QButtonGroup(self)
        self._seg_group.setExclusive(True)
        self._seg_proseg_btn = QPushButton("ProSeg")
        self._seg_cellpose_btn = QPushButton("Cellpose")
        self._seg_buttons = {"proseg": self._seg_proseg_btn, "cellpose": self._seg_cellpose_btn}
        for seg, button in self._seg_buttons.items():
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, s=seg: self._on_segmentation_clicked(s))
            self._seg_group.addButton(button)
        self._seg_proseg_btn.setChecked(True)

        # -- Broad vs fine cell-type level ------------------------------------
        self._level_label = QLabel("Colour by")
        self._level_group = QButtonGroup(self)
        self._level_group.setExclusive(True)
        self._level_broad_btn = QPushButton("Broad cell type")
        self._level_fine_btn = QPushButton("Fine cell type")
        self._level_buttons = {"broad": self._level_broad_btn, "fine": self._level_fine_btn}
        # Exclusive, but allow starting with none checked until the user picks.
        self._level_group.setExclusive(False)
        for kind, button in self._level_buttons.items():
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, k=kind: self._on_kind_clicked(k))
            self._level_group.addButton(button)

        # -- Mask opacity (0-100 % -> 0..1) -----------------------------------
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(int(round(float(opacity) * 100)))
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self._opacity_value = QLabel(f"{int(round(float(opacity) * 100))}%")

        self._show_all_check = QCheckBox("Show all cell types")
        self._show_all_check.setTristate(True)
        self._show_all_check.clicked.connect(self._on_show_all_clicked)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter cell types…")
        self._filter_edit.textChanged.connect(self._apply_filter)

        self._type_list = QListWidget()
        self._type_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._type_list.setUniformItemSizes(False)
        self._type_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self._title)
        root.addWidget(self._close_button)
        root.addWidget(self._seg_label)
        seg_row = QHBoxLayout()
        seg_row.addWidget(self._seg_proseg_btn)
        seg_row.addWidget(self._seg_cellpose_btn)
        root.addLayout(seg_row)
        root.addWidget(self._level_label)
        level_row = QHBoxLayout()
        level_row.addWidget(self._level_broad_btn)
        level_row.addWidget(self._level_fine_btn)
        root.addLayout(level_row)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Mask opacity"))
        opacity_row.addWidget(self._opacity_slider)
        opacity_row.addWidget(self._opacity_value)
        root.addLayout(opacity_row)
        root.addWidget(self._show_all_check)
        root.addWidget(self._filter_edit)
        root.addWidget(self._type_list, stretch=1)
        root.addWidget(self._status_label)
        self.setLayout(root)
        self.setMinimumWidth(240)

    # -- public API ---------------------------------------------------------
    def set_status(self, text: str):
        self.status_message.emit(str(text))

    @property
    def dataset(self) -> str | None:
        return self._dataset

    def opacity(self) -> float:
        return round(self._opacity_slider.value() / 100.0, 2)

    def set_dataset(self, dataset: str | None):
        """Point the panel at ``dataset`` and clear its rows (no level active)."""
        self._suppress = True
        try:
            self._type_list.clear()
            self._checkboxes.clear()
            self._items.clear()
            self._headers = []
            self._dataset = None if dataset is None else str(dataset)
            self._kind = ""
            self._show_all_check.setCheckState(Qt.Unchecked)
            for button in self._level_buttons.values():
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        finally:
            self._suppress = False

    def set_segmentation_available(self, available: dict[str, bool]):
        """Enable a segmentation button only when it has stored annotations.

        Keeps the current selection if it is still available, else falls back to
        the first available segmentation.
        """
        if not available.get(self._segmentation, False):
            self._segmentation = next((s for s, ok in available.items() if ok), self._segmentation)
        for seg, button in self._seg_buttons.items():
            button.blockSignals(True)
            button.setEnabled(bool(available.get(seg, False)))
            button.setChecked(seg == self._segmentation)
            button.blockSignals(False)

    def clear(self):
        self._suppress = True
        try:
            self._type_list.clear()
            self._checkboxes.clear()
            self._items.clear()
            self._headers = []
            self._dataset = None
            self._show_all_check.setCheckState(Qt.Unchecked)
        finally:
            self._suppress = False

    def populate(
        self,
        dataset: str,
        layout: list[tuple],
        colors: dict,
        counts: dict,
        enabled: set,
        segmentation: str,
        kind: str,
        opacity: float,
    ):
        """Build the cell-type rows for ``dataset`` (called on the GUI thread).

        ``layout`` is an ordered list of ``("header", title, rgba)`` and
        ``("type", label)`` entries -- the same convention the gene inspector
        uses, so fine subtypes render grouped under their broad class heading.
        """
        self._suppress = True
        try:
            self._type_list.clear()
            self._checkboxes.clear()
            self._items.clear()
            self._headers = []
            self._dataset = str(dataset)
            self._kind = str(kind)
            self._segmentation = str(segmentation)
            self._set_opacity_widgets(float(opacity))
            for seg, button in self._seg_buttons.items():
                button.blockSignals(True)
                button.setChecked(seg == self._segmentation)
                button.blockSignals(False)
            for k, button in self._level_buttons.items():
                button.blockSignals(True)
                button.setChecked(bool(self._kind) and k == self._kind)
                button.blockSignals(False)

            pending_header: tuple | None = None
            header_types: list[str] = []

            def _finalize_header():
                if pending_header is None:
                    return
                item, widget, title = pending_header
                types = list(header_types)
                self._headers.append((item, title, types))
                widget.clicked.connect(lambda ts=types: self._on_header_clicked(ts))

            for entry in layout:
                if entry[0] == "header":
                    _finalize_header()
                    _, title, rgba = entry
                    item = QListWidgetItem()
                    item.setFlags(Qt.NoItemFlags)
                    widget = self._make_header_widget(str(title), rgba)
                    hint = widget.sizeHint()
                    item.setSizeHint(QSize(hint.width(), hint.height() + 14))
                    self._type_list.addItem(item)
                    self._type_list.setItemWidget(item, widget)
                    pending_header = (item, widget, str(title))
                    header_types = []
                    continue

                label = str(entry[1])
                rgba = colors.get(label, (1.0, 1.0, 1.0, 1.0))
                count = int(counts.get(label, 0))
                text = f"{label}    ({count:,})" if count else str(label)
                check = QCheckBox(text)
                check.setIcon(QIcon(make_gene_marker_pixmap(rgba, "disc", px=22)))
                check.setChecked(label in enabled)
                check.toggled.connect(lambda on, t=label: self._on_type_toggled(t, on))
                item = QListWidgetItem()
                item.setSizeHint(check.sizeHint())
                self._type_list.addItem(item)
                self._type_list.setItemWidget(item, check)
                self._checkboxes[label] = check
                self._items[label] = item
                header_types.append(label)

            _finalize_header()
            self._refresh_row_visibility()
            self._refresh_show_all_state()
        finally:
            self._suppress = False

    def _make_header_widget(self, title: str, rgba) -> QLabel:
        """A clickable, bold group heading tinted by the broad class colour."""
        label = _GroupHeaderLabel(title)
        label.setCursor(Qt.PointingHandCursor)
        label.setToolTip("Click to show / hide all cell types in this group")
        r, g, b = (int(round(255 * float(c))) for c in (rgba[:3] if rgba else (0.6, 0.6, 0.6)))
        label.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 13pt; color: #f0f0f0; margin-top: 6px;"
            f" padding: 4px 4px 4px 8px; border-left: 7px solid rgb({r}, {g}, {b}); }}"
            "QLabel:hover { color: #ffffff; background: rgba(255, 255, 255, 20); }"
        )
        return label

    # -- helpers ------------------------------------------------------------
    def _set_opacity_widgets(self, opacity: float):
        pct = int(round(min(1.0, max(0.0, float(opacity))) * 100))
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(pct)
        self._opacity_slider.blockSignals(False)
        self._opacity_value.setText(f"{pct}%")

    def _refresh_row_visibility(self):
        needle = self._filter_edit.text().strip().lower()
        for label, item in self._items.items():
            item.setHidden(bool(needle) and needle not in label.lower())
        for header_item, _title, types in self._headers:
            any_visible = any(
                t in self._items and not self._items[t].isHidden() for t in types
            )
            header_item.setHidden(not any_visible)

    def _refresh_show_all_state(self):
        labels = list(self._checkboxes)
        checked = sum(1 for t in labels if self._checkboxes[t].isChecked())
        self._show_all_check.blockSignals(True)
        if not labels or checked == 0:
            self._show_all_check.setCheckState(Qt.Unchecked)
        elif checked == len(labels):
            self._show_all_check.setCheckState(Qt.Checked)
        else:
            self._show_all_check.setCheckState(Qt.PartiallyChecked)
        self._show_all_check.blockSignals(False)

    # -- event handlers -----------------------------------------------------
    def _on_close(self):
        if self._dataset is not None:
            self._close_callback(self._dataset)

    def _on_segmentation_clicked(self, seg: str):
        if seg == self._segmentation:
            return
        self._segmentation = str(seg)
        if self._suppress or self._dataset is None:
            return
        self._set_segmentation_callback(self._dataset, str(seg))

    def _on_kind_clicked(self, kind: str):
        # Enforce single selection (the group is non-exclusive so it can start
        # with neither Broad nor Fine chosen).
        for k, button in self._level_buttons.items():
            button.blockSignals(True)
            button.setChecked(k == kind)
            button.blockSignals(False)
        if kind == self._kind:
            return
        self._kind = str(kind)
        if self._suppress or self._dataset is None:
            return
        self._set_kind_callback(self._dataset, str(kind))

    def _on_type_toggled(self, label: str, on: bool):
        if self._suppress or self._dataset is None:
            return
        self._set_type_visible_callback(self._dataset, label, bool(on))
        self._refresh_show_all_state()

    def _on_show_all_clicked(self, _checked=None):
        if self._suppress or self._dataset is None:
            return
        turn_on = self._show_all_check.checkState() != Qt.Unchecked
        self._suppress = True
        try:
            for label in self._checkboxes:
                self._checkboxes[label].setChecked(turn_on)
        finally:
            self._suppress = False
        self._refresh_show_all_state()
        self._set_all_types_callback(self._dataset, turn_on)

    def _on_header_clicked(self, types: list[str]):
        if self._suppress or self._dataset is None:
            return
        visible = [
            t for t in types
            if t in self._checkboxes and t in self._items and not self._items[t].isHidden()
        ]
        if not visible:
            return
        turn_on = not all(self._checkboxes[t].isChecked() for t in visible)
        self._suppress = True
        try:
            for t in visible:
                self._checkboxes[t].setChecked(turn_on)
        finally:
            self._suppress = False
        self._refresh_show_all_state()
        self._set_types_visible_callback(self._dataset, visible, turn_on)

    def _on_opacity_changed(self, value: int):
        self._opacity_value.setText(f"{int(value)}%")
        if self._suppress or self._dataset is None:
            return
        self._set_opacity_callback(self._dataset, round(value / 100.0, 2))

    def _apply_filter(self, _text: str):
        self._refresh_row_visibility()


class CellInfoPieChart(QWidget):
    """Pie chart of a cell's transcripts, one slice per gene.

    Each slice is drawn in the gene's transcript colour darkened a shade, with
    the gene's marker glyph placed just inside the slice's outer edge.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slices: list[dict] = []
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_slices(self, slices: list[dict]):
        """``slices``: list of ``{count, rgba, glyph}`` (highest count first)."""
        self._slices = list(slices or [])
        self.update()

    def sizeHint(self):  # noqa: N802 (Qt override)
        return QSize(180, 180)

    def paintEvent(self, _event):  # noqa: N802 (Qt override)
        total = sum(max(0, int(s.get("count", 0))) for s in self._slices)
        if total <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        side = min(self.width(), self.height()) - 12
        if side <= 0:
            painter.end()
            return
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        rect = QRectF(left, top, side, side)
        cx, cy = rect.center().x(), rect.center().y()
        radius = side / 2.0

        start_deg = 90.0  # start at 12 o'clock, sweep clockwise
        glyph_font = QFont()
        glyph_font.setPointSizeF(max(7.0, side * 0.055))
        painter.setFont(glyph_font)
        for entry in self._slices:
            count = max(0, int(entry.get("count", 0)))
            if count <= 0:
                continue
            span_deg = 360.0 * count / total
            rgba = entry.get("rgba", (0.6, 0.6, 0.6, 1.0))
            color = QColor.fromRgbF(
                float(rgba[0]), float(rgba[1]), float(rgba[2]),
                float(rgba[3]) if len(rgba) >= 4 else 1.0,
            )
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.drawPie(rect, int(round(start_deg * 16)), int(round(-span_deg * 16)))

            glyph = str(entry.get("glyph", ""))
            if glyph:
                mid_deg = start_deg - span_deg / 2.0
                mid_rad = np.deg2rad(mid_deg)
                gx = cx + np.cos(mid_rad) * radius * 0.72
                gy = cy - np.sin(mid_rad) * radius * 0.72
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(
                    QRectF(gx - 12, gy - 12, 24, 24),
                    Qt.AlignCenter,
                    glyph,
                )
            start_deg -= span_deg
        painter.end()


def _cell_summary_html(cell: dict) -> str:
    area_um2 = cell.get("area")
    area_text = f"{area_um2:,.1f} µm²" if area_um2 is not None else "n/a"
    lines = [
        f"<b>Total transcripts:</b> {int(cell.get('total', 0)):,}",
        f"<b>Area:</b> {area_text}",
    ]
    intensity_rows = cell.get("intensities") or []
    if intensity_rows:
        lines.append("<b>Mean intensity:</b>")
        for channel, value in intensity_rows:
            value_text = f"{value:,.1f}" if value is not None else "n/a"
            lines.append(f"&nbsp;&nbsp;{html.escape(str(channel))}: {value_text}")
    else:
        lines.append("<b>Mean intensity:</b> no image channels loaded")
    return "<br>".join(lines)


def _cell_gene_list_html(cell: dict) -> str:
    lines = []
    for row in cell.get("gene_rows", []):
        rgba = row.get("rgba", (1.0, 1.0, 1.0, 1.0))
        rgb = tuple(max(0, min(255, int(round(float(v) * 255)))) for v in rgba[:3])
        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        glyph = html.escape(str(row.get("glyph", "")))
        gene = html.escape(str(row.get("gene", "")))
        count = int(row.get("count", 0))
        lines.append(
            f"<div style='margin:2px 0;'>"
            f"<span style='color:{color}; font-size:13pt;'>{glyph}</span> "
            f"<b>{gene}</b> "
            f"<span style='color:#9aa4b2;'>{count:,}</span></div>"
        )
    return "".join(lines) or "<i>No transcripts assigned</i>"


class _CellPanel(QWidget):
    """One selected cell's column: coloured Cell ID + summary, pie, gene list.

    The pie and title/summary are pinned to the top; the gene list fills the
    remaining panel height and scrolls internally, so a long gene list never
    pushes the pie down or makes the panel taller than the visible window.
    """

    def __init__(self, cell: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("CellPanel")
        self.setStyleSheet(" QLabel { background: transparent; }")
        # Fill the row's (viewport) height so the gene list can size to it.
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        color = str(cell.get("color", "#ffffff"))

        outer = QHBoxLayout()
        outer.setContentsMargins(10, 2, 10, 2)
        outer.setSpacing(12)

        # -- Left: Cell ID (in the cell's highlight colour) + summary -------
        left = QVBoxLayout()
        left.setSpacing(8)
        title = QLabel(f"Cell {html.escape(str(cell.get('cell_id', '')))}")
        title.setStyleSheet(f"color: {color}; font-size: 22pt; font-weight: bold;")
        title.setTextFormat(Qt.RichText)
        summary = QLabel(_cell_summary_html(cell))
        summary.setTextFormat(Qt.RichText)
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #e6ebf2; font-size: 12pt;")
        summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        left.addWidget(title)
        left.addWidget(summary)
        left.addStretch(1)  # keep title + summary pinned to the top
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(250)
        outer.addWidget(left_widget)

        # -- Middle: pie chart (slices darkened a shade), pinned to the top -
        pie = CellInfoPieChart()
        pie.setFixedSize(170, 170)
        pie.set_slices(
            [
                {
                    "count": row.get("count", 0),
                    "glyph": row.get("glyph", ""),
                    "rgba": darken_rgba(row.get("rgba", (0.6, 0.6, 0.6, 1.0)), CELL_PIE_SLICE_DARKEN),
                }
                for row in cell.get("gene_rows", [])
            ]
        )
        outer.addWidget(pie, 0, Qt.AlignTop)

        # -- Right: ranked gene list, fills panel height + scrolls ----------
        gene_list = QLabel(_cell_gene_list_html(cell))
        gene_list.setTextFormat(Qt.RichText)
        gene_list.setWordWrap(False)
        gene_list.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        gene_scroll = QScrollArea()
        gene_scroll.setWidgetResizable(True)
        gene_scroll.setFrameShape(QScrollArea.NoFrame)
        gene_scroll.setStyleSheet("background: transparent;")
        gene_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gene_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        gene_scroll.setWidget(gene_list)
        gene_scroll.setFixedWidth(230)
        # Expanding height + a small minimum so the panel never demands more
        # vertical space than the visible window; the list scrolls instead.
        gene_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        gene_scroll.setMinimumHeight(60)
        outer.addWidget(gene_scroll)

        self.setLayout(outer)


class CellInfoOverlay(QWidget):
    """Docked bottom bar summarising every currently highlighted cell.

    A persistent instruction header sits above a horizontally-scrolling row of
    :class:`_CellPanel` columns — one per selected cell, packed left-to-right,
    with a horizontal scrollbar once they overflow the viewport width.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CellInfoOverlay")
        self.setStyleSheet(
            "#CellInfoOverlay { background: #14161c; }"
            " QLabel { color: #f0f0f0; background: transparent; }"
        )
        self.setMaximumHeight(260)

        outer = QVBoxLayout()
        outer.setContentsMargins(8, 6, 8, 4)
        outer.setSpacing(4)

        self._instruction = QLabel(
            "To deselect all currently highlighted cells, close this window "
            "with the X in the top left of the window"
        )
        self._instruction.setStyleSheet("color: #9aa4b2; font-size: 9pt;")
        self._instruction.setWordWrap(True)
        outer.addWidget(self._instruction)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._row = QWidget()
        self._row_layout = QHBoxLayout()
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(0)
        self._row_layout.addStretch(1)  # trailing stretch packs panels to the left
        self._row.setLayout(self._row_layout)
        self._scroll.setWidget(self._row)
        outer.addWidget(self._scroll, 1)

        self.setLayout(outer)

    @staticmethod
    def _divider() -> QFrame:
        """A thick vertical rule drawn between adjacent cell panels."""
        line = QFrame()
        line.setObjectName("CellDivider")
        line.setFixedWidth(3)
        line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        line.setStyleSheet("#CellDivider { background: rgba(255, 255, 255, 90); }")
        return line

    def set_cells(self, cells: list[dict]):
        """Rebuild the panel row, one :class:`_CellPanel` per cell (in order)."""
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, cell in enumerate(cells):
            if index > 0:
                self._row_layout.addWidget(self._divider())  # separate the sets
            self._row_layout.addWidget(_CellPanel(cell))
        self._row_layout.addStretch(1)

    def cell_panels(self) -> list[_CellPanel]:
        panels = []
        for i in range(self._row_layout.count()):
            widget = self._row_layout.itemAt(i).widget()
            if isinstance(widget, _CellPanel):
                panels.append(widget)
        return panels


class _WidgetResizeRelay(QObject):
    """Forwards a parent widget's resize events to a callback (event filter)."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def eventFilter(self, _obj, event):  # noqa: N802 (Qt override)
        if event.type() == QEvent.Resize:
            try:
                self._callback()
            except Exception:
                pass
        return False


class ScaleBarOverlay(QWidget):
    """Bottom-right canvas overlay: a fixed-length bar labelled with the micron
    distance it currently spans.

    The bar is a fixed fraction of the canvas width and never changes size; only
    the label changes as the camera zooms. The label is large, bold, white with a
    black outline so it reads over any background.
    """

    def __init__(self, parent=None, width_fraction: float = 0.25, margin: int = 28):
        super().__init__(parent)
        self._um_per_px: float | None = None
        self._width_fraction = float(width_fraction)
        self._margin = int(margin)
        self._bar_px = 100
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.reposition()

    def set_um_per_px(self, um_per_px: float | None):
        self._um_per_px = float(um_per_px) if um_per_px and um_per_px > 0 else None
        self.update()

    def reposition(self):
        """Size the bar to a quarter of the canvas width and pin bottom-right."""
        parent = self.parentWidget()
        if parent is None:
            return
        pw, ph = int(parent.width()), int(parent.height())
        self._bar_px = max(20, int(pw * self._width_fraction))
        pad = 6
        w = self._bar_px + 2 * pad
        h = 66
        self.setGeometry(max(0, pw - self._margin - w), max(0, ph - self._margin - h), w, h)
        self.update()

    @staticmethod
    def _format_um(um: float) -> str:
        if um >= 100:
            return f"{um:.0f} µm"
        if um >= 10:
            return f"{um:.1f} µm"
        return f"{um:.2f} µm"

    def paintEvent(self, _event):  # noqa: N802 (Qt override)
        if self._um_per_px is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pad = 6
        x0, x1 = pad, pad + self._bar_px
        bar_y = self.height() - 12
        tick = 7
        # Draw the bar + end ticks twice: a thick black outline then a white line.
        for color, width in ((QColor(0, 0, 0), 7), (QColor(255, 255, 255), 3)):
            pen = QPen(color)
            pen.setWidth(width)
            pen.setCapStyle(Qt.FlatCap)
            painter.setPen(pen)
            painter.drawLine(x0, bar_y, x1, bar_y)
            painter.drawLine(x0, bar_y - tick, x0, bar_y + tick)
            painter.drawLine(x1, bar_y - tick, x1, bar_y + tick)

        text = self._format_um(self._bar_px * self._um_per_px)
        font = QFont()
        font.setBold(True)
        font.setPointSize(22)
        metrics = QFontMetricsF(font)
        tx = x0 + (self._bar_px - metrics.horizontalAdvance(text)) / 2.0
        ty = bar_y - tick - 8.0  # text baseline sits just above the bar
        path = QPainterPath()
        path.addText(tx, ty, font, text)
        # Draw the outline first (black stroke, no fill), then the white glyph
        # fill on top. A stroked pen is centred on the glyph edge, so filling
        # white afterwards covers the inner half and leaves only a thin outer
        # black outline -- keeping the white interior fully visible.
        outline = QPen(QColor(0, 0, 0))
        outline.setWidthF(5.0)
        outline.setJoinStyle(Qt.RoundJoin)
        painter.setPen(outline)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawPath(path)
        painter.end()


class ComparisonViewerController:
    """Coordinate loading/clearing napari layers for each dataset."""

    def __init__(self, viewer: napari.Viewer, datasets: dict[str, DatasetConfig], args):
        self.viewer = viewer
        self.datasets = datasets
        self.args = args
        self.active_dataset: str | None = None
        self._status_callback = None
        self._progress_callback = None
        self._shape_keys_callback = None
        self._loaded_shape_keys_callback = None
        self._image_entries_callback = None
        self._loaded_image_entries_callback = None
        self._datasets_changed_callback = None
        self._cellpose_value_options_callback = None
        self._current_cortical_depth_piece_id = CORTICAL_DEPTH_DEFAULT_PIECE_ID
        self._active_sdata = None
        self._active_images_sdata = None
        self._segmentation_keys: list[str] = []
        self._image_keys: list[str] = []
        self._image_channels: list[tuple[str, str]] = []
        self._x_transform: tuple[float, float, float] | None = None
        self._y_transform: tuple[float, float, float] | None = None
        self._cellpose_value_generation = 0
        self._cellpose_value_worker: object | None = None
        # Background build bookkeeping for the heavy label / image pipelines
        # that run in napari thread_workers so the UI stays responsive.
        self._label_build_generation = 0
        self._label_build_worker: object | None = None
        self._image_build_generation = 0
        self._image_build_worker: object | None = None
        # Per-gene transcript renderer (the default + only transcript view).
        self._gene_inspector_states: dict[str, GeneInspectorState] = {}
        self._gene_build_generation = 0
        self._gene_build_worker: object | None = None
        self._gene_inspector_widget = None
        # Cell-type mask-fill overlay (one coloured Labels layer per dataset).
        self._cell_type_states: dict[str, CellTypeOverlayState] = {}
        self._cell_type_widget = None
        self._cell_type_generation = 0
        self._cell_type_worker: object | None = None
        # Click-to-inspect cell state: a lazily-built per-dataset transcript
        # index, the currently selected cell, and the floating summary box.
        self._cell_transcript_index: dict[str, CellTranscriptIndex] = {}
        # Per dataset, an ordered list of highlighted cells (each a dict with the
        # cell id, highlight colour, geometry-derived draw data and panel stats).
        self._selected_cells: dict[str, list[dict]] = {}
        self._cell_info_overlay: CellInfoOverlay | None = None
        self._cell_info_dock = None
        self._suppress_dock_visibility = False
        # Bottom-right micron scale bar overlay + its canvas-resize relay.
        self._scale_bar: ScaleBarOverlay | None = None
        self._canvas_resize_relay: _WidgetResizeRelay | None = None
        self._install_gene_pick_callback()

    def set_status_callback(self, fn):
        self._status_callback = fn

    def set_progress_callback(self, fn):
        self._progress_callback = fn

    def set_gene_inspector_widget(self, widget):
        self._gene_inspector_widget = widget

    def set_cell_type_widget(self, widget):
        self._cell_type_widget = widget

    def _install_gene_pick_callback(self):
        callbacks = getattr(self.viewer, "mouse_drag_callbacks", None)
        if callbacks is None:
            return
        try:
            callbacks.append(self._on_viewer_mouse_press)
        except Exception as exc:
            log.debug("Could not install transcript pick callback: %s", exc)

    # -- Canvas overlays: micron scale bar + zoom-aware outline crispness ------
    def install_canvas_overlays(self):
        """Attach the bottom-right scale bar and start tracking camera zoom.

        Best-effort: if the napari canvas widget can't be reached (headless / a
        future napari layout change) the scale bar is skipped without error.
        """
        native = self._canvas_native_widget()
        if native is not None:
            try:
                self._scale_bar = ScaleBarOverlay(native)
                self._scale_bar.show()
                self._scale_bar.raise_()
                self._canvas_resize_relay = _WidgetResizeRelay(self._on_canvas_geometry_changed)
                native.installEventFilter(self._canvas_resize_relay)
            except Exception as exc:
                log.debug("Scale bar overlay unavailable: %s", exc)
                self._scale_bar = None
        try:
            self.viewer.camera.events.zoom.connect(self._on_camera_zoom)
        except Exception as exc:
            log.debug("Could not connect camera zoom for overlays: %s", exc)
        self._on_camera_zoom()

    def _canvas_native_widget(self):
        try:
            return self.viewer.window._qt_viewer.canvas.native
        except Exception:
            return None

    def _canvas_size_px(self) -> tuple[float, float] | None:
        try:
            size = self.viewer.window._qt_viewer.canvas.size
            w, h = float(size[0]), float(size[1])
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass
        native = self._canvas_native_widget()
        if native is not None:
            try:
                w, h = float(native.width()), float(native.height())
                if w > 0 and h > 0:
                    return (w, h)
            except Exception:
                pass
        return None

    def _um_per_canvas_px(self) -> float | None:
        camera = getattr(self.viewer, "camera", None)
        zoom = float(getattr(camera, "zoom", 0.0) or 0.0) if camera is not None else 0.0
        return (1.0 / zoom) if zoom > 0 else None

    def _on_canvas_geometry_changed(self):
        if self._scale_bar is not None:
            self._scale_bar.reposition()
        self._on_camera_zoom()

    def _on_camera_zoom(self, *_):
        um_per_px = self._um_per_canvas_px()
        if self._scale_bar is not None:
            self._scale_bar.set_um_per_px(um_per_px)
        self._apply_label_zoom_interpolation(um_per_px)

    def _apply_label_zoom_interpolation(self, um_per_px: float | None = None):
        """Crisp (nearest) outlines when zoomed in past the threshold, else linear.

        Only runs when the outline mode is the default 'linear'; an explicit
        --label-interpolation nearest keeps outlines crisp at every zoom.
        """
        if str(getattr(self.args, "label_interpolation", "linear")) != "linear":
            return
        if um_per_px is None:
            um_per_px = self._um_per_canvas_px()
        size = self._canvas_size_px()
        if um_per_px is None or size is None:
            return
        visible_um = min(size) * um_per_px
        mode = "nearest" if visible_um <= LABEL_CRISP_ZOOM_UM else "linear"
        for layer in list(self.viewer.layers):
            name = str(getattr(layer, "name", ""))
            if name.startswith("Segmentation | ") and hasattr(layer, "interpolation2d"):
                try:
                    if layer.interpolation2d != mode:
                        layer.interpolation2d = mode
                except Exception:
                    pass

    def set_shape_keys_callback(self, fn):
        self._shape_keys_callback = fn

    def set_loaded_shape_keys_callback(self, fn):
        self._loaded_shape_keys_callback = fn

    def set_image_entries_callback(self, fn):
        self._image_entries_callback = fn

    def set_loaded_image_entries_callback(self, fn):
        self._loaded_image_entries_callback = fn

    def set_datasets_changed_callback(self, fn):
        self._datasets_changed_callback = fn

    def set_cellpose_value_options_callback(self, fn):
        self._cellpose_value_options_callback = fn

    def _set_status(self, text: str):
        if self._status_callback is not None:
            self._status_callback(text)

    def _begin_progress(self, key: str, text: str):
        """Mark a loading stage as active (shows the busy bar + stage text)."""
        self._set_status(text)
        if self._progress_callback is not None:
            self._progress_callback(str(key), str(text), True)

    def _end_progress(self, key: str):
        """Mark a loading stage as finished (hides the busy bar when idle)."""
        if self._progress_callback is not None:
            self._progress_callback(str(key), "", False)

    def _reset_progress(self):
        """Clear all loading stages (e.g. when switching datasets)."""
        for key in ("dataset", "images", "masks", "transcripts"):
            self._end_progress(key)

    def _publish_shape_keys(self):
        if self._shape_keys_callback is not None:
            self._shape_keys_callback(list(self._segmentation_keys))

    def _segmentation_key_is_loaded(self, shape_key: str) -> bool:
        """True if any layer for this segmentation key is currently present."""
        if self.active_dataset is None:
            return False
        ds = self.active_dataset
        candidates = (
            make_layer_name(ds, self._segmentation_layer_type(), shape_key),
            make_layer_name(ds, "labels", shape_key),
            make_layer_name(ds, "labels", self._label_key_for_shape_key(shape_key)),
        )
        return any(self._get_layer_by_name(name) is not None for name in candidates)

    def _publish_loaded_segmentation_keys(self):
        if self._loaded_shape_keys_callback is None:
            return
        loaded = [k for k in self._segmentation_keys if self._segmentation_key_is_loaded(k)]
        self._loaded_shape_keys_callback(loaded)

    def _enumerate_image_channels(self) -> list[tuple[str, str]]:
        """List (image_key, channel) pairs for every image element.

        Both MERSCOPE (``MERSCOPE_z_projection``) and Xenium (``morphology_focus``
        et al.) store images as SpatialData image elements with a ``c`` channel
        coordinate, so channel names come from the same ``channel_labels`` path.
        """
        entries: list[tuple[str, str]] = []
        images_sdata = self._active_images_sdata
        if images_sdata is None:
            return entries
        try:
            image_keys = [str(k) for k in images_sdata.images.keys() if not is_derived_cache_key(str(k))]
        except Exception:
            return entries
        for image_key in sorted(image_keys):
            try:
                base = ensure_cyx(get_scale0_dataarray(images_sdata.images[image_key]))
                channels = channel_labels(base)
            except Exception as exc:
                log.warning("[%s] Could not read channels for image '%s' (%s)", self.active_dataset, image_key, exc)
                continue
            for channel in channels:
                entries.append((image_key, str(channel)))
        return entries

    def _publish_image_entries(self):
        if self._image_entries_callback is None:
            return
        entries = self._image_channels
        # Disambiguate identical channel names that appear under >1 image key.
        counts: dict[str, int] = {}
        for _key, channel in entries:
            counts[channel] = counts.get(channel, 0) + 1
        display = [
            (channel if counts.get(channel, 0) <= 1 else f"{image_key}: {channel}", image_key, channel)
            for image_key, channel in entries
        ]
        self._image_entries_callback(display)

    def _publish_loaded_image_entries(self):
        if self._loaded_image_entries_callback is None:
            return
        ds = self.active_dataset
        loaded = [
            (image_key, channel)
            for image_key, channel in self._image_channels
            if ds is not None and self._get_layer_by_name(make_layer_name(ds, "image", image_key, channel)) is not None
        ]
        self._loaded_image_entries_callback(loaded)

    def _publish_cellpose_value_options(self):
        if self._cellpose_value_options_callback is None:
            return
        if self.active_dataset is None or self.active_dataset != "MERSCOPE":
            self._cellpose_value_options_callback([], [], False)
            return

        cfg = self.datasets[self.active_dataset]
        try:
            if not cellpose_quantification_table_available(cfg.zarr_path):
                self._cellpose_value_options_callback([], [], False)
                return
            features = cellpose_quantification_features(cfg.zarr_path)
            channels = sorted({feature.channel for feature in features})
            statistic_order = ["min", "median", "mean", "max", "iqr"]
            statistics = sorted(
                {feature.statistic for feature in features},
                key=lambda value: (0, statistic_order.index(value)) if value in statistic_order else (1, value),
            )
            self._cellpose_value_options_callback(channels, statistics, True)
        except Exception as exc:
            log.warning("[%s] Could not inspect Cellpose value table (%s)", self.active_dataset, exc)
            self._cellpose_value_options_callback([], [], False)

    def _segmentation_source(self) -> str:
        return "labels" if str(self.args.segmentation_source).lower() == "labels" else "shapes"

    def _segmentation_layer_type(self) -> str:
        return "labels" if self._segmentation_source() == "labels" else "shapes"

    def _segmentation_unit_name(self) -> str:
        return "labels" if self._segmentation_source() == "labels" else "polygons"

    def _clear_layers(self):
        self._clear_gene_inspector_states()
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)

    def _resolve_optional_transform_paths(self, ds: str, cfg: DatasetConfig) -> tuple[Path | None, Path | None]:
        """Resolve optional transform/spec paths from explicit args or common defaults."""
        merscope_transform_path = cfg.merscope_transform_path
        xenium_spec_path = cfg.xenium_spec_path

        if ds == "MERSCOPE" and merscope_transform_path is None:
            candidate = cfg.zarr_path / "micron_to_mosaic_pixel_transform.csv"
            if candidate.exists():
                merscope_transform_path = candidate

        if ds == "XENIUM" and xenium_spec_path is None:
            candidates = [
                cfg.zarr_path / "experiment.xenium",
                cfg.zarr_path.parent / "experiment.xenium",
            ]
            for candidate in candidates:
                if candidate.exists():
                    xenium_spec_path = candidate
                    break

        return merscope_transform_path, xenium_spec_path

    def _remove_layer_by_name(self, layer_name: str):
        """Remove a layer by exact name if present."""
        for layer in list(self.viewer.layers):
            if str(layer.name) == str(layer_name):
                self.viewer.layers.remove(layer)
                return

    def _get_layer_by_name(self, layer_name: str):
        """Return a layer by exact name when present."""
        for layer in list(self.viewer.layers):
            if str(layer.name) == str(layer_name):
                return layer
        return None

    def _remove_layers_by_prefix(self, prefix: str):
        names = [str(layer.name) for layer in self.viewer.layers]
        for layer_name in matching_layer_names(names, prefix):
            self._remove_layer_by_name(layer_name)

    def _cortical_depth_layer_name(self, ds: str, role: str) -> str:
        return make_layer_name(ds, CORTICAL_DEPTH_LAYER_TYPE, role)

    def _find_cortical_depth_annotation_layer(self, ds: str, role: str):
        ds = str(ds).upper()
        role = str(role)
        expected_name = self._cortical_depth_layer_name(ds, role)
        for layer in self.viewer.layers:
            metadata = getattr(layer, "metadata", {}) or {}
            if metadata.get("cortical_depth_dataset") == ds and metadata.get("cortical_depth_role") == role:
                return layer
        return self._get_layer_by_name(expected_name)

    def _ensure_cortical_depth_annotation_layer(self, ds: str, role: str):
        ds = str(ds).upper()
        role = str(role)
        layer = self._find_cortical_depth_annotation_layer(ds, role)
        if layer is None:
            spec = CORTICAL_DEPTH_ROLE_SPECS[role]
            shape_type = "path" if spec.geometry_kind == "line" else "polygon"
            feature_kwargs = {}
            if role in CORTICAL_DEPTH_PIECE_ROLES:
                feature_kwargs = {
                    "features": {CORTICAL_DEPTH_PIECE_ID_PROPERTY: []},
                    "feature_defaults": {
                        CORTICAL_DEPTH_PIECE_ID_PROPERTY: [self._current_cortical_depth_piece_id]
                    },
                    "property_choices": {
                        CORTICAL_DEPTH_PIECE_ID_PROPERTY: [self._current_cortical_depth_piece_id]
                    },
                }
            layer = self.viewer.add_shapes(
                data=[],
                ndim=2,
                shape_type=shape_type,
                name=self._cortical_depth_layer_name(ds, role),
                edge_color=CORTICAL_DEPTH_LAYER_COLORS[role],
                face_color=CORTICAL_DEPTH_FILL_COLORS[role],
                edge_width=max(1.5, float(self.args.shape_edge_width) * 2.0),
                visible=True,
                **feature_kwargs,
            )
        metadata = getattr(layer, "metadata", None)
        if metadata is None:
            metadata = {}
            layer.metadata = metadata
        metadata["cortical_depth_dataset"] = ds
        metadata["cortical_depth_role"] = role
        metadata["cortical_depth_geojson_role"] = CORTICAL_DEPTH_ROLE_SPECS[role].geojson_role
        if role in CORTICAL_DEPTH_PIECE_ROLES:
            self._set_layer_current_piece(layer, self._current_cortical_depth_piece_id)
        return layer

    def _annotation_layer_shapes(self, layer) -> list[np.ndarray]:
        if layer is None:
            return []
        data = getattr(layer, "data", [])
        if data is None:
            return []
        if isinstance(data, np.ndarray) and data.ndim == 2:
            return [np.asarray(data, dtype=float)]
        return [np.asarray(shape, dtype=float) for shape in list(data)]

    def _annotation_layer_shape_inputs(self, layer, role: str) -> list[CorticalDepthShapeInput]:
        shapes = self._annotation_layer_shapes(layer)
        if role not in CORTICAL_DEPTH_PIECE_ROLES:
            return [CorticalDepthShapeInput(data=shape) for shape in shapes]
        piece_ids = self._layer_piece_ids(layer, len(shapes))
        return [
            CorticalDepthShapeInput(data=shape, tissue_piece_id=piece_ids[idx])
            for idx, shape in enumerate(shapes)
        ]

    def _layer_piece_ids(self, layer, n_shapes: int) -> list[str]:
        if layer is None or n_shapes <= 0:
            return []
        fallback = self._current_cortical_depth_piece_id or CORTICAL_DEPTH_DEFAULT_PIECE_ID
        try:
            features = getattr(layer, "features", None)
            if features is not None and CORTICAL_DEPTH_PIECE_ID_PROPERTY in features:
                values = list(features[CORTICAL_DEPTH_PIECE_ID_PROPERTY])
                out = [str(value).strip() or fallback for value in values[:n_shapes]]
                if len(out) < n_shapes:
                    out.extend([fallback] * (n_shapes - len(out)))
                return out
        except Exception:
            pass
        return [fallback] * n_shapes

    def _set_layer_current_piece(self, layer, piece_id: str):
        if layer is None:
            return
        piece_id = str(piece_id).strip() or CORTICAL_DEPTH_DEFAULT_PIECE_ID
        try:
            choices = getattr(layer, "property_choices", {}) or {}
            existing = list(choices.get(CORTICAL_DEPTH_PIECE_ID_PROPERTY, []))
            if piece_id not in [str(value) for value in existing]:
                existing.append(piece_id)
            choices[CORTICAL_DEPTH_PIECE_ID_PROPERTY] = existing
            layer.property_choices = choices
        except Exception:
            pass
        try:
            layer.current_properties = {CORTICAL_DEPTH_PIECE_ID_PROPERTY: [piece_id]}
        except Exception:
            pass

    def _collect_cortical_depth_annotations(self, ds: str) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
        ds = str(ds).upper()
        layers_by_role: dict[str, list[np.ndarray]] = {}
        layer_names: dict[str, str] = {}
        for role in CORTICAL_DEPTH_ROLE_ORDER:
            layer = self._find_cortical_depth_annotation_layer(ds, role)
            layers_by_role[role] = self._annotation_layer_shape_inputs(layer, role)
            if layer is not None:
                layer_names[role] = str(layer.name)
        return layers_by_role, layer_names

    def create_cortical_depth_annotation_layers(self, dataset_name: str):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        ds = self.active_dataset
        first_layer = None
        created = 0
        for role in CORTICAL_DEPTH_ROLE_ORDER:
            before = self._find_cortical_depth_annotation_layer(ds, role)
            layer = self._ensure_cortical_depth_annotation_layer(ds, role)
            if first_layer is None:
                first_layer = layer
            if before is None:
                created += 1
        if first_layer is not None:
            try:
                self.viewer.layers.selection.active = first_layer
                first_layer.mode = "add_path"
            except Exception:
                pass
        self._set_status(f"{ds} cortical-depth drawing layers ready ({created} created).")

    def set_cortical_depth_current_piece(self, dataset_name: str, piece_id: str):
        self._current_cortical_depth_piece_id = str(piece_id).strip() or CORTICAL_DEPTH_DEFAULT_PIECE_ID
        ds = str(dataset_name).upper()
        for role in CORTICAL_DEPTH_PIECE_ROLES:
            layer = self._find_cortical_depth_annotation_layer(ds, role)
            self._set_layer_current_piece(layer, self._current_cortical_depth_piece_id)

    def apply_cortical_depth_piece_to_selection(self, dataset_name: str, piece_id: str) -> int:
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return 0
        ds = self.active_dataset
        piece_id = str(piece_id).strip() or CORTICAL_DEPTH_DEFAULT_PIECE_ID
        changed = 0
        for role in CORTICAL_DEPTH_PIECE_ROLES:
            layer = self._find_cortical_depth_annotation_layer(ds, role)
            if layer is None:
                continue
            self._set_layer_current_piece(layer, piece_id)
            selected = sorted(int(idx) for idx in getattr(layer, "selected_data", set()) or set())
            if not selected:
                continue
            features = getattr(layer, "features", None)
            if features is None:
                continue
            features = features.copy()
            if CORTICAL_DEPTH_PIECE_ID_PROPERTY not in features:
                features[CORTICAL_DEPTH_PIECE_ID_PROPERTY] = [self._current_cortical_depth_piece_id] * len(features)
            for idx in selected:
                if 0 <= idx < len(features):
                    features.iloc[idx, features.columns.get_loc(CORTICAL_DEPTH_PIECE_ID_PROPERTY)] = piece_id
                    changed += 1
            layer.features = features
        self._current_cortical_depth_piece_id = piece_id
        self._set_status(f"{ds} applied {piece_id} to {changed} selected annotation shape(s).")
        return changed

    def validate_cortical_depth_annotations(self, dataset_name: str):
        if not self._ensure_dataset_is_active(dataset_name):
            empty = build_cortical_depth_annotation_geojson({}, dataset=str(dataset_name).upper())
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return empty
        ds = self.active_dataset
        layers_by_role, layer_names = self._collect_cortical_depth_annotations(ds)
        result = build_cortical_depth_annotation_geojson(
            layers_by_role,
            layer_names=layer_names,
            dataset=ds,
        )
        feature_count = len(result.geojson.get("features", []))
        if result.ok:
            self._set_status(
                f"{ds} cortical-depth annotations valid: features={feature_count}, warnings={len(result.warnings)}."
            )
        else:
            self._set_status(
                f"{ds} cortical-depth annotations invalid: errors={len(result.errors)}, warnings={len(result.warnings)}."
            )
        return result

    def export_cortical_depth_annotations(self, dataset_name: str, path: Path, *, allow_invalid: bool = False):
        result = self.validate_cortical_depth_annotations(dataset_name)
        if not result.ok and not allow_invalid:
            return result
        path = Path(path)
        write_cortical_depth_annotation_geojson(
            path,
            result,
            allow_invalid=allow_invalid,
            include_validation_report=allow_invalid,
        )
        if allow_invalid and not result.ok:
            self._set_status(
                f"{str(dataset_name).upper()} exported validation-failed cortical-depth annotations to {path} "
                f"(errors={len(result.errors)}, warnings={len(result.warnings)})."
            )
        else:
            self._set_status(
                f"{str(dataset_name).upper()} exported cortical-depth annotations to {path} "
                f"(features={len(result.geojson.get('features', []))}, warnings={len(result.warnings)})."
            )
        return result

    def export_separate_cortical_depth_annotations(self, dataset_name: str, output_dir: Path):
        result = self.validate_cortical_depth_annotations(dataset_name)
        if not result.ok:
            return result
        ds = str(dataset_name).upper()
        written = write_cortical_depth_separate_geojsons(
            output_dir,
            result,
            stem=f"{ds.lower()}_cortical_depth",
        )
        self._set_status(
            f"{ds} exported {len(written)} separate cortical-depth GeoJSON file(s) to {Path(output_dir)} "
            f"(warnings={len(result.warnings)})."
        )
        return result

    def snap_cortical_depth_side_edges(self, dataset_name: str) -> bool:
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return False
        ds = self.active_dataset
        pial_layer = self._find_cortical_depth_annotation_layer(ds, "pia")
        wm_layer = self._find_cortical_depth_annotation_layer(ds, "wm")
        side_layer = self._find_cortical_depth_annotation_layer(ds, "side")
        snapped = snap_cortical_depth_boundaries_to_edge(
            self._annotation_layer_shapes(pial_layer),
            self._annotation_layer_shapes(wm_layer),
            self._annotation_layer_shapes(side_layer),
        )
        if pial_layer is not None:
            pial_layer.data = snapped["pia"]
        if wm_layer is not None:
            wm_layer.data = snapped["wm"]
        try:
            active_layer = pial_layer or wm_layer
            if active_layer is not None:
                self.viewer.layers.selection.active = active_layer
                active_layer.mode = "select"
        except Exception:
            pass
        self._set_status(
            f"{ds} snapped {len(snapped['pia']) + len(snapped['wm'])} pial/WM boundary line(s) to the tissue edge."
        )
        return True

    def _distance_object_layer_name(self, ds: str, object_type: str) -> str:
        return make_layer_name(ds, DISTANCE_OBJECT_LAYER_TYPE, object_type)

    def _find_distance_object_annotation_layers(self, ds: str) -> list:
        ds = str(ds).upper()
        layers = []
        for layer in self.viewer.layers:
            metadata = getattr(layer, "metadata", {}) or {}
            if metadata.get("distance_object_dataset") == ds and metadata.get(
                "distance_object_type"
            ):
                layers.append(layer)
        return layers

    def _find_distance_object_annotation_layer(self, ds: str, object_type: str):
        ds = str(ds).upper()
        object_type = str(object_type).strip()
        for layer in self._find_distance_object_annotation_layers(ds):
            metadata = getattr(layer, "metadata", {}) or {}
            if str(metadata.get("distance_object_type")) == object_type:
                return layer
        return self._get_layer_by_name(
            self._distance_object_layer_name(ds, object_type)
        )

    def _ensure_distance_object_annotation_layer(
        self,
        ds: str,
        object_type: str,
    ):
        ds = str(ds).upper()
        object_type = str(object_type).strip()
        if not object_type:
            raise ValueError("Object set name must not be blank.")
        layer = self._find_distance_object_annotation_layer(ds, object_type)
        if layer is None:
            layer = self.viewer.add_shapes(
                data=[],
                ndim=2,
                shape_type="polygon",
                name=self._distance_object_layer_name(ds, object_type),
                edge_color=DISTANCE_OBJECT_EDGE_COLOR,
                face_color=DISTANCE_OBJECT_FILL_COLOR,
                edge_width=max(1.5, float(self.args.shape_edge_width) * 2.0),
                visible=True,
                # An empty dict/list makes napari infer float64, after which
                # its blank string default fails with "could not convert
                # string to float". Preserve string IDs explicitly.
                features=pd.DataFrame(
                    {OBJECT_ID_PROPERTY: pd.Series([], dtype="object")}
                ),
                feature_defaults={OBJECT_ID_PROPERTY: ""},
            )
        metadata = getattr(layer, "metadata", None)
        if metadata is None:
            metadata = {}
            layer.metadata = metadata
        metadata["distance_object_dataset"] = ds
        metadata["distance_object_type"] = object_type
        metadata["distance_object_role"] = "analysis_object"
        return layer

    def create_distance_object_annotation_layer(
        self,
        dataset_name: str,
        object_type: str,
    ):
        """Create or activate a named polygon layer for distance analysis."""
        if not self._ensure_dataset_is_active(dataset_name):
            raise RuntimeError(f"Could not activate dataset {dataset_name}.")
        layer = self._ensure_distance_object_annotation_layer(
            self.active_dataset,
            object_type,
        )
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "add_polygon"
        except Exception:
            pass
        self._set_status(
            f"{self.active_dataset} object layer {str(object_type).strip()!r} ready."
        )
        return layer

    def _distance_object_layer_shape_inputs(
        self,
        layer,
    ) -> list[ObjectAnnotationShapeInput]:
        shapes = self._annotation_layer_shapes(layer)
        object_ids: list[str | None] = [None] * len(shapes)
        try:
            features = getattr(layer, "features", None)
            if features is not None and OBJECT_ID_PROPERTY in features:
                values = list(features[OBJECT_ID_PROPERTY])
                for index, value in enumerate(values[: len(shapes)]):
                    object_ids[index] = str(value).strip() or None
        except Exception:
            pass
        return [
            ObjectAnnotationShapeInput(data=shape, object_id=object_ids[index])
            for index, shape in enumerate(shapes)
        ]

    def _collect_distance_object_annotations(
        self,
        ds: str,
    ) -> dict[str, list[ObjectAnnotationShapeInput]]:
        objects: dict[str, list[ObjectAnnotationShapeInput]] = {}
        for layer in self._find_distance_object_annotation_layers(ds):
            metadata = getattr(layer, "metadata", {}) or {}
            object_type = str(metadata.get("distance_object_type") or "").strip()
            if object_type:
                objects.setdefault(object_type, []).extend(
                    self._distance_object_layer_shape_inputs(layer)
                )
        return objects

    def validate_distance_object_annotations(self, dataset_name: str):
        """Validate every named object polygon layer for one dataset."""
        if not self._ensure_dataset_is_active(dataset_name):
            result = build_object_annotation_geojson(
                {}, dataset=str(dataset_name).upper()
            )
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return result
        ds = self.active_dataset
        result = build_object_annotation_geojson(
            self._collect_distance_object_annotations(ds),
            dataset=ds,
        )
        feature_count = len(result.geojson.get("features", []))
        if result.ok:
            self._set_status(
                f"{ds} object annotations valid: features={feature_count}, "
                f"warnings={len(result.warnings)}."
            )
        else:
            self._set_status(
                f"{ds} object annotations invalid: errors={len(result.errors)}."
            )
        return result

    def export_distance_object_annotations(
        self,
        dataset_name: str,
        path: Path,
    ):
        """Validate and write all named object layers to one GeoJSON file."""
        result = self.validate_distance_object_annotations(dataset_name)
        if not result.ok:
            return result
        write_object_annotation_geojson(path, result)
        self._set_status(
            f"{str(dataset_name).upper()} exported "
            f"{len(result.geojson.get('features', []))} object polygons to {path}."
        )
        return result

    def load_distance_object_annotations(
        self,
        dataset_name: str,
        path: Path,
    ) -> int:
        """Load or replace named object layers from an existing GeoJSON file."""
        if not self._ensure_dataset_is_active(dataset_name):
            raise RuntimeError(f"Could not activate dataset {dataset_name}.")
        objects = read_object_annotation_geojson(path)
        loaded = 0
        active_layer = None
        for object_type, shape_inputs in objects.items():
            layer = self._ensure_distance_object_annotation_layer(
                self.active_dataset,
                object_type,
            )
            layer.data = [shape_input.data for shape_input in shape_inputs]
            layer.features = pd.DataFrame(
                {
                    OBJECT_ID_PROPERTY: [
                        shape_input.object_id or "" for shape_input in shape_inputs
                    ]
                }
            )
            loaded += len(shape_inputs)
            active_layer = layer
        try:
            if active_layer is not None:
                self.viewer.layers.selection.active = active_layer
                active_layer.mode = "select"
        except Exception:
            pass
        self._set_status(
            f"{self.active_dataset} loaded {loaded} object polygons from {path}."
        )
        return loaded

    def _ensure_dataset_is_active(self, dataset_name: str) -> bool:
        ds = str(dataset_name).upper()
        if self.active_dataset != ds:
            self.load_dataset(ds, force=False)
        return self.active_dataset == ds and self._active_sdata is not None

    def _ensure_images_loaded(self, ds: str):
        if self._active_images_sdata is not None and len(self._active_images_sdata.images) > 0:
            return
        cfg = self.datasets[ds]
        self._active_images_sdata = sd.read_zarr(str(cfg.zarr_path), selection=("images",))

    def _startup_segmentation_keys(self) -> list[str]:
        """Segmentation keys to auto-load at startup (Cellpose + ProSeg masks).

        Both platforms write ``MOSAIK_cellpose`` and ``MOSAIK_proseg`` (see the
        MerXen ``segment`` stage). Prefer those exact keys; otherwise fall back to
        the first key containing the token so aligned/variant names still resolve.
        Each mask is skipped when its ``--skip-*`` flag is set.
        """
        keys = list(self._segmentation_keys)
        selected: list[str] = []
        for token, skip in (
            ("cellpose", bool(getattr(self.args, "skip_cellpose", False))),
            ("proseg", bool(getattr(self.args, "skip_proseg", False))),
        ):
            if skip:
                continue
            exact = f"MOSAIK_{token}"
            if exact in keys:
                selected.append(exact)
                continue
            match = next((k for k in keys if token in k.lower()), None)
            if match is not None:
                selected.append(match)
        return list(dict.fromkeys(selected))

    def _start_startup_autoload(self, ds: str):
        """Kick off the default background loads once dataset metadata is ready.

        Images, Cellpose+ProSeg masks and per-gene transcripts each load in their
        own napari thread_worker, so this returns immediately and the UI stays
        responsive while the busy progress bar tracks each stage. Cell-mask value
        overlays are intentionally NOT loaded here (on-demand only).

        Images are requested first so the camera auto-fits to the image extent (the
        tight tissue bounding box) rather than the transcript point cloud. The long
        gene-layer list is still kept at the bottom of the layer list -- not via
        load order, but by explicitly pinning it there once built (see
        :meth:`_move_gene_layers_to_bottom`), so this ordering and the list ordering
        are independent.
        """
        if not bool(getattr(self.args, "skip_images", False)) and self._image_keys:
            self.load_images_on_demand(ds)

        mask_keys = self._startup_segmentation_keys()
        if mask_keys:
            self.load_selected_labels(ds, mask_keys)

        if not bool(getattr(self.args, "skip_transcripts", False)):
            self.open_gene_inspector(ds)

    def load_dataset(self, dataset_name: str, force: bool = False):
        ds = str(dataset_name).upper()
        if ds not in self.datasets:
            self._set_status(f"Unknown dataset: {dataset_name}")
            return

        if (self.active_dataset == ds) and (not force):
            return

        self._reset_progress()
        mem_before = memory_snapshot_gb()
        t0 = time.time()
        cfg = self.datasets[ds]

        try:
            self._begin_progress("dataset", f"Loading {ds} metadata...")
            log.info("[%s] Loading dataset from %s", ds, cfg.zarr_path)

            self._clear_layers()
            gc.collect()
            self.active_dataset = None
            self._segmentation_keys = []
            self._image_keys = []
            self._image_channels = []
            self._publish_shape_keys()
            self._publish_loaded_segmentation_keys()
            self._publish_image_entries()
            self._publish_loaded_image_entries()
            if self._cellpose_value_options_callback is not None:
                self._cellpose_value_options_callback([], [], False)

            selection = startup_selection(self._segmentation_source())
            sdata = sd.read_zarr(str(cfg.zarr_path), selection=selection)
            self._active_sdata = sdata
            # Read images as a separate element so their keys populate the Images
            # tab even when auto-load is skipped; tolerate image-less stores.
            self._active_images_sdata = None
            try:
                self._ensure_images_loaded(ds)
            except Exception as exc:
                log.warning("[%s] Could not read images element (%s); continuing without images.", ds, exc)
                self._active_images_sdata = None
            ms_tf_path, xe_spec_path = self._resolve_optional_transform_paths(ds, cfg)
            x_transform, y_transform = resolve_dataset_mask_affine(
                ds,
                merscope_transform_path=ms_tf_path,
                xenium_spec_path=xe_spec_path,
            )
            self._x_transform = x_transform
            self._y_transform = y_transform
            log.info("[%s] Using image px->um transform x=%s y=%s", ds, x_transform, y_transform)

            self.active_dataset = ds
            if self._segmentation_source() == "labels":
                self._segmentation_keys = sorted(
                    str(k) for k in sdata.labels.keys() if not is_derived_cache_key(str(k))
                )
                if len(self._segmentation_keys) == 0:
                    raise RuntimeError(
                        "No labels found in this store. Generate label layers first "
                        "or use `--segmentation-source shapes`."
                    )
            else:
                self._segmentation_keys = sorted(str(k) for k in sdata.shapes.keys())
            images_source = self._active_images_sdata
            self._image_keys = sorted(
                str(k)
                for k in getattr(images_source, "images", {}).keys()
                if not is_derived_cache_key(str(k))
            )
            self._image_channels = self._enumerate_image_channels()
            self._publish_shape_keys()
            self._publish_loaded_segmentation_keys()
            self._publish_image_entries()
            self._publish_loaded_image_entries()
            self._publish_cellpose_value_options()
            self._publish_cell_type_options()

            elapsed = time.time() - t0
            mem_after = memory_snapshot_gb()
            summary = (
                f"{ds} metadata loaded in {elapsed:.1f}s | "
                f"images={len(self._image_keys)} "
                f"{self._segmentation_layer_type()}_keys={len(self._segmentation_keys)} | "
                f"RSS {mem_before['rss_gb']:.1f}->{mem_after['rss_gb']:.1f} GB | starting background loads..."
            )
            log.info(summary)
            self._set_status(summary)

            # Kick off the default background loads (images / masks / transcripts).
            self._start_startup_autoload(ds)

        except Exception as exc:
            self._set_status(f"Failed to load {ds}: {exc}")
            log.exception("[%s] Failed to load dataset", ds)
            self.active_dataset = None
            self._active_sdata = None
            self._active_images_sdata = None
            self._segmentation_keys = []
            self._image_keys = []
            self._image_channels = []
            self._publish_shape_keys()
            self._publish_loaded_segmentation_keys()
            self._publish_image_entries()
            self._publish_loaded_image_entries()
            if self._cellpose_value_options_callback is not None:
                self._cellpose_value_options_callback([], [], False)
        finally:
            self._end_progress("dataset")

    def _build_dataset_config(self, platform: str, zarr_path: Path) -> DatasetConfig:
        return DatasetConfig(
            name=platform,
            zarr_path=zarr_path,
            merscope_transform_path=getattr(self.args, "merscope_transform_path", None),
            xenium_spec_path=getattr(self.args, "xenium_spec_path", None),
        )

    def _replace_datasets(self, datasets: dict[str, DatasetConfig], initial: str):
        """Swap in a freshly browsed set of datasets and load the initial one."""
        self._clear_layers()
        self.active_dataset = None
        self._active_sdata = None
        self._active_images_sdata = None
        self.datasets = datasets
        if self._datasets_changed_callback is not None:
            self._datasets_changed_callback(list(datasets.keys()), initial)
        self.load_dataset(initial, force=True)

    def load_paired_dataset(self, merscope_path, xenium_path):
        """Open a MERSCOPE + Xenium pair browsed from the Dataset loader tab."""
        try:
            merscope_path = Path(merscope_path)
            xenium_path = Path(xenium_path)
            for label, path in (("MERSCOPE", merscope_path), ("Xenium", xenium_path)):
                if not path.exists():
                    raise FileNotFoundError(f"{label} zarr path not found: {path}")
                validate_spatialdata_store_compatibility(path)
            datasets = {
                "MERSCOPE": self._build_dataset_config("MERSCOPE", merscope_path),
                "XENIUM": self._build_dataset_config("XENIUM", xenium_path),
            }
        except Exception as exc:
            self._set_status(f"Could not open paired dataset: {exc}")
            log.exception("Failed to open paired dataset")
            return
        self._replace_datasets(datasets, initial="MERSCOPE")

    def load_standalone_dataset(self, platform: str, path):
        """Open a single MERSCOPE or Xenium store browsed from the loader tab."""
        platform = str(platform).upper()
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{platform} zarr path not found: {path}")
            validate_spatialdata_store_compatibility(path)
            datasets = {platform: self._build_dataset_config(platform, path)}
        except Exception as exc:
            self._set_status(f"Could not open {platform} dataset: {exc}")
            log.exception("Failed to open standalone %s dataset", platform)
            return
        self._replace_datasets(datasets, initial=platform)

    def _pyramid_levels_from_element(self, elem) -> list[object]:
        """Return the (c, y, x) DataArrays of a stored image element, coarsest last."""
        return [ensure_cyx(cyx) for _name, cyx in image_scale_dataarrays(elem)]

    def _ensure_image_pyramid_cache(self, image_key: str, image_elem) -> list[object]:
        """Ensure a materialized coarse-level pyramid exists for a single-scale image.

        Returns the coarse (c, y, x) levels (level 1..N, base excluded) read back
        from the persisted zarr cache, or [] if one could not be built. Building
        streams the full-resolution base once to write small mean-downsampled
        levels, so subsequent zoomed-out/mid views read tiny tiles instead of
        re-reading full-resolution chunks.
        """
        if self._active_sdata is None or self.active_dataset is None or da is None:
            return []

        step = max(2, int(self.args.image_pyramid_downsample))
        cache_key = derived_image_pyramid_cache_key(image_key, step)
        expected = {
            "kind": "image_pyramid",
            "source_image_key": str(image_key),
            "downsample": int(step),
            "min_size": int(SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE),
        }
        if (
            not bool(getattr(self.args, "overwrite_derived_caches", False))
            and self._derived_cache_complete("images", cache_key, expected)
            and self._refresh_image_key_from_store(cache_key)
        ):
            return self._pyramid_levels_from_element(self._active_sdata.images[cache_key])

        base_cyx = ensure_cyx(get_scale0_dataarray(image_elem))
        levels = lazy_coarsened_pyramid(base_cyx.data, step=step, reducer=np.mean)
        if len(levels) == 0:
            return []

        transform = get_transformation(image_elem, to_coordinate_system="global")
        channels = channel_labels(base_cyx)
        base_dtype = getattr(base_cyx, "dtype", None)
        pyramid_tree = self._datatree_from_levels(
            levels,
            dims=("c", "y", "x"),
            channels=channels,
            transform=transform,
            dtype=base_dtype,
        )
        Image2DModel.validate(pyramid_tree)
        self._discard_derived_cache_before_write("images", cache_key)
        self._active_sdata.images[cache_key] = pyramid_tree
        self._set_status(
            f"{self.active_dataset} writing image pyramid cache for {image_key} (downsample {step}x)..."
        )
        self._active_sdata.write_element(cache_key, overwrite=False)
        self._mark_derived_cache_complete("images", cache_key, {**expected, "levels": int(len(levels))})
        self._refresh_image_key_from_store(cache_key)
        log.info(
            "[%s] Built image pyramid cache images[%s] from images[%s] downsample=%sx levels=%s",
            self.active_dataset,
            cache_key,
            image_key,
            step,
            len(levels),
        )
        return self._pyramid_levels_from_element(self._active_sdata.images[cache_key])

    def _display_scale_levels_for_image(self, image_key: str, sdata) -> tuple[list[tuple[str, object]], str]:
        """Return (scale_levels, source) for display: stored pyramid, or base plus
        a materialized coarse pyramid for single-scale images."""
        elem = sdata.images[image_key]
        stored = [(name, ensure_cyx(cyx)) for name, cyx in image_scale_dataarrays(elem)]
        if len(stored) == 0:
            return [], "none"
        if len(stored) > 1:
            return stored, "stored"

        if da is not None:
            try:
                coarse = self._ensure_image_pyramid_cache(image_key, elem)
            except Exception as exc:
                log.warning(
                    "[%s] Image pyramid cache build failed for %s (%s); using synthetic fallback.",
                    self.active_dataset,
                    image_key,
                    exc,
                )
                coarse = []
            if coarse:
                extended = stored + [(f"imgpyr{idx + 1}", cyx) for idx, cyx in enumerate(coarse)]
                return extended, "materialized"
        return stored, "single"

    def _prime_image_pyramid_caches(self, ds: str, sdata, only_keys: set[str] | None = None) -> None:
        """Pre-build materialized pyramids for single-scale images (worker-thread safe)."""
        if da is None or sdata is None:
            return
        try:
            image_keys = [str(k) for k in sdata.images.keys() if not is_derived_cache_key(str(k))]
        except Exception:
            return
        if only_keys is not None:
            image_keys = [k for k in image_keys if k in only_keys]
        for image_key in image_keys:
            try:
                if len(image_scale_dataarrays(sdata.images[image_key])) > 1:
                    continue
                self._ensure_image_pyramid_cache(image_key, sdata.images[image_key])
            except Exception as exc:
                log.warning("[%s] Could not build image pyramid cache for %s (%s)", ds, image_key, exc)

    def _default_visible_channels(self, labels: list[str]) -> set[str]:
        """Channels shown by default on load; the rest are loaded but hidden.

        Fewer simultaneously-visible additive layers means fewer per-frame reads
        and blend passes. Every channel stays a toggleable layer in napari.
        """
        configured = getattr(self.args, "visible_channels", None)
        if configured:
            wanted = {token.strip().lower() for token in str(configured).split(",") if token.strip()}
            selected = {label for label in labels if str(label).lower() in wanted}
            if selected:
                return selected
        for label in labels:
            if "dapi" in str(label).lower():
                return {label}
        return {labels[0]} if labels else set()

    def _add_image_layers(
        self, ds: str, sdata, x_transform, y_transform, only_channels: set[tuple[str, str]] | None = None
    ) -> dict[str, int]:
        visible = not self.args.hide_images
        total_layers = 0
        failed_keys = 0

        try:
            image_keys = [str(k) for k in sdata.images.keys() if not is_derived_cache_key(str(k))]
        except Exception as exc:
            log.warning("[%s] Could not enumerate images; skipping image loading (%s)", ds, exc)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        if only_channels is not None:
            wanted_keys = {k for k, _c in only_channels}
            image_keys = [k for k in image_keys if k in wanted_keys]

        if len(image_keys) == 0:
            log.info("[%s] No images found in SpatialData; continuing without image layers.", ds)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        for image_key in image_keys:
            try:
                scale_levels, image_source = self._display_scale_levels_for_image(image_key, sdata)
                if len(scale_levels) == 0:
                    raise ValueError("image has no readable scale levels")

                base_scale_name, base_image_cyx = scale_levels[0]
                labels = channel_labels(base_image_cyx)
                default_visible = self._default_visible_channels(labels)

                x_coords = (
                    np.asarray(base_image_cyx.coords["x"].values)
                    if "x" in base_image_cyx.coords
                    else None
                )
                y_coords = (
                    np.asarray(base_image_cyx.coords["y"].values)
                    if "y" in base_image_cyx.coords
                    else None
                )
                affine = build_napari_affine_from_px_to_um(
                    x_transform=x_transform,
                    y_transform=y_transform,
                    x_coords=x_coords,
                    y_coords=y_coords,
                )

                for chan_idx, chan_name in enumerate(labels):
                    if only_channels is not None and (image_key, str(chan_name)) not in only_channels:
                        continue
                    layer_name = make_layer_name(ds, "image", image_key, chan_name)
                    channel_levels = [
                        image_cyx.isel(c=chan_idx).data
                        for _scale_name, image_cyx in scale_levels
                    ]
                    pyramid_source = image_source
                    if len(channel_levels) == 1:
                        channel_levels = lazy_subsampled_pyramid(channel_levels[0])
                        pyramid_source = "synthetic" if len(channel_levels) > 1 else "single"

                    multiscale = len(channel_levels) > 1
                    ch_data = channel_levels if multiscale else channel_levels[0]
                    cmap = image_colormap_for_channel(chan_name, chan_idx)
                    # Explicitly-selected channels are shown; the startup "all"
                    # load only shows the default (DAPI-like) channel.
                    if only_channels is not None:
                        chan_visible = visible
                    else:
                        chan_visible = visible and (chan_name in default_visible)

                    add_kwargs = dict(
                        name=layer_name,
                        affine=affine,
                        colormap=cmap,
                        blending="additive",
                        cache=NAPARI_DASK_CACHE_ENABLED,
                        opacity=self.args.image_opacity,
                        visible=chan_visible,
                    )
                    if multiscale:
                        add_kwargs["multiscale"] = True

                    contrast_limits = contrast_limits_from_dtype(channel_levels[0])
                    if contrast_limits is not None:
                        add_kwargs["contrast_limits"] = contrast_limits

                    self.viewer.add_image(ch_data, **add_kwargs)
                    total_layers += 1
                    shapes = [tuple(int(axis) for axis in level.shape) for level in channel_levels]
                    base_chunks = getattr(channel_levels[0], "chunksize", None) or getattr(
                        channel_levels[0], "chunks", None
                    )
                    log.info(
                        "[%s] Added image layer %s from %s levels=%s source=%s base=%s dtype=%s "
                        "base_shape=%s base_chunks=%s nchan=%s",
                        ds,
                        layer_name,
                        image_key,
                        len(channel_levels),
                        pyramid_source,
                        base_scale_name,
                        getattr(channel_levels[0], "dtype", "unknown"),
                        shapes[0] if shapes else None,
                        base_chunks,
                        len(labels),
                    )
                    log.debug("[%s] Image layer %s level shapes=%s", ds, layer_name, shapes)
            except Exception as exc:
                failed_keys += 1
                log.warning(
                    "[%s] Skipping image key '%s' due to load error (%s)",
                    ds,
                    image_key,
                    exc,
                )
                continue

        return {"layers": total_layers, "failed_keys": failed_keys, "skipped": False}

    def _label_key_for_shape_key(self, shape_key: str) -> str:
        shape_key = str(shape_key)
        if self._active_sdata is not None and shape_key in self._active_sdata.labels:
            return shape_key
        return f"{shape_key}_labels"

    def _read_labels_from_store(self):
        if self.active_dataset is None:
            return None
        cfg = self.datasets[self.active_dataset]
        return sd.read_zarr(str(cfg.zarr_path), selection=("labels",))

    def _read_images_from_store(self):
        if self.active_dataset is None:
            return None
        cfg = self.datasets[self.active_dataset]
        return sd.read_zarr(str(cfg.zarr_path), selection=("images",))

    def _read_shapes_from_store(self):
        if self.active_dataset is None:
            return None
        cfg = self.datasets[self.active_dataset]
        return sd.read_zarr(str(cfg.zarr_path), selection=("shapes",))

    def _refresh_label_key_from_store(self, label_key: str) -> bool:
        if self._active_sdata is None:
            return False
        labels_sdata = self._read_labels_from_store()
        if labels_sdata is None or label_key not in labels_sdata.labels:
            return False
        self._active_sdata.labels[label_key] = labels_sdata.labels[label_key]
        return True

    def _refresh_image_key_from_store(self, image_key: str) -> bool:
        if self._active_sdata is None:
            return False
        images_sdata = self._read_images_from_store()
        if images_sdata is None or image_key not in images_sdata.images:
            return False
        self._active_sdata.images[image_key] = images_sdata.images[image_key]
        return True

    def _derived_cache_path(self, element_type: str, key: str) -> Path | None:
        if self.active_dataset is None:
            return None
        cfg = self.datasets[self.active_dataset]
        return cfg.zarr_path / str(element_type) / str(key)

    def _remove_label_from_parent_metadata(self, label_key: str):
        if self.active_dataset is None:
            return
        labels_path = self.datasets[self.active_dataset].zarr_path / "labels"
        if not labels_path.exists():
            return
        try:
            group = zarr.open_group(str(labels_path), mode="a")
            labels = group.attrs.get("labels", None)
            if isinstance(labels, list) and label_key in labels:
                group.attrs["labels"] = [name for name in labels if name != label_key]
        except Exception as exc:
            log.debug("[%s] Could not update labels metadata for %s (%s)", self.active_dataset, label_key, exc)

    def _discard_derived_cache_before_write(self, element_type: str, key: str):
        """Remove a stale private cache so SpatialData can write it fresh."""
        if not is_derived_cache_key(key):
            raise ValueError(f"Refusing to delete non-derived cache element: {key}")
        if self._active_sdata is None:
            return

        collection = None
        if element_type == "labels":
            collection = self._active_sdata.labels
        elif element_type == "images":
            collection = self._active_sdata.images
        if collection is not None and key in collection:
            try:
                del collection[key]
            except Exception:
                pass

        path = self._derived_cache_path(element_type, key)
        if path is None or not path.exists():
            return
        if not path.is_dir():
            raise ValueError(f"Refusing to delete non-directory derived cache path: {path}")
        shutil.rmtree(path)
        if element_type == "labels":
            self._remove_label_from_parent_metadata(key)
        log.info("[%s] Removed stale derived cache %s/%s before rewrite", self.active_dataset, element_type, key)

    def _derived_cache_attrs(self, element_type: str, key: str) -> dict:
        path = self._derived_cache_path(element_type, key)
        if path is None or not path.exists():
            return {}
        try:
            group = zarr.open_group(str(path), mode="r")
            value = group.attrs.get(DERIVED_CACHE_ATTR, {})
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _derived_cache_complete(self, element_type: str, key: str, expected: dict[str, object]) -> bool:
        if bool(getattr(self.args, "overwrite_derived_caches", False)):
            return False
        attrs = self._derived_cache_attrs(element_type, key)
        if not attrs.get("complete"):
            return False
        if attrs.get("version") != DERIVED_CACHE_VERSION:
            return False
        for expected_key, expected_value in expected.items():
            if attrs.get(expected_key) != expected_value:
                return False
        return True

    def _mark_derived_cache_complete(self, element_type: str, key: str, attrs: dict[str, object]):
        path = self._derived_cache_path(element_type, key)
        if path is None:
            return
        group = zarr.open_group(str(path), mode="a")
        payload = {
            "version": DERIVED_CACHE_VERSION,
            "complete": True,
            **attrs,
        }
        group.attrs[DERIVED_CACHE_ATTR] = payload

    def _raster_scale_levels(self, elem) -> list[tuple[str, object]]:
        levels: list[tuple[str, object]] = []
        for scale_name, scale_da in image_scale_dataarrays(elem):
            if hasattr(scale_da, "dims"):
                for dim in ("z", "Z"):
                    if dim in scale_da.dims:
                        if int(scale_da.sizes[dim]) != 1:
                            raise ValueError(
                                f"Unsupported raster dims with non-singleton {dim}: "
                                f"{tuple(str(d) for d in scale_da.dims)}"
                            )
                        scale_da = scale_da.isel({dim: 0}, drop=True)
                dims = tuple(str(d) for d in scale_da.dims)
                if "c" in dims and "y" in dims and "x" in dims:
                    scale_da = scale_da.transpose("c", "y", "x")
                elif "y" in dims and "x" in dims:
                    scale_da = scale_da.transpose("y", "x")

            level_data = scale_da.data if hasattr(scale_da, "data") else scale_da
            if len(getattr(level_data, "shape", ())) in (2, 3):
                levels.append((str(scale_name), level_data))
        return levels

    def _napari_affine_from_element(self, elem) -> np.ndarray:
        tf = get_transformation(elem, to_coordinate_system="global")
        m = tf.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        return np.array(
            [
                [float(m[1, 1]), float(m[1, 0]), float(m[1, 2])],
                [float(m[0, 1]), float(m[0, 0]), float(m[0, 2])],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _label_display_affine(self, label_elem) -> np.ndarray:
        """Napari affine for a label raster, kept consistent with the images.

        Labels are rasterized on the morphology image's pixel grid, so their
        pixel->micron scale must match the image layers (and the transcripts),
        which the viewer derives from the resolved dataset ``pixel_size``
        (:meth:`resolve_dataset_mask_affine`). We deliberately do NOT trust the
        transformation stored on the label element: some upstream writers stamp
        Xenium masks with a MERSCOPE-default 0.108 um/px scale even though the
        raster is on the 0.2125 um/px Xenium grid, which shrinks the masks to
        the top-left quadrant of the image. Rebuilding the affine from the
        resolved pixel_size and the label's own (pixel-index) coordinates fixes
        that and is a no-op for MERSCOPE, where the two already agree.
        """
        if self._x_transform is None or self._y_transform is None:
            return self._napari_affine_from_element(label_elem)
        try:
            da_2d = get_scale0_dataarray(label_elem)
            coords = getattr(da_2d, "coords", {})
            x_coords = np.asarray(coords["x"].values) if "x" in coords else None
            y_coords = np.asarray(coords["y"].values) if "y" in coords else None
        except Exception as exc:
            log.warning(
                "[%s] Could not read label grid coords for %s (%s); "
                "falling back to stored transform.",
                self.active_dataset,
                getattr(label_elem, "name", "<label>"),
                exc,
            )
            return self._napari_affine_from_element(label_elem)
        return build_napari_affine_from_px_to_um(
            x_transform=self._x_transform,
            y_transform=self._y_transform,
            x_coords=x_coords,
            y_coords=y_coords,
        )

    def _image_grid_for_labels(self) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, Affine]:
        if self.active_dataset is None:
            raise RuntimeError("No active dataset.")
        ds = self.active_dataset
        self._ensure_images_loaded(ds)
        if self._active_images_sdata is None or len(self._active_images_sdata.images) == 0:
            raise RuntimeError("Cannot build labels without an image grid.")
        if self._x_transform is None or self._y_transform is None:
            raise RuntimeError("Image transform is not initialized.")

        image_key = next(iter(self._active_images_sdata.images.keys()))
        base_image_cyx = ensure_cyx(get_scale0_dataarray(self._active_images_sdata.images[image_key]))
        height = int(base_image_cyx.sizes["y"])
        width = int(base_image_cyx.sizes["x"])

        x_coords = np.asarray(base_image_cyx.coords["x"].values) if "x" in base_image_cyx.coords else None
        y_coords = np.asarray(base_image_cyx.coords["y"].values) if "y" in base_image_cyx.coords else None
        napari_affine = affine_matrix_from_px_to_um(
            x_transform=self._x_transform,
            y_transform=self._y_transform,
            x_coords=x_coords,
            y_coords=y_coords,
        )
        spatialdata_affine = Affine(
            [
                [float(napari_affine[1, 1]), float(napari_affine[1, 0]), float(napari_affine[1, 2])],
                [float(napari_affine[0, 1]), float(napari_affine[0, 0]), float(napari_affine[0, 2])],
                [0.0, 0.0, 1.0],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

        chunk = int(self.args.label_chunk_size)
        chunks = (min(chunk, height), min(chunk, width))
        return (height, width), chunks, napari_affine, spatialdata_affine

    def _create_label_element(self, label_key: str, shape: tuple[int, int], chunks: tuple[int, int], transform: Affine):
        if da is None:
            raise RuntimeError("dask is required to create lazy label arrays.")
        data = da.zeros(shape, chunks=chunks, dtype=np.uint32)
        label_da = xr.DataArray(data, dims=("y", "x"))
        return Labels2DModel.parse(label_da, transformations={"global": transform})

    def _write_empty_label_element(
        self,
        label_key: str,
        shape: tuple[int, int],
        chunks: tuple[int, int],
        transform: Affine,
        overwrite: bool,
    ):
        if self._active_sdata is None:
            raise RuntimeError("No active SpatialData object.")
        label_elem = self._create_label_element(label_key, shape, chunks, transform)
        self._active_sdata.labels[label_key] = label_elem
        self._active_sdata.write_element(label_key, overwrite=overwrite)

    def _label_cache_attrs(self, label_key: str) -> dict:
        if self.active_dataset is None:
            return {}
        cfg = self.datasets[self.active_dataset]
        label_path = cfg.zarr_path / "labels" / label_key
        if not label_path.exists():
            return {}
        try:
            group = zarr.open_group(str(label_path), mode="r")
            value = group.attrs.get(LABEL_CACHE_ATTR, {})
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _label_cache_is_complete(self, label_key: str, shape_key: str) -> bool:
        attrs = self._label_cache_attrs(label_key)
        return bool(
            attrs.get("complete")
            and attrs.get("source_shape_key") == str(shape_key)
            and attrs.get("version") == 1
        )

    def _mark_label_cache_complete(self, label_key: str, shape_key: str, shape: tuple[int, int], chunks: tuple[int, int]):
        if self.active_dataset is None:
            return
        cfg = self.datasets[self.active_dataset]
        label_path = cfg.zarr_path / "labels" / label_key
        group = zarr.open_group(str(label_path), mode="a")
        group.attrs[LABEL_CACHE_ATTR] = {
            "version": 1,
            "complete": True,
            "source_shape_key": str(shape_key),
            "shape": [int(shape[0]), int(shape[1])],
            "chunks": [int(chunks[0]), int(chunks[1])],
        }

    def _discard_label_cache_before_write(self, label_key: str, shape_key: str):
        if self._active_sdata is None:
            return
        path = self._derived_cache_path("labels", label_key)
        if path is None or not path.exists():
            return

        attrs = self._label_cache_attrs(label_key)
        generated_name = str(label_key) == f"{shape_key}_labels"
        has_matching_cache_attr = attrs.get("source_shape_key") == str(shape_key)
        explicit_overwrite = bool(getattr(self.args, "overwrite_labels", False))
        if not (generated_name or has_matching_cache_attr or explicit_overwrite):
            raise ValueError(
                f"Refusing to delete existing label element '{label_key}'. "
                "Pass --overwrite-labels if this should be rebuilt."
            )

        if label_key in self._active_sdata.labels:
            try:
                del self._active_sdata.labels[label_key]
            except Exception:
                pass
        if not path.is_dir():
            raise ValueError(f"Refusing to delete non-directory label cache path: {path}")
        shutil.rmtree(path)
        self._remove_label_from_parent_metadata(label_key)
        log.info("[%s] Removed stale label cache %s before rewrite", self.active_dataset, label_key)

    def _rasterize_label_payload(
        self,
        shape_key: str,
        label_key: str,
        shape: tuple[int, int],
        chunk_shape: tuple[int, int],
        napari_affine: np.ndarray,
    ) -> int:
        if self._active_sdata is None or self.active_dataset is None:
            return 0
        if shape_key not in self._active_sdata.shapes:
            raise KeyError(f"Shape key '{shape_key}' not found in current dataset.")

        ds = self.active_dataset
        cfg = self.datasets[ds]
        label_path = cfg.zarr_path / "labels" / label_key / "s0"
        label_arr = zarr.open(str(label_path), mode="r+")
        inv_affine = np.linalg.inv(np.asarray(napari_affine, dtype=float))
        gdf = self._active_sdata.shapes[shape_key].reset_index(drop=True)
        label_ids = np.arange(1, len(gdf) + 1, dtype=np.uint32)

        height, width = shape
        chunk_h, chunk_w = chunk_shape
        total_chunks = int(np.ceil(height / chunk_h) * np.ceil(width / chunk_w))
        touched_chunks = 0
        written_cells = 0
        processed = 0
        t0 = time.time()

        for y0 in range(0, height, chunk_h):
            y1 = min(y0 + chunk_h, height)
            for x0 in range(0, width, chunk_w):
                x1 = min(x0 + chunk_w, width)
                processed += 1
                bounds = pixel_window_global_bounds(napari_affine, y0, y1, x0, x1)
                candidates = query_geometries_for_bounds(gdf, bounds)
                if len(candidates) == 0:
                    continue

                ids = label_ids[candidates.index.to_numpy(dtype=np.int64, copy=False)]
                tile = rasterize_geometries_chunk(
                    candidates.geometry,
                    ids,
                    shape=(y1 - y0, x1 - x0),
                    inv_affine=inv_affine,
                    y0=y0,
                    x0=x0,
                    dtype=np.uint32,
                )
                if np.any(tile):
                    label_arr[y0:y1, x0:x1] = tile
                    touched_chunks += 1
                    written_cells += int(np.count_nonzero(tile))

                if processed % 10 == 0:
                    elapsed = time.time() - t0
                    self._set_status(
                        f"{ds} building labels {shape_key}: chunk {processed:,}/{total_chunks:,}, "
                        f"written_chunks={touched_chunks:,}, elapsed={elapsed:.1f}s"
                    )

        log.info(
            "[%s] Rasterized labels[%s] from shapes[%s]: shape=%s chunks=%s touched_chunks=%s nonzero_pixels=%s",
            ds,
            label_key,
            shape_key,
            shape,
            chunk_shape,
            f"{touched_chunks:,}",
            f"{written_cells:,}",
        )
        return int(len(gdf))

    def ensure_label_for_shape_key(self, shape_key: str) -> str:
        if self._active_sdata is None or self.active_dataset is None:
            raise RuntimeError("No active dataset.")

        label_key = self._label_key_for_shape_key(shape_key)
        exact_existing_label = label_key == str(shape_key)
        if (
            (not self.args.overwrite_labels)
            and exact_existing_label
            and label_key in self._active_sdata.labels
        ):
            return label_key
        if (
            (not self.args.overwrite_labels)
            and exact_existing_label
            and self._refresh_label_key_from_store(label_key)
        ):
            return label_key
        if (
            (not self.args.overwrite_labels)
            and self._refresh_label_key_from_store(label_key)
            and self._label_cache_is_complete(label_key, shape_key)
        ):
            return label_key

        mem_before = memory_snapshot_gb()
        t0 = time.time()
        self._set_status(f"{self.active_dataset} building labels for {shape_key}...")
        shape, chunks, napari_affine, spatialdata_affine = self._image_grid_for_labels()
        self._discard_label_cache_before_write(label_key, shape_key)
        self._write_empty_label_element(
            label_key=label_key,
            shape=shape,
            chunks=chunks,
            transform=spatialdata_affine,
            overwrite=False,
        )
        n_labels = self._rasterize_label_payload(
            shape_key=shape_key,
            label_key=label_key,
            shape=shape,
            chunk_shape=chunks,
            napari_affine=napari_affine,
        )
        self._mark_label_cache_complete(label_key, shape_key, shape, chunks)
        self._refresh_label_key_from_store(label_key)
        mem_after = memory_snapshot_gb()
        elapsed = time.time() - t0
        log.info(
            "[%s] Built cached label layer %s from %s labels in %.1fs | RSS %.1f->%.1f GB",
            self.active_dataset,
            label_key,
            f"{n_labels:,}",
            elapsed,
            mem_before["rss_gb"],
            mem_after["rss_gb"],
        )
        return label_key

    def _datatree_from_levels(
        self,
        levels: list[object],
        dims: tuple[str, ...],
        transform,
        channels: list[str] | None = None,
        dtype=None,
    ):
        if da is None:
            raise RuntimeError("dask is required to build derived raster caches.")
        arrays = []
        for level in levels:
            arr = da.asarray(level)
            if dtype is not None:
                arr = arr.astype(dtype)
            arrays.append(arr)
        tree = dask_arrays_to_datatree(arrays, dims=dims, channels=channels)
        set_transformation(tree, {"global": transform}, set_all=True)
        return tree

    def _ensure_label_outline_cache(self, label_key: str, width: int) -> str:
        if self._active_sdata is None:
            raise RuntimeError("No active dataset.")
        if is_derived_cache_key(label_key):
            return label_key

        width = max(1, int(width))
        cache_key = derived_outline_cache_key(label_key, width)
        expected = {
            "kind": "label_outline",
            "source_label_key": str(label_key),
            "width": int(width),
        }
        if (
            not bool(getattr(self.args, "overwrite_labels", False))
            and self._derived_cache_complete("labels", cache_key, expected)
            and self._refresh_label_key_from_store(cache_key)
        ):
            return cache_key

        if label_key not in self._active_sdata.labels:
            if not self._refresh_label_key_from_store(label_key):
                raise KeyError(f"Label key '{label_key}' not found in current dataset.")

        label_elem = self._active_sdata.labels[label_key]
        label_scale_levels = [
            (scale_name, level_data)
            for scale_name, level_data in self._raster_scale_levels(label_elem)
            if len(getattr(level_data, "shape", ())) == 2
        ]
        if len(label_scale_levels) == 0:
            raise ValueError(f"Expected 2D labels for {label_key}, found no readable 2D scale levels")

        if len(label_scale_levels) > 1:
            outline_levels = lazy_outline_pyramid_from_label_levels(
                [level_data for _scale_name, level_data in label_scale_levels],
                width=width,
            )
            source = "stored"
        else:
            outline_levels = lazy_outline_pyramid(label_scale_levels[0][1], width=width)
            source = "synthetic" if len(outline_levels) > 1 else "single"

        tf = get_transformation(label_elem, to_coordinate_system="global")
        outline_tree = self._datatree_from_levels(
            outline_levels,
            dims=("y", "x"),
            transform=tf,
            dtype=np.uint8,
        )
        Labels2DModel.validate(outline_tree)
        self._discard_derived_cache_before_write("labels", cache_key)
        self._active_sdata.labels[cache_key] = outline_tree
        self._set_status(f"{self.active_dataset} writing cached outline pyramid for {label_key}...")
        self._active_sdata.write_element(cache_key, overwrite=False)
        self._mark_derived_cache_complete(
            "labels",
            cache_key,
            {
                **expected,
                "source": source,
                "levels": int(len(outline_levels)),
                "source_shapes": [
                    [int(axis) for axis in getattr(level_data, "shape")]
                    for _scale_name, level_data in label_scale_levels
                ],
            },
        )
        self._refresh_label_key_from_store(cache_key)
        log.info(
            "[%s] Built cached outline pyramid labels[%s] from labels[%s] levels=%s source=%s width=%s",
            self.active_dataset,
            cache_key,
            label_key,
            len(outline_levels),
            source,
            width,
        )
        return cache_key

    def _prepare_label_outline_display(self, label_key: str) -> dict[str, object] | None:
        """Build/refresh the outline pyramid and gather display data (thread-safe).

        This does the heavy work (outline pyramid build + zarr I/O + lazy scale
        reads) but performs no napari layer operations, so it is safe to call
        from a worker thread. Pair it with :meth:`_finish_label_layer` on the GUI
        thread.
        """
        if self._active_sdata is None or self.active_dataset is None:
            return None
        if label_key not in self._active_sdata.labels:
            if not self._refresh_label_key_from_store(label_key):
                raise KeyError(f"Label key '{label_key}' not found in current dataset.")

        outline_width = max(1, int(self.args.label_contour_width))
        display_label_key = self._ensure_label_outline_cache(label_key, outline_width)
        label_elem = self._active_sdata.labels[display_label_key]
        label_scale_levels = [
            (scale_name, level_data)
            for scale_name, level_data in self._raster_scale_levels(label_elem)
            if len(getattr(level_data, "shape", ())) == 2
        ]

        if len(label_scale_levels) == 0:
            raise ValueError(f"Expected 2D cached outlines for {display_label_key}, found no readable 2D levels")

        data = label_scale_levels[0][1]
        if len(getattr(data, "shape", ())) != 2:
            raise ValueError(f"Expected 2D cached outlines for {display_label_key}, got shape {getattr(data, 'shape', None)}")

        napari_affine = self._label_display_affine(label_elem)
        outline_levels = [level_data for _scale_name, level_data in label_scale_levels]
        outline_data = outline_levels if len(outline_levels) > 1 else outline_levels[0]
        return {
            "display_label_key": display_label_key,
            "napari_affine": napari_affine,
            "outline_data": outline_data,
            "n_levels": len(outline_levels),
            "outline_width": outline_width,
            "base_shape": tuple(int(x) for x in data.shape),
            "base_dtype": getattr(data, "dtype", "unknown"),
        }

    def _finish_label_layer(self, ds: str, label_key: str, prepared: dict[str, object] | None) -> int:
        """Add the prepared label outline layer to the viewer (GUI thread only)."""
        if prepared is None:
            return 0
        n_levels = int(prepared["n_levels"])
        layer_name = make_layer_name(ds, "labels", label_key)
        self._remove_layer_by_name(layer_name)
        layer_color = np.asarray(stable_layer_color(label_key, alpha=1.0), dtype=np.float32)
        color_map = transparent_colormap(f"{label_key}_outline", layer_color, alpha=float(self.args.shape_opacity))

        seg_layer = self.viewer.add_image(
            prepared["outline_data"],
            name=layer_name,
            affine=prepared["napari_affine"],
            colormap=color_map,
            contrast_limits=(0.0, 1.0),
            cache=NAPARI_DASK_CACHE_ENABLED,
            interpolation2d=str(getattr(self.args, "label_interpolation", "linear")),
            multiscale=n_levels > 1,
            opacity=self.args.shape_opacity,
            blending="additive",
            visible=not self.args.hide_shapes,
        )
        # Hiding the ProSeg (cell-inspector) layer disables cell picking and drops
        # any current highlights; showing it re-enables clicking.
        try:
            seg_layer.events.visible.connect(self._on_segmentation_visibility_changed)
        except Exception:
            pass
        # Give the new outline the crisp/linear interpolation matching the current
        # zoom (a fresh layer would otherwise start at the base 'linear' setting).
        self._apply_label_zoom_interpolation()
        log.info(
            "[%s] Added label outline layer %s from cache=%s shape=%s dtype=%s levels=%s width=%s source=cache",
            ds,
            label_key,
            prepared["display_label_key"],
            prepared["base_shape"],
            prepared["base_dtype"],
            n_levels,
            prepared["outline_width"],
        )
        return 1

    def _resolve_points_columns(self):
        if self._active_sdata is None or self.active_dataset is None:
            raise RuntimeError("No active dataset.")
        if len(self._active_sdata.points) == 0:
            raise RuntimeError("No points available in SpatialData.")

        points_key = list(self._active_sdata.points.keys())[0]
        points_obj = self._active_sdata.points[points_key]
        x_col = first_existing_col(points_obj, ["x", "x_micron", "global_x", "x_location", "observed_x"])
        y_col = first_existing_col(points_obj, ["y", "y_micron", "global_y", "y_location", "observed_y"])
        assignment_col = first_existing_col(points_obj, ["assignment", "cell", "cell_id"])

        if x_col is None or y_col is None:
            raise KeyError(f"Could not resolve x/y columns in points[{points_key}]")
        return points_key, points_obj, x_col, y_col, assignment_col

    def load_selected_labels(self, dataset_name: str, shape_keys: list[str]):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return

        ds = self.active_dataset
        keys = [str(k) for k in shape_keys]
        if not keys:
            self._set_status("No segmentation selected.")
            return

        self._label_build_generation += 1
        generation = self._label_build_generation
        mem_before = memory_snapshot_gb()
        self._begin_progress(
            "masks",
            f"{ds}: loading/computing cell segmentation ({', '.join(keys)})...",
        )

        def compute():
            # Heavy: rasterizes shapes to labels (first run) and builds the
            # outline pyramid, both with zarr I/O. Returns display-ready specs;
            # the napari layers are added on the GUI thread in the apply step.
            t0 = time.time()
            specs: list[dict[str, object]] = []
            total_labels = 0
            for key in keys:
                if self._segmentation_source() == "labels" and key in self._active_sdata.labels:
                    label_key = key
                else:
                    label_key = self.ensure_label_for_shape_key(key)
                prepared = self._prepare_label_outline_display(label_key)
                try:
                    n_labels = int(len(self._active_sdata.shapes[key]))
                except Exception:
                    n_labels = 1
                specs.append({"label_key": label_key, "prepared": prepared, "n_labels": n_labels})
                total_labels += n_labels
            return {"specs": specs, "total_labels": total_labels, "build_seconds": time.time() - t0}

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_label_build_error(generation, ds, mem_before, exc)
                return
            self._apply_label_build(generation, ds, mem_before, payload)
            return

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, gen=generation, d=ds, mb=mem_before: self._apply_label_build(gen, d, mb, payload)
        )
        worker.errored.connect(
            lambda exc, gen=generation, d=ds, mb=mem_before: self._handle_label_build_error(gen, d, mb, exc)
        )
        self._label_build_worker = worker
        worker.start()

    def _apply_label_build(self, generation: int, ds: str, mem_before: dict[str, float], payload: dict[str, object]):
        if generation != self._label_build_generation or self.active_dataset != ds:
            return
        self._label_build_worker = None
        self._end_progress("masks")
        added_layers = 0
        for spec in payload["specs"]:
            added = self._finish_label_layer(ds, str(spec["label_key"]), spec["prepared"])
            if added > 0:
                added_layers += 1
        self._publish_loaded_segmentation_keys()
        mem_after = memory_snapshot_gb()
        self._set_status(
            f"{ds} loaded label outline layer(s)={added_layers}, "
            f"labels={int(payload['total_labels']):,} | "
            f"RSS {mem_before['rss_gb']:.1f}->{mem_after['rss_gb']:.1f} GB "
            f"(build={float(payload.get('build_seconds', 0.0)):.1f}s)"
        )

    def _handle_label_build_error(self, generation: int, ds: str, mem_before: dict[str, float], exc):
        if generation != self._label_build_generation:
            return
        self._label_build_worker = None
        self._end_progress("masks")
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{ds} label outline build failed: {message}")
        log.error("[%s] Label outline build failed: %s", ds, message)

    def _label_pyramid_levels_from_element(self, elem) -> list[object]:
        """Return the 2D label DataArrays of a stored label element, coarsest last."""
        return [
            level_data
            for _scale_name, level_data in self._raster_scale_levels(elem)
            if len(getattr(level_data, "shape", ())) == 2
        ]

    def _ensure_label_pyramid_cache(self, label_key: str, step: int) -> list[object]:
        """Ensure a materialized max-pooled label pyramid exists for a single-scale
        label element, returning the coarse levels (base excluded).

        Mirrors the image pyramid cache but uses ``np.max`` coarsening so label
        ids survive downsampling — this keeps the value-overlay colouring exactly
        what the previous lazy ``lazy_label_pyramid`` produced, just persisted so
        zoomed-out views no longer re-read the full-resolution label array.
        """
        if self._active_sdata is None or self.active_dataset is None or da is None:
            return []
        if is_derived_cache_key(label_key):
            return []

        step = max(2, int(step))
        cache_key = derived_label_pyramid_cache_key(label_key, step)
        expected = {
            "kind": "label_pyramid",
            "source_label_key": str(label_key),
            "downsample": int(step),
            "min_size": int(LABEL_OUTLINE_PYRAMID_MIN_SIZE),
        }
        if (
            not bool(getattr(self.args, "overwrite_derived_caches", False))
            and self._derived_cache_complete("labels", cache_key, expected)
            and self._refresh_label_key_from_store(cache_key)
        ):
            return self._label_pyramid_levels_from_element(self._active_sdata.labels[cache_key])

        if label_key not in self._active_sdata.labels and not self._refresh_label_key_from_store(label_key):
            raise KeyError(f"Label key '{label_key}' not found in current dataset.")

        label_elem = self._active_sdata.labels[label_key]
        base_levels = self._label_pyramid_levels_from_element(label_elem)
        if len(base_levels) == 0:
            raise ValueError(f"Expected 2D labels for {label_key}, found no readable 2D levels")

        base = base_levels[0]
        levels = lazy_coarsened_pyramid(base, step=step, reducer=np.max, min_size=LABEL_OUTLINE_PYRAMID_MIN_SIZE)
        if len(levels) == 0:
            return []

        transform = get_transformation(label_elem, to_coordinate_system="global")
        pyramid_tree = self._datatree_from_levels(
            levels,
            dims=("y", "x"),
            transform=transform,
            dtype=getattr(base, "dtype", None),
        )
        Labels2DModel.validate(pyramid_tree)
        self._discard_derived_cache_before_write("labels", cache_key)
        self._active_sdata.labels[cache_key] = pyramid_tree
        self._set_status(
            f"{self.active_dataset} writing label pyramid cache for {label_key} (downsample {step}x)..."
        )
        self._active_sdata.write_element(cache_key, overwrite=False)
        self._mark_derived_cache_complete("labels", cache_key, {**expected, "levels": int(len(levels))})
        self._refresh_label_key_from_store(cache_key)
        log.info(
            "[%s] Built label pyramid cache labels[%s] from labels[%s] downsample=%sx levels=%s",
            self.active_dataset,
            cache_key,
            label_key,
            step,
            len(levels),
        )
        return self._label_pyramid_levels_from_element(self._active_sdata.labels[cache_key])

    def _build_cellpose_label_display(self, label_key: str) -> tuple[list[object], np.ndarray]:
        """Return (label_levels, napari_affine) for the cell-value overlay.

        Uses the stored pyramid when present, otherwise the lazy base plus a
        materialized max-pooled coarse pyramid, so pan/zoom over the value
        overlay reads small tiles instead of the full-resolution label array.
        """
        if label_key not in self._active_sdata.labels and not self._refresh_label_key_from_store(label_key):
            raise KeyError(f"Label key '{label_key}' not found in current dataset.")
        label_elem = self._active_sdata.labels[label_key]
        stored = self._label_pyramid_levels_from_element(label_elem)
        if len(stored) == 0:
            raise ValueError(f"labels[{label_key}] has no readable 2D levels")

        levels = stored
        if len(stored) == 1 and da is not None:
            step = max(2, int(self.args.image_pyramid_downsample))
            try:
                coarse = self._ensure_label_pyramid_cache(label_key, step)
            except Exception as exc:
                log.warning(
                    "[%s] Label pyramid cache build failed for %s (%s); using lazy fallback.",
                    self.active_dataset,
                    label_key,
                    exc,
                )
                coarse = lazy_label_pyramid(stored[0])[1:]
            if coarse:
                levels = [stored[0]] + list(coarse)
        return levels, self._label_display_affine(label_elem)

    def _ensure_cellpose_label_key(self) -> str:
        if self._active_sdata is None or self.active_dataset is None:
            raise RuntimeError("No active dataset.")

        if CELLPOSE_LABEL_KEY in self._active_sdata.labels or self._refresh_label_key_from_store(CELLPOSE_LABEL_KEY):
            return CELLPOSE_LABEL_KEY

        if CELLPOSE_SHAPE_KEY not in self._active_sdata.shapes:
            shapes_sdata = self._read_shapes_from_store()
            if shapes_sdata is not None:
                for key, value in shapes_sdata.shapes.items():
                    self._active_sdata.shapes[key] = value

        return self.ensure_label_for_shape_key(CELLPOSE_SHAPE_KEY)

    def _compute_cellpose_value_payload(
        self,
        zarr_path: Path,
        channel: str,
        statistic: str,
        colormap_name: str,
    ) -> dict[str, object]:
        quantification = load_cellpose_quantification_values(
            zarr_path,
            channel=channel,
            statistic=statistic,
        )
        colors_rgba = ensure_colormap(colormap_name).map(
            np.linspace(0.0, 1.0, CELLPOSE_VALUE_BINS, dtype=np.float32)
        )
        mapping = build_binned_label_color_dict(
            quantification.label_ids,
            quantification.values,
            colors_rgba,
        )
        return {
            "channel": str(channel),
            "statistic": str(statistic),
            "colormap_name": str(colormap_name),
            "feature": quantification.feature,
            "mapping": mapping,
            "colormap": DirectLabelColormap(color_dict=mapping.color_dict),
        }

    def _apply_cellpose_value_payload(self, generation: int, payload: dict[str, object]):
        if generation != self._cellpose_value_generation or self.active_dataset != "MERSCOPE":
            return
        self._cellpose_value_worker = None
        if self._active_sdata is None:
            return

        label_key = str(payload["label_key"])
        label_levels = payload["label_levels"]
        napari_affine = payload["napari_affine"]
        if not label_levels:
            self._set_status(f"MERSCOPE Cellpose value overlay failed: labels[{label_key}] has no 2D levels.")
            return
        label_data = label_levels if len(label_levels) > 1 else label_levels[0]

        ds = self.active_dataset
        channel = str(payload["channel"])
        statistic = str(payload["statistic"])
        layer_name = make_layer_name(ds, "cell_values", CELLPOSE_SHAPE_KEY, f"{channel} {statistic}")
        self._remove_layers_by_prefix(layer_name_prefix(ds, "cell_values"))
        self.viewer.add_labels(
            label_data,
            name=layer_name,
            affine=napari_affine,
            cache=NAPARI_DASK_CACHE_ENABLED,
            colormap=payload["colormap"],
            multiscale=len(label_levels) > 1,
            opacity=min(1.0, max(0.0, float(self.args.shape_opacity))),
            blending="translucent",
            visible=True,
        )

        mapping = payload["mapping"]
        self._set_status(
            f"{ds} Cellpose values: {channel} {statistic} ({payload['colormap_name']}), "
            f"{mapping.lower_percentile:g}-{mapping.upper_percentile:g}%="
            f"{mapping.clip_low:.4g}..{mapping.clip_high:.4g}; "
            f"finite={mapping.finite_count:,}/{mapping.label_count:,}, "
            f"colors={mapping.unique_color_count:,}."
        )
        log.info(
            "[%s] Added Cellpose value overlay label=%s channel=%s statistic=%s colormap=%s "
            "clip=(%s, %s) finite=%s/%s colors=%s levels=%s",
            ds,
            label_key,
            channel,
            statistic,
            payload["colormap_name"],
            mapping.clip_low,
            mapping.clip_high,
            mapping.finite_count,
            mapping.label_count,
            mapping.unique_color_count,
            len(label_levels),
        )

    def _handle_cellpose_value_error(self, generation: int, exc):
        if generation != self._cellpose_value_generation:
            return
        self._cellpose_value_worker = None
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"MERSCOPE Cellpose value overlay failed: {message}")
        log.error("[MERSCOPE] Cellpose value overlay failed: %s", message)

    def load_cellpose_value_overlay(
        self,
        dataset_name: str,
        channel: str,
        statistic: str,
        colormap_name: str,
    ):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        if self.active_dataset != "MERSCOPE":
            self._set_status("Cellpose value overlay is only enabled for MERSCOPE datasets.")
            return

        cfg = self.datasets[self.active_dataset]
        if not cellpose_quantification_table_available(cfg.zarr_path):
            self._set_status(f"Missing {CELLPOSE_QUANTIFICATION_TABLE_KEY}; cannot load Cellpose value overlay.")
            return

        self._cellpose_value_generation += 1
        generation = self._cellpose_value_generation
        self._set_status(f"MERSCOPE building Cellpose value overlay: {channel} {statistic}...")

        def compute():
            # Heavy: prepares (rasterizes if needed) the Cellpose labels, builds
            # the color mapping, and materializes a max-pooled label pyramid so
            # the overlay pans/zooms without re-reading the full-resolution
            # labels. All off the GUI thread; layer creation happens on apply.
            label_key = self._ensure_cellpose_label_key()
            label_levels, napari_affine = self._build_cellpose_label_display(label_key)
            payload = self._compute_cellpose_value_payload(cfg.zarr_path, channel, statistic, colormap_name)
            return {
                **payload,
                "label_key": label_key,
                "label_levels": label_levels,
                "napari_affine": napari_affine,
            }

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_cellpose_value_error(generation, exc)
                return
            self._apply_cellpose_value_payload(generation, payload)
            return

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, gen=generation: self._apply_cellpose_value_payload(gen, payload)
        )
        worker.errored.connect(lambda exc, gen=generation: self._handle_cellpose_value_error(gen, exc))
        self._cellpose_value_worker = worker
        worker.start()

    def remove_cellpose_value_overlay(self, dataset_name: str):
        ds = str(dataset_name).upper()
        self._cellpose_value_generation += 1
        self._cellpose_value_worker = None
        removed = 0
        prefix = layer_name_prefix(ds, "cell_values")
        names = [str(layer.name) for layer in self.viewer.layers]
        for layer_name in matching_layer_names(names, prefix):
            before = len(self.viewer.layers)
            self._remove_layer_by_name(layer_name)
            if len(self.viewer.layers) < before:
                removed += 1
        self._set_status(f"{ds} removed {removed} Cellpose value overlay layer(s).")

    def unload_selected_shapes(self, dataset_name: str, shape_keys: list[str]):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        ds = str(dataset_name).upper()
        removed = 0
        for shape_key in shape_keys:
            layer_names = [
                make_layer_name(ds, self._segmentation_layer_type(), shape_key),
                make_layer_name(ds, "labels", shape_key),
                make_layer_name(ds, "labels", self._label_key_for_shape_key(shape_key)),
            ]
            for name in layer_names:
                before = len(self.viewer.layers)
                self._remove_layer_by_name(name)
                if len(self.viewer.layers) < before:
                    removed += 1
        self._publish_loaded_segmentation_keys()
        self._set_status(f"{ds} removed {removed} {self._segmentation_layer_type()} layer(s).")

    # ------------------------------------------------------------------
    # Per-gene transcript renderer
    # ------------------------------------------------------------------
    def _gene_layer_name(self, ds: str, symbol: str) -> str:
        return f"Genes | {gene_marker_symbol_label(symbol)}"

    def _resolve_gene_points_columns(self):
        points_key, points_obj, x_col, y_col, assignment_col = self._resolve_points_columns()
        gene_col = resolve_gene_column(points_obj)
        background_col = first_existing_col(points_obj, ["background"])
        return points_key, points_obj, x_col, y_col, assignment_col, gene_col, background_col

    def open_gene_inspector(self, dataset_name: str):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        ds = str(dataset_name).upper()
        try:
            (points_key, points_obj, x_col, y_col, assignment_col, gene_col,
             background_col) = self._resolve_gene_points_columns()
        except Exception as exc:
            self._set_status(f"{ds} inspect genes failed: {exc}")
            return
        if gene_col is None:
            self._set_status(f"{ds} has no gene/feature-name column in points[{points_key}].")
            return

        # Tear down any prior gene inspector layers before rebuilding.
        self._teardown_gene_inspector(ds)
        gc.collect()

        self._gene_build_generation += 1
        generation = self._gene_build_generation
        max_points = int(getattr(self.args, "gene_max_render_points", DEFAULT_GENE_MAX_RENDER_POINTS))
        random_state = int(getattr(self.args, "random_state", 42))
        cfg = self.datasets.get(ds)
        zarr_path = getattr(cfg, "zarr_path", None)
        self._begin_progress("transcripts", f"{ds}: building per-gene transcripts for {points_key}...")

        def compute():
            t0 = time.time()
            # Group genes by the cell type they mark when the store carries a
            # marker reference; otherwise fall back to the deterministic rainbow.
            reference = load_cell_type_marker_reference(zarr_path) if zarr_path is not None else None
            store = build_gene_point_groups(
                points_obj,
                x_col=x_col,
                y_col=y_col,
                gene_col=gene_col,
                assignment_col=assignment_col,
                background_col=background_col,
                reference=reference,
                max_points=max_points if max_points > 0 else None,
                random_state=random_state,
            )
            return {
                "points_key": points_key,
                "store": store,
                "reference": reference,
                "build_seconds": time.time() - t0,
            }

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_gene_inspector_build_error(generation, ds, exc)
                return
            self._apply_gene_inspector_build(generation, ds, payload)
            return

        worker = thread_worker(compute)()
        worker.returned.connect(
            lambda payload, gen=generation, d=ds: self._apply_gene_inspector_build(gen, d, payload)
        )
        worker.errored.connect(
            lambda exc, gen=generation, d=ds: self._handle_gene_inspector_build_error(gen, d, exc)
        )
        self._gene_build_worker = worker
        worker.start()

    def _apply_gene_inspector_build(self, generation: int, ds: str, payload: dict):
        if generation != self._gene_build_generation or self.active_dataset != ds:
            return
        self._gene_build_worker = None
        self._end_progress("transcripts")
        store = payload["store"]
        points_key = str(payload["points_key"])
        if store.total_points == 0 or not store.genes:
            self._set_status(f"{ds} inspect genes: no transcripts found.")
            return

        reference = payload.get("reference")
        # Two precomputed schemes over the same symbols (so switching only
        # recolours). Without a reference both fall back to the rainbow and only
        # A–Z ordering is offered.
        coarse_scheme = build_cell_type_gene_visuals(store.genes, reference, kind="coarse")
        fine_scheme = build_cell_type_gene_visuals(store.genes, reference, kind="fine")
        has_reference = bool(reference)
        gene_visuals = store.gene_visuals or coarse_scheme.visuals
        ordering = "coarse" if has_reference else "alphabetical"

        hide_assigned = bool(getattr(self.args, "gene_hide_assigned", False))
        hide_background = bool(getattr(self.args, "gene_hide_background", False))
        show_controls = bool(getattr(self.args, "gene_show_controls", False))
        spot_size = float(getattr(self.args, "gene_spot_size", DEFAULT_GENE_SPOT_SIZE))
        enabled = {g for g in store.genes if show_controls or g not in store.control_genes}

        timer = QTimer()
        timer.setSingleShot(True)
        state = GeneInspectorState(
            dataset=ds,
            points_key=points_key,
            store=store,
            gene_visuals=gene_visuals,
            layer_names=[],
            enabled_genes=enabled,
            spot_size=spot_size,
            hide_assigned=hide_assigned,
            hide_background=hide_background,
            show_controls=show_controls,
            rebuild_timer=timer,
            pending_groups=set(),
            highlighted_genes=[],
            group_display_ranges=[[] for _ in store.group_symbols],
            reference=reference,
            coarse_scheme=coarse_scheme,
            fine_scheme=fine_scheme,
            ordering=ordering,
            color_kind="coarse",
            colour_by_assignment=False,
        )
        timer.timeout.connect(lambda d=ds: self._flush_gene_group_rebuild(d))
        self._gene_inspector_states[ds] = state
        # Transcripts (re)loaded: drop any stale per-cell index and clear the
        # cell selection so a fresh click rebuilds against the new points.
        self._invalidate_cell_inspection(ds)

        # Create each layer already populated with its enabled genes' points +
        # per-point colours + symbol. Building populated (vs. empty then growing)
        # avoids napari's data-resize path that both drops the per-point symbol
        # back to the default disc and can reset colours to white.
        for gi, symbol in enumerate(store.group_symbols):
            coords, colors, sizes, ranges = self._gene_group_arrays(state, gi)
            state.group_display_ranges[gi] = ranges
            layer_name = self._gene_layer_name(ds, symbol)
            self._create_gene_points_layer(layer_name, symbol, spot_size, coords, colors, sizes)
            state.layer_names.append(layer_name)
        layer_names = state.layer_names
        # Pin the (long) gene-layer block to the bottom of the layer list so it
        # stays below images and masks regardless of which background load finishes
        # first. Layers added later by napari sit on top, so without this the
        # slow-to-build transcripts would land at the top.
        self._move_gene_layers_to_bottom(state)

        if self._gene_inspector_widget is not None:
            self._populate_gene_inspector(state)
            try:
                self._gene_inspector_widget.show()
                self._gene_inspector_widget.raise_()
            except Exception:
                pass

        source_total = int(store.source_total_points if store.source_total_points is not None else store.total_points)
        self._set_status("Click on any transcript to highlight that gene")
        log.info(
            "[%s] Gene inspector: genes=%s source_points=%s rendered_points=%s layers=%s sampled=%s build=%.1fs",
            ds, len(store.genes), source_total, store.total_points, len(layer_names), store.sampled,
            float(payload.get("build_seconds", 0.0)),
        )

    def _handle_gene_inspector_build_error(self, generation: int, ds: str, exc):
        if generation != self._gene_build_generation:
            return
        self._gene_build_worker = None
        self._end_progress("transcripts")
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{ds} inspect genes failed: {message}")
        log.error("[%s] Gene inspector build failed: %s", ds, message)

    def _gene_group_arrays(self, state: GeneInspectorState, gi: int):
        """Concatenate enabled genes for one marker group and track click ranges."""
        store = state.store
        gcoords = store.group_coords[gi]
        gcolors = store.group_colors[gi]
        coord_slices: list[np.ndarray] = []
        color_slices: list[np.ndarray] = []
        display_ranges: list[tuple[int, int, str]] = []
        cursor = 0
        for gene in store.genes:
            if gene not in state.enabled_genes:
                continue
            entry = store.gene_offsets.get(gene)
            if entry is None or entry[0] != gi:
                continue
            _g, fg_start, fg_end, bg_end = entry
            start = cursor
            if not state.hide_assigned and fg_end > fg_start:
                coord_slices.append(gcoords[fg_start:fg_end])
                if state.colour_by_assignment:
                    color_slices.append(
                        np.tile(GENE_ASSIGNED_RGBA, (int(fg_end - fg_start), 1))
                    )
                else:
                    color_slices.append(gcolors[fg_start:fg_end])
                cursor += int(fg_end - fg_start)
            if not state.hide_background and bg_end > fg_end:
                coord_slices.append(gcoords[fg_end:bg_end])
                if state.colour_by_assignment:
                    color_slices.append(
                        np.tile(GENE_UNASSIGNED_RGBA, (int(bg_end - fg_end), 1))
                    )
                else:
                    color_slices.append(gcolors[fg_end:bg_end])
                cursor += int(bg_end - fg_end)
            if cursor > start:
                display_ranges.append((start, cursor, str(gene)))
        if coord_slices:
            coords = np.concatenate(coord_slices, axis=0)
            colors = np.concatenate(color_slices, axis=0)
            sizes = self._gene_group_size_array(state, len(coords), display_ranges)
            return coords, colors, sizes, display_ranges
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            None,
            display_ranges,
        )

    def _gene_group_size_array(self, state: GeneInspectorState, n_points: int, ranges: list[tuple[int, int, str]]):
        highlighted_genes = set(state.highlighted_genes or [])
        if not highlighted_genes or n_points <= 0:
            return None
        if not any(gene in highlighted_genes for _start, _end, gene in ranges):
            return None
        sizes = np.full(int(n_points), float(state.spot_size), dtype=np.float32)
        for start, end, gene in ranges:
            if gene in highlighted_genes:
                sizes[int(start):int(end)] = float(GENE_HIGHLIGHT_SPOT_SIZE)
        return sizes

    def _create_gene_points_layer(self, name, symbol, spot_size, coords, colors, sizes=None):
        """Add a Points layer pre-populated with data + per-point colour + symbol."""
        size = sizes if sizes is not None else float(spot_size)
        kwargs = dict(name=name, size=size, opacity=1.0, visible=True, symbol=str(symbol))
        try:
            layer = self.viewer.add_points(coords, face_color=colors, border_color=colors, **kwargs)
        except TypeError:
            layer = self.viewer.add_points(coords, face_color=colors, edge_color=colors, **kwargs)
        # Antialiasing lets sub-pixel spots contribute partial opacity so density
        # structure shows when zoomed out; the min canvas-size floor is dropped so
        # spots actually shrink with zoom instead of staying a fixed 2px blanket.
        antialiasing = float(getattr(self.args, "gene_antialiasing", 1.0))
        min_px = float(getattr(self.args, "gene_min_canvas_px", 0.0))
        for attr, value in (("antialiasing", antialiasing), ("border_width", 0.0), ("symbol", str(symbol))):
            try:
                setattr(layer, attr, value)
            except Exception:
                pass
        try:
            layer.canvas_size_limits = (min_px, 10000.0)
        except Exception:
            pass
        return layer

    def _move_gene_layers_to_bottom(self, state: GeneInspectorState):
        """Move this dataset's gene Points layers to the bottom of the layer list.

        Preserves their relative order (first symbol group lowest). Silently no-ops
        when the layer list doesn't support reordering (e.g. the test fake viewer).
        """
        layers = self.viewer.layers
        names = list(state.layer_names)
        try:
            indices = [i for i, layer in enumerate(layers) if str(getattr(layer, "name", "")) in set(names)]
            if not indices:
                return
            if hasattr(layers, "move_multiple"):
                layers.move_multiple(sorted(indices), 0)
            elif hasattr(layers, "move"):
                # Move each gene layer to the front, last-to-first, so the final
                # bottom block keeps the original symbol-group order.
                for name in reversed(names):
                    src = next((i for i, layer in enumerate(layers) if str(getattr(layer, "name", "")) == name), None)
                    if src is not None:
                        layers.move(src, 0)
        except Exception as exc:  # pragma: no cover - defensive; ordering is cosmetic
            log.debug("Could not reorder gene layers to bottom: %s", exc)

    def _set_gene_layer_data(self, layer, coords, colors, symbol, spot_size, sizes=None):
        layer.data = coords
        try:
            layer.size = sizes if sizes is not None else float(spot_size)
        except Exception:
            pass
        if len(coords):
            try:
                layer.face_color = colors
            except Exception:
                pass
            try:
                layer.border_color = colors
            except (AttributeError, TypeError):
                try:
                    layer.edge_color = colors
                except Exception:
                    pass
            # Re-apply the symbol: changing ``data`` resets the per-point symbol
            # array back to the default disc, so it must be re-broadcast here.
            try:
                layer.symbol = str(symbol)
            except Exception:
                pass

    def _rebuild_gene_group_layers(self, state: GeneInspectorState, group_indices=None):
        store = state.store
        if group_indices is None:
            group_indices = range(len(store.group_symbols))
        for gi in group_indices:
            if gi < 0 or gi >= len(state.layer_names):
                continue
            layer = self._get_layer_by_name(state.layer_names[gi])
            if layer is None:
                continue
            coords, colors, sizes, ranges = self._gene_group_arrays(state, gi)
            if state.group_display_ranges is not None and gi < len(state.group_display_ranges):
                state.group_display_ranges[gi] = ranges
            self._set_gene_layer_data(layer, coords, colors, store.group_symbols[gi], state.spot_size, sizes)

    def _schedule_gene_group_rebuild(self, state: GeneInspectorState, group_index: int):
        if state.pending_groups is None:
            state.pending_groups = set()
        state.pending_groups.add(int(group_index))
        if state.rebuild_timer is not None:
            state.rebuild_timer.start(GENE_REBUILD_DEBOUNCE_MS)
        else:
            self._rebuild_gene_group_layers(state, group_indices={int(group_index)})

    def _flush_gene_group_rebuild(self, ds: str):
        state = self._gene_inspector_states.get(str(ds).upper())
        if state is None:
            return
        groups = state.pending_groups or set()
        state.pending_groups = set()
        self._rebuild_gene_group_layers(state, group_indices=sorted(groups) if groups else None)

    def _on_viewer_mouse_press(self, _viewer, event):
        event_type = str(getattr(event, "type", "mouse_press"))
        if "press" not in event_type:
            return
        if self.active_dataset is None:
            return
        ds = self.active_dataset
        state = self._gene_inspector_states.get(str(ds).upper())

        # 1) A click landing on a transcript highlights that gene (existing).
        if state is not None:
            gene = self._pick_gene_from_event(state, event)
            if gene is not None:
                self._add_highlighted_gene(state, gene)
                return

        # 2) Otherwise a click inside a segmentation mask adds it to the set of
        #    highlighted cells (existing highlights are kept) -- but only while the
        #    gating segmentation layer (ProSeg) is visible.
        if self._cell_inspector_pickable():
            picked = self._pick_cell_from_event(event)
            if picked is not None:
                self._add_cell_selection(ds, picked[0], picked[1])
                return

        # 3) A click in empty space only clears transcript-gene highlights; the
        #    selected cells stay highlighted so the user can pan/zoom freely.
        #    Cells are deselected only by closing the bottom window.
        if state is not None and state.highlighted_genes:
            self._clear_highlighted_genes(state)

    # -- Cell inspector: click a mask to summarise its cell -----------------
    def _event_world_xy(self, event) -> tuple[float, float] | None:
        """Return the click's global ``(x_um, y_um)``, or None."""
        position = getattr(event, "position", None)
        if position is None:
            return None
        try:
            coords = np.asarray(position, dtype=float).ravel()
        except Exception:
            return None
        if coords.size < 2:
            return None
        y_um, x_um = float(coords[-2]), float(coords[-1])
        return x_um, y_um

    def _cell_inspector_shape_key(self) -> str | None:
        """Resolve the segmentation GeoDataFrame to hit-test, preferring ProSeg."""
        if self._active_sdata is None:
            return None
        try:
            shape_keys = [str(k) for k in self._active_sdata.shapes.keys()]
        except Exception:
            return None
        if not shape_keys:
            return None
        for preferred in CELL_INSPECTOR_SHAPE_PREFERENCE:
            for key in shape_keys:
                if preferred in key.lower():
                    return key
        return shape_keys[0]

    def _cell_inspector_layer(self):
        """Return the rendered napari layer whose visibility gates cell picking.

        This is the layer for the same segmentation the click hit-test uses
        (ProSeg first, per :meth:`_cell_inspector_shape_key`), or ``None`` if it
        isn't currently loaded.
        """
        shape_key = self._cell_inspector_shape_key()
        if shape_key is None or self.active_dataset is None:
            return None
        ds = str(self.active_dataset).upper()
        try:
            label_key = self._label_key_for_shape_key(shape_key)
        except Exception:
            label_key = f"{shape_key}_labels"
        try:
            seg_type = self._segmentation_layer_type()
        except Exception:
            seg_type = "labels"
        for name in (
            make_layer_name(ds, "labels", label_key),
            make_layer_name(ds, "labels", shape_key),
            make_layer_name(ds, seg_type, shape_key),
        ):
            layer = self._get_layer_by_name(name)
            if layer is not None:
                return layer
        return None

    def _cell_inspector_pickable(self) -> bool:
        """Cells are clickable unless the gating segmentation layer is hidden.

        Only blocks when that layer is present *and* not visible; if no such layer
        is loaded there is nothing to hide, so picking behaves as before.
        """
        layer = self._cell_inspector_layer()
        return layer is None or bool(getattr(layer, "visible", True))

    def _on_segmentation_visibility_changed(self, _event=None):
        """Clear highlighted cells when the gating segmentation layer is hidden."""
        layer = self._cell_inspector_layer()
        if layer is not None and not bool(getattr(layer, "visible", True)):
            if any(self._selected_cells.values()):
                self._clear_all_cell_selections()

    def _pick_cell_from_event(self, event):
        """Return ``(cell_id, geometry)`` for the mask under the click, or None."""
        world = self._event_world_xy(event)
        if world is None:
            return None
        shape_key = self._cell_inspector_shape_key()
        if shape_key is None:
            return None
        try:
            gdf = self._active_sdata.shapes[shape_key]
            return pick_cell_at_point(gdf, world[0], world[1])
        except Exception as exc:
            log.debug("Cell pick failed: %s", exc)
            return None

    def _empty_cell_transcript_index(self) -> CellTranscriptIndex:
        return CellTranscriptIndex(
            coords_yx=np.empty((0, 2), dtype=np.float32),
            genes=np.empty((0,), dtype=object),
            slices={},
        )

    def _get_cell_transcript_index(self, ds: str) -> CellTranscriptIndex:
        """Lazily build (and cache) the assigned-transcript-per-cell index."""
        key = str(ds).upper()
        cached = self._cell_transcript_index.get(key)
        if cached is not None:
            return cached
        try:
            _pk, points_obj, x_col, y_col, assignment_col = self._resolve_points_columns()
            gene_col = resolve_gene_column(points_obj)
        except Exception as exc:
            log.debug("Cell transcript index unavailable: %s", exc)
            index = self._empty_cell_transcript_index()
            self._cell_transcript_index[key] = index
            return index
        if gene_col is None or assignment_col is None:
            index = self._empty_cell_transcript_index()
        else:
            self._set_status(f"{ds}: indexing transcripts by cell...")
            index = build_cell_transcript_index(points_obj, x_col, y_col, gene_col, assignment_col)
        self._cell_transcript_index[key] = index
        return index

    def _invalidate_cell_inspection(self, ds: str | None = None):
        """Drop cached cell index(es) and clear highlighted cells."""
        if ds is None:
            self._cell_transcript_index.clear()
            self._clear_all_cell_selections()
            return
        self._cell_transcript_index.pop(str(ds).upper(), None)
        if self._selected_cells.pop(str(ds).upper(), None) is not None:
            self._rebuild_cell_highlights(ds)

    def _cell_selection_display(self, ds: str, cell_id, geometry, color: str) -> dict:
        """Compute cached draw + panel data for one highlighted cell."""
        key = str(ds).upper()
        state = self._gene_inspector_states.get(key)
        index = self._get_cell_transcript_index(ds)
        coords_yx, genes = index.transcripts_for(cell_id)

        if state is not None:
            gene_visuals = state.gene_visuals
        else:
            gene_visuals = assign_gene_visuals(sorted({str(g) for g in genes.tolist()}))

        gene_rows = []
        for gene, count in ranked_gene_counts(genes):
            visual = gene_visuals.get(gene) if gene_visuals else None
            if visual is not None:
                rgba = tuple(visual.rgba)
                symbol = str(visual.symbol)
            else:
                rgba, symbol = (0.6, 0.6, 0.6, 1.0), "disc"
            gene_rows.append(
                {
                    "gene": gene,
                    "count": int(count),
                    "rgba": rgba,
                    "glyph": GENE_STATUS_SYMBOL_GLYPHS.get(symbol, "●"),
                }
            )

        try:
            centroid = geometry.centroid
            centroid_yx = (float(centroid.y), float(centroid.x))
        except Exception:
            centroid_yx = None
        try:
            area_um2 = float(geometry.area)
        except Exception:
            area_um2 = None

        return {
            "cell_id": cell_id,
            "color": str(color),
            "paths": self._polygon_to_napari_paths(geometry),
            "coords_yx": np.asarray(coords_yx, dtype=float),
            "centroid_yx": centroid_yx,
            "gene_rows": gene_rows,
            "total": int(len(genes)),
            "area": area_um2,
            "intensities": self._compute_cell_channel_intensities(geometry),
        }

    def _add_cell_selection(self, ds: str, cell_id, geometry):
        """Add a cell to the highlighted set (existing highlights are kept)."""
        if geometry is None:
            return
        key = str(ds).upper()
        entries = self._selected_cells.setdefault(key, [])
        norm = normalize_cell_key(cell_id)
        if any(normalize_cell_key(entry["cell_id"]) == norm for entry in entries):
            return  # already highlighted; ignore repeat clicks on the same cell
        color = CELL_HIGHLIGHT_COLORS[len(entries) % len(CELL_HIGHLIGHT_COLORS)]
        entries.append(self._cell_selection_display(ds, cell_id, geometry, color))
        self._rebuild_cell_highlights(ds)
        self._set_status(f"{ds}: {len(entries)} cell(s) highlighted (added {cell_id}).")

    def _rebuild_cell_highlights(self, ds: str):
        """Redraw boundary + link layers and the bottom bar for all highlights."""
        entries = self._selected_cells.get(str(ds).upper(), [])
        self._update_cell_boundary_layer(entries)
        self._update_cell_links_layer(entries)
        if not entries:
            if self._cell_info_overlay is not None:
                self._cell_info_overlay.set_cells([])
            self._hide_cell_dock()
            return
        overlay = self._ensure_cell_info_overlay()
        if overlay is not None:
            overlay.set_cells(entries)
            self._show_cell_dock()

    def _clear_all_cell_selections(self):
        """Remove every highlight (boundary, links, panels) and hide the bar."""
        self._selected_cells.clear()
        self._remove_layer_by_name(CELL_INSPECTOR_BOUNDARY_LAYER)
        self._remove_layer_by_name(CELL_INSPECTOR_LINKS_LAYER)
        if self._cell_info_overlay is not None:
            self._cell_info_overlay.set_cells([])
        self._hide_cell_dock()

    def _polygon_to_napari_paths(self, geometry) -> list[np.ndarray]:
        """Return ``(N, 2)`` napari ``(y, x)`` ring arrays for a (multi)polygon."""
        parts = geometry.geoms if getattr(geometry, "geom_type", "") == "MultiPolygon" else (geometry,)
        paths: list[np.ndarray] = []
        for part in parts:
            if part.is_empty:
                continue
            coords = np.asarray(part.exterior.coords, dtype=float)
            if coords.shape[0] < 3:
                continue
            paths.append(np.column_stack([coords[:, 1], coords[:, 0]]))  # (y, x)
        return paths

    def _mask_highlight_edge_width(self) -> float:
        """Outline width (in microns) marginally thicker than the loaded masks.

        The rasterised mask outlines are ``label_contour_width`` label-pixels
        wide, so their micron thickness is ``label_contour_width * um_per_pixel``.
        Drawing the highlight in micron (data) units at a small multiple of that
        makes it read as slightly thicker than neighbouring outlines at every
        zoom level without needing to track the camera.
        """
        prefix = "Segmentation | "
        outline_px = max(1, int(getattr(self.args, "label_contour_width", 1)))
        for layer in list(self.viewer.layers):
            if not str(getattr(layer, "name", "")).startswith(prefix):
                continue
            matrix = self._image_layer_affine_matrix(layer)
            if matrix is None:
                continue
            linear = matrix[:2, :2]
            sx = float(np.hypot(linear[0, 0], linear[1, 0]))
            sy = float(np.hypot(linear[0, 1], linear[1, 1]))
            um_per_px = (sx + sy) / 2.0
            if um_per_px > 0:
                return um_per_px * outline_px * CELL_BOUNDARY_WIDTH_FACTOR
        return CELL_BOUNDARY_FALLBACK_WIDTH_UM

    def _update_cell_boundary_layer(self, entries: list[dict]):
        self._remove_layer_by_name(CELL_INSPECTOR_BOUNDARY_LAYER)
        all_paths: list[np.ndarray] = []
        edge_colors: list[np.ndarray] = []
        for entry in entries:
            rgba = rgba_array(entry.get("color", "#ffffff"), alpha=1.0)
            for path in entry.get("paths", []):
                all_paths.append(path)
                edge_colors.append(rgba)
        if not all_paths:
            return
        try:
            self.viewer.add_shapes(
                all_paths,
                shape_type="polygon",
                name=CELL_INSPECTOR_BOUNDARY_LAYER,
                edge_color=np.asarray(edge_colors, dtype=float),
                face_color="transparent",
                edge_width=self._mask_highlight_edge_width(),
                opacity=1.0,
            )
        except Exception as exc:
            log.debug("Could not draw cell boundary highlight: %s", exc)

    def _update_cell_links_layer(self, entries: list[dict]):
        self._remove_layer_by_name(CELL_INSPECTOR_LINKS_LAYER)
        lines: list[np.ndarray] = []
        for entry in entries:
            centroid_yx = entry.get("centroid_yx")
            coords = entry.get("coords_yx")
            if centroid_yx is None or coords is None or len(coords) == 0:
                continue
            cy, cx = float(centroid_yx[0]), float(centroid_yx[1])
            for y, x in np.asarray(coords, dtype=float):
                lines.append(np.array([[float(y), float(x)], [cy, cx]], dtype=float))
        if not lines:
            return
        try:
            layer = self.viewer.add_shapes(
                lines,
                shape_type="line",
                name=CELL_INSPECTOR_LINKS_LAYER,
                edge_color="#ffffff",
                edge_width=CELL_LINK_WIDTH,
                opacity=float(CELL_LINK_COLOR[3]),
            )
        except Exception as exc:
            log.debug("Could not draw cell transcript links: %s", exc)
            return
        self._send_links_below_transcripts(layer)

    def _send_links_below_transcripts(self, links_layer):
        """Move the link layer beneath the transcript points so it never occludes them."""
        if self.active_dataset is None:
            return
        state = self._gene_inspector_states.get(str(self.active_dataset).upper())
        if state is None or not state.layer_names:
            return
        try:
            names = {str(n) for n in state.layer_names}
            layers = list(self.viewer.layers)
            gene_indices = [i for i, layer in enumerate(layers) if str(layer.name) in names]
            if not gene_indices:
                return
            target = min(gene_indices)
            src = self.viewer.layers.index(links_layer)
            if src > target:
                self.viewer.layers.move(src, target)
        except Exception as exc:
            log.debug("Could not reorder cell link layer: %s", exc)

    def _send_cell_types_below_transcripts(self, cell_type_layer):
        """Move the cell-type fill beneath the transcript points so they stay visible."""
        if self.active_dataset is None:
            return
        state = self._gene_inspector_states.get(str(self.active_dataset).upper())
        if state is None or not state.layer_names:
            return  # no transcripts loaded -> leave the fill where it is
        try:
            names = {str(n) for n in state.layer_names}
            layers = list(self.viewer.layers)
            gene_indices = [i for i, layer in enumerate(layers) if str(layer.name) in names]
            if not gene_indices:
                return
            target = min(gene_indices)
            src = self.viewer.layers.index(cell_type_layer)
            if src > target:
                self.viewer.layers.move(src, target)
        except Exception as exc:
            log.debug("Could not reorder cell-type overlay layer: %s", exc)

    def _image_layer_affine_matrix(self, layer) -> np.ndarray | None:
        """Return the 3x3 pixel->micron matrix for a 2D image layer."""
        affine = getattr(layer, "affine", None)
        if affine is None:
            return None
        matrix = getattr(affine, "affine_matrix", None)
        if matrix is None:
            try:
                matrix = np.asarray(affine, dtype=float)
            except Exception:
                return None
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return None
        return matrix[-3:, -3:]

    def _compute_cell_channel_intensities(self, geometry) -> list[tuple[str, float | None]]:
        """Mean intensity of every loaded image channel within the cell polygon."""
        rows: list[tuple[str, float | None]] = []
        for layer in list(self.viewer.layers):
            name = str(getattr(layer, "name", ""))
            if not name.startswith("Image | "):
                continue
            matrix = self._image_layer_affine_matrix(layer)
            if matrix is None:
                continue
            data = getattr(layer, "data", None)
            if isinstance(data, (list, tuple)):
                data = data[0] if data else None  # finest multiscale level
            if data is None:
                continue
            try:
                value = mean_intensity_in_polygon(data, matrix, geometry)
            except Exception as exc:
                log.debug("Intensity computation failed for %s: %s", name, exc)
                value = None
            channel = name.split(" | ", 1)[1] if " | " in name else name
            rows.append((channel, value))
        return rows

    def _ensure_cell_info_overlay(self) -> CellInfoOverlay | None:
        if self._cell_info_overlay is not None:
            return self._cell_info_overlay
        try:
            widget = CellInfoOverlay()
        except Exception as exc:
            log.debug("Could not create cell info overlay: %s", exc)
            return None
        self._cell_info_overlay = widget
        # Dock as a wide bar along the bottom (best-effort: a fake viewer used in
        # tests has no docking API, in which case the bare widget still works).
        window = getattr(self.viewer, "window", None)
        if window is not None and hasattr(window, "add_dock_widget"):
            try:
                dock = window.add_dock_widget(widget, area="bottom", name="Cell inspector")
                self._cell_info_dock = dock
                visibility_changed = getattr(dock, "visibilityChanged", None)
                if visibility_changed is not None:
                    try:
                        visibility_changed.connect(self._on_cell_dock_visibility_changed)
                    except Exception:
                        pass
                self._hide_cell_dock()
            except Exception as exc:
                log.debug("Could not dock cell info overlay: %s", exc)
                self._cell_info_dock = None
        return self._cell_info_overlay

    def _show_cell_dock(self):
        dock = self._cell_info_dock
        if dock is not None:
            self._suppress_dock_visibility = True
            try:
                dock.show()
                if hasattr(dock, "raise_"):
                    dock.raise_()
            except Exception:
                pass
            finally:
                self._suppress_dock_visibility = False
        elif self._cell_info_overlay is not None:
            self._cell_info_overlay.show()

    def _hide_cell_dock(self):
        dock = self._cell_info_dock
        if dock is not None:
            self._suppress_dock_visibility = True
            try:
                dock.hide()
            except Exception:
                pass
            finally:
                self._suppress_dock_visibility = False
        elif self._cell_info_overlay is not None:
            self._cell_info_overlay.hide()

    def _on_cell_dock_visibility_changed(self, visible: bool):
        """Closing the bottom window (its X) deselects every highlighted cell.

        napari's close button tears the dock down such that re-showing the same
        instance does not bring it back, so we also drop our references; the next
        cell selection then rebuilds a fresh dock (the window "comes back").
        """
        if visible or self._suppress_dock_visibility:
            return
        if not self._selected_cells and self._cell_info_dock is None:
            return
        self._clear_all_cell_selections()
        self._teardown_cell_info_dock()

    def _teardown_cell_info_dock(self):
        """Drop the current dock/overlay so a later selection recreates them."""
        dock = self._cell_info_dock
        self._cell_info_dock = None
        self._cell_info_overlay = None
        if dock is None:
            return
        window = getattr(self.viewer, "window", None)
        remove = getattr(window, "remove_dock_widget", None) if window is not None else None
        if remove is None:
            return

        def _remove(target=dock, remover=remove):
            try:
                remover(target)
            except Exception:
                pass

        # Defer removal so we do not mutate the dock area from inside the dock's
        # own close/visibility event.
        try:
            QTimer.singleShot(0, _remove)
        except Exception:
            _remove()

    def _pick_gene_from_event(self, state: GeneInspectorState, event) -> str | None:
        if state.group_display_ranges is None:
            return None
        # Later layers are visually on top, so query in reverse display order.
        for gi in reversed(range(len(state.layer_names))):
            layer = self._get_layer_by_name(state.layer_names[gi])
            if layer is None or not bool(getattr(layer, "visible", True)):
                continue
            idx = self._picked_point_index(layer, event)
            if idx is None:
                continue
            gene = self._gene_for_display_index(state, gi, idx)
            if gene is not None:
                return gene
        return None

    def _picked_point_index(self, layer, event) -> int | None:
        get_value = getattr(layer, "get_value", None)
        if get_value is None:
            return None
        position = getattr(event, "position", None)
        if position is None:
            return None
        kwargs = {
            "view_direction": getattr(event, "view_direction", None),
            "dims_displayed": getattr(event, "dims_displayed", None),
            "world": True,
        }
        for call_kwargs in (kwargs, {k: v for k, v in kwargs.items() if k != "world"}, {}):
            try:
                value = get_value(position, **call_kwargs)
                return self._coerce_picked_point_index(value)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _coerce_picked_point_index(self, value) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value) if int(value) >= 0 else None
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return self._coerce_picked_point_index(value.item())
            values = value.tolist()
        elif isinstance(value, (tuple, list)):
            values = list(value)
        else:
            return None
        for item in values:
            idx = self._coerce_picked_point_index(item)
            if idx is not None:
                return idx
        return None

    def _gene_for_display_index(self, state: GeneInspectorState, group_index: int, point_index: int) -> str | None:
        if state.group_display_ranges is None:
            return None
        if group_index < 0 or group_index >= len(state.group_display_ranges):
            return None
        idx = int(point_index)
        for start, end, gene in state.group_display_ranges[group_index]:
            if int(start) <= idx < int(end):
                return str(gene)
        return None

    def _add_highlighted_gene(self, state: GeneInspectorState, gene: str):
        new_gene = str(gene)
        if state.highlighted_genes is None:
            state.highlighted_genes = []
        if new_gene in state.highlighted_genes:
            return
        state.highlighted_genes.append(new_gene)
        entry = state.store.gene_offsets.get(new_gene)
        if entry is not None:
            self._apply_gene_group_sizes(state, {int(entry[0])})
        self._set_gene_highlight_status(state)

    def _clear_highlighted_genes(self, state: GeneInspectorState):
        if not state.highlighted_genes:
            return
        groups: set[int] = set()
        for gene in state.highlighted_genes:
            entry = state.store.gene_offsets.get(str(gene))
            if entry is not None:
                groups.add(int(entry[0]))
        state.highlighted_genes = []
        self._apply_gene_group_sizes(state, groups)
        self._set_gene_highlight_status(state)

    def _discard_highlighted_genes(self, state: GeneInspectorState, genes: set[str]):
        if not state.highlighted_genes:
            return
        removed = set(state.highlighted_genes) & set(genes)
        if not removed:
            return
        groups: set[int] = set()
        for gene in removed:
            entry = state.store.gene_offsets.get(str(gene))
            if entry is not None:
                groups.add(int(entry[0]))
        state.highlighted_genes = [gene for gene in state.highlighted_genes if gene not in removed]
        self._apply_gene_group_sizes(state, groups)
        self._set_gene_highlight_status(state)

    def _apply_gene_group_sizes(self, state: GeneInspectorState, group_indices: set[int]):
        if state.group_display_ranges is None:
            return
        for gi in sorted(group_indices):
            if gi < 0 or gi >= len(state.layer_names):
                continue
            layer = self._get_layer_by_name(state.layer_names[gi])
            if layer is None:
                continue
            n_points = len(getattr(layer, "data", ()))
            ranges = state.group_display_ranges[gi] if gi < len(state.group_display_ranges) else []
            sizes = self._gene_group_size_array(state, n_points, ranges)
            try:
                layer.size = sizes if sizes is not None else float(state.spot_size)
            except Exception:
                pass

    def _set_gene_highlight_status(self, state: GeneInspectorState):
        highlighted_genes = list(state.highlighted_genes or [])
        if not highlighted_genes:
            self._set_status("Click on any transcript to highlight that gene")
            return
        lines = [self._gene_highlight_status_line(state, gene) for gene in highlighted_genes]
        lines.append("<br>Click in empty space to deselect all genes.")
        self._set_status("".join(lines))

    def _gene_highlight_status_line(self, state: GeneInspectorState, gene: str) -> str:
        visual = state.gene_visuals.get(gene)
        symbol = getattr(visual, "symbol", state.store.gene_symbol(gene) or "disc")
        rgba = getattr(visual, "rgba", (1.0, 1.0, 1.0, 1.0))
        glyph = GENE_STATUS_SYMBOL_GLYPHS.get(str(symbol), "●")
        rgb = tuple(max(0, min(255, int(round(float(v) * 255)))) for v in rgba[:3])
        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        count = state.store.full_gene_count(gene)
        return (
            f"<span style='color:{color}; font-size:14pt'>{html.escape(glyph)}</span> "
            f"{html.escape(str(gene))}: {count:,} counts<br>"
        )

    def _gene_ordering_layout(self, state: GeneInspectorState):
        """Return ``(layout, labels)`` for the widget given the current ordering.

        ``layout`` is the ordered header/gene list; ``labels`` maps each gene to
        its "Broad / Fine" text (only used, and only populated, in A–Z mode).
        """
        store = state.store
        visuals = state.gene_visuals or {}
        layout: list[tuple] = []
        labels: dict[str, str] = {}

        if state.ordering == "alphabetical":
            layout = [("gene", gene) for gene in sorted(store.genes)]
            if state.reference:
                for gene in store.genes:
                    info = state.reference.get(gene)
                    if info:
                        fine = info.get("fine")
                        labels[gene] = f"{info['broad']} / {fine}" if fine else str(info["broad"])
            return layout, labels

        scheme = state.fine_scheme if state.ordering == "fine" else state.coarse_scheme
        for title, genes in getattr(scheme, "groups", []):
            rep = genes[len(genes) // 2] if genes else None
            rgba = getattr(visuals.get(rep), "rgba", (0.6, 0.6, 0.6, 1.0)) if rep else (0.6, 0.6, 0.6, 1.0)
            layout.append(("header", title, rgba))
            layout.extend(("gene", gene) for gene in genes)
        return layout, labels

    def _populate_gene_inspector(self, state: GeneInspectorState):
        """(Re)build the gene inspector list widget for ``state``'s ordering."""
        widget = self._gene_inspector_widget
        if widget is None:
            return
        store = state.store
        layout, labels = self._gene_ordering_layout(state)
        widget.set_ordering_available(bool(state.reference), state.ordering)
        widget.populate(
            state.dataset,
            layout,
            state.gene_visuals or {},
            dict(store.source_gene_counts or store.gene_counts),
            set(store.control_genes),
            set(state.enabled_genes),
            labels,
            state.ordering,
            state.hide_assigned,
            state.hide_background,
            state.show_controls,
            state.spot_size,
            state.colour_by_assignment,
        )

    def set_gene_ordering(self, dataset_name: str, kind: str):
        """Switch the gene list ordering (broad / fine / A–Z).

        Broad and fine each recolour the transcript points (symbols are shared, so
        only colours and the list grouping change); A–Z keeps the current colours
        and just relists the genes flat with their cell-type labels.
        """
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        kind = str(kind)
        if kind in ("coarse", "fine"):
            scheme = state.coarse_scheme if kind == "coarse" else state.fine_scheme
            if scheme is None:
                return
            if state.color_kind != kind:
                state.store.recolor(scheme.visuals)
                state.gene_visuals = scheme.visuals
                state.color_kind = kind
                self._rebuild_gene_group_layers(state, group_indices=None)
                if state.highlighted_genes:
                    self._set_gene_highlight_status(state)
            state.ordering = kind
        else:
            state.ordering = "alphabetical"
        self._populate_gene_inspector(state)

    def set_gene_visible(self, dataset_name: str, gene: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        gene = str(gene)
        entry = state.store.gene_offsets.get(gene)
        if entry is None:
            return
        if on:
            state.enabled_genes.add(gene)
        else:
            state.enabled_genes.discard(gene)
            self._discard_highlighted_genes(state, {gene})
        self._schedule_gene_group_rebuild(state, entry[0])

    def set_genes_visible(self, dataset_name: str, genes: list[str], on: bool):
        """Show/hide a batch of genes at once (used by group-heading clicks)."""
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        on = bool(on)
        groups: set[int] = set()
        for gene in genes:
            entry = state.store.gene_offsets.get(str(gene))
            if entry is None:
                continue
            if on:
                state.enabled_genes.add(str(gene))
            else:
                state.enabled_genes.discard(str(gene))
            groups.add(int(entry[0]))
        if not groups:
            return
        if not on:
            self._discard_highlighted_genes(state, {str(g) for g in genes})
        self._rebuild_gene_group_layers(state, group_indices=groups)

    def set_all_genes_visible(self, dataset_name: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        store = state.store
        if on:
            state.enabled_genes = {
                g for g in store.genes if state.show_controls or g not in store.control_genes
            }
        else:
            state.enabled_genes = set()
            self._clear_highlighted_genes(state)
        self._rebuild_gene_group_layers(state, group_indices=None)

    def set_gene_spot_size(self, dataset_name: str, size: float):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        state.spot_size = float(size)
        if state.highlighted_genes:
            self._apply_gene_group_sizes(state, set(range(len(state.layer_names))))
            return
        for name in state.layer_names:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                try:
                    layer.size = float(size)
                except Exception:
                    pass

    def set_gene_hide_background(self, dataset_name: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        state.hide_background = bool(on)
        self._rebuild_gene_group_layers(state, group_indices=None)

    def set_gene_hide_assigned(self, dataset_name: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        state.hide_assigned = bool(on)
        self._rebuild_gene_group_layers(state, group_indices=None)

    def set_gene_colour_by_assignment(self, dataset_name: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        state.colour_by_assignment = bool(on)
        self._rebuild_gene_group_layers(state, group_indices=None)
        self._populate_gene_inspector(state)

    def set_gene_show_controls(self, dataset_name: str, on: bool):
        state = self._gene_inspector_states.get(str(dataset_name).upper())
        if state is None:
            return
        state.show_controls = bool(on)
        if not on:
            # Turning controls off hides any control genes that were enabled.
            self._discard_highlighted_genes(state, set(state.store.control_genes))
            state.enabled_genes -= set(state.store.control_genes)
            self._rebuild_gene_group_layers(state, group_indices=None)

    # -- Cell-type mask colouring ------------------------------------------
    def _cell_type_state(self, dataset_name: str) -> CellTypeOverlayState:
        ds = str(dataset_name).upper()
        state = self._cell_type_states.get(ds)
        if state is None:
            state = CellTypeOverlayState(
                dataset=ds, opacity=float(getattr(self.args, "shape_opacity", 0.95))
            )
            self._cell_type_states[ds] = state
        return state

    def _cell_type_layer_name(self, dataset: str, segmentation: str) -> str:
        return make_layer_name(dataset, "cell_types", f"MOSAIK_{segmentation}")

    def _load_cell_type_data(self, state: CellTypeOverlayState, segmentation: str) -> bool:
        """Cache assignments + colour schemes for ``segmentation``; True if present.

        Results are memoised per segmentation (``None`` recorded when a store has
        no clustering table), so switching level or toggling a type never re-reads
        the store.
        """
        seg = str(segmentation)
        if seg in state.assignments:
            return state.assignments[seg] is not None
        cfg = self.datasets.get(state.dataset)
        zarr_path = getattr(cfg, "zarr_path", None)
        assignments = load_cell_type_assignments(zarr_path, seg) if zarr_path is not None else None
        state.assignments[seg] = assignments
        if assignments is None:
            return False
        schemes = build_cell_type_color_schemes(assignments.broad, assignments.fine, alpha=1.0)
        state.schemes[seg] = schemes
        # Default: every type shown, tracked independently for each level.
        state.enabled[seg] = {
            "broad": set(schemes["broad"].order),
            "fine": set(schemes["fine"].order),
        }
        return True

    @staticmethod
    def _cell_type_counts(assignments, kind: str) -> dict[str, int]:
        labels = np.asarray(assignments.labels_for(kind))
        values, counts = np.unique(labels, return_counts=True)
        return {str(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}

    def _populate_cell_type_panel(self, state: CellTypeOverlayState):
        """(Re)build the cell-type tick list for the active segmentation + level."""
        widget = self._cell_type_widget
        if widget is None:
            return
        seg = state.segmentation
        kind = state.kind or "broad"
        schemes = state.schemes.get(seg)
        assignments = state.assignments.get(seg)
        if schemes is None or assignments is None:
            widget.set_dataset(state.dataset)
            widget.set_status(f"No cell-type annotations stored for {seg} segmentation.")
            return
        scheme = schemes[kind]
        enabled = state.enabled[seg][kind]
        counts = self._cell_type_counts(assignments, kind)
        broad_colors = schemes["broad"].colors
        layout: list[tuple] = []
        for title, members in scheme.groups:
            # Fine headers take their broad class's hue; the single broad header
            # is a neutral grey (its rows already carry the distinct hues).
            header_rgba = broad_colors.get(title, (0.6, 0.6, 0.6, 1.0)) if kind == "fine" else (0.6, 0.6, 0.6, 1.0)
            layout.append(("header", title, header_rgba))
            layout.extend(("type", member) for member in members)
        widget.populate(
            state.dataset,
            layout,
            scheme.colors,
            counts,
            set(enabled),
            seg,
            kind,
            state.opacity,
        )

    def set_cell_type_segmentation(self, dataset_name: str, segmentation: str):
        state = self._cell_type_state(dataset_name)
        state.segmentation = str(segmentation)
        if not self._load_cell_type_data(state, state.segmentation):
            self._set_status(
                f"{state.dataset}: no cell-type annotations for {state.segmentation} segmentation."
            )
            if self._cell_type_widget is not None:
                self._cell_type_widget.set_dataset(state.dataset)
            return
        # Re-colour for the new segmentation only when a level is already active.
        if state.kind:
            self._populate_cell_type_panel(state)
            self._start_cell_type_overlay_build(state)

    def set_cell_type_kind(self, dataset_name: str, kind: str):
        state = self._cell_type_state(dataset_name)
        state.kind = str(kind)
        if not self._load_cell_type_data(state, state.segmentation):
            self._set_status(
                f"{state.dataset}: no cell-type annotations for {state.segmentation} segmentation."
            )
            if self._cell_type_widget is not None:
                self._cell_type_widget.set_dataset(state.dataset)
            return
        self._populate_cell_type_panel(state)
        self._start_cell_type_overlay_build(state)

    def set_cell_type_visible(self, dataset_name: str, label: str, on: bool):
        state = self._cell_type_state(dataset_name)
        enabled = state.enabled.get(state.segmentation, {}).get(state.kind or "broad")
        if enabled is None:
            return
        if on:
            enabled.add(str(label))
        else:
            enabled.discard(str(label))
        self._recolor_cell_type_layer(state)

    def set_cell_types_visible(self, dataset_name: str, labels: list[str], on: bool):
        state = self._cell_type_state(dataset_name)
        enabled = state.enabled.get(state.segmentation, {}).get(state.kind or "broad")
        if enabled is None:
            return
        for label in labels:
            if on:
                enabled.add(str(label))
            else:
                enabled.discard(str(label))
        self._recolor_cell_type_layer(state)

    def set_all_cell_types_visible(self, dataset_name: str, on: bool):
        state = self._cell_type_state(dataset_name)
        seg = state.segmentation
        kind = state.kind or "broad"
        scheme = state.schemes.get(seg, {}).get(kind) if state.schemes.get(seg) else None
        if scheme is None:
            return
        state.enabled[seg][kind] = set(scheme.order) if on else set()
        self._recolor_cell_type_layer(state)

    def set_cell_type_opacity(self, dataset_name: str, opacity: float):
        state = self._cell_type_state(dataset_name)
        state.opacity = float(opacity)
        layer = self._get_layer_by_name(state.layer_name) if state.layer_name else None
        if layer is not None:
            try:
                layer.opacity = min(1.0, max(0.0, float(opacity)))
            except Exception as exc:
                log.debug("Could not set cell-type overlay opacity: %s", exc)

    def _cell_type_color_dict(self, state: CellTypeOverlayState):
        seg = state.segmentation
        kind = state.kind or "broad"
        assignments = state.assignments.get(seg)
        schemes = state.schemes.get(seg)
        if assignments is None or schemes is None:
            return None
        return build_cell_type_color_dict(
            assignments.cell_ids,
            assignments.labels_for(kind),
            schemes[kind],
            enabled=state.enabled[seg][kind],
        )

    def _recolor_cell_type_layer(self, state: CellTypeOverlayState) -> bool:
        """Recolour the existing overlay in place (tick/show-all changes)."""
        layer = self._get_layer_by_name(state.layer_name) if state.layer_name else None
        if layer is None:
            return False
        color_dict = self._cell_type_color_dict(state)
        if color_dict is None:
            return False
        try:
            layer.colormap = DirectLabelColormap(color_dict=color_dict)
        except Exception as exc:
            log.debug("Cell-type overlay recolour failed: %s", exc)
            return False
        return True

    def _ensure_label_key_for_segmentation(self, segmentation: str) -> str:
        """Resolve (loading/rasterizing if needed) the mask label key for a seg.

        Prefers the stored ``MOSAIK_<seg>_labels`` element, whose pixel ids are
        the SpatialData instance-key values the annotations join on.
        """
        if self._active_sdata is None:
            raise RuntimeError("No active dataset.")
        shape_key = f"MOSAIK_{segmentation}"
        label_key = self._label_key_for_shape_key(shape_key)
        if label_key in self._active_sdata.labels or self._refresh_label_key_from_store(label_key):
            return label_key
        if shape_key not in self._active_sdata.shapes:
            shapes_sdata = self._read_shapes_from_store()
            if shapes_sdata is not None:
                for key, value in shapes_sdata.shapes.items():
                    self._active_sdata.shapes[key] = value
        return self.ensure_label_for_shape_key(shape_key)

    def _start_cell_type_overlay_build(self, state: CellTypeOverlayState):
        """Build (off the GUI thread) and draw the filled cell-type Labels layer."""
        seg = state.segmentation
        kind = state.kind or "broad"
        self._cell_type_generation += 1
        generation = self._cell_type_generation
        self._set_status(f"{state.dataset}: colouring {seg} masks by {kind} cell type...")

        def compute():
            label_key = self._ensure_label_key_for_segmentation(seg)
            label_levels, napari_affine = self._build_cellpose_label_display(label_key)
            return {"label_key": label_key, "label_levels": label_levels, "napari_affine": napari_affine}

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_cell_type_error(generation, exc)
                return
            self._apply_cell_type_layer(generation, state, payload)
            return

        worker = thread_worker(compute)()
        worker.returned.connect(
            lambda payload, gen=generation, st=state: self._apply_cell_type_layer(gen, st, payload)
        )
        worker.errored.connect(lambda exc, gen=generation: self._handle_cell_type_error(gen, exc))
        self._cell_type_worker = worker
        worker.start()

    def _apply_cell_type_layer(self, generation: int, state: CellTypeOverlayState, payload: dict):
        if generation != self._cell_type_generation or self.active_dataset != state.dataset:
            return
        self._cell_type_worker = None
        if self._active_sdata is None:
            return
        label_levels = payload["label_levels"]
        label_key = str(payload["label_key"])
        if not label_levels:
            self._set_status(f"{state.dataset} cell-type overlay failed: labels[{label_key}] has no 2D levels.")
            return
        color_dict = self._cell_type_color_dict(state)
        if color_dict is None:
            return
        label_data = label_levels if len(label_levels) > 1 else label_levels[0]
        name = self._cell_type_layer_name(state.dataset, state.segmentation)
        self._remove_layers_by_prefix(layer_name_prefix(state.dataset, "cell_types"))
        layer = self.viewer.add_labels(
            label_data,
            name=name,
            affine=payload["napari_affine"],
            cache=NAPARI_DASK_CACHE_ENABLED,
            colormap=DirectLabelColormap(color_dict=color_dict),
            multiscale=len(label_levels) > 1,
            opacity=min(1.0, max(0.0, float(state.opacity))),
            blending="translucent",
            visible=True,
        )
        # The cell-type fill must sit beneath the transcript points so it never
        # occludes them (add_labels drops it on top of the stack by default).
        self._send_cell_types_below_transcripts(layer)
        state.layer_name = name
        state.label_key = label_key
        assignments = state.assignments.get(state.segmentation)
        kind = state.kind or "broad"
        n_shown = len(state.enabled[state.segmentation][kind])
        n_cells = 0 if assignments is None else int(len(assignments.cell_ids))
        self._set_status(
            f"{state.dataset}: filled {state.segmentation} masks by {kind} cell type "
            f"({n_shown} types shown, {n_cells:,} annotated cells)."
        )
        log.info(
            "[%s] Cell-type overlay label=%s seg=%s kind=%s types_shown=%s cells=%s levels=%s",
            state.dataset, label_key, state.segmentation, kind, n_shown, n_cells, len(label_levels),
        )

    def _handle_cell_type_error(self, generation: int, exc):
        if generation != self._cell_type_generation:
            return
        self._cell_type_worker = None
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"Cell-type overlay failed: {message}")
        log.error("Cell-type overlay failed: %s", message)

    def close_cell_type_overlay(self, dataset_name: str):
        state = self._cell_type_state(dataset_name)
        self._cell_type_generation += 1  # cancel any in-flight build
        self._cell_type_worker = None
        self._remove_layers_by_prefix(layer_name_prefix(state.dataset, "cell_types"))
        state.layer_name = None
        if self._cell_type_widget is not None and self._cell_type_widget.dataset == state.dataset:
            self._cell_type_widget.set_dataset(state.dataset)
        self._set_status(f"{state.dataset}: removed cell-type colouring.")

    def _publish_cell_type_options(self):
        """Refresh the cell-type panel for the active dataset (on dataset load)."""
        widget = self._cell_type_widget
        if widget is None or self.active_dataset is None:
            return
        cfg = self.datasets.get(self.active_dataset)
        zarr_path = getattr(cfg, "zarr_path", None)
        available = {
            seg: (
                zarr_path is not None
                and clustering_table_key_for_segmentation(zarr_path, seg) is not None
            )
            for seg in ("proseg", "cellpose")
        }
        state = self._cell_type_state(self.active_dataset)
        # The previous overlay layer (if any) was dropped by _clear_layers().
        state.layer_name = None
        state.kind = ""
        if available.get(state.segmentation) is False:
            state.segmentation = next((s for s, ok in available.items() if ok), state.segmentation)
        widget.set_dataset(self.active_dataset)
        widget.set_segmentation_available(available)
        if any(available.values()):
            widget.set_status("Choose Broad or Fine cell type to colour the masks.")
        else:
            widget.set_status("No stored cell-type annotations in this dataset.")

    def close_gene_inspector(self, dataset_name: str):
        ds = str(dataset_name).upper()
        had = ds in self._gene_inspector_states
        self._teardown_gene_inspector(ds)
        self._end_progress("transcripts")
        if had:
            self._set_status(f"{ds} unloaded transcripts. Use 'Load transcripts' to bring them back.")

    def _teardown_gene_inspector(self, ds: str):
        ds = str(ds).upper()
        # Cancel any in-flight build so a stale result cannot install layers.
        self._gene_build_generation += 1
        self._gene_build_worker = None
        state = self._gene_inspector_states.pop(ds, None)
        if state is not None and state.rebuild_timer is not None:
            try:
                state.rebuild_timer.stop()
            except Exception:
                pass
        self._remove_layers_by_prefix(layer_name_prefix(ds, "genes"))
        self._invalidate_cell_inspection(ds)
        if self._gene_inspector_widget is not None and self._gene_inspector_widget.dataset == ds:
            self._gene_inspector_widget.clear()

    def _clear_gene_inspector_states(self):
        self._gene_build_generation += 1
        self._gene_build_worker = None
        for state in list(self._gene_inspector_states.values()):
            if state.rebuild_timer is not None:
                try:
                    state.rebuild_timer.stop()
                except Exception:
                    pass
        self._gene_inspector_states.clear()
        self._invalidate_cell_inspection(None)
        if self._gene_inspector_widget is not None:
            self._gene_inspector_widget.clear()

    def load_selected_image(self, dataset_name: str, image_channels):
        entries = [(str(k), str(c)) for k, c in image_channels]
        if not entries:
            self._set_status("No image channel selected.")
            return
        self.load_images_on_demand(dataset_name, image_channels=entries)

    def unload_selected_images(self, dataset_name: str, image_channels):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        ds = str(dataset_name).upper()
        removed = 0
        for image_key, channel in [(str(k), str(c)) for k, c in image_channels]:
            name = make_layer_name(ds, "image", image_key, channel)
            before = len(self.viewer.layers)
            self._remove_layer_by_name(name)
            if len(self.viewer.layers) < before:
                removed += 1
        self._publish_loaded_image_entries()
        self._set_status(f"{ds} removed {removed} image channel layer(s).")

    def load_images_on_demand(self, dataset_name: str, image_channels=None):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return

        ds = str(dataset_name).upper()
        self._ensure_images_loaded(ds)
        if self._x_transform is None or self._y_transform is None:
            raise RuntimeError("Image transform is not initialized.")

        images_sdata = self._active_images_sdata
        only_channels = {(str(k), str(c)) for k, c in image_channels} if image_channels is not None else None
        only_keys = {k for k, _c in only_channels} if only_channels is not None else None
        self._image_build_generation += 1
        generation = self._image_build_generation
        scope = "selected channels" if only_channels is not None else "all images"
        self._begin_progress("images", f"{ds}: computing image pyramids ({scope}; one-time cached build)...")

        def compute():
            # Heavy: streams full-resolution channels once to materialize small
            # coarse-level pyramids into the zarr. Runs on a worker thread so the
            # UI stays responsive; the layer creation happens on the GUI thread.
            t0 = time.time()
            self._prime_image_pyramid_caches(ds, images_sdata, only_keys=only_keys)
            return {"build_seconds": time.time() - t0, "only_channels": only_channels}

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_image_build_error(generation, ds, exc)
                return
            self._apply_image_build(generation, ds, payload)
            return

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, gen=generation, d=ds: self._apply_image_build(gen, d, payload)
        )
        worker.errored.connect(
            lambda exc, gen=generation, d=ds: self._handle_image_build_error(gen, d, exc)
        )
        self._image_build_worker = worker
        worker.start()

    def _apply_image_build(self, generation: int, ds: str, payload: dict[str, object]):
        if generation != self._image_build_generation or self.active_dataset != ds:
            return
        self._image_build_worker = None
        self._end_progress("images")
        only_channels = payload.get("only_channels")
        if only_channels is None:
            self._remove_layers_by_prefix(layer_name_prefix(ds, "image"))
        else:
            for image_key, channel in only_channels:
                self._remove_layer_by_name(make_layer_name(ds, "image", image_key, channel))
        stats = self._add_image_layers(
            ds, self._active_images_sdata, self._x_transform, self._y_transform, only_channels=only_channels
        )
        self._publish_loaded_image_entries()
        self._set_status(
            f"{ds} loaded image layers={stats['layers']} (failed={stats['failed_keys']}); "
            f"pyramid build={float(payload.get('build_seconds', 0.0)):.1f}s."
        )

    def _handle_image_build_error(self, generation: int, ds: str, exc):
        if generation != self._image_build_generation:
            return
        self._image_build_worker = None
        self._end_progress("images")
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{ds} image load failed: {message}")
        log.error("[%s] Image pyramid build/load failed: %s", ds, message)


def log_environment_diagnostics(viewer) -> None:
    """Log the rendering stack once at startup to aid performance triage.

    Records the machine architecture (to catch Rosetta), the Qt binding, the
    VisPy backend, and — deferred until the GL context is live — the OpenGL
    renderer/vendor strings. On Apple Silicon this confirms whether napari is
    running natively and which GL implementation Metal is backing.
    """
    import platform

    try:
        log.info(
            "Environment: machine=%s platform=%s python=%s",
            platform.machine(),
            platform.platform(),
            sys.version.split()[0],
        )
    except Exception:
        pass
    try:
        import qtpy

        log.info("Qt binding: %s (Qt %s)", qtpy.API_NAME, getattr(qtpy, "QT_VERSION", "unknown"))
    except Exception as exc:  # pragma: no cover - depends on runtime env
        log.debug("Could not read Qt binding info (%s)", exc)

    try:
        import vispy
        import vispy.app

        log.info("VisPy: %s backend=%s", vispy.__version__, vispy.app.use_app().backend_name)
    except Exception as exc:  # pragma: no cover - depends on runtime env
        log.debug("Could not read VisPy info (%s)", exc)

    def _log_opengl():
        try:
            from vispy.gloo import gl

            log.info(
                "OpenGL: renderer=%s vendor=%s version=%s",
                gl.glGetParameter(gl.GL_RENDERER),
                gl.glGetParameter(gl.GL_VENDOR),
                gl.glGetParameter(gl.GL_VERSION),
            )
        except Exception as exc:  # pragma: no cover - depends on GL context
            log.debug("Could not read OpenGL info (%s)", exc)

    # The GL context is not realized until the event loop paints; defer the
    # query so glGetParameter runs against a live context.
    try:
        QTimer.singleShot(0, _log_opengl)
    except Exception:  # pragma: no cover - depends on runtime env
        pass


def application_icon_path() -> Path:
    """Return the packaged application-icon path."""
    return Path(__file__).resolve().parent / "assets" / "app_icon.png"


def install_macos_dock_icon(path: Path) -> bool:
    """Set the native Cocoa application icon for an unbundled macOS process."""
    if sys.platform != "darwin":
        return True
    try:
        from AppKit import NSApplication, NSImage

        image = NSImage.alloc().initWithContentsOfFile_(str(path))
        if image is None:
            raise ValueError(f"Cocoa could not read {path}")
        NSApplication.sharedApplication().setApplicationIconImage_(image)
        return True
    except Exception as exc:  # pragma: no cover - depends on the macOS runtime
        log.warning("Could not install the native macOS Dock icon (%s).", exc)
        return False


def install_application_icon(viewer) -> bool:
    """Install the packaged icon on the Qt application and napari window.

    Setting the native window icon is important on macOS because Qt uses it for
    the running application's Dock icon.  The QApplication-level icon covers
    other top-level/dialog windows and the task switcher on other platforms.
    """
    path = application_icon_path()
    icon = QIcon(str(path))
    if not path.is_file() or icon.isNull():
        log.warning("Application icon is missing or unreadable: %s", path)
        return False

    app = QApplication.instance()
    if app is None:  # napari.Viewer normally creates it before this is called.
        log.warning("Cannot install application icon before QApplication exists.")
        return False
    app.setWindowIcon(icon)

    qt_window = getattr(getattr(viewer, "window", None), "_qt_window", None)
    if qt_window is None:
        log.warning("Could not find napari's native Qt window to install its icon.")
        return False
    qt_window.setWindowIcon(icon)
    window_handle = qt_window.windowHandle()
    if window_handle is not None:
        window_handle.setIcon(icon)
    if not install_macos_dock_icon(path):
        return False
    log.info("Installed application icon from %s", path)
    return True


def request_gl_core_profile() -> None:
    """Request an OpenGL 4.1 core-profile context before the Qt app is created.

    On macOS the default (compatibility) context is capped at legacy OpenGL 2.1
    layered over Metal. A core profile is the only way to obtain GL 4.1 on Apple
    Silicon. This is experimental: some VisPy visuals assume legacy GL, so it is
    opt-in via --gl-core-profile. Must run before the first QOpenGLContext, i.e.
    before napari.Viewer() creates the QApplication.
    """
    try:
        from qtpy.QtGui import QSurfaceFormat

        fmt = QSurfaceFormat()
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(4, 1)
        QSurfaceFormat.setDefaultFormat(fmt)
        log.info("Requested OpenGL 4.1 core profile (experimental --gl-core-profile).")
    except Exception as exc:  # pragma: no cover - depends on runtime env
        log.warning("Could not request an OpenGL core profile (%s)", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="napari-compare-xenium-merscope",
        description="Napari comparison viewer for MERSCOPE and Xenium SpatialData outputs."
    )
    parser.add_argument("--merscope-zarr", default=None, type=Path, help="Path to MERSCOPE .zarr output")
    parser.add_argument("--xenium-zarr", default=None, type=Path, help="Path to Xenium .zarr output")

    parser.add_argument(
        "--merscope-transform-path",
        default=None,
        type=Path,
        help="Optional MERSCOPE micron_to_mosaic_pixel_transform.csv path",
    )
    parser.add_argument(
        "--xenium-spec-path",
        default=None,
        type=Path,
        help="Optional Xenium experiment.xenium path (for pixel_size)",
    )

    parser.add_argument(
        "--initial-dataset",
        default="MERSCOPE",
        choices=["MERSCOPE", "XENIUM"],
        help="Dataset loaded first when the viewer opens",
    )

    parser.add_argument(
        "--segmentation-source",
        default="shapes",
        choices=["shapes", "labels"],
        help="Load segmentation layers from shapes (vector) or labels (raster masks).",
    )

    # Startup auto-load controls. By default the viewer loads all images, the
    # Cellpose + ProSeg cell segmentations, and all transcripts (per-gene points).
    # Each --skip-* flag suppresses only the *startup* load of that layer type for
    # low-RAM systems; the layer can still be loaded manually from its tab.
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not auto-load image layers at startup (load them manually from the Images tab).",
    )
    parser.add_argument(
        "--skip-cellpose",
        action="store_true",
        help="Do not auto-load the Cellpose segmentation at startup (load it manually from the Cell segmentation tab).",
    )
    parser.add_argument(
        "--skip-proseg",
        action="store_true",
        help="Do not auto-load the ProSeg segmentation at startup (load it manually from the Cell segmentation tab).",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Do not auto-load transcripts at startup (load them manually from the Gene inspector tab).",
    )
    parser.add_argument(
        "--label-chunk-size",
        default=2048,
        type=int,
        help="Chunk size in pixels for generated cached label masks.",
    )
    parser.add_argument(
        "--label-contour-width",
        default=1,
        type=int,
        help="Outline width in pixels for generated label outline display.",
    )
    parser.add_argument(
        "--label-interpolation",
        default="linear",
        choices=("linear", "nearest"),
        help=(
            "Screen interpolation for label-outline layers. 'linear' anti-aliases "
            "the outlines when zoomed out so they stay thin and fade rather than "
            "merging into blocks of colour; 'nearest' is crisper up close but blocky "
            "when zoomed out."
        ),
    )
    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Rebuild cached label elements even if labels with the same keys already exist.",
    )
    parser.add_argument(
        "--overwrite-derived-caches",
        action="store_true",
        help="Rebuild viewer-derived density/outline/image-pyramid caches even when matching cache metadata exists.",
    )
    parser.add_argument(
        "--image-pyramid-downsample",
        default=4,
        type=int,
        help=(
            "Downsample factor between materialized image-pyramid levels for single-scale "
            "images (>=2). 4 keeps caches small; 2 is smoothest but ~5x larger."
        ),
    )
    parser.add_argument(
        "--visible-channels",
        default=None,
        help=(
            "Comma-separated channel names to show by default when images load (e.g. 'DAPI,PolyT'). "
            "Others are loaded but hidden and can be toggled in napari. Default: a DAPI-like channel, "
            "else the first channel."
        ),
    )
    parser.add_argument("--random-state", default=42, type=int, help="Random seed used for sampling")

    parser.add_argument(
        "--gene-spot-size",
        default=DEFAULT_GENE_SPOT_SIZE,
        type=float,
        help="Initial world-micron spot size for the per-gene transcript renderer",
    )
    parser.add_argument(
        "--gene-max-render-points",
        default=DEFAULT_GENE_MAX_RENDER_POINTS,
        type=int,
        help="Cap on total per-gene points rendered at once; above this the store is uniformly subsampled",
    )
    parser.add_argument(
        "--gene-hide-assigned",
        action="store_true",
        help="Start the transcript view with assigned spots hidden",
    )
    parser.add_argument(
        "--gene-hide-background",
        action="store_true",
        help="Start the transcript view with background (unassigned) spots hidden",
    )
    parser.add_argument(
        "--gene-show-controls",
        action="store_true",
        help="Start the transcript view with control/blank probes shown",
    )
    parser.add_argument(
        "--gene-min-canvas-px",
        default=0.0,
        type=float,
        help=(
            "Minimum on-screen size (canvas pixels) for transcript spots. napari's "
            "default of 2 keeps zoomed-out spots at 2px so millions of them blanket "
            "the view into a solid mass; 0 lets spots shrink with zoom so density "
            "structure shows through."
        ),
    )
    parser.add_argument(
        "--gene-antialiasing",
        default=1.0,
        type=float,
        help=(
            "Antialiasing width (px) for transcript spots. >0 lets sub-pixel spots "
            "contribute partial opacity, so dense regions read darker than sparse "
            "ones (density structure appears when zoomed out). Set 0 for hard-edged "
            "spots if antialiasing costs too much performance."
        ),
    )
    parser.add_argument("--shape-edge-width", default=0.75, type=float, help="Segmentation/annotation edge width")
    parser.add_argument("--shape-opacity", default=0.95, type=float, help="Segmentation outline edge alpha")
    parser.add_argument("--image-opacity", default=1.0, type=float, help="Image layer opacity")

    parser.add_argument("--hide-images", action="store_true", help="Start with image layers hidden")
    parser.add_argument("--hide-shapes", action="store_true", help="Start with segmentation layers hidden")
    parser.add_argument(
        "--gl-core-profile",
        action="store_true",
        help=(
            "Experimental: request an OpenGL 4.1 core-profile context instead of the macOS "
            "default legacy 2.1 context. Check the 'OpenGL: version=...' startup log to confirm. "
            "May affect rendering on some setups; leave off if visuals look wrong."
        ),
    )
    parser.add_argument("--package-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--package-smoke-test-no-opengl", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.label_chunk_size <= 0:
        parser.error("--label-chunk-size must be positive.")
    if args.label_contour_width < 0:
        parser.error("--label-contour-width must be non-negative.")
    if args.image_pyramid_downsample < 2:
        parser.error("--image-pyramid-downsample must be >= 2.")
    return args


def run_package_smoke_test_without_opengl() -> None:
    """Exercise the frozen Qt control panel on CI hosts without OpenGL."""
    app = QApplication.instance() or QApplication([])
    callback = lambda *args, **kwargs: None
    panel = ViewerControlPanel(
        datasets=[],
        gene_inspector_widget=None,
        cell_type_widget=None,
        load_callback=callback,
        load_selected_labels_callback=callback,
        unload_selected_shapes_callback=callback,
        load_transcripts_callback=callback,
        unload_transcripts_callback=callback,
        load_selected_image_callback=callback,
        load_all_images_callback=callback,
        unload_selected_image_callback=callback,
        load_cellpose_values_callback=callback,
        remove_cellpose_values_callback=callback,
        create_annotation_layers_callback=callback,
        set_annotation_piece_callback=callback,
        apply_annotation_piece_callback=callback,
        snap_annotation_side_edges_callback=callback,
        validate_annotation_callback=callback,
        export_annotation_callback=callback,
        create_object_annotation_callback=callback,
        validate_object_annotations_callback=callback,
        export_object_annotations_callback=callback,
        load_object_annotations_callback=callback,
        load_paired_callback=callback,
        load_standalone_callback=callback,
        initial_dataset=None,
    )
    if panel._tab_stack.currentIndex() != 6:
        raise RuntimeError("The empty viewer did not select the Dataset loader tab.")
    panel.close()
    app.processEvents()


def main():
    args = parse_args()

    if args.package_smoke_test_no_opengl:
        run_package_smoke_test_without_opengl()
        return

    if args.merscope_zarr is not None and not args.merscope_zarr.exists():
        raise FileNotFoundError(f"MERSCOPE zarr path not found: {args.merscope_zarr}")
    if args.xenium_zarr is not None and not args.xenium_zarr.exists():
        raise FileNotFoundError(f"Xenium zarr path not found: {args.xenium_zarr}")

    datasets: dict[str, DatasetConfig] = {}
    if args.merscope_zarr is not None:
        datasets["MERSCOPE"] = DatasetConfig(
            name="MERSCOPE",
            zarr_path=args.merscope_zarr,
            merscope_transform_path=args.merscope_transform_path,
            xenium_spec_path=args.xenium_spec_path,
        )
    if args.xenium_zarr is not None:
        datasets["XENIUM"] = DatasetConfig(
            name="XENIUM",
            zarr_path=args.xenium_zarr,
            merscope_transform_path=args.merscope_transform_path,
            xenium_spec_path=args.xenium_spec_path,
        )

    available_datasets = list(datasets.keys())
    initial_dataset: str | None = args.initial_dataset if datasets else None
    if datasets and initial_dataset not in datasets:
        initial_dataset = available_datasets[0]
        log.info(
            "Initial dataset %s was not supplied; starting with %s.",
            args.initial_dataset,
            initial_dataset,
        )

    for cfg in datasets.values():
        validate_spatialdata_store_compatibility(cfg.zarr_path)

    if getattr(args, "gl_core_profile", False):
        request_gl_core_profile()

    dask_cache = install_thread_safe_napari_dask_cache()
    log.info(
        "Installed thread-safe napari Dask cache (budget %.1f GB)",
        float(dask_cache.cache.available_bytes) / (1024**3),
    )
    viewer = napari.Viewer(title="Spatialomics Viewer")
    install_application_icon(viewer)
    log_environment_diagnostics(viewer)
    controller = ComparisonViewerController(viewer=viewer, datasets=datasets, args=args)

    # The gene inspector widget is embedded as the "Gene inspector" tab of the
    # control panel, so build it first and pass it in.
    gene_inspector = GeneInspectorWidget(
        close_callback=controller.close_gene_inspector,
        set_gene_visible_callback=controller.set_gene_visible,
        set_all_genes_callback=controller.set_all_genes_visible,
        set_spot_size_callback=controller.set_gene_spot_size,
        set_hide_background_callback=controller.set_gene_hide_background,
        set_show_controls_callback=controller.set_gene_show_controls,
        set_hide_assigned_callback=controller.set_gene_hide_assigned,
        set_colour_by_assignment_callback=controller.set_gene_colour_by_assignment,
        set_ordering_callback=controller.set_gene_ordering,
        set_genes_visible_callback=controller.set_genes_visible,
        spot_size=args.gene_spot_size,
        hide_assigned=args.gene_hide_assigned,
        hide_background=args.gene_hide_background,
        show_controls=args.gene_show_controls,
    )
    # The cell-type mask-fill widget is the "Cell type labels" tab.
    cell_type_widget = CellTypeWidget(
        close_callback=controller.close_cell_type_overlay,
        set_segmentation_callback=controller.set_cell_type_segmentation,
        set_kind_callback=controller.set_cell_type_kind,
        set_type_visible_callback=controller.set_cell_type_visible,
        set_types_visible_callback=controller.set_cell_types_visible,
        set_all_types_callback=controller.set_all_cell_types_visible,
        set_opacity_callback=controller.set_cell_type_opacity,
        opacity=float(getattr(args, "shape_opacity", 0.95)),
    )
    panel = ViewerControlPanel(
        datasets=available_datasets,
        gene_inspector_widget=gene_inspector,
        cell_type_widget=cell_type_widget,
        load_callback=controller.load_dataset,
        load_selected_labels_callback=controller.load_selected_labels,
        unload_selected_shapes_callback=controller.unload_selected_shapes,
        load_transcripts_callback=controller.open_gene_inspector,
        unload_transcripts_callback=controller.close_gene_inspector,
        load_selected_image_callback=controller.load_selected_image,
        load_all_images_callback=controller.load_images_on_demand,
        unload_selected_image_callback=controller.unload_selected_images,
        load_cellpose_values_callback=controller.load_cellpose_value_overlay,
        remove_cellpose_values_callback=controller.remove_cellpose_value_overlay,
        create_annotation_layers_callback=controller.create_cortical_depth_annotation_layers,
        set_annotation_piece_callback=controller.set_cortical_depth_current_piece,
        apply_annotation_piece_callback=controller.apply_cortical_depth_piece_to_selection,
        snap_annotation_side_edges_callback=controller.snap_cortical_depth_side_edges,
        validate_annotation_callback=controller.validate_cortical_depth_annotations,
        export_annotation_callback=controller.export_cortical_depth_annotations,
        create_object_annotation_callback=controller.create_distance_object_annotation_layer,
        validate_object_annotations_callback=controller.validate_distance_object_annotations,
        export_object_annotations_callback=controller.export_distance_object_annotations,
        load_object_annotations_callback=controller.load_distance_object_annotations,
        load_paired_callback=controller.load_paired_dataset,
        load_standalone_callback=controller.load_standalone_dataset,
        initial_dataset=initial_dataset,
    )
    controller.set_status_callback(panel.set_status)
    controller.set_progress_callback(panel.set_progress)
    controller.set_shape_keys_callback(panel.set_shape_keys)
    controller.set_loaded_shape_keys_callback(panel.set_loaded_shape_keys)
    controller.set_image_entries_callback(panel.set_image_entries)
    controller.set_loaded_image_entries_callback(panel.set_loaded_image_entries)
    controller.set_datasets_changed_callback(panel.set_datasets)
    controller.set_cellpose_value_options_callback(panel.set_cellpose_value_options)
    controller.set_gene_inspector_widget(gene_inspector)
    controller.set_cell_type_widget(cell_type_widget)
    viewer.window.add_dock_widget(panel, area="right", name="Viewer Controls")

    controller.install_canvas_overlays()
    if initial_dataset is not None:
        controller.load_dataset(initial_dataset, force=True)
    else:
        panel.set_status("No dataset loaded. Choose a dataset from the Dataset loader tab.")
    if args.package_smoke_test:
        # Installer CI exercises the fully frozen Qt/napari application, then
        # exits without requiring human interaction or loading a dataset.
        viewer.close()
        return
    napari.run()


if __name__ == "__main__":
    main()
