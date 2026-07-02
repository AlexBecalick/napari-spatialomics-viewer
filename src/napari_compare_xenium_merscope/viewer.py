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
import json
import logging
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    from qtpy.QtCore import Qt, QTimer, Signal
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
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

from .utils import (
    CELLPOSE_LABEL_KEY,
    CELLPOSE_QUANTIFICATION_TABLE_KEY,
    CELLPOSE_SHAPE_KEY,
    CELLPOSE_VALUE_BINS,
    CORTICAL_DEPTH_DEFAULT_PIECE_ID,
    CORTICAL_DEPTH_PIECE_ID_PROPERTY,
    CORTICAL_DEPTH_ROLE_ORDER,
    CORTICAL_DEPTH_ROLE_SPECS,
    CorticalDepthShapeInput,
    DERIVED_CACHE_ATTR,
    assignment_mask,
    affine_matrix_from_px_to_um,
    build_binned_label_color_dict,
    build_cortical_depth_annotation_geojson,
    build_transcript_spatial_index,
    build_napari_affine_from_px_to_um,
    cellpose_quantification_features,
    cellpose_quantification_table_available,
    channel_labels,
    compute_transcript_density_array,
    derived_outline_cache_key,
    derived_transcript_density_cache_key,
    ensure_cyx,
    first_existing_col,
    geometry_to_napari_bounding_boxes,
    geometry_to_napari_centroids,
    geometry_to_napari_polygons,
    get_scale0_dataarray,
    image_scale_dataarrays,
    is_derived_cache_key,
    label_outline_mask_chunk,
    layer_name_prefix,
    load_cellpose_quantification_values,
    load_points_dataframe,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    query_geometries_for_bounds,
    query_transcript_spatial_index,
    rasterize_geometries_chunk,
    resolve_dataset_mask_affine,
    snap_cortical_depth_boundaries_to_edge,
    write_cortical_depth_annotation_geojson,
    write_cortical_depth_separate_geojsons,
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
class StreamedTranscriptLayerState:
    dataset: str
    points_key: str
    assigned_layer_name: str
    unassigned_layer_name: str
    timer: QTimer
    view_percent: int
    point_index: object | None
    density_layer_names: list[str]
    generation: int = 0
    updating: bool = False
    pending: bool = False
    worker: object | None = None
    camera_callback: object | None = None


FAST_DEFAULT_MAX_TRANSCRIPTS = 200_000
FAST_DEFAULT_MAX_SHAPES_PER_LAYER = 20_000
DEFAULT_TRANSCRIPT_VIEW_PERCENT = 0
DEFAULT_TRANSCRIPT_VIEW_MARGIN_FRACTION = 0.20
DEFAULT_TRANSCRIPT_STREAM_DEBOUNCE_MS = 300
DEFAULT_TRANSCRIPT_DENSITY_BIN_UM = 2.0
TRANSCRIPT_DENSITY_PYRAMID_NORMALIZATION = "mean_v1"
DEFAULT_TRANSCRIPT_DENSITY_MAX_PIXELS = 25_000_000
DEFAULT_TRANSCRIPT_DETAIL_MAX_VIEW_SIZE = 1_500.0
DEFAULT_TRANSCRIPT_DETAIL_MAX_POINTS = 300_000
DEFAULT_TRANSCRIPT_INDEX_MAX_POINTS = 25_000_000
DEFAULT_TRANSCRIPT_INDEX_TILE_UM = 250.0
SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE = 4096
SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS = 10
LABEL_CACHE_ATTR = "napari_compare_label_cache"
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


def startup_selection(startup_mode: str, skip_images: bool, segmentation_source: str) -> tuple[str, ...] | None:
    """Return the SpatialData read selection for startup loading."""
    seg = "labels" if str(segmentation_source).lower() == "labels" else "shapes"
    mode = str(startup_mode).lower()
    if mode == "full":
        if skip_images:
            return ("points", seg)
        return None
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


class DatasetSwitcherWidget(QWidget):
    """Dock widget with dataset controls and on-demand layer loading controls."""

    # Thread-safe status updates: heavy work runs in napari thread_workers that
    # call ``set_status`` from a background thread. Routing every update through a
    # Qt signal marshals the actual widget mutation back onto the GUI thread.
    status_message = Signal(str)

    def __init__(
        self,
        datasets: list[str],
        load_callback,
        load_selected_labels_callback,
        unload_selected_shapes_callback,
        load_full_transcripts_callback,
        set_transcript_view_percent_callback,
        load_images_callback,
        load_cellpose_values_callback,
        remove_cellpose_values_callback,
        create_annotation_layers_callback,
        set_annotation_piece_callback,
        apply_annotation_piece_callback,
        snap_annotation_side_edges_callback,
        validate_annotation_callback,
        export_annotation_callback,
        export_separate_annotations_callback,
        initial_dataset: str = "MERSCOPE",
        skip_images: bool = False,
        transcript_view_percent: int = DEFAULT_TRANSCRIPT_VIEW_PERCENT,
    ):
        super().__init__()
        self._load_callback = load_callback
        self._load_selected_labels_callback = load_selected_labels_callback
        self._unload_selected_shapes_callback = unload_selected_shapes_callback
        self._load_full_transcripts_callback = load_full_transcripts_callback
        self._set_transcript_view_percent_callback = set_transcript_view_percent_callback
        self._load_images_callback = load_images_callback
        self._load_cellpose_values_callback = load_cellpose_values_callback
        self._remove_cellpose_values_callback = remove_cellpose_values_callback
        self._create_annotation_layers_callback = create_annotation_layers_callback
        self._set_annotation_piece_callback = set_annotation_piece_callback
        self._apply_annotation_piece_callback = apply_annotation_piece_callback
        self._snap_annotation_side_edges_callback = snap_annotation_side_edges_callback
        self._validate_annotation_callback = validate_annotation_callback
        self._export_annotation_callback = export_annotation_callback
        self._export_separate_annotations_callback = export_separate_annotations_callback
        self._skip_images = bool(skip_images)

        self._dataset_combo = QComboBox()
        self._dataset_combo.addItems(datasets)
        if initial_dataset in datasets:
            self._dataset_combo.setCurrentText(initial_dataset)
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)

        self._reload_button = QPushButton("Reload Dataset")
        self._reload_button.clicked.connect(self._on_reload_clicked)

        self._shape_list = QListWidget()
        self._shape_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._shape_list.setMaximumHeight(120)
        self._shape_list.setMinimumHeight(80)
        self._shape_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._load_selected_labels_button = QPushButton("Load Selected Labels (Outlines)")
        self._load_selected_labels_button.clicked.connect(self._on_load_selected_labels)
        self._unload_selected_shapes_button = QPushButton("Unload Selected Segmentations")
        self._unload_selected_shapes_button.clicked.connect(self._on_unload_selected_shapes)

        self._cellpose_channel_combo = QComboBox()
        self._cellpose_statistic_combo = QComboBox()
        self._cellpose_colormap_combo = QComboBox()
        self._cellpose_colormap_combo.addItems(["viridis", "magma", "inferno", "plasma", "turbo", "gray"])
        self._load_cellpose_values_button = QPushButton("Load Cellpose Value Overlay")
        self._load_cellpose_values_button.clicked.connect(self._on_load_cellpose_values)
        self._remove_cellpose_values_button = QPushButton("Remove Cellpose Value Overlay")
        self._remove_cellpose_values_button.clicked.connect(self._on_remove_cellpose_values)
        self.set_cellpose_value_options([], [], enabled=False)

        self._load_full_tx_button = QPushButton("Load Full Transcripts (Viewport)")
        self._load_full_tx_button.clicked.connect(self._on_load_full_transcripts)
        self._tx_percent_slider = QSlider(Qt.Horizontal)
        self._tx_percent_slider.setRange(0, 100)
        self._tx_percent_slider.setSingleStep(1)
        self._tx_percent_slider.setPageStep(10)
        self._tx_percent_slider.setValue(int(transcript_view_percent))
        self._tx_percent_slider.valueChanged.connect(self._on_tx_percent_changed)
        self._tx_percent_label = QLabel()
        self._tx_percent_label.setWordWrap(True)
        self._update_tx_percent_label(self._tx_percent_slider.value())

        self._load_images_button = QPushButton("Load Images")
        self._load_images_button.setEnabled(not self._skip_images)
        self._load_images_button.clicked.connect(self._on_load_images)

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
        self._export_separate_annotations_button = QPushButton("Export Separate GeoJSONs")
        self._export_separate_annotations_button.clicked.connect(self._on_export_separate_annotations)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        self.status_message.connect(self._on_status_message)

        content = QWidget()
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(QLabel("Dataset"))
        root.addWidget(self._dataset_combo)
        root.addWidget(self._reload_button)
        root.addWidget(QLabel("Segmentations"))
        root.addWidget(self._shape_list)

        root.addWidget(self._load_selected_labels_button)
        root.addWidget(self._unload_selected_shapes_button)

        root.addWidget(QLabel("Cell Mask Values"))
        root.addWidget(QLabel("Channel"))
        root.addWidget(self._cellpose_channel_combo)
        root.addWidget(QLabel("Statistic"))
        root.addWidget(self._cellpose_statistic_combo)
        root.addWidget(QLabel("Colormap"))
        root.addWidget(self._cellpose_colormap_combo)
        root.addWidget(self._load_cellpose_values_button)
        root.addWidget(self._remove_cellpose_values_button)

        root.addWidget(QLabel("Transcripts"))
        root.addWidget(self._load_full_tx_button)
        tx_percent_row = QHBoxLayout()
        tx_percent_row.addWidget(QLabel("Viewport %"))
        tx_percent_row.addWidget(self._tx_percent_slider)
        root.addLayout(tx_percent_row)
        root.addWidget(self._tx_percent_label)

        if not self._skip_images:
            root.addWidget(self._load_images_button)

        root.addWidget(QLabel("Cortical Depth Annotations"))
        root.addWidget(self._create_annotations_button)
        root.addWidget(QLabel("Current Tissue Piece"))
        root.addWidget(self._piece_combo)
        piece_row = QHBoxLayout()
        piece_row.addWidget(self._new_piece_button)
        piece_row.addWidget(self._apply_piece_button)
        root.addLayout(piece_row)
        root.addWidget(self._snap_side_edges_button)
        root.addWidget(self._validate_annotations_button)
        root.addWidget(self._export_annotations_button)
        root.addWidget(self._export_separate_annotations_button)

        root.addWidget(self._status_label)
        root.addStretch(1)
        content.setLayout(root)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(220)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    @property
    def current_dataset(self) -> str:
        return str(self._dataset_combo.currentText())

    def set_status(self, text: str):
        # Safe to call from any thread; the queued signal hops to the GUI thread.
        self.status_message.emit(str(text))

    def _on_status_message(self, text: str):
        self._status_label.setText(str(text))

    def set_shape_keys(self, keys: list[str]):
        self._shape_list.clear()
        self._shape_list.addItems([str(k) for k in keys])

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

    def transcript_view_percent(self) -> int:
        return int(self._tx_percent_slider.value())

    def current_annotation_piece(self) -> str:
        text = str(self._piece_combo.currentText()).strip()
        return text or CORTICAL_DEPTH_DEFAULT_PIECE_ID

    def _update_tx_percent_label(self, value: int):
        if int(value) <= 0:
            text = "Auto: load as many viewport transcripts as the display cap allows."
        else:
            text = f"Manual: load {int(value)}% of transcripts in the current viewport."
        self._tx_percent_label.setText(text)

    def _on_dataset_changed(self, text: str):
        if not text:
            return
        self._load_callback(str(text), False)

    def _on_reload_clicked(self):
        self._load_callback(self.current_dataset, True)

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

    def _on_load_full_transcripts(self):
        self._load_full_transcripts_callback(self.current_dataset, self.transcript_view_percent())

    def _on_tx_percent_changed(self, value: int):
        value = int(value)
        self._update_tx_percent_label(value)
        self._set_transcript_view_percent_callback(self.current_dataset, value)

    def _on_load_images(self):
        self._load_images_callback(self.current_dataset)

    def _on_load_cellpose_values(self):
        if not self._load_cellpose_values_button.isEnabled():
            self.set_status("Cellpose value overlay is not available for this dataset.")
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

    def _on_export_separate_annotations(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Export Separate Cortical Depth GeoJSONs",
            "",
        )
        if not directory:
            return
        try:
            result = self._export_separate_annotations_callback(self.current_dataset, Path(directory))
        except Exception as exc:
            self.set_status(f"Separate annotation export failed: {exc}")
            QMessageBox.warning(self, "Export Separate GeoJSONs", str(exc))
            return
        self._show_annotation_validation_result("Export Separate GeoJSONs", result, export_path=Path(directory))

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


class ComparisonViewerController:
    """Coordinate loading/clearing napari layers for each dataset."""

    def __init__(self, viewer: napari.Viewer, datasets: dict[str, DatasetConfig], args):
        self.viewer = viewer
        self.datasets = datasets
        self.args = args
        self.active_dataset: str | None = None
        self._status_callback = None
        self._shape_keys_callback = None
        self._cellpose_value_options_callback = None
        self._current_cortical_depth_piece_id = CORTICAL_DEPTH_DEFAULT_PIECE_ID
        self._active_sdata = None
        self._active_images_sdata = None
        self._segmentation_keys: list[str] = []
        self._x_transform: tuple[float, float, float] | None = None
        self._y_transform: tuple[float, float, float] | None = None
        self._streamed_transcript_states: dict[str, StreamedTranscriptLayerState] = {}
        self._cellpose_value_generation = 0
        self._cellpose_value_worker: object | None = None
        # Background build bookkeeping for the heavy transcript / label pipelines
        # that now run in napari thread_workers so the UI stays responsive.
        self._transcript_build_generation = 0
        self._transcript_build_worker: object | None = None
        self._label_build_generation = 0
        self._label_build_worker: object | None = None
        self._canvas_size: tuple[int, int] | None = None

    def set_status_callback(self, fn):
        self._status_callback = fn

    def set_shape_keys_callback(self, fn):
        self._shape_keys_callback = fn

    def set_cellpose_value_options_callback(self, fn):
        self._cellpose_value_options_callback = fn

    def _set_status(self, text: str):
        if self._status_callback is not None:
            self._status_callback(text)

    def _publish_shape_keys(self):
        if self._shape_keys_callback is not None:
            self._shape_keys_callback(list(self._segmentation_keys))

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
        self._clear_streamed_transcript_states()
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)

    def _clear_streamed_transcript_states(self):
        # Invalidate any in-flight background transcript build so a stale result
        # cannot install layers/state after the dataset was cleared.
        self._transcript_build_generation += 1
        self._transcript_build_worker = None
        for state in list(self._streamed_transcript_states.values()):
            try:
                state.timer.stop()
            except Exception:
                pass
            self._disconnect_transcript_camera_callback(state)
        self._streamed_transcript_states.clear()

    def _disconnect_transcript_camera_callback(self, state: StreamedTranscriptLayerState):
        callback = getattr(state, "camera_callback", None)
        if callback is None:
            return
        for events in (self.viewer.camera.events.zoom, self.viewer.camera.events.center):
            try:
                events.disconnect(callback)
            except Exception:
                pass
        state.camera_callback = None

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

    def _add_points_layer(
        self,
        coords: np.ndarray,
        name: str,
        color,
        visible: bool,
        size: float | None = None,
        opacity: float | None = None,
    ):
        """Add a points layer with napari-version-compatible color kwargs."""
        common_kwargs = dict(
            size=self.args.point_size if size is None else size,
            opacity=self.args.point_opacity if opacity is None else opacity,
            visible=visible,
        )
        try:
            return self.viewer.add_points(
                coords,
                name=name,
                face_color=color,
                edge_color=color,
                **common_kwargs,
            )
        except TypeError:
            # Newer napari versions renamed edge_color -> border_color.
            return self.viewer.add_points(
                coords,
                name=name,
                face_color=color,
                border_color=color,
                **common_kwargs,
            )

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

    def _ensure_dataset_is_active(self, dataset_name: str) -> bool:
        ds = str(dataset_name).upper()
        if self.active_dataset != ds:
            self.load_dataset(ds, force=False)
        return self.active_dataset == ds and self._active_sdata is not None

    def _ensure_images_loaded(self, ds: str):
        if self.args.skip_images:
            raise RuntimeError("Image loading is disabled by --skip-images.")
        if self._active_images_sdata is not None and len(self._active_images_sdata.images) > 0:
            return
        cfg = self.datasets[ds]
        self._active_images_sdata = sd.read_zarr(str(cfg.zarr_path), selection=("images",))

    def load_dataset(self, dataset_name: str, force: bool = False):
        ds = str(dataset_name).upper()
        if ds not in self.datasets:
            self._set_status(f"Unknown dataset: {dataset_name}")
            return

        if (self.active_dataset == ds) and (not force):
            return

        mem_before = memory_snapshot_gb()
        t0 = time.time()
        cfg = self.datasets[ds]

        try:
            self._set_status(f"Loading {ds}...")
            log.info("[%s] Loading dataset from %s", ds, cfg.zarr_path)

            self._clear_layers()
            gc.collect()
            self._segmentation_keys = []
            self._publish_shape_keys()
            if self._cellpose_value_options_callback is not None:
                self._cellpose_value_options_callback([], [], False)

            selection = startup_selection(
                self.args.startup_mode,
                self.args.skip_images,
                self._segmentation_source(),
            )
            sdata = sd.read_zarr(str(cfg.zarr_path), selection=selection)
            self._active_sdata = sdata
            self._active_images_sdata = sdata if len(getattr(sdata, "images", {})) > 0 else None
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
            self._publish_shape_keys()
            self._publish_cellpose_value_options()

            if self.args.startup_mode == "full":
                img_stats = self._add_image_layers(ds, sdata, x_transform, y_transform)
                shape_stats = self._add_shape_layers(ds, sdata)
                tx_stats = self._add_transcripts_layer(self.args.max_transcripts)
            else:
                img_stats = {
                    "layers": 0,
                    "failed_keys": 0,
                    "skipped": bool(self.args.skip_images),
                    "deferred": not bool(self.args.skip_images),
                }
                shape_stats = {"layers": 0, "units": 0}
                tx_stats = self._add_transcripts_layer(self.args.max_transcripts)

            elapsed = time.time() - t0
            mem_after = memory_snapshot_gb()

            image_summary = f"images={img_stats['layers']}"
            if img_stats.get("deferred", False):
                image_summary = "images=deferred"
            elif img_stats.get("skipped", False):
                image_summary = "images=skipped"
            elif img_stats.get("failed_keys", 0) > 0:
                image_summary = (
                    f"images={img_stats['layers']} "
                    f"(failed_keys={img_stats['failed_keys']})"
                )

            summary = (
                f"{ds} loaded in {elapsed:.1f}s ({self.args.startup_mode} startup) | "
                f"{image_summary} | "
                f"{self._segmentation_layer_type()}_layers={shape_stats['layers']} "
                f"{self._segmentation_unit_name()}={shape_stats['units']:,} | "
                f"tx_total={tx_stats['total']:,} assigned={tx_stats['assigned']:,} "
                f"unassigned={tx_stats['unassigned']:,} | "
                f"RSS {mem_before['rss_gb']:.1f}->{mem_after['rss_gb']:.1f} GB"
            )
            self._set_status(summary)
            log.info(summary)

        except Exception as exc:
            self._set_status(f"Failed to load {ds}: {exc}")
            log.exception("[%s] Failed to load dataset", ds)
            self.active_dataset = None
            self._active_sdata = None
            self._active_images_sdata = None
            self._segmentation_keys = []
            self._publish_shape_keys()
            if self._cellpose_value_options_callback is not None:
                self._cellpose_value_options_callback([], [], False)

    def _add_image_layers(self, ds: str, sdata, x_transform, y_transform) -> dict[str, int]:
        if getattr(self.args, "skip_images", False):
            log.info("[%s] Image loading skipped (--skip-images).", ds)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        visible = not self.args.hide_images
        total_layers = 0
        failed_keys = 0

        try:
            image_keys = [str(k) for k in sdata.images.keys() if not is_derived_cache_key(str(k))]
        except Exception as exc:
            log.warning("[%s] Could not enumerate images; skipping image loading (%s)", ds, exc)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        if len(image_keys) == 0:
            log.info("[%s] No images found in SpatialData; continuing without image layers.", ds)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        for image_key in image_keys:
            try:
                scale_levels = [
                    (scale_name, ensure_cyx(da))
                    for scale_name, da in image_scale_dataarrays(sdata.images[image_key])
                ]
                if len(scale_levels) == 0:
                    raise ValueError("image has no readable scale levels")

                base_scale_name, base_image_cyx = scale_levels[0]
                labels = channel_labels(base_image_cyx)

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
                    layer_name = make_layer_name(ds, "image", image_key, chan_name)
                    channel_levels = [
                        image_cyx.isel(c=chan_idx).data
                        for _scale_name, image_cyx in scale_levels
                    ]
                    pyramid_source = "stored"
                    if len(channel_levels) == 1:
                        channel_levels = lazy_subsampled_pyramid(channel_levels[0])
                        pyramid_source = "synthetic" if len(channel_levels) > 1 else "single"

                    multiscale = len(channel_levels) > 1
                    ch_data = channel_levels if multiscale else channel_levels[0]
                    cmap = image_colormap_for_channel(chan_name, chan_idx)

                    add_kwargs = dict(
                        name=layer_name,
                        affine=affine,
                        colormap=cmap,
                        blending="additive",
                        opacity=self.args.image_opacity,
                        visible=visible,
                    )
                    if multiscale:
                        add_kwargs["multiscale"] = True

                    contrast_limits = contrast_limits_from_dtype(channel_levels[0])
                    if contrast_limits is not None:
                        add_kwargs["contrast_limits"] = contrast_limits

                    self.viewer.add_image(ch_data, **add_kwargs)
                    total_layers += 1
                    shapes = [tuple(int(axis) for axis in level.shape) for level in channel_levels]
                    log.info(
                        "[%s] Added image layer %s from %s levels=%s source=%s base=%s dtype=%s",
                        ds,
                        layer_name,
                        image_key,
                        len(channel_levels),
                        pyramid_source,
                        base_scale_name,
                        getattr(channel_levels[0], "dtype", "unknown"),
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

    def _add_shape_layer(
        self,
        shape_key: str,
        cap: int | None = None,
        simplify_tolerance: float | None = None,
        render_mode: str | None = None,
    ) -> int:
        if self._segmentation_source() == "labels":
            return self._add_label_layer(shape_key)

        if self._active_sdata is None or self.active_dataset is None:
            return 0
        ds = self.active_dataset
        if shape_key not in self._active_sdata.shapes:
            raise KeyError(f"Shape key '{shape_key}' not found in current dataset.")

        layer_name = make_layer_name(ds, self._segmentation_layer_type(), shape_key)
        self._remove_layer_by_name(layer_name)
        gc.collect()

        gdf = self._active_sdata.shapes[shape_key]
        edge_color = np.asarray(
            stable_layer_color(shape_key, alpha=self.args.shape_opacity),
            dtype=float,
        )
        mode = str(render_mode or self.args.shape_render_mode).lower()

        if mode == "centroid":
            coords = geometry_to_napari_centroids(gdf.geometry, max_shapes=cap)
            if len(coords) == 0:
                log.info("[%s] shapes[%s] has no drawable centroids", ds, shape_key)
                return 0
            self._add_points_layer(
                coords,
                name=layer_name,
                color=edge_color,
                visible=not self.args.hide_shapes,
                size=self.args.shape_centroid_size,
                opacity=self.args.shape_opacity,
            )
            units = int(len(coords))
        else:
            if mode == "bbox":
                shape_data = geometry_to_napari_bounding_boxes(gdf.geometry, max_shapes=cap)
                shape_type = "rectangle"
            else:
                shape_data = geometry_to_napari_polygons(
                    gdf.geometry,
                    max_shapes=cap,
                    simplify_tolerance=simplify_tolerance,
                    max_vertices_per_polygon=self.args.shape_max_vertices_per_polygon,
                )
                shape_type = mode

            if len(shape_data) == 0:
                log.info("[%s] shapes[%s] has no drawable geometries", ds, shape_key)
                return 0

            kwargs = dict(
                shape_type=shape_type,
                name=layer_name,
                edge_color=edge_color,
                edge_width=self.args.shape_edge_width,
                visible=not self.args.hide_shapes,
            )
            if shape_type in ("polygon", "rectangle"):
                kwargs["face_color"] = "transparent"
            self.viewer.add_shapes(shape_data, **kwargs)
            units = int(len(shape_data))

        limit_note = f", capped at {cap:,}" if cap is not None else ""
        simplify_note = f", simplify={simplify_tolerance}" if simplify_tolerance else ""
        vertex_note = ""
        if mode in ("path", "polygon") and self.args.shape_max_vertices_per_polygon is not None:
            vertex_note = f", max_vertices={self.args.shape_max_vertices_per_polygon}"
        mode_note = f", mode={mode}"
        log.info(
            "[%s] Added shape layer %s with %s geometries%s%s%s%s",
            ds,
            shape_key,
            f"{units:,}",
            limit_note,
            simplify_note,
            vertex_note,
            mode_note,
        )
        return units

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

        napari_affine = self._napari_affine_from_element(label_elem)
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

        self.viewer.add_image(
            prepared["outline_data"],
            name=layer_name,
            affine=prepared["napari_affine"],
            colormap=color_map,
            contrast_limits=(0.0, 1.0),
            interpolation2d="nearest",
            multiscale=n_levels > 1,
            opacity=self.args.shape_opacity,
            blending="additive",
            visible=not self.args.hide_shapes,
        )
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

    def _add_label_layer(self, label_key: str, layer_dataset: str | None = None) -> int:
        if self._active_sdata is None or self.active_dataset is None:
            return 0
        ds = layer_dataset or self.active_dataset
        prepared = self._prepare_label_outline_display(label_key)
        return self._finish_label_layer(ds, label_key, prepared)

    def _add_shape_layers(self, ds: str, sdata) -> dict[str, int]:
        total_layers = 0
        total_units = 0

        del ds, sdata
        for shape_key in self._segmentation_keys:
            added = self._add_shape_layer(
                shape_key,
                cap=self.args.max_shapes_per_layer,
                simplify_tolerance=self.args.shape_simplify_tolerance,
            )
            if added > 0:
                total_layers += 1
                total_units += added

        return {"layers": total_layers, "units": total_units}

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

    def _transcript_coords_from_pdf(
        self,
        points_pdf,
        x_col: str,
        y_col: str,
        assignment_col: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points_pdf) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        if assignment_col is not None and assignment_col in points_pdf.columns:
            assigned_mask = assignment_mask(points_pdf[assignment_col]).to_numpy(dtype=bool, copy=False)
        else:
            assigned_mask = np.ones(len(points_pdf), dtype=bool)

        x_vals = points_pdf[x_col].to_numpy(dtype=np.float32, copy=False)
        y_vals = points_pdf[y_col].to_numpy(dtype=np.float32, copy=False)
        good = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.all(good):
            x_vals = x_vals[good]
            y_vals = y_vals[good]
            assigned_mask = assigned_mask[good]

        assigned_coords = np.column_stack([y_vals[assigned_mask], x_vals[assigned_mask]]).astype(
            np.float32,
            copy=False,
        )
        unassigned_coords = np.column_stack([y_vals[~assigned_mask], x_vals[~assigned_mask]]).astype(
            np.float32,
            copy=False,
        )
        return assigned_coords, unassigned_coords

    def _compute_transcript_view_payload(
        self,
        state: StreamedTranscriptLayerState,
        bounds: tuple[float, float, float, float] | None,
        view_percent: int,
    ) -> dict[str, object]:
        empty = np.empty((0, 2), dtype=np.float32)
        if state.point_index is None or bounds is None:
            return {
                "assigned_coords": empty,
                "unassigned_coords": empty,
                "total_in_view": 0,
                "loaded": 0,
                "assigned": 0,
                "unassigned": 0,
                "loaded_fraction": 0.0,
                "bounds": bounds,
                "view_percent": int(view_percent),
                "detail_enabled": False,
                "detail_reason": "density-only",
                "view_margin_fraction": 0.0,
            }

        view_width = float(bounds[2] - bounds[0])
        view_height = float(bounds[3] - bounds[1])
        if max(view_width, view_height) > float(self.args.transcript_detail_max_view_size):
            return {
                "assigned_coords": empty,
                "unassigned_coords": empty,
                "total_in_view": 0,
                "loaded": 0,
                "assigned": 0,
                "unassigned": 0,
                "loaded_fraction": 0.0,
                "bounds": bounds,
                "view_percent": int(view_percent),
                "detail_enabled": False,
                "detail_reason": "zoom-in",
                "view_size": max(view_width, view_height),
                "view_margin_fraction": 0.0,
            }

        query_bounds = self._expanded_view_bounds(bounds)
        sample_percent = None if int(view_percent) <= 0 else float(view_percent)
        result = query_transcript_spatial_index(
            state.point_index,
            bounds=query_bounds,
            max_points=self.args.transcript_detail_max_points,
            sample_percent=sample_percent,
            random_state=self.args.random_state,
        )
        assigned_coords = result["assigned_coords"]
        unassigned_coords = result["unassigned_coords"]
        loaded = int(result["loaded"])
        return {
            "assigned_coords": assigned_coords,
            "unassigned_coords": unassigned_coords,
            "total_in_view": int(result["total_in_view"]),
            "loaded": loaded,
            "assigned": int(len(assigned_coords)),
            "unassigned": int(len(unassigned_coords)),
            "loaded_fraction": float(result["loaded_fraction"]),
            "bounds": query_bounds,
            "viewport_bounds": bounds,
            "view_percent": int(view_percent),
            "detail_enabled": True,
            "detail_reason": "indexed",
            "view_margin_fraction": float(self.args.transcript_view_margin_fraction),
        }

    def _set_points_layer_data(self, layer_name: str, coords: np.ndarray, color, visible: bool):
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            self._add_points_layer(coords, name=layer_name, color=color, visible=visible)
            return
        layer.data = coords
        layer.visible = bool(visible)

    def _set_layers_visible(self, layer_names: list[str], visible: bool):
        for layer_name in layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                layer.visible = bool(visible)

    def _transcript_state_key(self, ds: str) -> str:
        return make_layer_name(ds, "transcripts", "viewport")

    def _schedule_streamed_transcript_update(self, state_key: str, delay_ms: int | None = None):
        state = self._streamed_transcript_states.get(state_key)
        if state is None:
            return
        if state.updating:
            state.pending = True
            return
        delay = int(self.args.transcript_stream_debounce_ms if delay_ms is None else delay_ms)
        state.timer.start(max(0, delay))

    def _apply_streamed_transcript_update(self, state_key: str, generation: int, payload: dict[str, object]):
        state = self._streamed_transcript_states.get(state_key)
        if state is None or state.generation != generation or self.active_dataset != state.dataset:
            return

        state.updating = False
        state.worker = None
        transcripts_visible = not self.args.hide_transcripts
        detail_enabled = bool(payload.get("detail_enabled", False))
        self._set_layers_visible(state.density_layer_names, transcripts_visible and not detail_enabled)
        self._set_points_layer_data(
            state.assigned_layer_name,
            payload["assigned_coords"],
            self.args.assigned_color,
            transcripts_visible and detail_enabled,
        )
        self._set_points_layer_data(
            state.unassigned_layer_name,
            payload["unassigned_coords"],
            self.args.unassigned_color,
            transcripts_visible and detail_enabled,
        )

        loaded = int(payload["loaded"])
        total_in_view = int(payload["total_in_view"])
        mode = "auto" if int(payload["view_percent"]) <= 0 else f"{int(payload['view_percent'])}%"
        if detail_enabled:
            fraction = float(payload["loaded_fraction"]) * 100.0
            margin = max(0.0, float(payload.get("view_margin_fraction", 0.0)))
            margin_note = f", {margin * 100.0:.0f}% pan buffer" if margin > 0 else ""
            cap_note = ""
            if total_in_view > int(self.args.transcript_detail_max_points) and loaded >= int(self.args.transcript_detail_max_points):
                cap_note = f", capped at {int(self.args.transcript_detail_max_points):,}"
            self._set_status(
                f"{state.dataset} transcript point detail ({mode}): showing {loaded:,}/{total_in_view:,} "
                f"indexed points ({fraction:.1f}%{cap_note}{margin_note}); assigned={int(payload['assigned']):,}, "
                f"unassigned={int(payload['unassigned']):,}."
            )
        else:
            reason = str(payload.get("detail_reason", "density-only"))
            if reason == "zoom-in":
                self._set_status(
                    f"{state.dataset} transcript density visible; zoom in below "
                    f"{float(self.args.transcript_detail_max_view_size):.0f} um for point detail."
                )
            else:
                self._set_status(f"{state.dataset} transcript density visible; point detail index unavailable.")

        if state.pending:
            state.pending = False
            self._schedule_streamed_transcript_update(state_key, delay_ms=0)

    def _handle_streamed_transcript_error(self, state_key: str, generation: int, exc):
        state = self._streamed_transcript_states.get(state_key)
        if state is None or state.generation != generation:
            return
        state.updating = False
        state.worker = None
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{state.dataset} transcript viewport update failed: {message}")
        log.error("[%s] Transcript viewport update failed: %s", state.dataset, message)
        if state.pending:
            state.pending = False
            self._schedule_streamed_transcript_update(state_key, delay_ms=0)

    def _update_streamed_transcript_layer(self, state_key: str):
        state = self._streamed_transcript_states.get(state_key)
        if state is None or self.active_dataset != state.dataset:
            return
        if state.updating:
            state.pending = True
            return

        bounds = self._current_view_bounds()
        state.generation += 1
        generation = state.generation
        state.updating = True
        state.pending = False
        view_percent = int(state.view_percent)
        mode = "auto" if view_percent <= 0 else f"{view_percent}%"
        self._set_status(f"{state.dataset} updating viewport transcripts ({mode})...")

        if thread_worker is None:
            try:
                payload = self._compute_transcript_view_payload(state, bounds, view_percent)
            except Exception as exc:
                self._handle_streamed_transcript_error(state_key, generation, exc)
                return
            self._apply_streamed_transcript_update(state_key, generation, payload)
            return

        def compute():
            return self._compute_transcript_view_payload(state, bounds, view_percent)

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, key=state_key, gen=generation: self._apply_streamed_transcript_update(key, gen, payload)
        )
        worker.errored.connect(
            lambda exc, key=state_key, gen=generation: self._handle_streamed_transcript_error(key, gen, exc)
        )
        state.worker = worker
        worker.start()

    def _ensure_transcript_density_cache(
        self,
        points_key: str,
        points_obj,
        x_col: str,
        y_col: str,
        assignment_col: str | None,
    ) -> tuple[str, dict[str, object]]:
        if self._active_sdata is None or self.active_dataset is None:
            raise RuntimeError("No active dataset.")

        requested_bin_um = float(self.args.transcript_density_bin_um)
        cache_key = derived_transcript_density_cache_key(points_key, requested_bin_um)
        expected = {
            "kind": "transcript_density",
            "points_key": str(points_key),
            "x_col": str(x_col),
            "y_col": str(y_col),
            "assignment_col": None if assignment_col is None else str(assignment_col),
            "requested_bin_um": float(requested_bin_um),
            "max_pixels": int(self.args.transcript_density_max_pixels),
            "pyramid_normalization": TRANSCRIPT_DENSITY_PYRAMID_NORMALIZATION,
        }
        if self._derived_cache_complete("images", cache_key, expected) and self._refresh_image_key_from_store(cache_key):
            return cache_key, self._derived_cache_attrs("images", cache_key)

        self._set_status(f"{self.active_dataset} building transcript density cache for {points_key}...")
        density, density_meta = compute_transcript_density_array(
            points_obj,
            x_col=x_col,
            y_col=y_col,
            assignment_col=assignment_col,
            bin_um=requested_bin_um,
            max_pixels=int(self.args.transcript_density_max_pixels),
        )
        levels = lazy_density_pyramid(density)
        if da is not None:
            levels = [
                da.asarray(level).rechunk(
                    (
                        1,
                        min(1024, int(level.shape[1])),
                        min(1024, int(level.shape[2])),
                    )
                )
                for level in levels
            ]
        actual_bin_um = float(density_meta["actual_bin_um"])
        min_x, min_y, _max_x, _max_y = (float(v) for v in density_meta["bounds"])
        transform = Affine(
            [
                [actual_bin_um, 0.0, min_x],
                [0.0, actual_bin_um, min_y],
                [0.0, 0.0, 1.0],
            ],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        density_tree = self._datatree_from_levels(
            levels,
            dims=("c", "y", "x"),
            channels=["assigned", "unassigned"],
            transform=transform,
            dtype=np.float32,
        )
        Image2DModel.validate(density_tree)
        self._discard_derived_cache_before_write("images", cache_key)
        self._active_sdata.images[cache_key] = density_tree
        self._set_status(f"{self.active_dataset} writing transcript density pyramid for {points_key}...")
        self._active_sdata.write_element(cache_key, overwrite=False)
        attrs = {
            **expected,
            **density_meta,
            "levels": int(len(levels)),
        }
        self._mark_derived_cache_complete("images", cache_key, attrs)
        self._refresh_image_key_from_store(cache_key)
        log.info(
            "[%s] Built transcript density cache images[%s] from points[%s] shape=%s levels=%s bin=%s requested_bin=%s",
            self.active_dataset,
            cache_key,
            points_key,
            density.shape,
            len(levels),
            actual_bin_um,
            requested_bin_um,
        )
        return cache_key, attrs

    def _add_transcript_density_layers(self, cache_key: str, attrs: dict[str, object]) -> list[str]:
        if self._active_sdata is None or self.active_dataset is None:
            return []
        if cache_key not in self._active_sdata.images and not self._refresh_image_key_from_store(cache_key):
            raise KeyError(f"Transcript density cache '{cache_key}' not found.")

        ds = self.active_dataset
        image_elem = self._active_sdata.images[cache_key]
        levels = self._raster_scale_levels(image_elem)
        if len(levels) == 0:
            raise ValueError(f"Transcript density cache {cache_key} has no readable levels.")

        scale_levels = []
        for scale_name, data in levels:
            if len(getattr(data, "shape", ())) != 3:
                continue
            scale_levels.append((scale_name, data))
        if len(scale_levels) == 0:
            raise ValueError(f"Transcript density cache {cache_key} has no 3D c/y/x levels.")

        napari_affine = self._napari_affine_from_element(image_elem)
        visible = not self.args.hide_transcripts
        layer_names: list[str] = []
        channel_specs = [
            ("assigned", 0, self.args.assigned_color),
            ("unassigned", 1, self.args.unassigned_color),
        ]
        for label, channel_idx, color in channel_specs:
            channel_levels = [data[channel_idx, :, :] for _scale_name, data in scale_levels]
            ch_data = channel_levels if len(channel_levels) > 1 else channel_levels[0]
            layer_name = make_layer_name(ds, "transcript_density", label)
            self._remove_layer_by_name(layer_name)
            self.viewer.add_image(
                ch_data,
                name=layer_name,
                affine=napari_affine,
                colormap=transparent_colormap(f"{cache_key}_{label}", color, alpha=1.0),
                contrast_limits=(0.0, 35.0),
                interpolation2d="nearest",
                multiscale=len(channel_levels) > 1,
                opacity=1.0,
                blending="additive",
                visible=visible,
            )
            layer_names.append(layer_name)
        return layer_names

    def _add_streamed_transcripts_layer(self, view_percent: int | None = None) -> dict[str, int | bool]:
        if self._active_sdata is None or self.active_dataset is None:
            return {"total": 0, "assigned": 0, "unassigned": 0, "streamed": True}

        ds = self.active_dataset
        points_key, points_obj, x_col, y_col, assignment_col = self._resolve_points_columns()
        # _clear_streamed_transcript_states() bumps the build generation, so
        # capture the new generation afterwards.
        self._clear_streamed_transcript_states()
        self._remove_layers_by_prefix(layer_name_prefix(ds, "transcripts"))
        self._remove_layers_by_prefix(layer_name_prefix(ds, "transcript_density"))
        gc.collect()

        resolved_view_percent = (
            int(self.args.transcript_view_percent) if view_percent is None else int(view_percent)
        )
        self._transcript_build_generation += 1
        generation = self._transcript_build_generation
        self._set_status(f"{ds} building transcript density + detail index for {points_key}...")

        def compute():
            # Heavy: builds/persists the density pyramid and the in-memory point
            # index. Runs on a worker thread; only pure data is returned so the
            # napari layer creation can happen back on the GUI thread.
            t0 = time.time()
            density_cache_key, density_attrs = self._ensure_transcript_density_cache(
                points_key,
                points_obj,
                x_col,
                y_col,
                assignment_col,
            )
            self._set_status(f"{ds} building transcript point detail index for {points_key}...")
            point_index = build_transcript_spatial_index(
                points_obj,
                x_col=x_col,
                y_col=y_col,
                assignment_col=assignment_col,
                max_points=int(self.args.transcript_index_max_points),
                tile_um=float(self.args.transcript_index_tile_um),
                random_state=int(self.args.random_state),
            )
            return {
                "points_key": points_key,
                "density_cache_key": density_cache_key,
                "density_attrs": density_attrs,
                "point_index": point_index,
                "view_percent": resolved_view_percent,
                "build_seconds": time.time() - t0,
            }

        if thread_worker is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_streamed_transcript_build_error(generation, ds, exc)
                return {"total": 0, "assigned": 0, "unassigned": 0, "streamed": True}
            self._apply_streamed_transcripts_build(generation, ds, payload)
            return {"total": 0, "assigned": 0, "unassigned": 0, "streamed": True}

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, gen=generation, d=ds: self._apply_streamed_transcripts_build(gen, d, payload)
        )
        worker.errored.connect(
            lambda exc, gen=generation, d=ds: self._handle_streamed_transcript_build_error(gen, d, exc)
        )
        self._transcript_build_worker = worker
        worker.start()
        return {"total": 0, "assigned": 0, "unassigned": 0, "streamed": True}

    def _apply_streamed_transcripts_build(self, generation: int, ds: str, payload: dict[str, object]):
        if generation != self._transcript_build_generation or self.active_dataset != ds:
            return
        self._transcript_build_worker = None

        points_key = str(payload["points_key"])
        density_cache_key = str(payload["density_cache_key"])
        point_index = payload["point_index"]
        density_layer_names = self._add_transcript_density_layers(
            density_cache_key,
            payload["density_attrs"],
        )

        assigned_layer_name = make_layer_name(ds, "transcripts", "assigned")
        unassigned_layer_name = make_layer_name(ds, "transcripts", "unassigned")
        visible = not self.args.hide_transcripts
        empty = np.empty((0, 2), dtype=np.float32)
        self._add_points_layer(
            empty,
            name=unassigned_layer_name,
            color=self.args.unassigned_color,
            visible=visible,
        )
        self._add_points_layer(
            empty,
            name=assigned_layer_name,
            color=self.args.assigned_color,
            visible=visible,
        )

        state_key = self._transcript_state_key(ds)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda key=state_key: self._update_streamed_transcript_layer(key))
        state = StreamedTranscriptLayerState(
            dataset=ds,
            points_key=points_key,
            assigned_layer_name=assigned_layer_name,
            unassigned_layer_name=unassigned_layer_name,
            timer=timer,
            view_percent=int(payload["view_percent"]),
            point_index=point_index,
            density_layer_names=density_layer_names,
        )

        def schedule(_event=None, key=state_key):
            self._schedule_streamed_transcript_update(key)

        state.camera_callback = schedule
        self._streamed_transcript_states[state_key] = state

        try:
            self.viewer.camera.events.zoom.connect(schedule)
            self.viewer.camera.events.center.connect(schedule)
        except Exception:
            pass

        self._schedule_streamed_transcript_update(state_key, delay_ms=0)
        log.info(
            "[%s] Added hybrid transcript layers from points[%s] density_cache=%s viewport_percent=%s "
            "detail_max_points=%s indexed=%s build=%.1fs",
            ds,
            points_key,
            density_cache_key,
            state.view_percent,
            self.args.transcript_detail_max_points,
            0 if point_index is None else point_index.indexed_rows,
            float(payload.get("build_seconds", 0.0)),
        )

    def _handle_streamed_transcript_build_error(self, generation: int, ds: str, exc):
        if generation != self._transcript_build_generation:
            return
        self._transcript_build_worker = None
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{ds} transcript build failed: {message}")
        log.error("[%s] Transcript density/index build failed: %s", ds, message)

    def _add_transcripts_layer(self, cap: int | None) -> dict[str, int]:
        if self._active_sdata is None or self.active_dataset is None:
            return {"total": 0, "assigned": 0, "unassigned": 0}

        ds = self.active_dataset
        sdata = self._active_sdata
        self._clear_streamed_transcript_states()
        self._remove_layers_by_prefix(layer_name_prefix(ds, "transcripts"))
        self._remove_layers_by_prefix(layer_name_prefix(ds, "transcript_density"))

        if len(sdata.points) == 0:
            log.warning("[%s] No points available in SpatialData.", ds)
            return {"total": 0, "assigned": 0, "unassigned": 0}

        points_key, points_obj, x_col, y_col, assignment_col = self._resolve_points_columns()

        points_pdf = load_points_dataframe(
            points_obj,
            x_col=x_col,
            y_col=y_col,
            assignment_col=assignment_col,
            max_points=cap,
            random_state=self.args.random_state,
        )

        if assignment_col is not None and assignment_col in points_pdf.columns:
            assigned_mask = assignment_mask(points_pdf[assignment_col]).to_numpy(dtype=bool, copy=False)
        else:
            assigned_mask = np.ones(len(points_pdf), dtype=bool)

        x_vals = points_pdf[x_col].to_numpy(dtype=np.float32, copy=False)
        y_vals = points_pdf[y_col].to_numpy(dtype=np.float32, copy=False)

        assigned_coords = np.column_stack([y_vals[assigned_mask], x_vals[assigned_mask]])
        unassigned_coords = np.column_stack([y_vals[~assigned_mask], x_vals[~assigned_mask]])

        visible = not self.args.hide_transcripts

        if len(unassigned_coords) > 0:
            self._add_points_layer(
                unassigned_coords,
                name=make_layer_name(ds, "transcripts", "unassigned"),
                color=self.args.unassigned_color,
                visible=visible,
            )

        if len(assigned_coords) > 0:
            self._add_points_layer(
                assigned_coords,
                name=make_layer_name(ds, "transcripts", "assigned"),
                color=self.args.assigned_color,
                visible=visible,
            )

        total = int(len(points_pdf))
        assigned = int(assigned_mask.sum())
        unassigned = total - assigned

        limit_note = ""
        if cap is not None:
            limit_note = f" (sampled/capped to {cap:,})"

        log.info(
            "[%s] Added transcripts from points[%s]: total=%s assigned=%s unassigned=%s%s",
            ds,
            points_key,
            f"{total:,}",
            f"{assigned:,}",
            f"{unassigned:,}",
            limit_note,
        )

        return {"total": total, "assigned": assigned, "unassigned": unassigned}

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
        self._set_status(
            f"{ds} building label outline layer(s) for {len(keys)} segmentation(s)..."
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
        added_layers = 0
        for spec in payload["specs"]:
            added = self._finish_label_layer(ds, str(spec["label_key"]), spec["prepared"])
            if added > 0:
                added_layers += 1
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
        message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
        self._set_status(f"{ds} label outline build failed: {message}")
        log.error("[%s] Label outline build failed: %s", ds, message)

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

    def _apply_cellpose_value_payload(self, generation: int, label_key: str, payload: dict[str, object]):
        if generation != self._cellpose_value_generation or self.active_dataset != "MERSCOPE":
            return
        self._cellpose_value_worker = None
        if self._active_sdata is None:
            return

        if label_key not in self._active_sdata.labels and not self._refresh_label_key_from_store(label_key):
            self._set_status(f"MERSCOPE Cellpose value overlay failed: label key {label_key!r} not found.")
            return

        label_elem = self._active_sdata.labels[label_key]
        label_scale_levels = [
            (scale_name, level_data)
            for scale_name, level_data in self._raster_scale_levels(label_elem)
            if len(getattr(level_data, "shape", ())) == 2
        ]
        if len(label_scale_levels) == 0:
            self._set_status(f"MERSCOPE Cellpose value overlay failed: labels[{label_key}] has no 2D levels.")
            return

        label_levels = [level_data for _scale_name, level_data in label_scale_levels]
        if len(label_levels) == 1:
            label_levels = lazy_label_pyramid(label_levels[0])
        label_data = label_levels if len(label_levels) > 1 else label_levels[0]

        ds = self.active_dataset
        channel = str(payload["channel"])
        statistic = str(payload["statistic"])
        layer_name = make_layer_name(ds, "cell_values", CELLPOSE_SHAPE_KEY, f"{channel} {statistic}")
        self._remove_layers_by_prefix(layer_name_prefix(ds, "cell_values"))
        self.viewer.add_labels(
            label_data,
            name=layer_name,
            affine=self._napari_affine_from_element(label_elem),
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

        try:
            label_key = self._ensure_cellpose_label_key()
        except Exception as exc:
            self._set_status(f"Could not prepare {CELLPOSE_LABEL_KEY}: {exc}")
            log.exception("[MERSCOPE] Could not prepare Cellpose labels")
            return

        self._cellpose_value_generation += 1
        generation = self._cellpose_value_generation
        self._set_status(f"MERSCOPE building Cellpose value overlay: {channel} {statistic}...")

        if thread_worker is None:
            try:
                payload = self._compute_cellpose_value_payload(cfg.zarr_path, channel, statistic, colormap_name)
            except Exception as exc:
                self._handle_cellpose_value_error(generation, exc)
                return
            self._apply_cellpose_value_payload(generation, label_key, payload)
            return

        def compute():
            return self._compute_cellpose_value_payload(cfg.zarr_path, channel, statistic, colormap_name)

        worker_factory = thread_worker(compute)
        worker = worker_factory()
        worker.returned.connect(
            lambda payload, gen=generation, key=label_key: self._apply_cellpose_value_payload(gen, key, payload)
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

    def _get_canvas_size(self) -> tuple[int, int] | None:
        """Return the canvas size in pixels, robust across napari versions.

        napari has no stable public canvas-size API. We try the private
        ``_qt_viewer`` first (the real attribute) so the common path never
        touches the deprecated ``qt_viewer`` property, fall back to it if
        needed, and cache the last good value so a future napari that renames or
        removes either accessor keeps working.
        """
        window = getattr(self.viewer, "window", None)
        if window is not None:
            for attr in ("_qt_viewer", "qt_viewer"):
                try:
                    qt_viewer = getattr(window, attr, None)
                except Exception:
                    qt_viewer = None
                canvas = getattr(qt_viewer, "canvas", None) if qt_viewer is not None else None
                if canvas is None:
                    continue
                try:
                    size = tuple(int(v) for v in canvas.size)
                except Exception:
                    continue
                if len(size) >= 2 and size[0] > 0 and size[1] > 0:
                    self._canvas_size = (int(size[0]), int(size[1]))
                    return self._canvas_size
        return self._canvas_size

    def _current_view_bounds(self) -> tuple[float, float, float, float] | None:
        try:
            center = tuple(float(v) for v in self.viewer.camera.center)
            zoom = float(self.viewer.camera.zoom)
        except Exception:
            return None

        size = self._get_canvas_size()
        if size is None or len(center) < 2 or len(size) < 2 or zoom <= 0:
            return None

        y_center = center[-2]
        x_center = center[-1]
        width_px, height_px = size[0], size[1]
        half_x = max(width_px / (2.0 * zoom), 1.0)
        half_y = max(height_px / (2.0 * zoom), 1.0)
        return (x_center - half_x, y_center - half_y, x_center + half_x, y_center + half_y)

    def _expanded_view_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        margin = max(0.0, float(self.args.transcript_view_margin_fraction))
        min_x, min_y, max_x, max_y = (float(v) for v in bounds)
        pad_x = max(0.0, max_x - min_x) * margin
        pad_y = max(0.0, max_y - min_y) * margin
        return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

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
        self._set_status(f"{ds} removed {removed} {self._segmentation_layer_type()} layer(s).")

    def set_transcript_view_percent(self, dataset_name: str, percent: int):
        percent = int(percent)
        self.args.transcript_view_percent = percent
        ds = str(dataset_name).upper()
        state = self._streamed_transcript_states.get(self._transcript_state_key(ds))
        if state is None:
            return
        state.view_percent = percent
        if self.active_dataset == ds:
            self._schedule_streamed_transcript_update(self._transcript_state_key(ds), delay_ms=0)

    def load_full_transcripts(self, dataset_name: str, view_percent: int | None = None):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        if view_percent is not None:
            self.args.transcript_view_percent = int(view_percent)
        # The build now runs on a worker thread; status is driven by the build /
        # apply callbacks, so we intentionally do not overwrite it here.
        self._add_streamed_transcripts_layer(view_percent=view_percent)

    def load_images_on_demand(self, dataset_name: str):
        if self.args.skip_images:
            self._set_status("Image loading disabled (--skip-images).")
            return
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return

        ds = str(dataset_name).upper()
        self._ensure_images_loaded(ds)
        if self._x_transform is None or self._y_transform is None:
            raise RuntimeError("Image transform is not initialized.")

        self._remove_layers_by_prefix(layer_name_prefix(ds, "image"))
        stats = self._add_image_layers(ds, self._active_images_sdata, self._x_transform, self._y_transform)
        self._set_status(f"{ds} loaded image layers={stats['layers']} (failed={stats['failed_keys']}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
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
        "--startup-mode",
        default="fast",
        choices=["fast", "full"],
        help="Startup behavior: fast (on-demand layers) or full (eager legacy-style loading).",
    )
    parser.add_argument(
        "--segmentation-source",
        default="shapes",
        choices=["shapes", "labels"],
        help="Load segmentation layers from shapes (vector) or labels (raster masks).",
    )
    parser.add_argument(
        "--max-transcripts",
        default=None,
        type=int,
        help="Optional cap/sampling limit for transcripts per dataset load.",
    )
    parser.add_argument(
        "--transcript-view-percent",
        default=DEFAULT_TRANSCRIPT_VIEW_PERCENT,
        type=int,
        help=(
            "Percentage of transcripts in the current viewport to show in streamed full-transcript mode. "
            "Use 0 for auto, which loads up to --transcript-detail-max-points."
        ),
    )
    parser.add_argument(
        "--transcript-view-margin-fraction",
        default=DEFAULT_TRANSCRIPT_VIEW_MARGIN_FRACTION,
        type=float,
        help=(
            "Fraction of the current viewport width/height to add on each side when querying "
            "zoomed-in transcript point detail."
        ),
    )
    parser.add_argument(
        "--transcript-max-view-points",
        default=None,
        type=int,
        help="Deprecated alias for --transcript-detail-max-points.",
    )
    parser.add_argument(
        "--transcript-stream-debounce-ms",
        default=DEFAULT_TRANSCRIPT_STREAM_DEBOUNCE_MS,
        type=int,
        help="Delay after pan/zoom before refreshing viewport-streamed transcript layers.",
    )
    parser.add_argument(
        "--transcript-density-bin-um",
        default=DEFAULT_TRANSCRIPT_DENSITY_BIN_UM,
        type=float,
        help="Requested transcript density bin size in microns.",
    )
    parser.add_argument(
        "--transcript-density-max-pixels",
        default=DEFAULT_TRANSCRIPT_DENSITY_MAX_PIXELS,
        type=int,
        help="Maximum pixels per transcript density channel; bin size increases automatically above this.",
    )
    parser.add_argument(
        "--transcript-detail-max-view-size",
        default=DEFAULT_TRANSCRIPT_DETAIL_MAX_VIEW_SIZE,
        type=float,
        help="Maximum viewport width/height in microns before transcript point detail is shown.",
    )
    parser.add_argument(
        "--transcript-detail-max-points",
        default=None,
        type=int,
        help="Maximum transcript points displayed in the zoomed-in detail overlay.",
    )
    parser.add_argument(
        "--transcript-index-max-points",
        default=DEFAULT_TRANSCRIPT_INDEX_MAX_POINTS,
        type=int,
        help="Maximum transcript points loaded into the in-memory detail index.",
    )
    parser.add_argument(
        "--transcript-index-tile-um",
        default=DEFAULT_TRANSCRIPT_INDEX_TILE_UM,
        type=float,
        help="Tile size in microns for the in-memory transcript detail index.",
    )
    parser.add_argument(
        "--max-shapes-per-layer",
        default=None,
        type=int,
        help="Optional cap for polygons per shape layer.",
    )
    parser.add_argument(
        "--shape-simplify-tolerance",
        default=0.0,
        type=float,
        help="Optional simplify tolerance for polygons before adding shape layers (0 disables simplification).",
    )
    parser.add_argument(
        "--shape-render-mode",
        default="path",
        choices=["path", "polygon", "bbox", "centroid"],
        help="Render capped segmentation loads as path, polygon, bbox, or centroid.",
    )
    parser.add_argument(
        "--shape-max-vertices-per-polygon",
        default=None,
        type=int,
        help=(
            "Optional vertex cap per polygon for path/polygon rendering. "
            "Useful for loading large uncapped segmentation layers approximately."
        ),
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
        "--overwrite-labels",
        action="store_true",
        help="Rebuild cached label elements even if labels with the same keys already exist.",
    )
    parser.add_argument(
        "--overwrite-derived-caches",
        action="store_true",
        help="Rebuild viewer-derived density/outline caches even when matching cache metadata exists.",
    )
    parser.add_argument("--random-state", default=42, type=int, help="Random seed used for sampling")

    parser.add_argument("--point-size", default=2.0, type=float, help="Napari point size for transcript layers")
    parser.add_argument("--point-opacity", default=0.50, type=float, help="Napari point opacity")
    parser.add_argument("--shape-edge-width", default=0.75, type=float, help="Shape layer edge width")
    parser.add_argument("--shape-opacity", default=0.95, type=float, help="Shape layer edge alpha")
    parser.add_argument("--shape-centroid-size", default=1.0, type=float, help="Point size for centroid segmentation rendering")
    parser.add_argument("--image-opacity", default=1.0, type=float, help="Image layer opacity")
    parser.add_argument("--assigned-color", default="yellow", help="Color for assigned transcript points")
    parser.add_argument("--unassigned-color", default="#d62728", help="Color for unassigned transcript points")

    parser.add_argument("--hide-images", action="store_true", help="Start with image layers hidden")
    parser.add_argument("--hide-shapes", action="store_true", help="Start with shape layers hidden")
    parser.add_argument("--hide-transcripts", action="store_true", help="Start with transcript layers hidden")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not attempt to load image layers (useful for image-less zarr transfers).",
    )

    args = parser.parse_args()
    if args.merscope_zarr is None and args.xenium_zarr is None:
        parser.error("Pass at least one dataset path: --merscope-zarr and/or --xenium-zarr.")
    if args.max_transcripts is None:
        args.max_transcripts = FAST_DEFAULT_MAX_TRANSCRIPTS
    if args.max_transcripts <= 0:
        parser.error("--max-transcripts must be positive.")
    if not 0 <= args.transcript_view_percent <= 100:
        parser.error("--transcript-view-percent must be between 0 and 100.")
    if args.transcript_view_margin_fraction < 0:
        parser.error("--transcript-view-margin-fraction must be non-negative.")
    if args.transcript_detail_max_points is None:
        args.transcript_detail_max_points = (
            args.transcript_max_view_points
            if args.transcript_max_view_points is not None
            else DEFAULT_TRANSCRIPT_DETAIL_MAX_POINTS
        )
    args.transcript_max_view_points = args.transcript_detail_max_points
    if args.transcript_detail_max_points <= 0:
        parser.error("--transcript-detail-max-points must be positive.")
    if args.transcript_stream_debounce_ms < 0:
        parser.error("--transcript-stream-debounce-ms must be non-negative.")
    if args.transcript_density_bin_um <= 0:
        parser.error("--transcript-density-bin-um must be positive.")
    if args.transcript_density_max_pixels <= 0:
        parser.error("--transcript-density-max-pixels must be positive.")
    if args.transcript_detail_max_view_size <= 0:
        parser.error("--transcript-detail-max-view-size must be positive.")
    if args.transcript_index_max_points <= 0:
        parser.error("--transcript-index-max-points must be positive.")
    if args.transcript_index_tile_um <= 0:
        parser.error("--transcript-index-tile-um must be positive.")
    if args.max_shapes_per_layer is None:
        args.max_shapes_per_layer = FAST_DEFAULT_MAX_SHAPES_PER_LAYER
    if args.label_chunk_size <= 0:
        parser.error("--label-chunk-size must be positive.")
    if args.label_contour_width < 0:
        parser.error("--label-contour-width must be non-negative.")
    return args


def main():
    args = parse_args()

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
    initial_dataset = args.initial_dataset
    if initial_dataset not in datasets:
        initial_dataset = available_datasets[0]
        log.info(
            "Initial dataset %s was not supplied; starting with %s.",
            args.initial_dataset,
            initial_dataset,
        )

    for cfg in datasets.values():
        validate_spatialdata_store_compatibility(cfg.zarr_path)

    viewer = napari.Viewer(title="Xenium vs MERSCOPE Comparison")
    controller = ComparisonViewerController(viewer=viewer, datasets=datasets, args=args)

    switcher = DatasetSwitcherWidget(
        datasets=available_datasets,
        load_callback=controller.load_dataset,
        load_selected_labels_callback=controller.load_selected_labels,
        unload_selected_shapes_callback=controller.unload_selected_shapes,
        load_full_transcripts_callback=controller.load_full_transcripts,
        set_transcript_view_percent_callback=controller.set_transcript_view_percent,
        load_images_callback=controller.load_images_on_demand,
        load_cellpose_values_callback=controller.load_cellpose_value_overlay,
        remove_cellpose_values_callback=controller.remove_cellpose_value_overlay,
        create_annotation_layers_callback=controller.create_cortical_depth_annotation_layers,
        set_annotation_piece_callback=controller.set_cortical_depth_current_piece,
        apply_annotation_piece_callback=controller.apply_cortical_depth_piece_to_selection,
        snap_annotation_side_edges_callback=controller.snap_cortical_depth_side_edges,
        validate_annotation_callback=controller.validate_cortical_depth_annotations,
        export_annotation_callback=controller.export_cortical_depth_annotations,
        export_separate_annotations_callback=controller.export_separate_cortical_depth_annotations,
        initial_dataset=initial_dataset,
        skip_images=args.skip_images,
        transcript_view_percent=args.transcript_view_percent,
    )
    controller.set_status_callback(switcher.set_status)
    controller.set_shape_keys_callback(switcher.set_shape_keys)
    controller.set_cellpose_value_options_callback(switcher.set_cellpose_value_options)
    viewer.window.add_dock_widget(switcher, area="right", name="Dataset Switcher")

    controller.load_dataset(initial_dataset, force=True)
    napari.run()


if __name__ == "__main__":
    main()
