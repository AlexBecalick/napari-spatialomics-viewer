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
from napari.utils.colormaps import Colormap
from packaging.version import InvalidVersion, Version
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Affine
from spatialdata.transformations import get_transformation

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
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QLabel,
        QListWidget,
        QPushButton,
        QScrollArea,
        QSizePolicy,
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

from .utils import (
    assignment_mask,
    affine_matrix_from_px_to_um,
    build_napari_affine_from_px_to_um,
    channel_labels,
    ensure_cyx,
    first_existing_col,
    geometry_to_napari_bounding_boxes,
    geometry_to_napari_centroids,
    geometry_to_napari_polygons,
    get_scale0_dataarray,
    image_scale_dataarrays,
    label_outline_mask_chunk,
    layer_name_prefix,
    load_points_dataframe,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    query_geometries_for_bounds,
    rasterize_geometries_chunk,
    resolve_dataset_mask_affine,
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


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    zarr_path: Path
    merscope_transform_path: Path | None = None
    xenium_spec_path: Path | None = None


@dataclass
class StreamedShapeLayerState:
    dataset: str
    shape_key: str
    gdf: object
    detail_layer_name: str
    overview_layer_name: str
    timer: QTimer
    edge_color: np.ndarray


FAST_DEFAULT_MAX_TRANSCRIPTS = 200_000
FAST_DEFAULT_MAX_SHAPES_PER_LAYER = 20_000
SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE = 4096
SYNTHETIC_IMAGE_PYRAMID_MAX_LEVELS = 10
LABEL_CACHE_ATTR = "napari_compare_label_cache"
LABEL_OUTLINE_PYRAMID_MIN_SIZE = 4096
LABEL_OUTLINE_PYRAMID_MAX_LEVELS = 10


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

    if da is not None:
        labels = da.asarray(label_data)
        outline = labels.map_overlap(
            label_outline_mask_chunk,
            depth=max(1, width),
            boundary=0,
            trim=True,
            dtype=np.uint8,
            width=width,
        )
        levels: list[object] = [outline]
        data = outline
        while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
            data = da.coarsen(np.max, data, axes={0: 2, 1: 2}, trim_excess=True)
            if data.shape == levels[-1].shape:
                break
            levels.append(data.astype(np.uint8))
        return levels

    outline = label_outline_mask_chunk(label_data, width=width)
    levels = [outline]
    data = outline
    while len(levels) < max_levels and max(int(axis) for axis in data.shape) > min_size:
        y = (data.shape[0] // 2) * 2
        x = (data.shape[1] // 2) * 2
        if y < 2 or x < 2:
            break
        data = data[:y, :x].reshape(y // 2, 2, x // 2, 2).max(axis=(1, 3)).astype(np.uint8)
        if data.shape == levels[-1].shape:
            break
        levels.append(data)
    return levels


class DatasetSwitcherWidget(QWidget):
    """Dock widget with dataset controls and on-demand layer loading controls."""

    def __init__(
        self,
        datasets: list[str],
        load_callback,
        load_selected_shapes_callback,
        load_selected_full_shapes_callback,
        stream_selected_shapes_callback,
        load_selected_labels_callback,
        unload_selected_shapes_callback,
        load_all_shapes_callback,
        load_transcripts_callback,
        load_images_callback,
        initial_dataset: str = "MERSCOPE",
        skip_images: bool = False,
        segmentation_source: str = "shapes",
        full_shape_render_mode: str = "bbox",
    ):
        super().__init__()
        self._load_callback = load_callback
        self._load_selected_shapes_callback = load_selected_shapes_callback
        self._load_selected_full_shapes_callback = load_selected_full_shapes_callback
        self._stream_selected_shapes_callback = stream_selected_shapes_callback
        self._load_selected_labels_callback = load_selected_labels_callback
        self._unload_selected_shapes_callback = unload_selected_shapes_callback
        self._load_all_shapes_callback = load_all_shapes_callback
        self._load_transcripts_callback = load_transcripts_callback
        self._load_images_callback = load_images_callback
        self._skip_images = bool(skip_images)
        self._segmentation_source = "labels" if str(segmentation_source).lower() == "labels" else "shapes"
        self._full_shape_render_mode = str(full_shape_render_mode).lower()

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

        selected_label = (
            "Load Selected Segmentations"
            if self._segmentation_source == "labels"
            else "Load Selected Segmentations (Capped)"
        )
        self._load_selected_shapes_button = QPushButton(selected_label)
        self._load_selected_shapes_button.clicked.connect(self._on_load_selected_shapes)
        self._load_selected_full_shapes_button = QPushButton(
            f"Load Selected Segmentations (All: {self._full_shape_render_mode})"
        )
        self._load_selected_full_shapes_button.setEnabled(self._segmentation_source != "labels")
        self._load_selected_full_shapes_button.clicked.connect(self._on_load_selected_full_shapes)
        self._stream_selected_shapes_button = QPushButton("Stream Selected Polygons")
        self._stream_selected_shapes_button.setEnabled(self._segmentation_source != "labels")
        self._stream_selected_shapes_button.clicked.connect(self._on_stream_selected_shapes)
        self._load_selected_labels_button = QPushButton("Load Selected Labels (Outlines)")
        self._load_selected_labels_button.clicked.connect(self._on_load_selected_labels)
        self._unload_selected_shapes_button = QPushButton("Unload Selected Segmentations")
        self._unload_selected_shapes_button.clicked.connect(self._on_unload_selected_shapes)
        load_all_label = (
            "Load All Segmentations"
            if self._segmentation_source == "labels"
            else "Load All Segmentations (Capped)"
        )
        self._load_all_shapes_button = QPushButton(load_all_label)
        self._load_all_shapes_button.clicked.connect(self._on_load_all_shapes)

        self._load_sampled_tx_button = QPushButton("Load Sampled Transcripts")
        self._load_sampled_tx_button.clicked.connect(self._on_load_sampled_transcripts)
        self._load_full_tx_button = QPushButton("Load Full Transcripts")
        self._load_full_tx_button.clicked.connect(self._on_load_full_transcripts)

        self._load_images_button = QPushButton("Load Images")
        self._load_images_button.setEnabled(not self._skip_images)
        self._load_images_button.clicked.connect(self._on_load_images)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)

        content = QWidget()
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(QLabel("Dataset"))
        root.addWidget(self._dataset_combo)
        root.addWidget(self._reload_button)
        root.addWidget(QLabel("Segmentations"))
        root.addWidget(self._shape_list)

        root.addWidget(self._load_selected_shapes_button)
        if self._segmentation_source != "labels":
            root.addWidget(self._load_selected_full_shapes_button)
            root.addWidget(self._stream_selected_shapes_button)
        root.addWidget(self._unload_selected_shapes_button)
        root.addWidget(self._load_selected_labels_button)
        root.addWidget(self._load_all_shapes_button)

        root.addWidget(QLabel("Transcripts"))
        root.addWidget(self._load_sampled_tx_button)
        root.addWidget(self._load_full_tx_button)

        if not self._skip_images:
            root.addWidget(self._load_images_button)

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
        self._status_label.setText(str(text))

    def set_shape_keys(self, keys: list[str]):
        self._shape_list.clear()
        self._shape_list.addItems([str(k) for k in keys])

    def selected_shape_keys(self) -> list[str]:
        return [str(item.text()) for item in self._shape_list.selectedItems()]

    def _on_dataset_changed(self, text: str):
        if not text:
            return
        self._load_callback(str(text), False)

    def _on_reload_clicked(self):
        self._load_callback(self.current_dataset, True)

    def _on_load_selected_shapes(self):
        keys = self.selected_shape_keys()
        if not keys:
            self.set_status("No segmentation selected.")
            return
        self._load_selected_shapes_callback(self.current_dataset, keys)

    def _on_load_selected_full_shapes(self):
        keys = self.selected_shape_keys()
        if not keys:
            self.set_status("No segmentation selected.")
            return
        self._load_selected_full_shapes_callback(self.current_dataset, keys)

    def _on_stream_selected_shapes(self):
        keys = self.selected_shape_keys()
        if not keys:
            self.set_status("No segmentation selected.")
            return
        self._stream_selected_shapes_callback(self.current_dataset, keys)

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

    def _on_load_all_shapes(self):
        self._load_all_shapes_callback(self.current_dataset)

    def _on_load_sampled_transcripts(self):
        self._load_transcripts_callback(self.current_dataset, False)

    def _on_load_full_transcripts(self):
        self._load_transcripts_callback(self.current_dataset, True)

    def _on_load_images(self):
        self._load_images_callback(self.current_dataset)


class ComparisonViewerController:
    """Coordinate loading/clearing napari layers for each dataset."""

    def __init__(self, viewer: napari.Viewer, datasets: dict[str, DatasetConfig], args):
        self.viewer = viewer
        self.datasets = datasets
        self.args = args
        self.active_dataset: str | None = None
        self._status_callback = None
        self._shape_keys_callback = None
        self._active_sdata = None
        self._active_images_sdata = None
        self._segmentation_keys: list[str] = []
        self._x_transform: tuple[float, float, float] | None = None
        self._y_transform: tuple[float, float, float] | None = None
        self._streamed_shape_states: dict[str, StreamedShapeLayerState] = {}

    def set_status_callback(self, fn):
        self._status_callback = fn

    def set_shape_keys_callback(self, fn):
        self._shape_keys_callback = fn

    def _set_status(self, text: str):
        if self._status_callback is not None:
            self._status_callback(text)

    def _publish_shape_keys(self):
        if self._shape_keys_callback is not None:
            self._shape_keys_callback(list(self._segmentation_keys))

    def _segmentation_source(self) -> str:
        return "labels" if str(self.args.segmentation_source).lower() == "labels" else "shapes"

    def _segmentation_layer_type(self) -> str:
        return "labels" if self._segmentation_source() == "labels" else "shapes"

    def _segmentation_unit_name(self) -> str:
        return "labels" if self._segmentation_source() == "labels" else "polygons"

    def _clear_layers(self):
        self._clear_streamed_shape_states()
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)

    def _clear_streamed_shape_states(self):
        for state in list(self._streamed_shape_states.values()):
            try:
                state.timer.stop()
            except Exception:
                pass
        self._streamed_shape_states.clear()

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
            self.viewer.add_points(
                coords,
                name=name,
                face_color=color,
                edge_color=color,
                **common_kwargs,
            )
        except TypeError:
            # Newer napari versions renamed edge_color -> border_color.
            self.viewer.add_points(
                coords,
                name=name,
                face_color=color,
                border_color=color,
                **common_kwargs,
            )

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
                self._segmentation_keys = sorted(str(k) for k in sdata.labels.keys())
                if len(self._segmentation_keys) == 0:
                    raise RuntimeError(
                        "No labels found in this store. Generate label layers first "
                        "or use `--segmentation-source shapes`."
                    )
            else:
                self._segmentation_keys = sorted(str(k) for k in sdata.shapes.keys())
            self._publish_shape_keys()

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

    def _add_image_layers(self, ds: str, sdata, x_transform, y_transform) -> dict[str, int]:
        if getattr(self.args, "skip_images", False):
            log.info("[%s] Image loading skipped (--skip-images).", ds)
            return {"layers": 0, "failed_keys": 0, "skipped": True}

        visible = not self.args.hide_images
        total_layers = 0
        failed_keys = 0

        try:
            image_keys = list(sdata.images.keys())
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

    def _refresh_label_key_from_store(self, label_key: str) -> bool:
        if self._active_sdata is None:
            return False
        labels_sdata = self._read_labels_from_store()
        if labels_sdata is None or label_key not in labels_sdata.labels:
            return False
        self._active_sdata.labels[label_key] = labels_sdata.labels[label_key]
        return True

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
        self._write_empty_label_element(
            label_key=label_key,
            shape=shape,
            chunks=chunks,
            transform=spatialdata_affine,
            overwrite=True,
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

    def _add_label_layer(self, label_key: str, layer_dataset: str | None = None) -> int:
        if self._active_sdata is None or self.active_dataset is None:
            return 0
        ds = layer_dataset or self.active_dataset
        if label_key not in self._active_sdata.labels:
            if not self._refresh_label_key_from_store(label_key):
                raise KeyError(f"Label key '{label_key}' not found in current dataset.")

        label_elem = self._active_sdata.labels[label_key]
        label_da = get_scale0_dataarray(label_elem)
        data = label_da.data if hasattr(label_da, "data") else label_da
        if len(getattr(data, "shape", ())) != 2:
            raise ValueError(f"Expected 2D labels for {label_key}, got shape {getattr(data, 'shape', None)}")

        tf = get_transformation(label_elem, to_coordinate_system="global")
        m = tf.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        # Convert x/y affine to napari row/col affine.
        napari_affine = np.array(
            [
                [float(m[1, 1]), float(m[1, 0]), float(m[1, 2])],
                [float(m[0, 1]), float(m[0, 0]), float(m[0, 2])],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        layer_name = make_layer_name(ds, "labels", label_key)
        self._remove_layer_by_name(layer_name)
        outline_width = max(1, int(self.args.label_contour_width))
        outline_levels = lazy_outline_pyramid(data, width=outline_width)
        outline_data = outline_levels if len(outline_levels) > 1 else outline_levels[0]
        layer_color = np.asarray(stable_layer_color(label_key, alpha=float(self.args.shape_opacity)), dtype=np.float32)
        color_map = Colormap(
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    layer_color,
                ],
                dtype=np.float32,
            ),
            name=f"{label_key}_outline",
            controls=np.asarray([0.0, 1.0], dtype=np.float32),
        )

        self.viewer.add_image(
            outline_data,
            name=layer_name,
            affine=napari_affine,
            colormap=color_map,
            contrast_limits=(0.0, 1.0),
            interpolation2d="nearest",
            multiscale=len(outline_levels) > 1,
            opacity=self.args.shape_opacity,
            blending="additive",
            visible=not self.args.hide_shapes,
        )
        log.info(
            "[%s] Added label outline layer %s with shape=%s dtype=%s levels=%s width=%s",
            ds,
            label_key,
            tuple(int(x) for x in data.shape),
            getattr(data, "dtype", "unknown"),
            len(outline_levels),
            outline_width,
        )
        return 1

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

    def _add_transcripts_layer(self, cap: int | None) -> dict[str, int]:
        if self._active_sdata is None or self.active_dataset is None:
            return {"total": 0, "assigned": 0, "unassigned": 0}

        ds = self.active_dataset
        sdata = self._active_sdata
        self._remove_layers_by_prefix(layer_name_prefix(ds, "transcripts"))

        if len(sdata.points) == 0:
            log.warning("[%s] No points available in SpatialData.", ds)
            return {"total": 0, "assigned": 0, "unassigned": 0}

        points_key = list(sdata.points.keys())[0]
        points_obj = sdata.points[points_key]

        x_col = first_existing_col(points_obj, ["x", "x_micron", "global_x", "x_location", "observed_x"])
        y_col = first_existing_col(points_obj, ["y", "y_micron", "global_y", "y_location", "observed_y"])
        assignment_col = first_existing_col(points_obj, ["assignment", "cell", "cell_id"])

        if x_col is None or y_col is None:
            raise KeyError(f"Could not resolve x/y columns in points[{points_key}]")

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

    def _load_shape_keys(
        self,
        dataset_name: str,
        shape_keys: list[str],
        cap: int | None,
        mode: str,
        render_mode: str | None = None,
    ):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        added_layers = 0
        total_units = 0
        mem_before = memory_snapshot_gb()
        for shape_key in shape_keys:
            n = self._add_shape_layer(
                shape_key,
                cap=cap,
                simplify_tolerance=self.args.shape_simplify_tolerance,
                render_mode=render_mode,
            )
            if n > 0:
                added_layers += 1
                total_units += n
        mem_after = memory_snapshot_gb()
        self._set_status(
            f"{self.active_dataset} loaded {mode} {self._segmentation_layer_type()} "
            f"layer(s)={added_layers}, {self._segmentation_unit_name()}={total_units:,} | "
            f"RSS {mem_before['rss_gb']:.1f}->{mem_after['rss_gb']:.1f} GB"
        )

    def load_selected_shapes(self, dataset_name: str, shape_keys: list[str]):
        self._load_shape_keys(
            dataset_name,
            shape_keys,
            cap=self.args.max_shapes_per_layer,
            mode="capped selected",
            render_mode=self.args.shape_render_mode,
        )

    def load_selected_shapes_full(self, dataset_name: str, shape_keys: list[str]):
        self._load_shape_keys(
            dataset_name,
            shape_keys,
            cap=None,
            mode=f"full selected ({self.args.full_shape_render_mode})",
            render_mode=self.args.full_shape_render_mode,
        )

    def load_selected_labels(self, dataset_name: str, shape_keys: list[str]):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return

        added_layers = 0
        total_labels = 0
        mem_before = memory_snapshot_gb()
        for key in shape_keys:
            if self._segmentation_source() == "labels" and key in self._active_sdata.labels:
                label_key = key
            else:
                label_key = self.ensure_label_for_shape_key(key)
            added = self._add_label_layer(label_key)
            if added > 0:
                added_layers += 1
                try:
                    total_labels += int(len(self._active_sdata.shapes[key]))
                except Exception:
                    total_labels += 1

        mem_after = memory_snapshot_gb()
        self._set_status(
            f"{self.active_dataset} loaded label outline layer(s)={added_layers}, "
            f"labels={total_labels:,} | RSS {mem_before['rss_gb']:.1f}->{mem_after['rss_gb']:.1f} GB"
        )

    def _current_view_bounds(self) -> tuple[float, float, float, float] | None:
        try:
            center = tuple(float(v) for v in self.viewer.camera.center)
            zoom = float(self.viewer.camera.zoom)
            canvas = self.viewer.window.qt_viewer.canvas
            size = tuple(int(v) for v in canvas.size)
        except Exception:
            return None

        if len(center) < 2 or len(size) < 2 or zoom <= 0:
            return None

        y_center = center[-2]
        x_center = center[-1]
        width_px, height_px = size[0], size[1]
        half_x = max(width_px / (2.0 * zoom), 1.0)
        half_y = max(height_px / (2.0 * zoom), 1.0)
        return (x_center - half_x, y_center - half_y, x_center + half_x, y_center + half_y)

    def _streamed_state_key(self, ds: str, shape_key: str) -> str:
        return make_layer_name(ds, "streamed", shape_key)

    def _schedule_streamed_shape_update(self, state_key: str):
        state = self._streamed_shape_states.get(state_key)
        if state is None:
            return
        state.timer.start(int(self.args.stream_shapes_debounce_ms))

    def _update_streamed_shape_layer(self, state_key: str):
        state = self._streamed_shape_states.get(state_key)
        if state is None or self.active_dataset != state.dataset:
            return

        bounds = self._current_view_bounds()
        if bounds is None:
            return
        view_width = float(bounds[2] - bounds[0])
        view_height = float(bounds[3] - bounds[1])
        if max(view_width, view_height) > float(self.args.stream_shapes_max_view_size):
            self._remove_layer_by_name(state.detail_layer_name)
            self._set_status(
                f"{state.dataset} {state.shape_key}: zoom in to stream exact polygons "
                f"(view={max(view_width, view_height):.0f} > {self.args.stream_shapes_max_view_size:.0f})."
            )
            return

        candidates = query_geometries_for_bounds(state.gdf, bounds)
        n_candidates = int(len(candidates))
        if n_candidates > int(self.args.stream_shapes_max_polygons):
            self._remove_layer_by_name(state.detail_layer_name)
            self._set_status(
                f"{state.dataset} {state.shape_key}: {n_candidates:,} cells in view; "
                f"zoom further or raise --stream-shapes-max-polygons."
            )
            return

        data = geometry_to_napari_polygons(
            candidates.geometry,
            max_shapes=None,
            simplify_tolerance=self.args.shape_simplify_tolerance,
            max_vertices_per_polygon=self.args.shape_max_vertices_per_polygon,
        )
        self._remove_layer_by_name(state.detail_layer_name)
        if len(data) > 0:
            self.viewer.add_shapes(
                data,
                shape_type="path",
                name=state.detail_layer_name,
                edge_color=state.edge_color,
                edge_width=self.args.shape_edge_width,
                visible=not self.args.hide_shapes,
            )
        self._set_status(
            f"{state.dataset} streamed exact polygons for {state.shape_key}: "
            f"{len(data):,} polygons in current view."
        )

    def _add_streamed_shape_layer(self, shape_key: str) -> int:
        if self._active_sdata is None or self.active_dataset is None:
            return 0
        ds = self.active_dataset
        if shape_key not in self._active_sdata.shapes:
            raise KeyError(f"Shape key '{shape_key}' not found in current dataset.")

        state_key = self._streamed_state_key(ds, shape_key)
        old_state = self._streamed_shape_states.pop(state_key, None)
        if old_state is not None:
            old_state.timer.stop()

        overview_layer_name = make_layer_name(ds, "streamed_bbox", shape_key)
        detail_layer_name = state_key
        self._remove_layer_by_name(overview_layer_name)
        self._remove_layer_by_name(detail_layer_name)

        gdf = self._active_sdata.shapes[shape_key]
        edge_color = np.asarray(stable_layer_color(shape_key, alpha=self.args.shape_opacity), dtype=float)
        overview = geometry_to_napari_bounding_boxes(gdf.geometry, max_shapes=None)
        if len(overview) > 0:
            self.viewer.add_shapes(
                overview,
                shape_type="rectangle",
                name=overview_layer_name,
                edge_color=edge_color,
                edge_width=self.args.shape_edge_width,
                face_color="transparent",
                opacity=min(float(self.args.shape_opacity), 0.35),
                visible=not self.args.hide_shapes,
            )

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda key=state_key: self._update_streamed_shape_layer(key))
        state = StreamedShapeLayerState(
            dataset=ds,
            shape_key=shape_key,
            gdf=gdf,
            detail_layer_name=detail_layer_name,
            overview_layer_name=overview_layer_name,
            timer=timer,
            edge_color=edge_color,
        )
        self._streamed_shape_states[state_key] = state

        def schedule(_event=None, key=state_key):
            self._schedule_streamed_shape_update(key)

        try:
            self.viewer.camera.events.zoom.connect(schedule)
            self.viewer.camera.events.center.connect(schedule)
        except Exception:
            pass
        self._schedule_streamed_shape_update(state_key)
        return int(len(gdf))

    def stream_selected_shapes(self, dataset_name: str, shape_keys: list[str]):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        added_layers = 0
        total_units = 0
        for shape_key in shape_keys:
            n = self._add_streamed_shape_layer(shape_key)
            if n > 0:
                added_layers += 1
                total_units += n
        self._set_status(
            f"{self.active_dataset} added streamed polygon layer(s)={added_layers}, "
            f"overview cells={total_units:,}; zoom in for exact boundaries."
        )

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
                make_layer_name(ds, "streamed", shape_key),
                make_layer_name(ds, "streamed_bbox", shape_key),
            ]
            for name in layer_names:
                before = len(self.viewer.layers)
                self._remove_layer_by_name(name)
                if len(self.viewer.layers) < before:
                    removed += 1
            state_key = self._streamed_state_key(ds, shape_key)
            state = self._streamed_shape_states.pop(state_key, None)
            if state is not None:
                state.timer.stop()
        self._set_status(f"{ds} removed {removed} {self._segmentation_layer_type()} layer(s).")

    def load_all_shapes_capped(self, dataset_name: str):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        added_layers = 0
        total_units = 0
        for shape_key in self._segmentation_keys:
            n = self._add_shape_layer(
                shape_key,
                cap=self.args.max_shapes_per_layer,
                simplify_tolerance=self.args.shape_simplify_tolerance,
            )
            if n > 0:
                added_layers += 1
                total_units += n
        self._set_status(
            f"{self.active_dataset} loaded all capped {self._segmentation_layer_type()}: "
            f"{added_layers} layer(s), {self._segmentation_unit_name()}={total_units:,}"
        )

    def load_transcripts(self, dataset_name: str, full: bool):
        if not self._ensure_dataset_is_active(dataset_name):
            self._set_status(f"Could not activate dataset {dataset_name}.")
            return
        cap = None if full else self.args.max_transcripts
        stats = self._add_transcripts_layer(cap)
        mode = "full" if cap is None else f"sampled ({cap:,})"
        self._set_status(
            f"{self.active_dataset} loaded {mode} transcripts: total={stats['total']:,}, "
            f"assigned={stats['assigned']:,}, unassigned={stats['unassigned']:,}"
        )

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
        "--full-shape-render-mode",
        default="bbox",
        choices=["path", "polygon", "bbox", "centroid"],
        help=(
            "Render mode for uncapped selected segmentation loads. "
            "bbox is much lighter than full boundaries for large Cellpose layers."
        ),
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
        "--stream-shapes-max-view-size",
        default=2500.0,
        type=float,
        help="Maximum current view width/height in world units before exact polygon streaming is enabled.",
    )
    parser.add_argument(
        "--stream-shapes-max-polygons",
        default=2500,
        type=int,
        help="Maximum number of polygons to render in the streamed exact-detail layer.",
    )
    parser.add_argument(
        "--stream-shapes-debounce-ms",
        default=250,
        type=int,
        help="Delay after pan/zoom before refreshing streamed exact polygon detail.",
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
    if args.startup_mode == "fast":
        if args.max_transcripts is None:
            args.max_transcripts = FAST_DEFAULT_MAX_TRANSCRIPTS
    if args.max_shapes_per_layer is None:
        args.max_shapes_per_layer = FAST_DEFAULT_MAX_SHAPES_PER_LAYER
    if args.label_chunk_size <= 0:
        parser.error("--label-chunk-size must be positive.")
    if args.label_contour_width < 0:
        parser.error("--label-contour-width must be non-negative.")
    if args.stream_shapes_max_polygons <= 0:
        parser.error("--stream-shapes-max-polygons must be positive.")
    if args.stream_shapes_debounce_ms < 0:
        parser.error("--stream-shapes-debounce-ms must be non-negative.")
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
        load_selected_shapes_callback=controller.load_selected_shapes,
        load_selected_full_shapes_callback=controller.load_selected_shapes_full,
        stream_selected_shapes_callback=controller.stream_selected_shapes,
        load_selected_labels_callback=controller.load_selected_labels,
        unload_selected_shapes_callback=controller.unload_selected_shapes,
        load_all_shapes_callback=controller.load_all_shapes_capped,
        load_transcripts_callback=controller.load_transcripts,
        load_images_callback=controller.load_images_on_demand,
        initial_dataset=initial_dataset,
        skip_images=args.skip_images,
        segmentation_source=args.segmentation_source,
        full_shape_render_mode=args.full_shape_render_mode,
    )
    controller.set_status_callback(switcher.set_status)
    controller.set_shape_keys_callback(switcher.set_shape_keys)
    viewer.window.add_dock_widget(switcher, area="right", name="Dataset Switcher")

    controller.load_dataset(initial_dataset, force=True)
    napari.run()


if __name__ == "__main__":
    main()
