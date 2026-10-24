"""UMAP embedding view: loading, colouring, and the floating scatter window.

MerXen's clustering tables (``table_*_clustering_squidpy``) store each cell's
UMAP embedding in ``obsm['X_umap']``, row-aligned with the table's instance-key
column (the segmentation-mask ids the Cell type labels overlay also joins on).
This module reads just those two arrays -- never the full AnnData -- and draws
the embedding as a CPU-rasterised scatter, so the floating window needs no
second OpenGL context and renders headless in tests.

The controller owns all dataset state; the widgets here only display what they
are given and report clicks back through Qt signals and plain callbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from qtpy.QtCore import QPointF, QRectF, QSize, Qt, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .utils import (
    CELL_TYPE_SOURCES,
    analysis_source_metadata,
    cell_id_to_int as cell_id_to_int,
    cell_ids_as_int64,
    _resolve_instance_key,
    cell_type_source,
    cellpose_value_clip_range,
    cell_type_table_key_for_source,
    open_zarr_group_unconsolidated,
)

log = logging.getLogger("napari_compare")

UMAP_OBSM_KEY = "X_umap"
#: Colour-by modes offered by the UMAP tab, in display order.
UMAP_COLOUR_MODES: tuple[tuple[str, str], ...] = (
    ("broad", "Broad cell type"),
    ("fine", "Fine cell type"),
    ("statistic", "Per-cell statistic"),
)
UMAP_COLORMAPS = ("viridis", "magma", "inferno", "plasma", "turbo", "gray")
#: Plot background; matches the Cell inspector bar so the two read as a set.
UMAP_BACKGROUND_RGBA = (20, 22, 28, 255)
#: Cells with no annotation / no statistic value.
UMAP_MISSING_RGBA = (0.42, 0.44, 0.48, 1.0)
#: Cells whose type is unticked in the Cell type labels tab: kept (so the
#: embedding's shape stays readable) but pushed back into a dim layer.
UMAP_MUTED_RGBA = (0.25, 0.26, 0.30, 1.0)
UMAP_MISSING_LABEL = "Unannotated"
UMAP_DEFAULT_POINT_SIZE = 2  # slider units: radius = size / 2 logical px
UMAP_POINT_SIZE_RANGE = (1, 10)
UMAP_PICK_RADIUS_PX = 7.0
UMAP_CLICK_DRAG_THRESHOLD_PX = 4.0
UMAP_WHEEL_ZOOM_STEP = 1.2
UMAP_FIT_MARGIN = 1.06
#: Outline colour of a region of interest, in both views.
UMAP_ROI_COLOR = "#ffffff"
#: Minimum pointer travel (px) before a lasso records another vertex.
UMAP_LASSO_MIN_STEP_PX = 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _unique_id_index(ids: np.ndarray) -> tuple[pd.Index, np.ndarray]:
    """Return a unique ``pd.Index`` over ``ids`` and the rows it keeps (first wins)."""
    ids = np.asarray(ids)
    duplicated = pd.Index(ids).duplicated(keep="first")
    keep = np.flatnonzero(~duplicated)
    return pd.Index(ids[keep]), keep


@dataclass
class CellUmapEmbedding:
    """One clustering table's UMAP, row-aligned with its mask cell ids.

    ``segmentation`` is the :data:`utils.CELL_TYPE_SOURCES` key it was read for;
    ``cell_ids`` are unique exact integer instance-key values (the mask pixel ids);
    ``coords`` is ``(N, 2)`` ``float32`` UMAP-1 / UMAP-2.
    """

    segmentation: str
    table_key: str
    instance_key: str
    cell_ids: np.ndarray
    coords: np.ndarray
    _index: pd.Index | None = field(default=None, init=False, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.cell_ids.shape[0])

    def rows_for_ids(self, ids: Iterable[Any]) -> np.ndarray:
        """Return each id's row, or ``-1`` where that cell has no embedding."""
        if self._index is None:
            self._index = pd.Index(np.asarray(self.cell_ids))
        wanted, valid = cell_ids_as_int64(list(ids) if not isinstance(ids, np.ndarray) else ids)
        rows = np.full(wanted.shape[0], -1, dtype=np.int64)
        if valid.any():
            rows[valid] = self._index.get_indexer(wanted[valid])
        return rows

    def row_for_id(self, cell_id: Any) -> int:
        return int(self.rows_for_ids([cell_id])[0])


def _read_elem():
    try:
        from anndata.io import read_elem
    except ImportError:  # pragma: no cover - anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem


def cell_umap_available(zarr_path: Any, segmentation: str) -> bool:
    """Check for a stored two-dimensional embedding without loading its data."""
    try:
        key = cell_type_table_key_for_source(zarr_path, segmentation, embedding=True)
        if key is None:
            return False
        array = open_zarr_group_unconsolidated(zarr_path)[f"tables/{key}/obsm/{UMAP_OBSM_KEY}"]
        return len(array.shape) == 2 and array.shape[0] > 0 and array.shape[1] >= 2
    except Exception:
        return False


def load_cell_umap(
    zarr_path: Any,
    segmentation: str,
    obsm_key: str = UMAP_OBSM_KEY,
) -> CellUmapEmbedding | None:
    """Load a cell-type source's stored UMAP from its clustering table.

    Reads only ``obsm[obsm_key]`` and the instance-key obs column (resolved the
    same way as :func:`utils.load_cell_type_assignments`), so the table's
    expression matrix and neighbour graphs are never touched. Rows with a
    non-finite embedding or no integer id are dropped; duplicate ids keep their
    first row. Never raises -- a store without a UMAP yields ``None``.
    """
    if zarr_path is None:
        return None
    source = cell_type_source(segmentation)
    metadata = None
    if source is not None and source.method == "pciseq":
        metadata = analysis_source_metadata(zarr_path, segmentation)
        if metadata is None:
            return None
    table_key = cell_type_table_key_for_source(zarr_path, segmentation, embedding=True)
    if table_key is None:
        return None
    try:
        read_elem = _read_elem()
        root = open_zarr_group_unconsolidated(zarr_path)
        table = root[f"tables/{table_key}"]
        obsm = table["obsm"]
        if obsm_key not in obsm:
            return None
        coords = np.asarray(read_elem(obsm[obsm_key]), dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            return None
        coords = np.ascontiguousarray(coords[:, :2])

        attrs: dict = {}
        try:
            uns = table["uns"]
            if "spatialdata_attrs" in uns:
                attrs = dict(read_elem(uns["spatialdata_attrs"]))
        except Exception:
            attrs = {}
        obs = table["obs"]
        obs_attrs = dict(obs.attrs)
        columns = [str(c) for c in obs_attrs.get("column-order", [])]
        instance_key = _resolve_instance_key(
            SimpleNamespace(
                uns={"spatialdata_attrs": attrs},
                obs=SimpleNamespace(columns=columns),
            )
        )
        id_column = instance_key or str(obs_attrs.get("_index", "_index"))
        raw_ids = read_elem(obs[id_column])
        if isinstance(raw_ids, pd.Categorical):
            raw_ids = np.asarray(raw_ids)
        ids, valid = cell_ids_as_int64(raw_ids)
        if ids.shape[0] != coords.shape[0]:
            log.warning(
                "UMAP %s has %d rows but %d cell ids; ignoring it.",
                table_key,
                coords.shape[0],
                ids.shape[0],
            )
            return None
        valid &= np.isfinite(coords).all(axis=1)
        rows = np.flatnonzero(valid)
        _index, keep = _unique_id_index(ids[rows])
        rows = rows[keep]
        if rows.size == 0:
            return None
        if metadata is not None:
            current = analysis_source_metadata(zarr_path, segmentation)
            if current is None or current["analysis_id"] != metadata["analysis_id"]:
                return None
        return CellUmapEmbedding(
            segmentation=str(segmentation),
            table_key=str(table_key),
            instance_key=str(instance_key),
            cell_ids=ids[rows],
            coords=coords[rows],
        )
    except Exception as exc:
        log.warning("Could not read UMAP for %s (%s)", segmentation, exc)
        return None


def labels_for_cell_ids(source_ids, source_labels, target_ids) -> np.ndarray:
    """Look up each ``target_ids`` cell's label (``None`` where it has none)."""
    ids, valid = cell_ids_as_int64(source_ids)
    labels = np.asarray(source_labels, dtype=object)
    index, keep = _unique_id_index(ids[valid])
    labels = labels[valid][keep]
    wanted, wanted_valid = cell_ids_as_int64(target_ids)
    out = np.full(wanted.shape[0], None, dtype=object)
    positions = np.full(wanted.shape[0], -1, dtype=np.int64)
    positions[wanted_valid] = index.get_indexer(wanted[wanted_valid])
    found = positions >= 0
    out[found] = labels[positions[found]]
    return out


def values_for_cell_ids(source_ids, source_values, target_ids) -> np.ndarray:
    """Look up each ``target_ids`` cell's numeric value (``NaN`` where absent)."""
    ids, valid = cell_ids_as_int64(source_ids)
    values = np.asarray(source_values, dtype=np.float64)
    index, keep = _unique_id_index(ids[valid])
    values = values[valid][keep]
    wanted, wanted_valid = cell_ids_as_int64(target_ids)
    out = np.full(wanted.shape[0], np.nan, dtype=np.float64)
    positions = np.full(wanted.shape[0], -1, dtype=np.int64)
    positions[wanted_valid] = index.get_indexer(wanted[wanted_valid])
    found = positions >= 0
    out[found] = values[positions[found]]
    return out


def statistic_joins_source(source_key: str, statistic_shape_key: str) -> bool:
    """Whether per-cell statistics measured on ``statistic_shape_key`` masks
    share cell ids with the clustering ``source_key`` (same masks)."""
    source = cell_type_source(source_key)
    return source is not None and str(statistic_shape_key) in source.mask_shape_keys


def cell_type_source_label(source_key: str) -> str:
    source = cell_type_source(source_key)
    return source.label if source is not None else str(source_key)


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------


@dataclass
class UmapLegendEntry:
    label: str
    rgba: tuple[float, float, float, float]
    count: int
    enabled: bool = True


@dataclass
class UmapColouring:
    """Per-row colours and draw order for one UMAP colouring.

    ``order`` lists row indices back-to-front; its first ``n_background``
    entries (unannotated / unticked / no-value cells) are rasterised as a
    separate bottom layer so they never cover coloured cells.
    """

    colors: np.ndarray                     # uint8 (N, 4)
    order: np.ndarray                      # int64 row indices, back to front
    n_background: int
    title: str
    legend: list[UmapLegendEntry] = field(default_factory=list)
    colorbar: tuple[np.ndarray, float, float] | None = None
    hover_text: np.ndarray | None = None   # per-row category text
    values: np.ndarray | None = None       # per-row statistic values


def _rgba_uint8(rgba) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.float64)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:-1] + (1,))], axis=-1)
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def _stable_rank(n: int, seed: int) -> np.ndarray:
    """A fixed random rank per row so categorical draw order never shifts."""
    return np.random.default_rng(int(seed)).permutation(int(n))


def categorical_umap_colouring(
    labels,
    colors: Mapping[str, tuple],
    *,
    order: Iterable[str] | None = None,
    enabled: Iterable[str] | None = None,
    title: str = "",
    seed: int = 0,
    missing_label: str = UMAP_MISSING_LABEL,
) -> UmapColouring:
    """Colour rows by category, matching the Cell type labels palette.

    ``labels`` holds one entry per embedding row (``None``/empty = missing).
    Categories absent from ``colors`` count as missing. When ``enabled`` is
    given, categories outside it are drawn in :data:`UMAP_MUTED_RGBA` beneath
    the rest. Coloured rows are drawn in a fixed random order so no single type
    systematically overplots another.
    """
    text = pd.Series(np.asarray(labels, dtype=object), copy=False)
    text = text.where(text.notna(), "").astype(str).str.strip()
    codes, uniques = pd.factorize(text.to_numpy(dtype=object), sort=False)
    uniques = [str(u) for u in uniques]
    enabled_set = None if enabled is None else {str(e) for e in enabled}

    lut = np.empty((max(1, len(uniques)), 4), dtype=np.uint8)
    lut_missing = np.zeros(lut.shape[0], dtype=bool)
    lut_muted = np.zeros(lut.shape[0], dtype=bool)
    for i, label in enumerate(uniques):
        rgba = colors.get(label) if label else None
        if rgba is None:
            lut[i] = _rgba_uint8(UMAP_MISSING_RGBA)
            lut_missing[i] = True
        elif enabled_set is not None and label not in enabled_set:
            lut[i] = _rgba_uint8(UMAP_MUTED_RGBA)
            lut_muted[i] = True
        else:
            lut[i] = _rgba_uint8(rgba)
    n = int(codes.shape[0])
    rows_missing = lut_missing[codes] if n else np.zeros(0, dtype=bool)
    rows_muted = lut_muted[codes] if n else np.zeros(0, dtype=bool)
    rgba_rows = lut[codes] if n else np.empty((0, 4), dtype=np.uint8)

    background = rows_missing | rows_muted
    rank = _stable_rank(n, seed)
    back = np.flatnonzero(background)
    front = np.flatnonzero(~background)
    back = back[np.argsort(rank[back], kind="stable")]
    front = front[np.argsort(rank[front], kind="stable")]
    draw_order = np.concatenate([back, front]).astype(np.int64, copy=False)

    counts = np.bincount(codes, minlength=len(uniques)) if n else np.zeros(len(uniques), int)
    count_by_label = {label: int(counts[i]) for i, label in enumerate(uniques)}
    legend_order = list(order) if order is not None else sorted(colors)
    legend = [
        UmapLegendEntry(
            label=str(label),
            rgba=tuple(float(c) for c in colors[label]),
            count=count_by_label.get(str(label), 0),
            enabled=enabled_set is None or str(label) in enabled_set,
        )
        for label in legend_order
        if label in colors and count_by_label.get(str(label), 0) > 0
    ]
    missing_count = int(rows_missing.sum())
    if missing_count:
        legend.append(UmapLegendEntry(missing_label, UMAP_MISSING_RGBA, missing_count))

    hover = text.to_numpy(dtype=object).copy()
    if n:
        hover[rows_missing] = missing_label
    return UmapColouring(
        colors=np.ascontiguousarray(rgba_rows),
        order=draw_order,
        n_background=int(back.shape[0]),
        title=str(title),
        legend=legend,
        hover_text=hover,
    )


def continuous_umap_colouring(
    values,
    lut,
    *,
    clip: tuple[float, float] | None = None,
    title: str = "",
) -> UmapColouring:
    """Colour rows by a scalar through ``lut`` (``(K, 3|4)`` RGBA in 0..1).

    Uses the same binning as the spatial per-cell statistic overlay
    (``floor(normalised * (K - 1))`` after clipping), so a cell has the same
    colour in both views when ``clip`` is the overlay's clip range. Missing
    values are grey and drawn first; the rest are drawn low-to-high so bright
    outliers stay on top.
    """
    vals = np.asarray(values, dtype=np.float64)
    lut_arr = np.asarray(lut, dtype=np.float64)
    if lut_arr.ndim != 2 or lut_arr.shape[0] == 0:
        raise ValueError("lut must be a non-empty (K, 3|4) array.")
    finite = np.isfinite(vals)
    if clip is None:
        clip = cellpose_value_clip_range(vals) if finite.any() else (0.0, 1.0)
    low, high = float(clip[0]), float(clip[1])
    lut_u8 = _rgba_uint8(lut_arr)
    rgba_rows = np.empty((vals.shape[0], 4), dtype=np.uint8)
    rgba_rows[:] = _rgba_uint8(UMAP_MISSING_RGBA)
    if finite.any():
        if high <= low:
            bins = np.full(int(finite.sum()), lut_u8.shape[0] // 2, dtype=np.int64)
        else:
            normalised = np.clip((vals[finite] - low) / (high - low), 0.0, 1.0)
            bins = np.floor(normalised * (lut_u8.shape[0] - 1)).astype(np.int64)
        rgba_rows[finite] = lut_u8[bins]
    back = np.flatnonzero(~finite)
    front = np.flatnonzero(finite)
    front = front[np.argsort(vals[front], kind="stable")]
    return UmapColouring(
        colors=np.ascontiguousarray(rgba_rows),
        order=np.concatenate([back, front]).astype(np.int64, copy=False),
        n_background=int(back.shape[0]),
        title=str(title),
        colorbar=(lut_arr, low, high),
        values=vals,
    )


def points_in_polygon(points, polygon) -> np.ndarray:
    """Boolean mask of the ``(N, 2)`` x/y ``points`` inside a lasso ``polygon``.

    ``polygon`` is the ``(M, 2)`` x/y outline as drawn; it is closed from the
    last vertex back to the first. Freehand outlines often cross themselves, so
    the ring is repaired with ``make_valid`` (a figure-of-eight keeps both lobes)
    before a bounding-box-prefiltered vectorised containment test.
    """
    import shapely

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(pts.shape[0], dtype=bool)
    ring = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if ring.shape[0] >= 2:
        keep = np.ones(ring.shape[0], dtype=bool)
        keep[1:] = np.any(np.diff(ring, axis=0) != 0, axis=1)
        ring = ring[keep]
    if ring.shape[0] < 3 or pts.shape[0] == 0:
        return inside
    geometry = shapely.Polygon(ring)
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    if geometry.is_empty or geometry.area <= 0:
        return inside
    minx, miny, maxx, maxy = geometry.bounds
    candidates = np.flatnonzero(
        (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
    )
    if candidates.size:
        shapely.prepare(geometry)
        inside[candidates] = shapely.contains_xy(
            geometry, pts[candidates, 0], pts[candidates, 1]
        )
    return inside


def emphasise_rows(colouring: UmapColouring, rows) -> UmapColouring:
    """Mute every row outside ``rows`` and draw it behind the region's cells."""
    n = colouring.colors.shape[0]
    inside = np.zeros(n, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    inside[rows[(rows >= 0) & (rows < n)]] = True
    colors = colouring.colors.copy()
    colors[~inside] = _rgba_uint8(UMAP_MUTED_RGBA)
    order = colouring.order
    back = order[~inside[order]]
    front = order[inside[order]]
    return replace(
        colouring,
        colors=colors,
        order=np.concatenate([back, front]).astype(np.int64, copy=False),
        n_background=int(back.shape[0]),
    )


@dataclass
class UmapRoiSlice:
    label: str
    rgba: tuple[float, float, float, float]
    count: int


@dataclass
class UmapRoiSummary:
    """What the UMAP window's region panel shows for the current region."""

    total: int
    slices: list[UmapRoiSlice]
    level: str                 # "Broad cell type" / "Fine cell type"
    drawn_in: str              # "the spatial view" / "the UMAP"
    note: str = ""


def summarise_roi_labels(
    labels,
    colors: Mapping[str, tuple],
    *,
    level: str,
    drawn_in: str,
    note: str = "",
    missing_label: str = UMAP_MISSING_LABEL,
) -> UmapRoiSummary:
    """Count a region's cells per type, largest first (unannotated last)."""
    text = pd.Series(np.asarray(labels, dtype=object), copy=False)
    text = text.where(text.notna(), "").astype(str).str.strip()
    known = text.map(lambda value: value if value and value in colors else "")
    counts = known.value_counts()
    slices = [
        UmapRoiSlice(str(label), tuple(float(c) for c in colors[label]), int(count))
        for label, count in counts.items()
        if label
    ]
    slices.sort(key=lambda item: (-item.count, item.label))
    missing = int(counts.get("", 0))
    if missing:
        slices.append(UmapRoiSlice(missing_label, UMAP_MISSING_RGBA, missing))
    return UmapRoiSummary(
        total=int(text.shape[0]),
        slices=slices,
        level=str(level),
        drawn_in=str(drawn_in),
        note=str(note),
    )


# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------


def _swatch_pixmap(rgba, px: int = 14, muted: bool = False) -> QPixmap:
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    r, g, b = (int(round(255 * float(c))) for c in tuple(rgba)[:3])
    color = QColor(r, g, b, 90 if muted else 255)
    painter.setPen(QPen(QColor(0, 0, 0, 160), 1))
    painter.setBrush(color)
    painter.drawEllipse(QRectF(1, 1, px - 2, px - 2))
    painter.end()
    return pixmap


def _disk_offsets(radius: float, row_stride: int) -> np.ndarray:
    """Flat-buffer offsets covering a disk, ordered outer ring -> centre."""
    r = max(0.0, float(radius))
    reach = int(np.floor(r))
    offsets = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            dist2 = dx * dx + dy * dy
            if dist2 <= r * r + 1e-9:
                offsets.append((dist2, dy * row_stride + dx))
    if not offsets:
        offsets = [(0, 0)]
    offsets.sort(key=lambda item: -item[0])
    return np.asarray([off for _d, off in offsets], dtype=np.int64)


class UmapPlotCanvas(QWidget):
    """Pan/zoom scatter of an embedding, rasterised with NumPy into a QImage.

    170k points re-rasterise in a few tens of milliseconds, which keeps panning
    smooth without a GPU canvas. Highlighted cells are drawn as vector rings on
    top so they stay crisp at any zoom. The two axes share one scale, as is
    conventional for UMAPs.
    """

    point_clicked = Signal(int)
    point_hovered = Signal(int)
    #: A finished lasso: ``(M, 2)`` x/y vertices in UMAP units (closed on release).
    lasso_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self._coords = np.empty((0, 2), dtype=np.float64)
        self._colors32 = np.empty(0, dtype=np.uint32)
        self._order = np.empty(0, dtype=np.int64)
        self._n_background = 0
        self._point_radius = UMAP_DEFAULT_POINT_SIZE / 2.0
        self._center = np.zeros(2, dtype=np.float64)
        self._scale = 1.0  # logical px per UMAP unit
        self._fitted = False
        #: Set once the user zooms or pans; until then a resize re-fits the plot.
        self._view_adjusted = False
        self._image: QImage | None = None
        self._image_buffer: np.ndarray | None = None
        self._image_key: tuple | None = None
        self._dirty = True
        self._highlights: list[tuple[int, QColor]] = []
        self._hover_row = -1
        self._tree = None
        self._tree_built = False
        self._press_pos: QPointF | None = None
        self._last_drag_pos: QPointF | None = None
        self._dragging = False
        self._message = ""
        self._hover_text_fn: Callable[[int], str] | None = None
        self._lasso_mode = False
        self._lasso_points: list[tuple[float, float]] | None = None
        self._lasso_last_px: QPointF | None = None
        self._roi_polygon: np.ndarray | None = None
        self._background32 = int(
            np.asarray([UMAP_BACKGROUND_RGBA], dtype=np.uint8).view(np.uint32)[0, 0]
        )

    # -- public API ---------------------------------------------------------
    @property
    def point_count(self) -> int:
        return int(self._coords.shape[0])

    def set_message(self, text: str) -> None:
        """Text drawn centred over the plot (shown when there are no points)."""
        self._message = str(text)
        self.update()

    def set_hover_text_callback(self, fn: Callable[[int], str] | None) -> None:
        self._hover_text_fn = fn

    def set_embedding(self, coords) -> None:
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        self._coords = np.ascontiguousarray(coords)
        self._colors32 = np.full(coords.shape[0], self._background32, dtype=np.uint32)
        self._order = np.arange(coords.shape[0], dtype=np.int64)
        self._n_background = 0
        self._tree = None
        self._tree_built = False
        self._highlights = []
        self._hover_row = -1
        self._roi_polygon = None
        self._lasso_points = None
        self.reset_view()

    def clear(self) -> None:
        self.set_embedding(np.empty((0, 2)))

    def set_colouring(self, colors_rgba_u8, order, n_background: int = 0) -> None:
        colors = np.ascontiguousarray(np.asarray(colors_rgba_u8, dtype=np.uint8).reshape(-1, 4))
        if colors.shape[0] != self._coords.shape[0]:
            raise ValueError(
                f"Expected {self._coords.shape[0]} colours, got {colors.shape[0]}."
            )
        self._colors32 = colors.view(np.uint32).reshape(-1)
        self._order = np.asarray(order, dtype=np.int64).reshape(-1)
        self._n_background = int(max(0, min(int(n_background), self._order.shape[0])))
        self._dirty = True
        self.update()

    def set_highlights(self, highlights: Iterable[tuple[int, str]]) -> None:
        cleaned: list[tuple[int, QColor]] = []
        for row, color in highlights:
            row = int(row)
            if 0 <= row < self._coords.shape[0]:
                cleaned.append((row, QColor(str(color))))
        self._highlights = cleaned
        self.update()

    def highlighted_rows(self) -> list[int]:
        return [row for row, _color in self._highlights]

    def set_point_radius(self, radius_px: float) -> None:
        self._point_radius = max(0.25, float(radius_px))
        self._dirty = True
        self.update()

    @property
    def lasso_mode(self) -> bool:
        return self._lasso_mode

    def set_lasso_mode(self, enabled: bool) -> None:
        """While on, a left-button drag draws a lasso instead of panning."""
        self._lasso_mode = bool(enabled)
        self._lasso_points = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def set_roi_polygon(self, polygon) -> None:
        """Outline a region drawn in UMAP units (``None`` removes it)."""
        self._roi_polygon = (
            None
            if polygon is None
            else np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        )
        self.update()

    @property
    def roi_polygon(self) -> np.ndarray | None:
        return self._roi_polygon

    def reset_view(self) -> None:
        self._view_adjusted = False
        self._fit_view()
        self._dirty = True
        self.update()

    # -- geometry -----------------------------------------------------------
    def _fit_view(self) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        if self._coords.shape[0] == 0:
            self._center = np.zeros(2)
            self._scale = 1.0
            self._fitted = False
            return
        lo = self._coords.min(axis=0)
        hi = self._coords.max(axis=0)
        span = np.maximum(hi - lo, 1e-6) * UMAP_FIT_MARGIN
        self._center = (lo + hi) / 2.0
        self._scale = float(min(w / span[0], h / span[1]))
        self._fitted = True

    def data_to_view(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        px = (xy[:, 0] - self._center[0]) * self._scale + self.width() / 2.0
        py = self.height() / 2.0 - (xy[:, 1] - self._center[1]) * self._scale
        return np.column_stack([px, py])

    def view_to_data(self, px: float, py: float) -> np.ndarray:
        x = (float(px) - self.width() / 2.0) / self._scale + self._center[0]
        y = (self.height() / 2.0 - float(py)) / self._scale + self._center[1]
        return np.asarray([x, y], dtype=np.float64)

    def _ensure_tree(self):
        if not self._tree_built:
            self._tree_built = True
            try:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(self._coords) if self._coords.shape[0] else None
            except Exception:
                self._tree = None
        return self._tree

    def nearest_row(self, px: float, py: float, max_px: float | None = None) -> int:
        """Row of the point nearest a view position (``-1`` if none within reach)."""
        if self._coords.shape[0] == 0:
            return -1
        reach_px = UMAP_PICK_RADIUS_PX if max_px is None else float(max_px)
        reach_px = max(reach_px, self._point_radius + 2.0)
        reach = reach_px / max(self._scale, 1e-12)
        target = self.view_to_data(px, py)
        tree = self._ensure_tree()
        if tree is not None:
            dist, idx = tree.query(target, k=1, distance_upper_bound=reach)
            if not np.isfinite(dist) or int(idx) >= self._coords.shape[0]:
                return -1
            return int(idx)
        d2 = np.sum((self._coords - target) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx if d2[idx] <= reach * reach else -1

    # -- rendering ----------------------------------------------------------
    def _rasterize(self) -> None:
        dpr = float(self.devicePixelRatioF() or 1.0)
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        radius = self._point_radius * dpr
        pad = int(np.floor(radius)) + 1
        stride = w + 2 * pad
        buf = np.full((h + 2 * pad) * stride, self._background32, dtype=np.uint32)
        order = self._order
        if order.size:
            pts = self._coords[order]
            scale = self._scale * dpr
            px = np.floor((pts[:, 0] - self._center[0]) * scale + w / 2.0).astype(np.int64)
            py = np.floor(h / 2.0 - (pts[:, 1] - self._center[1]) * scale).astype(np.int64)
            visible = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            flat = (py + pad) * stride + (px + pad)
            colors = self._colors32[order]
            offsets = _disk_offsets(radius, stride)
            n_back = self._n_background
            for layer in (slice(0, n_back), slice(n_back, None)):
                keep = visible[layer]
                layer_flat = flat[layer][keep]
                layer_colors = colors[layer][keep]
                if layer_flat.size == 0:
                    continue
                # Repeated indices: the last write wins, i.e. later rows in
                # ``order`` (drawn "in front") keep their pixels.
                for offset in offsets:
                    buf[layer_flat + offset] = layer_colors
        image = buf.reshape(h + 2 * pad, stride)[pad : pad + h, pad : pad + w]
        image = np.ascontiguousarray(image)
        self._image_buffer = image  # QImage does not copy; keep the bytes alive
        qimage = QImage(image.data, w, h, w * 4, QImage.Format_RGBA8888)
        qimage.setDevicePixelRatio(dpr)
        self._image = qimage
        self._image_key = (self.width(), self.height(), dpr)
        self._dirty = False

    def rendered_rgba(self) -> np.ndarray | None:
        """The last rasterised plot as ``(H, W, 4)`` uint8 (for tests/export)."""
        if self._dirty or self._image is None:
            self._rasterize()
        if self._image_buffer is None:
            return None
        return self._image_buffer.view(np.uint8).reshape(self._image_buffer.shape + (4,))

    def paintEvent(self, _event):  # noqa: N802 (Qt override)
        dpr = float(self.devicePixelRatioF() or 1.0)
        if self._dirty or self._image is None or self._image_key != (self.width(), self.height(), dpr):
            self._rasterize()
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._roi_polygon is not None and self._roi_polygon.shape[0] >= 3:
            self._draw_outline(painter, self.data_to_view(self._roi_polygon), closed=True)
        if self._lasso_points:
            self._draw_outline(
                painter, self.data_to_view(np.asarray(self._lasso_points)), closed=False
            )
        ring = max(6.0, self._point_radius * 2.0 + 4.0)
        if self._highlights:
            rows = np.asarray([row for row, _c in self._highlights], dtype=np.int64)
            positions = self.data_to_view(self._coords[rows])
            for (x, y), (_row, color) in zip(positions, self._highlights):
                centre = QPointF(float(x), float(y))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(0, 0, 0, 200), 4.5))
                painter.drawEllipse(centre, ring, ring)
                painter.setPen(QPen(color, 2.5))
                painter.drawEllipse(centre, ring, ring)
        if 0 <= self._hover_row < self._coords.shape[0]:
            x, y = self.data_to_view(self._coords[self._hover_row])[0]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
            painter.drawEllipse(QPointF(float(x), float(y)), ring * 0.7, ring * 0.7)
        if self._message and self._coords.shape[0] == 0:
            painter.setPen(QColor(200, 205, 214))
            painter.drawText(self.rect().adjusted(16, 16, -16, -16), Qt.AlignCenter | Qt.TextWordWrap, self._message)
        painter.end()

    @staticmethod
    def _draw_outline(painter: QPainter, view_xy: np.ndarray, *, closed: bool) -> None:
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in view_xy])
        painter.setBrush(Qt.NoBrush)
        for pen in (QPen(QColor(0, 0, 0, 200), 4.0), QPen(QColor(UMAP_ROI_COLOR), 2.0)):
            painter.setPen(pen)
            if closed:
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)

    def resizeEvent(self, event):  # noqa: N802
        if not self._fitted or not self._view_adjusted:
            self._fit_view()
        self._dirty = True
        super().resizeEvent(event)

    # -- interaction --------------------------------------------------------
    @staticmethod
    def _event_pos(event) -> QPointF:
        pos = event.position() if hasattr(event, "position") else event.localPos()
        return QPointF(pos)

    def wheelEvent(self, event):  # noqa: N802
        if self._coords.shape[0] == 0:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        pos = self._event_pos(event)
        anchor = self.view_to_data(pos.x(), pos.y())
        self._scale *= UMAP_WHEEL_ZOOM_STEP ** steps
        self._view_adjusted = True
        # Keep the data point under the cursor fixed while zooming.
        self._center[0] = anchor[0] - (pos.x() - self.width() / 2.0) / self._scale
        self._center[1] = anchor[1] + (pos.y() - self.height() / 2.0) / self._scale
        self._dirty = True
        self.update()
        event.accept()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self._lasso_mode:
            pos = self._event_pos(event)
            self._lasso_points = [tuple(self.view_to_data(pos.x(), pos.y()))]
            self._lasso_last_px = pos
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._press_pos = self._event_pos(event)
            self._last_drag_pos = self._press_pos
            self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        pos = self._event_pos(event)
        if self._lasso_points is not None and (event.buttons() & Qt.LeftButton):
            last = self._lasso_last_px
            step = pos - last if last is not None else pos
            if last is None or (step.x() ** 2 + step.y() ** 2) ** 0.5 >= UMAP_LASSO_MIN_STEP_PX:
                self._lasso_points.append(tuple(self.view_to_data(pos.x(), pos.y())))
                self._lasso_last_px = pos
                self.update()
            return
        if self._press_pos is not None and (event.buttons() & Qt.LeftButton):
            delta = pos - self._press_pos
            if not self._dragging and (delta.x() ** 2 + delta.y() ** 2) ** 0.5 > UMAP_CLICK_DRAG_THRESHOLD_PX:
                self._dragging = True
                self.setCursor(Qt.ClosedHandCursor)
            if self._dragging and self._last_drag_pos is not None:
                step = pos - self._last_drag_pos
                self._center[0] -= step.x() / self._scale
                self._center[1] += step.y() / self._scale
                self._view_adjusted = True
                self._dirty = True
                self.update()
            self._last_drag_pos = pos
            return
        row = self.nearest_row(pos.x(), pos.y())
        if row != self._hover_row:
            self._hover_row = row
            self.point_hovered.emit(int(row))
            self.update()
        if row >= 0 and self._hover_text_fn is not None:
            text = self._hover_text_fn(row)
            if text:
                global_pos = self.mapToGlobal(pos.toPoint())
                QToolTip.showText(global_pos, text, self)
        elif row < 0:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self._lasso_points is not None:
            pos = self._event_pos(event)
            points = list(self._lasso_points)
            points.append(tuple(self.view_to_data(pos.x(), pos.y())))
            self._lasso_points = None
            self._lasso_last_px = None
            self.update()
            polygon = np.asarray(points, dtype=np.float64)
            if polygon.shape[0] >= 3 and np.ptp(polygon, axis=0).min() > 0:
                self.lasso_completed.emit(polygon)
            return
        if event.button() != Qt.LeftButton or self._press_pos is None:
            super().mouseReleaseEvent(event)
            return
        was_drag = self._dragging
        self._press_pos = None
        self._last_drag_pos = None
        self._dragging = False
        self.setCursor(Qt.CrossCursor)
        if was_drag:
            return
        pos = self._event_pos(event)
        row = self.nearest_row(pos.x(), pos.y())
        if row >= 0:
            self.point_clicked.emit(int(row))

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if not self._lasso_mode:
            self.reset_view()
        event.accept()

    def leaveEvent(self, event):  # noqa: N802
        if self._hover_row != -1:
            self._hover_row = -1
            self.point_hovered.emit(-1)
            self.update()
        super().leaveEvent(event)


def _format_bound(value: float) -> str:
    value = float(value)
    if abs(value) >= 1e4:
        return f"{value:,.0f}"
    return f"{value:.4g}"


class UmapColorBar(QWidget):
    """Vertical colour bar with the clip range for a statistic colouring."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut: np.ndarray | None = None
        self._low = 0.0
        self._high = 1.0
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def set_range(self, lut, low: float, high: float) -> None:
        self._lut = np.asarray(lut, dtype=np.float64)
        self._low = float(low)
        self._high = float(high)
        self.update()

    def sizeHint(self):  # noqa: N802
        return QSize(120, 220)

    def paintEvent(self, _event):  # noqa: N802
        if self._lut is None or self._lut.shape[0] == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        bar = QRectF(8, 12, 18, max(20, self.height() - 64))
        gradient = QLinearGradient(bar.bottomLeft(), bar.topLeft())
        n = self._lut.shape[0]
        for i in range(0, n, max(1, n // 32)):
            r, g, b = (int(round(255 * float(c))) for c in self._lut[i, :3])
            gradient.setColorAt(i / max(1, n - 1), QColor(r, g, b))
        r, g, b = (int(round(255 * float(c))) for c in self._lut[-1, :3])
        gradient.setColorAt(1.0, QColor(r, g, b))
        painter.fillRect(bar, gradient)
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
        painter.drawRect(bar)
        painter.setPen(QColor(230, 232, 236))
        painter.drawText(QPointF(bar.right() + 6, bar.top() + 9), _format_bound(self._high))
        painter.drawText(QPointF(bar.right() + 6, bar.bottom()), _format_bound(self._low))
        painter.setPen(QColor(154, 164, 178))
        note = QRectF(0, bar.bottom() + 10, self.width(), 44)
        painter.drawText(note, Qt.AlignLeft | Qt.TextWordWrap, "Clipped to the 1st–99th percentile")
        painter.end()


class UmapPieChart(QWidget):
    """Donut chart of a region's cell-type make-up with the total in the middle.

    Slices start at 12 o'clock and run clockwise in the given order (largest
    first), outlined in a dark stroke like the Cell inspector's gene pie.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slices: list[UmapRoiSlice] = []
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_slices(self, slices: Iterable[UmapRoiSlice]) -> None:
        self._slices = [item for item in slices if int(item.count) > 0]
        self.update()

    def sizeHint(self):  # noqa: N802
        return QSize(200, 170)

    def paintEvent(self, _event):  # noqa: N802
        total = sum(int(item.count) for item in self._slices)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        side = min(self.width(), self.height()) - 8
        if side <= 0:
            painter.end()
            return
        rect = QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)
        if total <= 0:
            painter.setPen(QPen(QColor(90, 96, 108), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(rect)
        start_deg = 90.0
        for item in self._slices:
            span_deg = 360.0 * int(item.count) / total
            r, g, b = (float(c) for c in tuple(item.rgba)[:3])
            painter.setBrush(QBrush(QColor.fromRgbF(r, g, b, 1.0)))
            painter.setPen(QPen(QColor(20, 22, 28), 1))
            if len(self._slices) == 1:
                painter.drawEllipse(rect)
            else:
                painter.drawPie(rect, int(round(start_deg * 16)), int(round(-span_deg * 16)))
            start_deg -= span_deg
        hole = side * 0.52
        centre = rect.center()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(20, 22, 28)))
        painter.drawEllipse(centre, hole / 2.0, hole / 2.0)
        painter.setPen(QColor(240, 240, 240))
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(8.0, side * 0.085))
        painter.setFont(font)
        text_rect = QRectF(centre.x() - hole / 2.0, centre.y() - hole / 2.0, hole, hole)
        painter.drawText(text_rect, Qt.AlignCenter, f"{total:,}\ncells")
        painter.end()


class UmapRoiPanel(QFrame):
    """Right-hand summary of the region of interest: donut + per-type table."""

    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("UmapRoiPanel")
        self.setStyleSheet(
            "#UmapRoiPanel { background: #1b1e25; border: 1px solid #2c313c; border-radius: 4px; }"
        )
        self._title = QLabel("Region of interest")
        self._title.setStyleSheet("font-weight: bold;")
        self._subtitle = QLabel("")
        self._subtitle.setWordWrap(True)
        self._subtitle.setStyleSheet("color: #9aa4b2;")
        self._pie = UmapPieChart()
        self._table = QListWidget()
        self._table.setSelectionMode(QListWidget.NoSelection)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setWordWrap(True)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.setMaximumHeight(160)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #c9ced8;")
        self._clear_button = QPushButton("Clear region")
        self._clear_button.clicked.connect(self.clear_requested)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._pie)
        layout.addWidget(self._table)
        layout.addWidget(self._note)
        layout.addWidget(self._clear_button)
        self.setLayout(layout)

    @property
    def pie(self) -> UmapPieChart:
        return self._pie

    def set_compact(self, compact: bool) -> None:
        """Smaller donut and table, for two stacked panels in a paired window."""
        self._pie.setMinimumSize(110, 110 if compact else 150)
        self._pie.setFixedHeight(120 if compact else 170)
        self._table.setMaximumHeight(96 if compact else 160)

    def rows_text(self) -> list[str]:
        return [self._table.item(i).text() for i in range(self._table.count())]

    def set_summary(self, summary: UmapRoiSummary) -> None:
        self._subtitle.setText(
            f"{summary.total:,} cells drawn in {summary.drawn_in} · {summary.level}"
        )
        self._pie.set_slices(summary.slices)
        self._table.clear()
        for item in summary.slices:
            share = 100.0 * item.count / summary.total if summary.total else 0.0
            share_text = "<0.1%" if 0 < share < 0.05 else f"{share:.1f}%"
            row = QListWidgetItem(
                QIcon(_swatch_pixmap(item.rgba)),
                f"{item.label}  {item.count:,} · {share_text}",
            )
            row.setToolTip(item.label)
            self._table.addItem(row)
        self._note.setText(summary.note)
        self._note.setVisible(bool(summary.note))


#: Panel headings for paired views (top to bottom).
UMAP_PLATFORM_HEADINGS = {"MERSCOPE": "MERSCOPE", "XENIUM": "Xenium"}


class UmapPanel(QWidget):
    """One embedding's view: heading, plot, region summary, legend, readout.

    The standalone window shows one panel; a paired session shows one per
    platform, stacked (MERSCOPE above Xenium).
    """

    point_clicked = Signal(int)
    lasso_completed = Signal(object)
    roi_clear_requested = Signal()

    def __init__(self, heading: str = "", parent=None):
        super().__init__(parent)
        self._cell_ids = np.empty(0, dtype=np.int64)
        self._colouring: UmapColouring | None = None

        self._heading = QLabel(str(heading))
        self._heading.setStyleSheet("font-weight: bold; font-size: 14pt; color: #ffffff;")
        self._heading.setVisible(bool(heading))
        self._title = QLabel("UMAP")
        self._title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet("color: #9aa4b2;")
        self._subtitle.setWordWrap(True)
        self._canvas = UmapPlotCanvas()
        self._canvas.point_clicked.connect(self.point_clicked)
        self._canvas.point_hovered.connect(self._on_hovered)
        self._canvas.set_hover_text_callback(self.describe_row)
        self._canvas.lasso_completed.connect(self.lasso_completed)
        self._roi_panel = UmapRoiPanel()
        self._roi_panel.clear_requested.connect(self.roi_clear_requested)
        self._roi_panel.setFixedWidth(240)
        self._roi_panel.setVisible(False)
        self._legend = QListWidget()
        self._legend.setFixedWidth(240)
        self._legend.setSelectionMode(QListWidget.NoSelection)
        self._legend.setFocusPolicy(Qt.NoFocus)
        self._legend.setWordWrap(True)
        self._legend.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._legend.setResizeMode(QListWidget.Adjust)
        self._colorbar = UmapColorBar()
        self._colorbar.setFixedWidth(150)
        self._colorbar.setVisible(False)
        self._hover_label = QLabel("")
        self._hover_label.setStyleSheet("color: #e6e8ec;")

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(self._heading)
        header.addWidget(self._title)
        header.addWidget(self._subtitle)
        body = QHBoxLayout()
        body.setSpacing(6)
        body.addWidget(self._canvas, stretch=1)
        side = QVBoxLayout()
        side.setSpacing(4)
        side.addWidget(self._roi_panel)
        side.addWidget(self._legend, stretch=1)
        side.addWidget(self._colorbar, stretch=1)
        body.addLayout(side)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addLayout(header)
        outer.addLayout(body, stretch=1)
        outer.addWidget(self._hover_label)
        self.setLayout(outer)

    # -- public API ---------------------------------------------------------
    @property
    def heading(self) -> str:
        return self._heading.text()

    @property
    def canvas(self) -> UmapPlotCanvas:
        return self._canvas

    @property
    def colouring(self) -> UmapColouring | None:
        return self._colouring

    @property
    def roi_panel(self) -> UmapRoiPanel:
        return self._roi_panel

    def set_compact(self, compact: bool) -> None:
        self._roi_panel.set_compact(compact)

    def set_roi_summary(self, summary: UmapRoiSummary | None) -> None:
        """Show the region panel for ``summary`` (``None`` hides it)."""
        if summary is None:
            self._roi_panel.setVisible(False)
            return
        self._roi_panel.set_summary(summary)
        self._roi_panel.setVisible(True)

    def set_message(self, title: str, message: str) -> None:
        """Replace the plot with an explanatory message (loading, missing...)."""
        self._title.setText(str(title))
        self._subtitle.setText("")
        self._cell_ids = np.empty(0, dtype=np.int64)
        self._colouring = None
        self._canvas.clear()
        self._canvas.set_message(str(message))
        self._legend.clear()
        self._legend.setVisible(False)
        self._colorbar.setVisible(False)
        self._roi_panel.setVisible(False)
        self._hover_label.setText("")

    def set_embedding(self, cell_ids, coords) -> None:
        """Show a new embedding (resets zoom); colours follow via set_colouring."""
        self._cell_ids = np.asarray(cell_ids)
        self._colouring = None
        self._canvas.set_message("")
        self._canvas.set_embedding(coords)

    def set_colouring(self, colouring: UmapColouring, title: str, subtitle: str) -> None:
        self._colouring = colouring
        self._title.setText(str(title))
        self._subtitle.setText(str(subtitle))
        self._canvas.set_colouring(colouring.colors, colouring.order, colouring.n_background)
        self._legend.clear()
        if colouring.colorbar is not None:
            lut, low, high = colouring.colorbar
            self._colorbar.set_range(lut, low, high)
            self._colorbar.setVisible(True)
            self._legend.setVisible(False)
        else:
            self._colorbar.setVisible(False)
            self._legend.setVisible(True)
            for entry in colouring.legend:
                suffix = "" if entry.enabled else "  (hidden)"
                item = QListWidgetItem(
                    QIcon(_swatch_pixmap(entry.rgba, muted=not entry.enabled)),
                    f"{entry.label}  ({entry.count:,}){suffix}",
                )
                if not entry.enabled:
                    item.setForeground(QColor(130, 136, 148))
                item.setToolTip(entry.label)
                self._legend.addItem(item)

    def set_highlights(self, highlights: Iterable[tuple[int, str]]) -> None:
        self._canvas.set_highlights(highlights)

    def set_point_radius(self, radius_px: float) -> None:
        self._canvas.set_point_radius(radius_px)

    def describe_row(self, row: int) -> str:
        if not (0 <= int(row) < self._cell_ids.shape[0]):
            return ""
        text = f"Cell {int(self._cell_ids[int(row)])}"
        colouring = self._colouring
        if colouring is None:
            return text
        if colouring.values is not None:
            value = float(colouring.values[int(row)])
            shown = f"{value:.4g}" if np.isfinite(value) else "no value"
            return f"{text} · {colouring.title}: {shown}"
        if colouring.hover_text is not None:
            return f"{text} · {colouring.hover_text[int(row)]}"
        return text

    def _on_hovered(self, row: int) -> None:
        self._hover_label.setText(self.describe_row(row) if row >= 0 else "")


class UmapWindow(QWidget):
    """The optional floating UMAP window.

    A ``Qt.Tool`` window parented to the napari main window, so it floats above
    the viewer without taking a dock slot. Closing it only hides it; the UMAP
    tab re-opens the same instance. It holds one :class:`UmapPanel` per view:
    ``""`` for a standalone dataset, or ``"MERSCOPE"`` above ``"XENIUM"`` for a
    pair. Panel signals are re-emitted with the panel's key.
    """

    panel_point_clicked = Signal(str, int)
    panel_lasso_completed = Signal(str, object)
    roi_clear_requested = Signal()
    visibility_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("UMAP")
        self.setObjectName("UmapWindow")
        self.setStyleSheet(
            "#UmapWindow { background: #14161c; }"
            " #UmapWindow QLabel { color: #f0f0f0; background: transparent; }"
            " #UmapWindow QListWidget { background: #1b1e25; color: #e6e8ec; border: none; }"
        )
        self.resize(820, 580)
        self._panels: dict[str, UmapPanel] = {}
        self._lasso_mode = False
        self._point_radius = UMAP_DEFAULT_POINT_SIZE / 2.0

        self._panel_layout = QVBoxLayout()
        self._panel_layout.setContentsMargins(0, 0, 0, 0)
        self._panel_layout.setSpacing(10)
        hint = QLabel(
            "Scroll to zoom · drag to pan · double-click to reset · click a cell "
            "to select it in the spatial view · use Draw region of interest in "
            "the UMAP tab to lasso cells"
        )
        hint.setStyleSheet("color: #9aa4b2; font-size: 9pt;")
        hint.setWordWrap(True)
        self._reset_button = QPushButton("Reset view")
        self._reset_button.clicked.connect(self._reset_views)
        footer = QHBoxLayout()
        footer.addWidget(hint, stretch=1)
        footer.addWidget(self._reset_button)
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addLayout(self._panel_layout, stretch=1)
        outer.addLayout(footer)
        self.setLayout(outer)
        self.set_panel_keys(("",))

    # -- panels -------------------------------------------------------------
    def panel_keys(self) -> tuple[str, ...]:
        return tuple(self._panels)

    def set_panel_keys(self, keys: Iterable[str]) -> None:
        """Show one panel per key, rebuilding only when the set changes."""
        keys = tuple(str(key) for key in keys) or ("",)
        if keys == tuple(self._panels):
            return
        for panel in self._panels.values():
            self._panel_layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._panels = {}
        compact = len(keys) > 1
        for key in keys:
            panel = UmapPanel(UMAP_PLATFORM_HEADINGS.get(key, key) if compact else "")
            panel.set_compact(compact)
            panel.set_point_radius(self._point_radius)
            panel.canvas.set_lasso_mode(self._lasso_mode)
            panel.point_clicked.connect(lambda row, k=key: self.panel_point_clicked.emit(k, row))
            panel.lasso_completed.connect(
                lambda polygon, k=key: self.panel_lasso_completed.emit(k, polygon)
            )
            panel.roi_clear_requested.connect(self.roi_clear_requested)
            self._panel_layout.addWidget(panel, stretch=1)
            self._panels[key] = panel
        if compact and self.height() < 880:
            self.resize(max(self.width(), 880), 920)

    def panel(self, key: str | None = None) -> UmapPanel:
        """The panel for ``key`` (the first panel when ``key`` is ``None``)."""
        if key is None or key not in self._panels:
            return next(iter(self._panels.values()))
        return self._panels[key]

    def panels(self) -> dict[str, UmapPanel]:
        return dict(self._panels)

    def set_lasso_mode(self, enabled: bool) -> None:
        self._lasso_mode = bool(enabled)
        for panel in self._panels.values():
            panel.canvas.set_lasso_mode(self._lasso_mode)

    def set_point_radius(self, radius_px: float) -> None:
        self._point_radius = float(radius_px)
        for panel in self._panels.values():
            panel.set_point_radius(self._point_radius)

    def _reset_views(self) -> None:
        for panel in self._panels.values():
            panel.canvas.reset_view()

    # -- single-panel conveniences (the first panel) ------------------------
    @property
    def canvas(self) -> UmapPlotCanvas:
        return self.panel().canvas

    @property
    def colouring(self) -> UmapColouring | None:
        return self.panel().colouring

    @property
    def roi_panel(self) -> UmapRoiPanel:
        return self.panel().roi_panel

    def describe_row(self, row: int) -> str:
        return self.panel().describe_row(row)

    # Spontaneous show/hide events come from the window system (minimising,
    # or macOS hiding tool windows while the app is in the background); only
    # explicit opens/closes should flip the tab's Open/Close button.
    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not event.spontaneous():
            self.visibility_changed.emit(True)

    def hideEvent(self, event):  # noqa: N802
        super().hideEvent(event)
        if not event.spontaneous():
            self.visibility_changed.emit(False)


class UmapControlWidget(QWidget):
    """The "UMAP" tab: open the window and choose its source and colouring.

    Callbacks (all argument-free of dataset names -- the controller acts on the
    active standalone dataset):

    - ``set_window_visible_callback(bool)``
    - ``set_source_callback(source_key)``
    - ``set_colour_callback(mode, channel, statistic, colormap)``
    - ``set_point_size_callback(radius_px)``
    - ``set_lasso_active_callback(bool)`` -- arm/disarm region drawing
    - ``clear_roi_callback()``
    """

    def __init__(
        self,
        set_window_visible_callback,
        set_source_callback,
        set_colour_callback,
        set_point_size_callback,
        set_lasso_active_callback=None,
        clear_roi_callback=None,
    ):
        super().__init__()
        self._set_window_visible_callback = set_window_visible_callback
        self._set_source_callback = set_source_callback
        self._set_colour_callback = set_colour_callback
        self._set_point_size_callback = set_point_size_callback
        self._set_lasso_active_callback = set_lasso_active_callback
        self._clear_roi_callback = clear_roi_callback
        self._suppress = False
        self._statistics_available = False

        description = QLabel(
            "Shows each cell's stored UMAP embedding in a floating window. "
            "Selecting a cell in the UMAP selects it in the spatial view, and "
            "cells selected in the spatial view are ringed in the UMAP."
        )
        description.setWordWrap(True)
        self._window_button = QPushButton("Open UMAP window")
        self._window_button.setCheckable(True)
        self._window_button.setCursor(Qt.PointingHandCursor)
        self._window_button.toggled.connect(self._on_window_toggled)

        self._source_combo = QComboBox()
        for source in CELL_TYPE_SOURCES:
            self._source_combo.addItem(source.label, source.key)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)

        self._colour_combo = QComboBox()
        for mode, label in UMAP_COLOUR_MODES:
            self._colour_combo.addItem(label, mode)
        self._colour_combo.currentIndexChanged.connect(self._on_colour_changed)

        self._channel_combo = QComboBox()
        self._statistic_combo = QComboBox()
        self._colormap_combo = QComboBox()
        self._colormap_combo.addItems(list(UMAP_COLORMAPS))
        for combo in (self._channel_combo, self._statistic_combo, self._colormap_combo):
            combo.currentIndexChanged.connect(self._on_colour_changed)
        self._statistic_box = QWidget()
        stat_layout = QVBoxLayout()
        stat_layout.setContentsMargins(12, 0, 0, 0)
        stat_layout.setSpacing(4)
        stat_layout.addWidget(QLabel("Channel"))
        stat_layout.addWidget(self._channel_combo)
        stat_layout.addWidget(QLabel("Statistic"))
        stat_layout.addWidget(self._statistic_combo)
        stat_layout.addWidget(QLabel("Colormap"))
        stat_layout.addWidget(self._colormap_combo)
        self._statistic_box.setLayout(stat_layout)
        self._statistic_note = QLabel(
            "Per-cell statistics are measured on the Cellpose masks, so they "
            "colour the Cellpose clustering's UMAP."
        )
        self._statistic_note.setWordWrap(True)
        self._statistic_note.setStyleSheet("color: #9aa4b2;")

        self._sync_check = QCheckBox("Follow the Cell type labels and Per cell statistics tabs")
        self._sync_check.setChecked(True)
        self._sync_check.setToolTip(
            "When ticked, choosing a source, Broad / Fine, or loading a per-cell "
            "statistic overlay on those tabs recolours the UMAP to match."
        )
        self._centre_check = QCheckBox("Centre the spatial view on cells picked in the UMAP")
        self._centre_check.setChecked(True)

        self._point_slider = QSlider(Qt.Horizontal)
        self._point_slider.setRange(*UMAP_POINT_SIZE_RANGE)
        self._point_slider.setValue(UMAP_DEFAULT_POINT_SIZE)
        self._point_slider.valueChanged.connect(self._on_point_size_changed)

        self._lasso_button = QPushButton("Draw region of interest")
        self._lasso_button.setCheckable(True)
        self._lasso_button.setCursor(Qt.PointingHandCursor)
        self._lasso_button.setToolTip(
            "Then hold the mouse button and drag around cells in the spatial view "
            "or the UMAP; releasing joins the end back to the start."
        )
        self._lasso_button.toggled.connect(self._on_lasso_toggled)
        self._clear_roi_button = QPushButton("Clear region of interest")
        self._clear_roi_button.setEnabled(False)
        self._clear_roi_button.clicked.connect(self._on_clear_roi)
        lasso_hint = QLabel(
            "Hold the mouse button and drag around cells in the spatial view or the "
            "UMAP. Releasing closes the outline; its cells are outlined in the "
            "spatial view, emphasised in the UMAP, and summarised in a pie chart."
        )
        lasso_hint.setWordWrap(True)
        lasso_hint.setStyleSheet("color: #9aa4b2;")

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QLabel("UMAP embedding"))
        layout.addWidget(description)
        layout.addWidget(self._window_button)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        layout.addWidget(QLabel("Cell type source"))
        layout.addWidget(self._source_combo)
        layout.addWidget(QLabel("Colour by"))
        layout.addWidget(self._colour_combo)
        layout.addWidget(self._statistic_box)
        layout.addWidget(self._statistic_note)
        point_row = QHBoxLayout()
        point_row.addWidget(QLabel("Point size"))
        point_row.addWidget(self._point_slider)
        layout.addLayout(point_row)
        layout.addWidget(self._sync_check)
        layout.addWidget(self._centre_check)
        roi_line = QFrame()
        roi_line.setFrameShape(QFrame.HLine)
        roi_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(roi_line)
        layout.addWidget(QLabel("Region of interest"))
        layout.addWidget(self._lasso_button)
        layout.addWidget(self._clear_roi_button)
        layout.addWidget(lasso_hint)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self.setLayout(layout)
        self.set_statistic_options([], [], False)
        self._refresh_statistic_box()

    # -- public API ---------------------------------------------------------
    @property
    def follow_other_tabs(self) -> bool:
        return self._sync_check.isChecked()

    @property
    def centre_on_pick(self) -> bool:
        return self._centre_check.isChecked()

    @property
    def point_radius(self) -> float:
        return self._point_slider.value() / 2.0

    def current_source(self) -> str:
        return str(self._source_combo.currentData() or "")

    def current_colour(self) -> tuple[str, str, str, str]:
        return (
            str(self._colour_combo.currentData() or "broad"),
            str(self._channel_combo.currentText()),
            str(self._statistic_combo.currentText()),
            str(self._colormap_combo.currentText()),
        )

    def set_status(self, text: str) -> None:
        self._status_label.setText(str(text))

    def set_window_open(self, is_open: bool) -> None:
        self._window_button.blockSignals(True)
        self._window_button.setChecked(bool(is_open))
        self._window_button.setText("Close UMAP window" if is_open else "Open UMAP window")
        self._window_button.blockSignals(False)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._window_button,
            self._source_combo,
            self._colour_combo,
            self._statistic_box,
            self._point_slider,
            self._lasso_button,
        ):
            widget.setEnabled(bool(enabled))
        if not enabled:
            self.set_lasso_active(False)
            self.set_roi_available(False)

    @property
    def lasso_active(self) -> bool:
        return self._lasso_button.isChecked()

    def set_lasso_active(self, active: bool) -> None:
        """Reflect the controller's drawing state without re-emitting it."""
        self._lasso_button.blockSignals(True)
        self._lasso_button.setChecked(bool(active))
        self._lasso_button.setText(
            "Drawing… drag in either view (click to cancel)" if active else "Draw region of interest"
        )
        self._lasso_button.blockSignals(False)

    def set_roi_available(self, available: bool) -> None:
        self._clear_roi_button.setEnabled(bool(available))

    def set_sources(self, available: Mapping[str, bool], current: str | None) -> None:
        """Offer only the cell-type sources whose clustering table is stored."""
        self._suppress = True
        try:
            self._source_combo.clear()
            for source in CELL_TYPE_SOURCES:
                if available.get(source.key, False):
                    self._source_combo.addItem(source.label, source.key)
            index = self._source_combo.findData(current) if current else -1
            if index >= 0:
                self._source_combo.setCurrentIndex(index)
            self._source_combo.setEnabled(self._source_combo.count() > 0)
        finally:
            self._suppress = False

    def set_colour_levels(self, levels: Mapping[str, str], current: str) -> None:
        """Offer only annotation levels present in the selected source."""
        self._suppress = True
        try:
            self._colour_combo.clear()
            for key, label in levels.items():
                self._colour_combo.addItem(label, key)
            self._colour_combo.addItem("Per-cell statistic", "statistic")
            self._colour_combo.model().item(self._colour_combo.count() - 1).setEnabled(
                self._statistics_available
            )
            index = self._colour_combo.findData(current)
            self._colour_combo.setCurrentIndex(index)
        finally:
            self._suppress = False
        self._refresh_statistic_box()

    def set_statistic_options(
        self, channels: list[str], statistics: list[str], available: bool
    ) -> None:
        self._suppress = True
        try:
            channel = self._channel_combo.currentText()
            statistic = self._statistic_combo.currentText()
            self._channel_combo.clear()
            self._statistic_combo.clear()
            self._channel_combo.addItems([str(c) for c in channels])
            self._statistic_combo.addItems([str(s) for s in statistics])
            if channel in channels:
                self._channel_combo.setCurrentText(channel)
            elif "DAPI" in channels:
                self._channel_combo.setCurrentText("DAPI")
            if statistic in statistics:
                self._statistic_combo.setCurrentText(statistic)
            elif "median" in statistics:
                self._statistic_combo.setCurrentText("median")
            self._statistics_available = bool(available and channels and statistics)
            stat_index = self._colour_combo.findData("statistic")
            item = self._colour_combo.model().item(stat_index) if stat_index >= 0 else None
            if item is not None:
                item.setEnabled(self._statistics_available)
            if not self._statistics_available and self._colour_combo.currentData() == "statistic":
                self._colour_combo.setCurrentIndex(self._colour_combo.findData("broad"))
        finally:
            self._suppress = False
        self._refresh_statistic_box()

    def set_selection(
        self,
        source: str | None = None,
        colour_by: str | None = None,
        channel: str | None = None,
        statistic: str | None = None,
        colormap: str | None = None,
    ) -> None:
        """Reflect a controller-side change without re-emitting callbacks."""
        self._suppress = True
        try:
            if source is not None:
                index = self._source_combo.findData(source)
                if index >= 0:
                    self._source_combo.setCurrentIndex(index)
            if colour_by is not None:
                index = self._colour_combo.findData(colour_by)
                if index >= 0:
                    self._colour_combo.setCurrentIndex(index)
            if channel:
                self._channel_combo.setCurrentText(str(channel))
            if statistic:
                self._statistic_combo.setCurrentText(str(statistic))
            if colormap:
                self._colormap_combo.setCurrentText(str(colormap))
        finally:
            self._suppress = False
        self._refresh_statistic_box()

    # -- handlers -----------------------------------------------------------
    def _refresh_statistic_box(self) -> None:
        is_statistic = self._colour_combo.currentData() == "statistic"
        self._statistic_box.setVisible(is_statistic)
        self._statistic_box.setEnabled(self._statistics_available)
        self._statistic_note.setVisible(is_statistic)

    def _on_window_toggled(self, checked: bool) -> None:
        self._window_button.setText("Close UMAP window" if checked else "Open UMAP window")
        if self._suppress:
            return
        self._set_window_visible_callback(bool(checked))

    def _on_source_changed(self, _index: int) -> None:
        if self._suppress:
            return
        source = self.current_source()
        if source:
            self._set_source_callback(source)

    def _on_colour_changed(self, _index: int) -> None:
        self._refresh_statistic_box()
        if self._suppress:
            return
        self._set_colour_callback(*self.current_colour())

    def _on_lasso_toggled(self, checked: bool) -> None:
        self.set_lasso_active(checked)
        if self._set_lasso_active_callback is not None:
            self._set_lasso_active_callback(bool(checked))

    def _on_clear_roi(self) -> None:
        if self._clear_roi_callback is not None:
            self._clear_roi_callback()

    def _on_point_size_changed(self, _value: int) -> None:
        if self._suppress:
            return
        self._set_point_size_callback(self.point_radius)
