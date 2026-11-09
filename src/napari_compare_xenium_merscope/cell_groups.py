"""Hand-curated cell groups: click Cellpose masks into named groups and export them.

The user builds groups of cells that share a feature they recognise visually.
Membership is keyed by the Cellpose mask ``label_id`` (the GeoDataFrame index of
``shapes/MOSAIK_cellpose`` and ``obs/label_id`` of the Cellpose image
quantification table), so the export can join each cell's stored per-channel
image statistics without recomputing anything from the images.

Pure helpers (statistics lookup, table building, CSV round trip) live beside the
Qt "Cell groups" tab widget; the controller in :mod:`viewer` owns the state and
the napari layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import zarr
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .utils import (
    CELLPOSE_QUANTIFICATION_TABLE_KEY,
    CELLPOSE_SHAPE_KEY,
    CellposeQuantificationFeature,
    cellpose_quantification_channel_labels,
    cellpose_quantification_features,
    open_zarr_group_unconsolidated,
)

#: Segmentation whose masks are grouped (the one the statistics are measured on).
CELL_GROUP_SHAPE_KEY = CELLPOSE_SHAPE_KEY
#: Napari layer-name prefix; one Shapes layer per group.
CELL_GROUP_LAYER_PREFIX = "Cell group | "
#: Distinct, dark-background-friendly outline colours, assigned in order.
CELL_GROUP_COLORS = (
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
    "#fabebe",  # pink
)
#: Identity columns that lead every exported row (statistics follow).
CELL_GROUP_ID_COLUMNS = (
    "group",
    "dataset",
    "store",
    "segmentation",
    "label_id",
    "centroid_x_um",
    "centroid_y_um",
    "area_um2",
)


@dataclass
class CellGroup:
    """One named group of Cellpose cells, in the order they were clicked."""

    name: str
    color: str
    #: Cellpose label id -> shapely geometry (global micron frame).
    cells: dict[int, object] = field(default_factory=dict)
    #: Label id owning each polygon of the group's napari layer, in layer order
    #: (a MultiPolygon cell contributes several polygons).
    path_owners: list[int] = field(default_factory=list)


def cell_group_layer_name(name: str) -> str:
    return f"{CELL_GROUP_LAYER_PREFIX}{name}"


def next_cell_group_name(existing: Iterable[str]) -> str:
    """Return the first free ``Group N`` name."""
    taken = {str(name).casefold() for name in existing}
    n = 1
    while f"group {n}" in taken:
        n += 1
    return f"Group {n}"


def next_cell_group_color(groups: Iterable[CellGroup]) -> str:
    """Return the first palette colour no group uses, cycling when exhausted."""
    groups = list(groups)
    used = {group.color.casefold() for group in groups}
    for color in CELL_GROUP_COLORS:
        if color.casefold() not in used:
            return color
    return CELL_GROUP_COLORS[len(groups) % len(CELL_GROUP_COLORS)]


def cellpose_statistic_column_names(features: list[CellposeQuantificationFeature]) -> list[str]:
    """``<channel>__<statistic>``, image-qualified when a channel name is shared."""
    labels = cellpose_quantification_channel_labels(features)
    return [
        f"{feature.channel}__{feature.statistic}"
        if label == feature.channel
        else f"{feature.image_key}__{feature.channel}__{feature.statistic}"
        for feature, label in zip(features, labels, strict=True)
    ]


def cellpose_statistics_for_labels(zarr_path: str | Path, label_ids) -> pd.DataFrame:
    """Read every stored image statistic for the given Cellpose label ids.

    Only the requested rows of ``X`` are read. The result is indexed by
    ``label_id`` in the requested order; ids absent from the table get ``NaN``.
    """
    wanted = np.asarray(list(label_ids), dtype=np.int64)
    features = cellpose_quantification_features(zarr_path)
    columns = cellpose_statistic_column_names(features)
    root = open_zarr_group_unconsolidated(zarr_path)
    table = root[f"tables/{CELLPOSE_QUANTIFICATION_TABLE_KEY}"]
    matrix = table["X"]
    if isinstance(matrix, zarr.Group):
        raise ValueError(
            f"{CELLPOSE_QUANTIFICATION_TABLE_KEY} stores a sparse X; only dense statistics are supported."
        )

    table_ids = pd.Index(np.asarray(table["obs/label_id"][:], dtype=np.int64))
    first = ~table_ids.duplicated()
    row_of_id = pd.Series(np.flatnonzero(first), index=table_ids[first])
    positions = row_of_id.reindex(wanted).to_numpy(dtype=float)
    found = np.isfinite(positions)

    values = np.full((wanted.size, len(columns)), np.nan, dtype=np.float64)
    if found.any():
        rows = np.unique(positions[found].astype(np.int64))
        block = np.asarray(matrix.get_orthogonal_selection((rows, slice(None))), dtype=np.float64)
        if block.shape[1] != len(columns):
            raise ValueError(
                f"{CELLPOSE_QUANTIFICATION_TABLE_KEY}: X has {block.shape[1]} columns "
                f"but var describes {len(columns)}."
            )
        values[found] = block[np.searchsorted(rows, positions[found].astype(np.int64))]
    return pd.DataFrame(values, index=pd.Index(wanted, name="label_id"), columns=columns)


def cell_group_table(
    groups: Iterable[CellGroup],
    *,
    dataset: str,
    store: str = "",
    segmentation: str = CELL_GROUP_SHAPE_KEY,
    statistics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per grouped cell: identity, geometry summary, then statistics."""
    rows = []
    for group in groups:
        for label_id, geometry in group.cells.items():
            centroid = geometry.centroid
            rows.append(
                (
                    group.name,
                    str(dataset),
                    str(store),
                    str(segmentation),
                    int(label_id),
                    float(centroid.x),
                    float(centroid.y),
                    float(geometry.area),
                )
            )
    frame = pd.DataFrame(rows, columns=list(CELL_GROUP_ID_COLUMNS))
    if statistics is not None:
        unique = statistics[~statistics.index.duplicated()]
        stats = unique.reindex(frame["label_id"].to_numpy()).reset_index(drop=True)
        frame = pd.concat([frame, stats], axis=1)
    return frame


def read_cell_groups_csv(path: str | Path) -> list[tuple[str, list[int]]]:
    """Read ``(group, [label_id, ...])`` pairs back from an exported CSV.

    Groups keep their first-appearance order, and cells their row order. Rows
    from another segmentation are rejected rather than joined to the wrong masks.
    """
    frame = pd.read_csv(path)
    missing = [column for column in ("group", "label_id") if column not in frame.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing column(s): {', '.join(missing)}.")
    if "segmentation" in frame.columns:
        others = sorted(set(frame["segmentation"].dropna().astype(str)) - {CELL_GROUP_SHAPE_KEY})
        if others:
            raise ValueError(
                f"{Path(path).name} groups {', '.join(others)} cells; only {CELL_GROUP_SHAPE_KEY} is supported."
            )
    frame = frame.dropna(subset=["group", "label_id"])
    groups: dict[str, list[int]] = {}
    for name, label_id in zip(frame["group"].astype(str), frame["label_id"], strict=True):
        ids = groups.setdefault(name, [])
        label_id = int(label_id)
        if label_id not in ids:
            ids.append(label_id)
    return list(groups.items())


# ---------------------------------------------------------------------------
# Qt tab
# ---------------------------------------------------------------------------


def _color_icon(color: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


class CellGroupWidget(QWidget):
    """The "Cell groups" tab: build named groups of Cellpose cells and export them.

    Callbacks (the controller acts on the active standalone dataset):

    - ``set_picking_callback(bool)`` -- arm/disarm click-to-group
    - ``new_group_callback(name)``
    - ``rename_group_callback(old, new)``
    - ``delete_group_callback(name)``
    - ``clear_group_callback(name)``
    - ``set_active_group_callback(name)``
    - ``export_callback(path)``
    - ``load_callback(path)``
    """

    def __init__(
        self,
        set_picking_callback,
        new_group_callback,
        rename_group_callback,
        delete_group_callback,
        clear_group_callback,
        set_active_group_callback,
        export_callback,
        load_callback,
    ):
        super().__init__()
        self._set_picking_callback = set_picking_callback
        self._new_group_callback = new_group_callback
        self._rename_group_callback = rename_group_callback
        self._delete_group_callback = delete_group_callback
        self._clear_group_callback = clear_group_callback
        self._set_active_group_callback = set_active_group_callback
        self._export_callback = export_callback
        self._load_callback = load_callback
        self._suppress = False
        self._names: list[str] = []

        description = QLabel(
            "Click Cellpose masks to collect cells that share a feature. Clicking a "
            "grouped cell again removes it from the selected group only; a cell can "
            "belong to several groups. Export writes one row per cell and group "
            "with the cell's per-channel statistics."
        )
        description.setWordWrap(True)
        self._availability_label = QLabel("")
        self._availability_label.setWordWrap(True)

        self._pick_button = QPushButton("Pick cells for group")
        self._pick_button.setCheckable(True)
        self._pick_button.setCursor(Qt.PointingHandCursor)
        self._pick_button.toggled.connect(self._on_pick_toggled)

        self._group_list = QListWidget()
        self._group_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._group_list.currentRowChanged.connect(self._on_current_row_changed)
        self._group_list.itemDoubleClicked.connect(lambda _item: self._on_rename())

        self._new_button = QPushButton("New group")
        self._new_button.clicked.connect(self._on_new)
        self._rename_button = QPushButton("Rename")
        self._rename_button.clicked.connect(self._on_rename)
        self._clear_button = QPushButton("Clear cells")
        self._clear_button.clicked.connect(self._on_clear)
        self._delete_button = QPushButton("Delete group")
        self._delete_button.clicked.connect(self._on_delete)
        self._export_button = QPushButton("Export groups to CSV…")
        self._export_button.clicked.connect(self._on_export)
        self._load_button = QPushButton("Load groups from CSV…")
        self._load_button.clicked.connect(self._on_load)

        edit_row = QHBoxLayout()
        edit_row.addWidget(self._new_button)
        edit_row.addWidget(self._rename_button)
        manage_row = QHBoxLayout()
        manage_row.addWidget(self._clear_button)
        manage_row.addWidget(self._delete_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(description)
        layout.addWidget(self._availability_label)
        layout.addWidget(self._pick_button)
        layout.addWidget(QLabel("Groups (the selected group receives clicks)"))
        layout.addWidget(self._group_list, stretch=1)
        layout.addLayout(edit_row)
        layout.addLayout(manage_row)
        layout.addWidget(self._export_button)
        layout.addWidget(self._load_button)
        self.setLayout(layout)
        self.set_available(False, "Load a standalone dataset to group cells.")

    # -- controller -> widget ------------------------------------------------
    def set_available(self, available: bool, reason: str = "") -> None:
        self._availability_label.setText("" if available else reason)
        self._availability_label.setVisible(not available and bool(reason))
        for widget in (
            self._pick_button,
            self._group_list,
            self._new_button,
            self._load_button,
        ):
            widget.setEnabled(bool(available))
        if not available:
            self.set_picking(False)
        self._update_group_buttons(bool(available))

    def set_picking(self, active: bool) -> None:
        self._suppress = True
        try:
            self._pick_button.setChecked(bool(active))
            self._pick_button.setText(
                "Picking cells… (click to stop)" if active else "Pick cells for group"
            )
        finally:
            self._suppress = False

    def set_groups(self, groups: list[tuple[str, str, int]], active: str | None) -> None:
        """Show ``(name, colour, cell count)`` rows and select ``active``."""
        self._suppress = True
        try:
            self._group_list.clear()
            self._names = [name for name, _color, _count in groups]
            for name, color, count in groups:
                item = QListWidgetItem(_color_icon(color), f"{name}  ({count} cell{'s' if count != 1 else ''})")
                self._group_list.addItem(item)
            if active in self._names:
                self._group_list.setCurrentRow(self._names.index(active))
        finally:
            self._suppress = False
        self._update_group_buttons(self._pick_button.isEnabled())

    def active_group(self) -> str | None:
        row = self._group_list.currentRow()
        return self._names[row] if 0 <= row < len(self._names) else None

    def _update_group_buttons(self, available: bool) -> None:
        has_active = available and self.active_group() is not None
        for button in (self._rename_button, self._clear_button, self._delete_button):
            button.setEnabled(has_active)
        self._export_button.setEnabled(available and bool(self._names))

    # -- widget -> controller ------------------------------------------------
    def _on_pick_toggled(self, checked: bool) -> None:
        self._pick_button.setText(
            "Picking cells… (click to stop)" if checked else "Pick cells for group"
        )
        if not self._suppress:
            self._set_picking_callback(bool(checked))

    def _on_current_row_changed(self, _row: int) -> None:
        self._update_group_buttons(self._pick_button.isEnabled())
        name = self.active_group()
        if not self._suppress and name is not None:
            self._set_active_group_callback(name)

    def _on_new(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New cell group", "Group name:", text=next_cell_group_name(self._names)
        )
        if ok and name.strip():
            self._new_group_callback(name.strip())

    def _on_rename(self) -> None:
        old = self.active_group()
        if old is None:
            return
        new, ok = QInputDialog.getText(self, "Rename cell group", "Group name:", text=old)
        if ok and new.strip() and new.strip() != old:
            self._rename_group_callback(old, new.strip())

    def _on_clear(self) -> None:
        name = self.active_group()
        if name is not None:
            self._clear_group_callback(name)

    def _on_delete(self) -> None:
        name = self.active_group()
        if name is not None:
            self._delete_group_callback(name)

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export cell groups", "cell_groups.csv", "CSV files (*.csv)"
        )
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self._export_callback(path)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load cell groups", "", "CSV files (*.csv)"
        )
        if path:
            self._load_callback(path)
