"""Read-only, on-demand text exploration of SpatialData and its backing store."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

PAGE_SIZE = 100
ELEMENTS = ("images", "labels", "points", "shapes", "tables")


@dataclass
class _Store:
    path: Path
    sdata: object
    images_sdata: object | None


@dataclass
class _Page:
    entries: object


def _short(value, limit=240):
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _summary(value):
    if isinstance(value, _Store):
        return "SpatialData", str(value.path)
    if isinstance(value, _Page):
        return "", "Next 100 entries"
    if isinstance(value, zarr.Group):
        encoding = value.attrs.get("encoding-type", "Zarr group")
        if encoding == "anndata" and "X" in value:
            matrix = value["X"]
            shape = matrix.shape if isinstance(matrix, zarr.Array) else matrix.attrs.get("shape")
            return encoding, f"shape={tuple(shape)}" if shape is not None else "Stored table"
        if encoding in ("csr_matrix", "csc_matrix"):
            return encoding, f"shape={tuple(value.attrs.get('shape', ()))}"
        if encoding == "dataframe":
            index = value.attrs.get("_index", "_index")
            rows = value[index].shape[0] if index in value else "?"
            return encoding, f"shape=({rows}, {len(value.attrs.get('column-order', []))})"
        return encoding, "Stored element"
    if hasattr(value, "children") and hasattr(value, "dataset"):
        return type(value).__name__, f"{len(value.children):,} child groups"
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        return type(value).__name__, _short(dict(value.sizes))
    if isinstance(value, Mapping):
        return type(value).__name__, f"{len(value):,} entries"
    if isinstance(value, (list, tuple)):
        return type(value).__name__, f"{len(value):,} entries"
    if isinstance(value, pd.DataFrame):
        return type(value).__name__, f"shape={value.shape}"
    if isinstance(value, (pd.Series, pd.Index)):
        return type(value).__name__, f"shape={value.shape}, dtype={value.dtype}"
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return type(value).__name__, _short(value)
    # Do not call repr on lazy arrays/dataframes: it may trigger computation.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return type(value).__name__, f"shape={value.shape}, dtype={value.dtype}"
    if hasattr(value, "columns"):
        return type(value).__name__, f"{len(value.columns):,} columns (lazy)"
    if hasattr(value, "obs") and hasattr(value, "var"):
        return type(value).__name__, f"shape={value.shape}"
    return type(value).__name__, ""


def _expandable(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return False
    return isinstance(value, (_Store, _Page, Mapping, list, tuple,
                              zarr.Group, pd.DataFrame, pd.Series, pd.Index,
                              xr.DataArray, xr.Dataset)) or any(
        hasattr(value, attr) for attr in ("shape", "columns", "children", "obs")
    )


def _children(value):
    if isinstance(value, _Store):
        # Viewer startup intentionally omits tables and unused segmentations.
        # Fill those gaps from the read-only Zarr hierarchy, without loading X.
        group = zarr.open_group(str(value.path), mode="r", use_consolidated=False)
        for name in ELEMENTS:
            loaded = dict(getattr(value.sdata, name, {}))
            if name == "images" and value.images_sdata is not None:
                loaded.update(getattr(value.images_sdata, name, {}))
            if name in group:
                stored = group[name]
                for key in stored.keys():
                    if key not in loaded:
                        loaded[key] = stored[key]
            yield name, loaded
        yield "attrs", dict(getattr(value.sdata, "attrs", group.attrs))
    elif isinstance(value, zarr.Group):
        if len(value.attrs):
            yield "attrs", dict(value.attrs)
        for key in value.keys():
            yield str(key), value[key]
    elif hasattr(value, "obs") and hasattr(value, "var"):
        for name in ("X", "obs", "var", "uns", "obsm", "varm", "obsp", "varp", "layers", "raw"):
            yield name, getattr(value, name, None)
    elif hasattr(value, "children") and hasattr(value, "dataset"):
        yield "attrs", dict(value.attrs)
        yield "dataset", value.dataset
        yield from value.children.items()
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield f"[{index}]", child
    elif isinstance(value, pd.DataFrame):
        yield "index", value.index
        for index, column in enumerate(value.columns):
            yield str(column), value.iloc[:, index]
    elif isinstance(value, (pd.Series, pd.Index)):
        yield "shape", str(value.shape)
        yield "dtype", str(value.dtype)
        if isinstance(value.dtype, pd.CategoricalDtype):
            yield "categories", value.dtype.categories
    elif isinstance(value, xr.DataArray):
        yield "dims", list(value.dims)
        yield "coords", {name: coord.variable for name, coord in value.coords.items()}
        yield "attrs", dict(value.attrs)
        yield "data", value.data
    elif isinstance(value, xr.Dataset):
        yield "coords", {name: coord.variable for name, coord in value.coords.items()}
        yield "attrs", dict(value.attrs)
        yield "data_vars", dict(value.data_vars)
    elif isinstance(value, Mapping):
        for key in value:
            yield str(key), value[key]
    elif hasattr(value, "columns"):
        # Dask frames: schema only, never scan the transcript table.
        yield "columns", {str(name): str(dtype) for name, dtype in value.dtypes.items()}
        yield "note", "Lazy dataframe: values remain on disk."
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        yield "shape", str(value.shape)
        yield "dtype", str(value.dtype)
        for name in ("chunks", "nbytes", "attrs"):
            if hasattr(value, name):
                child = getattr(value, name)
                yield name, dict(child) if name == "attrs" else child
        yield "note", "Summary only; array values are not read."


class _Signals(QObject):
    finished = Signal(object, object, object)


class _Load(QRunnable):
    def __init__(self, token, value):
        super().__init__()
        self.signals = _Signals()
        self.token = token
        self.value = value

    def run(self):
        try:
            entries = self.value.entries if isinstance(self.value, _Page) else iter(_children(self.value))
            # One-entry lookahead is retained in the iterator, never discarded.
            rows = list(islice(entries, PAGE_SIZE))
            extra = list(islice(entries, 1))
            if extra:
                rows.append(("Show more…", _Page(chain(extra, entries))))
            result = [(name, child, _summary(child), _expandable(child)) for name, child in rows]
            self.signals.finished.emit(self.token, result, None)
        except Exception as exc:
            self.signals.finished.emit(self.token, [], str(exc))


class SpatialDataExplorerWidget(QWidget):
    """Independent roots for a standalone store or both stores of a pair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generation = 0
        self._stores = []
        self._jobs = {}
        self._values = {}
        self._loaded = set()
        self._next_id = 0
        layout = QVBoxLayout(self)
        label = QLabel("Expand a branch to inspect its contents. Stored elements fill gaps in the loaded object. Images, matrices and columns show summaries only.")
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda: self.set_stores(self._stores))
        collapse = QPushButton("Collapse all")
        collapse.clicked.connect(lambda: self.tree.collapseAll())
        row.addWidget(refresh)
        row.addWidget(collapse)
        row.addStretch()
        layout.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Value / summary"])
        self.tree.setColumnWidth(0, 210)
        self.tree.setColumnWidth(1, 120)
        self.tree.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemExpanded.connect(self._expand)
        layout.addWidget(self.tree)
        self.set_stores([])

    def set_stores(self, stores):
        """Replace roots and invalidate outstanding reads from an older dataset."""
        self._generation += 1
        self._stores = list(stores)
        self.tree.clear()
        self._values.clear()
        self._loaded.clear()
        if not stores:
            self.tree.addTopLevelItem(QTreeWidgetItem(["Load a dataset to explore its SpatialData."]))
        for name, path, sdata, images_sdata in stores:
            value = _Store(Path(path), sdata, images_sdata)
            self._add(None, name, value, _summary(value), True)

    def _add(self, parent, name, value, summary, expandable):
        item = QTreeWidgetItem([name, *summary])
        self._next_id += 1
        key = self._next_id
        item.setData(0, Qt.UserRole, key)
        self._values[key] = (item, value)
        for column, text in enumerate((name, *summary)):
            item.setToolTip(column, text)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        if expandable:
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        return item

    def _expand(self, item):
        key = item.data(0, Qt.UserRole)
        if key in self._loaded or key not in self._values:
            return
        self._loaded.add(key)
        token = (self._generation, key)
        job = _Load(token, self._values[key][1])
        job.signals.finished.connect(self._finished)
        self._jobs[token] = job
        item.addChild(QTreeWidgetItem(["Loading…"]))
        QThreadPool.globalInstance().start(job)

    def _finished(self, token, rows, error):
        self._jobs.pop(token, None)
        generation, key = token
        if generation != self._generation or key not in self._values:
            return
        item = self._values[key][0]
        item.takeChildren()
        if error:
            item.addChild(QTreeWidgetItem(["Could not read branch", "", error]))
        elif not rows:
            item.addChild(QTreeWidgetItem(["(empty)"]))
        for name, value, summary, expandable in rows:
            self._add(item, name, value, summary, expandable)
