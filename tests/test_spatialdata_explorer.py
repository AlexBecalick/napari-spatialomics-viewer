"""On-demand object/store traversal and asynchronous explorer lifecycle."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from qtpy.QtWidgets import QApplication

from napari_compare_xenium_merscope import spatialdata_explorer as E


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _store(path):
    zarr.open_group(str(path), mode="w", zarr_format=2)
    return SimpleNamespace(images={}, labels={}, points={}, shapes={}, tables={}, attrs={"sample": "test"})


def _expand(qapp, widget, item):
    item.setExpanded(True)
    deadline = time.monotonic() + 10
    while widget._jobs and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert not widget._jobs


def _child(item, name):
    return next(item.child(i) for i in range(item.childCount()) if item.child(i).text(0) == name)


def test_stored_tables_expose_var_and_nested_uns_without_loading_matrix(tmp_path):
    obj = _store(tmp_path)
    root = zarr.open_group(str(tmp_path), mode="a")
    root.create_group("tables")
    table = ad.AnnData(np.zeros((2, 3)), var=pd.DataFrame({"symbol": ["A", "B", "C"]}, index=["a", "b", "c"]))
    table.uns["analysis"] = {"method": "example", "nested": {"threshold": 0.5}}
    table.write_zarr(str(tmp_path / "tables" / "cells"))

    elements = dict(E._children(E._Store(tmp_path, obj, None)))
    cells = elements["tables"]["cells"]
    assert isinstance(cells, zarr.Group)
    assert dict(E._children(cells))["X"].shape == (2, 3)
    var = dict(E._children(cells))["var"]
    assert dict(E._children(var))["symbol"][:].tolist() == ["A", "B", "C"]
    nested = cells["uns"]["analysis"]["nested"]
    assert float(dict(E._children(nested))["threshold"][()]) == 0.5


def test_loaded_tables_and_images_take_precedence(tmp_path):
    obj = _store(tmp_path)
    table = ad.AnnData(np.zeros((1, 2)))
    table.uns["nested"] = {"key": [1, 2]}
    obj.tables["cells"] = table
    image = np.ones((1, 2, 3))
    images = SimpleNamespace(images={"image": image})
    elements = dict(E._children(E._Store(tmp_path, obj, images)))
    assert elements["images"]["image"] is image
    assert dict(E._children(elements["tables"]["cells"]))["uns"]["nested"]["key"] == [1, 2]


def test_array_expansion_never_reads_values(qapp):
    class Array:
        shape = (10000, 20000)
        dtype = np.dtype("int64")
        chunks = (100, 100)
        nbytes = 1600000000

        def __getitem__(self, index):
            raise AssertionError("Explorer must not read array values")

        def __array__(self):
            raise AssertionError("Explorer must not materialize arrays")

    rows = dict(E._children(Array()))
    assert rows["shape"] == "(10000, 20000)"
    assert rows["dtype"] == "int64"
    assert rows["chunks"] == (100, 100)
    assert rows["nbytes"] == 1600000000


def test_long_collections_are_paged_without_skipping_entries(qapp):
    pages = []
    job = E._Load((0, 0), list(range(250)))
    job.signals.finished.connect(lambda token, rows, error: pages.append((rows, error)))
    job.run()
    rows, error = pages.pop()
    assert error is None
    assert len(rows) == E.PAGE_SIZE + 1
    assert rows[0][1] == 0 and rows[99][1] == 99
    job = E._Load((0, 1), rows[-1][1])
    job.signals.finished.connect(lambda token, rows, error: pages.append((rows, error)))
    job.run()
    assert pages[0][0][0][1] == 100
    assert pages[0][0][99][1] == 199


def test_paired_roots_expand_independently_and_switch_discards_old_reads(qapp, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    obj1, obj2 = _store(first), _store(second)
    widget = E.SpatialDataExplorerWidget()
    widget.set_stores([("MERSCOPE", first, obj1, None), ("XENIUM", second, obj2, None)])
    one, two = widget.tree.topLevelItem(0), widget.tree.topLevelItem(1)
    _expand(qapp, widget, one)
    assert _child(one, "tables") is not None
    assert two.childCount() == 0 and not two.isExpanded()
    _expand(qapp, widget, two)
    one.setExpanded(False)
    assert two.isExpanded()
    old_generation = widget._generation
    old_key = one.data(0, 256)
    widget.set_stores([("NEW", second, obj2, None)])
    widget._finished((old_generation, old_key), [], "stale error")
    assert widget.tree.topLevelItemCount() == 1
    assert widget.tree.topLevelItem(0).text(0) == "NEW"
    assert widget.tree.topLevelItem(0).childCount() == 0
    widget.close()


def test_missing_store_reports_branch_error(qapp, tmp_path):
    widget = E.SpatialDataExplorerWidget()
    widget.set_stores([("Missing", tmp_path / "missing", SimpleNamespace(), None)])
    root = widget.tree.topLevelItem(0)
    _expand(qapp, widget, root)
    assert root.child(0).text(0) == "Could not read branch"
    widget.close()


def test_multiscale_image_exposes_metadata_without_reading_pixels():
    image = E.xr.DataTree.from_dict({"scale0": E.xr.Dataset({"image": (("y", "x"), np.zeros((4, 5)))})})
    children = dict(E._children(image))
    assert "dataset" in children and "scale0" in children
    dataset = dict(E._children(children["scale0"]))["dataset"]
    data = dict(E._children(dataset))["data_vars"]["image"]
    array = dict(E._children(data))["data"]
    assert dict(E._children(array))["shape"] == "(4, 5)"


def test_controller_publishes_standalone_pair_and_empty_sources(tmp_path):
    from napari_compare_xenium_merscope.viewer import ComparisonViewerController

    controller = ComparisonViewerController.__new__(ComparisonViewerController)
    obj = _store(tmp_path)
    received = []
    controller._spatialdata_explorer = SimpleNamespace(set_stores=received.append)
    controller.datasets = {name: SimpleNamespace(zarr_path=tmp_path / name)
                           for name in ("MERSCOPE", "XENIUM")}
    controller._paired_session = None
    controller.active_dataset = "XENIUM"
    controller._active_sdata = obj
    controller._active_images_sdata = None
    controller._publish_spatialdata_explorer()
    assert [row[0] for row in received[-1]] == ["XENIUM"]
    controller._paired_session = SimpleNamespace(sdata={"MERSCOPE": obj, "XENIUM": obj})
    controller._publish_spatialdata_explorer()
    assert [row[0] for row in received[-1]] == ["MERSCOPE", "XENIUM"]
    controller._paired_session = None
    controller.active_dataset = None
    controller._publish_spatialdata_explorer()
    assert received[-1] == []
