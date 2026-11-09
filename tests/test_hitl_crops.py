"""Crop geometry, real mask-window estimates and the native Qt drawing tab."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import spatialdata as sd
from napari.components import ViewerModel
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QApplication, QFileDialog
from spatialdata.models import Image2DModel, Labels2DModel

from napari_compare_xenium_merscope.hitl_crop_widget import HitlCropWidget
from napari_compare_xenium_merscope.hitl_crops import (
    CropCellCounts,
    CropRectangle,
    HitlCropContext,
    count_crop_cells,
    save_crop_regions,
    validate_crop_rectangles,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "native.zarr"
    labels = np.zeros((1024, 1536), dtype=np.uint32)
    labels[100:132, 100:130] = np.arange(1, 961).reshape(32, 30)
    labels[500, 500] = 1
    sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                np.zeros((1, 1024, 1536), dtype=np.uint8),
                dims=("c", "y", "x"),
                c_coords=["DAPI"],
            )
        },
        labels={"cellpose_cell_labels": Labels2DModel.parse(labels, dims=("y", "x"))},
    ).write(path)
    metadata = next(
        path / name for name in ("zarr.json", ".zattrs") if (path / name).is_file()
    )
    context = HitlCropContext(
        "ATERA",
        "morphology_focus",
        "PILOT_A",
        1024,
        1536,
        0.5,
        2048,
        64,
        32,
        768,
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "2" * 64,
    )
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps(asdict(context)))
    return path, context, setup


def test_context_and_free_rectangle_roundtrip_without_overwriting(source, tmp_path):
    store, context, setup = source
    assert HitlCropContext.read(setup) == context
    context.validate_store(store, "ATERA")
    rectangle = CropRectangle(64, 64, 320, 480)
    assert context.rectangle(rectangle.vertices() + 0.2) == rectangle
    output = tmp_path / "regions.geojson"
    save_crop_regions(output, context, [rectangle.vertices()])
    payload = json.loads(output.read_text())
    assert payload["merxen_hitl_context"] == asdict(context)
    assert payload["merxen_hitl_regions_version"] == 2
    assert payload["features"][0]["geometry"]["coordinates"][0] == [
        [64, 64],
        [544, 64],
        [544, 384],
        [64, 384],
        [64, 64],
    ]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_crop_regions(output, context, [rectangle.vertices()])
    assert output.read_bytes() == before


def test_counts_read_selected_native_windows_and_deduplicate_labels(
    source, monkeypatch
):
    store, context, _ = source
    selections = []
    original = sd.read_zarr

    def read(*args, **kwargs):
        selections.append(kwargs["selection"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sd, "read_zarr", read)
    counts = count_crop_cells(
        store,
        context,
        [CropRectangle(64, 64, 320, 480), CropRectangle(400, 400, 256, 256)],
    )
    assert counts.per_crop == (960, 1)
    assert counts.total == 960
    assert counts.label_key == "cellpose_cell_labels"
    assert selections == [("labels",)]
    with pytest.raises(ValueError, match="resolution"):
        count_crop_cells(store, replace(context, height=1025), [])


def test_invalid_frames_rotations_margins_and_overlaps_are_refused(source):
    store, context, _ = source
    with pytest.raises(ValueError, match="different source"):
        replace(context, source_metadata_sha256="f" * 64).validate_store(store, "ATERA")
    with pytest.raises(ValueError, match="platform"):
        context.validate_store(store, "XENIUM")
    for box in (CropRectangle(0, 0, 320, 480), CropRectangle(64, 64, 32, 480)):
        with pytest.raises(ValueError):
            context.rectangle(box.vertices())
    with pytest.raises(ValueError, match="rotate"):
        context.rectangle(np.array([[0, 300], [300, 600], [600, 300], [300, 0]]))
    with pytest.raises(ValueError, match="overlap"):
        validate_crop_rectangles(
            context,
            [
                CropRectangle(64, 64, 320, 480).vertices(),
                CropRectangle(100, 100, 320, 480).vertices(),
            ],
        )


def wait_for(app, predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("Qt update did not complete")
        time.sleep(0.01)
    app.processEvents()


@pytest.fixture
def widget(source, qapp, tmp_path):
    store, _, setup = source
    viewer = ViewerModel()
    viewer.add_image(
        np.zeros((1024, 1536), dtype=np.uint8),
        name="ATERA morphology_focus DAPI",
        scale=(0.5, 0.5),
    )
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    widget = HitlCropWidget(viewer, settings)
    widget.set_dataset(store, "ATERA")
    widget.load_setup(setup)
    yield widget
    widget._timer.stop()
    wait_for(qapp, lambda: widget._worker is None)
    widget.close()


def test_real_click_drag_updates_list_count_target_and_saved_regions(
    widget, qapp, tmp_path, monkeypatch
):
    from napari.layers.shapes._shapes_mouse_bindings import add_rectangle

    widget._add_button.click()
    layer = widget._layer
    assert layer.mode == "add_rectangle"
    event = SimpleNamespace(position=(32, 32), type="mouse_press", modifiers=[])
    drag = add_rectangle(layer, event)
    next(drag)
    event.type = "mouse_move"
    event.position = (192, 272)
    next(drag)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(drag)
    wait_for(qapp, lambda: widget._counts is not None)
    assert layer.mode == "select" and not widget._add_button.isChecked()
    assert widget._crop_list.count() == 1
    assert "960 cells" in widget._crop_list.item(0).text()
    assert widget._counts.total == 960
    assert widget._target.property("targetReached")
    assert "#21633d" in widget._target.styleSheet()
    path = tmp_path / "selected.geojson"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    widget._save_button.click()
    assert json.loads(path.read_text())["features"][0]["geometry"]["coordinates"][0][
        2
    ] == [544, 384]
    widget._remove_button.click()
    wait_for(qapp, lambda: widget._crop_list.count() == 0)
    assert "0" in widget._counter.text()
    assert not widget._target.property("targetReached")


@pytest.mark.parametrize(
    "total,reached", [(799, False), (800, True), (1200, True), (1201, False)]
)
def test_target_range_is_inclusive_and_stale_counts_are_ignored(widget, total, reached):
    widget._set_count(CropCellCounts((), total, "cellpose_cell_labels"))
    assert widget._target.property("targetReached") is reached
    widget._count_done(widget._generation - 1, CropCellCounts((), 1000, ""))
    assert widget._target.property("targetReached") is reached


def test_dataset_switch_restores_regions_and_disables_paired_selection(
    widget, source, qapp
):
    store, _, _ = source
    widget._layer.add(
        CropRectangle(64, 64, 320, 480).vertices(), shape_type="rectangle"
    )
    wait_for(qapp, lambda: widget._counts is not None)
    widget.set_dataset(None)
    assert not widget._add_button.isEnabled()
    assert not widget._target.property("targetReached")
    widget.set_dataset(store, "ATERA")
    assert widget._crop_list.count() == 1
    wait_for(qapp, lambda: widget._counts is not None)
    assert widget._counts.total == 960


def test_native_control_panel_opens_hitl_tab_without_console(widget, qapp):
    from test_startup import _panel

    expanded = []
    panel = _panel(
        [],
        hitl_crop_widget=widget,
        expand_layer_controls_callback=lambda: expanded.append(True),
    )
    button = next(
        button
        for button in panel._tab_group.buttons()
        if button.text() == "HITL seg crops"
    )
    button.click()
    assert panel._tab_stack.currentWidget().isAncestorOf(widget)
    assert expanded == [True]


def test_overlapping_crops_keep_unique_estimate_but_cannot_be_saved(widget, qapp):
    widget._layer.data = [
        CropRectangle(64, 64, 320, 480).vertices(),
        CropRectangle(70, 70, 320, 480).vertices(),
    ]
    wait_for(qapp, lambda: widget._counts is not None)
    assert widget._counts.per_crop == (960, 960)
    assert widget._counts.total == 960
    assert not widget._save_button.isEnabled()
    assert "overlap" in widget._status.text()
