"""Tests of the viewer hooks for manual registration (panel, adapter, controller host, CLI).

The registration controller itself (VW2) is faked wherever possible so these tests do not depend
on its internals; two tests drive the real controller through the real host.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import manual_registration_stores as stores
import numpy as np
import pytest

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.manual_registration_controller import (
    ManualRegistrationController,
    RegistrationHost,
)
from napari_compare_xenium_merscope.manual_registration_widget import ManualRegistrationWidget


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _private_app_data(tmp_path, monkeypatch):
    """Never write drafts or settings into the real per-user application-data folder."""
    app_data = tmp_path / "app-data-root"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


# ---------------------------------------------------------------------------------------------
# Helpers and fakes
# ---------------------------------------------------------------------------------------------


def _panel(datasets: list[str], initial_dataset: str | None = None, **overrides):
    callback = lambda *args, **kwargs: None  # noqa: E731
    arguments = dict(
        datasets=datasets,
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
        initial_dataset=initial_dataset,
    )
    arguments.update(overrides)
    return V.ViewerControlPanel(**arguments)


def _settings(tmp_path, name="viewer-settings.ini"):
    from qtpy.QtCore import QSettings

    return QSettings(str(tmp_path / name), QSettings.IniFormat)


def _adapter(model, tmp_path):
    from napari._qt.containers.qt_layer_list import QtLayerList
    from qtpy.QtWidgets import QDockWidget, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

    class Buttons(QWidget):
        def __init__(self, names):
            super().__init__()
            layout = QHBoxLayout(self)
            for name in names:
                button = QPushButton(name)
                setattr(self, name, button)
                layout.addWidget(button)

    class Dock(QDockWidget):
        def __init__(self, widget):
            super().__init__()
            self.title = QWidget()
            self.title.setLayout(QHBoxLayout())
            self.setTitleBarWidget(self.title)
            self.setWidget(widget)

        def inner_widget(self):
            return self.widget()

    layer_buttons = Buttons(
        ["newPointsButton", "newShapesButton", "newLabelsButton", "deleteButton"]
    )
    viewer_buttons = Buttons(
        [
            "consoleButton",
            "ndisplayButton",
            "rollDimsButton",
            "transposeDimsButton",
            "gridViewButton",
            "resetViewButton",
        ]
    )
    layer_view = QtLayerList(model.layers)
    layer_container = QWidget()
    layer_layout = QVBoxLayout(layer_container)
    layer_layout.addWidget(layer_buttons)
    layer_layout.addWidget(layer_view)
    layer_layout.addWidget(viewer_buttons)
    controls = QWidget()
    qt_viewer = QWidget()
    qt_viewer.layerButtons = layer_buttons
    qt_viewer.viewerButtons = viewer_buttons
    qt_viewer.layers = layer_view
    qt_viewer.dockLayerList = Dock(layer_container)
    qt_viewer.dockLayerControls = Dock(controls)
    viewer = types.SimpleNamespace(
        layers=model.layers,
        window=types.SimpleNamespace(
            _qt_viewer=qt_viewer,
            _qt_window=types.SimpleNamespace(resizeDocks=lambda *_args: None),
        ),
    )
    adapter = V.NapariLeftPanelAdapter(viewer, settings=_settings(tmp_path, "adapter.ini"))
    # Keep the Qt objects alive for the duration of the test.
    adapter._test_keepalive = (qt_viewer, layer_container, controls, layer_buttons)
    return adapter


class _FakeRegistration:
    def __init__(self, *, active=False, confirm=True, gesture=None):
        self.active, self.confirm, self.gesture, self.calls = active, confirm, gesture, []

    def is_active(self):
        return self.active

    def deactivate(self):
        self.calls.append("deactivate")
        self.active = False

    def cancel_background(self):
        self.calls.append("cancel")

    def confirm_discard_unsaved(self):
        self.calls.append("confirm")
        return self.confirm

    def load_store(self, path):
        self.calls.append(("load", Path(path)))
        return True

    def mouse_gesture(self, event):
        self.calls.append("gesture")
        return self.gesture

    def set_widget(self, widget):
        self.calls.append(("widget", widget))

    def has_unsaved_changes(self):
        return False


class _Viewer:
    """Enough of a viewer for ``ComparisonViewerController.__init__``."""

    def __init__(self):
        self.layers = []  # a list supports .remove, used by _clear_layers
        self.mouse_drag_callbacks = []
        self.grid = SimpleNamespace(enabled=True)

    def reset_view(self):
        pass


def _controller(viewer=None, **args):
    defaults = dict(
        background_io_workers=1,
        session_cache_gb=0,
        image_pyramid_downsample=4,
        overwrite_derived_caches=False,
        skip_images=False,
        skip_transcripts=False,
        segmentation_source="shapes",
        hide_images=False,
        image_opacity=1.0,
    )
    defaults.update(args)
    return V.ComparisonViewerController(viewer or _Viewer(), {}, SimpleNamespace(**defaults))


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, value):
        for slot in list(self.slots):
            slot(value)


class _Worker:
    def __init__(self, fn):
        self.fn, self.returned, self.errored, self.started = fn, _Signal(), _Signal(), False

    def start(self):
        self.started = True

    def quit(self):
        pass

    def finish(self):
        try:
            value = self.fn()
        except BaseException as exc:  # noqa: BLE001
            self.errored.emit(exc)
            return
        self.returned.emit(value)


def _fake_thread_worker(created):
    def decorate(fn):
        def factory(*_a, **_k):
            worker = _Worker(fn)
            created.append(worker)
            return worker

        return factory

    return decorate


def _registration_store_dir(root: Path, attrs: bool = True) -> Path:
    store = root / "x_prepared.zarr"
    store.mkdir(parents=True)
    attributes = {V.STORE_ATTR: stores.prepared_store_attr()} if attrs else {}
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes})
    )
    return store


def _gesture(registration, world_rc):
    event = SimpleNamespace(
        type="mouse_press", button=1, position=tuple(world_rc), pos=np.array([100.0, 100.0])
    )
    gen = registration.mouse_gesture(event)
    assert gen is not None
    next(gen)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)


def _click_pair(registration, viewer, ref_xy, mov_xy):
    if not registration._landmark_mode:
        registration._widget.landmark_mode_changed.emit(True)
    reference, moving = viewer.layers["Reference | DAPI"], viewer.layers["Moving | DAPI"]
    _gesture(registration, reference.data_to_world((ref_xy[1], ref_xy[0])))
    _gesture(registration, moving.data_to_world((mov_xy[1], mov_xy[0])))


def _grid_pair():
    return (100.0, 150.0), (120.0, 130.0)


def _on_disk_labelling(path):
    from napari_compare_xenium_merscope.manual_registration_model import LABELLING_ATTR
    from napari_compare_xenium_merscope.manual_registration_store import read_root_attrs

    return read_root_attrs(path).get(LABELLING_ATTR)


def _statuses(ctrl):
    out: list[str] = []
    ctrl.set_status_callback(out.append)
    return out


def _gene_widget_with_stale_source():
    """A Gene inspector that still lists the pciSeq source of a previously shown dataset."""
    noop = lambda *args, **kwargs: None  # noqa: E731
    widget = V.GeneInspectorWidget(
        noop, noop, noop, noop, noop, noop, set_assignment_source_callback=noop
    )
    widget.set_assignment_sources(
        {"native": "Native assignments", "pciseq": "pciSeq (previous dataset)"},
        "pciseq",
        "previous dataset summary",
        dataset="XENIUM",
    )
    return widget


def _assignment_sources(widget) -> list[str]:
    combo = widget._assignment_combo
    return [combo.itemText(index) for index in range(combo.count())]


# ---------------------------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------------------------


def test_dataset_tab_has_manual_registration_section_after_paired_layout(qapp):
    from qtpy.QtWidgets import QLabel, QPushButton

    panel = _panel([], load_registration_callback=lambda path: None)
    layout = panel._tab_stack.widget(6).widget().layout()
    texts = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, (QLabel, QPushButton)):
            texts.append(widget.text())
    expected = [
        "Paired dataset layout",
        "Switch to side-by-side view",
        "Switch to Stacked overlay view",
        "Manual Registration",
        "Load image data",
        "Recently viewed",
    ]
    start = texts.index(expected[0])
    assert texts[start : start + len(expected)] == expected
    assert panel._load_registration_button.isEnabled()
    assert not _panel([])._load_registration_button.isEnabled()


def test_manual_registration_tab_is_appended_last(qapp):
    from qtpy.QtWidgets import QWidget

    sentinel = QWidget()
    panel = _panel([], registration_widget=sentinel)
    assert panel._tab_stack.count() == 10
    assert [panel._tab_group.button(i).text() for i in range(10)] == [
        "Gene inspector",
        "Cell segmentation",
        "Cell type labels",
        "Per cell statistics",
        "Draw tissue annotations",
        "Images",
        "Dataset loader",
        "UMAP",
        "Zarr metadata",
        "Manual registration",
    ]
    assert panel._tab_stack.currentIndex() == 6
    assert panel._tab_stack.widget(9).isAncestorOf(sentinel)
    assert not sentinel.isEnabled()
    panel.select_registration_tab()
    assert panel._tab_stack.currentIndex() == 9
    assert panel._tab_group.button(9).isChecked()
    assert _panel(["MERSCOPE"], initial_dataset="MERSCOPE")._tab_stack.currentIndex() == 0


def test_registration_mode_toggles_panel_tools(qapp):
    from qtpy.QtWidgets import QWidget

    sentinel = QWidget()
    panel = _panel(
        ["MERSCOPE"],
        initial_dataset="MERSCOPE",
        registration_widget=sentinel,
        load_registration_callback=lambda path: None,
    )
    panel.set_registration_mode(True)
    for button in (
        panel._dataset_combo,
        panel._reload_button,
        panel._create_annotations_button,
        panel._load_transcripts_button,
        panel._load_selected_labels_button,
        panel._load_cellpose_values_button,
    ):
        assert not button.isEnabled()
    for button in (
        panel._load_registration_button,
        panel._load_paired_side_button,
        panel._load_standalone_merscope_button,
    ):
        assert button.isEnabled()
    assert sentinel.isEnabled()
    assert panel._registration_hint.isHidden()

    panel.set_registration_mode(False)
    assert panel._create_annotations_button.isEnabled()
    assert panel._load_transcripts_button.isEnabled()
    assert panel._dataset_combo.isEnabled()
    assert not sentinel.isEnabled()
    assert not panel._registration_hint.isHidden()


def test_load_image_data_defers_recent_until_loaded(qapp, tmp_path, monkeypatch):
    store = tmp_path / "x_prepared.zarr"
    store.mkdir()
    got = []

    def callback(path):
        got.append(path)
        return result["value"]

    result = {"value": None}
    panel = _panel([], settings=_settings(tmp_path), load_registration_callback=callback)
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(store))
    opened = []
    panel.dataset_open_requested.connect(lambda: opened.append(True))

    panel._load_registration_button.click()
    assert got == [str(store)]
    assert panel.recent_datasets == [] and opened == []

    panel.registration_dataset_loaded(store)
    assert panel.recent_datasets[0] == {"kind": "registration", "path": str(store.absolute())}
    assert opened == [True]

    result["value"] = True
    other = tmp_path / "y_prepared.zarr"
    other.mkdir()
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(other))
    panel._load_registration_button.click()
    assert panel.recent_datasets[0]["path"] == str(other.absolute())
    assert opened == [True, True]

    result["value"] = False
    third = tmp_path / "z_prepared.zarr"
    third.mkdir()
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(third))
    panel._load_registration_button.click()
    assert all(entry["path"] != str(third.absolute()) for entry in panel.recent_datasets)
    assert opened == [True, True]


def test_registration_recents_round_trip(qapp, tmp_path):
    for name in ("a/spatialdata.zarr", "m/spatialdata.zarr", "x/spatialdata.zarr"):
        (tmp_path / name).mkdir(parents=True)
    store = tmp_path / "runs" / "x_prepared.zarr"
    store.mkdir(parents=True)
    settings = _settings(tmp_path)
    panel = _panel([], settings=settings, load_registration_callback=lambda path: False)
    panel.record_recent_dataset("MERSCOPE", tmp_path / "a" / "spatialdata.zarr")
    panel.record_recent_pair(
        tmp_path / "m" / "spatialdata.zarr",
        tmp_path / "x" / "spatialdata.zarr",
        V.ViewMode.SIDE_BY_SIDE,
    )
    panel.record_recent_registration(store)

    called = []
    restored = _panel(
        [],
        settings=settings,
        load_registration_callback=lambda path: called.append(path) or False,
    )
    assert len(restored.recent_datasets) == 3
    assert restored.recent_datasets[0]["kind"] == "registration"
    item = restored._recent_dataset_list.item(0)
    assert item.text() == "MANUAL REGISTRATION — x_prepared.zarr"
    assert str(store) in item.toolTip()
    assert restored._recent_browse_directory() == str(store.parent)

    restored._recent_dataset_list.setCurrentRow(0)
    restored._open_recent_dataset_button.click()
    assert called == [str(store.absolute())]

    # One row per store: a registration entry replaces a standalone entry with the same path.
    shared = tmp_path / "shared.zarr"
    shared.mkdir()
    restored.record_recent_dataset("MERSCOPE", shared)
    restored.record_recent_registration(shared)
    matching = [e for e in restored.recent_datasets if e.get("path") == str(shared.absolute())]
    assert matching == [{"kind": "registration", "path": str(shared.absolute())}]

    settings.setValue(
        V.RECENT_DATASETS_SETTINGS_KEY, json.dumps([{"kind": "registration", "path": ""}])
    )
    assert _panel([], settings=settings).recent_datasets == []


def test_open_recent_registration_without_callback_sets_status(qapp, tmp_path):
    settings = _settings(tmp_path)
    settings.setValue(
        V.RECENT_DATASETS_SETTINGS_KEY,
        json.dumps([{"kind": "registration", "path": str(tmp_path / "x_prepared.zarr")}]),
    )
    panel = _panel([], settings=settings)
    panel._recent_dataset_list.setCurrentRow(0)
    panel._open_recent_dataset_button.click()
    qapp.processEvents()
    assert panel._status_label.text() == "Manual registration is unavailable in this build."


def test_standalone_loader_ignores_pending_result(qapp, tmp_path, monkeypatch):
    panel = _panel([], settings=_settings(tmp_path), load_standalone_callback=lambda *_a: None)
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(tmp_path / "s.zarr"))
    panel._load_standalone_merscope_button.click()
    panel._load_standalone_xenium_button.click()
    assert panel.recent_datasets == []


# ---------------------------------------------------------------------------------------------
# Controller: routing, loading, teardown
# ---------------------------------------------------------------------------------------------


def test_standalone_load_redirects_registration_store(qapp, tmp_path, monkeypatch):
    store = _registration_store_dir(tmp_path / "with")
    ctrl = _controller()
    recorded = []
    monkeypatch.setattr(
        ctrl, "load_registration_store", lambda path: recorded.append(Path(path)) or None
    )

    def forbidden(*_a, **_k):
        raise AssertionError("a registration store must not be loaded as a standalone dataset")

    monkeypatch.setattr(ctrl, "_replace_datasets", forbidden)
    assert ctrl.load_standalone_dataset("MERSCOPE", store) is None
    assert recorded == [store]

    plain = _registration_store_dir(tmp_path / "without", attrs=False)
    monkeypatch.setattr(ctrl, "_replace_datasets", lambda *_a, **_k: True)
    assert ctrl.load_standalone_dataset("MERSCOPE", plain) is True
    assert recorded == [store]


def test_standalone_load_keeps_unsaved_registration_session(qapp, tmp_path, monkeypatch):
    plain = _registration_store_dir(tmp_path, attrs=False)
    ctrl = _controller()
    fake = _FakeRegistration(active=True, confirm=False)
    ctrl._registration = fake
    statuses = _statuses(ctrl)
    replaced = []

    def replace(*args, **kwargs):
        replaced.append(args)
        raise AssertionError("declining the discard must not replace the datasets")

    monkeypatch.setattr(ctrl, "_replace_datasets", replace)
    assert ctrl.load_standalone_dataset("XENIUM", plain) is False
    assert replaced == []
    assert fake.calls == ["confirm"]
    assert statuses[-1] == "Kept the current registration session (unsaved changes)."

    # Accepting the discard goes on to replace the datasets.
    fake.confirm = True
    monkeypatch.setattr(ctrl, "_replace_datasets", lambda *_a, **_k: True)
    assert ctrl.load_standalone_dataset("XENIUM", plain) is True


def test_load_dataset_routes_registration_name(qapp, tmp_path, monkeypatch):
    store = tmp_path / "x_prepared.zarr"
    store.mkdir()
    ctrl = _controller()
    ctrl.datasets = {V.REGISTRATION_DATASET: V.DatasetConfig(V.REGISTRATION_DATASET, store)}
    ctrl.active_dataset = V.REGISTRATION_DATASET
    ctrl._registration = _FakeRegistration(active=True)
    calls = []
    monkeypatch.setattr(
        ctrl, "load_registration_store", lambda path: calls.append(Path(path)) or None
    )

    def forbidden(*_a, **_k):
        raise AssertionError("_prepare_dataset_session must not be used")

    monkeypatch.setattr(ctrl, "_prepare_dataset_session", forbidden)
    assert ctrl.load_dataset(V.REGISTRATION_DATASET) is True
    assert calls == []
    assert ctrl.load_dataset(V.REGISTRATION_DATASET, force=True) is True
    assert calls == [store]


def test_load_registration_store_confirms_supersedes_and_delegates(qapp, tmp_path, monkeypatch):
    store = _registration_store_dir(tmp_path)
    monkeypatch.setattr(V, "validate_spatialdata_store_compatibility", lambda _p: None)
    ctrl = _controller()
    fake = _FakeRegistration(active=True, confirm=False)
    ctrl._registration = fake

    assert ctrl.load_registration_store(store) is False
    assert fake.calls == ["confirm"]

    fake.confirm = True
    fake.calls.clear()
    generation = ctrl._dataset_load_generation
    previous_cancel = ctrl._dataset_load_cancel
    assert ctrl.load_registration_store(store) is None
    assert ctrl._dataset_load_generation == generation + 1
    assert previous_cancel.is_set()
    assert fake.calls == ["confirm", "cancel", ("load", store)]

    statuses = _statuses(ctrl)
    assert ctrl.load_registration_store(tmp_path / "missing.zarr") is False
    assert statuses[-1].startswith("Could not open manual registration store")


def test_refused_registration_request_leaves_pending_loads_alone(qapp, tmp_path, monkeypatch):
    """A request that is refused (not a store, missing, declined) must not cancel anything."""
    monkeypatch.setattr(V, "validate_spatialdata_store_compatibility", lambda _p: None)
    store = _registration_store_dir(tmp_path / "with")
    plain = _registration_store_dir(tmp_path / "without", attrs=False)
    ctrl = _controller()
    fake = _FakeRegistration(active=True, confirm=True)
    ctrl._registration = fake
    statuses = _statuses(ctrl)
    progress: list[tuple] = []
    ctrl.set_progress_callback(lambda *args: progress.append(args))
    pending_load = ctrl._dataset_load_cancel
    sentinel_worker, sentinel_key = object(), "XENIUM"
    ctrl._dataset_load_worker, ctrl._dataset_loading_key = sentinel_worker, sentinel_key
    generation = ctrl._dataset_load_generation
    token = ctrl._registration_worker_tokens["registration_load"] = Event()

    def untouched():
        assert ctrl._dataset_load_generation == generation
        assert ctrl._dataset_load_cancel is pending_load and not pending_load.is_set()
        assert ctrl._dataset_load_worker is sentinel_worker
        assert ctrl._dataset_loading_key == sentinel_key
        assert not token.is_set()
        assert fake.calls == [] and progress == []

    # A plain SpatialData folder (no store attribute), then a missing one.
    assert ctrl.load_registration_store(plain) is False
    assert statuses[-1].startswith("Could not open manual registration store")
    untouched()
    assert ctrl.load_registration_store(tmp_path / "missing.zarr") is False
    untouched()

    # A store whose descriptor is malformed is refused the same way.
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {V.STORE_ATTR: {"schema": "merxen.manual_registration.store"}},
            }
        )
    )
    assert ctrl.load_registration_store(store) is False
    untouched()

    # Declining to discard unsaved work only asks; nothing is cancelled.
    good = _registration_store_dir(tmp_path / "good")
    fake.confirm = False
    assert ctrl.load_registration_store(good) is False
    assert fake.calls == ["confirm"]
    assert statuses[-1] == "Kept the current registration session (unsaved changes)."
    fake.calls.clear()
    assert ctrl._dataset_load_generation == generation
    assert ctrl._dataset_load_cancel is pending_load and not pending_load.is_set()
    assert ctrl._dataset_load_worker is sentinel_worker and not token.is_set()


# ---------------------------------------------------------------------------------------------
# Left-dock adapter
# ---------------------------------------------------------------------------------------------


def test_adapter_skips_tagged_layers_across_inserts_and_rotation(qapp, tmp_path):
    from napari.components import ViewerModel

    model = ViewerModel()
    image = model.add_image(np.zeros((2, 2)), name="Image | DAPI")
    matrix_a = np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1]])
    moving = model.add_image(
        np.zeros((4, 4)),
        name="Moving | DAPI",
        affine=matrix_a,
        metadata={V.LAYER_METADATA_KEY: {"role": "moving", "kind": "image", "channel": "DAPI"}},
    )
    adapter = _adapter(model, tmp_path)
    store = tmp_path / "data.zarr"
    store.mkdir()
    adapter.set_rotation_dataset([store])

    adapter.set_rotation_angle(90.0)
    assert np.allclose(image.data_to_world((0.0, 0.0)), (1.0, 0.0))
    assert np.allclose(moving.affine.affine_matrix, matrix_a, atol=1e-12)
    assert moving not in adapter._layer_base_affines

    matrix_b = np.array([[-0.5, 0, 30], [0, -0.5, 40], [0, 0, 1]])
    moving.affine = matrix_b
    model.add_points(np.zeros((1, 2)))  # an insert event re-registers the layers
    assert np.allclose(moving.affine.affine_matrix, matrix_b, atol=1e-12)

    before = adapter._rotation_origin()
    model.add_image(
        np.zeros((8, 8)),
        name="Moving | AF568",
        affine=np.diag([1000.0, 1000.0, 1.0]),
        metadata={V.LAYER_METADATA_KEY: {"role": "moving", "kind": "image", "channel": "AF568"}},
    )
    assert np.allclose(adapter._rotation_origin(), before, atol=1e-12)


def test_adapter_registration_mode_disables_rotation_and_guards_deletion(qapp, tmp_path):
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QAbstractItemView

    model = ViewerModel()
    model.add_image(np.zeros((2, 2)), name="Image | DAPI")
    adapter = _adapter(model, tmp_path)
    store = tmp_path / "data.zarr"
    store.mkdir()
    adapter.set_rotation_dataset([store])
    adapter.set_rotation_angle(37.25)
    assert adapter.rotation_save_button.isEnabled()

    adapter.set_registration_mode(True)
    assert adapter.rotation_spin.value() == 0.0
    assert not adapter.rotation_control.isEnabled()
    model.add_image(np.zeros((2, 2)), name="Another | DAPI")
    adapter._refresh_gene_presentation()
    assert not adapter.rotation_control.isEnabled()
    assert not adapter.rotation_save_button.isEnabled()
    assert not adapter._delete_button.isEnabled()
    assert adapter._layer_view.dragDropMode() == QAbstractItemView.NoDragDrop
    assert adapter._layer_view.editTriggers() == QAbstractItemView.NoEditTriggers

    adapter.set_registration_mode(False)
    assert adapter._delete_button.isEnabled()
    assert adapter.rotation_control.isEnabled()

    adapter.set_paired_mode(True)
    adapter.set_registration_mode(False)
    assert not adapter._delete_button.isEnabled()
    assert (
        adapter._delete_button.toolTip() == "Unload paired layers from the Viewer Controls panel."
    )


def test_mouse_press_delegates_to_registration_gesture(qapp, monkeypatch):
    ctrl = _controller()
    seen: list[str] = []

    def gesture():
        seen.append("press")
        yield
        seen.append("release")

    ctrl._registration = _FakeRegistration(active=True, gesture=gesture())

    def forbidden(*_a, **_k):
        raise AssertionError("no cell or transcript picking in a registration session")

    monkeypatch.setattr(ctrl, "_handle_viewer_click", forbidden, raising=False)
    event = SimpleNamespace(type="mouse_press", position=(0.0, 0.0), pos=(0.0, 0.0))
    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)
    assert seen == ["press", "release"]

    # No gesture and an active session: a plain pan, nothing else.
    ctrl._registration = _FakeRegistration(active=True, gesture=None)
    event.type = "mouse_press"
    with pytest.raises(StopIteration):
        next(ctrl._on_viewer_mouse_press(ctrl.viewer, event))

    # The lasso keeps priority and never reaches the registration controller.
    fake = _FakeRegistration(active=True, gesture=None)
    ctrl._registration = fake
    ctrl._umap_lasso_armed = True
    lasso_calls = []

    def lasso(_event):
        lasso_calls.append(True)
        yield

    monkeypatch.setattr(ctrl, "_spatial_lasso_gesture", lasso)
    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    assert lasso_calls == [True]
    assert "gesture" not in fake.calls


def test_teardown_paths_deactivate_registration(qapp, tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise Stop

    store = tmp_path / "s.zarr"
    store.mkdir()
    cfg = V.DatasetConfig("MERSCOPE", store)

    def new():
        ctrl = _controller()
        fake = _FakeRegistration(active=True)
        ctrl._registration = fake
        ctrl._clear_layers = stop
        ctrl.datasets = {"MERSCOPE": cfg}
        return ctrl, fake

    ctrl, fake = new()
    session = V.DatasetSession("MERSCOPE", cfg, object(), None, (1, 0, 0), (1, 0, 0), [], [], [])
    with pytest.raises(Stop):
        ctrl._apply_dataset_session(
            ctrl._dataset_load_generation,
            session,
            {"rss_gb": 0.0, "sys_used_gb": 0.0},
            time.time(),
        )
    assert fake.calls.index("deactivate") > fake.calls.index("cancel")

    ctrl, fake = new()
    with pytest.raises(Stop):
        ctrl._apply_paired_dataset_session(
            ctrl._dataset_load_generation,
            SimpleNamespace(configs={}, sdata={"MERSCOPE": None}),
            0.0,
        )
    assert fake.calls.index("deactivate") > fake.calls.index("cancel")

    ctrl, fake = new()
    with pytest.raises(Stop):
        ctrl._replace_datasets({"MERSCOPE": cfg}, "MERSCOPE")
    assert fake.calls.index("deactivate") > fake.calls.index("cancel")

    # load_paired_dataset only confirms; the teardown is deferred to the apply step.
    monkeypatch.setattr(V, "validate_spatialdata_store_compatibility", lambda _p: None)
    dir_a, dir_b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    dir_a.mkdir()
    dir_b.mkdir()
    ctrl, fake = new()
    fake.confirm = False
    assert ctrl.load_paired_dataset(dir_a, dir_b) is False
    assert "cancel" not in fake.calls

    created: list[_Worker] = []
    monkeypatch.setattr(V, "thread_worker", _fake_thread_worker(created))
    ctrl, fake = new()
    fake.confirm = True
    assert ctrl.load_paired_dataset(dir_a, dir_b) is None
    assert "deactivate" not in fake.calls


def test_cancel_and_reset_cover_registration(qapp):
    ctrl = _controller()
    fake = _FakeRegistration()
    ctrl._registration = fake
    captured = []
    ctrl.set_progress_callback(lambda key, text, active: captured.append((key, text, active)))
    ctrl.begin_progress("registration_load", "Loading…")
    ctrl._cancel_background_tasks()
    assert "cancel" in fake.calls
    ctrl._reset_progress()
    assert ("registration_load", "", False) in captured
    assert ("registration", "", False) in captured
    assert ctrl._registration_progress_keys == set()


def test_run_worker_sync_async_supersede_and_cancel(qapp, monkeypatch):
    ctrl = _controller()
    done: list = []
    errors: list = []

    monkeypatch.setattr(V, "thread_worker", None)
    ctrl.run_worker(lambda: 5, done.append, errors.append, key="k")
    assert done == [5] and errors == []

    def boom():
        raise ValueError("x")

    ctrl.run_worker(boom, done.append, errors.append, key="k")
    assert isinstance(errors[0], ValueError) and str(errors[0]) == "x"
    assert done == [5]

    created: list[_Worker] = []
    monkeypatch.setattr(V, "thread_worker", _fake_thread_worker(created))
    done.clear()
    ctrl.run_worker(lambda: "A", done.append, errors.append, key="k")
    ctrl.run_worker(lambda: "B", done.append, errors.append, key="k")
    first, second = created
    assert first.started and second.started
    first.finish()
    assert done == []
    second.finish()
    assert done == ["B"]

    created.clear()
    ctrl.run_worker(lambda: "C", done.append, errors.append, key="s")
    ctrl._cancel_background_tasks()
    created[0].finish()
    assert done == ["B"]

    created.clear()
    ctrl.run_worker(lambda: "D", done.append, errors.append, key="registration_save")
    ctrl._cancel_background_tasks()
    created[0].finish()
    assert done == ["B", "D"]
    assert "registration_save" not in ctrl._registration_worker_tokens

    # napari delivers errors as (type, value, traceback) tuples.
    created.clear()
    errors.clear()
    ctrl.run_worker(lambda: 1, done.append, errors.append, key="t")
    exc = RuntimeError("tuple")
    created[0].errored.emit((RuntimeError, exc, None))
    assert errors == [exc]


def test_images_tab_actions_are_inert_in_registration_session(qapp, monkeypatch):
    ctrl = _controller()
    ctrl._registration = _FakeRegistration(active=True)
    statuses = _statuses(ctrl)

    def forbidden(*_a, **_k):
        raise AssertionError("must not touch the dataset")

    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", forbidden)
    assert ctrl.load_images_on_demand(V.REGISTRATION_DATASET) is None
    assert ctrl.load_selected_image(V.REGISTRATION_DATASET, [("reference", "DAPI")]) is None
    assert ctrl.unload_selected_images(V.REGISTRATION_DATASET, [("reference", "DAPI")]) is None
    assert statuses[-1] == V.REGISTRATION_IMAGES_STATUS
    assert statuses.count(V.REGISTRATION_IMAGES_STATUS) == 3


def test_startup_autoload_skips_transcripts_without_points(qapp, monkeypatch):
    ctrl = _controller()
    ctrl._image_keys = []
    ctrl._segmentation_keys = []
    calls = []
    monkeypatch.setattr(
        ctrl, "open_gene_inspector", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    statuses = _statuses(ctrl)
    widget = _gene_widget_with_stale_source()
    ctrl.set_gene_inspector_widget(widget)
    ctrl.active_dataset = "X"

    ctrl._active_sdata = SimpleNamespace(points={})
    ctrl._start_startup_autoload("X")
    assert calls == []
    assert statuses[-1] == "X has no transcripts; images and masks only."
    # The previous dataset's pciSeq source and summary are gone.
    assert _assignment_sources(widget) == ["Native assignments"]
    assert widget._assignment_summary.text() == ""

    ctrl._active_sdata = SimpleNamespace(points={"transcripts": object()})
    ctrl._start_startup_autoload("X")
    assert calls == [(("X",), {"reuse_cached": True})]


def test_cursor_status_defers_to_napari_in_registration(qapp):
    ctrl = _controller()
    ctrl._napari_status_calculator = lambda: "napari-status"
    ctrl.active_dataset = V.REGISTRATION_DATASET
    ctrl._active_sdata = object()
    ctrl._registration = _FakeRegistration(active=True)
    assert ctrl._cell_aware_status_from_cursor() == "napari-status"


def test_parse_args_registration_store(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--registration-store", "s.zarr"])
    assert V.parse_args().registration_store == Path("s.zarr")
    monkeypatch.setattr(sys, "argv", ["prog"])
    assert V.parse_args().registration_store is None
    monkeypatch.setattr(
        sys, "argv", ["prog", "--registration-store", "s.zarr", "--merscope-zarr", "m.zarr"]
    )
    with pytest.raises(SystemExit):
        V.parse_args()


def test_controller_implements_registration_host(qapp):
    ctrl = _controller()
    for name in (
        "set_status",
        "begin_progress",
        "end_progress",
        "run_worker",
        "display_levels",
        "enter_registration_session",
        "registration_session_ready",
        "leave_registration_session",
        "app_data_dir",
    ):
        assert callable(getattr(ctrl, name)), name
    assert ctrl.viewer is not None
    if getattr(RegistrationHost, "_is_runtime_protocol", False):
        assert isinstance(ctrl, RegistrationHost)
    assert isinstance(ctrl._registration, ManualRegistrationController)


def test_app_data_dir_is_created(qapp, tmp_path, monkeypatch):
    base = tmp_path / "xdg"
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(base))
    path = _controller().app_data_dir()
    assert path == base / "napari-compare-xenium-merscope"
    assert path.is_dir()


def test_close_guard_blocks_close_when_discard_declined(qapp):
    from qtpy.QtWidgets import QWidget

    window = QWidget()
    window.show()
    guard = V.RegistrationCloseGuard(
        SimpleNamespace(confirm_close_registration=lambda: False), window
    )
    window.installEventFilter(guard)
    assert window.close() is False
    assert window.isVisible()
    window.removeEventFilter(guard)
    allowing = V.RegistrationCloseGuard(
        SimpleNamespace(confirm_close_registration=lambda: True), window
    )
    window.installEventFilter(allowing)
    assert window.close() is True


# ---------------------------------------------------------------------------------------------
# Host: display levels and the session lifecycle (real image data on disk)
# ---------------------------------------------------------------------------------------------


def _write_large_store(path: Path):
    import spatialdata as sd
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Scale

    data = np.random.default_rng(0).integers(0, 1000, (2, 2100, 2100), dtype=np.uint16)
    image = Image2DModel.parse(
        data,
        dims=("c", "y", "x"),
        c_coords=["DAPI", "AF568"],
        transformations={"global": Scale([0.5, 0.5], axes=("x", "y"))},
    )
    sd.SpatialData(images={"reference": image}).write(path)
    return path


def test_display_levels_builds_cache_into_registration_store(qapp, tmp_path, monkeypatch):
    import spatialdata as sd
    import zarr

    store = _write_large_store(tmp_path / "x_prepared.zarr")
    ctrl = _controller()
    assert ctrl.datasets == {} and ctrl.active_dataset is None
    sdata = sd.read_zarr(store)

    levels = ctrl.display_levels("reference", sdata, cancel_check=lambda: False)
    assert levels[0].shape == (2, 2100, 2100)
    assert len(levels) >= 2
    assert all(tuple(level.dims) == ("c", "y", "x") for level in levels)
    cache = store / "images" / "_napari_compare_imgpyr__reference__ds4"
    assert cache.is_dir()
    attrs = dict(zarr.open_group(str(cache), mode="r").attrs)
    assert attrs[V.DERIVED_CACHE_ATTR]["complete"] is True
    assert ctrl.datasets == {}

    # A second read reuses the cache and never writes.
    def no_write(*_a, **_k):
        raise AssertionError("the cache must be reused")

    monkeypatch.setattr(sd.SpatialData, "write_element", no_write)
    again = ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=lambda: False)
    assert len(again) >= 2
    monkeypatch.undo()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(tmp_path))

    with pytest.raises(RuntimeError, match="Background task cancelled"):
        ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=lambda: True)

    # A busy store: runtime overview pyramid, nothing written, status explains.
    busy_store = _write_large_store(tmp_path / "busy_prepared.zarr")

    @contextmanager
    def busy(*_a, **_k):
        raise V.StoreBusyError("busy")
        yield  # pragma: no cover

    monkeypatch.setattr(V, "store_write_lock", busy)
    statuses = _statuses(ctrl)
    levels = ctrl.display_levels("reference", sd.read_zarr(busy_store), cancel_check=None)
    assert len(levels) >= 2
    assert not (busy_store / "images" / "_napari_compare_imgpyr__reference__ds4").exists()
    assert statuses[-1] == (
        "The registration store is busy; showing images without the cached overview pyramid."
    )


def test_display_levels_falls_back_when_the_lock_file_cannot_be_opened(qapp, tmp_path, monkeypatch):
    import spatialdata as sd

    store = _write_large_store(tmp_path / "x_prepared.zarr")

    @contextmanager
    def unwritable(*_a, **_k):
        raise PermissionError(13, "Permission denied")
        yield  # pragma: no cover

    monkeypatch.setattr(V, "store_write_lock", unwritable)
    ctrl = _controller()
    statuses = _statuses(ctrl)
    levels = ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=None)
    assert levels[0].shape == (2, 2100, 2100)
    assert len(levels) >= 2
    assert not (store / "images" / "_napari_compare_imgpyr__reference__ds4").exists()
    assert statuses[-1] == (
        "The registration store is not writable; showing images without the cached overview "
        "pyramid."
    )


def test_display_levels_on_a_read_only_store_without_a_cache(qapp, tmp_path):
    import spatialdata as sd

    shared = tmp_path / "shared_results"
    shared.mkdir()
    store = _write_large_store(shared / "x_registered.zarr")
    sdata = sd.read_zarr(store)

    def set_writable(writable: bool) -> None:
        for dirpath, dirnames, filenames in os.walk(shared, topdown=False):
            for name in filenames:
                (Path(dirpath) / name).chmod(0o644 if writable else 0o444)
            for name in dirnames:
                (Path(dirpath) / name).chmod(0o755 if writable else 0o555)
        shared.chmod(0o755 if writable else 0o555)

    set_writable(False)
    try:
        if os.access(shared, os.W_OK):
            pytest.skip("file permissions are not enforced for this user")
        ctrl = _controller()
        statuses = _statuses(ctrl)
        levels = ctrl.display_levels("reference", sdata, cancel_check=None)
        assert levels[0].shape == (2, 2100, 2100)
        assert len(levels) >= 2
        assert "not writable" in statuses[-1]
        assert not (store / "images" / "_napari_compare_imgpyr__reference__ds4").exists()
        assert not list(shared.glob("*.lock"))
    finally:
        set_writable(True)


def test_display_levels_takes_the_store_lock_for_every_write(qapp, tmp_path, monkeypatch):
    import spatialdata as sd

    store = _write_large_store(tmp_path / "x_prepared.zarr")
    locks: list[Path] = []
    held = {"now": False}
    writes: list[bool] = []
    real_lock = V.store_write_lock
    real_write = sd.SpatialData.write_element

    @contextmanager
    def spy(path, **kwargs):
        locks.append(Path(path))
        with real_lock(path, **kwargs):
            held["now"] = True
            try:
                yield
            finally:
                held["now"] = False

    def write_element(self, *args, **kwargs):
        writes.append(held["now"])
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(V, "store_write_lock", spy)
    monkeypatch.setattr(sd.SpatialData, "write_element", write_element)
    ctrl = _controller()

    # Nothing cached: the cache is built under the lock.
    assert len(ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=None)) >= 2
    assert locks == [store] and writes == [True]

    # A complete, readable cache is reused: no lock and no write.
    locks.clear()
    writes.clear()
    assert len(ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=None)) >= 2
    assert locks == [] and writes == []

    # A complete marker whose element cannot be read back would be rebuilt: that write is
    # still under the lock.
    monkeypatch.setattr(ctrl, "_read_images_from_store", lambda *_a, **_k: None)
    assert len(ctrl.display_levels("reference", sd.read_zarr(store), cancel_check=None)) >= 2
    assert locks == [store] and writes == [True]


def test_display_levels_requires_a_store_path(qapp):
    with pytest.raises(RuntimeError, match="no store path"):
        _controller().display_levels("reference", SimpleNamespace(path=None), cancel_check=None)


def test_enter_ready_leave_registration_session(qapp, tmp_path):
    import spatialdata as sd

    store = stores.build_prepared_store(tmp_path / "x_prepared.zarr")
    sdata = sd.read_zarr(store)
    ctrl = _controller()
    modes, tabs, loaded, changed = [], [], [], []
    ctrl.set_registration_mode_callback(modes.append)
    ctrl.set_registration_tab_callback(lambda: tabs.append(True))
    ctrl.set_registration_dataset_loaded_callback(loaded.append)
    ctrl.set_datasets_changed_callback(lambda names, initial: changed.append((names, initial)))
    left_panel: list[bool] = []
    ctrl._left_panel_adapter = SimpleNamespace(set_registration_mode=left_panel.append)
    old_generation = ctrl._dataset_load_generation

    ctrl.enter_registration_session(store, sdata)
    assert left_panel == [True]
    assert ctrl.active_dataset == V.REGISTRATION_DATASET
    assert list(ctrl.datasets) == [V.REGISTRATION_DATASET]
    assert ctrl.datasets[V.REGISTRATION_DATASET].zarr_path == store
    assert ctrl.viewer.grid.enabled is False
    assert modes == [True]
    assert changed == [([V.REGISTRATION_DATASET], V.REGISTRATION_DATASET)]
    assert ctrl._dataset_load_generation == old_generation + 1

    ctrl.registration_session_ready(store, "prepared")
    assert tabs == [True]
    assert loaded == [store]

    ctrl.leave_registration_session()
    assert modes == [True, False]
    assert left_panel == [True, False]
    assert ctrl.active_dataset is None
    assert V.REGISTRATION_DATASET not in ctrl.datasets
    assert changed[-1] == ([], None)


def test_enter_registration_session_tears_down_the_paired_view(qapp, tmp_path):
    import spatialdata as sd

    store = stores.build_prepared_store(tmp_path / "x_prepared.zarr")
    ctrl = _controller()
    order: list[str] = []
    ctrl._paired_session = object()

    def deactivate_paired():
        order.append("deactivate_paired")
        ctrl._paired_session = None

    ctrl._deactivate_paired_view = deactivate_paired
    ctrl._clear_layers = lambda: order.append("clear_layers")
    ctrl.enter_registration_session(store, sd.read_zarr(store))
    assert order == ["deactivate_paired", "clear_layers"]
    assert ctrl.active_dataset == V.REGISTRATION_DATASET


def test_enter_registration_session_resets_the_assignment_sources(qapp, tmp_path):
    import spatialdata as sd

    store = stores.build_prepared_store(tmp_path / "x_prepared.zarr")
    ctrl = _controller()
    widget = _gene_widget_with_stale_source()
    ctrl.set_gene_inspector_widget(widget)
    ctrl.enter_registration_session(store, sd.read_zarr(store))
    assert _assignment_sources(widget) == ["Native assignments"]
    assert widget._assignment_summary.text() == ""
    assert widget.dataset == V.REGISTRATION_DATASET


def test_save_survives_loading_another_store(qapp, tmp_path, monkeypatch):
    """A save started in store A still reports after store B was loaded through the real host."""
    from napari.components import ViewerModel

    store_a = stores.build_prepared_store(tmp_path / "a_prepared.zarr")
    store_b = tmp_path / "b_prepared.zarr"
    shutil.copytree(store_a, store_b)
    for path in (store_a, store_b):  # build_prepared_store may add caches; keep them
        assert path.is_dir()

    created: list[_Worker] = []
    monkeypatch.setattr(V, "thread_worker", _fake_thread_worker(created))
    monkeypatch.setattr(V, "validate_spatialdata_store_compatibility", lambda _p: None)
    viewer = ViewerModel()
    ctrl = _controller(viewer)
    widget = ManualRegistrationWidget()
    ctrl.set_registration_widget(widget)
    registration = ctrl._registration
    monkeypatch.setattr(registration, "_ask", lambda *_a, default, **_k: "Discard")
    monkeypatch.setattr(registration, "_warn", lambda *_a, **_k: None)
    statuses = _statuses(ctrl)

    assert ctrl.load_registration_store(store_a) is None
    created.pop(0).finish()  # the load of A
    assert registration.is_active()
    registration.flush_pending_updates()

    ref_xy, mov_xy = _grid_pair()
    _click_pair(registration, viewer, ref_xy, mov_xy)
    widget.save_requested.emit()
    assert len(created) == 1
    save_a = created.pop(0)  # the save of A stays pending

    assert ctrl.load_registration_store(store_b) is None
    load_b = created.pop(0)
    load_b.finish()  # B is applied while A's save is still queued
    assert registration.session.path == store_b
    save_a.finish()  # the stale save now reports
    assert _on_disk_labelling(store_a)["revision"] == 1

    registration.flush_pending_updates()
    _click_pair(registration, viewer, ref_xy, mov_xy)
    widget.save_requested.emit()
    assert len(created) == 1  # a new save worker started for B
    assert statuses[-1] != "A save is already running."
    assert registration._saving is True
    created.pop(0).finish()
    assert registration._saving is False
    assert widget._finish_reason != "Saving…"
    assert _on_disk_labelling(store_b)["revision"] == 1
