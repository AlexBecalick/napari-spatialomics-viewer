"""Tests for empty and dataset-backed viewer startup states."""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from napari_compare_xenium_merscope import viewer as V


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


def _panel(datasets: list[str], initial_dataset: str | None = None, **overrides):
    callback = lambda *args, **kwargs: None
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


def test_parse_args_allows_launch_without_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["napari-compare-xenium-merscope"])

    args = V.parse_args()

    assert args.merscope_zarr is None
    assert args.xenium_zarr is None
    assert args.background_io_workers == 2
    assert args.session_cache_gb is None
    assert args.disable_async_slicing is False
    assert args.label_interpolation == "nearest"
    assert args.paired_view is None


@pytest.mark.parametrize("paired_view", ["side-by-side", "stacked-overlay"])
def test_parse_args_accepts_paired_view(monkeypatch, paired_view):
    monkeypatch.setattr(
        sys,
        "argv",
        ["napari-compare-xenium-merscope", "--paired-view", paired_view],
    )

    args = V.parse_args()

    assert args.paired_view == paired_view


def test_configure_napari_async_slicing_can_be_enabled_and_disabled():
    from napari.settings import get_settings

    original = bool(get_settings().experimental.async_)
    try:
        assert V.configure_napari_async_slicing(True)
        assert get_settings().experimental.async_ is True
        assert V.configure_napari_async_slicing(False) is False
        assert get_settings().experimental.async_ is False
    finally:
        V.configure_napari_async_slicing(original)


def test_empty_startup_selects_dataset_loader(qapp):
    panel = _panel([])

    assert panel.current_dataset == ""
    assert panel._tab_stack.currentIndex() == 6
    assert panel._tab_group.button(6).text() == "Dataset loader"
    assert panel._tab_group.button(6).isChecked()
    assert not panel._reload_button.isEnabled()


def test_no_opengl_package_smoke_test_builds_empty_control_panel(qapp):
    V.run_package_smoke_test_without_opengl()


def test_dataset_startup_keeps_gene_inspector_selected(qapp):
    panel = _panel(["MERSCOPE"], initial_dataset="MERSCOPE")

    assert panel.current_dataset == "MERSCOPE"
    assert panel._tab_stack.currentIndex() == 0
    assert panel._reload_button.isEnabled()


def test_paired_load_buttons_pass_their_view_modes(qapp, tmp_path, monkeypatch):
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QPushButton

    settings = QSettings(str(tmp_path / "viewer-settings.ini"), QSettings.IniFormat)
    opened = []
    panel = _panel(
        [],
        settings=settings,
        load_paired_callback=lambda merscope, xenium, mode: opened.append(
            (merscope, xenium, mode)
        )
        or True,
    )

    paired_button_labels = [
        button.text()
        for button in panel.findChildren(QPushButton)
        if button.text().startswith("Load new paired dataset")
    ]
    assert paired_button_labels == [
        "Load new paired dataset side-by-side",
        "Load new paired dataset stacked overlay",
    ]

    merscope = str(tmp_path / "merscope" / "spatialdata.zarr")
    xenium = str(tmp_path / "xenium" / "spatialdata.zarr")
    browsed = iter([merscope, xenium, merscope, xenium])
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: next(browsed))

    panel._load_paired_side_button.click()
    panel._load_paired_overlay_button.click()

    assert opened == [
        (merscope, xenium, V.ViewMode.SIDE_BY_SIDE),
        (merscope, xenium, V.ViewMode.STACKED_OVERLAY),
    ]


def test_pending_paired_load_is_recorded_only_after_validation(
    qapp, tmp_path, monkeypatch
):
    from qtpy.QtCore import QSettings

    settings = QSettings(str(tmp_path / "viewer-settings.ini"), QSettings.IniFormat)
    panel = _panel(
        [],
        settings=settings,
        load_paired_callback=lambda *_args: None,
    )
    merscope = tmp_path / "merscope" / "spatialdata.zarr"
    xenium = tmp_path / "xenium" / "spatialdata.zarr"
    browsed = iter([str(merscope), str(xenium)])
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: next(browsed))
    opened = []
    panel.dataset_open_requested.connect(lambda: opened.append(True))

    panel._load_paired_side_button.click()

    assert panel.recent_datasets == []
    assert opened == []

    panel.paired_dataset_loaded(
        merscope,
        xenium,
        V.ViewMode.SIDE_BY_SIDE,
    )

    assert panel.recent_datasets[0]["kind"] == "paired"
    assert opened == [True]


def test_annotations_tab_only_offers_combined_cortical_depth_export(qapp):
    from qtpy.QtWidgets import QPushButton

    panel = _panel(["MERSCOPE"], initial_dataset="MERSCOPE")
    button_labels = {button.text() for button in panel.findChildren(QPushButton)}

    assert "Export Combined GeoJSON" in button_labels
    assert "Export Separate GeoJSONs" not in button_labels


def test_annotations_tab_expands_layer_controls(qapp):
    expanded = []
    panel = _panel(
        ["MERSCOPE"],
        initial_dataset="MERSCOPE",
        expand_layer_controls_callback=lambda: expanded.append(True),
    )

    panel._tab_group.button(4).click()

    assert expanded == [True]


def test_recent_datasets_persist_newest_ten_and_reopen(qapp, tmp_path):
    from qtpy.QtCore import QSettings

    settings = QSettings(str(tmp_path / "viewer-settings.ini"), QSettings.IniFormat)
    opened = []
    panel = _panel(
        [],
        settings=settings,
        load_standalone_callback=lambda platform, path: opened.append((platform, path)) or True,
    )
    paths = [tmp_path / f"sample-{index}" / "spatialdata.zarr" for index in range(12)]
    for path in paths:
        panel.record_recent_dataset("MERSCOPE", path)

    assert len(panel.recent_datasets) == V.MAX_RECENT_DATASETS
    assert panel.recent_datasets[0]["path"] == str(paths[-1].absolute())
    assert panel.recent_datasets[-1]["path"] == str(paths[2].absolute())
    assert "sample-11/spatialdata.zarr" in panel._recent_dataset_list.item(0).text()

    # Reopening an older entry moves it to the top without creating a duplicate.
    panel.record_recent_dataset("XENIUM", paths[5])
    assert len(panel.recent_datasets) == V.MAX_RECENT_DATASETS
    assert panel.recent_datasets[0] == {
        "platform": "XENIUM",
        "path": str(paths[5].absolute()),
    }

    restored = _panel(
        [],
        settings=settings,
        load_standalone_callback=lambda platform, path: opened.append((platform, path)) or True,
    )
    restored._recent_dataset_list.setCurrentRow(0)
    restored._open_recent_dataset_button.click()

    assert opened[-1] == ("XENIUM", str(paths[5].absolute()))
    assert len(restored.recent_datasets) == V.MAX_RECENT_DATASETS


def test_paired_recent_persists_as_one_entry_and_reopens_with_mode(qapp, tmp_path):
    from qtpy.QtCore import QSettings

    settings = QSettings(str(tmp_path / "viewer-settings.ini"), QSettings.IniFormat)
    merscope = tmp_path / "merscope-sample" / "spatialdata.zarr"
    xenium = tmp_path / "xenium-sample" / "spatialdata.zarr"
    panel = _panel([], settings=settings)

    panel.record_recent_pair(merscope, xenium, V.ViewMode.STACKED_OVERLAY)

    assert panel.recent_datasets == [
        {
            "kind": "paired",
            "merscope_path": str(merscope.absolute()),
            "xenium_path": str(xenium.absolute()),
            "view_mode": "stacked-overlay",
        }
    ]
    label = panel._recent_dataset_list.item(0).text()
    assert label == (
        "PAIRED (stacked-overlay) — "
        "merscope-sample/spatialdata.zarr + xenium-sample/spatialdata.zarr"
    )

    opened = []
    restored = _panel(
        [],
        settings=settings,
        load_paired_callback=lambda merscope_path, xenium_path, mode: opened.append(
            (merscope_path, xenium_path, mode)
        )
        or True,
    )
    restored._recent_dataset_list.setCurrentRow(0)
    restored._open_recent_dataset_button.click()

    assert opened == [
        (
            str(merscope.absolute()),
            str(xenium.absolute()),
            V.ViewMode.STACKED_OVERLAY,
        )
    ]
    assert restored.recent_datasets == panel.recent_datasets


def test_left_panel_adapter_collapses_controls_and_aggregates_gene_rows(qapp):
    from napari._qt.containers.qt_layer_list import QtLayerList
    from napari.components import ViewerModel
    from napari_compare_xenium_merscope.paired_views import (
        LayerIdentity,
        attach_layer_identity,
    )
    from qtpy.QtWidgets import (
        QDockWidget,
        QHBoxLayout,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class Buttons(QWidget):
        def __init__(self, names):
            super().__init__()
            layout = QHBoxLayout(self)
            for name in names:
                button = QPushButton(name)
                setattr(self, name, button)
                layout.addWidget(button)

    class Title(QWidget):
        def __init__(self):
            super().__init__()
            self.setLayout(QHBoxLayout())

    class Dock(QDockWidget):
        def __init__(self, widget):
            super().__init__()
            self.title = Title()
            self.setTitleBarWidget(self.title)
            self.setWidget(widget)

        def inner_widget(self):
            return self.widget()

    class MainWindow(QWidget):
        def resizeDocks(self, *_args):
            pass

    model = ViewerModel()
    model.add_points(np.zeros((1, 2)), name="Genes | Disc")
    model.add_points(np.zeros((1, 2)), name="Genes | Ring")
    model.add_image(np.zeros((2, 2)), name="Image | DAPI")

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
        window=types.SimpleNamespace(_qt_viewer=qt_viewer, _qt_window=MainWindow()),
    )

    adapter = V.NapariLeftPanelAdapter(viewer)
    qapp.processEvents()
    adapter._refresh_gene_presentation()
    default_drag_mode = layer_view.dragDropMode()
    default_edit_triggers = layer_view.editTriggers()

    hidden_names = {
        str(layer_view.model().index(row, 0).data())
        for row in range(layer_view.model().rowCount())
        if layer_view.isRowHidden(row)
    }
    assert hidden_names == {"Genes | Disc", "Genes | Ring"}
    assert not adapter.gene_row.isHidden()
    # The native layer list consumes spare dock height instead of being capped
    # to its current rows; the separate aggregate remains immediately below it.
    assert layer_view.maximumHeight() == 16777215
    assert (
        layer_view.sizePolicy().verticalPolicy()
        == layer_view.sizePolicy().Expanding
    )
    assert layer_layout.stretch(layer_layout.indexOf(layer_view)) == 1
    assert layer_layout.indexOf(adapter.rotation_control) == 3
    assert adapter.rotation_slider.minimum() == 0
    assert adapter.rotation_slider.maximum() == 36000
    assert adapter.rotation_spin.minimum() == 0.0
    assert adapter.rotation_spin.maximum() == 360.0
    assert layer_layout.indexOf(viewer_buttons) == 4
    assert layer_buttons.newPointsButton.isHidden()
    assert layer_buttons.newShapesButton.isHidden()
    assert layer_buttons.newLabelsButton.isHidden()
    assert not layer_buttons.deleteButton.isHidden()
    assert viewer_buttons.consoleButton.isHidden()
    assert not viewer_buttons.resetViewButton.isHidden()
    assert not adapter._layer_controls_expanded
    assert controls.isHidden()

    merscope_genes = [
        attach_layer_identity(
            model.add_points(np.zeros((1, 2)), name=f"merscope-{symbol}"),
            LayerIdentity("MERSCOPE", "genes", channel=symbol),
        )
        for symbol in ("disc", "ring")
    ]
    xenium_genes = [
        attach_layer_identity(
            model.add_points(np.zeros((1, 2)), name=f"xenium-{symbol}"),
            LayerIdentity("XENIUM", "genes", channel=symbol),
        )
        for symbol in ("disc", "ring")
    ]
    adapter.set_paired_mode(True)
    qapp.processEvents()
    adapter._refresh_gene_presentation()
    assert not layer_buttons.deleteButton.isEnabled()
    assert layer_view.dragDropMode() == layer_view.NoDragDrop
    assert layer_view.editTriggers() == layer_view.NoEditTriggers
    assert adapter.gene_row.isHidden()
    assert all(not row.isHidden() for row in adapter.paired_gene_rows.values())
    hidden_names = {
        str(layer_view.model().index(row, 0).data())
        for row in range(layer_view.model().rowCount())
        if layer_view.isRowHidden(row)
    }
    assert hidden_names == {
        "Genes | Disc",
        "Genes | Ring",
        *(str(layer.name) for layer in merscope_genes),
        *(str(layer.name) for layer in xenium_genes),
    }

    adapter.paired_gene_rows["MERSCOPE"].toggle_visibility()
    assert not any(layer.visible for layer in merscope_genes)
    assert all(layer.visible for layer in xenium_genes)

    adapter.set_paired_mode(False)
    assert layer_buttons.deleteButton.isEnabled()
    assert layer_view.dragDropMode() == default_drag_mode
    assert layer_view.editTriggers() == default_edit_triggers
    assert not adapter.gene_row.isHidden()
    assert all(row.isHidden() for row in adapter.paired_gene_rows.values())

    adapter.gene_row.toggle_visibility()
    assert not any(layer.visible for layer in model.layers if layer.name.startswith("Genes | "))
    adapter.expand_layer_controls()
    assert adapter._layer_controls_expanded
    assert not controls.isHidden()


def test_left_panel_rotation_keeps_layers_registered_and_inputs_synchronised(qapp):
    from napari._qt.containers.qt_layer_list import QtLayerList
    from napari.components import ViewerModel
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

    model = ViewerModel()
    image = model.add_image(np.zeros((2, 2)), name="Image | DAPI")
    points = model.add_points(np.asarray([[0.0, 0.0]]), name="Genes | Disc")
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
    adapter = V.NapariLeftPanelAdapter(viewer)

    adapter.rotation_spin.setValue(90.0)
    expected_origin_rotation = np.asarray([1.0, 0.0])
    assert np.allclose(image.data_to_world((0.0, 0.0)), expected_origin_rotation)
    assert np.allclose(points.data_to_world((0.0, 0.0)), expected_origin_rotation)
    assert adapter.rotation_slider.value() == 9000

    # Layers loaded after the user rotates inherit the same world transform.
    labels = model.add_labels(np.zeros((2, 2), dtype=np.uint8), name="Segmentation")
    assert np.allclose(labels.data_to_world((0.0, 0.0)), expected_origin_rotation)

    adapter.rotation_slider.setValue(12345)
    assert adapter.rotation_spin.value() == pytest.approx(123.45)
    adapter.rotation_spin.setValue(360.0)
    assert adapter.rotation_slider.value() == 36000
    assert np.allclose(image.affine.affine_matrix, np.eye(3), atol=1e-12)
