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


def test_left_panel_adapter_collapses_controls_and_aggregates_gene_rows(qapp):
    from napari._qt.containers.qt_layer_list import QtLayerList
    from napari.components import ViewerModel
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

    hidden_names = {
        str(layer_view.model().index(row, 0).data())
        for row in range(layer_view.model().rowCount())
        if layer_view.isRowHidden(row)
    }
    assert hidden_names == {"Genes | Disc", "Genes | Ring"}
    assert not adapter.gene_row.isHidden()
    # Only the one non-gene row contributes to the native list height, so the
    # separate aggregate row sits directly beneath Image | DAPI.
    assert layer_view.maximumHeight() <= 40
    assert layer_layout.itemAt(3).spacerItem() is not None
    assert layer_buttons.newPointsButton.isHidden()
    assert layer_buttons.newShapesButton.isHidden()
    assert layer_buttons.newLabelsButton.isHidden()
    assert not layer_buttons.deleteButton.isHidden()
    assert viewer_buttons.consoleButton.isHidden()
    assert not viewer_buttons.resetViewButton.isHidden()
    assert not adapter._layer_controls_expanded
    assert controls.isHidden()

    adapter.gene_row.toggle_visibility()
    assert not any(layer.visible for layer in model.layers if layer.name.startswith("Genes | "))
    adapter.expand_layer_controls()
    assert adapter._layer_controls_expanded
    assert not controls.isHidden()
