"""Tests for empty and dataset-backed viewer startup states."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from napari_compare_xenium_merscope import viewer as V


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


def _panel(datasets: list[str], initial_dataset: str | None = None):
    callback = lambda *args, **kwargs: None
    return V.ViewerControlPanel(
        datasets=datasets,
        gene_inspector_widget=None,
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
        export_separate_annotations_callback=callback,
        load_paired_callback=callback,
        load_standalone_callback=callback,
        initial_dataset=initial_dataset,
    )


def test_parse_args_allows_launch_without_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["napari-compare-xenium-merscope"])

    args = V.parse_args()

    assert args.merscope_zarr is None
    assert args.xenium_zarr is None


def test_empty_startup_selects_dataset_loader(qapp):
    panel = _panel([])

    assert panel.current_dataset == ""
    assert panel._tab_stack.currentIndex() == 5
    assert panel._tab_group.button(5).text() == "Dataset loader"
    assert panel._tab_group.button(5).isChecked()
    assert not panel._reload_button.isEnabled()


def test_no_opengl_package_smoke_test_builds_empty_control_panel(qapp):
    V.run_package_smoke_test_without_opengl()


def test_dataset_startup_keeps_gene_inspector_selected(qapp):
    panel = _panel(["MERSCOPE"], initial_dataset="MERSCOPE")

    assert panel.current_dataset == "MERSCOPE"
    assert panel._tab_stack.currentIndex() == 0
    assert panel._reload_button.isEnabled()
