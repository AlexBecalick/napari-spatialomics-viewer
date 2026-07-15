"""Tests for the micron scale-bar overlay and zoom-aware label interpolation."""
from __future__ import annotations

import os
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


def _fake_zoom_viewer(zoom: float, size=(1000.0, 800.0)):
    canvas = types.SimpleNamespace(size=size, native=None)
    return types.SimpleNamespace(
        layers=[],
        mouse_drag_callbacks=[],
        camera=types.SimpleNamespace(
            zoom=zoom,
            events=types.SimpleNamespace(zoom=types.SimpleNamespace(connect=lambda *a, **k: None)),
        ),
        window=types.SimpleNamespace(_qt_viewer=types.SimpleNamespace(canvas=canvas)),
    )


def _controller(viewer, interpolation="linear"):
    return V.ComparisonViewerController(
        viewer, {}, types.SimpleNamespace(label_interpolation=interpolation)
    )


def test_scale_bar_format_um():
    fmt = V.ScaleBarOverlay._format_um
    assert fmt(250.0) == "250 µm"
    assert fmt(12.5) == "12.5 µm"
    assert fmt(3.14159) == "3.14 µm"
    assert fmt(1500.0) == "1500 µm"  # stays in microns as requested


def test_packaged_application_icon_is_installed_on_qt_window(qapp):
    from qtpy.QtWidgets import QWidget

    qt_window = QWidget()
    viewer = types.SimpleNamespace(window=types.SimpleNamespace(_qt_window=qt_window))

    assert V.application_icon_path().is_file()
    assert V.install_application_icon(viewer)
    assert not qapp.windowIcon().isNull()
    assert not qt_window.windowIcon().isNull()


def test_scale_bar_overlay_sizing_and_position(qapp):
    from qtpy.QtWidgets import QWidget

    parent = QWidget()
    parent.resize(1200, 900)
    bar = V.ScaleBarOverlay(parent, width_fraction=0.25, margin=28)
    bar.reposition()
    assert bar._bar_px == int(1200 * 0.25)  # a quarter of the canvas width
    geom = bar.geometry()
    # Pinned toward the bottom-right corner with a margin.
    assert geom.x() > parent.width() / 2
    assert geom.y() > parent.height() / 2
    assert parent.width() - geom.right() >= 20
    bar.set_um_per_px(2.0)  # 300 px * 2 µm/px = 600 µm; just ensure no paint crash
    bar.repaint()


def test_label_interpolation_toggles_with_zoom(qapp):
    viewer = _fake_zoom_viewer(zoom=8.0)  # min(1000,800)=800; 800/8 = 100 µm <= 150
    ctrl = _controller(viewer, "linear")
    layer = types.SimpleNamespace(
        name="Segmentation | Proseg", interpolation2d="linear"
    )
    viewer.layers.append(layer)

    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"  # zoomed in past 150 µm -> crisp

    viewer.camera.zoom = 1.0  # 800 µm visible -> smooth
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "linear"


def test_label_interpolation_respects_explicit_nearest(qapp):
    viewer = _fake_zoom_viewer(zoom=1.0)  # 800 µm visible (would be 'linear' if auto)
    ctrl = _controller(viewer, "nearest")
    layer = types.SimpleNamespace(name="Segmentation | Seg", interpolation2d="nearest")
    viewer.layers.append(layer)
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"  # explicit choice not overridden


def test_install_canvas_overlays_without_canvas_is_safe(qapp):
    viewer = _fake_zoom_viewer(zoom=4.0)
    ctrl = _controller(viewer, "linear")
    # canvas.native is None here -> scale bar skipped, camera hook + initial update
    # must still run without raising.
    ctrl.install_canvas_overlays()
    assert ctrl._scale_bar is None


def test_label_outline_uses_thread_safe_global_dask_cache(qapp):
    captured = {}

    class FakeViewer:
        layers = []
        mouse_drag_callbacks = []

        def add_image(self, data, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                events=types.SimpleNamespace(
                    visible=types.SimpleNamespace(connect=lambda *args, **kwargs: None)
                )
            )

    args = types.SimpleNamespace(
        shape_opacity=0.95,
        hide_shapes=False,
        label_interpolation="linear",
    )
    ctrl = V.ComparisonViewerController(FakeViewer(), {}, args)
    ctrl._apply_label_zoom_interpolation = lambda *args, **kwargs: None

    added = ctrl._finish_label_layer(
        "MERSCOPE",
        "MOSAIK_cellpose_labels",
        {
            "outline_data": [np.zeros((4, 4), dtype=np.uint8)],
            "napari_affine": np.eye(3),
            "n_levels": 1,
            "display_label_key": "cached_outline",
            "base_shape": (4, 4),
            "base_dtype": np.dtype("uint8"),
            "outline_width": 1,
        },
    )

    assert added == 1
    assert captured["cache"] is True


def test_named_object_layer_uses_string_object_ids(qapp):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    ctrl = V.ComparisonViewerController(
        viewer,
        {},
        types.SimpleNamespace(shape_edge_width=1.0),
    )

    layer = ctrl._ensure_distance_object_annotation_layer(
        "MERSCOPE",
        "test_plaques",
    )
    layer.add(
        np.asarray([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]),
        shape_type="polygon",
    )

    assert layer.metadata["distance_object_type"] == "test_plaques"
    assert layer.features[V.OBJECT_ID_PROPERTY].dtype == object
    assert list(layer.features[V.OBJECT_ID_PROPERTY]) == [""]


def _cell_type_controller_with_synthetic_store(qapp, monkeypatch):
    """A controller whose store read + heavy mask build are stubbed out.

    Four annotated cells (mask ids 1-4) over a 2x3 mask; id 5 in the mask is
    unannotated (must stay transparent). ProSeg has annotations, Cellpose does
    not, mirroring the real MERSCOPE store.
    """
    from pathlib import Path

    from napari.components import ViewerModel
    from napari_compare_xenium_merscope.utils import CellTypeAssignments

    viewer = ViewerModel()
    cfg = V.DatasetConfig(name="MERSCOPE", zarr_path=Path("/does/not/matter"))
    args = types.SimpleNamespace(
        shape_opacity=0.9, image_pyramid_downsample=4, label_interpolation="nearest"
    )
    ctrl = V.ComparisonViewerController(viewer, {"MERSCOPE": cfg}, args)
    ctrl.active_dataset = "MERSCOPE"
    ctrl._active_sdata = object()  # only truthiness is checked before the stubs

    assignments = CellTypeAssignments(
        segmentation="proseg",
        table_key="table_MOSAIK_proseg_clustering_squidpy",
        instance_key="cell",
        cell_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        broad=np.array(["Neurons", "Neurons", "Astrocytes", "Astrocytes"]),
        fine=np.array(["Neurons:0", "Neurons:1", "Astrocytes:0", "Astrocytes:0"]),
    )
    monkeypatch.setattr(
        V, "clustering_table_key_for_segmentation",
        lambda z, seg: assignments.table_key if seg == "proseg" else None,
    )
    monkeypatch.setattr(
        V, "load_cell_type_assignments",
        lambda z, seg: assignments if seg == "proseg" else None,
    )
    # id 5 is present in the mask but absent from the annotation table.
    mask = np.array([[0, 1, 2, 5], [3, 4, 0, 0]], dtype=np.uint32)
    monkeypatch.setattr(ctrl, "_ensure_label_key_for_segmentation", lambda seg: "MOSAIK_proseg_labels")
    monkeypatch.setattr(ctrl, "_build_cellpose_label_display", lambda key: ([mask], np.eye(3)))
    # Run the "heavy" build synchronously so assertions see the layer immediately.
    monkeypatch.setattr(V, "thread_worker", None)
    return viewer, ctrl


def _labels_layer(viewer):
    from napari.layers import Labels

    layers = [layer for layer in viewer.layers if isinstance(layer, Labels)]
    assert len(layers) == 1
    return layers[0]


def test_cell_type_overlay_fills_masks_by_broad_type(qapp, monkeypatch):
    viewer, ctrl = _cell_type_controller_with_synthetic_store(qapp, monkeypatch)

    ctrl.set_cell_type_kind("MERSCOPE", "broad")

    layer = _labels_layer(viewer)
    assert layer.opacity == pytest.approx(0.9)
    color_dict = layer.colormap.color_dict
    # Background (0) + unmapped (None) transparent; the 4 annotated cells coloured.
    from napari_compare_xenium_merscope.utils import TRANSPARENT_RGBA

    assert np.allclose(np.asarray(color_dict[0]), np.asarray(TRANSPARENT_RGBA))
    assert set(color_dict) == {0, None, 1, 2, 3, 4}
    # id 5 (unannotated) is absent -> renders transparent.
    assert 5 not in color_dict
    # The two Neuron cells share one broad colour; Astrocytes differ.
    assert np.array_equal(color_dict[1], color_dict[2])
    assert np.array_equal(color_dict[3], color_dict[4])
    assert not np.array_equal(color_dict[1], color_dict[3])


def test_cell_type_overlay_toggle_and_level_and_opacity(qapp, monkeypatch):
    viewer, ctrl = _cell_type_controller_with_synthetic_store(qapp, monkeypatch)
    ctrl.set_cell_type_kind("MERSCOPE", "broad")
    layer = _labels_layer(viewer)

    # Untick "Neurons" -> its two cells drop out of the colour dict (transparent).
    ctrl.set_cell_type_visible("MERSCOPE", "Neurons", False)
    color_dict = layer.colormap.color_dict
    assert set(color_dict) == {0, None, 3, 4}

    # Re-enable everything.
    ctrl.set_all_cell_types_visible("MERSCOPE", True)
    assert set(layer.colormap.color_dict) == {0, None, 1, 2, 3, 4}

    # Switch to fine: the two neuron cells now get distinct colours.
    ctrl.set_cell_type_kind("MERSCOPE", "fine")
    fine_layer = _labels_layer(viewer)
    fine_dict = fine_layer.colormap.color_dict
    assert set(fine_dict) == {0, None, 1, 2, 3, 4}
    assert not np.array_equal(fine_dict[1], fine_dict[2])  # Neurons:0 vs Neurons:1

    # Opacity slider drives the live layer opacity.
    ctrl.set_cell_type_opacity("MERSCOPE", 0.25)
    assert fine_layer.opacity == pytest.approx(0.25)


def test_cell_type_overlay_sits_below_transcripts(qapp, monkeypatch):
    from napari.layers import Labels, Points

    viewer, ctrl = _cell_type_controller_with_synthetic_store(qapp, monkeypatch)

    # Simulate loaded transcripts: a Points layer registered on the gene state.
    gene_layer = viewer.add_points(np.zeros((1, 2)), name="Genes | Disc")
    ctrl._gene_inspector_states["MERSCOPE"] = V.GeneInspectorState(
        dataset="MERSCOPE",
        points_key="transcripts",
        store=object(),
        gene_visuals={},
        layer_names=["Genes | Disc"],
        enabled_genes=set(),
        spot_size=0.5,
    )

    ctrl.set_cell_type_kind("MERSCOPE", "broad")

    labels = [layer for layer in viewer.layers if isinstance(layer, Labels)][0]
    points = [layer for layer in viewer.layers if isinstance(layer, Points)][0]
    # The fill must be below the transcript points in the layer stack.
    assert viewer.layers.index(labels) < viewer.layers.index(points)


def test_cell_type_overlay_absent_annotations_is_graceful(qapp, monkeypatch):
    viewer, ctrl = _cell_type_controller_with_synthetic_store(qapp, monkeypatch)

    # Cellpose has no clustering table -> no layer added, no exception.
    ctrl.set_cell_type_segmentation("MERSCOPE", "cellpose")
    ctrl.set_cell_type_kind("MERSCOPE", "broad")
    from napari.layers import Labels

    assert not any(isinstance(layer, Labels) for layer in viewer.layers)
