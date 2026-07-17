"""Tests for scale-bar overlays and segmentation display behaviour."""
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


def _controller(viewer, interpolation="nearest"):
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


def test_label_interpolation_stays_nearest_at_every_zoom(qapp):
    viewer = _fake_zoom_viewer(zoom=8.0)
    ctrl = _controller(viewer, "nearest")
    layer = types.SimpleNamespace(
        name="Segmentation | Proseg", interpolation2d="linear"
    )
    viewer.layers.append(layer)

    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"

    viewer.camera.zoom = 0.5
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"


def test_label_interpolation_respects_explicit_linear_compatibility(qapp):
    viewer = _fake_zoom_viewer(zoom=1.0)
    ctrl = _controller(viewer, "linear")
    layer = types.SimpleNamespace(name="Segmentation | Seg", interpolation2d="nearest")
    viewer.layers.append(layer)
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "linear"


def test_outline_coverage_colormap_keeps_partial_boundaries_readable():
    alpha = 0.8
    colormap = V.outline_coverage_colormap(
        "outline",
        np.asarray([1.0, 0.25, 0.0, 1.0], dtype=np.float32),
        alpha=alpha,
    )
    mapped = colormap.map(np.asarray([0.0, 0.25, 1.0], dtype=np.float32))

    assert mapped[0, 3] == pytest.approx(0.0)
    assert mapped[1, 3] == pytest.approx(alpha * 0.5, abs=1e-3)
    assert mapped[2, 3] == pytest.approx(alpha)
    assert np.allclose(mapped[:, :3], np.asarray([1.0, 0.25, 0.0]))


def test_display_pyramids_append_small_tiled_overview_levels():
    if V.da is None:
        pytest.skip("dask is unavailable")

    labels = np.zeros((16, 16), dtype=np.uint32)
    labels[2:14, 2:14] = 1
    outlines = V.lazy_outline_pyramid(labels, width=1, min_size=4, max_levels=4)
    finest = np.asarray(outlines[0].compute())
    coarse = np.asarray(outlines[1].compute())
    assert finest.dtype == np.uint8
    assert finest.max() == V.OUTLINE_COVERAGE_MAX
    assert np.any((coarse > 0) & (coarse < V.OUTLINE_COVERAGE_MAX))
    assert max(outlines[-1].shape) <= 4

    tiled = V.rechunk_raster_levels_for_display(outlines)
    assert all(
        max(max(chunks) for chunks in level.chunks) <= V.RASTER_DISPLAY_TILE_SIZE
        for level in tiled
    )

    image_levels = [
        (
            "scale0",
            V.xr.DataArray(
                V.da.zeros((1, 4096, 4096), chunks=(1, 1024, 1024)),
                dims=("c", "y", "x"),
                coords={"c": ["DAPI"]},
            ),
        ),
        (
            "scale1",
            V.xr.DataArray(
                V.da.zeros((1, 2048, 2048), chunks=(1, 1024, 1024)),
                dims=("c", "y", "x"),
                coords={"c": ["DAPI"]},
            ),
        ),
    ]
    completed_images = V.complete_image_pyramid_for_display(image_levels)
    assert (
        max(completed_images[-1][1].shape[-2:])
        <= V.SYNTHETIC_IMAGE_PYRAMID_MIN_SIZE
    )


def test_install_canvas_overlays_without_canvas_is_safe(qapp):
    viewer = _fake_zoom_viewer(zoom=4.0)
    ctrl = _controller(viewer, "linear")
    # canvas.native is None here -> scale bar skipped, camera hook + initial update
    # must still run without raising.
    ctrl.install_canvas_overlays()
    assert ctrl._scale_bar is None


def test_scale_bar_stays_hidden_until_dataset_is_loaded(qapp):
    from qtpy.QtWidgets import QWidget

    viewer = _fake_zoom_viewer(zoom=4.0)
    canvas = QWidget()
    canvas.resize(1000, 800)
    viewer.window._qt_viewer.canvas.native = canvas
    ctrl = _controller(viewer, "linear")

    ctrl.install_canvas_overlays()

    assert ctrl._scale_bar is not None
    assert ctrl._scale_bar.isHidden()

    ctrl.active_dataset = "MERSCOPE"
    ctrl._active_sdata = object()
    ctrl._update_scale_bar_visibility()
    assert not ctrl._scale_bar.isHidden()

    ctrl.active_dataset = None
    ctrl._update_scale_bar_visibility()
    assert ctrl._scale_bar.isHidden()


def test_dismissed_welcome_overlay_redraws_canvas_when_layer_arrives(qapp):
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QWidget

    class SceneBackend:
        def __init__(self):
            self.resize_calls = []

        def resizeGL(self, width, height):
            self.resize_calls.append((width, height))

    class SceneCanvas:
        def __init__(self):
            self.updates = 0
            self._backend = SceneBackend()

        def update(self):
            self.updates += 1

    native = QWidget()
    native.resize(1000, 800)
    target = QWidget()
    scene_canvas = SceneCanvas()
    canvas = types.SimpleNamespace(native=native, _scene_canvas=scene_canvas)
    model = ViewerModel()
    viewer = types.SimpleNamespace(
        layers=model.layers,
        theme="dark",
        welcome_screen=types.SimpleNamespace(visible=False),
        window=types.SimpleNamespace(_qt_viewer=types.SimpleNamespace(canvas=canvas)),
    )
    overlay = V.DatasetWelcomeOverlay(viewer, target, start_visible=True)

    # Recent-dataset activation dismisses the overlay before workers add layers.
    overlay.dismiss()
    qapp.processEvents()
    updates_after_dismiss = scene_canvas.updates
    assert updates_after_dismiss == 1
    resize_calls_after_dismiss = len(scene_canvas._backend.resize_calls)
    assert scene_canvas._backend.resize_calls[-1] == (1000, 800)

    # The later layer event must refresh both the GL viewport geometry and the
    # paint even though dismiss() has already hidden the overlay.
    model.add_image(np.zeros((8, 8)), name="Image | DAPI")
    qapp.processEvents()
    assert scene_canvas.updates == updates_after_dismiss + 1
    assert len(scene_canvas._backend.resize_calls) == resize_calls_after_dismiss + 1
    assert scene_canvas._backend.resize_calls[-1] == (1000, 800)


def test_canvas_visibility_repair_nudges_and_restores_dock_width(qapp):
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QDockWidget, QWidget

    class SceneCanvas:
        def __init__(self):
            self.updates = 0

        def update(self):
            self.updates += 1

    class MainWindow:
        def __init__(self):
            self.calls = []

        def resizeDocks(self, docks, sizes, orientation):
            self.calls.append((list(sizes), orientation))
            dock = docks[0]
            dock.resize(int(sizes[0]), dock.height())

    viewer = _fake_zoom_viewer(zoom=4.0)
    native = QWidget()
    native.resize(900, 700)
    scene_canvas = SceneCanvas()
    viewer.window._qt_viewer.canvas = types.SimpleNamespace(
        native=native, _scene_canvas=scene_canvas
    )
    main_window = MainWindow()
    viewer.window._qt_window = main_window
    dock = QDockWidget()
    dock.resize(280, 500)
    ctrl = _controller(viewer)
    ctrl.set_viewer_controls_dock(dock)

    original_width = dock.width()
    assert ctrl._nudge_canvas_dock_layout()
    assert main_window.calls[0][0] == [original_width + 1]

    QTest.qWait(30)
    qapp.processEvents()
    assert main_window.calls[-1][0] == [original_width]
    assert dock.width() == original_width
    assert scene_canvas.updates >= 1


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
    assert captured["contrast_limits"] == (0.0, float(V.OUTLINE_COVERAGE_MAX))


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
    monkeypatch.setattr(
        ctrl,
        "_ensure_label_key_for_segmentation",
        lambda seg, **_kwargs: "MOSAIK_proseg_labels",
    )
    monkeypatch.setattr(
        ctrl,
        "_build_cellpose_label_display",
        lambda key, **_kwargs: ([mask], np.eye(3)),
    )
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


def test_label_ids_for_shapes_use_true_instance_ids():
    """Raster labels are the shapes' own index (instance id), not positional +1.

    The v1 rasterizer labelled the Nth polygon ``N+1``, so the cell-type overlay
    (keyed on the true instance id) coloured every cell with its neighbour's
    colour. Labels must now equal the instance id so the join is exact.
    """
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    ctrl = V.ComparisonViewerController.__new__(V.ComparisonViewerController)

    def _square(cx, cy):
        return Polygon([(cx, cy), (cx + 1, cy), (cx + 1, cy + 1), (cx, cy + 1)])

    # Non-contiguous integer ids (as a QC-filtered proseg set would be).
    gdf = geopandas.GeoDataFrame(
        geometry=[_square(0, 0), _square(2, 0), _square(4, 0)],
        index=[2, 94831, 94832],
    )
    series, dtype = ctrl._label_ids_for_shapes(gdf)
    assert np.dtype(dtype) == np.uint32
    assert series.loc[94831] == 94831  # not 94832 (the old positional +1)
    assert series.loc[94832] == 94832
    assert series.loc[2] == 2

    # Large string ids (merscope EntityID) promote to uint64 and keep their value.
    big = "3715213700018100006"
    gdf_str = geopandas.GeoDataFrame(geometry=[_square(0, 0)], index=[big])
    series_str, dtype_str = ctrl._label_ids_for_shapes(gdf_str)
    assert np.dtype(dtype_str) == np.uint64
    assert int(series_str.loc[big]) == int(big)
