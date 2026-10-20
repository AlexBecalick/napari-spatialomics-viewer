"""Tests for the floating UMAP view.

Loading and colouring helpers run directly; the scatter canvas renders under an
offscreen QApplication; and the controller wiring (tab sync, two-way cell
selection) runs against a lightweight fake napari viewer with no OpenGL canvas.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from shapely.geometry import Polygon

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.cell_umap import (
    UMAP_MISSING_RGBA,
    UMAP_MUTED_RGBA,
    CellUmapEmbedding,
    UmapControlWidget,
    UmapPlotCanvas,
    categorical_umap_colouring,
    cell_ids_as_int64,
    continuous_umap_colouring,
    emphasise_rows,
    labels_for_cell_ids,
    load_cell_umap,
    points_in_polygon,
    summarise_roi_labels,
    values_for_cell_ids,
)
from napari_compare_xenium_merscope.utils import CellTypeAssignments


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def synchronous_workers(monkeypatch):
    monkeypatch.setattr(V, "thread_worker", None)


def _u8(rgba):
    return tuple(int(round(255 * float(c))) for c in rgba)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _write_clustering_table(tmp_path, table_key, obs, umap):
    import anndata as ad

    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    if umap is not None:
        adata.obsm["X_umap"] = umap
    adata.uns["spatialdata_attrs"] = {"instance_key": "instance_id", "region_key": "region"}
    store = tmp_path / "store.zarr"
    if not store.exists():
        root = zarr.open_group(str(store), mode="w", zarr_format=2)
        root.create_group("tables")
    adata.write_zarr(str(store / "tables" / table_key))
    return str(store)


def test_load_cell_umap_reads_embedding_and_exact_instance_ids(tmp_path):
    big = np.array(
        [3715213700018100064, 3715213700018100066, 3715213700018100069, 3715213700018100066],
        dtype=np.uint64,
    )
    obs = pd.DataFrame(
        {
            "instance_id": big,
            "broad_class": pd.Categorical(["Neurons"] * 4),
        },
        index=[f"c{i}" for i in range(4)],
    )
    umap = np.array([[0.0, 1.0], [2.0, 3.0], [np.nan, 5.0], [9.0, 9.0]], dtype=np.float32)
    store = _write_clustering_table(tmp_path, "table_original_clustering_squidpy", obs, umap)

    embedding = load_cell_umap(store, "original")
    assert embedding is not None
    assert embedding.instance_key == "instance_id"
    # The NaN row is dropped and the duplicate id keeps its first row; the
    # 19-digit ids survive without float rounding.
    assert embedding.cell_ids.tolist() == [3715213700018100064, 3715213700018100066]
    assert embedding.coords.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert embedding.row_for_id(3715213700018100066) == 1
    assert embedding.row_for_id("3715213700018100064") == 0
    assert embedding.row_for_id(3715213700018100069) == -1
    # Other sources' tables are absent.
    assert load_cell_umap(store, "proseg") is None


def test_load_cell_umap_without_obsm_umap_is_none(tmp_path):
    obs = pd.DataFrame({"instance_id": np.array([1, 2], dtype=np.int64)}, index=["a", "b"])
    store = _write_clustering_table(tmp_path, "table_MOSAIK_proseg_clustering_squidpy", obs, None)
    assert load_cell_umap(store, "proseg") is None
    assert load_cell_umap(None, "proseg") is None


def test_id_joins_handle_strings_and_missing_cells():
    ids, valid = cell_ids_as_int64(np.array(["5", "x", "7.0", None], dtype=object))
    assert valid.tolist() == [True, False, True, False]
    assert ids[valid].tolist() == [5, 7]

    labels = labels_for_cell_ids([1, 2, 3], ["A", "B", "C"], [3, 9, 1])
    assert labels.tolist() == ["C", None, "A"]
    values = values_for_cell_ids(np.array([10, 20], dtype=np.uint32), [1.5, 2.5], [20, 30])
    assert values[0] == 2.5 and np.isnan(values[1])


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------


def test_categorical_colouring_matches_palette_and_layers_muted_cells_behind():
    colors = {"Astro": (1.0, 0.0, 0.0, 1.0), "Neuron": (0.0, 0.0, 1.0, 1.0)}
    labels = np.array(["Astro", "Neuron", None, "Neuron", "Astro"], dtype=object)
    colouring = categorical_umap_colouring(
        labels, colors, order=["Astro", "Neuron"], enabled={"Astro"}, title="Broad"
    )
    assert tuple(colouring.colors[0]) == (255, 0, 0, 255)
    assert tuple(colouring.colors[1]) == _u8(UMAP_MUTED_RGBA)       # unticked type
    assert tuple(colouring.colors[2]) == _u8(UMAP_MISSING_RGBA)     # unannotated
    # Muted + missing rows form the back layer; ticked rows draw on top.
    assert colouring.n_background == 3
    assert set(colouring.order[:3].tolist()) == {1, 2, 3}
    assert set(colouring.order[3:].tolist()) == {0, 4}
    legend = [(e.label, e.count, e.enabled) for e in colouring.legend]
    assert legend == [("Astro", 2, True), ("Neuron", 2, False), ("Unannotated", 1, True)]
    assert colouring.hover_text.tolist() == ["Astro", "Neuron", "Unannotated", "Neuron", "Astro"]


def test_categorical_draw_order_is_stable_when_types_toggle():
    colors = {"A": (1, 0, 0, 1), "B": (0, 1, 0, 1)}
    labels = np.array(["A", "B"] * 50, dtype=object)
    all_on = categorical_umap_colouring(labels, colors)
    b_off = categorical_umap_colouring(labels, colors, enabled={"A"})
    a_rows_all = [r for r in all_on.order.tolist() if labels[r] == "A"]
    a_rows_off = [r for r in b_off.order.tolist() if labels[r] == "A"]
    assert a_rows_all == a_rows_off


def test_continuous_colouring_uses_overlay_binning_and_draws_high_values_last():
    lut = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1], [1, 1, 1, 1]], dtype=float)
    values = np.array([10.0, np.nan, 0.0, 5.0])
    colouring = continuous_umap_colouring(values, lut, clip=(0.0, 10.0), title="DAPI median")
    assert tuple(colouring.colors[1]) == _u8(UMAP_MISSING_RGBA)
    assert tuple(colouring.colors[2]) == (0, 0, 0, 255)
    assert tuple(colouring.colors[3]) == (128, 128, 128, 255)  # floor(0.5 * 2) -> bin 1
    assert tuple(colouring.colors[0]) == (255, 255, 255, 255)
    assert colouring.order.tolist() == [1, 2, 3, 0]
    assert colouring.n_background == 1
    assert colouring.colorbar[1:] == (0.0, 10.0)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def test_canvas_rasterises_points_and_picks_the_nearest(qapp):
    canvas = UmapPlotCanvas()
    canvas.resize(200, 200)
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    canvas.set_embedding(coords)
    canvas.set_colouring(
        np.array([[255, 0, 0, 255], [0, 255, 0, 255]], dtype=np.uint8), order=[0, 1]
    )
    image = canvas.rendered_rgba()
    dpr = image.shape[1] / canvas.width()
    for (x, y), rgba in zip(canvas.data_to_view(coords), ([255, 0, 0, 255], [0, 255, 0, 255])):
        assert image[int(y * dpr), int(x * dpr)].tolist() == rgba

    x0, y0 = canvas.data_to_view(coords[0])[0]
    assert canvas.nearest_row(x0 + 1, y0 + 1) == 0
    assert canvas.nearest_row(100, 100) == -1  # empty middle of the plot
    back = canvas.view_to_data(*canvas.data_to_view(coords[1])[0])
    assert np.allclose(back, coords[1])

    clicked = []
    canvas.point_clicked.connect(clicked.append)
    from qtpy.QtCore import QPoint
    from qtpy.QtTest import QTest

    x1, y1 = canvas.data_to_view(coords[1])[0]
    QTest.mouseClick(canvas, V.Qt.LeftButton, pos=QPoint(int(x1), int(y1)))
    assert clicked == [1]


def test_canvas_drag_pans_without_clicking_and_wheel_zooms_about_cursor(qapp):
    from qtpy.QtCore import QPoint, QPointF

    canvas = UmapPlotCanvas()
    canvas.resize(200, 200)
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    canvas.set_embedding(coords)
    clicked = []
    canvas.point_clicked.connect(clicked.append)
    x1, y1 = canvas.data_to_view(coords[1])[0]
    from qtpy.QtCore import QEvent
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    def send(kind, x, buttons):
        pos = QPointF(x, y1)
        QApplication.sendEvent(
            canvas, QMouseEvent(kind, pos, V.Qt.LeftButton, buttons, V.Qt.NoModifier)
        )

    # (QTest.mouseMove cannot carry a held button on Qt 5, so send events directly.)
    send(QEvent.MouseButtonPress, x1, V.Qt.LeftButton)
    send(QEvent.MouseMove, x1 + 15, V.Qt.LeftButton)
    send(QEvent.MouseMove, x1 + 30, V.Qt.LeftButton)
    send(QEvent.MouseButtonRelease, x1 + 30, V.Qt.NoButton)
    assert clicked == []  # a drag is a pan, not a pick
    moved = canvas.data_to_view(coords[1])[0]
    assert moved[0] == pytest.approx(x1 + 30, abs=1.0) and moved[1] == pytest.approx(y1, abs=1.0)

    before = canvas._scale
    anchor = canvas.data_to_view(coords[0])[0]
    from qtpy.QtGui import QWheelEvent

    event = QWheelEvent(
        QPointF(*anchor), QPointF(*anchor), QPoint(0, 0), QPoint(0, 120),
        V.Qt.NoButton, V.Qt.NoModifier, V.Qt.NoScrollPhase, False,
    )
    canvas.wheelEvent(event)
    assert canvas._scale == pytest.approx(before * 1.2)
    assert canvas.data_to_view(coords[0])[0] == pytest.approx(anchor)  # cursor point fixed


def test_canvas_refits_on_resize_until_the_user_zooms(qapp):
    from qtpy.QtWidgets import QApplication

    canvas = UmapPlotCanvas()
    canvas.resize(700, 700)
    canvas.show()
    QApplication.processEvents()
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    canvas.set_embedding(coords)
    canvas.resize(300, 300)  # shrinking would crop a stale fit
    QApplication.processEvents()
    x, y = canvas.data_to_view(coords).T
    assert (x >= 0).all() and (x <= canvas.width()).all()
    assert (y >= 0).all() and (y <= canvas.height()).all()
    fitted = canvas._scale
    canvas._view_adjusted = True  # as after a wheel zoom or drag
    canvas.resize(700, 700)
    QApplication.processEvents()
    assert canvas._scale == fitted  # the user's zoom is kept
    canvas.hide()


def test_control_widget_only_reports_user_changes(qapp):
    calls = []
    widget = UmapControlWidget(
        set_window_visible_callback=lambda v: calls.append(("window", v)),
        set_source_callback=lambda s: calls.append(("source", s)),
        set_colour_callback=lambda *a: calls.append(("colour",) + a),
        set_point_size_callback=lambda r: calls.append(("size", r)),
    )
    widget.set_sources({"proseg": True, "cellpose": True}, "proseg")
    widget.set_statistic_options(["DAPI"], ["median", "mean"], True)
    widget.set_selection(source="cellpose", colour_by="statistic", channel="DAPI", statistic="mean")
    assert calls == []
    assert widget.current_source() == "cellpose"
    assert widget.current_colour()[:3] == ("statistic", "DAPI", "mean")

    widget._colour_combo.setCurrentIndex(widget._colour_combo.findData("fine"))
    assert calls[-1][:2] == ("colour", "fine")

    widget.set_statistic_options([], [], False)  # statistics unavailable
    stat_item = widget._colour_combo.model().item(widget._colour_combo.findData("statistic"))
    assert not stat_item.isEnabled()


# ---------------------------------------------------------------------------
# Controller wiring
# ---------------------------------------------------------------------------


class _FakeLayer:
    def __init__(self, name, data, **kw):
        self.name = name
        self.data = data
        self.visible = kw.get("visible", True)
        self.kw = kw
        self.colormap = kw.get("colormap")
        self.contour = 0
        self.edge_width = kw.get("edge_width")

    def add(self, data, **kw):
        self.data = list(self.data) + list(data)
        self.kw.update(kw)

    def refresh(self):
        pass


class _FakeLayerList(list):
    def move(self, src, dst):
        self.insert(dst, self.pop(src))


class _FakeCamera:
    def __init__(self):
        self.center = (0.0, 0.0)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.mouse_drag_callbacks = []
        self.window = types.SimpleNamespace(_qt_viewer=None)
        self.camera = _FakeCamera()

    def add_shapes(self, data, name=None, **kw):
        layer = _FakeLayer(name, list(data), **kw)
        self.layers.append(layer)
        return layer

    add_points = add_shapes

    def add_labels(self, data, name=None, **kw):
        layer = _FakeLayer(name, data, **kw)
        self.layers.append(layer)
        return layer


def _square(x0, y0, size=10):
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def _controller(qapp):
    args = types.SimpleNamespace(
        gene_spot_size=2.0,
        gene_max_render_points=40_000_000,
        gene_hide_background=False,
        gene_show_controls=False,
        random_state=42,
        shape_opacity=0.9,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "MERSCOPE"
    # ProSeg (the cell-inspector masks) and Cellpose cover the same two places
    # with different ids, as in a real MERSCOPE store.
    proseg = gpd.GeoDataFrame(
        {"geometry": [_square(0, 0), _square(20, 20)]}, index=np.array([202, 303], dtype=np.uint64)
    )
    cellpose = gpd.GeoDataFrame(
        {"geometry": [_square(1, 1, 8), _square(21, 21, 8)]}, index=np.array([7, 8], dtype=np.uint64)
    )
    ctrl._active_sdata = types.SimpleNamespace(
        shapes={"MOSAIK_proseg": proseg, "MOSAIK_cellpose": cellpose},
        points={},
    )
    ctrl._dataset_sessions["MERSCOPE"] = types.SimpleNamespace(
        cell_type_available={"proseg": True, "proseg_mask": False, "cellpose": True, "original": False},
        umap_available={"proseg": True, "cellpose": True},
        cellpose_channels=["DAPI", "PolyT"],
        cellpose_statistics=["median", "mean"],
        cellpose_values_available=True,
    )
    cell_types = ctrl._cell_type_state("MERSCOPE")
    ctrl._cache_cell_type_data(
        cell_types,
        "proseg",
        CellTypeAssignments(
            segmentation="proseg",
            table_key="table_MOSAIK_proseg_clustering_squidpy",
            instance_key="instance_id",
            cell_ids=np.asarray([202, 303], dtype=np.int64),
            broad=np.asarray(["Astrocytes", "Neurons"], dtype=object),
            fine=np.asarray(["Astrocytes:0", "Neurons:1"], dtype=object),
        ),
    )
    ctrl._cache_cell_type_data(
        cell_types,
        "cellpose",
        CellTypeAssignments(
            segmentation="cellpose",
            table_key="table_MOSAIK_cellpose_clustering_squidpy",
            instance_key="instance_id",
            cell_ids=np.asarray([7, 8], dtype=np.int64),
            broad=np.asarray(["Astrocytes", "Neurons"], dtype=object),
            fine=np.asarray(["Astrocytes:0", "Neurons:0"], dtype=object),
        ),
    )
    widget = UmapControlWidget(
        set_window_visible_callback=ctrl.set_umap_window_visible,
        set_source_callback=ctrl.set_umap_source,
        set_colour_callback=ctrl.set_umap_colour,
        set_point_size_callback=ctrl.set_umap_point_size,
        set_lasso_active_callback=ctrl.set_umap_lasso_active,
        clear_roi_callback=ctrl.clear_umap_roi,
    )
    ctrl.set_umap_widget(widget)
    state = ctrl._umap_state()
    state.embeddings["proseg"] = CellUmapEmbedding(
        segmentation="proseg",
        table_key="table_MOSAIK_proseg_clustering_squidpy",
        instance_key="instance_id",
        cell_ids=np.asarray([303, 202], dtype=np.int64),
        coords=np.asarray([[5.0, 5.0], [-5.0, -5.0]], dtype=np.float32),
    )
    state.embeddings["cellpose"] = CellUmapEmbedding(
        segmentation="cellpose",
        table_key="table_MOSAIK_cellpose_clustering_squidpy",
        instance_key="instance_id",
        cell_ids=np.asarray([7, 8], dtype=np.int64),
        coords=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )
    state.statistic_values[("DAPI", "median")] = (
        np.asarray([7, 8, 99], dtype=np.uint32),
        np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        (1.0, 3.0),
    )
    return ctrl, widget


def _open_window(ctrl, widget):
    widget._window_button.click()
    window = ctrl._umap_window
    assert window is not None and window.isVisible()
    return window


def test_umap_tab_opens_window_coloured_like_the_cell_type_panel(qapp):
    ctrl, widget = _controller(qapp)
    window = _open_window(ctrl, widget)
    assert widget._window_button.isChecked()
    colouring = window.colouring
    assert colouring is not None and colouring.title == "Broad cell type"
    scheme = ctrl._cell_type_state("MERSCOPE").schemes["proseg"]["broad"]
    # Row 0 is cell 303 (Neurons), row 1 is cell 202 (Astrocytes).
    assert tuple(colouring.colors[0]) == _u8(scheme.colors["Neurons"])
    assert tuple(colouring.colors[1]) == _u8(scheme.colors["Astrocytes"])
    assert window.describe_row(0) == "Cell 303 · Neurons"

    window.close()
    assert not widget._window_button.isChecked()


def test_spatial_selection_rings_the_cell_in_the_umap(qapp):
    ctrl, widget = _controller(qapp)
    window = _open_window(ctrl, widget)
    gdf = ctrl._active_sdata.shapes["MOSAIK_proseg"]
    ctrl._add_cell_selection("MERSCOPE", gdf.index[0], gdf.geometry.iloc[0])  # cell 202
    assert window.canvas.highlighted_rows() == [1]

    ctrl._clear_all_cell_selections()
    assert window.canvas.highlighted_rows() == []


def test_umap_click_selects_the_cell_in_the_spatial_view(qapp):
    ctrl, widget = _controller(qapp)
    window = _open_window(ctrl, widget)
    window.panel().point_clicked.emit(0)  # cell 303
    entries = ctrl._selected_cells["MERSCOPE"]
    assert [int(e["cell_id"]) for e in entries] == [303]
    assert window.canvas.highlighted_rows() == [0]
    # The spatial view pans to the picked cell (centre of the 20..30 square).
    assert ctrl.viewer.camera.center == pytest.approx((25.0, 25.0))


def test_cellpose_umap_maps_selections_across_segmentations(qapp):
    ctrl, widget = _controller(qapp)
    window = _open_window(ctrl, widget)
    ctrl.set_umap_source("cellpose")
    # Picking Cellpose cell 8 selects the ProSeg cell at the same place (303).
    window.panel().point_clicked.emit(1)
    assert [int(e["cell_id"]) for e in ctrl._selected_cells["MERSCOPE"]] == [303]
    assert window.canvas.highlighted_rows() == [1]
    # A ProSeg click is mapped back to the Cellpose cell containing its centroid.
    gdf = ctrl._active_sdata.shapes["MOSAIK_proseg"]
    ctrl._add_cell_selection("MERSCOPE", gdf.index[0], gdf.geometry.iloc[0])  # 202 -> 7
    assert window.canvas.highlighted_rows() == [1, 0]


def test_cell_type_tab_clicks_sync_the_umap(qapp, monkeypatch):
    ctrl, widget = _controller(qapp)
    monkeypatch.setattr(ctrl, "_start_cell_type_overlay_build", lambda state: None)
    window = _open_window(ctrl, widget)

    ctrl.set_cell_type_kind("MERSCOPE", "fine")
    assert ctrl._umap_state().colour_by == "fine"
    assert widget.current_colour()[0] == "fine"
    assert window.colouring.title == "Fine cell type"

    ctrl.set_cell_type_segmentation("MERSCOPE", "cellpose")
    assert ctrl._umap_state().source == "cellpose"
    assert widget.current_source() == "cellpose"

    # Unticking a type in the Cell type labels tab mutes it in the UMAP.
    ctrl.set_cell_type_visible("MERSCOPE", "Neurons:0", False)
    assert tuple(window.colouring.colors[1]) == _u8(UMAP_MUTED_RGBA)

    widget._sync_check.setChecked(False)
    ctrl.set_cell_type_kind("MERSCOPE", "broad")
    assert ctrl._umap_state().colour_by == "fine"


def test_statistic_overlay_sync_switches_umap_to_cellpose(qapp):
    ctrl, widget = _controller(qapp)
    window = _open_window(ctrl, widget)
    assert ctrl._umap_state().source == "proseg"

    ctrl._sync_umap_from_statistic("DAPI", "median", "magma")
    state = ctrl._umap_state()
    assert (state.source, state.colour_by, state.colormap) == ("cellpose", "statistic", "magma")
    assert widget.current_source() == "cellpose"
    assert widget.current_colour() == ("statistic", "DAPI", "median", "magma")
    colouring = window.colouring
    assert colouring.colorbar is not None and colouring.colorbar[1:] == (1.0, 3.0)
    assert colouring.values.tolist() == [1.0, 3.0]
    assert window.describe_row(1) == "Cell 8 · DAPI median: 3"

    # Choosing a source the statistic cannot join falls back to cell types.
    ctrl.set_umap_source("proseg")
    assert ctrl._umap_state().colour_by == "broad"
    assert widget.current_colour()[0] == "broad"


# ---------------------------------------------------------------------------
# Region of interest (lasso)
# ---------------------------------------------------------------------------


def test_points_in_polygon_closes_the_outline_and_keeps_both_lobes_of_a_figure_eight():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]  # last -> first is implied
    pts = np.array([[5, 5], [15, 5], [0.5, 9.5]])
    assert points_in_polygon(pts, square).tolist() == [True, False, True]
    # A freehand outline crossing itself: both loops count as inside.
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    assert points_in_polygon(np.array([[1, 5], [9, 5], [5, 1]]), bowtie).tolist() == [True, True, False]
    assert not points_in_polygon(pts, [(0, 0), (1, 1)]).any()


def test_emphasise_rows_mutes_the_rest_and_draws_the_region_on_top():
    colors = {"A": (1, 0, 0, 1)}
    base = categorical_umap_colouring(np.array(["A", "A", "A"], dtype=object), colors)
    emphasised = emphasise_rows(base, [2])
    assert tuple(emphasised.colors[2]) == (255, 0, 0, 255)
    assert tuple(emphasised.colors[0]) == _u8(UMAP_MUTED_RGBA)
    assert emphasised.order[-1] == 2 and emphasised.n_background == 2


def test_summarise_roi_labels_orders_types_by_count_with_unannotated_last():
    colors = {"Astro": (1, 0, 0, 1), "Neuron": (0, 0, 1, 1)}
    labels = np.array(["Neuron", "Astro", "Neuron", None, "Weird"], dtype=object)
    summary = summarise_roi_labels(labels, colors, level="Broad cell type", drawn_in="the UMAP")
    assert summary.total == 5
    assert [(s.label, s.count) for s in summary.slices] == [
        ("Neuron", 2), ("Astro", 1), ("Unannotated", 2)
    ]


def test_canvas_lasso_emits_the_outline_instead_of_panning_or_picking(qapp):
    from qtpy.QtCore import QEvent, QPointF
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    canvas = UmapPlotCanvas()
    canvas.resize(300, 300)
    canvas.set_embedding(np.array([[0.0, 0.0], [10.0, 10.0]]))
    canvas.set_lasso_mode(True)
    lassos, clicks = [], []
    canvas.lasso_completed.connect(lassos.append)
    canvas.point_clicked.connect(clicks.append)
    before = canvas.data_to_view([[0.0, 0.0]])[0].copy()

    def send(kind, x, y, buttons):
        QApplication.sendEvent(
            canvas,
            QMouseEvent(kind, QPointF(x, y), V.Qt.LeftButton, buttons, V.Qt.NoModifier),
        )

    (x0, y0), (x1, y1) = canvas.data_to_view([[-1.0, -1.0], [11.0, 11.0]])
    send(QEvent.MouseButtonPress, x0, y0, V.Qt.LeftButton)
    for x, y in ((x1, y0), (x1, y1), (x0, y1)):
        send(QEvent.MouseMove, x, y, V.Qt.LeftButton)
    send(QEvent.MouseButtonRelease, x0, y1 + 10, V.Qt.NoButton)
    assert clicks == []
    assert len(lassos) == 1 and lassos[0].shape == (5, 2)
    # The outline is in UMAP units and encloses both points; the view did not pan.
    assert points_in_polygon(np.array([[0.0, 0.0], [10.0, 10.0]]), lassos[0]).all()
    assert canvas.data_to_view([[0.0, 0.0]])[0] == pytest.approx(before)


def _fake_label_display(ctrl, monkeypatch):
    built = []
    monkeypatch.setattr(ctrl, "_ensure_label_key_for_segmentation", lambda source, **_: f"{source}_labels")

    def display(label_key, **_):
        built.append(label_key)
        return [np.zeros((4, 4), dtype=np.uint32)], np.eye(3)

    monkeypatch.setattr(ctrl, "_build_cellpose_label_display", display)
    return built


def _drag(ctrl, points_xy):
    """Replay a napari press-move-release drag through the viewer callback."""
    x, y = points_xy[0]
    event = types.SimpleNamespace(
        type="mouse_press", position=(y, x), pos=(x * 10, y * 10), button=1,
        view_direction=None, dims_displayed=(0, 1),
    )
    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    for x, y in points_xy[1:]:
        event.type, event.position, event.pos = "mouse_move", (y, x), (x * 10, y * 10)
        next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)


def _rotated_drag(ctrl, angle):
    origin = np.array([20., 30.])
    adapter = types.SimpleNamespace(
        _rotation_angle=angle,
        _rotation_origin=lambda: origin,
        _world_rotation_matrix=V.NapariLeftPanelAdapter._world_rotation_matrix,
    )
    adapter.unrotate_world_xy = lambda xy: V.NapariLeftPanelAdapter.unrotate_world_xy(adapter, xy)
    ctrl._left_panel_adapter = adapter
    source = np.array([(-2, -2), (12, -2), (12, 12), (-2, 12)])
    transform = V.NapariLeftPanelAdapter._world_rotation_matrix(2, angle, origin)
    displayed = (transform @ np.column_stack([source[:, ::-1], np.ones(4)]).T).T[:, :2][:, ::-1]
    _drag(ctrl, displayed)
    return source


@pytest.mark.parametrize("angle", [0.0, 37.25, 90.0, 180.0])
def test_spatial_lasso_outlines_cells_and_summarises_them_in_the_umap(qapp, monkeypatch, angle):
    ctrl, widget = _controller(qapp)
    built = _fake_label_display(ctrl, monkeypatch)
    window = _open_window(ctrl, widget)

    widget._lasso_button.click()
    assert ctrl._umap_lasso_armed and window.canvas.lasso_mode
    assert ctrl.viewer.camera.mouse_pan is False  # dragging draws instead of panning

    source = _rotated_drag(ctrl, angle)
    assert np.allclose(ctrl._umap_state().roi.polygon[:4], source)
    state = ctrl._umap_state()
    assert state.roi.space == "spatial"
    assert state.roi.cell_ids.tolist() == [202] and state.roi.rows.tolist() == [1]
    # Drawing disarms, restoring panning; single-cell inspection is untouched.
    assert not ctrl._umap_lasso_armed and not widget._lasso_button.isChecked()
    assert ctrl.viewer.camera.mouse_pan is True
    assert ctrl._selected_cells.get("MERSCOPE", []) == []

    names = [layer.name for layer in ctrl.viewer.layers]
    assert V.ROI_OUTLINE_LAYER in names and V.ROI_CELLS_LAYER in names
    cells = ctrl._get_layer_by_name(V.ROI_CELLS_LAYER)
    assert cells.contour == V.ROI_CELL_CONTOUR_PX
    outlined = {k for k, v in cells.colormap.color_dict.items() if k and v[3] > 0}
    assert outlined == {202}
    assert built == ["proseg_labels"]

    # The UMAP emphasises the region and the side panel summarises it.
    assert tuple(window.colouring.colors[0]) == _u8(UMAP_MUTED_RGBA)  # 303 outside
    assert window.roi_panel.isVisible()
    assert window.roi_panel.rows_text() == ["Astrocytes  1 · 100.0%"]

    # Switching source re-selects the region's cells from the new masks.
    ctrl.set_umap_source("cellpose")
    assert state.roi.cell_ids.tolist() == [7]
    assert built == ["proseg_labels", "cellpose_labels"]

    widget._clear_roi_button.click()
    assert state.roi is None
    assert V.ROI_CELLS_LAYER not in [layer.name for layer in ctrl.viewer.layers]
    assert not window.roi_panel.isVisible()


def test_umap_lasso_selects_points_and_is_cleared_when_the_source_changes(qapp, monkeypatch):
    ctrl, widget = _controller(qapp)
    _fake_label_display(ctrl, monkeypatch)
    window = _open_window(ctrl, widget)
    widget._lasso_button.click()

    # Row 0 of the ProSeg UMAP sits at (5, 5): cell 303, a neuron.
    window.panel().lasso_completed.emit(np.array([[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]]))
    state = ctrl._umap_state()
    assert state.roi.space == "umap" and state.roi.cell_ids.tolist() == [303]
    assert window.canvas.roi_polygon is not None
    assert window.roi_panel.rows_text() == ["Neurons  1 · 100.0%"]
    outlined = ctrl._get_layer_by_name(V.ROI_CELLS_LAYER).colormap.color_dict
    assert {k for k, v in outlined.items() if k and v[3] > 0} == {303}
    assert V.ROI_OUTLINE_LAYER not in [layer.name for layer in ctrl.viewer.layers]

    ctrl.set_umap_source("cellpose")
    assert state.roi is None
    assert window.canvas.roi_polygon is None
    assert "Cleared the region" in widget._status_label.text()


def test_a_tiny_spatial_drag_keeps_drawing_armed(qapp, monkeypatch):
    ctrl, widget = _controller(qapp)
    _fake_label_display(ctrl, monkeypatch)
    _open_window(ctrl, widget)
    widget._lasso_button.click()
    _drag(ctrl, [(5, 5)])  # a click, not an outline
    assert ctrl._umap_state().roi is None
    assert ctrl._umap_lasso_armed


def test_lazy_labels_are_recoloured_once_their_first_slice_arrives(monkeypatch):
    class Emitter:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def disconnect(self, callback):
            self.callbacks.remove(callback)

        def __call__(self):
            for callback in list(self.callbacks):
                callback(None)

    layer = types.SimpleNamespace(events=types.SimpleNamespace(set_data=Emitter()))
    deferred, recolours = [], []
    monkeypatch.setattr(V.QTimer, "singleShot", lambda _delay, callback: deferred.append(callback))
    V.ComparisonViewerController._reapply_labels_colormap_after_first_slice(
        layer, lambda: recolours.append(True)
    )
    assert deferred == []  # nothing until napari has applied a slice
    layer.events.set_data()
    layer.events.set_data()  # later slices keep the colours; no more work
    assert len(deferred) == 1 and layer.events.set_data.callbacks == []
    deferred.pop()()
    assert recolours == [True]


# ---------------------------------------------------------------------------
# Paired MERSCOPE / Xenium
# ---------------------------------------------------------------------------


class _PairedLayerList(list):
    """List-backed layer list with napari's batched_update/move API."""

    def batched_update(self):
        from contextlib import nullcontext

        return nullcontext()

    def move(self, source, destination):
        layer = self.pop(source)
        if destination > source:
            destination -= 1
        self.insert(destination, layer)


class _PairedViewer:
    def __init__(self):
        self.layers = _PairedLayerList()
        self.grid = types.SimpleNamespace(enabled=False, shape=(-1, -1), stride=1, spacing=0.0)
        self.mouse_drag_callbacks = []
        self.window = types.SimpleNamespace(_qt_viewer=None)
        self.camera = _FakeCamera()


def _paired_controller(qapp, monkeypatch):
    from napari_compare_xenium_merscope.paired_data import (
        DatasetElementProfile,
        PairedDataContract,
        SegmentationProfile,
    )
    from napari_compare_xenium_merscope.paired_views import (
        LayerSlotKey,
        PairedLayerRegistry,
        ViewMode,
    )

    args = types.SimpleNamespace(
        gene_spot_size=2.0,
        gene_max_render_points=40_000_000,
        gene_hide_background=False,
        gene_show_controls=False,
        random_state=42,
        shape_opacity=0.9,
        shape_edge_width=0.5,
    )
    viewer = _PairedViewer()
    ctrl = V.ComparisonViewerController(viewer, {}, args)
    ctrl.datasets = {
        "MERSCOPE": types.SimpleNamespace(zarr_path=None),
        "XENIUM": types.SimpleNamespace(zarr_path=None),
    }
    ctrl.active_dataset = "PAIRED"
    # Both stores cover the same aligned places with their own cell ids.
    shapes = {
        "MERSCOPE": gpd.GeoDataFrame(
            {"geometry": [_square(0, 0), _square(20, 20)]}, index=np.array([202, 303], dtype=np.uint64)
        ),
        "XENIUM": gpd.GeoDataFrame(
            {"geometry": [_square(1, 1, 8), _square(21, 21, 8)]}, index=np.array([11, 12], dtype=np.uint64)
        ),
    }
    segmentations = {
        platform: SegmentationProfile(
            logical_key="proseg",
            shape_key=f"{platform.lower()}_proseg",
            label_key=f"{platform.lower()}_proseg_labels",
            points_key="transcripts",
            table_key=None,
        )
        for platform in ("MERSCOPE", "XENIUM")
    }
    profile = dict(
        points_key="transcripts", image_keys=(), image_pyramid_keys={}, segmentations=(),
        coordinate_system="global", pixel_to_world_affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    contract = PairedDataContract(
        pair_id="pair", coordinate_system="global", fixed_width=40, fixed_height=40,
        transform_fingerprint="t", native_fingerprint="n",
        merscope=DatasetElementProfile(platform="MERSCOPE", **profile),
        xenium=DatasetElementProfile(platform="XENIUM", **profile),
        mappings={},
    )
    ctrl._paired_session = types.SimpleNamespace(
        contract=contract,
        sdata={
            platform: types.SimpleNamespace(
                shapes={segmentations[platform].shape_key: shapes[platform]}, labels={}
            )
            for platform in ("MERSCOPE", "XENIUM")
        },
        configs=ctrl.datasets,
        view_mode=ViewMode.SIDE_BY_SIDE,
    )
    registry = PairedLayerRegistry(viewer, ctrl._paired_placeholder_layer, strict_layers=True)
    registry.ensure_slot(LayerSlotKey("reference", "layout-anchor"), side_visible=False)
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    ctrl._paired_registry = registry
    monkeypatch.setattr(ctrl, "_paired_pick_segmentation", lambda platform: segmentations[platform])
    monkeypatch.setattr(
        ctrl,
        "_paired_segmentation_for_cell_type",
        lambda platform, source: segmentations[platform] if source == "proseg" else None,
    )
    monkeypatch.setattr(
        ctrl,
        "_paired_label_levels",
        lambda platform, segmentation, outline: (
            [np.zeros((40, 40), dtype=np.uint32)],
            segmentation.label_key,
        ),
    )
    monkeypatch.setattr(
        ctrl,
        "_umap_view_sources",
        lambda state, view: {source: source == "proseg" for source in ("proseg", "proseg_mask", "cellpose", "original")},
    )
    labels = {
        "MERSCOPE": ([202, 303], ["Astrocytes", "Neurons"], ["Astrocytes:0", "Neurons:1"]),
        "XENIUM": ([11, 12], ["Neurons", "Oligodendrocytes"], ["Neurons:0", "Oligodendrocytes:0"]),
    }
    for platform, (ids, broad, fine) in labels.items():
        ctrl._cache_cell_type_data(
            ctrl._cell_type_state(platform),
            "proseg",
            CellTypeAssignments(
                segmentation="proseg",
                table_key="table_MOSAIK_proseg_clustering_squidpy",
                instance_key="instance_id",
                cell_ids=np.asarray(ids, dtype=np.int64),
                broad=np.asarray(broad, dtype=object),
                fine=np.asarray(fine, dtype=object),
            ),
        )
    widget = UmapControlWidget(
        set_window_visible_callback=ctrl.set_umap_window_visible,
        set_source_callback=ctrl.set_umap_source,
        set_colour_callback=ctrl.set_umap_colour,
        set_point_size_callback=ctrl.set_umap_point_size,
        set_lasso_active_callback=ctrl.set_umap_lasso_active,
        clear_roi_callback=ctrl.clear_umap_roi,
    )
    ctrl.set_umap_widget(widget)
    state = ctrl._umap_state()
    state.views["MERSCOPE"].embeddings["proseg"] = CellUmapEmbedding(
        segmentation="proseg", table_key="table_MOSAIK_proseg_clustering_squidpy", instance_key="instance_id",
        cell_ids=np.asarray([303, 202], dtype=np.int64),
        coords=np.asarray([[5.0, 5.0], [-5.0, -5.0]], dtype=np.float32),
    )
    state.views["XENIUM"].embeddings["proseg"] = CellUmapEmbedding(
        segmentation="proseg", table_key="table_MOSAIK_proseg_clustering_squidpy", instance_key="instance_id",
        cell_ids=np.asarray([11, 12], dtype=np.int64),
        coords=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )
    return ctrl, widget, registry


def _roi_identity(platform, part):
    from napari_compare_xenium_merscope.paired_views import LayerIdentity

    return LayerIdentity(platform, V.ROI_PAIRED_ROLE, V.ROI_PAIRED_ELEMENT, part)


def _outlined_ids(layer):
    return {k for k, v in layer.colormap.color_dict.items() if k and v[3] > 0}


def test_paired_umap_window_stacks_merscope_above_xenium_in_one_palette(qapp, monkeypatch):
    ctrl, widget, _registry = _paired_controller(qapp, monkeypatch)
    window = _open_window(ctrl, widget)
    assert window.panel_keys() == ("MERSCOPE", "XENIUM")
    assert [window.panel(k).heading for k in window.panel_keys()] == ["MERSCOPE", "Xenium"]
    merscope, xenium = window.panel("MERSCOPE"), window.panel("XENIUM")
    # Row 0 of MERSCOPE is 303 (Neurons); row 0 of Xenium is 11 (Neurons):
    # one palette spans both stores, like the paired cell-type fill.
    assert tuple(merscope.colouring.colors[0]) == tuple(xenium.colouring.colors[0])
    assert merscope.describe_row(1) == "Cell 202 · Astrocytes"
    assert xenium.describe_row(1) == "Cell 12 · Oligodendrocytes"
    # Per-cell statistics are unavailable for pairs.
    stat_item = widget._colour_combo.model().item(widget._colour_combo.findData("statistic"))
    assert not stat_item.isEnabled()


@pytest.mark.parametrize("angle", [0.0, 37.25, 90.0])
def test_paired_spatial_lasso_mirrors_the_outline_and_selects_each_store(qapp, monkeypatch, angle):
    ctrl, widget, registry = _paired_controller(qapp, monkeypatch)
    window = _open_window(ctrl, widget)
    widget._lasso_button.click()
    assert window.panel("XENIUM").canvas.lasso_mode

    _rotated_drag(ctrl, angle)
    state = ctrl._umap_state()
    assert state.views["MERSCOPE"].roi.cell_ids.tolist() == [202]
    assert state.views["XENIUM"].roi.cell_ids.tolist() == [11]
    # The identical outline is drawn in both panes (same aligned world coordinates).
    outlines = [registry.layer_for(_roi_identity(p, "outline")) for p in ("MERSCOPE", "XENIUM")]
    assert all(layer is not None for layer in outlines)
    assert np.allclose(outlines[0].data[0], outlines[1].data[0])
    # Each store outlines its own cells on its own masks.
    cells = {p: registry.layer_for(_roi_identity(p, "cells")) for p in ("MERSCOPE", "XENIUM")}
    assert _outlined_ids(cells["MERSCOPE"]) == {202} and _outlined_ids(cells["XENIUM"]) == {11}
    assert cells["XENIUM"].contour == V.ROI_CELL_CONTOUR_PX
    registry.assert_invariants()
    # Each UMAP emphasises its own region and has its own pie.
    assert window.panel("MERSCOPE").roi_panel.rows_text() == ["Astrocytes  1 · 100.0%"]
    assert window.panel("XENIUM").roi_panel.rows_text() == ["Neurons  1 · 100.0%"]
    assert tuple(window.panel("XENIUM").colouring.colors[1]) == _u8(UMAP_MUTED_RGBA)

    widget._clear_roi_button.click()
    assert all(view.roi is None for view in state.views.values())
    assert all(key.role != V.ROI_PAIRED_ROLE for key in registry.slot_keys)
    registry.assert_invariants()


def test_paired_arming_prepares_mirrored_outline_layers_and_cancel_removes_them(qapp, monkeypatch):
    ctrl, widget, registry = _paired_controller(qapp, monkeypatch)
    _open_window(ctrl, widget)
    widget._lasso_button.click()  # arm: the grid re-layout happens now, not mid-drag
    prepared = [registry.layer_for(_roi_identity(p, "outline")) for p in ("MERSCOPE", "XENIUM")]
    assert all(layer is not None and len(layer.data) == 0 for layer in prepared)
    _drag(ctrl, [(5, 5)])  # too small: still armed, layers kept
    assert ctrl._umap_lasso_armed
    assert registry.layer_for(_roi_identity("MERSCOPE", "outline")) is prepared[0]
    widget._lasso_button.click()  # cancel
    assert registry.layer_for(_roi_identity("MERSCOPE", "outline")) is None
    registry.assert_invariants()


def test_registry_deferred_order_applies_one_layout_for_several_registrations():
    from napari_compare_xenium_merscope.paired_views import LayerIdentity, PairedLayerRegistry, ViewMode

    viewer = _PairedViewer()
    registry = PairedLayerRegistry(viewer, lambda identity: _FakeLayer("placeholder", []), strict_layers=True)
    passes = []
    original = registry._apply_order_only

    def counted():
        passes.append(registry._order_hold)
        original()

    registry._apply_order_only = counted
    with registry.deferred_order():
        for platform in ("MERSCOPE", "XENIUM"):
            registry.register_layer(LayerIdentity(platform, "annotation", "x"), _FakeLayer(platform, []))
    assert passes[-1] == 0 and all(hold > 0 for hold in passes[:-1])
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    registry.assert_invariants()


def test_paired_umap_lasso_only_selects_that_modality(qapp, monkeypatch):
    ctrl, widget, registry = _paired_controller(qapp, monkeypatch)
    window = _open_window(ctrl, widget)
    widget._lasso_button.click()
    # Circle Xenium row 0 (cell 11 at UMAP (0, 0)).
    window.panel("XENIUM").lasso_completed.emit(
        np.array([[-1.0, -1.0], [0.5, -1.0], [0.5, 0.5], [-1.0, 0.5]])
    )
    state = ctrl._umap_state()
    assert state.views["XENIUM"].roi.cell_ids.tolist() == [11]
    assert state.views["MERSCOPE"].roi is None
    assert _outlined_ids(registry.layer_for(_roi_identity("XENIUM", "cells"))) == {11}
    assert registry.layer_for(_roi_identity("MERSCOPE", "cells")) is None  # placeholder only
    assert registry.layer_for(_roi_identity("XENIUM", "outline")) is None
    assert window.panel("XENIUM").canvas.roi_polygon is not None
    assert not window.panel("MERSCOPE").roi_panel.isVisible()
    assert "the XENIUM UMAP" in window.panel("XENIUM").roi_panel._subtitle.text()
    registry.assert_invariants()


def test_paired_umap_click_selects_only_that_store_and_spatial_picks_ring_back(qapp, monkeypatch):
    ctrl, widget, _registry = _paired_controller(qapp, monkeypatch)
    window = _open_window(ctrl, widget)
    window.panel("MERSCOPE").point_clicked.emit(0)  # cell 303
    assert [int(c) for c, _g in ctrl._paired_selected_cells["MERSCOPE"]] == [303]
    assert ctrl._paired_selected_cells["XENIUM"] == []
    assert window.panel("MERSCOPE").canvas.highlighted_rows() == [0]
    assert window.panel("XENIUM").canvas.highlighted_rows() == []

    # A spatial click selects both stores at one aligned point; each UMAP rings its cell.
    gdf = ctrl._paired_session.sdata["XENIUM"].shapes["xenium_proseg"]
    ctrl._paired_selected_cells["XENIUM"].append((gdf.index[1], gdf.geometry.iloc[1]))
    ctrl._update_paired_cell_info_overlay()
    assert window.panel("XENIUM").canvas.highlighted_rows() == [1]

    ctrl._clear_paired_cell_selections()
    assert window.panel("MERSCOPE").canvas.highlighted_rows() == []


def test_paired_cell_type_level_click_syncs_both_umaps(qapp, monkeypatch):
    ctrl, widget, _registry = _paired_controller(qapp, monkeypatch)
    monkeypatch.setattr(ctrl, "_apply_paired_cell_type_layers", lambda states: len(states))
    window = _open_window(ctrl, widget)
    ctrl.set_cell_type_kind("PAIRED", "fine")
    assert ctrl._umap_state().colour_by == "fine"
    assert window.panel("MERSCOPE").colouring.title == "Fine cell type"
    assert window.panel("XENIUM").colouring.title == "Fine cell type"
    # Unticking a (shared) type mutes it in the store that has it.
    ctrl.set_cell_type_visible("PAIRED", "Neurons:0", False)
    assert tuple(window.panel("XENIUM").colouring.colors[0]) == _u8(UMAP_MUTED_RGBA)


def test_control_panel_has_a_umap_tab(qapp):
    from test_startup import _panel

    widget = UmapControlWidget(lambda *_: None, lambda *_: None, lambda *_: None, lambda *_: None)
    panel = _panel([], umap_widget=widget)
    titles = [panel._tab_group.button(i).text() for i in range(len(panel._tab_group.buttons()))]
    assert titles[-1] == "UMAP"
    # An empty launch still opens on the Dataset loader.
    assert panel._tab_group.checkedButton().text() == "Dataset loader"


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_hybrid_allen_hierarchy_joins_masks_and_umap_without_loading_expression(tmp_path, monkeypatch, zarr_format):
    import anndata as ad
    from napari_compare_xenium_merscope.utils import (
        cell_type_levels, load_cell_type_assignments, build_assignment_color_schemes,
        build_cell_type_color_dict,
    )
    from napari_compare_xenium_merscope.cell_umap import cell_umap_available

    ids = [3715213700018100064, 3715213700018100066, 3715213700018100069]
    obs = pd.DataFrame({
        "instance_id": np.array(ids, dtype=np.uint64),
        "ct_final_name": pd.Categorical(["Astro", "Excitatory", "Unassigned"]),
        "ct_broad_name": pd.Categorical(["Astro", "Neurons", None]),
        "ct_supercluster_name": pd.Categorical([None, "L2/3 IT", None]),
        "ct_supercluster_status": pd.Categorical(["not_applicable", "confident", "low_counts"]),
        "ct_cluster_name": pd.Categorical([None, None, None]),
    }, index=["a", "b", "c"])
    store = tmp_path / "store.zarr"
    root = zarr.open_group(str(store), mode="w", zarr_format=zarr_format)
    root.create_group("tables")
    with ad.settings.override(zarr_write_format=zarr_format):
        store = _write_clustering_table(tmp_path, "table_MOSAIK_proseg_hybrid_clustering_squidpy",
                                        obs, np.array([[2, 3], [4, 5], [6, 7]], dtype=float))
    monkeypatch.setattr(ad, "read_zarr", lambda *a, **k: pytest.fail("Do not load the expression matrix"))
    assert cell_type_levels(store, "proseg_hybrid") == {
        "ct_final": "Allen · final assignment", "ct_broad": "Allen · broad type",
        "ct_supercluster": "Allen · supercluster",
    }
    assert cell_type_levels(store, "proseg") == {}
    assignments = load_cell_type_assignments(store, "proseg_hybrid")
    assert assignments.cell_ids.tolist() == ids
    assert assignments.labels_for("ct_supercluster").tolist() == [None, "L2/3 IT", None]
    assert assignments.labels_for("ct_cluster").tolist() == [None] * 3
    assert cell_umap_available(store, "proseg_hybrid")
    embedding = load_cell_umap(store, "proseg_hybrid")
    assert embedding.cell_ids.tolist() == ids
    labels = labels_for_cell_ids(assignments.cell_ids, assignments.labels_for("ct_supercluster"),
                                 embedding.cell_ids[::-1])
    assert labels.tolist() == [None, "L2/3 IT", None]
    scheme = build_assignment_color_schemes([assignments])["ct_supercluster"]
    mapping = build_cell_type_color_dict(assignments.cell_ids, assignments.labels_for("ct_supercluster"), scheme)
    assert ids[0] not in mapping and ids[2] not in mapping
    assert mapping[ids[1]] == scheme.colors["L2/3 IT"]
    assert V.ComparisonViewerController._cell_type_counts(assignments, "ct_supercluster") == {"L2/3 IT": 1}


def test_missing_hierarchy_and_embedding_are_not_offered(tmp_path, qapp):
    from napari_compare_xenium_merscope.utils import cell_type_levels
    from napari_compare_xenium_merscope.cell_umap import cell_umap_available

    obs = pd.DataFrame({"instance_id": [1, 2], "ct_broad_name": pd.Categorical([None, None])},
                       index=["a", "b"])
    store = _write_clustering_table(tmp_path, "table_MOSAIK_proseg_hybrid_clustering_squidpy", obs, None)
    assert cell_type_levels(store, "proseg_hybrid") == {}
    assert not cell_umap_available(store, "proseg_hybrid")
    ctrl, widget = _controller(qapp)
    ctrl.datasets = {"MERSCOPE": types.SimpleNamespace(zarr_path=store)}
    ctrl._dataset_sessions.clear()
    ctrl._publish_umap_options()
    assert widget._source_combo.count() == 0
    assert not widget._window_button.isEnabled()
    assert widget._colour_combo.findData("ct_broad") == -1


def _add_allen_levels(ctrl, platform, source="proseg", second_level=True):
    state = ctrl._cell_type_state(platform)
    assignments = state.assignments[source]
    assignments.levels = {"broad": assignments.broad, "fine": assignments.fine,
                          "ct_final": np.array(["Astro", "L2/3 IT"], dtype=object)}
    if second_level:
        assignments.levels["ct_supercluster"] = np.array([None, "L2/3 IT"], dtype=object)
    ctrl._cache_cell_type_data(state, source, assignments)
    return state


def test_allen_level_syncs_spatial_umap_and_roi_and_resets_on_source_change(qapp, monkeypatch):
    ctrl, widget = _controller(qapp)
    cell_state = _add_allen_levels(ctrl, "MERSCOPE")
    monkeypatch.setattr(ctrl, "_start_cell_type_overlay_build", lambda state: None)
    window = _open_window(ctrl, widget)
    ctrl.set_cell_type_kind("MERSCOPE", "ct_supercluster")
    assert widget.current_colour()[0] == "ct_supercluster"
    assert window.colouring.title == "Allen · supercluster"
    assert window.describe_row(0) == "Cell 303 · L2/3 IT"
    assert window.describe_row(1) == "Cell 202 · Unannotated"
    scheme = cell_state.schemes["proseg"]["ct_supercluster"]
    assert tuple(window.colouring.colors[0]) == _u8(scheme.colors["L2/3 IT"])
    ctrl.set_cell_type_visible("MERSCOPE", "L2/3 IT", False)
    assert tuple(window.colouring.colors[0]) == _u8(UMAP_MUTED_RGBA)
    ctrl.set_cell_type_visible("MERSCOPE", "L2/3 IT", True)
    ctrl._on_umap_lasso_completed("", np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]]))
    summary = ctrl._umap_roi_summary(ctrl._umap_state(), ctrl._umap_state().primary,
                                     ctrl._umap_state().embeddings["proseg"], window.colouring)
    assert summary.level == "Allen · supercluster"
    ctrl.set_cell_type_segmentation("MERSCOPE", "cellpose")
    assert widget._colour_combo.findData("ct_supercluster") == -1
    assert ctrl._umap_state().colour_by == "broad"
    assert cell_state.kind == "broad"
    window.close()


def test_paired_allen_colours_match_spatial_and_both_umaps(qapp, monkeypatch):
    ctrl, widget, registry = _paired_controller(qapp, monkeypatch)
    states = [_add_allen_levels(ctrl, platform) for platform in ("MERSCOPE", "XENIUM")]
    states[1].assignments["proseg"].levels["ct_final"] = np.array(["L2/3 IT", "Oligo"], dtype=object)
    window = _open_window(ctrl, widget)
    ctrl.set_cell_type_kind("MERSCOPE", "ct_final")
    for state in states:
        assert state.kind == "ct_final"
        assert state.schemes["proseg"]["ct_final"].colors == states[0].schemes["proseg"]["ct_final"].colors
    palette = states[0].schemes["proseg"]["ct_final"].colors
    for view in ctrl._umap_state().views.values():
        scheme = ctrl._umap_scheme(ctrl._umap_state(), view, "ct_final")
        assert scheme.colors == palette
    assert tuple(window.panel("MERSCOPE").colouring.colors[0]) == tuple(window.panel("XENIUM").colouring.colors[0])
    window.close()


def test_cell_type_widget_hides_unavailable_hierarchy_levels(qapp):
    callback = lambda *args: None
    widget = V.CellTypeWidget(
        close_callback=callback, set_segmentation_callback=callback, set_kind_callback=callback,
        set_type_visible_callback=callback, set_types_visible_callback=callback,
        set_all_types_callback=callback, set_opacity_callback=callback,
    )
    widget.set_levels_available({"ct_final": "Allen · final assignment"})
    assert widget._level_combo.findData("ct_final") == 0
    assert widget._level_combo.findData("ct_supercluster") == -1
    assert widget._level_combo.currentIndex() == -1
    widget.set_levels_available({"broad": "Broad cell type", "fine": "Fine cell type"})
    assert widget._level_combo.findData("ct_final") == -1
    assert widget._level_combo.findData("broad") == 0


def test_cell_type_dropdowns_dispatch_user_choices_only(qapp):
    changes = []
    callback = lambda *args: None
    widget = V.CellTypeWidget(
        close_callback=callback,
        set_segmentation_callback=lambda *args: changes.append(("source", *args)),
        set_kind_callback=lambda *args: changes.append(("level", *args)),
        set_type_visible_callback=callback, set_types_visible_callback=callback,
        set_all_types_callback=callback, set_opacity_callback=callback,
    )
    widget.set_dataset("MERSCOPE")
    sources = [source.key for source in V.CELL_TYPE_SOURCES[:2]]
    widget.set_segmentation_available(dict.fromkeys(sources, True))
    widget.set_levels_available({"broad": "Broad cell type", "fine": "Fine cell type"})
    assert changes == []
    source_index = 1 - widget._seg_combo.currentIndex()
    widget._seg_combo.setCurrentIndex(source_index)
    widget._seg_combo.activated.emit(source_index)
    widget._level_combo.setCurrentIndex(1)
    widget._level_combo.activated.emit(1)
    assert changes == [("source", "MERSCOPE", sources[source_index]), ("level", "MERSCOPE", "fine")]
    widget.set_levels_available({"broad": "Broad cell type"})
    assert widget._level_combo.currentIndex() == -1
    widget.set_segmentation_available({sources[0]: True})
    assert widget._seg_combo.count() == 1
    assert widget._seg_combo.currentData() == sources[0]
    assert len(changes) == 2


def test_paired_hierarchy_missing_on_one_platform_stays_unannotated(qapp, monkeypatch):
    ctrl, widget, registry = _paired_controller(qapp, monkeypatch)
    _add_allen_levels(ctrl, "MERSCOPE")
    window = _open_window(ctrl, widget)
    ctrl.set_umap_colour("ct_supercluster")
    assert window.panel("MERSCOPE").describe_row(0) == "Cell 303 · L2/3 IT"
    assert window.panel("XENIUM").describe_row(0) == "Cell 11 · Unannotated"
    assert window.panel("XENIUM").describe_row(1) == "Cell 12 · Unannotated"
    recorded = []
    ctrl._cell_type_widget = types.SimpleNamespace(set_levels_available=recorded.append)
    ctrl._publish_cell_type_levels()
    assert "ct_supercluster" not in recorded[-1]
    window.close()


def test_unreadable_annotation_store_fails_closed(tmp_path):
    from napari_compare_xenium_merscope.utils import cell_type_levels, load_cell_type_assignments
    root = zarr.open_group(str(tmp_path / "broken.zarr"), mode="w")
    root.create_group("tables").create_group("table_MOSAIK_proseg_hybrid_clustering_squidpy")
    assert cell_type_levels(tmp_path / "broken.zarr", "proseg_hybrid") == {}
    assert load_cell_type_assignments(tmp_path / "broken.zarr", "proseg_hybrid") is None


@pytest.mark.parametrize("suffix", ["mapfirst", "m8", "custom_run"])
def test_named_map_first_run_supersedes_legacy_for_labels_and_umap(tmp_path, suffix):
    from napari_compare_xenium_merscope.utils import (
        clustering_table_key_for_segmentation, cell_type_levels, load_cell_type_assignments,
    )
    base = "table_MOSAIK_proseg_hybrid_clustering_squidpy"
    legacy = pd.DataFrame({"instance_id": [1], "broad_class": ["Neurons"]}, index=["legacy"])
    store = _write_clustering_table(tmp_path, base, legacy, np.array([[0., 0.]]))
    atlas = pd.DataFrame({"instance_id": [7, 8], "ct_seaad_subclass_name": ["L2/3 IT", "Pvalb"]},
                         index=["a", "b"])
    key = base + "_" + suffix
    _write_clustering_table(tmp_path, key, atlas, np.array([[1., 2.], [3., 4.]]))
    assert clustering_table_key_for_segmentation(store, "proseg_hybrid") == key
    assert "ct_seaad_subclass" in cell_type_levels(store, "proseg_hybrid")
    assignments = load_cell_type_assignments(store, "proseg_hybrid")
    embedding = load_cell_umap(store, "proseg_hybrid")
    assert assignments.table_key == embedding.table_key == key
    assert assignments.labels_for("ct_seaad_subclass").tolist() == ["L2/3 IT", "Pvalb"]
    assert assignments.cell_ids.tolist() == embedding.cell_ids.tolist() == [7, 8]
    assert embedding.coords.tolist() == [[1., 2.], [3., 4.]]
    assert clustering_table_key_for_segmentation(store, "proseg") is None
    assert clustering_table_key_for_segmentation(store, "proseg_mask") is None


def test_map_first_run_preference_is_deterministic_and_ignores_non_annotation_variants(tmp_path):
    from napari_compare_xenium_merscope.utils import clustering_table_key_for_segmentation
    base = "table_MOSAIK_proseg_hybrid_clustering_squidpy"
    atlas = pd.DataFrame({"instance_id": [7], "ct_final_name": ["Pvalb"]}, index=["a"])
    legacy = pd.DataFrame({"instance_id": [1], "broad_class": ["Neurons"]}, index=["a"])
    # The exact coexistence seen in P7513, deliberately created out of order.
    store = _write_clustering_table(tmp_path, base + "_m8_state", atlas, np.array([[1., 2.]]))
    _write_clustering_table(tmp_path, base + "_m8", atlas, np.array([[3., 4.]]))
    _write_clustering_table(tmp_path, base, legacy, np.array([[0., 0.]]))
    _write_clustering_table(tmp_path, base + "_mapfirst", legacy, np.array([[0., 0.]]))
    assert clustering_table_key_for_segmentation(store, "proseg_hybrid") == base + "_m8"
    _write_clustering_table(tmp_path, base + "_mapfirst", atlas, np.array([[5., 6.]]))
    assert clustering_table_key_for_segmentation(store, "proseg_hybrid") == base + "_mapfirst"
    # After MerXen's default switches, an annotated unsuffixed run wins.
    _write_clustering_table(tmp_path, base, atlas, np.array([[7., 8.]]))
    assert clustering_table_key_for_segmentation(store, "proseg_hybrid") == base
