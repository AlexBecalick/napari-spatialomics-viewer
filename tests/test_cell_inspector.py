"""Tests for the click-a-cell-mask inspector.

Pure geometry / indexing / intensity helpers are exercised directly; the Qt
overlay + pie chart render under an offscreen QApplication; and the controller
wiring (click precedence, boundary + link layers, overlay population) runs
against a lightweight fake napari viewer with no OpenGL canvas.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import (
    GENE_MARKER_SYMBOLS,
    build_cell_transcript_index,
    build_gene_point_groups,
    darken_rgba,
    mean_intensity_in_polygon,
    normalize_cell_key,
    pick_cell_at_point,
    ranked_gene_counts,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def synchronous_workers(monkeypatch):
    """Make controller integration assertions deterministic in this module."""
    monkeypatch.setattr(V, "thread_worker", None)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _demo_gdf():
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),   # big outer
        Polygon([(2, 2), (6, 2), (6, 6), (2, 6)]),       # smaller, nested in outer
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
    ]
    return gpd.GeoDataFrame({"geometry": polys}, index=[101, 202, 303])


def test_pick_cell_prefers_smallest_containing_polygon():
    gdf = _demo_gdf()
    assert pick_cell_at_point(gdf, 4, 4)[0] == 202     # inside both -> smallest wins
    assert pick_cell_at_point(gdf, 8, 8)[0] == 101     # only in the outer
    assert pick_cell_at_point(gdf, 25, 25)[0] == 303
    assert pick_cell_at_point(gdf, 100, 100) is None   # empty space


def test_normalize_cell_key_bridges_int_and_string():
    assert normalize_cell_key(5) == normalize_cell_key(5.0) == normalize_cell_key("5") == "5"
    assert normalize_cell_key("cellA") == "cellA"
    assert normalize_cell_key(None) == ""
    assert normalize_cell_key(float("nan")) == ""


def test_build_cell_transcript_index_groups_assigned_only():
    pts = pd.DataFrame(
        {
            "x": [4.0, 4.5, 8.0, 25.0, 4.0],
            "y": [4.0, 4.2, 8.0, 25.0, 4.0],
            "gene": ["A", "B", "A", "C", "A"],
            "assignment": [202, 202, 101, 303, 0],  # last is background
        }
    )
    index = build_cell_transcript_index(pts, "x", "y", "gene", "assignment")

    coords, genes = index.transcripts_for(202)
    assert len(genes) == 2
    assert sorted(genes.tolist()) == ["A", "B"]
    # coords are napari (y, x); both cell-202 transcripts live near (y~4, x~4)
    assert coords.shape == (2, 2)
    assert np.allclose(np.sort(coords[:, 1]), [4.0, 4.5], atol=1e-3)  # x values

    assert len(index.transcripts_for(101)[1]) == 1
    assert len(index.transcripts_for(303)[1]) == 1
    assert len(index.transcripts_for(0)[1]) == 0        # background excluded
    assert len(index.transcripts_for(999)[1]) == 0      # unknown cell


def test_build_cell_transcript_index_without_assignment_is_empty():
    pts = pd.DataFrame({"x": [1.0], "y": [1.0], "gene": ["A"]})
    index = build_cell_transcript_index(pts, "x", "y", "gene", None)
    assert index.slices == {}
    assert len(index.transcripts_for(1)[1]) == 0


def test_ranked_gene_counts_orders_by_count_then_name():
    genes = np.array(["A", "A", "B", "C", "A", "B"], dtype=object)
    assert ranked_gene_counts(genes) == [("A", 3), ("B", 2), ("C", 1)]
    assert ranked_gene_counts(np.empty((0,), dtype=object)) == []


def test_darken_rgba_scales_rgb_only():
    assert darken_rgba((1.0, 0.5, 0.2, 0.8), 0.5) == (0.5, 0.25, 0.1, 0.8)
    r = darken_rgba((1.0, 1.0, 1.0), 0.6)
    assert r == (0.6, 0.6, 0.6, 1.0)


def test_mean_intensity_in_polygon():
    img = np.zeros((20, 20), dtype=np.float32)
    img[2:6, 2:6] = 10.0
    # affine maps pixel (row, col) -> micron (y, x) as identity (+0.5 offset)
    affine = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=float)
    poly = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    assert mean_intensity_in_polygon(img, affine, poly) == pytest.approx(10.0)

    outside = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    assert mean_intensity_in_polygon(img, affine, outside) is None


# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------


def _gene_rows():
    return [
        {"gene": "AAA", "count": 5, "rgba": (0.9, 0.2, 0.2, 1.0), "glyph": "●"},
        {"gene": "BBB", "count": 3, "rgba": (0.2, 0.7, 0.9, 1.0), "glyph": "■"},
        {"gene": "CCC", "count": 1, "rgba": (0.3, 0.9, 0.4, 1.0), "glyph": "◆"},
    ]


def test_pie_chart_renders(qapp):
    pie = V.CellInfoPieChart()
    pie.resize(160, 160)
    pie.set_slices(_gene_rows())
    pm = pie.grab()
    assert not pm.isNull()
    # An empty pie must not crash.
    pie.set_slices([])
    assert not pie.grab().isNull()


def _cell_dict(cell_id="202", color="#ffe14d"):
    return {
        "cell_id": cell_id,
        "color": color,
        "gene_rows": _gene_rows(),
        "total": 9,
        "area": 123.4,
        "intensities": [("DAPI", 42.0), ("PolyT", None)],
    }


def _divider_count(overlay):
    layout = overlay._row_layout
    count = 0
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is not None and widget.objectName() == "CellDivider":
            count += 1
    return count


def test_cell_info_overlay_shows_instruction_and_scales_panels(qapp):
    overlay = V.CellInfoOverlay(None)
    assert "close this window" in overlay._instruction.text().lower()

    overlay.set_cells([_cell_dict("202")])
    assert len(overlay.cell_panels()) == 1
    assert _divider_count(overlay) == 0  # no divider for a single cell
    assert not overlay.grab().isNull()

    # A second cell adds a second panel to the right (does not replace the first),
    # separated by a single divider rule.
    overlay.set_cells([_cell_dict("202", "#ffe14d"), _cell_dict("303", "#4dd2ff")])
    assert len(overlay.cell_panels()) == 2
    assert _divider_count(overlay) == 1
    assert not overlay.grab().isNull()

    overlay.set_cells([])
    assert overlay.cell_panels() == []


def test_cell_summary_and_gene_list_html():
    cell = _cell_dict("202")
    summary = V._cell_summary_html(cell)
    assert "Total transcripts:" in summary and "9" in summary
    assert "123.4" in summary and "µm²" in summary and "DAPI" in summary
    genes = V._cell_gene_list_html(cell)
    assert "AAA" in genes and "BBB" in genes and "CCC" in genes
    # No image channels -> explicit note.
    assert "no image channels loaded" in V._cell_summary_html(
        {"total": 0, "area": None, "gene_rows": [], "intensities": []}
    )


# ---------------------------------------------------------------------------
# Controller wiring against a fake viewer
# ---------------------------------------------------------------------------


class _FakeLayer:
    def __init__(self, name, data, **kw):
        self.name = name
        self.kw = dict(kw)
        self.visible = kw.get("visible", True)
        self.size = kw.get("size", 1.0)
        self.symbol = kw.get("symbol", "disc")
        self.face_color = kw.get("face_color")
        self.edge_width = kw.get("edge_width")
        self.affine = kw.get("affine")
        self._pick_value = None
        self._data = data if isinstance(data, list) else np.asarray(data)
        self.shown = np.asarray(
            kw.get("shown", np.ones(len(self._data), dtype=bool)), dtype=bool
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value if isinstance(value, list) else np.asarray(value)
        self.symbol = "disc"

    def get_value(self, _position, **_kwargs):
        return self._pick_value


class _FakeLayerList(list):
    def move(self, src, dst):
        layer = self.pop(src)
        self.insert(dst, layer)


class _FakeAffine:
    def __init__(self, matrix):
        self.affine_matrix = np.asarray(matrix, dtype=float)


class _FakeImageLayer:
    def __init__(self, name, data, matrix):
        self.name = name
        self.data = data
        self.affine = _FakeAffine(matrix)
        self.visible = True


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.mouse_drag_callbacks = []
        self.window = types.SimpleNamespace(_qt_viewer=None)

    def add_points(self, coords, name=None, **kw):
        layer = _FakeLayer(name, coords, **kw)
        self.layers.append(layer)
        return layer

    def add_shapes(self, data, name=None, **kw):
        layer = _FakeLayer(name, list(data), **kw)
        self.layers.append(layer)
        return layer


def _demo_store():
    rows = []
    for i in range(4):
        rows.append(("AAA", float(i), 10.0 + i, False))
    for i in range(2):
        rows.append(("BBB", 100.0 + i, 20.0 + i, False))
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    return build_gene_point_groups(df, x_col="x", y_col="y", gene_col="gene", background_col="background")


def _fake_sdata():
    # Two proseg cells; identity px<->um images. Cell 202 has 3 transcripts,
    # cell 303 has 2, so their pies/gene lists differ.
    polys = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),      # cell 202
        Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),  # cell 303
    ]
    gdf = gpd.GeoDataFrame({"geometry": polys}, index=[202, 303])
    points = pd.DataFrame(
        {
            "x": [2.0, 3.0, 4.0, 25.0, 26.0],
            "y": [2.0, 3.0, 4.0, 25.0, 26.0],
            "gene": ["AAA", "AAA", "BBB", "AAA", "BBB"],
            "assignment": [202, 202, 202, 303, 303],
        }
    )
    return types.SimpleNamespace(
        shapes={"MOSAIK_proseg": gdf},
        points={"transcripts": points},
    )


def _controller(qapp):
    args = types.SimpleNamespace(
        gene_spot_size=2.0,
        gene_max_render_points=40_000_000,
        gene_hide_background=False,
        gene_show_controls=False,
        random_state=42,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = None
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation,
        "TEST",
        {"points_key": "transcripts", "store": _demo_store(), "build_seconds": 0.0},
    )
    ctrl._active_sdata = _fake_sdata()
    # An identity-affine image channel so intensities can be computed.
    img = np.zeros((12, 12), dtype=np.float32)
    img[0:10, 0:10] = 7.0
    ctrl.viewer.layers.append(
        _FakeImageLayer("Image | DAPI", img, [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    )
    return ctrl


def _event(x_um, y_um):
    # napari world position is (y, x) for a 2D viewer.
    return types.SimpleNamespace(
        type="mouse_press",
        position=(y_um, x_um),
        pos=(x_um, y_um),
        view_direction=None,
        dims_displayed=(0, 1),
    )


def _dispatch_click(ctrl, event):
    event.type = "mouse_press"
    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)


def test_pick_cell_from_event_uses_proseg_shapes(qapp):
    ctrl = _controller(qapp)
    picked = ctrl._pick_cell_from_event(_event(5.0, 5.0))
    assert picked is not None and picked[0] == 202
    assert ctrl._pick_cell_from_event(_event(50.0, 50.0)) is None


def test_cell_picking_gated_on_proseg_layer_visibility(qapp):
    ctrl = _controller(qapp)
    seg = _FakeImageLayer(
        "Segmentation | Proseg", np.zeros((4, 4)),
        [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]],
    )
    ctrl.viewer.layers.append(seg)

    # Visible ProSeg layer -> a click inside a mask highlights the cell.
    _dispatch_click(ctrl, _event(5.0, 5.0))
    assert ctrl._selected_cells.get("TEST")

    # Hiding the ProSeg layer drops current highlights...
    seg.visible = False
    ctrl._on_segmentation_visibility_changed()
    assert not ctrl._selected_cells

    # ...and while hidden, clicks no longer highlight cells.
    _dispatch_click(ctrl, _event(5.0, 5.0))
    assert not ctrl._selected_cells.get("TEST")

    # Showing it again re-enables click-to-highlight.
    seg.visible = True
    _dispatch_click(ctrl, _event(5.0, 5.0))
    assert ctrl._selected_cells.get("TEST")


def test_add_cell_selection_draws_layers_and_populates_overlay(qapp):
    ctrl = _controller(qapp)
    _cid, geom = ctrl._pick_cell_from_event(_event(5.0, 5.0))
    ctrl._add_cell_selection("TEST", 202, geom)

    boundary = ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER)
    links = ctrl._get_layer_by_name(V.CELL_INSPECTOR_LINKS_LAYER)
    assert boundary is not None
    assert links is not None and len(links.data) == 3  # one line per transcript

    # Links must sit below every transcript points layer.
    state = ctrl._gene_inspector_states["TEST"]
    names = {str(n) for n in state.layer_names}
    layer_names = [str(layer.name) for layer in ctrl.viewer.layers]
    links_idx = layer_names.index(V.CELL_INSPECTOR_LINKS_LAYER)
    gene_indices = [i for i, n in enumerate(layer_names) if n in names]
    assert links_idx < min(gene_indices)

    entries = ctrl._selected_cells["TEST"]
    assert [e["cell_id"] for e in entries] == [202]
    assert entries[0]["color"] == V.CELL_HIGHLIGHT_COLORS[0]
    overlay = ctrl._cell_info_overlay
    assert overlay is not None and len(overlay.cell_panels()) == 1
    assert entries[0]["intensities"] and entries[0]["intensities"][0][0] == "DAPI"


def test_add_cell_selection_draws_loading_placeholder_before_worker_finishes(
    qapp, monkeypatch
):
    ctrl = _controller(qapp)
    monkeypatch.setattr(ctrl, "_start_cell_selection_build", lambda *args: None)
    _cid, geom = ctrl._pick_cell_from_event(_event(5.0, 5.0))

    ctrl._add_cell_selection("TEST", 202, geom)

    entry = ctrl._selected_cells["TEST"][0]
    assert entry["loading"] is True
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER) is not None
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_LINKS_LAYER) is None


def test_multiple_cells_accumulate_with_distinct_colours(qapp):
    ctrl = _controller(qapp)
    _c1, geom1 = ctrl._pick_cell_from_event(_event(5.0, 5.0))
    ctrl._add_cell_selection("TEST", 202, geom1)
    _c2, geom2 = ctrl._pick_cell_from_event(_event(25.0, 25.0))
    ctrl._add_cell_selection("TEST", 303, geom2)

    entries = ctrl._selected_cells["TEST"]
    assert [e["cell_id"] for e in entries] == [202, 303]
    assert entries[0]["color"] == V.CELL_HIGHLIGHT_COLORS[0]
    assert entries[1]["color"] == V.CELL_HIGHLIGHT_COLORS[1]

    # One boundary layer holds both cells' outlines, coloured per cell.
    boundary = ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER)
    assert len(boundary.data) == 2
    edge = np.asarray(boundary.kw["edge_color"], dtype=float)
    assert edge.shape == (2, 4) and not np.allclose(edge[0], edge[1])
    # Links now cover both cells (3 + 2 transcripts).
    assert len(ctrl._get_layer_by_name(V.CELL_INSPECTOR_LINKS_LAYER).data) == 5
    assert len(ctrl._cell_info_overlay.cell_panels()) == 2

    # Re-clicking an already-highlighted cell does not duplicate it.
    ctrl._add_cell_selection("TEST", 202, geom1)
    assert [e["cell_id"] for e in ctrl._selected_cells["TEST"]] == [202, 303]


def test_empty_click_keeps_cells_and_close_clears_all(qapp):
    ctrl = _controller(qapp)
    _cid, geom = ctrl._pick_cell_from_event(_event(5.0, 5.0))
    ctrl._add_cell_selection("TEST", 202, geom)

    # A click in empty space must NOT deselect the cell.
    _dispatch_click(ctrl, _event(50.0, 50.0))
    assert ctrl._selected_cells.get("TEST")
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER) is not None

    # Simulating the window being closed (visibilityChanged -> False) clears all
    # and drops the overlay so it can be rebuilt fresh.
    ctrl._on_cell_dock_visibility_changed(False)
    assert not ctrl._selected_cells
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER) is None
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_LINKS_LAYER) is None
    assert ctrl._cell_info_overlay is None  # references dropped on close

    # Selecting another cell after closing brings the window back.
    _cid2, geom2 = ctrl._pick_cell_from_event(_event(25.0, 25.0))
    ctrl._add_cell_selection("TEST", 303, geom2)
    assert ctrl._cell_info_overlay is not None
    assert len(ctrl._cell_info_overlay.cell_panels()) == 1
    assert ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER) is not None


def test_boundary_width_scales_from_loaded_mask_outline(qapp):
    ctrl = _controller(qapp)
    # A loaded label-outline layer at 0.2 µm/pixel; contour width defaults to 1px.
    ctrl.viewer.layers.append(
        _FakeImageLayer("Segmentation | Proseg", np.zeros((4, 4)), [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])
    )
    width = ctrl._mask_highlight_edge_width()
    assert width == pytest.approx(0.2 * 1 * V.CELL_BOUNDARY_WIDTH_FACTOR)

    _cid, geom = ctrl._pick_cell_from_event(_event(5.0, 5.0))
    ctrl._add_cell_selection("TEST", 202, geom)
    boundary = ctrl._get_layer_by_name(V.CELL_INSPECTOR_BOUNDARY_LAYER)
    assert boundary.edge_width == pytest.approx(width)
    # The highlight must be far thinner than the old fixed 4 µm and only a little
    # thicker than the 0.2 µm mask outline it sits on.
    assert 0.2 < width < 0.6


def test_boundary_width_falls_back_without_mask_layer(qapp):
    ctrl = _controller(qapp)  # only an "Image | DAPI" layer, no labels layer
    assert ctrl._mask_highlight_edge_width() == pytest.approx(V.CELL_BOUNDARY_FALLBACK_WIDTH_UM)


def test_transcript_click_still_highlights_gene(qapp):
    ctrl = _controller(qapp)
    state = ctrl._gene_inspector_states["TEST"]
    gi = state.store.gene_offsets["AAA"][0]
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    layer._pick_value = 0  # first display index in this group -> a gene
    _dispatch_click(ctrl, _event(2.0, 2.0))
    assert state.highlighted_genes            # a gene got highlighted
    assert "TEST" not in ctrl._selected_cells  # and no cell was selected
