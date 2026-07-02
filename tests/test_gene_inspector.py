"""Integration tests for the Inspect Genes per-gene transcript renderer.

These use a lightweight fake napari viewer (no OpenGL canvas) to exercise the
controller's layer rebuild / toggle logic, plus a headless QApplication for the
Qt widget and marker-icon drawing.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import GENE_MARKER_SYMBOLS, build_gene_point_groups


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    app = QApplication.instance() or QApplication([])
    return app


class _FakeLayer:
    """Mimics the napari Points quirk where assigning ``data`` resets the
    per-point ``symbol`` back to the default disc, so tests catch a regression
    if the controller forgets to re-apply the symbol after a data change."""

    def __init__(self, name, data, **kw):
        self.name = name
        self.visible = kw.get("visible", True)
        self.size = kw.get("size", 1.0)
        self.face_color = kw.get("face_color")
        self.edge_color = kw.get("edge_color")
        self.border_color = kw.get("border_color")
        self.symbol = "disc"
        self.antialiasing = 1
        self.border_width = 0.05
        self._data = np.asarray(data)
        if kw.get("symbol"):
            self.symbol = kw["symbol"]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)
        self.symbol = "disc"  # napari resets per-point symbol on data change


class _FakeViewer:
    def __init__(self):
        self.layers = []

    def add_points(self, coords, name=None, **kw):
        layer = _FakeLayer(name, coords, **kw)
        self.layers.append(layer)
        return layer


def _demo_store():
    rows = []
    for i in range(6):
        rows.append(("AAA", float(i), 10.0 + i, i >= 4))       # 4 fg, 2 bg
    for i in range(3):
        rows.append(("BBB", 100.0 + i, 20.0 + i, False))       # 3 fg
    for i in range(2):
        rows.append(("Blank-1", 200.0 + i, 30.0 + i, i == 1))  # control: 1 fg, 1 bg
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    return build_gene_point_groups(df, x_col="x", y_col="y", gene_col="gene", background_col="background")


def _controller(store, qapp):
    args = types.SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000,
        gene_hide_background=False, gene_show_controls=False,
        random_state=42, point_size=2.0, point_opacity=0.5,
        assigned_color="yellow", unassigned_color="#d62728", hide_transcripts=False,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = None
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation, "TEST",
        {"points_key": "transcripts", "store": store, "build_seconds": 0.0},
    )
    return ctrl


def _total(ctrl, state):
    return sum(len(ctrl._get_layer_by_name(n).data) for n in state.layer_names)


def test_controller_toggle_background_controls_and_teardown(qapp):
    store = _demo_store()
    assert store.group_symbols == list(GENE_MARKER_SYMBOLS[:3])
    assert store.control_genes == {"Blank-1"}
    ctrl = _controller(store, qapp)
    state = ctrl._gene_inspector_states["TEST"]

    # Controls excluded by default: AAA(6) + BBB(3).
    assert _total(ctrl, state) == 9

    # Per-gene toggle (debounced -> flush manually).
    ctrl.set_gene_visible("TEST", "AAA", False)
    ctrl._flush_gene_group_rebuild("TEST")
    assert _total(ctrl, state) == 3

    # Hide background: AAA fg(4) + BBB(3).
    ctrl.set_gene_visible("TEST", "AAA", True)
    ctrl._flush_gene_group_rebuild("TEST")
    ctrl.set_gene_hide_background("TEST", True)
    assert _total(ctrl, state) == 7

    # Show controls + all genes + background on: 6 + 3 + 2.
    ctrl.set_gene_hide_background("TEST", False)
    ctrl.set_gene_show_controls("TEST", True)
    state.show_controls = True
    ctrl.set_all_genes_visible("TEST", True)
    assert _total(ctrl, state) == 11

    # Turning controls off drops the control points.
    ctrl.set_gene_show_controls("TEST", False)
    assert _total(ctrl, state) == 9

    # Hide all.
    ctrl.set_all_genes_visible("TEST", False)
    assert _total(ctrl, state) == 0

    # Spot size applies to every layer.
    ctrl.set_gene_spot_size("TEST", 5.0)
    assert all(float(ctrl._get_layer_by_name(n).size) == 5.0 for n in state.layer_names)

    # Symbols + per-point colours are correct after a full rebuild. (show_controls
    # is off here, so the control-only group is empty; an empty layer's symbol is
    # irrelevant, hence the len() guard.)
    ctrl.set_all_genes_visible("TEST", True)
    for gi, n in enumerate(state.layer_names):
        layer = ctrl._get_layer_by_name(n)
        if len(layer.data):
            assert layer.symbol == store.group_symbols[gi]
            assert layer.face_color.shape == (len(layer.data), 4)

    # And the symbol survives a per-gene toggle (data change resets it in napari;
    # the controller must re-apply it).
    gi = store.gene_offsets["AAA"][0]
    ctrl.set_gene_visible("TEST", "AAA", False)
    ctrl._flush_gene_group_rebuild("TEST")
    ctrl.set_gene_visible("TEST", "AAA", True)
    ctrl._flush_gene_group_rebuild("TEST")
    assert ctrl._get_layer_by_name(state.layer_names[gi]).symbol == store.group_symbols[gi]

    ctrl._teardown_gene_inspector("TEST", restore=False)
    assert [l for l in ctrl.viewer.layers if "genes" in str(l.name)] == []


def test_marker_pixmaps_render_for_all_symbols(qapp):
    for sym in GENE_MARKER_SYMBOLS:
        pm = V.make_gene_marker_pixmap((0.2, 0.8, 0.3, 1.0), sym, px=22)
        assert not pm.isNull()


def test_gene_inspector_widget_populate_and_callbacks(qapp):
    store = _demo_store()
    calls = []
    w = V.GeneInspectorWidget(
        close_callback=lambda ds: calls.append(("close", ds)),
        set_gene_visible_callback=lambda *a: calls.append(("vis", a)),
        set_all_genes_callback=lambda *a: calls.append(("all", a)),
        set_spot_size_callback=lambda *a: calls.append(("size", a)),
        set_hide_background_callback=lambda *a: calls.append(("bg", a)),
        set_show_controls_callback=lambda *a: calls.append(("ctrl", a)),
    )
    from napari_compare_xenium_merscope.utils import assign_gene_visuals

    w.populate("TEST", list(store.genes), assign_gene_visuals(store.genes),
               dict(store.gene_counts), set(store.control_genes),
               {"AAA", "BBB"}, False, False, 2.0)
    assert len(w._checkboxes) == 3
    assert w._items["Blank-1"].isHidden()      # control hidden by default
    assert not w._items["AAA"].isHidden()
    assert w._checkboxes["AAA"].isChecked()

    w._checkboxes["AAA"].setChecked(False)
    assert ("vis", ("TEST", "AAA", False)) in calls

    # Showing controls reveals the control row.
    w._show_controls_check.setChecked(True)
    assert not w._items["Blank-1"].isHidden()
    assert ("ctrl", ("TEST", True)) in calls
