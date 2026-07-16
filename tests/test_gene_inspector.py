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
from napari_compare_xenium_merscope.utils import (
    GENE_MARKER_SYMBOLS,
    build_gene_point_groups,
    gene_marker_symbol_label,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    app = QApplication.instance() or QApplication([])
    return app


class _FakeLayer:
    """Small Points-layer double including napari's per-point ``shown`` mask."""

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
        self._pick_value = None
        self._data = np.asarray(data)
        self.shown = np.asarray(
            kw.get("shown", np.ones(len(self._data), dtype=bool)), dtype=bool
        )
        if kw.get("symbol"):
            self.symbol = kw["symbol"]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)
        self.symbol = "disc"  # napari resets per-point symbol on data change

    def get_value(self, _position, **_kwargs):
        return self._pick_value


class _FakeViewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []

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
        random_state=42,
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
    return sum(
        int(np.count_nonzero(ctrl._get_layer_by_name(n).shown))
        for n in state.layer_names
    )


def _dispatch_click(ctrl, event):
    event.type = "mouse_press"
    if not hasattr(event, "pos"):
        y, x = np.asarray(event.position, dtype=float).ravel()[-2:]
        event.pos = (x, y)
    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)


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
    ctrl.set_gene_hide_assigned("TEST", True)
    assert _total(ctrl, state) == 0
    ctrl.set_gene_hide_assigned("TEST", False)

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

    # A per-gene toggle updates only the visibility mask: fixed geometry and its
    # marker symbol both remain untouched.
    gi = store.gene_offsets["AAA"][0]
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    data_id = id(layer.data)
    ctrl.set_gene_visible("TEST", "AAA", False)
    ctrl._flush_gene_group_rebuild("TEST")
    ctrl.set_gene_visible("TEST", "AAA", True)
    ctrl._flush_gene_group_rebuild("TEST")
    assert layer.symbol == store.group_symbols[gi]
    assert id(layer.data) == data_id

    ctrl._teardown_gene_inspector("TEST")
    assert [l for l in ctrl.viewer.layers if "Genes" in str(l.name)] == []


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
        set_hide_assigned_callback=lambda *a: calls.append(("assigned", a)),
        set_colour_by_assignment_callback=lambda *a: calls.append(("assign_color", a)),
    )
    from napari_compare_xenium_merscope.utils import assign_gene_visuals

    layout = [("gene", g) for g in store.genes]
    w.populate("TEST", layout, assign_gene_visuals(store.genes),
               dict(store.gene_counts), set(store.control_genes),
               {"AAA", "BBB"}, {}, "alphabetical", False, False, False, 2.0)
    assert len(w._checkboxes) == 3
    assert w._hide_assigned_check.text() == "Hide assigned spots"
    assert w._hide_bg_check.text() == "Hide unassigned spots"
    assert w._colour_by_assignment_check.text() == "Colour transcripts by assigned/unassigned"
    assert w._items["Blank-1"].isHidden()      # control hidden by default
    assert not w._items["AAA"].isHidden()
    assert w._checkboxes["AAA"].isChecked()

    w._checkboxes["AAA"].setChecked(False)
    assert ("vis", ("TEST", "AAA", False)) in calls

    # Showing controls reveals the control row.
    w._show_controls_check.setChecked(True)
    assert not w._items["Blank-1"].isHidden()
    assert ("ctrl", ("TEST", True)) in calls

    w._hide_assigned_check.setChecked(True)
    assert ("assigned", ("TEST", True)) in calls

    w._colour_by_assignment_check.setChecked(True)
    assert ("assign_color", ("TEST", True)) in calls


_REF = {
    "SLC17A7": {"broad": "Neuron", "fine": "L2/3 IT"},
    "SST": {"broad": "Neuron", "fine": "Sst"},
    "AQP4": {"broad": "Astrocyte", "fine": "Protoplasmic astrocyte"},
    "MOG": {"broad": "Oligodendrocyte", "fine": "Myelinating oligodendrocyte"},
}


def _ref_controller(qapp):
    rows = []
    for gene in _REF:
        for i in range(4):
            rows.append((gene, float(i), float(i), i >= 3))  # 3 fg + 1 bg each
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background", reference=_REF,
    )
    args = types.SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000,
        gene_hide_background=False, gene_show_controls=False, random_state=42,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = None
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation, "TEST",
        {"points_key": "transcripts", "store": store, "reference": _REF, "build_seconds": 0.0},
    )
    return ctrl


def test_set_gene_ordering_recolors_and_keeps_symbols(qapp):
    ctrl = _ref_controller(qapp)
    state = ctrl._gene_inspector_states["TEST"]
    # A reference was found: default ordering is coarse and points are coarse-coloured.
    assert state.ordering == "coarse" and state.color_kind == "coarse"
    assert state.reference is not None
    symbols_before = list(state.store.group_symbols)
    coarse_rgba = tuple(state.gene_visuals["SLC17A7"].rgba)

    # Switch to fine: colours change, but the symbol grouping (and thus layers) is
    # unchanged so only a recolour happens.
    ctrl.set_gene_ordering("TEST", "fine")
    assert state.ordering == "fine" and state.color_kind == "fine"
    assert state.store.group_symbols == symbols_before
    fine_rgba = tuple(state.gene_visuals["SLC17A7"].rgba)
    assert coarse_rgba != fine_rgba

    # A–Z keeps whatever colours are currently applied (here: fine).
    ctrl.set_gene_ordering("TEST", "alphabetical")
    assert state.ordering == "alphabetical" and state.color_kind == "fine"
    assert tuple(state.gene_visuals["SLC17A7"].rgba) == fine_rgba

    # Back to broad recolours again.
    ctrl.set_gene_ordering("TEST", "coarse")
    assert state.color_kind == "coarse"
    assert tuple(state.gene_visuals["SLC17A7"].rgba) == coarse_rgba


def _display_range_for_gene(state, gene):
    gi = state.store.gene_offsets[gene][0]
    for start, end, name in state.group_display_ranges[gi]:
        if name == gene:
            return gi, start, end
    raise AssertionError(f"No display range for {gene}")


def test_colour_by_assignment_recolors_points_and_restores_current_scheme(qapp):
    ctrl = _ref_controller(qapp)
    state = ctrl._gene_inspector_states["TEST"]
    ctrl.set_gene_ordering("TEST", "fine")
    fine_rgba = np.asarray(state.gene_visuals["SLC17A7"].rgba, dtype=np.float32)

    gi, start, end = _display_range_for_gene(state, "SLC17A7")
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    np.testing.assert_allclose(
        layer.face_color[start:end], np.tile(fine_rgba, (end - start, 1)), atol=1e-6
    )
    assert layer.symbol == state.store.group_symbols[gi]

    ctrl.set_gene_colour_by_assignment("TEST", True)
    gi, start, end = _display_range_for_gene(state, "SLC17A7")
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    _g, _fg_start, fg_end, bg_end = state.store.gene_offsets["SLC17A7"]
    fg_count = fg_end - _fg_start
    bg_count = bg_end - fg_end
    np.testing.assert_allclose(
        layer.face_color[start:start + fg_count],
        np.tile(V.GENE_ASSIGNED_RGBA, (fg_count, 1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        layer.face_color[start + fg_count:start + fg_count + bg_count],
        np.tile(V.GENE_UNASSIGNED_RGBA, (bg_count, 1)),
        atol=1e-6,
    )
    assert layer.symbol == state.store.group_symbols[gi]

    ctrl.set_gene_colour_by_assignment("TEST", False)
    gi, start, end = _display_range_for_gene(state, "SLC17A7")
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    np.testing.assert_allclose(
        layer.face_color[start:end], np.tile(fine_rgba, (end - start, 1)), atol=1e-6
    )
    assert layer.symbol == state.store.group_symbols[gi]


def _order_widget(qapp):
    return V.GeneInspectorWidget(
        close_callback=lambda ds: None,
        set_gene_visible_callback=lambda *a: None,
        set_all_genes_callback=lambda *a: None,
        set_spot_size_callback=lambda *a: None,
        set_hide_background_callback=lambda *a: None,
        set_show_controls_callback=lambda *a: None,
        set_hide_assigned_callback=lambda *a: None,
        set_ordering_callback=lambda *a: None,
    )


def test_widget_grouped_populate_creates_headers_and_filters(qapp):
    from napari_compare_xenium_merscope.utils import assign_gene_visuals

    w = _order_widget(qapp)
    visuals = assign_gene_visuals(["SLC17A7", "SST", "AQP4"])
    layout = [
        ("header", "Neuron", (0.2, 0.3, 0.9, 1.0)),
        ("gene", "SLC17A7"), ("gene", "SST"),
        ("header", "Astrocyte", (0.2, 0.8, 0.3, 1.0)),
        ("gene", "AQP4"),
    ]
    counts = {"SLC17A7": 5, "SST": 3, "AQP4": 4}
    w.populate("TEST", layout, visuals, counts, set(), {"SLC17A7", "SST", "AQP4"},
               {}, "coarse", False, False, False, 2.0)
    assert len(w._checkboxes) == 3
    assert [t for _item, t, _g in w._headers] == ["Neuron", "Astrocyte"]

    # Filtering to SST leaves the Neuron header visible but hides the Astrocyte one.
    w._filter_edit.setText("SST")
    headers = {t: item for item, t, _g in w._headers}
    assert not headers["Neuron"].isHidden()
    assert headers["Astrocyte"].isHidden()


def test_clicking_group_header_toggles_group_genes(qapp):
    from napari_compare_xenium_merscope.utils import assign_gene_visuals

    calls = []
    w = V.GeneInspectorWidget(
        close_callback=lambda ds: None,
        set_gene_visible_callback=lambda *a: None,
        set_all_genes_callback=lambda *a: None,
        set_spot_size_callback=lambda *a: None,
        set_hide_background_callback=lambda *a: None,
        set_show_controls_callback=lambda *a: None,
        set_hide_assigned_callback=lambda *a: None,
        set_ordering_callback=lambda *a: None,
        set_genes_visible_callback=lambda ds, genes, on: calls.append((ds, tuple(genes), on)),
    )
    visuals = assign_gene_visuals(["SLC17A7", "SST", "AQP4"])
    layout = [
        ("header", "Neuron", (0.2, 0.3, 0.9, 1.0)),
        ("gene", "SLC17A7"), ("gene", "SST"),
        ("header", "Astrocyte", (0.2, 0.8, 0.3, 1.0)),
        ("gene", "AQP4"),
    ]
    w.populate("TEST", layout, visuals, {"SLC17A7": 1, "SST": 1, "AQP4": 1}, set(),
               {"SLC17A7", "SST", "AQP4"}, {}, "coarse", False, False, False, 2.0)
    neuron_item = next(item for item, t, _g in w._headers if t == "Neuron")
    header = w._gene_list.itemWidget(neuron_item)

    # All on -> a header click turns the group off (and leaves other groups alone).
    header.clicked.emit()
    assert calls[-1] == ("TEST", ("SLC17A7", "SST"), False)
    assert not w._checkboxes["SLC17A7"].isChecked()
    assert not w._checkboxes["SST"].isChecked()
    assert w._checkboxes["AQP4"].isChecked()

    # Clicking again turns the whole group back on.
    header.clicked.emit()
    assert calls[-1] == ("TEST", ("SLC17A7", "SST"), True)
    assert w._checkboxes["SLC17A7"].isChecked()


def test_controller_set_genes_visible_batch(qapp):
    ctrl = _ref_controller(qapp)
    state = ctrl._gene_inspector_states["TEST"]

    def total():
        return sum(
            int(np.count_nonzero(ctrl._get_layer_by_name(n).shown))
            for n in state.layer_names
        )

    ctrl.set_all_genes_visible("TEST", True)
    full = total()
    ctrl.set_genes_visible("TEST", ["SLC17A7", "SST"], False)
    assert "SLC17A7" not in state.enabled_genes and "SST" not in state.enabled_genes
    assert total() < full
    ctrl.set_genes_visible("TEST", ["SLC17A7", "SST"], True)
    assert "SLC17A7" in state.enabled_genes and total() == full


def test_widget_alphabetical_appends_celltype_labels(qapp):
    from napari_compare_xenium_merscope.utils import assign_gene_visuals

    w = _order_widget(qapp)
    visuals = assign_gene_visuals(["SLC17A7", "AQP4"])
    layout = [("gene", "AQP4"), ("gene", "SLC17A7")]
    labels = {"SLC17A7": "Neuron / L2/3 IT", "AQP4": "Astrocyte / Protoplasmic astrocyte"}
    w.populate("TEST", layout, visuals, {"SLC17A7": 1, "AQP4": 1}, set(),
               {"SLC17A7", "AQP4"}, labels, "alphabetical", False, False, False, 2.0)
    assert w._headers == []
    assert "Neuron / L2/3 IT" in w._checkboxes["SLC17A7"].text()


def test_move_gene_layers_to_bottom(qapp):
    store = _demo_store()
    ctrl = _controller(store, qapp)
    state = ctrl._gene_inspector_states["TEST"]

    class _MoveLayerList(list):
        def __init__(self, *a):
            super().__init__(*a)
            self.moved = None

        def move_multiple(self, sources, dest):
            self.moved = (list(sources), dest)

    # An image on top of the (already-created) gene layers, in list order.
    layers = _MoveLayerList(
        [ctrl._get_layer_by_name(n) for n in state.layer_names]
        + [types.SimpleNamespace(name="Image | DAPI")]
    )
    ctrl.viewer.layers = layers
    ctrl._move_gene_layers_to_bottom(state)
    # The gene layers (indices 0..n-1) are moved as a block to the bottom (0).
    assert layers.moved == (list(range(len(state.layer_names))), 0)


def test_gene_layer_names_use_marker_symbol_labels(qapp):
    store = _demo_store()
    ctrl = _controller(store, qapp)
    state = ctrl._gene_inspector_states["TEST"]

    assert state.layer_names == [
        f"Genes | {gene_marker_symbol_label(symbol)}"
        for symbol in store.group_symbols
    ]


def test_clicking_transcripts_accumulates_highlights_and_empty_click_clears(qapp):
    store = _demo_store()
    ctrl = _controller(store, qapp)
    messages = []
    ctrl.set_status_callback(messages.append)
    state = ctrl._gene_inspector_states["TEST"]
    ctrl.set_gene_spot_size("TEST", 0.5)

    aaa_layer = ctrl._get_layer_by_name(state.layer_names[store.gene_offsets["AAA"][0]])
    bbb_layer = ctrl._get_layer_by_name(state.layer_names[store.gene_offsets["BBB"][0]])
    aaa_layer._pick_value = 0
    event = types.SimpleNamespace(
        type="mouse_press",
        position=(10.0, 1.0),
        view_direction=None,
        dims_displayed=(0, 1),
    )

    _dispatch_click(ctrl, event)

    assert state.highlighted_genes == ["AAA"]
    assert "AAA: 6 counts" in messages[-1]
    assert "Click in empty space to deselect all genes." in messages[-1]
    assert isinstance(aaa_layer.size, np.ndarray)
    assert aaa_layer.size.tolist() == [V.GENE_HIGHLIGHT_SPOT_SIZE] * 6

    aaa_layer._pick_value = None
    bbb_layer._pick_value = 0
    _dispatch_click(ctrl, event)

    assert state.highlighted_genes == ["AAA", "BBB"]
    assert "AAA: 6 counts" in messages[-1]
    assert "BBB: 3 counts" in messages[-1]
    assert isinstance(aaa_layer.size, np.ndarray)
    assert aaa_layer.size.tolist() == [V.GENE_HIGHLIGHT_SPOT_SIZE] * 6
    assert isinstance(bbb_layer.size, np.ndarray)
    assert bbb_layer.size.tolist() == [V.GENE_HIGHLIGHT_SPOT_SIZE] * 3

    aaa_layer._pick_value = None
    bbb_layer._pick_value = None
    _dispatch_click(ctrl, event)

    assert state.highlighted_genes == []
    assert messages[-1] == "Click on any transcript to highlight that gene"
    assert float(aaa_layer.size) == 0.5
    assert float(bbb_layer.size) == 0.5


def test_drag_callback_never_runs_inspection_hit_tests(qapp, monkeypatch):
    ctrl = _controller(_demo_store(), qapp)
    event = types.SimpleNamespace(
        type="mouse_press",
        pos=(10.0, 10.0),
        position=(10.0, 10.0),
        view_direction=None,
        dims_displayed=(0, 1),
    )
    calls = []
    monkeypatch.setattr(
        ctrl, "_handle_viewer_click", lambda _event: calls.append("picked")
    )

    callback = ctrl._on_viewer_mouse_press(ctrl.viewer, event)
    next(callback)
    assert calls == []  # mouse-down yields immediately

    event.type = "mouse_move"
    event.pos = (10.0 + V.INSPECT_CLICK_DRAG_THRESHOLD_PX + 1.0, 10.0)
    next(callback)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(callback)

    assert calls == []
