"""Gene panel at Atera scale: the model/view list, selection mode and its helpers.

Everything here is synthetic (no store under ``/Users/...``) and runs offscreen. The
controller tests use the same fake napari viewer pattern as ``test_gene_inspector``.
"""
from __future__ import annotations

import os
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import anndata as ad
import atera_stores
import numpy as np
import pandas as pd
import pytest
import zarr

from napari_compare_xenium_merscope import platforms as P
from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import (
    CONTROL_GROUP_TITLE,
    GeneSlotAllocator,
    assign_gene_visuals,
    build_gene_point_groups,
    control_genes_from_is_gene,
    gene_palette_rgba,
    is_control_gene,
    load_cell_type_marker_reference,
)

#: CI machines are slower than a developer laptop; the measured local numbers
#: (populate + first show of 24.7k rows ~0.05 s, a keystroke ~5 ms) are far below
#: these bounds, which only catch a return to per-row widgets (224 s before).
POPULATE_BUDGET_S = 1.5
KEYSTROKE_BUDGET_S = 0.1


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


def _widget(calls=None, *, overview=True):
    calls = calls if calls is not None else []
    return V.GeneInspectorWidget(
        close_callback=lambda ds: calls.append(("close", ds)),
        set_gene_visible_callback=lambda *a: calls.append(("vis", a)),
        set_all_genes_callback=lambda *a: calls.append(("all", a)),
        set_spot_size_callback=lambda *a: calls.append(("size", a)),
        set_hide_background_callback=lambda *a: calls.append(("bg", a)),
        set_show_controls_callback=lambda *a: calls.append(("ctrl", a)),
        set_hide_assigned_callback=lambda *a: calls.append(("assigned", a)),
        set_ordering_callback=lambda *a: calls.append(("order", a)),
        set_genes_visible_callback=lambda ds, genes, on: calls.append(("genes", ds, tuple(genes), on)),
        set_overview_callback=(lambda *a: calls.append(("overview", a))) if overview else None,
    )


def _atera_scale_panel(n_genes=18_049, n_controls=6_628):
    genes = [f"GENE{i:05d}" for i in range(n_genes)]
    families = atera_stores.CONTROL_FAMILIES
    controls = [f"{families[i % len(families)]}_{i:05d}" for i in range(n_controls)]
    # Twelve broad groups of genes, as a marker reference would produce.
    layout: list[tuple] = []
    size = n_genes // 12 + 1
    for g in range(12):
        members = genes[g * size:(g + 1) * size]
        if members:
            layout.append(("header", f"Group {g}", (0.2, 0.4, 0.8, 1.0)))
            layout.extend(("gene", name) for name in members)
    layout.append(("header", CONTROL_GROUP_TITLE, (0.6, 0.6, 0.6, 1.0)))
    layout.extend(("gene", name) for name in controls)
    counts = {name: i for i, name in enumerate(genes + controls)}
    visuals = {name: V.GeneVisual(rgba=(0.5, 0.5, 0.5, 1.0), symbol="disc") for name in genes + controls}
    return genes, controls, layout, counts, visuals


# -- registry / helpers ---------------------------------------------------------------------


def test_gene_panel_mode_for_platform_and_panel_size():
    assert P.gene_panel_mode_for(P.ATERA, 10) == P.GENE_PANEL_MODE_SELECTION
    assert P.gene_panel_mode_for("atera", 0) == P.GENE_PANEL_MODE_SELECTION
    assert P.gene_panel_mode_for(P.XENIUM, 500) == P.GENE_PANEL_MODE_FULL
    assert P.gene_panel_mode_for(P.MERSCOPE, P.GENE_SELECTION_MODE_FEATURE_THRESHOLD) == P.GENE_PANEL_MODE_FULL
    assert P.gene_panel_mode_for(P.XENIUM, P.GENE_SELECTION_MODE_FEATURE_THRESHOLD + 1) == P.GENE_PANEL_MODE_SELECTION
    assert P.gene_panel_mode_for("TEST", 3) == P.GENE_PANEL_MODE_FULL
    assert P.GENE_SELECTION_SLOT_CAP == 32


def test_slot_allocator_distinct_lowest_free_reuse_and_cap():
    alloc = GeneSlotAllocator(cap=4)
    assert [alloc.allocate(g) for g in "ABCD"] == [0, 1, 2, 3]
    assert alloc.full and alloc.allocate("E") is None
    assert alloc.allocate("B") == 1  # idempotent
    colours = {alloc.visual(g).rgba for g in "ABCD"}
    assert len(colours) == 4  # distinct slots, distinct colours
    assert {alloc.visual(g).symbol for g in "ABCD"} == {"disc"}
    assert alloc.release("B") == 1 and alloc.release("B") is None
    assert alloc.allocate("E") == 1  # the freed slot is reused first
    assert alloc.visual("E").rgba == tuple(gene_palette_rgba(4)[1])
    assert alloc.genes == ["A", "E", "C", "D"]
    assert alloc.visual("B") is None
    alloc.clear()
    assert len(alloc) == 0


def test_is_control_gene_covers_the_five_atera_families_and_matches_is_gene():
    for family in atera_stores.CONTROL_FAMILIES:
        assert is_control_gene(f"{family}_0001")
    for name in ("GFAP", "SNAP25", "NEGR1", "INTS1", "DEPDC1", "UNC5C", "BLNK", "ACTB"):
        assert not is_control_gene(name), name
    frame = atera_stores.synthetic_transcripts()
    names = np.asarray(atera_stores.FEATURE_DICTIONARY)[frame["code"].to_numpy()]
    flagged = np.fromiter((is_control_gene(n) for n in names), dtype=bool, count=len(names))
    np.testing.assert_array_equal(flagged, ~frame["is_gene"].to_numpy(bool))


def test_control_genes_from_is_gene_trusts_the_flag_and_warns_on_disagreement(caplog):
    names = ["AAA", "MYSTERY1", "Blank-1", "NegControlProbe_00001", "UNFLAGGED_Blank"]
    flags = {"AAA": True, "MYSTERY1": False, "Blank-1": False, "NegControlProbe_00001": True}
    with caplog.at_level("WARNING", logger="napari_compare_utils"):
        controls = control_genes_from_is_gene(names, flags, source="test")
    # The flag wins both ways; an unflagged name falls back to the name rule.
    assert controls == {"MYSTERY1", "Blank-1", "UNFLAGGED_Blank"}
    text = caplog.text
    assert "is_gene disagrees" in text and "MYSTERY1" in text and "NegControlProbe_00001" in text
    caplog.clear()
    # Agreement is silent; no flags is exactly the name rule (Xenium/MERSCOPE).
    with caplog.at_level("WARNING", logger="napari_compare_utils"):
        assert control_genes_from_is_gene(["AAA", "Blank-1"], {"AAA": True, "Blank-1": False}) == {"Blank-1"}
        assert control_genes_from_is_gene(names, None) == {
            n for n in names if is_control_gene(n)
        }
    assert "is_gene disagrees" not in caplog.text


def _is_gene_frame():
    return pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "y": np.arange(8, dtype=float),
            "feature_name": ["AAA", "AAA", "BBB", "MYSTERY1", "MYSTERY1", "Blank-1", "BBB", "CCC"],
            # MYSTERY1 has a gene-like name but is a control by the vendor flag.
            "is_gene": [True, True, True, False, False, False, True, True],
        }
    )


def test_point_store_takes_controls_from_the_is_gene_column(caplog):
    frame = _is_gene_frame()
    with caplog.at_level("WARNING", logger="napari_compare_utils"):
        store = build_gene_point_groups(frame, "x", "y", "feature_name", is_gene_col="is_gene")
    assert store.control_genes == {"MYSTERY1", "Blank-1"}
    assert "MYSTERY1" in caplog.text
    # Without the column hook the name rule alone is used (unchanged behaviour).
    plain = build_gene_point_groups(frame, "x", "y", "feature_name")
    assert plain.control_genes == {"Blank-1"}


def test_point_store_controls_match_is_gene_on_the_synthetic_atera_frame():
    frame = atera_stores.synthetic_transcripts()
    names = np.asarray(atera_stores.FEATURE_DICTIONARY)[frame["code"].to_numpy()]
    points = pd.DataFrame(
        {"x": frame["x"].to_numpy(float), "y": frame["y"].to_numpy(float),
         "feature_name": names, "is_gene": frame["is_gene"].to_numpy(bool)}
    )
    store = build_gene_point_groups(points, "x", "y", "feature_name", is_gene_col="is_gene", max_points=10)
    assert store.control_genes == set(names[~points["is_gene"].to_numpy(bool)].tolist())


def test_selection_layout_files_an_is_gene_false_feature_under_the_control_heading():
    store = build_gene_point_groups(_is_gene_frame(), "x", "y", "feature_name", is_gene_col="is_gene")
    state = V.GeneInspectorState(
        dataset="ATERA", points_key="transcripts", store=store, gene_visuals={},
        layer_names=[], enabled_genes=set(), spot_size=1.0,
        panel_mode=P.GENE_PANEL_MODE_SELECTION,
    )
    # The scheme layout files MYSTERY1 among the genes (its name is not a control).
    layout = [
        ("header", "Neuron", (0.2, 0.3, 0.9, 1.0)),
        ("gene", "AAA"), ("gene", "MYSTERY1"), ("gene", "BBB"),
        ("header", "Unclassified", (0.5, 0.5, 0.5, 1.0)),
        ("gene", "CCC"),
        ("header", CONTROL_GROUP_TITLE, (0.6, 0.6, 0.6, 1.0)),
        ("gene", "Blank-1"),
    ]
    out = V.ComparisonViewerController._selection_gene_layout(
        types.SimpleNamespace(_gene_control_set=V.ComparisonViewerController._gene_control_set),
        state, layout,
    )
    heading = out.index(("header", CONTROL_GROUP_TITLE, (0.6, 0.6, 0.6, 1.0)))
    assert [e[1] for e in out[heading + 1:]] == ["Blank-1", "MYSTERY1"]
    assert ("gene", "MYSTERY1") not in out[:heading]
    assert [e[1] for e in out[:heading] if e[0] == "gene"] == ["AAA", "BBB", "CCC"]


def _write_marker_tables(path):
    root = zarr.open_group(path, mode="w", zarr_format=2)
    root.create_group("tables")
    # A large "cell" table with no marker data: it must never be read in full.
    big = ad.AnnData(
        X=np.ones((50, 40), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["x"] * 50}, index=[f"c{i}" for i in range(50)]),
    )
    big.write_zarr(path / "tables" / "table")
    marker = ad.AnnData(
        obs=pd.DataFrame(
            {"broad_cell_type": ["Neuron", "Astrocyte"], "fine_cell_type": ["L2/3 IT", "Protoplasmic"]},
            index=["SLC17A7", "AQP4"],
        ),
        X=np.zeros((2, 1), dtype=np.float32),
    )
    marker.write_zarr(path / "tables" / "markers_by_obs")


def test_marker_table_is_found_without_reading_whole_anndata_tables(tmp_path, monkeypatch):
    path = tmp_path / "store.zarr"
    _write_marker_tables(path)

    def forbidden(*_a, **_k):
        raise AssertionError("ad.read_zarr must not be used to look for marker tables")

    monkeypatch.setattr(ad, "read_zarr", forbidden)
    ref = load_cell_type_marker_reference(path)
    assert ref == {
        "SLC17A7": {"broad": "Neuron", "fine": "L2/3 IT"},
        "AQP4": {"broad": "Astrocyte", "fine": "Protoplasmic"},
    }


def test_marker_table_reads_uns_entry_only(tmp_path, monkeypatch):
    path = tmp_path / "store.zarr"
    zarr.open_group(path, mode="w", zarr_format=2).create_group("tables")
    table = ad.AnnData(X=np.zeros((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    table.uns["cell_type_marker_reference"] = {"genes": {"SST": {"broad": "Neuron", "fine": "Sst"}}}
    table.uns["unrelated"] = {"big": np.arange(10)}
    table.write_zarr(path / "tables" / "table")
    monkeypatch.setattr(ad, "read_zarr", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("read_zarr")))
    assert load_cell_type_marker_reference(path) == {"SST": {"broad": "Neuron", "fine": "Sst"}}


# -- model / view at Atera scale ------------------------------------------------------------


def test_atera_scale_panel_populates_and_filters_fast(qapp):
    genes, controls, layout, counts, visuals = _atera_scale_panel()
    w = _widget()
    w.resize(320, 700)
    t0 = time.perf_counter()
    w.populate("ATERA", layout, visuals, counts, set(controls), set(), {}, "coarse",
               False, False, False, 0.5, mode=P.GENE_PANEL_MODE_SELECTION)
    w.show()
    qapp.processEvents()
    populate_s = time.perf_counter() - t0
    assert populate_s < POPULATE_BUDGET_S, populate_s
    assert len(w.gene_names()) == len(genes) + len(controls)

    # Controls fold under one heading that stays visible with its count.
    assert w.is_gene_row_hidden(controls[0])
    assert not w.is_header_hidden(CONTROL_GROUP_TITLE)
    assert w.is_header_collapsed(CONTROL_GROUP_TITLE)
    assert not w.is_gene_row_hidden(genes[0])

    durations = []
    for text in ("G", "GE", "GENE1", "GENE17", "GENE1799", "GENE17999"):
        t0 = time.perf_counter()
        w._filter_edit.setText(text)
        w.flush_filter()  # the panel is above the debounce threshold
        qapp.processEvents()
        durations.append(time.perf_counter() - t0)
    assert max(durations) < KEYSTROKE_BUDGET_S, durations
    assert w.visible_gene_names() == ["GENE17999"]
    # Headers with no surviving gene disappear; the one holding the match stays.
    visible_headers = [t for t in w.header_titles() if not w.is_header_hidden(t)]
    assert len(visible_headers) == 1 and visible_headers[0].startswith("Group")
    assert w.is_header_hidden(CONTROL_GROUP_TITLE)

    # A pasted list matches names exactly (not as substrings).
    w._filter_edit.setText("GENE00001, gene00010  GENE00100;NOPE")
    w.flush_filter()
    assert w.visible_gene_names() == ["GENE00001", "GENE00010", "GENE00100"]

    # A control family search finds the folded heading but keeps its rows folded.
    w._filter_edit.setText("NegControlProbe")
    w.flush_filter()
    assert w.visible_gene_names() == []
    assert not w.is_header_hidden(CONTROL_GROUP_TITLE)
    w.click_header(CONTROL_GROUP_TITLE)  # unfolds (and ticks "show controls")
    assert not w.is_header_collapsed(CONTROL_GROUP_TITLE)
    assert w._show_controls_check.isChecked()
    shown = w.visible_gene_names()
    assert shown and all(name.startswith("NegControlProbe") for name in shown)
    w.close()


def test_typing_debounces_large_panels_but_not_small_ones(qapp):
    genes, controls, layout, counts, visuals = _atera_scale_panel(3000, 10)
    w = _widget()
    w.populate("ATERA", layout, visuals, counts, set(controls), set(), {}, "coarse",
               False, False, False, 0.5, mode=P.GENE_PANEL_MODE_SELECTION)
    w._filter_edit.setText("GENE00042")
    assert w._filter_timer.isActive()
    assert len(w.visible_gene_names()) == len(genes)  # not applied yet
    w.flush_filter()
    assert w.visible_gene_names() == ["GENE00042"]
    w.close()


# -- selection mode -----------------------------------------------------------------------


def _selection_widget(qapp, calls, n_genes=40, cap=4):
    genes = [f"G{i:02d}" for i in range(n_genes)]
    controls = ["NegControlProbe_00001", "UnassignedCodeword_0001"]
    layout = [("header", "Neuron", (0.2, 0.3, 0.9, 1.0))]
    layout += [("gene", g) for g in genes[: n_genes // 2]]
    layout += [("header", "Glia", (0.2, 0.8, 0.3, 1.0))]
    layout += [("gene", g) for g in genes[n_genes // 2:]]
    layout += [("header", CONTROL_GROUP_TITLE, (0.6, 0.6, 0.6, 1.0))]
    layout += [("gene", c) for c in controls]
    w = _widget(calls)
    w.populate("ATERA", layout, assign_gene_visuals(genes + controls),
               {g: 1 for g in genes + controls}, set(controls), set(), {}, "coarse",
               False, False, False, 0.5, mode=P.GENE_PANEL_MODE_SELECTION, selection_cap=cap)
    return w, genes, controls


def test_selection_mode_starts_empty_pins_selection_and_caps(qapp):
    calls = []
    w, genes, _controls = _selection_widget(qapp, calls)
    assert w.panel_mode == P.GENE_PANEL_MODE_SELECTION
    assert not any(w.is_gene_checked(g) for g in genes)
    assert w._show_all_check.text() == "All-transcript overview"
    assert not w._show_all_check.isTristate()
    assert w._selected_label.text() == "Selected (0/4)"
    assert w.selected_genes() == []

    for g in genes[:4]:
        assert w.set_gene_checked(g, True)
    assert w.selected_genes() == genes[:4]
    assert w._selected_label.text() == "Selected (4/4)"
    assert w._selected_model.rowCount() == 4
    # The fifth is refused with a status, and the callback never fires.
    n_calls = len(calls)
    assert not w.set_gene_checked(genes[4], True)
    assert not w.is_gene_checked(genes[4])
    assert len(calls) == n_calls
    assert "Selection is full" in w._status_label.text()

    # Unticking from the pinned list frees a place.
    first = w._selected_model.index(0, 0)
    assert w._selected_model.setData(first, V.Qt.Unchecked, V.Qt.CheckStateRole)
    assert not w.is_gene_checked(genes[0])
    assert ("vis", ("ATERA", genes[0], False)) in calls
    assert w.set_gene_checked(genes[4], True)
    assert w.selected_genes() == genes[1:5]
    w.close()


def test_selection_mode_header_above_cap_does_not_bulk_enable(qapp):
    calls = []
    w, genes, _controls = _selection_widget(qapp, calls, n_genes=40, cap=4)
    w.click_header("Neuron")  # 20 genes > cap 4
    assert not any(w.is_gene_checked(g) for g in genes)
    assert not [c for c in calls if c[0] == "genes"]
    assert "exceed" in w._status_label.text()

    # Narrowed by search to 3 genes, the same heading selects them all at once.
    w._filter_edit.setText("G0")  # G00..G09 under Neuron
    w._filter_edit.setText("G00 G01 G02")
    w.click_header("Neuron")
    assert calls[-1] == ("genes", "ATERA", ("G00", "G01", "G02"), True)
    assert w.selected_genes() == ["G00", "G01", "G02"]
    w.close()


def test_selection_mode_enter_selects_matches_and_show_all_is_the_overview_hook(qapp):
    calls = []
    w, genes, _controls = _selection_widget(qapp, calls, cap=4)
    w._filter_edit.setText("G1")  # G10..G19: ten matches > cap
    w._filter_edit.returnPressed.emit()
    assert w.selected_genes() == []
    w._filter_edit.setText("G15")
    w._filter_edit.returnPressed.emit()
    assert w.selected_genes() == ["G15"]

    w._show_all_check.click()
    assert ("overview", ("ATERA", True)) in calls
    assert not [c for c in calls if c[0] == "all"]
    assert not any(w.is_gene_checked(g) for g in genes if g != "G15")
    w._show_all_check.click()
    assert ("overview", ("ATERA", False)) in calls
    w.close()


def test_selection_mode_without_overview_hook_unticks_the_box(qapp):
    w, _genes, _controls = _selection_widget(qapp, [], cap=4)
    w._set_overview_callback = None
    w._show_all_check.click()
    assert w._show_all_check.checkState() == V.Qt.Unchecked
    assert "not available" in w._status_label.text()
    w.close()


def test_view_click_on_a_header_row_acts_like_click_header(qapp):
    calls = []
    w, _genes, _controls = _selection_widget(qapp, calls, n_genes=4, cap=8)
    proxy = w._proxy
    header_row = next(
        r for r in range(proxy.rowCount())
        if proxy.index(r, 0).data(V._GENE_IS_HEADER_ROLE)
        and proxy.index(r, 0).data(V._GENE_HEADER_TITLE_ROLE) == "Neuron"
    )
    w._gene_view.clicked.emit(proxy.index(header_row, 0))
    assert calls[-1] == ("genes", "ATERA", ("G00", "G01"), True)
    # Header text shows the group size in selection mode; the title accessor does not.
    assert proxy.index(header_row, 0).data(V.Qt.DisplayRole) == "Neuron  (2)"
    w.close()


# -- full mode regression ---------------------------------------------------------------------


def test_full_mode_500_gene_panel_with_blanks_is_unchanged(qapp):
    calls = []
    genes = [f"Gene{i:03d}" for i in range(500)]
    blanks = [f"Blank-{i}" for i in range(20)]
    w = _widget(calls)
    layout = [("gene", g) for g in sorted(genes + blanks)]
    w.populate("XENIUM", layout, assign_gene_visuals(genes + blanks),
               {g: 3 for g in genes + blanks}, set(blanks), set(genes), {}, "alphabetical",
               False, False, False, 1.0)
    assert w.panel_mode == P.GENE_PANEL_MODE_FULL
    assert w._show_all_check.text() == "Show all genes" and w._show_all_check.isTristate()
    assert not w._selected_view.isVisibleTo(w)
    assert w._show_all_check.checkState() == V.Qt.Checked
    assert all(w.is_gene_checked(g) for g in genes)
    assert all(w.is_gene_row_hidden(b) for b in blanks)
    assert w.header_titles() == []

    # No cap in full mode.
    assert w.set_gene_checked("Gene000", False) and w.set_gene_checked("Gene000", True)

    # Untick one -> partial; Show all -> everything eligible on, one callback.
    w.set_gene_checked("Gene007", False)
    assert w._show_all_check.checkState() == V.Qt.PartiallyChecked
    w._show_all_check.click()
    assert calls[-1] == ("all", ("XENIUM", True))
    assert all(w.is_gene_checked(g) for g in genes)
    assert not any(w.is_gene_checked(b) for b in blanks)
    w._show_all_check.click()
    assert calls[-1] == ("all", ("XENIUM", False))
    assert not any(w.is_gene_checked(g) for g in genes)

    # Controls appear (not folded) when asked for.
    w._show_controls_check.setChecked(True)
    assert not any(w.is_gene_row_hidden(b) for b in blanks)
    w._filter_edit.setText("gene04")
    assert w.visible_gene_names() == [f"Gene{i:03d}" for i in range(40, 50)]
    w.close()


def test_colour_refresh_and_visual_updates_do_not_reset_the_model(qapp):
    w, genes, _controls = _selection_widget(qapp, [], cap=4)
    resets = []
    w._proxy.modelReset.connect(lambda: resets.append(1))
    changed = []
    w._proxy.dataChanged.connect(lambda a, b, roles=None: changed.append(list(roles or [])))
    w.set_gene_visuals({genes[0]: V.GeneVisual(rgba=(1.0, 0.0, 0.0, 1.0), symbol="disc")})
    w.refresh_decorations()
    assert resets == []
    assert changed and all(int(V.Qt.DecorationRole) in roles for roles in changed)
    w.close()


# -- controller selection mode (eager point store) ---------------------------------------------


class _FakeLayer:
    def __init__(self, name, data, **kw):
        self.name = name
        self.visible = kw.get("visible", True)
        self.size = kw.get("size", 1.0)
        self.face_color = kw.get("face_color")
        self.border_color = kw.get("border_color")
        self.symbol = kw.get("symbol") or "disc"
        self.antialiasing = 1
        self.border_width = 0.05
        self._data = np.asarray(data)
        self.shown = np.asarray(kw.get("shown", np.ones(len(self._data), dtype=bool)), dtype=bool)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)


class _FakeViewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []

    def add_points(self, coords, name=None, **kw):
        layer = _FakeLayer(name, coords, **kw)
        self.layers.append(layer)
        return layer


def _selection_controller(monkeypatch, qapp, widget=None):
    monkeypatch.setattr(P, "GENE_SELECTION_MODE_FEATURE_THRESHOLD", 2)
    rows = []
    for gene, n in (("AAA", 4), ("BBB", 3), ("CCC", 2), ("Blank-1", 2)):
        rows.extend((gene, float(i), float(i), i == n - 1) for i in range(n))
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    store = build_gene_point_groups(df, x_col="x", y_col="y", gene_col="gene", background_col="background")
    args = types.SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000,
        gene_hide_background=False, gene_show_controls=False, random_state=42,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = widget
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation, "TEST",
        {"points_key": "transcripts", "store": store, "build_seconds": 0.0},
    )
    return ctrl, ctrl._gene_inspector_states["TEST"]


def _shown_total(ctrl, state):
    return sum(int(np.count_nonzero(ctrl._get_layer_by_name(n).shown)) for n in state.layer_names)


def _face(ctrl, state, gene):
    gi, fg_start, _fg_end, _bg_end = state.store.gene_offsets[gene]
    layer = ctrl._get_layer_by_name(state.layer_names[gi])
    for start, _end, name in state.group_display_ranges[gi]:
        if name == gene:
            return np.asarray(layer.face_color[start])
    raise AssertionError(gene)


def test_controller_selection_mode_starts_empty_and_paints_slot_colours(monkeypatch, qapp):
    ctrl, state = _selection_controller(monkeypatch, qapp)
    assert state.panel_mode == P.GENE_PANEL_MODE_SELECTION
    assert state.enabled_genes == set()
    assert _shown_total(ctrl, state) == 0

    ctrl.set_gene_visible("TEST", "BBB", True)
    ctrl._flush_gene_group_rebuild("TEST")
    assert _shown_total(ctrl, state) == 3
    slot0 = np.asarray(state.slot_allocator.visual("BBB").rgba, dtype=np.float32)
    np.testing.assert_allclose(_face(ctrl, state, "BBB"), slot0, atol=1e-6)

    ctrl.set_gene_visible("TEST", "AAA", True)
    ctrl._flush_gene_group_rebuild("TEST")
    assert state.slot_allocator.slot_of("AAA") == 1
    assert not np.allclose(_face(ctrl, state, "AAA"), slot0)

    # Freeing BBB restores its scheme colour, and the next gene reuses slot 0.
    ctrl.set_gene_visible("TEST", "BBB", False)
    ctrl._flush_gene_group_rebuild("TEST")
    np.testing.assert_allclose(
        _face(ctrl, state, "BBB"), np.asarray(state.gene_visuals["BBB"].rgba, dtype=np.float32), atol=1e-6
    )
    ctrl.set_gene_visible("TEST", "CCC", True)
    assert state.slot_allocator.slot_of("CCC") == 0


def test_controller_selection_cap_overview_and_layout(monkeypatch, qapp):
    w = _widget()
    ctrl, state = _selection_controller(monkeypatch, qapp, widget=w)
    state.slot_allocator = GeneSlotAllocator(cap=1)
    statuses = []
    ctrl.set_status_callback(statuses.append)

    # The widget opened in selection mode with the controls folded at the end.
    assert w.panel_mode == P.GENE_PANEL_MODE_SELECTION
    assert w.header_titles()[-1] == CONTROL_GROUP_TITLE
    assert w.is_header_collapsed(CONTROL_GROUP_TITLE)

    ctrl.set_gene_visible("TEST", "AAA", True)
    ctrl.set_gene_visible("TEST", "BBB", True)  # over the cap: refused
    assert state.enabled_genes == {"AAA"}
    assert "Selection is full" in statuses[-1]
    ctrl.set_genes_visible("TEST", ["BBB", "CCC"], True)  # all or nothing
    assert state.enabled_genes == {"AAA"}

    # The overview shows every real gene but never the controls.
    ctrl.set_gene_overview("TEST", True)
    assert _shown_total(ctrl, state) == 4 + 3 + 2
    assert w._show_all_check.checkState() == V.Qt.Checked
    ctrl.set_gene_overview("TEST", False)
    ctrl._flush_gene_group_rebuild("TEST")
    assert _shown_total(ctrl, state) == 4

    # "Show all" in selection mode routes to the overview instead of ticking all.
    ctrl.set_all_genes_visible("TEST", True)
    assert state.overview_visible and state.enabled_genes == {"AAA"}
    ctrl.set_all_genes_visible("TEST", False)
    assert state.enabled_genes == set() and len(state.slot_allocator) == 0
    w.close()


def test_controller_full_mode_is_untouched_by_selection_wiring(qapp):
    rows = [("AAA", 0.0, 0.0, False), ("BBB", 1.0, 1.0, False), ("Blank-1", 2.0, 2.0, False)]
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    store = build_gene_point_groups(df, x_col="x", y_col="y", gene_col="gene", background_col="background")
    args = types.SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000,
        gene_hide_background=False, gene_show_controls=False, random_state=42,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = None
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation, "TEST",
        {"points_key": "transcripts", "store": store, "build_seconds": 0.0},
    )
    state = ctrl._gene_inspector_states["TEST"]
    assert state.panel_mode == P.GENE_PANEL_MODE_FULL
    assert state.slot_allocator is None
    assert state.enabled_genes == {"AAA", "BBB"}


# -- paired selection mode (union panel above the threshold) -----------------------------------


def _paired_selection_controller(monkeypatch, widget=None, threshold=4):
    """Two platform states built and harmonised the way ``_apply_paired_gene_build`` does."""
    # Each local panel (3 features) is below the threshold; the union (5) is above it,
    # so selection mode can only come from the union decision.
    monkeypatch.setattr(P, "GENE_SELECTION_MODE_FEATURE_THRESHOLD", threshold)
    panels = {
        "MERSCOPE": (("AAA", 4), ("BBB", 3), ("Blank-1", 2)),
        "XENIUM": (("BBB", 2), ("CCC", 3), ("DDD", 2)),
    }
    args = types.SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000,
        gene_hide_background=False, gene_show_controls=False, random_state=42,
    )
    ctrl = V.ComparisonViewerController(_FakeViewer(), {}, args)
    ctrl._gene_inspector_widget = widget
    states = []
    for platform, panel in panels.items():
        rows = []
        for gene, n in panel:
            rows.extend((gene, float(i), float(i), i == n - 1) for i in range(n))
        df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
        store = build_gene_point_groups(df, x_col="x", y_col="y", gene_col="gene", background_col="background")
        state = ctrl._new_gene_inspector_state(platform, {"points_key": "transcripts", "store": store})
        ctrl._gene_inspector_states[platform] = state
        states.append(state)
    ctrl._harmonize_paired_gene_states(states)
    for state in states:
        for gi, symbol in enumerate(state.store.group_symbols):
            coords, colors, _sizes, ranges, shown = ctrl._gene_group_arrays(state, gi)
            state.group_display_ranges[gi] = ranges
            layer = ctrl.viewer.add_points(
                coords, name=f"{state.dataset} {symbol}", face_color=colors, shown=shown, symbol=symbol,
            )
            state.layer_names.append(layer.name)
    # Only its presence matters to the paired control routing.
    ctrl._paired_session = types.SimpleNamespace()
    return ctrl, states


def test_paired_union_above_threshold_opens_in_selection_mode_with_lockstep_slots(monkeypatch, qapp):
    w = _widget()
    ctrl, (merscope, xenium) = _paired_selection_controller(monkeypatch, widget=w)
    monkeypatch.setattr(ctrl, "_publish_transcript_sources", lambda *_a, **_k: None)
    statuses = []
    ctrl.set_status_callback(statuses.append)

    for state in (merscope, xenium):
        assert state.panel_mode == P.GENE_PANEL_MODE_SELECTION
        assert state.enabled_genes == set()
        assert state.slot_allocator is not None
        assert _shown_total(ctrl, state) == 0
    assert merscope.slot_allocator is not xenium.slot_allocator

    ctrl._populate_gene_inspector(merscope)
    assert w.panel_mode == P.GENE_PANEL_MODE_SELECTION
    assert w.header_titles()[-1] == CONTROL_GROUP_TITLE

    # A platform-only gene still takes the same slot on both platforms.
    ctrl.set_gene_visible("PAIRED", "CCC", True)
    ctrl.set_gene_visible("PAIRED", "BBB", True)
    for state in (merscope, xenium):
        ctrl._flush_gene_group_rebuild(state.dataset)
        assert state.enabled_genes == {"CCC", "BBB"}
        assert state.slot_allocator.slot_of("CCC") == 0
        assert state.slot_allocator.slot_of("BBB") == 1
    np.testing.assert_allclose(_face(ctrl, merscope, "BBB"), _face(ctrl, xenium, "BBB"), atol=1e-6)
    np.testing.assert_allclose(
        _face(ctrl, xenium, "BBB"),
        np.asarray(xenium.slot_allocator.visual("BBB").rgba, dtype=np.float32),
        atol=1e-6,
    )
    assert _shown_total(ctrl, merscope) == 3
    assert _shown_total(ctrl, xenium) == 2 + 3

    # Freeing a slot keeps both allocators in step; the next gene reuses it on both.
    ctrl.set_gene_visible("PAIRED", "CCC", False)
    ctrl.set_gene_visible("PAIRED", "AAA", True)
    assert merscope.slot_allocator.slot_of("AAA") == xenium.slot_allocator.slot_of("AAA") == 0

    # The cap is refused once for the pair, and neither platform changes.
    for state in (merscope, xenium):
        state.slot_allocator.cap = 2
    before = len(statuses)
    ctrl.set_gene_visible("PAIRED", "DDD", True)
    refusals = [s for s in statuses[before:] if "Selection is full" in s]
    assert len(refusals) == 1
    before = len(statuses)
    ctrl.set_genes_visible("PAIRED", ["CCC", "DDD"], True)
    assert len([s for s in statuses[before:] if "Selection is full" in s]) == 1
    for state in (merscope, xenium):
        assert state.enabled_genes == {"AAA", "BBB"}
        assert state.slot_allocator.genes == ["AAA", "BBB"]

    # Clearing releases every slot on both platforms and restores scheme colours.
    ctrl.set_all_genes_visible("PAIRED", False)
    for state in (merscope, xenium):
        assert state.enabled_genes == set() and len(state.slot_allocator) == 0
        np.testing.assert_allclose(
            _face(ctrl, state, "BBB"), np.asarray(state.gene_visuals["BBB"].rgba, dtype=np.float32), atol=1e-6
        )
    w.close()


def test_paired_union_at_or_below_threshold_stays_in_full_mode(monkeypatch, qapp):
    ctrl, (merscope, xenium) = _paired_selection_controller(monkeypatch, threshold=5)
    for state in (merscope, xenium):
        assert state.panel_mode == P.GENE_PANEL_MODE_FULL
        assert state.slot_allocator is None
        assert state.enabled_genes == {"AAA", "BBB", "CCC", "DDD"}
