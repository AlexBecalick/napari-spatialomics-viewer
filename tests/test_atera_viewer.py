"""ATERA viewer wiring (milestone M7): full/lazy transcript modes, budgets, viewport.

A fake viewer stands in for napari (vispy canvases cannot be built offscreen);
reads run synchronously (``thread_worker`` is None) on the synthetic store.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atera_stores
import numpy as np
import pytest

from napari_compare_xenium_merscope import atera_transcripts as T
from napari_compare_xenium_merscope import atera_viewer as AV
from napari_compare_xenium_merscope import transcript_cache as TC
from napari_compare_xenium_merscope import viewer as V


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return atera_stores.build_atera_latest_store(tmp_path_factory.mktemp("atera_m7"), pixel_size=0.5)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR", str(tmp_path / "transcript-cache"))
    monkeypatch.setattr(V, "thread_worker", None)
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


class _Emitter:
    def __init__(self):
        self.callbacks = []

    def connect(self, cb):
        self.callbacks.append(cb)

    def disconnect(self, cb):
        self.callbacks.remove(cb)

    def __call__(self):
        for cb in list(self.callbacks):
            cb(None)


class _Layer:
    def __init__(self, name, data, kind="points", **kw):
        self.name = name
        self.kind = kind
        self.visible = kw.get("visible", True)
        self.size = kw.get("size", 1.0)
        self.face_color = kw.get("face_color")
        self.border_color = kw.get("border_color")
        self.symbol = kw.get("symbol", "disc")
        self._data = np.asarray(data)
        self.shown = np.asarray(kw.get("shown", np.ones(len(self._data), dtype=bool)), dtype=bool)
        self.pick = None
        self.kwargs = kw
        self.data_sets = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)
        self.data_sets += 1
        # napari resets per-point symbol/colours on a resize
        self.symbol = "o"
        self.face_color = None
        self.border_color = None

    def get_value(self, _position, **_kw):
        return self.pick


class _Viewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []
        self.grid = SimpleNamespace(enabled=False)
        self.camera = SimpleNamespace(center=(20.0, 20.0), zoom=1.0, events=SimpleNamespace(center=_Emitter(), zoom=_Emitter()))
        self._canvas_size = (100, 100)

    def add_points(self, data, name=None, **kw):
        layer = _Layer(name, data, **kw)
        self.layers.append(layer)
        return layer

    def add_image(self, data, name=None, **kw):
        layer = _Layer(name, np.zeros((0, 2)), kind="image", **kw)
        layer.image = data
        self.layers.append(layer)
        return layer

    def reset_view(self):
        pass


def _args(**overrides):
    defaults = dict(
        background_io_workers=1, session_cache_gb=0, image_pyramid_downsample=4,
        overwrite_derived_caches=False, skip_images=True, skip_transcripts=False,
        segmentation_source="shapes", hide_images=False, image_opacity=1.0,
        merscope_transform_path=None, xenium_spec_path=None, atera_spec_path=None,
        gene_spot_size=0.5, gene_max_render_points=None, gene_hide_background=False,
        gene_show_controls=False, random_state=42, atera_transcript_mode="full",
        atera_viewport_max_points=V.DEFAULT_ATERA_VIEWPORT_MAX_POINTS,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _controller(store: Path, monkeypatch, **overrides):
    ctrl = V.ComparisonViewerController(_Viewer(), {}, _args(**overrides))
    ctrl.datasets = {"ATERA": ctrl._build_dataset_config("ATERA", store)}
    ctrl.active_dataset = "ATERA"
    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", lambda _ds: True)
    ctrl._gene_inspector_widget = None
    statuses: list[str] = []
    ctrl.set_status_callback(statuses.append)
    ctrl.statuses = statuses
    return ctrl


def _total_rows(store: Path) -> int:
    return T.AteraTranscriptEngine.open(store).index.n_rows


def _state(ctrl):
    return ctrl._gene_inspector_states["ATERA"]


def _layer(ctrl, name):
    return ctrl._get_layer_by_name(name)


def _real_genes(state):
    return [g for g in state.store.genes if g not in state.store.control_genes]


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------


def test_full_open_builds_a_gene_point_store_from_the_hash_sample(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    assert ctx.mode == "full" and ctx.budget.source == "requested"
    assert state.store.sampled and 0 < state.store.total_points < total
    assert state.store.source_total_points == total
    assert state.panel_mode == V.GENE_PANEL_MODE_SELECTION
    # The control set follows is_gene, as the overview computed it.
    assert state.store.control_genes == ctx.overview.control_names() & set(state.store.genes)
    assert "transcripts loaded" in ctrl.statuses[-1]
    # Layers exist under the Genes prefix; the viewport layer tops up the sample.
    assert all(name.startswith("Genes | ") for name in state.layer_names)
    assert _layer(ctrl, AV.ATERA_VIEWPORT_LAYER) is not None
    assert _layer(ctrl, AV.ATERA_DENSITY_LAYER) is None


def test_budget_covering_the_store_loads_every_transcript_without_a_viewport(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=0)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    assert state.store.total_points == total and not state.store.sampled
    assert state.atera.complete and not state.atera.viewport_enabled
    assert _layer(ctrl, AV.ATERA_VIEWPORT_LAYER) is None


def test_second_open_restores_from_the_external_cache(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    ctrl.open_gene_inspector("ATERA")
    first = _state(ctrl).store.total_points

    def forbidden(*_a, **_k):
        raise AssertionError("a cached open must not stream the store")

    monkeypatch.setattr(T.AteraTranscriptEngine, "build_overview", forbidden)
    monkeypatch.setattr(T.AteraTranscriptEngine, "sample_rows", forbidden)
    ctrl2 = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    ctrl2.open_gene_inspector("ATERA")
    assert _state(ctrl2).store.total_points == first
    assert "persistent cache" in ctrl2.statuses[-1]
    # A smaller budget is served by filtering the cached (nested) sample.
    ctrl2.reload_atera_transcripts("ATERA", "full", total // 4)
    smaller = _state(ctrl2).store.total_points
    assert smaller < first


def test_auto_budget_and_memory_guard(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    gib = 1024**3
    budget = AV.resolve_atera_budget(10**10, None, ram_bytes=64 * gib)
    assert budget.source == "auto" and not budget.clamped
    assert budget.budget == int(0.5 * 64 * gib // AV.ATERA_BYTES_PER_POINT)
    assert "AUTO budget" in budget.note and "64 GB RAM" in budget.note
    every = AV.resolve_atera_budget(10**10, 0, ram_bytes=64 * gib)
    assert every.clamped and every.budget == int(0.85 * 64 * gib // AV.ATERA_BYTES_PER_POINT)
    assert "clamped" in every.note
    # In the controller: a tiny machine clamps the requested budget and says so.
    monkeypatch.setattr(AV, "physical_memory_bytes", lambda: int(AV.ATERA_BYTES_PER_POINT * total * 0.3 / 0.85))
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=0)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    assert state.atera.budget.clamped
    assert state.store.total_points < total
    assert "clamped" in ctrl.statuses[-1]


def test_memory_error_halves_the_budget(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    monkeypatch.setattr(AV, "ATERA_MIN_BUDGET", 1)
    original = AV.gene_point_store_from_rows
    calls = []

    def flaky(*args, **kwargs):
        calls.append(len(args[1]))
        if len(calls) == 1:
            raise MemoryError("synthetic")
        return original(*args, **kwargs)

    monkeypatch.setattr(AV, "gene_point_store_from_rows", flaky)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    assert len(calls) == 2 and calls[1] <= calls[0]
    assert state.atera.budget.budget == (total // 2) // 2
    assert "Out of memory" in ctrl.statuses[-1]


def test_reload_button_triggers_exactly_one_reload(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    widget = V.GeneInspectorWidget(
        close_callback=lambda *_a: None,
        set_gene_visible_callback=ctrl.set_gene_visible,
        set_all_genes_callback=ctrl.set_all_genes_visible,
        set_spot_size_callback=ctrl.set_gene_spot_size,
        set_hide_background_callback=ctrl.set_gene_hide_background,
        set_show_controls_callback=ctrl.set_gene_show_controls,
        reload_atera_callback=ctrl.reload_atera_transcripts,
        set_atera_viewport_cap_callback=ctrl.set_atera_viewport_cap,
    )
    ctrl._gene_inspector_widget = widget
    ctrl.open_gene_inspector("ATERA")
    assert not widget._atera_box.isHidden()
    opens = []
    original = ctrl._open_atera_gene_inspector
    monkeypatch.setattr(ctrl, "_open_atera_gene_inspector", lambda ds, **kw: opens.append(ds) or original(ds, **kw))
    widget._atera_budget_spin.setValue(0.3)  # editing alone never reloads
    widget._atera_budget_spin.setValue(0.1)
    assert opens == []
    widget._atera_reload_button.click()
    assert opens == ["ATERA"]
    assert ctrl._atera_transcript_options["ATERA"]["budget"] == 100_000
    # Auto is the spin box's special value.
    widget._atera_budget_spin.setValue(0.0)
    assert widget.atera_requested_budget() is None
    # The viewport cap applies at once, without a reload.
    widget._atera_viewport_spin.setValue(3.0)
    assert opens == ["ATERA"]
    assert _state(ctrl).atera.viewport_cap == 3_000_000
    widget.clear()
    assert widget._atera_box.isHidden()


# ---------------------------------------------------------------------------
# Viewport
# ---------------------------------------------------------------------------


def test_viewport_tops_up_with_the_rows_the_sample_left_out(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None  # run requests synchronously
    genes = _real_genes(state)[:3]
    for gene in genes:
        ctrl.set_gene_visible("ATERA", gene, True)
    viewport = _layer(ctrl, AV.ATERA_VIEWPORT_LAYER)
    codes = [ctx.overview.vocabulary.index(g) for g in genes]
    everything = ctx.engine.read_window_rows((-1e9, -1e9, 1e9, 1e9), codes)
    sampled = sum(state.store.gene_counts.get(g, 0) for g in genes)
    # Whole slide in view (zoom 1 over a 100 px canvas around (20, 20) covers it).
    assert len(viewport.data) + sampled == len(everything["x"])
    assert len(viewport.data) > 0 and sampled > 0
    assert viewport.symbol == "disc" and viewport.face_color is not None
    assert len(viewport.face_color) == len(viewport.data)
    layer_before = viewport
    # Pans inside the padded window reuse the cached rows and the same layer.
    reads = []
    original = ctx.engine.read_window_rows
    monkeypatch.setattr(ctx.engine, "read_window_rows", lambda *a, **k: reads.append(a) or original(*a, **k))
    ctrl.viewer.camera.center = (21.0, 21.0)
    ctrl.viewer.camera.events.center()
    assert reads == []
    assert _layer(ctrl, AV.ATERA_VIEWPORT_LAYER) is layer_before
    # Click a viewport point: its gene is highlighted.
    viewport.pick = 0
    picked = ctrl._pick_gene_from_event(state, SimpleNamespace(position=(0, 0)))
    assert picked == str(ctx.overview.vocabulary[int(ctx.viewport_codes[0])]) and picked in genes
    # Hide-assigned and colour-by-assignment reach the viewport rows.
    n_all = len(viewport.data)
    ctrl.set_gene_hide_assigned("ATERA", True)
    rows = ctx.viewport_cache["rows"]
    keep = np.isin(rows["code"], codes)
    assert len(viewport.data) == int((keep & (rows["cell_id"] == 0)).sum()) <= n_all
    ctrl.set_gene_hide_assigned("ATERA", False)
    ctrl.set_gene_colour_by_assignment("ATERA", True)
    assigned = rows["cell_id"][keep] != 0
    np.testing.assert_array_equal(
        np.asarray(viewport.face_color)[assigned][:, :3],
        np.broadcast_to(V.GENE_ASSIGNED_RGBA[:3], (int(assigned.sum()), 3)),
    )
    # Spot size reaches the viewport layer; unticking empties it.
    ctrl.set_gene_spot_size("ATERA", 1.5)
    assert float(viewport.size) == 1.5
    for gene in genes:
        ctrl.set_gene_visible("ATERA", gene, False)
    assert len(viewport.data) == 0


def test_viewport_waits_for_zoom_when_the_window_is_too_dense(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    ctx.viewport_cap = 1
    reads = []
    monkeypatch.setattr(ctx.engine, "read_window_rows", lambda *a, **k: reads.append(a))
    ctrl.set_gene_overview("ATERA", True)
    assert reads == []
    assert "zoom in" in ctrl.statuses[-1]


def test_scheduler_is_single_flight_and_latest_wins():
    started = []
    runner = AV.LatestWinsRunner(lambda request, gen, cancel: started.append((request, gen, cancel)))
    for i in range(20):
        runner.submit(i)
    assert len(started) == 1 and runner.pending is not None and runner.pending[0] == 19
    first_request, first_gen, first_cancel = started[0]
    assert first_cancel.is_set()  # superseded while running
    assert runner.finish(first_gen) is False  # stale result is dropped
    assert len(started) == 2 and started[1][0] == 19 and runner.pending is None
    assert runner.finish(started[1][1]) is True
    assert not runner.busy
    runner.submit("x")
    runner.cancel()
    assert runner.finish(started[-1][1]) is False


def test_burst_of_camera_events_gives_one_read_in_flight(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    started = []
    ctx.view_runner = AV.LatestWinsRunner(lambda request, gen, cancel: started.append(gen))
    ctrl.set_gene_visible("ATERA", _real_genes(state)[0], True)
    for i in range(20):
        ctrl.viewer.camera.center = (20.0 + 100 * i, 20.0)
        ctrl.viewer.camera.events.center()
    assert len(started) == 1
    assert ctx.view_runner.pending is not None
    # A view the latest request already covers adds nothing while it runs.
    pending = ctx.view_runner.pending
    ctrl.viewer.camera.events.zoom()
    assert ctx.view_runner.pending is pending and len(started) == 1
    # A dataset switch cancels both lazy kinds.
    ctrl._cancel_dataset_background_tasks()
    assert ctx.view_runner.pending is None
    assert ctx.view_runner.finish(started[0]) is False


def test_lazy_reads_do_not_wait_for_the_store_io_slots(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    state.atera.view_timer = None
    acquired = 0
    while ctrl._store_io_slots.acquire(blocking=False):
        acquired += 1
    try:
        ctrl.set_gene_visible("ATERA", _real_genes(state)[0], True)
        assert state.atera.viewport_cache is not None
    finally:
        for _ in range(acquired):
            ctrl._store_io_slots.release()


# ---------------------------------------------------------------------------
# Lazy mode
# ---------------------------------------------------------------------------


def test_lazy_mode_density_overview_and_batched_gene_reads(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    assert ctx.mode == "lazy" and state.store.total_points == 0
    density = _layer(ctrl, AV.ATERA_DENSITY_LAYER)
    assert density is not None and density.kind == "image"
    assert density.image.shape == ctx.overview.density_counts.shape
    # The panel lists every observed feature with exact counts.
    genes, _lut, counts, _controls = T.overview_panel(ctx.overview)
    assert state.store.genes == genes and state.store.source_gene_counts == counts
    reads = []
    original = ctx.engine.read_genes
    monkeypatch.setattr(ctx.engine, "read_genes", lambda g, **k: reads.append(list(g)) or original(g, **k))
    ctx.view_timer = None
    wanted = _real_genes(state)[:3]
    for gene in wanted:  # inside one debounce window (the QTimer never fires here)
        ctrl.set_gene_visible("ATERA", gene, True)
    assert reads == []
    ctrl._atera_flush_gene_reads("ATERA")
    assert reads == [sorted(wanted)]
    assert set(ctx.loaded) == set(wanted)
    for gene in wanted:
        assert state.store.gene_counts[gene] == counts[gene]
    shown = sum(int(np.count_nonzero(_layer(ctrl, n).shown)) for n in state.layer_names)
    assert shown == sum(counts[g] for g in wanted)
    # Each new group layer starts with its symbol and colours (no resize reset).
    for gi, name in enumerate(state.layer_names):
        layer = _layer(ctrl, name)
        assert layer.symbol == state.store.group_symbols[gi]
        assert layer.face_color is not None and len(layer.face_color) == len(layer.data)
    # Selection slots paint the loaded points; unticking frees the slot.
    gene = wanted[0]
    assert gene in state.slot_allocator
    ctrl.set_gene_visible("ATERA", gene, False)
    ctrl._flush_gene_group_rebuild("ATERA")
    assert gene not in state.slot_allocator and gene not in state.enabled_genes
    shown = sum(int(np.count_nonzero(_layer(ctrl, n).shown)) for n in state.layer_names)
    assert shown == sum(counts[g] for g in wanted[1:])
    # The debounced flush removes the unticked gene's points from the layers
    # (never kept hidden); its rows stay only in the LRU.
    ctrl._atera_flush_gene_reads("ATERA")
    assert len(reads) == 1
    assert gene in ctx.loaded and gene not in ctx.store_genes
    assert state.store.gene_offsets.get(gene) is None
    in_layers = sum(len(_layer(ctrl, n).data) for n in state.layer_names)
    assert in_layers == state.store.total_points == sum(counts[g] for g in wanted[1:])
    # A re-tick of a loaded gene reads nothing and puts its points back.
    ctrl.set_gene_visible("ATERA", gene, True)
    ctrl._atera_flush_gene_reads("ATERA")
    assert len(reads) == 1
    assert gene in ctx.store_genes
    in_layers = sum(len(_layer(ctrl, n).data) for n in state.layer_names)
    assert in_layers == sum(counts[g] for g in wanted)
    # The lazy viewport shows the other genes under the overview, never the
    # ticked or loaded ones (they have their own full-resolution layers).
    ctrl.set_gene_overview("ATERA", True)
    assert ctx.viewport_cache is not None and len(ctx.viewport_codes) > 0
    assert not np.isin(ctx.viewport_codes, [ctx.overview.vocabulary.index(g) for g in wanted]).any()
    assert not np.isin(ctx.viewport_codes, [ctx.overview.vocabulary.index(g) for g in state.store.control_genes]).any()


def test_lazy_gene_cache_is_an_lru_that_never_evicts_ticked_genes():
    def rows(n):
        return {"x": np.zeros(n, np.float32), "y": np.zeros(n, np.float32)}

    loaded = {"A": rows(10), "B": rows(10), "C": rows(10)}  # 80 B each
    # A re-tick of A makes it the most recent; a new gene D then evicts B, the
    # oldest unticked gene, and never the ticked ones.
    merged = AV.lru_gene_cache(loaded, {"D": rows(10)}, {"A", "D"}, 3 * 80)
    assert list(merged) == ["C", "A", "D"]
    merged = AV.lru_gene_cache(merged, {}, {"A", "D"}, 80)
    assert list(merged) == ["A", "D"]  # over budget, but ticked genes stay


def test_teardown_drops_the_lazy_gene_rows(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    ctrl.set_gene_visible("ATERA", _real_genes(state)[0], True)
    ctrl._atera_flush_gene_reads("ATERA")
    assert ctx.loaded
    session = ctrl._dataset_sessions.get("ATERA")
    if session is not None and session.gene_payload is not None:
        assert session.gene_payload["atera"] is ctx
    ctrl._teardown_gene_inspector("ATERA")
    # Nothing references the per-gene rows from a retained session any more.
    assert ctx.loaded == {} and ctx.store_genes == frozenset()
    assert ctx.viewport_cache is None and len(ctx.viewport_codes) == 0


def test_memory_error_creating_the_layers_reloads_at_half_the_budget(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    monkeypatch.setattr(AV, "ATERA_MIN_BUDGET", 1)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    original = ctrl._create_gene_points_layer
    calls = []

    def flaky(name, *args, **kwargs):
        if name != AV.ATERA_VIEWPORT_LAYER:
            calls.append(name)
            if len(calls) == 1:
                raise MemoryError("synthetic")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ctrl, "_create_gene_points_layer", flaky)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    assert state.atera.budget.budget == (total // 2) // 2
    assert state.atera.requested_budget == (total // 2) // 2
    assert "Out of memory creating the layers" in ctrl.statuses[-1]
    # The partial layers were torn down: one layer per group, none left over.
    names = [str(layer.name) for layer in ctrl.viewer.layers]
    assert sorted(n for n in names if n in state.layer_names) == sorted(state.layer_names)
    assert len(names) == len(set(names))
    assert all(_layer(ctrl, n) is not None for n in state.layer_names)


def test_memory_error_creating_lazy_layers_is_reported(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")

    def boom(*_a, **_k):
        raise MemoryError("synthetic")

    monkeypatch.setattr(ctrl, "_atera_add_density_layer", boom)
    ctrl.open_gene_inspector("ATERA")
    assert "ATERA" not in ctrl._gene_inspector_states
    assert "out of memory" in ctrl.statuses[-1]


def test_teardown_removes_atera_layers_and_stops_reads(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    ctx = _state(ctrl).atera
    assert ctrl.viewer.camera.events.center.callbacks
    ctrl._teardown_gene_inspector("ATERA")
    assert _layer(ctrl, AV.ATERA_DENSITY_LAYER) is None
    assert _layer(ctrl, AV.ATERA_VIEWPORT_LAYER) is None
    assert not ctrl.viewer.camera.events.center.callbacks
    assert ctx.camera_callback is None


def test_helpers_bounds_estimate_and_cap():
    assert AV.visible_bounds((10.0, 20.0), 2.0, (100, 200)) == (-30.0, -15.0, 70.0, 35.0)
    assert AV.visible_bounds((10.0, 20.0), 0.0, (100, 200)) is None
    assert AV.pad_bounds((0, 0, 10, 20), 0.5) == (-5.0, -10.0, 15.0, 30.0)
    assert AV.bounds_contain((0, 0, 10, 10), (1, 1, 9, 9))
    assert not AV.bounds_contain((0, 0, 10, 10), (1, 1, 11, 9))
    rows = {"x": np.arange(10.0), "key": np.array([9, 1, 8, 2, 7, 3, 6, 4, 5, 0], dtype=np.uint32)}
    capped = AV.cap_rows_by_key(rows, 3)
    np.testing.assert_array_equal(capped["x"], [1.0, 3.0, 9.0])


def test_aggregate_row_and_layer_order_include_the_dynamic_layers(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    state.atera.view_timer = None
    ctrl.set_gene_visible("ATERA", _real_genes(state)[0], True)
    ctrl._atera_flush_gene_reads("ATERA")
    ctrl.set_gene_overview("ATERA", True)
    row = V.GeneLayerAggregateRow(ctrl.viewer)
    names = {layer.name for layer in row.gene_layers()}
    assert set(state.layer_names) <= names
    assert {AV.ATERA_DENSITY_LAYER, AV.ATERA_VIEWPORT_LAYER} <= names
    assert state.layer_names  # the per-gene groups were created


# ---------------------------------------------------------------------------
# Review follow-ups: runner identity, lazy MemoryError, density visibility,
# is_gene-driven controls on a real open
# ---------------------------------------------------------------------------


def test_lazy_memory_error_drawing_genes_never_wedges_the_gene_reads(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    ctx.gene_timer = None  # flush synchronously
    original = ctrl._create_gene_points_layer
    calls = []

    def flaky(name, *args, **kwargs):
        if name != AV.ATERA_VIEWPORT_LAYER:
            calls.append(name)
            if len(calls) == 1:
                raise MemoryError("synthetic")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ctrl, "_create_gene_points_layer", flaky)
    first, second = _real_genes(state)[:2]
    ctrl.set_gene_visible("ATERA", first, True)
    # The runner finished, the gene that did not fit was unticked (its rows
    # stay in the LRU), no partial layer is left and the status says why.
    assert not ctx.gene_runner.busy
    assert first not in state.enabled_genes and first not in state.slot_allocator
    assert first in ctx.loaded and ctx.store_genes == frozenset()
    assert state.store.total_points == 0 and state.layer_names == []
    fixed = {AV.ATERA_DENSITY_LAYER, AV.ATERA_VIEWPORT_LAYER}
    assert not any(
        str(layer.name).startswith("Genes | ") and str(layer.name) not in fixed for layer in ctrl.viewer.layers
    )
    assert "out of memory" in ctrl.statuses[-1] and "Tick fewer genes" in ctrl.statuses[-1]
    # A later tick still loads.
    ctrl.set_gene_visible("ATERA", second, True)
    assert not ctx.gene_runner.busy
    assert ctx.store_genes == frozenset({second})
    assert state.store.gene_counts[second] == state.store.source_gene_counts[second]
    assert all(_layer(ctrl, n) is not None for n in state.layer_names)


def test_restored_payload_ignores_the_torn_down_installs_reads(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl._dataset_sessions["ATERA"] = SimpleNamespace(gene_payload=None, last_used=0)
    monkeypatch.setattr(ctrl, "_evict_dataset_sessions", lambda *a, **k: None)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    jobs = []
    monkeypatch.setattr(ctrl, "_atera_run", lambda compute, done, failed: jobs.append((compute, done, failed)))
    genes = _real_genes(state)
    ctrl.set_gene_visible("ATERA", genes[0], True)
    old_compute, old_done, old_failed = jobs[-1]
    old_runner = ctx.view_runner
    old_cancel = old_runner.cancel_event
    # Re-activating the dataset restores the cached payload: same context,
    # fresh runners whose generations restart.
    ctrl._open_atera_gene_inspector("ATERA", reuse_cached=True, announce=False)
    restored = _state(ctrl)
    assert restored.atera is ctx and ctx.view_runner is not old_runner
    ctx.view_timer = None
    ctrl.set_gene_visible("ATERA", genes[1], True)
    assert ctx.view_runner.busy
    new_generation = ctx.view_runner.running
    # The old job ends, cancelled or not: the new runner stays single-flight.
    old_failed(AV.TranscriptBuildCancelled("cancelled"))
    assert ctx.view_runner.busy and ctx.view_runner.running == new_generation
    rendered = ctx.viewport_cache
    old_cancel.clear()  # a read that completed just as the restore cancelled it
    old_done(old_compute())
    assert ctx.view_runner.busy and ctx.viewport_cache is rendered
    # The new job still lands normally.
    compute, done, _failed = jobs[-1]
    done(compute())
    assert not ctx.view_runner.busy and ctx.viewport_cache is not None


def test_restored_payload_ignores_the_torn_down_installs_gene_reads(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl._dataset_sessions["ATERA"] = SimpleNamespace(gene_payload=None, last_used=0)
    monkeypatch.setattr(ctrl, "_evict_dataset_sessions", lambda *a, **k: None)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    ctx.gene_timer = None
    jobs = []
    monkeypatch.setattr(ctrl, "_atera_run", lambda compute, done, failed: jobs.append((compute, done, failed)))
    genes = _real_genes(state)
    ctrl.set_gene_visible("ATERA", genes[0], True)
    old_compute, old_done, _old_failed = jobs[-1]
    old_cancel = ctx.gene_runner.cancel_event
    ctrl._open_atera_gene_inspector("ATERA", reuse_cached=True, announce=False)
    restored = _state(ctrl)
    assert restored.atera is ctx
    ctx.view_timer = None
    ctx.gene_timer = None
    ctrl.set_gene_visible("ATERA", genes[1], True)
    assert ctx.gene_runner.busy
    # The old install's gene store (built for genes[0] only) never lands.
    old_cancel.clear()  # a read that completed just as the restore cancelled it
    old_done(old_compute())
    assert ctx.gene_runner.busy and ctx.store_genes == frozenset()
    compute, done, _failed = jobs[-1]
    done(compute())
    assert not ctx.gene_runner.busy
    assert set(ctx.store_genes) == {g for g in restored.enabled_genes if g in ctx.loaded}
    assert genes[1] in ctx.store_genes


def test_lazy_density_follows_the_overview_switch_not_gene_ticks(qapp, atera_store, monkeypatch):
    ctrl = _controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    ctx.gene_timer = None
    density = _layer(ctrl, AV.ATERA_DENSITY_LAYER)
    assert density.visible  # the lazy whole-slide view before anything is ticked
    ctrl.set_gene_overview("ATERA", True)
    assert density.visible
    ctrl.set_gene_overview("ATERA", False)
    assert not density.visible
    genes = _real_genes(state)
    ctrl.set_gene_visible("ATERA", genes[0], True)
    ctrl.set_genes_visible("ATERA", genes[1:3], True)
    assert not density.visible  # ticks never re-show it
    ctrl.set_gene_overview("ATERA", True)
    assert density.visible
    density.visible = False  # hidden from the layer list
    ctrl.set_gene_visible("ATERA", genes[0], False)
    ctrl.set_gene_visible("ATERA", genes[3], True)
    assert not density.visible


def _store_with_is_gene_disagreeing(src: Path, dst: Path, gene: str, control: str) -> Path:
    """Copy ``src`` with ``gene`` flagged is_gene=False and ``control`` flagged True."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shutil.copytree(src, dst)
    files = sorted((dst / "points").rglob("*.parquet"))
    files = [f for f in files if f.is_file()]
    assert files
    for path in files:
        table = pq.read_table(path)
        if "is_gene" not in table.column_names:
            continue
        names = np.asarray(table.column("feature_name").to_pylist(), dtype=object)
        flags = np.asarray(table.column("is_gene").to_pylist(), dtype=bool)
        flags[names == gene] = False
        flags[names == control] = True
        table = table.set_column(table.schema.get_field_index("is_gene"), "is_gene", pa.array(flags))
        pq.write_table(table, path)
    return dst


def test_lazy_open_files_an_is_gene_false_feature_under_the_control_heading(qapp, atera_store, monkeypatch, tmp_path):
    fake_gene = "ACTB"  # gene-like name, flagged as a control
    named_control = "NegControlProbe_00002"  # control-like name, flagged as a gene
    store = _store_with_is_gene_disagreeing(atera_store, tmp_path / "flags.zarr", fake_gene, named_control)
    for mode in ("lazy", "full"):
        ctrl = _controller(store, monkeypatch, atera_transcript_mode=mode, gene_max_render_points=0)
        widget = V.GeneInspectorWidget(
            close_callback=lambda *_a: None,
            set_gene_visible_callback=ctrl.set_gene_visible,
            set_all_genes_callback=ctrl.set_all_genes_visible,
            set_spot_size_callback=ctrl.set_gene_spot_size,
            set_hide_background_callback=ctrl.set_gene_hide_background,
            set_show_controls_callback=ctrl.set_gene_show_controls,
            reload_atera_callback=ctrl.reload_atera_transcripts,
            set_atera_viewport_cap_callback=ctrl.set_atera_viewport_cap,
        )
        ctrl._gene_inspector_widget = widget
        ctrl.open_gene_inspector("ATERA")
        state = _state(ctrl)
        assert state.atera.overview.is_gene_flags_used
        assert fake_gene in state.store.genes and named_control in state.store.genes
        assert fake_gene in state.store.control_genes, mode
        assert named_control not in state.store.control_genes, mode
        model = widget._model
        group_of = {
            model.genes[int(gi)]: model.header_titles[int(group)]
            for group, members in enumerate(model.group_genes)
            for gi in members
        }
        # Genes listed before any heading (A-Z ordering) have no group.
        assert fake_gene in model.genes and named_control in model.genes
        assert group_of.get(fake_gene) == V.CONTROL_GROUP_TITLE, mode
        assert group_of.get(named_control) != V.CONTROL_GROUP_TITLE, mode
        assert bool(model.is_control[model.gene_index[fake_gene]])
        assert not bool(model.is_control[model.gene_index[named_control]])
        widget.clear()
        ctrl._teardown_gene_inspector("ATERA")


# ---------------------------------------------------------------------------
# M10(a): all-genes display cap - a nested sub-sample of the loaded sample
# ---------------------------------------------------------------------------


def _shown_points(ctrl, state) -> int:
    return int(sum(int(np.count_nonzero(_layer(ctrl, name).shown)) for name in state.layer_names))


def _shown_keys(ctrl, state, sample) -> list[int]:
    key_of = {(float(y), float(x)): int(k) for x, y, k in zip(sample.x, sample.y, sample.key)}
    keys = []
    for name in state.layer_names:
        layer = _layer(ctrl, name)
        for y, x in np.asarray(layer.data)[np.asarray(layer.shown, dtype=bool)]:
            keys.append(key_of[(float(y), float(x))])
    return keys


def _widget(ctrl):
    widget = V.GeneInspectorWidget(
        close_callback=lambda *_a: None,
        set_gene_visible_callback=ctrl.set_gene_visible,
        set_all_genes_callback=ctrl.set_all_genes_visible,
        set_spot_size_callback=ctrl.set_gene_spot_size,
        set_hide_background_callback=ctrl.set_gene_hide_background,
        set_show_controls_callback=ctrl.set_gene_show_controls,
        reload_atera_callback=ctrl.reload_atera_transcripts,
        set_atera_viewport_cap_callback=ctrl.set_atera_viewport_cap,
    )
    ctrl._gene_inspector_widget = widget
    return widget


def test_overview_cap_draws_a_nested_sub_sample_and_ticked_genes_draw_everything(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    budget, cap = total // 2, total // 6
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=budget, atera_overview_max_points=cap)
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    assert ctx.overview_capped and ctx.overview_cap == cap
    assert ctx.overview_threshold == T.sample_threshold(cap, total) < ctx.threshold
    assert state.store.overview_offsets is not None
    assert "All-transcript overview draws a nested" in ctrl.statuses[-1]
    engine = T.AteraTranscriptEngine.open(atera_store)
    full_sample = engine.sample_rows(ctx.threshold)
    real = set(_real_genes(state))
    real_code = np.isin(
        np.asarray(ctx.overview.vocabulary, dtype=object)[full_sample.code.astype(int)], sorted(real)
    )
    # Overview on: exactly the real-gene rows with key < cap threshold.
    ctrl.set_gene_overview("ATERA", True)
    keys = _shown_keys(ctrl, state, full_sample)
    expected = int(np.count_nonzero(real_code & (full_sample.key < np.uint32(ctx.overview_threshold))))
    assert len(keys) == expected > 0
    assert max(keys) < ctx.overview_threshold
    capped_count = len(keys)
    # A ticked gene draws its whole sampled run on top of the capped overview.
    gene = _real_genes(state)[0]
    ctrl.set_gene_visible("ATERA", gene, True)
    ctrl._flush_gene_group_rebuild("ATERA")  # gene ticks rebuild on a debounce timer
    _g, fg0, fg1, bg1 = state.store.gene_offsets[gene]
    fg_stop, bg_stop = state.store.overview_offsets[gene]
    extra = (fg1 - fg_stop) + (bg1 - bg_stop)
    assert _shown_points(ctrl, state) == capped_count + extra
    # Overview off: only the ticked gene, whole.
    ctrl.set_gene_overview("ATERA", False)
    assert _shown_points(ctrl, state) == bg1 - fg0
    # The panel's summary names the all-genes view size.
    assert AV.AteraTranscriptMixin._atera_overview_points(state) == expected


def test_overview_cap_at_or_above_the_budget_changes_nothing(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    for cap in (total // 2, total, 0):
        ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2, atera_overview_max_points=cap)
        ctrl.open_gene_inspector("ATERA")
        state = _state(ctrl)
        assert not state.atera.overview_capped
        assert state.store.overview_offsets is None
        ctrl.set_gene_overview("ATERA", True)
        real_points = sum(state.store.gene_counts[g] for g in _real_genes(state) if g in state.store.gene_counts)
        assert _shown_points(ctrl, state) == real_points
        ctrl._teardown_gene_inspector("ATERA")


def test_overview_cap_reload_reuses_the_cached_sample(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    ctrl.open_gene_inspector("ATERA")
    # The synthetic store is far below the 40M default: uncapped.
    assert ctrl._atera_overview_cap("ATERA") == AV.DEFAULT_ATERA_OVERVIEW_MAX_POINTS
    assert not _state(ctrl).atera.overview_capped
    loaded = _state(ctrl).store.total_points

    def forbidden(*_a, **_k):
        raise AssertionError("an all-genes cap change must not stream the store")

    monkeypatch.setattr(T.AteraTranscriptEngine, "build_overview", forbidden)
    monkeypatch.setattr(T.AteraTranscriptEngine, "sample_rows", forbidden)
    ctrl.reload_atera_transcripts("ATERA", "full", total // 2, overview_cap=total // 6)
    state = _state(ctrl)
    assert ctrl._atera_transcript_options["ATERA"]["overview_cap"] == total // 6
    assert state.store.total_points == loaded  # ticked genes keep the whole sample
    assert state.atera.overview_capped and state.atera.options_key()[2] == total // 6
    # A reload without a cap keeps the chosen one; 0 draws every loaded point.
    ctrl.reload_atera_transcripts("ATERA", "full", total // 2)
    assert _state(ctrl).atera.overview_capped
    ctrl.reload_atera_transcripts("ATERA", "full", total // 2, overview_cap=0)
    assert not _state(ctrl).atera.overview_capped
    assert _state(ctrl).store.overview_offsets is None


def test_overview_cap_field_is_applied_by_one_reload(qapp):
    calls = []
    widget = V.GeneInspectorWidget(
        close_callback=lambda *_a: None,
        set_gene_visible_callback=lambda *_a: None,
        set_all_genes_callback=lambda *_a: None,
        set_spot_size_callback=lambda *_a: None,
        set_hide_background_callback=lambda *_a: None,
        set_show_controls_callback=lambda *_a: None,
        reload_atera_callback=lambda *a, **k: calls.append((a, k)),
        set_atera_viewport_cap_callback=lambda *_a: None,
    )
    widget.set_atera_controls(
        True, dataset="ATERA", budget=None, auto_budget=171_800_000, total=1_409_600_000,
        loaded=171_800_000, overview_cap=40_000_000, overview_points=40_000_000,
    )
    assert widget._atera_overview_spin.value() == 40.0
    assert "all-genes view 40.0 M" in widget._atera_note.text()
    assert "Auto 171.8 M" in widget._atera_note.text()
    # The AUTO derivation sits in the tooltip; a clamp warning stays visible.
    assert "AUTO" not in widget._atera_note.text()
    widget.set_atera_controls(
        True, dataset="ATERA", budget=None, auto_budget=171_800_000, total=1_409_600_000,
        loaded=100_000_000, note="budget 300.0M requested; clamped to 100.0M",
    )
    assert "clamped to 100.0M" in widget._atera_note.text()
    assert "clamped" in widget._atera_note.toolTip()
    widget.set_atera_controls(
        True, dataset="ATERA", budget=None, auto_budget=171_800_000, total=1_409_600_000,
        loaded=171_800_000, overview_cap=40_000_000, overview_points=40_000_000,
        note="AUTO budget 171.8M = 0.5 x 64 GB RAM / 200 B per point",
    )
    assert "AUTO budget" in widget._atera_note.toolTip()
    assert "171.8" in widget._atera_budget_spin.toolTip()
    assert widget._atera_budget_spin.text() == "Auto"
    widget._atera_overview_spin.setValue(12.5)  # editing alone never reloads
    assert calls == []
    widget._atera_reload_button.click()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:3] == ("ATERA", "full", None) and kwargs == {"overview_cap": 12_500_000}
    widget._atera_overview_spin.setValue(0.0)
    assert widget._atera_overview_spin.text() == "Off"
    assert widget.atera_requested_overview_cap() == 0


def test_viewport_under_a_capped_overview_fills_in_without_duplicates(qapp, atera_store, monkeypatch):
    total = _total_rows(atera_store)
    ctrl = _controller(
        atera_store, monkeypatch, gene_max_render_points=total // 2, atera_overview_max_points=total // 6
    )
    ctrl.open_gene_inspector("ATERA")
    state = _state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    gene = _real_genes(state)[0]
    ctrl.set_gene_visible("ATERA", gene, True)
    ctrl._flush_gene_group_rebuild("ATERA")
    ctrl.set_gene_overview("ATERA", True)
    assert ctx.viewport_cache is not None
    assert ctx.viewport_cache["exclude"] == ctx.overview_threshold
    viewport = _layer(ctrl, AV.ATERA_VIEWPORT_LAYER)
    real = _real_genes(state)
    codes = [ctx.overview.vocabulary.index(g) for g in real]
    everything = ctx.engine.read_window_rows((-1e9, -1e9, 1e9, 1e9), codes)
    # Whole slide in view: store (capped overview + whole ticked gene) plus the
    # viewport is every real-gene row exactly once.
    assert _shown_points(ctrl, state) + len(viewport.data) == len(everything["x"])
    # Overview off: back to the ticked gene's skipped rows only.
    ctrl.set_gene_overview("ATERA", False)
    assert ctx.viewport_cache["exclude"] == ctx.threshold
    gene_rows = ctx.engine.read_window_rows((-1e9, -1e9, 1e9, 1e9), [ctx.overview.vocabulary.index(gene)])
    assert _shown_points(ctrl, state) + len(viewport.data) == len(gene_rows["x"])


def test_overview_cap_cli_default_is_40m(monkeypatch):
    monkeypatch.setattr("sys.argv", ["napari-compare"])
    try:
        args = V.parse_args()
    except SystemExit:
        pytest.skip("parse_args needs a dataset path")
    assert args.atera_overview_max_points == 40_000_000 == AV.DEFAULT_ATERA_OVERVIEW_MAX_POINTS


# ---------------------------------------------------------------------------
# M10(c): the Atera controls neither overlap nor clip at the panel's minimum width
# ---------------------------------------------------------------------------


def _atera_controls_widget():
    widget = V.GeneInspectorWidget(
        close_callback=lambda *_a: None,
        set_gene_visible_callback=lambda *_a: None,
        set_all_genes_callback=lambda *_a: None,
        set_spot_size_callback=lambda *_a: None,
        set_hide_background_callback=lambda *_a: None,
        set_show_controls_callback=lambda *_a: None,
        reload_atera_callback=lambda *_a, **_k: None,
        set_atera_viewport_cap_callback=lambda *_a: None,
    )
    # Real-store numbers: the longest texts the fields will hold.
    widget.set_atera_controls(
        True, dataset="ATERA", mode="full", budget=None, auto_budget=171_798_691,
        total=1_409_601_228, loaded=171_798_691, viewport_cap=20_000_000,
        overview_cap=40_000_000, overview_points=40_012_345,
        note="AUTO budget 171.8M = 0.5 x 64 GB RAM / 200 B per point",
    )
    return widget


def _text_fits(qapp, child) -> tuple[bool, str]:
    from qtpy.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QLabel,
        QPushButton,
        QStyle,
        QStyleOptionButton,
        QStyleOptionComboBox,
    )

    if isinstance(child, QCheckBox):
        # The text beside the indicator (a check box never wraps).
        option = QStyleOptionButton()
        child.initStyleOption(option)
        room = child.style().subElementRect(QStyle.SE_CheckBoxContents, option, child).width()
        needed = child.fontMetrics().horizontalAdvance(child.text())
        return needed <= room, f"check box '{child.text()}' {needed} > {room}"
    if isinstance(child, QComboBox):
        option = QStyleOptionComboBox()
        child.initStyleOption(option)
        field = child.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, child)
        widest = max(child.fontMetrics().horizontalAdvance(child.itemText(i)) for i in range(child.count()))
        return widest <= field.width(), f"combo {widest} > {field.width()}"
    if isinstance(child, QAbstractSpinBox):
        edit = child.lineEdit()
        needed = edit.fontMetrics().horizontalAdvance(edit.text())
        return needed <= edit.contentsRect().width(), f"spin '{edit.text()}' {needed} > {edit.contentsRect().width()}"
    if isinstance(child, QPushButton):
        needed = child.fontMetrics().horizontalAdvance(child.text())
        return needed <= child.contentsRect().width(), f"button {needed} > {child.width()}"
    if isinstance(child, QLabel) and not child.wordWrap():
        from qtpy.QtGui import QTextDocument

        doc = QTextDocument()
        doc.setDefaultFont(child.font())
        doc.setHtml(child.text())
        needed = int(doc.idealWidth()) - 2 * int(doc.documentMargin())
        return needed <= child.contentsRect().width(), f"label '{child.text()}' {needed} > {child.width()}"
    return True, ""


@pytest.mark.parametrize("panel_width", [240, 280, 320])
def test_atera_controls_neither_overlap_nor_clip(qapp, panel_width):
    widget = _atera_controls_widget()
    widget.resize(panel_width, 1200)
    widget.show()
    qapp.processEvents()
    try:
        assert widget.width() == panel_width
        box = widget._atera_box
        assert box.isVisible()
        children = [
            widget._atera_title, *widget._atera_form_labels.values(), widget._atera_mode_combo,
            widget._atera_budget_spin, widget._atera_overview_spin, widget._atera_viewport_spin,
            widget._atera_reload_button, widget._atera_note,
        ]
        rects = []
        for child in children:
            assert child.isVisible()
            rect = child.geometry()
            assert child.parentWidget() is box
            assert rect.left() >= 0 and rect.right() < box.width(), (type(child).__name__, rect)
            fits, why = _text_fits(qapp, child)
            assert fits, why
            rects.append((child, rect))
        for i, (a, ra) in enumerate(rects):
            for b, rb in rects[i + 1:]:
                assert not ra.intersects(rb), (type(a).__name__, ra, type(b).__name__, rb)
        # Each label sits beside or above its own field, never beside another's.
        form_fields = {
            "mode": widget._atera_mode_combo, "budget": widget._atera_budget_spin,
            "overview": widget._atera_overview_spin, "viewport": widget._atera_viewport_spin,
        }
        for key, label in widget._atera_form_labels.items():
            assert label.buddy() is form_fields[key]
            field = form_fields[key].geometry()
            assert label.geometry().top() <= field.top() <= label.geometry().bottom() + 8
        # The Reload button spans the box on its own row under every field.
        button = widget._atera_reload_button.geometry()
        assert button.width() == box.width()
        assert all(button.top() > f.geometry().bottom() for f in form_fields.values())
        # The note wraps inside the box instead of widening it.
        note = widget._atera_note
        assert note.wordWrap() and note.geometry().right() < box.width()
        assert "Auto 171.8 M" in note.text()
    finally:
        widget.close()


@pytest.mark.parametrize("napari_style", [False, True])
@pytest.mark.parametrize("datasets", [["ATERA"], ["XENIUM", "MERSCOPE"]])
def test_gene_inspector_fits_the_panel_at_its_minimum_width_with_the_scrollbar_showing(qapp, datasets, napari_style):
    """Inside the real panel (not the bare widget): at the panel's minimum width,
    in a window short enough that the Genes tab scrolls vertically, the
    scrollbar's width must not push the inspector's right edge out of view."""
    from qtpy.QtCore import QPoint

    from test_manual_registration_hooks import _panel

    widget = _atera_controls_widget()
    if datasets != ["ATERA"]:
        widget.set_atera_controls(False)
    panel = _panel(datasets, gene_inspector_widget=widget)
    if napari_style:
        # The real dock inherits napari's stylesheet: wider button padding and
        # a fixed 13 px scrollbar inside a bordered scroll area.
        from napari.qt import get_stylesheet

        panel.setStyleSheet(get_stylesheet("dark"))
    page = panel._tab_stack.widget(0)
    assert page.isAncestorOf(widget)
    panel._tab_stack.setCurrentWidget(page)
    panel.show()
    try:
        panel.resize(panel.minimumWidth(), 500)
        qapp.processEvents()
        assert panel.width() == panel.minimumWidth() == 240
        assert page.verticalScrollBar().isVisible()
        assert not page.horizontalScrollBar().isVisible()
        viewport = page.viewport()

        def right_in_viewport(child):
            return child.mapTo(viewport, QPoint(child.width() - 1, 0)).x()

        assert right_in_viewport(widget) < viewport.width()
        from qtpy.QtWidgets import QWidget

        # Every visible control of the inspector (not only the Atera box), plus
        # the tab's load/unload buttons: inside the viewport, text unclipped.
        children = [panel._load_transcripts_button, panel._unload_transcripts_button]
        children += [child for child in widget.findChildren(QWidget) if child.isVisibleTo(widget)]
        if datasets == ["ATERA"]:
            assert widget._atera_reload_button in children and widget._atera_mode_combo in children
        else:
            assert widget._atera_box not in children
        for name in ("_colour_by_assignment_check", "_show_controls_check", "_order_coarse_btn",
                     "_order_fine_btn", "_order_alpha_btn", "_spot_spin", "_spot_slider"):
            assert getattr(widget, name) in children, name
        for child in children:
            assert child.isVisible()
            label = f"{type(child).__name__} {getattr(child, 'text', lambda: '')()!r}"
            assert child.mapTo(viewport, QPoint(0, 0)).x() >= 0, label
            assert right_in_viewport(child) < viewport.width(), label
            fits, why = _text_fits(qapp, child)
            assert fits, why
        # The order buttons wrap rather than overlap.
        order = [widget._order_coarse_btn, widget._order_fine_btn, widget._order_alpha_btn]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                assert not a.geometry().intersects(b.geometry())
        if datasets == ["ATERA"]:
            box = widget._atera_box
            atera = [
                widget._atera_title, *widget._atera_form_labels.values(), widget._atera_mode_combo,
                widget._atera_budget_spin, widget._atera_overview_spin, widget._atera_viewport_spin,
                widget._atera_reload_button, widget._atera_note,
            ]
            for child in atera:
                assert child.geometry().left() >= 0 and child.geometry().right() < box.width(), type(child).__name__
                assert child.width() >= min(child.minimumSizeHint().width(), box.width()), type(child).__name__
            for i, a in enumerate(atera):
                for b in atera[i + 1:]:
                    assert not a.geometry().intersects(b.geometry()), (type(a).__name__, type(b).__name__)
        # The load/unload buttons wrap rather than overlap.
        assert not panel._load_transcripts_button.geometry().intersects(
            panel._unload_transcripts_button.geometry()
        )
        # A wider panel still gives the inspector the whole viewport width.
        panel.resize(400, 500)
        qapp.processEvents()
        assert widget.width() == viewport.width()
    finally:
        panel.close()


def test_atera_form_puts_labels_beside_fields_only_when_every_row_fits(qapp):
    from qtpy.QtWidgets import QFormLayout

    widget = _atera_controls_widget()
    form = widget._atera_box._form
    widget.show()
    try:
        widget.resize(700, 1200)
        qapp.processEvents()
        assert form.rowWrapPolicy() == QFormLayout.DontWrapRows
        for key, label in widget._atera_form_labels.items():
            field = label.buddy()
            assert field.geometry().left() > label.geometry().right()
            assert abs(field.geometry().center().y() - label.geometry().center().y()) <= 4
        widget.resize(widget.minimumWidth(), 1200)
        qapp.processEvents()
        assert form.rowWrapPolicy() == QFormLayout.WrapAllRows
        for key, label in widget._atera_form_labels.items():
            field = label.buddy()
            assert field.geometry().top() > label.geometry().bottom()
            assert field.width() >= field.minimumSizeHint().width()
            assert field.geometry().right() < widget._atera_box.width()
    finally:
        widget.close()


def test_atera_mode_items_are_short_with_the_description_in_the_tooltip(qapp):
    widget = _atera_controls_widget()
    combo = widget._atera_mode_combo
    from qtpy.QtCore import Qt

    assert [combo.itemText(i) for i in range(combo.count())] == ["Whole-slide sample", "Lazy (density)"]
    assert [combo.itemData(i) for i in range(combo.count())] == ["full", "lazy"]
    for i in range(combo.count()):
        assert len(str(combo.itemData(i, Qt.ToolTipRole))) > len(combo.itemText(i))
    assert widget._atera_budget_spin.suffix() == " M"
    assert widget._atera_budget_spin.specialValueText() == "Auto"


def test_short_genes_tab_scrolls_instead_of_squashing_the_atera_controls(qapp):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QScrollArea
    from test_manual_registration_hooks import _panel

    widget = _atera_controls_widget()
    panel = _panel([], gene_inspector_widget=widget)
    page = next(
        panel._tab_stack.widget(i)
        for i in range(panel._tab_stack.count())
        if panel._tab_stack.widget(i).isAncestorOf(widget)
    )
    assert isinstance(page, QScrollArea)
    # The horizontal bar is only a last resort (AsNeeded) for a dock narrower
    # than the panel's minimum; at any real panel width only the height scrolls.
    assert page.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    panel._tab_stack.setCurrentWidget(page)
    # A window far too short for the whole inspector (the real viewer's
    # 848 px window leaves the tab about 500 px).
    panel.resize(300, 420)
    panel.show()
    qapp.processEvents()
    try:
        box = widget._atera_box
        assert box.height() >= box.minimumSizeHint().height()
        assert widget.height() >= widget.minimumSizeHint().height()
        assert page.verticalScrollBar().maximum() > 0
        assert not page.horizontalScrollBar().isVisible()
        assert page.horizontalScrollBar().maximum() == 0
        assert widget.width() <= page.viewport().width()
    finally:
        panel.close()


@pytest.mark.parametrize("transient", [False, True])
def test_genes_tab_keeps_content_clear_of_an_overlay_scrollbar(qapp, transient):
    """macOS lays transient scrollbars over the viewport's right edge: the
    panel's scroll areas then reserve the bar's width so no control sits under it."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QProxyStyle, QStyle

    from test_manual_registration_hooks import _panel

    class Overlay(QProxyStyle):
        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QStyle.SH_ScrollBar_Transient:
                return int(transient)
            return super().styleHint(hint, option, widget, returnData)

    widget = _atera_controls_widget()
    panel = _panel(["ATERA"], gene_inspector_widget=widget)
    page = panel._tab_stack.widget(0)
    assert isinstance(page, V.PanelScrollArea)
    style = Overlay()
    style.setParent(page)
    page.setStyle(style)
    panel._tab_stack.setCurrentWidget(page)
    panel.show()
    try:
        panel.resize(panel.minimumWidth(), 500)
        qapp.processEvents()
        bar = page.verticalScrollBar()
        assert bar.isVisible() and page.overlay_scrollbars() is transient
        viewport = page.viewport()
        right = widget.mapTo(page, QPoint(widget.width() - 1, 0)).x()
        bar_left = bar.mapTo(page, QPoint(0, 0)).x()
        assert right < bar_left
        # The reserved right margin is the bar's width exactly when bars overlay.
        assert page._overlay_margin == (bar.sizeHint().width() if transient else 0)
        if transient:
            assert viewport.geometry().right() < page.width() - bar.sizeHint().width() + 1
        for child in (widget._atera_budget_spin, widget._atera_mode_combo, widget._spot_spin):
            assert child.mapTo(page, QPoint(child.width() - 1, 0)).x() < bar_left
        # Short of scrolling, the margin goes away and the content takes the full width.
        panel.resize(panel.minimumWidth(), 4000)
        qapp.processEvents()
        if not page.verticalScrollBar().maximum():
            assert page._overlay_margin == 0 and viewport.width() == page.width()
    finally:
        panel.close()
