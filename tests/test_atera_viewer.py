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
    # A re-tick of a loaded gene reads nothing.
    ctrl.set_gene_visible("ATERA", gene, True)
    ctrl._atera_flush_gene_reads("ATERA")
    assert len(reads) == 1
    # The lazy viewport shows other genes, not the loaded ones.
    ctrl.set_gene_overview("ATERA", True)
    if ctx.viewport_cache is not None:
        assert not np.isin(ctx.viewport_codes, [ctx.overview.vocabulary.index(g) for g in wanted]).any()


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
