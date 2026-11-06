"""Determinate progress: the reporter, the panel's bar and label, and the wiring.

Fake clocks and callbacks only; the Atera parts run on the synthetic store
from ``atera_stores`` (nothing reads /Users/... data).
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atera_stores
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope import atera_viewer as AV
from napari_compare_xenium_merscope import atera_transcripts as T
from napari_compare_xenium_merscope import progress as P
from napari_compare_xenium_merscope import transcript_cache as TC
from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import build_gene_point_groups

import test_atera_viewer as ATV
import test_gene_inspector as GI
import test_startup as ST


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory):
    return atera_stores.build_atera_latest_store(tmp_path_factory.mktemp("atera_progress"), pixel_size=0.5)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR", str(tmp_path / "transcript-cache"))
    monkeypatch.setattr(V, "thread_worker", None)
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


class _Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _reporter(clock=None):
    sent = []
    reporter = P.ProgressReporter(lambda *a: sent.append(a), clock=clock or _Clock())
    return reporter, sent


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


def test_reporter_throttles_to_ten_per_second_but_never_drops_first_or_final():
    clock = _Clock()
    reporter, sent = _reporter(clock)
    assert reporter.report("t", 0, 409, "streaming", "files")  # first value passes
    for done in range(1, 409):
        clock.t += 0.01  # 100 reports per second
        reporter.report("t", done, 409, "streaming", "files")
    clock.t += 0.001
    assert reporter.report("t", 409, 409, "streaming", "files")  # final always passes
    dones = [d for _k, d, _n, _t, _u in sent]
    assert dones[0] == 0 and dones[-1] == 409
    # 4.09 s of reports at <= 10/s, plus the first and the final one.
    assert len(sent) <= 4.09 * 10 + 2
    assert all(b > a for a, b in zip(dones, dones[1:]))


def test_reporter_values_are_monotonic_within_a_phase_and_reset_by_a_new_phase():
    clock = _Clock()
    reporter, sent = _reporter(clock)
    reporter.report("t", 5, 10, "a")
    clock.t += 1
    assert not reporter.report("t", 3, 10, "a")  # backwards: dropped
    assert not reporter.report("t", 5, 10, "a")  # repeat: dropped
    clock.t += 1
    assert reporter.report("t", 2, 20, "b")  # new phase (total/text) starts over
    assert reporter.report("t", 0, 0, "busy phase")  # back to a busy bar
    assert sent[-1] == ("t", 0, 0, "busy phase", "")
    # Out-of-range values are clamped to [0, total].
    clock.t += 1
    reporter.report("u", 50, 10)
    assert sent[-1][1:3] == (10, 10)


def test_reporter_begin_and_end_reset_a_stage():
    clock = _Clock()
    reporter, sent = _reporter(clock)
    reporter.report("t", 8, 10)
    reporter.end("t")
    # Right after a reset a lower value of the same phase passes immediately.
    assert reporter.report("t", 1, 10)
    reporter.begin("t")
    assert reporter.report("t", 1, 10)
    reporter.reset()
    assert reporter.report("t", 1, 10)
    assert [d for _k, d, *_ in sent] == [8, 1, 1, 1]


def test_reporter_keeps_order_across_threads():
    sent = []
    reporter = P.ProgressReporter(lambda *a: sent.append(a[1]), min_interval=0.0)
    counter = iter(range(1, 4001))
    lock = threading.Lock()

    def work():
        for _ in range(1000):
            with lock:
                done = next(counter)
            reporter.report("t", done, 4000)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sent == sorted(sent) and sent[-1] == 4000


# ---------------------------------------------------------------------------
# ProgressDisplay
# ---------------------------------------------------------------------------


def test_display_busy_then_determinate_with_count_percent_and_eta():
    clock = _Clock()
    display = P.ProgressDisplay(clock)
    display.set_stage("t", "ATERA: loading", True)
    assert display.bar() == (0, 0, 0)
    assert display.label() == "ATERA: loading"
    display.set_value("t", 0, 409, "ATERA: streaming transcripts", "files")
    assert display.bar() == (0, 409, 0)
    clock.t += 1.0
    display.set_value("t", 123, 409, "ATERA: streaming transcripts", "files")
    assert display.bar() == (0, 409, 123)
    # No ETA before 2 s.
    assert display.label() == "ATERA: streaming transcripts — 123/409 files (30%)"
    clock.t += 2.0
    display.set_value("t", 204, 409, "ATERA: streaming transcripts", "files")
    assert display.label().startswith("ATERA: streaming transcripts — 204/409 files (49%), about 3 s left")
    # Back to busy with a new text.
    display.set_value("t", 0, 0, "ATERA: grouping the points by gene…")
    assert display.bar() == (0, 0, 0)
    assert display.label() == "ATERA: grouping the points by gene…"


def test_display_ignores_values_for_inactive_stages_and_end_clears_them():
    display = P.ProgressDisplay(_Clock())
    assert not display.set_value("t", 1, 2)  # never begun
    display.set_stage("t", "x", True)
    display.set_value("t", 1, 2)
    display.set_stage("t", "", False)  # cancel / end
    assert not display.active and display.bar() == (0, 0, 0)
    assert not display.set_value("t", 2, 2)  # a late worker value is dropped
    display.set_stage("t", "again", True)
    assert display.bar() == (0, 0, 0)  # a fresh stage starts busy


def test_display_concurrent_stages_stay_with_the_first_determinate_one():
    clock = _Clock()
    display = P.ProgressDisplay(clock)
    display.set_stage("images", "images…", True)
    display.set_stage("transcripts", "transcripts…", True)
    assert display.bar() == (0, 0, 0)
    display.set_value("transcripts", 3, 10, unit="files")
    clock.t += 0.5
    display.set_value("masks", 1, 2)  # inactive: ignored
    assert display.bar() == (0, 10, 3)
    display.set_stage("masks", "masks…", True)
    clock.t += 0.5
    display.set_value("masks", 1, 2, unit="segmentations")
    # The newer stage shows in the label only; the bar keeps the first one.
    assert display.bar() == (0, 10, 3)
    assert display.label() == "images… | transcripts… — 3/10 files (30%) | masks… — 1/2 segmentations (50%)"
    display.set_stage("transcripts", "", False)
    assert display.bar() == (0, 2, 1)  # the first stage ended: the next one takes over
    display.set_stage("transcripts", "transcripts…", True)
    display.set_value("transcripts", 5, 10, unit="files")
    assert display.bar() == (0, 2, 1)
    # Values within a phase never go backwards on the display either.
    display.set_value("masks", 0, 2, unit="segmentations")
    assert display.bar() == (0, 2, 1)
    # A stage that goes busy again gives the bar up until it has a total again.
    display.set_value("masks", 0, 0, text="masks: saving…")
    assert display.bar() == (0, 10, 5)


def test_display_bar_never_decreases_while_interleaved_stages_report():
    clock = _Clock()
    display = P.ProgressDisplay(clock)
    display.set_stage("transcripts", "partitions…", True)
    display.set_stage("masks", "segmentations…", True)
    fractions = []
    # The interleaving the real viewer produced: 2/4, 1/3, 3/4, 2/3, 4/4, 3/3.
    for key, done, total in [("transcripts", 1, 4), ("transcripts", 2, 4), ("masks", 1, 3), ("transcripts", 3, 4),
                             ("masks", 2, 3), ("transcripts", 4, 4), ("masks", 3, 3)]:
        clock.t += 0.1
        display.set_value(key, done, total)
        lo, hi, value = display.bar()
        fractions.append(value / hi)
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_display_scales_totals_beyond_int32_and_formats_bytes():
    display = P.ProgressDisplay(_Clock())
    display.set_stage("t", "rows", True)
    display.set_value("t", 3 * 10**9, 6 * 10**9, unit="rows")
    assert display.bar() == (0, P.PROGRESS_BAR_STEPS, P.PROGRESS_BAR_STEPS // 2)
    assert P.format_count(512 * 1024**2, 2 * 1024**3, "bytes") == "512.0 MB of 2.0 GB (25%)"
    assert P.format_eta(125) == "about 2 min 05 s left"


# ---------------------------------------------------------------------------
# The panel's QProgressBar
# ---------------------------------------------------------------------------


def test_panel_bar_switches_between_busy_and_determinate(qapp):
    panel = ST._panel(["XENIUM"])
    bar = panel._progress_bar
    panel.set_progress("transcripts", "XENIUM: loading…", True)
    assert not bar.isHidden() and (bar.minimum(), bar.maximum()) == (0, 0)
    panel.set_progress_value("transcripts", 0, 40, "XENIUM: streaming transcripts", "partition reads")
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 40, 0)
    panel.set_progress_value("transcripts", 12, 40, "XENIUM: streaming transcripts", "partition reads")
    assert bar.value() == 12 and bar.isTextVisible()
    assert "12/40 partition reads (30%)" in panel._progress_label.text()
    panel.set_progress_value("transcripts", 0, 0, "XENIUM: grouping…")
    assert bar.maximum() == 0 and panel._progress_label.text() == "XENIUM: grouping…"
    panel.set_progress("transcripts", "", False)
    assert bar.isHidden() and panel._progress_label.text() == ""
    # A late worker value after the stage ended does not bring the bar back.
    panel.set_progress_value("transcripts", 39, 40, "late", "files")
    assert bar.isHidden() and (bar.minimum(), bar.maximum()) == (0, 0)


# ---------------------------------------------------------------------------
# Controller API
# ---------------------------------------------------------------------------


def _bare_controller():
    ctrl = V.ComparisonViewerController(GI._FakeViewer(), {}, SimpleNamespace())
    stages, values = [], []
    ctrl.set_progress_callback(lambda key, text, active: stages.append((key, text, active)))
    ctrl.set_progress_value_callback(lambda *a: values.append(a))
    return ctrl, stages, values


def test_controller_begin_with_total_starts_determinate_and_end_resets(qapp):
    ctrl, stages, values = _bare_controller()
    ctrl._begin_progress("transcripts", "loading", total=8, status=False)
    assert stages == [("transcripts", "loading", True)]
    assert values == [("transcripts", 0, 8, "", "")]
    ctrl._report_progress("transcripts", 8, 8, unit="files")
    assert values[-1] == ("transcripts", 8, 8, "", "files")
    ctrl._end_progress("transcripts")
    assert stages[-1] == ("transcripts", "", False)
    ctrl._begin_progress("transcripts", "again")
    ctrl._report_progress("transcripts", 1, 8)  # a fresh run: never clamped by the old one
    assert values[-1][1] == 1
    ctrl._reset_progress()
    keys = {k for k, _t, active in stages if not active}
    assert {"transcripts", V.GENE_LAYERS_PROGRESS_KEY, AV.ATERA_GENES_PROGRESS_KEY} <= keys


def test_controller_without_a_value_callback_ignores_reports():
    ctrl = V.ComparisonViewerController.__new__(V.ComparisonViewerController)
    ctrl._progress_callback = None
    ctrl._status_callback = None
    ctrl._begin_progress("t", "x", total=3)
    ctrl._report_progress("t", 1, 3)
    ctrl._end_progress("t")


# ---------------------------------------------------------------------------
# Chunked gene-layer creation on the Qt thread
# ---------------------------------------------------------------------------


def _chunked_controller(store, monkeypatch, steps):
    args = SimpleNamespace(
        gene_spot_size=2.0, gene_max_render_points=40_000_000, gene_hide_background=False,
        gene_show_controls=False, random_state=42, gene_layer_chunk_min_points=1,
    )
    ctrl = V.ComparisonViewerController(GI._FakeViewer(), {}, args)
    ctrl.active_dataset = "TEST"
    ctrl._gene_inspector_widget = None
    monkeypatch.setattr(ctrl, "_defer_gene_layer_step", steps.append)
    stages, values = [], []
    ctrl.set_progress_callback(lambda key, text, active: stages.append((key, text, active)))
    ctrl.set_progress_value_callback(lambda *a: values.append(a))
    return ctrl, stages, values


def _apply(ctrl, store):
    ctrl._gene_build_generation += 1
    ctrl._apply_gene_inspector_build(
        ctrl._gene_build_generation, "TEST", {"points_key": "transcripts", "store": store, "build_seconds": 0.0},
    )


def test_large_stores_create_their_layers_one_group_per_event_loop_turn(qapp, monkeypatch):
    store = GI._demo_store()
    n = len(store.group_symbols)
    assert n == 3
    steps = []
    ctrl, stages, values = _chunked_controller(store, monkeypatch, steps)
    _apply(ctrl, store)
    key = V.GENE_LAYERS_PROGRESS_KEY
    state = ctrl._gene_inspector_states["TEST"]
    assert ctrl.viewer.layers == [] and len(steps) == 1
    assert (key, "TEST: creating the transcript layers…", True) in stages
    assert [v[1:5:3] for v in values if v[0] == key] == [(0, "layers")]
    created = []
    while steps:
        steps.pop(0)()
        created.append(len(ctrl.viewer.layers))
    # Each group: one turn builds its arrays and layer, the next adds (draws)
    # it; the finish (ordering, widget) runs on a turn of its own.
    assert created == [0, 1, 1, 2, 2, 3, 3]
    assert [v[1:3] for v in values if v[0] == key] == [(0, n), (1, n), (2, n), (3, n)]
    assert stages[-1] == (key, "", False)
    assert state.layer_chain is None
    # The finished state equals the one-shot build's.
    sync = GI._controller(store, qapp)
    sync_state = sync._gene_inspector_states["TEST"]
    assert state.layer_names == sync_state.layer_names
    assert GI._total(ctrl, state) == GI._total(sync, sync_state) == 9


def test_a_generation_bump_mid_chain_still_finishes_the_registered_build(qapp, monkeypatch):
    # ``load_dataset(other)`` cancels the background tasks (a generation bump)
    # before the switch, which may fail and keep this dataset: the chain must
    # not leave a half-built state behind.
    store = GI._demo_store()
    steps = []
    ctrl, stages, _values = _chunked_controller(store, monkeypatch, steps)
    finished = []
    original = ctrl._finish_gene_inspector_build
    monkeypatch.setattr(ctrl, "_finish_gene_inspector_build", lambda *a: (finished.append(a), original(*a)))
    _apply(ctrl, store)
    steps.pop(0)()
    steps.pop(0)()
    assert len(ctrl.viewer.layers) == 1
    ctrl._cancel_background_tasks()
    while steps:
        steps.pop(0)()
    state = ctrl._gene_inspector_states["TEST"]
    assert len(state.layer_names) == len(store.group_symbols) == len(ctrl.viewer.layers)
    assert state.layer_chain is None and ctrl._gene_layer_chain is None
    assert len(finished) == 1
    assert stages[-1] == (V.GENE_LAYERS_PROGRESS_KEY, "", False)
    # A session apply (what a successful switch does) still stops a chain.
    _apply(ctrl, store)
    ctrl._teardown_gene_inspector("TEST")
    _apply(ctrl, store)
    steps.pop(0)()
    ctrl._clear_layers()
    while steps:
        steps.pop(0)()
    assert ctrl._gene_inspector_states == {} and ctrl.viewer.layers == []


class _FakePairedRegistry:
    def __init__(self, viewer):
        self.viewer = viewer
        self.slot_keys = []
        self.repaired = 0

    def register_layer(self, identity, layer):
        self.viewer.layers.append(layer)
        self.slot_keys.append(identity.slot_key)
        return layer

    def repair_order(self):
        self.repaired += 1


def _paired_chunked_controller(monkeypatch, steps, chunk_min_points=1):
    store = GI._demo_store()
    ctrl, stages, values = _chunked_controller(store, monkeypatch, steps)
    ctrl.args.gene_layer_chunk_min_points = chunk_min_points
    ctrl._paired_session = SimpleNamespace(
        contract=SimpleNamespace(pair_id="pair", profile=lambda ds: SimpleNamespace(points_key="transcripts")),
    )
    ctrl._paired_registry = _FakePairedRegistry(ctrl.viewer)
    monkeypatch.setattr(ctrl, "_transcript_payload_is_current", lambda *_a: True)
    monkeypatch.setattr(ctrl, "_remove_paired_anchor", lambda: None)
    monkeypatch.setattr(ctrl, "_schedule_canvas_visibility_repair", lambda *_a: None)
    statuses = []
    ctrl.set_status_callback(statuses.append)
    payloads = {
        ds: {"points_key": "transcripts", "store": GI._demo_store(), "build_seconds": 0.0}
        for ds in ("MERSCOPE", "XENIUM")
    }
    ctrl._gene_build_generation += 1
    ctrl._apply_paired_gene_build(ctrl._gene_build_generation, "pair", payloads, None)
    return ctrl, stages, values, statuses


def test_large_pairs_create_their_layers_one_per_event_loop_turn_across_both_platforms(qapp, monkeypatch):
    steps = []
    ctrl, stages, values, statuses = _paired_chunked_controller(monkeypatch, steps)
    key = V.GENE_LAYERS_PROGRESS_KEY
    n = 6  # 3 groups on each platform
    assert ctrl.viewer.layers == [] and len(steps) == 1
    assert (key, "Paired: creating the transcript layers…", True) in stages
    created = []
    while steps:
        steps.pop(0)()
        created.append(len(ctrl.viewer.layers))
    assert created == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert [v[1:3] for v in values if v[0] == key] == [(i, n) for i in range(n + 1)]
    assert stages[-1] == (key, "", False)
    states = [ctrl._gene_inspector_states[ds] for ds in ("MERSCOPE", "XENIUM")]
    assert [len(state.layer_names) for state in states] == [3, 3]
    assert all(state.layer_chain is None for state in states)
    assert ctrl._paired_registry.repaired == 1
    assert statuses[-1].startswith("Paired transcripts: 22 points in 6 platform-qualified layer(s)")
    assert all(isinstance(layer, V.FastPoints) for layer in ctrl.viewer.layers)


def test_small_pairs_keep_the_one_shot_layer_build(qapp, monkeypatch):
    steps = []
    ctrl, _stages, _values, statuses = _paired_chunked_controller(monkeypatch, steps, chunk_min_points=10**9)
    assert steps == [] and len(ctrl.viewer.layers) == 6
    assert ctrl._paired_registry.repaired == 1
    assert "6 platform-qualified layer(s)" in statuses[-1]


def test_paired_teardown_mid_way_stops_the_chunks_and_hides_their_progress(qapp, monkeypatch):
    steps = []
    ctrl, stages, _values, _statuses = _paired_chunked_controller(monkeypatch, steps)
    steps.pop(0)()
    steps.pop(0)()
    assert len(ctrl.viewer.layers) == 1
    monkeypatch.setattr(ctrl, "_remove_paired_slot", lambda key: ctrl._paired_registry.slot_keys.remove(key))
    ctrl._teardown_paired_gene_inspector()
    assert stages[-1] == (V.GENE_LAYERS_PROGRESS_KEY, "", False)
    ended = len(stages)
    while steps:
        steps.pop(0)()
    assert ctrl._gene_inspector_states == {}
    assert len(stages) == ended and ctrl._paired_registry.repaired == 0


def test_small_stores_keep_the_one_shot_layer_build(qapp, monkeypatch):
    store = GI._demo_store()
    steps = []
    ctrl, stages, _values = _chunked_controller(store, monkeypatch, steps)
    ctrl.args.gene_layer_chunk_min_points = 10**9
    _apply(ctrl, store)
    assert steps == [] and len(ctrl.viewer.layers) == 3
    assert not any(k == V.GENE_LAYERS_PROGRESS_KEY for k, _t, _a in stages)


def test_teardown_mid_way_stops_the_chunks_and_hides_their_progress(qapp, monkeypatch):
    store = GI._demo_store()
    steps = []
    ctrl, stages, _values = _chunked_controller(store, monkeypatch, steps)
    _apply(ctrl, store)
    steps.pop(0)()  # builds group 0
    steps.pop(0)()  # adds it
    assert len(ctrl.viewer.layers) == 1
    ctrl._teardown_gene_inspector("TEST")
    assert stages[-1] == (V.GENE_LAYERS_PROGRESS_KEY, "", False)
    while steps:
        steps.pop(0)()
    assert "TEST" not in ctrl._gene_inspector_states
    assert ctrl.viewer.layers == [] or all(
        not str(getattr(layer, "name", "")).startswith("Genes") for layer in ctrl.viewer.layers
    )


def test_a_superseded_chain_never_touches_the_new_builds_layers_or_progress(qapp, monkeypatch):
    store = GI._demo_store()
    steps = []
    ctrl, stages, _values = _chunked_controller(store, monkeypatch, steps)
    _apply(ctrl, store)
    old_step = steps.pop(0)
    ctrl._teardown_gene_inspector("TEST")
    _apply(ctrl, store)
    ended = len([s for s in stages if s == (V.GENE_LAYERS_PROGRESS_KEY, "", False)])
    old_step()  # the stale chain's step: a no-op
    assert ctrl.viewer.layers == []
    assert len([s for s in stages if s == (V.GENE_LAYERS_PROGRESS_KEY, "", False)]) == ended
    while steps:
        steps.pop(0)()
    assert len(ctrl._gene_inspector_states["TEST"].layer_names) == 3


# ---------------------------------------------------------------------------
# Worker-side counts
# ---------------------------------------------------------------------------


def test_eager_build_reports_partitions_over_both_passes():
    rows = [(f"G{i % 5}", float(i), float(i % 7), i % 3 == 0) for i in range(200)]
    frame = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    ddf = dd.from_pandas(frame, npartitions=4)
    seen = []
    store = build_gene_point_groups(
        ddf, x_col="x", y_col="y", gene_col="gene", background_col="background",
        progress=lambda done, total: seen.append((done, total)),
    )
    plain = build_gene_point_groups(ddf, x_col="x", y_col="y", gene_col="gene", background_col="background")
    n = ddf.npartitions
    assert seen[-1] == (2 * n, 2 * n) and len(seen) == 2 * n
    assert [d for d, _t in seen] == list(range(1, 2 * n + 1))
    assert store.total_points == plain.total_points == 200
    for a, b in zip(store.group_coords, plain.group_coords):
        np.testing.assert_array_equal(a, b)


def test_atera_reads_and_sample_restore_report_their_counts(atera_store, tmp_path):
    engine = T.AteraTranscriptEngine.open(atera_store)
    genes = [g for g in engine.index.vocabulary[:3]]
    seen = []
    engine.read_genes(genes, progress=lambda d, n: seen.append((d, n)))
    assert seen[0][0] == 0 and seen[-1][0] == seen[-1][1] >= 1
    x0, x1, y0, y1 = engine.index.extent_um()
    seen.clear()
    engine.read_window_rows((x0, y0, x1 + 1, y1 + 1), progress=lambda d, n: seen.append((d, n)))
    assert seen[-1] == (len(engine.files_in_window((x0, y0, x1 + 1, y1 + 1))),) * 2
    cache = tmp_path / "cache"
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 2, total)
    TC.save_atera_sample(engine.index, engine.sample_rows(threshold), cache_root=cache)
    seen.clear()
    got = TC.load_atera_sample(engine.index, threshold // 2, cache_root=cache, progress=lambda d, n: seen.append((d, n)))
    assert got is not None
    expected_bytes = sum(getattr(got, name).nbytes for name in ("x", "y", "code", "cell_id", "key"))
    assert seen[0] == (0, expected_bytes) and seen[-1] == (expected_bytes, expected_bytes)
    assert [d for d, _n in seen] == sorted(d for d, _n in seen)


def test_atera_full_load_reports_files_and_ends_its_stages(qapp, atera_store, monkeypatch):
    total = ATV._total_rows(atera_store)
    ctrl = ATV._controller(atera_store, monkeypatch, gene_max_render_points=total // 2)
    stages, values = [], []
    ctrl.set_progress_callback(lambda key, text, active: stages.append((key, text, active)))
    ctrl.set_progress_value_callback(lambda *a: values.append(a))
    ctrl.open_gene_inspector("ATERA")
    files = [v for v in values if v[0] == "transcripts" and v[4] == "files"]
    assert files and files[-1][1] == files[-1][2] == len(T.AteraTranscriptEngine.open(atera_store).index.files)
    assert any(v[0] == "transcripts" and v[2] == 0 and "grouping" in v[3] for v in values)
    assert ("transcripts", "", False) in stages
    # A second open restores the cached sample with a byte count.
    stages.clear()
    values.clear()
    ctrl._teardown_gene_inspector("ATERA")
    ctrl._dataset_sessions.clear()
    ctrl.open_gene_inspector("ATERA")
    restored = [v for v in values if v[0] == "transcripts" and v[4] == "bytes"]
    assert restored and restored[-1][1] == restored[-1][2] > 0


def _atera_progress(ctrl):
    stages, values = [], []
    ctrl.set_progress_callback(lambda key, text, active: stages.append((key, text, active)))
    ctrl.set_progress_value_callback(lambda *a: values.append(a))
    return stages, values


def test_lazy_gene_reads_show_and_then_hide_a_file_count(qapp, atera_store, monkeypatch):
    ctrl = ATV._controller(atera_store, monkeypatch, atera_transcript_mode="lazy")
    ctrl.open_gene_inspector("ATERA")
    state = ATV._state(ctrl)
    state.atera.view_timer = None
    stages, values = _atera_progress(ctrl)
    for gene in ATV._real_genes(state)[:2]:
        ctrl.set_gene_visible("ATERA", gene, True)
    ctrl._atera_flush_gene_reads("ATERA")
    key = AV.ATERA_GENES_PROGRESS_KEY
    began = [s for s in stages if s[0] == key and s[2]]
    assert began and "reading 2 gene(s)" in began[0][1]
    counts = [v[1:3] for v in values if v[0] == key]
    assert counts[0][0] == 0 and counts[-1][0] == counts[-1][1] >= 1
    assert stages[-1] == (key, "", False) or (key, "", False) in stages
    assert not state.atera.gene_runner.busy


def test_wide_viewport_reads_show_progress_and_narrow_ones_do_not(qapp, atera_store, monkeypatch):
    total = ATV._total_rows(atera_store)
    ctrl = ATV._controller(atera_store, monkeypatch, gene_max_render_points=total // 3)
    ctrl.open_gene_inspector("ATERA")
    state = ATV._state(ctrl)
    ctx = state.atera
    ctx.view_timer = None
    key = AV.ATERA_VIEWPORT_PROGRESS_KEY
    n_files = len(ctx.engine.files_in_window((-1e9, -1e9, 1e9, 1e9)))
    stages, values = _atera_progress(ctrl)
    monkeypatch.setattr(AV, "ATERA_VIEWPORT_PROGRESS_MIN_FILES", 10**6)
    ctrl.set_gene_visible("ATERA", ATV._real_genes(state)[0], True)
    assert not any(s[0] == key and s[2] for s in stages)  # a small read never flashes the bar
    monkeypatch.setattr(AV, "ATERA_VIEWPORT_PROGRESS_MIN_FILES", 1)
    ctrl.set_gene_visible("ATERA", ATV._real_genes(state)[1], True)
    assert (key, "", False) in stages and any(s[0] == key and s[2] for s in stages)
    counts = [v[1:3] for v in values if v[0] == key]
    assert counts and counts[-1][0] == counts[-1][1] and counts[-1][1] <= n_files
