"""Source-aware transcript preparation and inspection, without an OpenGL viewer."""

from __future__ import annotations

from threading import Semaphore
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import (
    CellTranscriptIndex,
    CellTypeAssignments,
    MarkerReferenceResolution,
)

SOURCE = "pciseq_proseg_hybrid"
POINTS = "transcripts_pciseq_proseg_hybrid"


def _metadata(identity="a" * 64):
    return dict(
        method="pciseq",
        points=POINTS,
        source_points="transcripts",
        assignment_column="assignment",
        background_column="background",
        status_column="pciseq_assignment_status",
        analysis_id=identity,
        segmentation="proseg_hybrid",
        shape="MOSAIK_proseg_hybrid",
    )


def _points():
    frame = pd.DataFrame(
        dict(
            x=[1.0, 2.0, 3.0],
            y=[4.0, 5.0, 6.0],
            gene=["A", "B", "A"],
            assignment=pd.array([2**64 - 1, None, None], dtype="UInt64"),
            background=[False, True, False],
            pciseq_assignment_status=["assigned", "background", "not_modelled"],
        )
    )
    native = frame.copy()
    native["assignment"] = [11, 11, 11]
    return SimpleNamespace(points={POINTS: frame, "transcripts": native}, shapes={})


@pytest.fixture
def controller(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "thread_worker", None)
    c = V.ComparisonViewerController.__new__(V.ComparisonViewerController)
    c.args = SimpleNamespace(gene_max_render_points=2, random_state=42)
    c.active_dataset = "XENIUM"
    c._active_sdata = _points()
    c._paired_session = None
    c._paired_registry = None
    c.datasets = {
        ds: SimpleNamespace(zarr_path=tmp_path / ds) for ds in ("MERSCOPE", "XENIUM")
    }
    c._dataset_sessions = {}
    c._transcript_assignment_sources = {}
    c._gene_inspector_widget = None
    c._gene_inspector_states = {}
    c._gene_build_generation = 0
    c._gene_build_worker = None
    c._task_cancel_events = {}
    c._task_keys = {}
    c._store_io_slots = Semaphore(1)
    c._cell_transcript_index = {}
    c._cell_intensity_cache = {}
    c._cell_inspection_workers = {}
    c._cell_inspection_generation = 0
    c._selected_cells = {}
    c._paired_selected_cells = {}
    c._progress_callback = None
    c.statuses = []
    c._set_status = c.statuses.append
    c._remove_layers_by_prefix = lambda *_: None
    c._rebuild_cell_highlights = lambda *_: None
    c._clear_all_cell_selections = lambda: None
    c._ensure_dataset_is_active = lambda _: True
    metadata = _metadata()
    monkeypatch.setattr(
        V,
        "assignment_source_points",
        lambda path, source: (
            dict(metadata)
            if source == SOURCE and path == c.datasets["XENIUM"].zarr_path
            else None
        ),
    )
    monkeypatch.setattr(
        V,
        "resolve_cell_type_marker_reference",
        lambda *_: MarkerReferenceResolution(reference=None),
    )
    return c, metadata


def test_native_points_are_selected_explicitly_even_when_pciseq_is_first(controller):
    c, _ = controller
    columns = c._resolve_gene_points_columns()
    assert columns[0] == "transcripts"
    assert columns[1] is c._active_sdata.points["transcripts"]
    c._transcript_assignment_sources["XENIUM"] = SOURCE
    assert c._resolve_gene_points_columns()[0] == POINTS
    assert c._cell_inspector_shape_key() == "MOSAIK_proseg_hybrid"


def test_switch_prepares_exact_pciseq_index_and_cancels_native_build(controller):
    c, _ = controller
    old = c._begin_task_token("transcripts", ("old-native",))
    c._cell_transcript_index["XENIUM"] = CellTranscriptIndex.empty()
    c._dataset_sessions["XENIUM"] = SimpleNamespace(
        gene_payload={"assignment_source": "native"}
    )
    results = []
    c._apply_gene_inspector_build = lambda _gen, _ds, payload, _token: results.append(
        payload
    )
    assert c.set_transcript_assignment_source("XENIUM", SOURCE)
    assert old.is_set()
    assert "XENIUM" not in c._cell_transcript_index
    assert c._dataset_sessions["XENIUM"].gene_payload is None
    payload = results[0]
    assert payload["assignment_source"] == SOURCE
    assert payload["analysis_id"] == "a" * 64
    assert payload["store"].source_status_counts == {
        "assigned": 1,
        "background": 1,
        "not_modelled": 1,
    }
    assert payload["store"].total_points == 2
    coords, genes = payload["store"].cell_transcript_index.transcripts_for(2**64 - 1)
    assert genes.tolist() == ["A"]
    np.testing.assert_array_equal(coords, [[4.0, 1.0]])
    assert len(payload["store"].cell_transcript_index.transcripts_for(11)[1]) == 0
    assert SOURCE in c._task_keys["transcripts"]
    assert "a" * 64 in c._task_keys["transcripts"]


def test_stale_analysis_build_cannot_replace_current_session(controller):
    c, metadata = controller
    c._transcript_assignment_sources["XENIUM"] = SOURCE
    c._gene_build_generation = 7
    sentinel = object()
    c._gene_build_worker = sentinel
    metadata["analysis_id"] = "b" * 64
    c._apply_gene_inspector_build(
        7, "XENIUM", dict(assignment_source=SOURCE, analysis_id="a" * 64)
    )
    assert c._gene_build_worker is sentinel
    c._apply_gene_inspector_build(
        7, "XENIUM", dict(assignment_source="native", analysis_id="")
    )
    assert c._gene_build_worker is sentinel


def test_cell_index_is_never_reused_across_source_or_analysis_changes(controller):
    c, metadata = controller
    index = SimpleNamespace(marker="native")
    c._cache_cell_transcript_index("XENIUM", index)
    assert c._get_cell_transcript_index("XENIUM") is index
    c._transcript_assignment_sources["XENIUM"] = SOURCE
    assert c._get_cell_transcript_index("XENIUM") is not index
    c._cache_cell_transcript_index("XENIUM", index)
    assert c._get_cell_transcript_index("XENIUM") is index
    metadata["analysis_id"] = "b" * 64
    assert c._get_cell_transcript_index("XENIUM") is not index
    c._transcript_source_metadata = lambda *_: None
    c._cache_cell_transcript_index("XENIUM", index, (SOURCE, ""))
    assert c._get_cell_transcript_index("XENIUM") is not index


def _pair(c):
    profiles = {
        ds: SimpleNamespace(points_key="transcripts", segmentations=())
        for ds in ("MERSCOPE", "XENIUM")
    }
    c._paired_session = SimpleNamespace(
        contract=SimpleNamespace(
            pair_id="pair", profile=profiles.__getitem__, mappings={}
        ),
        configs=c.datasets,
        sdata={"MERSCOPE": _points(), "XENIUM": c._active_sdata},
    )
    c._transcript_assignment_sources = {ds: SOURCE for ds in profiles}


def test_paired_pciseq_build_skips_unavailable_platform_without_native_fallback(
    controller,
):
    c, _ = controller
    _pair(c)
    results = []
    c._apply_paired_gene_build = lambda _generation, _pair, payloads, _token: (
        results.append(payloads)
    )
    assert c._open_paired_gene_inspector()
    assert set(results[0]) == {"XENIUM"}
    assert results[0]["XENIUM"]["points_key"] == POINTS
    with pytest.raises(KeyError, match="unavailable"):
        c._paired_gene_build_input("MERSCOPE")
    c._paired_registry = object()
    assert c._paired_pick_segmentation("MERSCOPE") is None


def test_paired_async_build_rejects_changed_analysis_identity(controller):
    c, metadata = controller
    _pair(c)
    c._gene_build_generation = 4
    sentinel = object()
    c._gene_build_worker = sentinel
    metadata["analysis_id"] = "b" * 64
    c._apply_paired_gene_build(
        4,
        "pair",
        {"XENIUM": dict(assignment_source=SOURCE, analysis_id="a" * 64)},
        None,
    )
    assert c._gene_build_worker is sentinel


def test_source_panel_names_single_side_and_separates_background_from_not_modelled(
    controller,
):
    c, _ = controller
    _pair(c)
    calls = []
    c._gene_inspector_widget = SimpleNamespace(
        set_assignment_sources=lambda *args, **kw: calls.append((args, kw))
    )
    c._gene_inspector_states["XENIUM"] = SimpleNamespace(
        assignment_source=SOURCE,
        store=SimpleNamespace(
            source_status_counts={"assigned": 5, "background": 2, "not_modelled": 3}
        ),
    )
    c._publish_transcript_sources()
    options, selected, summary = calls[0][0]
    assert options["native"] == "Native assignments"
    assert "XENIUM only" in options[SOURCE]
    assert selected == SOURCE
    assert "MERSCOPE" in summary
    assert "2 modelled background" in summary and "3 not modelled" in summary


def test_paired_cell_types_can_colour_xenium_alone(controller):
    c, _ = controller
    _pair(c)
    states = tuple(
        V.CellTypeOverlayState(dataset=ds, segmentation=SOURCE)
        for ds in ("MERSCOPE", "XENIUM")
    )
    assignments = CellTypeAssignments(
        segmentation=SOURCE,
        table_key="pciseq_cells",
        instance_key="instance_id",
        cell_ids=np.array([3]),
        broad=np.array(["Astro"]),
        fine=np.array(["Astro"]),
        levels={"pciseq": np.array(["Astro"])},
    )
    c._cache_cell_type_data(states[1], SOURCE, assignments)
    c._paired_cell_type_states = lambda: states
    c._load_cell_type_data = lambda state, _: state.dataset == "XENIUM"
    c._paired_segmentation_for_cell_type = lambda ds, _: (
        object() if ds == "XENIUM" else None
    )
    seen = []
    c._apply_paired_cell_type_layers = lambda available: (
        seen.append(tuple(s.dataset for s in available)) or len(available)
    )
    c._populate_paired_cell_type_panel = lambda _: None
    c._set_paired_cell_type_kind("pciseq")
    assert seen == [("XENIUM",)]
    assert states[1].kind == "pciseq"


def test_old_umap_result_cannot_populate_new_analysis(controller):
    c, metadata = controller
    state = object()
    c._umap_state_obj = state
    metadata["analysis_id"] = "b" * 64
    # Any attempt to inspect views would fail: the stale result must exit first.
    c._apply_umap_load(
        state, SOURCE, None, {"source_identities": {"XENIUM": "a" * 64}, "views": {}}
    )


def test_switch_failure_removes_old_native_layers_and_indexes(controller):
    c, _ = controller
    del c._active_sdata.points[POINTS]
    removed = []
    c._remove_layers_by_prefix = removed.append
    c._cell_transcript_index["XENIUM"] = CellTranscriptIndex.empty()
    assert c.set_transcript_assignment_source("XENIUM", SOURCE)
    assert removed and "XENIUM" not in c._cell_transcript_index
    assert c._transcript_source_key("XENIUM") == SOURCE
    assert "missing" in c.statuses[-1]


def test_stale_cell_inspection_and_mask_overlay_results_are_discarded(controller):
    c, metadata = controller
    c._transcript_assignment_sources["XENIUM"] = SOURCE
    metadata["analysis_id"] = "b" * 64
    c._apply_cell_selection_payload(
        c._cell_inspection_generation,
        ("XENIUM", "1", SOURCE, "a" * 64),
        {},
        CellTranscriptIndex.empty(),
    )
    assert "XENIUM" not in c._cell_transcript_index
    c._cell_type_generation = 1
    sentinel = object()
    c._cell_type_worker = sentinel
    state = V.CellTypeOverlayState(dataset="XENIUM", segmentation=SOURCE)
    c._apply_cell_type_layer(1, state, {"source_key": SOURCE, "analysis_id": "a" * 64})
    assert c._cell_type_worker is sentinel


def test_pciseq_cell_annotation_uses_exact_uint64_id_and_own_probability(controller):
    c, _ = controller
    c._transcript_assignment_sources["XENIUM"] = SOURCE
    state = V.CellTypeOverlayState(dataset="XENIUM", segmentation=SOURCE)
    ids = np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64)
    assignments = CellTypeAssignments(
        segmentation=SOURCE,
        table_key="pciseq_cells",
        instance_key="instance_id",
        cell_ids=ids,
        broad=np.array(["Neuron", "Astro"]),
        fine=np.array(["Neuron", "Astro"]),
        levels={"pciseq": np.array(["Neuron", "Astro"])},
        probabilities=np.array([0.7, 0.95]),
        statuses=np.array(["assigned", "assigned"]),
        analysis_id="a" * 64,
    )
    c._cache_cell_type_data(state, SOURCE, assignments)
    c._cell_type_states = {"XENIUM": state}
    result = c._cell_type_annotation_for_cell("XENIUM", str(2**64 - 1))
    assert result["broad_cell_type"] == "Astro"
    assert result["annotation_method"] == "pciseq"
    assert result["pciseq_probability"] == 0.95
    assert "95.0%" in V._cell_summary_html(result)
    _, metadata = controller
    metadata["analysis_id"] = "b" * 64
    stale = c._cell_type_annotation_for_cell("XENIUM", str(2**64 - 1))
    assert stale["broad_cell_type"] == "Unannotated"


def test_assignment_selector_emits_only_user_changes():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    calls = []

    def noop(*_):
        pass

    widget = V.GeneInspectorWidget(
        noop,
        noop,
        noop,
        noop,
        noop,
        noop,
        set_assignment_source_callback=lambda *args: calls.append(args),
    )
    widget.set_assignment_sources(
        {"native": "Native assignments", SOURCE: "pciSeq · ProSeg hybrid"},
        SOURCE,
        "3 not modelled",
        dataset="XENIUM",
    )
    assert calls == []
    widget._assignment_combo.setCurrentIndex(0)
    assert calls == [("XENIUM", "native")]
    widget.close()
    assert app is not None


def test_paired_moving_platform_requires_exact_ids_for_alignment(
    controller, monkeypatch
):
    c, metadata = controller
    _pair(c)
    c._paired_session.contract.profile("MERSCOPE").points_key = "transcripts_aligned"
    store = c._paired_session.sdata["MERSCOPE"]
    store.points["transcripts_aligned"] = store.points["transcripts"].copy()
    monkeypatch.setattr(V, "assignment_source_points", lambda *_: dict(metadata))
    assert not c._open_paired_gene_inspector()
    assert "transcript_id" in c.statuses[-1]


@pytest.mark.parametrize("partitioned", [False, True])
def test_paired_pciseq_joins_assignments_onto_exact_aligned_transcript_ids(
    controller, monkeypatch, partitioned
):
    c, metadata = controller
    _pair(c)
    c.args.gene_max_render_points = 0
    c._paired_session.contract.profile("MERSCOPE").points_key = "transcripts_aligned"
    store = c._paired_session.sdata["MERSCOPE"]
    native = store.points[POINTS]
    native["transcript_id"] = np.array(
        [2**64 - 3, 2**64 - 2, 2**64 - 1], dtype=np.uint64
    )
    aligned = native[["x", "y", "gene", "transcript_id"]].copy()
    aligned["x"] = [101.0, 202.0, 303.0]
    aligned["y"] = [404.0, 505.0, 606.0]
    # A permuted, nonrigid aligned table rules out positional or affine guesses.
    store.points["transcripts_aligned"] = aligned.iloc[[2, 0, 1]].copy()
    if partitioned:
        import dask.dataframe as dd

        store.points[POINTS] = dd.from_pandas(native, npartitions=2, sort=False)
        store.points["transcripts_aligned"] = dd.from_pandas(
            store.points["transcripts_aligned"], npartitions=2, sort=False
        )
    monkeypatch.setattr(V, "assignment_source_points", lambda *_: dict(metadata))
    results = []
    c._apply_paired_gene_build = lambda _gen, _pair, payloads, _token: results.append(
        payloads
    )
    assert c._open_paired_gene_inspector()
    assert set(results[0]) == {"MERSCOPE", "XENIUM"}
    moving = results[0]["MERSCOPE"]
    assert moving["display_points_key"] == "transcripts_aligned"
    coords, genes = moving["store"].cell_transcript_index.transcripts_for(2**64 - 1)
    assert genes.tolist() == ["A"]
    np.testing.assert_array_equal(coords, [[404.0, 101.0]])
    np.testing.assert_array_equal(native.x, [1.0, 2.0, 3.0])
    assert moving["store"].source_status_counts == {
        "assigned": 1,
        "background": 1,
        "not_modelled": 1,
    }


def test_display_spool_is_removed_after_streaming_error(tmp_path, monkeypatch):
    import dask.dataframe as dd
    from tempfile import TemporaryDirectory

    created = []

    def temporary(**kwargs):
        context = TemporaryDirectory(dir=tmp_path, **kwargs)
        created.append(context.name)
        return context

    monkeypatch.setattr(V, "TemporaryDirectory", temporary)
    points = dd.from_pandas(
        pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}), npartitions=2
    )
    with pytest.raises(ValueError, match="stop after streaming"):
        with V.ComparisonViewerController._assignment_display_spool(points) as spooled:
            np.testing.assert_array_equal(spooled.compute().x, [1.0, 2.0])
            raise ValueError("stop after streaming")
    assert created and not any(Path(path).exists() for path in created)


def test_paired_aligned_cache_hit_bypasses_join_and_spool(
    controller, monkeypatch, tmp_path
):
    from napari_compare_xenium_merscope import transcript_cache as TC

    c, metadata = controller
    _pair(c)
    c.args.gene_max_render_points = 0
    c._paired_session.contract.profile("MERSCOPE").points_key = "transcripts_aligned"
    store = c._paired_session.sdata["MERSCOPE"]
    store.points[POINTS]["transcript_id"] = np.array([1, 2, 3], dtype=np.uint64)
    display = store.points[POINTS][["x", "y", "gene", "transcript_id"]].copy()
    display.x += 100.0
    store.points["transcripts_aligned"] = display
    for key in (POINTS, "transcripts_aligned"):
        root = c.datasets["MERSCOPE"].zarr_path / "points" / key
        root.mkdir(parents=True)
        (root / "part.0.parquet").write_bytes(b"source fingerprint")
    monkeypatch.setenv(TC.TRANSCRIPT_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(V, "assignment_source_points", lambda *_: dict(metadata))
    monkeypatch.setattr(
        TC,
        "resolve_cell_type_marker_reference",
        lambda *_: MarkerReferenceResolution(reference=None),
    )
    results = []
    c._apply_paired_gene_build = lambda _gen, _pair, payloads, _token: results.append(
        payloads
    )
    assert c._open_paired_gene_inspector()
    assert results[0]["MERSCOPE"].get("cache_saved")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cache hit must not repeat the exact-ID join or spool")

    monkeypatch.setattr(V, "assignment_points_in_display_frame", forbidden)
    monkeypatch.setattr(c, "_assignment_display_spool", forbidden)
    assert c._open_paired_gene_inspector()
    assert results[1]["MERSCOPE"]["cache_hit"]
    coords, _ = results[1]["MERSCOPE"]["store"].cell_transcript_index.transcripts_for(
        2**64 - 1
    )
    np.testing.assert_array_equal(coords, [[4.0, 101.0]])
