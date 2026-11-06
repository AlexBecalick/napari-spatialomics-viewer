"""ATERA cell inspector without a global index (milestone M8).

A cell click paints the count-table row first, then the cell's exact assigned
transcripts from a two-phase read of only its tiles.  Pies, gene lists and
links are capped at WTA scale; below the caps the output is unchanged.

Everything runs on the synthetic store from ``atera_stores``; no test reads
``/Users/becalia/data``.  Viewers are napari ``ViewerModel`` instances, never
vispy canvases.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from threading import Event
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atera_stores
import numpy as np
import pyarrow.parquet as pq
import pytest

from napari_compare_xenium_merscope import atera_transcripts as T
from napari_compare_xenium_merscope import transcript_cache as TC
from napari_compare_xenium_merscope import utils as U
from napari_compare_xenium_merscope import viewer as V

PS = 0.5


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return atera_stores.build_atera_latest_store(tmp_path_factory.mktemp("atera_m8"), pixel_size=PS)


@pytest.fixture
def store_copy(atera_store, tmp_path) -> Path:
    dst = tmp_path / "copy" / atera_store.name
    shutil.copytree(atera_store, dst)
    return dst


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR", str(tmp_path / "transcript-cache"))
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


@pytest.fixture(scope="module")
def kept() -> "object":
    """Every transcript the store keeps, with its tile and cell (NA -> 0)."""
    tx = atera_stores.synthetic_transcripts()
    tx = tx[tx["keep"]].copy()
    tx["cell"] = tx["original_assignment"].fillna(0).astype(np.int64)
    return tx


def _expected(kept, cell: int):
    rows = kept[kept["cell"] == cell]
    names = [atera_stores.FEATURE_DICTIONARY[c] for c in rows["code"]]
    return sorted(zip(rows["x"].astype(np.float32), rows["y"].astype(np.float32), names))


def _got(engine, rows):
    names = [engine.index.vocabulary[c] for c in rows.code]
    return sorted(zip(rows.x, rows.y, names))


def _bounds(cell: int):
    _cid, xmin, ymin, xmax, ymax = next(c for c in atera_stores.CELLS if c[0] == cell)
    return (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# Engine: cell -> tile map and the two-phase read
# ---------------------------------------------------------------------------


def test_cell_tile_map_from_the_overview_matches_the_standalone_pass(atera_store, kept):
    engine = T.AteraTranscriptEngine.open(atera_store)
    overview = engine.build_overview(5.0)
    cell_tiles = overview.cell_tiles
    assert cell_tiles is not None and cell_tiles.validate(engine.index)
    standalone = engine.build_cell_tile_map()
    assert np.array_equal(standalone.cell_ids, cell_tiles.cell_ids)
    assert np.array_equal(standalone.cell_ptr, cell_tiles.cell_ptr)
    assert np.array_equal(standalone.tile_index, cell_tiles.tile_index)
    for cid, *_rect in atera_stores.CELLS:
        expected = set(kept.loc[kept["cell"] == cid, "tile"])
        assert set(cell_tiles.tiles_for(cid)) == expected
    # Cell 3 straddles the x = 20 um tile boundary: its transcripts sit in two tiles.
    assert set(cell_tiles.tiles_for(3)) == {"0,0", "1,0"}
    assert cell_tiles.n_straddling == 1
    assert cell_tiles.tiles_for(999) == ()
    assert 0 not in set(cell_tiles.cell_ids.tolist())


@pytest.mark.parametrize("use_map", [True, False])
def test_read_cell_returns_exactly_the_assigned_rows(atera_store, kept, use_map):
    engine = T.AteraTranscriptEngine.open(atera_store)
    cell_tiles = engine.build_cell_tile_map() if use_map else None
    for cid, *_rect in atera_stores.CELLS:
        rows = engine.read_cell(cid, None if use_map else _bounds(cid), cell_tiles=cell_tiles)
        assert rows is not None and rows.used_cell_tiles is use_map
        assert _got(engine, rows) == _expected(kept, cid), cid
    straddler = engine.read_cell(3, None if use_map else _bounds(3), cell_tiles=cell_tiles)
    assert set(straddler.tiles) == {"0,0", "1,0"}
    assert straddler.n == int((kept["cell"] == 3).sum())


def test_read_cell_never_returns_unassigned_rows(atera_store, kept):
    engine = T.AteraTranscriptEngine.open(atera_store)
    assert (kept["cell"] == 0).any()  # the store does hold unassigned (NA) rows
    for cell in (0, -1, "not-a-cell"):
        rows = engine.read_cell(cell, (0.0, 0.0, 40.0, 40.0))
        assert rows is None or rows.n == 0
    # A cell with no assigned transcripts: the map proves it without reading.
    cell_tiles = engine.build_cell_tile_map()
    rows = engine.read_cell(999, cell_tiles=cell_tiles)
    assert rows.n == 0 and rows.files_read == 0


def test_phase_two_reads_only_row_groups_with_hits(atera_store, kept, monkeypatch):
    engine = T.AteraTranscriptEngine.open(atera_store)
    original = T._read_row_group
    calls = []

    def spy(parquet_file, rg, columns):
        calls.append((parquet_file, int(rg), tuple(columns)))
        return original(parquet_file, rg, columns)

    monkeypatch.setattr(T, "_read_row_group", spy)
    rows = engine.read_cell(3, _bounds(3))
    phase1 = [c for c in calls if c[2] == (T.ASSIGNMENT_COLUMN,)]
    phase2 = [c for c in calls if c[2] != (T.ASSIGNMENT_COLUMN,)]
    assert phase1 and phase2
    assert all(set(c[2]) == {T.X_COLUMN, T.Y_COLUMN, T.GENE_COLUMN} for c in phase2)
    assert T.ASSIGNMENT_COLUMN not in {col for c in phase2 for col in c[2]}
    hits_total = 0
    for parquet_file, rg, _cols in phase2:
        ids = original(parquet_file, rg, [T.ASSIGNMENT_COLUMN]).column(0).to_pylist()
        hits = sum(1 for v in ids if v == 3)
        assert hits > 0  # never a phase-2 read of a row group without the cell
        hits_total += hits
    assert hits_total == rows.n
    assert rows.row_groups_read == len(phase2) <= rows.row_groups_scanned == len(phase1)


def test_a_changed_file_makes_the_read_stale(store_copy):
    engine = T.AteraTranscriptEngine.open(store_copy)
    assert engine.read_cell(1, _bounds(1)) is not None
    touched = next(f for f in engine.index.files if f.tile == "0,0")
    st = os.stat(touched.path)
    os.utime(touched.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert engine.read_cell(1, _bounds(1)) is None
    assert T.indexed_cell_transcripts(engine, 1, _bounds(1)) is None
    # Files of other tiles are unaffected.
    assert engine.read_cell(7, _bounds(7)) is not None


def test_a_map_from_other_points_is_ignored(atera_store, kept):
    engine = T.AteraTranscriptEngine.open(atera_store)
    good = engine.build_cell_tile_map()
    stale = T.CellTileMap(
        tiles=good.tiles, cell_ids=good.cell_ids, cell_ptr=good.cell_ptr,
        tile_index=good.tile_index, fingerprint={"n_files": 1, "n_rows": 1},
    )
    assert not stale.validate(engine.index)
    rows = engine.read_cell(3, _bounds(3), cell_tiles=stale)  # falls back to the box
    assert rows is not None and not rows.used_cell_tiles
    assert _got(engine, rows) == _expected(kept, 3)
    assert engine.read_cell(3, None, cell_tiles=stale) is None


def test_an_index_less_store_has_no_exact_cell_reads(tmp_path):
    store = atera_stores.build_atera_latest_store(tmp_path, with_index=False)
    engine = T.AteraTranscriptEngine.open(store)
    assert engine is not None and not engine.index.validated
    assert engine.read_cell(1, _bounds(1)) is None
    assert engine.build_cell_tile_map() is None
    assert engine.build_overview(5.0).cell_tiles is None


def test_indexed_cell_transcripts_builds_a_cell_index(atera_store, kept):
    engine = T.AteraTranscriptEngine.open(atera_store)
    index = T.indexed_cell_transcripts(engine, 3, _bounds(3))
    coords, genes = index.transcripts_for(3)
    expected = _expected(kept, 3)
    assert len(genes) == len(expected)
    assert sorted(zip(coords[:, 1], coords[:, 0], genes.tolist())) == expected
    assert index.transcripts_for(4)[1].size == 0


def test_cell_tiles_persist_in_the_overview_cache_and_a_sidecar(atera_store, tmp_path):
    cache = tmp_path / "c"
    engine = T.AteraTranscriptEngine.open(atera_store)
    overview = engine.build_overview(5.0)
    assert TC.save_atera_overview(engine.index, overview, cache_root=cache) is not None
    restored = TC.load_atera_overview(engine.index, bin_um=5.0, cache_root=cache)
    assert restored.cell_tiles is not None
    assert np.array_equal(restored.cell_tiles.cell_ids, overview.cell_tiles.cell_ids)
    assert restored.cell_tiles.tiles_for(3) == overview.cell_tiles.tiles_for(3)
    sidecar = TC.load_atera_cell_tiles(engine.index, cache_root=cache)
    assert sidecar is not None and sidecar.tiles_for(3) == overview.cell_tiles.tiles_for(3)
    assert sidecar.fingerprint == engine.index.fingerprint


# ---------------------------------------------------------------------------
# Utils: table rows and the cell index builder
# ---------------------------------------------------------------------------


def test_table_counts_match_the_csr_row(atera_store):
    import anndata as ad

    adata = ad.read_zarr(atera_store / "tables" / "table_original")
    matrix = adata.X.tocsr()
    names = list(adata.var_names)
    for row, cid in enumerate(adata.obs["instance_id"].to_numpy()):
        counts = U.table_cell_gene_counts(atera_store, "table_original", int(cid))
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        expected = {names[j]: int(v) for j, v in zip(matrix.indices[start:end], matrix.data[start:end]) if v}
        assert dict(counts.ranked()) == expected
        assert counts.total == sum(expected.values())
        assert list(counts.counts) == sorted(counts.counts, reverse=True)
    assert U.table_cell_gene_counts(atera_store, "table_original", 999) is None
    assert U.table_cell_gene_counts(atera_store, "table_original", "1") is not None


@pytest.mark.parametrize("dense", [False, True])
def test_table_counts_handle_sparse_ids_and_exclude_controls(tmp_path, dense):
    import anndata as ad
    import pandas as pd
    import spatialdata as sd
    from scipy import sparse
    from spatialdata.models import TableModel

    ids = np.array([5, 900, 2**40 + 3], dtype=np.uint64)  # non-contiguous, beyond int32
    var = ["GENE_A", "NegControlProbe_00002", "GENE_B", "UnassignedCodeword_0001"]
    x = np.array([[2, 1, 0, 0], [0, 4, 7, 1], [3, 0, 3, 0]], dtype=np.float32)
    obs = pd.DataFrame(
        {"cell": ids, "region": pd.Categorical(["cells"] * 3)},
        index=[f"c{i}" for i in range(3)],
    )
    table = TableModel.parse(
        ad.AnnData(X=x if dense else sparse.csr_matrix(x), obs=obs, var=pd.DataFrame(index=var)),
        region="cells", region_key="region", instance_key="cell",
    )
    store = tmp_path / "t.zarr"
    sd.SpatialData(tables={"counts": table}).write(store)
    counts = U.table_cell_gene_counts(store, "counts", 900)
    assert counts.ranked() == [("GENE_B", 7)]  # controls dropped
    assert U.table_cell_gene_counts(store, "counts", 2**40 + 3).ranked() == [("GENE_A", 3), ("GENE_B", 3)]
    assert U.table_cell_gene_counts(store, "counts", 5).ranked() == [("GENE_A", 2)]
    assert U.table_cell_gene_counts(store, "counts", 6) is None


def test_cell_transcript_index_from_arrays_groups_exact_uint64_ids():
    big = np.uint64(2**63 + 11)
    cells = np.array([big, 2, big, 2, 2], dtype=np.uint64)
    x = np.arange(5, dtype=np.float32)
    y = x + 10
    codes = np.array([3, 0, -1, 3, 1])
    index = U.cell_transcript_index_from_arrays(cells, x, y, codes, ["A", "B", "C", "D"])
    assert index.gene_names == ("A", "B", "D")
    coords, genes = index.transcripts_for(2**63 + 11)
    assert genes.tolist() == ["D"] and coords.tolist() == [[10.0, 0.0]]
    coords, genes = index.transcripts_for("2")
    assert genes.tolist() == ["A", "D", "B"]
    assert U.cell_transcript_index_from_arrays([], [], [], [], ["A"]).slices == {}


# ---------------------------------------------------------------------------
# Controller on ATERA
# ---------------------------------------------------------------------------


def _args(**overrides):
    defaults = dict(
        background_io_workers=1, session_cache_gb=0, image_pyramid_downsample=4,
        overwrite_derived_caches=False, overwrite_labels=False, skip_images=True,
        skip_transcripts=True, segmentation_source="shapes", hide_images=False,
        hide_shapes=False, image_opacity=1.0, visible_channels=None, shape_opacity=0.5,
        label_contour_width=1, merscope_transform_path=None, xenium_spec_path=None,
        atera_spec_path=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _activate(store, monkeypatch):
    from napari.components import ViewerModel

    ctrl = V.ComparisonViewerController(ViewerModel(), {}, _args())
    cfg = ctrl._build_dataset_config("ATERA", store)
    ctrl.datasets = {"ATERA": cfg}
    session = ctrl._prepare_dataset_session("ATERA", cfg, Event())
    ctrl.active_dataset = "ATERA"
    ctrl._active_sdata = session.sdata
    ctrl._active_images_sdata = session.images_sdata
    ctrl._x_transform = session.x_transform
    ctrl._y_transform = session.y_transform
    ctrl._segmentation_keys = list(session.segmentation_keys)
    ctrl._dataset_sessions["ATERA"] = session
    monkeypatch.setattr(V, "thread_worker", None)
    statuses: list[str] = []
    ctrl.set_status_callback(statuses.append)
    ctrl.statuses = statuses
    return ctrl


def _forbid_scans(ctrl, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("ATERA cell inspection must not scan the store")

    import dask.base

    monkeypatch.setattr(V, "build_gene_point_groups", forbidden)
    monkeypatch.setattr(V, "build_cell_transcript_index", forbidden)
    monkeypatch.setattr(U, "build_gene_point_groups", forbidden)
    monkeypatch.setattr(ctrl, "_resolve_points_columns_for", forbidden)
    monkeypatch.setattr(ctrl, "_resolve_gene_points_columns", forbidden)
    monkeypatch.setattr(dask.base.DaskMethodsMixin, "compute", forbidden)
    monkeypatch.setattr(dask.base, "compute", forbidden)


def test_atera_inspection_defaults_to_the_original_cells(qapp, atera_store, monkeypatch):
    ctrl = _activate(atera_store, monkeypatch)
    assert ctrl._inspection_source("ATERA") == "original"
    assert ctrl._inspection_shape("ATERA") == "atera_cell_boundaries"
    assert ctrl._cell_inspector_shape_key() == "atera_cell_boundaries"
    assert ctrl._lazy_cell_counts_table("ATERA") == "table_original"


def test_cell_click_paints_table_counts_then_the_indexed_transcripts(qapp, atera_store, kept, monkeypatch):
    ctrl = _activate(atera_store, monkeypatch)
    _forbid_scans(ctrl, monkeypatch)
    applied = []
    original = ctrl._apply_cell_selection_payload

    def record(generation, worker_key, payload, index, status=None):
        applied.append(dict(payload))
        return original(generation, worker_key, payload, index, status=status)

    monkeypatch.setattr(ctrl, "_apply_cell_selection_payload", record)
    gdf = ctrl._active_sdata.shapes["atera_cell_boundaries"]
    cell_id, geometry = ctrl._shape_for_cell_id(gdf, 3)
    ctrl._add_cell_selection("ATERA", cell_id, geometry)

    assert [p["transcript_source"] for p in applied] == ["table", "indexed"]
    assert all(p["wta_caps"] is True for p in applied)
    table, indexed = applied
    table_counts = U.table_cell_gene_counts(atera_store, "table_original", 3)
    assert [(r["gene"], r["count"]) for r in table["gene_rows"]] == table_counts.ranked()
    assert table["transcripts_loading"] is True and len(table["coords_yx"]) == 0
    assert table["total"] == table_counts.total

    rows = kept[kept["cell"] == 3]
    is_control = ~rows["is_gene"].to_numpy()
    assert indexed["total"] == int((~is_control).sum())
    assert indexed["control_total"] == int(is_control.sum())
    assert all(not U.is_control_gene(r["gene"]) for r in indexed["gene_rows"])
    assert {r["gene"] for r in indexed["control_rows"]} == {
        atera_stores.FEATURE_DICTIONARY[c] for c in rows["code"][is_control]
    }
    assert len(indexed["coords_yx"]) == len(rows)
    # The real genes agree with the count table (X holds the assigned genes).
    assert sorted((r["gene"], r["count"]) for r in indexed["gene_rows"]) == sorted(table_counts.ranked())
    entry = ctrl._selected_cells["ATERA"][0]
    assert entry["transcript_source"] == "indexed" and not entry["transcripts_loading"]
    links = ctrl._get_layer_by_name(V.CELL_INSPECTOR_LINKS_LAYER)
    assert links is not None and len(links.data) == len(rows)
    assert any("table counts shown" in s for s in ctrl.statuses)
    html = V._cell_summary_html(entry)
    assert "Gene transcripts:" in html and "Control features:" in html


def test_inspection_index_reuses_the_loaded_engine_and_map(qapp, atera_store, monkeypatch):
    ctrl = _activate(atera_store, monkeypatch)
    engine = T.AteraTranscriptEngine.open(atera_store)
    overview = engine.build_overview(5.0)
    ctx = SimpleNamespace(engine=engine, overview=overview)
    ctrl._gene_inspector_states["ATERA"] = SimpleNamespace(atera=ctx, gene_visuals={}, store=None)
    monkeypatch.setattr(T.AteraTranscriptEngine, "open", classmethod(lambda *_a, **_k: pytest.fail("reopened")))
    seen = []
    original = engine.read_cell

    def spy(cell_id, bounds_um=None, *, cell_tiles=None, cancel=None):
        seen.append((bounds_um, cell_tiles))
        return original(cell_id, bounds_um, cell_tiles=cell_tiles, cancel=cancel)

    monkeypatch.setattr(engine, "read_cell", spy)
    index = ctrl._inspection_transcript_index("ATERA", 3, None, ctrl._empty_cell_transcript_index())
    assert index.transcripts_for(3)[1].size > 0
    assert seen[-1][1] is overview.cell_tiles
    # Cached per cell: a second click does not read again.
    ctrl._inspection_transcript_index("ATERA", 3, None, ctrl._empty_cell_transcript_index())
    assert len(seen) == 1


def test_without_an_exact_read_the_table_counts_stay(qapp, tmp_path, monkeypatch):
    store = atera_stores.build_atera_latest_store(tmp_path, with_index=False)
    ctrl = _activate(store, monkeypatch)
    _forbid_scans(ctrl, monkeypatch)
    gdf = ctrl._active_sdata.shapes["atera_cell_boundaries"]
    cell_id, geometry = ctrl._shape_for_cell_id(gdf, 2)
    payload = ctrl._cell_selection_display("ATERA", cell_id, geometry, "#ffffff")
    assert payload["transcript_source"] == "table" and not payload["transcripts_loading"]
    assert payload["wta_caps"] is True
    assert payload["gene_rows"] and "unavailable" in payload["transcript_note"]


@pytest.mark.parametrize("failure", ["exact_read", "common", "memory"])
def test_a_failed_exact_read_keeps_the_table_counts(qapp, atera_store, monkeypatch, failure):
    ctrl = _activate(atera_store, monkeypatch)
    _forbid_scans(ctrl, monkeypatch)

    def boom(*_a, **_k):
        if failure == "memory":
            raise MemoryError("out of memory")
        raise RuntimeError("phase two failed")

    if failure == "common":
        # An image-statistics error alone must not wipe the table counts.
        monkeypatch.setattr(ctrl, "_cell_selection_common", boom)
    else:
        monkeypatch.setattr(V, "indexed_cell_transcripts", boom)
    gdf = ctrl._active_sdata.shapes["atera_cell_boundaries"]
    cell_id, geometry = ctrl._shape_for_cell_id(gdf, 3)
    ctrl._add_cell_selection("ATERA", cell_id, geometry)

    entry = ctrl._selected_cells["ATERA"][0]
    table_counts = U.table_cell_gene_counts(atera_store, "table_original", 3)
    assert [(r["gene"], r["count"]) for r in entry["gene_rows"]] == table_counts.ranked()
    assert entry["transcript_source"] == "table" and entry["wta_caps"] is True
    assert not entry.get("inspection_error")
    assert entry["transcripts_loading"] is False and entry["intensities_loading"] is False
    assert "could not be read" in entry["transcript_note"]
    assert not ctrl._cell_inspection_workers
    summary = V._cell_summary_html(entry)
    assert "Reading the assigned transcripts" not in summary
    assert "could not be read; showing the table counts" in summary
    genes = V._cell_gene_list_html(entry)
    assert "could not be loaded" not in genes and table_counts.ranked()[0][0] in genes
    assert any("showing the table counts" in s for s in ctrl.statuses)


def test_a_failed_table_read_still_reports_an_inspection_error(qapp, atera_store, monkeypatch):
    ctrl = _activate(atera_store, monkeypatch)
    _forbid_scans(ctrl, monkeypatch)
    monkeypatch.setattr(ctrl, "_lazy_cell_table_display",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("table failed")))
    gdf = ctrl._active_sdata.shapes["atera_cell_boundaries"]
    cell_id, geometry = ctrl._shape_for_cell_id(gdf, 3)
    ctrl._add_cell_selection("ATERA", cell_id, geometry)
    entry = ctrl._selected_cells["ATERA"][0]
    assert "table failed" in entry["inspection_error"]


# ---------------------------------------------------------------------------
# WTA-scale presentation (size-gated)
# ---------------------------------------------------------------------------


def _rows(n):
    return [
        {"gene": f"G{i:05d}", "count": n - i, "rgba": (0.2, 0.4, 0.6, 1.0), "glyph": "●"}
        for i in range(n)
    ]


def test_a_wta_cell_is_capped_in_the_pie_list_and_links(qapp):
    rows = _rows(6_000)
    slices = V._cell_pie_slices(rows, V.CELL_PIE_MAX_SLICES)
    assert len(slices) == V.CELL_PIE_MAX_SLICES + 1
    other = slices[-1]
    assert other["rgba"] == V.CELL_OTHER_SLICE_RGBA and other["other_genes"] == 6_000 - V.CELL_PIE_MAX_SLICES
    assert sum(s["count"] for s in slices) == sum(r["count"] for r in rows)

    html = V._cell_gene_list_html({"gene_rows": rows, "wta_caps": True})
    assert html.count("<b>G") == V.CELL_GENE_LIST_MAX_ROWS
    assert f"{6_000 - V.CELL_GENE_LIST_MAX_ROWS:,} more genes" in html

    coords = np.column_stack([np.arange(17_202.0), np.arange(17_202.0)])
    entry = {"centroid_yx": (1.0, 2.0), "coords_yx": coords, "wta_caps": True}
    segments = V.ComparisonViewerController._cell_link_segments(entry)
    assert segments.shape == (V.CELL_LINKS_MAX_PER_CELL, 2, 2)
    assert np.array_equal(segments, V.ComparisonViewerController._cell_link_segments(entry))
    assert segments[0, 0].tolist() == [0.0, 0.0] and segments[-1, 0].tolist() == [17_201.0, 17_201.0]
    assert np.all(segments[:, 1] == [1.0, 2.0])
    summary = V._cell_summary_html({"total": 17_202, "coords_yx": coords, "wta_caps": True})
    assert "Links drawn: 2,000 of 17,202" in summary

    panel = V._CellPanel({"cell_id": 1, "gene_rows": rows, "coords_yx": coords, "wta_caps": True})
    assert len(panel._pie._slices) == V.CELL_PIE_MAX_SLICES + 1


def test_a_large_non_lazy_cell_keeps_every_slice_row_and_link(qapp):
    """MERSCOPE/XENIUM cells (no ``wta_caps``) render as before the caps:
    one pie slice per gene, every gene row and one link per transcript."""
    n_genes = V.CELL_PIE_MAX_SLICES + 164  # P7513 MERSCOPE max is 188 genes
    rows = _rows(n_genes)
    big_rows = _rows(V.CELL_GENE_LIST_MAX_ROWS + 50)
    n_links = V.CELL_LINKS_MAX_PER_CELL + 421  # P7513 MERSCOPE max is 2,421
    coords = np.column_stack([np.arange(float(n_links)), np.arange(float(n_links))])
    cell = {"cell_id": 7, "gene_rows": rows, "coords_yx": coords,
            "centroid_yx": (1.0, 2.0), "total": n_links}

    expected = [
        {"count": r["count"], "glyph": r["glyph"], "rgba": U.darken_rgba(r["rgba"], V.CELL_PIE_SLICE_DARKEN)}
        for r in rows
    ]
    assert V._cell_pie_slices(rows) == expected
    panel = V._CellPanel(cell)
    assert panel._pie._slices == expected
    assert all(s["rgba"] != V.CELL_OTHER_SLICE_RGBA for s in panel._pie._slices)

    html = V._cell_gene_list_html({"gene_rows": big_rows})
    assert html.count("<b>G") == len(big_rows) and "more genes" not in html

    segments = V.ComparisonViewerController._cell_link_segments(cell)
    assert segments.shape == (n_links, 2, 2)
    assert np.array_equal(segments[:, 0], coords)
    assert "Links drawn" not in V._cell_summary_html(cell)


def test_a_small_cell_renders_exactly_as_before(qapp):
    rows = _rows(20)
    slices = V._cell_pie_slices(rows)
    assert slices == [
        {"count": r["count"], "glyph": r["glyph"], "rgba": U.darken_rgba(r["rgba"], V.CELL_PIE_SLICE_DARKEN)}
        for r in rows
    ]
    html = V._cell_gene_list_html({"gene_rows": rows})
    assert "more genes" not in html and html.count("<b>G") == 20
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    segments = V.ComparisonViewerController._cell_link_segments({"centroid_yx": (0.0, 0.5), "coords_yx": coords})
    assert [s.tolist() for s in segments] == [[[1.0, 2.0], [0.0, 0.5]], [[3.0, 4.0], [0.0, 0.5]]]
    summary = V._cell_summary_html({"total": 2, "coords_yx": coords})
    assert "Links drawn" not in summary and "Control features" not in summary
    assert "<b>Total transcripts:</b> 2" in summary


# ---------------------------------------------------------------------------
# Real store (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("NAPARI_COMPARE_ATERA_STORE"),
    reason="set NAPARI_COMPARE_ATERA_STORE to a real Atera store to run",
)
def test_real_store_cell_1001_counts_and_transcripts():
    store = Path(os.environ["NAPARI_COMPARE_ATERA_STORE"])
    counts = U.table_cell_gene_counts(store, "table_original", 1001)
    assert counts is not None and counts.total == 17_199
    engine = T.AteraTranscriptEngine.open(store)
    import spatialdata as sd

    shapes = sd.read_zarr(store, selection=("shapes",)).shapes["atera_cell_boundaries"]
    rows = engine.read_cell(1001, shapes.geometry.loc[1001].bounds)
    assert rows is not None and rows.n == 17_202


def test_an_overview_cached_without_the_map_gets_it_once(atera_store, tmp_path, monkeypatch):
    """Overviews cached before M8 lack the map: the load builds and persists it once."""
    from napari_compare_xenium_merscope import atera_viewer as AV

    engine = T.AteraTranscriptEngine.open(atera_store)
    progress = []

    def report(phase):
        return lambda done, n: progress.append((phase, done, n))

    restore = AV.AteraTranscriptMixin._atera_restore_cell_tiles
    first = restore(None, "ATERA", engine, Event(), report)
    assert first is not None and set(first.tiles_for(3)) == {"0,0", "1,0"}
    assert progress and progress[-1][0] == "indexing cells"
    assert TC.load_atera_cell_tiles(engine.index) is not None
    monkeypatch.setattr(T.AteraTranscriptEngine, "build_cell_tile_map", lambda *_a, **_k: pytest.fail("rebuilt"))
    second = restore(None, "ATERA", engine, Event(), report)
    assert np.array_equal(second.cell_ids, first.cell_ids)
