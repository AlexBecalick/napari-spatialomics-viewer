"""ATERA lazy transcript engine (milestone M6): index, code-range map, reads, overview, cache.

Everything runs on the synthetic store from ``atera_stores`` (or tiny hand-written
parquet stores); the only real-data test is gated on ``NAPARI_COMPARE_ATERA_STORE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import atera_stores as A
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_compare_xenium_merscope import atera_transcripts as T
from napari_compare_xenium_merscope import transcript_cache as TC
from napari_compare_xenium_merscope.utils import TranscriptBuildCancelled

COLUMNS = ("x", "y", "feature_name", "original_assignment", "transcript_id")
GENE_CODES = {name: code for code, name in enumerate(A.FEATURE_DICTIONARY)}


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return A.build_atera_latest_store(tmp_path_factory.mktemp("atera_m6"))


@pytest.fixture
def store_copy(atera_store, tmp_path) -> Path:
    dst = tmp_path / "copy" / atera_store.name
    shutil.copytree(atera_store, dst)
    return dst


def _index_entries(store: Path) -> list[dict]:
    raw = json.loads((store / T.ATERA_POINTS_INDEX_JSON).read_text(encoding="utf-8"))
    return raw["points"]["transcripts"]["files"]


def _truth(store: Path) -> pd.DataFrame:
    """Every row in index file order, with names decoded and ids filled with 0."""
    frames = []
    directory = store / "points" / "transcripts" / "points.parquet"
    for position, entry in enumerate(_index_entries(store)):
        table = pq.read_table(directory / entry["file"])
        names = table.column("feature_name").to_pylist()
        ids = table.column("original_assignment").to_pylist()
        frames.append(
            pd.DataFrame(
                {
                    "file": position,
                    "tile": entry["tile"],
                    "x": table.column("x").to_numpy(),
                    "y": table.column("y").to_numpy(),
                    "name": names,
                    "code": [GENE_CODES[n] for n in names],
                    "original_assignment": [0 if v is None else int(v) for v in ids],
                    "transcript_id": table.column("transcript_id").to_numpy(),
                    "is_gene": table.column("is_gene").to_numpy(zero_copy_only=False),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _assert_rows_equal(got: dict[str, np.ndarray], expected: pd.DataFrame) -> None:
    assert len(got["x"]) == len(expected)
    np.testing.assert_array_equal(got["x"], expected["x"].to_numpy(np.float32))
    np.testing.assert_array_equal(got["y"], expected["y"].to_numpy(np.float32))
    if "code" in got:
        np.testing.assert_array_equal(got["code"].astype(np.int64), expected["code"].to_numpy(np.int64))
    if "original_assignment" in got:
        np.testing.assert_array_equal(
            got["original_assignment"].astype(np.uint64), expected["original_assignment"].to_numpy(np.uint64)
        )
    if "transcript_id" in got:
        np.testing.assert_array_equal(got["transcript_id"], expected["transcript_id"].to_numpy(np.uint64))


class _PathSpy:
    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, int]] = []
        self.rows = 0
        original = T._read_row_group
        engine_parquet = T.AteraTranscriptEngine._parquet

        def parquet(engine, position):
            handle = engine_parquet(engine, position)
            handle._spy_name = engine.index.files[position].name
            return handle

        def spy(parquet_file, row_group, columns):
            table = original(parquet_file, row_group, columns)
            self.calls.append((getattr(parquet_file, "_spy_name", "?"), int(row_group)))
            self.rows += table.num_rows
            return table

        monkeypatch.setattr(T.AteraTranscriptEngine, "_parquet", parquet)
        monkeypatch.setattr(T, "_read_row_group", spy)

    def reset(self) -> None:
        self.calls.clear()
        self.rows = 0


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_loads_in_index_file_order_with_one_shared_dictionary(atera_store):
    index = T.AteraPointsIndex.load(atera_store)
    assert index is not None and index.validated
    names = [f.name for f in index.files]
    assert names == [e["file"] for e in _index_entries(atera_store)]
    # Natural part order, which differs from a lexicographic directory listing.
    assert names != sorted(names)
    assert names.index("part.10.parquet") > names.index("part.2.parquet")
    assert index.vocabulary == A.FEATURE_DICTIONARY
    assert index.shared_dictionary and all(lut is None for lut in index.file_luts)
    assert index.code_dtype == np.int16
    assert index.assignment_dtype == np.uint32
    assert index.fingerprint["n_files"] == len(index.files)
    assert index.fingerprint["n_rows"] == index.n_rows
    assert index.tiles == A.TILE_NAMES
    off_grid = index.files[0]
    assert off_grid.tile == "-1,1" and off_grid.bbox_um[0] < 0 and off_grid.block_bounds_um is None


@pytest.mark.parametrize("breakage", ["n_files", "n_rows", "missing_file", "footer_rows", "footer_row_groups"])
def test_index_validation_returns_none_and_the_engine_falls_back(store_copy, breakage):
    index_path = store_copy / T.ATERA_POINTS_INDEX_JSON
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    element = raw["points"]["transcripts"]
    if breakage == "n_files":
        element["fingerprint"]["n_files"] += 1
    elif breakage == "n_rows":
        element["fingerprint"]["n_rows"] += 1
    elif breakage == "missing_file":
        (store_copy / element["directory"] / element["files"][3]["file"]).unlink()
    elif breakage == "footer_rows":
        element["files"][2]["n_rows"] += 1
        element["fingerprint"]["n_rows"] += 1
    elif breakage == "footer_row_groups":
        element["files"][2]["n_row_groups"] += 1
    index_path.write_text(json.dumps(raw), encoding="utf-8")

    assert T.AteraPointsIndex.load(store_copy) is None
    engine = T.AteraTranscriptEngine.open(store_copy)
    assert engine is not None and not engine.index.validated
    assert not engine.window_reads_available
    assert engine.read_window((0, 0, 10, 10)) is None
    with pytest.raises(RuntimeError):
        engine.read_tile("0,0")
    names = [f.name for f in engine.index.files]
    assert names == sorted(names, key=T._natural_key)
    if breakage != "missing_file":
        truth = _truth(store_copy)
        got = engine.read_genes(["GFAP"], COLUMNS)["GFAP"]
        # Scanned order is the natural part order, as in the index.
        _assert_rows_equal(got, truth[truth["name"] == "GFAP"])
        overview = engine.build_overview(bin_um=5.0)
        assert int(overview.counts.sum()) == len(truth)


def test_stale_index_fixture_is_rejected(tmp_path):
    store = A.build_atera_latest_store(tmp_path, stale_index=True)
    assert T.AteraPointsIndex.load(store) is None
    assert T.AteraPointsIndex.scan(store) is not None


# ---------------------------------------------------------------------------
# Gene reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_map", [True, False])
@pytest.mark.parametrize(
    "genes",
    [["GFAP"], ["ACTB", "MBP", "ZIC1", "NegControlProbe_00002", A.ABSENT_GENE]],
)
def test_read_genes_equals_brute_force(atera_store, genes, with_map):
    engine = T.AteraTranscriptEngine.open(atera_store)
    if with_map:
        assert engine.build_code_map() is not None
    truth = _truth(atera_store)
    result = engine.read_genes(genes + ["NOT_A_GENE"], COLUMNS)
    assert list(result) == genes + ["NOT_A_GENE"]
    for gene in genes:
        _assert_rows_equal(result[gene], truth[truth["name"] == gene])
    if A.ABSENT_GENE in genes:
        assert len(result[A.ABSENT_GENE]["x"]) == 0
    empty = result["NOT_A_GENE"]
    assert len(empty["x"]) == 0 and empty["x"].dtype == np.float32
    assert empty["original_assignment"].dtype == np.uint32 and empty["code"].dtype == np.int16


@pytest.mark.parametrize("with_map", [True, False])
def test_each_row_group_is_read_at_most_once_per_call(atera_store, monkeypatch, with_map):
    engine = T.AteraTranscriptEngine.open(atera_store)
    if with_map:
        engine.build_code_map()
    spy = _PathSpy(monkeypatch)
    engine.read_genes(["ACTB", "AQP4", "GFAP", "MBP", "SNAP25", "ZIC1"], COLUMNS)
    assert spy.calls
    assert len(spy.calls) == len(set(spy.calls))


def test_code_map_reads_fewer_rows_than_string_statistics(atera_store, monkeypatch):
    # The tiny last file's single row group spans the first gene, the last gene
    # and a control, so its string statistics cover every gene name.
    index = T.AteraPointsIndex.load(atera_store)
    last = index.files[-1]
    assert last.n_row_groups == 1
    gene = "MBP"
    stats_engine = T.AteraTranscriptEngine(index)
    assert 0 in stats_engine._stats_row_groups(last.position, [gene])

    spy = _PathSpy(monkeypatch)
    by_stats = stats_engine.read_genes([gene], COLUMNS)[gene]
    stats_rows = spy.rows
    stats_calls = list(spy.calls)

    map_engine = T.AteraTranscriptEngine(index)
    code_map = map_engine.build_code_map()
    spy.reset()
    by_map = map_engine.read_genes([gene], COLUMNS)[gene]
    assert spy.rows < stats_rows
    assert (last.name, 0) in stats_calls and (last.name, 0) not in spy.calls
    assert code_map.rows_for([GENE_CODES[gene]]) == len(by_map["x"])
    for key in by_map:
        np.testing.assert_array_equal(by_map[key], by_stats[key])


def _rewrite_with_reversed_dictionary(store: Path, file_name: str) -> None:
    path = store / "points" / "transcripts" / "points.parquet" / file_name
    table = pq.read_table(path)
    metadata = pq.read_metadata(path)
    names = table.column("feature_name").to_pylist()
    reversed_dictionary = list(reversed(A.FEATURE_DICTIONARY))
    local = {name: i for i, name in enumerate(reversed_dictionary)}
    feature = pa.DictionaryArray.from_arrays(
        pa.array([local[n] for n in names], type=pa.int16()), pa.array(reversed_dictionary, type=pa.string())
    )
    table = table.set_column(table.schema.get_field_index("feature_name"), "feature_name", feature)
    pq.write_table(table, path, row_group_size=metadata.row_group(0).num_rows)
    assert pq.read_metadata(path).num_row_groups == metadata.num_row_groups


def test_dictionary_mismatch_falls_back_to_a_lut_with_the_same_results(store_copy):
    truth = _truth(store_copy)
    _rewrite_with_reversed_dictionary(store_copy, "part.2.parquet")
    index = T.AteraPointsIndex.load(store_copy)
    assert index is not None and not index.shared_dictionary
    assert index.vocabulary == A.FEATURE_DICTIONARY
    assert index.file_luts[2] is not None
    genes = ["ACTB", "GFAP", "ZIC1", "UnassignedCodeword_0001"]
    for build_map in (False, True):
        engine = T.AteraTranscriptEngine(index)
        if build_map:
            engine.build_code_map()
        result = engine.read_genes(genes, COLUMNS)
        for gene in genes:
            _assert_rows_equal(result[gene], truth[truth["name"] == gene])
    overview = T.AteraTranscriptEngine(index).build_overview()
    expected = truth.groupby("code").size()
    for code, count in expected.items():
        assert overview.counts[code] == count


# ---------------------------------------------------------------------------
# Tile and window reads
# ---------------------------------------------------------------------------


def _inside(frame: pd.DataFrame, bounds) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bounds
    mask = (frame["x"] >= xmin) & (frame["x"] < xmax) & (frame["y"] >= ymin) & (frame["y"] < ymax)
    return frame[mask]


def test_read_tile_equals_brute_force_in_index_order(atera_store):
    engine = T.AteraTranscriptEngine.open(atera_store)
    truth = _truth(atera_store)
    for tile in A.TILE_NAMES:
        got = engine.read_tile(tile, COLUMNS)
        _assert_rows_equal(got, truth[truth["tile"] == tile])
    # The off-grid tile has a negative-x box and null block bounds.
    off = engine.read_tile("-1,1", COLUMNS)
    assert len(off["x"]) and (off["x"] < 0).all()


@pytest.mark.parametrize(
    "bounds",
    [
        (5.0, 2.0, 35.0, 12.0),  # non-square, straddles the 0,0 / 1,0 boundary
        (17.0, 11.0, 24.0, 17.0),  # cell 3, which straddles x = 20 um
        (20.0, 0.0, 30.0, 20.0),  # starts exactly at a tile edge
        (0.0, 0.0, 20.0, 20.0),  # ends exactly at a tile edge (half-open)
        (-5.0, 15.0, 1.0, 35.0),  # the negative-bbox off-grid tile and its neighbours
        (30.0, 30.0, 40.0, 40.0),  # tile 1,1 including its tiny single-row-group file
        (100.0, 100.0, 120.0, 110.0),  # nothing
    ],
)
@pytest.mark.parametrize("with_map", [True, False])
def test_read_window_equals_brute_force(atera_store, bounds, with_map):
    engine = T.AteraTranscriptEngine.open(atera_store)
    if with_map:
        engine.build_code_map()
    truth = _truth(atera_store)
    window = engine.read_window(bounds, columns=COLUMNS)
    expected = _inside(truth, bounds)
    _assert_rows_equal(window.columns, expected)
    assert set(expected["tile"]).issubset(window.tiles)

    genes = ["GFAP", "MBP", "NegControlCodeword_0500"]
    filtered = engine.read_window(bounds, genes=genes, columns=COLUMNS)
    _assert_rows_equal(filtered.columns, expected[expected["name"].isin(genes)])

    whole = engine.read_window(bounds, columns=COLUMNS, clip=False)
    _assert_rows_equal(whole.columns, truth[truth["tile"].isin(whole.tiles)])


def test_window_includes_the_tiny_single_row_group_file(atera_store, monkeypatch):
    engine = T.AteraTranscriptEngine.open(atera_store)
    last = engine.index.files[-1]
    box = last.bbox_um
    spy = _PathSpy(monkeypatch)
    window = engine.read_window((box[0], box[2], box[1] + 0.001, box[3] + 0.001), columns=COLUMNS)
    assert (last.name, 0) in spy.calls
    assert len(window.columns["x"]) >= last.n_rows


# ---------------------------------------------------------------------------
# Nullable ids
# ---------------------------------------------------------------------------


def _mini_store(root: Path, ids: list[int | None]) -> Path:
    store = root / "mini.zarr"
    directory = store / "points" / "transcripts" / "points.parquet"
    directory.mkdir(parents=True)
    n = len(ids)
    table = pa.table(
        {
            "x": pa.array(np.linspace(0.0, 1.0, n, dtype=np.float32)),
            "y": pa.array(np.linspace(0.0, 1.0, n, dtype=np.float32)),
            "feature_name": pa.DictionaryArray.from_arrays(
                pa.array(np.zeros(n, dtype=np.int16)), pa.array(["GENE_A", "NegControlProbe_1"])
            ),
            "is_gene": pa.array(np.ones(n, dtype=bool)),
            "original_assignment": pa.array(ids, type=pa.uint64()),
        }
    )
    pq.write_table(table, directory / "part.0.parquet")
    payload = {
        "format_version": 1,
        "points": {
            "transcripts": {
                "directory": "points/transcripts/points.parquet",
                "fingerprint": {"n_files": 1, "n_rows": n},
                "files": [
                    {"file": "part.0.parquet", "tile": "0,0", "tile_rows": [0, n], "block_bounds_um": None,
                     "n_rows": n, "n_row_groups": 1, "bbox_um": [0.0, 1.0, 0.0, 1.0],
                     "transcript_id_range": [1, n]}
                ],
            }
        },
    }
    (store / T.ATERA_POINTS_INDEX_JSON).write_text(json.dumps(payload), encoding="utf-8")
    return store


def test_null_ids_are_unassigned_and_large_uint64_ids_round_trip(tmp_path):
    big = 2**63 + 5
    store = _mini_store(tmp_path, [None, 7, big, None])
    engine = T.AteraTranscriptEngine.open(store)
    assert engine.index.validated and engine.index.assignment_dtype == np.uint64
    read_paths = [
        engine.read_genes(["GENE_A"])["GENE_A"]["original_assignment"],
        engine.read_tile("0,0")["original_assignment"],
        engine.read_window((0.0, 0.0, 2.0, 2.0)).columns["original_assignment"],
    ]
    engine.build_code_map()
    read_paths.append(engine.read_genes(["GENE_A"])["GENE_A"]["original_assignment"])
    for ids in read_paths:
        assert ids.dtype == np.uint64
        assert ids.tolist() == [0, 7, big, 0]
    overview = engine.build_overview(bin_um=1.0)
    assert overview.assigned_counts.tolist() == [2, 0]


def test_ids_below_2_pow_32_are_uint32(tmp_path):
    store = _mini_store(tmp_path, [None, 2**32 - 1])
    engine = T.AteraTranscriptEngine.open(store)
    ids = engine.read_genes(["GENE_A"])["GENE_A"]["original_assignment"]
    assert ids.dtype == np.uint32 and ids.tolist() == [0, 2**32 - 1]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def test_overview_counts_density_and_code_map_are_exact(atera_store):
    engine = T.AteraTranscriptEngine.open(atera_store)
    truth = _truth(atera_store)
    progress = []
    overview = engine.build_overview(bin_um=5.0, progress=lambda done, total: progress.append((done, total)))
    assert progress[-1] == (len(engine.index.files), len(engine.index.files))
    expected = np.bincount(truth["code"], minlength=len(A.FEATURE_DICTIONARY))
    np.testing.assert_array_equal(overview.counts, expected)
    assigned = np.bincount(truth.loc[truth["original_assignment"] > 0, "code"], minlength=len(A.FEATURE_DICTIONARY))
    np.testing.assert_array_equal(overview.assigned_counts, assigned)
    assert overview.total_rows == len(truth)
    assert overview.control_names() == set(A.CONTROLS)
    assert overview.is_gene_flags_used
    genes = truth[truth["is_gene"]]
    assert int(overview.density_counts.sum()) == len(genes) == int(overview.counts[~overview.is_control].sum())
    assert overview.counts[GENE_CODES[A.ABSENT_GENE]] == 0
    assert A.ABSENT_GENE in overview.gene_names()
    # The raster places each row in its bin, including the negative-x tile.
    x0, y0 = overview.origin_um
    assert x0 <= truth["x"].min() and y0 <= truth["y"].min()
    ix = np.floor((genes["x"].to_numpy(np.float64) - x0) / 5.0).astype(int)
    iy = np.floor((genes["y"].to_numpy(np.float64) - y0) / 5.0).astype(int)
    raster = np.zeros_like(overview.density_counts, dtype=np.int64)
    np.add.at(raster, (iy, ix), 1)
    np.testing.assert_array_equal(overview.density_counts, raster)
    # The pass also yields the exact code-range map.
    assert overview.code_map is not None and engine.code_map is overview.code_map
    for name in ("GFAP", "ZIC1", "Intergenic_Region_1"):
        assert overview.code_map.rows_for([GENE_CODES[name]]) == int((truth["name"] == name).sum())


def test_overview_split_assigned_density(atera_store):
    engine = T.AteraTranscriptEngine.open(atera_store)
    truth = _truth(atera_store)
    overview = engine.build_overview(bin_um=10.0, split_assigned=True)
    assert overview.assigned_density_counts.shape == overview.density_counts.shape
    expected = int(((truth["original_assignment"] > 0) & truth["is_gene"]).sum())
    assert int(overview.assigned_density_counts.sum()) == expected


def test_overview_sample_is_deterministic_per_seed(atera_store):
    engine = T.AteraTranscriptEngine.open(atera_store)
    first = engine.build_overview(sample_size=120, seed=3).sample
    again = engine.build_overview(sample_size=120, seed=3).sample
    other = engine.build_overview(sample_size=120, seed=4).sample
    for key in first:
        np.testing.assert_array_equal(first[key], again[key])
    assert not np.array_equal(first["x"], other["x"])
    assert 0 < len(first["x"]) <= 120
    assert abs(len(first["x"]) - 120 * 0.88) < 40  # ~12% controls are dropped
    controls = {GENE_CODES[name] for name in A.CONTROLS}
    assert not set(first["code"].tolist()) & controls
    assert first["original_assignment"].dtype == np.uint32


def test_overview_honours_cancellation_between_files(atera_store):
    engine = T.AteraTranscriptEngine.open(atera_store)
    done = []
    cancelled = lambda: len(done) >= 2  # noqa: E731
    with pytest.raises(TranscriptBuildCancelled):
        engine.build_overview(cancel=cancelled, progress=lambda d, _t: done.append(d))
    assert len(done) < len(engine.index.files)


def test_reads_honour_a_set_event(atera_store):
    import threading

    engine = T.AteraTranscriptEngine.open(atera_store)
    event = threading.Event()
    event.set()
    with pytest.raises(TranscriptBuildCancelled):
        engine.read_genes(["GFAP", "MBP"], cancel=event)
    with pytest.raises(TranscriptBuildCancelled):
        engine.read_tile("0,0", cancel=event)


def test_sample_global_positions_is_safe_above_1e9():
    total = 1_500_000_000
    positions = T.sample_global_positions(total, 1000, seed=11)
    assert positions.dtype == np.int64 and len(positions) == 1000
    assert np.all(np.diff(positions) > 0)
    assert positions.min() >= 0 and positions.max() < total
    np.testing.assert_array_equal(positions, T.sample_global_positions(total, 1000, seed=11))
    assert not np.array_equal(positions, T.sample_global_positions(total, 1000, seed=12))
    assert len(T.sample_global_positions(5, 10, seed=1)) == 5
    assert len(T.sample_global_positions(0, 10, seed=1)) == 0


# ---------------------------------------------------------------------------
# Overview cache
# ---------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_overview_cache_round_trip_and_source_tree_untouched(store_copy, tmp_path):
    cache = tmp_path / "cache"
    before = _tree_digest(store_copy)
    engine = T.AteraTranscriptEngine.open(store_copy)
    overview = engine.build_overview(bin_um=5.0, sample_size=50, seed=2, split_assigned=True)
    path = TC.save_atera_overview(engine.index, overview, cache_root=cache)
    assert path is not None and path.is_dir() and not str(path).startswith(str(store_copy))
    restored = TC.load_atera_overview(
        T.AteraPointsIndex.load(store_copy), bin_um=5.0, sample_size=50, sample_seed=2,
        split_assigned=True, cache_root=cache,
    )
    assert restored is not None
    for name in ("counts", "assigned_counts", "is_control", "density_counts", "assigned_density_counts"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(overview, name))
    assert restored.vocabulary == overview.vocabulary and restored.origin_um == overview.origin_um
    for key in overview.sample:
        np.testing.assert_array_equal(restored.sample[key], overview.sample[key])
    for name in ("code_ptr", "row_group", "row_start", "row_end"):
        np.testing.assert_array_equal(getattr(restored.code_map, name), getattr(overview.code_map, name))
    # Other options are other entries.
    assert TC.load_atera_overview(engine.index, bin_um=10.0, cache_root=cache) is None
    # The code-range map sidecar is written alongside.
    code_map = TC.load_atera_code_map(engine.index, cache_root=cache)
    assert code_map is not None and code_map.n_runs == overview.code_map.n_runs
    assert _tree_digest(store_copy) == before


def test_overview_cache_is_invalidated_by_index_or_mtime_but_not_ctime(store_copy, tmp_path):
    cache = tmp_path / "cache"
    engine = T.AteraTranscriptEngine.open(store_copy)
    overview = engine.build_overview(bin_um=5.0)
    assert TC.save_atera_overview(engine.index, overview, cache_root=cache) is not None

    part = engine.index.files[4].path
    # ctime only: a hardlink (MerXen's churn) changes ctime, not size or mtime.
    stat = os.stat(part)
    link = tmp_path / "hardlink.parquet"
    os.link(part, link)
    link.unlink()
    assert os.stat(part).st_mtime_ns == stat.st_mtime_ns
    assert all(set(item) == {"file", "size", "mtime_ns"} for item in T.file_stat_identity(engine.index))
    assert TC.load_atera_overview(T.AteraPointsIndex.load(store_copy), bin_um=5.0, cache_root=cache) is not None

    # mtime change.
    os.utime(part, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert TC.load_atera_overview(T.AteraPointsIndex.load(store_copy), bin_um=5.0, cache_root=cache) is None
    os.utime(part, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert TC.load_atera_overview(T.AteraPointsIndex.load(store_copy), bin_um=5.0, cache_root=cache) is not None

    # Index JSON content change (still valid).
    index_path = store_copy / T.ATERA_POINTS_INDEX_JSON
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    raw["points"]["transcripts"]["files"][1]["tile_rows"] = [0, 1]
    index_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    changed = T.AteraPointsIndex.load(store_copy)
    assert changed is not None
    assert TC.load_atera_overview(changed, bin_um=5.0, cache_root=cache) is None
    assert TC.load_atera_code_map(changed, cache_root=cache) is None


def test_overview_cache_coexists_with_v1_entries(store_copy, tmp_path):
    from napari_compare_xenium_merscope.utils import GenePointStore, GeneVisual, MarkerReferenceResolution

    cache = tmp_path / "cache"
    request = TC.TranscriptCacheRequest.create(
        dataset_path=store_copy, points_key="transcripts", x_col="x", y_col="y",
        gene_col="feature_name", cache_root=cache,
    )
    assert request is not None
    store = GenePointStore(
        group_symbols=["disc"],
        group_coords=[np.zeros((1, 2), dtype=np.float32)],
        group_colors=[np.ones((1, 4), dtype=np.float32)],
        gene_offsets={"GFAP": (0, 0, 1, 1)},
        gene_counts={"GFAP": 1},
        genes=["GFAP"],
        control_genes=set(),
        total_points=1,
        gene_visuals={"GFAP": GeneVisual(rgba=(1.0, 0.0, 0.0, 1.0), symbol="disc")},
    )
    v1_path = TC.save_transcript_cache(request, store, MarkerReferenceResolution(reference=None))
    assert v1_path is not None
    manifest = json.loads((v1_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == TC.TRANSCRIPT_POINTS_KIND

    engine = T.AteraTranscriptEngine.open(store_copy)
    overview_path = TC.save_atera_overview(engine.index, engine.build_overview(), cache_root=cache)
    assert overview_path is not None
    # Saving either never prunes the other.
    assert v1_path.is_dir()
    assert TC.save_transcript_cache(request, store, MarkerReferenceResolution(reference=None)) is not None
    assert overview_path.is_dir()
    assert TC.load_transcript_cache(request, None) is not None

    # A v1 reader ignores an overview manifest even at a v1 address.
    forged = json.loads((v1_path / "manifest.json").read_text(encoding="utf-8"))
    forged["kind"] = TC.ATERA_OVERVIEW_CACHE_KIND
    (v1_path / "manifest.json").write_text(json.dumps(forged), encoding="utf-8")
    assert TC.load_transcript_cache(request, None) is None
    # A pre-kind v1 manifest (no field) still loads.
    del forged["kind"]
    (v1_path / "manifest.json").write_text(json.dumps(forged), encoding="utf-8")
    assert TC.load_transcript_cache(request, None) is not None


def test_overview_cache_is_disabled_inside_the_dataset(store_copy):
    engine = T.AteraTranscriptEngine.open(store_copy)
    overview = engine.build_overview(bin_um=10.0)
    assert TC.save_atera_overview(engine.index, overview, cache_root=store_copy / "cache") is None
    assert not (store_copy / "cache").exists()


def test_code_map_sidecar_round_trip_and_rejects_a_foreign_map(store_copy, tmp_path):
    cache = tmp_path / "cache"
    engine = T.AteraTranscriptEngine.open(store_copy)
    code_map = engine.build_code_map()
    assert TC.save_atera_code_map(engine.index, code_map, cache_root=cache) is not None
    fresh = T.AteraTranscriptEngine.open(store_copy)
    restored = TC.load_atera_code_map(fresh.index, cache_root=cache)
    assert fresh.attach_code_map(restored) and fresh.code_map is restored
    truth = _truth(store_copy)
    _assert_rows_equal(fresh.read_genes(["SNAP25"], COLUMNS)["SNAP25"], truth[truth["name"] == "SNAP25"])
    foreign = T.CodeRangeMap(
        code_ptr=np.zeros(3, dtype=np.int64), row_group=np.zeros(0, np.int32),
        row_start=np.zeros(0, np.int64), row_end=np.zeros(0, np.int64),
    )
    assert not fresh.attach_code_map(foreign)
    assert fresh.code_map is restored


# ---------------------------------------------------------------------------
# Hygiene and real data
# ---------------------------------------------------------------------------


def test_engine_module_imports_no_qt():
    code = (
        "import sys; import napari_compare_xenium_merscope.atera_transcripts as m; "
        "bad = [n for n in sys.modules if n.split('.')[0] in ('qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'napari', 'vispy')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.environ.get("NAPARI_COMPARE_ATERA_STORE"), reason="NAPARI_COMPARE_ATERA_STORE not set")
def test_real_atera_store_index():
    index = T.AteraPointsIndex.load(os.environ["NAPARI_COMPARE_ATERA_STORE"])
    assert index is not None and index.validated
    assert len(index.files) == 409
    assert index.n_rows == 1_409_601_228
    assert index.shared_dictionary and len(index.vocabulary) == 24_677
    assert index.assignment_dtype == np.uint32
