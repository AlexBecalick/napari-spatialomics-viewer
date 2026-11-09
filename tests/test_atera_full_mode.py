"""ATERA whole-slide ("full") transcript load: hash sample, budgets, cache, point store.

Runs on the synthetic store from ``atera_stores``; nothing reads /Users/... data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import atera_stores as A
import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope import atera_transcripts as T
from napari_compare_xenium_merscope import transcript_cache as TC
from napari_compare_xenium_merscope import utils as U


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return A.build_atera_latest_store(tmp_path_factory.mktemp("atera_full"))


@pytest.fixture
def store_copy(atera_store, tmp_path) -> Path:
    dst = tmp_path / "copy" / atera_store.name
    shutil.copytree(atera_store, dst)
    return dst


def _engine(store: Path) -> T.AteraTranscriptEngine:
    engine = T.AteraTranscriptEngine.open(store)
    assert engine is not None and engine.index.validated
    return engine


def _rows(sample) -> set[tuple]:
    return set(
        zip(
            sample.x.tolist(), sample.y.tolist(), sample.code.astype(int).tolist(),
            sample.cell_id.astype(int).tolist(), sample.key.astype(int).tolist(),
        )
    )


def test_sample_keys_are_deterministic_and_spread():
    a = T.row_sample_keys(3, 100, 5000)
    b = T.row_sample_keys(3, 100, 5000)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint32
    # A different file gives different keys; the fraction below half is about half.
    assert not np.array_equal(a, T.row_sample_keys(4, 100, 5000))
    assert 0.45 < float(np.mean(a < np.uint32(1 << 31))) < 0.55
    # Keys of a block equal the keys of its pieces.
    np.testing.assert_array_equal(a[:10], T.row_sample_keys(3, 100, 10))
    np.testing.assert_array_equal(a[10:], T.row_sample_keys(3, 110, 4990))


def test_sample_threshold_bounds():
    assert T.sample_threshold(0, 100) == 0
    assert T.sample_threshold(100, 100) == T.SAMPLE_KEY_SPACE
    assert T.sample_threshold(10**12, 100) == T.SAMPLE_KEY_SPACE
    assert T.sample_threshold(50, 100) == T.SAMPLE_KEY_SPACE // 2
    assert T.sample_threshold(1, 1_409_601_228) > 0


def test_budget_at_or_above_total_keeps_every_row(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    sample = engine.sample_rows(T.sample_threshold(total, total))
    assert sample.complete
    assert sample.n == total
    full = engine.read_window((-1e9, -1e9, 1e9, 1e9), columns=("x", "y", "feature_name"))
    assert sorted(zip(full.columns["x"].tolist(), full.columns["y"].tolist())) == sorted(
        zip(sample.x.tolist(), sample.y.tolist())
    )


def test_samples_nest_and_subset_matches_a_fresh_read(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    small_t = T.sample_threshold(total // 5, total)
    large_t = T.sample_threshold(total // 2, total)
    small = engine.sample_rows(small_t)
    large = engine.sample_rows(large_t)
    assert 0 < small.n < large.n < total
    assert _rows(small) <= _rows(large)
    assert _rows(large.subset(small_t)) == _rows(small)
    with pytest.raises(ValueError):
        small.subset(large_t)
    # Deterministic: a second read is identical, in the same order.
    again = engine.sample_rows(small_t)
    np.testing.assert_array_equal(again.key, small.key)
    np.testing.assert_array_equal(again.x, small.x)


def test_sample_size_tracks_the_budget(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    for budget in (total // 4, total // 2):
        n = engine.sample_rows(T.sample_threshold(budget, total)).n
        assert abs(n - budget) < 5 * np.sqrt(budget) + 5


def test_overview_pass_keeps_the_same_hash_sample(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 3, total)
    overview = engine.build_overview(5.0, keep_threshold=threshold)
    direct = engine.sample_rows(threshold)
    assert overview.hash_sample is not None
    for name in ("x", "y", "code", "cell_id", "key"):
        np.testing.assert_array_equal(getattr(overview.hash_sample, name), getattr(direct, name))
    # Unassigned rows are 0, assigned ids survive as integers.
    assert overview.hash_sample.cell_id.dtype in (np.uint32, np.uint64)


def test_auto_budget_and_memory_guard_arithmetic():
    gib = 1024**3
    assert T.auto_point_budget(10**10, bytes_per_point=200, ram_bytes=64 * gib) == int(0.5 * 64 * gib // 200)
    assert T.auto_point_budget(1000, bytes_per_point=200, ram_bytes=64 * gib) == 1000
    assert T.auto_point_budget(1000, bytes_per_point=200, ram_bytes=None) == 1000
    budget, clamped = T.clamp_point_budget(10**10, bytes_per_point=200, ram_bytes=64 * gib)
    assert clamped and budget == int(0.85 * 64 * gib // 200)
    assert T.clamp_point_budget(1000, bytes_per_point=200, ram_bytes=64 * gib) == (1000, False)
    assert T.physical_memory_bytes() is None or T.physical_memory_bytes() > 0


def test_window_rows_exclude_exactly_the_sampled_rows(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 3, total)
    sample = engine.sample_rows(threshold)
    bounds = (-1e9, -1e9, 1e9, 1e9)
    rest = engine.read_window_rows(bounds, exclude_below=threshold)
    everything = engine.read_window_rows(bounds)
    assert len(rest["x"]) + sample.n == len(everything["x"]) == total
    assert bool(np.all(rest["key"] >= np.uint32(threshold)))
    # Restricted to some codes and a window.
    codes = sorted(set(sample.code.astype(int).tolist()))[:3]
    x0, x1 = float(np.quantile(everything["x"], 0.2)), float(np.quantile(everything["x"], 0.8))
    y0, y1 = float(np.quantile(everything["y"], 0.2)), float(np.quantile(everything["y"], 0.8))
    win = engine.read_window_rows((x0, y0, x1, y1), codes, exclude_below=threshold)
    inside = (sample.x >= x0) & (sample.x < x1) & (sample.y >= y0) & (sample.y < y1) & np.isin(sample.code, codes)
    allin = engine.read_window_rows((x0, y0, x1, y1), codes)
    assert len(win["x"]) + int(inside.sum()) == len(allin["x"])
    assert set(win["code"].astype(int).tolist()) <= set(codes)


def test_sample_cache_reuses_a_larger_sample_and_never_shrinks(store_copy, tmp_path):
    cache = tmp_path / "cache"
    engine = _engine(store_copy)
    index = engine.index
    total = index.n_rows
    small_t = T.sample_threshold(total // 4, total)
    large_t = T.sample_threshold(total // 2, total)
    large = engine.sample_rows(large_t)
    assert TC.load_atera_sample(index, small_t, cache_root=cache) is None
    assert TC.save_atera_sample(index, large, cache_root=cache) is not None
    assert TC.cached_atera_sample_threshold(index, cache_root=cache) == large_t
    got = TC.load_atera_sample(index, small_t, cache_root=cache)
    assert got is not None and got.threshold == small_t
    assert _rows(got) == _rows(engine.sample_rows(small_t))
    same = TC.load_atera_sample(index, large_t, cache_root=cache)
    assert same is not None and _rows(same) == _rows(large)
    # A smaller save never replaces the larger entry; a larger request misses.
    assert TC.save_atera_sample(index, engine.sample_rows(small_t), cache_root=cache) is None
    assert TC.cached_atera_sample_threshold(index, cache_root=cache) == large_t
    assert TC.load_atera_sample(index, T.SAMPLE_KEY_SPACE, cache_root=cache) is None
    # The row cap keeps huge samples out of the cache.
    assert TC.save_atera_sample(index, engine.sample_rows(T.SAMPLE_KEY_SPACE), cache_root=cache, max_rows=10) is None


def test_sample_cache_does_not_disturb_the_overview_or_v1_entries(store_copy, tmp_path):
    cache = tmp_path / "cache"
    engine = _engine(store_copy)
    overview = engine.build_overview(5.0)
    assert TC.save_atera_overview(engine.index, overview, cache_root=cache) is not None
    total = engine.index.n_rows
    assert TC.save_atera_sample(engine.index, engine.sample_rows(T.sample_threshold(total // 2, total)), cache_root=cache)
    assert TC.load_atera_overview(engine.index, bin_um=5.0, cache_root=cache) is not None
    kinds = sorted(p.name for p in cache.iterdir())
    assert kinds == ["atera-celltiles-v1", "atera-codemap-v1", "atera-overview-v1", "atera-sample-v1"]


def _reference_store_order(genes, codes, background, visuals):
    """The pre-refactor ordering: argsort of group * 2N + code * 2 + background."""
    symbol_to_idx = {sym: i for i, sym in enumerate(U.GENE_MARKER_SYMBOLS)}
    sidx_by_code = np.array([symbol_to_idx[visuals[g].symbol] for g in genes], dtype=np.int64)
    per_point = sidx_by_code[codes]
    used = np.unique(per_point)
    to_group = np.full(len(U.GENE_MARKER_SYMBOLS), -1, dtype=np.int64)
    to_group[used] = np.arange(len(used))
    group = to_group[per_point]
    bucket = group.astype(np.uint64) * np.uint64(2 * len(genes)) + codes.astype(np.uint64) * np.uint64(2) + background.astype(np.uint64)
    return np.argsort(bucket, kind="stable"), [U.GENE_MARKER_SYMBOLS[int(s)] for s in used]


@pytest.mark.parametrize("n_genes", [7, 40_000])
def test_assembled_store_matches_the_original_ordering(n_genes):
    rng = np.random.default_rng(5)
    genes = [f"G{i:05d}" for i in range(n_genes)]
    n = 20_000
    codes = rng.integers(0, min(n_genes, 300), size=n).astype(np.uint32)
    background = rng.random(n) < 0.3
    x = rng.random(n).astype(np.float32)
    y = rng.random(n).astype(np.float32)
    visuals = U.assign_gene_visuals(genes)
    store = U.assemble_gene_point_store(genes, x, y, codes, background, gene_visuals=visuals)
    order, symbols = _reference_store_order(genes, codes, background, visuals)
    assert store.group_symbols == symbols
    coords = np.concatenate(store.group_coords)
    np.testing.assert_array_equal(coords[:, 0], y[order])
    np.testing.assert_array_equal(coords[:, 1], x[order])
    for gene, (gi, fg0, fg1, bg1) in store.gene_offsets.items():
        code = genes.index(gene)
        assert store.gene_counts[gene] == int(np.sum(codes == code))
        assert fg1 - fg0 == int(np.sum((codes == code) & ~background))
        assert bg1 - fg0 == store.gene_counts[gene]


def test_gene_point_store_from_rows_lists_every_observed_feature(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 2, total)
    overview = engine.build_overview(5.0, keep_threshold=threshold)
    sample = overview.hash_sample
    store = T.gene_point_store_from_rows(
        overview, sample.x, sample.y, sample.code, sample.cell_id, sampled=not sample.complete
    )
    genes, _lut, counts, controls = T.overview_panel(overview)
    assert store.genes == sorted(genes) == genes
    assert store.source_gene_counts == counts
    assert store.control_genes == controls
    assert store.total_points == sample.n
    assert store.source_total_points == total
    assert store.sampled
    names = np.asarray(overview.vocabulary, dtype=object)[sample.code.astype(int)]
    frame = pd.DataFrame({"gene": names, "bg": sample.cell_id == 0})
    for gene, (gi, fg0, fg1, bg1) in store.gene_offsets.items():
        rows = frame[frame.gene == gene]
        assert bg1 - fg0 == len(rows)
        assert fg1 - fg0 == int((~rows.bg).sum())


def test_window_rows_for_many_genes_filter_by_a_code_table(atera_store, monkeypatch):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 3, total)
    codes = list(range(0, len(engine.index.vocabulary), 2))
    planned = engine.read_window_rows((-1e9, -1e9, 1e9, 1e9), codes, exclude_below=threshold)
    monkeypatch.setattr(T, "WINDOW_PLAN_MAX_CODES", 1)
    tabled = engine.read_window_rows((-1e9, -1e9, 1e9, 1e9), codes, exclude_below=threshold)
    key = lambda rows: sorted(zip(rows["x"].tolist(), rows["y"].tolist(), rows["code"].astype(int).tolist()))  # noqa: E731
    assert key(planned) == key(tabled)
    assert set(tabled["code"].astype(int).tolist()) <= set(codes)


# ---------------------------------------------------------------------------
# All-genes display cap (M10a): a nested sub-sample leads every gene run
# ---------------------------------------------------------------------------


def test_partition_by_key_leads_with_the_nested_sub_sample(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    sample = engine.sample_rows(T.sample_threshold(total // 2, total))
    before = {name: getattr(sample, name).copy() for name in ("x", "y", "code", "cell_id", "key")}
    cap_threshold = T.sample_threshold(total // 6, total)
    member = before["key"] < np.uint32(cap_threshold)
    lead = sample.partition_by_key(cap_threshold)
    assert 0 < lead < sample.n and lead == int(member.sum())
    # Members lead, non-members follow, each side in its original order.
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(sample, name)[:lead], values[member])
        np.testing.assert_array_equal(getattr(sample, name)[lead:], values[~member])
    # The leading rows are exactly a fresh read at the smaller threshold.
    assert _rows(sample.subset(cap_threshold)) == _rows(engine.sample_rows(cap_threshold))
    # A cap at or above the sample is a no-op.
    keys = sample.key.copy()
    assert sample.partition_by_key(sample.threshold) == sample.n
    assert sample.partition_by_key(T.SAMPLE_KEY_SPACE) == sample.n
    np.testing.assert_array_equal(sample.key, keys)


class _FancyIndexOOM(np.ndarray):
    """A column whose full-length fancy indexing (a permutation) runs out of memory."""

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) and idx.dtype.kind == "i" and len(idx) == len(self):
            raise MemoryError("simulated")
        return np.asarray(self)[idx]


def _oom_sample(n=64):
    rng = np.random.default_rng(3)
    x = np.arange(n, dtype=np.float32)
    return T.SampledTranscripts(
        x=x, y=x.copy(), code=np.arange(n, dtype=np.int32), cell_id=np.arange(n, dtype=np.uint32),
        key=rng.integers(0, 1000, n).astype(np.uint32), threshold=1000, total_rows=n,
    )


@pytest.mark.parametrize("failing", ["y", "code", "cell_id", "key"])
def test_partition_by_key_out_of_memory_part_way_keeps_rows_aligned(failing):
    sample = _oom_sample()
    before = {name: np.asarray(getattr(sample, name)).copy() for name in ("x", "y", "code", "cell_id", "key")}
    setattr(sample, failing, getattr(sample, failing).view(_FancyIndexOOM))
    with pytest.raises(MemoryError):
        sample.partition_by_key(500)
    # Rolled back: every column is back in its original order.
    assert sample.aligned
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(getattr(sample, name)), values)
    # The OOM retry path filters it by key: x, y, code and cell id still match per row.
    sub = sample.subset(800)
    np.testing.assert_array_equal(sub.x, np.asarray(sub.y))
    np.testing.assert_array_equal(sub.x.astype(np.int64), np.asarray(sub.code).astype(np.int64))
    np.testing.assert_array_equal(sub.x.astype(np.int64), np.asarray(sub.cell_id).astype(np.int64))
    np.testing.assert_array_equal(np.asarray(sub.key), before["key"][before["key"] < 800])


def test_partition_by_key_marks_the_sample_unaligned_when_the_rollback_also_fails(monkeypatch):
    sample = _oom_sample()
    sample.code = sample.code.view(_FancyIndexOOM)
    real_empty_like = np.empty_like

    def no_memory(*args, **kwargs):
        raise MemoryError("simulated rollback")

    monkeypatch.setattr(T.np, "empty_like", no_memory)
    with pytest.raises(MemoryError):
        sample.partition_by_key(500)
    monkeypatch.setattr(T.np, "empty_like", real_empty_like)
    assert not sample.aligned
    with pytest.raises(ValueError):
        sample.subset(800)


def test_store_records_the_capped_overview_prefix_of_every_run(atera_store):
    engine = _engine(atera_store)
    total = engine.index.n_rows
    threshold = T.sample_threshold(total // 2, total)
    overview = engine.build_overview(5.0, keep_threshold=threshold)
    sample = overview.hash_sample
    cap_threshold = T.sample_threshold(total // 6, total)
    lead = sample.partition_by_key(cap_threshold)
    store = T.gene_point_store_from_rows(
        overview, sample.x, sample.y, sample.code, sample.cell_id, sampled=True, overview_prefix=lead
    )
    assert store.overview_offsets is not None
    # Map each store point back to its sample key through its coordinates.
    key_of = {(float(y), float(x)): int(k) for x, y, k in zip(sample.x, sample.y, sample.key)}
    drawn = 0
    for gene, (gi, fg0, fg1, bg1) in store.gene_offsets.items():
        fg_stop, bg_stop = store.overview_offsets[gene]
        assert fg0 <= fg_stop <= fg1 <= bg_stop <= bg1
        coords = store.group_coords[gi]
        keys = np.array([key_of[(float(y), float(x))] for y, x in coords[fg0:bg1]], dtype=np.int64)
        in_cap = keys < cap_threshold
        prefix = np.zeros(bg1 - fg0, dtype=bool)
        prefix[: fg_stop - fg0] = True
        prefix[fg1 - fg0 : bg_stop - fg0] = True
        np.testing.assert_array_equal(prefix, in_cap)
        drawn += int(prefix.sum())
    assert drawn == lead
    # Without a prefix (Xenium/MERSCOPE, or an uncapped sample) nothing is recorded.
    plain = T.gene_point_store_from_rows(
        overview, sample.x, sample.y, sample.code, sample.cell_id, sampled=True
    )
    assert plain.overview_offsets is None
    assert U.assemble_gene_point_store(
        ["A", "B"], np.zeros(3, np.float32), np.zeros(3, np.float32),
        np.array([0, 1, 1], np.uint16), np.array([False, True, False]), overview_prefix=3,
    ).overview_offsets is None
