"""The vectorised eager build reproduces the frozen per-row build bit for bit.

``tests/_frozen_gene_point_groups.py`` is the build as it was before the
vectorisation; every store field, dtype, dict order and cell-index slice must
match it across gene and assignment dtypes, NA ids, 19-digit and uint64 ids,
render sampling and multi-partition dask frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frozen_gene_point_groups import frozen_build_gene_point_groups  # noqa: E402

from napari_compare_xenium_merscope import utils as U  # noqa: E402
from napari_compare_xenium_merscope.atera_transcripts import sample_global_positions  # noqa: E402

N_ROWS = 600
GENES = ["Gfap", "Aqp4", "Blank-0003", "NegControlProbe_1", "Snap25", "Mbp", "Olig2"]


def _assert_arrays_identical(a, b, label):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.dtype == b.dtype, f"{label}: dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    if a.dtype.kind == "f":
        assert a.tobytes() == b.tobytes(), f"{label}: values differ"
    else:
        assert np.array_equal(a, b), f"{label}: values differ"


def _assert_dicts_identical(a, b, label):
    if a is None or b is None:
        assert a is b, label
        return
    assert list(a.items()) == list(b.items()), f"{label}: dicts differ (content or order)"


def assert_stores_identical(new, old):
    assert new.group_symbols == old.group_symbols
    assert len(new.group_coords) == len(old.group_coords)
    for i, (a, b) in enumerate(zip(new.group_coords, old.group_coords)):
        _assert_arrays_identical(a, b, f"group_coords[{i}]")
    for i, (a, b) in enumerate(zip(new.group_colors, old.group_colors)):
        _assert_arrays_identical(a, b, f"group_colors[{i}]")
    _assert_dicts_identical(new.gene_offsets, old.gene_offsets, "gene_offsets")
    _assert_dicts_identical(new.gene_counts, old.gene_counts, "gene_counts")
    assert new.genes == old.genes
    assert new.control_genes == old.control_genes
    assert new.total_points == old.total_points
    assert new.sampled == old.sampled
    _assert_dicts_identical(new.source_gene_counts, old.source_gene_counts, "source_gene_counts")
    assert new.source_total_points == old.source_total_points
    _assert_dicts_identical(new.source_status_counts, old.source_status_counts, "source_status_counts")
    assert (new.gene_visuals is None) == (old.gene_visuals is None)
    if new.gene_visuals is not None:
        assert list(new.gene_visuals.items()) == list(old.gene_visuals.items())
    a, b = new.cell_transcript_index, old.cell_transcript_index
    assert (a is None) == (b is None)
    if a is not None:
        _assert_arrays_identical(a.coords_yx, b.coords_yx, "cell coords")
        _assert_arrays_identical(a.gene_codes, b.gene_codes, "cell gene codes")
        assert a.gene_names == b.gene_names
        _assert_dicts_identical(a.slices, b.slices, "cell slices")
        for key in list(b.slices)[:50]:
            ca, ga = a.transcripts_for(key)
            cb, gb = b.transcripts_for(key)
            _assert_arrays_identical(ca, cb, f"transcripts_for({key}) coords")
            _assert_arrays_identical(ga, gb, f"transcripts_for({key}) genes")


def _base_frame(seed=0, n=N_ROWS):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 500, n).astype(np.float64)
    y = rng.uniform(0, 500, n).astype(np.float32)
    x[rng.choice(n, 7, replace=False)] = np.nan
    y[rng.choice(n, 5, replace=False)] = np.inf
    genes = np.asarray(GENES, dtype=object)[rng.integers(0, len(GENES), n)]
    cells = rng.integers(0, 40, n)
    return rng, x, y, genes, cells


def _gene_column(kind, genes, rng):
    if kind == "object":
        out = genes.copy()
        out[rng.choice(len(out), 6, replace=False)] = None
        out[rng.choice(len(out), 3, replace=False)] = np.nan
        return pd.Series(out, dtype=object)
    if kind == "string":
        return pd.Series(genes, dtype="string")
    if kind == "string_pyarrow":
        out = genes.copy()
        out[:4] = None
        return pd.Series(out, dtype="string[pyarrow]")
    if kind == "arrow_dictionary":
        import pyarrow as pa

        array = pa.array(list(genes)).dictionary_encode()
        return pd.Series(pd.arrays.ArrowExtensionArray(array))
    if kind == "categorical":
        cat = pd.Categorical(genes, categories=sorted(set(GENES)) + ["Unused"])
        series = pd.Series(cat)
        series[rng.choice(len(series), 4, replace=False)] = None
        return series
    if kind == "categorical_int":
        codes = pd.Series(pd.factorize(genes)[0]).astype(np.int64)
        return pd.Series(pd.Categorical(codes, categories=list(range(len(GENES)))))
    if kind == "int":
        return pd.Series(pd.factorize(genes)[0].astype(np.int32))
    if kind == "float":
        values = pd.factorize(genes)[0].astype(np.float64)
        values[:3] = -0.0
        values[3:5] = np.nan
        return pd.Series(values)
    if kind == "mixed_object":  # 1 and "1" stringify alike: per-row fallback
        values = pd.factorize(genes)[0].astype(object)
        values[::5] = [str(v) for v in values[::5]]
        return pd.Series(values, dtype=object)
    raise AssertionError(kind)


def _assignment_column(kind, cells, rng):
    n = len(cells)
    if kind == "uint32":
        return pd.Series(cells.astype(np.uint32))
    if kind == "int64_neg":
        return pd.Series(np.where(cells == 1, -1, cells).astype(np.int64))
    if kind == "float":
        values = cells.astype(np.float64)
        values[rng.choice(n, 9, replace=False)] = np.nan
        values[:2] = 2.0**60  # beyond float's exact integer range -> no key
        values[2:4] = np.inf
        return pd.Series(values)
    if kind == "string":
        values = np.asarray([f"cell_{c}" if c % 7 else ("" if c % 2 else "0") for c in cells], dtype=object)
        values[rng.choice(n, 5, replace=False)] = None
        values[5:8] = ["nan", "None", "-1"]
        values[8:12] = [" 12", "12", "12.0", "1.2e1"]  # one cell key, four spellings
        return pd.Series(values, dtype=object)
    if kind == "arrow_int64":
        return pd.Series(pd.array(cells.astype(np.int64), dtype="int64[pyarrow]"))
    if kind == "string_dtype":
        return pd.Series([str(c) if c else "<NA>" for c in cells], dtype="string")
    if kind == "digits19":
        base = 1_234_567_890_123_456_789
        return pd.Series(np.asarray([str(base + c) if c else "0" for c in cells], dtype=object))
    if kind == "uint64_max":
        values = (np.uint64(2**64 - 1) - cells.astype(np.uint64)).astype(np.uint64)
        values[cells == 0] = 0
        return pd.Series(values, dtype=np.uint64)
    if kind == "UInt64_na":
        values = pd.array(
            [None if c in (0, 3) else (2**63 + 11 + int(c)) for c in cells], dtype="UInt64"
        )
        return pd.Series(values)
    if kind == "Int64_na":
        return pd.Series(pd.array([None if c == 0 else int(c) for c in cells], dtype="Int64"))
    if kind == "categorical":
        return pd.Series(pd.Categorical([f"c{c}" if c else "0" for c in cells]))
    if kind == "mixed_object":
        values = cells.astype(object)
        values[::3] = [str(v) for v in values[::3]]
        values[1] = True
        values[4] = 7.0
        return pd.Series(values, dtype=object)
    raise AssertionError(kind)


def _frame(gene_kind="object", assignment_kind="uint32", *, status=False, is_gene=False, background=False, seed=0):
    rng, x, y, genes, cells = _base_frame(seed)
    df = pd.DataFrame({"x": x, "y": y})
    df["gene"] = _gene_column(gene_kind, genes, rng)
    if assignment_kind is not None:
        df["assignment"] = _assignment_column(assignment_kind, cells, rng)
    if background:
        df["background"] = rng.random(len(df)) < 0.3
    if status:
        states = np.asarray(["assigned", "unassigned", "excluded", None], dtype=object)
        df["status"] = pd.Series(states[rng.integers(0, 4, len(df))], dtype=object)
    if is_gene:
        flags = ~pd.Series(genes).str.startswith(("Blank", "NegControl")).to_numpy()
        flags[0] = not flags[0]
        df["is_gene"] = pd.Series(flags).astype("boolean")
        df.loc[3, "is_gene"] = pd.NA
    return df


def _both(points, **kwargs):
    new = U.build_gene_point_groups(points, **kwargs)
    old = frozen_build_gene_point_groups(points, **kwargs)
    assert_stores_identical(new, old)
    return new


GENE_KINDS = ["object", "string", "string_pyarrow", "arrow_dictionary", "categorical", "categorical_int", "int", "float", "mixed_object"]
ASSIGNMENT_KINDS = [
    "uint32", "int64_neg", "float", "string", "string_dtype", "arrow_int64", "digits19",
    "uint64_max", "UInt64_na", "Int64_na", "categorical", "mixed_object",
]


@pytest.mark.parametrize("gene_kind", GENE_KINDS)
@pytest.mark.parametrize("max_points", [None, 97])
def test_gene_dtypes_match_frozen_build(gene_kind, max_points):
    df = _frame(gene_kind, "uint32")
    _both(
        df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment",
        max_points=max_points, random_state=3, build_cell_index=True,
    )


@pytest.mark.parametrize("assignment_kind", ASSIGNMENT_KINDS)
@pytest.mark.parametrize("max_points", [None, 150])
def test_assignment_dtypes_match_frozen_build(assignment_kind, max_points):
    df = _frame("categorical", assignment_kind)
    store = _both(
        df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment",
        max_points=max_points, random_state=11, build_cell_index=True,
    )
    assert store.cell_transcript_index is not None


@pytest.mark.parametrize("npartitions", [1, 3, 7])
@pytest.mark.parametrize("read_ahead", [0, 1, 3])
@pytest.mark.parametrize("max_points", [None, 1, 233, N_ROWS * 4])
def test_dask_partitions_and_sampling_match_frozen_build(monkeypatch, npartitions, read_ahead, max_points):
    monkeypatch.setattr(U, "_POINT_PARTITION_READ_AHEAD", read_ahead)
    frames = [_frame("object", "string", status=True, seed=s) for s in range(3)]
    df = pd.concat(frames, ignore_index=True)
    ddf = dd.from_pandas(df, npartitions=npartitions)
    _both(
        ddf, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment",
        status_col="status", max_points=max_points, random_state=5, build_cell_index=True,
    )


@pytest.mark.parametrize("gene_kind", ["object", "categorical"])
def test_background_status_and_is_gene_columns_match_frozen_build(gene_kind):
    df = _frame(gene_kind, "digits19", status=True, is_gene=True, background=True)
    _both(
        df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment",
        background_col="background", status_col="status", is_gene_col="is_gene",
        max_points=211, random_state=8, build_cell_index=True,
    )


def test_no_assignment_and_empty_partitions_match_frozen_build():
    df = _frame("string", None)
    df.loc[: N_ROWS // 2, "x"] = np.nan  # the first partitions are empty
    ddf = dd.from_pandas(df, npartitions=6)
    _both(ddf, x_col="x", y_col="y", gene_col="gene", max_points=50, random_state=1)
    _both(ddf, x_col="x", y_col="y", gene_col="gene")


def test_categorical_status_compares_codes():
    df = _frame("object", "uint32", status=True)
    df["status"] = pd.Categorical(df["status"])
    _both(df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment", status_col="status")


def test_nineteen_digit_and_uint64_max_ids_survive_into_the_cell_index():
    """Exactness scope: CellTranscriptIndex slices and transcripts_for keep every digit."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [5.0, 6.0, 7.0, 8.0],
        "gene": ["A", "B", "A", "B"],
        "assignment": np.asarray([2**64 - 1, 2**64 - 2, 2**64 - 1, 0], dtype=np.uint64),
    })
    store = _both(df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment", build_cell_index=True)
    index = store.cell_transcript_index
    assert set(index.slices) == {"18446744073709551615", "18446744073709551614"}
    coords, genes = index.transcripts_for(2**64 - 1)
    assert genes.tolist() == ["A", "A"]
    assert coords.tolist() == [[5.0, 1.0], [7.0, 3.0]]
    assert index.transcripts_for(np.uint64(2**64 - 2))[1].tolist() == ["B"]

    digits = pd.DataFrame({
        "x": [1.0, 2.0], "y": [3.0, 4.0], "gene": ["A", "B"],
        "assignment": ["1234567890123456789", "1234567890123456788"],
    })
    store = _both(digits, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment", build_cell_index=True)
    assert store.cell_transcript_index.transcripts_for(1234567890123456789)[1].tolist() == ["A"]
    assert store.cell_transcript_index.transcripts_for("1234567890123456788")[1].tolist() == ["B"]


def test_transcripts_for_reuses_one_gene_name_array():
    index = U.CellTranscriptIndex(
        coords_yx=np.zeros((3, 2), dtype=np.float32),
        gene_codes=np.asarray([0, 1, 0], dtype=np.uint8),
        gene_names=("A", "B"),
        slices={"1": (0, 3)},
    )
    first = index.transcripts_for(1)[1]
    names = index._gene_name_array
    assert first.tolist() == ["A", "B", "A"]
    assert index.transcripts_for("1")[1].tolist() == ["A", "B", "A"]
    assert index._gene_name_array is names


# ---------------------------------------------------------------------------
# Sampler beyond numpy's hypergeometric limit
# ---------------------------------------------------------------------------


def _frozen_hypergeometric_keeps(counts, render_count, seed):
    """The historical pass-2 sampling loop, on partition sizes only."""
    rng = np.random.default_rng(seed)
    remaining_source = int(sum(counts))
    remaining_sample = int(render_count)
    out = []
    for part_count in counts:
        if remaining_sample == remaining_source:
            keep = np.arange(part_count, dtype=np.int64)
        elif remaining_sample <= 0:
            keep = np.empty((0,), dtype=np.int64)
        else:
            take = int(rng.hypergeometric(
                ngood=part_count, nbad=max(0, remaining_source - part_count), nsample=remaining_sample,
            ))
            keep = (np.sort(rng.choice(part_count, size=take, replace=False))
                    if take > 0 else np.empty((0,), dtype=np.int64))
        out.append(keep)
        remaining_source -= part_count
        remaining_sample -= len(keep)
    return out


def test_sampler_keeps_hypergeometric_draws_at_and_above_1e9_total_when_numpy_accepts():
    counts = [999_999_000, 300_000_000, 600_000_000]  # total ~1.9e9, but every nbad < 1e9
    assert sum(counts) >= 10**9
    sampler = U._StreamingSampler(counts, 1_000, 21)
    assert sampler.uses_hypergeometric
    reference = _frozen_hypergeometric_keeps(counts, 1_000, 21)
    for count, expected in zip(counts, reference):
        _assert_arrays_identical(sampler.take(count), expected, "hypergeometric keeps")


def test_sampler_switches_to_global_positions_when_numpy_would_refuse():
    counts = [500_000_000, 500_000_000, 500_000_000]  # total - first == 1e9
    with pytest.raises(ValueError):
        _frozen_hypergeometric_keeps(counts, 1_000, 21)
    sampler = U._StreamingSampler(counts, 1_000, 21)
    assert not sampler.uses_hypergeometric
    positions = sample_global_positions(sum(counts), 1_000, 21)
    keeps = [sampler.take(count) for count in counts]
    offsets = np.cumsum([0] + counts[:-1])
    merged = np.concatenate([keep + off for keep, off in zip(keeps, offsets)])
    _assert_arrays_identical(merged, positions, "global positions")
    for keep, count in zip(keeps, counts):
        assert np.all((keep >= 0) & (keep < count))
        assert np.all(np.diff(keep) > 0)


def test_sampler_partition_ngood_at_limit_uses_global_positions():
    assert not U._StreamingSampler.hypergeometric_accepts([10**9, 5])
    assert not U._StreamingSampler.hypergeometric_accepts([5, 10**9])
    assert U._StreamingSampler.hypergeometric_accepts([10**9 - 1, 10**9 - 1])


# ---------------------------------------------------------------------------
# Read-ahead
# ---------------------------------------------------------------------------


def test_read_ahead_keeps_order_and_honours_cancel(monkeypatch):
    df = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0), "gene": ["A"] * 40})
    ddf = dd.from_pandas(df, npartitions=8)
    parts = list(U._iter_point_partitions(ddf, ["x", "y", "gene"], read_ahead=2))
    assert [p["x"].iloc[0] for p in parts] == sorted(p["x"].iloc[0] for p in parts)
    assert pd.concat(parts)["x"].tolist() == list(range(40))

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(U.TranscriptBuildCancelled):
        list(U._iter_point_partitions(ddf, ["x", "y", "gene"], cancel_check=cancel, read_ahead=2))
