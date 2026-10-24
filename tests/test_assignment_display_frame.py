"""Aligned display coordinates join to pciSeq assignments by exact transcript IDs."""

from __future__ import annotations

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope.utils import assignment_points_in_display_frame


@pytest.fixture
def points():
    largest = 2**64 - 1
    assignments = pd.DataFrame(
        {
            "transcript_id": np.array([largest, 0, 7, 8], dtype=np.uint64),
            "feature_name": ["A", "B", "C", "D"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [4.0, 3.0, 2.0, 1.0],
            "assignment": pd.array([largest, None, None, 2], dtype="UInt64"),
            "background": [False, True, False, False],
            "pciseq_assignment_status": pd.Categorical(
                ["assigned", "background", "not_modelled", "assigned"],
                categories=["assigned", "background", "not_modelled"],
            ),
        }
    )
    display = pd.DataFrame(
        {
            "transcript_id": np.array([7, largest, 8, 0], dtype=np.uint64),
            "gene": ["C", "A", "D", "B"],
            "aligned_x": [111.0, 777.0, -321.0, 909.0],
            "aligned_y": [-7.0, 99.0, 1200.0, 2.0],
            # Existing display assignments must never replace pciSeq's posterior assignments.
            "assignment": [999, 999, 999, 999],
            "background": [True] * 4,
        }
    )
    return assignments, display


def _join(assignments, display):
    return assignment_points_in_display_frame(
        assignments,
        display,
        assignment_gene_col="feature_name",
        display_gene_col="gene",
        coordinate_cols=("aligned_x", "aligned_y"),
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_shuffled_ids_use_nonlinear_display_coordinates_and_pciseq_assignments(
    points, lazy
):
    assignments, display = points
    if lazy:
        assignments = dd.from_pandas(assignments, npartitions=3, sort=False)
        display = dd.from_pandas(display, npartitions=2, sort=False)
    joined = _join(assignments, display)
    if lazy:
        assert isinstance(joined, dd.DataFrame)
        joined = joined.compute(scheduler="synchronous")
    by_id = joined.set_index("transcript_id")
    assert set(joined.columns) == {
        "transcript_id",
        "gene",
        "aligned_x",
        "aligned_y",
        "assignment",
        "background",
        "pciseq_assignment_status",
    }
    assert by_id.loc[2**64 - 1, "aligned_x"] == 777
    assert by_id.loc[2**64 - 1, "aligned_y"] == 99
    assert by_id.loc[2**64 - 1, "assignment"] == 2**64 - 1
    assert by_id.loc[8, "assignment"] == 2
    assert pd.isna(by_id.loc[0, "assignment"])
    assert by_id.loc[0, "pciseq_assignment_status"] == "background"
    assert bool(by_id.loc[0, "background"])
    assert pd.isna(by_id.loc[7, "assignment"])
    assert by_id.loc[7, "pciseq_assignment_status"] == "not_modelled"
    assert not bool(by_id.loc[7, "background"])
    assert by_id.loc[7, "gene"] == "C"


@pytest.mark.parametrize("side", ["assignments", "display"])
@pytest.mark.parametrize("problem", ["duplicate", "missing", "gene"])
def test_global_id_or_gene_mismatch_fails_across_partitions(points, side, problem):
    assignments, display = points
    frame = assignments if side == "assignments" else display
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "missing":
        frame = frame.iloc[1:].copy()
    else:
        frame = frame.copy()
        frame.loc[
            frame.index[0], "feature_name" if side == "assignments" else "gene"
        ] = "wrong"
    if side == "assignments":
        assignments = frame
    else:
        display = frame
    joined = _join(
        dd.from_pandas(assignments, npartitions=3, sort=False),
        dd.from_pandas(display, npartitions=2, sort=False),
    )
    with pytest.raises(ValueError, match="duplicate|missing|different genes"):
        joined.compute(scheduler="synchronous")


def test_dask_display_join_does_not_materialize_inputs_during_discovery(points):
    assignments, display = points
    seen = []

    @dask.delayed
    def read(frame, key):
        seen.append(key)
        return frame

    assignment_lazy = dd.from_delayed(
        [read(assignments, "assignment")], meta=assignments.iloc[:0]
    )
    display_lazy = dd.from_delayed([read(display, "display")], meta=display.iloc[:0])
    joined = _join(assignment_lazy, display_lazy)
    assert seen == []
    joined.compute(scheduler="synchronous")
    assert sorted(seen) == ["assignment", "display"]


def test_dask_display_join_preserves_spatialdata_attrs_accessor(points):
    import spatialdata  # noqa: F401 -- installs the supported Dask attrs accessor

    assignments, display = points
    assignment_lazy = dd.from_pandas(assignments, npartitions=2, sort=False)
    display_lazy = dd.from_pandas(display, npartitions=2, sort=False)
    display_lazy.attrs["display_frame"] = "aligned"
    joined = _join(assignment_lazy, display_lazy)
    assert joined.attrs["display_frame"] == "aligned"
    assert (
        len(joined[["aligned_x", "assignment"]].compute(scheduler="synchronous")) == 4
    )


@pytest.mark.parametrize("bad_id", [-1, float(2**64 - 1), None, str(2**64)])
def test_unsafe_transcript_ids_are_rejected(points, bad_id):
    assignments, display = points
    assignments = assignments.copy()
    assignments["transcript_id"] = assignments.transcript_id.astype(object)
    assignments.loc[0, "transcript_id"] = bad_id
    with pytest.raises(ValueError, match="exact uint64"):
        _join(assignments, display)
