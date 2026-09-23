"""Focused tests for the external prepared-transcript cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_compare_xenium_merscope.transcript_cache import (
    TranscriptCacheRequest,
    default_transcript_cache_dir,
    load_transcript_cache,
    save_transcript_cache,
)
from napari_compare_xenium_merscope.utils import (
    CellTranscriptIndex,
    MarkerReferenceResolution,
    TranscriptBuildCancelled,
    build_gene_point_groups,
    gene_panel_fingerprint,
)


REFERENCE = {
    "A": {"broad": "Neuron", "fine": "Excitatory"},
    "B": {"broad": "Astrocyte", "fine": "Astrocyte"},
}


def _source_store(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "sample" / "spatialdata.zarr"
    points = dataset / "points" / "transcripts"
    points.mkdir(parents=True)
    # These need only model the local files whose identity guards a cache entry;
    # transcript streaming itself is exercised with the DataFrame below.
    (points / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group"}), encoding="utf-8"
    )
    part = points / "part.0.parquet"
    part.write_bytes(b"first parquet identity")
    return dataset, part


def _request(dataset: Path, cache_root: Path, **overrides) -> TranscriptCacheRequest:
    values = {
        "dataset_path": dataset,
        "points_key": "transcripts",
        "x_col": "x",
        "y_col": "y",
        "gene_col": "gene",
        "assignment_col": "assignment",
        "background_col": None,
        "max_points": 10_000,
        "random_state": 42,
        "build_cell_index": True,
        "cache_root": cache_root,
    }
    values.update(overrides)
    request = TranscriptCacheRequest.create(**values)
    assert request is not None
    return request


def _store(reference=REFERENCE):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [11.0, 12.0, 13.0, 14.0, 15.0],
            "gene": ["A", "B", "A", "Blank-1", "B"],
            "assignment": [101, 101, 0, 202, 202],
        }
    )
    return build_gene_point_groups(
        frame,
        x_col="x",
        y_col="y",
        gene_col="gene",
        assignment_col="assignment",
        reference=reference,
        build_cell_index=True,
    )


def _resolution(reference=REFERENCE, *, source="test reference"):
    matched = 0 if reference is None else len(reference)
    return MarkerReferenceResolution(
        reference=reference,
        source=source,
        panel_fingerprint=gene_panel_fingerprint(["A", "B"]),
        panel_gene_count=2,
        matched_gene_count=matched,
        broad_gene_count=matched,
        fine_gene_count=matched,
    )


def _dataset_snapshot(dataset: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(dataset)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in dataset.rglob("*")
        if path.is_file()
    }


def test_round_trip_restores_complete_store_and_memory_maps(tmp_path):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "external-cache")
    original = _store()
    resolution = _resolution()

    cache_path = save_transcript_cache(request, original, resolution)
    assert cache_path is not None
    hit = load_transcript_cache(request, lambda genes: resolution)

    assert hit is not None
    assert hit.path == cache_path
    restored = hit.store
    assert restored.group_symbols == original.group_symbols
    assert restored.gene_offsets == original.gene_offsets
    assert restored.gene_counts == original.gene_counts
    assert restored.genes == original.genes
    assert restored.control_genes == original.control_genes
    assert restored.total_points == original.total_points
    assert restored.sampled == original.sampled
    assert restored.source_gene_counts == original.source_gene_counts
    assert restored.source_total_points == original.source_total_points
    assert restored.gene_visuals == original.gene_visuals
    for expected, actual in zip(original.group_coords, restored.group_coords, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert isinstance(actual, np.memmap)
    for expected, actual in zip(original.group_colors, restored.group_colors, strict=True):
        np.testing.assert_allclose(actual, expected)

    expected_index = original.cell_transcript_index
    actual_index = restored.cell_transcript_index
    assert expected_index is not None and actual_index is not None
    np.testing.assert_array_equal(actual_index.coords_yx, expected_index.coords_yx)
    np.testing.assert_array_equal(actual_index.gene_codes, expected_index.gene_codes)
    assert isinstance(actual_index.coords_yx, np.memmap)
    assert isinstance(actual_index.gene_codes, np.memmap)
    assert actual_index.gene_names == expected_index.gene_names
    assert actual_index.slices == expected_index.slices
    assert hit.reference_resolution == resolution

    # The on-disk representation remains inspectable and cannot execute pickle.
    assert {path.suffix for path in cache_path.iterdir()} <= {".json", ".npy"}
    assert not list(cache_path.rglob("*.pkl"))


def test_round_trip_allows_panel_gene_omitted_by_render_sampling(tmp_path):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "cache", max_points=1)
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [11.0, 12.0],
            "gene": ["A", "B"],
            "assignment": [101, 202],
        }
    )
    original = build_gene_point_groups(
        frame,
        x_col="x",
        y_col="y",
        gene_col="gene",
        assignment_col="assignment",
        reference=REFERENCE,
        max_points=1,
        build_cell_index=True,
    )
    assert len(original.genes) == 2
    assert len(original.gene_offsets) == 1

    assert save_transcript_cache(request, original, _resolution()) is not None
    hit = load_transcript_cache(request, lambda genes: _resolution())

    assert hit is not None
    assert hit.store.genes == original.genes
    assert hit.store.gene_offsets == original.gene_offsets
    assert hit.store.gene_counts == original.gene_counts


def test_large_cell_slice_map_is_stored_outside_manifest(tmp_path):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "cache")
    store = _store()
    cell_count = 10_000
    store.cell_transcript_index = CellTranscriptIndex(
        coords_yx=np.zeros((cell_count, 2), dtype=np.float32),
        gene_codes=np.zeros(cell_count, dtype=np.uint8),
        gene_names=("A",),
        slices={f"cell-{index:08d}": (index, index + 1) for index in range(cell_count)},
    )

    cache_path = save_transcript_cache(request, store, _resolution())
    assert cache_path is not None
    manifest_path = cache_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_manifest = manifest["cell_transcript_index"]

    assert manifest_path.stat().st_size < 100_000
    assert "slices" not in cell_manifest
    assert (cache_path / cell_manifest["slice_keys"]).is_file()
    assert (cache_path / cell_manifest["slice_spans"]).is_file()
    hit = load_transcript_cache(request, lambda genes: _resolution())
    assert hit is not None
    assert len(hit.store.cell_transcript_index.slices) == cell_count


def test_point_file_stat_change_invalidates_request(tmp_path):
    dataset, point_file = _source_store(tmp_path)
    cache_root = tmp_path / "cache"
    before = _request(dataset, cache_root)
    assert save_transcript_cache(before, _store(), _resolution()) is not None

    # Preserve both bytes and size: the source file's stat identity alone must
    # invalidate the cache, as it would after a parquet partition is replaced.
    old_stat = point_file.stat()
    os.utime(
        point_file,
        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000),
    )
    after = _request(dataset, cache_root)

    assert after.digest != before.digest
    assert load_transcript_cache(after, lambda genes: _resolution()) is None


def test_successful_save_prunes_superseded_source_and_reference_entries(tmp_path):
    dataset, point_file = _source_store(tmp_path)
    cache_root = tmp_path / "cache"
    before = _request(dataset, cache_root)
    first_path = save_transcript_cache(before, _store(), _resolution())
    assert first_path is not None

    old_stat = point_file.stat()
    os.utime(
        point_file,
        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 2_000_000_000),
    )
    after = _request(dataset, cache_root)
    assert after.family_digest == before.family_digest
    assert after.digest != before.digest
    current_path = save_transcript_cache(after, _store(), _resolution())

    assert current_path is not None
    assert not before.entry_root.exists()
    changed_reference = {
        **REFERENCE,
        "A": {"broad": "Neuron", "fine": "Inhibitory"},
    }
    changed_resolution = _resolution(changed_reference)
    changed_path = save_transcript_cache(
        after, _store(changed_reference), changed_resolution
    )
    assert changed_path is not None
    assert changed_path.exists()
    assert not current_path.exists()


def test_cancelled_save_removes_partial_entry(tmp_path):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "cache")
    checks = 0

    def cancel_after_work_starts():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(TranscriptBuildCancelled):
        save_transcript_cache(
            request,
            _store(),
            _resolution(),
            cancel_check=cancel_after_work_starts,
        )

    assert not list(request.entry_root.glob(".writing-*"))


@pytest.mark.parametrize(
    "changed",
    [
        {"x_col": "global_x"},
        {"assignment_col": None},
        {"max_points": 4},
        {"random_state": 7},
        {"build_cell_index": False},
    ],
)
def test_build_configuration_changes_invalidate_request(tmp_path, changed):
    dataset, _part = _source_store(tmp_path)
    cache_root = tmp_path / "cache"
    before = _request(dataset, cache_root)
    after = _request(dataset, cache_root, **changed)

    assert after.digest != before.digest


def test_current_marker_reference_must_match_cached_reference(tmp_path):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "cache")
    old_resolution = _resolution()
    assert save_transcript_cache(request, _store(), old_resolution) is not None

    changed_reference = {
        **REFERENCE,
        "A": {"broad": "Neuron", "fine": "Inhibitory"},
    }
    changed_resolution = _resolution(changed_reference)

    assert load_transcript_cache(request, lambda genes: changed_resolution) is None
    assert load_transcript_cache(request, lambda genes: old_resolution) is not None


@pytest.mark.parametrize("corrupt", ["manifest", "array"])
def test_corrupt_entry_is_a_cache_miss(tmp_path, corrupt):
    dataset, _part = _source_store(tmp_path)
    request = _request(dataset, tmp_path / "cache")
    resolution = _resolution()
    cache_path = save_transcript_cache(request, _store(), resolution)
    assert cache_path is not None

    if corrupt == "manifest":
        (cache_path / "manifest.json").write_text("{not json", encoding="utf-8")
    else:
        array_path = next(cache_path.glob("group_*_coords.npy"))
        array_path.write_bytes(b"not a numpy array")

    assert load_transcript_cache(request, lambda genes: resolution) is None


def test_cache_is_external_and_never_mutates_source_dataset(tmp_path):
    dataset, _part = _source_store(tmp_path)
    before = _dataset_snapshot(dataset)
    cache_root = tmp_path / "cache-sibling"
    request = _request(dataset, cache_root)

    cache_path = save_transcript_cache(request, _store(), _resolution())
    assert cache_path is not None
    assert load_transcript_cache(request, lambda genes: _resolution()) is not None
    assert _dataset_snapshot(dataset) == before
    assert not cache_path.is_relative_to(dataset)

    forbidden = dataset / ".viewer-transcript-cache"
    assert TranscriptCacheRequest.create(
        dataset_path=dataset,
        points_key="transcripts",
        x_col="x",
        y_col="y",
        gene_col="gene",
        cache_root=forbidden,
    ) is None
    assert not forbidden.exists()
    assert _dataset_snapshot(dataset) == before


def test_unwritable_cache_location_does_not_fail_a_completed_build(tmp_path):
    dataset, _part = _source_store(tmp_path)
    cache_root = tmp_path / "not-a-directory"
    cache_root.write_text("blocks cache directory creation", encoding="utf-8")
    request = _request(dataset, cache_root)

    assert save_transcript_cache(request, _store(), _resolution()) is None


def test_default_cache_directory_honours_external_environment_override(tmp_path, monkeypatch):
    configured = tmp_path / "custom-cache"
    monkeypatch.setenv("NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR", str(configured))
    assert default_transcript_cache_dir() == configured.resolve()
