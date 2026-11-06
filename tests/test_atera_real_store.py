"""Opt-in smoke test against a real Atera SpatialData store (read-only).

Skipped unless ``NAPARI_COMPARE_ATERA_STORE`` points at a MerXen Atera store,
for example ``.../Atera/latest_spatialdata.zarr/latest_spatialdata.zarr``. CI
never sets it, so every other Atera test runs on the synthetic fixture in
``tests/atera_stores.py``.

Nothing here writes to the store or to the transcript cache: the engine reads
parquet footers and row groups, the code-range map is built in memory, and the
pixel size and detection read JSON metadata only. The expected numbers are
those of the MerXen Atera run the viewer was developed against (409 files,
1,409,601,228 transcripts); a different store fails the exact-count checks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

STORE_ENV = "NAPARI_COMPARE_ATERA_STORE"

pytestmark = pytest.mark.skipif(
    not os.environ.get(STORE_ENV),
    reason=f"set {STORE_ENV} to a real Atera store to run",
)

EXPECTED_FILES = 409
EXPECTED_ROWS = 1_409_601_228
EXPECTED_VOCABULARY = 24_677
EXPECTED_PIXEL_SIZE_UM = 0.21249222
CELL_ID = 1001
CELL_TABLE_TOTAL = 17_199
CELL_EXACT_ROWS = 17_202


@pytest.fixture(scope="module")
def store() -> Path:
    path = Path(os.environ[STORE_ENV])
    assert path.is_dir(), f"{STORE_ENV}={path} is not a directory"
    return path


@pytest.fixture(scope="module")
def engine(store):
    from napari_compare_xenium_merscope import atera_transcripts as T

    opened = T.AteraTranscriptEngine.open(store)
    assert opened is not None
    return opened


def test_detection_is_atera_and_authoritative(store):
    from napari_compare_xenium_merscope.platforms import ATERA, detect_store_platform

    detection = detect_store_platform(store)
    assert detection.platform == ATERA
    assert detection.authoritative, detection.evidence


def test_pixel_size_comes_from_the_store(store):
    from napari_compare_xenium_merscope import utils as U

    (sx, kx, tx), (ky, sy, ty) = U.resolve_dataset_mask_affine("ATERA", zarr_path=store)
    assert sx == pytest.approx(EXPECTED_PIXEL_SIZE_UM, rel=1e-9)
    assert sy == pytest.approx(EXPECTED_PIXEL_SIZE_UM, rel=1e-9)
    assert (kx, tx, ky, ty) == (0.0, 0.0, 0.0, 0.0)


def test_points_index_is_validated_with_one_shared_dictionary(engine):
    index = engine.index
    assert index.validated
    assert len(index.files) == EXPECTED_FILES
    assert index.n_rows == EXPECTED_ROWS
    assert index.shared_dictionary
    assert len(index.vocabulary) == EXPECTED_VOCABULARY
    assert index.assignment_dtype == np.uint32


def test_code_range_gene_read_matches_the_statistics_read(engine):
    """One gene read twice: string-statistics pruning, then the exact code-range map."""
    gene = "GFAP"
    assert engine.index.code_of(gene) is not None
    if engine.code_map is None:
        by_stats = engine.read_genes([gene])[gene]
        code_map = engine.build_code_map()
        assert code_map is not None and engine.code_map is code_map
    else:  # an already attached map: compare against a fresh statistics read
        attached = engine.code_map
        engine.attach_code_map(None)
        try:
            by_stats = engine.read_genes([gene])[gene]
        finally:
            engine.attach_code_map(attached)
    by_map = engine.read_genes([gene])[gene]
    n = len(by_map["x"])
    assert n > 1_000_000
    assert n == len(by_stats["x"]) == engine.code_map.rows_for([engine.index.code_of(gene)])
    for column in by_map:
        assert np.array_equal(by_map[column], by_stats[column]), column


def test_cell_1001_table_counts_and_exact_rows(store, engine):
    import spatialdata as sd

    from napari_compare_xenium_merscope import utils as U

    counts = U.table_cell_gene_counts(store, "table_original", CELL_ID)
    assert counts is not None and counts.total == CELL_TABLE_TOTAL
    shapes = sd.read_zarr(store, selection=("shapes",)).shapes["atera_cell_boundaries"]
    rows = engine.read_cell(CELL_ID, shapes.geometry.loc[CELL_ID].bounds)
    assert rows is not None and rows.n == CELL_EXACT_ROWS
