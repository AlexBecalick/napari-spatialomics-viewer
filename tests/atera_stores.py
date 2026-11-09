"""Synthetic 10x Atera ``latest_spatialdata.zarr`` stores (and P7513-like attribute stores).

This is a plain helper module (no ``test_`` prefix, no pytest fixtures). pytest's default
``prepend`` import mode puts ``tests/`` on ``sys.path``, so test files import it with
``import atera_stores``.

:func:`build_atera_latest_store` mirrors the edge cases of a real MerXen Atera store at a
tiny size (about 0.6 s, under 100 KB):

* ``images/morphology_focus``: four channels with the real names, two of them containing ``/``.
* ``labels``: vendor ``cell_labels`` / ``nucleus_labels`` plus the MerXen-managed
  ``atera_*_labels`` aliases carrying the ``napari_compare_label_cache`` marker
  (``source: vendor_labels``).
* ``shapes/atera_cell_boundaries``: uint64 index named ``instance_id`` (from 1) with a
  duplicate ``instance_id`` column, ``source_cell_id`` and ``segmentation_method``.
  ``shapes/atera_nucleus`` has its own ids plus ``cell_instance_id``: cell 7 has two nuclei,
  cell 6 has none and nucleus 8 straddles the boundary of cell 8. Cell 3 straddles the
  ``0,0`` / ``1,0`` tile boundary, so its transcripts sit in two tiles.
* ``points/transcripts``: per-tile files split into fixed raw-row chunks, each filtered and
  sorted by feature code, sharing ONE ``feature_name`` dictionary (int16 indices) with the
  genes first in sorted order and five control families appended. ``original_assignment``
  is nullable uint64. Files are ``part.N.parquet`` (so ``part.10`` sorts before ``part.2``);
  ``part.0`` is the off-grid tile ``-1,1`` with a negative-x bbox; the last file is a tiny
  single-row-group file. Row groups are ``row_group_size`` rows.
* ``tables``: ``table_original`` (vendor Azimuth columns, ``X_umap_vendor``), the clustering
  table with one cell dropped, and the image-quantification table whose
  ``mask_label_key`` is ``cell_labels``.
* Root ``merxen_schema`` (platform ATERA, ``original`` / ``original_nucleus``
  segmentations), ``merxen_atera.experiment.pixel_size``, ``merxen_native_latest.json``,
  ``atera/experiment.spatial`` (optional override) and ``merxen_atera_points_index.json``
  in the real nested shape.

All values are synthetic.
"""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_NAME = "SYN_ATERA"  # synthetic alias
CHANNELS = ("DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin")
GENES = (
    "ACTB", "ALDH1L1", "AQP4", "BDNF", "CALB1", "CAMK2A", "CNP", "CUX2", "DCN", "EGFR",
    "FOXP2", "GAD1", "GAD2", "GFAP", "GJA1", "HEXB", "IBA1", "KCNIP4", "LAMP5", "MBP",
    "MOG", "NRGN", "OLIG2", "PDGFRA", "PLP1", "PVALB", "RORB", "SATB2", "SLC17A7", "SNAP25",
    "SST", "TBR1", "VIP", "ZIC1",
)
#: In the dictionary but never observed (a panel gene with no transcripts).
ABSENT_GENE = "CUX2"
CONTROLS = (
    "NegControlProbe_00002",
    "NegControlProbe_00003",
    "NegControlCodeword_0500",
    "NegControlCodeword_0501",
    "Intergenic_Region_1",
    "UnassignedCodeword_0001",
    "UnassignedCodeword_0002",
    "DeprecatedCodeword_0001",
)
CONTROL_FAMILIES = (
    "NegControlProbe",
    "NegControlCodeword",
    "Intergenic_Region",
    "UnassignedCodeword",
    "DeprecatedCodeword",
)
FEATURE_DICTIONARY = GENES + CONTROLS
TILE_SIZE_UM = 20.0
EXTENT_UM = 40.0  # the on-grid tiles cover [0, 40) x [0, 40) um
RAW_CHUNK_ROWS = 48  # vendor rows per file before the quality filter

#: (instance_id, xmin, ymin, xmax, ymax) in um.
CELLS: tuple[tuple[int, float, float, float, float], ...] = (
    (1, 2.0, 2.0, 8.0, 8.0),
    (2, 10.0, 3.0, 16.0, 9.0),
    (3, 17.0, 11.0, 24.0, 17.0),  # straddles the x = 20 um tile boundary
    (4, 24.0, 4.0, 31.0, 10.0),
    (5, 3.0, 23.0, 9.0, 29.0),
    (6, 12.0, 25.0, 18.0, 32.0),  # no nucleus
    (7, 25.0, 24.0, 33.0, 31.0),  # two nuclei
    (8, 30.0, 33.0, 37.0, 38.0),
)
#: (instance_id, cell_instance_id, xmin, ymin, xmax, ymax) in um.
NUCLEI: tuple[tuple[int, int, float, float, float, float], ...] = (
    (1, 1, 4.0, 4.0, 6.0, 6.0),
    (2, 2, 12.0, 5.0, 14.0, 7.0),
    (3, 3, 19.0, 13.0, 21.0, 15.0),
    (4, 4, 26.0, 6.0, 28.0, 8.0),
    (5, 5, 5.0, 25.0, 7.0, 27.0),
    (6, 7, 26.0, 26.0, 28.0, 28.0),
    (7, 7, 30.0, 27.0, 32.0, 29.0),
    (8, 8, 35.0, 35.0, 39.0, 37.0),  # straddles the cell 8 boundary (x = 37 um)
)
SEGMENTATION_METHODS = (
    "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)",
    "Segmented by interior stain (18S)",
    "Segmented by nucleus expansion of 5.0µm",
)
#: Cell dropped from the clustering table (filtered by QC there).
CLUSTERING_DROPPED_CELL = 8
#: (tile, raw rows, x range, y range). Order fixes the part numbering.
TILES: tuple[tuple[str, int, tuple[float, float], tuple[float, float]], ...] = (
    ("-1,1", 5, (-2.0, -0.03125), (20.3, 30.0)),
    ("0,0", 144, (0.0, 19.984375), (0.0, 19.984375)),
    ("1,0", 96, (20.0, 39.984375), (0.0, 19.984375)),
    ("0,1", 96, (0.0, 19.984375), (20.0, 39.984375)),
    ("1,1", 106, (20.0, 39.984375), (20.0, 39.984375)),
)
TILE_NAMES = tuple(tile for tile, *_rest in TILES)
QUANT_STATISTICS = ("min", "median", "mean", "max", "iqr")


def control_family(name: str) -> str | None:
    """Return the control family of a feature name, or ``None`` for a gene."""
    for family in CONTROL_FAMILIES:
        if str(name).startswith(family):
            return family
    return None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_zarr_attributes(group_dir: Path, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into a zarr v3 group's ``zarr.json`` attributes."""
    path = Path(group_dir) / "zarr.json"
    meta = _read_json(path)
    attrs = dict(meta.get("attributes") or {})
    attrs.update(updates)
    meta["attributes"] = attrs
    _write_json(path, meta)


def _inside(xs: np.ndarray, ys: np.ndarray, rect: Sequence[float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = rect
    return (xs >= xmin) & (xs < xmax) & (ys >= ymin) & (ys < ymax)


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def synthetic_transcripts(seed: int = 7513) -> pd.DataFrame:
    """Every vendor row (before the quality filter) with tile, chunk and keep flags."""
    rng = np.random.default_rng(seed)
    present_genes = [i for i, g in enumerate(GENES) if g != ABSENT_GENE]
    control_codes = list(range(len(GENES), len(FEATURE_DICTIONARY)))
    frames = []
    next_id = 1
    for tile, n_raw, (x0, x1), (y0, y1) in TILES:
        xs = rng.uniform(x0, x1, n_raw).astype(np.float32)
        ys = rng.uniform(y0, y1, n_raw).astype(np.float32)
        # Pull a third of the rows into the cells overlapping this tile, so every cell has
        # transcripts and cell 3 (which straddles x = 20 um) has them in two tiles.
        cells = [c for c in CELLS if c[1] < x1 and c[3] > x0 and c[2] < y1 and c[4] > y0]
        for k in range(n_raw // 3 if cells else 0):
            _cid, cx0, cy0, cx1, cy1 = cells[k % len(cells)]
            xs[k] = rng.uniform(max(cx0, x0), min(cx1, x1))
            ys[k] = rng.uniform(max(cy0, y0), min(cy1, y1))
        is_control = rng.random(n_raw) < 0.12
        codes = np.where(
            is_control,
            rng.choice(control_codes, n_raw),
            rng.choice(present_genes, n_raw),
        ).astype(np.int16)
        qv = rng.uniform(10.0, 40.0, n_raw).astype(np.float32)
        frame = pd.DataFrame(
            {
                "tile": tile,
                "raw_row": np.arange(n_raw),
                "x": xs,
                "y": ys,
                "z": rng.uniform(6.0, 18.0, n_raw).astype(np.float32),
                "code": codes,
                "qv": qv,
                "transcript_id": np.arange(next_id, next_id + n_raw, dtype=np.uint64),
            }
        )
        next_id += n_raw
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["chunk"] = out["raw_row"] // RAW_CHUNK_ROWS
    out["keep"] = out["qv"] >= 20.0
    # Every file keeps at least one row.
    first_rows = out.groupby(["tile", "chunk"]).head(1).index
    out.loc[first_rows, "keep"] = True
    out.loc[first_rows, "qv"] = np.float32(20.0)
    # The tiny last file (one row group) spans the first gene, the last gene and a
    # control, so its feature_name statistics cover 'A..'..'Z..' plus controls.
    tiny = out[(out["tile"] == TILES[-1][0]) & (out["chunk"] == out.loc[out["tile"] == TILES[-1][0], "chunk"].max())]
    spanning = tiny.index[:3]
    out.loc[spanning, "code"] = np.array([0, len(GENES) - 1, len(GENES)], dtype=np.int16)
    out.loc[spanning, "keep"] = True
    out.loc[spanning, "qv"] = np.float32(30.0)

    xs = out["x"].to_numpy()
    ys = out["y"].to_numpy()
    assignment = np.zeros(len(out), dtype=np.uint64)
    for cid, *rect in CELLS:
        assignment[_inside(xs, ys, rect)] = cid
    out["original_assignment"] = pd.array(
        np.where(assignment > 0, assignment, 0), dtype="UInt64"
    )
    out.loc[assignment == 0, "original_assignment"] = pd.NA
    overlaps = np.zeros(len(out), dtype=bool)
    for _nid, _cid, *rect in NUCLEI:
        overlaps |= _inside(xs, ys, rect)
    out["overlaps_nucleus"] = overlaps
    out["is_gene"] = out["code"] < len(GENES)
    return out


def _arrow_file_table(frame: pd.DataFrame):
    import pyarrow as pa

    frame = frame.sort_values(["code", "transcript_id"], kind="stable").reset_index(drop=True)
    n = len(frame)
    feature = pa.DictionaryArray.from_arrays(
        pa.array(frame["code"].to_numpy(dtype=np.int16), type=pa.int16()),
        pa.array(list(FEATURE_DICTIONARY), type=pa.string()),
    )
    tile_codes = np.array([TILE_NAMES.index(t) for t in frame["tile"]], dtype=np.int8)
    source_tile = pa.DictionaryArray.from_arrays(
        pa.array(tile_codes, type=pa.int8()), pa.array(list(TILE_NAMES), type=pa.string())
    )
    assignment = frame["original_assignment"]
    columns = {
        "x": pa.array(frame["x"].to_numpy(np.float32)),
        "y": pa.array(frame["y"].to_numpy(np.float32)),
        "z": pa.array(frame["z"].to_numpy(np.float32)),
        "feature_name": feature,
        "is_gene": pa.array(frame["is_gene"].to_numpy(bool)),
        "qv": pa.array(frame["qv"].to_numpy(np.float32)),
        "overlaps_nucleus": pa.array(frame["overlaps_nucleus"].to_numpy(bool)),
        "original_assignment": pa.array(
            [None if pd.isna(v) else int(v) for v in assignment], type=pa.uint64()
        ),
        "transcript_id": pa.array(frame["transcript_id"].to_numpy(np.uint64)),
        "source_tile": source_tile,
        "__null_dask_index__": pa.array(np.arange(n, dtype=np.int64)),
    }
    table = pa.table(columns)
    pandas_meta = {
        "index_columns": ["__null_dask_index__"],
        "column_indexes": [
            {"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object",
             "metadata": {"encoding": "UTF-8"}}
        ],
        "columns": [
            {"name": "x", "field_name": "x", "pandas_type": "float32", "numpy_type": "float32", "metadata": None},
            {"name": "y", "field_name": "y", "pandas_type": "float32", "numpy_type": "float32", "metadata": None},
            {"name": "z", "field_name": "z", "pandas_type": "float32", "numpy_type": "float32", "metadata": None},
            {"name": "feature_name", "field_name": "feature_name", "pandas_type": "categorical",
             "numpy_type": "int16", "metadata": {"num_categories": len(FEATURE_DICTIONARY), "ordered": False}},
            {"name": "is_gene", "field_name": "is_gene", "pandas_type": "bool", "numpy_type": "bool", "metadata": None},
            {"name": "qv", "field_name": "qv", "pandas_type": "float32", "numpy_type": "float32", "metadata": None},
            {"name": "overlaps_nucleus", "field_name": "overlaps_nucleus", "pandas_type": "bool",
             "numpy_type": "bool", "metadata": None},
            {"name": "original_assignment", "field_name": "original_assignment", "pandas_type": "uint64",
             "numpy_type": "UInt64", "metadata": None},
            {"name": "transcript_id", "field_name": "transcript_id", "pandas_type": "uint64",
             "numpy_type": "uint64", "metadata": None},
            {"name": "source_tile", "field_name": "source_tile", "pandas_type": "categorical",
             "numpy_type": "int8", "metadata": {"num_categories": len(TILE_NAMES), "ordered": False}},
            {"name": "__null_dask_index__", "field_name": "__null_dask_index__", "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
        ],
        "attributes": {},
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": pd.__version__,
    }
    return table.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})


def _write_points_files(store: Path, transcripts: pd.DataFrame, row_group_size: int) -> list[dict]:
    """Replace the writer's points parquet with Atera-like per-tile files; return index entries."""
    import pyarrow.parquet as pq

    directory = store / "points" / "transcripts" / "points.parquet"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    entries: list[dict] = []
    part = 0
    for tile, *_rest in TILES:
        tile_rows = transcripts[transcripts["tile"] == tile]
        for chunk, raw in tile_rows.groupby("chunk", sort=True):
            kept = raw[raw["keep"]]
            table = _arrow_file_table(kept)
            name = f"part.{part}.parquet"
            pq.write_table(table, directory / name, row_group_size=int(row_group_size))
            xs = kept["x"].to_numpy(float)
            ys = kept["y"].to_numpy(float)
            start = int(chunk) * RAW_CHUNK_ROWS
            entries.append(
                {
                    "file": name,
                    "tile": tile,
                    "tile_rows": [start, start + int(len(raw))],
                    "block_bounds_um": None,
                    "n_rows": int(len(kept)),
                    "n_row_groups": int(pq.ParquetFile(directory / name).metadata.num_row_groups),
                    "bbox_um": [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
                    "transcript_id_range": [int(raw["transcript_id"].min()), int(raw["transcript_id"].max())],
                }
            )
            part += 1
    return entries


def points_index_payload(entries: list[dict], *, stale: bool = False) -> dict:
    """The ``merxen_atera_points_index.json`` payload (nested like the real one)."""
    n_rows = sum(int(e["n_rows"]) for e in entries)
    files = [dict(e) for e in entries]
    if stale:
        # Written before the points were rebuilt: one more row than the files now hold.
        n_rows += 1
        files[1]["n_rows"] = int(files[1]["n_rows"]) + 1
    return {
        "format_version": 1,
        "points": {
            "transcripts": {
                "directory": "points/transcripts/points.parquet",
                "fingerprint": {"n_files": len(files), "n_rows": n_rows},
                "files": files,
            }
        },
    }


# ---------------------------------------------------------------------------
# Rasters, shapes and tables
# ---------------------------------------------------------------------------


def _raster_shape(pixel_size: float) -> tuple[int, int]:
    n = int(np.ceil(EXTENT_UM / float(pixel_size)))
    return (n, n)


def _rasterize(rects: Sequence[tuple[int, float, float, float, float]], pixel_size: float) -> np.ndarray:
    out = np.zeros(_raster_shape(pixel_size), dtype=np.uint32)
    for label, xmin, ymin, xmax, ymax in rects:
        r0, r1 = int(round(ymin / pixel_size)), int(round(ymax / pixel_size))
        c0, c1 = int(round(xmin / pixel_size)), int(round(xmax / pixel_size))
        out[r0:r1, c0:c1] = label
    return out


def _image(pixel_size: float) -> np.ndarray:
    h, w = _raster_shape(pixel_size)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((len(CHANNELS), h, w), dtype=np.uint16)
    for i in range(len(CHANNELS)):
        plane = 800.0 + 400.0 * np.sin(xx / (7.0 + i)) * np.cos(yy / 11.0)
        out[i] = np.clip(plane, 0, 65535).astype(np.uint16)
    return out


def _cell_frame() -> pd.DataFrame:
    from shapely.geometry import box

    ids = np.array([c[0] for c in CELLS], dtype=np.uint64)
    frame = pd.DataFrame(
        {
            "instance_id": ids,
            "source_cell_id": [f"aaaa{chr(ord('a') + int(i))}aaa-1" for i in ids],
            "segmentation_method": [SEGMENTATION_METHODS[int(i) % 3] for i in ids],
            "geometry": [box(c[1], c[2], c[3], c[4]) for c in CELLS],
        },
        index=pd.Index(ids, name="instance_id"),
    )
    return frame


def _nucleus_frame() -> pd.DataFrame:
    from shapely.geometry import box

    ids = np.array([n[0] for n in NUCLEI], dtype=np.uint64)
    return pd.DataFrame(
        {
            "instance_id": ids,
            "cell_instance_id": np.array([n[1] for n in NUCLEI], dtype=np.uint64),
            "source_cell_id": [f"aaaa{chr(ord('a') + n[1])}aaa-1" for n in NUCLEI],
            "segmentation_method": "Segmented by nuclear stain (DAPI) multi-focus",
            "geometry": [box(n[2], n[3], n[4], n[5]) for n in NUCLEI],
        },
        index=pd.Index(ids, name="instance_id"),
    )


def _cell_counts(transcripts: pd.DataFrame) -> np.ndarray:
    kept = transcripts[transcripts["keep"] & transcripts["is_gene"] & transcripts["original_assignment"].notna()]
    counts = np.zeros((len(CELLS), len(GENES)), dtype=np.float32)
    for cid, code in zip(kept["original_assignment"].astype(int), kept["code"].astype(int)):
        counts[cid - 1, code] += 1
    return counts


def _vendor_obs(rng: np.random.Generator) -> pd.DataFrame:
    ids = np.array([c[0] for c in CELLS], dtype=np.uint64)
    broad = ["Neuron", "Neuron", "Glial cell", "Glial cell", "Neuron", "Endothelial cell", "Glial cell", "Neuron"]
    fine = ["L2/3 IT", "Pvalb", "Astrocyte", "Oligodendrocyte", "L5 ET", "Arterial EC", "Microglia", "Sst"]
    return pd.DataFrame(
        {
            "instance_id": ids,
            "source_cell_id": [f"aaaa{chr(ord('a') + int(i))}aaa-1" for i in ids],
            "region": pd.Categorical(["atera_cell_boundaries"] * len(ids)),
            "vendor_azimuth_broad": pd.Categorical(broad),
            "vendor_azimuth_coarse": pd.Categorical(fine),
            "vendor_azimuth_fine": pd.Categorical(fine),
            "vendor_azimuth_full_hierarchy": pd.Categorical([f"{b}|{f}" for b, f in zip(broad, fine)]),
            "vendor_azimuth_score": rng.uniform(0.5, 1.0, len(ids)).astype(np.float32),
            "cell_area": np.array([(c[3] - c[1]) * (c[4] - c[2]) for c in CELLS], dtype=float),
            "nucleus_count": np.array([sum(n[1] == c[0] for n in NUCLEI) for c in CELLS], dtype=np.int64),
        },
        index=pd.Index([str(int(i)) for i in ids]),
    )


def _tables(transcripts: pd.DataFrame, pixel_size: float, seed: int) -> dict[str, Any]:
    import anndata as ad
    from scipy import sparse
    from spatialdata.models import TableModel

    rng = np.random.default_rng(seed)
    counts = _cell_counts(transcripts)
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(len(GENES))]}, index=list(GENES))
    obs = _vendor_obs(rng)
    centroids = np.array([((c[1] + c[3]) / 2, (c[2] + c[4]) / 2) for c in CELLS], dtype=float)
    original = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=obs,
        var=var,
        obsm={
            "X_umap_vendor": rng.normal(size=(len(CELLS), 2)),
            "X_pca_vendor": rng.normal(size=(len(CELLS), 10)),
            "spatial": centroids,
        },
        uns={"vendor_azimuth_broad_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
    )
    tables = {
        "table_original": TableModel.parse(
            original, region="atera_cell_boundaries", region_key="region", instance_key="instance_id"
        )
    }

    keep = obs["instance_id"].to_numpy() != CLUSTERING_DROPPED_CELL
    clust_obs = obs.loc[keep].copy()
    n = int(keep.sum())
    broad_class = ["Neurons", "Neurons", "Astrocytes", "Oligodendrocytes", "Neurons", "Vascular", "Microglia"][:n]
    clust_obs["broad_class"] = pd.Categorical(broad_class)
    clust_obs["hierarchical_cluster"] = pd.Categorical([f"{b}:{i % 2}" for i, b in enumerate(broad_class)])
    clust_obs["neuron_split_label"] = pd.Categorical(
        ["Excitatory" if b == "Neurons" and i % 2 == 0 else ("Inhibitory" if b == "Neurons" else "not_neuron")
         for i, b in enumerate(broad_class)],
        categories=["Excitatory", "Inhibitory", "not_neuron"],
    )
    clust_obs["leiden"] = pd.Categorical([str(i % 3) for i in range(n)])
    clustering = ad.AnnData(
        X=sparse.csr_matrix(counts[keep]),
        obs=clust_obs,
        var=var.copy(),
        obsm={
            "X_umap": rng.normal(size=(n, 2)),
            "X_umap_vendor": original.obsm["X_umap_vendor"][keep],
            "spatial": centroids[keep],
        },
    )
    tables["table_original_clustering_squidpy"] = TableModel.parse(
        clustering, region="atera_cell_boundaries", region_key="region", instance_key="instance_id"
    )

    features = [(ch, stat) for ch in CHANNELS for stat in QUANT_STATISTICS]
    qvar = pd.DataFrame(
        {
            "image_key": ["morphology_focus"] * len(features),
            "channel": [ch for ch, _ in features],
            "statistic": [stat for _, stat in features],
        },
        index=[f"morphology_focus__{ch}__{stat}" for ch, stat in features],
    )
    ids = obs["instance_id"].to_numpy()
    pixel_area = float(pixel_size) ** 2
    qobs = pd.DataFrame(
        {
            "instance_id": ids,
            "source_cell_id": obs["source_cell_id"].to_numpy(),
            "label_id": ids.astype(np.int64),
            "mask_pixel_count": np.round(obs["cell_area"].to_numpy() / pixel_area).astype(np.int64),
            "region": pd.Categorical(["atera_cell_boundaries"] * len(ids)),
        },
        index=pd.Index([str(int(i)) for i in ids], name="obs_id"),
    )
    quant = ad.AnnData(
        X=rng.uniform(100, 2000, size=(len(ids), len(features))).astype(np.float32),
        obs=qobs,
        var=qvar,
        uns={
            "mask_image_quantification": {
                "statistics": list(QUANT_STATISTICS),
                "table_key": "table_atera_cell_boundaries_image_quantification",
                "shape_key": "atera_cell_boundaries",
                "mask_label_key": "cell_labels",
            }
        },
    )
    tables["table_atera_cell_boundaries_image_quantification"] = TableModel.parse(
        quant, region="atera_cell_boundaries", region_key="region", instance_key="instance_id"
    )
    return tables


def merxen_schema_attr() -> dict[str, Any]:
    """The root ``merxen_schema`` attribute of a MerXen Atera store (trimmed)."""
    return {
        "schema_version": "1.0.0",
        "primary_points": "transcripts",
        "instance_id": {"column": "instance_id", "dtype": "uint64", "minimum": 1,
                        "raster_background": 0, "unassigned": None},
        "transcript_id": {"column": "transcript_id", "dtype": "uint64", "minimum": 1},
        "segmentations": {
            "original": {
                "points": "transcripts",
                "assignment_column": "original_assignment",
                "shape": "atera_cell_boundaries",
                "table": "table_original",
                "instance_key": "instance_id",
                "id_namespace": "source_segmentation_v1",
                "background_column": None,
                "assignment_source_column": None,
                "legacy_aliases": ["original_seg"],
            },
            "original_nucleus": {
                "points": "transcripts",
                "assignment_column": None,
                "shape": "atera_nucleus",
                "table": None,
                "instance_key": "instance_id",
                "id_namespace": "original_nucleus_row_v1",
                "background_column": None,
                "assignment_source_column": None,
                "legacy_aliases": [],
                "cell_key": "cell_instance_id",
            },
        },
        "platform": "ATERA",
        "segmentation_source": "native",
        "native_segmentation": {
            "source": "vendor",
            "platform": "ATERA",
            "cell_shapes": "atera_cell_boundaries",
            "nucleus_shapes": "atera_nucleus",
            "label_elements": ["cell_labels", "nucleus_labels"],
        },
    }


def experiment_spatial_payload(pixel_size: float) -> dict[str, Any]:
    return {
        "major_version": 1,
        "minor_version": 2,
        "run_name": "SYNTHETIC",
        "region_name": DATASET_NAME,
        "panel_wta": {"panel_num_targets": len(GENES)},
        "pixel_size": float(pixel_size),
        "z_step_size": 3.0,
        "num_cells": len(CELLS),
    }


def _label_marker(source_label_key: str, shape_key: str, shape: tuple[int, int], *,
                  id_namespace: str, cell_join_key: str | None) -> dict[str, Any]:
    marker = {
        "version": 2,
        "complete": True,
        "source_shape_key": shape_key,
        "shape": [int(shape[0]), int(shape[1])],
        "chunks": [int(shape[0]), int(shape[1])],
        "source": "vendor_labels",
        "source_label_key": source_label_key,
        "source_fingerprint": "sha256:" + ("a" if source_label_key == "cell_labels" else "b") * 64,
        "link_mode": "hardlink",
        "id_namespace": id_namespace,
    }
    if cell_join_key is not None:
        marker["cell_join_key"] = cell_join_key
    return marker


def build_atera_latest_store(
    root: Path,
    *,
    pixel_size: float = 0.5,
    with_index: bool = True,
    stale_index: bool = False,
    with_experiment: bool = True,
    with_override: bool = False,
    override_pixel_size: float | None = None,
    with_atera_attrs: bool = True,
    with_schema: bool = True,
    with_native_json: bool = True,
    row_group_size: int = 16,
    seed: int = 7513,
    name: str = "latest_spatialdata.zarr",
) -> Path:
    """Write a synthetic Atera store under ``root`` and return its path.

    ``pixel_size`` (um/px) sets the raster grid, the shapes/points ``global``
    scale (1 / pixel_size), ``atera/experiment.spatial`` and
    ``merxen_atera.experiment.pixel_size``. ``with_override`` adds
    ``atera/experiment_override.spatial`` holding ``override_pixel_size``
    (default ``pixel_size``).
    """
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel
    from spatialdata.transformations import Identity, Scale

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = root / name
    if store.exists():
        raise FileExistsError(store)
    ps = float(pixel_size)
    um_to_px = 1.0 / ps
    transcripts = synthetic_transcripts(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import dask.dataframe as dd

        image = Image2DModel.parse(
            _image(ps),
            dims=("c", "y", "x"),
            c_coords=list(CHANNELS),
            transformations={"global": Identity()},
            scale_factors=[2],
        )
        cell_raster = _rasterize(CELLS, ps)
        nucleus_raster = _rasterize([(n[0], *n[2:]) for n in NUCLEI], ps)
        labels = {
            key: Labels2DModel.parse(array, dims=("y", "x"), transformations={"global": Identity()})
            for key, array in (
                ("cell_labels", cell_raster),
                ("nucleus_labels", nucleus_raster),
                ("atera_cell_boundaries_labels", cell_raster),
                ("atera_nucleus_labels", nucleus_raster),
            )
        }
        shapes_scale = Scale([um_to_px, um_to_px], axes=("x", "y"))
        shapes = {
            "atera_cell_boundaries": ShapesModel.parse(
                gpd.GeoDataFrame(_cell_frame(), geometry="geometry"),
                transformations={"global": shapes_scale},
            ),
            "atera_nucleus": ShapesModel.parse(
                gpd.GeoDataFrame(_nucleus_frame(), geometry="geometry"),
                transformations={"global": shapes_scale},
            ),
        }
        first = transcripts[transcripts["keep"]].head(4)
        seed_points = pd.DataFrame(
            {
                "x": first["x"].to_numpy(np.float32),
                "y": first["y"].to_numpy(np.float32),
                "z": first["z"].to_numpy(np.float32),
                "feature_name": pd.Categorical(
                    [FEATURE_DICTIONARY[c] for c in first["code"]], categories=list(FEATURE_DICTIONARY)
                ),
                "original_assignment": first["original_assignment"].to_numpy(),
            }
        )
        points = PointsModel.parse(
            dd.from_pandas(seed_points, npartitions=1),
            coordinates={"x": "x", "y": "y", "z": "z"},
            feature_key="feature_name",
            instance_key="original_assignment",
            transformations={"global": Scale([um_to_px, um_to_px, 1.0], axes=("x", "y", "z"))},
        )
        sd.SpatialData(
            images={"morphology_focus": image},
            labels=labels,
            shapes=shapes,
            points={"transcripts": points},
            tables=_tables(transcripts, ps, seed),
        ).write(str(store))

    entries = _write_points_files(store, transcripts, row_group_size)

    raster_shape = _raster_shape(ps)
    update_zarr_attributes(
        store / "labels" / "atera_cell_boundaries_labels",
        {"napari_compare_label_cache": _label_marker(
            "cell_labels", "atera_cell_boundaries", raster_shape,
            id_namespace="source_segmentation_v1", cell_join_key=None)},
    )
    update_zarr_attributes(
        store / "labels" / "atera_nucleus_labels",
        {"napari_compare_label_cache": _label_marker(
            "nucleus_labels", "atera_nucleus", raster_shape,
            id_namespace="original_nucleus_row_v1", cell_join_key="cell_instance_id")},
    )

    root_attrs: dict[str, Any] = {}
    if with_schema:
        root_attrs["merxen_schema"] = merxen_schema_attr()
    if with_atera_attrs:
        root_attrs["merxen_atera"] = {"experiment": experiment_spatial_payload(ps)}
    if root_attrs:
        update_zarr_attributes(store, root_attrs)

    if with_native_json:
        _write_json(
            store / "merxen_native_latest.json",
            {
                "format_version": 1,
                "platform": "ATERA",
                "dataset_name": DATASET_NAME,
                "source_build_fingerprint": "0" * 64,
                "link_mode": "hardlink",
                "readonly_payloads": True,
                "elements_written": [
                    "shapes/atera_cell_boundaries",
                    "shapes/atera_nucleus",
                    "tables/table_original",
                ],
            },
        )
    if with_experiment:
        _write_json(store / "atera" / "experiment.spatial", experiment_spatial_payload(ps))
    if with_override:
        value = ps if override_pixel_size is None else float(override_pixel_size)
        _write_json(store / "atera" / "experiment_override.spatial", experiment_spatial_payload(value))
    if with_index:
        _write_json(store / "merxen_atera_points_index.json", points_index_payload(entries, stale=stale_index))
    return store


# ---------------------------------------------------------------------------
# P7513-like attribute stores (metadata only)
# ---------------------------------------------------------------------------


def _write_group(path: Path, attributes: dict[str, Any] | None = None) -> None:
    _write_json(
        path / "zarr.json",
        {"attributes": dict(attributes or {}), "zarr_format": 3, "node_type": "group"},
    )


def build_p7513_like_attrs_store(root: Path, platform: str, *, with_schema: bool = True) -> Path:
    """A metadata-only store with the root attributes of a paired P7513 store.

    XENIUM: ``merxen_schema`` + ``merxen_alignment_pair_reference`` (counterpart MERSCOPE)
    and ``shapes/xenium_cell_boundaries``. MERSCOPE: ``merxen_schema`` +
    ``merxen_alignment`` (roles fixed XENIUM / moving MERSCOPE), a ``_merxen_alignment``
    directory and ``shapes/merscope_cell_boundaries``.
    """
    platform = str(platform).upper()
    store = Path(root) / f"p7513_like_{platform.lower()}.zarr"
    attrs: dict[str, Any] = {"spatialdata_attrs": {"version": "0.2"}}
    if with_schema:
        attrs["merxen_schema"] = {"schema_version": "1.0.0", "primary_points": "transcripts", "platform": platform}
    if platform == "XENIUM":
        attrs["merxen_alignment_pair_reference"] = {
            "version": 1,
            "pair_id": "SYN1",
            "counterpart_platform": "MERSCOPE",
            "counterpart_native_fingerprint": "1" * 64,
            "transform_fingerprint": "2" * 64,
        }
        shape_key = "xenium_cell_boundaries"
    elif platform == "MERSCOPE":
        attrs["merxen_alignment"] = {
            "version": 2,
            "pair_id": "SYN1",
            "roles": {"fixed": "XENIUM", "moving": "MERSCOPE"},
            "common_coordinate_system": "merxen_xenium",
        }
        (store / "_merxen_alignment" / "SYN1").mkdir(parents=True, exist_ok=True)
        shape_key = "merscope_cell_boundaries"
    else:
        raise ValueError(f"P7513-like stores are XENIUM or MERSCOPE, not {platform}")
    _write_group(store, attrs)
    _write_group(store / "shapes")
    _write_group(store / "shapes" / shape_key, {"encoding-type": "ngff:shapes"})
    return store
