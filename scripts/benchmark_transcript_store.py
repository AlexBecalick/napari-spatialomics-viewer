#!/usr/bin/env python3
"""Synthetic benchmark for the bounded transcript-store builder.

Run from an installed/editable checkout, for example:

    python scripts/benchmark_transcript_store.py --points 1000000
    python scripts/benchmark_transcript_store.py --points 10000000 --render-cap 2000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# This repository also has a legacy scripts/napari_compare_xenium_merscope.py;
# put the package source ahead of that same-named module for direct script runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from napari_compare_xenium_merscope.utils import build_gene_point_groups


def _array_bytes(store) -> int:
    arrays = list(store.group_coords) + list(store.group_colors)
    index = store.cell_transcript_index
    if index is not None:
        arrays.extend((index.coords_yx, index.gene_codes))
    return sum(int(array.nbytes) for array in arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--genes", type=int, default=500)
    parser.add_argument("--cells", type=int, default=100_000)
    parser.add_argument("--render-cap", type=int, default=250_000)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.points, args.genes, args.cells, args.render_cap, args.partitions) <= 0:
        parser.error("all numeric arguments must be positive")

    rng = np.random.default_rng(args.seed)
    gene_names = np.asarray([f"GENE_{index:04d}" for index in range(args.genes)])
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0, 20_000, args.points).astype(np.float32),
            "y": rng.uniform(0, 20_000, args.points).astype(np.float32),
            "gene": gene_names[rng.integers(0, args.genes, args.points)],
            "assignment": rng.integers(1, args.cells + 1, args.points, dtype=np.uint32),
        }
    )
    try:
        import dask.dataframe as dd

        points = dd.from_pandas(frame, npartitions=args.partitions)
    except Exception:
        points = frame

    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    store = build_gene_point_groups(
        points,
        x_col="x",
        y_col="y",
        gene_col="gene",
        assignment_col="assignment",
        max_points=args.render_cap,
        random_state=args.seed,
        build_cell_index=True,
    )
    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    index_points = (
        0 if store.cell_transcript_index is None else len(store.cell_transcript_index.coords_yx)
    )
    print(f"source points:       {store.source_total_points:,}")
    print(f"rendered points:     {store.total_points:,}")
    print(f"indexed points:      {index_points:,}")
    print(f"build seconds:       {elapsed:.2f}")
    print(f"retained array GiB:  {_array_bytes(store) / (1024 ** 3):.3f}")
    print(f"RSS delta GiB:       {(rss_after - rss_before) / (1024 ** 3):.3f}")


if __name__ == "__main__":
    main()
