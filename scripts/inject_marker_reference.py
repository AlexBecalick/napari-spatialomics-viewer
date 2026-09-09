#!/usr/bin/env python3
"""Inject a portable cell-type marker-reference JSON into SpatialData stores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import TableModel

ELEMENT = "cell_type_marker_reference"


def inject(reference_path: Path, zarr_path: Path, *, force: bool = False) -> str:
    raw = json.loads(reference_path.read_text())
    genes = raw.get("genes")
    if not isinstance(genes, dict) or not genes:
        raise ValueError(f"{reference_path} has no non-empty genes mapping")

    element_path = zarr_path / "tables" / ELEMENT
    if element_path.exists() and not force:
        return f"SKIPPED (already present): {zarr_path}"

    names = sorted(genes)
    obs = pd.DataFrame(
        {
            "broad_cell_type": [genes[name]["broad"] for name in names],
            "fine_cell_type": [genes[name].get("fine", "") for name in names],
        },
        index=pd.Index(names, name="gene"),
    )
    table = ad.AnnData(X=np.zeros((len(names), 1), dtype=np.float32), obs=obs)
    table.uns[ELEMENT] = json.dumps(raw)
    table = TableModel.parse(table)

    sdata = sd.read_zarr(str(zarr_path), selection=("tables",))
    if element_path.exists():
        sdata.delete_element_from_disk(ELEMENT)
    sdata.tables[ELEMENT] = table
    sdata.write_element(ELEMENT, overwrite=False)
    try:
        sdata.write_consolidated_metadata()
    except Exception:
        pass
    return f"INJECTED {len(names)} genes: {zarr_path}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("zarr", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for zarr_path in args.zarr:
        print(inject(args.reference, zarr_path, force=args.force))


if __name__ == "__main__":
    main()
