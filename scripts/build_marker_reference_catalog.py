#!/usr/bin/env python3
"""Build portable gene-panel cell-type references for the viewer and MerXen.

Existing hand-curated references are normalized and copied into the catalogue.
For a clustered mouse-brain store, every panel gene is assigned to the Allen
whole-mouse-brain supercluster in which its mean normalized expression is
highest. Canonical lineage markers are then corrected with a small, reviewable
curation layer. The source table, winning/runner-up expression, enrichment ratio,
and any curated override are recorded per gene so the result remains auditable.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from napari_compare_xenium_merscope.utils import gene_panel_fingerprint

SCHEMA = "napari-compare cell_type_marker_reference/2"
DEFAULT_TABLE = "table_MOSAIK_proseg_clustering_squidpy"

# Broad-lineage anchors are deliberately conservative: only genes with a strong,
# well-established CNS lineage association are included. This prevents a noisy or
# anatomically restricted dataset from assigning markers such as Olig1/Pdgfra to
# the neuronal class with the largest raw group mean. Fine labels for overridden
# genes use the corresponding Allen whole-mouse-brain class rather than retaining
# an incompatible neuronal supercluster.
CANONICAL_MOUSE_BROAD_MARKERS: dict[str, tuple[str, ...]] = {
    "Astrocyte": (
        "Aldh1l1", "Aldoc", "Aqp4", "Fabp7", "Gfap", "Glul", "S100b",
        "Slc1a2", "Slc1a3", "Sox9",
    ),
    "Oligodendrocyte": (
        "Bcas1", "Cnp", "Cspg4", "Enpp6", "Ermn", "Fa2h", "Gal3st1",
        "Gpr17", "Mag", "Mal", "Mbp", "Mobp", "Mog", "Olig1", "Olig2",
        "Opalin", "Pdgfra", "Plp1", "Sox10", "Tcf7l2", "Ugt8a",
    ),
    "Vascular": (
        "Abcc9", "Acta2", "Adgrl4", "Bmx", "Cldn5", "Col1a1", "Col1a2",
        "Col3a1", "Col4a1", "Col4a2", "Dcn", "Dpt", "Egfl7", "Emcn",
        "Eng", "Esam", "Flt1", "Kcnj8", "Kdr", "Lum", "Mgp", "Notch3",
        "Pdgfrb", "Pecam1", "Pi16", "Ptprb", "Ramp2", "Rgs5", "Tagln",
        "Tek", "Tie1", "Vwf",
    ),
    "Microglia": (
        "Aif1", "C1qa", "C1qb", "C1qc", "Cd68", "Csf1r", "Ctss", "Cx3cr1",
        "Fcrls", "Gpr34", "Hexb", "Itgam", "Laptm5", "Lilrb4", "Lpl", "Lyz2",
        "P2ry12", "Ptprc", "Sall1", "Tmem119", "Trem2", "Tyrobp",
    ),
    "Neuron": (
        "Elavl3", "Elavl4", "Gad1", "Gad2", "Map2", "Rbfox3", "Slc17a6",
        "Slc17a7", "Slc17a8", "Snap25", "Stmn2", "Syt1", "Tubb3",
    ),
}

CANONICAL_MOUSE_FINE_LABELS = {
    "Astrocyte": "30 Astro-Epen",
    "Oligodendrocyte": "31 OPC-Oligo",
    "Vascular": "33 Vascular",
    "Microglia": "34 Immune",
}


def _canonical_mouse_marker_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for broad, genes in CANONICAL_MOUSE_BROAD_MARKERS.items():
        for gene in genes:
            previous = lookup.setdefault(gene.casefold(), broad)
            if previous != broad:
                raise ValueError(f"Canonical marker {gene!r} is assigned twice")
    return lookup


def _viewer_broad_label(value: str) -> str:
    return {
        "Neurons": "Neuron",
        "Astrocytes/Ependymal": "Astrocyte",
        "Oligodendrocyte lineage": "Oligodendrocyte",
        "Vascular cells": "Vascular",
        "Microglia": "Microglia",
        "OEC": "Olfactory ensheathing cell",
    }.get(str(value), str(value))


def derive_mouse_brain_reference(zarr_path: Path, table_key: str = DEFAULT_TABLE) -> dict:
    table_path = zarr_path / "tables" / table_key
    table = ad.read_zarr(table_path)
    required = {"broad_class", "broad_atlas_label"}
    missing = required - set(table.obs.columns)
    if missing:
        raise ValueError(f"{table_path} lacks required obs columns: {sorted(missing)}")

    broad = table.obs["broad_class"].astype(str).to_numpy()
    fine = table.obs["broad_atlas_label"].astype(str).to_numpy()
    pairs = sorted(set(zip(broad.tolist(), fine.tolist())))
    means = np.zeros((len(pairs), table.n_vars), dtype=np.float64)
    cell_counts: list[int] = []
    for index, pair in enumerate(pairs):
        mask = (broad == pair[0]) & (fine == pair[1])
        count = int(mask.sum())
        cell_counts.append(count)
        block = table.X[mask]
        means[index] = np.asarray(block.mean(axis=0)).ravel()

    canonical = _canonical_mouse_marker_lookup()
    genes: dict[str, dict] = {}
    for gene_index, gene in enumerate(map(str, table.var_names)):
        values = means[:, gene_index]
        order = np.argsort(values)[::-1]
        best_index = int(order[0])
        runner_index = int(order[1]) if len(order) > 1 else best_index
        best_broad, best_fine = pairs[best_index]
        best_value = float(values[best_index])
        runner_value = float(values[runner_index])
        ratio = (best_value + 1e-9) / (runner_value + 1e-9)
        genes[gene] = {
            "broad": _viewer_broad_label(best_broad),
            "fine": best_fine,
            "evidence": {
                "method": "highest mean normalized expression by Allen-atlas-labelled cell class",
                "source_table": table_key,
                "winning_mean_expression": round(best_value, 6),
                "runner_up_mean_expression": round(runner_value, 6),
                "winning_to_runner_up_ratio": round(ratio, 6),
                "winning_group_cells": cell_counts[best_index],
            },
        }
        curated_broad = canonical.get(gene.casefold())
        if curated_broad is not None and curated_broad != genes[gene]["broad"]:
            genes[gene]["evidence"]["data_derived_assignment"] = {
                "broad": genes[gene]["broad"],
                "fine": genes[gene]["fine"],
            }
            genes[gene]["evidence"]["curation"] = (
                "canonical CNS broad-lineage marker override"
            )
            genes[gene]["broad"] = curated_broad
            genes[gene]["fine"] = CANONICAL_MOUSE_FINE_LABELS.get(
                curated_broad, genes[gene]["fine"]
            )

    return {
        "schema": SCHEMA,
        "created": date.today().isoformat(),
        "derivation": {
            "kind": "data-derived mouse-brain association with curated canonical-marker overrides",
            "source_dataset": zarr_path.parent.name,
            "source_store": zarr_path.name,
            "table": table_key,
            "fine_level": "Allen whole-mouse-brain supercluster",
            "selection": "maximum group mean of the normalized clustering matrix",
            "curation": "conservative canonical CNS broad-lineage marker overrides",
        },
        "genes": genes,
    }


def load_existing_reference(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw.get("genes"), dict):
        raise ValueError(f"{path} does not contain a genes mapping")
    raw = dict(raw)
    raw["schema"] = SCHEMA
    raw["created"] = date.today().isoformat()
    raw["derivation"] = {
        "kind": "existing hand-curated reference",
        "source_dataset": path.parent.name,
        "source_file": path.name,
    }
    return raw


def write_reference(output_dir: Path, panel_id: str, reference: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = dict(reference)
    genes = reference["genes"]
    reference["panel_id"] = panel_id
    reference["panel_fingerprint"] = gene_panel_fingerprint(genes)
    reference["panel_gene_count"] = len(genes)
    reference["broad_types"] = sorted({str(info["broad"]) for info in genes.values()})
    reference["fine_types"] = sorted({str(info.get("fine", "")) for info in genes.values() if info.get("fine")})
    reference["genes"] = {gene: genes[gene] for gene in sorted(genes)}

    json_path = output_dir / f"{panel_id}.json"
    csv_path = output_dir / f"{panel_id}.csv"
    json_path.write_text(json.dumps(reference, indent=2) + "\n")
    rows = [
        {
            "gene": gene,
            "broad_cell_type": info["broad"],
            "fine_cell_type": info.get("fine", ""),
            "evidence": json.dumps(info.get("evidence", {}), separators=(",", ":")),
        }
        for gene, info in reference["genes"].items()
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ag7-zarr", type=Path)
    parser.add_argument("--vizgen-815-zarr", type=Path)
    parser.add_argument("--p-merscope-reference", type=Path)
    parser.add_argument("--p-xenium-reference", type=Path)
    parser.add_argument("--nmv-reference", type=Path)
    args = parser.parse_args()

    generated: list[tuple[str, dict]] = []
    vizgen_reference = (
        derive_mouse_brain_reference(args.vizgen_815_zarr)
        if args.vizgen_815_zarr is not None
        else None
    )
    if vizgen_reference is not None:
        generated.append(("vizgen_mouse_brain_815", vizgen_reference))
    if args.ag7_zarr is not None:
        ag7_reference = derive_mouse_brain_reference(args.ag7_zarr)
        # The 815-gene whole-mouse-brain dataset has all major glial lineages and
        # is the stronger reference for shared genes. Keep ag7's own enrichment
        # result as supporting evidence, but harmonize the displayed assignment.
        if vizgen_reference is not None:
            for gene in set(ag7_reference["genes"]) & set(vizgen_reference["genes"]):
                ag7_info = ag7_reference["genes"][gene]
                vizgen_info = vizgen_reference["genes"][gene]
                ag7_info["evidence"]["ag7_data_derived_assignment"] = {
                    "broad": ag7_info["broad"],
                    "fine": ag7_info["fine"],
                }
                ag7_info["evidence"]["harmonized_from"] = "vizgen_mouse_brain_815"
                ag7_info["broad"] = vizgen_info["broad"]
                ag7_info["fine"] = vizgen_info["fine"]
        generated.append(("ag7_mouse_500", ag7_reference))

    for panel_id, source in (
        ("p_series_merscope_300", args.p_merscope_reference),
        ("p_series_xenium_300", args.p_xenium_reference),
        ("nmv2p35_18_human_496", args.nmv_reference),
    ):
        if source is not None:
            generated.append((panel_id, load_existing_reference(source)))

    for panel_id, reference in generated:
        json_path, csv_path = write_reference(args.output_dir, panel_id, reference)
        print(f"{panel_id}: {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
