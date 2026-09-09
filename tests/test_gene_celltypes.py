"""Tests for the cell-type marker reference: colour/symbol schemes, the reference
loader/normaliser, and reference-driven point grouping."""
from __future__ import annotations

import colorsys
import json
import types

import numpy as np
import pandas as pd
import anndata as ad
import zarr

from napari_compare_xenium_merscope.utils import (
    COARSE_CELL_TYPE_HUES,
    GENE_MARKER_SYMBOLS,
    UNCLASSIFIED_GROUP_TITLE,
    build_cell_type_gene_visuals,
    build_gene_point_groups,
    gene_panel_fingerprint,
    is_control_gene,
    load_cell_type_marker_reference,
    resolve_cell_type_marker_reference,
)

# A tiny reference spanning several broad types (two neuron subtypes so we can
# check that fine subtypes of one broad cluster in hue) plus a functional group.
REF = {
    "SLC17A7": {"broad": "Neuron", "fine": "L2/3 IT"},
    "RORB": {"broad": "Neuron", "fine": "L4 IT"},
    "GAD1": {"broad": "Neuron", "fine": "Pvalb"},
    "SST": {"broad": "Neuron", "fine": "Sst"},
    "AQP4": {"broad": "Astrocyte", "fine": "Protoplasmic astrocyte"},
    "GJA1": {"broad": "Astrocyte", "fine": "Fibrous astrocyte"},
    "P2RY12": {"broad": "Microglia", "fine": "Homeostatic microglia"},
    "MOG": {"broad": "Oligodendrocyte", "fine": "Myelinating oligodendrocyte"},
    "APP": {"broad": "Alzheimer's disease", "fine": "Amyloid / secretase"},
    "MAPT": {"broad": "Alzheimer's disease", "fine": "Tau"},
}
GENES = sorted(REF) + ["NegControlProbe_00002", "BLANK_0007"]


def _hue(rgba):
    return colorsys.rgb_to_hsv(*rgba[:3])[0]


def test_coarse_scheme_groups_alphabetical_controls_last():
    scheme = build_cell_type_gene_visuals(GENES, REF, kind="coarse")
    titles = [t for t, _ in scheme.groups]
    # Broad groups alphabetical, control group last.
    assert titles[:-1] == sorted(titles[:-1])
    assert titles[-1] == "Control / blank probes"
    assert "Neuron" in titles and "Alzheimer's disease" in titles
    # Every gene (incl. controls) gets a visual.
    assert set(scheme.visuals) == set(GENES)


def test_coarse_genes_share_their_broad_hue():
    scheme = build_cell_type_gene_visuals(GENES, REF, kind="coarse")
    for gene, info in REF.items():
        base = COARSE_CELL_TYPE_HUES.get(info["broad"])
        if base is None:
            continue  # functional group hue lives elsewhere
        assert abs(_hue(scheme.visuals[gene].rgba) - base) < 0.02, gene


def test_symbols_are_stable_between_coarse_and_fine():
    coarse = build_cell_type_gene_visuals(GENES, REF, kind="coarse")
    fine = build_cell_type_gene_visuals(GENES, REF, kind="fine")
    for gene in GENES:
        assert coarse.visuals[gene].symbol == fine.visuals[gene].symbol
    # Within the (largest) Neuron broad group, symbols cycle the marker list.
    neuron_syms = [coarse.visuals[g].symbol for g in ("GAD1", "RORB", "SLC17A7", "SST")]
    assert len(set(neuron_syms)) == len(neuron_syms)  # 4 distinct symbols
    assert all(s in GENE_MARKER_SYMBOLS for s in neuron_syms)


def test_fine_scheme_orders_by_broad_then_fine_and_clusters_hue():
    scheme = build_cell_type_gene_visuals(GENES, REF, kind="fine")
    titles = [t for t, _ in scheme.groups if t != "Control / blank probes"]
    # Ordered by (broad, fine): Alzheimer's before Astrocyte before Neuron, and
    # within Neuron the fine subtypes are alphabetical.
    assert titles == sorted(titles)
    # Two neuron subtypes are closer in hue to each other than to an astrocyte.
    h_l23 = _hue(scheme.visuals["SLC17A7"].rgba)   # Neuron / L2/3 IT
    h_sst = _hue(scheme.visuals["SST"].rgba)        # Neuron / Sst
    h_astro = _hue(scheme.visuals["AQP4"].rgba)     # Astrocyte
    assert abs(h_l23 - h_sst) < abs(h_l23 - h_astro)


def test_no_reference_falls_back_to_rainbow():
    scheme = build_cell_type_gene_visuals(["ZZZ", "AAA", "BBB"], reference=None, kind="coarse")
    assert scheme.groups == [("Genes", ["AAA", "BBB", "ZZZ"])]
    assert set(scheme.visuals) == {"AAA", "BBB", "ZZZ"}


def test_partial_reference_separates_unclassified_genes_from_controls():
    scheme = build_cell_type_gene_visuals(
        ["SLC17A7", "UNKNOWN", "UnassignedCodeword_0001"],
        {"SLC17A7": REF["SLC17A7"]},
        kind="coarse",
    )
    grouped = dict(scheme.groups)
    assert grouped[UNCLASSIFIED_GROUP_TITLE] == ["UNKNOWN"]
    assert grouped["Control / blank probes"] == ["UnassignedCodeword_0001"]


def test_unassigned_codeword_is_control_and_excluded_from_fingerprint():
    assert is_control_gene("UnassignedCodeword_0042")
    assert gene_panel_fingerprint(["A", "UnassignedCodeword_0042"]) == gene_panel_fingerprint(["A"])


def test_normalize_reference_accepts_json_string_and_wrapped():
    # A JSON string (as stored in uns) and a {"genes": ...} wrapper both work.
    scheme = build_cell_type_gene_visuals(["APP", "MAPT"], json.dumps({"genes": REF}), kind="coarse")
    assert scheme.visuals["APP"].symbol  # populated, no crash
    titles = [t for t, _ in scheme.groups]
    assert "Alzheimer's disease" in titles


def test_load_reference_from_fake_table_uns():
    fake_table = types.SimpleNamespace(
        uns={"cell_type_marker_reference": json.dumps({"genes": REF})},
        var=pd.DataFrame(index=list(REF)),
    )
    fake_sdata = types.SimpleNamespace(tables={"table": fake_table})
    ref = load_cell_type_marker_reference(fake_sdata)
    assert ref is not None
    assert ref["SLC17A7"]["broad"] == "Neuron"
    assert ref["SLC17A7"]["fine"] == "L2/3 IT"


def test_load_reference_from_fake_table_var_columns():
    var = pd.DataFrame(
        {
            "broad_cell_type": [REF[g]["broad"] for g in REF],
            "fine_cell_type": [REF[g]["fine"] for g in REF],
        },
        index=list(REF),
    )
    fake_table = types.SimpleNamespace(uns={}, var=var)
    fake_sdata = types.SimpleNamespace(tables={"table": fake_table})
    ref = load_cell_type_marker_reference(fake_sdata)
    assert ref["AQP4"]["broad"] == "Astrocyte"


def test_load_reference_returns_none_without_metadata():
    fake_table = types.SimpleNamespace(uns={}, var=pd.DataFrame(index=["A"]))
    fake_sdata = types.SimpleNamespace(tables={"table": fake_table})
    assert load_cell_type_marker_reference(fake_sdata) is None
    assert load_cell_type_marker_reference(None) is None


def test_load_reference_from_zarr_path_reads_anndata_table_directly(tmp_path):
    path = tmp_path / "spatialdata.zarr"
    marker_path = path / "tables" / "cell_type_marker_reference"
    zarr.open_group(path, mode="w", zarr_format=2).create_group("tables")
    table = ad.AnnData(X=np.zeros((1, 1)), var=pd.DataFrame(index=["SLC17A7"]))
    table.uns["cell_type_marker_reference"] = {"genes": REF}
    table.write_zarr(marker_path)

    ref = load_cell_type_marker_reference(path)

    assert ref is not None
    assert ref["SLC17A7"] == {"broad": "Neuron", "fine": "L2/3 IT"}


def test_resolver_reads_json_sidecar_and_reports_coverage(tmp_path):
    path = tmp_path / "dataset" / "spatialdata.zarr"
    zarr.open_group(path, mode="w", zarr_format=2)
    sidecar = path.parent / "cell_type_marker_reference.json"
    sidecar.write_text(json.dumps({"genes": {"Slc17a7": REF["SLC17A7"]}}))

    result = resolve_cell_type_marker_reference(path, ["Slc17a7", "Blank-1"])

    assert result.reference == {"Slc17a7": REF["SLC17A7"]}
    assert result.matched_gene_count == result.panel_gene_count == 1
    assert result.broad_available and result.fine_available
    assert str(sidecar) in result.source


def test_resolver_uses_only_an_exact_external_panel_fingerprint(tmp_path, monkeypatch):
    catalogue = tmp_path / "catalogue"
    catalogue.mkdir()
    genes = ["SLC17A7", "AQP4"]
    payload = {
        "panel_fingerprint": gene_panel_fingerprint(genes),
        "genes": {name: REF[name] for name in genes},
    }
    (catalogue / "matching.json").write_text(json.dumps(payload))
    monkeypatch.setenv("NAPARI_COMPARE_MARKER_REFERENCE_DIR", str(catalogue))

    matching = resolve_cell_type_marker_reference(None, genes)
    nonmatching = resolve_cell_type_marker_reference(None, [*genes, "MOG"])

    assert matching.reference == payload["genes"]
    assert matching.source.endswith("matching.json")
    assert nonmatching.reference is None


def test_build_store_groups_points_by_cell_type_symbol():
    rows = []
    for gene in ("SLC17A7", "RORB", "GAD1", "SST", "AQP4", "MOG"):
        for i in range(3):
            rows.append((gene, float(i), float(i), False))
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background", reference=REF,
    )
    coarse = build_cell_type_gene_visuals(store.genes, REF, kind="coarse")
    # The store's per-gene marker symbol matches the coarse scheme's assignment
    # (grouping is cell-type driven, not alphabetical-index driven).
    for gene in ("SLC17A7", "RORB", "GAD1", "SST", "AQP4", "MOG"):
        assert store.gene_symbol(gene) == coarse.visuals[gene].symbol


def test_store_recolor_keeps_symbols_changes_colours():
    # Two Neuron subtypes so the fine hue offset differs from the single broad hue.
    rows = [(g, 0.0, 0.0, False) for g in ("SLC17A7", "SST", "AQP4", "MOG")]
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background", reference=REF,
    )
    symbols_before = list(store.group_symbols)
    before = [c.copy() for c in store.group_colors]
    fine = build_cell_type_gene_visuals(store.genes, REF, kind="fine")
    store.recolor(fine.visuals)
    assert store.group_symbols == symbols_before  # grouping unchanged
    changed = any(not np.allclose(a, b) for a, b in zip(before, store.group_colors))
    assert changed  # colours updated in place
