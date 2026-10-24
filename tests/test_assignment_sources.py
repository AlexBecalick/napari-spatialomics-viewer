"""Registered pciSeq sources keep native masks and all-cell annotations separate."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from napari_compare_xenium_merscope.utils import (
    analysis_source_metadata,
    assignment_source_points,
    build_assignment_color_schemes,
    cell_id_to_int,
    cell_ids_as_int64,
    cell_type_levels,
    cell_type_shape_key_for_source,
    cell_type_table_key_for_source,
    clustering_table_key_for_segmentation,
    load_cell_type_assignments,
    normalize_cell_key,
)


def _write_table(store, key, obs, shape, umap=None):
    table = ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs)
    table.uns["spatialdata_attrs"] = {
        "region": [shape],
        "region_key": "region",
        "instance_key": "instance_id",
    }
    table.uns["merxen_pciseq_viewer"] = {"analysis_id": "a" * 64}
    if umap is not None:
        table.obsm["X_umap"] = np.asarray(umap, dtype=np.float32)
    table.write_zarr(str(store / "tables" / key))


@pytest.fixture
def pciseq_store(tmp_path):
    store = tmp_path / "store.zarr"
    root = zarr.open_group(str(store), mode="w", zarr_format=2)
    for group in ("shapes", "points", "tables"):
        root.create_group(group)
    root["shapes"].create_group("MOSAIK_proseg_hybrid")
    root["points"].create_group("transcripts")
    root["points"].create_group("transcripts_pciseq_proseg_hybrid")
    root["points"]["transcripts_pciseq_proseg_hybrid"].attrs["merxen_pciseq_viewer"] = {
        "analysis_id": "a" * 64,
    }
    native = {
        "shape": "MOSAIK_proseg_hybrid",
        "points": "transcripts",
        "table": "table_MOSAIK_proseg_hybrid",
        "instance_key": "instance_id",
    }
    entry = {
        "method": "pciseq",
        "label": "pciSeq · ProSeg hybrid",
        "segmentation": "proseg_hybrid",
        "shape": native["shape"],
        "points": "transcripts_pciseq_proseg_hybrid",
        "source_points": "transcripts",
        "assignment_column": "assignment",
        "background_column": "background",
        "status_column": "pciseq_assignment_status",
        "instance_key": "instance_id",
        "table": "table_MOSAIK_proseg_hybrid_pciseq_cells",
        "embedding_table": "table_MOSAIK_proseg_hybrid_pciseq",
        "analysis_id": "a" * 64,
    }
    root.attrs["merxen_schema"] = {
        "primary_points": "transcripts",
        "segmentations": {"proseg_hybrid": native},
        "analysis_sources": {"pciseq_proseg_hybrid": entry},
    }
    ids = np.array([2**63 + 7, 2**63 + 8, 2**64 - 1], dtype=np.uint64)
    obs = pd.DataFrame(
        {
            "instance_id": ids,
            "pciseq_cell_type": pd.Categorical(["Astro_52", "Zero", None]),
            "pciseq_probability": [0.9, 0.8, np.nan],
            "pciseq_status": pd.Categorical(
                ["assigned", "zero_expression", "excluded_overlap"]
            ),
            # A pciSeq source must never display these copied clustering labels.
            "broad_class": ["Legacy neuron"] * 3,
            "ct_final_name": ["Legacy Allen"] * 3,
        },
        index=["full_a", "full_b", "full_c"],
    )
    _write_table(store, entry["table"], obs, native["shape"])
    _write_table(
        store,
        entry["embedding_table"],
        obs.iloc[[1, 2]],
        native["shape"],
        [[1, 2], [3, 4]],
    )
    legacy = pd.DataFrame(
        {"instance_id": [5], "broad_class": ["Legacy neuron"]}, index=["old"]
    )
    _write_table(
        store, "table_MOSAIK_proseg_hybrid_clustering_squidpy", legacy, native["shape"]
    )
    return store, entry, ids


def test_pciseq_annotations_and_embedding_resolve_independently(pciseq_store):
    store, entry, ids = pciseq_store
    key = "pciseq_proseg_hybrid"
    assert analysis_source_metadata(store, key) == entry
    assert assignment_source_points(store, key) == entry
    assert assignment_source_points(store, "proseg_hybrid") is None
    assert cell_type_shape_key_for_source(store, key) == "MOSAIK_proseg_hybrid"
    assert cell_type_table_key_for_source(store, key) == entry["table"]
    assert (
        cell_type_table_key_for_source(store, key, embedding=True)
        == entry["embedding_table"]
    )
    assert clustering_table_key_for_segmentation(store, "proseg_hybrid") == (
        "table_MOSAIK_proseg_hybrid_clustering_squidpy"
    )
    assert cell_type_levels(store, key) == {"pciseq": "pciSeq · cell type"}
    labels = load_cell_type_assignments(store, key)
    assert labels is not None
    assert labels.cell_ids.tolist() == ids[:2].tolist()
    assert labels.cell_ids.dtype == np.uint64
    assert labels.labels_for("pciseq").tolist() == ["Astro_52", "Zero"]
    assert labels.labels_for("broad").tolist() == [None, None]
    assert labels.probabilities.tolist() == [0.9, 0.8]
    assert labels.statuses.tolist() == ["assigned", "zero_expression"]
    assert labels.shape_key == "MOSAIK_proseg_hybrid"
    assert labels.analysis_id == entry["analysis_id"]
    assert set(build_assignment_color_schemes([labels])) == {"pciseq"}


@pytest.mark.parametrize(
    "branch,shape",
    [
        ("proseg_hybrid", "MOSAIK_proseg_hybrid"),
        ("proseg", "MOSAIK_proseg"),
        ("cellpose", "MOSAIK_cellpose"),
        ("original", "xenium_cell_boundaries"),
    ],
)
def test_each_pciseq_branch_uses_registered_native_shape(pciseq_store, branch, shape):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    if shape not in root["shapes"]:
        root["shapes"].create_group(shape)
    schema = root.attrs["merxen_schema"]
    schema["segmentations"][branch] = {"shape": shape, "points": "transcripts"}
    entry = dict(entry, shape=shape, segmentation=branch)
    schema["analysis_sources"][f"pciseq_{branch}"] = entry
    root.attrs["merxen_schema"] = schema
    obs = pd.DataFrame({"instance_id": [1], "pciseq_cell_type": ["A"]}, index=["a"])
    _write_table(store, entry["table"], obs, shape)
    _write_table(store, entry["embedding_table"], obs, shape, [[0, 1]])
    assert cell_type_shape_key_for_source(store, f"pciseq_{branch}") == shape


@pytest.mark.parametrize(
    "change",
    [
        {"method": "clustering"},
        {"segmentation": "proseg"},
        {"shape": "wrong_mask"},
        {"points": "transcripts"},
        {"source_points": "other_points"},
        {"table": "missing_table"},
        {"analysis_id": "invalid"},
    ],
)
def test_invalid_or_cross_branch_registry_is_not_offered(pciseq_store, change):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    schema = root.attrs["merxen_schema"]
    schema["analysis_sources"]["pciseq_proseg_hybrid"] = dict(entry, **change)
    root.attrs["merxen_schema"] = schema
    assert analysis_source_metadata(store, "pciseq_proseg_hybrid") is None
    assert load_cell_type_assignments(store, "pciseq_proseg_hybrid") is None


def test_wrong_table_shape_link_is_not_joined(pciseq_store):
    store, entry, _ = pciseq_store
    obs = pd.DataFrame({"instance_id": [1], "pciseq_cell_type": ["A"]}, index=["a"])
    _write_table(store, entry["table"], obs, "MOSAIK_proseg")
    assert analysis_source_metadata(store, "pciseq_proseg_hybrid") is None


def test_missing_embedding_does_not_hide_all_cell_labels(pciseq_store):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    del root["tables"][entry["embedding_table"]]
    assert cell_type_levels(store, "pciseq_proseg_hybrid") == {
        "pciseq": "pciSeq · cell type"
    }
    assert (
        cell_type_table_key_for_source(store, "pciseq_proseg_hybrid", embedding=True)
        is None
    )


@pytest.mark.parametrize("element", ["points", "table", "embedding_table"])
@pytest.mark.parametrize("marker", [None, {"analysis_id": "b" * 64}])
def test_uncommitted_or_missing_generation_markers_fail_closed(
    pciseq_store, element, marker
):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    if element == "points":
        attrs = root["points"][entry["points"]].attrs
        if marker is None:
            del attrs["merxen_pciseq_viewer"]
        else:
            attrs["merxen_pciseq_viewer"] = marker
    else:
        from anndata.io import write_elem

        uns = root["tables"][entry[element]]["uns"]
        if marker is None:
            del uns["merxen_pciseq_viewer"]
        else:
            write_elem(uns, "merxen_pciseq_viewer", marker)
    assert analysis_source_metadata(store, "pciseq_proseg_hybrid") is None
    assert cell_type_levels(store, "pciseq_proseg_hybrid") == {}


@pytest.mark.parametrize("variant", ["native", "registered", "legacy"])
def test_qc_embedding_keeps_its_valid_native_or_aligned_region(pciseq_store, variant):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    native = entry["shape"]
    shape = {
        "native": native,
        "registered": "registered_destination",
        "legacy": f"{native}_aligned_nonrigid",
    }[variant]
    schema = root.attrs["merxen_schema"]
    if shape not in root["shapes"]:
        root["shapes"].create_group(shape)
    schema["analysis_sources"]["pciseq_proseg_hybrid"]["embedding_shape"] = shape
    if variant == "registered":
        schema["segmentations"]["proseg_hybrid_aligned"] = {
            "shape": shape,
            "coordinate_variant_of": "proseg_hybrid",
        }
    root.attrs["merxen_schema"] = schema
    obs = pd.DataFrame({"instance_id": [1], "pciseq_cell_type": ["A"]}, index=["a"])
    _write_table(store, entry["embedding_table"], obs, shape, [[4, 7]])
    assert (
        analysis_source_metadata(store, "pciseq_proseg_hybrid")["embedding_shape"]
        == shape
    )
    assert cell_type_shape_key_for_source(store, "pciseq_proseg_hybrid") == native


def test_embedding_cannot_use_another_branch_even_with_matching_cell_ids(pciseq_store):
    store, entry, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    wrong = "MOSAIK_proseg_aligned_nonrigid"
    root["shapes"].create_group(wrong)
    schema = root.attrs["merxen_schema"]
    schema["analysis_sources"]["pciseq_proseg_hybrid"]["embedding_shape"] = wrong
    schema["segmentations"]["proseg_aligned"] = {
        "shape": wrong,
        "coordinate_variant_of": "proseg",
    }
    root.attrs["merxen_schema"] = schema
    obs = pd.DataFrame({"instance_id": [1], "pciseq_cell_type": ["A"]}, index=["a"])
    _write_table(store, entry["embedding_table"], obs, wrong, [[4, 7]])
    assert analysis_source_metadata(store, "pciseq_proseg_hybrid") is None


def test_pciseq_custom_class_names_are_not_legacy_missing_label_tokens(pciseq_store):
    store, entry, _ = pciseq_store
    obs = pd.DataFrame(
        {
            "instance_id": np.arange(1, 6, dtype=np.uint64),
            "pciseq_cell_type": pd.Categorical(["NA", "None", "nan", "", None]),
            "pciseq_probability": [0.9, 0.8, 0.7, np.nan, np.nan],
            "pciseq_status": ["assigned"] * 5,
        },
        index=[f"c{i}" for i in range(5)],
    )
    _write_table(store, entry["table"], obs, entry["shape"])
    labels = load_cell_type_assignments(store, "pciseq_proseg_hybrid")
    assert labels.labels_for("pciseq").tolist() == ["NA", "None", "nan"]
    assert build_assignment_color_schemes([labels])["pciseq"].order == [
        "NA",
        "None",
        "nan",
    ]


@pytest.mark.parametrize("read_kind", ["annotations", "embedding"])
def test_refresh_during_metadata_array_read_does_not_return_mixed_generation(
    pciseq_store,
    monkeypatch,
    read_kind,
):
    import anndata.io

    from napari_compare_xenium_merscope.cell_umap import load_cell_umap

    store, entry, _ = pciseq_store
    original = anndata.io.read_elem
    touched = []
    target = "pciseq_cell_type" if read_kind == "annotations" else "X_umap"

    def read(element, *args, **kwargs):
        result = original(element, *args, **kwargs)
        if element.path.endswith(f"/{target}"):
            root = zarr.open_group(str(store), mode="a", use_consolidated=False)
            root["points"][entry["points"]].attrs["merxen_pciseq_viewer"] = {
                "analysis_id": "b" * 64,
            }
            touched.append(target)
        return result

    monkeypatch.setattr(anndata.io, "read_elem", read)
    loader = (
        load_cell_type_assignments if read_kind == "annotations" else load_cell_umap
    )
    assert loader(store, "pciseq_proseg_hybrid") is None
    assert touched == [target]


def test_unregistered_pciseq_tables_do_not_replace_clustering(pciseq_store):
    store, _, _ = pciseq_store
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    schema = root.attrs["merxen_schema"]
    schema.pop("analysis_sources")
    root.attrs["merxen_schema"] = schema
    assert cell_type_levels(store, "pciseq_proseg_hybrid") == {}
    assert cell_type_levels(store, "proseg_hybrid") == {
        "broad": "Broad cell type",
        "fine": "Fine cell type",
    }


def test_integer_normalization_preserves_adjacent_full_range_uint64():
    large = 2**64 - 1
    assert normalize_cell_key(large) == str(large)
    assert normalize_cell_key(np.uint64(large - 1)) == str(large - 1)
    assert normalize_cell_key(f"{large}.0") == str(large)
    assert normalize_cell_key("1.8446744073709551615e19") == str(large)
    assert normalize_cell_key(float(large)) == ""
    assert normalize_cell_key(pd.NA) == ""
    assert cell_id_to_int(str(large)) == large
    assert cell_id_to_int(float(large)) is None
    ids, valid = cell_ids_as_int64([str(large - 1), str(large), "bad", None])
    assert ids.dtype == np.uint64
    assert ids[valid].tolist() == [large - 1, large]
    ids, valid = cell_ids_as_int64([-1, large, large + 1])
    assert ids.dtype == object
    assert valid.tolist() == [True, True, False]
    assert ids[:2].tolist() == [-1, large]


def test_pciseq_umap_joins_qc_rows_to_full_annotations_by_exact_id(pciseq_store):
    from napari_compare_xenium_merscope.cell_umap import (
        cell_umap_available,
        labels_for_cell_ids,
        load_cell_umap,
        values_for_cell_ids,
    )

    store, entry, ids = pciseq_store
    source = "pciseq_proseg_hybrid"
    assert cell_umap_available(store, source)
    embedding = load_cell_umap(store, source)
    annotations = load_cell_type_assignments(store, source)
    assert embedding is not None and annotations is not None
    assert embedding.table_key == entry["embedding_table"]
    assert embedding.cell_ids.tolist() == ids[1:].tolist()
    assert embedding.cell_ids.dtype == np.uint64
    assert embedding.coords.tolist() == [[1, 2], [3, 4]]
    assert embedding.row_for_id(str(ids[1])) == 0
    assert embedding.row_for_id(str(ids[2])) == 1
    assert embedding.row_for_id(-1) == -1
    assert embedding.row_for_id(str(ids[0])) == -1
    labels = labels_for_cell_ids(
        annotations.cell_ids,
        annotations.labels_for("pciseq"),
        embedding.cell_ids,
    )
    assert labels.tolist() == ["Zero", None]
    values = values_for_cell_ids(
        annotations.cell_ids,
        annotations.probabilities,
        embedding.cell_ids,
    )
    assert values[0] == 0.8 and np.isnan(values[1])


def test_exact_id_joins_do_not_alias_signed_minus_one_and_uint64_maximum():
    from napari_compare_xenium_merscope.cell_umap import (
        labels_for_cell_ids,
        values_for_cell_ids,
    )

    largest = 2**64 - 1
    assert labels_for_cell_ids(
        [largest - 1, largest], ["A", "B"], [-1, largest]
    ).tolist() == [
        None,
        "B",
    ]
    assert labels_for_cell_ids(
        [-1, largest], ["negative", "largest"], [largest, -1]
    ).tolist() == [
        "largest",
        "negative",
    ]
    values = values_for_cell_ids([largest - 1, largest], [1, 2], [-1, str(largest - 1)])
    assert np.isnan(values[0]) and values[1] == 1


def test_umap_panel_keeps_full_uint64_cell_ids():
    from qtpy.QtWidgets import QApplication

    from napari_compare_xenium_merscope.cell_umap import UmapPanel

    app = QApplication.instance() or QApplication([])
    panel = UmapPanel()
    try:
        ids = np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64)
        panel.set_embedding(ids, np.array([[0, 1], [2, 3]], dtype=np.float32))
        assert panel._cell_ids.dtype == np.uint64
        assert panel.describe_row(0) == f"Cell {2**64 - 2}"
        assert panel.describe_row(1) == f"Cell {2**64 - 1}"
    finally:
        panel.close()
        app.processEvents()
