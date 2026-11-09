"""Platform registry, store detection, ATERA pixel size and the synthetic Atera fixture."""

from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import atera_stores
from napari_compare_xenium_merscope import platforms as P
from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import make_layer_name, resolve_dataset_mask_affine

PS = 0.5


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return atera_stores.build_atera_latest_store(tmp_path_factory.mktemp("atera"), pixel_size=PS)


def _copy_store(src: Path, dst_root: Path) -> Path:
    dst = dst_root / "copy" / src.name
    shutil.copytree(src, dst)
    return dst


def _drop_root_attr(store: Path, key: str) -> None:
    meta = json.loads((store / "zarr.json").read_text())
    meta["attributes"].pop(key, None)
    (store / "zarr.json").write_text(json.dumps(meta))


def _empty_group(path: Path, attributes: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"attributes": attributes or {}, "zarr_format": 3, "node_type": "group"})
    )
    return path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_order_and_paired_subset():
    assert P.supported_platforms() == ("MERSCOPE", "XENIUM", "ATERA")
    assert P.standalone_platforms() == P.supported_platforms()
    assert P.PAIRED_PLATFORMS == ("MERSCOPE", "XENIUM")
    assert set(P.PAIRED_PLATFORMS) <= set(P.supported_platforms())
    assert [spec.name for spec in P.PLATFORM_SPECS] == list(P.supported_platforms())


def test_lookup_is_case_insensitive_and_unknown_names_raise():
    assert P.platform_spec("atera").name == "ATERA"
    assert P.platform_spec(" Xenium ").name == "XENIUM"
    assert P.find_platform_spec("merscope").name == "MERSCOPE"
    assert P.find_platform_spec("VISIUM") is None
    assert P.find_platform_spec(None) is None
    with pytest.raises(P.UnknownPlatformError):
        P.platform_spec("VISIUM")
    assert issubclass(P.UnknownPlatformError, ValueError)


def test_spec_values():
    merscope, xenium, atera = (P.platform_spec(n) for n in ("MERSCOPE", "XENIUM", "ATERA"))
    assert merscope.pixel_transform == "merscope_affine"
    assert xenium.pixel_transform == atera.pixel_transform == "spec_file"
    assert (merscope.fallback_um_per_px, xenium.fallback_um_per_px, atera.fallback_um_per_px) == (
        0.108, 0.2125, 0.21249222,
    )
    assert atera.original_cell_shape_key == "atera_cell_boundaries"
    assert atera.original_nucleus_shape_key == "atera_nucleus"
    assert atera.nucleus_cell_join_key == "cell_instance_id"
    assert atera.paired_role is None and not atera.is_paired
    assert atera.assignment_instance_key_fallback
    assert not merscope.assignment_instance_key_fallback and not xenium.assignment_instance_key_fallback
    assert not (atera.supports_pciseq or atera.supports_cortical_depth or atera.supports_object_annotations)
    assert atera.lazy_transcripts and atera.gene_panel_mode == "selection"
    for spec in (merscope, xenium):
        assert spec.supports_pciseq and spec.supports_cortical_depth and spec.supports_object_annotations
        assert not spec.lazy_transcripts and spec.gene_panel_mode == "full"
    assert dict(atera.vendor_label_aliases) == {
        "atera_cell_boundaries_labels": "cell_labels",
        "atera_nucleus_labels": "nucleus_labels",
    }


# ---------------------------------------------------------------------------
# detect_store_platform
# ---------------------------------------------------------------------------


def test_detect_fixture_store_is_authoritative_atera(atera_store):
    found = P.detect_store_platform(atera_store)
    assert found.platform == "ATERA"
    assert found.authoritative
    assert any("merxen_schema" in text for text in found.evidence)


def test_detect_native_latest_json_alone_is_authoritative(tmp_path):
    store = _empty_group(tmp_path / "s.zarr")
    (store / "merxen_native_latest.json").write_text(json.dumps({"platform": "atera"}))
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == ("ATERA", True)


def test_detect_experiment_spatial_alone_is_heuristic(tmp_path):
    store = _empty_group(tmp_path / "s.zarr")
    (store / "atera").mkdir()
    (store / "atera" / "experiment.spatial").write_text(json.dumps({"pixel_size": 0.3}))
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == ("ATERA", False)


def test_detect_root_attr_and_shapes_are_heuristic(tmp_path):
    store = _empty_group(tmp_path / "a.zarr", {"merxen_atera": {"experiment": {}}})
    assert P.detect_store_platform(store) == P.StoreDetection(
        "ATERA", False, ("heuristic ATERA: root attribute merxen_atera",)
    )
    store = _empty_group(tmp_path / "x.zarr")
    _empty_group(store / "shapes" / "xenium_cell_boundaries")
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == ("XENIUM", False)


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_detect_p7513_like_attrs_ignore_alignment_attrs(tmp_path, platform):
    store = atera_stores.build_p7513_like_attrs_store(tmp_path, platform)
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == (platform, True)
    # Without merxen_schema, the alignment attrs alone are not evidence; the
    # original shapes element is (heuristically).
    bare = atera_stores.build_p7513_like_attrs_store(tmp_path / "bare", platform, with_schema=False)
    found = P.detect_store_platform(bare)
    assert (found.platform, found.authoritative) == (platform, False)
    assert not any("alignment" in text for text in found.evidence)


def test_detect_ignores_parent_dir_sidecars(tmp_path):
    (tmp_path / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.2125}))
    store = _empty_group(tmp_path / "s.zarr")
    found = P.detect_store_platform(store)
    assert found.platform is None and not found.authoritative


def test_detect_empty_or_missing_store(tmp_path):
    empty = tmp_path / "empty.zarr"
    empty.mkdir()
    assert P.detect_store_platform(empty).platform is None
    assert P.detect_store_platform(tmp_path / "missing.zarr").platform is None


def test_detect_conflicting_evidence_merxen_schema_wins(tmp_path):
    store = _empty_group(
        tmp_path / "s.zarr", {"merxen_schema": {"platform": "XENIUM"}, "merxen_atera": {}}
    )
    (store / "merxen_native_latest.json").write_text(json.dumps({"platform": "ATERA"}))
    (store / "atera").mkdir()
    (store / "atera" / "experiment.spatial").write_text("{}")
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == ("XENIUM", True)
    assert any("conflict" in text for text in found.evidence)


def test_detect_unknown_schema_platform_falls_through(tmp_path):
    store = _empty_group(tmp_path / "s.zarr", {"merxen_schema": {"platform": "VISIUM"}})
    (store / "merxen_native_latest.json").write_text(json.dumps({"platform": "MERSCOPE"}))
    found = P.detect_store_platform(store)
    assert (found.platform, found.authoritative) == ("MERSCOPE", True)


def test_detect_disagreeing_heuristics_give_none(tmp_path):
    store = _empty_group(tmp_path / "s.zarr")
    (store / "experiment.xenium").write_text("{}")
    (store / "micron_to_mosaic_pixel_transform.csv").write_text("1 0 0\n0 1 0\n0 0 1\n")
    found = P.detect_store_platform(store)
    assert found.platform is None
    assert any("disagree" in text for text in found.evidence)


# ---------------------------------------------------------------------------
# find_spec_path
# ---------------------------------------------------------------------------


def test_find_spec_path_override_beats_experiment_spatial(tmp_path):
    store = _empty_group(tmp_path / "s.zarr")
    (store / "atera").mkdir()
    (store / "atera" / "experiment.spatial").write_text("{}")
    assert P.find_spec_path(store, "ATERA") == store / "atera" / "experiment.spatial"
    (store / "atera" / "experiment_override.spatial").write_text("{}")
    assert P.find_spec_path(store, "atera") == store / "atera" / "experiment_override.spatial"


def test_find_spec_path_parent_lookup(tmp_path):
    store = _empty_group(tmp_path / "s.zarr")
    assert P.find_spec_path(store, "ATERA") is None
    (tmp_path / "experiment.spatial").write_text("{}")
    assert P.find_spec_path(store, "ATERA") == tmp_path / "experiment.spatial"
    (tmp_path / "experiment.xenium").write_text("{}")
    assert P.find_spec_path(store, "XENIUM") == tmp_path / "experiment.xenium"
    (store / "experiment.xenium").write_text("{}")
    assert P.find_spec_path(store, "XENIUM") == store / "experiment.xenium"


# ---------------------------------------------------------------------------
# Pixel size
# ---------------------------------------------------------------------------


def _mpp(affine) -> tuple[float, float]:
    (sx, shx, tx), (shy, sy, ty) = affine
    assert shx == shy == tx == ty == 0.0
    return sx, sy


def test_atera_pixel_size_from_spec_path(atera_store):
    spec = atera_store / "atera" / "experiment.spatial"
    assert _mpp(resolve_dataset_mask_affine("ATERA", spec_path=spec)) == (PS, PS)
    assert _mpp(resolve_dataset_mask_affine("atera", zarr_path=atera_store)) == (PS, PS)


def test_atera_override_wins(tmp_path):
    store = atera_stores.build_atera_latest_store(
        tmp_path, pixel_size=PS, with_override=True, override_pixel_size=0.25, with_index=False
    )
    assert _mpp(resolve_dataset_mask_affine("ATERA", zarr_path=store)) == (0.25, 0.25)


def test_atera_missing_explicit_spec_falls_back_to_store_sidecar(tmp_path, caplog):
    store = atera_stores.build_atera_latest_store(
        tmp_path, pixel_size=PS, with_override=True, override_pixel_size=0.25, with_index=False
    )
    missing = store / "nope.spatial"
    with caplog.at_level(logging.WARNING):
        affine = resolve_dataset_mask_affine("ATERA", spec_path=missing, zarr_path=store)
    assert _mpp(affine) == (0.25, 0.25)
    assert str(missing) in caplog.text and "missing" in caplog.text
    assert "No experiment.spatial found" not in caplog.text

    unreadable = tmp_path / "bad.spatial"
    unreadable.write_text("not json")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        affine = resolve_dataset_mask_affine("ATERA", spec_path=unreadable, zarr_path=store)
    assert _mpp(affine) == (0.25, 0.25)
    assert str(unreadable) in caplog.text and "unreadable" in caplog.text


def test_atera_missing_explicit_spec_without_store_sidecar_warns(atera_store, tmp_path, caplog):
    store = _copy_store(atera_store, tmp_path)
    (store / "atera" / "experiment.spatial").unlink()
    missing = tmp_path / "typo.spatial"
    with caplog.at_level(logging.WARNING):
        affine = resolve_dataset_mask_affine("ATERA", spec_path=missing, zarr_path=store)
    assert _mpp(affine) == (PS, PS)
    assert str(missing) in caplog.text and "merxen_atera" in caplog.text


def test_atera_root_attr_and_shapes_fallbacks(atera_store, tmp_path, caplog):
    store = _copy_store(atera_store, tmp_path)
    (store / "atera" / "experiment.spatial").unlink()
    with caplog.at_level(logging.WARNING):
        assert _mpp(resolve_dataset_mask_affine("ATERA", zarr_path=store)) == (PS, PS)
    assert "merxen_atera" in caplog.text

    _drop_root_attr(store, "merxen_atera")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        sx, sy = _mpp(resolve_dataset_mask_affine("ATERA", zarr_path=store))
    assert sx == pytest.approx(PS) and sy == pytest.approx(PS)
    assert sx != 0.2125
    assert "shapes/atera_cell_boundaries" in caplog.text


def test_atera_refuses_without_any_pixel_size(tmp_path):
    store = _empty_group(tmp_path / "s.zarr", {"merxen_schema": {"platform": "ATERA"}})
    with pytest.raises(ValueError, match="ATERA pixel size unknown"):
        resolve_dataset_mask_affine("ATERA", zarr_path=store)
    with pytest.raises(ValueError, match="ATERA pixel size unknown"):
        resolve_dataset_mask_affine("ATERA")


def test_xenium_reads_experiment_xenium_pixel_size(tmp_path):
    spec = tmp_path / "experiment.xenium"
    spec.write_text(json.dumps({"pixel_size": 0.3}))
    assert _mpp(resolve_dataset_mask_affine("XENIUM", xenium_spec_path=spec)) == (0.3, 0.3)
    assert _mpp(resolve_dataset_mask_affine("XENIUM", spec_path=spec)) == (0.3, 0.3)
    spec.write_text(json.dumps({"microns_per_pixel": 0.4}))
    assert _mpp(resolve_dataset_mask_affine("XENIUM", spec_path=spec)) == (0.4, 0.4)


def test_xenium_fallback_unchanged(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert _mpp(resolve_dataset_mask_affine("XENIUM")) == (0.2125, 0.2125)
    assert "[XENIUM] Spec file missing/unreadable" in caplog.text
    bad = tmp_path / "experiment.xenium"
    bad.write_text("not json")
    assert _mpp(resolve_dataset_mask_affine("XENIUM", xenium_spec_path=bad)) == (0.2125, 0.2125)


def test_unknown_dataset_still_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset_mask_affine("VISIUM")


# ---------------------------------------------------------------------------
# Layer names and colormaps
# ---------------------------------------------------------------------------


def test_make_layer_name_uses_platform_dapi_aliases():
    assert make_layer_name("XENIUM", "image", "morphology_focus", "DAPI") == "Image | DAPI Xenium"
    assert make_layer_name("xenium", "image", "post_xenium_if", "DAPI") == "Image | DAPI Post Xenium-IF"
    assert make_layer_name("XENIUM", "image", "other", "DAPI") == "Image | DAPI"
    assert make_layer_name("ATERA", "image", "morphology_focus", "DAPI") == "Image | DAPI"
    assert (
        make_layer_name("ATERA", "image", "morphology_focus", "ATP1A1/CD45/E-Cadherin")
        == "Image | ATP1A1/CD45/E-Cadherin"
    )
    assert make_layer_name("MERSCOPE", "image", "morphology_focus", "DAPI") == "Image | DAPI"


def test_atera_channel_colormaps_are_distinct_and_explicit():
    fallback = {"gray", "magenta", "cyan", "yellow", "red", "orange"}
    # The store's channel order (DAPI, ATP1A1/..., 18S, alphaSMA/...) lines the
    # rotating fallback up with the spec by coincidence, so pin fallback_index=0
    # (gray) to prove the explicit platform mapping is what assigns the colours.
    cmaps = [
        V.image_colormap_for_channel(name, 0, platform="ATERA")
        for name in atera_stores.CHANNELS
    ]
    assert len(set(cmaps)) == len(cmaps)
    assert "gray" not in cmaps
    assert cmaps == ["blue", "magenta", "green", "yellow"]
    spec = P.platform_spec("ATERA")
    for name, cmap in zip(atera_stores.CHANNELS, cmaps):
        assert spec.channel_colormap(name) == cmap
    # Without the platform lookup the same calls fall back to gray.
    assert V.image_colormap_for_channel("ATP1A1/CD45/E-Cadherin", 0) == "gray"
    assert V.image_colormap_for_channel("alphaSMA/Vimentin", 0) == "gray"
    assert V.image_colormap_for_channel("ATP1A1/CD45/E-Cadherin", 0, platform="XENIUM") == "gray"
    # Existing rules are unchanged for other platforms and without a platform.
    assert V.image_colormap_for_channel("DAPI", 3, platform="XENIUM") == "blue"
    assert V.image_colormap_for_channel("PolyT", 0, platform="MERSCOPE") == "green"
    assert V.image_colormap_for_channel("ATP1A1/CD45/E-Cadherin", 1) in fallback
    assert V.image_colormap_for_channel("Cellbound1", 2) == "cyan"


# ---------------------------------------------------------------------------
# Fixture round trip
# ---------------------------------------------------------------------------


def test_fixture_round_trips_through_spatialdata(atera_store):
    import spatialdata as sd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdata = sd.read_zarr(str(atera_store))
    points = sdata.points["transcripts"]
    assert str(points["original_assignment"].dtype) == "UInt64"
    assert isinstance(points["feature_name"].dtype, pd.CategoricalDtype)
    assert not points["feature_name"].cat.known
    frame = points.compute()
    assert frame["original_assignment"].isna().any()
    assert frame["original_assignment"].notna().any()
    assert set(frame["feature_name"].astype(str)) <= set(atera_stores.FEATURE_DICTIONARY)
    assert atera_stores.ABSENT_GENE not in set(frame["feature_name"].astype(str))

    cells = sdata.shapes["atera_cell_boundaries"]
    assert cells.index.name == "instance_id" and "instance_id" in cells.columns
    assert cells.index.dtype == np.uint64 and int(cells.index.min()) == 1
    nuclei = sdata.shapes["atera_nucleus"]
    joins = nuclei["cell_instance_id"].astype(int).tolist()
    assert joins.count(7) == 2 and 6 not in joins

    channels = [str(c) for c in sdata.images["morphology_focus"]["scale0"]["image"].c.values]
    assert channels == list(atera_stores.CHANNELS)
    assert set(sdata.labels) == {
        "cell_labels", "nucleus_labels", "atera_cell_boundaries_labels", "atera_nucleus_labels",
    }
    assert len(sdata.tables["table_original_clustering_squidpy"]) == len(atera_stores.CELLS) - 1
    assert "X_umap_vendor" in sdata.tables["table_original"].obsm
    quant = sdata.tables["table_atera_cell_boundaries_image_quantification"]
    assert quant.uns["mask_image_quantification"]["mask_label_key"] == "cell_labels"


def test_fixture_parquet_layout_mirrors_real_store(atera_store):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = atera_store / "points" / "transcripts" / "points.parquet"
    names = sorted(p.name for p in directory.glob("*.parquet"))
    assert names[:3] == ["part.0.parquet", "part.1.parquet", "part.10.parquet"]
    dictionaries = set()
    for path in directory.glob("*.parquet"):
        meta = pq.ParquetFile(path)
        assert meta.schema_arrow.field("feature_name").type == pa.dictionary(pa.int16(), pa.string())
        assert meta.schema_arrow.field("original_assignment").type == pa.uint64()
        chunk = meta.read_row_group(0, columns=["feature_name"]).column(0).chunk(0)
        dictionaries.add(tuple(chunk.dictionary.to_pylist()))
    assert dictionaries == {atera_stores.FEATURE_DICTIONARY}
    families = {atera_stores.control_family(n) for n in atera_stores.CONTROLS}
    assert families == set(atera_stores.CONTROL_FAMILIES)

    index = json.loads((atera_store / "merxen_atera_points_index.json").read_text())
    entry = index["points"]["transcripts"]
    assert entry["fingerprint"]["n_files"] == len(names)
    assert entry["fingerprint"]["n_rows"] == sum(
        pq.ParquetFile(directory / e["file"]).metadata.num_rows for e in entry["files"]
    )
    first = entry["files"][0]
    assert first["file"] == "part.0.parquet" and first["tile"] == "-1,1"
    assert first["bbox_um"][0] < 0 and first["block_bounds_um"] is None and first["n_row_groups"] == 1
    tiny = pq.ParquetFile(directory / entry["files"][-1]["file"])
    assert tiny.metadata.num_row_groups == 1
    stats = tiny.metadata.row_group(0).column(3).statistics
    assert (stats.min, stats.max) == (atera_stores.GENES[0], atera_stores.GENES[-1])
    assert any(e["n_row_groups"] > 1 for e in entry["files"])

    # Cell 3 straddles a tile boundary, so its assigned transcripts sit in two tiles.
    tiles = {
        e["tile"]
        for e in entry["files"]
        if 3 in set(
            pq.read_table(directory / e["file"], columns=["original_assignment"])
            .column(0).drop_null().to_pylist()
        )
    }
    assert tiles == {"0,0", "1,0"}


def test_fixture_label_alias_markers_and_stale_index(atera_store, tmp_path):
    cell = json.loads((atera_store / "labels" / "atera_cell_boundaries_labels" / "zarr.json").read_text())
    marker = cell["attributes"]["napari_compare_label_cache"]
    assert marker["source"] == "vendor_labels" and marker["source_label_key"] == "cell_labels"
    assert marker["version"] == 2 and marker["complete"] is True
    nucleus = json.loads((atera_store / "labels" / "atera_nucleus_labels" / "zarr.json").read_text())
    assert nucleus["attributes"]["napari_compare_label_cache"]["cell_join_key"] == "cell_instance_id"

    stale = atera_stores.build_atera_latest_store(tmp_path, stale_index=True, with_experiment=False)
    index = json.loads((stale / "merxen_atera_points_index.json").read_text())
    good = json.loads((atera_store / "merxen_atera_points_index.json").read_text())
    assert (
        index["points"]["transcripts"]["fingerprint"]["n_rows"]
        == good["points"]["transcripts"]["fingerprint"]["n_rows"] + 1
    )
    assert not (stale / "atera" / "experiment.spatial").exists()
