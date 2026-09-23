"""Tests for the MerXen version-2 paired-data contract reader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from spatialdata.transformations import Affine

from napari_compare_xenium_merscope.paired_data import (
    PairedDataContractError,
    resolve_paired_data_contract,
)


def _pair() -> tuple[SimpleNamespace, SimpleNamespace]:
    fingerprint = "transform-sha"
    native_fingerprint = "native-sha"
    mappings = {
        "points": {"transcripts": "transcripts_aligned_nonrigid"},
        "shapes": {
            "MOSAIK_proseg": "MOSAIK_proseg_aligned_nonrigid",
            "cellpose_nuclei": "cellpose_nuclei_aligned_nonrigid",
        },
        "tables": {"table": "table_aligned_nonrigid"},
        "labels": {
            "MOSAIK_proseg": "MOSAIK_proseg_aligned_nonrigid_labels",
            "cellpose_nuclei": "cellpose_nuclei_aligned_nonrigid_labels",
        },
        "images": {
            "MERSCOPE_z_projection": "MERSCOPE_z_projection_aligned_nonrigid"
        },
    }
    artifacts = {
        f"{kind}/{output}": {
            "status": "complete",
            "boundary_type": "nucleus" if "nuclei" in output else "cell",
        }
        for kind, values in mappings.items()
        for output in values.values()
    }
    native_proseg = {
        "points": "transcripts",
        "shape": "MOSAIK_proseg",
        "table": "table",
        "coordinate_variant_of": None,
    }
    native_nuclei = {
        "points": "transcripts",
        "shape": "cellpose_nuclei",
        "table": None,
        "coordinate_variant_of": None,
    }
    merscope_schema = {
        "primary_points": "transcripts",
        "segmentations": {
            "proseg": native_proseg,
            "cellpose_nuclei": native_nuclei,
            "proseg_aligned_nonrigid": {
                "points": "transcripts_aligned_nonrigid",
                "shape": "MOSAIK_proseg_aligned_nonrigid",
                "table": "table_aligned_nonrigid",
                "coordinate_variant_of": "proseg",
            },
            "cellpose_nuclei_aligned_nonrigid": {
                "points": "transcripts_aligned_nonrigid",
                "shape": "cellpose_nuclei_aligned_nonrigid",
                "table": None,
                "coordinate_variant_of": "cellpose_nuclei",
            },
        },
    }
    xenium_schema = {
        "primary_points": "transcripts",
        "segmentations": {
            "proseg": {
                "points": "transcripts",
                "shape": "MOSAIK_proseg",
                "table": "table",
            }
        },
    }
    manifest = {
        "version": 2,
        "pair_id": "pair-1",
        "roles": {"moving": "MERSCOPE", "fixed": "XENIUM"},
        "common_coordinate_system": "merxen_xenium",
        "fixed_grid": {
            "image_key": "morphology_focus",
            "dimensions": {"width": 12, "height": 10},
            "coordinate_system": "merxen_xenium",
            "pixel_to_world_affine": [
                [0.2125, 0.0, 10.0],
                [0.0, 0.2125, 20.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "transform": {"fingerprint": fingerprint},
        "native_input": {"fingerprint": native_fingerprint},
        "mappings": mappings,
        "artifacts": artifacts,
        "complete": True,
    }
    merscope = SimpleNamespace(
        attrs={"merxen_alignment": manifest, "merxen_schema": merscope_schema},
        points={"transcripts_aligned_nonrigid": object()},
        shapes={
            "MOSAIK_proseg_aligned_nonrigid": object(),
            "cellpose_nuclei_aligned_nonrigid": object(),
        },
        tables={"table_aligned_nonrigid": object()},
        labels={
            "MOSAIK_proseg_aligned_nonrigid_labels": object(),
            "cellpose_nuclei_aligned_nonrigid_labels": object(),
        },
        images={"MERSCOPE_z_projection_aligned_nonrigid": object()},
    )
    xenium = SimpleNamespace(
        attrs={
            "merxen_alignment_pair_reference": {
                "version": 1,
                "pair_id": "pair-1",
                "counterpart_platform": "MERSCOPE",
                "counterpart_native_fingerprint": native_fingerprint,
                "transform_fingerprint": fingerprint,
            },
            "merxen_schema": xenium_schema,
        },
        points={"transcripts": object()},
        shapes={"MOSAIK_proseg": object()},
        tables={"table": object()},
        labels={"MOSAIK_proseg_labels": object()},
        images={"morphology_focus": object()},
    )
    return merscope, xenium


def _transformed_raster(
    matrix: list[list[float]],
    *,
    coordinate_system: str = "global",
) -> xr.DataArray:
    raster = xr.DataArray(np.zeros((2, 2), dtype=np.uint8), dims=("y", "x"))
    raster.attrs["transform"] = {
        coordinate_system: Affine(
            matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
    }
    return raster


def _write_marker(
    store: Path,
    group: str,
    key: str,
    attr_name: str,
    marker: dict[str, object],
) -> None:
    element = zarr.open_group(str(store / group / key), mode="a")
    element.attrs[attr_name] = marker


def _persisted_xenium_caches(
    tmp_path: Path,
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, str]]:
    moving, fixed = _pair()
    store = tmp_path / "xenium.zarr"
    fixed.path = store
    fixed_affine = moving.attrs["merxen_alignment"]["fixed_grid"][
        "pixel_to_world_affine"
    ]
    identity = np.eye(3).tolist()
    keys = {
        "image": "_napari_compare_imgpyr__morphology_focus__ds4",
        "label": "_napari_compare_labelpyr__MOSAIK_proseg_labels__ds4",
        "outline": "_napari_compare_outline__MOSAIK_proseg_labels__w1",
    }
    fixed.images["morphology_focus"] = _transformed_raster(identity)
    fixed.images[keys["image"]] = _transformed_raster(identity)
    fixed.labels["MOSAIK_proseg_labels"] = _transformed_raster(fixed_affine)
    fixed.labels[keys["label"]] = _transformed_raster(fixed_affine)
    fixed.labels[keys["outline"]] = _transformed_raster(fixed_affine)

    _write_marker(
        store,
        "labels",
        "MOSAIK_proseg_labels",
        "napari_compare_label_cache",
        {
            "version": 2,
            "complete": True,
            "source_shape_key": "MOSAIK_proseg",
            "shape": [10, 12],
            "chunks": [10, 12],
        },
    )
    _write_marker(
        store,
        "images",
        keys["image"],
        "napari_compare_derived_cache",
        {
            "version": 2,
            "complete": True,
            "kind": "image_pyramid",
            "source_image_key": "morphology_focus",
            "downsample": 4,
            "min_size": 1024,
            "levels": 2,
        },
    )
    _write_marker(
        store,
        "labels",
        keys["label"],
        "napari_compare_derived_cache",
        {
            "version": 2,
            "complete": True,
            "kind": "label_pyramid",
            "source_label_key": "MOSAIK_proseg_labels",
            "downsample": 4,
            "min_size": 1024,
            "levels": 2,
        },
    )
    _write_marker(
        store,
        "labels",
        keys["outline"],
        "napari_compare_derived_cache",
        {
            "version": 2,
            "complete": True,
            "kind": "label_outline",
            "source_label_key": "MOSAIK_proseg_labels",
            "width": 1,
            "min_size": 1024,
            "tile_size": 1024,
            "pyramid_mode": "coverage_mean_v1",
            "value_max": 255,
            "levels": 2,
        },
    )
    return moving, fixed, keys


def test_resolve_contract_uses_exact_materialized_merscope_elements() -> None:
    merscope, xenium = _pair()

    contract = resolve_paired_data_contract(merscope, xenium)

    assert contract.pair_id == "pair-1"
    assert contract.fixed_width == 12
    assert contract.fixed_height == 10
    assert contract.merscope.points_key == "transcripts_aligned_nonrigid"
    assert contract.merscope.image_keys == (
        "MERSCOPE_z_projection_aligned_nonrigid",
    )
    assert contract.merscope.segmentation("proseg").label_key == (
        "MOSAIK_proseg_aligned_nonrigid_labels"
    )
    assert contract.merscope.segmentation("cellpose_nuclei").boundary_type == (
        "nucleus"
    )
    assert contract.xenium.points_key == "transcripts"
    assert contract.xenium.image_keys == ("morphology_focus",)
    assert contract.xenium.segmentation("proseg").label_key == (
        "MOSAIK_proseg_labels"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda moving, _fixed: moving.attrs["merxen_alignment"].update(
                complete=False
            ),
            "incomplete or stale",
        ),
        (
            lambda _moving, fixed: fixed.attrs[
                "merxen_alignment_pair_reference"
            ].update(pair_id="other"),
            "different alignment pairs",
        ),
        (
            lambda _moving, fixed: fixed.attrs[
                "merxen_alignment_pair_reference"
            ].update(transform_fingerprint="other"),
            "different alignment transforms",
        ),
        (
            lambda moving, _fixed: moving.images.pop(
                "MERSCOPE_z_projection_aligned_nonrigid"
            ),
            "missing materialized outputs",
        ),
    ],
)
def test_resolve_contract_rejects_incomplete_or_mismatched_pairs(
    mutation,
    message: str,
) -> None:
    moving, fixed = deepcopy(_pair())
    mutation(moving, fixed)

    with pytest.raises(PairedDataContractError, match=message):
        resolve_paired_data_contract(moving, fixed)


def test_resolve_contract_rejects_incomplete_artifact() -> None:
    moving, fixed = _pair()
    moving.attrs["merxen_alignment"]["artifacts"][
        "images/MERSCOPE_z_projection_aligned_nonrigid"
    ]["status"] = "stale"

    with pytest.raises(PairedDataContractError, match="not complete"):
        resolve_paired_data_contract(moving, fixed)


def test_resolve_contract_trusts_complete_source_matched_xenium_caches(
    tmp_path: Path,
) -> None:
    moving, fixed, keys = _persisted_xenium_caches(tmp_path)

    contract = resolve_paired_data_contract(moving, fixed)

    assert contract.xenium.image_pyramid_keys == {
        "morphology_focus": keys["image"]
    }
    segmentation = contract.xenium.segmentation("proseg")
    assert segmentation is not None
    assert segmentation.label_pyramid_key == keys["label"]
    assert segmentation.outline_key == keys["outline"]


def test_resolve_contract_ignores_xenium_caches_with_untrusted_markers_or_cs(
    tmp_path: Path,
) -> None:
    moving, fixed, keys = _persisted_xenium_caches(tmp_path)
    image_group = zarr.open_group(
        str(fixed.path / "images" / keys["image"]), mode="a"
    )
    image_marker = dict(image_group.attrs["napari_compare_derived_cache"])
    image_marker["source_image_key"] = "another_image"
    image_group.attrs["napari_compare_derived_cache"] = image_marker

    label_group = zarr.open_group(
        str(fixed.path / "labels" / keys["label"]), mode="a"
    )
    label_marker = dict(label_group.attrs["napari_compare_derived_cache"])
    label_marker["version"] = 1
    label_group.attrs["napari_compare_derived_cache"] = label_marker

    fixed_affine = moving.attrs["merxen_alignment"]["fixed_grid"][
        "pixel_to_world_affine"
    ]
    fixed.labels[keys["outline"]] = _transformed_raster(
        fixed_affine,
        coordinate_system="another_coordinate_system",
    )

    contract = resolve_paired_data_contract(moving, fixed)

    assert contract.xenium.image_pyramid_keys == {}
    segmentation = contract.xenium.segmentation("proseg")
    assert segmentation is not None
    assert segmentation.label_pyramid_key is None
    assert segmentation.outline_key is None


@pytest.mark.parametrize(
    ("marker_mutation", "transform_mutation", "message"),
    [
        (
            {"source_shape_key": "another_shape"},
            False,
            "cache marker",
        ),
        (
            {},
            True,
            "fixed-grid affine",
        ),
    ],
)
def test_resolve_contract_rejects_untrusted_xenium_base_labels(
    tmp_path: Path,
    marker_mutation: dict[str, object],
    transform_mutation: bool,
    message: str,
) -> None:
    moving, fixed, _keys = _persisted_xenium_caches(tmp_path)
    label_group = zarr.open_group(
        str(fixed.path / "labels" / "MOSAIK_proseg_labels"), mode="a"
    )
    marker = dict(label_group.attrs["napari_compare_label_cache"])
    marker.update(marker_mutation)
    label_group.attrs["napari_compare_label_cache"] = marker
    if transform_mutation:
        fixed.labels["MOSAIK_proseg_labels"] = _transformed_raster(
            np.eye(3).tolist()
        )

    with pytest.raises(PairedDataContractError, match=message):
        resolve_paired_data_contract(moving, fixed)
