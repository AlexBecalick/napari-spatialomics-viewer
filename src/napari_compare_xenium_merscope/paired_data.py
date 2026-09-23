"""Strict reader for MerXen's viewer-ready paired alignment contract.

The comparison viewer deliberately does not calculate registrations.  Paired
views are enabled only when the moving MERSCOPE store contains MerXen's complete
version-2 materialization manifest and the fixed Xenium store carries the
matching pair reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

ALIGNMENT_MANIFEST_ATTR = "merxen_alignment"
ALIGNMENT_PAIR_REFERENCE_ATTR = "merxen_alignment_pair_reference"
MERXEN_SCHEMA_ATTR = "merxen_schema"
ALIGNMENT_MANIFEST_VERSION = 2
DERIVED_IMAGE_PYRAMID_PREFIX = "_napari_compare_imgpyr__"
DERIVED_LABEL_PYRAMID_PREFIX = "_napari_compare_labelpyr__"
DERIVED_OUTLINE_PREFIX = "_napari_compare_outline__"
DERIVED_CACHE_ATTR = "napari_compare_derived_cache"
LABEL_CACHE_ATTR = "napari_compare_label_cache"
DERIVED_CACHE_VERSION = 2
LABEL_CACHE_VERSION = 2
OUTLINE_PYRAMID_MODE = "coverage_mean_v1"
OUTLINE_COVERAGE_MAX = 255


class PairedDataContractError(ValueError):
    """Raised when two stores are not a complete, compatible aligned pair."""


@dataclass(frozen=True)
class SegmentationProfile:
    """Exact element keys for one logical segmentation in one platform."""

    logical_key: str
    shape_key: str
    label_key: str | None
    points_key: str
    table_key: str | None
    boundary_type: str = "cell"
    label_pyramid_key: str | None = None
    outline_key: str | None = None


@dataclass(frozen=True)
class DatasetElementProfile:
    """Exact, non-heuristic SpatialData element selection for one platform."""

    platform: str
    points_key: str
    image_keys: tuple[str, ...]
    image_pyramid_keys: Mapping[str, str]
    segmentations: tuple[SegmentationProfile, ...]
    coordinate_system: str
    pixel_to_world_affine: tuple[tuple[float, float, float], ...]

    @property
    def shape_keys(self) -> tuple[str, ...]:
        """Return shape keys in stable logical-segmentation order."""
        return tuple(item.shape_key for item in self.segmentations)

    @property
    def label_keys(self) -> tuple[str, ...]:
        """Return available label keys in stable logical-segmentation order."""
        return tuple(
            item.label_key for item in self.segmentations if item.label_key is not None
        )

    @property
    def table_keys(self) -> tuple[str, ...]:
        """Return unique available table keys in stable order."""
        return tuple(
            dict.fromkeys(
                item.table_key
                for item in self.segmentations
                if item.table_key is not None
            )
        )

    def segmentation(self, logical_key: str) -> SegmentationProfile | None:
        """Return the segmentation matching ``logical_key``, if present."""
        wanted = str(logical_key)
        return next(
            (item for item in self.segmentations if item.logical_key == wanted),
            None,
        )


@dataclass(frozen=True)
class PairedDataContract:
    """Validated MERSCOPE/Xenium pair and its exact viewer element profiles."""

    pair_id: str
    coordinate_system: str
    fixed_width: int
    fixed_height: int
    transform_fingerprint: str
    native_fingerprint: str
    merscope: DatasetElementProfile
    xenium: DatasetElementProfile
    mappings: Mapping[str, Mapping[str, str]]

    def profile(self, platform: str) -> DatasetElementProfile:
        """Return the profile for ``platform``."""
        token = str(platform).upper()
        if token == "MERSCOPE":
            return self.merscope
        if token == "XENIUM":
            return self.xenium
        raise KeyError(f"Unsupported paired platform: {platform!r}")


def resolve_paired_data_contract(
    merscope_sdata: Any,
    xenium_sdata: Any,
) -> PairedDataContract:
    """Validate two SpatialData objects and resolve exact paired-view elements.

    Args:
        merscope_sdata: Moving MERSCOPE SpatialData object.
        xenium_sdata: Fixed Xenium SpatialData object.

    Returns:
        A validated, immutable paired-data contract.

    Raises:
        PairedDataContractError: If the stores are stale, incomplete, mismatched,
            or missing a declared artifact.
    """
    manifest = _mapping_attr(merscope_sdata, ALIGNMENT_MANIFEST_ATTR)
    reference = _mapping_attr(xenium_sdata, ALIGNMENT_PAIR_REFERENCE_ATTR)

    if manifest.get("version") != ALIGNMENT_MANIFEST_VERSION:
        raise PairedDataContractError(
            "MERSCOPE store does not contain a MerXen version-2 alignment manifest."
        )
    if manifest.get("complete") is not True:
        reason = manifest.get("invalidation_reason")
        suffix = f" ({reason})" if reason else ""
        raise PairedDataContractError(
            f"MERSCOPE alignment materialization is incomplete or stale{suffix}."
        )
    roles = _as_mapping(manifest.get("roles"), "manifest.roles")
    if str(roles.get("moving", "")).upper() != "MERSCOPE" or str(
        roles.get("fixed", "")
    ).upper() != "XENIUM":
        raise PairedDataContractError(
            "Paired viewing requires moving MERSCOPE and fixed Xenium roles."
        )

    pair_id = _required_text(manifest, "pair_id", "manifest")
    if reference.get("version") != 1:
        raise PairedDataContractError(
            "Xenium store does not contain a MerXen alignment pair reference."
        )
    if str(reference.get("pair_id", "")) != pair_id:
        raise PairedDataContractError(
            "The MERSCOPE and Xenium stores belong to different alignment pairs."
        )
    if str(reference.get("counterpart_platform", "")).upper() != "MERSCOPE":
        raise PairedDataContractError(
            "The Xenium pair reference does not identify a MERSCOPE counterpart."
        )

    transform = _as_mapping(manifest.get("transform"), "manifest.transform")
    transform_fingerprint = _required_text(
        transform, "fingerprint", "manifest.transform"
    )
    if str(reference.get("transform_fingerprint", "")) != transform_fingerprint:
        raise PairedDataContractError(
            "The two stores reference different alignment transforms."
        )
    native_input = _as_mapping(manifest.get("native_input"), "manifest.native_input")
    native_fingerprint = _required_text(
        native_input, "fingerprint", "manifest.native_input"
    )
    if (
        str(reference.get("counterpart_native_fingerprint", ""))
        != native_fingerprint
    ):
        raise PairedDataContractError(
            "The Xenium pair reference points to a different MERSCOPE revision."
        )

    coordinate_system = _required_text(
        manifest, "common_coordinate_system", "manifest"
    )
    fixed_grid = _as_mapping(manifest.get("fixed_grid"), "manifest.fixed_grid")
    dimensions = _as_mapping(
        fixed_grid.get("dimensions"), "manifest.fixed_grid.dimensions"
    )
    try:
        width = int(dimensions["width"])
        height = int(dimensions["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PairedDataContractError(
            "Alignment manifest has invalid fixed-grid dimensions."
        ) from exc
    if width <= 0 or height <= 0:
        raise PairedDataContractError(
            "Alignment manifest fixed-grid dimensions must be positive."
        )
    if str(fixed_grid.get("coordinate_system", "")) != coordinate_system:
        raise PairedDataContractError(
            "Alignment manifest fixed grid uses a different coordinate system."
        )
    affine = _affine_tuple(fixed_grid.get("pixel_to_world_affine"))
    fixed_image_key = _required_text(fixed_grid, "image_key", "manifest.fixed_grid")
    if fixed_image_key not in getattr(xenium_sdata, "images", {}):
        raise PairedDataContractError(
            f"Xenium fixed image {fixed_image_key!r} is missing."
        )

    mappings = _manifest_mappings(manifest)
    artifacts = _as_mapping(manifest.get("artifacts"), "manifest.artifacts")
    _validate_declared_outputs(merscope_sdata, mappings, artifacts)

    merscope_profile = _merscope_profile(
        merscope_sdata,
        mappings=mappings,
        artifacts=artifacts,
        coordinate_system=coordinate_system,
        affine=affine,
    )
    xenium_profile = _xenium_profile(
        xenium_sdata,
        fixed_image_key=fixed_image_key,
        coordinate_system=coordinate_system,
        affine=affine,
    )
    frozen_mappings = MappingProxyType(
        {
            kind: MappingProxyType(dict(kind_mapping))
            for kind, kind_mapping in mappings.items()
        }
    )
    return PairedDataContract(
        pair_id=pair_id,
        coordinate_system=coordinate_system,
        fixed_width=width,
        fixed_height=height,
        transform_fingerprint=transform_fingerprint,
        native_fingerprint=native_fingerprint,
        merscope=merscope_profile,
        xenium=xenium_profile,
        mappings=frozen_mappings,
    )


def _merscope_profile(
    sdata_obj: Any,
    *,
    mappings: Mapping[str, Mapping[str, str]],
    artifacts: Mapping[str, Any],
    coordinate_system: str,
    affine: tuple[tuple[float, float, float], ...],
) -> DatasetElementProfile:
    schema = _schema(sdata_obj)
    primary_native = str(schema.get("primary_points") or "transcripts")
    points_key = mappings["points"].get(primary_native)
    if points_key is None:
        raise PairedDataContractError(
            f"Alignment manifest does not map primary points {primary_native!r}."
        )

    native_registry = {
        str(branch): dict(entry)
        for branch, entry in _registry(schema).items()
        if isinstance(entry, Mapping) and entry.get("coordinate_variant_of") is None
    }
    aligned_registry = {
        str(entry.get("coordinate_variant_of")): dict(entry)
        for entry in _registry(schema).values()
        if isinstance(entry, Mapping) and entry.get("coordinate_variant_of") is not None
    }
    segmentations: list[SegmentationProfile] = []
    for logical_key, native_entry in sorted(native_registry.items()):
        aligned_entry = aligned_registry.get(logical_key)
        if aligned_entry is None:
            continue
        native_shape = str(native_entry.get("shape", ""))
        aligned_shape = mappings["shapes"].get(native_shape)
        if aligned_shape is None or aligned_shape != str(aligned_entry.get("shape", "")):
            continue
        native_table = native_entry.get("table")
        table_key = (
            None
            if native_table is None
            else mappings["tables"].get(str(native_table))
        )
        label_key = mappings["labels"].get(native_shape)
        artifact = artifacts.get(f"labels/{label_key}", {}) if label_key else {}
        segmentations.append(
            SegmentationProfile(
                logical_key=logical_key,
                shape_key=aligned_shape,
                label_key=label_key,
                points_key=str(aligned_entry.get("points") or points_key),
                table_key=table_key,
                boundary_type=str(
                    artifact.get("boundary_type", "cell")
                    if isinstance(artifact, Mapping)
                    else "cell"
                ),
                label_pyramid_key=(
                    str(artifact["pyramid_key"])
                    if isinstance(artifact, Mapping) and artifact.get("pyramid_key")
                    else None
                ),
                outline_key=(
                    str(artifact["outline_key"])
                    if isinstance(artifact, Mapping) and artifact.get("outline_key")
                    else None
                ),
            )
        )
    if not segmentations:
        raise PairedDataContractError(
            "MERSCOPE manifest has no registered aligned segmentation branches."
        )
    image_keys = tuple(dict.fromkeys(mappings["images"].values()))
    if not image_keys:
        raise PairedDataContractError(
            "MERSCOPE manifest has no materialized aligned image."
        )
    return DatasetElementProfile(
        platform="MERSCOPE",
        points_key=points_key,
        image_keys=image_keys,
        image_pyramid_keys=MappingProxyType(
            {
                output_key: str(artifact["pyramid_key"])
                for output_key in image_keys
                for artifact in [artifacts.get(f"images/{output_key}")]
                if isinstance(artifact, Mapping) and artifact.get("pyramid_key")
            }
        ),
        segmentations=tuple(segmentations),
        coordinate_system=coordinate_system,
        pixel_to_world_affine=affine,
    )


def _xenium_profile(
    sdata_obj: Any,
    *,
    fixed_image_key: str,
    coordinate_system: str,
    affine: tuple[tuple[float, float, float], ...],
) -> DatasetElementProfile:
    schema = _schema(sdata_obj)
    points_key = str(schema.get("primary_points") or "transcripts")
    if points_key not in getattr(sdata_obj, "points", {}):
        raise PairedDataContractError(
            f"Xenium primary points {points_key!r} are missing."
        )
    segmentations: list[SegmentationProfile] = []
    labels = getattr(sdata_obj, "labels", {})
    shapes = getattr(sdata_obj, "shapes", {})
    tables = getattr(sdata_obj, "tables", {})
    for logical_key, raw_entry in sorted(_registry(schema).items()):
        entry = dict(raw_entry)
        if entry.get("coordinate_variant_of") is not None:
            continue
        shape_key = str(entry.get("shape", ""))
        if not shape_key or shape_key not in shapes:
            continue
        label_key = _native_label_key(shape_key, labels)
        if label_key is not None:
            _validate_xenium_base_label(
                sdata_obj,
                label_key=label_key,
                shape_key=shape_key,
                fixed_grid_affine=affine,
            )
        label_pyramid_key = _derived_cache_key(
            sdata_obj,
            labels,
            group="labels",
            prefix=DERIVED_LABEL_PYRAMID_PREFIX,
            source_key=label_key,
            kind="label_pyramid",
        )
        outline_key = _derived_cache_key(
            sdata_obj,
            labels,
            group="labels",
            prefix=DERIVED_OUTLINE_PREFIX,
            source_key=label_key,
            kind="label_outline",
        )
        table_key = entry.get("table")
        if table_key is not None and str(table_key) not in tables:
            table_key = None
        token = f"{logical_key} {shape_key}".lower()
        segmentations.append(
            SegmentationProfile(
                logical_key=str(logical_key),
                shape_key=shape_key,
                label_key=label_key,
                points_key=str(entry.get("points") or points_key),
                table_key=None if table_key is None else str(table_key),
                boundary_type="nucleus" if "nucle" in token else "cell",
                label_pyramid_key=label_pyramid_key,
                outline_key=outline_key,
            )
        )
    if not segmentations:
        raise PairedDataContractError(
            "Xenium store has no registered native segmentation branches."
        )
    return DatasetElementProfile(
        platform="XENIUM",
        points_key=points_key,
        image_keys=(fixed_image_key,),
        image_pyramid_keys=MappingProxyType(
            {
                fixed_image_key: pyramid_key
                for pyramid_key in [
                    _derived_cache_key(
                        sdata_obj,
                        getattr(sdata_obj, "images", {}),
                        group="images",
                        prefix=DERIVED_IMAGE_PYRAMID_PREFIX,
                        source_key=fixed_image_key,
                        kind="image_pyramid",
                    )
                ]
                if pyramid_key is not None
            }
        ),
        segmentations=tuple(segmentations),
        coordinate_system=coordinate_system,
        pixel_to_world_affine=affine,
    )


def _manifest_mappings(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    raw = _as_mapping(manifest.get("mappings"), "manifest.mappings")
    result: dict[str, dict[str, str]] = {}
    for kind in ("points", "shapes", "tables", "labels", "images"):
        values = _as_mapping(raw.get(kind), f"manifest.mappings.{kind}")
        result[kind] = {str(key): str(value) for key, value in values.items()}
    return result


def _validate_declared_outputs(
    sdata_obj: Any,
    mappings: Mapping[str, Mapping[str, str]],
    artifacts: Mapping[str, Any],
) -> None:
    missing: list[str] = []
    incomplete: list[str] = []
    for kind, kind_mapping in mappings.items():
        container = getattr(sdata_obj, kind, {})
        for output_key in kind_mapping.values():
            if output_key not in container:
                missing.append(f"{kind}/{output_key}")
            artifact_key = f"{kind}/{output_key}"
            artifact = artifacts.get(artifact_key)
            if not isinstance(artifact, Mapping) or artifact.get("status") != "complete":
                incomplete.append(artifact_key)
                continue
            _validate_declared_viewer_caches(
                sdata_obj,
                kind=kind,
                artifact_key=artifact_key,
                artifact=artifact,
                missing=missing,
            )
    if missing:
        raise PairedDataContractError(
            "MERSCOPE store is missing materialized outputs: " + ", ".join(missing)
        )
    if incomplete:
        raise PairedDataContractError(
            "MERSCOPE materialized outputs are not complete: "
            + ", ".join(incomplete)
        )


def _validate_declared_viewer_caches(
    sdata_obj: Any,
    *,
    kind: str,
    artifact_key: str,
    artifact: Mapping[str, Any],
    missing: list[str],
) -> None:
    """Validate materializer-declared cache elements without inventing names."""
    if kind == "images":
        pyramid_key = artifact.get("pyramid_key")
        pyramid_status = str(artifact.get("pyramid_status", ""))
        if pyramid_status in {"built", "skipped"}:
            if not pyramid_key or str(pyramid_key) not in getattr(
                sdata_obj, "images", {}
            ):
                missing.append(f"{artifact_key}:pyramid/{pyramid_key or '<undeclared>'}")
        return
    if kind != "labels":
        return
    cache_status = artifact.get("cache_status")
    status = cache_status if isinstance(cache_status, Mapping) else {}
    for field, status_field in (
        ("pyramid_key", "label_pyramid"),
        ("outline_key", "outline"),
    ):
        cache_key = artifact.get(field)
        cache_state = str(status.get(status_field, ""))
        if cache_state in {"built", "skipped"}:
            if not cache_key or str(cache_key) not in getattr(
                sdata_obj, "labels", {}
            ):
                missing.append(
                    f"{artifact_key}:{field}/{cache_key or '<undeclared>'}"
                )


def _native_label_key(shape_key: str, labels: Any) -> str | None:
    for candidate in (shape_key, f"{shape_key}_labels"):
        if candidate in labels:
            return candidate
    return None


def _derived_cache_key(
    sdata_obj: Any,
    collection: Any,
    *,
    group: str,
    prefix: str,
    source_key: str | None,
    kind: str,
) -> str | None:
    """Return a complete, source-matched cache with its source transform.

    MerXen's completion markers are raw Zarr group attrs; SpatialData does not
    expose them on the loaded xarray object.  A filename prefix alone is not a
    provenance guarantee, so persisted paired stores only use a cache after its
    marker and SpatialData transform agree with the declared source element.
    """
    if not source_key:
        return None
    expected_prefix = f"{prefix}{source_key}__"
    candidates = sorted(
        str(key) for key in collection if str(key).startswith(expected_prefix)
    )
    if not candidates:
        return None

    # Simple in-memory objects are useful to callers/tests and have no raw Zarr
    # marker to inspect.  Real ``read_zarr`` objects always expose ``path``.
    if _spatialdata_store_path(sdata_obj) is None:
        return candidates[0]

    source_element = collection.get(source_key)
    if source_element is None:
        return None
    for candidate in candidates:
        marker = _zarr_element_marker(
            sdata_obj,
            group=group,
            key=candidate,
            attr_name=DERIVED_CACHE_ATTR,
        )
        if not _valid_derived_cache_marker(
            marker,
            key=candidate,
            kind=kind,
            source_key=str(source_key),
        ):
            continue
        if not _elements_share_spatial_transform(
            source_element,
            collection[candidate],
        ):
            continue
        return candidate
    return None


def _validate_xenium_base_label(
    sdata_obj: Any,
    *,
    label_key: str,
    shape_key: str,
    fixed_grid_affine: tuple[tuple[float, float, float], ...],
) -> None:
    """Reject persisted fixed-side label rasters not tied to the fixed grid."""
    if _spatialdata_store_path(sdata_obj) is None:
        return
    marker = _zarr_element_marker(
        sdata_obj,
        group="labels",
        key=label_key,
        attr_name=LABEL_CACHE_ATTR,
    )
    if not (
        marker.get("complete") is True
        and marker.get("version") == LABEL_CACHE_VERSION
        and marker.get("source_shape_key") == str(shape_key)
    ):
        raise PairedDataContractError(
            f"Xenium labels/{label_key} has no complete version-"
            f"{LABEL_CACHE_VERSION} cache marker for shapes/{shape_key}."
        )
    label_element = getattr(sdata_obj, "labels", {}).get(label_key)
    if label_element is None or not _element_matches_affine(
        label_element,
        np.asarray(fixed_grid_affine, dtype=float),
    ):
        raise PairedDataContractError(
            f"Xenium labels/{label_key} does not use the declared fixed-grid "
            "affine."
        )


def _valid_derived_cache_marker(
    marker: Mapping[str, Any],
    *,
    key: str,
    kind: str,
    source_key: str,
) -> bool:
    if not (
        marker.get("complete") is True
        and marker.get("version") == DERIVED_CACHE_VERSION
        and marker.get("kind") == kind
    ):
        return False
    source_field = (
        "source_image_key" if kind == "image_pyramid" else "source_label_key"
    )
    if marker.get(source_field) != source_key:
        return False
    try:
        levels = int(marker["levels"])
        min_size = int(marker["min_size"])
    except (KeyError, TypeError, ValueError):
        return False
    if levels < 1 or min_size < 1:
        return False
    if kind in {"image_pyramid", "label_pyramid"}:
        try:
            downsample = int(marker["downsample"])
        except (KeyError, TypeError, ValueError):
            return False
        return downsample >= 2 and key.endswith(f"__ds{downsample}")
    if kind == "label_outline":
        try:
            width = int(marker["width"])
            tile_size = int(marker["tile_size"])
            value_max = int(marker["value_max"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            width >= 1
            and tile_size >= 1
            and key.endswith(f"__w{width}")
            and marker.get("pyramid_mode") == OUTLINE_PYRAMID_MODE
            and value_max == OUTLINE_COVERAGE_MAX
        )
    return False


def _spatialdata_store_path(sdata_obj: Any) -> Path | None:
    value = getattr(sdata_obj, "path", None)
    if value is None:
        value = getattr(sdata_obj, "_path", None)
    if value is None:
        return None
    try:
        return Path(value)
    except TypeError:
        return None


def _zarr_element_marker(
    sdata_obj: Any,
    *,
    group: str,
    key: str,
    attr_name: str,
) -> dict[str, Any]:
    store_path = _spatialdata_store_path(sdata_obj)
    if store_path is None:
        return {}
    try:
        import zarr

        zarr_group = zarr.open_group(
            str(store_path / str(group) / str(key)),
            mode="r",
        )
        value = zarr_group.attrs.get(attr_name, {})
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _element_transform_matrices(element: Any) -> dict[str, np.ndarray] | None:
    try:
        from spatialdata.transformations import get_transformation

        transformations = get_transformation(element, get_all=True)
    except Exception:
        return None
    result: dict[str, np.ndarray] = {}
    for coordinate_system, transformation in transformations.items():
        try:
            matrix = np.asarray(
                transformation.to_affine_matrix(
                    input_axes=("x", "y"),
                    output_axes=("x", "y"),
                ),
                dtype=float,
            )
        except Exception:
            return None
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return None
        result[str(coordinate_system)] = matrix
    return result or None


def _elements_share_spatial_transform(source: Any, cache: Any) -> bool:
    source_matrices = _element_transform_matrices(source)
    cache_matrices = _element_transform_matrices(cache)
    if source_matrices is None or cache_matrices is None:
        return False
    if source_matrices.keys() != cache_matrices.keys():
        return False
    return all(
        np.allclose(source_matrices[key], cache_matrices[key], atol=1e-8)
        for key in source_matrices
    )


def _element_matches_affine(element: Any, expected: np.ndarray) -> bool:
    matrices = _element_transform_matrices(element)
    if matrices is None:
        return False
    return any(
        np.allclose(matrix, expected, atol=1e-8)
        for matrix in matrices.values()
    )


def _schema(sdata_obj: Any) -> dict[str, Any]:
    value = getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR)
    if not isinstance(value, Mapping):
        raise PairedDataContractError("SpatialData store is missing merxen_schema.")
    return dict(value)


def _registry(schema: Mapping[str, Any]) -> dict[str, Any]:
    value = schema.get("segmentations")
    if not isinstance(value, Mapping):
        raise PairedDataContractError(
            "SpatialData merxen_schema has no segmentation registry."
        )
    return dict(value)


def _mapping_attr(sdata_obj: Any, key: str) -> dict[str, Any]:
    value = getattr(sdata_obj, "attrs", {}).get(key)
    if not isinstance(value, Mapping):
        raise PairedDataContractError(f"SpatialData store is missing {key!r}.")
    return dict(value)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedDataContractError(f"{name} must be an object.")
    return value


def _required_text(value: Mapping[str, Any], key: str, name: str) -> str:
    text = str(value.get(key, "")).strip()
    if not text:
        raise PairedDataContractError(f"{name}.{key} is required.")
    return text


def _affine_tuple(value: Any) -> tuple[tuple[float, float, float], ...]:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PairedDataContractError(
            "Alignment manifest fixed-grid affine is invalid."
        ) from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise PairedDataContractError(
            "Alignment manifest fixed-grid affine must be a finite 3x3 matrix."
        )
    if abs(float(np.linalg.det(matrix))) < 1e-15:
        raise PairedDataContractError(
            "Alignment manifest fixed-grid affine is singular."
        )
    return tuple(tuple(float(item) for item in row) for row in matrix)
