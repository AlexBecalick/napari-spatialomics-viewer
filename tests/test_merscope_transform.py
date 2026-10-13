"""MERSCOPE pixel->micron placement when the store carries no transform CSV.

MerXen's published ``latest_spatialdata.zarr`` stores do not ship
``micron_to_mosaic_pixel_transform.csv``, but the SpatialData MERSCOPE reader
leaves the full micron->mosaic-pixel affine on ``shapes/merscope_cell_boundaries``
(its ``global`` transformation). Images must be placed with the inverse of that
matrix, translation included, or they land away from the micron-space masks and
transcripts whenever the mosaic origin is not (0, 0).
"""

from __future__ import annotations

import logging
import types
from pathlib import Path
from threading import Event

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import zarr
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import (
    MERSCOPE_TRANSFORM_CSV_NAME,
    build_napari_affine_from_px_to_um,
    ensure_cyx,
    get_scale0_dataarray,
    read_merscope_micron_to_mosaic_from_zarr,
    resolve_dataset_mask_affine,
    validate_micron_to_mosaic_matrix,
)

# Vizgen mouse reference (VZG2) micron->mosaic-pixel transform: mosaic pixel 0
# sits at (-21.86, -111.78) um.
VZG2_MICRON_TO_MOSAIC = np.array(
    [
        [9.259471893310547, 0.0, 202.43661499023438],
        [0.0, 9.259332656860352, 1034.9761962890625],
        [0.0, 0.0, 1.0],
    ]
)
VZG2_PIXEL0_UM_XY = (-202.43661499023438 / 9.259471893310547, -1034.9761962890625 / 9.259332656860352)


class _Viewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []
        self.added_images: list[dict] = []

    def add_image(self, data, **kwargs):
        self.added_images.append({"data": data, **kwargs})
        return types.SimpleNamespace(name=kwargs.get("name"))


def _controller(**args):
    defaults = dict(
        background_io_workers=1,
        session_cache_gb=0,
        segmentation_source="shapes",
        hide_images=False,
        image_opacity=1.0,
        visible_channels=None,
        overwrite_labels=False,
        overwrite_derived_caches=False,
    )
    defaults.update(args)
    return V.ComparisonViewerController(_Viewer(), {}, types.SimpleNamespace(**defaults))


def _affine(matrix: np.ndarray) -> Affine:
    return Affine(np.asarray(matrix, dtype=float), input_axes=("x", "y"), output_axes=("x", "y"))


def _translated(tx: float, ty: float, scale: float = 9.2594) -> np.ndarray:
    return np.array([[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]])


def _write_merscope_store(
    root: Path,
    cell_boundaries_transform=None,
    *,
    with_cell_boundaries: bool = True,
) -> Path:
    """Write a tiny MerXen-like MERSCOPE store under ``root``."""
    store = root / "merscope" / "latest" / "latest_spatialdata.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    image = Image2DModel.parse(
        np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16),
        dims=("c", "y", "x"),
        c_coords=["DAPI", "PolyT"],
    )
    points = PointsModel.parse(
        pd.DataFrame(
            {"x": [-20.0, -19.0], "y": [-110.0, -109.0], "gene": ["GeneA", "GeneB"], "assignment": [1, 0]}
        ),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
    )
    proseg = ShapesModel.parse(
        gpd.GeoDataFrame(
            {"geometry": [Polygon([(-21.0, -111.0), (-20.0, -111.0), (-20.0, -110.0)])]},
            geometry="geometry",
        )
    )
    shapes = {"MOSAIK_proseg": proseg}
    if with_cell_boundaries:
        transform = (
            _affine(VZG2_MICRON_TO_MOSAIC) if cell_boundaries_transform is None else cell_boundaries_transform
        )
        shapes["merscope_cell_boundaries"] = ShapesModel.parse(
            gpd.GeoDataFrame(
                {"geometry": [Polygon([(-21.0, -111.0), (-20.0, -111.0), (-20.0, -110.0)])]},
                geometry="geometry",
            ),
            transformations={"global": transform},
        )
    sd.SpatialData(
        images={"MERSCOPE_z_projection": image},
        points={"transcripts": points},
        shapes=shapes,
    ).write(store)
    return store


def _write_csv(path: Path, matrix: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float))
    return path


def _resolve(ctrl, cfg: V.DatasetConfig) -> np.ndarray:
    x_tf, y_tf = ctrl._resolve_dataset_pixel_transform("MERSCOPE", cfg)
    return build_napari_affine_from_px_to_um(x_tf, y_tf)


def _pixel0_um_xy(napari_affine: np.ndarray) -> tuple[float, float]:
    y_um, x_um, _ = napari_affine @ np.array([0.0, 0.0, 1.0])
    return float(x_um), float(y_um)


def test_merscope_image_uses_zarr_cell_boundary_transform_without_csv(tmp_path):
    store = _write_merscope_store(tmp_path / "data")
    csv = _write_csv(tmp_path / "reference" / MERSCOPE_TRANSFORM_CSV_NAME, VZG2_MICRON_TO_MOSAIC)
    csv_x, csv_y = resolve_dataset_mask_affine("MERSCOPE", merscope_transform_path=csv)

    ctrl = _controller()
    cfg = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    session = ctrl._prepare_dataset_session("MERSCOPE", cfg, Event())

    np.testing.assert_allclose(session.x_transform, csv_x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(session.y_transform, csv_y, rtol=0, atol=1e-9)

    stats = ctrl._add_image_layers(
        "MERSCOPE",
        session.images_sdata,
        session.x_transform,
        session.y_transform,
        build_cache=False,
    )
    assert stats["layers"] == 2
    image = ensure_cyx(get_scale0_dataarray(session.images_sdata.images["MERSCOPE_z_projection"]))
    x_coords = np.asarray(image.coords["x"].values) if "x" in image.coords else None
    y_coords = np.asarray(image.coords["y"].values) if "y" in image.coords else None
    expected = build_napari_affine_from_px_to_um(csv_x, csv_y, x_coords=x_coords, y_coords=y_coords)
    # Array index of mosaic pixel coordinate 0 (SpatialData labels pixel centres).
    x0 = 0.0 if x_coords is None else -float(x_coords[0]) / float(x_coords[1] - x_coords[0])
    y0 = 0.0 if y_coords is None else -float(y_coords[0]) / float(y_coords[1] - y_coords[0])
    for added in ctrl.viewer.added_images:
        np.testing.assert_allclose(added["affine"], expected, rtol=0, atol=1e-9)
        # Mosaic pixel 0 lands at the dataset's micron origin, not at (0, 0).
        y_um, x_um, _ = np.asarray(added["affine"]) @ np.array([y0, x0, 1.0])
        np.testing.assert_allclose((x_um, y_um), VZG2_PIXEL0_UM_XY, atol=1e-6)


def test_read_merscope_transform_from_zarr_metadata_matches_csv(tmp_path):
    store = _write_merscope_store(tmp_path)
    matrix = read_merscope_micron_to_mosaic_from_zarr(store)
    np.testing.assert_allclose(matrix, VZG2_MICRON_TO_MOSAIC, rtol=0, atol=1e-9)


def test_read_merscope_transform_falls_back_to_consolidated_metadata(tmp_path):
    store = _write_merscope_store(tmp_path)
    (store / "shapes" / "merscope_cell_boundaries" / "zarr.json").unlink()
    matrix = read_merscope_micron_to_mosaic_from_zarr(store)
    np.testing.assert_allclose(matrix, VZG2_MICRON_TO_MOSAIC, rtol=0, atol=1e-9)


def test_merscope_transform_precedence(tmp_path):
    zarr_matrix = _translated(202.0, 1035.0)
    store = _write_merscope_store(tmp_path / "data", _affine(zarr_matrix))
    explicit = _write_csv(tmp_path / "explicit" / MERSCOPE_TRANSFORM_CSV_NAME, _translated(10.0, 20.0))
    inside = _write_csv(store / MERSCOPE_TRANSFORM_CSV_NAME, _translated(30.0, 40.0))
    parent = _write_csv(store.parent / MERSCOPE_TRANSFORM_CSV_NAME, _translated(50.0, 60.0))
    ctrl = _controller()

    def pixel0(cfg):
        return _pixel0_um_xy(_resolve(ctrl, cfg))

    def origin(matrix):
        inv = np.linalg.inv(matrix)
        return float(inv[0, 2]), float(inv[1, 2])

    base = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    with_explicit = V.DatasetConfig(name="MERSCOPE", zarr_path=store, merscope_transform_path=explicit)

    np.testing.assert_allclose(pixel0(with_explicit), origin(_translated(10.0, 20.0)))
    np.testing.assert_allclose(pixel0(base), origin(_translated(30.0, 40.0)))
    inside.unlink()
    np.testing.assert_allclose(pixel0(base), origin(_translated(50.0, 60.0)))
    parent.unlink()
    np.testing.assert_allclose(pixel0(base), origin(zarr_matrix))
    # An explicit path that does not exist falls through to the next source.
    missing = V.DatasetConfig(
        name="MERSCOPE", zarr_path=store, merscope_transform_path=tmp_path / "nope.csv"
    )
    np.testing.assert_allclose(pixel0(missing), origin(zarr_matrix))


def test_merscope_fallback_warns_and_names_missing_sources(tmp_path, caplog):
    store = _write_merscope_store(tmp_path, with_cell_boundaries=False)
    ctrl = _controller()
    with caplog.at_level(logging.WARNING):
        x_tf, y_tf = ctrl._resolve_dataset_pixel_transform(
            "MERSCOPE", V.DatasetConfig(name="MERSCOPE", zarr_path=store)
        )
    assert x_tf == (0.108, 0.0, 0.0)
    assert y_tf == (0.0, 0.108, 0.0)
    text = caplog.text
    assert "0.108" in text
    assert MERSCOPE_TRANSFORM_CSV_NAME in text
    assert "merscope_cell_boundaries" in text


def test_identity_cell_boundary_transform_is_not_used_as_micron_to_pixel(tmp_path, caplog):
    # A store whose cell boundaries are already in the micron frame carries an
    # identity: using it would place images at 1 um/px.
    store = _write_merscope_store(tmp_path, Identity())
    with caplog.at_level(logging.WARNING):
        x_tf, y_tf = resolve_dataset_mask_affine("MERSCOPE", zarr_path=store)
    assert x_tf == (0.108, 0.0, 0.0)
    assert y_tf == (0.0, 0.108, 0.0)
    assert "merscope_cell_boundaries" in caplog.text


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.array([[9.26, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        np.array([[np.nan, 0.0, 1.0], [0.0, 9.26, 2.0], [0.0, 0.0, 1.0]]),
        np.array([[9.26, 0.0, 1.0], [0.0, 9.26, 2.0], [0.1, 0.0, 1.0]]),
        np.array([[9.26, 0.0, 1.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]),
        np.array([[-9.26, 0.0, 1.0], [0.0, 9.26, 2.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["identity", "singular", "non-finite", "projective", "anisotropic", "mirrored"],
)
def test_validate_micron_to_mosaic_matrix_rejects_implausible(matrix):
    with pytest.raises(ValueError):
        validate_micron_to_mosaic_matrix(matrix, source="test")


def test_validate_micron_to_mosaic_matrix_accepts_small_rotation(tmp_path):
    theta = np.deg2rad(0.5)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = np.eye(3)
    matrix[:2, :2] = 9.2594 * rotation
    matrix[:2, 2] = (12.0, -34.0)
    np.testing.assert_allclose(validate_micron_to_mosaic_matrix(matrix, source="test"), matrix)
    x_tf, y_tf = resolve_dataset_mask_affine(
        "MERSCOPE",
        merscope_transform_path=_write_csv(tmp_path / MERSCOPE_TRANSFORM_CSV_NAME, matrix),
    )
    inv = np.linalg.inv(matrix)
    np.testing.assert_allclose(x_tf, inv[0], atol=1e-12)
    np.testing.assert_allclose(y_tf, inv[1], atol=1e-12)


def _mark_label_cache(store: Path, label_key: str, shape_key: str, shape) -> None:
    group = zarr.open_group(str(store / "labels" / label_key), mode="a")
    group.attrs[V.LABEL_CACHE_ATTR] = {
        "version": 2,
        "complete": True,
        "source_shape_key": shape_key,
        "shape": [int(shape[0]), int(shape[1])],
        "chunks": [int(shape[0]), int(shape[1])],
    }


def _label_cache_controller(store: Path, stored_napari_affine: np.ndarray):
    """Controller whose store holds a viewer-built label cache with a given transform."""
    m = stored_napari_affine
    stored = Affine(
        [[m[1, 1], m[1, 0], m[1, 2]], [m[0, 1], m[0, 0], m[0, 2]], [0.0, 0.0, 1.0]],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    labels = Labels2DModel.parse(
        np.zeros((16, 16), dtype=np.uint32), dims=("y", "x"), transformations={"global": stored}
    )
    sdata = sd.read_zarr(store)
    sdata.labels["MOSAIK_proseg_labels"] = labels
    sdata.write_element("MOSAIK_proseg_labels")
    _mark_label_cache(store, "MOSAIK_proseg_labels", "MOSAIK_proseg", (16, 16))

    ctrl = _controller()
    cfg = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    ctrl.datasets = {"MERSCOPE": cfg}
    ctrl.active_dataset = "MERSCOPE"
    ctrl._active_sdata = sd.read_zarr(store)
    ctrl._x_transform, ctrl._y_transform = ctrl._resolve_dataset_pixel_transform("MERSCOPE", cfg)
    return ctrl


def test_label_cache_rasterized_with_old_fallback_transform_is_stale(tmp_path):
    store = _write_merscope_store(tmp_path)
    old_fallback = build_napari_affine_from_px_to_um((0.108, 0.0, 0.0), (0.0, 0.108, 0.0))
    ctrl = _label_cache_controller(store, old_fallback)

    assert not ctrl._label_cache_is_complete("MOSAIK_proseg_labels", "MOSAIK_proseg")


def test_label_cache_rasterized_with_resolved_transform_is_reused(tmp_path):
    store = _write_merscope_store(tmp_path)
    csv = _write_csv(tmp_path / "ref" / MERSCOPE_TRANSFORM_CSV_NAME, VZG2_MICRON_TO_MOSAIC)
    x_tf, y_tf = resolve_dataset_mask_affine("MERSCOPE", merscope_transform_path=csv)
    ctrl = _label_cache_controller(store, build_napari_affine_from_px_to_um(x_tf, y_tf))

    assert ctrl._label_cache_is_complete("MOSAIK_proseg_labels", "MOSAIK_proseg")


def test_derived_label_cache_is_stale_when_its_source_transform_changed(tmp_path):
    store = _write_merscope_store(tmp_path)
    old_fallback = build_napari_affine_from_px_to_um((0.108, 0.0, 0.0), (0.0, 0.108, 0.0))
    ctrl = _label_cache_controller(store, old_fallback)
    sdata = ctrl._active_sdata
    source_tf = sdata.labels["MOSAIK_proseg_labels"].attrs["transform"]["global"]
    sdata.labels["outline_same"] = Labels2DModel.parse(
        np.zeros((16, 16), dtype=np.uint8), dims=("y", "x"), transformations={"global": source_tf}
    )
    sdata.labels["outline_old"] = Labels2DModel.parse(
        np.zeros((16, 16), dtype=np.uint8),
        dims=("y", "x"),
        transformations={"global": _affine(np.linalg.inv(VZG2_MICRON_TO_MOSAIC))},
    )

    assert ctrl._derived_label_cache_matches_source("outline_same", "MOSAIK_proseg_labels", sdata)
    assert not ctrl._derived_label_cache_matches_source("outline_old", "MOSAIK_proseg_labels", sdata)


def test_xenium_label_caches_are_not_checked_against_the_merscope_transform(tmp_path):
    store = _write_merscope_store(tmp_path)
    old_fallback = build_napari_affine_from_px_to_um((0.108, 0.0, 0.0), (0.0, 0.108, 0.0))
    ctrl = _label_cache_controller(store, old_fallback)
    ctrl.datasets = {"XENIUM": ctrl.datasets["MERSCOPE"]}
    ctrl.active_dataset = "XENIUM"

    assert ctrl._label_cache_is_complete("MOSAIK_proseg_labels", "MOSAIK_proseg")
