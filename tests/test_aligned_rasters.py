"""Placement of MerXen's aligned rasters in the standalone MERSCOPE view.

MerXen resamples a MERSCOPE section onto its paired Xenium pixel grid and stores
the results beside the native data as ``*_aligned_nonrigid`` images and labels.
They carry a single transformation, to the ``merxen_xenium`` frame, with the
Xenium pixel size (0.2125 um/px), and no ``global`` entry. The dataset's
MERSCOPE pixel->micron transform (about 0.108 um/px) does not apply to them:
placed with it, an aligned raster is drawn at about half size.
"""

from __future__ import annotations

import types
from pathlib import Path
from threading import Event

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import zarr
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, get_transformation

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import (
    MERXEN_ALIGNMENT_COORDINATE_SYSTEM,
    aligned_raster_transformation,
    build_napari_affine_from_px_to_um,
    derived_image_pyramid_cache_key,
    derived_label_pyramid_cache_key,
    ensure_cyx,
    get_scale0_dataarray,
    napari_affine_from_transformation,
    raster_display_transformation,
)

XENIUM_UM_PER_PX = 0.2125
NATIVE_IMAGE = "MERSCOPE_z_projection"
ALIGNED_IMAGE = "MERSCOPE_z_projection_aligned_nonrigid"
NATIVE_LABELS = "MOSAIK_proseg_labels"
ALIGNED_SHAPES = "MOSAIK_proseg_aligned_nonrigid"
ALIGNED_LABELS = "MOSAIK_proseg_aligned_nonrigid_labels"
VZG2_MICRON_TO_MOSAIC = np.array(
    [
        [9.259471893310547, 0.0, 202.43661499023438],
        [0.0, 9.259332656860352, 1034.9761962890625],
        [0.0, 0.0, 1.0],
    ]
)
# What MerXen's alignment writes on an aligned raster: Xenium pixel centres.
ALIGNED_PIXEL_TO_UM = np.array(
    [
        [XENIUM_UM_PER_PX, 0.0, XENIUM_UM_PER_PX / 2],
        [0.0, XENIUM_UM_PER_PX, XENIUM_UM_PER_PX / 2],
        [0.0, 0.0, 1.0],
    ]
)
# The same map as a napari (row, col) -> (y, x) affine.
ALIGNED_NAPARI_AFFINE = np.array(
    [
        [XENIUM_UM_PER_PX, 0.0, XENIUM_UM_PER_PX / 2],
        [0.0, XENIUM_UM_PER_PX, XENIUM_UM_PER_PX / 2],
        [0.0, 0.0, 1.0],
    ]
)


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
        image_pyramid_downsample=4,
        label_chunk_size=1024,
    )
    defaults.update(args)
    return V.ComparisonViewerController(_Viewer(), {}, types.SimpleNamespace(**defaults))


def _affine(matrix: np.ndarray) -> Affine:
    return Affine(np.asarray(matrix, dtype=float), input_axes=("x", "y"), output_axes=("x", "y"))


def _polygon_shapes(transformations=None):
    return ShapesModel.parse(
        gpd.GeoDataFrame(
            {"geometry": [Polygon([(-21.0, -111.0), (-20.0, -111.0), (-20.0, -110.0)])]},
            geometry="geometry",
        ),
        transformations=transformations,
    )


def _write_store(
    root: Path,
    *,
    native_size: int = 16,
    aligned_size: int = 8,
    aligned_image_key: str = ALIGNED_IMAGE,
) -> Path:
    """Write a MERSCOPE store holding native and MerXen-aligned rasters."""
    store = root / "merscope" / "latest" / "latest_spatialdata.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    aligned_tf = {MERXEN_ALIGNMENT_COORDINATE_SYSTEM: _affine(ALIGNED_PIXEL_TO_UM)}
    native_label_tf = _affine(
        np.array([[0.108, 0.0, 0.054], [0.0, 0.108, 0.054], [0.0, 0.0, 1.0]])
    )
    images = {
        NATIVE_IMAGE: Image2DModel.parse(
            np.ones((2, native_size, native_size), dtype=np.uint16),
            dims=("c", "y", "x"),
            c_coords=["DAPI", "PolyT"],
        ),
        aligned_image_key: Image2DModel.parse(
            np.ones((2, aligned_size, aligned_size), dtype=np.uint16),
            dims=("c", "y", "x"),
            c_coords=["DAPI", "PolyT"],
            transformations=aligned_tf,
        ),
    }
    labels = {
        NATIVE_LABELS: Labels2DModel.parse(
            np.zeros((native_size, native_size), dtype=np.uint32),
            dims=("y", "x"),
            transformations={"global": native_label_tf},
        ),
        ALIGNED_LABELS: Labels2DModel.parse(
            np.zeros((aligned_size, aligned_size), dtype=np.uint32),
            dims=("y", "x"),
            transformations=aligned_tf,
        ),
    }
    points = PointsModel.parse(
        pd.DataFrame({"x": [-20.0], "y": [-110.0], "gene": ["GeneA"]}),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
    )
    sd.SpatialData(
        images=images,
        labels=labels,
        points={"transcripts": points},
        shapes={
            "MOSAIK_proseg": _polygon_shapes(),
            ALIGNED_SHAPES: _polygon_shapes({MERXEN_ALIGNMENT_COORDINATE_SYSTEM: Identity()}),
            "merscope_cell_boundaries": _polygon_shapes(
                {"global": _affine(VZG2_MICRON_TO_MOSAIC)}
            ),
        },
    ).write(store)
    # MerXen stamps the viewer's label-cache marker on the rasters it materialises.
    group = zarr.open_group(str(store / "labels" / ALIGNED_LABELS), mode="a")
    group.attrs[V.LABEL_CACHE_ATTR] = {
        "version": 2,
        "complete": True,
        "source_shape_key": ALIGNED_SHAPES,
        "shape": [aligned_size, aligned_size],
        "chunks": [aligned_size, aligned_size],
    }
    return store


def _active_controller(store: Path):
    ctrl = _controller()
    cfg = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    ctrl.datasets = {"MERSCOPE": cfg}
    ctrl.active_dataset = "MERSCOPE"
    ctrl._active_sdata = sd.read_zarr(store)
    ctrl._active_images_sdata = sd.read_zarr(store, selection=("images",))
    ctrl._x_transform, ctrl._y_transform = ctrl._resolve_dataset_pixel_transform("MERSCOPE", cfg)
    return ctrl


def _session_affine(ctrl, element) -> np.ndarray:
    raster = ensure_cyx(get_scale0_dataarray(element)) if "c" in element.dims else get_scale0_dataarray(element)
    x_coords = np.asarray(raster.coords["x"].values) if "x" in raster.coords else None
    y_coords = np.asarray(raster.coords["y"].values) if "y" in raster.coords else None
    return build_napari_affine_from_px_to_um(
        ctrl._x_transform, ctrl._y_transform, x_coords=x_coords, y_coords=y_coords
    )


def _base_shape(data) -> tuple[int, ...]:
    base = data[0] if isinstance(data, list) else data
    return tuple(int(axis) for axis in np.shape(base))


def test_aligned_image_is_placed_on_the_xenium_grid(tmp_path):
    store = _write_store(tmp_path)
    ctrl = _controller()
    cfg = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    session = ctrl._prepare_dataset_session("MERSCOPE", cfg, Event())
    ctrl._x_transform, ctrl._y_transform = session.x_transform, session.y_transform

    stats = ctrl._add_image_layers(
        "MERSCOPE",
        session.images_sdata,
        session.x_transform,
        session.y_transform,
        build_cache=False,
    )

    assert stats["layers"] == 4
    by_shape: dict[tuple[int, ...], list[np.ndarray]] = {}
    for added in ctrl.viewer.added_images:
        by_shape.setdefault(_base_shape(added["data"]), []).append(np.asarray(added["affine"]))
    aligned = by_shape[(8, 8)]
    native = by_shape[(16, 16)]
    assert len(aligned) == len(native) == 2
    for affine in aligned:
        np.testing.assert_allclose(affine, ALIGNED_NAPARI_AFFINE, rtol=0, atol=1e-12)
    expected_native = _session_affine(ctrl, session.images_sdata.images[NATIVE_IMAGE])
    for affine in native:
        np.testing.assert_allclose(affine, expected_native, rtol=0, atol=1e-12)
    # The native image keeps the ~0.108 um/px MERSCOPE pixel size.
    assert abs(native[0][0, 0] - 1.0 / 9.259332656860352) < 1e-9


def test_aligned_labels_are_placed_by_their_own_transform_and_kept(tmp_path):
    store = _write_store(tmp_path)
    ctrl = _active_controller(store)
    aligned = ctrl._active_sdata.labels[ALIGNED_LABELS]
    native = ctrl._active_sdata.labels[NATIVE_LABELS]

    np.testing.assert_allclose(
        ctrl._label_display_affine(aligned), ALIGNED_NAPARI_AFFINE, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        ctrl._label_display_affine(native), _session_affine(ctrl, native), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        ctrl._napari_affine_from_element(aligned), ALIGNED_NAPARI_AFFINE, rtol=0, atol=1e-12
    )
    # The MerXen raster is not judged stale and rebuilt on the MERSCOPE grid.
    assert ctrl._label_cache_matches_pixel_transform(ALIGNED_LABELS)
    assert ctrl._label_cache_is_complete(ALIGNED_LABELS, ALIGNED_SHAPES)


def test_label_pyramid_cache_of_an_aligned_raster_stays_in_its_frame(tmp_path):
    store = _write_store(tmp_path, aligned_size=1100)
    ctrl = _active_controller(store)

    levels = ctrl._ensure_label_pyramid_cache(ALIGNED_LABELS, 4)

    assert levels
    cache = sd.read_zarr(store, selection=("labels",)).labels[
        derived_label_pyramid_cache_key(ALIGNED_LABELS, 4)
    ]
    transformations = get_transformation(cache, get_all=True)
    assert set(transformations) == {MERXEN_ALIGNMENT_COORDINATE_SYSTEM}
    np.testing.assert_allclose(
        napari_affine_from_transformation(transformations[MERXEN_ALIGNMENT_COORDINATE_SYSTEM]),
        ALIGNED_NAPARI_AFFINE,
        rtol=0,
        atol=1e-12,
    )


def test_image_pyramid_cache_of_an_aligned_image_stays_in_its_frame(tmp_path):
    store = _write_store(tmp_path, aligned_size=1100)
    ctrl = _active_controller(store)
    image = ctrl._active_sdata.images[ALIGNED_IMAGE]

    levels = ctrl._ensure_image_pyramid_cache(ALIGNED_IMAGE, image)

    assert levels
    cache = sd.read_zarr(store, selection=("images",)).images[
        derived_image_pyramid_cache_key(ALIGNED_IMAGE, 4)
    ]
    assert set(get_transformation(cache, get_all=True)) == {MERXEN_ALIGNMENT_COORDINATE_SYSTEM}


def test_label_grid_is_sized_off_a_native_image(tmp_path):
    # An aligned image listed first must not size (or place) rasterised labels.
    store = _write_store(tmp_path, aligned_image_key="A_aligned_nonrigid")
    ctrl = _active_controller(store)
    assert next(iter(ctrl._active_images_sdata.images)) == "A_aligned_nonrigid"

    shape, _chunks, napari_affine, _spatialdata_affine = ctrl._image_grid_for_labels()

    assert shape == (16, 16)
    np.testing.assert_allclose(
        napari_affine,
        _session_affine(ctrl, ctrl._active_images_sdata.images[NATIVE_IMAGE]),
        rtol=0,
        atol=1e-12,
    )


def test_raster_frame_helpers():
    native = Labels2DModel.parse(np.zeros((4, 4), dtype=np.uint32), dims=("y", "x"))
    both = Labels2DModel.parse(
        np.zeros((4, 4), dtype=np.uint32),
        dims=("y", "x"),
        transformations={
            "global": Identity(),
            MERXEN_ALIGNMENT_COORDINATE_SYSTEM: _affine(ALIGNED_PIXEL_TO_UM),
        },
    )
    aligned = Labels2DModel.parse(
        np.zeros((4, 4), dtype=np.uint32),
        dims=("y", "x"),
        transformations={MERXEN_ALIGNMENT_COORDINATE_SYSTEM: _affine(ALIGNED_PIXEL_TO_UM)},
    )
    other = Labels2DModel.parse(
        np.zeros((4, 4), dtype=np.uint32),
        dims=("y", "x"),
        transformations={"elsewhere": Identity()},
    )

    assert aligned_raster_transformation(native) is None
    assert aligned_raster_transformation(both) is None
    assert aligned_raster_transformation(aligned) is not None
    assert raster_display_transformation(both)[0] == "global"
    assert raster_display_transformation(aligned)[0] == MERXEN_ALIGNMENT_COORDINATE_SYSTEM
    try:
        raster_display_transformation(other)
    except ValueError as exc:
        assert "elsewhere" in str(exc)
    else:
        raise AssertionError("a raster outside both frames must be refused")
