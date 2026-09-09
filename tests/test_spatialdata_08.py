"""Integration coverage for stores written by the supported SpatialData release."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from packaging.version import Version
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import image_scale_dataarrays


def test_viewer_reads_spatialdata_080_zarr_v3_store(tmp_path):
    """Exercise the same selective reads used by dataset startup and image loading."""
    assert Version(sd.__version__) >= Version("0.8.0")

    store = tmp_path / "spatialdata.zarr"
    image = Image2DModel.parse(
        np.arange(32, dtype=np.uint16).reshape(2, 4, 4),
        dims=("c", "y", "x"),
        c_coords=["DAPI", "PolyT"],
    )
    labels = Labels2DModel.parse(
        np.asarray([[0, 1], [2, 2]], dtype=np.uint16),
        dims=("y", "x"),
    )
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [2.0, 3.0],
                "feature_name": ["GeneA", "GeneB"],
                "assignment": [1, 0],
            }
        ),
        coordinates={"x": "x", "y": "y"},
        feature_key="feature_name",
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {"geometry": [Polygon([(0, 0), (1, 0), (1, 1)])]},
            geometry="geometry",
        )
    )
    sd.SpatialData(
        images={"mosaic": image},
        labels={"cell_labels": labels},
        points={"transcripts": points},
        shapes={"cell_shapes": shapes},
    ).write(store)

    metadata = json.loads((store / "zarr.json").read_text())
    assert metadata["zarr_format"] == 3
    assert Version(
        metadata["attributes"]["spatialdata_attrs"]["spatialdata_software_version"]
    ) >= Version("0.8.0")
    V.validate_spatialdata_store_compatibility(store)

    shapes_startup = sd.read_zarr(store, selection=V.startup_selection("shapes"))
    labels_startup = sd.read_zarr(store, selection=V.startup_selection("labels"))
    images = sd.read_zarr(store, selection=("images",))

    assert list(shapes_startup.points) == ["transcripts"]
    assert list(shapes_startup.shapes) == ["cell_shapes"]
    assert list(labels_startup.labels) == ["cell_labels"]
    assert [name for name, _array in image_scale_dataarrays(images.images["mosaic"])] == [
        "scale0"
    ]
    assert V.ComparisonViewerController._enumerate_image_channels_for(images) == [
        ("mosaic", "DAPI"),
        ("mosaic", "PolyT"),
    ]
