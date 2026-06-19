from __future__ import annotations

import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from napari_compare_xenium_merscope.utils import (
    build_transcript_spatial_index,
    compute_transcript_density_array,
    derived_outline_cache_key,
    derived_transcript_density_cache_key,
    geometry_to_napari_polygons,
    geometry_to_napari_bounding_boxes,
    geometry_to_napari_centroids,
    image_scale_dataarrays,
    ensure_cyx,
    is_derived_cache_key,
    label_outline_mask_chunk,
    layer_name_prefix,
    load_viewport_points_dataframe,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    pick_default_shape_key,
    query_geometries_for_bounds,
    query_transcript_spatial_index,
    rasterize_geometries_chunk,
)


def _wavy_polygon(n_points: int = 256) -> Polygon:
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_points, endpoint=False)
    radius = 10.0 + 1.25 * np.sin(9.0 * theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return Polygon(np.column_stack([x, y]))


def test_pick_default_shape_key_priority():
    keys = ["foo", "MOSAIK_proseg", "cell_boundaries"]
    assert pick_default_shape_key(keys) == "MOSAIK_proseg"

    keys = ["z_key", "cell_boundaries", "a_key"]
    assert pick_default_shape_key(keys) == "cell_boundaries"

    keys = ["z_key", "b_key", "a_key"]
    assert pick_default_shape_key(keys) == "a_key"

    assert pick_default_shape_key([]) is None


def test_layer_name_generation_and_matching():
    ds = "merscope"
    base = make_layer_name(ds, "shapes", "MOSAIK_proseg")
    with_suffix = f"{base} | duplicate"
    unrelated = make_layer_name(ds, "shapes", "cell_boundaries")
    other_dataset = make_layer_name("xenium", "shapes", "MOSAIK_proseg")
    names = [base, with_suffix, unrelated, other_dataset]

    prefix = layer_name_prefix(ds, "shapes", "MOSAIK_proseg")
    matches = matching_layer_names(names, prefix)
    assert matches == [base, with_suffix]


def test_derived_cache_key_naming_and_filtering():
    outline_key = derived_outline_cache_key("cell_boundaries", 2)
    density_key = derived_transcript_density_cache_key("transcripts", 4.0)
    weird_key = derived_outline_cache_key("foo/bar baz", 1)

    assert outline_key == "_napari_compare_outline__cell_boundaries__w2"
    assert density_key == "_napari_compare_tx_density__transcripts__bin4"
    assert is_derived_cache_key(outline_key)
    assert is_derived_cache_key(density_key)
    assert "/" not in weird_key
    assert not is_derived_cache_key("cell_boundaries")


def test_geometry_to_napari_polygons_cap_and_simplify():
    p1 = _wavy_polygon(512)
    p2 = _wavy_polygon(384)
    p3 = _wavy_polygon(320)
    mp = MultiPolygon([p2, p3])

    all_polys = geometry_to_napari_polygons([p1, mp], max_shapes=None, simplify_tolerance=None)
    capped_polys = geometry_to_napari_polygons([p1, mp], max_shapes=2, simplify_tolerance=None)
    assert len(all_polys) == 3
    assert len(capped_polys) == 2

    raw = geometry_to_napari_polygons([p1], max_shapes=None, simplify_tolerance=None)[0]
    simplified = geometry_to_napari_polygons([p1], max_shapes=None, simplify_tolerance=2.0)[0]
    assert simplified.shape[0] < raw.shape[0]


def test_geometry_to_napari_polygons_vertex_limit():
    poly = _wavy_polygon(512)

    limited = geometry_to_napari_polygons(
        [poly],
        max_shapes=None,
        simplify_tolerance=None,
        max_vertices_per_polygon=32,
    )[0]

    assert limited.shape[0] <= 32


def test_geometry_to_lightweight_shape_representations():
    poly = Polygon([(1.0, 2.0), (5.0, 2.0), (5.0, 7.0), (1.0, 7.0)])

    boxes = geometry_to_napari_bounding_boxes([poly])
    centroids = geometry_to_napari_centroids([poly])

    assert boxes.shape == (1, 4, 2)
    assert centroids.shape == (1, 2)
    assert np.allclose(boxes[0], [[2.0, 1.0], [2.0, 5.0], [7.0, 5.0], [7.0, 1.0]])
    assert np.allclose(centroids[0], [4.5, 3.0])


def test_pixel_window_global_bounds_and_query():
    import geopandas as gpd

    affine = np.array(
        [
            [2.0, 0.0, 10.0],
            [0.0, 3.0, 20.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bounds = pixel_window_global_bounds(affine, y0=0, y1=5, x0=0, x1=10)

    assert bounds == (20.0, 10.0, 50.0, 20.0)

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(21, 11), (22, 11), (22, 12), (21, 12)]),
            Polygon([(100, 100), (101, 100), (101, 101), (100, 101)]),
        ]
    )

    matches = query_geometries_for_bounds(gdf, bounds)

    assert len(matches) == 1


def test_rasterize_geometries_chunk():
    affine = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    poly = Polygon([(2.5, 2.5), (6.5, 2.5), (6.5, 6.5), (2.5, 6.5)])

    tile = rasterize_geometries_chunk([poly], [7], (10, 10), np.linalg.inv(affine))

    assert tile.dtype == np.uint32
    assert tile.max() == 7
    assert np.count_nonzero(tile) == 25


def test_label_outline_mask_chunk_marks_label_boundaries():
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 1

    outline = label_outline_mask_chunk(labels)

    assert outline.dtype == np.uint8
    assert np.count_nonzero(outline) == 12
    assert np.all(outline[3:5, 3:5] == 0)
    assert outline[2, 2] == 1
    assert outline[5, 5] == 1


def test_load_viewport_points_dataframe_filters_and_caps_pandas():
    import pandas as pd

    points = pd.DataFrame(
        {
            "x": np.arange(100, dtype=float),
            "y": np.arange(100, dtype=float),
            "assignment": np.arange(100),
        }
    )

    sampled, total, fraction = load_viewport_points_dataframe(
        points,
        x_col="x",
        y_col="y",
        assignment_col="assignment",
        bounds=(10.0, 10.0, 39.0, 39.0),
        sample_percent=None,
        max_points=7,
        random_state=1,
    )

    assert total == 30
    assert len(sampled) == 7
    assert np.isclose(fraction, 7 / 30)
    assert sampled["x"].between(10.0, 39.0).all()
    assert sampled["y"].between(10.0, 39.0).all()


def test_load_viewport_points_dataframe_percent_samples_dask():
    import dask.dataframe as dd
    import pandas as pd

    points = pd.DataFrame(
        {
            "x": np.arange(200, dtype=float),
            "y": np.arange(200, dtype=float),
            "assignment": np.ones(200, dtype=int),
        }
    )
    ddf = dd.from_pandas(points, npartitions=4)

    sampled, total, fraction = load_viewport_points_dataframe(
        ddf,
        x_col="x",
        y_col="y",
        assignment_col="assignment",
        bounds=(0.0, 0.0, 99.0, 99.0),
        sample_percent=10,
        max_points=50,
        random_state=1,
    )

    assert total == 100
    assert 1 <= len(sampled) <= 50
    assert fraction <= 0.5
    assert sampled["x"].between(0.0, 99.0).all()
    assert sampled["y"].between(0.0, 99.0).all()


def test_compute_transcript_density_array_pandas():
    import pandas as pd

    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 4.0],
            "y": [0.0, 1.0, 0.0],
            "assignment": [1, 0, 2],
        }
    )

    density, meta = compute_transcript_density_array(
        points,
        x_col="x",
        y_col="y",
        assignment_col="assignment",
        bin_um=2.0,
        max_pixels=100,
    )

    assert density.shape == (2, 1, 2)
    assert density.dtype == np.uint32
    assert density[0, 0, 0] == 1
    assert density[1, 0, 0] == 1
    assert density[0, 0, 1] == 1
    assert meta["total"] == 3
    assert meta["assigned"] == 2
    assert meta["unassigned"] == 1


def test_compute_transcript_density_array_dask_adjusts_bin():
    import dask.dataframe as dd
    import pandas as pd

    points = pd.DataFrame(
        {
            "x": np.linspace(0.0, 99.0, 100),
            "y": np.linspace(0.0, 99.0, 100),
            "assignment": np.ones(100, dtype=int),
        }
    )
    ddf = dd.from_pandas(points, npartitions=4)

    density, meta = compute_transcript_density_array(
        ddf,
        x_col="x",
        y_col="y",
        assignment_col="assignment",
        bin_um=1.0,
        max_pixels=25,
    )

    assert density.shape[1] * density.shape[2] <= 25
    assert meta["actual_bin_um"] > 1.0
    assert int(density.sum()) == 100


def test_transcript_spatial_index_query_caps_and_splits_assignment():
    import pandas as pd

    points = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 10.0],
            "y": [0.0, 1.0, 2.0, 3.0, 10.0],
            "assignment": [1, 0, 2, 0, 1],
        }
    )
    index = build_transcript_spatial_index(
        points,
        x_col="x",
        y_col="y",
        assignment_col="assignment",
        max_points=10,
        tile_um=2.0,
        random_state=1,
    )

    result = query_transcript_spatial_index(
        index,
        bounds=(0.0, 0.0, 3.0, 3.0),
        max_points=3,
        sample_percent=None,
        random_state=1,
    )

    assert result["total_in_view"] == 4
    assert result["loaded"] == 3
    assert result["assigned_coords"].shape[1] == 2
    assert result["unassigned_coords"].shape[1] == 2


def test_transcript_spatial_index_query_empty_view():
    import pandas as pd

    points = pd.DataFrame({"x": [0.0], "y": [0.0], "assignment": [1]})
    index = build_transcript_spatial_index(points, "x", "y", "assignment", tile_um=2.0)

    result = query_transcript_spatial_index(
        index,
        bounds=(10.0, 10.0, 11.0, 11.0),
        max_points=10,
    )

    assert result["total_in_view"] == 0
    assert result["loaded"] == 0
    assert result["assigned_coords"].shape == (0, 2)


class _ImageNode:
    def __init__(self, image):
        self.ds = xr.Dataset({"image": image})


def test_image_scale_dataarrays_returns_sorted_pyramid_levels():
    image = {
        "scale10": _ImageNode(xr.DataArray(np.zeros((1, 2, 2)), dims=("c", "y", "x"))),
        "scale2": _ImageNode(xr.DataArray(np.zeros((1, 4, 4)), dims=("c", "y", "x"))),
        "scale0": _ImageNode(xr.DataArray(np.zeros((1, 8, 8)), dims=("c", "y", "x"))),
    }

    scales = image_scale_dataarrays(image)

    assert [name for name, _da in scales] == ["scale0", "scale2", "scale10"]
    assert [da.shape for _name, da in scales] == [(1, 8, 8), (1, 4, 4), (1, 2, 2)]


def test_ensure_cyx_squeezes_singleton_z_plane():
    da = xr.DataArray(np.zeros((2, 1, 4, 8)), dims=("c", "z", "y", "x"))

    cyx = ensure_cyx(da)

    assert cyx.dims == ("c", "y", "x")
    assert cyx.shape == (2, 4, 8)
