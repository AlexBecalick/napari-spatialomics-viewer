from __future__ import annotations

import numpy as np
import xarray as xr
import zarr
from shapely.geometry import MultiPolygon, Polygon

from napari_compare_xenium_merscope.utils import (
    CELLPOSE_LABEL_KEY,
    CELLPOSE_QUANTIFICATION_TABLE_KEY,
    build_cortical_depth_annotation_geojson,
    build_binned_label_color_dict,
    build_transcript_spatial_index,
    cellpose_quantification_features,
    cellpose_quantification_table_available,
    cellpose_value_clip_range,
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
    load_cellpose_quantification_values,
    load_viewport_points_dataframe,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    pick_default_shape_key,
    query_geometries_for_bounds,
    query_transcript_spatial_index,
    rasterize_geometries_chunk,
    resolve_cellpose_quantification_feature,
    snap_cortical_depth_boundaries_to_edge,
    write_cortical_depth_annotation_geojson,
    write_cortical_depth_separate_geojsons,
    _unique_valid_polygons,
)


def test_empty_polygon_annotation_layer_colors_are_napari_compatible():
    from napari.layers import Shapes
    from napari_compare_xenium_merscope.viewer import (
        CORTICAL_DEPTH_FILL_COLORS,
        CORTICAL_DEPTH_LAYER_COLORS,
    )

    layer = Shapes(
        data=[],
        ndim=2,
        shape_type="polygon",
        edge_color=CORTICAL_DEPTH_LAYER_COLORS["exclusion"],
        face_color=CORTICAL_DEPTH_FILL_COLORS["exclusion"],
        name="test cortical depth exclusion",
    )

    assert len(layer.data) == 0


def _create_string_array(group, name: str, values: list[str]):
    max_len = max(1, max(len(value) for value in values))
    group.create_array(name, data=np.asarray(values, dtype=f"<U{max_len}"))


def _make_cellpose_quantification_zarr(tmp_path):
    path = tmp_path / "cellpose_quant.zarr"
    root = zarr.open_group(str(path), mode="w")
    tables = root.create_group("tables")
    labels = root.create_group("labels")
    labels.create_group(CELLPOSE_LABEL_KEY)
    table = tables.create_group(CELLPOSE_QUANTIFICATION_TABLE_KEY)
    obs = table.create_group("obs")
    var = table.create_group("var")

    label_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    obs.create_array("label_id", data=label_ids)
    _create_string_array(obs, "cell_id", [f"cellpose_{label_id}" for label_id in label_ids])

    image_key = ["MERSCOPE_z_projection"] * 10
    channels = ["DAPI"] * 5 + ["PolyT"] * 5
    statistics = ["min", "median", "mean", "max", "iqr"] * 2
    features = [
        f"{image}__{channel}__{statistic}"
        for image, channel, statistic in zip(image_key, channels, statistics, strict=True)
    ]
    _create_string_array(var, "image_key", image_key)
    _create_string_array(var, "channel", channels)
    _create_string_array(var, "statistic", statistics)
    _create_string_array(var, "feature", features)

    x = np.arange(40, dtype=np.float64).reshape(4, 10)
    table.create_array("X", data=x)
    return path


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


def test_cellpose_quantification_feature_lookup_and_value_load(tmp_path):
    path = _make_cellpose_quantification_zarr(tmp_path)

    assert cellpose_quantification_table_available(path)
    features = cellpose_quantification_features(path)
    assert len(features) == 10

    dapi_median = resolve_cellpose_quantification_feature(path, "DAPI", "median")
    polyt_mean = resolve_cellpose_quantification_feature(path, "PolyT", "mean")

    assert dapi_median.column_index == 1
    assert dapi_median.feature == "MERSCOPE_z_projection__DAPI__median"
    assert polyt_mean.column_index == 7
    assert polyt_mean.feature == "MERSCOPE_z_projection__PolyT__mean"

    loaded = load_cellpose_quantification_values(path, "DAPI", "median")
    assert loaded.feature == dapi_median
    assert loaded.label_ids.tolist() == [10, 20, 30, 40]
    assert loaded.values.tolist() == [1.0, 11.0, 21.0, 31.0]


def test_cellpose_value_clip_range_uses_percentiles():
    values = np.arange(100, dtype=float)

    low, high = cellpose_value_clip_range(values, lower_percentile=1, upper_percentile=99)

    assert np.isclose(low, 0.99)
    assert np.isclose(high, 98.01)


def test_binned_label_color_dict_uses_label_ids_and_transparent_defaults():
    label_ids = np.asarray([10, 20, 30, 40], dtype=np.uint32)
    values = np.asarray([0.0, 5.0, np.nan, 10.0], dtype=np.float32)
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    mapping = build_binned_label_color_dict(
        label_ids,
        values,
        colors,
        lower_percentile=0,
        upper_percentile=100,
    )

    assert mapping.color_dict[0] == (0.0, 0.0, 0.0, 0.0)
    assert mapping.color_dict[None] == (0.0, 0.0, 0.0, 0.0)
    assert mapping.color_dict[10] == (1.0, 0.0, 0.0, 1.0)
    assert mapping.color_dict[20] == (0.0, 1.0, 0.0, 1.0)
    assert mapping.color_dict[40] == (1.0, 1.0, 0.0, 1.0)
    assert mapping.color_dict.get(30, mapping.color_dict[None]) == (0.0, 0.0, 0.0, 0.0)
    assert 1 not in mapping.color_dict


def test_binned_label_color_dict_constant_values_are_safe():
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    mapping = build_binned_label_color_dict(
        np.asarray([5, 6], dtype=np.uint32),
        np.asarray([42.0, 42.0], dtype=np.float32),
        colors,
    )

    assert mapping.clip_low == 42.0
    assert mapping.clip_high == 42.0
    assert mapping.color_dict[5] == mapping.color_dict[6]
    assert mapping.color_dict[5] != (0.0, 0.0, 0.0, 0.0)


def test_binned_label_color_dict_keeps_unique_colors_bounded():
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    mapping = build_binned_label_color_dict(
        np.arange(1, 21, dtype=np.uint32),
        np.linspace(0.0, 1.0, 20, dtype=np.float32),
        colors,
        lower_percentile=0,
        upper_percentile=100,
    )

    assert mapping.unique_color_count <= colors.shape[0] + 1

    from napari.utils.colormaps import DirectLabelColormap

    colormap = DirectLabelColormap(color_dict=mapping.color_dict)
    assert colormap._num_unique_colors <= colors.shape[0] + 1


def test_cortical_depth_geojson_export_preserves_spatialdata_xy(tmp_path):
    result = build_cortical_depth_annotation_geojson(
        {
            # Napari layer data are y/x; exported GeoJSON must be x/y.
            "side": [np.asarray([[200.0, 100.0], [250.0, 100.0], [265.0, 200.0], [215.0, 200.0]])],
            "pia": [(np.asarray([[200.0, 100.0], [205.0, 150.0], [215.0, 200.0]]), "piece_a")],
            "wm": [(np.asarray([[250.0, 100.0], [255.0, 150.0], [265.0, 200.0]]), "piece_a")],
            "exclusion": [
                (
                    np.asarray([[225.0, 140.0], [225.0, 160.0], [235.0, 160.0], [235.0, 140.0]]),
                    "piece_a",
                )
            ],
        },
        dataset="XENIUM",
    )

    assert result.errors == ()
    roles = [feature["properties"]["role"] for feature in result.geojson["features"]]
    assert roles == ["side_boundary", "pial_boundary", "gray_white_boundary", "exclusion"]
    assert result.geojson["features"][1]["properties"]["tissue_piece_id"] == "piece_a"
    assert result.geojson["features"][1]["properties"]["piece_mode"] == "depth"
    assert result.geojson["features"][1]["geometry"] == {
        "type": "LineString",
        "coordinates": [[100.0, 200.0], [150.0, 205.0], [200.0, 215.0]],
    }
    assert result.geojson["features"][2]["geometry"]["coordinates"] == [
        [100.0, 250.0],
        [150.0, 255.0],
        [200.0, 265.0],
    ]

    out = tmp_path / "annotations.geojson"
    write_cortical_depth_annotation_geojson(out, result)
    saved = out.read_text()
    assert '"role": "pial_boundary"' in saved
    assert "[\n            100.0,\n            200.0\n          ]" in saved

    written = write_cortical_depth_separate_geojsons(tmp_path / "separate", result, stem="xenium_cortical_depth")
    assert set(written) == {"pia", "wm", "side", "exclusion"}
    assert written["pia"].name == "xenium_cortical_depth_pial_boundary.geojson"
    assert written["wm"].name == "xenium_cortical_depth_wm_boundary.geojson"
    assert '"tissue_piece_id": "piece_a"' in written["pia"].read_text()
    assert '"role": "exclusion"' in written["exclusion"].read_text()


def test_cortical_depth_geojson_export_accepts_pial_only_piece():
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [np.asarray([[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]])],
            "pia": [(np.asarray([[10.0, 0.0], [10.0, 100.0]]), "surface_only")],
        }
    )

    assert result.ok
    pial = next(feature for feature in result.geojson["features"] if feature["properties"]["annotation_role"] == "pia")
    assert pial["properties"]["tissue_piece_id"] == "surface_only"
    assert pial["properties"]["piece_mode"] == "mask_qc_only"


def test_cortical_depth_geojson_export_accepts_edge_overhang_with_near_snapped_endpoints():
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [
                np.asarray(
                    [
                        [-20.0, 0.0],
                        [80.0, 0.0],
                        [80.0, 100.0],
                        [-20.0, 100.0],
                    ]
                )
            ],
            "pia": [(np.asarray([[10.0, 0.0001], [10.0, 100.0]]), "piece_a")],
            "wm": [(np.asarray([[50.0, 0.0], [50.0, 100.0001]]), "piece_a")],
        }
    )

    assert result.ok
    assert not any("do not form a polygon" in error for error in result.errors)


def test_cortical_depth_geojson_export_deduplicates_near_identical_candidate_polygons():
    square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    near_square = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.000001), (0.0, 10.0)])
    distinct = Polygon([(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)])

    assert _unique_valid_polygons([square, near_square, distinct]) == [square, distinct]


def test_cortical_depth_geojson_export_rejects_wm_without_pia():
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [np.asarray([[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]])],
            "wm": [(np.asarray([[30.0, 0.0], [30.0, 100.0]]), "piece_a")],
        }
    )

    assert not result.ok
    assert any("gray/white boundary but no pial boundary" in error for error in result.errors)


def test_cortical_depth_geojson_export_rejects_multiple_edge_lines():
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [
                np.asarray([[0.0, 0.0], [50.0, 0.0]]),
                np.asarray([[0.0, 100.0], [50.0, 100.0]]),
            ],
            "pia": [np.asarray([[10.0, 0.0], [10.0, 100.0]])],
        }
    )

    assert not result.ok
    assert any("exactly one tissue-edge" in error for error in result.errors)


def test_cortical_depth_geojson_export_can_force_write_invalid_debug_file(tmp_path):
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [np.asarray([[0.0, 0.0], [50.0, 0.0]])],
            "wm": [(np.asarray([[30.0, 0.0], [30.0, 50.0]]), "piece_a")],
        }
    )
    assert not result.ok

    out = tmp_path / "debug_annotations.geojson"
    write_cortical_depth_annotation_geojson(
        out,
        result,
        allow_invalid=True,
        include_validation_report=True,
    )

    saved = out.read_text()
    assert '"napari_compare_validation"' in saved
    assert '"ok": false' in saved
    assert "gray/white boundary but no pial boundary" in saved


def test_cortical_depth_geojson_export_flags_crossing_and_exclusion_touch():
    result = build_cortical_depth_annotation_geojson(
        {
            "side": [np.asarray([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]])],
            "pia": [np.asarray([[0.0, 0.0], [100.0, 100.0]])],
            "wm": [np.asarray([[100.0, 0.0], [0.0, 100.0]])],
            "exclusion": [np.asarray([[20.0, 20.0], [20.0, 40.0], [40.0, 40.0], [40.0, 20.0]])],
        }
    )

    assert not result.ok
    assert any("do not form a polygon" in error for error in result.errors)
    assert any("intersect" in warning for warning in result.warnings)
    assert any("exclusion polygon 1 touches or overlaps" in warning for warning in result.warnings)


def test_cortical_depth_snap_moves_boundary_endpoints_to_edge():
    snapped = snap_cortical_depth_boundaries_to_edge(
        pial_shapes=[np.asarray([[5.0, 2.0], [5.0, 98.0]])],
        wm_shapes=[np.asarray([[40.0, 2.0], [40.0, 98.0]])],
        edge_shapes=[np.asarray([[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]])],
    )

    assert set(snapped) == {"pia", "wm"}
    assert np.allclose(snapped["pia"][0], np.asarray([[5.0, 0.0], [5.0, 100.0]]))
    assert np.allclose(snapped["wm"][0], np.asarray([[40.0, 0.0], [40.0, 100.0]]))


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
