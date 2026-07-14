from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr
from shapely.geometry import MultiPolygon, Polygon

from napari_compare_xenium_merscope.utils import (
    GENE_MARKER_SYMBOLS,
    GeneVisual,
    assign_gene_visuals,
    build_gene_point_groups,
    gene_palette_rgba,
    is_control_gene,
    resolve_gene_column,
    CELLPOSE_LABEL_KEY,
    CELLPOSE_QUANTIFICATION_TABLE_KEY,
    build_cortical_depth_annotation_geojson,
    build_object_annotation_geojson,
    build_binned_label_color_dict,
    cellpose_quantification_features,
    cellpose_quantification_table_available,
    cellpose_value_clip_range,
    derived_outline_cache_key,
    image_scale_dataarrays,
    ensure_cyx,
    is_derived_cache_key,
    label_outline_mask_chunk,
    layer_name_prefix,
    load_cellpose_quantification_values,
    make_layer_name,
    matching_layer_names,
    pixel_window_global_bounds,
    pick_default_shape_key,
    query_geometries_for_bounds,
    read_object_annotation_geojson,
    rasterize_geometries_chunk,
    resolve_cellpose_quantification_feature,
    snap_cortical_depth_boundaries_to_edge,
    write_cortical_depth_annotation_geojson,
    write_cortical_depth_separate_geojsons,
    write_object_annotation_geojson,
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

    image_name = make_layer_name(ds, "image", "MERSCOPE_z_projection", "PolyT")
    assert image_name == "Image | PolyT"
    assert matching_layer_names([image_name, "Genes | Tailed arrow"], layer_name_prefix(ds, "image")) == [
        image_name
    ]
    assert matching_layer_names(["Genes | Tailed arrow"], layer_name_prefix(ds, "genes")) == [
        "Genes | Tailed arrow"
    ]


def test_derived_cache_key_naming_and_filtering():
    outline_key = derived_outline_cache_key("cell_boundaries", 2)
    weird_key = derived_outline_cache_key("foo/bar baz", 1)

    assert outline_key == "_napari_compare_outline__cell_boundaries__w2"
    assert is_derived_cache_key(outline_key)
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


def test_object_annotation_geojson_round_trip_preserves_names_ids_and_xy(tmp_path):
    result = build_object_annotation_geojson(
        {
            "Amyloid plaques": [
                (
                    np.asarray(
                        [[200.0, 100.0], [200.0, 120.0], [220.0, 120.0]]
                    ),
                    "plaque_a",
                ),
                np.asarray(
                    [[300.0, 400.0], [300.0, 420.0], [320.0, 420.0]]
                ),
            ],
            "Tau tangles": [
                np.asarray([[10.0, 50.0], [20.0, 60.0], [20.0, 50.0]])
            ],
        },
        dataset="XENIUM",
    )

    assert result.ok
    assert len(result.geojson["features"]) == 3
    first = result.geojson["features"][0]
    assert first["properties"]["object_type"] == "Amyloid plaques"
    assert first["properties"]["object_id"] == "plaque_a"
    assert first["geometry"]["coordinates"][0][0] == [100.0, 200.0]

    path = tmp_path / "objects.geojson"
    write_object_annotation_geojson(path, result)
    loaded = read_object_annotation_geojson(path)

    assert set(loaded) == {"Amyloid plaques", "Tau tangles"}
    assert loaded["Amyloid plaques"][0].object_id == "plaque_a"
    np.testing.assert_allclose(
        loaded["Amyloid plaques"][0].data,
        np.asarray([[200.0, 100.0], [200.0, 120.0], [220.0, 120.0]]),
    )


def test_object_annotation_export_rejects_duplicate_ids():
    polygon = np.asarray([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    result = build_object_annotation_geojson(
        {"Plaques": [(polygon, "same"), (polygon + 20.0, "same")]}
    )

    assert not result.ok
    assert any("Duplicate object_id" in error for error in result.errors)


def test_object_annotation_generated_id_avoids_reloaded_id_collision():
    result = build_object_annotation_geojson(
        {
            "Amyloid plaques": [
                (
                    np.asarray([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
                    "amyloid_plaques_0002",
                ),
                np.asarray([[10.0, 10.0], [10.0, 15.0], [15.0, 15.0]]),
            ]
        }
    )

    assert result.ok
    assert [
        feature["properties"]["object_id"]
        for feature in result.geojson["features"]
    ] == ["amyloid_plaques_0002", "amyloid_plaques_0003"]


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


def test_rasterize_geometries_chunk_rejects_nonfinite_affine():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    inv_affine = np.eye(3)
    inv_affine[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite 3x3"):
        rasterize_geometries_chunk([poly], [1], (2, 2), inv_affine)


def test_label_outline_mask_chunk_marks_label_boundaries():
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[2:6, 2:6] = 1

    outline = label_outline_mask_chunk(labels)

    assert outline.dtype == np.uint8
    assert np.count_nonzero(outline) == 12
    assert np.all(outline[3:5, 3:5] == 0)
    assert outline[2, 2] == 1
    assert outline[5, 5] == 1


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


def test_resolve_gene_column_prefers_known_names():
    import pandas as pd

    assert resolve_gene_column(pd.DataFrame(columns=["x", "y", "gene"])) == "gene"
    assert resolve_gene_column(pd.DataFrame(columns=["x", "y", "feature_name"])) == "feature_name"
    assert resolve_gene_column(pd.DataFrame(columns=["x", "y"])) is None


def test_is_control_gene_matches_blanks_and_controls():
    assert is_control_gene("Blank-10")
    assert is_control_gene("NegControlProbe_00042")
    assert is_control_gene("antisense-FOO")
    assert not is_control_gene("APP")
    assert not is_control_gene("GJA1")
    assert not is_control_gene("")


def test_assign_gene_visuals_deterministic_alphabetical_and_unique():
    names = [f"G{i:03d}" for i in range(300)]
    shuffled = list(reversed(names))
    a = assign_gene_visuals(shuffled)
    b = assign_gene_visuals(names)

    # Deterministic regardless of input order.
    assert a == b
    # Alphabetical assignment: first gene gets the first symbol.
    assert a["G000"].symbol == GENE_MARKER_SYMBOLS[0]
    assert a["G001"].symbol == GENE_MARKER_SYMBOLS[1]
    # Every (colour, symbol) pair is unique across the whole panel.
    pairs = {(v.rgba, v.symbol) for v in a.values()}
    assert len(pairs) == len(names)
    assert all(isinstance(v, GeneVisual) for v in a.values())


def test_gene_palette_rgba_shape_and_range():
    palette = gene_palette_rgba(5, alpha=0.5)
    assert len(palette) == 5
    for rgba in palette:
        assert len(rgba) == 4
        assert rgba[3] == 0.5
        assert all(0.0 <= c <= 1.0 for c in rgba)


def _gene_points_frame():
    import pandas as pd

    # 10 points across 3 genes + 1 control. background True == unassigned noise.
    rows = [
        # gene,     x,    y,    background
        ("AAA", 1.0, 10.0, False),
        ("AAA", 2.0, 11.0, False),
        ("AAA", 3.0, 12.0, True),
        ("BBB", 4.0, 20.0, False),
        ("BBB", 5.0, 21.0, False),
        ("Blank-1", 6.0, 30.0, False),
        ("CCC", 7.0, 40.0, False),
        ("CCC", 8.0, 41.0, True),
        ("CCC", 9.0, 42.0, True),
        ("CCC", 10.0, 43.0, True),
    ]
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])
    df["assignment"] = np.where(df["background"].to_numpy(), 0, 7).astype("uint32")
    return df


def test_build_gene_point_groups_structure_and_offsets():
    df = _gene_points_frame()
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background"
    )

    assert store.total_points == 10
    assert not store.sampled
    assert store.genes == ["AAA", "BBB", "Blank-1", "CCC"]
    assert store.control_genes == {"Blank-1"}
    # 4 distinct genes -> 4 distinct symbols -> 4 groups.
    assert store.group_symbols == list(GENE_MARKER_SYMBOLS[:4])
    assert store.gene_counts == {"AAA": 3, "BBB": 2, "Blank-1": 1, "CCC": 4}

    # Each gene occupies its own group here; offsets are relative to the group.
    assert store.gene_offsets["AAA"] == (0, 0, 2, 3)   # 2 fg then 1 bg
    assert store.gene_offsets["BBB"] == (1, 0, 2, 2)   # all fg
    assert store.gene_offsets["Blank-1"] == (2, 0, 1, 1)
    assert store.gene_offsets["CCC"] == (3, 0, 1, 4)   # 1 fg then 3 bg

    # Coords are (y, x); foreground rows come before background rows.
    aaa = store.group_coords[0]
    assert aaa.shape == (3, 2)
    assert set(aaa[:2, 0].tolist()) == {10.0, 11.0}    # fg y-values
    assert aaa[2, 0] == 12.0                            # bg y-value
    assert aaa[0, 1] in (1.0, 2.0)                      # x column

    # Colours match the assigned gene visuals.
    visuals = assign_gene_visuals(store.genes)
    np.testing.assert_allclose(store.group_colors[0][0], visuals["AAA"].rgba, atol=1e-6)
    assert store.gene_symbol("CCC") == GENE_MARKER_SYMBOLS[3]


def test_build_gene_point_groups_assignment_fallback_matches_background():
    df = _gene_points_frame()
    from_bg = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background"
    )
    from_assignment = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", assignment_col="assignment"
    )
    assert from_assignment.gene_offsets == from_bg.gene_offsets
    assert from_assignment.gene_counts == from_bg.gene_counts


def test_build_gene_point_groups_subsamples_when_over_cap():
    df = _gene_points_frame()
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background",
        max_points=5, random_state=0,
    )
    assert store.sampled
    assert store.total_points == 5
    assert sum(store.gene_counts.values()) == 5
    assert store.source_total_points == 10
    assert store.source_gene_counts == {"AAA": 3, "BBB": 2, "Blank-1": 1, "CCC": 4}


def test_build_gene_point_groups_empty_input():
    import pandas as pd

    df = pd.DataFrame({"x": [np.nan], "y": [1.0], "gene": ["AAA"], "background": [False]})
    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background"
    )
    assert store.total_points == 0
    assert store.genes == []
    assert store.group_coords == []


def test_build_gene_point_groups_many_genes_share_symbol_groups():
    # >14 genes force several genes into the same symbol group, where a gene's
    # points are NOT ordered by code value. Regression test that per-gene counts
    # and offsets stay correct in that interleaved layout.
    import pandas as pd

    rows = []
    for i in range(20):                       # G000..G019; gene i has i+1 points
        gene = f"G{i:03d}"
        for j in range(i + 1):
            bg = (i == 5 and j == 0)          # one background point in a shared group
            rows.append((gene, float(i * 1000 + j), float(j), bg))
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "background"])

    store = build_gene_point_groups(
        df, x_col="x", y_col="y", gene_col="gene", background_col="background"
    )

    # All 14 symbols used; counts match ground truth.
    assert len(store.group_symbols) == 14
    expected = {f"G{i:03d}": i + 1 for i in range(20)}
    assert store.gene_counts == expected
    assert sum(store.gene_counts.values()) == store.total_points == sum(range(1, 21))

    # Each gene's stored points are exactly its own (verify via x signature).
    for i in range(20):
        gene = f"G{i:03d}"
        gi, fg_start, fg_end, bg_end = store.gene_offsets[gene]
        xs = store.group_coords[gi][fg_start:bg_end][:, 1]  # coords are (y, x)
        assert len(xs) == i + 1
        assert xs.min() >= i * 1000 and xs.max() < i * 1000 + (i + 1)

    # G005 has its one background point ordered last (fg then bg).
    gi, fg_start, fg_end, bg_end = store.gene_offsets["G005"]
    assert bg_end - fg_end == 1
