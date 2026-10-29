"""Hand-curated Cellpose cell groups: statistics lookup, CSV round trip, controller."""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from shapely.geometry import MultiPolygon, Polygon

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope import utils as U
from napari_compare_xenium_merscope.cell_groups import (
    CELL_GROUP_ID_COLUMNS,
    CellGroup,
    CellGroupWidget,
    cell_group_layer_name,
    cell_group_table,
    cellpose_statistics_for_labels,
    next_cell_group_color,
    next_cell_group_name,
    read_cell_groups_csv,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _square(x0, y0, size=10.0):
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def _write_strings(group, name, values):
    group.create_array(name, data=np.asarray(values, dtype="U64"))


def _make_store(tmp_path, *, label_ids=(10, 20, 30), images=("MERSCOPE_z_projection",)):
    """A store with Cellpose labels + quantification (X[i, j] = 100 * i + j)."""
    path = tmp_path / "groups.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_group("labels").create_group(U.CELLPOSE_LABEL_KEY)
    table = root.create_group("tables").create_group(U.CELLPOSE_QUANTIFICATION_TABLE_KEY)
    table.create_group("obs").create_array("label_id", data=np.asarray(label_ids, dtype=np.int64))
    image_keys, channels, statistics = [], [], []
    for image in images:
        for channel in ("DAPI", "PolyT"):
            for statistic in ("mean", "median", "iqr"):
                image_keys.append(image)
                channels.append(channel)
                statistics.append(statistic)
    var = table.create_group("var")
    _write_strings(var, "image_key", image_keys)
    _write_strings(var, "channel", channels)
    _write_strings(var, "statistic", statistics)
    n_rows, n_cols = len(label_ids), len(channels)
    x = 100.0 * np.arange(n_rows)[:, None] + np.arange(n_cols)[None, :]
    table.create_array("X", data=x)
    return path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_statistics_follow_requested_order_and_mark_missing_cells(tmp_path):
    store = _make_store(tmp_path)
    stats = cellpose_statistics_for_labels(store, [30, 99, 10])
    assert list(stats.index) == [30, 99, 10]
    assert list(stats.columns) == [
        "DAPI__mean", "DAPI__median", "DAPI__iqr",
        "PolyT__mean", "PolyT__median", "PolyT__iqr",
    ]
    assert stats.loc[30].tolist() == [200.0, 201.0, 202.0, 203.0, 204.0, 205.0]
    assert stats.loc[10].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert stats.loc[99].isna().all()


def test_statistic_columns_are_image_qualified_when_channels_repeat(tmp_path):
    store = _make_store(tmp_path, images=("morphology_focus", "post_xenium_if"))
    stats = cellpose_statistics_for_labels(store, [20])
    assert stats.columns[0] == "morphology_focus__DAPI__mean"
    assert stats.columns[6] == "post_xenium_if__DAPI__mean"
    assert stats.loc[20, "post_xenium_if__DAPI__mean"] == 106.0


def test_group_table_joins_statistics_and_round_trips_through_csv(tmp_path):
    store = _make_store(tmp_path)
    groups = [
        CellGroup("swollen", "#e6194b", {30: _square(0, 0), 10: _square(20, 0, 4)}),
        CellGroup("normal", "#3cb44b", {20: _square(40, 40)}),
    ]
    stats = cellpose_statistics_for_labels(store, [30, 10, 20])
    table = cell_group_table(groups, dataset="MERSCOPE", store=str(store), statistics=stats)

    assert tuple(table.columns[: len(CELL_GROUP_ID_COLUMNS)]) == CELL_GROUP_ID_COLUMNS
    assert table["group"].tolist() == ["swollen", "swollen", "normal"]
    assert table["label_id"].tolist() == [30, 10, 20]
    assert table["segmentation"].unique().tolist() == [U.CELLPOSE_SHAPE_KEY]
    assert table.loc[0, ["centroid_x_um", "centroid_y_um", "area_um2"]].tolist() == [5.0, 5.0, 100.0]
    assert table.loc[1, "area_um2"] == 16.0
    assert table.loc[2, "DAPI__iqr"] == 102.0

    path = tmp_path / "groups.csv"
    table.to_csv(path, index=False)
    assert read_cell_groups_csv(path) == [("swollen", [30, 10]), ("normal", [20])]


def test_reading_groups_rejects_other_segmentations(tmp_path):
    path = tmp_path / "proseg.csv"
    pd.DataFrame({"group": ["a"], "label_id": [1], "segmentation": ["MOSAIK_proseg"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="MOSAIK_proseg"):
        read_cell_groups_csv(path)
    pd.DataFrame({"group": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="label_id"):
        read_cell_groups_csv(path)


def test_default_names_and_colours_skip_used_ones():
    assert next_cell_group_name(["Group 1", "group 2"]) == "Group 3"
    first = CellGroup("a", next_cell_group_color([]))
    assert next_cell_group_color([first]) != first.color


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@pytest.fixture
def ctrl(tmp_path, qapp):
    from napari.components import ViewerModel

    store = _make_store(tmp_path)
    viewer = ViewerModel()
    config = V.DatasetConfig(name="MERSCOPE", zarr_path=store)
    controller = V.ComparisonViewerController(
        viewer,
        {"MERSCOPE": config},
        types.SimpleNamespace(
            background_io_workers=1,
            segmentation_source="shapes",
            shape_opacity=0.5,
            shape_edge_width=1.0,
            label_contour_width=1,
        ),
    )
    two_part = MultiPolygon([_square(40, 0, 4), _square(50, 0, 4)])
    cellpose = gpd.GeoDataFrame(
        {"geometry": [_square(0, 0), _square(20, 0), two_part]}, index=[10, 20, 30]
    )
    controller.active_dataset = "MERSCOPE"
    controller._active_sdata = types.SimpleNamespace(
        shapes={U.CELLPOSE_SHAPE_KEY: cellpose}, images={}, labels={}
    )
    return controller


def _click(ctrl, x, y):
    ctrl._handle_viewer_click(types.SimpleNamespace(position=(y, x), viewbox=None))


def _layer(ctrl, name):
    return ctrl._get_layer_by_name(cell_group_layer_name(name))


def test_picking_adds_and_removes_cells_without_opening_the_inspector(ctrl):
    ctrl.set_cell_group_picking(True)
    assert ctrl._cell_group_picking
    assert [group.name for group in ctrl._cell_groups()] == ["Group 1"]

    _click(ctrl, 5, 5)
    _click(ctrl, 42, 2)  # first part of the two-polygon cell 30
    group = ctrl._cell_group("Group 1")
    assert list(group.cells) == [10, 30]
    assert len(_layer(ctrl, "Group 1").data) == 3  # cell 30 contributes two polygons
    assert not ctrl._selected_cells  # no inspector selection

    _click(ctrl, 52, 2)  # second part of cell 30 -> removes the whole cell
    assert list(group.cells) == [10]
    assert len(_layer(ctrl, "Group 1").data) == 1
    _click(ctrl, 500, 500)  # empty space changes nothing
    assert list(group.cells) == [10]

    ctrl.set_cell_group_picking(False)
    _click(ctrl, 25, 5)
    assert list(group.cells) == [10]


def test_a_cell_can_join_several_groups_and_renaming_keeps_the_layer(ctrl):
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    assert ctrl.add_cell_group("swollen") == "swollen"
    assert ctrl.add_cell_group("swollen") is None  # duplicate names are refused
    _click(ctrl, 5, 5)  # cell 10 joins swollen and stays in Group 1
    _click(ctrl, 25, 5)
    assert list(ctrl._cell_group("Group 1").cells) == [10]
    assert len(_layer(ctrl, "Group 1").data) == 1
    assert list(ctrl._cell_group("swollen").cells) == [10, 20]

    _click(ctrl, 5, 5)  # removing it from swollen leaves Group 1 alone
    assert list(ctrl._cell_group("swollen").cells) == [20]
    assert list(ctrl._cell_group("Group 1").cells) == [10]
    _click(ctrl, 5, 5)

    assert ctrl.rename_cell_group("swollen", "dystrophic")
    assert _layer(ctrl, "swollen") is None
    assert len(_layer(ctrl, "dystrophic").data) == 2
    _click(ctrl, 42, 2)
    assert list(ctrl._cell_group("dystrophic").cells) == [20, 10, 30]

    assert ctrl.clear_cell_group("dystrophic")
    assert _layer(ctrl, "dystrophic") is None
    assert list(ctrl._cell_group("Group 1").cells) == [10]
    assert ctrl.delete_cell_group("dystrophic")
    assert ctrl._active_cell_group().name == "Group 1"


def test_a_cell_in_two_groups_exports_and_reloads_as_two_rows(ctrl, tmp_path):
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    ctrl.add_cell_group("bright")
    _click(ctrl, 5, 5)
    path = tmp_path / "overlap.csv"
    ctrl.export_cell_groups(path)
    exported = pd.read_csv(path)
    assert exported[["group", "label_id"]].values.tolist() == [["Group 1", 10], ["bright", 10]]
    assert exported["DAPI__mean"].tolist() == [0.0, 0.0]

    for name in ("Group 1", "bright"):
        ctrl.delete_cell_group(name)
    assert ctrl.load_cell_groups(path) == 2
    assert {group.name: list(group.cells) for group in ctrl._cell_groups()} == {
        "Group 1": [10], "bright": [10],
    }


def test_a_layer_deleted_in_napari_is_rebuilt_on_the_next_click(ctrl):
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    ctrl.viewer.layers.remove(_layer(ctrl, "Group 1"))
    _click(ctrl, 25, 5)
    assert len(_layer(ctrl, "Group 1").data) == 2


def test_picking_uses_unrotated_dataset_coordinates(ctrl):
    # The display is rotated: the click lands on cell 20's on-screen position.
    ctrl._left_panel_adapter = types.SimpleNamespace(unrotate_world_xy=lambda xy: (xy[0] + 20.0, xy[1]))
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    assert list(ctrl._active_cell_group().cells) == [20]


def test_export_and_reload_groups(ctrl, tmp_path):
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 42, 2)
    ctrl.add_cell_group("normal")
    _click(ctrl, 5, 5)
    path = tmp_path / "export.csv"
    table = ctrl.export_cell_groups(path)
    exported = pd.read_csv(path)
    assert exported["group"].tolist() == ["Group 1", "normal"]
    assert exported["label_id"].tolist() == [30, 10]
    assert exported["dataset"].unique().tolist() == ["MERSCOPE"]
    assert exported.loc[0, "DAPI__mean"] == 200.0
    assert exported.loc[1, "PolyT__iqr"] == 5.0
    assert exported.loc[0, "area_um2"] == pytest.approx(32.0)
    assert len(table) == 2

    for name in ("Group 1", "normal"):
        ctrl.delete_cell_group(name)
    assert ctrl._cell_groups() == []
    pd.concat(
        [exported, pd.DataFrame({"group": ["normal"], "label_id": [777]})], ignore_index=True
    ).to_csv(path, index=False)
    assert ctrl.load_cell_groups(path) == 2
    assert [group.name for group in ctrl._cell_groups()] == ["Group 1", "normal"]
    assert list(ctrl._cell_group("normal").cells) == [10]
    assert len(_layer(ctrl, "Group 1").data) == 2
    assert ctrl._active_cell_group().name == "Group 1"


def test_export_without_quantification_table_keeps_ids(ctrl, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "cellpose_quantification_table_available", lambda *_: False)
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    table = ctrl.export_cell_groups(tmp_path / "ids.csv")
    assert list(table.columns) == list(CELL_GROUP_ID_COLUMNS)


def test_groups_are_kept_per_store_and_redrawn_on_return(ctrl):
    ctrl.set_cell_group_picking(True)
    _click(ctrl, 5, 5)
    ctrl._clear_layers()
    ctrl._publish_cell_groups()
    assert len(_layer(ctrl, "Group 1").data) == 1


def test_paired_mode_disables_grouping(ctrl, qapp):
    widget = CellGroupWidget(*([lambda *_: None] * 8))
    ctrl.set_cell_group_widget(widget)
    assert widget._pick_button.isEnabled()
    ctrl._paired_session = object()
    ctrl._publish_cell_groups()
    assert not widget._pick_button.isEnabled()
    assert "standalone" in widget._availability_label.text()
    ctrl.set_cell_group_picking(True)
    assert not ctrl._cell_group_picking


def test_missing_cellpose_segmentation_blocks_picking(ctrl, monkeypatch):
    ctrl._active_sdata.shapes.clear()
    monkeypatch.setattr(ctrl, "_read_shapes_from_store", lambda: None)
    ctrl.set_cell_group_picking(True)
    assert not ctrl._cell_group_picking


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def test_widget_reports_active_group_and_forwards_actions(qapp):
    calls = []
    widget = CellGroupWidget(
        lambda active: calls.append(("pick", active)),
        lambda name: calls.append(("new", name)),
        lambda old, new: calls.append(("rename", old, new)),
        lambda name: calls.append(("delete", name)),
        lambda name: calls.append(("clear", name)),
        lambda name: calls.append(("active", name)),
        lambda path: calls.append(("export", path)),
        lambda path: calls.append(("load", path)),
    )
    widget.set_available(True)
    assert not widget._export_button.isEnabled()
    widget.set_groups([("a", "#e6194b", 1), ("b", "#3cb44b", 0)], "b")
    assert widget.active_group() == "b"
    assert widget._group_list.item(0).text() == "a  (1 cell)"
    assert widget._export_button.isEnabled()
    assert calls == []  # programmatic updates do not echo back

    widget._group_list.setCurrentRow(0)
    widget._pick_button.click()
    widget._delete_button.click()
    assert calls == [("active", "a"), ("pick", True), ("delete", "a")]
    widget.set_picking(False)
    assert not widget._pick_button.isChecked()
    assert calls[-1] == ("delete", "a")


def test_control_panel_has_a_cell_groups_tab(qapp):
    from test_startup import _panel

    widget = CellGroupWidget(*([lambda *_: None] * 8))
    panel = _panel([], cell_group_widget=widget)
    titles = [panel._tab_group.button(i).text() for i in range(len(panel._tab_group.buttons()))]
    assert titles.index("Cell groups") == titles.index("UMAP") + 1
    assert titles.index("Dataset loader") == 6
