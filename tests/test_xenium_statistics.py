"""Xenium statistics retain image identity from selection through overlay display."""
import os
import types
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import zarr

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope import utils as U


@pytest.fixture
def store(tmp_path):
    root = zarr.open_group(str(tmp_path / "statistics.zarr"), mode="w")
    root.create_group("labels").create_group(U.CELLPOSE_LABEL_KEY)
    table = root.create_group("tables").create_group(U.CELLPOSE_QUANTIFICATION_TABLE_KEY)
    table.create_group("obs").create_array("label_id", data=np.array([1, 2]))
    var = table.create_group("var")
    for key, values in {
        "image_key": ["morphology_focus", "post_xenium_if"],
        "channel": ["DAPI", "DAPI"],
        "statistic": ["mean", "mean"],
    }.items():
        var.create_array(key, data=np.array(values, dtype="U32"))
    table.create_array("X", data=np.array([[10., 100.], [20., 200.]]))
    return tmp_path / "statistics.zarr"


def test_source_qualified_statistics_load_distinct_values(store):
    features = U.cellpose_quantification_features(store)
    assert U.cellpose_quantification_channel_labels(features) == [
        "morphology_focus: DAPI", "post_xenium_if: DAPI"
    ]
    for image, expected in [("morphology_focus", [10., 20.]), ("post_xenium_if", [100., 200.])]:
        selected = f"{image}: DAPI"
        values = U.load_cellpose_quantification_values(store, selected, "mean")
        assert values.feature.image_key == image
        assert values.values.tolist() == expected
        assert U.resolve_cellpose_quantification_feature(store, selected, "mean") == values.feature
    with pytest.raises(ValueError, match="Ambiguous"):
        U.load_cellpose_quantification_values(store, "DAPI", "mean")
    with pytest.raises(ValueError, match="Ambiguous"):
        U.resolve_cellpose_quantification_feature(store, "DAPI", "mean")


def test_xenium_session_options_and_overlay(store, monkeypatch):
    from qtpy.QtWidgets import QApplication
    from napari.components import ViewerModel

    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    config = V.DatasetConfig(name="XENIUM", zarr_path=store)
    ctrl = V.ComparisonViewerController(
        viewer, {"XENIUM": config},
        types.SimpleNamespace(background_io_workers=1, segmentation_source="shapes", shape_opacity=0.5),
    )
    sdata = types.SimpleNamespace(shapes={}, images={}, labels={})
    monkeypatch.setattr(V.sd, "read_zarr", lambda *a, **k: sdata)
    monkeypatch.setattr(ctrl, "_resolve_dataset_pixel_transform", lambda *a: ((1., 0., 0.), (1., 0., 0.)))
    monkeypatch.setattr(V, "cell_type_levels", lambda *a: [])
    monkeypatch.setattr(V, "cell_umap_available", lambda *a: False)
    session = ctrl._prepare_dataset_session("XENIUM", config, Event())
    assert session.cellpose_values_available
    assert session.cellpose_channels == ["morphology_focus: DAPI", "post_xenium_if: DAPI"]
    ctrl.active_dataset = "XENIUM"
    ctrl._active_sdata = sdata
    ctrl._dataset_sessions["XENIUM"] = session
    published = []
    ctrl.set_cellpose_value_options_callback(lambda *args: published.append(args))
    ctrl._publish_cellpose_value_options()
    assert published == [(session.cellpose_channels, ["mean"], True)]
    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", lambda *a: True)
    monkeypatch.setattr(ctrl, "_sync_umap_from_statistic", lambda *a: None)
    monkeypatch.setattr(ctrl, "_ensure_cellpose_label_key", lambda **k: U.CELLPOSE_LABEL_KEY)
    monkeypatch.setattr(ctrl, "_build_cellpose_label_display", lambda *a, **k: (
        [np.array([[0, 1], [2, 0]], dtype=np.uint32)], np.eye(3)
    ))
    monkeypatch.setattr(V, "thread_worker", None)
    ctrl.load_cellpose_value_overlay("XENIUM", "post_xenium_if: DAPI", "mean", "viridis")
    assert len(viewer.layers) == 1
    assert "post_xenium_if: DAPI mean" in viewer.layers[0].name
    assert viewer.layers[0].data.tolist() == [[0, 1], [2, 0]]
    ctrl.remove_cellpose_value_overlay("XENIUM")
    assert len(viewer.layers) == 0
    session.cellpose_values_available = False
    ctrl._publish_cellpose_value_options()
    assert published[-1][-1] is False
    ctrl._paired_session = object()
    ctrl._publish_cellpose_value_options()
    assert published[-1] == ([], [], False)
    assert app is not None
