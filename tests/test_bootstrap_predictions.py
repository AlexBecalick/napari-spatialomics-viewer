"""Raw labels, conditional alternatives and bootstrap filters retain provenance."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from napari_compare_xenium_merscope.utils import (
    cell_type_levels,
    load_cell_type_assignments,
)
from napari_compare_xenium_merscope.viewer import _cell_summary_html


def _store(tmp_path):
    # Above 2**53 and 2**63: confidence joins must never round the mask IDs.
    ids = np.array(
        [9223372036854775809, 9223372036854775811, 9223372036854775813], dtype=np.uint64
    )
    obs = pd.DataFrame({"instance_id": ids}, index=["a", "b", "empty"])
    for stem in ["ct_seaad_subclass", "ct_whb_supercluster"]:
        obs[f"{stem}_raw_name"] = pd.Categorical(["Type <A>", "Type B", None])
        obs[f"{stem}_name"] = pd.Categorical([None, "Type B", None])
        obs[f"{stem}_bp"] = [0.4, 1.0, np.nan]
        obs[f"{stem}_aggregate_probability"] = [0.2, 1.0, np.nan]
        obs[f"{stem}_resolution_status"] = [
            "not_calibrated",
            "accepted_existing_atlas_policy",
            "no_expression",
        ]
        for rank in range(1, 6):
            obs[f"{stem}_runner_up_{rank}_name"] = pd.Categorical(
                [f"Alternative {rank}", None, None]
            )
            obs[f"{stem}_runner_up_{rank}_probability"] = [0.1, np.nan, np.nan]
    # Standard Allen accepted alias is separate from the WHB raw fields.
    obs["ct_supercluster_name"] = obs.ct_whb_supercluster_name
    data = ad.AnnData(np.zeros((3, 1)), obs=obs)
    data.obsm["X_umap"] = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    data.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    path = tmp_path / "example.zarr"
    zarr.open_group(path, mode="w", zarr_format=2).create_group("tables")
    data.write_zarr(path / "tables" / "table_original_clustering_squidpy")
    return path, ids


def test_raw_predictions_and_accepted_labels_remain_separate(tmp_path):
    path, ids = _store(tmp_path)
    levels = cell_type_levels(path, "original")
    assert "ct_seaad_subclass_raw" in levels
    assert "ct_whb_supercluster_raw" in levels
    labels = load_cell_type_assignments(path, "original")
    assert labels.cell_ids.tolist() == ids[:2].tolist()
    assert labels.labels_for("ct_seaad_subclass_raw").tolist() == ["Type <A>", "Type B"]
    assert labels.labels_for("ct_seaad_subclass").tolist() == [None, "Type B"]
    assert labels.labels_for("ct_supercluster").tolist() == [None, "Type B"]
    # Zero shows all; thresholds are inclusive and don't mutate saved labels.
    assert labels.labels_for("ct_seaad_subclass_raw", 0.4).tolist() == [
        "Type <A>",
        "Type B",
    ]
    assert labels.labels_for("ct_seaad_subclass_raw", 0.5).tolist() == [None, "Type B"]
    assert labels.labels_for("ct_whb_supercluster_raw", 1).tolist() == [None, "Type B"]
    assert labels.labels_for("ct_seaad_subclass_raw").tolist() == ["Type <A>", "Type B"]
    assert labels.labels_for("ct_supercluster", 0.99).tolist() == [None, "Type B"]


def test_inspector_shows_up_to_five_supported_candidates_with_bootstrap_semantics(
    tmp_path,
):
    path, _ = _store(tmp_path)
    labels = load_cell_type_assignments(path, "original")
    predictions = labels.predictions_for_cell(0)
    assert len(predictions) == 2
    assert len(predictions[0]["candidates"]) == 5
    assert predictions[0]["conditional"]
    assert predictions[0]["aggregate_bootstrap"] == 0.2
    assert len(labels.predictions_for_cell(1)[0]["candidates"]) == 1
    text = _cell_summary_html({"bootstrap_predictions": predictions})
    assert "Type &lt;A&gt;" in text and "Type <A>" not in text
    assert "40.0%" in text and "20.0%" in text
    assert "not biological probability" in text
    assert "selected parent class" in text
    assert "not_calibrated" in text


def test_existing_export_without_candidate_columns_still_exposes_top_call(tmp_path):
    path, _ = _store(tmp_path)
    table = path / "tables" / "table_original_clustering_squidpy"
    data = ad.read_zarr(table)
    data.obs = data.obs.drop(
        columns=[
            c for c in data.obs if "runner_up" in c or "aggregate_probability" in c
        ]
    )
    data.write_zarr(table)
    labels = load_cell_type_assignments(path, "original")
    assert labels.labels_for("ct_seaad_subclass_raw", 0.5).tolist() == [None, "Type B"]
    details = labels.predictions_for_cell(0)
    assert len(details[0]["candidates"]) == 1
    assert details[0]["aggregate_bootstrap"] is None


def test_bootstrap_control_converts_percent_and_does_not_dispatch_refreshes():
    from qtpy.QtWidgets import QApplication
    from napari_compare_xenium_merscope.prediction_controls import BootstrapFilterWidget

    app = QApplication.instance() or QApplication([])
    changes = []
    widget = BootstrapFilterWidget(changes.append)
    widget.set_filter(0.3, True)
    assert changes == []
    widget.cutoff.setValue(90)
    assert changes == [0.9]
    widget.set_filter(0, False)
    assert changes == [0.9]
    assert not widget.isEnabled()
    widget.close()
    assert app is not None
