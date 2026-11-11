"""Local package export, verified save tracking and resume of managed annotation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import spatialdata as sd
from qtpy.QtWidgets import QMessageBox
from spatialdata.models import Image2DModel
from test_hitl_crops import qapp as qapp
from test_hitl_crops import source as source
from test_hitl_crops import wait_for
from test_hitl_crops import widget as widget

from napari_compare_xenium_merscope.hitl_annotation_session import (
    annotation_progress,
    file_sha256,
    open_annotation_package,
    validate_saved_annotation,
)
from napari_compare_xenium_merscope.hitl_crops import CropRectangle
from napari_compare_xenium_merscope.hitl_normalization import normalize_model_plane
from napari_compare_xenium_merscope.hitl_package import (
    AnnotationExportSettings,
    export_local_annotation_package,
    load_annotation_export_settings,
    upgrade_legacy_annotation_package,
    validate_local_density,
)


def write_verified_save(package, *, nonce="test-session", masks=None):
    crop = package.crops[0]
    if masks is None:
        masks = np.zeros(crop.shape, np.uint32)
        masks[30:60, 30:60] = 1
        masks[90:120, 90:120] = 2
    segment = package.root / f"{crop.crop_id}_seg.npy"
    # Importers must only hash this file, never unpickle it.
    segment.write_bytes(b"GUI output is opaque to the viewer and server importer")
    numeric = package.root / "masks" / f"{crop.crop_id}.npz"
    np.savez_compressed(numeric, masks=masks)
    receipt = {
        "schema_version": 1,
        "package_id": package.package_id,
        "manifest_sha256": package.manifest_sha256,
        "crop_id": crop.crop_id,
        "session_nonce": nonce,
        "cellpose_version": "4.2.1.1",
        "image_sha256": crop.files_sha256[f"{crop.crop_id}.tif"],
        "seg_sha256": file_sha256(segment),
        "masks_sha256": file_sha256(numeric),
        "cell_count": int(np.count_nonzero(np.unique(masks))),
    }
    (package.root / "receipts" / f"{crop.crop_id}.json").write_text(json.dumps(receipt))
    return masks


@pytest.fixture
def managed(source):
    store, context, _ = source
    before = {
        str(path.relative_to(store)): file_sha256(path)
        for path in store.rglob("*")
        if path.is_file()
    }
    package = export_local_annotation_package(
        store,
        context,
        [CropRectangle(64, 64, 320, 480).vertices()],
        AnnotationExportSettings(("DAPI",), "joint", None, None),
    )
    assert package.root.parent == store.parent
    assert before == {
        str(path.relative_to(store)): file_sha256(path)
        for path in store.rglob("*")
        if path.is_file()
    }
    return package


def test_local_export_has_gui_prefills_model_arrays_and_transfer_location(managed):
    package = open_annotation_package(managed.root)
    crop = package.crops[0]
    assert crop.shape == (320, 480)
    assert package.transfer_relative_path == f"annotations/{package.package_id}/"
    assert (package.root / f"{crop.crop_id}_seg.npy").is_file()
    image = np.load(
        package.root / "model_input" / f"{crop.crop_id}.npy", allow_pickle=False
    )
    assert image.shape == (3, 320, 480)
    np.testing.assert_array_equal(image[0], image[1])
    assert annotation_progress(package) == {}
    settings = load_annotation_export_settings(package.root / "roi_context.json")
    assert settings.channels == ("DAPI",)


def test_normalization_handles_constant_channels_and_keeps_complete_halo_statistics():
    plane = np.arange(512 * 768, dtype=np.float32).reshape(512, 768)
    normalized = normalize_model_plane(plane)
    expected8 = (
        np.clip(
            (plane - np.percentile(plane, (2, 98))[0])
            / (np.ptp(np.percentile(plane, (2, 98))) + 1e-8),
            0,
            1,
        )
        * 255
    ).astype(np.uint8)
    low = np.percentile(expected8.astype(np.float32)[None], 1.0)
    high = np.percentile(expected8.astype(np.float32)[None], 99.0)
    expected = expected8.astype(np.float32)
    expected -= low
    expected /= high - low
    np.testing.assert_array_equal(
        normalized.values[100:356, 100:420], expected[100:356, 100:420]
    )
    assert not normalize_model_plane(np.ones((256, 256))).values.any()


def test_atera_export_uses_existing_density_as_third_native_channel(source, tmp_path):
    original, context, _ = source
    data = sd.read_zarr(original)
    rows, columns = np.indices((context.height, context.width), dtype=np.float32)
    source_path = tmp_path / "three-channel-source.zarr"
    sd.SpatialData(
        images={
            "morphology_focus": Image2DModel.parse(
                np.stack((rows, columns)),
                dims=("c", "y", "x"),
                c_coords=["DAPI", "18S"],
            )
        },
        labels=data.labels,
    ).write(source_path)
    metadata = next(
        source_path / name
        for name in ("zarr.json", ".zattrs")
        if (source_path / name).is_file()
    )
    context = replace(context, source_metadata_sha256=file_sha256(metadata))
    density_path = tmp_path / "transcript_density.zarr"
    density_values = (rows + columns)[None]
    sd.SpatialData(
        images={
            "atera_transcript_density": Image2DModel.parse(
                density_values, dims=("c", "y", "x"), c_coords=["transcript_density"]
            )
        }
    ).write(density_path)
    record = {
        "profile": {"sigma_um": 0.15},
        "geometry": {
            "height": context.height,
            "width": context.width,
            "pixel_size_um": context.pixel_size_um,
        },
    }
    record["fingerprint"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    (density_path / "transcript_density.json").write_text(json.dumps(record))
    settings = AnnotationExportSettings(
        ("DAPI", "18S", "transcript_density"), "per_channel", record, None
    )
    package = export_local_annotation_package(
        source_path,
        context,
        [CropRectangle(64, 64, 320, 480).vertices()],
        settings,
        density_path=density_path,
    )
    raw = np.load(package.root / "raw/crop_001.npy", allow_pickle=False)
    model = np.load(package.root / "model_input/crop_001.npy", allow_pickle=False)
    np.testing.assert_array_equal(raw[2], density_values[0, 64:384, 64:544])
    np.testing.assert_array_equal(
        model[2], normalize_model_plane(density_values[0]).values[64:384, 64:544]
    )
    assert package.manifest["model_channels"] == ["DAPI", "18S", "transcript_density"]
    record["profile"]["sigma_um"] = 0.5
    (density_path / "transcript_density.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        validate_local_density(density_path, context, settings)


def test_save_receipt_accepts_unchanged_masks_but_refuses_stale_or_changed_files(
    managed,
):
    write_verified_save(managed)
    crop = managed.crops[0]
    assert (
        validate_saved_annotation(
            managed, crop, session_nonce="test-session"
        ).cell_count
        == 2
    )
    with pytest.raises(ValueError, match="this crop"):
        validate_saved_annotation(managed, crop, session_nonce="another-session")
    (managed.root / f"{crop.crop_id}_seg.npy").write_bytes(b"changed afterwards")
    assert annotation_progress(managed) == {}


def test_disconnected_masks_remain_unfinished(managed):
    masks = np.zeros(managed.crops[0].shape, np.uint32)
    masks[10:20, 10:20] = 1
    masks[30:40, 30:40] = 1
    write_verified_save(managed, masks=masks)
    with pytest.raises(ValueError, match="connected"):
        validate_saved_annotation(managed, managed.crops[0])
    assert not annotation_progress(managed)


def test_legacy_upgrade_preserves_edited_gui_file_and_original_package(managed, source):
    manifest_path = managed.root / "manifest.json"
    values = json.loads(manifest_path.read_text())
    values.pop("managed_annotation_version")
    manifest_path.write_text(json.dumps(values))
    segment = managed.root / "crop_001_seg.npy"
    segment.write_bytes(b"existing mask edits must stay intact")
    before = {
        str(path.relative_to(managed.root)): file_sha256(path)
        for path in managed.root.rglob("*")
        if path.is_file()
    }
    _, context, _ = source
    upgraded = upgrade_legacy_annotation_package(
        managed.root, managed.root.parent, context
    )
    assert upgraded.root != managed.root
    assert (upgraded.root / "crop_001_seg.npy").read_bytes() == segment.read_bytes()
    assert not annotation_progress(upgraded)
    assert before == {
        str(path.relative_to(managed.root)): file_sha256(path)
        for path in managed.root.rglob("*")
        if path.is_file()
    }


def test_package_can_be_annotated_without_loading_full_spatialdata(widget, managed):
    widget.set_dataset(None)
    widget.open_annotation_package(managed.root)
    assert widget._session.package.package_id == managed.package_id
    assert widget._annotate_button.isEnabled()
    assert widget._crop_list.count() == 1
    widget._new_crop_selection()
    assert widget._session is None


def test_legacy_upgrade_never_converts_evaluation_crops_into_training(managed, source):
    path = managed.root / "manifest.json"
    values = json.loads(path.read_text())
    values.pop("managed_annotation_version")
    values["evaluation_eligible"] = True
    path.write_text(json.dumps(values))
    before = list(managed.root.parent.glob("hitl-*"))
    with pytest.raises(ValueError, match="held-out"):
        upgrade_legacy_annotation_package(managed.root, managed.root.parent, source[1])
    assert list(managed.root.parent.glob("hitl-*")) == before
    assert json.loads(path.read_text())["evaluation_eligible"] is True


@pytest.mark.skipif(
    not os.environ.get("MERXEN_TEST_CELLPOSE_PYTHON"),
    reason="Separate Cellpose GUI environment is optional",
)
@pytest.mark.parametrize("action", ["save", "close", "failed_save"])
def test_real_cellpose_gui_records_only_successful_save(managed, action):
    from importlib.resources import files

    crop = managed.crops[0]
    masks = np.zeros(crop.shape, np.uint32)
    masks[30:65, 30:65] = 1
    masks[100:150, 100:150] = 2
    np.save(
        managed.root / "crop_001_seg.npy",
        {
            "masks": masks,
            "outlines": masks,
            "filename": "crop_001.tif",
            "ismanual": np.ones(2, bool),
        },
    )
    launcher = str(
        files("napari_compare_xenium_merscope").joinpath(
            "resources/cellpose_annotation_runner.txt"
        )
    )
    driver = """
import runpy, sys
from pathlib import Path
import numpy as np
from qtpy.QtWidgets import QApplication, QMessageBox
from cellpose.gui.gui import MainW
from cellpose.gui import io
launcher, package, action = sys.argv[1:]
def run(app):
    window = next(w for w in app.topLevelWidgets() if isinstance(w, MainW))
    assert window.loaded and int(window.cellpix.max()) == 2
    if action == "failed_save":
        original = np.save
        def fail(file, *args, **kwargs):
            if str(file).endswith("_seg.npy"):
                raise OSError("simulated failed disk write")
            return original(file, *args, **kwargs)
        np.save = fail
    if action != "close":
        io._save_sets(window)
    window.close()
    return 0
QApplication.exec_ = run
QMessageBox.warning = lambda *args: None
sys.argv = [launcher, "--package", package, "--crop", "crop_001", "--session", "real-save"]
runpy.run_path(launcher, run_name="__main__")
"""
    completed = subprocess.run(
        [
            os.environ["MERXEN_TEST_CELLPOSE_PYTHON"],
            "-I",
            "-c",
            driver,
            launcher,
            str(managed.root),
            action,
        ],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if action == "save":
        assert (
            validate_saved_annotation(
                managed, crop, session_nonce="real-save"
            ).cell_count
            == 2
        )
    else:
        assert not annotation_progress(managed)


def test_save_areas_prepares_package_and_annotation_updates_colour_and_completion(
    widget, source, qapp, monkeypatch
):
    _, _, setup = source
    (setup.parent / "hitl_export_settings.json").write_text(
        json.dumps(
            {
                "model_channels": ["DAPI"],
                "input_normalization": "joint",
                "transcript_density": None,
            }
        )
    )
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args: messages.append(args[-1])
    )
    widget._layer.add(
        CropRectangle(64, 64, 320, 480).vertices(), shape_type="rectangle"
    )
    wait_for(qapp, lambda: widget._save_button.isEnabled())
    widget._save_button.click()
    wait_for(qapp, lambda: widget._session.package is not None)
    package = widget._session.package
    assert widget._annotate_button.isEnabled()
    assert not widget._layer.editable
    write_verified_save(package, nonce="fresh-save")
    process = SimpleNamespace(poll=lambda: 0, returncode=0)
    widget._annotation_launch = (
        package,
        0,
        "fresh-save",
        process,
        time.monotonic(),
        Path("log"),
    )
    widget._poll_annotation_gui()
    assert "Annotated" in widget._crop_list.item(0).text()
    assert widget._crop_list.item(0).foreground().color().name() == "#3ecf6b"
    assert "Annotation complete" in messages[-1]
    assert (
        str(package.root) in messages[-1]
        and package.transfer_relative_path in messages[-1]
    )
    with (package.root / "review.tsv").open() as stream:
        assert next(csv.DictReader(stream, delimiter="\t"))["status"] == "reviewed"
    widget._sessions.clear()
    store, _, _ = source
    widget.set_dataset(store, "ATERA")
    assert widget._session.package.root == package.root
    assert "Annotated" in widget._crop_list.item(0).text()


def test_closing_gui_without_save_does_not_mark_crop_annotated(
    widget, managed, monkeypatch
):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    widget.open_annotation_package(managed.root)
    process = SimpleNamespace(poll=lambda: 0, returncode=0)
    widget._annotation_launch = (
        managed,
        0,
        "not-saved",
        process,
        time.monotonic(),
        Path("log"),
    )
    widget._poll_annotation_gui()
    assert "Annotated" not in widget._crop_list.item(0).text()
    assert "No new verified save" in widget._status.text()


def test_annotate_button_launches_selected_crop_in_external_environment(
    widget, managed, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command, **kwargs: (
            calls.append((command, kwargs)) or SimpleNamespace(poll=lambda: None)
        ),
    )
    widget.open_annotation_package(managed.root)
    widget._settings.setValue("hitl_crops/cellpose_python", sys.executable)
    widget._annotate_button.click()
    assert calls[0][0][0] == sys.executable
    assert calls[0][0][calls[0][0].index("--crop") + 1] == "crop_001"
    assert str(managed.root) in calls[0][0]
    assert "PYTHONPATH" not in calls[0][1]["env"]
    widget._gui_timer.stop()
    widget._annotation_launch = None
