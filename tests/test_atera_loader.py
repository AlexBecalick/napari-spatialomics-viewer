"""ATERA standalone loading: CLI, loader buttons, recents, detection, paired refusal, gates."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atera_stores
import pytest

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.paired_views import normalize_platform

PS = 0.5


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def atera_store(tmp_path_factory) -> Path:
    return atera_stores.build_atera_latest_store(tmp_path_factory.mktemp("atera"), pixel_size=PS)


@pytest.fixture(autouse=True)
def _private_app_data(tmp_path, monkeypatch):
    app_data = tmp_path / "app-data-root"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


def _panel(**overrides):
    callback = lambda *args, **kwargs: None  # noqa: E731
    arguments = dict(
        datasets=[],
        gene_inspector_widget=None,
        cell_type_widget=None,
        load_callback=callback,
        load_selected_labels_callback=callback,
        unload_selected_shapes_callback=callback,
        load_transcripts_callback=callback,
        unload_transcripts_callback=callback,
        load_selected_image_callback=callback,
        load_all_images_callback=callback,
        unload_selected_image_callback=callback,
        load_cellpose_values_callback=callback,
        remove_cellpose_values_callback=callback,
        create_annotation_layers_callback=callback,
        set_annotation_piece_callback=callback,
        apply_annotation_piece_callback=callback,
        snap_annotation_side_edges_callback=callback,
        validate_annotation_callback=callback,
        export_annotation_callback=callback,
        create_object_annotation_callback=callback,
        validate_object_annotations_callback=callback,
        export_object_annotations_callback=callback,
        load_object_annotations_callback=callback,
        load_paired_callback=callback,
        load_standalone_callback=callback,
        initial_dataset=None,
    )
    arguments.update(overrides)
    return V.ViewerControlPanel(**arguments)


def _settings(tmp_path):
    from qtpy.QtCore import QSettings

    return QSettings(str(tmp_path / "viewer-settings.ini"), QSettings.IniFormat)


class _Viewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []
        self.grid = SimpleNamespace(enabled=True)

    def reset_view(self):
        pass


def _args(**overrides):
    defaults = dict(
        background_io_workers=1,
        session_cache_gb=0,
        image_pyramid_downsample=4,
        overwrite_derived_caches=False,
        skip_images=False,
        skip_transcripts=False,
        segmentation_source="shapes",
        hide_images=False,
        image_opacity=1.0,
        merscope_transform_path=None,
        xenium_spec_path=None,
        atera_spec_path=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _controller(**overrides):
    return V.ComparisonViewerController(_Viewer(), {}, _args(**overrides))


def _statuses(ctrl):
    out: list[str] = []
    ctrl.set_status_callback(out.append)
    return out


def _copy_store(src: Path, dst_root: Path) -> Path:
    dst = dst_root / src.name
    shutil.copytree(src, dst)
    return dst


def _heuristic_only_copy(src: Path, dst_root: Path) -> Path:
    """A copy with no authoritative platform metadata (sidecars and names remain)."""
    store = _copy_store(src, dst_root)
    meta = json.loads((store / "zarr.json").read_text())
    meta["attributes"].pop("merxen_schema", None)
    (store / "zarr.json").write_text(json.dumps(meta))
    (store / "merxen_native_latest.json").unlink(missing_ok=True)
    assert V.detect_store_platform(store).authoritative is False
    return store


def _xenium_spec(tmp_path, pixel_size=0.3) -> Path:
    path = tmp_path / "elsewhere" / "experiment.xenium"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pixel_size": pixel_size}))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_atera_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = V.parse_args()
    assert args.atera_zarr is None
    assert args.atera_spec_path is None
    assert args.initial_dataset is None


def test_parse_args_accepts_atera_zarr_and_initial_dataset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--atera-zarr", "a.zarr", "--initial-dataset", "ATERA", "--atera-spec-path", "s.json"],
    )
    args = V.parse_args()
    assert args.atera_zarr == Path("a.zarr")
    assert args.initial_dataset == "ATERA"
    assert args.atera_spec_path == Path("s.json")


@pytest.mark.parametrize(
    "extra",
    [
        ["--merscope-zarr", "m.zarr"],
        ["--xenium-zarr", "x.zarr"],
        ["--paired-view", "side-by-side"],
        ["--registration-store", "r.zarr"],
    ],
)
def test_parse_args_atera_zarr_is_exclusive(monkeypatch, capsys, extra):
    monkeypatch.setattr(sys, "argv", ["prog", "--atera-zarr", "a.zarr", *extra])
    with pytest.raises(SystemExit):
        V.parse_args()
    assert "--atera-zarr" in capsys.readouterr().err


def test_startup_configs_build_atera_with_its_own_spec_override(tmp_path):
    args = SimpleNamespace(
        merscope_zarr=None,
        xenium_zarr=None,
        atera_zarr=tmp_path / "a.zarr",
        merscope_transform_path=tmp_path / "m.csv",
        xenium_spec_path=tmp_path / "experiment.xenium",
        atera_spec_path=tmp_path / "experiment.spatial",
    )
    configs = V.startup_dataset_configs(args)
    assert list(configs) == ["ATERA"]
    cfg = configs["ATERA"]
    assert cfg.name == "ATERA" and cfg.platform == "ATERA"
    assert cfg.spec_path == tmp_path / "experiment.spatial"
    assert cfg.xenium_spec_path is None and cfg.merscope_transform_path is None


def test_startup_detection_relabels_only_on_authoritative_evidence(atera_store, tmp_path):
    args = _args(initial_dataset="XENIUM")
    datasets = {"XENIUM": V.dataset_config_for("XENIUM", atera_store, args)}
    relabelled, initial = V.apply_startup_store_detection(datasets, args, paired=False)
    assert list(relabelled) == ["ATERA"] and initial == "ATERA"
    assert relabelled["ATERA"].platform == "ATERA"

    # A paired startup only warns; load_paired_dataset refuses the slot.
    kept, _ = V.apply_startup_store_detection(datasets, args, paired=True)
    assert list(kept) == ["XENIUM"]

    heuristic = _heuristic_only_copy(atera_store, tmp_path / "heuristic")
    datasets = {"XENIUM": V.dataset_config_for("XENIUM", heuristic, args)}
    kept, initial = V.apply_startup_store_detection(datasets, args, paired=False)
    assert list(kept) == ["XENIUM"] and initial == "XENIUM"


def test_main_builds_an_atera_dataset_config(atera_store, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["prog", "--atera-zarr", str(atera_store), "--initial-dataset", "ATERA"]
    )
    captured = {}

    class _Stop(Exception):
        pass

    def fake_controller(*, viewer, datasets, args):
        captured["datasets"] = datasets
        raise _Stop

    monkeypatch.setattr(V, "install_thread_safe_napari_dask_cache", lambda: SimpleNamespace(
        cache=SimpleNamespace(available_bytes=0)))
    monkeypatch.setattr(V, "configure_napari_async_slicing", lambda enabled: enabled)
    monkeypatch.setattr(V.napari, "Viewer", lambda **_k: SimpleNamespace())
    monkeypatch.setattr(V, "install_application_icon", lambda _viewer: None)
    monkeypatch.setattr(V, "log_environment_diagnostics", lambda _viewer: None)
    monkeypatch.setattr(V, "ComparisonViewerController", fake_controller)
    with pytest.raises(_Stop):
        V.main()
    cfg = captured["datasets"]["ATERA"]
    assert list(captured["datasets"]) == ["ATERA"]
    assert cfg == V.DatasetConfig(name="ATERA", zarr_path=atera_store, platform="ATERA")


# ---------------------------------------------------------------------------
# Loader panel: buttons and recents
# ---------------------------------------------------------------------------


def test_loader_has_one_button_per_platform(qapp):
    panel = _panel()
    assert list(panel._load_standalone_buttons) == ["MERSCOPE", "XENIUM", "ATERA"]
    assert panel._load_standalone_merscope_button.text() == "Load new standalone MERSCOPE dataset"
    assert panel._load_standalone_xenium_button.text() == "Load new standalone Xenium dataset"
    assert panel._load_standalone_atera_button.text() == "Load new standalone Atera dataset"


def test_atera_button_loads_atera_and_records_the_effective_platform(qapp, tmp_path, monkeypatch):
    calls = []
    result = {"value": None, "effective": None}

    def load(platform, path):
        calls.append((platform, path))
        return result["value"]

    panel = _panel(
        settings=_settings(tmp_path),
        load_standalone_callback=load,
        standalone_platform_callback=lambda: result["effective"],
    )
    store = tmp_path / "atera.zarr"
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(store))
    opened = []
    panel.dataset_open_requested.connect(lambda: opened.append(True))

    for value in (None, False):
        result["value"] = value
        panel._load_standalone_atera_button.click()
    assert calls == [("ATERA", str(store)), ("ATERA", str(store))]
    assert panel.recent_datasets == [] and opened == []

    result["value"] = True
    panel._load_standalone_atera_button.click()
    assert panel.recent_datasets[0] == {"platform": "ATERA", "path": str(store.absolute())}
    assert opened == [True]

    # The Xenium button on an Atera store records the platform the controller opened.
    result["effective"] = "ATERA"
    other = tmp_path / "relabelled.zarr"
    monkeypatch.setattr(panel, "_browse_zarr", lambda _title: str(other))
    panel._load_standalone_xenium_button.click()
    assert calls[-1] == ("XENIUM", str(other))
    assert panel.recent_datasets[0] == {"platform": "ATERA", "path": str(other.absolute())}


def test_recents_keep_atera_and_drop_unknown_platforms(qapp, tmp_path):
    settings = _settings(tmp_path)
    atera = tmp_path / "run" / "latest_spatialdata.zarr"
    settings.setValue(
        V.RECENT_DATASETS_SETTINGS_KEY,
        json.dumps(
            [
                {"platform": "ATERA", "path": str(atera)},
                {"platform": "FOO", "path": str(tmp_path / "foo.zarr")},
                {"platform": "XENIUM", "path": str(tmp_path / "x.zarr")},
            ]
        ),
    )
    opened = []
    panel = _panel(
        settings=settings,
        load_standalone_callback=lambda platform, path: opened.append((platform, path)) or True,
    )
    assert panel.recent_datasets == [
        {"platform": "ATERA", "path": str(atera)},
        {"platform": "XENIUM", "path": str(tmp_path / "x.zarr")},
    ]
    assert panel._recent_dataset_list.item(0).text() == "Atera — latest_spatialdata.zarr"
    panel._recent_dataset_list.setCurrentRow(0)
    panel._open_recent_dataset_button.click()
    assert opened == [("ATERA", str(atera))]

    panel.record_recent_dataset("ATERA", atera)
    assert _panel(settings=settings).recent_datasets[0] == {
        "platform": "ATERA",
        "path": str(atera.absolute()),
    }


def test_registration_mode_keeps_the_atera_button_enabled(qapp):
    from qtpy.QtWidgets import QWidget

    panel = _panel(registration_widget=QWidget(), load_registration_callback=lambda path: None)
    panel.set_registration_mode(True)
    assert panel._load_standalone_atera_button.isEnabled()


# ---------------------------------------------------------------------------
# Controller: configs and pixel transforms
# ---------------------------------------------------------------------------


def test_atera_config_ignores_the_xenium_spec(qapp, atera_store, tmp_path):
    xenium_spec = _xenium_spec(tmp_path)
    ctrl = _controller(xenium_spec_path=xenium_spec, merscope_transform_path=tmp_path / "m.csv")
    cfg = ctrl._build_dataset_config("ATERA", atera_store)
    assert cfg.platform == "ATERA"
    assert cfg.spec_path is None and cfg.xenium_spec_path is None
    assert cfg.merscope_transform_path is None
    _, spec_path = ctrl._resolve_optional_transform_paths("ATERA", cfg)
    assert spec_path == atera_store / "atera" / "experiment.spatial"
    x_tf, y_tf = ctrl._resolve_dataset_pixel_transform("ATERA", cfg)
    assert (x_tf[0], y_tf[1]) == (PS, PS)

    xenium_cfg = ctrl._build_dataset_config("XENIUM", atera_store)
    assert xenium_cfg.spec_path == xenium_spec


def test_atera_spec_override_is_used(qapp, atera_store, tmp_path):
    override = tmp_path / "override" / "experiment.spatial"
    override.parent.mkdir()
    override.write_text(json.dumps({"pixel_size": 0.25}))
    ctrl = _controller(atera_spec_path=override)
    cfg = ctrl._build_dataset_config("ATERA", atera_store)
    x_tf, y_tf = ctrl._resolve_dataset_pixel_transform("ATERA", cfg)
    assert (x_tf[0], y_tf[1]) == (0.25, 0.25)


def test_xenium_spec_resolution_is_unchanged(qapp, tmp_path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    (tmp_path / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.4}))
    ctrl = _controller()
    cfg = V.DatasetConfig(name="XENIUM", zarr_path=store)
    _, spec_path = ctrl._resolve_optional_transform_paths("XENIUM", cfg)
    assert spec_path == tmp_path / "experiment.xenium"
    x_tf, _ = ctrl._resolve_dataset_pixel_transform("XENIUM", cfg)
    assert x_tf[0] == 0.4

    explicit = _xenium_spec(tmp_path, 0.3)
    legacy = V.DatasetConfig(name="XENIUM", zarr_path=store, xenium_spec_path=explicit)
    assert ctrl._resolve_dataset_pixel_transform("XENIUM", legacy)[0][0] == 0.3


def test_atera_session_is_prepared_with_the_store_pixel_size(qapp, atera_store):
    ctrl = _controller()
    cfg = ctrl._build_dataset_config("ATERA", atera_store)
    ctrl.datasets = {"ATERA": cfg}
    session = ctrl._prepare_dataset_session("ATERA", cfg, Event())
    assert session.x_transform == (PS, 0.0, 0.0)
    assert session.y_transform == (0.0, PS, 0.0)
    assert "atera_cell_boundaries" in session.segmentation_keys


# ---------------------------------------------------------------------------
# Controller: standalone detection
# ---------------------------------------------------------------------------


def test_standalone_relabels_an_authoritative_atera_store(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    statuses = _statuses(ctrl)
    replaced = []
    monkeypatch.setattr(
        ctrl, "_replace_datasets", lambda datasets, initial: replaced.append((datasets, initial)) or True
    )
    assert ctrl.load_standalone_dataset("XENIUM", atera_store) is True
    datasets, initial = replaced[-1]
    assert initial == "ATERA" and list(datasets) == ["ATERA"]
    assert datasets["ATERA"].platform == "ATERA"
    assert ctrl.last_standalone_platform == "ATERA"
    assert "Detected Atera store; opening as ATERA." in statuses


def _run_real_standalone_load(ctrl, monkeypatch, platform, store):
    """Run load_standalone_dataset through _replace_datasets/load_dataset for real.

    Only the worker thread and the post-load background autoloads are removed;
    the metadata session is prepared and applied synchronously.
    """
    monkeypatch.setattr(V, "thread_worker", None)
    monkeypatch.setattr(ctrl, "_start_startup_autoload", lambda ds: None)
    monkeypatch.setattr(ctrl, "_schedule_canvas_visibility_repair", lambda ds: None)
    progress: list[tuple[str, str, bool]] = []
    ctrl.set_progress_callback(lambda key, text, active: progress.append((key, text, active)))
    return ctrl.load_standalone_dataset(platform, store), progress


def test_standalone_relabel_notice_survives_the_real_load(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    statuses = _statuses(ctrl)
    result, progress = _run_real_standalone_load(ctrl, monkeypatch, "XENIUM", atera_store)
    notice = "Detected Atera store; opening as ATERA."
    assert result is True
    assert ctrl.active_dataset == "ATERA"
    # The progress stage text shown while metadata loads carries the notice ...
    started = [text for key, text, active in progress if key == "dataset" and active]
    assert started and started[-1].startswith(notice)
    assert "Loading ATERA metadata" in started[-1]
    # ... and so does the status that is actually left on screen afterwards.
    assert statuses[-1].startswith(notice)
    assert "ATERA metadata loaded" in statuses[-1]
    # The notice is one-shot: a later reload does not repeat it.
    assert ctrl._pending_load_notice is None
    statuses.clear()
    ctrl.load_dataset("ATERA", force=True)
    assert statuses and not any(notice in text for text in statuses)


def test_standalone_relabel_notice_is_kept_on_a_failed_load(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    statuses = _statuses(ctrl)

    def boom(ds, cfg, cancel_token):
        raise RuntimeError("metadata unreadable")

    monkeypatch.setattr(ctrl, "_prepare_dataset_session", boom)
    result, _ = _run_real_standalone_load(ctrl, monkeypatch, "XENIUM", atera_store)
    assert result is False
    assert statuses[-1].startswith("Detected Atera store; opening as ATERA.")
    assert "Failed to load ATERA: metadata unreadable" in statuses[-1]
    assert ctrl._pending_load_notice is None


def test_standalone_load_without_relabel_has_no_notice(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    statuses = _statuses(ctrl)
    result, _ = _run_real_standalone_load(ctrl, monkeypatch, "ATERA", atera_store)
    assert result is True
    assert not any("Detected" in text for text in statuses)
    assert statuses[-1].startswith("ATERA metadata loaded")


def test_standalone_heuristic_store_keeps_the_requested_platform(qapp, atera_store, tmp_path, monkeypatch):
    store = _heuristic_only_copy(atera_store, tmp_path)
    ctrl = _controller()
    statuses = _statuses(ctrl)
    replaced = []
    monkeypatch.setattr(
        ctrl, "_replace_datasets", lambda datasets, initial: replaced.append((datasets, initial)) or True
    )
    assert ctrl.load_standalone_dataset("XENIUM", store) is True
    assert replaced[-1][1] == "XENIUM"
    assert ctrl.last_standalone_platform == "XENIUM"
    assert not any(text.startswith("Detected") for text in statuses)


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_standalone_p7513_like_stores_keep_their_platform(qapp, tmp_path, monkeypatch, platform):
    store = atera_stores.build_p7513_like_attrs_store(tmp_path, platform)
    ctrl = _controller()
    replaced = []
    monkeypatch.setattr(
        ctrl, "_replace_datasets", lambda datasets, initial: replaced.append(initial) or True
    )
    assert ctrl.load_standalone_dataset(platform, store) is True
    assert replaced == [platform]
    assert ctrl.last_standalone_platform == platform


def test_standalone_refuses_an_unknown_platform(qapp, tmp_path, monkeypatch):
    store = tmp_path / "plain.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    ctrl = _controller()
    statuses = _statuses(ctrl)

    def forbidden(*_a, **_k):
        raise AssertionError("an unknown platform must not replace the datasets")

    monkeypatch.setattr(ctrl, "_replace_datasets", forbidden)
    assert ctrl.load_standalone_dataset("FOO", store) is False
    assert "standalone viewing supports MERSCOPE, Xenium, Atera" in statuses[-1]


# ---------------------------------------------------------------------------
# Controller: paired refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", ["merscope", "xenium"])
def test_paired_load_refuses_an_atera_store_before_cancelling(qapp, atera_store, tmp_path, monkeypatch, slot):
    other = atera_stores.build_p7513_like_attrs_store(
        tmp_path, "XENIUM" if slot == "merscope" else "MERSCOPE"
    )
    ctrl = _controller()
    statuses = _statuses(ctrl)

    def forbidden(*_a, **_k):
        raise AssertionError("a refused paired load must not cancel or read anything")

    monkeypatch.setattr(ctrl, "_cancel_background_tasks", forbidden)
    monkeypatch.setattr(ctrl, "_confirm_leave_registration", forbidden)
    monkeypatch.setattr(V.sd, "read_zarr", forbidden)
    generation = ctrl._dataset_load_generation
    pair = (atera_store, other) if slot == "merscope" else (other, atera_store)
    assert ctrl.load_paired_dataset(*pair) is False
    assert ctrl._dataset_load_generation == generation
    assert statuses[-1].startswith(
        "Atera stores open standalone only; paired viewing supports MERSCOPE + Xenium"
    )


def test_paired_load_refuses_swapped_authoritative_stores(qapp, tmp_path, monkeypatch):
    merscope = atera_stores.build_p7513_like_attrs_store(tmp_path, "MERSCOPE")
    xenium = atera_stores.build_p7513_like_attrs_store(tmp_path, "XENIUM")
    ctrl = _controller()
    statuses = _statuses(ctrl)
    monkeypatch.setattr(
        ctrl, "_cancel_background_tasks", lambda: pytest.fail("must not cancel")
    )
    assert ctrl.load_paired_dataset(xenium, merscope) is False
    assert "is a Xenium store but was given as the MERSCOPE store" in statuses[-1]


def test_paired_backstops_still_reject_atera():
    with pytest.raises(ValueError):
        normalize_platform("ATERA")
    assert "ATERA" not in V.PAIRED_PLATFORMS


# ---------------------------------------------------------------------------
# Transcript gates
# ---------------------------------------------------------------------------


def test_open_gene_inspector_on_atera_builds_nothing(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    ctrl.datasets = {"ATERA": ctrl._build_dataset_config("ATERA", atera_store)}
    statuses = _statuses(ctrl)

    def forbidden(*_a, **_k):
        raise AssertionError("ATERA must not build eager transcripts or read the marker table")

    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", lambda _ds: True)
    monkeypatch.setattr(ctrl, "_resolve_gene_points_columns", forbidden)
    monkeypatch.setattr(V, "build_gene_point_groups", forbidden)
    monkeypatch.setattr(V, "resolve_cell_type_marker_reference", forbidden)
    monkeypatch.setattr(V, "make_transcript_cache_request", forbidden)
    assert ctrl.open_gene_inspector("ATERA") is None
    assert statuses[-1] == "Atera transcripts: lazy engine not available yet"
    assert ctrl._uses_lazy_cell_queries("ATERA")
    assert not ctrl._uses_lazy_cell_queries("MERSCOPE")
    assert not ctrl._uses_lazy_cell_queries("TEST")


def test_atera_cell_inspection_never_scans_transcripts(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    ctrl.datasets = {"ATERA": ctrl._build_dataset_config("ATERA", atera_store)}

    def forbidden(*_a, **_k):
        raise AssertionError("ATERA cell inspection must not resolve a whole-store scan")

    monkeypatch.setattr(ctrl, "_inspection_identity", forbidden)
    monkeypatch.setattr(ctrl, "_inspection_sdata", forbidden)
    monkeypatch.setattr(ctrl, "_resolve_points_columns_for", forbidden)
    index = ctrl._inspection_transcript_index(
        "ATERA", 1, None, ctrl._empty_cell_transcript_index()
    )
    assert index.transcripts_for(1)[1].size == 0


def test_merscope_label_cache_gate_follows_the_pixel_transform(qapp, atera_store):
    ctrl = _controller()
    ctrl.datasets = {
        "MERSCOPE": V.DatasetConfig(name="MERSCOPE", zarr_path=Path("/m.zarr")),
        "ATERA": ctrl._build_dataset_config("ATERA", atera_store),
    }
    ctrl.active_dataset = "MERSCOPE"
    assert ctrl._active_pixel_transform_is_merscope_affine()
    for name in ("ATERA", "XENIUM", None):
        ctrl.active_dataset = name
        assert not ctrl._active_pixel_transform_is_merscope_affine()
