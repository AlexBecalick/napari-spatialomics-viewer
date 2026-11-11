"""ATERA segmentation, cell types, UMAP, statistics and inspector wiring (milestone M3).

Everything runs on the synthetic store from ``atera_stores``; no test reads
``/Users/becalia/data``. Viewers are napari ``ViewerModel`` instances or plain
stubs, never vispy canvases.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from threading import Event
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atera_stores
import numpy as np
import pytest

from napari_compare_xenium_merscope import cell_umap as C
from napari_compare_xenium_merscope import utils as U
from napari_compare_xenium_merscope import viewer as V

PS = 0.5
ALIASES = ("atera_cell_boundaries_labels", "atera_nucleus_labels")


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


@pytest.fixture
def store_copy(atera_store, tmp_path) -> Path:
    """A writable per-test copy (the viewer may add its own derived caches)."""
    dst = tmp_path / "copy" / atera_store.name
    shutil.copytree(atera_store, dst)
    return dst


@pytest.fixture(autouse=True)
def _private_app_data(tmp_path, monkeypatch):
    app_data = tmp_path / "app-data-root"
    app_data.mkdir()
    monkeypatch.setattr(V.QStandardPaths, "writableLocation", lambda *_a: str(app_data))


class _StubViewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []
        self.added_images: list[dict] = []
        self.grid = SimpleNamespace(enabled=False)

    def add_image(self, data, **kwargs):
        self.added_images.append({"data": data, **kwargs})
        return SimpleNamespace(name=kwargs.get("name"))

    def reset_view(self):
        pass


def _args(**overrides):
    defaults = dict(
        background_io_workers=1,
        session_cache_gb=0,
        image_pyramid_downsample=4,
        overwrite_derived_caches=False,
        overwrite_labels=False,
        skip_images=True,
        skip_transcripts=True,
        segmentation_source="shapes",
        hide_images=False,
        hide_shapes=False,
        image_opacity=1.0,
        visible_channels=None,
        shape_opacity=0.5,
        label_contour_width=1,
        merscope_transform_path=None,
        xenium_spec_path=None,
        atera_spec_path=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _controller(viewer=None, **overrides):
    return V.ComparisonViewerController(viewer or _StubViewer(), {}, _args(**overrides))


def _activate(ctrl, store, platform="ATERA"):
    """Prepare and apply the metadata session synchronously (no autoloads)."""
    cfg = ctrl._build_dataset_config(platform, store)
    ctrl.datasets = {platform: cfg}
    session = ctrl._prepare_dataset_session(platform, cfg, Event())
    ctrl.active_dataset = platform
    ctrl._active_sdata = session.sdata
    ctrl._active_images_sdata = session.images_sdata
    ctrl._x_transform = session.x_transform
    ctrl._y_transform = session.y_transform
    ctrl._segmentation_keys = list(session.segmentation_keys)
    ctrl._dataset_sessions[platform] = session
    return session


def _alias_snapshot(store: Path) -> dict[str, bytes]:
    return {key: (store / "labels" / key / "zarr.json").read_bytes() for key in ALIASES}


def _payload_snapshot(store: Path) -> dict[str, bytes]:
    """Every element file under labels/, shapes/, points/, tables/ and images/.

    The viewer's own ``_napari_compare_*`` caches are excluded, and so is each
    element-type group's ``zarr.json`` (SpatialData lists a new derived cache
    there); the elements themselves must stay byte-identical.
    """
    out = {}
    for group in ("labels", "shapes", "points", "tables", "images"):
        for path in sorted((store / group).rglob("*")):
            rel = path.relative_to(store)
            if len(rel.parts) == 2:
                continue
            if path.is_file() and not any(U.is_derived_cache_key(part) for part in rel.parts):
                out[str(rel)] = path.read_bytes()
    return out


# ---------------------------------------------------------------------------
# utils: sources, assignments, UMAP, quantification
# ---------------------------------------------------------------------------


def test_original_source_resolves_the_atera_cell_boundaries(atera_store):
    assert U.cell_type_shape_key_for_source(atera_store, "original") == "atera_cell_boundaries"
    assert U.cell_type_shape_key_for_source(atera_store, "proseg") is None
    assert U.cell_type_shape_key_for_source(atera_store, "pciseq_original") is None


def test_original_source_never_falls_back_to_the_merxen_schema(atera_store, monkeypatch):
    # Only exact candidate keys resolve: without atera_cell_boundaries in the
    # candidates, the merxen_schema registration of it is not consulted.
    original = U.cell_type_source("original")
    without_atera = dataclasses.replace(
        original,
        mask_shape_keys=tuple(k for k in original.mask_shape_keys if k != "atera_cell_boundaries"),
    )
    monkeypatch.setitem(U._CELL_TYPE_SOURCE_BY_KEY, "original", without_atera)
    assert U.cell_type_shape_key_for_source(atera_store, "original") is None


def test_cell_type_assignments_join_the_uint64_instance_ids(atera_store):
    assignments = U.load_cell_type_assignments(atera_store, "original")
    assert assignments is not None
    assert assignments.instance_key == "instance_id"
    assert assignments.shape_key == "atera_cell_boundaries"
    assert assignments.table_key == "table_original_clustering_squidpy"
    ids = [int(v) for v in assignments.cell_ids]
    assert ids == [c[0] for c in atera_stores.CELLS if c[0] != atera_stores.CLUSTERING_DROPPED_CELL]
    assert atera_stores.CLUSTERING_DROPPED_CELL not in ids  # unannotated, stays transparent
    assert set(assignments.levels) >= {"broad", "fine"}


def test_cell_umap_reads_x_umap(atera_store):
    assert C.cell_umap_available(atera_store, "original")
    embedding = C.load_cell_umap(atera_store, "original")
    assert embedding is not None
    assert embedding.coords.shape == (len(atera_stores.CELLS) - 1, 2)
    assert int(embedding.cell_ids[0]) == 1
    assert C.UMAP_PLATFORM_HEADINGS["ATERA"] == "Atera"


def test_quantification_discovery_finds_the_atera_table(atera_store):
    found = U.discover_mask_quantification(atera_store)
    assert found == U.MaskQuantification(
        table_key="table_atera_cell_boundaries_image_quantification",
        shape_key="atera_cell_boundaries",
        label_key="atera_cell_boundaries_labels",  # the alias, not uns mask_label_key 'cell_labels'
    )
    assert not found.is_cellpose
    assert found.display_name == "Per-cell image statistics (atera_cell_boundaries)"
    assert not U.cellpose_quantification_table_available(atera_store)
    features = U.cellpose_quantification_features(atera_store, found.table_key)
    assert sorted(set(U.cellpose_quantification_channel_labels(features))) == sorted(atera_stores.CHANNELS)
    values = U.load_cellpose_quantification_values(
        atera_store, "ATP1A1/CD45/E-Cadherin", "mean", table_key=found.table_key
    )
    assert values.label_ids.tolist() == [c[0] for c in atera_stores.CELLS]
    means = dict(U.cellpose_cell_mean_intensities(atera_store, 3, found.table_key))
    assert set(means) == set(atera_stores.CHANNELS)
    assert all(value is not None for value in means.values())


def test_quantification_discovery_prefers_cellpose(tmp_path):
    import zarr

    root = zarr.open_group(str(tmp_path / "c.zarr"), mode="w")
    root.create_group("labels").create_group(U.CELLPOSE_LABEL_KEY)
    table = root.create_group("tables").create_group(U.CELLPOSE_QUANTIFICATION_TABLE_KEY)
    table.create_group("obs").create_array("label_id", data=np.array([1, 2]))
    var = table.create_group("var")
    for key in ("image_key", "channel", "statistic"):
        var.create_array(key, data=np.array(["a", "b"], dtype="U8"))
    table.create_array("X", data=np.zeros((2, 2)))
    assert U.discover_mask_quantification(tmp_path / "c.zarr") == U.CELLPOSE_MASK_QUANTIFICATION
    assert U.discover_mask_quantification(tmp_path / "missing.zarr") is None
    assert U.discover_mask_quantification(None) is None


def test_vendor_label_markers_hide_their_vendor_elements(atera_store):
    markers = U.vendor_label_alias_markers(atera_store)
    assert sorted(markers) == sorted(ALIASES)
    assert U.hidden_vendor_label_keys(markers) == {"cell_labels", "nucleus_labels"}
    assert U.is_vendor_label_marker(markers["atera_cell_boundaries_labels"])
    assert not U.is_vendor_label_marker({"version": 2, "complete": True})


def test_atera_nucleus_displays_as_nucleus():
    assert U.make_layer_name("ATERA", "labels", "atera_nucleus_labels") == "Segmentation | Nucleus"
    assert (
        U.make_layer_name("ATERA", "labels", "atera_cell_boundaries_labels")
        == "Segmentation | Atera cell boundaries"
    )


# ---------------------------------------------------------------------------
# Session smoke test
# ---------------------------------------------------------------------------


def test_atera_session_smoke(qapp, atera_store):
    ctrl = _controller()
    session = _activate(ctrl, atera_store)
    assert session.x_transform == (PS, 0.0, 0.0)
    assert session.y_transform == (0.0, PS, 0.0)
    assert session.segmentation_keys == ["atera_cell_boundaries", "atera_nucleus"]
    assert ctrl._startup_segmentation_keys() == ["atera_cell_boundaries"]
    assert session.cellpose_values_available
    assert session.quantification.shape_key == "atera_cell_boundaries"
    assert session.cellpose_statistics == list(atera_stores.QUANT_STATISTICS)
    assert session.cell_type_available["original"] is True
    assert session.umap_available["original"] is True
    assert not any(session.cell_type_available[k] for k in session.cell_type_available if k != "original")

    stats = ctrl._add_image_layers(
        "ATERA", session.images_sdata, session.x_transform, session.y_transform, build_cache=False
    )
    assert stats["layers"] == 4
    added = {entry["name"]: entry["colormap"] for entry in ctrl.viewer.added_images}
    assert set(added) == {f"Image | {channel}" for channel in atera_stores.CHANNELS}
    assert added["Image | DAPI"] == "blue"
    assert added["Image | ATP1A1/CD45/E-Cadherin"] == "magenta"
    assert added["Image | alphaSMA/Vimentin"] == "yellow"
    assert added["Image | 18S"] == "green"


def test_embedded_density_is_discovered_and_auto_loaded_as_a_red_image(qapp, store_copy, monkeypatch):
    import spatialdata as sd
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import get_transformation

    images = sd.read_zarr(store_copy, selection=("images",))
    morphology = images.images["morphology_focus"]
    base = U.get_scale0_dataarray(morphology)
    images.images["atera_transcript_density"] = Image2DModel.parse(
        np.ones((1, *base.shape[-2:]), np.float32), dims=("c", "y", "x"),
        c_coords=["transcript_density"],
        transformations=get_transformation(morphology, get_all=True),
    )
    images.write_element("atera_transcript_density")
    images.write_consolidated_metadata()
    ctrl = _controller(skip_images=False)
    session = _activate(ctrl, store_copy)
    assert ("atera_transcript_density", "transcript_density") in session.image_channels
    ctrl._add_image_layers("ATERA", session.images_sdata,
        session.x_transform, session.y_transform, build_cache=False)
    density = next(entry for entry in ctrl.viewer.added_images
                   if entry["name"] == "Image | Transcript density")
    assert density["colormap"] == "red"
    assert density["visible"]
    loads = []
    ctrl._image_keys = session.image_keys
    monkeypatch.setattr(ctrl, "load_images_on_demand", loads.append)
    monkeypatch.setattr(ctrl, "_startup_segmentation_keys", lambda: [])
    ctrl._start_startup_autoload("ATERA")
    assert loads == ["ATERA"]


def test_atera_labels_mode_lists_each_raster_once(qapp, atera_store):
    ctrl = _controller(segmentation_source="labels")
    session = _activate(ctrl, atera_store)
    assert session.segmentation_keys == list(ALIASES)  # no vendor cell_labels/nucleus_labels
    assert ctrl._startup_segmentation_keys() == ["atera_cell_boundaries_labels"]


def test_mosaik_startup_keys_are_unchanged(qapp):
    ctrl = _controller()
    ctrl.active_dataset = "XENIUM"
    ctrl.datasets = {"XENIUM": V.DatasetConfig(name="XENIUM", zarr_path=Path("/x.zarr"))}
    ctrl._segmentation_keys = ["MOSAIK_cellpose", "xenium_cell_boundaries", "atera_cell_boundaries"]
    assert ctrl._startup_segmentation_keys() == ["MOSAIK_cellpose"]
    ctrl._segmentation_keys = ["xenium_cell_boundaries"]
    assert ctrl._startup_segmentation_keys() == []


# ---------------------------------------------------------------------------
# Vendor alias safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["shapes", "labels"])
def test_standalone_open_never_touches_the_vendor_aliases(qapp, store_copy, monkeypatch, mode):
    from napari.components import ViewerModel

    before_aliases = _alias_snapshot(store_copy)
    before_payload = _payload_snapshot(store_copy)
    ctrl = _controller(ViewerModel(), segmentation_source=mode)
    monkeypatch.setattr(V, "thread_worker", None)
    monkeypatch.setattr(ctrl, "_schedule_canvas_visibility_repair", lambda ds: None)
    assert ctrl.load_standalone_dataset("ATERA", store_copy) is True
    names = [layer.name for layer in ctrl.viewer.layers]
    assert "Segmentation | Atera cell boundaries" in names
    assert "Segmentation | Nucleus" not in names  # nucleus is off by default

    # The cell-type overlay and statistics join on the alias as stored.
    state = ctrl._cell_type_state("ATERA")
    assert state.segmentation == "original"
    assert ctrl._ensure_label_key_for_segmentation("original") == "atera_cell_boundaries_labels"
    quantification = ctrl._dataset_quantification("ATERA")
    assert ctrl._ensure_cellpose_label_key(quantification=quantification) == "atera_cell_boundaries_labels"

    assert _alias_snapshot(store_copy) == before_aliases
    assert _payload_snapshot(store_copy) == before_payload
    listed = json.loads((store_copy / "labels" / "zarr.json").read_text())["attributes"]["ome"]["labels"]
    assert set(ALIASES) | {"cell_labels", "nucleus_labels"} <= set(listed)
    for key in ALIASES + ("cell_labels", "nucleus_labels"):
        assert (store_copy / "labels" / key / "s0").is_dir() or (store_copy / "labels" / key).is_dir()


def test_vendor_alias_writes_are_refused(qapp, store_copy):
    ctrl = _controller(overwrite_labels=True)
    _activate(ctrl, store_copy)
    before = _alias_snapshot(store_copy)
    key = "atera_cell_boundaries_labels"
    with pytest.raises(ValueError, match="vendor"):
        ctrl._discard_label_cache_before_write(key, "atera_cell_boundaries")
    with pytest.raises(ValueError, match="vendor"):
        ctrl._mark_label_cache_complete(key, "atera_cell_boundaries", (80, 80), (80, 80))
    with pytest.raises(ValueError, match="vendor"):
        ctrl._rasterize_label_payload("atera_cell_boundaries", key, (80, 80), (80, 80), np.eye(3))
    # Even --overwrite-labels resolves the alias instead of rebuilding it.
    assert ctrl.ensure_label_for_shape_key("atera_cell_boundaries") == key
    assert (store_copy / "labels" / key).is_dir()
    assert _alias_snapshot(store_copy) == before


def test_stale_vendor_alias_displays_the_vendor_labels(qapp, store_copy):
    atera_stores.update_zarr_attributes(
        store_copy / "labels" / "atera_cell_boundaries_labels",
        {
            V.LABEL_CACHE_ATTR: {
                **U.vendor_label_alias_markers(store_copy)["atera_cell_boundaries_labels"],
                "complete": False,
            }
        },
    )
    before = _alias_snapshot(store_copy)
    ctrl = _controller()
    _activate(ctrl, store_copy)
    assert ctrl.ensure_label_for_shape_key("atera_cell_boundaries") == "cell_labels"
    assert _alias_snapshot(store_copy) == before


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("overwrite_labels", [False, True])
def test_missing_shape_never_writes_a_label_element(qapp, store_copy, overwrite_labels):
    """A shape the store lacks raises before any discard or empty-element write.

    Regression: an absent ``merscope_cell_boundaries`` once left a marker-less,
    all-background ``merscope_cell_boundaries_labels`` shell (and a duplicate
    ``ome.labels`` entry per attempt) in an Atera store.
    """
    ctrl = _controller(overwrite_labels=overwrite_labels)
    _activate(ctrl, store_copy)
    labels_dir = store_copy / "labels"
    before = _tree_snapshot(labels_dir)
    for _attempt in range(2):
        with pytest.raises(KeyError, match="merscope_cell_boundaries"):
            ctrl.ensure_label_for_shape_key("merscope_cell_boundaries")
    assert _tree_snapshot(labels_dir) == before
    assert not (labels_dir / "merscope_cell_boundaries_labels").exists()
    assert "merscope_cell_boundaries_labels" not in ctrl._active_sdata.labels


def test_shape_only_in_store_is_loaded_before_the_label_build(qapp, store_copy, monkeypatch):
    """A shape missing from memory but present in the store still builds."""
    ctrl = _controller()
    _activate(ctrl, store_copy)
    shape_key = "atera_nucleus"
    if shape_key in ctrl._active_sdata.shapes:
        del ctrl._active_sdata.shapes[shape_key]
    calls = []
    monkeypatch.setattr(ctrl, "_discard_label_cache_before_write", lambda *a: calls.append("discard"))
    monkeypatch.setattr(ctrl, "_write_empty_label_element", lambda **k: calls.append("write"))
    monkeypatch.setattr(ctrl, "_rasterize_label_payload", lambda **k: calls.append("raster") or 0)
    monkeypatch.setattr(ctrl, "_mark_label_cache_complete", lambda *a: None)
    monkeypatch.setattr(ctrl, "_label_key_for_shape_key", lambda key: "nucleus_rebuilt_labels")
    monkeypatch.setattr(ctrl, "_refresh_label_key_from_store", lambda *a, **k: False)
    monkeypatch.setattr(
        ctrl, "_image_grid_for_labels", lambda: ((8, 8), (8, 8), np.eye(3), None)
    )
    assert ctrl.ensure_label_for_shape_key(shape_key) == "nucleus_rebuilt_labels"
    assert shape_key in ctrl._active_sdata.shapes
    assert calls == ["discard", "write", "raster"]


def test_viewer_generated_label_caches_are_still_rebuilt(qapp, tmp_path):
    """A non-vendor marker keeps the existing rebuild behaviour."""
    ctrl = _controller()
    ctrl.active_dataset = "XENIUM"
    store = tmp_path / "x.zarr"
    (store / "labels" / "MOSAIK_cellpose_labels").mkdir(parents=True)
    ctrl.datasets = {"XENIUM": V.DatasetConfig(name="XENIUM", zarr_path=store)}
    assert ctrl._vendor_label_marker("MOSAIK_cellpose_labels") is None
    ctrl._refuse_vendor_label_write("MOSAIK_cellpose_labels", "rasterize into")  # no raise


# ---------------------------------------------------------------------------
# Inspector shape key and nucleus clicks
# ---------------------------------------------------------------------------


def test_cell_inspector_shape_key_is_the_atera_cell_boundaries(qapp, atera_store):
    ctrl = _controller()
    _activate(ctrl, atera_store)
    assert ctrl._cell_inspector_shape_key() == "atera_cell_boundaries"
    assert ctrl._inspection_source("ATERA") == "original"
    # Also when the selected source has no mask on this store.
    ctrl._cell_type_state("ATERA").segmentation = "proseg"
    assert ctrl._cell_inspector_shape_key() == "atera_cell_boundaries"


def test_cell_inspector_shape_key_is_unchanged_for_mosaik_platforms(qapp):
    for platform in ("MERSCOPE", "XENIUM"):
        ctrl = _controller()
        ctrl.active_dataset = platform
        ctrl.datasets = {platform: V.DatasetConfig(name=platform, zarr_path=Path("/nope.zarr"))}
        ctrl._active_sdata = SimpleNamespace(
            shapes={"cell_boundaries": None, "MOSAIK_cellpose": None, "MOSAIK_proseg": None},
            attrs={},
        )
        ctrl._cell_type_states[platform] = V.CellTypeOverlayState(
            dataset=platform, store_path=str(Path("/nope.zarr").absolute()), segmentation="pciseq_x"
        )
        assert ctrl._cell_inspector_shape_key() == "MOSAIK_proseg"


class _Layer:
    def __init__(self, name, visible=True):
        self.name = name
        self.visible = visible


def _click(x_um, y_um):
    return SimpleNamespace(position=(y_um, x_um))


def test_nucleus_click_selects_its_cell_by_cell_instance_id(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    _activate(ctrl, atera_store)
    ctrl.viewer.layers.append(_Layer("Segmentation | Nucleus"))

    def forbidden(*_a, **_k):
        raise AssertionError("a nucleus click must map through cell_instance_id, not overlap")

    monkeypatch.setattr(ctrl, "_match_inspection_geometry", forbidden)
    # Nucleus 8 straddles cell 8's right edge (x = 37 um): click the part outside it.
    picked = ctrl._pick_cell_from_event(_click(38.0, 36.0))
    assert picked is not None and int(picked[0]) == 8
    # The selected cell comes from the cell_instance_id column, not the
    # nucleus's own instance id: remap nucleus 8 to cell 4 and click again.
    nuclei = ctrl._active_sdata.shapes["atera_nucleus"]
    assert 4 in {c[0] for c in atera_stores.CELLS}
    nuclei.loc[8, "cell_instance_id"] = 4
    picked = ctrl._pick_cell_from_event(_click(38.0, 36.0))
    assert picked is not None and int(picked[0]) == 4
    nuclei.loc[8, "cell_instance_id"] = 8
    # Inside a cell, the cell itself wins.
    assert int(ctrl._pick_cell_from_event(_click(5.0, 5.0))[0]) == 1
    # Outside every cell and nucleus: nothing.
    assert ctrl._pick_cell_from_event(_click(39.5, 1.0)) is None


def test_nucleus_without_a_cell_selects_nothing(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    _activate(ctrl, atera_store)
    ctrl.viewer.layers.append(_Layer("Segmentation | Nucleus"))
    nuclei = ctrl._active_sdata.shapes["atera_nucleus"]
    nuclei.loc[8, "cell_instance_id"] = 0
    monkeypatch.setattr(
        ctrl, "_match_inspection_geometry", lambda *_a: pytest.fail("no overlap fallback")
    )
    assert ctrl._pick_cell_from_event(_click(38.0, 36.0)) is None
    assert ctrl._nucleus_cell_join_key("atera_nucleus", "atera_cell_boundaries") == "cell_instance_id"
    assert ctrl._nucleus_cell_join_key("atera_nucleus", "MOSAIK_proseg") is None


# ---------------------------------------------------------------------------
# Statistics overlay, inspector intensities and UMAP
# ---------------------------------------------------------------------------


def test_statistic_overlay_uses_the_atera_quantification(qapp, store_copy, monkeypatch):
    from napari.components import ViewerModel

    before = _alias_snapshot(store_copy)
    ctrl = _controller(ViewerModel())
    _activate(ctrl, store_copy)
    published = []
    ctrl.set_cellpose_value_options_callback(lambda *args: published.append(args))
    ctrl._publish_cellpose_value_options()
    assert published[-1][2] is True
    assert sorted(published[-1][0]) == sorted(atera_stores.CHANNELS)
    statuses = []
    ctrl.set_status_callback(statuses.append)
    monkeypatch.setattr(V, "thread_worker", None)
    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", lambda *_a: True)
    monkeypatch.setattr(ctrl, "_sync_umap_from_statistic", lambda *_a: None)
    ctrl.load_cellpose_value_overlay("ATERA", "DAPI", "mean", "viridis")
    names = [layer.name for layer in ctrl.viewer.layers]
    assert names == ["ATERA | cell_values | atera_cell_boundaries | DAPI mean"], statuses
    assert any("Per-cell image statistics (atera_cell_boundaries) values" in s for s in statuses)
    assert _alias_snapshot(store_copy) == before


def test_inspector_reads_atera_stored_means(qapp, atera_store):
    ctrl = _controller()
    _activate(ctrl, atera_store)
    gdf = ctrl._active_sdata.shapes["atera_cell_boundaries"]
    cell_id, geometry = ctrl._shape_for_cell_id(gdf, 3)
    intensities, matched = ctrl._inspection_intensities("ATERA", cell_id, geometry, [])
    assert int(matched) == 3
    assert [name for name, _ in intensities] == list(atera_stores.CHANNELS)
    assert all(value is not None for _, value in intensities)


def test_umap_statistics_join_the_original_source(qapp, atera_store):
    ctrl = _controller()
    _activate(ctrl, atera_store)
    ctrl._publish_umap_options()
    state = ctrl._umap_state()
    assert state.statistics_available
    assert state.statistics_shape_key == "atera_cell_boundaries"
    assert state.available_sources["original"] is True
    assert ctrl._umap_statistic_source(state) == "original"


def test_umap_statistic_source_defaults_to_cellpose():
    state = V.CellUmapState(dataset="XENIUM")
    state.statistics_available = True
    state.available_sources = {"cellpose": True, "original": True}
    assert state.statistics_shape_key == U.CELLPOSE_SHAPE_KEY
    assert V.ComparisonViewerController._umap_statistic_source(state) == "cellpose"


# ---------------------------------------------------------------------------
# Capabilities: cortical depth and object annotations
# ---------------------------------------------------------------------------


def _panel():
    callback = lambda *args, **kwargs: None  # noqa: E731
    return V.ViewerControlPanel(
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


def test_atera_explains_why_pciseq_sources_are_not_offered(qapp, atera_store, monkeypatch):
    ctrl = _controller()
    session = _activate(ctrl, atera_store)
    pciseq_keys = [s.key for s in V.CELL_TYPE_SOURCES if getattr(s, "method", "") == "pciseq"]
    assert pciseq_keys
    # Even a session that (wrongly) reports pciSeq annotations must not offer them.
    for key in pciseq_keys:
        session.cell_type_available[key] = True
    monkeypatch.setattr(ctrl, "_start_cell_annotation_prefetch", lambda *_a: None)
    widget = V.CellTypeWidget(*(lambda *_a: None,) * 7)
    ctrl._cell_type_widget = widget
    statuses = []
    widget.status_message.connect(statuses.append)
    ctrl._publish_cell_type_options()

    offered = [widget._seg_combo.itemData(i) for i in range(widget._seg_combo.count())]
    assert "original" in offered
    assert not any(key in offered for key in pciseq_keys)
    reason = "pciSeq cell types are not available for Atera stores"
    assert widget._seg_combo.toolTip().startswith(reason)
    assert reason in statuses[-1]

    # A MERSCOPE dataset has no such explanation.
    assert ctrl._unsupported_tool_reason("MERSCOPE", "pciseq") is None


def test_atera_disables_cortical_depth_and_object_tools(qapp, atera_store):
    panel = _panel()
    ctrl = _controller()
    ctrl.set_dataset_capabilities_callback(panel.set_dataset_capabilities)
    _activate(ctrl, atera_store)
    ctrl._publish_dataset_capabilities()
    tools = panel._cortical_depth_tools() + panel._object_annotation_tools()
    assert not any(widget.isEnabled() for widget in tools)
    assert "Atera" in panel._create_annotations_button.toolTip()
    assert "Atera" in panel._create_object_annotations_button.toolTip()

    statuses = []
    ctrl.set_status_callback(statuses.append)
    ctrl.create_cortical_depth_annotation_layers("ATERA")
    assert statuses[-1].startswith("Cortical-depth annotations are not available for Atera")
    with pytest.raises(RuntimeError, match="Distance-from-object"):
        ctrl.create_distance_object_annotation_layer("ATERA", "vessel")
    with pytest.raises(RuntimeError, match="Distance-from-object"):
        ctrl.load_distance_object_annotations("ATERA", Path("objects.geojson"))

    # A MERSCOPE dataset re-enables them; paired mode still disables them.
    ctrl.datasets = {"MERSCOPE": V.DatasetConfig(name="MERSCOPE", zarr_path=Path("/m.zarr"))}
    ctrl.active_dataset = "MERSCOPE"
    ctrl._publish_dataset_capabilities()
    assert all(widget.isEnabled() for widget in tools)
    assert panel._create_annotations_button.toolTip() == ""
    panel.set_paired_view_mode("side-by-side")
    assert not any(widget.isEnabled() for widget in tools)
    panel.set_paired_view_mode(None)
    assert all(widget.isEnabled() for widget in tools)
