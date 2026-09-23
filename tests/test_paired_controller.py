"""Controller-level tests for paired MERSCOPE/Xenium presentation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.paired_data import (
    DatasetElementProfile,
    PairedDataContract,
    SegmentationProfile,
)
from napari_compare_xenium_merscope.paired_views import (
    LayerIdentity,
    LayerSlotKey,
    PairedLayerRegistry,
    ViewMode,
)


class _Emitter:
    def __init__(self, source):
        self.source = source
        self.callbacks = []

    def connect(self, callback, **_kwargs):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def __call__(self):
        event = SimpleNamespace(source=self.source)
        for callback in list(self.callbacks):
            callback(event)


class _Layer:
    def __init__(self, *, visible=True):
        self.name = ""
        self.metadata = {}
        self._visible = bool(visible)
        self.events = SimpleNamespace(visible=_Emitter(self))

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        self.events.visible()


class _LayerList(list):
    @contextmanager
    def batched_update(self):
        yield

    def move(self, source, destination):
        layer = self.pop(source)
        if destination > source:
            destination -= 1
        self.insert(destination, layer)


class _Grid:
    def __init__(self):
        self.enabled = False
        self.shape = (-1, -1)
        self.stride = 1
        self.spacing = 0.0


class _Viewer:
    def __init__(self):
        self.layers = _LayerList()
        self.grid = _Grid()


def _placeholder(_identity):
    return _Layer(visible=False)


def _contract(affine) -> PairedDataContract:
    profile_kwargs = dict(
        points_key="transcripts",
        image_keys=(),
        image_pyramid_keys={},
        segmentations=(),
        coordinate_system="global",
        pixel_to_world_affine=tuple(tuple(float(value) for value in row) for row in affine),
    )
    return PairedDataContract(
        pair_id="pair-1",
        coordinate_system="global",
        fixed_width=100,
        fixed_height=80,
        transform_fingerprint="transform",
        native_fingerprint="native",
        merscope=DatasetElementProfile(platform="MERSCOPE", **profile_kwargs),
        xenium=DatasetElementProfile(platform="XENIUM", **profile_kwargs),
        mappings={},
    )


def _controller_without_init() -> V.ComparisonViewerController:
    return V.ComparisonViewerController.__new__(V.ComparisonViewerController)


def test_paired_napari_affine_converts_full_xy_matrix_to_yx():
    xy_affine = np.asarray(
        [
            [2.0, 3.0, 5.0],
            [7.0, 11.0, 13.0],
            [0.0, 0.0, 1.0],
        ]
    )

    yx_affine = V.ComparisonViewerController._paired_napari_affine(
        _contract(xy_affine)
    )

    # Napari supplies and consumes coordinates as (y, x), so both the input and
    # output axes must be swapped.  Off-diagonal terms make a row-only swap fail.
    expected = np.asarray(
        [
            [11.0, 7.0, 13.0],
            [3.0, 2.0, 5.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(yx_affine, expected)
    np.testing.assert_allclose(
        yx_affine @ np.asarray([4.0, 6.0, 1.0]),
        np.asarray([99.0, 29.0, 1.0]),
    )


def test_switch_paired_view_preserves_layers_and_updates_grid_and_session():
    viewer = _Viewer()
    registry = PairedLayerRegistry(
        viewer,
        _placeholder,
        strict_layers=True,
        spacing=4.0,
    )
    for role, key in (("image", "DAPI"), ("segmentation", "proseg")):
        for platform in ("MERSCOPE", "XENIUM"):
            registry.register_layer(LayerIdentity(platform, role, key), _Layer())

    controller = _controller_without_init()
    controller.viewer = viewer
    controller._paired_registry = registry
    controller._paired_session = SimpleNamespace(
        view_mode=ViewMode.STACKED_OVERLAY
    )
    mode_notifications = []
    statuses = []
    redraws = []
    controller._paired_view_mode_callback = mode_notifications.append
    controller._set_status = statuses.append
    controller._force_canvas_redraw = lambda: redraws.append(True)
    original_layers = {id(layer): layer for layer in viewer.layers}

    assert controller.switch_paired_view("side")
    assert controller._paired_session.view_mode is ViewMode.SIDE_BY_SIDE
    assert viewer.grid.enabled
    assert viewer.grid.shape == (1, 2)
    assert viewer.grid.stride == registry.slot_count == 2
    assert [
        registry.identity_for_layer(layer).platform for layer in viewer.layers
    ] == ["MERSCOPE", "MERSCOPE", "XENIUM", "XENIUM"]
    assert {id(layer): layer for layer in viewer.layers} == original_layers

    assert controller.switch_paired_view("overlay")
    assert controller._paired_session.view_mode is ViewMode.STACKED_OVERLAY
    assert not viewer.grid.enabled
    assert {id(layer): layer for layer in viewer.layers} == original_layers
    assert mode_notifications == [
        ViewMode.SIDE_BY_SIDE,
        ViewMode.STACKED_OVERLAY,
    ]
    assert len(statuses) == len(redraws) == 2
    registry.assert_invariants()


def test_paired_gene_visibility_control_fans_out_to_both_platform_states():
    controller = _controller_without_init()
    controller._paired_session = SimpleNamespace(contract=_contract(np.eye(3)))
    merscope = SimpleNamespace(
        dataset="MERSCOPE",
        store=SimpleNamespace(gene_offsets={"Gad1": (2, 0, 1, 1)}),
        enabled_genes=set(),
    )
    xenium = SimpleNamespace(
        dataset="XENIUM",
        store=SimpleNamespace(gene_offsets={"Gad1": (7, 0, 1, 1)}),
        enabled_genes=set(),
    )
    controller._gene_inspector_states = {
        "MERSCOPE": merscope,
        "XENIUM": xenium,
    }
    rebuilds = []
    controller._schedule_gene_group_rebuild = (
        lambda state, group: rebuilds.append((state.dataset, group))
    )

    controller.set_gene_visible("MERSCOPE", "Gad1", True)

    assert merscope.enabled_genes == {"Gad1"}
    assert xenium.enabled_genes == {"Gad1"}
    assert rebuilds == [("MERSCOPE", 2), ("XENIUM", 7)]


def test_paired_cell_type_visibility_control_fans_out_to_both_platform_states():
    controller = _controller_without_init()
    controller._paired_session = object()
    controller.args = SimpleNamespace(shape_opacity=0.95)
    controller.datasets = {
        "MERSCOPE": SimpleNamespace(zarr_path=None),
        "XENIUM": SimpleNamespace(zarr_path=None),
    }
    merscope = V.CellTypeOverlayState(
        dataset="MERSCOPE",
        enabled={"proseg": {"broad": {"Neuron", "Glia"}}},
    )
    xenium = V.CellTypeOverlayState(
        dataset="XENIUM",
        enabled={"proseg": {"broad": {"Neuron", "Glia"}}},
    )
    controller._cell_type_states = {
        "MERSCOPE": merscope,
        "XENIUM": xenium,
    }
    recolored = []
    controller._recolor_cell_type_layer = (
        lambda state: recolored.append(state.dataset)
    )

    controller.set_cell_type_visible("MERSCOPE", "Neuron", False)

    assert merscope.enabled["proseg"]["broad"] == {"Glia"}
    assert xenium.enabled["proseg"]["broad"] == {"Glia"}
    assert recolored == ["MERSCOPE", "XENIUM"]


def test_paired_cell_type_layer_is_recolored_and_refreshed_after_initial_load():
    controller = _controller_without_init()
    controller.active_dataset = "PAIRED"
    controller._paired_session = object()
    state = V.CellTypeOverlayState(dataset="MERSCOPE", segmentation="proseg")
    state.layer_name = "MERSCOPE | Cell types | Proseg"
    layer = _Layer()
    layer.name = state.layer_name
    refreshed = []
    layer.refresh = lambda: refreshed.append(True)
    controller._paired_registry = SimpleNamespace(
        identity_for_layer=lambda candidate: (
            LayerIdentity("MERSCOPE", "cell_types", "proseg")
            if candidate is layer
            else None
        )
    )
    controller._get_layer_by_name = (
        lambda name: layer if name == state.layer_name else None
    )
    recolored = []
    controller._recolor_cell_type_layer = (
        lambda candidate: recolored.append(candidate.dataset) or True
    )
    redraws = []
    controller._force_canvas_redraw = lambda: redraws.append(True)

    controller._refresh_cell_type_layer_after_load(state, layer)

    assert recolored == ["MERSCOPE"]
    assert refreshed == [True]
    assert redraws == [True]


def test_paired_cell_type_refresh_waits_for_first_lazy_slice_completion(monkeypatch):
    controller = _controller_without_init()
    controller.active_dataset = "PAIRED"
    controller._paired_session = object()
    state = V.CellTypeOverlayState(dataset="MERSCOPE", segmentation="proseg")
    state.layer_name = "MERSCOPE | Cell types | Proseg"
    layer = _Layer()
    layer.name = state.layer_name
    layer.loaded = True
    layer.events.loaded = _Emitter(layer)
    refreshed = []
    layer.refresh = lambda: refreshed.append(True)
    controller._paired_registry = SimpleNamespace(
        identity_for_layer=lambda candidate: (
            LayerIdentity("MERSCOPE", "cell_types", "proseg")
            if candidate is layer
            else None
        )
    )
    controller._get_layer_by_name = (
        lambda name: layer if name == state.layer_name else None
    )
    recolored = []
    controller._recolor_cell_type_layer = (
        lambda candidate: recolored.append(candidate.dataset) or True
    )
    redraws = []
    controller._force_canvas_redraw = lambda: redraws.append(True)
    deferred = []
    monkeypatch.setattr(
        V.QTimer,
        "singleShot",
        lambda _delay, callback: deferred.append(callback),
    )

    controller._refresh_cell_type_layer_after_load(state, layer)

    assert recolored == ["MERSCOPE"]
    assert refreshed == [True]
    assert redraws == [True]
    assert layer.events.loaded.callbacks

    layer.loaded = False
    layer.events.loaded()
    assert deferred == []

    layer.loaded = True
    layer.events.loaded()
    assert len(deferred) == 1
    assert layer.events.loaded.callbacks == []

    deferred.pop()()
    assert recolored == ["MERSCOPE", "MERSCOPE"]
    assert refreshed == [True, True]
    assert redraws == [True, True]


def test_clear_paired_cell_selections_removes_managed_layers_and_dock_panels():
    controller = _controller_without_init()
    controller._paired_selected_cells = {
        "MERSCOPE": [("m1", object())],
        "XENIUM": [("x1", object())],
    }
    selection_key = LayerSlotKey("selection", "clicked-cells")
    controller._paired_registry = SimpleNamespace(
        slot_keys=(LayerSlotKey("image", "DAPI"), selection_key)
    )
    removed = []
    controller._remove_paired_slot = lambda key: removed.append(key) or ()
    cells = []
    controller._cell_info_overlay = SimpleNamespace(
        set_cells=lambda value: cells.append(value)
    )
    hidden = []
    controller._hide_cell_dock = lambda: hidden.append(True)

    controller._clear_paired_cell_selections()

    assert controller._paired_selected_cells == {
        "MERSCOPE": [],
        "XENIUM": [],
    }
    assert removed == [selection_key]
    assert cells == [[]]
    assert hidden == [True]


def test_hiding_paired_segmentation_clears_both_platform_selections():
    viewer = _Viewer()
    registry = PairedLayerRegistry(viewer, _placeholder)
    layer = registry.register_layer(
        LayerIdentity("MERSCOPE", "segmentation", "proseg"),
        _Layer(),
    )
    controller = _controller_without_init()
    controller._paired_session = SimpleNamespace(contract=_contract(np.eye(3)))
    controller._paired_registry = registry
    controller._cell_type_states = {}
    controller._paired_selected_cells = {
        "MERSCOPE": [("m1", object())],
        "XENIUM": [("x1", object())],
    }
    cleared = []
    controller._clear_paired_cell_selections = lambda: cleared.append(True)

    layer.visible = False
    controller._on_segmentation_visibility_changed(
        SimpleNamespace(source=layer)
    )

    assert cleared == [True]


def test_cell_type_source_resolves_schema_defined_branches_by_shape_mapping():
    contract = _contract(np.eye(3))
    merscope_segmentation = SegmentationProfile(
        logical_key="custom_moving_branch",
        shape_key="MOSAIK_cellpose_aligned_nonrigid",
        label_key="MOSAIK_cellpose_aligned_nonrigid_labels",
        points_key="transcripts_aligned_nonrigid",
        table_key=None,
    )
    xenium_segmentation = SegmentationProfile(
        logical_key="custom_fixed_branch",
        shape_key="MOSAIK_cellpose",
        label_key="MOSAIK_cellpose_labels",
        points_key="transcripts",
        table_key=None,
    )
    contract = replace(
        contract,
        merscope=replace(
            contract.merscope,
            segmentations=(merscope_segmentation,),
        ),
        xenium=replace(
            contract.xenium,
            segmentations=(xenium_segmentation,),
        ),
        mappings={
            "shapes": {
                "MOSAIK_cellpose": "MOSAIK_cellpose_aligned_nonrigid"
            }
        },
    )
    controller = _controller_without_init()
    controller._paired_session = SimpleNamespace(contract=contract)

    assert controller._paired_segmentation_for_cell_type(
        "MERSCOPE", "cellpose"
    ) is merscope_segmentation
    assert controller._paired_segmentation_for_cell_type(
        "XENIUM", "cellpose"
    ) is xenium_segmentation
