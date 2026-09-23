"""Focused tests for paired layer/grid presentation coordination."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from napari_compare_xenium_merscope.paired_views import (
    LayerIdentity,
    LayerSlotKey,
    PairedLayerRegistry,
    PairedViewInvariantError,
    ViewMode,
    attach_layer_identity,
    is_placeholder_layer,
    layer_identity_from_layer,
    platform_qualified_layer_name,
)


class _Emitter:
    def __init__(self, source):
        self.source = source
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def __call__(self):
        event = SimpleNamespace(source=self.source)
        for callback in list(self.callbacks):
            callback(event)


class _Layer:
    def __init__(self, name="", visible=True, metadata=None):
        self.name = name
        self.metadata = dict(metadata or {})
        self._visible = bool(visible)
        self.events = SimpleNamespace(visible=_Emitter(self))

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        # Napari's Layer.visible setter emits even if the value is unchanged.
        self.events.visible()


class _ImageLayer(_Layer):
    def __init__(self, name="", visible=True, metadata=None):
        super().__init__(name=name, visible=visible, metadata=metadata)
        self._opacity = 1.0
        self._blending = "translucent"
        self._contrast_limits = (0.0, 1.0)
        self._gamma = 1.0
        self._colormap = "gray"
        self._interpolation2d = "nearest"
        for property_name in (
            "opacity",
            "blending",
            "contrast_limits",
            "gamma",
            "colormap",
            "interpolation2d",
        ):
            setattr(self.events, property_name, _Emitter(self))

    @staticmethod
    def _display_property(name):
        def getter(self):
            return getattr(self, f"_{name}")

        def setter(self, value):
            setattr(self, f"_{name}", value)
            getattr(self.events, name)()

        return property(getter, setter)

    opacity = _display_property("opacity")
    blending = _display_property("blending")
    contrast_limits = _display_property("contrast_limits")
    gamma = _display_property("gamma")
    colormap = _display_property("colormap")
    interpolation2d = _display_property("interpolation2d")


class _LayerList(list):
    def __init__(self):
        super().__init__()
        self.batch_depth = 0
        self.batch_entries = 0
        self.moves = []

    @contextmanager
    def batched_update(self):
        self.batch_depth += 1
        self.batch_entries += 1
        try:
            yield
        finally:
            self.batch_depth -= 1

    def move(self, source, destination):
        self.moves.append((source, destination))
        item = self.pop(source)
        if destination > source:
            destination -= 1
        self.insert(destination, item)


class _Grid:
    def __init__(self):
        self.enabled = False
        self.shape = (-1, -1)
        self.stride = 1
        self.spacing = 0.0

    def position(self, index, nlayers):
        if not self.enabled:
            return (0, 0)
        group = (int(index) // abs(int(self.stride))) % (
            int(self.shape[0]) * int(self.shape[1])
        )
        return divmod(group, int(self.shape[1]))


class _Viewer:
    def __init__(self):
        self.layers = _LayerList()
        self.grid = _Grid()


def _placeholder(identity):
    return _Layer(name=f"placeholder {identity.platform}", visible=False)


def _registry(*, strict_layers=True):
    viewer = _Viewer()
    registry = PairedLayerRegistry(
        viewer,
        _placeholder,
        strict_layers=strict_layers,
        spacing=7,
    )
    return viewer, registry


def _identity(platform, role, key="", channel=""):
    return LayerIdentity(platform, role, key, channel)


def _real_layers(registry):
    return [layer for layer in registry.layers if not is_placeholder_layer(layer)]


def test_view_mode_aliases_and_invalid_value():
    assert ViewMode.coerce("side") is ViewMode.SIDE_BY_SIDE
    assert ViewMode.coerce("side_by_side") is ViewMode.SIDE_BY_SIDE
    assert ViewMode.coerce("overlay") is ViewMode.STACKED_OVERLAY
    assert ViewMode.coerce("stacked_overlay") is ViewMode.STACKED_OVERLAY
    assert ViewMode.coerce(ViewMode.STANDALONE) is ViewMode.STANDALONE

    with pytest.raises(ValueError, match="Unknown view mode"):
        ViewMode.coerce("flick-between-platforms")


def test_identity_metadata_and_platform_qualified_name_are_unambiguous():
    identity = _identity("merscope", "image", "morphology_focus", "DAPI")
    layer = _Layer(metadata={"upstream": "kept"})

    attach_layer_identity(layer, identity)

    assert layer.name == "MERSCOPE | Image | Morphology focus | DAPI"
    assert platform_qualified_layer_name(
        "xenium", "segmentation", "cellpose"
    ) == "XENIUM | Segmentation | Cellpose"
    assert layer.metadata["upstream"] == "kept"
    assert layer_identity_from_layer(layer) == identity
    assert not is_placeholder_layer(layer)


def test_one_real_layer_keeps_placeholder_out_of_layer_list_until_side_mode():
    viewer, registry = _registry()
    merscope = _Layer(visible=True)
    identity = _identity("MERSCOPE", "image", "warped", "DAPI")

    registry.register_layer(identity, merscope)

    xenium_placeholder = registry.layer_for(
        identity.for_platform("XENIUM"), include_placeholder=True
    )
    assert registry.slot_count == 1
    assert viewer.layers == [merscope]
    assert registry.layer_for(identity) is merscope
    assert xenium_placeholder is not None
    assert is_placeholder_layer(xenium_placeholder)
    assert not xenium_placeholder.visible

    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    assert viewer.layers == [merscope, xenium_placeholder]
    registry.assert_invariants()


def test_overlay_removes_missing_counterpart_placeholder_and_side_restores_it():
    viewer, registry = _registry()
    identity = _identity("MERSCOPE", "segmentation", "proseg")
    real = registry.register_layer(identity, _Layer(visible=True))
    placeholder = registry.counterpart(real, include_placeholder=True)

    registry.set_mode(ViewMode.STACKED_OVERLAY)

    assert viewer.layers == [real]
    assert placeholder not in viewer.layers
    assert registry.counterpart(real, include_placeholder=True) is placeholder
    registry.assert_invariants()

    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    assert viewer.layers == [real, placeholder]
    registry.assert_invariants()

    registry.set_mode(ViewMode.STACKED_OVERLAY)
    assert viewer.layers == [real]
    assert placeholder not in viewer.layers
    registry.assert_invariants()


def test_side_by_side_uses_equal_canonical_platform_blocks():
    viewer, registry = _registry()
    registrations = [
        (_identity("XENIUM", "transcripts", "aligned", "disc"), 400),
        (_identity("MERSCOPE", "segmentation", "proseg"), 300),
        (_identity("XENIUM", "image", "morphology", "DAPI"), 100),
        (_identity("MERSCOPE", "transcripts", "aligned", "disc"), 400),
        (_identity("XENIUM", "segmentation", "proseg"), 300),
        (_identity("MERSCOPE", "image", "morphology", "DAPI"), 100),
    ]
    for identity, order in registrations:
        registry.register_layer(identity, _Layer(), semantic_order=order)

    original_ids = {id(layer) for layer in viewer.layers}
    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    assert viewer.grid.enabled
    assert viewer.grid.shape == (1, 2)
    assert viewer.grid.stride == registry.slot_count == 3
    assert viewer.grid.spacing == 7
    assert {id(layer) for layer in viewer.layers} == original_ids

    identities = [registry.identity_for_layer(layer) for layer in viewer.layers]
    assert [identity.platform for identity in identities] == [
        "MERSCOPE",
        "MERSCOPE",
        "MERSCOPE",
        "XENIUM",
        "XENIUM",
        "XENIUM",
    ]
    assert [identity.role for identity in identities[:3]] == [
        "image",
        "segmentation",
        "transcripts",
    ]
    assert [viewer.grid.position(i, len(viewer.layers)) for i in range(6)] == [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
        (0, 1),
    ]
    registry.assert_invariants()


def test_stacked_mode_disables_grid_and_interleaves_by_semantic_order():
    viewer, registry = _registry()
    for role, order in (("transcripts", 400), ("image", 100), ("segmentation", 300)):
        for platform in ("MERSCOPE", "XENIUM"):
            registry.register_layer(
                _identity(platform, role, role),
                _Layer(),
                semantic_order=order,
            )
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    layer_ids = {id(layer) for layer in viewer.layers}

    registry.set_mode(ViewMode.STACKED_OVERLAY)

    identities = [registry.identity_for_layer(layer) for layer in viewer.layers]
    assert not viewer.grid.enabled
    assert [(identity.role, identity.platform) for identity in identities] == [
        ("image", "MERSCOPE"),
        ("image", "XENIUM"),
        ("segmentation", "MERSCOPE"),
        ("segmentation", "XENIUM"),
        ("transcripts", "MERSCOPE"),
        ("transcripts", "XENIUM"),
    ]
    assert {id(layer) for layer in viewer.layers} == layer_ids


def test_visibility_is_linked_only_in_side_mode_and_overlay_state_round_trips():
    _viewer, registry = _registry()
    ms_identity = _identity("MERSCOPE", "segmentation", "proseg")
    xe_identity = ms_identity.for_platform("XENIUM")
    ms = registry.register_layer(ms_identity, _Layer(visible=True))
    xe = registry.register_layer(xe_identity, _Layer(visible=False))

    registry.set_mode(ViewMode.STACKED_OVERLAY)
    ms.visible = True
    xe.visible = False
    assert ms.visible and not xe.visible

    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    # Entering linked mode uses the union, so one visible counterpart shows both.
    assert ms.visible and xe.visible
    ms.visible = False
    assert not ms.visible and not xe.visible
    xe.visible = True
    assert ms.visible and xe.visible

    registry.set_mode(ViewMode.STACKED_OVERLAY)
    # Independent overlay choices from before linked mode are restored.
    assert ms.visible and not xe.visible
    xe.visible = True
    assert ms.visible and xe.visible
    ms.visible = False
    assert not ms.visible and xe.visible


def test_unload_keeps_slot_count_and_reload_restores_overlay_visibility():
    viewer, registry = _registry()
    identity = _identity("MERSCOPE", "image", "warped", "DAPI")
    original = registry.register_layer(identity, _Layer(visible=False))
    registry.register_layer(identity.for_platform("XENIUM"), _Layer(visible=True))
    registry.set_mode(ViewMode.STACKED_OVERLAY)

    assert registry.unload_layer(identity) is original
    placeholder = registry.layer_for(identity, include_placeholder=True)
    assert is_placeholder_layer(placeholder)
    assert placeholder not in viewer.layers
    assert len(viewer.layers) == 1

    replacement = registry.register_layer(identity, _Layer(visible=True))
    assert replacement is registry.layer_for(identity)
    assert not replacement.visible
    assert len(viewer.layers) == 2


def test_dynamic_slot_in_side_mode_updates_stride_without_cross_pane_leakage():
    viewer, registry = _registry()
    for platform in ("MERSCOPE", "XENIUM"):
        registry.register_layer(
            _identity(platform, "image", "morphology", "DAPI"), _Layer()
        )
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    assert viewer.grid.stride == 1

    registry.register_layer(
        _identity("MERSCOPE", "segmentation", "proseg"), _Layer()
    )

    assert registry.slot_count == 2
    assert viewer.grid.stride == 2
    assert [
        registry.identity_for_layer(layer).platform for layer in viewer.layers
    ] == ["MERSCOPE", "MERSCOPE", "XENIUM", "XENIUM"]
    registry.assert_invariants()


def test_viewbox_routing_is_explicit_and_overlay_is_ambiguous():
    _viewer, registry = _registry()
    registry.ensure_slot(LayerSlotKey("sentinel"))
    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    assert registry.platform_for_viewbox((0, 0)) == "MERSCOPE"
    assert registry.platform_for_viewbox((0, 1)) == "XENIUM"
    assert registry.platform_for_viewbox((1, 0)) is None
    assert registry.viewbox_for_platform("merscope") == (0, 0)
    assert registry.viewbox_for_platform("XENIUM") == (0, 1)

    registry.set_mode(ViewMode.STACKED_OVERLAY)
    assert registry.platform_for_viewbox((0, 0)) is None
    assert registry.viewbox_for_platform("MERSCOPE") is None


def test_repair_order_recovers_from_external_layer_drag():
    viewer, registry = _registry()
    for role in ("image", "segmentation"):
        for platform in ("MERSCOPE", "XENIUM"):
            registry.register_layer(_identity(platform, role, role), _Layer())
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    expected = tuple(viewer.layers)
    viewer.layers.insert(0, viewer.layers.pop())

    with pytest.raises(PairedViewInvariantError, match="out of order"):
        registry.assert_invariants()

    registry.repair_order()
    assert tuple(viewer.layers) == expected
    registry.assert_invariants()


def test_repair_order_restores_deleted_and_renamed_managed_layers():
    viewer, registry = _registry()
    identity = _identity("MERSCOPE", "image", "morphology", "DAPI")
    layer = registry.register_layer(identity, _Layer())
    registry.register_layer(identity.for_platform("XENIUM"), _Layer())
    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    viewer.layers.remove(layer)
    layer.name = "user rename"
    layer.metadata.clear()

    registry.repair_order()

    assert any(candidate is layer for candidate in viewer.layers)
    assert layer.name == platform_qualified_layer_name(identity)
    assert layer_identity_from_layer(layer) == identity
    registry.assert_invariants()


def test_side_mode_rejects_unmanaged_layers_that_would_change_grid_grouping():
    viewer, registry = _registry()
    registry.ensure_slot(LayerSlotKey("sentinel"))
    viewer.layers.append(_Layer(name="unmanaged annotation"))

    with pytest.raises(PairedViewInvariantError, match="unmanaged layers"):
        registry.set_mode(ViewMode.SIDE_BY_SIDE)
    assert registry.mode is ViewMode.STANDALONE
    assert not viewer.grid.enabled


def test_placeholder_visibility_never_turns_on_when_counterpart_is_mirrored():
    _viewer, registry = _registry()
    identity = _identity("MERSCOPE", "transcripts", "aligned", "disc")
    real = registry.register_layer(identity, _Layer(visible=True))
    placeholder = registry.counterpart(real, include_placeholder=True)
    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    real.visible = False
    real.visible = True

    assert real.visible
    assert is_placeholder_layer(placeholder)
    assert not placeholder.visible


def test_side_unload_reload_preserves_linked_and_independent_visibility_states():
    _viewer, registry = _registry()
    ms_identity = _identity("MERSCOPE", "image", "morphology", "DAPI")
    xe_identity = ms_identity.for_platform("XENIUM")
    ms = registry.register_layer(ms_identity, _Layer(visible=False))
    registry.register_layer(xe_identity, _Layer(visible=True))
    registry.set_mode(ViewMode.STACKED_OVERLAY)
    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    ms.visible = False

    registry.unload_layer(ms_identity)
    replacement = registry.register_layer(ms_identity, _Layer(visible=True))

    assert not replacement.visible
    assert not registry.layer_for(xe_identity).visible
    registry.set_mode(ViewMode.STACKED_OVERLAY)
    assert not replacement.visible
    assert registry.layer_for(xe_identity).visible


def test_final_side_slot_cannot_be_removed_because_grid_needs_two_nonempty_blocks():
    _viewer, registry = _registry()
    key = LayerSlotKey("sentinel")
    registry.ensure_slot(key)
    registry.set_mode(ViewMode.SIDE_BY_SIDE)

    with pytest.raises(PairedViewInvariantError, match="final slot"):
        registry.remove_slot(key)

    registry.assert_invariants()


def test_native_image_display_controls_are_linked_only_in_side_by_side_mode():
    _viewer, registry = _registry()
    merscope_identity = _identity("MERSCOPE", "image", "morphology", "DAPI")
    xenium_identity = merscope_identity.for_platform("XENIUM")
    merscope = registry.register_layer(merscope_identity, _ImageLayer())
    xenium = registry.register_layer(xenium_identity, _ImageLayer())

    registry.set_mode(ViewMode.SIDE_BY_SIDE)
    merscope.opacity = 0.35
    xenium.contrast_limits = (10.0, 200.0)
    merscope.gamma = 0.7
    xenium.colormap = "magenta"

    assert xenium.opacity == pytest.approx(0.35)
    assert merscope.contrast_limits == (10.0, 200.0)
    assert xenium.gamma == pytest.approx(0.7)
    assert merscope.colormap == "magenta"

    registry.set_mode(ViewMode.STACKED_OVERLAY)
    merscope.opacity = 0.8
    merscope.contrast_limits = (20.0, 80.0)

    assert xenium.opacity == pytest.approx(0.35)
    assert xenium.contrast_limits == (10.0, 200.0)
