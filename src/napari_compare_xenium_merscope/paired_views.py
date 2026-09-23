"""Presentation coordination for paired MERSCOPE/Xenium views.

The module deliberately knows very little about napari.  A compatible viewer
only needs a mutable ``layers`` sequence and a ``grid`` object exposing
``enabled``, ``shape``, ``stride`` and ``spacing``.  This keeps the ordering and
linking rules testable without constructing a Qt/OpenGL canvas while matching
napari 0.7's grid semantics:

* consecutive blocks of ``grid.stride`` layers are assigned to viewboxes;
* a ``(1, 2)`` grid therefore needs two equal-sized platform blocks; and
* disabling the grid draws the same layer objects together in one view.

The controller remains responsible for constructing real napari layers.  When
one platform has no loaded layer for a logical slot, the registry calls the
supplied ``placeholder_factory``.  That factory must return a lightweight layer
object without adding it to the viewer itself.  Placeholders are inserted into
Napari's real LayerList only while side-by-side mode needs equal platform
blocks; stacked overlay and standalone modes expose real data layers only.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, MutableSequence


MERSCOPE = "MERSCOPE"
XENIUM = "XENIUM"
DEFAULT_PLATFORM_ORDER = (MERSCOPE, XENIUM)

# Native layer controls which have a meaningful counterpart operation in the
# linked grid.  Array-valued styling (point sizes/colours, label colormaps) is
# deliberately excluded because the two platforms can have different element
# counts or instance ids.  Image colormaps are safe because paired image slots
# represent the same logical channel.
_COMMON_LINKED_PROPERTIES = ("opacity", "blending")
_IMAGE_LINKED_PROPERTIES = (
    "contrast_limits",
    "gamma",
    "colormap",
    "interpolation2d",
)

PAIRED_LAYER_METADATA_KEY = "napari_compare_paired_layer"


class ViewMode(str, Enum):
    """Supported viewer presentation modes."""

    STANDALONE = "standalone"
    SIDE_BY_SIDE = "side-by-side"
    STACKED_OVERLAY = "stacked-overlay"

    @classmethod
    def coerce(cls, value: "ViewMode | str") -> "ViewMode":
        """Return a mode while accepting convenient CLI/UI spellings."""

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("_", "-")
        aliases = {
            "side": cls.SIDE_BY_SIDE,
            "side-by-side": cls.SIDE_BY_SIDE,
            "stacked": cls.STACKED_OVERLAY,
            "overlay": cls.STACKED_OVERLAY,
            "stacked-overlay": cls.STACKED_OVERLAY,
            "standalone": cls.STANDALONE,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unknown view mode {value!r}; expected one of {choices}.") from exc


def normalize_platform(platform: str) -> str:
    """Normalize and validate a paired platform name."""

    normalized = str(platform).strip().upper()
    if normalized not in DEFAULT_PLATFORM_ORDER:
        raise ValueError(
            f"Unknown paired platform {platform!r}; expected {MERSCOPE!r} or {XENIUM!r}."
        )
    return normalized


def normalize_role(role: str) -> str:
    """Return a stable snake-case role identifier."""

    return "_".join(str(role).strip().lower().replace("-", " ").split())


@dataclass(frozen=True, order=True)
class LayerSlotKey:
    """Platform-independent identity of one logical display layer slot."""

    role: str
    element_key: str = ""
    channel: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", normalize_role(self.role))
        object.__setattr__(self, "element_key", str(self.element_key or ""))
        object.__setattr__(self, "channel", str(self.channel or ""))


@dataclass(frozen=True, order=True)
class LayerIdentity:
    """Stable identity for a platform-owned layer.

    Display names are intentionally excluded from identity.  Names may change
    for presentation, whereas this value is safe for counterpart lookup and
    event routing.
    """

    platform: str
    role: str
    element_key: str = ""
    channel: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", normalize_platform(self.platform))
        object.__setattr__(self, "role", normalize_role(self.role))
        object.__setattr__(self, "element_key", str(self.element_key or ""))
        object.__setattr__(self, "channel", str(self.channel or ""))

    @property
    def slot_key(self) -> LayerSlotKey:
        return LayerSlotKey(self.role, self.element_key, self.channel)

    def for_platform(self, platform: str) -> "LayerIdentity":
        return LayerIdentity(
            platform=platform,
            role=self.role,
            element_key=self.element_key,
            channel=self.channel,
        )


_ROLE_LABELS = {
    "image": "Image",
    "images": "Image",
    "segmentation": "Segmentation",
    "labels": "Segmentation",
    "shapes": "Segmentation",
    "genes": "Genes",
    "transcripts": "Genes",
    "cell_types": "Cell types",
    "cell_type": "Cell types",
    "cell_values": "Cell values",
    "selection": "Selection",
    "annotation": "Annotation",
}


def _display_component(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def platform_qualified_layer_name(
    identity_or_platform: LayerIdentity | str,
    role: str | None = None,
    element_key: str = "",
    channel: str | None = None,
) -> str:
    """Build an unambiguous paired-mode layer name.

    Either pass a :class:`LayerIdentity` or the four identity components.
    Element keys are retained because two images can expose the same channel.
    """

    if isinstance(identity_or_platform, LayerIdentity):
        identity = identity_or_platform
    else:
        if role is None:
            raise TypeError("role is required when passing a platform string")
        identity = LayerIdentity(
            identity_or_platform,
            role,
            element_key,
            "" if channel is None else channel,
        )
    role_label = _ROLE_LABELS.get(identity.role, _display_component(identity.role))
    parts = [identity.platform, role_label]
    if identity.element_key:
        parts.append(_display_component(identity.element_key))
    if identity.channel:
        parts.append(str(identity.channel))
    return " | ".join(parts)


def layer_identity_metadata(
    identity: LayerIdentity, *, placeholder: bool = False
) -> dict[str, Any]:
    """Return the namespaced metadata payload stored on a paired layer."""

    return {
        "platform": identity.platform,
        "role": identity.role,
        "element_key": identity.element_key,
        "channel": identity.channel,
        "placeholder": bool(placeholder),
    }


def attach_layer_identity(
    layer: Any, identity: LayerIdentity, *, placeholder: bool = False
) -> Any:
    """Attach identity metadata and a platform-qualified name to ``layer``."""

    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[PAIRED_LAYER_METADATA_KEY] = layer_identity_metadata(
        identity, placeholder=placeholder
    )
    layer.metadata = metadata
    layer.name = platform_qualified_layer_name(identity)
    return layer


def layer_identity_from_metadata(metadata: Mapping[str, Any] | None) -> LayerIdentity | None:
    """Decode a paired identity, returning ``None`` for unrelated metadata."""

    if not metadata:
        return None
    payload = metadata.get(PAIRED_LAYER_METADATA_KEY)
    if not isinstance(payload, Mapping):
        return None
    try:
        return LayerIdentity(
            platform=str(payload["platform"]),
            role=str(payload["role"]),
            element_key=str(payload.get("element_key", "")),
            channel=str(payload.get("channel", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def layer_identity_from_layer(layer: Any) -> LayerIdentity | None:
    """Decode paired identity metadata from a layer-like object."""

    return layer_identity_from_metadata(getattr(layer, "metadata", None))


def is_placeholder_layer(layer: Any) -> bool:
    """Return whether a layer is a registry-created empty placeholder."""

    metadata = getattr(layer, "metadata", {}) or {}
    payload = metadata.get(PAIRED_LAYER_METADATA_KEY, {})
    return isinstance(payload, Mapping) and bool(payload.get("placeholder", False))


# Lower numbers are drawn first (below later layers) in napari's LayerList.
DEFAULT_ROLE_ORDER: dict[str, int] = {
    "image": 100,
    "images": 100,
    "cell_types": 200,
    "cell_type": 200,
    "cell_values": 220,
    "segmentation": 300,
    "labels": 300,
    "shapes": 300,
    "genes": 400,
    "transcripts": 400,
    "selection": 500,
    "annotation": 600,
}


def semantic_order_for_role(role: str) -> int:
    """Return a stable bottom-to-top order for an unknown or known role."""

    return int(DEFAULT_ROLE_ORDER.get(normalize_role(role), 350))


class PairedViewInvariantError(RuntimeError):
    """Raised when layer-list state cannot satisfy napari grid grouping."""


@dataclass
class _PairedSlot:
    key: LayerSlotKey
    semantic_order: int
    sequence: int
    layers: dict[str, Any] = field(default_factory=dict)
    real_platforms: set[str] = field(default_factory=set)
    side_visible: bool = True
    ever_real: bool = False


PlaceholderFactory = Callable[[LayerIdentity], Any]


class PairedLayerRegistry:
    """Own paired layer slots, ordering, grid mode, and linked visibility.

    Parameters
    ----------
    viewer:
        Viewer-like object with mutable ``layers`` and an evented-or-plain
        ``grid`` model.
    placeholder_factory:
        Called for missing platform counterparts.  It must return a layer-like
        object but must not insert it into ``viewer.layers``.
    platform_order:
        First platform renders left in side-by-side mode and below its
        counterpart within a stacked semantic slot.
    strict_layers:
        If true, side-by-side activation rejects unmanaged viewer layers.  This
        prevents a stray layer from changing Napari's index/stride grouping.
    spacing:
        Gap between the two Napari grid viewboxes.
    """

    def __init__(
        self,
        viewer: Any,
        placeholder_factory: PlaceholderFactory,
        *,
        platform_order: Iterable[str] = DEFAULT_PLATFORM_ORDER,
        strict_layers: bool = True,
        spacing: float = 4.0,
    ) -> None:
        self.viewer = viewer
        self.layers: MutableSequence[Any] = viewer.layers
        self.grid = viewer.grid
        platforms = tuple(normalize_platform(value) for value in platform_order)
        if len(platforms) != 2 or len(set(platforms)) != 2:
            raise ValueError("platform_order must contain MERSCOPE and XENIUM exactly once")
        if set(platforms) != set(DEFAULT_PLATFORM_ORDER):
            raise ValueError("platform_order must contain MERSCOPE and XENIUM exactly once")
        self.platform_order = platforms
        self.placeholder_factory = placeholder_factory
        self.strict_layers = bool(strict_layers)
        self.spacing = float(spacing)
        self.mode = ViewMode.STANDALONE

        self._slots: dict[LayerSlotKey, _PairedSlot] = {}
        self._layer_records: dict[int, tuple[Any, LayerIdentity]] = {}
        self._visibility_callbacks: dict[int, Callable[..., None]] = {}
        self._property_callbacks: dict[
            int, list[tuple[Any, Callable[..., None]]]
        ] = {}
        self._overlay_visibility: dict[LayerIdentity, bool] = {}
        self._sequence = 0
        self._syncing_visibility = False

    # -- inspection -----------------------------------------------------
    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @property
    def slot_keys(self) -> tuple[LayerSlotKey, ...]:
        return tuple(slot.key for slot in self._ordered_slots())

    def identity_for_layer(self, layer: Any) -> LayerIdentity | None:
        record = self._layer_records.get(id(layer))
        if record is not None and record[0] is layer:
            return record[1]
        return layer_identity_from_layer(layer)

    def layer_for(self, identity: LayerIdentity, *, include_placeholder: bool = False) -> Any | None:
        slot = self._slots.get(identity.slot_key)
        if slot is None:
            return None
        layer = slot.layers.get(identity.platform)
        if layer is None or (not include_placeholder and identity.platform not in slot.real_platforms):
            return None
        return layer

    def counterpart(self, layer_or_identity: Any, *, include_placeholder: bool = False) -> Any | None:
        identity = (
            layer_or_identity
            if isinstance(layer_or_identity, LayerIdentity)
            else self.identity_for_layer(layer_or_identity)
        )
        if identity is None:
            return None
        other = next(p for p in self.platform_order if p != identity.platform)
        return self.layer_for(identity.for_platform(other), include_placeholder=include_placeholder)

    def layers_for_platform(
        self, platform: str, *, include_placeholders: bool = True
    ) -> tuple[Any, ...]:
        platform = normalize_platform(platform)
        result = []
        for slot in self._ordered_slots():
            if include_placeholders or platform in slot.real_platforms:
                result.append(slot.layers[platform])
        return tuple(result)

    def ordered_layers(self, mode: ViewMode | str | None = None) -> tuple[Any, ...]:
        mode = self.mode if mode is None else ViewMode.coerce(mode)
        slots = self._ordered_slots()
        if mode is ViewMode.SIDE_BY_SIDE:
            return tuple(
                slot.layers[platform]
                for platform in self.platform_order
                for slot in slots
            )
        return tuple(
            slot.layers[platform]
            for slot in slots
            for platform in self.platform_order
            if platform in slot.real_platforms
        )

    # -- registration ---------------------------------------------------
    def ensure_slot(
        self,
        key: LayerSlotKey,
        *,
        semantic_order: int | None = None,
        side_visible: bool = True,
    ) -> LayerSlotKey:
        """Ensure a complete two-platform slot exists, using placeholders."""

        key = LayerSlotKey(key.role, key.element_key, key.channel)
        order = semantic_order_for_role(key.role) if semantic_order is None else int(semantic_order)
        slot = self._slots.get(key)
        if slot is None:
            slot = _PairedSlot(
                key=key,
                semantic_order=order,
                sequence=self._sequence,
                side_visible=bool(side_visible),
            )
            self._sequence += 1
            self._slots[key] = slot
        elif semantic_order is not None and slot.semantic_order != order:
            raise ValueError(
                f"Conflicting semantic order for {key!r}: "
                f"{slot.semantic_order} != {order}."
            )

        with self._batch_layers():
            for platform in self.platform_order:
                if platform not in slot.layers:
                    identity = LayerIdentity(
                        platform, key.role, key.element_key, key.channel
                    )
                    placeholder = self.placeholder_factory(identity)
                    if placeholder is None:
                        raise TypeError("placeholder_factory returned None")
                    attach_layer_identity(placeholder, identity, placeholder=True)
                    self._install_layer_record(identity, placeholder, connect=False)
                    slot.layers[platform] = placeholder
        self._apply_order_only()
        return key

    def register_layer(
        self,
        identity: LayerIdentity,
        layer: Any,
        *,
        semantic_order: int | None = None,
    ) -> Any:
        """Install or replace the real layer for one platform slot."""

        if not isinstance(identity, LayerIdentity):
            raise TypeError("identity must be a LayerIdentity")
        if layer is None:
            raise TypeError("layer cannot be None")
        existing_record = self._layer_records.get(id(layer))
        if (
            existing_record is not None
            and existing_record[0] is layer
            and existing_record[1] != identity
        ):
            raise PairedViewInvariantError(
                f"Layer object is already registered as {existing_record[1]!r}."
            )
        self.ensure_slot(identity.slot_key, semantic_order=semantic_order)
        slot = self._slots[identity.slot_key]
        old = slot.layers[identity.platform]
        old_was_real = identity.platform in slot.real_platforms
        had_real_history = slot.ever_real
        incoming_visibility = bool(getattr(layer, "visible", True))
        if old_was_real:
            self._remember_visibility(identity, old)

        attach_layer_identity(layer, identity, placeholder=False)
        with self._batch_layers():
            self._disconnect_visibility(old)
            self._remove_if_present(old)
            self._forget_layer_record(old)
            slot.layers[identity.platform] = layer
            slot.real_platforms.add(identity.platform)
            slot.ever_real = True
            self._install_layer_record(identity, layer, connect=True)
            self._append_if_missing(layer)

            if self.mode is ViewMode.SIDE_BY_SIDE:
                if not had_real_history:
                    slot.side_visible = incoming_visibility
                self._set_layer_visible(layer, slot.side_visible)
                self._mirror_slot_visibility(slot, source_platform=identity.platform)
            else:
                restored = self._overlay_visibility.get(identity, incoming_visibility)
                self._set_layer_visible(layer, restored)
                self._overlay_visibility[identity] = restored
        self._apply_order_only()
        return layer

    def unload_layer(self, identity: LayerIdentity) -> Any | None:
        """Replace a real layer with an invisible lightweight placeholder."""

        slot = self._slots.get(identity.slot_key)
        if slot is None or identity.platform not in slot.real_platforms:
            return None
        old = slot.layers[identity.platform]
        self._remember_visibility(identity, old)
        placeholder = self.placeholder_factory(identity)
        if placeholder is None:
            raise TypeError("placeholder_factory returned None")
        attach_layer_identity(placeholder, identity, placeholder=True)
        with self._batch_layers():
            self._disconnect_visibility(old)
            self._remove_if_present(old)
            self._forget_layer_record(old)
            slot.layers[identity.platform] = placeholder
            slot.real_platforms.remove(identity.platform)
            self._install_layer_record(identity, placeholder, connect=False)
        self._apply_order_only()
        return old

    def remove_slot(self, key: LayerSlotKey) -> tuple[Any, ...]:
        """Remove both platform entries for a logical slot."""

        key = LayerSlotKey(key.role, key.element_key, key.channel)
        if self.mode is ViewMode.SIDE_BY_SIDE and key in self._slots and len(self._slots) == 1:
            raise PairedViewInvariantError(
                "Cannot remove the final slot while side-by-side mode is active. "
                "Install a sentinel slot first or leave side-by-side mode."
            )
        slot = self._slots.pop(key, None)
        if slot is None:
            return ()
        removed = tuple(slot.layers[p] for p in self.platform_order)
        with self._batch_layers():
            for platform, layer in slot.layers.items():
                identity = LayerIdentity(platform, key.role, key.element_key, key.channel)
                self._remember_visibility(identity, layer)
                self._disconnect_visibility(layer)
                self._remove_if_present(layer)
                self._forget_layer_record(layer)
        self._apply_order_only()
        return removed

    # -- mode and event coordination -----------------------------------
    def set_mode(self, mode: ViewMode | str) -> ViewMode:
        """Apply grid state, canonical order, and mode-specific visibility."""

        requested = ViewMode.coerce(mode)
        previous = self.mode
        # Validate before changing logical or grid state so a failed transition
        # leaves the prior mode fully usable.
        if requested is ViewMode.SIDE_BY_SIDE:
            if not self._slots:
                raise PairedViewInvariantError(
                    "Side-by-side mode needs at least one paired slot."
                )
            self._assert_no_unmanaged_layers()
        if previous is not ViewMode.SIDE_BY_SIDE and requested is ViewMode.SIDE_BY_SIDE:
            self._snapshot_overlay_visibility()
            for slot in self._slots.values():
                visible = [
                    bool(getattr(slot.layers[p], "visible", False))
                    for p in self.platform_order
                    if p in slot.real_platforms
                ]
                if visible:
                    # Union is least surprising: entering a linked view does not
                    # make a layer disappear merely because one overlay was hidden.
                    slot.side_visible = any(visible)
        elif previous is ViewMode.SIDE_BY_SIDE and requested is not ViewMode.SIDE_BY_SIDE:
            self._snapshot_side_visibility()

        self.mode = requested
        if requested is ViewMode.SIDE_BY_SIDE:
            self.grid.enabled = False
            self._apply_order_only()
            self.grid.shape = (1, 2)
            self.grid.stride = self.slot_count
            self.grid.spacing = self.spacing
            self._apply_side_visibility()
            self.grid.enabled = True
        else:
            self.grid.enabled = False
            self._apply_order_only()
            self._restore_overlay_visibility()
        return requested

    def repair_order(self) -> None:
        """Restore canonical identity and ordering after a native UI mutation."""

        self._assert_no_unmanaged_layers()
        for slot in self._slots.values():
            for platform, layer in slot.layers.items():
                identity = LayerIdentity(
                    platform,
                    slot.key.role,
                    slot.key.element_key,
                    slot.key.channel,
                )
                if (
                    layer_identity_from_layer(layer) != identity
                    or str(getattr(layer, "name", ""))
                    != platform_qualified_layer_name(identity)
                ):
                    attach_layer_identity(
                        layer,
                        identity,
                        placeholder=platform not in slot.real_platforms,
                    )
        self._apply_order_only()
        if self.mode is ViewMode.SIDE_BY_SIDE:
            self.grid.shape = (1, 2)
            self.grid.stride = self.slot_count
            self.grid.spacing = self.spacing
            self.grid.enabled = True

    def platform_for_viewbox(self, viewbox: Any) -> str | None:
        """Map a Napari mouse-event viewbox to its source platform.

        Stacked overlay is intentionally ambiguous and returns ``None``.
        """

        if self.mode is not ViewMode.SIDE_BY_SIDE or viewbox is None:
            return None
        try:
            position = tuple(int(value) for value in viewbox)
        except (TypeError, ValueError):
            return None
        if position == (0, 0):
            return self.platform_order[0]
        if position == (0, 1):
            return self.platform_order[1]
        return None

    def viewbox_for_platform(self, platform: str) -> tuple[int, int] | None:
        """Return a platform's grid cell in side-by-side mode."""

        if self.mode is not ViewMode.SIDE_BY_SIDE:
            return None
        platform = normalize_platform(platform)
        return (0, self.platform_order.index(platform))

    def assert_invariants(self) -> None:
        """Raise if paired slots or current Napari grid state are inconsistent."""

        if any(set(slot.layers) != set(self.platform_order) for slot in self._slots.values()):
            raise PairedViewInvariantError("Every logical slot must have both platform entries.")
        registered = tuple(
            layer
            for slot in self._slots.values()
            for layer in slot.layers.values()
        )
        if len({id(layer) for layer in registered}) != len(registered):
            raise PairedViewInvariantError("A layer object is registered in more than one slot.")
        ordered = self.ordered_layers()
        displayed_ids = {id(layer) for layer in ordered}
        for layer in registered:
            occurrences = sum(candidate is layer for candidate in self.layers)
            expected = 1 if id(layer) in displayed_ids else 0
            if occurrences != expected:
                raise PairedViewInvariantError(
                    "Registered layers must appear exactly when required by the current mode."
                )
        if self.mode is ViewMode.SIDE_BY_SIDE:
            self._assert_no_unmanaged_layers()
            if tuple(self.layers) != ordered:
                raise PairedViewInvariantError("Side-by-side platform blocks are out of order.")
            if not bool(self.grid.enabled):
                raise PairedViewInvariantError("Napari grid is disabled in side-by-side mode.")
            if tuple(self.grid.shape) != (1, 2) or int(self.grid.stride) != self.slot_count:
                raise PairedViewInvariantError("Napari grid shape/stride does not match paired slots.")

    # -- internal helpers ------------------------------------------------
    def _ordered_slots(self) -> list[_PairedSlot]:
        return sorted(
            self._slots.values(),
            key=lambda slot: (
                slot.semantic_order,
                slot.key.role,
                slot.key.element_key.casefold(),
                slot.key.channel.casefold(),
                slot.sequence,
            ),
        )

    def _batch_layers(self):
        batch = getattr(self.layers, "batched_update", None)
        return batch() if callable(batch) else nullcontext()

    def _append_if_missing(self, layer: Any) -> None:
        if not any(candidate is layer for candidate in self.layers):
            self.layers.append(layer)

    def _remove_if_present(self, layer: Any) -> None:
        for index, candidate in enumerate(self.layers):
            if candidate is layer:
                self.layers.pop(index)
                return

    def _move(self, source: int, destination: int) -> None:
        move = getattr(self.layers, "move", None)
        if callable(move):
            # napari's EventedList.move destination is in *pre-move* space and
            # means "insert before".  Adjust a forward move so the resulting
            # post-move index is exactly ``destination``.
            pre_move_destination = destination + 1 if source < destination else destination
            move(source, pre_move_destination)
            return
        layer = self.layers.pop(source)
        self.layers.insert(destination, layer)

    def _apply_order_only(self) -> None:
        refresh_live_grid = (
            self.mode is ViewMode.SIDE_BY_SIDE and bool(self.grid.enabled)
        )
        if not self._slots:
            if self.mode is ViewMode.SIDE_BY_SIDE:
                self.grid.enabled = False
            return
        desired = self.ordered_layers()
        desired_ids = {id(layer) for layer in desired}
        registered_ids = self._managed_layer_ids()
        with self._batch_layers():
            # Missing-counterpart placeholders are structural Napari layers
            # only in side-by-side mode.  Remove them when entering overlay or
            # standalone mode so the LayerList presents real datasets only.
            for layer in tuple(self.layers):
                if id(layer) in registered_ids and id(layer) not in desired_ids:
                    self._remove_if_present(layer)
            for target, layer in enumerate(desired):
                current = next(
                    (i for i, candidate in enumerate(self.layers) if candidate is layer),
                    None,
                )
                if current is None:
                    self.layers.insert(target, layer)
                elif current != target:
                    self._move(current, target)
        if refresh_live_grid:
            # A newly discovered image/channel/annotation adds one slot to both
            # platform blocks.  Napari's stride must change in the same logical
            # operation or the old boundary sends layers to the wrong viewbox.
            self.grid.shape = (1, 2)
            self.grid.stride = self.slot_count
            self.grid.spacing = self.spacing
            self.grid.enabled = True

    def _install_layer_record(
        self, identity: LayerIdentity, layer: Any, *, connect: bool
    ) -> None:
        existing = self._layer_records.get(id(layer))
        if existing is not None and existing[0] is layer and existing[1] != identity:
            raise PairedViewInvariantError(
                f"Layer object is already registered as {existing[1]!r}."
            )
        self._layer_records[id(layer)] = (layer, identity)
        if connect:
            self._connect_visibility(layer)
            self._connect_linked_properties(identity, layer)

    def _forget_layer_record(self, layer: Any) -> None:
        record = self._layer_records.get(id(layer))
        if record is not None and record[0] is layer:
            self._layer_records.pop(id(layer), None)

    def _connect_visibility(self, layer: Any) -> None:
        emitter = getattr(getattr(layer, "events", None), "visible", None)
        connect = getattr(emitter, "connect", None)
        if not callable(connect):
            return

        def callback(_event=None, target=layer):
            self._on_layer_visibility_changed(target)

        connect(callback)
        self._visibility_callbacks[id(layer)] = callback

    def _disconnect_visibility(self, layer: Any) -> None:
        callback = self._visibility_callbacks.pop(id(layer), None)
        if callback is not None:
            emitter = getattr(getattr(layer, "events", None), "visible", None)
            disconnect = getattr(emitter, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(callback)
                except (TypeError, ValueError):
                    pass
        for emitter, property_callback in self._property_callbacks.pop(
            id(layer), []
        ):
            disconnect_property = getattr(emitter, "disconnect", None)
            if callable(disconnect_property):
                try:
                    disconnect_property(property_callback)
                except (TypeError, ValueError):
                    pass

    def _connect_linked_properties(
        self, identity: LayerIdentity, layer: Any
    ) -> None:
        """Mirror compatible native display controls in the linked grid."""
        names = list(_COMMON_LINKED_PROPERTIES)
        if identity.role in {"image", "images"}:
            names.extend(_IMAGE_LINKED_PROPERTIES)
        callbacks: list[tuple[Any, Callable[..., None]]] = []
        events = getattr(layer, "events", None)
        for name in names:
            emitter = getattr(events, name, None)
            connect = getattr(emitter, "connect", None)
            if not callable(connect) or not hasattr(layer, name):
                continue

            def callback(
                _event=None,
                target=layer,
                property_name=name,
            ):
                self._on_linked_property_changed(target, property_name)

            connect(callback)
            callbacks.append((emitter, callback))
        if callbacks:
            self._property_callbacks[id(layer)] = callbacks

    def _on_linked_property_changed(
        self, layer: Any, property_name: str
    ) -> None:
        if self.mode is not ViewMode.SIDE_BY_SIDE or self._syncing_visibility:
            return
        identity = self.identity_for_layer(layer)
        if identity is None:
            return
        counterpart = self.counterpart(identity)
        if counterpart is None or not hasattr(counterpart, property_name):
            return
        try:
            source_value = getattr(layer, property_name)
            target_value = getattr(counterpart, property_name)
            equal = source_value == target_value
            if hasattr(equal, "all"):
                equal = bool(equal.all())
            if bool(equal):
                return
        except Exception:
            pass
        self._syncing_visibility = True
        try:
            setattr(counterpart, property_name, getattr(layer, property_name))
        except (AttributeError, TypeError, ValueError):
            # Layer subclasses and napari versions expose slightly different
            # controls; an unsupported counterpart property is non-fatal.
            pass
        finally:
            self._syncing_visibility = False

    def _on_layer_visibility_changed(self, layer: Any) -> None:
        if self._syncing_visibility:
            return
        identity = self.identity_for_layer(layer)
        if identity is None:
            return
        visible = bool(getattr(layer, "visible", True))
        if self.mode is ViewMode.SIDE_BY_SIDE:
            slot = self._slots.get(identity.slot_key)
            if slot is None:
                return
            slot.side_visible = visible
            self._mirror_slot_visibility(slot, source_platform=identity.platform)
        else:
            self._overlay_visibility[identity] = visible

    def _set_layer_visible(self, layer: Any, visible: bool) -> None:
        if is_placeholder_layer(layer):
            visible = False
        if bool(getattr(layer, "visible", True)) != bool(visible):
            layer.visible = bool(visible)

    def _mirror_slot_visibility(
        self, slot: _PairedSlot, *, source_platform: str | None = None
    ) -> None:
        self._syncing_visibility = True
        try:
            for platform in self.platform_order:
                if platform == source_platform or platform not in slot.real_platforms:
                    continue
                self._set_layer_visible(slot.layers[platform], slot.side_visible)
        finally:
            self._syncing_visibility = False

    def _apply_side_visibility(self) -> None:
        self._syncing_visibility = True
        try:
            for slot in self._slots.values():
                for platform in slot.real_platforms:
                    self._set_layer_visible(slot.layers[platform], slot.side_visible)
        finally:
            self._syncing_visibility = False

    def _remember_visibility(self, identity: LayerIdentity, layer: Any) -> None:
        # Side-by-side has its own linked visibility state.  Replacing/unloading
        # a layer there must not overwrite the independent stacked snapshot.
        if self.mode is not ViewMode.SIDE_BY_SIDE and not is_placeholder_layer(layer):
            self._overlay_visibility[identity] = bool(getattr(layer, "visible", True))

    def _snapshot_overlay_visibility(self) -> None:
        for slot in self._slots.values():
            for platform in slot.real_platforms:
                identity = LayerIdentity(
                    platform, slot.key.role, slot.key.element_key, slot.key.channel
                )
                self._remember_visibility(identity, slot.layers[platform])

    def _snapshot_side_visibility(self) -> None:
        for slot in self._slots.values():
            visible = [
                bool(getattr(slot.layers[p], "visible", False))
                for p in self.platform_order
                if p in slot.real_platforms
            ]
            if visible:
                slot.side_visible = any(visible)

    def _restore_overlay_visibility(self) -> None:
        self._syncing_visibility = True
        try:
            for slot in self._slots.values():
                for platform in slot.real_platforms:
                    identity = LayerIdentity(
                        platform, slot.key.role, slot.key.element_key, slot.key.channel
                    )
                    visible = self._overlay_visibility.get(identity, slot.side_visible)
                    self._set_layer_visible(slot.layers[platform], visible)
        finally:
            self._syncing_visibility = False

    def _managed_layer_ids(self) -> set[int]:
        return {
            id(layer)
            for slot in self._slots.values()
            for layer in slot.layers.values()
        }

    def _assert_no_unmanaged_layers(self) -> None:
        if not self.strict_layers:
            return
        managed = self._managed_layer_ids()
        extras = [
            str(getattr(layer, "name", type(layer).__name__))
            for layer in self.layers
            if id(layer) not in managed
        ]
        if extras:
            raise PairedViewInvariantError(
                "Side-by-side grid cannot contain unmanaged layers because they "
                f"change stride grouping: {extras}."
            )


__all__ = [
    "DEFAULT_PLATFORM_ORDER",
    "DEFAULT_ROLE_ORDER",
    "LayerIdentity",
    "LayerSlotKey",
    "MERSCOPE",
    "PAIRED_LAYER_METADATA_KEY",
    "PairedLayerRegistry",
    "PairedViewInvariantError",
    "ViewMode",
    "XENIUM",
    "attach_layer_identity",
    "is_placeholder_layer",
    "layer_identity_from_layer",
    "layer_identity_from_metadata",
    "layer_identity_metadata",
    "normalize_platform",
    "platform_qualified_layer_name",
    "semantic_order_for_role",
]
