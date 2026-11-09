"""Manual landmark registration session controller (napari, no Qt widgets of its own).

The controller opens an image-only registration store (a MerXen *prepared store* or *result
store*), shows the reference and moving images in napari with the moving image placed by a
coarse rotate/flip/shift transform, lets the user optionally crop each image and place landmark
pairs by clicking, keeps a live robust fit, saves the labelling into the store's root
attributes, restores it on reload (including local drafts) and shows a finished registration
result read-only with landmark overlays. The Qt panel is VW1's ``ManualRegistrationWidget``;
the maths, the attrs contract and the store I/O live in ``manual_registration_model`` and
``manual_registration_store``. This module never imports ``viewer.py`` (the viewer imports it)
and never imports ``merxen``.

Coordinate conventions (identical to MerXen):

* A landmark is ``(x, y)`` in level-0 pixels of the full, uncropped image it belongs to;
  ``x`` is the column, ``y`` the row; an integer is a pixel *centre* (pixel ``i`` covers
  ``[i - 0.5, i + 0.5)``). The valid range in an image of shape ``(H, W)`` is
  ``-0.5 <= x < W - 0.5`` and ``-0.5 <= y < H - 0.5``.
* napari layer data are ``(row, col) = (y, x)``. Every matrix maps *moving px -> reference px*.
* The whole placement of a layer goes into ``layer.affine`` (``scale``, ``rotate``,
  ``translate`` and ``shear`` stay identity; a flip is never a negative scale). Reference layers
  use the reference ``base_affine_rc`` of the store; every moving-side layer uses
  ``napari_affine_rc(A_ref, M)`` with ``M`` the coarse matrix (or the robust-fit matrix while
  "Preview fit" is on). The world unit is micrometres.
* Crops are ``{"x0", "y0", "x1", "y1"}`` ints, half-open in the full level-0 frame of their own
  image.

Threading: ``load_store`` runs ``prepare_session`` (read, descriptor, display levels, contrast
limits; no Qt, no viewer access) and ``save_labelling`` through ``host.run_worker``; everything
else (fit, affines, markers, dialogs) runs on the GUI thread.

Keys bound for the lifetime of a session: ``F`` (flick), ``Shift-F`` (hold to peek),
``Control-Z`` (undo), ``Escape`` (cancel), and the guards ``Control-Alt-T``,
``Control-Delete`` and ``Control-Backspace``. They are restored when the session ends.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from napari.layers import Image, Points, Shapes
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.key_bindings import KeyBinding
from qtpy.QtCore import QCoreApplication, QTimer

from .manual_registration_model import (
    LAYER_METADATA_KEY,
    ROLES,
    CoarseState,
    CropRect,
    LabellingMismatch,
    LabellingState,
    LandmarkEvaluation,
    LandmarkPair,
    NotARegistrationStore,
    ResidualRow,
    StoreDescriptor,
    UnsupportedSchemaVersion,
    coarse_from_matrix,
    coarse_matrix_xy,
    evaluate_landmarks,
    finish_allowed,
    labelling_content_sha256,
    labelling_from_attrs,
    labelling_to_attrs,
    napari_affine_rc,
    pair_colour,
    parse_store_descriptor,
    points_outside_crop,
    utc_now,
)
from .manual_registration_store import (
    StoreBusyError,
    StoreChangedError,
    clear_local_draft,
    read_local_draft,
    read_root_attrs,
    read_store_descriptor,
    save_labelling,
    write_local_draft,
)

log = logging.getLogger(__name__)

REGISTRATION_DATASET = "MANUAL_REGISTRATION"

CLICK_DRAG_THRESHOLD_PX = 4.0  # same as viewer.INSPECT_CLICK_DRAG_THRESHOLD_PX
AFFINE_UPDATE_DELAY_MS = 30
DRAFT_WRITE_DELAY_MS = 500
MARKER_SCREEN_PX = 30.0  # ring diameter on screen
CROSS_SIZE_FACTOR = 0.6  # cross size relative to the ring
RING_BORDER_WIDTH = 0.16  # relative
CROSS_BORDER_WIDTH = 0.30  # relative
LABEL_OFFSET_FACTOR = 0.45  # label offset = factor x marker world size, up and right
LABEL_TEXT_SIZE = 11
CROP_EDGE_SCREEN_PX = 2.0
CROP_COLOUR = "#ffff00"
CROP_MIN_DRAW_PX = 16
CROP_SNAP_PX = 64  # mirrors MerXen; used only for the size warning
CROP_MIN_SIZE_PX = 1024
DISABLED_PAIR_COLOUR = "#9e9e9e"
REFERENCE_COLORMAP = "green"
MOVING_COLORMAP = "magenta"
OTHER_CHANNEL_COLORMAP = "gray"
CONTRAST_PERCENTILES = (0.5, 99.5)
CONTRAST_MAX_PIXELS = 4_000_000
PREVIEW_MIN_ENABLED = 3
UNDO_LIMIT = 500
SESSION_KEYS = (
    "F",
    "Shift-F",
    "Control-Z",
    "Escape",
    "Control-Alt-T",
    "Control-Delete",
    "Control-Backspace",
)

FLICK_NEXT = {"reference": "moving", "moving": "reference", "both": "reference"}
PEEK_OF = {"reference": "moving", "moving": "reference", "both": "reference"}
SHOWN_ROLES = {
    "reference": {"reference"},
    "moving": {"moving"},
    "both": {"reference", "moving"},
}

_TRANSFORM_TEXT = (
    "Transform mode is disabled during manual registration; use the Coarse alignment controls."
)
_DELETE_TEXT = (
    "Layers of a registration session cannot be deleted; load another dataset to close the session."
)
_NO_SESSION_TEXT = "Load a registration store first."
_OVERLAY_LAYERS = {
    "reference_landmarks": "Reference landmarks",
    "registered_landmarks": "Registered moving landmarks",
    "landmark_fit_positions": "Landmark-fit positions",
}


class RegistrationLoadCancelled(Exception):
    """Raised on the worker thread when a newer load or a cancel superseded the running one."""


# ----------------------------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------------------------


def contrast_limits_from_level(
    level_cyx: Any,
    channel_index: int,
    *,
    percentiles: tuple[float, float] = CONTRAST_PERCENTILES,
    max_pixels: int = CONTRAST_MAX_PIXELS,
) -> tuple[float, float]:
    """Return display contrast limits of one channel of a ``(c, y, x)`` level.

    The limits are the ``percentiles`` of the non-zero pixels of a strided subsample of at most
    about ``max_pixels`` pixels (zero is padding or background).

    Args:
        level_cyx: A ``(c, y, x)`` DataArray (typically the coarsest pyramid level).
        channel_index: Index of the channel.
        percentiles: Lower and upper percentile.
        max_pixels: Subsampling budget in pixels.

    Returns:
        ``(low, high)`` with ``high > low``; ``(0.0, 1.0)`` when the channel has no non-zero pixel.
    """
    arr = level_cyx.isel(c=channel_index)
    height, width = arr.shape
    step = max(1, math.ceil(math.sqrt(height * width / max_pixels)))
    values = np.asarray(arr.data[::step, ::step])
    values = values[np.isfinite(values)] if values.dtype.kind == "f" else values.ravel()
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return (0.0, 1.0)
    low, high = (float(v) for v in np.percentile(nonzero, percentiles))
    if not high > low:
        high = low + 1.0
    return (low, high)


def marker_world_size(zoom: float) -> float | None:
    """Return the world size (micrometres) of a marker that is ``MARKER_SCREEN_PX`` on screen."""
    try:
        zoom = float(zoom)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(zoom) or zoom <= 0:
        return None
    return MARKER_SCREEN_PX / zoom


def layer_linear_scale(affine_rc: np.ndarray) -> float:
    """Return ``sqrt(|det L|)``, the micrometres per data unit of a layer affine."""
    linear = np.asarray(affine_rc, dtype=float)[:2, :2]
    return float(math.sqrt(abs(float(np.linalg.det(linear)))))


def marker_data_size(zoom: float, affine_rc: np.ndarray) -> float | None:
    """Return the Points ``size`` (data units) of a marker of constant screen size."""
    world = marker_world_size(zoom)
    scale = layer_linear_scale(affine_rc)
    if world is None or not scale > 0:
        return None
    return world / scale


def label_translation_data(affine_rc: np.ndarray, world_size: float) -> np.ndarray:
    """Return the ``text.translation`` that puts a label up and right of its marker.

    ``Points.text.translation`` is in *data* coordinates of the layer (it is added before the
    layer transform), so the world offset (row up, column right) is mapped back through ``L``.
    """
    linear = np.asarray(affine_rc, dtype=float)[:2, :2]
    offset = np.array([-LABEL_OFFSET_FACTOR * world_size, LABEL_OFFSET_FACTOR * world_size])
    return np.asarray(np.linalg.solve(linear, offset), dtype=float)


def crop_edge_data_width(zoom: float, affine_rc: np.ndarray) -> float:
    """Return the Shapes ``edge_width`` (data units) of a crop outline of constant screen width."""
    scale = layer_linear_scale(affine_rc)
    try:
        zoom = float(zoom)
    except (TypeError, ValueError):
        return 1.0
    if not (math.isfinite(zoom) and zoom > 0 and scale > 0):
        return 1.0
    return CROP_EDGE_SCREEN_PX / zoom / scale


def point_inside_image(xy: Sequence[float], shape_yx: tuple[int, int]) -> bool:
    """Whether ``(x, y)`` lies in ``-0.5 <= x < W - 0.5``, ``-0.5 <= y < H - 0.5``."""
    height, width = shape_yx
    x, y = float(xy[0]), float(xy[1])
    return -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def crop_from_drag(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    shape_yx: tuple[int, int],
    *,
    min_px: int = CROP_MIN_DRAW_PX,
) -> CropRect | None:
    """Return the crop of a drag between two ``(x, y)`` points, or ``None`` if too small.

    ``x0 = round(min x)`` and ``x1 = round(max x) + 1`` (same for y) with a deterministic
    half-up rounding, clipped to the image.
    """
    height, width = shape_yx
    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])
    x0 = _round_half_up(min(sx, ex))
    x1 = _round_half_up(max(sx, ex)) + 1
    y0 = _round_half_up(min(sy, ey))
    y1 = _round_half_up(max(sy, ey)) + 1
    x0 = min(max(x0, 0), width - 1)
    x1 = min(max(x1, 1), width)
    y0 = min(max(y0, 0), height - 1)
    y1 = min(max(y1, 1), height)
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return CropRect(x0, y0, x1, y1)


def snapped_crop_size(
    crop: CropRect | None, shape_yx: tuple[int, int], *, multiple: int = CROP_SNAP_PX
) -> tuple[int, int]:
    """Return the ``(width, height)`` of the crop after MerXen's inward snapping (warning only)."""
    height, width = shape_yx
    if crop is None:
        return (int(width), int(height))
    x0 = -(-crop.x0 // multiple) * multiple
    y0 = -(-crop.y0 // multiple) * multiple
    x1 = width if crop.x1 == width else (crop.x1 // multiple) * multiple
    y1 = height if crop.y1 == height else (crop.y1 // multiple) * multiple
    return (max(0, x1 - x0), max(0, y1 - y0))


def crop_rectangle_rc(crop: CropRect) -> np.ndarray:
    """Return the four ``(row, col)`` vertices of a crop at pixel *edges*."""
    return np.array(
        [
            [crop.y0 - 0.5, crop.x0 - 0.5],
            [crop.y0 - 0.5, crop.x1 - 0.5],
            [crop.y1 - 0.5, crop.x1 - 0.5],
            [crop.y1 - 0.5, crop.x0 - 0.5],
        ],
        dtype=float,
    )


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def result_rows(result: dict[str, Any]) -> tuple[ResidualRow, ...]:
    """Return one ``ResidualRow`` per landmark of a ``RESULT_ATTR`` payload, sorted by id."""
    rows = []
    for record in sorted(result.get("landmarks") or [], key=lambda r: int(r["id"])):
        residual = record.get("residual_um") or {}
        affine = residual.get("affine")
        final = residual.get("final")
        rows.append(
            ResidualRow(
                id=int(record["id"]),
                colour=str(record.get("colour", DISABLED_PAIR_COLOUR)),
                enabled=bool(record.get("enabled", True)),
                used=bool(record.get("used_in_fit", False)),
                outlier=bool(record.get("outlier", False)),
                outside_crop=bool(record.get("outside_crop", False)),
                fit_um=_float_or_nan(residual.get("landmark")),
                loo_um=_float_or_nan(residual.get("loo")),
                affine_um=None if affine is None else float(affine),
                final_um=None if final is None else float(final),
            )
        )
    return tuple(rows)


def _fmt2(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}" if math.isfinite(number) else "n/a"


def result_banner(result: dict[str, Any]) -> str:
    """Return the three-line summary shown above the result table."""

    def text(value: Any) -> str:
        return "n/a" if value is None or value == "" else str(value)

    def status_part(block: Any) -> str:
        block = block if isinstance(block, dict) else {}
        reason = block.get("reason")
        return text(block.get("status")) + (f" – {reason}" if reason else "")

    label = result.get("run_label")
    pearson = (result.get("tissue_qc") or {}).get("pearson") or {}
    return (
        f"{text(result.get('run_id'))} · preset {text(result.get('preset'))}"
        f"{f' ({label})' if label else ''}\n"
        f"Refinement: {status_part(result.get('refinement'))} · "
        f"Flow: {status_part(result.get('nonrigid'))}\n"
        f"Tissue Pearson: coarse {_fmt2(pearson.get('coarse'))} → "
        f"landmark {_fmt2(pearson.get('landmark'))} → affine {_fmt2(pearson.get('affine'))} "
        f"→ final {_fmt2(pearson.get('final'))}"
    )


def _default_colour_seed(descriptor: StoreDescriptor) -> int:
    key = descriptor.grid_digest or str(descriptor.path)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100_000


def _fresh_state(descriptor: StoreDescriptor, revision: int | None) -> LabellingState:
    state = LabellingState(
        coarse=CoarseState(),
        crops={"reference": None, "moving": None},
        pairs=[],
        next_id=1,
        colour_seed=_default_colour_seed(descriptor),
        status="new",
        revision=revision,
    )
    state.loaded_content_sha = state.content_sha256()
    return state


def _software_version() -> str:
    try:
        return importlib.metadata.version("napari-compare-xenium-merscope")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ----------------------------------------------------------------------------------------------
# Host protocol and session
# ----------------------------------------------------------------------------------------------


class RegistrationHost(Protocol):
    """What the controller needs from the application (``ComparisonViewerController`` in VW3)."""

    viewer: Any

    def set_status(self, text: str) -> None:
        """Show a status message."""

    def begin_progress(self, key: str, text: str) -> None:
        """Start a progress indicator."""

    def end_progress(self, key: str) -> None:
        """End a progress indicator."""

    def run_worker(
        self,
        compute: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[BaseException], None],
        *,
        key: str,
    ) -> None:
        """Run ``compute`` on a worker (under the store-IO semaphore); callbacks on the GUI thread.

        A superseded or cancelled task never calls back; synchronous when there is no thread
        worker (tests).
        """

    def display_levels(
        self, image_key: str, sdata: Any, *, cancel_check: Callable[[], bool] | None
    ) -> list[Any]:
        """Return ``(c, y, x)`` DataArray levels (level 0 first); called on the worker thread."""

    def enter_registration_session(self, path: Path, sdata: Any) -> None:
        """Tear down the other views and clear the layers (GUI thread)."""

    def registration_session_ready(self, path: Path, kind: str) -> None:
        """The layers are built (GUI thread): recents, explorer refresh, view reset, tab."""

    def leave_registration_session(self) -> None:
        """The session ended (GUI thread)."""

    def app_data_dir(self) -> Path:
        """Return the per-user application data directory (callable from a worker thread)."""


@dataclass
class RegistrationSession:
    """Everything ``prepare_session`` read on the worker, handed to ``apply_session``.

    Attributes:
        path: The store directory.
        sdata: The SpatialData object (images only).
        descriptor: The parsed store descriptor.
        kind: ``"prepared"`` or ``"result"``.
        levels: Role to ``(c, y, x)`` levels, level 0 first.
        contrast: ``(role, channel)`` to contrast limits (from the coarsest level).
        state: The labelling state (prepared stores only).
        result: ``RESULT_ATTR`` (result stores only).
        draft: The local draft payload, if one exists (prepared stores only).
        labelling_mismatch: ``LabellingMismatch`` text, decided on the GUI thread.
        stored_revision: Revision of ``LABELLING_ATTR`` on disk at load (``None``: none).
        notes: Non-fatal warnings appended to the ready status.
    """

    path: Path
    sdata: Any
    descriptor: StoreDescriptor
    kind: str
    levels: dict[str, list[Any]]
    contrast: dict[tuple[str, str], tuple[float, float]]
    state: LabellingState | None
    result: dict[str, Any] | None
    draft: dict[str, Any] | None
    labelling_mismatch: str | None = None
    stored_revision: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class UndoAction:
    """One undoable edit."""

    kind: Literal["add", "delete", "enable", "crop"]
    pairs: tuple[LandmarkPair, ...] = ()
    pair_id: int | None = None
    enabled_before: bool | None = None
    role: str | None = None
    crop_before: CropRect | None = None


def _slot(*, prepared: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Guard a Qt slot or key handler: never raise, optionally require a prepared session."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: ManualRegistrationController, *args: Any, **kwargs: Any) -> Any:
            try:
                if prepared and not self._prepared_active():
                    self._host.set_status(_NO_SESSION_TEXT)
                    return None
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 (a slot must never raise into Qt)
                log.exception("manual registration: %s failed", fn.__name__)
                try:
                    self._host.set_status(f"Manual registration error: {exc}")
                except Exception:  # noqa: BLE001
                    log.exception("manual registration: could not report an error")
                return None

        return wrapper

    return decorate


# ----------------------------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------------------------


class ManualRegistrationController:
    """Drives a manual registration session in the napari viewer of the host."""

    def __init__(self, host: RegistrationHost) -> None:
        self._host = host
        self._viewer = host.viewer
        self._widget: Any = None
        self._widget_connections: list[tuple[Any, Callable[..., Any]]] = []
        self._session: RegistrationSession | None = None
        self._session_serial = 0
        self._display_mode = "both"
        self._auto_switch = True
        self._landmark_mode = False
        self._crop_role: str | None = None
        self._crop_restore_mode: str | None = None
        self._pending: tuple[float, float] | None = None
        self._undo: list[UndoAction] = []
        self._preview = False
        self._floor_um = 10.0
        self._live_fit: LandmarkEvaluation | None = None
        self._layers: dict[str, Any] = {}
        self._image_layers: dict[str, list[Any]] = {"reference": [], "moving": []}
        self._reg_layer: dict[str, Any] = {}
        self._layer_role: dict[int, str] = {}
        self._user_opacity: dict[int, float] = {}
        self._opacity_guard = False
        self._active_guard = False
        self._size_factors: dict[int, np.ndarray] = {}
        self._saved_keys: dict[str, Callable[..., Any] | None] = {}
        self._dblclick_removed: list[tuple[int, Callable[..., Any]]] = []
        self._load_generation = 0
        self._load_cancel = threading.Event()
        self._saving = False
        self._affine_timer: Any = None
        self._draft_timer: Any = None

    # ------------------------------------------------------------------ properties

    @property
    def session(self) -> RegistrationSession | None:
        """The active session (``None`` when no registration store is shown)."""
        return self._session

    @property
    def state(self) -> LabellingState | None:
        """The labelling state of the active prepared session."""
        return self._session.state if self._session is not None else None

    @property
    def display_mode(self) -> str:
        """The shown image(s): ``"reference"``, ``"moving"`` or ``"both"``."""
        return self._display_mode

    @property
    def pending_reference_xy(self) -> tuple[float, float] | None:
        """The reference point waiting for its moving partner, ``(x, y)``."""
        return self._pending

    def is_active(self) -> bool:
        """Whether a registration session is shown."""
        return self._session is not None

    def _prepared_active(self) -> bool:
        return self._session is not None and self._session.kind == "prepared"

    # ------------------------------------------------------------------ widget wiring

    def set_widget(self, widget: Any) -> None:
        """Connect the registration panel; pushes the current state when a session is active.

        Calling it again disconnects the previous widget first.
        """
        for signal, slot in self._widget_connections:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._widget_connections = []
        self._widget = widget
        if widget is None:
            return
        wiring = (
            (widget.rotate_by, self._on_rotate_by),
            (widget.rotation_set, self._on_rotation_set),
            (widget.shift_by, self._on_shift_by),
            (widget.offset_set, self._on_offset_set),
            (widget.flip_horizontal, self._on_flip_horizontal),
            (widget.flip_vertical, self._on_flip_vertical),
            (widget.reset_coarse, self._on_reset_coarse),
            (widget.adopt_fit, self._on_adopt_fit),
            (widget.display_mode_changed, self._on_display_mode_changed),
            (widget.flick_requested, self._on_flick_requested),
            (widget.auto_switch_changed, self._on_auto_switch_changed),
            (widget.landmark_mode_changed, self._on_landmark_mode_changed),
            (widget.pair_selected, self._on_pair_selected),
            (widget.pair_enabled_changed, self._on_pair_enabled_changed),
            (widget.delete_pairs, self._on_delete_pairs),
            (widget.undo_requested, self._on_undo_requested),
            (widget.preview_changed, self._on_preview_changed),
            (widget.outlier_floor_changed, self._on_outlier_floor_changed),
            (widget.crop_draw_requested, self._on_crop_draw_requested),
            (widget.crop_set, self._on_crop_set),
            (widget.save_requested, self._on_save_requested),
            (widget.finish_requested, self._on_finish_requested),
            (widget.overlay_changed, self._on_overlay_changed),
        )
        for signal, slot in wiring:
            signal.connect(slot)
            self._widget_connections.append((signal, slot))
        try:
            self._auto_switch = bool(widget.auto_switch())
        except Exception:  # noqa: BLE001
            self._auto_switch = True
        if self._session is not None:
            self._push_widget_state()

    # ------------------------------------------------------------------ loading

    def load_store(self, path: Path) -> bool:
        """Open a registration store: cheap checks now, the heavy read on the host's worker.

        This never prompts about unsaved changes of the current session (the caller does, with
        ``confirm_discard_unsaved``).

        Returns:
            ``True`` if a load was started; ``False`` (with a status message) otherwise.
        """
        path = Path(path)
        if not path.is_dir():
            self._host.set_status(f"Registration store not found: {path}")
            return False
        try:
            read_store_descriptor(path)
        except (NotARegistrationStore, UnsupportedSchemaVersion, OSError, ValueError) as exc:
            self._host.set_status(f"Could not open {path.name} as a registration store: {exc}")
            return False
        self._load_generation += 1
        generation = self._load_generation
        self._load_cancel.set()
        self._load_cancel = threading.Event()
        token = self._load_cancel
        self._host.begin_progress("registration", f"Loading registration store {path.name}…")
        self._host.run_worker(
            lambda: self.prepare_session(path, token.is_set),
            lambda session: self._on_session_prepared(generation, session),
            lambda exc: self._on_load_error(generation, path, exc),
            key="registration_load",
        )
        return True

    def _on_session_prepared(self, generation: int, session: RegistrationSession) -> None:
        if generation != self._load_generation:
            return
        try:
            self.apply_session(session)
        except Exception as exc:  # noqa: BLE001 (a broken session must not leave a half view)
            log.exception("manual registration: applying %s failed", session.path.name)
            self._host.end_progress("registration")
            if self._session is session:  # only a half-built NEW session is torn down
                self._teardown_session(call_host=True)
            self._host.set_status(f"Could not load registration store {session.path.name}: {exc}")

    def _on_load_error(self, generation: int, path: Path, exc: BaseException) -> None:
        if generation != self._load_generation:
            return
        self._host.end_progress("registration")
        if isinstance(exc, RegistrationLoadCancelled):
            self._host.set_status("Loading cancelled.")
            return
        log.error("manual registration: loading %s failed", path.name, exc_info=exc)
        self._host.set_status(f"Could not load registration store {path.name}: {exc}")

    def prepare_session(
        self, path: Path, cancel_check: Callable[[], bool] | None = None
    ) -> RegistrationSession:
        """Read a store into a ``RegistrationSession`` (worker thread; no Qt, no viewer access).

        Args:
            path: The registration store directory.
            cancel_check: Polled between steps; a true value raises ``RegistrationLoadCancelled``.

        Raises:
            RegistrationLoadCancelled: If cancelled.
            ValueError: If an image element is missing or its channels do not match the store.
            NotARegistrationStore: If the directory is not a registration store.
        """
        import spatialdata

        def check() -> None:
            if cancel_check is not None and cancel_check():
                raise RegistrationLoadCancelled()

        path = Path(path)
        root = read_root_attrs(path)
        descriptor = parse_store_descriptor(path, root)
        check()
        sdata = spatialdata.read_zarr(str(path), selection=("images",))
        check()
        levels: dict[str, list[Any]] = {}
        contrast: dict[tuple[str, str], tuple[float, float]] = {}
        for role in ROLES:
            record = descriptor.images[role]
            key = record.element_key
            if key not in sdata.images:
                raise ValueError(f"Store {path.name} has no image element {key!r} ({role}).")
            levels[role] = self._host.display_levels(key, sdata, cancel_check=cancel_check)
            check()
            if not levels[role]:
                raise ValueError(f"No readable levels for image {key!r}.")
            names = list(record.channel_names)
            n_channels = int(levels[role][0].shape[0])
            if len(names) != n_channels:
                raise ValueError(
                    f"Image {key!r} has {n_channels} channels but the store lists {names}."
                )
            for index, channel in enumerate(names):
                contrast[(role, channel)] = contrast_limits_from_level(levels[role][-1], index)
                check()
        if descriptor.kind == "result":
            if descriptor.result is None:
                raise ValueError(
                    f"Store {path.name} has no merxen_manual_registration_result; "
                    "the register run did not finish."
                )
            return RegistrationSession(
                path=path,
                sdata=sdata,
                descriptor=descriptor,
                kind="result",
                levels=levels,
                contrast=contrast,
                state=None,
                result=descriptor.result,
                draft=None,
            )
        stored = descriptor.labelling
        stored_revision = stored.get("revision") if stored else None
        if not isinstance(stored_revision, int) or isinstance(stored_revision, bool):
            stored_revision = None
        mismatch: str | None = None
        notes: list[str] = []
        if stored is None:
            state = _fresh_state(descriptor, revision=None)
        else:
            try:
                state = labelling_from_attrs(stored, descriptor)
            except UnsupportedSchemaVersion:
                raise
            except LabellingMismatch as exc:
                mismatch = str(exc)
                state = _fresh_state(descriptor, revision=stored_revision)
            except ValueError as exc:
                mismatch = f"the stored landmarks could not be read: {exc}"
                state = _fresh_state(descriptor, revision=stored_revision)
            if stored.get("store_grid_digest") not in (None, descriptor.grid_digest):
                notes.append(
                    "The images were re-prepared since these landmarks were saved (grid digest "
                    "changed); check that the landmarks still sit on the right structures."
                )
            notes.extend(
                w
                for w in state.load_warnings
                if not w.startswith("the store's source images changed fingerprint")
            )
        draft_file = read_local_draft(path, self._host.app_data_dir())
        draft = draft_file["payload"] if draft_file is not None else None
        return RegistrationSession(
            path=path,
            sdata=sdata,
            descriptor=descriptor,
            kind="prepared",
            levels=levels,
            contrast=contrast,
            state=state,
            result=None,
            draft=draft,
            labelling_mismatch=mismatch,
            stored_revision=stored_revision,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------ applying a session

    def apply_session(self, session: RegistrationSession) -> None:
        """Show a prepared session (GUI thread): dialogs first, then layers, keys and widget.

        Every dialog runs before anything is torn down, so cancelling leaves the current view
        intact.
        """
        host = self._host
        descriptor = session.descriptor
        if session.kind == "prepared":
            if session.labelling_mismatch:
                choice = self._ask(
                    "Landmarks do not match these images",
                    self._mismatch_text(session),
                    ["Start fresh", "Cancel"],
                    default="Cancel",
                )
                if choice != "Start fresh":
                    host.end_progress("registration")
                    host.set_status("Loading cancelled.")
                    return
            if session.draft is not None:
                self._decide_draft(session)
        if self._session is not None:
            self.flush_pending_updates()  # the old session's pending draft/affine run on IT
            self._teardown_session(call_host=False)
        self._saving = False
        if self._widget is not None:
            self._widget.set_busy(False)
        host.enter_registration_session(session.path, session.sdata)
        self._session = session
        self._session_serial += 1
        self._undo = []
        self._pending = None
        self._preview = False
        self._landmark_mode = False
        self._crop_role = None
        self._crop_restore_mode = None
        self._display_mode = "both"
        self._live_fit = None
        self._viewer.grid.enabled = False
        if session.kind == "prepared":
            assert session.state is not None
            saved = session.state.outlier_floor_um
            if saved is not None:
                self._floor_um = float(saved)
            elif self._widget is not None:
                self._floor_um = float(self._widget.outlier_floor_um())
            else:
                self._floor_um = 10.0
            self._build_image_layers(session)
            self._build_prepared_overlays(session)
        else:
            self._build_image_layers(session)
            self._build_result_overlays(session)
        self._bind_session_keys()
        self._connect_viewer_events()
        self._viewer.layers.selection.active = self._reg_layer["reference"]
        if session.kind == "prepared":
            self._refit()
            self._apply_moving_affine()
            self._sync_markers()
            self._sync_crops()
        else:
            self._resize_markers()
        self._apply_display("both")
        self._push_widget_state()
        host.registration_session_ready(session.path, session.kind)
        if session.kind == "prepared":
            assert session.state is not None
            n_pairs = len(session.state.pairs)
            status = session.state.status
            text = (
                f"{descriptor.dataset_name}: {n_pairs} landmark pairs "
                f"({'not saved yet' if status == 'new' else status}). "
                "Turn on Landmark mode to add pairs; F flicks."
            )
        else:
            run_id = (session.result or {}).get("run_id", descriptor.dataset_name)
            text = f"{run_id}: reference and registered moving image. F flicks; the table lists residuals."
        if session.notes:
            text += " " + " ".join(session.notes)
        host.set_status(text)

    @staticmethod
    def _mismatch_text(session: RegistrationSession) -> str:
        name = session.path.name
        mismatch = session.labelling_mismatch or ""
        if mismatch.startswith("the stored landmarks could not be read"):
            return (
                f"The landmarks saved in {name} cannot be used ({mismatch}). Start fresh? "
                "The old landmarks stay in the store until you save."
            )
        return (
            f"The landmarks saved in {name} were placed on images of a different size or pixel "
            f"size ({mismatch}). Start fresh? The old landmarks stay in the store until you save."
        )

    def _decide_draft(self, session: RegistrationSession) -> None:
        """Offer a newer local draft for restore (prepared stores; may replace ``session.state``)."""
        assert session.state is not None
        app_dir = self._host.app_data_dir()
        path = session.path
        draft = session.draft
        assert draft is not None
        stored = session.descriptor.labelling
        try:
            draft_sha = labelling_content_sha256(draft)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            self._discard_bad_draft(
                session, f"The local draft could not be read ({exc}); discarded."
            )
            return
        # A malformed STORED labelling must never cost the user the draft (their only copy of
        # that work): treat it as different from the draft and older, so Restore is offered.
        same = False
        newer = True
        if stored is not None:
            try:
                same = draft_sha == labelling_content_sha256(stored)
                newer = str(draft.get("updated_at", "")) > str(stored.get("updated_at", ""))
            except (KeyError, TypeError, ValueError, AttributeError):
                same, newer = False, True
        if same or not newer:
            clear_local_draft(path, app_dir)
            return
        choice = self._ask(
            "Restore local draft?",
            f"A local draft from {draft.get('updated_at')} is newer than the landmarks saved in "
            f"the store ({stored.get('updated_at') if stored else 'none'}). Restore it?",
            ["Restore", "Discard"],
            default="Restore",
        )
        if choice != "Restore":
            clear_local_draft(path, app_dir)
            return
        try:
            restored = labelling_from_attrs(draft, session.descriptor)
        except LabellingMismatch:
            self._discard_bad_draft(
                session, "The local draft does not match these images; discarded."
            )
            return
        except ValueError as exc:
            self._discard_bad_draft(
                session, f"The local draft could not be read ({exc}); discarded."
            )
            return
        restored.revision = session.stored_revision  # conflict detection uses the STORE revision
        restored.loaded_content_sha = session.state.loaded_content_sha  # so "unsaved" is True
        restored.status = session.state.status
        session.state = restored

    def _discard_bad_draft(self, session: RegistrationSession, message: str) -> None:
        clear_local_draft(session.path, self._host.app_data_dir())
        self._host.set_status(message)
        session.notes = (*session.notes, message)

    # ------------------------------------------------------------------ layers

    def _role_affine(self, role: str) -> np.ndarray:
        assert self._session is not None
        return np.asarray(self._session.descriptor.images[role].base_affine_rc, dtype=float)

    def _coarse_matrix(self) -> np.ndarray:
        assert self._session is not None and self._session.state is not None
        images = self._session.descriptor.images
        return coarse_matrix_xy(
            self._session.state.coarse,
            moving_shape_yx=images["moving"].shape_yx,
            reference_shape_yx=images["reference"].shape_yx,
            moving_pixel_size_um=images["moving"].pixel_size_um,
            reference_pixel_size_um=images["reference"].pixel_size_um,
        )

    def _moving_affine_rc(self) -> np.ndarray:
        matrix = self._coarse_matrix()
        if self._preview and self._live_fit is not None and self._live_fit.matrix is not None:
            matrix = self._live_fit.matrix
        return napari_affine_rc(self._role_affine("reference"), matrix)

    def _register_layer(self, layer: Any, role: str) -> None:
        self._layer_role[id(layer)] = role

    def _build_image_layers(self, session: RegistrationSession) -> None:
        prepared = session.kind == "prepared"
        self._layers = {}
        self._image_layers = {"reference": [], "moving": []}
        self._reg_layer = {}
        self._layer_role = {}
        self._user_opacity = {}
        self._size_factors = {}
        moving_affine = self._moving_affine_rc() if prepared else None
        for role in ROLES:
            record = session.descriptor.images[role]
            prefix = (
                "Reference"
                if role == "reference"
                else ("Moving" if prepared else "Registered moving")
            )
            for index, channel in enumerate(record.channel_names):
                data = [level.isel(c=index).data for level in session.levels[role]]
                is_reg = channel == record.registration_channel
                if role == "reference":
                    affine = self._role_affine("reference")
                elif prepared:
                    affine = moving_affine
                else:
                    affine = self._role_affine("moving")
                colormap = (
                    (REFERENCE_COLORMAP if role == "reference" else MOVING_COLORMAP)
                    if is_reg
                    else OTHER_CHANNEL_COLORMAP
                )
                layer = Image(
                    data if len(data) > 1 else data[0],
                    multiscale=len(data) > 1,
                    name=f"{prefix} | {channel}",
                    affine=affine,
                    colormap=colormap,
                    blending="additive",
                    contrast_limits=session.contrast[(role, channel)],
                    visible=is_reg,
                    opacity=1.0,
                    cache=True,
                    metadata={
                        LAYER_METADATA_KEY: {"role": role, "kind": "image", "channel": channel}
                    },
                )
                self._viewer.add_layer(layer)
                layer.events.mode.connect(self._on_layer_mode)
                layer.events.opacity.connect(self._on_layer_opacity)
                self._layers[layer.name] = layer
                self._image_layers[role].append(layer)
                self._register_layer(layer, role)
                if is_reg:
                    self._reg_layer[role] = layer

    def _new_points_layer(self, name: str, role: str, kind: str, affine: np.ndarray) -> Any:
        layer = Points(
            np.empty((0, 2)),
            ndim=2,
            name=name,
            affine=affine,
            features=pd.DataFrame(
                {"label": pd.Series([], dtype=object), "colour": pd.Series([], dtype=object)}
            ),
            text={
                "string": "{label}",
                "color": {"feature": "colour"},
                "size": LABEL_TEXT_SIZE,
                "anchor": "center",
            },
            border_color="transparent",
            border_width_is_relative=True,
            blending="translucent",
            metadata={LAYER_METADATA_KEY: {"role": role, "kind": kind, "channel": None}},
        )
        layer.editable = False
        layer.mode = "pan_zoom"
        return layer

    def _new_crop_layer(self, name: str, role: str, affine: np.ndarray) -> Any:
        layer = Shapes(
            name=name,
            ndim=2,
            affine=affine,
            edge_color=CROP_COLOUR,
            face_color="transparent",
            metadata={LAYER_METADATA_KEY: {"role": role, "kind": "crop", "channel": None}},
        )
        layer.editable = False
        layer.mode = "pan_zoom"
        return layer

    def _add_controller_layer(self, layer: Any, role: str) -> None:
        self._viewer.add_layer(layer)
        self._layers[layer.name] = layer
        self._register_layer(layer, role)

    def _build_prepared_overlays(self, session: RegistrationSession) -> None:
        moving_affine = self._moving_affine_rc()
        reference_affine = self._role_affine("reference")
        self._add_controller_layer(
            self._new_crop_layer("Reference crop", "reference", reference_affine), "reference"
        )
        self._add_controller_layer(
            self._new_crop_layer("Moving crop", "moving", moving_affine), "moving"
        )
        self._add_controller_layer(
            self._new_points_layer("Moving landmarks", "moving", "landmarks", moving_affine),
            "moving",
        )
        self._add_controller_layer(
            self._new_points_layer(
                "Reference landmarks", "reference", "landmarks", reference_affine
            ),
            "reference",
        )

    def _build_result_overlays(self, session: RegistrationSession) -> None:
        p_r = session.descriptor.images["reference"].pixel_size_um
        affine = np.diag([p_r, p_r, 1.0])
        result = session.result or {}
        records = sorted(result.get("landmarks") or [], key=lambda r: int(r["id"]))
        fit_layer = self._new_points_layer("Landmark-fit positions", "reference", "overlay", affine)
        reg_layer = self._new_points_layer(
            "Registered moving landmarks", "reference", "overlay", affine
        )
        ref_layer = self._new_points_layer("Reference landmarks", "reference", "overlay", affine)
        fit_layer.visible = False
        for layer in (fit_layer, reg_layer, ref_layer):
            self._add_controller_layer(layer, "reference")
        colours = {int(r["id"]): self._row_colour(r) for r in records}

        coords: list[Any] = []
        symbols: list[str] = []
        faces: list[str] = []
        labels: list[str] = []
        factors: list[float] = []
        widths: list[float] = []
        for record in records:
            pid = int(record["id"])
            xy = record.get("reference_xy_px")
            if xy is None:
                log.debug("result landmark %s has no reference position; skipped", pid)
                continue
            label = f"{pid}*" if record.get("outside_crop") else str(pid)
            for symbol, factor, width, text in (
                ("ring", 1.0, RING_BORDER_WIDTH, label),
                ("cross", CROSS_SIZE_FACTOR, CROSS_BORDER_WIDTH, ""),
            ):
                coords.append((float(xy[1]), float(xy[0])))
                symbols.append(symbol)
                faces.append(colours[pid])
                labels.append(text)
                factors.append(factor)
                widths.append(width)
        self._assign_points(ref_layer, coords, symbols, faces, labels, factors, widths)

        for layer, stage, symbol, factor, width, with_label in (
            (reg_layer, "final", "x", 0.8, CROSS_BORDER_WIDTH, True),
            (fit_layer, "landmark", "diamond", 0.6, 0.0, False),
        ):
            coords = []
            faces = []
            labels = []
            for record in records:
                pid = int(record["id"])
                position = (record.get("moving_in_reference_xy_px") or {}).get(stage)
                if position is None:
                    log.debug("result landmark %s has no %s position; skipped", pid, stage)
                    continue
                coords.append((float(position[1]), float(position[0])))
                faces.append(colours[pid])
                labels.append(str(pid) if with_label else "")
            n = len(coords)
            self._assign_points(
                layer, coords, [symbol] * n, faces, labels, [factor] * n, [width] * n
            )

    @staticmethod
    def _row_colour(record: dict[str, Any]) -> str:
        if not record.get("enabled", True):
            return DISABLED_PAIR_COLOUR
        colour = record.get("colour")
        return colour if isinstance(colour, str) else DISABLED_PAIR_COLOUR

    # ------------------------------------------------------------------ points and shapes

    def _zoom(self) -> float:
        try:
            return float(self._viewer.camera.zoom)
        except (TypeError, ValueError, AttributeError):
            return float("nan")

    @staticmethod
    def _layer_matrix(layer: Any) -> np.ndarray:
        return np.asarray(layer.affine.affine_matrix, dtype=float)

    def _assign_points(
        self,
        layer: Any,
        coords_rc: Sequence[Sequence[float]],
        symbols: Sequence[str],
        colours: Sequence[str],
        labels: Sequence[str],
        size_factors: Sequence[float],
        border_widths: Sequence[float],
    ) -> None:
        """Replace every row of a marker layer at once (all properties together)."""
        n = len(coords_rc)
        matrix = self._layer_matrix(layer)
        zoom = self._zoom()
        world = marker_world_size(zoom)
        if world is None:
            world = MARKER_SCREEN_PX
        base = marker_data_size(zoom, matrix)
        if base is None:
            base = world / max(layer_linear_scale(matrix), 1e-12)
        factors = np.asarray(size_factors, dtype=float).reshape(-1)
        layer.selected_data = set()
        with layer.events.blocker_all():
            layer.data = np.asarray(coords_rc, dtype=float).reshape(n, 2)
            layer.symbol = list(symbols) if n else "ring"
            if n:  # (an empty colour array makes napari warn; 0 rows need no colours)
                layer.face_color = transform_color(list(colours))
                layer.border_color = np.zeros((n, 4))
            layer.border_width = np.asarray(border_widths, dtype=float).reshape(-1)
            layer.border_width_is_relative = True
            layer.features = pd.DataFrame(
                {"label": list(labels), "colour": list(colours)}, dtype=object
            )
            layer.size = base * factors
            layer.text.translation = label_translation_data(matrix, world)
        layer.refresh()
        self._size_factors[id(layer)] = factors

    def _sync_markers(self) -> None:
        """Redraw the landmark markers of both roles from the labelling state."""
        session = self._session
        if session is None or session.kind != "prepared" or session.state is None:
            return
        state = session.state
        outside = {
            row.id: row.outside_crop for row in (self._live_fit.rows if self._live_fit else ())
        }
        for role in ROLES:
            layer = self._layers.get(
                f"{'Reference' if role == 'reference' else 'Moving'} landmarks"
            )
            if layer is None:
                continue
            coords: list[tuple[float, float]] = []
            symbols: list[str] = []
            colours: list[str] = []
            labels: list[str] = []
            factors: list[float] = []
            widths: list[float] = []
            for pair in sorted(state.pairs, key=lambda p: p.id):
                x, y = pair.reference_xy if role == "reference" else pair.moving_xy
                colour = pair.colour if pair.enabled else DISABLED_PAIR_COLOUR
                label = f"{pair.id}*" if outside.get(pair.id) else str(pair.id)
                for symbol, factor, width, text in (
                    ("ring", 1.0, RING_BORDER_WIDTH, label),
                    ("cross", CROSS_SIZE_FACTOR, CROSS_BORDER_WIDTH, ""),
                ):
                    coords.append((y, x))
                    symbols.append(symbol)
                    colours.append(colour)
                    labels.append(text)
                    factors.append(factor)
                    widths.append(width)
            if role == "reference" and self._pending is not None:
                x, y = self._pending
                coords.append((y, x))
                symbols.append("ring")
                colours.append(pair_colour(state.next_id, state.colour_seed))
                labels.append(str(state.next_id))
                factors.append(1.0)
                widths.append(RING_BORDER_WIDTH)
            self._assign_points(layer, coords, symbols, colours, labels, factors, widths)

    def _sync_crops(self) -> None:
        """Redraw the crop outlines."""
        session = self._session
        if session is None or session.kind != "prepared" or session.state is None:
            return
        zoom = self._zoom()
        for role in ROLES:
            layer = self._layers.get(f"{'Reference' if role == 'reference' else 'Moving'} crop")
            if layer is None:
                continue
            self._draw_crop(layer, session.state.crops.get(role), zoom)

    def _draw_crop(self, layer: Any, crop: CropRect | None, zoom: float) -> None:
        matrix = self._layer_matrix(layer)
        layer.selected_data = set()
        layer.data = []
        if crop is not None:
            layer.add_rectangles(
                crop_rectangle_rc(crop),
                edge_color=CROP_COLOUR,
                face_color="transparent",
                edge_width=crop_edge_data_width(zoom, matrix),
            )

    def _resize_markers(self) -> None:
        """Recompute marker sizes, label offsets and crop widths for the current zoom and affines."""
        zoom = self._zoom()
        if not math.isfinite(zoom) or zoom <= 0:
            return
        world = marker_world_size(zoom)
        for layer in self._layers.values():
            if isinstance(layer, Points):
                factors = self._size_factors.get(id(layer))
                if factors is None or len(layer.data) == 0 or world is None:
                    continue
                matrix = self._layer_matrix(layer)
                base = marker_data_size(zoom, matrix)
                if base is None:
                    continue
                layer.size = base * factors
                layer.text.translation = label_translation_data(matrix, world)
            elif isinstance(layer, Shapes):
                n = len(layer.data)
                if n:
                    layer.edge_width = [crop_edge_data_width(zoom, self._layer_matrix(layer))] * n

    def _on_zoom(self, event: Any = None) -> None:
        try:
            self._resize_markers()
        except Exception:  # noqa: BLE001
            log.exception("manual registration: resizing markers failed")

    # ------------------------------------------------------------------ viewer events and keys

    def _connect_viewer_events(self) -> None:
        self._viewer.camera.events.zoom.connect(self._on_zoom)
        self._viewer.layers.selection.events.active.connect(self._keep_markers_inactive)
        for layer in self._layers.values():
            if self._session is not None and self._session.kind == "result":
                if layer.name in _OVERLAY_LAYERS.values():
                    layer.events.visible.connect(self._on_overlay_layer_visible)

    def _keep_markers_inactive(self, event: Any = None) -> None:
        if self._active_guard or self._session is None:
            return
        active = self._viewer.layers.selection.active
        if active is None:
            return
        meta = (getattr(active, "metadata", None) or {}).get(LAYER_METADATA_KEY) or {}
        if meta.get("kind") in ("landmarks", "crop", "overlay") and "reference" in self._reg_layer:
            self._active_guard = True
            try:
                self._viewer.layers.selection.active = self._reg_layer["reference"]
            finally:
                self._active_guard = False

    def _on_layer_mode(self, event: Any) -> None:
        layer = getattr(event, "source", None)
        if layer is not None and str(layer.mode) == "transform":
            layer.mode = "pan_zoom"
            self._host.set_status(_TRANSFORM_TEXT)

    def _on_layer_opacity(self, event: Any) -> None:
        if self._opacity_guard:
            return
        layer = getattr(event, "source", None)
        if layer is None:
            return
        role = self._layer_role.get(id(layer))
        if role is not None and role in SHOWN_ROLES[self._display_mode]:
            self._user_opacity[id(layer)] = float(layer.opacity)

    def _bind_session_keys(self) -> None:
        handlers: dict[str, Callable[..., Any]] = {
            "F": self._key_flick,
            "Shift-F": self._key_peek,
            "Control-Z": self._key_undo,
            "Escape": self._key_escape,
            "Control-Alt-T": self._key_transform,
            "Control-Delete": self._key_delete,
            "Control-Backspace": self._key_delete,
        }
        self._saved_keys = {}
        for key in SESSION_KEYS:
            self._saved_keys[key] = self._viewer.keymap.get(KeyBinding.from_str(key))
            self._viewer.bind_key(key, handlers[key], overwrite=True)

    def _restore_session_keys(self) -> None:
        for key, saved in self._saved_keys.items():
            try:
                if saved is not None:
                    self._viewer.bind_key(key, saved, overwrite=True)
                else:
                    self._viewer.bind_key(key, None)
            except Exception:  # noqa: BLE001
                log.exception("manual registration: could not restore key %s", key)
        self._saved_keys = {}

    def _key_flick(self, viewer: Any = None) -> None:
        self._guarded_call(lambda: self._apply_display(FLICK_NEXT[self._display_mode]))

    def _key_peek(self, viewer: Any = None) -> Iterator[None]:
        self._guarded_call(lambda: self._apply_display(PEEK_OF[self._display_mode], remember=False))
        yield
        self._guarded_call(lambda: self._apply_display(self._display_mode, remember=False))

    def _key_undo(self, viewer: Any = None) -> None:
        if self._prepared_active():
            self._guarded_call(self._undo_or_cancel)
        else:
            self._host.set_status("Nothing to undo in a result store.")

    def _key_escape(self, viewer: Any = None) -> None:
        if self._prepared_active():
            self._guarded_call(self._escape)
        else:
            self._host.set_status("Nothing to cancel in a result store.")

    def _key_transform(self, viewer: Any = None) -> None:
        self._host.set_status(_TRANSFORM_TEXT)

    def _key_delete(self, viewer: Any = None) -> None:
        self._host.set_status(_DELETE_TEXT)

    def _guarded_call(self, fn: Callable[[], Any]) -> None:
        try:
            if self._session is None:
                return
            fn()
        except Exception as exc:  # noqa: BLE001
            log.exception("manual registration: key handler failed")
            self._host.set_status(f"Manual registration error: {exc}")

    def _update_double_click(self) -> None:
        """Remove ``double_click_to_zoom`` while landmark or crop mode is armed; restore after."""
        callbacks = self._viewer.mouse_double_click_callbacks
        armed = self._session is not None and (self._landmark_mode or self._crop_role is not None)
        if armed:
            for index in reversed(range(len(callbacks))):
                if getattr(callbacks[index], "__name__", "") == "double_click_to_zoom":
                    self._dblclick_removed.append((index, callbacks.pop(index)))
        else:
            for index, func in sorted(self._dblclick_removed, key=lambda item: item[0]):
                if func not in callbacks:
                    callbacks.insert(min(index, len(callbacks)), func)
            self._dblclick_removed = []

    # ------------------------------------------------------------------ display (flick)

    def _apply_display(self, mode: str, *, remember: bool = True) -> None:
        shown = SHOWN_ROLES[mode]
        self._opacity_guard = True
        try:
            for layer in self._layers.values():
                role = self._layer_role.get(id(layer))
                if role is None:
                    continue
                target = self._user_opacity.get(id(layer), 1.0) if role in shown else 0.0
                if layer.opacity != target:
                    layer.opacity = target
        finally:
            self._opacity_guard = False
        if remember:
            self._display_mode = mode
            if self._widget is not None:
                self._widget.set_display_mode(mode)

    # ------------------------------------------------------------------ timers and affines

    def _timer(self, name: str) -> Any:
        """Return the lazily created single-shot timer ``name`` (``None`` without a Qt app)."""
        attr = "_affine_timer" if name == "affine" else "_draft_timer"
        timer = getattr(self, attr)
        if timer is not None:
            return timer
        if QCoreApplication.instance() is None:
            return None
        timer = QTimer()
        timer.setSingleShot(True)
        if name == "affine":
            timer.setInterval(AFFINE_UPDATE_DELAY_MS)
            timer.timeout.connect(self._on_affine_timeout)
        else:
            timer.setInterval(DRAFT_WRITE_DELAY_MS)
            timer.timeout.connect(self._on_draft_timeout)
        setattr(self, attr, timer)
        return timer

    def _on_affine_timeout(self) -> None:
        try:
            self._apply_moving_affine()
        except Exception:  # noqa: BLE001
            log.exception("manual registration: affine update failed")

    def _on_draft_timeout(self) -> None:
        self._write_draft()

    def _schedule_affine_update(self) -> None:
        timer = self._timer("affine")
        if timer is None:
            self._apply_moving_affine()
        elif not timer.isActive():
            timer.start()

    def _schedule_draft_write(self) -> None:
        timer = self._timer("draft")
        if timer is None:
            self._write_draft()
        else:
            timer.start()  # restarts: written 500 ms after the last change

    def _apply_moving_affine(self) -> None:
        """Assign the current moving placement to every moving-side layer (prepared only)."""
        session = self._session
        if session is None or session.kind != "prepared":
            return
        affine = self._moving_affine_rc()
        for layer in (
            *self._image_layers["moving"],
            self._layers.get("Moving landmarks"),
            self._layers.get("Moving crop"),
        ):
            if layer is not None:
                layer.affine = affine
        self._resize_markers()

    def flush_pending_updates(self) -> None:
        """Run the pending coalesced affine update and the pending draft write now."""
        timer = self._affine_timer
        if timer is not None and timer.isActive():
            timer.stop()
            self._apply_moving_affine()
        timer = self._draft_timer
        if timer is not None and timer.isActive():
            timer.stop()
            self._write_draft()

    # ------------------------------------------------------------------ live fit and widget push

    def _refit(self) -> None:
        session = self._session
        if session is None or session.kind != "prepared" or session.state is None:
            return
        images = session.descriptor.images
        previous = self._live_fit.matrix if self._live_fit is not None else None
        fit = evaluate_landmarks(
            session.state,
            moving_pixel_size_um=images["moving"].pixel_size_um,
            reference_pixel_size_um=images["reference"].pixel_size_um,
            outlier_floor_um=self._floor_um,
        )
        self._live_fit = fit
        if self._preview:
            if len(session.state.enabled_pairs()) < PREVIEW_MIN_ENABLED or fit.matrix is None:
                self._preview = False
                if self._widget is not None:
                    self._widget.set_preview(False)
                    self._widget.set_coarse_enabled(True)
                self._schedule_affine_update()
            elif previous is None or not np.array_equal(previous, fit.matrix):
                self._schedule_affine_update()
        if fit.error:
            self._host.set_status(fit.error)

    def _content_sha(self) -> str:
        assert self._session is not None and self._session.state is not None
        return self._session.state.content_sha256()

    def has_unsaved_changes(self) -> bool:
        """Whether the labelling differs from what was last loaded or saved.

        A pending half-pair does not count; a result store never has unsaved changes.
        """
        session = self._session
        if session is None or session.kind != "prepared" or session.state is None:
            return False
        return self._content_sha() != session.state.loaded_content_sha

    def _dirty_text(self) -> str:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        return (
            f"{len(state.pairs)} pairs · {'not saved' if state.status == 'new' else state.status}"
            f" · {'unsaved changes' if self.has_unsaved_changes() else 'saved'}"
        )

    def _update_dirty(self) -> None:
        if self._widget is None or not self._prepared_active():
            return
        assert self._session is not None and self._session.state is not None
        n_enabled = len(self._session.state.enabled_pairs())
        self._widget.set_dirty(self.has_unsaved_changes(), self._dirty_text())
        self._widget.set_finish_state(
            *finish_allowed(n_enabled, self._pending is not None, self._saving)
        )

    def _push_widget_rows(self) -> None:
        widget = self._widget
        session = self._session
        if widget is None or session is None or session.state is None:
            return
        state = session.state
        fit = self._live_fit
        if fit is not None:
            widget.set_rows(fit.rows)
            widget.set_fit_summary(fit.summary)
        n_enabled = len(state.enabled_pairs())
        widget.set_preview_available(n_enabled >= PREVIEW_MIN_ENABLED)
        for role in ROLES:
            crop = state.crops.get(role)
            if state.pairs:
                points = np.array(
                    [p.reference_xy if role == "reference" else p.moving_xy for p in state.pairs],
                    dtype=float,
                )
                n_outside = int(points_outside_crop(points, crop).sum())
            else:
                n_outside = 0
            widget.set_crop(role, crop, session.descriptor.images[role].shape_yx, n_outside)
        self._update_dirty()

    def _push_widget_state(self) -> None:
        """Push the complete state of the session to the widget (setters never emit)."""
        widget = self._widget
        session = self._session
        if widget is None or session is None:
            return
        if session.kind == "result":
            result = session.result or {}
            widget.set_session("result", str(result.get("run_id", session.descriptor.dataset_name)))
            widget.set_display_mode(self._display_mode)
            widget.set_result(result_rows(result), result_banner(result))
            for key, name in _OVERLAY_LAYERS.items():
                layer = self._layers.get(name)
                if layer is not None:
                    widget.set_overlay_visibility(key, bool(layer.visible))
            return
        assert session.state is not None
        widget.set_session("prepared", session.descriptor.dataset_name)
        widget.set_busy(self._saving)
        widget.set_coarse(session.state.coarse)
        widget.set_coarse_enabled(not self._preview)
        widget.set_preview(self._preview)
        widget.set_display_mode(self._display_mode)
        widget.set_landmark_mode(self._landmark_mode)
        widget.set_crop_drawing(self._crop_role)
        widget.set_outlier_floor(self._floor_um)
        if self._landmark_mode:
            widget.set_pending(
                "moving" if self._pending is not None else "reference", session.state.next_id
            )
        else:
            widget.set_pending(None, None)
        self._push_widget_rows()

    # ------------------------------------------------------------------ editing core

    def _content_changed(self) -> None:
        """A pair, crop or coarse value changed: refit, redraw, update the widget, schedule a draft."""
        self._refit()
        self._sync_markers()
        self._push_widget_rows()
        self._schedule_draft_write()

    def _push_undo(self, action: UndoAction) -> None:
        self._undo.append(action)
        if len(self._undo) > UNDO_LIMIT:
            del self._undo[: len(self._undo) - UNDO_LIMIT]

    def _set_crop(self, role: str, crop: CropRect | None, *, undoable: bool) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        if undoable:
            self._push_undo(UndoAction("crop", role=role, crop_before=state.crops.get(role)))
        state.crops[role] = crop
        self._sync_crops()
        self._content_changed()

    def _undo_or_cancel(self) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        if self._pending is not None:
            self._clear_pending("Pending point removed.")
            return
        if not self._undo:
            self._host.set_status("Nothing to undo.")
            return
        action = self._undo.pop()
        if action.kind == "add":
            state.remove_pairs(p.id for p in action.pairs)
        elif action.kind == "delete":
            state.restore_pairs(action.pairs)
        elif action.kind == "enable" and action.pair_id is not None:
            state.set_enabled(action.pair_id, bool(action.enabled_before))
        elif action.kind == "crop" and action.role is not None:
            state.crops[action.role] = action.crop_before
            self._sync_crops()
        self._content_changed()
        self._host.set_status(f"Undid: {action.kind}.")

    def _clear_pending(self, message: str | None) -> None:
        had = self._pending is not None
        self._pending = None
        if had and self._landmark_mode and self._auto_switch and self._display_mode == "moving":
            self._apply_display("reference")  # the next click is a reference click again
        self._sync_markers()
        if self._widget is not None and self._prepared_active():
            assert self._session is not None and self._session.state is not None
            if self._landmark_mode:
                self._widget.set_pending("reference", self._session.state.next_id)
            self._update_dirty()
        if message and had:
            self._host.set_status(message)

    def _escape(self) -> None:
        if self._pending is not None:
            self._clear_pending("Pending point removed.")
        elif self._crop_role is not None:
            self._disarm_crop("Crop drawing cancelled.")
        elif self._landmark_mode:
            self._set_landmark_mode(False)
            if self._widget is not None:
                self._widget.set_landmark_mode(False)

    def _set_landmark_mode(self, active: bool) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        if active:
            self._landmark_mode = True
            self._update_double_click()
            if self._widget is not None:
                self._widget.set_pending("reference", state.next_id)
            self._host.set_status(
                f"Landmark mode: click the REFERENCE point of pair {state.next_id}."
            )
            if self._auto_switch:
                self._apply_display("reference")
        else:
            self._landmark_mode = False
            had_pending = self._pending is not None
            self._pending = None
            self._update_double_click()
            if had_pending:
                self._sync_markers()
                self._host.set_status("Pending reference point discarded.")
            if self._widget is not None:
                self._widget.set_pending(None, None)
            self._update_dirty()

    def _arm_crop(self, role: str) -> None:
        if self._crop_role is None:
            self._crop_restore_mode = self._display_mode
        self._crop_role = role
        self._apply_display(role)
        self._update_double_click()
        if self._widget is not None:
            self._widget.set_crop_drawing(role)
        self._host.set_status(f"Drag a rectangle on the {role} image (Escape cancels).")

    def _disarm_crop(self, message: str | None = None) -> None:
        restore = self._crop_restore_mode
        self._crop_role = None
        self._crop_restore_mode = None
        self._sync_crops()
        if restore is not None:
            self._apply_display(restore)
        self._update_double_click()
        if self._widget is not None:
            self._widget.set_crop_drawing(None)
        if message:
            self._host.set_status(message)

    # ------------------------------------------------------------------ mouse gestures

    @staticmethod
    def _canvas_position(event: Any) -> np.ndarray | None:
        pos = getattr(event, "pos", None)
        if pos is None:
            return None
        try:
            return np.asarray(pos, dtype=float).ravel()[:2].copy()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _world_position(event: Any) -> np.ndarray:
        return np.asarray(event.position, dtype=float).ravel()[-2:].copy()

    def mouse_gesture(self, event: Any) -> Iterator[None] | None:
        """Return a press-move-release generator if the press belongs to the session, else ``None``.

        Only the left button of a *press* on a prepared session is consumed, and only while crop
        drawing or landmark mode is armed (crop drawing wins). The decision is made here,
        synchronously; the generator performs the gesture.
        """
        session = self._session
        if session is None or session.kind != "prepared":
            return None
        if "press" not in str(getattr(event, "type", "mouse_press")):
            return None
        if getattr(event, "button", 1) not in (1, None):
            return None
        if self._crop_role is not None:
            return self._crop_gesture(event, self._crop_role)
        if self._landmark_mode:
            return self._landmark_gesture(event)
        return None

    def _moved_beyond_threshold(self, press_px: np.ndarray | None, event: Any) -> bool:
        px = self._canvas_position(event)
        return (
            press_px is None
            or px is None
            or float(np.linalg.norm(px - press_px)) > CLICK_DRAG_THRESHOLD_PX
        )

    def _landmark_gesture(self, event: Any) -> Iterator[None]:
        press_world = self._world_position(event)  # napari mutates the event
        press_px = self._canvas_position(event)
        dragged = False
        yield
        while str(event.type) == "mouse_move":
            if self._moved_beyond_threshold(press_px, event):
                dragged = True
            yield
        if dragged or "release" not in str(event.type):
            return
        try:
            self._handle_landmark_click(press_world)
        except Exception as exc:  # noqa: BLE001
            log.exception("manual registration: landmark click failed")
            self._host.set_status(f"Manual registration error: {exc}")

    def _handle_landmark_click(self, world: np.ndarray) -> None:
        session = self._session
        if session is None or session.state is None:
            return
        state = session.state
        role = (
            "moving"
            if (self._display_mode == "moving" or (self._display_mode == "both" and self._pending))
            else "reference"
        )
        rc = np.asarray(self._reg_layer[role].world_to_data(world), dtype=float).ravel()[-2:]
        xy = (round(float(rc[1]), 3), round(float(rc[0]), 3))
        if not point_inside_image(xy, session.descriptor.images[role].shape_yx):
            self._host.set_status(f"Click inside the {role} image.")
            return
        if role == "reference":
            self._pending = xy
            self._sync_markers()
            if self._widget is not None:
                self._widget.set_pending("moving", state.next_id)
            self._update_dirty()
            self._host.set_status(
                f"Pair {state.next_id}: reference point placed; "
                "now click the same structure in the MOVING image."
            )
            if self._auto_switch and self._display_mode != "both":
                self._apply_display("moving")
            return
        if self._pending is None:
            self._host.set_status("Place the reference point first (F).")
            return
        pid = state.next_id
        state.next_id += 1
        pair = LandmarkPair(
            id=pid,
            colour=pair_colour(pid, state.colour_seed),
            reference_xy=self._pending,
            moving_xy=xy,
            enabled=True,
        )
        state.pairs.append(pair)
        state.pairs.sort(key=lambda p: p.id)
        self._pending = None
        self._push_undo(UndoAction("add", pairs=(pair,)))
        self._content_changed()
        if self._widget is not None:
            self._widget.set_pending("reference", state.next_id)
        row = next(
            (r for r in (self._live_fit.rows if self._live_fit else ()) if r.id == pid), None
        )
        if row is not None and math.isfinite(row.fit_um):
            self._host.set_status(f"Pair {pid} added (fit residual {row.fit_um:.1f} µm).")
        else:
            self._host.set_status(f"Pair {pid} added.")
        if self._auto_switch and self._display_mode != "both":
            self._apply_display("reference")

    def _crop_gesture(self, event: Any, role: str) -> Iterator[None]:
        camera = self._viewer.camera
        prev_pan = camera.mouse_pan
        camera.mouse_pan = False
        start = self._world_position(event)
        press_px = self._canvas_position(event)
        dragged = False
        last_draw = 0.0
        try:
            yield
            while str(event.type) == "mouse_move":
                if self._moved_beyond_threshold(press_px, event):
                    dragged = True
                now = time.monotonic()
                if dragged and now - last_draw >= AFFINE_UPDATE_DELAY_MS / 1000:
                    self._preview_crop(role, start, self._world_position(event))
                    last_draw = now
                yield
            if not dragged or "release" not in str(event.type):
                self._sync_crops()
                self._host.set_status("Drag a rectangle to draw the crop (Escape cancels).")
                return
            self._finish_crop_draw(role, start, self._world_position(event))
        finally:
            camera.mouse_pan = prev_pan if prev_pan is not None else True

    def _world_to_xy(self, role: str, world: np.ndarray) -> tuple[float, float]:
        rc = np.asarray(self._reg_layer[role].world_to_data(world), dtype=float).ravel()[-2:]
        return (float(rc[1]), float(rc[0]))

    def _preview_crop(self, role: str, w0: np.ndarray, w1: np.ndarray) -> None:
        session = self._session
        layer = self._layers.get(f"{'Reference' if role == 'reference' else 'Moving'} crop")
        if session is None or layer is None:
            return
        crop = crop_from_drag(
            self._world_to_xy(role, w0),
            self._world_to_xy(role, w1),
            session.descriptor.images[role].shape_yx,
            min_px=1,
        )
        self._draw_crop(layer, crop, self._zoom())

    def _finish_crop_draw(self, role: str, w0: np.ndarray, w1: np.ndarray) -> None:
        session = self._session
        assert session is not None and session.state is not None
        shape = session.descriptor.images[role].shape_yx
        crop = crop_from_drag(self._world_to_xy(role, w0), self._world_to_xy(role, w1), shape)
        if crop is None:
            self._sync_crops()
            self._host.set_status(
                f"Crop too small: drag at least {CROP_MIN_DRAW_PX} px in the {role} image."
            )
            return
        self._set_crop(role, crop, undoable=True)
        self._disarm_crop()
        message = (
            f"{role.capitalize()} crop: x {crop.x0}–{crop.x1}, y {crop.y0}–{crop.y1} "
            f"({crop.width}×{crop.height} px)."
        )
        snapped = snapped_crop_size(crop, shape)
        if min(snapped) < CROP_MIN_SIZE_PX:
            message += (
                f" Warning: after snapping to {CROP_SNAP_PX} px it is {snapped[0]}×{snapped[1]} px; "
                f"the register workflow needs at least {CROP_MIN_SIZE_PX} px per side."
            )
        self._host.set_status(message)

    # ------------------------------------------------------------------ widget slots

    def _coarse_edit(self, change: Callable[[CoarseState], CoarseState]) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        if self._preview:
            self._host.set_status("Turn off Preview fit to change the coarse alignment.")
            if self._widget is not None:
                self._widget.set_coarse(state.coarse)
            return
        state.coarse = change(state.coarse)
        if self._widget is not None:
            self._widget.set_coarse(state.coarse)
        self._schedule_affine_update()
        self._content_changed()

    @_slot(prepared=True)
    def _on_rotate_by(self, delta_deg: float) -> None:
        self._coarse_edit(lambda c: c.rotated(float(delta_deg)))

    @_slot(prepared=True)
    def _on_rotation_set(self, rotation_deg: float) -> None:
        self._coarse_edit(lambda c: c.with_rotation(float(rotation_deg)))

    @_slot(prepared=True)
    def _on_shift_by(self, dx_um: float, dy_um: float) -> None:
        self._coarse_edit(lambda c: c.shifted(float(dx_um), float(dy_um)))

    @_slot(prepared=True)
    def _on_offset_set(self, x_um: float, y_um: float) -> None:
        self._coarse_edit(lambda c: c.with_offset(float(x_um), float(y_um)))

    @_slot(prepared=True)
    def _on_flip_horizontal(self) -> None:
        self._coarse_edit(lambda c: c.flipped_horizontal())

    @_slot(prepared=True)
    def _on_flip_vertical(self) -> None:
        self._coarse_edit(lambda c: c.flipped_vertical())

    @_slot(prepared=True)
    def _on_reset_coarse(self) -> None:
        self._coarse_edit(lambda c: CoarseState())

    @_slot(prepared=True)
    def _on_adopt_fit(self) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        fit = self._live_fit
        if fit is None or fit.matrix is None or len(state.enabled_pairs()) < PREVIEW_MIN_ENABLED:
            self._host.set_status("Add at least 3 enabled pairs first.")
            return
        images = self._session.descriptor.images
        coarse, ratio = coarse_from_matrix(
            fit.matrix,
            moving_shape_yx=images["moving"].shape_yx,
            reference_shape_yx=images["reference"].shape_yx,
            moving_pixel_size_um=images["moving"].pixel_size_um,
            reference_pixel_size_um=images["reference"].pixel_size_um,
        )
        state.coarse = coarse
        self._preview = False
        if self._widget is not None:
            self._widget.set_preview(False)
            self._widget.set_coarse_enabled(True)
            self._widget.set_coarse(coarse)
        self._schedule_affine_update()
        self._content_changed()
        self._host.set_status(
            f"Coarse alignment set from the fit (scale ratio {ratio:.4f}; "
            "scale is not part of the coarse transform)."
        )

    @_slot()
    def _on_display_mode_changed(self, mode: str) -> None:
        if self._session is None:
            self._host.set_status(_NO_SESSION_TEXT)
            return
        self._apply_display(str(mode))

    @_slot()
    def _on_flick_requested(self) -> None:
        if self._session is None:
            self._host.set_status(_NO_SESSION_TEXT)
            return
        self._apply_display(FLICK_NEXT[self._display_mode])

    @_slot()
    def _on_auto_switch_changed(self, value: bool) -> None:
        self._auto_switch = bool(value)

    @_slot(prepared=True)
    def _on_landmark_mode_changed(self, active: bool) -> None:
        self._set_landmark_mode(bool(active))

    @_slot()
    def _on_pair_selected(self, pair_id: int) -> None:
        session = self._session
        if session is None:
            self._host.set_status(_NO_SESSION_TEXT)
            return
        if session.kind == "prepared":
            assert session.state is not None
            try:
                pair = session.state.pair(int(pair_id))
            except KeyError:
                return
            role = "moving" if self._display_mode == "moving" else "reference"
            x, y = pair.moving_xy if role == "moving" else pair.reference_xy
            layer = self._layers[f"{'Moving' if role == 'moving' else 'Reference'} landmarks"]
        else:
            records = {int(r["id"]): r for r in (session.result or {}).get("landmarks") or []}
            record = records.get(int(pair_id))
            if record is None or record.get("reference_xy_px") is None:
                return
            x, y = record["reference_xy_px"]
            layer = self._layers["Reference landmarks"]
        world = np.asarray(layer.data_to_world((float(y), float(x))), dtype=float).ravel()[-2:]
        self._viewer.camera.center = (0.0, float(world[0]), float(world[1]))

    @_slot(prepared=True)
    def _on_pair_enabled_changed(self, pair_id: int, enabled: bool) -> None:
        assert self._session is not None and self._session.state is not None
        state = self._session.state
        try:
            current = state.pair(int(pair_id))
        except KeyError:
            return
        if current.enabled == bool(enabled):
            return
        self._push_undo(UndoAction("enable", pair_id=int(pair_id), enabled_before=current.enabled))
        state.pairs[state.pairs.index(current)] = dataclasses.replace(
            current, enabled=bool(enabled)
        )
        self._content_changed()

    @_slot(prepared=True)
    def _on_delete_pairs(self, ids: Sequence[int]) -> None:
        assert self._session is not None and self._session.state is not None
        removed = self._session.state.remove_pairs(int(i) for i in ids)
        if not removed:
            return
        self._push_undo(UndoAction("delete", pairs=tuple(removed)))
        self._content_changed()

    @_slot(prepared=True)
    def _on_undo_requested(self) -> None:
        self._undo_or_cancel()

    @_slot(prepared=True)
    def _on_preview_changed(self, on: bool) -> None:
        assert self._session is not None and self._session.state is not None
        n_enabled = len(self._session.state.enabled_pairs())
        if on and n_enabled < PREVIEW_MIN_ENABLED:
            if self._widget is not None:
                self._widget.set_preview(False)
            self._host.set_status("Preview needs at least 3 enabled pairs.")
            return
        self._preview = bool(on)
        if self._widget is not None:
            self._widget.set_coarse_enabled(not on)
        self._schedule_affine_update()

    @_slot(prepared=True)
    def _on_outlier_floor_changed(self, value: float) -> None:
        assert self._session is not None and self._session.state is not None
        self._floor_um = float(value)
        self._session.state.outlier_floor_um = self._floor_um
        self._refit()
        self._sync_markers()
        self._push_widget_rows()

    @_slot(prepared=True)
    def _on_crop_draw_requested(self, role: str) -> None:
        if role not in ROLES:
            return
        self._arm_crop(str(role))

    @_slot(prepared=True)
    def _on_crop_set(self, role: str, crop: CropRect | None) -> None:
        session = self._session
        assert session is not None and session.state is not None
        if role not in ROLES:
            return
        height, width = session.descriptor.images[role].shape_yx
        valid = crop is None or (
            0 <= crop.x0 < crop.x1 <= width
            and 0 <= crop.y0 < crop.y1 <= height
            and crop.width >= CROP_MIN_DRAW_PX
            and crop.height >= CROP_MIN_DRAW_PX
        )
        current = session.state.crops.get(role)
        if not valid:
            self._host.set_status(
                f"Crop must lie inside the {role} image ({width}×{height} px) and be at least "
                f"{CROP_MIN_DRAW_PX} px wide and high."
            )
            self._push_widget_rows()  # revert the spin boxes
            return
        if crop != current:
            self._set_crop(role, crop, undoable=True)

    @_slot()
    def _on_overlay_changed(self, key: str, visible: bool) -> None:
        session = self._session
        if session is None or session.kind != "result":
            return
        name = _OVERLAY_LAYERS.get(key)
        layer = self._layers.get(name) if name else None
        if layer is not None:
            layer.visible = bool(visible)

    def _on_overlay_layer_visible(self, event: Any) -> None:
        layer = getattr(event, "source", None)
        if layer is None or self._widget is None:
            return
        for key, name in _OVERLAY_LAYERS.items():
            if layer.name == name:
                try:
                    self._widget.set_overlay_visibility(key, bool(layer.visible))
                except Exception:  # noqa: BLE001
                    log.exception("manual registration: could not mirror overlay visibility")

    # ------------------------------------------------------------------ saving

    @_slot(prepared=True)
    def _on_save_requested(self) -> None:
        self._save("draft")

    @_slot(prepared=True)
    def _on_finish_requested(self) -> None:
        self._save("finished")

    def _build_payload(self, status: str) -> dict[str, Any]:
        assert self._session is not None and self._session.state is not None
        return labelling_to_attrs(
            self._session.state,
            self._session.descriptor,
            status=status,
            software_version=_software_version(),
            outlier_floor_um=self._floor_um,
        )

    def _save(self, status: str) -> None:
        session = self._session
        assert session is not None and session.state is not None
        if self._saving:
            self._host.set_status("A save is already running.")
            return
        if status == "finished":
            ok, reason = finish_allowed(
                len(session.state.enabled_pairs()), self._pending is not None, False
            )
            if not ok:
                self._host.set_status(reason)
                return
        payload = self._build_payload(status)
        expected = session.state.revision
        path = session.path
        serial = self._session_serial
        version = _software_version()
        self._saving = True
        if self._widget is not None:
            self._widget.set_busy(True)
            self._widget.set_finish_state(False, "Saving…")
        self._host.run_worker(
            lambda: save_labelling(
                path, payload, expected_revision=expected, software_version=version
            ),
            lambda written: self._on_saved(serial, written),
            lambda exc: self._on_save_error(serial, exc),
            key="registration_save",
        )

    def _session_is_current(self, serial: int) -> bool:
        return self._session is not None and serial == self._session_serial

    def _on_saved(self, serial: int, written: dict[str, Any]) -> None:
        if not self._session_is_current(serial):
            log.debug("manual registration: ignored the result of a save of an earlier session")
            return
        self._saving = False
        if self._widget is not None:
            self._widget.set_busy(False)
        try:
            self._record_saved(written)
        except Exception as exc:  # noqa: BLE001
            log.exception("manual registration: bookkeeping after a save failed")
            self._host.set_status(f"Manual registration error: {exc}")

    def _record_saved(self, written: dict[str, Any]) -> None:
        session = self._session
        assert session is not None and session.state is not None
        state = session.state
        state.revision = int(written["revision"])
        state.status = str(written["status"])
        state.loaded_content_sha = str(written["content_sha256"])
        clear_local_draft(session.path, self._host.app_data_dir())
        self._update_dirty()
        n_pairs = len(state.pairs)
        if written["status"] == "finished":
            self._host.set_status(
                f"Labelling finished: {len(state.enabled_pairs())} enabled pairs saved "
                f"(revision {state.revision}). Run manual_registration_register.nf next."
            )
        else:
            self._host.set_status(f"Saved {n_pairs} pairs (revision {state.revision}, draft).")

    def _on_save_error(self, serial: int, exc: BaseException) -> None:
        if not self._session_is_current(serial):
            log.warning("manual registration: a save of an earlier session failed: %s", exc)
            return
        self._saving = False
        if self._widget is not None:
            self._widget.set_busy(False)
        self._handle_save_error(exc)
        self._update_dirty()

    def _handle_save_error(self, exc: BaseException) -> None:
        session = self._session
        assert session is not None
        name = session.path.name
        if isinstance(exc, StoreChangedError):
            choice = self._ask(
                "Store changed",
                f"The landmarks in {name} were changed by another session since you loaded them "
                f"({exc}). Reload the store (discards your changes here) or cancel and keep editing?",
                ["Reload", "Cancel"],
                default="Cancel",
            )
            if choice == "Reload":
                clear_local_draft(session.path, self._host.app_data_dir())
                if session.state is not None:
                    session.state.loaded_content_sha = self._content_sha()  # clean: no prompt
                self.load_store(session.path)
            else:
                self._host.set_status("Not saved: the store changed on disk.")
        elif isinstance(exc, StoreBusyError):
            self._warn(
                "Store busy",
                "The store is busy (another program is writing to it). Try again in a moment.",
            )
        else:
            log.error("manual registration: saving the landmarks failed", exc_info=exc)
            self._warn("Could not save landmarks", str(exc))

    def _save_sync(self, status: str) -> bool:
        """Save on the GUI thread (a single small attrs write); ``False`` after an error dialog."""
        session = self._session
        assert session is not None and session.state is not None
        if self._saving:
            self._host.set_status("A save is already running.")
            return False
        payload = self._build_payload(status)
        try:
            written = save_labelling(
                session.path,
                payload,
                expected_revision=session.state.revision,
                software_version=_software_version(),
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_save_error(exc)
            return False
        self._record_saved(written)
        return True

    # ------------------------------------------------------------------ drafts

    def _write_draft(self) -> None:
        session = self._session
        if session is None or session.kind != "prepared" or session.state is None:
            return
        try:
            app_dir = self._host.app_data_dir()
            if not self.has_unsaved_changes():
                clear_local_draft(session.path, app_dir)
                return
            payload = labelling_to_attrs(
                session.state,
                session.descriptor,
                status="draft",
                software_version=_software_version(),
                outlier_floor_um=self._floor_um,
            )
            payload["updated_at"] = utc_now()
            payload["revision"] = session.state.revision or 1  # placeholder, see _decide_draft
            write_local_draft(session.path, app_dir, payload)
        except Exception:  # noqa: BLE001
            log.warning("manual registration: could not write the local draft", exc_info=True)

    # ------------------------------------------------------------------ dialogs

    def _ask(self, title: str, text: str, buttons: Sequence[str], *, default: str) -> str:
        """Ask the user to pick one of ``buttons``; returns ``default`` when the box is closed."""
        from qtpy.QtWidgets import QMessageBox

        box = QMessageBox(self._widget)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Question)
        created = {label: box.addButton(label, QMessageBox.ActionRole) for label in buttons}
        box.setDefaultButton(created[default])
        box.setEscapeButton(created[default])
        run = getattr(box, "exec", None) or box.exec_
        run()
        clicked = box.clickedButton()
        for label, button in created.items():
            if button is clicked:
                return label
        return default

    def _warn(self, title: str, text: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.warning(self._widget, title, text)

    # ------------------------------------------------------------------ leaving

    def confirm_discard_unsaved(self) -> bool:
        """Return ``True`` if it is fine to leave the session (asking about unsaved changes).

        "Save draft" saves synchronously on the GUI thread, "Discard" keeps the local draft that
        is offered at the next load, "Cancel" returns ``False``.
        """
        if not self.has_unsaved_changes():
            return True
        assert self._session is not None
        choice = self._ask(
            "Unsaved landmarks",
            f"You have unsaved landmark changes in {self._session.path.name}. Save a draft first? "
            "(Discard keeps a local draft that is offered next time.)",
            ["Save draft", "Discard", "Cancel"],
            default="Cancel",
        )
        if choice == "Save draft":
            return self._save_sync("draft")
        if choice == "Discard":
            self.flush_pending_updates()
            self._write_draft()
            return True
        return False

    def cancel_background(self) -> None:
        """Cancel a running load; its late callbacks are ignored (saves are not cancelled)."""
        self._load_cancel.set()
        self._load_generation += 1

    def deactivate(self) -> None:
        """End the session (idempotent): flush the draft, remove layers, restore keys."""
        if self._session is None:
            return
        self.flush_pending_updates()
        self._write_draft()  # up to date even when the 500 ms timer had not run
        for name in ("_affine_timer", "_draft_timer"):
            timer = getattr(self, name)
            if timer is not None:
                timer.stop()
        self._teardown_session(call_host=True)

    def _teardown_session(self, *, call_host: bool) -> None:
        viewer = self._viewer
        for name in ("_affine_timer", "_draft_timer"):
            timer = getattr(self, name)
            if timer is not None:
                timer.stop()
        try:
            viewer.camera.events.zoom.disconnect(self._on_zoom)
        except (TypeError, ValueError, RuntimeError):
            pass
        try:
            viewer.layers.selection.events.active.disconnect(self._keep_markers_inactive)
        except (TypeError, ValueError, RuntimeError):
            pass
        for layer in list(self._layers.values()):
            for emitter, callback in (
                (getattr(layer.events, "mode", None), self._on_layer_mode),
                (getattr(layer.events, "opacity", None), self._on_layer_opacity),
                (getattr(layer.events, "visible", None), self._on_overlay_layer_visible),
            ):
                if emitter is None:
                    continue
                try:
                    emitter.disconnect(callback)
                except (TypeError, ValueError, RuntimeError):
                    pass
        self._restore_session_keys()
        self._landmark_mode = False
        self._crop_role = None
        self._update_double_click()
        self._dblclick_removed = []
        for layer in list(self._layers.values()):
            if layer in viewer.layers:
                try:
                    viewer.layers.remove(layer)
                except Exception:  # noqa: BLE001
                    log.exception("manual registration: could not remove layer %s", layer.name)
        self._layers = {}
        self._image_layers = {"reference": [], "moving": []}
        self._reg_layer = {}
        self._layer_role = {}
        self._user_opacity = {}
        self._size_factors = {}
        self._session = None
        self._pending = None
        self._undo = []
        self._preview = False
        self._crop_restore_mode = None
        self._live_fit = None
        self._display_mode = "both"
        self._saving = False
        if self._widget is not None:
            self._widget.set_busy(False)
            self._widget.set_session(None, "")
        if call_host:
            self._host.leave_registration_session()
