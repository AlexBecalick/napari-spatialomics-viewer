"""Pure model of the manual-registration feature: constants, state, maths and attrs contract.

This module holds everything about landmark-based manual registration that has no side effects:
the coordinate conventions, the coarse transform, the napari affine placement, the Umeyama
similarity fit with leave-one-out residuals and outlier trimming, the (de)serialisation of the
labelling payload with its digests, and the parser of the store descriptor.

It must not import Qt, zarr or napari (only numpy and the standard library), and it must not
import ``merxen``. The maths is a re-implementation of MerXen's ``geometry.py`` and
``landmark_fit.py``; the golden vectors in ``tests/data/manual_registration_golden_v1.json`` tie
both copies together.

Conventions (identical to MerXen):

* A point is ``(x, y)`` (x = column, y = row) in level-0 pixels of the full, uncropped image it
  belongs to; an integer value is a pixel centre.
* Every matrix is 3x3 with last row ``[0, 0, 1]`` and maps moving to reference:
  ``p_ref = M @ p_mov``.
* ``R(theta) = [[cos, sin], [-sin, cos]]``; a positive angle is counter-clockwise on screen.
* ``M_coarse = T(c_r + offset / p_r) R(theta) F(mirror) S(p_m / p_r) T(-c_m)``.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import random
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
STORE_ATTR = "merxen_manual_registration_store"
LABELLING_ATTR = "merxen_manual_registration"
RESULT_ATTR = "merxen_manual_registration_result"
STORE_SCHEMA = "merxen.manual_registration.store"
LABELLING_SCHEMA = "merxen.manual_registration.labelling"
RESULT_SCHEMA = "merxen.manual_registration.result"
LAYER_METADATA_KEY = "merxen_manual_registration"  # viewer only (layer.metadata key)
MIN_LANDMARK_PAIRS = 8
DEFAULT_OUTLIER_FLOOR_UM = 10.0
OUTLIER_K = 3.0
MAX_OUTLIER_FRACTION = 0.25
MIN_USED_PAIRS = 6
CROP_SNAP_PX = 64
CROP_MIN_SIZE_PX = 1024
MIN_PREVIEW_PAIRS = 3  # preview / "Use fit" need at least 3 enabled pairs
MIN_DRAWN_CROP_PX = 16  # smallest crop side the UI accepts
SOFTWARE_NAME = "napari-compare-xenium-merscope"  # written_by.software
ROLES = ("reference", "moving")
DISPLAY_MODES = ("reference", "moving", "both")
SERIAL_SECTION_FLOOR_HINT_UM = 30.0  # outlier floor of the serial_section preset
SAME_SECTION_FLOOR_UM = 10.0  # outlier floor of the same_section preset

# Converts between (x, y) and napari (row, col). Never mutate.
SWAP: np.ndarray = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
SWAP.setflags(write=False)

_COLOUR_RE = re.compile(r"^#[0-9a-f]{6}$")


# ----------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------


class NotARegistrationStore(ValueError):
    """The path is not a (complete) manual-registration store."""


class UnsupportedSchemaVersion(ValueError):
    """A payload was written with a newer schema version than this viewer reads."""


class LabellingMismatch(ValueError):
    """The stored landmarks belong to different images than the store holds."""


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------


def _round(value: float, ndigits: int) -> float:
    """Round to ``ndigits`` decimals; ``-0.0`` becomes ``0.0`` (stable JSON and digests)."""
    return round(float(value), ndigits) + 0.0


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to (-180, 180] degrees.

    Args:
        angle_deg: Angle in degrees.

    Returns:
        The equivalent angle in ``(-180, 180]``.

    Raises:
        ValueError: If the angle is not finite.
    """
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle_deg!r}")
    return angle - 360.0 * math.ceil((angle - 180.0) / 360.0)


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _float_pair(value: object, what: str) -> tuple[float, float]:
    """Parse two finite numbers or raise ``ValueError`` naming ``what``."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence | np.ndarray)
        or len(value) != 2
        or not all(_is_number(v) for v in value)
    ):
        raise ValueError(f"{what} must be two numbers, got {value!r}")
    pair = (float(value[0]), float(value[1]))
    if not (math.isfinite(pair[0]) and math.isfinite(pair[1])):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return pair


# ----------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CoarseState:
    """Coarse (rotation, mirror, offset) placement of the moving image.

    Attributes:
        rotation_deg: Rotation about the displayed moving centre, wrapped to ``(-180, 180]``.
        mirror_x: Whether the moving image is mirrored (a ``det < 0`` matrix).
        offset_um: Displayed moving centre relative to the reference centre, in reference
            micrometres, x right and y down.
    """

    rotation_deg: float = 0.0
    mirror_x: bool = False
    offset_um: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_deg", wrap_deg(self.rotation_deg))
        object.__setattr__(self, "mirror_x", bool(self.mirror_x))
        offset = (float(self.offset_um[0]), float(self.offset_um[1]))
        object.__setattr__(self, "offset_um", offset)

    def rotated(self, delta_deg: float) -> CoarseState:
        """Return the state rotated by ``delta_deg`` about the displayed moving centre."""
        return replace(self, rotation_deg=wrap_deg(self.rotation_deg + delta_deg))

    def with_rotation(self, rotation_deg: float) -> CoarseState:
        """Return the state with the rotation set to ``rotation_deg``."""
        return replace(self, rotation_deg=wrap_deg(rotation_deg))

    def shifted(self, dx_um: float, dy_um: float) -> CoarseState:
        """Return the state shifted by ``(dx_um, dy_um)`` reference micrometres."""
        return replace(self, offset_um=(self.offset_um[0] + dx_um, self.offset_um[1] + dy_um))

    def with_offset(self, x_um: float, y_um: float) -> CoarseState:
        """Return the state with the offset set to ``(x_um, y_um)``."""
        return replace(self, offset_um=(float(x_um), float(y_um)))

    def flipped_horizontal(self) -> CoarseState:
        """Mirror the displayed moving image about the vertical axis through its centre."""
        return replace(self, mirror_x=not self.mirror_x, rotation_deg=wrap_deg(-self.rotation_deg))

    def flipped_vertical(self) -> CoarseState:
        """Mirror the displayed moving image about the horizontal axis through its centre."""
        return replace(
            self, mirror_x=not self.mirror_x, rotation_deg=wrap_deg(180.0 - self.rotation_deg)
        )

    def to_attrs(self) -> dict[str, Any]:
        """Return the JSON form written to the store (rotation and offset rounded to 4)."""
        rotation = _round(self.rotation_deg, 4)
        if rotation <= -180.0:
            rotation = 180.0
        return {
            "rotation_deg": rotation,
            "mirror_x": bool(self.mirror_x),
            "offset_um": [_round(self.offset_um[0], 4), _round(self.offset_um[1], 4)],
        }


@dataclass(frozen=True)
class CropRect:
    """An integer crop, half-open in the full level-0 pixel frame: ``arr[y0:y1, x0:x1]``."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(_is_int(v) for v in values):
            raise ValueError(f"crop values must be integers, got {values!r}")
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if not (0 <= self.x0 < self.x1 and 0 <= self.y0 < self.y1):
            raise ValueError(
                f"invalid crop (x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1}): "
                "need 0 <= x0 < x1 and 0 <= y0 < y1"
            )

    @property
    def width(self) -> int:
        """Crop width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Crop height in pixels."""
        return self.y1 - self.y0

    def contains_xy(self, points_xy: np.ndarray) -> np.ndarray:
        """Return an ``(N,)`` bool mask of points inside the crop.

        A point is inside when ``x0 - 0.5 <= x < x1 - 0.5`` and ``y0 - 0.5 <= y < y1 - 0.5``.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        return np.asarray(
            (x >= self.x0 - 0.5) & (x < self.x1 - 0.5) & (y >= self.y0 - 0.5) & (y < self.y1 - 0.5),
            dtype=bool,
        )

    def validate_within(self, shape_yx: tuple[int, int]) -> None:
        """Raise ``ValueError`` unless the crop lies inside an image of shape ``(H, W)``."""
        height, width = shape_yx
        if self.x1 > width or self.y1 > height:
            raise ValueError(
                f"crop x {self.x0}:{self.x1}, y {self.y0}:{self.y1} exceeds the image "
                f"({width} x {height} px)"
            )

    def to_attrs(self) -> dict[str, int]:
        """Return the JSON form ``{"x0", "y0", "x1", "y1"}``."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_attrs(cls, value: Mapping[str, Any] | None) -> CropRect | None:
        """Parse the JSON form; ``None`` gives ``None``.

        Raises:
            ValueError: If ``value`` is not a mapping with four valid integer keys.
        """
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"x0", "y0", "x1", "y1"}:
            raise ValueError(f"invalid crop {value!r}")
        return cls(x0=value["x0"], y0=value["y0"], x1=value["x1"], y1=value["y1"])


@dataclass(frozen=True)
class LandmarkPair:
    """One landmark pair; points are ``(x, y)`` level-0 pixels of the full images."""

    id: int
    colour: str
    reference_xy: tuple[float, float]
    moving_xy: tuple[float, float]
    enabled: bool = True

    def to_attrs(self) -> dict[str, Any]:
        """Return the JSON form with exactly the five payload keys (coordinates rounded to 3)."""
        return {
            "id": int(self.id),
            "colour": str(self.colour),
            "enabled": bool(self.enabled),
            "reference_xy_px": [_round(self.reference_xy[0], 3), _round(self.reference_xy[1], 3)],
            "moving_xy_px": [_round(self.moving_xy[0], 3), _round(self.moving_xy[1], 3)],
        }


@dataclass(frozen=True)
class RoleImage:
    """One image of the store (reference or moving) as the viewer places it."""

    role: str  # "reference" | "moving"
    element_key: str  # SpatialData images element name
    channel_names: tuple[str, ...]
    registration_channel: str
    shape_yx: tuple[int, int]  # shape of the element's level 0
    pixel_size_um: float  # level-0 um/px of the element
    base_affine_rc: np.ndarray = field(compare=False)  # 3x3 (row, col) -> world um


@dataclass(frozen=True)
class StoreDescriptor:
    """What the viewer needs to know about a manual-registration store."""

    path: Path
    kind: str  # "prepared" | "result"
    dataset_name: str
    images: dict[str, RoleImage]  # keys "reference", "moving"
    grid_digest: str | None
    store_attr: dict[str, Any]
    labelling: dict[str, Any] | None  # LABELLING_ATTR verbatim (prepared only)
    result: dict[str, Any] | None  # RESULT_ATTR verbatim (result only)
    warnings: tuple[str, ...] = ()


def pair_colour(pair_id: int, seed: int) -> str:
    """Return the deterministic ``#rrggbb`` colour of a pair (golden-ratio hue walk)."""
    h0 = ((seed * 2654435761) % 2**32) / 2**32
    hue = (h0 + 0.6180339887498949 * pair_id) % 1.0
    sat = (0.75, 1.0, 0.9)[pair_id % 3]
    r, g, b = colorsys.hsv_to_rgb(hue, sat, 1.0)
    return f"#{round(255 * r):02x}{round(255 * g):02x}{round(255 * b):02x}"


def _default_crops() -> dict[str, CropRect | None]:
    return {"reference": None, "moving": None}


@dataclass
class LabellingState:
    """Mutable labelling state: coarse placement, crops and landmark pairs."""

    coarse: CoarseState = field(default_factory=CoarseState)
    crops: dict[str, CropRect | None] = field(default_factory=_default_crops)
    pairs: list[LandmarkPair] = field(default_factory=list)  # kept sorted by id
    next_id: int = 1
    colour_seed: int = 0
    status: str = "new"  # "new" (never saved) | "draft" | "finished"
    revision: int | None = None  # on-disk revision this state is based on
    outlier_floor_um: float | None = None  # informative, from the stored labelling
    loaded_content_sha: str | None = None
    load_warnings: tuple[str, ...] = ()

    @classmethod
    def new(cls, colour_seed: int | None = None) -> LabellingState:
        """Return an empty state; ``colour_seed=None`` draws a random seed."""
        if colour_seed is None:
            colour_seed = random.SystemRandom().randrange(1, 2**31)
        return cls(colour_seed=int(colour_seed))

    def enabled_pairs(self) -> list[LandmarkPair]:
        """Return the enabled pairs in ascending id order."""
        return [p for p in self.pairs if p.enabled]

    def pair(self, pair_id: int) -> LandmarkPair:
        """Return the pair with ``pair_id``.

        Raises:
            KeyError: If there is no such pair.
        """
        for p in self.pairs:
            if p.id == pair_id:
                return p
        raise KeyError(pair_id)

    def add_pair(self, reference_xy: Sequence[float], moving_xy: Sequence[float]) -> LandmarkPair:
        """Add a pair with the next id and its colour; coordinates are rounded to 3 decimals."""
        pair = LandmarkPair(
            id=self.next_id,
            colour=pair_colour(self.next_id, self.colour_seed),
            reference_xy=(_round(reference_xy[0], 3), _round(reference_xy[1], 3)),
            moving_xy=(_round(moving_xy[0], 3), _round(moving_xy[1], 3)),
            enabled=True,
        )
        self.pairs.append(pair)
        self.pairs.sort(key=lambda p: p.id)
        self.next_id += 1
        return pair

    def remove_pairs(self, ids: Iterable[int]) -> list[LandmarkPair]:
        """Remove the pairs with the given ids and return them in id order.

        Ids that match no pair are ignored. ``next_id`` is unchanged (ids are never reused).
        """
        wanted = set(ids)
        removed = [p for p in self.pairs if p.id in wanted]
        self.pairs = [p for p in self.pairs if p.id not in wanted]
        return removed

    def restore_pairs(self, pairs: Iterable[LandmarkPair]) -> None:
        """Re-insert removed pairs (same ids and colours), keeping the list sorted.

        Raises:
            ValueError: If an id is already present (nothing is restored then).
        """
        incoming = list(pairs)
        present = {p.id for p in self.pairs}
        seen: set[int] = set()
        for p in incoming:
            if p.id in present or p.id in seen:
                raise ValueError(f"landmark id {p.id} is already present")
            seen.add(p.id)
        self.pairs.extend(incoming)
        self.pairs.sort(key=lambda p: p.id)
        if incoming:
            self.next_id = max(self.next_id, max(p.id for p in incoming) + 1)

    def set_enabled(self, pair_id: int, enabled: bool) -> None:
        """Enable or disable a pair (KeyError if absent)."""
        for index, p in enumerate(self.pairs):
            if p.id == pair_id:
                self.pairs[index] = replace(p, enabled=bool(enabled))
                return
        raise KeyError(pair_id)

    def content_sha256(self) -> str:
        """Digest of the content this state would write (coarse, crops and pairs)."""
        return canonical_sha256(_content_dict(self))


@dataclass(frozen=True)
class ResidualRow:
    """Per-pair diagnostics of the live fit (or of a register result)."""

    id: int
    colour: str
    enabled: bool
    used: bool
    outlier: bool
    outside_crop: bool
    fit_um: float  # NaN when unavailable
    loo_um: float  # NaN when unavailable
    affine_um: float | None = None  # result mode only
    final_um: float | None = None  # result mode only


@dataclass(frozen=True)
class FitSummary:
    """Summary of the live fit shown under the landmark table."""

    n_enabled: int
    n_used: int
    scale: float
    expected_scale: float  # p_m / p_r
    scale_ratio: float
    rotation_deg: float
    reflected: bool
    rms_um: float  # sqrt(mean(residuals_um[used] ** 2))
    other_class_median_um: float
    chosen_class_median_um: float
    spread_ratio: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SimilarityFit:
    """A similarity fit of moving to reference landmarks."""

    matrix: np.ndarray  # 3x3 moving px -> reference px
    scale: float  # sqrt|det L|
    rotation_deg: float  # applied after the mirror, wrapped to (-180, 180]
    reflected: bool
    residuals_px: np.ndarray  # in-sample, reference px, (N,)


@dataclass(frozen=True)
class RobustFitOptions:
    """Options of ``robust_similarity``."""

    floor_um: float = DEFAULT_OUTLIER_FLOOR_UM
    k: float = OUTLIER_K
    max_outlier_fraction: float = MAX_OUTLIER_FRACTION
    min_used_pairs: int = MIN_USED_PAIRS
    trim_outliers: bool = True


@dataclass(frozen=True)
class RobustSimilarityFit:
    """A similarity fit after trimming outlier pairs, with diagnostics."""

    fit: SimilarityFit  # on the used pairs
    used: np.ndarray  # (N,) bool
    outlier: np.ndarray  # (N,) bool
    residuals_um: np.ndarray  # every pair under fit.matrix
    loo_um: np.ndarray  # used: LOO within the used set; others: = residuals_um
    chosen_class_median_um: float  # median in-sample residual of the used pairs
    other_class_median_um: float  # the same under a fit of the opposite reflection class
    spread_ratio: float  # of the used moving points
    scale_ratio: float  # fit.scale / (p_m / p_r)


@dataclass(frozen=True)
class LandmarkEvaluation:
    """Everything the controller needs after every change of the landmarks."""

    rows: tuple[ResidualRow, ...]  # all pairs, ascending id
    summary: FitSummary | None
    matrix: np.ndarray | None  # robust fit matrix (moving px -> reference px) or None
    error: str | None = None  # e.g. a degenerate-geometry message


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------


def rotation_matrix_xy(angle_deg: float) -> np.ndarray:
    """Return the 3x3 rotation ``R(theta)``; positive is counter-clockwise on screen."""
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def translation_xy(dx: float, dy: float) -> np.ndarray:
    """Return the 3x3 translation ``T(dx, dy)``."""
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = dx
    matrix[1, 2] = dy
    return matrix


def image_centre_xy(shape_yx: tuple[int, int]) -> tuple[float, float]:
    """Return the centre ``((W - 1) / 2, (H - 1) / 2)`` of an image of shape ``(H, W)``.

    Raises:
        ValueError: If the shape is not two positive integers.
    """
    _check_shape(shape_yx)
    height, width = shape_yx
    return ((width - 1) / 2.0, (height - 1) / 2.0)


def _check_shape(shape_yx: Sequence[int]) -> None:
    if len(shape_yx) != 2 or not all(_is_int(v) and v > 0 for v in shape_yx):
        raise ValueError(f"image shape must be two positive integers, got {tuple(shape_yx)!r}")


def _check_pixel_size(name: str, value: float) -> None:
    if not (isinstance(value, (int, float, np.floating, np.integer)) and value > 0):
        raise ValueError(f"{name} must be positive, got {value!r}")


def apply_matrix_xy(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to ``(x, y)`` points of shape ``(2,)`` or ``(N, 2)``."""
    mat = np.asarray(matrix, dtype=np.float64)
    pts = np.asarray(points_xy, dtype=np.float64)
    return np.asarray(pts @ mat[:2, :2].T + mat[:2, 2], dtype=np.float64)


def coarse_matrix_xy(
    coarse: CoarseState,
    *,
    moving_shape_yx: tuple[int, int],
    reference_shape_yx: tuple[int, int],
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
) -> np.ndarray:
    """Build the moving-to-reference matrix of a coarse state.

    ``M = T(c_r + offset / p_r) R(theta) F(mirror) S(p_m / p_r) T(-c_m)`` with ``c_m`` and
    ``c_r`` the centres of the full moving and reference images.

    Raises:
        ValueError: On non-positive pixel sizes or invalid shapes.
    """
    _check_pixel_size("moving_pixel_size_um", moving_pixel_size_um)
    _check_pixel_size("reference_pixel_size_um", reference_pixel_size_um)
    scale = moving_pixel_size_um / reference_pixel_size_um
    c_ref = image_centre_xy(reference_shape_yx)
    c_mov = image_centre_xy(moving_shape_yx)
    ox, oy = coarse.offset_um
    mirror = np.diag([-1.0 if coarse.mirror_x else 1.0, 1.0, 1.0])
    return np.asarray(
        translation_xy(
            c_ref[0] + ox / reference_pixel_size_um, c_ref[1] + oy / reference_pixel_size_um
        )
        @ rotation_matrix_xy(coarse.rotation_deg)
        @ mirror
        @ np.diag([scale, scale, 1.0])
        @ translation_xy(-c_mov[0], -c_mov[1]),
        dtype=np.float64,
    )


def _similarity_parameters(linear: np.ndarray) -> tuple[float, float, bool]:
    """Return ``(scale, rotation_deg, reflected)`` of a 2x2 similarity linear part."""
    det = float(np.linalg.det(linear))
    if det == 0.0 or not math.isfinite(det):
        raise ValueError("matrix is singular")
    reflected = det < 0.0
    scale = math.sqrt(abs(det))
    unit = linear / scale
    rotation = unit @ np.diag([-1.0, 1.0]) if reflected else unit
    angle = wrap_deg(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])))
    return scale, angle, reflected


def coarse_from_matrix(
    matrix_xy: np.ndarray,
    *,
    moving_shape_yx: tuple[int, int],
    reference_shape_yx: tuple[int, int],
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
) -> tuple[CoarseState, float]:
    """Recover the coarse state of a similarity matrix.

    Returns:
        ``(coarse, scale_ratio)`` where ``scale_ratio = k / (p_m / p_r)`` (informative: the
        scale is not part of the coarse state).

    Raises:
        ValueError: On non-positive pixel sizes, invalid shapes or a singular matrix.
    """
    _check_pixel_size("moving_pixel_size_um", moving_pixel_size_um)
    _check_pixel_size("reference_pixel_size_um", reference_pixel_size_um)
    mat = np.asarray(matrix_xy, dtype=np.float64)
    scale, angle, mirror = _similarity_parameters(mat[:2, :2])
    c_ref = image_centre_xy(reference_shape_yx)
    c_mov = image_centre_xy(moving_shape_yx)
    mapped = apply_matrix_xy(mat, np.asarray(c_mov))
    offset = (
        float((mapped[0] - c_ref[0]) * reference_pixel_size_um),
        float((mapped[1] - c_ref[1]) * reference_pixel_size_um),
    )
    coarse = CoarseState(rotation_deg=angle, mirror_x=mirror, offset_um=offset)
    return coarse, scale / (moving_pixel_size_um / reference_pixel_size_um)


def napari_affine_rc(reference_base_rc: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    """Return the napari ``(row, col)`` affine of a moving layer.

    ``A_mov = A_ref @ SWAP @ M @ SWAP`` where ``A_ref`` is the reference layer's base affine in
    napari ``(row, col)`` order (world unit: micrometres).
    """
    return np.asarray(
        np.asarray(reference_base_rc, dtype=np.float64)
        @ SWAP
        @ np.asarray(matrix_xy, dtype=np.float64)
        @ SWAP,
        dtype=np.float64,
    )


def snap_crop(
    crop: CropRect | None,
    shape_yx: tuple[int, int],
    *,
    multiple: int = CROP_SNAP_PX,
    min_size_px: int = CROP_MIN_SIZE_PX,
) -> CropRect:
    """Return the effective (workflow) crop: snapped inward to ``multiple`` pixels.

    The origin is rounded up and the far edge down, except an edge on the image border, which
    stays there. ``None`` gives the full image (no size check).

    Raises:
        ValueError: If the crop exceeds the image or a snapped side is shorter than
            ``min_size_px``.
    """
    if multiple < 1:
        raise ValueError(f"multiple must be at least 1, got {multiple}")
    height, width = shape_yx
    if crop is None:
        return CropRect(0, 0, width, height)
    crop.validate_within((height, width))
    x0 = -(-crop.x0 // multiple) * multiple
    y0 = -(-crop.y0 // multiple) * multiple
    x1 = width if crop.x1 == width else (crop.x1 // multiple) * multiple
    y1 = height if crop.y1 == height else (crop.y1 // multiple) * multiple
    w, h = x1 - x0, y1 - y0
    if w < min_size_px or h < min_size_px:
        raise ValueError(
            f"effective crop {max(w, 0)} x {max(h, 0)} px after snapping inward to {multiple} px "
            f"is smaller than {min_size_px} px per side"
        )
    return CropRect(x0, y0, x1, y1)


def points_outside_crop(points_xy: np.ndarray, crop: CropRect | None) -> np.ndarray:
    """Return an ``(N,)`` bool mask of points outside ``crop`` (``None``: none are outside)."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if crop is None:
        return np.zeros(len(pts), dtype=bool)
    return np.asarray(~crop.contains_xy(pts), dtype=bool)


def pairs_outside_crop(
    pairs: Sequence[LandmarkPair], crops: Mapping[str, CropRect | None]
) -> np.ndarray:
    """Return an ``(N,)`` bool mask: reference point outside its crop or moving outside its own."""
    if not pairs:
        return np.zeros(0, dtype=bool)
    ref = np.array([p.reference_xy for p in pairs], dtype=np.float64)
    mov = np.array([p.moving_xy for p in pairs], dtype=np.float64)
    return np.asarray(
        points_outside_crop(ref, crops.get("reference"))
        | points_outside_crop(mov, crops.get("moving")),
        dtype=bool,
    )


def point_in_image(point_xy: Sequence[float], shape_yx: tuple[int, int]) -> bool:
    """Whether a point lies in the valid range ``-0.5 <= x < W - 0.5``, ``-0.5 <= y < H - 0.5``."""
    height, width = shape_yx
    x, y = float(point_xy[0]), float(point_xy[1])
    return -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5


# ----------------------------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------------------------


def _as_pairs(moving_xy: Any, reference_xy: Any) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(moving_xy, dtype=np.float64)
    dst = np.asarray(reference_xy, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise ValueError("moving_xy and reference_xy must be (N, 2) arrays of the same length")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("landmark coordinates must be finite")
    return src, dst


def umeyama_similarity(moving_xy: Any, reference_xy: Any, *, reflection: bool) -> np.ndarray:
    """Least-squares similarity inside one reflection class (Umeyama 1991).

    Args:
        moving_xy: ``(N, 2)`` moving points in px (source).
        reference_xy: ``(N, 2)`` reference points in px (destination).
        reflection: Fit a reflecting (``det < 0``) similarity instead of a proper one.

    Returns:
        The 3x3 matrix mapping moving to reference.

    Raises:
        ValueError: On malformed input, fewer than 2 pairs or coincident moving points.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    n = len(src)
    if n < 2:
        raise ValueError("at least 2 landmark pairs are needed for a fit")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    sigma = dc.T @ sc / n
    u, d, vt = np.linalg.svd(sigma)
    sign = float(np.sign(np.linalg.det(u) * np.linalg.det(vt)))
    if sign == 0.0:
        sign = 1.0
    s_mat = np.diag([1.0, -sign if reflection else sign])
    rot = u @ s_mat @ vt
    var_s = float((sc**2).sum() / n)
    if var_s == 0.0:
        raise ValueError("landmark positions are degenerate (all moving points coincide)")
    c = float(np.trace(np.diag(d) @ s_mat) / var_s)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = c * rot
    matrix[:2, 2] = mu_d - c * rot @ mu_s
    return matrix


def fit_similarity(moving_xy: Any, reference_xy: Any, *, reflection: bool) -> SimilarityFit:
    """Fit a similarity and report its in-sample residuals (reference px).

    Raises:
        ValueError: With fewer than 2 pairs, malformed input or degenerate geometry.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    if len(src) < 2:
        raise ValueError("at least 2 landmark pairs are needed for a fit")
    matrix = umeyama_similarity(src, dst, reflection=reflection)
    scale, angle, reflected = _similarity_parameters(matrix[:2, :2])
    residuals = np.linalg.norm(apply_matrix_xy(matrix, src) - dst, axis=1)
    return SimilarityFit(
        matrix=matrix,
        scale=scale,
        rotation_deg=angle,
        reflected=reflected,
        residuals_px=residuals,
    )


def leave_one_out_residuals(moving_xy: Any, reference_xy: Any, *, reflection: bool) -> np.ndarray:
    """Residual (reference px) of each pair under a fit that excludes it.

    All NaN when there are fewer than 3 pairs; a degenerate sub-fit gives NaN for that pair.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    n = len(src)
    result = np.full(n, np.nan)
    if n < 3:
        return result
    for i in range(n):
        keep = np.arange(n) != i
        try:
            m = umeyama_similarity(src[keep], dst[keep], reflection=reflection)
        except ValueError:
            continue
        result[i] = float(np.linalg.norm(apply_matrix_xy(m, src[i]) - dst[i]))
    return result


def spread_ratio(points_xy: Any) -> float:
    """Return ``sqrt(lambda_min / lambda_max)`` of the points' covariance (1 round, 0 collinear)."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    w = np.linalg.eigvalsh(np.cov(pts.T, bias=True))
    if w[-1] == 0:
        return 0.0
    return float(math.sqrt(max(float(w[0]), 0.0) / float(w[-1])))


def robust_similarity(
    moving_xy: Any,
    reference_xy: Any,
    *,
    reflection: bool,
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
    options: RobustFitOptions = RobustFitOptions(),  # noqa: B008 (frozen dataclass)
) -> RobustSimilarityFit:
    """Fit a similarity after iteratively trimming outlier pairs.

    A pair is flagged when its leave-one-out residual (within the pairs still used) exceeds both
    ``floor_um`` and ``k`` times the median of the finite residuals. The worst flagged pair is
    dropped, one at a time, until none is flagged, ``floor(max_outlier_fraction * N)`` were
    dropped, or dropping one more would leave fewer than ``min_used_pairs``. Deterministic
    (ties: lowest index).

    Raises:
        ValueError: On malformed input, fewer than 2 pairs or non-positive pixel sizes.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    n = len(src)
    if n < 2:
        raise ValueError("at least 2 landmark pairs are needed for a fit")
    _check_pixel_size("moving_pixel_size_um", moving_pixel_size_um)
    _check_pixel_size("reference_pixel_size_um", reference_pixel_size_um)
    p_r, p_m = float(reference_pixel_size_um), float(moving_pixel_size_um)
    used = np.ones(n, dtype=bool)
    if options.trim_outliers:
        trimmed = 0
        max_trim = math.floor(options.max_outlier_fraction * n)
        while True:
            if int(used.sum()) < 3:
                break
            idx = np.flatnonzero(used)
            loo = leave_one_out_residuals(src[idx], dst[idx], reflection=reflection) * p_r
            finite = np.isfinite(loo)
            if not finite.any():
                break
            med = float(np.median(loo[finite]))
            flagged = finite & (loo > options.floor_um) & (loo > options.k * med)
            if (
                not flagged.any()
                or trimmed >= max_trim
                or int(used.sum()) - 1 < options.min_used_pairs
            ):
                break
            j = idx[int(np.argmax(np.where(flagged, loo, -np.inf)))]
            used[j] = False
            trimmed += 1
        outlier = ~used
    else:
        loo_all = leave_one_out_residuals(src, dst, reflection=reflection) * p_r
        finite = np.isfinite(loo_all)
        med = float(np.median(loo_all[finite])) if finite.any() else math.nan
        outlier = finite & (loo_all > options.floor_um) & (loo_all > options.k * med)
    fit = fit_similarity(src[used], dst[used], reflection=reflection)
    residuals_um = np.linalg.norm(apply_matrix_xy(fit.matrix, src) - dst, axis=1) * p_r
    loo_um = residuals_um.copy()
    loo_um[used] = leave_one_out_residuals(src[used], dst[used], reflection=reflection) * p_r
    chosen = float(np.median(residuals_um[used]))
    other = fit_similarity(src[used], dst[used], reflection=not reflection)
    other_median = float(np.median(other.residuals_px * p_r))
    return RobustSimilarityFit(
        fit=fit,
        used=used,
        outlier=np.asarray(outlier, dtype=bool),
        residuals_um=residuals_um,
        loo_um=loo_um,
        chosen_class_median_um=chosen,
        other_class_median_um=other_median,
        spread_ratio=spread_ratio(src[used]),
        scale_ratio=fit.scale / (p_m / p_r),
    )


def evaluate_landmarks(
    state: LabellingState,
    *,
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
    outlier_floor_um: float = DEFAULT_OUTLIER_FLOOR_UM,
) -> LandmarkEvaluation:
    """Evaluate the live fit of a labelling state (the only live-fit implementation).

    The reflection class is ``state.coarse.mirror_x``. Enabled pairs go through
    ``robust_similarity``; disabled pairs are scored under the resulting matrix. Missing values
    are NaN. With fewer than 2 enabled pairs, or on degenerate geometry (``error`` is set), no
    summary or matrix is returned. Summary warnings come in the order of the master design.

    Args:
        state: The labelling state.
        moving_pixel_size_um: Moving level-0 pixel size.
        reference_pixel_size_um: Reference level-0 pixel size.
        outlier_floor_um: Outlier floor of the robust fit (10 same_section, 30 serial_section).
    """
    pairs = sorted(state.pairs, key=lambda p: p.id)
    enabled = [p for p in pairs if p.enabled]
    n = len(enabled)
    outside = pairs_outside_crop(pairs, state.crops)
    nan = float("nan")

    def blank_rows() -> tuple[ResidualRow, ...]:
        return tuple(
            ResidualRow(
                id=p.id,
                colour=p.colour,
                enabled=p.enabled,
                used=False,
                outlier=False,
                outside_crop=bool(outside[i]),
                fit_um=nan,
                loo_um=nan,
            )
            for i, p in enumerate(pairs)
        )

    if n < 2:
        return LandmarkEvaluation(rows=blank_rows(), summary=None, matrix=None)
    src = np.array([p.moving_xy for p in enabled], dtype=np.float64)
    dst = np.array([p.reference_xy for p in enabled], dtype=np.float64)
    try:
        robust = robust_similarity(
            src,
            dst,
            reflection=state.coarse.mirror_x,
            moving_pixel_size_um=moving_pixel_size_um,
            reference_pixel_size_um=reference_pixel_size_um,
            options=RobustFitOptions(floor_um=outlier_floor_um),
        )
    except ValueError as exc:
        return LandmarkEvaluation(rows=blank_rows(), summary=None, matrix=None, error=str(exc))

    matrix = robust.fit.matrix
    enabled_index = {p.id: i for i, p in enumerate(enabled)}
    rows: list[ResidualRow] = []
    for i, p in enumerate(pairs):
        if p.enabled:
            k = enabled_index[p.id]
            rows.append(
                ResidualRow(
                    id=p.id,
                    colour=p.colour,
                    enabled=True,
                    used=bool(robust.used[k]),
                    outlier=bool(robust.outlier[k]),
                    outside_crop=bool(outside[i]),
                    fit_um=float(robust.residuals_um[k]),
                    loo_um=float(robust.loo_um[k]),
                )
            )
        else:
            err = float(
                np.linalg.norm(
                    apply_matrix_xy(matrix, np.asarray(p.moving_xy)) - np.asarray(p.reference_xy)
                )
                * reference_pixel_size_um
            )
            rows.append(
                ResidualRow(
                    id=p.id,
                    colour=p.colour,
                    enabled=False,
                    used=False,
                    outlier=False,
                    outside_crop=bool(outside[i]),
                    fit_um=err,
                    loo_um=err,
                )
            )
    n_used = int(robust.used.sum())
    warns: list[str] = []
    if n < 3:
        warns.append("Leave-one-out residuals need at least 3 enabled pairs.")
    if n_used >= 3 and robust.other_class_median_um * 2 <= robust.chosen_class_median_um:
        warns.append(
            "The opposite mirror state fits these landmarks at least 2× better: "
            "check Flip H / Flip V."
        )
    if abs(robust.scale_ratio - 1) > 0.05:
        warns.append(
            "Fitted scale differs from the pixel-size ratio by "
            f"{100 * (robust.scale_ratio - 1):+.1f} %: check the pixel sizes or the landmarks."
        )
    if n_used >= 3 and robust.spread_ratio < 0.2:
        warns.append(
            f"Landmarks are nearly collinear (spread {robust.spread_ratio:.2f}): "
            "add pairs across the tissue."
        )
    outlier_ids = [r.id for r in rows if r.outlier]
    if outlier_ids:
        warns.append(
            f"{len(outlier_ids)} pair(s) flagged as outliers and excluded: "
            f"{', '.join(str(i) for i in outlier_ids)}"
        )
    outside_ids = [r.id for r in rows if r.outside_crop]
    if outside_ids:
        warns.append(
            f"{len(outside_ids)} pair(s) outside the crops: "
            f"{', '.join(str(i) for i in outside_ids)}. The register workflow uses them "
            "unless landmarks_outside_crop is set."
        )
    used_res = robust.residuals_um[robust.used]
    summary = FitSummary(
        n_enabled=n,
        n_used=n_used,
        scale=float(robust.fit.scale),
        expected_scale=float(moving_pixel_size_um / reference_pixel_size_um),
        scale_ratio=float(robust.scale_ratio),
        rotation_deg=float(robust.fit.rotation_deg),
        reflected=bool(robust.fit.reflected),
        rms_um=float(math.sqrt(float(np.mean(used_res**2)))),
        other_class_median_um=float(robust.other_class_median_um),
        chosen_class_median_um=float(robust.chosen_class_median_um),
        spread_ratio=float(robust.spread_ratio),
        warnings=tuple(warns),
    )
    return LandmarkEvaluation(rows=tuple(rows), summary=summary, matrix=matrix)


# ----------------------------------------------------------------------------------------------
# Serialisation and digests
# ----------------------------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON text of ``obj`` (sorted keys, compact, ASCII, no NaN).

    Raises:
        ValueError: If ``obj`` contains NaN or infinity.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_sha256(obj: Any) -> str:
    """Return ``"sha256:"`` plus the hex SHA-256 of ``canonical_json(obj)``."""
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def labelling_content_sha256(payload: Mapping[str, Any]) -> str:
    """Digest of the content of a labelling payload (coarse, crops and pairs only)."""
    return canonical_sha256(
        {
            "coarse": {k: payload["coarse"][k] for k in ("rotation_deg", "mirror_x", "offset_um")},
            "crops": payload["crops"],
            "pairs": payload["landmarks"]["pairs"],
        }
    )


def _content_dict(state: LabellingState) -> dict[str, Any]:
    crops = state.crops
    return {
        "coarse": state.coarse.to_attrs(),
        "crops": {
            role: (crops[role].to_attrs() if crops.get(role) is not None else None)
            for role in ROLES
        },
        "pairs": [p.to_attrs() for p in sorted(state.pairs, key=lambda p: p.id)],
    }


def utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def labelling_to_attrs(
    state: LabellingState,
    descriptor: StoreDescriptor,
    *,
    status: str,
    software_version: str,
    outlier_floor_um: float | None = None,
) -> dict[str, Any]:
    """Build the labelling payload of ``state`` for a prepared store.

    ``revision`` is a placeholder (``state.revision or 0``); ``save_labelling`` replaces it, and
    also ``updated_at``, ``finished_at``, ``written_by`` and ``content_sha256``.

    Raises:
        ValueError: If the store is not a prepared one, ``status`` is invalid or a value is
            NaN or infinite.
    """
    if descriptor.kind != "prepared":
        raise ValueError("landmarks can only be written to a prepared store")
    if status not in ("draft", "finished"):
        raise ValueError(f"status must be 'draft' or 'finished', got {status!r}")
    now = utc_now()
    matrix = coarse_matrix_xy(
        state.coarse,
        moving_shape_yx=descriptor.images["moving"].shape_yx,
        reference_shape_yx=descriptor.images["reference"].shape_yx,
        moving_pixel_size_um=descriptor.images["moving"].pixel_size_um,
        reference_pixel_size_um=descriptor.images["reference"].pixel_size_um,
    )
    content = _content_dict(state)
    coarse = dict(content["coarse"])
    coarse["matrix_xy"] = [[_round(v, 7) for v in row] for row in matrix.tolist()]
    result: dict[str, Any] = {
        "schema": LABELLING_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "revision": int(state.revision or 0),
        "updated_at": now,
        "finished_at": now if status == "finished" else None,
        "written_by": {"software": SOFTWARE_NAME, "version": software_version},
        "store_grid_digest": descriptor.grid_digest,
        "images": {
            role: {
                "shape_yx": [int(v) for v in descriptor.images[role].shape_yx],
                "pixel_size_um": float(descriptor.images[role].pixel_size_um),
            }
            for role in ROLES
        },
        "coarse": coarse,
        "crops": content["crops"],
        "landmarks": {
            "next_id": int(state.next_id),
            "colour_seed": int(state.colour_seed),
            "pairs": content["pairs"],
        },
    }
    result["content_sha256"] = labelling_content_sha256(result)
    if outlier_floor_um is not None:
        result["outlier_floor_um"] = _round(outlier_floor_um, 1)
    canonical_json(result)  # raises ValueError on NaN / inf
    return result


def _parse_coarse(payload: Mapping[str, Any]) -> CoarseState:
    coarse = payload.get("coarse")
    if not isinstance(coarse, Mapping):
        raise ValueError("landmark labelling has no 'coarse' block")
    rotation = coarse.get("rotation_deg")
    if not _is_number(rotation) or not math.isfinite(float(rotation)):
        raise ValueError(f"invalid coarse rotation_deg {rotation!r}")
    mirror = coarse.get("mirror_x")
    if not isinstance(mirror, bool):
        raise ValueError(f"coarse mirror_x must be a bool, got {mirror!r}")
    offset = _float_pair(coarse.get("offset_um"), "coarse offset_um")
    return CoarseState(rotation_deg=float(rotation), mirror_x=mirror, offset_um=offset)


def labelling_from_attrs(payload: Mapping[str, Any], descriptor: StoreDescriptor) -> LabellingState:
    """Parse a labelling payload (store attribute or local draft) into a state.

    Raises:
        ValueError: If the payload is not a labelling or a value is invalid.
        UnsupportedSchemaVersion: If it was written with a newer schema version.
        LabellingMismatch: If it was placed on images of another shape or pixel size.
    """
    if not isinstance(payload, Mapping) or payload.get("schema") != LABELLING_SCHEMA:
        raise ValueError("not a landmark labelling payload")
    version = payload.get("schema_version")
    if not _is_int(version):
        raise ValueError(f"landmark labelling schema_version must be an integer, got {version!r}")
    if version > SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"landmark labelling schema_version {version} was written by a newer MerXen/viewer; "
            f"update this viewer (it reads version {SCHEMA_VERSION})"
        )
    warnings: list[str] = []

    images = payload.get("images")
    if not isinstance(images, Mapping):
        raise LabellingMismatch(
            "the stored landmarks have no 'images' block, so they cannot be matched to the "
            "store's images; start fresh"
        )
    for role in ROLES:
        have = descriptor.images[role]
        try:
            rec = images[role]
            stored_shape = tuple(int(v) for v in rec["shape_yx"])
            stored_px = float(rec["pixel_size_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LabellingMismatch(
                f"the stored landmarks have no valid image record for {role}; start fresh"
            ) from exc
        if stored_shape != tuple(have.shape_yx) or not math.isclose(
            stored_px, have.pixel_size_um, rel_tol=1e-9
        ):
            raise LabellingMismatch(
                f"landmarks were placed on different images ({role}: stored shape "
                f"{stored_shape} at {stored_px} µm/px, store has {tuple(have.shape_yx)} at "
                f"{have.pixel_size_um} µm/px); start fresh"
            )
    stored_digest = payload.get("store_grid_digest")
    if stored_digest and descriptor.grid_digest and stored_digest != descriptor.grid_digest:
        warnings.append(
            "the store's source images changed fingerprint since the landmarks were saved; "
            "the image shapes still match"
        )

    coarse = _parse_coarse(payload)
    crops_raw = payload.get("crops")
    crops: dict[str, CropRect | None] = {}
    for role in ROLES:
        value = crops_raw.get(role) if isinstance(crops_raw, Mapping) else None
        try:
            crop = CropRect.from_attrs(value)
            if crop is not None:
                crop.validate_within(descriptor.images[role].shape_yx)
        except ValueError as exc:
            raise ValueError(f"invalid {role} crop {value!r}") from exc
        crops[role] = crop

    landmarks = payload.get("landmarks")
    if not isinstance(landmarks, Mapping):
        raise ValueError("landmark labelling has no 'landmarks' block")
    seed = landmarks.get("colour_seed", 0)
    if not _is_int(seed):
        raise ValueError(f"colour_seed must be an integer, got {seed!r}")
    pairs: list[LandmarkPair] = []
    seen: set[int] = set()
    raw_pairs = landmarks.get("pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError("landmarks.pairs must be a list")
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid landmark pair {raw!r}")
        pid = raw.get("id")
        if not _is_int(pid) or pid < 1:
            raise ValueError(f"landmark id must be an integer >= 1, got {pid!r}")
        pid = int(pid)
        if pid in seen:
            raise ValueError(f"duplicate landmark id {pid}")
        seen.add(pid)
        colour = raw.get("colour")
        if not isinstance(colour, str) or not _COLOUR_RE.match(colour):
            warnings.append(f"landmark {pid} had an invalid colour; replaced")
            colour = pair_colour(pid, int(seed))
        ref = _float_pair(raw.get("reference_xy_px"), f"landmark {pid} reference_xy_px")
        mov = _float_pair(raw.get("moving_xy_px"), f"landmark {pid} moving_xy_px")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"landmark {pid} enabled must be a bool, got {enabled!r}")
        for role, pt in (("reference", ref), ("moving", mov)):
            if not point_in_image(pt, descriptor.images[role].shape_yx):
                warnings.append(f"landmark {pid} {role} point {pt} lies outside its image")
        pairs.append(
            LandmarkPair(id=pid, colour=colour, reference_xy=ref, moving_xy=mov, enabled=enabled)
        )
    pairs.sort(key=lambda p: p.id)

    next_id = landmarks.get("next_id")
    max_id = max((p.id for p in pairs), default=0)
    if not _is_int(next_id):
        raise ValueError(f"landmarks.next_id must be an integer, got {next_id!r}")
    next_id = int(next_id)
    if next_id <= max_id:
        warnings.append(f"next_id {next_id} was not above the highest landmark id; repaired")
        next_id = max_id + 1

    status = payload.get("status")
    if status not in ("draft", "finished"):
        raise ValueError(f"status must be 'draft' or 'finished', got {status!r}")
    revision = payload.get("revision")
    if not _is_int(revision) or revision < 1:
        raise ValueError(f"revision must be an integer >= 1, got {revision!r}")

    floor: float | None = None
    if "outlier_floor_um" in payload and payload["outlier_floor_um"] is not None:
        raw_floor = payload["outlier_floor_um"]
        if _is_number(raw_floor) and math.isfinite(float(raw_floor)) and float(raw_floor) > 0:
            floor = float(raw_floor)
        else:
            warnings.append(f"ignored an invalid outlier_floor_um {raw_floor!r}")

    try:
        digest_matches = payload.get("content_sha256") == labelling_content_sha256(payload)
    except (KeyError, TypeError):  # e.g. no 'crops' block: the stored digest cannot match
        digest_matches = False
    if not digest_matches:
        warnings.append(
            "content_sha256 does not match the stored landmarks (edited outside the viewer?)"
        )

    state = LabellingState(
        coarse=coarse,
        crops=crops,
        pairs=pairs,
        next_id=next_id,
        colour_seed=int(seed),
        status=status,
        revision=int(revision),
        outlier_floor_um=floor,
        load_warnings=tuple(warnings),
    )
    state.loaded_content_sha = state.content_sha256()
    return state


def finish_allowed(n_enabled: int, has_pending: bool, busy: bool) -> tuple[bool, str]:
    """Whether "Finish labelling" is allowed, with the reason when it is not."""
    if busy:
        return False, "Saving… please wait."
    if has_pending:
        return False, "Finish the pending pair (click its moving point) or press Esc to cancel it."
    if n_enabled < MIN_LANDMARK_PAIRS:
        return (
            False,
            f"Finish needs at least {MIN_LANDMARK_PAIRS} enabled landmark pairs "
            f"({n_enabled} so far).",
        )
    return True, ""


# ----------------------------------------------------------------------------------------------
# Store descriptor
# ----------------------------------------------------------------------------------------------


def _unsupported(what: str, version: int) -> UnsupportedSchemaVersion:
    return UnsupportedSchemaVersion(
        f"manual-registration {what} schema_version {version} was written by a newer "
        f"MerXen/viewer; update this viewer (it reads version {SCHEMA_VERSION})"
    )


def _role_record(images: Any, role: str) -> dict[str, Any]:
    """Validate ``images[role]`` of a store attribute and return it."""

    def bad(field_name: str) -> NotARegistrationStore:
        return NotARegistrationStore(
            f"store attribute images.{role}.{field_name} is missing or invalid"
        )

    rec = images.get(role) if isinstance(images, Mapping) else None
    if not isinstance(rec, Mapping):
        raise bad("element")
    if not isinstance(rec.get("element"), str) or not rec["element"]:
        raise bad("element")
    shape = rec.get("shape_yx")
    if not (
        isinstance(shape, Sequence)
        and not isinstance(shape, (str, bytes))
        and len(shape) == 2
        and all(_is_int(v) and v > 0 for v in shape)
    ):
        raise bad("shape_yx")
    px = rec.get("pixel_size_um")
    if not (_is_number(px) and math.isfinite(float(px)) and float(px) > 0):
        raise bad("pixel_size_um")
    names = rec.get("channel_names")
    if not (
        isinstance(names, Sequence)
        and not isinstance(names, (str, bytes))
        and len(names) > 0
        and all(isinstance(v, str) for v in names)
    ):
        raise bad("channel_names")
    if rec.get("registration_channel") not in names:
        raise bad("registration_channel")
    return dict(rec)


def parse_store_descriptor(path: Path, root_attrs: Mapping[str, Any]) -> StoreDescriptor:
    """Build a ``StoreDescriptor`` from the root attributes of a store.

    Raises:
        NotARegistrationStore: Missing or malformed store attribute, bad image record or an
            incomplete result store.
        UnsupportedSchemaVersion: A payload with a schema version above 1.
    """
    path = Path(path)
    sa = root_attrs.get(STORE_ATTR)
    if not isinstance(sa, dict):
        raise NotARegistrationStore(
            f"{path.name} is not a manual-registration store (no '{STORE_ATTR}' root attribute). "
            "Create one with the MerXen manual_registration_prepare workflow."
        )
    if sa.get("schema") != STORE_SCHEMA:
        raise NotARegistrationStore(
            f"{path.name} is not a manual-registration store (unknown store schema "
            f"{sa.get('schema')!r})"
        )
    version = sa.get("schema_version")
    if not _is_int(version):
        raise NotARegistrationStore(f"{path.name}: store schema_version is missing or invalid")
    if version > SCHEMA_VERSION:
        raise _unsupported("store", version)
    kind = sa.get("kind")
    if kind not in ("prepared", "result"):
        raise NotARegistrationStore(f"{path.name}: store kind {kind!r} is not prepared or result")

    warnings: list[str] = []
    images_attr = sa.get("images")
    records = {role: _role_record(images_attr, role) for role in ROLES}

    origin_x = origin_y = 0.0
    p_ref = float(records["reference"]["pixel_size_um"])
    if kind == "result":
        crop_value = records["reference"].get("crop")
        if crop_value is None:
            warnings.append("result store has no reference crop; assuming the full grid")
        else:
            try:
                crop = CropRect.from_attrs(crop_value)
            except ValueError as exc:
                raise NotARegistrationStore(
                    "store attribute images.reference.crop is missing or invalid"
                ) from exc
            assert crop is not None
            origin_x, origin_y = float(crop.x0), float(crop.y0)

    images: dict[str, RoleImage] = {}
    for role in ROLES:
        rec = records[role]
        px = float(rec["pixel_size_um"])
        if kind == "prepared":
            base = np.diag([px, px, 1.0])
        else:
            base = np.array(
                [[p_ref, 0.0, p_ref * origin_y], [0.0, p_ref, p_ref * origin_x], [0.0, 0.0, 1.0]]
            )
        images[role] = RoleImage(
            role=role,
            element_key=str(rec["element"]),
            channel_names=tuple(rec["channel_names"]),
            registration_channel=str(rec["registration_channel"]),
            shape_yx=(int(rec["shape_yx"][0]), int(rec["shape_yx"][1])),
            pixel_size_um=px,
            base_affine_rc=np.asarray(base, dtype=np.float64),
        )

    labelling: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    if kind == "prepared":
        grid_digest = sa.get("grid_digest")
        lab = root_attrs.get(LABELLING_ATTR)
        if lab is not None:
            if not isinstance(lab, dict):
                warnings.append("ignored a malformed landmark labelling attribute")
            else:
                lab_version = lab.get("schema_version")
                if _is_int(lab_version) and lab_version > SCHEMA_VERSION:
                    raise _unsupported("landmark labelling", lab_version)
                labelling = lab
    else:
        grid_digest = (sa.get("prepared_store") or {}).get("grid_digest")
        res = root_attrs.get(RESULT_ATTR)
        if not isinstance(res, dict) or res.get("schema") != RESULT_SCHEMA:
            raise NotARegistrationStore(
                "result store has no complete merxen_manual_registration_result attribute; "
                "the register run did not finish"
            )
        res_version = res.get("schema_version")
        if _is_int(res_version) and res_version > SCHEMA_VERSION:
            raise _unsupported("result", res_version)
        result = res
    dataset_name = sa.get("dataset_name")
    return StoreDescriptor(
        path=path,
        kind=str(kind),
        dataset_name=dataset_name if isinstance(dataset_name, str) else path.stem,
        images=images,
        grid_digest=grid_digest if isinstance(grid_digest, str) else None,
        store_attr=json.loads(json.dumps(sa)),
        labelling=labelling,
        result=result,
        warnings=tuple(warnings),
    )
