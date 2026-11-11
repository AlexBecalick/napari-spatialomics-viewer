"""Native-frame crop selection and Cellpose label estimates for HITL exports.

The portable context/GeoJSON contract is shared with MerXen. This module does
not require MerXen or load transcripts, image pixels or a whole-slide mask.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np


class SpatialImageSource(TypedDict):
    """Backing SpatialData identity of a displayed image, independent of its name."""

    store_path: str
    platform: str
    image_key: str
    channel: str
    coordinate_frame: Literal["native", "aligned"]


@dataclass(frozen=True)
class CropRectangle:
    """One axis-aligned rectangle in native level-zero pixels."""

    y0: int
    x0: int
    height: int
    width: int

    def vertices(self) -> np.ndarray:
        """Return napari's four rectangle vertices in native (y, x) order."""
        y, x, h, w = self.y0, self.x0, self.height, self.width
        return np.asarray([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], float)


@dataclass(frozen=True)
class HitlCropContext:
    """Source identity and original inference grid supplied by MerXen."""

    platform: str
    image_key: str
    dataset_alias: str
    height: int
    width: int
    pixel_size_um: float
    tile_size: int
    overlap_px: int
    margin_px: int
    crop_size_px: int
    source_metadata_sha256: str
    baseline_stats_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.platform not in {"ATERA", "XENIUM"}:
            raise ValueError("Choose an Atera or Xenium HITL crop setup from MerXen.")
        if self.image_key != "morphology_focus":
            raise ValueError("Crop setup requires the native morphology image.")
        if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError("Crop setup has no valid native pixel size.")
        if min(self.height, self.width, self.tile_size) <= 0 or (
            self.overlap_px < 0
            or self.margin_px < 0
            or self.tile_size <= 2 * self.overlap_px
        ):
            raise ValueError("Crop setup has an invalid image or inference grid.")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (self.source_metadata_sha256, self.baseline_stats_sha256)
        ):
            raise ValueError("Crop setup is missing its source/grid identity.")

    @classmethod
    def read(cls, path: str | Path) -> HitlCropContext:
        """Read a portable roi_context.json produced by MerXen."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_store(self, path: Path, platform: str) -> None:
        """Refuse setup files belonging to another source or platform."""
        if platform != self.platform:
            raise ValueError("This crop setup belongs to a different platform.")
        metadata = next(
            (
                path / name
                for name in ("zarr.json", ".zattrs")
                if (path / name).is_file()
            ),
            None,
        )
        if (
            metadata is None
            or hashlib.sha256(metadata.read_bytes()).hexdigest()
            != self.source_metadata_sha256
        ):
            raise ValueError(
                "This crop setup belongs to a different source store. "
                "Open the matching native review store and its roi_context.json."
            )

    def usable_areas(self) -> list[CropRectangle]:
        """Return the tile interiors that retain the original inference margins."""
        core = self.tile_size - 2 * self.overlap_px
        result = []
        for y in range(0, self.height, core):
            for x in range(0, self.width, core):
                h = min(y + core, self.height) - y - 2 * self.margin_px
                w = min(x + core, self.width) - x - 2 * self.margin_px
                if min(h, w) >= 256:
                    result.append(
                        CropRectangle(y + self.margin_px, x + self.margin_px, h, w)
                    )
        return result

    def rectangle(self, vertices: np.ndarray) -> CropRectangle:
        """Validate a drawn rectangle, rounding its edges onto native pixels."""
        points = np.asarray(vertices, float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Draw a rectangle with four finite corners.")
        points = np.rint(points).astype(np.int64)
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        corners = {(y0, x0), (y0, x1), (y1, x1), (y1, x0)}
        edges = np.diff(np.vstack([points, points[0]]), axis=0)
        if set(map(tuple, points)) != corners or not np.all(
            np.count_nonzero(edges, axis=1) == 1
        ):
            raise ValueError(
                "Keep crop edges horizontal and vertical; do not rotate crops."
            )
        rectangle = CropRectangle(int(y0), int(x0), int(y1 - y0), int(x1 - x0))
        if min(rectangle.height, rectangle.width) < 256:
            raise ValueError("Make each crop at least 256 native pixels wide and high.")
        if not any(
            y0 >= area.y0
            and x0 >= area.x0
            and y1 <= area.y0 + area.height
            and x1 <= area.x0 + area.width
            for area in self.usable_areas()
        ):
            raise ValueError("Keep the entire crop inside one cyan guide.")
        return rectangle


def validate_crop_rectangles(
    context: HitlCropContext, vertices: Sequence[np.ndarray]
) -> list[CropRectangle]:
    """Validate all crop geometry and refuse overlapping annotation fields."""
    rectangles = [context.rectangle(points) for points in vertices]
    for index, box in enumerate(rectangles):
        if any(
            box.y0 < other.y0 + other.height
            and box.y0 + box.height > other.y0
            and box.x0 < other.x0 + other.width
            and box.x0 + box.width > other.x0
            for other in rectangles[:index]
        ):
            raise ValueError(
                "Crops overlap. Move or remove overlapping crops before saving."
            )
    return rectangles


@dataclass(frozen=True)
class CropCellCounts:
    """Per-crop estimates and their unique union of nonzero Cellpose labels."""

    per_crop: tuple[int, ...]
    total: int
    label_key: str


def count_crop_cells(
    store_path: Path, context: HitlCropContext, rectangles: Sequence[CropRectangle]
) -> CropCellCounts:
    """Read only the selected mask windows and count each cell once overall."""
    import spatialdata as sd

    from .utils import get_scale0_dataarray

    labels = sd.read_zarr(str(store_path), selection=("labels",)).labels
    key = next(
        (
            name
            for name in ("cellpose_cell_labels", "MOSAIK_cellpose_labels")
            if name in labels
        ),
        None,
    )
    if key is None:
        raise ValueError(
            "No Cellpose masks found. "
            "Open a store containing Cellpose masks to estimate cells."
        )
    array = get_scale0_dataarray(labels[key])
    if set(array.dims) != {"y", "x"}:
        raise ValueError("Cellpose masks must be a two-dimensional native label image.")
    array = array.transpose("y", "x")
    if tuple(array.shape) != (context.height, context.width):
        raise ValueError("Cellpose masks do not match the native image resolution.")
    all_cells: set[int] = set()
    counts = []
    for box in rectangles:
        window = array.data[box.y0 : box.y0 + box.height, box.x0 : box.x0 + box.width]
        if hasattr(window, "compute"):
            window = window.compute()
        values = np.unique(np.asarray(window))
        cells = {int(value) for value in values if value > 0}
        all_cells.update(cells)
        counts.append(len(cells))
    return CropCellCounts(tuple(counts), len(all_cells), key)


def save_crop_regions(
    path: Path, context: HitlCropContext, vertices: Sequence[np.ndarray]
) -> None:
    """Save exporter-compatible native-pixel GeoJSON without replacing a file."""
    rectangles = validate_crop_rectangles(context, vertices)
    if not rectangles:
        raise ValueError("Add at least one crop before saving.")
    features = []
    for index, box in enumerate(rectangles, 1):
        xy = box.vertices()[:, ::-1].astype(int).tolist()
        features.append(
            {
                "type": "Feature",
                "properties": {"roi_id": f"roi_{index:03d}"},
                "geometry": {"type": "Polygon", "coordinates": [[*xy, xy[0]]]},
            }
        )
    payload = {
        "type": "FeatureCollection",
        "coordinate_system": "native_pixels",
        "merxen_hitl_regions_version": 2,
        "merxen_hitl_context": asdict(context),
        "features": features,
    }
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
