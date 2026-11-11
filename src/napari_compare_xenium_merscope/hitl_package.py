"""Create complete local Cellpose correction packages from native viewer crops."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import spatialdata as sd
import tifffile
from skimage.measure import label

from .hitl_annotation_session import (
    AnnotationPackage,
    file_sha256,
    open_annotation_package,
)
from .hitl_crops import (
    CropRectangle,
    HitlCropContext,
    save_crop_regions,
    validate_crop_rectangles,
)
from .hitl_normalization import normalize_model_plane
from .utils import ensure_cyx, get_scale0_dataarray


@dataclass(frozen=True)
class AnnotationExportSettings:
    """Portable model-channel and normalization settings supplied by MerXen."""

    channels: tuple[str, ...]
    input_normalization: str
    density: dict[str, Any] | None
    baseline_identity: dict[str, Any] | None
    guide_version: str = "pilot-v2"


def load_annotation_export_settings(setup_path: Path) -> AnnotationExportSettings:
    """Read the companion setup or an existing annotation package's manifest."""
    companion = setup_path.parent / "hitl_export_settings.json"
    manifest = setup_path.parent / "manifest.json"
    if companion.is_file():
        values = json.loads(companion.read_text())
    elif manifest.is_file():
        values = json.loads(manifest.read_text())
    else:
        raise ValueError(
            "This crop setup needs hitl_export_settings.json. Prepare it with "
            "the updated MerXen crop-setup command, or choose roi_context.json "
            "from an existing annotation package."
        )
    channels = tuple(values["model_channels"])
    mode = values["input_normalization"]
    if not 1 <= len(channels) <= 3 or len(set(channels)) != len(channels):
        raise ValueError("Crop setup must specify one to three unique model channels.")
    if mode not in {"joint", "per_channel"}:
        raise ValueError("Unsupported model input normalization.")
    density = values.get("transcript_density")
    if "transcript_density" in channels and (density is None or mode != "per_channel"):
        raise ValueError(
            "Density inputs need a matching raster and per-channel normalization."
        )
    return AnnotationExportSettings(
        channels,
        mode,
        density,
        values.get("baseline_stats_model_identity"),
        values.get("guide_version", "pilot-v2"),
    )


def validate_local_density(
    path: Path, context: HitlCropContext, settings: AnnotationExportSettings
) -> dict[str, Any]:
    """Check the portable raster manifest, recipe and native image geometry."""
    record = json.loads((path / "transcript_density.json").read_text())
    expected = settings.density
    if expected is None:
        raise ValueError("The crop setup does not select a density channel.")
    payload = {key: value for key, value in record.items() if key != "fingerprint"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != record.get("fingerprint") or digest != expected.get("fingerprint"):
        raise ValueError("This density raster does not match the selected crop setup.")
    if record.get("profile") != expected.get("profile"):
        raise ValueError("This density raster uses a different generation recipe.")
    geometry = record["geometry"]
    if (geometry["height"], geometry["width"]) != (
        context.height,
        context.width,
    ) or not np.isclose(
        geometry["pixel_size_um"], context.pixel_size_um, rtol=0, atol=1e-10
    ):
        raise ValueError("Density raster does not match the native image geometry.")
    return record


def find_local_density(
    source: Path, setup_path: Path, settings: AnnotationExportSettings
) -> Path | None:
    """Find an existing raster by local convention or the original setup hint."""
    candidates = [
        source.parent / "transcript_density.zarr",
        source.parent / "transcript_density" / "transcript_density.zarr",
        setup_path.parent / "transcript_density.zarr",
    ]
    if settings.density and settings.density.get("raster_path"):
        candidates.append(Path(settings.density["raster_path"]))
    return next(
        (path for path in candidates if (path / "transcript_density.json").is_file()),
        None,
    )


def _read(array: Any) -> np.ndarray:
    return np.asarray(array.compute() if hasattr(array, "compute") else array)


def _tile_for(box: CropRectangle, context: HitlCropContext) -> dict[str, int]:
    core = context.tile_size - 2 * context.overlap_px
    y, x = box.y0 // core * core, box.x0 // core * core
    return {
        "core_y0": y,
        "core_y1": min(y + core, context.height),
        "core_x0": x,
        "core_x1": min(x + core, context.width),
        "tile_y0": max(0, y - context.overlap_px),
        "tile_y1": min(y + core + context.overlap_px, context.height),
        "tile_x0": max(0, x - context.overlap_px),
        "tile_x1": min(x + core + context.overlap_px, context.width),
    }


_GUIDE = """# Correct the selected cell masks

Open crops from this package through the viewer's **Annotate cell masks** button.
Each image starts with the current Cellpose masks. Correct every visible cell,
including missing cells, false detections and incorrect merges/splits.

Use DAPI to separate touching nuclei. Include a visible contiguous perinuclear
rim; do not invent missing cytoplasm. For neurons, include the soma and an
unbroken proximal dendrite trunk up to half the soma diameter, capped at 10 µm.
Exclude thin branches, distal processes, neuropil and debris. Keep one connected
component per cell. Record uncertainty in review.tsv's notes column.

Press **Ctrl+S**, then close Cellpose and return to the viewer. Save even when no
changes were needed. A verified save marks that crop annotated. Keep correcting
crops until the viewer reports completion. GUI contrast changes affect display;
leave original image, raw and model_input files unchanged.

These are training-pilot crops, not held-out evaluation material. Training
runs later on the MerXen server. Keep the complete folder together when copying.

A DAPI nucleus counts even if its cytoplasmic stain is weak. Draw a compact,
clearly bounded anucleate soma only when the staining is convincing and the soma
is at least about 5 µm across. Separate oligodendrocyte rows into individual
cells. Draw vascular cells, not the vessel lumen. If a boundary is still unclear
after about ten seconds, draw the clear part and record the uncertainty in
review.tsv. Include partial cells at the crop edge; avoid overlapping masks.
"""


def upgrade_legacy_annotation_package(
    original: Path, destination_parent: Path, context: HitlCropContext
) -> AnnotationPackage:
    """Copy an older pilot into a managed package while retaining all GUI edits."""
    from .hitl_annotation_session import package_file

    original = original.resolve()
    old = json.loads((original / "manifest.json").read_text())
    if old.get("evaluation_eligible") is not False or any(
        row.get("split") != "pilot_train" for row in old["crops"]
    ):
        raise ValueError(
            "Only training-pilot packages can be upgraded. "
            "Keep held-out evaluation crops separate."
        )
    old_context = HitlCropContext.read(original / "roi_context.json")
    if old_context != context:
        raise ValueError("The older package belongs to a different source/grid.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package_id = f"hitl-{stamp}-{uuid.uuid4().hex[:8]}"
    output = destination_parent / package_id
    staging = destination_parent / f".{package_id}.preparing"
    staging.mkdir()
    try:
        for name in ("raw", "model_input", "context", "masks", "receipts"):
            (staging / name).mkdir()
        (staging / "annotation-guide.md").write_text(_GUIDE)
        shutil.copy2(original / "roi_context.json", staging / "roi_context.json")
        old_review = {}
        if (original / "review.tsv").is_file():
            with (original / "review.tsv").open(newline="") as stream:
                old_review = {
                    row["crop_id"]: row
                    for row in csv.DictReader(stream, delimiter="\t")
                }
        rows, review = [], []
        for index, old_row in enumerate(old["crops"], 1):
            old_id, crop_id = old_row["crop_id"], f"crop_{index:03d}"
            row = dict(old_row)
            row.update(crop_id=crop_id, original_crop_id=old_id, files_sha256={})
            for old_relative, relative, immutable in (
                (f"{old_id}.tif", f"{crop_id}.tif", True),
                (f"{old_id}_seg.npy", f"{crop_id}_seg.npy", False),
                (f"raw/{old_id}.npy", f"raw/{crop_id}.npy", True),
                (f"model_input/{old_id}.npy", f"model_input/{crop_id}.npy", True),
            ):
                source = package_file(original, old_relative)
                if immutable and file_sha256(source) != old_row["files_sha256"].get(
                    old_relative
                ):
                    raise ValueError(
                        f"An older package input has changed: {old_relative}"
                    )
                shutil.copyfile(source, staging / relative)
                row["files_sha256"][relative] = file_sha256(staging / relative)
            row["split"] = "pilot_train"
            rows.append(row)
            previous = dict(old_review.get(old_id, {}))
            review.append(
                {
                    "crop_id": crop_id,
                    "status": "pending",
                    "reason": previous.get("reason", ""),
                    "minutes": previous.get("minutes", ""),
                    "notes": previous.get("notes", ""),
                }
            )
        manifest = dict(old)
        manifest.update(
            managed_annotation_version=1,
            package_id=package_id,
            crops=rows,
            roi_context=asdict(context),
            source_metadata_sha256=context.source_metadata_sha256,
            guide_version=f"{old.get('guide_version', 'pilot-v2')}-managed-v1",
            transfer_relative_path=f"annotations/{package_id}/",
            evaluation_eligible=False,
            origin_manifest_sha256=file_sha256(original / "manifest.json"),
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (staging / "review.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["crop_id", "status", "reason", "minutes", "notes"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(review)
        (staging / "training-template.json").write_text(
            json.dumps(
                {
                    "hitl_root": None,
                    "annotation_package_ids": [package_id],
                    "model_name": None,
                    "platform": context.platform,
                    "species": None,
                    "n_epochs": 100,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.1,
                    "batch_size": 1,
                    "seed": 0,
                },
                indent=2,
            )
            + "\n"
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return open_annotation_package(output)


def export_local_annotation_package(
    source: Path,
    context: HitlCropContext,
    rectangles_yx: list[np.ndarray],
    settings: AnnotationExportSettings,
    *,
    density_path: Path | None = None,
) -> AnnotationPackage:
    """Create a complete, immutable-input package beside the SpatialData store.

    Image preparation measures both percentile stages on original inference
    halos. Only per-crop arrays and GUI images are retained in the package.
    """
    source = Path(source).resolve()
    context.validate_store(source, context.platform)
    boxes = validate_crop_rectangles(context, rectangles_yx)
    if not boxes:
        raise ValueError("Select at least one crop.")
    sdata = sd.read_zarr(source, selection=("images", "labels"))
    native = ensure_cyx(get_scale0_dataarray(sdata.images[context.image_key]))
    if tuple(native.shape[-2:]) != (context.height, context.width):
        raise ValueError("Source morphology dimensions differ from the crop setup.")
    channels = [str(value) for value in native.coords["c"].values]
    providers = {}
    density_record = None
    for channel in settings.channels:
        if channel == "transcript_density":
            if density_path is None:
                raise ValueError(
                    "Choose the existing matching transcript density raster."
                )
            density_record = validate_local_density(density_path, context, settings)
            density_store = sd.read_zarr(density_path, selection=("images",))
            density = ensure_cyx(
                get_scale0_dataarray(density_store.images["atera_transcript_density"])
            )
            if tuple(density.shape) != (1, context.height, context.width):
                raise ValueError("Density image dimensions differ from the crop setup.")
            providers[channel] = density.data[0]
        elif channel in channels:
            providers[channel] = native.data[channels.index(channel)]
        else:
            raise ValueError(f"Source morphology is missing model channel: {channel}")
    mask_key = next(
        (
            key
            for key in ("cellpose_cell_labels", "MOSAIK_cellpose_labels")
            if key in sdata.labels
        ),
        None,
    )
    if mask_key is None:
        raise ValueError("This store has no Cellpose masks for prefilling the crops.")
    masks = get_scale0_dataarray(sdata.labels[mask_key]).transpose("y", "x")
    if tuple(masks.shape) != (context.height, context.width):
        raise ValueError("Cellpose masks differ from the native morphology grid.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package_id = f"hitl-{stamp}-{uuid.uuid4().hex[:8]}"
    output = source.parent / package_id
    staging = source.parent / f".{package_id}.preparing"
    staging.mkdir()
    try:
        for name in ("raw", "model_input", "context", "masks", "receipts"):
            (staging / name).mkdir()
        (staging / "annotation-guide.md").write_text(_GUIDE)
        (staging / "roi_context.json").write_text(
            json.dumps(asdict(context), indent=2) + "\n"
        )
        save_crop_regions(staging / "selected_regions.geojson", context, rectangles_yx)
        tiles = [_tile_for(box, context) for box in boxes]
        model_arrays = [
            np.zeros((3, box.height, box.width), np.float32) for box in boxes
        ]
        raw_arrays = [
            np.zeros((len(settings.channels), box.height, box.width), np.float32)
            for box in boxes
        ]
        normalization = [{} for box in boxes]
        tile_keys = sorted({(tile["core_y0"], tile["core_x0"]) for tile in tiles})
        for tile_key in tile_keys:
            members = [
                index
                for index, tile in enumerate(tiles)
                if (tile["core_y0"], tile["core_x0"]) == tile_key
            ]
            tile = tiles[members[0]]
            ys = slice(tile["tile_y0"], tile["tile_y1"])
            xs = slice(tile["tile_x0"], tile["tile_x1"])
            joint = None
            cached_planes = None
            if settings.input_normalization == "joint":
                cached_planes = [
                    _read(providers[channel][ys, xs]).astype(np.float32)
                    for channel in settings.channels
                ]
                joint = tuple(np.percentile(np.stack(cached_planes), (2, 98)))
            records, image_bounds = [], []
            for channel_index, channel in enumerate(settings.channels):
                plane = (
                    cached_planes[channel_index]
                    if cached_planes is not None
                    else _read(providers[channel][ys, xs]).astype(np.float32)
                )
                prepared = normalize_model_plane(plane, image_bounds=joint)
                records.append(
                    {
                        "low": prepared.low,
                        "high": prepared.high,
                        "scale": prepared.high - prepared.low,
                        "bounds_dtype": prepared.bounds_dtype,
                        "stride": list(prepared.stride),
                    }
                )
                image_bounds.append([prepared.image_low, prepared.image_high])
                for index in members:
                    box = boxes[index]
                    y = box.y0 - tile["tile_y0"]
                    x = box.x0 - tile["tile_x0"]
                    raw_arrays[index][channel_index] = plane[
                        y : y + box.height, x : x + box.width
                    ]
                    model_arrays[index][channel_index] = prepared.values[
                        y : y + box.height, x : x + box.width
                    ]
                del plane, prepared
            if len(settings.channels) == 1:
                for index in members:
                    model_arrays[index][1:] = model_arrays[index][0]
                records = records * 3
            elif len(settings.channels) == 2:
                records.append(
                    {
                        "low": 0.0,
                        "high": 0.0,
                        "scale": 0.0,
                        "bounds_dtype": "float32",
                        "stride": records[0]["stride"],
                    }
                )
            for index in members:
                normalization[index] = {
                    "route": "A",
                    "batch_axis": True,
                    "percentile": [1, 99],
                    "input_normalization": settings.input_normalization,
                    "image_p2_p98": image_bounds,
                    "channels": records,
                }
            del cached_planes
        rows = []
        for index, box in enumerate(boxes):
            crop_id = f"crop_{index + 1:03d}"
            crop_masks = _read(
                masks.data[box.y0 : box.y0 + box.height, box.x0 : box.x0 + box.width]
            )
            if crop_masks.dtype.kind not in "iu" or np.any(crop_masks < 0):
                raise ValueError(
                    "Prefills must contain nonnegative integer mask labels."
                )
            crop_masks = label(crop_masks, background=0, connectivity=2).astype(
                np.uint32
            )
            padded = np.pad(crop_masks, 1)
            boundary = np.zeros(crop_masks.shape, dtype=bool)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                boundary |= (
                    crop_masks
                    != padded[1 + dy : 1 + dy + box.height, 1 + dx : 1 + dx + box.width]
                )
            outlines = crop_masks * boundary
            # The GUI recomputes outlines when loading; the explicit prefill is
            # only a display convenience, never a training target after review.
            tifffile.imwrite(
                staging / f"{crop_id}.tif",
                raw_arrays[index],
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )
            np.save(
                staging / f"{crop_id}_seg.npy",
                {
                    "masks": crop_masks,
                    "outlines": outlines,
                    "filename": f"{crop_id}.tif",
                    "ismanual": np.zeros(int(crop_masks.max()), bool),
                },
            )
            np.save(staging / "raw" / f"{crop_id}.npy", raw_arrays[index])
            np.save(staging / "model_input" / f"{crop_id}.npy", model_arrays[index])
            files = [
                f"{crop_id}.tif",
                f"{crop_id}_seg.npy",
                f"raw/{crop_id}.npy",
                f"model_input/{crop_id}.npy",
            ]
            rows.append(
                {
                    "crop_id": crop_id,
                    "roi_id": f"roi_{index + 1:03d}",
                    "split": "pilot_train",
                    "stratum": "viewer_selected",
                    "bbox_y0_y1_x0_x1": [
                        box.y0,
                        box.y0 + box.height,
                        box.x0,
                        box.x0 + box.width,
                    ],
                    "tile": tiles[index],
                    "normalization": normalization[index],
                    "prefill_cells": int(crop_masks.max()),
                    "files_sha256": {
                        name: file_sha256(staging / name) for name in files
                    },
                }
            )
        manifest = {
            "schema_version": 1,
            "managed_annotation_version": 1,
            "package_id": package_id,
            "guide_version": "pilot-v2-managed-v1",
            "platform": context.platform,
            "image_key": context.image_key,
            "source_metadata_sha256": context.source_metadata_sha256,
            "baseline_stats_sha256": context.baseline_stats_sha256,
            "roi_context": asdict(context),
            "pixel_size_um": context.pixel_size_um,
            "raw_channels": list(settings.channels),
            "model_channels": list(settings.channels),
            "view_channels": list(settings.channels),
            "input_normalization": settings.input_normalization,
            "transcript_density": settings.density
            if density_record is not None
            else None,
            "baseline_stats_model_identity": settings.baseline_identity,
            "training_normalize": False,
            "evaluation_eligible": False,
            "transfer_relative_path": f"annotations/{package_id}/",
            "crops": rows,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "training-template.json").write_text(
            json.dumps(
                {
                    "hitl_root": None,
                    "annotation_package_ids": [package_id],
                    "model_name": None,
                    "platform": context.platform,
                    "species": None,
                    "n_epochs": 100,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.1,
                    "batch_size": 1,
                    "seed": 0,
                },
                indent=2,
            )
            + "\n"
        )
        with (staging / "review.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["crop_id", "status", "reason", "minutes", "notes"])
            writer.writerows([[row["crop_id"], "pending", "", "", ""] for row in rows])
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return open_annotation_package(output)
