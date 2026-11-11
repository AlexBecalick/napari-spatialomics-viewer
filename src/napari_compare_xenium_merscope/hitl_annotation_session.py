"""Persisted annotation progress and validation without unpickling GUI outputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    """Hash a package artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_file(root: Path, relative: str) -> Path:
    """Resolve a relative artifact, refusing traversal and external symlinks."""
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Annotation artifact paths must stay inside the package.")
    result = root / relative_path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Annotation artifact points outside its package.")
    return result


@dataclass(frozen=True)
class AnnotationCrop:
    """One immutable training crop and its mutable annotation artifacts."""

    crop_id: str
    bounds: tuple[int, int, int, int]
    files_sha256: dict[str, str]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native height and width."""
        y0, y1, x0, x1 = self.bounds
        return y1 - y0, x1 - x0


@dataclass(frozen=True)
class AnnotationPackage:
    """An exported package with a stable manifest and per-crop save receipts."""

    root: Path
    package_id: str
    manifest_sha256: str
    manifest: dict[str, Any]
    crops: tuple[AnnotationCrop, ...]

    @property
    def transfer_relative_path(self) -> str:
        """Return the destination below the configured MerXen HITL root."""
        return f"annotations/{self.package_id}/"


def open_annotation_package(root: Path) -> AnnotationPackage:
    """Read a managed package and validate its IDs and immutable input hashes."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("managed_annotation_version") != 1:
        raise ValueError(
            "Choose a package created by Save areas and prepare annotation."
        )
    package_id = manifest.get("package_id", "")
    if not re.fullmatch(r"hitl-[A-Za-z0-9_-]{1,80}", package_id):
        raise ValueError("Invalid annotation package ID.")
    crops = []
    seen = set()
    for row in manifest["crops"]:
        crop_id = row["crop_id"]
        if not re.fullmatch(r"crop_[0-9]{3,6}", crop_id) or crop_id in seen:
            raise ValueError("Crop IDs must be safe and unique.")
        seen.add(crop_id)
        bounds = tuple(row["bbox_y0_y1_x0_x1"])
        if len(bounds) != 4 or any(type(value) is not int for value in bounds):
            raise ValueError("Invalid native crop bounds.")
        crop = AnnotationCrop(crop_id, bounds, row["files_sha256"])
        if min(crop.shape) < 256 or min(bounds[0], bounds[2]) < 0:
            raise ValueError("Invalid annotation crop dimensions.")
        required = (
            f"{crop_id}.tif",
            f"raw/{crop_id}.npy",
            f"model_input/{crop_id}.npy",
        )
        for relative in required:
            expected = crop.files_sha256.get(relative)
            if not expected or file_sha256(package_file(root, relative)) != expected:
                raise ValueError(f"Immutable annotation input changed: {relative}")
        crops.append(crop)
    if not crops:
        raise ValueError("Annotation package contains no crops.")
    return AnnotationPackage(
        root, package_id, file_sha256(manifest_path), manifest, tuple(crops)
    )


@dataclass(frozen=True)
class SavedAnnotation:
    """A verified numeric mask saved by the managed Cellpose GUI session."""

    crop_id: str
    cell_count: int
    masks_path: Path
    receipt_path: Path


def validate_saved_annotation(
    package: AnnotationPackage,
    crop: AnnotationCrop,
    *,
    session_nonce: str | None = None,
) -> SavedAnnotation:
    """Check the save receipt, image identity, GUI output and numeric mask."""
    root, crop_id = package.root, crop.crop_id
    receipt_path = package_file(root, f"receipts/{crop_id}.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("package_id") != package.package_id
        or receipt.get("manifest_sha256") != package.manifest_sha256
        or receipt.get("crop_id") != crop_id
        or (session_nonce is not None and receipt.get("session_nonce") != session_nonce)
    ):
        raise ValueError("No verified save from this crop's Cellpose session.")
    if receipt.get("image_sha256") != crop.files_sha256[f"{crop_id}.tif"]:
        raise ValueError("Saved masks refer to a different crop image.")
    for relative, field in (
        (f"{crop_id}.tif", "image_sha256"),
        (f"{crop_id}_seg.npy", "seg_sha256"),
        (f"masks/{crop_id}.npz", "masks_sha256"),
    ):
        if file_sha256(package_file(root, relative)) != receipt.get(field):
            raise ValueError(
                f"Saved annotation changed after its save receipt: {relative}"
            )
    masks_path = package_file(root, f"masks/{crop_id}.npz")
    with np.load(masks_path, allow_pickle=False) as arrays:
        if arrays.files != ["masks"]:
            raise ValueError("Numeric annotation must contain only masks.")
        masks = arrays["masks"]
    if masks.shape != crop.shape or masks.dtype.kind not in "iu" or np.any(masks < 0):
        raise ValueError(
            "Saved masks must be a native-size, nonnegative integer image."
        )
    count = int(np.count_nonzero(np.unique(masks)))
    from skimage.measure import label

    if (
        int(masks.max()) > masks.size
        or int(label(masks, background=0, connectivity=2).max()) != count
    ):
        raise ValueError(
            "Each cell must have one connected mask. "
            "Reopen and fix disconnected pieces."
        )
    if count != receipt.get("cell_count"):
        raise ValueError("Saved mask count differs from its receipt.")
    return SavedAnnotation(crop_id, count, masks_path, receipt_path)


def annotation_progress(package: AnnotationPackage) -> dict[str, SavedAnnotation]:
    """Return verified saved crops; absent or invalid saves remain unfinished."""
    result = {}
    for crop in package.crops:
        try:
            result[crop.crop_id] = validate_saved_annotation(package, crop)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result
