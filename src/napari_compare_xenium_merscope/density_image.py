"""Portable discovery of MerXen's embedded native transcript-density image."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DENSITY_ATTR = "merxen_transcript_density"
DENSITY_IMAGE_KEY = "atera_transcript_density"


def embedded_density_record(path: Path) -> dict[str, Any] | None:
    """Read signed embedded provenance without scanning chunks or transcripts.

    Copied stores may have different timestamp precision on the annotation
    computer. Exported training arrays receive their own content hashes.
    """
    if (path / "zarr.json").is_file():
        attrs = json.loads((path / "zarr.json").read_text()).get("attributes", {})
    elif (path / ".zattrs").is_file():
        attrs = json.loads((path / ".zattrs").read_text())
    else:
        return None
    binding = attrs.get(DENSITY_ATTR)
    if binding is None:
        return None
    signed = {key: value for key, value in binding.items() if key != "binding_sha256"}
    digest = hashlib.sha256(json.dumps(signed, sort_keys=True).encode()).hexdigest()
    if (
        binding.get("schema_version") != 1
        or binding.get("image_key") not in {DENSITY_IMAGE_KEY, "xenium_transcript_density"}
        or binding.get("binding_sha256") != digest
        or not (path / "images" / binding["image_key"]).is_dir()
    ):
        raise ValueError("Embedded transcript density is incomplete or inconsistent.")
    record: dict[str, Any] = binding["record"]
    return record
