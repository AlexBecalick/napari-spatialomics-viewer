"""Platform registry: one source of truth for the platforms this viewer opens.

Standard library only (no numpy, zarr or spatialdata), so it is cheap to import
from the launcher, the CLI and the tests.

* :class:`PlatformSpec` describes one platform: sidecar files, fallback pixel
  size, original mask keys, image-channel display hints and capability flags.
* :data:`PLATFORM_SPECS` lists them in a stable order (MERSCOPE, XENIUM, ATERA).
  :data:`PAIRED_PLATFORMS` is the subset paired mode supports.
* :func:`detect_store_platform` reads a store's platform from JSON metadata
  only. ``merxen_schema.platform`` in the root ``zarr.json`` and the
  ``platform`` of ``merxen_native_latest.json`` are authoritative; element
  names and sidecars inside the store are heuristic hints only.
* :func:`find_spec_path` finds the pixel-size sidecar in or beside a store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MERSCOPE = "MERSCOPE"
XENIUM = "XENIUM"
ATERA = "ATERA"

PIXEL_TRANSFORM_MERSCOPE_AFFINE = "merscope_affine"
PIXEL_TRANSFORM_SPEC_FILE = "spec_file"

GENE_PANEL_MODE_FULL = "full"
GENE_PANEL_MODE_SELECTION = "selection"
#: A panel with more features than this opens the gene inspector in selection
#: mode (empty by default, search first) whatever its platform.
GENE_SELECTION_MODE_FEATURE_THRESHOLD = 2000
#: Maximum number of genes selection mode shows at once (one colour slot each).
GENE_SELECTION_SLOT_CAP = 32

MERXEN_NATIVE_LATEST_JSON = "merxen_native_latest.json"
ATERA_ROOT_ATTR = "merxen_atera"


class UnknownPlatformError(ValueError):
    """Raised for a platform name the registry does not know."""


@dataclass(frozen=True)
class PlatformSpec:
    """Static description of one platform.

    ``image_channel_aliases`` maps ``(image_key, channel)`` to the display label
    used in layer names; ``image_channel_colormaps`` maps an exact channel name
    (case-insensitive) to a napari colormap. Both are tuples of pairs so the
    spec stays immutable.
    """

    name: str
    display_name: str
    pixel_transform: str
    spec_sidecar_names: tuple[str, ...]
    fallback_um_per_px: float
    original_cell_shape_key: str | None
    original_nucleus_shape_key: str | None = None
    nucleus_cell_join_key: str | None = None
    vendor_label_aliases: tuple[tuple[str, str], ...] = ()
    startup_shape_keys: tuple[str, ...] = ()
    inspector_shape_preference: tuple[str, ...] = ("proseg", "cellpose", "cell")
    image_channel_aliases: tuple[tuple[tuple[str, str], str], ...] = ()
    image_channel_colormaps: tuple[tuple[str, str], ...] = ()
    paired_role: str | None = None
    assignment_instance_key_fallback: bool = False
    supports_pciseq: bool = True
    supports_cortical_depth: bool = True
    supports_object_annotations: bool = True
    lazy_transcripts: bool = False
    gene_panel_mode: str = GENE_PANEL_MODE_FULL
    marker_shape_keys: tuple[str, ...] = ()

    def channel_alias(self, image_key: str, channel: str) -> str | None:
        """Return the display label for ``channel`` of ``image_key``, if any."""
        for (key, chan), alias in self.image_channel_aliases:
            if key == str(image_key) and chan == str(channel):
                return alias
        return None

    def channel_colormap(self, channel: str) -> str | None:
        """Return the explicit colormap for ``channel`` (case-insensitive), if any."""
        wanted = str(channel).strip().lower()
        for name, colormap in self.image_channel_colormaps:
            if name.lower() == wanted:
                return colormap
        return None

    @property
    def is_paired(self) -> bool:
        return self.paired_role is not None


PLATFORM_SPECS: tuple[PlatformSpec, ...] = (
    PlatformSpec(
        name=MERSCOPE,
        display_name="MERSCOPE",
        pixel_transform=PIXEL_TRANSFORM_MERSCOPE_AFFINE,
        spec_sidecar_names=("micron_to_mosaic_pixel_transform.csv",),
        fallback_um_per_px=0.108,
        original_cell_shape_key="merscope_cell_boundaries",
        paired_role="moving",
        marker_shape_keys=("merscope_cell_boundaries",),
    ),
    PlatformSpec(
        name=XENIUM,
        display_name="Xenium",
        pixel_transform=PIXEL_TRANSFORM_SPEC_FILE,
        spec_sidecar_names=("experiment.xenium",),
        fallback_um_per_px=0.2125,
        original_cell_shape_key="xenium_cell_boundaries",
        original_nucleus_shape_key="xenium_nucleus",
        image_channel_aliases=(
            (("morphology_focus", "DAPI"), "DAPI Xenium"),
            (("post_xenium_if", "DAPI"), "DAPI Post Xenium-IF"),
        ),
        paired_role="fixed",
        marker_shape_keys=("xenium_cell_boundaries",),
    ),
    PlatformSpec(
        name=ATERA,
        display_name="Atera",
        pixel_transform=PIXEL_TRANSFORM_SPEC_FILE,
        spec_sidecar_names=(
            "atera/experiment_override.spatial",
            "atera/experiment.spatial",
            "experiment.spatial",
        ),
        fallback_um_per_px=0.21249222,
        original_cell_shape_key="atera_cell_boundaries",
        original_nucleus_shape_key="atera_nucleus",
        nucleus_cell_join_key="cell_instance_id",
        vendor_label_aliases=(
            ("atera_cell_boundaries_labels", "cell_labels"),
            ("atera_nucleus_labels", "nucleus_labels"),
        ),
        startup_shape_keys=("atera_cell_boundaries",),
        inspector_shape_preference=("proseg", "cellpose", "atera_cell_boundaries", "cell"),
        image_channel_colormaps=(
            ("DAPI", "blue"),
            ("18S", "green"),
            ("ATP1A1/CD45/E-Cadherin", "magenta"),
            ("alphaSMA/Vimentin", "yellow"),
        ),
        paired_role=None,
        assignment_instance_key_fallback=True,
        supports_pciseq=False,
        supports_cortical_depth=False,
        supports_object_annotations=False,
        lazy_transcripts=True,
        gene_panel_mode=GENE_PANEL_MODE_SELECTION,
        marker_shape_keys=("atera_cell_boundaries",),
    ),
)

_SPECS_BY_NAME: dict[str, PlatformSpec] = {spec.name: spec for spec in PLATFORM_SPECS}

#: Platforms paired mode supports (MERSCOPE moving onto XENIUM fixed).
PAIRED_PLATFORMS: tuple[str, ...] = tuple(spec.name for spec in PLATFORM_SPECS if spec.is_paired)


def supported_platforms() -> tuple[str, ...]:
    """Every platform name, in registry order."""
    return tuple(spec.name for spec in PLATFORM_SPECS)


def standalone_platforms() -> tuple[str, ...]:
    """Platforms that open on their own (all of them)."""
    return supported_platforms()


def gene_panel_mode_for(platform: object, feature_count: int) -> str:
    """Gene-inspector mode for a panel of ``feature_count`` features on ``platform``.

    The platform's own mode wins when it asks for selection mode (ATERA); any
    other panel switches to selection mode above
    :data:`GENE_SELECTION_MODE_FEATURE_THRESHOLD` features, where listing and
    drawing every gene at once stops being useful.
    """
    spec = find_platform_spec(platform)
    if spec is not None and spec.gene_panel_mode == GENE_PANEL_MODE_SELECTION:
        return GENE_PANEL_MODE_SELECTION
    if int(feature_count) > GENE_SELECTION_MODE_FEATURE_THRESHOLD:
        return GENE_PANEL_MODE_SELECTION
    return GENE_PANEL_MODE_FULL


def find_platform_spec(name: object) -> PlatformSpec | None:
    """Return the spec for ``name`` (case-insensitive), or ``None`` if unknown."""
    if name is None:
        return None
    return _SPECS_BY_NAME.get(str(name).strip().upper())


def platform_spec(name: object) -> PlatformSpec:
    """Return the spec for ``name`` (case-insensitive); raise for unknown names."""
    spec = find_platform_spec(name)
    if spec is None:
        raise UnknownPlatformError(
            f"Unknown platform {name!r}; expected one of {', '.join(supported_platforms())}"
        )
    return spec


def find_spec_path(zarr_path: str | Path, platform: str) -> Path | None:
    """Return the first existing pixel-size sidecar for ``platform``.

    Looks inside ``zarr_path`` first, then in its parent directory, trying the
    platform's ``spec_sidecar_names`` in order within each directory (so an
    ATERA ``atera/experiment_override.spatial`` beats ``atera/experiment.spatial``).
    """
    spec = platform_spec(platform)
    root = Path(zarr_path)
    for base in (root, root.parent):
        for name in spec.spec_sidecar_names:
            candidate = base / name
            if candidate.is_file():
                return candidate
    return None


@dataclass(frozen=True)
class StoreDetection:
    """Result of :func:`detect_store_platform`.

    ``platform`` is a canonical registry name or ``None``. ``authoritative`` is
    true only when it came from ``merxen_schema.platform`` or
    ``merxen_native_latest.json``. ``evidence`` lists what was seen, in the
    order it was considered.
    """

    platform: str | None
    authoritative: bool
    evidence: tuple[str, ...] = ()


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _root_attributes(root: Path) -> dict:
    v3 = _read_json(root / "zarr.json")
    if v3 is not None and isinstance(v3.get("attributes"), dict):
        return v3["attributes"]
    v2 = _read_json(root / ".zattrs")
    return v2 if v2 is not None else {}


def _consolidated_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    v3 = _read_json(root / "zarr.json")
    consolidated = v3.get("consolidated_metadata") if v3 is not None else None
    metadata = consolidated.get("metadata") if isinstance(consolidated, dict) else None
    if isinstance(metadata, dict):
        keys.update(str(key) for key in metadata)
    v2 = _read_json(root / ".zmetadata")
    metadata = v2.get("metadata") if v2 is not None else None
    if isinstance(metadata, dict):
        keys.update(str(key).rsplit("/.z", 1)[0] for key in metadata)
    return keys


def _has_element(root: Path, group: str, key: str, consolidated: set[str]) -> bool:
    return (root / group / key).is_dir() or f"{group}/{key}" in consolidated


def detect_store_platform(zarr_path: str | Path) -> StoreDetection:
    """Detect a store's platform from JSON metadata alone (never opens data).

    Authoritative, in order: root ``zarr.json`` ``attributes.merxen_schema.platform``,
    then ``merxen_native_latest.json`` ``platform``; the first wins on conflict.
    Heuristic, used only when nothing authoritative names a known platform, and
    only from inside the store (never its parent): the ``merxen_atera`` root
    attribute, ``atera/experiment.spatial``, ``experiment.xenium``,
    ``micron_to_mosaic_pixel_transform.csv`` and the platforms' original
    ``shapes/*_cell_boundaries`` elements. Heuristics naming more than one
    platform give ``None``. ``merxen_alignment*`` attributes are ignored: paired
    MERSCOPE and XENIUM stores both carry them.
    """
    root = Path(zarr_path)
    evidence: list[str] = []
    if not root.is_dir():
        return StoreDetection(None, False, (f"{root} is not a directory",))

    attrs = _root_attributes(root)
    schema = attrs.get("merxen_schema")
    if isinstance(schema, dict) and schema.get("platform") is not None:
        raw = schema.get("platform")
        spec = find_platform_spec(raw)
        if spec is not None:
            evidence.append(f"zarr.json merxen_schema.platform={raw}")
        else:
            evidence.append(f"zarr.json merxen_schema.platform={raw!r} is not a known platform")
    else:
        spec = None

    native = _read_json(root / MERXEN_NATIVE_LATEST_JSON)
    native_spec = None
    if native is not None and native.get("platform") is not None:
        raw = native.get("platform")
        native_spec = find_platform_spec(raw)
        if native_spec is not None:
            evidence.append(f"{MERXEN_NATIVE_LATEST_JSON} platform={raw}")
        else:
            evidence.append(f"{MERXEN_NATIVE_LATEST_JSON} platform={raw!r} is not a known platform")

    if spec is not None:
        if native_spec is not None and native_spec.name != spec.name:
            evidence.append(
                f"conflict: merxen_schema says {spec.name}, {MERXEN_NATIVE_LATEST_JSON} says "
                f"{native_spec.name}; merxen_schema wins"
            )
        return StoreDetection(spec.name, True, tuple(evidence))
    if native_spec is not None:
        return StoreDetection(native_spec.name, True, tuple(evidence))

    consolidated = _consolidated_keys(root)
    hints: dict[str, list[str]] = {}

    def hint(platform: str, text: str) -> None:
        hints.setdefault(platform, []).append(text)

    if ATERA_ROOT_ATTR in attrs:
        hint(ATERA, f"root attribute {ATERA_ROOT_ATTR}")
    if (root / "atera" / "experiment.spatial").is_file():
        hint(ATERA, "atera/experiment.spatial")
    if (root / "experiment.xenium").is_file():
        hint(XENIUM, "experiment.xenium")
    if (root / "micron_to_mosaic_pixel_transform.csv").is_file():
        hint(MERSCOPE, "micron_to_mosaic_pixel_transform.csv")
    for platform_spec_entry in PLATFORM_SPECS:
        for key in platform_spec_entry.marker_shape_keys:
            if _has_element(root, "shapes", key, consolidated):
                hint(platform_spec_entry.name, f"shapes/{key}")

    for platform, texts in hints.items():
        evidence.extend(f"heuristic {platform}: {text}" for text in texts)
    if len(hints) == 1:
        return StoreDetection(next(iter(hints)), False, tuple(evidence))
    if len(hints) > 1:
        evidence.append("heuristics disagree: " + ", ".join(sorted(hints)))
    return StoreDetection(None, False, tuple(evidence))
