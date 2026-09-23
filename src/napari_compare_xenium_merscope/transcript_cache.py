"""Persistent, external cache for prepared transcript point stores.

Transcript preparation streams the complete points table twice and can dominate
the time needed to reopen a dataset.  This module stores the resulting compact
``GenePointStore`` outside the SpatialData zarr.  Cache entries contain JSON and
``.npy`` files only (never pickle), and coordinate/index arrays are restored as
read-only memory maps.

Entries are content addressed in two stages.  The first digest describes the
local points tree and every option that affects transcript preparation.  The
second describes the marker-reference resolution used to assign colours and
symbols.  Consequently a changed points file, build option, or marker reference
is a cache miss rather than a potentially incorrect hit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import blake2s
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from .utils import (
    CellTranscriptIndex,
    GenePointStore,
    GeneVisual,
    MarkerReferenceResolution,
    TranscriptBuildCancelled,
    resolve_cell_type_marker_reference,
)


log = logging.getLogger(__name__)

TRANSCRIPT_CACHE_VERSION = 1
TRANSCRIPT_CACHE_ENV = "NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR"
_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_CACHE_IO_CHUNK_BYTES = 64 * 1024 * 1024
_METADATA_NAMES = {
    ".zarray",
    ".zattrs",
    ".zgroup",
    ".zmetadata",
    "zarr.json",
    "_common_metadata",
    "_metadata",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return blake2s(_canonical_json(value), digest_size=20).hexdigest()


def _cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise TranscriptBuildCancelled("Transcript cache operation cancelled")


def default_transcript_cache_dir() -> Path:
    """Return the platform-appropriate external transcript cache directory."""
    configured = os.environ.get(TRANSCRIPT_CACHE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "napari-compare-xenium-merscope" / "transcripts"


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _metadata_content_digest(path: Path) -> str | None:
    name = path.name
    if name not in _METADATA_NAMES and path.suffix.lower() != ".json":
        return None
    hasher = blake2s(digest_size=16)
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _point_tree_fingerprint(points_path: Path) -> list[dict[str, Any]] | None:
    """Fingerprint local point files without reading large parquet payloads."""
    try:
        files = sorted(
            (path for path in points_path.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(points_path).as_posix(),
        )
    except OSError:
        return None
    if not files:
        return None

    entries: list[dict[str, Any]] = []
    try:
        for path in files:
            stat = path.stat()
            entry: dict[str, Any] = {
                "path": path.relative_to(points_path).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
            }
            content_digest = _metadata_content_digest(path)
            if content_digest is not None:
                entry["content"] = content_digest
            entries.append(entry)
    except (OSError, ValueError):
        return None
    return entries


@dataclass(frozen=True)
class TranscriptCacheRequest:
    """Immutable identity of one transcript preparation request."""

    dataset_path: Path
    points_path: Path
    points_key: str
    x_col: str
    y_col: str
    gene_col: str
    assignment_col: str | None
    background_col: str | None
    max_points: int | None
    random_state: int
    build_cell_index: bool
    cache_root: Path
    family_digest: str
    digest: str

    @classmethod
    def create(
        cls,
        dataset_path: str | Path,
        points_key: str,
        x_col: str,
        y_col: str,
        gene_col: str,
        assignment_col: str | None = None,
        background_col: str | None = None,
        max_points: int | None = None,
        random_state: int = 42,
        build_cell_index: bool = False,
        cache_root: str | Path | None = None,
    ) -> "TranscriptCacheRequest | None":
        """Create a request for a readable local SpatialData points element.

        ``None`` disables caching for remote/missing sources and whenever the
        requested cache directory is inside the source dataset.
        """
        try:
            dataset = Path(dataset_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError):
            return None
        if not dataset.is_dir():
            return None

        key = str(points_key)
        key_path = Path(key)
        if not key or key_path.is_absolute() or ".." in key_path.parts:
            return None
        points = (dataset / "points" / key_path).resolve()
        if not _is_relative_to(points, dataset) or not points.is_dir():
            return None

        root = (
            default_transcript_cache_dir()
            if cache_root is None
            else Path(cache_root).expanduser().resolve()
        )
        if root == dataset or _is_relative_to(root, dataset):
            log.warning(
                "Transcript cache directory %s is inside dataset %s; cache disabled",
                root,
                dataset,
            )
            return None

        tree = _point_tree_fingerprint(points)
        if tree is None:
            return None
        cap = None if max_points is None or int(max_points) <= 0 else int(max_points)
        family_identity = {
            "version": TRANSCRIPT_CACHE_VERSION,
            "dataset_path": str(dataset),
            "points_key": key,
            "columns": {
                "x": str(x_col),
                "y": str(y_col),
                "gene": str(gene_col),
                "assignment": None if assignment_col is None else str(assignment_col),
                "background": None if background_col is None else str(background_col),
            },
            "options": {
                "max_points": cap,
                "random_state": int(random_state),
                "build_cell_index": bool(build_cell_index),
            },
        }
        identity = {**family_identity, "point_tree": tree}
        return cls(
            dataset_path=dataset,
            points_path=points,
            points_key=key,
            x_col=str(x_col),
            y_col=str(y_col),
            gene_col=str(gene_col),
            assignment_col=None if assignment_col is None else str(assignment_col),
            background_col=None if background_col is None else str(background_col),
            max_points=cap,
            random_state=int(random_state),
            build_cell_index=bool(build_cell_index),
            cache_root=root,
            family_digest=_digest(family_identity),
            digest=_digest(identity),
        )

    @property
    def entry_root(self) -> Path:
        return self.cache_root / f"v{TRANSCRIPT_CACHE_VERSION}" / self.digest


@dataclass(frozen=True)
class TranscriptCacheHit:
    """Successfully restored transcript payload and its cache location."""

    store: GenePointStore
    reference_resolution: MarkerReferenceResolution
    path: Path


def _resolution_payload(value: Any) -> dict[str, Any]:
    if value is None:
        value = MarkerReferenceResolution(reference=None)
    if isinstance(value, MarkerReferenceResolution):
        raw = asdict(value)
    elif is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        # A plain reference mapping is a useful lightweight resolver result.
        raw = asdict(MarkerReferenceResolution(reference=dict(value)))
    else:
        raw = {
            name: getattr(value, name, default)
            for name, default in (
                ("reference", None),
                ("source", ""),
                ("panel_fingerprint", ""),
                ("panel_gene_count", 0),
                ("matched_gene_count", 0),
                ("broad_gene_count", 0),
                ("fine_gene_count", 0),
                ("warnings", ()),
            )
        }
    reference = raw.get("reference")
    normalized_reference = None
    if isinstance(reference, Mapping):
        normalized_reference = {
            str(gene): {
                str(key): str(item)
                for key, item in sorted(dict(info).items())
            }
            for gene, info in sorted(reference.items(), key=lambda item: str(item[0]))
            if isinstance(info, Mapping)
        }
    return {
        "reference": normalized_reference,
        "source": str(raw.get("source", "")),
        "panel_fingerprint": str(raw.get("panel_fingerprint", "")),
        "panel_gene_count": int(raw.get("panel_gene_count", 0)),
        "matched_gene_count": int(raw.get("matched_gene_count", 0)),
        "broad_gene_count": int(raw.get("broad_gene_count", 0)),
        "fine_gene_count": int(raw.get("fine_gene_count", 0)),
        "warnings": [str(item) for item in raw.get("warnings", ())],
    }


def _resolution_from_payload(raw: Mapping[str, Any]) -> MarkerReferenceResolution:
    reference = raw.get("reference")
    if reference is not None and not isinstance(reference, dict):
        raise ValueError("invalid reference")
    warnings = raw.get("warnings", [])
    if not isinstance(warnings, list):
        raise ValueError("invalid reference warnings")
    return MarkerReferenceResolution(
        reference=reference,
        source=str(raw.get("source", "")),
        panel_fingerprint=str(raw.get("panel_fingerprint", "")),
        panel_gene_count=int(raw.get("panel_gene_count", 0)),
        matched_gene_count=int(raw.get("matched_gene_count", 0)),
        broad_gene_count=int(raw.get("broad_gene_count", 0)),
        fine_gene_count=int(raw.get("fine_gene_count", 0)),
        warnings=tuple(str(item) for item in warnings),
    )


def _visual_payload(visual: GeneVisual) -> dict[str, Any]:
    rgba = [float(value) for value in visual.rgba]
    if len(rgba) != 4 or not np.isfinite(rgba).all():
        raise ValueError("gene visual must have a finite RGBA quadruple")
    return {"rgba": rgba, "symbol": str(visual.symbol)}


def _safe_member(entry: Path, name: str) -> Path:
    candidate = (entry / str(name)).resolve()
    if not _is_relative_to(candidate, entry.resolve()):
        raise ValueError("cache member escapes its entry")
    return candidate


def _load_array(entry: Path, name: str, *, ndim: int, dtype: np.dtype | None = None) -> np.ndarray:
    path = _safe_member(entry, name)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != ndim or (dtype is not None and array.dtype != np.dtype(dtype)):
        raise ValueError(f"invalid array {name}")
    return array


def _chunk_rows(array: np.ndarray) -> int:
    trailing = int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1
    row_bytes = max(1, trailing * int(array.dtype.itemsize))
    return max(1, _CACHE_IO_CHUNK_BYTES // row_bytes)


def _all_true(
    array: np.ndarray,
    cancel_check: Callable[[], bool] | None,
) -> bool:
    rows = _chunk_rows(array)
    for start in range(0, len(array), rows):
        _cancelled(cancel_check)
        if not bool(np.asarray(array[start : start + rows]).all()):
            return False
    return True


def _maximum_unsigned_code(
    array: np.ndarray,
    cancel_check: Callable[[], bool] | None,
) -> int:
    maximum = -1
    rows = _chunk_rows(array)
    for start in range(0, len(array), rows):
        _cancelled(cancel_check)
        chunk = np.asarray(array[start : start + rows])
        if len(chunk):
            maximum = max(maximum, int(chunk.max()))
    return maximum


def _read_manifest(entry: Path) -> dict[str, Any]:
    manifest_path = entry / _MANIFEST
    if not manifest_path.is_file() or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("missing or oversized transcript cache manifest")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid transcript cache manifest")
    return raw


def _restore_store(
    entry: Path,
    raw: Mapping[str, Any],
    cancel_check: Callable[[], bool] | None = None,
) -> GenePointStore:
    groups = raw.get("groups")
    genes = raw.get("genes")
    offsets_raw = raw.get("gene_offsets")
    counts_raw = raw.get("gene_counts")
    visuals_raw = raw.get("gene_visuals")
    if not all(isinstance(value, expected) for value, expected in (
        (groups, list), (genes, list), (offsets_raw, dict),
        (counts_raw, dict), (visuals_raw, dict),
    )):
        raise ValueError("invalid transcript store metadata")

    group_symbols: list[str] = []
    group_coords: list[np.ndarray] = []
    for item in groups:
        _cancelled(cancel_check)
        if not isinstance(item, dict):
            raise ValueError("invalid transcript group")
        coords = _load_array(entry, str(item.get("coords", "")), ndim=2, dtype=np.float32)
        if coords.shape[1:] != (2,) or coords.shape[0] != int(item.get("count", -1)):
            raise ValueError("invalid transcript group coordinates")
        group_symbols.append(str(item["symbol"]))
        group_coords.append(coords)

    gene_names = [str(value) for value in genes]
    if len(set(gene_names)) != len(gene_names):
        raise ValueError("duplicate cached genes")
    visuals: dict[str, GeneVisual] = {}
    for gene in gene_names:
        _cancelled(cancel_check)
        item = visuals_raw.get(gene)
        if not isinstance(item, dict):
            raise ValueError("missing cached gene visual")
        rgba = tuple(float(value) for value in item.get("rgba", ()))
        if len(rgba) != 4 or not np.isfinite(rgba).all():
            raise ValueError("invalid cached gene visual")
        visuals[gene] = GeneVisual(rgba=rgba, symbol=str(item.get("symbol", "")))

    offsets: dict[str, tuple[int, int, int, int]] = {}
    counts: dict[str, int] = {}
    group_colors: list[np.ndarray] = []
    filled: list[np.ndarray] = []
    for coords in group_coords:
        _cancelled(cancel_check)
        group_colors.append(np.empty((len(coords), 4), dtype=np.float32))
        filled.append(np.zeros(len(coords), dtype=bool))
    offset_genes = {str(gene) for gene in offsets_raw}
    count_genes = {str(gene) for gene in counts_raw}
    if offset_genes != count_genes or not offset_genes.issubset(gene_names):
        raise ValueError("cached gene ranges do not match the gene panel")
    # A uniformly sampled render can legitimately contain no points for a
    # low-abundance panel gene. Such genes remain in ``genes`` and
    # ``source_gene_counts`` for the inspector, but have no rendered offset.
    for gene in gene_names:
        _cancelled(cancel_check)
        if gene not in offset_genes:
            continue
        values = offsets_raw[gene]
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("invalid cached gene offsets")
        gi, fg_start, fg_end, bg_end = (int(value) for value in values)
        if not (0 <= gi < len(group_coords)):
            raise ValueError("cached gene group is out of range")
        if not (0 <= fg_start <= fg_end <= bg_end <= len(group_coords[gi])):
            raise ValueError("cached gene offsets are out of range")
        if visuals[gene].symbol != group_symbols[gi]:
            raise ValueError("cached gene visual symbol does not match its group")
        count = int(counts_raw.get(gene, -1))
        if count != bg_end - fg_start:
            raise ValueError("cached gene count does not match its offsets")
        rows = _chunk_rows(group_colors[gi])
        rgba = np.asarray(visuals[gene].rgba, dtype=np.float32)
        for start in range(fg_start, bg_end, rows):
            _cancelled(cancel_check)
            end = min(bg_end, start + rows)
            if filled[gi][start:end].any():
                raise ValueError("overlapping cached gene offsets")
            filled[gi][start:end] = True
            group_colors[gi][start:end] = rgba
        offsets[gene] = (gi, fg_start, fg_end, bg_end)
        counts[gene] = count
    for mask in filled:
        if not _all_true(mask, cancel_check):
            raise ValueError("cached transcript coordinates are not fully indexed")

    cell_index_raw = raw.get("cell_transcript_index")
    cell_index = None
    if cell_index_raw is not None:
        if not isinstance(cell_index_raw, dict):
            raise ValueError("invalid cell transcript index")
        cell_coords = _load_array(
            entry, str(cell_index_raw.get("coords", "")), ndim=2, dtype=np.float32
        )
        cell_codes = _load_array(entry, str(cell_index_raw.get("gene_codes", "")), ndim=1)
        if cell_coords.shape[1:] != (2,) or len(cell_coords) != len(cell_codes):
            raise ValueError("invalid cell transcript arrays")
        if cell_codes.dtype.kind != "u":
            raise ValueError("cell transcript gene codes must be unsigned")
        cell_gene_names_raw = cell_index_raw.get("gene_names")
        if not isinstance(cell_gene_names_raw, list):
            raise ValueError("invalid cell transcript gene names")
        cell_gene_names = tuple(str(value) for value in cell_gene_names_raw)
        maximum_code = _maximum_unsigned_code(cell_codes, cancel_check)
        if maximum_code >= len(cell_gene_names):
            raise ValueError("cell transcript gene code is out of range")
        cell_keys = _load_array(
            entry, str(cell_index_raw.get("slice_keys", "")), ndim=1
        )
        cell_spans = _load_array(
            entry,
            str(cell_index_raw.get("slice_spans", "")),
            ndim=2,
            dtype=np.int64,
        )
        if cell_keys.dtype.kind not in {"U", "S"}:
            raise ValueError("cell transcript slice keys must be strings")
        if cell_spans.shape != (len(cell_keys), 2):
            raise ValueError("invalid cell transcript slice arrays")
        slices: dict[str, tuple[int, int]] = {}
        previous_end = 0
        for index, (key, span) in enumerate(zip(cell_keys, cell_spans, strict=True)):
            if index % 4096 == 0:
                _cancelled(cancel_check)
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            key = str(key)
            if not key or key in slices:
                raise ValueError("invalid or duplicate cell transcript slice key")
            start, end = (int(value) for value in span)
            if not (0 <= start <= end <= len(cell_codes)):
                raise ValueError("cell transcript slice is out of range")
            if start != previous_end:
                raise ValueError("cell transcript slices are not contiguous")
            previous_end = end
            slices[key] = (start, end)
        if previous_end != len(cell_codes):
            raise ValueError("cell transcript slices do not cover the index")
        cell_index = CellTranscriptIndex(
            coords_yx=cell_coords,
            gene_codes=cell_codes,
            gene_names=cell_gene_names,
            slices=slices,
        )

    total_points = int(raw.get("total_points", -1))
    if total_points != sum(len(coords) for coords in group_coords):
        raise ValueError("cached total point count does not match arrays")
    source_counts_raw = raw.get("source_gene_counts")
    source_counts = (
        None
        if source_counts_raw is None
        else {str(key): int(value) for key, value in dict(source_counts_raw).items()}
    )
    source_total_raw = raw.get("source_total_points")
    return GenePointStore(
        group_symbols=group_symbols,
        group_coords=group_coords,
        group_colors=group_colors,
        gene_offsets=offsets,
        gene_counts=counts,
        genes=gene_names,
        control_genes={str(value) for value in raw.get("control_genes", ())},
        total_points=total_points,
        sampled=bool(raw.get("sampled", False)),
        source_gene_counts=source_counts,
        source_total_points=None if source_total_raw is None else int(source_total_raw),
        gene_visuals=visuals,
        cell_transcript_index=cell_index,
    )


def load_transcript_cache(
    request: TranscriptCacheRequest | None,
    reference_resolver: Callable[[list[str]], Any] | None,
    cancel_check: Callable[[], bool] | None = None,
) -> TranscriptCacheHit | None:
    """Restore a valid entry, or return ``None`` for a miss/corruption."""
    if request is None or not request.entry_root.is_dir():
        return None
    try:
        candidates = sorted(path for path in request.entry_root.iterdir() if path.is_dir())
    except OSError:
        return None
    for entry in candidates:
        _cancelled(cancel_check)
        try:
            raw = _read_manifest(entry)
            if (
                int(raw.get("version", -1)) != TRANSCRIPT_CACHE_VERSION
                or str(raw.get("request_digest", "")) != request.digest
                or str(raw.get("request_family_digest", ""))
                != request.family_digest
                or str(raw.get("points_key", "")) != request.points_key
            ):
                continue
            genes_raw = raw.get("genes")
            if not isinstance(genes_raw, list):
                continue
            current = (
                MarkerReferenceResolution(reference=None)
                if reference_resolver is None
                else reference_resolver([str(gene) for gene in genes_raw])
            )
            current_payload = _resolution_payload(current)
            reference_digest = _digest(current_payload)
            if (
                entry.name != reference_digest
                or str(raw.get("reference_digest", "")) != reference_digest
            ):
                continue
            cached_resolution_raw = raw.get("reference_resolution")
            if not isinstance(cached_resolution_raw, dict):
                continue
            if _digest(cached_resolution_raw) != reference_digest:
                continue
            store = _restore_store(entry, raw, cancel_check=cancel_check)
            resolution = _resolution_from_payload(cached_resolution_raw)
            _cancelled(cancel_check)
            return TranscriptCacheHit(store=store, reference_resolution=resolution, path=entry)
        except TranscriptBuildCancelled:
            raise
        except Exception as exc:
            log.warning("Ignoring invalid transcript cache entry %s: %s", entry, exc)
    return None


def _write_npy(
    path: Path,
    value: np.ndarray,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Write one safe NumPy array in bounded, cancellable chunks."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not permitted in transcript caches")
    _cancelled(cancel_check)
    if array.size == 0:
        with path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        return
    target = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
    )
    try:
        if array.ndim == 0:
            target[...] = array
        else:
            rows = _chunk_rows(array)
            for start in range(0, len(array), rows):
                _cancelled(cancel_check)
                target[start : start + rows] = array[start : start + rows]
        target.flush()
        _cancelled(cancel_check)
    finally:
        del target


def _prune_superseded_entries(
    request: TranscriptCacheRequest,
    keep: Path,
) -> None:
    """Best-effort removal of stale source/reference versions for one request."""
    try:
        # Only the currently resolved marker reference is useful for an exact
        # point/build identity. Reference changes otherwise duplicate the full
        # coordinate store below the same request directory.
        for entry in request.entry_root.iterdir():
            if entry == keep or not entry.is_dir() or entry.name.startswith("."):
                continue
            shutil.rmtree(entry, ignore_errors=True)

        # A point-file replacement changes ``digest`` but not ``family_digest``.
        # Retain alternate render caps/seeds as distinct useful families while
        # pruning obsolete source fingerprints for the same build settings.
        version_root = request.entry_root.parent
        for sibling in version_root.iterdir():
            if sibling == request.entry_root or not sibling.is_dir():
                continue
            same_family = False
            for entry in sibling.iterdir():
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                try:
                    raw = _read_manifest(entry)
                except Exception:
                    continue
                same_family = (
                    str(raw.get("request_family_digest", ""))
                    == request.family_digest
                )
                if same_family:
                    break
            if same_family:
                shutil.rmtree(sibling, ignore_errors=True)
    except OSError as exc:
        log.debug("Could not prune superseded transcript cache entries: %s", exc)


def save_transcript_cache(
    request: TranscriptCacheRequest | None,
    store: GenePointStore,
    reference_resolution: Any,
    cancel_check: Callable[[], bool] | None = None,
) -> Path | None:
    """Atomically persist ``store`` outside its source SpatialData zarr."""
    if request is None:
        return None
    _cancelled(cancel_check)
    resolution_payload = _resolution_payload(reference_resolution)
    reference_digest = _digest(resolution_payload)
    final = request.entry_root / reference_digest
    temp: Path | None = None
    stale: Path | None = None
    try:
        # Cache failures are always best-effort: an unwritable user cache must
        # never turn a successful transcript build into a failed viewer load.
        request.entry_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".writing-", dir=request.entry_root))
        groups: list[dict[str, Any]] = []
        for index, (symbol, coords) in enumerate(
            zip(store.group_symbols, store.group_coords, strict=True)
        ):
            _cancelled(cancel_check)
            name = f"group_{index:03d}_coords.npy"
            array = np.asarray(coords, dtype=np.float32)
            if array.ndim != 2 or array.shape[1:] != (2,):
                raise ValueError("transcript group coordinates must have shape (N, 2)")
            _write_npy(temp / name, array, cancel_check=cancel_check)
            groups.append({"symbol": str(symbol), "coords": name, "count": int(len(array))})

        visuals = store.gene_visuals or {}
        if set(store.genes) != set(visuals):
            raise ValueError("a cacheable transcript store needs one visual per gene")
        cell_payload = None
        if store.cell_transcript_index is not None:
            index = store.cell_transcript_index
            cell_coords = np.asarray(index.coords_yx, dtype=np.float32)
            codes = np.asarray(index.gene_codes)
            if cell_coords.ndim != 2 or cell_coords.shape[1:] != (2,):
                raise ValueError("cell transcript coordinates must have shape (N, 2)")
            if codes.ndim != 1 or len(cell_coords) != len(codes):
                raise ValueError("cell transcript coordinate/code lengths do not match")
            if codes.dtype.kind != "u":
                raise ValueError("cell transcript gene codes must use an unsigned dtype")
            gene_names = [str(value) for value in index.gene_names]
            if _maximum_unsigned_code(codes, cancel_check) >= len(gene_names):
                raise ValueError("cell transcript gene code is out of range")
            slice_items = sorted(
                (str(key), (int(span[0]), int(span[1])))
                for key, span in index.slices.items()
            )
            slice_keys = [key for key, _span in slice_items]
            if len(set(slice_keys)) != len(slice_keys) or any(not key for key in slice_keys):
                raise ValueError("cell transcript slice keys must be unique and non-empty")
            slice_spans = np.asarray(
                [span for _key, span in slice_items], dtype=np.int64
            ).reshape((-1, 2))
            previous_end = 0
            for slice_index, (start, end) in enumerate(slice_spans):
                if slice_index % 4096 == 0:
                    _cancelled(cancel_check)
                if not (int(start) == previous_end <= int(end) <= len(codes)):
                    raise ValueError("cell transcript slices must be contiguous and in range")
                previous_end = int(end)
            if previous_end != len(codes):
                raise ValueError("cell transcript slices do not cover the index")

            _write_npy(temp / "cell_coords.npy", cell_coords, cancel_check=cancel_check)
            _write_npy(temp / "cell_gene_codes.npy", codes, cancel_check=cancel_check)
            _write_npy(
                temp / "cell_slice_keys.npy",
                np.asarray(slice_keys, dtype=np.str_),
                cancel_check=cancel_check,
            )
            _write_npy(
                temp / "cell_slice_spans.npy",
                slice_spans,
                cancel_check=cancel_check,
            )
            cell_payload = {
                "coords": "cell_coords.npy",
                "gene_codes": "cell_gene_codes.npy",
                "gene_names": gene_names,
                "slice_keys": "cell_slice_keys.npy",
                "slice_spans": "cell_slice_spans.npy",
            }

        manifest = {
            "version": TRANSCRIPT_CACHE_VERSION,
            "request_digest": request.digest,
            "request_family_digest": request.family_digest,
            "reference_digest": reference_digest,
            "points_key": request.points_key,
            "groups": groups,
            "gene_offsets": {
                str(gene): [int(value) for value in offsets]
                for gene, offsets in sorted(store.gene_offsets.items())
            },
            "gene_counts": {
                str(gene): int(count) for gene, count in sorted(store.gene_counts.items())
            },
            "genes": [str(gene) for gene in store.genes],
            "control_genes": sorted(str(gene) for gene in store.control_genes),
            "total_points": int(store.total_points),
            "sampled": bool(store.sampled),
            "source_gene_counts": None if store.source_gene_counts is None else {
                str(gene): int(count)
                for gene, count in sorted(store.source_gene_counts.items())
            },
            "source_total_points": (
                None if store.source_total_points is None else int(store.source_total_points)
            ),
            "gene_visuals": {
                str(gene): _visual_payload(visuals[gene]) for gene in store.genes
            },
            "cell_transcript_index": cell_payload,
            "reference_resolution": resolution_payload,
        }
        _cancelled(cancel_check)
        manifest_bytes = _canonical_json(manifest)
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("transcript cache manifest exceeds its safe size limit")
        (temp / _MANIFEST).write_bytes(manifest_bytes)

        if final.exists():
            stale = request.entry_root / f".stale-{uuid4().hex}"
            try:
                os.replace(final, stale)
            except FileNotFoundError:  # another writer replaced it first
                stale = None
        try:
            os.replace(temp, final)
        except Exception:
            # If replacement fails after moving a previous valid entry aside,
            # make a best effort to restore it under its content address.
            if stale is not None and stale.exists() and not final.exists():
                try:
                    os.replace(stale, final)
                    stale = None
                except OSError:
                    pass
            raise
        temp = None
        if stale is not None:
            shutil.rmtree(stale, ignore_errors=True)
        _prune_superseded_entries(request, final)
        return final
    except TranscriptBuildCancelled:
        raise
    except Exception as exc:
        log.warning("Could not save transcript cache %s: %s", final, exc)
        return None
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


# Compact payload-oriented API used by the viewer.  Keep the lower-level
# request/hit API above available for focused tests and future cache management.
def make_transcript_cache_request(
    *,
    zarr_path: str | Path,
    points_key: str,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None = None,
    background_col: str | None = None,
    max_points: int | None = None,
    random_state: int = 42,
    build_cell_index: bool = False,
    cache_dir: str | Path | None = None,
) -> TranscriptCacheRequest | None:
    """Return the cache request corresponding to one viewer transcript build."""
    return TranscriptCacheRequest.create(
        dataset_path=zarr_path,
        points_key=points_key,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
        assignment_col=assignment_col,
        background_col=background_col,
        max_points=max_points,
        random_state=random_state,
        build_cell_index=build_cell_index,
        cache_root=cache_dir,
    )


def load_transcript_payload(
    request: TranscriptCacheRequest | None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Restore the viewer payload while revalidating current marker metadata."""
    if request is None:
        return None
    hit = load_transcript_cache(
        request,
        lambda genes: resolve_cell_type_marker_reference(request.dataset_path, genes),
        cancel_check=cancel_check,
    )
    if hit is None:
        return None
    resolution = hit.reference_resolution
    return {
        "points_key": request.points_key,
        "store": hit.store,
        "reference": resolution.reference,
        "reference_resolution": resolution,
        "build_seconds": 0.0,
        "cache_hit": True,
        "cache_path": hit.path,
    }


def save_transcript_payload(
    request: TranscriptCacheRequest | None,
    payload: Mapping[str, Any],
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Persist a viewer transcript payload, returning whether it was cached."""
    store = payload.get("store")
    if not isinstance(store, GenePointStore):
        return False
    return save_transcript_cache(
        request,
        store,
        payload.get("reference_resolution"),
        cancel_check=cancel_check,
    ) is not None
