"""Lazy reader for 10x Atera transcripts (pure: numpy and pyarrow, no Qt).

An Atera store holds about 1.4e9 transcripts in ~400 per-tile parquet files.
Within a tile the rows are cut into files by gene, and every file is sorted by
its ``feature_name`` dictionary code, so one gene lives in a few row groups per
tile.  The viewer never materialises the whole table; this module serves the
pieces it needs:

* :class:`AteraPointsIndex` describes the files.  :meth:`AteraPointsIndex.load`
  validates MerXen's ``merxen_atera_points_index.json`` against the parquet
  footers (``None`` on any mismatch); :meth:`AteraPointsIndex.scan` is the
  index-less fallback (natural ``part.N`` order, no tiles or boxes).
* :class:`CodeRangeMap` maps a global feature code to the exact
  ``(row group, row_start, row_end)`` runs holding it.  Before one exists,
  per-gene reads prune row groups with the ``feature_name`` string statistics,
  which over-read (control probes sort among the gene names).
* :class:`AteraTranscriptEngine` reads batches of genes (each row group read
  once per call), whole tiles and tile-granular windows, and builds the
  :class:`AteraOverview`: exact per-feature totals and assigned counts, a
  density raster over real genes and (optionally) a small seeded sample, in one
  pass that also yields the code-range map.

All file reads go through one module-level pool of :data:`READ_POOL_WORKERS`
threads; each file is read with ``use_threads=False`` and reduced at once, and a
cancel event or callable is checked between files.

Arrow conventions: ``feature_name`` comes back as global integer codes under
the key :data:`CODE_KEY`; ``original_assignment`` always goes through
``fill_null(0)`` (0 is never a cell id) and is returned as ``uint32`` when every
id fits, else ``uint64`` - never float.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .utils import (
    IS_GENE_COLUMN,
    TranscriptBuildCancelled,
    control_genes_from_is_gene,
    is_control_gene,
)

log = logging.getLogger(__name__)

ATERA_POINTS_INDEX_JSON = "merxen_atera_points_index.json"
ATERA_POINTS_INDEX_FORMAT = 1
DEFAULT_POINTS_KEY = "transcripts"
X_COLUMN = "x"
Y_COLUMN = "y"
GENE_COLUMN = "feature_name"
ASSIGNMENT_COLUMN = "original_assignment"
#: Output key for ``feature_name`` (global codes into the vocabulary).
CODE_KEY = "code"
DEFAULT_READ_COLUMNS = (X_COLUMN, Y_COLUMN, GENE_COLUMN, ASSIGNMENT_COLUMN)
#: Bounded reader pool shared by every lazy read (about 70 MB transient per
#: 3.7M-row file and worker).
READ_POOL_WORKERS = 4
DEFAULT_OVERVIEW_BIN_UM = 5.0
#: Largest vocabulary that still fits int16 codes.
INT16_CODE_LIMIT = 32767
#: Refuse a code-range map with more runs than this (a file that is not sorted
#: by feature would otherwise blow it up); reads then use string statistics.
MAX_CODE_MAP_RUNS = 50_000_000

_PART_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Cancellation and the shared reader pool
# ---------------------------------------------------------------------------


def _is_cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(cancel())


def _check_cancel(cancel: Any) -> None:
    if _is_cancelled(cancel):
        raise TranscriptBuildCancelled("Atera transcript read cancelled")


#: Arrow-internal decode threads per read (the pool already parallelises files).
_READ_USE_THREADS = False

_POOL_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None
_IN_POOL = threading.local()


def read_pool() -> ThreadPoolExecutor:
    """Return the module-level bounded reader pool (created on first use)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(
                max_workers=READ_POOL_WORKERS, thread_name_prefix="atera-read"
            )
        return _POOL


def _pool_task(func: Callable[[int], Any], position: int, cancel: Any) -> Any:
    _IN_POOL.active = True
    try:
        _check_cancel(cancel)
        return func(position)
    finally:
        _IN_POOL.active = False


def _map_files(
    func: Callable[[int], Any],
    positions: Sequence[int],
    cancel: Any,
    consume: Callable[[int, Any], None],
) -> None:
    """Run ``func`` per file position on the pool; ``consume`` results in order.

    At most ``2 * READ_POOL_WORKERS`` files are in flight, so memory stays
    bounded however many files there are.  Cancellation is checked between
    files; pending work is cancelled and :class:`TranscriptBuildCancelled`
    raised.  A call from inside a pool worker runs sequentially (no deadlock).
    """
    positions = list(positions)
    if getattr(_IN_POOL, "active", False) or len(positions) <= 1:
        for position in positions:
            _check_cancel(cancel)
            consume(position, func(position))
        return
    pool = read_pool()
    window = 2 * READ_POOL_WORKERS
    pending: deque[tuple[int, Future]] = deque()
    queue = iter(positions)
    try:
        for position in queue:
            _check_cancel(cancel)
            pending.append((position, pool.submit(_pool_task, func, position, cancel)))
            if len(pending) >= window:
                done_position, future = pending.popleft()
                consume(done_position, future.result())
        while pending:
            _check_cancel(cancel)
            done_position, future = pending.popleft()
            consume(done_position, future.result())
    except BaseException:
        for _position, future in pending:
            future.cancel()
        for _position, future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except BaseException:
                    pass
        raise


def _read_row_group(parquet_file: pq.ParquetFile, row_group: int, columns: Sequence[str]) -> pa.Table:
    """The single place a row group is read (tests spy on it)."""
    return parquet_file.read_row_group(int(row_group), columns=list(columns), use_threads=_READ_USE_THREADS)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AteraPointsFile:
    """One parquet file of the points element, in index order."""

    position: int
    name: str
    path: Path
    n_rows: int
    n_row_groups: int
    tile: str | None = None
    #: ``(xmin, xmax, ymin, ymax)`` in um as the index stores it (inclusive), or None.
    bbox_um: tuple[float, float, float, float] | None = None
    block_bounds_um: Any = None
    transcript_id_range: tuple[int, int] | None = None


def _natural_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in _PART_RE.split(name))


def _column_index(metadata: pq.FileMetaData, name: str) -> int | None:
    schema = metadata.schema
    for index in range(len(schema)):
        if schema.column(index).path == name:
            return index
    return None


def _first_dictionary(parquet_file: pq.ParquetFile) -> pa.Array | None:
    if parquet_file.metadata.num_rows == 0:
        return None
    for batch in parquet_file.iter_batches(batch_size=1, columns=[GENE_COLUMN], use_threads=False):
        column = batch.column(0)
        if pa.types.is_dictionary(column.type):
            return column.dictionary
        return None
    return None


def _plain_feature_names(parquet_file: pq.ParquetFile) -> list[str]:
    """Distinct non-null ``feature_name`` values of a file stored as plain strings.

    Only the index-less fallback needs this (MerXen writes a dictionary); it
    reads the one column batch by batch, in first-seen order.
    """
    seen: dict[str, None] = {}
    for batch in parquet_file.iter_batches(columns=[GENE_COLUMN], use_threads=False):
        column = batch.column(0)
        if pa.types.is_dictionary(column.type):
            column = column.dictionary_decode()
        for name in pc.unique(column.drop_null()).to_pylist():
            seen.setdefault(str(name), None)
    return list(seen)


def _assignment_dtype(metadatas: Sequence[pq.FileMetaData]) -> np.dtype:
    """uint32 when the footer statistics prove every id is below 2**32."""
    maximum = 0
    for metadata in metadatas:
        index = _column_index(metadata, ASSIGNMENT_COLUMN)
        if index is None:
            continue
        for rg in range(metadata.num_row_groups):
            group = metadata.row_group(rg)
            stats = group.column(index).statistics
            if group.num_rows == 0:
                continue
            if stats is not None and stats.has_null_count and stats.null_count == group.num_rows:
                continue  # every id null
            if stats is None or not stats.has_min_max:
                return np.dtype(np.uint64)
            try:
                maximum = max(maximum, int(stats.max))
            except (TypeError, ValueError):
                return np.dtype(np.uint64)
    return np.dtype(np.uint32) if maximum < 2**32 else np.dtype(np.uint64)


@dataclass
class AteraPointsIndex:
    """The validated (or scanned) file list of one Atera points element."""

    zarr_path: Path
    points_key: str
    directory: Path
    files: tuple[AteraPointsFile, ...]
    metadata: tuple[pq.FileMetaData, ...]
    vocabulary: tuple[str, ...]
    #: Per file: local dictionary index -> global code, or None when the file's
    #: dictionary is the vocabulary itself.
    file_luts: tuple[np.ndarray | None, ...]
    shared_dictionary: bool
    assignment_dtype: np.dtype
    #: True only for an index JSON that matched every footer.
    validated: bool
    index_path: Path | None = None
    fingerprint: dict[str, int] = field(default_factory=dict)
    rg_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    rg_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _vocab_lookup: dict[str, int] = field(default_factory=dict, repr=False)
    _file_dictionaries: tuple[pa.Array | None, ...] = field(default=(), repr=False)

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, zarr_path: str | Path, points_key: str = DEFAULT_POINTS_KEY) -> "AteraPointsIndex | None":
        """Load and validate the MerXen points index; ``None`` on any mismatch."""
        store = Path(zarr_path)
        index_path = store / ATERA_POINTS_INDEX_JSON
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("[ATERA] Unreadable points index %s: %s", index_path, exc)
            return None
        try:
            return cls._from_payload(store, str(points_key), index_path, raw)
        except (_IndexMismatch, TypeError, ValueError, AttributeError, KeyError) as exc:
            # A parseable but malformed payload (null or non-numeric counts,
            # non-float bbox values, a non-dict ``points``) is a mismatch too.
            log.warning("[ATERA] Points index %s does not match the files (%s); using the fallback reader.", index_path, exc)
            return None

    @classmethod
    def _from_payload(cls, store: Path, points_key: str, index_path: Path, raw: Any) -> "AteraPointsIndex":
        if not isinstance(raw, dict):
            raise _IndexMismatch("not a JSON object")
        if int(raw.get("format_version", -1)) != ATERA_POINTS_INDEX_FORMAT:
            raise _IndexMismatch(f"unsupported format_version {raw.get('format_version')!r}")
        element = (raw.get("points") or {}).get(points_key)
        if not isinstance(element, dict):
            raise _IndexMismatch(f"no entry for points {points_key!r}")
        directory_text = element.get("directory")
        if not isinstance(directory_text, str) or not directory_text:
            raise _IndexMismatch("missing directory")
        directory = (store / directory_text).resolve()
        if not _is_within(directory, store.resolve()) or not directory.is_dir():
            raise _IndexMismatch(f"directory {directory_text!r} is not inside the store")
        fingerprint = element.get("fingerprint") or {}
        entries = element.get("files")
        if not isinstance(entries, list) or not entries:
            raise _IndexMismatch("no files")
        try:
            n_files = int(fingerprint["n_files"])
            n_rows = int(fingerprint["n_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _IndexMismatch("missing fingerprint") from exc
        if n_files != len(entries):
            raise _IndexMismatch(f"fingerprint n_files {n_files} != {len(entries)} entries")
        files: list[AteraPointsFile] = []
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise _IndexMismatch("invalid file entry")
            name = entry.get("file")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise _IndexMismatch(f"invalid file name {name!r}")
            bbox = entry.get("bbox_um")
            bbox_tuple = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(v is not None for v in bbox):
                bbox_tuple = tuple(float(v) for v in bbox)
            id_range = entry.get("transcript_id_range")
            files.append(
                AteraPointsFile(
                    position=position,
                    name=name,
                    path=directory / name,
                    n_rows=int(entry.get("n_rows", -1)),
                    n_row_groups=int(entry.get("n_row_groups", -1)),
                    tile=None if entry.get("tile") is None else str(entry.get("tile")),
                    bbox_um=bbox_tuple,
                    block_bounds_um=entry.get("block_bounds_um"),
                    transcript_id_range=(
                        tuple(int(v) for v in id_range)
                        if isinstance(id_range, (list, tuple)) and len(id_range) == 2
                        else None
                    ),
                )
            )
        if sum(f.n_rows for f in files) != n_rows:
            raise _IndexMismatch(f"fingerprint n_rows {n_rows} != the sum of the entries")
        metadatas: list[pq.FileMetaData] = []
        for item in files:
            try:
                metadata = pq.read_metadata(item.path)
            except (OSError, pa.ArrowException) as exc:
                raise _IndexMismatch(f"{item.name}: {exc}") from exc
            if metadata.num_rows != item.n_rows:
                raise _IndexMismatch(f"{item.name}: footer num_rows {metadata.num_rows} != {item.n_rows}")
            if metadata.num_row_groups != item.n_row_groups:
                raise _IndexMismatch(
                    f"{item.name}: footer num_row_groups {metadata.num_row_groups} != {item.n_row_groups}"
                )
            metadatas.append(metadata)
        return cls._build(
            store, points_key, directory, tuple(files), tuple(metadatas),
            validated=True, index_path=index_path,
            fingerprint={"n_files": n_files, "n_rows": n_rows},
        )

    @classmethod
    def scan(cls, zarr_path: str | Path, points_key: str = DEFAULT_POINTS_KEY) -> "AteraPointsIndex | None":
        """Index-less fallback: the parquet files in natural ``part.N`` order."""
        store = Path(zarr_path)
        element = store / "points" / str(points_key)
        candidates = [element / "points.parquet", element]
        paths: list[Path] = []
        directory = element
        for candidate in candidates:
            if candidate.is_file():
                paths, directory = [candidate], candidate.parent
                break
            if candidate.is_dir():
                found = sorted(
                    (p for p in candidate.iterdir() if p.is_file() and p.suffix == ".parquet"),
                    key=lambda p: _natural_key(p.name),
                )
                if found:
                    paths, directory = found, candidate
                    break
        if not paths:
            return None
        files = []
        metadatas = []
        for position, path in enumerate(paths):
            try:
                metadata = pq.read_metadata(path)
            except (OSError, pa.ArrowException) as exc:
                log.warning("[ATERA] Unreadable points file %s: %s", path, exc)
                return None
            metadatas.append(metadata)
            files.append(
                AteraPointsFile(
                    position=position,
                    name=path.name,
                    path=path,
                    n_rows=int(metadata.num_rows),
                    n_row_groups=int(metadata.num_row_groups),
                )
            )
        try:
            return cls._build(
                store, str(points_key), directory, tuple(files), tuple(metadatas),
                validated=False, index_path=None,
                fingerprint={"n_files": len(files), "n_rows": int(sum(f.n_rows for f in files))},
            )
        except _IndexMismatch as exc:
            log.warning("[ATERA] Cannot read the points files under %s: %s", directory, exc)
            return None

    @classmethod
    def _build(
        cls,
        store: Path,
        points_key: str,
        directory: Path,
        files: tuple[AteraPointsFile, ...],
        metadatas: tuple[pq.FileMetaData, ...],
        *,
        validated: bool,
        index_path: Path | None,
        fingerprint: dict[str, int],
    ) -> "AteraPointsIndex":
        dictionaries: list[pa.Array | None] = [None] * len(files)
        plain_names: list[list[str]] = [[] for _ in files]

        def first_dictionary(position: int) -> tuple[pa.Array | None, list[str]]:
            parquet_file = pq.ParquetFile(files[position].path, metadata=metadatas[position])
            dictionary = _first_dictionary(parquet_file)
            if dictionary is None and metadatas[position].num_rows > 0:
                # Plain strings: the vocabulary must come from the values.
                return None, _plain_feature_names(parquet_file)
            return dictionary, []

        def keep(position: int, value: tuple[pa.Array | None, list[str]]) -> None:
            dictionaries[position], plain_names[position] = value

        try:
            _map_files(first_dictionary, range(len(files)), None, keep)
        except (OSError, pa.ArrowException) as exc:
            raise _IndexMismatch(f"unreadable feature dictionary: {exc}") from exc
        reference = next((d for d in dictionaries if d is not None), None)
        vocabulary: list[str] = [] if reference is None else [str(v) for v in reference.to_pylist()]
        lookup = {name: code for code, name in enumerate(vocabulary)}
        shared = True
        luts: list[np.ndarray | None] = []
        for dictionary in dictionaries:
            if dictionary is None or dictionary is reference or dictionary.equals(reference):
                luts.append(None)
                continue
            shared = False
            lut = np.empty(len(dictionary), dtype=np.int32)
            for local, name in enumerate(dictionary.to_pylist()):
                name = str(name)
                code = lookup.get(name)
                if code is None:
                    code = len(vocabulary)
                    vocabulary.append(name)
                    lookup[name] = code
                lut[local] = code
            luts.append(lut)
        plain = [names for names in plain_names if names]
        if plain:
            # Plain-string files are resolved through the vocabulary lookup.
            shared = False
            for names in plain:
                for name in names:
                    if name not in lookup:
                        lookup[name] = len(vocabulary)
                        vocabulary.append(name)
            log.info("[ATERA] Some points files store feature_name as plain strings; codes come from the name lookup.")
        if not shared and not plain:
            log.info("[ATERA] Points files do not share one feature dictionary; remapping codes per file.")
        rg_counts = np.array([m.num_row_groups for m in metadatas], dtype=np.int64)
        rg_offsets = np.concatenate([[0], np.cumsum(rg_counts)]).astype(np.int64)
        rg_rows = np.array(
            [m.row_group(g).num_rows for m in metadatas for g in range(m.num_row_groups)],
            dtype=np.int64,
        )
        return cls(
            zarr_path=store,
            points_key=points_key,
            directory=directory,
            files=files,
            metadata=metadatas,
            vocabulary=tuple(vocabulary),
            file_luts=tuple(luts),
            shared_dictionary=shared,
            assignment_dtype=_assignment_dtype(metadatas),
            validated=validated,
            index_path=index_path,
            fingerprint=dict(fingerprint),
            rg_offsets=rg_offsets,
            rg_rows=rg_rows,
            _vocab_lookup=lookup,
            _file_dictionaries=tuple(dictionaries),
        )

    # -- properties -------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return int(sum(f.n_rows for f in self.files))

    @property
    def n_row_groups(self) -> int:
        return int(self.rg_offsets[-1])

    @property
    def code_dtype(self) -> np.dtype:
        return np.dtype(np.int16) if len(self.vocabulary) <= INT16_CODE_LIMIT else np.dtype(np.int32)

    @property
    def tiles(self) -> tuple[str, ...]:
        """Tile names in index order (empty without a validated index)."""
        seen: dict[str, None] = {}
        for item in self.files:
            if item.tile is not None:
                seen.setdefault(item.tile, None)
        return tuple(seen)

    def code_of(self, name: str) -> int | None:
        return self._vocab_lookup.get(str(name))

    def extent_um(self) -> tuple[float, float, float, float] | None:
        """``(xmin, xmax, ymin, ymax)`` over every row: index boxes, else footer stats."""
        boxes = [f.bbox_um for f in self.files if f.n_rows > 0]
        if self.validated and boxes and all(box is not None for box in boxes):
            arr = np.asarray(boxes, dtype=np.float64)
            return (float(arr[:, 0].min()), float(arr[:, 1].max()), float(arr[:, 2].min()), float(arr[:, 3].max()))
        values: list[tuple[float, float, float, float]] = []
        for metadata in self.metadata:
            xi = _column_index(metadata, X_COLUMN)
            yi = _column_index(metadata, Y_COLUMN)
            if xi is None or yi is None:
                return None
            for rg in range(metadata.num_row_groups):
                group = metadata.row_group(rg)
                if group.num_rows == 0:
                    continue
                xs, ys = group.column(xi).statistics, group.column(yi).statistics
                if xs is None or ys is None or not xs.has_min_max or not ys.has_min_max:
                    return None
                values.append((float(xs.min), float(xs.max), float(ys.min), float(ys.max)))
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return (float(arr[:, 0].min()), float(arr[:, 1].max()), float(arr[:, 2].min()), float(arr[:, 3].max()))

    # -- conversion helpers -----------------------------------------------

    def global_codes(self, position: int, column: pa.ChunkedArray | pa.Array) -> np.ndarray:
        """Global codes of a ``feature_name`` column of file ``position`` (-1 for null)."""
        chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
        out: list[np.ndarray] = []
        file_dictionary = self._file_dictionaries[position] if self._file_dictionaries else None
        for chunk in chunks:
            if not pa.types.is_dictionary(chunk.type):
                names = chunk.to_pylist()
                out.append(np.array([self._vocab_lookup.get(str(n), -1) if n is not None else -1 for n in names], dtype=np.int32))
                continue
            indices = chunk.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            dictionary = chunk.dictionary
            if file_dictionary is not None and (dictionary is file_dictionary or dictionary.equals(file_dictionary)):
                lut = self.file_luts[position]
            else:
                lut = np.array([self._vocab_lookup.get(str(n), -1) for n in dictionary.to_pylist()], dtype=np.int32)
                if np.any(lut < 0):
                    log.warning("[ATERA] %s holds feature names outside the vocabulary; their rows are dropped.", self.files[position].name)
            if lut is None:
                out.append(indices)
            else:
                valid = indices >= 0
                mapped = np.full(len(indices), -1, dtype=np.int32)
                mapped[valid] = lut[indices[valid]]
                out.append(mapped)
        if not out:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(out) if len(out) > 1 else out[0]

    def column_to_numpy(self, name: str, column: pa.ChunkedArray, position: int) -> np.ndarray:
        if name == GENE_COLUMN:
            return self.global_codes(position, column).astype(self.code_dtype, copy=False)
        if name == ASSIGNMENT_COLUMN:
            filled = pc.fill_null(column, 0)
            values = filled.to_numpy() if isinstance(filled, pa.ChunkedArray) else filled.to_numpy(zero_copy_only=False)
            return np.asarray(values).astype(self.assignment_dtype, copy=False)
        if pa.types.is_dictionary(column.type):
            column = column.dictionary_decode() if isinstance(column, pa.Array) else pa.chunked_array(
                [c.dictionary_decode() for c in column.chunks], type=column.type.value_type
            )
        return np.asarray(column.to_numpy())

    def empty_column(self, name: str) -> np.ndarray:
        if name == GENE_COLUMN:
            return np.zeros(0, dtype=self.code_dtype)
        if name == ASSIGNMENT_COLUMN:
            return np.zeros(0, dtype=self.assignment_dtype)
        schema = self.metadata[0].schema.to_arrow_schema() if self.metadata else None
        if schema is not None and name in schema.names:
            arrow_type = schema.field(name).type
            if pa.types.is_dictionary(arrow_type):
                arrow_type = arrow_type.value_type
            try:
                return np.zeros(0, dtype=arrow_type.to_pandas_dtype())
            except (NotImplementedError, TypeError):
                return np.zeros(0, dtype=object)
        return np.zeros(0, dtype=np.float32)

    def row_group_ids(self, position: int) -> range:
        return range(int(self.rg_offsets[position]), int(self.rg_offsets[position + 1]))

    def locate_row_group(self, global_rg: int) -> tuple[int, int]:
        position = int(np.searchsorted(self.rg_offsets, global_rg, side="right") - 1)
        return position, int(global_rg - self.rg_offsets[position])


_CURRENT_MAP = object()  # plan_gene_reads sentinel: use the attached map


class _IndexMismatch(ValueError):
    pass


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def output_key(column: str) -> str:
    """The key a requested column is returned under."""
    return CODE_KEY if column == GENE_COLUMN else column


# ---------------------------------------------------------------------------
# Code-range map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeRangeMap:
    """Exact ``code -> [(global row group, row_start, row_end)]`` runs.

    Runs of one code are ``row_group[code_ptr[c]:code_ptr[c + 1]]`` (with the
    matching ``row_start`` / ``row_end``), ordered by file then row group.
    """

    code_ptr: np.ndarray  # int64, n_codes + 1
    row_group: np.ndarray  # int32
    row_start: np.ndarray  # int64
    row_end: np.ndarray  # int64

    @property
    def n_codes(self) -> int:
        return int(len(self.code_ptr) - 1)

    @property
    def n_runs(self) -> int:
        return int(len(self.row_group))

    def runs(self, code: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = int(self.code_ptr[code]), int(self.code_ptr[code + 1])
        return self.row_group[lo:hi], self.row_start[lo:hi], self.row_end[lo:hi]

    def rows_for(self, codes: Iterable[int]) -> int:
        """Rows the runs of ``codes`` cover (what a read of them keeps)."""
        total = 0
        for code in codes:
            _rgs, starts, ends = self.runs(int(code))
            total += int((ends - starts).sum())
        return total

    def validate(self, index: AteraPointsIndex) -> bool:
        """True when the map fits ``index`` (vocabulary size, row groups, rows)."""
        try:
            if self.n_codes != len(index.vocabulary):
                return False
            if self.code_ptr[0] != 0 or self.code_ptr[-1] != self.n_runs:
                return False
            if np.any(np.diff(self.code_ptr) < 0):
                return False
            if not (len(self.row_start) == len(self.row_end) == self.n_runs):
                return False
            if self.n_runs == 0:
                return True
            if self.row_group.min() < 0 or self.row_group.max() >= index.n_row_groups:
                return False
            if np.any(self.row_start < 0) or np.any(self.row_end <= self.row_start):
                return False
            return bool(np.all(self.row_end <= index.rg_rows[self.row_group]))
        except (IndexError, ValueError):
            return False


def _code_runs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(code, start, end)`` of each run of equal codes (null runs dropped)."""
    n = len(codes)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty.astype(np.int32), empty, empty
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    ends = np.concatenate([change, [n]]).astype(np.int64)
    run_codes = codes[starts].astype(np.int32)
    keep = run_codes >= 0
    return run_codes[keep], starts[keep], ends[keep]


class _CodeMapBuilder:
    def __init__(self, index: AteraPointsIndex) -> None:
        self.index = index
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.n_runs = 0
        self.failed = False

    def add(self, global_rg: int, codes: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> None:
        if self.failed:
            return
        self.n_runs += len(codes)
        if self.n_runs > MAX_CODE_MAP_RUNS:
            log.warning("[ATERA] More than %d feature runs; not building a code-range map.", MAX_CODE_MAP_RUNS)
            self.failed = True
            self.parts.clear()
            return
        self.parts.append((codes, np.full(len(codes), global_rg, dtype=np.int32), starts, ends))

    def finish(self) -> CodeRangeMap | None:
        if self.failed:
            return None
        n_codes = len(self.index.vocabulary)
        if self.parts:
            codes = np.concatenate([p[0] for p in self.parts])
            rgs = np.concatenate([p[1] for p in self.parts])
            starts = np.concatenate([p[2] for p in self.parts])
            ends = np.concatenate([p[3] for p in self.parts])
        else:
            codes = np.zeros(0, dtype=np.int32)
            rgs = np.zeros(0, dtype=np.int32)
            starts = ends = np.zeros(0, dtype=np.int64)
        # Parts arrive in global row-group order; a stable sort keeps it per code.
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=n_codes) if len(codes) else np.zeros(n_codes, dtype=np.int64)
        code_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return CodeRangeMap(
            code_ptr=code_ptr,
            row_group=rgs[order].astype(np.int32),
            row_start=starts[order].astype(np.int64),
            row_end=ends[order].astype(np.int64),
        )


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@dataclass
class AteraOverview:
    """Exact per-feature counts, a density raster and the code-range map."""

    vocabulary: tuple[str, ...]
    counts: np.ndarray  # int64 per code
    assigned_counts: np.ndarray  # int64 per code
    is_control: np.ndarray  # bool per code
    #: Real-gene transcripts per bin; row = y, column = x.
    density_counts: np.ndarray  # uint32 (H, W)
    origin_um: tuple[float, float]  # (x0, y0) of bin (0, 0)
    bin_um: float
    total_rows: int
    fingerprint: dict[str, int]
    code_map: CodeRangeMap | None = None
    assigned_density_counts: np.ndarray | None = None
    #: ``{"x", "y", "code", "original_assignment"}`` of the optional sample.
    sample: dict[str, np.ndarray] | None = None
    sample_size: int = 0
    sample_seed: int = 0
    is_gene_flags_used: bool = False
    #: Whole-slide hash sample kept during the pass (``keep_threshold``); never cached here.
    hash_sample: "SampledTranscripts | None" = None

    @property
    def density(self) -> np.ndarray:
        return self.density_counts.astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.density_counts.shape)  # type: ignore[return-value]

    def control_names(self) -> set[str]:
        return {self.vocabulary[i] for i in np.flatnonzero(self.is_control)}

    def gene_names(self) -> list[str]:
        """Real (non-control) features in vocabulary order, observed or not."""
        return [self.vocabulary[i] for i in np.flatnonzero(~self.is_control)]

    def counts_by_name(self, *, include_controls: bool = True) -> dict[str, int]:
        return {
            name: int(self.counts[i])
            for i, name in enumerate(self.vocabulary)
            if include_controls or not self.is_control[i]
        }

    def assigned_by_name(self, *, include_controls: bool = True) -> dict[str, int]:
        return {
            name: int(self.assigned_counts[i])
            for i, name in enumerate(self.vocabulary)
            if include_controls or not self.is_control[i]
        }


def _grid(extent: tuple[float, float, float, float], bin_um: float) -> tuple[float, float, int, int]:
    xmin, xmax, ymin, ymax = extent
    x0 = math.floor(xmin / bin_um) * bin_um
    y0 = math.floor(ymin / bin_um) * bin_um
    width = int(math.floor((xmax - x0) / bin_um)) + 1
    height = int(math.floor((ymax - y0) / bin_um)) + 1
    return float(x0), float(y0), max(1, width), max(1, height)


def _bin_block(
    xs: np.ndarray, ys: np.ndarray, x0: float, y0: float, bin_um: float, width: int, height: int
) -> tuple[int, int, np.ndarray] | None:
    """Count rows per bin over their own bounding block: ``(row0, col0, counts)``.

    Bin indices are ``floor((v - origin) / bin_um)`` in float64 (truncation
    equals floor for the non-negative offsets of in-extent rows); rows outside
    the extent (a stale box) are clipped onto its border, never dropped.
    """
    if len(xs) == 0:
        return None
    ix = np.subtract(xs, x0, dtype=np.float64)
    ix /= bin_um
    ix = ix.astype(np.int32)
    iy = np.subtract(ys, y0, dtype=np.float64)
    iy /= bin_um
    iy = iy.astype(np.int32)
    c0, c1 = int(ix.min()), int(ix.max())
    r0, r1 = int(iy.min()), int(iy.max())
    if c0 < 0 or c1 >= width:
        np.clip(ix, 0, width - 1, out=ix)
        c0, c1 = min(max(c0, 0), width - 1), min(max(c1, 0), width - 1)
    if r0 < 0 or r1 >= height:
        np.clip(iy, 0, height - 1, out=iy)
        r0, r1 = min(max(r0, 0), height - 1), min(max(r1, 0), height - 1)
    w = c1 - c0 + 1
    # flat = (iy - r0) * w + (ix - c0), in place.
    iy -= r0
    iy *= w
    iy += ix
    iy -= c0
    block = np.bincount(iy, minlength=(r1 - r0 + 1) * w).reshape(r1 - r0 + 1, w)
    return r0, c0, block


def _run_sums(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray, *, contiguous: bool = False) -> np.ndarray:
    """Number of True rows of ``mask`` inside each ``[start, end)`` run.

    ``contiguous`` runs tile the whole array (no null-code gaps), which allows
    the faster ``reduceat``.
    """
    if len(starts) == 0:
        return np.zeros(0, dtype=np.int64)
    if contiguous:
        return np.add.reduceat(mask.view(np.uint8), starts, dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    return cumulative[ends] - cumulative[starts]


def _may_hold_zero(metadata: pq.FileMetaData, rg: int, column: int | None) -> bool:
    """False when the statistics prove no stored id is 0 (nulls aside)."""
    if column is None:
        return True
    stats = metadata.row_group(rg).column(column).statistics
    if stats is None or not stats.has_min_max:
        return True
    try:
        return int(stats.min) <= 0
    except (TypeError, ValueError):
        return True


def sample_global_positions(total: int, k: int, seed: int) -> np.ndarray:
    """``k`` sorted, unique row positions in ``[0, total)``; safe for total >= 1e9.

    ``rng.hypergeometric`` refuses populations of 1e9 or more; ``rng.choice``
    without replacement does not.  ``shuffle=False`` does not sort, so the
    result is sorted explicitly.  Deterministic per ``seed``.
    """
    total = int(total)
    k = int(k)
    if total < 0 or k < 0:
        raise ValueError("total and k must be non-negative")
    k = min(k, total)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    positions = rng.choice(total, size=k, replace=False, shuffle=False)
    return np.sort(np.asarray(positions, dtype=np.int64))


# ---------------------------------------------------------------------------
# Whole-slide hash sample (the "full" transcript mode)
# ---------------------------------------------------------------------------

#: Sample keys are the top 32 bits of a splitmix64 hash, so a threshold in
#: ``[0, SAMPLE_KEY_SPACE]`` selects a fraction of the rows.
SAMPLE_KEY_SPACE = 1 << 32
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)
#: Bits reserved for the row index in the hashed (file, row) word.
_ROW_BITS = 40


def sample_threshold(budget: int, total: int) -> int:
    """Key threshold that keeps about ``budget`` of ``total`` rows.

    Every row with a key below the threshold is kept, so a smaller budget is
    always a subset of a larger one (the samples nest), and a budget at or
    above ``total`` keeps every row.
    """
    budget = int(budget)
    total = int(total)
    if budget <= 0 or total <= 0:
        return 0
    if budget >= total:
        return SAMPLE_KEY_SPACE
    return int((budget * SAMPLE_KEY_SPACE) // total)


def threshold_fraction(threshold: int) -> float:
    return min(1.0, max(0.0, float(threshold) / float(SAMPLE_KEY_SPACE)))


def row_sample_keys(position: int, row_start: int, n: int) -> np.ndarray:
    """``uint32`` sample keys of rows ``row_start .. row_start + n`` of file ``position``.

    A splitmix64 hash of ``(file position, row index)``: stable across runs,
    machines and budgets, and independent of the scheduling order.
    """
    n = int(n)
    base = (int(position) + 1) << _ROW_BITS
    z = np.arange(base + int(row_start), base + int(row_start) + n, dtype=np.uint64)
    z += _SPLITMIX_GAMMA
    z ^= z >> np.uint64(30)
    z *= _SPLITMIX_M1
    z ^= z >> np.uint64(27)
    z *= _SPLITMIX_M2
    z ^= z >> np.uint64(31)
    return (z >> np.uint64(32)).astype(np.uint32)


def _keep_mask(position: int, row_start: int, n: int, threshold: int) -> np.ndarray | None:
    """Rows of a block kept by ``threshold`` (``None`` = every row)."""
    if threshold >= SAMPLE_KEY_SPACE:
        return None
    if threshold <= 0:
        return np.zeros(int(n), dtype=bool)
    return row_sample_keys(position, row_start, n) < np.uint32(threshold)


#: Bytes one sampled row takes in memory and in the external cache:
#: float32 x and y, int16 code, uint32 cell id and uint32 sample key.
SAMPLE_ROW_BYTES = 18


@dataclass
class SampledTranscripts:
    """A deterministic whole-slide sample: every row whose key is below ``threshold``."""

    x: np.ndarray  # float32
    y: np.ndarray  # float32
    code: np.ndarray  # global feature code (int16/int32)
    cell_id: np.ndarray  # uint32/uint64, 0 = unassigned
    key: np.ndarray  # uint32 sample key
    threshold: int
    total_rows: int

    @property
    def n(self) -> int:
        return int(len(self.x))

    @property
    def complete(self) -> bool:
        """True when every row of the store is in the sample."""
        return self.threshold >= SAMPLE_KEY_SPACE

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.x, self.y, self.code, self.cell_id, self.key)))

    def subset(self, threshold: int) -> "SampledTranscripts":
        """The nested sample of a smaller ``threshold`` (no re-read)."""
        threshold = int(threshold)
        if threshold >= self.threshold:
            if threshold > self.threshold and not self.complete:
                raise ValueError("cannot grow a sample by filtering it")
            return self
        keep = np.flatnonzero(self.key < np.uint32(threshold)) if threshold > 0 else np.zeros(0, np.int64)
        return SampledTranscripts(
            x=self.x[keep], y=self.y[keep], code=self.code[keep], cell_id=self.cell_id[keep],
            key=self.key[keep], threshold=threshold, total_rows=self.total_rows,
        )


def physical_memory_bytes() -> int | None:
    """Physical RAM of this machine (psutil, else ``os.sysconf``), or ``None``."""
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def auto_point_budget(
    total: int, *, bytes_per_point: float, ram_bytes: int | None, fraction: float = 0.5
) -> int:
    """``floor(fraction * RAM / bytes_per_point)``, capped at ``total``."""
    total = max(0, int(total))
    if ram_bytes is None or ram_bytes <= 0 or bytes_per_point <= 0:
        return total
    return int(min(total, math.floor(float(fraction) * float(ram_bytes) / float(bytes_per_point))))


def clamp_point_budget(
    budget: int, *, bytes_per_point: float, ram_bytes: int | None, limit_fraction: float = 0.85
) -> tuple[int, bool]:
    """Clamp ``budget`` so its estimated memory stays under ``limit_fraction`` of RAM.

    Returns ``(budget, clamped)``.
    """
    budget = max(0, int(budget))
    if ram_bytes is None or ram_bytes <= 0 or bytes_per_point <= 0:
        return budget, False
    limit = int(math.floor(float(limit_fraction) * float(ram_bytes) / float(bytes_per_point)))
    if budget > limit:
        return max(0, limit), True
    return budget, False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class WindowRead:
    """Rows of a window read and the tiles they came from."""

    tiles: tuple[str, ...]
    columns: dict[str, np.ndarray]
    files_read: int = 0


class AteraTranscriptEngine:
    """Batched gene, tile and window reads and the overview pass."""

    def __init__(self, index: AteraPointsIndex, code_map: CodeRangeMap | None = None) -> None:
        self.index = index
        self.code_map: CodeRangeMap | None = None
        if code_map is not None:
            self.attach_code_map(code_map)

    @classmethod
    def open(cls, zarr_path: str | Path, points_key: str = DEFAULT_POINTS_KEY) -> "AteraTranscriptEngine | None":
        """Validated index when possible, else the scanned fallback (or ``None``)."""
        index = AteraPointsIndex.load(zarr_path, points_key)
        if index is None:
            index = AteraPointsIndex.scan(zarr_path, points_key)
        return None if index is None else cls(index)

    @property
    def window_reads_available(self) -> bool:
        return bool(self.index.validated)

    def attach_code_map(self, code_map: CodeRangeMap | None) -> bool:
        """Use ``code_map`` for gene reads when it fits the index."""
        if code_map is None:
            self.code_map = None
            return False
        if not code_map.validate(self.index):
            log.warning("[ATERA] Ignoring a code-range map that does not fit the points index.")
            return False
        self.code_map = code_map
        return True

    # -- shared helpers ---------------------------------------------------

    def _parquet(self, position: int) -> pq.ParquetFile:
        item = self.index.files[position]
        return pq.ParquetFile(item.path, metadata=self.index.metadata[position])

    def _columns(self, columns: Sequence[str]) -> list[str]:
        out: list[str] = []
        for name in columns:
            if name not in out:
                out.append(str(name))
        return out

    def _convert(self, table: pa.Table, columns: Sequence[str], position: int) -> dict[str, np.ndarray]:
        return {
            output_key(name): self.index.column_to_numpy(name, table.column(name), position)
            for name in columns
        }

    def _concat(self, parts: list[dict[str, np.ndarray]], columns: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in columns:
            key = output_key(name)
            arrays = [p[key] for p in parts if key in p]
            out[key] = np.concatenate(arrays) if arrays else self.index.empty_column(name)
        return out

    def _stats_row_groups(self, position: int, names: Sequence[str]) -> list[int]:
        """Row groups whose ``feature_name`` string statistics may hold ``names``."""
        metadata = self.index.metadata[position]
        column = _column_index(metadata, GENE_COLUMN)
        selected = []
        for rg in range(metadata.num_row_groups):
            group = metadata.row_group(rg)
            if group.num_rows == 0:
                continue
            stats = None if column is None else group.column(column).statistics
            if stats is None or not stats.has_min_max:
                selected.append(rg)
                continue
            lo, hi = str(stats.min), str(stats.max)
            if any(lo <= name <= hi for name in names):
                selected.append(rg)
        return selected

    def plan_gene_reads(
        self, codes: Sequence[int], code_map: CodeRangeMap | None | object = _CURRENT_MAP
    ) -> dict[int, dict[int, list[tuple[int, int, int]] | None]]:
        """``file -> row group -> [(code, start, end)]`` (``None`` = filter by code).

        ``code_map`` defaults to the attached map; callers that act on the plan
        pass their own snapshot so a map attached by another thread mid-call
        cannot make the plan and the read disagree.
        """
        if code_map is _CURRENT_MAP:
            code_map = self.code_map
        plan: dict[int, dict[int, list[tuple[int, int, int]] | None]] = {}
        if code_map is not None:
            for code in codes:
                rgs, starts, ends = code_map.runs(int(code))  # type: ignore[union-attr]
                for g, s, e in zip(rgs.tolist(), starts.tolist(), ends.tolist()):
                    position, local = self.index.locate_row_group(g)
                    runs = plan.setdefault(position, {}).setdefault(local, [])
                    runs.append((int(code), int(s), int(e)))  # type: ignore[union-attr]
            for groups in plan.values():
                for runs in groups.values():
                    runs.sort(key=lambda run: run[1])  # type: ignore[union-attr]
            return plan
        names = [self.index.vocabulary[c] for c in codes]
        for position in range(len(self.index.files)):
            for rg in self._stats_row_groups(position, names):
                plan.setdefault(position, {})[rg] = None
        return plan

    # -- gene reads -------------------------------------------------------

    def read_genes(
        self,
        genes: Iterable[str],
        columns: Sequence[str] = DEFAULT_READ_COLUMNS,
        cancel: Any = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Rows of each gene, reading the union of their row groups once.

        Unknown names map to empty arrays.  Rows are in index file order, then
        row-group and row order.
        """
        names = list(dict.fromkeys(str(g) for g in genes))
        columns = self._columns(columns)
        codes = {name: self.index.code_of(name) for name in names}
        wanted = sorted({c for c in codes.values() if c is not None})
        # One snapshot: build_overview may attach a map from another thread.
        plan = self.plan_gene_reads(wanted, self.code_map)
        # Decide from the plan itself: any row group without runs is filtered
        # by code, so it needs the gene column.
        needs_codes = any(runs is None for groups in plan.values() for runs in groups.values())
        read_columns = [*columns, GENE_COLUMN] if needs_codes and GENE_COLUMN not in columns else list(columns)
        wanted_set = set(wanted)

        def work(position: int) -> dict[int, list[dict[str, np.ndarray]]]:
            parquet_file = self._parquet(position)
            found: dict[int, list[dict[str, np.ndarray]]] = {}
            for rg in sorted(plan[position]):
                table = _read_row_group(parquet_file, rg, read_columns)
                runs = plan[position][rg]
                if runs is not None:
                    for code, start, end in runs:
                        found.setdefault(code, []).append(self._convert(table.slice(start, end - start), columns, position))
                    continue
                row_codes = self.index.global_codes(position, table.column(GENE_COLUMN))
                for code in wanted_set.intersection(np.unique(row_codes).tolist()):
                    rows = np.flatnonzero(row_codes == code)
                    found.setdefault(int(code), []).append(self._convert(table.take(rows), columns, position))
            return found

        parts: dict[int, list[dict[str, np.ndarray]]] = {c: [] for c in wanted}

        def consume(_position: int, found: dict[int, list[dict[str, np.ndarray]]]) -> None:
            for code, items in found.items():
                parts[code].extend(items)

        _map_files(work, sorted(plan), cancel, consume)
        result: dict[str, dict[str, np.ndarray]] = {}
        for name in names:
            code = codes[name]
            result[name] = self._concat(parts.get(code, []) if code is not None else [], columns)
        return result

    # -- tile / window reads ----------------------------------------------

    def _require_index(self) -> None:
        if not self.index.validated:
            raise RuntimeError("tile and window reads need a validated Atera points index")

    def read_tile(
        self,
        tile: str,
        columns: Sequence[str] = DEFAULT_READ_COLUMNS,
        cancel: Any = None,
    ) -> dict[str, np.ndarray]:
        """Every row of one tile (all of its files, in index order)."""
        self._require_index()
        columns = self._columns(columns)
        positions = [f.position for f in self.index.files if f.tile == str(tile)]
        return self._read_files({p: None for p in positions}, columns, cancel, rect=None, codes=None)

    def files_in_window(self, bounds_um: Sequence[float]) -> list[int]:
        """File positions whose box meets ``[xmin, xmax) x [ymin, ymax)``."""
        xmin, ymin, xmax, ymax = (float(v) for v in bounds_um)
        out = []
        for item in self.index.files:
            if item.n_rows == 0:
                continue
            box = item.bbox_um
            if box is None:
                out.append(item.position)
                continue
            bx0, bx1, by0, by1 = box
            if bx0 < xmax and bx1 >= xmin and by0 < ymax and by1 >= ymin:
                out.append(item.position)
        return out

    def read_window(
        self,
        bounds_um: Sequence[float],
        genes: Iterable[str] | None = None,
        columns: Sequence[str] = DEFAULT_READ_COLUMNS,
        cancel: Any = None,
        *,
        clip: bool = True,
    ) -> WindowRead | None:
        """Rows of the tiles meeting ``bounds_um = (xmin, ymin, xmax, ymax)`` (um).

        Reads are tile-granular (files are cut by gene within a tile).  With
        ``clip`` only rows inside the half-open rectangle are returned; without
        it every row of each touched tile.  ``genes`` restricts to those
        features.  ``None`` without a validated index (window reads disabled).
        """
        if not self.window_reads_available:
            return None
        columns = self._columns([X_COLUMN, Y_COLUMN, *columns])
        positions = self.files_in_window(bounds_um)
        tiles = tuple(dict.fromkeys(self.index.files[p].tile for p in positions if self.index.files[p].tile is not None))
        if not clip:
            positions = [f.position for f in self.index.files if f.tile in tiles or (f.position in positions)]
        codes: list[int] | None = None
        selection: dict[int, dict[int, list[tuple[int, int, int]] | None] | None]
        if genes is not None:
            codes = sorted({c for c in (self.index.code_of(g) for g in genes) if c is not None})
            plan = self.plan_gene_reads(codes, self.code_map)
            selection = {p: plan[p] for p in positions if p in plan}
        else:
            selection = {p: None for p in positions}
        rect = tuple(float(v) for v in bounds_um) if clip else None
        data = self._read_files(selection, columns, cancel, rect=rect, codes=codes)
        return WindowRead(tiles=tiles, columns=data, files_read=len(selection))

    def _read_files(
        self,
        selection: Mapping[int, Mapping[int, list[tuple[int, int, int]] | None] | None],
        columns: list[str],
        cancel: Any,
        *,
        rect: tuple[float, ...] | None,
        codes: Sequence[int] | None,
    ) -> dict[str, np.ndarray]:
        need = list(columns)
        if rect is not None:
            for name in (X_COLUMN, Y_COLUMN):
                if name not in need:
                    need.append(name)
        code_filter = None if codes is None else np.asarray(sorted(codes), dtype=np.int64)

        def work(position: int) -> list[dict[str, np.ndarray]]:
            parquet_file = self._parquet(position)
            groups = selection[position]
            metadata = self.index.metadata[position]
            row_groups = sorted(groups) if groups is not None else list(range(metadata.num_row_groups))
            out = []
            for rg in row_groups:
                if rect is not None and not _row_group_meets(metadata, rg, rect):
                    continue
                runs = None if groups is None else groups[rg]
                read_columns = need if (runs is not None or code_filter is None or GENE_COLUMN in need) else [*need, GENE_COLUMN]
                table = _read_row_group(parquet_file, rg, read_columns)
                pieces = [table] if runs is None else [table.slice(s, e - s) for _c, s, e in runs]
                for piece in pieces:
                    if runs is None and code_filter is not None:
                        row_codes = self.index.global_codes(position, piece.column(GENE_COLUMN))
                        piece = piece.take(np.flatnonzero(np.isin(row_codes, code_filter)))
                    if rect is not None and piece.num_rows:
                        xs = piece.column(X_COLUMN).to_numpy()
                        ys = piece.column(Y_COLUMN).to_numpy()
                        inside = (xs >= rect[0]) & (xs < rect[2]) & (ys >= rect[1]) & (ys < rect[3])
                        if not inside.all():
                            piece = piece.filter(pa.array(inside))
                    if piece.num_rows:
                        out.append(self._convert(piece, columns, position))
            return out

        parts: list[dict[str, np.ndarray]] = []
        _map_files(work, sorted(selection), cancel, lambda _p, found: parts.extend(found))
        return self._concat(parts, columns)

    # -- whole-slide sample ----------------------------------------------

    def _sample_block(
        self,
        table: pa.Table,
        codes: np.ndarray,
        ids: np.ndarray | None,
        keep: np.ndarray | None,
        position: int,
        row_start: int,
        has_assignment: bool,
    ) -> dict[str, np.ndarray]:
        """The kept rows of one row group as compact sample columns."""
        index = self.index
        n = len(codes)
        rows = None if keep is None else np.flatnonzero(keep)
        take = (lambda a: a) if rows is None else (lambda a: a[rows])
        xs = np.asarray(table.column(X_COLUMN).to_numpy(), dtype=np.float32)
        ys = np.asarray(table.column(Y_COLUMN).to_numpy(), dtype=np.float32)
        if has_assignment:
            if ids is None:
                column = table.column(ASSIGNMENT_COLUMN)
                if rows is not None:
                    column = column.take(pa.array(rows))
                picked_ids = np.asarray(pc.fill_null(column, 0).to_numpy())
            else:
                picked_ids = take(ids)
        else:
            picked_ids = np.zeros(n if rows is None else len(rows), dtype=index.assignment_dtype)
        keys = row_sample_keys(position, row_start, n)
        return {
            "x": take(xs),
            "y": take(ys),
            "code": take(codes).astype(index.code_dtype),
            "cell_id": np.asarray(picked_ids).astype(index.assignment_dtype, copy=False),
            "key": take(keys),
        }

    def _merge_sample(self, parts: list[dict[str, np.ndarray]], threshold: int) -> SampledTranscripts:
        index = self.index
        dtypes = {
            "x": np.float32, "y": np.float32, "code": index.code_dtype,
            "cell_id": index.assignment_dtype, "key": np.uint32,
        }
        merged: dict[str, np.ndarray] = {}
        for name, dtype in dtypes.items():
            total = sum(len(p[name]) for p in parts)
            out = np.empty(total, dtype=dtype)
            cursor = 0
            for part in parts:
                values = part.pop(name)
                out[cursor : cursor + len(values)] = values
                cursor += len(values)
            merged[name] = out
        return SampledTranscripts(threshold=int(threshold), total_rows=index.n_rows, **merged)

    def sample_rows(
        self,
        threshold: int,
        cancel: Any = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> SampledTranscripts:
        """Stream every file and keep the rows whose sample key is below ``threshold``."""
        index = self.index
        has_assignment = all(_column_index(m, ASSIGNMENT_COLUMN) is not None for m in index.metadata)
        read_columns = [X_COLUMN, Y_COLUMN, GENE_COLUMN] + ([ASSIGNMENT_COLUMN] if has_assignment else [])
        threshold = int(threshold)

        def work(position: int) -> list[dict[str, np.ndarray]]:
            parquet_file = self._parquet(position)
            out = []
            row0 = 0
            for rg in range(index.metadata[position].num_row_groups):
                n_rg = int(index.metadata[position].row_group(rg).num_rows)
                start = row0
                row0 += n_rg
                if n_rg == 0 or threshold <= 0:
                    continue
                table = _read_row_group(parquet_file, rg, read_columns)
                codes = index.global_codes(position, table.column(GENE_COLUMN))
                keep = _keep_mask(position, start, n_rg, threshold)
                if table.column(GENE_COLUMN).null_count:
                    valid = codes >= 0
                    keep = valid if keep is None else (keep & valid)
                out.append(self._sample_block(table, codes, None, keep, position, start, has_assignment))
            return out

        parts: list[dict[str, np.ndarray]] = []
        done = 0
        n_files = len(index.files)

        def consume(_position: int, found: list[dict[str, np.ndarray]]) -> None:
            nonlocal done
            parts.extend(found)
            done += 1
            if progress is not None:
                progress(done, n_files)

        _map_files(work, range(n_files), cancel, consume)
        return self._merge_sample(parts, threshold)

    def read_window_rows(
        self,
        bounds_um: Sequence[float],
        codes: Iterable[int] | None = None,
        cancel: Any = None,
        *,
        exclude_below: int = 0,
        clip: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """Rows of ``codes`` in the window whose sample key is NOT below ``exclude_below``.

        This is what a viewport adds on top of a whole-slide sample of the same
        threshold: the rows the sample left out, never a row twice.  Returns
        ``x``, ``y``, ``code``, ``cell_id`` and ``key``, or ``None`` without a
        validated index.
        """
        if not self.window_reads_available:
            return None
        index = self.index
        exclude_below = int(exclude_below)
        if exclude_below >= SAMPLE_KEY_SPACE:
            empty = self._merge_sample([], 0)
            return {"x": empty.x, "y": empty.y, "code": empty.code, "cell_id": empty.cell_id, "key": empty.key}
        positions = self.files_in_window(bounds_um)
        rect = tuple(float(v) for v in bounds_um) if clip else None
        wanted = None if codes is None else sorted({int(c) for c in codes})
        plan = None if wanted is None else self.plan_gene_reads(wanted, self.code_map)
        code_filter = None if wanted is None else np.asarray(wanted, dtype=np.int64)
        has_assignment = all(_column_index(m, ASSIGNMENT_COLUMN) is not None for m in index.metadata)
        read_columns = [X_COLUMN, Y_COLUMN, GENE_COLUMN] + ([ASSIGNMENT_COLUMN] if has_assignment else [])
        if plan is not None:
            positions = [p for p in positions if p in plan]

        def work(position: int) -> list[dict[str, np.ndarray]]:
            parquet_file = self._parquet(position)
            metadata = index.metadata[position]
            starts = np.concatenate([[0], np.cumsum([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)])])
            groups = None if plan is None else plan[position]
            row_groups = sorted(groups) if groups is not None else list(range(metadata.num_row_groups))
            out = []
            for rg in row_groups:
                if rect is not None and not _row_group_meets(metadata, rg, rect):
                    continue
                table = _read_row_group(parquet_file, rg, read_columns)
                runs = None if groups is None else groups[rg]
                pieces = [(table, 0)] if runs is None else [(table.slice(s, e - s), s) for _c, s, e in runs]
                for piece, offset in pieces:
                    n = piece.num_rows
                    if not n:
                        continue
                    codes_piece = index.global_codes(position, piece.column(GENE_COLUMN))
                    keep = codes_piece >= 0
                    if runs is None and code_filter is not None:
                        keep &= np.isin(codes_piece, code_filter)
                    if rect is not None:
                        xs = piece.column(X_COLUMN).to_numpy()
                        ys = piece.column(Y_COLUMN).to_numpy()
                        keep &= (xs >= rect[0]) & (xs < rect[2]) & (ys >= rect[1]) & (ys < rect[3])
                    if exclude_below > 0:
                        keys = row_sample_keys(position, int(starts[rg]) + int(offset), n)
                        keep &= keys >= np.uint32(exclude_below)
                    if keep.any():
                        out.append(self._sample_block(piece, codes_piece, None, keep, position, int(starts[rg]) + int(offset), has_assignment))
            return out

        parts: list[dict[str, np.ndarray]] = []
        _map_files(work, positions, cancel, lambda _p, found: parts.extend(found))
        sample = self._merge_sample(parts, 0)
        return {"x": sample.x, "y": sample.y, "code": sample.code, "cell_id": sample.cell_id, "key": sample.key}

    # -- code map and overview --------------------------------------------

    def build_code_map(self, cancel: Any = None) -> CodeRangeMap | None:
        """Decode every ``feature_name`` column once and attach the map."""
        builder = _CodeMapBuilder(self.index)

        def work(position: int) -> list[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            parquet_file = self._parquet(position)
            out = []
            for rg in range(self.index.metadata[position].num_row_groups):
                table = _read_row_group(parquet_file, rg, [GENE_COLUMN])
                codes = self.index.global_codes(position, table.column(0))
                out.append((int(self.index.rg_offsets[position] + rg), _code_runs(codes)))
            return out

        def consume(_position: int, found) -> None:
            for global_rg, runs in found:
                builder.add(global_rg, *runs)

        _map_files(work, range(len(self.index.files)), cancel, consume)
        code_map = builder.finish()
        self.attach_code_map(code_map)
        return code_map

    def build_overview(
        self,
        bin_um: float = DEFAULT_OVERVIEW_BIN_UM,
        cancel: Any = None,
        *,
        sample_size: int = 0,
        seed: int = 0,
        split_assigned: bool = False,
        progress: Callable[[int, int], None] | None = None,
        keep_threshold: int | None = None,
    ) -> AteraOverview:
        """One pass: exact counts, assigned counts, density raster(s), code map.

        Controls come from ``is_gene`` when the files carry it (via
        :func:`control_genes_from_is_gene`), else from the name rule; they are
        counted but excluded from the density.  ``sample_size > 0`` keeps a
        seeded per-file Bernoulli sample at rate ``sample_size / total`` (per
        file and row group, so it is deterministic whatever the scheduling),
        then drops controls and trims to at most ``sample_size`` rows.

        ``keep_threshold`` (see :func:`sample_threshold`) also keeps, in the
        same pass, the whole-slide hash sample of every row (controls included)
        whose key is below it, returned as ``overview.hash_sample``.
        """
        bin_um = float(bin_um)
        if not bin_um > 0:
            raise ValueError("bin_um must be positive")
        index = self.index
        n_codes = len(index.vocabulary)
        extent = index.extent_um() or (0.0, 0.0, 0.0, 0.0)
        x0, y0, width, height = _grid(extent, bin_um)
        density = np.zeros((height, width), dtype=np.uint32)
        assigned_density = np.zeros((height, width), dtype=np.uint32) if split_assigned else None
        counts = np.zeros(n_codes, dtype=np.int64)
        assigned = np.zeros(n_codes, dtype=np.int64)
        not_gene = np.zeros(n_codes, dtype=np.int64)
        name_control = np.array([is_control_gene(name) for name in index.vocabulary], dtype=bool)
        total = index.n_rows
        rate = 0.0 if sample_size <= 0 or total == 0 else min(1.0, float(sample_size) / float(total))
        builder = _CodeMapBuilder(index)
        sample_parts: list[dict[str, np.ndarray]] = []
        has_is_gene = all(_column_index(m, IS_GENE_COLUMN) is not None for m in index.metadata) and bool(index.metadata)
        has_assignment = all(_column_index(m, ASSIGNMENT_COLUMN) is not None for m in index.metadata)
        read_columns = [X_COLUMN, Y_COLUMN, GENE_COLUMN]
        if has_assignment:
            read_columns.append(ASSIGNMENT_COLUMN)
        if has_is_gene:
            read_columns.append(IS_GENE_COLUMN)
        null_rows = 0
        kept_parts: list[dict[str, np.ndarray]] = []

        def work(position: int) -> dict[str, Any]:
            parquet_file = self._parquet(position)
            rng = np.random.default_rng([int(seed), int(position)]) if rate > 0 else None
            f_counts = np.zeros(n_codes, dtype=np.int64)
            f_assigned = np.zeros(n_codes, dtype=np.int64)
            f_not_gene = np.zeros(n_codes, dtype=np.int64)
            blocks: list[tuple[int, int, np.ndarray]] = []
            a_blocks: list[tuple[int, int, np.ndarray]] = []
            runs = []
            samples = []
            kept: list[dict[str, np.ndarray]] = []
            nulls = 0
            metadata = index.metadata[position]
            assignment_index = _column_index(metadata, ASSIGNMENT_COLUMN)
            row0 = 0
            for rg in range(metadata.num_row_groups):
                table = _read_row_group(parquet_file, rg, read_columns)
                codes = index.global_codes(position, table.column(GENE_COLUMN))
                n = len(codes)
                rg_row0 = row0
                row0 += n
                # Per-run sums replace per-row bincounts: codes come in runs.
                run_codes, starts, ends = _code_runs(codes)
                runs.append((int(index.rg_offsets[position] + rg), (run_codes, starts, ends)))
                np.add.at(f_counts, run_codes, ends - starts)
                valid = None
                if table.column(GENE_COLUMN).null_count:
                    valid = codes >= 0
                    nulls += int(n - valid.sum())
                ids = None
                is_assigned = None
                if has_assignment:
                    column = table.column(ASSIGNMENT_COLUMN)
                    is_assigned = np.asarray(column.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
                    if _may_hold_zero(metadata, rg, assignment_index):
                        ids = np.asarray(pc.fill_null(column, 0).to_numpy())
                        is_assigned &= ids != 0
                    np.add.at(f_assigned, run_codes, _run_sums(is_assigned, starts, ends, contiguous=valid is None))
                real = None
                if has_is_gene:
                    flags = np.asarray(
                        pc.fill_null(table.column(IS_GENE_COLUMN), False).to_numpy(zero_copy_only=False), dtype=bool
                    )
                    if not flags.all():
                        np.add.at(f_not_gene, run_codes, _run_sums(~flags, starts, ends, contiguous=valid is None))
                        real = flags
                elif name_control[run_codes].any():
                    real = np.ones(n, dtype=bool)
                    for code, start, end in zip(run_codes, starts, ends):
                        if name_control[code]:
                            real[start:end] = False
                if valid is not None:
                    real = valid if real is None else (real & valid)
                xs = table.column(X_COLUMN).to_numpy()
                ys = table.column(Y_COLUMN).to_numpy()
                if real is None:
                    block = _bin_block(xs, ys, x0, y0, bin_um, width, height)
                else:
                    block = _bin_block(xs[real], ys[real], x0, y0, bin_um, width, height)
                if block is not None:
                    blocks.append(block)
                if split_assigned:
                    mask = is_assigned if is_assigned is not None else np.zeros(n, dtype=bool)
                    if real is not None:
                        mask = mask & real
                    a_block = _bin_block(xs[mask], ys[mask], x0, y0, bin_um, width, height)
                    if a_block is not None:
                        a_blocks.append(a_block)
                if keep_threshold is not None and keep_threshold > 0 and n:
                    keep = _keep_mask(position, rg_row0, n, keep_threshold)
                    if valid is not None:
                        keep = valid if keep is None else (keep & valid)
                    kept.append(self._sample_block(table, codes, ids, keep, position, rg_row0, has_assignment))
                if rng is not None:
                    pick = rng.random(n) < rate
                    if valid is not None:
                        pick &= valid
                    if pick.any():
                        if has_assignment and ids is None:
                            picked_ids = np.asarray(
                                pc.fill_null(table.column(ASSIGNMENT_COLUMN).take(np.flatnonzero(pick)), 0).to_numpy()
                            )
                        elif ids is not None:
                            picked_ids = ids[pick]
                        else:
                            picked_ids = np.zeros(int(pick.sum()), dtype=np.uint64)
                        samples.append(
                            {
                                "x": np.asarray(xs[pick], dtype=np.float32),
                                "y": np.asarray(ys[pick], dtype=np.float32),
                                "code": codes[pick].astype(np.int32),
                                ASSIGNMENT_COLUMN: picked_ids.astype(index.assignment_dtype),
                            }
                        )
            return {
                "counts": f_counts, "assigned": f_assigned, "not_gene": f_not_gene,
                "blocks": blocks, "a_blocks": a_blocks, "runs": runs, "samples": samples, "nulls": nulls,
                "kept": kept,
            }

        done = 0
        n_files = len(index.files)

        def consume(_position: int, result: dict[str, Any]) -> None:
            nonlocal done, null_rows
            np.add(counts, result["counts"], out=counts)
            np.add(assigned, result["assigned"], out=assigned)
            np.add(not_gene, result["not_gene"], out=not_gene)
            null_rows += result["nulls"]
            for r0, c0, block in result["blocks"]:
                density[r0 : r0 + block.shape[0], c0 : c0 + block.shape[1]] += block.astype(np.uint32)
            if assigned_density is not None:
                for r0, c0, block in result["a_blocks"]:
                    assigned_density[r0 : r0 + block.shape[0], c0 : c0 + block.shape[1]] += block.astype(np.uint32)
            for global_rg, runs in result["runs"]:
                builder.add(global_rg, *runs)
            sample_parts.extend(result["samples"])
            kept_parts.extend(result["kept"])
            done += 1
            if progress is not None:
                progress(done, n_files)

        _map_files(work, range(n_files), cancel, consume)

        if has_is_gene:
            flags = {
                index.vocabulary[i]: bool(not_gene[i] == 0)
                for i in np.flatnonzero(counts > 0)
            }
            controls = control_genes_from_is_gene(index.vocabulary, flags, source=str(index.zarr_path))
            is_control = np.array([name in controls for name in index.vocabulary], dtype=bool)
        else:
            is_control = name_control
        sample = None
        if rate > 0:
            if sample_parts:
                merged = {
                    key: np.concatenate([p[key] for p in sample_parts])
                    for key in ("x", "y", "code", ASSIGNMENT_COLUMN)
                }
                keep = ~is_control[merged["code"]]
            else:
                merged = {
                    "x": np.zeros(0, np.float32), "y": np.zeros(0, np.float32),
                    "code": np.zeros(0, np.int32), ASSIGNMENT_COLUMN: np.zeros(0, index.assignment_dtype),
                }
                keep = np.zeros(0, dtype=bool)
            rows = np.flatnonzero(keep)
            if len(rows) > sample_size:
                rows = rows[sample_global_positions(len(rows), sample_size, seed)]
            sample = {key: value[rows] for key, value in merged.items()}
            sample["code"] = sample["code"].astype(index.code_dtype)
        hash_sample = None
        if keep_threshold is not None:
            hash_sample = self._merge_sample(kept_parts, int(keep_threshold))
            kept_parts.clear()
        code_map = builder.finish()
        self.attach_code_map(code_map)
        if int(counts.sum()) + null_rows != total:
            log.warning("[ATERA] Overview counted %d rows; the footers say %d.", int(counts.sum()) + null_rows, total)
        return AteraOverview(
            vocabulary=index.vocabulary,
            counts=counts,
            assigned_counts=assigned,
            is_control=is_control,
            density_counts=density,
            origin_um=(x0, y0),
            bin_um=bin_um,
            total_rows=total,
            fingerprint=dict(index.fingerprint),
            code_map=code_map,
            assigned_density_counts=assigned_density,
            sample=sample,
            sample_size=int(max(0, sample_size)),
            sample_seed=int(seed),
            is_gene_flags_used=bool(has_is_gene),
            hash_sample=hash_sample,
        )


def _row_group_meets(metadata: pq.FileMetaData, rg: int, rect: Sequence[float]) -> bool:
    """False only when the x/y statistics prove the row group misses ``rect``."""
    group = metadata.row_group(rg)
    if group.num_rows == 0:
        return False
    xi = _column_index(metadata, X_COLUMN)
    yi = _column_index(metadata, Y_COLUMN)
    if xi is None or yi is None:
        return True
    xs, ys = group.column(xi).statistics, group.column(yi).statistics
    if xs is None or ys is None or not xs.has_min_max or not ys.has_min_max:
        return True
    return bool(float(xs.min) < rect[2] and float(xs.max) >= rect[0] and float(ys.min) < rect[3] and float(ys.max) >= rect[1])


def index_content_digest(index: AteraPointsIndex) -> str | None:
    """blake2s of the index JSON bytes (``None`` for a scanned index)."""
    if index.index_path is None:
        return None
    try:
        return blake2s(index.index_path.read_bytes(), digest_size=20).hexdigest()
    except OSError:
        return None


def file_stat_identity(index: AteraPointsIndex) -> list[dict[str, Any]] | None:
    """Per file ``size`` and ``mtime_ns`` (ctime is excluded: hardlinks change it)."""
    out = []
    for item in index.files:
        try:
            stat = os.stat(item.path)
        except OSError:
            return None
        out.append({"file": item.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return out


# ---------------------------------------------------------------------------
# Gene point stores for the viewer
# ---------------------------------------------------------------------------


def overview_panel(overview: AteraOverview) -> tuple[list[str], np.ndarray, dict[str, int], set[str]]:
    """``(genes, lut, source_counts, controls)`` of the viewer's gene panel.

    ``genes`` are the observed features in alphabetical order (what
    :func:`~.utils.build_gene_point_groups` lists), ``lut`` maps a global code
    to its panel index (-1 for unobserved), ``source_counts`` are the exact
    per-feature totals and ``controls`` the observed control features (from
    ``is_gene`` when the points carry it).
    """
    observed = np.flatnonzero(np.asarray(overview.counts) > 0)
    names = [str(overview.vocabulary[i]) for i in observed]
    order = sorted(range(len(names)), key=lambda i: names[i])
    genes = [names[i] for i in order]
    lut = np.full(len(overview.vocabulary), -1, dtype=np.int64)
    for panel_index, i in enumerate(order):
        lut[int(observed[i])] = panel_index
    counts = {str(overview.vocabulary[i]): int(overview.counts[i]) for i in observed}
    controls = {str(overview.vocabulary[i]) for i in observed if bool(overview.is_control[i])}
    return genes, lut, counts, controls


def gene_point_store_from_rows(
    overview: AteraOverview,
    x: np.ndarray,
    y: np.ndarray,
    code: np.ndarray,
    cell_id: np.ndarray,
    *,
    sampled: bool,
    gene_visuals: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    reference_resolver: Callable[[list[str]], Any] | None = None,
    alpha: float = 1.0,
):
    """A :class:`~.utils.GenePointStore` over compact Atera rows (vectorised).

    Every observed feature is listed with its exact count, whether or not
    ``code`` holds rows of it; unassigned rows (``cell_id == 0``) are the
    background.
    """
    from .utils import assemble_gene_point_store

    genes, lut, counts, controls = overview_panel(overview)
    if reference is None and reference_resolver is not None:
        reference = reference_resolver(genes)
    panel_dtype = np.uint16 if len(genes) <= np.iinfo(np.uint16).max + 1 else np.uint32
    panel_index = lut[np.asarray(code, dtype=np.int64)] if len(code) else np.zeros(0, np.int64)
    if len(panel_index) and int(panel_index.min()) < 0:
        raise ValueError("rows hold features the overview never counted")
    panel_codes = panel_index.astype(panel_dtype)
    del panel_index
    return assemble_gene_point_store(
        genes,
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        panel_codes,
        np.asarray(cell_id) == 0,
        gene_visuals=gene_visuals,
        reference=reference,
        alpha=alpha,
        control_genes=controls,
        sampled=bool(sampled),
        source_gene_counts=counts,
        source_total_points=int(np.asarray(overview.counts).sum()),
    )
