"""Store I/O of the manual-registration feature: root-attrs reads, write lock, save, drafts.

This module reads the root attributes of a manual-registration Zarr store as metadata only,
serialises writes with the same lock file MerXen uses, saves the landmark labelling payload
(revision check, history backup, read-back verification) and keeps local drafts. It must not
import Qt (it has no GUI dependencies) and never opens image arrays or SpatialData.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import zarr

from .manual_registration_model import (
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    MIN_LANDMARK_PAIRS,
    SCHEMA_VERSION,
    SOFTWARE_NAME,
    STORE_ATTR,
    NotARegistrationStore,
    StoreDescriptor,
    UnsupportedSchemaVersion,
    canonical_json,
    labelling_content_sha256,
    parse_store_descriptor,
    utc_now,
)

try:
    import fcntl
except ImportError:  # Windows builds of the viewer
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

LOCK_SUFFIX = ".merxen-write.lock"
HISTORY_SUFFIX = ".landmark-history"
DRAFT_DIR_NAME = "manual_registration_drafts"
DRAFT_FORMAT = 1

_NO_LOCK_LOGGED = False


class StoreChangedError(RuntimeError):
    """The store changed on disk since it was loaded (another session or a rebuild)."""


class StoreBusyError(RuntimeError):
    """The store write lock is held by another program."""


def write_lock_path(path: Path) -> Path:
    """Return the lock file of a store, the one MerXen locks (``<name>.merxen-write.lock``)."""
    resolved = Path(path).resolve()
    return resolved.with_name(resolved.name + LOCK_SUFFIX)


def history_dir(path: Path) -> Path:
    """Return the directory that keeps backups of overwritten labellings of a store."""
    resolved = Path(path).resolve()
    return resolved.with_name(resolved.name + HISTORY_SUFFIX)


def read_root_attrs(path: Path) -> dict[str, Any]:
    """Read the root attributes of a Zarr store as plain JSON (metadata only).

    Args:
        path: The store directory (Zarr v3 ``zarr.json`` or v2 ``.zattrs``).

    Returns:
        The root attributes (an empty dict for a v2 group without attributes).

    Raises:
        FileNotFoundError: If ``path`` is not a directory.
        ValueError: If ``zarr.json`` cannot be parsed.
        NotARegistrationStore: If the directory is not a Zarr group.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist or is not a directory")
    zarr_json = path / "zarr.json"
    if zarr_json.exists():
        try:
            meta = json.loads(zarr_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable zarr.json in {path.name}: {exc}") from exc
        attrs = meta.get("attributes") if isinstance(meta, dict) else None
        return dict(attrs or {})
    zattrs = path / ".zattrs"
    if zattrs.exists():
        try:
            data = json.loads(zattrs.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable .zattrs in {path.name}: {exc}") from exc
        return dict(data) if isinstance(data, dict) else {}
    if (path / ".zgroup").exists():
        return {}
    raise NotARegistrationStore(f"{path.name} is not a Zarr store")


def read_store_descriptor(path: Path) -> StoreDescriptor:
    """Read and parse the store descriptor of a manual-registration store."""
    return parse_store_descriptor(Path(path), read_root_attrs(path))


@contextmanager
def store_write_lock(path: Path, *, timeout_s: float = 10.0) -> Iterator[None]:
    """Hold the store's write lock (the file MerXen locks) for the duration of the block.

    The lock polls every 50 ms until ``timeout_s`` has passed. It is **not re-entrant**: a
    nested call on the same store from the same thread times out with ``StoreBusyError``. The
    lock file is never deleted. Without ``fcntl`` (Windows) no lock is taken.

    Raises:
        StoreBusyError: If another program holds the lock for longer than ``timeout_s``.
        OSError: If the lock file cannot be opened (for example a read-only location).
    """
    global _NO_LOCK_LOGGED
    if fcntl is None:
        if not _NO_LOCK_LOGGED:
            log.debug("fcntl is unavailable; store writes are not serialised with MerXen")
            _NO_LOCK_LOGGED = True
        yield
        return
    lock = write_lock_path(path)
    with lock.open("a", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise StoreBusyError(
                        f"The store is busy: another program holds {lock.name}. "
                        "Try again in a moment."
                    ) from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != LABELLING_SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("payload is not a version-1 landmark labelling")
    status = payload.get("status")
    if status not in ("draft", "finished"):
        raise ValueError(f"status must be 'draft' or 'finished', got {status!r}")
    if status == "finished":
        pairs = (payload.get("landmarks") or {}).get("pairs") or []
        n_enabled = sum(1 for p in pairs if p.get("enabled", True))
        if n_enabled < MIN_LANDMARK_PAIRS:
            raise ValueError(f"Finish needs at least {MIN_LANDMARK_PAIRS} enabled landmark pairs")
    result: dict[str, Any] = json.loads(canonical_json(payload))  # NaN / inf -> ValueError
    return result


def _write_backup(path: Path, current: dict[str, Any], on_disk: Any) -> None:
    """Best-effort backup of the labelling that is about to be overwritten."""
    try:
        directory = history_dir(path)
        directory.mkdir(exist_ok=True)
        stem = f"{on_disk}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        target = directory / f"{stem}.json"
        n = 0
        while target.exists():
            n += 1
            target = directory / f"{stem}-{n}.json"
        target.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.warning("could not back up the previous landmarks: %s", exc)


def save_labelling(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_revision: int | None,
    software_version: str,
) -> dict[str, Any]:
    """Save the labelling payload into a prepared store (the only key this viewer writes).

    The payload is validated first. Under the store write lock the root attributes are
    re-read, the store kind, grid digest and on-disk revision are checked, the previous
    labelling is backed up (best effort), ``revision``, ``updated_at``, ``finished_at``,
    ``written_by`` and ``content_sha256`` are filled in, the attribute is written (other root
    keys and the consolidated metadata stay) and read back for verification.

    Args:
        path: The prepared store directory.
        payload: A payload built by ``labelling_to_attrs``.
        expected_revision: The revision the caller loaded (``None`` for a store without a
            labelling).
        software_version: The viewer version for ``written_by``.

    Returns:
        The payload exactly as written.

    Raises:
        ValueError: On an invalid payload (nothing is written).
        NotARegistrationStore: If the store is not a prepared manual-registration store.
        StoreChangedError: If the store was rebuilt or saved by another session.
        UnsupportedSchemaVersion: If the labelling on disk is newer than this viewer reads.
        StoreBusyError: If the write lock cannot be taken.
        RuntimeError: If the read-back differs from what was written.
    """
    path = Path(path)
    out = _validate_payload(payload)
    with store_write_lock(path):
        read_root_attrs(path)  # never create a store: raises for a missing or non-zarr path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", zarr.errors.ZarrUserWarning)  # v3 consolidated metadata
            root = zarr.open_group(str(path), mode="a")
        attrs = json.loads(json.dumps(dict(root.attrs)))
        sa = attrs.get(STORE_ATTR)
        if not isinstance(sa, dict) or sa.get("kind") != "prepared":
            raise NotARegistrationStore(
                "landmarks can only be saved into a prepared manual-registration store"
            )
        stored_digest = out.get("store_grid_digest")
        if stored_digest and sa.get("grid_digest") and stored_digest != sa.get("grid_digest"):
            raise StoreChangedError(
                "The prepared store was rebuilt from different images since it was loaded. "
                "Reload it."
            )
        current = attrs.get(LABELLING_ATTR)
        if (
            isinstance(current, dict)
            and isinstance(current.get("schema_version"), int)
            and current["schema_version"] > SCHEMA_VERSION
        ):
            raise UnsupportedSchemaVersion(
                f"landmark labelling schema_version {current['schema_version']} was written by "
                f"a newer MerXen/viewer; update this viewer (it reads version {SCHEMA_VERSION})"
            )
        on_disk = current.get("revision") if isinstance(current, dict) else None
        if on_disk != expected_revision:
            raise StoreChangedError(
                f"The landmarks in this store were saved by another session (revision {on_disk} "
                f"on disk, {expected_revision} when loaded). Reload the store (discards local "
                "changes) or cancel."
            )
        if isinstance(current, dict):
            _write_backup(path, current, on_disk)
        now = utc_now()
        out["revision"] = (expected_revision or 0) + 1
        out["updated_at"] = now
        out["finished_at"] = (
            now
            if out["status"] == "finished"
            else (current.get("finished_at") if isinstance(current, dict) else None)
        )
        out["written_by"] = {"software": SOFTWARE_NAME, "version": software_version}
        out["content_sha256"] = labelling_content_sha256(out)
        root.attrs.update({LABELLING_ATTR: out})
        back = read_root_attrs(path).get(LABELLING_ATTR)
        if back is None or canonical_json(back) != canonical_json(out):
            raise RuntimeError(
                "The saved landmarks could not be read back identically; the store was not "
                "updated reliably."
            )
    return out


# ----------------------------------------------------------------------------------------------
# Local drafts
# ----------------------------------------------------------------------------------------------


def draft_path(store: Path, base_dir: Path) -> Path:
    """Return the draft file of a store, keyed by its resolved path."""
    key = hashlib.sha1(str(Path(store).resolve()).encode("utf-8")).hexdigest()
    return Path(base_dir) / DRAFT_DIR_NAME / f"{key}.json"


def write_local_draft(store: Path, base_dir: Path, payload: dict[str, Any]) -> None:
    """Write the local draft of a store atomically.

    Raises:
        OSError: If the draft cannot be written.
        ValueError: If the payload is not JSON (NaN or infinity).
    """
    target = draft_path(store, base_dir)
    text = json.dumps(
        {
            "draft_format": DRAFT_FORMAT,
            "store_path": str(Path(store).resolve()),
            "saved_at": utc_now(),
            "payload": payload,
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_local_draft(store: Path, base_dir: Path) -> dict[str, Any] | None:
    """Return the local draft of a store, or ``None`` if missing, corrupt or for another store."""
    target = draft_path(store, base_dir)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("could not read the local draft %s: %s", target.name, exc)
        return None
    try:
        draft = json.loads(text)
    except ValueError as exc:
        log.warning("ignoring a corrupt local draft %s: %s", target.name, exc)
        return None
    if (
        not isinstance(draft, dict)
        or draft.get("draft_format") != DRAFT_FORMAT
        or not isinstance(draft.get("payload"), dict)
        or draft.get("store_path") != str(Path(store).resolve())
    ):
        return None
    return draft


def clear_local_draft(store: Path, base_dir: Path) -> None:
    """Delete the local draft of a store (no error when there is none)."""
    draft_path(store, base_dir).unlink(missing_ok=True)
