"""Synthetic manual-registration stores and a fake host (shared by the controller and VW3 tests).

This is a plain helper module (no ``test_`` prefix, no pytest fixtures). pytest's default
``prepend`` import mode puts ``tests/`` on ``sys.path``, so test files import it with
``import manual_registration_stores as stores``.

All samples are synthetic: the dataset name is the alias ``SYN1`` and every path inside the
attributes is the placeholder ``"<synthetic>"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from napari_compare_xenium_merscope.manual_registration_model import (
    LABELLING_ATTR,
    LABELLING_SCHEMA,
    RESULT_ATTR,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    STORE_ATTR,
    STORE_SCHEMA,
    labelling_content_sha256,
)
from napari_compare_xenium_merscope.utils import ensure_cyx, get_scale0_dataarray

DATASET_NAME = "SYN1"  # synthetic alias (not a sample name)
REFERENCE_SHAPE_YX = (1100, 1000)
REFERENCE_PIXEL_SIZE_UM = 0.5
MOVING_SHAPE_YX = (960, 1100)
MOVING_PIXEL_SIZE_UM = 0.4
REFERENCE_CHANNELS = ("DAPI", "AF568")
MOVING_CHANNELS = ("DAPI", "AF568", "AF647")
RESULT_CROP = {"x0": 64, "y0": 128, "x1": 1000, "y1": 1088}  # effective crop
GRID_DIGEST = "sha256:" + "1" * 64
CONTENT_DIGEST = "sha256:" + "2" * 64
CREATED_AT = "2026-11-02T10:15:00Z"
PLACEHOLDER = "<synthetic>"


def synthetic_image(shape_yx: tuple[int, int], n_channels: int, *, seed: int) -> np.ndarray:
    """Return a ``(c, y, x)`` uint16 test image with a 16-px zero border on every side.

    Channel ``i`` is ``1000 + 600 sin(x/37 + i) cos(y/53)`` plus eight Gaussian blobs
    (sigma 6 px, +20000) at positions drawn from ``default_rng(seed)``.
    """
    height, width = shape_yx
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    border = 16
    blobs = [
        (
            rng.uniform(border + 20, height - border - 20),
            rng.uniform(border + 20, width - border - 20),
        )
        for _ in range(8)
    ]
    out = np.zeros((n_channels, height, width), dtype=np.uint16)
    for i in range(n_channels):
        plane = 1000.0 + 600.0 * np.sin(xx / 37.0 + i) * np.cos(yy / 53.0)
        for by, bx in blobs:
            plane += 20000.0 * np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / (2.0 * 6.0**2))
        plane[:border, :] = 0
        plane[-border:, :] = 0
        plane[:, :border] = 0
        plane[:, -border:] = 0
        out[i] = np.clip(plane, 0, 65535).astype(np.uint16)
    return out


def _level_shapes(shape_yx: tuple[int, int]) -> list[list[int]]:
    shapes = [[int(shape_yx[0]), int(shape_yx[1])]]
    while min(shapes[-1]) // 2 >= 256:
        shapes.append([shapes[-1][0] // 2, shapes[-1][1] // 2])
    return shapes


def _image_record(
    role: str,
    shape_yx: tuple[int, int],
    pixel_size_um: float,
    channels: Sequence[str],
) -> dict[str, Any]:
    return {
        "element": role,
        "role": role,
        "shape_yx": [int(shape_yx[0]), int(shape_yx[1])],
        "dtype": "uint16",
        "pixel_size_um": float(pixel_size_um),
        "pixel_size_source": "metadata",
        "channel_names": list(channels),
        "registration_channel": channels[0],
        "source_channel_names": list(channels),
        "canonical_channel_names": list(channels),
        "name_sources": ["Name"] * len(channels),
        "ome_tiff": {
            "path": PLACEHOLDER,
            "fingerprint": "sha256:" + ("3" if role == "reference" else "4") * 64,
            "written_by_prepare": False,
            "level_shapes_yx": _level_shapes(shape_yx),
            "exact_level_factors": False,
        },
        "source": {
            "format": "ome_tiff",
            "path": PLACEHOLDER,
            "fingerprint": "sha256:" + ("5" if role == "reference" else "6") * 64,
            "size_bytes": 0,
        },
        "czi": None,
    }


def prepared_store_attr(
    *,
    dataset_name: str = DATASET_NAME,
    reference_shape_yx: tuple[int, int] = REFERENCE_SHAPE_YX,
    moving_shape_yx: tuple[int, int] = MOVING_SHAPE_YX,
    reference_pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    moving_pixel_size_um: float = MOVING_PIXEL_SIZE_UM,
    reference_channels: Sequence[str] = REFERENCE_CHANNELS,
    moving_channels: Sequence[str] = MOVING_CHANNELS,
) -> dict[str, Any]:
    """Return a complete ``STORE_ATTR`` payload of kind ``"prepared"`` (master design 3.5)."""
    kept = [c for c in reference_channels if c in moving_channels]
    return {
        "schema": STORE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": "prepared",
        "dataset_name": dataset_name,
        "created_at": CREATED_AT,
        "merxen_version": "0.0.0+synthetic",
        "conventions": {
            "pixel": "level0_xy_integer_is_pixel_centre",
            "matrix": "moving_level0_px_xy_to_reference_level0_px_xy",
            "rotation": "positive_ccw_on_screen_y_down",
            "crop": "half_open_level0_px",
        },
        "images": {
            "reference": _image_record(
                "reference", reference_shape_yx, reference_pixel_size_um, reference_channels
            ),
            "moving": _image_record(
                "moving", moving_shape_yx, moving_pixel_size_um, moving_channels
            ),
        },
        "channel_matching": {
            "reference_kept": kept,
            "reference_dropped": [c for c in reference_channels if c not in kept],
            "moving_unmatched": [c for c in moving_channels if c not in reference_channels],
        },
        "grid_digest": GRID_DIGEST,
        "content_digest": CONTENT_DIGEST,
    }


def result_store_attr(
    *,
    crop: dict[str, int] = RESULT_CROP,
    preset: str = "serial_section",
    run_label: str | None = None,
    dataset_name: str = DATASET_NAME,
    reference_shape_yx: tuple[int, int] = REFERENCE_SHAPE_YX,
    moving_shape_yx: tuple[int, int] = MOVING_SHAPE_YX,
    reference_pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    moving_pixel_size_um: float = MOVING_PIXEL_SIZE_UM,
    reference_channels: Sequence[str] = REFERENCE_CHANNELS,
    moving_channels: Sequence[str] = MOVING_CHANNELS,
) -> dict[str, Any]:
    """Return a complete ``STORE_ATTR`` payload of kind ``"result"`` (master design 3.5)."""
    cropped = (int(crop["y1"] - crop["y0"]), int(crop["x1"] - crop["x0"]))
    attr = prepared_store_attr(
        dataset_name=dataset_name,
        reference_shape_yx=reference_shape_yx,
        moving_shape_yx=moving_shape_yx,
        reference_pixel_size_um=reference_pixel_size_um,
        moving_pixel_size_um=moving_pixel_size_um,
        reference_channels=reference_channels,
        moving_channels=moving_channels,
    )
    for key in ("channel_matching", "content_digest", "grid_digest"):
        del attr[key]  # a result store records them under "prepared_store" instead
    attr["kind"] = "result"
    attr["run_id"] = f"{dataset_name}_{preset}"
    attr["preset"] = preset
    attr["run_label"] = run_label
    attr["prepared_store"] = {
        "path": PLACEHOLDER,
        "content_digest": CONTENT_DIGEST,
        "grid_digest": GRID_DIGEST,
    }
    attr["images"] = {
        "reference": {
            "element": "reference",
            "role": "reference",
            "shape_yx": list(cropped),
            "full_shape_yx": [int(reference_shape_yx[0]), int(reference_shape_yx[1])],
            "crop": dict(crop),
            "pixel_size_um": float(reference_pixel_size_um),
            "dtype": "uint16",
            "channel_names": list(reference_channels),
            "registration_channel": reference_channels[0],
        },
        "moving": {
            "element": "moving_registered",
            "role": "moving",
            "shape_yx": list(cropped),
            "pixel_size_um": float(reference_pixel_size_um),
            "dtype": "uint16",
            "channel_names": list(moving_channels),
            "registration_channel": moving_channels[0],
            "source_shape_yx": [int(moving_shape_yx[0]), int(moving_shape_yx[1])],
            "source_pixel_size_um": float(moving_pixel_size_um),
            "source_crop": {
                "x0": 0,
                "y0": 0,
                "x1": int(moving_shape_yx[1]),
                "y1": int(moving_shape_yx[0]),
            },
        },
    }
    return attr


def labelling_payload(
    *,
    pairs: Sequence[dict[str, Any]],
    coarse: dict[str, Any] | None = None,
    crops: dict[str, Any] | None = None,
    revision: int = 1,
    status: str = "draft",
    reference_shape_yx: tuple[int, int] = REFERENCE_SHAPE_YX,
    moving_shape_yx: tuple[int, int] = MOVING_SHAPE_YX,
    reference_pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    moving_pixel_size_um: float = MOVING_PIXEL_SIZE_UM,
    colour_seed: int = 4211,
    updated_at: str = "2026-11-02T14:02:11Z",
    outlier_floor_um: float | None = 30.0,
) -> dict[str, Any]:
    """Return a full ``LABELLING_ATTR`` payload (master design 3.6).

    Each pair is a dict with ``id``, ``reference_xy_px``, ``moving_xy_px`` and optionally
    ``colour`` and ``enabled``. ``content_sha256`` comes from VW1's ``labelling_content_sha256``.
    """
    from napari_compare_xenium_merscope.manual_registration_model import pair_colour

    coarse_in = coarse or {}
    coarse_out = {
        "rotation_deg": float(coarse_in.get("rotation_deg", 0.0)),
        "mirror_x": bool(coarse_in.get("mirror_x", False)),
        "offset_um": [float(v) for v in coarse_in.get("offset_um", (0.0, 0.0))],
        "matrix_xy": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    full_pairs = []
    for p in pairs:
        full_pairs.append(
            {
                "id": int(p["id"]),
                "colour": p.get("colour", pair_colour(int(p["id"]), colour_seed)),
                "enabled": bool(p.get("enabled", True)),
                "reference_xy_px": [float(v) for v in p["reference_xy_px"]],
                "moving_xy_px": [float(v) for v in p["moving_xy_px"]],
            }
        )
    full_pairs.sort(key=lambda p: p["id"])
    payload: dict[str, Any] = {
        "schema": LABELLING_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "revision": int(revision),
        "updated_at": updated_at,
        "finished_at": updated_at if status == "finished" else None,
        "written_by": {"software": "napari-compare-xenium-merscope", "version": "0.0.0"},
        "store_grid_digest": GRID_DIGEST,
        "images": {
            "reference": {
                "shape_yx": [int(v) for v in reference_shape_yx],
                "pixel_size_um": float(reference_pixel_size_um),
            },
            "moving": {
                "shape_yx": [int(v) for v in moving_shape_yx],
                "pixel_size_um": float(moving_pixel_size_um),
            },
        },
        "coarse": coarse_out,
        "crops": {
            "reference": (crops or {}).get("reference"),
            "moving": (crops or {}).get("moving"),
        },
        "landmarks": {
            "next_id": max((p["id"] for p in full_pairs), default=0) + 1,
            "colour_seed": int(colour_seed),
            "pairs": full_pairs,
        },
    }
    payload["content_sha256"] = labelling_content_sha256(payload)
    if outlier_floor_um is not None:
        payload["outlier_floor_um"] = float(outlier_floor_um)
    return payload


def default_result_landmarks() -> list[dict[str, Any]]:
    """Return three landmark records for a result attribute (positions in full reference px)."""
    records = [
        # id, reference, moving, final, landmark, affine, used, outlier, outside_crop, enabled
        (1, (500.0, 300.0), (400.0, 250.0), (501.5, 299.0), (503.0, 297.5), (501.0, 299.5)),
        (2, (700.25, 800.5), (640.0, 700.0), (700.0, 801.0), (702.0, 799.0), (700.5, 800.75)),
        (3, (250.0, 900.0), (210.0, 780.0), (249.0, 899.5), (251.5, 902.0), (249.5, 899.75)),
    ]
    from napari_compare_xenium_merscope.manual_registration_model import pair_colour

    out = []
    for pid, ref, mov, final, landmark, affine in records:
        out.append(
            {
                "id": pid,
                "colour": pair_colour(pid, 4211),
                "enabled": True,
                "used_in_fit": pid != 3,
                "outlier": pid == 3,
                "outside_crop": False,
                "reference_xy_px": list(ref),
                "moving_xy_px": list(mov),
                "moving_in_reference_xy_px": {
                    "landmark": list(landmark),
                    "affine": list(affine),
                    "final": list(final),
                },
                "residual_um": {
                    "landmark": 1.5 * pid,
                    "loo": 2.0 * pid,
                    "affine": 1.0 * pid,
                    "final": 0.5 * pid,
                },
                "final_includes_flow": True,
                "push_converged": True,
            }
        )
    return out


def result_attr(
    *,
    landmarks: Sequence[dict[str, Any]],
    crop: dict[str, int] = RESULT_CROP,
    preset: str = "serial_section",
    run_label: str | None = None,
) -> dict[str, Any]:
    """Return a ``RESULT_ATTR`` payload with the keys the controller reads (master design 3.7)."""
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": f"{DATASET_NAME}_{preset}",
        "dataset_name": DATASET_NAME,
        "preset": preset,
        "preset_version": f"{preset}_v1",
        "run_label": run_label,
        "pixel_size_um": {
            "reference": REFERENCE_PIXEL_SIZE_UM,
            "moving": MOVING_PIXEL_SIZE_UM,
        },
        "crops": {
            "reference": {"requested": None, "effective": dict(crop)},
            "moving": {"requested": None, "effective": None},
        },
        "fit": {"model": "similarity", "n_pairs": len(landmarks), "outlier_floor_um": 30.0},
        "refinement": {"status": "applied", "reason": None},
        "nonrigid": {"enabled": True, "status": "skipped", "reason": "too few confident windows"},
        "tissue_qc": {
            "pearson": {"coarse": 0.31, "landmark": 0.52, "affine": 0.58, "final": 0.66},
        },
        "landmarks": [dict(row) for row in landmarks],
        "warnings": [],
    }


def write_root_attr(path: Path, key: str, payload: dict[str, Any]) -> None:
    """Write one root attribute of a Zarr group (other keys are kept)."""
    import zarr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.open_group(str(path), mode="a").attrs.update({key: payload})


def _write_images(path: Path, images: dict[str, Any]) -> None:
    import spatialdata as sd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd.SpatialData(images=images).write(str(path))


def _image_model(array: np.ndarray, channels: Sequence[str], transformation: Any) -> Any:
    from spatialdata.models import Image2DModel

    return Image2DModel.parse(
        array,
        dims=("c", "y", "x"),
        c_coords=list(channels),
        transformations={"global": transformation},
        chunks=(1, 512, 512),
    )


def build_prepared_store(
    path: Path,
    *,
    labelling: dict[str, Any] | None = None,
    reference_shape_yx: tuple[int, int] = REFERENCE_SHAPE_YX,
    moving_shape_yx: tuple[int, int] = MOVING_SHAPE_YX,
    reference_pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    moving_pixel_size_um: float = MOVING_PIXEL_SIZE_UM,
    reference_channels: Sequence[str] = REFERENCE_CHANNELS,
    moving_channels: Sequence[str] = MOVING_CHANNELS,
) -> Path:
    """Write a synthetic prepared store (two images + ``STORE_ATTR``, optional labelling)."""
    from spatialdata.transformations import Scale

    path = Path(path)
    p_r, p_m = reference_pixel_size_um, moving_pixel_size_um
    reference = _image_model(
        synthetic_image(reference_shape_yx, len(reference_channels), seed=11),
        reference_channels,
        Scale([p_r, p_r], axes=("x", "y")),
    )
    moving = _image_model(
        synthetic_image(moving_shape_yx, len(moving_channels), seed=23),
        moving_channels,
        Scale([p_m, p_m], axes=("x", "y")),
    )
    _write_images(path, {"reference": reference, "moving": moving})
    write_root_attr(
        path,
        STORE_ATTR,
        prepared_store_attr(
            reference_shape_yx=reference_shape_yx,
            moving_shape_yx=moving_shape_yx,
            reference_pixel_size_um=p_r,
            moving_pixel_size_um=p_m,
            reference_channels=reference_channels,
            moving_channels=moving_channels,
        ),
    )
    if labelling is not None:
        write_root_attr(path, LABELLING_ATTR, labelling)
    return path


def build_result_store(
    path: Path,
    *,
    landmarks: Sequence[dict[str, Any]] | None = None,
    crop: dict[str, int] = RESULT_CROP,
    **kw: Any,
) -> Path:
    """Write a synthetic result store (cropped reference + registered moving, result attrs)."""
    from spatialdata.transformations import Affine

    path = Path(path)
    p_r = REFERENCE_PIXEL_SIZE_UM
    cropped = (int(crop["y1"] - crop["y0"]), int(crop["x1"] - crop["x0"]))
    transformation = Affine(
        [[p_r, 0.0, p_r * crop["x0"]], [0.0, p_r, p_r * crop["y0"]], [0.0, 0.0, 1.0]],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    reference_full = synthetic_image(REFERENCE_SHAPE_YX, len(REFERENCE_CHANNELS), seed=11)
    reference = _image_model(
        reference_full[:, crop["y0"] : crop["y1"], crop["x0"] : crop["x1"]],
        REFERENCE_CHANNELS,
        transformation,
    )
    registered = _image_model(
        synthetic_image(cropped, len(MOVING_CHANNELS), seed=37), MOVING_CHANNELS, transformation
    )
    _write_images(path, {"reference": reference, "moving_registered": registered})
    write_root_attr(path, STORE_ATTR, result_store_attr(crop=crop, **kw))
    rows = default_result_landmarks() if landmarks is None else list(landmarks)
    write_root_attr(path, RESULT_ATTR, result_attr(landmarks=rows, crop=crop))
    return path


class FakeHost:
    """A synchronous implementation of ``RegistrationHost`` that records what the controller does."""

    def __init__(self, viewer: Any, app_dir: Path) -> None:
        self.viewer = viewer
        self._app_dir = Path(app_dir)
        self.statuses: list[str] = []
        self.progress: list[tuple[str, str, str | None]] = []
        self.entered: list[Path] = []
        self.ready: list[tuple[Path, str]] = []
        self.left = 0
        self.worker_keys: list[str] = []

    def set_status(self, text: str) -> None:
        self.statuses.append(text)

    def begin_progress(self, key: str, text: str) -> None:
        self.progress.append(("begin", key, text))

    def end_progress(self, key: str) -> None:
        self.progress.append(("end", key, None))

    def run_worker(self, compute, on_done, on_error, *, key: str) -> None:  # noqa: ANN001
        self.worker_keys.append(key)
        try:
            result = compute()
        except Exception as exc:  # noqa: BLE001 (mirror the real worker: errors go to on_error)
            on_error(exc)
            return
        on_done(result)

    def display_levels(self, image_key: str, sdata: Any, *, cancel_check=None) -> list[Any]:  # noqa: ANN001
        base = ensure_cyx(get_scale0_dataarray(sdata.images[image_key]))
        return [base, base.isel(y=slice(None, None, 4), x=slice(None, None, 4))]

    def enter_registration_session(self, path: Path, sdata: Any) -> None:
        self.entered.append(Path(path))
        self.viewer.layers.clear()

    def registration_session_ready(self, path: Path, kind: str) -> None:
        self.ready.append((Path(path), kind))
        self.end_progress("registration")

    def leave_registration_session(self) -> None:
        self.left += 1

    def app_data_dir(self) -> Path:
        return self._app_dir


class DeferredHost(FakeHost):
    """A host whose ``run_worker`` queues the task; ``run_pending`` runs the queued tasks."""

    def __init__(self, viewer: Any, app_dir: Path) -> None:
        super().__init__(viewer, app_dir)
        self.pending: list[tuple[Any, Any, Any, str]] = []

    def run_worker(self, compute, on_done, on_error, *, key: str) -> None:  # noqa: ANN001
        self.worker_keys.append(key)
        self.pending.append((compute, on_done, on_error, key))

    def run_pending(self, *, keys: Sequence[str] | None = None) -> int:
        """Run (and remove) the queued tasks, optionally only those with a key in ``keys``."""
        ran = 0
        for task in list(self.pending):
            compute, on_done, on_error, key = task
            if keys is not None and key not in keys:
                continue
            self.pending.remove(task)
            try:
                result = compute()
            except Exception as exc:  # noqa: BLE001
                on_error(exc)
            else:
                on_done(result)
            ran += 1
        return ran

    def drop_pending(self, *, keys: Sequence[str] | None = None) -> int:
        """Discard queued tasks without running them (what a host cancel does)."""
        dropped = [t for t in self.pending if keys is None or t[3] in keys]
        for task in dropped:
            self.pending.remove(task)
        return len(dropped)
