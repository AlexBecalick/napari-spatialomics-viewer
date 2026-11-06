"""Viewer wiring of the Atera transcript engine (milestone M7).

Two transcript modes for a standalone ATERA store:

* ``full`` (default): one background pass streams every parquet file and keeps
  a deterministic, nested hash sample whose size is the point budget (every
  transcript when the budget covers the store).  The sample becomes an
  ordinary :class:`~.utils.GenePointStore`, so the gene inspector's layers,
  colours, filters, highlights and selection slots work exactly as for
  Xenium/MERSCOPE.  When the budget leaves rows out, a viewport layer tops the
  visible field up to full density with exactly the rows the sample skipped.
* ``lazy``: a 5 um density image of every real gene, full-resolution layers
  only for the genes ticked (batched per-gene reads), and the same viewport
  layer showing every gene once zoomed in far enough.

Every transcript read runs on a worker thread through the engine's bounded
reader pool (never the viewer's store I/O slots), single-flight and
latest-wins per kind, so image and mask loads are never blocked.

This module holds Qt-free helpers (budgets, bounds, the scheduler) and a mixin
of :class:`~.viewer.ComparisonViewerController` methods.
"""

from __future__ import annotations

import gc
import logging
from contextlib import ExitStack
import math
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

import numpy as np

from .atera_transcripts import (
    ASSIGNMENT_COLUMN,
    DEFAULT_POINTS_KEY,
    GENE_COLUMN,
    SAMPLE_KEY_SPACE,
    X_COLUMN,
    Y_COLUMN,
    AteraTranscriptEngine,
    auto_point_budget,
    clamp_point_budget,
    gene_point_store_from_rows,
    physical_memory_bytes,
    sample_threshold,
    threshold_fraction,
)
from .transcript_cache import (
    load_atera_code_map,
    load_atera_overview,
    load_atera_sample,
    save_atera_overview,
    save_atera_sample,
)
from .utils import TranscriptBuildCancelled, resolve_cell_type_marker_reference

log = logging.getLogger(__name__)

ATERA_MODE_FULL = "full"
ATERA_MODE_LAZY = "lazy"
ATERA_TRANSCRIPT_MODES = (ATERA_MODE_FULL, ATERA_MODE_LAZY)
DEFAULT_ATERA_TRANSCRIPT_MODE = ATERA_MODE_FULL
#: Cap of the viewport top-up layer (rows streamed for the visible field).
DEFAULT_ATERA_VIEWPORT_MAX_POINTS = 20_000_000
#: Estimated memory per loaded point: the GenePointStore (coords + RGBA, 24 B),
#: the napari Points layer (~110 B: data, face/border colours, sizes, shown)
#: and the GPU vertex buffer (~56 B) - calibrated on the real store (M7 notes).
ATERA_BYTES_PER_POINT = 200.0
#: AUTO budget: this fraction of physical RAM over ATERA_BYTES_PER_POINT.
ATERA_AUTO_RAM_FRACTION = 0.5
#: Memory guard: never load more than this fraction of physical RAM.
ATERA_MEMORY_GUARD_FRACTION = 0.85
#: Below this a MemoryError is reported instead of halving again.
ATERA_MIN_BUDGET = 1_000_000
ATERA_OVERVIEW_BIN_UM = 5.0
ATERA_VIEWPORT_DEBOUNCE_MS = 200
ATERA_GENE_DEBOUNCE_MS = 120
ATERA_PROGRESS_POLL_MS = 500
#: The viewport reads a window this much larger than the view on each side,
#: so small pans are served from the cached rows.
ATERA_VIEWPORT_PAD_FRACTION = 0.5
#: Lazy mode keeps per-gene rows for at most this many rows (LRU over
#: unticked genes; ticked genes are never evicted).
ATERA_LAZY_GENE_CACHE_ROWS = 100_000_000
ATERA_DENSITY_LAYER = "Genes | density overview"
ATERA_VIEWPORT_LAYER = "Genes | viewport"
_READ_COLUMNS = (X_COLUMN, Y_COLUMN, GENE_COLUMN, ASSIGNMENT_COLUMN)


# ---------------------------------------------------------------------------
# Qt-free helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AteraBudget:
    """The point budget of a full-mode load and how it was chosen."""

    budget: int
    auto_budget: int
    total: int
    clamped: bool
    source: str  # "auto" | "requested" | "all"
    ram_bytes: int | None
    note: str


def resolve_atera_budget(
    total: int,
    requested: int | None,
    *,
    ram_bytes: int | None,
    bytes_per_point: float = ATERA_BYTES_PER_POINT,
) -> AteraBudget:
    """Pick the full-mode point budget.

    ``requested`` ``None`` means AUTO (``ATERA_AUTO_RAM_FRACTION`` of RAM over
    ``bytes_per_point``), ``<= 0`` means every transcript; any budget is then
    clamped by the memory guard (``ATERA_MEMORY_GUARD_FRACTION`` of RAM).
    """
    total = max(0, int(total))
    auto = auto_point_budget(total, bytes_per_point=bytes_per_point, ram_bytes=ram_bytes, fraction=ATERA_AUTO_RAM_FRACTION)
    if requested is None:
        budget, source = auto, "auto"
    elif int(requested) <= 0:
        budget, source = total, "all"
    else:
        budget, source = min(int(requested), total), "requested"
    budget, clamped = clamp_point_budget(
        budget, bytes_per_point=bytes_per_point, ram_bytes=ram_bytes, limit_fraction=ATERA_MEMORY_GUARD_FRACTION
    )
    ram_text = "unknown RAM" if not ram_bytes else f"{ram_bytes / 1024**3:.0f} GB RAM"
    if source == "auto":
        note = (
            f"AUTO budget {budget / 1e6:,.1f}M = {ATERA_AUTO_RAM_FRACTION:g} x {ram_text} / "
            f"{bytes_per_point:g} B per point"
        )
    elif source == "all":
        note = f"budget: every transcript ({total / 1e6:,.1f}M)"
    else:
        note = f"budget {budget / 1e6:,.1f}M requested"
    if clamped:
        note += (
            f"; clamped to {budget / 1e6:,.1f}M so the estimated "
            f"{bytes_per_point:g} B/point stays under {ATERA_MEMORY_GUARD_FRACTION:.0%} of {ram_text}"
        )
    return AteraBudget(budget, auto, total, clamped, source, ram_bytes, note)


def visible_bounds(center, zoom: float, canvas_hw) -> tuple[float, float, float, float] | None:
    """World ``(xmin, ymin, xmax, ymax)`` seen by a 2D camera (``None`` if unknown)."""
    try:
        cy, cx = (float(v) for v in list(center)[-2:])
        zoom = float(zoom)
        h, w = (float(v) for v in list(canvas_hw)[-2:])
    except (TypeError, ValueError):
        return None
    if not (zoom > 0 and h > 0 and w > 0) or not all(math.isfinite(v) for v in (cx, cy)):
        return None
    half_w = w / (2.0 * zoom)
    half_h = h / (2.0 * zoom)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def pad_bounds(bounds, fraction: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in bounds)
    dx = (x1 - x0) * float(fraction)
    dy = (y1 - y0) * float(fraction)
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def bounds_contain(outer, inner) -> bool:
    return bool(outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3])


def estimate_window_rows(overview, bounds, codes, exclude_below: int = 0) -> float:
    """Rows of ``codes`` a window read would return, from the density raster.

    The raster counts real genes only; the selected codes take their share of
    the real-gene total (controls are counted at the same rate).
    """
    density = overview.density_counts
    if density.size == 0:
        return 0.0
    x0, y0 = overview.origin_um
    b = float(overview.bin_um)
    h, w = density.shape
    c0 = int(np.clip(math.floor((bounds[0] - x0) / b), 0, w))
    c1 = int(np.clip(math.ceil((bounds[2] - x0) / b), 0, w))
    r0 = int(np.clip(math.floor((bounds[1] - y0) / b), 0, h))
    r1 = int(np.clip(math.ceil((bounds[3] - y0) / b), 0, h))
    if c1 <= c0 or r1 <= r0:
        return 0.0
    window = float(density[r0:r1, c0:c1].sum(dtype=np.int64))
    counts = np.asarray(overview.counts)
    real_total = float(counts[~np.asarray(overview.is_control)].sum())
    if real_total <= 0:
        return 0.0
    selected = float(counts[np.asarray(sorted(codes), dtype=np.int64)].sum()) if len(codes) else 0.0
    return window * (selected / real_total) * (1.0 - threshold_fraction(exclude_below))


def cap_rows_by_key(rows: dict[str, np.ndarray], cap: int) -> dict[str, np.ndarray]:
    """Keep the ``cap`` rows with the smallest sample keys (deterministic)."""
    n = len(rows["x"])
    if cap <= 0 or n <= cap:
        return rows
    keep = np.sort(np.argpartition(rows["key"], cap - 1)[:cap])
    return {name: values[keep] for name, values in rows.items()}


class LatestWinsRunner:
    """Single-flight, latest-wins scheduling of one kind of background read.

    At most one job runs and at most one waits; a new request replaces the
    waiting one (and cancels the running one's event).  ``start(request,
    generation, cancel)`` launches a job, which must call :meth:`finish` with
    its generation when it ends; ``finish`` says whether the result is the
    latest and starts the waiting job.
    """

    def __init__(self, start: Callable[[Any, int, Event], None]) -> None:
        self._start = start
        self.generation = 0
        self.running: int | None = None
        self.pending: tuple[Any, int] | None = None
        self.cancel_event: Event | None = None
        self.launched = 0

    @property
    def busy(self) -> bool:
        return self.running is not None

    def submit(self, request: Any, *, supersede: bool = True) -> None:
        self.generation += 1
        if self.running is None:
            self._launch(request, self.generation)
            return
        self.pending = (request, self.generation)
        if supersede and self.cancel_event is not None:
            self.cancel_event.set()

    def _launch(self, request: Any, generation: int) -> None:
        self.running = generation
        self.cancel_event = Event()
        self.launched += 1
        self._start(request, generation, self.cancel_event)

    def finish(self, generation: int) -> bool:
        current = generation == self.generation
        if self.running == generation:
            self.running = None
            self.cancel_event = None
        if self.running is None and self.pending is not None:
            request, pending_generation = self.pending
            self.pending = None
            self._launch(request, pending_generation)
        return current

    def cancel(self) -> None:
        """Drop the waiting job and make any running result stale."""
        self.generation += 1
        self.pending = None
        if self.cancel_event is not None:
            self.cancel_event.set()


@dataclass
class AteraTranscriptContext:
    """Per-dataset lazy-engine state attached to a GeneInspectorState."""

    dataset: str
    zarr_path: str
    points_key: str
    engine: Any
    overview: Any
    mode: str
    budget: AteraBudget | None
    threshold: int
    total_rows: int
    requested_budget: int | None = None
    viewport_cap: int = DEFAULT_ATERA_VIEWPORT_MAX_POINTS
    memory_note: str = ""
    loaded: dict = field(default_factory=dict)  # lazy: gene -> rows
    gene_runner: LatestWinsRunner | None = None
    view_runner: LatestWinsRunner | None = None
    view_timer: Any = None
    gene_timer: Any = None
    viewport_cache: dict | None = None
    viewport_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    camera_callback: Any = None
    last_hint: str = ""
    #: The latest viewport request submitted (in flight or pending).
    view_request: dict | None = None
    #: _atera_render_key of what the viewport layer currently shows.
    rendered_key: tuple | None = None

    @property
    def complete(self) -> bool:
        return self.mode == ATERA_MODE_FULL and self.threshold >= SAMPLE_KEY_SPACE

    @property
    def viewport_enabled(self) -> bool:
        return bool(getattr(self.engine, "window_reads_available", False)) and not self.complete

    def options_key(self) -> tuple:
        return (self.mode, self.requested_budget)


def _empty_rows(engine) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = engine.index
    return (
        np.zeros(0, np.float32), np.zeros(0, np.float32),
        np.zeros(0, index.code_dtype), np.zeros(0, index.assignment_dtype),
    )


def _concat_gene_rows(engine, rows_by_gene: dict) -> tuple[np.ndarray, ...]:
    parts = [rows for rows in rows_by_gene.values() if len(rows.get("x", ()))]
    if not parts:
        return _empty_rows(engine)
    return (
        np.concatenate([p["x"] for p in parts]).astype(np.float32, copy=False),
        np.concatenate([p["y"] for p in parts]).astype(np.float32, copy=False),
        np.concatenate([p["code"] for p in parts]),
        np.concatenate([p[ASSIGNMENT_COLUMN] for p in parts]),
    )


def _rows_count(rows: dict) -> int:
    return int(len(rows.get("x", ())))


# ---------------------------------------------------------------------------
# Controller mixin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cancelled:
    """A worker result standing for a cancelled read (no traceback printed)."""

    exc: BaseException


def _thread_worker():
    from . import viewer as _viewer

    return getattr(_viewer, "thread_worker", None)


def _qtimer(interval_ms: int, callback):
    try:
        from qtpy.QtCore import QTimer
    except Exception:  # pragma: no cover - Qt always present in the app
        return None
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(int(interval_ms))
    timer.timeout.connect(callback)
    return timer


class AteraTranscriptMixin:
    """Atera transcript wiring for :class:`~.viewer.ComparisonViewerController`."""

    # -- options --------------------------------------------------------

    def _atera_options_store(self) -> dict:
        store = getattr(self, "_atera_transcript_options", None)
        if store is None:
            store = {}
            self._atera_transcript_options = store
        return store

    def _atera_requested_options(self, ds: str) -> tuple[str, int | None, int]:
        """``(mode, requested budget or None for AUTO, viewport cap)`` for ``ds``."""
        opts = self._atera_options_store().get(str(ds).upper(), {})
        mode = str(opts.get("mode") or getattr(self.args, "atera_transcript_mode", None) or DEFAULT_ATERA_TRANSCRIPT_MODE)
        if mode not in ATERA_TRANSCRIPT_MODES:
            mode = DEFAULT_ATERA_TRANSCRIPT_MODE
        if "budget" in opts:
            budget = opts["budget"]
        else:
            budget = getattr(self.args, "gene_max_render_points", None)
        cap = opts.get("viewport_cap")
        if cap is None:
            cap = getattr(self.args, "atera_viewport_max_points", None) or DEFAULT_ATERA_VIEWPORT_MAX_POINTS
        return mode, (None if budget is None else int(budget)), int(cap)

    # -- open / build -----------------------------------------------------

    def _atera_points_key(self) -> str:
        sdata = getattr(self, "_active_sdata", None)
        try:
            keys = list(getattr(sdata, "points", {}) or {})
        except Exception:
            keys = []
        if DEFAULT_POINTS_KEY in keys or not keys:
            return DEFAULT_POINTS_KEY
        return str(keys[0])

    def _open_atera_gene_inspector(self, ds: str, reuse_cached: bool = False, announce: bool = True):
        ds = str(ds).upper()
        mode, requested, cap = self._atera_requested_options(ds)
        session = self._dataset_sessions.get(ds)
        cached = None if session is None else session.gene_payload
        if (
            reuse_cached
            and cached is not None
            and cached.get("atera") is not None
            and cached["atera"].options_key() == (mode, requested)
            and self._transcript_payload_is_current(ds, cached)
        ):
            self._teardown_gene_inspector(ds)
            self._gene_build_generation += 1
            generation = self._gene_build_generation
            self._begin_progress("transcripts", f"{ds}: restoring prepared transcripts…")
            payload = dict(cached)
            payload["atera"].viewport_cap = cap
            self._apply_gene_inspector_build(generation, ds, payload, token=None)
            return
        cfg = self.datasets.get(ds)
        zarr_path = getattr(cfg, "zarr_path", None)
        if zarr_path is None:
            self._set_status(f"{ds}: no store path for transcripts.")
            return
        points_key = self._atera_points_key()
        self._teardown_gene_inspector(ds)
        self._publish_transcript_sources()
        gc.collect()
        self._gene_build_generation += 1
        generation = self._gene_build_generation
        task_key = (ds, "atera", str(zarr_path), points_key, mode, requested)
        cancel_token = self._begin_task_token("transcripts", task_key)
        if cancel_token is None:
            self._set_status(f"{ds}: transcript load is already running.")
            return
        text = (
            f"{ds}: streaming every transcript for a whole-slide sample…"
            if mode == ATERA_MODE_FULL
            else f"{ds}: loading the transcript density overview…"
        )
        if announce:
            self._begin_progress("transcripts", text)
        elif self._progress_callback is not None:
            self._progress_callback("transcripts", text, True)
        progress: dict = {"done": 0, "total": 0, "phase": ""}
        self._atera_start_progress_poll(ds, generation, progress, announce)

        def compute():
            return self._atera_compute_payload(
                ds, str(zarr_path), points_key, mode, requested, cap, cancel_token, progress
            )

        factory = _thread_worker()
        if factory is None:
            try:
                payload = compute()
            except Exception as exc:
                self._handle_gene_inspector_build_error(generation, ds, exc, cancel_token)
                return
            self._apply_gene_inspector_build(generation, ds, payload, cancel_token)
            return
        worker = factory(compute)()
        worker.returned.connect(
            lambda payload, gen=generation, d=ds, token=cancel_token: self._apply_gene_inspector_build(gen, d, payload, token)
        )
        worker.errored.connect(
            lambda exc, gen=generation, d=ds, token=cancel_token: self._handle_gene_inspector_build_error(gen, d, exc, token)
        )
        self._gene_build_worker = worker
        worker.start()

    def _atera_start_progress_poll(self, ds: str, generation: int, progress: dict, announce: bool) -> None:
        old = getattr(self, "_atera_progress_timer", None)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        self._atera_progress_timer = None
        if not announce:
            return

        def poll():
            if generation != self._gene_build_generation or self._gene_build_worker is None:
                return
            done, total = int(progress.get("done", 0)), int(progress.get("total", 0))
            if total:
                self._set_status(f"{ds}: {progress.get('phase') or 'streaming transcripts'}… {done}/{total} files")
            if timer is not None:
                timer.start()

        timer = _qtimer(ATERA_PROGRESS_POLL_MS, poll)
        if timer is not None:
            self._atera_progress_timer = timer
            timer.start()

    def _atera_compute_payload(
        self,
        ds: str,
        zarr_path: str,
        points_key: str,
        mode: str,
        requested: int | None,
        viewport_cap: int,
        cancel_token: Event,
        progress: dict,
    ) -> dict:
        """Worker thread: engine, overview, (full) sample and the point store."""
        t0 = time.time()
        engine = AteraTranscriptEngine.open(zarr_path, points_key)
        if engine is None:
            raise RuntimeError(f"no readable Atera points under points/{points_key}")
        index = engine.index
        code_map = load_atera_code_map(index)
        if code_map is not None:
            engine.attach_code_map(code_map)
        total = int(index.n_rows)
        budget = None
        threshold = 0
        if mode == ATERA_MODE_FULL:
            budget = resolve_atera_budget(total, requested, ram_bytes=physical_memory_bytes())
            threshold = sample_threshold(budget.budget, total)
            log.info(
                "[%s] Atera full transcript load: %s (total %d rows, threshold %d/%d)",
                ds, budget.note, total, threshold, SAMPLE_KEY_SPACE,
            )

        def report(phase: str):
            def _report(done: int, n: int) -> None:
                progress.update(done=int(done), total=int(n), phase=phase)
            return _report

        cancel = cancel_token.is_set
        overview = load_atera_overview(index, bin_um=ATERA_OVERVIEW_BIN_UM)
        cache_hit = overview is not None
        sample = None
        if threshold > 0:
            sample = load_atera_sample(index, threshold, cancel_check=cancel)
            cache_hit = cache_hit and sample is not None
        memory_note = ""
        resolution = None

        def resolve_reference(genes):
            nonlocal resolution
            try:
                resolution = resolve_cell_type_marker_reference(zarr_path, genes)
            except Exception as exc:  # metadata only; never fail the load for it
                log.warning("[%s] Cell-type marker reference lookup failed: %s", ds, exc)
                return None
            return resolution.reference

        while True:
            try:
                if overview is None:
                    overview = engine.build_overview(
                        ATERA_OVERVIEW_BIN_UM, cancel_token, progress=report("streaming transcripts"),
                        keep_threshold=threshold if threshold > 0 and sample is None else None,
                    )
                    if overview.hash_sample is not None:
                        sample = overview.hash_sample
                        overview.hash_sample = None
                    if not cancel():
                        save_atera_overview(index, overview, cancel_check=cancel)
                    if sample is not None and not cancel():
                        save_atera_sample(index, sample, cancel_check=cancel)
                elif threshold > 0 and sample is None:
                    sample = engine.sample_rows(threshold, cancel_token, progress=report("sampling transcripts"))
                    if not cancel():
                        save_atera_sample(index, sample, cancel_check=cancel)
                if overview.code_map is not None and engine.code_map is None:
                    engine.attach_code_map(overview.code_map)
                if cancel():
                    raise TranscriptBuildCancelled("Atera transcript load cancelled")
                rows = (
                    (sample.x, sample.y, sample.code, sample.cell_id)
                    if sample is not None
                    else _empty_rows(engine)
                )
                store = gene_point_store_from_rows(
                    overview, *rows,
                    sampled=bool(mode == ATERA_MODE_FULL and threshold < SAMPLE_KEY_SPACE),
                    reference_resolver=resolve_reference,
                )
                break
            except MemoryError:
                if budget is None or budget.budget <= ATERA_MIN_BUDGET:
                    raise
                halved = budget.budget // 2
                memory_note = (
                    f"Out of memory at {budget.budget / 1e6:,.1f}M points; "
                    f"retrying with half ({halved / 1e6:,.1f}M)."
                )
                log.warning("[%s] %s", ds, memory_note)
                budget = AteraBudget(
                    halved, budget.auto_budget, total, True, budget.source, budget.ram_bytes,
                    budget.note + "; " + memory_note,
                )
                threshold = sample_threshold(halved, total)
                if sample is not None and sample.threshold >= threshold:
                    sample = sample.subset(threshold)
                else:
                    sample = None
                gc.collect()
        n_points = int(store.total_points)
        del sample
        context = AteraTranscriptContext(
            dataset=ds,
            zarr_path=zarr_path,
            points_key=points_key,
            engine=engine,
            overview=overview,
            mode=mode,
            budget=budget,
            threshold=int(threshold),
            total_rows=total,
            requested_budget=requested,
            viewport_cap=int(viewport_cap),
            memory_note=memory_note,
        )
        seconds = time.time() - t0
        log.info(
            "[%s] Atera transcripts (%s): %d of %d rows as points in %.1fs (cache %s)",
            ds, mode, n_points, total, seconds, "hit" if cache_hit else "miss",
        )
        return {
            "points_key": points_key,
            "store": store,
            "reference": None if resolution is None else resolution.reference,
            "reference_resolution": resolution,
            "cache_hit": cache_hit,
            "build_seconds": seconds,
            "assignment_source": "native",
            "analysis_id": "",
            "atera": context,
        }

    # -- install / teardown ---------------------------------------------

    def _atera_install(self, state) -> None:
        """Attach density/viewport layers, schedulers and camera hooks to ``state``."""
        ctx = state.atera
        ds = state.dataset
        ctx.viewport_cache = None
        ctx.viewport_codes = np.zeros(0, dtype=np.int32)
        ctx.last_hint = ""
        ctx.view_request = None
        if ctx.mode == ATERA_MODE_LAZY:
            ctx.loaded = {}
        ctx.gene_runner = LatestWinsRunner(
            lambda request, gen, cancel, d=ds: self._atera_start_gene_read(d, request, gen, cancel)
        )
        ctx.view_runner = LatestWinsRunner(
            lambda request, gen, cancel, d=ds: self._atera_start_view_read(d, request, gen, cancel)
        )
        ctx.view_timer = _qtimer(ATERA_VIEWPORT_DEBOUNCE_MS, lambda d=ds: self._atera_request_viewport(d))
        ctx.gene_timer = _qtimer(ATERA_GENE_DEBOUNCE_MS, lambda d=ds: self._atera_flush_gene_reads(d))
        if ctx.mode == ATERA_MODE_LAZY:
            self._atera_add_density_layer(state)
        if ctx.viewport_enabled:
            self._atera_add_viewport_layer(state)
            self._atera_connect_camera(state)
        self._atera_order_layers(state)
        self._atera_publish_controls(state)

    def _atera_add_density_layer(self, state) -> None:
        ctx = state.atera
        overview = ctx.overview
        density = np.log1p(np.asarray(overview.density_counts, dtype=np.float32))
        high = float(np.percentile(density[density > 0], 99.5)) if np.any(density > 0) else 1.0
        x0, y0 = overview.origin_um
        b = float(overview.bin_um)
        add_image = getattr(self.viewer, "add_image", None)
        if add_image is None:
            return
        self._remove_layer_by_name(ATERA_DENSITY_LAYER)
        try:
            add_image(
                density,
                name=ATERA_DENSITY_LAYER,
                scale=(b, b),
                translate=(y0 + b / 2.0, x0 + b / 2.0),
                colormap="magma",
                contrast_limits=(0.0, max(high, 1e-3)),
                blending="additive",
                opacity=0.85,
                visible=True,
            )
        except Exception as exc:  # cosmetic overview; never fail the load
            log.warning("[%s] Could not add the transcript density layer: %s", state.dataset, exc)

    def _atera_add_viewport_layer(self, state) -> None:
        self._remove_layer_by_name(ATERA_VIEWPORT_LAYER)
        # An empty layer takes one RGBA colour (napari rejects a (0, 4) array).
        self._create_gene_points_layer(
            ATERA_VIEWPORT_LAYER, "disc", state.spot_size,
            np.zeros((0, 2), dtype=np.float32), np.ones(4, dtype=np.float32),
        )

    def _atera_order_layers(self, state) -> None:
        """Gene groups stay at the bottom; the density image goes below them."""
        layers = self.viewer.layers
        try:
            if not hasattr(layers, "move"):
                return
            for name in (ATERA_VIEWPORT_LAYER, ATERA_DENSITY_LAYER):
                src = next((i for i, layer in enumerate(layers) if str(getattr(layer, "name", "")) == name), None)
                if src is None:
                    continue
                if name == ATERA_DENSITY_LAYER:
                    layers.move(src, 0)
                else:
                    top = max(
                        (i for i, layer in enumerate(layers) if str(getattr(layer, "name", "")) in set(state.layer_names)),
                        default=-1,
                    )
                    layers.move(src, top + 1 if src > top else top)
        except Exception as exc:  # pragma: no cover - ordering is cosmetic
            log.debug("Could not order the Atera transcript layers: %s", exc)

    def _atera_connect_camera(self, state) -> None:
        camera = getattr(self.viewer, "camera", None)
        events = getattr(camera, "events", None)
        if events is None:
            return
        ds = state.dataset

        def on_camera(_event=None, d=ds):
            self._atera_schedule_viewport(d)

        for name in ("center", "zoom"):
            emitter = getattr(events, name, None)
            if emitter is not None:
                try:
                    emitter.connect(on_camera)
                except Exception:
                    pass
        state.atera.camera_callback = on_camera

    def _atera_teardown(self, state) -> None:
        ctx = getattr(state, "atera", None)
        if ctx is None:
            return
        for runner in (ctx.gene_runner, ctx.view_runner):
            if runner is not None:
                runner.cancel()
        for timer in (ctx.view_timer, ctx.gene_timer):
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        callback = ctx.camera_callback
        events = getattr(getattr(self.viewer, "camera", None), "events", None)
        if callback is not None and events is not None:
            for name in ("center", "zoom"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    try:
                        emitter.disconnect(callback)
                    except Exception:
                        pass
        ctx.camera_callback = None
        ctx.viewport_cache = None
        ctx.view_request = None
        widget = getattr(self, "_gene_inspector_widget", None)
        if widget is not None and hasattr(widget, "set_atera_controls"):
            widget.set_atera_controls(False)

    def _atera_cancel_all(self) -> None:
        for state in list(getattr(self, "_gene_inspector_states", {}).values()):
            ctx = getattr(state, "atera", None)
            if ctx is None:
                continue
            for runner in (ctx.gene_runner, ctx.view_runner):
                if runner is not None:
                    runner.cancel()
            ctx.view_request = None

    def _atera_publish_controls(self, state) -> None:
        widget = getattr(self, "_gene_inspector_widget", None)
        if widget is None or not hasattr(widget, "set_atera_controls"):
            return
        ctx = state.atera
        budget = ctx.budget
        auto = budget.auto_budget if budget is not None else resolve_atera_budget(
            ctx.total_rows, None, ram_bytes=physical_memory_bytes()
        ).auto_budget
        note = "" if budget is None else budget.note
        if ctx.memory_note:
            note = f"{note}. {ctx.memory_note}" if note else ctx.memory_note
        widget.set_atera_controls(
            True,
            dataset=state.dataset,
            mode=ctx.mode,
            budget=None if ctx.requested_budget is None else int(ctx.requested_budget),
            auto_budget=int(auto),
            total=int(ctx.total_rows),
            loaded=int(state.store.total_points),
            viewport_cap=int(ctx.viewport_cap),
            note=note,
        )

    def _atera_status_text(self, state, payload: dict) -> str:
        ctx = state.atera
        seconds = float(payload.get("build_seconds", 0.0))
        source = "restored from the persistent cache" if payload.get("cache_hit") else "built"
        if ctx.mode == ATERA_MODE_FULL:
            text = (
                f"{state.dataset}: {state.store.total_points / 1e6:,.1f}M of {ctx.total_rows / 1e6:,.1f}M "
                f"transcripts loaded ({ctx.budget.note if ctx.budget else ''}) - {source} in {seconds:.1f}s."
            )
        else:
            text = (
                f"{state.dataset}: density overview of {ctx.total_rows / 1e6:,.1f}M transcripts {source} in "
                f"{seconds:.1f}s; ticked genes load at full resolution."
            )
        if ctx.memory_note:
            text += f" {ctx.memory_note}"
        return text + " Tick genes, or use the All-transcript overview."

    # -- user options -----------------------------------------------------

    def reload_atera_transcripts(self, dataset_name: str, mode: str, budget: int | None, viewport_cap: int | None = None):
        """Reload with a new mode/budget (one reload per call; nested samples reuse the cache)."""
        ds = str(dataset_name).upper()
        opts = self._atera_options_store().setdefault(ds, {})
        if mode in ATERA_TRANSCRIPT_MODES:
            opts["mode"] = str(mode)
        opts["budget"] = None if budget is None else int(budget)
        if viewport_cap is not None and int(viewport_cap) > 0:
            opts["viewport_cap"] = int(viewport_cap)
        self.open_gene_inspector(ds)

    def set_atera_viewport_cap(self, dataset_name: str, cap: int) -> None:
        ds = str(dataset_name).upper()
        cap = max(1, int(cap))
        self._atera_options_store().setdefault(ds, {})["viewport_cap"] = cap
        state = self._gene_inspector_states.get(ds)
        if state is not None and getattr(state, "atera", None) is not None:
            state.atera.viewport_cap = cap
            state.atera.viewport_cache = None
            self._atera_request_viewport(ds)

    # -- selection hooks --------------------------------------------------

    def _atera_states(self, dataset_name: str | None = None):
        states = (
            self._gene_states_for_control(dataset_name)
            if dataset_name is not None
            else tuple(getattr(self, "_gene_inspector_states", {}).values())
        )
        return tuple(state for state in states if getattr(state, "atera", None) is not None)

    def _atera_selection_changed(self, dataset_name: str) -> None:
        """Genes or the overview changed: load missing lazy genes, refresh the viewport."""
        for state in self._atera_states(dataset_name):
            ctx = state.atera
            if ctx.mode == ATERA_MODE_LAZY:
                self._atera_set_density_visible(state, True)
                if self._atera_missing_genes(state):
                    if ctx.gene_timer is not None:
                        ctx.gene_timer.start()
                    else:
                        self._atera_flush_gene_reads(state.dataset)
            self._atera_schedule_viewport(state.dataset)

    def _atera_display_changed(self, dataset_name: str) -> None:
        """Colours, filters, spot size or highlights changed: re-render the viewport rows."""
        for state in self._atera_states(dataset_name):
            self._atera_render_viewport(state)

    def _atera_set_density_visible(self, state, on: bool) -> None:
        layer = self._get_layer_by_name(ATERA_DENSITY_LAYER)
        if layer is not None:
            try:
                layer.visible = bool(on)
            except Exception:
                pass

    # -- lazy per-gene reads ----------------------------------------------

    def _atera_missing_genes(self, state) -> list[str]:
        ctx = state.atera
        counts = state.store.source_gene_counts or {}
        return sorted(g for g in state.enabled_genes if g not in ctx.loaded and int(counts.get(g, 0)) > 0)

    def _atera_flush_gene_reads(self, ds: str) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        if state is None or getattr(state, "atera", None) is None:
            return
        ctx = state.atera
        if ctx.mode != ATERA_MODE_LAZY or not self._atera_missing_genes(state):
            return
        # The job reads whatever is missing when it starts, so requests made
        # while one runs are merged rather than lost (never superseded).
        ctx.gene_runner.submit("genes", supersede=False)

    def _atera_start_gene_read(self, ds: str, _request, generation: int, cancel: Event) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        ctx = None if state is None else getattr(state, "atera", None)
        if ctx is None:
            return
        wanted = self._atera_missing_genes(state)
        if not wanted:
            ctx.gene_runner.finish(generation)
            return
        enabled = set(state.enabled_genes)
        loaded = dict(ctx.loaded)
        # Broad and fine schemes share symbols, so the current colours keep the
        # layer grouping of the symbol each gene was listed with.
        visuals = dict(state.gene_visuals or {})
        self._set_status(f"{ds}: loading {len(wanted)} gene(s) at full resolution…")

        def compute():
            rows = ctx.engine.read_genes(wanted, columns=_READ_COLUMNS, cancel=cancel)
            merged = dict(loaded)
            merged.update(rows)
            budget = ATERA_LAZY_GENE_CACHE_ROWS
            total = sum(_rows_count(r) for r in merged.values())
            for gene in list(merged):
                if total <= budget:
                    break
                if gene in enabled:
                    continue
                total -= _rows_count(merged.pop(gene))
            shown = {g: r for g, r in merged.items()}
            store = gene_point_store_from_rows(
                ctx.overview, *_concat_gene_rows(ctx.engine, shown), sampled=False, gene_visuals=visuals,
            )
            return merged, store

        def done(result, gen=generation):
            self._atera_gene_read_done(ds, ctx, gen, result)

        def failed(exc, gen=generation):
            ctx.gene_runner.finish(gen)
            message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
            if not isinstance(message, TranscriptBuildCancelled):
                self._set_status(f"{ds}: gene read failed: {message}")
                log.error("[%s] Atera gene read failed: %s", ds, message)

        self._atera_run(compute, done, failed)

    def _atera_gene_read_done(self, ds: str, ctx, generation: int, result) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        if state is None or getattr(state, "atera", None) is not ctx:
            return
        merged, store = result
        ctx.loaded = merged
        self._atera_swap_store(state, store)
        ctx.gene_runner.finish(generation)
        if self._atera_missing_genes(state) and not ctx.gene_runner.busy:
            self._atera_flush_gene_reads(ds)
        self._atera_schedule_viewport(ds)
        self._set_status(
            f"{ds}: {len(merged)} gene(s) at full resolution ({store.total_points / 1e6:,.2f}M points)."
        )

    def _atera_swap_store(self, state, store) -> None:
        """Replace the state's point store and its group layers (lazy mode)."""
        for name in list(state.layer_names):
            self._remove_layer_by_name(name)
        state.store = store
        state.layer_names = []
        state.pending_groups = set()
        state.group_display_ranges = [[] for _ in store.group_symbols]
        state.group_shown_masks = [np.zeros(len(c), dtype=bool) for c in store.group_coords]
        self._reapply_selection_slot_colours(state)
        for gi, symbol in enumerate(store.group_symbols):
            coords, colors, sizes, ranges, shown = self._gene_group_arrays(state, gi)
            state.group_display_ranges[gi] = ranges
            name = self._gene_layer_name(state.dataset, symbol)
            self._create_gene_points_layer(name, symbol, state.spot_size, coords, colors, sizes, shown=shown)
            state.layer_names.append(name)
        self._move_gene_layers_to_bottom(state)
        self._atera_order_layers(state)

    # -- viewport ---------------------------------------------------------

    def _atera_schedule_viewport(self, ds: str) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        ctx = None if state is None else getattr(state, "atera", None)
        if ctx is None or not ctx.viewport_enabled:
            return
        if ctx.view_timer is not None:
            ctx.view_timer.start()
        else:
            self._atera_request_viewport(ds)

    def _atera_visible_bounds(self):
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return None
        canvas = getattr(self.viewer, "_canvas_size", None)
        if canvas is None:
            return None
        return visible_bounds(getattr(camera, "center", None), getattr(camera, "zoom", 0.0), canvas)

    def _atera_display_codes(self, state) -> np.ndarray:
        """Global codes the viewport must show.

        Full mode: the ticked genes, plus every real gene under the overview
        (the rows the sample left out).  Lazy mode: only the overview genes
        that are neither ticked nor loaded - ticked genes arrive at full
        resolution through their own per-gene read.
        """
        ctx = state.atera
        vocab_codes = {name: i for i, name in enumerate(ctx.overview.vocabulary)}
        names = set() if ctx.mode == ATERA_MODE_LAZY else set(state.enabled_genes)
        if getattr(state, "overview_visible", False):
            names |= {g for g in state.store.genes if g not in state.store.control_genes}
        if ctx.mode == ATERA_MODE_LAZY:
            names -= set(ctx.loaded) | set(state.enabled_genes)
        codes = sorted(vocab_codes[n] for n in names if n in vocab_codes)
        return np.asarray(codes, dtype=np.int64)

    def _atera_request_viewport(self, ds: str) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        ctx = None if state is None else getattr(state, "atera", None)
        if ctx is None or not ctx.viewport_enabled:
            return
        bounds = self._atera_visible_bounds()
        codes = self._atera_display_codes(state)
        if bounds is None or not len(codes):
            ctx.viewport_cache = None
            self._atera_render_viewport(state)
            return
        exclude = int(ctx.threshold) if ctx.mode == ATERA_MODE_FULL else 0
        cache = ctx.viewport_cache
        code_set = frozenset(int(c) for c in codes)
        if (
            cache is not None
            and cache["exclude"] == exclude
            and code_set <= cache["codes"]
            and bounds_contain(cache["bounds"], bounds)
        ):
            # A pan inside the cached window: the layer already holds these rows.
            if ctx.rendered_key != self._atera_render_key(state):
                self._atera_render_viewport(state)
            return
        padded = pad_bounds(bounds, ATERA_VIEWPORT_PAD_FRACTION)
        estimate = estimate_window_rows(ctx.overview, padded, codes, exclude)
        if estimate > ctx.viewport_cap:
            ctx.viewport_cache = None
            self._atera_render_viewport(state)
            hint = (
                f"{ds}: zoom in to stream full-density transcripts here "
                f"(about {estimate / 1e6:,.1f}M in view; viewport cap {ctx.viewport_cap / 1e6:,.1f}M)."
            )
            if hint.split("(")[0] != ctx.last_hint.split("(")[0]:
                self._set_status(hint)
            ctx.last_hint = hint
            return
        ctx.last_hint = ""
        last = ctx.view_request
        if (
            ctx.view_runner.busy
            and last is not None
            and last["exclude"] == exclude
            and code_set <= last["codes"]
            and bounds_contain(last["bounds"], bounds)
        ):
            return  # the read in flight already covers this view
        ctx.view_request = {"bounds": padded, "codes": code_set, "exclude": exclude}
        ctx.view_runner.submit(ctx.view_request)

    def _atera_start_view_read(self, ds: str, request: dict, generation: int, cancel: Event) -> None:
        state = self._gene_inspector_states.get(str(ds).upper())
        ctx = None if state is None else getattr(state, "atera", None)
        if ctx is None:
            return
        cap = int(ctx.viewport_cap)

        def compute():
            t0 = time.perf_counter()
            rows = ctx.engine.read_window_rows(
                request["bounds"], sorted(request["codes"]), cancel, exclude_below=request["exclude"]
            )
            if rows is None:
                return None
            rows = cap_rows_by_key(rows, cap)
            log.debug("[%s] Viewport read: %d rows in %.2fs", ds, len(rows["x"]), time.perf_counter() - t0)
            return rows

        def done(rows, gen=generation):
            self._atera_view_read_done(ds, ctx, gen, request, rows)

        def failed(exc, gen=generation):
            ctx.view_runner.finish(gen)
            message = exc[1] if isinstance(exc, tuple) and len(exc) > 1 else exc
            if not isinstance(message, TranscriptBuildCancelled):
                log.warning("[%s] Atera viewport read failed: %s", ds, message)

        self._atera_run(compute, done, failed)

    def _atera_view_read_done(self, ds: str, ctx, generation: int, request: dict, rows) -> None:
        current = ctx.view_runner.finish(generation)
        state = self._gene_inspector_states.get(str(ds).upper())
        if not current or state is None or getattr(state, "atera", None) is not ctx or rows is None:
            return
        ctx.viewport_cache = {**request, "rows": rows}
        self._atera_render_viewport(state)

    def _atera_viewport_rgba(self, state) -> np.ndarray:
        """Per-global-code RGBA: slot colours, else the current scheme."""
        ctx = state.atera
        vocabulary = ctx.overview.vocabulary
        table = np.zeros((len(vocabulary), 4), dtype=np.float32)
        visuals = state.gene_visuals or {}
        for i, name in enumerate(vocabulary):
            visual = self._selection_slot_visual(state, name) or visuals.get(name)
            if visual is not None:
                table[i] = np.asarray(visual.rgba, dtype=np.float32)
            else:
                table[i] = (0.7, 0.7, 0.7, 1.0)
        return table

    def _atera_render_key(self, state) -> tuple:
        """What the viewport layer's content depends on (besides the cached rows)."""
        ctx = state.atera
        allocator = getattr(state, "slot_allocator", None)
        return (
            id(ctx.viewport_cache),
            tuple(int(c) for c in self._atera_display_codes(state)),
            bool(state.hide_assigned), bool(state.hide_background), bool(state.colour_by_assignment),
            float(state.spot_size), tuple(state.highlighted_genes or ()), state.color_kind,
            tuple(getattr(allocator, "genes", ()) or ()),
        )

    def _atera_render_viewport(self, state) -> None:
        """Push the cached window rows that are displayable now into the viewport layer."""
        from .viewer import GENE_ASSIGNED_RGBA, GENE_HIGHLIGHT_SPOT_SIZE, GENE_UNASSIGNED_RGBA

        ctx = state.atera
        layer = self._get_layer_by_name(ATERA_VIEWPORT_LAYER)
        if layer is None:
            return
        cache = ctx.viewport_cache
        codes = self._atera_display_codes(state)
        if cache is None or not len(codes):
            keep = None
            rows = None
        else:
            rows = cache["rows"]
            keep = np.isin(rows["code"], codes)
            assigned = rows["cell_id"] != 0
            if state.hide_assigned:
                keep &= ~assigned
            if state.hide_background:
                keep &= assigned
        if rows is None or keep is None or not keep.any():
            coords = np.zeros((0, 2), dtype=np.float32)
            colors = np.zeros((0, 4), dtype=np.float32)
            point_codes = np.zeros(0, dtype=np.int32)
            sizes: Any = float(state.spot_size)
        else:
            idx = np.flatnonzero(keep)
            point_codes = rows["code"][idx].astype(np.int32)
            coords = np.empty((len(idx), 2), dtype=np.float32)
            coords[:, 0] = rows["y"][idx]
            coords[:, 1] = rows["x"][idx]
            if state.colour_by_assignment:
                colors = np.where(
                    (rows["cell_id"][idx] != 0)[:, None], GENE_ASSIGNED_RGBA[None, :], GENE_UNASSIGNED_RGBA[None, :]
                ).astype(np.float32)
            else:
                colors = self._atera_viewport_rgba(state)[point_codes]
            sizes = float(state.spot_size)
            highlighted = set(state.highlighted_genes or [])
            if highlighted:
                vocab_codes = {name: i for i, name in enumerate(ctx.overview.vocabulary)}
                hot = [vocab_codes[g] for g in highlighted if g in vocab_codes]
                if hot and np.isin(point_codes, hot).any():
                    sizes = np.where(np.isin(point_codes, hot), GENE_HIGHLIGHT_SPOT_SIZE, state.spot_size).astype(np.float32)
        ctx.viewport_codes = point_codes
        ctx.rendered_key = self._atera_render_key(state)
        t0 = time.perf_counter()
        self._atera_set_viewport_data(layer, coords, colors, sizes)
        log.debug("[%s] Viewport layer: %d points in %.2fs", state.dataset, len(coords), time.perf_counter() - t0)

    def _atera_set_viewport_data(self, layer, coords, colors, sizes) -> None:
        """Replace the persistent viewport layer's points with one vispy upload.

        Every per-point change (data, colours, size, shown) would otherwise
        re-upload the whole vertex buffer.  The layer's and its colour
        managers' events are blocked while the arrays are swapped (the view
        slice is recomputed synchronously, since asynchronous slicing would
        leave it indexing the old points), then one refresh draws the result.
        """
        blockers = []
        for owner in (layer, getattr(layer, "_face", None), getattr(layer, "_border", None)):
            events = getattr(owner, "events", None)
            blocker = getattr(events, "blocker_all", None) if events is not None else None
            if callable(blocker):
                blockers.append(blocker())
        with ExitStack() as stack:
            for blocker in blockers:
                stack.enter_context(blocker)
            layer.data = coords
            reslice = getattr(layer, "set_view_slice", None)
            if callable(reslice):
                try:
                    reslice()
                except Exception as exc:  # pragma: no cover - napari internals
                    log.debug("Viewport re-slice failed: %s", exc)
            if len(coords):
                self._set_gene_layer_colors(layer, colors, "disc")
            for name, value in (("symbol", "disc"), ("size", sizes), ("shown", np.ones(len(coords), dtype=bool))):
                try:
                    setattr(layer, name, value)
                except Exception:
                    pass
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    def _atera_pick_viewport_gene(self, state, event) -> str | None:
        ctx = getattr(state, "atera", None)
        if ctx is None or not len(ctx.viewport_codes):
            return None
        layer = self._get_layer_by_name(ATERA_VIEWPORT_LAYER)
        if layer is None or not bool(getattr(layer, "visible", True)):
            return None
        idx = self._picked_point_index(layer, event)
        if idx is None or idx >= len(ctx.viewport_codes):
            return None
        return str(ctx.overview.vocabulary[int(ctx.viewport_codes[idx])])

    # -- worker plumbing --------------------------------------------------

    def _atera_run(self, compute, done, failed) -> None:
        """Run ``compute`` off the Qt thread; a cancelled read reports quietly."""

        def guarded():
            try:
                return compute()
            except TranscriptBuildCancelled as exc:
                return _Cancelled(exc)

        def returned(result):
            if isinstance(result, _Cancelled):
                failed(result.exc)
            else:
                done(result)

        factory = _thread_worker()
        if factory is None:
            try:
                result = guarded()
            except Exception as exc:
                failed(exc)
                return
            returned(result)
            return
        worker = factory(guarded)()
        worker.returned.connect(returned)
        worker.errored.connect(failed)
        keep = getattr(self, "_atera_workers", None)
        if keep is None:
            keep = set()
            self._atera_workers = keep
        keep.add(worker)
        try:
            worker.finished.connect(lambda w=worker: keep.discard(w))
        except Exception:
            pass
        worker.start()
