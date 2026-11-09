"""Determinate progress for the control panel's shared progress bar.

Two small, Qt-free pieces:

* :class:`ProgressReporter` sits on the controller.  Worker threads call
  :meth:`ProgressReporter.report` with ``(done, total)`` counts; it keeps each
  stage's values monotonic within a phase and throttles them to at most one
  update per :data:`PROGRESS_MIN_INTERVAL_S` (the first and the final value of
  a phase always pass), so a 409-file pass never floods the Qt event queue.
* :class:`ProgressDisplay` sits on the panel.  It combines the active stages
  into one bar (determinate for the first stage that got a total, until it
  ends; else busy) and one label such as
  ``"ATERA: streaming transcripts — 123/409 files (30%), about 40 s left"``.

The controller marshals updates to the GUI thread through a queued Qt signal,
so neither class touches widgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable

#: At most this many seconds between two forwarded updates of one stage (~10/s).
PROGRESS_MIN_INTERVAL_S = 0.1
#: An ETA is shown once a phase has run this long and is this far along.
PROGRESS_ETA_MIN_ELAPSED_S = 2.0
PROGRESS_ETA_MIN_FRACTION = 0.02
#: QProgressBar works in int32; larger totals are scaled to this many steps.
PROGRESS_BAR_STEPS = 1000
_INT32_MAX = 2**31 - 1


def format_count(done: int, total: int, unit: str = "") -> str:
    """``"123/409 files (30%)"``; ``unit="bytes"`` prints sizes."""
    done = int(done)
    total = int(total)
    pct = int(100 * done / total) if total > 0 else 0
    if unit == "bytes":
        return f"{_format_bytes(done)} of {_format_bytes(total)} ({pct}%)"
    suffix = f" {unit}" if unit else ""
    return f"{done:,}/{total:,}{suffix} ({pct}%)"


def _format_bytes(value: int) -> str:
    value = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024.0
    return f"{value:,.1f} GB"  # pragma: no cover - loop always returns


def format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"about {max(1, int(round(seconds)))} s left"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"about {minutes} min {secs:02d} s left"


@dataclass
class _Phase:
    total: int
    text: str
    done: int
    emitted_at: float


class ProgressReporter:
    """Thread-safe, throttled, per-stage ``(done, total)`` counts."""

    def __init__(
        self,
        emit: Callable[[str, int, int, str, str], None] | None = None,
        *,
        min_interval: float = PROGRESS_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.emit = emit
        self.min_interval = float(min_interval)
        self.clock = clock
        self._lock = Lock()
        self._phases: dict[str, _Phase] = {}

    def begin(self, key: str) -> None:
        """Start ``key`` afresh: the next report is never throttled or clamped."""
        with self._lock:
            self._phases.pop(str(key), None)

    end = begin

    def reset(self) -> None:
        with self._lock:
            self._phases.clear()

    def report(
        self,
        key: str,
        done: int,
        total: int,
        text: str | None = None,
        unit: str = "",
        *,
        force: bool = False,
    ) -> bool:
        """Forward ``done/total`` for ``key`` unless throttled; ``True`` if forwarded.

        ``total <= 0`` switches the stage back to a busy bar (with ``text``).
        Within one phase (same total and text) values never go backwards; a
        new total or text starts a new phase, whose first value always passes.
        """
        key = str(key)
        total = max(0, int(total))
        done = min(max(0, int(done)), total) if total else 0
        label = "" if text is None else str(text)
        now = self.clock()
        with self._lock:
            phase = self._phases.get(key)
            same = phase is not None and phase.total == total and phase.text == label
            if same:
                if done < phase.done or (done == phase.done and not force):
                    return False
                final = total > 0 and done >= total
                if not force and not final and now - phase.emitted_at < self.min_interval:
                    return False
            self._phases[key] = _Phase(total, label, done, now)
            # Emitted under the lock so two threads' values keep their order
            # (the emit is a queued Qt signal: cheap, never re-enters here).
            emit = self.emit
            if emit is not None:
                emit(key, done, total, label, str(unit or ""))
        return True


@dataclass
class _StageValue:
    done: int
    total: int
    text: str
    unit: str
    started_at: float
    updated_at: float


class ProgressDisplay:
    """What the panel's bar and label show for the active stages."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.stages: dict[str, str] = {}
        self.values: dict[str, _StageValue] = {}

    # -- updates -----------------------------------------------------------

    def set_stage(self, key: str, text: str, active: bool) -> None:
        key = str(key)
        self.values.pop(key, None)
        if active:
            self.stages[key] = str(text)
        else:
            self.stages.pop(key, None)

    def set_value(self, key: str, done: int, total: int, text: str = "", unit: str = "") -> bool:
        """Record a count for an active stage; ``False`` (ignored) for an inactive one."""
        key = str(key)
        if key not in self.stages:
            return False
        now = self.clock()
        old = self.values.get(key)
        total = max(0, int(total))
        if total <= 0:
            self.values.pop(key, None)
            if text:
                self.stages[key] = str(text)
            return True
        same_phase = old is not None and old.total == total and old.text == str(text)
        started = old.started_at if same_phase else now
        done = min(max(0, int(done)), total)
        if same_phase:
            done = max(done, old.done)
        self.values[key] = _StageValue(done, total, str(text), str(unit), started, now)
        return True

    # -- what to draw --------------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self.stages)

    def current(self) -> tuple[str, _StageValue] | None:
        """The determinate stage the bar follows.

        The bar stays with the first stage that got a total until that stage
        ends or goes busy again, so interleaved reports from concurrent stages
        (different totals) never make it jump back and forth; the other
        stages' counts are shown in the label only. ``values`` keeps the order
        in which stages first got a total (a new phase of the same stage keeps
        its place; a busy or ended stage is removed).
        """
        for key, value in self.values.items():
            if key in self.stages:
                return key, value
        return None

    def bar(self) -> tuple[int, int, int]:
        """``(minimum, maximum, value)``; ``(0, 0, 0)`` is Qt's busy bar."""
        current = self.current()
        if current is None:
            return 0, 0, 0
        value = current[1]
        if value.total <= _INT32_MAX:
            return 0, int(value.total), int(value.done)
        return 0, PROGRESS_BAR_STEPS, int(PROGRESS_BAR_STEPS * value.done / value.total)

    def stage_text(self, key: str) -> str:
        text = self.stages.get(key, "")
        value = self.values.get(key)
        if value is None:
            return text
        if value.text:
            text = value.text
        count = format_count(value.done, value.total, value.unit)
        eta = self._eta(value)
        return f"{text} — {count}{', ' + eta if eta else ''}" if text else count + (f", {eta}" if eta else "")

    def _eta(self, value: _StageValue) -> str:
        if value.done <= 0 or value.done >= value.total:
            return ""
        elapsed = value.updated_at - value.started_at
        fraction = value.done / value.total
        if elapsed < PROGRESS_ETA_MIN_ELAPSED_S or fraction < PROGRESS_ETA_MIN_FRACTION:
            return ""
        return format_eta(elapsed * (1.0 - fraction) / fraction)

    def label(self) -> str:
        return " | ".join(t for t in (self.stage_text(k) for k in self.stages) if t)
