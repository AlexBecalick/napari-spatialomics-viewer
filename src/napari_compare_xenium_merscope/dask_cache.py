"""Concurrency-safe opportunistic Dask caching for napari raster layers."""

from __future__ import annotations

import sys
from threading import RLock
from time import perf_counter
from typing import Any

from dask.cache import Cache, overhead
from dask.callbacks import Callback


class ThreadSafeDaskCache(Cache):
    """A Dask ``Cache`` that supports overlapping computations.

    Dask's standard ``Cache`` stores task start times and durations in two
    dictionaries shared by every computation using that callback. Napari also
    shares one cache across all lazy layers, so concurrent image/label slices
    can clear each other's dictionaries and raise from ``Cache._posttask``.

    Keep the normal shared, bounded cachey store (and therefore cache hits
    across layers), but maintain callback bookkeeping separately for each Dask
    scheduler state. Cache access and callback registration are synchronized;
    actual array reads and computations remain concurrent.
    """

    def __init__(self, cache: Any):
        super().__init__(cache)
        self._lock = RLock()
        self._active_contexts = 0
        self._starttimes_by_state: dict[int, dict[Any, float]] = {}
        self._durations_by_state: dict[int, dict[Any, float]] = {}

    def __enter__(self):
        # Callback.active is a process-wide set. Reference-count overlapping
        # contexts so one layer cannot unregister the callback while another
        # layer is still slicing.
        with self._lock:
            if self._active_contexts == 0:
                Callback.active.add(self._callback)
            self._active_contexts += 1
        return self

    def __exit__(self, *_args):
        with self._lock:
            if self._active_contexts <= 0:
                raise RuntimeError("Unbalanced ThreadSafeDaskCache context exit")
            self._active_contexts -= 1
            if self._active_contexts == 0:
                Callback.active.discard(self._callback)

    def _start(self, dsk):
        # Substituting cached task values is the hot path that makes revisiting
        # a viewport avoid repeated Zarr I/O and decompression.
        with self._lock:
            overlap = set(dsk).intersection(self.cache.data)
            for key in overlap:
                dsk[key] = self.cache.data[key]

    def _pretask(self, key, _dsk, state):
        state_id = id(state)
        with self._lock:
            self._starttimes_by_state.setdefault(state_id, {})[key] = perf_counter()

    def _posttask(self, key, value, _dsk, state, _worker_id):
        state_id = id(state)
        finished = perf_counter()
        with self._lock:
            starttimes = self._starttimes_by_state.setdefault(state_id, {})
            durations = self._durations_by_state.setdefault(state_id, {})
            started = starttimes.pop(key, finished)
            duration = finished - started
            dependencies = state.get("dependencies", {}).get(key, ())
            if dependencies:
                duration += max((durations.get(dep, 0.0) for dep in dependencies), default=0.0)
            durations[key] = duration

            # Concurrent graphs can finish the same key. Cachey does not
            # replace an existing key atomically, so retain the first result
            # instead of double-counting its bytes.
            if key in self.cache.data:
                return
            nbytes = self._nbytes(value) + overhead + sys.getsizeof(key) * 4
            self.cache.put(key, value, cost=duration / nbytes / 1e9, nbytes=nbytes)

    def _finish(self, _dsk, state, _errored):
        state_id = id(state)
        with self._lock:
            self._starttimes_by_state.pop(state_id, None)
            self._durations_by_state.pop(state_id, None)


def install_thread_safe_napari_dask_cache() -> ThreadSafeDaskCache:
    """Replace napari's shared Dask callback while preserving its RAM budget."""
    from napari.utils import _dask_utils

    current = _dask_utils._DASK_CACHE
    if isinstance(current, ThreadSafeDaskCache):
        return current

    safe_cache = ThreadSafeDaskCache(current.cache)
    _dask_utils._DASK_CACHE = safe_cache
    # Match napari's default behavior (25% of physical RAM), but initialize it
    # now so the startup diagnostic reports the real budget. Cachey allocates
    # memory only as viewport chunks are inserted.
    return _dask_utils.resize_dask_cache()
