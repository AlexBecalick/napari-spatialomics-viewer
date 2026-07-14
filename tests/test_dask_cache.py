from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import dask
from cachey import Cache as CacheyCache

from napari_compare_xenium_merscope.dask_cache import ThreadSafeDaskCache


def test_overlapping_computations_do_not_corrupt_callback_state():
    """Finishing one graph must not clear another graph's task timing."""
    cache = ThreadSafeDaskCache(CacheyCache(1024 * 1024))
    slow_started = Event()
    release_slow = Event()

    def slow_task():
        slow_started.set()
        assert release_slow.wait(timeout=5)
        return 1

    def run_slow():
        with cache:
            return dask.get({"slow": (slow_task,)}, "slow")

    def run_fast():
        assert slow_started.wait(timeout=5)
        with cache:
            return dask.get({"fast": (lambda: 2,)}, "fast")

    with ThreadPoolExecutor(max_workers=2) as executor:
        slow_future = executor.submit(run_slow)
        fast_future = executor.submit(run_fast)
        assert fast_future.result(timeout=5) == 2
        release_slow.set()
        assert slow_future.result(timeout=5) == 1

    assert cache._active_contexts == 0
    assert not cache._starttimes_by_state
    assert not cache._durations_by_state


def test_completed_tasks_are_reused_from_memory_cache():
    cache = ThreadSafeDaskCache(CacheyCache(1024 * 1024))
    calls = 0
    calls_lock = Lock()

    def expensive_task():
        nonlocal calls
        with calls_lock:
            calls += 1
        return b"cached viewport chunk"

    graph = {"viewport-chunk": (expensive_task,)}
    with cache:
        assert dask.get(graph, "viewport-chunk") == b"cached viewport chunk"
    with cache:
        assert dask.get(graph, "viewport-chunk") == b"cached viewport chunk"

    assert calls == 1
    assert "viewport-chunk" in cache.cache.data
