"""Test-wide setup: each test's Qt objects and workers end with that test."""

from __future__ import annotations

import gc
import sys

import pytest


@pytest.fixture(autouse=True)
def _finish_qt_work_on_the_main_thread():
    """Finish a test's background workers and free its Qt objects before the next test.

    Tests build controllers and widgets without running an event loop. Left
    alone, a test's ``thread_worker`` results arrive during a later test's
    event processing, after their controller is gone, and its ``deleteLater``
    calls and reference cycles are torn down whenever an allocation triggers
    the garbage collector, on whichever thread made it (zarr's I/O loop, a
    worker pool). Either can abort or segfault the interpreter far from the
    test that caused it. Doing this here, on the main thread, keeps any
    failure in the test that caused it.
    """
    yield
    if "qtpy.QtCore" not in sys.modules:
        return
    from qtpy.QtCore import QCoreApplication, QEvent, QThreadPool

    app = QCoreApplication.instance()
    if app is None:
        return
    QThreadPool.globalInstance().waitForDone(30_000)
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
