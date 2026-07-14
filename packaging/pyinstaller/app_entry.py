"""Entry point for self-contained macOS, Windows, and Linux applications."""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from pathlib import Path


SMOKE_TEST_LOG_ENV = "NAPARI_COMPARE_SMOKE_TEST_LOG"


def _finish_smoke_test(exit_code: int, error: str | None = None) -> None:
    """Terminate frozen smoke tests without waiting for GUI worker threads."""
    log_path = os.environ.get(SMOKE_TEST_LOG_ENV)
    if error is not None and log_path:
        try:
            Path(log_path).write_text(error, encoding="utf-8")
        except OSError:
            pass
    os._exit(exit_code)


def main() -> None:
    multiprocessing.freeze_support()
    from napari_compare_xenium_merscope.viewer import main as viewer_main

    smoke_test = "--package-smoke-test" in sys.argv
    try:
        viewer_main()
    except BaseException:
        if smoke_test:
            _finish_smoke_test(1, traceback.format_exc())
        raise
    if smoke_test:
        _finish_smoke_test(0)


if __name__ == "__main__":
    main()
