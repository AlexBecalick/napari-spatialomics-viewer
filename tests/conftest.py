"""Test-wide setup: Qt fonts on Windows, and each test's Qt work ends with that test."""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# The test modules all default Qt to offscreen, but this file is imported first
# and on Windows creates the QApplication below: without this default it would
# start on the native platform, whose windows the runner's 1024x768 screen
# limits (a release build failed on that when QT_QPA_PLATFORM was unset).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Windows UI fonts: Segoe UI as the native viewer uses it, and Segoe UI Symbol
# for glyphs such as the gene markers.
_WINDOWS_TEST_FONTS = ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf", "seguisym.ttf")


def _give_offscreen_qt_windows_fonts() -> None:
    """Point Qt's offscreen platform at real fonts on Windows.

    On Windows the offscreen platform reads fonts only from PyQt5's own font
    directory, which the wheels leave empty ("Qt no longer ships fonts"), so
    every text metric is a fallback and the layout tests fail; the viewer
    itself runs on the native platform with the system fonts. Qt picks its
    default family from whatever the directory holds, so it gets only the
    Segoe UI files (or the whole system font directory if they are missing).
    Must run before the first QApplication.
    """
    if "QT_QPA_FONTDIR" in os.environ:
        return
    system_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = [system_fonts / name for name in _WINDOWS_TEST_FONTS if (system_fonts / name).is_file()]
    if not fonts:
        os.environ["QT_QPA_FONTDIR"] = str(system_fonts)
        return
    font_dir = Path(tempfile.mkdtemp(prefix="qt-test-fonts-"))
    for font in fonts:
        shutil.copy2(font, font_dir / font.name)
    os.environ["QT_QPA_FONTDIR"] = str(font_dir)


def _native_sized_windows_test_app():
    """The run's QApplication on offscreen Windows, with the native UI font.

    Offscreen Qt gives the default font 12 pt, while the viewer on Windows
    gets the system UI font, Segoe UI 9 pt; at 12 pt every text line is half
    as tall again and the window-height tests measure a window no user sees.
    An application font only lasts as long as its application, so this one is
    created here and kept for the run; the test modules' ``qapp`` fixtures
    (``QApplication.instance() or QApplication([])``) reuse it.
    """
    if sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM", "offscreen") != "offscreen":
        return None
    _give_offscreen_qt_windows_fonts()
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 9))
    return app


_WINDOWS_TEST_APP = _native_sized_windows_test_app()


_qt_text_setup: list[str] = []


def _record_qt_text_setup(app) -> None:
    """Remember the Qt platform and font the layout tests measured text with."""
    if _qt_text_setup or not hasattr(app, "font"):
        return
    from qtpy.QtGui import QFontInfo, QFontMetrics

    font = app.font()
    line = (
        f"Qt {app.platformName()}: font {QFontInfo(font).family()!r} {font.pointSizeF():g} pt, "
        f"line {QFontMetrics(font).lineSpacing()} px"
    )
    if "QT_QPA_FONTDIR" in os.environ:
        line += f", QT_QPA_FONTDIR={os.environ['QT_QPA_FONTDIR']}"
    _qt_text_setup.append(line)


def pytest_terminal_summary(terminalreporter) -> None:
    """Print the Qt text setup, which explains layout failures across platforms."""
    for line in _qt_text_setup:
        terminalreporter.write_line(line)


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
    _record_qt_text_setup(app)
    QThreadPool.globalInstance().waitForDone(30_000)
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
