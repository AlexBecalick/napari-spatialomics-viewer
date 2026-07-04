"""Tests for the micron scale-bar overlay and zoom-aware label interpolation."""
from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from napari_compare_xenium_merscope import viewer as V


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("qtpy/Qt not available")
    return QApplication.instance() or QApplication([])


def _fake_zoom_viewer(zoom: float, size=(1000.0, 800.0)):
    canvas = types.SimpleNamespace(size=size, native=None)
    return types.SimpleNamespace(
        layers=[],
        mouse_drag_callbacks=[],
        camera=types.SimpleNamespace(
            zoom=zoom,
            events=types.SimpleNamespace(zoom=types.SimpleNamespace(connect=lambda *a, **k: None)),
        ),
        window=types.SimpleNamespace(_qt_viewer=types.SimpleNamespace(canvas=canvas)),
    )


def _controller(viewer, interpolation="linear"):
    return V.ComparisonViewerController(
        viewer, {}, types.SimpleNamespace(label_interpolation=interpolation)
    )


def test_scale_bar_format_um():
    fmt = V.ScaleBarOverlay._format_um
    assert fmt(250.0) == "250 µm"
    assert fmt(12.5) == "12.5 µm"
    assert fmt(3.14159) == "3.14 µm"
    assert fmt(1500.0) == "1500 µm"  # stays in microns as requested


def test_scale_bar_overlay_sizing_and_position(qapp):
    from qtpy.QtWidgets import QWidget

    parent = QWidget()
    parent.resize(1200, 900)
    bar = V.ScaleBarOverlay(parent, width_fraction=0.25, margin=28)
    bar.reposition()
    assert bar._bar_px == int(1200 * 0.25)  # a quarter of the canvas width
    geom = bar.geometry()
    # Pinned toward the bottom-right corner with a margin.
    assert geom.x() > parent.width() / 2
    assert geom.y() > parent.height() / 2
    assert parent.width() - geom.right() >= 20
    bar.set_um_per_px(2.0)  # 300 px * 2 µm/px = 600 µm; just ensure no paint crash
    bar.repaint()


def test_label_interpolation_toggles_with_zoom(qapp):
    viewer = _fake_zoom_viewer(zoom=8.0)  # min(1000,800)=800; 800/8 = 100 µm <= 150
    ctrl = _controller(viewer, "linear")
    layer = types.SimpleNamespace(
        name="TEST | labels | MOSAIK_proseg_labels", interpolation2d="linear"
    )
    viewer.layers.append(layer)

    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"  # zoomed in past 150 µm -> crisp

    viewer.camera.zoom = 1.0  # 800 µm visible -> smooth
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "linear"


def test_label_interpolation_respects_explicit_nearest(qapp):
    viewer = _fake_zoom_viewer(zoom=1.0)  # 800 µm visible (would be 'linear' if auto)
    ctrl = _controller(viewer, "nearest")
    layer = types.SimpleNamespace(name="TEST | labels | seg", interpolation2d="nearest")
    viewer.layers.append(layer)
    ctrl._apply_label_zoom_interpolation()
    assert layer.interpolation2d == "nearest"  # explicit choice not overridden


def test_install_canvas_overlays_without_canvas_is_safe(qapp):
    viewer = _fake_zoom_viewer(zoom=4.0)
    ctrl = _controller(viewer, "linear")
    # canvas.native is None here -> scale bar skipped, camera hook + initial update
    # must still run without raising.
    ctrl.install_canvas_overlays()
    assert ctrl._scale_bar is None
