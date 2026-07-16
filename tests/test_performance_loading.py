"""Focused tests for responsive loading and task deduplication."""

from __future__ import annotations

import types

from napari_compare_xenium_merscope import viewer as V


class _Viewer:
    def __init__(self):
        self.layers = []
        self.mouse_drag_callbacks = []


def _controller(**args):
    defaults = dict(background_io_workers=1, session_cache_gb=0)
    defaults.update(args)
    return V.ComparisonViewerController(
        _Viewer(), {}, types.SimpleNamespace(**defaults)
    )


def test_task_tokens_deduplicate_and_cancel_superseded_work():
    ctrl = _controller()
    first = ctrl._begin_task_token("images", ("TEST", "DAPI"))

    assert first is not None
    assert ctrl._begin_task_token("images", ("TEST", "DAPI")) is None

    replacement = ctrl._begin_task_token("images", ("TEST", "PolyT"))
    assert replacement is not None
    assert first.is_set()
    assert not replacement.is_set()


def test_image_load_installs_preview_before_materialized_replacement(monkeypatch):
    ctrl = _controller(
        hide_images=False,
        image_pyramid_downsample=4,
        overwrite_derived_caches=False,
    )
    ctrl.active_dataset = "TEST"
    ctrl._active_sdata = object()
    ctrl._active_images_sdata = object()
    ctrl._x_transform = (1.0, 0.0, 0.0)
    ctrl._y_transform = (1.0, 0.0, 0.0)
    calls: list[bool] = []
    primed: list[bool] = []

    monkeypatch.setattr(V, "thread_worker", None)
    monkeypatch.setattr(ctrl, "_ensure_dataset_is_active", lambda _ds: True)
    monkeypatch.setattr(ctrl, "_ensure_images_loaded", lambda _ds: None)
    monkeypatch.setattr(
        ctrl,
        "_add_image_layers",
        lambda *args, build_cache=True, **kwargs: calls.append(build_cache)
        or {"layers": 1, "failed_keys": 0, "skipped": False},
    )
    monkeypatch.setattr(
        ctrl,
        "_prime_image_pyramid_caches",
        lambda *args, **kwargs: primed.append(True),
    )

    ctrl.load_images_on_demand("TEST", [("mosaic", "DAPI")])

    assert calls == [False, True]
    assert primed == [True]
