"""Focused tests for prepared dataset session reuse."""

from __future__ import annotations

import types
from collections import OrderedDict
from pathlib import Path

import numpy as np

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


def _session(name: str, retained_bytes: int) -> V.DatasetSession:
    store = types.SimpleNamespace(
        group_coords=[np.empty((retained_bytes // 4, 1), dtype=np.float32)],
        group_colors=[],
        cell_transcript_index=None,
    )
    return V.DatasetSession(
        dataset=name,
        config=V.DatasetConfig(name=name, zarr_path=Path(f"/{name}.zarr")),
        sdata=object(),
        images_sdata=None,
        x_transform=(1.0, 0.0, 0.0),
        y_transform=(1.0, 0.0, 0.0),
        segmentation_keys=[],
        image_keys=[],
        image_channels=[],
        gene_payload={"store": store},
    )


def test_session_lru_evicts_inactive_prepared_arrays_first():
    ctrl = _controller(session_cache_gb=0)
    ctrl.active_dataset = "NEW"
    old = _session("OLD", 64)
    new = _session("NEW", 64)
    ctrl._dataset_sessions = OrderedDict((("OLD", old), ("NEW", new)))

    ctrl._evict_dataset_sessions()

    assert list(ctrl._dataset_sessions) == ["NEW"]
