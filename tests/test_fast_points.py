"""FastPoints: a one-element ``_view_symbol`` only while every point shares one symbol."""

from __future__ import annotations

import numpy as np

from napari_compare_xenium_merscope.fast_points import FastPoints


def _layer(n=50, symbol="disc"):
    layer = FastPoints(np.random.default_rng(0).random((n, 2)), symbol=symbol)
    _in_view(layer)
    return layer


def _in_view(layer):
    # What napari's slice step does; no viewer or canvas needed.
    layer._slicing_state._indices_view = np.arange(len(layer.data))


def _names(layer):
    return [str(x) for x in layer._view_symbol]


def test_one_symbol_is_one_view_entry_and_keeps_napari_semantics():
    layer = _layer()
    assert _names(layer) == ["disc"]
    assert layer._type_string == "points"
    assert len(layer.symbol) == 50 and {str(s) for s in layer.symbol} == {"disc"}
    layer.symbol = "square"
    assert _names(layer) == ["square"]


def test_a_mixed_symbol_array_falls_back_to_one_entry_per_point():
    layer = _layer(4)
    layer.symbol = np.array(["disc", "square", "disc", "disc"])
    assert _names(layer) == ["disc", "square", "disc", "disc"]
    layer.symbol = np.array(["star"] * 4)  # an array that happens to be uniform
    assert _names(layer) == ["star"]


def test_an_in_place_edit_of_selected_points_is_seen():
    layer = _layer(5)
    assert _names(layer) == ["disc"]
    layer.selected_data = {1, 3}
    layer.current_symbol = "star"
    assert _names(layer) == ["disc", "star", "disc", "star", "disc"]


def test_adding_and_removing_points_rechecks_uniformity():
    layer = _layer(5)
    layer.current_symbol = "cross"  # nothing selected: no in-place edit
    assert _names(layer) == ["disc"]
    layer.add(np.array([[0.5, 0.5]]))
    _in_view(layer)
    assert _names(layer)[-1] == "cross" and len(_names(layer)) == 6
    layer.selected_data = {5}
    layer.remove_selected()
    _in_view(layer)
    assert _names(layer) == ["disc"]


def test_nothing_in_view_keeps_napari_behaviour():
    layer = FastPoints(np.zeros((3, 2)), symbol="disc", shown=np.zeros(3, dtype=bool))
    _in_view(layer)
    assert len(layer._view_symbol) == 0


def test_the_data_extent_is_cached_until_the_points_change(monkeypatch):
    layer = FastPoints(np.array([[0.0, 0.0], [10.0, 20.0]]), symbol="disc")
    calls = []
    original = FastPoints.__mro__[1]._extent_data.fget
    monkeypatch.setattr(FastPoints.__mro__[1], "_extent_data",
                        property(lambda self: (calls.append(1), original(self))[1]))
    first = layer._extent_data
    np.testing.assert_array_equal(first, [[0, 0], [10, 20]])
    first[0, 0] = -99  # callers may modify the returned array
    layer._clear_extent()  # what every napari async slice does
    np.testing.assert_array_equal(layer._extent_data, [[0, 0], [10, 20]])
    assert len(calls) == 1
    layer.data = np.array([[1.0, 1.0], [5.0, 5.0]])
    np.testing.assert_array_equal(layer._extent_data, [[1, 1], [5, 5]])
    layer.add(np.array([[30.0, 40.0]]))
    np.testing.assert_array_equal(layer._extent_data, [[1, 1], [30, 40]])
    # Dragging selected points edits the same array in place.
    layer._slicing_state._indices_view = np.arange(3)
    layer.selected_data = {2}
    layer._move([2], np.array([30.0, 40.0]))  # sets the drag start
    layer._move([2], np.array([50.0, 60.0]))
    np.testing.assert_array_equal(layer._extent_data[1], layer.data.max(axis=0))
    assert layer.data[2].tolist() != [30.0, 40.0]
    assert len(calls) == 4
    assert layer.extent.data.tolist() == layer._extent_data.tolist()
