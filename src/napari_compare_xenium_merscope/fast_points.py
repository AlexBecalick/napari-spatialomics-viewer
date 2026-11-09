"""A napari ``Points`` layer that keeps large gene layers cheap to draw.

Two napari costs scale with every point of a layer and recur far more often
than the points change; both are removed here without changing behaviour.

**Symbols.**
napari's vispy points visual builds ``[str(x) for x in layer._view_symbol]``
on every data change, a Python loop over every point in view. For a gene
layer of 11M points that loop alone takes about 3.5 s of the 4.4 s it takes
to show the layer. Every gene layer uses one symbol, and vispy broadcasts a
one-element symbol list to all points, so :class:`FastPoints`
returns a one-element ``_view_symbol`` while every point has the same symbol.

Uniformity is known without a scan when the symbol was set to one value
through the ``symbol`` setter (the normal case). Any other change to the
symbol array is caught because it either replaces the array (``add``,
``remove``, ``data``: a new ``id``/length) or goes through ``current_symbol``
on a selection (an in-place edit, invalidated here). Only then is one
vectorised scan run, and its result is cached for that array.

**Extent.** napari clears a layer's cached extent after every asynchronous
slice (``_refresh_sync(extent=True)``), and the next draw recomputes it as a
min/max over every point: about 0.2 s per 13M-point layer, three or four
times per layer while a 14-layer build is added. The data extent is cached
here per data array and dropped whenever napari changes the points (the
``data`` setter, ``add``/``remove`` (new arrays) and dragging selected points
in place), so napari's own invalidation still drives every recompute that a
real change needs.
"""

from __future__ import annotations

import numpy as np
from napari.layers import Points

try:  # napari >= 0.4.19
    from napari.layers.points._points_utils import coerce_symbols
except Exception:  # pragma: no cover - older napari
    coerce_symbols = None


class FastPoints(Points):
    """``Points`` with a constant-time one-symbol ``_view_symbol`` and a cached data extent."""

    # Class-level defaults: ``Points.__init__`` sets the symbol before any
    # instance attribute of this subclass could be assigned.
    _ncx_symbol_hint = None
    _ncx_symbol_cache: tuple | None = None
    _ncx_extent_cache: tuple | None = None

    @property
    def _type_string(self) -> str:
        # napari derives this from the class name; keep the layer a "points"
        # layer for its controls, context keys, writers and layer menus.
        return "points"

    @property
    def symbol(self) -> np.ndarray:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol) -> None:
        hint = None
        if coerce_symbols is not None:
            try:
                coerced = coerce_symbols(symbol)
                if coerced.size == 1:
                    hint = coerced.reshape(-1)[0]
            except Exception:
                hint = None
        self._ncx_symbol_cache = None
        self._ncx_symbol_hint = hint
        try:
            Points.symbol.fset(self, symbol)
        except BaseException:
            self._ncx_symbol_cache = None
            raise
        else:
            if hint is not None:
                self._bind_symbol_cache(hint)
        finally:
            self._ncx_symbol_hint = None

    @property
    def current_symbol(self):
        return Points.current_symbol.fget(self)

    @current_symbol.setter
    def current_symbol(self, symbol) -> None:
        # napari edits the selected points' symbols in place on this path.
        if getattr(self, "_update_properties", False) and len(getattr(self, "selected_data", ())) > 0:
            self._ncx_symbol_cache = None
        Points.current_symbol.fset(self, symbol)

    def _symbol_key(self) -> tuple:
        symbols = self._symbol
        return id(symbols), symbols.shape

    def _bind_symbol_cache(self, value) -> None:
        self._ncx_symbol_cache = (self._symbol_key(), value)

    def _uniform_symbol(self):
        """The one symbol every point has, else ``None``."""
        symbols = getattr(self, "_symbol", None)
        if symbols is None or len(symbols) == 0:
            return None
        cache = self._ncx_symbol_cache
        if cache is not None and cache[0] == self._symbol_key():
            return cache[1]
        if self._ncx_symbol_hint is not None:
            # Inside the ``symbol`` setter: the array was just filled with it.
            value = self._ncx_symbol_hint
        else:
            first = symbols[0]
            value = first if bool(np.all(symbols == first)) else None
        self._bind_symbol_cache(value)
        return value

    @property
    def _view_symbol(self) -> np.ndarray:
        value = self._uniform_symbol()
        if value is None or len(self._indices_view) == 0:
            return Points._view_symbol.fget(self)
        return np.array([value], dtype=object)

    # -- extent --------------------------------------------------------------

    @property
    def data(self):
        return Points.data.fget(self)

    @data.setter
    def data(self, data) -> None:
        self._ncx_extent_cache = None
        Points.data.fset(self, data)

    def _move(self, selection_indices, position) -> None:
        # napari moves the selected points in place (same array object).
        self._ncx_extent_cache = None
        super()._move(selection_indices, position)
        self._ncx_extent_cache = None

    @property
    def _extent_data(self) -> np.ndarray:
        data = self.data
        key = (id(data), data.shape)
        cache = self._ncx_extent_cache
        if cache is None or cache[0] != key:
            cache = (key, Points._extent_data.fget(self))
            self._ncx_extent_cache = cache
        # napari's own property returns a fresh array that callers may modify.
        return cache[1].copy()
