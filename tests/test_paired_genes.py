"""Focused tests for the paired transcript visual/control contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from napari_compare_xenium_merscope import viewer as V
from napari_compare_xenium_merscope.utils import build_gene_point_groups


def _store(*genes: str, background=None):
    if background is None:
        background = [False] * len(genes)
    frame = pd.DataFrame(
        {
            "x": np.arange(len(genes), dtype=np.float32) + 0.25,
            "y": np.arange(len(genes), dtype=np.float32) + 10.5,
            "gene": list(genes),
            "background": list(background),
        }
    )
    return build_gene_point_groups(
        frame,
        x_col="x",
        y_col="y",
        gene_col="gene",
        background_col="background",
    )


def _state(dataset: str, store, reference=None):
    return SimpleNamespace(
        dataset=dataset,
        store=store,
        reference=reference,
        show_controls=False,
        highlighted_genes=[],
    )


def _gene_coords(store, gene: str) -> np.ndarray:
    group, start, _foreground_end, end = store.gene_offsets[gene]
    return np.asarray(store.group_coords[group][start:end]).copy()


def _assert_gene_uses_visual(state, gene: str) -> None:
    visual = state.gene_visuals[gene]
    group, start, _foreground_end, end = state.store.gene_offsets[gene]
    assert state.store.group_symbols[group] == visual.symbol
    expected = np.broadcast_to(
        np.asarray(visual.rgba, dtype=np.float32),
        (end - start, 4),
    )
    np.testing.assert_allclose(
        state.store.group_colors[group][start:end],
        expected,
    )


def test_paired_unequal_panels_are_regrouped_with_one_union_visual_scheme():
    # SHARED starts at a different alphabetical index in each local panel, so
    # independent builds deliberately give it a different symbol and colour.
    merscope_store = _store(
        "SHARED", "ZZZ", "SHARED", background=[True, False, False]
    )
    xenium_store = _store("AAA", "SHARED")
    assert (
        merscope_store.gene_visuals["SHARED"]
        != xenium_store.gene_visuals["SHARED"]
    )
    original_coords = {
        "MERSCOPE": {
            gene: _gene_coords(merscope_store, gene)
            for gene in merscope_store.genes
        },
        "XENIUM": {
            gene: _gene_coords(xenium_store, gene)
            for gene in xenium_store.genes
        },
    }
    original_counts = {
        "MERSCOPE": {
            gene: (
                merscope_store.gene_offsets[gene][2]
                - merscope_store.gene_offsets[gene][1],
                merscope_store.gene_offsets[gene][3]
                - merscope_store.gene_offsets[gene][2],
            )
            for gene in merscope_store.genes
        },
        "XENIUM": {
            gene: (
                xenium_store.gene_offsets[gene][2]
                - xenium_store.gene_offsets[gene][1],
                xenium_store.gene_offsets[gene][3]
                - xenium_store.gene_offsets[gene][2],
            )
            for gene in xenium_store.genes
        },
    }
    merscope = _state("MERSCOPE", merscope_store)
    xenium = _state("XENIUM", xenium_store)
    controller = V.ComparisonViewerController.__new__(
        V.ComparisonViewerController
    )

    controller._harmonize_paired_gene_states([merscope, xenium])

    union = ["AAA", "SHARED", "ZZZ"]
    for state in (merscope, xenium):
        assert state.panel_genes == union
        assert state.enabled_genes == set(union)
        assert state.store.genes == (
            ["SHARED", "ZZZ"]
            if state.dataset == "MERSCOPE"
            else ["AAA", "SHARED"]
        )
        for gene in state.store.genes:
            _assert_gene_uses_visual(state, gene)
            np.testing.assert_allclose(
                _gene_coords(state.store, gene),
                original_coords[state.dataset][gene],
            )
            _group, start, foreground_end, end = state.store.gene_offsets[gene]
            assert (
                foreground_end - start,
                end - foreground_end,
            ) == original_counts[state.dataset][gene]

    assert merscope.gene_visuals == xenium.gene_visuals
    assert (
        merscope.store.gene_visuals["SHARED"]
        == xenium.store.gene_visuals["SHARED"]
    )
    assert merscope.panel_gene_counts == xenium.panel_gene_counts == {
        "AAA": 1,
        "SHARED": 3,
        "ZZZ": 1,
    }
    layout, _labels = controller._gene_ordering_layout(merscope)
    assert [entry[1] for entry in layout if entry[0] == "gene"] == union


def test_paired_identical_panels_reuse_point_arrays_during_harmonization():
    merscope = _state("MERSCOPE", _store("AAA", "SHARED", "AAA"))
    xenium = _state("XENIUM", _store("AAA", "SHARED"))
    states = (merscope, xenium)
    array_ids = {
        state.dataset: (
            [id(array) for array in state.store.group_coords],
            [id(array) for array in state.store.group_colors],
        )
        for state in states
    }
    controller = V.ComparisonViewerController.__new__(
        V.ComparisonViewerController
    )

    controller._harmonize_paired_gene_states(list(states))

    for state in states:
        coord_ids, color_ids = array_ids[state.dataset]
        assert [id(array) for array in state.store.group_coords] == coord_ids
        assert [id(array) for array in state.store.group_colors] == color_ids
        for gene in state.store.genes:
            _assert_gene_uses_visual(state, gene)


def _conflicting_reference_states():
    merscope = _state(
        "MERSCOPE",
        _store("M_ONLY", "SHARED"),
        {
            "M_ONLY": {"broad": "M broad", "fine": "M fine"},
            "SHARED": {"broad": "M broad", "fine": "M fine"},
        },
    )
    xenium = _state(
        "XENIUM",
        _store("SHARED", "X_ONLY"),
        {
            "SHARED": {"broad": "X broad", "fine": "X fine"},
            "X_ONLY": {"broad": "X broad", "fine": "X fine"},
        },
    )
    return merscope, xenium


def test_paired_union_visuals_are_deterministic_when_state_order_changes():
    controller = V.ComparisonViewerController.__new__(
        V.ComparisonViewerController
    )
    forward = _conflicting_reference_states()
    reverse = _conflicting_reference_states()

    controller._harmonize_paired_gene_states(list(forward))
    controller._harmonize_paired_gene_states(list(reversed(reverse)))

    for states in (forward, reverse):
        merscope, xenium = states
        assert merscope.reference["SHARED"] == {
            "broad": "M broad",
            "fine": "M fine",
        }
        assert merscope.reference == xenium.reference
        assert merscope.coarse_scheme.visuals == xenium.coarse_scheme.visuals
        assert merscope.fine_scheme.visuals == xenium.fine_scheme.visuals
        assert (
            merscope.store.gene_visuals["SHARED"]
            == xenium.store.gene_visuals["SHARED"]
        )

    assert forward[0].coarse_scheme.visuals == reverse[0].coarse_scheme.visuals
    assert forward[0].fine_scheme.visuals == reverse[0].fine_scheme.visuals

    controller._paired_session = object()
    controller._gene_inspector_states = {
        state.dataset: state for state in forward
    }
    recolored = []
    controller._recolor_gene_group_layers = (
        lambda state, group_indices=None: recolored.append(state.dataset)
    )
    controller._populate_gene_inspector = lambda _state: None
    controller.set_gene_ordering("PAIRED", "fine")
    assert recolored == ["MERSCOPE", "XENIUM"]
    for state in forward:
        _assert_gene_uses_visual(state, "SHARED")
    assert (
        forward[0].store.gene_visuals["SHARED"]
        == forward[1].store.gene_visuals["SHARED"]
    )


def test_paired_platform_only_gene_toggle_updates_union_state_and_local_store():
    merscope, xenium = _conflicting_reference_states()
    controller = V.ComparisonViewerController.__new__(
        V.ComparisonViewerController
    )
    controller._harmonize_paired_gene_states([merscope, xenium])
    controller._paired_session = object()
    controller._gene_inspector_states = {
        "MERSCOPE": merscope,
        "XENIUM": xenium,
    }
    rebuilt = []
    controller._schedule_gene_group_rebuild = (
        lambda state, group: rebuilt.append((state.dataset, group))
    )
    merscope.enabled_genes.discard("M_ONLY")
    xenium.enabled_genes.discard("M_ONLY")

    controller.set_gene_visible("MERSCOPE", "M_ONLY", True)

    assert "M_ONLY" in merscope.enabled_genes
    assert "M_ONLY" in xenium.enabled_genes
    assert rebuilt == [
        ("MERSCOPE", merscope.store.gene_offsets["M_ONLY"][0])
    ]
