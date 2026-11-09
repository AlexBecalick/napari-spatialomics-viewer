"""Frozen copy of the per-row eager transcript build (before the M4 vectorisation).

Kept verbatim (apart from imports and the ``_frozen_`` prefixes) as the
reference that :func:`build_gene_point_groups` must reproduce bit for bit.
Do not "improve" this file: its value is that it never changes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from napari_compare_xenium_merscope.utils import (
    CellTranscriptIndex,
    GenePointStore,
    GeneVisual,
    _check_transcript_cancelled,
    _empty_gene_point_store,
    _unsigned_code_dtype,
    accumulate_is_gene_flags,
    assemble_gene_point_store,
    assignment_mask,
    control_genes_from_is_gene,
    normalize_cell_key,
    to_pandas,
)


def _frozen_iter_point_partitions(points_obj, cols: list[str]):
    work = points_obj[cols]
    if hasattr(work, "to_delayed"):
        for part in work.to_delayed():
            pdf = part.compute()
            if len(pdf) > 0:
                yield pdf
        return

    pdf = to_pandas(work)
    if len(pdf) > 0:
        yield pdf



def _frozen_point_partition_arrays(
    pdf: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None,
    background_col: str | None,
    status_col: str | None = None,
):
    """Return compact valid arrays for one transcript partition."""
    x_all = pdf[x_col].to_numpy(dtype=np.float32, copy=False)
    y_all = pdf[y_col].to_numpy(dtype=np.float32, copy=False)
    gene_all = pdf[gene_col].astype("string").to_numpy(dtype=object)
    good = np.isfinite(x_all) & np.isfinite(y_all) & pd.notna(gene_all)
    if not np.any(good):
        return None

    if background_col is not None and background_col in pdf.columns:
        background_all = pdf[background_col].to_numpy(dtype=bool, copy=False)
    elif assignment_col is not None and assignment_col in pdf.columns:
        background_all = ~assignment_mask(pdf[assignment_col]).to_numpy(dtype=bool, copy=False)
    else:
        background_all = np.zeros(len(pdf), dtype=bool)
    if status_col is not None:
        # Rendering's unassigned group includes excluded transcripts. Their
        # scientific background flag remains false in the original points.
        background_all = background_all | (pdf[status_col].astype("string").fillna("unknown") != "assigned").to_numpy(dtype=bool)

    assignment_all = None
    assigned_all = None
    if assignment_col is not None and assignment_col in pdf.columns:
        assignment_all = pdf[assignment_col].to_numpy(dtype=object)
        assigned_all = assignment_mask(pdf[assignment_col]).to_numpy(dtype=bool, copy=False)

    return (
        np.ascontiguousarray(x_all[good]),
        np.ascontiguousarray(y_all[good]),
        gene_all[good].astype(str),
        np.ascontiguousarray(background_all[good]),
        None if assignment_all is None else assignment_all[good],
        None if assigned_all is None else np.ascontiguousarray(assigned_all[good]),
    )


def frozen_build_gene_point_groups(
    points_obj,
    x_col: str,
    y_col: str,
    gene_col: str,
    assignment_col: str | None = None,
    background_col: str | None = None,
    gene_visuals: Mapping[str, GeneVisual] | None = None,
    reference: Mapping[str, Any] | None = None,
    reference_resolver: Callable[[Iterable[str]], Mapping[str, Any] | None] | None = None,
    max_points: int | None = None,
    random_state: int = 42,
    alpha: float = 1.0,
    build_cell_index: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    status_col: str | None = None,
    is_gene_col: str | None = None,
) -> GenePointStore:
    """Read all transcripts into a symbol-grouped, gene-sorted point store.

    ``background`` is taken from ``background_col`` (bool) if present, else from
    ``assignment_col`` (unassigned == not :func:`assignment_mask`), else all
    foreground. If the total exceeds ``max_points`` a uniform random subsample is
    taken (this preserves per-gene proportions in expectation).

    ``is_gene_col`` (Atera's ``is_gene``) makes the store's ``control_genes`` the
    features flagged not-a-gene, cross-checked against the name rule by
    :func:`control_genes_from_is_gene`.  Without it controls are name-only.
    """
    cols = [x_col, y_col, gene_col]
    if is_gene_col is not None and is_gene_col not in cols:
        cols.append(is_gene_col)
    is_gene_flags: dict[str, bool] | None = {} if is_gene_col is not None else None
    if background_col is not None:
        cols.append(background_col)
    if assignment_col is not None and assignment_col not in cols:
        cols.append(assignment_col)
    if status_col is not None and status_col not in cols:
        cols.append(status_col)

    # Pass 1 retains only small aggregate dictionaries.  In particular, it does
    # not keep a Python string object for every transcript and it lets us size
    # all output arrays exactly before the second streaming pass.
    source_gene_counts: dict[str, int] = {}
    source_status_counts: dict[str, int] | None = {} if status_col else None
    cell_counts: dict[str, int] = {}
    total_source = 0
    for pdf in _frozen_iter_point_partitions(points_obj, cols):
        _check_transcript_cancelled(cancel_check)
        if is_gene_flags is not None:
            # Raw rows, before coordinate filtering: the flag is per feature.
            accumulate_is_gene_flags(
                is_gene_flags, pdf[gene_col].astype("string").to_numpy(dtype=object), pdf[is_gene_col]
            )
        if status_col is not None:
            # Status totals describe the complete source, independent of the
            # render cap and coordinate filtering used for the point layers.
            for status, count in pdf[status_col].value_counts(dropna=False).items():
                key = str(status) if pd.notna(status) else "unknown"
                source_status_counts[key] = source_status_counts.get(key, 0) + int(count)
        arrays = _frozen_point_partition_arrays(
            pdf,
            x_col=x_col,
            y_col=y_col,
            gene_col=gene_col,
            assignment_col=assignment_col,
            background_col=background_col,
            status_col=status_col,
        )
        if arrays is None:
            continue
        _x, _y, genes_part, _bg, assignment_part, assigned_part = arrays
        total_source += int(len(genes_part))
        names, counts = np.unique(genes_part, return_counts=True)
        for name, count in zip(names.tolist(), counts.tolist(), strict=True):
            text = str(name)
            source_gene_counts[text] = source_gene_counts.get(text, 0) + int(count)

        if build_cell_index and assignment_part is not None and assigned_part is not None:
            assigned_values = assignment_part[assigned_part]
            keys = np.fromiter(
                (normalize_cell_key(value) for value in assigned_values),
                dtype=object,
                count=int(len(assigned_values)),
            )
            keys = keys[keys != ""]
            if len(keys):
                unique_keys, key_counts = np.unique(keys, return_counts=True)
                for key, count in zip(unique_keys.tolist(), key_counts.tolist(), strict=True):
                    text = str(key)
                    cell_counts[text] = cell_counts.get(text, 0) + int(count)

    if total_source == 0:
        return _empty_gene_point_store()

    source_genes = sorted(source_gene_counts)
    if reference is None and reference_resolver is not None:
        reference = reference_resolver(source_genes)
    gene_to_code = {name: index for index, name in enumerate(source_genes)}
    gene_code_dtype = _unsigned_code_dtype(len(source_genes))

    render_count = total_source
    if max_points is not None and 0 < int(max_points) < total_source:
        render_count = int(max_points)
    sampled = render_count < total_source

    render_x = np.empty(render_count, dtype=np.float32)
    render_y = np.empty(render_count, dtype=np.float32)
    render_gene_codes = np.empty(render_count, dtype=gene_code_dtype)
    render_background = np.empty(render_count, dtype=bool)

    cell_index = CellTranscriptIndex.empty() if build_cell_index else None
    cell_keys = sorted(cell_counts)
    cell_code_by_key = {key: index for index, key in enumerate(cell_keys)}
    cell_offsets = np.zeros(len(cell_keys) + 1, dtype=np.int64)
    if build_cell_index and cell_keys:
        cell_offsets[1:] = np.cumsum(
            np.asarray([cell_counts[key] for key in cell_keys], dtype=np.int64)
        )
        cell_coords = np.empty((int(cell_offsets[-1]), 2), dtype=np.float32)
        cell_gene_codes = np.empty(int(cell_offsets[-1]), dtype=gene_code_dtype)
        cell_write = cell_offsets[:-1].copy()
    else:
        cell_coords = np.empty((0, 2), dtype=np.float32)
        cell_gene_codes = np.empty((0,), dtype=gene_code_dtype)
        cell_write = np.empty((0,), dtype=np.int64)

    # Pass 2 fills fixed-size outputs.  Sequential hypergeometric allocation
    # selects an exact uniform sample without ever materialising the uncapped
    # coordinate set in memory.
    rng = np.random.default_rng(int(random_state))
    remaining_source = int(total_source)
    remaining_sample = int(render_count)
    render_cursor = 0
    for pdf in _frozen_iter_point_partitions(points_obj, cols):
        _check_transcript_cancelled(cancel_check)
        arrays = _frozen_point_partition_arrays(
            pdf,
            x_col=x_col,
            y_col=y_col,
            gene_col=gene_col,
            assignment_col=assignment_col,
            background_col=background_col,
            status_col=status_col,
        )
        if arrays is None:
            continue
        x_part, y_part, genes_part, bg_part, assignment_part, assigned_part = arrays
        part_count = int(len(genes_part))
        part_gene_codes = np.fromiter(
            (gene_to_code[str(name)] for name in genes_part),
            dtype=gene_code_dtype,
            count=part_count,
        )

        if remaining_sample == remaining_source:
            keep = np.arange(part_count, dtype=np.int64)
        elif remaining_sample <= 0:
            keep = np.empty((0,), dtype=np.int64)
        else:
            take = int(
                rng.hypergeometric(
                    ngood=part_count,
                    nbad=max(0, remaining_source - part_count),
                    nsample=remaining_sample,
                )
            )
            keep = (
                np.sort(rng.choice(part_count, size=take, replace=False))
                if take > 0
                else np.empty((0,), dtype=np.int64)
            )
        take = int(len(keep))
        if take:
            end = render_cursor + take
            render_x[render_cursor:end] = x_part[keep]
            render_y[render_cursor:end] = y_part[keep]
            render_gene_codes[render_cursor:end] = part_gene_codes[keep]
            render_background[render_cursor:end] = bg_part[keep]
            render_cursor = end
        remaining_source -= part_count
        remaining_sample -= take

        if build_cell_index and assignment_part is not None and assigned_part is not None and cell_keys:
            assigned_positions = np.flatnonzero(assigned_part)
            if len(assigned_positions):
                normalized = np.fromiter(
                    (normalize_cell_key(assignment_part[pos]) for pos in assigned_positions),
                    dtype=object,
                    count=int(len(assigned_positions)),
                )
                valid_keys = normalized != ""
                assigned_positions = assigned_positions[valid_keys]
                normalized = normalized[valid_keys]
                if len(assigned_positions):
                    codes = np.fromiter(
                        (cell_code_by_key[str(key)] for key in normalized),
                        dtype=np.int64,
                        count=int(len(normalized)),
                    )
                    order = np.argsort(codes, kind="stable")
                    codes_sorted = codes[order]
                    positions_sorted = assigned_positions[order]
                    starts = np.concatenate(
                        ([0], np.flatnonzero(np.diff(codes_sorted) != 0) + 1)
                    )
                    ends = np.concatenate((starts[1:], [len(codes_sorted)]))
                    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                        code = int(codes_sorted[start])
                        count = int(end - start)
                        dest = int(cell_write[code])
                        src = positions_sorted[start:end]
                        cell_coords[dest:dest + count, 0] = y_part[src]
                        cell_coords[dest:dest + count, 1] = x_part[src]
                        cell_gene_codes[dest:dest + count] = part_gene_codes[src]
                        cell_write[code] += count

    if render_cursor != render_count:
        raise RuntimeError(
            f"Transcript streaming sample filled {render_cursor} points; expected {render_count}"
        )

    if cell_keys:
        slices = {
            key: (int(cell_offsets[index]), int(cell_offsets[index + 1]))
            for index, key in enumerate(cell_keys)
        }
        cell_index = CellTranscriptIndex(
            coords_yx=cell_coords,
            gene_codes=cell_gene_codes,
            gene_names=tuple(source_genes),
            slices=slices,
        )

    control_genes = control_genes_from_is_gene(
        source_genes, is_gene_flags, source=f"column {is_gene_col!r}" if is_gene_col else ""
    )
    return assemble_gene_point_store(
        list(source_genes),
        render_x,
        render_y,
        render_gene_codes,
        render_background,
        gene_visuals=gene_visuals,
        reference=reference,
        alpha=alpha,
        control_genes=control_genes,
        sampled=sampled,
        source_gene_counts=source_gene_counts,
        source_total_points=total_source,
        source_status_counts=source_status_counts,
        cell_transcript_index=cell_index,
    )


