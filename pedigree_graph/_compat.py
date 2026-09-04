"""The 0.7.1 subsample constructor, kept working until slice 7 deletes it.

# 0.8.0-DELETE: this whole module.  ``PedigreeGraph.from_subsample`` is the
pre-view way to restrict pair extraction to a subset of rows, which
``full.view(ids=...)`` replaces (ADR 0006).  The body lives here rather than in
:mod:`pedigree_graph._core` so its removal is a file deletion plus one thin
classmethod.
"""

from __future__ import annotations

__all__ = ["from_subsample"]

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._frames import _coerce_to_array_dict
from pedigree_graph._input import _map_ids_to_rows, validate_id_field

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._frames import FrameLike


def from_subsample(
    cls: type[PedigreeGraph],
    full_pedigree: dict[str, np.ndarray] | FrameLike,
    df: dict[str, np.ndarray] | FrameLike,
) -> PedigreeGraph:
    """Build the full graph, then mark which of its rows *df* selects.

    The full pedigree is what gets built, so multi-hop relationships through
    ancestors absent from *df* are still found; the sample mask and the remap
    then restrict and re-coordinate what ``extract_pairs`` returns.

    Args:
        cls: The graph class to construct.
        full_pedigree: Complete pedigree as a dict of columns or a frame.
        df: Subsample of *full_pedigree* in the same form.  Ids must be unique
            and present in *full_pedigree*.  Empty is allowed.

    Returns:
        A graph over *full_pedigree* whose pair output is restricted to *df*.

    Raises:
        PedigreeValidationError: ``missing_field`` when *df* has no id column,
            ``duplicate_id`` for a repeated id, ``unknown_view_id`` for an id
            absent from *full_pedigree*.
    """
    df_arrays = _coerce_to_array_dict(df)
    if "id" not in df_arrays:
        raise PedigreeValidationError(
            "missing_field",
            "from_subsample: df is missing the required 'id' field",
            field="id",
        )
    # Same id-column contract as the constructor (unique, nonnegative).
    df_ids = validate_id_field(df_arrays["id"])

    full_arrays = _coerce_to_array_dict(full_pedigree)
    full_ids = np.asarray(full_arrays["id"])
    if len(df_ids) > 0:
        in_full = np.isin(df_ids, full_ids)
        if not in_full.all():
            positions = np.flatnonzero(~in_full)
            first = int(positions[0])
            raise PedigreeValidationError(
                "unknown_view_id",
                f"from_subsample: {positions.size} id(s) in df are not present in full_pedigree",
                id=int(df_ids[first]),
                position=first,
                missing_count=int(positions.size),
            )

    # Constructing the full graph validates the full pedigree's id column.
    pg = cls(full_arrays)
    full_ids = pg._ids  # validated int64 copy

    if len(df_ids) == 0:
        # Empty subsample → mask filters everything; remap unused.
        pg._sample_mask = np.zeros(len(full_ids), dtype=bool)
        pg._subsample_remap = np.full(len(full_ids), -1, dtype=np.intp)
        return pg

    pg._sample_mask = np.isin(full_ids, df_ids)

    # Build full-graph-row → df-row table via the same searchsorted remap
    # used for parent ids: target = df ids, query = full ids.
    pg._subsample_remap = _map_ids_to_rows(df_ids, full_ids, np.intp)

    # Build the inverse df-row → full-graph-row table so consumers that
    # need graph coordinates (e.g. compute_pair_kinship indexing the
    # full kinship matrix) can map caller-coordinate pairs back.
    graph_rows_in_sub = np.flatnonzero(pg._subsample_remap >= 0)
    inverse = np.full(len(df_ids), -1, dtype=np.intp)
    inverse[pg._subsample_remap[graph_rows_in_sub]] = graph_rows_in_sub
    pg._subsample_inverse = inverse

    return pg
