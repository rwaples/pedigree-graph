"""The 0.7.1 subsample constructor and pair adapter, kept working until slice 7 deletes them.

# 0.8.0-DELETE: this whole module.  ``PedigreeGraph.from_subsample`` is the
pre-view way to restrict pair extraction to a subset of rows, which
``full.view(ids=...)`` replaces, and ``PedigreeGraph.extract_pairs`` is the
0.7.1 pair surface that ``relationship_pairs`` replaces (ADR 0006).  The bodies
live here rather than in :mod:`pedigree_graph._core` so their removal is a file
deletion plus two thin methods.
"""

from __future__ import annotations

__all__ = ["from_subsample", "legacy_extract_pairs"]

import logging
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._frames import _coerce_to_array_dict
from pedigree_graph._input import _map_ids_to_rows, validate_id_field
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import remap_pairs_to_caller
from pedigree_graph._registry import RELATIONSHIPS, _validate_max_degree

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._frames import FrameLike

logger = logging.getLogger(__name__)

# 0.7.1 stored these ten collateral codes as (min row, max row); the engine
# now orients them by role, so the adapter folds them back.
_LEGACY_CANONICAL_CODES = frozenset(
    code for code, category in RELATIONSHIPS.items() if category.down > 0 and not category.symmetric
)


def legacy_extract_pairs(
    pg: PedigreeGraph,
    max_degree: int,
    min_kinship: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run the matrix engine with 0.7.1 output semantics.

    The body of :meth:`PedigreeGraph.extract_pairs`: the 0.7.1 code selection,
    ``(min, max)`` orientation for the collateral codes, sample-mask filtering,
    caller-space remap, the count-cache write, and matrix release.  Within a
    code the emission order may differ from 0.7.1 (blocks arrive sorted by
    canonical key); membership and orientation do not.

    Args:
        pg: The graph to extract from.
        max_degree: Degree cutoff, validated against ``[0, 5]``.
        min_kinship: Codes with nominal kinship below this are skipped.

    Returns:
        ``{code: (idx1, idx2)}`` intp arrays in caller-input coordinates.

    Raises:
        PedigreeValidationError: ``max_degree_out_of_range``.
    """
    max_degree = _validate_max_degree(max_degree)
    codes = frozenset(
        code
        for code, category in RELATIONSHIPS.items()
        if category.degree <= max_degree and category.nominal_kinship >= min_kinship
    )
    pairs = MatrixPairExtractor(pg, max_workers=None).extract(codes)
    for code in _LEGACY_CANONICAL_CODES:
        first, second = pairs[code]
        pairs[code] = (np.minimum(first, second), np.maximum(first, second))

    raw_counts = {code: len(block[0]) for code, block in pairs.items()}
    if pg._sample_mask is not None:
        for code, (idx1, idx2) in pairs.items():
            keep = pg._sample_mask[idx1] & pg._sample_mask[idx2]
            pairs[code] = (idx1[keep], idx2[keep])
        logger.info("Filtered to sample_mask: %s", ", ".join(f"{k}: {len(v[0])}" for k, v in pairs.items()))
    if pg._subsample_remap is not None:
        remap_pairs_to_caller(pairs, pg._subsample_remap)
    subsample_counts = {code: len(block[0]) for code, block in pairs.items()}

    pg._pair_count_cache[("matrix", int(max_degree), float(min_kinship))] = (raw_counts, subsample_counts)
    pg._release_pair_matrices()
    return pairs


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
