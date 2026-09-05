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

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._frames import _coerce_to_array_dict
from pedigree_graph._input import validate_id_field
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import project_pairs
from pedigree_graph._registry import RELATIONSHIPS, _validate_max_degree

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph._frames import FrameLike

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
    ``(min, max)`` orientation for the collateral codes, projection through
    the legacy view of a ``from_subsample`` graph, the count-cache write, and
    matrix release.  Within a code the emission order may differ from 0.7.1
    (blocks arrive sorted by canonical key); membership and orientation do not.

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
    if pg._legacy_view is not None:
        # 0.7.1 re-canonicalised every code after the remap, lineal ones
        # included; the frozen parity arrays encode that.
        graph_to_view = pg._legacy_view._graph_to_view()
        for code, (first, second) in pairs.items():
            view_first, view_second = project_pairs(first, second, graph_to_view)
            pairs[code] = (np.minimum(view_first, view_second), np.maximum(view_first, view_second))
    subsample_counts = {code: len(block[0]) for code, block in pairs.items()}

    pg._pair_count_cache[("matrix", int(max_degree), float(min_kinship))] = (raw_counts, subsample_counts)
    pg._release_pair_matrices()
    return pairs


def from_subsample(
    cls: type[PedigreeGraph],
    full_pedigree: dict[str, np.ndarray] | FrameLike,
    df: dict[str, np.ndarray] | FrameLike,
) -> PedigreeGraph:
    """Build the full graph and attach the legacy view that *df* selects.

    The full pedigree is what gets built, so multi-hop relationships through
    ancestors absent from *df* are still found; ``extract_pairs`` then
    projects its output through the view, in *df* row order.

    Args:
        cls: The graph class to construct.
        full_pedigree: Complete pedigree as a dict of columns or a frame.
        df: Subsample of *full_pedigree* in the same form.  Ids must be unique
            and present in *full_pedigree*.  Empty is allowed.

    Returns:
        A graph over *full_pedigree* whose pair output is restricted to *df*.

    Raises:
        PedigreeValidationError: ``missing_field`` when *df* has no id column,
            ``duplicate_id`` for a repeated id, the constructor's own codes
            for an invalid *full_pedigree*, then ``unknown_view_id`` for an
            id absent from it.
    """
    df_arrays = _coerce_to_array_dict(df)
    if "id" not in df_arrays:
        raise PedigreeValidationError(
            "missing_field",
            "from_subsample: df is missing the required 'id' field",
            field="id",
        )
    # The 0.7.1 tests pin duplicate_id here, not the view's duplicate_view_id.
    df_ids = validate_id_field(df_arrays["id"])
    pg = cls(_coerce_to_array_dict(full_pedigree))
    pg._legacy_view = pg.view(ids=df_ids)
    return pg
