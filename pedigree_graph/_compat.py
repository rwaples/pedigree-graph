"""The 0.7.1 subsample constructor and the pair / count adapters, kept working until slice 7 deletes them.

# 0.8.0-DELETE: this whole module.  ``PedigreeGraph.from_subsample`` is the
pre-view way to restrict pair extraction to a subset of rows, which
``full.view(ids=...)`` replaces; ``PedigreeGraph.extract_pairs`` and
``count_pairs`` are the 0.7.1 pair surface that ``relationship_pairs`` /
``relationship_counts`` replace; ``count_pairs_streaming`` is the 0.7.1 scalar
estimate that ``estimate_relationship_counts`` replaces (ADR 0006).  The bodies
live here rather than in :mod:`pedigree_graph._core` so their removal is a file
deletion plus four thin methods.
"""

from __future__ import annotations

__all__ = ["from_subsample", "legacy_count_pairs", "legacy_count_pairs_streaming", "legacy_extract_pairs"]

from typing import TYPE_CHECKING, Literal

import numpy as np

from pedigree_graph._errors import PedigreeValidationError, ResourceError
from pedigree_graph._frames import _coerce_to_array_dict
from pedigree_graph._input import validate_id_field
from pedigree_graph._kinship_kernel import _compute_theta_per_gen, _scatter_summary
from pedigree_graph._lineage import descendant_path_counts, distinct_ancestor_counts
from pedigree_graph._ne_common import _require_complete_generation_labels
from pedigree_graph._ne_rates import _generation_kinship_summary, _per_gen_mean_kinship
from pedigree_graph._pair_extractor import MatrixPairExtractor
from pedigree_graph._pair_utils import project_pairs
from pedigree_graph._registry import RELATIONSHIPS, _validate_max_degree
from pedigree_graph._streaming_counter import _estimate

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


def legacy_count_pairs(
    pg: PedigreeGraph,
    max_degree: int,
    scope: Literal["subsample", "full"],
) -> dict[str, int]:
    """Count every relationship category with the 0.7.1 matrix engine.

    The body of :meth:`PedigreeGraph.count_pairs`: returns the counts cached
    by an earlier :meth:`PedigreeGraph.extract_pairs` for this cutoff, else
    runs it.  Overlapping categories are counted in each (no precedence fold).

    Args:
        pg: The graph to count on.
        max_degree: Degree cutoff, validated against ``[0, 5]``.
        scope: ``"subsample"`` counts the pairs ``extract_pairs`` returns
            (mask-filtered on a ``from_subsample`` graph); ``"full"`` counts
            the underlying graph.  Equivalent on a plain graph.

    Returns:
        ``{code: count}`` over all 23 codes, ``0`` above the cutoff.

    Raises:
        ValueError: *scope* is neither literal.
        PedigreeValidationError: ``max_degree_out_of_range``.
    """
    if scope not in ("subsample", "full"):
        raise ValueError(f"scope must be 'subsample' or 'full', got {scope!r}")
    max_degree = _validate_max_degree(max_degree)

    key = ("matrix", int(max_degree), 0.0)
    entry = pg._pair_count_cache.get(key)
    if entry is None:
        pg.extract_pairs(max_degree=max_degree)
        entry = pg._pair_count_cache[key]
    raw, sub = entry
    return dict(raw) if scope == "full" else dict(sub)


def legacy_count_pairs_streaming(
    pg: PedigreeGraph,
    max_degree: int,
    scope: Literal["subsample", "full"],
) -> dict[str, int]:
    """Return the 0.7.1 scalar-estimate dict.

    The body of :meth:`PedigreeGraph.count_pairs_streaming`: the scope
    validation, the refusal of ``scope="subsample"`` on a ``from_subsample``
    graph (the scalar path is full-graph only), and the 0.8 estimate's *raw*
    counts (unfolded, ``0`` above the cutoff, as 0.7.1 returned them).  The
    estimate's ``RuntimeWarning`` for clamped codes is not suppressed, and its
    per-cutoff cache is shared, so an ``estimate_relationship_counts`` call
    after this one for the same cutoff is a silent cache hit.  Like every
    0.7.1 adapter it does not commit the package thread budget.

    Args:
        pg: The graph to count on.
        max_degree: Degree cutoff, validated against ``[0, 5]``.
        scope: ``"full"`` or ``"subsample"``; equivalent on a plain graph.

    Returns:
        ``{code: count}`` over all 23 codes, ``0`` above the cutoff.

    Raises:
        ValueError: *scope* is neither literal.
        PedigreeValidationError: ``max_degree_out_of_range``.
        NotImplementedError: ``scope="subsample"`` on a ``from_subsample`` graph.
    """
    if scope not in ("subsample", "full"):
        raise ValueError(f"scope must be 'subsample' or 'full', got {scope!r}")
    max_degree = _validate_max_degree(max_degree)
    if scope == "subsample" and pg._legacy_view is not None:
        raise NotImplementedError(
            "count_pairs_streaming(scope='subsample') is not supported on "
            "graphs constructed via from_subsample; the scalar path is "
            "full-graph only.  Use count_pairs() for subsample-restricted "
            "counts, or call count_pairs_streaming(scope='full') for the "
            "underlying full-pedigree counts.",
        )

    return dict(_estimate(pg, max_degree).raw)


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


def legacy_per_gen_mean_kinship(pg: PedigreeGraph, min_kinship: float) -> np.ndarray:
    """Return the 0.7.1 ``max(label) + 1`` mean-kinship array.

    The body of :meth:`PedigreeGraph.per_gen_mean_kinship`.  Partial labels
    are rejected, since the estimators that consume this array cannot
    represent an excluded cohort.  At ``min_kinship == 0.0`` the array is the
    0.8 summary scattered over the raw labels, NaN in the gaps; a positive
    threshold re-runs the pruned DP as before; a matrix cached at the same
    threshold is walked instead.  Cached per threshold on the graph.
    """
    key = float(min_kinship)
    cached = pg._theta_per_gen_cache.get(key)
    if cached is not None:
        return cached
    _require_complete_generation_labels(pg, "per_gen_mean_kinship")

    labels = np.asarray(pg.generation)
    K = pg._kinship_cache.get(key)
    if K is not None:
        theta = _per_gen_mean_kinship(K, labels, np.asarray(pg.twin))
    elif key == 0.0:
        theta = _scatter_summary(_generation_kinship_summary(pg), labels)
    else:
        theta = _compute_theta_per_gen(
            pg.n,
            pg.mother,
            pg.father,
            pg.twin,
            pg.depth,
            min_kinship,
            labels=labels,
        )
    pg._theta_per_gen_cache[key] = theta
    return theta


def legacy_n_ancestors(pg: PedigreeGraph) -> np.ndarray:
    """Return the 0.7.1 ``compute_n_ancestors`` array.

    The same distinct counts as :meth:`PedigreeGraph.distinct_ancestor_counts`,
    as a writeable int32 array cached on the graph, so repeated calls return
    the same object as 0.7.1 did.
    """
    if pg._n_ancestors is None:
        pg._n_ancestors = distinct_ancestor_counts(pg).copy()
    return pg._n_ancestors


def legacy_n_descendants(pg: PedigreeGraph) -> np.ndarray:
    """Return the 0.7.1 ``compute_n_descendants`` array.

    The same path counts as :meth:`PedigreeGraph.descendant_path_counts`,
    cast to int32 after the 0.7.1 bounds check, so a loop-heavy pedigree
    raises ``arithmetic_overflow`` rather than wrapping.  Cached on the graph
    as a writeable array.

    Raises:
        ResourceError: ``arithmetic_overflow`` when any count exceeds
            ``np.iinfo(np.int32).max``.
    """
    if pg._n_descendants is None:
        counts = descendant_path_counts(pg)
        if counts.size and int(counts.max()) > np.iinfo(np.int32).max:
            raise ResourceError(
                "arithmetic_overflow",
                "compute_n_descendants: at least one path count exceeds "
                f"int32 max ({np.iinfo(np.int32).max:,}); the pedigree is "
                "too inbred / loop-heavy for the int32-cached output.  "
                "Use PedigreeGraph.descendant_path_counts() for the int64 counts.",
                operation="compute_n_descendants",
                dtype="int32",
            )
        pg._n_descendants = counts.astype(np.int32)
    return pg._n_descendants
