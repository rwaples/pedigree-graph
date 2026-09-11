"""Rate-of-inbreeding / coancestry Ne estimators (PGQ-006).

The regression-based estimators built on F̄ and θ̄ series, plus the
Gutiérrez 2008 individual-ΔF estimator:

* :func:`ne_inbreeding`         — regression of ``ln(1 − F̄_t)`` on t.
* :func:`ne_coancestry`         — regression of ``ln(1 − θ̄_t)`` on t.
* :func:`ne_individual_delta_f` — Gutiérrez individual increase in inbreeding.

Each public estimator resolves its prerequisites from the graph and hands
them to a private evaluator (``_inbreeding_from`` and friends) that works
on an :class:`~pedigree_graph._cohorts.ObservedCohorts` grouping.  The
orchestrator calls the same evaluators, so a direct call and an
orchestrated call cannot disagree.

Also owns the two routes to :meth:`PedigreeGraph.mean_kinship_by_generation`
— :func:`_summary_from_matrix` over a cached kinship matrix and the streamed
DP — and :func:`_kinship_summary_for_labels`, the one place that chooses
between them for any labelling, the genome-node one included.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._errors import PedigreeValidationError
from pedigree_graph._input import _INT32_MAX, _check_duplicate_rows, _coerce_row_selection, _FieldSpec, _own
from pedigree_graph._kinship_kernel import (
    _compute_eqg,
    _compute_generation_kinship_summary,
    _densify_labels,
    _finalize_summary,
)
from pedigree_graph._ne_common import (
    _scalar_ne_from_log_regression,
    _transition_ne,
)
from pedigree_graph._ne_results import (
    NeCoancestryResult,
    NeInbreedingResult,
    NeIndividualDeltaFResult,
)

if TYPE_CHECKING:
    import scipy.sparse as sp

    from pedigree_graph._core import PedigreeGraph
    from pedigree_graph.summaries import GenerationKinshipSummary


logger = logging.getLogger(__name__)


def _summary_from_matrix(
    K: sp.csc_matrix,
    labels: np.ndarray,
    twin_idx: np.ndarray,
) -> GenerationKinshipSummary:
    """Generation kinship summary walked from a complete kinship matrix.

    Same grouping and MZ rule as the streamed DP path
    (:func:`~pedigree_graph._kinship_dp._compute_generation_kinship_summary`),
    so the two are interchangeable oracles: upper-triangle entries whose rows
    share an observed label, minus ``(i, twin[i])`` pairs, summed per label
    and divided by :func:`~pedigree_graph._kinship_dp._finalize_summary`.

    Args:
        K: full-symmetric sparse kinship (φ-scale) from
            :meth:`PedigreeGraph.kinship_matrix`.
        labels: per-row cohort label, ``-1`` unknown.
        twin_idx: per-row twin partner row index, ``-1`` for non-twins.
    """
    dense, observed, n_unlabelled = _densify_labels(labels)
    twin = np.ascontiguousarray(twin_idx, dtype=np.int32)
    k = int(observed.shape[0])
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    pair_mask = (rows < cols) & (dense[rows] == dense[cols]) & (dense[rows] < k)
    pair_mask &= ~((twin[rows] >= 0) & (twin[rows] == cols))
    sum_theta = np.bincount(
        dense[rows[pair_mask]].astype(np.intp),
        weights=vals[pair_mask].astype(np.float64),
        minlength=k + 1,
    )
    return _finalize_summary(sum_theta, dense, twin, observed, n_unlabelled)


def _kinship_summary_for_labels(pg: PedigreeGraph, labels: np.ndarray) -> GenerationKinshipSummary:
    """Generation kinship summary of *labels*, by whichever route the graph affords.

    Walks the complete kinship matrix when the graph already caches it, else
    streams the retiring DP.  :func:`_summary_from_matrix` is written to be
    the matrix oracle of the DP path, so the route is an implementation
    choice and never a semantic one; both the memoised graph-label summary
    and the masked-label summary the group-coancestry prerequisite needs go
    through here rather than each picking a route of its own.

    Args:
        pg: Pedigree graph supplying the structure.
        labels: per-row cohort label to group by; ``-1`` for a row that
            belongs to no cohort.

    Returns:
        The :class:`~pedigree_graph.summaries.GenerationKinshipSummary` over
        those labels.
    """
    K = pg._complete_kinship_cache
    if K is not None:
        return _summary_from_matrix(K, np.asarray(labels), np.asarray(pg.twin_rows))
    return _compute_generation_kinship_summary(
        pg.n_individuals,
        pg.mother_rows,
        pg.father_rows,
        pg.twin_rows,
        pg.depth,
        0.0,
        labels=labels,
    )


def _generation_kinship_summary(pg: PedigreeGraph) -> GenerationKinshipSummary:
    """Memoised body of :meth:`PedigreeGraph.mean_kinship_by_generation`.

    Groups by the supplied labels, or by structural depth when none were
    supplied, over whichever route :func:`_kinship_summary_for_labels`
    picks.  Stored on the graph, so every later call returns the same
    frozen object.
    """
    cached = pg._generation_kinship_summary
    if cached is not None:
        return cached
    labels = pg.generation_labels
    if labels is None:
        labels = pg.depth
    t0 = time.perf_counter()
    summary = _kinship_summary_for_labels(pg, labels)
    pg._generation_kinship_summary = summary
    logger.info(
        "mean_kinship_by_generation: n=%d, groups=%d, unlabelled=%d, %.2fs",
        pg.n_individuals,
        len(summary),
        summary.unlabelled_individual_count,
        time.perf_counter() - t0,
    )
    return summary


def _cohort_means(values: np.ndarray, cohorts: ObservedCohorts) -> np.ndarray:
    """Mean of ``values`` within each observed cohort, NaN for an empty one."""
    out = np.full(cohorts.k, np.nan, dtype=np.float64)
    for b, rows in enumerate(cohorts.members()):
        if rows.shape[0]:
            out[b] = float(values[rows].mean())
    return out


def _inbreeding_from(cohorts: ObservedCohorts, F: np.ndarray) -> NeInbreedingResult:
    mean_f = _cohort_means(F, cohorts)
    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_f, cohorts.generations)
    return NeInbreedingResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_f_per_gen=mean_f,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(mean_f, cohorts.generations),
        slope=slope,
        n_generations_used=n_used,
    )


def ne_inbreeding(pg: PedigreeGraph) -> NeInbreedingResult:
    """Inbreeding-rate Ne (Ne_I).

    Computes per-cohort mean F over the observed generation labels.
    Each adjacent observed-cohort transition reports the gap-corrected
    ``Ne = 1 / (2·ΔF)`` of :func:`~pedigree_graph._ne_common._transition_ne`;
    the aggregate Ne comes from the regression slope of ``ln(1 − F̄)`` on the
    label offset, first observed cohort excluded.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_inbreeding")
    return _inbreeding_from(cohorts, pg._inbreeding_values())


def _coancestry_from(cohorts: ObservedCohorts, summary: GenerationKinshipSummary) -> NeCoancestryResult:
    if not np.array_equal(summary.generations, cohorts.generations):
        raise ValueError("generation kinship summary does not describe the estimator's observed cohorts")
    mean_theta = np.asarray(summary.mean_kinship, dtype=np.float64)
    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(mean_theta, cohorts.generations)
    return NeCoancestryResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_theta_per_gen=mean_theta,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(mean_theta, cohorts.generations),
        slope=slope,
        n_generations_used=n_used,
    )


def ne_coancestry(pg: PedigreeGraph) -> NeCoancestryResult:
    """Coancestry-rate Ne (Ne_C).

    Same regression form as Ne_I but on the per-cohort mean kinship θ over
    within-cohort unordered pairs (excluding the diagonal and MZ twin
    pairs) that :meth:`PedigreeGraph.mean_kinship_by_generation` reports.
    The summary is streamed from the DP without materializing K, or walked
    from a complete kinship matrix the graph already caches.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_coancestry")
    return _coancestry_from(cohorts, _generation_kinship_summary(pg))


def _ne_from_delta_f(delta_f: np.ndarray) -> float | None:
    """``1/(2·ΔF̄)`` over *delta_f*; ``None`` when it is empty or ΔF̄ is not positive.

    The one reduction behind every Ne this estimator reports — the scalar, each
    cohort's, and the unrelated-founder diagnostic — so the "no rate, no
    estimate" rule is stated once rather than guarded at each site.
    """
    if delta_f.shape[0] == 0:
        return None
    mean = float(delta_f.mean())
    return 1.0 / (2.0 * mean) if mean > 0.0 else None


def _individual_delta_f_from(
    cohorts: ObservedCohorts,
    F: np.ndarray,
    eqg: np.ndarray,
    reference: np.ndarray | None = None,
) -> NeIndividualDeltaFResult:
    valid = (eqg > 0.0) & (F < 1.0)
    delta_f = np.full(F.shape[0], np.nan, dtype=np.float64)
    if valid.any():
        delta_f[valid] = 1.0 - np.power(1.0 - F[valid], 1.0 / eqg[valid])

    k = cohorts.k
    members = cohorts.members()
    ne_per_gen = np.full(k, np.nan, dtype=np.float64)
    mean_eqg_per_gen = np.full(k, np.nan, dtype=np.float64)
    n_used_per_gen = np.zeros(k, dtype=np.int64)
    for b, rows in enumerate(members):
        in_b = rows[valid[rows]]
        n_used_per_gen[b] = int(in_b.shape[0])
        if in_b.shape[0] == 0:
            continue
        mean_eqg_per_gen[b] = float(eqg[in_b].mean())
        cohort_ne = _ne_from_delta_f(delta_f[in_b])
        if cohort_ne is not None:
            ne_per_gen[b] = cohort_ne

    if reference is None:
        reference = members[-1] if k else np.zeros(0, dtype=np.int32)
    eligible = reference[valid[reference]]
    n_reference = int(eligible.shape[0])
    reference_df = delta_f[eligible]
    ne = _ne_from_delta_f(reference_df)
    standard_error: float | None = None
    if ne is not None and n_reference > 1:
        standard_error = 2.0 / math.sqrt(n_reference) * ne**2 * float(reference_df.std(ddof=1))

    lagged = eligible[eqg[eligible] > 1.0]
    lagged_df = 1.0 - np.power(1.0 - F[lagged], 1.0 / (eqg[lagged] - 1.0))
    ne_unrelated_founders = _ne_from_delta_f(lagged_df)

    buckets = np.unique(cohorts.dense[eligible])
    shared = buckets.shape[0] == 1 and int(buckets[0]) < k
    return NeIndividualDeltaFResult(
        ne=ne,
        generations=cohorts.generations,
        ne_per_gen=ne_per_gen,
        mean_eqg_per_gen=mean_eqg_per_gen,
        n_used_per_gen=n_used_per_gen,
        standard_error=standard_error,
        n_reference=n_reference,
        reference_generation=int(cohorts.generations[buckets[0]]) if shared else None,
        ne_unrelated_founders=ne_unrelated_founders,
    )


_REFERENCE = _FieldSpec("reference", True, 0, _INT32_MAX, np.int32)


def _reference_out_of_range(value: object, position: int, n_individuals: int) -> PedigreeValidationError:
    return PedigreeValidationError(
        "reference_row_out_of_range",
        f"row {value} at position {position} is outside the {n_individuals}-row pedigree",
        row=value,
        position=position,
        n_individuals=n_individuals,
    )


def _reference_rows(pg: PedigreeGraph, selection: object) -> np.ndarray:
    """Validate a reference-subpopulation selection against the graph's row range.

    The same shape, integer-form, range, then duplicate order a view selection
    follows, so a caller sees a single-entry failure before a whole-argument
    one.  Duplicates are rejected rather than collapsed: a repeated row would
    silently weight one individual twice in ΔF̄.

    Args:
        pg: The graph whose rows are being selected.
        selection: The caller's array-like of graph rows.

    Returns:
        The rows as an owned read-only int32 array, in the order given.

    Raises:
        PedigreeValidationError: ``invalid_shape`` or ``invalid_integer_value``
            for a malformed selection, ``reference_row_out_of_range`` for a row
            outside the pedigree, then ``duplicate_reference_row``.
    """
    n_individuals = pg.n_individuals
    rows = _coerce_row_selection(
        _REFERENCE,
        selection,
        n_individuals,
        lambda value, position: _reference_out_of_range(value, position, n_individuals),
    )
    _check_duplicate_rows(rows, n_individuals, "duplicate_reference_row", "row", rows)
    return _own(rows, np.int32)


def ne_individual_delta_f(pg: PedigreeGraph, *, reference: object | None = None) -> NeIndividualDeltaFResult:
    """Gutiérrez et al. 2008 individual increase in inbreeding (Ne_iΔF).

    Gutiérrez, Cervantes, Molina, Valera and Goyache, *Individual increase in
    inbreeding allows estimating effective sizes from pedigrees*, Genet. Sel.
    Evol. 40(4):359-378, eq. 2::

        ΔF_i = 1 − (1 − F_i)^(1/t_i)

    where ``t_i`` is the individual's equivalent complete generations, the sum
    over its known ancestors of ``(1/2)^n`` for meiotic distance ``n``
    (:func:`~pedigree_graph._kinship_depth._compute_eqg`).  A row is eligible
    when ``t_i > 0``: a founder has ``t = 0`` and no rate.  Rows with
    ``F_i = 1`` are dropped as well, which is this package's guard and not the
    paper's — eq. 2 is finite there and would report ``ΔF_i = 1``.

    Their §2.1 averages ΔF_i over a **reference subpopulation** to give ΔF̄ and
    reports ``Ne = 1/(2·ΔF̄)``, with standard error
    ``σ_Ne = (2/√N)·Ne²·σ_ΔF`` over that subpopulation's ``N`` individuals.
    σ_ΔF is the sample standard deviation (``ddof=1``): the paper does not
    state a ddof, and the difference is O(1/N).

    Equivalent complete generations already count each individual's own
    pedigree depth, so labels only group the per-cohort series; each cohort is
    itself a valid reference subpopulation, which is what ``ne_per_gen``
    reports.

    The result also carries ``ne_unrelated_founders``, a package-defined
    diagnostic that no line of the paper contains, dividing by ``t_i − 1``
    over the reference rows with ``t_i > 1``.  Its justification is
    measurement, not theory.  Eq. 1 is ``F_t = 1 − (1 − ΔF)^t``, so eq. 2
    recovers the census size only where the pedigree has accumulated ``t``
    generations of drift, and founders that really are unrelated and
    non-inbred leave generation 1 at ``F = 0`` exactly, one generation behind.
    The paper's own remedy is choosing the reference subpopulation by pedigree
    depth (its Table II) and reading ΔF_i against equivalent generations (its
    Figs. 3-4), which is what ``reference=`` does here.  A real pedigree's
    founders are merely where record-keeping stopped and are generally
    related, so there the lag does not exist and the field biases downward.
    ``ne`` remains the estimator; see
    :class:`~pedigree_graph.effective_size.NeIndividualDeltaFResult`.

    Args:
        pg: Pedigree graph.
        reference: Graph rows of the reference subpopulation, unique and in
            range.  ``None`` selects the rows of the last observed cohort.

    Returns:
        The :class:`~pedigree_graph.effective_size.NeIndividualDeltaFResult`.

    Raises:
        PedigreeValidationError: ``invalid_shape``, ``invalid_integer_value``,
            ``reference_row_out_of_range``, or ``duplicate_reference_row`` for
            a *reference* the graph cannot accept.
        MissingMetadataError: ``missing_generation_labels`` when the supplied
            labels are partly ``-1``.
    """
    rows = None if reference is None else _reference_rows(pg, reference)
    cohorts = ObservedCohorts.for_graph(pg, "ne_individual_delta_f")
    F = pg._inbreeding_values()
    eqg = _compute_eqg(np.asarray(pg.mother_rows), np.asarray(pg.father_rows), np.asarray(pg.depth), pg.n_individuals)
    return _individual_delta_f_from(cohorts, F, eqg, rows)
