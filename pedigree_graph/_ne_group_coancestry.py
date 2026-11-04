"""Group-coancestry Ne estimator and its per-cohort prerequisite (PGQ-006).

Caballero & Toro 2000 (Genet. Res. 75:331) eq. 3 group coancestry, evaluated
per observed cohort over the genome-node pedigree of ADR 0008 and reduced to
a scalar by the ``ln(1 − x)`` regression the other rate estimators share.

The genome-node collapse is a row mask: a masked co-twin's ``-1`` label
sends it to the sentinel bucket
:func:`~pedigree_graph._cohorts._densify_labels` already discards.  So the
prerequisite reads per-cohort θ sums off the generation kinship summary
rather than materialising a kinship matrix, and it reuses outright the
summary the graph memoises for :func:`~pedigree_graph.effective_size.ne_coancestry`:
since issue #25 that summary is itself genome-node, so both estimators share
one convention and one computation whether or not the pedigree has MZ twins.
That computation holds no kinship rows (issue #38), so memory stays linear in
the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._ne_common import (
    _genome_node_labels,
    _log_fit_mask,
    _scalar_ne_from_log_regression,
    _transition_ne,
)
from pedigree_graph._ne_rates import _generation_kinship_summary
from pedigree_graph._ne_results import NeGroupCoancestryResult

if TYPE_CHECKING:
    from pedigree_graph._cohorts import ObservedCohorts
    from pedigree_graph._core import PedigreeGraph


class GroupCoancestryByCohort(NamedTuple):
    """Per-cohort group coancestry over the genome-node representatives.

    Built by :func:`_group_coancestry_by_cohort` and consumed by
    :func:`_group_coancestry_from`.

    Attributes:
        generations: observed labels, int32, ascending, aligned with the
            estimator's :class:`~pedigree_graph._cohorts.ObservedCohorts`.
        mean_group_coancestry: float64 ``f̄_g`` per cohort; NaN where the
            cohort holds no genome-node representative.
        n_genomes: int64 genome-node representatives per cohort — the ``n_g``
            of the ``f̄_g`` denominator.
    """

    generations: np.ndarray
    mean_group_coancestry: np.ndarray
    n_genomes: np.ndarray


def _group_coancestry_by_cohort(pg: PedigreeGraph, cohorts: ObservedCohorts) -> GroupCoancestryByCohort:
    """Per-cohort genome-node group coancestry, Caballero & Toro 2000 eq. 3.

    Eq. 3 is ``f̄ = Σ_i Σ_j a_ij / 2N²`` over all ``N²`` ordered pairs of a
    group, self-coancestries and reciprocals included.  With ``a_ij = 2φ_ij``
    off the diagonal and ``a_ii = 1 + F_i`` on it, that is::

        f̄_g = (2·Σ_{i<j} φ_ij + Σ_i s_i) / n_g²,   s_i = (1 + F_i) / 2

    over cohort ``g``'s genome-node representatives.  The pair sum is read
    back off a kinship summary of :func:`_genome_node_labels` as
    ``mean_kinship · pair_counts``; the diagonal term needs only F.

    A ``-1`` in those labels is a collapsed co-twin and nothing else,
    because the estimator's cohorts reject partly-unknown labels before this
    runs.  The masked summary is therefore exactly the one
    :func:`~pedigree_graph._ne_rates._generation_kinship_summary` memoises on
    the graph, so the pair sum costs no pass of its own.  It also leaves the
    base labels free of ``-1``, so ``cohorts.dense`` is a cohort index for
    every row and never the sentinel bucket, and the kept rows bincount
    straight into cohort space.

    Args:
        pg: Pedigree graph.
        cohorts: The estimator's observed-cohort grouping.

    Returns:
        A :class:`GroupCoancestryByCohort` aligned with ``cohorts.generations``.
    """
    labels = _genome_node_labels(pg)
    summary = _generation_kinship_summary(pg)

    # Co-twins with differing generation labels are constructible: the MZ codes
    # in _errors.py's VALIDATION_CODES cover reciprocity, parents and sex, and
    # there is no label one. Masking can therefore empty a cohort out of the
    # summary entirely, so the summary's cohorts are a subset of the
    # estimator's and its pair sums scatter into cohort space.
    summary_generations = np.asarray(summary.generations)
    if not np.all(np.isin(summary_generations, cohorts.generations)):
        raise ValueError("generation kinship summary does not describe the estimator's observed cohorts")
    pair_counts = np.asarray(summary.pair_counts, dtype=np.float64)
    mean_theta = np.asarray(summary.mean_kinship, dtype=np.float64)
    sum_theta = np.zeros(cohorts.k, dtype=np.float64)
    sum_theta[np.searchsorted(cohorts.generations, summary_generations)] = np.where(
        pair_counts > 0.0, mean_theta * pair_counts, 0.0
    )

    kept = labels >= 0
    bucket = cohorts.dense[kept]
    n_genomes = np.bincount(bucket, minlength=cohorts.k).astype(np.int64)
    s = (1.0 + np.asarray(pg._inbreeding_values(), dtype=np.float64)[kept]) / 2.0
    sum_s = np.bincount(bucket, weights=s, minlength=cohorts.k)

    f_bar = np.full(cohorts.k, np.nan, dtype=np.float64)
    present = n_genomes > 0
    f_bar[present] = (2.0 * sum_theta[present] + sum_s[present]) / n_genomes[present].astype(np.float64) ** 2
    return GroupCoancestryByCohort(np.asarray(cohorts.generations, dtype=np.int32), f_bar, n_genomes)


def _group_coancestry_from(cohorts: ObservedCohorts, gc: GroupCoancestryByCohort) -> NeGroupCoancestryResult:
    """Reduce per-cohort group coancestry to the result record.

    The first cohort's f̄ is the computed baseline and is reported and rated
    as computed.  It is never overwritten with the ``0.5`` self-coancestry of
    a non-inbred individual, which is a per-individual quantity and not a
    group one: where the baseline cohort is unrelated and non-inbred the
    computed value is ``1/(2n)``, and the first transition is then Caballero
    & Toro's own eq. 11 ``Δf₀,₁``.

    ``census_ratio`` reads the census of exactly the cohorts the fit uses,
    off :func:`~pedigree_graph._ne_common._log_fit_mask` — the one predicate
    that also selects the fitted points and counts ``n_generations_used``,
    so the diagnostic and the count can never describe different cohorts.
    Every cohort it selects has a finite f̄, and f̄ is NaN wherever the
    cohort holds no genome-node representative, so the minimum census over
    that set is at least 1 and the ratio never divides by zero.

    Args:
        cohorts: The estimator's observed-cohort grouping.
        gc: Prerequisite from :func:`_group_coancestry_by_cohort`.

    Returns:
        The :class:`~pedigree_graph._ne_results.NeGroupCoancestryResult`.

    Raises:
        ValueError: when *gc* describes different cohorts than *cohorts*.
    """
    if not np.array_equal(gc.generations, cohorts.generations):
        raise ValueError("group coancestry does not describe the estimator's observed cohorts")
    f_bar = np.asarray(gc.mean_group_coancestry, dtype=np.float64)
    fit = _scalar_ne_from_log_regression(f_bar, cohorts.generations)
    fitted_census = np.asarray(gc.n_genomes, dtype=np.float64)[1:][_log_fit_mask(f_bar[1:])]
    return NeGroupCoancestryResult(
        **fit._asdict(),
        generations=cohorts.generations,
        mean_group_coancestry_per_gen=f_bar,
        n_genomes_per_gen=gc.n_genomes,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(f_bar, cohorts.generations),
        census_ratio=float(fitted_census.max() / fitted_census.min()) if fitted_census.size else float("nan"),
    )
