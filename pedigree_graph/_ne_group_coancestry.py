"""Group-coancestry Ne estimator and its per-cohort prerequisite (PGQ-006).

Caballero & Toro 2000 (Genet. Res. 75:331) eq. 3 group coancestry, evaluated
per observed cohort over the genome-node pedigree of ADR 0008 and reduced to
a scalar by the ``ln(1 − x)`` regression the other rate estimators share.

The genome-node collapse is a row mask: a masked co-twin's ``-1`` label
sends it to the sentinel bucket
:func:`~pedigree_graph._cohorts._densify_labels` already discards.  So the
prerequisite reads per-cohort θ sums off the streaming DP rather than
materialising a kinship matrix, and it reuses outright the summary the graph
memoises for :func:`~pedigree_graph._ne_rates.ne_coancestry`: since issue #25
that summary is itself genome-node, so both estimators share one convention
and one DP pass whether or not the pedigree has MZ twins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_common import (
    _genome_node_labels,
    _log_fit_mask,
    _scalar_ne_from_log_regression,
    _transition_ne,
)
from pedigree_graph._ne_rates import _generation_kinship_summary
from pedigree_graph._ne_results import NeGroupCoancestryResult

if TYPE_CHECKING:
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
    ne_scalar, slope, n_used = _scalar_ne_from_log_regression(f_bar, cohorts.generations)
    fitted_census = np.asarray(gc.n_genomes, dtype=np.float64)[1:][_log_fit_mask(f_bar[1:])]
    return NeGroupCoancestryResult(
        ne=ne_scalar,
        generations=cohorts.generations,
        mean_group_coancestry_per_gen=f_bar,
        n_genomes_per_gen=gc.n_genomes,
        transition_from=cohorts.transition_from(),
        transition_to=cohorts.transition_to(),
        ne_per_gen=_transition_ne(f_bar, cohorts.generations),
        slope=slope,
        n_generations_used=n_used,
        census_ratio=float(fitted_census.max() / fitted_census.min()) if fitted_census.size else float("nan"),
    )


def ne_group_coancestry(pg: PedigreeGraph) -> NeGroupCoancestryResult:
    """Group-coancestry rate Ne (Ne_GC).

    Caballero & Toro, *Interrelations between effective population size and
    other pedigree tools for the management of conserved populations*,
    Genet. Res. 75(3):331-343, 2000, eq. 3::

        f̄ = Σ_i Σ_j a_ij / 2N²

    over all ``N²`` ordered pairs of a group, self-coancestries and
    reciprocals included, with ``a_ij = 2φ_ij`` off the diagonal and
    ``a_ii = 1 + F_i`` on it.  Each observed cohort is one such group,
    taken over the genome-node pedigree of ADR 0008: an MZ pair is one
    genome, not two.  The paper has no MZ twins, so it is silent there
    rather than contradicted.

    The reduction to a scalar is **not** their eq. 11 ``Ne ≈ t/(2Δf₀,ₜ)``,
    which the authors themselves flag as "only accurate if F̄ₖ₋₁ is small".
    It is the regression of ``ln(1 − f̄_g)`` on the label offset that
    :func:`~pedigree_graph._ne_rates.ne_inbreeding` and
    :func:`~pedigree_graph._ne_rates.ne_coancestry` share, whose exact
    cumulative form ``1 − x_t = (1 − Δ)^t`` is Gutiérrez et al. 2008
    (Genet. Sel. Evol. 40:359) eq. 1.

    The first observed cohort is the regression baseline, and its f̄ is
    computed like every other cohort's rather than assumed.  Where that
    cohort is unrelated and non-inbred the computed value is ``1/(2n)``
    exactly — the ``f̄₀ = 1/(2N)`` the paper states for unrelated founders —
    and the first entry of ``ne_per_gen`` is then their eq. 11
    ``Δf₀,₁ = (f̄₁ − f̄₀)/(1 − f̄₀)``.

    **The scalar assumes a constant census**, and ``census_ratio`` on the
    record is the evidence for or against that.
    ``f̄_g = θ̄_g·(n_g − 1)/n_g + s̄_g/n_g`` is an identity whose second term
    is a self-coancestry near ``0.5`` over the cohort size; that term does
    not accumulate at the drift rate, so ``ln(1 − f̄)`` reads a change in
    census as drift.  Eq. 11 carries the same assumption, fixing ``N`` at
    ``f̄₀ = 1/(2N)`` and holding it after, so this is faithful to the source
    rather than a defect in the implementation.  Measured over 6 cohorts and
    8 seeds against :func:`~pedigree_graph._ne_rates.ne_coancestry`, which
    reads the same drift without the diagonal term: a constant census of 40
    gives ``Ne_GC/Ne_C = 1.00``, a decline from 40 to 4 gives ``0.57``,
    growth from 10 to 40 gives ``1.28``, and a lone trailing individual
    gives ``0.08`` — ``ne`` of 3.40 where ``ne_coancestry`` is unmoved at
    40.73.  The number is always reported; weigh it against
    ``census_ratio``.

    Requires complete generation labels, or none, in which case structural
    depth groups the cohorts.  It reads no founder contributions, so unlike
    :func:`~pedigree_graph._ne_founders.ne_long_term_contributions` it does
    not require closed represented parentage.

    Args:
        pg: Pedigree graph.

    Returns:
        The :class:`~pedigree_graph._ne_results.NeGroupCoancestryResult`.

    Raises:
        MissingMetadataError: ``missing_generation_labels`` when the supplied
            labels are partly ``-1``.
    """
    cohorts = ObservedCohorts.for_graph(pg, "ne_group_coancestry")
    return _group_coancestry_from(cohorts, _group_coancestry_by_cohort(pg, cohorts))
