"""Cohort eligibility utilities for overlapping-generation Ne estimators.

The Hill 1979 separate-sex Ne_H estimator needs to restrict to
*eligible* birth cohorts whose members have had time to complete
their reproductive lifespans by the end of the observed pedigree.
This module provides the public utility :func:`eligible_cohort_range`
that returns a :class:`CohortWindow` ``(c_min, c_max,
reproductive_age_p95)``.

The eligibility heuristic is the in-sample 95th percentile of the
parent-child birth-year difference: any cohort whose youngest 5%
of lifetime reproduction would land outside the pedigree's
observation window is excluded.
"""

from __future__ import annotations

__all__ = ["CohortWindow", "eligible_cohort_range"]

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


class CohortWindow(NamedTuple):
    """Eligible birth-year window for cohort-based Ne estimation.

    Attributes:
        c_min: Inclusive lower bound on eligible birth-year cohort.
        c_max: Inclusive upper bound on eligible birth-year cohort.
        reproductive_age_p95: The in-sample 95th percentile of
            ``child.birth_year − parent.birth_year`` used as the
            right-censoring cutoff.
    """

    c_min: int
    c_max: int
    reproductive_age_p95: float


def eligible_cohort_range(
    pg: PedigreeGraph,
    *,
    percentile: float = 95.0,
    c_min: int | None = None,
    c_max: int | None = None,
) -> CohortWindow:
    """Return the eligible birth-year window for cohort-Ne estimation.

    The default window is ``(y_min, y_max - reproductive_age_p95)``
    where ``y_min`` / ``y_max`` are the min / max of known
    ``pg.birth_year`` and ``reproductive_age_p95`` is the requested
    percentile (default 95) of ``child.birth_year − parent.birth_year``
    over all parent-child edges with both endpoints having known
    ``birth_year`` (sentinel ``-1`` skipped).

    Args:
        pg: Pedigree graph with ``birth_year`` attached.
        percentile: Percentile (0–100) of the parent-child age difference
            distribution to use as the right-censoring cutoff.  Defaults
            to ``95.0``.
        c_min: Override the lower bound of the window.  ``None`` keeps
            the heuristic default.
        c_max: Override the upper bound of the window.  ``None`` keeps
            the heuristic default.

    Returns:
        :class:`CohortWindow` ``(c_min, c_max, reproductive_age_p95)``.

    Raises:
        ValueError: If ``pg.birth_year is None``, or if no parent-child
            edges have both endpoints with known birth_year (so the
            percentile is undefined).

    Notes:
        The in-sample percentile is self-referentially biased under
        severe right-censoring of the pedigree: if the pedigree is
        truncated before older reproductive ages have been observed,
        ``reproductive_age_p95`` will under-estimate the true cutoff
        and the returned ``c_max`` will be too lax.  Users with known
        truncation should pass an explicit ``c_max`` based on their
        domain knowledge (e.g., the species' maximum reproductive age
        from a life table).
    """
    if pg.birth_year is None:
        raise ValueError("eligible_cohort_range requires pg.birth_year to be set")

    by = pg.birth_year
    known = by[by >= 0]
    if known.size == 0:
        raise ValueError("eligible_cohort_range: no individuals have known birth_year")

    y_min = int(known.min())
    y_max = int(known.max())

    # Parent-child age-difference distribution, both endpoints known.
    diffs: list[np.ndarray] = []
    for parent_arr in (pg.mother, pg.father):
        edge_rows = np.where(parent_arr >= 0)[0]
        if edge_rows.size == 0:
            continue
        parents = parent_arr[edge_rows]
        by_child = by[edge_rows]
        by_parent = by[parents]
        both = (by_child >= 0) & (by_parent >= 0)
        if np.any(both):
            diffs.append((by_child[both] - by_parent[both]).astype(np.float64))

    if not diffs:
        raise ValueError(
            "eligible_cohort_range: no parent-child edges with both birth_years known; "
            "cannot compute reproductive_age_p95"
        )

    all_diffs = np.concatenate(diffs)
    reproductive_age_p95 = float(np.percentile(all_diffs, percentile))

    default_c_min = y_min
    default_c_max = int(np.floor(y_max - reproductive_age_p95))

    return CohortWindow(
        c_min=default_c_min if c_min is None else int(c_min),
        c_max=default_c_max if c_max is None else int(c_max),
        reproductive_age_p95=reproductive_age_p95,
    )
