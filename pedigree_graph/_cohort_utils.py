"""Birth-year utilities for overlapping-generation Ne estimators.

:func:`generation_interval` is the body of
:attr:`PedigreeGraph.generation_interval`, the sex-split Hill 1979 ``L``.
The Hill 1979 separate-sex Ne_H estimator also needs to restrict to
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

from pedigree_graph._errors import MissingMetadataError
from pedigree_graph._ne_results import GenerationInterval

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def generation_interval(pg: PedigreeGraph) -> GenerationInterval | None:
    """Sex-split generation interval (Hill 1979 ``L``) of ``pg``.

    ``T_m`` is the mean ``child.birth_year − sire.birth_year`` over
    sire-offspring edges with both birth years known, ``T_f`` the symmetric
    dam form, ``T = (T_m + T_f) / 2``; skip-generation edges are included
    unconditionally.  Returns ``None`` only when ``pg.birth_year is None``.

    Raises:
        MissingMetadataError: ``insufficient_parent_age_data`` when birth
            years are present but a parent role has no edge with both birth
            years known; ``missing_parent_roles`` lists the roles in
            ``("mother", "father")`` order.
    """
    if pg.birth_year is None:
        return None
    diffs_f = pg._known_parent_edges_for("mother")[1]
    diffs_m = pg._known_parent_edges_for("father")[1]
    missing_roles = tuple(role for role, diffs in (("mother", diffs_f), ("father", diffs_m)) if diffs.size == 0)
    if missing_roles:
        raise MissingMetadataError(
            "insufficient_parent_age_data",
            f"generation_interval: no parent-child edge with both birth years known for {', '.join(missing_roles)}",
            operation="generation_interval",
            missing_parent_roles=missing_roles,
        )
    T_m = float(diffs_m.mean())
    T_f = float(diffs_f.mean())
    return GenerationInterval(T=(T_m + T_f) / 2.0, T_m=T_m, T_f=T_f, n_edges=int(diffs_m.size + diffs_f.size))


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
        MissingMetadataError: ``missing_birth_year`` (``status="absent"``,
            every row counted) if ``pg.birth_year is None``;
            ``insufficient_parent_age_data`` with both roles listed if no
            parent-child edge has both birth years known, so the
            percentile is undefined.  One role with qualifying edges is
            enough for the percentile.

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
        raise MissingMetadataError(
            "missing_birth_year",
            "eligible_cohort_range: the pedigree carries no birth years",
            operation="eligible_cohort_range",
            status="absent",
            missing_count=pg.n_individuals,
        )

    by = pg.birth_year
    known = by[by >= 0]

    y_min = int(known.min())
    y_max = int(known.max())

    diffs = [d for parent_label in ("mother", "father") if (d := pg._known_parent_edges_for(parent_label)[1]).size > 0]
    if not diffs:
        raise MissingMetadataError(
            "insufficient_parent_age_data",
            "eligible_cohort_range: no parent-child edge with both birth years known; the percentile is undefined",
            operation="eligible_cohort_range",
            missing_parent_roles=("mother", "father"),
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
