"""Metadata guards the effective-size estimators run before any work (ADR 0006).

Each guard raises :class:`~pedigree_graph.MissingMetadataError` with a stable
code and the fields that code requires, naming the estimator in
``operation`` so an orchestrated failure identifies the estimator it
disables.  An empty graph is a valid no-estimate case and passes every
guard.  Which estimator needs which guard, and in which order, is the
metadata dependency matrix documented on
:mod:`pedigree_graph.effective_size`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._errors import MissingMetadataError

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def _require_complete_generation_labels(pg: PedigreeGraph, operation: str) -> None:
    """Reject a graph whose supplied generation labels are only partly known.

    A ``-1`` label indexes no cohort; left unchecked it wraps into the last
    bucket of every label-indexed accumulator and silently biases the result.
    Absent labels are not an error here: the estimators fall back to
    structural depth through
    :meth:`~pedigree_graph._cohorts.ObservedCohorts.for_graph`.

    Raises:
        MissingMetadataError: ``missing_generation_labels`` with
            ``status="partial"`` and the number of unknown labels.
    """
    labels = pg.generation_labels
    if labels is None:
        return
    missing = int(np.count_nonzero(labels < 0))
    if missing:
        raise MissingMetadataError(
            "missing_generation_labels",
            f"{operation}: {missing} of {labels.size} generation labels are unknown (-1)",
            operation=operation,
            status="partial",
            missing_count=missing,
        )


def _require_complete_sex(pg: PedigreeGraph, operation: str) -> None:
    """Reject a graph whose sex is absent or partly unknown.

    The sex-dependent estimators count offspring per parent and members per
    cohort; dropping unknown rows would change both the numerators and the
    denominators, so they are never filtered.  Uniform but fully known sex
    is valid here (the estimator warns and reports no estimate).

    Raises:
        MissingMetadataError: ``missing_sex`` with ``status="absent"`` and
            every row counted when the graph carries no sex, or
            ``status="partial"`` with the number of ``-1`` rows.
    """
    if pg.n_individuals == 0:
        return
    sex = pg.sex
    if sex is None:
        raise MissingMetadataError(
            "missing_sex",
            f"{operation}: the pedigree carries no sex",
            operation=operation,
            status="absent",
            missing_count=pg.n_individuals,
        )
    missing = int(np.count_nonzero(sex < 0))
    if missing:
        raise MissingMetadataError(
            "missing_sex",
            f"{operation}: {missing} of {sex.size} sex values are unknown (-1)",
            operation=operation,
            status="partial",
            missing_count=missing,
        )


def _require_closed_parentage(pg: PedigreeGraph, operation: str) -> None:
    """Reject a graph in which some row has exactly one represented parent.

    The founder-contribution recurrences halve each parent's row into the
    child; a child with one represented parent would keep only half its
    ancestry, so the long-term-contribution and Caballero-Toro estimators
    require every row to have zero represented parents (a represented
    founder) or two.  The unrepresented parent is ``"missing"`` when its id
    is ``-1`` and ``"external"`` when it names an id outside the graph.

    Raises:
        MissingMetadataError: ``incomplete_parentage`` with the number of
            affected rows and the first one's row, id, and parent roles.
    """
    mother = np.asarray(pg.mother_rows)
    father = np.asarray(pg.father_rows)
    affected = np.flatnonzero((mother < 0) != (father < 0))
    if affected.size == 0:
        return
    first = int(affected[0])
    mother_present = mother[first] >= 0
    unrepresented_id = int(pg.father_ids[first] if mother_present else pg.mother_ids[first])
    raise MissingMetadataError(
        "incomplete_parentage",
        f"{operation}: {affected.size} rows have exactly one represented parent (first: row {first})",
        operation=operation,
        affected_count=int(affected.size),
        first_row=first,
        first_id=int(pg.ids[first]),
        represented_parent_role="mother" if mother_present else "father",
        unrepresented_parent_role="father" if mother_present else "mother",
        unrepresented_parent_status="missing" if unrepresented_id < 0 else "external",
    )
