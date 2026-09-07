"""Observed generation cohorts: the grouping every cohort-indexed result shares.

Generation labels are nonnegative int32 or ``-1`` (unknown), may be rebased
or sparse, and never drive pedigree structure.  :class:`ObservedCohorts`
maps them to dense bucket indices once, so the kinship summary and every
effective-size estimator allocate one slot per observed label rather than
one per label value, and so all of them agree on what "cohort ``b``" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._ne_metadata import _require_complete_generation_labels

if TYPE_CHECKING:
    from pedigree_graph._core import PedigreeGraph


def _densify_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Map cohort labels to dense bucket indices.

    Observed labels (``>= 0``) become ``0 .. k-1`` in ascending label order;
    every ``-1`` row becomes the sentinel bucket ``k``.  Accumulators then
    allocate ``k + 1`` buckets from ``dense.max()`` — bounded by the number of
    distinct labels, never by the label values — and the sentinel bucket,
    which only ever collects unlabelled-with-unlabelled pairs, is discarded
    by the consumer.

    Returns:
        ``(dense, observed, n_unlabelled)``: int32 bucket index per row, the
        ascending int32 observed labels, and the count of ``-1`` rows.
    """
    raw = np.ascontiguousarray(labels, dtype=np.int32)
    labelled = raw >= 0
    observed, inverse = np.unique(raw[labelled], return_inverse=True)
    dense = np.full(raw.shape[0], observed.shape[0], dtype=np.int32)
    dense[labelled] = inverse.astype(np.int32)
    n_unlabelled = int(raw.shape[0] - np.count_nonzero(labelled))
    return dense, observed.astype(np.int32), n_unlabelled


@dataclass(frozen=True, slots=True)
class ObservedCohorts:
    """Rows grouped by observed generation label, in ascending label order.

    Attributes:
        generations: Observed labels, int32, ascending, length ``k``.
        dense: Bucket index per row, int32; ``k`` for an unlabelled row.
        counts: Rows per bucket, int64, length ``k``.
        unlabelled_individual_count: Rows whose label is ``-1``.
    """

    generations: np.ndarray
    dense: np.ndarray
    counts: np.ndarray
    unlabelled_individual_count: int

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> ObservedCohorts:
        """Group by the labels as given; ``-1`` rows belong to no cohort."""
        dense, observed, n_unlabelled = _densify_labels(labels)
        k = int(observed.shape[0])
        counts = np.bincount(dense, minlength=k + 1)[:k].astype(np.int64)
        return cls(observed, dense, counts, n_unlabelled)

    @classmethod
    def for_graph(cls, pg: PedigreeGraph, operation: str) -> ObservedCohorts:
        """Cohorts an estimator groups by: supplied labels, else structural depth.

        Raises:
            MissingMetadataError: ``missing_generation_labels`` when the
                supplied labels are partly ``-1``.
        """
        _require_complete_generation_labels(pg, operation)
        labels = pg.generation_labels
        return cls.from_labels(pg.depth if labels is None else labels)

    @property
    def k(self) -> int:
        """Number of observed cohorts."""
        return int(self.generations.shape[0])

    def __len__(self) -> int:
        return self.k

    def members(self) -> list[np.ndarray]:
        """Row indices per bucket, each ascending, in one pass over the rows."""
        k = self.k
        order = np.argsort(self.dense, kind="stable")
        bounds = np.searchsorted(self.dense[order], np.arange(k + 1))
        return [order[bounds[b] : bounds[b + 1]] for b in range(k)]

    def transition_from(self) -> np.ndarray:
        """Label of the earlier cohort of each adjacent observed pair."""
        return self.generations[:-1] if self.k else self.generations

    def transition_to(self) -> np.ndarray:
        """Label of the later cohort of each adjacent observed pair."""
        return self.generations[1:] if self.k else self.generations
