"""Public summary records over a whole pedigree graph.

:class:`GenerationKinshipSummary` is the result of
:meth:`~pedigree_graph.PedigreeGraph.mean_kinship_by_generation` (ADR 0006).
It is data only; the DP that fills it lives in ``pedigree_graph._kinship_dp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pedigree_graph._topology import owned_readonly

__all__ = ["GenerationKinshipSummary"]


@dataclass(frozen=True, slots=True)
class GenerationKinshipSummary:
    """Mean pedigree-expected kinship within each observed generation.

    One row per generation label that at least one individual carries, in
    ascending label order.  ``mean_kinship[g]`` averages the kinship
    coefficient over the unordered pairs of distinct individuals in group
    ``g``, excluding MZ twin pairs whose two co-twins are both in ``g``.
    ``pair_counts[g]`` is that pair count; ``mean_kinship[g]`` is NaN where
    it is zero.  Individuals whose label is unknown (``-1``) belong to no
    group and are tallied in ``unlabelled_individual_count``.

    Attributes:
        generations: Observed labels, int32, ascending.
        mean_kinship: Float64 mean kinship per observed label.
        pair_counts: Int64 number of averaged pairs per observed label.
        unlabelled_individual_count: Rows whose label is ``-1``.
    """

    generations: np.ndarray
    mean_kinship: np.ndarray
    pair_counts: np.ndarray
    unlabelled_individual_count: int

    def __post_init__(self) -> None:
        """Freeze the arrays and check they describe the same groups."""
        generations = owned_readonly(self.generations, np.int32)
        mean_kinship = owned_readonly(self.mean_kinship, np.float64)
        pair_counts = owned_readonly(self.pair_counts, np.int64)
        if not (generations.shape == mean_kinship.shape == pair_counts.shape) or generations.ndim != 1:
            raise ValueError("generations, mean_kinship, and pair_counts must be 1-D arrays of one length")
        object.__setattr__(self, "generations", generations)
        object.__setattr__(self, "mean_kinship", mean_kinship)
        object.__setattr__(self, "pair_counts", pair_counts)
        object.__setattr__(self, "unlabelled_individual_count", int(self.unlabelled_individual_count))

    def __len__(self) -> int:
        """Number of observed generations."""
        return int(self.generations.shape[0])
