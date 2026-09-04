"""The ADR 0006 read-only property surface of :class:`~pedigree_graph._core.PedigreeGraph`.

Every array here is the graph's own storage, handed out unchanged: the same
object on every access, marked read-only when it was built, so a caller can
neither copy-cost its way through a loop nor mutate the graph by writing into
what it read. Only :attr:`PedigreeProperties.depth` is derived, and it is
computed on first access rather than at construction.

The mixin reads two attributes the graph sets while building itself: the
parsed :class:`~pedigree_graph._input.PedigreeInput` and the flag that says
whether this graph was built through a 0.7.1 entry point, which is the only
thing that turns absent sex into the all-female default.
"""

from __future__ import annotations

__all__ = ["PedigreeProperties"]

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from pedigree_graph._topology import readonly, structural_depth

if TYPE_CHECKING:
    from pedigree_graph._input import PedigreeInput


class PedigreeProperties:
    """Read-only views onto a graph's owned input arrays and its structural depth."""

    _input: PedigreeInput
    _legacy_defaults: bool  # 0.8.0-DELETE

    @property
    def ids(self) -> np.ndarray:
        """Read-only int64 id per graph row, unique and nonnegative."""
        return self._input.ids

    @property
    def mother_ids(self) -> np.ndarray:
        """Read-only int64 mother id per graph row, ``-1`` when missing.

        A nonnegative id whose :attr:`mother_rows` entry is ``-1`` is an
        external reference: a real parent outside the represented rows.
        """
        return self._input.mother_ids

    @property
    def father_ids(self) -> np.ndarray:
        """Read-only int64 father id per graph row, as :attr:`mother_ids`."""
        return self._input.father_ids

    @property
    def twin_ids(self) -> np.ndarray:
        """Read-only int64 MZ co-twin id per graph row, as :attr:`mother_ids`."""
        return self._input.twin_ids

    @property
    def mother_rows(self) -> np.ndarray:
        """Read-only int32 mother row per graph row, ``-1`` when missing or external."""
        return self._input.mother_rows

    @property
    def father_rows(self) -> np.ndarray:
        """Read-only int32 father row per graph row, as :attr:`mother_rows`."""
        return self._input.father_rows

    @property
    def twin_rows(self) -> np.ndarray:
        """Read-only int32 MZ co-twin row per graph row, as :attr:`mother_rows`."""
        return self._input.twin_rows

    @property
    def sex(self) -> np.ndarray | None:
        """Read-only int8 sex per graph row, or ``None`` when no sex is known.

        ``0`` female, ``1`` male, ``-1`` unknown. A wholly unknown column
        reads as ``None``, exactly like an omitted one.
        """
        parsed = self._input.sex
        if parsed is None and self._legacy_defaults:  # 0.8.0-DELETE
            return self._legacy_sex  # 0.8.0-DELETE
        return parsed

    @cached_property
    def depth(self) -> np.ndarray:
        """Read-only int32 structural depth per graph row, founders ``0``.

        Derived from the parent edges alone, so supplied generation labels
        never move it. Computed on first access: graphs that only extract
        pairs in graph order never pay for it.
        """
        return structural_depth(self._input.mother_rows, self._input.father_rows, self._input.n_individuals)

    @property
    def generation_labels(self) -> np.ndarray | None:
        """Read-only int32 supplied generation labels, or ``None`` when absent.

        Cohort labels as the caller supplied them, ``-1`` where unknown. They
        describe the pedigree; they never drive an ordering. :attr:`depth` does.
        """
        return self._input.generation

    @property
    def birth_year(self) -> np.ndarray | None:
        """Read-only int32 birth year per graph row, or ``None`` when absent."""
        return self._input.birth_year

    @property
    def n_individuals(self) -> int:
        """Number of represented individuals, one per graph row."""
        return self._input.n_individuals

    def __len__(self) -> int:
        return self._input.n_individuals

    # 0.8.0-DELETE: the 0.7.1 all-female default for a graph built without sex.
    @cached_property
    def _legacy_sex(self) -> np.ndarray:
        """Read-only all-female int8 sex column."""
        return readonly(np.zeros(self._input.n_individuals, dtype=np.int8))

    # 0.8.0-DELETE: superseded by generation_labels plus depth; the 0.7.1
    # estimators read one array and expect depth when no labels were supplied.
    @property
    def generation(self) -> np.ndarray:
        """Supplied generation labels when present, else structural depth."""
        labels = self._input.generation
        return self.depth if labels is None else labels
