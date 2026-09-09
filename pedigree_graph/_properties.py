"""The ADR 0006 read-only property surface of :class:`~pedigree_graph._core.PedigreeGraph`.

Every array here is the graph's own storage, handed out unchanged: the same
object on every access, marked read-only when it was built, so a caller can
neither copy-cost its way through a loop nor mutate the graph by writing into
what it read. Only :attr:`PedigreeProperties.depth` is derived, and it is
computed on first access rather than at construction.

The mixin reads one attribute the graph sets while building itself: the
natively built columns, a ``pedigree_graph._native.BuiltPedigree``.
"""

from __future__ import annotations

__all__ = ["PedigreeProperties"]

from functools import cached_property
from typing import TYPE_CHECKING

from pedigree_graph import _native
from pedigree_graph._topology import structural_depth

if TYPE_CHECKING:
    import numpy as np


class PedigreeProperties:
    """Read-only views onto a graph's owned input arrays and its structural depth."""

    _built: _native.BuiltPedigree

    @property
    def ids(self) -> np.ndarray:
        """Read-only int64 id per graph row, unique and nonnegative."""
        return self._built.ids

    @property
    def mother_ids(self) -> np.ndarray:
        """Read-only int64 mother id per graph row, ``-1`` when missing.

        A nonnegative id whose :attr:`mother_rows` entry is ``-1`` is an
        external reference: a real parent outside the represented rows.
        """
        return self._built.mother_ids

    @property
    def father_ids(self) -> np.ndarray:
        """Read-only int64 father id per graph row, as :attr:`mother_ids`."""
        return self._built.father_ids

    @property
    def twin_ids(self) -> np.ndarray:
        """Read-only int64 MZ co-twin id per graph row, as :attr:`mother_ids`."""
        return self._built.twin_ids

    @property
    def mother_rows(self) -> np.ndarray:
        """Read-only int32 mother row per graph row, ``-1`` when missing or external."""
        return self._built.mother_rows

    @property
    def father_rows(self) -> np.ndarray:
        """Read-only int32 father row per graph row, as :attr:`mother_rows`."""
        return self._built.father_rows

    @property
    def twin_rows(self) -> np.ndarray:
        """Read-only int32 MZ co-twin row per graph row, as :attr:`mother_rows`."""
        return self._built.twin_rows

    @property
    def sex(self) -> np.ndarray | None:
        """Read-only int8 sex per graph row, or ``None`` when no sex is known.

        ``0`` female, ``1`` male, ``-1`` unknown. A wholly unknown column
        reads as ``None``, exactly like an omitted one.
        """
        return self._built.sex

    @cached_property
    def depth(self) -> np.ndarray:
        """Read-only int32 structural depth per graph row, founders ``0``.

        Derived from the parent edges alone, so supplied generation labels
        never move it. Computed on first access: graphs that only extract
        pairs in graph order never pay for it.
        """
        return structural_depth(self._built.mother_rows, self._built.father_rows)

    @property
    def generation_labels(self) -> np.ndarray | None:
        """Read-only int32 supplied generation labels, or ``None`` when absent.

        Cohort labels as the caller supplied them, ``-1`` where unknown. They
        describe the pedigree; they never drive an ordering. :attr:`depth` does.
        """
        return self._built.generation

    @property
    def birth_year(self) -> np.ndarray | None:
        """Read-only int32 birth year per graph row, or ``None`` when absent."""
        return self._built.birth_year

    @property
    def n_individuals(self) -> int:
        """Number of represented individuals, one per graph row."""
        return len(self._built.ids)

    @cached_property
    def _id_index(self) -> _native.IdIndex:
        """Sorted-id lookup built on first id selection and reused by every later one."""
        return _native.IdIndex(self._built.ids)

    def __len__(self) -> int:
        return len(self._built.ids)
