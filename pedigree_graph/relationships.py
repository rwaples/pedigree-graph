"""Public relationship vocabulary and the typed relationship-pair results.

:data:`RELATIONSHIPS` maps each of the 23 relationship codes to an immutable
:class:`RelationshipCategory`.  Iteration order is the documented same-degree
precedence for closest-category classification.  See
:class:`RelationshipCategory` for the ``first`` / ``second`` orientation rule
and :data:`RelationshipRole` for the closed set of role names.

:class:`RelationshipPairs` is what ``PedigreeGraph.relationship_pairs`` and
``PedigreeView.relationship_pairs`` return: an immutable mapping over all 23
codes whose :class:`RelationshipPairBlock` values own read-only int32 rows of
the receiver in the category's semantic orientation (ADR 0006).
:class:`RelationshipCountResult` is the matching ``relationship_counts``
result: one ``int | None`` per code plus the sets naming how each count was
obtained.
"""

from __future__ import annotations

__all__ = [
    "RELATIONSHIPS",
    "RelationshipCategory",
    "RelationshipCountResult",
    "RelationshipPairBlock",
    "RelationshipPairs",
    "RelationshipRole",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from pedigree_graph._registry import RELATIONSHIPS, RelationshipCategory, RelationshipRole

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from pedigree_graph._view import CoordinateToken


# eq=False: the numpy fields have no scalar equality, and two blocks are the
# same result only when they came from the same call.
@dataclass(frozen=True, slots=True, eq=False)
class RelationshipPairBlock:
    """The pairs of one relationship category, in the receiver's rows.

    ``first_rows[k]`` and ``second_rows[k]`` are the two members of pair ``k``,
    as graph rows for a graph receiver and view rows for a view receiver.
    For an asymmetric category ``first`` carries :attr:`first_role` and
    ``second`` carries :attr:`second_role`; a symmetric category stores
    ``first < second``.  Pairs are sorted by the canonical unordered row key,
    and a pair belongs to exactly one category across the whole result.

    Attributes:
        category: The registry record this block belongs to.
        first_rows: Owned, read-only int32 rows of the first member.
        second_rows: Owned, read-only int32 rows of the second member.
        requested: Whether the caller's selector named this category.  An
            unrequested block is always empty.
    """

    category: RelationshipCategory
    first_rows: np.ndarray
    second_rows: np.ndarray
    requested: bool
    _coordinate_token: CoordinateToken = field(repr=False)

    @property
    def code(self) -> str:
        """The category's registry code."""
        return self.category.code

    @property
    def first_role(self) -> RelationshipRole | None:
        """Role of ``first_rows``, ``None`` for a symmetric category."""
        return self.category.first_role

    @property
    def second_role(self) -> RelationshipRole | None:
        """Role of ``second_rows``, ``None`` for a symmetric category."""
        return self.category.second_role

    def __len__(self) -> int:
        return len(self.first_rows)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.first_rows
        yield self.second_rows


class RelationshipPairs(Mapping[str, RelationshipPairBlock]):
    """Immutable mapping from every registry code to its :class:`RelationshipPairBlock`.

    Iteration follows :data:`RELATIONSHIPS` order and always yields all 23
    codes; a category the selector did not name is present as an empty block
    with ``requested=False``.

    Args:
        blocks: One block per registry code, already in registry order.
    """

    __slots__ = ("_blocks",)

    def __init__(self, blocks: Mapping[str, RelationshipPairBlock]) -> None:
        assert tuple(blocks) == tuple(RELATIONSHIPS), "RelationshipPairs needs every registry code in registry order"
        self._blocks = dict(blocks)

    def __getitem__(self, code: str) -> RelationshipPairBlock:
        return self._blocks[code]

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        counts = ", ".join(f"{code}={len(block)}" for code, block in self._blocks.items() if block.requested)
        return f"RelationshipPairs({counts})"


# eq=False keeps Mapping's value equality; a generated __hash__ would fail on
# the dict field.
@dataclass(frozen=True, slots=True, eq=False)
class RelationshipCountResult(Mapping[str, int | None]):
    """Immutable mapping from every registry code to its pair count, or ``None``.

    Iteration follows :data:`RELATIONSHIPS` order and always yields all 23
    codes; a category the selector did not name maps to ``None``.  Build one
    with :meth:`from_pairs`.

    Attributes:
        requested: Codes the selector named.
        exact: Requested codes whose count is exact.
        approximate: Requested codes whose count is an estimate.
        clamped: Requested codes whose inclusion-exclusion residual underflowed
            and was floored at 0; that 0 is not a true absence.
    """

    _counts: dict[str, int | None]
    requested: frozenset[str]
    exact: frozenset[str]
    approximate: frozenset[str]
    clamped: frozenset[str]

    def __post_init__(self) -> None:
        assert tuple(self._counts) == tuple(RELATIONSHIPS), "RelationshipCountResult needs every code in registry order"

    @classmethod
    def from_pairs(cls, pairs: RelationshipPairs) -> Self:
        """Return the exact counts of *pairs*: block lengths, ``None`` where unrequested."""
        requested = frozenset(code for code, block in pairs.items() if block.requested)
        counts = {code: len(block) if block.requested else None for code, block in pairs.items()}
        return cls(counts, requested, requested, frozenset(), frozenset())

    def __getitem__(self, code: str) -> int | None:
        return self._counts[code]

    def __iter__(self) -> Iterator[str]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)

    def __repr__(self) -> str:
        counts = ", ".join(f"{code}={count}" for code, count in self._counts.items() if code in self.requested)
        return f"RelationshipCountResult({counts})"
