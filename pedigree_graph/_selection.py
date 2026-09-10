"""The resolved form of one relationship selector, parsed at the public boundary.

Every public relationship endpoint — ``relationship_pairs``,
``relationship_counts``, ``relationship_kinship_matrix``, and their view
counterparts — takes the same one-of-two selector: ``max_degree=`` or
``categories=``.  :class:`RelationshipSelection` is what that selector becomes
once validated, so the pair, count, and matrix engines share one parse instead
of each reaching into a sibling's privates.

Parsing once at the boundary buys three things the engines used to arrange for
themselves.  A one-shot ``categories`` iterable is consumed exactly once, so no
caller has to materialise it defensively before handing it on.  The resolved
codes carry their own registry order, which is a canonical cache key: two
selectors naming the same codes are one selection and hit one cache entry.  And
the highest requested degree — the cutoff the row-streaming counter runs at — is
computed here rather than rediscovered per call.

The type is deliberately private.  No consumer constructs a selection; they all
pass ``max_degree=``/``categories=`` keywords, and the public surface is frozen
by ``tests/test_architecture_guardrails.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pedigree_graph._registry import RELATIONSHIPS, categories_up_to_degree, select_categories

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class RelationshipSelection:
    """The registry codes one selector names, in registry order.

    Build these with :meth:`parse`; it is the only constructor that
    establishes the invariants below.

    Attributes:
        codes: The requested codes, for membership tests.
        ordered: The same codes in registry order — the canonical cache key.
            Two selectors naming the same codes produce an equal tuple
            regardless of the order or duplicates the caller wrote.
        top_degree: The highest degree among *codes*, or ``None`` when the
            selection is empty.  ``top_degree is None`` exactly when
            ``codes`` is empty, so it doubles as the empty test on the paths
            that need the cutoff.
    """

    codes: frozenset[str]
    ordered: tuple[str, ...]
    top_degree: int | None

    @classmethod
    def parse(cls, max_degree: int | None, categories: Iterable[str] | None) -> RelationshipSelection:
        """Validate the one-of-two selector and resolve the codes it names.

        Args:
            max_degree: Select every category at or below this degree.
            categories: Select these registry codes.  Consumed once, so a
                one-shot iterable is safe.

        Returns:
            The resolved selection; empty only for an empty *categories*.

        Raises:
            TypeError: Both selectors, neither, a bare ``str`` for
                *categories*, or a non-``str`` code.
            PedigreeValidationError: ``max_degree_out_of_range`` or
                ``unknown_relationship_category``.
        """
        if (max_degree is None) == (categories is None):
            raise TypeError("exactly one of max_degree= or categories= is required")
        if max_degree is not None:
            selected = categories_up_to_degree(max_degree)
        else:
            if isinstance(categories, str):
                raise TypeError("categories must be an iterable of codes, not a single str")
            assert categories is not None
            selected = select_categories(categories)
        # Both selectors return registry order already, so ordered needs no re-sort.
        ordered = tuple(category.code for category in selected)
        top = max((RELATIONSHIPS[code].degree for code in ordered), default=None)
        return cls(frozenset(ordered), ordered, top)
