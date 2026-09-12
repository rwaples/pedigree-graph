"""Relationship-category registry: the shared vocabulary of pair codes.

:data:`RELATIONSHIPS` is the canonical mapping of the 23 relationship codes to
immutable :class:`RelationshipCategory` records.  Iteration order is
kinship-descending and degree-ascending, and within one degree it is the
documented precedence for closest-category classification: when a pair matches
several categories of the same degree, the one appearing first in this order
wins.

Orientation: ``first`` is the pair member with at least as many meioses to the
shared ancestor(s), ``up`` counts meioses from ``first`` up to the ancestor(s),
and ``down`` counts meioses from the ancestor(s) down to ``second``.  So
``up >= down`` holds for every category, with equality exactly for the seven
symmetric ones.

Imported by ``_core`` (PedigreeGraph), both pair engines, and the public
:mod:`pedigree_graph.relationships` facade, so the codes, kinship coefficients,
and degree range have a single source of truth.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

from pedigree_graph._errors import PedigreeValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "RELATIONSHIPS",
    "REL_PLAN",
    "EngineSupport",
    "RelationshipCategory",
    "RelationshipRole",
    "categories_up_to_degree",
    "estimate_exact_codes",
    "select_categories",
]


RelationshipRole = Literal[
    "offspring",
    "mother",
    "father",
    "descendant",
    "ancestor",
    "niece_nephew",
    "aunt_uncle",
    "junior_cousin",
    "senior_cousin",
]
"""The closed set of positional roles an asymmetric category can assign."""


@dataclass(frozen=True, slots=True)
class RelationshipCategory:
    """One relationship category: its code, its path shape, and its two roles.

    ``first`` is the pair member with at least as many meioses to the shared
    ancestor(s); ``up`` counts meioses from ``first`` up to the ancestor(s) and
    ``down`` counts them from the ancestor(s) down to ``second``.  ``up >= down``
    therefore holds for every category, and the category is symmetric exactly
    when ``up == down``.  For a cousin category, "junior" means generationally
    further from the shared ancestors, not younger by birth year.

    Attributes:
        code: Short registry key, e.g. ``"FS"``.
        label: Human-readable display label.
        degree: Kinship distance: 0 for MZ twins, 1 for parent-offspring and
            full sibs, and so on.
        nominal_kinship: Kinship coefficient of a single connecting path
            through the shared ancestor(s), with no inbreeding.
        up: Meioses from ``first`` up to the shared ancestor(s).
        down: Meioses from the shared ancestor(s) down to ``second``.
        ancestor_count: 1 for half / lineal, 2 for a mated pair, 0 for MZ.
        first_role: Role of the first pair member, ``None`` if symmetric.
        second_role: Role of the second pair member, ``None`` if symmetric.
    """

    code: str
    label: str
    degree: int
    nominal_kinship: float
    up: int
    down: int
    ancestor_count: int
    first_role: RelationshipRole | None
    second_role: RelationshipRole | None

    @property
    def symmetric(self) -> bool:
        """Whether the two positions carry no distinct biological roles."""
        return self.first_role is None


_RELATIONSHIPS: dict[str, RelationshipCategory] = {
    category.code: category
    for category in (
        # --- degree 0 (kinship 1/2) ---
        RelationshipCategory("MZ", "MZ twin", 0, 0.5, 0, 0, 0, None, None),
        # --- degree 1 (kinship 1/4) ---
        RelationshipCategory("MO", "Mother-offspring", 1, 0.25, 1, 0, 1, "offspring", "mother"),
        RelationshipCategory("FO", "Father-offspring", 1, 0.25, 1, 0, 1, "offspring", "father"),
        RelationshipCategory("FS", "Full sib", 1, 0.25, 1, 1, 2, None, None),
        # --- degree 2 (kinship 1/8) ---
        RelationshipCategory("MHS", "Maternal half sib", 2, 0.125, 1, 1, 1, None, None),
        RelationshipCategory("PHS", "Paternal half sib", 2, 0.125, 1, 1, 1, None, None),
        RelationshipCategory("GP", "Grandparent", 2, 0.125, 2, 0, 1, "descendant", "ancestor"),
        RelationshipCategory("Av", "Avuncular", 2, 0.125, 2, 1, 2, "niece_nephew", "aunt_uncle"),
        # --- degree 3 (kinship 1/16) ---
        RelationshipCategory("GGP", "Great-grandparent", 3, 0.0625, 3, 0, 1, "descendant", "ancestor"),
        RelationshipCategory("HAv", "Half-avuncular", 3, 0.0625, 2, 1, 1, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("GAv", "Great-avuncular", 3, 0.0625, 3, 1, 2, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("1C", "1st cousin", 3, 0.0625, 2, 2, 2, None, None),
        # --- degree 4 (kinship 1/32) ---
        RelationshipCategory("GGGP", "Great²-grandparent", 4, 0.03125, 4, 0, 1, "descendant", "ancestor"),
        RelationshipCategory("HGAv", "Half-great-avuncular", 4, 0.03125, 3, 1, 1, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("GGAv", "Great²-avuncular", 4, 0.03125, 4, 1, 2, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("H1C", "Half-1st-cousin", 4, 0.03125, 2, 2, 1, None, None),
        RelationshipCategory("1C1R", "1st cousin 1R", 4, 0.03125, 3, 2, 2, "junior_cousin", "senior_cousin"),
        # --- degree 5 (kinship 1/64) ---
        RelationshipCategory("G3GP", "Great³-grandparent", 5, 0.015625, 5, 0, 1, "descendant", "ancestor"),
        RelationshipCategory("HGGAv", "Half-great²-avuncular", 5, 0.015625, 4, 1, 1, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("G3Av", "Great³-avuncular", 5, 0.015625, 5, 1, 2, "niece_nephew", "aunt_uncle"),
        RelationshipCategory("H1C1R", "Half-1st-cousin 1R", 5, 0.015625, 3, 2, 1, "junior_cousin", "senior_cousin"),
        RelationshipCategory("1C2R", "1st cousin 2R", 5, 0.015625, 4, 2, 2, "junior_cousin", "senior_cousin"),
        RelationshipCategory("2C", "2nd cousin", 5, 0.015625, 3, 3, 2, None, None),
    )
}

RELATIONSHIPS: Mapping[str, RelationshipCategory] = MappingProxyType(_RELATIONSHIPS)

# Valid ``max_degree`` range for the public pair APIs.  Degree 0 = MZ only
# (still a useful query — twins-only counts); degree 5 = full registry.
_MAX_DEGREE_MIN = 0
_MAX_DEGREE_MAX = 5


def _validate_max_degree(max_degree: int) -> int:
    """Return the integer value of *max_degree*, rejecting coercion and bools."""
    if isinstance(max_degree, bool):
        raise TypeError("max_degree must be an integer, not bool")
    md = operator.index(max_degree)
    if md < _MAX_DEGREE_MIN or md > _MAX_DEGREE_MAX:
        raise PedigreeValidationError(
            "max_degree_out_of_range",
            f"max_degree must be in [{_MAX_DEGREE_MIN}, {_MAX_DEGREE_MAX}], got {max_degree!r}",
            value=max_degree,
            minimum=_MAX_DEGREE_MIN,
            maximum=_MAX_DEGREE_MAX,
        )
    return md


def categories_up_to_degree(max_degree: int) -> tuple[RelationshipCategory, ...]:
    """Select every category at or below *max_degree*.

    Args:
        max_degree: Degree cutoff, validated against ``[0, 5]``.

    Returns:
        The matching categories in registry order.

    Raises:
        TypeError: *max_degree* is not an integer or is a boolean.
        PedigreeValidationError: *max_degree* is outside ``[0, 5]``
            (code ``max_degree_out_of_range``).
    """
    cutoff = _validate_max_degree(max_degree)
    return tuple(category for category in _RELATIONSHIPS.values() if category.degree <= cutoff)


def select_categories(codes: Iterable[str]) -> tuple[RelationshipCategory, ...]:
    """Select the categories named by *codes*.

    Args:
        codes: Relationship codes in any order; duplicates are ignored.

    Returns:
        The named categories in registry order, not in the order given.

    Raises:
        PedigreeValidationError: One or more codes are not registry codes
            (code ``unknown_relationship_category``, with the offending codes
            in ``fields["codes"]`` sorted lexically).
        TypeError: A code is not a string.
    """
    requested = set()
    for code in codes:
        if not isinstance(code, str):
            raise TypeError(f"relationship code must be str, got {type(code).__name__}")
        requested.add(code)
    unknown = requested - _RELATIONSHIPS.keys()
    if unknown:
        raise PedigreeValidationError(
            "unknown_relationship_category",
            f"unknown relationship category code(s): {', '.join(sorted(unknown))}",
            codes=tuple(sorted(unknown)),
        )
    return tuple(category for code, category in _RELATIONSHIPS.items() if code in requested)


# ---------------------------------------------------------------------------
# Engine plan: how each engine handles a code, beyond the structural category
# ---------------------------------------------------------------------------


class EngineSupport(NamedTuple):
    """Per-code engine handling, beyond the structural :class:`RelationshipCategory`.

    ``relationship_counts`` and ``relationship_pairs`` apply closest-category
    precedence for every code. This record names the six codes also available
    through scalar ``close_relative_counts``.
    """

    estimate_exact: bool
    """Whether ``close_relative_counts`` computes this code, exactly.

    The name is retained from the former estimator. False means the scalar
    method returns None for this code, not an approximation. See ADR 0011.
    """


# Keyed by relationship code; covers exactly the RELATIONSHIPS key set (asserted
# below and in tests). All public counts use closest-category precedence.
REL_PLAN: dict[str, EngineSupport] = {
    # --- the six scalar close-relative codes, degrees 0 through 2 ---
    "MZ": EngineSupport(estimate_exact=True),
    "MO": EngineSupport(estimate_exact=True),
    "FO": EngineSupport(estimate_exact=True),
    "FS": EngineSupport(estimate_exact=True),
    "MHS": EngineSupport(estimate_exact=True),
    "PHS": EngineSupport(estimate_exact=True),
    # --- degree 2 ---
    "GP": EngineSupport(estimate_exact=False),
    "Av": EngineSupport(estimate_exact=False),
    # --- degree 3 ---
    "GGP": EngineSupport(estimate_exact=False),
    "HAv": EngineSupport(estimate_exact=False),
    "GAv": EngineSupport(estimate_exact=False),
    "1C": EngineSupport(estimate_exact=False),
    # --- degree 4 ---
    "GGGP": EngineSupport(estimate_exact=False),
    "HGAv": EngineSupport(estimate_exact=False),
    "GGAv": EngineSupport(estimate_exact=False),
    "H1C": EngineSupport(estimate_exact=False),
    "1C1R": EngineSupport(estimate_exact=False),
    # --- degree 5 ---
    "G3GP": EngineSupport(estimate_exact=False),
    "HGGAv": EngineSupport(estimate_exact=False),
    "G3Av": EngineSupport(estimate_exact=False),
    "H1C1R": EngineSupport(estimate_exact=False),
    "1C2R": EngineSupport(estimate_exact=False),
    "2C": EngineSupport(estimate_exact=False),
}

assert REL_PLAN.keys() == _RELATIONSHIPS.keys(), "REL_PLAN and RELATIONSHIPS cover different codes"


def estimate_exact_codes() -> frozenset[str]:
    """The six exact codes returned by ``close_relative_counts``; legacy helper name."""
    return frozenset(code for code, plan in REL_PLAN.items() if plan.estimate_exact)
