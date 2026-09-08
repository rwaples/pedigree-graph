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

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np  # 0.8.0-DELETE

from pedigree_graph._errors import PedigreeValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "PAIR_KINSHIP",
    "RELATIONSHIPS",
    "REL_PLAN",
    "REL_REGISTRY",
    "EngineSupport",
    "RelType",
    "RelationshipCategory",
    "RelationshipRole",
    "bfs_divergent_codes",
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
    """Coerce *max_degree* to int and reject values outside ``[0, 5]``."""
    md = int(max_degree)
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

    The matrix engine (``relationship_counts`` / ``relationship_pairs``) is the
    reference: it counts *paths* through shared ancestors and is exact for
    every code on every input.  This record captures where the other two
    engines deviate, so the divergence lives in one place instead of being
    re-stated in three docstrings (PGQ-004).
    """

    estimate_exact: bool
    """``estimate_relationship_counts`` equals ``relationship_counts`` for
    this code on every input (ADR 0011): the MZ, parent-offspring, and
    sibling codes.  ``False`` → the code is reported in the result's
    ``approximate`` set."""

    bfs_diverges_under_inbreeding: bool
    """``count_pairs_bfs`` counts *distinct* shared ancestors while the
    matrix engine counts *paths*; the two differ for this code on inbred
    input.  ``False`` → BFS matches the matrix engine's unfolded blocks on
    every input."""


# Keyed by relationship code; covers exactly the RELATIONSHIPS key set (asserted
# below and in tests).  Matrix engine is the exact paths-counting reference.
REL_PLAN: dict[str, EngineSupport] = {
    # --- degree 0 / 1: lineal + sibling, exact everywhere ---
    "MZ": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    "MO": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    "FO": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    "FS": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    "MHS": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    "PHS": EngineSupport(estimate_exact=True, bfs_diverges_under_inbreeding=False),
    # --- degree 2 ---
    "GP": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "Av": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    # --- degree 3 ---
    "GGP": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "HAv": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "GAv": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "1C": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    # --- degree 4 ---
    "GGGP": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "HGAv": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "GGAv": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "H1C": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "1C1R": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=True),
    # --- degree 5 ---
    "G3GP": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "HGGAv": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "G3Av": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=False),
    "H1C1R": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=True),
    "1C2R": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=True),
    "2C": EngineSupport(estimate_exact=False, bfs_diverges_under_inbreeding=True),
}

assert REL_PLAN.keys() == _RELATIONSHIPS.keys(), "REL_PLAN and RELATIONSHIPS cover different codes"


def estimate_exact_codes() -> frozenset[str]:
    """Codes for which ``estimate_relationship_counts`` equals ``relationship_counts``."""
    return frozenset(code for code, plan in REL_PLAN.items() if plan.estimate_exact)


def bfs_divergent_codes() -> frozenset[str]:
    """Codes where ``count_pairs_bfs`` can diverge from the matrix engine.

    BFS counts distinct ancestors and the matrix engine counts paths, so
    these codes differ on inbred input.
    """
    return frozenset(code for code, plan in REL_PLAN.items() if plan.bfs_diverges_under_inbreeding)


# ---------------------------------------------------------------------------
# 0.7.1 compatibility: detached snapshots, built once from RELATIONSHIPS
# ---------------------------------------------------------------------------


class RelType(NamedTuple):  # 0.8.0-DELETE
    """Relationship category defined by path through pedigree."""

    up: int  # meioses A → common ancestor(s)
    down: int  # meioses common ancestor(s) → B
    n_anc: int  # 1 = half/lineal, 2 = full (mated-pair ancestors)
    code: str  # short dict key
    label: str  # human-readable display label

    @property
    def kinship(self) -> float:
        """Kinship coefficient derived from path length and ancestor count."""
        if self.code == "MZ":
            return 0.5
        return self.n_anc * 0.5 ** (self.up + self.down + 1)

    @property
    def degree(self) -> int:
        """Kinship degree (0 for MZ, 1 for parent-offspring/full-sib, etc.)."""
        if self.code == "MZ":
            return 0
        return round(-1 - np.log2(self.kinship))


def _as_rel_type(category: RelationshipCategory) -> RelType:  # 0.8.0-DELETE
    """Restore the 0.7.1 orientation, which stored collateral categories up ≤ down."""
    up, down = (category.up, 0) if category.down == 0 else (category.down, category.up)
    return RelType(up, down, category.ancestor_count, category.code, category.label)


REL_REGISTRY: dict[str, RelType] = {code: _as_rel_type(c) for code, c in _RELATIONSHIPS.items()}  # 0.8.0-DELETE

PAIR_KINSHIP: dict[str, float] = {code: c.nominal_kinship for code, c in _RELATIONSHIPS.items()}  # 0.8.0-DELETE
