"""Relationship-category registry: the shared vocabulary of pair codes.

Every relationship category is parameterised by ``(up, down, n_ancestors)``:
  - up:   meioses from individual A up to common ancestor(s), canonicalised up ≤ down
  - down: meioses from common ancestor(s) down to individual B
  - n_ancestors: 1 (half / lineal) or 2 (full, i.e. mated pair)
  - kinship = n_ancestors × (1/2)^(up + down + 1)

Imported by ``_core`` (PedigreeGraph), both pair engines, and the package
``__init__`` so the codes, kinship coefficients, and degree range have a
single source of truth.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "PAIR_KINSHIP",
    "REL_REGISTRY",
    "RelType",
]


class RelType(NamedTuple):
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


# Ordered registry: kinship-descending, degree-ascending.
# MZ twins are a special case (up=down=n_anc=0).
REL_REGISTRY: dict[str, RelType] = {}
for _rt in [
    # --- special ---
    RelType(0, 0, 0, "MZ", "MZ twin"),
    # --- degree 1 (kinship 1/4) ---
    RelType(1, 0, 1, "MO", "Mother-offspring"),
    RelType(1, 0, 1, "FO", "Father-offspring"),
    RelType(1, 1, 2, "FS", "Full sib"),
    # --- degree 2 (kinship 1/8) ---
    RelType(1, 1, 1, "MHS", "Maternal half sib"),
    RelType(1, 1, 1, "PHS", "Paternal half sib"),
    RelType(2, 0, 1, "GP", "Grandparent"),
    RelType(1, 2, 2, "Av", "Avuncular"),
    # --- degree 3 (kinship 1/16) ---
    RelType(3, 0, 1, "GGP", "Great-grandparent"),
    RelType(1, 2, 1, "HAv", "Half-avuncular"),
    RelType(1, 3, 2, "GAv", "Great-avuncular"),
    RelType(2, 2, 2, "1C", "1st cousin"),
    # --- degree 4 (kinship 1/32) ---
    RelType(4, 0, 1, "GGGP", "Great²-grandparent"),
    RelType(1, 3, 1, "HGAv", "Half-great-avuncular"),
    RelType(1, 4, 2, "GGAv", "Great²-avuncular"),
    RelType(2, 2, 1, "H1C", "Half-1st-cousin"),
    RelType(2, 3, 2, "1C1R", "1st cousin 1R"),
    # --- degree 5 (kinship 1/64) ---
    RelType(5, 0, 1, "G3GP", "Great³-grandparent"),
    RelType(1, 4, 1, "HGGAv", "Half-great²-avuncular"),
    RelType(1, 5, 2, "G3Av", "Great³-avuncular"),
    RelType(2, 3, 1, "H1C1R", "Half-1st-cousin 1R"),
    RelType(2, 4, 2, "1C2R", "1st cousin 2R"),
    RelType(3, 3, 2, "2C", "2nd cousin"),
]:
    REL_REGISTRY[_rt.code] = _rt

# Kinship lookup by code — single source of truth for all consumers
PAIR_KINSHIP: dict[str, float] = {rt.code: rt.kinship for rt in REL_REGISTRY.values()}

# Valid ``max_degree`` range for the public pair APIs.  Degree 0 = MZ only
# (still a useful query — twins-only counts); degree 5 = full registry.
_MAX_DEGREE_MIN = 0
_MAX_DEGREE_MAX = 5


def _validate_max_degree(max_degree: int) -> int:
    """Coerce *max_degree* to int and reject values outside ``[0, 5]``."""
    md = int(max_degree)
    if md < _MAX_DEGREE_MIN or md > _MAX_DEGREE_MAX:
        raise ValueError(
            f"max_degree must be in [{_MAX_DEGREE_MIN}, {_MAX_DEGREE_MAX}], "
            f"got {max_degree!r}",
        )
    return md
