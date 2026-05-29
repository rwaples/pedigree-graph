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
    "REL_PLAN",
    "REL_REGISTRY",
    "EngineSupport",
    "RelType",
    "bfs_divergent_codes",
    "streaming_approximate_codes",
    "streaming_exact_codes",
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


# ---------------------------------------------------------------------------
# Engine plan: how each engine handles a code, beyond the structural RelType
# ---------------------------------------------------------------------------


class EngineSupport(NamedTuple):
    """Per-code engine handling, beyond the structural :class:`RelType`.

    The matrix engine (``count_pairs`` / ``extract_pairs``) is the
    reference: it counts *paths* through shared ancestors and is exact for
    every code on every input.  This record captures where the other two
    engines deviate, so the divergence lives in one place instead of being
    re-stated in three docstrings (PGQ-004).
    """

    streaming_exact: bool
    """``count_pairs_streaming`` is bit-identical to the matrix engine for
    this code.  ``False`` → the scalar formula is approximate (it assumes a
    full complement of known ancestors and diverges on shallow / inbred /
    twin-having pedigrees; see :meth:`count_pairs_streaming`)."""

    bfs_diverges_under_inbreeding: bool
    """``count_pairs_bfs`` counts *distinct* shared ancestors while the
    matrix engine counts *paths*; the two differ for this code on inbred
    input.  ``False`` → BFS matches the matrix engine on every input."""


# Keyed by relationship code; must cover exactly the REL_REGISTRY key set
# (asserted in tests).  Matrix engine is the exact paths-counting reference.
REL_PLAN: dict[str, EngineSupport] = {
    # --- degree 0 / 1: lineal + sibling, exact everywhere ---
    "MZ": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "MO": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "FO": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "FS": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "MHS": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "PHS": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    # --- degree 2 ---
    "GP": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "Av": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    # --- degree 3 ---
    "GGP": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "HAv": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "GAv": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "1C": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    # --- degree 4 ---
    "GGGP": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "HGAv": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "GGAv": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "H1C": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "1C1R": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=True),
    # --- degree 5 ---
    "G3GP": EngineSupport(streaming_exact=True, bfs_diverges_under_inbreeding=False),
    "HGGAv": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "G3Av": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=False),
    "H1C1R": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=True),
    "1C2R": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=True),
    "2C": EngineSupport(streaming_exact=False, bfs_diverges_under_inbreeding=True),
}


def streaming_exact_codes() -> frozenset[str]:
    """Codes for which ``count_pairs_streaming`` matches the matrix engine exactly."""
    return frozenset(code for code, plan in REL_PLAN.items() if plan.streaming_exact)


def streaming_approximate_codes() -> frozenset[str]:
    """Codes for which ``count_pairs_streaming`` is an approximation."""
    return frozenset(code for code, plan in REL_PLAN.items() if not plan.streaming_exact)


def bfs_divergent_codes() -> frozenset[str]:
    """Codes where ``count_pairs_bfs`` can diverge from the matrix engine.

    BFS counts distinct ancestors and the matrix engine counts paths, so
    these codes differ on inbred input.
    """
    return frozenset(code for code, plan in REL_PLAN.items() if plan.bfs_diverges_under_inbreeding)
