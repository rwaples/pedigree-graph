"""Relationship categories by explicit path enumeration, the specification oracle for classification.

Test code only (ADR 0007).  ``tests/oracle/relationship_pairs.py`` is the 0.8
engine moved verbatim and changed in lockstep with the Rust engine (issue
#29), so a definition defect shared by both passes their parity.  This oracle
states the ADR 0010 definitions (the 0.7.1 ones, which that copy is the record
of) pair by pair over explicit ancestor path counts, from the id, mother,
father and twin columns alone.  It imports nothing from the engine, the copied
oracle, or the selection parser; the registry is read as specification.

The rules, each with the line of ``relationship_pairs.py`` that defines it:

* ``MZ``: the twin column (``_mz_twin_pairs``, :327).
* ``MO``/``FO``: each represented parent link, offspring first (:335).
* Siblings by *original* parent id, external ids included, over rows with a
  known parent id, co-twins included (:384): ``FS`` both ids known and equal
  (:398), ``MHS``/``PHS`` the known mother/father id equal, minus ``FS``
  (:411, :421).
* Lineal ``GP`` .. ``G3GP``: an upward path of exactly k hops, descendant first (:493).
* ``Av``: a represented parent of the first is a full sib of the second, not
  self and not parent-offspring either way (:596).
* ``HAv`` .. ``G3Av``: an ancestor of the first at exactly ``down - 1`` hops
  is a full (or half) sib of the second, minus the subtract list (:501, :778-913).
* ``1C``/``H1C``: two or more / exactly one *distinct* shared grandparent,
  never a pair sharing a known parent id (:523).
* ``1C1R``/``H1C1R``/``1C2R``: the 2-up by 3-up (or 4-up) path product is
  at least two / exactly one / at least two, junior cousin first, minus the
  subtract list (:800, :848, :871).
* ``2C``: the 3-up path product is at least two and no grandparent path is
  shared (:618).
* An asymmetric pair valid both ways takes the lower row first (:101); each
  pair is then reported under its earliest registry code (:929).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pedigree_graph._registry import RELATIONSHIPS

if TYPE_CHECKING:
    import numpy as np

_PO = ("MO", "FO")
_SUBTRACT = {
    "HAv": (*_PO, "GP"),
    "GAv": (*_PO, "GP", "Av"),
    "HGAv": (*_PO, "GP", "GGP", "HAv"),
    "GGAv": (*_PO, "GP", "GGP", "Av", "GAv"),
    "HGGAv": (*_PO, "GP", "GGP", "GGGP", "HAv", "HGAv"),
    "G3Av": (*_PO, "GP", "GGP", "GGGP", "Av", "GAv", "GGAv"),
    "1C1R": (*_PO, "GP", "GGP", "Av", "GAv", "FS", "MHS", "PHS", "1C"),
    "H1C1R": (*_PO, "GP", "GGP", "GGGP", "HAv", "HGAv", "FS", "MHS", "PHS", "1C", "H1C", "1C1R"),
    "1C2R": (*_PO, "GP", "GGP", "GGGP", "Av", "GAv", "GGAv", "FS", "MHS", "PHS", "1C", "H1C", "1C1R"),
}
#: Collateral codes: (sib kind, hops from the first member down to the sib link's near end + 1).
_COLLATERAL = {
    "HAv": ("half", 2),
    "GAv": ("full", 3),
    "HGAv": ("half", 3),
    "GGAv": ("full", 4),
    "HGGAv": ("half", 4),
    "G3Av": ("full", 5),
}
_LINEAL = {"GP": 2, "GGP": 3, "GGGP": 4, "G3GP": 5}
#: Removed cousins: (junior's hops up, senior's hops up, predicate on the path product).
_REMOVED = {
    "1C1R": (3, 2, lambda paths: paths >= 2),
    "H1C1R": (3, 2, lambda paths: paths == 1),
    "1C2R": (4, 2, lambda paths: paths >= 2),
}


def _product(up_a: Counter, up_b: Counter) -> int:
    return sum(count * up_b[ancestor] for ancestor, count in up_a.items() if ancestor in up_b)


def relationship_paths(
    ids: np.ndarray, mother: np.ndarray, father: np.ndarray, twin: np.ndarray, *, max_degree: int = 5
) -> dict[str, set[tuple[int, int]]]:
    """Closest-category pairs of input rows, every registry code up to *max_degree*.

    Args:
        ids: Individual ids, one per row, any order.
        mother: Mother id per row, ``-1`` when missing; an id that is no row is external.
        father: Father id per row, as ``mother``.
        twin: MZ co-twin id per row, ``-1`` when none.
        max_degree: Highest registry degree reported.

    Returns:
        ``{code: {(first, second), ...}}`` in input rows, the category's
        orientation, symmetric codes ``first < second``.
    """
    n = len(ids)
    row_of = {int(v): r for r, v in enumerate(ids)}
    mother_id = [int(v) for v in mother]
    father_id = [int(v) for v in father]
    mother_row = [row_of.get(v, -1) if v != -1 else -1 for v in mother_id]
    father_row = [row_of.get(v, -1) if v != -1 else -1 for v in father_id]
    parents = [[p for p in (mother_row[r], father_row[r]) if p != -1] for r in range(n)]

    # up[r][k]: k-hop ancestors of r, each with its number of distinct paths.
    up = []
    for r in range(n):
        levels = [Counter({r: 1})]
        for _ in range(5):
            nxt: Counter = Counter()
            for ancestor, count in levels[-1].items():
                for p in parents[ancestor]:
                    nxt[p] += count
            levels.append(nxt)
        up.append(levels)

    def full_sib(a: int, b: int) -> bool:
        known = -1 not in (mother_id[a], father_id[a], mother_id[b], father_id[b])
        return a != b and known and mother_id[a] == mother_id[b] and father_id[a] == father_id[b]

    def shares_mother(a: int, b: int) -> bool:
        return a != b and mother_id[a] != -1 and mother_id[a] == mother_id[b]

    def shares_father(a: int, b: int) -> bool:
        return a != b and father_id[a] != -1 and father_id[a] == father_id[b]

    def half_sib(a: int, b: int) -> bool:
        return (shares_mother(a, b) or shares_father(a, b)) and not full_sib(a, b)

    def oriented(ordered: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """One entry per unordered pair: the lower row first when both orientations hold."""
        return {(a, b) for a, b in ordered if a != b and not (b < a and (b, a) in ordered)}

    def unordered(pairs: set[tuple[int, int]]) -> set[frozenset[int]]:
        return {frozenset(pair) for pair in pairs}

    raw: dict[str, set[tuple[int, int]]] = {}
    everyone = range(n)
    raw["MZ"] = {(r, row_of[int(twin[r])]) for r in everyone if twin[r] != -1 and r < row_of[int(twin[r])]}
    raw["MO"] = {(r, mother_row[r]) for r in everyone if mother_row[r] != -1}
    raw["FO"] = {(r, father_row[r]) for r in everyone if father_row[r] != -1}
    pairs_lo_hi = [(a, b) for a in everyone for b in everyone if a < b]
    raw["FS"] = {(a, b) for a, b in pairs_lo_hi if full_sib(a, b)}
    raw["MHS"] = {(a, b) for a, b in pairs_lo_hi if shares_mother(a, b) and not full_sib(a, b)}
    raw["PHS"] = {(a, b) for a, b in pairs_lo_hi if shares_father(a, b) and not full_sib(a, b)}
    for code, hops in _LINEAL.items():
        raw[code] = {(r, ancestor) for r in everyone for ancestor in up[r][hops]}
    parent_child = unordered(raw["MO"] | raw["FO"])
    raw["Av"] = oriented(
        {
            (c, u)
            for c in everyone
            for u in everyone
            if c != u and frozenset((c, u)) not in parent_child and any(full_sib(p, u) for p in parents[c])
        }
    )
    for code, (kind, down) in _COLLATERAL.items():
        sib = full_sib if kind == "full" else half_sib
        found = {(i, j) for i in everyone for j in everyone if any(sib(a, j) for a in up[i][down - 1])}
        drop = set().union(*(unordered(raw[c]) for c in _SUBTRACT[code]))
        raw[code] = {pair for pair in oriented(found) if frozenset(pair) not in drop}

    raw["1C"], raw["H1C"] = set(), set()
    for a, b in pairs_lo_hi:
        if shares_mother(a, b) or shares_father(a, b):
            continue
        shared = len(up[a][2].keys() & up[b][2].keys())
        if shared >= 2:
            raw["1C"].add((a, b))
        elif shared == 1:
            raw["H1C"].add((a, b))
    for code, (junior_up, senior_up, keep) in _REMOVED.items():
        found = {
            (j, i) for i in everyone for j in everyone if i != j and keep(_product(up[i][senior_up], up[j][junior_up]))
        }
        drop = set().union(*(unordered(raw[c]) for c in _SUBTRACT[code]))
        raw[code] = {pair for pair in oriented(found) if frozenset(pair) not in drop}
    raw["2C"] = {
        (a, b) for a, b in pairs_lo_hi if _product(up[a][3], up[b][3]) >= 2 and _product(up[a][2], up[b][2]) == 0
    }

    claimed: set[frozenset[int]] = set()
    folded: dict[str, set[tuple[int, int]]] = {}
    for code, category in RELATIONSHIPS.items():
        if category.degree > max_degree:
            continue
        folded[code] = {pair for pair in raw[code] if frozenset(pair) not in claimed}
        claimed |= unordered(folded[code])
    return folded
