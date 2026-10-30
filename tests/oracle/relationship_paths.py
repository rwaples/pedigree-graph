"""Relationship categories by explicit path enumeration, the specification oracle for classification.

Test code only (ADR 0007).  It states the ADR 0018 definitions pair by pair
over explicit ancestor path counts, from the id, mother, father and twin
columns alone, and imports nothing from the engine or the selection parser;
the registry is read as specification.

A *path* joins the pair through one shared ancestor.  Upward it follows
represented parent rows, each route counted once; a sib link counts one path
per shared known parent id, external ids included, so full sibs are joined
by two paths and half sibs by one.  A collateral category is *full* when at
least two paths of its shape join the pair, summed over both orientations,
and *half* when exactly one does.

* ``MZ``: the twin column.
* ``MO``/``FO``: each represented parent link, offspring first.
* Siblings by *original* parent id, external ids included, over rows with a
  known parent id, co-twins included: ``FS`` both ids known and equal,
  ``MHS``/``PHS`` the known mother/father id equal, minus ``FS``.
* Lineal ``GP`` .. ``G3GP``: an upward path of exactly k hops, descendant first.
* The avuncular family at depth k (``Av``/``HAv`` k=1, ``GAv``/``HGAv`` 2,
  ``GGAv``/``HGGAv`` 3, ``G3Av`` 4): each k-hop ancestor of the niece or
  nephew, times the parent ids it shares with the aunt or uncle, minus the
  subtract list.
* ``1C``/``H1C``: the 2-up by 2-up path product, never a pair sharing a
  known parent id.
* ``1C1R``/``H1C1R``/``1C2R``: the senior-up by junior-up path product, 2 by
  3 (or 4), junior cousin first, minus the subtract list.
* ``2C``: the 3-up path product is at least two and no grandparent is shared.
* A pair joined in both orientations of an asymmetric code takes the lower
  row first; each pair is then reported under its earliest registry code.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from oracle.pair_utils import sort_by_canonical_key
from pedigree_graph._registry import RELATIONSHIPS
from pedigree_graph._selection import RelationshipSelection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pedigree_graph import PedigreeGraph, PedigreeView

_PairArrays = tuple[np.ndarray, np.ndarray]

_PO = ("MO", "FO")
_SUBTRACT = {
    "Av": _PO,
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
#: The avuncular family: depth (hops from the niece up to the aunt's sib) -> (full code, half code).
_AVUNCULAR = {1: ("Av", "HAv"), 2: ("GAv", "HGAv"), 3: ("GGAv", "HGGAv"), 4: ("G3Av", None)}
_LINEAL = {"GP": 2, "GGP": 3, "GGGP": 4, "G3GP": 5}
#: Removed cousins: (junior's hops up, senior's hops up, predicate on the path count).
_REMOVED = {
    "1C1R": (3, 2, lambda paths: paths >= 2),
    "H1C1R": (3, 2, lambda paths: paths == 1),
    "1C2R": (4, 2, lambda paths: paths >= 2),
}


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

    children: list[list[int]] = [[] for _ in range(n)]
    for r in range(n):
        for p in parents[r]:
            children[p].append(r)

    def levels(r: int, step: list[list[int]]) -> list[Counter]:
        """``[k]``: the rows k hops from *r* along *step*, each with its number of distinct paths."""
        out = [Counter({r: 1})]
        for _ in range(5):
            nxt: Counter = Counter()
            for x, count in out[-1].items():
                for y in step[x]:
                    nxt[y] += count
            out.append(nxt)
        return out

    up = [levels(r, parents) for r in range(n)]
    down = [levels(r, children) for r in range(n)]
    by_mother: dict[int, list[int]] = {}
    by_father: dict[int, list[int]] = {}
    for r in range(n):
        if mother_id[r] != -1:
            by_mother.setdefault(mother_id[r], []).append(r)
        if father_id[r] != -1:
            by_father.setdefault(father_id[r], []).append(r)

    def shares_mother(a: int, b: int) -> bool:
        return a != b and mother_id[a] != -1 and mother_id[a] == mother_id[b]

    def shares_father(a: int, b: int) -> bool:
        return a != b and father_id[a] != -1 and father_id[a] == father_id[b]

    def full_sib(a: int, b: int) -> bool:
        return shares_mother(a, b) and shares_father(a, b)

    def sib_paths(a: int) -> Counter:
        """Each sib of *a*, with one path per parent id the two share."""
        out: Counter = Counter()
        for group in (by_mother.get(mother_id[a], ()), by_father.get(father_id[a], ())):
            out.update(j for j in group if j != a)
        return out

    def avuncular_paths(niece: int, depth: int) -> Counter:
        """Each aunt or uncle of *niece* at *depth*, with the paths through the niece's ancestors."""
        out: Counter = Counter()
        for a, count in up[niece][depth].items():
            for aunt, links in sib_paths(a).items():
                out[aunt] += count * links
        return out

    def cousin_paths(i: int, up_hops: int, down_hops: int) -> Counter:
        """Each row *down_hops* below an ancestor *up_hops* above *i*, with the paths through those ancestors."""
        out: Counter = Counter()
        for ancestor, count in up[i][up_hops].items():
            for j, below in down[ancestor][down_hops].items():
                out[j] += count * below
        return out

    def oriented(ordered: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """One entry per unordered pair: the lower row first when both orientations hold."""
        return {(a, b) for a, b in ordered if a != b and not (b < a and (b, a) in ordered)}

    def unordered(pairs: set[tuple[int, int]]) -> set[frozenset[int]]:
        return {frozenset(pair) for pair in pairs}

    def minus_closer(code: str, pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        drop = set().union(*(unordered(raw[c]) for c in _SUBTRACT[code]))
        return {pair for pair in oriented(pairs) if frozenset(pair) not in drop}

    raw: dict[str, set[tuple[int, int]]] = {}
    everyone = range(n)
    raw["MZ"] = {(r, row_of[int(twin[r])]) for r in everyone if twin[r] != -1 and r < row_of[int(twin[r])]}
    raw["MO"] = {(r, mother_row[r]) for r in everyone if mother_row[r] != -1}
    raw["FO"] = {(r, father_row[r]) for r in everyone if father_row[r] != -1}
    sibs = {(a, b) for a in everyone for b in sib_paths(a) if a < b}
    raw["FS"] = {(a, b) for a, b in sibs if full_sib(a, b)}
    raw["MHS"] = {(a, b) for a, b in sibs if shares_mother(a, b) and not full_sib(a, b)}
    raw["PHS"] = {(a, b) for a, b in sibs if shares_father(a, b) and not full_sib(a, b)}
    for code, hops in _LINEAL.items():
        raw[code] = {(r, ancestor) for r in everyone for ancestor in up[r][hops]}

    # Registry order computes every subtracted code before the codes that subtract it.
    for depth, (full, half) in _AVUNCULAR.items():
        forward = [avuncular_paths(i, depth) for i in everyone]
        found_full, found_half = set(), set()
        for i in everyone:
            for j, paths in forward[i].items():
                if i != j:
                    total = paths + forward[j][i]
                    (found_full if total >= 2 else found_half).add((i, j))
        raw[full] = minus_closer(full, found_full)
        if half is not None:
            raw[half] = minus_closer(half, found_half)

    raw["1C"], raw["H1C"] = set(), set()
    for a in everyone:
        for b, paths in cousin_paths(a, 2, 2).items():
            if a < b and not (shares_mother(a, b) or shares_father(a, b)):
                raw["1C" if paths >= 2 else "H1C"].add((a, b))
    for code, (junior_up, senior_up, keep) in _REMOVED.items():
        # forward[i][j]: paths with i the senior cousin, j the junior.
        forward = [cousin_paths(i, senior_up, junior_up) for i in everyone]
        found_removed = {
            (j, i) for i in everyone for j, paths in forward[i].items() if i != j and keep(paths + forward[j][i])
        }
        raw[code] = minus_closer(code, found_removed)
    raw["2C"] = set()
    for a in everyone:
        shares_grandparent = cousin_paths(a, 2, 2)
        for b, paths in cousin_paths(a, 3, 3).items():
            if a < b and paths >= 2 and b not in shares_grandparent:
                raw["2C"].add((a, b))

    claimed: set[frozenset[int]] = set()
    folded: dict[str, set[tuple[int, int]]] = {}
    for code, category in RELATIONSHIPS.items():
        if category.degree > max_degree:
            continue
        folded[code] = {pair for pair in raw[code] if frozenset(pair) not in claimed}
        claimed |= unordered(folded[code])
    return folded


#: The degree-5 result per graph, keyed by its columns: every selector is a
#: filter of it, since a code is claimed only by earlier registry codes.
_CACHE: dict[bytes, dict[str, set[tuple[int, int]]]] = {}


def _graph_paths(graph: PedigreeGraph) -> dict[str, set[tuple[int, int]]]:
    ids = np.asarray(graph.ids, dtype=np.int64)
    mother = np.asarray(graph.mother_ids, dtype=np.int64)
    father = np.asarray(graph.father_ids, dtype=np.int64)
    twin_rows = np.asarray(graph.twin_rows)
    twin = np.where(twin_rows >= 0, ids[np.maximum(twin_rows, 0)], -1)
    key = b"".join(column.tobytes() for column in (ids, mother, father, twin))
    if key not in _CACHE:
        _CACHE[key] = relationship_paths(ids, mother, father, twin, max_degree=5)
    return _CACHE[key]


def _arrays(pairs: set[tuple[int, int]], n: int) -> _PairArrays:
    block = np.array(sorted(pairs), dtype=np.intp).reshape(-1, 2)
    return sort_by_canonical_key(block[:, 0], block[:, 1], n)


def oracle_pairs(
    graph: PedigreeGraph, *, max_degree: int | None = None, categories: Iterable[str] | None = None
) -> dict[str, _PairArrays]:
    """``{code: (first, second)}`` intp graph rows for the selector, every code present, unrequested empty."""
    codes = RelationshipSelection.parse(max_degree, categories).codes
    paths = _graph_paths(graph)
    n = graph.n_individuals
    return {code: _arrays(paths[code] if code in codes else set(), n) for code in RELATIONSHIPS}


def oracle_view_pairs(
    view: PedigreeView, *, max_degree: int | None = None, categories: Iterable[str] | None = None
) -> dict[str, _PairArrays]:
    """As :func:`oracle_pairs`, in view rows: both endpoints selected, symmetric re-canonicalised, view-key sorted."""
    codes = RelationshipSelection.parse(max_degree, categories).codes
    graph_to_view = view._graph_to_view()
    paths = _graph_paths(view._graph)
    n = len(view)
    out = {}
    for code in RELATIONSHIPS:
        kept = set()
        if code in codes:
            for first, second in paths[code]:
                a, b = int(graph_to_view[first]), int(graph_to_view[second])
                if a >= 0 and b >= 0:
                    kept.add((min(a, b), max(a, b)) if RELATIONSHIPS[code].symmetric else (a, b))
        out[code] = _arrays(kept, n)
    return out
