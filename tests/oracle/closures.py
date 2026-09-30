"""Lineage counts and components from set closures, walk enumeration and union-find.

Test code only (ADR 0007).  ``tests/oracle/lineage.py`` is the 0.9.3 kernel
verbatim and the old component oracle called the same SciPy routine as
production; this one shares neither.  It reads only the id and parent
columns: a parent id that is no row is external and contributes no edge, and
an MZ twin link is not an edge.  Walks are enumerated one by one, so
:func:`descendant_walks` is for small pedigrees only; the path count of an
inbred lineage grows exponentially with its depth.
"""

from __future__ import annotations


def _edges(columns: dict) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]]]:
    ids = [int(v) for v in columns["id"]]
    known = set(ids)
    parents = {
        i: [int(p) for p in (m, f) if int(p) in known]
        for i, m, f in zip(ids, columns["mother"].tolist(), columns["father"].tolist(), strict=True)
    }
    children: dict[int, list[int]] = {i: [] for i in ids}
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    return ids, parents, children


def _closure(start: int, step: dict[int, list[int]]) -> set[int]:
    seen: set[int] = set()
    stack = list(step[start])
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(step[v])
    return seen


def distinct_ancestors(columns: dict) -> dict[int, int]:
    """Distinct strict represented ancestors per id."""
    ids, parents, _ = _edges(columns)
    return {i: len(_closure(i, parents)) for i in ids}


def distinct_descendants(columns: dict) -> dict[int, int]:
    """Distinct strict descendants per id."""
    ids, _, children = _edges(columns)
    return {i: len(_closure(i, children)) for i in ids}


def descendant_walks(columns: dict) -> dict[int, int]:
    """Downward walks from each id, one counted per walk."""
    ids, _, children = _edges(columns)

    def walks(start: int) -> int:
        count, stack = 0, list(children[start])
        while stack:
            count += 1
            stack.extend(children[stack.pop()])
        return count

    return {i: walks(i) for i in ids}


def component_minimum(columns: dict) -> dict[int, int]:
    """Smallest id of each id's component under represented parent edges, by union-find."""
    ids, parents, _ = _edges(columns)
    root = {i: i for i in ids}

    def find(v: int) -> int:
        while root[v] != v:
            root[v] = root[root[v]]
            v = root[v]
        return v

    for child, ps in parents.items():
        for p in ps:
            a, b = find(child), find(p)
            if a != b:
                root[max(a, b)] = min(a, b)
    smallest: dict[int, int] = {}
    for i in ids:
        smallest[find(i)] = min(smallest.get(find(i), i), i)
    return {i: smallest[find(i)] for i in ids}
