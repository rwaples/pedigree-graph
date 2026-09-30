"""Depth, lineage counts and components held to independent oracles and exact identities.

The oracle (``tests/oracle/closures.py``) reads only the id and parent
columns: set closures for distinct ancestors and descendants, explicit walk
enumeration for descendant paths, and union-find for components.  The lineage arrays measure different
things and are never compared elementwise with each other:
``distinct_ancestor_counts`` counts distinct strict ancestors and
``descendant_path_counts`` counts descendant *paths* (``_lineage.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import pedigree_arrays, pedigree_columns
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle.closures import component_minimum, descendant_walks, distinct_ancestors, distinct_descendants

from pedigree_graph import PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=100)


def _append(columns: dict, rows: dict) -> dict:
    return {key: np.concatenate([columns[key], np.asarray(rows[key], dtype=columns[key].dtype)]) for key in columns}


def _by_id(graph: PedigreeGraph) -> dict[str, dict[int, int]]:
    ids = graph.ids.tolist()
    return {
        "depth": dict(zip(ids, np.asarray(graph.depth).tolist(), strict=True)),
        "ancestors": dict(zip(ids, graph.distinct_ancestor_counts().tolist(), strict=True)),
        "walks": dict(zip(ids, graph.descendant_path_counts().tolist(), strict=True)),
        "component": dict(zip(ids, graph.connected_component_ids().tolist(), strict=True)),
    }


@_SETTINGS
@given(pedigree_columns(max_n=25))
def test_lineage_counts_equal_independent_closures(columns):
    # Rejects dedup applied to paths, path counting applied to ancestors, a
    # sweep that drops an edge, and components that follow external parents
    # or twin links.  Walks are at least the distinct descendants, and both
    # sums count ancestor-descendant pairs.
    got = _by_id(PedigreeGraph.from_frame(columns))
    ancestors, descendants, walks = (
        distinct_ancestors(columns),
        distinct_descendants(columns),
        descendant_walks(columns),
    )
    assert got["ancestors"] == ancestors
    assert got["walks"] == walks
    assert got["component"] == component_minimum(columns)
    assert all(walks[i] >= descendants[i] for i in walks)
    assert sum(ancestors.values()) == sum(descendants.values())


@_SETTINGS
@given(pedigree_arrays(non_inbred=True))
def test_descendant_walks_are_distinct_descendants_without_loops(arrays):
    # Mates with disjoint closed ancestor sets leave no ancestor two paths to
    # a descendant (conftest.pedigree_arrays), so walks and distinct
    # descendants coincide there and nowhere is the path count inflated.
    ids, mother, father, sex = arrays
    columns = {"id": ids, "mother": mother, "father": father}
    graph = PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, sex=sex)
    assert dict(zip(ids.tolist(), graph.descendant_path_counts().tolist(), strict=True)) == distinct_descendants(
        columns
    )


def _one_to_zero(ids: np.ndarray) -> np.ndarray:
    out = ids + 1
    if len(ids):
        out[np.argmax(ids)] = 0
    return out


_ID_MAPS = {
    "reversed": lambda ids: ids.max(initial=0) - ids,
    "near_int64_max": lambda ids: np.iinfo(np.int64).max - ids,
    "above_int32": lambda ids: 2**40 + 7 * ids,
    "one_id_to_zero": _one_to_zero,
}


@pytest.mark.parametrize("mapping", list(_ID_MAPS))
@_SETTINGS
@given(columns=pedigree_columns())
def test_structure_is_independent_of_id_values(mapping, columns):
    # Rejects a component labelled by its old minimum or its first row, ids
    # narrowed to int32 (no other test builds ids above 2**31 - 1), and a
    # ``> 0`` sentinel test.  Depth and both lineage counts move with the id;
    # the component id is the minimum of the mapped ids.
    universe = np.unique(np.concatenate([columns[k][columns[k] != -1] for k in ("id", "mother", "father")]))
    new = dict(zip(universe.tolist(), _ID_MAPS[mapping](universe).tolist(), strict=True))
    relabel = lambda v: np.array([new[int(x)] if x != -1 else -1 for x in v], dtype=np.int64)  # noqa: E731
    mapped = {**columns, **{k: relabel(columns[k]) for k in ("id", "mother", "father", "twin")}}
    before, after = _by_id(PedigreeGraph.from_frame(columns)), _by_id(PedigreeGraph.from_frame(mapped))
    for field in ("depth", "ancestors", "walks"):
        assert after[field] == {new[i]: v for i, v in before[field].items()}, field
    assert after["component"] == component_minimum(mapped)


@_SETTINGS
@given(pedigree_columns(max_n=20), pedigree_columns(max_n=20), st.data())
def test_a_disjoint_union_in_any_row_order_keeps_each_family_s_values(columns, other, data):
    # Rejects per-id outputs that depend on rows of an unrelated family,
    # including through interleaving.  F is exact on these depths
    # (test_kinship_exact.py), so it is compared bit for bit.
    shifted = {k: (np.where(v == -1, -1, v + 100_000) if k != "sex" else v) for k, v in other.items()}
    union = _append(columns, shifted)
    order = np.array(data.draw(st.permutations(range(len(union["id"])))), dtype=np.int64)
    union = {k: v[order] for k, v in union.items()}
    together = PedigreeGraph.from_frame(union)
    joint = _by_id(together)
    joint_f = dict(zip(together.ids.tolist(), together.inbreeding().tolist(), strict=True))
    for part in (columns, shifted):
        alone = PedigreeGraph.from_frame(part)
        for field, values in _by_id(alone).items():
            assert {i: joint[field][i] for i in values} == values, field
        assert {i: joint_f[i] for i in alone.ids.tolist()} == dict(
            zip(alone.ids.tolist(), alone.inbreeding().tolist(), strict=True)
        )


def _closed_subset(columns: dict, seeds: set[int], *, upward: bool) -> dict:
    """Rows of *columns* closed under represented parents (``upward``) or children, co-twins included."""
    ids = columns["id"].tolist()
    known = set(ids)
    twin = {i: int(t) for i, t in zip(ids, columns["twin"].tolist(), strict=True) if t != -1}
    step: dict[int, set[int]] = {i: set() for i in ids}
    for i, m, f in zip(ids, columns["mother"].tolist(), columns["father"].tolist(), strict=True):
        for p in (m, f):
            if p in known:
                if upward:
                    step[i].add(p)
                else:
                    step[p].add(i)
    keep, stack = set(), list(seeds)
    while stack:
        v = stack.pop()
        if v in keep:
            continue
        keep.add(v)
        stack.extend(step[v])
        if v in twin:
            stack.append(twin[v])
    mask = np.array([i in keep for i in ids], dtype=bool)
    return {k: v[mask] for k, v in columns.items()}


@_SETTINGS
@given(pedigree_columns().filter(lambda c: len(c["id"]) > 0), st.data())
def test_a_closed_subset_keeps_the_counts_its_closure_decides(columns, data):
    # A parent-closed subset keeps every ancestor, so depth and distinct
    # ancestors are unchanged by id; a child-closed subset keeps every
    # descendant, its dropped parents becoming external ids, so descendant
    # walks are unchanged.  Rejects an external parent counted as an ancestor.
    seeds = set(data.draw(st.sets(st.sampled_from(columns["id"].tolist()), min_size=1)))
    full = _by_id(PedigreeGraph.from_frame(columns))
    upward = _by_id(PedigreeGraph.from_frame(_closed_subset(columns, seeds, upward=True)))
    for field in ("depth", "ancestors"):
        assert upward[field] == {i: full[field][i] for i in upward[field]}, field
    downward = _by_id(PedigreeGraph.from_frame(_closed_subset(columns, seeds, upward=False)))
    assert downward["walks"] == {i: full["walks"][i] for i in downward["walks"]}
