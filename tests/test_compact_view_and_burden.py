"""Parity of the compact view engine and the streamed burden sink."""

from __future__ import annotations

import numpy as np
import pedigrees
import pytest
from _support import CHILD_PRELUDE, _run_child

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, _native


@pytest.mark.parametrize(
    "fixture",
    [*pedigrees.motif_fixtures().values(), pedigrees.build_random("random_1k", pedigrees.RANDOM_FIXTURES["random_1k"])],
)
def test_compact_view_pairs_match_full_graph(fixture: dict[str, np.ndarray]) -> None:
    """Parent closure, external parent IDs, MZ, and view order preserve every block."""
    graph = PedigreeGraph.from_frame(
        {
            "id": fixture["ids"],
            "mother": fixture["mother"],
            "father": fixture["father"],
            "twin": fixture["twin"],
            "sex": fixture["sex"],
        }
    )
    n = graph.n_individuals
    selected = np.arange(n - 1, -1, -3, dtype=np.int32)
    if len(selected) < 2:
        return
    view_map = graph.view(rows=selected)._graph_to_view()
    mask = view_map >= 0
    counts = _native.relationship_counts(graph._built, max_degree=5, threads=1, selected=mask)
    compact_counts = _native.compact_view_counts(graph._built, view_map, max_degree=5, threads=1)
    assert compact_counts == counts
    for execution in ("speed", "memory"):
        old = _native.relationship_pairs(
            graph._built,
            max_degree=5,
            requested=list(RELATIONSHIPS),
            threads=1,
            execution=execution,
            view_rows=view_map,
        )
        compact = _native.relationship_pairs(
            graph._built,
            max_degree=5,
            requested=list(RELATIONSHIPS),
            threads=1,
            execution=execution,
            view_rows=view_map,
            compact=True,
        )
        for code in RELATIONSHIPS:
            np.testing.assert_array_equal(compact[code][0], old[code][0], err_msg=code)
            np.testing.assert_array_equal(compact[code][1], old[code][1], err_msg=code)


@pytest.mark.parametrize("name", ["random_1k", "deep_inbred_60g"])
def test_burden_matches_pair_blocks(name: str) -> None:
    """One native summary equals aggregation of the materialised closest-category pairs."""
    fixture = pedigrees.build_random(name, pedigrees.RANDOM_FIXTURES[name])
    graph = PedigreeGraph.from_frame(
        {
            "id": fixture["ids"],
            "mother": fixture["mother"],
            "father": fixture["father"],
            "twin": fixture["twin"],
            "sex": fixture["sex"],
        }
    )
    pairs = graph.relationship_pairs(max_degree=5)
    burden = graph.relationship_burden()
    expected = np.zeros((graph.n_individuals, 5), dtype=np.uint32)
    same_depth = np.zeros(len(burden.same_depth_pairs), dtype=np.uint64)
    for code, block in pairs.items():
        assert burden.category_counts[code] == len(block)
        degree = RELATIONSHIPS[code].degree
        for a, b in zip(block.first_rows, block.second_rows, strict=True):
            if degree > 0:
                expected[a, degree - 1] += 1
                expected[b, degree - 1] += 1
            if graph.depth[a] == graph.depth[b]:
                same_depth[graph.depth[a]] += 1
    np.testing.assert_array_equal(burden.per_person, expected)
    np.testing.assert_array_equal(burden.same_depth_pairs, same_depth)


def test_burden_rejects_depth_outside_graph_before_allocating() -> None:
    """The native boundary cannot allocate an array sized by arbitrary depth input."""
    graph = PedigreeGraph.from_frame({"id": [1, 2], "mother": [-1, 1], "father": [-1, -1], "sex": [0, 1]})
    with pytest.raises(ValueError, match="graph row range"):
        _native.relationship_burden(graph._built, np.array([0, 2_000_000_000], dtype=np.int32), threads=1)


def test_sparse_large_view_uses_compact_path_without_changing_public_results() -> None:
    """The automatic dispatch keeps graph/view coordinates and category counts."""
    fixture = pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"])
    graph = PedigreeGraph.from_frame(
        {
            "id": fixture["ids"],
            "mother": fixture["mother"],
            "father": fixture["father"],
            "twin": fixture["twin"],
            "sex": fixture["sex"],
        }
    )
    selected = np.random.default_rng(42).permutation(graph.n_individuals)[:303]
    view = graph.view(rows=selected)
    map_rows = view._graph_to_view()
    expected = _native.relationship_pairs(
        graph._built, max_degree=5, requested=list(RELATIONSHIPS), threads=1, execution="speed", view_rows=map_rows
    )
    got = view.relationship_pairs(max_degree=5)
    counts = view.relationship_counts(max_degree=5)
    for code in RELATIONSHIPS:
        np.testing.assert_array_equal(got[code].first_rows, expected[code][0])
        np.testing.assert_array_equal(got[code].second_rows, expected[code][1])
        assert counts[code] == len(got[code])


def test_burden_is_bit_identical_under_every_thread_budget() -> None:
    """Budgets 1 and 4 give the same bytes; the pool is built once per process, so each runs in its own."""
    body = """
        burden = graph.relationship_burden()
        digest = hashlib.sha256()
        digest.update(np.array(list(burden.category_counts.values()), dtype=np.uint64).tobytes())
        digest.update(np.ascontiguousarray(burden.per_person).tobytes())
        digest.update(np.ascontiguousarray(burden.same_depth_pairs).tobytes())
        print(thread_budget(), int(burden.per_person.sum()), digest.hexdigest())
    """
    one = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="1").split()
    four = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="4").split()
    assert one[0] == "1"
    assert four[0] == "4"
    assert int(one[1]) > 0
    assert one[1:] == four[1:]
