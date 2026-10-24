"""The native Meuwissen-Luo walk against the 0.9.3 Numba kernel kept under ``tests/oracle``.

``PedigreeGraph.inbreeding`` runs in the Rust core, which walks
graph rows in its own depth-major order.  The oracle runs as 0.9.3's facade
ran it: parents remapped into the depth-major order, the kernel, the result
scattered back.  F is held to ``rtol 1e-9, atol 1e-12`` on every parity
fixture in permuted row orders; whether the bits also matched is recorded,
not asserted, since the plan does not promise it.  The ancestor counts the
same walk returns are held exactly to the ancestor-set sweep behind
``_native.distinct_ancestor_counts``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import _PAIRWISE_FIXTURES, ADR_0008_FIXTURES, CHILD_PRELUDE, _mz_frame, _run_child
from conftest import FIXTURE_NAMES, parity_graph
from oracle.inbreeding import _compute_F_meuwissen_luo
from oracle.remap import build_topology

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._lineage import sweep_depth
from pedigree_graph._threads import thread_budget

PERMUTATION_SEEDS = (None, 5, 11)
RTOL, ATOL = 1e-9, 1e-12


def _oracle_F(graph: PedigreeGraph) -> np.ndarray:
    topo = build_topology(graph.depth)
    F = _compute_F_meuwissen_luo(
        topo.to_topological(graph.mother_rows),
        topo.to_topological(graph.father_rows),
        topo.to_topological(graph.twin_rows),
        topo.gather(graph.depth),
        graph.n_individuals,
    )
    return topo.per_row_to_graph(F)


def _walk_F(graph: PedigreeGraph) -> np.ndarray:
    """The walk's F, checking its ancestor counts against the set sweep on the way."""
    F, ancestors = _native.inbreeding(graph._built, graph.depth, threads=thread_budget())
    sets = _native.distinct_ancestor_counts(graph._built, sweep_depth(graph))
    assert ancestors.tobytes() == sets.tobytes()
    return F


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_f_matches_the_oracle(name, seed, record_property):
    graph = parity_graph(name, seed)
    native = _walk_F(graph)
    oracle = _oracle_F(graph)
    np.testing.assert_allclose(native, oracle, rtol=RTOL, atol=ATOL)
    record_property("bit_identical", native.tobytes() == oracle.tobytes())


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
def test_mz_and_inbred_constructions_match_the_oracle(build):
    graph = PedigreeGraph.from_frame(build())
    np.testing.assert_allclose(_walk_F(graph), _oracle_F(graph), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", ADR_0008_FIXTURES, ids=[case[0] for case in ADR_0008_FIXTURES])
def test_adr_0008_constructions_match_the_oracle(case):
    _name, mother, father, twin, _expected = case
    graph = PedigreeGraph.from_frame(_mz_frame(list(range(len(mother))), mother, father, twin))
    np.testing.assert_allclose(_walk_F(graph), _oracle_F(graph), rtol=RTOL, atol=ATOL)


def test_the_oracle_handles_selfing():
    """Selfing (``same_parent_id``) is refused at construction, so no differential reaches it; pin it here."""
    mother = father = np.array([-1, 0], dtype=np.int32)
    depth = np.array([0, 1], dtype=np.int32)
    F = _compute_F_meuwissen_luo(mother, father, np.full(2, -1, dtype=np.int32), depth, 2)
    assert F[1] == 0.5


class TestBoundary:
    def test_the_arrays_are_owned_and_contiguous(self):
        graph = parity_graph("random_1k")
        F, ancestors = _native.inbreeding(graph._built, graph.depth, threads=thread_budget())
        for array, dtype in ((F, np.float64), (ancestors, np.int32)):
            assert array.dtype == dtype
            assert array.shape == (graph.n_individuals,)
            assert array.flags.c_contiguous
            assert not isinstance(array.base, np.ndarray)

    def test_the_public_array_is_the_binding_without_a_copy(self):
        graph = parity_graph("random_1k")
        F = graph.inbreeding()
        assert not isinstance(F.base, np.ndarray)
        assert not F.flags.writeable
        assert F.tobytes() == _native.inbreeding(graph._built, graph.depth, threads=thread_budget())[0].tobytes()

    def test_inbreeding_fills_the_ancestor_count_memo(self, monkeypatch):
        graph = parity_graph("random_1k")
        want = parity_graph("random_1k").distinct_ancestor_counts()
        graph.inbreeding()

        def refuse(*_args):
            raise AssertionError("the ancestor-set sweep ran after the walk")

        monkeypatch.setattr(_native, "distinct_ancestor_counts", refuse)
        counts = graph.distinct_ancestor_counts()
        assert counts.tobytes() == want.tobytes()
        assert not counts.flags.writeable

    def test_an_empty_pedigree_gives_empty_arrays(self):
        empty = np.zeros(0, np.int64)
        graph = PedigreeGraph.from_frame({"id": empty, "mother": empty, "father": empty})
        F, ancestors = _native.inbreeding(graph._built, graph.depth, threads=thread_budget())
        assert F.shape == ancestors.shape == (0,)

    def test_a_non_structural_depth_is_rejected(self):
        graph = parity_graph("nuclear_full_sibs")
        with pytest.raises(PedigreeValidationError) as info:
            _native.inbreeding(graph._built, np.zeros(graph.n_individuals, dtype=np.int32), threads=thread_budget())
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "depth"


def test_a_refused_allocation_raises_a_resource_error():
    body = """
        from pedigree_graph import ResourceError
        _native.fail_next_allocation("inbreeding_walk", n)
        try:
            _native.inbreeding(graph._built, graph.depth, threads=thread_budget())
        except ResourceError as e:
            print(e.code, e.fields["operation"], e.fields["requested_elements"] >= n)
        else:
            print("no error")
        _native.fail_next_allocation(None)
        _native.inbreeding(graph._built, graph.depth, threads=thread_budget())
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines == ["allocation_failed inbreeding_walk True", "recovered"]


THREAD_BODY = """
    import hashlib
    import numpy as np
    from pedigree_graph import PedigreeGraph
    from pedigree_graph._threads import thread_budget
    rng = np.random.default_rng(3)
    mother, father, twin = [-1] * 300, [-1] * 300, [-1] * 300
    for _ in range(6):
        earlier = len(mother)
        for _ in range(700):
            m, f = int(rng.integers(earlier)), int(rng.integers(earlier))
            first = len(mother)
            for _ in range(int(rng.integers(1, 5))):
                mother.append(m); father.append(f if f != m else -1); twin.append(-1)
            if len(mother) - first >= 2 and rng.random() < 0.3:
                twin[first], twin[first + 1] = first + 1, first
    rows = rng.permutation(len(mother))
    graph = PedigreeGraph.from_frame({
        "id": rows,
        "mother": np.asarray(mother)[rows],
        "father": np.asarray(father)[rows],
        "twin": np.asarray(twin)[rows],
    })
    F = graph.inbreeding()
    counts = graph.distinct_ancestor_counts()
    print(thread_budget(), hashlib.sha256(F.tobytes() + counts.tobytes()).hexdigest())
"""


def test_f_and_counts_are_bit_identical_under_every_thread_budget():
    """Each budget runs in its own interpreter, since the package pool is built once per process.

    Every depth holds hundreds of sibships, many opening with MZ co-twins, in
    shuffled rows, so the walks split across workers and full sibs and
    co-twins stand apart in the input.
    """
    one = _run_child(THREAD_BODY, PEDIGREE_GRAPH_THREADS="1").split()
    four = _run_child(THREAD_BODY, PEDIGREE_GRAPH_THREADS="4").split()
    assert one[0] == "1"
    assert four[0] == "4"
    assert one[1] == four[1]
