"""The native Meuwissen-Luo walk against the 0.9.3 Numba kernel kept under ``tests/oracle``.

``PedigreeGraph.inbreeding`` runs in the Rust core, which walks
graph rows in its own depth-major order.  The oracle runs as 0.9.3's facade
ran it: parents remapped into the depth-major order, the kernel, the result
scattered back.  F is held to ``rtol 1e-9, atol 1e-12`` on every parity
fixture in permuted row orders; whether the bits also matched is recorded,
not asserted, since the plan does not promise it.
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import _PAIRWISE_FIXTURES, ADR_0008_FIXTURES, CHILD_PRELUDE, _mz_frame, _run_child
from conftest import FIXTURE_NAMES, parity_graph
from oracle.inbreeding import _compute_F_meuwissen_luo
from oracle.remap import build_topology

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native

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


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_f_matches_the_oracle(name, seed, record_property):
    graph = parity_graph(name, seed)
    native = _native.inbreeding(graph._built, graph.depth)
    oracle = _oracle_F(graph)
    np.testing.assert_allclose(native, oracle, rtol=RTOL, atol=ATOL)
    record_property("bit_identical", native.tobytes() == oracle.tobytes())


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
def test_mz_and_inbred_constructions_match_the_oracle(build):
    graph = PedigreeGraph.from_frame(build())
    np.testing.assert_allclose(_native.inbreeding(graph._built, graph.depth), _oracle_F(graph), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", ADR_0008_FIXTURES, ids=[case[0] for case in ADR_0008_FIXTURES])
def test_adr_0008_constructions_match_the_oracle(case):
    _name, mother, father, twin, _expected = case
    graph = PedigreeGraph.from_frame(_mz_frame(list(range(len(mother))), mother, father, twin))
    np.testing.assert_allclose(_native.inbreeding(graph._built, graph.depth), _oracle_F(graph), rtol=RTOL, atol=ATOL)


def test_the_oracle_handles_selfing():
    """Selfing (``same_parent_id``) is refused at construction, so no differential reaches it; pin it here."""
    mother = father = np.array([-1, 0], dtype=np.int32)
    depth = np.array([0, 1], dtype=np.int32)
    F = _compute_F_meuwissen_luo(mother, father, np.full(2, -1, dtype=np.int32), depth, 2)
    assert F[1] == 0.5


class TestBoundary:
    def test_the_array_is_owned_and_contiguous(self):
        graph = parity_graph("random_1k")
        F = _native.inbreeding(graph._built, graph.depth)
        assert F.dtype == np.float64
        assert F.shape == (graph.n_individuals,)
        assert F.flags.c_contiguous
        assert not isinstance(F.base, np.ndarray)

    def test_the_public_array_is_the_binding_without_a_copy(self):
        graph = parity_graph("random_1k")
        F = graph.inbreeding()
        assert not isinstance(F.base, np.ndarray)
        assert not F.flags.writeable
        assert F.tobytes() == _native.inbreeding(graph._built, graph.depth).tobytes()

    def test_an_empty_pedigree_gives_an_empty_array(self):
        empty = np.zeros(0, np.int64)
        graph = PedigreeGraph.from_frame({"id": empty, "mother": empty, "father": empty})
        assert _native.inbreeding(graph._built, graph.depth).shape == (0,)

    def test_a_non_structural_depth_is_rejected(self):
        graph = parity_graph("nuclear_full_sibs")
        with pytest.raises(PedigreeValidationError) as info:
            _native.inbreeding(graph._built, np.zeros(graph.n_individuals, dtype=np.int32))
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "depth"


def test_a_refused_allocation_raises_a_resource_error():
    body = """
        from pedigree_graph import ResourceError
        _native.fail_next_allocation("inbreeding_walk", n)
        try:
            _native.inbreeding(graph._built, graph.depth)
        except ResourceError as e:
            print(e.code, e.fields["operation"], e.fields["requested_elements"] >= n)
        else:
            print("no error")
        _native.fail_next_allocation(None)
        _native.inbreeding(graph._built, graph.depth)
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines == ["allocation_failed inbreeding_walk True", "recovered"]
