"""The native Meuwissen-Luo walk against the 0.9.3 Numba kernel kept under ``tests/oracle``.

Slice 15 moves ``PedigreeGraph.inbreeding`` to the Rust core, which walks
graph rows in its own depth-major order.  The oracle runs as 0.9.3's facade
ran it: parents remapped into the depth-major order, the kernel, the result
scattered back.  F is held to ``rtol 1e-9, atol 1e-12`` on every parity
fixture in permuted row orders; whether the bits also matched is recorded,
not asserted, since the plan does not promise it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from oracle.inbreeding import _compute_F_meuwissen_luo
from oracle.remap import build_topology
from test_native_relationship_pairs import CHILD_PRELUDE, _run_child
from test_pedigree_graph import _PAIRWISE_FIXTURES

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
PERMUTATION_SEEDS = (None, 5, 11)
RTOL, ATOL = 1e-9, 1e-12


def _graph(name: str, seed: int | None = None) -> PedigreeGraph:
    columns = parity_columns(FIXTURES[name])
    if seed is not None:
        perm = np.random.default_rng(seed).permutation(len(columns["id"]))
        columns = {key: value[perm] for key, value in columns.items()}
    return PedigreeGraph.from_frame(columns)


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
    graph = _graph(name, seed)
    native = _native.inbreeding(graph._built, graph.depth)
    oracle = _oracle_F(graph)
    np.testing.assert_allclose(native, oracle, rtol=RTOL, atol=ATOL)
    record_property("bit_identical", native.tobytes() == oracle.tobytes())


@pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
def test_mz_and_inbred_constructions_match_the_oracle(build):
    graph = PedigreeGraph.from_frame(build())
    np.testing.assert_allclose(_native.inbreeding(graph._built, graph.depth), _oracle_F(graph), rtol=RTOL, atol=ATOL)


class TestBoundary:
    def test_the_array_is_owned_and_contiguous(self):
        graph = _graph("random_1k")
        F = _native.inbreeding(graph._built, graph.depth)
        assert F.dtype == np.float64
        assert F.shape == (graph.n_individuals,)
        assert F.flags.c_contiguous
        assert not isinstance(F.base, np.ndarray)

    def test_the_public_array_is_the_binding_without_a_copy(self):
        graph = _graph("random_1k")
        F = graph.inbreeding()
        assert not isinstance(F.base, np.ndarray)
        assert not F.flags.writeable
        assert F.tobytes() == _native.inbreeding(graph._built, graph.depth).tobytes()

    def test_an_empty_pedigree_gives_an_empty_array(self):
        empty = np.zeros(0, np.int64)
        graph = PedigreeGraph.from_frame({"id": empty, "mother": empty, "father": empty})
        assert _native.inbreeding(graph._built, graph.depth).shape == (0,)

    def test_a_non_structural_depth_is_rejected(self):
        graph = _graph("nuclear_full_sibs")
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
