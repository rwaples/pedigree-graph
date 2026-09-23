"""The native lineage counts against the 0.9.3 Numba kernels kept under ``tests/oracle``.

Slice 15 moves ``distinct_ancestor_counts`` and ``descendant_path_counts`` to
the Rust core.  Counts do not depend on row labels, so both are held byte
for byte to the oracle, run as 0.9.3's facade ran it, on every parity
fixture in permuted row orders.  The descendant sweep's checked adds are
pinned by a sib-mating ladder whose path counts double each generation.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from oracle.lineage import _compute_n_ancestors, _compute_n_descendants
from oracle.remap import build_topology
from test_native_relationship_pairs import CHILD_PRELUDE, _run_child

from pedigree_graph import PedigreeGraph, PedigreeValidationError, ResourceError, _native

FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)
PERMUTATION_SEEDS = (None, 5, 11)


def _graph(name: str, seed: int | None = None) -> PedigreeGraph:
    columns = parity_columns(FIXTURES[name])
    if seed is not None:
        perm = np.random.default_rng(seed).permutation(len(columns["id"]))
        columns = {key: value[perm] for key, value in columns.items()}
    return PedigreeGraph.from_frame(columns)


def _oracle(kernel, graph: PedigreeGraph) -> np.ndarray:
    topo = build_topology(graph.depth)
    counts = kernel(topo.to_topological(graph.mother_rows), topo.to_topological(graph.father_rows), graph.n_individuals)
    return topo.per_row_to_graph(counts)


def _sib_mating_ladder(generations: int) -> PedigreeGraph:
    """Two founders, then two rows per generation, each a child of both rows above.

    A row ``g`` generations above the last has ``2^(g + 1) - 2`` descendant
    paths, so 62 generations put the founders one below int64's maximum and
    64 pass it.  Sib mating is not ``same_parent_id``, so the graph is valid.
    """
    n = 2 + 2 * generations
    rows = np.arange(n)
    mother = np.where(rows < 2, -1, (rows // 2 - 1) * 2)
    father = np.where(rows < 2, -1, (rows // 2 - 1) * 2 + 1)
    return PedigreeGraph.from_arrays(ids=rows, mother_ids=mother, father_ids=father)


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_counts_match_the_oracle_bytes(name, seed):
    graph = _graph(name, seed)
    ancestors = _native.distinct_ancestor_counts(graph._built, graph.depth)
    descendants = _native.descendant_path_counts(graph._built, graph.depth)
    assert ancestors.tobytes() == _oracle(_compute_n_ancestors, graph).astype(np.int32).tobytes()
    assert descendants.tobytes() == _oracle(_compute_n_descendants, graph).astype(np.int64).tobytes()


class TestOverflow:
    def test_sixty_two_generations_fit(self):
        counts = _sib_mating_ladder(62).descendant_path_counts()
        assert counts[0] == counts[1] == np.iinfo(np.int64).max - 1

    def test_sixty_four_generations_raise_rather_than_wrap(self):
        graph = _sib_mating_ladder(64)
        assert graph.n_individuals == 130
        with pytest.raises(ResourceError) as info:
            graph.descendant_path_counts()
        assert info.value.code == "arithmetic_overflow"
        assert info.value.fields == {"operation": "descendant_path_counts", "dtype": "int64"}


class TestBoundary:
    @pytest.mark.parametrize(
        ("binding", "dtype"),
        [(_native.distinct_ancestor_counts, np.int32), (_native.descendant_path_counts, np.int64)],
    )
    def test_the_arrays_are_owned_and_contiguous(self, binding, dtype):
        graph = _graph("random_1k")
        counts = binding(graph._built, graph.depth)
        assert counts.dtype == dtype
        assert counts.shape == (graph.n_individuals,)
        assert counts.flags.c_contiguous
        assert not isinstance(counts.base, np.ndarray)

    def test_the_public_arrays_are_the_bindings_without_a_copy(self):
        graph = _graph("random_1k")
        for public in (graph.distinct_ancestor_counts(), graph.descendant_path_counts()):
            assert not isinstance(public.base, np.ndarray)
            assert not public.flags.writeable

    def test_a_non_structural_depth_is_rejected_when_the_rows_need_sorting(self):
        graph = _graph("random_1k", seed=5)
        assert not graph._built.rows_topological
        flat = np.zeros(graph.n_individuals, dtype=np.int32)
        for binding in (_native.distinct_ancestor_counts, _native.descendant_path_counts):
            with pytest.raises(PedigreeValidationError) as info:
                binding(graph._built, flat)
            assert info.value.code == "value_out_of_range"
            assert info.value.fields["field"] == "depth"

    def test_depth_is_optional_only_for_parents_first_rows(self):
        topological, permuted = _graph("random_1k"), _graph("random_1k", seed=5)
        for binding in (_native.distinct_ancestor_counts, _native.descendant_path_counts):
            assert (
                binding(topological._built, None).tobytes() == binding(topological._built, topological.depth).tobytes()
            )
            with pytest.raises(PedigreeValidationError) as info:
                binding(permuted._built, None)
            assert info.value.code == "length_mismatch"
            assert info.value.fields["field"] == "depth"

    def test_parents_first_rows_never_read_depth(self):
        graph = _graph("random_1k")
        assert graph._built.rows_topological
        flat = np.zeros(graph.n_individuals, dtype=np.int32)
        for binding in (_native.distinct_ancestor_counts, _native.descendant_path_counts):
            assert binding(graph._built, flat).tobytes() == binding(graph._built, graph.depth).tobytes()


SEAM_CASES = [
    ("lineage_sets", "distinct_ancestor_counts"),
    ("lineage_output", "distinct_ancestor_counts"),
    ("lineage_output", "descendant_path_counts"),
]


@pytest.mark.parametrize(("family", "product"), SEAM_CASES, ids=[f"{f}-{p}" for f, p in SEAM_CASES])
def test_a_refused_allocation_raises_a_resource_error(family, product):
    body = f"""
        from pedigree_graph import ResourceError
        call = lambda: getattr(_native, {product!r})(graph._built, graph.depth)
        _native.fail_next_allocation({family!r}, n)
        try:
            call()
        except ResourceError as e:
            print(e.code, e.fields["operation"], e.fields["requested_elements"] >= n)
        else:
            print("no error")
        _native.fail_next_allocation(None)
        call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines == [f"allocation_failed {family} True", "recovered"]
