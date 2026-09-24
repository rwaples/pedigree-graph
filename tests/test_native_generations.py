"""The native Ne prerequisites against their 0.9.3 forms kept under ``tests/oracle``.

Maignel's equivalent complete generations (the Numba ``_compute_eqg``) and
the per-cohort founder-contribution means (the NumPy adjoint sweep of
``_per_gen_founder_means``) run in the Rust core.  Both are
held to ``rtol 1e-9, atol 1e-12`` on every parity fixture in permuted row
orders, with bit identity recorded rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import CHILD_PRELUDE, _run_child
from conftest import parity_columns, parity_fixtures
from oracle.eqg import _compute_eqg
from oracle.founder_means import _per_gen_founder_means as _oracle_founder_means

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_founders import _founder_columns, _founder_idx, _per_gen_founder_means

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


def _cohort_sets(graph: PedigreeGraph) -> dict[str, ObservedCohorts]:
    """Cohorts by depth, and by merged depths with every third row unlabelled."""
    depth = np.asarray(graph.depth)
    merged = np.where(np.arange(graph.n_individuals) % 3 == 0, -1, depth // 2).astype(np.int32)
    return {"depth": ObservedCohorts.from_labels(depth), "merged": ObservedCohorts.from_labels(merged)}


@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_equivalent_generations_match_the_oracle(name, seed, record_property):
    graph = _graph(name, seed)
    native = _native.equivalent_generations(graph._built, graph.depth)
    oracle = _compute_eqg(
        np.asarray(graph.mother_rows), np.asarray(graph.father_rows), np.asarray(graph.depth), graph.n_individuals
    )
    np.testing.assert_allclose(native, oracle, rtol=RTOL, atol=ATOL)
    record_property("bit_identical", native.tobytes() == oracle.tobytes())


@pytest.mark.parametrize("cohorts", ["depth", "merged"])
@pytest.mark.parametrize("seed", PERMUTATION_SEEDS)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_founder_means_match_the_oracle(name, seed, cohorts, record_property):
    graph = _graph(name, seed)
    grouping = _cohort_sets(graph)[cohorts]
    founder_idx = _founder_idx(graph)
    native = _per_gen_founder_means(graph, founder_idx=founder_idx, cohorts=grouping)
    oracle = _oracle_founder_means(graph, founder_idx=founder_idx, cohorts=grouping)
    np.testing.assert_array_equal(native.founder_idx, oracle.founder_idx)
    assert native.m_g.shape == oracle.m_g.shape
    np.testing.assert_allclose(native.m_g, oracle.m_g, rtol=RTOL, atol=ATOL)
    record_property("bit_identical", native.m_g.tobytes() == oracle.m_g.tobytes())


class TestBoundary:
    def test_the_arrays_are_owned_and_contiguous(self):
        graph = _graph("random_1k")
        grouping = _cohort_sets(graph)["depth"]
        founder_idx = _founder_idx(graph)
        eqg = _native.equivalent_generations(graph._built, graph.depth)
        means = _native.founder_contribution_means(
            graph._built,
            graph.depth,
            grouping.dense,
            grouping.k,
            _founder_columns(graph, founder_idx),
            founder_idx.shape[0],
        )
        assert eqg.shape == (graph.n_individuals,)
        assert means.shape == (grouping.k * founder_idx.shape[0],)
        for array in (eqg, means):
            assert array.dtype == np.float64
            assert array.flags.c_contiguous
            assert not isinstance(array.base, np.ndarray)

    def test_the_facade_means_are_read_only(self):
        m_g = _per_gen_founder_means(_graph("random_1k")).m_g
        assert not m_g.flags.writeable

    def test_bad_cohorts_and_founder_columns_are_rejected(self):
        graph = _graph("nuclear_full_sibs")
        n = graph.n_individuals
        cohort = np.zeros(n, np.int32)
        column = np.full(n, -1, np.int64)
        column[0] = 0

        def call(cohort=cohort, n_cohorts=1, column=column, n_genomes=1):
            _native.founder_contribution_means(graph._built, graph.depth, cohort, n_cohorts, column, n_genomes)

        for kwargs, field in [
            ({"cohort": np.full(n, 2, np.int32)}, "cohort"),
            ({"cohort": np.full(n, -1, np.int32)}, "cohort"),
            ({"column": np.full(n, 1, np.int64)}, "founder_column"),
            ({"column": np.full(n, -2, np.int64)}, "founder_column"),
        ]:
            with pytest.raises(PedigreeValidationError) as info:
                call(**kwargs)
            assert info.value.code == "value_out_of_range"
            assert info.value.fields["field"] == field
        with pytest.raises(PedigreeValidationError) as info:
            call(cohort=cohort[:-1])
        assert info.value.code == "length_mismatch"

    def test_a_non_structural_depth_is_rejected_when_the_rows_need_sorting(self):
        graph = _graph("random_1k", seed=5)
        assert not graph._built.rows_topological
        with pytest.raises(PedigreeValidationError) as info:
            _native.equivalent_generations(graph._built, np.zeros(graph.n_individuals, dtype=np.int32))
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "depth"


SEAM_CASES = [("lineage_output", "equivalent_generations"), ("founder_means", "founder_contribution_means")]


@pytest.mark.parametrize(("family", "product"), SEAM_CASES, ids=[f"{f}-{p}" for f, p in SEAM_CASES])
def test_a_refused_allocation_raises_a_resource_error(family, product):
    body = f"""
        from pedigree_graph import ResourceError
        cohort = np.minimum(graph.depth, 3).astype(np.int32)
        column = np.where((graph.mother_rows < 0) & (graph.father_rows < 0), np.arange(n), -1).astype(np.int64)
        calls = {{
            "equivalent_generations": lambda: _native.equivalent_generations(graph._built, graph.depth),
            "founder_contribution_means": lambda: _native.founder_contribution_means(
                graph._built, graph.depth, cohort, 4, column, n
            ),
        }}
        call = calls[{product!r}]
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
