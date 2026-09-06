"""``pair_kinship`` on graphs and views: the pinned float32 recurrence (ADR 0009, slice 5a).

Within one receiver every value is bit-identical to the ``kinship_matrix``
entry for the same pair, to the pure-Python oracle, to the reversed endpoint
order, and to itself before and after a matrix is cached.  Across two row
orders of one pedigree the values stay inside the ADR 0009 envelope, and the
ULP distance is reported.  Errors carry structured codes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from pedigree_graph import (
    RELATIONSHIPS,
    PedigreeGraph,
    PedigreeValidationError,
    ResourceError,
)
from pedigree_graph._kinship_pairwise import _pairwise_kinship_py, _run_kernel, pairwise_kinship
from pedigree_graph._threads import _reset_thread_state, configure_threads

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees

MAX_DEGREE = 5
ENVELOPE_UNIT = 2.0**-25


def _fixtures() -> dict[str, dict[str, np.ndarray]]:
    fixtures = dict(pedigrees.motif_fixtures())
    for name in ("random_1k", "deep_inbred_60g"):
        fixtures[name] = pedigrees.build_random(name, pedigrees.RANDOM_FIXTURES[name])
    return fixtures


FIXTURES = _fixtures()
FIXTURE_NAMES = sorted(FIXTURES)
MOTIF_NAMES = sorted(pedigrees.motif_fixtures())


def _columns(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "id": fixture["ids"],
        "mother": fixture["mother"],
        "father": fixture["father"],
        "twin": fixture["twin"],
        "sex": fixture["sex"],
    }


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph(_columns(FIXTURES[name]))


def _all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n)


def _kernel_inputs(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """Parent arrays and endpoints in the private depth-major order the kernel runs in."""
    mother, father, twin = graph._topological_parents
    return mother, father, twin, graph._topology.translate(first), graph._topology.translate(second)


def _matrix_values(graph: PedigreeGraph, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(graph.kinship_matrix(0.0)[first, second], dtype=np.float32).ravel()


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def _sib_mating() -> PedigreeGraph:
    # 0,1 founders; 2,3 full sibs; 4 = child(2,3) with F = 0.25.
    return PedigreeGraph(
        {
            "id": np.arange(5),
            "mother": np.array([-1, -1, 0, 0, 2]),
            "father": np.array([-1, -1, 1, 1, 3]),
            "twin": np.full(5, -1),
            "sex": np.array([0, 1, 0, 1, 0]),
        }
    )


def _mz_twins_with_descendants() -> PedigreeGraph:
    # 4,5 MZ twins of full-sib parents (2,3); 8 = child(4,6), 9 = child(5,7).
    return PedigreeGraph(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, 0, 0, 2, 2, -1, -1, 4, 5]),
            "father": np.array([-1, -1, 1, 1, 3, 3, -1, -1, 6, 7]),
            "twin": np.array([-1, -1, -1, -1, 5, 4, -1, -1, -1, -1]),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0]),
        }
    )


def _double_first_cousins() -> PedigreeGraph:
    # (8,9) and (9,10) are double first cousins: phi = 0.125, not the nominal 1/16.
    return PedigreeGraph(
        {
            "id": np.arange(11),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(11, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
        }
    )


class TestCallForms:
    def test_rows_form_returns_read_only_float32(self):
        graph = _sib_mating()
        values = graph.pair_kinship([4, 2, 0], [4, 3, 1])
        assert values.dtype == np.float32
        assert not values.flags.writeable
        assert values.tolist() == [0.625, 0.25, 0.0]

    def test_rows_accept_integral_floats_and_int32(self):
        graph = _sib_mating()
        expected = graph.pair_kinship([4, 2], [4, 3])
        assert (
            graph.pair_kinship(np.array([4.0, 2.0]), np.array([4, 3], dtype=np.int32)).tobytes() == expected.tobytes()
        )

    def test_empty_rows_give_an_empty_float32(self):
        values = _sib_mating().pair_kinship([], [])
        assert values.dtype == np.float32
        assert values.shape == (0,)

    def test_block_form_aligns_to_the_block(self):
        graph = _double_first_cousins()
        block = graph.relationship_pairs(max_degree=3)["1C"]
        values = graph.pair_kinship(block)
        assert len(values) == len(block)
        assert values.dtype == np.float32
        assert not values.flags.writeable
        assert values.tolist() == [0.125, 0.125]

    def test_collection_form_covers_all_23_codes(self):
        graph = _double_first_cousins()
        pairs = graph.relationship_pairs(max_degree=2)
        values = graph.pair_kinship(pairs)
        assert isinstance(values, MappingProxyType)
        assert tuple(values) == tuple(RELATIONSHIPS)
        for code, block in pairs.items():
            assert values[code].dtype == np.float32
            assert not values[code].flags.writeable
            assert len(values[code]) == len(block)
            assert values[code].tobytes() == graph.pair_kinship(block).tobytes()
        assert not pairs["1C"].requested
        assert len(values["1C"]) == 0

    def test_collection_form_of_an_empty_selection(self):
        graph = _sib_mating()
        values = graph.pair_kinship(graph.relationship_pairs(categories=["2C"]))
        assert all(len(value) == 0 for value in values.values())

    def test_collection_is_immutable(self):
        graph = _sib_mating()
        values = graph.pair_kinship(graph.relationship_pairs(max_degree=1))
        with pytest.raises(TypeError):
            values["FS"] = np.zeros(1, dtype=np.float32)  # type: ignore[index]

    def test_endpoint_reversal_is_bit_identical(self):
        graph = _graph("deep_inbred_60g")
        first, second = _all_pairs(graph.n_individuals)
        assert graph.pair_kinship(first, second).tobytes() == graph.pair_kinship(second, first).tobytes()

    def test_self_pairs_encode_inbreeding(self):
        graph = _sib_mating()
        rows = np.arange(graph.n_individuals)
        values = graph.pair_kinship(rows, rows).astype(np.float64)
        np.testing.assert_array_equal(2.0 * values - 1.0, graph.compute_inbreeding())


class TestValues:
    def test_mz_ancestry_raises_descendant_kinship(self):
        graph = _mz_twins_with_descendants()
        assert graph.pair_kinship([4, 4, 8], [5, 4, 9]).tolist() == [0.625, 0.625, 0.15625]

    def test_double_first_cousins_are_not_nominal(self):
        assert _double_first_cousins().pair_kinship([8, 9], [9, 10]).tolist() == [0.125, 0.125]

    def test_half_first_cousin_parents_are_not_pruned(self):
        graph = PedigreeGraph(
            {
                "id": np.arange(10),
                "mother": np.array([-1, -1, -1, -1, -1, 0, 0, 5, 6, 7]),
                "father": np.array([-1, -1, -1, -1, -1, 1, 2, 3, 4, 8]),
                "twin": np.full(10, -1),
                "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            }
        )
        assert graph.pair_kinship([7, 9], [8, 7]).tolist() == [0.03125, 0.265625]

    @pytest.mark.parametrize("name", ["one_parent_known", "external_parents", "founder_mz_twins"])
    def test_partial_parentage_matches_the_matrix(self, name):
        graph = _graph(name)
        first, second = _all_pairs(graph.n_individuals)
        assert graph.pair_kinship(first, second).tobytes() == _matrix_values(graph, first, second).tobytes()

    def test_zero_is_exact(self):
        graph = _graph("random_1k")
        first, second = _all_pairs(graph.n_individuals)
        values = graph.pair_kinship(first, second)
        matrix = graph.kinship_matrix(0.0)
        support = np.asarray(matrix[first, second] != 0).ravel()
        np.testing.assert_array_equal(values != 0, support)


class TestWithinGraphParity:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_every_pair_matches_the_matrix_bit_for_bit(self, name):
        graph = _graph(name)
        first, second = _all_pairs(graph.n_individuals)
        assert graph.pair_kinship(first, second).tobytes() == _matrix_values(graph, first, second).tobytes()

    @pytest.mark.parametrize("name", [*MOTIF_NAMES, "deep_inbred_60g"])
    def test_kernel_matches_the_python_oracle(self, name):
        graph = _graph(name)
        first, second = _all_pairs(graph.n_individuals)
        kernel = pairwise_kinship(*_kernel_inputs(graph, first, second))
        oracle = _pairwise_kinship_py(graph.mother, graph.father, graph.twin, graph.depth, first, second)
        assert kernel.tobytes() == oracle.tobytes()

    @pytest.mark.parametrize("name", ["deep_inbred_60g", "random_1k"])
    def test_result_is_the_same_before_and_after_the_matrix(self, name):
        def pairs_of(graph):
            return graph.pair_kinship(graph.relationship_pairs(max_degree=MAX_DEGREE))

        fresh = _graph(name)
        before = pairs_of(fresh)
        cached = _graph(name)
        cached.kinship_matrix(0.0)
        after = pairs_of(cached)
        for code in RELATIONSHIPS:
            assert before[code].tobytes() == after[code].tobytes(), code
        fresh.kinship_matrix(0.0)
        again = pairs_of(fresh)
        for code in RELATIONSHIPS:
            assert before[code].tobytes() == again[code].tobytes(), code

    def test_relationship_pairs_of_a_reordered_graph_match_its_own_matrix(self):
        fixture = FIXTURES["deep_inbred_60g"]
        perm = np.random.default_rng(5).permutation(len(fixture["ids"]))
        graph = PedigreeGraph({key: value[perm] for key, value in _columns(fixture).items()})
        pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
        values = graph.pair_kinship(pairs)
        for code, block in pairs.items():
            if len(block):
                assert values[code].tobytes() == _matrix_values(graph, *block).tobytes(), code


class TestCrossOrderEnvelope:
    def test_permuted_deep_inbred_graphs_agree_within_the_envelope(self, capsys):
        fixture = FIXTURES["deep_inbred_60g"]
        reference = _graph("deep_inbred_60g")
        n = reference.n_individuals
        first, second = _all_pairs(n)
        want = reference.pair_kinship(first, second)
        depth = reference.depth
        tolerance = 2.0 * (depth[first] + depth[second] + 1) * ENVELOPE_UNIT
        worst_ulp = 0
        differing = 0
        for seed in (11, 12):
            perm = np.random.default_rng(seed).permutation(n)
            permuted = PedigreeGraph({key: value[perm] for key, value in _columns(fixture).items()})
            inverse = np.empty(n, dtype=np.intp)
            inverse[perm] = np.arange(n)
            got = permuted.pair_kinship(inverse[first], inverse[second])
            deviation = np.abs(want.astype(np.float64) - got.astype(np.float64))
            assert np.all(deviation <= tolerance), f"seed {seed}: beyond the ADR 0009 envelope"
            np.testing.assert_array_equal(got == 0, want == 0)
            ulp = _ulp_distance(want, got)
            worst_ulp = max(worst_ulp, int(ulp.max()))
            differing += int((ulp > 0).sum())
        with capsys.disabled():
            print(
                f"cross-order pair_kinship deep_inbred_60g: {differing} of {2 * len(first)} differ, max {worst_ulp} ulp"
            )


class TestViews:
    def test_view_rows_are_converted_to_graph_rows(self):
        graph = _graph("double_first_cousins")
        rows = np.arange(graph.n_individuals)[::-1]
        view = graph.view(rows=rows)
        first, second = _all_pairs(view.n_individuals)
        assert view.pair_kinship(first, second).tobytes() == graph.pair_kinship(rows[first], rows[second]).tobytes()

    def test_view_blocks_and_collections_match_the_graph(self):
        graph = _graph("random_1k")
        view = graph.view(ids=FIXTURES["random_1k"]["ids"][::3])
        pairs = view.relationship_pairs(max_degree=3)
        values = view.pair_kinship(pairs)
        for code, block in pairs.items():
            expected = graph.pair_kinship(view.graph_rows[block.first_rows], view.graph_rows[block.second_rows])
            assert values[code].tobytes() == expected.tobytes(), code
            assert view.pair_kinship(block).tobytes() == expected.tobytes(), code

    def test_unselected_ancestors_still_count(self):
        graph = _double_first_cousins()
        view = graph.view(rows=[8, 9, 10])
        assert view.pair_kinship([0, 1], [1, 2]).tolist() == [0.125, 0.125]

    def test_empty_view_accepts_only_empty_queries(self):
        view = _sib_mating().view(rows=[])
        assert view.pair_kinship([], []).shape == (0,)
        with pytest.raises(PedigreeValidationError) as info:
            view.pair_kinship([0], [0])
        assert info.value.code == "pair_row_out_of_range"
        assert info.value.fields["n_individuals"] == 0

    def test_graph_block_is_rejected_by_a_view(self):
        graph = _sib_mating()
        view = graph.view(rows=[4, 3, 2])
        with pytest.raises(PedigreeValidationError) as info:
            view.pair_kinship(graph.relationship_pairs(max_degree=1)["FS"])
        assert info.value.code == "coordinate_space_mismatch"
        assert dict(info.value.fields) == {
            "operation": "pair_kinship",
            "receiver_type": "PedigreeView",
            "result_type": "RelationshipPairBlock",
        }

    def test_view_block_is_rejected_by_the_graph(self):
        graph = _sib_mating()
        view = graph.view(rows=[4, 3, 2])
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship(view.relationship_pairs(max_degree=1))
        assert info.value.code == "coordinate_space_mismatch"
        assert info.value.fields["receiver_type"] == "PedigreeGraph"

    def test_equivalent_views_are_distinct_receivers(self):
        graph = _sib_mating()
        one, two = graph.view(rows=[2, 3]), graph.view(rows=[2, 3])
        with pytest.raises(PedigreeValidationError) as info:
            two.pair_kinship(one.relationship_pairs(max_degree=1)["FS"])
        assert info.value.code == "coordinate_space_mismatch"

    def test_another_graphs_block_is_rejected(self):
        one, two = _sib_mating(), _sib_mating()
        with pytest.raises(PedigreeValidationError) as info:
            two.pair_kinship(one.relationship_pairs(max_degree=1)["FS"])
        assert info.value.code == "coordinate_space_mismatch"


class TestErrors:
    @pytest.fixture
    def graph(self):
        return _sib_mating()

    def test_block_with_a_second_argument(self, graph):
        block = graph.relationship_pairs(max_degree=1)["FS"]
        with pytest.raises(TypeError):
            graph.pair_kinship(block, [0])

    def test_collection_with_a_second_argument(self, graph):
        with pytest.raises(TypeError):
            graph.pair_kinship(graph.relationship_pairs(max_degree=1), [0])

    def test_rows_without_a_second_argument(self, graph):
        with pytest.raises(TypeError):
            graph.pair_kinship([0, 1])

    def test_length_mismatch(self, graph):
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship([0, 1, 2], [0, 1])
        assert info.value.code == "pair_length_mismatch"
        assert dict(info.value.fields) == {"first_length": 3, "second_length": 2}

    @pytest.mark.parametrize(
        ("first", "second", "argument", "row", "position"),
        [
            ([0, 5], [0, 0], "first_rows", 5, 1),
            ([0, 0], [-1, 0], "second_rows", -1, 0),
            ([np.uint64(2**63)], [0], "first_rows", 2**63, 0),
            ([0], [1e19], "second_rows", 1e19, 0),
        ],
    )
    def test_row_out_of_range_names_the_argument(self, graph, first, second, argument, row, position):
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship(first, second)
        assert info.value.code == "pair_row_out_of_range"
        assert dict(info.value.fields) == {
            "argument": argument,
            "row": row,
            "position": position,
            "n_individuals": graph.n_individuals,
        }

    def test_two_dimensional_rows(self, graph):
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship([[0, 1]], [0])
        assert info.value.code == "invalid_shape"
        assert info.value.fields["field"] == "first_rows"

    def test_fractional_row(self, graph):
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship([0], [1.5])
        assert info.value.code == "invalid_integer_value"
        assert info.value.fields["field"] == "second_rows"

    def test_null_row(self, graph):
        with pytest.raises(PedigreeValidationError) as info:
            graph.pair_kinship([0, None], [0, 0])
        assert info.value.code == "invalid_integer_value"
        assert info.value.fields["value"] == "null"

    def test_memo_capacity_limit_is_a_resource_error(self):
        graph = _graph("deep_inbred_60g")
        first, second = _all_pairs(graph.n_individuals)
        with pytest.raises(ResourceError) as info:
            _run_kernel(*_kernel_inputs(graph, first, second), cap_limit=1 << 16)
        assert info.value.code == "memo_capacity_exceeded"
        assert dict(info.value.fields) == {"operation": "pair_kinship", "capacity": 1 << 16, "maximum": 1 << 16}


class TestLegacyAdapter:
    def test_compute_pair_kinship_returns_the_new_values(self):
        graph = _graph("deep_inbred_60g")
        pairs = graph.extract_pairs(max_degree=MAX_DEGREE)
        legacy = graph.compute_pair_kinship(pairs)
        for code, (first, second) in pairs.items():
            assert legacy[code].dtype == np.float32
            assert legacy[code].tobytes() == graph.pair_kinship(np.asarray(first), np.asarray(second)).tobytes(), code

    def test_from_subsample_rows_are_caller_space(self):
        import polars as pl

        graph = _mz_twins_with_descendants()
        frame = pl.DataFrame(
            {
                "id": np.arange(10),
                "mother": np.asarray(graph.mother),
                "father": np.asarray(graph.father),
                "twin": np.asarray(graph.twin),
                "sex": np.asarray(graph.sex),
            }
        )
        sub = PedigreeGraph.from_subsample(frame, frame.filter(pl.col("id").is_in([8, 9])).reverse())
        assert sub.compute_pair_kinship({"x": (np.array([0]), np.array([1]))})["x"].tolist() == [0.15625]


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        _reset_thread_state()
        yield
        _reset_thread_state()

    def test_public_call_commits_the_budget(self):
        _sib_mating().pair_kinship([0], [1])
        with pytest.raises(RuntimeError):
            configure_threads(3)

    def test_view_call_commits_the_budget(self):
        _sib_mating().view(rows=[0, 1]).pair_kinship([0], [1])
        with pytest.raises(RuntimeError):
            configure_threads(3)

    def test_adapter_leaves_the_budget_uncommitted(self):
        _sib_mating().compute_pair_kinship({"x": (np.array([0]), np.array([1]))})
        configure_threads(3)


@pytest.mark.slow
def test_random_30k_integration():
    fixture = pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"])
    graph = PedigreeGraph(_columns(fixture))
    pairs = graph.relationship_pairs(max_degree=3)
    values = graph.pair_kinship(pairs)
    assert sum(len(block) for block in pairs.values()) > 0
    for code, block in pairs.items():
        assert len(values[code]) == len(block), code
        if len(block):
            assert float(values[code].min()) > 0, code
    block = pairs["1C"]
    assert values["1C"].tobytes() == graph.pair_kinship(block).tobytes()
    assert values["1C"].tobytes() == graph.pair_kinship(block.first_rows, block.second_rows).tobytes()
    rows = np.arange(graph.n_individuals)
    self_kinship = graph.pair_kinship(rows, rows).astype(np.float64)
    assert np.abs(2.0 * self_kinship - 1.0 - graph.compute_inbreeding()).max() <= 2.0**-22
