"""``PedigreeGraph.relationship_pairs`` and its result types (slice 4a, ADR 0006).

Fixtures come from ``tests/parity/pedigrees.py``; the 0.7.1 adapter
``extract_pairs`` is the parity-locked membership oracle, and
``relationship_predicates.AncestorWalk`` checks orientation without the engine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from relationship_predicates import AncestorWalk

import pedigree_graph._pair_extractor as pair_extractor
from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, RelationshipPairs
from pedigree_graph._pair_extractor import check_exclusive, dependency_closure
from pedigree_graph._threads import _reset_thread_state, configure_threads
from pedigree_graph._view import CoordinateToken

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import pedigrees

if TYPE_CHECKING:
    import polars as pl

    from pedigree_graph import RelationshipPairBlock

PARITY_DIR = Path(__file__).resolve().parent / "parity"
CODES = tuple(RELATIONSHIPS)
ASYMMETRIC = tuple(code for code, category in RELATIONSHIPS.items() if not category.symmetric)
SYMMETRIC = tuple(code for code, category in RELATIONSHIPS.items() if category.symmetric)


def _fixtures() -> dict[str, dict[str, np.ndarray]]:
    fixtures = dict(pedigrees.motif_fixtures())
    fixtures["random_1k"] = pedigrees.build_random("random_1k", pedigrees.RANDOM_FIXTURES["random_1k"])
    return fixtures


FIXTURES = _fixtures()
FIXTURE_NAMES = sorted(FIXTURES)


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


def _unordered(first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(np.minimum(first, second).tolist(), np.maximum(first, second).tolist(), strict=True))


def _oriented(block: RelationshipPairBlock) -> list[tuple[int, int]]:
    return list(zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True))


def _folded_oracle(graph: PedigreeGraph) -> tuple[dict[str, set[tuple[int, int]]], dict[str, int]]:
    """0.7.1 membership folded by registry precedence; also how many pairs each code lost."""
    legacy = graph.extract_pairs(max_degree=5)
    seen: set[tuple[int, int]] = set()
    folded: dict[str, set[tuple[int, int]]] = {}
    removed: dict[str, int] = {}
    for code in CODES:
        pairs = _unordered(*legacy[code])
        folded[code] = pairs - seen
        removed[code] = len(pairs & seen)
        seen |= pairs
    return folded, removed


@pytest.fixture(scope="module")
def full_results() -> dict[str, RelationshipPairs]:
    return {name: _graph(name).relationship_pairs(max_degree=5) for name in FIXTURE_NAMES}


class TestResultShape:
    def test_all_codes_in_registry_order(self, full_results):
        result = full_results["avuncular_and_cousins"]
        assert list(result) == list(CODES)
        assert len(result) == 23
        assert isinstance(result, RelationshipPairs)

    def test_mapping_is_immutable(self, full_results):
        result = full_results["avuncular_and_cousins"]
        with pytest.raises(TypeError):
            result["MZ"] = result["FS"]  # ty: ignore[invalid-assignment]
        with pytest.raises(TypeError):
            del result["MZ"]  # ty: ignore[invalid-argument-type]

    def test_block_is_frozen(self, full_results):
        block = full_results["avuncular_and_cousins"]["FS"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            block.requested = False  # ty: ignore[invalid-assignment]

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_arrays_are_owned_int32_readonly(self, full_results, name):
        for block in full_results[name].values():
            for rows in block:
                assert rows.dtype == np.int32
                assert rows.flags.c_contiguous
                assert not rows.flags.writeable

    def test_mutating_the_input_afterwards_changes_nothing(self):
        columns = {
            key: np.array(value, copy=True) for key, value in _columns(FIXTURES["avuncular_and_cousins"]).items()
        }
        result = PedigreeGraph(columns).relationship_pairs(max_degree=5)
        before = {code: _oriented(block) for code, block in result.items()}
        for value in columns.values():
            value[:] = 0
        assert {code: _oriented(block) for code, block in result.items()} == before

    def test_block_unpacks_and_has_len(self, full_results):
        block = full_results["avuncular_and_cousins"]["Av"]
        first, second = block
        assert first is block.first_rows
        assert second is block.second_rows
        assert len(block) == len(first) == 2

    def test_roles_and_code_come_from_the_registry(self, full_results):
        for code, block in full_results["avuncular_and_cousins"].items():
            assert block.code == code
            assert block.category is RELATIONSHIPS[code]
            assert block.first_role == RELATIONSHIPS[code].first_role
            assert block.second_role == RELATIONSHIPS[code].second_role

    def test_no_public_attribute_is_a_token(self, full_results):
        for owner in (full_results["avuncular_and_cousins"], full_results["avuncular_and_cousins"]["FS"]):
            public = [name for name in dir(owner) if not name.startswith("_")]
            assert public
            assert not any(isinstance(getattr(owner, name), CoordinateToken) for name in public)

    def test_blocks_carry_the_exact_receiver_token(self):
        graph = _graph("avuncular_and_cousins")
        other = _graph("avuncular_and_cousins")
        result = graph.relationship_pairs(max_degree=5)
        assert all(block._coordinate_token is graph._coordinate_token for block in result.values())
        assert graph._coordinate_token is not other._coordinate_token
        assert result["FS"]._coordinate_token is not other._coordinate_token

    def test_repr_lists_requested_counts(self, full_results):
        text = repr(full_results["nuclear_full_sibs"])
        assert text.startswith("RelationshipPairs(")
        assert "FS=3" in text

    def test_root_exports(self):
        import pedigree_graph

        assert "RelationshipPairs" in pedigree_graph.__all__
        assert "RelationshipPairBlock" in pedigree_graph.__all__
        assert pedigree_graph.relationships.RelationshipPairs is RelationshipPairs


class TestSelectors:
    def test_both_selectors_is_a_type_error(self):
        with pytest.raises(TypeError):
            _graph("nuclear_full_sibs").relationship_pairs(max_degree=1, categories=["FS"])

    def test_neither_selector_is_a_type_error(self):
        with pytest.raises(TypeError):
            _graph("nuclear_full_sibs").relationship_pairs()

    def test_bare_string_is_a_type_error(self):
        with pytest.raises(TypeError):
            _graph("nuclear_full_sibs").relationship_pairs(categories="FS")

    def test_non_string_code_is_a_type_error(self):
        with pytest.raises(TypeError):
            _graph("nuclear_full_sibs").relationship_pairs(categories=["FS", 3])  # ty: ignore[invalid-argument-type]

    def test_unknown_code(self):
        with pytest.raises(PedigreeValidationError) as info:
            _graph("nuclear_full_sibs").relationship_pairs(categories=["FS", "zz", "aa"])
        assert info.value.code == "unknown_relationship_category"
        assert info.value.fields["codes"] == ("aa", "zz")

    @pytest.mark.parametrize("max_degree", [-1, 6])
    def test_max_degree_out_of_range(self, max_degree):
        with pytest.raises(PedigreeValidationError) as info:
            _graph("nuclear_full_sibs").relationship_pairs(max_degree=max_degree)
        assert info.value.code == "max_degree_out_of_range"
        assert info.value.fields["value"] == max_degree
        assert (info.value.fields["minimum"], info.value.fields["maximum"]) == (0, 5)

    def test_empty_categories_computes_nothing(self):
        result = _graph("avuncular_and_cousins").relationship_pairs(categories=())
        assert all(len(block) == 0 and not block.requested for block in result.values())

    def test_max_degree_zero_requests_only_mz(self):
        result = _graph("mz_twins_with_children").relationship_pairs(max_degree=0)
        assert [code for code, block in result.items() if block.requested] == ["MZ"]
        assert len(result["MZ"]) == 1
        assert all(len(block) == 0 for code, block in result.items() if code != "MZ")

    @pytest.mark.parametrize("max_degree", range(6))
    def test_requested_flags_match_max_degree(self, max_degree):
        result = _graph("avuncular_and_cousins").relationship_pairs(max_degree=max_degree)
        for code, block in result.items():
            assert block.requested == (RELATIONSHIPS[code].degree <= max_degree)

    def test_requested_flags_match_categories(self):
        result = _graph("avuncular_and_cousins").relationship_pairs(categories=["2C", "MO", "MO"])
        assert {code for code, block in result.items() if block.requested} == {"2C", "MO"}

    def test_unrequested_block_is_empty_even_when_computed(self):
        result = _graph("lineal_five_generations").relationship_pairs(categories=["1C1R"])
        assert not result["GGP"].requested
        assert len(result["GGP"]) == 0
        assert len(_graph("lineal_five_generations").relationship_pairs(max_degree=3)["GGP"]) > 0


class TestDependencyClosure:
    def test_closure_of_1c1r_is_the_registry_prefix(self):
        expected = frozenset(
            (
                "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv", "1C",
                "GGGP", "HGAv", "GGAv", "H1C", "1C1R",
            )
        )  # fmt: skip
        assert dependency_closure(frozenset({"1C1R"})) == expected

    def test_closure_of_nothing_is_nothing(self):
        assert dependency_closure(frozenset()) == frozenset()

    def test_closure_is_idempotent_and_contains_its_input(self):
        for code in CODES:
            closure = dependency_closure(frozenset({code}))
            assert code in closure
            assert dependency_closure(closure) == closure

    @pytest.mark.parametrize("code", ["1C1R", "H1C", "2C", "HAv"])
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_single_category_equals_the_full_run(self, full_results, name, code):
        alone = _graph(name).relationship_pairs(categories=[code])[code]
        full = full_results[name][code]
        np.testing.assert_array_equal(alone.first_rows, full.first_rows)
        np.testing.assert_array_equal(alone.second_rows, full.second_rows)


class TestMembershipAndPrecedence:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_matches_the_folded_0_7_1_oracle(self, full_results, name):
        folded, _ = _folded_oracle(_graph(name))
        for code, block in full_results[name].items():
            assert _unordered(block.first_rows, block.second_rows) == folded[code], code

    def test_the_fold_removes_pairs_on_the_backcross_fixture(self):
        _, removed = _folded_oracle(_graph("backcross_and_selfing_like"))
        assert sum(removed.values()) > 0
        assert removed["GP"] > 0


class TestOrientation:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_asymmetric_blocks_satisfy_their_roles(self, full_results, name):
        walk = AncestorWalk(_graph(name))
        for code in ASYMMETRIC:
            for first, second in _oriented(full_results[name][code]):
                assert walk.oriented_pair_is_valid(code, first, second), (code, first, second)

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_symmetric_blocks_are_canonical(self, full_results, name):
        for code in SYMMETRIC:
            block = full_results[name][code]
            assert np.all(block.first_rows < block.second_rows), code

    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_blocks_are_sorted_by_canonical_key(self, full_results, name):
        n = _graph(name).n_individuals
        for code, block in full_results[name].items():
            keys = np.minimum(block.first_rows, block.second_rows).astype(np.int64) * n + np.maximum(
                block.first_rows, block.second_rows
            )
            assert np.all(np.diff(keys) > 0), code

    def test_mo_and_fo_name_the_actual_parent(self, full_results):
        graph = _graph("random_1k")
        result = full_results["random_1k"]
        np.testing.assert_array_equal(graph.mother_rows[result["MO"].first_rows], result["MO"].second_rows)
        np.testing.assert_array_equal(graph.father_rows[result["FO"].first_rows], result["FO"].second_rows)


class TestDualValid:
    """random_1k holds H1C1R pairs valid in both orientations through different paths."""

    def test_random_1k_has_dual_valid_pairs_reported_once_lower_row_first(self, full_results):
        graph = _graph("random_1k")
        walk = AncestorWalk(graph)
        block = full_results["random_1k"]["H1C1R"]
        duals = [(a, b) for a, b in _oriented(block) if walk.dual_valid("H1C1R", a, b)]
        assert duals
        assert all(a < b for a, b in duals)
        unordered = _unordered(block.first_rows, block.second_rows)
        assert len(unordered) == len(block)

    def test_hand_built_dual_valid_half_avuncular_pair(self):
        # 4's mother 2 is a paternal half sib of 5 (father 0), and 5's mother 3
        # is a paternal half sib of 4 (father 1), so (4, 5) is HAv both ways.
        graph = PedigreeGraph(
            {
                "id": np.array([0, 1, 2, 3, 4, 5]),
                "mother": np.array([-1, -1, -1, -1, 2, 3]),
                "father": np.array([-1, -1, 0, 1, 1, 0]),
            }
        )
        walk = AncestorWalk(graph)
        assert walk.dual_valid("HAv", 4, 5)
        result = graph.relationship_pairs(max_degree=5)
        check_exclusive(result)
        assert _oriented(result["HAv"]) == [(4, 5)]


class TestCheckExclusive:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_passes_on_every_fixture(self, full_results, name):
        check_exclusive(full_results[name])

    def test_passes_on_the_shipped_parquet(self, small_pedigree: pl.DataFrame):
        check_exclusive(PedigreeGraph(small_pedigree).relationship_pairs(max_degree=5))

    def test_fails_when_a_pair_is_in_two_blocks(self, full_results):
        result = full_results["avuncular_and_cousins"]
        blocks = dict(result)
        source = blocks["FS"]
        blocks["MHS"] = dataclasses.replace(blocks["MHS"], first_rows=source.first_rows, second_rows=source.second_rows)
        with pytest.raises(AssertionError, match="both FS and MHS"):
            check_exclusive(RelationshipPairs(blocks))

    def test_fails_on_both_orientations_in_one_block(self, full_results):
        result = full_results["avuncular_and_cousins"]
        blocks = dict(result)
        first, second = blocks["Av"]
        blocks["Av"] = dataclasses.replace(
            blocks["Av"],
            first_rows=np.concatenate([first, second[:1]]),
            second_rows=np.concatenate([second, first[:1]]),
        )
        with pytest.raises(AssertionError):
            check_exclusive(RelationshipPairs(blocks))

    def test_fails_on_an_unsorted_block(self, full_results):
        result = full_results["avuncular_and_cousins"]
        blocks = dict(result)
        first, second = blocks["MO"]
        blocks["MO"] = dataclasses.replace(blocks["MO"], first_rows=first[::-1].copy(), second_rows=second[::-1].copy())
        with pytest.raises(AssertionError, match="not strictly sorted"):
            check_exclusive(RelationshipPairs(blocks))

    def test_env_var_runs_the_check_at_runtime(self, monkeypatch):
        calls: list[RelationshipPairs] = []
        monkeypatch.setattr(pair_extractor, "check_exclusive", calls.append)
        monkeypatch.delenv("PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY", raising=False)
        _graph("nuclear_full_sibs").relationship_pairs(max_degree=1)
        assert calls == []
        monkeypatch.setenv("PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY", "1")
        result = _graph("nuclear_full_sibs").relationship_pairs(max_degree=1)
        assert calls == [result]

    def test_env_var_surfaces_a_corrupted_engine_result(self, monkeypatch):
        monkeypatch.setenv("PEDIGREE_GRAPH_DEBUG_EXCLUSIVITY", "1")
        original = pair_extractor.MatrixPairExtractor.extract

        def unsorted(self, codes):
            pairs = original(self, codes)
            first, second = pairs["MO"]
            pairs["MO"] = (first[::-1].copy(), second[::-1].copy())
            return pairs

        monkeypatch.setattr(pair_extractor.MatrixPairExtractor, "extract", unsorted)
        with pytest.raises(AssertionError, match="MO"):
            _graph("nuclear_full_sibs").relationship_pairs(max_degree=1)


BLOCK_DIGEST_SCRIPT = """
import hashlib, sys
sys.path.insert(0, {parity!r})
import pedigrees
from pedigree_graph import PedigreeGraph, configure_threads
configure_threads({threads})
fx = pedigrees.build_random("random_1k", pedigrees.RANDOM_FIXTURES["random_1k"])
graph = PedigreeGraph({{"id": fx["ids"], "mother": fx["mother"], "father": fx["father"], "twin": fx["twin"], "sex": fx["sex"]}})
digest = hashlib.sha256()
for code, block in graph.relationship_pairs(max_degree=5).items():
    digest.update(code.encode())
    digest.update(block.first_rows.tobytes())
    digest.update(block.second_rows.tobytes())
print(digest.hexdigest())
"""


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self):
        _reset_thread_state()
        yield
        _reset_thread_state()

    @staticmethod
    def _digest_in_fresh_process(threads: int) -> str:
        env = dict(os.environ)
        env.pop("PEDIGREE_GRAPH_THREADS", None)
        script = BLOCK_DIGEST_SCRIPT.format(parity=str(PARITY_DIR), threads=threads)
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        return proc.stdout.strip()

    def test_thread_budget_does_not_change_the_blocks(self):
        assert self._digest_in_fresh_process(1) == self._digest_in_fresh_process(4)

    def test_in_process_budget_of_four_matches_the_one_thread_result(self, full_results, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        configure_threads(4)
        result = _graph("random_1k").relationship_pairs(max_degree=5)
        reference = full_results["random_1k"]
        for code in CODES:
            np.testing.assert_array_equal(result[code].first_rows, reference[code].first_rows)
            np.testing.assert_array_equal(result[code].second_rows, reference[code].second_rows)


def _digest(result: RelationshipPairs) -> str:
    digest = hashlib.sha256()
    for code, block in result.items():
        digest.update(code.encode())
        digest.update(block.first_rows.tobytes())
        digest.update(block.second_rows.tobytes())
    return digest.hexdigest()


def test_repeated_calls_are_bit_identical(full_results):
    graph = _graph("random_1k")
    assert _digest(graph.relationship_pairs(max_degree=5)) == _digest(full_results["random_1k"])
    assert _digest(graph.relationship_pairs(max_degree=5)) == _digest(full_results["random_1k"])
