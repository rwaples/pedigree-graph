"""``PedigreeGraph.relationship_pairs`` and its result types (ADR 0006).

Fixtures come from ``tests/parity/pedigrees.py``; the frozen 0.7.1 pair arrays
in ``tests/data/parity_v0.7.1`` are the parity-locked membership oracle, and
``relationship_predicates.AncestorWalk`` checks orientation without the engine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pedigrees
import pytest
from _support import ASYMMETRIC, CODES, SYMMETRIC, _ped_double_first_cousins
from conftest import FIXTURE_NAMES, FIXTURES, parity_columns, parity_graph
from oracle.relationship_pairs import check_exclusive, dependency_closure
from relationship_predicates import AncestorWalk

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, RelationshipPairs
from pedigree_graph._view import CoordinateToken

PARITY_DIR = Path(__file__).resolve().parent / "parity"
BASELINE_DIR = Path(__file__).resolve().parent / "data" / "parity_v0.7.1"
BASELINE = json.loads((BASELINE_DIR / "manifest.json").read_text())["fixtures"]

if TYPE_CHECKING:
    import polars as pl

    from pedigree_graph import RelationshipPairBlock


def _unordered(first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(np.minimum(first, second).tolist(), np.maximum(first, second).tolist(), strict=True))


def _oriented(block: RelationshipPairBlock) -> list[tuple[int, int]]:
    return list(zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True))


def _frozen_pairs(name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The frozen 0.7.1 pair arrays for *name*, checked against the input hash they were taken from."""
    entry = BASELINE[name]
    assert pedigrees.input_hash(FIXTURES[name]) == entry["input_hash"], name
    with np.load(BASELINE_DIR / entry["file"]) as npz:
        return {code: (npz[f"pairs/{code}/first"], npz[f"pairs/{code}/second"]) for code in CODES}


def _folded_oracle(name: str) -> tuple[dict[str, set[tuple[int, int]]], dict[str, int]]:
    """0.7.1 membership folded by registry precedence; also how many pairs each code lost."""
    frozen = _frozen_pairs(name)
    seen: set[tuple[int, int]] = set()
    folded: dict[str, set[tuple[int, int]]] = {}
    removed: dict[str, int] = {}
    for code in CODES:
        pairs = _unordered(*frozen[code])
        folded[code] = pairs - seen
        removed[code] = len(pairs & seen)
        seen |= pairs
    return folded, removed


@pytest.fixture(scope="module")
def full_results() -> dict[str, RelationshipPairs]:
    return {name: parity_graph(name).relationship_pairs(max_degree=5) for name in FIXTURE_NAMES}


class TestResultShape:
    def test_all_codes_in_registry_order(self, full_results):
        result = full_results["avuncular_and_cousins"]
        assert list(result) == list(CODES)
        assert len(result) == 23
        assert isinstance(result, RelationshipPairs)

    def test_mapping_is_immutable(self, full_results):
        result = RelationshipPairs(dict(full_results["avuncular_and_cousins"]))
        with pytest.raises(TypeError):
            result["MZ"] = result["FS"]  # ty: ignore[invalid-assignment]
        with pytest.raises(TypeError):
            del result["MZ"]  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError):
            result._blocks["MZ"] = result["FS"]  # ty: ignore[invalid-assignment]

    def test_constructor_copies_its_mapping(self, full_results):
        original = full_results["avuncular_and_cousins"]
        blocks = dict(original)
        result = RelationshipPairs(blocks)
        blocks["MZ"] = blocks["FS"]
        assert result["MZ"] is original["MZ"]

    @pytest.mark.parametrize("malformed", ["missing", "reordered"])
    def test_constructor_rejects_invalid_registry_shape(self, full_results, malformed):
        blocks = dict(full_results["avuncular_and_cousins"])
        if malformed == "missing":
            blocks.pop("MZ")
        else:
            blocks = dict(reversed(tuple(blocks.items())))
        with pytest.raises(ValueError, match="registry code in registry order"):
            RelationshipPairs(blocks)

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
            key: np.array(value, copy=True) for key, value in parity_columns(FIXTURES["avuncular_and_cousins"]).items()
        }
        result = PedigreeGraph.from_frame(columns).relationship_pairs(max_degree=5)
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
        graph = parity_graph("avuncular_and_cousins")
        other = parity_graph("avuncular_and_cousins")
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
            parity_graph("nuclear_full_sibs").relationship_pairs(max_degree=1, categories=["FS"])

    def test_neither_selector_is_a_type_error(self):
        with pytest.raises(TypeError):
            parity_graph("nuclear_full_sibs").relationship_pairs()

    def test_bare_string_is_a_type_error(self):
        with pytest.raises(TypeError):
            parity_graph("nuclear_full_sibs").relationship_pairs(categories="FS")

    def test_non_string_code_is_a_type_error(self):
        with pytest.raises(TypeError):
            parity_graph("nuclear_full_sibs").relationship_pairs(categories=["FS", 3])  # ty: ignore[invalid-argument-type]

    def test_unknown_code(self):
        with pytest.raises(PedigreeValidationError) as info:
            parity_graph("nuclear_full_sibs").relationship_pairs(categories=["FS", "zz", "aa"])
        assert info.value.code == "unknown_relationship_category"
        assert info.value.fields["codes"] == ("aa", "zz")

    @pytest.mark.parametrize("max_degree", [-1, 6])
    def test_max_degree_out_of_range(self, max_degree):
        with pytest.raises(PedigreeValidationError) as info:
            parity_graph("nuclear_full_sibs").relationship_pairs(max_degree=max_degree)
        assert info.value.code == "max_degree_out_of_range"
        assert info.value.fields["value"] == max_degree
        assert (info.value.fields["minimum"], info.value.fields["maximum"]) == (0, 5)

    def test_empty_categories_computes_nothing(self):
        result = parity_graph("avuncular_and_cousins").relationship_pairs(categories=())
        assert all(len(block) == 0 and not block.requested for block in result.values())

    def test_max_degree_zero_requests_only_mz(self):
        result = parity_graph("mz_twins_with_children").relationship_pairs(max_degree=0)
        assert [code for code, block in result.items() if block.requested] == ["MZ"]
        assert len(result["MZ"]) == 1
        assert all(len(block) == 0 for code, block in result.items() if code != "MZ")

    @pytest.mark.parametrize("max_degree", range(6))
    def test_requested_flags_match_max_degree(self, max_degree):
        result = parity_graph("avuncular_and_cousins").relationship_pairs(max_degree=max_degree)
        for code, block in result.items():
            assert block.requested == (RELATIONSHIPS[code].degree <= max_degree)

    def test_requested_flags_match_categories(self):
        result = parity_graph("avuncular_and_cousins").relationship_pairs(categories=["2C", "MO", "MO"])
        assert {code for code, block in result.items() if block.requested} == {"2C", "MO"}

    def test_unrequested_block_is_empty_even_when_computed(self):
        result = parity_graph("lineal_five_generations").relationship_pairs(categories=["1C1R"])
        assert not result["GGP"].requested
        assert len(result["GGP"]) == 0
        assert len(parity_graph("lineal_five_generations").relationship_pairs(max_degree=3)["GGP"]) > 0


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
        alone = parity_graph(name).relationship_pairs(categories=[code])[code]
        full = full_results[name][code]
        np.testing.assert_array_equal(alone.first_rows, full.first_rows)
        np.testing.assert_array_equal(alone.second_rows, full.second_rows)


class TestMembershipAndPrecedence:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_matches_the_folded_0_7_1_oracle(self, full_results, name):
        folded, _ = _folded_oracle(name)
        for code, block in full_results[name].items():
            assert _unordered(block.first_rows, block.second_rows) == folded[code], code

    def test_the_fold_removes_pairs_on_the_backcross_fixture(self):
        _, removed = _folded_oracle("backcross_and_selfing_like")
        assert sum(removed.values()) > 0
        assert removed["GP"] > 0


class TestOrientation:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_asymmetric_blocks_satisfy_their_roles(self, full_results, name):
        walk = AncestorWalk(parity_graph(name))
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
        n = parity_graph(name).n_individuals
        for code, block in full_results[name].items():
            keys = np.minimum(block.first_rows, block.second_rows).astype(np.int64) * n + np.maximum(
                block.first_rows, block.second_rows
            )
            assert np.all(np.diff(keys) > 0), code

    def test_every_pair_on_the_shipped_parquet_satisfies_its_predicate(self, small_pedigree: pl.DataFrame):
        graph = PedigreeGraph.from_frame(small_pedigree)
        walk = AncestorWalk(graph)
        pairs = graph.relationship_pairs(max_degree=5)
        np.testing.assert_array_equal(graph.twin_rows[pairs["MZ"].first_rows], pairs["MZ"].second_rows)
        for code in (code for code in CODES if code != "MZ"):
            for first, second in _oriented(pairs[code]):
                assert walk.oriented_pair_is_valid(code, first, second), (code, first, second)

    def test_mo_and_fo_name_the_actual_parent(self, full_results):
        graph = parity_graph("random_1k")
        result = full_results["random_1k"]
        np.testing.assert_array_equal(graph.mother_rows[result["MO"].first_rows], result["MO"].second_rows)
        np.testing.assert_array_equal(graph.father_rows[result["FO"].first_rows], result["FO"].second_rows)


class TestDualValid:
    """random_1k holds H1C1R pairs valid in both orientations through different paths."""

    def test_random_1k_has_dual_valid_pairs_reported_once_lower_row_first(self, full_results):
        graph = parity_graph("random_1k")
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
        graph = PedigreeGraph.from_frame(
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
        check_exclusive(PedigreeGraph.from_frame(small_pedigree).relationship_pairs(max_degree=5))

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


BLOCK_DIGEST_SCRIPT = """
import hashlib, sys
sys.path.insert(0, {parity!r})
import pedigrees
from pedigree_graph import PedigreeGraph, configure_threads
configure_threads({threads})
fx = pedigrees.build_random("random_1k", pedigrees.RANDOM_FIXTURES["random_1k"])
graph = PedigreeGraph.from_frame({{"id": fx["ids"], "mother": fx["mother"], "father": fx["father"], "twin": fx["twin"], "sex": fx["sex"]}})
digest = hashlib.sha256()
for code, block in graph.relationship_pairs(max_degree=5).items():
    digest.update(code.encode())
    digest.update(block.first_rows.tobytes())
    digest.update(block.second_rows.tobytes())
print(digest.hexdigest())
"""


# ---------------------------------------------------------------------------
# Hand-built pedigrees and the independent pandas reference
# ---------------------------------------------------------------------------


def _reference_relationship_pairs(df) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Independent pandas derivation of the seven original categories, the golden for the engine.

    Deliberately pandas: an independent derivation of the same pairs using a
    different toolchain than the production sparse-matrix path. The polars
    fixture converts at this boundary only.
    """
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    ids_arr = df["id"].to_numpy().astype(np.int64)
    id_to_row = np.full(ids_arr.max() + 1, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df), dtype=np.int32)

    def resolve_rows(ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        mask = (ids >= 0) & (ids < len(id_to_row))
        result = np.full(len(ids), -1, dtype=np.int32)
        result[mask] = id_to_row[ids[mask]]
        return result

    pairs = {}

    twins = df[df["twin"] != -1]
    ta = twins["id"].to_numpy().astype(int)
    tb = twins["twin"].to_numpy().astype(int)
    mask = ta < tb
    pairs["MZ"] = (resolve_rows(ta[mask]), resolve_rows(tb[mask]))

    non_twin_nf = df[(df["mother"] != -1) & (df["twin"] == -1)].copy()
    non_twin_nf["_row"] = non_twin_nf.index.to_numpy()

    full_rows_1, full_rows_2 = [], []
    mat_half_rows_1, mat_half_rows_2 = [], []

    sib_counts = non_twin_nf.groupby("mother").size()
    multi_mothers = sib_counts[sib_counts >= 2].index
    mat_sib = non_twin_nf[non_twin_nf["mother"].isin(multi_mothers)]

    if len(mat_sib) > 0:
        mat_pairs = mat_sib[["mother", "father", "_row"]].merge(
            mat_sib[["mother", "father", "_row"]],
            on="mother",
            suffixes=("_1", "_2"),
        )
        mat_pairs = mat_pairs[mat_pairs["_row_1"] < mat_pairs["_row_2"]]
        same_father = mat_pairs["father_1"] == mat_pairs["father_2"]
        full_rows_1.append(mat_pairs.loc[same_father, "_row_1"].to_numpy())
        full_rows_2.append(mat_pairs.loc[same_father, "_row_2"].to_numpy())
        mat_half_rows_1.append(mat_pairs.loc[~same_father, "_row_1"].to_numpy())
        mat_half_rows_2.append(mat_pairs.loc[~same_father, "_row_2"].to_numpy())

    pat_half_rows_1, pat_half_rows_2 = [], []
    pat_counts = non_twin_nf.groupby("father").size()
    multi_fathers = pat_counts[pat_counts >= 2].index
    pat_sib = non_twin_nf[non_twin_nf["father"].isin(multi_fathers)]

    if len(pat_sib) > 0:
        pat_pairs = pat_sib[["mother", "father", "_row"]].merge(
            pat_sib[["mother", "father", "_row"]],
            on="father",
            suffixes=("_1", "_2"),
        )
        pat_pairs = pat_pairs[pat_pairs["_row_1"] < pat_pairs["_row_2"]]
        diff_mother = pat_pairs["mother_1"] != pat_pairs["mother_2"]
        pat_half_rows_1.append(pat_pairs.loc[diff_mother, "_row_1"].to_numpy())
        pat_half_rows_2.append(pat_pairs.loc[diff_mother, "_row_2"].to_numpy())

    pairs["FS"] = (
        np.concatenate(full_rows_1) if full_rows_1 else np.array([], dtype=int),
        np.concatenate(full_rows_2) if full_rows_1 else np.array([], dtype=int),
    )
    pairs["MHS"] = (
        np.concatenate(mat_half_rows_1) if mat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(mat_half_rows_2) if mat_half_rows_1 else np.array([], dtype=int),
    )
    pairs["PHS"] = (
        np.concatenate(pat_half_rows_1) if pat_half_rows_1 else np.array([], dtype=int),
        np.concatenate(pat_half_rows_2) if pat_half_rows_1 else np.array([], dtype=int),
    )

    all_nf = df[df["mother"] != -1]
    child_rows = all_nf.index.to_numpy()
    mother_rows = resolve_rows(all_nf["mother"].to_numpy().astype(int))
    father_rows = resolve_rows(all_nf["father"].to_numpy().astype(int))

    m_valid = mother_rows >= 0
    f_valid = father_rows >= 0
    pairs["MO"] = (child_rows[m_valid], mother_rows[m_valid])
    pairs["FO"] = (child_rows[f_valid], father_rows[f_valid])

    child_ids = all_nf["id"].to_numpy().astype(np.int64)
    mother_ids = all_nf["mother"].to_numpy().astype(np.int64)
    father_ids = all_nf["father"].to_numpy().astype(np.int64)
    n_children = len(child_ids)

    df_mothers_col = df["mother"].to_numpy().astype(np.int64)
    df_fathers_col = df["father"].to_numpy().astype(np.int64)
    mother_row = resolve_rows(mother_ids)
    father_row = resolve_rows(father_ids)

    gp_ids = np.full((n_children, 4), -1, dtype=np.int64)
    m_ok = mother_row >= 0
    gp_ids[m_ok, 0] = df_mothers_col[mother_row[m_ok]]
    gp_ids[m_ok, 1] = df_fathers_col[mother_row[m_ok]]
    f_ok = father_row >= 0
    gp_ids[f_ok, 2] = df_mothers_col[father_row[f_ok]]
    gp_ids[f_ok, 3] = df_fathers_col[father_row[f_ok]]

    gp_child = np.tile(child_ids, 4)
    gp_parent = np.concatenate([mother_ids, mother_ids, father_ids, father_ids])
    gp_gp = np.concatenate([gp_ids[:, 0], gp_ids[:, 1], gp_ids[:, 2], gp_ids[:, 3]])

    valid_gp = gp_gp >= 0
    gp_child = gp_child[valid_gp]
    gp_parent = gp_parent[valid_gp]
    gp_gp = gp_gp[valid_gp]

    sort_idx = np.argsort(gp_gp, kind="mergesort")
    gp_child = gp_child[sort_idx]
    gp_parent = gp_parent[sort_idx]
    gp_gp = gp_gp[sort_idx]

    _, group_starts, group_counts = np.unique(gp_gp, return_index=True, return_counts=True)

    multi = group_counts >= 2
    group_starts = group_starts[multi]
    group_counts = group_counts[multi]

    pair_i_parts = []
    pair_j_parts = []
    for size in np.unique(group_counts):
        gs = group_starts[group_counts == size]
        ii, jj = np.triu_indices(size, k=1)
        all_i = (gs[:, np.newaxis] + ii[np.newaxis, :]).ravel()
        all_j = (gs[:, np.newaxis] + jj[np.newaxis, :]).ravel()
        pair_i_parts.append(all_i)
        pair_j_parts.append(all_j)

    pair_i = np.concatenate(pair_i_parts)
    pair_j = np.concatenate(pair_j_parts)

    diff_parent = gp_parent[pair_i] != gp_parent[pair_j]
    c1_raw = gp_child[pair_i[diff_parent]]
    c2_raw = gp_child[pair_j[diff_parent]]

    c1 = np.minimum(c1_raw, c2_raw)
    c2 = np.maximum(c1_raw, c2_raw)

    max_id = int(c2.max()) + 1
    pair_keys = c1.astype(np.int64) * max_id + c2.astype(np.int64)
    unique_keys = np.unique(pair_keys)
    c1_final = unique_keys // max_id
    c2_final = unique_keys % max_id

    c_idx1 = resolve_rows(c1_final)
    c_idx2 = resolve_rows(c2_final)
    c_valid = (c_idx1 >= 0) & (c_idx2 >= 0)
    pairs["1C"] = (c_idx1[c_valid], c_idx2[c_valid])

    return pairs


def _pairs_to_set(idx1, idx2):
    """Convert pair arrays to a set of sorted tuples for comparison."""
    return {(min(a, b), max(a, b)) for a, b in zip(idx1.tolist(), idx2.tolist(), strict=True)}


def _pairs_to_set(idx1, idx2):
    """Convert pair arrays to a set of sorted tuples for comparison."""
    return {(min(a, b), max(a, b)) for a, b in zip(idx1.tolist(), idx2.tolist(), strict=True)}


class TestPandasReference:
    """The engine produces the pandas reference's pair sets for the original seven categories."""

    def test_golden_pairs_match(self, small_pedigree):
        """Identical sets for the six sibling/parent keys; the reference's cousins are a subset of 1C | H1C."""
        df = small_pedigree
        reference = _reference_relationship_pairs(df)
        new = PedigreeGraph.from_frame(df).relationship_pairs(max_degree=4)

        exact_keys = [
            "MZ",
            "FS",
            "MHS",
            "PHS",
            "MO",
            "FO",
        ]
        for key in exact_keys:
            reference_set = _pairs_to_set(*reference[key])
            new_set = _pairs_to_set(*new[key])
            assert reference_set == new_set, (
                f"{key}: reference has {len(reference_set)} pairs, engine has {len(new_set)} pairs, "
                f"diff: {reference_set.symmetric_difference(new_set)}"
            )

        # Cousins: the reference lumps full 1C and half-1C into one category.
        # The engine splits them: pairs["1C"] = full only (>= 2 shared GPs),
        # pairs["H1C"] = half only (1 shared GP). The union should match the reference
        # (after removing self-pairs and sibling-pairs from the reference).
        reference_cousins = _pairs_to_set(*reference["1C"])
        new_1c = _pairs_to_set(*new["1C"])
        new_h1c = _pairs_to_set(*new["H1C"])
        new_all_cousins = new_1c | new_h1c
        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()
        # Filter out self-pairs and sibling-pairs from the reference
        reference_proper = set()
        for a, b in reference_cousins:
            if a == b:
                continue
            if mother[a] == mother[b] or father[a] == father[b]:
                continue
            reference_proper.add((a, b))
        assert reference_proper <= new_all_cousins, (
            f"1st cousin: reference has {len(reference_proper - new_all_cousins)} pairs not in the engine"
        )
        # 1C and H1C must be disjoint
        assert not (new_1c & new_h1c), f"1C and H1C overlap: {len(new_1c & new_h1c)} pairs"


class TestKnownTinyPedigree:
    """The double-first-cousin pedigree, with every pair counted by hand."""

    @pytest.fixture
    def tiny_pedigree(self):
        """3-generation pedigree:
        Gen 0: 4 founders (0=F, 1=M, 2=F, 3=M)
        Gen 1: 4 offspring
          - 4,5 children of (0,1) — full sibs
          - 6   child of (2,3)
          - 7   child of (2,3) — full sib with 6
        Gen 2: 3 offspring
          - 8   child of (4=F, 6=M) — 4 is female, 6 is male
          - 9   child of (5=F, 7=M) — 5 is female, 7 is male
          - 10  child of (4=F, 6=M) — full sib with 8

        Expected:
          Full sibs: (4,5), (6,7), (8,10) = 3 pairs
          1st cousins: (8,9), (9,10) = 2 pairs (parents 4&5 are full sibs, parents 6&7 are full sibs)
          Mother-offspring: (4,0), (5,0), (6,2), (7,2), (8,4), (9,5), (10,4) = 7
          Father-offspring: (4,1), (5,1), (6,3), (7,3), (8,6), (9,7), (10,6) = 7
          Grandparent-grandchild: 8→{0,1,2,3}, 9→{0,1,2,3}, 10→{0,1,2,3} = 12
          Avuncular: 5 is aunt of 8,10; 4 is aunt of 9; 7 is uncle of 8,10; 6 is uncle of 9
                     = 6 pairs
        """
        return _ped_double_first_cousins()

    def test_full_sib_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        sib_set = _pairs_to_set(*pairs["FS"])
        expected = {(4, 5), (6, 7), (8, 10)}
        assert sib_set == expected, f"Got {sib_set}"

    def test_mother_offspring_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        mo_set = _pairs_to_set(*pairs["MO"])
        assert len(mo_set) == 7

    def test_father_offspring_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        fo_set = _pairs_to_set(*pairs["FO"])
        assert len(fo_set) == 7

    def test_cousin_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        cousin_set = _pairs_to_set(*pairs["1C"])
        # 8's parents: (4, 6). 9's parents: (5, 7).
        # 4 & 5 share grandparents 0,1. 6 & 7 share grandparents 2,3.
        # So 8 and 9 are double 1st cousins (share all 4 grandparents).
        # 10's parents: (4, 6) same as 8, so 10 is full sib of 8.
        # 10 and 9: parents (4,6) vs (5,7) — same as 8 vs 9
        expected = {(8, 9), (9, 10)}
        assert cousin_set == expected, f"Got {cousin_set}"

    def test_grandparent_grandchild_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        gp_set = _pairs_to_set(*pairs["GP"])
        # 8 → grandparents 0,1,2,3
        # 9 → grandparents 0,1,2,3
        # 10 → grandparents 0,1,2,3
        expected = {
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (0, 9),
            (1, 9),
            (2, 9),
            (3, 9),
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 10),
        }
        assert gp_set == expected, f"Got {gp_set}, expected {expected}"

    def test_avuncular_count(self, tiny_pedigree):
        pairs = PedigreeGraph.from_frame(tiny_pedigree).relationship_pairs(max_degree=3)
        avunc_set = _pairs_to_set(*pairs["Av"])
        # 5 is full sib of 4 (mother of 8, 10) → 5 is aunt of 8, 10
        # 4 is full sib of 5 (mother of 9) → 4 is aunt of 9
        # 7 is full sib of 6 (father of 8, 10) → 7 is uncle of 8, 10
        # 6 is full sib of 7 (father of 9) → 6 is uncle of 9
        expected = {(5, 8), (5, 10), (4, 9), (7, 8), (7, 10), (6, 9)}
        assert avunc_set == expected, f"Got {avunc_set}"


class TestSecondCousinFullVsHalf:
    def test_only_the_full_second_cousins_are_2c(self):
        """Full 2C share two great-grandparents through a mated pair; half 2C share one and are excluded.

        Full branch: 2,3 full sibs of (0,1); 6=child(4,2), 7=child(5,3) are
        full first cousins; 10=child(8,6), 11=child(9,7) share GGPs {0,1}.
        Half branch: 15=child(12,13), 16=child(12,14) are maternal half sibs;
        19=child(17,15), 20=child(18,16); 23=child(21,19), 24=child(22,20)
        share only GGP 12.
        """
        # fmt: off
        mother = [-1, -1, 0, 0, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1, -1, 12, 12, -1, -1, 17, 18, -1, -1, 21, 22]
        father = [-1, -1, 1, 1, -1, -1, 2, 3, -1, -1, 6, 7, -1, -1, -1, 13, 14, -1, -1, 15, 16, -1, -1, 19, 20]
        # fmt: on
        graph = PedigreeGraph.from_frame({"id": np.arange(25), "mother": np.array(mother), "father": np.array(father)})
        assert _unordered(*graph.relationship_pairs(max_degree=5)["2C"]) == {(10, 11)}


class TestThreads:
    """The blocks are the same for every thread budget; each budget runs in a fresh interpreter."""

    @staticmethod
    def _digest_in_fresh_process(threads: int) -> str:
        env = dict(os.environ)
        env.pop("PEDIGREE_GRAPH_THREADS", None)
        script = BLOCK_DIGEST_SCRIPT.format(parity=str(PARITY_DIR), threads=threads)
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        return proc.stdout.strip()

    def test_thread_budget_does_not_change_the_blocks(self):
        assert self._digest_in_fresh_process(1) == self._digest_in_fresh_process(4)


def _digest(result: RelationshipPairs) -> str:
    digest = hashlib.sha256()
    for code, block in result.items():
        digest.update(code.encode())
        digest.update(block.first_rows.tobytes())
        digest.update(block.second_rows.tobytes())
    return digest.hexdigest()


def test_repeated_calls_are_bit_identical(full_results):
    graph = parity_graph("random_1k")
    assert _digest(graph.relationship_pairs(max_degree=5)) == _digest(full_results["random_1k"])
    assert _digest(graph.relationship_pairs(max_degree=5)) == _digest(full_results["random_1k"])
