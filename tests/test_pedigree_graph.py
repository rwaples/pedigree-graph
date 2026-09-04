"""Tests for pedigree_graph relationship extraction."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pedigree_graph import PedigreeGraph, PedigreeValidationError
from pedigree_graph._kinship_pairwise import (
    _pairwise_kinship_py,
    _pairwise_kinship_with_stats,
    pairwise_kinship,
)

logger = logging.getLogger(__name__)


def _extract_relationship_pairs_legacy(df, seed: int = 42) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Legacy implementation kept for golden testing.

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

    unique_gp_arr = np.unique(gp_gp)
    if len(unique_gp_arr) > 100000:
        logger.info(
            "extract_relationship_pairs: %d grandparents exceed 100K cap, sampling subset",
            len(unique_gp_arr),
        )
        rng = np.random.default_rng(seed)
        selected_gp = rng.choice(unique_gp_arr, 100000, replace=False)
        gp_mask = np.isin(gp_gp, selected_gp)
        gp_child = gp_child[gp_mask]
        gp_parent = gp_parent[gp_mask]
        gp_gp = gp_gp[gp_mask]

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


class TestGoldenComparison:
    """New implementation must produce identical pair sets as legacy for original 7 categories."""

    def test_golden_pairs_match(self, small_pedigree):
        """Run old and new on the same pedigree; assert identical pairs for original 7 keys.

        For cousins, the legacy version applies a 100K grandparent cap (subsamples),
        so we only check that legacy is a subset of new. For the small fixture
        (N=1000, G=3), the cap shouldn't trigger, so they should be equal.
        """
        df = small_pedigree
        legacy = _extract_relationship_pairs_legacy(df, seed=42)
        new = PedigreeGraph(df).extract_pairs(max_degree=4)

        exact_keys = [
            "MZ",
            "FS",
            "MHS",
            "PHS",
            "MO",
            "FO",
        ]
        for key in exact_keys:
            legacy_set = _pairs_to_set(*legacy[key])
            new_set = _pairs_to_set(*new[key])
            assert legacy_set == new_set, (
                f"{key}: legacy has {len(legacy_set)} pairs, new has {len(new_set)} pairs, "
                f"diff: {legacy_set.symmetric_difference(new_set)}"
            )

        # Cousins: legacy lumps full 1C and half-1C into one category.
        # New implementation splits them: pairs["1C"] = full only (>= 2 shared GPs),
        # pairs["H1C"] = half only (1 shared GP). The union should match legacy
        # (after removing self-pairs and sibling-pairs from legacy).
        legacy_cousins = _pairs_to_set(*legacy["1C"])
        new_1c = _pairs_to_set(*new["1C"])
        new_h1c = _pairs_to_set(*new.get("H1C", (np.array([]), np.array([]))))
        new_all_cousins = new_1c | new_h1c
        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()
        # Filter out self-pairs and sibling-pairs from legacy
        legacy_proper = set()
        for a, b in legacy_cousins:
            if a == b:
                continue
            if mother[a] == mother[b] or father[a] == father[b]:
                continue
            legacy_proper.add((a, b))
        assert legacy_proper <= new_all_cousins, (
            f"1st cousin: legacy has {len(legacy_proper - new_all_cousins)} pairs not in new"
        )
        # 1C and H1C must be disjoint
        assert not (new_1c & new_h1c), f"1C and H1C overlap: {len(new_1c & new_h1c)} pairs"


class TestNewRelationships:
    """Test the 3 new relationship categories."""

    def test_has_new_keys(self, small_pedigree):
        pairs = PedigreeGraph(small_pedigree).extract_pairs()
        assert "GP" in pairs
        assert "Av" in pairs
        assert "2C" in pairs

    def test_grandparent_grandchild_structure(self, small_pedigree):
        """Each grandparent-grandchild pair must be separated by 2 generations."""
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        gc, gp = pairs["GP"]
        if len(gc) == 0:
            pytest.skip("No grandparent-grandchild pairs found")

        gen = df["generation"].to_numpy()
        gen_diff = np.abs(gen[gc] - gen[gp])
        assert np.all(gen_diff == 2), f"Generation diffs: {np.unique(gen_diff)}"

    def test_grandparent_grandchild_ancestry(self, small_pedigree):
        """Each grandchild must have the grandparent as a parent of a parent."""
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        gc_arr, gp_arr = pairs["GP"]
        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()

        for gc, gp in zip(gc_arr[:100], gp_arr[:100], strict=True):
            m = mother[gc]
            f = father[gc]
            grandparents = set()
            if m >= 0:
                if mother[m] >= 0:
                    grandparents.add(mother[m])
                if father[m] >= 0:
                    grandparents.add(father[m])
            if f >= 0:
                if mother[f] >= 0:
                    grandparents.add(mother[f])
                if father[f] >= 0:
                    grandparents.add(father[f])
            assert gp in grandparents, f"gc={gc}, gp={gp}, actual grandparents={grandparents}"

    def test_avuncular_structure(self, small_pedigree):
        """Avuncular pairs span exactly 1 generation."""
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        a1, a2 = pairs["Av"]
        if len(a1) == 0:
            pytest.skip("No avuncular pairs found")

        gen = df["generation"].to_numpy()
        gen_diff = np.abs(gen[a1] - gen[a2])
        # Avuncular: uncle/aunt is same gen as parent, so 1 gen from nephew/niece
        assert np.all(gen_diff <= 1), f"Generation diffs: {np.unique(gen_diff)}"


class TestStructuralCorrectness:
    """Verify structural properties of extracted pairs."""

    def test_full_sibs_share_both_parents(self, small_pedigree):
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        idx1, idx2 = pairs["FS"]
        if len(idx1) == 0:
            pytest.skip("No full sib pairs")

        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()
        assert np.all(mother[idx1] == mother[idx2])
        assert np.all(father[idx1] == father[idx2])

    def test_half_sibs_share_exactly_one_parent(self, small_pedigree):
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()

        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()

        for key in ["MHS", "PHS"]:
            idx1, idx2 = pairs[key]
            if len(idx1) == 0:
                continue

            same_m = mother[idx1] == mother[idx2]
            same_f = father[idx1] == father[idx2]

            if key == "MHS":
                assert np.all(same_m), "Maternal half sibs must share mother"
                assert np.all(~same_f), "Maternal half sibs must NOT share father"
            else:
                assert np.all(same_f), "Paternal half sibs must share father"
                assert np.all(~same_m), "Paternal half sibs must NOT share mother"

    def test_cousins_share_grandparent(self, small_pedigree):
        """Every 1st cousin pair must share at least one grandparent and not be a self-pair."""
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        idx1, idx2 = pairs["1C"]
        if len(idx1) == 0:
            pytest.skip("No cousin pairs")

        mother = df["mother"].to_numpy()
        father = df["father"].to_numpy()

        for i in range(min(200, len(idx1))):
            a, b = idx1[i], idx2[i]
            assert a != b, f"Self-pair found: ({a}, {b})"

            # Share a grandparent
            gp_a = set()
            for p in [mother[a], father[a]]:
                if p >= 0:
                    if mother[p] >= 0:
                        gp_a.add(mother[p])
                    if father[p] >= 0:
                        gp_a.add(father[p])
            gp_b = set()
            for p in [mother[b], father[b]]:
                if p >= 0:
                    if mother[p] >= 0:
                        gp_b.add(mother[p])
                    if father[p] >= 0:
                        gp_b.add(father[p])
            assert gp_a & gp_b, f"Cousins {a},{b} share no grandparent"

    def test_no_pair_overlap_within_sibling_types(self, small_pedigree):
        """Full sib, maternal half sib, paternal half sib should be mutually exclusive."""
        pairs = PedigreeGraph(small_pedigree).extract_pairs()
        sib_keys = ["FS", "MHS", "PHS"]
        sib_sets = {k: _pairs_to_set(*pairs[k]) for k in sib_keys}

        for i in range(len(sib_keys)):
            for j in range(i + 1, len(sib_keys)):
                overlap = sib_sets[sib_keys[i]] & sib_sets[sib_keys[j]]
                assert len(overlap) == 0, f"Overlap between '{sib_keys[i]}' and '{sib_keys[j]}': {len(overlap)} pairs"

    def test_no_pair_overlap_twins_and_siblings(self, small_pedigree):
        """MZ twins should not also appear as siblings."""
        pairs = PedigreeGraph(small_pedigree).extract_pairs()
        twin_set = _pairs_to_set(*pairs["MZ"])
        for key in ["FS", "MHS", "PHS"]:
            sib_set = _pairs_to_set(*pairs[key])
            overlap = twin_set & sib_set
            assert len(overlap) == 0, f"Overlap between 'MZ twin' and '{key}': {len(overlap)} pairs"


class TestNoSubsamplingLoss:
    """The new implementation should not subsample cousins."""

    def test_exact_cousin_count(self, small_pedigree):
        """Verify cousin count is deterministic (no RNG-dependent cap)."""
        pairs1 = PedigreeGraph(small_pedigree).extract_pairs()
        pairs2 = PedigreeGraph(small_pedigree).extract_pairs()
        assert len(pairs1["1C"][0]) == len(pairs2["1C"][0])


class TestEdgeCases:
    def test_founders_only(self):
        """DataFrame with only founders should produce no pairs except possibly twins."""
        df = pl.DataFrame(
            {
                "id": np.arange(100),
                "mother": np.full(100, -1),
                "father": np.full(100, -1),
                "twin": np.full(100, -1),
                "sex": np.random.default_rng(42).binomial(1, 0.5, 100),
                "generation": np.zeros(100, dtype=int),
            }
        )
        pairs = PedigreeGraph(df).extract_pairs()
        for key in [
            "FS",
            "MHS",
            "PHS",
            "MO",
            "FO",
            "1C",
            "GP",
            "Av",
            "2C",
        ]:
            assert len(pairs[key][0]) == 0, f"{key} should be empty for founders-only"

    def test_single_child_families(self):
        """No sibling pairs when every family has exactly 1 child."""
        n_founders = 50
        n_children = 25
        ids = np.arange(n_founders + n_children)
        mothers = np.full(n_founders + n_children, -1, dtype=int)
        fathers = np.full(n_founders + n_children, -1, dtype=int)
        twins = np.full(n_founders + n_children, -1, dtype=int)
        sex = np.zeros(n_founders + n_children, dtype=int)
        gen = np.zeros(n_founders + n_children, dtype=int)

        # Assign unique parents to each child
        females = [i for i in range(n_founders) if i % 2 == 0]
        males = [i for i in range(n_founders) if i % 2 == 1]
        sex[:n_founders:2] = 0  # females
        sex[1:n_founders:2] = 1  # males

        for i in range(n_children):
            child_id = n_founders + i
            mothers[child_id] = females[i]
            fathers[child_id] = males[i]
            gen[child_id] = 1

        df = pl.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "twin": twins,
                "sex": sex,
                "generation": gen,
            }
        )
        pairs = PedigreeGraph(df).extract_pairs()
        assert len(pairs["FS"][0]) == 0
        assert len(pairs["MHS"][0]) == 0
        assert len(pairs["PHS"][0]) == 0


class TestKnownTinyPedigree:
    """Hand-built 3-generation pedigree with manually counted expected pairs."""

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
        n = 11
        data = {
            "id": np.arange(n),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(n, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        }
        return pl.DataFrame(data)

    def test_full_sib_count(self, tiny_pedigree):
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
        sib_set = _pairs_to_set(*pairs["FS"])
        expected = {(4, 5), (6, 7), (8, 10)}
        assert sib_set == expected, f"Got {sib_set}"

    def test_mother_offspring_count(self, tiny_pedigree):
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
        mo_set = _pairs_to_set(*pairs["MO"])
        assert len(mo_set) == 7

    def test_father_offspring_count(self, tiny_pedigree):
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
        fo_set = _pairs_to_set(*pairs["FO"])
        assert len(fo_set) == 7

    def test_cousin_count(self, tiny_pedigree):
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
        cousin_set = _pairs_to_set(*pairs["1C"])
        # 8's parents: (4, 6). 9's parents: (5, 7).
        # 4 & 5 share grandparents 0,1. 6 & 7 share grandparents 2,3.
        # So 8 and 9 are double 1st cousins (share all 4 grandparents).
        # 10's parents: (4, 6) same as 8, so 10 is full sib of 8.
        # 10 and 9: parents (4,6) vs (5,7) — same as 8 vs 9
        expected = {(8, 9), (9, 10)}
        assert cousin_set == expected, f"Got {cousin_set}"

    def test_grandparent_grandchild_count(self, tiny_pedigree):
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
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
        pairs = PedigreeGraph(tiny_pedigree).extract_pairs()
        avunc_set = _pairs_to_set(*pairs["Av"])
        # 5 is full sib of 4 (mother of 8, 10) → 5 is aunt of 8, 10
        # 4 is full sib of 5 (mother of 9) → 4 is aunt of 9
        # 7 is full sib of 6 (father of 8, 10) → 7 is uncle of 8, 10
        # 6 is full sib of 7 (father of 9) → 6 is uncle of 9
        expected = {(5, 8), (5, 10), (4, 9), (7, 8), (7, 10), (6, 9)}
        assert avunc_set == expected, f"Got {avunc_set}"


class TestSecondCousinFullVsHalf:
    """Verify that 2C extraction returns only full 2nd cousins (≥ 2 shared great-grandparents)
    and excludes half-2nd-cousins (1 shared great-grandparent)."""

    @pytest.fixture
    def pedigree_with_half_2c(self):
        """4-generation pedigree with one full-2C pair and one half-2C pair.

        Full-2C branch (share 2 great-grandparents via a mated pair):
          Gen 0: 0(F), 1(M)
          Gen 1: 2(M)=child(0,1), 3(M)=child(0,1)  — full sibs
                 4(F) founder, 5(F) founder
          Gen 2: 6(M)=child(4,2), 7(M)=child(5,3)   — full 1st cousins
                 8(F) founder, 9(F) founder
          Gen 3: 10=child(8,6), 11=child(9,7)        — full 2nd cousins
            GGPs of 10: {0,1} (via 6→2→{0,1}). GGPs of 11: {0,1} (via 7→3→{0,1}).
            Shared GGPs = {0,1} → count=2 → full 2C ✓

        Half-2C branch (share only 1 great-grandparent):
          Gen 0: 12(F), 13(M), 14(M)
          Gen 1: 15(M)=child(12,13), 16(M)=child(12,14) — maternal half-sibs
                 17(F) founder, 18(F) founder
          Gen 2: 19(M)=child(17,15), 20(M)=child(18,16) — half 1st cousins
                 21(F) founder, 22(F) founder
          Gen 3: 23=child(21,19), 24=child(22,20)       — half 2nd cousins
            GGPs of 23: {12,13} (via 19→15→{12,13}). GGPs of 24: {12,14} (via 20→16→{12,14}).
            Shared GGPs = {12} → count=1 → half 2C, should be EXCLUDED
        """
        n = 25
        data = {
            "id": np.arange(n),
            "mother": np.array(
                [
                    -1,
                    -1,
                    0,
                    0,  # 0-3
                    -1,
                    -1,
                    4,
                    5,  # 4-7
                    -1,
                    -1,
                    8,
                    9,  # 8-11
                    -1,
                    -1,
                    -1,
                    12,
                    12,  # 12-16
                    -1,
                    -1,
                    17,
                    18,  # 17-20
                    -1,
                    -1,
                    21,
                    22,  # 21-24
                ]
            ),
            "father": np.array(
                [
                    -1,
                    -1,
                    1,
                    1,  # 0-3
                    -1,
                    -1,
                    2,
                    3,  # 4-7
                    -1,
                    -1,
                    6,
                    7,  # 8-11
                    -1,
                    -1,
                    -1,
                    13,
                    14,  # 12-16
                    -1,
                    -1,
                    15,
                    16,  # 17-20
                    -1,
                    -1,
                    19,
                    20,  # 21-24
                ]
            ),
            "twin": np.full(n, -1),
            "sex": np.array(
                [
                    0,
                    1,
                    1,
                    1,  # 0=F, 1=M, 2=M, 3=M
                    0,
                    0,
                    1,
                    1,  # 4=F, 5=F, 6=M, 7=M
                    0,
                    0,
                    0,
                    0,  # 8=F, 9=F, 10, 11
                    0,
                    1,
                    1,
                    1,
                    1,  # 12=F, 13=M, 14=M, 15=M, 16=M
                    0,
                    0,
                    1,
                    1,  # 17=F, 18=F, 19=M, 20=M
                    0,
                    0,
                    0,
                    0,  # 21=F, 22=F, 23, 24
                ]
            ),
            "generation": np.array(
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                ]
            ),
        }
        return pl.DataFrame(data)

    def test_full_2c_included(self, pedigree_with_half_2c):
        """Full 2nd cousin pair (10, 11) must be in 2C."""
        pairs = PedigreeGraph(pedigree_with_half_2c).extract_pairs(max_degree=5)
        sc_set = _pairs_to_set(*pairs["2C"])
        assert (10, 11) in sc_set, f"Full 2C pair (10,11) missing from 2C: {sc_set}"

    def test_half_2c_excluded(self, pedigree_with_half_2c):
        """Half 2nd cousin pair (23, 24) must NOT be in 2C."""
        pairs = PedigreeGraph(pedigree_with_half_2c).extract_pairs(max_degree=5)
        sc_set = _pairs_to_set(*pairs["2C"])
        assert (23, 24) not in sc_set, f"Half 2C pair (23,24) incorrectly in 2C: {sc_set}"

    def test_2c_count(self, pedigree_with_half_2c):
        """Only 1 full 2C pair should exist in this pedigree."""
        pairs = PedigreeGraph(pedigree_with_half_2c).extract_pairs(max_degree=5)
        sc_set = _pairs_to_set(*pairs["2C"])
        assert sc_set == {(10, 11)}, f"Expected exactly {{(10,11)}}, got {sc_set}"


# ---------------------------------------------------------------------------
# Constructors: from_arrays, from_subsample
# ---------------------------------------------------------------------------


class TestFromArrays:
    def test_round_trip_matches_dataframe_construction(self, small_pedigree):
        ids = small_pedigree["id"].to_numpy()
        mothers = small_pedigree["mother"].to_numpy()
        fathers = small_pedigree["father"].to_numpy()
        twins = small_pedigree["twin"].to_numpy()
        generation = small_pedigree["generation"].to_numpy()

        pg_df = PedigreeGraph(small_pedigree)
        pg_arr = PedigreeGraph.from_arrays(ids, mothers, fathers, twins, generation)

        # Same n, same remapped parent indices
        assert pg_df.n == pg_arr.n
        np.testing.assert_array_equal(pg_df.mother, pg_arr.mother)
        np.testing.assert_array_equal(pg_df.father, pg_arr.father)
        np.testing.assert_array_equal(pg_df.generation, pg_arr.generation)

    def test_derives_generation_when_none(self):
        # Three-gen lineage: 0 founder, 1 child of 0, 2 child of 1
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, 0, 1]),
            fathers=np.array([-1, -1, -1]),
            twins=None,
            generation=None,
        )
        np.testing.assert_array_equal(pg.generation, [0, 1, 2])

    def test_default_twins_is_no_twins(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            twins=None,
            generation=np.array([0, 0, 1]),
        )
        # All twin entries remap to -1 (no twins)
        assert np.all(pg.twin == -1)

    def test_birth_year_omitted_is_none(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
        )
        assert pg.birth_year is None

    def test_birth_year_round_trips_via_from_arrays(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([1990, 1990, 2010]),
        )
        assert pg.birth_year is not None
        assert pg.birth_year.dtype == np.int32
        np.testing.assert_array_equal(pg.birth_year, [1990, 1990, 2010])

    def test_birth_year_round_trips_via_dataframe(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "mother": [-1, -1, 0],
                "father": [-1, -1, 1],
                "twin": [-1, -1, -1],
                "sex": [0, 1, 0],
                "generation": [0, 0, 1],
                "birth_year": [1990, 1992, 2010],
            }
        )
        pg = PedigreeGraph(df)
        np.testing.assert_array_equal(pg.birth_year, [1990, 1992, 2010])

    def test_birth_year_nan_coerced_to_sentinel(self):
        # NaN floats (e.g. pandas Series with missing values) collapse to -1.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([1990.0, np.nan, 2010.0]),
        )
        np.testing.assert_array_equal(pg.birth_year, [1990, -1, 2010])

    def test_birth_year_topology_violation_raises(self):
        # Child born before mother.
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_arrays(
                ids=np.array([0, 1, 2]),
                mothers=np.array([-1, -1, 0]),
                fathers=np.array([-1, -1, 1]),
                birth_year=np.array([2010, 1990, 1990]),
            )
        assert info.value.code == "birth_year_topology"
        assert info.value.fields["parent_role"] == "mother"
        assert info.value.fields["child_row"] == 2
        assert info.value.fields["parent_row"] == 0
        assert info.value.fields["child_id"] == 2
        assert info.value.fields["parent_id"] == 0
        assert info.value.fields["child_birth_year"] == 1990
        assert info.value.fields["parent_birth_year"] == 2010
        assert info.value.fields["violation_count"] == 1

    def test_birth_year_topology_violation_reports_the_father_role(self):
        # Child born before father (mother edge OK).
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_arrays(
                ids=np.array([0, 1, 2]),
                mothers=np.array([-1, -1, 0]),
                fathers=np.array([-1, -1, 1]),
                birth_year=np.array([1990, 2010, 1995]),
            )
        assert info.value.code == "birth_year_topology"
        assert info.value.fields["parent_role"] == "father"

    def test_birth_year_partial_parent_validated_against_known_only(self):
        # Child has only the mother known. Father edge unconstrained.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, -1]),
            birth_year=np.array([1990, 1990, 2010]),
        )
        np.testing.assert_array_equal(pg.birth_year, [1990, 1990, 2010])

    def test_birth_year_unknown_endpoints_skip_validation(self):
        # Mother has unknown birth_year (-1); edge is not constrained.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([-1, 1990, 2010]),
        )
        np.testing.assert_array_equal(pg.birth_year, [-1, 1990, 2010])

    def test_birth_year_equal_to_parent_is_allowed(self):
        # Edge case: child.birth_year == parent.birth_year (unrealistic but
        # not a topological error).
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([2010, 2010, 2010]),
        )
        np.testing.assert_array_equal(pg.birth_year, [2010, 2010, 2010])


class TestGenerationInterval:
    def test_returns_none_when_birth_year_missing(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
        )
        assert pg.generation_interval is None

    def test_basic_two_parent_pedigree(self):
        # Mother 0 born 1990; father 1 born 1992; child 2 born 2010.
        # Only one mother-edge (Δ=20y) and one father-edge (Δ=18y).
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([1990, 1992, 2010]),
        )
        gi = pg.generation_interval
        assert gi is not None
        assert gi.T_m == pytest.approx(18.0)  # father-side
        assert gi.T_f == pytest.approx(20.0)  # mother-side
        assert gi.T == pytest.approx(19.0)  # noqa: SIM300 (gi.T is an attribute, not a constant)
        assert gi.n_edges == 2

    def test_returns_none_when_one_sex_has_no_edges(self):
        # All fathers unknown → T_m undefined → result is None.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 2]),
            mothers=np.array([-1, 0]),
            fathers=np.array([-1, -1]),
            birth_year=np.array([1990, 2010]),
        )
        assert pg.generation_interval is None

    def test_returns_none_when_no_edges_have_known_birth_years(self):
        # Edges exist but parent birth_years all unknown.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([-1, -1, 2010]),
        )
        assert pg.generation_interval is None

    def test_unknown_endpoints_skipped_from_mean(self):
        # Father 1's birth_year is unknown, so its two outgoing edges
        # are excluded from T_m even though the children's birth_years
        # are known.  All four endpoints known → both means defined.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2, 3, 4, 5]),
            mothers=np.array([-1, -1, -1, 0, 0, 2]),
            fathers=np.array([-1, -1, -1, 1, 1, 1]),
            # Mother 2's birth_year is unknown; the 2 → 5 mother edge is
            # skipped from T_f, but the 0 → 3 and 0 → 4 edges remain.
            birth_year=np.array([1990, 1990, -1, 2010, 2012, 2014]),
        )
        gi = pg.generation_interval
        assert gi is not None
        # T_m: edges 1→3 (Δ=20), 1→4 (Δ=22), 1→5 (Δ=24). Mean = 22.0.
        assert gi.T_m == pytest.approx(22.0)
        # T_f: edges 0→3 (Δ=20), 0→4 (Δ=22). The 2→5 edge is skipped
        # because mother 2's birth_year is unknown.  Mean = 21.0.
        assert gi.T_f == pytest.approx(21.0)
        assert gi.T == pytest.approx(21.5)  # noqa: SIM300 (gi.T is an attribute, not a constant)

    def test_includes_skip_generation_edges(self):
        # Founder 0 (1900) and founder 1 (1920) are the parents of 2 (1940),
        # so the mother edge spans two generations' worth of years.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([1900, 1920, 1940]),
        )
        gi = pg.generation_interval
        assert gi is not None
        # Dam edge (T_f): 0 → 2, Δ = 40 (skip-gen); sire edge (T_m): 1 → 2, Δ = 20.
        assert gi.T_m == pytest.approx(20.0)
        assert gi.T_f == pytest.approx(40.0)
        assert gi.n_edges == 2

    def test_cached_on_second_access(self):
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),
            fathers=np.array([-1, -1, 1]),
            birth_year=np.array([1990, 1992, 2010]),
        )
        gi1 = pg.generation_interval
        gi2 = pg.generation_interval
        assert gi1 is gi2  # cached_property returns the same object


class TestInputValidation:
    """PGQ-002: constructor input boundary and sparse-safe id remapping."""

    _BASE = {
        "mother": [-1, -1],
        "father": [-1, -1],
        "twin": [-1, -1],
        "sex": [0, 0],
        "generation": [0, 0],
    }

    def test_duplicate_ids_raise(self):
        df = pl.DataFrame({"id": [0, 0], **self._BASE})
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph(df)
        assert info.value.code == "duplicate_id"
        assert info.value.fields["id"] == 0
        assert info.value.fields["rows"] == (0, 1)

    def test_negative_ids_raise(self):
        df = pl.DataFrame({"id": [-1, 0], **self._BASE})
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph(df)
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "id"
        assert info.value.fields["minimum"] == 0

    def test_non_integer_ids_raise(self):
        df = pl.DataFrame({"id": [0.0, 0.5], **self._BASE})
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph(df)
        assert info.value.code == "invalid_integer_value"
        assert info.value.fields["value"] == 0.5

    def test_integral_float_ids_are_accepted(self):
        df = pl.DataFrame({"id": [0.0, 1.0], **self._BASE})
        assert PedigreeGraph(df)._ids.tolist() == [0, 1]

    def test_mismatched_column_length_raises(self):
        data = {
            "id": np.array([0, 1]),
            "mother": np.array([-1]),  # short
            "father": np.array([-1, -1]),
            "twin": np.array([-1, -1]),
            "sex": np.array([0, 0]),
            "generation": np.array([0, 0]),
        }
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph(data)
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "mother"
        assert info.value.fields["expected_length"] == 2
        assert info.value.fields["actual_length"] == 1

    def test_missing_required_column_raises(self):
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph({"id": np.array([0, 1])})
        assert info.value.code == "missing_field"
        assert info.value.fields["field"] == "mother"

    def test_sparse_high_ids_do_not_allocate_dense_table(self):
        # Pre-fix this allocated a max(id)+1 (~2e9) int32 table → ~8 GB / OOM.
        # searchsorted keeps it O(n log n).
        ids = np.array([0, 1, 2_000_000_000, 2_000_000_001])
        pg = PedigreeGraph.from_arrays(
            ids=ids,
            mothers=np.array([-1, -1, 0, 0]),
            fathers=np.array([-1, -1, 1, 1]),
            generation=np.array([0, 0, 1, 1]),
        )
        assert pg.mother.tolist() == [-1, -1, 0, 0]
        assert pg.father.tolist() == [-1, -1, 1, 1]

    def test_unsorted_ids_remap_correctly(self):
        # searchsorted must honor argsort order, not assume sorted ids.
        df = pl.DataFrame(
            {
                "id": [5, 2, 9, 7],
                "mother": [-1, -1, 5, 2],
                "father": [-1, -1, 2, 5],
                "twin": [-1, -1, -1, -1],
                "sex": [0, 0, 0, 0],
                "generation": [0, 0, 1, 1],
            }
        )
        pg = PedigreeGraph(df)
        # row 2 (id 9): mother id 5 -> row 0, father id 2 -> row 1
        assert pg.mother.tolist() == [-1, -1, 0, 1]
        assert pg.father.tolist() == [-1, -1, 1, 0]

    def test_absent_parent_ids_remap_leniently(self):
        # Partial pedigree (falconer's PedigreeGraph(df) path): parents
        # outside the rows remap to -1, but orig ids drive sib grouping.
        df = pl.DataFrame(
            {
                "id": [10, 11],
                "mother": [99, 99],  # 99 not a row
                "father": [98, 98],  # 98 not a row
                "twin": [-1, -1],
                "sex": [0, 0],
                "generation": [1, 1],
            }
        )
        pg = PedigreeGraph(df)
        assert pg.mother.tolist() == [-1, -1]
        assert pg._orig_mother.tolist() == [99, 99]

    def test_absent_twin_id_remaps_leniently(self):
        df = pl.DataFrame(
            {
                "id": [0, 1],
                "mother": [-1, -1],
                "father": [-1, -1],
                "twin": [7, -1],  # 7 not a row (co-twin outside sample)
                "sex": [0, 0],
                "generation": [0, 0],
            }
        )
        pg = PedigreeGraph(df)
        assert pg.twin.tolist() == [-1, -1]

    def test_duplicate_full_pedigree_ids_raise_in_from_subsample(self):
        full = pl.DataFrame(
            {
                "id": [0, 0, 1],  # duplicate full-pedigree id
                "mother": [-1, -1, 0],
                "father": [-1, -1, -1],
                "twin": [-1, -1, -1],
                "sex": [0, 0, 0],
                "generation": [0, 0, 1],
            }
        )
        sub = full[2:]
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_subsample(full, sub)
        assert info.value.code == "duplicate_id"


class TestFromSubsample:
    @pytest.fixture
    def lineage_pedigree(self):
        # 3-gen lineage: founders 0, 2 first; then 1=child(0,2), 3=child(0,2),
        # 4=child(1,3).  Rows ordered so that parent row indices precede their
        # children (topological invariant required by PedigreeGraph).
        return pl.DataFrame(
            {
                "id": np.array([0, 2, 1, 3, 4]),
                "mother": np.array([-1, -1, 0, 0, 1]),
                "father": np.array([-1, -1, 2, 2, 3]),
                "twin": np.full(5, -1),
                "sex": np.array([0, 0, 1, 1, 0]),
                "generation": np.array([0, 0, 1, 1, 2]),
            }
        )

    def test_extract_pairs_filters_to_subsample(self, lineage_pedigree):
        sub = lineage_pedigree.filter(pl.col("id").is_in([1, 3, 4]))
        pg = PedigreeGraph.from_subsample(lineage_pedigree, sub)
        pairs = pg.extract_pairs(max_degree=2)
        # All extracted pair endpoints must be valid sub-row indices
        for code, (idx1, idx2) in pairs.items():
            for i, j in zip(idx1, idx2, strict=True):
                assert 0 <= i < len(sub), f"{code}: i={i} out of range"
                assert 0 <= j < len(sub), f"{code}: j={j} out of range"

    def test_empty_subsample_yields_no_pairs(self, lineage_pedigree):
        empty = lineage_pedigree.head(0)
        pg = PedigreeGraph.from_subsample(lineage_pedigree, empty)
        pairs = pg.extract_pairs(max_degree=2)
        for code, (idx1, _) in pairs.items():
            assert len(idx1) == 0, f"{code} should be empty"

    def test_duplicate_ids_raises(self, lineage_pedigree):
        dup = pl.concat([lineage_pedigree.head(2), lineage_pedigree.head(1)])
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_subsample(lineage_pedigree, dup)
        assert info.value.code == "duplicate_id"

    def test_id_not_in_full_pedigree_raises(self, lineage_pedigree):
        bogus = pl.DataFrame(
            {
                "id": [99],
                "mother": [-1],
                "father": [-1],
                "twin": [-1],
                "sex": [0],
                "generation": [0],
            }
        )
        with pytest.raises(PedigreeValidationError) as info:
            PedigreeGraph.from_subsample(lineage_pedigree, bogus)
        assert info.value.code == "unknown_view_id"
        assert info.value.fields["id"] == 99
        assert info.value.fields["position"] == 0
        assert info.value.fields["missing_count"] == 1

    def test_count_pairs_full_vs_subsample(self, lineage_pedigree):
        sub = lineage_pedigree.filter(pl.col("id").is_in([1, 3, 4]))
        pg = PedigreeGraph.from_subsample(lineage_pedigree, sub)
        full = pg.count_pairs(max_degree=2, scope="full")
        partial = pg.count_pairs(max_degree=2, scope="subsample")
        # Full counts >= subsample counts for every relationship code
        for code in full:
            assert full[code] >= partial[code], f"{code}: full={full[code]} < sub={partial[code]}"

    def test_count_pairs_invalid_scope_raises(self, lineage_pedigree):
        pg = PedigreeGraph(lineage_pedigree)
        with pytest.raises(ValueError, match="scope must be"):
            pg.count_pairs(scope="bogus")

    def test_birth_year_round_trips_through_subsample(self, lineage_pedigree):
        # Attach birth_year to the full pedigree and ensure it survives
        # the subsample construction path.
        full = lineage_pedigree.with_columns(birth_year=pl.Series(np.array([1990, 1990, 2000, 2000, 2010])))
        sub = full.filter(pl.col("id").is_in([1, 3, 4]))
        pg = PedigreeGraph.from_subsample(full, sub)
        assert pg.birth_year is not None
        # The subsample graph is built over the FULL pedigree, so
        # pg.birth_year is the full vector indexed by full-row index.
        # Verify the values match the original full ordering.
        np.testing.assert_array_equal(pg.birth_year, [1990, 1990, 2000, 2000, 2010])


# ---------------------------------------------------------------------------
# kinship + compute_pair_kinship (inbred branch)
# ---------------------------------------------------------------------------


class TestComputePairKinship:
    def _inbred_pedigree(self):
        # Same layout as the kinship-kernel inbred-MZ test:
        # G0: 0,1 founders; G1: 2,3 full-sibs of (0,1); G2: 4,5 MZ twins of (2,3)
        return pl.DataFrame(
            {
                "id": np.arange(6),
                "mother": np.array([-1, -1, 0, 0, 2, 2]),
                "father": np.array([-1, -1, 1, 1, 3, 3]),
                "twin": np.array([-1, -1, -1, -1, 5, 4]),
                "sex": np.array([0, 1, 0, 1, 0, 1]),
                "generation": np.array([0, 0, 1, 1, 2, 2]),
            }
        )

    def test_non_inbred_single_path_is_exact(self):
        # Plain full sibs (no inbreeding, no duplicate paths): exact kinship
        # equals the nominal 0.25.  (There is no nominal fast path anymore; the
        # exact recurrence simply returns the same value here.)
        df = pl.DataFrame(
            {
                "id": np.arange(4),
                "mother": np.array([-1, -1, 0, 0]),
                "father": np.array([-1, -1, 1, 1]),
                "twin": np.full(4, -1),
                "sex": np.array([0, 1, 0, 1]),
                "generation": np.array([0, 0, 1, 1]),
            }
        )
        pg = PedigreeGraph(df)
        pairs = pg.extract_pairs(max_degree=1)
        out = pg.compute_pair_kinship(pairs)
        # Full siblings → kinship 0.25
        assert np.all(out["FS"] == 0.25)

    def test_inbred_path_returns_correct_mz_and_full_sib(self):
        df = self._inbred_pedigree()
        pg = PedigreeGraph(df)
        pairs = pg.extract_pairs(max_degree=1)
        F = pg.compute_inbreeding()
        # Twins (rows 4, 5) inbred with F = 0.25
        assert F[4] == pytest.approx(0.25)
        assert F[5] == pytest.approx(0.25)

        out = pg.compute_pair_kinship(pairs)
        # MZ off-diagonal = (1 + F) / 2 = 0.625
        assert out["MZ"].tolist() == [pytest.approx(0.625)]
        # G1 full sibs (2, 3) — both non-inbred, so FS kinship = 0.25
        assert all(v == pytest.approx(0.25) for v in out["FS"])

    def test_reversed_subsample_pair_kinship_uses_graph_coords(self):
        # PGQ-001 regression: extract_pairs returns caller (df-row) coordinates
        # on a from_subsample graph, but the kinship matrix is built in
        # full-graph coordinates.  A reordered subsample must still yield the
        # correct exact kinship — exercising the inbred/MZ slow path.
        full = self._inbred_pedigree()
        K_full = PedigreeGraph(full).kinship_matrix(min_kinship=0.0).tocsr()
        assert K_full[4, 5] == pytest.approx(0.625)

        # Reversed subsample of the MZ twins: df rows are [id 5, id 4].
        sub = full.filter(pl.col("id").is_in([4, 5])).reverse()
        pg = PedigreeGraph.from_subsample(full, sub)
        pairs = pg.extract_pairs(max_degree=1)
        out = pg.compute_pair_kinship(pairs)
        assert out["MZ"].tolist() == [pytest.approx(0.625)]

    def test_reordered_subsample_pairs_are_canonically_ordered(self):
        # The graph→caller remap can permute rows; remapped pairs must keep
        # the lo < hi invariant that downstream pair-key encoders rely on.
        full = self._inbred_pedigree()
        sub = full.filter(pl.col("id").is_in([2, 3, 4, 5])).reverse()
        pg = PedigreeGraph.from_subsample(full, sub)
        pairs = pg.extract_pairs(max_degree=2)
        for code, (idx1, idx2) in pairs.items():
            assert np.all(idx1 <= idx2), f"{code} not canonically ordered after remap"

    def test_compute_inbreeding_idempotent(self):
        df = self._inbred_pedigree()
        pg = PedigreeGraph(df)
        F1 = pg.compute_inbreeding()
        F2 = pg.compute_inbreeding()
        np.testing.assert_array_equal(F1, F2)

    def test_kinship_matrix_max_degree_shortcut(self):
        df = pl.DataFrame(
            {
                "id": np.arange(4),
                "mother": np.array([-1, 0, 1, 2]),
                "father": np.full(4, -1),
                "twin": np.full(4, -1),
                "sex": np.zeros(4, dtype=int),
                "generation": np.array([0, 1, 2, 3]),
            }
        )
        pg = PedigreeGraph(df)
        # max_degree=2 → threshold = 0.5**3 - 1e-9 ≈ 0.125
        # Kinship 0,1 = 0.25 (kept), 0,2 = 0.125 (kept boundary), 0,3 = 0.0625 (dropped)
        K = pg.kinship_matrix(max_degree=2).toarray()
        assert K[0, 1] == 0.25
        assert K[0, 2] == 0.125
        assert K[0, 3] == 0.0  # dropped by max_degree=2 threshold

    def test_empty_pair_array_handled(self):
        df = self._inbred_pedigree()
        pg = PedigreeGraph(df)
        # Force inbred path with empty MHS pairs
        pairs = pg.extract_pairs(max_degree=1)
        if len(pairs["MHS"][0]) == 0:
            out = pg.compute_pair_kinship(pairs)
            assert out["MHS"].dtype == np.float64
            assert len(out["MHS"]) == 0

    def test_constructor_accepts_a_parent_row_after_its_child(self):
        # Row 3's mother is row 5: acyclic but not topological.  The private
        # depth-major order carries every order-dependent kernel.
        df = pl.DataFrame(
            {
                "id": np.array([0, 1, 2, 3, 4, 5]),
                "mother": np.array([-1, -1, 0, 5, -1, -1]),
                "father": np.array([-1, -1, 1, -1, -1, -1]),
                "twin": np.full(6, -1),
                "sex": np.array([1, 0, 0, 1, 0, 1], dtype=np.int8),
                "generation": np.array([0, 0, 1, 1, 0, 0]),
            }
        )
        pg = PedigreeGraph(df)
        assert pg._depth.tolist() == [0, 0, 1, 1, 0, 0]
        assert pg.compute_pair_kinship({"PO": (np.array([3]), np.array([5]))})["PO"].tolist() == [0.25]
        assert pg.kinship_matrix(0.0)[3, 5] == np.float32(0.25)

    # -- exactness battery for the public compute_pair_kinship API --

    def test_double_first_cousins_are_exact_not_nominal(self):
        # The bug the removed fast path produced: double first cousins were
        # returned as the nominal 1C value (0.0625) instead of their true 0.125.
        pg = PedigreeGraph(_ped_double_first_cousins())
        pairs = pg.extract_pairs(max_degree=3)
        out = pg.compute_pair_kinship(pairs)
        assert list(zip(pairs["1C"][0], pairs["1C"][1], strict=True)) == [(8, 9), (9, 10)]
        np.testing.assert_array_equal(out["1C"], np.array([0.125, 0.125]))

    def test_half_first_cousin_parent_offspring_custom_pairs(self):
        # Disproof case via custom (non-extract_pairs) pairs: a sub-threshold
        # 1/32 parental kinship must still raise the parent-offspring kinship to
        # 0.265625 (a pruned matrix would return 0.25).
        pg = PedigreeGraph(_ped_half_first_cousin_parents())
        pairs = {"x": (np.array([7, 9]), np.array([8, 7]))}
        out = pg.compute_pair_kinship(pairs)
        np.testing.assert_allclose(out["x"], np.array([0.03125, 0.265625]))

    def test_mz_twins_with_descendants_full_parity(self):
        pg = PedigreeGraph(_ped_mz_twins_with_descendants())
        K = pg.kinship_matrix(0.0).toarray()
        pairs = pg.extract_pairs(max_degree=5)
        out = pg.compute_pair_kinship(pairs)
        for code, (idx1, idx2) in pairs.items():
            if len(idx1):
                np.testing.assert_allclose(out[code], K[idx1, idx2], atol=1e-6)

    def test_consanguinity_full_parity(self):
        pg = PedigreeGraph(_ped_sib_mating())
        K = pg.kinship_matrix(0.0).toarray()
        pairs = pg.extract_pairs(max_degree=5)
        out = pg.compute_pair_kinship(pairs)
        for code, (idx1, idx2) in pairs.items():
            if len(idx1):
                np.testing.assert_allclose(out[code], K[idx1, idx2], atol=1e-6)

    def test_custom_dict_self_pair_and_unknown_code(self):
        # Arbitrary code keys (not in REL_REGISTRY) and self-pairs are computed
        # exactly; input orientation is preserved.
        pg = PedigreeGraph(_ped_sib_mating())
        pairs = {
            "self": (np.array([4, 0]), np.array([4, 0])),  # F_4=0.25 -> 0.625
            "reversed": (np.array([2, 4]), np.array([4, 2])),
            "made_up_code": (np.array([2]), np.array([3])),  # full sibs -> 0.25
        }
        out = pg.compute_pair_kinship(pairs)
        np.testing.assert_allclose(out["self"], np.array([0.625, 0.5]))
        np.testing.assert_array_equal(out["reversed"], out["reversed"][::-1])
        np.testing.assert_allclose(out["made_up_code"], np.array([0.25]))

    def test_cached_matrix_path_matches_direct(self):
        # Whether or not kinship_matrix(0.0) is pre-cached, the result is the
        # same (the cached-CSC sampling branch vs the direct recurrence branch).
        df = _ped_double_first_cousins()
        pairs = PedigreeGraph(df).extract_pairs(max_degree=5)

        pg_direct = PedigreeGraph(df)  # no matrix cached -> recurrence
        out_direct = pg_direct.compute_pair_kinship(pairs)

        pg_cached = PedigreeGraph(df)
        pg_cached.kinship_matrix(0.0)  # populate the cache -> sampling branch
        out_cached = pg_cached.compute_pair_kinship(pairs)

        for code in pairs:
            np.testing.assert_allclose(out_direct[code], out_cached[code], atol=1e-6)

    def test_all_empty_returns_empty_per_code(self):
        pg = PedigreeGraph(_ped_inbred_mz())
        empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        out = pg.compute_pair_kinship({"FS": empty, "MZ": empty})
        assert set(out) == {"FS", "MZ"}
        for v in out.values():
            assert v.dtype == np.float64
            assert v.shape == (0,)


# ---------------------------------------------------------------------------
# Direct pairwise-kinship recurrence (_pairwise_kinship_py) vs matrix oracle
# ---------------------------------------------------------------------------


def _oracle_all_pairs(pg: PedigreeGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All upper-triangle (incl. diagonal) pairs, reference vs kinship_matrix(0.0).

    Returns ``(got, exp, ii, jj)`` so callers can both assert parity and inspect
    individual cells.
    """
    K = pg.kinship_matrix(0.0).toarray()
    ii, jj = np.triu_indices(pg.n)
    got = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, ii, jj)
    exp = K[ii, jj]
    return got, exp, ii, jj


def _ped_inbred_mz() -> pl.DataFrame:
    # G0: 0,1 founders; G1: 2,3 full-sibs of (0,1); G2: 4,5 MZ twins of (2,3).
    return pl.DataFrame(
        {
            "id": np.arange(6),
            "mother": np.array([-1, -1, 0, 0, 2, 2]),
            "father": np.array([-1, -1, 1, 1, 3, 3]),
            "twin": np.array([-1, -1, -1, -1, 5, 4]),
            "sex": np.array([0, 1, 0, 1, 0, 1]),
            "generation": np.array([0, 0, 1, 1, 2, 2]),
        }
    )


def _ped_double_first_cousins() -> pl.DataFrame:
    # 4,5 full sibs of (0,1); 6,7 full sibs of (2,3); 8=child(4,6),
    # 9=child(5,7), 10=child(4,6).  (8,9) and (9,10) are DOUBLE first cousins
    # (both parent-couples are full-sib pairs) -> phi = 0.125, twice the nominal
    # 1C lookup of 0.0625.
    return pl.DataFrame(
        {
            "id": np.arange(11),
            "mother": np.array([-1, -1, -1, -1, 0, 0, 2, 2, 4, 5, 4]),
            "father": np.array([-1, -1, -1, -1, 1, 1, 3, 3, 6, 7, 6]),
            "twin": np.full(11, -1),
            "sex": np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
            "generation": np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
        }
    )


def _ped_half_first_cousin_parents() -> pl.DataFrame:
    # 0..4 founders; 5=child(0,1), 6=child(0,2) share founder 0 -> half sibs;
    # 7=child(5,3), 8=child(6,4) -> half-first-cousins (phi=1/32);
    # 9=child(7,8) -> phi(9,7) = 0.5*((1+F_7)/2 + phi(8,7)) = 0.265625.
    # The disproof of threshold-pruning: a sub-threshold (1/32) parental kinship
    # feeds an above-threshold parent-offspring kinship.
    return pl.DataFrame(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, -1, -1, -1, 0, 0, 5, 6, 7]),
            "father": np.array([-1, -1, -1, -1, -1, 1, 2, 3, 4, 8]),
            "twin": np.full(10, -1),
            "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            "generation": np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3]),
        }
    )


def _ped_mz_twins_with_descendants() -> pl.DataFrame:
    # 0,1 founders; 2,3 full sibs of (0,1); 4,5 MZ twins of (2,3);
    # 6,7 unrelated founders (mates); 8=child(4,6), 9=child(5,7).
    # 4 and 5 are genome-identical, so 8 and 9 are half-sib-equivalent with phi
    # elevated above the non-inbred maternal-half-sib value by co-coalescence.
    return pl.DataFrame(
        {
            "id": np.arange(10),
            "mother": np.array([-1, -1, 0, 0, 2, 2, -1, -1, 4, 5]),
            "father": np.array([-1, -1, 1, 1, 3, 3, -1, -1, 6, 7]),
            "twin": np.array([-1, -1, -1, -1, 5, 4, -1, -1, -1, -1]),
            "sex": np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0]),
            "generation": np.array([0, 0, 1, 1, 2, 2, 0, 0, 3, 3]),
        }
    )


def _ped_sib_mating() -> pl.DataFrame:
    # 0,1 founders; 2,3 full sibs; 4=child(2,3) (sib-mating, F_4=0.25).
    return pl.DataFrame(
        {
            "id": np.arange(5),
            "mother": np.array([-1, -1, 0, 0, 2]),
            "father": np.array([-1, -1, 1, 1, 3]),
            "twin": np.full(5, -1),
            "sex": np.array([0, 1, 0, 1, 0]),
            "generation": np.array([0, 0, 1, 1, 2]),
        }
    )


_PAIRWISE_FIXTURES = [
    _ped_inbred_mz,
    _ped_double_first_cousins,
    _ped_half_first_cousin_parents,
    _ped_mz_twins_with_descendants,
    _ped_sib_mating,
]


def _random_pedigree(rng: np.random.Generator, p_twin: float = 0.3) -> pl.DataFrame:
    """Generate a small valid (topologically ordered) random pedigree.

    Parent-index-only; no simace dependency.  Mixes inbreeding (mates drawn
    from the same cohort) and occasional MZ twins so the fuzz exercises both
    correction paths.  Children always get a higher id than their parents, so
    the topological invariant holds.
    """
    n_founders = int(rng.integers(2, 5))
    n_gen = int(rng.integers(1, 5))
    per_gen = int(rng.integers(1, 4))
    ids = list(range(n_founders))
    mother = [-1] * n_founders
    father = [-1] * n_founders
    twin = [-1] * n_founders
    gen = [0] * n_founders
    sex = [int(rng.integers(0, 2)) for _ in range(n_founders)]
    cur = list(range(n_founders))
    next_id = n_founders
    for g in range(1, n_gen + 1):
        new_gen: list[int] = []
        females = [i for i in cur if sex[i] == 0] or cur
        males = [i for i in cur if sex[i] == 1] or cur
        for _ in range(per_gen):
            m = int(rng.choice(females))
            # A child cannot name one individual in both parent roles.
            mates = [i for i in males if i != m] or [i for i in cur if i != m]
            f = int(rng.choice(mates)) if mates else -1
            ids.append(next_id)
            mother.append(m)
            father.append(f)
            twin.append(-1)
            gen.append(g)
            sex.append(int(rng.integers(0, 2)))
            new_gen.append(next_id)
            next_id += 1
        # Occasionally turn the last two new individuals into MZ twins.
        if len(new_gen) >= 2 and rng.random() < p_twin:
            a, b = new_gen[-1], new_gen[-2]
            mother[b] = mother[a]
            father[b] = father[a]
            twin[a] = b
            twin[b] = a
            sex[b] = sex[a]
        cur = new_gen
    return pl.DataFrame({"id": ids, "mother": mother, "father": father, "twin": twin, "sex": sex, "generation": gen})


class TestPairwiseKinshipReference:
    """`_pairwise_kinship_py` must equal `kinship_matrix(0.0)` on every pair.

    The matrix DP is the independent exact oracle.  These cover the cases the
    nominal-lookup fast path gets wrong (multiple relationship paths) and the
    inbreeding / MZ-co-coalescence cases that motivated the routine.
    """

    _inbred_mz = staticmethod(_ped_inbred_mz)
    _double_first_cousins = staticmethod(_ped_double_first_cousins)
    _half_first_cousin_parents = staticmethod(_ped_half_first_cousin_parents)
    _mz_twins_with_descendants = staticmethod(_ped_mz_twins_with_descendants)
    _sib_mating = staticmethod(_ped_sib_mating)

    def _phi(self, pg, a, b):
        return _pairwise_kinship_py(pg.mother, pg.father, pg.twin, np.array([a]), np.array([b]))[0]

    @pytest.mark.parametrize(
        "fixture",
        [
            "_inbred_mz",
            "_double_first_cousins",
            "_half_first_cousin_parents",
            "_mz_twins_with_descendants",
            "_sib_mating",
        ],
    )
    def test_all_pairs_match_matrix_oracle(self, fixture):
        pg = PedigreeGraph(getattr(self, fixture)())
        got, exp, _, _ = _oracle_all_pairs(pg)
        # Dyadic rationals at these shallow depths are exact in both float32 and
        # float64, so parity is effectively bit-exact; use a loose atol anyway.
        np.testing.assert_allclose(got, exp, atol=1e-6)

    def test_double_first_cousins_exceed_nominal(self):
        pg = PedigreeGraph(self._double_first_cousins())
        # Both double-cousin pairs: true phi 0.125, NOT the nominal 1C 0.0625.
        assert self._phi(pg, 8, 9) == pytest.approx(0.125)
        assert self._phi(pg, 9, 10) == pytest.approx(0.125)

    def test_half_first_cousin_parent_offspring_not_pruned(self):
        # Permanent guard against threshold-pruning regressions.
        pg = PedigreeGraph(self._half_first_cousin_parents())
        assert self._phi(pg, 7, 8) == pytest.approx(0.03125)  # half-1C parents
        assert self._phi(pg, 9, 7) == pytest.approx(0.265625)  # NOT 0.25

    def test_inbred_mz_self_and_cross_kinship(self):
        pg = PedigreeGraph(self._inbred_mz())
        # MZ twins (4,5) inbred F=0.25 -> cross-kinship = self-kinship = 0.625.
        assert self._phi(pg, 4, 5) == pytest.approx(0.625)
        assert self._phi(pg, 4, 4) == pytest.approx(0.625)
        # Their non-inbred full-sib parents (2,3) stay at 0.25.
        assert self._phi(pg, 2, 3) == pytest.approx(0.25)

    def test_mz_descendants_elevated_above_nominal_cousin(self):
        pg = PedigreeGraph(self._mz_twins_with_descendants())
        # 8=child(4,6), 9=child(5,7) with 4,5 MZ twins (genome-identical) and
        # 6,7 unrelated.  They share one genome source, so they are half-sib-
        # equivalent: phi = 0.5 * 0.5 * phi(4,5) = 0.25 * 0.625 = 0.15625.
        # That exceeds the non-inbred maternal-half-sib value of 0.125 because
        # the shared twins are themselves inbred (F=0.25) — MZ co-coalescence.
        assert self._phi(pg, 8, 9) == pytest.approx(0.15625)
        assert self._phi(pg, 8, 9) > 0.125

    def test_input_orientation_preserved(self):
        # phi is symmetric; reversed input order must give the same value and
        # not reorder the output.
        pg = PedigreeGraph(self._sib_mating())
        fwd = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, np.array([2, 4]), np.array([4, 2]))
        rev = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, np.array([4, 2]), np.array([2, 4]))
        np.testing.assert_array_equal(fwd, rev[::-1])

    def test_self_pairs_return_diagonal(self):
        pg = PedigreeGraph(self._sib_mating())
        # F_4 = 0.25 (sib-mating) -> self-kinship 0.625; founders -> 0.5.
        assert self._phi(pg, 4, 4) == pytest.approx(0.625)
        assert self._phi(pg, 0, 0) == pytest.approx(0.5)

    def test_empty_input_returns_empty_float64(self):
        pg = PedigreeGraph(self._sib_mating())
        out = _pairwise_kinship_py(
            pg.mother, pg.father, pg.twin, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        assert out.dtype == np.float64
        assert out.shape == (0,)


class TestPairwiseKinshipNumba:
    """The numba kernel must be bit-identical to the pure-Python reference.

    Both compute float64 with the same IEEE ops, so they agree to the last bit
    regardless of traversal order — making `_pairwise_kinship_py` a true bit
    oracle.  Parity against the (float32) matrix is checked at `atol=1e-6`.
    """

    @pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
    def test_numba_bit_exact_vs_python(self, build):
        pg = PedigreeGraph(build())
        ii, jj = np.triu_indices(pg.n)
        py = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, ii, jj)
        nb = pairwise_kinship(pg.mother, pg.father, pg.twin, ii, jj)
        # Bit-exact: no tolerance.
        np.testing.assert_array_equal(nb, py)

    @pytest.mark.parametrize("build", _PAIRWISE_FIXTURES, ids=lambda b: b.__name__)
    def test_numba_matches_matrix_oracle(self, build):
        pg = PedigreeGraph(build())
        K = pg.kinship_matrix(0.0).toarray()
        ii, jj = np.triu_indices(pg.n)
        nb = pairwise_kinship(pg.mother, pg.father, pg.twin, ii, jj)
        np.testing.assert_allclose(nb, K[ii, jj], atol=1e-6)

    def test_fuzz_numba_equals_python_and_oracle(self):
        rng = np.random.default_rng(20240609)
        checked = 0
        for _ in range(200):
            pg = PedigreeGraph(_random_pedigree(rng))
            if pg.n < 2:
                continue
            K = pg.kinship_matrix(0.0).toarray()
            ii, jj = np.triu_indices(pg.n)
            py = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, ii, jj)
            nb = pairwise_kinship(pg.mother, pg.father, pg.twin, ii, jj)
            np.testing.assert_array_equal(nb, py)  # bit-exact
            np.testing.assert_allclose(nb, K[ii, jj], atol=1e-6)  # vs matrix
            checked += 1
        assert checked > 100  # the generator should mostly yield n >= 2

    def test_input_orientation_preserved(self):
        pg = PedigreeGraph(_ped_sib_mating())
        fwd = pairwise_kinship(pg.mother, pg.father, pg.twin, np.array([2, 4]), np.array([4, 2]))
        rev = pairwise_kinship(pg.mother, pg.father, pg.twin, np.array([4, 2]), np.array([2, 4]))
        np.testing.assert_array_equal(fwd, rev[::-1])

    def test_empty_input_returns_empty_float64(self):
        pg = PedigreeGraph(_ped_sib_mating())
        empty = np.array([], dtype=np.int64)
        out = pairwise_kinship(pg.mother, pg.father, pg.twin, empty, empty)
        assert out.dtype == np.float64
        assert out.shape == (0,)

    def test_stats_wrapper_reports_bounded_memo(self):
        # On a moderately related pedigree the memo and stack stay small
        # relative to n**2 — the scaling guarantee in miniature.
        rng = np.random.default_rng(7)
        pg = PedigreeGraph(_random_pedigree(rng))
        ii, jj = np.triu_indices(pg.n)
        out, stats = _pairwise_kinship_with_stats(pg.mother, pg.father, pg.twin, ii, jj)
        assert out.shape == ii.shape
        assert stats["memo_entries"] <= pg.n * pg.n
        assert stats["max_stack_depth"] >= 1
        assert stats["memo_capacity"] >= stats["memo_entries"]
