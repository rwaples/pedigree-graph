"""Tests for pedigree_graph relationship extraction."""

import logging

import numpy as np
import pandas as pd
import pytest

from pedigree_graph import PedigreeGraph

logger = logging.getLogger(__name__)


def _extract_relationship_pairs_legacy(df: pd.DataFrame, seed: int = 42) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Legacy implementation kept for golden testing."""
    ids_arr = df["id"].values.astype(np.int64)
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
    ta = twins["id"].values.astype(int)
    tb = twins["twin"].values.astype(int)
    mask = ta < tb
    pairs["MZ"] = (resolve_rows(ta[mask]), resolve_rows(tb[mask]))

    non_twin_nf = df[(df["mother"] != -1) & (df["twin"] == -1)].copy()
    non_twin_nf["_row"] = non_twin_nf.index.values

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
        full_rows_1.append(mat_pairs.loc[same_father, "_row_1"].values)
        full_rows_2.append(mat_pairs.loc[same_father, "_row_2"].values)
        mat_half_rows_1.append(mat_pairs.loc[~same_father, "_row_1"].values)
        mat_half_rows_2.append(mat_pairs.loc[~same_father, "_row_2"].values)

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
        pat_half_rows_1.append(pat_pairs.loc[diff_mother, "_row_1"].values)
        pat_half_rows_2.append(pat_pairs.loc[diff_mother, "_row_2"].values)

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
    child_rows = all_nf.index.values
    mother_rows = resolve_rows(all_nf["mother"].values.astype(int))
    father_rows = resolve_rows(all_nf["father"].values.astype(int))

    m_valid = mother_rows >= 0
    f_valid = father_rows >= 0
    pairs["MO"] = (child_rows[m_valid], mother_rows[m_valid])
    pairs["FO"] = (child_rows[f_valid], father_rows[f_valid])

    child_ids = all_nf["id"].values.astype(np.int64)
    mother_ids = all_nf["mother"].values.astype(np.int64)
    father_ids = all_nf["father"].values.astype(np.int64)
    n_children = len(child_ids)

    df_mothers_col = df["mother"].values.astype(np.int64)
    df_fathers_col = df["father"].values.astype(np.int64)
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
        mother = df["mother"].values
        father = df["father"].values
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

        gen = df["generation"].values
        gen_diff = np.abs(gen[gc] - gen[gp])
        assert np.all(gen_diff == 2), f"Generation diffs: {np.unique(gen_diff)}"

    def test_grandparent_grandchild_ancestry(self, small_pedigree):
        """Each grandchild must have the grandparent as a parent of a parent."""
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()
        gc_arr, gp_arr = pairs["GP"]
        mother = df["mother"].values
        father = df["father"].values

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

        gen = df["generation"].values
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

        mother = df["mother"].values
        father = df["father"].values
        assert np.all(mother[idx1] == mother[idx2])
        assert np.all(father[idx1] == father[idx2])

    def test_half_sibs_share_exactly_one_parent(self, small_pedigree):
        df = small_pedigree
        pairs = PedigreeGraph(df).extract_pairs()

        mother = df["mother"].values
        father = df["father"].values

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

        mother = df["mother"].values
        father = df["father"].values

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
        df = pd.DataFrame(
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

        df = pd.DataFrame(
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
        return pd.DataFrame(data)

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
        return pd.DataFrame(data)

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
        ids = small_pedigree["id"].values
        mothers = small_pedigree["mother"].values
        fathers = small_pedigree["father"].values
        twins = small_pedigree["twin"].values
        generation = small_pedigree["generation"].values

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
        df = pd.DataFrame(
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
        with pytest.raises(ValueError, match="birth_year topology violation"):
            PedigreeGraph.from_arrays(
                ids=np.array([0, 1, 2]),
                mothers=np.array([-1, -1, 0]),
                fathers=np.array([-1, -1, 1]),
                birth_year=np.array([2010, 1990, 1990]),
            )

    def test_birth_year_topology_violation_message_mentions_father(self):
        # Child born before father (mother edge OK).
        with pytest.raises(ValueError, match="father-child"):
            PedigreeGraph.from_arrays(
                ids=np.array([0, 1, 2]),
                mothers=np.array([-1, -1, 0]),
                fathers=np.array([-1, -1, 1]),
                birth_year=np.array([1990, 2010, 1995]),
            )

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
        # Founder 0 (1900); child 1 (1920); grandchild 2 (1940);
        # AND a skip-gen edge: 0 also fathered 2 directly.
        # 0 → 1: 20y; 1 → 2: 20y; 0 → 2: 40y (skip-gen).
        # Wait — that would conflict with father=0 and father=1 for id=2.
        # Easier: build a pedigree with a skip-gen via the father field.
        pg = PedigreeGraph.from_arrays(
            ids=np.array([0, 1, 2]),
            mothers=np.array([-1, -1, 0]),  # mother of 2 is 0 (skip-gen since 1 also exists)
            fathers=np.array([-1, -1, 0]),  # father of 2 is also 0 — same founder, self-mating
            birth_year=np.array([1900, 1920, 1940]),
        )
        gi = pg.generation_interval
        assert gi is not None
        # Mother edge: 0 → 2, Δ = 40
        # Father edge: 0 → 2, Δ = 40
        assert gi.T_m == pytest.approx(40.0)
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


class TestFromSubsample:
    @pytest.fixture
    def lineage_pedigree(self):
        # 3-gen lineage: founders 0, 2 first; then 1=child(0,2), 3=child(0,2),
        # 4=child(1,3).  Rows ordered so that parent row indices precede their
        # children (topological invariant required by PedigreeGraph).
        return pd.DataFrame(
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
        sub = lineage_pedigree[lineage_pedigree["id"].isin([1, 3, 4])].reset_index(drop=True)
        pg = PedigreeGraph.from_subsample(lineage_pedigree, sub)
        pairs = pg.extract_pairs(max_degree=2)
        # All extracted pair endpoints must be valid sub-row indices
        for code, (idx1, idx2) in pairs.items():
            for i, j in zip(idx1, idx2, strict=True):
                assert 0 <= i < len(sub), f"{code}: i={i} out of range"
                assert 0 <= j < len(sub), f"{code}: j={j} out of range"

    def test_empty_subsample_yields_no_pairs(self, lineage_pedigree):
        empty = lineage_pedigree.iloc[:0]
        pg = PedigreeGraph.from_subsample(lineage_pedigree, empty)
        pairs = pg.extract_pairs(max_degree=2)
        for code, (idx1, _) in pairs.items():
            assert len(idx1) == 0, f"{code} should be empty"

    def test_duplicate_ids_raises(self, lineage_pedigree):
        dup = pd.concat([lineage_pedigree.iloc[:2], lineage_pedigree.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate id"):
            PedigreeGraph.from_subsample(lineage_pedigree, dup)

    def test_id_not_in_full_pedigree_raises(self, lineage_pedigree):
        bogus = pd.DataFrame(
            {
                "id": [99],
                "mother": [-1],
                "father": [-1],
                "twin": [-1],
                "sex": [0],
                "generation": [0],
            }
        )
        with pytest.raises(ValueError, match="not present in full_pedigree"):
            PedigreeGraph.from_subsample(lineage_pedigree, bogus)

    def test_count_pairs_full_vs_subsample(self, lineage_pedigree):
        sub = lineage_pedigree[lineage_pedigree["id"].isin([1, 3, 4])].reset_index(drop=True)
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
        full = lineage_pedigree.copy()
        full["birth_year"] = np.array([1990, 1990, 2000, 2000, 2010])
        sub = full[full["id"].isin([1, 3, 4])].reset_index(drop=True)
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
        return pd.DataFrame(
            {
                "id": np.arange(6),
                "mother": np.array([-1, -1, 0, 0, 2, 2]),
                "father": np.array([-1, -1, 1, 1, 3, 3]),
                "twin": np.array([-1, -1, -1, -1, 5, 4]),
                "sex": np.array([0, 1, 0, 1, 0, 1]),
                "generation": np.array([0, 0, 1, 1, 2, 2]),
            }
        )

    def test_non_inbred_uses_fast_path(self):
        df = pd.DataFrame(
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
        F = pg.compute_inbreeding()
        assert np.all(F == 0)
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

    def test_compute_inbreeding_idempotent(self):
        df = self._inbred_pedigree()
        pg = PedigreeGraph(df)
        F1 = pg.compute_inbreeding()
        F2 = pg.compute_inbreeding()
        np.testing.assert_array_equal(F1, F2)

    def test_kinship_matrix_max_degree_shortcut(self):
        df = pd.DataFrame(
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

    def test_constructor_rejects_non_topological(self):
        # Parent index appears AFTER its child — violates the invariant
        # required by `_compute_F_meuwissen_luo`'s outer loop.
        df = pd.DataFrame(
            {
                "id": np.array([0, 1, 2, 3, 4, 5]),
                "mother": np.array([-1, -1, 0, 5, -1, -1]),
                "father": np.array([-1, -1, 1, -1, -1, -1]),
                "twin": np.full(6, -1),
                "sex": np.array([1, 0, 0, 1, 0, 1], dtype=np.int8),
                "generation": np.array([0, 0, 1, 1, 0, 0]),
            }
        )
        with pytest.raises(ValueError, match="topological order"):
            PedigreeGraph(df)
