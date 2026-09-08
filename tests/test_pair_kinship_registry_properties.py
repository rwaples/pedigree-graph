"""Registry<->kinship oracle: the primary downstream-bias guard.

For each relationship code, a motif builder produces a single-lineage pedigree
guaranteed to contain a pair of exactly that code (one path, no inbreeding), so
the exact pairwise kinship equals the category's nominal kinship. The matrix
engine must also classify the pair under that code (non-vacuity).
"""

from __future__ import annotations

import numpy as np
import pytest

from pedigree_graph import RELATIONSHIPS, PedigreeGraph

# Codes whose single shared ancestor must be the father (else mother).
_SHARED_IS_FATHER = frozenset({"FO", "PHS"})


def _motif(up, down, n_anc, *, shared_is_mother):
    """Build a single-lineage pedigree containing an (up, down, n_anc) pair.

    Every marry-in partner is a fresh, never-reused founder, so the returned
    pair (a, b) shares exactly one most-recent-common-ancestor lineage.
    """
    ids: list[int] = []
    mo: list[int] = []
    fa: list[int] = []

    def add(m=-1, f=-1):
        i = len(ids)
        ids.append(i)
        mo.append(m)
        fa.append(f)
        return i

    if down == 0:
        # Lineal: b is the common ancestor; a is `up` meioses below.
        b = add()
        cur = b
        for _ in range(up):
            fr = add()
            cur = add(cur, fr) if shared_is_mother else add(fr, cur)
        a = cur
    else:
        if n_anc == 2:
            cm, cf = add(), add()
            a1, b1 = add(cm, cf), add(cm, cf)  # full sibs
        elif shared_is_mother:
            ca, pa, pb = add(), add(), add()
            a1, b1 = add(ca, pa), add(ca, pb)  # share mother ca (half sibs)
        else:
            ca, pa, pb = add(), add(), add()
            a1, b1 = add(pa, ca), add(pb, ca)  # share father ca
        a = a1
        for _ in range(up - 1):
            fr = add()
            a = add(a, fr)
        b = b1
        for _ in range(down - 1):
            fr = add()
            b = add(b, fr)

    return (
        np.array(ids, dtype=np.int64),
        np.array(mo, dtype=np.int64),
        np.array(fa, dtype=np.int64),
        a,
        b,
    )


@pytest.mark.parametrize("code", [c for c in RELATIONSHIPS if c != "MZ"])
def test_extracted_pair_kinship_matches_registry(code):
    category = RELATIONSHIPS[code]
    up, down = (category.up, 0) if category.down == 0 else (category.down, category.up)
    ids, mo, fa, a, b = _motif(up, down, category.ancestor_count, shared_is_mother=code not in _SHARED_IS_FATHER)
    pg = PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa)

    phi = pg.pair_kinship(np.array([a]), np.array([b]))[0]
    assert phi == pytest.approx(category.nominal_kinship)

    # Non-vacuity: the matrix engine must classify (a, b) under exactly this
    # code. Directed codes (parent-offspring, grandparent, ...) keep a
    # meaningful (descendant, ancestor) orientation rather than canonical lo<hi,
    # so accept either orientation.
    pairs = pg.relationship_pairs(max_degree=5)
    extracted = set(zip(pairs[code].first_rows.tolist(), pairs[code].second_rows.tolist(), strict=True))
    assert (a, b) in extracted or (b, a) in extracted


def test_mz_twin_kinship():
    ids = np.array([0, 1, 2, 3], dtype=np.int64)
    mo = np.array([-1, -1, 0, 0], dtype=np.int64)
    fa = np.array([-1, -1, 1, 1], dtype=np.int64)
    twins = np.array([-1, -1, 3, 2], dtype=np.int64)  # 2 and 3 are MZ
    pg = PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa, twin_ids=twins)

    phi = pg.pair_kinship(np.array([2]), np.array([3]))[0]
    assert phi == pytest.approx(RELATIONSHIPS["MZ"].nominal_kinship)
    pairs = pg.relationship_pairs(max_degree=5)
    assert (2, 3) in set(zip(pairs["MZ"].first_rows.tolist(), pairs["MZ"].second_rows.tolist(), strict=True))
