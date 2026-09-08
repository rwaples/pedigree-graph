"""Property-based tests for relationship-pair extraction integrity.

Pair keys are canonical and duplicate-free; full vs half sibling classification
is consistent with the parent arrays (the >=2-shared-ancestors rule); and pair
counts are invariant under id relabelling (incl. large/sparse ids).
"""

from __future__ import annotations

import numpy as np
from conftest import pedigree_arrays, random_pedigree, relabel_pedigree
from hypothesis import given, settings
from hypothesis import strategies as st

from pedigree_graph import PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=50)


@_SETTINGS
@given(pg=random_pedigree())
def test_pair_keys_no_self_pairs_and_unique(pg):
    # Symmetric codes are stored canonical lo<hi; directed codes (parent-
    # offspring, grandparent, ...) keep a (descendant, ancestor) orientation.
    # The invariant common to both: no self-pairs and no duplicate unordered pair.
    for code, (i1, i2) in pg.relationship_pairs(max_degree=5).items():
        assert np.all(i1 != i2), code  # no self-pairs
        unordered = [(min(x, y), max(x, y)) for x, y in zip(i1.tolist(), i2.tolist(), strict=True)]
        assert len(unordered) == len(set(unordered)), code  # no duplicate unordered pairs


@_SETTINGS
@given(pg=random_pedigree())
def test_full_vs_half_sib_classification(pg):
    pairs = pg.relationship_pairs(max_degree=5)
    mo, fa = pg.mother_rows, pg.father_rows

    for i, j in zip(*pairs["FS"], strict=True):
        assert mo[i] != -1
        assert fa[i] != -1
        assert mo[i] == mo[j]
        assert fa[i] == fa[j]

    for i, j in zip(*pairs["MHS"], strict=True):
        assert mo[i] != -1  # share mother
        assert mo[i] == mo[j]
        assert not (fa[i] != -1 and fa[i] == fa[j])  # but not full sibs

    for i, j in zip(*pairs["PHS"], strict=True):
        assert fa[i] != -1  # share father
        assert fa[i] == fa[j]
        assert not (mo[i] != -1 and mo[i] == mo[j])  # but not full sibs


@_SETTINGS
@given(arrays=pedigree_arrays(), data=st.data())
def test_pair_counts_id_relabel_invariant(arrays, data):
    ids, mo, fa, sex = arrays
    pg1 = PedigreeGraph.from_arrays(ids=ids, mother_ids=mo, father_ids=fa, sex=sex)
    pg2 = relabel_pedigree(arrays, data)
    assert dict(pg1.relationship_counts(max_degree=5)) == dict(pg2.relationship_counts(max_degree=5))
