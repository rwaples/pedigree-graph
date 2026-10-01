"""Relationship classification held to its definitions and to exact metamorphic identities.

The engine's differential oracle (``tests/oracle/relationship_pairs.py``) is
the 0.8 engine moved verbatim and changed in lockstep with the Rust engine, so
it catches algorithm defects but not definition defects the two share.
``tests/oracle/relationship_paths.py`` states the ADR 0010 definitions by
explicit path enumeration; these properties hold the engine to both oracles on
general topologies and to identities that need no expected pair list.

Every registry code occurs in every run: the spec differential is
parametrized over the 23 codes and appends that code's motif
(``_support._motif``) to the drawn pedigree.
"""

from __future__ import annotations

import numpy as np
import pytest
from _support import _SHARED_IS_FATHER, _motif
from conftest import pedigree_columns
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle.exact_kinship import exact_kinship
from oracle.relationship_pairs import oracle_pairs, oracle_view_pairs
from oracle.relationship_paths import relationship_paths

from pedigree_graph import RELATIONSHIPS, PedigreeGraph

_SETTINGS = settings(deadline=None, max_examples=100)
_MOTIF_OFFSET = 1_000_000


def _pair_sets(pairs) -> dict[str, set[tuple[int, int]]]:
    return {code: set(zip(b.first_rows.tolist(), b.second_rows.tolist(), strict=True)) for code, b in pairs.items()}


def _append(columns: dict, rows: dict) -> dict:
    return {key: np.concatenate([columns[key], np.asarray(rows[key], dtype=columns[key].dtype)]) for key in columns}


def _motif_rows(code: str) -> tuple[dict, int, int]:
    """Columns of *code*'s single-lineage motif at ids above ``_MOTIF_OFFSET``, and its pair's local rows."""
    category = RELATIONSHIPS[code]
    if code == "MZ":
        ids = np.arange(4, dtype=np.int64)
        mother, father = np.array([-1, -1, 0, 0]), np.array([-1, -1, 1, 1])
        twin = np.array([-1, -1, 3, 2])
        a, b = 2, 3
    else:
        up, down = (category.up, 0) if category.down == 0 else (category.down, category.up)
        ids, mother, father, a, b = _motif(
            up, down, category.ancestor_count, shared_is_mother=code not in _SHARED_IS_FATHER
        )
        twin = np.full(len(ids), -1)
    sex = np.array([1 if i in set(father.tolist()) else 0 for i in ids.tolist()], dtype=np.int8)
    shift = lambda v: np.where(v == -1, -1, v + _MOTIF_OFFSET)  # noqa: E731
    return (
        {"id": ids + _MOTIF_OFFSET, "mother": shift(mother), "father": shift(father), "twin": shift(twin), "sex": sex},
        a,
        b,
    )


def _paths(columns: dict, max_degree: int = 5) -> dict[str, set[tuple[int, int]]]:
    return relationship_paths(
        columns["id"], columns["mother"], columns["father"], columns["twin"], max_degree=max_degree
    )


@settings(deadline=None, max_examples=150)
@given(pedigree_columns(), st.integers(min_value=0, max_value=5))
def test_engine_equals_the_copied_oracle_on_generated_pedigrees(columns, max_degree):
    # The copied oracle has only seen fixtures; this feeds it the general
    # strategy.  Rejects row-streaming scratch leaks and compaction errors on
    # topologies no fixture has, in graph and view coordinates alike.
    graph = PedigreeGraph.from_frame(columns)
    got = graph.relationship_pairs(max_degree=max_degree)
    for code, (first, second) in oracle_pairs(graph, max_degree=max_degree).items():
        np.testing.assert_array_equal(got[code].first_rows, first, err_msg=code)
        np.testing.assert_array_equal(got[code].second_rows, second, err_msg=code)
    if graph.n_individuals >= 2:
        view = graph.view(rows=np.arange(graph.n_individuals)[::2])
        got = view.relationship_pairs(max_degree=max_degree)
        for code, (first, second) in oracle_view_pairs(view, max_degree=max_degree).items():
            np.testing.assert_array_equal(got[code].first_rows, first, err_msg=code)
            np.testing.assert_array_equal(got[code].second_rows, second, err_msg=code)


@pytest.mark.parametrize("code", list(RELATIONSHIPS))
@settings(deadline=None, max_examples=25)
@given(columns=pedigree_columns(max_n=30))
def test_engine_equals_the_path_definitions(code, columns):
    # Rejects ``> 0`` in place of ``>= 2``, multiplicity collapsed before the
    # full/half test, and a swapped exclusion row: the definitions, not the
    # algorithm.  The appended motif makes *code* occur in every draw.
    motif, a, b = _motif_rows(code)
    grown = _append(columns, motif)
    graph = PedigreeGraph.from_frame(grown)
    got = _pair_sets(graph.relationship_pairs(max_degree=5))
    want = _paths(grown)
    assert got == want
    n = len(columns["id"])
    assert frozenset((n + a, n + b)) in {frozenset(pair) for pair in got[code]}


@_SETTINGS
@given(pedigree_columns())
def test_a_shared_parent_id_is_exactly_one_sibling_or_parent_code(columns):
    # By id, external ids included: FS when both ids are shared, MHS / PHS
    # when only the mother / father is, MZ for co-twins and MO / FO for a
    # parent and child first.  Rejects a maternal/paternal swap, which
    # simACE's maternal-household C turns into a bias.
    graph = PedigreeGraph.from_frame(columns)
    code_of = {
        frozenset(pair): code
        for code, pairs in _pair_sets(graph.relationship_pairs(max_degree=2)).items()
        for pair in pairs
    }
    ids = columns["id"].tolist()
    row_of = {v: r for r, v in enumerate(ids)}
    mother, father, twin = columns["mother"].tolist(), columns["father"].tolist(), columns["twin"].tolist()
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            same_mother = mother[a] != -1 and mother[a] == mother[b]
            same_father = father[a] != -1 and father[a] == father[b]
            if not (same_mother or same_father):
                continue
            if twin[a] != -1 and row_of[twin[a]] == b:
                want = "MZ"
            elif row_of.get(mother[a]) == b or row_of.get(mother[b]) == a:
                want = "MO"
            elif row_of.get(father[a]) == b or row_of.get(father[b]) == a:
                want = "FO"
            elif same_mother and same_father:
                want = "FS"
            else:
                want = "MHS" if same_mother else "PHS"
            assert code_of.get(frozenset((a, b))) == want, (a, b)


@_SETTINGS
@given(pedigree_columns())
def test_a_classified_pair_is_at_least_as_related_as_its_code(columns):
    # With every parent id resolved to a row, one path of a code's shape
    # already carries its nominal kinship, so exact kinship is at least that
    # much; an inequality, because double first cousins exceed 1C's nominal
    # value.  Rejects a pair classified closer than it is.  Kinship is the
    # exact oracle's, the kernel being held to it in test_kinship_exact.py.
    rows = set(columns["id"].tolist())
    resolved = {
        **columns,
        "mother": np.array([v if v in rows else -1 for v in columns["mother"].tolist()], dtype=np.int64),
        "father": np.array([v if v in rows else -1 for v in columns["father"].tolist()], dtype=np.int64),
    }
    graph = PedigreeGraph.from_frame(resolved)
    exact = exact_kinship(resolved["id"], resolved["mother"], resolved["father"], resolved["twin"])
    for code, pairs in _pair_sets(graph.relationship_pairs(max_degree=5)).items():
        nominal = int(RELATIONSHIPS[code].nominal_kinship * 2**exact.shift)
        for a, b in pairs:
            assert exact.numerator[a][b] >= nominal, (code, a, b)


def test_siblings_by_a_shared_external_parent_are_full_sibs_with_zero_kinship():
    # Pinned so a change to this semantic is deliberate (plan open question 1):
    # siblings are grouped by original parent id, and a person with only
    # external parents is a represented founder.
    graph = PedigreeGraph.from_frame(
        {"id": np.array([10, 11]), "mother": np.array([900, 900]), "father": np.array([901, 901])}
    )
    pairs = graph.relationship_pairs(max_degree=1)
    assert _pair_sets(pairs)["FS"] == {(0, 1)}
    assert graph.pair_kinship(np.array([0]), np.array([1])).tolist() == [0.0]


def _full_sib_pair(columns: dict, data) -> tuple[dict, int, int]:
    """*columns* plus a new row that is a full sib, of the same sex, of a drawn row with both parent ids known.

    When no row has both parent ids known, a child of two drawn founders of
    each sex is appended first, so the pair always exists.
    """
    grown = {key: value.copy() for key, value in columns.items()}
    known = [
        r
        for r in range(len(grown["id"]))
        if grown["mother"][r] != -1 and grown["father"][r] != -1 and grown["twin"][r] == -1
    ]
    if not known:
        grown = _append(
            grown, {"id": [60_001, 60_002], "mother": [-1, -1], "father": [-1, -1], "twin": [-1, -1], "sex": [0, 1]}
        )
        grown = _append(grown, {"id": [60_003], "mother": [60_001], "father": [60_002], "twin": [-1], "sex": [0]})
        known = [len(grown["id"]) - 1]
    row = data.draw(st.sampled_from(known))
    grown = _append(
        grown,
        {
            "id": [60_100],
            "mother": [grown["mother"][row]],
            "father": [grown["father"][row]],
            "twin": [-1],
            "sex": [grown["sex"][row]],
        },
    )
    return grown, row, len(grown["id"]) - 1


@_SETTINGS
@given(pedigree_columns(max_n=30), st.data())
def test_a_twin_link_between_full_sibs_moves_only_that_pair(columns, data):
    # Co-twins take part in sibling groups (issue #29) and every pair through
    # a twin keeps its structural category, so linking two full sibs changes
    # exactly one pair, FS to MZ.  Then each co-twin stands in for the other:
    # (t, r) and (t', r) share code and role for every r outside the pair and
    # its descendants.  Rejects twin handling applied only to the pair.
    base, t, t_prime = _full_sib_pair(columns, data)
    linked = {**base, "twin": base["twin"].copy()}
    linked["twin"][t], linked["twin"][t_prime] = base["id"][t_prime], base["id"][t]
    before = _pair_sets(PedigreeGraph.from_frame(base).relationship_pairs(max_degree=5))
    after = _pair_sets(PedigreeGraph.from_frame(linked).relationship_pairs(max_degree=5))
    pair = (min(t, t_prime), max(t, t_prime))
    assert before["FS"] - after["FS"] == {pair}
    assert after["MZ"] - before["MZ"] == {pair}
    assert {c: v for c, v in before.items() if c not in ("FS", "MZ")} == {
        c: v for c, v in after.items() if c not in ("FS", "MZ")
    }

    descendants = set()
    frontier = {int(base["id"][t]), int(base["id"][t_prime])}
    while frontier:
        children = {
            int(i)
            for i, m, f in zip(base["id"], base["mother"], base["father"], strict=True)
            if m in frontier or f in frontier
        }
        frontier = children - descendants
        descendants |= children
    row_of = {int(v): r for r, v in enumerate(base["id"])}
    outside = set(range(len(base["id"]))) - {t, t_prime} - {row_of[d] for d in descendants}

    def roles(me: int) -> dict[int, tuple[str, bool]]:
        return {
            (b if a == me else a): (code, a == me)
            for code, pairs in after.items()
            for a, b in pairs
            if me in (a, b) and (b if a == me else a) in outside
        }

    # An asymmetric pair valid in both orientations of an inbred pedigree
    # puts its lower row first (ADR 0006), so when t and t' straddle r the
    # roles may differ, but only as that tie-break: both lower-row first.
    for_t, for_t_prime = roles(t), roles(t_prime)
    for r in outside:
        want, got = for_t.get(r), for_t_prime.get(r)
        assert (got and got[0]) == (want and want[0]), r
        if want is not None and not RELATIONSHIPS[want[0]].symmetric and got != want:
            assert want[1] == (t < r), r
            assert got[1] == (t_prime < r), r


@_SETTINGS
@given(pedigree_columns(max_n=25), pedigree_columns(max_n=15), st.data())
def test_appended_descendants_and_families_leave_original_pairs_unchanged(columns, other, data):
    # Rejects classification that reads a global row index or a sibling
    # group polluted by unrelated rows; no pair crosses the two families.
    n = len(columns["id"])
    shifted = {k: (np.where(v == -1, -1, v + 100_000) if k != "sex" else v) for k, v in other.items()}
    grown = _append(columns, shifted)
    females = [int(v) for v, s in zip(columns["id"], columns["sex"], strict=True) if s == 0]
    males = [int(v) for v, s in zip(columns["id"], columns["sex"], strict=True) if s == 1]
    for k in range(data.draw(st.integers(min_value=0, max_value=3))):
        mate_sex = data.draw(st.integers(min_value=0, max_value=1))
        mate = 70_000 + 2 * k
        parent = data.draw(st.sampled_from([-1, *(males if mate_sex == 0 else females)]))
        mother, father = (mate, parent) if mate_sex == 0 else (parent, mate)
        grown = _append(grown, {"id": [mate], "mother": [-1], "father": [-1], "twin": [-1], "sex": [mate_sex]})
        grown = _append(grown, {"id": [mate + 1], "mother": [mother], "father": [father], "twin": [-1], "sex": [0]})

    before = _pair_sets(PedigreeGraph.from_frame(columns).relationship_pairs(max_degree=5))
    after = _pair_sets(PedigreeGraph.from_frame(grown).relationship_pairs(max_degree=5))
    m = len(shifted["id"])
    for code, pairs in after.items():
        assert {(a, b) for a, b in pairs if a < n and b < n} == before[code], code
        assert not {(a, b) for a, b in pairs if (a < n <= b < n + m) or (b < n <= a < n + m)}, code


@settings(deadline=None, max_examples=60)
@given(pedigree_columns())
def test_every_single_category_selection_equals_its_full_run_block(columns):
    # Today four codes are checked (test_relationship_pairs.py).  Rejects the
    # "lower degree not computed below the cutoff" class for all 23.
    graph = PedigreeGraph.from_frame(columns)
    full = graph.relationship_pairs(max_degree=5)
    for code in RELATIONSHIPS:
        alone = graph.relationship_pairs(categories=[code])
        np.testing.assert_array_equal(alone[code].first_rows, full[code].first_rows, err_msg=code)
        np.testing.assert_array_equal(alone[code].second_rows, full[code].second_rows, err_msg=code)


_ID_MAPS = {
    "reversed": lambda ids: ids.max(initial=0) - ids,
    "near_int64_max": lambda ids: np.iinfo(np.int64).max - ids,
    "above_int32": lambda ids: 2**40 + 7 * ids,
}


@pytest.mark.parametrize("mapping", list(_ID_MAPS))
@_SETTINGS
@given(columns=pedigree_columns())
def test_pairs_in_id_space_are_invariant_under_relabelling(mapping, columns):
    # Only counts were compared under relabelling before.  Rejects a pair
    # emitted by old row index or narrowed id.
    universe = np.unique(np.concatenate([columns[k][columns[k] != -1] for k in ("id", "mother", "father")]))
    new = dict(zip(universe.tolist(), _ID_MAPS[mapping](universe).tolist(), strict=True))
    relabel = lambda v: np.array([new.get(int(x), -1) if x != -1 else -1 for x in v], dtype=np.int64)  # noqa: E731
    mapped = {**columns, **{k: relabel(columns[k]) for k in ("id", "mother", "father", "twin")}}

    def by_id(cols: dict) -> dict[str, set[tuple[int, int]]]:
        ids = cols["id"].tolist()
        pairs = _pair_sets(PedigreeGraph.from_frame(cols).relationship_pairs(max_degree=5))
        return {code: {(ids[a], ids[b]) for a, b in block} for code, block in pairs.items()}

    original = by_id(columns)
    assert by_id(mapped) == {code: {(new[a], new[b]) for a, b in block} for code, block in original.items()}
