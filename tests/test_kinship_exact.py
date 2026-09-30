"""Kinship and F held to the exact rational definition and to exact metamorphic identities.

The differential oracles for kinship (``tests/oracle/pair_kinship.py``,
``inbreeding.py``, ``kinship_dp``) are earlier production code moved verbatim,
so a recurrence defect shared with the Rust kernel passes parity.
``tests/oracle/exact_kinship.py`` is Henderson's tabular recurrence in
exact scaled integers, written from the definition; these properties hold the kernel to
it and to identities that need no expected value at all.

When float32 is exact.  By induction over the recurrence,
``phi(a, b) * 2**(d_a + d_b + 2)`` is an integer: a founder's self kinship is
``1/2``; peeling ``a`` gives ``(phi(m, b) + phi(f, b)) / 2`` with both parents
at depth ``<= d_a - 1``; and the self step ``(1 + phi(m, f)) / 2`` has
``d_m + d_f <= 2 d_a - 2``.  Kinship is at most 1, so every value on the peel
path of a pair with ``d_a + d_b <= 22`` fits float32's 24-bit significand, and
every rounding is exact: the kernel's value *is* the rational.  Deeper pairs
round once per step, each step at most half a float32 ulp below 1
(``2**-25``, the correctly rounded half-sum ADR 0009 pins), and each step lowers
``d_a + d_b`` by at least 1, so ``|phi32 - phi| <= (d_a + d_b + 1) * 2**-25``.
That is half the ADR 0009 cross-order envelope, which bounds two such paths.

When float64 F is exact.  The Meuwissen-Luo kernel only adds and multiplies by
0.5, 0.25 and 0.75 (``crates/core/src/kinship/inbreeding.rs``).  With
``A = 2 phi``, ``A[i, i] * 2**(2 d_i + 1)`` is an integer; an ancestor ``j``'s
path weight is a multiple of ``2**-(d_i - d_j)`` and its within-family variance
a multiple of ``2**-(2 d_j + 1)``, so every term and partial sum is a multiple
of ``2**-(2 d_i + 1)`` below 2, and fits float64's 53 bits when
``d_i <= 25``: then ``F`` is exactly ``A[i, i] - 1``.

The peel rule itself is not checked here: a different valid rule stays inside
the bound.  ``test_pair_kinship.py::TestWithinGraphParity::
test_every_pair_matches_the_matrix_bit_for_bit[deep_inbred_60g]`` owns it;
peeling depth ties to the lesser row fails it and 14 other bitwise tests
(scratch mutation, 2026-09-30).
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _support import ENVELOPE_UNIT, EXACT_DEPTH_SUM, EXACT_F_DEPTH
from conftest import FIXTURES, parity_columns, pedigree_columns
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle.exact_kinship import exact_kinship

from pedigree_graph import PedigreeGraph

if TYPE_CHECKING:
    from oracle.exact_kinship import ExactKinship

_SETTINGS = settings(deadline=None, max_examples=150)

#: A row id above every id ``pedigree_columns`` makes, for rows a test appends.
_FRESH_ID = 50_000


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n)


def _assert_equals_exact(got: np.ndarray, first: np.ndarray, second: np.ndarray, exact: ExactKinship) -> None:
    """Each kernel value is the rational where float32 is exact, else inside ``(D + 1) * 2**-25``."""
    unit = 1 << (exact.shift - 25)
    depth, numerator = exact.depth, exact.numerator
    for a, b, value in zip(first.tolist(), second.tolist(), exact.scaled(got), strict=True):
        want = numerator[a][b]
        depth_sum = depth[a] + depth[b]
        if depth_sum <= EXACT_DEPTH_SUM:
            assert value == want, (a, b)
        else:
            assert abs(value - want) <= (depth_sum + 1) * unit, (a, b)
        assert (value == 0) == (want == 0), (a, b)


def _assert_same_kinship(before: np.ndarray, after: np.ndarray, depth_sum: np.ndarray) -> None:
    """Bit equality where both values are exact, the ADR 0009 two-path envelope beyond."""
    exact = depth_sum <= EXACT_DEPTH_SUM
    assert before[exact].tobytes() == after[exact].tobytes()
    gap = np.abs(before[~exact].astype(np.float64) - after[~exact].astype(np.float64))
    assert np.all(gap <= 2.0 * (depth_sum[~exact] + 1) * ENVELOPE_UNIT)


def _append(columns: dict, rows: dict) -> dict:
    return {key: np.concatenate([columns[key], np.asarray(rows[key], dtype=columns[key].dtype)]) for key in columns}


def _with_a_twin(columns: dict, data) -> tuple[dict, int, int]:
    """*columns* with a co-twin appended for a drawn row that has none; returns the pair's rows."""
    n = len(columns["id"])
    row = data.draw(st.sampled_from([r for r in range(n) if columns["twin"][r] == -1]))
    new_id = _FRESH_ID + 1
    grown = _append(
        columns,
        {
            "id": [new_id],
            "mother": [columns["mother"][row]],
            "father": [columns["father"][row]],
            "twin": [columns["id"][row]],
            "sex": [columns["sex"][row]],
        },
    )
    grown["twin"][row] = new_id
    return grown, row, n


def _closed_sets(columns: dict) -> list[frozenset[int]]:
    """Each row's genome and its represented ancestors' genomes, co-twins as the lower row."""
    ids = columns["id"].tolist()
    row_of = {v: r for r, v in enumerate(ids)}
    genome = [min(r, row_of[t]) if t != -1 else r for r, t in enumerate(columns["twin"].tolist())]
    closed: dict[int, frozenset[int]] = {}

    def visit(row: int) -> frozenset[int]:
        if row not in closed:
            members = {genome[row]}
            for parent in (columns["mother"][row], columns["father"][row]):
                if int(parent) in row_of:
                    members |= visit(row_of[int(parent)])
            closed[row] = frozenset(members)
        return closed[row]

    return [visit(r) for r in range(len(ids))]


@_SETTINGS
@given(pedigree_columns())
def test_kinship_and_inbreeding_equal_the_exact_rational(columns):
    # Rejects a recurrence defect the copied oracles share: a missing or
    # external parent counted as 1/2, the MZ self branch applied to the pair
    # only, a one-parent row peeled as a founder.
    graph = PedigreeGraph.from_frame(columns)
    exact = exact_kinship(columns["id"], columns["mother"], columns["father"], columns["twin"])
    assert np.asarray(graph.depth).tolist() == exact.depth
    first, second = _pairs(graph.n_individuals)
    _assert_equals_exact(graph.pair_kinship(first, second), first, second, exact)
    rows = [row for row in range(graph.n_individuals) if exact.depth[row] <= EXACT_F_DEPTH]
    inbreeding = exact.scaled(graph.inbreeding()[rows])
    for row, value in zip(rows, inbreeding, strict=True):
        assert value == 2 * exact.numerator[row][row] - (1 << exact.shift), row


@pytest.mark.parametrize("name", ["deep_inbred_60g", "twins_mating_loop", "external_parents", "one_parent_known"])
def test_fixture_kinship_equals_the_exact_rational(name):
    # deep_inbred_60g has pairs past the exact range, so it is the one
    # fixture where the bound, not equality, is what holds.
    columns = parity_columns(FIXTURES[name])
    graph = PedigreeGraph.from_frame(columns)
    exact = exact_kinship(columns["id"], columns["mother"], columns["father"], columns["twin"])
    first, second = _pairs(graph.n_individuals)
    _assert_equals_exact(graph.pair_kinship(first, second), first, second, exact)


@_SETTINGS
@given(pedigree_columns())
def test_zero_kinship_is_exactly_disjoint_ancestry(columns):
    # Set logic, no recurrence.  Rejects a memo-key collision, a stored
    # explicit zero, and a spurious nonzero through a co-twin lookup.
    graph = PedigreeGraph.from_frame(columns)
    closed = _closed_sets(columns)
    n = graph.n_individuals
    first, second = _pairs(n)
    related = np.array([not closed[a].isdisjoint(closed[b]) for a, b in zip(first, second, strict=True)], dtype=bool)
    np.testing.assert_array_equal(graph.pair_kinship(first, second) != 0, related)

    matrix = graph.kinship_matrix().tocoo()
    stored = {(int(a), int(b)) for a, b in zip(matrix.row, matrix.col, strict=True)}
    expected = {(a, b) for a, b, r in zip(first.tolist(), second.tolist(), related.tolist(), strict=True) if r}
    assert stored == expected | {(b, a) for a, b in expected}

    row_of = {v: r for r, v in enumerate(columns["id"].tolist())}
    inbreeding = graph.inbreeding()
    for row in range(n):
        m, f = row_of.get(int(columns["mother"][row])), row_of.get(int(columns["father"][row]))
        inbred = m is not None and f is not None and not closed[m].isdisjoint(closed[f])
        assert (inbreeding[row] != 0) == inbred, row


@_SETTINGS
@given(pedigree_columns(max_n=25), pedigree_columns(max_n=15), st.data())
def test_appended_rows_leave_every_original_value_bit_identical(columns, other, data):
    # Rejects dependence on a global row index and DP slot reuse that
    # unrelated rows can trigger.  Appending after the original rows keeps
    # their depth and row order, so the pinned peel path is the same and the
    # values are bit-identical at any depth.
    n = len(columns["id"])
    shifted = {
        key: np.where(value == -1, -1, value + 100_000) if key != "sex" else value for key, value in other.items()
    }
    grown = _append(columns, shifted)
    females = [int(v) for v, s in zip(columns["id"], columns["sex"], strict=True) if s == 0]
    males = [int(v) for v, s in zip(columns["id"], columns["sex"], strict=True) if s == 1]
    for k in range(data.draw(st.integers(min_value=0, max_value=4))):
        grown = _append(
            grown,
            {
                "id": [_FRESH_ID + k],
                "mother": [data.draw(st.sampled_from([-1, *females]))],
                "father": [data.draw(st.sampled_from([-1, *males]))],
                "twin": [-1],
                "sex": [data.draw(st.integers(min_value=0, max_value=1))],
            },
        )

    base, big = PedigreeGraph.from_frame(columns), PedigreeGraph.from_frame(grown)
    first, second = _pairs(n)
    assert big.pair_kinship(first, second).tobytes() == base.pair_kinship(first, second).tobytes()
    assert big.inbreeding()[:n].tobytes() == base.inbreeding().tobytes()
    m = len(shifted["id"])
    cross_first, cross_second = np.repeat(np.arange(n), m), np.tile(np.arange(n, n + m), n)
    assert not big.pair_kinship(cross_first, cross_second).any()


def test_appending_a_disjoint_copy_leaves_the_deep_fixture_bit_identical():
    # The generated pedigrees are shallow enough that every value is exact;
    # here rounding is live, so bit identity is the pinned rule's, not arithmetic's.
    columns = parity_columns(FIXTURES["deep_inbred_60g"])
    copy = {
        key: np.where(value == -1, -1, value + 10_000_000) if key != "sex" else value for key, value in columns.items()
    }
    base, big = PedigreeGraph.from_frame(columns), PedigreeGraph.from_frame(_append(columns, copy))
    first, second = _pairs(len(columns["id"]))
    assert big.pair_kinship(first, second).tobytes() == base.pair_kinship(first, second).tobytes()
    assert big.inbreeding()[: base.n_individuals].tobytes() == base.inbreeding().tobytes()


@_SETTINGS
@given(pedigree_columns(max_n=30).filter(lambda c: len(c["id"]) > 0), st.data())
def test_completing_or_externalising_a_missing_parent_changes_no_value(columns, data):
    # Rejects a one-parent row treated as a founder and an external id
    # resolved to a row.  Co-twins must share parents, so the replacement goes
    # to both; a one-sided change would test the refusal, not kinship.
    n = len(columns["id"])
    open_slots = [(row, key) for row in range(n) for key in ("mother", "father") if columns[key][row] == -1]
    row, key = data.draw(st.sampled_from(open_slots))
    completion = data.draw(st.sampled_from(["fresh_founder", "external"]))
    new_parent = _FRESH_ID if completion == "fresh_founder" else 77_777
    twin_row = {int(v): r for r, v in enumerate(columns["id"])}.get(int(columns["twin"][row]))

    changed = {k: v.copy() for k, v in columns.items()}
    for target in {row, twin_row} - {None}:
        changed[key][target] = new_parent
    if completion == "fresh_founder":
        sex = 0 if key == "mother" else 1
        changed = _append(changed, {"id": [new_parent], "mother": [-1], "father": [-1], "twin": [-1], "sex": [sex]})

    base, after = PedigreeGraph.from_frame(columns), PedigreeGraph.from_frame(changed)
    first, second = _pairs(n)
    depth_sum = np.maximum(base.depth[first] + base.depth[second], after.depth[first] + after.depth[second])
    _assert_same_kinship(base.pair_kinship(first, second), after.pair_kinship(first, second), depth_sum)
    exact = np.maximum(base.depth, after.depth[:n]) <= EXACT_F_DEPTH
    assert after.inbreeding()[:n][exact].tobytes() == base.inbreeding()[exact].tobytes()


@_SETTINGS
@given(pedigree_columns())
def test_swapping_parent_roles_changes_no_value(columns):
    # Rejects sex-asymmetric memo canonicalisation.  Depth and row order are
    # unchanged and the half-sum commutes, so kinship is bit-identical at any
    # depth; F is compared where it is exact.
    swapped = {
        **columns,
        "mother": columns["father"],
        "father": columns["mother"],
        "sex": (1 - columns["sex"]).astype(np.int8),
    }
    base, mirror = PedigreeGraph.from_frame(columns), PedigreeGraph.from_frame(swapped)
    first, second = _pairs(base.n_individuals)
    assert mirror.pair_kinship(first, second).tobytes() == base.pair_kinship(first, second).tobytes()
    exact = np.asarray(base.depth) <= EXACT_F_DEPTH
    assert mirror.inbreeding()[exact].tobytes() == base.inbreeding()[exact].tobytes()


@_SETTINGS
@given(pedigree_columns().filter(lambda c: (c["twin"] == -1).any()), st.data())
def test_a_co_twin_substitutes_for_its_twin(columns, data):
    # Rejects twin handling applied only to the pair (t, t'), the issue #5
    # class.  The two rows peel at different row positions, so a depth tie
    # can split their paths: bits where exact, the envelope beyond.
    grown, t, t_prime = _with_a_twin(columns, data)
    graph = PedigreeGraph.from_frame(grown)
    others = np.array([r for r in range(graph.n_individuals) if r not in (t, t_prime)], dtype=np.int64)
    via_t = graph.pair_kinship(np.full(len(others), t), others)
    via_t_prime = graph.pair_kinship(np.full(len(others), t_prime), others)
    _assert_same_kinship(via_t, via_t_prime, graph.depth[t] + graph.depth[others])
    assert graph.inbreeding()[t] == graph.inbreeding()[t_prime]
    self_kinship = graph.pair_kinship(np.array([t]), np.array([t]))
    assert graph.pair_kinship(np.array([t]), np.array([t_prime])).tobytes() == self_kinship.tobytes()


@_SETTINGS
@given(pedigree_columns(), st.data())
def test_mean_kinship_by_generation_is_the_mean_over_genome_pairs(columns, data):
    # Rejects cohort-grouping and MZ-collapse errors on random topologies.
    # The representative of a twin pair is the co-twin with a label, the
    # earlier label when both have one (``mean_kinship_by_generation``).
    n = len(columns["id"])
    labels = np.array(
        data.draw(st.lists(st.integers(min_value=-1, max_value=3), min_size=n, max_size=n)), dtype=np.int64
    )
    graph = PedigreeGraph.from_frame({**columns, "generation": labels})
    # Unlabelled rows are counted as rows, a collapsed co-twin included
    # (``_ne_rates.py``); a wholly unlabelled pedigree groups by structural
    # depth instead and counts none (CONTEXT.md, "generation label").
    unlabelled = int((labels == -1).sum())
    if unlabelled == n:
        assert graph.generation_labels is None
        labels, unlabelled = np.asarray(graph.depth, dtype=np.int64), 0
    row_of = {int(v): r for r, v in enumerate(columns["id"])}

    representatives = []
    for row in range(n):
        twin = row_of.get(int(columns["twin"][row]))
        if twin is None:
            representatives.append(row)
            continue
        key = {r: (labels[r] == -1, labels[r], r) for r in (row, twin)}
        if min((row, twin), key=key.get) == row:
            representatives.append(row)

    groups = sorted({int(labels[r]) for r in representatives if labels[r] != -1})
    summary = graph.mean_kinship_by_generation()
    assert summary.generations.tolist() == groups
    assert summary.unlabelled_individual_count == unlabelled
    for index, label in enumerate(groups):
        members = np.array([r for r in representatives if labels[r] == label], dtype=np.int64)
        first, second = np.triu_indices(len(members), k=1)
        pairs = len(first)
        assert summary.pair_counts[index] == pairs
        if pairs == 0:
            assert math.isnan(summary.mean_kinship[index])
            continue
        values = graph.pair_kinship(members[first], members[second]).astype(np.float64)
        want = math.fsum(values) / pairs
        # The kernel sums the float32 values in float64 in its own order, at
        # most (pairs - 1) roundings of 2**-53 relative to the nonnegative sum,
        # then divides once; fsum is exact and the reference divides once.
        assert abs(summary.mean_kinship[index] - want) <= (pairs + 3) * 2.0**-53 * want


def _support(matrix) -> set[tuple[int, int]]:
    coo = matrix.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist(), strict=True))


@_SETTINGS
@given(
    pedigree_columns(),
    st.lists(st.sampled_from([0.0, 2.0**-6, 2.0**-4, 0.1, 0.25, 0.3, 0.5, 1.0]), min_size=2, max_size=4),
)
def test_approximate_support_shrinks_as_the_threshold_rises(columns, thresholds):
    # The thresholded pass reads a pruned entry as 0 and drops a value at or
    # below the threshold (``matrix.rs``, ``Dp``); the float32 half-sum is
    # monotone in its inputs, so a higher threshold only lowers propagated
    # values and cuts more.  By induction the support is nested, and at
    # threshold 0 nothing positive is dropped.  Rejects a comparison with the
    # wrong sense, a zero threshold that prunes, and a support that is not
    # the complete matrix's.
    graph = PedigreeGraph.from_frame(columns)
    complete = _support(graph.kinship_matrix())
    assert _support(graph.approximate_kinship_matrix(min_propagated_kinship=0.0)) == complete
    supports = [_support(graph.approximate_kinship_matrix(min_propagated_kinship=t)) for t in sorted(thresholds)]
    for looser, stricter in itertools.pairwise(supports):
        assert stricter <= looser <= complete
