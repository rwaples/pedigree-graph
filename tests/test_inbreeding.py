"""``inbreeding`` on graphs: the memoised MZ-aware Meuwissen-Luo F (ADR 0008).

On the ADR 0008 fixtures F is exactly ``2 * phi(i, i) - 1`` against both the
``pair_kinship`` self pair and the ``kinship_matrix`` diagonal, and matches the
hand-derived values the fixture table carries; the classic matings (sib,
half-sib, parent-offspring, the Crow & Kimura closed line) carry their
textbook values.  Sixty generations of accumulated
inbreeding hold that identity inside ``2**-22``.  The array is float64 and
read-only, and it is computed once: a second call hands back the same object
without re-entering the kernel, and the call commits the package thread budget.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _support import ADR_0008_FIXTURES, _mz_frame
from conftest import parity_columns, parity_fixtures

import pedigree_graph._core
from pedigree_graph import PedigreeGraph
from pedigree_graph._threads import configure_threads

DEEP_FIXTURE = parity_fixtures("deep_inbred_60g")["deep_inbred_60g"]
DEEP_ENVELOPE = 2.0**-22

MZ_ONLY_LINK = next(case for case in ADR_0008_FIXTURES if case[0] == "mz_only_link")

MZ_CASES = pytest.mark.parametrize("case", ADR_0008_FIXTURES, ids=[case[0] for case in ADR_0008_FIXTURES])


def _graph(case) -> PedigreeGraph:
    _name, mother, father, twin, _expected = case
    return PedigreeGraph.from_frame(_mz_frame(list(range(len(mother))), mother, father, twin))


def _self_kinship_identity(graph: PedigreeGraph) -> np.ndarray:
    rows = np.arange(graph.n_individuals)
    return 2.0 * graph.pair_kinship(rows, rows).astype(np.float64) - 1.0


def _diagonal_identity(graph: PedigreeGraph) -> np.ndarray:
    return 2.0 * graph.kinship_matrix().diagonal().astype(np.float64) - 1.0


def _forbid_kernel(*_args, **_kwargs):
    raise AssertionError("inbreeding() re-entered the Meuwissen-Luo kernel after the memo was populated")


@MZ_CASES
def test_mz_fixture_values(case):
    name, _mother, _father, _twin, expected = case
    F = _graph(case).inbreeding()
    for row, value in expected.items():
        assert F[row] == value, f"{name} row {row}"


@MZ_CASES
def test_mz_fixture_self_kinship_identity_is_exact(case):
    name = case[0]
    graph = _graph(case)
    F = graph.inbreeding()
    np.testing.assert_array_equal(F, _self_kinship_identity(graph), err_msg=name)
    np.testing.assert_array_equal(F, _diagonal_identity(graph), err_msg=name)


def test_deep_inbred_60g_holds_the_identity_inside_the_envelope():
    graph = PedigreeGraph.from_frame(parity_columns(DEEP_FIXTURE))
    F = graph.inbreeding()
    assert np.abs(F - _self_kinship_identity(graph)).max() <= DEEP_ENVELOPE
    assert np.abs(F - _diagonal_identity(graph)).max() <= DEEP_ENVELOPE


@pytest.mark.parametrize(
    ("mother", "father", "expected"),
    [
        ([-1, -1, -1], [-1, -1, -1], [0.0, 0.0, 0.0]),  # founders
        ([-1, -1, 0, -1], [-1, -1, -1, 0], [0.0] * 4),  # one known parent breaks every path
        ([-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3], [0, 0, 0, 0, 0.25]),  # full-sib mating
        ([-1, -1, -1, 0, 0, 3], [-1, -1, -1, 1, 2, 4], [0, 0, 0, 0, 0, 0.125]),  # half-sib mating
        ([-1, -1, 0, 0], [-1, -1, 1, 2], [0, 0, 0, 0.25]),  # mother x her child
        (  # Crow & Kimura full-sib closed line: F_5 = 5/8 - 1/32
            [-1, -1, 0, 0, 2, 2, 4, 4, 6, 6, 8],
            [-1, -1, 1, 1, 3, 3, 5, 5, 7, 7, 9],
            [0, 0, 0, 0, 0.25, 0.25, 0.375, 0.375, 0.5, 0.5, 0.59375],
        ),
        ([-1, *range(14)], [-1] * 15, [0.0] * 15),  # 15-generation single-parent chain
    ],
    ids=["founders", "one_parent", "full_sib", "half_sib", "parent_offspring", "closed_line", "chain_15"],
)
def test_hand_derived_values(mother, father, expected):
    n = len(mother)
    F = PedigreeGraph.from_arrays(
        ids=np.arange(n), mother_ids=np.array(mother), father_ids=np.array(father)
    ).inbreeding()
    np.testing.assert_allclose(F, expected, rtol=0, atol=1e-12)


def test_skip_generation_edges_match_the_matrix_diagonal():
    # 8 = (3, 6) mates across a generation gap.
    pg = PedigreeGraph.from_arrays(
        ids=np.arange(10),
        mother_ids=np.array([-1, -1, -1, -1, 1, 1, 5, 3, 3, 7]),
        father_ids=np.array([-1, -1, -1, -1, 0, 0, 4, 2, 6, 6]),
    )
    np.testing.assert_allclose(pg.inbreeding(), _diagonal_identity(pg), atol=1e-12)


@pytest.mark.parametrize("strip_twins", [False, True])
def test_shipped_parquet_matches_the_matrix_diagonal(small_pedigree, strip_twins):
    if strip_twins:
        small_pedigree = small_pedigree.with_columns(pl.lit(-1).cast(small_pedigree.schema["twin"]).alias("twin"))
    pg = PedigreeGraph.from_frame(small_pedigree)
    np.testing.assert_allclose(pg.inbreeding(), _diagonal_identity(pg), atol=1e-10)


def test_absent_co_twin_is_not_an_mz_pair():
    # Co-twin outside the subsample remaps to -1: the row is an ordinary individual.
    full = _mz_frame([0, 1, 2, 3, 4], [-1, -1, 0, 0, 2], [-1, -1, 1, 1, 3], [-1, -1, 3, 2, -1])
    F = PedigreeGraph.from_frame(full.filter(pl.col("id") != 3)).inbreeding()
    assert F[3] == 0.0


def test_second_call_returns_the_memo_without_recomputing(monkeypatch):
    graph = _graph(MZ_ONLY_LINK)
    first = graph.inbreeding()
    monkeypatch.setattr(pedigree_graph._native, "inbreeding", _forbid_kernel)
    assert graph.inbreeding() is first


def test_result_is_read_only():
    F = _graph(MZ_ONLY_LINK).inbreeding()
    with pytest.raises(ValueError, match="read-only"):
        F[0] = 0.5


def test_result_is_float64():
    assert _graph(MZ_ONLY_LINK).inbreeding().dtype == np.float64


@pytest.mark.usefixtures("fresh_thread_state")
class TestThreads:
    def test_computing_call_commits_the_budget(self):
        _graph(MZ_ONLY_LINK).inbreeding()
        with pytest.raises(RuntimeError):
            configure_threads(3)
