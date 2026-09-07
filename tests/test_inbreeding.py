"""``inbreeding`` on graphs: the memoised MZ-aware Meuwissen-Luo F (ADR 0008, slice 5c).

On the ADR 0008 fixtures F is exactly ``2 * phi(i, i) - 1`` against both the
``pair_kinship`` self pair and the ``kinship_matrix`` diagonal, and matches the
hand-derived values the fixture table carries.  Sixty generations of accumulated
inbreeding hold that identity inside ``2**-22``.  The array is float64 and
read-only, and it is computed once: a second call hands back the same object
without re-entering the kernel.  A computing call commits the package thread
budget, and ``compute_inbreeding`` is the 0.7.1 adapter onto that same array.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from test_inbreeding_kernel import ADR_0008_FIXTURES, _mz_frame

import pedigree_graph._core
from pedigree_graph import PedigreeGraph
from pedigree_graph._threads import _reset_thread_state, configure_threads

DEEP_FIXTURE = parity_fixtures("deep_inbred_60g")["deep_inbred_60g"]
DEEP_ENVELOPE = 2.0**-22

MZ_ONLY_LINK = next(case for case in ADR_0008_FIXTURES if case[0] == "mz_only_link")

MZ_CASES = pytest.mark.parametrize("case", ADR_0008_FIXTURES, ids=[case[0] for case in ADR_0008_FIXTURES])


def _graph(case) -> PedigreeGraph:
    _name, mother, father, twin, _expected = case
    return PedigreeGraph(_mz_frame(list(range(len(mother))), mother, father, twin))


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
    graph = PedigreeGraph(parity_columns(DEEP_FIXTURE))
    F = graph.inbreeding()
    assert np.abs(F - _self_kinship_identity(graph)).max() <= DEEP_ENVELOPE
    assert np.abs(F - _diagonal_identity(graph)).max() <= DEEP_ENVELOPE


def test_second_call_returns_the_memo_without_recomputing(monkeypatch):
    graph = _graph(MZ_ONLY_LINK)
    first = graph.inbreeding()
    monkeypatch.setattr(pedigree_graph._core, "_compute_F_meuwissen_luo", _forbid_kernel)
    assert graph.inbreeding() is first


def test_result_is_read_only():
    F = _graph(MZ_ONLY_LINK).inbreeding()
    with pytest.raises(ValueError, match="read-only"):
        F[0] = 0.5


def test_result_is_float64():
    assert _graph(MZ_ONLY_LINK).inbreeding().dtype == np.float64


class TestThreads:
    @pytest.fixture(autouse=True)
    def reset_thread_state(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        _reset_thread_state()
        yield
        _reset_thread_state()

    def test_computing_call_commits_the_budget(self):
        _graph(MZ_ONLY_LINK).inbreeding()
        with pytest.raises(RuntimeError):
            configure_threads(3)


def test_adapter_returns_the_canonical_array():
    graph = _graph(MZ_ONLY_LINK)
    F = graph.compute_inbreeding()
    assert F is graph.inbreeding()
    with pytest.raises(ValueError, match="read-only"):
        F[0] = 0.5
