"""Every relationship query commits the thread budget, and only after its arguments pass (issue #52).

The core answers a call with nothing to walk (an empty selection, a receiver
of fewer than two rows), so no host shortcut can skip the commit any more:
``relationship_counts`` with an empty selection used to return without
committing, unlike every other operation.  Rejected arguments still leave
the budget uncommitted, as #51 fixed for ``pair_kinship``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph, PedigreeValidationError
from pedigree_graph._threads import configure_threads


def _graph() -> PedigreeGraph:
    """0,1 founders; 2,3 full sibs; 4,5 their children's cousins-to-be."""
    return PedigreeGraph.from_frame(
        {
            "id": np.arange(6),
            "mother": [-1, -1, 0, 0, 2, 3],
            "father": [-1, -1, 1, 1, -1, -1],
        }
    )


def _call(name: str, receiver, n: int, **selector):
    """One relationship query on *receiver* with inputs sized to its *n* rows."""
    if name == "moments":
        return receiver.relationship_moments(**selector, values={"x": np.ones(n)})
    if name == "relatives":
        return receiver.relatives_per_person(**selector, thresholds={"t": (np.zeros(n), 0.0)})
    return getattr(receiver, f"relationship_{name}")(**selector)


OPERATIONS = ("pairs", "counts", "moments", "relatives")


@pytest.mark.usefixtures("fresh_thread_state")
class TestEveryCallCommits:
    @pytest.mark.parametrize("name", OPERATIONS)
    @pytest.mark.parametrize(
        ("rows", "selector"),
        [
            (None, {"categories": []}),
            ([0, 2, 4], {"categories": []}),
            ([3], {"max_degree": 2}),
            ([], {"max_degree": 2}),
        ],
        ids=["empty-graph", "empty-view", "one-row-view", "zero-row-view"],
    )
    def test_a_call_with_nothing_to_walk_commits_the_budget(self, name, rows, selector):
        graph = _graph()
        receiver = graph if rows is None else graph.view(rows=np.array(rows, dtype=np.int64))
        n = graph.n_individuals if rows is None else len(rows)
        _call(name, receiver, n, **selector)
        with pytest.raises(RuntimeError, match="committed thread budget"):
            configure_threads(3)


@pytest.mark.usefixtures("fresh_thread_state")
class TestRejectedArgumentsLeaveTheBudgetOpen:
    @pytest.mark.parametrize("name", OPERATIONS)
    def test_an_unknown_category_wins_over_a_bad_thread_variable(self, name, monkeypatch):
        monkeypatch.setenv("PEDIGREE_GRAPH_THREADS", "two")
        graph = _graph()
        with pytest.raises(PedigreeValidationError) as info:
            _call(name, graph, graph.n_individuals, categories=["cousin"])
        assert info.value.code == "unknown_relationship_category"
        configure_threads(3)

    @pytest.mark.parametrize("name", ["moments", "relatives"])
    @pytest.mark.parametrize("categories", [[], ["FS"]])
    def test_a_short_column_wins_over_a_bad_thread_variable(self, name, categories, monkeypatch):
        """With nothing to walk too: the host checks its columns before the core decides that."""
        monkeypatch.setenv("PEDIGREE_GRAPH_THREADS", "two")
        graph = _graph()
        with pytest.raises(PedigreeValidationError):
            _call(name, graph, graph.n_individuals - 1, categories=categories)
        configure_threads(3)
