"""Every kinship binding refuses a depth it cannot use with one error (issue #65).

``PedigreeGraph`` always passes the structural depth, which orders the parents
and puts co-twins level.  A depth handed to ``_native`` directly need not, and
every kernel checks it in one order: the parent ordering first, then co-twins.
"""

from __future__ import annotations

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._threads import thread_budget

# 1, 2 founders; 3, 4 their MZ co-twins; 5 a child of 1 alone.
_GRAPH = PedigreeGraph.from_frame(
    {
        "id": [1, 2, 3, 4, 5],
        "mother": [-1, -1, 1, 1, 1],
        "father": [-1, -1, 2, 2, -1],
        "sex": [0, 1, 0, 0, 1],
        "twin": [-1, -1, 4, 3, -1],
    }
)


def _row(person: int) -> int:
    return int(np.flatnonzero(_GRAPH.ids == person)[0])


def _every_kernel(depth: np.ndarray) -> dict[str, dict]:
    built, n, threads = _GRAPH._built, _GRAPH.n_individuals, thread_budget()
    first, second = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    diagonal = np.arange(n, dtype=np.int32)
    founder_column = np.full(n, -1, dtype=np.int64)
    founder_column[[_row(1), _row(2)]] = [0, 1]
    calls = {
        "pair_kinship": lambda: _native.pair_kinship(built, depth, first, second, threads=threads),
        "kinship_support_values": lambda: _native.kinship_support_values(
            built, depth, np.arange(n + 1, dtype=np.int64), diagonal, threads=threads
        ),
        "kinship_csc": lambda: _native.kinship_csc(built, depth),
        "approximate_kinship_csc": lambda: _native.approximate_kinship_csc(built, depth, 0.0),
        "inbreeding": lambda: _native.inbreeding(built, depth, threads=threads),
        "generation_kinship_sums": lambda: _native.generation_kinship_sums(
            built, depth, np.zeros(n), np.zeros(n, dtype=np.int32), 1
        ),
        "founder_contribution_means": lambda: _native.founder_contribution_means(
            built, depth, np.zeros(n, dtype=np.int32), 1, founder_column, 2
        ),
    }
    errors = {}
    for name, call in calls.items():
        with pytest.raises(PedigreeValidationError) as info:
            call()
        assert info.value.code == "value_out_of_range", name
        errors[name] = dict(info.value.fields)
    return errors


def _depth(raised: dict[int, int]) -> np.ndarray:
    """The structural depth with each person in *raised* moved to its value."""
    depth = np.array(_GRAPH.depth, dtype=np.int32)
    for person, value in raised.items():
        depth[_row(person)] = value
    return depth


def _assert_every_kernel_reports(depth: np.ndarray, want: dict) -> None:
    for name, fields in _every_kernel(depth).items():
        assert {key: fields[key] for key in want} == want, name


def test_every_kernel_reports_the_parent_ordering_before_the_co_twins():
    # Co-twin 4 raised above 3, and 5 not below its mother 1.
    _assert_every_kernel_reports(
        _depth({4: 2, 5: 0}),
        {"field": "depth", "position": _row(5), "value": 0, "minimum": 1, "maximum": 4},
    )


def test_every_kernel_reports_co_twins_at_two_depths():
    # The first co-twin row is reported, at its depth against its partner's.
    first, partner = sorted([_row(3), _row(4)])
    depth = _depth({4: 2})
    _assert_every_kernel_reports(
        depth,
        {"field": "depth", "position": first, "value": int(depth[first]), "minimum": int(depth[partner])},
    )
