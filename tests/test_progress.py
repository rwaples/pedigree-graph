"""Progress reporting and cancellation of the long relationship calls (ADR 0017, issue #37).

The binding wakes the calling thread every ``_progress.TICK_S`` seconds; the
default logger writes a line every ``_progress.LOG_INTERVAL_S``.  Tests that
need progress on a sub-second call shrink both; the rest keep the real ones.
Signals and the thread budget are per process, so the Ctrl-C and
thread-budget tests run in a child interpreter.
"""

from __future__ import annotations

import inspect
import logging
import re

import numpy as np
import pytest
from _support import _run_child

from pedigree_graph import PedigreeGraph, RelationshipProgress, _progress
from pedigree_graph._progress import format_progress


def random_graph(n: int, seed: int = 7) -> PedigreeGraph:
    """Overlapping generations: each row after the first 20 has parents among the 60 rows before it."""
    rng = np.random.default_rng(seed)
    mother = np.full(n, -1)
    father = np.full(n, -1)
    for i in range(20, n):
        lo = max(0, i - 60)
        mother[i], father[i] = rng.integers(lo, i), rng.integers(lo, i)
        if father[i] == mother[i]:
            father[i] = -1
    return PedigreeGraph.from_frame({"id": np.arange(n), "mother": mother, "father": father})


# The same builder for the child scripts.
RANDOM_GRAPH = "import numpy as np\nfrom pedigree_graph import PedigreeGraph\n" + inspect.getsource(random_graph)

# Each method walks this pedigree in about half a second at one thread, so a
# 0.01 s tick sees it walking many times.
SLOW_N = 20_000

# The five methods with the keyword; the graph argument may be a view.
METHODS = {
    "relationship_counts": lambda g, p: g.relationship_counts(max_degree=5, progress=p),
    "relationship_pairs": lambda g, p: g.relationship_pairs(max_degree=5, progress=p),
    "relationship_moments": lambda g, p: g.relationship_moments(max_degree=5, progress=p),
    "relatives_per_person": lambda g, p: g.relatives_per_person(max_degree=5, progress=p),
    "relationship_burden": lambda g, p: g.relationship_burden(progress=p),
}
LOGGERS = {
    "relationship_counts": "pedigree_graph._relationship_counts",
    "relationship_pairs": "pedigree_graph._relationship_pairs",
    "relationship_moments": "pedigree_graph._relationship_moments",
    "relatives_per_person": "pedigree_graph._relatives_per_person",
    "relationship_burden": "pedigree_graph._burden",
}
# Burden logs no start or total line (ADR 0017, D10).
WITH_START_AND_TOTAL = [m for m in METHODS if m != "relationship_burden"]
VIEW_METHODS = WITH_START_AND_TOTAL

PROGRESS_LINE = re.compile(r"^\w+: (preparing after |[\d,]+/[\d,]+ rows \(\d+%\) after |all [\d,]+ rows walked)")
WALKING_LINE = re.compile(r"^(\w+): [\d,]+/[\d,]+ rows \(\d+%\) after \d")
PHASES = ("preparing", "walking", "finishing")


@pytest.fixture(scope="module")
def slow_graph() -> PedigreeGraph:
    return random_graph(SLOW_N)


@pytest.fixture(scope="module")
def small_graph() -> PedigreeGraph:
    return random_graph(400)


@pytest.fixture
def fast_ticks(monkeypatch):
    monkeypatch.setattr(_progress, "TICK_S", 0.01)
    monkeypatch.setattr(_progress, "LOG_INTERVAL_S", 0.02)


def _messages(caplog, method: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGERS[method]]


def _progress_lines(caplog, method: str) -> list[str]:
    return [m for m in _messages(caplog, method) if PROGRESS_LINE.match(m)]


@pytest.mark.parametrize("method", METHODS)
def test_a_long_call_logs_walking_progress(method, slow_graph, fast_ticks, caplog):
    """Acceptance 1: rows done and total, and the elapsed time."""
    caplog.set_level(logging.INFO)
    METHODS[method](slow_graph, None)
    walking = [m for m in _progress_lines(caplog, method) if WALKING_LINE.match(m)]
    assert walking, _messages(caplog, method)
    assert all(WALKING_LINE.match(m)[1] == method for m in walking)


@pytest.mark.parametrize("method", METHODS)
def test_a_short_call_logs_no_progress(method, small_graph, caplog):
    """Acceptance 2, at the real interval: only the start and total lines, and none for burden."""
    caplog.set_level(logging.INFO)
    METHODS[method](small_graph, None)
    lines = _messages(caplog, method)
    if method in WITH_START_AND_TOTAL:
        assert len(lines) == 2, lines
        assert lines[0].startswith(f"{method}: max_degree=")
        assert lines[1].startswith(f"{method} total: ")
    else:
        assert lines == []


@pytest.mark.parametrize("method", METHODS)
def test_progress_false_keeps_the_start_and_total_lines_only(method, slow_graph, fast_ticks, caplog):
    caplog.set_level(logging.INFO)
    METHODS[method](slow_graph, False)
    lines = _messages(caplog, method)
    assert not any(PROGRESS_LINE.match(m) for m in lines), lines
    assert len(lines) == (2 if method in WITH_START_AND_TOTAL else 0), lines


def _check_reports(seen: list[RelationshipProgress]) -> None:
    assert seen
    ranks = [PHASES.index(p.phase) for p in seen]
    assert ranks == sorted(ranks)
    assert all((p.rows_total is None) == (p.phase == "preparing") for p in seen)
    done = [p.rows_done for p in seen]
    assert done == sorted(done)
    assert all(p.rows_total is None or p.rows_done <= p.rows_total for p in seen)
    elapsed = [p.elapsed for p in seen]
    assert elapsed == sorted(elapsed)
    assert any(p.phase == "walking" for p in seen)


@pytest.mark.parametrize("method", METHODS)
def test_a_callback_sees_a_consistent_call_and_nothing_is_logged(method, slow_graph, fast_ticks, caplog):
    caplog.set_level(logging.INFO)
    seen: list[RelationshipProgress] = []
    METHODS[method](slow_graph, seen.append)
    _check_reports(seen)
    assert _progress_lines(caplog, method) == []


@pytest.mark.parametrize("method", VIEW_METHODS)
@pytest.mark.parametrize("every", [2, 7], ids=["half", "seventh"])
def test_a_view_call_reports_its_own_walk(method, every, slow_graph, fast_ticks):
    """A view walks its ancestry-compact pedigree or the whole graph; either way the reports are consistent."""
    seen: list[RelationshipProgress] = []
    METHODS[method](slow_graph.view(rows=np.arange(0, SLOW_N, every)), seen.append)
    _check_reports(seen)
    assert all(p.rows_total is None or p.rows_total <= SLOW_N for p in seen)


def test_format_progress_has_one_form_per_phase():
    assert (
        format_progress("relationship_pairs", RelationshipProgress("preparing", 0, None, 45.9))
        == "relationship_pairs: preparing after 45s"
    )
    assert (
        format_progress("relationship_counts", RelationshipProgress("walking", 195_758, 783_029, 192.4))
        == "relationship_counts: 195,758/783,029 rows (25%) after 3m12s"
    )
    assert (
        format_progress("relationship_pairs", RelationshipProgress("finishing", 783_029, 783_029, 580.0))
        == "relationship_pairs: all 783,029 rows walked, assembling after 9m40s"
    )
    # Truncated, so a walk one row short never reads 100%.
    assert (
        format_progress("x", RelationshipProgress("walking", 999, 1000, 7509.0))
        == "x: 999/1,000 rows (99%) after 2h05m09s"
    )


@pytest.mark.parametrize("method", METHODS)
def test_a_raising_callback_cancels_the_call_and_the_graph_stays_usable(method, slow_graph, fast_ticks):
    reference = slow_graph.relationship_counts(max_degree=5, progress=False)

    def stop(_: RelationshipProgress) -> None:
        raise RuntimeError("stop here")

    with pytest.raises(RuntimeError, match="stop here"):
        METHODS[method](slow_graph, stop)
    assert slow_graph.relationship_counts(max_degree=5, progress=False) == reference


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [1, True, "log"])
def test_progress_must_be_none_false_or_a_callable(method, bad, small_graph):
    with pytest.raises(TypeError, match="progress must be None, False or a callable"):
        METHODS[method](small_graph, bad)


def test_ctrl_c_interrupts_a_long_call():
    """Acceptance for D4: KeyboardInterrupt well before the call would have finished."""
    body = """
        import _thread, threading, time
        from pedigree_graph import _progress
        _progress.TICK_S = 0.01
        graph = random_graph(40_000)
        start = time.perf_counter()
        graph.relationship_counts(max_degree=5, progress=False)
        full = time.perf_counter() - start
        threading.Timer(0.05, _thread.interrupt_main).start()
        start = time.perf_counter()
        try:
            graph.relationship_counts(max_degree=5, progress=False)
        except KeyboardInterrupt:
            print("raised", (time.perf_counter() - start) / full)
        else:
            print("finished")
    """
    word, *rest = _run_child(RANDOM_GRAPH, body, timeout=120).split()
    assert word == "raised"
    assert float(rest[0]) < 0.5


def test_results_do_not_depend_on_progress_at_any_thread_budget():
    """Acceptance 3: bit-identical under ``None``, ``False`` and a callable, at budgets 1 and 4."""
    body = """
        import hashlib
        from pedigree_graph import _progress
        from pedigree_graph._threads import thread_budget
        _progress.TICK_S = 0.001
        graph = random_graph(4_000)
        x = np.random.default_rng(3).normal(size=graph.n_individuals)
        calls = []

        def digest(*arrays):
            h = hashlib.sha256()
            for a in arrays:
                h.update(np.ascontiguousarray(a).tobytes())
            return h.hexdigest()[:16]

        def run(p):
            counts = graph.relationship_counts(max_degree=5, progress=p)
            pairs = graph.relationship_pairs(max_degree=5, progress=p)
            moments = graph.relationship_moments(max_degree=5, values={"x": x}, progress=p)
            relatives = graph.relatives_per_person(max_degree=5, thresholds={"t": (x, 0.0)}, progress=p)
            burden = graph.relationship_burden(progress=p)
            return [
                digest(np.array(list(counts.values()), dtype=np.int64)),
                digest(*(rows for block in pairs.values() for rows in block)),
                digest(moments.counts, moments.sum_first, moments.cross, moments.m2_first),
                digest(relatives.counts),
                digest(np.array(list(burden.category_counts.values()), dtype=np.uint64), burden.per_person, burden.same_depth_pairs),
            ]

        logged = run(None)
        same = logged == run(False) == run(calls.append)
        print(thread_budget(), len(calls) > 0, same, " ".join(logged))
    """
    one = _run_child(RANDOM_GRAPH, body, PEDIGREE_GRAPH_THREADS="1").split()
    four = _run_child(RANDOM_GRAPH, body, PEDIGREE_GRAPH_THREADS="4").split()
    assert one[:3] == ["1", "True", "True"]
    assert four[:3] == ["4", "True", "True"]
    assert one[3:] == four[3:]
