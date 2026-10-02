"""A Rust panic inside a native call reaches Python as an exception, not an abort (ADR 0007).

The core forbids user-reachable panics, so the probe is a test-only entry
point compiled with the ``test-hooks`` feature (``pixi run build-dev``).  It
runs in a fresh interpreter, so a panic that did abort would fail the child
rather than kill the test runner.  ``PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS=1``
(set by ``pixi run test`` and ``test-all``) turns a missing hook into a
failure; only installed-wheel runs, which never carry the feature, skip.
"""

from __future__ import annotations

import os

import pytest
from _support import CHILD_PRELUDE, _run_child

from pedigree_graph import _native


def _require_hook(name: str) -> None:
    if not hasattr(_native, name):
        if os.environ.get("PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail("the extension was built without the test-hooks feature; run `pixi run build-dev`")
        pytest.skip("installed without the test-hooks feature")


def test_a_panic_raises_and_the_process_stays_usable():
    _require_hook("_panic_for_test")
    body = """
        try:
            _native._panic_for_test()
        except BaseException as e:
            print("raised", type(e).__name__)
        print("after", sum(c for c in graph.relationship_counts(max_degree=1).values() if c is not None))
    """
    lines = _run_child(CHILD_PRELUDE, body).splitlines()
    assert lines[0] == "raised PanicException"
    assert lines[1].startswith("after ")
    assert int(lines[1].split()[1]) > 0


def test_a_panic_in_a_watched_worker_raises_at_once_and_the_pool_stays_usable():
    """The job's completion guard wakes the watcher while it unwinds (ADR 0017).

    The tick is 30 s, so a watcher that only noticed the panic on its next
    tick would take about 30 s, not under 5.
    """
    _require_hook("_panic_in_watched_worker_for_test")
    body = """
        import time
        before = graph.relationship_counts(max_degree=2)
        start = time.perf_counter()
        try:
            _native._panic_in_watched_worker_for_test(threads=thread_budget(), tick=30.0)
        except BaseException as e:
            print("raised", type(e).__name__)
        print("seconds", time.perf_counter() - start)
        print("same", graph.relationship_counts(max_degree=2) == before)
    """
    lines = _run_child(CHILD_PRELUDE, body, timeout=60).splitlines()
    assert lines[0] == "raised PanicException"
    assert float(lines[1].split()[1]) < 5
    assert lines[2] == "same True"


def test_a_watched_call_returns_when_its_job_ends_not_on_the_next_tick():
    body = """
        import time
        start = time.perf_counter()
        _native.relationship_counts(graph._built, max_degree=5, threads=thread_budget(), tick=30.0)
        print(time.perf_counter() - start)
    """
    assert float(_run_child(CHILD_PRELUDE, body, timeout=60)) < 5
