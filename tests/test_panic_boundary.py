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


def test_a_panic_raises_and_the_process_stays_usable():
    if not hasattr(_native, "_panic_for_test"):
        if os.environ.get("PEDIGREE_GRAPH_REQUIRE_TEST_HOOKS") == "1":
            pytest.fail("the extension was built without the test-hooks feature; run `pixi run build-dev`")
        pytest.skip("installed without the test-hooks feature")
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
