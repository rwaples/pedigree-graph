"""The boundary contract of the raw ``_native.relationship_pairs`` binding.

The binding is the seam pair extraction crosses (ADR 0006 and 0007): the
Rust row-streaming engine classifies, orients, and assembles; Python keeps
the selector and the result type.  The public API calls this binding
directly, so its values are held against the oracle in
``test_relationship_pairs_execution``; these tests pin what only the raw
binding shows: owned int32 arrays, registry-ordered keys, both executions
element for element equal, and structured errors instead of aborts.
Anything that needs a different package pool runs in a fresh interpreter,
because the pool is built once per process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from _support import CHILD_PRELUDE, _run_child

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, _native
from pedigree_graph._threads import thread_budget


def _native_pairs(graph: PedigreeGraph, execution: str, **selector):
    """The binding's result for *graph* and *selector*, as the facade calls it."""
    from pedigree_graph._selection import RelationshipSelection

    selection = RelationshipSelection.parse(selector.get("max_degree"), selector.get("categories"))
    return _native.relationship_pairs(
        graph._built,
        max_degree=selection.top_degree or 0,
        requested=list(selection.ordered),
        threads=thread_budget(),
        execution=execution,
        view_rows=None,
    )


class TestBoundary:
    def test_arrays_are_owned_contiguous_int32_with_a_native_base(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        first, second = _native_pairs(graph, "speed", max_degree=2)["MO"]
        for array in (first, second):
            assert array.dtype == np.int32
            assert array.ndim == 1
            assert array.flags.c_contiguous
            assert not isinstance(array.base, np.ndarray)
        assert len(first) == len(second) > 0

    def test_the_two_executions_agree_element_for_element(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        speed = _native_pairs(graph, "speed", max_degree=5)
        memory = _native_pairs(graph, "memory", max_degree=5)
        for code in RELATIONSHIPS:
            np.testing.assert_array_equal(speed[code][0], memory[code][0])
            np.testing.assert_array_equal(speed[code][1], memory[code][1])

    def test_block_lengths_equal_the_counts(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        pairs = _native_pairs(graph, "memory", max_degree=5)
        counts = graph.relationship_counts(max_degree=5)
        assert tuple(pairs) == tuple(RELATIONSHIPS)
        assert {code: len(pairs[code][0]) for code in RELATIONSHIPS} == dict(counts)

    def test_malformed_arguments_raise_before_any_work(self, small_pedigree):
        graph = PedigreeGraph.from_frame(small_pedigree)
        call = {"max_degree": 5, "requested": ["FS"], "threads": thread_budget(), "execution": "speed"}
        with pytest.raises(ValueError, match="execution"):
            _native.relationship_pairs(graph._built, **{**call, "execution": "buffered"})
        with pytest.raises(ValueError, match="unknown relationship code"):
            _native.relationship_pairs(graph._built, **{**call, "requested": ["cousin"]})
        with pytest.raises(ValueError, match="view_rows"):
            _native.relationship_pairs(graph._built, **call, view_rows=np.array([0], dtype=np.int32))
        with pytest.raises(ValueError, match="threads"):
            _native.relationship_pairs(graph._built, **{**call, "threads": 0})
        with pytest.raises(PedigreeValidationError, match="max_degree"):
            _native.relationship_pairs(graph._built, **{**call, "max_degree": 6})
        with pytest.raises(RuntimeError, match="test seam is off"):
            _native.fail_next_allocation("heap")


def test_a_forked_child_gets_its_own_pool():
    """The pool is per process: a fork inherits its memory but none of its workers.

    Before the slot recorded its process, the child pushed work onto a queue
    with no live worker and blocked for ever.
    """
    script = textwrap.dedent(
        """
        import multiprocessing as mp
        import numpy as np
        from pedigree_graph import PedigreeGraph

        def build():
            return PedigreeGraph.from_arrays(
                ids=np.arange(6),
                mother_ids=np.array([-1, -1, 0, 0, 2, 2]),
                father_ids=np.array([-1, -1, 1, 1, 3, 3]),
            )

        def child(q):
            q.put(int(build().relationship_counts(max_degree=2)["FS"]))

        if __name__ == "__main__":
            parent = int(build().relationship_counts(max_degree=2)["FS"])
            ctx = mp.get_context("fork")
            q = ctx.Queue()
            p = ctx.Process(target=child, args=(q,))
            p.start()
            p.join(60)
            if p.is_alive():
                p.kill()
                raise SystemExit("the forked child blocked on the inherited pool")
            print(parent, q.get())
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["2", "2"]


def test_the_allocation_seam_is_off_unless_the_environment_unlocks_it():
    """A released wheel refuses the plant; a process that opts in still validates the name."""
    body = """
        try:
            _native.fail_next_allocation("heap")
        except ValueError as e:
            print("named:", e)
        _native.fail_next_allocation(None)
        print("cleared")
    """
    assert "named: unknown allocation family" in _run_child(CHILD_PRELUDE, body)

    locked = subprocess.run(
        [sys.executable, "-c", "from pedigree_graph import _native; _native.fail_next_allocation(None)"],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "PEDIGREE_GRAPH_ALLOW_TEST_SEAM"},
        check=False,
    )
    assert locked.returncode != 0
    assert "test seam is off" in locked.stderr


class TestProcessWidePool:
    def test_blocks_are_identical_under_every_thread_budget(self):
        body = """
            codes = list(graph.relationship_counts(max_degree=5))
            pairs = _native.relationship_pairs(
                graph._built, max_degree=5, requested=codes, threads=thread_budget(), execution="speed", view_rows=view
            )
            digest = hashlib.sha256()
            for code, (first, second) in pairs.items():
                digest.update(first.tobytes()); digest.update(second.tobytes())
            print(thread_budget(), digest.hexdigest())
        """
        one = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="1").split()
        four = _run_child(CHILD_PRELUDE, body, PEDIGREE_GRAPH_THREADS="4").split()
        assert one[0] == "1"
        assert four[0] == "4"
        assert one[1] == four[1]

    def test_the_pool_keeps_its_first_size(self):
        body = """
            print(_native.configure_pool(2), _native.configure_pool(2))
            try:
                _native.configure_pool(3)
            except RuntimeError as e:
                print("conflict:", e)
            _native.relationship_counts(graph._built, max_degree=2, threads=2)
            try:
                _native.relationship_counts(graph._built, max_degree=2, threads=5)
            except RuntimeError as e:
                print("conflict:", e)
        """
        out = _run_child(CHILD_PRELUDE, body)
        assert out.startswith("2 2")
        assert out.count("conflict: the thread pool is already configured with 2 threads") == 2


#: The canonical list, read from the core so it cannot drift from ``Family::ALL``.
FAMILIES = tuple(_native.allocation_families())

#: The families the counting path reserves.  The test asserts both
#: directions, so a family that starts or stops being reserved by counts
#: fails here instead of silently losing its case.
COUNT_FAMILIES = frozenset({"parent_edges", "csr", "sibling_index", "accumulator", "row_set"})

#: The families only the kinship walk, the kinship matrix DP and the
#: inbreeding, lineage and generation sweeps reserve; ``test_native_pair_kinship``,
#: ``test_native_kinship_matrix``, ``test_native_inbreeding``,
#: ``test_native_lineage`` and ``test_native_generations`` hold those, and here a
#: call that never reaches them has to succeed.
KINSHIP_FAMILIES = frozenset(
    {
        "kinship_memo",
        "kinship_stack",
        "kinship_output",
        "kinship_rows",
        "kinship_csc",
        "kinship_sums",
        "kinship_scratch",
        "inbreeding_walk",
        "lineage_sets",
        "lineage_output",
        "founder_means",
    }
)

#: The plant's size floor. Without one it fires on whichever reservation of
#: the family comes first, which for a collected iterator is its zero lower
#: bound, leaving ``requested_elements`` meaningless. One element is the
#: largest floor every family clears on a fixture this size: a task table
#: holds one entry per task range, so it is a handful of elements however
#: many rows there are.
SEAM_MIN_ELEMENTS = 1


def test_the_family_list_covers_the_counting_families():
    """``COUNT_FAMILIES`` and ``KINSHIP_FAMILIES`` name families the core still has."""
    assert set(FAMILIES) >= COUNT_FAMILIES | KINSHIP_FAMILIES
    assert len(FAMILIES) == len(set(FAMILIES))


@pytest.mark.parametrize("family", FAMILIES)
def test_a_refused_allocation_raises_a_resource_error(family):
    """Every allocation family surfaces as ``ResourceError("allocation_failed")`` in a fresh process.

    The plant carries a size floor, so it skips zero-sized reservations and
    ``requested_elements`` is a real count. A call whose
    path never reserves the family has to succeed: the plant is consumed
    only by a reservation, so success is what proves the family unreached.
    """
    body = f"""
        from pedigree_graph import ResourceError
        family = {family!r}
        floor = {SEAM_MIN_ELEMENTS}
        def pairs():
            return _native.relationship_pairs(
                graph._built, max_degree=5, requested=["FS", "2C"], threads=1, execution="speed", view_rows=view
            )
        def counts():
            return _native.relationship_counts(graph._built, max_degree=5, threads=1)
        expect_failure = {{
            "pairs": family not in {sorted(KINSHIP_FAMILIES)!r},
            "counts": family in {sorted(COUNT_FAMILIES)!r},
        }}
        for label, call in (("pairs", pairs), ("counts", counts)):
            _native.fail_next_allocation(family, floor)
            try:
                call()
            except ResourceError as e:
                print(label, e.code, e.fields["operation"], e.fields["dtype"], e.fields["requested_elements"] >= floor)
            else:
                print(label, "no error")
            _native.fail_next_allocation(None)
            print(label, "expected", expect_failure[label])
        call()
        print("recovered")
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines[-1] == "recovered"
    outcomes = {}
    for line in lines[:-1]:
        parts = line.split()
        if parts[1] == "expected":
            assert outcomes[parts[0]] == (parts[2] == "True"), line
            continue
        if parts[1] == "no":
            outcomes[parts[0]] = False
            continue
        label, code, operation, dtype, counted = parts
        outcomes[label] = True
        assert (code, operation, counted) == ("allocation_failed", family, "True"), line
        assert dtype in {"bool", "int32", "int64", "intp", "uint8", "uint64", "object"}
