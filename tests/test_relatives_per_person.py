"""``relatives_per_person`` (issue #33): the per-person reducer, its boundary and its result.

The oracle is ``relationship_pairs`` plus ``np.add.at``, as the epimight and
pedsum kernels count today: a directional category credits its first
(junior) member only, a symmetric one both members, and a threshold column
credits a relative when ``relative[relative] <= threshold[person]``.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from _support import CHILD_PRELUDE, _run_child
from conftest import parity_columns, parity_fixtures

from pedigree_graph import (
    RELATIONSHIPS,
    PedigreeGraph,
    PedigreeValidationError,
    RelativesPerPerson,
    _native,
)
from pedigree_graph import _relatives_per_person as boundary

CODES = tuple(RELATIONSHIPS)
DIRECTIONAL = frozenset(code for code, cat in RELATIONSHIPS.items() if cat.first_role is not None)
FIXTURES = parity_fixtures("random_1k")


def _graph(name: str) -> PedigreeGraph:
    return PedigreeGraph.from_frame(parity_columns(FIXTURES[name]))


def _reordered_half(graph: PedigreeGraph, seed: int):
    n = graph.n_individuals
    return graph.view(rows=np.random.default_rng(seed).permutation(n)[: n // 2])


def _synthetic(n: int, seed: int) -> PedigreeGraph:
    """A random graph of *n* rows, each non-founder's parents drawn from the 60 rows before it."""
    rng = np.random.default_rng(seed)
    rows = np.arange(n)
    lo = np.maximum(0, rows - 60)
    mother = lo + (rng.random(n) * (rows - lo)).astype(np.int64)
    father = lo + (rng.random(n) * (rows - lo)).astype(np.int64)
    founder = rows < 20
    mother[founder] = -1
    father[founder | (father == mother)] = -1
    return PedigreeGraph.from_frame({"id": rows, "mother": mother, "father": father})


def _thresholds(n: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray | float]]:
    """Columns with NaN and ±inf on both sides, a scalar threshold, a float32 side and an integer one."""
    rng = np.random.default_rng(seed)
    onset = rng.uniform(0, 80, n)
    onset[rng.random(n) < 0.3] = np.nan
    onset[rng.random(n) < 0.05] = -np.inf
    cutoff = rng.uniform(0, 100, n)
    cutoff[rng.random(n) < 0.05] = np.nan
    cutoff[rng.random(n) < 0.05] = np.inf
    return {
        "aligned": (onset, cutoff),
        "scalar": (onset, 40.0),
        "f32": (onset.astype(np.float32), cutoff),
        "int": (rng.integers(0, 5, n), 2),
    }


def _oracle(receiver, codes: tuple[str, ...], thresholds: dict) -> np.ndarray:
    """int64 ``[rows, codes, 1 + K]`` counts from the receiver's pair blocks and ``np.add.at``."""
    columns = [
        (np.asarray(rel, dtype=np.float64), thr if np.ndim(thr) == 0 else np.asarray(thr, dtype=np.float64))
        for rel, thr in thresholds.values()
    ]
    out = np.zeros((receiver.n_individuals, len(codes), 1 + len(columns)), dtype=np.int64)
    if not codes:
        return out
    pairs = receiver.relationship_pairs(categories=codes)
    for c, code in enumerate(codes):
        first, second = pairs[code]
        sides = [(first, second)] if code in DIRECTIONAL else [(first, second), (second, first)]
        for person, relative in sides:
            np.add.at(out[:, c, 0], person, 1)
            for k, (rel, thr) in enumerate(columns):
                bound = thr if np.ndim(thr) == 0 else thr[person]
                np.add.at(out[:, c, 1 + k], person, (rel[relative] <= bound).astype(np.int64))
    return out


def _assert_parity(target, thresholds: dict, **context) -> None:
    got = target.relatives_per_person(max_degree=5, thresholds=thresholds)
    assert got.categories == CODES
    assert got.columns == ("relatives", *thresholds)
    assert got.counts.dtype == np.uint32
    np.testing.assert_array_equal(got.counts, _oracle(target, CODES, thresholds), err_msg=str(context))


class TestParity:
    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("receiver", ["graph", "view"])
    @pytest.mark.parametrize("with_thresholds", [False, True], ids=["totals", "thresholds"])
    def test_counts_equal_the_pair_oracle(self, name, receiver, with_thresholds):
        graph = _graph(name)
        target = graph if receiver == "graph" else _reordered_half(graph, 11)
        thresholds = _thresholds(target.n_individuals, 3) if with_thresholds else {}
        _assert_parity(target, thresholds, fixture=name, receiver=receiver)

    @pytest.mark.parametrize("receiver", [pytest.param("graph", marks=pytest.mark.slow), "view"])
    def test_random_30k_equals_the_pair_oracle(self, receiver):
        import pedigrees

        graph = PedigreeGraph.from_frame(
            parity_columns(pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"]))
        )
        target = graph if receiver == "graph" else _reordered_half(graph, 5)
        _assert_parity(target, _thresholds(target.n_individuals, 8), receiver=receiver)

    def test_a_selection_counts_only_its_categories(self):
        graph = _graph("random_1k")
        thresholds = _thresholds(graph.n_individuals, 4)
        got = graph.relatives_per_person(categories=["1C", "MO", "FS"], thresholds=thresholds)
        assert got.categories == ("MO", "FS", "1C")
        np.testing.assert_array_equal(got.counts, _oracle(graph, got.categories, thresholds))

    def test_compact_and_full_view_execution_are_identical(self, monkeypatch):
        graph = _synthetic(20_000, 3)
        view = graph.view(rows=np.random.default_rng(6).permutation(graph.n_individuals)[:2_000])
        assert boundary._should_compact_view(graph.n_individuals, len(view))
        thresholds = _thresholds(len(view), 9)
        results = []
        for compact in (False, True):
            monkeypatch.setattr(boundary, "_should_compact_view", lambda *_, flag=compact: flag)
            results.append(view.relatives_per_person(max_degree=5, thresholds=thresholds).counts)
        assert np.array_equal(results[0], results[1])
        assert results[0].sum() > 0


#: Epimight's relationship groups: pair codes and whether only the junior is credited.
EPIMIGHT_GROUPS = {
    "PO": (("MO", "FO"), True),
    "FS": (("FS", "MZ"), False),
    "HS": (("MHS", "PHS"), False),
    "mHS": (("MHS",), False),
    "pHS": (("PHS",), False),
    "Av": (("Av",), True),
    "1G": (("GP",), True),
    "1C": (("1C",), False),
}


def _count_affected_relatives(pair_list, affected, n, unidirectional=False, onset=None, cutoff=None):
    """fitACE_epimight's ``pair_extraction.count_affected_relatives``, inlined."""

    def hits(person, relative):
        hit = affected[relative]
        if cutoff is not None:
            hit = hit & (onset[relative] <= cutoff[person])
        return hit.astype(np.int64)

    counts = np.zeros(n, dtype=np.int64)
    for idx1, idx2 in pair_list:
        if len(idx1) == 0:
            continue
        np.add.at(counts, idx1, hits(idx1, idx2))
        if not unidirectional:
            np.add.at(counts, idx2, hits(idx2, idx1))
    return counts.astype(np.int32, copy=False)


def test_threshold_columns_reproduce_the_epimight_counts():
    graph = _graph("random_1k")
    n = graph.n_individuals
    rng = np.random.default_rng(21)
    affected = rng.random(n) < 0.4
    onset = rng.uniform(0, 80, n).astype(np.float32)
    onset[affected & (rng.random(n) < 0.15)] = np.nan
    cutoff = rng.uniform(0, 90, n)
    cutoff[rng.random(n) < 0.05] = np.nan
    cutoff[rng.random(n) < 0.05] = np.inf
    cutoff[rng.random(n) < 0.05] = -np.inf
    codes = sorted({code for group, _ in EPIMIGHT_GROUPS.values() for code in group})
    aligned = np.where(affected, onset, np.nan)
    assert aligned.dtype == np.float32
    got = graph.relatives_per_person(
        categories=codes,
        thresholds={"aligned": (aligned, cutoff), "unaligned": (np.where(affected, 0.0, np.nan), 0.0)},
    )
    pairs = graph.relationship_pairs(categories=codes)
    for group, (group_codes, unidirectional) in EPIMIGHT_GROUPS.items():
        blocks = [pairs[code] for code in group_codes]
        ones = np.ones(n, dtype=bool)
        want = {
            "relatives": _count_affected_relatives(blocks, ones, n, unidirectional),
            "aligned": _count_affected_relatives(blocks, affected, n, unidirectional, onset=onset, cutoff=cutoff),
            "unaligned": _count_affected_relatives(blocks, affected, n, unidirectional),
        }
        for column, expected in want.items():
            np.testing.assert_array_equal(got.sum(group_codes, column), expected, err_msg=f"{group} {column}")
        assert want["aligned"].sum() < want["unaligned"].sum() <= want["relatives"].sum(), group


THREAD_BODY = """
    import logging
    records = []
    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)
    log = logging.getLogger("pedigree_graph._relatives_per_person")
    log.addHandler(Keep())
    log.setLevel(logging.INFO)
    onset = rng.uniform(0, 80, n)
    onset[rng.random(n) < 0.3] = np.nan
    cutoff = rng.uniform(0, 100, n)
    thresholds = {"t": (onset, cutoff), "s": (onset, 30.0)}
    h = hashlib.sha256()
    h.update(graph.relatives_per_person(max_degree=5, thresholds=thresholds).counts.tobytes())
    lanes, lane_pairs = records[-1].args[2], records[-1].args[3]
    view = graph.view(rows=rng.permutation(n)[: n // 2])
    half = {k: (r[view.graph_rows], t if np.ndim(t) == 0 else t[view.graph_rows]) for k, (r, t) in thresholds.items()}
    h.update(view.relatives_per_person(max_degree=5, thresholds=half).counts.tobytes())
    print(thread_budget(), lanes, sum(1 for p in lane_pairs if p > 0), h.hexdigest())
"""

#: A graph of more than three 2048-row task ranges, so several lanes get work.
THREAD_PRELUDE = """
    import hashlib
    import numpy as np
    from pedigree_graph import PedigreeGraph
    from pedigree_graph._threads import thread_budget
    n = 3 * 2048 + 300
    rng = np.random.default_rng(7)
    mother = np.full(n, -1); father = np.full(n, -1)
    for i in range(20, n):
        lo = max(0, i - 60)
        mother[i], father[i] = rng.integers(lo, i), rng.integers(lo, i)
        if father[i] == mother[i]:
            father[i] = -1
    graph = PedigreeGraph.from_frame({"id": np.arange(n), "mother": mother, "father": father})
"""


def test_counts_are_bit_identical_under_every_thread_budget():
    one = _run_child(THREAD_PRELUDE, THREAD_BODY, PEDIGREE_GRAPH_THREADS="1").split()
    four = _run_child(THREAD_PRELUDE, THREAD_BODY, PEDIGREE_GRAPH_THREADS="4").split()
    assert one[:3] == ["1", "1", "1"]
    assert four[:2] == ["4", "4"]
    assert int(four[2]) >= 2, four
    assert one[3] == four[3]


def test_a_refused_count_allocation_raises_and_the_next_call_recovers():
    body = """
        from pedigree_graph import ResourceError
        floor = n
        x = rng.normal(size=n)
        def run():
            return graph.relatives_per_person(max_degree=2, thresholds={"t": (x, 0.5)}).counts
        clean = run().copy()
        _native.fail_next_allocation("relative_counts", floor)
        try:
            run()
        except ResourceError as e:
            print(e.code, e.fields["operation"], e.fields["dtype"], e.fields["requested_elements"] >= floor)
        else:
            print("no error")
        _native.fail_next_allocation(None)
        print(np.array_equal(run(), clean), int(clean.sum()) > 0)
    """
    lines = _run_child(CHILD_PRELUDE, body).strip().splitlines()
    assert lines == ["allocation_failed relative_counts uint32 True", "True True"]


class TestHandover:
    def test_counts_are_a_read_only_reshape_of_the_native_array(self):
        graph = _graph("random_1k")
        got = graph.relatives_per_person(max_degree=3, thresholds={"t": (np.zeros(graph.n_individuals), 1.0)})
        flat = got.counts.base
        assert isinstance(flat, np.ndarray)
        assert flat.ndim == 1
        assert flat.size == got.counts.size
        assert not isinstance(flat.base, np.ndarray)
        assert np.shares_memory(got.counts, flat)
        assert not got.counts.flags.writeable
        assert not flat.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            got.counts[0, 0, 0] = 1

    def test_get_is_a_read_only_view(self):
        graph = _graph("random_1k")
        got = graph.relatives_per_person(max_degree=2, thresholds={"t": (np.zeros(graph.n_individuals), 1.0)})
        column = got.get("MHS", "t")
        assert np.shares_memory(column, got.counts)
        assert not column.flags.writeable
        np.testing.assert_array_equal(column, got.counts[:, got.categories.index("MHS"), 1])


class TestSum:
    def test_sum_folds_codes_into_int64(self):
        graph = _graph("random_1k")
        got = graph.relatives_per_person(max_degree=2)
        want = got.counts[:, got.categories.index("MO"), 0].astype(np.int64) + got.get("FO")
        total = got.sum(iter(["MO", "FO"]))
        assert total.dtype == np.int64
        np.testing.assert_array_equal(total, want)
        np.testing.assert_array_equal(got.sum([]), np.zeros(graph.n_individuals, dtype=np.int64))

    @pytest.mark.parametrize(
        ("codes", "column", "error", "match"),
        [
            (["MO", "FO", "MO"], "relatives", ValueError, r"repeats \['MO'\]"),
            (["MO", "1C"], "relatives", ValueError, "'1C' was not requested"),
            (["MO"], "t", ValueError, "no column 't'"),
            ("MO", "relatives", TypeError, "not a single str"),
        ],
    )
    def test_sum_refuses_bad_codes_and_columns(self, codes, column, error, match):
        got = _graph("random_1k").relatives_per_person(max_degree=2)
        with pytest.raises(error, match=match):
            got.sum(codes, column)

    def test_get_refuses_an_unrequested_code_and_an_unknown_column(self):
        got = _graph("random_1k").relatives_per_person(categories=["FS"])
        with pytest.raises(ValueError, match="'MO' was not requested"):
            got.get("MO")
        with pytest.raises(ValueError, match="no column 'x'"):
            got.get("FS", "x")

    def test_sum_allocates_only_its_result(self):
        graph = _synthetic(20_000, 5)
        got = graph.relatives_per_person(max_degree=2)
        codes = got.categories
        assert len(codes) >= 6
        rows = len(got.counts)
        tracemalloc.start()
        try:
            total = got.sum(codes)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # A gathering fold would hold 4 bytes per row per code on top of the result.
        assert peak <= 8 * rows + 80_000, peak
        assert 4 * rows * len(codes) > 80_000
        np.testing.assert_array_equal(total, got.counts[:, :, 0].sum(axis=1, dtype=np.int64))


class TestBorrowing:
    def _peak(self, graph, thresholds) -> int:
        graph.relatives_per_person(max_degree=1, thresholds=thresholds)
        tracemalloc.start()
        try:
            graph.relatives_per_person(max_degree=1, thresholds=thresholds)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_contiguous_float64_columns_and_scalar_thresholds_are_not_copied(self):
        graph = _synthetic(20_000, 7)
        n = graph.n_individuals
        rng = np.random.default_rng(1)
        onset, cutoff = rng.uniform(0, 80, n), rng.uniform(0, 100, n)
        peak = self._peak(graph, {"a": (onset, cutoff), "b": (onset, cutoff), "s": (onset, 40.0)})
        assert peak < 2 * n, peak

    def test_a_converted_input_shared_by_two_columns_is_converted_once(self):
        graph = _synthetic(20_000, 7)
        n = graph.n_individuals
        rng = np.random.default_rng(2)
        onset, cutoff = rng.uniform(0, 80, n), rng.uniform(0, 100, n).astype(np.float32)
        peak = self._peak(graph, {"a": (onset, cutoff), "b": (onset, cutoff)})
        assert 8 * n <= peak < 16 * n, peak

    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.int64, np.uint8])
    def test_other_real_dtypes_are_converted(self, dtype):
        graph = _graph("random_1k")
        n = graph.n_individuals
        values = np.random.default_rng(3).integers(0, 5, n).astype(dtype)
        strided = np.repeat(values, 2)[::2]
        got = graph.relatives_per_person(max_degree=2, thresholds={"t": (values, strided), "s": (values, dtype(2))})
        want = graph.relatives_per_person(
            max_degree=2, thresholds={"t": (values.astype(np.float64), values.astype(np.float64)), "s": (values, 2.0)}
        )
        np.testing.assert_array_equal(got.counts, want.counts)


@pytest.fixture(scope="module")
def graph():
    return _graph("random_1k")


class TestBoundaryErrors:
    def _call(self, graph, **kwargs):
        kwargs.setdefault("max_degree", 2)
        return graph.relatives_per_person(**kwargs)

    def test_length_mismatch(self, graph):
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            self._call(graph, thresholds={"t": (np.zeros(n - 1), 0.0)})
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "thresholds['t'].relative"
        with pytest.raises(PedigreeValidationError) as info:
            self._call(graph, thresholds={"t": (np.zeros(n), np.zeros(n + 1))})
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "thresholds['t'].threshold"

    def test_two_dimensional_input(self, graph):
        n = graph.n_individuals
        with pytest.raises(PedigreeValidationError) as info:
            self._call(graph, thresholds={"t": (np.zeros((n, 1)), 0.0)})
        assert info.value.code == "invalid_shape"

    @pytest.mark.parametrize("threshold", [True, np.True_, "bool-array"])
    def test_bool_is_refused(self, graph, threshold):
        n = graph.n_individuals
        if isinstance(threshold, str):
            threshold = np.ones(n, dtype=bool)
        with pytest.raises(TypeError, match="bool"):
            self._call(graph, thresholds={"t": (np.zeros(n), threshold)})
        with pytest.raises(TypeError, match="bool"):
            self._call(graph, thresholds={"t": (np.zeros(n, dtype=bool), 0.0)})

    def test_non_numeric_arrays_are_refused(self, graph):
        n = graph.n_individuals
        with pytest.raises(TypeError, match="real numeric"):
            self._call(graph, thresholds={"t": (np.zeros(n, dtype=complex), 0.0)})

    @pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8, reason="longdouble is float64 here")
    def test_floats_wider_than_float64_are_refused(self, graph):
        n = graph.n_individuals
        wide = np.full(n, 1 + np.finfo(np.longdouble).eps, dtype=np.longdouble)
        with pytest.raises(TypeError, match="wider than float64"):
            self._call(graph, thresholds={"t": (wide, 0.0)})
        with pytest.raises(TypeError, match="wider than float64"):
            self._call(graph, thresholds={"t": (np.zeros(n), wide[0])})

    @pytest.mark.parametrize(
        ("relative", "threshold", "value"),
        [
            ("big", 0.0, 2**53 + 1),
            ("big_negative", 0.0, -(2**53) - 1),
            ("uint64", 0.0, 2**63),
            ("zeros", 2**53 + 1, 2**53 + 1),
            ("zeros", np.int64(-(2**60)), -(2**60)),
        ],
    )
    def test_integers_beyond_2_53_are_refused(self, graph, relative, threshold, value):
        n = graph.n_individuals
        arrays = {
            "big": np.full(n, 2**53 + 1, dtype=np.int64),
            "big_negative": np.full(n, -(2**53) - 1, dtype=np.int64),
            "uint64": np.full(n, 2**63, dtype=np.uint64),
            "zeros": np.zeros(n, dtype=np.int64),
        }
        with pytest.raises(PedigreeValidationError) as info:
            self._call(graph, thresholds={"t": (arrays[relative], threshold)})
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["value"] == value

    def test_2_53_itself_is_exact(self, graph):
        n = graph.n_individuals
        got = self._call(graph, thresholds={"t": (np.full(n, 2**53, dtype=np.int64), -(2**53))})
        assert not got.get("MO", "t").any()

    def test_at_most_32_columns(self, graph):
        n = graph.n_individuals
        x = np.zeros(n)
        assert len(self._call(graph, thresholds={f"c{i}": (x, 0.0) for i in range(32)}).columns) == 33
        with pytest.raises(ValueError, match="at most 32"):
            self._call(graph, thresholds={f"c{i}": (x, 0.0) for i in range(33)})

    @pytest.mark.parametrize(
        ("name", "error"),
        [("", TypeError), (3, TypeError), (None, TypeError), ("relatives", ValueError)],
    )
    def test_column_names(self, graph, name, error):
        x = np.zeros(graph.n_individuals)
        with pytest.raises(error):
            self._call(graph, thresholds={name: (x, 0.0)})

    def test_an_entry_must_be_a_pair(self, graph):
        x = np.zeros(graph.n_individuals)
        with pytest.raises(TypeError, match="tuple"):
            self._call(graph, thresholds={"t": [x, 0.0]})
        with pytest.raises(TypeError, match="tuple"):
            self._call(graph, thresholds={"t": (x,)})

    def test_selectors(self, graph):
        with pytest.raises(PedigreeValidationError) as info:
            graph.relatives_per_person(categories=["XX"])
        assert info.value.code == "unknown_relationship_category"
        with pytest.raises(TypeError, match="exactly one"):
            graph.relatives_per_person(max_degree=2, categories=["FS"])
        with pytest.raises(TypeError, match="exactly one"):
            graph.relatives_per_person()
        with pytest.raises(TypeError, match="single str"):
            graph.relatives_per_person(categories="FS")


class TestEmptyReceivers:
    def test_fewer_than_two_rows_skip_the_engine(self, monkeypatch):
        def refuse(*_, **__):
            raise AssertionError("the engine ran")

        graph = _graph("random_1k")
        monkeypatch.setattr(_native, "relatives_per_person", refuse)
        for rows in ([], [7]):
            view = graph.view(rows=np.array(rows, dtype=np.int64))
            x = np.zeros(len(rows))
            got = view.relatives_per_person(max_degree=2, thresholds={"t": (x, x), "s": (x, 1.0)})
            assert isinstance(got, RelativesPerPerson)
            assert got.counts.shape == (len(rows), 8, 3)
            assert got.counts.dtype == np.uint32
            assert not got.counts.any()
            assert not got.counts.flags.writeable
        with pytest.raises(PedigreeValidationError, match="length"):
            graph.view(rows=np.array([7])).relatives_per_person(max_degree=2, thresholds={"t": (np.zeros(2), 0.0)})

    def test_an_empty_selection_skips_the_engine(self, monkeypatch):
        def refuse(*_, **__):
            raise AssertionError("the engine ran")

        graph = _graph("random_1k")
        monkeypatch.setattr(_native, "relatives_per_person", refuse)
        got = graph.relatives_per_person(categories=[], thresholds={"t": (np.zeros(graph.n_individuals), 0.0)})
        assert got.categories == ()
        assert got.counts.shape == (graph.n_individuals, 0, 2)


def test_repr_names_rows_totals_and_columns():
    graph = _graph("nuclear_full_sibs")
    got = graph.relatives_per_person(categories=["MO", "FS"], thresholds={"t": (np.zeros(graph.n_individuals), 0.0)})
    assert repr(got).startswith(f"RelativesPerPerson(rows={graph.n_individuals}; MO=")
    assert repr(got).endswith("columns=('relatives', 't'))")
