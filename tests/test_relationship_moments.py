"""``relationship_moments`` (ADR 0013): the engine sink, its numeric contract, and the result algebra.

The oracle is ``relationship_pairs`` plus Python integers: every pair block
of the same receiver, the package's own quantizer, and unbounded-int sums
whose exact rational value (with ``Fraction``) is rounded once to float64,
so counts, sums and centered moments are compared bit for bit.
"""

from __future__ import annotations

import math
import tracemalloc
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _support import _run_child
from conftest import parity_columns, parity_fixtures

from pedigree_graph import RELATIONSHIPS, PedigreeGraph, PedigreeValidationError, ResourceError
from pedigree_graph import _relationship_moments as boundary
from pedigree_graph._threads import thread_budget
from pedigree_graph.moments import CONVERSION_CHUNK, MomentAxis, host_bytes

if TYPE_CHECKING:
    from pedigree_graph.moments import RelationshipMoments

ARRAYS = (
    "counts",
    "sum_first",
    "sum_second",
    "sumsq_first",
    "sumsq_second",
    "cross",
    "m2_first",
    "m2_second",
    "comoment",
)
EXACT = ("q_sum_first", "q_sum_second", "q_sumsq_first", "q_sumsq_second", "q_cross")
SYMMETRIC_CODES = frozenset(code for code, cat in RELATIONSHIPS.items() if cat.first_role is None)
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "crates" / "core" / "tests" / "fixtures"


def _sim_graph(name: str) -> PedigreeGraph:
    """A graph from an engine-input TSV whose original ids are its rows."""
    table = np.loadtxt(FIXTURE_DIR / f"{name}.tsv", dtype=np.int64, skiprows=1)
    n = len(table)
    return PedigreeGraph.from_frame(
        {"id": np.arange(n), "mother": table[:, 3], "father": table[:, 4], "twin": table[:, 2]}
    )


def _graph(name: str) -> PedigreeGraph:
    if name.startswith("sim_"):
        return _sim_graph(name)
    return PedigreeGraph.from_frame(parity_columns(parity_fixtures(name)[name]))


def _inputs(n: int, seed: int, *, per_role: bool) -> dict:
    rng = np.random.default_rng(seed)
    first = {"g": rng.integers(0, 3, n), "s": rng.integers(0, 2, n)}
    second = {"s": first["s"], "a": rng.integers(0, 2, n)} if per_role else None
    values = {"x": rng.normal(size=n), "y": rng.normal(size=n) * 1e-3 + 7.0}
    same = {"hh": rng.integers(-1, 4, n)}
    products = [("first.x", "second.x"), ("first.y", "second.y"), ("first.x", "first.y"), ("second.y", "second.x")]
    return {"first": first, "second": second, "values": values, "same": same, "products": products}


def _oracle(receiver, inputs: dict, codes: tuple[str, ...], symmetric: str) -> dict[str, np.ndarray]:
    """The result arrays from the receiver's pair blocks and exact Python integers."""
    n = receiver.n_individuals
    lf, n1, _ = boundary._pack("first", inputs["first"], n)
    if inputs["second"] is None:
        ls, n2 = lf, n1
    else:
        ls, n2, _ = boundary._pack("second", inputs["second"], n)
    columns = {name: np.asarray(v, dtype=np.float64) for name, v in inputs["values"].items()}
    names = tuple(columns)
    q, exps = boundary._quantize(columns, n)
    _, products = boundary._parse_products(inputs["products"], names)
    keys = np.empty((n, len(inputs["same"])), dtype=np.int64)
    for j, v in enumerate(inputs["same"].values()):
        keys[:, j] = np.asarray(v, dtype=np.int64)
    k, p, s = len(names), len(products), keys.shape[1]
    cells = n1 * n2 << s
    stride = 1 + 4 * k + p
    acc = [[[0] * stride for _ in range(cells)] for _ in codes]
    pairs = receiver.relationship_pairs(categories=list(codes))
    q_list = q.tolist()
    keys_list = keys.tolist()

    def add(ci: int, a: int, b: int) -> None:
        cell = int(lf[a]) * n2 + int(ls[b])
        for j in range(s):
            cell = (cell << 1) | int(keys_list[a][j] >= 0 and keys_list[a][j] == keys_list[b][j])
        row = acc[ci][cell]
        row[0] += 1
        va, vb = q_list[a], q_list[b]
        for c in range(k):
            row[1 + c] += va[c]
            row[1 + k + c] += vb[c]
            row[1 + 2 * k + c] += va[c] * va[c]
            row[1 + 3 * k + c] += vb[c] * vb[c]
        for i, (sa, ca, sb, cb) in enumerate(products):
            row[1 + 4 * k + i] += (va, vb)[sa][ca] * (va, vb)[sb][cb]

    for ci, code in enumerate(codes):
        block = pairs[code]
        for a, b in zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True):
            add(ci, a, b)
            if symmetric == "both" and code in SYMMETRIC_CODES:
                add(ci, b, a)

    def scaled(value: int, e: int, n_pairs: int = 1) -> float:
        return float(Fraction(value, n_pairs) / Fraction(2) ** e) if n_pairs else 0.0

    out = {name: [] for name in (*ARRAYS, *EXACT)}
    for ci in range(len(codes)):
        for row in acc[ci]:
            n_pairs = row[0]
            sa, sb = row[1 : 1 + k], row[1 + k : 1 + 2 * k]
            qa, qb = row[1 + 2 * k : 1 + 3 * k], row[1 + 3 * k : 1 + 4 * k]
            cross = row[1 + 4 * k :]
            out["counts"].append(n_pairs)
            for name, ints in zip(EXACT, (sa, sb, qa, qb, cross), strict=True):
                out[name].append(list(ints))
            out["sum_first"].append([scaled(v, int(exps[c])) for c, v in enumerate(sa)])
            out["sum_second"].append([scaled(v, int(exps[c])) for c, v in enumerate(sb)])
            out["sumsq_first"].append([scaled(v, 2 * int(exps[c])) for c, v in enumerate(qa)])
            out["sumsq_second"].append([scaled(v, 2 * int(exps[c])) for c, v in enumerate(qb)])
            out["cross"].append(
                [scaled(v, int(exps[ca] + exps[cb])) for v, (_, ca, _, cb) in zip(cross, products, strict=True)]
            )
            out["m2_first"].append(
                [scaled(n_pairs * qa[c] - sa[c] * sa[c], 2 * int(exps[c]), n_pairs) for c in range(k)]
            )
            out["m2_second"].append(
                [scaled(n_pairs * qb[c] - sb[c] * sb[c], 2 * int(exps[c]), n_pairs) for c in range(k)]
            )
            sums = (sa, sb)
            out["comoment"].append(
                [
                    scaled(n_pairs * v - sums[sx][cx] * sums[sy][cy], int(exps[cx] + exps[cy]), n_pairs)
                    for v, (sx, cx, sy, cy) in zip(cross, products, strict=True)
                ]
            )
    shape = (len(codes), n1 * n2 << s)
    return {
        "counts": np.array(out["counts"], dtype=np.int64).reshape(shape),
        **{name: np.array(out[name], dtype=np.float64).reshape(*shape, -1) for name in ARRAYS[1:]},
        **{name: np.array(out[name], dtype=object).reshape(*shape, -1) for name in EXACT},
    }


def _flat(result: RelationshipMoments) -> dict[str, np.ndarray]:
    n_cat = result.shape[0]
    return {
        "counts": result.counts.reshape(n_cat, -1),
        **{
            name: getattr(result, name).reshape(n_cat, -1, getattr(result, name).shape[-1])
            for name in (*ARRAYS[1:], *EXACT)
        },
    }


def _assert_same(got: dict[str, np.ndarray], want: dict[str, np.ndarray], **context) -> None:
    for name in (*ARRAYS, *EXACT):
        np.testing.assert_array_equal(got[name], want[name], err_msg=f"{name} {context}", strict=True)


def _reordered_half(graph: PedigreeGraph, seed: int):
    n = graph.n_individuals
    return graph.view(rows=np.random.default_rng(seed).permutation(n)[: n // 2])


PARITY_FIXTURES = ("random_1k", "sim_d2_n3000", "sim_d3_n3000", "sim_d6_n3000")


class TestParity:
    @pytest.mark.parametrize("name", PARITY_FIXTURES)
    @pytest.mark.parametrize("receiver", ["graph", "view"])
    @pytest.mark.parametrize("per_role", [False, True], ids=["shared", "per_role"])
    @pytest.mark.parametrize("symmetric", ["canonical", "both"])
    def test_moments_equal_the_pair_oracle(self, name, receiver, per_role, symmetric):
        graph = _graph(name)
        target = graph if receiver == "graph" else _reordered_half(graph, 11)
        inputs = _inputs(target.n_individuals, 3, per_role=per_role)
        codes = tuple(RELATIONSHIPS)
        got = target.relationship_moments(max_degree=5, symmetric=symmetric, **inputs)
        assert got.categories == codes
        _assert_same(_flat(got), _oracle(target, inputs, codes, symmetric), fixture=name, receiver=receiver)

    @pytest.mark.slow
    @pytest.mark.parametrize("receiver", ["graph", "view"])
    def test_random_30k_equals_the_pair_oracle(self, receiver):
        import pedigrees

        graph = PedigreeGraph.from_frame(
            parity_columns(pedigrees.build_random("random_30k", pedigrees.LARGE_FIXTURES["random_30k"]))
        )
        target = graph if receiver == "graph" else _reordered_half(graph, 5)
        inputs = _inputs(target.n_individuals, 8, per_role=True)
        codes = tuple(RELATIONSHIPS)
        got = target.relationship_moments(max_degree=5, **inputs)
        _assert_same(_flat(got), _oracle(target, inputs, codes, "canonical"), receiver=receiver)

    def test_compact_and_full_view_execution_are_bit_identical(self, monkeypatch):
        graph = _graph("random_1k")
        view = graph.view(rows=np.random.default_rng(2).permutation(graph.n_individuals)[:100])
        inputs = _inputs(len(view), 4, per_role=True)
        results = []
        for compact in (False, True):
            monkeypatch.setattr(boundary, "_should_compact_view", lambda *_, flag=compact: flag)
            results.append(_flat(view.relationship_moments(max_degree=5, **inputs)))
        _assert_same(results[1], results[0])

    def test_a_view_of_fewer_than_two_rows_has_no_pairs(self):
        graph = _graph("random_1k")
        for rows in ([], [7]):
            view = graph.view(rows=np.array(rows, dtype=np.int64))
            got = view.relationship_moments(
                max_degree=2, first={"s": np.ones(len(rows), dtype=np.int64)}, values={"x": np.full(len(rows), 2.0)}
            )
            # No rows means no factor levels; a lone row keeps its one level.
            assert got.shape == (8, len(rows), len(rows))
            assert got.counts.sum() == 0
            assert got.lanes == 0


class TestQuantization:
    def test_exponents_follow_the_binary_exponent_rule(self):
        for magnitude, want in [(0.0, 0), (1.0, 43), (1024.0, 33), (3.0, 41), (0.75, 43), (1e-3, 52)]:
            got = boundary._exponent(np.array([magnitude, -magnitude / 2]))
            assert got == want, magnitude
            if magnitude:
                assert Fraction(magnitude) * Fraction(2) ** got <= Fraction(2) ** 43
                assert Fraction(magnitude) * Fraction(2) ** (got + 1) > Fraction(2) ** 43

    def test_halfway_values_round_to_even_at_both_signs(self):
        magnitude = 16.0
        e = boundary._exponent(np.array([magnitude]))
        step = Fraction(1, 2**e)
        halfway = [float(step * (k + Fraction(1, 2))) for k in (2, 3, -3, -4)]
        column = np.array([magnitude, *halfway])
        q, exps = boundary._quantize({"x": column}, len(column))
        assert exps[0] == e
        assert q[1:, 0].tolist() == [2, 4, -2, -4]
        assert q[0, 0] == 2**43

    def test_quantization_error_is_below_the_bound(self):
        rng = np.random.default_rng(1)
        column = rng.normal(size=1000) * 37.0
        q, exps = boundary._quantize({"x": column}, len(column))
        bound = Fraction(max(abs(Fraction(x)) for x in column.tolist())) / 2**43
        for x, qx in zip(column.tolist(), q[:, 0].tolist(), strict=True):
            assert abs(Fraction(qx, 2 ** int(exps[0])) - Fraction(x)) <= Fraction(1, 2 ** (int(exps[0]) + 1)) < bound

    def test_an_all_zero_column_has_exponent_zero_and_exact_zero_moments(self):
        graph = _graph("random_1k")
        got = graph.relationship_moments(categories=["FS", "MO"], values={"z": np.zeros(graph.n_individuals)})
        assert boundary._exponent(np.zeros(3)) == 0
        assert got.counts.sum() > 0
        for name in ARRAYS[1:]:
            assert np.all(getattr(got, name) == 0.0), name

    @pytest.mark.parametrize("magnitude", [5e-324, 2.2250738585072014e-308, 1e300, 1.7976931348623157e308])
    def test_extreme_finite_inputs_quantize_without_overflow(self, magnitude):
        rng = np.random.default_rng(3)
        column = rng.uniform(-1, 1, 50) * magnitude
        column[0] = magnitude
        q, exps = boundary._quantize({"x": column}, len(column))
        assert np.all(np.isfinite(column))
        assert 2**42 < np.max(np.abs(q)) <= 2**43
        if magnitude < 1e308:
            back = np.ldexp(q.astype(np.float64), -exps)
            assert np.all(np.isfinite(back))
            assert np.all(np.abs(back[:, 0] - column) <= np.ldexp(1.0, -int(exps[0]) - 1))
        else:
            # The largest finite float64 quantizes to exactly 2^43, which no
            # longer scales back: the output overflow is the ValueError below.
            assert q[0, 0] == 2**43

    def test_an_output_that_overflows_float64_names_the_column(self):
        graph = _graph("random_1k")
        huge = np.full(graph.n_individuals, 1.5e308)
        got = graph.relationship_moments(categories=["FS"], values={"big": huge, "fine": np.ones(graph.n_individuals)})
        # The integers are exact; the overflow is in the float view.
        assert got.q_sum_first[0, 0] > got.counts[0] * 2**42
        with pytest.raises(ValueError, match=r"sum_first of 'big'"):
            _ = got.sum_first
        with pytest.raises(ValueError, match=r"sumsq_first of 'big'"):
            _ = got.sumsq_first
        # A subnormal maximum scales far past 2^1023 and back without loss.
        tiny = np.full(graph.n_individuals, 5e-324)
        got = graph.relationship_moments(categories=["FS"], values={"tiny": tiny})
        assert np.all(got.sum_first == got.counts[..., np.newaxis] * 5e-324)

    def test_non_finite_values_are_rejected(self):
        graph = _graph("random_1k")
        bad = np.ones(graph.n_individuals)
        bad[5] = np.nan
        with pytest.raises(ValueError, match=r"values\['x'\] is not finite at position 5"):
            graph.relationship_moments(categories=["FS"], values={"x": bad})
        bad[5] = -np.inf
        with pytest.raises(ValueError, match="not finite"):
            graph.relationship_moments(categories=["FS"], values={"x": bad})


class TestCenteredMoments:
    def test_a_constant_column_has_an_exactly_zero_second_moment(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        got = graph.relationship_moments(
            max_degree=3, values={"c": np.full(n, 12345.678), "x": np.random.default_rng(6).normal(size=n)}
        )
        assert np.all(got.m2_first[..., 0] == 0.0)
        assert np.all(got.m2_second[..., 0] == 0.0)
        assert np.all(got.comoment[..., 0] == 0.0)
        assert np.any(got.m2_first[..., 1] > 0)
        assert np.all(got.counts[..., np.newaxis] * got.sumsq_first >= got.sum_first**2 * 0.999)

    def test_no_cancellation_when_the_mean_dwarfs_the_spread(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(9)
        spread = rng.normal(size=n)
        got = graph.relationship_moments(categories=["FS"], values={"x": spread + 1e3, "y": spread})
        first = graph.relationship_pairs(categories=["FS"])["FS"].first_rows
        want = float(np.sum((spread[first] - spread[first].mean()) ** 2))
        # The shifted column is quantized at 2^-33, so its moment carries that
        # resolution and nothing more: no cancellation term at 1e6 x SD^2.
        assert math.isclose(got.m2_first[0, 1], want, rel_tol=1e-12)
        assert math.isclose(got.m2_first[0, 0], want, rel_tol=1e-8)

    def test_reversing_a_view_swaps_the_member_moments_and_keeps_r_bit_identical(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(12)
        x, y = rng.normal(size=n), rng.normal(size=n) + 3.0
        forward = graph.view(rows=np.arange(n))
        backward = graph.view(rows=np.arange(n)[::-1])
        kwargs = {"categories": ["FS", "MHS", "PHS", "1C"], "products": [("first.x", "second.x")]}
        a = forward.relationship_moments(values={"x": x, "y": y}, **kwargs)
        b = backward.relationship_moments(values={"x": x[::-1], "y": y[::-1]}, **kwargs)
        np.testing.assert_array_equal(a.counts, b.counts)
        np.testing.assert_array_equal(a.m2_first, b.m2_second)
        np.testing.assert_array_equal(a.m2_second, b.m2_first)
        np.testing.assert_array_equal(a.comoment, b.comoment)
        # The pair set is the same and the diagonal product is symmetric, so
        # the correlation cannot depend on which member came first.
        np.testing.assert_array_equal(a.pearson("first.x", "second.x"), b.pearson("first.x", "second.x"))


def _lane_and_output_bytes(result: RelationshipMoments) -> tuple[int, int]:
    n_cat, cells = result.shape[0], int(np.prod(result.shape[1:]))
    accumulators = n_cat * cells * (1 + 4 * len(result.columns) + len(result.products))
    return accumulators * 16, host_bytes(accumulators)


#: A graph of more than three 2048-row task ranges, so several lanes get work.
MOMENTS_PRELUDE = """
    import hashlib
    import numpy as np
    from pedigree_graph import PedigreeGraph, _native
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

MOMENTS_BODY = """
    values = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    first = {"g": rng.integers(0, 4, n), "s": rng.integers(0, 2, n)}
    same = {"hh": rng.integers(-1, 30, n)}
    def run(**kw):
        return graph.relationship_moments(max_degree=5, first=first, values=values, same=same, **kw)
    def digest(m):
        h = hashlib.sha256()
        for name in ("counts", "sum_first", "sum_second", "sumsq_first", "sumsq_second", "cross", "m2_first", "m2_second", "comoment"):
            h.update(getattr(m, name).tobytes())
        h.update(repr([[int(v) for v in getattr(m, name).ravel()] for name in ("q_sum_first", "q_cross")]).encode())
        return h.hexdigest()
"""


class TestBudgetAndLanes:
    def test_the_host_conversion_stays_within_its_budget_term(self):
        accumulators = 3 * CONVERSION_CHUNK + 5
        rng = np.random.default_rng(31)
        hi = rng.integers(-(1 << 62), 1 << 62, accumulators, dtype=np.int64)
        lo = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, accumulators, dtype=np.int64)
        tracemalloc.start()
        try:
            exact = boundary._split_halves(hi, lo)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The halves themselves (16 bytes each) predate the trace.
        assert peak <= host_bytes(accumulators) - 16 * accumulators
        want = [(int(high) << 64) + int(low) for high, low in zip(hi[:3], lo[:3].view(np.uint64), strict=True)]
        assert [int(v) for v in exact[:3]] == want

    def test_moments_are_bit_identical_under_every_thread_budget(self):
        body = (
            MOMENTS_BODY
            + """
    m = run()
    print(thread_budget(), m.lanes, sum(1 for p in m.lane_pairs if p > 0), sum(m.lane_pairs) == int(m.counts.sum()), digest(m))
"""
        )
        one = _run_child(MOMENTS_PRELUDE, body, PEDIGREE_GRAPH_THREADS="1").split()
        four = _run_child(MOMENTS_PRELUDE, body, PEDIGREE_GRAPH_THREADS="4").split()
        assert one[:4] == ["1", "1", "1", "True"]
        assert four[:2] == ["4", "4"]
        assert int(four[2]) >= 2, four
        assert four[3] == "True"
        assert one[4] == four[4]

    def test_a_smaller_budget_cuts_the_lanes_and_keeps_the_result(self):
        body = (
            MOMENTS_BODY
            + """
    full = run()
    n_cat, cells = full.shape[0], int(np.prod(full.shape[1:]))
    k, p = len(full.columns), len(full.products)
    accumulators = n_cat * cells * (1 + 4 * k + p)
    lane = accumulators * 16
    out = host_bytes(accumulators)
    print(full.lanes, full.estimated_peak_bytes == 4 * lane + out, sum(1 for p in full.lane_pairs if p > 0) >= 2)
    for w in (1, 2, 3):
        m = run(memory_budget_bytes=w * lane + out + lane - 1)
        print(m.lanes, m.estimated_peak_bytes == w * lane + out, digest(m) == digest(full), len(m.lane_pairs) == w)
"""
        )
        lines = _run_child(
            MOMENTS_PRELUDE, "    from pedigree_graph.moments import host_bytes", body, PEDIGREE_GRAPH_THREADS="4"
        ).splitlines()
        assert lines[0] == "4 True True"
        assert lines[1:] == ["1 True True True", "2 True True True", "3 True True True"]

    def test_a_budget_below_one_lane_is_refused_before_any_accumulator_exists(self):
        body = (
            MOMENTS_BODY
            + """
    from pedigree_graph import ResourceError
    full = run()
    n_cat, cells = full.shape[0], int(np.prod(full.shape[1:]))
    k, p = len(full.columns), len(full.products)
    accumulators = n_cat * cells * (1 + 4 * k + p)
    lane = accumulators * 16
    out = host_bytes(accumulators)
    _native.fail_next_allocation("moment_lanes")
    try:
        run(memory_budget_bytes=lane + out - 1)
    except ResourceError as e:
        print(e.code, e.fields["operation"], e.fields["estimated_bytes"] == lane + out, e.fields["budget_bytes"] == lane + out - 1)
    # The plant is still armed: no lane was reserved by the refused call.
    try:
        run()
    except ResourceError as e:
        print(e.code, e.fields["operation"], e.fields["dtype"])
    _native.fail_next_allocation("moment_output")
    try:
        run()
    except ResourceError as e:
        print(e.code, e.fields["operation"], e.fields["dtype"])
    _native.fail_next_allocation(None)
    print(run().lanes)
"""
        )
        lines = _run_child(
            MOMENTS_PRELUDE, "    from pedigree_graph.moments import host_bytes", body, PEDIGREE_GRAPH_THREADS="2"
        ).splitlines()
        assert lines == [
            "memory_budget_exceeded relationship_moments True True",
            "allocation_failed moment_lanes int128",
            "allocation_failed moment_output int64",
            "2",
        ]

    def test_the_lane_count_never_exceeds_the_thread_budget(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        got = graph.relationship_moments(max_degree=2, values={"x": np.ones(graph.n_individuals)})
        assert 1 <= got.lanes <= thread_budget()
        assert len(got.lane_pairs) == got.lanes
        assert sum(got.lane_pairs) == got.counts.sum()
        lane, out = _lane_and_output_bytes(got)
        assert got.estimated_peak_bytes == got.lanes * lane + out
        # The degenerate paths plan the same sizes and honour the same budget.
        empty = graph.view(rows=np.array([3])).relationship_moments(max_degree=2, values={"x": np.ones(1)})
        assert empty.lanes == 0
        assert empty.estimated_peak_bytes == lane + out
        with pytest.raises(ResourceError) as info:
            graph.view(rows=np.array([3])).relationship_moments(
                max_degree=2, values={"x": np.ones(1)}, memory_budget_bytes=lane + out - 1
            )
        assert info.value.code == "memory_budget_exceeded"
        # No categories means no accumulators, which any budget fits.
        assert graph.relationship_moments(categories=[], values={"x": np.ones(n)}, memory_budget_bytes=0).shape == (0,)

    def test_each_same_key_doubles_the_cells_in_the_estimate(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(4)
        keys = {f"k{i}": rng.integers(0, 3, n) for i in range(3)}
        peaks = []
        for count in range(4):
            got = graph.relationship_moments(
                categories=["FS"], first={"s": rng.integers(0, 2, n)}, same=dict(list(keys.items())[:count])
            )
            assert got.shape == (1, 2, 2, *([2] * count))
            lane, out = _lane_and_output_bytes(got)
            assert got.estimated_peak_bytes == lane + out
            peaks.append(got.estimated_peak_bytes)
        assert peaks[1:] == [2 * peaks[0], 4 * peaks[0], 8 * peaks[0]]

    def test_labels_past_the_int32_range_are_refused_before_the_engine(self):
        n = 50_000
        graph = PedigreeGraph.from_frame({"id": np.arange(n), "mother": np.full(n, -1), "father": np.full(n, -1)})
        with pytest.raises(PedigreeValidationError) as info:
            graph.relationship_moments(categories=["FS"], first={"a": np.arange(n), "b": np.arange(n)})
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["field"] == "first label"
        assert info.value.fields["maximum"] == 2**31 - 1
        # Below the int32 range the engine sizes the cells and refuses the budget.
        with pytest.raises(ResourceError) as info:
            graph.relationship_moments(categories=["FS"], first={"a": np.arange(n)}, second={"a": np.arange(n)})
        assert info.value.code == "memory_budget_exceeded"
        assert info.value.fields["estimated_bytes"] > info.value.fields["budget_bytes"] == 1 << 30
        with pytest.raises(ValueError, match="at most 16 same keys"):
            graph.relationship_moments(
                categories=["FS"], first={"a": np.arange(n)}, same={f"k{i}": np.arange(n) for i in range(17)}
            )
        with pytest.raises(ResourceError) as info:
            graph.relationship_moments(
                categories=["FS"], first={"a": np.arange(n)}, same={f"k{i}": np.arange(n) for i in range(16)}
            )
        accumulators = 50_000**2 * 2**16
        assert info.value.fields["estimated_bytes"] == accumulators * 16 + host_bytes(accumulators)


class TestSurface:
    def test_factor_packing_round_trips_noncontiguous_and_negative_levels(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(7)
        g = rng.choice(np.array([-5, 3, 100]), n)
        s = rng.integers(0, 2, n).astype(bool)
        got = graph.relationship_moments(categories=["FS", "MO"], first={"g": g, "s": s}, second={"s": s})
        assert [a.name for a in got.axes] == ["category", "first_g", "first_s", "second_s"]
        np.testing.assert_array_equal(got.axis("first_g").levels, [-5, 3, 100])
        np.testing.assert_array_equal(got.axis("second_s").levels, [0, 1])
        assert got.categories == ("MO", "FS")
        pairs = graph.relationship_pairs(categories=["FS", "MO"])
        for ci, code in enumerate(got.categories):
            first, second = pairs[code]
            for gi, level in enumerate([-5, 3, 100]):
                for si in (0, 1):
                    for sj in (0, 1):
                        want = np.count_nonzero((g[first] == level) & (s[first] == si) & (s[second] == sj))
                        assert got.counts[ci, gi, si, sj] == want
                        assert got.select(category=code, first_g=level, first_s=si, second_s=sj).counts.item() == want
        with pytest.raises(ValueError, match="not a level"):
            got.select(first_g=4)
        with pytest.raises(ValueError, match="more than once"):
            got.select(first_g=[3, 3])
        assert got.select(first_g=range(3, 4)).shape == (2, 1, 2, 2)
        assert got.select(first_g={-5, 100}).shape == (2, 2, 2, 2)
        assert got.select(first_g=np.array(100)).axis("first_g").levels.tolist() == [100]
        assert got.select(category="FS").shape == (1, 3, 2, 2)
        zero = got.select(first_g=[]).sum("first_g")
        assert zero.shape == (2, 2, 2)
        assert zero.counts.sum() == 0
        assert zero.q_sum_first.shape == (2, 2, 2, 0)

    def test_select_sum_and_merge_equal_a_direct_engine_call(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(8)
        g, s, hh = rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(-1, 5, n)
        values = {"x": rng.normal(size=n), "y": rng.normal(size=n) + 2}
        products = [("first.x", "second.x"), ("first.x", "second.y"), ("first.y", "first.x")]
        full = graph.relationship_moments(
            max_degree=3, first={"g": g, "s": s}, second={"s": s}, values=values, products=products, same={"hh": hh}
        )
        oracle_inputs = {"first": {"g": g}, "second": {"s": s}, "values": values, "same": {}, "products": products}
        direct = graph.relationship_moments(
            max_degree=3, first={"g": g}, second={"s": s}, values=values, products=products
        )
        folded = full.sum("first_s", "same_hh")
        assert [a.name for a in folded.axes] == [a.name for a in direct.axes]
        _assert_same(_flat(folded), _flat(direct))
        np.testing.assert_array_equal(folded.pearson("first.x", "second.y"), direct.pearson("first.x", "second.y"))
        _assert_same(_flat(direct), _oracle(graph, oracle_inputs, tuple(direct.categories), "canonical"))
        # Merging the two halves of the key axis is the same fold, exactly.
        halves = full.select(same_hh=0).sum("same_hh").merge(full.select(same_hh=1).sum("same_hh"))
        _assert_same(_flat(halves), _flat(full.sum("same_hh")))
        po = full.select(category=["MO", "FO"]).sum("category")
        assert po.shape == full.shape[1:]
        assert po.categories == ()
        assert po.counts.sum() == full.select(category=["MO", "FO"]).counts.sum()
        doubled = full.merge(full)
        np.testing.assert_array_equal(doubled.counts, 2 * full.counts)
        np.testing.assert_array_equal(doubled.q_cross, 2 * full.q_cross)
        np.testing.assert_array_equal(doubled.m2_first, 2 * full.m2_first)
        np.testing.assert_array_equal(doubled.comoment, 2 * full.comoment)
        np.testing.assert_array_equal(doubled.pearson("first.x", "second.x"), full.pearson("first.x", "second.x"))
        with pytest.raises(ValueError, match="same axes"):
            full.merge(direct)
        with pytest.raises(ValueError, match="symmetric rule"):
            full.merge(
                graph.relationship_moments(
                    max_degree=3,
                    first={"g": g, "s": s},
                    second={"s": s},
                    values=values,
                    products=products,
                    same={"hh": hh},
                    symmetric="both",
                )
            )

    def test_a_constant_non_dyadic_column_stays_exactly_constant_through_the_algebra(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(21)
        fam = rng.integers(0, 3, n)
        got = graph.relationship_moments(
            max_degree=2,
            first={"fam": fam},
            values={"c": np.full(n, 0.3), "x": rng.normal(size=n)},
            products=[("first.c", "second.x"), ("first.c", "second.c"), ("first.x", "second.x")],
        )
        for result in (got, got.sum("first_fam"), got.sum("first_fam", "second_fam"), got.merge(got).sum("category")):
            assert np.all(result.m2_first[..., 0] == 0.0)
            assert np.all(result.m2_second[..., 0] == 0.0)
            assert np.all(result.comoment[..., 0] == 0.0)
            assert np.all(np.isnan(result.pearson("first.c", "second.x")))
            assert np.all(np.isnan(result.pearson("first.c", "second.c")))
            r = result.pearson("first.x", "second.x")
            assert np.all(np.isnan(r) | (np.abs(r) <= 1.0))
            mean = result.mean("first.c")
            assert np.all(np.isnan(mean) | (np.abs(mean - 0.3) <= 0.3 * 2**-43))

    def test_merge_aligns_different_exponents_exactly(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(22)
        x = rng.normal(size=n)
        big = x.copy()
        big[0] = 1e6
        halves = graph.view(rows=np.arange(n // 2)), graph.view(rows=np.arange(n // 2, n))
        a = halves[0].relationship_moments(categories=["FS", "MO"], values={"x": big[: n // 2]})
        b = halves[1].relationship_moments(categories=["FS", "MO"], values={"x": x[n // 2 :]})
        assert a.exponents[0] < b.exponents[0]
        merged = a.merge(b)
        assert merged.exponents[0] == b.exponents[0]
        shift = int(b.exponents[0] - a.exponents[0])
        np.testing.assert_array_equal(merged.q_sum_first, a.q_sum_first * 2**shift + b.q_sum_first)
        np.testing.assert_array_equal(merged.q_sumsq_first, a.q_sumsq_first * 4**shift + b.q_sumsq_first)
        np.testing.assert_array_equal(merged.q_cross, a.q_cross * 4**shift + b.q_cross)
        np.testing.assert_array_equal(merged.counts, a.counts + b.counts)
        np.testing.assert_array_equal(b.merge(a).q_sumsq_first, merged.q_sumsq_first)
        assert merged.m2_first.shape == a.m2_first.shape

    def test_merging_extreme_scales_converts_without_a_false_overflow(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        x = np.random.default_rng(23).normal(size=n)
        halves = graph.view(rows=np.arange(n // 2)), graph.view(rows=np.arange(n // 2, n))
        a = halves[0].relationship_moments(categories=["FS"], values={"x": x[: n // 2] * 1e100})
        b = halves[1].relationship_moments(categories=["FS"], values={"x": x[n // 2 :] * 1e-100})
        merged = a.merge(b)
        e = int(merged.exponents[0])
        n_pairs = int(merged.counts[0])
        s, q = int(merged.q_sum_first[0, 0]), int(merged.q_sumsq_first[0, 0])
        assert q.bit_length() > 1100, "the aligned integers must exceed float64 for this test to bite"
        assert merged.sumsq_first[0, 0] == float(Fraction(q) / Fraction(2) ** (2 * e))
        assert merged.m2_first[0, 0] == float(Fraction(n_pairs * q - s * s, n_pairs) / Fraction(2) ** (2 * e))
        assert np.isfinite(merged.comoment).all()
        r = merged.pearson("first.x", "second.x")
        assert np.all(np.abs(r) <= 1.0)

    def test_folding_every_axis_leaves_a_zero_dimensional_cell(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        x = np.random.default_rng(24).normal(size=n)
        per_category = graph.relationship_moments(categories=["MHS", "PHS"], values={"x": x})
        pooled = per_category.sum("category")
        assert pooled.shape == () == pooled.counts.shape
        assert int(pooled.counts) == int(per_category.counts.sum())
        assert pooled.q_sum_first.shape == pooled.m2_first.shape == (1,)
        assert pooled.pearson("first.x", "second.x").shape == ()
        assert pooled.mean("first.x").shape == ()

    def test_count_only_and_product_free_results_have_empty_float_views(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        counted = graph.relationship_moments(max_degree=1, first={"s": np.arange(n) % 2})
        for name in ARRAYS[1:]:
            assert getattr(counted, name).shape == (*counted.shape, 0), name
        plain = graph.relationship_moments(max_degree=1, values={"x": np.arange(n, dtype=float)}, products=[])
        assert plain.cross.shape == plain.comoment.shape == (*plain.shape, 0)
        assert plain.m2_first.shape == (*plain.shape, 1)

    def test_axis_levels_are_read_only_and_copied(self):
        got = _graph("random_1k").relationship_moments(max_degree=1)
        with pytest.raises(ValueError, match="read-only"):
            got.axis("category").levels[0] = "MZ"
        levels = np.array([3, 5])
        MomentAxis("first_g", levels)
        levels[0] = 4
        assert levels.flags.writeable

    def test_both_orientations_equal_canonical_plus_its_transpose_for_symmetric_codes(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(10)
        factors = {"s": rng.integers(0, 2, n), "g": rng.integers(0, 3, n)}
        values = {"x": rng.normal(size=n)}
        canonical = graph.relationship_moments(max_degree=3, first=factors, values=values)
        both = graph.relationship_moments(max_degree=3, first=factors, values=values, symmetric="both")
        assert both.symmetric == "both"
        transpose = (0, 3, 4, 1, 2)
        for ci, code in enumerate(canonical.categories):
            counts_t = np.transpose(canonical.counts, transpose)[ci]
            sum_first_t = np.transpose(canonical.sum_second, (*transpose, 5))[ci]
            if code in SYMMETRIC_CODES:
                np.testing.assert_array_equal(both.counts[ci], canonical.counts[ci] + counts_t, err_msg=code)
                np.testing.assert_array_equal(both.sum_first[ci], canonical.sum_first[ci] + sum_first_t, err_msg=code)
            else:
                for name in ARRAYS:
                    np.testing.assert_array_equal(getattr(both, name)[ci], getattr(canonical, name)[ci], err_msg=code)

    def test_products_on_one_side_match_numpy(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=n), rng.normal(size=n)
        got = graph.relationship_moments(
            categories=["FS", "Av"],
            values={"x": x, "y": y},
            products=[("first.x", "first.y"), ("second.y", "second.y"), ("first.x", "second.y")],
        )
        pairs = graph.relationship_pairs(categories=["FS", "Av"])
        for ci, code in enumerate(got.categories):
            first, second = pairs[code]
            for pi, want in enumerate(
                [np.sum(x[first] * y[first]), np.sum(y[second] ** 2), np.sum(x[first] * y[second])]
            ):
                assert math.isclose(got.cross[ci, pi], want, rel_tol=1e-12, abs_tol=1e-12), (code, pi)
            r = np.corrcoef(x[first], y[second])[0, 1]
            assert math.isclose(got.pearson("first.x", "second.y")[ci], r, rel_tol=1e-10)
            assert math.isclose(got.pearson("second.y", "first.x")[ci], r, rel_tol=1e-10)
            assert math.isclose(got.mean("first.x")[ci], x[first].mean(), rel_tol=1e-12)
        with pytest.raises(ValueError, match="not a requested product"):
            got.pearson("first.x", "second.x")

    def test_equality_keys_match_a_pair_list_oracle_and_unknowns_never_match(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(14)
        hh = rng.integers(-2, 6, n)
        got = graph.relationship_moments(categories=["FS", "MHS", "PHS"], same={"hh": hh})
        assert got.axis("same_hh").levels.tolist() == [0, 1]
        pairs = graph.relationship_pairs(categories=["FS", "MHS", "PHS"])
        for ci, code in enumerate(got.categories):
            first, second = pairs[code]
            same = (hh[first] >= 0) & (hh[first] == hh[second])
            assert got.counts[ci, 1] == np.count_nonzero(same)
            assert got.counts[ci, 0] == np.count_nonzero(~same)
        both_unknown = np.full(n, -1)
        got = graph.relationship_moments(categories=["FS"], same={"hh": both_unknown})
        assert got.counts[0, 1] == 0
        assert got.counts[0, 0] == len(pairs["FS"])

    def test_table_folds_every_other_axis(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        rng = np.random.default_rng(15)
        a, b, g = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 3, n)
        got = graph.relationship_moments(categories=["FS", "MO"], first={"g": g, "a": a}, second={"b": b})
        table = got.table("first_a", "second_b")
        assert table.shape == (2, 2, 2)
        np.testing.assert_array_equal(got.sum("category").table("first_a", "second_b"), table.sum(axis=0))
        with pytest.raises(ValueError, match="two different axes"):
            got.table("first_a", "first_a")
        pairs = graph.relationship_pairs(categories=["FS", "MO"])
        for ci, code in enumerate(got.categories):
            first, second = pairs[code]
            for i in (0, 1):
                for j in (0, 1):
                    assert table[ci, i, j] == np.count_nonzero((a[first] == i) & (b[second] == j))
        assert got.cell_levels()["first_g"].shape == got.shape

    def test_default_products_are_the_diagonal_and_second_defaults_to_first(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        got = graph.relationship_moments(
            categories=["FS"], first={"s": np.zeros(n, dtype=int)}, values={"x": np.ones(n), "y": np.ones(n)}
        )
        assert got.products == (("first.x", "second.x"), ("first.y", "second.y"))
        assert [a.name for a in got.axes] == ["category", "first_s", "second_s"]

    def test_arguments_are_validated_at_the_boundary(self):
        graph = _graph("random_1k")
        n = graph.n_individuals
        ones = np.ones(n)
        with pytest.raises(ValueError, match="symmetric must be one of"):
            graph.relationship_moments(categories=["FS"], symmetric="either")
        for budget in (-1, 2**64, True, 1.5):
            with pytest.raises(ValueError, match="memory_budget_bytes"):
                graph.relationship_moments(categories=["FS"], memory_budget_bytes=budget)
        with pytest.raises(TypeError, match=r"\(str, str\) tuple"):
            graph.relationship_moments(categories=["FS"], values={"x": ones}, products=["first.x"])
        with pytest.raises(TypeError, match=r"\(str, str\) tuple"):
            graph.relationship_moments(categories=["FS"], values={"x": ones}, products=[("first.x", 1)])
        with pytest.raises(PedigreeValidationError) as info:
            graph.relationship_moments(categories=["FS"], same={"hh": np.full(n, 2**63, dtype=np.uint64)})
        assert info.value.code == "value_out_of_range"
        assert info.value.fields["value"] == 2**63
        assert (
            graph.relationship_moments(categories=["FS"], same={"hh": np.full(n, 2**63 - 1, dtype=np.uint64)})
            .counts[0, 1]
            .item()
            > 0
        )
        with pytest.raises(ValueError, match="at most 32 value columns"):
            graph.relationship_moments(categories=["FS"], values={f"c{i}": ones for i in range(33)})
        with pytest.raises(ValueError, match="product operand"):
            graph.relationship_moments(categories=["FS"], values={"x": ones}, products=[("first.x", "third.x")])
        with pytest.raises(TypeError, match="integer or boolean"):
            graph.relationship_moments(categories=["FS"], first={"f": ones * 0.5})
        with pytest.raises(TypeError, match="exactly one of"):
            graph.relationship_moments()
        with pytest.raises(PedigreeValidationError) as info:
            graph.relationship_moments(categories=["FS"], first={"f": np.zeros(n - 1, dtype=int)})
        assert info.value.code == "length_mismatch"
        assert info.value.fields["field"] == "first['f']"
        with pytest.raises(PedigreeValidationError) as info:
            graph.relationship_moments(categories=["FS"], values={"x": np.zeros((n, 1))})
        assert info.value.code == "invalid_shape"
        with pytest.raises(TypeError, match="non-empty str"):
            graph.relationship_moments(categories=["FS"], values={"": ones})
        view = graph.view(rows=np.arange(10))
        with pytest.raises(PedigreeValidationError) as info:
            view.relationship_moments(categories=["FS"], values={"x": ones})
        assert info.value.fields == {"field": "values['x']", "expected_length": 10, "actual_length": n}

    def test_arrays_are_read_only_and_the_result_repr_names_its_axes(self):
        graph = _graph("random_1k")
        got = graph.relationship_moments(
            categories=["FS"],
            first={"s": np.zeros(graph.n_individuals, dtype=int)},
            values={"x": np.ones(graph.n_individuals)},
        )
        for name in (*ARRAYS, *EXACT, "exponents"):
            with pytest.raises(ValueError, match="read-only"):
                getattr(got, name)[...] = 0
        assert repr(got).startswith("RelationshipMoments(category=1, first_s=1, second_s=1; pairs=")
        assert got.count() is got.counts
