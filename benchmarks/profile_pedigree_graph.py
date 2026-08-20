#!/usr/bin/env python
"""Profiling harness for the pedigree_graph consumed surface.

Builds a representative pedigree with simACE's ``run_simulation`` (the same
DataFrame real consumers feed in), then profiles each public ``PedigreeGraph``
operation to attribute wall-time and peak memory to one of three layers:

  * **scipy / SuiteSparse sparse matmuls** — ``extract_pairs`` is dominated by
    ``A2 @ A2.T``, ``A3 @ A3.T``, ... (compiled C; can't fuse a threshold into
    the product, so the full intermediate ``nnz`` is materialised first).
  * **numba ``@njit`` kernels** — ``compute_inbreeding`` (Meuwissen-Luo) and
    ``per_gen_mean_kinship`` / kinship-matrix DP. LLVM-compiled; ~C++ parity.
  * **python orchestration** — the rest.

Why this answers the "rewrite in C++?" question:

  * time concentrated in scipy matmuls **with high intermediate nnz** -> the
    lever is a *fused thresholded sparse product* (emit an entry only when the
    shared-ancestor count >= 2), achievable in numba or C++. Memory win, not a
    language win.
  * time concentrated in the numba kernels -> already at the compiled ceiling;
    a C++ port buys ~nothing (see docs/adr/0001).

Two memory signals are reported per op:
  * ``rss_delta_mb``      — peak RSS over a background sampler thread. Captures
                            scipy's C-side buffers (the real matmul footprint).
  * ``tracemalloc_mb``    — peak Python-level allocation only. If rss_delta far
                            exceeds tracemalloc, the memory lives in C/scipy.

Usage:
  python benchmarks/profile_pedigree_graph.py --scale medium
  python benchmarks/profile_pedigree_graph.py --n 50000 --g 6 --max-degree 5
  python benchmarks/profile_pedigree_graph.py --scale small --ops extract5,inbreeding
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import gc
import io
import logging
import pstats
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from simace.simulation.simulate import run_simulation

from pedigree_graph import PedigreeGraph, compute_all_ne
from pedigree_graph._kinship_pairwise import (
    _pairwise_kinship_py,
    _pairwise_kinship_with_stats,
    pairwise_kinship,
)

# --- representative pedigree presets -------------------------------------
# (N = individuals per generation, G_ped = recorded generations.)
SCALES: dict[str, dict[str, int]] = {
    "small": {"N": 2_000, "G_ped": 8},  # ~16k  — fast smoke
    "medium": {"N": 10_000, "G_ped": 8},  # ~80k  — default
    "large": {"N": 50_000, "G_ped": 6},  # ~300k
    "xlarge": {"N": 100_000, "G_ped": 6},  # ~600k — matches config/_default.yaml
}

# Variance components + mating mirror config/_default.yaml's pedigree block,
# so the simulated structure (full/half sibs, MZ twins, cousins) matches what
# fitACE / pedsum actually consume. p_mztwin > 0 deliberately puts MZ twins in
# the pedigree; previously this forced compute_pair_kinship onto its full
# DP-matrix path. As of the direct-recurrence routine the matrix is no longer
# built — the twins still make the case representative (MZ co-coalescence).
SIM_PARAMS = {
    "mating_lambda": 0.5,
    "p_mztwin": 0.02,
    "A1": 0.5,
    "C1": 0.0,
    "E1": 0.5,
    "A2": 0.4,
    "C2": 0.2,
    "E2": 0.4,
    "rA": 0.0,
    "rC": 0.0,
    "rE": 0.0,
}


class _RSSSampler(threading.Thread):
    """Background thread that records peak RSS while an op runs."""

    def __init__(self, interval: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.peak = self._proc.memory_info().rss

    def run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join()
        return self.peak


class _LogCapture(logging.Handler):
    """Collect formatted DEBUG records from the ``pedigree_graph`` logger.

    The package already logs per-matmul timing + nnz, e.g.
    ``A3 @ A3.T computed in 1.42s (nnz=88123456)`` — the smoking gun for the
    intermediate-nnz blowup. We just grab those lines.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def measure(label: str, fn, logcap: _LogCapture) -> tuple[dict, object]:
    """Run ``fn`` once, capturing wall-time, peak RSS, tracemalloc, and logs."""
    gc.collect()
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    logcap.records.clear()

    sampler = _RSSSampler()
    sampler.peak = baseline
    sampler.start()
    tracemalloc.start()

    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0

    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = sampler.stop()

    rec = {
        "label": label,
        "seconds": elapsed,
        "rss_baseline_mb": baseline / 1e6,
        "rss_peak_mb": peak_rss / 1e6,
        "rss_delta_mb": (peak_rss - baseline) / 1e6,
        "tracemalloc_mb": tm_peak / 1e6,
        "logs": list(logcap.records),
    }
    return rec, result


def warmup() -> float:
    """Trigger numba JIT compilation on a tiny pedigree (excluded from timings).

    Kernels use ``cache=True`` so this is ~free on a warm disk cache, but we
    always pay it once per fresh environment. Reported separately so it never
    pollutes the measured kernel timings.
    """
    t0 = time.perf_counter()
    df = run_simulation(seed=0, N=64, G_ped=3, G_sim=4, **SIM_PARAMS)
    pg = PedigreeGraph(df)
    pg.compute_inbreeding()
    pg.per_gen_mean_kinship()
    pg.extract_pairs(max_degree=5)
    pg2 = PedigreeGraph(df)
    pg2.compute_pair_kinship(pg2.extract_pairs(max_degree=2))
    # Warm the direct pairwise-kinship kernel explicitly (also reached via
    # compute_pair_kinship above, but warm it directly so the stats wrapper and
    # bare-kernel microbench below never pay first-call JIT).
    pairwise_kinship(pg2.mother, pg2.father, pg2.twin, np.array([0, 1]), np.array([1, 2]))
    _pairwise_kinship_with_stats(pg2.mother, pg2.father, pg2.twin, np.array([0]), np.array([1]))
    with contextlib.suppress(Exception):  # tiny pedigree may degenerate some estimators
        compute_all_ne(PedigreeGraph(df), skip_ne_coancestry=True)
    return time.perf_counter() - t0


def build_ops(df, max_degree: int, logcap: _LogCapture, skip_coancestry: bool):
    """Return {key: thunk} where each thunk builds a FRESH pg then measures one op.

    A fresh graph per op keeps the per-op memory delta clean (no cross-op cache
    contamination). Construction happens *outside* the timed region.
    """

    def _construct():
        return measure("PedigreeGraph(df)", lambda: PedigreeGraph(df), logcap)

    def _extract(degree):
        pg = PedigreeGraph(df)
        return measure(f"extract_pairs(max_degree={degree})", lambda: pg.extract_pairs(max_degree=degree), logcap)

    def _inbreeding():
        pg = PedigreeGraph(df)
        return measure("compute_inbreeding", pg.compute_inbreeding, logcap)

    def _per_gen():
        pg = PedigreeGraph(df)
        return measure("per_gen_mean_kinship", pg.per_gen_mean_kinship, logcap)

    def _pair_kinship():
        pg = PedigreeGraph(df)
        pairs = pg.extract_pairs(max_degree=max_degree)  # prerequisite, untimed
        return measure("compute_pair_kinship", lambda: pg.compute_pair_kinship(pairs), logcap)

    def _all_ne():
        pg = PedigreeGraph(df)
        return measure(
            f"compute_all_ne(skip_coancestry={skip_coancestry})",
            lambda: compute_all_ne(pg, skip_ne_coancestry=skip_coancestry),
            logcap,
        )

    return {
        "construct": _construct,
        "extract2": lambda: _extract(2),
        "extract5": lambda: _extract(max_degree),
        "inbreeding": _inbreeding,
        "per_gen": _per_gen,
        "pair_kinship": _pair_kinship,
        "all_ne": _all_ne,
    }


def cprofile_pass(df, max_degree: int) -> str:
    """CProfile a heavy representative sequence; return top-N by tottime+cumtime.

    cProfile can't see *inside* njit/scipy-C frames, but it attributes wall-time
    to the dispatch boundaries — so you can read off how much sits in scipy
    sparse matmul entry points vs numba dispatch vs python orchestration.
    """
    pr = cProfile.Profile()
    pg = PedigreeGraph(df)
    pr.enable()
    pg.extract_pairs(max_degree=max_degree)
    pg2 = PedigreeGraph(df)
    pairs = pg2.extract_pairs(max_degree=2)
    pg2.compute_pair_kinship(pairs)
    pg2.per_gen_mean_kinship()
    pg2.compute_inbreeding()
    pr.disable()

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    buf.write("\n=== TOP 30 BY CUMULATIVE TIME (where wall-time is spent) ===\n")
    st.sort_stats("cumulative").print_stats(30)
    buf.write("\n=== TOP 30 BY TOTAL (self) TIME (the actual hot frames) ===\n")
    st.sort_stats("tottime").print_stats(30)
    return buf.getvalue()


def fmt_summary(records: list[dict]) -> str:
    """Render the per-op wall-time + memory table from measured records."""
    head = f"{'operation':<34}{'wall_s':>10}{'rss_delta_MB':>14}{'tracemalloc_MB':>16}"
    lines = [head, "-" * len(head)]
    lines.extend(
        f"{r['label']:<34}{r['seconds']:>10.3f}{r['rss_delta_mb']:>14.1f}{r['tracemalloc_mb']:>16.1f}" for r in records
    )
    return "\n".join(lines)


def fmt_nnz_logs(records: list[dict]) -> str:
    """Render the captured per-matmul timing + intermediate-nnz DEBUG logs."""
    out = ["=== Per-matmul timing + intermediate nnz (from pedigree_graph DEBUG logs) ==="]
    any_logs = False
    for r in records:
        relevant = [ln for ln in r["logs"] if "nnz" in ln or "computed in" in ln or "total:" in ln]
        if relevant:
            any_logs = True
            out.append(f"\n[{r['label']}]")
            out.extend(f"  {ln}" for ln in relevant)
    if not any_logs:
        out.append("  (none captured)")
    return "\n".join(out)


def _flatten_pairs(pairs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every code's (idx1, idx2) into flat graph-space arrays."""
    non_empty = [v for v in pairs.values() if len(v[0])]
    if not non_empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    a = np.concatenate([v[0] for v in non_empty]).astype(np.int64)
    b = np.concatenate([v[1] for v in non_empty]).astype(np.int64)
    return a, b


def _deep_inbred_pedigree(n_gen: int = 22, per_gen: int = 40, n_founders: int = 6, seed: int = 1) -> pl.DataFrame:
    """Build a deep, heavily-inbred pedigree for worst-case stress testing.

    Each generation mates only within the immediately preceding one, descending
    from a handful of founders.  Over many generations this maximizes the
    distinct-ancestor count ``A`` and
    inbreeding depth — the worst case for the ``O(P * A^2)`` direct recurrence
    and the case that makes the full kinship matrix near-dense.
    """
    rng = np.random.default_rng(seed)
    ids = list(range(n_founders))
    mother = [-1] * n_founders
    father = [-1] * n_founders
    twin = [-1] * n_founders
    gen = [0] * n_founders
    sex = [i % 2 for i in range(n_founders)]
    cur = list(range(n_founders))
    next_id = n_founders
    for g in range(1, n_gen + 1):
        new_gen: list[int] = []
        for _ in range(per_gen):
            m = int(rng.choice(cur))
            f = int(rng.choice(cur))
            while f == m and len(cur) > 1:
                f = int(rng.choice(cur))
            ids.append(next_id)
            mother.append(m)
            father.append(f)
            twin.append(-1)
            gen.append(g)
            sex.append(int(rng.integers(0, 2)))
            new_gen.append(next_id)
            next_id += 1
        cur = new_gen
    return pl.DataFrame({"id": ids, "mother": mother, "father": father, "twin": twin, "sex": sex, "generation": gen})


def pairwise_diagnostics(df, max_degree: int, seed: int, py_cap: int = 10_000) -> str:
    """Memo/stack stats for the direct kernel + a Python-vs-numba microbench.

    Confirms the memo stays far below ``n**2`` on the representative simulated
    pedigree and quantifies the numba speedup over the pure-Python reference.
    The Python reference is timed on a capped random subset (it is far slower);
    the cap is reported, never silent.
    """
    pg = PedigreeGraph(df)
    pairs = pg.extract_pairs(max_degree=max_degree)
    a, b = _flatten_pairs(pairs)
    n, p = pg.n, a.shape[0]

    t0 = time.perf_counter()
    _, stats = _pairwise_kinship_with_stats(pg.mother, pg.father, pg.twin, a, b)
    nb_full = time.perf_counter() - t0

    if p > py_cap:
        rng = np.random.default_rng(seed)
        sel = rng.choice(p, py_cap, replace=False)
        sa, sb = a[sel], b[sel]
        cap_note = f"subset {py_cap:,} of {p:,} (Python reference too slow on full set)"
    else:
        sa, sb = a, b
        cap_note = f"all {p:,} pairs"

    t0 = time.perf_counter()
    nb_sub = pairwise_kinship(pg.mother, pg.father, pg.twin, sa, sb)
    nb_sub_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    py_sub = _pairwise_kinship_py(pg.mother, pg.father, pg.twin, sa, sb)
    py_sub_t = time.perf_counter() - t0
    bit_exact = bool(np.array_equal(nb_sub, py_sub))
    speedup = py_sub_t / nb_sub_t if nb_sub_t > 0 else float("nan")

    return "\n".join(
        [
            "=== Direct pairwise-kinship diagnostics (simulated pedigree) ===",
            f"  individuals n         : {n:,}   (n^2 = {n * n:,}, the full-matrix cell count)",
            f"  requested pairs P     : {p:,}",
            f"  numba kernel (full P) : {nb_full:.3f}s",
            f"  memo entries          : {stats['memo_entries']:,}  "
            f"({stats['memo_entries'] / max(n * n, 1):.3%} of n^2)",
            f"  memo capacity / grows : {stats['memo_capacity']:,} / {stats['memo_grows']}",
            f"  max work-stack depth  : {stats['max_stack_depth']:,}",
            f"  py-vs-numba microbench [{cap_note}]:",
            f"    pure-Python ref     : {py_sub_t:.3f}s",
            f"    numba kernel        : {nb_sub_t:.3f}s",
            f"    speedup             : {speedup:.1f}x   bit-exact: {bit_exact}",
        ]
    )


def stress_diagnostics(seed: int) -> str:
    """Deep/inbred worst-case guard: memo must stay bounded and beat the matrix.

    Builds a deep, heavily-inbred pedigree (large ancestor sets), runs the
    direct kernel on all extracted pairs, and compares wall time + correctness
    against the full ``kinship_matrix(0.0)`` path it replaces.
    """
    df = _deep_inbred_pedigree(seed=seed)
    pg = PedigreeGraph(df)
    pairs = pg.extract_pairs(max_degree=5)
    a, b = _flatten_pairs(pairs)
    n, p = pg.n, a.shape[0]

    overflowed = False
    try:
        t0 = time.perf_counter()
        out_nb, stats = _pairwise_kinship_with_stats(pg.mother, pg.father, pg.twin, a, b)
        nb_t = time.perf_counter() - t0
    except ValueError:
        overflowed = True
        stats, nb_t, out_nb = (
            {"memo_entries": -1, "memo_capacity": -1, "memo_grows": -1, "max_stack_depth": -1},
            float("nan"),
            None,
        )

    pg_mat = PedigreeGraph(df)
    t0 = time.perf_counter()
    k = pg_mat.kinship_matrix(0.0)
    mat_t = time.perf_counter() - t0
    nnz = k.nnz
    ok = "n/a"
    if out_nb is not None:
        exp = np.asarray(k.tocsr()[a, b]).ravel()
        ok = str(bool(np.allclose(out_nb, exp, atol=1e-9)))

    return "\n".join(
        [
            "=== Worst-case stress: deep/inbred pedigree ===",
            f"  individuals n         : {n:,}   (n^2 = {n * n:,})",
            f"  requested pairs P     : {p:,}",
            f"  capacity-limit raised : {overflowed}",
            f"  memo entries          : {stats['memo_entries']:,}  "
            f"({stats['memo_entries'] / max(n * n, 1):.3%} of n^2)",
            f"  memo capacity / grows : {stats['memo_capacity']:,} / {stats['memo_grows']}",
            f"  max work-stack depth  : {stats['max_stack_depth']:,}",
            f"  matrix nnz (~density) : {nnz:,}  ({nnz / max(n * n, 1):.1%} dense)",
            f"  direct kernel wall    : {nb_t:.3f}s",
            f"  full-matrix wall      : {mat_t:.3f}s  (the path the kernel replaces)",
            f"  direct == matrix      : {ok}",
        ]
    )


def main() -> None:
    """Parse CLI args, run the profiling sequence, and print the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=list(SCALES), default="medium")
    ap.add_argument("--n", type=int, default=None, help="individuals per generation (overrides --scale)")
    ap.add_argument("--g", type=int, default=None, help="recorded generations G_ped (overrides --scale)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-degree", type=int, default=5, help="max_degree for the heavy extract_pairs op (1-5)")
    ap.add_argument(
        "--ops",
        default="construct,extract2,extract5,inbreeding,per_gen,pair_kinship,all_ne",
        help="comma-separated subset of ops to run",
    )
    ap.add_argument(
        "--skip-coancestry",
        action="store_true",
        default=True,
        help="skip O(n^2)-ish ne_coancestry (default True; protects large scales)",
    )
    ap.add_argument("--with-coancestry", dest="skip_coancestry", action="store_false")
    ap.add_argument("--no-cprofile", action="store_true", help="skip the cProfile attribution pass")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "reports")
    args = ap.parse_args()

    N = args.n if args.n is not None else SCALES[args.scale]["N"]
    G_ped = args.g if args.g is not None else SCALES[args.scale]["G_ped"]
    tag = f"n{N}_g{G_ped}_d{args.max_degree}_s{args.seed}"

    # Capture the package's own per-matmul nnz/timing instrumentation.
    logcap = _LogCapture()
    pg_logger = logging.getLogger("pedigree_graph")
    pg_logger.setLevel(logging.DEBUG)
    pg_logger.addHandler(logcap)

    print(f"# pedigree_graph profiling harness — scale={args.scale} N={N:,} G_ped={G_ped}")

    print("[1/4] numba JIT warmup (excluded from measured timings) ...", flush=True)
    warm_s = warmup()
    print(f"      warmup: {warm_s:.2f}s\n")

    print("[2/4] simulating pedigree ...", flush=True)
    t0 = time.perf_counter()
    df = run_simulation(seed=args.seed, N=N, G_ped=G_ped, G_sim=G_ped + 2, **SIM_PARAMS)
    sim_s = time.perf_counter() - t0
    n_ind = len(df)
    n_founders = int(((df["mother"].to_numpy() < 0) & (df["father"].to_numpy() < 0)).sum())
    n_twins = int((df["twin"].to_numpy() >= 0).sum())
    print(f"      {n_ind:,} individuals  ({n_founders:,} founders, {n_twins:,} twin rows)  in {sim_s:.2f}s\n")

    ops = build_ops(df, args.max_degree, logcap, args.skip_coancestry)
    selected = [o.strip() for o in args.ops.split(",") if o.strip()]

    print("[3/4] measuring operations ...", flush=True)
    records: list[dict] = []
    for key in selected:
        if key not in ops:
            print(f"      ! unknown op '{key}' (have: {', '.join(ops)})")
            continue
        rec, _ = ops[key]()
        records.append(rec)
        print(f"      {rec['label']:<34} {rec['seconds']:>8.3f}s  rssΔ={rec['rss_delta_mb']:>8.1f}MB", flush=True)

    pairwise_text = ""
    if "pair_kinship" in selected:
        print("\n[3b/4] direct pairwise-kinship diagnostics + worst-case stress ...", flush=True)
        pairwise_text = pairwise_diagnostics(df, args.max_degree, args.seed) + "\n\n" + stress_diagnostics(args.seed)
        print(pairwise_text)

    print("\n[4/4] cProfile attribution pass ...", flush=True)
    prof_text = "" if args.no_cprofile else cprofile_pass(df, args.max_degree)

    # ---- assemble report ----
    sections = [
        "pedigree_graph profiling report",
        f"scale={args.scale}  N={N:,}  G_ped={G_ped}  individuals={n_ind:,}  "
        f"founders={n_founders:,}  twin_rows={n_twins:,}",
        f"numba warmup (excluded): {warm_s:.2f}s   |   simulate: {sim_s:.2f}s",
        "",
        fmt_summary(records),
        "",
        fmt_nnz_logs(records),
    ]
    if pairwise_text:
        sections += ["", pairwise_text]
    if prof_text:
        sections += ["", prof_text]
    report = "\n".join(sections)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f"profile_{tag}.txt"
    report_path.write_text(report)

    print("\n" + "=" * 72)
    print(fmt_summary(records))
    print()
    print(fmt_nnz_logs(records))
    print("=" * 72)
    print(f"\nFull report (incl. cProfile top-30): {report_path}")
    print(
        "\nReading it:\n"
        "  * extract_pairs dominates wall-time AND rss_delta >> tracemalloc, with a\n"
        "    large `A3 @ A3.T (nnz=...)` line  -> the lever is a fused thresholded\n"
        "    sparse product (numba or C++); a straight C++ port of the njit kernels\n"
        "    is not the win.\n"
        "  * inbreeding / per_gen / pair_kinship dominate  -> numba is already at the\n"
        "    compiled ceiling (see docs/adr/0001); C++ buys ~nothing there."
    )


if __name__ == "__main__":
    main()
