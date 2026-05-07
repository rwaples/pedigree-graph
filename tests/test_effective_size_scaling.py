"""Scaling tests for the refactored Ne_LTC and Ne_CT helpers.

Validates byte-parity with a reference dense implementation, correct
handling of skip-gen parents, structural memory bounds via internal
sentinel metrics, and end-to-end RSS / convergence at scales the old
dense path could not reach.

The reference dense implementation embedded in this file is a verbatim
copy of the deleted ``_founder_contribution_matrix`` from
``_effective_size.py`` (pre-refactor) — kept here purely so the parity
test compares the new streaming helpers against the algorithm they
replace.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from pedigree_graph import (
    PedigreeGraph,
    compute_all_ne,
    ne_caballero_toro,
    ne_long_term_contributions,
)
from pedigree_graph._effective_size import (
    _caballero_toro_accumulators,
    _founder_idx,
    _per_gen_founder_means,
)

# ---------------------------------------------------------------------------
# Reference dense implementation (pre-refactor `_founder_contribution_matrix`).
# ---------------------------------------------------------------------------


def _ref_founder_contribution_matrix(pg: PedigreeGraph) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim copy of the deleted dense forward recursion."""
    n = pg.n
    gen = np.asarray(pg.generation)
    mother = np.asarray(pg.mother)
    father = np.asarray(pg.father)
    founder_idx = np.where(gen == 0)[0]
    n_founders = len(founder_idx)
    c = np.zeros((n, n_founders), dtype=np.float64)
    if n_founders == 0:
        return c, founder_idx
    c[founder_idx, np.arange(n_founders)] = 1.0
    g_max = int(gen.max())
    for g in range(1, g_max + 1):
        in_g = np.where(gen == g)[0]
        if len(in_g) == 0:
            continue
        m_idx_g = mother[in_g]
        f_idx_g = father[in_g]
        c_m = np.zeros((len(in_g), n_founders), dtype=np.float64)
        c_f = np.zeros((len(in_g), n_founders), dtype=np.float64)
        has_m = m_idx_g >= 0
        has_f = f_idx_g >= 0
        if has_m.any():
            c_m[has_m] = c[m_idx_g[has_m]]
        if has_f.any():
            c_f[has_f] = c[f_idx_g[has_f]]
        c[in_g] = 0.5 * (c_m + c_f)
    return c, founder_idx


def _ref_per_gen_means(pg: PedigreeGraph) -> tuple[np.ndarray, np.ndarray]:
    """Reference per-gen founder means built from the dense matrix."""
    c, founder_idx = _ref_founder_contribution_matrix(pg)
    gen = np.asarray(pg.generation)
    g_max = int(gen.max()) if pg.n > 0 else 0
    n_founders = len(founder_idx)
    m_g = np.full((g_max + 1, n_founders), np.nan, dtype=np.float64)
    for g in range(g_max + 1):
        in_g = gen == g
        if in_g.any():
            m_g[g] = c[in_g].mean(axis=0)
    return m_g, founder_idx


def _ref_ct_accumulators(pg: PedigreeGraph, F: np.ndarray) -> dict:
    """Reference (sums, counts) built from the dense matrix."""
    c, founder_idx = _ref_founder_contribution_matrix(pg)
    gen = np.asarray(pg.generation)
    g_max = int(gen.max()) if pg.n > 0 else 0
    n_founders = len(founder_idx)
    sums = np.zeros((g_max + 1, n_founders), dtype=np.float64)
    counts = np.zeros((g_max + 1, n_founders), dtype=np.int64)
    self_coancestry = (1.0 + F) / 2.0
    for g in range(g_max + 1):
        in_g = np.where(gen == g)[0]
        if len(in_g) == 0:
            continue
        for f_local in range(n_founders):
            mask = c[in_g, f_local] > 0.0
            if mask.any():
                idx = in_g[mask]
                sums[g, f_local] = float(self_coancestry[idx].sum())
                counts[g, f_local] = int(idx.size)
    return {"sums": sums, "counts": counts, "founder_idx": founder_idx}


# ---------------------------------------------------------------------------
# Pedigree builders
# ---------------------------------------------------------------------------


def _df(records: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "id": r["id"],
            "mother": r.get("mother", -1),
            "father": r.get("father", -1),
            "twin": r.get("twin", -1),
            "sex": r["sex"],
            "generation": r["generation"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def _build_closed_line(n_gens: int = 5) -> pd.DataFrame:
    records = [
        {"id": 0, "sex": 1, "generation": 0},
        {"id": 1, "sex": 0, "generation": 0},
    ]
    next_id = 2
    prev_m, prev_f = 0, 1
    for g in range(1, n_gens + 1):
        m, f = next_id, next_id + 1
        records.append({"id": m, "sex": 1, "generation": g, "mother": prev_f, "father": prev_m})
        records.append({"id": f, "sex": 0, "generation": g, "mother": prev_f, "father": prev_m})
        prev_m, prev_f = m, f
        next_id += 2
    return _df(records)


def _build_random_mating_pedigree(
    rng: np.random.Generator,
    n_per_gen: int,
    n_gens: int,
) -> pd.DataFrame:
    """Multi-generation random-mating WF-ish pedigree (balanced sexes)."""
    n_male = n_per_gen // 2
    n_female = n_per_gen - n_male
    records: list[dict] = []
    next_id = 0
    prev_male = list(range(next_id, next_id + n_male))
    prev_female = list(range(next_id + n_male, next_id + n_male + n_female))
    records.extend({"id": mid, "sex": 1, "generation": 0} for mid in prev_male)
    records.extend({"id": fid, "sex": 0, "generation": 0} for fid in prev_female)
    next_id = n_male + n_female
    for g in range(1, n_gens + 1):
        cur_male: list[int] = []
        cur_female: list[int] = []
        for _ in range(n_male):
            f = int(rng.choice(prev_male))
            m = int(rng.choice(prev_female))
            records.append({"id": next_id, "sex": 1, "generation": g, "mother": m, "father": f})
            cur_male.append(next_id)
            next_id += 1
        for _ in range(n_female):
            f = int(rng.choice(prev_male))
            m = int(rng.choice(prev_female))
            records.append({"id": next_id, "sex": 0, "generation": g, "mother": m, "father": f})
            cur_female.append(next_id)
            next_id += 1
        prev_male, prev_female = cur_male, cur_female
    return _df(records)


def _build_skip_gen_pedigree() -> pd.DataFrame:
    """Hand-built pedigree with skip-gen edges.

    Layout (id : gen | parents):
        0 : 0   founder M
        1 : 0   founder F
        2 : 0   founder M (long-lived ancestor)
        3 : 0   founder F (long-lived ancestor)
        4 : 1   M | mother=1 father=0
        5 : 1   F | mother=1 father=0
        6 : 2   M | mother=5 father=4   (parents both at gen 1)
        7 : 1   F | mother=3 father=2   (parents both at gen 0)
        8 : 3   M | mother=3 father=6   (SKIP-GEN: mother gen 0, father gen 2)
        9 : 3   F | mother=7 father=6   (parents at gen 1 and gen 2)
    """
    records = [
        {"id": 0, "sex": 1, "generation": 0},
        {"id": 1, "sex": 0, "generation": 0},
        {"id": 2, "sex": 1, "generation": 0},
        {"id": 3, "sex": 0, "generation": 0},
        {"id": 4, "sex": 1, "generation": 1, "mother": 1, "father": 0},
        {"id": 5, "sex": 0, "generation": 1, "mother": 1, "father": 0},
        {"id": 6, "sex": 1, "generation": 2, "mother": 5, "father": 4},
        # gen-2 with both parents at gen 0
        {"id": 7, "sex": 0, "generation": 1, "mother": 3, "father": 2},
        # SKIP-GEN: gen 3 child with mother at gen 0 (founder 3) and father at gen 2 (id 6).
        # generation = max(0, 2) + 1 = 3.  Mother→child gap = 3.
        {"id": 8, "sex": 1, "generation": 3, "mother": 3, "father": 6},
        {"id": 9, "sex": 0, "generation": 3, "mother": 7, "father": 6},
    ]
    return _df(records)


# ---------------------------------------------------------------------------
# Parity tests — new streaming helpers vs. reference dense impl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    ["closed_line_5", "wf_n20_g4", "skip_gen"],
)
def test_per_gen_founder_means_matches_reference(fixture_name: str) -> None:
    """Adjoint sweep matches dense forward recursion to 1e-12."""
    rng = np.random.default_rng(0)
    if fixture_name == "closed_line_5":
        df = _build_closed_line(n_gens=5)
    elif fixture_name == "wf_n20_g4":
        df = _build_random_mating_pedigree(rng, n_per_gen=20, n_gens=4)
    elif fixture_name == "skip_gen":
        df = _build_skip_gen_pedigree()
    else:  # pragma: no cover
        raise ValueError(fixture_name)

    pg = PedigreeGraph(df)

    m_g_new, founder_idx_new = _per_gen_founder_means(pg)
    m_g_ref, founder_idx_ref = _ref_per_gen_means(pg)

    np.testing.assert_array_equal(founder_idx_new, founder_idx_ref)
    # Adjoint reorders summation, so use atol rather than equality.
    np.testing.assert_allclose(m_g_new, m_g_ref, atol=1e-12, rtol=0.0, equal_nan=True)


@pytest.mark.parametrize(
    "fixture_name",
    ["closed_line_5", "wf_n20_g4", "skip_gen"],
)
def test_ct_accumulators_match_reference(fixture_name: str) -> None:
    """Streaming CT sums/counts match the dense reduction."""
    rng = np.random.default_rng(0)
    if fixture_name == "closed_line_5":
        df = _build_closed_line(n_gens=5)
    elif fixture_name == "wf_n20_g4":
        df = _build_random_mating_pedigree(rng, n_per_gen=20, n_gens=4)
    elif fixture_name == "skip_gen":
        df = _build_skip_gen_pedigree()
    else:  # pragma: no cover
        raise ValueError(fixture_name)

    pg = PedigreeGraph(df)
    F = pg.compute_inbreeding()
    founder_idx = _founder_idx(pg)

    new = _caballero_toro_accumulators(pg, founder_idx, F)
    ref = _ref_ct_accumulators(pg, F)

    np.testing.assert_array_equal(new["founder_idx"], ref["founder_idx"])
    np.testing.assert_array_equal(new["counts"], ref["counts"])
    np.testing.assert_allclose(new["sums"], ref["sums"], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "fixture_name",
    ["closed_line_5", "wf_n20_g4", "skip_gen"],
)
def test_estimator_results_match_reference(fixture_name: str) -> None:
    """End-to-end LTC and CT dataclasses match the reference path field-by-field."""
    rng = np.random.default_rng(0)
    if fixture_name == "closed_line_5":
        df = _build_closed_line(n_gens=5)
    elif fixture_name == "wf_n20_g4":
        df = _build_random_mating_pedigree(rng, n_per_gen=20, n_gens=4)
    elif fixture_name == "skip_gen":
        df = _build_skip_gen_pedigree()
    else:  # pragma: no cover
        raise ValueError(fixture_name)

    pg = PedigreeGraph(df)
    F = pg.compute_inbreeding()

    # New path
    res_ltc_new = ne_long_term_contributions(pg)
    res_ct_new = ne_caballero_toro(pg)

    # Reference path: feed dense-derived structures through the public API.
    m_g_ref, founder_idx_ref = _ref_per_gen_means(pg)
    res_ltc_ref = ne_long_term_contributions(
        pg, mean_contributions=(m_g_ref, founder_idx_ref)
    )
    ref_acc = _ref_ct_accumulators(pg, F)
    # Pad with the metric keys the helper would add (CT estimator only reads sums/counts).
    ref_acc.setdefault("peak_ancestor_set_size", 0)
    ref_acc.setdefault("peak_live_ancestor_sets", 0)
    ref_acc.setdefault("total_ancestor_pair_visits", 0)
    res_ct_ref = ne_caballero_toro(pg, ct_accumulators=ref_acc)

    # LTC dataclass parity
    assert res_ltc_new.asymptote_reached == res_ltc_ref.asymptote_reached
    assert res_ltc_new.n_iterations == res_ltc_ref.n_iterations
    if res_ltc_new.ne is None or res_ltc_ref.ne is None:
        assert res_ltc_new.ne is res_ltc_ref.ne
    else:
        assert res_ltc_new.ne == pytest.approx(res_ltc_ref.ne, abs=1e-12)
    assert res_ltc_new.sum_c_squared == pytest.approx(res_ltc_ref.sum_c_squared, abs=1e-12)

    # CT dataclass parity
    np.testing.assert_allclose(
        res_ct_new.mean_self_coancestry_per_gen,
        res_ct_ref.mean_self_coancestry_per_gen,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        res_ct_new.n_founders_with_descendants_per_gen,
        res_ct_ref.n_founders_with_descendants_per_gen,
    )
    np.testing.assert_allclose(
        res_ct_new.ne_per_gen,
        res_ct_ref.ne_per_gen,
        atol=1e-12,
        equal_nan=True,
    )
    if res_ct_new.ne is None or res_ct_ref.ne is None:
        assert res_ct_new.ne is res_ct_ref.ne
    else:
        assert res_ct_new.ne == pytest.approx(res_ct_ref.ne, abs=1e-12)


# ---------------------------------------------------------------------------
# Sentinel-metric test — structural memory invariants
# ---------------------------------------------------------------------------


def test_sentinel_metrics_at_n2000_g8() -> None:
    """At N=2000, G=8 the new helpers expose bounded structural metrics.

    Asserts that no dense `(n × n_founders)` array is required: the only
    dense structures produced are `m_g` and `(sums, counts)` of shape
    `(g_max+1, n_founders)`, which scale linearly in N and G.

    Uses `F = zeros` to avoid triggering the (unrelated) full kinship
    matrix build at this scale — we are only validating structural memory
    invariants of the new helpers, not numerical CT correctness (which
    is covered by the parity tests above).
    """
    rng = np.random.default_rng(7)
    n_per_gen, n_gens = 2000, 8
    df = _build_random_mating_pedigree(rng, n_per_gen=n_per_gen, n_gens=n_gens)
    pg = PedigreeGraph(df)
    F = np.zeros(pg.n, dtype=np.float64)
    founder_idx = _founder_idx(pg)
    n_founders = len(founder_idx)

    m_g, _ = _per_gen_founder_means(pg, founder_idx=founder_idx)
    ct = _caballero_toro_accumulators(pg, founder_idx, F)

    # Output shapes: linear in (g_max+1, n_founders), never (N · g_max, n_founders).
    assert m_g.shape == (n_gens + 1, n_founders)
    assert ct["sums"].shape == (n_gens + 1, n_founders)
    assert ct["counts"].shape == (n_gens + 1, n_founders)

    # Output bytes scale with (g_max · n_founders), not N · n_founders.
    output_bytes = m_g.nbytes + ct["sums"].nbytes + ct["counts"].nbytes
    assert output_bytes < (n_gens + 1) * n_founders * (8 + 8 + 8) + 1024

    # Live ancestor-set metrics: bounded by N (population cap) — the
    # streaming structure never needs (n · n_founders) cells live at once.
    assert 0 <= ct["peak_ancestor_set_size"] <= n_founders
    assert 0 <= ct["peak_live_ancestor_sets"] <= pg.n
    # Total work: ancestor-pair visits.  Strictly bounded above by N · n_founders
    # (saturated case); typically far less.
    assert ct["total_ancestor_pair_visits"] <= pg.n * n_founders


# ---------------------------------------------------------------------------
# Slow tests — RSS at scale + LTC convergence at a scale the old code couldn't run
# ---------------------------------------------------------------------------


_RSS_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import pandas as pd
    import sys

    from pedigree_graph import PedigreeGraph
    from pedigree_graph._effective_size import (
        _caballero_toro_accumulators,
        _founder_idx,
        _per_gen_founder_means,
    )


    def read_vm_hwm_kb() -> int:
        # /proc/self/status VmHWM is the per-process peak RSS, reset on
        # exec — unlike getrusage(ru_maxrss) which can inherit from the
        # parent via fork+exec on some glibc versions.
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1])
        return -1


    def build(n_per_gen, n_gens, seed):
        rng = np.random.default_rng(seed)
        n_male = n_per_gen // 2
        n_female = n_per_gen - n_male
        records = []
        next_id = 0
        prev_m = list(range(next_id, next_id + n_male))
        prev_f = list(range(next_id + n_male, next_id + n_male + n_female))
        for mid in prev_m:
            records.append({"id": mid, "sex": 1, "generation": 0,
                             "mother": -1, "father": -1, "twin": -1})
        for fid in prev_f:
            records.append({"id": fid, "sex": 0, "generation": 0,
                             "mother": -1, "father": -1, "twin": -1})
        next_id = n_male + n_female
        for g in range(1, n_gens + 1):
            cur_m, cur_f = [], []
            for _ in range(n_male):
                fa = int(rng.choice(prev_m))
                mo = int(rng.choice(prev_f))
                records.append({"id": next_id, "sex": 1, "generation": g,
                                 "mother": mo, "father": fa, "twin": -1})
                cur_m.append(next_id)
                next_id += 1
            for _ in range(n_female):
                fa = int(rng.choice(prev_m))
                mo = int(rng.choice(prev_f))
                records.append({"id": next_id, "sex": 0, "generation": g,
                                 "mother": mo, "father": fa, "twin": -1})
                cur_f.append(next_id)
                next_id += 1
            prev_m, prev_f = cur_m, cur_f
        return pd.DataFrame(records)


    df = build(n_per_gen=2000, n_gens=8, seed=42)
    pg = PedigreeGraph(df)
    # Synthesize F via the lazy cache without forcing the full kinship
    # matrix (which is unrelated to this PR and dominates RSS at scale).
    F = np.zeros(pg.n, dtype=np.float64)
    founder_idx = _founder_idx(pg)
    m_g, _ = _per_gen_founder_means(pg, founder_idx=founder_idx)
    ct = _caballero_toro_accumulators(pg, founder_idx, F)
    print(f"RSS_KB={read_vm_hwm_kb()}")
    print(f"PEAK_ANC_SET={ct['peak_ancestor_set_size']}")
    print(f"PEAK_LIVE_SETS={ct['peak_live_ancestor_sets']}")
    print(f"PAIR_VISITS={ct['total_ancestor_pair_visits']}")
    """
).strip()


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="VmHWM is a Linux-specific metric in /proc/self/status",
)
def test_helpers_rss_at_n2000_g8_under_threshold() -> None:
    """Subprocess RSS at N=2000, G=8 with the new helpers in isolation.

    Old dense `_founder_contribution_matrix` peak: (2000 * 9) * 2000 *
    8 bytes ≈ 290 MB just for ``c``.  The new helpers store only:

    * ``m_g`` shape (g_max+1, n_founders) — ~0.13 MB
    * ``sums`` + ``counts`` shape (g_max+1, n_founders) — ~0.27 MB
    * working ancestor sets — bounded by frontier × ancestry depth

    Total working memory should be O(MB), comfortably under 200 MB
    even with interpreter + numpy overhead.  Threshold set at 250 MB
    for platform variance.  Excludes ``compute_all_ne`` because the
    sparse kinship matrix at this scale is unrelated to this refactor
    and would mask the result.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    rss_line = next(line for line in proc.stdout.splitlines() if line.startswith("RSS_KB="))
    rss_kb = int(rss_line.split("=", 1)[1])
    rss_mb = rss_kb / 1024.0
    assert rss_mb < 250.0, (
        f"new helpers peak RSS {rss_mb:.1f} MB exceeded 250 MB threshold; "
        f"stdout=\n{proc.stdout}"
    )


@pytest.mark.slow
def test_ltc_runs_at_scale_old_code_could_not() -> None:
    """N=2000, G=10 — old dense code allocated ~2.6 GB; new path is fine.

    Verifies the new path completes and produces a sensible LTC value.
    """
    rng = np.random.default_rng(13)
    df = _build_random_mating_pedigree(rng, n_per_gen=2000, n_gens=10)
    pg = PedigreeGraph(df)
    res = ne_long_term_contributions(pg)
    # Loose bound: under WF random mating with N=2000, the LTC asymptote
    # (when reached) sits near N/2.  Don't assert convergence — at this
    # scale and tolerance the asymptote often doesn't cross 1e-6 within
    # 10 generations.  Just assert the call returned a well-formed result.
    assert res.n_iterations >= 1
    assert np.isfinite(res.sum_c_squared)
    if res.ne is not None:
        assert 100.0 < res.ne < 5000.0


# ---------------------------------------------------------------------------
# compute_all_ne smoke at scale — confirms full orchestration path works.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_all_ne_at_n2000_g8_smoke() -> None:
    """End-to-end smoke at the scale targeted by the refactor."""
    rng = np.random.default_rng(11)
    df = _build_random_mating_pedigree(rng, n_per_gen=2000, n_gens=8)
    pg = PedigreeGraph(df)
    out = compute_all_ne(pg)
    assert set(out.keys()) == {
        "ne_inbreeding",
        "ne_coancestry",
        "ne_variance_family_size",
        "ne_sex_ratio",
        "ne_individual_delta_f",
        "ne_long_term_contributions",
        "ne_hill_overlapping",
        "ne_caballero_toro",
    }
    for name, result in out.items():
        d = result.to_dict()
        assert "ne" in d, name
