"""Spike harness: Python matrix engine vs Rust port, per code, on several pedigrees."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from pedigree_graph import PedigreeGraph

HERE = Path(__file__).parent
BIN = HERE / "rust/target/release/pgr-spike"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)


def random_inbred(seed, n_gen=8, pop=60):
    """Small overlapping-generation pedigree with heavy inbreeding, twins, and missing parents."""
    rng = np.random.default_rng(seed)
    ids, mothers, fathers, twins, gens = [], [], [], [], []
    nxt = 1
    pool = []
    for g in range(n_gen):
        cohort = []
        for _ in range(pop):
            if g == 0 or rng.random() < 0.1:
                m = f = -1
            else:
                cand = pool[-3 * pop :]
                m = int(rng.choice(cand))
                f = int(rng.choice(cand)) if rng.random() < 0.85 else -1
                if f == m:
                    f = -1
                if rng.random() < 0.08:
                    m = -1
            ids.append(nxt); mothers.append(m); fathers.append(f); twins.append(-1); gens.append(g)
            cohort.append(nxt)
            nxt += 1
            if rng.random() < 0.03 and m != -1:
                ids.append(nxt); mothers.append(m); fathers.append(f); twins.append(nxt - 1); gens.append(g)
                twins[-2] = nxt
                cohort.append(nxt)
                nxt += 1
        pool.extend(cohort)
    return pl.DataFrame(
        {"id": ids, "mother": mothers, "father": fathers, "twin": twins, "sex": [0] * len(ids), "generation": gens}
    )


def simulated(seed, N, G_ped):
    from simace.simulation.simulate import run_simulation

    return run_simulation(seed=seed, N=N, G_ped=G_ped, mating_lambda=1.0, p_mztwin=0.02,
                          A1=0.5, C1=0.2, E1=0.3, A2=0.4, C2=0.3, E2=0.3, rA=0.3, rC=0.5)


def run_case(name, df):
    pg = PedigreeGraph(df)
    tsv = WORK / f"{name}.tsv"
    pl.DataFrame(
        {"mother": pg.mother, "father": pg.father, "twin": pg.twin,
         "orig_mother": pg._orig_mother, "orig_father": pg._orig_father}
    ).write_csv(tsv, separator="\t")

    t0 = time.perf_counter()
    py = pg.extract_pairs(max_degree=5)
    t_py = time.perf_counter() - t0

    out = WORK / f"{name}.rust.tsv"
    t0 = time.perf_counter()
    res = subprocess.run([str(BIN), str(tsv), str(out)], capture_output=True, text=True, check=True)
    t_rs = time.perf_counter() - t0
    rs_lines = res.stderr.strip().splitlines()
    t_rs_internal = float(rs_lines[-2].split("s")[0])

    rs = {code: set() for code in py}
    if out.stat().st_size:
        rdf = pl.read_csv(out, separator="\t", has_header=False, new_columns=["code", "i", "j"])
        for code, i, j in rdf.iter_rows():
            rs[code].add((i, j))

    print(f"\n== {name}: n={pg.n}  python={t_py:.3f}s  rust={t_rs:.3f}s (compute {t_rs_internal:.3f}s)")
    ok = True
    for code, (a, b) in py.items():
        pyset = set(zip(a.tolist(), b.tolist()))
        if pyset != rs[code]:
            ok = False
            only_py = sorted(pyset - rs[code])[:5]
            only_rs = sorted(rs[code] - pyset)[:5]
            print(f"  MISMATCH {code}: py={len(pyset)} rs={len(rs[code])}  only_py={only_py} only_rs={only_rs}")
        else:
            print(f"  ok {code:6s} {len(pyset)}")
    return ok


if __name__ == "__main__":
    which = sys.argv[1:] or ["small", "inbred", "sim"]
    all_ok = True
    if "small" in which:
        all_ok &= run_case("small", pl.read_parquet(HERE.parent / "tests/data/small_pedigree.parquet"))
    if "inbred" in which:
        for s in range(5):
            all_ok &= run_case(f"inbred{s}", random_inbred(s))
    if "sim" in which:
        all_ok &= run_case("sim_5k_6", simulated(1, 5000, 6))
    if "big" in which:
        all_ok &= run_case("sim_50k_6", simulated(2, 50000, 6))
    print("\nALL MATCH" if all_ok else "\nMISMATCHES FOUND")
    sys.exit(0 if all_ok else 1)
