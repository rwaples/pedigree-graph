from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import norm

from fitace.kinship.grm_io import build_household_matrix
from fitace_pcgc import fit_pcgc


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, pl.DataFrame): return x.to_dicts()
    if isinstance(x, Path): return str(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype")
    ap.add_argument("threshold_npz")
    ap.add_argument("degree5_npz")
    ap.add_argument("--prevalence", type=float, default=0.1)
    ap.add_argument("--iter-reml", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    ph = pl.read_parquet(args.phenotype)
    K = {
        "threshold": sp.load_npz(args.threshold_npz).tocsc(),
        "degree5": sp.load_npz(args.degree5_npz).tocsc(),
    }
    liability = ph["liability1"].to_numpy().astype(np.float64)
    ybin = (liability >= norm.ppf(1.0 - args.prevalence)).astype(np.float64)
    hh = ph["household_id"].to_numpy()
    C = build_household_matrix(hh)
    iids = ph["id"].to_numpy().astype(str)
    report = {"n": len(ph), "prevalence": args.prevalence, "sample_prevalence": float(ybin.mean()), "pcgc": {}}
    for name, kmat in K.items():
        t0 = time.perf_counter()
        result = fit_pcgc(
            ybin,
            2.0 * kmat,
            kinship_C=C,
            prevalence=args.prevalence,
            sample_prevalence=args.prevalence,
            iids=iids,
            grm_threshold=0.0,
            jackknife_blocks=30,
            model="ACE",
            backend="reference",
            moment="auto",
            threads=1,
        )
        report["pcgc"][name] = {
            "wall_outer_s": time.perf_counter() - t0,
            "vc": result.vc.to_dicts(),
            "cov": result.cov.to_dicts(),
            "backend": result.backend,
            "n_pairs_A": result.n_pairs_A,
            "n_pairs_C": result.n_pairs_C,
            "c_factor": result.c_factor,
            "moment": result.moment,
            "newton_converged": result.newton_converged,
            "newton_iterations": result.newton_iters,
        }
    if args.iter_reml:
        from fitace_iter_reml import fit_iter_reml
        report["iter_reml"] = {}
        for name, kmat in K.items():
            work = Path(f"/tmp/threshold-iter-{name}")
            t0 = time.perf_counter()
            result = fit_iter_reml(
                y=liability,
                kinship=kmat,
                household_id=hh,
                iids=iids,
                grm_threshold=0.0,
                phase1_probes=40,
                phase2_probes=30,
                max_iter=20,
                tol=1e-2,
                pcg_tol=1e-5,
                pcg_max_iter=500,
                compute_logdet=False,
                seed=42,
                threads=1,
                work_dir=work,
                cleanup=False,
                binary=Path("fitACE/fitACE_iter_reml/ace_iter_reml/build-fp32/ace_iter_reml").resolve(),
            )
            report["iter_reml"][name] = {
                "wall_outer_s": time.perf_counter() - t0,
                "vc": result.vc.to_dicts(),
                "cov": result.cov.to_dicts(),
                "converged": result.converged,
                "n_iter": result.n_iter,
                "wall_s": result.wall_s,
                "iter_log": result.iter_log.to_dicts(),
                "bench": result.bench.to_dicts() if result.bench is not None else None,
            }
    text = json.dumps(report, default=jsonable, indent=2, sort_keys=True)
    print(text)
    if args.out: Path(args.out).write_text(text + "\n")

if __name__ == "__main__": main()
