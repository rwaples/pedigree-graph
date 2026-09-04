from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from pedigree_graph import PedigreeGraph


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_graph(df: pl.DataFrame) -> PedigreeGraph:
    return PedigreeGraph.from_arrays(
        ids=df["id"].to_numpy(),
        mothers=df["mother"].to_numpy(),
        fathers=df["father"].to_numpy(),
        twins=df["twin"].to_numpy() if "twin" in df.columns else None,
        generation=df["generation"].to_numpy() if "generation" in df.columns else None,
    )


def subset(K: sp.csc_matrix, ids: np.ndarray, subset_ids: np.ndarray | None) -> sp.csc_matrix:
    if subset_ids is None:
        return K
    pos = {int(v): i for i, v in enumerate(ids)}
    rows = np.asarray([pos[int(v)] for v in subset_ids], dtype=np.intp)
    return K.tocsr()[rows].tocsc()[:, rows]


def structure_keys(K: sp.csc_matrix) -> np.ndarray:
    coo = sp.triu(K, k=1).tocoo()
    return np.sort(coo.row.astype(np.int64) * K.shape[0] + coo.col.astype(np.int64))


def build_degree5(df: pl.DataFrame) -> tuple[sp.csc_matrix, dict[str, int], float]:
    started = time.perf_counter()
    pair_graph = build_graph(df)
    pairs = pair_graph.extract_pairs(max_degree=5)
    counts = {k: len(v[0]) for k, v in pairs.items()}
    first = np.concatenate([v[0] for v in pairs.values()]).astype(np.int64, copy=False)
    second = np.concatenate([v[1] for v in pairs.values()]).astype(np.int64, copy=False)
    n = len(df)
    diag_rows = np.arange(n, dtype=np.int64)
    value_graph = build_graph(df)
    values = value_graph.compute_pair_kinship({"off": (first, second), "diag": (diag_rows, diag_rows)})
    off = values["off"].astype(np.float32, copy=False)
    diag = values["diag"].astype(np.float32, copy=False)
    K = sp.coo_matrix(
        (
            np.concatenate([off, off, diag]),
            (np.concatenate([first, second, diag_rows]), np.concatenate([second, first, diag_rows])),
        ),
        shape=(n, n),
        dtype=np.float32,
    ).tocsc()
    K.sum_duplicates()
    K.eliminate_zeros()
    return K, counts, time.perf_counter() - started


def matrix_summary(name: str, K: sp.csc_matrix, wall: float) -> dict:
    upper = sp.triu(K, k=1).tocoo()
    vals = upper.data.astype(np.float64)
    return {
        "name": name,
        "n": K.shape[0],
        "nnz": int(K.nnz),
        "offdiag_pairs": int(upper.nnz),
        "wall_s": wall,
        "rss_mb": rss_mb(),
        "bytes": int(K.data.nbytes + K.indices.nbytes + K.indptr.nbytes),
        "value_min": float(vals.min()) if vals.size else None,
        "value_max": float(vals.max()) if vals.size else None,
        "value_quantiles": np.quantile(vals, [0, .01, .1, .5, .9, .99, 1]).tolist() if vals.size else [],
    }


def top_eigs(K: sp.csc_matrix, k: int = 5) -> list[float]:
    if K.shape[0] <= 2:
        return []
    vals = eigsh(K.astype(np.float64), k=min(k, K.shape[0] - 1), which="LA", return_eigenvectors=False, tol=1e-5)
    return np.sort(vals)[::-1].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pedigree")
    ap.add_argument("--phenotype")
    ap.add_argument("--out")
    ap.add_argument("--complete", action="store_true")
    args = ap.parse_args()

    ped = pl.read_parquet(args.pedigree)
    ids = ped["id"].to_numpy()
    subset_ids = pl.read_parquet(args.phenotype)["id"].to_numpy() if args.phenotype else None
    report: dict = {"pedigree": args.pedigree, "phenotype": args.phenotype, "n_full": len(ped)}

    t0 = time.perf_counter()
    K_thr = build_graph(ped).kinship_matrix(min_kinship=0.001)
    t_thr = time.perf_counter() - t0
    report["threshold_full"] = matrix_summary("threshold_full", K_thr, t_thr)

    K_deg, counts, t_deg = build_degree5(ped)
    report["degree5_counts"] = counts
    report["degree5_full"] = matrix_summary("degree5_full", K_deg, t_deg)

    keys_t = structure_keys(K_thr)
    keys_d = structure_keys(K_deg)
    report["structure_full"] = {
        "intersection": int(np.intersect1d(keys_t, keys_d, assume_unique=True).size),
        "threshold_only": int(np.setdiff1d(keys_t, keys_d, assume_unique=True).size),
        "degree5_only": int(np.setdiff1d(keys_d, keys_t, assume_unique=True).size),
    }
    overlap = K_thr.multiply(K_deg != 0).tocoo()
    dvals = np.asarray(K_deg[overlap.row, overlap.col]).ravel()
    diffs = np.abs(overlap.data.astype(np.float64) - dvals.astype(np.float64))
    report["overlap_value_diff_full"] = {
        "n": int(diffs.size),
        "nonzero": int(np.count_nonzero(diffs)),
        "max_abs": float(diffs.max()) if diffs.size else 0.0,
        "mean_abs": float(diffs.mean()) if diffs.size else 0.0,
        "quantiles": np.quantile(diffs, [0, .5, .9, .99, 1]).tolist() if diffs.size else [],
    }

    K_thr_s = subset(K_thr, ids, subset_ids)
    K_deg_s = subset(K_deg, ids, subset_ids)
    report["threshold_subset"] = matrix_summary("threshold_subset", K_thr_s, t_thr)
    report["degree5_subset"] = matrix_summary("degree5_subset", K_deg_s, t_deg)
    keys_ts = structure_keys(K_thr_s)
    keys_ds = structure_keys(K_deg_s)
    report["structure_subset"] = {
        "intersection": int(np.intersect1d(keys_ts, keys_ds, assume_unique=True).size),
        "threshold_only": int(np.setdiff1d(keys_ts, keys_ds, assume_unique=True).size),
        "degree5_only": int(np.setdiff1d(keys_ds, keys_ts, assume_unique=True).size),
    }
    report["top_eigs_subset"] = {"threshold": top_eigs(K_thr_s), "degree5": top_eigs(K_deg_s)}

    if args.complete:
        t0 = time.perf_counter()
        K_full = build_graph(ped).kinship_matrix(min_kinship=0.0)
        t_full = time.perf_counter() - t0
        K_full_s = subset(K_full, ids, subset_ids)
        report["complete_full"] = matrix_summary("complete_full", K_full, t_full)
        report["complete_subset"] = matrix_summary("complete_subset", K_full_s, t_full)
        report["top_eigs_subset"]["complete"] = top_eigs(K_full_s)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        sp.save_npz(out.with_suffix(".threshold.npz"), K_thr_s)
        sp.save_npz(out.with_suffix(".degree5.npz"), K_deg_s)


if __name__ == "__main__":
    main()
