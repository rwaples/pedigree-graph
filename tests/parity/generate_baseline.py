"""Freeze pedigree-graph 0.7.1 outputs as the differential baseline for 0.8.

Run this against a ``v0.7.1`` worktree, never against the 0.8 branch::

    pixi run python tests/parity/generate_baseline.py \
        --package-root ../pedigree-graph-v0.7.1 --out tests/data/parity_v0.7.1

The script imports ``pedigree_graph`` from ``--package-root`` and refuses to
continue if the import resolves anywhere else. It calls only the 0.7.1 API,
frozen at base commit ``aa71c35``; the 0.8 branch deleted that API, so this
module is never executed by the suite and imports nothing from the 0.8 surface
at test time.  ``capture_v08.py`` reproduces its layout for the test.

Small fixtures (motifs, ``random_1k``, ``deep_inbred_60g``, and the shipped
``small_pedigree.parquet``) get their full oriented arrays saved in one
``.npz`` each. Large fixtures get per-code counts and SHA-256 hashes only.
``manifest.json`` records the generator version, package commit, fixture
parameters, input hashes, and every hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pedigrees  # noqa: E402

MAX_DEGREE = 5
APPROX_THRESHOLD = 0.001
SUBSAMPLE_SEED = 7


def _import_package(root: Path):
    sys.path.insert(0, str(root))
    for name in [m for m in sys.modules if m == "pedigree_graph" or m.startswith("pedigree_graph.")]:
        del sys.modules[name]
    import pedigree_graph

    resolved = Path(pedigree_graph.__file__).resolve()
    if root.resolve() not in resolved.parents:
        sys.exit(f"pedigree_graph resolved to {resolved}, not under {root}")
    return pedigree_graph


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        c = np.ascontiguousarray(arr)
        h.update(str(c.dtype).encode())
        h.update(str(c.shape).encode())
        h.update(c.tobytes())
    return h.hexdigest()


def _sorted_pairs(i: np.ndarray, j: np.ndarray, *cols: np.ndarray):
    order = np.lexsort((j, i))
    return (i[order], j[order], *[c[order] for c in cols])


def _upper_coo(K):
    coo = K.tocoo()
    keep = coo.row <= coo.col
    r, c, v = coo.row[keep].astype(np.int32), coo.col[keep].astype(np.int32), coo.data[keep].astype(np.float32)
    return _sorted_pairs(r, c, v)


def _build(pg_mod, fx):
    return pg_mod.PedigreeGraph.from_arrays(
        ids=fx["ids"],
        mothers=fx["mother"],
        fathers=fx["father"],
        twins=fx["twin"],
        sex=fx["sex"],
    )


def _subsample_frames(fx, keep, depth):
    """0.7.1's dict constructor requires ``generation``; pass the structural depth."""
    full = {
        "id": fx["ids"],
        "mother": fx["mother"],
        "father": fx["father"],
        "twin": fx["twin"],
        "sex": fx["sex"],
        "generation": depth,
    }
    sub = {k: v[keep] for k, v in full.items()}
    return full, sub


def _capture(
    pg_mod,
    fx: dict[str, np.ndarray],
    *,
    full_arrays: bool,
    approximate_matrix: Callable | None = None,
) -> tuple[dict, dict]:
    """Return ``(arrays, summary)``; ``arrays`` is empty when ``full_arrays`` is False.

    ``approximate_matrix`` lets the large differential test isolate frozen
    support parity; the separate 30k matrix integration test runs the complete
    public exact-value path.
    """
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n": len(fx["ids"]), "counts": {}, "hashes": {}}
    g = _build(pg_mod, fx)

    pairs = g.extract_pairs(max_degree=MAX_DEGREE)
    kin = g.compute_pair_kinship(pairs)
    for code, (raw_i, raw_j) in pairs.items():
        si, sj, sk = _sorted_pairs(
            np.asarray(raw_i, dtype=np.int32),
            np.asarray(raw_j, dtype=np.int32),
            np.asarray(kin[code], dtype=np.float64),
        )
        summary["counts"][code] = len(si)
        summary["hashes"][f"pairs/{code}"] = _sha(si, sj)
        summary["hashes"][f"pair_kinship/{code}"] = _sha(sk)
        if full_arrays:
            arrays[f"pairs/{code}/first"] = si
            arrays[f"pairs/{code}/second"] = sj
            arrays[f"pair_kinship/{code}"] = sk

    summary["streaming_counts"] = {k: int(v) for k, v in g.count_pairs_streaming(max_degree=MAX_DEGREE).items()}

    F = np.asarray(g.compute_inbreeding(), dtype=np.float64)
    n_anc = np.asarray(g.compute_n_ancestors())
    try:
        n_desc = np.asarray(g.compute_n_descendants())
    except (OverflowError, RuntimeError):
        # 0.7.1 raised OverflowError; 0.8 raises ResourceError(RuntimeError).
        n_desc = None
        summary["n_descendants_overflow"] = True
    theta = np.asarray(g.per_gen_mean_kinship(), dtype=np.float64)
    depth = np.asarray(g.generation, dtype=np.int32)
    vectors = [("inbreeding", F), ("n_ancestors", n_anc), ("per_gen_mean_kinship", theta), ("depth", depth)]
    if n_desc is not None:
        vectors.append(("n_descendants", n_desc))
    for name, arr in vectors:
        summary["hashes"][name] = _sha(arr)
        if full_arrays:
            arrays[name] = arr

    approximate = (
        g.kinship_matrix(min_kinship=APPROX_THRESHOLD) if approximate_matrix is None else approximate_matrix(g)
    )
    r, c, v = _upper_coo(approximate)
    summary["counts"]["approx_support_upper_nnz"] = len(r)
    summary["hashes"]["approx_support"] = _sha(r, c)
    summary["hashes"]["approx_values"] = _sha(v)
    if full_arrays:
        arrays["approx/row"], arrays["approx/col"], arrays["approx/val"] = r, c, v
        r0, c0, v0 = _upper_coo(g.kinship_matrix(min_kinship=0.0))
        summary["counts"]["complete_upper_nnz"] = len(r0)
        summary["hashes"]["complete_support"] = _sha(r0, c0)
        summary["hashes"]["complete_values"] = _sha(v0)
        arrays["complete/row"], arrays["complete/col"], arrays["complete/val"] = r0, c0, v0

    keep = pedigrees.subsample_selection(fx, SUBSAMPLE_SEED)
    full, sub = _subsample_frames(fx, keep, depth)
    gs = pg_mod.PedigreeGraph.from_subsample(full, sub)
    sub_pairs = gs.extract_pairs(max_degree=MAX_DEGREE)
    summary["subsample"] = {"seed": SUBSAMPLE_SEED, "n": len(keep), "counts": {}, "hashes": {}}
    summary["hashes"]["subsample/rows"] = _sha(keep.astype(np.int64))
    if full_arrays:
        arrays["subsample/rows"] = keep.astype(np.int64)
    for code, (i, j) in sub_pairs.items():
        si, sj = _sorted_pairs(np.asarray(i, dtype=np.int32), np.asarray(j, dtype=np.int32))
        summary["subsample"]["counts"][code] = len(si)
        summary["subsample"]["hashes"][code] = _sha(si, sj)
        if full_arrays:
            arrays[f"subsample/pairs/{code}/first"] = si
            arrays[f"subsample/pairs/{code}/second"] = sj
    return arrays, summary


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "parity_v0.7.1")
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()

    pg_mod = _import_package(args.package_root)
    import polars as pl

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator_version": pedigrees.GENERATOR_VERSION,
        "package_commit": _git_commit(args.package_root),
        "package_version": getattr(pg_mod, "__version__", None),
        "max_degree": MAX_DEGREE,
        "approx_threshold": APPROX_THRESHOLD,
        "fixtures": {},
    }

    fixtures: list[tuple[str, dict, dict[str, np.ndarray], bool]] = []
    for name, fx in pedigrees.motif_fixtures().items():
        fixtures.append((name, {"kind": "motif"}, fx, True))
    df = pl.read_parquet(HERE.parent / "data" / "small_pedigree.parquet")
    shipped = {
        "ids": df["id"].to_numpy().astype(np.int64),
        "mother": df["mother"].to_numpy().astype(np.int64),
        "father": df["father"].to_numpy().astype(np.int64),
        "twin": df["twin"].to_numpy().astype(np.int64) if "twin" in df.columns else np.full(len(df), -1, np.int64),
        "sex": df["sex"].to_numpy().astype(np.int8),
    }
    fixtures.append(("small_pedigree", {"kind": "shipped", "file": "tests/data/small_pedigree.parquet"}, shipped, True))
    for name, params in pedigrees.RANDOM_FIXTURES.items():
        fixtures.append((name, {"kind": "random", **params}, pedigrees.build_random(name, params), True))
    if not args.skip_large:
        for name, params in pedigrees.LARGE_FIXTURES.items():
            fixtures.append((name, {"kind": "large", **params}, pedigrees.build_random(name, params), False))

    for name, params, fx, full_arrays in fixtures:
        t0 = time.perf_counter()
        arrays, summary = _capture(pg_mod, fx, full_arrays=full_arrays)
        summary["params"] = params
        summary["input_hash"] = pedigrees.input_hash(fx)
        summary["seconds"] = round(time.perf_counter() - t0, 2)
        if full_arrays:
            path = args.out / f"{name}.npz"
            np.savez_compressed(path, **arrays, **{f"input/{k}": v for k, v in fx.items()})
            summary["file"] = path.name
        manifest["fixtures"][name] = summary
        print(
            f"{name}: n={summary['n']} pairs={sum(v for k, v in summary['counts'].items() if '/' not in k and 'nnz' not in k)} {summary['seconds']}s"
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
