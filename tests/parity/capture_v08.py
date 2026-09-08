"""Reproduce the frozen 0.7.1 ``(arrays, summary)`` capture through the 0.8 API.

``generate_baseline.py`` is frozen at the 0.7.1 surface and only ever runs
against a 0.7.1 worktree; this module is what ``tests/test_parity_v071.py``
replays against the installed package.  It produces the same key layout as
``generate_baseline._capture`` so every count and SHA-256 in the manifest
still compares bit for bit where 0.8 preserves the value:

* ``pairs/<code>``: the matrix engine's degree-5 blocks *before* the ADR 0006
  closest-category precedence fold, read through the private
  ``MatrixPairExtractor`` exactly as ``relationship_pairs`` reads them.
  0.7.1 reported a pair under every category it satisfied, so the public
  ``relationship_pairs``, which keeps one category per pair, cannot reproduce
  the frozen collateral blocks on the random fixtures (its own contract is
  locked by ``tests/data/relationship_pairs_v0.8``).  The ten collateral
  asymmetric codes (``down > 0``) are folded back to the ``(min, max)`` row
  orientation 0.7.1 stored, lineal codes stay offspring-first as 0.7.1 had
  them, and every block is sorted by ``(first, second)``.
* ``pair_kinship/<code>``: ``pair_kinship(first, second)`` over those rows,
  widened from float32 to the float64 the frozen arrays hold.
* ``inbreeding``, ``n_ancestors`` (``distinct_ancestor_counts`` as int32),
  ``n_descendants`` (``descendant_path_counts`` cast to int32 after a bounds
  check; the fixture that overflowed int32 under 0.7.1 records
  ``n_descendants_overflow`` and no array), ``depth``.
* ``per_gen_mean_kinship``: ``mean_kinship_by_generation()`` scattered onto
  the dense ``max(depth) + 1`` layout with NaN in the gaps, which is what
  0.7.1 produced at threshold 0.
* ``approx/*``: ``approximate_kinship_matrix(min_propagated_kinship=0.001)``;
  ``complete/*``: ``kinship_matrix()``.
* ``subsample/*``: the same pre-precedence blocks projected through
  ``view(ids=...)`` into view rows, every code folded to ``(min, max)`` as
  0.7.1's ``from_subsample`` did.

Not reproduced: the manifest's ``streaming_counts``.  0.7.1's
``count_pairs_streaming`` reported unfolded raw counts; the 0.8
``estimate_relationship_counts`` subtracts the half-sib pairs a
parent-offspring category claims under the precedence fold, so the frozen
values have no 0.8 equivalent and the test does not compare them.

``generate_baseline`` imports nothing from ``pedigree_graph`` at module
import time, so its hashing helpers and constants are shared from there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pedigrees
from generate_baseline import APPROX_THRESHOLD, MAX_DEGREE, SUBSAMPLE_SEED, _sha, _sorted_pairs, _upper_coo

from pedigree_graph import RELATIONSHIPS, PedigreeGraph
from pedigree_graph._pair_extractor import MatrixPairExtractor, dependency_closure
from pedigree_graph._pair_utils import project_pairs
from pedigree_graph._registry import categories_up_to_degree

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["APPROX_THRESHOLD", "build", "capture"]

_FOLDED_CODES = frozenset(
    code for code, category in RELATIONSHIPS.items() if category.down > 0 and not category.symmetric
)


def build(fx: dict[str, np.ndarray]) -> PedigreeGraph:
    """Build the graph of one parity fixture, without generation labels."""
    return PedigreeGraph.from_arrays(
        ids=fx["ids"],
        mother_ids=fx["mother"],
        father_ids=fx["father"],
        twin_ids=fx["twin"],
        sex=fx["sex"],
    )


def _engine_pairs(graph: PedigreeGraph) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every degree-5 block as the engine emits it, before the precedence fold."""
    codes = dependency_closure(frozenset(category.code for category in categories_up_to_degree(MAX_DEGREE)))
    pairs = MatrixPairExtractor(graph, max_workers=1).extract(codes)
    graph._release_pair_matrices()
    return pairs


def _folded(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]], *, fold_all: bool
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{code: (first, second)}`` in the orientation 0.7.1 stored."""
    return {
        code: (np.minimum(first, second), np.maximum(first, second))
        if fold_all or code in _FOLDED_CODES
        else (first, second)
        for code, (first, second) in pairs.items()
    }


def _dense_mean_kinship(graph: PedigreeGraph) -> np.ndarray:
    summary = graph.mean_kinship_by_generation()
    theta = np.full(int(graph.depth.max()) + 1, np.nan, dtype=np.float64)
    theta[summary.generations] = summary.mean_kinship
    return theta


def _int32_descendant_counts(graph: PedigreeGraph) -> np.ndarray | None:
    counts = graph.descendant_path_counts()
    if counts.size and int(counts.max()) > np.iinfo(np.int32).max:
        return None
    return counts.astype(np.int32)


def capture(
    fx: dict[str, np.ndarray],
    *,
    full_arrays: bool,
    approximate_matrix: Callable[[PedigreeGraph], object] | None = None,
) -> tuple[dict, dict]:
    """Return ``(arrays, summary)`` in the frozen layout; ``arrays`` is empty when *full_arrays* is False.

    ``approximate_matrix`` lets the large differential test isolate frozen
    support parity; the separate 30k matrix integration test runs the complete
    public exact-value path.
    """
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n": len(fx["ids"]), "counts": {}, "hashes": {}}
    g = build(fx)

    pairs = _engine_pairs(g)
    for code, (raw_i, raw_j) in _folded(pairs, fold_all=False).items():
        si, sj, sk = _sorted_pairs(
            np.asarray(raw_i, dtype=np.int32),
            np.asarray(raw_j, dtype=np.int32),
            np.asarray(g.pair_kinship(raw_i, raw_j), dtype=np.float64),
        )
        summary["counts"][code] = len(si)
        summary["hashes"][f"pairs/{code}"] = _sha(si, sj)
        summary["hashes"][f"pair_kinship/{code}"] = _sha(sk)
        if full_arrays:
            arrays[f"pairs/{code}/first"] = si
            arrays[f"pairs/{code}/second"] = sj
            arrays[f"pair_kinship/{code}"] = sk

    F = np.asarray(g.inbreeding(), dtype=np.float64)
    n_anc = np.asarray(g.distinct_ancestor_counts(), dtype=np.int32)
    n_desc = _int32_descendant_counts(g)
    if n_desc is None:
        summary["n_descendants_overflow"] = True
    theta = _dense_mean_kinship(g)
    depth = np.asarray(g.depth, dtype=np.int32)
    vectors = [("inbreeding", F), ("n_ancestors", n_anc), ("per_gen_mean_kinship", theta), ("depth", depth)]
    if n_desc is not None:
        vectors.append(("n_descendants", n_desc))
    for name, arr in vectors:
        summary["hashes"][name] = _sha(arr)
        if full_arrays:
            arrays[name] = arr

    approximate = (
        g.approximate_kinship_matrix(min_propagated_kinship=APPROX_THRESHOLD)
        if approximate_matrix is None
        else approximate_matrix(g)
    )
    r, c, v = _upper_coo(approximate)
    summary["counts"]["approx_support_upper_nnz"] = len(r)
    summary["hashes"]["approx_support"] = _sha(r, c)
    summary["hashes"]["approx_values"] = _sha(v)
    if full_arrays:
        arrays["approx/row"], arrays["approx/col"], arrays["approx/val"] = r, c, v
        r0, c0, v0 = _upper_coo(g.kinship_matrix())
        summary["counts"]["complete_upper_nnz"] = len(r0)
        summary["hashes"]["complete_support"] = _sha(r0, c0)
        summary["hashes"]["complete_values"] = _sha(v0)
        arrays["complete/row"], arrays["complete/col"], arrays["complete/val"] = r0, c0, v0

    keep = pedigrees.subsample_selection(fx, SUBSAMPLE_SEED)
    graph_to_view = g.view(ids=fx["ids"][keep])._graph_to_view()
    sub_pairs = _folded(
        {code: project_pairs(first, second, graph_to_view) for code, (first, second) in pairs.items()},
        fold_all=True,
    )
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
