"""Hand-checked kinship matrices on small pedigrees: textbook values, MZ edges, inbreeding, pruning."""

from __future__ import annotations

import numpy as np
import polars as pl

from pedigree_graph import PedigreeGraph


def _graph(mothers, fathers, twins) -> PedigreeGraph:
    n = len(mothers)
    return PedigreeGraph.from_frame(
        {
            "id": np.arange(n),
            "mother": np.asarray(mothers, dtype=np.int64),
            "father": np.asarray(fathers, dtype=np.int64),
            "twin": np.asarray(twins, dtype=np.int64),
        }
    )


def _build(mothers, fathers, twins, min_kinship=0.0):
    graph = _graph(mothers, fathers, twins)
    if min_kinship == 0.0:
        return graph.kinship_matrix().toarray()
    return graph.approximate_kinship_matrix(min_propagated_kinship=min_kinship).toarray()


def test_founders_only():
    K = _build([-1, -1], [-1, -1], [-1, -1])
    assert np.array_equal(K, np.array([[0.5, 0.0], [0.0, 0.5]]))


def test_parent_offspring_fullsib_grandparent():
    # 0, 1 founders; 2 = child(0,1); 3 = child(2, -1) — grandchild of 0 and 1.
    K = _build(mothers=[-1, -1, 0, 2], fathers=[-1, -1, 1, -1], twins=[-1, -1, -1, -1])
    assert np.allclose(np.diag(K), [0.5, 0.5, 0.5, 0.5])
    assert K[0, 2] == 0.25
    assert K[1, 2] == 0.25
    assert K[2, 3] == 0.25
    assert K[0, 3] == 0.125
    assert K[1, 3] == 0.125


def test_mz_twins_noninbred():
    # 0, 1 founders; 2, 3 MZ twin children of 0 and 1.
    K = _build(mothers=[-1, -1, 0, 0], fathers=[-1, -1, 1, 1], twins=[-1, -1, 3, 2])
    assert K[2, 3] == 0.5
    assert K[2, 2] == 0.5
    assert K[3, 3] == 0.5
    assert K[0, 2] == K[0, 3] == 0.25
    assert K[1, 2] == K[1, 3] == 0.25


def test_founder_mz_twins():
    # Regression, rwaples/pedigree-graph#5: 0, 1 are MZ co-twins *and*
    # founders.  The MZ twin pass used to run only inside the depth >= 1
    # loop, so a depth-0 twin pair never got its edge and came back 0.0.
    K = _build(mothers=[-1, -1, 0], fathers=[-1, -1, 1], twins=[1, 0, -1])
    assert K[0, 1] == 0.5
    assert K[1, 0] == 0.5
    # 2's parents are MZ co-twins: kinship(mother, father) = 0.5, so
    # self-kinship = (1 + 0.5) / 2.
    assert K[2, 2] == 0.75


def test_founder_mz_twins_propagate_to_descendants():
    # Founders 0/1 are MZ co-twins mating with unrelated 2/3; their children
    # 4 and 5 are genetically half-sibs, phi = 0.125.  A zero at depth 0
    # would propagate to everything below the pair.
    K = _build(
        mothers=[-1, -1, -1, -1, 2, 3],
        fathers=[-1, -1, -1, -1, 0, 1],
        twins=[1, 0, -1, -1, -1, -1],
    )
    assert K[0, 1] == 0.5
    assert K[4, 5] == 0.125
    assert K[5, 4] == 0.125


def test_founder_mz_twins_match_pairwise_kinship():
    # kinship_matrix() and pair_kinship() disagreed on exactly these pairs;
    # #5 was filed on that disagreement.
    ped = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "mother": [-1, -1, -1, -1, 2, 3],
            "father": [-1, -1, -1, -1, 0, 1],
            "twin": [1, 0, -1, -1, -1, -1],
            "sex": [0, 0, 1, 1, 0, 0],
            "generation": [0, 0, 0, 0, 1, 1],
        }
    )
    rows, cols = np.triu_indices(6)
    graph = PedigreeGraph.from_frame(ped)
    exact = graph.pair_kinship(rows.astype(np.int64), cols.astype(np.int64))
    K = graph.kinship_matrix().toarray()
    np.testing.assert_allclose(K[rows, cols], exact)


def test_inbred_mz_regression():
    # G0: 0, 1 founders; G1: 2, 3 full-sibs child(0,1);
    # G2: 4, 5 MZ twins child(2, 3), inbred with F = 0.25.
    K = _build(
        mothers=[-1, -1, 0, 0, 2, 2],
        fathers=[-1, -1, 1, 1, 3, 3],
        twins=[-1, -1, -1, -1, 5, 4],
    )
    F = 2 * np.diag(K) - 1
    assert F[4] == 0.25
    assert F[5] == 0.25
    assert K[4, 5] == 0.625
    assert K[4, 4] == 0.625
    assert K[5, 5] == 0.625
    # 0.5 * (K[parent, parent] + K[parent, other_parent]) = 0.5 * (0.5 + 0.25)
    assert K[2, 4] == 0.375
    assert K[3, 5] == 0.375


def test_symmetric_and_sorted():
    graph = _graph(
        mothers=[-1, -1, 0, 0, 2, 2],
        fathers=[-1, -1, 1, 1, 3, 3],
        twins=[-1, -1, -1, -1, 5, 4],
    )
    matrix = graph.kinship_matrix()
    assert matrix.has_sorted_indices
    K = matrix.toarray()
    assert np.allclose(K, K.T)


def test_min_kinship_prunes_offdiag():
    # 3-generation lineage; kinships 0.25, 0.125, 0.0625.  A propagation
    # threshold of 0.1 drops the great-grandparent pair and keeps the rest.
    lineage = {"mothers": [-1, 0, 1, 2], "fathers": [-1, -1, -1, -1], "twins": [-1, -1, -1, -1]}
    K_full = _build(**lineage)
    K_pruned = _build(**lineage, min_kinship=0.1)
    assert K_full[0, 3] == 0.0625
    assert K_pruned[0, 3] == 0.0
    assert K_pruned[0, 1] == 0.25
    assert K_pruned[0, 2] == 0.125


def test_dp_reorders_topological_but_not_depth_major_rows():
    # Rows 0, 1 founders; row 2 = child(0, 1); row 3 a founder that sorts
    # after its own child by input row order.  The DP runs in depth-major
    # rows and hands the matrix back in caller rows.
    depth_major = _build([-1, -1, 0, -1], [-1, -1, 1, -1], [-1, -1, -1, -1])
    permuted = _build([-1, 2, -1, -1], [-1, 3, -1, -1], [-1, -1, -1, -1])
    assert depth_major[0, 2] == permuted[2, 1]
    assert depth_major[2, 2] == permuted[1, 1]
