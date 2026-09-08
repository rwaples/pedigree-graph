"""Every public operation is invariant under acyclic input row order (slice 1b).

The reference graph is built from a fixture in its given (topological) order.
Each permutation reorders every column of the same table; ``perm[k]`` is the
reference row that ends up at permuted position k.  Results are keyed by
pedigree ID rather than row, so the comparison needs no mapping and a stray
row/ID confusion cannot pass.

Integer, category and pair results must be exactly invariant.  Float kinship
may re-round: ADR 0009 pins the peel rule to depth then row, so a permutation
can change the evaluation order of a deep inbred pair.  Those comparisons use
the ADR's cross-order envelope and report ULP distance alongside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import parity_columns, parity_fixtures
from relationship_predicates import AncestorWalk

from pedigree_graph import RELATIONSHIPS, MissingMetadataError, PedigreeGraph
from pedigree_graph.effective_size import eligible_cohort_range, estimate_effective_sizes
from pedigree_graph.experimental import count_pairs_bfs

if TYPE_CHECKING:
    from pedigree_graph import PedigreeView
    from pedigree_graph.summaries import GenerationKinshipSummary

MAX_DEGREE = 5


# lineal_five_generations, random_1k and every motif with a skip-generation
# edge are already topological-but-not-depth-major, so the permuted routing
# runs even for the reference graph.
FIXTURES = parity_fixtures("random_1k", "deep_inbred_60g")
FIXTURE_NAMES = sorted(FIXTURES)


def _reference_depth(fixture: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(PedigreeGraph.from_frame(parity_columns(fixture)).depth, dtype=np.int64)


def _dated_columns(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    depth = _reference_depth(fixture)
    return parity_columns(fixture, 25 * depth + (np.arange(len(depth)) % 3))


def _permutations(depth: np.ndarray) -> dict[str, np.ndarray]:
    n = len(depth)
    return {
        "reversed": np.arange(n)[::-1].copy(),
        "shuffle_11": np.random.default_rng(11).permutation(n),
        "shuffle_12": np.random.default_rng(12).permutation(n),
        "children_first": np.argsort(-depth, kind="stable"),
    }


def _permute(columns: dict[str, np.ndarray], perm: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[perm] for key, value in columns.items()}


def _build(columns: dict[str, np.ndarray], constructor: str) -> PedigreeGraph:
    if constructor == "dict":
        return PedigreeGraph.from_frame(columns)
    return PedigreeGraph.from_arrays(
        ids=columns["id"],
        mother_ids=columns["mother"],
        father_ids=columns["father"],
        twin_ids=columns["twin"],
        sex=columns["sex"],
        birth_year=columns.get("birth_year"),
    )


def _unordered_pair_sets(pairs: dict[str, tuple[np.ndarray, np.ndarray]], ids: np.ndarray) -> dict[str, set]:
    """``{code: unordered ID-pair set}`` for symmetric pair arrays in graph rows."""
    return {
        code: {
            frozenset(pair)
            for pair in zip(
                ids[np.asarray(first, dtype=np.intp)].tolist(),
                ids[np.asarray(second, dtype=np.intp)].tolist(),
                strict=True,
            )
        }
        for code, (first, second) in pairs.items()
    }


def _relationship_pair_sets(
    receiver: PedigreeGraph | PedigreeView,
    ids: np.ndarray,
    *,
    walk: AncestorWalk | None = None,
    rows: np.ndarray | None = None,
) -> dict:
    """``{code: (requested, unordered ID-pair set, oriented ID-pair set, dual-valid unordered set)}``.

    Symmetric blocks and dual-valid asymmetric pairs are oriented by graph row
    (ADR 0006 pair contracts 2 and 5), so their orientation legitimately
    follows the permutation; every other asymmetric pair is compared exactly.
    For a view, *walk* is the graph's and *rows* maps view rows to graph rows
    so dual validity is still decided on the pedigree.
    """
    if walk is None:
        assert isinstance(receiver, PedigreeGraph)
        walk = AncestorWalk(receiver)
    if rows is None:
        rows = np.arange(len(ids))
    sets = {}
    for code, block in receiver.relationship_pairs(max_degree=MAX_DEGREE).items():
        pairs = list(zip(block.first_rows.tolist(), block.second_rows.tolist(), strict=True))
        oriented = {(int(ids[a]), int(ids[b])) for a, b in pairs}
        dual = set()
        if not block.category.symmetric:
            dual = {
                frozenset((int(ids[a]), int(ids[b])))
                for a, b in pairs
                if walk.dual_valid(code, int(rows[a]), int(rows[b]))
            }
        sets[code] = (block.requested, {frozenset(pair) for pair in oriented}, oriented, dual)
    return sets


def _assert_relationship_pairs_match(expected: dict, actual: dict, label: str) -> None:
    assert list(actual) == list(expected), f"{label}: code order changed"
    for code, (requested, unordered, oriented, dual) in expected.items():
        got_requested, got_unordered, got_oriented, got_dual = actual[code]
        assert got_requested == requested, f"{label}/{code}: requested flag changed"
        assert got_unordered == unordered, f"{label}/{code}: unordered ID-pair set changed"
        assert got_dual == dual, f"{label}/{code}: dual-valid set changed"
        if RELATIONSHIPS[code].symmetric:
            continue
        fixed = {pair for pair in oriented if frozenset(pair) not in dual}
        got_fixed = {pair for pair in got_oriented if frozenset(pair) not in dual}
        assert got_fixed == fixed, f"{label}/{code}: oriented ID-pair set changed"


def _id_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _matrix_by_id(matrix, ids: np.ndarray) -> dict[tuple[int, int], float]:
    K = matrix.tocoo()
    keep = K.row <= K.col
    rows, cols, values = ids[K.row[keep]].tolist(), ids[K.col[keep]].tolist(), K.data[keep].tolist()
    return {_id_key(a, b): value for a, b, value in zip(rows, cols, values, strict=True)}


def _float32_ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer-view ULP distance between two nonnegative float32 arrays."""
    ia = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def _assert_within_envelope(expected: dict, actual: dict, depth_by_id: dict, label: str) -> int:
    """Check the ADR 0009 cross-order envelope; return the worst ULP distance."""
    assert set(actual) == set(expected), f"{label}: kinship support changed"
    if not expected:
        return 0
    keys = list(expected)
    want = np.array([expected[key] for key in keys], dtype=np.float32)
    got = np.array([actual[key] for key in keys], dtype=np.float32)
    tolerance = np.array(
        [2.0 * (depth_by_id[a] + depth_by_id[b] + 1) * 2.0**-25 for a, b in keys],
        dtype=np.float64,
    )
    deviation = np.abs(want.astype(np.float64) - got.astype(np.float64))
    worst = int(np.argmax(deviation - tolerance))
    assert np.all(deviation <= tolerance), (
        f"{label}: outside the ADR 0009 envelope at {keys[worst]}: "
        f"{want[worst]!r} vs {got[worst]!r}, tolerance {tolerance[worst]:.3e}"
    )
    return int(_float32_ulp_distance(want, got).max())


def _assert_theta_matches(expected: GenerationKinshipSummary, actual: GenerationKinshipSummary, label: str) -> None:
    """θ̄_g is a float64 mean of float32 kinship, so it inherits the ADR 0009
    cross-order envelope of its terms.  Cohort g here is structural depth (the
    fixtures carry no supplied labels), so every within-cohort pair has both
    endpoints at depth g and the per-cohort bound is ``2 * (2g + 1) * 2**-25``.
    """
    np.testing.assert_array_equal(actual.generations, expected.generations, err_msg=f"{label}: cohorts changed")
    np.testing.assert_array_equal(actual.pair_counts, expected.pair_counts, err_msg=f"{label}: pair counts changed")
    assert actual.unlabelled_individual_count == expected.unlabelled_individual_count, f"{label}: unlabelled changed"
    np.testing.assert_array_equal(
        np.isnan(actual.mean_kinship), np.isnan(expected.mean_kinship), err_msg=f"{label}: NaN cohorts moved"
    )
    cohorts = expected.generations.astype(np.float64)
    tolerance = np.maximum(1e-9, 2.0 * (2.0 * cohorts + 1.0) * 2.0**-25)
    known = ~np.isnan(expected.mean_kinship)
    deviation = np.abs(actual.mean_kinship[known] - expected.mean_kinship[known])
    assert np.all(deviation <= tolerance[known]), f"{label}: max deviation {deviation.max():.3e}"


APPROX_THRESHOLD = 0.001
# Fraction of the propagation-pruned support allowed to move under a
# permutation; see _assert_approximate_matrix.  0.055% is the worst observed
# on this corpus. Retained values are independently corrected after selection.
APPROX_SUPPORT_DRIFT = 0.005

NE_ATOL = 1e-12
# Six of the eight estimators are exactly invariant under permutation.  Two are
# not, for reasons that are properties of the estimator rather than of the
# routing, and each gets the tolerance its measured cause needs:
#   ne_caballero_toro / ne_inbreeding reduce per-individual float64 quantities in
#     row order (the founder-set sweep, the per-cohort mean of F), so a
#     permutation moves the last bits; worst observed 2.0e-12.
#   ne_coancestry is derived from theta-bar, a float64 mean of float32 kinship
#     that carries the ADR 0009 cross-order envelope, and then amplified through
#     ``1 / (2 * delta)``; worst observed 6.3e-6 on deep_inbred_60g.
# A routing bug moves all of these by percent, well outside either tolerance.
NE_RTOL = 1e-12
NE_RTOL_BY_ESTIMATOR = {
    "ne_caballero_toro": 1e-10,
    "ne_inbreeding": 1e-10,
    "ne_coancestry": 1e-4,
}


def _assert_approximate_matrix(reference: _Snapshot, actual: _Snapshot, label: str) -> float:
    """Compare corrected values on propagation-pruned support across row orders.

    The support remains approximate: pruning an intermediate at
    ``val <= min_propagated_kinship`` can admit or omit a pair relative to a
    final pedigree-expected-value cutoff, and the row tie-break can move that
    support under permutation.  Values are no longer propagated
    approximations.  Every retained value must equal the same graph's complete
    matrix bit, and values on support shared by two row orders obey ADR 0009's
    ordinary cross-order recurrence envelope.
    """
    for snapshot, which in ((reference, "reference"), (actual, "permuted")):
        complete = snapshot.complete_kinship
        assert set(snapshot.approx_kinship) <= set(complete), f"{label}: {which} support is not a subset"
        differing = [
            key
            for key, value in snapshot.approx_kinship.items()
            if np.float32(value).tobytes() != np.float32(complete[key]).tobytes()
        ]
        assert not differing, f"{label}: {which} retained values are not pedigree-expected at {differing[:3]}"

    moved = set(reference.approx_kinship) ^ set(actual.approx_kinship)
    assert len(moved) <= APPROX_SUPPORT_DRIFT * len(reference.approx_kinship), (
        f"{label}: {len(moved)} of {len(reference.approx_kinship)} support pairs moved"
    )
    shared = set(reference.approx_kinship) & set(actual.approx_kinship)
    if not shared:
        return 0.0
    keys = list(shared)
    want = np.array([reference.approx_kinship[key] for key in keys], dtype=np.float32)
    got = np.array([actual.approx_kinship[key] for key in keys], dtype=np.float32)
    depth = reference.depth_by_id
    tolerance = np.array(
        [2.0 * (depth[first] + depth[second] + 1) * 2.0**-25 for first, second in keys],
        dtype=np.float64,
    )
    deviation = np.abs(want.astype(np.float64) - got.astype(np.float64))
    assert np.all(deviation <= tolerance), f"{label}: retained values exceed the ADR 0009 envelope"
    return float(deviation.max())


def _numeric_equal(expected, actual, path: str, rtol: float = NE_RTOL) -> None:
    if expected is None or actual is None:
        assert expected is None, f"{path}: expected {expected!r}, got None"
        assert actual is None, f"{path}: expected None, got {actual!r}"
    elif isinstance(expected, dict):
        assert sorted(actual) == sorted(expected), f"{path}: keys changed"
        for key in expected:
            _numeric_equal(expected[key], actual[key], f"{path}.{key}", rtol)
    elif isinstance(expected, bool | int | str | tuple):
        assert actual == expected, f"{path}: {expected!r} vs {actual!r}"
    else:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=rtol,
            atol=NE_ATOL,
            equal_nan=True,
            err_msg=path,
        )


# The row a refusal reports first depends on the row order by construction.
_ORDER_DEPENDENT_FIELDS = frozenset(
    {"first_row", "first_id", "represented_parent_role", "unrepresented_parent_role", "unrepresented_parent_status"}
)


def _value_or_refusal(compute):
    try:
        return compute()
    except MissingMetadataError as exc:
        return (exc.code, {k: v for k, v in exc.fields.items() if k not in _ORDER_DEPENDENT_FIELDS})


def _effective_sizes(graph: PedigreeGraph) -> dict:
    """Every estimator's serialized result, refusals stripped of their order-dependent fields."""
    results = estimate_effective_sizes(graph).to_dict()
    for record in results.values():
        if "fields" in record:
            record["fields"] = {k: v for k, v in record["fields"].items() if k not in _ORDER_DEPENDENT_FIELDS}
    return results


def _assert_all_ne_matches(expected: dict, actual: dict, label: str) -> None:
    assert sorted(actual) == sorted(expected), f"{label}: estimator set changed"
    for estimator, fields in expected.items():
        _numeric_equal(
            fields,
            actual[estimator],
            f"{label}.{estimator}",
            NE_RTOL_BY_ESTIMATOR.get(estimator, NE_RTOL),
        )


class _Snapshot:
    """Every operation reachable at slice 1b, keyed so row order cannot show."""

    def __init__(self, columns: dict[str, np.ndarray], constructor: str, subsample_ids: np.ndarray):
        ids = np.asarray(columns["id"])
        graph = _build(columns, constructor)
        id_list = ids.tolist()
        self.depth_by_id = dict(zip(id_list, np.asarray(graph.depth).tolist(), strict=True))

        self.by_id = {
            "depth": dict(self.depth_by_id),
            "distinct_ancestor_counts": dict(zip(id_list, graph.distinct_ancestor_counts().tolist(), strict=True)),
            "descendant_path_counts": dict(zip(id_list, graph.descendant_path_counts().tolist(), strict=True)),
            "connected_component_ids": dict(zip(id_list, graph.connected_component_ids().tolist(), strict=True)),
            "inbreeding": dict(zip(id_list, np.asarray(graph.inbreeding()).tolist(), strict=True)),
        }

        pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
        kinship = graph.pair_kinship(pairs)
        self.pair_kinship = {
            _id_key(int(a), int(b)): float(value)
            for code, (first, second) in pairs.items()
            for a, b, value in zip(ids[first], ids[second], kinship[code], strict=True)
        }
        full_sib, maternal, paternal = graph._sibling_pairs()
        self.sibling_pairs = _unordered_pair_sets({"FS": full_sib, "MHS": maternal, "PHS": paternal}, ids)
        self.relationship_pairs = _relationship_pair_sets(graph, ids)

        self.relationship_counts = dict(graph.relationship_counts(max_degree=MAX_DEGREE))
        self.estimated_counts = dict(graph.estimate_relationship_counts(max_degree=MAX_DEGREE))
        self.count_pairs_bfs = count_pairs_bfs(graph, max_degree=MAX_DEGREE)

        self.complete_kinship = _matrix_by_id(graph.kinship_matrix(), ids)
        self.approx_kinship = _matrix_by_id(
            graph.approximate_kinship_matrix(min_propagated_kinship=APPROX_THRESHOLD), ids
        )
        self.theta = graph.mean_kinship_by_generation()

        # A fixture without a parent-child edge to date is refused by the
        # age-based helpers in every row order; compare the refusal itself.
        self.generation_interval = _value_or_refusal(lambda: graph.generation_interval)
        self.cohort_range = _value_or_refusal(lambda: eligible_cohort_range(graph))
        self.all_ne = _effective_sizes(graph)

        view = graph.view(ids=subsample_ids)
        self.view_relationship_pairs = _relationship_pair_sets(
            view, view.ids, walk=AncestorWalk(graph), rows=view.graph_rows
        )


@pytest.mark.parametrize("constructor", ["dict", "from_arrays"])
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_operation_is_invariant_under_row_order(name, constructor, capsys):
    fixture = FIXTURES[name]
    columns = _dated_columns(fixture)
    subsample_ids = np.asarray(fixture["ids"])[::2]
    reference = _Snapshot(columns, constructor, subsample_ids)
    worst = {"ulp_matrix_0.0": 0, "ulp_pair_kinship": 0, "maxdev_matrix_0.001": 0.0}

    for label, perm in _permutations(_reference_depth(fixture)).items():
        actual = _Snapshot(_permute(columns, perm), constructor, subsample_ids)
        where = f"{name}/{constructor}/{label}"

        for field, expected in reference.by_id.items():
            assert actual.by_id[field] == expected, f"{where}: {field} changed"

        assert actual.sibling_pairs == reference.sibling_pairs, f"{where}: sibling pair sets changed"
        _assert_relationship_pairs_match(
            reference.relationship_pairs, actual.relationship_pairs, f"{where}/relationship_pairs"
        )
        _assert_relationship_pairs_match(
            reference.view_relationship_pairs, actual.view_relationship_pairs, f"{where}/view.relationship_pairs"
        )

        assert actual.relationship_counts == reference.relationship_counts, f"{where}: relationship_counts changed"
        assert actual.estimated_counts == reference.estimated_counts, f"{where}: estimated counts changed"
        assert actual.count_pairs_bfs == reference.count_pairs_bfs, f"{where}: count_pairs_bfs changed"

        depths = reference.depth_by_id
        worst["ulp_matrix_0.0"] = max(
            worst["ulp_matrix_0.0"],
            _assert_within_envelope(reference.complete_kinship, actual.complete_kinship, depths, f"{where}/K(0.0)"),
        )
        worst["maxdev_matrix_0.001"] = max(
            worst["maxdev_matrix_0.001"],
            _assert_approximate_matrix(reference, actual, f"{where}/K({APPROX_THRESHOLD})"),
        )
        worst["ulp_pair_kinship"] = max(
            worst["ulp_pair_kinship"],
            _assert_within_envelope(reference.pair_kinship, actual.pair_kinship, depths, f"{where}/pair_kinship"),
        )

        _assert_theta_matches(reference.theta, actual.theta, f"{where}/mean_kinship_by_generation")
        assert actual.generation_interval == reference.generation_interval, f"{where}: generation_interval changed"
        assert actual.cohort_range == reference.cohort_range, f"{where}: eligible_cohort_range changed"
        _assert_all_ne_matches(reference.all_ne, actual.all_ne, f"{where}/estimate_effective_sizes")

    with capsys.disabled():
        print(f"cross-order drift {name}/{constructor}: {worst}")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_pair_kinship_is_bit_identical_to_the_matrix_under_permutation(name):
    # ADR 0009: pair/matrix bit parity is a within-graph property and holds in
    # every row order, deep inbred pedigrees included.
    fixture = FIXTURES[name]
    columns = _dated_columns(fixture)
    for label, perm in _permutations(_reference_depth(fixture)).items():
        graph = _build(_permute(columns, perm), "dict")
        pairs = graph.relationship_pairs(max_degree=MAX_DEGREE)
        kinship = graph.pair_kinship(pairs)
        K = graph.kinship_matrix()
        for code, values in kinship.items():
            if len(values) == 0:
                continue
            first, second = pairs[code]
            matrix = np.asarray(K[first, second], dtype=np.float32).ravel()
            assert values.tobytes() == matrix.tobytes(), f"{name}/{label}/{code}: pair vs matrix bits differ"


@pytest.mark.parametrize("name", ["random_1k", "deep_inbred_60g"])
def test_mean_kinship_by_generation_groups_by_the_supplied_label(name):
    """Cohorts follow the label even when a cohort spans two structural depths.

    ``depth // 2`` merges each pair of adjacent depths into one cohort, so the
    within-cohort pairs the retiring DP accumulates inline no longer share a
    depth.  The cached-matrix branch groups the same labels by walking the
    kinship COO instead, which makes it an independent oracle.
    """
    fixture = FIXTURES[name]
    labels = _reference_depth(fixture) // 2
    columns = {**parity_columns(fixture), "generation": labels}

    streamed = PedigreeGraph.from_frame(columns)
    from_matrix = PedigreeGraph.from_frame(columns)
    from_matrix.kinship_matrix()

    theta = streamed.mean_kinship_by_generation()
    walked = from_matrix.mean_kinship_by_generation()
    np.testing.assert_array_equal(theta.generations, np.unique(labels))
    np.testing.assert_array_equal(walked.generations, theta.generations)
    np.testing.assert_array_equal(walked.pair_counts, theta.pair_counts)
    np.testing.assert_allclose(theta.mean_kinship, walked.mean_kinship, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("labelling", ["zeros", "shuffled", "offset"])
@pytest.mark.parametrize("name", ["random_1k", "deep_inbred_60g"])
def test_generation_labels_do_not_drive_structure(name, labelling):
    fixture = FIXTURES[name]
    depth = _reference_depth(fixture)
    columns = parity_columns(fixture)
    labels = {
        "zeros": np.zeros(len(depth), dtype=np.int64),
        "shuffled": np.random.default_rng(99).permutation(depth),
        "offset": depth + 7,
    }[labelling]

    unlabelled = PedigreeGraph.from_frame(columns)
    labelled = PedigreeGraph.from_frame({**columns, "generation": labels})

    assert labelled.generation_labels.tolist() == labels.tolist()
    np.testing.assert_array_equal(labelled.depth, unlabelled.depth)
    np.testing.assert_array_equal(labelled.inbreeding(), unlabelled.inbreeding())
    np.testing.assert_array_equal(labelled.descendant_path_counts(), unlabelled.descendant_path_counts())

    expected_pairs = unlabelled.relationship_pairs(max_degree=MAX_DEGREE)
    actual_pairs = labelled.relationship_pairs(max_degree=MAX_DEGREE)
    assert list(actual_pairs) == list(expected_pairs)
    for code, (first, second) in expected_pairs.items():
        np.testing.assert_array_equal(actual_pairs[code].first_rows, first)
        np.testing.assert_array_equal(actual_pairs[code].second_rows, second)

    expected_kinship = unlabelled.pair_kinship(expected_pairs)
    actual_kinship = labelled.pair_kinship(actual_pairs)
    for code, values in expected_kinship.items():
        assert actual_kinship[code].tobytes() == values.tobytes(), f"{code}: pair kinship moved with the label"

    expected_k, actual_k = unlabelled.kinship_matrix(), labelled.kinship_matrix()
    np.testing.assert_array_equal(expected_k.indptr, actual_k.indptr)
    np.testing.assert_array_equal(expected_k.indices, actual_k.indices)
    assert actual_k.data.tobytes() == expected_k.data.tobytes()
