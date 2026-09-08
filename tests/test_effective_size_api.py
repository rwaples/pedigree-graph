"""API-surface contract tests for the eight final Ne estimators.

Covers the shape of every result record, the read-only ownership of its
arrays, label rebasing / sparsity / depth-merging, the represented-founder
rules, and the structured allocation guards.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields, replace

import numpy as np
import polars as pl
import pytest

from pedigree_graph import PedigreeGraph, ResourceError, effective_size
from pedigree_graph import _ne_common as ne_common
from pedigree_graph._cohorts import _densify_labels as _cohorts_densify_labels
from pedigree_graph._kinship_kernel import _densify_labels as _kernel_densify_labels
from pedigree_graph._ne_common import (
    _checked_founder_matrix,
    _scalar_ne_from_log_regression,
    _transition_ne,
)
from pedigree_graph._ne_founders import _founder_columns, _founder_idx, _per_gen_founder_means
from pedigree_graph.effective_size import NeInbreedingResult, estimate_effective_sizes


def _df(records: list[dict]) -> pl.DataFrame:
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
    return pl.DataFrame(rows)


def _closed_line(n_gens: int) -> pl.DataFrame:
    records = [
        {"id": 0, "sex": 1, "generation": 0},
        {"id": 1, "sex": 0, "generation": 0},
    ]
    next_id = 2
    prev_m, prev_f = 0, 1
    for g in range(1, n_gens + 1):
        m = next_id
        records.append({"id": m, "sex": 1, "generation": g, "mother": prev_f, "father": prev_m})
        f = next_id + 1
        records.append({"id": f, "sex": 0, "generation": g, "mother": prev_f, "father": prev_m})
        prev_m, prev_f = m, f
        next_id += 2
    return _df(records)


def _relabelled(df: pl.DataFrame, mapping: dict[int, int]) -> pl.DataFrame:
    return df.with_columns(pl.col("generation").replace_strict(mapping).alias("generation"))


def _mz_founder_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 0, "generation": 0, "twin": 1},
                {"id": 1, "sex": 0, "generation": 0, "twin": 0},
                {"id": 2, "sex": 1, "generation": 0},
                {"id": 3, "sex": 1, "generation": 1, "mother": 0, "father": 2},
                {"id": 4, "sex": 0, "generation": 1, "mother": 1, "father": 2},
                {"id": 5, "sex": 1, "generation": 2, "mother": 4, "father": 3},
                {"id": 6, "sex": 0, "generation": 2, "mother": 4, "father": 3},
            ]
        )
    )


def _single_founder_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 0, "generation": 0},
                {"id": 1, "sex": 1, "generation": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 0, "father": 1},
                {"id": 3, "sex": 0, "generation": 1, "mother": 0, "father": 1},
                {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
                {"id": 5, "sex": 0, "generation": 2, "mother": 3, "father": 2},
            ]
        )
    )


def _late_founder_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 1, "generation": 0},
                {"id": 1, "sex": 0, "generation": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
                {"id": 3, "sex": 0, "generation": 1},
                {"id": 4, "sex": 1, "generation": 2, "mother": 3, "father": 2},
            ]
        )
    )


def _off_label_founder_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 1, "generation": 0},
                {"id": 1, "sex": 0, "generation": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0},
                {"id": 3, "sex": 0, "generation": 3},
            ]
        )
    )


def _external_parent_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_arrays(
        ids=[10, 11, 12],
        mother_ids=[99, -1, 10],
        father_ids=[-1, -1, 11],
        sex=[0, 1, 0],
        generation=[0, 0, 1],
    )


def _two_cohort_reproduction_pedigree() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 1, "generation": 0},
                {"id": 1, "sex": 1, "generation": 0},
                {"id": 2, "sex": 0, "generation": 0},
                {"id": 3, "sex": 0, "generation": 0},
                {"id": 4, "sex": 1, "generation": 1, "father": 0, "mother": 2},
                {"id": 5, "sex": 1, "generation": 1, "father": 1, "mother": 3},
                {"id": 6, "sex": 0, "generation": 1, "father": 0, "mother": 2},
                {"id": 7, "sex": 0, "generation": 1, "father": 1, "mother": 3},
                {"id": 8, "sex": 1, "generation": 1, "father": 4, "mother": 6},
                {"id": 9, "sex": 0, "generation": 1, "father": 5, "mother": 7},
            ]
        )
    )


def _birth_year_pedigree() -> PedigreeGraph:
    rows = [
        {"id": 0, "sex": 1, "generation": 0, "birth_year": 1900},
        {"id": 1, "sex": 1, "generation": 0, "birth_year": 1900},
        {"id": 2, "sex": 0, "generation": 0, "birth_year": 1900},
        {"id": 3, "sex": 0, "generation": 0, "birth_year": 1900},
        {"id": 4, "sex": 1, "generation": 1, "birth_year": 1910, "father": 0, "mother": 2},
        {"id": 5, "sex": 1, "generation": 1, "birth_year": 1910, "father": 1, "mother": 3},
        {"id": 6, "sex": 0, "generation": 1, "birth_year": 1910, "father": 0, "mother": 2},
        {"id": 7, "sex": 0, "generation": 1, "birth_year": 1910, "father": 1, "mother": 3},
    ]
    return PedigreeGraph.from_frame(_df(rows).with_columns(pl.Series("birth_year", [r["birth_year"] for r in rows])))


@dataclass(frozen=True)
class _Estimator:
    name: str
    call: object
    labels: str | None
    transitions: frozenset


_RATE_TRANSITIONS = frozenset({"transition_from", "transition_to", "ne_per_gen"})

ESTIMATORS = (
    _Estimator("ne_inbreeding", effective_size.ne_inbreeding, "generations", _RATE_TRANSITIONS),
    _Estimator("ne_coancestry", effective_size.ne_coancestry, "generations", _RATE_TRANSITIONS),
    _Estimator("ne_variance_family_size", effective_size.ne_variance_family_size, "parent_generations", frozenset()),
    _Estimator("ne_sex_ratio", effective_size.ne_sex_ratio, "generations", frozenset()),
    _Estimator("ne_individual_delta_f", effective_size.ne_individual_delta_f, "generations", frozenset()),
    _Estimator("ne_long_term_contributions", effective_size.ne_long_term_contributions, None, frozenset()),
    _Estimator("ne_hill_overlapping", effective_size.ne_hill_overlapping, None, frozenset()),
    _Estimator("ne_caballero_toro", effective_size.ne_caballero_toro, "generations", _RATE_TRANSITIONS),
)

ESTIMATOR_NAMES = frozenset(e.name for e in ESTIMATORS)
LABELLED = tuple(e for e in ESTIMATORS if e.labels is not None)
RATE_BASED = tuple(e for e in ESTIMATORS if e.transitions)


def _over(estimators: tuple[_Estimator, ...]):
    return pytest.mark.parametrize("est", estimators, ids=[e.name for e in estimators])


def _array_fields(result: object) -> dict[str, np.ndarray]:
    found = {}
    for f in fields(result):
        value = getattr(result, f.name)
        if isinstance(value, np.ndarray):
            found[f.name] = value
    return found


def _assert_owned_read_only(result: object) -> None:
    for name, value in _array_fields(result).items():
        assert not value.flags.writeable, name
        assert value.flags.c_contiguous, name
        assert value.flags.owndata, name
        with pytest.raises(ValueError, match="read-only"):
            value[...] = 0


def _assert_plain_python(value: object, where: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        assert not math.isnan(value), f"{where} is NaN"
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _assert_plain_python(item, f"{where}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, f"{where} key {key!r}"
            _assert_plain_python(item, f"{where}.{key}")
        return
    raise AssertionError(f"{where} is {type(value).__name__}, not plain Python")


@pytest.fixture(scope="module")
def empty_graph() -> PedigreeGraph:
    return PedigreeGraph.from_arrays(ids=[], mother_ids=[], father_ids=[], sex=[])


@pytest.fixture(scope="module")
def line_graph() -> PedigreeGraph:
    return PedigreeGraph.from_frame(_closed_line(4))


@pytest.fixture(scope="module")
def sparse_line_graph() -> PedigreeGraph:
    return PedigreeGraph.from_frame(_relabelled(_closed_line(3), {0: 0, 1: 0, 2: 2, 3: 5}))


def test_empty_graph_constructs(empty_graph):
    assert empty_graph.n_individuals == 0
    assert empty_graph.generation_labels is None


@_over(ESTIMATORS)
def test_empty_graph_yields_no_estimate(empty_graph, est):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = est.call(empty_graph)
    assert [str(w.message) for w in caught] == []
    assert result.ne is None
    for name, value in _array_fields(result).items():
        assert value.shape == (0,), name


def test_empty_graph_ltc_has_no_final_generation(empty_graph):
    result = effective_size.ne_long_term_contributions(empty_graph)
    assert result.final_generation is None
    assert result.sum_c_squared == 0.0


def test_empty_graph_hill_collapses_to_ne_v(empty_graph):
    result = effective_size.ne_hill_overlapping(empty_graph)
    assert result.collapses_to_ne_v is True
    assert result.generation_interval == 1.0


def test_empty_graph_batch_serializes(empty_graph):
    results = estimate_effective_sizes(empty_graph)
    assert set(results) == ESTIMATOR_NAMES
    for name, result in results.items():
        for field_name, value in _array_fields(result).items():
            assert value.shape == (0,), f"{name}.{field_name}"
        assert isinstance(result.to_dict(), dict)


@_over(ESTIMATORS)
@pytest.mark.parametrize("keyword", ["mean_contributions", "ct_accumulators", "theta_per_gen", "K"])
def test_final_estimator_rejects_injected_prerequisites(line_graph, est, keyword):
    with pytest.raises(TypeError):
        est.call(line_graph, **{keyword: None})


@_over(ESTIMATORS)
def test_final_estimator_rejects_a_second_positional_argument(line_graph, est):
    with pytest.raises(TypeError):
        est.call(line_graph, None)


@_over(LABELLED)
def test_arrays_align_with_the_observed_labels(line_graph, est):
    result = est.call(line_graph)
    labels = getattr(result, est.labels)
    assert labels.dtype == np.int32
    assert np.all(np.diff(labels) > 0)
    k = labels.shape[0]
    for name, value in _array_fields(result).items():
        expected = max(k - 1, 0) if name in est.transitions else k
        assert value.shape == (expected,), name


def test_variance_result_is_indexed_by_parent_cohort(line_graph):
    result = effective_size.ne_variance_family_size(line_graph)
    arrays = _array_fields(result)
    k = result.parent_generations.shape[0]
    assert set(arrays) - {"parent_generations"} == {
        "ne_per_transition",
        "v_mm",
        "v_mf",
        "v_fm",
        "v_ff",
        "cov_m",
        "cov_f",
    }
    assert not hasattr(result, "generations")
    for name, value in arrays.items():
        assert value.shape == (k,), name


@_over(RATE_BASED)
def test_transition_labels_are_the_adjacent_cohort_pairs(line_graph, est):
    result = est.call(line_graph)
    assert np.array_equal(result.transition_from, result.generations[:-1])
    assert np.array_equal(result.transition_to, result.generations[1:])
    assert result.ne_per_gen.shape == (max(result.generations.shape[0] - 1, 0),)


def test_ltc_final_generation_names_the_stopping_cohort(line_graph):
    result = effective_size.ne_long_term_contributions(line_graph)
    generations = effective_size.ne_inbreeding(line_graph).generations
    assert result.n_iterations == 1
    assert result.final_generation == int(generations[result.n_iterations])


def test_ltc_counts_every_adjacent_cohort_comparison_it_makes():
    """The MZ pedigree's cohort 0 holds an extra co-twin row, so the first comparison misses tol."""
    pg = _mz_founder_pedigree()
    result = effective_size.ne_long_term_contributions(pg)
    generations = effective_size.ne_caballero_toro(pg).generations
    assert result.n_iterations == 2
    assert result.final_generation == int(generations[result.n_iterations])


@_over(ESTIMATORS)
def test_result_arrays_are_owned_and_read_only(line_graph, est):
    _assert_owned_read_only(est.call(line_graph))


def test_record_does_not_alias_its_constructor_inputs():
    generations = np.array([0, 1, 2], dtype=np.int32)
    mean_f = np.array([0.0, 0.25, 0.5])
    result = NeInbreedingResult(
        ne=None,
        generations=generations,
        mean_f_per_gen=mean_f,
        transition_from=generations[:-1],
        transition_to=generations[1:],
        ne_per_gen=np.array([1.0, 2.0]),
    )
    generations[0] = 99
    mean_f[0] = 99.0
    assert result.generations[0] == 0
    assert result.mean_f_per_gen[0] == 0.0


def test_hill_age_table_is_an_immutable_mapping_of_read_only_arrays():
    result = effective_size.ne_hill_overlapping(_birth_year_pedigree())
    assert result.collapses_to_ne_v is False
    assert type(result.age_table).__name__ == "mappingproxy"
    assert set(result.age_table) == {"ages_m", "offspring_count_m", "ages_f", "offspring_count_f"}
    for name, value in result.age_table.items():
        assert not value.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            value[...] = 0
    with pytest.raises(TypeError):
        result.age_table["ages_m"] = np.array([0])
    assert type(result.to_dict()["age_table"]) is dict
    _assert_owned_read_only(result)


@_over(ESTIMATORS)
def test_to_dict_returns_plain_python(line_graph, est):
    _assert_plain_python(est.call(line_graph).to_dict(), est.name)


@pytest.mark.parametrize("d", [1e-3, 1e-9])
def test_transition_ne_recovers_a_constant_rate_across_label_gaps(d):
    generations = np.array([0, 2, 5, 9])
    x = -np.expm1(generations * np.log1p(-d))
    assert _transition_ne(x, generations) == pytest.approx(1.0 / (2.0 * d), rel=1e-9)


def test_transition_ne_at_unit_gap_is_the_explicit_one_step_arithmetic():
    x = np.array([0.1, 0.25, 0.375])
    generations = np.array([0, 1, 2])
    out = _transition_ne(x, generations)
    expected = [1.0 / (2.0 * ((x[i + 1] - x[i]) / (1.0 - x[i]))) for i in range(2)]
    assert list(out) == expected


def test_scalar_ne_from_log_regression_is_label_shift_invariant():
    series = np.array([0.0, 0.25, 0.375, 0.5])
    generations = np.array([0, 1, 2, 3], dtype=np.int32)
    assert _scalar_ne_from_log_regression(series, generations) == _scalar_ne_from_log_regression(
        series, generations + 7
    )


@_over(LABELLED)
def test_rebasing_the_labels_leaves_the_estimate_unchanged(est):
    base = est.call(PedigreeGraph.from_frame(_closed_line(4)))
    shifted = est.call(
        PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") + 10).alias("generation")))
    )
    assert base.ne == shifted.ne
    for name, value in _array_fields(base).items():
        if value.dtype.kind != "f":
            continue
        assert np.array_equal(value, getattr(shifted, name), equal_nan=True), name


def test_rebasing_the_labels_shifts_only_the_ltc_final_generation():
    base = effective_size.ne_long_term_contributions(PedigreeGraph.from_frame(_closed_line(4)))
    shifted = effective_size.ne_long_term_contributions(
        PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") + 10).alias("generation")))
    )
    assert base.final_generation == 1
    assert shifted.final_generation == 11
    assert replace(base, final_generation=shifted.final_generation) == shifted


def test_sparse_labels_are_reported_as_observed():
    result = effective_size.ne_inbreeding(PedigreeGraph.from_frame(_relabelled(_closed_line(2), {0: 0, 1: 2, 2: 5})))
    assert np.array_equal(result.generations, [0, 2, 5])
    assert np.array_equal(result.transition_from, [0, 2])
    assert np.array_equal(result.transition_to, [2, 5])


def test_sparse_labels_spread_one_delta_f_over_the_label_gap():
    sparse = effective_size.ne_inbreeding(PedigreeGraph.from_frame(_relabelled(_closed_line(2), {0: 0, 1: 2, 2: 5})))
    dense = effective_size.ne_inbreeding(PedigreeGraph.from_frame(_closed_line(2)))
    assert np.isnan(sparse.ne_per_gen[0])
    assert np.isnan(dense.ne_per_gen[0])
    assert dense.ne_per_gen[1] == pytest.approx(2.0)
    assert sparse.ne_per_gen[1] == pytest.approx(1.0 / (2.0 * (1.0 - (1.0 - 0.25) ** (1.0 / 3.0))))
    assert sparse.ne_per_gen[1] != dense.ne_per_gen[1]


def test_unresolved_external_parent_still_makes_a_represented_founder():
    pg = _external_parent_pedigree()
    assert np.asarray(pg.mother_rows)[0] == -1
    founder_idx = _founder_idx(pg)
    assert np.array_equal(founder_idx, [0, 1])
    assert np.array_equal(_founder_columns(pg, founder_idx), [0, 1, -1])


def test_founder_status_ignores_the_generation_label():
    pg = _off_label_founder_pedigree()
    assert np.array_equal(_founder_idx(pg), [0, 1, 3])


def test_parentless_mz_cotwins_share_one_founder_genome():
    pg = _mz_founder_pedigree()
    founder_idx = _founder_idx(pg)
    assert np.array_equal(founder_idx, [0, 2])
    columns = _founder_columns(pg, founder_idx)
    assert columns[0] == columns[1]
    assert np.array_equal(columns, [0, 0, 1, -1, -1, -1, -1])


def test_mz_founder_pair_matches_a_single_founder_beyond_cohort_zero():
    """Cohort 0 differs by construction: the MZ pedigree carries an extra co-twin row there."""
    mz = _per_gen_founder_means(_mz_founder_pedigree()).m_g
    single = _per_gen_founder_means(_single_founder_pedigree()).m_g
    assert mz[1:] == pytest.approx(single[1:])
    assert mz[0] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert single[0] == pytest.approx([0.5, 0.5])
    assert mz.sum(axis=1) == pytest.approx(1.0)


def test_mz_founder_pair_matches_a_single_founder_in_caballero_toro():
    mz = effective_size.ne_caballero_toro(_mz_founder_pedigree())
    single = effective_size.ne_caballero_toro(_single_founder_pedigree())
    assert np.array_equal(mz.mean_self_coancestry_per_gen, single.mean_self_coancestry_per_gen, equal_nan=True)
    assert np.array_equal(mz.n_founders_with_descendants_per_gen, single.n_founders_with_descendants_per_gen)
    assert mz.mean_self_coancestry_per_gen[1:] == pytest.approx([0.5, 0.625])
    assert np.isnan(mz.mean_self_coancestry_per_gen[0])
    assert np.array_equal(mz.n_founders_with_descendants_per_gen, [0, 2, 2])


def test_caballero_toro_first_cohort_is_the_baseline():
    result = effective_size.ne_caballero_toro(_late_founder_pedigree())
    assert np.isnan(result.mean_self_coancestry_per_gen[0])
    assert result.n_founders_with_descendants_per_gen[0] == 0


def test_a_founder_is_not_its_own_descendant():
    pg = _late_founder_pedigree()
    assert np.array_equal(_founder_idx(pg), [0, 1, 3])
    result = effective_size.ne_caballero_toro(pg)
    assert np.array_equal(result.n_founders_with_descendants_per_gen, [0, 2, 3])


def test_caballero_toro_first_transition_starts_from_the_half_baseline():
    """Self-coancestry is (1 + F) / 2, so a non-inbred baseline is 0.5, not 0."""
    pg = PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") // 2).alias("generation")))
    result = effective_size.ne_caballero_toro(pg)
    expected = _transition_ne(
        np.array([0.5, result.mean_self_coancestry_per_gen[1]]),
        result.generations[:2],
    )
    assert result.ne_per_gen[0] == pytest.approx(expected[0])


def test_labels_that_merge_structural_depths_propagate_by_structure():
    """Two structural depths share each label, so grouping changes but ancestry does not."""
    pg = PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") // 2).alias("generation")))
    assert np.array_equal(pg.generation_labels, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert np.array_equal(pg.depth, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    m_g = _per_gen_founder_means(pg).m_g
    assert m_g == pytest.approx(np.full((3, 2), 0.5))
    assert m_g.sum(axis=1) == pytest.approx(1.0)
    assert effective_size.ne_long_term_contributions(pg).final_generation == 1
    assert np.array_equal(effective_size.ne_caballero_toro(pg).generations, [0, 1, 2])


def test_a_parent_and_its_child_in_one_label_group_still_run():
    pg = PedigreeGraph.from_frame(_relabelled(_closed_line(2), {0: 0, 1: 1, 2: 1}))
    assert effective_size.ne_long_term_contributions(pg).final_generation == 1
    assert np.array_equal(effective_size.ne_caballero_toro(pg).generations, [0, 1])


def test_variance_keeps_the_maximum_parent_cohort():
    result = effective_size.ne_variance_family_size(_two_cohort_reproduction_pedigree())
    assert np.array_equal(result.parent_generations, [0, 1])
    assert np.isnan(result.ne_per_transition[0])
    assert result.ne_per_transition[1] == pytest.approx(4.0)
    assert result.v_mm == pytest.approx([0.0, 1.0 / 3.0])
    for name, value in _array_fields(result).items():
        assert value.shape == (2,), name


def test_checked_founder_matrix_rejects_an_unrepresentable_shape():
    with pytest.raises(ResourceError) as excinfo:
        _checked_founder_matrix(2**40, 2**40, "op", np.float64, 0.0)
    assert excinfo.value.code == "arithmetic_overflow"
    assert excinfo.value.fields["operation"] == "op"
    assert excinfo.value.fields["dtype"] == "float64"


def test_checked_founder_matrix_reports_a_refused_allocation(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(ne_common.np, "full", boom)
    with pytest.raises(ResourceError) as excinfo:
        _checked_founder_matrix(3, 4, "op", np.int64, 0)
    assert excinfo.value.code == "allocation_failed"
    assert excinfo.value.fields["operation"] == "op"
    assert excinfo.value.fields["requested_elements"] == 12
    assert excinfo.value.fields["dtype"] == "int64"


def test_final_inbreeding_reports_the_observed_cohorts(sparse_line_graph):
    result = effective_size.ne_inbreeding(sparse_line_graph)
    assert np.array_equal(result.generations, [0, 2, 5])
    assert result.mean_f_per_gen == pytest.approx([0.0, 0.25, 0.375])
    assert result.ne_per_gen == pytest.approx([3.73205081, 8.47975451])


def test_a_batch_returns_the_eight_estimators(sparse_line_graph):
    assert set(estimate_effective_sizes(sparse_line_graph)) == ESTIMATOR_NAMES


def test_mean_kinship_by_generation_reports_unlabelled_rows():
    pg = PedigreeGraph.from_arrays(
        ids=list(range(8)),
        mother_ids=[-1, -1, 0, 0, 2, 2, 4, 4],
        father_ids=[-1, -1, 1, 1, 3, 3, 5, 5],
        sex=[0, 1, 0, 1, 0, 1, 0, 1],
        generation=[0, 0, 1, 1, 2, 2, -1, -1],
    )
    summary = pg.mean_kinship_by_generation()
    assert np.array_equal(summary.generations, [0, 1, 2])
    assert summary.mean_kinship == pytest.approx([0.0, 0.25, 0.375])
    assert np.array_equal(summary.pair_counts, [1, 1, 1])
    assert summary.unlabelled_individual_count == 2


def test_cohorts_and_kinship_kernel_share_one_densify_labels():
    assert _cohorts_densify_labels is _kernel_densify_labels
