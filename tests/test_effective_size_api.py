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

from pedigree_graph import PedigreeGraph, PedigreeValidationError, ResourceError, effective_size
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


def _birth_year_pedigree_without_an_eligible_cohort() -> PedigreeGraph:
    rows = [
        {"id": 0, "sex": 1, "generation": 0, "birth_year": 1900},
        {"id": 1, "sex": 0, "generation": 0, "birth_year": 1900},
        {"id": 2, "sex": 1, "generation": 1, "birth_year": 1910, "father": 0, "mother": 1},
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
    _Estimator("ne_group_coancestry", effective_size.ne_group_coancestry, "generations", _RATE_TRANSITIONS),
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
    assert result.ne is None
    assert result.n_effective_founders is None
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
@pytest.mark.parametrize("keyword", ["mean_contributions", "group_coancestry", "theta_per_gen", "K"])
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


def test_ltc_final_generation_names_the_last_observed_cohort(line_graph):
    result = effective_size.ne_long_term_contributions(line_graph)
    generations = effective_size.ne_inbreeding(line_graph).generations
    assert result.n_cohorts == 5
    assert result.final_generation == int(generations[-1])


def test_a_parentless_mz_pair_is_one_effective_founder():
    """Record level: the co-twins' shared contribution column (ADR 0008) leaves the record two founder genomes."""
    mz = effective_size.ne_long_term_contributions(_mz_founder_pedigree())
    assert mz.n_effective_founders == 2.0
    assert mz.n_cohorts == 3
    assert mz.final_generation == 2
    assert mz.to_dict() == effective_size.ne_long_term_contributions(_single_founder_pedigree()).to_dict()


def _rows_labelled(pg: PedigreeGraph, *labels: int) -> np.ndarray:
    return np.flatnonzero(np.isin(np.asarray(pg.generation_labels), labels))


def test_the_default_delta_f_reference_is_the_last_observed_cohort(line_graph):
    default = effective_size.ne_individual_delta_f(line_graph)
    explicit = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 4))
    assert default == explicit
    assert default.reference_generation == 4


def test_a_narrower_delta_f_reference_moves_the_scalar_but_not_the_series(line_graph):
    whole = effective_size.ne_individual_delta_f(line_graph)
    part = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 2))
    assert part.n_reference == 2
    assert part.reference_generation == 2
    assert part.ne != whole.ne
    assert np.array_equal(part.ne_per_gen, whole.ne_per_gen, equal_nan=True)

    assert whole.ne == pytest.approx(3.142606753941622, rel=1e-12)
    assert whole.ne_unrelated_founders == pytest.approx(2.423661050931537, rel=1e-12)
    assert part.ne == pytest.approx(3.732050807568876, rel=1e-12)
    assert part.ne_unrelated_founders == pytest.approx(2.0, abs=1e-12)
    assert part.ne_unrelated_founders != whole.ne_unrelated_founders


def test_a_delta_f_reference_spanning_two_cohorts_has_no_single_label(line_graph):
    result = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 3, 4))
    assert result.n_reference == 4
    assert result.reference_generation is None


def test_the_delta_f_reference_label_follows_the_eligible_rows(line_graph):
    """Founders carry no rate, so they neither count toward N nor claim the label."""
    result = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 0, 2))
    assert result.n_reference == 2
    assert result.reference_generation == 2


def test_a_one_generation_deep_delta_f_reference_yields_no_unrelated_founder_diagnostic(line_graph):
    """``ne`` accepts ``t > 0`` and the diagnostic ``t > 1``, so only this direction of disagreement exists.

    These rows have ``t = 1`` and ``F = 0``, eligible for ``ne`` but carrying
    no drift, so ΔF̄ is 0 and neither field reports a number.  A reference
    where ``ne`` is a number and the diagnostic is ``None`` needs some row with
    ``F_i > 0`` at ``t_i ≤ 1``, and ``F_i > 0`` needs two known parents (1.0 of
    ``t_i`` between them) that are themselves related, which adds a further
    known ancestor path at meiotic distance 2 worth 0.25.  So ``F_i > 0``
    forces ``t_i ≥ 1.25``, a floor realised by parent-offspring mating
    (``F = 0.25`` at ``t = 1.25``; full-sib mating gives 2.0).

    The one structure that escapes the floor is a child of two MZ co-twins,
    ``F = 0.5`` at ``t = 1``, which the test below pins.
    """
    result = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 1))
    assert result.n_reference == 2
    assert result.ne is None
    assert result.ne_unrelated_founders is None


def _mz_parent_pedigree() -> PedigreeGraph:
    """A child of two MZ co-twins: one genome node in both parent roles.

    ``check_same_parent`` forbids one *row* in both roles and the MZ sex check
    forbids co-twins of differing known sex, but neither forbids this, and ADR
    0008 makes the genome node the semantic unit that makes it meaningful.
    """
    return PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 0, "generation": 0, "twin": 1},
                {"id": 1, "sex": 0, "generation": 0, "twin": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 0, "father": 1},
            ]
        )
    )


def test_the_stricter_diagnostic_eligibility_can_empty_out_while_ne_still_reports():
    """``F = 0.5`` at ``t = 1`` is the one place the two eligibility rules disagree.

    Selfing a genome node puts a full generation of drift into a row one
    generation deep, which is the only way past the ``t_i ≥ 1.25`` floor the
    test above derives.  ``ne`` takes the row at ``t > 0`` and reports
    ``1/(2·0.5)``; the diagnostic wants ``t > 1``, finds nothing, and reports
    ``None`` rather than a number over an empty average.
    """
    result = effective_size.ne_individual_delta_f(_mz_parent_pedigree(), reference=[2])
    assert result.n_reference == 1
    assert result.ne == pytest.approx(1.0, abs=1e-12)
    assert result.ne_unrelated_founders is None


def test_the_unrelated_founder_diagnostic_narrows_the_reference_without_a_count(line_graph):
    """All four rows feed ``ne``; only the two at ``t = 2`` feed the diagnostic, and nothing reports that."""
    result = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 1, 2))
    assert result.n_reference == 4
    assert result.ne == pytest.approx(7.464101615137752, rel=1e-12)
    assert result.ne_unrelated_founders == pytest.approx(2.0, abs=1e-12)


def test_an_empty_delta_f_reference_and_an_empty_graph_have_no_unrelated_founder_diagnostic(line_graph, empty_graph):
    assert effective_size.ne_individual_delta_f(line_graph, reference=[]).n_reference == 0
    assert effective_size.ne_individual_delta_f(line_graph, reference=[]).ne_unrelated_founders is None
    assert effective_size.ne_individual_delta_f(empty_graph).ne_unrelated_founders is None


def test_a_delta_f_reference_of_founders_alone_yields_no_estimate(line_graph):
    result = effective_size.ne_individual_delta_f(line_graph, reference=_rows_labelled(line_graph, 0))
    assert result.n_reference == 0
    assert result.ne is None
    assert result.standard_error is None
    assert result.reference_generation is None


def test_a_delta_f_reference_row_outside_the_pedigree_is_rejected(line_graph):
    with pytest.raises(PedigreeValidationError) as excinfo:
        effective_size.ne_individual_delta_f(line_graph, reference=[0, 10])
    assert excinfo.value.code == "reference_row_out_of_range"
    assert excinfo.value.fields == {"row": 10, "position": 1, "n_individuals": 10}


def test_a_repeated_delta_f_reference_row_is_rejected(line_graph):
    """A repeat would weight one individual twice in ΔF̄ with nothing to show for it."""
    with pytest.raises(PedigreeValidationError) as excinfo:
        effective_size.ne_individual_delta_f(line_graph, reference=[8, 9, 8])
    assert excinfo.value.code == "duplicate_reference_row"
    assert excinfo.value.fields == {"row": 8, "positions": (0, 2), "duplicate_count": 1}


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


def test_hill_result_producers_cover_all_three_states(empty_graph):
    sentinel = effective_size.ne_hill_overlapping(empty_graph)
    birth_year_empty = effective_size.ne_hill_overlapping(_birth_year_pedigree_without_an_eligible_cohort())
    birth_year_populated = effective_size.ne_hill_overlapping(_birth_year_pedigree())

    assert (sentinel.collapses_to_ne_v, sentinel.n_eligible_cohorts) == (True, 0)
    assert (birth_year_empty.collapses_to_ne_v, birth_year_empty.n_eligible_cohorts) == (False, 0)
    assert birth_year_empty.ne is None
    assert birth_year_empty.cohort_years is None
    assert (birth_year_populated.collapses_to_ne_v, birth_year_populated.n_eligible_cohorts) == (False, 1)


@pytest.mark.parametrize("changes", [{"generation_interval": 2.0}, {"T_m": 1.0}, {"n_eligible_cohorts": 1}])
def test_hill_result_rejects_an_invalid_sentinel_state(empty_graph, changes):
    result = effective_size.ne_hill_overlapping(empty_graph)
    with pytest.raises(ValueError, match="sentinel state"):
        replace(result, **changes)


@pytest.mark.parametrize("changes", [{"ne": 10.0}, {"cohort_years": np.array([], dtype=np.int32)}])
def test_hill_result_rejects_an_invalid_birth_year_empty_state(changes):
    result = effective_size.ne_hill_overlapping(_birth_year_pedigree_without_an_eligible_cohort())
    with pytest.raises(ValueError, match="birth-year-empty state"):
        replace(result, **changes)


@pytest.mark.parametrize("changes", [{"Vk_m": None}, {"n_eligible_cohorts": 2}])
def test_hill_result_rejects_an_invalid_birth_year_populated_state(changes):
    result = effective_size.ne_hill_overlapping(_birth_year_pedigree())
    with pytest.raises(ValueError, match="birth-year-populated state"):
        replace(result, **changes)


@_over(ESTIMATORS)
def test_to_dict_returns_plain_python(line_graph, est):
    _assert_plain_python(est.call(line_graph).to_dict(), est.name)


def test_the_serialized_delta_f_record_always_carries_the_diagnostic_key(line_graph, empty_graph):
    """The key's presence is how a consumer tells a post-ADR-0012 record from a 0.8 one.

    simACE reads it that way: only the corrected estimator reports
    ``ne_unrelated_founders``, so the key must survive serialization even
    where the value is ``None``, or the probe silently misreads a current
    record as an old one.
    """
    for pg in (line_graph, empty_graph):
        payload = effective_size.ne_individual_delta_f(pg).to_dict()
        assert "ne_unrelated_founders" in payload
    assert effective_size.ne_individual_delta_f(empty_graph).to_dict()["ne_unrelated_founders"] is None


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
    assert base.final_generation == 4
    assert shifted.final_generation == 14
    assert replace(base, final_generation=shifted.final_generation) == shifted


def test_rebasing_the_labels_shifts_only_the_delta_f_reference_generation():
    base = effective_size.ne_individual_delta_f(PedigreeGraph.from_frame(_closed_line(4)))
    shifted = effective_size.ne_individual_delta_f(
        PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") + 10).alias("generation")))
    )
    assert base.reference_generation == 4
    assert shifted.reference_generation == 14
    assert replace(base, generations=shifted.generations, reference_generation=shifted.reference_generation) == shifted


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


def test_an_mz_founder_pair_matches_a_single_founder_in_group_coancestry():
    """One genome is one genome, in the founder cohort as in every later one.

    :func:`test_mz_founder_pair_matches_a_single_founder_beyond_cohort_zero`
    has to exclude cohort 0, because the founder-contribution means count
    rows there and the MZ pedigree carries an extra one.  The genome-node
    collapse removes that exception: these two records are equal field for
    field, cohort 0 included.
    """
    mz = effective_size.ne_group_coancestry(_mz_founder_pedigree())

    assert mz == effective_size.ne_group_coancestry(_single_founder_pedigree())
    assert np.array_equal(mz.n_genomes_per_gen, [2, 2, 2])
    assert np.array_equal(mz.mean_group_coancestry_per_gen, [0.25, 0.375, 0.5])


def test_group_coancestry_computes_its_baseline_around_a_late_founder():
    """The first cohort's f̄ is eq. 3 over its own rows, never a sentinel.

    Those two rows are unrelated and non-inbred, so the baseline is C&T's
    ``1/(2N)`` at ``N = 2``.  The cohort after it holds one child and one
    newly appearing founder, unrelated to each other, so its f̄ is that same
    ``0.25``: no drift where the pedigree records none.  That newly
    appearing founder counts toward its own cohort's census, which leaves a
    two-row cohort beside a one-row one and a ``census_ratio`` that says so.
    """
    pg = _late_founder_pedigree()
    assert np.array_equal(_founder_idx(pg), [0, 1, 3])

    result = effective_size.ne_group_coancestry(pg)

    assert np.array_equal(result.n_genomes_per_gen, [2, 2, 1])
    assert np.array_equal(result.mean_group_coancestry_per_gen, [0.25, 0.25, 0.5])
    assert result.census_ratio == 2.0


def test_labels_that_merge_structural_depths_propagate_by_structure():
    """Two structural depths share each label, so grouping changes but ancestry does not."""
    pg = PedigreeGraph.from_frame(_closed_line(4).with_columns((pl.col("generation") // 2).alias("generation")))
    assert np.array_equal(pg.generation_labels, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert np.array_equal(pg.depth, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    m_g = _per_gen_founder_means(pg).m_g
    assert m_g == pytest.approx(np.full((3, 2), 0.5))
    assert m_g.sum(axis=1) == pytest.approx(1.0)
    assert effective_size.ne_long_term_contributions(pg).final_generation == 2
    assert np.array_equal(effective_size.ne_group_coancestry(pg).generations, [0, 1, 2])


def test_a_parent_and_its_child_in_one_label_group_still_run():
    pg = PedigreeGraph.from_frame(_relabelled(_closed_line(2), {0: 0, 1: 1, 2: 1}))
    assert effective_size.ne_long_term_contributions(pg).final_generation == 1
    assert np.array_equal(effective_size.ne_group_coancestry(pg).generations, [0, 1])


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
