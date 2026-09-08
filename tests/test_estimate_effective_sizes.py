"""Contract tests for ``estimate_effective_sizes`` and its result mapping.

Covers selector validation and its ordering against the thread budget, the
immutable eight-key mapping, the ``UnavailableEffectiveSize`` sentinel, the
per-call prerequisite memo (what each selection builds and shares), parity
with the eight direct estimators, and serialization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph, _threads, configure_threads
from pedigree_graph import _ne_estimate as ne_estimate
from pedigree_graph import effective_size as es
from pedigree_graph.effective_size import (
    ALL_EFFECTIVE_SIZE_ESTIMATORS,
    EffectiveSizeResults,
    UnavailableEffectiveSize,
    estimate_effective_sizes,
)

_IDS = np.arange(8)
_MOTHER = np.array([-1, -1, 0, 0, 2, 2, 4, 4])
_FATHER = np.array([-1, -1, 1, 1, 3, 3, 5, 5])
_SEX = np.array([0, 1, 0, 1, 0, 1, 0, 1])
_GEN = np.array([0, 0, 1, 1, 2, 2, 3, 3])
_BIRTH = np.array([1900, 1900, 1920, 1920, 1940, 1940, 1960, 1960])
_ONE_PARENT_FATHER = np.array([-1, -1, 1, 1, 3, -1, 5, 5])

_DIRECT = {name: getattr(es, name) for name in ALL_EFFECTIVE_SIZE_ESTIMATORS}
_NEEDS_PARENTAGE = ("ne_long_term_contributions", "ne_caballero_toro")
_NEEDS_SEX = ("ne_variance_family_size", "ne_sex_ratio", "ne_hill_overlapping")


def _graph(**overrides):
    columns = {"id": _IDS, "mother": _MOTHER, "father": _FATHER, "sex": _SEX, "generation": _GEN}
    columns.update(overrides)
    return PedigreeGraph.from_frame({k: v for k, v in columns.items() if v is not None})


def _empty_graph():
    return PedigreeGraph.from_frame({"id": [], "mother": [], "father": []})


def _without(*names):
    return [name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name not in names]


@pytest.fixture
def no_estimator_work(monkeypatch):
    """Turn any estimator dispatch into a failure, so ordering violations are loud."""

    def refuse(self, name):
        raise AssertionError("work")

    monkeypatch.setattr(ne_estimate._Prerequisites, "_compute", refuse)


@pytest.fixture
def prerequisites(monkeypatch):
    """Collect the per-call memo instances so a test can read ``computed()``."""
    built = []

    class Spy(ne_estimate._Prerequisites):
        __slots__ = ()

        def __init__(self, pg, hill_vk_scale):
            super().__init__(pg, hill_vk_scale)
            built.append(self)

    monkeypatch.setattr(ne_estimate, "_Prerequisites", Spy)
    return built


def _count_calls(monkeypatch, name):
    calls = []
    real = getattr(ne_estimate, name)

    def counted(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(ne_estimate, name, counted)
    return calls


@pytest.mark.usefixtures("no_estimator_work")
class TestSelectorValidation:
    @pytest.mark.parametrize(
        "estimators",
        [None, "ne_inbreeding", b"ne_inbreeding", [1], ["ne_inbreeding", None]],
        ids=["none", "str", "bytes", "int_element", "none_element"],
    )
    def test_a_bad_selector_raises_type_error_before_any_work(self, estimators):
        with pytest.raises(TypeError):
            estimate_effective_sizes(_graph(), estimators)

    @pytest.mark.parametrize("estimators", [["nope"], ["ne_inbreeding", "nope"]], ids=["only", "trailing"])
    def test_an_unknown_name_raises_value_error_before_any_work(self, estimators):
        with pytest.raises(ValueError, match="unknown estimator"):
            estimate_effective_sizes(_graph(), estimators)

    @pytest.mark.parametrize(
        "hill_vk_scale",
        [1, "yes", None, np.True_],
        ids=["int", "str", "none", "numpy_bool"],
    )
    def test_hill_vk_scale_must_be_an_actual_bool(self, hill_vk_scale):
        with pytest.raises(TypeError):
            estimate_effective_sizes(_graph(), ["ne_hill_overlapping"], hill_vk_scale=hill_vk_scale)


class TestSelectorAcceptance:
    def test_a_one_shot_generator_is_materialized(self):
        names = (name for name in ("ne_sex_ratio", "ne_inbreeding"))
        result = estimate_effective_sizes(_graph(), names)
        assert result["ne_sex_ratio"].ne == 2.0
        assert result["ne_inbreeding"].ne is not None

    def test_an_empty_selection_is_valid_and_builds_nothing(self, prerequisites):
        result = estimate_effective_sizes(_graph(), [])
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert all(value == UnavailableEffectiveSize.not_requested() for value in result.values())
        assert prerequisites[0].computed() == frozenset()

    def test_duplicates_behave_like_one_name(self):
        pg = _graph()
        assert estimate_effective_sizes(pg, ["ne_inbreeding", "ne_inbreeding"]) == estimate_effective_sizes(
            pg, ["ne_inbreeding"]
        )

    def test_input_order_does_not_change_output_order(self):
        result = estimate_effective_sizes(_graph(), ["ne_caballero_toro", "ne_inbreeding"])
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)

    def test_hill_vk_scale_true_reaches_the_hill_record(self):
        result = estimate_effective_sizes(_graph(birth_year=_BIRTH), ["ne_hill_overlapping"], hill_vk_scale=True)
        assert result["ne_hill_overlapping"].vk_scaled is True


class TestResultMapping:
    def test_it_is_a_mapping_over_the_eight_keys(self):
        result = estimate_effective_sizes(_graph(), ["ne_inbreeding"])
        assert isinstance(result, EffectiveSizeResults)
        assert isinstance(result, Mapping)
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert len(result) == 8

    def test_an_unknown_key_raises_key_error(self):
        result = estimate_effective_sizes(_graph(), [])
        with pytest.raises(KeyError):
            result["nope"]

    def test_item_assignment_is_rejected(self):
        result = estimate_effective_sizes(_graph(), [])
        with pytest.raises(TypeError):
            result["x"] = 1

    def test_attribute_mutation_is_rejected(self):
        result = estimate_effective_sizes(_graph(), [])
        with pytest.raises(AttributeError):
            result._items = ()

    def test_attribute_deletion_is_rejected(self):
        result = estimate_effective_sizes(_graph(), [])
        with pytest.raises(AttributeError):
            del result._items

    def test_two_calls_on_one_graph_are_equal(self):
        pg = _graph()
        assert estimate_effective_sizes(pg) == estimate_effective_sizes(pg)

    def test_it_equals_a_plain_dict_of_its_items(self):
        result = estimate_effective_sizes(_graph())
        assert result == dict(result.items())

    def test_different_selections_differ(self):
        pg = _graph()
        assert estimate_effective_sizes(pg, ["ne_inbreeding"]) != estimate_effective_sizes(pg, ["ne_sex_ratio"])

    def test_repr_names_the_class(self):
        assert "EffectiveSizeResults" in repr(estimate_effective_sizes(_graph(), []))


class TestUnavailable:
    @pytest.mark.parametrize("name", _without("ne_inbreeding"))
    def test_unselected_keys_are_not_requested(self, name):
        result = estimate_effective_sizes(_graph(), ["ne_inbreeding"])
        assert result[name] == UnavailableEffectiveSize(reason="not_requested", code=None, fields={})
        assert result[name].to_dict() == {"reason": "not_requested", "code": None, "fields": {}}

    def test_fields_are_an_immutable_proxy(self):
        value = estimate_effective_sizes(_graph(), [])["ne_inbreeding"]
        assert isinstance(value.fields, MappingProxyType)
        with pytest.raises(TypeError):
            value.fields["x"] = 1

    @pytest.mark.parametrize(
        ("reason", "code", "match"),
        [
            ("missing_metadata", None, "code is None"),
            ("not_requested", "x", "code is None"),
            ("bogus", None, "unknown reason"),
        ],
        ids=["metadata_without_code", "not_requested_with_code", "unknown_reason"],
    )
    def test_inconsistent_construction_is_rejected(self, reason, code, match):
        with pytest.raises(ValueError, match=match):
            UnavailableEffectiveSize(reason, code, {})


class TestMissingMetadata:
    @pytest.mark.parametrize("name", _NEEDS_PARENTAGE)
    def test_incomplete_parentage_names_the_estimator_it_disables(self, name):
        value = estimate_effective_sizes(_graph(father=_ONE_PARENT_FATHER))[name]
        assert value.reason == "missing_metadata"
        assert value.code == "incomplete_parentage"
        assert value.fields["operation"] == name

    @pytest.mark.parametrize("name", _without(*_NEEDS_PARENTAGE))
    def test_incomplete_parentage_leaves_the_other_six_intact(self, name):
        value = estimate_effective_sizes(_graph(father=_ONE_PARENT_FATHER))[name]
        assert not isinstance(value, UnavailableEffectiveSize)

    @pytest.mark.parametrize("name", _NEEDS_SEX)
    def test_absent_sex_names_the_estimator_it_disables(self, name):
        value = estimate_effective_sizes(_graph(sex=None))[name]
        assert value.reason == "missing_metadata"
        assert value.code == "missing_sex"
        assert value.fields["operation"] == name

    @pytest.mark.parametrize("name", _without(*_NEEDS_SEX))
    def test_absent_sex_leaves_the_other_five_intact(self, name):
        value = estimate_effective_sizes(_graph(sex=None))[name]
        assert not isinstance(value, UnavailableEffectiveSize)

    def test_a_non_metadata_error_propagates(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(ne_estimate, "_inbreeding_from", boom)
        with pytest.raises(RuntimeError):
            estimate_effective_sizes(_graph(), ["ne_inbreeding"])


class TestPrerequisiteClosure:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("ne_inbreeding", {"observed_cohorts", "inbreeding", "ne_inbreeding"}),
            ("ne_sex_ratio", {"observed_cohorts", "ne_sex_ratio"}),
            ("ne_coancestry", {"observed_cohorts", "theta_summary", "ne_coancestry"}),
            (
                "ne_long_term_contributions",
                {"observed_cohorts", "represented_founders", "founder_means", "ne_long_term_contributions"},
            ),
            (
                "ne_caballero_toro",
                {
                    "observed_cohorts",
                    "inbreeding",
                    "represented_founders",
                    "ct_accumulators",
                    "ne_caballero_toro",
                },
            ),
            (
                "ne_variance_family_size",
                {"observed_cohorts", "generation_family_table", "ne_variance_family_size"},
            ),
        ],
    )
    def test_a_single_selection_builds_exactly_its_closure(self, prerequisites, name, expected):
        estimate_effective_sizes(_graph(), [name])
        assert prerequisites[0].computed() == expected

    def test_long_term_contributions_does_not_build_the_caballero_toro_accumulators(self, prerequisites):
        estimate_effective_sizes(_graph(), ["ne_long_term_contributions"])
        assert "ct_accumulators" not in prerequisites[0].computed()

    def test_hill_without_birth_years_collapses_through_a_private_variance(self, prerequisites):
        result = estimate_effective_sizes(_graph(), ["ne_hill_overlapping"])
        assert prerequisites[0].computed() == {
            "observed_cohorts",
            "generation_family_table",
            "ne_variance_family_size",
            "ne_hill_overlapping",
        }
        assert result["ne_variance_family_size"] == UnavailableEffectiveSize.not_requested()

    def test_hill_with_birth_years_skips_the_generation_cohorts(self, prerequisites):
        estimate_effective_sizes(_graph(birth_year=_BIRTH), ["ne_hill_overlapping"])
        assert prerequisites[0].computed() == {
            "generation_interval",
            "cohort_window",
            "birth_year_family_table",
            "ne_hill_overlapping",
        }

    def test_hill_and_variance_share_one_variance_computation(self, monkeypatch):
        calls = _count_calls(monkeypatch, "_variance_from")
        estimate_effective_sizes(_graph(), ["ne_hill_overlapping", "ne_variance_family_size"])
        assert len(calls) == 1

    def test_long_term_contributions_and_caballero_toro_share_one_founder_index(self, monkeypatch):
        calls = _count_calls(monkeypatch, "_founder_idx")
        estimate_effective_sizes(_graph(), list(_NEEDS_PARENTAGE))
        assert len(calls) == 1

    def test_a_failed_guard_memoizes_nothing_for_that_estimator(self, prerequisites):
        estimate_effective_sizes(_graph(father=_ONE_PARENT_FATHER))
        computed = prerequisites[0].computed()
        assert computed.isdisjoint(_NEEDS_PARENTAGE)


class TestDirectParity:
    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    @pytest.mark.parametrize("birth_year", [None, _BIRTH], ids=["no_birth_years", "birth_years"])
    def test_the_orchestrated_record_equals_the_direct_one(self, name, birth_year):
        pg = _graph(birth_year=birth_year)
        assert estimate_effective_sizes(pg)[name] == _DIRECT[name](pg)

    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    def test_the_empty_graph_yields_real_records_without_an_estimate(self, name):
        value = estimate_effective_sizes(_empty_graph())[name]
        assert not isinstance(value, UnavailableEffectiveSize)
        assert value.ne is None


class TestSerialization:
    def test_to_dict_is_a_plain_ordered_dict_of_plain_dicts(self):
        payload = estimate_effective_sizes(_graph(birth_year=_BIRTH)).to_dict()
        assert type(payload) is dict
        assert list(payload) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert all(type(value) is dict for value in payload.values())
        json.dumps(payload)

    def test_unavailable_entries_serialize_to_plain_field_dicts(self):
        payload = estimate_effective_sizes(_graph(father=_ONE_PARENT_FATHER)).to_dict()
        not_requested = estimate_effective_sizes(_graph(), []).to_dict()["ne_inbreeding"]
        assert set(not_requested) == {"reason", "code", "fields"}
        entry = payload["ne_caballero_toro"]
        assert set(entry) == {"reason", "code", "fields"}
        assert type(entry["fields"]) is dict
        assert entry["fields"]["operation"] == "ne_caballero_toro"
        assert entry["fields"]["unrepresented_parent_status"] == "missing"
        json.dumps(payload)


class TestThreadBudget:
    @pytest.fixture(autouse=True)
    def _thread_state(self, monkeypatch):
        monkeypatch.delenv("PEDIGREE_GRAPH_THREADS", raising=False)
        _threads._reset_thread_state()
        yield
        _threads._reset_thread_state()

    def test_a_call_commits_the_default_budget(self):
        estimate_effective_sizes(_graph(), ["ne_sex_ratio"])
        assert _threads._STATE.committed == 1

    def test_a_call_commits_a_configured_budget(self):
        configure_threads(3)
        estimate_effective_sizes(_graph(), ["ne_sex_ratio"])
        assert _threads._STATE.committed == 3

    def test_results_do_not_depend_on_the_budget(self):
        under_one = estimate_effective_sizes(_graph(birth_year=_BIRTH))
        _threads._reset_thread_state()
        configure_threads(3)
        under_three = estimate_effective_sizes(_graph(birth_year=_BIRTH))
        assert _threads._STATE.committed == 3
        assert under_one == under_three

    def test_selector_validation_precedes_the_commit(self):
        with pytest.raises(TypeError):
            estimate_effective_sizes(_graph(), None)
        assert _threads._STATE.committed is None


@pytest.mark.parametrize(
    "kwargs",
    [{"n_threads": 2}, {"skip_ne_coancestry": True}, {"theta_per_gen": {}}],
    ids=["n_threads", "skip_ne_coancestry", "theta_per_gen"],
)
def test_injected_keywords_are_rejected(kwargs):
    with pytest.raises(TypeError):
        estimate_effective_sizes(_graph(), **kwargs)
