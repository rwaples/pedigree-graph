"""Contract tests for ``estimate_effective_sizes`` and its result mapping.

Covers selector validation and its ordering against the thread budget, the
immutable eight-key mapping, the ``UnavailableEffectiveSize`` sentinel, the
per-call prerequisite memo (what each selection builds and shares), parity
with the eight direct estimators, and serialization.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from _support import CHAIN_BIRTH, _chain_graph

from pedigree_graph import MissingMetadataError, PedigreeGraph, _threads, configure_threads
from pedigree_graph import _ne_estimate as ne_estimate
from pedigree_graph import _ne_rates as ne_rates
from pedigree_graph import effective_size as es
from pedigree_graph.effective_size import (
    ALL_EFFECTIVE_SIZE_ESTIMATORS,
    EffectiveSizeResults,
    UnavailableEffectiveSize,
    estimate_effective_sizes,
)

_ONE_PARENT_FATHER = np.array([-1, -1, 1, 1, 3, -1, 5, 5])
_PARTIAL_GEN = np.array([0, 0, 1, 1, -1, 2, 3, 3])
_PARTIAL_SEX = np.array([0, 1, 0, 1, -1, 1, 0, 1])
_UNIFORM_SEX = np.zeros(8, dtype=np.int64)

_DEGENERATE = {
    "clean": {},
    "partial_labels": {"generation": _PARTIAL_GEN},
    "absent_sex": {"sex": None},
    "partial_sex": {"sex": _PARTIAL_SEX},
    "uniform_sex": {"sex": _UNIFORM_SEX},
    "one_parent": {"father": _ONE_PARENT_FATHER},
    "labels+sex": {"generation": _PARTIAL_GEN, "sex": None},
    "labels+parentage": {"generation": _PARTIAL_GEN, "father": _ONE_PARENT_FATHER},
    "sex+parentage": {"sex": None, "father": _ONE_PARENT_FATHER},
    "labels+sex+parentage": {"generation": _PARTIAL_GEN, "sex": None, "father": _ONE_PARENT_FATHER},
    "clean+birth": {"birth_year": CHAIN_BIRTH},
    "labels+birth": {"generation": _PARTIAL_GEN, "birth_year": CHAIN_BIRTH},
    "labels+sex+birth": {"generation": _PARTIAL_GEN, "sex": None, "birth_year": CHAIN_BIRTH},
    "uniform_sex+birth": {"sex": _UNIFORM_SEX, "birth_year": CHAIN_BIRTH},
}

_DIRECT = {name: getattr(es, name) for name in ALL_EFFECTIVE_SIZE_ESTIMATORS}
_NEEDS_PARENTAGE = ("ne_long_term_contributions",)
_NEEDS_SEX = ("ne_variance_family_size", "ne_sex_ratio", "ne_hill_overlapping")


def _empty_graph():
    return PedigreeGraph.from_frame({"id": [], "mother": [], "father": []})


def _refusal(code, fields):
    return ("missing_metadata", code, tuple(sorted(fields.items())))


def _outcome(call):
    """``(outcome, warnings)`` for one estimator call, on either path.

    The two paths report a refusal differently, the standalone function
    raising where the orchestrator returns a sentinel, so both normalize to
    one tuple here.  What is compared is then the refusal itself rather than
    the shape it arrived in.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except MissingMetadataError as error:
            outcome = _refusal(error.code, error.fields)
        else:
            if isinstance(value, UnavailableEffectiveSize):
                outcome = _refusal(value.code, value.fields)
            else:
                outcome = ("value", value)
    return outcome, tuple(sorted(str(record.message) for record in caught))


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


def _count_calls(monkeypatch, name, module=ne_estimate):
    calls = []
    real = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
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
            estimate_effective_sizes(_chain_graph(), estimators)

    @pytest.mark.parametrize("estimators", [["nope"], ["ne_inbreeding", "nope"]], ids=["only", "trailing"])
    def test_an_unknown_name_raises_value_error_before_any_work(self, estimators):
        with pytest.raises(ValueError, match="unknown estimator"):
            estimate_effective_sizes(_chain_graph(), estimators)

    @pytest.mark.parametrize(
        "hill_vk_scale",
        [1, "yes", None, np.True_],
        ids=["int", "str", "none", "numpy_bool"],
    )
    def test_hill_vk_scale_must_be_an_actual_bool(self, hill_vk_scale):
        with pytest.raises(TypeError):
            estimate_effective_sizes(_chain_graph(), ["ne_hill_overlapping"], hill_vk_scale=hill_vk_scale)


class TestSelectorAcceptance:
    def test_a_one_shot_generator_is_materialized(self):
        names = (name for name in ("ne_sex_ratio", "ne_inbreeding"))
        result = estimate_effective_sizes(_chain_graph(), names)
        assert result["ne_sex_ratio"].ne == 2.0
        assert result["ne_inbreeding"].ne is not None

    def test_an_empty_selection_is_valid_and_builds_nothing(self, prerequisites):
        result = estimate_effective_sizes(_chain_graph(), [])
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert all(value == UnavailableEffectiveSize.not_requested() for value in result.values())
        assert prerequisites[0].computed() == frozenset()

    def test_duplicates_behave_like_one_name(self):
        pg = _chain_graph()
        assert estimate_effective_sizes(pg, ["ne_inbreeding", "ne_inbreeding"]) == estimate_effective_sizes(
            pg, ["ne_inbreeding"]
        )

    def test_input_order_does_not_change_output_order(self):
        result = estimate_effective_sizes(_chain_graph(), ["ne_group_coancestry", "ne_inbreeding"])
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)

    def test_hill_vk_scale_true_reaches_the_hill_record(self):
        result = estimate_effective_sizes(
            _chain_graph(birth_year=CHAIN_BIRTH), ["ne_hill_overlapping"], hill_vk_scale=True
        )
        assert result["ne_hill_overlapping"].vk_scaled is True


class TestResultMapping:
    def test_it_is_a_mapping_over_the_eight_keys(self):
        result = estimate_effective_sizes(_chain_graph(), ["ne_inbreeding"])
        assert isinstance(result, EffectiveSizeResults)
        assert isinstance(result, Mapping)
        assert list(result) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert len(result) == 8

    def test_an_unknown_key_raises_key_error(self):
        result = estimate_effective_sizes(_chain_graph(), [])
        with pytest.raises(KeyError):
            result["nope"]

    def test_item_assignment_is_rejected(self):
        result = estimate_effective_sizes(_chain_graph(), [])
        with pytest.raises(TypeError):
            result["x"] = 1

    def test_attribute_mutation_is_rejected(self):
        result = estimate_effective_sizes(_chain_graph(), [])
        with pytest.raises(AttributeError):
            result._items = ()

    def test_attribute_deletion_is_rejected(self):
        result = estimate_effective_sizes(_chain_graph(), [])
        with pytest.raises(AttributeError):
            del result._items

    def test_two_calls_on_one_graph_are_equal(self):
        pg = _chain_graph()
        assert estimate_effective_sizes(pg) == estimate_effective_sizes(pg)

    def test_it_equals_a_plain_dict_of_its_items(self):
        result = estimate_effective_sizes(_chain_graph())
        assert result == dict(result.items())

    def test_different_selections_differ(self):
        pg = _chain_graph()
        assert estimate_effective_sizes(pg, ["ne_inbreeding"]) != estimate_effective_sizes(pg, ["ne_sex_ratio"])

    def test_repr_names_the_class(self):
        assert "EffectiveSizeResults" in repr(estimate_effective_sizes(_chain_graph(), []))


class TestUnavailable:
    @pytest.mark.parametrize("name", _without("ne_inbreeding"))
    def test_unselected_keys_are_not_requested(self, name):
        result = estimate_effective_sizes(_chain_graph(), ["ne_inbreeding"])
        assert result[name] == UnavailableEffectiveSize(reason="not_requested", code=None, fields={})
        assert result[name].to_dict() == {"reason": "not_requested", "code": None, "fields": {}}

    def test_fields_are_an_immutable_proxy(self):
        value = estimate_effective_sizes(_chain_graph(), [])["ne_inbreeding"]
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
        value = estimate_effective_sizes(_chain_graph(father=_ONE_PARENT_FATHER))[name]
        assert value.reason == "missing_metadata"
        assert value.code == "incomplete_parentage"
        assert value.fields["operation"] == name

    @pytest.mark.parametrize("name", _without(*_NEEDS_PARENTAGE))
    def test_incomplete_parentage_leaves_the_other_seven_intact(self, name):
        value = estimate_effective_sizes(_chain_graph(father=_ONE_PARENT_FATHER))[name]
        assert not isinstance(value, UnavailableEffectiveSize)

    @pytest.mark.parametrize("name", _NEEDS_SEX)
    def test_absent_sex_names_the_estimator_it_disables(self, name):
        value = estimate_effective_sizes(_chain_graph(sex=None))[name]
        assert value.reason == "missing_metadata"
        assert value.code == "missing_sex"
        assert value.fields["operation"] == name

    @pytest.mark.parametrize("name", _without(*_NEEDS_SEX))
    def test_absent_sex_leaves_the_other_five_intact(self, name):
        value = estimate_effective_sizes(_chain_graph(sex=None))[name]
        assert not isinstance(value, UnavailableEffectiveSize)

    def test_a_non_metadata_error_propagates(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(ne_estimate, "_inbreeding_from", boom)
        with pytest.raises(RuntimeError):
            estimate_effective_sizes(_chain_graph(), ["ne_inbreeding"])


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
                "ne_group_coancestry",
                {"observed_cohorts", "group_coancestry", "ne_group_coancestry"},
            ),
            (
                "ne_variance_family_size",
                {"observed_cohorts", "generation_family_table", "ne_variance_family_size"},
            ),
            (
                "ne_individual_delta_f",
                {"observed_cohorts", "inbreeding", "eqg", "ne_individual_delta_f"},
            ),
        ],
    )
    def test_a_single_selection_builds_exactly_its_closure(self, prerequisites, name, expected):
        estimate_effective_sizes(_chain_graph(), [name])
        assert prerequisites[0].computed() == expected

    def test_long_term_contributions_does_not_build_the_group_coancestry(self, prerequisites):
        estimate_effective_sizes(_chain_graph(), ["ne_long_term_contributions"])
        assert "group_coancestry" not in prerequisites[0].computed()

    def test_hill_without_birth_years_collapses_through_a_private_variance(self, prerequisites):
        result = estimate_effective_sizes(_chain_graph(), ["ne_hill_overlapping"])
        assert prerequisites[0].computed() == {
            "observed_cohorts",
            "generation_family_table",
            "ne_variance_family_size",
            "ne_hill_overlapping",
        }
        assert result["ne_variance_family_size"] == UnavailableEffectiveSize.not_requested()

    def test_hill_with_birth_years_skips_the_generation_cohorts(self, prerequisites):
        estimate_effective_sizes(_chain_graph(birth_year=CHAIN_BIRTH), ["ne_hill_overlapping"])
        assert prerequisites[0].computed() == {
            "generation_interval",
            "cohort_window",
            "birth_year_family_table",
            "ne_hill_overlapping",
        }

    def test_hill_and_variance_share_one_variance_computation(self, monkeypatch):
        calls = _count_calls(monkeypatch, "_variance_from")
        estimate_effective_sizes(_chain_graph(), ["ne_hill_overlapping", "ne_variance_family_size"])
        assert len(calls) == 1

    def test_group_coancestry_and_coancestry_share_one_kinship_summary(self, monkeypatch):
        """The two kinship estimators walk the pedigree once between them.

        ``_kinship_summary_for_labels`` is the single place either summary
        can run its DP or matrix walk, and since issue #25 the graph-label
        summary is itself genome-node, so the group-coancestry prerequisite
        reads the memo ``ne_coancestry`` fills instead of masking a private
        one.  Two walks would mean the shared memo was missed, so one call
        is the proof.  This pedigree has no MZ twins; the twin-bearing case
        is ``test_ne_group_coancestry`` ::
        ``test_the_kinship_summary_is_shared_with_ne_coancestry``, which was
        the pedigree that used to pay for two.
        """
        calls = _count_calls(monkeypatch, "_kinship_summary_for_labels", module=ne_rates)
        estimate_effective_sizes(_chain_graph(), ["ne_coancestry", "ne_group_coancestry"])
        assert len(calls) == 1

    def test_a_failed_guard_memoizes_nothing_for_that_estimator(self, prerequisites):
        estimate_effective_sizes(_chain_graph(father=_ONE_PARENT_FATHER))
        computed = prerequisites[0].computed()
        assert computed.isdisjoint(_NEEDS_PARENTAGE)


class TestDirectParity:
    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    @pytest.mark.parametrize("birth_year", [None, CHAIN_BIRTH], ids=["no_birth_years", "birth_years"])
    def test_the_orchestrated_record_equals_the_direct_one(self, name, birth_year):
        pg = _chain_graph(birth_year=birth_year)
        assert estimate_effective_sizes(pg)[name] == _DIRECT[name](pg)

    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    def test_the_empty_graph_yields_real_records_without_an_estimate(self, name):
        value = estimate_effective_sizes(_empty_graph())[name]
        assert not isinstance(value, UnavailableEffectiveSize)
        assert value.ne is None


_LABELS = ("_require_complete_generation_labels",)
_LABELS_AND_SEX = (*_LABELS, "_require_complete_sex", "_warn_if_uniform_sex")

_DOCUMENTED_GUARDS = {
    "ne_inbreeding": _LABELS,
    "ne_coancestry": _LABELS,
    "ne_variance_family_size": _LABELS_AND_SEX,
    "ne_sex_ratio": _LABELS_AND_SEX,
    "ne_individual_delta_f": _LABELS,
    "ne_long_term_contributions": (*_LABELS, "_require_closed_parentage"),
    "ne_hill_overlapping": _LABELS_AND_SEX,
    "ne_group_coancestry": _LABELS,
}


class TestRegistryCoverage:
    """The registry is the dispatch, so its rows are worth reading directly."""

    def test_every_estimator_name_has_exactly_one_row(self):
        assert tuple(ne_estimate._REGISTRY) == ALL_EFFECTIVE_SIZE_ESTIMATORS

    def test_the_empty_graph_delegate_covers_the_same_names(self):
        assert tuple(ne_estimate._DIRECT) == ALL_EFFECTIVE_SIZE_ESTIMATORS

    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    def test_each_row_declares_the_documented_guards(self, name):
        """The rows restate the metadata dependency matrix on ``effective_size``.

        Prose and table can drift apart; this is the one place they are read
        side by side.  Hill appears here under its collapse branch, which
        inherits Ne_V's requirements; its birth-year branch is below.
        """
        guards = ne_estimate._REGISTRY[name].guards(_chain_graph())
        assert tuple(guard.__name__ for guard in guards) == _DOCUMENTED_GUARDS[name]

    def test_hills_birth_year_branch_drops_the_generation_labels(self):
        """It groups by birth year and never reads a label, so it must not refuse one."""
        guards = ne_estimate._REGISTRY["ne_hill_overlapping"].guards(_chain_graph(birth_year=CHAIN_BIRTH))
        assert tuple(guard.__name__ for guard in guards) == ("_require_complete_sex",)


class TestPathEquivalence:
    """The batch and standalone paths agree on every degenerate graph.

    Each estimator is reachable two ways, and each way wires its own guards
    and prerequisites: the standalone function does it inline, the
    orchestrator does it in ``_Prerequisites``.  A graph that fails more than
    one guard is where two wirings would part company, since only the first
    guard to run gets to name the refusal.

    Issue #19 asserted they already have.  They have not.  The standalone path
    reaches its guards through
    :meth:`~pedigree_graph._cohorts.ObservedCohorts.for_graph`, which runs
    ``_require_complete_generation_labels`` before it densifies, so building
    cohorts first *is* checking labels first.  Both paths refuse in the order
    labels, sex, then the estimator's own guard.

    This pins that agreement rather than the order itself, so a reshape of
    either wiring has to keep the two in step without freezing which guard
    happens to be checked where.  Swapping the two guards in ``_compute``
    moves eight of these cells, which is what makes the pin worth its
    runtime.
    """

    @pytest.mark.parametrize("name", ALL_EFFECTIVE_SIZE_ESTIMATORS)
    @pytest.mark.parametrize("case", list(_DEGENERATE), ids=list(_DEGENERATE))
    def test_both_paths_refuse_alike_and_warn_alike(self, name, case):
        overrides = _DEGENERATE[case]
        direct = _outcome(lambda: _DIRECT[name](_chain_graph(**overrides)))
        orchestrated = _outcome(lambda: estimate_effective_sizes(_chain_graph(**overrides), [name])[name])
        assert orchestrated == direct


class TestSerialization:
    def test_to_dict_is_a_plain_ordered_dict_of_plain_dicts(self):
        payload = estimate_effective_sizes(_chain_graph(birth_year=CHAIN_BIRTH)).to_dict()
        assert type(payload) is dict
        assert list(payload) == list(ALL_EFFECTIVE_SIZE_ESTIMATORS)
        assert all(type(value) is dict for value in payload.values())
        json.dumps(payload)

    def test_unavailable_entries_serialize_to_plain_field_dicts(self):
        payload = estimate_effective_sizes(_chain_graph(father=_ONE_PARENT_FATHER)).to_dict()
        not_requested = estimate_effective_sizes(_chain_graph(), []).to_dict()["ne_inbreeding"]
        assert set(not_requested) == {"reason", "code", "fields"}
        entry = payload["ne_long_term_contributions"]
        assert set(entry) == {"reason", "code", "fields"}
        assert type(entry["fields"]) is dict
        assert entry["fields"]["operation"] == "ne_long_term_contributions"
        assert entry["fields"]["unrepresented_parent_status"] == "missing"
        json.dumps(payload)


@pytest.mark.usefixtures("fresh_thread_state")
class TestThreadBudget:
    def test_a_call_commits_the_default_budget(self):
        estimate_effective_sizes(_chain_graph(), ["ne_sex_ratio"])
        assert _threads._STATE.committed == 1

    def test_a_call_commits_a_configured_budget(self):
        configure_threads(3)
        estimate_effective_sizes(_chain_graph(), ["ne_sex_ratio"])
        assert _threads._STATE.committed == 3

    def test_results_do_not_depend_on_the_budget(self):
        under_one = estimate_effective_sizes(_chain_graph(birth_year=CHAIN_BIRTH))
        _threads._reset_thread_state()
        configure_threads(3)
        under_three = estimate_effective_sizes(_chain_graph(birth_year=CHAIN_BIRTH))
        assert _threads._STATE.committed == 3
        assert under_one == under_three

    def test_selector_validation_precedes_the_commit(self):
        with pytest.raises(TypeError):
            estimate_effective_sizes(_chain_graph(), None)
        assert _threads._STATE.committed is None


@pytest.mark.parametrize(
    "kwargs",
    [{"n_threads": 2}, {"skip_ne_coancestry": True}, {"theta_per_gen": {}}],
    ids=["n_threads", "skip_ne_coancestry", "theta_per_gen"],
)
def test_injected_keywords_are_rejected(kwargs):
    with pytest.raises(TypeError):
        estimate_effective_sizes(_chain_graph(), **kwargs)


class TestWarningAttribution:
    """The uniform-sex notice names the caller, on whichever path fired it.

    ``_warn_if_uniform_sex`` sits at different call depths on the two paths,
    the orchestrator's memo adding frames the standalone call does not have,
    so a fixed ``stacklevel`` can only ever be right for one of them.  It used
    to be right for the standalone path and blame ``_Prerequisites.result``
    for the other, which tells a caller nothing about their own code.
    """

    @pytest.mark.parametrize(
        "call",
        [
            es.ne_sex_ratio,
            lambda pg: estimate_effective_sizes(pg, ["ne_sex_ratio"]),
        ],
        ids=["direct", "orchestrated"],
    )
    def test_the_uniform_sex_notice_points_outside_the_package(self, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(_chain_graph(sex=_UNIFORM_SEX))
        [record] = [r for r in caught if "pg.sex is uniform" in str(r.message)]
        assert Path(record.filename) == Path(__file__)


class TestHillFallbackWarningScope:
    """The Hill fallback hides only the duplicate uniform-sex notice."""

    def _run(self, monkeypatch, *, noisy):
        pg = _chain_graph(sex=_UNIFORM_SEX)
        if noisy:
            real = ne_estimate._variance_from

            def unrelated(cohorts, table):
                warnings.warn("nan encountered in the variance table", RuntimeWarning, stacklevel=2)
                return real(cohorts, table)

            monkeypatch.setattr(ne_estimate, "_variance_from", unrelated)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            estimate_effective_sizes(pg, ["ne_hill_overlapping"])
        return [str(record.message) for record in seen]

    def test_an_unrelated_runtime_warning_still_reaches_the_caller(self, monkeypatch):
        assert any("nan encountered" in message for message in self._run(monkeypatch, noisy=True))

    def test_the_duplicate_uniform_sex_notice_stays_hidden(self, monkeypatch):
        messages = self._run(monkeypatch, noisy=False)
        assert any(message.startswith("ne_hill_overlapping: pg.sex is uniform") for message in messages)
        assert not any(message.startswith("ne_variance_family_size: pg.sex is uniform") for message in messages)
