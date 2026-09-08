"""Regressions the slice 6c review found."""

from __future__ import annotations

import numpy as np
import pytest

from pedigree_graph import PedigreeGraph
from pedigree_graph import effective_size as es
from pedigree_graph._ne_common import _scalar_ne_from_log_regression


def _closed_line(n_gens: int, labels: list[int] | None = None) -> PedigreeGraph:
    ids, mother, father, sex, gen = [0, 1], [-1, -1], [-1, -1], [1, 0], [0, 0]
    prev = (0, 1)
    for g in range(1, n_gens + 1):
        m, f = len(ids), len(ids) + 1
        ids += [m, f]
        mother += [prev[1], prev[1]]
        father += [prev[0], prev[0]]
        sex += [1, 0]
        gen += [g, g]
        prev = (m, f)
    if labels is not None:
        gen = [labels[g] for g in gen]
    return PedigreeGraph.from_frame({"id": ids, "mother": mother, "father": father, "sex": sex, "generation": gen})


def _non_inbred() -> PedigreeGraph:
    return PedigreeGraph.from_frame(
        {
            "id": list(range(8)),
            "mother": [-1, -1, -1, -1, 0, 0, 2, 2],
            "father": [-1, -1, -1, -1, 1, 1, 3, 3],
            "sex": [0, 1, 0, 1, 0, 1, 0, 1],
            "generation": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_flat_series_regresses_to_no_estimate():
    assert _scalar_ne_from_log_regression(np.array([np.nan, 0.5, 0.5, 0.5]), np.arange(4))[0] is None
    assert _scalar_ne_from_log_regression(np.array([0.0, 0.1, 0.19, 0.271]), np.arange(4))[0] is not None


@pytest.mark.parametrize("labels", [None, [10, 12, 15]])
def test_non_inbred_pedigree_has_no_rate_estimate(labels):
    pg = _closed_line(2, labels) if labels else _non_inbred()
    for estimator in (es.ne_coancestry, es.ne_caballero_toro):
        res = estimator(pg)
        assert res.ne is None or res.ne < 1e9, (estimator.__name__, res.ne)


def _empty() -> PedigreeGraph:
    return PedigreeGraph.from_frame({"id": [], "mother": [], "father": []})


@pytest.mark.parametrize("build", [_non_inbred, _empty])
def test_vk_scaled_records_the_request_on_every_branch(build):
    pg = build()
    assert es.ne_hill_overlapping(pg, vk_scale=True).vk_scaled is True
    assert es.estimate_effective_sizes(pg, ["ne_hill_overlapping"], hill_vk_scale=True)["ne_hill_overlapping"].vk_scaled


def test_generation_kinship_summaries_compare_and_do_not_hash():
    a = _closed_line(3).mean_kinship_by_generation()
    b = _closed_line(3).mean_kinship_by_generation()
    assert a == b
    assert a != _closed_line(2).mean_kinship_by_generation()
    with pytest.raises(TypeError):
        hash(a)


def test_coancestry_on_an_empty_graph_reports_zero_length_arrays():
    empty = _empty()
    for result in (es.ne_coancestry(empty), es.estimate_effective_sizes(empty)["ne_coancestry"]):
        assert result.ne is None
        assert result.ne_per_gen.shape == (0,)
        assert result.mean_theta_per_gen.shape == (0,)


def test_an_unselected_coancestry_key_carries_no_record():
    unavailable = es.estimate_effective_sizes(_empty(), ["ne_inbreeding"])["ne_coancestry"]
    assert unavailable == es.UnavailableEffectiveSize.not_requested()
