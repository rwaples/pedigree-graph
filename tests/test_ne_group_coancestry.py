"""Group-coancestry Ne (Ne_GC): Caballero & Toro 2000 eq. 3 on the genome-node pedigree.

The per-cohort prerequisite is checked against a dense ``Σ_i Σ_j φ_ij / n²``
oracle built from the complete kinship matrix, the baseline against the
``f̄₀ = 1/(2N)`` the paper states for unrelated founders, and the genome-node
collapse against the same pedigree with the co-twin row deleted outright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from _support import _assert_owned_read_only, _assert_plain_python, _build_closed_line, _df, _random_mating

from pedigree_graph import MissingMetadataError, PedigreeGraph
from pedigree_graph._cohorts import ObservedCohorts
from pedigree_graph._ne_group_coancestry import (
    _genome_node_labels,
    _group_coancestry_by_cohort,
    _group_coancestry_from,
    ne_group_coancestry,
)
from pedigree_graph.effective_size import ne_coancestry, ne_long_term_contributions

if TYPE_CHECKING:
    from collections.abc import Sequence


def _with_mz_pair(frame: pl.DataFrame, generation: int) -> pl.DataFrame:
    """Make the first two rows of *generation* an MZ pair, by copying parents onto the second.

    The two rows already share a sex in :func:`_random_mating`, so only their
    parentage has to be reconciled for the package's MZ identity checks.
    """
    rows = frame.to_dicts()
    cohort = [r for r in rows if r["generation"] == generation]
    first, second = cohort[0], cohort[1]
    second["mother"], second["father"] = first["mother"], first["father"]
    first["twin"], second["twin"] = second["id"], first["id"]
    return pl.DataFrame(rows)


def _without(frame: pl.DataFrame, drop_id: int) -> pl.DataFrame:
    return frame.filter(pl.col("id") != drop_id).with_columns(pl.lit(-1).cast(pl.Int64).alias("twin"))


def _dense_group_coancestry(pg: PedigreeGraph) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 3 summed over the dense kinship matrix, genome-node representatives only.

    ``f̄ = Σ_i Σ_j a_ij / 2N²`` with ``a_ij = 2φ_ij`` off the diagonal and
    ``a_ii = 1 + F_i`` on it is ``Σ_i Σ_j φ'_ij / n²`` where ``φ'`` is the
    kinship matrix with ``(1 + F)/2`` written onto its diagonal.
    """
    phi = pg.kinship_matrix().toarray().astype(np.float64)
    np.fill_diagonal(phi, (1.0 + np.asarray(pg.inbreeding(), dtype=np.float64)) / 2.0)
    labels = _genome_node_labels(pg)
    observed = np.unique(labels[labels >= 0])
    f_bar = np.array(
        [phi[np.ix_(rows, rows)].sum() / rows.shape[0] ** 2 for rows in (np.flatnonzero(labels == g) for g in observed)]
    )
    return observed, f_bar


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        ("no_twins", _random_mating(6, 4, 11)),
        ("mz_mid_cohort", _with_mz_pair(_random_mating(6, 4, 11), 2)),
        ("no_twins_wide", _random_mating(20, 5, 7)),
        ("mz_first_cohort", _with_mz_pair(_random_mating(20, 5, 7), 1)),
        ("closed_line", _build_closed_line(5)),
    ],
)
def test_per_cohort_group_coancestry_matches_the_dense_eq_3_sum(name, frame):
    pg = PedigreeGraph.from_frame(frame)
    result = ne_group_coancestry(pg)

    observed, expected = _dense_group_coancestry(pg)

    np.testing.assert_array_equal(result.generations, observed)
    np.testing.assert_allclose(result.mean_group_coancestry_per_gen, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("cached_matrix", [False, True], ids=["dp", "matrix"])
def test_the_two_kinship_summary_routes_agree(cached_matrix):
    """The cached-matrix walk and the streamed DP are interchangeable oracles.

    The estimator does not choose between them —
    :func:`~pedigree_graph._ne_rates._kinship_summary_for_labels` does, off
    whether the graph already holds a complete kinship matrix — so a caller
    that happened to ask for one first must get the same numbers.
    """
    frame = _with_mz_pair(_random_mating(20, 5, 7), 3)
    pg = PedigreeGraph.from_frame(frame)
    if cached_matrix:
        pg.kinship_matrix()

    result = ne_group_coancestry(pg)

    observed, expected = _dense_group_coancestry(PedigreeGraph.from_frame(frame))
    np.testing.assert_array_equal(result.generations, observed)
    np.testing.assert_allclose(result.mean_group_coancestry_per_gen, expected, rtol=0.0, atol=1e-12)


def test_the_closed_line_matches_the_hand_derived_eq_3_values():
    """Eq. 3 on a 2-per-cohort full-sib line, derived by hand rather than from the package.

    The dense oracle above reads the package's own kinship matrix, so it
    cannot catch a kinship error the two share.  These four values come from
    the full-sib recursion instead: ``F = 0, 0, 0.25, 0.375`` down the line
    (``tests/test_effective_size.py::test_toy4_closed_line_F_recursion``), and
    within-cohort φ of ``0, 0.25, 0.375, 0.5``, giving
    ``f̄ = (2φ + (1 + F)) / 4`` per cohort.
    """
    pg = PedigreeGraph.from_frame(_build_closed_line(3))

    result = ne_group_coancestry(pg)

    assert list(result.n_genomes_per_gen) == [2, 2, 2, 2]
    assert list(result.mean_group_coancestry_per_gen) == [0.25, 0.375, 0.5, 0.59375]


@pytest.mark.parametrize(("n_gens", "expected"), [(4, 2.3357), (5, 2.3446), (8, 2.3534)])
def test_the_scalar_lands_on_the_full_sib_chain_rate(n_gens, expected):
    """The reported Ne, not just the series behind it, is pinned to a value.

    A full-sib mating chain has asymptotic ``Ne ≈ 2.62`` (the eigenvalue
    ``(1 + √5)/4 ≈ 0.809`` of the F recursion), which the regression
    approaches from below on a finite chain the way
    ``test_toy4_closed_line_F_recursion`` bands ``ne_inbreeding``.  Pinning
    the value as well as the band is what separates this reduction from one
    off by a factor of two, a sign, or the wrong series.
    """
    result = ne_group_coancestry(PedigreeGraph.from_frame(_build_closed_line(n_gens)))

    assert 1.5 < result.ne < 4.0
    assert result.ne == pytest.approx(expected, abs=1e-4)
    assert result.slope < 0.0
    assert result.n_generations_used == n_gens


@pytest.mark.parametrize("n_per_gen", [6, 20, 60])
def test_the_baseline_cohort_is_the_computed_one_over_two_n(n_per_gen):
    """Unrelated non-inbred founders give C&T's own ``f̄₀ = 1/(2N)``, exactly.

    Eq. 3 over ``N`` unrelated non-inbred individuals keeps only the ``N``
    diagonal terms, each ``(1 + 0)/2``, over ``N²``.  The baseline is
    therefore computed like every other cohort rather than assumed, and it
    lands on the value the paper states for unrelated founders — which is
    what makes the first ``ne_per_gen`` entry their eq. 11 ``Δf₀,₁``.
    """
    pg = PedigreeGraph.from_frame(_random_mating(n_per_gen, 4, 2026))
    result = ne_group_coancestry(pg)

    assert int(result.n_genomes_per_gen[0]) == n_per_gen
    assert result.mean_group_coancestry_per_gen[0] == 1.0 / (2.0 * n_per_gen)


def test_the_first_transition_reads_from_the_computed_baseline_not_a_half():
    """``ne_per_gen[0]`` is eq. 11 off ``f̄₀ = 1/(2N)``, not off a ``0.5`` self-coancestry.

    ``0.5`` is the self-coancestry of one non-inbred individual, a
    per-individual quantity; group coancestry over ``N`` unrelated founders
    is ``1/(2N)``.  Anchoring the first transition at ``0.5`` would report a
    rate two orders of magnitude off here, so the two are never within
    rounding of each other.
    """
    n_per_gen = 20
    pg = PedigreeGraph.from_frame(_random_mating(n_per_gen, 4, 2026))
    result = ne_group_coancestry(pg)

    f_0 = 1.0 / (2.0 * n_per_gen)
    f_1 = float(result.mean_group_coancestry_per_gen[1])
    assert result.ne_per_gen[0] == pytest.approx(1.0 / (2.0 * (f_1 - f_0) / (1.0 - f_0)), rel=1e-12)

    from_a_half = 1.0 / (2.0 * (f_1 - 0.5) / (1.0 - 0.5))
    assert not np.isclose(result.ne_per_gen[0], from_a_half)


def test_an_mz_pair_collapses_to_one_genome_node():
    """The co-twin is masked out, so the cohort counts genomes rather than rows."""
    frame = _with_mz_pair(_random_mating(20, 5, 7), 5)
    pg = PedigreeGraph.from_frame(frame)

    result = ne_group_coancestry(pg)

    assert int(pg.n_individuals) == 20 * 6
    assert list(result.n_genomes_per_gen) == [20, 20, 20, 20, 20, 19]


def test_collapsing_an_mz_pair_equals_deleting_the_co_twin_row():
    """A genome counted once is a genome counted once, however the frame spells it.

    The co-twin here is childless, so dropping its row leaves the same
    genomes with the same ancestry and the estimator has to agree to the
    bit — that is what makes the mask a collapse rather than a reweighting.
    """
    frame = _with_mz_pair(_random_mating(20, 5, 7), 5)
    co_twin = int(frame.filter(pl.col("twin") >= 0)["id"].max())

    collapsed = ne_group_coancestry(PedigreeGraph.from_frame(frame))
    deleted = ne_group_coancestry(PedigreeGraph.from_frame(_without(frame, co_twin)))

    np.testing.assert_array_equal(collapsed.mean_group_coancestry_per_gen, deleted.mean_group_coancestry_per_gen)
    np.testing.assert_array_equal(collapsed.n_genomes_per_gen, deleted.n_genomes_per_gen)
    assert collapsed.ne == deleted.ne


@pytest.mark.parametrize("twins", [0, 2])
def test_the_kinship_summary_is_shared_with_ne_coancestry(twins):
    """One genome-node summary serves both estimators, twins or not (issue #25).

    ``_generation_kinship_summary`` is the sole writer of the graph memo, so
    its presence afterwards is the evidence that the pedigree paid for one DP
    pass and not a second private one.  Before #25 a pedigree with MZ twins
    paid twice, because the two estimators disagreed on the MZ convention.
    """
    frame = _random_mating(20, 4, 5)
    pg = PedigreeGraph.from_frame(frame if twins == 0 else _with_mz_pair(frame, twins))
    assert pg._generation_kinship_summary is None

    ne_group_coancestry(pg)

    assert pg._generation_kinship_summary is not None
    assert pg._generation_kinship_summary is pg.mean_kinship_by_generation()
    assert ne_coancestry(pg).mean_theta_per_gen[2] == pytest.approx(
        float(pg.mean_kinship_by_generation().mean_kinship[2]), rel=1e-12
    )


def test_group_coancestry_does_not_require_closed_parentage():
    """It reads no founder contributions, so a half-represented parentage is fine.

    ``ne_long_term_contributions`` halves each represented parent into the
    child and so rejects the same graph; this estimator only ever reads
    kinship and F, both defined row by row.
    """
    columns = {
        "id": np.arange(8),
        "mother": np.array([-1, -1, 0, 0, 2, 2, 4, 4]),
        "father": np.array([-1, -1, 1, 1, 3, -1, 5, 5]),
        "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "generation": np.array([0, 0, 1, 1, 2, 2, 3, 3]),
    }
    pg = PedigreeGraph.from_frame(columns)

    with pytest.raises(MissingMetadataError) as info:
        ne_long_term_contributions(pg)
    assert info.value.code == "incomplete_parentage"

    result = ne_group_coancestry(pg)
    assert result.ne is not None
    assert list(result.n_genomes_per_gen) == [2, 2, 2, 2]


def test_co_twins_in_different_cohorts_keep_the_cohort_alignment():
    """Masking can empty a cohort, and the record still describes the graph's cohorts.

    The package enforces MZ reciprocity, parent identity and sex identity but
    not label identity, so co-twins with different generation labels are
    constructible.  Masking is unconditional, so the higher-labelled cohort
    here loses its only row and drops out of the masked summary; the record
    reports it as an empty cohort rather than losing the label.
    """
    pg = PedigreeGraph.from_frame(
        _df(
            [
                {"id": 0, "sex": 1, "generation": 0},
                {"id": 1, "sex": 0, "generation": 0},
                {"id": 2, "sex": 1, "generation": 1, "mother": 1, "father": 0, "twin": 3},
                {"id": 3, "sex": 1, "generation": 2, "mother": 1, "father": 0, "twin": 2},
            ]
        )
    )

    result = ne_group_coancestry(pg)

    np.testing.assert_array_equal(result.generations, np.unique(pg.generation_labels))
    assert list(result.n_genomes_per_gen) == [2, 1, 0]
    assert np.isnan(result.mean_group_coancestry_per_gen[2])
    assert result.mean_group_coancestry_per_gen[0] == 0.25


def test_an_empty_graph_yields_no_estimate():
    """Same shape as the sibling rate estimators: no estimate, no warning, empty series."""
    pg = PedigreeGraph.from_arrays(ids=[], mother_ids=[], father_ids=[], sex=[])

    result = ne_group_coancestry(pg)
    sibling = ne_coancestry(pg)

    assert result.ne is sibling.ne is None
    assert result.n_generations_used == sibling.n_generations_used
    for name in ("generations", "mean_group_coancestry_per_gen", "n_genomes_per_gen", "ne_per_gen"):
        assert getattr(result, name).shape == (0,), name


def test_partly_unknown_generation_labels_disable_the_estimator():
    pg = PedigreeGraph.from_frame(
        {
            "id": np.arange(8),
            "mother": np.array([-1, -1, 0, 0, 2, 2, 4, 4]),
            "father": np.array([-1, -1, 1, 1, 3, 3, 5, 5]),
            "sex": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            "generation": np.array([0, 0, 1, 1, 2, 2, -1, -1]),
        }
    )
    with pytest.raises(MissingMetadataError) as info:
        ne_group_coancestry(pg)
    assert info.value.code == "missing_generation_labels"
    assert info.value.fields["operation"] == "ne_group_coancestry"


def test_absent_generation_labels_fall_back_to_depth():
    labelled = _build_closed_line(4)
    pg_labelled = PedigreeGraph.from_frame(labelled)
    pg_unlabelled = PedigreeGraph.from_frame(labelled.drop("generation"))

    assert pg_unlabelled.generation_labels is None
    assert ne_group_coancestry(pg_labelled).to_dict() == ne_group_coancestry(pg_unlabelled).to_dict()


def test_the_evaluator_rejects_a_prerequisite_for_other_cohorts():
    pg = PedigreeGraph.from_frame(_build_closed_line(4))
    cohorts = ObservedCohorts.for_graph(pg, "ne_group_coancestry")
    other = ObservedCohorts.from_labels(np.array([0, 0, 7, 7, 9, 9, 11, 11, 13, 13], dtype=np.int32))

    with pytest.raises(ValueError, match="observed cohorts"):
        _group_coancestry_from(other, _group_coancestry_by_cohort(pg, cohorts))


def test_result_arrays_are_owned_and_read_only():
    _assert_owned_read_only(ne_group_coancestry(PedigreeGraph.from_frame(_build_closed_line(4))))


def test_the_record_does_not_alias_its_prerequisite():
    pg = PedigreeGraph.from_frame(_build_closed_line(4))
    cohorts = ObservedCohorts.for_graph(pg, "ne_group_coancestry")
    gc = _group_coancestry_by_cohort(pg, cohorts)

    result = _group_coancestry_from(cohorts, gc)
    gc.mean_group_coancestry[0] = 99.0
    gc.n_genomes[0] = 99

    assert result.mean_group_coancestry_per_gen[0] != 99.0
    assert result.n_genomes_per_gen[0] != 99


def test_to_dict_returns_plain_python():
    result = ne_group_coancestry(PedigreeGraph.from_frame(_build_closed_line(4)))
    _assert_plain_python(result.to_dict(), "ne_group_coancestry")


def _varying_census(sizes: Sequence[int], seed: int) -> pl.DataFrame:
    """Closed random-mating pedigree whose cohort ``g`` holds ``sizes[g]`` rows.

    :func:`~test_effective_size._random_mating` fixes one census for every
    cohort, which is exactly the case ``census_ratio`` cannot distinguish.
    Mating is otherwise identical: each cohort after the first draws both
    parents uniformly from the previous one.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    next_id = 0
    previous_m: list[int] = []
    previous_f: list[int] = []
    for generation, n in enumerate(sizes):
        current_m: list[int] = []
        current_f: list[int] = []
        for j in range(n):
            sex = 1 if j < n // 2 else 0
            record = {"id": next_id, "sex": sex, "generation": generation}
            if generation > 0:
                record["mother"] = int(rng.choice(previous_f))
                record["father"] = int(rng.choice(previous_m))
            records.append(record)
            (current_m if sex == 1 else current_f).append(next_id)
            next_id += 1
        previous_m, previous_f = current_m, current_f
    return _df(records)


def _truncated_last_cohort(frame: pl.DataFrame, keep: int) -> pl.DataFrame:
    """The same pedigree with all but *keep* rows of the final cohort dropped.

    Every earlier cohort keeps its rows, its parents and its drift, so the
    only difference between the two graphs is the final census.
    """
    last = frame["generation"].max()
    return pl.concat(
        [frame.filter(pl.col("generation") != last), frame.filter(pl.col("generation") == last).head(keep)]
    )


def _with_a_masked_trailing_cohort(frame: pl.DataFrame) -> pl.DataFrame:
    """Append a cohort holding nothing but the MZ co-twin of the last row.

    The co-twin is the higher row of the pair, so the genome-node mask sends
    it to the sentinel bucket and the appended cohort ends up with no
    representative at all: a census of zero and a NaN f̄.
    """
    rows = frame.to_dicts()
    last = rows[-1]
    co_twin_id = max(int(r["id"]) for r in rows) + 1
    last["twin"] = co_twin_id
    rows.append({**last, "id": co_twin_id, "generation": last["generation"] + 1, "twin": last["id"]})
    return pl.DataFrame(rows)


def _fitted_census(result) -> np.ndarray:
    """The censuses the ``ln(1 − f̄)`` fit actually saw, derived from the record alone."""
    f_bar = np.asarray(result.mean_group_coancestry_per_gen)[1:]
    return np.asarray(result.n_genomes_per_gen)[1:][np.isfinite(f_bar) & (f_bar < 1.0)]


def test_census_ratio_is_one_when_the_census_never_changes():
    """The constant-census case the scalar is faithful on reports exactly 1."""
    result = ne_group_coancestry(PedigreeGraph.from_frame(_varying_census((40,) * 6, 2026)))

    assert list(result.n_genomes_per_gen) == [40] * 6
    assert result.census_ratio == 1.0


@pytest.mark.parametrize(
    ("name", "sizes", "expected"),
    [
        ("growing", (10, 16, 22, 28, 34, 40), 40 / 16),
        ("declining", (40, 33, 26, 19, 11, 4), 33 / 4),
    ],
)
def test_census_ratio_is_the_max_over_min_of_the_fitted_cohorts(name, sizes, expected):
    """It reads the cohorts the fit used, which excludes the baseline.

    The baseline cohort sets the intercept and contributes no ``ln(1 − f̄)``
    term, so a pedigree that grows from 10 to 40 reports ``40/16`` and not
    ``40/10``.  Reading it off the record's own arrays rather than off
    *sizes* keeps the oracle independent of the builder.
    """
    result = ne_group_coancestry(PedigreeGraph.from_frame(_varying_census(sizes, 2026)))

    fitted = _fitted_census(result)
    assert list(fitted) == list(sizes[1:])
    assert result.census_ratio == pytest.approx(fitted.max() / fitted.min())
    assert result.census_ratio == pytest.approx(expected)


def test_census_ratio_and_n_generations_used_describe_the_same_cohorts():
    """A cohort the fit drops leaves the count and the ratio alike.

    Masking can empty a cohort, giving it a NaN f̄ and a census of zero.  The
    two fields are derived from one predicate, so that cohort is absent from
    both: the count does not include it and the ratio does not divide by it.
    """
    frame = _with_a_masked_trailing_cohort(_varying_census((20, 20, 30, 30, 40, 40), 7))
    result = ne_group_coancestry(PedigreeGraph.from_frame(frame))

    assert list(result.n_genomes_per_gen) == [20, 20, 30, 30, 40, 40, 0]
    assert np.isnan(result.mean_group_coancestry_per_gen[-1])
    assert result.n_generations_used == len(_fitted_census(result)) == 5
    assert result.census_ratio == pytest.approx(2.0)


def test_a_lone_trailing_cohort_collapses_ne_while_ne_coancestry_holds():
    """What ``census_ratio`` exists to warn about, on a pedigree that is otherwise unchanged.

    Truncating the final cohort to one individual leaves every earlier
    cohort, its parents and its drift untouched, so no real rate has moved.
    But ``f̄`` over one genome is that genome's own self-coancestry, near
    ``0.5``, against ``0.057`` for the cohort it replaces — the ``s̄_g/n_g``
    term of ``f̄_g = θ̄_g·(n_g − 1)/n_g + s̄_g/n_g`` with ``n_g = 1`` — and the
    ``ln(1 − f̄)`` fit reads that jump as a generation of drift.  Ne falls
    from 43.2 to 3.5.  ``ne_coancestry`` reads the same pedigree without the
    diagonal term, finds no within-cohort pair there at all, and holds at
    40.5.
    """
    frame = _varying_census((40,) * 6, 2026)
    full = PedigreeGraph.from_frame(frame)
    trailing = PedigreeGraph.from_frame(_truncated_last_cohort(frame, 1))

    intact = ne_group_coancestry(full)
    damaged = ne_group_coancestry(trailing)

    assert intact.census_ratio == 1.0
    assert damaged.census_ratio == 40.0
    assert intact.ne == pytest.approx(43.22, abs=0.01)
    assert damaged.ne == pytest.approx(3.53, abs=0.01)
    assert damaged.mean_group_coancestry_per_gen[-1] == pytest.approx(0.5127, abs=1e-4)

    assert ne_coancestry(full).ne == pytest.approx(43.14, abs=0.01)
    assert ne_coancestry(trailing).ne == pytest.approx(40.55, abs=0.01)
