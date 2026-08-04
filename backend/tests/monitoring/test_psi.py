"""Tests for the Population Stability Index detector (P12.2).

:mod:`backend.monitoring.psi` §1 names this file as the place the metric's
classic silent failure is pinned. That failure — recomputing the bin edges from
each period's own sample — is the organising idea of the suite, because it is
the one that leaves a detector *reporting*, in the right units, on a schedule,
forever saying "no drift".

Six claims, each a place PSI is quietly got wrong, each asserted in both
directions where it has two:

1. **Detection works both ways.** A detector that always fires is exactly as
   useless as one that never does, and a suite that only shifts the
   distribution catches neither. So every detection test has a matching
   no-drift test drawn from the *same* generator with a different seed, and
   :func:`test_no_drift_and_drift_are_separated_across_every_seed` asserts the
   two populations do not overlap — the strongest form available without
   claiming a distributional result the module does not have.

2. **Bins are fixed at reference time.** Asserted three ways, because each
   alone is weak. *Behaviourally*, the reference's edges, counts, fractions and
   fingerprint are identical before and after measuring wildly different
   samples. *Structurally*, no public callable in the module accepts two
   array-like samples, so there is no surface through which per-period binning
   could be requested. And by *counterexample*: :func:`_psi_with_per_period_bins`
   below is the broken implementation — the one that bins each sample by its own
   quantiles — and it is asserted to return exactly ``0.0`` on a fifty-sigma
   shift, which is what makes the passing assertions evidence rather than
   decoration. It is the same device
   ``backend/tests/features/test_transforms_properties.py`` uses for cross-date
   leakage, and for the same reason: a property only worth asserting is one that
   something plausible fails.

3. **Empty bins.** ``(q - p) ln(q/p)`` is undefined when a bin empties, which is
   the *normal* consequence of real drift. The implemented policy — floor both
   fractions at ``epsilon`` — is checked against the closed form
   ``(epsilon - 0.1) * ln(10 * epsilon)`` on a reference whose deciles hold
   exactly 2,000 observations each, so the assertion is arithmetic rather than a
   previously observed number. The floor's size is then asserted to be *visible*:
   named bins, a separated contribution, and the epsilon itself, all in
   ``to_dict()``.

4. **Small samples raise rather than report.** Including at the exact boundary,
   in both directions: ``minimum_sample_size(B) - 1`` refuses and
   ``minimum_sample_size(B)`` measures.

5. **A change in the NaN rate is drift.** Asserted in the form that matters: a
   fixture whose surviving names are drawn from the reference distribution
   itself, so the distributional PSI is at its noise floor while 40% of the
   universe silently stops having a value. An implementation that dropped
   ``NaN`` would report that fixture as perfectly stable.

6. **No data raises; it never returns 0.0.** Behaviourally for the empty and
   all-absent cross-sections, and structurally by asserting the entry point has
   no return path that produces a constant.

Every array in this file is synthetic and named so (I3 —
``backend/tests/monitoring/fixtures.py``). No number here is a measurement of a
feature; they are measurements of the arithmetic.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from backend.monitoring import psi as psi_module
from backend.monitoring.errors import (
    InsufficientSampleError,
    MonitoringInputError,
    ReferenceDistributionError,
)
from backend.monitoring.psi import (
    AVAILABILITY_CATEGORIES,
    DEFAULT_BINS,
    DEFAULT_EPSILON,
    MINIMUM_BINS,
    NULL_PSI_NOISE_BUDGET,
    ReferenceDistribution,
    minimum_sample_size,
    population_stability_index,
)
from backend.tests.monitoring.fixtures import (
    FIXTURE_UNITS,
    OBSERVED_SIZE,
    REFERENCE_SEED,
    REFERENCE_SIZE,
    equal_decile_reference,
    fixture_stamp,
    gaussian_reference,
    gaussian_sample,
    with_absent,
)

if TYPE_CHECKING:
    from backend.monitoring.psi import FloatArray

SEEDS: Final = tuple(range(1000, 1040))
"""Forty independent live cross-sections per condition. Fixed, so a failure is
reproducible and a bound that holds here is a bound, not a coin flip."""

MODERATE: Final = 0.10
MAJOR: Final = 0.25
"""The conventional ladder, quoted here only so the detection assertions can be
read against the numbers an operator would see. :mod:`backend.monitoring.drift`
owns them; this module knows nothing about them."""


# ---------------------------------------------------------------------------
# The broken implementation, kept as a counterexample (claim 2)
# ---------------------------------------------------------------------------


def _own_quantile_fractions(sample: FloatArray, n_bins: int) -> FloatArray:
    """Return ``sample``'s share of each of *its own* quantile bins."""
    probabilities = [index / n_bins for index in range(1, n_bins)]
    edges = np.unique(np.quantile(sample, probabilities))
    counts, _ = np.histogram(sample, bins=[-np.inf, *edges, np.inf])
    return np.asarray(counts / sample.size, dtype=np.float64)


def _psi_with_per_period_bins(
    reference_sample: FloatArray,
    observed_sample: FloatArray,
    *,
    n_bins: int = DEFAULT_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI as it is written when the bins are recomputed every period.

    This is the mistake, written out. It takes two raw samples — the shape of
    API :mod:`backend.monitoring.psi` deliberately does not offer — and bins
    each by its own quantiles. Quantile edges put ``1/B`` of their own sample in
    each bin by construction, so ``p_i == q_i == 1/B`` whatever either
    distribution did, and the sum is identically zero.

    It exists so the suite can assert that it *fails*: a detector test that no
    plausible wrong implementation fails is not evidence.
    """
    p = np.maximum(_own_quantile_fractions(reference_sample, n_bins), epsilon)
    q = np.maximum(_own_quantile_fractions(observed_sample, n_bins), epsilon)
    return float(np.sum((q - p) * np.log(q / p)))


# ---------------------------------------------------------------------------
# 1. Both directions of detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_an_unshifted_sample_stays_at_the_noise_floor(seed: int) -> None:
    reference = gaussian_reference()
    result = population_stability_index(reference=reference, observed=gaussian_sample(seed=seed))
    # Far below the conventional "moderate" rung, and of the order of the null
    # expectation for this sample size — (B-1)/n = 9/2000 = 0.0045 — rather than
    # merely "small".
    assert result.value < 0.02
    assert result.value < MODERATE
    assert result.value < 6.0 * result.null_expected_value
    assert result.null_expected_value == pytest.approx(9 / OBSERVED_SIZE)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_one_sigma_shift_clears_the_major_threshold(seed: int) -> None:
    reference = gaussian_reference()
    result = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=seed, shift=1.0)
    )
    assert result.value > 0.8
    assert result.value > MAJOR


@pytest.mark.parametrize("seed", SEEDS)
def test_a_half_sigma_shift_clears_the_moderate_threshold(seed: int) -> None:
    reference = gaussian_reference()
    result = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=seed, shift=0.5)
    )
    assert result.value > MODERATE


@pytest.mark.parametrize("scale", [0.5, 2.0])
@pytest.mark.parametrize("seed", SEEDS[:10])
def test_a_pure_dispersion_change_is_detected_with_the_mean_unmoved(
    seed: int, scale: float
) -> None:
    # The shape a vendor unit change takes: the location is untouched, so a
    # monitor watching only the cross-sectional mean sees nothing. Recentred on
    # the training sample's own mean so "untouched" is exact rather than likely.
    reference = gaussian_reference()
    training = gaussian_sample(seed=REFERENCE_SEED, size=REFERENCE_SIZE)
    drawn = gaussian_sample(seed=seed, scale=scale)
    observed = drawn - float(np.mean(drawn)) + float(np.mean(training))
    assert float(np.mean(observed)) == pytest.approx(float(np.mean(training)), abs=1e-12)
    assert population_stability_index(reference=reference, observed=observed).value > MAJOR


def test_no_drift_and_drift_are_separated_across_every_seed() -> None:
    reference = gaussian_reference()

    def value(seed: int, shift: float) -> float:
        return population_stability_index(
            reference=reference, observed=gaussian_sample(seed=seed, shift=shift)
        ).value

    quiet = [value(seed, 0.0) for seed in SEEDS]
    moved = [value(seed, 1.0) for seed in SEEDS]
    # The two populations must not touch, and the gap must be wide: a detector
    # whose "drift" and "no drift" distributions overlap has no usable threshold
    # whatever its individual values look like.
    assert max(quiet) < MODERATE < MAJOR < min(moved)
    assert min(moved) > 40.0 * max(quiet)


def test_measuring_the_reference_against_itself_is_exactly_zero() -> None:
    reference = equal_decile_reference()
    result = population_stability_index(
        reference=reference, observed=np.arange(20_000, dtype=np.float64)
    )
    assert result.value == 0.0
    assert result.observed_fractions == reference.fractions


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
def test_psi_is_non_negative_and_grows_with_the_shift(shift: float) -> None:
    reference = gaussian_reference()
    ladder = [
        population_stability_index(
            reference=reference, observed=gaussian_sample(seed=4242, shift=step)
        ).value
        for step in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    ]
    assert all(value >= 0.0 for value in ladder)
    assert ladder == sorted(ladder)
    index = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0).index(shift)
    assert ladder[index] >= 0.0


def test_psi_is_invariant_to_the_order_of_the_cross_section() -> None:
    reference = gaussian_reference()
    observed = gaussian_sample(seed=99, shift=0.7)
    shuffled = np.random.default_rng(7).permutation(observed)
    first = population_stability_index(reference=reference, observed=observed)
    second = population_stability_index(reference=reference, observed=shuffled)
    assert first.value == second.value


# ---------------------------------------------------------------------------
# 2. Bins are fixed at reference time
# ---------------------------------------------------------------------------


def test_recomputing_bins_per_period_destroys_the_detector() -> None:
    """The pinned failure mode (:mod:`backend.monitoring.psi` §1)."""
    training = gaussian_sample(seed=20240101, size=20_000)
    reference = gaussian_reference()
    for shift in (0.0, 1.0, 2.0, 5.0, 50.0):
        observed = gaussian_sample(seed=77, shift=shift)
        measured = population_stability_index(reference=reference, observed=observed).value
        collapsed = _psi_with_per_period_bins(training, observed)
        # The broken form does not merely under-report: it returns exactly zero,
        # because each sample's own quantile bins each hold 1/B of it.
        assert collapsed == pytest.approx(0.0, abs=1e-12)
        if shift:
            assert measured > MAJOR
    # And the collapse is total: fifty sigma is reported as identically stable.
    far = gaussian_sample(seed=77, shift=50.0)
    assert _psi_with_per_period_bins(training, far) == pytest.approx(0.0, abs=1e-12)
    assert population_stability_index(reference=reference, observed=far).value > 10.0


def test_the_reference_is_unchanged_by_measuring_against_it() -> None:
    reference = gaussian_reference()
    before = (
        reference.edges,
        reference.bin_counts,
        reference.fractions,
        reference.fingerprint,
        reference.n_absent,
        reference.epsilon,
    )
    for shift in (0.0, 3.0, -3.0):
        for scale in (0.2, 1.0, 5.0):
            population_stability_index(
                reference=reference, observed=gaussian_sample(seed=11, shift=shift, scale=scale)
            )
    after = (
        reference.edges,
        reference.bin_counts,
        reference.fractions,
        reference.fingerprint,
        reference.n_absent,
        reference.epsilon,
    )
    assert before == after


def test_every_result_reports_the_reference_edges_and_fractions_verbatim() -> None:
    reference = gaussian_reference()
    for shift in (0.0, 2.5):
        result = population_stability_index(
            reference=reference, observed=gaussian_sample(seed=13, shift=shift)
        )
        assert tuple(bin_.lower for bin_ in result.bins) == reference.edges[:-1]
        assert tuple(bin_.upper for bin_ in result.bins) == reference.edges[1:]
        assert tuple(bin_.reference_count for bin_ in result.bins) == reference.bin_counts
        assert tuple(bin_.reference_fraction for bin_ in result.bins) == reference.fractions


def test_two_periods_are_summed_over_the_same_partition() -> None:
    # Comparability across dates is the property per-period binning destroys,
    # and it is a statement about the partition, not about the values.
    reference = gaussian_reference()
    early = population_stability_index(reference=reference, observed=gaussian_sample(seed=1))
    late = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=2, shift=1.0)
    )
    assert early.reference.fingerprint == late.reference.fingerprint
    assert [bin_.lower for bin_ in early.bins] == [bin_.lower for bin_ in late.bins]
    assert [bin_.reference_fraction for bin_ in early.bins] == [
        bin_.reference_fraction for bin_ in late.bins
    ]


def test_no_public_entry_point_accepts_two_raw_samples() -> None:
    """Structural: per-period binning is unreachable, not merely unused."""
    source = ast.parse(Path(inspect.getsourcefile(psi_module) or "").read_text(encoding="utf-8"))
    array_like_parameters: dict[str, list[str]] = {}
    for node in ast.walk(source):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        array_like = [
            argument.arg
            for argument in arguments
            if argument.annotation is not None and "ArrayLike" in ast.unparse(argument.annotation)
        ]
        array_like_parameters[node.name] = array_like
    assert array_like_parameters, "no public functions found; the scan is not looking at psi.py"
    for name, parameters in array_like_parameters.items():
        assert len(parameters) <= 1, (
            f"{name} takes {parameters}; a public entry point accepting two raw samples is "
            f"the surface through which bins get recomputed per period"
        )
    signature = inspect.signature(population_stability_index)
    assert list(signature.parameters) == ["reference", "observed"]
    assert signature.parameters["reference"].annotation == "ReferenceDistribution"


def test_the_fingerprint_distinguishes_two_references_with_the_same_name() -> None:
    # Reference identifiers get reused across re-cuts. Two PSI values quoted
    # under one identifier and different edges look like a time series and are
    # not one, so the digest — not the name — is what establishes comparability.
    ten = gaussian_reference(reference_id="FIXTURE_same_name")
    five = gaussian_reference(reference_id="FIXTURE_same_name", n_bins=5)
    assert ten.reference_id == five.reference_id
    assert ten.fingerprint != five.fingerprint
    rebuilt = gaussian_reference(reference_id="FIXTURE_same_name")
    assert rebuilt.fingerprint == ten.fingerprint


def test_the_fingerprint_moves_with_the_epsilon_floor() -> None:
    base = gaussian_reference()
    floored = ReferenceDistribution(
        feature=base.feature,
        units=base.units,
        reference_id=base.reference_id,
        stamp=base.stamp,
        edges=base.edges,
        bin_counts=base.bin_counts,
        n_absent=base.n_absent,
        epsilon=1e-4,
    )
    assert floored.fingerprint != base.fingerprint


# ---------------------------------------------------------------------------
# 3. Empty bins
# ---------------------------------------------------------------------------


EMPTIED_DECILE_TERM: Final = (DEFAULT_EPSILON - 0.1) * math.log(DEFAULT_EPSILON / 0.1)
"""Closed form for one emptied decile at the default floor: ``≈ 1.1513``."""


def test_an_emptied_decile_contributes_the_documented_closed_form() -> None:
    reference = equal_decile_reference()
    assert reference.bin_counts == (2_000,) * DEFAULT_BINS
    assert reference.fractions == (0.1,) * DEFAULT_BINS
    # Every live value inside bin 5 -> the other nine deciles empty.
    result = population_stability_index(
        reference=reference, observed=np.full(OBSERVED_SIZE, 10_500.0)
    )
    emptied = [bin_ for bin_ in result.bins if bin_.observed_count == 0]
    assert len(emptied) == 9
    for bin_ in emptied:
        assert bin_.contribution == pytest.approx(EMPTIED_DECILE_TERM)
        assert bin_.contribution == pytest.approx(1.1512810, abs=1e-6)
    occupied = next(bin_ for bin_ in result.bins if bin_.observed_count)
    assert occupied.index == 5
    assert occupied.contribution == pytest.approx(0.9 * math.log(10.0))
    assert result.value == pytest.approx(9 * EMPTIED_DECILE_TERM + 0.9 * math.log(10.0))


def test_the_floor_is_visible_rather_than_buried() -> None:
    reference = equal_decile_reference()
    result = population_stability_index(
        reference=reference, observed=np.full(OBSERVED_SIZE, 10_500.0)
    )
    assert result.floored_bins == (0, 1, 2, 3, 4, 6, 7, 8, 9)
    assert result.floor_contribution == pytest.approx(9 * EMPTIED_DECILE_TERM)
    # The floor supplies more than four fifths of the total. A reader who was not
    # told this would read 12.4 as a measurement of the data.
    assert result.floor_contribution > 0.8 * result.value
    payload = result.to_dict()
    assert payload["epsilon"] == DEFAULT_EPSILON
    assert payload["floored_bins"] == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert payload["floor_contribution"] == pytest.approx(9 * EMPTIED_DECILE_TERM)
    serialised_bins = json.loads(json.dumps(payload["bins"]))
    assert [entry["index"] for entry in serialised_bins if entry["floored"]] == [
        0,
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
    ]
    json.dumps(payload)


def test_floor_contribution_is_a_float_even_when_nothing_was_floored() -> None:
    # `sum()` over an empty selection returns the integer 0, which serialises as
    # `0` rather than `0.0` and silently changes a dashboard column's type
    # between dates.
    reference = gaussian_reference()
    result = population_stability_index(reference=reference, observed=gaussian_sample(seed=5))
    assert result.floored_bins == ()
    assert isinstance(result.floor_contribution, float)
    assert result.floor_contribution == 0.0
    assert isinstance(result.to_dict()["floor_contribution"], float)


def test_no_bin_is_dropped_and_the_contributions_sum_to_the_value() -> None:
    reference = equal_decile_reference()
    for observed in (
        np.full(OBSERVED_SIZE, 10_500.0),
        np.arange(OBSERVED_SIZE, dtype=np.float64),
        gaussian_sample(seed=3, size=OBSERVED_SIZE) * 5_000.0 + 10_000.0,
    ):
        result = population_stability_index(reference=reference, observed=observed)
        assert len(result.bins) == reference.n_bins
        assert [bin_.index for bin_ in result.bins] == list(range(reference.n_bins))
        assert sum(bin_.observed_count for bin_ in result.bins) == result.n_observed_present
        assert result.value == pytest.approx(
            sum(bin_.contribution for bin_ in result.bins), rel=1e-12
        )
        assert all(bin_.contribution >= 0.0 for bin_ in result.bins)


def test_a_value_beyond_the_training_range_lands_in_the_extreme_bin() -> None:
    # Finite outer edges would drop exactly the observations most likely to be
    # the drift. The reference's edges are infinite, so nothing is discarded.
    reference = equal_decile_reference()
    result = population_stability_index(reference=reference, observed=np.full(OBSERVED_SIZE, 1e12))
    assert result.n_observed_present == OBSERVED_SIZE
    assert result.bins[-1].observed_count == OBSERVED_SIZE
    assert result.value > MAJOR


def test_a_reference_with_an_empty_bin_is_refused() -> None:
    # `p == 0` gives the comparison an arbitrary baseline, so it is refused at
    # construction rather than floored like the live side.
    with pytest.raises(ReferenceDistributionError, match="hold no observations"):
        ReferenceDistribution.from_sample(
            np.arange(20_000, dtype=np.float64),
            feature="FIXTURE_gapped",
            units=FIXTURE_UNITS,
            reference_id="FIXTURE_gapped_edges",
            stamp=fixture_stamp(),
            edges=[-1e9, 5_000.0, 10_000.0, 1e9],
        )


def test_a_distribution_on_mass_points_is_refused_rather_than_binned_anyway() -> None:
    mass_points = np.concatenate(
        [np.zeros(18_000), np.linspace(1.0, 2.0, 2_000)],
    )
    with pytest.raises(ReferenceDistributionError, match="concentrated on mass points"):
        ReferenceDistribution.from_sample(
            mass_points,
            feature="FIXTURE_mass_points",
            units=FIXTURE_UNITS,
            reference_id="FIXTURE_mass_points",
            stamp=fixture_stamp(),
        )


def test_asking_for_too_few_bins_blames_the_request_not_the_data() -> None:
    with pytest.raises(ReferenceDistributionError, match=f"below the minimum of {MINIMUM_BINS}"):
        gaussian_reference(n_bins=3)


# ---------------------------------------------------------------------------
# 4. Small samples raise rather than report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [2, 4, 5, 10, 20, 50])
def test_the_minimum_is_derived_from_the_noise_budget(n_bins: int) -> None:
    minimum = minimum_sample_size(n_bins)
    assert minimum == math.ceil((n_bins - 1) / NULL_PSI_NOISE_BUDGET)
    # The definition of the budget: at the minimum, pure multinomial noise
    # produces a PSI no larger than a quarter of the "moderate" rung.
    assert (n_bins - 1) / minimum <= NULL_PSI_NOISE_BUDGET
    assert (n_bins - 1) / (minimum - 1) > NULL_PSI_NOISE_BUDGET


def test_the_documented_minima_hold() -> None:
    assert minimum_sample_size(DEFAULT_BINS) == 360
    assert minimum_sample_size(MINIMUM_BINS) == 120
    assert minimum_sample_size(AVAILABILITY_CATEGORIES) == 40


@pytest.mark.parametrize("n_observations", [40, 100, 200, 359])
def test_a_sample_below_the_minimum_refuses_instead_of_returning_a_number(
    n_observations: int,
) -> None:
    reference = gaussian_reference()
    with pytest.raises(InsufficientSampleError) as raised:
        population_stability_index(
            reference=reference, observed=gaussian_sample(seed=8, size=n_observations)
        )
    error = raised.value
    assert error.n_observations == n_observations
    assert error.minimum == 360
    assert error.n_bins == DEFAULT_BINS
    assert error.quantity == "present (non-NaN) observations"
    # I2: a refusal that does not name its reference is as uninterpretable as a
    # number that does not.
    assert error.feature == reference.feature
    assert error.reference_id == reference.reference_id
    assert reference.reference_id in str(error)
    # The refusal quotes the noise a number would have been, so the reader can
    # see why it is a refusal rather than a small reading.
    assert f"{(DEFAULT_BINS - 1) / n_observations:.4f}" in str(error)


def test_the_boundary_holds_in_both_directions() -> None:
    reference = gaussian_reference()
    minimum = minimum_sample_size(reference.n_bins)
    with pytest.raises(InsufficientSampleError):
        population_stability_index(
            reference=reference, observed=gaussian_sample(seed=9, size=minimum - 1)
        )
    just_enough = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=9, size=minimum)
    )
    assert just_enough.n_observed_present == minimum
    # And the refusal was worth making: at the minimum the noise floor is still
    # a quarter of the "moderate" rung.
    assert just_enough.null_expected_value == pytest.approx(NULL_PSI_NOISE_BUDGET, rel=1e-9)


def test_the_refusal_states_the_noise_a_number_would_have_been() -> None:
    reference = gaussian_reference()
    with pytest.raises(InsufficientSampleError) as raised:
        population_stability_index(reference=reference, observed=gaussian_sample(seed=10, size=100))
    assert "0.0900" in str(raised.value)  # (10 - 1) / 100
    assert "noise presented as a signal" in str(raised.value)


def test_a_thin_reference_is_refused_at_construction() -> None:
    with pytest.raises(ReferenceDistributionError, match="below the minimum of 360"):
        gaussian_reference(size=359)


def test_the_refusal_is_not_a_valueerror_a_caller_would_swallow() -> None:
    # `except ValueError` around a numeric pipeline is ordinary. A detector's
    # refusal must not disappear into one.
    assert not issubclass(InsufficientSampleError, ValueError)


# ---------------------------------------------------------------------------
# 5. A change in the NaN rate is itself drift
# ---------------------------------------------------------------------------


def test_an_availability_collapse_is_reported_while_the_survivors_look_stable() -> None:
    """The failure mode: dropping ``NaN`` reports this fixture as perfectly stable."""
    reference = gaussian_reference()
    # The surviving names are drawn from the reference distribution itself, so
    # nothing about the *shape* has moved. 40% of the universe has simply
    # stopped having a value — the signature of a broken upstream source.
    observed = with_absent(gaussian_sample(seed=21, size=OBSERVED_SIZE), absent=800)
    result = population_stability_index(reference=reference, observed=observed)

    assert result.value < 0.05  # the survivors are stable, and correctly so
    assert result.availability.value > 3.0  # and the break is loud
    assert result.availability.observed_absent_fraction == pytest.approx(0.4)
    assert result.availability.reference_absent_fraction == 0.0
    assert result.availability.rate_change == pytest.approx(0.4)
    assert result.n_observed_total == OBSERVED_SIZE
    assert result.n_observed_present == OBSERVED_SIZE - 800


def test_absent_values_are_counted_not_dropped() -> None:
    reference = gaussian_reference()
    present_only = gaussian_sample(seed=22, size=1_200)
    padded = np.concatenate([present_only, np.full(800, np.nan)])
    dropped = population_stability_index(reference=reference, observed=present_only)
    counted = population_stability_index(reference=reference, observed=padded)
    # The distributional PSI is identical — the same names survive — but the two
    # results must not be interchangeable, because one of them knows that a
    # third of the universe went missing.
    assert counted.value == pytest.approx(dropped.value)
    assert counted.n_observed_total == 2_000
    assert dropped.n_observed_total == 1_200
    assert counted.availability.value > dropped.availability.value
    assert counted.availability.observed_absent_fraction == pytest.approx(0.4)
    assert dropped.availability.observed_absent_fraction == 0.0


def test_availability_is_never_folded_into_the_distribution_psi() -> None:
    reference = gaussian_reference()
    observed = with_absent(gaussian_sample(seed=23, shift=1.0), absent=600)
    result = population_stability_index(reference=reference, observed=observed)
    # A large availability break must not be able to hide inside a moderate
    # distributional number, nor the reverse. The value is the bin sum and
    # nothing else.
    assert result.value == pytest.approx(sum(bin_.contribution for bin_ in result.bins), rel=1e-12)
    assert result.availability.value > 1.0
    assert result.value < result.value + result.availability.value
    payload = result.to_dict()
    assert payload["value"] == result.value
    assert isinstance(payload["availability"], dict)
    assert payload["availability"]["value"] == result.availability.value


def test_the_reference_records_its_own_availability_rate() -> None:
    reference = gaussian_reference(absent=2_000)
    assert reference.n_absent == 2_000
    assert reference.n_present == 18_000
    assert reference.absent_fraction == pytest.approx(0.1)
    # A live sample matching the reference's own NaN rate is not an availability
    # finding, which is what "compared against the reference" has to mean.
    observed = with_absent(gaussian_sample(seed=24, size=2_000), absent=200)
    result = population_stability_index(reference=reference, observed=observed)
    assert result.availability.rate_change == pytest.approx(0.0)
    assert result.availability.value == pytest.approx(0.0, abs=1e-12)
    # Whereas a live sample with none missing at all *is* one.
    intact = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=24, size=2_000)
    )
    assert intact.availability.rate_change == pytest.approx(-0.1)
    assert intact.availability.value > 1.0
    assert intact.availability.floored is True


def test_an_all_absent_cross_section_refuses_and_carries_the_finding() -> None:
    reference = gaussian_reference()
    with pytest.raises(InsufficientSampleError) as raised:
        population_stability_index(reference=reference, observed=np.full(OBSERVED_SIZE, np.nan))
    error = raised.value
    assert error.n_observations == 0
    assert error.quantity == "present (non-NaN) observations"
    assert error.availability is not None
    assert error.availability.observed_absent_fraction == 1.0
    assert error.availability.value > 10.0
    assert "The NaN rate moved from 0.0000 to 1.0000" in str(error)


def test_a_sample_too_thin_to_bin_still_reports_the_availability_it_could_measure() -> None:
    reference = gaussian_reference()
    observed = np.full(1_000, np.nan)
    observed[:100] = gaussian_sample(seed=25, size=100)
    with pytest.raises(InsufficientSampleError) as raised:
        population_stability_index(reference=reference, observed=observed)
    assert raised.value.availability is not None
    assert raised.value.availability.observed_absent_fraction == pytest.approx(0.9)
    assert "attached to this exception as `.availability`" in str(raised.value)


def test_the_availability_shift_carries_its_own_noise_scale() -> None:
    reference = gaussian_reference()
    result = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=26, size=2_000)
    )
    assert result.availability.null_expected_value == pytest.approx(1 / 2_000)
    assert result.availability.n_observed == 2_000
    assert result.availability.n_reference == reference.n_total


# ---------------------------------------------------------------------------
# 6. No data raises; it never returns 0.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_observations", [0, 1, 2, 10, 39])
def test_no_data_raises_rather_than_reporting_no_drift(n_observations: int) -> None:
    reference = gaussian_reference()
    with pytest.raises(InsufficientSampleError) as raised:
        population_stability_index(
            reference=reference, observed=np.zeros(n_observations, dtype=np.float64)
        )
    error = raised.value
    assert error.n_observations == n_observations
    assert error.minimum == minimum_sample_size(AVAILABILITY_CATEGORIES)
    assert error.quantity == "observations"
    assert "Zero is not returned either" in str(error)
    assert "most dangerous" in str(error)


def test_the_entry_point_has_no_constant_return_path() -> None:
    """Structural: there is nowhere for a ``0.0`` to be returned from."""
    source = ast.parse(Path(inspect.getsourcefile(psi_module) or "").read_text(encoding="utf-8"))
    entry = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "population_stability_index"
    )
    returns = [node for node in ast.walk(entry) if isinstance(node, ast.Return)]
    assert len(returns) == 1, "more than one return path; one of them is not a measurement"
    only = returns[0].value
    assert isinstance(only, ast.Call)
    assert isinstance(only.func, ast.Name)
    assert only.func.id == "PSIResult"
    assert any(isinstance(node, ast.Raise) for node in ast.walk(entry))


def test_an_empty_reference_sample_is_refused() -> None:
    with pytest.raises(ReferenceDistributionError, match="no present observations"):
        ReferenceDistribution.from_sample(
            np.full(500, np.nan),
            feature="FIXTURE_empty",
            units=FIXTURE_UNITS,
            reference_id="FIXTURE_empty",
            stamp=fixture_stamp(),
        )


def test_a_reference_without_a_name_is_refused() -> None:
    # I2: a PSI quoted against an unnamed reference cannot be interpreted or
    # reproduced, so the name is a construction-time requirement.
    for blank in ("", "   "):
        with pytest.raises(ReferenceDistributionError, match="is empty"):
            gaussian_reference(reference_id=blank)


def test_a_reference_without_a_stamp_is_refused() -> None:
    base = gaussian_reference()
    with pytest.raises(ReferenceDistributionError, match="ReproducibilityStamp"):
        ReferenceDistribution(
            feature=base.feature,
            units=base.units,
            reference_id=base.reference_id,
            stamp="0" * 40,  # type: ignore[arg-type]
            edges=base.edges,
            bin_counts=base.bin_counts,
            n_absent=base.n_absent,
        )


# ---------------------------------------------------------------------------
# Input refusals shared by every claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_an_infinity_is_refused_rather_than_binned_into_the_tail(infinity: float) -> None:
    reference = gaussian_reference()
    observed = gaussian_sample(seed=31)
    observed[17] = infinity
    with pytest.raises(MonitoringInputError, match="infinite value"):
        population_stability_index(reference=reference, observed=observed)


def test_a_two_dimensional_panel_is_refused() -> None:
    reference = gaussian_reference()
    with pytest.raises(MonitoringInputError, match="one-dimensional cross-section"):
        population_stability_index(
            reference=reference, observed=gaussian_sample(seed=32, size=2_000).reshape(100, 20)
        )


def test_a_non_numeric_cross_section_is_refused() -> None:
    reference = gaussian_reference()
    with pytest.raises(MonitoringInputError, match="must be numeric"):
        population_stability_index(reference=reference, observed=["a"] * 500)


def test_the_caller_may_mutate_its_input_afterwards() -> None:
    reference = gaussian_reference()
    observed = gaussian_sample(seed=33)
    result = population_stability_index(reference=reference, observed=observed)
    observed[:] = 1e9
    assert population_stability_index(reference=reference, observed=observed).value != result.value
    assert result.n_observed_present == OBSERVED_SIZE


def test_the_result_serialises_to_json_with_its_reference_identity() -> None:
    reference = gaussian_reference()
    result = population_stability_index(
        reference=reference, observed=gaussian_sample(seed=34, shift=1.0)
    )
    payload = result.to_dict()
    text = json.dumps(payload)
    assert reference.reference_id in text
    assert reference.fingerprint in text
    assert reference.stamp.data_version in text
    assert payload["units"] == "dimensionless (Jeffreys divergence between two histograms)"
    assert isinstance(payload["reference"], dict)
    assert payload["reference"]["binning"] == (
        "quantile bins of the reference sample, fixed at reference time"
    )
    # ±inf edges survive JSON as strings rather than being dropped.
    assert payload["reference"]["edges"][0] == "-inf"  # type: ignore[index]
    assert payload["reference"]["edges"][-1] == "inf"  # type: ignore[index]


@pytest.mark.parametrize("bad", [0.0, 1.0, -1e-6, 2.0, float("nan")])
def test_an_epsilon_outside_the_open_unit_interval_is_refused(bad: float) -> None:
    base = gaussian_reference()
    with pytest.raises(ReferenceDistributionError, match="epsilon"):
        ReferenceDistribution(
            feature=base.feature,
            units=base.units,
            reference_id=base.reference_id,
            stamp=base.stamp,
            edges=base.edges,
            bin_counts=base.bin_counts,
            n_absent=base.n_absent,
            epsilon=bad,
        )


@pytest.mark.parametrize("n_bins", [0, 1, -3])
def test_a_bin_count_below_two_has_no_divergence_to_measure(n_bins: int) -> None:
    with pytest.raises(MonitoringInputError, match="at least 2"):
        minimum_sample_size(n_bins)


def test_a_bool_is_not_a_bin_count() -> None:
    with pytest.raises(MonitoringInputError, match="must be an int"):
        # `True` is an int in Python, and a bin count of 1 has no divergence to
        # measure — silently accepting it would be the wrong kind of tolerant.
        minimum_sample_size(True)
