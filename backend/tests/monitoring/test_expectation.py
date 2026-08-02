"""Tests for live-versus-expected and the auto-halt (P12.1, gate G12).

The gate is "injected performance deviation triggers halt", and a suite that
only asserted that would pass with a band of ``[-inf, +inf]`` inverted, or with a
``decide`` that halts on everything. So the tests are organised around the five
ways this component fails *silently*:

1. **The band is not what it claims.** Its edges are asserted against an
   independent transcription of the window statistic
   (:func:`~backend.tests.monitoring.expectation_fixtures.pooled_window_sharpes`)
   and against the *fraction* of windows that fall outside them. A band widened
   to the extremes of the sample — the mutation that makes the auto-halt never
   fire — fails those assertions and fails the deviation test too, because the
   injected deviation is placed just outside the quantile edge and well inside
   the sample's own minimum. That relationship is asserted in the test itself,
   so the test says what it is protecting.

2. **"Unavailable" reads as "in band".** Every unavailability is asserted to
   *halt*: no band, no live window, a live window of the wrong length, stale
   data, a gross cost basis, and an unexpected exception from inside the
   comparison. Structurally, ``CONTINUE`` is asserted to be unconstructible
   without a completed non-breaching comparison.

3. **The horizon silently mismatches.** A band cut at one window length compared
   against a live window of another is refused rather than scaled.

4. **The comparison is not net of costs on both sides (I4).**

5. **"Inside the band" is presented as validation.** The disclosures are
   asserted to be *in the payload*, and the comparison is asserted to carry no
   attribute a template could render as a pass mark.

Both directions of the gate are here: a deviation halts, an in-band series does
not, and the in-band decision's own text is asserted to say it is not a
validation.

All fixtures are synthetic and say so (I3 — see ``expectation_fixtures.py``).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.monitoring import expectation
from backend.monitoring.errors import (
    ComparisonUnavailableError,
    CostBasisError,
    ExpectationBandError,
    MonitoringError,
)
from backend.monitoring.expectation import (
    DEFAULT_WINDOW_PERIODS,
    MINIMUM_WINDOW_PERIODS,
    SELECTION_CONTAMINATION_DISCLOSURE,
    CostTreatment,
    ExpectationBand,
    ExpectationComparison,
    HaltAction,
    HaltCause,
    HaltDecision,
    HaltPolicy,
    HaltSide,
    LiveWindow,
    WindowStatistic,
    compare,
    decide,
    path_digest,
)
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    DEFAULT_N_OBSERVATIONS,
    FIXTURE_LIVE_SOURCE,
    LIVE_AS_OF,
    fixture_artefact,
    fixture_band,
    live_window,
    pooled_window_sharpes,
    series_with_sharpe,
    synthetic_paths,
)
from backend.tests.monitoring.fixtures import fixture_stamp

if TYPE_CHECKING:
    from backend.backtest.metrics import FloatArray

# ---------------------------------------------------------------------------
# 1. The policy arithmetic: width is derived, not chosen
# ---------------------------------------------------------------------------


def test_the_tail_mass_is_the_sidak_correction_of_the_annual_budget() -> None:
    """Independently transcribed: 1 - (1 - budget) ** (1 / evaluations)."""
    policy = HaltPolicy(window_periods=63, periods_per_year=252.0, annual_false_halt_budget=0.10)
    assert policy.evaluations_per_year == pytest.approx(4.0)
    expected_alpha = 1.0 - 0.90**0.25
    assert policy.per_evaluation_alpha == pytest.approx(expected_alpha)
    # Two-sided by default, so the budget is split across the halting edges.
    assert policy.tail_mass == pytest.approx(expected_alpha / 2.0)
    assert policy.n_halting_sides == 2


def test_a_one_sided_policy_spends_the_whole_budget_on_the_lower_edge() -> None:
    two_sided = HaltPolicy()
    one_sided = HaltPolicy(side=HaltSide.LOWER)
    assert one_sided.tail_mass == pytest.approx(2.0 * two_sided.tail_mass)


def test_a_larger_budget_buys_a_narrower_band() -> None:
    """The direction of the trade-off, asserted rather than described."""
    tolerant = fixture_band(policy=HaltPolicy(annual_false_halt_budget=0.30))
    strict = fixture_band(policy=HaltPolicy(annual_false_halt_budget=0.08))
    assert tolerant.width < strict.width


def test_a_window_below_the_floor_is_refused_rather_than_offered_as_a_knob() -> None:
    with pytest.raises(ExpectationBandError, match="below the minimum"):
        HaltPolicy(window_periods=MINIMUM_WINDOW_PERIODS - 1)


def test_a_window_longer_than_a_year_is_refused() -> None:
    """Fewer than one evaluation a year makes the annual budget uninterpretable."""
    with pytest.raises(ExpectationBandError, match="exceeds periods_per_year"):
        HaltPolicy(window_periods=300, periods_per_year=252.0)


@pytest.mark.parametrize("budget", [0.0, 1.0, -0.1, 1.5])
def test_a_budget_outside_the_open_unit_interval_is_refused(budget: float) -> None:
    with pytest.raises(ExpectationBandError, match="annual_false_halt_budget"):
        HaltPolicy(annual_false_halt_budget=budget)


def test_the_policy_payload_states_what_the_correction_does_not_cover() -> None:
    payload = HaltPolicy().to_dict()
    correction = payload["correction"]
    assert isinstance(correction, str)
    assert "Sidak" in correction
    assert "parallel" in correction  # the unhandled multiplicity, named in the payload


# ---------------------------------------------------------------------------
# 2. The band is a quantile of matched-horizon windows — the non-vacuity anchor
# ---------------------------------------------------------------------------


def test_the_band_edges_are_quantiles_of_an_independently_computed_window_sample() -> None:
    """Asserted against a second transcription, not against the module itself."""
    paths = synthetic_paths()
    band = fixture_band(paths=paths)
    pooled = pooled_window_sharpes(paths, window=band.window_periods)

    assert band.n_pooled_windows == pooled.size
    assert band.lower == pytest.approx(float(np.quantile(pooled, band.policy.tail_mass)))
    assert band.upper == pytest.approx(float(np.quantile(pooled, 1.0 - band.policy.tail_mass)))
    assert band.median == pytest.approx(float(np.median(pooled)))


def test_the_band_is_a_quantile_and_not_the_extremes_of_the_sample() -> None:
    """The mutation this catches: widening the band until nothing ever halts.

    A band built from ``min``/``max`` — or from any tail mass wide enough to hold
    the whole sample — leaves nothing outside it, and an auto-halt that can never
    fire is indistinguishable from no auto-halt at all. So the fraction of
    backtest windows outside the band is asserted to be non-zero and of the order
    of the configured tail mass.
    """
    paths = synthetic_paths()
    band = fixture_band(paths=paths)
    pooled = pooled_window_sharpes(paths, window=band.window_periods)

    below = float(np.mean(pooled < band.lower))
    above = float(np.mean(pooled > band.upper))
    assert below > 0.0, "no backtest window falls below the band: the edge is not a quantile"
    assert above > 0.0, "no backtest window falls above the band: the edge is not a quantile"
    assert below <= 2.0 * band.policy.tail_mass
    assert above <= 2.0 * band.policy.tail_mass
    assert float(np.min(pooled)) < band.lower
    assert float(np.max(pooled)) > band.upper


def test_the_horizon_is_matched_so_a_longer_window_gives_a_narrower_band() -> None:
    """Spread falls roughly with 1/sqrt(window); the point of cutting at the live horizon."""
    paths = synthetic_paths()
    quarter = fixture_band(paths=paths, policy=HaltPolicy(window_periods=63))
    year = fixture_band(paths=paths, policy=HaltPolicy(window_periods=252))
    assert year.dispersion < quarter.dispersion


def test_a_tail_the_sample_cannot_resolve_is_refused_not_approximated() -> None:
    """The quantile-resolution refusal: 1/tail_mass independent windows or nothing."""
    short = synthetic_paths(n_observations=300)
    policy = HaltPolicy(window_periods=63, annual_false_halt_budget=0.01)
    with pytest.raises(ExpectationBandError, match="effectively independent windows"):
        ExpectationBand.from_paths(short, artefact=fixture_artefact(short), policy=policy)


def test_the_resolution_refusal_counts_non_overlapping_windows_only() -> None:
    """Overlapping windows sharpen the shape; they are not independent draws.

    The default fixture gives 100 non-overlapping 63-period windows and 5,990
    overlapping ones. A tail mass of 0.0064 is comfortably resolvable by 5,990
    draws and is *not* resolvable by 100 — so a band that counted overlap would
    accept the finer budget. It is refused, which is the assertion.
    """
    paths = synthetic_paths()
    accepted = ExpectationBand.from_paths(
        paths,
        artefact=fixture_artefact(paths),
        policy=HaltPolicy(window_periods=63, annual_false_halt_budget=0.10),
    )
    assert accepted.n_independent_windows == 5 * (DEFAULT_N_OBSERVATIONS // 63)
    assert accepted.n_pooled_windows == 5 * (DEFAULT_N_OBSERVATIONS - 63 + 1)
    assert accepted.n_independent_windows >= accepted.policy.minimum_independent_windows

    finer = HaltPolicy(window_periods=63, annual_false_halt_budget=0.05)
    assert finer.minimum_independent_windows > accepted.n_independent_windows
    assert finer.minimum_independent_windows < accepted.n_pooled_windows
    with pytest.raises(ExpectationBandError, match="non-overlapping windows"):
        ExpectationBand.from_paths(paths, artefact=fixture_artefact(paths), policy=finer)


def test_a_window_longer_than_the_paths_is_refused() -> None:
    paths = synthetic_paths(n_observations=100)
    with pytest.raises(ExpectationBandError, match="exceeds"):
        ExpectationBand.from_paths(
            paths,
            artefact=fixture_artefact(paths),
            policy=HaltPolicy(window_periods=200, periods_per_year=252.0),
        )


def test_paths_that_disagree_with_the_artefact_shape_are_refused() -> None:
    """A band labelled with another run's identity is worse than an unlabelled one (I2)."""
    paths = synthetic_paths()
    other = synthetic_paths(n_paths=3, seed=7)
    with pytest.raises(ExpectationBandError, match="another run's identity"):
        ExpectationBand.from_paths(paths, artefact=fixture_artefact(other))


def test_non_finite_paths_are_refused() -> None:
    paths = synthetic_paths().copy()
    paths[0, 0] = np.nan
    with pytest.raises(ExpectationBandError, match="finite"):
        ExpectationBand.from_paths(paths, artefact=fixture_artefact(paths))


def test_a_degenerate_window_is_refused_rather_than_dropped() -> None:
    """Dropping the windows the statistic refuses would narrow the band silently."""
    paths = synthetic_paths().copy()
    paths[0, 100:200] = 0.0  # a zero-variance stretch leaves the Sharpe undefined
    with pytest.raises(ExpectationBandError, match="undefined"):
        ExpectationBand.from_paths(
            paths,
            artefact=fixture_artefact(paths),
            policy=HaltPolicy(window_periods=21, annual_false_halt_budget=0.2),
        )


# ---------------------------------------------------------------------------
# 3. Power and latency are reported, not asserted away
# ---------------------------------------------------------------------------


def test_detection_power_rises_with_the_size_of_the_deterioration() -> None:
    band = fixture_band()
    weak = band.detection_power(1.0)
    strong = band.detection_power(3.0)
    assert weak["power"] < strong["power"]
    assert weak["expected_evaluations"] > strong["expected_evaluations"]
    assert weak["expected_evaluations"] == pytest.approx(1.0 / weak["power"])


def test_a_quarterly_test_is_weak_and_the_numbers_say_so() -> None:
    """The honest headline of §2: one band standard deviation is mostly missed."""
    band = fixture_band()
    assert band.detection_power(1.0)["power"] < 0.25
    assert band.detection_power(3.0)["power"] > 0.5
    # Latency in years, which is the number an operator actually has to accept.
    assert band.detection_power(1.0)["expected_years"] > 1.0


def test_power_is_undefined_rather_than_perfect_on_a_zero_dispersion_band() -> None:
    band = fixture_band()
    flat = ExpectationBand(
        statistic=band.statistic,
        policy=band.policy,
        artefact=band.artefact,
        lower=0.0,
        upper=0.0,
        median=0.0,
        dispersion=0.0,
        n_pooled_windows=band.n_pooled_windows,
        n_independent_windows=band.n_independent_windows,
        risk_free_rate=0.0,
    )
    with pytest.raises(ExpectationBandError, match="undefined rather than perfect"):
        flat.detection_power(1.0)


# ---------------------------------------------------------------------------
# 4. What the band is not: the disclosures are payload, not prose
# ---------------------------------------------------------------------------


def test_the_selection_contamination_caveat_travels_in_the_payload() -> None:
    payload = fixture_band().to_dict()
    disclosures = payload["disclosures"]
    assert isinstance(disclosures, list)
    assert SELECTION_CONTAMINATION_DISCLOSURE in disclosures
    joined = " ".join(str(item) for item in disclosures)
    assert "SELECTED" in joined
    assert "never a validation" in joined
    assert "trial(s)" in joined  # §6.7: the search size beside the result


def test_the_band_names_the_cpcv_artefact_it_came_from() -> None:
    """I2: which artefact supplied the band, with the four stamp components."""
    paths = synthetic_paths()
    band = fixture_band(paths=paths)
    artefact = band.to_dict()["artefact"]
    assert isinstance(artefact, dict)
    assert artefact["path_digest"] == path_digest(paths)
    assert artefact["data_version"] == fixture_stamp().data_version
    assert artefact["config_hash"] == fixture_stamp().config_hash
    assert artefact["git_reference"] == fixture_stamp().git_reference
    assert artefact["seed"] == fixture_stamp().seed
    assert artefact["trials"] == band.artefact.trials


def test_the_whole_band_payload_is_json_serialisable() -> None:
    """It is rendered by the dashboard; a payload that cannot be encoded is not a payload."""
    encoded = json.dumps(fixture_band().to_dict())
    assert "cpcv_n_groups" in encoded


def test_a_comparison_carries_no_attribute_that_reads_as_a_pass_mark() -> None:
    """Structural: "inside the band" must not be renderable as validation."""
    band = fixture_band()
    comparison = compare(
        band=band,
        live=live_window(series_with_sharpe(band.median, n_periods=band.window_periods)),
        now=DECISION_AT,
    )
    for forbidden in ("validated", "passed", "healthy", "ok", "confirmed", "verified"):
        assert not hasattr(comparison, forbidden), forbidden
    assert forbidden not in comparison.to_dict()
    assert comparison.to_dict()["in_band_is_not_validation"] == SELECTION_CONTAMINATION_DISCLOSURE


def test_an_artefact_without_a_trial_count_cannot_be_built() -> None:
    """§6.7: the search size is part of the band's identity, not an optional extra."""
    paths = synthetic_paths()
    with pytest.raises(ExpectationBandError, match="trials"):
        fixture_artefact(paths, trials=0)


# ---------------------------------------------------------------------------
# 5. Cost realism on both sides (I4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("treatment", [CostTreatment.GROSS, CostTreatment.UNKNOWN])
def test_a_band_cut_from_returns_that_are_not_net_is_refused(treatment: CostTreatment) -> None:
    paths = synthetic_paths()
    with pytest.raises(CostBasisError, match="cost drag"):
        fixture_artefact(paths, cost_treatment=treatment)


@pytest.mark.parametrize("treatment", [CostTreatment.GROSS, CostTreatment.UNKNOWN])
def test_a_live_window_that_is_not_net_is_refused(treatment: CostTreatment) -> None:
    with pytest.raises(CostBasisError, match="gross live series"):
        live_window(series_with_sharpe(0.0, n_periods=63), cost_treatment=treatment)


def test_the_comparison_states_that_modelled_and_realised_costs_are_not_the_same_net() -> None:
    band = fixture_band()
    comparison = compare(
        band=band,
        live=live_window(series_with_sharpe(band.median, n_periods=band.window_periods)),
        now=DECISION_AT,
    )
    assert "MODELLED" in comparison.cost_note
    assert "REALISED" in comparison.cost_note
    assert comparison.to_dict()["cost_note"] == comparison.cost_note


# ---------------------------------------------------------------------------
# 6. The gate, both directions
# ---------------------------------------------------------------------------


def _deviated_live(band: ExpectationBand, *, sigmas_below: float) -> LiveWindow:
    """Build a live window whose Sharpe sits ``sigmas_below`` band sigmas under the edge."""
    target = band.lower - sigmas_below * band.dispersion
    return live_window(series_with_sharpe(target, n_periods=band.window_periods))


def test_an_injected_performance_deviation_triggers_a_halt() -> None:
    """Gate G12, the halting direction — with the deviation's size pinned.

    The injected value is placed just outside the lower edge and is asserted to
    lie **above** the worst backtest window. A band widened to the sample's
    extremes, or doubled in width, therefore fails this test rather than passing
    it by accident, which is what makes the assertion evidence about the band and
    not merely about a very bad number.
    """
    paths = synthetic_paths()
    band = fixture_band(paths=paths)
    pooled = pooled_window_sharpes(paths, window=band.window_periods)
    live = _deviated_live(band, sigmas_below=0.15)

    value = float(live.returns.mean() / live.returns.std(ddof=1))
    assert value < band.lower, "the fixture must be outside the band for the gate to mean anything"
    assert value > float(np.min(pooled)), (
        "the injected deviation must lie inside the sample's own range, so a band widened to "
        "its extremes stops halting and this test fails"
    )
    assert value > band.median - 2.0 * (band.median - band.lower), (
        "the injected deviation must sit inside a band of twice the width, so doubling the "
        "band stops halting and this test fails"
    )

    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND
    assert decision.should_halt
    assert decision.comparison is not None
    assert decision.comparison.below_band
    assert "below the expected band" in decision.detail
    assert "selected on" in decision.detail  # the caveat travels into the halt reason


def test_an_in_band_series_does_not_halt() -> None:
    """Gate G12, the other direction — and the verdict refuses to call itself validation."""
    band = fixture_band()
    live = live_window(series_with_sharpe(band.median, n_periods=band.window_periods))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.CONTINUE
    assert decision.cause is None
    assert not decision.should_halt
    assert decision.comparison is not None
    assert decision.comparison.within_band
    assert "not a validation" in decision.detail


def test_a_series_just_inside_the_lower_edge_does_not_halt() -> None:
    """The boundary from the permitting side: the band must not be tighter than it says."""
    band = fixture_band()
    target = band.lower + 0.05 * band.dispersion
    live = live_window(series_with_sharpe(target, n_periods=band.window_periods))
    assert decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT).action is (
        HaltAction.CONTINUE
    )


def test_an_upside_breach_halts_under_the_default_two_sided_policy() -> None:
    """Outperforming a band cut from your own backtest is an accounting question."""
    band = fixture_band()
    target = band.upper + 0.5 * band.dispersion
    live = live_window(series_with_sharpe(target, n_periods=band.window_periods))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.ABOVE_EXPECTED_BAND
    assert "position scale" in decision.detail


def test_an_upside_breach_under_a_one_sided_policy_continues_but_says_so() -> None:
    band = fixture_band(policy=HaltPolicy(side=HaltSide.LOWER))
    target = band.upper + 0.5 * band.dispersion
    live = live_window(series_with_sharpe(target, n_periods=band.window_periods))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.CONTINUE
    assert decision.comparison is not None
    assert decision.comparison.above_band
    assert not decision.comparison.breaches_halting_side
    assert "does not halt on" in decision.detail


def test_the_same_deviation_halts_on_the_mean_return_statistic_too() -> None:
    """The statistic is a choice; the mechanism is not specific to the Sharpe ratio."""
    paths = synthetic_paths()
    band = fixture_band(paths=paths, statistic=WindowStatistic.MEAN_RETURN)
    target = band.lower - 0.2 * band.dispersion
    returns = np.full(band.window_periods, target, dtype=np.float64)
    decision = decide(band=band, live=live_window(returns), stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND


# ---------------------------------------------------------------------------
# 7. Fail closed: every unavailability halts, and CONTINUE needs evidence
# ---------------------------------------------------------------------------


def test_no_band_halts_rather_than_passing() -> None:
    decision = decide(
        band=None,
        live=live_window(series_with_sharpe(0.1, n_periods=63)),
        stamp=fixture_stamp(),
        now=DECISION_AT,
    )
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.COMPARISON_UNAVAILABLE


def test_no_live_window_halts_rather_than_passing() -> None:
    decision = decide(band=fixture_band(), live=None, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.COMPARISON_UNAVAILABLE
    assert "manufacture" in decision.detail  # I3, said in the halt reason


def test_a_live_window_of_the_wrong_length_halts_rather_than_being_rescaled() -> None:
    band = fixture_band()
    live = live_window(series_with_sharpe(band.median, n_periods=band.window_periods // 2))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.COMPARISON_UNAVAILABLE
    assert "another horizon" in decision.detail


def test_stale_live_data_halts() -> None:
    band = fixture_band()
    live = live_window(
        series_with_sharpe(band.median, n_periods=band.window_periods),
        as_of=LIVE_AS_OF - dt.timedelta(days=30),
    )
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.STALE_LIVE_DATA
    assert "repeats its last verdict" in decision.detail


def test_a_gross_side_halts_with_its_own_cause() -> None:
    """Constructed past ``LiveWindow``'s own guard to reach ``decide``'s I4 branch."""
    band = fixture_band()
    live = live_window(series_with_sharpe(band.median, n_periods=band.window_periods))
    object.__setattr__(live, "cost_treatment", CostTreatment.GROSS)
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.COST_BASIS


def test_an_unexpected_exception_inside_the_comparison_halts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monitor that crashed checked nothing, which must not look like a pass."""

    def explode(**_: object) -> ExpectationComparison:
        msg = "simulated defect inside the comparison"
        raise RuntimeError(msg)

    monkeypatch.setattr(expectation, "compare", explode)
    band = fixture_band()
    decision = decide(
        band=band,
        live=live_window(series_with_sharpe(band.median, n_periods=band.window_periods)),
        stamp=fixture_stamp(),
        now=DECISION_AT,
    )
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.INTERNAL_ERROR
    assert "RuntimeError" in decision.detail


def test_continue_cannot_be_constructed_without_a_comparison() -> None:
    """The fail-closed guarantee as a type property, not a code review note."""
    with pytest.raises(ExpectationBandError, match="CONTINUE requires a completed comparison"):
        HaltDecision(
            action=HaltAction.CONTINUE,
            cause=None,
            detail="everything looks fine",
            comparison=None,
            stamp=fixture_stamp(),
            decided_at=DECISION_AT,
            as_of=LIVE_AS_OF,
        )


def test_continue_cannot_be_constructed_over_a_breaching_comparison() -> None:
    band = fixture_band()
    breaching = compare(band=band, live=_deviated_live(band, sigmas_below=0.5), now=DECISION_AT)
    with pytest.raises(ExpectationBandError, match="breached the halting side"):
        HaltDecision(
            action=HaltAction.CONTINUE,
            cause=None,
            detail="ignoring the breach",
            comparison=breaching,
            stamp=fixture_stamp(),
            decided_at=DECISION_AT,
            as_of=LIVE_AS_OF,
        )


def test_a_halt_must_name_a_cause() -> None:
    with pytest.raises(ExpectationBandError, match="must name its cause"):
        HaltDecision(
            action=HaltAction.HALT,
            cause=None,
            detail="stopped",
            comparison=None,
            stamp=fixture_stamp(),
            decided_at=DECISION_AT,
            as_of=None,
        )


def test_a_decision_without_an_i2_stamp_cannot_be_constructed() -> None:
    with pytest.raises(ExpectationBandError, match="ReproducibilityStamp"):
        HaltDecision(
            action=HaltAction.HALT,
            cause=HaltCause.INTERNAL_ERROR,
            detail="stopped",
            comparison=None,
            stamp="0" * 40,  # type: ignore[arg-type]
            decided_at=DECISION_AT,
            as_of=None,
        )


def test_the_unavailability_refusal_is_not_a_value_error() -> None:
    """A stray ``except ValueError`` must not be able to swallow the halting refusal."""
    assert not issubclass(ComparisonUnavailableError, ValueError)
    assert issubclass(ComparisonUnavailableError, MonitoringError)


def test_compare_raises_rather_than_returning_an_unavailable_verdict() -> None:
    """The dangerous shape does not exist: there is no 'unavailable' comparison object."""
    with pytest.raises(ComparisonUnavailableError):
        compare(band=None, live=live_window(series_with_sharpe(0.1, n_periods=63)))
    with pytest.raises(ComparisonUnavailableError):
        compare(band=fixture_band(), live=None)


def test_an_empty_live_window_is_refused_at_construction() -> None:
    with pytest.raises(ComparisonUnavailableError, match="no comparison"):
        LiveWindow(
            returns=np.asarray([], dtype=np.float64),
            as_of=LIVE_AS_OF,
            cost_treatment=CostTreatment.NET_REALISED,
            source=FIXTURE_LIVE_SOURCE,
        )


def test_a_live_window_with_a_nan_is_refused() -> None:
    returns: FloatArray = np.asarray([0.01, math.nan, 0.02], dtype=np.float64)
    with pytest.raises(ComparisonUnavailableError):
        live_window(returns)


def test_a_live_window_without_a_source_is_refused() -> None:
    """A halt whose evidence cannot be traced to a reconciliation run cannot be reviewed."""
    with pytest.raises(ExpectationBandError, match="source must say"):
        live_window(series_with_sharpe(0.1, n_periods=63), source="  ")


# ---------------------------------------------------------------------------
# 8. Reporting details
# ---------------------------------------------------------------------------


def test_the_percentile_position_stays_inside_the_resolvable_range() -> None:
    """A value beyond the edge is never extrapolated into a tail the sample cannot support."""
    band = fixture_band()
    far_below = compare(band=band, live=_deviated_live(band, sigmas_below=5.0), now=DECISION_AT)
    assert far_below.percentile == pytest.approx(band.policy.tail_mass)
    middle = compare(
        band=band,
        live=live_window(series_with_sharpe(band.median, n_periods=band.window_periods)),
        now=DECISION_AT,
    )
    assert middle.percentile == pytest.approx(0.5)


def test_the_decision_payload_is_json_serialisable_and_carries_the_band() -> None:
    band = fixture_band()
    decision = decide(
        band=band,
        live=_deviated_live(band, sigmas_below=0.2),
        stamp=fixture_stamp(),
        now=DECISION_AT,
    )
    payload = json.loads(json.dumps(decision.to_dict()))
    assert payload["action"] == "halt"
    assert payload["cause"] == "below_expected_band"
    assert payload["comparison"]["band"]["artefact"]["artefact_id"] == band.artefact.artefact_id
    assert payload["data_version"] == fixture_stamp().data_version


def test_the_default_window_is_a_quarter() -> None:
    """Pinned so a change to the observation window is a deliberate, reviewed edit."""
    assert DEFAULT_WINDOW_PERIODS == 63
    assert HaltPolicy().window_periods == DEFAULT_WINDOW_PERIODS
