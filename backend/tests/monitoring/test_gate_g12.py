"""Gate G12: injected drift is detected; an injected performance deviation halts.

This module is a **verification**, not a feature. Every mechanism it exercises
was built and unit-tested in P12.1, P12.2 and P12.4; what is asserted here is
that the mechanisms compose into the two sentences the directive's Phase 12 gate
actually states (§5 Phase 12), end to end, at the layer an operator sees.

What each clause is asked to show, and in both directions
---------------------------------------------------------

**Clause 1 — injected drift is detected.** Not "the PSI number moves" — P12.2
proves that at the metric. Here a feature distribution is corrupted on purpose
and the *reporting* layer says so: the drift report bands the injected feature
``MAJOR``, names it, stays ``complete``, and the alerting layer turns that into
one ``CRITICAL`` alert that is persisted and delivered.

**Clause 2 — an injected performance deviation triggers a halt.** The decision
half lives here: a live window placed a stated distance below the expectation
band produces ``HALT`` / ``BELOW_EXPECTED_BAND``, and the cycle raises and
delivers the ``CRITICAL`` alert that a halt row will point at. The *landing* half
— the ``monitoring_halt_event`` row, and
:func:`~backend.monitoring.history.require_not_halted` refusing afterwards —
needs a database and lives in ``backend/tests/integration/test_gate_g12_db.py``.

**Both clauses assert the negative too**, because a detector that always fires
passes the positive arm of any gate trivially. The unshifted panel bands
``STABLE`` on every feature and raises nothing at all; the in-band series
continues, and a continue is refused entry to the halt history without the
database being touched. The two arms differ only by the injection.

What this module does NOT establish
-----------------------------------

1. **That real drift will be caught at a useful latency.** Every condition here
   is synthetic, seeded, and injected at a magnitude chosen to be decisive
   (:mod:`backend.tests.monitoring.gate_g12_fixtures`). The gate shows the
   pipeline reacts to a corruption it was handed; it says nothing about the size
   or speed of the corruptions the market produces.
2. **That the halt is a powerful test.** P12.1's own measured power is
   ``0.11 / 0.41 / 0.78`` at one, two and three band standard deviations of
   deterioration — one evaluation per quarter, so a 1-sigma decay takes ~2.3
   years to catch in expectation. The test named
   ``test_the_gate_does_not_establish_that_this_is_a_powerful_test`` asserts
   those numbers rather than leaving them in a docstring, so the gate cannot be
   quoted without them.
3. **That Phase 12 is complete.** P12.3 (extraction-quality drift on the golden
   set) is blocked on B3 and P12.5 (the monitoring UI) is unstarted. Neither is
   part of G12's stated condition, and neither is verified anywhere below.
4. **Anything about the thresholds themselves.** The 0.10/0.25 ladder is
   credit-scoring convention (D-031), and the gate uses it unchanged and
   unweakened. A gate passed by moving a threshold would be worthless, so no
   test here constructs a non-default :class:`~backend.monitoring.drift.DriftBands`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import TYPE_CHECKING, Any, cast

import pytest

from backend import monitoring
from backend.monitoring.alerts import (
    AlertSeverity,
    MonitoringSnapshot,
    default_rules,
    evaluate_rules,
)
from backend.monitoring.drift import DEFAULT_MODERATE_THRESHOLD, DriftBand, DriftBands
from backend.monitoring.errors import CostBasisError, HaltHistoryError
from backend.monitoring.expectation import (
    CostTreatment,
    HaltAction,
    HaltCause,
    HaltPolicy,
    HaltSide,
    WindowStatistic,
    compare,
    decide,
)
from backend.monitoring.history import record_halt
from backend.tests.monitoring.doubles import InMemoryAlertStore, RecordingChannel
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    FIXTURE_LIVE_SOURCE,
    fixture_band,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import FIXTURE_DATA_VERSION, fixture_stamp
from backend.tests.monitoring.gate_g12_fixtures import (
    DEVIATION_SIGMAS_BELOW_EDGE,
    GATE_AS_OF,
    GATE_FEATURES,
    INJECTED_ABSENT_COUNT,
    INJECTED_INDEX,
    INJECTED_LOCATION_SHIFT,
    INJECTED_SCALE_FACTOR,
    deviated_live_window,
    gate_band,
    gate_drift_report,
    in_band_live_window,
    run_monitoring_cycle,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.monitoring.drift import DriftReport, FeatureDrift

INJECTED_FEATURE = GATE_FEATURES[INJECTED_INDEX]
CONTROL_FEATURES = tuple(
    name for index, name in enumerate(GATE_FEATURES) if index != INJECTED_INDEX
)


def _finding(report: DriftReport, feature: str) -> FeatureDrift:
    """Return the measured finding for ``feature``, failing if it was refused.

    A lookup that raises rather than returning ``None`` on purpose: an
    unmeasurable feature is exactly what would make a gate assertion vacuous,
    so it fails the test instead of skipping the assertion.
    """
    for measured in report.measured:
        if measured.feature == feature:
            return measured
    message = f"{feature!r} was not measured in this report"
    raise AssertionError(message)


class _NoDatabase:
    """A session stand-in that fails the test if it is touched at all."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - deliberately refuses everything
        message = f"the database was used ({name}) before the argument check refused the call"
        raise AssertionError(message)


def _unusable_session() -> AsyncSession:
    """Return the unusable session, typed for the functions under test."""
    return cast("AsyncSession", _NoDatabase())


# ===========================================================================
# Clause 1 — injected drift is detected, through the reporting layer
# ===========================================================================


def test_clause_1_negative_an_uninjected_panel_reports_no_drift() -> None:
    """The control arm: three fresh draws from their own references drift nowhere.

    This runs first because it is the assertion that makes the positive arm mean
    anything. The observed cross-sections are drawn with seeds the references
    never saw, so the PSI here is the detector's noise floor on a 2,000-name
    panel — not a re-measurement of the training sample, which would be zero by
    construction and would prove nothing.
    """
    report = gate_drift_report()

    assert report.complete, "no feature was refused, so the negative arm is a real measurement"
    assert report.worst_band is DriftBand.STABLE
    assert report.by_band(DriftBand.MODERATE) == ()
    assert report.by_band(DriftBand.MAJOR) == ()

    for feature in GATE_FEATURES:
        finding = _finding(report, feature)
        assert finding.band is DriftBand.STABLE, f"{feature} banded {finding.band}"
        assert finding.psi.value < DEFAULT_MODERATE_THRESHOLD
        # An order of magnitude below the ladder's first rung, not marginally so.
        assert finding.psi.value < 0.02, f"{feature} PSI {finding.psi.value:.6f}"
        assert finding.availability_band is DriftBand.STABLE


async def test_clause_1_negative_an_uninjected_panel_raises_no_alert() -> None:
    """Nothing measured, nothing said: the operator's inbox stays empty."""
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    snapshot = MonitoringSnapshot(
        as_of=GATE_AS_OF, stamp=fixture_stamp(), drift=gate_drift_report()
    )

    results = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)

    assert results == ()
    assert store.alerts == {}
    assert channel.delivered == []


@pytest.mark.parametrize(
    ("injection", "kwargs"),
    [
        ("location", {"location_shift": INJECTED_LOCATION_SHIFT}),
        ("scale", {"scale_factor": INJECTED_SCALE_FACTOR}),
        ("availability", {"absent": INJECTED_ABSENT_COUNT}),
    ],
)
def test_clause_1_positive_each_injection_is_detected_and_localised(
    injection: str, kwargs: dict[str, float | int]
) -> None:
    """Every injection kind bands ``MAJOR``, and only on the feature it was put in.

    Three orthogonal corruptions, because a detector that only notices a mean
    shift would pass a one-injection gate and still be blind to a variance
    explosion or to a source that stopped delivering.
    """
    report = gate_drift_report(**kwargs)  # type: ignore[arg-type]

    assert report.complete, "the injections are all measurable; a refusal here is a different bug"
    assert report.worst_band is DriftBand.MAJOR
    assert [finding.feature for finding in report.by_band(DriftBand.MAJOR)] == [INJECTED_FEATURE]

    for feature in CONTROL_FEATURES:
        control = _finding(report, feature)
        assert control.band is DriftBand.STABLE, (
            f"the {injection} injection lit up {feature}, which was never injected into"
        )

    injected = _finding(report, INJECTED_FEATURE)
    assert injected.band is DriftBand.MAJOR
    assert not injected.floor_driven, "the finding must not be an artefact of the epsilon floor"


def test_clause_1_positive_a_location_injection_is_measured_far_above_the_ladder() -> None:
    """The location injection's magnitude, stated rather than implied.

    A 1-sigma shift of the whole cross-section is not a marginal call: it lands
    roughly nine times the conventional ``MAJOR`` threshold, which is what makes
    this arm of the gate a demonstration rather than a coin flip.
    """
    report = gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT)
    injected = _finding(report, INJECTED_FEATURE)

    assert injected.psi.value > 8.0 * DEFAULT_MODERATE_THRESHOLD
    assert injected.distribution_band is DriftBand.MAJOR
    assert injected.exceeds_sampling_noise, (
        "a value inside this sample's own multinomial noise is the ladder speaking, not the data"
    )
    assert injected.availability_band is DriftBand.STABLE, (
        "a location shift moves shape, not availability; the two axes must not bleed"
    )


def test_clause_1_positive_a_scale_injection_moves_shape_without_moving_the_mean() -> None:
    """Dispersion drift is caught although the centre of the distribution did not move."""
    report = gate_drift_report(scale_factor=INJECTED_SCALE_FACTOR)
    injected = _finding(report, INJECTED_FEATURE)

    assert injected.distribution_band is DriftBand.MAJOR
    assert injected.exceeds_sampling_noise
    assert injected.availability_band is DriftBand.STABLE


def test_clause_1_positive_an_availability_break_surfaces_on_its_own_axis() -> None:
    """Two fifths of the panel stops arriving; the shape is fine and the report is not.

    The survivors are drawn from the reference distribution itself, so the
    distributional PSI is *correctly* stable. D-031's whole point is that
    averaging the availability change into that number would report this fixture
    as perfectly healthy — the failure mode where an upstream source has broken
    and the dashboard is green.
    """
    report = gate_drift_report(absent=INJECTED_ABSENT_COUNT)
    injected = _finding(report, INJECTED_FEATURE)

    assert injected.distribution_band is DriftBand.STABLE
    assert injected.psi.value < DEFAULT_MODERATE_THRESHOLD
    assert not injected.exceeds_sampling_noise, (
        "the survivors really are the reference distribution; the shape did not move"
    )
    assert injected.availability_band is DriftBand.MAJOR
    assert injected.psi.availability.value > 1.0
    assert injected.band is DriftBand.MAJOR, "the report must take the worse of the two axes"
    # Two orders of magnitude apart: averaging them would erase the finding.
    assert injected.psi.availability.value > 100.0 * injected.psi.value


def test_clause_1_the_ladder_walks_with_the_size_of_the_injection() -> None:
    """PSI is monotone in the injected shift, and the bands climb with it.

    Stronger evidence than any single threshold crossing: the detector is
    tracking the injection's magnitude rather than firing on contact. The
    intermediate rungs are the ones that would be missing from an always-on
    detector.
    """
    values = [
        _finding(gate_drift_report(location_shift=shift), INJECTED_FEATURE)
        for shift in (0.0, 0.25, 0.5, 1.0)
    ]

    psis = [finding.psi.value for finding in values]
    assert psis == sorted(psis), f"PSI is not monotone in the injected shift: {psis}"
    assert [str(finding.band) for finding in values] == ["stable", "stable", "moderate", "major"]


async def test_clause_1_positive_the_injection_reaches_the_operator_as_one_alert() -> None:
    """End to end: injected shift, reported drift, one CRITICAL alert, persisted and delivered.

    This is clause 1's actual claim. The assertions walk the whole path — the
    report, the rule set, the store, the channel — because "detected" means an
    operator was told, not that a float exceeded a constant somewhere.
    """
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    report = gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT)
    snapshot = MonitoringSnapshot(as_of=GATE_AS_OF, stamp=fixture_stamp(), drift=report)

    results = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)

    assert len(results) == 1
    dispatched = results[0]
    alert = dispatched.stored.alert
    assert alert.rule_id == "feature_drift"
    assert alert.severity is AlertSeverity.CRITICAL
    assert INJECTED_FEATURE in alert.subject
    assert "major" in alert.subject
    # Delivered, not merely written.
    assert dispatched.delivered_by == channel.channel_id
    assert [delivered.dedup_key for delivered in channel.delivered] == [alert.dedup_key]
    # And visible to the operator query the monitoring page is built on.
    open_alerts = await store.open_alerts(minimum_severity=AlertSeverity.CRITICAL)
    assert [item.alert.dedup_key for item in open_alerts] == [alert.dedup_key]


async def test_clause_1_the_same_injection_on_the_same_date_alerts_once() -> None:
    """A monitoring job restarting mid-incident does not re-page anybody.

    Part of the gate rather than a nicety: an alerting layer that fires on every
    cycle is one whose alerts get muted, and a muted detector has not detected
    anything.
    """
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    snapshot = MonitoringSnapshot(
        as_of=GATE_AS_OF,
        stamp=fixture_stamp(),
        drift=gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT),
    )

    first = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)
    second = await run_monitoring_cycle(
        snapshot,
        store=store,
        channels=[channel],
        now=DECISION_AT + dt.timedelta(hours=1),
    )

    assert first[0].stored.alert.dedup_key == second[0].stored.alert.dedup_key
    assert len(store.alerts) == 1


def test_clause_1_the_gate_uses_the_shipped_ladder_unweakened() -> None:
    """No threshold was moved to make this gate pass (directive §9.3, §0.2).

    The report under test is built by :func:`~backend.monitoring.drift.drift_report`
    with no ``bands`` argument, so it carries the shipped defaults. Asserted
    here explicitly so that weakening them later breaks the gate rather than
    silently re-scoring it.
    """
    report = gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT)

    assert report.bands == DriftBands()
    assert report.bands.moderate == DEFAULT_MODERATE_THRESHOLD
    assert (report.bands.moderate, report.bands.major) == (0.10, 0.25)
    assert "convention" in report.bands.basis


# ===========================================================================
# Clause 2 — an injected performance deviation triggers a halt
# ===========================================================================


def test_clause_2_negative_an_in_band_series_does_not_halt() -> None:
    """The control arm: a live window at the band's median keeps trading.

    Constructed from the same band, the same horizon and the same cost basis as
    the halting arm — the only difference between the two is where the window's
    Sharpe ratio was placed.
    """
    band = gate_band()
    decision = decide(
        band=band, live=in_band_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
    )

    assert decision.action is HaltAction.CONTINUE
    assert decision.cause is None
    assert not decision.should_halt
    assert decision.comparison is not None
    assert decision.comparison.within_band
    assert not decision.comparison.breaches_halting_side


def test_clause_2_negative_a_window_just_inside_the_edge_does_not_halt() -> None:
    """The halt is located at the band edge, not fired on approach.

    A window one hundredth of a band standard deviation *inside* the lower edge
    continues. Together with the halting arm one half-sigma outside it, this
    brackets the edge from both sides.
    """
    band = gate_band()
    inside = live_window(
        series_with_sharpe(band.lower + 0.01 * band.dispersion, n_periods=band.window_periods)
    )

    decision = decide(band=band, live=inside, stamp=fixture_stamp(), now=DECISION_AT)

    assert decision.action is HaltAction.CONTINUE
    assert decision.comparison is not None
    assert not decision.comparison.below_band


async def test_clause_2_negative_an_in_band_cycle_raises_no_alert() -> None:
    """A healthy cycle is silent — including when a stable drift report is in it.

    The full negative of both clauses in one snapshot: nothing injected
    anywhere, and the operator hears nothing. If this test ever starts producing
    an alert, every positive assertion in this module has lost its meaning.
    """
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    band = gate_band()
    snapshot = MonitoringSnapshot(
        as_of=GATE_AS_OF,
        stamp=fixture_stamp(),
        decision=decide(
            band=band, live=in_band_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
        ),
        drift=gate_drift_report(),
    )

    results = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)

    assert results == ()
    assert store.alerts == {}
    assert channel.delivered == []


async def test_clause_2_negative_a_continue_cannot_enter_the_halt_history() -> None:
    """A continue decision is refused by the halt log before the database is touched.

    The session handed in raises on any attribute access at all, so passing
    proves the refusal came from the argument check rather than from a
    constraint — and therefore that no row was written on the negative arm.
    """
    band = gate_band()
    decision = decide(
        band=band, live=in_band_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
    )

    with pytest.raises(HaltHistoryError, match="requires a halting decision"):
        await record_halt(_unusable_session(), decision)


def test_clause_2_positive_an_injected_deviation_halts_with_the_named_cause() -> None:
    """The clause itself: a placed deviation produces HALT / BELOW_EXPECTED_BAND.

    The live window's Sharpe ratio is set to exactly
    ``band.lower - 0.5 * band.dispersion``, so the assertion is about a distance
    the gate chose rather than about a series that happened to fall out.
    """
    band = gate_band()
    live = deviated_live_window(band)
    comparison = compare(band=band, live=live, now=DECISION_AT)

    # The injection landed where it was aimed, to floating point.
    assert comparison.value == pytest.approx(
        band.lower - DEVIATION_SIGMAS_BELOW_EDGE * band.dispersion
    )
    assert comparison.below_band
    assert comparison.breaches_halting_side

    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND
    assert decision.should_halt
    assert decision.as_of == GATE_AS_OF
    # The detail carries the numbers, not just the verdict: an operator reading
    # the halt row must see what was compared against what.
    assert f"{comparison.value:.6g}" in decision.detail
    assert f"{band.lower:.6g}" in decision.detail
    assert f"{band.upper:.6g}" in decision.detail
    assert band.artefact.artefact_id in decision.detail


def test_clause_2_positive_the_deviation_halts_on_a_second_statistic_too() -> None:
    """The halt is not an artefact of the Sharpe ratio's arithmetic.

    A mean-return band, cut from the same paths at the same horizon, halts on a
    live window placed the same half-sigma below its own lower edge.
    """
    band = fixture_band(statistic=WindowStatistic.MEAN_RETURN)
    assert band.statistic is WindowStatistic.MEAN_RETURN
    target = band.lower - DEVIATION_SIGMAS_BELOW_EDGE * band.dispersion
    # A zero-mean series shifted to `target`, so the window's mean return is
    # exactly the placed value.
    live = live_window(series_with_sharpe(0.0, n_periods=band.window_periods) + target)

    comparison = compare(band=band, live=live, now=DECISION_AT)
    assert comparison.value == pytest.approx(target)

    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND


def test_clause_2_positive_an_upside_deviation_halts_under_the_default_policy() -> None:
    """An injected deviation *above* the band halts too, and names its own cause.

    Beating a band cut from your own backtest is normally an accounting, scale
    or data error rather than good fortune, so the default two-sided policy
    stops trading for it. The gate checks the deviation is detected in the
    direction nobody wants to look at.
    """
    band = gate_band()
    high = live_window(
        series_with_sharpe(
            band.upper + DEVIATION_SIGMAS_BELOW_EDGE * band.dispersion,
            n_periods=band.window_periods,
        )
    )

    decision = decide(band=band, live=high, stamp=fixture_stamp(), now=DECISION_AT)

    assert band.policy.side is HaltSide.BOTH
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.ABOVE_EXPECTED_BAND


@pytest.mark.parametrize(
    ("condition", "cause"),
    [
        ("no_band", HaltCause.COMPARISON_UNAVAILABLE),
        ("no_live_window", HaltCause.COMPARISON_UNAVAILABLE),
        ("wrong_window_length", HaltCause.COMPARISON_UNAVAILABLE),
        ("stale_live_data", HaltCause.STALE_LIVE_DATA),
    ],
)
def test_clause_2_a_monitor_that_could_not_check_halts_rather_than_passing(
    condition: str, cause: HaltCause
) -> None:
    """The continue arm is narrow: only a completed, in-band comparison keeps trading.

    Included in the gate because "an injected deviation triggers a halt" is
    worth little if the surrounding failures quietly pass. Each row here is a
    way the check does not happen, and each one stops trading with its own
    cause. Every one of them is a condition whose live series is *in band* —
    the only difference from the continue arm is that the comparison could not
    be trusted.
    """
    band = gate_band()
    live = in_band_live_window(band)
    if condition == "no_band":
        decision = decide(band=None, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    elif condition == "no_live_window":
        decision = decide(band=band, live=None, stamp=fixture_stamp(), now=DECISION_AT)
    elif condition == "wrong_window_length":
        short = live_window(series_with_sharpe(band.median, n_periods=band.window_periods // 2))
        decision = decide(band=band, live=short, stamp=fixture_stamp(), now=DECISION_AT)
    else:
        stale = in_band_live_window(
            band, as_of=GATE_AS_OF - dt.timedelta(days=band.policy.max_live_age_days + 10)
        )
        decision = decide(band=band, live=stale, stamp=fixture_stamp(), now=DECISION_AT)

    assert decision.action is HaltAction.HALT, condition
    assert decision.cause is cause


@pytest.mark.parametrize("treatment", [CostTreatment.GROSS, CostTreatment.UNKNOWN])
def test_clause_2_a_gross_live_series_cannot_reach_the_comparison_at_all(
    treatment: CostTreatment,
) -> None:
    """I4 is enforced before the band is consulted, not by halting afterwards.

    A gross live series sits above a net-of-cost band by the cost drag for as
    long as costs are positive, which reads as "live is fine" — the most
    dangerous direction for this comparison to be wrong in. So the window is
    refused at construction and the gate's halting arm can never be a
    cost-accounting artefact.
    """
    band = gate_band()

    with pytest.raises(CostBasisError):
        live_window(
            series_with_sharpe(band.median, n_periods=band.window_periods),
            cost_treatment=treatment,
        )


async def test_clause_2_positive_the_halt_reaches_the_operator_as_one_critical_alert() -> None:
    """The halt is announced, persisted and delivered, carrying the decision payload.

    The alert's dedup key is what the ``monitoring_halt_event`` row records, and
    what the acknowledgement gate on resuming later reads — so this is the
    handshake the database half of the gate depends on.
    """
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    band = gate_band()
    decision = decide(
        band=band, live=deviated_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
    )
    snapshot = MonitoringSnapshot(as_of=GATE_AS_OF, stamp=fixture_stamp(), decision=decision)

    results = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)

    assert len(results) == 1
    alert = results[0].stored.alert
    assert alert.rule_id == "live_vs_expected"
    assert alert.severity is AlertSeverity.CRITICAL
    assert "Trading halted" in alert.subject
    assert str(HaltCause.BELOW_EXPECTED_BAND) in alert.subject
    assert alert.payload["cause"] == str(HaltCause.BELOW_EXPECTED_BAND)
    assert results[0].delivered_by == channel.channel_id
    assert channel.delivered == [alert]


async def test_both_clauses_in_one_cycle_produce_two_independent_alerts() -> None:
    """A single monitoring run carrying both injections says both things.

    One alert per condition, not one rolled-up "something is wrong": an operator
    acknowledges a feature or a halt, and a merged alert would be signed for
    once while the other condition ran on.
    """
    store = InMemoryAlertStore()
    channel = RecordingChannel()
    band = gate_band()
    snapshot = MonitoringSnapshot(
        as_of=GATE_AS_OF,
        stamp=fixture_stamp(),
        decision=decide(
            band=band, live=deviated_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
        ),
        drift=gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT),
    )

    results = await run_monitoring_cycle(snapshot, store=store, channels=[channel], now=DECISION_AT)

    rules = [result.stored.alert.rule_id for result in results]
    assert sorted(rules) == ["feature_drift", "live_vs_expected"]
    assert len({result.stored.alert.dedup_key for result in results}) == 2
    assert all(result.stored.alert.severity is AlertSeverity.CRITICAL for result in results)
    assert len(channel.delivered) == 2


# ===========================================================================
# What the gate does not establish, asserted so it cannot be quoted without it
# ===========================================================================


def test_the_gate_does_not_establish_that_this_is_a_powerful_test() -> None:
    """G12's own weakness, in numbers: the halt is slow at plausible deteriorations.

    P12.1 measured, and this re-measures: a strategy whose window statistic has
    fallen by one band standard deviation is caught in a given quarterly
    evaluation with probability ``0.11``, which is roughly nine evaluations —
    over two years — in expectation. G12 passing means the pipeline reacts to an
    injected condition. It does not mean real decay is caught in time to matter.
    """
    band = gate_band()

    power = {sigmas: band.detection_power(sigmas)["power"] for sigmas in (1.0, 2.0, 3.0)}
    assert power[1.0] == pytest.approx(0.11, abs=0.01)
    assert power[2.0] == pytest.approx(0.41, abs=0.01)
    assert power[3.0] == pytest.approx(0.78, abs=0.01)

    slowest = band.detection_power(1.0)
    assert slowest["expected_evaluations"] > 9.0
    assert slowest["expected_years"] > 2.0

    # And the deviation this gate halts on is far larger than the deterioration
    # a real strategy would decay through: it sits more than two and a half band
    # standard deviations below the band's own median. The gate is a
    # demonstration at a decisive magnitude, not a sensitivity measurement.
    injected_value = band.lower - DEVIATION_SIGMAS_BELOW_EDGE * band.dispersion
    injected_sigmas_below_median = (band.median - injected_value) / band.dispersion
    assert injected_sigmas_below_median > 2.5


def test_the_gate_does_not_cover_extraction_quality_drift() -> None:
    """P12.3 is not part of G12's stated condition and is not verified here.

    The directive's Phase 12 gate reads "injected drift is detected; injected
    performance deviation triggers halt". Extraction-quality drift on the golden
    set is a Phase 12 *deliverable* blocked on B3; passing G12 therefore does
    not make Phase 12 complete (§9.10). Asserted structurally — there is no
    golden-set drift surface in the monitoring package to import — so this
    statement cannot go stale silently.
    """
    exported = set(monitoring.__all__)
    assert not {name for name in exported if "golden" in name.lower()}
    assert not {name for name in exported if "extraction" in name.lower()}


def test_every_number_in_this_gate_is_synthetic_and_says_so() -> None:
    """I3: gate evidence must be unmistakably constructed, never presentable as real.

    Both clauses' artefacts are checked, because the failure this guards against
    is a plausible-looking synthetic result escaping into a report as if it were
    a measurement.
    """
    band = gate_band()
    report = gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT)
    live = deviated_live_window(band)

    assert FIXTURE_DATA_VERSION.startswith("FIXTURE")
    assert "not-a-data-version" in FIXTURE_DATA_VERSION
    assert report.stamp.data_version == FIXTURE_DATA_VERSION
    assert band.artefact.stamp.data_version == FIXTURE_DATA_VERSION
    assert band.artefact.artefact_id.startswith("FIXTURE_")
    assert live.source == FIXTURE_LIVE_SOURCE
    assert "no paper account exists" in live.source
    assert all(feature.startswith("FIXTURE_") for feature in GATE_FEATURES)
    for finding in report.measured:
        assert finding.reference_id.startswith("FIXTURE_")
        assert finding.psi.reference.units.startswith("fixture units")


def test_the_gate_evidence_carries_the_i2_stamp_end_to_end() -> None:
    """I2: every artefact this gate produces names the run that produced it.

    A gate result nobody can regenerate is an anecdote. The stamp's four
    components travel from the drift report and the halt decision into the alert
    payload, which is the object that reaches the operator and the database.
    """
    band = gate_band()
    report = gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT)
    decision = decide(
        band=band, live=deviated_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
    )
    snapshot = MonitoringSnapshot(
        as_of=GATE_AS_OF, stamp=fixture_stamp(), decision=decision, drift=report
    )
    alerts = evaluate_rules(default_rules(), snapshot, now=DECISION_AT)

    assert len(alerts) == 2
    for payload in (report.to_dict(), decision.to_dict(), *(alert.to_dict() for alert in alerts)):
        # Serialisable, because evidence that cannot be written down is not evidence.
        round_tripped = json.loads(json.dumps(payload))
        for component in ("git_reference", "data_version", "config_hash", "seed"):
            assert component in round_tripped, component
        assert round_tripped["data_version"] == FIXTURE_DATA_VERSION


def test_the_gate_ran_against_the_shipped_defaults() -> None:
    """Neither clause was passed by loosening the configuration it is judged under.

    The drift ladder and the halt policy under test are the ones the package
    ships. Directive §0.2: "Never mark a gate passed by weakening its test."
    """
    band = gate_band()

    assert band.policy == HaltPolicy()
    assert band.policy.window_periods == 63
    assert band.policy.annual_false_halt_budget == 0.10
    assert band.policy.side is HaltSide.BOTH
    assert band.policy.max_live_age_days == 4
    assert math.isfinite(band.policy.tail_mass)
    assert gate_drift_report().bands == DriftBands()
