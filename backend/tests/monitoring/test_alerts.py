"""Tests for the alert rules, dispatch and escalation (P12.4).

Three claims, and the tests are grouped by them:

1. **A rule fires once per condition.** Asserted through the *dedup key*, which
   is where the property actually lives: the same condition evaluated twice —
   by a monitor that restarted, with different wall-clock times — must produce
   the same key, and a different condition must not.

2. **Delivery does not silently drop.** The alert is persisted *before* any
   channel is called (asserted by dispatching to a channel that raises and then
   reading the alert back), a channel that fails escalates to the next, and
   total failure raises rather than returning. A channel that raises something
   other than ``AlertDeliveryError`` — the ordinary case for a third-party
   client — is asserted to be caught too.

3. **Acknowledgement is state.** Covered as a store property in
   ``store_contract.py``; here it is the *escalation* side that is asserted —
   an alert nobody signed for is found by
   :func:`~backend.monitoring.alerts.pending_escalations`, and escalating it
   produces one new alert per round rather than one per poll.

All fixtures are synthetic (I3 — ``alert_fixtures.py``, ``expectation_fixtures.py``).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from backend.monitoring.alerts import (
    DEFAULT_ESCALATION_AFTER,
    Alert,
    AlertSeverity,
    DriftReportRule,
    HaltDecisionRule,
    MonitoringSnapshot,
    MonitorSilenceRule,
    StructlogChannel,
    alert_dedup_key,
    default_rules,
    dispatch,
    escalate,
    evaluate_rules,
    pending_escalations,
)
from backend.monitoring.drift import DriftBand, DriftBands, drift_report
from backend.monitoring.errors import AlertError, AlertUndeliverableError
from backend.monitoring.expectation import HaltAction, HaltCause, HaltPolicy, HaltSide, decide
from backend.tests.monitoring.alert_fixtures import LATER, RAISED_AT, fixture_alert
from backend.tests.monitoring.doubles import (
    ExplodingChannel,
    FailingChannel,
    InMemoryAlertStore,
    RecordingChannel,
)
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    LIVE_AS_OF,
    fixture_band,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import (
    OBSERVED_SIZE,
    fixture_stamp,
    gaussian_reference,
    gaussian_sample,
)

# ---------------------------------------------------------------------------
# 1. One condition, one alert
# ---------------------------------------------------------------------------


def test_the_same_condition_produces_the_same_key_at_a_different_time() -> None:
    """The mechanism: the key is content, so a restarted monitor re-derives it."""
    condition = {"as_of": "2026-08-01", "cause": "below_expected_band"}
    first = alert_dedup_key(rule_id="live_vs_expected", condition=condition)
    second = alert_dedup_key(
        rule_id="live_vs_expected", condition=dict(reversed(list(condition.items())))
    )
    assert first == second
    assert len(first) == 64


def test_a_different_condition_produces_a_different_key() -> None:
    base = {"as_of": "2026-08-01", "cause": "below_expected_band"}
    other = {"as_of": "2026-08-02", "cause": "below_expected_band"}
    different_cause = {"as_of": "2026-08-01", "cause": "comparison_unavailable"}
    keys = {
        alert_dedup_key(rule_id="live_vs_expected", condition=base),
        alert_dedup_key(rule_id="live_vs_expected", condition=other),
        alert_dedup_key(rule_id="live_vs_expected", condition=different_cause),
        alert_dedup_key(rule_id="feature_drift", condition=base),
    }
    assert len(keys) == 4


def _halting_snapshot() -> MonitoringSnapshot:
    """Build a snapshot whose decision is a halt from an injected deviation."""
    band = fixture_band()
    live = live_window(
        series_with_sharpe(band.lower - 0.2 * band.dispersion, n_periods=band.window_periods)
    )
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    return MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp(), decision=decision)


def test_a_halt_raises_one_critical_alert() -> None:
    alerts = HaltDecisionRule().evaluate(_halting_snapshot(), now=RAISED_AT)
    assert len(alerts) == 1
    assert alerts[0].severity is AlertSeverity.CRITICAL
    assert "Trading halted" in alerts[0].subject
    assert alerts[0].payload["cause"] == "below_expected_band"


def test_the_same_halt_evaluated_twice_yields_one_stored_alert() -> None:
    """Evaluated at two different instants, absorbed into one row by the store."""
    snapshot = _halting_snapshot()
    first = HaltDecisionRule().evaluate(snapshot, now=RAISED_AT)[0]
    second = HaltDecisionRule().evaluate(snapshot, now=LATER)[0]
    assert first.dedup_key == second.dedup_key
    assert first.raised_at != second.raised_at


async def test_a_repeat_evaluation_does_not_produce_a_second_row() -> None:
    store = InMemoryAlertStore()
    snapshot = _halting_snapshot()
    alert = HaltDecisionRule().evaluate(snapshot, now=RAISED_AT)[0]
    channel = RecordingChannel()
    await dispatch(alert, store=store, channels=[channel], now=RAISED_AT)
    repeat = HaltDecisionRule().evaluate(snapshot, now=LATER)[0]
    await dispatch(repeat, store=store, channels=[channel], now=LATER)
    assert len(store.alerts) == 1


def test_a_continue_decision_raises_nothing() -> None:
    band = fixture_band()
    live = live_window(series_with_sharpe(band.median, n_periods=band.window_periods))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    snapshot = MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp(), decision=decision)
    assert HaltDecisionRule().evaluate(snapshot, now=RAISED_AT) == ()


def test_a_non_halting_upper_breach_still_warns() -> None:
    """The one-sided policy continues trading; it does not continue silently."""
    band = fixture_band(policy=HaltPolicy(side=HaltSide.LOWER))
    live = live_window(
        series_with_sharpe(band.upper + 0.5 * band.dispersion, n_periods=band.window_periods)
    )
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.CONTINUE
    snapshot = MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp(), decision=decision)
    alerts = HaltDecisionRule().evaluate(snapshot, now=RAISED_AT)
    assert len(alerts) == 1
    assert alerts[0].severity is AlertSeverity.WARNING
    assert "above the expected band" in alerts[0].subject


def test_an_unavailable_comparison_alerts_as_critically_as_underperformance() -> None:
    """Both stopped trading; both need the same human."""
    decision = decide(band=None, live=None, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.cause is HaltCause.COMPARISON_UNAVAILABLE
    snapshot = MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp(), decision=decision)
    alerts = HaltDecisionRule().evaluate(snapshot, now=RAISED_AT)
    assert alerts[0].severity is AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 2. Drift and silence rules
# ---------------------------------------------------------------------------


def _drift_snapshot(*, shift: float, thin: bool = False) -> MonitoringSnapshot:
    """Build a snapshot carrying a drift report over one or two fixture features."""
    stable_reference = gaussian_reference(feature="FIXTURE_stable", reference_id="FIXTURE_ref_a")
    moved_reference = gaussian_reference(feature="FIXTURE_moved", reference_id="FIXTURE_ref_b")
    observations: list[tuple[object, object]] = [
        (stable_reference, gaussian_sample(seed=5, size=OBSERVED_SIZE)),
        (moved_reference, gaussian_sample(seed=6, size=OBSERVED_SIZE, shift=shift)),
    ]
    if thin:
        thin_reference = gaussian_reference(feature="FIXTURE_thin", reference_id="FIXTURE_ref_c")
        observations.append((thin_reference, np.asarray([0.1, 0.2, 0.3], dtype=np.float64)))
    report = drift_report(
        as_of=LIVE_AS_OF,
        stamp=fixture_stamp(),
        observations=observations,  # type: ignore[arg-type]
        bands=DriftBands(),
    )
    return MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp(), drift=report)


def test_one_alert_per_drifting_feature_not_one_per_report() -> None:
    """An operator acknowledges a feature; a rolled-up alert would be signed once."""
    snapshot = _drift_snapshot(shift=1.5)
    alerts = DriftReportRule().evaluate(snapshot, now=RAISED_AT)
    assert len(alerts) == 1
    assert "FIXTURE_moved" in alerts[0].subject
    assert alerts[0].severity is AlertSeverity.CRITICAL
    assert len({alert.dedup_key for alert in alerts}) == len(alerts)


def test_a_stable_panel_raises_nothing() -> None:
    """The other direction: the rule must be capable of staying quiet."""
    assert DriftReportRule().evaluate(_drift_snapshot(shift=0.0), now=RAISED_AT) == ()


def test_a_lower_threshold_admits_moderate_drift() -> None:
    snapshot = _drift_snapshot(shift=0.35)
    assert DriftReportRule().evaluate(snapshot, now=RAISED_AT) == ()
    lenient = DriftReportRule(minimum_band=DriftBand.MODERATE).evaluate(snapshot, now=RAISED_AT)
    assert len(lenient) == 1
    assert lenient[0].severity is AlertSeverity.WARNING


def test_an_incomplete_drift_report_is_itself_an_alert() -> None:
    """D-031: a feature whose data broke is the one most likely to have moved."""
    alerts = DriftReportRule().evaluate(_drift_snapshot(shift=0.0, thin=True), now=RAISED_AT)
    assert len(alerts) == 1
    assert "incomplete" in alerts[0].subject
    assert "FIXTURE_thin" in alerts[0].detail
    assert "NOT stable" in alerts[0].detail


def test_a_monitoring_run_that_observed_nothing_is_critical() -> None:
    """A silent monitor and a healthy system look identical from the dashboard."""
    empty = MonitoringSnapshot(as_of=LIVE_AS_OF, stamp=fixture_stamp())
    alerts = MonitorSilenceRule().evaluate(empty, now=RAISED_AT)
    assert len(alerts) == 1
    assert alerts[0].severity is AlertSeverity.CRITICAL
    assert "no observations" in alerts[0].subject


def test_the_silence_rule_stays_quiet_when_anything_was_observed() -> None:
    assert MonitorSilenceRule().evaluate(_halting_snapshot(), now=RAISED_AT) == ()


def test_the_default_rule_set_covers_halt_drift_and_silence() -> None:
    snapshot = _halting_snapshot()
    with_drift = MonitoringSnapshot(
        as_of=snapshot.as_of,
        stamp=snapshot.stamp,
        decision=snapshot.decision,
        drift=_drift_snapshot(shift=1.5).drift,
    )
    alerts = evaluate_rules(default_rules(), with_drift, now=RAISED_AT)
    assert {alert.rule_id for alert in alerts} == {"live_vs_expected", "feature_drift"}
    assert len({alert.dedup_key for alert in alerts}) == len(alerts)


def test_duplicate_keys_across_rules_collapse_to_one_alert() -> None:
    same = fixture_alert(condition="shared")

    class Echo:
        rule_id = "echo"

        def evaluate(self, snapshot: MonitoringSnapshot, *, now: dt.datetime) -> tuple[Alert, ...]:
            assert snapshot is not None
            assert now is not None
            return (same,)

    alerts = evaluate_rules([Echo(), Echo()], _halting_snapshot(), now=RAISED_AT)
    assert len(alerts) == 1


def test_every_alert_payload_is_json_serialisable() -> None:
    """The payload is what the dashboard renders and what the row stores."""
    for alert in HaltDecisionRule().evaluate(_halting_snapshot(), now=RAISED_AT):
        encoded = json.loads(json.dumps(alert.to_dict()))
        assert encoded["payload"]["comparison"]["band"]["artefact"]["trials"] >= 1


def test_an_alert_without_an_i2_stamp_cannot_be_constructed() -> None:
    with pytest.raises(AlertError, match="ReproducibilityStamp"):
        Alert(
            rule_id="r",
            severity=AlertSeverity.INFO,
            subject="s",
            detail="d",
            dedup_key="0" * 64,
            payload={},
            stamp="not a stamp",  # type: ignore[arg-type]
            raised_at=RAISED_AT,
        )


def test_a_naive_raised_at_is_refused() -> None:
    with pytest.raises(AlertError, match="timezone-naive"):
        fixture_alert(condition="naive", raised_at=dt.datetime(2026, 8, 3, 9, 30))  # noqa: DTZ001


# ---------------------------------------------------------------------------
# 3. Delivery: persist first, escalate to the next channel, never drop
# ---------------------------------------------------------------------------


async def test_an_alert_is_persisted_before_any_channel_is_called() -> None:
    """The ordering that survives a process dying mid-dispatch."""
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="persist-first")
    failing = FailingChannel()
    with pytest.raises(AlertUndeliverableError):
        await dispatch(alert, store=store, channels=[failing], now=RAISED_AT)
    stored = await store.get(alert.dedup_key)
    assert stored.alert.dedup_key == alert.dedup_key
    assert stored.escalated
    assert stored.attempts[0].detail == failing.detail


async def test_the_first_accepting_channel_stops_the_walk() -> None:
    """Later channels are fallbacks, not copies."""
    store = InMemoryAlertStore()
    primary = RecordingChannel(channel_id="primary")
    secondary = RecordingChannel(channel_id="secondary")
    result = await dispatch(
        fixture_alert(condition="fallback"),
        store=store,
        channels=[primary, secondary],
        now=RAISED_AT,
    )
    assert result.delivered_by == "primary"
    assert len(primary.delivered) == 1
    assert secondary.delivered == []


async def test_a_failing_channel_escalates_to_the_next_and_the_failure_is_recorded() -> None:
    store = InMemoryAlertStore()
    failing = FailingChannel()
    backup = RecordingChannel(channel_id="backup")
    alert = fixture_alert(condition="failover")
    result = await dispatch(alert, store=store, channels=[failing, backup], now=RAISED_AT)
    assert result.delivered_by == "backup"
    assert result.failures == ((failing.channel_id, failing.detail),)
    stored = await store.get(alert.dedup_key)
    assert [attempt.outcome.value for attempt in stored.attempts] == ["failed", "delivered"]
    assert stored.delivered
    assert not stored.escalated


async def test_a_channel_that_raises_something_else_is_still_a_failure() -> None:
    """A third-party client raising TimeoutError is the ordinary case."""
    store = InMemoryAlertStore()
    backup = RecordingChannel(channel_id="backup")
    alert = fixture_alert(condition="exploding")
    result = await dispatch(
        alert, store=store, channels=[ExplodingChannel(), backup], now=RAISED_AT
    )
    assert result.delivered_by == "backup"
    assert result.failures[0][0] == "exploding"
    assert "TimeoutError" in result.failures[0][1]


async def test_an_undeliverable_alert_raises_rather_than_returning() -> None:
    """Recording alone would let a caller ignoring the result treat failure as success."""
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="undeliverable")
    with pytest.raises(AlertUndeliverableError) as raised:
        await dispatch(
            alert,
            store=store,
            channels=[FailingChannel(channel_id="a"), FailingChannel(channel_id="b")],
            now=RAISED_AT,
        )
    assert raised.value.dedup_key == alert.dedup_key
    assert {channel for channel, _ in raised.value.failures} == {"a", "b"}
    assert "persisted and escalated" in str(raised.value)
    assert len((await store.get(alert.dedup_key)).attempts) == 2


async def test_dispatching_with_no_channels_is_total_failure() -> None:
    """An alerting system with no configured channel is not an alerting system."""
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="no-channels")
    with pytest.raises(AlertUndeliverableError):
        await dispatch(alert, store=store, channels=[], now=RAISED_AT)
    assert (await store.get(alert.dedup_key)).alert.dedup_key == alert.dedup_key


async def test_the_structlog_channel_delivers_and_returns_a_receipt() -> None:
    store = InMemoryAlertStore()
    result = await dispatch(
        fixture_alert(condition="structlog"),
        store=store,
        channels=[StructlogChannel()],
        now=RAISED_AT,
    )
    assert result.delivered_by == "structlog"
    assert result.stored.attempts[0].detail == "monitoring.alert"


# ---------------------------------------------------------------------------
# 4. Escalation of what nobody acknowledged
# ---------------------------------------------------------------------------


async def test_an_unacknowledged_alert_escalates_after_the_deadline() -> None:
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="stale-ack")
    await dispatch(alert, store=store, channels=[RecordingChannel()], now=RAISED_AT)

    early = await pending_escalations(
        store, now=RAISED_AT + dt.timedelta(minutes=5), after=DEFAULT_ESCALATION_AFTER
    )
    assert early == ()

    late = await pending_escalations(store, now=LATER, after=DEFAULT_ESCALATION_AFTER)
    assert [stored.alert.dedup_key for stored in late] == [alert.dedup_key]


async def test_an_acknowledged_alert_never_escalates() -> None:
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="acked")
    await dispatch(alert, store=store, channels=[RecordingChannel()], now=RAISED_AT)
    await store.acknowledge(
        dedup_key=alert.dedup_key,
        acknowledged_by="operator@example.invalid",
        note="looked at it",
        now=RAISED_AT,
    )
    assert await pending_escalations(store, now=LATER) == ()


async def test_an_undelivered_alert_escalates_immediately() -> None:
    """No deadline applies to an alert that reached nobody."""
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="never-delivered")
    with pytest.raises(AlertUndeliverableError):
        await dispatch(alert, store=store, channels=[FailingChannel()], now=RAISED_AT)
    pending = await pending_escalations(store, now=RAISED_AT + dt.timedelta(minutes=1))
    assert [stored.alert.dedup_key for stored in pending] == [alert.dedup_key]


async def test_escalation_raises_the_severity_and_fires_once_per_round() -> None:
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="escalate-me", severity=AlertSeverity.WARNING)
    await dispatch(alert, store=store, channels=[RecordingChannel()], now=RAISED_AT)
    stored = (await pending_escalations(store, now=LATER))[0]

    channel = RecordingChannel(channel_id="pager")
    first = await escalate(stored, store=store, channels=[channel], now=LATER, round_number=1)
    assert first.stored.alert.severity is AlertSeverity.CRITICAL
    assert first.stored.alert.rule_id.endswith(".escalation")
    assert "ESCALATION 1" in first.stored.alert.subject

    # Polling again in the same round must not page anyone a second time.
    repeat = await escalate(stored, store=store, channels=[channel], now=LATER, round_number=1)
    assert repeat.stored.alert.dedup_key == first.stored.alert.dedup_key
    assert len(store.alerts) == 2

    second_round = await escalate(
        stored, store=store, channels=[channel], now=LATER, round_number=2
    )
    assert second_round.stored.alert.dedup_key != first.stored.alert.dedup_key
    assert len(store.alerts) == 3


async def test_escalating_an_undelivered_alert_says_so() -> None:
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="undelivered-escalation")
    with pytest.raises(AlertUndeliverableError):
        await dispatch(alert, store=store, channels=[FailingChannel()], now=RAISED_AT)
    stored = await store.get(alert.dedup_key)
    result = await escalate(
        stored, store=store, channels=[RecordingChannel()], now=LATER, round_number=1
    )
    assert "no channel accepted" in result.stored.alert.detail


async def test_a_round_number_below_one_is_refused() -> None:
    store = InMemoryAlertStore()
    alert = fixture_alert(condition="bad-round")
    await dispatch(alert, store=store, channels=[RecordingChannel()], now=RAISED_AT)
    stored = await store.get(alert.dedup_key)
    with pytest.raises(AlertError, match="round_number"):
        await escalate(
            stored, store=store, channels=[RecordingChannel()], now=LATER, round_number=0
        )
