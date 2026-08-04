"""The :class:`~backend.monitoring.alerts.AlertStore` contract, run against every implementation.

One suite, two backends. ``test_alerts_store_memory.py`` runs it against the
in-memory double; ``backend/tests/integration/test_monitoring_alerts_db.py``
runs the identical assertions against
:class:`~backend.monitoring.alerts.PostgresAlertStore` on a real TimescaleDB
container. Writing the assertions once is the point: D-034's failure
was a double that quietly disagreed with the database, and two hand-written
suites would have hidden that just as well.

The module name deliberately does not begin with ``test_``, so pytest collects
these bodies only through the two runners rather than as tests of their own.

What this contract covers, in the order the properties matter:

1. a repeat of one condition is absorbed — one row, one acknowledgement;
2. delivery attempts accumulate as a history, failures included;
3. an acknowledgement persists, is readable afterwards, and cannot be replaced;
4. an unknown alert is a refusal, never an empty result;
5. blank text is refused before it can become an unactionable record.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.monitoring.alerts import AlertSeverity, AlertStore, DeliveryOutcome
from backend.monitoring.errors import AlertError, AlreadyAcknowledgedError, UnknownAlertError
from backend.tests.monitoring.alert_fixtures import LATER, RAISED_AT, fixture_alert


async def assert_a_repeat_condition_is_absorbed(store: AlertStore) -> None:
    """One condition produces one alert however often the monitor re-observes it."""
    alert = fixture_alert(condition="repeat")
    first = await store.record(alert, now=RAISED_AT)
    assert first.is_new

    # The same condition, observed again an hour later by a monitor that
    # restarted: a fresh Alert object, an identical dedup key.
    again = fixture_alert(condition="repeat", raised_at=RAISED_AT + dt.timedelta(hours=1))
    assert again.dedup_key == alert.dedup_key
    second = await store.record(again, now=RAISED_AT + dt.timedelta(hours=1))
    assert not second.is_new
    assert second.alert_id == first.alert_id
    # The stored copy keeps the FIRST observation's timestamp: an incident began
    # once, and re-observing it does not move when it started.
    assert second.alert.raised_at == RAISED_AT


async def assert_a_distinct_condition_is_a_distinct_alert(store: AlertStore) -> None:
    """Deduplication must not merge two different findings."""
    one = await store.record(fixture_alert(condition="alpha"), now=RAISED_AT)
    two = await store.record(fixture_alert(condition="beta"), now=RAISED_AT)
    assert one.alert_id != two.alert_id


async def assert_delivery_attempts_accumulate_as_a_history(store: AlertStore) -> None:
    """Failures are rows. "Which alerts reached nobody" must be a query."""
    alert = fixture_alert(condition="delivery")
    await store.record(alert, now=RAISED_AT)
    first = await store.record_attempt(
        dedup_key=alert.dedup_key,
        channel_id="pager",
        outcome=DeliveryOutcome.FAILED,
        detail="simulated outage",
        now=RAISED_AT,
    )
    second = await store.record_attempt(
        dedup_key=alert.dedup_key,
        channel_id="pager",
        outcome=DeliveryOutcome.FAILED,
        detail="simulated outage again",
        now=RAISED_AT,
    )
    other = await store.record_attempt(
        dedup_key=alert.dedup_key,
        channel_id="email",
        outcome=DeliveryOutcome.DELIVERED,
        detail="message-id 1",
        now=RAISED_AT,
    )
    assert (first.attempt, second.attempt) == (1, 2)
    assert other.attempt == 1  # the counter is per channel, not per alert

    stored = await store.get(alert.dedup_key)
    assert len(stored.attempts) == 3
    assert stored.delivered  # one channel accepted it
    assert not stored.escalated


async def assert_an_alert_no_channel_accepted_reads_as_escalated(store: AlertStore) -> None:
    """Escalated means "attempted and refused" — not "never dispatched"."""
    alert = fixture_alert(condition="escalated")
    stored = await store.record(alert, now=RAISED_AT)
    assert not stored.escalated  # no attempts yet: undispatched, a different condition
    await store.record_attempt(
        dedup_key=alert.dedup_key,
        channel_id="pager",
        outcome=DeliveryOutcome.FAILED,
        detail="simulated outage",
        now=RAISED_AT,
    )
    after = await store.get(alert.dedup_key)
    assert after.escalated
    assert not after.delivered


async def assert_an_acknowledgement_persists_and_is_read_back(store: AlertStore) -> None:
    """Acknowledgement is state, and the store is where it lives."""
    alert = fixture_alert(condition="acknowledge")
    await store.record(alert, now=RAISED_AT)
    assert not (await store.get(alert.dedup_key)).acknowledged

    signature = await store.acknowledge(
        dedup_key=alert.dedup_key,
        acknowledged_by="operator@example.invalid",
        note="reviewed the CPCV band and the reconciliation; investigating",
        now=LATER,
    )
    assert signature.acknowledged_by == "operator@example.invalid"

    stored = await store.get(alert.dedup_key)
    assert stored.acknowledged
    assert stored.acknowledgement is not None
    assert stored.acknowledgement.acknowledged_by == "operator@example.invalid"
    assert stored.acknowledgement.acknowledged_at == LATER
    assert "investigating" in stored.acknowledgement.note


async def assert_a_second_acknowledgement_is_refused(store: AlertStore) -> None:
    """Who took responsibility is never silently rewritten."""
    alert = fixture_alert(condition="double-acknowledge")
    await store.record(alert, now=RAISED_AT)
    await store.acknowledge(
        dedup_key=alert.dedup_key,
        acknowledged_by="first@example.invalid",
        note="taking this",
        now=LATER,
    )
    with pytest.raises(AlreadyAcknowledgedError) as raised:
        await store.acknowledge(
            dedup_key=alert.dedup_key,
            acknowledged_by="second@example.invalid",
            note="also taking this",
            now=LATER,
        )
    assert raised.value.acknowledged_by == "first@example.invalid"
    stored = await store.get(alert.dedup_key)
    assert stored.acknowledgement is not None
    assert stored.acknowledgement.acknowledged_by == "first@example.invalid"


async def assert_open_alerts_excludes_acknowledged_ones(store: AlertStore) -> None:
    """The operator's queue is "what has nobody signed for", filterable by severity."""
    critical = fixture_alert(condition="open-critical", severity=AlertSeverity.CRITICAL)
    warning = fixture_alert(condition="open-warning", severity=AlertSeverity.WARNING)
    signed = fixture_alert(condition="open-signed", severity=AlertSeverity.CRITICAL)
    for alert in (critical, warning, signed):
        await store.record(alert, now=RAISED_AT)
    await store.acknowledge(
        dedup_key=signed.dedup_key,
        acknowledged_by="operator@example.invalid",
        note="handled",
        now=LATER,
    )

    keys = {stored.alert.dedup_key for stored in await store.open_alerts()}
    assert keys == {critical.dedup_key, warning.dedup_key}

    severe = await store.open_alerts(minimum_severity=AlertSeverity.CRITICAL)
    assert {stored.alert.dedup_key for stored in severe} == {critical.dedup_key}


async def assert_an_unknown_alert_is_a_refusal_not_an_empty_result(store: AlertStore) -> None:
    """A missing alert must not read as an alert with nothing wrong."""
    with pytest.raises(UnknownAlertError):
        await store.get("0" * 64)
    with pytest.raises(UnknownAlertError):
        await store.acknowledge(dedup_key="0" * 64, acknowledged_by="x", note="y", now=RAISED_AT)
    with pytest.raises(UnknownAlertError):
        await store.record_attempt(
            dedup_key="0" * 64,
            channel_id="pager",
            outcome=DeliveryOutcome.FAILED,
            detail="nowhere to record this",
            now=RAISED_AT,
        )


async def assert_blank_text_is_refused(store: AlertStore) -> None:
    """A failure with no reason, or a signature with no note, is not a record."""
    alert = fixture_alert(condition="blank-text")
    await store.record(alert, now=RAISED_AT)
    with pytest.raises(AlertError):
        await store.record_attempt(
            dedup_key=alert.dedup_key,
            channel_id="pager",
            outcome=DeliveryOutcome.FAILED,
            detail="   ",
            now=RAISED_AT,
        )
    with pytest.raises(AlertError):
        await store.acknowledge(
            dedup_key=alert.dedup_key,
            acknowledged_by="operator@example.invalid",
            note="",
            now=LATER,
        )
    with pytest.raises(AlertError):
        await store.acknowledge(
            dedup_key=alert.dedup_key, acknowledged_by="", note="something", now=LATER
        )


CONTRACT = (
    assert_a_repeat_condition_is_absorbed,
    assert_a_distinct_condition_is_a_distinct_alert,
    assert_delivery_attempts_accumulate_as_a_history,
    assert_an_alert_no_channel_accepted_reads_as_escalated,
    assert_an_acknowledgement_persists_and_is_read_back,
    assert_a_second_acknowledgement_is_refused,
    assert_open_alerts_excludes_acknowledged_ones,
    assert_an_unknown_alert_is_a_refusal_not_an_empty_result,
    assert_blank_text_is_refused,
)
"""Every contract body, so a runner cannot silently omit one.

Both runners assert their own test count against ``len(CONTRACT)``: adding a
property here without wiring it into both backends fails the build rather than
quietly covering one implementation.
"""
