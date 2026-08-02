"""Synthetic alert fixtures (I3: nothing here describes a real incident).

Every alert built here names a ``FIXTURE_`` rule, carries the fixture
reproducibility stamp whose ``data_version`` announces itself as not a data
version, and describes a condition that never happened. An acknowledgement
recorded against one of these is a statement about the store, never about an
operator.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

from backend.monitoring.alerts import Alert, AlertSeverity, alert_dedup_key
from backend.tests.monitoring.fixtures import fixture_stamp

if TYPE_CHECKING:
    from backend.monitoring.psi import JsonValue

FIXTURE_RULE_ID: Final = "FIXTURE_rule"
"""Stands where a rule identifier would, and says it is a fixture."""

RAISED_AT: Final = dt.datetime(2026, 8, 3, 9, 30, tzinfo=dt.UTC)
"""When the fixture conditions were 'observed'."""

LATER: Final = dt.datetime(2026, 8, 3, 18, 0, tzinfo=dt.UTC)
"""A later instant, past :data:`~backend.monitoring.alerts.DEFAULT_ESCALATION_AFTER`."""


def fixture_alert(
    *,
    condition: str = "synthetic-condition-a",
    severity: AlertSeverity = AlertSeverity.CRITICAL,
    rule_id: str = FIXTURE_RULE_ID,
    raised_at: dt.datetime = RAISED_AT,
) -> Alert:
    """Build a synthetic alert whose dedup key is derived from ``condition``.

    Args:
        condition: the fixture condition identity. Two alerts with the same
            condition share a dedup key, which is the property the store tests
            exercise.
        severity: how urgent the fixture alert claims to be.
        rule_id: the fixture rule identifier.
        raised_at: when the condition was 'observed' (UTC).

    Returns:
        A valid :class:`~backend.monitoring.alerts.Alert`.
    """
    payload: dict[str, JsonValue] = {"condition": condition, "fixture": True}
    return Alert(
        rule_id=rule_id,
        severity=severity,
        subject=f"FIXTURE alert for {condition}",
        detail=f"FIXTURE condition {condition}; no real incident is described here (I3).",
        dedup_key=alert_dedup_key(rule_id=rule_id, condition={"fixture_condition": condition}),
        payload=payload,
        stamp=fixture_stamp(),
        raised_at=raised_at,
    )
