"""Alerting: rules, delivery, acknowledgement (P12.4).

An alert nobody acknowledges is a log line
------------------------------------------

That sentence is the whole design. Three things follow from it, and each is
enforced by a structure rather than by a convention:

**1. Firing is idempotent per condition, not per poll.** An alert's identity is
a SHA-256 digest of the *condition* — the rule, the date, and the specific thing
that was wrong — never a counter, a timestamp or a UUID
(:func:`alert_dedup_key`). A monitoring job that runs every hour and finds the
same halt condition raises the same key every hour, and the store's ``UNIQUE
(dedup_key)`` absorbs the repeats: one row, one acknowledgement, one thing for
an operator to act on. The reasoning is D-033's, one domain over: a counter must
be *remembered* across a restart, and the process that restarted mints a new one
and pages someone twice for one condition. Content needs no memory.

**2. Delivery cannot silently drop.** Three separate properties:

* An alert is **persisted before any delivery is attempted.** If the process
  dies mid-dispatch, the alert exists; only the notification was lost, and the
  absence of a successful attempt row is queryable.
* A channel **raises** (:class:`~backend.monitoring.errors.AlertDeliveryError`)
  rather than returning a status. A boolean return is a boolean a caller can
  ignore, and the ignored branch is where alerts go missing.
* When every channel refuses, :func:`dispatch` records each failure, marks the
  alert escalated, **and raises**
  :class:`~backend.monitoring.errors.AlertUndeliverableError`. Recording alone
  would let a caller that ignores the return value treat total delivery failure
  as success; raising alone would lose the audit trail. Both, or the alert can
  disappear.

**3. Acknowledgement is state, and it is persisted state.** One row, one
acknowledger, one instant, and a ``UNIQUE (alert_id)`` that makes a second
acknowledgement a refusal rather than an overwrite — an acknowledgement records
*who took responsibility*, and silently replacing that name is worse than losing
it. Acknowledgement is never a boolean column on the alert: it is derived from
the presence of a row, for the same reason
:mod:`backend.execution.store` derives order state from its transition log.

And it is load-bearing rather than decorative:
:func:`backend.monitoring.history.record_resume` refuses to resume trading from a
halt whose alert has no acknowledgement row. If nobody has signed for the alert,
nobody has read it, and the halt stands.

What escalation means here
--------------------------

Two distinct escalations, because there are two ways an alert dies:

* **Undelivered** — no channel accepted it. Handled by :func:`dispatch` as
  above.
* **Unacknowledged** — it was delivered and nobody responded.
  :func:`pending_escalations` finds those past a deadline, and
  :func:`escalate` re-raises each as a *new* alert at a higher severity, keyed on
  the original condition **plus the escalation round**, so a poll every ten
  minutes produces one escalation per round rather than one per poll.

The store is a Protocol with two implementations
------------------------------------------------

:class:`AlertStore` is the seam, :class:`PostgresAlertStore` is the real one, and
the tests carry an in-memory double. D-034 is the reason the double is not
trusted on its own: a double that models the *wrong layer* — the primary key
when a trigger is what actually refuses, or a refusal with no SQLSTATE — makes a
broken store indistinguishable from a correct one. So the two are exercised by
**one** contract suite (``backend/tests/monitoring/store_contract.py``), the
double is itself under test, and the Postgres side is the authority whenever
they can both be run.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import sqlalchemy as sa
import structlog
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db.models import MonitoringAlert as MonitoringAlertRow
from backend.db.models import MonitoringAlertAcknowledgement as AcknowledgementRow
from backend.db.models import MonitoringAlertDelivery as DeliveryRow
from backend.monitoring.drift import DriftBand, DriftReport
from backend.monitoring.errors import (
    AlertDeliveryError,
    AlertError,
    AlertUndeliverableError,
    AlreadyAcknowledgedError,
    UnknownAlertError,
)
from backend.monitoring.expectation import HaltDecision
from backend.monitoring.psi import JsonValue
from backend.tracking.stamp import ReproducibilityStamp, canonical_config_hash

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "DEFAULT_ESCALATION_AFTER",
    "UNIQUE_VIOLATION_SQLSTATE",
    "Acknowledgement",
    "Alert",
    "AlertChannel",
    "AlertRule",
    "AlertSeverity",
    "AlertStore",
    "DeliveryAttempt",
    "DeliveryOutcome",
    "DispatchResult",
    "DriftReportRule",
    "HaltDecisionRule",
    "MonitorSilenceRule",
    "MonitoringSnapshot",
    "StoredAlert",
    "StructlogChannel",
    "alert_dedup_key",
    "default_rules",
    "dispatch",
    "escalate",
    "evaluate_rules",
    "pending_escalations",
]

UNIQUE_VIOLATION_SQLSTATE: Final = "23505"
"""SQLSTATE ``unique_violation`` — what a repeat condition looks like arriving from Postgres.

Decided on the SQLSTATE rather than on the exception class, per D-034: a CHECK
violation is an ``IntegrityError`` too, and treating one as "this alert already
exists" would return a stored alert that does not exist.
"""

DEFAULT_ESCALATION_AFTER: Final = dt.timedelta(hours=4)
"""How long an unacknowledged alert waits before it escalates.

A judgement, and a conservative one: long enough that an operator working the
alert is not paged twice for it, short enough that an alert raised at the start
of a session is escalated within it. It is a parameter on every function that
uses it, never a hidden constant.
"""

_LOGGER: Final = structlog.get_logger(__name__)

_SQLSTATE_LENGTH: Final = 5
"""Characters in a SQLSTATE code."""


class AlertSeverity(StrEnum):
    """How urgently an alert needs a human.

    Attributes:
        INFO: a fact worth recording; no action implied.
        WARNING: something needs looking at before the next rebalance.
        CRITICAL: something is already wrong — trading is halted, a monitor is
            blind, or an alert could not be delivered.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Ordinal severity, ascending, for comparisons and escalation."""
        return _SEVERITY_ORDER.index(self)


_SEVERITY_ORDER: Final = (AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL)


def alert_dedup_key(*, rule_id: str, condition: Mapping[str, JsonValue]) -> str:
    """Return the content-derived identity of one alert condition.

    SHA-256 over canonical JSON of the rule id and the condition mapping. The
    condition must describe *what was wrong and when*, and must contain nothing
    that changes between two evaluations of the same condition — no timestamp,
    no counter, no attempt number, no UUID. That exclusion is the mechanism: a
    monitoring job restarting mid-incident recomputes the identical key and the
    store absorbs it, so one condition produces one alert and one
    acknowledgement no matter how often the job runs.

    Args:
        rule_id: identifier of the rule raising the alert.
        condition: the condition's identity — for a halt, the date and cause;
            for feature drift, the date and feature. JSON-serialisable.

    Returns:
        A 64-character lowercase hex digest.

    Raises:
        backend.tracking.stamp.ConfigHashError: if ``condition`` cannot be
            canonicalised (a non-string key, a NaN, an unserialisable value).
    """
    return canonical_config_hash({"rule_id": rule_id, "condition": dict(condition)})


@dataclass(frozen=True, slots=True)
class Alert:
    """One alert condition, before it has been stored or delivered.

    Attributes:
        rule_id: which rule raised it.
        severity: how urgently it needs a human.
        subject: one line, for a channel that has room for one line.
        detail: the full message, including the numbers that produced it.
        dedup_key: the condition's content-derived identity
            (:func:`alert_dedup_key`).
        payload: the machine-readable finding, as JSON-safe values — typically
            the ``to_dict()`` of the decision or report that raised it.
        stamp: the I2 stamp of the monitoring run that raised it. Required: an
            alert is a decision, and every decision in this platform carries the
            four components it can be regenerated from.
        raised_at: when the condition was observed (UTC, timezone-aware).
    """

    rule_id: str
    severity: AlertSeverity
    subject: str
    detail: str
    dedup_key: str
    payload: dict[str, JsonValue]
    stamp: ReproducibilityStamp
    raised_at: dt.datetime

    def __post_init__(self) -> None:
        """Validate the alert.

        Raises:
            AlertError: if any text field is blank, the severity is not an
                :class:`AlertSeverity`, the stamp is not a
                :class:`~backend.tracking.stamp.ReproducibilityStamp`, or
                ``raised_at`` is timezone-naive. A naive instant in an incident
                record is a number whose meaning depends on the machine that
                wrote it.
        """
        for name, value in (
            ("rule_id", self.rule_id),
            ("subject", self.subject),
            ("detail", self.detail),
            ("dedup_key", self.dedup_key),
        ):
            supplied: object = value
            if not isinstance(supplied, str) or not supplied.strip():
                msg = f"{name} must be a non-empty string"
                raise AlertError(msg)
        supplied_severity: object = self.severity
        if not isinstance(supplied_severity, AlertSeverity):
            msg = f"severity must be an AlertSeverity, got {type(supplied_severity).__name__}"
            raise AlertError(msg)
        supplied_stamp: object = self.stamp
        if not isinstance(supplied_stamp, ReproducibilityStamp):
            msg = (
                f"stamp must be a ReproducibilityStamp, got {type(supplied_stamp).__name__}; "
                f"an alert that cannot say which run raised it cannot be investigated (I2)"
            )
            raise AlertError(msg)
        moment = self.raised_at
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            msg = f"raised_at={moment!r} is timezone-naive"
            raise AlertError(msg)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the alert as a JSON-safe mapping."""
        return {
            "rule_id": self.rule_id,
            "severity": str(self.severity),
            "subject": self.subject,
            "detail": self.detail,
            "dedup_key": self.dedup_key,
            "raised_at": self.raised_at.isoformat(),
            "git_reference": self.stamp.git_reference,
            "data_version": self.stamp.data_version,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class Acknowledgement:
    """A person's signature on an alert.

    Attributes:
        acknowledged_by: who. Free text, because the operator identity model is
            not this task's to invent; never blank.
        acknowledged_at: when (UTC).
        note: what they concluded. Never blank — an acknowledgement with no note
            is a click, and the point of the record is that someone looked.
    """

    acknowledged_by: str
    acknowledged_at: dt.datetime
    note: str


class DeliveryOutcome(StrEnum):
    """Whether one channel accepted one alert.

    Attributes:
        DELIVERED: the channel accepted it.
        FAILED: the channel refused or errored. Always carries a detail.
    """

    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeliveryAttempt:
    """One recorded attempt to hand one alert to one channel.

    Attributes:
        channel_id: the channel.
        attempt: 1-based attempt number for this ``(alert, channel)`` pair.
        outcome: delivered or failed.
        detail: why it failed, or how it was delivered.
        attempted_at: when (UTC).
    """

    channel_id: str
    attempt: int
    outcome: DeliveryOutcome
    detail: str
    attempted_at: dt.datetime


@dataclass(frozen=True, slots=True)
class StoredAlert:
    """A persisted alert and everything derived from other rows about it.

    ``delivered``, ``escalated`` and ``acknowledged`` are **derived**, never
    stored flags: a boolean column can drift from the rows it summarises, and
    the rows are the audit trail.

    Attributes:
        alert_id: database key.
        alert: the alert as raised.
        recorded_at: when the row was written (UTC).
        is_new: whether *this* record call wrote it, as opposed to absorbing a
            repeat of an existing condition.
        attempts: every delivery attempt, oldest first.
        acknowledgement: the signature, or ``None``.
    """

    alert_id: int
    alert: Alert
    recorded_at: dt.datetime
    is_new: bool
    attempts: tuple[DeliveryAttempt, ...] = ()
    acknowledgement: Acknowledgement | None = None

    @property
    def delivered(self) -> bool:
        """Whether any channel accepted this alert."""
        return any(attempt.outcome is DeliveryOutcome.DELIVERED for attempt in self.attempts)

    @property
    def escalated(self) -> bool:
        """Whether delivery was attempted and every attempt failed.

        An alert with no attempts at all is **not** escalated — it is
        undispatched, which is a different and more alarming condition
        (:func:`pending_escalations` reports both).
        """
        return bool(self.attempts) and not self.delivered

    @property
    def acknowledged(self) -> bool:
        """Whether someone has signed for this alert."""
        return self.acknowledgement is not None

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the stored alert and its derived state as a JSON-safe mapping."""
        return {
            "alert_id": self.alert_id,
            "recorded_at": self.recorded_at.isoformat(),
            "delivered": self.delivered,
            "escalated": self.escalated,
            "acknowledged": self.acknowledged,
            "acknowledged_by": (
                None if self.acknowledgement is None else self.acknowledgement.acknowledged_by
            ),
            "attempts": [
                {
                    "channel_id": attempt.channel_id,
                    "attempt": attempt.attempt,
                    "outcome": str(attempt.outcome),
                    "detail": attempt.detail,
                    "attempted_at": attempt.attempted_at.isoformat(),
                }
                for attempt in self.attempts
            ],
            "alert": self.alert.to_dict(),
        }


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    """Everything one monitoring run observed, as the rules see it.

    Attributes:
        as_of: the date the observations belong to. Part of every condition key,
            so the same condition on a later date is a new alert.
        stamp: the I2 stamp of the run.
        decision: the live-versus-expected verdict, or ``None`` when the run did
            not produce one — which is itself an alertable condition, see
            :class:`MonitorSilenceRule`.
        drift: the feature-drift report, or ``None``.
    """

    as_of: dt.date
    stamp: ReproducibilityStamp
    decision: HaltDecision | None = None
    drift: DriftReport | None = None


@runtime_checkable
class AlertRule(Protocol):
    """A rule that turns a monitoring snapshot into zero or more alerts.

    Implementations must be **pure and deterministic**: given the same snapshot
    they must produce the same alerts with the same dedup keys, because that is
    what makes "fires once per condition" true across restarts.
    """

    @property
    def rule_id(self) -> str:
        """Stable identifier, part of every dedup key this rule produces."""
        ...

    def evaluate(self, snapshot: MonitoringSnapshot, *, now: dt.datetime) -> tuple[Alert, ...]:
        """Return the alerts this rule raises for ``snapshot``.

        Args:
            snapshot: what the monitoring run observed.
            now: the instant to stamp raised alerts with (UTC).

        Returns:
            Zero or more alerts, each with a condition-derived dedup key.
        """
        ...


@dataclass(frozen=True, slots=True)
class HaltDecisionRule:
    """Alerts on the live-versus-expected verdict.

    Two conditions, deliberately distinct:

    * a **halt** — always ``CRITICAL``, whatever the cause. A halt because the
      comparison was unavailable is exactly as urgent as a halt because the
      strategy underperformed: in both cases trading has stopped and a human is
      required to restart it;
    * an **upper-band breach that did not halt** (policy ``HaltSide.LOWER``) —
      ``WARNING``. Beating a band cut from your own backtest is not good news,
      and the operator who chose the one-sided policy still has to be told.
    """

    rule_id: str = "live_vs_expected"

    def evaluate(self, snapshot: MonitoringSnapshot, *, now: dt.datetime) -> tuple[Alert, ...]:
        """Raise alerts for a halt, or for a non-halting upper breach.

        Args:
            snapshot: the monitoring run's observations.
            now: raised-at instant (UTC).

        Returns:
            Zero or one alert.
        """
        decision = snapshot.decision
        if decision is None:
            return ()
        if decision.should_halt:
            cause = "unknown" if decision.cause is None else str(decision.cause)
            return (
                Alert(
                    rule_id=self.rule_id,
                    severity=AlertSeverity.CRITICAL,
                    subject=f"Trading halted ({cause}) as of {snapshot.as_of.isoformat()}",
                    detail=decision.detail,
                    dedup_key=alert_dedup_key(
                        rule_id=self.rule_id,
                        condition={
                            "as_of": snapshot.as_of.isoformat(),
                            "action": str(decision.action),
                            "cause": cause,
                        },
                    ),
                    payload=decision.to_dict(),
                    stamp=snapshot.stamp,
                    raised_at=now,
                ),
            )
        comparison = decision.comparison
        if comparison is not None and comparison.above_band:
            return (
                Alert(
                    rule_id=self.rule_id,
                    severity=AlertSeverity.WARNING,
                    subject=(
                        f"Live performance above the expected band as of "
                        f"{snapshot.as_of.isoformat()} (not halting under this policy)"
                    ),
                    detail=decision.detail,
                    dedup_key=alert_dedup_key(
                        rule_id=self.rule_id,
                        condition={
                            "as_of": snapshot.as_of.isoformat(),
                            "action": str(decision.action),
                            "cause": "above_expected_band_not_halting",
                        },
                    ),
                    payload=decision.to_dict(),
                    stamp=snapshot.stamp,
                    raised_at=now,
                ),
            )
        return ()


@dataclass(frozen=True, slots=True)
class DriftReportRule:
    """Alerts on feature drift, and on a drift report that could not be completed.

    One alert per drifting feature rather than one per report: an operator
    acknowledges a *feature*, and a single rolled-up alert would be acknowledged
    once while thirty features drifted on.

    An **incomplete** report raises its own alert. D-031's reasoning applies
    directly — a feature the detector refused is not a stable feature, and a
    monitoring page silently blind to exactly the features whose data broke is
    the failure the refusal machinery exists to prevent.

    Attributes:
        minimum_band: the least severe band that raises an alert. Default
            :attr:`~backend.monitoring.drift.DriftBand.MAJOR`.
        rule_id: stable identifier.
    """

    minimum_band: DriftBand = DriftBand.MAJOR
    rule_id: str = "feature_drift"

    def evaluate(self, snapshot: MonitoringSnapshot, *, now: dt.datetime) -> tuple[Alert, ...]:
        """Raise one alert per drifting feature, plus one if the report is incomplete.

        Args:
            snapshot: the monitoring run's observations.
            now: raised-at instant (UTC).

        Returns:
            Zero or more alerts.
        """
        report = snapshot.drift
        if report is None:
            return ()
        threshold = _SEVERITY_BY_BAND[self.minimum_band]
        alerts: list[Alert] = []
        for finding in report.measured:
            severity = _SEVERITY_BY_BAND[finding.band]
            if severity.rank < threshold.rank:
                continue
            alerts.append(
                Alert(
                    rule_id=self.rule_id,
                    severity=severity,
                    subject=(
                        f"{finding.feature}: {finding.band} drift as of {report.as_of.isoformat()}"
                    ),
                    detail=(
                        f"PSI {finding.psi.value:.4f} against reference "
                        f"{finding.reference_id!r} (fingerprint "
                        f"{finding.reference_fingerprint[:12]}…); distribution band "
                        f"{finding.distribution_band}, availability band "
                        f"{finding.availability_band}. exceeds_sampling_noise="
                        f"{finding.exceeds_sampling_noise}, floor_driven="
                        f"{finding.floor_driven}. The ladder is convention, not a "
                        f"measurement: {finding.bands.basis}"
                    ),
                    dedup_key=alert_dedup_key(
                        rule_id=self.rule_id,
                        condition={
                            "as_of": report.as_of.isoformat(),
                            "feature": finding.feature,
                            "reference_fingerprint": finding.reference_fingerprint,
                            "band": str(finding.band),
                        },
                    ),
                    payload=finding.to_dict(),
                    stamp=snapshot.stamp,
                    raised_at=now,
                )
            )
        if report.unmeasurable:
            features = sorted(entry.feature for entry in report.unmeasurable)
            condition: dict[str, JsonValue] = {
                "as_of": report.as_of.isoformat(),
                "unmeasurable": list(features),
            }
            alerts.append(
                Alert(
                    rule_id=self.rule_id,
                    severity=AlertSeverity.WARNING,
                    subject=(
                        f"Drift report incomplete as of {report.as_of.isoformat()}: "
                        f"{len(features)} feature(s) could not be measured"
                    ),
                    detail=(
                        f"Unmeasurable: {', '.join(features)}. These features are NOT stable; "
                        f"no PSI exists for them on this date. A feature whose data broke is "
                        f"the one most likely to have moved, so an incomplete report is a "
                        f"finding rather than a gap (D-031)."
                    ),
                    dedup_key=alert_dedup_key(rule_id=self.rule_id, condition=condition),
                    payload={"unmeasurable": [entry.to_dict() for entry in report.unmeasurable]},
                    stamp=snapshot.stamp,
                    raised_at=now,
                )
            )
        return tuple(alerts)


_SEVERITY_BY_BAND: Final[dict[DriftBand, AlertSeverity]] = {
    DriftBand.STABLE: AlertSeverity.INFO,
    DriftBand.MODERATE: AlertSeverity.WARNING,
    DriftBand.MAJOR: AlertSeverity.CRITICAL,
}
"""Drift band to alert severity. A mapping rather than a comparison chain so a
new band cannot silently fall through to ``INFO``."""


@dataclass(frozen=True, slots=True)
class MonitorSilenceRule:
    """Alerts when a monitoring run observed nothing at all.

    The failure this catches is the one no other rule can: a scheduled job that
    ran, produced neither a verdict nor a drift report, and left the dashboard
    showing yesterday's green tiles. A monitor that observed nothing is
    indistinguishable, from the outside, from a monitor that found nothing
    wrong — so it says so, at ``CRITICAL``.

    Attributes:
        rule_id: stable identifier.
    """

    rule_id: str = "monitor_silence"

    def evaluate(self, snapshot: MonitoringSnapshot, *, now: dt.datetime) -> tuple[Alert, ...]:
        """Raise an alert when the snapshot carries no observations.

        Args:
            snapshot: the monitoring run's observations.
            now: raised-at instant (UTC).

        Returns:
            Zero or one alert.
        """
        if snapshot.decision is not None or snapshot.drift is not None:
            return ()
        return (
            Alert(
                rule_id=self.rule_id,
                severity=AlertSeverity.CRITICAL,
                subject=f"Monitoring produced no observations for {snapshot.as_of.isoformat()}",
                detail=(
                    "The monitoring run for this date carried neither a live-versus-expected "
                    "decision nor a drift report. Nothing was checked. This is reported at "
                    "CRITICAL because a silent monitor and a healthy system look identical "
                    "from the dashboard."
                ),
                dedup_key=alert_dedup_key(
                    rule_id=self.rule_id,
                    condition={"as_of": snapshot.as_of.isoformat()},
                ),
                payload={"as_of": snapshot.as_of.isoformat()},
                stamp=snapshot.stamp,
                raised_at=now,
            ),
        )


def default_rules() -> tuple[AlertRule, ...]:
    """Return the standard rule set, in evaluation order.

    Returns:
        A halt rule, a drift rule at the ``MAJOR`` band, and the silence rule.
    """
    return (HaltDecisionRule(), DriftReportRule(), MonitorSilenceRule())


def evaluate_rules(
    rules: Sequence[AlertRule], snapshot: MonitoringSnapshot, *, now: dt.datetime
) -> tuple[Alert, ...]:
    """Run every rule over one snapshot and return the alerts, de-duplicated.

    Two rules producing the same dedup key produce one alert — the first — which
    keeps "one condition, one alert" true even when rule sets overlap.

    Args:
        rules: the rules to run, in order.
        snapshot: what the monitoring run observed.
        now: raised-at instant for every alert (UTC).

    Returns:
        The alerts, in rule order, with duplicate dedup keys removed.
    """
    seen: set[str] = set()
    alerts: list[Alert] = []
    for rule in rules:
        for alert in rule.evaluate(snapshot, now=now):
            if alert.dedup_key in seen:
                continue
            seen.add(alert.dedup_key)
            alerts.append(alert)
    return tuple(alerts)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


@runtime_checkable
class AlertChannel(Protocol):
    """Somewhere an alert can be sent.

    Implementations **raise**
    :class:`~backend.monitoring.errors.AlertDeliveryError` on failure rather
    than returning a status, because a status is something a caller can ignore
    and the ignored branch is where alerts are lost.
    """

    @property
    def channel_id(self) -> str:
        """Stable identifier of this channel, recorded on every attempt."""
        ...

    async def deliver(self, alert: Alert) -> str:
        """Deliver ``alert``, or raise.

        Args:
            alert: the alert to deliver.

        Returns:
            A short human-readable delivery receipt (a message id, a log event
            name) recorded on the attempt row.

        Raises:
            AlertDeliveryError: if the alert could not be delivered.
        """
        ...


@dataclass(frozen=True, slots=True)
class StructlogChannel:
    """Delivers an alert as a structured log event.

    **This is a record, not a notification.** It never fails, which makes it a
    poor last line of defence: configured alone it would make every alert look
    delivered while nobody is paged. Use it alongside a channel that actually
    reaches a person, and treat its receipts as evidence the alert was written
    down rather than evidence it was read.

    Attributes:
        channel_id: stable identifier. Default ``"structlog"``.
    """

    channel_id: str = "structlog"

    async def deliver(self, alert: Alert) -> str:
        """Write the alert to the structured log.

        Args:
            alert: the alert to record.

        Returns:
            The log event name.
        """
        _LOGGER.warning(
            "monitoring.alert",
            rule_id=alert.rule_id,
            severity=str(alert.severity),
            subject=alert.subject,
            dedup_key=alert.dedup_key,
            git_reference=alert.stamp.git_reference,
            data_version=alert.stamp.data_version,
        )
        return "monitoring.alert"


class AlertStore(Protocol):
    """Persistence for alerts, their delivery attempts and their acknowledgements.

    Every method is ``async`` because the real implementation is
    :class:`PostgresAlertStore`. None of them commits — the caller owns the
    transaction, matching :mod:`backend.execution.store`.
    """

    async def record(self, alert: Alert, *, now: dt.datetime) -> StoredAlert:
        """Persist ``alert``, absorbing a repeat of an existing condition.

        Args:
            alert: the alert to store.
            now: the instant the row is written (UTC).

        Returns:
            The stored alert, with ``is_new`` false when an existing row for the
            same ``dedup_key`` was returned instead of a new one.
        """
        ...

    async def record_attempt(
        self,
        *,
        dedup_key: str,
        channel_id: str,
        outcome: DeliveryOutcome,
        detail: str,
        now: dt.datetime,
    ) -> DeliveryAttempt:
        """Record one delivery attempt.

        Args:
            dedup_key: the alert's condition key.
            channel_id: the channel attempted.
            outcome: delivered or failed.
            detail: receipt or failure reason. Never blank.
            now: when the attempt was made (UTC).

        Returns:
            The recorded attempt, carrying its 1-based attempt number.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
        """
        ...

    async def acknowledge(
        self, *, dedup_key: str, acknowledged_by: str, note: str, now: dt.datetime
    ) -> Acknowledgement:
        """Record a person's signature on an alert.

        Args:
            dedup_key: the alert's condition key.
            acknowledged_by: who is signing. Never blank.
            note: what they concluded. Never blank.
            now: when (UTC).

        Returns:
            The recorded acknowledgement.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
            AlreadyAcknowledgedError: if one is already recorded.
        """
        ...

    async def get(self, dedup_key: str) -> StoredAlert:
        """Return the stored alert for ``dedup_key`` with its derived state.

        Args:
            dedup_key: the alert's condition key.

        Returns:
            The stored alert.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
        """
        ...

    async def open_alerts(
        self, *, minimum_severity: AlertSeverity | None = None
    ) -> tuple[StoredAlert, ...]:
        """Return every unacknowledged alert, oldest first.

        Args:
            minimum_severity: when given, only alerts at least this severe.

        Returns:
            The unacknowledged alerts.
        """
        ...


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """The outcome of dispatching one alert.

    Attributes:
        stored: the persisted alert, including every attempt made here.
        delivered_by: the channel that accepted it, or ``None`` when none did
            (in which case :func:`dispatch` has already raised).
        failures: ``(channel_id, detail)`` for every channel that refused.
    """

    stored: StoredAlert
    delivered_by: str | None
    failures: tuple[tuple[str, str], ...] = field(default_factory=tuple)


async def dispatch(
    alert: Alert,
    *,
    store: AlertStore,
    channels: Sequence[AlertChannel],
    now: dt.datetime,
) -> DispatchResult:
    """Persist an alert, then deliver it, escalating if nobody accepts it.

    Order matters and is the point: **persist, then deliver.** If the process
    dies between the two, the alert exists with no delivery attempts, and
    :func:`pending_escalations` reports it as undispatched. The reverse order
    would lose the alert entirely on the same failure.

    Channels are tried in order and the first success stops the walk — the later
    channels are fallbacks, not copies. Every attempt, successful or not, is
    recorded.

    Args:
        alert: the alert to dispatch.
        store: where alerts, attempts and acknowledgements live.
        channels: delivery channels in priority order. An empty sequence is
            treated as total failure, because an alerting system with no
            configured channel is not an alerting system.
        now: the dispatch instant (UTC).

    Returns:
        A :class:`DispatchResult` when at least one channel accepted the alert.

    Raises:
        AlertUndeliverableError: when every channel refused, or none was
            configured. Raised **after** the alert and its failed attempts are
            persisted, so the exception means "nobody was told", never "the
            alert is gone".
    """
    await store.record(alert, now=now)
    failures: list[tuple[str, str]] = []
    attempts: list[DeliveryAttempt] = []
    for channel in channels:
        try:
            receipt = await channel.deliver(alert)
        except AlertDeliveryError as exc:
            failures.append((channel.channel_id, exc.detail))
            attempts.append(
                await store.record_attempt(
                    dedup_key=alert.dedup_key,
                    channel_id=channel.channel_id,
                    outcome=DeliveryOutcome.FAILED,
                    detail=exc.detail,
                    now=now,
                )
            )
            continue
        except Exception as exc:  # a channel that raised anything else still failed
            detail = f"{type(exc).__name__}: {exc}"
            failures.append((channel.channel_id, detail))
            attempts.append(
                await store.record_attempt(
                    dedup_key=alert.dedup_key,
                    channel_id=channel.channel_id,
                    outcome=DeliveryOutcome.FAILED,
                    detail=detail,
                    now=now,
                )
            )
            continue
        attempts.append(
            await store.record_attempt(
                dedup_key=alert.dedup_key,
                channel_id=channel.channel_id,
                outcome=DeliveryOutcome.DELIVERED,
                detail=receipt,
                now=now,
            )
        )
        return DispatchResult(
            stored=await store.get(alert.dedup_key),
            delivered_by=channel.channel_id,
            failures=tuple(failures),
        )
    raise AlertUndeliverableError(dedup_key=alert.dedup_key, failures=tuple(failures))


async def pending_escalations(
    store: AlertStore,
    *,
    now: dt.datetime,
    after: dt.timedelta = DEFAULT_ESCALATION_AFTER,
    minimum_severity: AlertSeverity | None = None,
) -> tuple[StoredAlert, ...]:
    """Return alerts that have gone unacknowledged too long, or reached nobody.

    Three conditions, all of which mean the alert is not doing its job:

    * **undelivered** — every attempt failed;
    * **undispatched** — no attempt was ever made (the process died between
      persisting and delivering);
    * **unacknowledged** — delivered ``after`` ago and still unsigned.

    Args:
        store: the alert store.
        now: the current instant (UTC).
        after: how long an unacknowledged alert may sit. Default
            :data:`DEFAULT_ESCALATION_AFTER`.
        minimum_severity: when given, only alerts at least this severe.

    Returns:
        The alerts needing escalation, oldest first.
    """
    deadline = now - after
    open_alerts = await store.open_alerts(minimum_severity=minimum_severity)
    return tuple(
        stored
        for stored in open_alerts
        if not stored.delivered or stored.alert.raised_at <= deadline
    )


async def escalate(
    stored: StoredAlert,
    *,
    store: AlertStore,
    channels: Sequence[AlertChannel],
    now: dt.datetime,
    round_number: int = 1,
) -> DispatchResult:
    """Re-raise an unacknowledged alert at a higher severity, once per round.

    The escalation is a *new* alert whose condition key is the original's plus
    the round number. That is what stops a poll every ten minutes from
    generating an escalation every ten minutes: round 1 fires once, and only a
    caller that decides to move to round 2 produces another.

    Args:
        stored: the alert that has not been acknowledged.
        store: the alert store.
        channels: delivery channels for the escalation, in priority order.
        now: the escalation instant (UTC).
        round_number: which escalation round this is, 1-based.

    Returns:
        The dispatch result for the escalation alert.

    Raises:
        AlertError: if ``round_number`` is below 1.
        AlertUndeliverableError: if the escalation itself reaches nobody.
    """
    if round_number < 1:
        msg = f"round_number must be at least 1; got {round_number}"
        raise AlertError(msg)
    original = stored.alert
    raised = _SEVERITY_ORDER[min(original.severity.rank + 1, len(_SEVERITY_ORDER) - 1)]
    reason = (
        "no channel accepted the original alert"
        if not stored.delivered
        else f"delivered but unacknowledged since {original.raised_at.isoformat()}"
    )
    escalation = Alert(
        rule_id=f"{original.rule_id}.escalation",
        severity=raised,
        subject=f"ESCALATION {round_number}: {original.subject}",
        detail=(
            f"Escalating alert {original.dedup_key} ({reason}). Original detail follows.\n"
            f"{original.detail}"
        ),
        dedup_key=alert_dedup_key(
            rule_id=f"{original.rule_id}.escalation",
            condition={"original": original.dedup_key, "round": round_number},
        ),
        payload={"original": original.to_dict(), "round": round_number, "reason": reason},
        stamp=original.stamp,
        raised_at=now,
    )
    return await dispatch(escalation, store=store, channels=channels, now=now)


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------


def _sqlstate(exc: DBAPIError) -> str | None:
    """Return the SQLSTATE a driver exception carries, if any.

    Args:
        exc: the wrapped database error.

    Returns:
        The five-character SQLSTATE, or ``None`` when the driver exposes none.
    """
    original: object = exc.orig
    for attribute in ("sqlstate", "pgcode"):
        code: object = getattr(original, attribute, None)
        if isinstance(code, str) and len(code) == _SQLSTATE_LENGTH:
            return code
    return None


def _is_unique_violation(exc: DBAPIError) -> bool:
    """Return whether ``exc`` is a uniqueness refusal rather than another constraint.

    Decided on the SQLSTATE (D-034): a CHECK violation is an ``IntegrityError``
    too, and treating one as "this condition is already recorded" would return
    an alert that was never written.

    Args:
        exc: the wrapped database error.

    Returns:
        ``True`` for ``23505``; when the driver exposes no SQLSTATE, falls back
        to the exception class, which is the best answer a silent driver allows.
    """
    sqlstate = _sqlstate(exc)
    if sqlstate is not None:
        return sqlstate == UNIQUE_VIOLATION_SQLSTATE
    return isinstance(exc, IntegrityError)


def _require_text(name: str, value: str) -> str:
    """Return ``value`` stripped, refusing a blank.

    Args:
        name: field name, for the message.
        value: the text to check.

    Returns:
        The stripped text.

    Raises:
        AlertError: if the value is blank.
    """
    supplied: object = value
    if not isinstance(supplied, str) or not supplied.strip():
        msg = f"{name} must be a non-empty string"
        raise AlertError(msg)
    return value.strip()


@dataclass(frozen=True, slots=True)
class PostgresAlertStore:
    """The real :class:`AlertStore`, over migration 0017's four tables.

    Does not commit: the caller owns the transaction, so a monitoring cycle's
    alerts and its halt record land together or not at all.

    Attributes:
        session: the ``AsyncSession`` to read and write through. These tables
            are not bitemporal (an alert is something *we* observed, not a fact
            about the market), so no as-of scoping applies — same reasoning as
            migrations 0005, 0007, 0009-0011 and 0014.
    """

    session: AsyncSession

    async def record(self, alert: Alert, *, now: dt.datetime) -> StoredAlert:
        """Persist an alert, absorbing a repeat of the same condition.

        The insert is attempted first and ``UNIQUE (dedup_key)`` decides. A
        look-then-insert has a window two monitoring workers can both pass
        through, and both would page the operator for one condition.

        Args:
            alert: the alert to store.
            now: when the row is written (UTC).

        Returns:
            The stored alert; ``is_new`` is false for an absorbed repeat.

        Raises:
            sqlalchemy.exc.DBAPIError: any refusal that is not a uniqueness
                violation, re-raised unchanged — this method interprets exactly
                one constraint and refuses to guess about the rest.
        """
        statement = (
            sa.insert(MonitoringAlertRow)
            .values(
                dedup_key=alert.dedup_key,
                rule_id=alert.rule_id,
                severity=str(alert.severity),
                subject=alert.subject,
                detail=alert.detail,
                payload=dict(alert.payload),
                raised_at=alert.raised_at,
                git_commit=alert.stamp.git_commit,
                git_dirty=alert.stamp.git_dirty,
                data_version=alert.stamp.data_version,
                config_hash=alert.stamp.config_hash,
                seed=alert.stamp.seed,
                recorded_at=now,
            )
            .returning(MonitoringAlertRow.alert_id, MonitoringAlertRow.recorded_at)
        )
        try:
            async with self.session.begin_nested():
                result = await self.session.execute(statement)
                alert_id, recorded_at = result.one()
        except DBAPIError as exc:
            if not _is_unique_violation(exc):
                raise
            return await self.get(alert.dedup_key)
        return StoredAlert(
            alert_id=int(alert_id),
            alert=alert,
            recorded_at=recorded_at,
            is_new=True,
        )

    async def record_attempt(
        self,
        *,
        dedup_key: str,
        channel_id: str,
        outcome: DeliveryOutcome,
        detail: str,
        now: dt.datetime,
    ) -> DeliveryAttempt:
        """Record one delivery attempt against a stored alert.

        Args:
            dedup_key: the alert's condition key.
            channel_id: the channel attempted.
            outcome: delivered or failed.
            detail: receipt or failure reason.
            now: when (UTC).

        Returns:
            The recorded attempt with its 1-based attempt number.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
            AlertError: if ``detail`` is blank — a failed delivery that does not
                say why cannot be acted on.
        """
        text = _require_text("detail", detail)
        alert_id = await self._alert_id(dedup_key)
        previous = await self.session.execute(
            sa.select(sa.func.count())
            .select_from(DeliveryRow)
            .where(DeliveryRow.alert_id == alert_id, DeliveryRow.channel_id == channel_id)
        )
        attempt = int(previous.scalar_one()) + 1
        await self.session.execute(
            sa.insert(DeliveryRow).values(
                alert_id=alert_id,
                channel_id=channel_id,
                attempt=attempt,
                outcome=str(outcome),
                detail=text,
                attempted_at=now,
            )
        )
        return DeliveryAttempt(
            channel_id=channel_id,
            attempt=attempt,
            outcome=outcome,
            detail=text,
            attempted_at=now,
        )

    async def acknowledge(
        self, *, dedup_key: str, acknowledged_by: str, note: str, now: dt.datetime
    ) -> Acknowledgement:
        """Record a person's signature on an alert, refusing a second one.

        Args:
            dedup_key: the alert's condition key.
            acknowledged_by: who is signing.
            note: what they concluded.
            now: when (UTC).

        Returns:
            The recorded acknowledgement.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
            AlreadyAcknowledgedError: if one is already recorded. The incumbent's
                name travels on the exception rather than being overwritten.
            AlertError: if either text field is blank.
        """
        who = _require_text("acknowledged_by", acknowledged_by)
        text = _require_text("note", note)
        alert_id = await self._alert_id(dedup_key)
        statement = sa.insert(AcknowledgementRow).values(
            alert_id=alert_id,
            acknowledged_by=who,
            note=text,
            acknowledged_at=now,
        )
        try:
            async with self.session.begin_nested():
                await self.session.execute(statement)
        except DBAPIError as exc:
            if not _is_unique_violation(exc):
                raise
            incumbent = await self.session.execute(
                sa.select(AcknowledgementRow.acknowledged_by).where(
                    AcknowledgementRow.alert_id == alert_id
                )
            )
            raise AlreadyAcknowledgedError(
                dedup_key=dedup_key, acknowledged_by=str(incumbent.scalar_one())
            ) from exc
        return Acknowledgement(acknowledged_by=who, acknowledged_at=now, note=text)

    async def get(self, dedup_key: str) -> StoredAlert:
        """Return the stored alert and everything derived about it.

        Args:
            dedup_key: the alert's condition key.

        Returns:
            The stored alert, with ``is_new`` false — this is a read.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
        """
        row = (
            await self.session.execute(
                sa.select(MonitoringAlertRow).where(MonitoringAlertRow.dedup_key == dedup_key)
            )
        ).scalar_one_or_none()
        if row is None:
            msg = f"no alert is stored under dedup key {dedup_key!r}"
            raise UnknownAlertError(msg)
        return await self._hydrate(row)

    async def open_alerts(
        self, *, minimum_severity: AlertSeverity | None = None
    ) -> tuple[StoredAlert, ...]:
        """Return every unacknowledged alert, oldest first.

        Args:
            minimum_severity: when given, only alerts at least this severe.

        Returns:
            The unacknowledged alerts.
        """
        acknowledged = sa.select(AcknowledgementRow.alert_id)
        statement = (
            sa.select(MonitoringAlertRow)
            .where(MonitoringAlertRow.alert_id.notin_(acknowledged))
            .order_by(MonitoringAlertRow.alert_id)
        )
        if minimum_severity is not None:
            allowed = [
                str(severity)
                for severity in _SEVERITY_ORDER
                if severity.rank >= minimum_severity.rank
            ]
            statement = statement.where(MonitoringAlertRow.severity.in_(allowed))
        rows = (await self.session.execute(statement)).scalars().all()
        return tuple([await self._hydrate(row) for row in rows])

    async def _alert_id(self, dedup_key: str) -> int:
        """Return the database key of the alert stored under ``dedup_key``.

        Args:
            dedup_key: the alert's condition key.

        Returns:
            The alert id.

        Raises:
            UnknownAlertError: if no alert carries ``dedup_key``.
        """
        found = (
            await self.session.execute(
                sa.select(MonitoringAlertRow.alert_id).where(
                    MonitoringAlertRow.dedup_key == dedup_key
                )
            )
        ).scalar_one_or_none()
        if found is None:
            msg = f"no alert is stored under dedup key {dedup_key!r}"
            raise UnknownAlertError(msg)
        return int(found)

    async def _hydrate(self, row: MonitoringAlertRow) -> StoredAlert:
        """Assemble a :class:`StoredAlert` from a row plus its attempts and signature.

        Args:
            row: the ORM alert row.

        Returns:
            The stored alert with derived delivery and acknowledgement state.
        """
        attempts = (
            (
                await self.session.execute(
                    sa.select(DeliveryRow)
                    .where(DeliveryRow.alert_id == row.alert_id)
                    .order_by(DeliveryRow.delivery_id)
                )
            )
            .scalars()
            .all()
        )
        signature = (
            await self.session.execute(
                sa.select(AcknowledgementRow).where(AcknowledgementRow.alert_id == row.alert_id)
            )
        ).scalar_one_or_none()
        payload: object = row.payload
        return StoredAlert(
            alert_id=row.alert_id,
            alert=Alert(
                rule_id=row.rule_id,
                severity=AlertSeverity(row.severity),
                subject=row.subject,
                detail=row.detail,
                dedup_key=row.dedup_key,
                payload=dict(payload) if isinstance(payload, dict) else {},
                stamp=ReproducibilityStamp(
                    git_commit=row.git_commit,
                    git_dirty=row.git_dirty,
                    data_version=row.data_version,
                    config_hash=row.config_hash,
                    seed=row.seed,
                ),
                raised_at=row.raised_at,
            ),
            recorded_at=row.recorded_at,
            is_new=False,
            attempts=tuple(
                DeliveryAttempt(
                    channel_id=attempt.channel_id,
                    attempt=attempt.attempt,
                    outcome=DeliveryOutcome(attempt.outcome),
                    detail=attempt.detail,
                    attempted_at=attempt.attempted_at,
                )
                for attempt in attempts
            ),
            acknowledgement=(
                None
                if signature is None
                else Acknowledgement(
                    acknowledged_by=signature.acknowledged_by,
                    acknowledged_at=signature.acknowledged_at,
                    note=signature.note,
                )
            ),
        )
