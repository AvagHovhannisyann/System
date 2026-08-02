"""An in-memory :class:`~backend.monitoring.alerts.AlertStore`, and what it does not model.

D-034 is the reason this file has a docstring rather than just code. A double
that stands in for the *wrong layer* makes a broken store indistinguishable from
a correct one, and the specific way it did that in P11.2 was by modelling the
primary key when a ``BEFORE INSERT`` trigger was what actually refused the row.

So, explicitly, what this double **does** model:

* ``UNIQUE (dedup_key)`` on ``monitoring_alert`` — a repeat condition is absorbed
  and returned with ``is_new`` false;
* ``UNIQUE (alert_id)`` on ``monitoring_alert_acknowledgement`` — a second
  acknowledgement raises
  :class:`~backend.monitoring.errors.AlreadyAcknowledgedError` carrying the
  incumbent's name;
* the per-``(alert, channel)`` attempt counter that
  ``UNIQUE (alert_id, channel_id, attempt)`` makes monotonic;
* the ``detail <> ''``, ``acknowledged_by <> ''`` and ``note <> ''`` CHECKs,
  as the same :class:`~backend.monitoring.errors.AlertError` the real store
  raises before it reaches SQL.

And what it **does not**, so no test here may be read as evidence about them:

* the append-only triggers — an in-memory dict cannot refuse an UPDATE it has no
  concept of;
* the remaining CHECK constraints (severity, digest shape, git/config hash
  shapes), which bind a writer that skips Python entirely;
* concurrency. Every refusal here happens in one thread with no window in it,
  which is exactly the property a real store has to earn from the database.

Those three are asserted against Postgres in
``backend/tests/integration/test_monitoring_alerts_db.py``, and nowhere else.
The contract suite in ``store_contract.py`` runs against both.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.monitoring.alerts import (
    Acknowledgement,
    Alert,
    AlertSeverity,
    DeliveryAttempt,
    DeliveryOutcome,
    StoredAlert,
)
from backend.monitoring.errors import (
    AlertDeliveryError,
    AlertError,
    AlreadyAcknowledgedError,
    UnknownAlertError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _require_text(name: str, value: str) -> str:
    """Mirror the store's blank-text refusal.

    Args:
        name: field name for the message.
        value: text to check.

    Returns:
        The stripped text.

    Raises:
        AlertError: if the value is blank.
    """
    if not value.strip():
        msg = f"{name} must be a non-empty string"
        raise AlertError(msg)
    return value.strip()


@dataclass
class InMemoryAlertStore:
    """A dict-backed alert store with the same refusals as the Postgres one.

    Attributes:
        alerts: alerts by dedup key, in insertion order.
        attempts: delivery attempts by dedup key, in attempt order.
        acknowledgements: signatures by dedup key.
        recorded_at: when each alert row was written.
    """

    alerts: dict[str, Alert] = field(default_factory=dict)
    attempts: dict[str, list[DeliveryAttempt]] = field(default_factory=dict)
    acknowledgements: dict[str, Acknowledgement] = field(default_factory=dict)
    recorded_at: dict[str, dt.datetime] = field(default_factory=dict)

    async def record(self, alert: Alert, *, now: dt.datetime) -> StoredAlert:
        """Store an alert, absorbing a repeat of the same condition.

        Args:
            alert: the alert to store.
            now: when the row is written (UTC).

        Returns:
            The stored alert; ``is_new`` false for an absorbed repeat.
        """
        if alert.dedup_key in self.alerts:
            return await self.get(alert.dedup_key)
        self.alerts[alert.dedup_key] = alert
        self.recorded_at[alert.dedup_key] = now
        self.attempts.setdefault(alert.dedup_key, [])
        return StoredAlert(
            alert_id=len(self.alerts),
            alert=alert,
            recorded_at=now,
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
        """Record one delivery attempt.

        Args:
            dedup_key: the alert's condition key.
            channel_id: the channel attempted.
            outcome: delivered or failed.
            detail: receipt or failure reason.
            now: when (UTC).

        Returns:
            The recorded attempt with its 1-based number.

        Raises:
            UnknownAlertError: if the alert is not stored.
            AlertError: if ``detail`` is blank.
        """
        text = _require_text("detail", detail)
        self._require_known(dedup_key)
        made = self.attempts.setdefault(dedup_key, [])
        attempt = DeliveryAttempt(
            channel_id=channel_id,
            attempt=sum(1 for item in made if item.channel_id == channel_id) + 1,
            outcome=outcome,
            detail=text,
            attempted_at=now,
        )
        made.append(attempt)
        return attempt

    async def acknowledge(
        self, *, dedup_key: str, acknowledged_by: str, note: str, now: dt.datetime
    ) -> Acknowledgement:
        """Record a signature, refusing a second one.

        Args:
            dedup_key: the alert's condition key.
            acknowledged_by: who is signing.
            note: what they concluded.
            now: when (UTC).

        Returns:
            The recorded acknowledgement.

        Raises:
            UnknownAlertError: if the alert is not stored.
            AlreadyAcknowledgedError: if one is already recorded.
            AlertError: if either text is blank.
        """
        who = _require_text("acknowledged_by", acknowledged_by)
        text = _require_text("note", note)
        self._require_known(dedup_key)
        incumbent = self.acknowledgements.get(dedup_key)
        if incumbent is not None:
            raise AlreadyAcknowledgedError(
                dedup_key=dedup_key, acknowledged_by=incumbent.acknowledged_by
            )
        signature = Acknowledgement(acknowledged_by=who, acknowledged_at=now, note=text)
        self.acknowledgements[dedup_key] = signature
        return signature

    async def get(self, dedup_key: str) -> StoredAlert:
        """Return the stored alert and its derived state.

        Args:
            dedup_key: the alert's condition key.

        Returns:
            The stored alert.

        Raises:
            UnknownAlertError: if the alert is not stored.
        """
        self._require_known(dedup_key)
        keys = list(self.alerts)
        return StoredAlert(
            alert_id=keys.index(dedup_key) + 1,
            alert=self.alerts[dedup_key],
            recorded_at=self.recorded_at[dedup_key],
            is_new=False,
            attempts=tuple(self.attempts.get(dedup_key, ())),
            acknowledgement=self.acknowledgements.get(dedup_key),
        )

    async def open_alerts(
        self, *, minimum_severity: AlertSeverity | None = None
    ) -> tuple[StoredAlert, ...]:
        """Return every unacknowledged alert, oldest first.

        Args:
            minimum_severity: when given, only alerts at least this severe.

        Returns:
            The unacknowledged alerts.
        """
        found: list[StoredAlert] = []
        for key, alert in self.alerts.items():
            if key in self.acknowledgements:
                continue
            if minimum_severity is not None and alert.severity.rank < minimum_severity.rank:
                continue
            found.append(await self.get(key))
        return tuple(found)

    def _require_known(self, dedup_key: str) -> None:
        """Raise if ``dedup_key`` is not stored.

        Args:
            dedup_key: the key to check.

        Raises:
            UnknownAlertError: if it is not stored.
        """
        if dedup_key not in self.alerts:
            msg = f"no alert is stored under dedup key {dedup_key!r}"
            raise UnknownAlertError(msg)


@dataclass
class RecordingChannel:
    """A channel that accepts everything and remembers what it was given.

    Attributes:
        channel_id: identifier.
        delivered: every alert handed to it, in order.
    """

    channel_id: str = "recording"
    delivered: list[Alert] = field(default_factory=list)

    async def deliver(self, alert: Alert) -> str:
        """Accept the alert.

        Args:
            alert: the alert delivered.

        Returns:
            A receipt naming the position in the delivery order.
        """
        self.delivered.append(alert)
        return f"recorded#{len(self.delivered)}"


@dataclass
class FailingChannel:
    """A channel that always refuses, the way a real one does — by raising.

    Attributes:
        channel_id: identifier.
        detail: the refusal reason.
        calls: how many times it was asked (count).
    """

    channel_id: str = "failing"
    detail: str = "simulated channel outage"
    calls: int = 0

    async def deliver(self, alert: Alert) -> str:  # noqa: ARG002 - the signature is the contract
        """Refuse the alert.

        Args:
            alert: the alert that will not be delivered.

        Raises:
            AlertDeliveryError: always.
        """
        self.calls += 1
        raise AlertDeliveryError(channel_id=self.channel_id, detail=self.detail)


@dataclass
class ExplodingChannel:
    """A channel that raises something other than ``AlertDeliveryError``.

    A third-party client raising ``TimeoutError`` or ``KeyError`` is the ordinary
    case, not the exotic one, and a dispatcher that only catches its own
    exception type would let that escape *after* the alert was persisted and
    *before* the remaining channels were tried — losing the notification while
    the audit trail says an attempt was in progress.

    Attributes:
        channel_id: identifier.
    """

    channel_id: str = "exploding"

    async def deliver(self, alert: Alert) -> str:  # noqa: ARG002 - the signature is the contract
        """Raise a non-alerting exception.

        Args:
            alert: the alert that will not be delivered.

        Raises:
            TimeoutError: always.
        """
        msg = "third-party client timed out"
        raise TimeoutError(msg)


def channel_ids(channels: Sequence[object]) -> tuple[str, ...]:
    """Return the ``channel_id`` of each channel, for assertion messages.

    Args:
        channels: the channels.

    Returns:
        Their identifiers in order.
    """
    return tuple(str(getattr(channel, "channel_id", "?")) for channel in channels)
