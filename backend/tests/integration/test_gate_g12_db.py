"""Gate G12, the half only a database can answer: the halt actually lands.

``backend/tests/monitoring/test_gate_g12.py`` proves the two clauses as far as
in-process objects can carry them — an injected drift becomes a banded report
and a delivered alert; an injected deviation becomes a
``HALT``/``BELOW_EXPECTED_BAND`` decision and a delivered alert. Neither of
those is yet a halt. A halt is a **row**, and the gate's second clause is only
demonstrated when that row exists, when the kill-switch seam reads it, and when
:func:`~backend.monitoring.history.require_not_halted` refuses afterwards.

So this module asserts the part of G12 that lives on disk:

* the injected deviation's decision becomes a ``monitoring_halt_event`` row,
  checked by raw SQL rather than by the object :func:`record_halt` returned;
* :func:`~backend.monitoring.history.active_halt` derives the outage from the
  history and :func:`~backend.monitoring.history.require_not_halted` raises,
  which is the whole contract the execution-side kill switch is written against;
* the negative arm: an **in-band** decision writes nothing, and the seam keeps
  returning ``None`` — the halt table stays empty, which is what makes the
  positive arm evidence rather than a tautology;
* clause 1's landing: the drift alert raised by an injected shift is a
  ``monitoring_alert`` row an operator can query, and the uninjected cycle
  leaves the table empty.

**Not executed where this was written, and deliberately neither skipped nor
xfailed (I6, directive §9.3).** No Docker daemon is available in the authoring
environment, so ``backend/tests/integration/conftest.py`` cannot start the
TimescaleDB container and every test here errors on the environment rather than
reporting a pass it did not earn. CI has a daemon and runs it. Until that run is
green, **G12's second clause is verified as a decision and unverified as a
persisted halt** — a distinction that belongs in any statement about this gate.

Isolation follows the neighbouring monitoring module: every test runs inside one
transaction and rolls back, so nothing here needs the privileged TRUNCATE path.
The autouse fixture in ``conftest.py`` resets the schema between tests anyway.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import ingest_writer_session
from backend.monitoring.alerts import (
    AlertSeverity,
    MonitoringSnapshot,
    PostgresAlertStore,
)
from backend.monitoring.errors import (
    HaltHistoryError,
    SystemHaltedError,
    UnacknowledgedHaltError,
)
from backend.monitoring.expectation import HaltAction, HaltCause, HaltDecision, decide
from backend.monitoring.history import (
    AUTOMATIC_ACTOR,
    HaltEventKind,
    active_halt,
    halt_history,
    record_halt,
    record_resume,
    require_not_halted,
)
from backend.tests.monitoring.alert_fixtures import LATER
from backend.tests.monitoring.doubles import RecordingChannel
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import FIXTURE_DATA_VERSION, fixture_stamp
from backend.tests.monitoring.gate_g12_fixtures import (
    DEVIATION_SIGMAS_BELOW_EDGE,
    GATE_AS_OF,
    GATE_FEATURES,
    INJECTED_INDEX,
    INJECTED_LOCATION_SHIFT,
    deviated_live_window,
    gate_band,
    gate_drift_report,
    in_band_live_window,
    run_monitoring_cycle,
)

_OPERATOR = "operator@example.invalid"
_INJECTED_FEATURE = GATE_FEATURES[INJECTED_INDEX]


def _injected_deviation_decision() -> HaltDecision:
    """Build the halting decision from the gate's injected performance deviation.

    The live window's per-period Sharpe ratio is placed exactly
    ``DEVIATION_SIGMAS_BELOW_EDGE`` band standard deviations below the band's
    lower edge, so what reaches the database is a condition of a stated size.
    """
    band = gate_band()
    decision = decide(
        band=band,
        live=deviated_live_window(band, sigmas_below_edge=DEVIATION_SIGMAS_BELOW_EDGE),
        stamp=fixture_stamp(),
        now=DECISION_AT,
    )
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND
    return decision


def _in_band_decision() -> HaltDecision:
    """Build the control decision: the same band, a window at its median."""
    band = gate_band()
    decision = decide(
        band=band, live=in_band_live_window(band), stamp=fixture_stamp(), now=DECISION_AT
    )
    assert decision.action is HaltAction.CONTINUE
    return decision


async def _halt_rows(session: AsyncSession) -> list[sa.Row[tuple[str, str | None, str, str]]]:
    """Return every halt-history row, read by raw SQL rather than through the ORM.

    Deliberately not ``halt_history()``: the claim under test is that a row
    exists on disk, and asking the module that wrote it would be asking the
    accused. ``kind``, ``cause``, ``actor`` and ``data_version`` are enough to
    identify what was written and by which run.
    """
    result = await session.execute(
        sa.text(
            "SELECT kind, cause, actor, data_version FROM monitoring_halt_event "
            "ORDER BY halt_event_id"
        )
    )
    return list(result.all())


async def _alert_subjects(session: AsyncSession) -> list[tuple[str, str, str]]:
    """Return ``(rule_id, severity, subject)`` for every persisted alert, by raw SQL."""
    result = await session.execute(
        sa.text("SELECT rule_id, severity, subject FROM monitoring_alert ORDER BY alert_id")
    )
    return [(str(row[0]), str(row[1]), str(row[2])) for row in result.all()]


# ===========================================================================
# Clause 2 — the injected deviation lands as a halt, and the seam refuses
# ===========================================================================


async def test_clause_2_the_injected_deviation_writes_a_halt_row() -> None:
    """The row exists on disk, carries its cause, and names the run that wrote it.

    This is the sentence G12's second clause actually makes. Everything before
    it is a decision object; this is the halt.
    """
    async with ingest_writer_session() as session:
        assert await _halt_rows(session) == [], "the gate must start from an unhalted system"

        store = PostgresAlertStore(session=session)
        decision = _injected_deviation_decision()
        snapshot = MonitoringSnapshot(as_of=GATE_AS_OF, stamp=fixture_stamp(), decision=decision)
        dispatched = await run_monitoring_cycle(
            snapshot, store=store, channels=[RecordingChannel()], now=DECISION_AT
        )
        assert len(dispatched) == 1
        alert = dispatched[0].stored.alert

        event = await record_halt(session, decision, alert_dedup_key=alert.dedup_key)

        rows = await _halt_rows(session)
        assert len(rows) == 1, "exactly one halt row for one halting decision"
        kind, cause, actor, data_version = rows[0]
        assert kind == str(HaltEventKind.HALT)
        assert cause == str(HaltCause.BELOW_EXPECTED_BAND)
        assert actor == AUTOMATIC_ACTOR
        assert data_version == FIXTURE_DATA_VERSION, "I2: the row names the run (a fixture one)"

        # The decision payload travels onto the row, so the halt can be reviewed
        # without the objects that produced it.
        assert event.decision is not None
        assert event.decision["cause"] == str(HaltCause.BELOW_EXPECTED_BAND)
        assert event.alert_dedup_key == alert.dedup_key
        await session.rollback()


async def test_clause_2_the_halt_closes_the_kill_switch_seam() -> None:
    """``require_not_halted`` refuses while the injected deviation's halt is open.

    The execution side is written against exactly this: an exception, never a
    boolean, because a boolean is a value a caller can forget to check and the
    forgotten check leaves orders flowing during a halt.
    """
    async with ingest_writer_session() as session:
        # Before the halt: the seam permits trading. It returns None; the whole
        # contract is that it RAISES when halted, so a permitted system is the
        # call completing rather than a value anybody has to inspect.
        await require_not_halted(session)

        event = await record_halt(session, _injected_deviation_decision())

        current = await active_halt(session)
        assert current is not None
        assert current.halt_event_id == event.halt_event_id
        assert current.cause is HaltCause.BELOW_EXPECTED_BAND

        with pytest.raises(SystemHaltedError) as gated:
            await require_not_halted(session)
        assert gated.value.halt_event_id == event.halt_event_id
        assert gated.value.cause == str(HaltCause.BELOW_EXPECTED_BAND)
        await session.rollback()


async def test_clause_2_negative_an_in_band_decision_writes_nothing_and_gates_nothing() -> None:
    """The control arm at the storage layer: no row, no halt, no refusal.

    The only difference from the positive arm is where the live window's Sharpe
    ratio was placed. A halt log that filled up on healthy cycles would make
    every assertion in this module meaningless, so it is checked directly rather
    than assumed.
    """
    async with ingest_writer_session() as session:
        decision = _in_band_decision()

        with pytest.raises(HaltHistoryError, match="requires a halting decision"):
            await record_halt(session, decision)

        assert await _halt_rows(session) == []
        assert await active_halt(session) is None
        # Returns None; the seam's whole contract is that it RAISES when halted,
        # so a permitted system is the call completing rather than a value.
        await require_not_halted(session)
        await session.rollback()


async def test_clause_2_the_halt_is_cleared_only_by_an_acknowledged_operator_resume() -> None:
    """Both directions of the gate, in one round trip: halted, then permitted again.

    The resume is the negative direction of clause 2 at the storage layer — it
    is what proves the refusal above was a *state* rather than a permanent
    failure. It is also where acknowledgement becomes load-bearing: an alert
    nobody signed for keeps trading stopped.
    """
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        decision = _injected_deviation_decision()
        snapshot = MonitoringSnapshot(as_of=GATE_AS_OF, stamp=fixture_stamp(), decision=decision)
        dispatched = await run_monitoring_cycle(
            snapshot, store=store, channels=[RecordingChannel()], now=DECISION_AT
        )
        alert = dispatched[0].stored.alert
        event = await record_halt(session, decision, alert_dedup_key=alert.dedup_key)

        with pytest.raises(SystemHaltedError):
            await require_not_halted(session)

        # Nobody has read the alert, so trading does not restart.
        with pytest.raises(UnacknowledgedHaltError):
            await record_resume(
                session,
                halt_event_id=event.halt_event_id,
                actor=_OPERATOR,
                reason="looks fine to me",
                stamp=fixture_stamp(),
                now=LATER,
            )
        with pytest.raises(SystemHaltedError):
            await require_not_halted(session)

        await store.acknowledge(
            dedup_key=alert.dedup_key,
            acknowledged_by=_OPERATOR,
            note="synthetic G12 condition; reviewed the injected deviation and the band",
            now=LATER,
        )
        resume = await record_resume(
            session,
            halt_event_id=event.halt_event_id,
            actor=_OPERATOR,
            reason="G12 fixture halt cleared after review",
            stamp=fixture_stamp(),
            now=LATER,
        )
        assert resume.resolves_halt_event_id == event.halt_event_id

        assert await active_halt(session) is None
        # Returns None; the seam's whole contract is that it RAISES when halted,
        # so a permitted system is the call completing rather than a value.
        await require_not_halted(session)

        history = await halt_history(session)
        assert [entry.kind for entry in history] == [HaltEventKind.RESUME, HaltEventKind.HALT]
        assert history[1].cause is HaltCause.BELOW_EXPECTED_BAND
        await session.rollback()


async def test_clause_2_an_upside_deviation_also_lands_as_a_halt() -> None:
    """A deviation above the band stops trading too, with its own cause on the row.

    Beating a band cut from your own backtest is normally an accounting, scale
    or data error. The gate checks the halt lands in the direction nobody
    thinks to look at, because that is the one an operator would otherwise
    never see recorded.
    """
    band = gate_band()
    high = live_window(
        series_with_sharpe(
            band.upper + DEVIATION_SIGMAS_BELOW_EDGE * band.dispersion,
            n_periods=band.window_periods,
        )
    )
    decision = decide(band=band, live=high, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.cause is HaltCause.ABOVE_EXPECTED_BAND

    async with ingest_writer_session() as session:
        await record_halt(session, decision)
        rows = await _halt_rows(session)
        assert [row[1] for row in rows] == [str(HaltCause.ABOVE_EXPECTED_BAND)]

        with pytest.raises(SystemHaltedError):
            await require_not_halted(session)
        await session.rollback()


# ===========================================================================
# Clause 1 — the injected drift reaches the operator's alert table
# ===========================================================================


async def test_clause_1_the_injected_drift_is_persisted_as_a_queryable_alert() -> None:
    """The drift alert is a row an operator can find, naming the injected feature.

    Clause 1 says drift is *detected*. Detection that stays in a process is not
    detection; the monitoring page (§6.10) reads this table.
    """
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        snapshot = MonitoringSnapshot(
            as_of=GATE_AS_OF,
            stamp=fixture_stamp(),
            drift=gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT),
        )

        dispatched = await run_monitoring_cycle(
            snapshot, store=store, channels=[RecordingChannel()], now=DECISION_AT
        )

        assert len(dispatched) == 1
        persisted = await _alert_subjects(session)
        assert len(persisted) == 1
        rule_id, severity, subject = persisted[0]
        assert rule_id == "feature_drift"
        assert severity == str(AlertSeverity.CRITICAL)
        assert _INJECTED_FEATURE in subject

        # And it is unacknowledged, so it appears in the operator's open queue.
        open_alerts = await store.open_alerts(minimum_severity=AlertSeverity.CRITICAL)
        assert [item.alert.rule_id for item in open_alerts] == ["feature_drift"]
        await session.rollback()


async def test_clause_1_negative_an_uninjected_panel_persists_nothing() -> None:
    """The control arm at the storage layer: a healthy panel leaves no row behind."""
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        snapshot = MonitoringSnapshot(
            as_of=GATE_AS_OF, stamp=fixture_stamp(), drift=gate_drift_report()
        )

        dispatched = await run_monitoring_cycle(
            snapshot, store=store, channels=[RecordingChannel()], now=DECISION_AT
        )

        assert dispatched == ()
        assert await _alert_subjects(session) == []
        assert await store.open_alerts() == ()
        await session.rollback()


async def test_both_clauses_land_together_in_one_cycle() -> None:
    """One monitoring run, both injections: two alert rows and one halt row.

    The composite the gate's two clauses describe. Two alerts rather than one
    rolled-up notice, because an operator acknowledges a condition and a merged
    alert would be signed for once while the other condition ran on.
    """
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        decision = _injected_deviation_decision()
        snapshot = MonitoringSnapshot(
            as_of=GATE_AS_OF,
            stamp=fixture_stamp(),
            decision=decision,
            drift=gate_drift_report(location_shift=INJECTED_LOCATION_SHIFT),
        )

        dispatched = await run_monitoring_cycle(
            snapshot, store=store, channels=[RecordingChannel()], now=DECISION_AT
        )
        halt_alert = next(
            result for result in dispatched if result.stored.alert.rule_id == "live_vs_expected"
        )
        await record_halt(session, decision, alert_dedup_key=halt_alert.stored.alert.dedup_key)

        persisted = await _alert_subjects(session)
        assert sorted(rule_id for rule_id, _, _ in persisted) == [
            "feature_drift",
            "live_vs_expected",
        ]
        assert len(await _halt_rows(session)) == 1
        with pytest.raises(SystemHaltedError):
            await require_not_halted(session)
        await session.rollback()
