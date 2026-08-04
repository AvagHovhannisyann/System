"""Halt-history refusals that happen before any database call (P12.4).

The behaviour of :mod:`backend.monitoring.history` against a real schema —
derived halt state, one resume per halt, the acknowledgement gate, the
append-only triggers — is in ``test_alerts_store_db.py`` and needs a container.

What is testable without one is a narrower but real property: **the validations
happen before the I/O**. Every function here is handed a session that raises on
any use at all, so a test passing is proof the refusal came from the argument
check rather than from the database. That matters because these are the
refusals a caller hits in an operator flow, where an exception raised *after* a
partial write leaves the transaction poisoned and the caller with a worse
problem than the one they asked about.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pytest

from backend.monitoring.errors import HaltHistoryError
from backend.monitoring.expectation import HaltAction, HaltCause, decide
from backend.monitoring.history import (
    AUTOMATIC_ACTOR,
    HaltEventKind,
    halt_history,
    record_halt,
    record_resume,
)
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    LIVE_AS_OF,
    fixture_band,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import fixture_stamp

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class _NoDatabase:
    """A session stand-in that fails the test if it is touched at all."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - deliberately refuses everything
        message = f"the database was used ({name}) before the argument check refused the call"
        raise AssertionError(message)


def _session() -> AsyncSession:
    """Return the unusable session, typed for the functions under test."""
    return cast("AsyncSession", _NoDatabase())


async def test_recording_a_continue_decision_in_the_halt_history_is_refused() -> None:
    """A continue row in the halt history would read as an outage that never happened."""
    band = fixture_band()
    live = live_window(series_with_sharpe(band.median, n_periods=band.window_periods))
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.CONTINUE
    with pytest.raises(HaltHistoryError, match="an outage that never happened"):
        await record_halt(_session(), decision)


async def test_recording_something_that_is_not_a_decision_is_refused() -> None:
    with pytest.raises(HaltHistoryError, match="must be a HaltDecision"):
        await record_halt(_session(), cast("Any", {"action": "halt"}))


@pytest.mark.parametrize(
    ("actor", "reason"),
    [("", "looked at it"), ("   ", "looked at it"), ("operator", ""), ("operator", "  ")],
)
async def test_a_resume_without_an_actor_or_a_reason_is_refused(actor: str, reason: str) -> None:
    """The decision to halt is a machine's; the decision to resume is a person's."""
    with pytest.raises(HaltHistoryError, match="which human"):
        await record_resume(
            _session(),
            halt_event_id=1,
            actor=actor,
            reason=reason,
            stamp=fixture_stamp(),
            now=dt.datetime(2026, 8, 4, tzinfo=dt.UTC),
        )


@pytest.mark.parametrize("limit", [0, -1])
async def test_a_non_positive_history_limit_is_refused(limit: int) -> None:
    with pytest.raises(HaltHistoryError, match="limit"):
        await halt_history(_session(), limit=limit)


def test_every_halt_cause_is_a_distinct_operator_response() -> None:
    """The enum is the filter an operator uses; a single 'halted' would be useless."""
    causes = {str(cause) for cause in HaltCause}
    assert causes == {
        "below_expected_band",
        "above_expected_band",
        "comparison_unavailable",
        "stale_live_data",
        "cost_basis",
        "internal_error",
    }


def test_an_automatic_halt_names_the_job_rather_than_leaving_the_actor_unknown() -> None:
    assert AUTOMATIC_ACTOR.startswith("auto:")
    assert AUTOMATIC_ACTOR.strip() == AUTOMATIC_ACTOR


def test_the_history_records_exactly_two_kinds_of_event() -> None:
    assert {str(kind) for kind in HaltEventKind} == {"halt", "resume"}
    assert LIVE_AS_OF.isoformat() == "2026-08-01"
