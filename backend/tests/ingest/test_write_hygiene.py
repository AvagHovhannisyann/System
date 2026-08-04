"""Future ``knowledge_time`` is refused in the write path (P3.1, D-011 control).

Two levels, both here because neither needs a database:

- the pure validator, whose boundary (``knowledge_time == now`` is fine, one
  microsecond later is not) is the whole specification;
- the class-level ``before_flush`` listener, which fires on a bindless
  ``Session`` — the flush is refused before any bind is resolved, so the proof
  that nothing reaches the database is that no database is involved at all.

The end-to-end version, against real TimescaleDB, lives in
``backend/tests/integration/test_ingest_write_path.py``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from backend.db.models import PriceBar, SecurityMaster
from backend.ingest import FUTURE_KNOWLEDGE_TIME_TOLERANCE
from backend.ingest.errors import FutureKnowledgeTimeError
from backend.ingest.write import validate_knowledge_time

_NOW = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)


def _bar(knowledge_time: dt.datetime) -> PriceBar:
    """Build one transient PriceBar; values are irrelevant to the check."""
    day = dt.datetime(2024, 1, 4, tzinfo=dt.UTC)
    price = Decimal("10")
    return PriceBar(
        security_id=1,
        valid_from=day,
        valid_to=day + dt.timedelta(days=1),
        knowledge_time=knowledge_time,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal("1"),
        volume_shares=1000,
    )


# --- the validator ----------------------------------------------------------


def test_tolerance_is_zero_by_decision() -> None:
    """Documented in backend/ingest/write.py; asserted so a change is deliberate."""
    assert dt.timedelta(0) == FUTURE_KNOWLEDGE_TIME_TOLERANCE


def test_past_knowledge_time_is_accepted() -> None:
    past = _NOW - dt.timedelta(days=3650)
    assert validate_knowledge_time(past, context="probe", now=_NOW) is past


def test_knowledge_time_equal_to_now_is_accepted() -> None:
    """The boundary is inclusive: a fact knowable *now* is writable now."""
    assert validate_knowledge_time(_NOW, context="probe", now=_NOW) is _NOW


def test_knowledge_time_one_microsecond_ahead_is_refused() -> None:
    """One microsecond past the boundary is still a claim about the future."""
    with pytest.raises(FutureKnowledgeTimeError, match="is in the future"):
        validate_knowledge_time(_NOW + dt.timedelta(microseconds=1), context="probe", now=_NOW)


def test_naive_knowledge_time_is_refused_with_a_specific_error() -> None:
    naive = dt.datetime(2024, 1, 4)  # noqa: DTZ001 — the point of the test
    with pytest.raises(TypeError, match="timezone-aware"):
        validate_knowledge_time(naive, context="probe", now=_NOW)


# --- the class-level flush listener ----------------------------------------


def test_flush_of_a_future_knowledge_time_row_is_refused_before_any_io() -> None:
    """A bindless Session proves the refusal happens before the database is touched."""
    future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=6)
    with Session() as session:
        session.add(_bar(future))
        with pytest.raises(FutureKnowledgeTimeError, match=r"price_bar|PriceBar"):
            session.flush()


def test_flush_error_names_the_offending_row() -> None:
    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    with Session() as session:
        session.add(
            SecurityMaster(
                security_id=7,
                ticker="ABC",
                name="ABC Corp.",
                exchange="XNAS",
                valid_from=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
                knowledge_time=future,
            )
        )
        with pytest.raises(FutureKnowledgeTimeError) as raised:
            session.flush()
    message = str(raised.value)
    assert "SecurityMaster" in message
    assert "security_id=7" in message


def test_flush_of_past_knowledge_time_rows_is_not_blocked_by_the_listener() -> None:
    """The guard is narrow: only future values are refused, and nothing else changes.

    A bindless session cannot execute SQL, so reaching a bind-resolution error
    is exactly the evidence wanted — the listener let the flush proceed.
    """
    past = dt.datetime(2016, 1, 4, tzinfo=dt.UTC)
    with Session() as session:
        session.add(_bar(past))
        with pytest.raises(Exception) as raised:  # noqa: PT011 — see assertion below
            session.flush()
    assert not isinstance(raised.value, FutureKnowledgeTimeError)


def test_listener_ignores_rows_without_a_knowledge_time() -> None:
    """A missing knowledge_time is the database's NOT NULL to report, not ours."""
    day = dt.datetime(2024, 1, 4, tzinfo=dt.UTC)
    with Session() as session:
        session.add(
            SecurityMaster(
                security_id=1,
                ticker="ABC",
                name="ABC Corp.",
                exchange="XNAS",
                valid_from=day,
            )
        )
        with pytest.raises(Exception) as raised:  # noqa: PT011 — see assertion below
            session.flush()
    assert not isinstance(raised.value, FutureKnowledgeTimeError)
