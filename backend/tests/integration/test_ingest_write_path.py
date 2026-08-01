"""P3.1 integration: knowledge-time hygiene and open-interval supersession.

Two D-011 controls, exercised end to end against real TimescaleDB:

1. a row whose ``knowledge_time`` is in the future never reaches the store;
2. an open-ended fact is closed by a later-knowledge correction and not by an
   UPDATE — and the resulting intervals do not overlap, where the naive
   alternative demonstrably does.

The second half deliberately includes the *failure* it prevents: a test that
writes a second open-ended row and asserts the overlap is visible. Without it,
the passing supersession test would prove only that the helper produces some
rows, not that it fixes anything.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from backend.db import BitemporalBypassError, as_of, ingest_writer_session
from backend.db.bitemporal import INFINITY
from backend.db.models import SecurityMaster
from backend.ingest.errors import FutureKnowledgeTimeError
from backend.ingest.supersession import supersede_open_interval
from backend.tests.integration.factories import create_security, insert_rows

_LISTED = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
_RENAMED = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)
_KNOWN_FIRST = dt.datetime(2020, 1, 2, tzinfo=dt.UTC)
_KNOWN_LATER = dt.datetime(2024, 6, 2, tzinfo=dt.UTC)
_EVENT_TIME_AFTER_RENAME = dt.datetime(2024, 7, 1, tzinfo=dt.UTC)


def _open_master(
    security_id: int, ticker: str, *, valid_from: dt.datetime, knowledge_time: dt.datetime
) -> SecurityMaster:
    """Build an open-ended identity version ``[valid_from, infinity)``."""
    return SecurityMaster(
        security_id=security_id,
        ticker=ticker,
        name=f"{ticker} Corp.",
        exchange="XNAS",
        valid_from=valid_from,
        valid_to=INFINITY,
        knowledge_time=knowledge_time,
    )


async def _masters_at(as_of_ts: dt.datetime) -> list[SecurityMaster]:
    async with as_of(as_of_ts) as session:
        result = await session.scalars(select(SecurityMaster).order_by(SecurityMaster.valid_from))
        return list(result.all())


def _covering(masters: list[SecurityMaster], event_time: dt.datetime) -> list[SecurityMaster]:
    """Versions whose half-open event interval contains ``event_time``."""
    return [m for m in masters if m.valid_from <= event_time < m.valid_to]


# --- knowledge-time hygiene at the real write path --------------------------


async def test_future_knowledge_time_never_reaches_the_store() -> None:
    security_id = await create_security()
    future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    with pytest.raises(FutureKnowledgeTimeError, match="is in the future"):
        await insert_rows(
            _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=future)
        )
    assert await _masters_at(dt.datetime.now(dt.UTC)) == []


async def test_a_future_row_does_not_poison_the_rest_of_its_batch() -> None:
    """The whole flush is refused: no partial batch, no orphaned good rows."""
    security_id = await create_security()
    future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    with pytest.raises(FutureKnowledgeTimeError):
        await insert_rows(
            _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST),
            _open_master(security_id, "XYZ", valid_from=_RENAMED, knowledge_time=future),
        )
    assert await _masters_at(dt.datetime.now(dt.UTC)) == []


async def test_a_past_knowledge_time_still_writes_normally() -> None:
    """The guard is narrow — backfill-style historical knowledge is unaffected."""
    security_id = await create_security()
    await insert_rows(
        _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    )
    assert [m.ticker for m in await _masters_at(_KNOWN_FIRST)] == ["ABC"]


# --- the failure the supersession helper exists to prevent ------------------


async def test_naive_second_open_row_produces_overlapping_intervals() -> None:
    """The audit finding, demonstrated: two open intervals cover one event time.

    Writing the successor without closing its predecessor leaves
    ``[2020-01-01, inf)`` and ``[2024-06-01, inf)`` both visible. Every row is
    individually valid and nothing raises, which is exactly why this needs a
    helper rather than care: a point-in-time query at 2024-07-01 gets **two**
    identities for one security.
    """
    security_id = await create_security()
    await insert_rows(
        _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST),
        _open_master(security_id, "XYZ", valid_from=_RENAMED, knowledge_time=_KNOWN_LATER),
    )
    visible = await _masters_at(dt.datetime(2025, 1, 1, tzinfo=dt.UTC))
    covering = _covering(visible, _EVENT_TIME_AFTER_RENAME)
    assert len(covering) == 2
    assert {m.ticker for m in covering} == {"ABC", "XYZ"}
    assert all(m.valid_to == INFINITY for m in covering)


# --- the supersession contract ----------------------------------------------


async def test_supersession_leaves_exactly_one_version_per_event_time() -> None:
    security_id = await create_security()
    original = _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    await insert_rows(original)
    correction, successor = supersede_open_interval(
        original,
        boundary=_RENAMED,
        knowledge_time=_KNOWN_LATER,
        changes={"ticker": "XYZ", "name": "XYZ Corp."},
    )
    await insert_rows(correction, successor)

    visible = await _masters_at(dt.datetime(2025, 1, 1, tzinfo=dt.UTC))
    assert [(m.ticker, m.valid_from, m.valid_to) for m in visible] == [
        ("ABC", _LISTED, _RENAMED),
        ("XYZ", _RENAMED, INFINITY),
    ]
    assert len(_covering(visible, _EVENT_TIME_AFTER_RENAME)) == 1
    assert _covering(visible, _EVENT_TIME_AFTER_RENAME)[0].ticker == "XYZ"
    assert len(_covering(visible, dt.datetime(2022, 1, 1, tzinfo=dt.UTC))) == 1


async def test_an_as_of_before_the_correction_still_sees_the_open_interval() -> None:
    """Immutability of the past: we honestly believed it was open-ended then."""
    security_id = await create_security()
    original = _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    await insert_rows(original)
    correction, successor = supersede_open_interval(
        original, boundary=_RENAMED, knowledge_time=_KNOWN_LATER, changes={"ticker": "XYZ"}
    )
    await insert_rows(correction, successor)

    before = await _masters_at(_KNOWN_LATER - dt.timedelta(microseconds=1))
    assert [(m.ticker, m.valid_to) for m in before] == [("ABC", INFINITY)]

    at_boundary = await _masters_at(_KNOWN_LATER)
    assert [(m.ticker, m.valid_to) for m in at_boundary] == [
        ("ABC", _RENAMED),
        ("XYZ", INFINITY),
    ]


async def test_the_correction_supersedes_rather_than_duplicating() -> None:
    """Same (key, valid_from), later knowledge: one visible version, not two."""
    security_id = await create_security()
    original = _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    await insert_rows(original)
    correction, successor = supersede_open_interval(
        original, boundary=_RENAMED, knowledge_time=_KNOWN_LATER, changes={"ticker": "XYZ"}
    )
    await insert_rows(correction, successor)

    visible = await _masters_at(dt.datetime(2025, 1, 1, tzinfo=dt.UTC))
    at_original_valid_from = [m for m in visible if m.valid_from == _LISTED]
    assert len(at_original_valid_from) == 1
    assert at_original_valid_from[0].valid_to == _RENAMED


async def test_both_physical_versions_remain_on_disk() -> None:
    """Append-only: the correction adds a row, it does not overwrite one.

    Read through two as-ofs rather than a raw scan — the raw scan is exactly
    what the D-011 guard forbids — and the two answers differing proves both
    versions are physically present.
    """
    security_id = await create_security()
    original = _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    await insert_rows(original)
    correction, successor = supersede_open_interval(
        original, boundary=_RENAMED, knowledge_time=_KNOWN_LATER, changes={"ticker": "XYZ"}
    )
    await insert_rows(correction, successor)

    old_view = await _masters_at(_KNOWN_LATER - dt.timedelta(days=1))
    new_view = await _masters_at(_KNOWN_LATER)
    assert [m.valid_to for m in old_view] == [INFINITY]
    assert [m.valid_to for m in new_view] == [_RENAMED, INFINITY]


async def test_supersession_writes_are_append_only_inserts() -> None:
    """A second supersession of the successor chains correctly."""
    security_id = await create_security()
    original = _open_master(security_id, "ABC", valid_from=_LISTED, knowledge_time=_KNOWN_FIRST)
    await insert_rows(original)
    first_correction, successor = supersede_open_interval(
        original, boundary=_RENAMED, knowledge_time=_KNOWN_LATER, changes={"ticker": "XYZ"}
    )
    await insert_rows(first_correction, successor)

    second_boundary = dt.datetime(2025, 3, 1, tzinfo=dt.UTC)
    second_knowledge = dt.datetime(2025, 3, 2, tzinfo=dt.UTC)
    second_correction, third = supersede_open_interval(
        successor,
        boundary=second_boundary,
        knowledge_time=second_knowledge,
        changes={"ticker": "PQR"},
    )
    await insert_rows(second_correction, third)

    visible = await _masters_at(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    assert [(m.ticker, m.valid_from, m.valid_to) for m in visible] == [
        ("ABC", _LISTED, _RENAMED),
        ("XYZ", _RENAMED, second_boundary),
        ("PQR", second_boundary, INFINITY),
    ]


async def test_writer_session_still_refuses_to_read_fact_tables() -> None:
    """The ingest write path gained no read capability (D-011 layer 2 unchanged)."""
    async with ingest_writer_session() as session:
        with pytest.raises(BitemporalBypassError):
            await session.execute(select(SecurityMaster))
