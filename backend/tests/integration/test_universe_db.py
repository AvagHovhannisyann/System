"""Phase 4 against a real TimescaleDB: the as-of reads, the schema, gate G4.

Everything here needs Postgres. The unit suite in ``backend/tests/universe``
covers the arithmetic against hand-built inputs; what it cannot cover is the part
that only exists inside the database:

- the **as-of read** itself. ``read_listings`` and ``read_price_window`` write no
  ``knowledge_time`` predicate of their own — the as-of layer rewrites their
  SELECTs — so whether a fact learned after the rebalance date can leak into a
  screen is a property of the rewrite plus these statements together, and only a
  real query answers it.
- **gate G4 end to end.** A name delisted after a past rebalance date must be in
  that date's universe. Here the delisting is a real later-knowledge row in
  ``security_master``, read back at an as-of years afterwards — the exact
  situation in which survivorship bias enters, and one no in-memory fixture can
  reproduce.
- **migration 0011's constraints**: the append-only triggers, the uniqueness of
  a snapshot's ``(rebalance_date, criteria_hash, as_of)`` identity, and the CHECK
  tying ``included`` to an empty ``failed_filters``.

**Not executed in this environment.** The Docker daemon is unavailable where this
was written, so the Testcontainers fixture cannot start and every test in this
module errors on the environment rather than on its assertions. They are
deliberately **not** skipped or xfailed (I6): a skipped test reports success it
did not earn, and the next environment with a working daemon must see these run
and either pass or fail on their merits.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import as_of, create_admin_engine, ingest_writer_session
from backend.db.models import PriceBar, Security, SecurityMaster
from backend.universe import builder
from backend.universe.builder import (
    adv_window_start,
    assemble_candidates,
    build_universe,
    read_listings,
    read_price_window,
    screen_candidates,
)
from backend.universe.criteria import UniverseCriteria
from backend.universe.errors import UniverseInputUnavailableError, UniverseSessionError
from backend.universe.history import UniverseHistory, load_history, persist_history
from backend.universe.snapshot import load_snapshots, persist_snapshot
from backend.universe.waterfall import filter_waterfall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from backend.universe.snapshot import UniverseSnapshot

REBALANCE_DATE = dt.date(2020, 3, 31)
"""A Tuesday in the middle of the fixture history."""

EARLY_KNOWLEDGE = dt.datetime(2015, 1, 1, tzinfo=dt.UTC)
"""When the fixture identities and bars became knowable."""

DELISTING_KNOWLEDGE = dt.datetime(2020, 6, 30, tzinfo=dt.UTC)
"""When the fixture delisting became knowable — after the rebalance date."""


def fixture_criteria(*, require_borrow: bool = False) -> UniverseCriteria:
    """Round-number criteria. Not a recommended configuration; a readable one."""
    return UniverseCriteria(
        min_adv_usd=Decimal("1000000"),
        min_price_usd=Decimal("5"),
        min_market_cap_usd=Decimal("300000000"),
        allowed_exchanges=frozenset({"XNYS", "XNAS"}),
        require_borrow=require_borrow,
        adv_lookback_days=20,
    )


@pytest.fixture(autouse=True)
async def _clean_universe_tables() -> AsyncIterator[None]:
    """Truncate the two universe tables after every test in this module.

    The shared integration fixture truncates the Phase 2 fact tables and reaches
    ``universe_member`` through its CASCADE (that table holds a foreign key to
    ``security``), but nothing there reaches ``universe_snapshot``. TRUNCATE is
    deliberately left unblocked by migration 0011's append-only trigger, matching
    revisions 0003/0004/0007/0009/0010, precisely so this sanctioned reset path
    exists. An ordinary admin engine suffices: the Core guard refuses SQL naming
    a *bitemporal* fact table, and neither of these is one.
    """
    yield
    engine = create_admin_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(
                sa.text(
                    "TRUNCATE TABLE universe_member, universe_snapshot RESTART IDENTITY CASCADE"
                )
            )
    finally:
        await engine.dispose()


async def _create_security() -> int:
    """Create one identity anchor and return its database-generated key."""
    async with ingest_writer_session() as session:
        anchor = Security()
        session.add(anchor)
        await session.flush()
        security_id = anchor.security_id
        await session.commit()
    return security_id


def _master(
    security_id: int,
    *,
    knowledge_time: dt.datetime = EARLY_KNOWLEDGE,
    exchange: str = "XNYS",
    first_listed_on: dt.date | None = dt.date(2010, 1, 4),
    delisted_on: dt.date | None = None,
) -> SecurityMaster:
    """Build one identity version, open-ended in event time from 2010."""
    return SecurityMaster(
        security_id=security_id,
        ticker=f"FIX{security_id}",
        name=f"Fixture Issuer {security_id}",
        exchange=exchange,
        first_listed_on=first_listed_on,
        delisted_on=delisted_on,
        valid_from=dt.datetime(2010, 1, 4, tzinfo=dt.UTC),
        knowledge_time=knowledge_time,
    )


def _bar(
    security_id: int,
    day: dt.date,
    *,
    knowledge_time: dt.datetime | None = None,
    close_usd: str = "10",
    volume_shares: int = 1_000_000,
) -> PriceBar:
    """Build one daily bar, event-time-encoded as ``[D 00:00Z, D+1 00:00Z)``.

    ``knowledge_time`` defaults to the close of the trading day itself, which is
    the earliest instant the bar could honestly be known.
    """
    valid_from = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)
    price = Decimal(close_usd)
    return PriceBar(
        security_id=security_id,
        valid_from=valid_from,
        valid_to=valid_from + dt.timedelta(days=1),
        knowledge_time=knowledge_time or (valid_from + dt.timedelta(hours=21)),
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal("1"),
        volume_shares=volume_shares,
    )


def _weekdays_ending(last_date: dt.date, count: int) -> list[dt.date]:
    """Return ``count`` weekday dates ending at ``last_date``, ascending."""
    dates: list[dt.date] = []
    cursor = last_date
    while len(dates) < count:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor -= dt.timedelta(days=1)
    return sorted(dates)


async def _insert(*rows: PriceBar | SecurityMaster) -> None:
    """Insert rows through the sanctioned append-only write path."""
    async with ingest_writer_session() as session:
        session.add_all(list(rows))
        await session.commit()


async def _survivor_and_later_delisted() -> tuple[int, int]:
    """Create two names: one still listed, one delisted three months later.

    Both trade identically for the 40 weekdays ending at the rebalance date, so
    the only thing that can separate them is the delisting — which is the point.
    """
    survivor = await _create_security()
    delisted = await _create_security()
    await _insert(
        _master(survivor),
        # The delisting became knowable on 2020-06-30, which is *after* the
        # rebalance date. Read at today's as-of, this is the version the store
        # returns, and it is the row a naive builder would reject.
        _master(delisted, knowledge_time=DELISTING_KNOWLEDGE, delisted_on=dt.date(2020, 6, 30)),
    )
    bars = [
        _bar(security_id, day)
        for security_id in (survivor, delisted)
        for day in _weekdays_ending(REBALANCE_DATE, 40)
    ]
    await _insert(*bars)
    return survivor, delisted


async def _screen_with_supplied_inputs(
    *, rebalance_date: dt.date, as_of_ts: dt.datetime
) -> UniverseSnapshot:
    """Read point-in-time, then screen with market caps supplied by the test.

    The market-cap and borrow inputs have no source in this system (B1, B2), so
    :func:`~backend.universe.builder.build_universe` refuses before it ever gets
    here. Supplying them explicitly, in the test, is the only way to exercise the
    screen that runs once those feeds land — and it is legitimate because the
    values are written out here rather than invented inside the system.
    """
    criteria = fixture_criteria()
    async with as_of(as_of_ts) as session:
        listings = await read_listings(session, rebalance_date=rebalance_date)
        bars = await read_price_window(
            session,
            first_date=adv_window_start(rebalance_date, criteria.adv_lookback_days),
            last_date=rebalance_date,
        )
    candidates = [
        dataclasses.replace(candidate, market_cap_usd=Decimal("1000000000"), borrow_available=True)
        for candidate in assemble_candidates(
            listings, bars, rebalance_date=rebalance_date, criteria=criteria
        )
    ]
    return screen_candidates(
        candidates, rebalance_date=rebalance_date, criteria=criteria, as_of=as_of_ts
    )


# --- Gate G4: survivorship-bias elimination --------------------------------


class TestSurvivorshipBiasElimination:
    async def test_a_name_delisted_after_the_date_is_a_candidate_in_that_universe(
        self,
    ) -> None:
        survivor, delisted = await _survivor_and_later_delisted()
        now = dt.datetime.now(dt.UTC)
        async with as_of(now) as session:
            listings = await read_listings(session, rebalance_date=REBALANCE_DATE)
        by_id = {listing.security_id: listing for listing in listings}
        # The store returns the version that already records the delisting...
        assert by_id[delisted].delisted_on == dt.date(2020, 6, 30)
        # ...and the predicate keeps the name anyway, because on 2020-03-31 it
        # was listed and tradeable.
        assert by_id[delisted].is_listed_on(REBALANCE_DATE) is True
        assert by_id[survivor].is_listed_on(REBALANCE_DATE) is True

    async def test_the_later_delisted_name_is_a_member_of_the_past_universe(self) -> None:
        survivor, delisted = await _survivor_and_later_delisted()
        snapshot = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=dt.datetime.now(dt.UTC)
        )
        assert set(snapshot.members) == {survivor, delisted}

    async def test_the_name_is_absent_from_a_universe_built_after_its_delisting(self) -> None:
        # The other half of the same property: the predicate is not simply
        # permissive. A universe built for a date *after* the delisting must not
        # contain the name.
        survivor, delisted = await _survivor_and_later_delisted()
        later = dt.date(2020, 9, 30)
        await _insert(
            *[_bar(survivor, day) for day in _weekdays_ending(later, 40)],
        )
        snapshot = await _screen_with_supplied_inputs(
            rebalance_date=later, as_of_ts=dt.datetime.now(dt.UTC)
        )
        assert delisted not in snapshot.members
        assert survivor in snapshot.members

    async def test_the_whole_builder_considers_the_later_delisted_name(self) -> None:
        # Through build_universe itself, against real SQL, with the availability
        # gate lifted — membership is unreachable without a market-cap source,
        # but candidacy is where survivorship bias enters and candidacy is
        # reachable.
        survivor, delisted = await _survivor_and_later_delisted()
        now = dt.datetime.now(dt.UTC)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            async with as_of(now) as session:
                snapshot = await build_universe(
                    session, rebalance_date=REBALANCE_DATE, criteria=fixture_criteria()
                )
        assert {outcome.security_id for outcome in snapshot.outcomes} == {survivor, delisted}


# --- Invariant I1: the as-of read ------------------------------------------


class TestTemporalIntegrity:
    async def test_a_bar_knowable_only_later_does_not_enter_the_adv(self) -> None:
        security_id = await _create_security()
        await _insert(_master(security_id))
        days = _weekdays_ending(REBALANCE_DATE, 20)
        await _insert(*[_bar(security_id, day) for day in days[:-1]])
        # The last bar was not knowable until a year after the rebalance date.
        await _insert(
            _bar(
                security_id,
                days[-1],
                knowledge_time=dt.datetime(2021, 3, 31, tzinfo=dt.UTC),
            )
        )
        early = dt.datetime(2020, 4, 1, tzinfo=dt.UTC)
        criteria = fixture_criteria()
        async with as_of(early) as session:
            listings = await read_listings(session, rebalance_date=REBALANCE_DATE)
            bars = await read_price_window(
                session,
                first_date=adv_window_start(REBALANCE_DATE, criteria.adv_lookback_days),
                last_date=REBALANCE_DATE,
            )
        candidates = assemble_candidates(
            listings, bars, rebalance_date=REBALANCE_DATE, criteria=criteria
        )
        # 19 of the 20 requested bars were knowable, so the ADV is unmeasurable
        # rather than computed from a shorter window.
        assert candidates[0].adv_usd is None

    async def test_a_delisting_learned_later_is_invisible_to_an_earlier_read(self) -> None:
        security_id = await _create_security()
        await _insert(_master(security_id))
        await _insert(
            _master(
                security_id,
                knowledge_time=DELISTING_KNOWLEDGE,
                delisted_on=dt.date(2020, 6, 30),
            )
        )
        before = dt.datetime(2020, 4, 1, tzinfo=dt.UTC)
        async with as_of(before) as session:
            early_listings = await read_listings(session, rebalance_date=REBALANCE_DATE)
        async with as_of(dt.datetime.now(dt.UTC)) as session:
            late_listings = await read_listings(session, rebalance_date=REBALANCE_DATE)
        assert early_listings[0].delisted_on is None
        assert late_listings[0].delisted_on == dt.date(2020, 6, 30)
        # Both reads agree on the thing that matters: the name was listed on the
        # rebalance date. The later knowledge changes what we know about its
        # future, not what the universe was.
        assert early_listings[0].is_listed_on(REBALANCE_DATE)
        assert late_listings[0].is_listed_on(REBALANCE_DATE)

    async def test_an_ingestion_writer_session_cannot_build_a_universe(self) -> None:
        # A writer session carries no as-of bound, so the refusal happens at the
        # builder's entry point rather than deep inside a statement rewrite.
        async with ingest_writer_session() as session:
            with pytest.raises(UniverseSessionError, match="not scoped by as_of"):
                await build_universe(
                    session, rebalance_date=REBALANCE_DATE, criteria=fixture_criteria()
                )


# --- Invariant I3: the refusal ---------------------------------------------


class TestRefusalAgainstARealDatabase:
    async def test_the_build_refuses_because_the_market_cap_input_does_not_exist(
        self,
    ) -> None:
        await _survivor_and_later_delisted()
        async with as_of(dt.datetime.now(dt.UTC)) as session:
            with pytest.raises(UniverseInputUnavailableError) as caught:
                await build_universe(
                    session, rebalance_date=REBALANCE_DATE, criteria=fixture_criteria()
                )
        assert caught.value.filter_name == "market_cap"
        assert caught.value.blocker == "B1"

    async def test_no_snapshot_is_written_when_the_build_refuses(self) -> None:
        await _survivor_and_later_delisted()
        criteria = fixture_criteria()
        async with as_of(dt.datetime.now(dt.UTC)) as session:
            with pytest.raises(UniverseInputUnavailableError):
                await build_universe(session, rebalance_date=REBALANCE_DATE, criteria=criteria)
        async with ingest_writer_session() as session:
            stored = await load_snapshots(session, criteria_hash=criteria.criteria_hash())
        assert stored == ()


# --- Persistence: migration 0011 -------------------------------------------


class TestSnapshotPersistence:
    async def test_a_snapshot_survives_the_round_trip_through_the_database(self) -> None:
        survivor, delisted = await _survivor_and_later_delisted()
        built = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=dt.datetime.now(dt.UTC)
        )
        async with ingest_writer_session() as session:
            await persist_snapshot(session, built)
            await session.commit()
        async with ingest_writer_session() as session:
            (loaded,) = await load_snapshots(session, criteria_hash=built.criteria_hash)
        assert loaded == built
        assert set(loaded.members) == {survivor, delisted}

    async def test_every_candidate_gets_a_row_not_only_the_members(self) -> None:
        # The waterfall is unreconstructible from a members-only table, and P4.2
        # requires exactly that reconstruction.
        survivor = await _create_security()
        excluded = await _create_security()
        await _insert(_master(survivor), _master(excluded, exchange="XLON"))
        await _insert(
            *[
                _bar(security_id, day)
                for security_id in (survivor, excluded)
                for day in _weekdays_ending(REBALANCE_DATE, 40)
            ]
        )
        built = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=dt.datetime.now(dt.UTC)
        )
        async with ingest_writer_session() as session:
            await persist_snapshot(session, built)
            await session.commit()
            (loaded,) = await load_snapshots(session, criteria_hash=built.criteria_hash)
        assert loaded.candidate_count == 2
        assert loaded.members == (survivor,)
        waterfall = filter_waterfall(loaded)
        assert waterfall.total_removed + waterfall.member_count == waterfall.candidate_count
        exchange_step = waterfall.step_for("exchange")
        assert exchange_step is not None
        assert exchange_step.removed == 1

    async def test_the_same_build_cannot_be_stored_twice(self) -> None:
        await _survivor_and_later_delisted()
        instant = dt.datetime.now(dt.UTC)
        built = await _screen_with_supplied_inputs(rebalance_date=REBALANCE_DATE, as_of_ts=instant)

        async def store() -> None:
            async with ingest_writer_session() as session:
                await persist_snapshot(session, built)
                await session.commit()

        await store()
        with pytest.raises(IntegrityError):
            await store()

    async def test_rebuilding_at_a_later_as_of_appends_rather_than_corrects(self) -> None:
        # Same criteria, same rebalance date, later knowledge instant: a new row.
        # The difference between the two answers is information about the data,
        # and a schema that overwrote the first would destroy it.
        await _survivor_and_later_delisted()
        first_instant = dt.datetime(2020, 4, 1, tzinfo=dt.UTC)
        second_instant = dt.datetime.now(dt.UTC)
        first = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=first_instant
        )
        second = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=second_instant
        )
        async with ingest_writer_session() as session:
            await persist_snapshot(session, first)
            await persist_snapshot(session, second)
            await session.commit()
            latest = await load_snapshots(session, criteria_hash=first.criteria_hash)
            as_known_then = await load_snapshots(
                session,
                criteria_hash=first.criteria_hash,
                knowledge_cutoff=first_instant,
            )
        # One snapshot per rebalance date, latest knowledge wins...
        assert [item.as_of for item in latest] == [second_instant]
        # ...and the point-in-time browser still reaches the earlier build.
        assert [item.as_of for item in as_known_then] == [first_instant]

    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE universe_snapshot SET member_count = 0",
            "DELETE FROM universe_snapshot",
            "UPDATE universe_member SET included = false",
            "DELETE FROM universe_member",
        ],
    )
    async def test_the_universe_tables_are_append_only_in_the_database(
        self, statement: str
    ) -> None:
        await _survivor_and_later_delisted()
        built = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=dt.datetime.now(dt.UTC)
        )
        async with ingest_writer_session() as session:
            await persist_snapshot(session, built)
            await session.commit()
        engine = create_admin_engine()
        try:
            with pytest.raises(DBAPIError, match="append-only"):
                async with engine.begin() as connection:
                    await connection.execute(sa.text(statement))
        finally:
            await engine.dispose()

    async def test_a_member_row_cannot_claim_inclusion_while_recording_a_failure(self) -> None:
        # The CHECK is the database's copy of "a member is exactly a candidate
        # that failed nothing", so the two can never come to disagree.
        await _survivor_and_later_delisted()
        built = await _screen_with_supplied_inputs(
            rebalance_date=REBALANCE_DATE, as_of_ts=dt.datetime.now(dt.UTC)
        )
        async with ingest_writer_session() as session:
            snapshot_id = await persist_snapshot(session, built)
            await session.commit()
        engine = create_admin_engine()
        try:
            with pytest.raises(DBAPIError, match="included_iff_no_failures"):
                async with engine.begin() as connection:
                    await connection.execute(
                        sa.text(
                            "INSERT INTO universe_member "
                            "(snapshot_id, security_id, included, failed_filters) "
                            "VALUES (:snapshot_id, :security_id, true, '[\"price\"]'::jsonb)"
                        ),
                        {"snapshot_id": snapshot_id, "security_id": built.members[0] + 1000},
                    )
        finally:
            await engine.dispose()


# --- P4.2: history over a schedule ------------------------------------------


class TestHistoryPersistence:
    async def test_a_history_round_trips_and_yields_its_series(self) -> None:
        survivor, delisted = await _survivor_and_later_delisted()
        later = dt.date(2020, 9, 30)
        await _insert(*[_bar(survivor, day) for day in _weekdays_ending(later, 40)])
        now = dt.datetime.now(dt.UTC)
        snapshots = [
            await _screen_with_supplied_inputs(rebalance_date=date, as_of_ts=now)
            for date in (REBALANCE_DATE, later)
        ]
        history = UniverseHistory.from_snapshots(snapshots)
        async with ingest_writer_session() as session:
            identifiers = await persist_history(session, history)
            await session.commit()
        assert len(identifiers) == 2

        async with ingest_writer_session() as session:
            loaded = await load_history(session, criteria=fixture_criteria())
        assert loaded is not None
        assert loaded.rebalance_dates == (REBALANCE_DATE, later)
        assert [point.member_count for point in loaded.size_series()] == [2, 1]
        (turnover,) = loaded.turnover_series()
        assert turnover.exited == (delisted,)
        # (0 + 1) / (2 + 1) = 1/3 — the delisted name left, nothing joined.
        assert turnover.turnover_fraction == pytest.approx(1 / 3)

    async def test_loading_a_history_that_was_never_stored_gives_nothing(self) -> None:
        async with ingest_writer_session() as session:
            assert await load_history(session, criteria=fixture_criteria()) is None
