"""P3.8 integration: macro vintage ingestion against real TimescaleDB.

What is proved here and nowhere else:

- **the revision case, end to end through the as-of read path.** A series
  revised twice returns the *original* value at an ``as_of`` before the first
  revision, the first revision's value between them, and the latest after —
  from real rows, through :func:`backend.db.as_of`, with no test-local
  reimplementation of the read semantics. This is the property macro data
  exists to threaten and the reason P3.8 ingests vintages at all;
- **the knowledge boundary**: an observation is invisible to every ``as_of``
  before its derived vintage instant and visible from that instant on (I1);
- FRED's missing marker survives the round trip as ``NULL`` + ``is_missing``,
  and a genuine ``0`` survives as ``0`` — checked after a database round trip,
  because a coercion could as easily happen in the driver as in the parser;
- the schema migration 0008 installs is the one D-011 requires (as-of indices,
  append-only triggers, the missing-value and event-time CHECKs), and those
  CHECKs actually refuse the rows they are meant to;
- the checkpoint survives across runs, and a re-run neither loses nor
  duplicates observations;
- the CC.8 connector contract: an unavailable source raises and writes nothing.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import as_of, ingest_writer_session
from backend.db.bitemporal import INFINITY

# TID251: schema introspection and the fixture-local reset must reach Postgres
# beneath the Core guard, which rejects textual SQL naming a fact table by
# design (D-011/D-012). Same sanctioned use as in ``test_edgar_ingestion.py``.
from backend.db.engine import _create_migration_engine as _migration_engine  # noqa: TID251
from backend.db.models import MacroObservation, MacroSeries
from backend.ingest.errors import TransientSourceError
from backend.ingest.fred.client import FredClient
from backend.ingest.fred.connector import CHECKPOINT_VINTAGE_PREFIX, FredConnector
from backend.ingest.fred.parse import vintage_date_to_knowledge_time
from backend.ingest.runs import RunKind, RunStatus, latest_checkpoint
from backend.tests.ingest.test_fred_fixtures import TEST_API_KEY, fred_transport
from backend.tests.integration.connector_contract import ConnectorContractTests

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx

    from backend.ingest.base import Connector

# The three vintages of observation date 2024-01-01 in the revision payload,
# and the instants they become knowable under the connector's declared policy.
_OBSERVED = dt.date(2024, 1, 1)
_VINTAGE_1 = dt.date(2024, 4, 25)
_VINTAGE_2 = dt.date(2024, 5, 30)
_VINTAGE_3 = dt.date(2024, 6, 27)
_KNOWN_1 = vintage_date_to_knowledge_time(_VINTAGE_1)
_KNOWN_2 = vintage_date_to_knowledge_time(_VINTAGE_2)
_KNOWN_3 = vintage_date_to_knowledge_time(_VINTAGE_3)


@pytest.fixture(autouse=True)
async def _clean_macro_tables() -> AsyncIterator[None]:
    """Truncate the macro fact tables after every test in this module.

    The shared integration fixture truncates the Phase 2 tables; these are new
    in migration 0008 and have no foreign key to them, so CASCADE does not
    reach them. Run on the unguarded migration engine for the same reason the
    shared fixture does: TRUNCATE names a fact table in textual SQL, which the
    Core guard refuses on every guarded engine by design.
    """
    yield
    engine = _migration_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text("TRUNCATE TABLE macro_observation, macro_series"))
    finally:
        await engine.dispose()


def _connector(
    *,
    series_ids: tuple[str, ...] = ("REVISEDTWICE",),
    transport: httpx.MockTransport | None = None,
) -> FredConnector:
    """Build a connector on the test payloads with a real wall clock.

    The clock is real on purpose: ``run()`` stamps the ingestion-run record
    from it, and a fabricated 2024 clock would write a run that ended before it
    started.
    """
    return FredConnector(
        client=FredClient(
            api_key=TEST_API_KEY,
            transport=transport if transport is not None else fred_transport(),
        ),
        api_key=TEST_API_KEY,
        series_ids=series_ids,
    )


async def _value_at(as_of_ts: dt.datetime, *, series_id: str, observed: dt.date) -> Decimal | None:
    """Return one observation's value as visible through the as-of read path."""
    async with as_of(as_of_ts) as session:
        rows = (
            await session.scalars(
                sa.select(MacroObservation).where(
                    MacroObservation.series_id == series_id,
                    MacroObservation.observation_date == observed,
                )
            )
        ).all()
    assert len(rows) <= 1, (
        f"as_of {as_of_ts.isoformat()} returned {len(rows)} rows for one (series, "
        "observation date); latest-knowledge-wins must yield exactly one version"
    )
    return rows[0].value if rows else None


async def _write(row: MacroObservation) -> None:
    """Add one row through the sanctioned writer session and commit.

    A named helper so the constraint tests below can put a single statement
    inside ``pytest.raises`` — otherwise the assertion could pass because some
    *other* line in the block raised.
    """
    async with ingest_writer_session() as session:
        session.add(row)
        await session.commit()


async def _stored_observations() -> list[MacroObservation]:
    """Return every observation visible now, through the sanctioned read path."""
    async with as_of(dt.datetime.now(dt.UTC)) as session:
        return list(
            (
                await session.scalars(
                    sa.select(MacroObservation).order_by(MacroObservation.observation_date)
                )
            ).all()
        )


# --- the revision case: the reason this connector exists ---------------------


async def test_a_twice_revised_observation_returns_the_value_believed_at_each_as_of() -> None:
    """The central P3.8 property, through real rows and the real read path.

    Before the first revision the store must answer with the ORIGINAL number —
    the one that was wrong but believed — and only after each revision's
    knowledge instant with the revised one. A connector that ingested current
    values would pass nothing here except by accident.
    """
    await _connector().run(RunKind.BACKFILL)

    just_after_first = _KNOWN_1 + dt.timedelta(seconds=1)
    just_before_second = _KNOWN_2 - dt.timedelta(seconds=1)
    just_after_second = _KNOWN_2 + dt.timedelta(seconds=1)
    just_before_third = _KNOWN_3 - dt.timedelta(seconds=1)
    just_after_third = _KNOWN_3 + dt.timedelta(seconds=1)

    assert await _value_at(just_after_first, series_id="REVISEDTWICE", observed=_OBSERVED) == (
        Decimal("1.1")
    )
    assert await _value_at(just_before_second, series_id="REVISEDTWICE", observed=_OBSERVED) == (
        Decimal("1.1")
    )
    assert await _value_at(just_after_second, series_id="REVISEDTWICE", observed=_OBSERVED) == (
        Decimal("2.2")
    )
    assert await _value_at(just_before_third, series_id="REVISEDTWICE", observed=_OBSERVED) == (
        Decimal("2.2")
    )
    assert await _value_at(just_after_third, series_id="REVISEDTWICE", observed=_OBSERVED) == (
        Decimal("3.3")
    )
    assert await _value_at(
        dt.datetime.now(dt.UTC), series_id="REVISEDTWICE", observed=_OBSERVED
    ) == Decimal("3.3")


async def test_an_observation_is_invisible_before_its_vintage_becomes_knowable() -> None:
    """I1 at the boundary: not one second early.

    The boundary sits at the derived vintage instant, not at the observation
    date — which for this row is nearly four months earlier. Querying at the
    observation date must see nothing at all.
    """
    await _connector().run(RunKind.BACKFILL)

    assert (
        await _value_at(
            dt.datetime.combine(_OBSERVED, dt.time.min, tzinfo=dt.UTC),
            series_id="REVISEDTWICE",
            observed=_OBSERVED,
        )
        is None
    )
    assert (
        await _value_at(
            _KNOWN_1 - dt.timedelta(seconds=1), series_id="REVISEDTWICE", observed=_OBSERVED
        )
        is None
    )
    assert await _value_at(_KNOWN_1, series_id="REVISEDTWICE", observed=_OBSERVED) == Decimal("1.1")


async def test_every_vintage_is_stored_as_its_own_row() -> None:
    """Revisions are appended, never applied: three rows, one logical fact."""
    await _connector().run(RunKind.BACKFILL)
    engine = _migration_engine()
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    sa.text(
                        "SELECT knowledge_time, value, vintage_start_date FROM macro_observation "
                        "WHERE series_id = 'REVISEDTWICE' AND observation_date = :observed "
                        "ORDER BY knowledge_time"
                    ),
                    {"observed": _OBSERVED},
                )
            ).all()
    finally:
        await engine.dispose()
    assert [row.value for row in rows] == [Decimal("1.1"), Decimal("2.2"), Decimal("3.3")]
    assert [row.vintage_start_date for row in rows] == [_VINTAGE_1, _VINTAGE_2, _VINTAGE_3]
    assert [row.knowledge_time for row in rows] == [_KNOWN_1, _KNOWN_2, _KNOWN_3]


async def test_the_as_of_read_never_returns_two_versions_of_one_observation() -> None:
    """Overlapping open event-time intervals must not fan out.

    Consecutive macro observations deliberately carry overlapping open
    ``[observation_date, infinity)`` intervals. That is safe only because they
    are distinct logical keys; this walks a range of as-ofs to prove the read
    path really does collapse to one version per (series, observation date).
    """
    await _connector().run(RunKind.BACKFILL)
    probe = _KNOWN_1
    while probe < _KNOWN_3 + dt.timedelta(days=60):
        await _value_at(probe, series_id="REVISEDTWICE", observed=_OBSERVED)  # asserts internally
        probe += dt.timedelta(days=7)


async def test_series_metadata_carries_the_units_at_each_vintage() -> None:
    """Directive §8: a macro value is meaningless without its unit.

    The metadata is versioned too, so the units visible at an ``as_of`` are the
    ones that were published then.
    """
    await _connector().run(RunKind.BACKFILL)
    async with as_of(dt.datetime.now(dt.UTC)) as session:
        series = (
            await session.scalars(
                sa.select(MacroSeries).where(MacroSeries.series_id == "REVISEDTWICE")
            )
        ).all()
    assert len(series) == 1  # latest metadata version wins
    assert series[0].units
    assert series[0].frequency_short == "Q"
    assert series[0].title.endswith("(Renamed)")

    earlier_only = vintage_date_to_knowledge_time(_VINTAGE_1) + dt.timedelta(days=1)
    async with as_of(earlier_only) as session:
        earlier = (
            await session.scalars(
                sa.select(MacroSeries).where(MacroSeries.series_id == "REVISEDTWICE")
            )
        ).all()
    assert not earlier[0].title.endswith("(Renamed)")


# --- missing values, through a real round trip -------------------------------


async def test_a_missing_value_round_trips_as_null_and_never_as_zero() -> None:
    """FRED's '.' must not become 0 or NaN in the database either."""
    await _connector(series_ids=("WITHMISSING",)).run(RunKind.BACKFILL)
    stored = {row.observation_date: row for row in await _stored_observations()}
    assert len(stored) == 3  # nothing dropped

    missing = stored[dt.date(2024, 1, 2)]
    assert missing.is_missing is True
    assert missing.value is None

    real_zero = stored[dt.date(2024, 1, 3)]
    assert real_zero.is_missing is False
    assert real_zero.value == Decimal("0")

    present = stored[dt.date(2024, 1, 1)]
    assert present.is_missing is False
    assert present.value == Decimal("5.5")


async def test_the_missing_value_check_refuses_an_inconsistent_row() -> None:
    """The invariant is in the schema, not only in the connector.

    A future writer that set ``is_missing`` beside a value — or a NULL value
    without the flag — would make "no data" and "a real number" the same row.
    The database refuses both.
    """
    for is_missing, value in ((True, Decimal("1.0")), (False, None)):
        row = MacroObservation(
            series_id="CHECKPROBE",
            observation_date=dt.date(2024, 1, 1),
            value=value,
            is_missing=is_missing,
            vintage_start_date=dt.date(2024, 2, 1),
            vintage_end_date=None,
            valid_from=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            valid_to=INFINITY,
            knowledge_time=vintage_date_to_knowledge_time(dt.date(2024, 2, 1)),
        )
        with pytest.raises(IntegrityError, match="ck_macro_observation_missing_value"):
            await _write(row)


async def test_the_event_time_check_refuses_a_valid_from_that_drifts() -> None:
    """``valid_from`` and ``observation_date`` are redundant on purpose.

    The CHECK is what keeps the redundancy from becoming a divergence: an
    observation date that disagrees with the event-time coordinate the as-of
    layer indexes would make point-in-time reads silently wrong.
    """
    row = MacroObservation(
        series_id="CHECKPROBE",
        observation_date=dt.date(2024, 1, 1),
        value=Decimal("1.0"),
        is_missing=False,
        vintage_start_date=dt.date(2024, 2, 1),
        vintage_end_date=None,
        valid_from=dt.datetime(2024, 1, 2, tzinfo=dt.UTC),  # one day off
        valid_to=INFINITY,
        knowledge_time=vintage_date_to_knowledge_time(dt.date(2024, 2, 1)),
    )
    with pytest.raises(IntegrityError, match="ck_macro_observation_valid_from_matches_date"):
        await _write(row)


# --- schema shape (migration 0008) -------------------------------------------


@pytest.mark.parametrize("table", ["macro_series", "macro_observation"])
async def test_the_macro_tables_are_append_only_in_the_database(table: str) -> None:
    """UPDATE and DELETE are refused by trigger, on every role (D-012)."""
    await _connector().run(RunKind.BACKFILL)
    engine = _migration_engine()
    try:
        for statement in (
            f"UPDATE {table} SET knowledge_time = now()",  # noqa: S608 — parametrized table name
            f"DELETE FROM {table}",  # noqa: S608 — parametrized table name
        ):
            with pytest.raises(DBAPIError):
                async with engine.begin() as connection:
                    await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


async def test_migration_0008_installs_the_asof_indices() -> None:
    """The composite index must match the DISTINCT ON / ORDER BY exactly."""
    engine = _migration_engine()
    try:
        async with engine.connect() as connection:
            definitions = dict(
                (
                    await connection.execute(
                        sa.text(
                            "SELECT indexname, indexdef FROM pg_indexes "
                            "WHERE tablename IN ('macro_series', 'macro_observation')"
                        )
                    )
                ).all()
            )
    finally:
        await engine.dispose()
    observation_index = definitions["ix_macro_observation_asof_lookup"]
    assert "series_id, observation_date, valid_from, knowledge_time DESC" in observation_index
    assert (
        "series_id, valid_from, knowledge_time DESC" in definitions["ix_macro_series_asof_lookup"]
    )


async def test_the_primary_key_makes_a_revision_a_new_row() -> None:
    """``knowledge_time`` is in the key, so versions cannot collide."""
    engine = _migration_engine()
    try:
        async with engine.connect() as connection:
            columns = (
                await connection.execute(
                    sa.text(
                        "SELECT a.attname FROM pg_index i "
                        "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                        "AND a.attnum = ANY(i.indkey) "
                        "WHERE i.indrelid = 'macro_observation'::regclass AND i.indisprimary"
                    )
                )
            ).scalars()
            key = set(columns)
    finally:
        await engine.dispose()
    assert key == {"series_id", "observation_date", "valid_from", "knowledge_time"}


# --- run bookkeeping and resumption ------------------------------------------


async def test_a_successful_run_records_its_checkpoint_and_metrics() -> None:
    """The run record is the checkpoint store; there is no second source of truth."""
    result = await _connector().run(RunKind.BACKFILL)
    assert result.rows_written > 0
    assert result.checkpoint_before is None
    assert result.checkpoint_after == {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "2024-07-25"}
    assert await latest_checkpoint("fred") == result.checkpoint_after
    names = {metric.name for metric in result.metrics}
    assert {"observation_rows_read", "vintages_already_known", "series_metadata_versions"} <= names


async def test_a_backfill_run_reports_no_lag_figure() -> None:
    """Backfill knowledge times are historical by design; a lag would be noise."""
    result = await _connector().run(RunKind.BACKFILL)
    assert result.lag_report is None


async def test_a_second_run_writes_nothing_new_and_keeps_the_checkpoint() -> None:
    """Resumption is idempotent: no duplicates, no lost rows."""
    first = await _connector().run(RunKind.BACKFILL)
    before = len(await _stored_observations())
    second = await _connector().run(RunKind.LIVE)
    assert second.rows_written == 0
    assert second.checkpoint_before == first.checkpoint_after
    assert second.checkpoint_after == first.checkpoint_after
    assert len(await _stored_observations()) == before


async def test_rewriting_the_same_vintage_is_a_no_op_rather_than_a_collision() -> None:
    """A run killed between commit and checkpoint must not wedge the source.

    Driven from a fresh connector with no checkpoint, so it re-reads and
    re-writes everything the first run already wrote.
    """
    await _connector().run(RunKind.BACKFILL)
    before = await _stored_observations()
    replayed = _connector()
    async for batch in replayed.fetch_batches(None):
        await replayed._write_batch(batch.rows)
    after = await _stored_observations()
    assert len(after) == len(before)


async def test_a_failed_run_is_recorded_and_writes_nothing() -> None:
    """I3: an unavailable source raises, and leaves nothing plausible behind."""
    connector = _connector(transport=fred_transport(unavailable=True))
    with pytest.raises(TransientSourceError):
        await connector.run(RunKind.LIVE)
    assert await _stored_observations() == []
    async with ingest_writer_session() as session:
        status = (
            await session.execute(
                sa.text(
                    "SELECT status FROM ingestion_run WHERE source = 'fred' "
                    "ORDER BY run_id DESC LIMIT 1"
                )
            )
        ).scalar_one()
    assert status == RunStatus.FAILED.value


class TestFredConnectorContract(ConnectorContractTests):
    """CC.8: the shared unavailable-source contract, against this connector."""

    expected_error: ClassVar[type[TransientSourceError]] = TransientSourceError

    def build_connector(self) -> Connector:
        """Return a FRED connector whose source cannot be reached."""
        return _connector(transport=fred_transport(unavailable=True))
