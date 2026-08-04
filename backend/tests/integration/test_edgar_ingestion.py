"""P3.2/P3.9 integration: EDGAR ingestion against real TimescaleDB.

What is proved here and nowhere else:

- filings parsed from captured EDGAR responses are written through the
  sanctioned writer session and read back through :func:`backend.db.as_of`;
- **the knowledge-time boundary holds on real data**: a filing is invisible to
  every as-of before its acceptance instant and visible from that instant on.
  That is invariant I1 stated in the terms this connector exists to get right —
  the boundary sits at the acceptance instant, not at the filing date;
- the checkpoint survives across runs and an interrupted run neither loses nor
  duplicates filings;
- the schema migration 0006 installs is the one D-011 requires (hypertable,
  as-of indices, append-only triggers);
- the P3.9 report measures what is there and reports ``None`` for what is not.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, ClassVar

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from backend.db import as_of, ingest_writer_session
from backend.db.bitemporal import INFINITY, BitemporalMixin

# TID251: schema introspection and the fixture-local reset must reach Postgres
# beneath the Core guard, which rejects textual SQL naming a fact table by
# design (D-011/D-012). This is the same sanctioned use as in
# ``test_hypertable_append_only.py`` and ``conftest.py``.
from backend.db.engine import _create_migration_engine as _migration_engine  # noqa: TID251
from backend.db.models import EdgarFiling, EdgarFilingDocument
from backend.ingest.base import ConnectorRuntime
from backend.ingest.edgar.client import EdgarClient
from backend.ingest.edgar.connector import CHECKPOINT_LAST_INDEX_DATE, SecEdgarConnector
from backend.ingest.edgar.report import build_edgar_report, detect_gaps
from backend.ingest.errors import PermanentSourceError, TransientSourceError
from backend.ingest.runs import RunKind, RunStatus, latest_checkpoint
from backend.tests.ingest.test_edgar_fixtures import NonBlockingClock, edgar_transport
from backend.tests.integration.connector_contract import ConnectorContractTests

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from backend.ingest.base import Connector

_USER_AGENT = "Quant Research Platform test-contact@example.com"
_CAPTURED_FORMS = frozenset({"8-K", "4", "D", "CORRESP"})
# The 2024-03-08 sample lists Form 4 rows whose headers were not captured,
# so tests that span it select the forms the fixtures cover. The selection
# mechanism is the same one; only the day range differs.
_MULTI_DAY_FORMS = frozenset({"8-K", "D", "CORRESP"})
_MARCH_08 = dt.date(2024, 3, 8)
_MARCH_11 = dt.date(2024, 3, 11)
_MARCH_12 = dt.date(2024, 3, 12)

# The two acceptance instants the boundary assertions pivot on, both taken from
# the captured headers: a Form 4 dated 2024-03-11 but accepted after midnight
# UTC, and a Form D accepted the previous Friday evening.
_FORM_4 = "0001225208-24-004041"
_FORM_4_ACCEPTED = dt.datetime(2024, 3, 12, 0, 13, 8, tzinfo=dt.UTC)
_JOINT_FORM_4 = "0001193805-24-000360"
_FORM_D = "0000950172-24-000037"
_FORM_D_ACCEPTED = dt.datetime(2024, 3, 8, 22, 59, 9, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
async def _clean_edgar_tables() -> AsyncIterator[None]:
    """Truncate the EDGAR fact tables after every test in this module.

    The shared integration fixture truncates the Phase 2 tables; these are new
    in migration 0006 and have no foreign key to them, so CASCADE does not
    reach them. Run on the unguarded migration engine for the same reason the
    shared fixture does: TRUNCATE names a fact table in textual SQL, which the
    Core guard refuses on every guarded engine by design.
    """
    yield
    engine = _migration_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text("TRUNCATE TABLE edgar_filing, edgar_filing_document"))
    finally:
        await engine.dispose()


def _connector(
    *,
    start_date: dt.date = _MARCH_11,
    end_date: dt.date = _MARCH_12,
    max_index_days_per_run: int | None = 5,
    transport: httpx.MockTransport | None = None,
    form_types: frozenset[str] = _CAPTURED_FORMS,
) -> SecEdgarConnector:
    """Build a connector on the captured responses with a real wall clock.

    The clock is real on purpose: ``run()`` stamps the ingestion-run record
    from it, and a fabricated 2024 clock would write a run that ended before it
    started.
    """
    return SecEdgarConnector(
        client=EdgarClient(
            user_agent=_USER_AGENT,
            transport=transport if transport is not None else edgar_transport(),
        ),
        user_agent=_USER_AGENT,
        form_types=form_types,
        start_date=start_date,
        end_date=end_date,
        max_index_days_per_run=max_index_days_per_run,
    )


async def _accessions_at(as_of_ts: dt.datetime) -> set[str]:
    """Return the accession numbers visible through the as-of read path."""
    async with as_of(as_of_ts) as session:
        return set((await session.scalars(sa.select(EdgarFiling.accession_number))).all())


async def _filing_keys(as_of_ts: dt.datetime) -> set[tuple[str, int]]:
    """Return the (accession, CIK) logical keys visible through the as-of read path."""
    async with as_of(as_of_ts) as session:
        rows = (
            await session.execute(sa.select(EdgarFiling.accession_number, EdgarFiling.cik))
        ).all()
    return {(accession, cik) for accession, cik in rows}


async def _unguarded_scalar(query: str) -> object:
    """Run one catalog query beneath the Core guard and return a scalar."""
    engine = _migration_engine()
    try:
        async with engine.connect() as connection:
            return (await connection.execute(sa.text(query))).scalar()
    finally:
        await engine.dispose()


# --- the write path and the as-of boundary ----------------------------------


async def test_run_writes_filings_and_documents_readable_through_as_of() -> None:
    """A full run's rows are durable and readable only through the sanctioned path."""
    result = await _connector().run(RunKind.BACKFILL)
    assert result.rows_written > 0
    assert result.checkpoint_after == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-12"}

    now = dt.datetime.now(dt.UTC)
    async with as_of(now) as session:
        filings = list((await session.scalars(sa.select(EdgarFiling))).all())
        documents = list((await session.scalars(sa.select(EdgarFilingDocument))).all())
    assert {filing.accession_number for filing in filings} == {
        "0000950170-24-012183",
        "0000950170-24-029225",
        "0000950170-24-030003",
        "0001214659-24-004416",
        _JOINT_FORM_4,
        _FORM_4,
        _FORM_D,
    }
    # The joint Form 4 is one submission listed under seven CIKs: seven filing
    # rows, one document set, one acceptance instant.
    joint = [f for f in filings if f.accession_number == _JOINT_FORM_4]
    assert len({filing.cik for filing in joint}) == 7
    assert len({filing.knowledge_time for filing in joint}) == 1
    assert (
        len([d for d in documents if d.accession_number == _JOINT_FORM_4])
        == joint[0].document_count
    )
    assert len(filings) + len(documents) == result.rows_written
    assert all(filing.document_count > 0 for filing in filings)
    assert {document.accession_number for document in documents} <= {
        filing.accession_number for filing in filings
    }
    # The idempotent Core INSERT must still produce D-011's row shape: an
    # open-ended event interval stored as PG 'infinity' (surfacing as the aware
    # sentinel), the audit column stamped by the server default, and the
    # retraction flag defaulted false. A write path that bypassed the column
    # types or the server defaults would show up right here.
    written: list[BitemporalMixin] = [*filings, *documents]
    for row in written:
        assert row.valid_to == INFINITY
        assert row.is_retraction is False
        assert row.ingested_at.tzinfo is not None
        assert row.valid_from.tzinfo is not None
        assert row.knowledge_time.tzinfo is not None


async def test_knowledge_time_boundary_holds_at_the_acceptance_instant() -> None:
    """Invisible strictly before acceptance, visible from acceptance onward (I1).

    The boundary is inclusive at ``knowledge_time`` (D-011 read semantics), so
    an as-of exactly at the acceptance instant sees the filing and one
    microsecond earlier does not.
    """
    await _connector().run(RunKind.BACKFILL)
    assert _FORM_4 not in await _accessions_at(_FORM_4_ACCEPTED - dt.timedelta(microseconds=1))
    assert _FORM_4 in await _accessions_at(_FORM_4_ACCEPTED)


async def test_a_filing_is_invisible_on_its_own_filing_date() -> None:
    """The lookahead this connector exists to prevent, asserted end to end.

    Form 4 ``0001225208-24-004041`` carries filing date 2024-03-11 and was
    accepted 2024-03-12T00:13:08Z. Every instant of its EDGAR filing date is
    before it existed, so no as-of within that date may return it. A connector
    that used the filing date as ``knowledge_time`` would fail exactly here.
    """
    await _connector().run(RunKind.BACKFILL)
    end_of_filing_date = dt.datetime(2024, 3, 11, 23, 59, 59, tzinfo=dt.UTC)
    assert _FORM_4 not in await _accessions_at(end_of_filing_date)
    # ...while the Form D, dated the same 2024-03-11, was already knowable the
    # previous Friday evening. Filing date orders them identically; acceptance
    # does not.
    assert _FORM_D in await _accessions_at(end_of_filing_date)
    assert _FORM_D in await _accessions_at(_FORM_D_ACCEPTED)
    assert _FORM_D not in await _accessions_at(_FORM_D_ACCEPTED - dt.timedelta(seconds=1))


async def test_documents_share_their_filings_knowledge_time_boundary() -> None:
    """A manifest never becomes visible before the submission that contains it."""
    await _connector().run(RunKind.BACKFILL)
    async with as_of(_FORM_4_ACCEPTED - dt.timedelta(microseconds=1)) as session:
        before = (
            await session.scalars(
                sa.select(EdgarFilingDocument.accession_number).where(
                    EdgarFilingDocument.accession_number == _FORM_4
                )
            )
        ).all()
    async with as_of(_FORM_4_ACCEPTED) as session:
        at = (
            await session.scalars(
                sa.select(EdgarFilingDocument.accession_number).where(
                    EdgarFilingDocument.accession_number == _FORM_4
                )
            )
        ).all()
    assert list(before) == []
    assert list(at) == [_FORM_4]


# --- resumption and idempotence ---------------------------------------------


async def test_checkpoint_carries_across_runs_and_skips_completed_days() -> None:
    """Run one consumes one index day; run two resumes at the next."""
    first = await _connector(
        start_date=_MARCH_08,
        end_date=_MARCH_12,
        max_index_days_per_run=1,
        form_types=_MULTI_DAY_FORMS,
    ).run(RunKind.BACKFILL)
    assert first.checkpoint_before is None
    assert first.checkpoint_after == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-08"}
    assert await latest_checkpoint("sec_edgar") == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-08"}

    log: list[str] = []
    second = await _connector(
        start_date=_MARCH_08,
        end_date=_MARCH_12,
        max_index_days_per_run=1,
        transport=edgar_transport(request_log=log),
        form_types=_MULTI_DAY_FORMS,
    ).run(RunKind.BACKFILL)
    assert second.checkpoint_before == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-08"}
    assert second.checkpoint_after == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-11"}
    assert not any("master.20240308.idx" in url for url in log)


async def test_an_interrupted_run_neither_loses_nor_duplicates_filings() -> None:
    """Re-writing a day whose checkpoint was lost is a no-op, not a collision.

    The framework records ``checkpoint_after`` on the run record only after the
    batch commits, so a process killed between the commit and the record leaves
    the day written and unrecorded, and the next run re-reads it. That is
    simulated exactly: the second pass is handed ``None`` as its resume
    position — the state a lost checkpoint leaves — and re-writes the identical
    rows. It must neither raise on the append-only primary key nor leave a
    second copy.
    """
    first = await _connector(start_date=_MARCH_11, end_date=_MARCH_11).run(RunKind.BACKFILL)
    before = await _filing_keys(dt.datetime.now(dt.UTC))
    assert before

    connector = _connector(start_date=_MARCH_11, end_date=_MARCH_11)
    rewritten = 0
    async for batch in connector.fetch_batches(None):
        await connector._write_batch(batch.rows)
        rewritten += len(batch.rows)

    async with as_of(dt.datetime.now(dt.UTC)) as session:
        keys = list(
            (await session.execute(sa.select(EdgarFiling.accession_number, EdgarFiling.cik))).all()
        )

    assert rewritten == first.rows_written
    assert set(keys) == before
    # One row per logical key, not one per write: the second pass inserted the
    # same rows again and left nothing behind.
    assert len(keys) == len(set(keys))


async def test_a_failed_run_keeps_the_days_it_completed() -> None:
    """A run that dies part-way keeps its committed days and records where it got to."""
    transport = edgar_transport(available_index_dates=("20240308", "20240311"))
    connector = _connector(
        start_date=_MARCH_08,
        end_date=_MARCH_12,
        max_index_days_per_run=5,
        transport=transport,
        form_types=_MULTI_DAY_FORMS,
    )
    # 403 is what EDGAR answers for an archive key it does not have, and it is
    # classified permanent: retrying an identical request cannot help, and a
    # retry loop would spend the fair-access budget hiding the defect.
    with pytest.raises(PermanentSourceError, match="403"):
        await connector.run(RunKind.BACKFILL)
    assert await latest_checkpoint("sec_edgar") == {CHECKPOINT_LAST_INDEX_DATE: "2024-03-11"}
    assert "0000950170-24-029225" in await _accessions_at(dt.datetime.now(dt.UTC))


# --- D-011 live-run knowledge-time lag check --------------------------------


async def test_live_run_lag_check_flags_historical_knowledge_times() -> None:
    """A live run writing 2024 acceptance instants is flagged, and the data is kept.

    This is D-011's compensating control: the run still succeeds (a lagging
    feed is a source problem, not a corrupt row) and the breach is recorded as
    a measurement the P3.9 report reads back.
    """
    result = await _connector(start_date=_MARCH_11, end_date=_MARCH_11).run(RunKind.LIVE)
    assert result.lag_report is not None
    assert result.lag_report.breached is True
    assert result.lag_report.observed_max_lag is not None
    assert result.rows_written > 0

    report = await build_edgar_report()
    assert report.knowledge_time_lag is not None
    assert report.knowledge_time_lag.breached is True
    assert report.knowledge_time_lag.run_id == result.run_id
    assert report.knowledge_time_lag.declared_max_lag == dt.timedelta(hours=36)


# --- P3.9 report -------------------------------------------------------------


async def test_report_on_an_empty_store_measures_nothing_and_says_so() -> None:
    """No data means ``None``, never zero. A "0 days stale" tile would be a lie."""
    report = await build_edgar_report()
    assert report.coverage.first_index_date is None
    assert report.coverage.last_index_date is None
    assert report.coverage.weekdays_in_window is None
    assert report.coverage.filings == 0
    assert report.gaps == ()
    assert report.staleness.newest_index_date_age is None
    assert report.staleness.newest_knowledge_time_age is None
    assert report.staleness.last_successful_run_age is None
    assert report.knowledge_time_lag is None
    assert report.runs == ()


async def test_report_measures_coverage_staleness_and_run_history() -> None:
    """Every populated figure is something the report actually observed."""
    result = await _connector(
        start_date=_MARCH_08, end_date=_MARCH_12, form_types=_MULTI_DAY_FORMS
    ).run(RunKind.BACKFILL)
    report = await build_edgar_report()

    assert report.source == "sec_edgar"
    assert report.coverage.first_index_date == _MARCH_11
    assert report.coverage.last_index_date == _MARCH_12
    assert report.coverage.index_dates_present == 2
    assert report.coverage.weekdays_in_window == 2
    assert report.coverage.filings == 5
    assert report.coverage.documents > report.coverage.filings

    assert report.staleness.newest_index_date == _MARCH_12
    assert report.staleness.newest_index_date_age is not None
    assert report.staleness.newest_knowledge_time is not None
    assert report.staleness.last_successful_run_age is not None
    assert report.staleness.consecutive_failed_runs_since_success == 0

    assert [run.run_id for run in report.runs] == [result.run_id]
    persisted = report.runs[0].quality_metrics
    assert persisted["index_days_consumed"]["value"] == 3
    assert persisted["index_entries_selected"]["value"] == 5
    assert persisted["filing_headers_fetched"]["value"] == 5
    assert persisted["rows_written"]["unit"] == "rows"


async def test_report_displays_the_connectors_knowledge_time_policy() -> None:
    """D-011 requires the derivation to be visible beside the numbers."""
    report = await build_edgar_report()
    assert "acceptance" in report.knowledge_time_policy.description.lower()
    assert "US/Eastern" in report.knowledge_time_policy.description
    assert report.knowledge_time_policy.max_live_lag == dt.timedelta(hours=36)


async def test_report_counts_failed_runs_since_the_last_success() -> None:
    """A run of failures after a success is a measurement, and zero is one too."""
    await _connector(start_date=_MARCH_11, end_date=_MARCH_11).run(RunKind.BACKFILL)
    broken = _connector(
        start_date=_MARCH_12,
        end_date=_MARCH_12,
        transport=edgar_transport(available_index_dates=()),
    )
    with pytest.raises(PermanentSourceError):
        await broken.run(RunKind.BACKFILL)
    report = await build_edgar_report()
    assert report.staleness.consecutive_failed_runs_since_success == 1
    assert report.runs[0].error_detail is not None


def test_gap_detection_reports_weekday_holes_only() -> None:
    """Weekends are not gaps: EDGAR publishes no daily index on them."""
    covered = [dt.date(2024, 3, 8), dt.date(2024, 3, 14)]
    gaps = detect_gaps(covered)
    assert [(gap.first_date, gap.last_date, gap.weekdays) for gap in gaps] == [
        (dt.date(2024, 3, 11), dt.date(2024, 3, 13), 3)
    ]
    assert detect_gaps([dt.date(2024, 3, 8)]) == ()
    assert detect_gaps([]) == ()


async def test_a_fully_consumed_window_reports_no_gaps() -> None:
    """The captured window is 2024-03-11 to 2024-03-12, contiguous, so no gap exists.

    Stated as an assertion because the interesting failure is the *false*
    positive: a gap detector that counted weekends, or that treated the
    2024-03-08 index (from which nothing was selected) as a hole, would report
    one here.
    """
    await _connector(start_date=_MARCH_08, end_date=_MARCH_12, form_types=_MULTI_DAY_FORMS).run(
        RunKind.BACKFILL
    )
    report = await build_edgar_report()
    assert report.coverage.first_index_date == _MARCH_11
    assert report.coverage.last_index_date == _MARCH_12
    assert report.gaps == ()


# --- schema: what migration 0006 installed ----------------------------------


async def test_edgar_filing_is_a_hypertable_partitioned_on_valid_from() -> None:
    """Acceptance instant is the partition key, with the same 1-month chunks as price_bar."""
    dimension = await _unguarded_scalar(
        "SELECT column_name || '|' || time_interval::text "
        "FROM timescaledb_information.dimensions WHERE hypertable_name = 'edgar_filing'"
    )
    assert dimension == "valid_from|30 days"


async def test_edgar_filing_document_is_not_a_hypertable() -> None:
    """Its access pattern is a point lookup by accession; chunk pruning would not bite."""
    hypertables = await _unguarded_scalar(
        "SELECT array_agg(hypertable_name) FROM timescaledb_information.hypertables"
    )
    assert isinstance(hypertables, list)
    assert "edgar_filing" in hypertables
    assert "edgar_filing_document" not in hypertables


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("edgar_filing", "(accession_number, cik, valid_from, knowledge_time DESC)"),
        (
            "edgar_filing_document",
            "(accession_number, document_sequence, valid_from, knowledge_time DESC)",
        ),
    ],
)
async def test_asof_composite_index_exists(table: str, columns: str) -> None:
    """The index matches the DISTINCT ON / ORDER BY shape the query layer emits."""
    definition = await _unguarded_scalar(
        # S608: `table` is a fixed parametrize literal, not external input.
        f"SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_{table}_asof_lookup'"  # noqa: S608
    )
    assert definition is not None
    assert columns in str(definition)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE edgar_filing SET form_type = 'EVIL'",
        "DELETE FROM edgar_filing",
        "UPDATE edgar_filing_document SET filename = 'evil.htm'",
        "DELETE FROM edgar_filing_document",
    ],
)
async def test_append_only_triggers_reject_mutation(statement: str) -> None:
    """Corrections are new rows with a later knowledge_time; nothing is ever rewritten."""
    await _connector(start_date=_MARCH_11, end_date=_MARCH_11).run(RunKind.BACKFILL)
    engine = _migration_engine()
    try:
        with pytest.raises(DBAPIError, match="append-only"):
            async with engine.begin() as connection:
                await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


async def test_knowledge_time_has_no_database_default() -> None:
    """A writer that omits knowledge_time is refused by the database (D-011)."""
    async with ingest_writer_session() as session:
        incomplete = EdgarFiling(
            accession_number="0000000000-24-000000",
            cik=1,
            company_name="X",
            form_type="8-K",
            filing_date=_MARCH_11,
            index_date=_MARCH_11,
            period_of_report=None,
            declared_document_count=None,
            document_count=0,
            source_url="https://www.sec.gov/",
            valid_from=dt.datetime(2024, 3, 11, tzinfo=dt.UTC),
        )
        session.add(incomplete)
        with pytest.raises(sa.exc.IntegrityError, match="knowledge_time"):
            await session.flush()


# --- CC.8 connector contract -------------------------------------------------


class TestEdgarConnectorContract(ConnectorContractTests):
    """CC.8: an unavailable EDGAR raises; it never becomes data."""

    expected_error: ClassVar[type[TransientSourceError]] = TransientSourceError

    def build_connector(self) -> Connector:
        """Return an EDGAR connector whose transport cannot reach the source."""

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("name resolution failed", request=request)

        clock = NonBlockingClock()
        return SecEdgarConnector(
            # The limiter's clock must be the one the sleep advances, and it
            # must advance generously — see _NonBlockingClock.
            ConnectorRuntime(
                sleep=clock.sleep, monotonic=clock.monotonic, jitter=lambda ceiling: ceiling
            ),
            client=EdgarClient(user_agent=_USER_AGENT, transport=httpx.MockTransport(refuse)),
            user_agent=_USER_AGENT,
            start_date=_MARCH_11,
            end_date=_MARCH_11,
        )

    async def test_no_edgar_rows_survive_an_unavailable_source(self) -> None:
        """The EDGAR tables specifically are untouched, and the run is recorded failed."""
        connector = self.build_connector()
        with pytest.raises(self.expected_error):
            await connector.run(RunKind.LIVE)
        assert await _accessions_at(dt.datetime.now(dt.UTC)) == set()
        report = await build_edgar_report()
        assert report.runs[0].status == RunStatus.FAILED.value
        assert report.coverage.filings == 0
