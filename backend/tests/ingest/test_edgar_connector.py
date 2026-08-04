"""P3.2 unit tests: fair-access enforcement, pacing, resumption, and failure.

These drive :meth:`~backend.ingest.edgar.connector.SecEdgarConnector.fetch_batches`
directly against the captured responses, so they exercise the connector's own
logic — which days it asks for, what it builds, how it paces — without a
database. The write path, the as-of boundary and run bookkeeping are exercised
against real Postgres in ``backend/tests/integration/test_edgar_ingestion.py``.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import TYPE_CHECKING

import httpx
import pytest

from backend.core.config import Settings, get_settings
from backend.db.bitemporal import INFINITY
from backend.db.models import EdgarFiling, EdgarFilingDocument
from backend.ingest.base import ConnectorRuntime
from backend.ingest.edgar.client import EdgarClient, SecUserAgentNotConfiguredError
from backend.ingest.edgar.connector import (
    CHECKPOINT_LAST_INDEX_DATE,
    DEFAULT_FORM_TYPES,
    SecEdgarConnector,
)
from backend.ingest.edgar.parse import EARLIEST_ISO_NAMED_DAILY_INDEX
from backend.ingest.errors import PermanentSourceError, TransientSourceError
from backend.ingest.ratelimit import RateLimit
from backend.ingest.registry import connector_class
from backend.tests.ingest.test_edgar_fixtures import NonBlockingClock, edgar_transport

if TYPE_CHECKING:
    from collections.abc import Iterable

    from backend.ingest.base import Batch
    from backend.ingest.checkpoint import Checkpoint

_USER_AGENT = "Quant Research Platform test-contact@example.com"
_ALL_CAPTURED_FORMS = frozenset({"8-K", "4", "D", "CORRESP"})
# The 2024-03-08 sample holds Form 4 rows whose headers are not captured, so
# multi-day tests select the forms that are: this is the same selection
# mechanism, over the days the fixtures cover.
_MULTI_DAY_FORMS = frozenset({"8-K", "D", "CORRESP"})
_MARCH_11 = dt.date(2024, 3, 11)
_MARCH_12 = dt.date(2024, 3, 12)


def _runtime(now: dt.datetime | None = None) -> ConnectorRuntime:
    """Return a runtime with a fixed wall clock and a non-blocking virtual clock."""
    instant = now if now is not None else dt.datetime(2024, 3, 13, 12, 0, tzinfo=dt.UTC)
    clock = NonBlockingClock()
    return ConnectorRuntime(
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda ceiling: ceiling,
        now=lambda: instant,
    )


def _connector(
    *,
    transport: httpx.MockTransport | None = None,
    form_types: frozenset[str] | None = _ALL_CAPTURED_FORMS,
    start_date: dt.date = _MARCH_11,
    end_date: dt.date | None = _MARCH_12,
    runtime: ConnectorRuntime | None = None,
    max_index_days_per_run: int | None = 5,
) -> SecEdgarConnector:
    """Build a connector wired to the captured EDGAR responses."""
    return SecEdgarConnector(
        runtime if runtime is not None else _runtime(),
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


async def _collect(
    connector: SecEdgarConnector, checkpoint: Checkpoint | None = None
) -> list[Batch]:
    """Drain ``fetch_batches`` into a list."""
    return [batch async for batch in connector.fetch_batches(checkpoint)]


def _filings(batches: Iterable[Batch]) -> dict[str, EdgarFiling]:
    """Index every EdgarFiling row across batches by accession number."""
    return {
        row.accession_number: row
        for batch in batches
        for row in batch.rows
        if isinstance(row, EdgarFiling)
    }


# --- SEC fair access: fail-closed -------------------------------------------


def test_connector_refuses_to_exist_without_a_configured_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SEC_USER_AGENT, no connector — not an anonymous or invented one.

    Refusal is at construction, before any request, so a misconfigured
    deployment fails where it is obvious rather than at the first outbound
    call.
    """
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    get_settings.cache_clear()
    with pytest.raises(SecUserAgentNotConfiguredError, match="SEC_USER_AGENT"):
        SecEdgarConnector()


@pytest.mark.parametrize("configured", ["", "   "])
def test_blank_contact_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    """A blank header is indistinguishable from sending none, so it is refused too."""
    monkeypatch.setenv("SEC_USER_AGENT", configured)
    get_settings.cache_clear()
    with pytest.raises(SecUserAgentNotConfiguredError):
        SecEdgarConnector()


def test_configured_contact_is_accepted_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a contact configured, construction succeeds and uses that value."""
    monkeypatch.setenv("SEC_USER_AGENT", _USER_AGENT)
    get_settings.cache_clear()
    assert Settings().sec_user_agent == _USER_AGENT
    SecEdgarConnector(client=EdgarClient(user_agent=_USER_AGENT, transport=edgar_transport()))


async def test_every_request_declares_the_configured_contact() -> None:
    """SEC's published sample headers are sent on every request, not just the first."""
    seen: list[httpx.Headers] = []
    captured = edgar_transport()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        response = captured.handler(request)
        assert isinstance(response, httpx.Response)
        return response

    connector = _connector(transport=httpx.MockTransport(handler))
    await _collect(connector)
    assert seen
    assert all(headers["user-agent"] == _USER_AGENT for headers in seen)
    assert all(headers["accept-encoding"] == "gzip, deflate" for headers in seen)


def test_client_rejects_a_blank_user_agent_directly() -> None:
    """The client refuses too, so no code path can construct an anonymous one."""
    with pytest.raises(ValueError, match="non-empty contact string"):
        EdgarClient(user_agent="  ")


# --- rate limiting -----------------------------------------------------------


def test_declared_rate_limit_is_secs_published_ceiling() -> None:
    """10 requests/second, and a burst of one so the ceiling is never exceeded.

    A bucket holding ten tokens would permit ten immediate requests plus ten
    refilled inside the same second — twenty in a one-second window. A burst of
    one paces at a strict 100 ms.
    """
    assert SecEdgarConnector.rate_limit == RateLimit(requests_per_second=10.0, burst=1)


async def test_requests_are_never_issued_faster_than_the_published_ceiling() -> None:
    """A real run of six requests cannot finish faster than 10 requests/second allows.

    Deliberately against the wall clock and the real scheduler: compliance is a
    statement about elapsed time, and a virtual clock would assert the
    connector's arithmetic rather than its behaviour. The bound is exact in the
    direction that matters — if the declared burst were the full rate, six
    requests would complete in milliseconds and this would fail.
    """
    log: list[str] = []
    connector = SecEdgarConnector(
        client=EdgarClient(user_agent=_USER_AGENT, transport=edgar_transport(request_log=log)),
        user_agent=_USER_AGENT,
        form_types=_ALL_CAPTURED_FORMS,
        start_date=_MARCH_11,
        end_date=_MARCH_11,
    )
    started = time.monotonic()
    await _collect(connector)
    elapsed = time.monotonic() - started

    # 1 quarter listing + 1 daily index + 4 selected filing headers.
    assert len(log) == 6
    minimum = (len(log) - 1) / SecEdgarConnector.rate_limit.requests_per_second
    assert elapsed >= minimum
    # The reported wait is the time spent *blocked on the limiter*, so it is
    # positive and cannot exceed the run's elapsed time — the bucket refills
    # while a request is in flight, which is why it is not simply `minimum`.
    assert 0 < connector.rate_limit_wait_s <= elapsed
    assert connector.transient_retries == 0


# --- knowledge time ----------------------------------------------------------


async def test_knowledge_time_and_event_time_are_the_acceptance_instant() -> None:
    """Both temporal columns carry the acceptance instant, in UTC and aware."""
    batches = await _collect(_connector())
    filings = _filings(batches)
    form_4 = filings["0001225208-24-004041"]
    assert form_4.knowledge_time == dt.datetime(2024, 3, 12, 0, 13, 8, tzinfo=dt.UTC)
    assert form_4.valid_from == form_4.knowledge_time
    assert form_4.acceptance_datetime == form_4.knowledge_time
    assert form_4.valid_to == INFINITY
    assert form_4.knowledge_time.tzinfo is not None


async def test_knowledge_time_is_never_derived_from_the_filing_date() -> None:
    """The two divergence cases keep their acceptance instants, in both directions."""
    filings = _filings(await _collect(_connector()))
    form_4 = filings["0001225208-24-004041"]
    assert form_4.filing_date == _MARCH_11
    assert form_4.knowledge_time > dt.datetime(2024, 3, 12, tzinfo=dt.UTC)

    form_d = filings["0000950172-24-000037"]
    assert form_d.filing_date == _MARCH_11
    assert form_d.knowledge_time == dt.datetime(2024, 3, 8, 22, 59, 9, tzinfo=dt.UTC)
    assert form_d.index_date == _MARCH_11


async def test_index_date_and_filing_date_are_stored_separately() -> None:
    """A CORRESP filed 2024-02-07 and disseminated 2024-03-11 keeps both dates."""
    filings = _filings(await _collect(_connector()))
    corresp = filings["0000950170-24-012183"]
    assert corresp.filing_date == dt.date(2024, 2, 7)
    assert corresp.index_date == _MARCH_11
    assert corresp.knowledge_time == dt.datetime(2024, 2, 7, 21, 5, 2, tzinfo=dt.UTC)


async def test_documents_ride_in_the_same_batch_as_their_filing() -> None:
    """A filing is never committable without its manifest."""
    batches = await _collect(_connector())
    for batch in batches:
        accessions = {row.accession_number for row in batch.rows if isinstance(row, EdgarFiling)}
        documents = [row for row in batch.rows if isinstance(row, EdgarFilingDocument)]
        assert {document.accession_number for document in documents} <= accessions
        for document in documents:
            assert document.knowledge_time == next(
                row.knowledge_time
                for row in batch.rows
                if isinstance(row, EdgarFiling)
                and row.accession_number == document.accession_number
            )


# --- incremental sync and resumption ----------------------------------------


async def test_one_accession_under_many_ciks_becomes_one_row_per_cik() -> None:
    """Seven filers, one header request, seven filing rows, one document set.

    Accession 0001193805-24-000360 is listed under seven CIKs in the real
    2024-03-12 index. Each is a fact the source states, so each gets a row;
    the submission behind them is one thing, so its header is fetched once and
    its documents are stored once.
    """
    log: list[str] = []
    connector = _connector(
        transport=edgar_transport(request_log=log), start_date=_MARCH_12, end_date=_MARCH_12
    )
    batches = await _collect(connector)
    rows = [row for batch in batches for row in batch.rows]
    joint = [
        row
        for row in rows
        if isinstance(row, EdgarFiling) and row.accession_number == "0001193805-24-000360"
    ]
    assert len(joint) == 7
    assert len({row.cik for row in joint}) == 7
    assert len({row.knowledge_time for row in joint}) == 1
    assert {row.company_name for row in joint} >= {"Flynn James E", "AdaptHealth Corp."}

    documents = [
        row
        for row in rows
        if isinstance(row, EdgarFilingDocument) and row.accession_number == "0001193805-24-000360"
    ]
    assert len(documents) == joint[0].document_count
    header_requests = [url for url in log if "0001193805-24-000360-index-headers" in url]
    assert len(header_requests) == 1

    metrics = {metric.name: metric for metric in connector._metrics(0, len(batches), None)}
    assert metrics["index_entries_selected"].value == 9
    assert metrics["filing_headers_fetched"].value == 3


async def test_checkpoint_marks_the_index_date_the_batch_completes() -> None:
    """One batch per index date, each checkpointing that date."""
    batches = await _collect(_connector(start_date=_MARCH_11, end_date=_MARCH_12))
    assert [batch.checkpoint[CHECKPOINT_LAST_INDEX_DATE] for batch in batches] == [
        "2024-03-11",
        "2024-03-12",
    ]


async def test_a_checkpointed_day_is_not_requested_again() -> None:
    """Resumption is proved by the requests that were *not* made.

    Asserting on the URLs requested is the only way to distinguish a connector
    that resumed from one that re-read everything and happened to write the
    same rows.
    """
    log: list[str] = []
    connector = _connector(transport=edgar_transport(request_log=log))
    batches = await _collect(connector, {CHECKPOINT_LAST_INDEX_DATE: "2024-03-11"})
    assert [batch.checkpoint[CHECKPOINT_LAST_INDEX_DATE] for batch in batches] == ["2024-03-12"]
    assert not any("master.20240311.idx" in url for url in log)
    assert any("master.20240312.idx" in url for url in log)


async def test_a_day_with_no_selected_filings_still_advances_the_checkpoint() -> None:
    """Otherwise the day would be re-fetched forever.

    The captured 2024-03-08 index holds only Form 4 rows, which the default
    form selection excludes.
    """
    connector = _connector(
        form_types=_MULTI_DAY_FORMS, start_date=dt.date(2024, 3, 8), end_date=_MARCH_12
    )
    batches = await _collect(connector)
    by_date = {batch.checkpoint[CHECKPOINT_LAST_INDEX_DATE]: batch for batch in batches}
    assert by_date["2024-03-08"].rows == ()
    assert by_date["2024-03-11"].rows != ()


async def test_per_run_cap_bounds_the_work_and_leaves_the_rest_resumable() -> None:
    """A capped run stops early; its last checkpoint is where the next one starts."""
    connector = _connector(
        form_types=_MULTI_DAY_FORMS,
        start_date=dt.date(2024, 3, 8),
        end_date=_MARCH_12,
        max_index_days_per_run=1,
    )
    batches = await _collect(connector)
    assert [batch.checkpoint[CHECKPOINT_LAST_INDEX_DATE] for batch in batches] == ["2024-03-08"]


async def test_first_run_starts_at_the_beginning_of_history_not_at_now() -> None:
    """A missing checkpoint means "from the beginning", per the framework contract.

    Starting at "now" would create a permanent, invisible hole in the history.
    """
    assert (
        SecEdgarConnector(
            client=EdgarClient(user_agent=_USER_AGENT, transport=edgar_transport()),
            user_agent=_USER_AGENT,
        )._start_date
        == EARLIEST_ISO_NAMED_DAILY_INDEX
    )


def test_start_date_before_edgars_readable_history_is_refused() -> None:
    """The pre-1998 naming schemes are out of scope and say so, loudly."""
    with pytest.raises(ValueError, match="precedes EDGAR"):
        SecEdgarConnector(
            client=EdgarClient(user_agent=_USER_AGENT, transport=edgar_transport()),
            user_agent=_USER_AGENT,
            start_date=dt.date(1996, 1, 2),
        )


@pytest.mark.parametrize(
    "checkpoint", [{"last_index_date": 20240311}, {"last_index_date": "11/03"}]
)
async def test_an_unreadable_checkpoint_raises_rather_than_resetting(
    checkpoint: dict[str, object],
) -> None:
    """Resetting past a broken resume position would silently skip or duplicate history."""
    with pytest.raises(PermanentSourceError, match="last_index_date"):
        await _collect(_connector(), checkpoint)  # type: ignore[arg-type]


# --- unavailable source ------------------------------------------------------


async def test_unreachable_source_raises_and_yields_nothing() -> None:
    """A transport failure becomes a TransientSourceError; no batch is produced."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed", request=request)

    connector = SecEdgarConnector(
        _runtime(),
        client=EdgarClient(user_agent=_USER_AGENT, transport=httpx.MockTransport(refuse)),
        user_agent=_USER_AGENT,
        start_date=_MARCH_11,
        end_date=_MARCH_11,
    )
    with pytest.raises(TransientSourceError, match="ConnectError"):
        await _collect(connector)


async def test_a_day_missing_from_the_listing_is_never_requested() -> None:
    """EDGAR answers a non-existent index with 403, so the connector must not guess.

    The transport serves only the dates given; a connector that constructed
    dates instead of reading the listing would hit the 403 branch and fail.
    """
    log: list[str] = []
    connector = _connector(
        transport=edgar_transport(available_index_dates=("20240311",), request_log=log),
        start_date=dt.date(2024, 3, 9),
        end_date=dt.date(2024, 3, 10),
    )
    assert await _collect(connector) == []
    assert not any(".idx" in url for url in log)


async def test_a_403_from_the_archive_is_a_permanent_failure() -> None:
    """403 is not retried: EDGAR uses it both for absent keys and access blocks."""
    connector = _connector(
        transport=edgar_transport(available_index_dates=()),
        start_date=_MARCH_11,
        end_date=_MARCH_11,
    )
    with pytest.raises(PermanentSourceError, match="403"):
        await _collect(connector)


# --- registration and measurement -------------------------------------------


def test_connector_is_registered_under_its_source_name() -> None:
    """The scheduler resolves 'sec_edgar' to this class."""
    assert connector_class("sec_edgar") is SecEdgarConnector


def test_default_form_selection_is_the_periodic_and_current_reports() -> None:
    """The default selection is the Phase 7 input set, and nothing broader.

    Pinned as a test because it is a scope decision with a request-budget
    consequence: EDGAR disseminates about 5,000 filings a day and each selected
    one costs its own header request.
    """
    expected = frozenset(
        {
            "10-K",
            "10-K/A",
            "10-Q",
            "10-Q/A",
            "8-K",
            "8-K/A",
            "20-F",
            "20-F/A",
            "40-F",
            "40-F/A",
            "6-K",
            "6-K/A",
        }
    )
    assert expected == DEFAULT_FORM_TYPES
    assert "4" not in DEFAULT_FORM_TYPES
    assert "CORRESP" not in DEFAULT_FORM_TYPES


async def test_metrics_report_selection_separately_from_coverage() -> None:
    """Filings listed and filings selected are distinct measurements.

    A single "filings" number would make a form filter indistinguishable from
    EDGAR having published nothing.
    """
    connector = _connector(start_date=_MARCH_11, end_date=_MARCH_11)
    batches = await _collect(connector)
    metrics = {metric.name: metric for metric in connector._metrics(0, len(batches), None)}
    assert metrics["index_rows_read"].value == 19
    assert metrics["index_duplicate_rows_collapsed"].value == 13
    assert metrics["index_entries_selected"].value == 4
    assert metrics["index_days_consumed"].value == 1
    assert metrics["index_rows_read"].unit == "rows"


async def test_acceptance_before_dissemination_is_measured_not_hidden() -> None:
    """The CORRESP case shows up as a number, so the caveat is visible in the report."""
    connector = _connector(start_date=_MARCH_11, end_date=_MARCH_11)
    await _collect(connector)
    metrics = {metric.name: metric for metric in connector._metrics(0, 1, None)}
    assert metrics["filings_accepted_before_index_date"].value == 2
    assert metrics["max_acceptance_to_index_lag"].value == 33
    assert metrics["max_acceptance_to_index_lag"].unit == "days"


async def test_no_divergence_reports_none_not_zero() -> None:
    """Absence of the phenomenon is None; a measured zero would be a different claim."""
    connector = _connector(start_date=_MARCH_12, end_date=_MARCH_12)
    await _collect(connector)
    metrics = {metric.name: metric for metric in connector._metrics(0, 1, None)}
    assert metrics["filings_accepted_before_index_date"].value == 0
    assert metrics["max_acceptance_to_index_lag"].value is None
