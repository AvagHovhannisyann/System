"""P3.8 unit tests: key refusal, vintage selection, resumption, secret hygiene.

These drive :meth:`~backend.ingest.fred.connector.FredConnector.fetch_batches`
directly against the test payloads, so they exercise the connector's own logic
— which vintages it writes, which it defers, what it builds — without a
database. The write path, the as-of boundary and run bookkeeping are exercised
against real Postgres in ``backend/tests/integration/test_fred_ingestion.py``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import pytest

from backend.db.bitemporal import INFINITY
from backend.db.models import MacroObservation, MacroSeries
from backend.ingest.base import ConnectorRuntime
from backend.ingest.errors import PermanentSourceError, TransientSourceError
from backend.ingest.fred.client import (
    FRED_API_KEY_ENV,
    FredApiKeyNotConfiguredError,
    FredClient,
    redacted_url,
    resolve_fred_api_key,
)
from backend.ingest.fred.connector import (
    CHECKPOINT_VINTAGE_PREFIX,
    DEFAULT_SERIES_IDS,
    FRED_SOURCE,
    FredConnector,
)
from backend.ingest.fred.parse import (
    SERIES_DEFINITION_VALID_FROM,
    vintage_date_to_knowledge_time,
)
from backend.ingest.registry import connector_class
from backend.tests.ingest.test_fred_fixtures import (
    TEST_API_KEY,
    NonBlockingClock,
    fred_transport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from backend.ingest.base import Batch
    from backend.ingest.checkpoint import Checkpoint

# Well after every vintage in the test payloads, so nothing is deferred by the
# not-yet-knowable rule unless a test asks for that explicitly.
_NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)


def _runtime(now: dt.datetime = _NOW) -> ConnectorRuntime:
    """Return a runtime with a fixed wall clock and a non-blocking virtual clock."""
    clock = NonBlockingClock()
    return ConnectorRuntime(
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda ceiling: ceiling,
        now=lambda: now,
    )


def _connector(
    *,
    series_ids: tuple[str, ...] = ("REVISEDTWICE",),
    transport: httpx.MockTransport | None = None,
    runtime: ConnectorRuntime | None = None,
    max_series_per_run: int | None = 25,
) -> FredConnector:
    """Build a connector wired to the FRED test payloads."""
    return FredConnector(
        runtime if runtime is not None else _runtime(),
        client=FredClient(
            api_key=TEST_API_KEY,
            transport=transport if transport is not None else fred_transport(),
        ),
        api_key=TEST_API_KEY,
        series_ids=series_ids,
        max_series_per_run=max_series_per_run,
    )


async def _collect(connector: FredConnector, checkpoint: Checkpoint | None = None) -> list[Batch]:
    """Drain ``fetch_batches`` into a list."""
    return [batch async for batch in connector.fetch_batches(checkpoint)]


def _observations(batches: Iterable[Batch]) -> list[MacroObservation]:
    """Return every MacroObservation row across batches, in build order."""
    return [row for batch in batches for row in batch.rows if isinstance(row, MacroObservation)]


def _series_rows(batches: Iterable[Batch]) -> list[MacroSeries]:
    """Return every MacroSeries row across batches, in build order."""
    return [row for batch in batches for row in batch.rows if isinstance(row, MacroSeries)]


# --- fail-closed without a key ----------------------------------------------


def test_connector_refuses_to_exist_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No FRED_API_KEY, no connector — and no keyless or degraded fallback.

    FRED requires a key on every endpoint, so there is nothing to degrade to.
    Refusal is at construction, before any request, so a misconfigured
    deployment fails where it is obvious.
    """
    monkeypatch.delenv(FRED_API_KEY_ENV, raising=False)
    with pytest.raises(FredApiKeyNotConfiguredError, match=FRED_API_KEY_ENV):
        FredConnector()


@pytest.mark.parametrize("configured", ["", "   ", "\t\n"])
def test_a_blank_key_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
    """A blank key produces the same HTTP 400 as none, so it is refused here."""
    monkeypatch.setenv(FRED_API_KEY_ENV, configured)
    with pytest.raises(FredApiKeyNotConfiguredError):
        FredConnector()


def test_the_refusal_message_names_the_variable_and_the_reason() -> None:
    """An operator must learn what to set and why there is no default."""
    with pytest.raises(FredApiKeyNotConfiguredError) as raised:
        resolve_fred_api_key({})
    message = str(raised.value)
    assert FRED_API_KEY_ENV in message
    assert "no keyless tier" in message
    assert "fred.stlouisfed.org/docs/api/api_key.html" in message


def test_a_configured_key_is_accepted_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a key present the resolver returns it, stripped."""
    monkeypatch.setenv(FRED_API_KEY_ENV, f"  {TEST_API_KEY}  ")
    assert resolve_fred_api_key() == TEST_API_KEY


def test_the_client_refuses_a_blank_key_directly() -> None:
    """Defence in depth: the client will not be built around an empty credential."""
    with pytest.raises(ValueError, match="non-empty FRED API key"):
        FredClient(api_key="   ")


# --- secret isolation (I5) ---------------------------------------------------


def test_the_api_key_never_appears_in_a_rendered_url() -> None:
    """Rendered URLs omit the key entirely rather than masking it."""
    rendered = redacted_url("series/observations", {"series_id": "GDPC1", "api_key": TEST_API_KEY})
    assert TEST_API_KEY not in rendered
    assert "api_key" not in rendered
    assert "series_id=GDPC1" in rendered


async def test_the_api_key_never_appears_in_a_stored_source_url() -> None:
    """The provenance column is written to the database, so it must be clean."""
    batches = await _collect(_connector())
    for row in _series_rows(batches):
        assert TEST_API_KEY not in row.source_url
        assert "api_key" not in row.source_url
        assert row.source_url.startswith("https://api.stlouisfed.org/fred/series?")


async def test_the_api_key_never_appears_in_an_error_message() -> None:
    """Error text reaches logs; a key in it would be a permanent leak."""
    connector = _connector(transport=fred_transport(status=500))
    with pytest.raises(TransientSourceError) as raised:
        await _collect(connector)
    assert TEST_API_KEY not in str(raised.value)
    assert "api_key" not in str(raised.value)


async def test_the_api_key_never_appears_in_a_transport_failure_message() -> None:
    """An httpx error can embed the request URL; only the exception type is reported."""
    connector = _connector(transport=fred_transport(unavailable=True))
    with pytest.raises(TransientSourceError) as raised:
        await _collect(connector)
    assert TEST_API_KEY not in str(raised.value)


async def test_the_key_is_actually_sent_on_the_wire() -> None:
    """Redaction must not have removed the credential from the request itself."""
    log: list[httpx.URL] = []
    await _collect(_connector(transport=fred_transport(request_log=log)))
    assert log
    assert all(request.params.get("api_key") == TEST_API_KEY for request in log)
    assert all(request.params.get("file_type") == "json" for request in log)


# --- the revision case ------------------------------------------------------


async def test_every_vintage_becomes_its_own_row() -> None:
    """The core of P3.8: revisions are extra rows, never overwrites."""
    batches = await _collect(_connector())
    first_quarter = [
        row for row in _observations(batches) if row.observation_date == dt.date(2024, 1, 1)
    ]
    assert len(first_quarter) == 3
    assert [row.value for row in first_quarter] == [
        Decimal("1.1"),
        Decimal("2.2"),
        Decimal("3.3"),
    ]
    # Same event time, three distinct knowledge times: exactly D-011's shape.
    assert {row.valid_from for row in first_quarter} == {dt.datetime(2024, 1, 1, tzinfo=dt.UTC)}
    assert len({row.knowledge_time for row in first_quarter}) == 3


async def test_knowledge_times_are_derived_from_vintages_not_observation_dates() -> None:
    """The two axes must not be conflated; this asserts the actual values."""
    batches = await _collect(_connector())
    by_value = {row.value: row for row in _observations(batches)}
    assert by_value[Decimal("1.1")].knowledge_time == vintage_date_to_knowledge_time(
        dt.date(2024, 4, 25)
    )
    assert by_value[Decimal("3.3")].knowledge_time == vintage_date_to_knowledge_time(
        dt.date(2024, 6, 27)
    )
    # And every knowledge time is far later than the event time it describes,
    # because a quarter's number is published months after the quarter.
    for row in _observations(batches):
        assert row.knowledge_time > row.valid_from


async def test_observations_are_open_ended_in_event_time() -> None:
    """A reading about a period stays true; the period label lives in a column."""
    batches = await _collect(_connector())
    for row in _observations(batches):
        assert row.valid_to == INFINITY
        assert row.valid_from == dt.datetime.combine(
            row.observation_date, dt.time.min, tzinfo=dt.UTC
        )


async def test_the_open_vintage_end_is_stored_as_null() -> None:
    """No year-9999 sentinel reaches the column."""
    batches = await _collect(_connector())
    rows = _observations(batches)
    assert any(row.vintage_end_date is None for row in rows)
    assert all(row.vintage_end_date != dt.date(9999, 12, 31) for row in rows)


async def test_series_metadata_versions_ride_in_the_same_batch() -> None:
    """An observation is never committed without the units it must be read with."""
    batches = await _collect(_connector())
    assert len(batches) == 1
    assert _series_rows(batches)
    assert _observations(batches)
    series_rows = _series_rows(batches)
    for row in series_rows:
        assert row.units
        assert row.valid_to == INFINITY
        # Every metadata version shares ONE event-time anchor and differs only in
        # knowledge_time — that is what makes republished metadata a *correction*
        # to one fact rather than a second, competing fact. Asserting these two
        # were equal would collapse the two temporal axes D-011 keeps apart, and
        # would pass just as happily if the connector stamped event time from the
        # vintage.
        assert row.valid_from == SERIES_DEFINITION_VALID_FROM
        assert row.knowledge_time > row.valid_from
    # The invariant the anchor exists to provide: versions of one series are
    # versions, not rivals. Same logical key + same valid_from, distinct
    # knowledge_times, so latest-knowledge-wins leaves exactly one visible.
    by_series: dict[str, list[dt.datetime]] = {}
    for row in series_rows:
        by_series.setdefault(row.series_id, []).append(row.knowledge_time)
    for knowledge_times in by_series.values():
        assert len(set(knowledge_times)) == len(knowledge_times)


# --- missing values ----------------------------------------------------------


async def test_a_missing_value_is_written_as_null_and_flagged() -> None:
    """FRED's '.' becomes NULL + is_missing, never 0 and never a dropped row."""
    batches = await _collect(_connector(series_ids=("WITHMISSING",)))
    rows = {row.observation_date: row for row in _observations(batches)}
    assert len(rows) == 3
    missing = rows[dt.date(2024, 1, 2)]
    assert missing.is_missing is True
    assert missing.value is None


async def test_a_real_zero_survives_as_a_real_zero() -> None:
    """The failure mode a lax parser creates: 0 and absent becoming the same row."""
    batches = await _collect(_connector(series_ids=("WITHMISSING",)))
    rows = {row.observation_date: row for row in _observations(batches)}
    zero = rows[dt.date(2024, 1, 3)]
    assert zero.is_missing is False
    assert zero.value == Decimal("0")


# --- knowledge-time policy: nothing is written before it is knowable ---------


async def test_a_vintage_whose_knowledge_instant_has_not_passed_is_deferred() -> None:
    """Today's vintage is not yet knowable under this connector's policy.

    Writing it would be refused by the write-path guard once the clock caught
    up, or would leak into every as_of after that instant. It is skipped and
    left for the next run instead.
    """
    # Just before the newest vintage (2024-07-25) becomes knowable.
    just_before = vintage_date_to_knowledge_time(dt.date(2024, 7, 25)) - dt.timedelta(minutes=1)
    connector = _connector(runtime=_runtime(just_before))
    batches = await _collect(connector)
    rows = _observations(batches)
    assert all(row.knowledge_time <= just_before for row in rows)
    assert dt.date(2024, 4, 1) not in {row.observation_date for row in rows}


async def test_a_deferred_vintage_does_not_advance_the_checkpoint() -> None:
    """Otherwise the next run would step over data it never wrote."""
    just_before = vintage_date_to_knowledge_time(dt.date(2024, 7, 25)) - dt.timedelta(minutes=1)
    batches = await _collect(_connector(runtime=_runtime(just_before)))
    watermark = batches[-1].checkpoint[f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE"]
    assert watermark == "2024-06-27"  # the newest vintage actually written


async def test_a_deferred_vintage_is_picked_up_once_it_becomes_knowable() -> None:
    """The deferral is a delay, not a permanent gap."""
    just_before = vintage_date_to_knowledge_time(dt.date(2024, 7, 25)) - dt.timedelta(minutes=1)
    first = await _collect(_connector(runtime=_runtime(just_before)))
    later = await _collect(_connector(runtime=_runtime(_NOW)), first[-1].checkpoint)
    assert {row.observation_date for row in _observations(later)} == {dt.date(2024, 4, 1)}


async def test_no_row_is_ever_built_with_a_future_knowledge_time() -> None:
    """Across a sweep of clocks, the invariant holds by construction."""
    for days in range(0, 400, 17):
        now = dt.datetime(2024, 4, 1, tzinfo=dt.UTC) + dt.timedelta(days=days)
        batches = await _collect(_connector(runtime=_runtime(now)))
        for batch in batches:
            for row in batch.rows:
                assert isinstance(row, MacroObservation | MacroSeries)
                assert row.knowledge_time <= now


# --- incremental resumption --------------------------------------------------


async def test_a_first_run_starts_at_the_beginning_of_history() -> None:
    """A missing checkpoint means "from the start", never "from now"."""
    log: list[httpx.URL] = []
    await _collect(_connector(transport=fred_transport(request_log=log)))
    assert all(request.params.get("realtime_start") == "1776-07-04" for request in log)
    assert all(request.params.get("realtime_end") == "9999-12-31" for request in log)


async def test_a_resumed_run_requests_from_its_watermark() -> None:
    """The next run asks FRED only for the window it has not consumed."""
    log: list[httpx.URL] = []
    connector = _connector(transport=fred_transport(request_log=log))
    await _collect(connector, {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "2024-05-30"})
    assert all(request.params.get("realtime_start") == "2024-05-30" for request in log)


async def test_already_known_vintages_are_not_rewritten() -> None:
    """FRED returns still-current old rows on every windowed request.

    Re-writing them would be a no-op against the primary key, but counting
    their decade-old knowledge times in the live-lag check would make that
    check permanent noise. They are filtered before reaching a batch.
    """
    batches = await _collect(
        _connector(), {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "2024-05-30"}
    )
    vintages = {row.vintage_start_date for row in _observations(batches)}
    assert vintages == {dt.date(2024, 6, 27), dt.date(2024, 7, 25)}


async def test_resumption_is_idempotent_at_the_watermark() -> None:
    """Re-running at the same checkpoint writes nothing new."""
    checkpoint = {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "2024-07-25"}
    batches = await _collect(_connector(), checkpoint)
    assert _observations(batches) == []
    assert batches[-1].checkpoint == checkpoint


async def test_the_checkpoint_is_cumulative_across_series() -> None:
    """A run that fails part-way must not lose committed series' progress."""
    batches = await _collect(_connector(series_ids=("REVISEDTWICE", "WITHMISSING")))
    assert len(batches) == 2
    # The second batch still carries the first series' watermark.
    assert batches[-1].checkpoint == {
        f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "2024-07-25",
        f"{CHECKPOINT_VINTAGE_PREFIX}WITHMISSING": "2024-02-15",
    }


async def test_a_checkpoint_for_an_unconfigured_series_is_carried_forward() -> None:
    """Narrowing the configured list must not silently discard progress."""
    batches = await _collect(
        _connector(series_ids=("WITHMISSING",)),
        {f"{CHECKPOINT_VINTAGE_PREFIX}RETIREDSERIES": "2024-01-01"},
    )
    assert batches[-1].checkpoint[f"{CHECKPOINT_VINTAGE_PREFIX}RETIREDSERIES"] == "2024-01-01"


@pytest.mark.parametrize(
    "checkpoint",
    [
        {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": 20240530},
        {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": "30/05/2024"},
        {f"{CHECKPOINT_VINTAGE_PREFIX}REVISEDTWICE": None},
    ],
)
async def test_an_unreadable_checkpoint_raises_rather_than_resetting(
    checkpoint: Checkpoint,
) -> None:
    """Resetting past a corrupt resume position would silently skip or re-ingest."""
    with pytest.raises(PermanentSourceError, match="vintage_through"):
        await _collect(_connector(), checkpoint)


# --- paging ------------------------------------------------------------------


async def test_paging_follows_the_declared_counters() -> None:
    """A three-row history split at limit=2 is read completely, in two requests."""
    log: list[httpx.URL] = []
    batches = await _collect(
        _connector(series_ids=("PAGEDSERIES",), transport=fred_transport(request_log=log))
    )
    rows = _observations(batches)
    assert [row.value for row in rows] == [Decimal("6.1"), Decimal("6.2"), Decimal("6.3")]
    observation_requests = [
        request for request in log if request.path.endswith("/series/observations")
    ]
    assert [request.params.get("offset") for request in observation_requests] == ["0", "2"]


# --- unavailable source (I3) -------------------------------------------------


async def test_an_unreachable_source_raises_rather_than_returning_data() -> None:
    """I3, at the connector level: no placeholder, no default, no empty success."""
    with pytest.raises(TransientSourceError):
        await _collect(_connector(transport=fred_transport(unavailable=True)))


@pytest.mark.parametrize(
    ("status", "expected"), [(429, TransientSourceError), (500, TransientSourceError)]
)
async def test_retryable_statuses_surface_as_transient(
    status: int, expected: type[Exception]
) -> None:
    """FRED documents 429 and 500; both are worth retrying and are classified so."""
    with pytest.raises(expected):
        await _collect(_connector(transport=fred_transport(status=status)))


@pytest.mark.parametrize("status", [400, 404, 423])
async def test_non_retryable_statuses_surface_as_permanent(status: int) -> None:
    """FRED's other documented statuses are permanent: retrying cannot help.

    Note 423 Locked, which FRED documents but the shared taxonomy has no
    special case for, so it lands in the permanent bucket. Fail-closed and
    loud is the right default; if FRED's 423 turns out to be temporary the
    taxonomy is where that belongs, not here.
    """
    with pytest.raises(PermanentSourceError):
        await _collect(_connector(transport=fred_transport(status=status)))


async def test_a_non_json_body_is_refused() -> None:
    """A body that is not the documented JSON object is a defect, not an empty page."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>", request=request)

    with pytest.raises(PermanentSourceError, match="not JSON"):
        await _collect(_connector(transport=httpx.MockTransport(handler)))


async def test_an_unknown_series_is_not_silently_skipped() -> None:
    """A series with no payload answers 404 and fails the run rather than vanishing."""
    with pytest.raises(PermanentSourceError):
        await _collect(_connector(series_ids=("NOTAFIXTURE",)))


# --- declaration and configuration -------------------------------------------


def test_the_connector_declares_the_p31_contract() -> None:
    """source_name, knowledge_time_policy and rate_limit, all present and specific."""
    assert FredConnector.source_name == FRED_SOURCE
    assert connector_class(FRED_SOURCE) is FredConnector
    policy = FredConnector.knowledge_time_policy
    assert "vintage" in policy.description.lower()
    assert "never the raw vintage date" in policy.description.lower()
    assert policy.max_live_lag > dt.timedelta(0)
    assert FredConnector.rate_limit.requests_per_second > 0


def test_the_default_series_list_is_non_empty_and_unique() -> None:
    """The shipped default must not contain a duplicate that eats its own progress."""
    assert DEFAULT_SERIES_IDS
    assert len(set(DEFAULT_SERIES_IDS)) == len(DEFAULT_SERIES_IDS)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"series_ids": ()}, "at least one FRED series"),
        ({"series_ids": ("GDPC1", "")}, "blank identifier"),
        ({"series_ids": ("GDPC1", "GDPC1")}, "must be unique"),
        ({"max_series_per_run": 0}, "must be >= 1"),
    ],
)
def test_invalid_configuration_is_refused(kwargs: dict[str, object], match: str) -> None:
    """A misconfigured connector fails at construction, not mid-run."""
    with pytest.raises(ValueError, match=match):
        FredConnector(api_key=TEST_API_KEY, **kwargs)  # type: ignore[arg-type]


async def test_the_per_run_series_cap_bounds_the_run() -> None:
    """A bounded run makes resumable progress instead of one unbounded execution."""
    batches = await _collect(
        _connector(series_ids=("REVISEDTWICE", "WITHMISSING"), max_series_per_run=1)
    )
    assert len(batches) == 1
    assert f"{CHECKPOINT_VINTAGE_PREFIX}WITHMISSING" not in batches[-1].checkpoint


async def test_every_request_is_paced_through_the_framework() -> None:
    """The declared rate limit is binding, not advisory.

    Proven by the limiter having been consulted: with burst=1 and a virtual
    clock, the second request must wait.
    """
    clock = NonBlockingClock()
    runtime = ConnectorRuntime(
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda ceiling: ceiling,
        now=lambda: _NOW,
    )
    connector = _connector(runtime=runtime)
    await _collect(connector)
    assert clock.sleeps
    assert connector.rate_limit_wait_s > 0


async def test_the_output_type_is_sent_explicitly() -> None:
    """Correctness rests on one row per (observation date, real-time period)."""
    log: list[httpx.URL] = []
    await _collect(_connector(transport=fred_transport(request_log=log)))
    observation_requests = [
        request for request in log if request.path.endswith("/series/observations")
    ]
    assert observation_requests
    assert all(request.params.get("output_type") == "1" for request in observation_requests)
