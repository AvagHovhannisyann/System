"""P3.8 unit tests: vintage-to-knowledge-time derivation and response parsing.

The derivation tests are the important ones. ``knowledge_time`` is the column
invariant I1 rests on, and for FRED it is *computed* rather than read — from a
date, under a policy this connector chose. A wrong policy would not fail
anywhere else: the rows would look perfectly ordinary and every backtest built
on them would be quietly optimistic.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.ingest.errors import PermanentSourceError
from backend.ingest.fred.parse import (
    EARLIEST_REALTIME_DATE,
    LATEST_REALTIME_DATE,
    MISSING_VALUE_MARKER,
    VINTAGE_TIMEZONE,
    observation_date_to_valid_from,
    parse_observations,
    parse_series_versions,
    vintage_date_to_knowledge_time,
)
from backend.tests.ingest.test_fred_fixtures import read_fixture

# --- knowledge_time: a documented conservative lag, never the raw date -------


@pytest.mark.parametrize(
    ("vintage", "expected"),
    [
        # Eastern Daylight Time (UTC-04:00): midnight ET on the following day.
        (dt.date(2024, 3, 11), dt.datetime(2024, 3, 12, 4, 0, tzinfo=dt.UTC)),
        # Eastern Standard Time (UTC-05:00).
        (dt.date(2024, 1, 15), dt.datetime(2024, 1, 16, 5, 0, tzinfo=dt.UTC)),
        # The day before US DST begins in 2024 (2024-03-10): the *following*
        # midnight is already EDT, so the offset must be 4 hours, not 5.
        (dt.date(2024, 3, 9), dt.datetime(2024, 3, 10, 5, 0, tzinfo=dt.UTC)),
        (dt.date(2024, 3, 10), dt.datetime(2024, 3, 11, 4, 0, tzinfo=dt.UTC)),
        # Month and year boundaries.
        (dt.date(2023, 12, 31), dt.datetime(2024, 1, 1, 5, 0, tzinfo=dt.UTC)),
    ],
)
def test_knowledge_time_is_midnight_eastern_after_the_vintage_date(
    vintage: dt.date, expected: dt.datetime
) -> None:
    """The derivation is the policy: end of the vintage day, US/Eastern, in UTC."""
    assert vintage_date_to_knowledge_time(vintage) == expected


def test_knowledge_time_is_never_the_raw_vintage_date() -> None:
    """D-011 forbids a date-only source's raw date as a knowledge time.

    Stated as its own property rather than left implied by the table above,
    because this is the rule being obeyed — the specific hour is just how.
    """
    for offset in range(400):
        vintage = dt.date(2024, 1, 1) + dt.timedelta(days=offset)
        knowledge_time = vintage_date_to_knowledge_time(vintage)
        assert knowledge_time.date() > vintage
        # And strictly after every moment of the vintage day in the market's
        # own timezone, which is the property that makes it conservative.
        end_of_vintage_day = dt.datetime.combine(
            vintage, dt.time.max, tzinfo=VINTAGE_TIMEZONE
        ).astimezone(dt.UTC)
        assert knowledge_time > end_of_vintage_day


def test_knowledge_time_is_after_the_us_close_on_the_vintage_day() -> None:
    """A number released the morning of day V is not tradable on day V.

    16:00 US/Eastern is the regular-session close; the derived instant must sit
    after it, or a strategy rebalancing at the close could act on a release the
    connector cannot prove was public.
    """
    vintage = dt.date(2024, 6, 12)
    close = dt.datetime.combine(vintage, dt.time(16, 0), tzinfo=VINTAGE_TIMEZONE)
    assert vintage_date_to_knowledge_time(vintage) > close.astimezone(dt.UTC)


def test_knowledge_time_is_monotonic_in_the_vintage_date() -> None:
    """A later vintage is never knowable earlier — including across DST shifts."""
    previous = vintage_date_to_knowledge_time(dt.date(2023, 1, 1))
    for offset in range(1, 800):
        current = vintage_date_to_knowledge_time(dt.date(2023, 1, 1) + dt.timedelta(days=offset))
        assert current > previous
        previous = current


def test_a_datetime_vintage_is_refused() -> None:
    """``datetime`` subclasses ``date``; accepting one would silently truncate."""
    with pytest.raises(TypeError, match="must be a date, not a datetime"):
        vintage_date_to_knowledge_time(dt.datetime(2024, 3, 11, 8, 30, tzinfo=dt.UTC))


def test_valid_from_is_the_observation_date_at_utc_midnight() -> None:
    """Event time is UTC-anchored; migration 0008 enforces the same equality."""
    assert observation_date_to_valid_from(dt.date(2024, 3, 11)) == dt.datetime(
        2024, 3, 11, tzinfo=dt.UTC
    )


def test_observation_date_and_vintage_date_are_different_axes() -> None:
    """The distinction the whole table exists for, asserted directly.

    One observation date carries several knowledge times (one per vintage), and
    one vintage date carries several observation dates. Conflating them is the
    single most likely way to get macro point-in-time wrong.
    """
    observation_date = dt.date(2024, 1, 1)
    early = vintage_date_to_knowledge_time(dt.date(2024, 4, 25))
    late = vintage_date_to_knowledge_time(dt.date(2024, 6, 27))
    assert observation_date_to_valid_from(observation_date) == dt.datetime(
        2024, 1, 1, tzinfo=dt.UTC
    )
    assert early < late
    # Both revisions describe the same event-time period.
    assert observation_date_to_valid_from(observation_date) < early


# --- observations: revisions, missing values, envelope -----------------------


def test_revision_history_parses_into_one_row_per_vintage() -> None:
    """Three vintages of one observation date stay three rows, not one."""
    page = parse_observations(read_fixture("observations-revised-twice.json"), series_id="X")
    first_quarter = [
        observation
        for observation in page.observations
        if observation.observation_date == dt.date(2024, 1, 1)
    ]
    assert len(first_quarter) == 3
    assert [observation.value for observation in first_quarter] == [
        Decimal("1.1"),
        Decimal("2.2"),
        Decimal("3.3"),
    ]
    assert [observation.vintage_start_date for observation in first_quarter] == [
        dt.date(2024, 4, 25),
        dt.date(2024, 5, 30),
        dt.date(2024, 6, 27),
    ]
    # Derived knowledge times are strictly increasing, which is what makes
    # latest-knowledge-wins return the original value before a revision.
    knowledge_times = [
        vintage_date_to_knowledge_time(observation.vintage_start_date)
        for observation in first_quarter
    ]
    assert knowledge_times == sorted(knowledge_times)
    assert len(set(knowledge_times)) == 3


def test_open_real_time_period_parses_to_none_not_year_9999() -> None:
    """A year-9999 sentinel must not leak into date arithmetic downstream."""
    page = parse_observations(read_fixture("observations-revised-twice.json"), series_id="X")
    open_rows = [row for row in page.observations if row.vintage_end_date is None]
    closed_rows = [row for row in page.observations if row.vintage_end_date is not None]
    assert len(open_rows) == 2
    assert all(row.vintage_end_date != LATEST_REALTIME_DATE for row in page.observations)
    assert closed_rows[0].vintage_end_date == dt.date(2024, 5, 29)


def test_missing_marker_becomes_none_and_is_flagged() -> None:
    """FRED's '.' is an explicit absence: kept as one, never zeroed or dropped."""
    page = parse_observations(read_fixture("observations-missing-value.json"), series_id="X")
    by_date = {row.observation_date: row for row in page.observations}
    missing = by_date[dt.date(2024, 1, 2)]
    assert missing.is_missing is True
    assert missing.value is None


def test_a_genuine_zero_is_not_confused_with_a_missing_value() -> None:
    """The other half of the same rule, and the one a lax parser breaks.

    If '.' became 0 then a real 0 and an absent value would be the same row,
    and any feature averaging them would be averaging invented observations.
    """
    page = parse_observations(read_fixture("observations-missing-value.json"), series_id="X")
    by_date = {row.observation_date: row for row in page.observations}
    zero = by_date[dt.date(2024, 1, 3)]
    assert zero.is_missing is False
    assert zero.value == Decimal("0")
    assert len(page.observations) == 3  # nothing was dropped


def test_no_observation_is_silently_dropped() -> None:
    """Row count matches FRED's own declared count for the page."""
    for name, series_id in (
        ("observations-revised-twice.json", "REVISEDTWICE"),
        ("observations-missing-value.json", "WITHMISSING"),
    ):
        payload = read_fixture(name)
        page = parse_observations(payload, series_id=series_id)
        assert len(page.observations) == page.count == payload["count"]


def test_documented_example_response_parses() -> None:
    """The captured payload: FRED's own published example, parsed end to end.

    This is the only fixture that is genuinely FRED output, so it is what pins
    the envelope contract. It carries a single real-time period across all 84
    observations, which is why the revision cases need constructed payloads.
    """
    page = parse_observations(read_fixture("observations-gnpca-docs-example.json"), series_id="X")
    assert len(page.observations) == page.count == 84
    assert page.offset == 0
    assert page.has_more is False
    assert {row.vintage_start_date for row in page.observations} == {dt.date(2013, 8, 14)}
    assert {row.vintage_end_date for row in page.observations} == {dt.date(2013, 8, 14)}
    assert page.observations[0].observation_date == dt.date(1929, 1, 1)
    assert page.observations[0].value == Decimal("1065.9")
    assert not any(row.is_missing for row in page.observations)


def test_has_more_follows_declared_counters_not_page_fullness() -> None:
    """Paging is driven by count/offset, so an exactly-full last page terminates."""
    first = parse_observations(read_fixture("observations-paged-page1.json"), series_id="X")
    second = parse_observations(read_fixture("observations-paged-page2.json"), series_id="X")
    assert first.has_more is True
    assert second.has_more is False


# --- refusals: nothing is guessed past -------------------------------------


@pytest.mark.parametrize(
    "value",
    ["", " ", "n/a", "NA", "1.2.3", "--", "1,234.5"],
)
def test_an_unrecognized_value_marker_raises_rather_than_being_coerced(value: str) -> None:
    """Only '.' means missing. Anything else is a contract change to surface.

    Coercing an unknown marker to NULL would silently turn a source change into
    invisible data loss; coercing it to a number would invent one.
    """
    payload = {
        "count": 1,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "9999-12-31",
                "date": "2024-01-01",
                "value": value,
            }
        ],
    }
    with pytest.raises(PermanentSourceError, match="missing marker"):
        parse_observations(payload, series_id="X")


@pytest.mark.parametrize("value", ["NaN", "-NaN", "sNaN", "Infinity", "-Infinity", "inf"])
def test_non_finite_decimals_are_refused(value: str) -> None:
    """``Decimal("NaN")`` succeeds — so this guard is the only thing stopping it.

    A NaN in a numeric fact column is silent corruption: it propagates through
    every aggregation without raising and turns downstream features into NaN
    without ever failing a test.
    """
    payload = {
        "count": 1,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "9999-12-31",
                "date": "2024-01-01",
                "value": value,
            }
        ],
    }
    with pytest.raises(PermanentSourceError, match=r"non-finite|missing marker"):
        parse_observations(payload, series_id="X")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"count": 1, "offset": 0, "limit": 1}, "must be a JSON array"),
        ({"observations": {}, "count": 0, "offset": 0, "limit": 1}, "must be a JSON array"),
        ({"observations": [], "offset": 0, "limit": 1}, "'count' must be an integer"),
        ({"observations": [], "count": True, "offset": 0, "limit": 1}, "must be an integer"),
        ({"observations": [], "count": -1, "offset": 0, "limit": 1}, "must be >= 0"),
    ],
)
def test_a_malformed_envelope_raises(payload: dict[str, object], match: str) -> None:
    """A shape this parser does not understand is a defect, not a zero-row page."""
    with pytest.raises(PermanentSourceError, match=match):
        parse_observations(payload, series_id="X")


@pytest.mark.parametrize("field", ["date", "realtime_start", "realtime_end", "value"])
def test_a_row_missing_a_required_field_raises(field: str) -> None:
    """Every documented row field is required; none is defaulted."""
    row = {
        "realtime_start": "2024-01-01",
        "realtime_end": "9999-12-31",
        "date": "2024-01-01",
        "value": "1.0",
    }
    del row[field]
    payload = {"count": 1, "offset": 0, "limit": 1, "observations": [row]}
    with pytest.raises(PermanentSourceError, match=field):
        parse_observations(payload, series_id="X")


def test_a_malformed_date_raises() -> None:
    """A date that is not YYYY-MM-DD is never repaired or skipped."""
    payload = {
        "count": 1,
        "offset": 0,
        "limit": 1,
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "9999-12-31",
                "date": "01/01/2024",
                "value": "1.0",
            }
        ],
    }
    with pytest.raises(PermanentSourceError, match="not a YYYY-MM-DD date"):
        parse_observations(payload, series_id="X")


# --- series metadata ---------------------------------------------------------


def test_series_metadata_versions_parse_with_their_units() -> None:
    """Units come from the source and are carried per version (directive §8)."""
    versions = parse_series_versions(
        read_fixture("series-revised-twice.json"), series_id="REVISEDTWICE"
    )
    assert len(versions) == 2
    assert versions[0].vintage_end_date == dt.date(2024, 6, 26)
    assert versions[1].vintage_end_date is None
    assert versions[0].title != versions[1].title  # metadata versions too
    assert all(version.units for version in versions)
    assert all(version.frequency_short == "Q" for version in versions)


def test_metadata_for_a_different_series_is_refused() -> None:
    """A response about another series must not be stored under the requested id."""
    payload = read_fixture("series-revised-twice.json")
    with pytest.raises(PermanentSourceError, match="not the requested"):
        parse_series_versions(payload, series_id="SOMETHINGELSE")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "must be a JSON array"),
        ({"seriess": {}}, "must be a JSON array"),
        ({"seriess": []}, "is empty"),
    ],
)
def test_a_malformed_series_envelope_raises(payload: dict[str, object], match: str) -> None:
    """An empty or wrong-shaped metadata envelope is refused, never read as absence."""
    with pytest.raises(PermanentSourceError, match=match):
        parse_series_versions(payload, series_id="X")


def test_realtime_bounds_match_freds_documented_values() -> None:
    """The complete-real-time-period idiom, pinned to the documented dates."""
    assert EARLIEST_REALTIME_DATE.isoformat() == "1776-07-04"
    assert LATEST_REALTIME_DATE.isoformat() == "9999-12-31"
    assert MISSING_VALUE_MARKER == "."
