"""Parsing FRED/ALFRED responses, and deriving ``knowledge_time`` from a vintage (P3.8).

Everything below was read from St. Louis Fed documentation on 2026-08-01. The
pages are named per claim so a later reader can re-verify instead of trusting
this docstring; nothing here was inferred from how the data "probably" looks.

Vintages are the whole point
----------------------------

Macro series are **revised**. Real GDP for a quarter is published, then revised,
then revised again; payrolls are revised every month; almost everything is.
Ingesting only today's values would hand every backtest the benefit of
revisions that had not happened yet — invariant I1's failure mode, and the one
that makes a backtest look excellent and be worthless.

FRED exposes revision history through ALFRED **real-time periods**
(``https://fred.stlouisfed.org/docs/api/fred/realtime_period.html``):

    *"The real-time period marks when facts were true or when information was
    known until it changed. ... The real-time period set by realtime_start and
    realtime_end is a (closed, closed) period."*

and ``fred/series/observations`` with ``output_type=1`` (*"Observations by
Real-Time Period"*, the documented default) returns one row per (observation
date, real-time period). ALFRED's Download Data Help
(``https://alfred.stlouisfed.org/help/downloaddata``) defines the two dates
exactly:

    *"The real-time period start date defines the first vintage date for which
    a data value is the latest revision available. The real-time period end
    date defines the last vintage date for which a data value is the latest
    revision available."*

That is precisely D-011's bitemporal shape. The observation date is event time
(``valid_from``); the real-time period **start** is the vintage from which
``knowledge_time`` is derived. Requesting ``realtime_start=1776-07-04`` and
``realtime_end=9999-12-31`` returns the complete revision history — the
documented idiom for *"all known information from the first available to the
last available"*.

``knowledge_time`` — a documented conservative lag, never the raw date
---------------------------------------------------------------------

FRED gives a vintage **date**, not an instant. D-011 is explicit about what to
do with a date-only source: apply *"a documented conservative lag ... never the
report date itself, because same-day availability at the open cannot be
assumed"*.

:func:`vintage_date_to_knowledge_time` therefore maps vintage date *V* to
**midnight America/New_York at the start of V+1**, converted to UTC. Read
plainly: a value first available on vintage date *V* is treated as knowable
only once day *V* is over in the timezone US markets trade in.

Why that direction and that size:

- FRED publishes no intraday release instant through this API, so any
  intraday guess would be invented. Releases do land during the morning (BLS
  and BEA release at 8:30 a.m. ET), so the *true* instant is somewhere inside
  day *V* — meaning the only safe assumption is the **end** of it. Choosing
  the start of *V* would assert knowledge hours before it existed, which is
  lookahead in the one direction that flatters a backtest.
- The boundary lands after the US market close on day *V*, so no strategy
  trading on day *V* can see a number released that morning. Conservative by a
  few hours, and conservative is the side to be wrong on.
- D-011's stated default is the next *trading* day; this is the next
  **calendar** day. The difference is deliberate and is a downgrade in
  conservatism only across weekends and holidays, where no trading decision
  occurs between the two candidates anyway. No trading calendar exists in the
  codebase before Phase 4, and inventing one here would be a guess (§9.4).
  Recorded so the choice is visible rather than discovered.

``last_updated`` from ``fred/series`` is **not** used as a knowledge time,
though it is a real timestamp. It describes when the *series* was last touched
as a whole, so it is identical for every observation in it and says nothing
about when any particular revision became available. Using it would be a
lookahead for old observations and a lag for new ones.

Missing values
--------------

FRED marks a missing observation with the string ``"."``
(:data:`MISSING_VALUE_MARKER`); ALFRED's Download Data Help documents the same
marker for an undefined real-time period end. It is parsed into
``value=None, is_missing=True`` and never into ``0``, ``NaN`` or a dropped row.
Anything else non-numeric raises :class:`PermanentSourceError` rather than
being coerced — including the strings ``Decimal`` would happily accept as
``NaN`` or ``Infinity``, which is a real hazard: ``Decimal("NaN")`` succeeds.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

from backend.ingest.errors import PermanentSourceError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "EARLIEST_REALTIME_DATE",
    "LATEST_REALTIME_DATE",
    "MISSING_VALUE_MARKER",
    "OBSERVATIONS_OUTPUT_TYPE",
    "SERIES_DEFINITION_VALID_FROM",
    "VINTAGE_TIMEZONE",
    "Observation",
    "ObservationPage",
    "SeriesVersion",
    "observation_date_to_valid_from",
    "parse_observations",
    "parse_series_versions",
    "vintage_date_to_knowledge_time",
]

VINTAGE_TIMEZONE: Final = ZoneInfo("America/New_York")
"""Timezone the vintage-date boundary is taken in: the US market's own.

The question ``knowledge_time`` answers is "could a participant in *this*
market have acted on it", so the day boundary that matters is Eastern, not UTC
and not the St. Louis Fed's Central.
"""

EARLIEST_REALTIME_DATE: Final = dt.date(1776, 7, 4)
"""FRED's documented "first available" real-time date (``realtime_start`` default floor).

From ``fred/series/observations``: ``observation_start`` defaults to
``1776-07-04 (earliest available)``, and the Real-Time Periods page uses the
same value for the complete real-time period. Used verbatim rather than
approximated so a first run genuinely starts at the beginning of the source's
history — the framework's rule for a missing checkpoint.
"""

LATEST_REALTIME_DATE: Final = dt.date(9999, 12, 31)
"""FRED's documented "last available" real-time date, and its open-period sentinel.

Appears both as the ``realtime_end`` to request for a complete real-time period
and as the ``realtime_end`` of a row that is still the latest revision. Parsed
into ``None`` (see :attr:`Observation.vintage_end_date`) rather than stored as a
literal year-9999 date that could silently enter date arithmetic.
"""

MISSING_VALUE_MARKER: Final = "."
"""FRED's marker for a value that does not exist at a vintage."""

OBSERVATIONS_OUTPUT_TYPE: Final = "1"
"""``output_type=1`` — *"Observations by Real-Time Period"*.

Sent explicitly even though it is the documented default: this connector's
entire correctness rests on getting one row per (observation date, real-time
period), and relying on a remote default for that is a silent dependency.
"""

_MAX_OBSERVATIONS_PER_PAGE: Final = 100000
"""``limit``'s documented maximum (and default) for ``fred/series/observations``."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One FRED observation at one vintage — one row of one revision history.

    Attributes:
        observation_date: the period the reading measures, as FRED labels it
            (period start for aggregated frequencies). Calendar date.
        vintage_start_date: FRED ``realtime_start`` — the first vintage date
            at which this value was the latest revision. Calendar date; the
            input to :func:`vintage_date_to_knowledge_time`, never itself a
            knowledge time.
        vintage_end_date: FRED ``realtime_end``, or ``None`` when FRED
            returned :data:`LATEST_REALTIME_DATE` (still the latest revision
            at the moment of the request).
        value: the observed value, or ``None`` when ``is_missing``. Arbitrary
            precision :class:`~decimal.Decimal`, parsed from FRED's decimal
            string without rounding. Unit is the series' — see
            :class:`SeriesVersion`.
        is_missing: ``True`` when FRED reported :data:`MISSING_VALUE_MARKER`.
            A stated absence is knowledge, and is kept as such rather than
            dropped or zeroed.
    """

    observation_date: dt.date
    vintage_start_date: dt.date
    vintage_end_date: dt.date | None
    value: Decimal | None
    is_missing: bool


@dataclass(frozen=True, slots=True)
class ObservationPage:
    """One page of an observations response, with the paging counters FRED states.

    Attributes:
        observations: the page's rows, in the order FRED returned them.
        count: FRED's total row count for the request (dimensionless).
        offset: the offset this page starts at (dimensionless).
        limit: the page size FRED applied (dimensionless).
    """

    observations: tuple[Observation, ...]
    count: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        """Whether more rows remain after this page.

        Derived from FRED's own ``count``/``offset`` rather than from "the page
        came back full", which would mis-handle a total that is an exact
        multiple of the limit.
        """
        return self.offset + len(self.observations) < self.count


@dataclass(frozen=True, slots=True)
class SeriesVersion:
    """One real-time version of a FRED series' definition.

    Attributes:
        series_id: FRED identifier, e.g. ``"GDPC1"``.
        vintage_start_date: FRED ``realtime_start`` of this metadata version.
        vintage_end_date: FRED ``realtime_end``, or ``None`` when open.
        title: series title.
        frequency: long frequency label, e.g. ``"Quarterly"``.
        frequency_short: short frequency code, e.g. ``"Q"``. Stored verbatim;
            **no period length is derived from it** (see
            :class:`~backend.db.models.MacroObservation` for why).
        units: FRED's units string — the authoritative unit of every value of
            this series at this vintage (directive §8).
        units_short: FRED's abbreviated units string.
        seasonal_adjustment_short: e.g. ``"SA"``, ``"NSA"``, ``"SAAR"``.
        observation_start: earliest observation date FRED reported here.
        observation_end: latest observation date FRED reported here.
    """

    series_id: str
    vintage_start_date: dt.date
    vintage_end_date: dt.date | None
    title: str
    frequency: str
    frequency_short: str
    units: str
    units_short: str
    seasonal_adjustment_short: str
    observation_start: dt.date
    observation_end: dt.date


def vintage_date_to_knowledge_time(vintage_date: dt.date) -> dt.datetime:
    """Convert a FRED vintage date to a conservative UTC ``knowledge_time``.

    Applies the connector's documented lag: the value is treated as knowable
    from **midnight America/New_York at the start of the day after**
    ``vintage_date``. See the module docstring for why the end of the vintage
    day is the only assumption that cannot introduce lookahead.

    Args:
        vintage_date: FRED ``realtime_start``, a calendar date on the US
            Eastern calendar.

    Returns:
        A timezone-aware UTC instant, strictly later than every moment of
        ``vintage_date`` in US/Eastern. For 2024-03-11 (EDT, UTC-04:00) that is
        ``2024-03-12T04:00:00+00:00``.

    Raises:
        TypeError: if ``vintage_date`` is a :class:`datetime.datetime` rather
            than a :class:`datetime.date`. ``datetime`` is a subclass of
            ``date``, so this would otherwise pass silently and truncate an
            instant nobody meant to supply.
    """
    if isinstance(vintage_date, dt.datetime):
        msg = (
            f"vintage_date must be a date, not a datetime; got {vintage_date!r}. "
            "FRED supplies vintages as dates, and accepting a datetime here would "
            "silently discard a time component the source never gave"
        )
        raise TypeError(msg)
    following_day = vintage_date + dt.timedelta(days=1)
    boundary = dt.datetime.combine(following_day, dt.time.min, tzinfo=VINTAGE_TIMEZONE)
    return boundary.astimezone(dt.UTC)


SERIES_DEFINITION_VALID_FROM: Final = dt.datetime.combine(
    EARLIEST_REALTIME_DATE, dt.time.min, tzinfo=dt.UTC
)
"""The single event-time anchor shared by **every** version of a series definition.

Not a date anyone measured — a deliberate constant, and the reasoning matters
because getting it wrong reintroduces a defect
:mod:`backend.ingest.supersession` exists to warn about.

A series *definition* ("GDPC1 means real GDP, in billions of chained dollars,
quarterly") is one fact that holds for all event time. What changes is our
**belief** about it, when FRED republishes the metadata. Under D-011 that makes
each metadata vintage a *correction* — same logical key, same ``valid_from``,
later ``knowledge_time`` — and latest-knowledge-wins then yields exactly one
visible definition at any ``as_of``.

Giving each version its own ``valid_from`` instead (say, its own vintage
instant) would make them distinct *facts* rather than versions of one, because
the as-of read is scoped per (logical key, ``valid_from``). Both would then be
visible simultaneously, and a join from an observation to its units would fan
out and silently double every row. That is precisely the two-identities-for-one
-security failure ``supersession.py`` documents; it was caught here by an
integration test returning two ``macro_series`` rows where one was required.

The value is FRED's own documented "first available" real-time date rather than
an arbitrary epoch, and it is a **constant** so it cannot drift between runs —
an anchor derived from the earliest vintage actually seen would move the moment
a backfill reached further back, orphaning every previously written version.
"""


def observation_date_to_valid_from(observation_date: dt.date) -> dt.datetime:
    """Convert an observation date to its event-time ``valid_from``.

    Args:
        observation_date: the period label FRED assigns the reading.

    Returns:
        That date at 00:00 UTC, timezone-aware. Event time is UTC-anchored
        (unlike knowledge time, which is anchored to the market's calendar)
        because an observation date is a period label, not a market moment;
        migration 0008 enforces the correspondence with a CHECK.
    """
    return dt.datetime.combine(observation_date, dt.time.min, tzinfo=dt.UTC)


def parse_observations(payload: Mapping[str, Any], *, series_id: str) -> ObservationPage:
    """Parse one ``fred/series/observations`` JSON body.

    Args:
        payload: the decoded JSON object.
        series_id: the series requested, used in error messages only — FRED
            does not repeat it inside the observations envelope.

    Returns:
        An :class:`ObservationPage` holding the rows and FRED's paging
        counters.

    Raises:
        PermanentSourceError: if the envelope or any row departs from the
            documented shape — a missing ``observations`` array, a
            non-integer counter, a row missing ``date``/``value``/
            ``realtime_start``/``realtime_end``, a date that is not
            ``YYYY-MM-DD``, or a value that is neither
            :data:`MISSING_VALUE_MARKER` nor a finite decimal. Nothing is
            defaulted, skipped or coerced: a shape this parser does not
            understand is a defect to surface, and guessing past it is how
            fabricated numbers enter an append-only store.
    """
    rows = payload.get("observations")
    if not isinstance(rows, list):
        msg = (
            f"fred observations for {series_id!r}: 'observations' must be a JSON array; "
            f"got {type(rows).__name__}"
        )
        raise PermanentSourceError(msg)
    observations = tuple(
        _parse_observation(row, series_id=series_id, index=index) for index, row in enumerate(rows)
    )
    return ObservationPage(
        observations=observations,
        count=_require_int(payload, "count", series_id=series_id),
        offset=_require_int(payload, "offset", series_id=series_id),
        limit=_require_int(payload, "limit", series_id=series_id),
    )


def parse_series_versions(
    payload: Mapping[str, Any], *, series_id: str
) -> tuple[SeriesVersion, ...]:
    """Parse one ``fred/series`` JSON body into its real-time versions.

    Args:
        payload: the decoded JSON object. FRED's documented envelope names the
            array ``"seriess"`` (its own spelling, not a typo here).
        series_id: the series requested, used in error messages and to reject
            a response describing a different series.

    Returns:
        One :class:`SeriesVersion` per real-time period FRED returned, in
        response order.

    Raises:
        PermanentSourceError: if ``seriess`` is missing or not an array, if any
            entry omits a documented field, or if an entry's ``id`` is not the
            series requested — answering with a different series is a source
            defect that must not be written under the requested key.
    """
    entries = payload.get("seriess")
    if not isinstance(entries, list):
        msg = (
            f"fred series metadata for {series_id!r}: 'seriess' must be a JSON array; "
            f"got {type(entries).__name__}"
        )
        raise PermanentSourceError(msg)
    if not entries:
        msg = (
            f"fred series metadata for {series_id!r}: 'seriess' is empty. FRED answers an "
            "unknown series with an error status, so an empty array is an undocumented "
            "shape and is refused rather than read as 'no such series'"
        )
        raise PermanentSourceError(msg)
    return tuple(
        _parse_series_version(entry, series_id=series_id, index=index)
        for index, entry in enumerate(entries)
    )


def _parse_observation(row: object, *, series_id: str, index: int) -> Observation:
    """Parse one observation row; raise on anything the contract does not cover."""
    context = f"fred observation {index} of {series_id!r}"
    if not isinstance(row, dict):
        msg = f"{context}: expected a JSON object; got {type(row).__name__}"
        raise PermanentSourceError(msg)
    value_text = _require_str(row, "value", context=context)
    is_missing = value_text.strip() == MISSING_VALUE_MARKER
    return Observation(
        observation_date=_require_date(row, "date", context=context),
        vintage_start_date=_require_date(row, "realtime_start", context=context),
        vintage_end_date=_optional_end_date(row, "realtime_end", context=context),
        value=None if is_missing else _parse_decimal(value_text, context=context),
        is_missing=is_missing,
    )


def _parse_series_version(entry: object, *, series_id: str, index: int) -> SeriesVersion:
    """Parse one ``seriess`` entry; raise on anything the contract does not cover."""
    context = f"fred series metadata version {index} of {series_id!r}"
    if not isinstance(entry, dict):
        msg = f"{context}: expected a JSON object; got {type(entry).__name__}"
        raise PermanentSourceError(msg)
    reported_id = _require_str(entry, "id", context=context)
    if reported_id != series_id:
        msg = (
            f"{context}: FRED returned metadata for series {reported_id!r}, not the "
            f"requested {series_id!r}; refusing to store it under the requested id"
        )
        raise PermanentSourceError(msg)
    return SeriesVersion(
        series_id=reported_id,
        vintage_start_date=_require_date(entry, "realtime_start", context=context),
        vintage_end_date=_optional_end_date(entry, "realtime_end", context=context),
        title=_require_str(entry, "title", context=context),
        frequency=_require_str(entry, "frequency", context=context),
        frequency_short=_require_str(entry, "frequency_short", context=context),
        units=_require_str(entry, "units", context=context),
        units_short=_require_str(entry, "units_short", context=context),
        seasonal_adjustment_short=_require_str(entry, "seasonal_adjustment_short", context=context),
        observation_start=_require_date(entry, "observation_start", context=context),
        observation_end=_require_date(entry, "observation_end", context=context),
    )


def _require_str(row: Mapping[str, Any], field: str, *, context: str) -> str:
    """Return a required string field, or raise naming the field and what arrived."""
    value = row.get(field)
    if not isinstance(value, str):
        msg = f"{context}: field {field!r} must be a string; got {type(value).__name__} {value!r}"
        raise PermanentSourceError(msg)
    return value


def _require_int(payload: Mapping[str, Any], field: str, *, series_id: str) -> int:
    """Return a required integer counter, or raise.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so a JSON
    ``true`` would otherwise be accepted as the count ``1``.
    """
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = (
            f"fred observations for {series_id!r}: field {field!r} must be an integer; "
            f"got {type(value).__name__} {value!r}"
        )
        raise PermanentSourceError(msg)
    if value < 0:
        msg = f"fred observations for {series_id!r}: field {field!r} must be >= 0; got {value}"
        raise PermanentSourceError(msg)
    return value


def _require_date(row: Mapping[str, Any], field: str, *, context: str) -> dt.date:
    """Return a required ``YYYY-MM-DD`` field as a date, or raise."""
    text = _require_str(row, field, context=context)
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        msg = f"{context}: field {field!r} is not a YYYY-MM-DD date: {text!r}"
        raise PermanentSourceError(msg) from exc


def _optional_end_date(row: Mapping[str, Any], field: str, *, context: str) -> dt.date | None:
    """Return a real-time period end, mapping FRED's open sentinels to ``None``.

    FRED closes an open period with :data:`LATEST_REALTIME_DATE` in JSON, and
    ALFRED's Download Data Help documents :data:`MISSING_VALUE_MARKER` for the
    same thing in its text downloads. Both map to ``None``; anything else must
    be a real date.
    """
    text = _require_str(row, field, context=context).strip()
    if text == MISSING_VALUE_MARKER:
        return None
    parsed = _require_date({field: text}, field, context=context)
    return None if parsed >= LATEST_REALTIME_DATE else parsed


def _parse_decimal(text: str, *, context: str) -> Decimal:
    """Parse a FRED value string into a finite :class:`~decimal.Decimal`.

    Raises:
        PermanentSourceError: if the string is not a decimal number, or is one
            of the non-finite forms :class:`~decimal.Decimal` accepts —
            ``NaN``, ``sNaN``, ``Infinity``, ``-Infinity``. That guard is the
            point of this function: ``Decimal("NaN")`` succeeds, and a NaN
            reaching a numeric column is exactly the silent corruption the
            missing-value handling exists to prevent.
    """
    stripped = text.strip()
    try:
        value = Decimal(stripped)
    except (InvalidOperation, ValueError) as exc:
        msg = (
            f"{context}: value {text!r} is neither the missing marker "
            f"{MISSING_VALUE_MARKER!r} nor a decimal number; refusing to guess what it "
            "means rather than writing a number the source did not state"
        )
        raise PermanentSourceError(msg) from exc
    if not value.is_finite():
        msg = (
            f"{context}: value {text!r} parses to the non-finite Decimal {value!r}; "
            "refusing to write it (a NaN or infinity in a numeric fact column is "
            "silent corruption, not data)"
        )
        raise PermanentSourceError(msg)
    return value


def observations_page_limit() -> int:
    """Return the page size to request from ``fred/series/observations`` (rows).

    FRED documents ``limit`` as *"integer between 1 and 100000, optional,
    default: 100000"*. The maximum is requested deliberately: a full vintage
    history is one logical unit and fetching it in the fewest calls keeps the
    connector well inside a rate budget FRED does not publish a number for.
    """
    return _MAX_OBSERVATIONS_PER_PAGE
