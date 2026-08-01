"""Per-source data-quality report: coverage, gaps, staleness (P3.9).

The directive's Phase 3 gate requires a report showing "coverage, gaps, and
staleness per source", and §6.2 makes it the Data Health page's content. This
module builds it.

**Everything here is a measurement or it is ``None``.** That is the whole
design rule, and it is why so many fields are optional. A source with no
ingestion runs has no staleness — not a staleness of zero. A run that wrote no
rows measured no knowledge-time lag — not a lag of zero. A store holding no
filings has no coverage window — not an empty one starting today. Rendering a
default as though it were an observation is the exact I3 failure this report
would otherwise be the most natural place to commit, because a dashboard tile
showing "0 days stale" is indistinguishable from one showing a real zero.

Two halves, with different provenance, kept visibly separate:

- **Per-run metrics** are *persisted at run time* by the connector into
  ``ingestion_run.quality_metrics`` (P3.1 emits the framework's; P3.2 adds
  EDGAR's own). The report reads them back; it does not recompute them, so what
  the dashboard shows is what the run actually observed, including for runs
  whose code has since changed.
- **Store-side coverage, gaps and staleness** are measured *at report time* by
  querying the fact tables through :func:`backend.db.as_of` — the only read
  path there is (D-011). They therefore describe the store as believed at the
  report's ``as_of`` instant, which is stated in the report rather than
  implied.

Placement note: this module lives under ``backend/ingest/edgar/`` because P3.2
is the only source that exists. The dataclasses, :func:`run_history`,
:func:`detect_gaps` and the staleness assembly are source-agnostic already;
only the store-side measurement functions name EDGAR's tables. When a second
connector lands, the generic half belongs in ``backend/ingest/`` with each
connector contributing its own coverage probe — a move, not a redesign.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import sqlalchemy as sa

from backend.db import as_of, ingest_writer_session
from backend.db.models import EdgarFiling, EdgarFilingDocument
from backend.ingest.edgar.connector import SEC_EDGAR_SOURCE, SecEdgarConnector
from backend.ingest.runs import IngestionRun, RunStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.ingest.checkpoint import JsonScalar
    from backend.ingest.quality import KnowledgeTimePolicy

__all__ = [
    "CoverageWindow",
    "Gap",
    "KnowledgeTimeLagObservation",
    "RunSummary",
    "SourceQualityReport",
    "Staleness",
    "build_edgar_report",
    "detect_gaps",
    "edgar_coverage",
    "run_history",
]

_DEFAULT_RUN_HISTORY: Final = 20
"""Runs included in a report by default (count)."""

_SATURDAY: Final = 5
"""``date.weekday()`` value for Saturday."""


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of consecutive weekdays inside the covered window holding no filings.

    Attributes:
        first_date: first weekday of the gap (inclusive).
        last_date: last weekday of the gap (inclusive).
        weekdays: number of weekdays in the gap (count).

    Weekends are excluded because EDGAR publishes no daily index on Saturdays
    or Sundays (verified against the published 2024 QTR1 listing, which
    contains weekday entries only). Weekday gaps still include US federal
    holidays, days EDGAR was closed, and days on which no *selected* form type
    was filed — so a gap is a prompt to investigate, never by itself proof that
    data is missing. Severity is deliberately not assigned here: a threshold
    would be an invented judgement, and the consumer (P3.10) owns that policy.
    """

    first_date: dt.date
    last_date: dt.date
    weekdays: int


@dataclass(frozen=True, slots=True)
class CoverageWindow:
    """What the store actually holds for one source, at the report's ``as_of``.

    Attributes:
        first_index_date: earliest daily-index date any stored filing came
            from, or ``None`` when the store holds no filings.
        last_index_date: latest such date, or ``None``.
        index_dates_present: distinct daily-index dates with at least one
            stored filing (count).
        weekdays_in_window: weekdays between ``first_index_date`` and
            ``last_index_date`` inclusive (count), or ``None`` when there is no
            window. The denominator a coverage ratio would use; the ratio
            itself is left to the consumer so it is never shown for an empty
            store.
        filings: stored ``edgar_filing`` rows visible at the report's
            ``as_of`` (count). One row per *(filing, filer CIK)* pair, which is
            what the daily index states — a joint filing under seven CIKs is
            seven rows. Not a count of distinct submissions, and named
            ``filings`` only because that is the table it counts.
        documents: stored filing documents visible at the same instant (count).
            Keyed on the accession, so a joint filing's documents are counted
            once however many filers it has.
    """

    first_index_date: dt.date | None
    last_index_date: dt.date | None
    index_dates_present: int
    weekdays_in_window: int | None
    filings: int
    documents: int


@dataclass(frozen=True, slots=True)
class Staleness:
    """How far behind the source the store is, measured at the report instant.

    Attributes:
        newest_index_date: latest daily-index date in the store, or ``None``.
        newest_index_date_age: report instant minus the *end* of that index
            date (its 00:00 UTC plus one day), or ``None`` when the store is
            empty.
        newest_knowledge_time: latest ``knowledge_time`` — i.e. latest
            acceptance instant — among stored filings, or ``None``.
        newest_knowledge_time_age: report instant minus that acceptance
            instant, or ``None``.
        last_successful_run_ended_at: when the most recent ``succeeded`` run of
            this source finished, or ``None`` if it never has.
        last_successful_run_age: report instant minus that, or ``None``.
        consecutive_failed_runs_since_success: failed runs recorded after the
            last successful one (count). ``0`` here is a real measurement — it
            means runs have been examined and none failed since — which is why
            it is not optional.
    """

    newest_index_date: dt.date | None
    newest_index_date_age: dt.timedelta | None
    newest_knowledge_time: dt.datetime | None
    newest_knowledge_time_age: dt.timedelta | None
    last_successful_run_ended_at: dt.datetime | None
    last_successful_run_age: dt.timedelta | None
    consecutive_failed_runs_since_success: int


@dataclass(frozen=True, slots=True)
class KnowledgeTimeLagObservation:
    """The most recent live-run knowledge-time lag check, read back from a run.

    D-011 promises this check as the compensating control for the absence of a
    database constraint relating ``knowledge_time`` to ``ingested_at``. The
    check runs inside the connector run (P3.1) and is persisted with it; this
    is the report's view of it.

    Attributes:
        run_id: the run that measured it.
        observed_max_lag: the largest ``write instant - knowledge_time`` that
            run saw, or ``None`` when the run wrote no rows.
        declared_max_lag: the connector's declared bound at the time.
        breached: whether the observation exceeded the bound.
    """

    run_id: int
    observed_max_lag: dt.timedelta | None
    declared_max_lag: dt.timedelta
    breached: bool


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One ingestion run as the operational log recorded it.

    Attributes:
        run_id: surrogate run key.
        run_kind: ``backfill`` or ``live``.
        status: ``running``, ``succeeded`` or ``failed``.
        started_at: when the run opened (UTC).
        ended_at: when it closed (UTC), or ``None`` while running.
        rows_written: rows durably committed (count).
        checkpoint_after: furthest durably-written position, or ``None``.
        error_detail: failure detail, or ``None``.
        quality_metrics: the metrics the run persisted, verbatim, as
            ``{name: {"value": ..., "unit": ...}}``. Not recomputed — see the
            module docstring.
    """

    run_id: int
    run_kind: str
    status: str
    started_at: dt.datetime
    ended_at: dt.datetime | None
    rows_written: int
    checkpoint_after: dict[str, JsonScalar] | None
    error_detail: str | None
    quality_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SourceQualityReport:
    """The data-quality report for one source (P3.9).

    Attributes:
        source: connector source name.
        measured_at: the instant this report was built, and the ``as_of`` its
            store-side measurements were taken at. Stated rather than implied,
            because every count below is "as believed at" this instant.
        knowledge_time_policy: the connector's declared derivation of
            ``knowledge_time``. D-011 requires the report to display it: a
            coverage number means nothing without knowing what the timestamps
            behind it mean.
        coverage: what the store holds.
        gaps: weekday gaps inside the covered window (see :class:`Gap`).
        staleness: how far behind the source the store is.
        knowledge_time_lag: the most recent persisted live-run lag check, or
            ``None`` when no live run has ever measured one.
        runs: recent run history, newest first.
    """

    source: str
    measured_at: dt.datetime
    knowledge_time_policy: KnowledgeTimePolicy
    coverage: CoverageWindow
    gaps: tuple[Gap, ...]
    staleness: Staleness
    knowledge_time_lag: KnowledgeTimeLagObservation | None
    runs: tuple[RunSummary, ...]


async def run_history(source: str, *, limit: int = _DEFAULT_RUN_HISTORY) -> tuple[RunSummary, ...]:
    """Return the most recent ingestion runs for ``source``, newest first.

    Args:
        source: connector source name.
        limit: maximum runs to return (count, >= 1).

    Returns:
        Run summaries carrying each run's persisted quality metrics verbatim.

    Raises:
        ValueError: if ``limit`` is not positive.
    """
    if limit < 1:
        msg = f"limit must be >= 1; got {limit}"
        raise ValueError(msg)
    async with ingest_writer_session() as session:
        runs = (
            await session.scalars(
                sa.select(IngestionRun)
                .where(IngestionRun.source == source)
                .order_by(IngestionRun.started_at.desc(), IngestionRun.run_id.desc())
                .limit(limit)
            )
        ).all()
        return tuple(
            RunSummary(
                run_id=run.run_id,
                run_kind=run.run_kind,
                status=run.status,
                started_at=run.started_at,
                ended_at=run.ended_at,
                rows_written=run.rows_written,
                checkpoint_after=run.checkpoint_after,
                error_detail=run.error_detail,
                quality_metrics=dict(run.quality_metrics),
            )
            for run in runs
        )


async def edgar_coverage(as_of_ts: dt.datetime) -> tuple[CoverageWindow, tuple[dt.date, ...]]:
    """Measure EDGAR coverage in the store as believed at ``as_of_ts``.

    Args:
        as_of_ts: the knowledge-time instant to read the store at. Must be
            timezone-aware and not in the future (the query layer enforces
            both).

    Returns:
        ``(window, index_dates)`` — the coverage window and the sorted distinct
        daily-index dates that actually hold filings. The second element is
        returned rather than folded into the window because gap detection needs
        the dates themselves, and recomputing them from a second query could
        observe a different store.

    Raises:
        AsOfTimestampError: if ``as_of_ts`` is naive or in the future.
    """
    async with as_of(as_of_ts) as session:
        # Counts are aggregated in the database, not by materializing rows: this
        # report is a dashboard endpoint and the store is meant to hold years of
        # filings. The distinct index dates *are* materialized, deliberately —
        # there is one per trading day and gap detection needs them all.
        index_dates = tuple(
            sorted((await session.scalars(sa.select(EdgarFiling.index_date).distinct())).all())
        )
        filings = (
            await session.scalars(sa.select(sa.func.count(EdgarFiling.accession_number)))
        ).one()
        documents = (
            await session.scalars(sa.select(sa.func.count(EdgarFilingDocument.accession_number)))
        ).one()
    first = index_dates[0] if index_dates else None
    last = index_dates[-1] if index_dates else None
    window = CoverageWindow(
        first_index_date=first,
        last_index_date=last,
        index_dates_present=len(index_dates),
        weekdays_in_window=(
            None if first is None or last is None else _count_weekdays(first, last)
        ),
        filings=filings,
        documents=documents,
    )
    return window, index_dates


def detect_gaps(index_dates: Sequence[dt.date]) -> tuple[Gap, ...]:
    """Return the weekday gaps inside the window ``index_dates`` spans.

    Args:
        index_dates: distinct daily-index dates present in the store, sorted
            ascending. Fewer than two dates cannot bound a gap, so an empty
            tuple is returned.

    Returns:
        One :class:`Gap` per maximal run of consecutive weekdays between the
        first and last date that hold no filings. See :class:`Gap` for what a
        gap does and does not prove.
    """
    if len(index_dates) < 2:
        return ()
    present = set(index_dates)
    gaps: list[Gap] = []
    current: list[dt.date] = []
    day = index_dates[0]
    while day <= index_dates[-1]:
        if day.weekday() < _SATURDAY and day not in present:
            current.append(day)
        elif current:
            gaps.append(Gap(first_date=current[0], last_date=current[-1], weekdays=len(current)))
            current = []
        day += dt.timedelta(days=1)
    if current:
        gaps.append(Gap(first_date=current[0], last_date=current[-1], weekdays=len(current)))
    return tuple(gaps)


async def build_edgar_report(
    *,
    as_of_ts: dt.datetime | None = None,
    run_limit: int = _DEFAULT_RUN_HISTORY,
) -> SourceQualityReport:
    """Build the SEC EDGAR data-quality report (P3.9).

    Args:
        as_of_ts: knowledge-time instant to measure the store at; defaults to
            now. Passing an explicit instant makes the report reproducible.
        run_limit: how many recent runs to include (count).

    Returns:
        A :class:`SourceQualityReport`. Store-side figures describe the store
        as believed at ``as_of_ts``; per-run figures are read back verbatim
        from what each run persisted.

    Raises:
        AsOfTimestampError: if ``as_of_ts`` is naive or in the future.
        ValueError: if ``run_limit`` is not positive.
    """
    measured_at = as_of_ts if as_of_ts is not None else dt.datetime.now(dt.UTC)
    coverage, index_dates = await edgar_coverage(measured_at)
    runs = await run_history(SEC_EDGAR_SOURCE, limit=run_limit)
    newest_knowledge_time = await _newest_knowledge_time(measured_at)
    return SourceQualityReport(
        source=SEC_EDGAR_SOURCE,
        measured_at=measured_at,
        knowledge_time_policy=SecEdgarConnector.knowledge_time_policy,
        coverage=coverage,
        gaps=detect_gaps(index_dates),
        staleness=_staleness(measured_at, coverage, newest_knowledge_time, runs),
        knowledge_time_lag=_latest_lag_observation(runs),
        runs=runs,
    )


async def _newest_knowledge_time(as_of_ts: dt.datetime) -> dt.datetime | None:
    """Return the latest stored EDGAR ``knowledge_time``, or ``None`` if none.

    ``None`` when the store holds no filings: there is no newest acceptance
    instant to report, and reporting the epoch (or ``as_of_ts``) would be a
    fabricated observation.
    """
    async with as_of(as_of_ts) as session:
        return (await session.scalars(sa.select(sa.func.max(EdgarFiling.knowledge_time)))).one()


def _count_weekdays(first: dt.date, last: dt.date) -> int:
    """Return the number of Mon-Fri days in ``[first, last]`` inclusive."""
    day = first
    count = 0
    while day <= last:
        if day.weekday() < _SATURDAY:
            count += 1
        day += dt.timedelta(days=1)
    return count


def _staleness(
    measured_at: dt.datetime,
    coverage: CoverageWindow,
    newest_knowledge_time: dt.datetime | None,
    runs: Sequence[RunSummary],
) -> Staleness:
    """Assemble the staleness section, leaving unmeasurable ages as ``None``."""
    last_success = next(
        (run for run in runs if run.status == RunStatus.SUCCEEDED.value and run.ended_at), None
    )
    failed_since = 0
    for run in runs:
        if run.status == RunStatus.SUCCEEDED.value:
            break
        if run.status == RunStatus.FAILED.value:
            failed_since += 1
    index_end = (
        None
        if coverage.last_index_date is None
        else dt.datetime.combine(
            coverage.last_index_date + dt.timedelta(days=1), dt.time.min, tzinfo=dt.UTC
        )
    )
    return Staleness(
        newest_index_date=coverage.last_index_date,
        newest_index_date_age=None if index_end is None else measured_at - index_end,
        newest_knowledge_time=newest_knowledge_time,
        newest_knowledge_time_age=(
            None if newest_knowledge_time is None else measured_at - newest_knowledge_time
        ),
        last_successful_run_ended_at=None if last_success is None else last_success.ended_at,
        last_successful_run_age=(
            None
            if last_success is None or last_success.ended_at is None
            else measured_at - last_success.ended_at
        ),
        consecutive_failed_runs_since_success=failed_since,
    )


def _latest_lag_observation(runs: Sequence[RunSummary]) -> KnowledgeTimeLagObservation | None:
    """Return the newest persisted live-run lag check, or ``None`` if never measured.

    Reads the metric names the framework persists
    (:meth:`backend.ingest.base.Connector._metrics`). A run that recorded no
    lag metrics measured no lag — it is skipped rather than reported as zero.
    """
    for run in runs:
        metrics = run.quality_metrics
        declared = _metric_number(metrics, "knowledge_time_declared_max_lag")
        if declared is None:
            continue
        observed = _metric_number(metrics, "knowledge_time_max_lag")
        return KnowledgeTimeLagObservation(
            run_id=run.run_id,
            observed_max_lag=None if observed is None else dt.timedelta(seconds=observed),
            declared_max_lag=dt.timedelta(seconds=declared),
            breached=_metric_value(metrics, "knowledge_time_lag_breach") is True,
        )
    return None


def _metric_value(metrics: dict[str, Any], name: str) -> object | None:
    """Return one persisted metric's raw value, or ``None`` when absent.

    Defensive about shape on purpose: the metrics come back from a ``JSONB``
    column written by a possibly older revision of the emitting code, so this
    reads what is there rather than what the current schema would produce.
    """
    entry = metrics.get(name)
    return entry.get("value") if isinstance(entry, dict) else None


def _metric_number(metrics: dict[str, Any], name: str) -> float | None:
    """Return a persisted metric's numeric value, or ``None`` when absent/non-numeric."""
    value = _metric_value(metrics, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
