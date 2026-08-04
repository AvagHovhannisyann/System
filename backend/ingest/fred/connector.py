"""The FRED macro-series connector: vintages as bitemporal rows (P3.8).

Built on P3.1's contracts in the shape :mod:`backend.ingest.edgar` established.
What is specific to this source is the temporal problem, and it is the reason
this connector is not three lines around "get the latest values".

Why every vintage, not the latest value
---------------------------------------

Macro series are revised. A quarter's real GDP is published, then revised, then
revised again; payrolls are revised monthly; almost every series in this
connector's default list has a revision history. A store holding only today's
numbers would answer *"what was unemployment in March 2020"* with the figure
settled on years later — and every backtest reading it would silently trade on
information that did not exist at the time. That is invariant I1's failure
mode, and macro data is the easiest place in this platform to introduce it,
because the lookahead is invisible: the series still looks like a time series,
just a slightly more accurate one.

So the connector requests the **complete real-time period**
(``realtime_start=1776-07-04``, ``realtime_end=9999-12-31``,
``output_type=1``) and writes one row per (observation date, vintage). Under
D-011's read semantics — latest ``knowledge_time <= as_of`` wins per (logical
key, ``valid_from``) — a revision is therefore a **later-knowledge row**, never
an update, and an ``as_of`` before the revision returns the number that was
believed then. The append-only store gives this for free; what the connector
must not do is collapse the vintages before writing them.

``knowledge_time``
------------------

Derived from the vintage date by
:func:`~backend.ingest.fred.parse.vintage_date_to_knowledge_time`: midnight
US/Eastern at the **start of the day after** the vintage date. FRED publishes a
date, not an instant, and D-011 requires a documented conservative lag for such
sources rather than the raw date. The full reasoning, and why the end of the
vintage day is the only assumption that cannot create lookahead, is in
:mod:`backend.ingest.fred.parse`.

Two consequences the connector implements rather than leaves implicit:

- **A vintage whose derived knowledge instant has not yet passed is not
  written.** Today's vintage becomes knowable, under this policy, at midnight
  tonight. Writing it now would either be refused by
  :func:`~backend.ingest.write.validate_knowledge_time` (a future
  ``knowledge_time`` is not a late fact, it is a wrong one) or, worse, would
  leak into every ``as_of`` after that instant. It is skipped, counted as
  ``vintages_not_yet_knowable``, and picked up by the next run — the checkpoint
  deliberately does not advance past it.
- **The checkpoint is a per-series vintage watermark**, so a live run only
  writes genuinely new revisions.

Incremental sync
----------------

The checkpoint is flat by necessity (:mod:`backend.ingest.checkpoint` rejects
nesting, because a value that changes type across a JSONB round-trip silently
moves a resume position), so per-series progress is stored under prefixed keys:
``{"vintage_through:GDPC1": "2026-07-30", ...}``. Each batch carries the
**cumulative** map rather than just the series it covers, so a run that fails
part-way does not lose the progress of the series it already committed.

The next run re-requests each series from its watermark **inclusively** and
filters client-side to vintages strictly after it. Inclusive because FRED's
real-time period is closed on both ends and a request starting at *W* returns
every row whose period merely *intersects* ``[W, ∞)`` — including long-standing
values last revised years ago. Those are already stored; re-writing them would
be a no-op against the primary key, but counting their ancient knowledge times
in the live-lag check would make that check permanent noise. Filtering them out
before they reach a batch keeps the D-011 lag signal meaningful: a live run's
rows are exactly the revisions it learned about.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.bitemporal import INFINITY, BitemporalMixin
from backend.db.models import MacroObservation, MacroSeries
from backend.ingest.base import Batch, Connector
from backend.ingest.errors import PermanentSourceError
from backend.ingest.fred.client import FredClient, redacted_url, resolve_fred_api_key
from backend.ingest.fred.parse import (
    EARLIEST_REALTIME_DATE,
    LATEST_REALTIME_DATE,
    OBSERVATIONS_OUTPUT_TYPE,
    SERIES_DEFINITION_VALID_FROM,
    Observation,
    SeriesVersion,
    observation_date_to_valid_from,
    observations_page_limit,
    parse_observations,
    parse_series_versions,
    vintage_date_to_knowledge_time,
)
from backend.ingest.quality import DataQualityMetric, KnowledgeTimePolicy
from backend.ingest.ratelimit import RateLimit
from backend.ingest.registry import register_connector
from backend.ingest.write import validate_knowledge_time

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from sqlalchemy import Table

    from backend.db.base import Base
    from backend.ingest.base import ConnectorRuntime
    from backend.ingest.checkpoint import Checkpoint, JsonScalar
    from backend.ingest.quality import KnowledgeTimeLagReport

__all__ = [
    "CHECKPOINT_VINTAGE_PREFIX",
    "DEFAULT_SERIES_IDS",
    "FRED_SOURCE",
    "FredConnector",
]

_logger = get_logger(__name__)

FRED_SOURCE: Final = "fred"
"""Source name under which runs, checkpoints and metrics are recorded."""

CHECKPOINT_VINTAGE_PREFIX: Final = "vintage_through:"
"""Checkpoint key prefix; the suffix is the series id, the value an ISO date."""

DEFAULT_SERIES_IDS: Final = (
    "GDPC1",
    "UNRATE",
    "PAYEMS",
    "CPIAUCSL",
    "INDPRO",
    "UMCSENT",
    "FEDFUNDS",
    "DGS10",
    "T10Y2Y",
    "BAMLH0A0HYM2",
)
"""Default macro series: US activity, prices, labour, policy and credit conditions.

Every identifier was confirmed to exist on 2026-08-01 by requesting its public
FRED series page (``https://fred.stlouisfed.org/series/<id>``, all HTTP 200);
none was recalled or guessed. They are the conventional macro conditioning
variables for a US equity cross-section — output (``GDPC1``, ``INDPRO``),
labour (``UNRATE``, ``PAYEMS``), prices (``CPIAUCSL``), sentiment
(``UMCSENT``), policy and the curve (``FEDFUNDS``, ``DGS10``, ``T10Y2Y``) and
credit stress (``BAMLH0A0HYM2``).

A **default**, not a fixed list: the constructor takes ``series_ids``, and the
list is expected to be operator configuration once the Data Health page (P3.10)
can edit it. Note that revision behaviour differs sharply across them —
``GDPC1`` and ``PAYEMS`` are revised repeatedly, while ``DGS10`` and
``FEDFUNDS`` are market rates that essentially never are — which is exactly why
the connector must not assume either pattern.
"""

_MAX_LIVE_LAG: Final = dt.timedelta(hours=60)
"""Declared bound on ``write_instant - knowledge_time`` for live runs.

A **declaration**, not a measurement — it is the threshold D-011's live-lag
check compares against, and the check's own observation is what gets recorded.
Derived from this connector's own policy rather than from an assumption about
FRED: a live run writes only vintages it has not seen whose knowledge instant
has already passed, so the oldest such instant is midnight ET after the oldest
unseen vintage. With a daily schedule that is at most ~36 hours old at write
time (a vintage dated yesterday becomes knowable at midnight tonight and is
written by tomorrow's run); 60 hours absorbs one entirely missed run and a
weekend boundary. Exceeding it flags the run and never rejects data.
"""

_DEFAULT_MAX_SERIES_PER_RUN: Final = 25
"""Series consumed per run by default (count).

Bounds a run's wall-clock cost while leaving the checkpoint free to walk
forward, exactly as EDGAR bounds daily indices per run. Larger than the default
series list on purpose, so the default configuration completes in one run and
the cap only bites on a list an operator has grown.
"""


@register_connector
class FredConnector(Connector):
    """Vintage-aware FRED ingestion over ALFRED real-time periods (P3.8).

    One instance per run. Constructing it resolves the FRED API key and
    **refuses to proceed without one**, so a misconfigured deployment fails at
    construction rather than at the first request — and never by degrading to a
    keyless or invented path, which FRED does not offer and invariant I3
    forbids.

    Resumption: the checkpoint maps each series to the newest vintage date
    durably written for it (see the module docstring). The framework records it
    only after the batch's transaction commits, so an interrupted run can lose
    at most the batch that did not commit; the next run re-reads that series
    and re-writes it, which is harmless because :meth:`_write_batch` inserts
    idempotently.
    """

    source_name: ClassVar[str] = FRED_SOURCE
    knowledge_time_policy: ClassVar[KnowledgeTimePolicy] = KnowledgeTimePolicy(
        description=(
            "FRED/ALFRED vintage date (the observation's realtime_start, i.e. the first "
            "vintage date at which the value was the latest revision), plus a documented "
            "conservative lag: knowledge_time is midnight America/New_York at the START OF "
            "THE DAY AFTER that vintage date, in UTC. Never the raw vintage date — FRED "
            "publishes a date, not a release instant, and assuming availability at the "
            "start of the vintage day would claim knowledge hours before it existed "
            "(D-011 date-only-source rule). The boundary falls after the US market close "
            "on the vintage day, so a release that landed that morning cannot be traded on "
            "that day. Revisions are separate rows with later knowledge times, never "
            "updates, so an as_of before a revision returns the value believed then."
        ),
        max_live_lag=_MAX_LIVE_LAG,
    )
    rate_limit: ClassVar[RateLimit] = RateLimit(requests_per_second=1.0, burst=1)
    """One request per second — **self-imposed**, not a published figure.

    FRED documents *that* it rate-limits and returns 429 when exceeded
    (``https://fred.stlouisfed.org/docs/api/fred/errors.html``, read
    2026-08-01) but publishes **no numeric ceiling** anywhere in its API
    documentation. Rather than quote a number the source does not state, this
    connector declares a deliberately conservative budget it has no need to
    exceed: a full run of the default list is one metadata call plus one or two
    observation pages per series — tens of requests, seconds of pacing. If FRED
    later publishes a ceiling, this should become that ceiling divided by the
    worker count (the token bucket binds per process).
    """

    def __init__(
        self,
        runtime: ConnectorRuntime | None = None,
        *,
        client: FredClient | None = None,
        api_key: str | None = None,
        series_ids: Sequence[str] = DEFAULT_SERIES_IDS,
        max_series_per_run: int | None = _DEFAULT_MAX_SERIES_PER_RUN,
    ) -> None:
        """Create a connector, refusing to exist without a configured API key.

        Args:
            runtime: injectable clocks/scheduler (framework default in
                production).
            client: a :class:`~backend.ingest.fred.client.FredClient` to use
                instead of building one. Tests pass one wired to captured or
                constructed responses; production leaves it ``None``.
            api_key: FRED API key; defaults to the ``FRED_API_KEY``
                environment variable via
                :func:`~backend.ingest.fred.client.resolve_fred_api_key`.
            series_ids: the series to ingest, in order. Defaults to
                :data:`DEFAULT_SERIES_IDS`.
            max_series_per_run: series consumed per run, or ``None`` for
                unbounded. Defaults to :data:`_DEFAULT_MAX_SERIES_PER_RUN`.

        Raises:
            FredApiKeyNotConfiguredError: if no key is configured and none was
                passed. FRED requires a key on every endpoint, so this is
                fail-closed with no degraded mode.
            ValueError: if ``series_ids`` is empty or holds a blank or
                duplicated identifier, or if ``max_series_per_run`` is not
                positive. A duplicate would make one series' checkpoint
                overwrite its own progress mid-run.
        """
        super().__init__(runtime)
        if max_series_per_run is not None and max_series_per_run < 1:
            msg = f"max_series_per_run must be >= 1 or None; got {max_series_per_run}"
            raise ValueError(msg)
        cleaned = tuple(series_id.strip() for series_id in series_ids)
        if not cleaned:
            msg = (
                "series_ids must name at least one FRED series; an empty list would make "
                "a run that fetches nothing look like a successful ingestion"
            )
            raise ValueError(msg)
        if any(not series_id for series_id in cleaned):
            msg = f"series_ids must not contain a blank identifier; got {series_ids!r}"
            raise ValueError(msg)
        duplicates = sorted({sid for sid in cleaned if cleaned.count(sid) > 1})
        if duplicates:
            msg = f"series_ids must be unique; repeated: {duplicates}"
            raise ValueError(msg)
        self._api_key = api_key if api_key is not None else resolve_fred_api_key()
        self._client = client
        self._series_ids = cleaned
        self._max_series = max_series_per_run
        self._series_consumed = 0
        self._observation_rows_read = 0
        self._observation_pages_fetched = 0
        self._series_metadata_versions = 0
        self._missing_value_observations = 0
        self._already_known_vintages = 0
        self._not_yet_knowable_vintages = 0

    async def fetch_batches(self, checkpoint: Checkpoint | None) -> AsyncIterator[Batch]:
        """Yield one batch per series: its metadata versions and every new vintage.

        Args:
            checkpoint: the previous run's per-series vintage watermarks, or
                ``None`` on the source's first run — which starts at
                :data:`~backend.ingest.fred.parse.EARLIEST_REALTIME_DATE`, the
                beginning of the source's history, per the framework's rule
                that a missing checkpoint never means "start from now".

        Yields:
            One :class:`~backend.ingest.base.Batch` per series consumed,
            holding that series' new metadata versions and new observation
            vintages, with the cumulative checkpoint. Metadata and observations
            ride in the same batch so they commit in one transaction — an
            observation is never visible without the units it must be read
            with.

        Raises:
            SourceUnavailableError: from any request that fails. Nothing is
                substituted (I3).
            PermanentSourceError: if a checkpoint entry is not an ISO date. A
                resume position that cannot be read is a defect to surface, not
                to reset past — resetting would silently re-ingest or skip
                history.
        """
        watermarks = self._resume_watermarks(checkpoint)
        progress: dict[str, dt.date] = dict(watermarks)
        client = self._client if self._client is not None else self._build_client()
        owns_client = self._client is None
        try:
            for series_id in self._selected_series():
                known_through = watermarks.get(series_id)
                rows, newest_vintage = await self._rows_for_series(client, series_id, known_through)
                self._series_consumed += 1
                if newest_vintage is not None:
                    progress[series_id] = newest_vintage
                yield Batch(rows=rows, checkpoint=self._checkpoint(progress))
        finally:
            if owns_client:
                await client.aclose()

    def _build_client(self) -> FredClient:
        """Build the production HTTP client from the resolved API key."""
        return FredClient(api_key=self._api_key)

    def _selected_series(self) -> tuple[str, ...]:
        """Return the series to consume this run, respecting the per-run cap."""
        if self._max_series is None:
            return self._series_ids
        return self._series_ids[: self._max_series]

    def _resume_watermarks(self, checkpoint: Checkpoint | None) -> dict[str, dt.date]:
        """Return the per-series vintage watermarks recorded by the previous run.

        Entries for series not in this run's list are ignored here but are
        carried forward by :meth:`_checkpoint`, so narrowing the configured
        list temporarily does not discard the progress of the series left out.
        """
        if checkpoint is None:
            return {}
        watermarks: dict[str, dt.date] = {}
        for key, raw in checkpoint.items():
            if not key.startswith(CHECKPOINT_VINTAGE_PREFIX):
                continue
            series_id = key[len(CHECKPOINT_VINTAGE_PREFIX) :]
            if not isinstance(raw, str):
                msg = (
                    f"checkpoint {key!r} must be an ISO date string; "
                    f"got {type(raw).__name__} {raw!r}"
                )
                raise PermanentSourceError(msg)
            try:
                watermarks[series_id] = dt.date.fromisoformat(raw)
            except ValueError as exc:
                msg = f"checkpoint {key!r} is not an ISO date: {raw!r}"
                raise PermanentSourceError(msg) from exc
        return watermarks

    def _checkpoint(self, progress: Mapping[str, dt.date]) -> dict[str, JsonScalar]:
        """Render the cumulative per-series watermarks as a flat checkpoint."""
        return {
            f"{CHECKPOINT_VINTAGE_PREFIX}{series_id}": vintage.isoformat()
            for series_id, vintage in sorted(progress.items())
        }

    async def _rows_for_series(
        self, client: FredClient, series_id: str, known_through: dt.date | None
    ) -> tuple[tuple[Base, ...], dt.date | None]:
        """Fetch one series' metadata and observations, and build its new rows.

        Returns:
            ``(rows, newest_vintage)`` — the rows to write, and the newest
            vintage date among them, which becomes the series' checkpoint
            watermark. ``newest_vintage`` is ``None`` when nothing new was
            written, leaving the watermark where it was so the next run
            re-examines the same window rather than stepping over it.
        """
        realtime_start = (
            EARLIEST_REALTIME_DATE if known_through is None else known_through
        ).isoformat()
        realtime_end = LATEST_REALTIME_DATE.isoformat()
        now = self.runtime.now()

        metadata_params = {
            "series_id": series_id,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
        }
        metadata_payload = await self._get(
            "series", metadata_params, f"series metadata {series_id}", client=client
        )
        versions = parse_series_versions(metadata_payload, series_id=series_id)
        self._series_metadata_versions += len(versions)
        metadata_url = redacted_url("series", metadata_params)

        observations = await self._observations(client, series_id, realtime_start, realtime_end)

        rows: list[Base] = []
        newest_vintage: dt.date | None = None
        for version in versions:
            if not self._is_new_and_knowable(version.vintage_start_date, known_through, now):
                continue
            rows.append(self._series_row(version, metadata_url))
            newest_vintage = _later(newest_vintage, version.vintage_start_date)
        for observation in observations:
            if not self._is_new_and_knowable(observation.vintage_start_date, known_through, now):
                continue
            if observation.is_missing:
                self._missing_value_observations += 1
            rows.append(self._observation_row(series_id, observation))
            newest_vintage = _later(newest_vintage, observation.vintage_start_date)
        _logger.info(
            "ingest.fred.series_parsed",
            source=self.source_name,
            series_id=series_id,
            resume_from=realtime_start,
            metadata_versions=len(versions),
            observation_rows_read=len(observations),
            rows_built=len(rows),
            newest_vintage=None if newest_vintage is None else newest_vintage.isoformat(),
        )
        return tuple(rows), newest_vintage

    async def _observations(
        self, client: FredClient, series_id: str, realtime_start: str, realtime_end: str
    ) -> list[Observation]:
        """Fetch every page of one series' complete vintage history.

        Paging follows FRED's own ``count``/``offset``/``limit`` counters
        rather than "the page came back full", which would mishandle a total
        that is an exact multiple of the page size.

        Raises:
            PermanentSourceError: if FRED does not advance the offset between
                pages. Continuing would loop forever re-reading one page;
                stopping quietly would drop the remainder and report a
                truncated history as complete.
        """
        collected: list[Observation] = []
        offset = 0
        limit = observations_page_limit()
        while True:
            params = {
                "series_id": series_id,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "output_type": OBSERVATIONS_OUTPUT_TYPE,
                "sort_order": "asc",
                "limit": str(limit),
                "offset": str(offset),
            }
            page = parse_observations(
                await self._get(
                    "series/observations",
                    params,
                    f"observations {series_id} offset {offset}",
                    client=client,
                ),
                series_id=series_id,
            )
            self._observation_pages_fetched += 1
            self._observation_rows_read += len(page.observations)
            collected.extend(page.observations)
            if not page.has_more:
                return collected
            if not page.observations:
                msg = (
                    f"fred observations for {series_id!r}: FRED reports {page.count} rows "
                    f"but returned an empty page at offset {offset}; refusing to loop or to "
                    "report a truncated history as complete"
                )
                raise PermanentSourceError(msg)
            offset += len(page.observations)

    def _is_new_and_knowable(
        self, vintage_date: dt.date, known_through: dt.date | None, now: dt.datetime
    ) -> bool:
        """Decide whether one vintage should be written on this run.

        Two independent reasons to skip, counted separately because they mean
        opposite things:

        - **already known** — the vintage is at or before the resume watermark,
          so it is durably stored. FRED returns these because a real-time
          period merely intersecting the requested window is returned; keeping
          them would re-write no-ops and drown the live-lag check in decade-old
          knowledge times.
        - **not yet knowable** — the derived knowledge instant has not passed.
          Writing it would assert knowledge that does not yet exist; the
          checkpoint deliberately does not advance past it, so the next run
          picks it up once it is genuinely knowable.
        """
        if known_through is not None and vintage_date <= known_through:
            self._already_known_vintages += 1
            return False
        if vintage_date_to_knowledge_time(vintage_date) > now:
            self._not_yet_knowable_vintages += 1
            return False
        return True

    def _series_row(self, version: SeriesVersion, source_url: str) -> MacroSeries:
        """Build one ``macro_series`` row from a parsed metadata version.

        Event time is the **shared** anchor
        :data:`~backend.ingest.fred.parse.SERIES_DEFINITION_VALID_FROM`, not
        this version's vintage instant. Every version of one series must carry
        the same ``valid_from`` or the as-of read treats them as separate facts
        and returns all of them at once — see that constant's docstring for the
        fan-out that produces. Only ``knowledge_time`` distinguishes versions,
        which is exactly D-011's correction shape.
        """
        knowledge_time = vintage_date_to_knowledge_time(version.vintage_start_date)
        return MacroSeries(
            series_id=version.series_id,
            title=version.title,
            frequency=version.frequency,
            frequency_short=version.frequency_short,
            units=version.units,
            units_short=version.units_short,
            seasonal_adjustment_short=version.seasonal_adjustment_short,
            observation_start=version.observation_start,
            observation_end=version.observation_end,
            vintage_start_date=version.vintage_start_date,
            vintage_end_date=version.vintage_end_date,
            source_url=source_url,
            valid_from=SERIES_DEFINITION_VALID_FROM,
            valid_to=INFINITY,
            knowledge_time=knowledge_time,
        )

    def _observation_row(self, series_id: str, observation: Observation) -> MacroObservation:
        """Build one ``macro_observation`` row from a parsed observation vintage.

        Event time is the observation date at 00:00 UTC, open-ended; knowledge
        time is the derived vintage instant. The two are independent here in a
        way they are not for EDGAR, and that independence is the entire point:
        it is what lets one observation date carry several values, each
        knowable from a different moment.
        """
        return MacroObservation(
            series_id=series_id,
            observation_date=observation.observation_date,
            value=observation.value,
            is_missing=observation.is_missing,
            vintage_start_date=observation.vintage_start_date,
            vintage_end_date=observation.vintage_end_date,
            valid_from=observation_date_to_valid_from(observation.observation_date),
            valid_to=INFINITY,
            knowledge_time=vintage_date_to_knowledge_time(observation.vintage_start_date),
        )

    async def _get(
        self, path: str, params: Mapping[str, str], description: str, *, client: FredClient
    ) -> dict[str, Any]:
        """Fetch one FRED endpoint through the framework's pacing and retry.

        Routing every outbound call through
        :meth:`~backend.ingest.base.Connector.request` is what makes the
        declared rate limit binding rather than advisory. ``description`` never
        carries the API key (invariant I5).
        """
        return await self.request(
            lambda: client.get_json(path, params, source=self.source_name),
            description=description,
        )

    async def _write_batch(self, rows: Sequence[Base]) -> None:
        """Commit one series' rows idempotently, in a single transaction.

        Overrides the framework default for the same reason EDGAR does: **a
        re-run of a series must not fail.** The checkpoint is recorded on the
        run record only after a batch commits, so a process killed between the
        commit and the run record leaves the series written but unrecorded, and
        the next run re-reads it. With the framework's plain ``add_all`` that
        re-write collides with the append-only primary key and the source is
        stuck until someone intervenes. ``ON CONFLICT DO NOTHING`` on the full
        primary key makes the re-write a no-op instead — and it can only ever
        suppress a row identical in its versioning coordinates, since a genuine
        revision carries a later vintage and therefore a later
        ``knowledge_time`` and a different key.

        The framework's ``before_flush`` knowledge-time guard does **not** see
        Core DML, so this method calls
        :func:`~backend.ingest.write.validate_knowledge_time` on every row
        explicitly. That is the same gap in the base-class contract EDGAR
        reported; it is worked around identically rather than silently.
        """
        if not rows:
            return
        reference = self.runtime.now()
        for row in rows:
            if isinstance(row, BitemporalMixin):
                validate_knowledge_time(
                    row.knowledge_time, context=type(row).__name__, now=reference
                )
        async with ingest_writer_session() as session:
            for model, values in _grouped_insert_values(rows):
                table = cast("Table", sa.inspect(model).local_table)
                await session.execute(pg_insert(table).on_conflict_do_nothing(), values)
            await session.commit()

    def _metrics(
        self,
        rows_written: int,
        batches_written: int,
        lag_report: KnowledgeTimeLagReport | None,
    ) -> list[DataQualityMetric]:
        """Extend the framework metrics with what this run measured at FRED.

        Every entry is a count observed while running; none is defaulted. Three
        are worth reading together, because they are how a coverage figure is
        kept from being mistaken for a claim about the source:
        ``observation_rows_read`` is what FRED returned,
        ``vintages_already_known`` is what the resume filter dropped, and
        ``vintages_not_yet_knowable`` is what the knowledge-time policy
        deferred to the next run.
        """
        metrics = super()._metrics(rows_written, batches_written, lag_report)
        metrics.extend(
            [
                DataQualityMetric("series_consumed", self._series_consumed, "series"),
                DataQualityMetric(
                    "series_metadata_versions", self._series_metadata_versions, "versions"
                ),
                DataQualityMetric(
                    "observation_pages_fetched", self._observation_pages_fetched, "pages"
                ),
                DataQualityMetric(
                    "observation_rows_read", self._observation_rows_read, "observations"
                ),
                DataQualityMetric(
                    "missing_value_observations", self._missing_value_observations, "observations"
                ),
                DataQualityMetric("vintages_already_known", self._already_known_vintages, "rows"),
                DataQualityMetric(
                    "vintages_not_yet_knowable", self._not_yet_knowable_vintages, "rows"
                ),
            ]
        )
        return metrics


def _later(current: dt.date | None, candidate: dt.date) -> dt.date:
    """Return the later of an optional running maximum and a candidate date."""
    return candidate if current is None or candidate > current else current


def _grouped_insert_values(rows: Sequence[Base]) -> list[tuple[type[Base], list[dict[str, Any]]]]:
    """Group ORM instances by model and render them as uniform INSERT parameter dicts.

    A near-copy of the equivalent helper in
    :mod:`backend.ingest.edgar.connector`. Duplicated rather than shared
    because promoting it into :mod:`backend.ingest.base` is a change to the
    P3.1 framework, which this task does not own; the duplication is recorded
    so it can be collapsed deliberately rather than discovered.

    Args:
        rows: transient ORM instances built by the connector, in write order.

    Returns:
        ``[(model, [values, ...]), ...]`` in first-appearance order, so series
        metadata inserts before the observations it gives units to.

    Raises:
        ValueError: if two instances of one model set different attributes. An
            executemany needs uniform keys, and quietly filling a missing one
            with ``None`` would write a value the source never supplied.
    """
    grouped: dict[type[Base], list[dict[str, Any]]] = {}
    for row in rows:
        state = sa.inspect(row)
        values = {
            attribute.key: getattr(row, attribute.key)
            for attribute in state.mapper.column_attrs
            if attribute.key in state.dict
        }
        model = type(row)
        bucket = grouped.setdefault(model, [])
        if bucket and set(bucket[0]) != set(values):
            msg = (
                f"{model.__name__} rows in one batch set different columns "
                f"({sorted(set(bucket[0]) ^ set(values))}); an executemany cannot express "
                "that and filling the difference with NULL would write values the source "
                "never gave"
            )
            raise ValueError(msg)
        bucket.append(values)
    return list(grouped.items())
