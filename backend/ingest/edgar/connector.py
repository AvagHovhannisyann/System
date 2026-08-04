"""The SEC EDGAR connector: P3.1's contracts against a real, adversarial source (P3.2).

This is the reference implementation of the connector contract. It was built
first because acceptance timestamps are the hardest temporal-correctness
problem in Phase 3, and because getting them wrong is invisible: a lookahead of
a few hours produces backtests that look better and are worthless.

``knowledge_time`` — the whole reason this connector exists
-----------------------------------------------------------

``knowledge_time`` is the filing's **acceptance instant**, converted from the
submission header's US/Eastern value to UTC by
:func:`~backend.ingest.edgar.parse.acceptance_datetime_to_utc`. It is never the
filing date, and the difference is not cosmetic. Under 17 CFR 232.13, EDGAR
assigns Forms 3/4/5, Schedules 13D/13G/14N, Form 144 and Rule 462(b) filings
the *same* business day until 10 p.m. Eastern. Accession
``0001225208-24-004041`` is a Form 4 with filing date 2024-03-11 and acceptance
2024-03-12T00:13:08Z: treating its filing date as knowledge time would assert
the market knew it about a day before it existed. In the other direction,
accession ``0000950172-24-000037`` was accepted 2024-03-08T22:59:09Z and
assigned filing date 2024-03-11 — knowable on the Friday, dated the Monday.
Both are in the committed fixtures.

The acceptance instant is not in the daily index. The index carries
``CIK|Company Name|Form Type|Date Filed|File Name`` and nothing else, so the
connector must fetch each selected submission's ``-index-headers.html``, which
carries the acceptance datetime *and* the document manifest in one request.
That is why the request budget is one per index day plus one per selected
accession, and why ``form_types`` exists.

One accession, several filers
------------------------------

A daily-index row is a *(filing, filer)* association, not a filing: EDGAR lists
one accession once per associated CIK, and the majority of accessions do have
more than one (1,809 of the 2024-03-11 index's 3,394; accession
``0001193805-24-000360`` is listed under seven). The connector therefore
fetches each accession's header once and writes one ``edgar_filing`` row per
listed CIK, with the documents stored once against the accession. Collapsing
those rows would mean electing one filer's CIK and name to represent the
filing — a choice the source does not make and this connector will not invent.

What is deliberately **not** done
---------------------------------

- **``data.sec.gov`` is not used.** Its ``acceptanceDateTime`` field is a
  correct UTC conversion for some filers and the raw Eastern wall clock with a
  ``Z`` suffix for others; the evidence and the discriminating test are in
  :mod:`backend.ingest.edgar.parse` and in
  ``backend/tests/fixtures/edgar/acceptance-datetime-crosscheck.json``.
- **Dates are never guessed.** The quarter listing is enumerated and its file
  names are used verbatim, because EDGAR answers a request for a non-existent
  daily-index file with 403 — the same status a fair-access block returns.
- **Nothing is substituted for a missing source.** Every failure path raises;
  there is no empty-batch fallback, no default acceptance time, no "assume the
  market open". A day EDGAR did not publish is a day absent from the listing,
  which is a fact about EDGAR, not a value this connector invents.

Known caveat, recorded rather than silently absorbed
-----------------------------------------------------

Some filings are **disseminated long after they are accepted**. Accession
``0000950170-24-012183`` is a ``CORRESP`` carrying acceptance 2024-02-07
16:05:02 ET, and it appears in the daily index for 2024-03-11 — a month later.
That is the observed fact; what it implies is the open question. If the daily
index is the moment a filing becomes retrievable, then for such filings the
acceptance instant is *earlier* than knowability, and using it as
``knowledge_time`` is anti-conservative — the one direction that produces
lookahead.

The connector still records the acceptance instant: it is the timestamp the
directive specifies, the one EDGAR publishes per filing, and the correct one
for the periodic-report forms selected by default (none of the captured 8-K,
10-K or 6-K filings shows this divergence). What it additionally does is
**measure** the divergence — ``filings_accepted_before_index_date`` and
``max_acceptance_to_index_lag`` — so a run that ingests affected forms says so
in the data-quality report rather than hiding it. Whether a per-form
dissemination-lag policy should override the acceptance instant for
``CORRESP``, ``UPLOAD`` and their kin is a decision for the operator, and it is
recorded here rather than made silently.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.bitemporal import INFINITY, BitemporalMixin
from backend.db.models import EdgarFiling, EdgarFilingDocument
from backend.ingest.base import Batch, Connector
from backend.ingest.edgar.client import EdgarClient, resolve_sec_user_agent
from backend.ingest.edgar.parse import (
    ARCHIVES_BASE,
    EARLIEST_ISO_NAMED_DAILY_INDEX,
    EDGAR_TIMEZONE,
    DailyIndexEntry,
    DailyIndexFile,
    FilingHeader,
    daily_index_file_url,
    filing_index_headers_url,
    parse_daily_index,
    parse_filing_header,
    parse_quarter_listing,
    quarter_listing_url,
    quarter_of,
)
from backend.ingest.errors import PermanentSourceError
from backend.ingest.quality import DataQualityMetric, KnowledgeTimePolicy
from backend.ingest.ratelimit import RateLimit
from backend.ingest.registry import register_connector
from backend.ingest.write import validate_knowledge_time

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from sqlalchemy import Table

    from backend.db.base import Base
    from backend.ingest.base import ConnectorRuntime
    from backend.ingest.checkpoint import Checkpoint
    from backend.ingest.quality import KnowledgeTimeLagReport

__all__ = [
    "CHECKPOINT_LAST_INDEX_DATE",
    "DEFAULT_FORM_TYPES",
    "SEC_EDGAR_SOURCE",
    "SecEdgarConnector",
]

_logger = get_logger(__name__)

SEC_EDGAR_SOURCE: Final = "sec_edgar"
"""Source name under which runs, checkpoints and metrics are recorded."""

CHECKPOINT_LAST_INDEX_DATE: Final = "last_index_date"
"""Checkpoint key: ISO date of the last daily index durably written."""

DEFAULT_FORM_TYPES: Final = frozenset(
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
"""Form types selected by default: periodic and current reports.

The platform's use of EDGAR is the Phase 7 extraction pipeline, whose inputs
are annual, quarterly and current reports and their foreign equivalents.
Selecting everything EDGAR disseminates would be several thousand submissions
a day (the real 2024-03-11 index carries 5,395 rows covering 3,394 distinct
accessions), each needing its own header request — about six minutes per index
day at SEC's published rate ceiling, almost all of it spent on Forms 3/4/5,
prospectuses and fund filings this platform does not read.

The filter is a **selection**, never an absence: the rows the index listed
(``index_rows_read``), the entries selected (``index_entries_selected``) and
the submissions actually fetched (``filing_headers_fetched``) are three
separate measurements, so a coverage figure can never be mistaken for "EDGAR
published nothing".
"""

_DEFAULT_MAX_INDEX_DAYS_PER_RUN: Final = 5
"""Daily indices consumed per run by default (count).

Bounds a run's wall-clock cost while leaving the checkpoint free to walk
forward: a live run that has fallen behind catches up over successive runs
rather than in one unbounded execution, and a first run — which per the
framework contract starts at the beginning of the source's history, never at
"now" — makes bounded, resumable progress instead of attempting three decades.
"""

_LAST_QUARTER: Final = 4
"""Quarter number after which the walk rolls into the next year."""

_MAX_LIVE_LAG: Final = dt.timedelta(hours=36)
"""Declared upper bound on ``write_instant - knowledge_time`` for live runs.

A **declaration**, not a measurement — it is the threshold the D-011 live-lag
check compares against, and the check's own observation is what gets recorded.
It is set from two things actually observed on 2026-08-01: a day's daily index
carries a ``last-modified`` of 10:05—10:06 p.m. throughout the published 2024
QTR1 listing (so a day's index exists only late on that day), and the earliest
acceptance times in the captured indices are just after 06:00 Eastern. A live
run consuming the newest published index therefore legitimately writes rows
whose oldest knowledge time is roughly sixteen hours old the moment it runs.
Thirty-six hours leaves room for a schedule anchored to UTC rather than Eastern
and for one missed run, and flags a feed that has fallen further behind than
that. Exceeding it flags the run; it never rejects data.
"""


@register_connector
class SecEdgarConnector(Connector):
    """Incremental EDGAR ingestion over the daily index (P3.2).

    One instance per run. Constructing it resolves the SEC contact string and
    **refuses to proceed without one**, so a misconfigured deployment fails at
    construction rather than at the first request.

    Resumption: the checkpoint is ``{"last_index_date": "YYYY-MM-DD"}``, the
    last daily index whose rows are durably committed. The framework records it
    only after the batch's transaction commits, so an interrupted run can lose
    at most the batch that did not commit; the next run re-reads that index day
    and re-writes it, which is harmless because
    :meth:`_write_batch` inserts idempotently (see its docstring).
    """

    source_name: ClassVar[str] = SEC_EDGAR_SOURCE
    knowledge_time_policy: ClassVar[KnowledgeTimePolicy] = KnowledgeTimePolicy(
        description=(
            "SEC EDGAR submission acceptance instant: the <ACCEPTANCE-DATETIME> value of "
            "the filing's own -index-headers.html, read as US/Eastern local time "
            "(America/New_York, so -05:00 or -04:00 depending on the date) and converted "
            "to UTC. Never the filing date: under 17 CFR 232.13 EDGAR dates Forms 3/4/5, "
            "Schedules 13D/13G/14N, Form 144 and Rule 462(b) filings to the same business "
            "day until 10 p.m. Eastern, so a filing date can precede its own acceptance "
            "instant in UTC by most of a day."
        ),
        max_live_lag=_MAX_LIVE_LAG,
    )
    rate_limit: ClassVar[RateLimit] = RateLimit(requests_per_second=10.0, burst=1)
    """SEC's published ceiling: "Current max request rate: 10 requests/second".

    ``burst=1`` rather than a burst equal to the rate, deliberately: a bucket
    holding ten tokens permits ten immediate requests *plus* ten refilled ones
    inside the same second, which is twenty requests in a one-second window and
    therefore over the published ceiling. A burst of one paces requests at a
    strict 100 ms apart, which is the ceiling read as written. The framework's
    per-process caveat applies — the limit binds per worker, so a multi-worker
    deployment must divide it.
    """

    def __init__(
        self,
        runtime: ConnectorRuntime | None = None,
        *,
        client: EdgarClient | None = None,
        user_agent: str | None = None,
        form_types: frozenset[str] | None = DEFAULT_FORM_TYPES,
        start_date: dt.date = EARLIEST_ISO_NAMED_DAILY_INDEX,
        end_date: dt.date | None = None,
        max_index_days_per_run: int | None = _DEFAULT_MAX_INDEX_DAYS_PER_RUN,
    ) -> None:
        """Create a connector, refusing to exist without an SEC contact string.

        Args:
            runtime: injectable clocks/scheduler (framework default in
                production).
            client: an :class:`~backend.ingest.edgar.client.EdgarClient` to use
                instead of building one. Tests pass one wired to captured
                responses; production leaves it ``None``.
            user_agent: SEC contact string; defaults to
                ``settings.sec_user_agent``.
            form_types: the form types to select, or ``None`` to select every
                filing the index lists. Defaults to
                :data:`DEFAULT_FORM_TYPES`.
            start_date: earliest daily index to consume when no checkpoint
                exists. Defaults to
                :data:`~backend.ingest.edgar.parse.EARLIEST_ISO_NAMED_DAILY_INDEX`
                — the beginning of the source's ``YYYYMMDD``-named history,
                per the framework's rule that a missing checkpoint means "start
                at the beginning", never "start at now".
            end_date: latest daily index to consume, inclusive. Defaults to
                today's date in US/Eastern, which is EDGAR's own calendar; a
                day whose index is not yet published simply does not appear in
                the quarter listing.
            max_index_days_per_run: daily indices consumed per run, or ``None``
                for unbounded. Defaults to
                :data:`_DEFAULT_MAX_INDEX_DAYS_PER_RUN`.

        Raises:
            SecUserAgentNotConfiguredError: if no contact string is configured
                and none was passed. SEC fair access is fail-closed here.
            ValueError: if ``max_index_days_per_run`` is not positive, or if
                ``start_date`` precedes EDGAR's ``YYYYMMDD``-named history (the
                two earlier naming schemes are not consumable — see
                :data:`~backend.ingest.edgar.parse.EARLIEST_ISO_NAMED_DAILY_INDEX`).
        """
        super().__init__(runtime)
        if max_index_days_per_run is not None and max_index_days_per_run < 1:
            msg = f"max_index_days_per_run must be >= 1 or None; got {max_index_days_per_run}"
            raise ValueError(msg)
        if start_date < EARLIEST_ISO_NAMED_DAILY_INDEX:
            msg = (
                f"start_date {start_date.isoformat()} precedes EDGAR's YYYYMMDD-named daily "
                f"index history (starts {EARLIEST_ISO_NAMED_DAILY_INDEX.isoformat()}); the "
                "1994 MMDDYY and 1995-1998 YYMMDD naming schemes are not consumed by this "
                "connector"
            )
            raise ValueError(msg)
        self._user_agent = user_agent if user_agent is not None else resolve_sec_user_agent()
        self._client = client
        self._form_types = form_types
        self._start_date = start_date
        self._end_date = end_date
        self._max_index_days = max_index_days_per_run
        self._index_days_consumed = 0
        self._index_rows_read = 0
        self._duplicate_rows_collapsed = 0
        self._index_entries_selected = 0
        self._filing_headers_fetched = 0
        self._legacy_named_index_files = 0
        self._accepted_before_index_date = 0
        self._max_acceptance_to_index_lag: dt.timedelta | None = None
        self._declared_document_count_mismatches = 0

    async def fetch_batches(self, checkpoint: Checkpoint | None) -> AsyncIterator[Batch]:
        """Yield one batch per daily index, resuming after the checkpointed date.

        Args:
            checkpoint: the previous run's position, or ``None`` on the
                source's first run (which starts at ``start_date``, the
                beginning of EDGAR's consumable history).

        Yields:
            One :class:`~backend.ingest.base.Batch` per daily index consumed,
            holding that day's selected filings and all their documents, with
            the checkpoint that day's date. Filings and their documents ride in
            the same batch so they commit in one transaction — a filing can
            never be visible without its manifest.

        Raises:
            SourceUnavailableError: from any request that fails. Nothing is
                substituted (I3).
            PermanentSourceError: if the checkpoint holds a value that is not
                an ISO date; a resume position that cannot be read is a defect
                to surface, not to reset past (resetting would silently
                re-ingest or skip history).
        """
        resume_after = self._resume_after(checkpoint)
        end_date = self._end_date if self._end_date is not None else self._today_eastern()
        client = self._client if self._client is not None else self._build_client()
        owns_client = self._client is None
        try:
            for index_file in await self._index_files(client, resume_after, end_date):
                rows = await self._rows_for_index_date(client, index_file)
                self._index_days_consumed += 1
                yield Batch(
                    rows=rows,
                    checkpoint={CHECKPOINT_LAST_INDEX_DATE: index_file.index_date.isoformat()},
                )
        finally:
            if owns_client:
                await client.aclose()

    def _build_client(self) -> EdgarClient:
        """Build the production HTTP client from the resolved contact string."""
        return EdgarClient(user_agent=self._user_agent)

    def _today_eastern(self) -> dt.date:
        """Return today's date on EDGAR's own calendar (US/Eastern).

        EDGAR's business day, its filing-date rule and its index naming are all
        Eastern; using the UTC date would make the connector skip or re-read a
        day for the five hours a night the two calendars disagree.
        """
        return self.runtime.now().astimezone(EDGAR_TIMEZONE).date()

    def _resume_after(self, checkpoint: Checkpoint | None) -> dt.date | None:
        """Return the last durably-written index date, or ``None`` on a first run."""
        if checkpoint is None:
            return None
        raw = checkpoint.get(CHECKPOINT_LAST_INDEX_DATE)
        if raw is None:
            return None
        if not isinstance(raw, str):
            msg = (
                f"checkpoint {CHECKPOINT_LAST_INDEX_DATE!r} must be an ISO date string; "
                f"got {type(raw).__name__} {raw!r}"
            )
            raise PermanentSourceError(msg)
        try:
            return dt.date.fromisoformat(raw)
        except ValueError as exc:
            msg = f"checkpoint {CHECKPOINT_LAST_INDEX_DATE!r} is not an ISO date: {raw!r}"
            raise PermanentSourceError(msg) from exc

    async def _index_files(
        self, client: EdgarClient, resume_after: dt.date | None, end_date: dt.date
    ) -> list[DailyIndexFile]:
        """Return the daily-index files to consume this run, in date order.

        Walks quarter listings forward from the resume position, taking only
        dates EDGAR actually published, and stops as soon as the per-run cap is
        reached so a bounded run does not fetch listings it will not use.
        """
        first = self._start_date if resume_after is None else resume_after + dt.timedelta(days=1)
        selected: list[DailyIndexFile] = []
        year, quarter = first.year, quarter_of(first)
        while (year, quarter) <= (end_date.year, quarter_of(end_date)):
            listing = parse_quarter_listing(
                await self._get(
                    quarter_listing_url(year, quarter),
                    f"quarter listing {year}Q{quarter}",
                    client=client,
                )
            )
            self._legacy_named_index_files += listing.legacy_named_files
            for index_file in listing.files:
                if first <= index_file.index_date <= end_date:
                    selected.append(index_file)
                if self._max_index_days is not None and len(selected) >= self._max_index_days:
                    return selected
            year, quarter = (year + 1, 1) if quarter == _LAST_QUARTER else (year, quarter + 1)
        return selected

    async def _rows_for_index_date(
        self, client: EdgarClient, index_file: DailyIndexFile
    ) -> tuple[Base, ...]:
        """Fetch and parse one daily index and every filing it selects.

        The header is fetched **once per accession**, not once per index entry:
        a filing listed under seven filer CIKs is one submission with one
        acceptance instant and one document manifest, and EDGAR serves the
        identical header under every one of those CIKs. It then becomes one
        filing row per (accession, CIK) pair — what the index states — and one
        set of document rows.
        """
        body = await self._get(
            daily_index_file_url(
                index_file.index_date.year, quarter_of(index_file.index_date), index_file.file_name
            ),
            f"daily index {index_file.index_date.isoformat()}",
            client=client,
        )
        parsed = parse_daily_index(body)
        self._index_rows_read += parsed.rows_read
        self._duplicate_rows_collapsed += parsed.duplicate_rows_collapsed
        by_accession: dict[str, list[DailyIndexEntry]] = {}
        for entry in parsed.entries:
            if self._form_types is None or entry.form_type in self._form_types:
                by_accession.setdefault(entry.accession_number, []).append(entry)
        selected_entries = sum(len(filers) for filers in by_accession.values())
        self._index_entries_selected += selected_entries
        self._filing_headers_fetched += len(by_accession)
        rows: list[Base] = []
        for accession_number, filers in by_accession.items():
            url = filing_index_headers_url(filers[0].cik, accession_number)
            header = parse_filing_header(
                await self._get(url, f"filing header {accession_number}", client=client),
                accession_number=accession_number,
            )
            self._measure_divergence(header, index_file.index_date)
            rows.extend(self._build_rows(filers, header, index_file.index_date, url))
        _logger.info(
            "ingest.edgar.index_day_parsed",
            source=self.source_name,
            index_date=index_file.index_date.isoformat(),
            index_rows=parsed.rows_read,
            distinct_filer_entries=len(parsed.entries),
            distinct_accessions=parsed.distinct_accessions,
            selected_filer_entries=selected_entries,
            selected_accessions=len(by_accession),
        )
        return tuple(rows)

    async def _get(self, url: str, description: str, *, client: EdgarClient) -> str:
        """Fetch one URL through the framework's pacing and retry.

        Routing every outbound call through
        :meth:`~backend.ingest.base.Connector.request` is what makes the
        declared rate limit binding rather than advisory: the framework paces
        each call against the token bucket and retries only transient failures.
        """
        return await self.request(
            lambda: client.get_text(url, source=self.source_name), description=description
        )

    def _measure_divergence(self, header: FilingHeader, index_date: dt.date) -> None:
        """Record how far the acceptance instant sits from the dissemination date.

        Both counters exist because the acceptance instant is an
        anti-conservative knowledge time exactly when it precedes public
        dissemination (the CORRESP/UPLOAD case in the module docstring). They
        are measurements of what this run actually saw; neither is defaulted.
        """
        acceptance_date = header.acceptance_datetime.astimezone(EDGAR_TIMEZONE).date()
        if acceptance_date < index_date:
            self._accepted_before_index_date += 1
            lag = index_date - acceptance_date
            if self._max_acceptance_to_index_lag is None or lag > self._max_acceptance_to_index_lag:
                self._max_acceptance_to_index_lag = lag
        if header.declared_document_count is not None and header.declared_document_count != len(
            header.documents
        ):
            self._declared_document_count_mismatches += 1

    def _build_rows(
        self,
        filers: Sequence[DailyIndexEntry],
        header: FilingHeader,
        index_date: dt.date,
        source_url: str,
    ) -> list[Base]:
        """Build one filing row per filer plus one document set, from parsed source data.

        Event time is the acceptance instant, open-ended: a filing's existence
        becomes true when EDGAR accepts it and stays true. Knowledge time is
        the same instant — acceptance is when the submission became publicly
        retrievable — which is what makes ``as_of`` before acceptance correctly
        blind to it.

        Args:
            filers: the index entries sharing one accession number, one per
                filer CIK, in index order. Every one gets a filing row, because
                every one is something the source states.
            header: the submission header, parsed once for that accession.
            index_date: the daily index this filing was disseminated in.
            source_url: the ``-index-headers.html`` URL the header came from.

        Returns:
            The filing rows followed by the document rows, in write order.
        """
        acceptance = header.acceptance_datetime
        accession_number = header.accession_number
        rows: list[Base] = [
            EdgarFiling(
                accession_number=accession_number,
                cik=filer.cik,
                company_name=filer.company_name,
                form_type=header.form_type,
                filing_date=header.filing_date,
                index_date=index_date,
                period_of_report=header.period_of_report,
                declared_document_count=header.declared_document_count,
                document_count=len(header.documents),
                source_url=source_url,
                valid_from=acceptance,
                valid_to=INFINITY,
                knowledge_time=acceptance,
            )
            for filer in filers
        ]
        # The archive directory is served under every associated CIK; the first
        # filer's is used so the URL is deterministic rather than arbitrary.
        directory = (
            f"{ARCHIVES_BASE}/edgar/data/{filers[0].cik}/{accession_number.replace('-', '')}"
        )
        rows.extend(
            EdgarFilingDocument(
                accession_number=accession_number,
                document_sequence=document.sequence,
                document_type=document.document_type,
                filename=document.filename,
                description=document.description,
                document_url=f"{directory}/{document.filename}",
                valid_from=acceptance,
                valid_to=INFINITY,
                knowledge_time=acceptance,
            )
            for document in header.documents
        )
        return rows

    async def _write_batch(self, rows: Sequence[Base]) -> None:
        """Commit one index day's rows idempotently, in a single transaction.

        Overrides the framework default for one reason: **a re-run of an index
        day must not fail.** The checkpoint is recorded on the run record only
        after a batch commits, so a process killed between the commit and the
        run record leaves the day written but unrecorded, and the next run
        re-reads it. With the framework's plain ``add_all`` that re-write
        collides with the append-only primary key and the source is stuck until
        someone intervenes. ``ON CONFLICT DO NOTHING`` on the full primary key
        makes the re-write a no-op instead — and it can only ever suppress a
        row that is byte-identical in its versioning coordinates, since a
        genuine correction carries a later ``knowledge_time`` and therefore a
        different key.

        The framework's ``before_flush`` knowledge-time guard does **not** see
        Core DML, so this method calls
        :func:`~backend.ingest.write.validate_knowledge_time` on every row
        explicitly. That is a real gap in the base-class contract, whose
        docstring states an override "cannot skip" the guard; it is reported
        rather than worked around silently.
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
            for model, values in _group_insert_values(rows):
                table = cast("Table", sa.inspect(model).local_table)
                await session.execute(pg_insert(table).on_conflict_do_nothing(), values)
            await session.commit()

    def _metrics(
        self,
        rows_written: int,
        batches_written: int,
        lag_report: KnowledgeTimeLagReport | None,
    ) -> list[DataQualityMetric]:
        """Extend the framework metrics with what this run measured at EDGAR.

        Every entry is a count or a duration observed while running. The
        acceptance-to-index lag is ``None`` when no filing in the run was
        accepted before its dissemination date — absence of the phenomenon, not
        a measured zero.
        """
        metrics = super()._metrics(rows_written, batches_written, lag_report)
        metrics.extend(
            [
                DataQualityMetric("index_days_consumed", self._index_days_consumed, "days"),
                DataQualityMetric("index_rows_read", self._index_rows_read, "rows"),
                DataQualityMetric(
                    "index_duplicate_rows_collapsed", self._duplicate_rows_collapsed, "rows"
                ),
                DataQualityMetric(
                    "index_entries_selected", self._index_entries_selected, "entries"
                ),
                DataQualityMetric(
                    "filing_headers_fetched", self._filing_headers_fetched, "accessions"
                ),
                DataQualityMetric(
                    "legacy_named_index_files_skipped", self._legacy_named_index_files, "files"
                ),
                DataQualityMetric(
                    "filings_accepted_before_index_date",
                    self._accepted_before_index_date,
                    "filings",
                ),
                DataQualityMetric(
                    "max_acceptance_to_index_lag",
                    (
                        None
                        if self._max_acceptance_to_index_lag is None
                        else self._max_acceptance_to_index_lag.days
                    ),
                    "days",
                ),
                DataQualityMetric(
                    "declared_document_count_mismatches",
                    self._declared_document_count_mismatches,
                    "filings",
                ),
            ]
        )
        return metrics


def _group_insert_values(rows: Sequence[Base]) -> list[tuple[type[Base], list[dict[str, Any]]]]:
    """Group ORM instances by model and render them as uniform INSERT parameter dicts.

    Args:
        rows: transient ORM instances built by the connector, in write order.

    Returns:
        ``[(model, [values, ...]), ...]`` in first-appearance order, so filings
        insert before the documents that reference them.

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
                f"({sorted(set(bucket[0]) ^ set(values))}); an executemany cannot express that "
                "and filling the difference with NULL would write values the source never gave"
            )
            raise ValueError(msg)
        bucket.append(values)
    return list(grouped.items())
