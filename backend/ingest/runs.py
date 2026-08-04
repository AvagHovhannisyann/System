"""Ingestion-run tracking: the operational log of every connector execution (P3.1).

Every connector execution opens a row here before touching a source and closes
it when it finishes or fails. The row answers, for an operator and for the
data-quality report (P3.9): which source, when, how it ended, how many rows it
wrote, where it resumed from and where it got to, what went wrong, and —
crucially for D-011 — whether it was a ``backfill`` or a ``live`` run.

Why ``run_kind`` matters
------------------------

D-011 deliberately imposes **no** database constraint relating
``knowledge_time`` to ``ingested_at``, because backfills legitimately break any
such relation: a 2016 filing ingested today carries an honest 2016 knowledge
time, and a CHECK forbidding that would make historical loading impossible.
The compensating control D-011 promises in its place is exactly this column
plus the live-run lag check in :mod:`backend.ingest.quality`: a live run's
knowledge times *should* track ingestion closely, and when they do not the run
is flagged. Without ``run_kind`` recorded per run there is no way to tell a
legitimate historical load from a live feed that has silently fallen a day
behind — and the second is invisible corruption of point-in-time integrity.

Why this table is **not** a bitemporal fact table
-------------------------------------------------

It carries no :class:`~backend.db.bitemporal.BitemporalMixin`, deliberately:

1. **It is not a fact about the world.** The bitemporal columns describe when
   a fact *was true* (``valid_from``/``valid_to``) and when it *became knowable
   to the market* (``knowledge_time``). "We started a job at 04:00" has no
   market knowability; any ``knowledge_time`` we invented for it would be a
   fabricated number in a column whose entire meaning is that it is not
   fabricated (I3).
2. **Its rows must mutate.** A run legitimately transitions ``running`` →
   ``succeeded``/``failed``, which is a *state change of an operational
   record*, not a later correction to a historical belief. The append-only
   triggers on fact tables exist precisely to forbid that, and the
   supersession machinery that models corrections
   (:mod:`backend.ingest.supersession`) would turn a two-field status update
   into a second row that no operator query wants.
3. **It must be readable without an as-of.** Bitemporal tables can only be
   read through :func:`backend.db.as_of`, which by design refuses future
   timestamps and returns the world as believed at an instant. Operational
   questions — "is a run in flight right now?", "what did last night's run
   fail on?" — are questions about the present state of our own pipeline; they
   have no as-of and must not acquire a fake one.

Concretely, it is therefore **not** in the bitemporal registry, so the
Core-level read guard does not scope it, and ordinary reads and updates go
through the ingestion writer session like any non-fact table.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from sqlalchemy import CursorResult
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import ingest_writer_session
from backend.db.base import Base
from backend.ingest.checkpoint import JsonScalar, normalize_checkpoint
from backend.ingest.errors import RunRecordError

if TYPE_CHECKING:
    from backend.ingest.checkpoint import Checkpoint

__all__ = [
    "IngestionRun",
    "RunKind",
    "RunStatus",
    "finish_run",
    "get_run",
    "latest_checkpoint",
    "start_run",
]


class RunKind(StrEnum):
    """Whether a run loads history or keeps up with a live feed (D-011).

    ``BACKFILL`` runs write knowledge times far in the past on purpose and are
    exempt from the live-lag check. ``LIVE`` runs are expected to write
    knowledge times close to now, and are flagged when they do not.
    """

    BACKFILL = "backfill"
    LIVE = "live"


class RunStatus(StrEnum):
    """Lifecycle state of an ingestion run.

    ``RUNNING`` is set at start and is the only state with a NULL ``ended_at``.
    A process killed mid-run leaves its row ``RUNNING`` forever — deliberately
    visible as an unfinished run rather than silently rewritten to a
    conclusion nobody observed.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _sql_in_list(values: type[StrEnum]) -> str:
    """Render a ``StrEnum``'s values as a SQL ``IN`` list literal."""
    return ", ".join(f"'{member.value}'" for member in values)


class IngestionRun(Base):
    """One execution of one connector: operational metadata, not a fact.

    See the module docstring for why this table carries no bitemporal mixin.
    All timestamps are ``TIMESTAMPTZ`` in UTC; ``rows_written`` counts
    database rows (dimensionless).
    """

    __tablename__ = "ingestion_run"
    __table_args__ = (
        sa.CheckConstraint(f"run_kind IN ({_sql_in_list(RunKind)})", name="run_kind"),
        sa.CheckConstraint(f"status IN ({_sql_in_list(RunStatus)})", name="status"),
        sa.CheckConstraint("rows_written >= 0", name="rows_written_non_negative"),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="end_after_start"),
        sa.CheckConstraint(
            f"(status = '{RunStatus.RUNNING.value}') = (ended_at IS NULL)",
            name="running_iff_open",
        ),
        sa.Index("ix_ingestion_run_source_started", "source", sa.text("started_at DESC")),
    )

    run_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.Identity(),
        primary_key=True,
        doc="Surrogate run key, database-generated. Dimensionless.",
    )
    source: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Connector source name (Connector.source_name), e.g. 'sec_edgar'.",
    )
    run_kind: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="'backfill' or 'live' — D-011's compensating control for knowledge-time lag.",
    )
    status: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="'running' | 'succeeded' | 'failed'. 'running' iff ended_at IS NULL.",
    )
    started_at: Mapped[dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
        doc="When the run opened, UTC (database clock).",
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=True,
        doc="When the run finished, UTC; NULL while running.",
    )
    rows_written: Mapped[int] = mapped_column(
        sa.BigInteger,
        nullable=False,
        server_default=sa.text("0"),
        doc="Rows durably committed by this run (count).",
    )
    checkpoint_before: Mapped[dict[str, JsonScalar] | None] = mapped_column(
        # none_as_null: without it SQLAlchemy stores Python None as JSON 'null',
        # which is *not* SQL NULL — "checkpoint_after IS NOT NULL" would then
        # match a run that recorded no position and latest_checkpoint() would
        # return None for a source that has a perfectly good one.
        JSONB(none_as_null=True),
        nullable=True,
        doc="Resume position the run started from; SQL NULL on a source's first run.",
    )
    checkpoint_after: Mapped[dict[str, JsonScalar] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
        doc="Furthest durably-written position reached; SQL NULL if no batch committed.",
    )
    error_detail: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
        doc="Failure detail (exception type and message); NULL unless status='failed'.",
    )
    quality_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
        doc="Emitted data-quality metrics as {name: {value, unit}} (P3.9 consumes this).",
    )


async def start_run(
    *,
    source: str,
    run_kind: RunKind,
    checkpoint_before: Checkpoint | None = None,
) -> int:
    """Open a ``running`` run record and return its id.

    Committed immediately and in its own transaction, so a process that dies
    mid-run leaves visible evidence that a run started and never finished.

    Args:
        source: connector source name.
        run_kind: ``BACKFILL`` or ``LIVE`` (D-011 compensating control).
        checkpoint_before: resume position handed to the connector, or
            ``None`` for a source that has never run.

    Returns:
        The database-generated ``run_id``.

    Raises:
        TypeError: if ``checkpoint_before`` is not a flat JSON-scalar mapping.
    """
    stored_before = None if checkpoint_before is None else normalize_checkpoint(checkpoint_before)
    async with ingest_writer_session() as session:
        run = IngestionRun(
            source=source,
            run_kind=run_kind.value,
            status=RunStatus.RUNNING.value,
            rows_written=0,
            checkpoint_before=stored_before,
            quality_metrics={},
        )
        session.add(run)
        await session.flush()
        run_id = run.run_id
        await session.commit()
    return run_id


async def finish_run(
    run_id: int,
    *,
    status: RunStatus,
    rows_written: int,
    checkpoint_after: Checkpoint | None = None,
    error_detail: str | None = None,
    quality_metrics: dict[str, Any] | None = None,
    ended_at: dt.datetime | None = None,
) -> None:
    """Close a ``running`` run record, recording its outcome.

    Args:
        run_id: id returned by :func:`start_run`.
        status: ``SUCCEEDED`` or ``FAILED``.
        rows_written: rows durably committed by the run (count, >= 0).
        checkpoint_after: furthest durably-written position, or ``None`` if no
            batch committed. Recorded even for a failed run, because the rows
            it did write are real and re-fetching them would collide with the
            append-only primary key.
        error_detail: failure detail; required when ``status`` is ``FAILED``.
            Must contain no secrets (invariant I5).
        quality_metrics: rendered metrics (:func:`~backend.ingest.quality.metrics_as_json`).
        ended_at: override for the end instant (timezone-aware UTC); defaults
            to now. Injected by tests.

    Raises:
        ValueError: if ``status`` is ``RUNNING`` (finishing into the open
            state is meaningless), if ``rows_written`` is negative, or if a
            failed run carries no ``error_detail`` — an unexplained failure in
            the operational log is a failure to record, not a record.
        RunRecordError: if no ``running`` row with this id exists (unknown id,
            or the run was already finished).
        TypeError: if ``checkpoint_after`` is not a flat JSON-scalar mapping.
    """
    if status is RunStatus.RUNNING:
        msg = "finish_run cannot set status back to 'running'"
        raise ValueError(msg)
    if rows_written < 0:
        msg = f"rows_written must be >= 0; got {rows_written}"
        raise ValueError(msg)
    if status is RunStatus.FAILED and not error_detail:
        msg = "a failed run must record error_detail; an unexplained failure is not a record"
        raise ValueError(msg)
    stored_after = None if checkpoint_after is None else normalize_checkpoint(checkpoint_after)
    end_instant = ended_at if ended_at is not None else dt.datetime.now(dt.UTC)
    async with ingest_writer_session() as session:
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                sa.update(IngestionRun)
                .where(
                    IngestionRun.run_id == run_id,
                    IngestionRun.status == RunStatus.RUNNING.value,
                )
                .values(
                    status=status.value,
                    ended_at=end_instant,
                    rows_written=rows_written,
                    checkpoint_after=stored_after,
                    error_detail=error_detail,
                    quality_metrics=quality_metrics if quality_metrics is not None else {},
                )
            ),
        )
        if result.rowcount != 1:
            await session.rollback()
            msg = (
                f"cannot finish run {run_id}: no run with that id is in state "
                f"'{RunStatus.RUNNING.value}' (unknown id, or already finished)"
            )
            raise RunRecordError(msg)
        await session.commit()


async def get_run(run_id: int) -> IngestionRun:
    """Return one run record by id.

    Args:
        run_id: id returned by :func:`start_run`.

    Returns:
        The :class:`IngestionRun` row, detached from its session.

    Raises:
        RunRecordError: if no run with that id exists.
    """
    async with ingest_writer_session() as session:
        run = await session.get(IngestionRun, run_id)
        if run is None:
            msg = f"no ingestion run with id {run_id}"
            raise RunRecordError(msg)
        return run


async def latest_checkpoint(source: str) -> dict[str, JsonScalar] | None:
    """Return the resume position a new run of ``source`` should start from.

    Defined as the ``checkpoint_after`` of the most recent run of that source
    which recorded one, newest first by ``started_at`` then ``run_id``.

    Failed runs count. That is deliberate: a run that committed three batches
    and then died wrote three batches of real rows, and the fact tables are
    append-only, so re-fetching them would collide on the primary key rather
    than silently duplicate. Resuming from the furthest durably-written
    position is both cheaper and the only thing the store will accept.

    Args:
        source: connector source name.

    Returns:
        The stored checkpoint mapping, or ``None`` when the source has never
        recorded one (first-ever run).
    """
    async with ingest_writer_session() as session:
        result = await session.execute(
            sa.select(IngestionRun.checkpoint_after)
            .where(
                IngestionRun.source == source,
                IngestionRun.checkpoint_after.is_not(None),
            )
            .order_by(IngestionRun.started_at.desc(), IngestionRun.run_id.desc())
            .limit(1)
        )
        return result.scalars().one_or_none()
