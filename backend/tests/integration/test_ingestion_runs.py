"""P3.1 integration: the ingestion-run table against real TimescaleDB.

Covers the run lifecycle, the checkpoint round-trip through ``JSONB`` (types
preserved exactly — a checkpoint that changes type between runs silently moves
the resume position), the resume-position query, and the database CHECK
constraints that encode the lifecycle rather than trusting application code to
maintain it.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from backend.db import ingest_writer_session
from backend.db._guard import fact_table_names
from backend.db.bitemporal import BitemporalMixin
from backend.ingest.checkpoint import JsonScalar
from backend.ingest.errors import RunRecordError
from backend.ingest.runs import (
    IngestionRun,
    RunKind,
    RunStatus,
    finish_run,
    get_run,
    latest_checkpoint,
    start_run,
)

_SOURCE = "probe_source"


async def _insert_raw(**values: object) -> None:
    """Insert a run row bypassing the repository, to exercise DB constraints."""
    async with ingest_writer_session() as session:
        await session.execute(sa.insert(IngestionRun).values(**values))
        await session.commit()


# --- lifecycle --------------------------------------------------------------


async def test_a_started_run_is_open_and_records_its_kind() -> None:
    run_id = await start_run(
        source=_SOURCE, run_kind=RunKind.BACKFILL, checkpoint_before={"page": 1}
    )
    run = await get_run(run_id)
    assert run.source == _SOURCE
    assert run.run_kind == RunKind.BACKFILL.value
    assert run.status == RunStatus.RUNNING.value
    assert run.ended_at is None
    assert run.rows_written == 0
    assert run.checkpoint_before == {"page": 1}
    assert run.checkpoint_after is None
    assert run.started_at.tzinfo is not None


async def test_a_first_run_has_no_checkpoint_before() -> None:
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    assert (await get_run(run_id)).checkpoint_before is None


async def test_finishing_a_run_records_the_outcome() -> None:
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(
        run_id,
        status=RunStatus.SUCCEEDED,
        rows_written=17,
        checkpoint_after={"page": 2},
        quality_metrics={"rows_written": {"value": 17, "unit": "rows"}},
    )
    run = await get_run(run_id)
    assert run.status == RunStatus.SUCCEEDED.value
    assert run.rows_written == 17
    assert run.checkpoint_after == {"page": 2}
    assert run.error_detail is None
    assert run.ended_at is not None
    assert run.ended_at >= run.started_at
    assert run.quality_metrics == {"rows_written": {"value": 17, "unit": "rows"}}


async def test_a_failed_run_records_its_error_and_partial_progress() -> None:
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(
        run_id,
        status=RunStatus.FAILED,
        rows_written=3,
        checkpoint_after={"page": 1},
        error_detail="TransientSourceError: gateway timeout",
    )
    run = await get_run(run_id)
    assert run.status == RunStatus.FAILED.value
    assert run.rows_written == 3
    assert run.checkpoint_after == {"page": 1}
    assert "gateway timeout" in (run.error_detail or "")


async def test_finishing_a_run_twice_is_refused() -> None:
    """A closed run is closed; a second transition would rewrite history."""
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(run_id, status=RunStatus.SUCCEEDED, rows_written=1)
    with pytest.raises(RunRecordError, match="already finished"):
        await finish_run(run_id, status=RunStatus.FAILED, rows_written=0, error_detail="late")


async def test_finishing_an_unknown_run_is_refused() -> None:
    with pytest.raises(RunRecordError):
        await finish_run(987654321, status=RunStatus.SUCCEEDED, rows_written=0)


async def test_a_failed_run_must_carry_an_explanation() -> None:
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    with pytest.raises(ValueError, match="must record error_detail"):
        await finish_run(run_id, status=RunStatus.FAILED, rows_written=0)


async def test_finish_cannot_reopen_a_run() -> None:
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    with pytest.raises(ValueError, match="cannot set status back"):
        await finish_run(run_id, status=RunStatus.RUNNING, rows_written=0)


async def test_get_run_raises_for_an_unknown_id() -> None:
    with pytest.raises(RunRecordError, match="no ingestion run"):
        await get_run(4242)


# --- checkpoint round-trip --------------------------------------------------


async def test_checkpoint_scalar_types_survive_the_jsonb_round_trip() -> None:
    """Type drift across the round-trip is how a resume position silently moves."""
    checkpoint: dict[str, JsonScalar] = {
        "last_index_date": "2024-01-05",
        "documents_seen": 412,
        "coverage_fraction": 0.98,
        "complete": True,
        "next_cursor": None,
    }
    run_id = await start_run(
        source=_SOURCE, run_kind=RunKind.BACKFILL, checkpoint_before=checkpoint
    )
    await finish_run(
        run_id, status=RunStatus.SUCCEEDED, rows_written=0, checkpoint_after=checkpoint
    )
    stored = (await get_run(run_id)).checkpoint_after
    assert stored == checkpoint
    assert stored is not None
    assert isinstance(stored["documents_seen"], int)
    assert isinstance(stored["coverage_fraction"], float)
    assert isinstance(stored["complete"], bool)
    assert stored["next_cursor"] is None


async def test_latest_checkpoint_is_none_for_a_source_that_never_ran() -> None:
    assert await latest_checkpoint("probe_never_ran") is None


async def test_latest_checkpoint_returns_the_most_recent_recorded_position() -> None:
    for page in (1, 2, 3):
        run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
        await finish_run(
            run_id,
            status=RunStatus.SUCCEEDED,
            rows_written=1,
            checkpoint_after={"page": page},
        )
    assert await latest_checkpoint(_SOURCE) == {"page": 3}


async def test_latest_checkpoint_ignores_runs_that_recorded_none() -> None:
    """A run that committed nothing must not erase the last known position."""
    first = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(
        first, status=RunStatus.SUCCEEDED, rows_written=5, checkpoint_after={"page": 7}
    )
    second = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(second, status=RunStatus.SUCCEEDED, rows_written=0)
    assert await latest_checkpoint(_SOURCE) == {"page": 7}


async def test_latest_checkpoint_honors_a_failed_runs_durable_progress() -> None:
    """Rows a failed run committed are real; re-fetching them would collide."""
    first = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(
        first, status=RunStatus.SUCCEEDED, rows_written=5, checkpoint_after={"page": 1}
    )
    second = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(
        second,
        status=RunStatus.FAILED,
        rows_written=2,
        checkpoint_after={"page": 2},
        error_detail="TransientSourceError: reset",
    )
    assert await latest_checkpoint(_SOURCE) == {"page": 2}


async def test_latest_checkpoint_is_scoped_per_source() -> None:
    run_id = await start_run(source="probe_a", run_kind=RunKind.LIVE)
    await finish_run(
        run_id, status=RunStatus.SUCCEEDED, rows_written=1, checkpoint_after={"page": 9}
    )
    assert await latest_checkpoint("probe_b") is None


# --- database-level lifecycle constraints -----------------------------------


async def test_database_rejects_an_unknown_run_kind() -> None:
    """The backfill/live distinction is D-011's compensating control; not free text."""
    with pytest.raises(IntegrityError, match="run_kind"):
        await _insert_raw(source=_SOURCE, run_kind="whenever", status="running")


async def test_database_rejects_an_unknown_status() -> None:
    with pytest.raises(IntegrityError, match="status"):
        await _insert_raw(source=_SOURCE, run_kind="live", status="probably_fine")


async def test_database_rejects_a_finished_run_with_no_end_time() -> None:
    with pytest.raises(IntegrityError, match="running_iff_open"):
        await _insert_raw(source=_SOURCE, run_kind="live", status="succeeded")


async def test_database_rejects_a_running_run_that_already_ended() -> None:
    """started_at/ended_at are given explicitly so only running_iff_open is violated."""
    with pytest.raises(IntegrityError, match="running_iff_open"):
        await _insert_raw(
            source=_SOURCE,
            run_kind="live",
            status="running",
            started_at=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            ended_at=dt.datetime(2020, 1, 2, tzinfo=dt.UTC),
        )


async def test_database_rejects_an_end_before_the_start() -> None:
    with pytest.raises(IntegrityError, match="end_after_start"):
        await _insert_raw(
            source=_SOURCE,
            run_kind="live",
            status="succeeded",
            started_at=dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC),
            ended_at=dt.datetime(2026, 8, 1, 11, 0, tzinfo=dt.UTC),
        )


async def test_database_rejects_negative_rows_written() -> None:
    with pytest.raises(IntegrityError, match="rows_written"):
        await _insert_raw(
            source=_SOURCE,
            run_kind="live",
            status="running",
            rows_written=-1,
        )


async def test_database_check_names_match_the_model_declaration() -> None:
    """Migration/model drift guard: the DDL that ran names what the model names.

    Alembic applies the metadata naming convention
    (``ck_%(table_name)s_%(constraint_name)s``) to explicitly named CHECK
    constraints too, so a migration passing an already-prefixed name silently
    creates ``ck_ingestion_run_ck_ingestion_run_...``. That drift is invisible
    until an operator reads a constraint-violation message, so it is asserted.
    """
    declared = {
        constraint.name
        for constraint in cast("sa.Table", IngestionRun.__table__).constraints
        if isinstance(constraint, sa.CheckConstraint)
    }
    async with ingest_writer_session() as session:
        result = await session.execute(
            sa.text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'ingestion_run'::regclass AND contype = 'c'"
            )
        )
        actual = set(result.scalars().all())
    assert actual == declared


def test_the_run_table_is_not_bitemporal() -> None:
    """Operational metadata must stay outside the as-of read path (module docstring)."""
    assert not issubclass(IngestionRun, BitemporalMixin)
    assert not hasattr(IngestionRun, "knowledge_time")
    assert IngestionRun.__tablename__ not in fact_table_names()


async def test_a_run_row_can_be_updated_unlike_a_fact_row() -> None:
    """Fact tables carry append-only triggers; this table deliberately does not."""
    run_id = await start_run(source=_SOURCE, run_kind=RunKind.LIVE)
    await finish_run(run_id, status=RunStatus.SUCCEEDED, rows_written=1)
    assert (await get_run(run_id)).status == RunStatus.SUCCEEDED.value
