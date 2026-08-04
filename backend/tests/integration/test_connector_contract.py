"""P3.1 + CC.8 integration: the connector base class against real TimescaleDB.

Two halves:

- **CC.8, the no-fabrication contract.** Test doubles simulating an
  unavailable source are run through the real framework and the real database,
  and the contract in ``connector_contract.py`` asserts the failure shape:
  the exception propagates, no rows exist, the run is recorded ``failed``, and
  no :class:`~backend.ingest.base.RunResult` is produced. Two doubles are used
  — one that fails before writing anything and one that fails *after* a
  committed batch — because a framework can pass the first and still quietly
  report a partial run as a success.
- **the run orchestration** the base class provides around every connector:
  checkpointed resume, per-batch atomicity, retry accounting, and the live-run
  knowledge-time lag flag (D-011's compensating control).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, cast

import pytest
from sqlalchemy import select

from backend.db import as_of
from backend.db.models import PriceBar
from backend.ingest.base import Batch, Connector, ConnectorRuntime
from backend.ingest.errors import (
    PermanentSourceError,
    SourceUnavailableError,
    TransientSourceError,
)
from backend.ingest.quality import KnowledgeTimePolicy
from backend.ingest.ratelimit import RateLimit
from backend.ingest.retry import RetryPolicy
from backend.ingest.runs import RunKind, RunStatus
from backend.tests.ingest.clock import ControlledClock
from backend.tests.integration.connector_contract import (
    ConnectorContractTests,
    latest_run,
    stored_fact_row_count,
)
from backend.tests.integration.factories import create_security

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from backend.ingest.checkpoint import Checkpoint

_DAY = dt.datetime(2024, 1, 4, tzinfo=dt.UTC)
_HISTORIC_KNOWLEDGE = dt.datetime(2024, 1, 4, 21, 0, tzinfo=dt.UTC)
_POLICY = KnowledgeTimePolicy(
    description="probe: knowledge_time is the batch's stated availability instant",
    max_live_lag=dt.timedelta(hours=2),
)
_LIMIT = RateLimit(requests_per_second=1000.0, burst=1000)


def _bar(security_id: int, day_offset: int, knowledge_time: dt.datetime) -> PriceBar:
    """Build one PriceBar; prices USD/share, volume shares, factor dimensionless."""
    valid_from = _DAY + dt.timedelta(days=day_offset)
    price = Decimal("100")
    return PriceBar(
        security_id=security_id,
        valid_from=valid_from,
        valid_to=valid_from + dt.timedelta(days=1),
        knowledge_time=knowledge_time,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal("1"),
        volume_shares=1000,
    )


class _ProbeConnector(Connector):
    """Shared declaration for the doubles below; each overrides fetch_batches."""

    source_name = "probe_contract"
    knowledge_time_policy = _POLICY
    rate_limit = _LIMIT
    retry_policy = RetryPolicy(max_attempts=2, initial_backoff_s=0.01, max_backoff_s=0.02)


class UnreachableSourceConnector(_ProbeConnector):
    """A source that cannot be reached at all: raises before yielding anything."""

    source_name = "probe_unreachable"

    async def fetch_batches(
        self,
        checkpoint: Checkpoint | None,  # noqa: ARG002 — the source is never reached
    ) -> AsyncIterator[Batch]:
        """Raise on the first iteration, as an unreachable source does."""
        for _ in ():
            yield Batch(rows=(), checkpoint={})
        msg = "probe: connection refused"
        raise TransientSourceError(msg)


class FailsAfterOneBatchConnector(_ProbeConnector):
    """A source that dies part-way: one batch commits, then it becomes unavailable."""

    source_name = "probe_partial"

    def __init__(self, security_id: int, runtime: ConnectorRuntime | None = None) -> None:
        """Store the anchor id the yielded rows reference."""
        super().__init__(runtime)
        self.security_id = security_id

    async def fetch_batches(
        self,
        checkpoint: Checkpoint | None,  # noqa: ARG002 — always starts from the beginning
    ) -> AsyncIterator[Batch]:
        """Yield one good batch, then fail."""
        yield Batch(
            rows=(_bar(self.security_id, 0, _HISTORIC_KNOWLEDGE),),
            checkpoint={"day_offset": 0},
        )
        msg = "probe: 404 no such index file"
        raise PermanentSourceError(msg)


class WorkingConnector(_ProbeConnector):
    """A source that yields two batches successfully."""

    source_name = "probe_working"

    def __init__(
        self,
        security_id: int,
        *,
        knowledge_time: dt.datetime = _HISTORIC_KNOWLEDGE,
        runtime: ConnectorRuntime | None = None,
    ) -> None:
        """Store the anchor id and the knowledge time every yielded row carries."""
        super().__init__(runtime)
        self.security_id = security_id
        self.knowledge_time = knowledge_time
        self.seen_checkpoint: Checkpoint | None = None
        self.checkpoint_was_seen = False

    async def fetch_batches(self, checkpoint: Checkpoint | None) -> AsyncIterator[Batch]:
        """Yield two single-row batches, resuming after any recorded checkpoint."""
        self.seen_checkpoint = checkpoint
        self.checkpoint_was_seen = True
        start = 0 if checkpoint is None else cast("int", checkpoint["day_offset"]) + 1
        for offset in (start, start + 1):
            yield Batch(
                rows=(_bar(self.security_id, offset, self.knowledge_time),),
                checkpoint={"day_offset": offset},
            )


class FlakyThenWorkingConnector(_ProbeConnector):
    """A source that fails transiently once per batch before succeeding."""

    source_name = "probe_flaky"

    def __init__(self, security_id: int, runtime: ConnectorRuntime | None = None) -> None:
        """Store the anchor id and the per-batch attempt counter."""
        super().__init__(runtime)
        self.security_id = security_id
        self.attempts = 0

    async def _fetch_one(self, offset: int) -> Batch:
        self.attempts += 1
        if self.attempts % 2 == 1:
            msg = "probe: 503 service unavailable"
            raise TransientSourceError(msg)
        return Batch(
            rows=(_bar(self.security_id, offset, _HISTORIC_KNOWLEDGE),),
            checkpoint={"day_offset": offset},
        )

    async def fetch_batches(
        self,
        checkpoint: Checkpoint | None,  # noqa: ARG002 — always starts from the beginning
    ) -> AsyncIterator[Batch]:
        """Yield two batches, each of which needs one retry to arrive."""
        for offset in (0, 1):
            yield await self.request(
                lambda offset=offset: self._fetch_one(offset),  # type: ignore[misc]
                description=f"probe batch {offset}",
            )


# --- CC.8: the base-class no-fabrication contract ---------------------------


class TestUnreachableSourceContract(ConnectorContractTests):
    """CC.8 against a source that cannot be reached at all."""

    expected_error: ClassVar[type[SourceUnavailableError]] = TransientSourceError

    def build_connector(self) -> Connector:
        """Return the unreachable-source double."""
        return UnreachableSourceConnector()


class TestPartialFailureContract(ConnectorContractTests):
    """CC.8 against a source that fails *after* committing a batch.

    The harder case: a framework that quietly reported the partial run as a
    success would pass the unreachable-source contract and fail here.
    """

    expected_error: ClassVar[type[SourceUnavailableError]] = PermanentSourceError
    security_id: int

    @pytest.fixture(autouse=True)
    async def _anchor(self) -> None:
        """Create the identity anchor the yielded rows reference."""
        type(self).security_id = await create_security()

    def build_connector(self) -> Connector:
        """Return the partial-failure double."""
        return FailsAfterOneBatchConnector(type(self).security_id)

    async def test_unavailable_source_writes_no_rows(self) -> None:
        """Overridden: this double *does* commit its first batch, honestly.

        The contract's general claim ("nothing is written") is about a source
        that never delivered anything. Here one batch genuinely arrived and was
        genuinely committed — discarding it would be the dishonesty, not
        keeping it. What must hold is that the rows present are exactly the
        ones that really arrived, and that the run is still recorded failed.
        """
        before = await stored_fact_row_count()
        with pytest.raises(self.expected_error):
            await self.build_connector().run(RunKind.LIVE)
        assert await stored_fact_row_count() == before + 1

    async def test_unavailable_source_records_the_run_as_failed(self) -> None:
        """Overridden: rows_written reflects the batch that really committed."""
        connector = self.build_connector()
        with pytest.raises(self.expected_error):
            await connector.run(RunKind.LIVE)
        run = await latest_run(connector.source_name)
        assert run.status == RunStatus.FAILED.value
        assert run.rows_written == 1
        assert "PermanentSourceError" in (run.error_detail or "")
        assert run.checkpoint_after == {"day_offset": 0}


# --- run orchestration ------------------------------------------------------


async def test_a_successful_run_writes_rows_and_records_its_checkpoint() -> None:
    security_id = await create_security()
    connector = WorkingConnector(security_id)
    result = await connector.run(RunKind.BACKFILL)

    assert result.rows_written == 2
    assert result.batches_written == 2
    assert result.checkpoint_before is None
    assert result.checkpoint_after == {"day_offset": 1}

    run = await latest_run(connector.source_name)
    assert run.status == RunStatus.SUCCEEDED.value
    assert run.run_kind == RunKind.BACKFILL.value
    assert run.rows_written == 2
    assert run.error_detail is None
    assert run.checkpoint_after == {"day_offset": 1}


async def test_a_second_run_resumes_from_the_recorded_checkpoint() -> None:
    """Incremental sync: the framework hands back what the last run reached."""
    security_id = await create_security()
    await WorkingConnector(security_id).run(RunKind.BACKFILL)

    second = WorkingConnector(security_id)
    result = await second.run(RunKind.BACKFILL)

    assert second.seen_checkpoint == {"day_offset": 1}
    assert result.checkpoint_before == {"day_offset": 1}
    assert result.checkpoint_after == {"day_offset": 3}
    assert await stored_fact_row_count() == 4


async def test_a_first_ever_run_is_handed_no_checkpoint() -> None:
    """None must mean 'start from the beginning', never 'start from now'."""
    security_id = await create_security()
    connector = WorkingConnector(security_id)
    await connector.run(RunKind.BACKFILL)
    assert connector.checkpoint_was_seen
    assert connector.seen_checkpoint is None


async def test_written_rows_are_readable_through_the_as_of_layer() -> None:
    security_id = await create_security()
    await WorkingConnector(security_id).run(RunKind.BACKFILL)
    async with as_of(dt.datetime.now(dt.UTC)) as session:
        bars = list((await session.scalars(select(PriceBar).order_by(PriceBar.valid_from))).all())
    assert [bar.valid_from for bar in bars] == [_DAY, _DAY + dt.timedelta(days=1)]
    assert all(bar.knowledge_time == _HISTORIC_KNOWLEDGE for bar in bars)


async def test_emitted_metrics_are_recorded_on_the_run() -> None:
    """Every number carries its unit; the metrics are measured, not defaulted."""
    security_id = await create_security()
    connector = WorkingConnector(security_id)
    await connector.run(RunKind.BACKFILL)
    run = await latest_run(connector.source_name)
    assert run.quality_metrics["rows_written"] == {"value": 2, "unit": "rows"}
    assert run.quality_metrics["batches_written"] == {"value": 2, "unit": "batches"}
    assert run.quality_metrics["transient_retries"] == {"value": 0, "unit": "count"}
    assert run.quality_metrics["rate_limit_wait"]["unit"] == "seconds"


async def test_transient_failures_are_retried_and_counted() -> None:
    security_id = await create_security()
    clock = ControlledClock()
    connector = FlakyThenWorkingConnector(
        security_id,
        ConnectorRuntime(sleep=clock.sleep, monotonic=clock.monotonic, jitter=lambda c: c),
    )
    result = await connector.run(RunKind.BACKFILL)

    assert result.rows_written == 2
    assert connector.transient_retries == 2
    run = await latest_run(connector.source_name)
    assert run.quality_metrics["transient_retries"] == {"value": 2, "unit": "count"}


# --- D-011 compensating control: the live-run knowledge-time lag flag -------


async def test_a_live_run_within_the_declared_lag_is_not_flagged() -> None:
    security_id = await create_security()
    fresh = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)
    connector = WorkingConnector(security_id, knowledge_time=fresh)
    result = await connector.run(RunKind.LIVE)

    assert result.lag_report is not None
    assert not result.lag_report.breached
    run = await latest_run(connector.source_name)
    assert run.quality_metrics["knowledge_time_lag_breach"]["value"] is False


async def test_a_live_run_trailing_the_declared_lag_is_flagged_but_still_succeeds() -> None:
    """Flag, do not reject: the data is real, the source's timeliness is not.

    D-011's compensating control. Dropping a lagging feed's rows would discard
    real data; ignoring the lag is how "point in time" silently becomes "point
    in time, minus a day".
    """
    security_id = await create_security()
    connector = WorkingConnector(security_id, knowledge_time=_HISTORIC_KNOWLEDGE)
    result = await connector.run(RunKind.LIVE)

    assert result.lag_report is not None
    assert result.lag_report.breached
    assert result.lag_report.observed_max_lag is not None
    assert result.lag_report.declared_max_lag == dt.timedelta(hours=2)
    assert result.rows_written == 2

    run = await latest_run(connector.source_name)
    assert run.status == RunStatus.SUCCEEDED.value
    assert run.quality_metrics["knowledge_time_lag_breach"]["value"] is True
    assert run.quality_metrics["knowledge_time_max_lag"]["unit"] == "seconds"
    assert run.quality_metrics["knowledge_time_declared_max_lag"]["value"] == 7200.0


async def test_a_backfill_run_with_ancient_knowledge_times_is_not_flagged() -> None:
    """A backfill's lag is correct by construction; flagging it would be noise."""
    security_id = await create_security()
    connector = WorkingConnector(security_id, knowledge_time=_HISTORIC_KNOWLEDGE)
    result = await connector.run(RunKind.BACKFILL)

    assert result.lag_report is None
    run = await latest_run(connector.source_name)
    assert "knowledge_time_lag_breach" not in run.quality_metrics


async def test_a_live_run_that_wrote_nothing_reports_no_lag_measurement() -> None:
    """No rows means no evidence — which must not be recorded as zero lag."""

    class _EmptyConnector(_ProbeConnector):
        source_name = "probe_empty"

        async def fetch_batches(
            self,
            checkpoint: Checkpoint | None,  # noqa: ARG002 — yields nothing
        ) -> AsyncIterator[Batch]:
            """Yield no batches at all."""
            for _ in ():
                yield Batch(rows=(), checkpoint={})

    result = await _EmptyConnector().run(RunKind.LIVE)
    assert result.rows_written == 0
    assert result.lag_report is None
    assert result.checkpoint_after is None
