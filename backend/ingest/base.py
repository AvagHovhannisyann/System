"""Connector base class: the contract every data source implements (P3.1).

A connector's only job is to turn a source into batches of ORM rows. Everything
that must be true of *every* source — pacing, retry, resumption, run
bookkeeping, knowledge-time hygiene, data-quality measurement — lives here, so
it is done once and identically rather than remembered per connector.

What a concrete connector declares (all three are required, checked at
class-definition time):

- ``source_name`` — the stable identifier under which its runs, checkpoints
  and quality metrics are recorded;
- ``knowledge_time_policy`` — **how it derives ``knowledge_time``**, in prose,
  plus the lag it expects on live runs. D-011 requires this declaration in
  code, not in a comment, because ``knowledge_time`` is the column invariant
  I1 rests on and a source-specific derivation is the thing most likely to be
  silently wrong;
- ``rate_limit`` — the source's published request budget.

What a concrete connector implements: :meth:`Connector.fetch_batches`, an
async generator yielding :class:`Batch` objects. It calls
:meth:`Connector.request` for every outbound call, which paces and retries it.

What the framework guarantees around it:

1. the run is recorded from before the first request to after the last write,
   with its ``run_kind`` (D-011's compensating control) and its resume
   position both before and after;
2. every batch is committed in its own transaction, so a failure part-way
   through keeps what was already written and records how far it got;
3. **failure is never converted into data.** If the source is unavailable the
   exception propagates out of :meth:`Connector.run` after the run is recorded
   ``failed``. There is no fallback, no default, no empty-batch substitution —
   invariant I3, proven by the base-class contract test (CC.8);
4. live runs are checked against the connector's declared knowledge-time lag
   and flagged when they trail it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeVar

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.bitemporal import BitemporalMixin
from backend.ingest.checkpoint import normalize_checkpoint
from backend.ingest.errors import ConnectorDeclarationError
from backend.ingest.quality import (
    DataQualityMetric,
    KnowledgeTimeLagReport,
    KnowledgeTimePolicy,
    knowledge_time_lag_report,
    metrics_as_json,
)
from backend.ingest.ratelimit import RateLimit, TokenBucket
from backend.ingest.retry import RetryPolicy, full_jitter, retry_async
from backend.ingest.runs import RunKind, RunStatus, finish_run, latest_checkpoint, start_run

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from backend.db.base import Base
    from backend.ingest.checkpoint import Checkpoint, JsonScalar

__all__ = ["Batch", "Connector", "ConnectorRuntime", "RunResult"]

T = TypeVar("T")

_logger = get_logger(__name__)


def _utc_now() -> dt.datetime:
    """Return the current instant as a timezone-aware UTC datetime."""
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class ConnectorRuntime:
    """Injectable clocks and schedulers, so pacing and retry are testable.

    Production uses the defaults. Tests substitute a controlled clock and a
    sleep that advances it, which is what lets the retry schedule and the rate
    limiter be asserted exactly instead of by sleeping and hoping.

    Attributes:
        sleep: awaitable sleep taking seconds.
        monotonic: monotonic clock in seconds, for rate limiting.
        jitter: maps a backoff ceiling in seconds to an actual wait in
            seconds.
        now: current timezone-aware UTC instant, for run and lag bookkeeping.
    """

    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic
    jitter: Callable[[float], float] = full_jitter
    now: Callable[[], dt.datetime] = _utc_now


@dataclass(frozen=True, slots=True)
class Batch:
    """One unit of work: rows to write, and the position reached by writing them.

    Attributes:
        rows: ORM instances to insert. Written with ``session.add_all`` in a
            single committed transaction, so a batch is atomic.
        checkpoint: the resume position **after** these rows are durably
            written. Recorded on the run only once the commit succeeds, so a
            checkpoint can never claim progress that was rolled back.
    """

    rows: tuple[Base, ...]
    checkpoint: Checkpoint


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one successful :meth:`Connector.run`.

    A failed run does not produce one of these — it raises. Attributes:
        run_id: the ingestion-run record's id.
        source: connector source name.
        run_kind: ``backfill`` or ``live``.
        rows_written: rows durably committed (count).
        batches_written: batches durably committed (count).
        checkpoint_before: position the run resumed from, or ``None``.
        checkpoint_after: position reached, or ``None`` if nothing committed.
        lag_report: live-run knowledge-time lag check, or ``None`` for a
            backfill (where a large lag is correct by construction).
        metrics: the data-quality metrics emitted, as recorded on the run.
    """

    run_id: int
    source: str
    run_kind: RunKind
    rows_written: int
    batches_written: int
    checkpoint_before: dict[str, JsonScalar] | None
    checkpoint_after: dict[str, JsonScalar] | None
    lag_report: KnowledgeTimeLagReport | None
    metrics: tuple[DataQualityMetric, ...] = field(default_factory=tuple)


class Connector(ABC):
    """Abstract base for every data-source connector.

    Subclasses declare ``source_name``, ``knowledge_time_policy`` and
    ``rate_limit`` as class attributes and implement
    :meth:`fetch_batches`. ``retry_policy`` may be overridden; its default is
    five attempts with exponential backoff from 0.5s, capped at 30s.

    Instances are cheap and single-use per run: the rate-limiter state and the
    per-run counters live on the instance, so two concurrent runs of one source
    must not share one instance.
    """

    source_name: ClassVar[str]
    knowledge_time_policy: ClassVar[KnowledgeTimePolicy]
    rate_limit: ClassVar[RateLimit]
    retry_policy: ClassVar[RetryPolicy] = RetryPolicy()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject a subclass that does not declare the required contract.

        Checked at class-definition time so a misdeclared connector cannot be
        instantiated, registered or scheduled. Subclasses that are themselves
        abstract (they still declare :meth:`fetch_batches` as abstract) are
        exempt, so an intermediate base may legitimately leave the
        declarations to its concrete children.

        Raises:
            ConnectorDeclarationError: if a concrete subclass omits or
                mis-types ``source_name``, ``knowledge_time_policy`` or
                ``rate_limit``.
        """
        super().__init_subclass__(**kwargs)
        fetch = cls.__dict__.get("fetch_batches", getattr(cls, "fetch_batches", None))
        if getattr(fetch, "__isabstractmethod__", False):
            return
        expected: tuple[tuple[str, type], ...] = (
            ("source_name", str),
            ("knowledge_time_policy", KnowledgeTimePolicy),
            ("rate_limit", RateLimit),
        )
        for attribute, attribute_type in expected:
            value = getattr(cls, attribute, None)
            if value is None:
                msg = (
                    f"{cls.__name__} must declare {attribute!r}: every connector states its "
                    "source name, its knowledge_time derivation policy (D-011) and its "
                    "rate limit"
                )
                raise ConnectorDeclarationError(msg)
            if not isinstance(value, attribute_type):
                msg = (
                    f"{cls.__name__}.{attribute} must be a {attribute_type.__name__}; "
                    f"got {type(value).__name__}"
                )
                raise ConnectorDeclarationError(msg)
        if not cls.source_name.strip():
            msg = f"{cls.__name__}.source_name must be a non-empty identifier"
            raise ConnectorDeclarationError(msg)

    def __init__(self, runtime: ConnectorRuntime | None = None) -> None:
        """Create a connector instance with its own rate-limiter state.

        Args:
            runtime: injectable clocks/scheduler; defaults to real time.
        """
        self.runtime = runtime if runtime is not None else ConnectorRuntime()
        self._bucket: TokenBucket = self.rate_limit.bucket(
            monotonic=self.runtime.monotonic, sleep=self.runtime.sleep
        )
        self._retries = 0
        self._rate_limit_wait_s = 0.0

    @property
    def transient_retries(self) -> int:
        """Number of transient-failure retries performed so far (count)."""
        return self._retries

    @property
    def rate_limit_wait_s(self) -> float:
        """Seconds spent waiting on the rate limiter so far."""
        return self._rate_limit_wait_s

    async def request(self, operation: Callable[[], Awaitable[T]], *, description: str) -> T:
        """Perform one source call: rate-limited, then retried if transient.

        Every outbound call a connector makes must go through this method —
        that is what makes the declared rate limit and retry policy actually
        binding rather than advisory.

        Args:
            operation: zero-argument coroutine factory performing one attempt.
                Called afresh per attempt, so it must be idempotent at the
                source (GET-shaped).
            description: short label for logs; no secrets (invariant I5).

        Returns:
            Whatever ``operation`` returned on its first successful attempt.

        Raises:
            TransientSourceError: re-raised unchanged after the retry budget
                is exhausted.
            PermanentSourceError: propagated immediately, never retried.
        """

        async def paced() -> T:
            self._rate_limit_wait_s += await self._bucket.acquire()
            return await operation()

        def count_retry(attempt: int, waited_s: float) -> None:  # noqa: ARG001 — callback signature
            self._retries += 1

        return await retry_async(
            paced,
            policy=self.retry_policy,
            description=description,
            sleep=self.runtime.sleep,
            jitter=self.runtime.jitter,
            on_retry=count_retry,
        )

    @abstractmethod
    def fetch_batches(self, checkpoint: Checkpoint | None) -> AsyncIterator[Batch]:
        """Yield batches of rows to write, resuming from ``checkpoint``.

        Implemented as an ``async def`` generator. Every outbound call must go
        through :meth:`request`.

        Args:
            checkpoint: the position the previous run reached, or ``None`` on
                a source's first ever run. Implementations must treat ``None``
                as "start from the beginning of the source's history", never
                as "start from now" — the second silently creates a permanent
                gap.

        Yields:
            :class:`Batch` objects in the order they must be written. The
            checkpoint on each batch describes the position *after* that batch.

        Raises:
            SourceUnavailableError: when the source cannot be read. The
                implementation must raise rather than yield a placeholder,
                partial or invented batch (invariant I3).
        """
        raise NotImplementedError

    async def run(self, run_kind: RunKind) -> RunResult:
        """Execute one full ingestion run and record it.

        Args:
            run_kind: ``BACKFILL`` for historical loading (knowledge times far
                behind ingestion are correct), ``LIVE`` for keeping up with a
                feed (knowledge times are checked against the declared lag).

        Returns:
            A :class:`RunResult` describing what was written.

        Raises:
            Exception: whatever the source raised, re-raised unchanged after
                the run is recorded ``failed`` with the error detail and the
                furthest checkpoint durably reached. Nothing is substituted
                for missing data and no partial run is reported as success
                (invariant I3, directive §9.1—9.2).
        """
        checkpoint_before = await latest_checkpoint(self.source_name)
        run_id = await start_run(
            source=self.source_name, run_kind=run_kind, checkpoint_before=checkpoint_before
        )
        rows_written = 0
        batches_written = 0
        checkpoint_after: dict[str, JsonScalar] | None = None
        knowledge_times: list[dt.datetime] = []
        last_write_at: dt.datetime | None = None
        _logger.info(
            "ingest.run.started",
            source=self.source_name,
            run_id=run_id,
            run_kind=run_kind.value,
            checkpoint_before=checkpoint_before,
        )
        try:
            async for batch in self.fetch_batches(checkpoint_before):
                await self._write_batch(batch.rows)
                last_write_at = self.runtime.now()
                rows_written += len(batch.rows)
                batches_written += 1
                knowledge_times.extend(
                    row.knowledge_time for row in batch.rows if isinstance(row, BitemporalMixin)
                )
                checkpoint_after = normalize_checkpoint(batch.checkpoint)
        except BaseException as exc:
            await finish_run(
                run_id,
                status=RunStatus.FAILED,
                rows_written=rows_written,
                checkpoint_after=checkpoint_after,
                error_detail=f"{type(exc).__name__}: {exc}",
                quality_metrics=metrics_as_json(self._metrics(rows_written, batches_written, None)),
                ended_at=self.runtime.now(),
            )
            _logger.error(
                "ingest.run.failed",
                source=self.source_name,
                run_id=run_id,
                run_kind=run_kind.value,
                rows_written=rows_written,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        lag_report = self._lag_report(run_kind, knowledge_times, last_write_at)
        metrics = self._metrics(rows_written, batches_written, lag_report)
        await finish_run(
            run_id,
            status=RunStatus.SUCCEEDED,
            rows_written=rows_written,
            checkpoint_after=checkpoint_after,
            quality_metrics=metrics_as_json(metrics),
            ended_at=self.runtime.now(),
        )
        if lag_report is not None and lag_report.breached:
            _logger.warning(
                "ingest.run.knowledge_time_lag_breach",
                source=self.source_name,
                run_id=run_id,
                observed_max_lag_s=(
                    None
                    if lag_report.observed_max_lag is None
                    else lag_report.observed_max_lag.total_seconds()
                ),
                declared_max_lag_s=lag_report.declared_max_lag.total_seconds(),
                policy=self.knowledge_time_policy.description,
            )
        _logger.info(
            "ingest.run.succeeded",
            source=self.source_name,
            run_id=run_id,
            run_kind=run_kind.value,
            rows_written=rows_written,
            batches_written=batches_written,
            checkpoint_after=checkpoint_after,
        )
        return RunResult(
            run_id=run_id,
            source=self.source_name,
            run_kind=run_kind,
            rows_written=rows_written,
            batches_written=batches_written,
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            lag_report=lag_report,
            metrics=tuple(metrics),
        )

    async def _write_batch(self, rows: Sequence[Base]) -> None:
        """Commit one batch through the sanctioned append-only writer session.

        Overridable by connectors needing a different write shape (e.g. one
        that must resolve identity anchors first). The default adds every row
        and commits, so a batch is atomic. Knowledge-time hygiene is enforced
        beneath this by the class-level flush listener in
        :mod:`backend.ingest.write`, so an override cannot skip it.
        """
        if not rows:
            return
        async with ingest_writer_session() as session:
            session.add_all(list(rows))
            await session.commit()

    def _lag_report(
        self,
        run_kind: RunKind,
        knowledge_times: Sequence[dt.datetime],
        last_write_at: dt.datetime | None,
    ) -> KnowledgeTimeLagReport | None:
        """Run the live-run knowledge-time lag check, or skip it for a backfill.

        Returns ``None`` for a backfill run (its knowledge times are
        historical by design, so a lag figure would be meaningless) and for a
        live run that wrote nothing (no rows, no evidence — reporting zero lag
        would be an invented measurement).
        """
        if run_kind is not RunKind.LIVE or last_write_at is None:
            return None
        return knowledge_time_lag_report(
            knowledge_times, observed_at=last_write_at, policy=self.knowledge_time_policy
        )

    def _metrics(
        self,
        rows_written: int,
        batches_written: int,
        lag_report: KnowledgeTimeLagReport | None,
    ) -> list[DataQualityMetric]:
        """Assemble the data-quality metrics emitted by a run.

        Every metric is a measurement taken during the run; none is inferred
        or defaulted. The lag metrics are present only when a lag check
        actually ran (live run that wrote rows), so their absence means "not
        measured" rather than "measured as zero".
        """
        metrics = [
            DataQualityMetric("rows_written", rows_written, "rows"),
            DataQualityMetric("batches_written", batches_written, "batches"),
            DataQualityMetric("transient_retries", self._retries, "count"),
            DataQualityMetric("rate_limit_wait", round(self._rate_limit_wait_s, 6), "seconds"),
        ]
        if lag_report is not None:
            observed = lag_report.observed_max_lag
            metrics.extend(
                [
                    DataQualityMetric(
                        "knowledge_time_max_lag",
                        None if observed is None else observed.total_seconds(),
                        "seconds",
                    ),
                    DataQualityMetric(
                        "knowledge_time_declared_max_lag",
                        lag_report.declared_max_lag.total_seconds(),
                        "seconds",
                    ),
                    DataQualityMetric("knowledge_time_lag_breach", lag_report.breached, "boolean"),
                ]
            )
        return metrics
