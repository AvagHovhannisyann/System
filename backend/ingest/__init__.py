"""Ingestion framework: the connector contract and everything it enforces (P3.1).

Public surface of ``backend.ingest``. One module per concern:

- :mod:`backend.ingest.base` — the :class:`~backend.ingest.base.Connector`
  abstract base every data source implements, and the run orchestration around
  it (rate limiting, retry, checkpointed resume, run recording, data-quality
  metric emission);
- :mod:`backend.ingest.errors` — the failure taxonomy, whose central rule is
  that an unavailable source raises and is never converted into a value
  (invariant I3);
- :mod:`backend.ingest.retry` / :mod:`backend.ingest.ratelimit` — transient-only
  exponential backoff with jitter, and the per-source token bucket;
- :mod:`backend.ingest.checkpoint` / :mod:`backend.ingest.runs` — incremental
  sync position, and the operational log of every run including its
  ``backfill``/``live`` kind (D-011's compensating control);
- :mod:`backend.ingest.quality` — the knowledge-time policy each connector must
  declare, and the live-run lag check;
- :mod:`backend.ingest.supersession` — the append-only way to close an
  open-ended fact, which is the only way there is;
- :mod:`backend.ingest.registry` — source name to connector class.

**Importing this package installs knowledge-time write enforcement.**
:mod:`backend.ingest.write` registers a ``before_flush`` listener on the ORM
``Session`` class, so any flush anywhere in the process carrying a bitemporal
row with a ``knowledge_time`` in the future is refused before I/O. It is
imported here rather than left to the modules that happen to need it, so the
guarantee holds for every consumer of this package.

:mod:`backend.ingest.celery_app` is deliberately **not** imported here: it is
the worker/beat entrypoint, and importing Celery into every consumer of the
connector contract would buy nothing.
"""

from __future__ import annotations

from backend.ingest.base import Batch, Connector, ConnectorRuntime, RunResult
from backend.ingest.checkpoint import Checkpoint, JsonScalar, normalize_checkpoint
from backend.ingest.errors import (
    ConnectorDeclarationError,
    FutureKnowledgeTimeError,
    IngestError,
    PermanentSourceError,
    RunRecordError,
    SourceUnavailableError,
    SupersessionError,
    TransientSourceError,
    source_error_for_status,
)
from backend.ingest.quality import (
    DataQualityMetric,
    KnowledgeTimeLagReport,
    KnowledgeTimePolicy,
    knowledge_time_lag_report,
    metrics_as_json,
)
from backend.ingest.ratelimit import RateLimit, TokenBucket
from backend.ingest.registry import connector_class, register_connector, registered_sources
from backend.ingest.retry import RetryPolicy, full_jitter, retry_async
from backend.ingest.runs import (
    IngestionRun,
    RunKind,
    RunStatus,
    finish_run,
    get_run,
    latest_checkpoint,
    start_run,
)
from backend.ingest.supersession import close_open_interval, supersede_open_interval
from backend.ingest.write import FUTURE_KNOWLEDGE_TIME_TOLERANCE, validate_knowledge_time

__all__ = [
    "FUTURE_KNOWLEDGE_TIME_TOLERANCE",
    "Batch",
    "Checkpoint",
    "Connector",
    "ConnectorDeclarationError",
    "ConnectorRuntime",
    "DataQualityMetric",
    "FutureKnowledgeTimeError",
    "IngestError",
    "IngestionRun",
    "JsonScalar",
    "KnowledgeTimeLagReport",
    "KnowledgeTimePolicy",
    "PermanentSourceError",
    "RateLimit",
    "RetryPolicy",
    "RunKind",
    "RunRecordError",
    "RunResult",
    "RunStatus",
    "SourceUnavailableError",
    "SupersessionError",
    "TokenBucket",
    "TransientSourceError",
    "close_open_interval",
    "connector_class",
    "finish_run",
    "full_jitter",
    "get_run",
    "knowledge_time_lag_report",
    "latest_checkpoint",
    "metrics_as_json",
    "normalize_checkpoint",
    "register_connector",
    "registered_sources",
    "retry_async",
    "source_error_for_status",
    "start_run",
    "supersede_open_interval",
    "validate_knowledge_time",
]
