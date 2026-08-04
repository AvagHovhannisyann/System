"""Celery application and beat scaffolding for scheduled ingestion (P3.1).

Broker and result backend are Redis, already in the compose stack. Workers and
the scheduler are started against this module::

    celery -A backend.ingest.celery_app worker --loglevel=INFO
    celery -A backend.ingest.celery_app beat   --loglevel=INFO

**The beat schedule is deliberately empty.** Directive §9.4 and §1.2 forbid
inventing what does not exist, and no connector exists yet: P3.2 (SEC EDGAR)
lands the first one, and the sources behind blocker B1 are not chosen-and-keyed
until the operator provisions them. A schedule entry naming a source with no
implementation would be a fabricated operational fact — the scheduler would
report a configured job that can never run. Entries are added to
:data:`BEAT_SCHEDULE` by the task that lands the corresponding connector, at
which point the cadence is a real decision about a real source.

:func:`run_connector_task` is the task those future entries point at: it
resolves a source name through the registry and executes one run. It is
defined now because it is source-independent; scheduling it is not.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from celery import Celery

from backend.core.config import get_settings
from backend.ingest.registry import connector_class
from backend.ingest.runs import RunKind

if TYPE_CHECKING:
    from backend.core.config import Settings

__all__ = ["BEAT_SCHEDULE", "build_celery_app", "celery_app", "run_connector_task"]

BEAT_SCHEDULE: dict[str, dict[str, Any]] = {}
"""Periodic ingestion schedule: empty until connectors exist (module docstring)."""


def build_celery_app(settings: Settings | None = None) -> Celery:
    """Build the Celery application from settings.

    Args:
        settings: configuration to read the Redis URL from; defaults to the
            cached application settings.

    Returns:
        A configured :class:`celery.Celery` instance. Both broker and result
        backend point at ``settings.redis_url``.

    Notes:
        Configuration choices worth stating: JSON serialization only (pickle
        would let a queue entry execute arbitrary code); UTC everywhere,
        matching the store's timezone discipline (D-011); ``task_acks_late``
        with ``worker_prefetch_multiplier = 1`` so a worker killed mid-run
        leaves the task to be retried rather than silently lost — the
        ingestion-run record would otherwise stay ``running`` with nothing
        driving it.
    """
    resolved = settings if settings is not None else get_settings()
    app = Celery(
        "quant_research_platform",
        broker=resolved.redis_url,
        backend=resolved.redis_url,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        beat_schedule=BEAT_SCHEDULE,
    )
    return app


celery_app = build_celery_app()
"""Process-wide Celery application (the ``-A backend.ingest.celery_app`` target)."""


@celery_app.task(name="backend.ingest.run_connector")
def run_connector_task(source: str, run_kind: str) -> dict[str, Any]:
    """Run one ingestion cycle for ``source`` and return a summary.

    The single task every future beat-schedule entry points at.

    Args:
        source: registered connector source name.
        run_kind: ``"backfill"`` or ``"live"``.

    Returns:
        A JSON-serializable summary: run id, source, rows and batches written,
        and whether the live knowledge-time lag check was breached (``None``
        when no check ran).

    Raises:
        KeyError: if no connector is registered for ``source``.
        ValueError: if ``run_kind`` is not a valid run kind.
        Exception: whatever the connector raised, after its run is recorded
            ``failed``. The task deliberately does not swallow it: a task that
            returned a summary for a failed run would report ingestion that
            did not happen (invariant I3).
    """
    connector = connector_class(source)()
    result = asyncio.run(connector.run(RunKind(run_kind)))
    return {
        "run_id": result.run_id,
        "source": result.source,
        "run_kind": result.run_kind.value,
        "rows_written": result.rows_written,
        "batches_written": result.batches_written,
        "knowledge_time_lag_breach": (
            None if result.lag_report is None else result.lag_report.breached
        ),
    }
