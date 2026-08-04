"""Connector registry and Celery/beat wiring (P3.1).

The beat-schedule assertion is a scope guard, not a formality: directive §9.4
forbids inventing what does not exist, and a schedule entry naming a source
with no connector would be a configured job that can never run. It stays empty
until P3.2 lands the first connector.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest

from backend.core.config import Settings
from backend.ingest.base import Batch, Connector
from backend.ingest.celery_app import (
    BEAT_SCHEDULE,
    build_celery_app,
    celery_app,
    run_connector_task,
)
from backend.ingest.quality import KnowledgeTimePolicy
from backend.ingest.ratelimit import RateLimit
from backend.ingest.registry import connector_class, register_connector, registered_sources

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from backend.ingest.checkpoint import Checkpoint


class _RegistrableConnector(Connector):
    """Minimal connector used to exercise registration."""

    source_name = "probe_registry"
    knowledge_time_policy = KnowledgeTimePolicy(
        description="probe: source-provided availability timestamp",
        max_live_lag=dt.timedelta(hours=1),
    )
    rate_limit = RateLimit(requests_per_second=5.0, burst=5)

    async def fetch_batches(
        self,
        checkpoint: Checkpoint | None,  # noqa: ARG002 — registration test; never iterated
    ) -> AsyncIterator[Batch]:
        """Yield nothing."""
        for _ in ():
            yield Batch(rows=(), checkpoint={})


class _CollidingConnector(_RegistrableConnector):
    """A different class claiming the same source name."""

    source_name = "probe_registry"


# --- registry ---------------------------------------------------------------


def test_registration_is_explicit_and_round_trips() -> None:
    register_connector(_RegistrableConnector)
    assert connector_class("probe_registry") is _RegistrableConnector
    assert "probe_registry" in registered_sources()


def test_registering_the_same_class_twice_is_idempotent() -> None:
    register_connector(_RegistrableConnector)
    register_connector(_RegistrableConnector)
    assert connector_class("probe_registry") is _RegistrableConnector


def test_a_second_class_cannot_claim_a_registered_source() -> None:
    """Silent replacement would let import order decide what runs for a source."""
    register_connector(_RegistrableConnector)
    with pytest.raises(ValueError, match="already registered"):
        register_connector(_CollidingConnector)


def test_unknown_source_raises_and_names_what_is_registered() -> None:
    with pytest.raises(KeyError, match="no connector registered"):
        connector_class("probe_never_registered")


def test_defining_a_connector_does_not_register_it() -> None:
    """Subclassing must not enrol test doubles into the production registry (I3)."""

    class _UnregisteredConnector(_RegistrableConnector):
        source_name = "probe_unregistered"

    assert "probe_unregistered" not in registered_sources()
    assert _UnregisteredConnector.source_name == "probe_unregistered"


# --- celery wiring ----------------------------------------------------------


def test_broker_and_backend_point_at_the_configured_redis() -> None:
    app = build_celery_app(Settings(redis_url="redis://probe-host:6379/3"))
    assert app.conf.broker_url == "redis://probe-host:6379/3"
    assert app.conf.result_backend == "redis://probe-host:6379/3"


def test_serialization_is_json_only() -> None:
    """Pickle on a broker is remote code execution; JSON is the only content type."""
    app = build_celery_app(Settings(redis_url="redis://probe-host:6379/0"))
    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert list(app.conf.accept_content) == ["json"]


def test_timezone_is_utc_matching_the_store() -> None:
    app = build_celery_app(Settings(redis_url="redis://probe-host:6379/0"))
    assert app.conf.enable_utc is True
    assert str(app.conf.timezone) == "UTC"


def test_a_killed_worker_leaves_its_task_redeliverable() -> None:
    """Late ack + prefetch 1: a half-run ingestion is retried, not silently lost."""
    app = build_celery_app(Settings(redis_url="redis://probe-host:6379/0"))
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1


def test_beat_schedule_is_empty_until_connectors_exist() -> None:
    """Scope guard: no invented cadence for a source that does not exist (section 9.4)."""
    assert BEAT_SCHEDULE == {}
    assert dict(celery_app.conf.beat_schedule) == {}


def test_the_generic_run_task_is_registered_under_a_stable_name() -> None:
    """Future beat entries point here; the name must not drift."""
    assert run_connector_task.name == "backend.ingest.run_connector"
    assert "backend.ingest.run_connector" in celery_app.tasks
