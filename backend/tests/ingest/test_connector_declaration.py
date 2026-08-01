"""What a connector must declare, checked at class-definition time (P3.1, D-011).

D-011 requires every connector to declare, in code, how it derives
``knowledge_time``. A declaration that can be forgotten is not a requirement,
so the base class refuses to define a concrete subclass that omits it — before
the class exists, let alone before it can be registered or scheduled.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, ClassVar

import pytest

from backend.ingest.base import Batch, Connector, ConnectorRuntime
from backend.ingest.errors import ConnectorDeclarationError
from backend.ingest.quality import KnowledgeTimePolicy
from backend.ingest.ratelimit import RateLimit
from backend.ingest.retry import RetryPolicy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from backend.ingest.checkpoint import Checkpoint

_POLICY = KnowledgeTimePolicy(
    description="probe: source-provided availability timestamp",
    max_live_lag=dt.timedelta(hours=1),
)
_LIMIT = RateLimit(requests_per_second=10.0, burst=10)


class _DeclaredConnector(Connector):
    """A fully-declared connector; the positive control for this module."""

    source_name = "probe_declared"
    knowledge_time_policy = _POLICY
    rate_limit = _LIMIT

    async def fetch_batches(
        self,
        checkpoint: Checkpoint | None,  # noqa: ARG002 — declaration test; never iterated
    ) -> AsyncIterator[Batch]:
        """Yield nothing: this module tests declaration, not fetching."""
        for _ in ():
            yield Batch(rows=(), checkpoint={})


def test_a_fully_declared_connector_is_definable_and_instantiable() -> None:
    connector = _DeclaredConnector()
    assert connector.source_name == "probe_declared"
    assert connector.knowledge_time_policy is _POLICY
    assert connector.rate_limit is _LIMIT
    assert isinstance(connector.retry_policy, RetryPolicy)


def test_missing_source_name_is_refused_at_class_definition() -> None:
    with pytest.raises(ConnectorDeclarationError, match="source_name"):

        class _NoName(Connector):
            knowledge_time_policy = _POLICY
            rate_limit = _LIMIT

            async def fetch_batches(
                self,
                checkpoint: Checkpoint | None,  # noqa: ARG002 — never reached
            ) -> AsyncIterator[Batch]:
                for _ in ():
                    yield Batch(rows=(), checkpoint={})


def test_missing_knowledge_time_policy_is_refused() -> None:
    """The D-011 declaration is the one that cannot be optional."""
    with pytest.raises(ConnectorDeclarationError, match="knowledge_time_policy"):

        class _NoPolicy(Connector):
            source_name = "probe_no_policy"
            rate_limit = _LIMIT

            async def fetch_batches(
                self,
                checkpoint: Checkpoint | None,  # noqa: ARG002 — never reached
            ) -> AsyncIterator[Batch]:
                for _ in ():
                    yield Batch(rows=(), checkpoint={})


def test_missing_rate_limit_is_refused() -> None:
    with pytest.raises(ConnectorDeclarationError, match="rate_limit"):

        class _NoLimit(Connector):
            source_name = "probe_no_limit"
            knowledge_time_policy = _POLICY

            async def fetch_batches(
                self,
                checkpoint: Checkpoint | None,  # noqa: ARG002 — never reached
            ) -> AsyncIterator[Batch]:
                for _ in ():
                    yield Batch(rows=(), checkpoint={})


def test_mistyped_declaration_is_refused() -> None:
    with pytest.raises(ConnectorDeclarationError, match="must be a KnowledgeTimePolicy"):

        class _BadPolicy(Connector):
            source_name = "probe_bad_policy"
            knowledge_time_policy: ClassVar[KnowledgeTimePolicy] = "one hour"  # type: ignore[assignment]
            rate_limit = _LIMIT

            async def fetch_batches(
                self,
                checkpoint: Checkpoint | None,  # noqa: ARG002 — never reached
            ) -> AsyncIterator[Batch]:
                for _ in ():
                    yield Batch(rows=(), checkpoint={})


def test_empty_source_name_is_refused() -> None:
    with pytest.raises(ConnectorDeclarationError, match="non-empty"):

        class _BlankName(Connector):
            source_name = "   "
            knowledge_time_policy = _POLICY
            rate_limit = _LIMIT

            async def fetch_batches(
                self,
                checkpoint: Checkpoint | None,  # noqa: ARG002 — never reached
            ) -> AsyncIterator[Batch]:
                for _ in ():
                    yield Batch(rows=(), checkpoint={})


def test_an_intermediate_abstract_subclass_may_defer_the_declarations() -> None:
    """A shared base (e.g. a common HTTP connector) need not name a source."""

    class _AbstractHttpConnector(Connector):
        """Intermediate base that leaves the declarations to its children."""

        rate_limit = _LIMIT

    assert _AbstractHttpConnector.rate_limit is _LIMIT


def test_runtime_defaults_are_real_clocks() -> None:
    """The injection points exist for tests; production must not depend on them."""
    runtime = ConnectorRuntime()
    assert runtime.now().tzinfo is not None
    assert isinstance(runtime.monotonic(), float)
