"""CC.8: the reusable connector contract every concrete connector must satisfy.

Invariant I3 says a connector faced with an unavailable source **raises**; it
never returns placeholder data, an empty result that reads as "the source says
there is nothing", or a partially-written run reported as a success. That is a
property of the base class, so it is asserted once, here, as a shared test
class — and every concrete connector (P3.2 EDGAR first) inherits it by
subclassing :class:`ConnectorContractTests` and overriding
:meth:`ConnectorContractTests.build_connector`.

Test doubles are legitimate scaffolding and belong here, under
``backend/tests``: simulating an unavailable source is the only way to prove
the framework raises when one occurs. What is forbidden — and what
``scripts/check_no_fabrication.py`` enforces — is production code under
``backend/ingest`` doing the same thing.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import pytest
from sqlalchemy import select

from backend.db import as_of, ingest_writer_session
from backend.db.models import PriceBar, SecurityMaster
from backend.ingest.base import Connector, RunResult
from backend.ingest.errors import SourceUnavailableError
from backend.ingest.runs import IngestionRun, RunKind, RunStatus, get_run


async def stored_fact_row_count() -> int:
    """Return how many fact rows are visible now, through the sanctioned read path."""
    async with as_of(dt.datetime.now(dt.UTC)) as session:
        bars = (await session.scalars(select(PriceBar))).all()
        masters = (await session.scalars(select(SecurityMaster))).all()
    return len(bars) + len(masters)


async def latest_run(source: str) -> IngestionRun:
    """Return the most recent ingestion-run record for ``source``."""
    async with ingest_writer_session() as session:
        run_id = (
            await session.execute(
                select(IngestionRun.run_id)
                .where(IngestionRun.source == source)
                .order_by(IngestionRun.run_id.desc())
                .limit(1)
            )
        ).scalar_one()
    return await get_run(run_id)


class ConnectorContractTests:
    """Contract every connector must satisfy when its source is unavailable.

    Subclass in a test module, override :meth:`build_connector` to return a
    connector whose source is unavailable, and pytest collects the inherited
    tests against it.

    Attributes:
        expected_error: the exception the unavailable source raises. Must be
            (a subclass of) :class:`SourceUnavailableError` — the one family
            the framework's callers are told to expect.
    """

    expected_error: ClassVar[type[SourceUnavailableError]] = SourceUnavailableError

    def build_connector(self) -> Connector:
        """Return the connector under test, wired to an unavailable source."""
        raise NotImplementedError

    async def test_unavailable_source_raises_rather_than_returning_data(self) -> None:
        """I3: no placeholder, no default, no empty success. The call raises."""
        with pytest.raises(self.expected_error):
            await self.build_connector().run(RunKind.LIVE)

    async def test_unavailable_source_writes_no_rows(self) -> None:
        """Nothing plausible-looking is left behind for a later query to find."""
        before = await stored_fact_row_count()
        with pytest.raises(self.expected_error):
            await self.build_connector().run(RunKind.LIVE)
        assert await stored_fact_row_count() == before

    async def test_unavailable_source_records_the_run_as_failed(self) -> None:
        """The failure is recorded, with its cause, not silently swallowed."""
        connector = self.build_connector()
        with pytest.raises(self.expected_error) as raised:
            await connector.run(RunKind.LIVE)
        run = await latest_run(connector.source_name)
        assert run.status == RunStatus.FAILED.value
        assert run.rows_written == 0
        assert run.error_detail is not None
        assert type(raised.value).__name__ in run.error_detail

    async def test_run_returns_no_result_object_on_failure(self) -> None:
        """A RunResult is only ever produced by a run that actually succeeded."""
        result: RunResult | None = None
        with pytest.raises(self.expected_error):
            result = await self.build_connector().run(RunKind.LIVE)
        assert result is None

    async def test_the_error_belongs_to_the_source_unavailable_family(self) -> None:
        """Callers catch one family; a connector must not invent its own."""
        with pytest.raises(SourceUnavailableError):
            await self.build_connector().run(RunKind.LIVE)
