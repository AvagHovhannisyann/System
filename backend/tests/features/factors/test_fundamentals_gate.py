"""The six B1-blocked factors refuse, and say what is actually missing (P5.3).

Invariant I3 and directive §9.1-9.2 make this the whole specification for
book-to-price, earnings yield, gross profitability, ROIC, accruals and asset
growth today: their input feed does not exist, so they raise. What is tested
here is that they raise *for the right reason and with the right information* —
naming the missing table, the blocker that must clear, and the connector task
that will supply it — because an exception nobody can act on is only marginally
better than a fabricated number.

The last test in this file is a **canary**. It asserts that the fundamentals
table really is absent from the mapped schema, and it is expected to start
failing the day P3.5 lands. That failure is the signal to write the six
computations; until then it is the evidence that this package's refusals
describe reality rather than a hard-coded belief about it.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import pytest

from backend.db.base import Base
from backend.features.compute import FeatureComputeRequest
from backend.features.errors import FeatureComputeError, FeatureError
from backend.features.factors import BASELINE_FACTORS
from backend.features.factors._fundamentals import (
    FUNDAMENTALS_BLOCKER,
    FUNDAMENTALS_CONNECTOR_TASK,
    FUNDAMENTALS_TABLE,
    FundamentalsComputationNotWrittenError,
    FundamentalsSourceUnavailableError,
    fundamentals_source_present,
    require_fundamentals_source,
)
from backend.features.factors.growth import accruals, asset_growth
from backend.features.factors.quality import gross_profitability, roic
from backend.features.factors.value import book_to_price, earnings_yield

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputation
    from backend.features.spec import FeatureSpec

COMPUTE_DATE = dt.date(2026, 3, 2)

BLOCKED: list[tuple[str, FeatureComputation]] = [
    ("book_to_price", book_to_price),
    ("earnings_yield", earnings_yield),
    ("gross_profitability", gross_profitability),
    ("roic", roic),
    ("accruals", accruals),
    ("asset_growth", asset_growth),
]
BLOCKED_IDS = [name for name, _ in BLOCKED]

SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in BASELINE_FACTORS}


class NoSession:
    """A stand-in that fails loudly if a blocked factor tries to read anything.

    The refusal must happen before any I/O. If a computation reached for the
    database, this object's missing ``execute`` would raise ``AttributeError``
    rather than the named error the tests expect, and the assertion on the
    exception type would fail.
    """


def request_for(name: str) -> FeatureComputeRequest:
    """Build a request for one blocked feature at its declared cutoff."""
    spec = SPEC_BY_NAME[name]
    return FeatureComputeRequest(
        feature=name,
        compute_date=COMPUTE_DATE,
        as_of=spec.knowledge_cutoff(COMPUTE_DATE),
        security_ids=(1, 2, 3),
    )


@pytest.mark.parametrize(("name", "computation"), BLOCKED, ids=BLOCKED_IDS)
async def test_a_blocked_factor_raises_instead_of_returning_a_plausible_number(
    name: str, computation: FeatureComputation
) -> None:
    """The one behaviour I3 permits for a feature whose source does not exist."""
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        await computation(cast("AsyncSession", NoSession()), request_for(name))
    assert raised.value.feature == name


@pytest.mark.parametrize(("name", "computation"), BLOCKED, ids=BLOCKED_IDS)
async def test_the_refusal_names_the_table_the_blocker_and_the_connector_task(
    name: str, computation: FeatureComputation
) -> None:
    """An operator reading the message must know what to unblock and where to look."""
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        await computation(cast("AsyncSession", NoSession()), request_for(name))
    message = str(raised.value)
    assert FUNDAMENTALS_TABLE in message
    assert FUNDAMENTALS_BLOCKER in message
    assert FUNDAMENTALS_CONNECTOR_TASK in message
    assert "BLOCKERS.md" in message
    assert raised.value.table == FUNDAMENTALS_TABLE
    assert raised.value.blocker == FUNDAMENTALS_BLOCKER
    assert raised.value.task == FUNDAMENTALS_CONNECTOR_TASK


@pytest.mark.parametrize(("name", "computation"), BLOCKED, ids=BLOCKED_IDS)
async def test_the_refusal_is_part_of_the_feature_error_taxonomy(
    name: str, computation: FeatureComputation
) -> None:
    """A caller catching ``FeatureComputeError`` catches this without knowing about B1."""
    with pytest.raises(FeatureComputeError):
        await computation(cast("AsyncSession", NoSession()), request_for(name))
    assert issubclass(FundamentalsSourceUnavailableError, FeatureComputeError)
    assert issubclass(FundamentalsComputationNotWrittenError, FeatureComputeError)
    assert issubclass(FundamentalsSourceUnavailableError, FeatureError)


@pytest.mark.parametrize(("name", "computation"), BLOCKED, ids=BLOCKED_IDS)
async def test_the_refusal_does_not_depend_on_which_securities_were_asked_for(
    name: str, computation: FeatureComputation
) -> None:
    """An empty universe must not make a blocked factor look like it succeeded.

    Returning an empty array for zero securities would be a perfectly typed,
    entirely wrong answer: it says "computed, nothing to report" for a feature
    that computed nothing at all.
    """
    spec = SPEC_BY_NAME[name]
    empty = FeatureComputeRequest(
        feature=name,
        compute_date=COMPUTE_DATE,
        as_of=spec.knowledge_cutoff(COMPUTE_DATE),
        security_ids=(),
    )
    with pytest.raises(FundamentalsSourceUnavailableError):
        await computation(cast("AsyncSession", NoSession()), empty)


def test_the_gate_reports_the_other_error_once_the_feed_has_landed() -> None:
    """After P3.5, "blocked on B1" would be the wrong answer, so a different error fires.

    Probed against a supplied table set rather than by registering a fake table
    in the process-wide metadata, which would corrupt every other test in the
    run.
    """
    with pytest.raises(FundamentalsComputationNotWrittenError) as raised:
        require_fundamentals_source("book_to_price", tables={FUNDAMENTALS_TABLE})
    assert raised.value.feature == "book_to_price"
    message = str(raised.value)
    assert FUNDAMENTALS_TABLE in message
    # The blocker is named only to say it is no longer the explanation; the
    # outstanding work is now an engineer's, and the message must say so.
    assert f"{FUNDAMENTALS_BLOCKER} no longer explains" in message
    assert "must be written" in message


def test_the_gate_reports_the_blocker_when_the_feed_is_absent() -> None:
    """The other branch of the same probe, so neither is asserted by omission."""
    with pytest.raises(FundamentalsSourceUnavailableError):
        require_fundamentals_source("book_to_price", tables=set())


def test_the_probe_reads_the_mapped_schema_rather_than_a_flag() -> None:
    """``fundamentals_source_present()`` answers from ``Base.metadata``, not a constant."""
    assert fundamentals_source_present({FUNDAMENTALS_TABLE}) is True
    assert fundamentals_source_present(set()) is False
    assert fundamentals_source_present({"price_bar"}) is False


def test_canary_the_fundamentals_table_is_still_absent_from_the_schema() -> None:
    """Expected to fail the day P3.5 lands. That failure is the signal to write the six.

    Until then this is what makes the refusals above evidence rather than
    assertion: the table this package says does not exist genuinely does not
    exist, while ``price_bar`` — which the price factors read — genuinely does.
    """
    mapped = set(Base.metadata.tables)
    assert FUNDAMENTALS_TABLE not in mapped
    assert "price_bar" in mapped
