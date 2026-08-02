"""Short interest: the lag is the deliverable, and the refusal names a new gap (P5.3).

This factor has no arithmetic to test, because it has no data and — unlike every
other blocked factor in the package — no connector task and no blocker entry
either. What it does have is the most consequential availability lag in the
library, and a refusal that has to say something the six B1 refusals do not.

**The lag.** US short interest is published on a *settlement-date* basis roughly
eight business days after that settlement date, so every row carries a prominent
date that is not the date the number became public. A connector keying
``knowledge_time`` to that settlement date — the obvious join key — would grant
eight business days of foresight twice a month, and the resulting factor would
look completely healthy: real ratios, right magnitude, right sign, plausible
distribution, better backtest. The 17-day margin exists to survive that mistake,
and it is asserted here against literals written out by hand from the module's
stated derivation, per D-027's standing lesson that a test recomputing a formula
cannot see an error in it.

**The refusal.** Reusing "blocked on B1" would have been the comfortable option
and would have been wrong twice: B1's decision selected Sharadar SF1, SEP, SFP
and ACTIONS, none of which carries short interest, and an unregistered gap riding
behind a blocker already marked DECIDED is exactly how work goes quietly missing
(directive §9.8). So there is a distinct error, and the tests below check it says
which of *three* different things is outstanding — an unregistered source, a
registered blocker, or an engineer's unwritten query — rather than collapsing
them.

The last test is a **canary**: it asserts the short-interest table really is
absent from the mapped schema, and it is expected to start failing the day a
connector lands. Until then it is what makes the refusal evidence rather than
assertion.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import pytest

from backend.db.base import Base
from backend.features.compute import FeatureComputeRequest, resolve_as_of
from backend.features.errors import (
    AvailabilityLagViolationError,
    FeatureComputeError,
    FeatureError,
)
from backend.features.factors._fundamentals import (
    FUNDAMENTALS_BLOCKER,
    FUNDAMENTALS_TABLE,
    FundamentalsSourceUnavailableError,
)
from backend.features.factors.short_interest import (
    SHORT_INTEREST,
    SHORT_INTEREST_AVAILABILITY_LAG,
    SHORT_INTEREST_SOURCE_OF_RECORD,
    SHORT_INTEREST_TABLE,
    ShortInterestComputationNotWrittenError,
    ShortInterestSourceUnavailableError,
    require_short_interest_source,
    short_interest,
    short_interest_source_present,
)
from backend.features.factors.value import FUNDAMENTAL_AVAILABILITY_LAG
from backend.features.registry import default_registry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

COMPUTE_DATE = dt.date(2026, 3, 2)
"""An arbitrary Monday used wherever a rebalance date is needed."""

DECLARED_LAG = dt.timedelta(days=17)
"""What the module docstring argues for: 14 + 1 + 2, written out here by hand."""

PUBLICATION_GAP = dt.timedelta(days=14)
"""Eight business days from settlement date to dissemination, bounded in calendar days.

Two weekends (+4) and up to two market holidays (+2) on top of eight business
days. This is the component the whole declaration exists for: it is the size of
the foresight a settlement-date knowledge time would grant.
"""


class NoSession:
    """A stand-in that fails loudly if ``short_interest`` tries to read anything.

    The refusal must happen before any I/O. If the computation reached for the
    database, this object's missing ``execute`` would raise ``AttributeError``
    rather than the named error, and the assertion on the exception type would
    fail.
    """


def request_for(securities: tuple[int, ...] = (1, 2, 3)) -> FeatureComputeRequest:
    """Build a request for ``short_interest`` at its declared cutoff."""
    return FeatureComputeRequest(
        feature=SHORT_INTEREST.name,
        compute_date=COMPUTE_DATE,
        as_of=SHORT_INTEREST.knowledge_cutoff(COMPUTE_DATE),
        security_ids=securities,
    )


# ---------------------------------------------------------------------------
# The declaration, and the lag it exists for
# ---------------------------------------------------------------------------


def test_the_declaration_is_registered_in_the_process_wide_catalog() -> None:
    """Importing the package populates the registry the 30-feature cap is about."""
    registry = default_registry()
    assert SHORT_INTEREST.name in registry
    assert registry.spec("short_interest") is SHORT_INTEREST


def test_the_declared_availability_lag_is_seventeen_days() -> None:
    """The number the module docstring derives: 14 + 1 + 2, asserted as a literal."""
    assert SHORT_INTEREST.availability_lag == DECLARED_LAG
    assert SHORT_INTEREST_AVAILABILITY_LAG == DECLARED_LAG


def test_the_lag_covers_the_whole_settlement_to_dissemination_gap() -> None:
    """The margin's entire purpose, stated as an inequality rather than a sum.

    A connector that stamped ``knowledge_time`` from the settlement date printed
    on every row would make each observation visible about eight business days
    early. The declared lag has to be at least that gap, or the margin would leave
    part of the foresight it exists to absorb.
    """
    assert SHORT_INTEREST.availability_lag >= PUBLICATION_GAP


def test_the_lag_is_the_maximum_of_the_two_regimes_and_not_their_sum() -> None:
    """This factor reads two feeds at one cutoff, so one bound covers both legs.

    The share-count denominator comes from the B1-blocked fundamentals feed, whose
    own margin is 7 days. A lag bounds how fresh the *freshest* input may be, so 17
    days covers the 7-day requirement outright; adding them would be arithmetic on
    a bound rather than reasoning about one.
    """
    assert SHORT_INTEREST.availability_lag > FUNDAMENTAL_AVAILABILITY_LAG
    assert SHORT_INTEREST.availability_lag < dt.timedelta(days=24)


def test_the_knowledge_cutoff_is_the_stated_instant() -> None:
    """The lag's operational content on a concrete date, against a literal instant."""
    assert SHORT_INTEREST.knowledge_cutoff(COMPUTE_DATE).isoformat() == (
        "2026-02-13T00:00:00+00:00"
    )


def test_a_caller_may_not_ask_for_an_instant_fresher_than_the_lag_permits() -> None:
    """One microsecond past the cutoff is refused, before any session is opened.

    That granularity matters: a lag is not a hint about staleness, it is the
    boundary of what the feature is entitled to know, and a boundary that
    tolerates "just a bit fresher" is not a boundary.
    """
    permitted = SHORT_INTEREST.knowledge_cutoff(COMPUTE_DATE)
    assert resolve_as_of(SHORT_INTEREST, COMPUTE_DATE) == permitted
    with pytest.raises(AvailabilityLagViolationError) as raised:
        resolve_as_of(
            SHORT_INTEREST, COMPUTE_DATE, requested_as_of=permitted + dt.timedelta(microseconds=1)
        )
    assert raised.value.availability_lag == DECLARED_LAG


def test_a_caller_may_reconstruct_the_factor_at_an_older_instant() -> None:
    """Reading older data is always I1-safe: it is what a historical replay does."""
    older = SHORT_INTEREST.knowledge_cutoff(COMPUTE_DATE) - dt.timedelta(days=30)
    assert resolve_as_of(SHORT_INTEREST, COMPUTE_DATE, requested_as_of=older) == older


def test_the_declared_source_tables_are_both_feeds_the_ratio_needs() -> None:
    """Impact analysis depends on this, and neither table exists."""
    assert SHORT_INTEREST.source_tables == frozenset({SHORT_INTEREST_TABLE, FUNDAMENTALS_TABLE})


def test_the_units_state_the_numerator_and_the_denominator() -> None:
    """Directive §8: shares-over-shares is a fraction, and which shares matters."""
    assert "dimensionless fraction" in SHORT_INTEREST.units
    assert "shares sold short per share of common stock" in SHORT_INTEREST.units


def test_the_units_are_not_percent_or_basis_points() -> None:
    """The two forms that are the silent bug class this project spends effort on."""
    units = SHORT_INTEREST.units.lower()
    assert "percent" not in units
    assert "basis point" not in units


def test_the_definition_states_the_expected_premium_sign() -> None:
    """P5.4 tests a premium against a stated expectation, not a remembered one."""
    assert "Expected premium sign: NEGATIVE" in SHORT_INTEREST.definition


def test_the_definition_pins_dissemination_rather_than_the_settlement_date() -> None:
    """The whole temporal content of this factor, written where a reviewer sees it.

    The Features page (§6.4) renders the definition, and this is the sentence that
    tells whoever writes the connector which of the two dates on every row is the
    knowledge time.
    """
    definition = SHORT_INTEREST.definition
    assert "DISSEMINATED" in definition
    assert "never the settlement date" in definition
    assert "eight business days" in definition


def test_the_definition_records_that_shares_outstanding_is_not_free_float() -> None:
    """An unstated modelling choice inside a number labelled as a measurement."""
    assert "rather than free float" in SHORT_INTEREST.definition


# ---------------------------------------------------------------------------
# The refusal, and the three states it distinguishes
# ---------------------------------------------------------------------------


async def test_the_factor_raises_instead_of_returning_a_plausible_ratio() -> None:
    """The one behaviour I3 permits for a feature whose source does not exist."""
    with pytest.raises(ShortInterestSourceUnavailableError) as raised:
        await short_interest(cast("AsyncSession", NoSession()), request_for())
    assert raised.value.feature == "short_interest"
    assert raised.value.table == SHORT_INTEREST_TABLE
    assert raised.value.source_of_record == SHORT_INTEREST_SOURCE_OF_RECORD


async def test_the_refusal_says_the_gap_is_unregistered_and_names_the_source_of_record() -> None:
    """An error nobody can act on is only marginally better than a fabricated number.

    Three things have to be in the message: the missing table, the fact that
    nothing is tracking it, and where such data would have to come from. Without
    the third, "no data" is where the conversation ends.
    """
    with pytest.raises(ShortInterestSourceUnavailableError) as raised:
        await short_interest(cast("AsyncSession", NoSession()), request_for())
    message = str(raised.value)
    assert SHORT_INTEREST_TABLE in message
    assert "UNREGISTERED" in message
    assert "BLOCKERS.md" in message
    assert "FINRA" in message
    assert "settlement date" in message


async def test_the_refusal_does_not_borrow_b1_as_its_explanation() -> None:
    """B1 is DECIDED and its four selected feeds do not carry short interest.

    Naming B1 here would claim a decision covers a feed it does not, and would let
    an unregistered gap ride invisibly behind a blocker that already has an owner.
    The message may mention B1 only to say it is *not* the explanation.
    """
    with pytest.raises(ShortInterestSourceUnavailableError) as raised:
        await short_interest(cast("AsyncSession", NoSession()), request_for())
    message = str(raised.value)
    assert "B1 does not cover it" in message
    assert not hasattr(raised.value, "blocker")


async def test_the_refusal_is_part_of_the_feature_error_taxonomy() -> None:
    """A caller catching ``FeatureComputeError`` catches this without knowing the details."""
    with pytest.raises(FeatureComputeError):
        await short_interest(cast("AsyncSession", NoSession()), request_for())
    assert issubclass(ShortInterestSourceUnavailableError, FeatureComputeError)
    assert issubclass(ShortInterestComputationNotWrittenError, FeatureComputeError)
    assert issubclass(ShortInterestSourceUnavailableError, FeatureError)


async def test_the_refusal_does_not_depend_on_which_securities_were_asked_for() -> None:
    """An empty universe must not make a blocked factor look like it succeeded.

    Returning an empty array for zero securities would be a perfectly typed,
    entirely wrong answer: it says "computed, nothing to report" for a feature that
    computed nothing at all.
    """
    with pytest.raises(ShortInterestSourceUnavailableError):
        await short_interest(cast("AsyncSession", NoSession()), request_for(securities=()))


def test_the_gate_reports_the_unregistered_source_when_no_feed_exists() -> None:
    """Today's state, and the branch the factor actually takes."""
    with pytest.raises(ShortInterestSourceUnavailableError) as raised:
        require_short_interest_source("short_interest", tables=set())
    assert raised.value.feature == "short_interest"


def test_the_short_interest_gap_is_reported_ahead_of_the_registered_one() -> None:
    """Most-specific-first: an unregistered gap outranks a blocker with an owner.

    With the fundamentals feed present and short interest still missing, the
    answer must still be the short-interest error — otherwise the factor would
    report "blocked on B1", which is true of its denominator and says nothing
    about the half nobody is working on.
    """
    with pytest.raises(ShortInterestSourceUnavailableError):
        require_short_interest_source("short_interest", tables={FUNDAMENTALS_TABLE})


def test_the_gate_falls_through_to_b1_once_a_short_interest_feed_lands() -> None:
    """The denominator is separately blocked, and that error already has an owner.

    Probed against a supplied table set rather than by registering a fake table in
    process-wide metadata, which would corrupt every other test in the run.
    """
    with pytest.raises(FundamentalsSourceUnavailableError) as raised:
        require_short_interest_source("short_interest", tables={SHORT_INTEREST_TABLE})
    assert raised.value.feature == "short_interest"
    assert raised.value.blocker == FUNDAMENTALS_BLOCKER


def test_the_gate_reports_unwritten_arithmetic_once_both_feeds_have_landed() -> None:
    """After both connectors, "no feed" would be the wrong answer, so a third error fires.

    A single error covering all three states would keep reporting "no connector
    exists" long after one did, which is how a task goes quietly missing.
    """
    with pytest.raises(ShortInterestComputationNotWrittenError) as raised:
        require_short_interest_source(
            "short_interest", tables={SHORT_INTEREST_TABLE, FUNDAMENTALS_TABLE}
        )
    assert raised.value.feature == "short_interest"
    message = str(raised.value)
    assert SHORT_INTEREST_TABLE in message
    assert FUNDAMENTALS_TABLE in message
    assert "must be written" in message


def test_the_probe_reads_the_mapped_schema_rather_than_a_flag() -> None:
    """``short_interest_source_present()`` answers from ``Base.metadata``, not a constant."""
    assert short_interest_source_present({SHORT_INTEREST_TABLE}) is True
    assert short_interest_source_present(set()) is False
    assert short_interest_source_present({"price_bar", FUNDAMENTALS_TABLE}) is False


def test_canary_the_short_interest_table_is_still_absent_from_the_schema() -> None:
    """Expected to fail the day a short-interest connector lands. That is the signal.

    Until then this is what makes the refusal above evidence rather than
    assertion: the table this module says does not exist genuinely does not exist,
    while ``price_bar`` — which the price factors read — genuinely does.
    """
    mapped = set(Base.metadata.tables)
    assert SHORT_INTEREST_TABLE not in mapped
    assert FUNDAMENTALS_TABLE not in mapped
    assert "price_bar" in mapped
