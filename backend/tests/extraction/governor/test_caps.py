"""P7.7: there is no default cap, and a cap belongs to exactly one provider (B4, §6.5).

The refusals in this file are the feature. B4 says the governor refuses to run
without configured caps, and the failure mode it exists to prevent is not a
crash — it is a governor that runs, shows a green gauge, and enforces a number
nobody chose.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.extraction.governor.caps import (
    CapBook,
    CapPolicy,
    ProviderCap,
    SpendWindow,
)
from backend.extraction.governor.errors import (
    CapNotConfiguredError,
    CapsNotConfiguredError,
    CurrencyMismatchError,
    GovernorConfigurationError,
)
from backend.extraction.governor.estimate import Money
from backend.extraction.providers.catalog import Provider
from backend.tests.extraction.governor.doubles import (
    CHEAP_MODEL,
    OTHER_PROVIDER_MODEL,
    cap,
    usd,
)

# --------------------------------------------------------------------------
# No default cap (B4)
# --------------------------------------------------------------------------


def test_an_empty_cap_book_refuses_to_exist() -> None:
    """B4, stated in the constructor: no caps means nothing may be spent."""
    with pytest.raises(CapsNotConfiguredError, match="number nobody chose"):
        CapBook({})
    with pytest.raises(CapsNotConfiguredError):
        CapBook.of()


def test_a_provider_absent_from_the_book_is_refused_not_defaulted() -> None:
    """Missing is 'ungoverned', which is the same answer as 'no'."""
    book = CapBook.of(cap(provider=Provider.ANTHROPIC))
    with pytest.raises(CapNotConfiguredError, match="no spend cap is configured"):
        book.cap_for(Provider.OPENAI)


def test_two_caps_for_one_provider_are_refused_rather_than_silently_merged() -> None:
    """Keeping the last would make the budget depend on argument order."""
    with pytest.raises(GovernorConfigurationError, match="two caps"):
        CapBook.of(cap(daily="1.00"), cap(daily="2.00"))


def test_a_cap_filed_under_another_providers_key_is_refused() -> None:
    book_input = {Provider.OPENAI: cap(provider=Provider.ANTHROPIC)}
    with pytest.raises(GovernorConfigurationError, match="filed under"):
        CapBook(book_input)


# --------------------------------------------------------------------------
# One cap, one currency, non-negative
# --------------------------------------------------------------------------


def test_a_cap_may_not_be_denominated_in_two_currencies() -> None:
    with pytest.raises(CurrencyMismatchError, match="two currencies"):
        ProviderCap(
            provider=Provider.ANTHROPIC,
            daily_limit=usd("1"),
            monthly_limit=Money(Decimal("1"), "EUR"),
            policy=CapPolicy.HALT,
        )


def test_a_negative_limit_is_refused_and_zero_is_not() -> None:
    """Zero is a legal, deliberate 'spend nothing'; negative is incoherent."""
    assert cap(daily="0", monthly="0").daily_limit == usd("0")
    with pytest.raises(ValueError, match="must not be negative"):
        cap(daily="-0.01")


# --------------------------------------------------------------------------
# Halt versus degrade is a per-provider statement about a specific pair of models
# --------------------------------------------------------------------------


def test_degrade_without_a_target_is_refused() -> None:
    with pytest.raises(ValueError, match="model to degrade to"):
        ProviderCap(
            provider=Provider.ANTHROPIC,
            daily_limit=usd("1"),
            monthly_limit=usd("10"),
            policy=CapPolicy.DEGRADE,
        )


def test_a_halting_cap_may_not_carry_a_fallback_it_will_never_use() -> None:
    """A fallback under a halt policy reads as protection that does not exist."""
    with pytest.raises(ValueError, match="never be reached"):
        cap(policy=CapPolicy.HALT, degrade_to=CHEAP_MODEL)


def test_degrading_across_providers_is_refused() -> None:
    """A cross-provider substitution spends a different cap under a different key."""
    with pytest.raises(GovernorConfigurationError, match="belongs to 'openai'"):
        cap(policy=CapPolicy.DEGRADE, degrade_to=OTHER_PROVIDER_MODEL)


def test_a_halting_cap_has_no_degrade_target_to_give() -> None:
    with pytest.raises(GovernorConfigurationError, match="nothing to substitute"):
        cap().require_degrade_target()


def test_a_degrading_cap_returns_its_target() -> None:
    assert cap(policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL).require_degrade_target() == (
        CHEAP_MODEL
    )


# --------------------------------------------------------------------------
# Windows are UTC calendar periods
# --------------------------------------------------------------------------


def test_window_keys_are_utc_calendar_periods() -> None:
    at = dt.datetime(2026, 8, 2, 23, 59, tzinfo=dt.UTC)
    assert SpendWindow.DAILY.key(at) == "2026-08-02"
    assert SpendWindow.MONTHLY.key(at) == "2026-08"


def test_a_local_time_is_converted_to_utc_rather_than_read_as_utc() -> None:
    """23:00 in UTC+02:00 is 21:00 UTC — the same day here, and the test proves the shift.

    The interesting case is the one that crosses: 01:00 on the 3rd at UTC+02:00
    is 23:00 on the 2nd UTC, so it spends the 2nd's budget.
    """
    east = dt.timezone(dt.timedelta(hours=2))
    crossing = dt.datetime(2026, 8, 3, 1, 0, tzinfo=east)
    assert SpendWindow.DAILY.key(crossing) == "2026-08-02"
    assert SpendWindow.MONTHLY.key(dt.datetime(2026, 9, 1, 1, 0, tzinfo=east)) == "2026-08"


def test_a_naive_timestamp_names_no_window() -> None:
    """Interpreted as UTC on one host and local on another, the two would disagree."""
    with pytest.raises(ValueError, match="naive"):
        SpendWindow.DAILY.key(dt.datetime(2026, 8, 2, 12, 0))  # noqa: DTZ001 — the point


def test_both_windows_are_always_constrained() -> None:
    """Daily bounds a burst, monthly bounds a backfill; they are not alternatives."""
    limits = cap(daily="1.00", monthly="10.00").limits
    assert set(limits) == {SpendWindow.DAILY, SpendWindow.MONTHLY}
    windows = cap().windows(dt.datetime(2026, 8, 2, 12, 0, tzinfo=dt.UTC))
    assert windows == {SpendWindow.DAILY: "2026-08-02", SpendWindow.MONTHLY: "2026-08"}
