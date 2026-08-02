"""Screening criteria: validation, canonical identity, and the per-name decision.

Every threshold below is a round number and every expected answer is arithmetic
a reader can check without running anything. The boundary tests matter more than
they look: the comparison is ``value >= floor``, so a name sitting exactly on a
floor is a member, and "at least" versus "more than" silently moves the edge of
the universe.

Overrides are passed as zero-argument builders rather than as ``**kwargs``
dictionaries, so each case stays fully typed under ``mypy --strict`` instead of
being smuggled past it through a widened mapping.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from backend.tests.universe.fixtures import fixture_criteria, passing_candidate
from backend.universe.criteria import (
    FILTER_ORDER,
    MAX_ADV_LOOKBACK_DAYS,
    MIN_ADV_LOOKBACK_DAYS,
    FilterOutcome,
    UniverseCandidate,
    UniverseCriteria,
    evaluate_candidate,
)
from backend.universe.errors import UniverseConsistencyError, UniverseCriteriaError

if TYPE_CHECKING:
    from collections.abc import Callable


class TestCriteriaValidation:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: fixture_criteria(min_adv_usd="0"),
            lambda: fixture_criteria(min_price_usd="0"),
            lambda: fixture_criteria(min_market_cap_usd="0"),
            lambda: fixture_criteria(min_adv_usd="-1"),
            lambda: fixture_criteria(min_price_usd="-1"),
            lambda: fixture_criteria(min_market_cap_usd="-1"),
        ],
        ids=["adv=0", "price=0", "cap=0", "adv<0", "price<0", "cap<0"],
    )
    def test_a_floor_that_is_not_strictly_positive_is_refused(
        self, build: Callable[[], UniverseCriteria]
    ) -> None:
        # A floor of zero is not a lenient screen, it is an absent one: nothing
        # fails `value >= 0`, yet it still appears in the criteria hash and in
        # the waterfall with zero removals.
        with pytest.raises(UniverseCriteriaError, match="strictly positive"):
            build()

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_floor_is_refused(self, bad: str) -> None:
        with pytest.raises(UniverseCriteriaError, match="finite"):
            fixture_criteria(min_adv_usd=bad)

    def test_an_empty_exchange_whitelist_is_refused(self) -> None:
        with pytest.raises(UniverseCriteriaError, match="empty"):
            fixture_criteria(allowed_exchanges=frozenset())

    def test_a_blank_exchange_code_is_refused(self) -> None:
        with pytest.raises(UniverseCriteriaError, match="blank"):
            fixture_criteria(allowed_exchanges=frozenset({"XNYS", "  "}))

    @pytest.mark.parametrize(
        "lookback", [MIN_ADV_LOOKBACK_DAYS - 1, 0, -5, MAX_ADV_LOOKBACK_DAYS + 1]
    )
    def test_an_adv_lookback_outside_the_permitted_window_is_refused(self, lookback: int) -> None:
        with pytest.raises(UniverseCriteriaError, match="adv_lookback_days"):
            fixture_criteria(adv_lookback_days=lookback)

    @pytest.mark.parametrize("lookback", [MIN_ADV_LOOKBACK_DAYS, 20, MAX_ADV_LOOKBACK_DAYS])
    def test_the_permitted_window_boundaries_are_accepted(self, lookback: int) -> None:
        assert fixture_criteria(adv_lookback_days=lookback).adv_lookback_days == lookback

    def test_a_bool_is_not_an_integer_lookback(self) -> None:
        # True == 1 in Python, so a slipped boolean would compare as though it
        # were a real lookback rather than being caught by the window check.
        with pytest.raises(UniverseCriteriaError, match="bool is refused"):
            UniverseCriteria(
                min_adv_usd=Decimal("1"),
                min_price_usd=Decimal("1"),
                min_market_cap_usd=Decimal("1"),
                allowed_exchanges=frozenset({"XNYS"}),
                require_borrow=False,
                # bool is a subtype of int, so this type-checks; the refusal is
                # a runtime one, which is exactly why it has to exist.
                adv_lookback_days=True,
            )

    def test_a_truthy_value_cannot_turn_the_borrow_screen_on(self) -> None:
        with pytest.raises(UniverseCriteriaError, match="require_borrow must be a bool"):
            UniverseCriteria(
                min_adv_usd=Decimal("1"),
                min_price_usd=Decimal("1"),
                min_market_cap_usd=Decimal("1"),
                allowed_exchanges=frozenset({"XNYS"}),
                require_borrow=1,  # type: ignore[arg-type]
            )


class TestAppliedFilters:
    def test_the_borrow_screen_is_absent_when_it_is_not_required(self) -> None:
        applied = fixture_criteria(require_borrow=False).applied_filters()
        assert applied == ("exchange", "price", "market_cap", "adv")
        assert "borrow" not in applied

    def test_the_borrow_screen_appears_when_required(self) -> None:
        assert fixture_criteria(require_borrow=True).applied_filters() == FILTER_ORDER

    @pytest.mark.parametrize("require_borrow", [True, False])
    def test_applied_filters_are_a_subsequence_of_the_declared_order(
        self, require_borrow: bool
    ) -> None:
        applied = fixture_criteria(require_borrow=require_borrow).applied_filters()
        positions = [FILTER_ORDER.index(name) for name in applied]
        assert positions == sorted(positions)


class TestCriteriaIdentity:
    def test_the_hash_is_64_lowercase_hex_characters(self) -> None:
        digest = fixture_criteria().criteria_hash()
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_equal_amounts_written_differently_hash_the_same(self) -> None:
        # 1E+6, 1000000 and 1000000.00 are one number written three ways. Hashing
        # their str() forms would give one set of criteria three identities.
        spellings = ["1E+6", "1000000", "1000000.00"]
        digests = {fixture_criteria(min_adv_usd=spelling).criteria_hash() for spelling in spellings}
        assert len(digests) == 1

    def test_the_exchange_set_order_does_not_change_the_hash(self) -> None:
        first = fixture_criteria(allowed_exchanges=frozenset({"XNYS", "XNAS", "ARCX"}))
        second = fixture_criteria(allowed_exchanges=frozenset({"ARCX", "XNAS", "XNYS"}))
        assert first.criteria_hash() == second.criteria_hash()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: fixture_criteria(min_adv_usd="1000001"),
            lambda: fixture_criteria(min_price_usd="6"),
            lambda: fixture_criteria(min_market_cap_usd="300000001"),
            lambda: fixture_criteria(require_borrow=True),
            lambda: fixture_criteria(adv_lookback_days=21),
            lambda: fixture_criteria(allowed_exchanges=frozenset({"XNYS"})),
        ],
        ids=["adv", "price", "cap", "borrow", "lookback", "exchanges"],
    )
    def test_changing_any_criterion_changes_the_hash(
        self, build: Callable[[], UniverseCriteria]
    ) -> None:
        assert build().criteria_hash() != fixture_criteria().criteria_hash()

    def test_the_config_round_trips_through_json_shaped_values(self) -> None:
        original = fixture_criteria(require_borrow=True, adv_lookback_days=63)
        restored = UniverseCriteria.from_config(original.as_config())
        assert restored == original
        assert restored.criteria_hash() == original.criteria_hash()

    def test_the_declared_filter_order_is_part_of_the_stored_config(self) -> None:
        assert fixture_criteria().as_config()["filter_order"] == list(FILTER_ORDER)

    def test_a_stored_config_under_a_different_filter_order_is_refused(self) -> None:
        # A snapshot whose waterfall was attributed under another order is not
        # the same measurement as one attributed under this order.
        config = dict(fixture_criteria().as_config())
        config["filter_order"] = list(reversed(FILTER_ORDER))
        with pytest.raises(UniverseCriteriaError, match="filter_order"):
            UniverseCriteria.from_config(config)

    @pytest.mark.parametrize(
        "key",
        ["adv_lookback_days", "allowed_exchanges", "filter_order", "require_borrow", "min_adv_usd"],
    )
    def test_a_missing_key_in_a_stored_config_is_refused(self, key: str) -> None:
        config = dict(fixture_criteria().as_config())
        del config[key]
        with pytest.raises(UniverseCriteriaError, match="missing key"):
            UniverseCriteria.from_config(config)

    def test_a_stored_amount_that_is_not_a_decimal_string_is_refused(self) -> None:
        config = dict(fixture_criteria().as_config())
        config["min_adv_usd"] = 1000000  # a JSON number, not the canonical string
        with pytest.raises(UniverseCriteriaError, match="decimal string"):
            UniverseCriteria.from_config(config)


class TestCandidateValidation:
    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_security_id_is_refused(self, bad: int) -> None:
        with pytest.raises(UniverseConsistencyError, match="identity-anchor key"):
            passing_candidate(bad)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: passing_candidate(1, price_usd="-1"),
            lambda: passing_candidate(1, adv_usd="-1"),
            lambda: passing_candidate(1, market_cap_usd="-1"),
            lambda: passing_candidate(1, price_usd="NaN"),
            lambda: passing_candidate(1, adv_usd="Infinity"),
            lambda: passing_candidate(1, market_cap_usd="NaN"),
        ],
        ids=["price<0", "adv<0", "cap<0", "price=NaN", "adv=inf", "cap=NaN"],
    )
    def test_an_impossible_measurement_is_a_corrupt_input_not_an_exclusion(
        self, build: Callable[[], UniverseCandidate]
    ) -> None:
        with pytest.raises(UniverseConsistencyError, match="corrupt input"):
            build()

    def test_zero_is_a_legitimate_measurement(self) -> None:
        # A name that traded no shares has an ADV of exactly zero. That is a
        # measurement, and it fails the screen; it is not a corrupt input.
        assert passing_candidate(1, adv_usd="0").adv_usd == Decimal("0")


class TestEvaluateCandidate:
    def test_a_candidate_passing_everything_is_a_member_with_no_failures(self) -> None:
        outcome = evaluate_candidate(passing_candidate(1), fixture_criteria())
        assert outcome == FilterOutcome(security_id=1, included=True, failed_filters=())
        assert outcome.attributed_filter is None

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: passing_candidate(1, exchange="XLON"), ("exchange",)),
            (lambda: passing_candidate(1, price_usd="4.99"), ("price",)),
            (lambda: passing_candidate(1, market_cap_usd="299999999"), ("market_cap",)),
            (lambda: passing_candidate(1, adv_usd="999999"), ("adv",)),
        ],
        ids=["exchange", "price", "market_cap", "adv"],
    )
    def test_each_screen_fails_on_its_own_input(
        self, build: Callable[[], UniverseCandidate], expected: tuple[str, ...]
    ) -> None:
        outcome = evaluate_candidate(build(), fixture_criteria())
        assert outcome.failed_filters == expected
        assert outcome.included is False
        assert outcome.attributed_filter == expected[0]

    @pytest.mark.parametrize(
        ("build", "screen"),
        [
            (lambda: passing_candidate(1, price_usd="5"), "price"),
            (lambda: passing_candidate(1, market_cap_usd="300000000"), "market_cap"),
            (lambda: passing_candidate(1, adv_usd="1000000"), "adv"),
        ],
        ids=["price", "market_cap", "adv"],
    )
    def test_a_name_exactly_on_a_floor_is_included(
        self, build: Callable[[], UniverseCandidate], screen: str
    ) -> None:
        # The comparison is >=. This test is the definition of the boundary.
        outcome = evaluate_candidate(build(), fixture_criteria())
        assert screen not in outcome.failed_filters
        assert outcome.included is True

    @pytest.mark.parametrize(
        "build",
        [
            lambda: passing_candidate(1, price_usd=None),
            lambda: passing_candidate(1, adv_usd=None),
            lambda: passing_candidate(1, market_cap_usd=None),
        ],
        ids=["price", "adv", "market_cap"],
    )
    def test_an_unknowable_input_fails_its_screen_rather_than_passing_it(
        self, build: Callable[[], UniverseCandidate]
    ) -> None:
        # Excluding a name we cannot show to be eligible drops a trade; including
        # one we cannot show to be eligible invents a trade. Only the second
        # corrupts a backtest.
        assert evaluate_candidate(build(), fixture_criteria()).included is False

    def test_every_failure_is_recorded_not_only_the_attributed_one(self) -> None:
        candidate = passing_candidate(1, exchange="XLON", price_usd="1", adv_usd="1")
        outcome = evaluate_candidate(candidate, fixture_criteria())
        assert outcome.failed_filters == ("exchange", "price", "adv")
        assert outcome.attributed_filter == "exchange"

    def test_failures_come_back_in_the_declared_order(self) -> None:
        candidate = passing_candidate(
            1,
            exchange="XLON",
            price_usd="1",
            market_cap_usd="1",
            adv_usd="1",
            borrow_available=False,
        )
        outcome = evaluate_candidate(candidate, fixture_criteria(require_borrow=True))
        assert outcome.failed_filters == FILTER_ORDER

    def test_the_borrow_screen_is_not_applied_when_it_is_not_required(self) -> None:
        candidate = passing_candidate(1, borrow_available=None)
        assert evaluate_candidate(candidate, fixture_criteria(require_borrow=False)).included

    @pytest.mark.parametrize("borrow", [None, False])
    def test_an_unborrowable_or_unknown_name_fails_the_borrow_screen(
        self, borrow: bool | None
    ) -> None:
        candidate = passing_candidate(1, borrow_available=borrow)
        outcome = evaluate_candidate(candidate, fixture_criteria(require_borrow=True))
        assert outcome.failed_filters == ("borrow",)

    def test_the_exchange_comparison_is_verbatim(self) -> None:
        # A case mismatch is a symbology defect worth surfacing as an exclusion,
        # not something to alias away.
        outcome = evaluate_candidate(passing_candidate(1, exchange="xnys"), fixture_criteria())
        assert outcome.failed_filters == ("exchange",)


class TestFilterOutcomeValidation:
    def test_an_included_outcome_may_not_carry_failures(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="failed nothing"):
            FilterOutcome(security_id=1, included=True, failed_filters=("price",))

    def test_an_excluded_outcome_must_carry_a_failure(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="failed nothing"):
            FilterOutcome(security_id=1, included=False, failed_filters=())

    def test_an_unknown_screen_name_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="unknown screen"):
            FilterOutcome(security_id=1, included=False, failed_filters=("liquidity",))

    def test_failures_out_of_declared_order_are_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="FILTER_ORDER"):
            FilterOutcome(security_id=1, included=False, failed_filters=("price", "exchange"))

    def test_a_repeated_failure_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="FILTER_ORDER"):
            FilterOutcome(security_id=1, included=False, failed_filters=("price", "price"))


def test_the_declared_filter_order_is_the_five_phase_four_screens() -> None:
    # Directive Phase 4 names exactly these five. The order is frozen because it
    # decides every waterfall's attribution, and it is hashed into the criteria.
    assert FILTER_ORDER == ("exchange", "price", "market_cap", "adv", "borrow")


def test_a_candidate_is_immutable() -> None:
    candidate = UniverseCandidate(security_id=1, exchange="XNYS")
    with pytest.raises(AttributeError):
        candidate.exchange = "XNAS"  # type: ignore[misc]
