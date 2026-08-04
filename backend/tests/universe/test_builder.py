"""The universe builder: refusals, the survivorship predicate, assembly, screening.

The centrepiece is :class:`TestSurvivorshipBiasElimination` — gate G4. Everything
else in this file exists so that test cannot pass for an accidental reason.

Nothing here touches a database. The pure steps are exercised directly, and the
orchestration is exercised against :class:`~backend.tests.universe.fixtures.FakeAsOfSession`,
which serves canned rows and executes nothing. What the SQL itself does is
``backend/tests/integration/test_universe_db.py``'s job.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, cast

import pytest

from backend.db.asof import AS_OF_INFO_KEY
from backend.tests.universe.fixtures import (
    FIXTURE_AS_OF,
    FIXTURE_REBALANCE_DATE,
    FakeAsOfSession,
    RawRowSession,
    fixture_criteria,
    listed_security,
    passing_candidate,
    weekday_bars,
)
from backend.universe import builder
from backend.universe.builder import (
    ADV_WINDOW_PADDING_DAYS,
    FILTERS_WITH_AVAILABLE_INPUTS,
    UNAVAILABLE_FILTER_INPUTS,
    DailyBar,
    ListedSecurity,
    adv_window_calendar_days,
    adv_window_start,
    assemble_candidates,
    bound_as_of,
    build_universe,
    median_dollar_volume_usd,
    read_listings,
    read_price_window,
    require_available_inputs,
    screen_candidates,
)
from backend.universe.criteria import FILTER_ORDER, UniverseCandidate
from backend.universe.errors import (
    UniverseConsistencyError,
    UniverseInputUnavailableError,
    UniverseSessionError,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _as_session(fake: FakeAsOfSession) -> AsyncSession:
    """Present the canned-row double as the session type the builder declares."""
    return cast("AsyncSession", fake)


def _with_supplied_market_data(candidate: UniverseCandidate) -> UniverseCandidate:
    """Attach a market cap and a borrow flag that no code path can produce today.

    Both inputs are blocked (B1, B2), so :func:`assemble_candidates` leaves them
    ``None`` and every real build refuses before reaching the screen. Supplying
    them here — explicitly, in the test, on a value object the test already owns
    — is what lets the screening arithmetic *downstream* of that refusal be
    exercised at all. Nothing in ``backend/universe`` does this, and the test
    that asserts the refusal is what keeps it that way.
    """
    return dataclasses.replace(
        candidate, market_cap_usd=Decimal("1000000000"), borrow_available=True
    )


# --- Session scoping (I1) --------------------------------------------------


class TestSessionScoping:
    def test_a_session_with_no_as_of_bound_is_refused(self) -> None:
        fake = FakeAsOfSession()
        fake.info.clear()
        with pytest.raises(UniverseSessionError, match="not scoped by as_of"):
            bound_as_of(_as_session(fake))

    def test_a_naive_as_of_bound_is_refused(self) -> None:
        fake = FakeAsOfSession()
        fake.info[AS_OF_INFO_KEY] = dt.datetime(2026, 1, 2)  # noqa: DTZ001 — the defect under test
        with pytest.raises(UniverseSessionError, match="timezone-aware UTC"):
            bound_as_of(_as_session(fake))

    def test_a_non_utc_as_of_bound_is_refused(self) -> None:
        fake = FakeAsOfSession()
        offset = dt.timezone(dt.timedelta(hours=-5))
        fake.info[AS_OF_INFO_KEY] = dt.datetime(2026, 1, 2, tzinfo=offset)
        with pytest.raises(UniverseSessionError, match="timezone-aware UTC"):
            bound_as_of(_as_session(fake))

    def test_a_bound_that_is_not_a_datetime_is_refused(self) -> None:
        fake = FakeAsOfSession()
        fake.info[AS_OF_INFO_KEY] = "2026-01-02T00:00:00Z"
        with pytest.raises(UniverseSessionError, match="not a datetime"):
            bound_as_of(_as_session(fake))

    def test_a_scoped_session_yields_its_instant(self) -> None:
        assert bound_as_of(_as_session(FakeAsOfSession())) == FIXTURE_AS_OF

    async def test_an_unscoped_session_is_refused_before_any_statement_is_issued(self) -> None:
        fake = FakeAsOfSession(listings=[listed_security(1)], bars=weekday_bars(1))
        fake.info.clear()
        with pytest.raises(UniverseSessionError):
            await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert fake.executed == []


# --- Unavailable filter inputs (I3) ----------------------------------------


class TestRefusalWhenAFilterInputIsUnavailable:
    def test_the_screens_split_into_computable_and_blocked(self) -> None:
        # The split is the honest state of the system, asserted rather than
        # described: exchange/price/adv come from tables that exist, market_cap
        # and borrow come from sources that do not.
        assert FILTERS_WITH_AVAILABLE_INPUTS == ("exchange", "price", "adv")
        assert set(UNAVAILABLE_FILTER_INPUTS) == {"market_cap", "borrow"}
        assert set(FILTERS_WITH_AVAILABLE_INPUTS) | set(UNAVAILABLE_FILTER_INPUTS) == set(
            FILTER_ORDER
        )

    def test_the_market_cap_screen_refuses_and_names_its_blocker(self) -> None:
        with pytest.raises(UniverseInputUnavailableError) as caught:
            require_available_inputs(fixture_criteria())
        assert caught.value.filter_name == "market_cap"
        assert caught.value.blocker == "B1"
        assert "shares-outstanding" in caught.value.source

    def test_the_borrow_screen_refuses_and_names_its_blocker(self) -> None:
        # market_cap is checked first (FILTER_ORDER), so reaching the borrow
        # refusal requires temporarily removing the market-cap entry. This is
        # the only way to observe the second refusal until B1 resolves.
        remaining = {"borrow": UNAVAILABLE_FILTER_INPUTS["borrow"]}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", remaining)
            with pytest.raises(UniverseInputUnavailableError) as caught:
                require_available_inputs(fixture_criteria(require_borrow=True))
        assert caught.value.filter_name == "borrow"
        assert caught.value.blocker == "B2"

    def test_a_borrow_screen_that_is_switched_off_needs_no_borrow_feed(self) -> None:
        # "Not applied" and "applied and passed everything" are different facts,
        # and only the first can be honoured without the feed.
        remaining = {"borrow": UNAVAILABLE_FILTER_INPUTS["borrow"]}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", remaining)
            require_available_inputs(fixture_criteria(require_borrow=False))

    def test_the_refusal_message_says_why_skipping_the_screen_is_worse(self) -> None:
        with pytest.raises(UniverseInputUnavailableError) as caught:
            require_available_inputs(fixture_criteria())
        message = str(caught.value)
        assert "refused rather than run without this screen" in message
        assert "B1" in message

    async def test_the_build_refuses_before_reading_anything(self) -> None:
        # The whole point: the caller is never left wondering whether a partial
        # universe was computed, and no name is ever screened on four of five.
        fake = FakeAsOfSession(listings=[listed_security(1)], bars=weekday_bars(1))
        with pytest.raises(UniverseInputUnavailableError):
            await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert fake.executed == []

    async def test_the_build_never_under_filters_by_dropping_the_blocked_screen(self) -> None:
        # The failure this package exists to prevent, stated as a test: a name
        # that would pass every *available* screen must NOT come back as a
        # member while the market-cap screen cannot run.
        fake = FakeAsOfSession(listings=[listed_security(1)], bars=weekday_bars(1))
        with pytest.raises(UniverseInputUnavailableError):
            await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )

    async def test_lifting_the_gate_without_a_source_empties_the_universe(self) -> None:
        """The counterfactual the refusal protects against, made visible.

        With the availability gate removed but no market-cap source wired, every
        candidate's market cap is ``None`` and every name fails the screen. The
        build "succeeds" and returns an empty universe — which reads as a
        mistake in the criteria rather than as a missing feed. That is precisely
        why the gate refuses instead of proceeding.
        """
        fake = FakeAsOfSession(listings=[listed_security(1)], bars=weekday_bars(1))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            snapshot = await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert snapshot.candidate_count == 1
        assert snapshot.members == ()
        outcome = snapshot.outcome_for(1)
        assert outcome is not None
        assert outcome.failed_filters == ("market_cap",)


# --- The listing predicate: gate G4 ----------------------------------------


class TestListingPredicate:
    @pytest.mark.parametrize(
        ("delisted_on", "expected"),
        [
            (None, True),
            (dt.date(2020, 4, 1), True),  # delisted the day after — still listed
            (FIXTURE_REBALANCE_DATE, True),  # delisting date is the last trading day
            (dt.date(2020, 3, 30), False),  # delisted the day before
        ],
        ids=["never", "delisted-tomorrow", "delisted-today", "delisted-yesterday"],
    )
    def test_delisting_is_compared_against_the_rebalance_date(
        self, delisted_on: dt.date | None, expected: bool
    ) -> None:
        listing = listed_security(1, delisted_on=delisted_on)
        assert listing.is_listed_on(FIXTURE_REBALANCE_DATE) is expected

    @pytest.mark.parametrize(
        ("first_listed_on", "expected"),
        [
            (None, True),
            (dt.date(2019, 1, 1), True),
            (FIXTURE_REBALANCE_DATE, True),  # an IPO trades on its first day
            (dt.date(2020, 4, 1), False),  # not yet listed
        ],
        ids=["unknown", "long-listed", "listed-today", "lists-tomorrow"],
    )
    def test_first_listing_is_compared_against_the_rebalance_date(
        self, first_listed_on: dt.date | None, expected: bool
    ) -> None:
        listing = listed_security(1, first_listed_on=first_listed_on)
        assert listing.is_listed_on(FIXTURE_REBALANCE_DATE) is expected

    def test_a_name_listed_and_delisted_entirely_after_the_date_is_absent(self) -> None:
        listing = listed_security(
            1, first_listed_on=dt.date(2021, 1, 4), delisted_on=dt.date(2022, 6, 30)
        )
        assert listing.is_listed_on(FIXTURE_REBALANCE_DATE) is False


class TestSurvivorshipBiasElimination:
    """Gate G4: a name delisted *after* a past rebalance date is in that universe.

    The setup is the one every research read is actually in. The as-of instant
    is 2026 — years after the rebalance date — so the store returns the identity
    version that already records the 2020 delisting. A builder that asks that row
    "is this name still listed" gets "no" and drops it. A builder that asks "was
    it listed on 2020-03-31" gets "yes" and keeps it.

    The bias is invisible without this test: the universe is still the right sort
    of size, every screen still runs, no error is raised, and the only symptom is
    that every backtest downstream looks better than the strategy was.
    """

    SURVIVOR = 1
    DELISTED_AFTER = 2
    DELISTED_BEFORE = 3

    def _listings(self) -> tuple[ListedSecurity, ...]:
        return (
            listed_security(self.SURVIVOR),
            # Acquired three months after the rebalance date. On the rebalance
            # date it was an ordinary, tradeable listed company.
            listed_security(self.DELISTED_AFTER, delisted_on=dt.date(2020, 6, 30)),
            # Already gone a year before the rebalance date: correctly absent.
            listed_security(self.DELISTED_BEFORE, delisted_on=dt.date(2019, 3, 29)),
        )

    def _bars(self) -> tuple[DailyBar, ...]:
        return tuple(
            bar
            for security_id in (self.SURVIVOR, self.DELISTED_AFTER, self.DELISTED_BEFORE)
            for bar in weekday_bars(security_id)
        )

    def test_the_later_delisted_name_is_screened_at_all(self) -> None:
        candidates = assemble_candidates(
            self._listings(),
            self._bars(),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        considered = {candidate.security_id for candidate in candidates}
        assert self.DELISTED_AFTER in considered
        assert self.SURVIVOR in considered
        assert self.DELISTED_BEFORE not in considered

    def test_the_later_delisted_name_is_a_member_of_the_past_universe(self) -> None:
        candidates = assemble_candidates(
            self._listings(),
            self._bars(),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        snapshot = screen_candidates(
            [_with_supplied_market_data(candidate) for candidate in candidates],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
            as_of=FIXTURE_AS_OF,
        )
        assert snapshot.members == (self.SURVIVOR, self.DELISTED_AFTER)
        assert snapshot.as_of > dt.datetime(
            FIXTURE_REBALANCE_DATE.year,
            FIXTURE_REBALANCE_DATE.month,
            FIXTURE_REBALANCE_DATE.day,
            tzinfo=dt.UTC,
        ), "the read must be made after the delisting for the test to mean anything"

    async def test_the_builder_considers_the_later_delisted_name_end_to_end(self) -> None:
        # The same property through build_universe itself, with the availability
        # gate lifted (no market-cap source exists, so membership is not
        # reachable — but candidacy is, and candidacy is where the bias enters).
        fake = FakeAsOfSession(listings=self._listings(), bars=self._bars())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            snapshot = await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        considered = {outcome.security_id for outcome in snapshot.outcomes}
        assert considered == {self.SURVIVOR, self.DELISTED_AFTER}
        delisted = snapshot.outcome_for(self.DELISTED_AFTER)
        assert delisted is not None
        assert "exchange" not in delisted.failed_filters
        assert "price" not in delisted.failed_filters
        assert "adv" not in delisted.failed_filters

    def test_a_survivors_only_predicate_would_lose_the_name(self) -> None:
        """Non-vacuity: the naive predicate this test exists to forbid.

        Spelled out rather than described, so the test file itself records what
        the failure looks like. If :meth:`ListedSecurity.is_listed_on` were ever
        rewritten as ``delisted_on is None``, the assertions above would go from
        passing to failing — this shows the two answers differ on this fixture.
        """
        listing = listed_security(self.DELISTED_AFTER, delisted_on=dt.date(2020, 6, 30))
        survivors_only = listing.delisted_on is None
        assert survivors_only is False
        assert listing.is_listed_on(FIXTURE_REBALANCE_DATE) is True


# --- ADV window and median -------------------------------------------------


class TestAdvWindow:
    @pytest.mark.parametrize("lookback", [2, 5, 20, 63, 126, 252])
    def test_the_calendar_span_holds_the_requested_trading_days_with_room(
        self, lookback: int
    ) -> None:
        span = adv_window_calendar_days(lookback)
        # Weekends alone remove two days in seven. The span must still leave
        # room for the requested bars plus a realistic holiday count.
        weekday_days = span * 5 // 7
        assert weekday_days >= lookback + 5, (lookback, span, weekday_days)

    def test_the_span_grows_with_the_lookback(self) -> None:
        spans = [adv_window_calendar_days(n) for n in range(2, 253)]
        assert spans == sorted(spans)

    def test_the_padding_is_included(self) -> None:
        assert adv_window_calendar_days(2) == 3 + ADV_WINDOW_PADDING_DAYS

    def test_the_window_is_inclusive_of_the_rebalance_date(self) -> None:
        start = adv_window_start(FIXTURE_REBALANCE_DATE, 20)
        span = (FIXTURE_REBALANCE_DATE - start).days + 1
        assert span == adv_window_calendar_days(20)


class TestMedianDollarVolume:
    def test_an_odd_count_takes_the_middle_value(self) -> None:
        volumes = [Decimal("1"), Decimal("5"), Decimal("100")]
        assert median_dollar_volume_usd(volumes) == Decimal("5")

    def test_an_even_count_averages_the_two_middle_values(self) -> None:
        volumes = [Decimal("1"), Decimal("4"), Decimal("6"), Decimal("100")]
        assert median_dollar_volume_usd(volumes) == Decimal("5")

    def test_the_input_order_does_not_matter(self) -> None:
        forward = [Decimal(n) for n in (1, 2, 3, 4, 5)]
        assert median_dollar_volume_usd(forward) == median_dollar_volume_usd(forward[::-1])

    def test_a_spike_does_not_drag_the_median(self) -> None:
        # The reason the screen uses a median: one earnings-day volume spike
        # must not qualify an otherwise untradeable name.
        quiet = [Decimal("1000")] * 19
        assert median_dollar_volume_usd([*quiet, Decimal("1000000000")]) == Decimal("1000")

    def test_the_halving_is_exact_at_realistic_magnitudes(self) -> None:
        # Two adjacent values around $10 trillion with six decimal places: about
        # twenty significant digits, more than a default Decimal context leaves
        # room for after summing.
        low = Decimal("9999999999999.000001")
        high = Decimal("9999999999999.000003")
        assert median_dollar_volume_usd([low, high]) == Decimal("9999999999999.000002")

    def test_an_empty_window_has_no_median(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="median of zero"):
            median_dollar_volume_usd([])


# --- Assembly --------------------------------------------------------------


class TestAssembleCandidates:
    def test_the_price_is_the_unadjusted_close_of_the_most_recent_bar(self) -> None:
        bars = [
            *weekday_bars(1, count=19, last_date=dt.date(2020, 3, 30), close_usd="7"),
            DailyBar(
                security_id=1,
                trade_date=FIXTURE_REBALANCE_DATE,
                close_raw_usd=Decimal("11"),
                volume_shares=1_000_000,
            ),
        ]
        candidates = assemble_candidates(
            [listed_security(1)],
            bars,
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].price_usd == Decimal("11")

    def test_the_adv_is_the_median_of_price_times_volume(self) -> None:
        # 20 identical bars of $10 x 1,000,000 shares. The median of a constant
        # series is that constant: $10,000,000/day.
        candidates = assemble_candidates(
            [listed_security(1)],
            weekday_bars(1, count=20, close_usd="10", volume_shares=1_000_000),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].adv_usd == Decimal("10000000")

    def test_only_the_last_lookback_bars_enter_the_median(self) -> None:
        # 40 bars available, 20 requested. The older twenty are ten times as
        # heavy and must not move the answer.
        old = weekday_bars(1, count=20, last_date=dt.date(2020, 3, 2), volume_shares=10_000_000)
        recent = weekday_bars(1, count=20, volume_shares=1_000_000)
        candidates = assemble_candidates(
            [listed_security(1)],
            [*old, *recent],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].adv_usd == Decimal("10000000")

    def test_too_few_bars_leaves_the_adv_unmeasured_rather_than_estimated(self) -> None:
        candidates = assemble_candidates(
            [listed_security(1)],
            weekday_bars(1, count=19),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(adv_lookback_days=20),
        )
        assert candidates[0].adv_usd is None
        assert candidates[0].price_usd == Decimal("10")

    def test_a_name_with_no_bars_has_neither_price_nor_adv(self) -> None:
        candidates = assemble_candidates(
            [listed_security(1)],
            [],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].price_usd is None
        assert candidates[0].adv_usd is None

    def test_the_blocked_inputs_are_left_unmeasured(self) -> None:
        # Not zero, not a guess, not a price-derived stand-in. The build refuses
        # upstream so these Nones are never screened on.
        candidates = assemble_candidates(
            [listed_security(1)],
            weekday_bars(1),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].market_cap_usd is None
        assert candidates[0].borrow_available is None

    def test_the_exchange_comes_from_the_version_in_force(self) -> None:
        candidates = assemble_candidates(
            [listed_security(1, exchange="XLON")],
            weekday_bars(1),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].exchange == "XLON"

    def test_bars_belonging_to_other_securities_are_ignored(self) -> None:
        candidates = assemble_candidates(
            [listed_security(1)],
            weekday_bars(2),
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
        )
        assert candidates[0].price_usd is None

    def test_candidates_come_back_ascending_by_security_id(self) -> None:
        listings = [listed_security(n) for n in (7, 2, 5)]
        candidates = assemble_candidates(
            listings, [], rebalance_date=FIXTURE_REBALANCE_DATE, criteria=fixture_criteria()
        )
        assert [candidate.security_id for candidate in candidates] == [2, 5, 7]

    def test_two_bars_for_one_day_are_an_unversioned_read_not_a_data_quirk(self) -> None:
        duplicated = DailyBar(
            security_id=1,
            trade_date=FIXTURE_REBALANCE_DATE,
            close_raw_usd=Decimal("10"),
            volume_shares=1,
        )
        bars = [*weekday_bars(1), duplicated]
        with pytest.raises(UniverseConsistencyError, match="unversioned read"):
            assemble_candidates(
                [listed_security(1)],
                bars,
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )

    def test_a_datetime_rebalance_date_is_refused_rather_than_truncated(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="no time component"):
            assemble_candidates(
                [],
                [],
                rebalance_date=dt.datetime(2020, 3, 31, 16, 0, tzinfo=dt.UTC),
                criteria=fixture_criteria(),
            )


# --- Screening -------------------------------------------------------------


class TestScreenCandidates:
    def test_every_candidate_gets_an_outcome_members_and_exclusions_alike(self) -> None:
        candidates = [
            passing_candidate(1),
            passing_candidate(2, price_usd="1"),
            passing_candidate(3, exchange="XLON"),
        ]
        snapshot = screen_candidates(
            candidates,
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
            as_of=FIXTURE_AS_OF,
        )
        assert snapshot.candidate_count == 3
        assert snapshot.members == (1,)
        assert snapshot.excluded_count == 2

    def test_the_snapshot_carries_the_criteria_hash_and_the_as_of(self) -> None:
        criteria = fixture_criteria()
        snapshot = screen_candidates(
            [passing_candidate(1)],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=criteria,
            as_of=FIXTURE_AS_OF,
        )
        assert snapshot.criteria_hash == criteria.criteria_hash()
        assert snapshot.as_of == FIXTURE_AS_OF

    def test_outcomes_come_back_ascending_by_security_id(self) -> None:
        snapshot = screen_candidates(
            [passing_candidate(n) for n in (9, 3, 6)],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
            as_of=FIXTURE_AS_OF,
        )
        assert [outcome.security_id for outcome in snapshot.outcomes] == [3, 6, 9]

    def test_a_candidate_screened_twice_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="more than once"):
            screen_candidates(
                [passing_candidate(1), passing_candidate(1)],
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
                as_of=FIXTURE_AS_OF,
            )

    def test_an_empty_candidate_set_gives_an_empty_universe_not_an_error(self) -> None:
        # A rebalance date before any security was listed is a legitimate,
        # empty answer — distinguishable from a refusal, which is the point.
        snapshot = screen_candidates(
            [],
            rebalance_date=FIXTURE_REBALANCE_DATE,
            criteria=fixture_criteria(),
            as_of=FIXTURE_AS_OF,
        )
        assert snapshot.candidate_count == 0
        assert snapshot.members == ()


# --- Orchestration ---------------------------------------------------------


class TestBuildUniverseOrchestration:
    async def test_both_source_tables_are_read_through_the_supplied_session(self) -> None:
        fake = FakeAsOfSession(listings=[listed_security(1)], bars=weekday_bars(1))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert len(fake.executed) == 2
        assert any("security_master" in sql for sql in fake.executed)
        assert any("price_bar" in sql for sql in fake.executed)

    async def test_a_rebalance_date_after_the_as_of_is_refused(self) -> None:
        # Nothing about that date was knowable at the as-of instant, so the only
        # universe it could produce is an empty one wearing a date it was never
        # screened at.
        fake = FakeAsOfSession(as_of=dt.datetime(2020, 1, 2, tzinfo=dt.UTC))
        with pytest.raises(UniverseConsistencyError, match="after the session's as-of"):
            await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert fake.executed == []

    async def test_a_datetime_rebalance_date_is_refused_at_the_entry_point(self) -> None:
        fake = FakeAsOfSession()
        with pytest.raises(UniverseConsistencyError, match="no time component"):
            await build_universe(
                _as_session(fake),
                rebalance_date=dt.datetime(2020, 3, 31, tzinfo=dt.UTC),
                criteria=fixture_criteria(),
            )
        assert fake.executed == []

    async def test_two_identity_versions_in_force_at_one_instant_are_refused(self) -> None:
        # Versions of one logical key must have disjoint [valid_from, valid_to)
        # intervals; choosing between two would make the universe depend on the
        # order the database happened to return rows in.
        rows = [
            (1, "XNYS", None, None),
            (1, "XNAS", None, None),
        ]
        session = cast("AsyncSession", RawRowSession(rows=rows))
        with pytest.raises(UniverseConsistencyError, match="more than one identity version"):
            await read_listings(session, rebalance_date=FIXTURE_REBALANCE_DATE)

    async def test_a_bar_not_starting_at_midnight_utc_is_refused(self) -> None:
        # D-011 encodes trading day D as [D 00:00Z, D+1 00:00Z). Truncating an
        # intraday valid_from to a date would fold two bars onto one day and
        # double that day's weight in the median.
        rows = [
            (
                1,
                dt.datetime(2020, 3, 31, 16, 0, tzinfo=dt.UTC),
                Decimal("10"),
                1_000_000,
            )
        ]
        session = cast("AsyncSession", RawRowSession(rows=rows))
        with pytest.raises(UniverseConsistencyError, match="not midnight UTC"):
            await read_price_window(
                session, first_date=dt.date(2020, 3, 1), last_date=FIXTURE_REBALANCE_DATE
            )

    async def test_the_snapshot_records_the_sessions_as_of_instant(self) -> None:
        instant = dt.datetime(2025, 7, 1, 12, 30, tzinfo=dt.UTC)
        fake = FakeAsOfSession(as_of=instant, listings=[listed_security(1)], bars=weekday_bars(1))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            snapshot = await build_universe(
                _as_session(fake),
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
            )
        assert snapshot.as_of == instant
