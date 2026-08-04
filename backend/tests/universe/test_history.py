"""Historical reconstruction: size series, turnover series, and membership spans.

Every fixture history below is small enough to count on fingers, so each expected
turnover fraction is arithmetic in the test's own comment. The definition being
checked is the one the module states::

    turnover = (entered + exited) / (previous_member_count + member_count)

a **fraction**, never a percent.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, cast

import pytest

from backend.tests.universe.fixtures import (
    FIXTURE_AS_OF,
    FakeAsOfSession,
    fixture_criteria,
    listed_security,
    weekday_bars,
)
from backend.universe import builder
from backend.universe.criteria import FilterOutcome
from backend.universe.errors import UniverseConsistencyError, UniverseInputUnavailableError
from backend.universe.history import (
    UniverseHistory,
    UniverseTurnoverPoint,
    build_history,
)
from backend.universe.snapshot import UniverseSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

JANUARY = dt.date(2020, 1, 31)
FEBRUARY = dt.date(2020, 2, 28)
MARCH = dt.date(2020, 3, 31)
APRIL = dt.date(2020, 4, 30)


def _snapshot_with(
    rebalance_date: dt.date,
    members: Iterable[int],
    *,
    excluded: Iterable[int] = (),
) -> UniverseSnapshot:
    """Build a snapshot whose universe is exactly ``members``.

    Excluded names are attributed to the ADV screen, arbitrarily — the history
    module never reads the attribution, and the waterfall tests cover it.
    """
    criteria = fixture_criteria()
    outcomes = tuple(
        sorted(
            [
                *(FilterOutcome(security_id=n, included=True, failed_filters=()) for n in members),
                *(
                    FilterOutcome(security_id=n, included=False, failed_filters=("adv",))
                    for n in excluded
                ),
            ],
            key=lambda outcome: outcome.security_id,
        )
    )
    return UniverseSnapshot(
        rebalance_date=rebalance_date,
        criteria=criteria,
        criteria_hash=criteria.criteria_hash(),
        as_of=FIXTURE_AS_OF,
        members=tuple(outcome.security_id for outcome in outcomes if outcome.included),
        outcomes=outcomes,
    )


class TestHistoryInvariants:
    def test_snapshots_are_sorted_into_ascending_order(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(MARCH, [1]), _snapshot_with(JANUARY, [1])]
        )
        assert history.rebalance_dates == (JANUARY, MARCH)

    def test_two_snapshots_for_one_date_are_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="strictly ascending and unique"):
            UniverseHistory.from_snapshots(
                [_snapshot_with(JANUARY, [1]), _snapshot_with(JANUARY, [2])]
            )

    def test_a_history_under_mixed_criteria_is_refused(self) -> None:
        # A size series that mixes a $300m floor with a $1bn one plots a change
        # of definition as though it were a change in the market.
        other = fixture_criteria(min_market_cap_usd="1000000000")
        january = _snapshot_with(JANUARY, [1])
        february = UniverseSnapshot(
            rebalance_date=FEBRUARY,
            criteria=other,
            criteria_hash=other.criteria_hash(),
            as_of=FIXTURE_AS_OF,
            members=(1,),
            outcomes=(FilterOutcome(security_id=1, included=True, failed_filters=()),),
        )
        with pytest.raises(UniverseConsistencyError, match="different criteria"):
            UniverseHistory.from_snapshots([january, february])

    def test_a_history_of_zero_snapshots_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="zero snapshots"):
            UniverseHistory.from_snapshots([])

    def test_the_history_reports_the_criteria_its_snapshots_share(self) -> None:
        history = UniverseHistory.from_snapshots([_snapshot_with(JANUARY, [1])])
        assert history.criteria == fixture_criteria()
        assert history.criteria_hash == fixture_criteria().criteria_hash()


class TestSizeSeries:
    def test_one_point_per_rebalance_date_with_both_counts(self) -> None:
        history = UniverseHistory.from_snapshots(
            [
                _snapshot_with(JANUARY, [1, 2], excluded=[3]),
                _snapshot_with(FEBRUARY, [1], excluded=[2, 3, 4]),
            ]
        )
        series = history.size_series()
        assert [(p.rebalance_date, p.candidate_count, p.member_count) for p in series] == [
            (JANUARY, 3, 2),
            (FEBRUARY, 4, 1),
        ]
        assert [p.excluded_count for p in series] == [1, 3]

    def test_the_inclusion_rate_is_a_fraction(self) -> None:
        history = UniverseHistory.from_snapshots([_snapshot_with(JANUARY, [1], excluded=[2, 3, 4])])
        assert history.size_series()[0].inclusion_rate == 0.25

    def test_a_date_with_no_candidates_has_no_inclusion_rate(self) -> None:
        # Undefined, not zero: plotting it as zero draws a cliff where there is
        # only an absence of data.
        history = UniverseHistory.from_snapshots([_snapshot_with(JANUARY, [])])
        assert history.size_series()[0].inclusion_rate is None


class TestTurnoverSeries:
    def test_there_is_one_fewer_point_than_there_are_dates(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(d, [1]) for d in (JANUARY, FEBRUARY, MARCH)]
        )
        assert len(history.turnover_series()) == 2

    def test_a_history_of_one_date_has_an_empty_turnover_series(self) -> None:
        # No interval was observed, which is not the same as an interval over
        # which nothing changed.
        history = UniverseHistory.from_snapshots([_snapshot_with(JANUARY, [1, 2])])
        assert history.turnover_series() == ()

    def test_an_unchanged_universe_has_zero_turnover(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, [1, 2, 3]), _snapshot_with(FEBRUARY, [1, 2, 3])]
        )
        point = history.turnover_series()[0]
        assert point.entered == ()
        assert point.exited == ()
        assert point.turnover_fraction == 0.0

    def test_a_completely_replaced_universe_has_turnover_one(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, [1, 2]), _snapshot_with(FEBRUARY, [3, 4])]
        )
        point = history.turnover_series()[0]
        assert point.entered == (3, 4)
        assert point.exited == (1, 2)
        # (2 + 2) / (2 + 2) = 1.0
        assert point.turnover_fraction == 1.0

    def test_a_partial_change_at_constant_size_matches_one_way_turnover(self) -> None:
        before = list(range(1, 11))  # ten names
        after = [*before[:9], 11]  # one out, one in
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, before), _snapshot_with(FEBRUARY, after)]
        )
        point = history.turnover_series()[0]
        assert point.entered == (11,)
        assert point.exited == (10,)
        assert point.retained_count == 9
        # (1 + 1) / (10 + 10) = 0.10 — "10% of the universe turned over".
        assert point.turnover_fraction == pytest.approx(0.10)

    def test_turnover_stays_within_zero_and_one_when_the_universe_shrinks(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, [1, 2, 3, 4]), _snapshot_with(FEBRUARY, [1])]
        )
        point = history.turnover_series()[0]
        # (0 + 3) / (4 + 1) = 0.6 — dividing by the later count alone would give
        # 3.0, which is arithmetically fine and reads as a bug.
        assert point.turnover_fraction == pytest.approx(0.6)
        assert 0.0 <= point.turnover_fraction <= 1.0

    def test_two_empty_universes_have_zero_turnover(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, []), _snapshot_with(FEBRUARY, [])]
        )
        assert history.turnover_series()[0].turnover_fraction == 0.0

    def test_the_interval_carries_both_of_its_dates(self) -> None:
        # A quarterly history and a monthly one are not comparable, and only the
        # interval's two ends say which this is.
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, [1]), _snapshot_with(APRIL, [1])]
        )
        point = history.turnover_series()[0]
        assert (point.previous_date, point.rebalance_date) == (JANUARY, APRIL)

    def test_a_turnover_point_that_does_not_reconcile_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="does not reconcile"):
            UniverseTurnoverPoint(
                previous_date=JANUARY,
                rebalance_date=FEBRUARY,
                previous_member_count=10,
                member_count=10,
                entered=(11,),
                exited=(),  # net +1 against a net 0 move in the counts
            )

    def test_a_turnover_interval_that_runs_backwards_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="does not run forwards"):
            UniverseTurnoverPoint(
                previous_date=FEBRUARY,
                rebalance_date=JANUARY,
                previous_member_count=0,
                member_count=0,
                entered=(),
                exited=(),
            )


class TestMembershipSpans:
    def test_a_name_present_throughout_has_one_open_span(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(d, [1]) for d in (JANUARY, FEBRUARY, MARCH)]
        )
        (span,) = history.membership_spans()
        assert (span.security_id, span.entered_on, span.last_seen_on) == (1, JANUARY, MARCH)
        assert span.is_open is True

    def test_a_name_that_left_has_a_closed_span(self) -> None:
        history = UniverseHistory.from_snapshots(
            [
                _snapshot_with(JANUARY, [1]),
                _snapshot_with(FEBRUARY, [1]),
                _snapshot_with(MARCH, [], excluded=[1]),
            ]
        )
        (span,) = history.membership_spans()
        assert (span.entered_on, span.last_seen_on, span.is_open) == (JANUARY, FEBRUARY, False)

    def test_a_name_that_left_and_returned_has_two_spans(self) -> None:
        # The gap is the interesting part: a name that dropped below the price
        # floor for a quarter is a different fact from one that never left.
        history = UniverseHistory.from_snapshots(
            [
                _snapshot_with(JANUARY, [1]),
                _snapshot_with(FEBRUARY, [], excluded=[1]),
                _snapshot_with(MARCH, [1]),
                _snapshot_with(APRIL, [1]),
            ]
        )
        first, second = history.membership_spans()
        assert (first.entered_on, first.last_seen_on, first.is_open) == (JANUARY, JANUARY, False)
        assert (second.entered_on, second.last_seen_on, second.is_open) == (MARCH, APRIL, True)

    def test_spans_are_ordered_by_security_then_entry(self) -> None:
        history = UniverseHistory.from_snapshots(
            [_snapshot_with(JANUARY, [2, 1]), _snapshot_with(FEBRUARY, [1, 2])]
        )
        assert [span.security_id for span in history.membership_spans()] == [1, 2]


class TestReport:
    def test_the_report_carries_every_date_and_its_turnover(self) -> None:
        history = UniverseHistory.from_snapshots(
            [
                _snapshot_with(JANUARY, [1, 2], excluded=[3]),
                _snapshot_with(FEBRUARY, [1, 4], excluded=[3]),
            ]
        )
        report = history.report()
        assert JANUARY.isoformat() in report
        assert FEBRUARY.isoformat() in report
        # (1 + 1) / (2 + 2) = 0.5
        assert "0.5000" in report


class TestBuildHistory:
    def _session(self) -> AsyncSession:
        listings = [listed_security(1), listed_security(2)]
        bars = [*weekday_bars(1, last_date=MARCH), *weekday_bars(2, last_date=MARCH)]
        return cast("AsyncSession", FakeAsOfSession(listings=listings, bars=bars))

    async def test_an_empty_schedule_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="zero rebalance dates"):
            await build_history(self._session(), rebalance_dates=[], criteria=fixture_criteria())

    @pytest.mark.parametrize(
        "dates",
        [[FEBRUARY, JANUARY], [JANUARY, JANUARY]],
        ids=["out-of-order", "duplicated"],
    )
    async def test_a_schedule_that_is_not_strictly_ascending_is_refused(
        self, dates: list[dt.date]
    ) -> None:
        # The schedule is the history's definition, so it is refused rather than
        # silently sorted or deduplicated.
        with pytest.raises(UniverseConsistencyError, match="strictly ascending and unique"):
            await build_history(self._session(), rebalance_dates=dates, criteria=fixture_criteria())

    async def test_the_run_refuses_on_the_first_date_so_no_partial_history_exists(self) -> None:
        session = self._session()
        with pytest.raises(UniverseInputUnavailableError) as caught:
            await build_history(
                session,
                rebalance_dates=[JANUARY, FEBRUARY, MARCH],
                criteria=fixture_criteria(),
            )
        assert caught.value.filter_name == "market_cap"

    async def test_every_date_is_built_through_the_one_session(self) -> None:
        # One session means one as-of, which is what makes the series
        # internally comparable: a turnover spike caused by a vendor backfill
        # between two reads is indistinguishable from one caused by the market.
        fake = FakeAsOfSession(
            listings=[listed_security(1), listed_security(2)],
            bars=[*weekday_bars(1, last_date=MARCH), *weekday_bars(2, last_date=MARCH)],
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "UNAVAILABLE_FILTER_INPUTS", {})
            history = await build_history(
                cast("AsyncSession", fake),
                rebalance_dates=[JANUARY, FEBRUARY, MARCH],
                criteria=fixture_criteria(),
            )
        assert history.rebalance_dates == (JANUARY, FEBRUARY, MARCH)
        assert {snapshot.as_of for snapshot in history.snapshots} == {FIXTURE_AS_OF}
        assert len(fake.executed) == 6  # two reads per rebalance date
