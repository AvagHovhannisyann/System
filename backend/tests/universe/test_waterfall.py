"""The filter-impact waterfall: attribution, reconciliation, and what it refuses.

The waterfall is an accounting identity before it is a chart::

    sum(removed per screen) + members == candidates considered

so most of this file is that identity, checked on hand-countable outcome sets
where the expected column for every name is derivable by reading
:data:`~backend.universe.criteria.FILTER_ORDER`.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.tests.universe.fixtures import (
    FIXTURE_AS_OF,
    FIXTURE_REBALANCE_DATE,
    fixture_criteria,
)
from backend.universe.criteria import FILTER_ORDER, FilterOutcome, UniverseCriteria
from backend.universe.errors import UniverseConsistencyError
from backend.universe.snapshot import UniverseSnapshot
from backend.universe.waterfall import (
    FilterWaterfall,
    WaterfallStep,
    filter_waterfall,
    waterfall_series,
)


def _outcome(security_id: int, *failed: str) -> FilterOutcome:
    return FilterOutcome(security_id=security_id, included=not failed, failed_filters=tuple(failed))


def _snapshot(
    *outcomes: FilterOutcome,
    criteria: UniverseCriteria | None = None,
    rebalance_date: dt.date = FIXTURE_REBALANCE_DATE,
) -> UniverseSnapshot:
    screens = criteria if criteria is not None else fixture_criteria()
    return UniverseSnapshot(
        rebalance_date=rebalance_date,
        criteria=screens,
        criteria_hash=screens.criteria_hash(),
        as_of=FIXTURE_AS_OF,
        members=tuple(outcome.security_id for outcome in outcomes if outcome.included),
        outcomes=outcomes,
    )


class TestAttribution:
    def test_each_exclusion_is_counted_once_against_its_earliest_failure(self) -> None:
        # Five names: one member, and one attributed to each of the four applied
        # screens. The third fails price *and* adv and belongs to price.
        snapshot = _snapshot(
            _outcome(1),
            _outcome(2, "exchange", "adv"),
            _outcome(3, "price", "adv"),
            _outcome(4, "market_cap"),
            _outcome(5, "adv"),
        )
        waterfall = filter_waterfall(snapshot)
        removed = {step.filter_name: step.removed for step in waterfall.steps}
        assert removed == {"exchange": 1, "price": 1, "market_cap": 1, "adv": 1}

    def test_names_failing_a_screen_under_another_attribution_are_counted_separately(
        self,
    ) -> None:
        snapshot = _snapshot(
            _outcome(1),
            _outcome(2, "exchange", "adv"),
            _outcome(3, "price", "adv"),
            _outcome(4, "adv"),
        )
        adv = filter_waterfall(snapshot).step_for("adv")
        assert adv is not None
        assert adv.removed == 1  # attributed to adv alone
        assert adv.also_failed == 2  # attributed to exchange and to price
        assert adv.failed_in_total == 3  # the upper bound on loosening the floor

    def test_the_chain_of_considered_counts_runs_through_the_screens(self) -> None:
        snapshot = _snapshot(
            _outcome(1),
            _outcome(2, "exchange"),
            _outcome(3, "price"),
            _outcome(4, "market_cap"),
            _outcome(5, "adv"),
        )
        waterfall = filter_waterfall(snapshot)
        assert [step.considered for step in waterfall.steps] == [5, 4, 3, 2]
        assert [step.remaining for step in waterfall.steps] == [4, 3, 2, 1]
        assert waterfall.member_count == 1

    def test_only_applied_screens_appear(self) -> None:
        # "Not applied" and "applied and removed nobody" are different facts
        # about the build, and a zero bar states the second one.
        waterfall = filter_waterfall(_snapshot(_outcome(1)))
        assert waterfall.applied_filters == ("exchange", "price", "market_cap", "adv")
        assert waterfall.step_for("borrow") is None

    def test_the_borrow_screen_appears_when_it_was_applied(self) -> None:
        criteria = fixture_criteria(require_borrow=True)
        waterfall = filter_waterfall(_snapshot(_outcome(1, "borrow"), criteria=criteria))
        assert waterfall.applied_filters == FILTER_ORDER
        borrow = waterfall.step_for("borrow")
        assert borrow is not None
        assert borrow.removed == 1


class TestReconciliation:
    @pytest.mark.parametrize(
        "outcomes",
        [
            (),
            (_outcome(1),),
            (_outcome(1, "adv"),),
            (_outcome(1), _outcome(2, "price"), _outcome(3, "exchange", "price", "adv")),
            tuple(_outcome(n, "market_cap") for n in range(1, 40)),
        ],
        ids=["empty", "one-member", "one-exclusion", "mixed", "all-excluded"],
    )
    def test_removals_plus_members_equal_the_candidates_considered(
        self, outcomes: tuple[FilterOutcome, ...]
    ) -> None:
        snapshot = _snapshot(*outcomes)
        waterfall = filter_waterfall(snapshot)
        assert waterfall.total_removed + waterfall.member_count == waterfall.candidate_count
        assert waterfall.candidate_count == snapshot.candidate_count
        assert waterfall.member_count == snapshot.member_count

    def test_a_broken_chain_of_steps_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="chain of steps is broken"):
            FilterWaterfall(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria_hash="a" * 64,
                candidate_count=10,
                member_count=8,
                steps=(
                    WaterfallStep(filter_name="exchange", considered=10, removed=1, also_failed=0),
                    # considered should be 9
                    WaterfallStep(filter_name="price", considered=10, removed=1, also_failed=0),
                ),
            )

    def test_steps_that_do_not_sum_back_to_the_candidates_are_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match=r"does not reconcile|names standing"):
            FilterWaterfall(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria_hash="a" * 64,
                candidate_count=10,
                member_count=5,
                steps=(
                    WaterfallStep(filter_name="exchange", considered=10, removed=1, also_failed=0),
                ),
            )

    def test_steps_out_of_declared_order_are_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="not in FILTER_ORDER"):
            FilterWaterfall(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria_hash="a" * 64,
                candidate_count=2,
                member_count=0,
                steps=(
                    WaterfallStep(filter_name="price", considered=2, removed=1, also_failed=0),
                    WaterfallStep(filter_name="exchange", considered=1, removed=1, also_failed=0),
                ),
            )

    def test_a_screen_cannot_remove_more_names_than_it_considered(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="cannot remove a name"):
            WaterfallStep(filter_name="price", considered=3, removed=4, also_failed=0)

    def test_an_unknown_screen_name_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="unknown screen"):
            WaterfallStep(filter_name="liquidity", considered=1, removed=0, also_failed=0)

    def test_a_snapshot_recording_a_screen_its_criteria_did_not_apply_is_refused(self) -> None:
        # The snapshot and its criteria disagree about what was run, and the
        # chart would carry a column that should not exist.
        snapshot = _snapshot(_outcome(1, "borrow"), criteria=fixture_criteria(require_borrow=False))
        with pytest.raises(UniverseConsistencyError, match="did not apply"):
            filter_waterfall(snapshot)


class TestWaterfallSeries:
    def test_one_waterfall_per_snapshot_in_rebalance_date_order(self) -> None:
        snapshots = [
            _snapshot(_outcome(1), rebalance_date=dt.date(2020, 2, 28)),
            _snapshot(_outcome(1), _outcome(2, "adv"), rebalance_date=dt.date(2020, 1, 31)),
        ]
        series = waterfall_series(snapshots)
        assert [item.rebalance_date for item in series] == [
            dt.date(2020, 1, 31),
            dt.date(2020, 2, 28),
        ]
        assert [item.candidate_count for item in series] == [2, 1]

    def test_an_empty_sequence_gives_an_empty_series(self) -> None:
        assert waterfall_series([]) == ()


class TestReport:
    def test_the_report_states_the_reconciliation_it_checked(self) -> None:
        snapshot = _snapshot(_outcome(1), _outcome(2, "price"), _outcome(3, "adv"))
        report = filter_waterfall(snapshot).report()
        assert "3 candidates" in report
        assert "2 removed + 1 members = 3 candidates" in report
        for name in ("exchange", "price", "market_cap", "adv"):
            assert name in report
