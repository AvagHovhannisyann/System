"""The snapshot value object: its invariants and its round trip through stored rows.

A snapshot is the record a waterfall, a turnover series and a point-in-time
browser are all derived from, so most of what is tested here is the set of
statements it refuses to make: members that disagree with the screening record, a
criteria hash that does not identify its own criteria, header counts that
disagree with the rows beneath them. Each of those would make some downstream
number quietly wrong rather than loudly absent.

Persistence itself — the append-only triggers, the unique constraint on a
snapshot's identity — needs a real database and lives in
``backend/tests/integration/test_universe_db.py``. What is tested here is the
pure translation in both directions, against ORM row objects constructed in
memory and never attached to a session.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.db import models
from backend.tests.universe.fixtures import (
    FIXTURE_AS_OF,
    FIXTURE_REBALANCE_DATE,
    fixture_criteria,
)
from backend.universe.criteria import FilterOutcome
from backend.universe.errors import UniverseConsistencyError, UniverseCriteriaError
from backend.universe.snapshot import (
    UniverseSnapshot,
    snapshot_from_rows,
    snapshots_by_date,
)


def _outcome(security_id: int, *failed: str) -> FilterOutcome:
    return FilterOutcome(security_id=security_id, included=not failed, failed_filters=tuple(failed))


def _snapshot(
    *outcomes: FilterOutcome,
    rebalance_date: dt.date = FIXTURE_REBALANCE_DATE,
    as_of: dt.datetime = FIXTURE_AS_OF,
) -> UniverseSnapshot:
    criteria = fixture_criteria()
    return UniverseSnapshot(
        rebalance_date=rebalance_date,
        criteria=criteria,
        criteria_hash=criteria.criteria_hash(),
        as_of=as_of,
        members=tuple(outcome.security_id for outcome in outcomes if outcome.included),
        outcomes=outcomes,
    )


class TestSnapshotInvariants:
    def test_counts_agree_with_the_outcomes(self) -> None:
        snapshot = _snapshot(_outcome(1), _outcome(2, "price"), _outcome(3))
        assert snapshot.candidate_count == 3
        assert snapshot.member_count == 2
        assert snapshot.excluded_count == 1
        assert snapshot.member_set == frozenset({1, 3})

    def test_members_must_be_exactly_the_included_outcomes(self) -> None:
        criteria = fixture_criteria()
        with pytest.raises(UniverseConsistencyError, match="ascending list of included"):
            UniverseSnapshot(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=criteria,
                criteria_hash=criteria.criteria_hash(),
                as_of=FIXTURE_AS_OF,
                members=(1, 2),  # 2 was excluded
                outcomes=(_outcome(1), _outcome(2, "price")),
            )

    def test_outcomes_must_be_ascending_without_duplicates(self) -> None:
        criteria = fixture_criteria()
        with pytest.raises(UniverseConsistencyError, match="ascending security_id"):
            UniverseSnapshot(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=criteria,
                criteria_hash=criteria.criteria_hash(),
                as_of=FIXTURE_AS_OF,
                members=(2, 1),
                outcomes=(_outcome(2), _outcome(1)),
            )

    def test_a_hash_that_does_not_identify_its_criteria_is_refused(self) -> None:
        with pytest.raises(UniverseConsistencyError, match="does not match the criteria"):
            UniverseSnapshot(
                rebalance_date=FIXTURE_REBALANCE_DATE,
                criteria=fixture_criteria(),
                criteria_hash="0" * 64,
                as_of=FIXTURE_AS_OF,
                members=(),
                outcomes=(),
            )

    @pytest.mark.parametrize(
        "as_of",
        [
            dt.datetime(2026, 1, 2),  # noqa: DTZ001 — naive, the defect under test
            dt.datetime(2026, 1, 2, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        ],
        ids=["naive", "offset"],
    )
    def test_a_naive_or_offset_as_of_is_refused(self, as_of: dt.datetime) -> None:
        with pytest.raises(UniverseConsistencyError, match="timezone-aware UTC"):
            _snapshot(_outcome(1), as_of=as_of)

    def test_a_name_that_was_never_a_candidate_is_distinguishable_from_an_excluded_one(
        self,
    ) -> None:
        # "Not listed at this date" and "listed and screened out" are different
        # answers to "why is this name missing", and conflating them would send
        # an operator hunting for a screen that never ran.
        snapshot = _snapshot(_outcome(1), _outcome(2, "adv"))
        assert snapshot.outcome_for(99) is None
        excluded = snapshot.outcome_for(2)
        assert excluded is not None
        assert excluded.failed_filters == ("adv",)


class TestRowRoundTrip:
    def _rows(
        self, snapshot: UniverseSnapshot, snapshot_id: int = 7
    ) -> tuple[models.UniverseSnapshot, list[models.UniverseMember]]:
        header = models.UniverseSnapshot(
            snapshot_id=snapshot_id,
            rebalance_date=snapshot.rebalance_date,
            criteria_hash=snapshot.criteria_hash,
            criteria=snapshot.criteria.as_config(),
            as_of=snapshot.as_of,
            candidate_count=snapshot.candidate_count,
            member_count=snapshot.member_count,
        )
        members = [
            models.UniverseMember(
                snapshot_id=snapshot_id,
                security_id=outcome.security_id,
                included=outcome.included,
                failed_filters=list(outcome.failed_filters),
            )
            for outcome in snapshot.outcomes
        ]
        return header, members

    def test_a_snapshot_survives_the_round_trip_through_rows(self) -> None:
        original = _snapshot(_outcome(1), _outcome(2, "exchange", "price"), _outcome(3))
        header, members = self._rows(original)
        assert snapshot_from_rows(header, members) == original

    def test_member_rows_may_arrive_in_any_order(self) -> None:
        original = _snapshot(_outcome(1), _outcome(2, "adv"), _outcome(5))
        header, members = self._rows(original)
        assert snapshot_from_rows(header, list(reversed(members))) == original

    def test_a_member_row_from_another_snapshot_is_refused(self) -> None:
        original = _snapshot(_outcome(1))
        header, members = self._rows(original)
        members.append(
            models.UniverseMember(snapshot_id=999, security_id=2, included=True, failed_filters=[])
        )
        with pytest.raises(UniverseConsistencyError, match="different snapshot"):
            snapshot_from_rows(header, members)

    def test_header_counts_that_disagree_with_the_rows_are_refused(self) -> None:
        original = _snapshot(_outcome(1), _outcome(2, "price"))
        header, members = self._rows(original)
        header.member_count = 2
        with pytest.raises(UniverseConsistencyError, match="header counts"):
            snapshot_from_rows(header, members)

    def test_a_stored_hash_that_does_not_match_its_stored_criteria_is_refused(self) -> None:
        original = _snapshot(_outcome(1))
        header, members = self._rows(original)
        header.criteria_hash = "f" * 64
        with pytest.raises(UniverseConsistencyError, match="does not match the criteria"):
            snapshot_from_rows(header, members)

    def test_stored_criteria_that_no_longer_describe_a_filter_are_refused(self) -> None:
        original = _snapshot(_outcome(1))
        header, members = self._rows(original)
        stored = dict(header.criteria)
        stored["min_adv_usd"] = "0"
        header.criteria = stored
        with pytest.raises(UniverseCriteriaError, match="strictly positive"):
            snapshot_from_rows(header, members)


class TestSnapshotsByDate:
    def test_snapshots_are_indexed_by_rebalance_date(self) -> None:
        first = _snapshot(_outcome(1), rebalance_date=dt.date(2020, 1, 31))
        second = _snapshot(_outcome(2), rebalance_date=dt.date(2020, 2, 28))
        indexed = snapshots_by_date([first, second])
        assert indexed == {dt.date(2020, 1, 31): first, dt.date(2020, 2, 28): second}

    def test_two_snapshots_for_one_date_are_refused(self) -> None:
        # They differ by as_of; silently keeping one would make every derived
        # series depend on argument order.
        first = _snapshot(_outcome(1))
        second = _snapshot(_outcome(2), as_of=FIXTURE_AS_OF + dt.timedelta(days=1))
        with pytest.raises(UniverseConsistencyError, match="two snapshots supplied"):
            snapshots_by_date([first, second])
