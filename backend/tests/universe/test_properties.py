"""Property tests for the universe screen, the waterfall, and the turnover series.

The example-based tests elsewhere pin down behaviour on inputs a person chose.
These pin down the statements that must hold for *every* input, which is where
the accounting identities live:

- a member is exactly a candidate that failed nothing, and its failure list is
  always an ordered subsequence of the screens actually applied;
- loosening a screen never removes a name from the universe (monotonicity) —
  the property an operator implicitly assumes every time they lower a floor to
  see what comes back;
- the waterfall's removals plus its members always equal the candidates
  considered, whatever the outcomes;
- turnover is always a fraction in ``[0, 1]``, and it is zero exactly when the
  membership did not change;
- a name listed on a date is a candidate on that date **whatever its later
  delisting date is** — the survivorship property, generalised past the single
  fixture in ``test_builder.py``.

Amounts are generated as two-decimal-place ``Decimal`` values because that is
what the screens compare: the boundary is ``value >= floor`` and exact decimal
arithmetic is the point of the comparison.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.tests.universe.fixtures import FIXTURE_AS_OF, FIXTURE_REBALANCE_DATE
from backend.universe.builder import (
    ListedSecurity,
    assemble_candidates,
    median_dollar_volume_usd,
    screen_candidates,
)
from backend.universe.criteria import (
    FILTER_ORDER,
    FilterOutcome,
    UniverseCandidate,
    UniverseCriteria,
    evaluate_candidate,
)
from backend.universe.history import UniverseHistory
from backend.universe.snapshot import UniverseSnapshot
from backend.universe.waterfall import filter_waterfall

_EXCHANGES = ("XNYS", "XNAS", "XLON", "ARCX")

positive_amounts = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("1000000000000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
non_negative_amounts = st.decimals(
    min_value=Decimal("0"),
    max_value=Decimal("1000000000000"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)
optional_amounts = st.none() | non_negative_amounts


@st.composite
def criteria_strategy(draw: st.DrawFn) -> UniverseCriteria:
    """Draw any valid set of screening criteria."""
    return UniverseCriteria(
        min_adv_usd=draw(positive_amounts),
        min_price_usd=draw(positive_amounts),
        min_market_cap_usd=draw(positive_amounts),
        allowed_exchanges=draw(st.frozensets(st.sampled_from(_EXCHANGES), min_size=1)),
        require_borrow=draw(st.booleans()),
        adv_lookback_days=draw(st.integers(min_value=2, max_value=252)),
    )


@st.composite
def candidate_strategy(draw: st.DrawFn, *, security_id: int = 1) -> UniverseCandidate:
    """Draw any valid candidate, with every optional input possibly unknowable."""
    return UniverseCandidate(
        security_id=security_id,
        exchange=draw(st.sampled_from(_EXCHANGES)),
        price_usd=draw(optional_amounts),
        adv_usd=draw(optional_amounts),
        market_cap_usd=draw(optional_amounts),
        borrow_available=draw(st.none() | st.booleans()),
    )


@st.composite
def candidate_set_strategy(draw: st.DrawFn) -> list[UniverseCandidate]:
    """Draw a set of candidates with distinct identity-anchor keys."""
    identifiers = draw(
        st.lists(st.integers(min_value=1, max_value=500), min_size=0, max_size=40, unique=True)
    )
    return [draw(candidate_strategy(security_id=n)) for n in identifiers]


def _loosen(criteria: UniverseCriteria, *, exchange: str) -> UniverseCriteria:
    """Return criteria at least as permissive as ``criteria`` for ``exchange``."""
    return UniverseCriteria(
        min_adv_usd=criteria.min_adv_usd / 2,
        min_price_usd=criteria.min_price_usd / 2,
        min_market_cap_usd=criteria.min_market_cap_usd / 2,
        allowed_exchanges=criteria.allowed_exchanges | {exchange},
        require_borrow=False,
        adv_lookback_days=criteria.adv_lookback_days,
    )


# --- The screen ------------------------------------------------------------


@hypothesis_settings(max_examples=400, deadline=None)
@given(candidate=candidate_strategy(), criteria=criteria_strategy())
def test_a_member_is_exactly_a_candidate_that_failed_nothing(
    candidate: UniverseCandidate, criteria: UniverseCriteria
) -> None:
    outcome = evaluate_candidate(candidate, criteria)
    assert outcome.included == (outcome.failed_filters == ())
    assert outcome.security_id == candidate.security_id


@hypothesis_settings(max_examples=400, deadline=None)
@given(candidate=candidate_strategy(), criteria=criteria_strategy())
def test_failures_are_an_ordered_subsequence_of_the_applied_screens(
    candidate: UniverseCandidate, criteria: UniverseCriteria
) -> None:
    outcome = evaluate_candidate(candidate, criteria)
    applied = criteria.applied_filters()
    assert set(outcome.failed_filters) <= set(applied)
    positions = [FILTER_ORDER.index(name) for name in outcome.failed_filters]
    assert positions == sorted(set(positions))


@hypothesis_settings(max_examples=400, deadline=None)
@given(candidate=candidate_strategy(), criteria=criteria_strategy())
def test_the_attributed_screen_is_the_earliest_one_failed(
    candidate: UniverseCandidate, criteria: UniverseCriteria
) -> None:
    outcome = evaluate_candidate(candidate, criteria)
    if outcome.failed_filters:
        earliest = min(outcome.failed_filters, key=FILTER_ORDER.index)
        assert outcome.attributed_filter == earliest
    else:
        assert outcome.attributed_filter is None


@hypothesis_settings(max_examples=400, deadline=None)
@given(candidate=candidate_strategy(), criteria=criteria_strategy())
def test_loosening_every_screen_never_removes_a_member(
    candidate: UniverseCandidate, criteria: UniverseCriteria
) -> None:
    # Monotonicity. An operator who halves a floor expects names to come back,
    # never to disappear; a screen whose direction was inverted would show up
    # here and nowhere else.
    strict = evaluate_candidate(candidate, criteria)
    loose = evaluate_candidate(candidate, _loosen(criteria, exchange=candidate.exchange))
    assert not strict.included or loose.included


@hypothesis_settings(max_examples=300, deadline=None)
@given(candidate=candidate_strategy(), criteria=criteria_strategy())
def test_an_unknowable_input_can_only_ever_exclude(
    candidate: UniverseCandidate, criteria: UniverseCriteria
) -> None:
    # Replacing a known value with "not knowable" must never turn an exclusion
    # into an inclusion: excluding a name we cannot show to be eligible drops a
    # trade, including one invents a trade.
    blanked = UniverseCandidate(
        security_id=candidate.security_id,
        exchange=candidate.exchange,
        price_usd=None,
        adv_usd=None,
        market_cap_usd=None,
        borrow_available=None,
    )
    assert not evaluate_candidate(blanked, criteria).included


# --- The waterfall ---------------------------------------------------------


@hypothesis_settings(max_examples=300, deadline=None)
@given(candidates=candidate_set_strategy(), criteria=criteria_strategy())
def test_the_waterfall_always_reconciles(
    candidates: list[UniverseCandidate], criteria: UniverseCriteria
) -> None:
    snapshot = screen_candidates(
        candidates,
        rebalance_date=FIXTURE_REBALANCE_DATE,
        criteria=criteria,
        as_of=FIXTURE_AS_OF,
    )
    waterfall = filter_waterfall(snapshot)
    assert waterfall.total_removed + waterfall.member_count == waterfall.candidate_count
    assert waterfall.candidate_count == len(candidates)


@hypothesis_settings(max_examples=300, deadline=None)
@given(candidates=candidate_set_strategy(), criteria=criteria_strategy())
def test_every_screens_removals_are_bounded_by_the_names_that_failed_it(
    candidates: list[UniverseCandidate], criteria: UniverseCriteria
) -> None:
    snapshot = screen_candidates(
        candidates,
        rebalance_date=FIXTURE_REBALANCE_DATE,
        criteria=criteria,
        as_of=FIXTURE_AS_OF,
    )
    waterfall = filter_waterfall(snapshot)
    for step in waterfall.steps:
        failed_here = sum(
            1 for outcome in snapshot.outcomes if step.filter_name in outcome.failed_filters
        )
        assert step.removed <= failed_here == step.failed_in_total


# --- Turnover --------------------------------------------------------------


def _snapshot_for(rebalance_date: dt.date, members: set[int]) -> UniverseSnapshot:
    criteria = UniverseCriteria(
        min_adv_usd=Decimal("1"),
        min_price_usd=Decimal("1"),
        min_market_cap_usd=Decimal("1"),
        allowed_exchanges=frozenset({"XNYS"}),
        require_borrow=False,
    )
    outcomes = tuple(
        FilterOutcome(security_id=n, included=True, failed_filters=()) for n in sorted(members)
    )
    return UniverseSnapshot(
        rebalance_date=rebalance_date,
        criteria=criteria,
        criteria_hash=criteria.criteria_hash(),
        as_of=FIXTURE_AS_OF,
        members=tuple(sorted(members)),
        outcomes=outcomes,
    )


member_sets = st.sets(st.integers(min_value=1, max_value=60), max_size=30)


@hypothesis_settings(max_examples=300, deadline=None)
@given(before=member_sets, after=member_sets)
def test_turnover_is_a_fraction_and_is_zero_exactly_when_membership_is_unchanged(
    before: set[int], after: set[int]
) -> None:
    history = UniverseHistory.from_snapshots(
        [
            _snapshot_for(dt.date(2020, 1, 31), before),
            _snapshot_for(dt.date(2020, 2, 29), after),
        ]
    )
    point = history.turnover_series()[0]
    assert 0.0 <= point.turnover_fraction <= 1.0
    assert (point.turnover_fraction == 0.0) == (before == after)
    assert set(point.entered) == after - before
    assert set(point.exited) == before - after
    assert point.retained_count == len(before & after)


@hypothesis_settings(max_examples=200, deadline=None)
@given(memberships=st.lists(member_sets, min_size=1, max_size=8))
def test_membership_spans_reconstruct_the_membership_exactly(
    memberships: list[set[int]],
) -> None:
    # The spans are what §6.3 renders as entry and exit dates, so losing or
    # inventing a date is losing or inventing a constituency. The spans must
    # therefore cover exactly the dates each name was a member, and no date
    # twice — the second half is what forbids merging a name's two separate
    # runs into one span with a hole in it.
    dates = [
        dt.date(2020, 1, 31) + dt.timedelta(days=31 * index) for index in range(len(memberships))
    ]
    snapshots = [
        _snapshot_for(date, members) for date, members in zip(dates, memberships, strict=True)
    ]
    history = UniverseHistory.from_snapshots(snapshots)

    expected: dict[int, set[dt.date]] = {}
    for date, members in zip(dates, memberships, strict=True):
        for security_id in members:
            expected.setdefault(security_id, set()).add(date)

    covered: dict[int, set[dt.date]] = {}
    for span in history.membership_spans():
        within = {date for date in dates if span.entered_on <= date <= span.last_seen_on}
        already = covered.setdefault(span.security_id, set())
        assert already.isdisjoint(within), "two spans of one name overlap"
        already.update(within)
    assert covered == expected


# --- The median ------------------------------------------------------------


@hypothesis_settings(max_examples=300, deadline=None)
@given(volumes=st.lists(non_negative_amounts, min_size=1, max_size=60))
def test_the_median_lies_between_the_extremes_and_ignores_order(
    volumes: list[Decimal],
) -> None:
    median = median_dollar_volume_usd(volumes)
    assert min(volumes) <= median <= max(volumes)
    assert median_dollar_volume_usd(list(reversed(volumes))) == median
    assert median_dollar_volume_usd(sorted(volumes)) == median


# --- Survivorship ----------------------------------------------------------


@hypothesis_settings(max_examples=300, deadline=None)
@given(
    days_listed_before=st.integers(min_value=0, max_value=5000),
    days_delisted_after=st.integers(min_value=0, max_value=5000),
)
def test_a_name_listed_on_the_date_is_a_candidate_however_it_later_ends(
    days_listed_before: int, days_delisted_after: int
) -> None:
    # Generalises the single fixture in test_builder.py: no delisting date at or
    # after the rebalance date, however far in the future, may remove the name
    # from that date's candidate set.
    listing = ListedSecurity(
        security_id=1,
        exchange="XNYS",
        first_listed_on=FIXTURE_REBALANCE_DATE - dt.timedelta(days=days_listed_before),
        delisted_on=FIXTURE_REBALANCE_DATE + dt.timedelta(days=days_delisted_after),
    )
    assert listing.is_listed_on(FIXTURE_REBALANCE_DATE)
    candidates = assemble_candidates(
        [listing],
        [],
        rebalance_date=FIXTURE_REBALANCE_DATE,
        criteria=UniverseCriteria(
            min_adv_usd=Decimal("1"),
            min_price_usd=Decimal("1"),
            min_market_cap_usd=Decimal("1"),
            allowed_exchanges=frozenset({"XNYS"}),
            require_borrow=False,
        ),
    )
    assert [candidate.security_id for candidate in candidates] == [1]


@hypothesis_settings(max_examples=300, deadline=None)
@given(days_before=st.integers(min_value=1, max_value=5000))
def test_a_name_delisted_before_the_date_is_never_a_candidate(days_before: int) -> None:
    listing = ListedSecurity(
        security_id=1,
        exchange="XNYS",
        first_listed_on=None,
        delisted_on=FIXTURE_REBALANCE_DATE - dt.timedelta(days=days_before),
    )
    assert not listing.is_listed_on(FIXTURE_REBALANCE_DATE)
