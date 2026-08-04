"""Historical universe reconstruction, size series, and turnover series (P4.2).

Phase 4 asks for the universe *"reconstructed historically — never using
present-day membership"*, and gate G4 asks for its size and turnover plotted
over history and inspected for discontinuities. This module builds the sequence
and derives the two series; :mod:`backend.universe.waterfall` derives the
filter-impact view of the same snapshots.

A history is a sequence of independent point-in-time builds, one per rebalance
date, **all under one set of criteria**. The criteria constraint is not
bookkeeping: a size series that mixes a 300-million market-cap floor with a
1-billion one is not a series at all, and its "discontinuity" would be a change
of definition wearing the appearance of a change in the market. So
:class:`UniverseHistory` refuses snapshots whose ``criteria_hash`` disagree.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Rebalance dates are exchange-calendar dates**, strictly ascending and unique
within a history. Turnover is defined between *consecutive* rebalance dates in
the history, so a gap in the sequence is a longer interval rather than a missing
observation — a history of quarter-ends and one of month-ends are both valid and
their turnover numbers are not comparable to each other. That is a property of
the schedule, not of this code, and it is why the interval's two dates are
carried on every turnover point instead of only the later one.

**Turnover is a FRACTION in ``[0, 1]``, never a percent.** Defined as::

    turnover = (entered + exited) / (previous_member_count + member_count)

the symmetric difference of the two membership sets over the sum of their sizes.
Zero means the membership did not change; one means it changed completely. When
the universe holds a constant number of names this equals the conventional
one-way turnover — 10 names in and 10 out of a 100-name universe gives
``20 / 200 = 0.10``, "10% turned over" — which is why this normalisation is the
one chosen over dividing by a single side's count. Dividing by the later count
alone would report turnover above 1 whenever the universe shrinks, which is
arithmetically fine and reads as a bug.

**Both universes empty gives 0.0.** The symmetric difference is empty, so
nothing turned over. That is a statement about the two sets rather than an
estimate standing in for a number nobody has, which is why it is a defined value
here and not a ``None``.

**Counts are counts of securities**, and ``entered``/``exited`` are the actual
identity-anchor keys rather than only their sizes: §6.3 asks for constituents
with entry and exit dates, and a count cannot answer "which name left".

--------------------------------------------------------------------------
Sessions
--------------------------------------------------------------------------

:func:`build_history` takes the ``as_of()``-scoped session it is handed and
passes it to every build, so an entire history is reconstructed at **one**
knowledge instant. That is what makes the series internally comparable: two
snapshots read at different as-ofs can differ because a vendor backfilled data
between them, and a turnover spike caused by a backfill is indistinguishable
from one caused by the market. It never opens a session of its own (I1).

Nothing here commits. Persisting a run is :func:`persist_history`, and the
transaction boundary belongs to the caller, who normally wants the whole run to
land together or not at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.universe.builder import build_universe
from backend.universe.errors import UniverseConsistencyError
from backend.universe.snapshot import UniverseSnapshot, load_snapshots, persist_snapshot

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.universe.criteria import UniverseCriteria

__all__ = [
    "MembershipSpan",
    "UniverseHistory",
    "UniverseSizePoint",
    "UniverseTurnoverPoint",
    "build_history",
    "load_history",
    "persist_history",
]


@dataclass(frozen=True, slots=True)
class UniverseSizePoint:
    """One rebalance date's universe size, beside the number of names screened.

    Both numbers are reported because the interesting discontinuities are the
    ones where they move differently. Members falling while candidates hold
    steady is a screen biting harder — a market-wide drawdown pushing names
    through the price floor, say. Both falling together is usually the data:
    a coverage gap, or an ingestion run that did not finish.

    Attributes:
        rebalance_date: the date, an exchange-calendar date.
        candidate_count: securities considered, before any screen (count).
        member_count: securities passing every applied screen (count).
    """

    rebalance_date: dt.date
    candidate_count: int
    member_count: int

    @property
    def excluded_count(self) -> int:
        """Candidates removed by at least one screen (count)."""
        return self.candidate_count - self.member_count

    @property
    def inclusion_rate(self) -> float | None:
        """Members as a fraction of candidates, in ``[0, 1]`` (dimensionless).

        Returns:
            ``member_count / candidate_count``, or ``None`` when nothing was
            screened. ``None`` rather than 0.0: a rebalance date with no
            candidates has an *undefined* inclusion rate, and plotting it as
            zero would draw a cliff where there is only an absence of data.
        """
        if self.candidate_count == 0:
            return None
        return self.member_count / self.candidate_count


@dataclass(frozen=True, slots=True)
class UniverseTurnoverPoint:
    """Membership change across one interval between consecutive rebalance dates.

    Attributes:
        previous_date: the earlier rebalance date of the interval.
        rebalance_date: the later rebalance date of the interval.
        previous_member_count: universe size at ``previous_date`` (count).
        member_count: universe size at ``rebalance_date`` (count).
        entered: identity-anchor keys that are members now and were not before,
            ascending.
        exited: identity-anchor keys that were members before and are not now,
            ascending.
    """

    previous_date: dt.date
    rebalance_date: dt.date
    previous_member_count: int
    member_count: int
    entered: tuple[int, ...]
    exited: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the interval's direction and the arithmetic between its counts.

        Raises:
            UniverseConsistencyError: if the interval does not run forwards, if
                either count is negative, or if the entries and exits cannot be
                reconciled with the two counts. The last check is the one that
                matters: ``member_count - previous_member_count`` must equal
                ``len(entered) - len(exited)``, and a violation means the two
                snapshots this point was derived from disagree with the sets
                derived from them.
        """
        if self.previous_date >= self.rebalance_date:
            msg = (
                f"turnover interval {self.previous_date.isoformat()} → "
                f"{self.rebalance_date.isoformat()} does not run forwards; a history's "
                f"rebalance dates are strictly ascending"
            )
            raise UniverseConsistencyError(msg)
        if self.previous_member_count < 0 or self.member_count < 0:
            msg = (
                f"member counts must be non-negative; got previous="
                f"{self.previous_member_count}, current={self.member_count}"
            )
            raise UniverseConsistencyError(msg)
        net = self.member_count - self.previous_member_count
        if net != len(self.entered) - len(self.exited):
            msg = (
                f"turnover at {self.rebalance_date.isoformat()} does not reconcile: the "
                f"universe moved from {self.previous_member_count} to {self.member_count} "
                f"names (net {net:+d}) while {len(self.entered)} entered and "
                f"{len(self.exited)} left (net "
                f"{len(self.entered) - len(self.exited):+d})"
            )
            raise UniverseConsistencyError(msg)

    @property
    def entered_count(self) -> int:
        """Names that joined the universe over this interval (count)."""
        return len(self.entered)

    @property
    def exited_count(self) -> int:
        """Names that left the universe over this interval (count)."""
        return len(self.exited)

    @property
    def retained_count(self) -> int:
        """Names that were members at both ends of the interval (count)."""
        return self.member_count - self.entered_count

    @property
    def turnover_fraction(self) -> float:
        """Membership turnover over the interval, a **fraction in ``[0, 1]``**.

        ``(entered + exited) / (previous_member_count + member_count)`` — see the
        module docstring for the definition and for why this normalisation was
        chosen over dividing by one side's count.

        Returns:
            The fraction. ``0.0`` when both universes are empty, because the
            symmetric difference is empty; never a percent.
        """
        denominator = self.previous_member_count + self.member_count
        if denominator == 0:
            return 0.0
        return (self.entered_count + self.exited_count) / denominator


@dataclass(frozen=True, slots=True)
class MembershipSpan:
    """One contiguous run of rebalance dates over which a security was a member.

    §6.3's "constituents with entry and exit dates". A name that leaves and
    later returns produces **two** spans rather than one long one with a hole in
    it, because the hole is the interesting part: a name that dropped below the
    price floor for a quarter and came back is a different fact from one that
    was in the universe throughout, and merging them would erase it.

    Contiguity is defined over the history's own rebalance dates, not over the
    calendar — consecutive means "adjacent in this history's schedule".

    Attributes:
        security_id: identity-anchor key (dimensionless).
        entered_on: first rebalance date of the run.
        last_seen_on: last rebalance date of the run.
        is_open: whether the run reaches the history's final rebalance date, so
            the name is a current constituent as far as this history goes. An
            open span has no exit date, which is different from having one that
            happens to be the last date.
    """

    security_id: int
    entered_on: dt.date
    last_seen_on: dt.date
    is_open: bool

    def __post_init__(self) -> None:
        """Validate that the run does not run backwards.

        Raises:
            UniverseConsistencyError: if ``last_seen_on`` precedes
                ``entered_on``.
        """
        if self.last_seen_on < self.entered_on:
            msg = (
                f"security_id={self.security_id} has a membership span ending "
                f"{self.last_seen_on.isoformat()} before it began "
                f"{self.entered_on.isoformat()}"
            )
            raise UniverseConsistencyError(msg)


@dataclass(frozen=True, slots=True)
class UniverseHistory:
    """A run of point-in-time universes under one set of criteria (P4.2).

    Attributes:
        criteria_hash: the criteria digest every snapshot in the run shares.
        snapshots: the builds, ascending by rebalance date, one per date.
    """

    criteria_hash: str
    snapshots: tuple[UniverseSnapshot, ...]

    def __post_init__(self) -> None:
        """Validate the ordering, the uniqueness of the dates, and the shared criteria.

        Raises:
            UniverseConsistencyError: if the snapshots are not in strictly
                ascending rebalance-date order, or if any snapshot was built
                under different criteria than :attr:`criteria_hash`. Mixing
                criteria would make the size and turnover series describe two
                different measurements plotted on one axis.
        """
        dates = [snapshot.rebalance_date for snapshot in self.snapshots]
        if dates != sorted(set(dates)):
            msg = (
                f"a history's rebalance dates must be strictly ascending and unique; got "
                f"{len(dates)} date(s) covering {len(set(dates))} distinct date(s)"
            )
            raise UniverseConsistencyError(msg)
        mismatched = [
            snapshot.rebalance_date
            for snapshot in self.snapshots
            if snapshot.criteria_hash != self.criteria_hash
        ]
        if mismatched:
            msg = (
                f"{len(mismatched)} snapshot(s) in this history were built under different "
                f"criteria than {self.criteria_hash!r} (first: "
                f"{mismatched[0].isoformat()}). A size or turnover series across changing "
                f"criteria plots a change of definition as though it were a change in the "
                f"market"
            )
            raise UniverseConsistencyError(msg)

    @classmethod
    def from_snapshots(cls, snapshots: Iterable[UniverseSnapshot]) -> UniverseHistory:
        """Build a history from snapshots, taking the criteria hash from them.

        Args:
            snapshots: the builds, in any order. Sorted by rebalance date here.

        Returns:
            The history.

        Raises:
            UniverseConsistencyError: if the sequence is empty (a history with
                no criteria has no identity to check the members against), or if
                the snapshots are mutually inconsistent (see
                :meth:`__post_init__`).
        """
        ordered = tuple(sorted(snapshots, key=lambda snapshot: snapshot.rebalance_date))
        if not ordered:
            msg = (
                "cannot build a universe history from zero snapshots: the criteria hash a "
                "history is defined by would have to be invented"
            )
            raise UniverseConsistencyError(msg)
        return cls(criteria_hash=ordered[0].criteria_hash, snapshots=ordered)

    @property
    def criteria(self) -> UniverseCriteria:
        """The screens every snapshot in this history was built under."""
        return self.snapshots[0].criteria

    @property
    def rebalance_dates(self) -> tuple[dt.date, ...]:
        """The history's schedule, ascending."""
        return tuple(snapshot.rebalance_date for snapshot in self.snapshots)

    def size_series(self) -> tuple[UniverseSizePoint, ...]:
        """Return universe size per rebalance date, ascending (G4's size history).

        Returns:
            One :class:`UniverseSizePoint` per snapshot.
        """
        return tuple(
            UniverseSizePoint(
                rebalance_date=snapshot.rebalance_date,
                candidate_count=snapshot.candidate_count,
                member_count=snapshot.member_count,
            )
            for snapshot in self.snapshots
        )

    def turnover_series(self) -> tuple[UniverseTurnoverPoint, ...]:
        """Return membership turnover per interval, ascending (G4's turnover history).

        There is one point fewer than there are snapshots: turnover is a
        property of an interval, and the first rebalance date has no predecessor.
        A history of one snapshot therefore has an **empty** turnover series
        rather than a zero — no interval was observed, which is not the same as
        an interval over which nothing changed.

        Returns:
            One :class:`UniverseTurnoverPoint` per consecutive pair of
            snapshots.
        """
        points: list[UniverseTurnoverPoint] = []
        for previous, current in zip(self.snapshots, self.snapshots[1:], strict=False):
            before = previous.member_set
            after = current.member_set
            points.append(
                UniverseTurnoverPoint(
                    previous_date=previous.rebalance_date,
                    rebalance_date=current.rebalance_date,
                    previous_member_count=previous.member_count,
                    member_count=current.member_count,
                    entered=tuple(sorted(after - before)),
                    exited=tuple(sorted(before - after)),
                )
            )
        return tuple(points)

    def membership_spans(self) -> tuple[MembershipSpan, ...]:
        """Return each security's contiguous runs of membership, ascending.

        Ordered by ``(security_id, entered_on)``. A security that left and
        returned appears more than once — see :class:`MembershipSpan` on why the
        gap is preserved rather than smoothed over.

        Returns:
            One :class:`MembershipSpan` per contiguous run.
        """
        last_date = self.snapshots[-1].rebalance_date if self.snapshots else None
        open_runs: dict[int, tuple[dt.date, dt.date]] = {}
        spans: list[MembershipSpan] = []

        def close(security_id: int) -> None:
            entered_on, last_seen_on = open_runs.pop(security_id)
            spans.append(
                MembershipSpan(
                    security_id=security_id,
                    entered_on=entered_on,
                    last_seen_on=last_seen_on,
                    is_open=last_seen_on == last_date,
                )
            )

        for snapshot in self.snapshots:
            members = snapshot.member_set
            for security_id in sorted(set(open_runs) - members):
                close(security_id)
            for security_id in sorted(members):
                run = open_runs.get(security_id)
                entered_on = run[0] if run is not None else snapshot.rebalance_date
                open_runs[security_id] = (entered_on, snapshot.rebalance_date)
        for security_id in sorted(open_runs):
            close(security_id)
        return tuple(sorted(spans, key=lambda span: (span.security_id, span.entered_on)))

    def report(self) -> str:
        """Render the size and turnover series as text for human inspection.

        Gate G4 requires the size and turnover history to be *inspected for
        discontinuities*. The dashboard page that plots it is blocked (B6 —
        UI work is routed away from this agent), so this renders the same
        numbers in a form a human can read without it. It is the programmatic
        half of that gate and is **not** a substitute for the plot: a table of
        120 rows does not show a shape.

        Returns:
            A multi-line table: one row per rebalance date with the candidate
            count, member count, and — from the second date on — the entries,
            exits, and turnover fraction of the interval that ended there.
        """
        turnover_by_date = {point.rebalance_date: point for point in self.turnover_series()}
        header = (
            f"Universe history — {len(self.snapshots)} rebalance date(s), "
            f"criteria {self.criteria_hash[:12]}…"
        )
        lines = [
            header,
            "  date          candidates   members   entered   exited   turnover",
        ]
        for point in self.size_series():
            turnover = turnover_by_date.get(point.rebalance_date)
            if turnover is None:
                tail = "        -        -          -"
            else:
                tail = (
                    f"{turnover.entered_count:>9}{turnover.exited_count:>9}"
                    f"{turnover.turnover_fraction:>11.4f}"
                )
            lines.append(
                f"  {point.rebalance_date.isoformat()}{point.candidate_count:>13}"
                f"{point.member_count:>10}{tail}"
            )
        return "\n".join(lines)


async def build_history(
    session: AsyncSession,
    *,
    rebalance_dates: Sequence[dt.date],
    criteria: UniverseCriteria,
) -> UniverseHistory:
    """Reconstruct the universe at every rebalance date, at one knowledge instant (P4.2).

    Each date is built independently by
    :func:`~backend.universe.builder.build_universe` through the **same**
    ``as_of()``-scoped session, so no snapshot in the run can see data that
    arrived after the run's as-of. Present-day membership plays no part at any
    date: what a name is today is not an input to any screen.

    Args:
        session: an ``as_of()``-scoped ``AsyncSession``. Never opened here (I1).
        rebalance_dates: the schedule, which must be strictly ascending and
            unique. Order is the caller's statement of what the history *is*, so
            an unordered or duplicated schedule is refused rather than sorted
            silently.
        criteria: the screens, identical at every date — see the module
            docstring on why a history may not mix criteria.

    Returns:
        The :class:`UniverseHistory`.

    Raises:
        UniverseConsistencyError: if ``rebalance_dates`` is empty, or is not
            strictly ascending and unique.
        UniverseSessionError: if the session is not ``as_of()``-scoped
            (propagated from the builder).
        UniverseInputUnavailableError: if a requested screen has no data source.
            Raised on the **first** date and therefore before any snapshot is
            produced, so a partial history is not a reachable state. Today this
            is every history; see :mod:`backend.universe.builder`.
    """
    dates = list(rebalance_dates)
    if not dates:
        msg = "cannot reconstruct a universe history over zero rebalance dates"
        raise UniverseConsistencyError(msg)
    if dates != sorted(set(dates)):
        msg = (
            f"rebalance_dates must be strictly ascending and unique; got {len(dates)} "
            f"date(s) covering {len(set(dates))} distinct date(s). The schedule is the "
            f"history's definition, so it is refused rather than silently sorted"
        )
        raise UniverseConsistencyError(msg)
    snapshots = [
        await build_universe(session, rebalance_date=date, criteria=criteria) for date in dates
    ]
    return UniverseHistory(criteria_hash=criteria.criteria_hash(), snapshots=tuple(snapshots))


async def persist_history(session: AsyncSession, history: UniverseHistory) -> tuple[int, ...]:
    """Append every snapshot in a history, without committing.

    Args:
        session: any writable ``AsyncSession``; neither universe table is
            bitemporal, so no as-of scoping is required. The caller is normally
            still holding the ``as_of()`` session that built the run.
        history: the reconstruction to persist.

    Returns:
        The database-generated ``snapshot_id`` of each snapshot, in the
        history's rebalance-date order.

    Raises:
        sqlalchemy.exc.IntegrityError: if any snapshot's
            ``(rebalance_date, criteria_hash, as_of)`` is already stored. The
            caller owns the transaction, so a mid-run failure rolls the whole
            run back rather than leaving half a history behind — which is the
            reason this does not commit per snapshot.
    """
    return tuple([await persist_snapshot(session, snapshot) for snapshot in history.snapshots])


async def load_history(
    session: AsyncSession,
    *,
    criteria: UniverseCriteria,
    first_rebalance_date: dt.date | None = None,
    last_rebalance_date: dt.date | None = None,
    knowledge_cutoff: dt.datetime | None = None,
) -> UniverseHistory | None:
    """Load a persisted history for one set of criteria, latest build per date.

    Thin wrapper over :func:`~backend.universe.snapshot.load_snapshots` that
    applies this module's invariants to the result. ``knowledge_cutoff`` makes
    this the point-in-time browser §6.3 asks for: a past cutoff returns the
    history as it was *reconstructible then*, not as it is reconstructible now.

    Args:
        session: any readable ``AsyncSession``; neither universe table is
            bitemporal.
        criteria: the screens whose history to load. Snapshots under other
            criteria are a different measurement and are never mixed in.
        first_rebalance_date: earliest date to include, inclusive; ``None`` for
            no lower bound.
        last_rebalance_date: latest date to include, inclusive; ``None`` for no
            upper bound.
        knowledge_cutoff: ignore builds whose ``as_of`` is later than this
            timezone-aware UTC instant; ``None`` takes the latest build of each
            date.

    Returns:
        The history, or ``None`` when no snapshot matches. ``None`` rather than
        an empty history: a history is defined by the criteria its snapshots
        agree on, and there is nothing to agree on here.

    Raises:
        UniverseConsistencyError: if ``knowledge_cutoff`` is not a
            timezone-aware UTC instant, or if the stored rows are inconsistent
            (propagated).
    """
    snapshots = await load_snapshots(
        session,
        criteria_hash=criteria.criteria_hash(),
        first_rebalance_date=first_rebalance_date,
        last_rebalance_date=last_rebalance_date,
        knowledge_cutoff=knowledge_cutoff,
    )
    if not snapshots:
        return None
    return UniverseHistory.from_snapshots(snapshots)
