"""Filter-impact waterfall: how many names each screen removed (P4.2, §6.3).

The dashboard's Universe page asks for *"a filter-impact waterfall showing how
many names each screen removes"*. This module derives it from the stored
screening outcomes, which is the only way it can honestly be derived: the number
cannot be recovered from a membership list — a universe of 480 names says
nothing about whether the market-cap floor removed 40 names or 4,000 — and it
cannot be recomputed later either, because recomputing needs the store as it
stood at the original ``as_of`` and the answer would move as data arrives. So
:class:`~backend.universe.snapshot.UniverseSnapshot` keeps one outcome per
candidate considered, and this module counts them.

--------------------------------------------------------------------------
Attribution, and why the order is frozen
--------------------------------------------------------------------------

A name that fails three screens is counted **once**, against the first screen it
failed in :data:`~backend.universe.criteria.FILTER_ORDER`. Any other rule breaks
the arithmetic a waterfall is: counting it against every screen it failed makes
the removals sum to more than the names removed, and the chart stops adding up
to its own total.

The consequence is that the *shape* of the waterfall is a function of the order,
not only of the data. Move ``price`` before ``exchange`` and a name that fails
both changes column. That is why :data:`~backend.universe.criteria.FILTER_ORDER`
is part of the criteria hash: reordering makes new waterfalls incomparable with
published ones, and the hash is what says so out loud instead of leaving two
different measurements looking like the same one.

Attribution is not the whole story, though, and the operator's real question —
*would loosening this screen bring names back* — is not answerable from it. A
name attributed to ``exchange`` may also have failed ``price``, so loosening the
price floor alone would not recover it. :attr:`WaterfallStep.also_failed` carries
that second number: names that failed this screen but were attributed to an
earlier one. The two together bound the effect of loosening a screen from both
sides, and neither alone does.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

Every field is a **count of securities**. There are no percentages here at all:
a waterfall is an accounting identity, and the identity is what the tests check.

Only the screens the criteria actually applied appear
(:meth:`~backend.universe.criteria.UniverseCriteria.applied_filters`). A borrow
screen that was switched off is **absent**, not present with zero removals —
"not applied" and "applied and removed nobody" are different facts about the
build, and a zero bar in a chart states the second one.

The identity every waterfall satisfies, and the reason this module exists as
something other than a dictionary of counts::

    sum(step.removed for step in steps) + member_count == candidate_count

It is checked on construction, so a waterfall that does not reconcile raises
instead of being rendered.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.universe.criteria import FILTER_ORDER
from backend.universe.errors import UniverseConsistencyError

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterable

    from backend.universe.snapshot import UniverseSnapshot

__all__ = [
    "FilterWaterfall",
    "WaterfallStep",
    "filter_waterfall",
    "waterfall_series",
]


@dataclass(frozen=True, slots=True)
class WaterfallStep:
    """One screen's contribution to the waterfall.

    Attributes:
        filter_name: the screen, one of
            :data:`~backend.universe.criteria.FILTER_ORDER`.
        considered: names still standing when this screen is reached (count) —
            the candidate count for the first step, the previous step's
            :attr:`remaining` after that.
        removed: names **attributed** to this screen: they failed it, and failed
            no earlier one (count).
        also_failed: names that failed this screen but were attributed to an
            earlier one (count). Zero on the first step by construction. Not
            part of the waterfall's arithmetic — see the module docstring on why
            it is carried anyway.
    """

    filter_name: str
    considered: int
    removed: int
    also_failed: int

    def __post_init__(self) -> None:
        """Validate the step's own counts.

        Raises:
            UniverseConsistencyError: if the filter name is not a declared
                screen, if any count is negative, or if more names were removed
                than were considered.
        """
        if self.filter_name not in FILTER_ORDER:
            msg = (
                f"waterfall step names unknown screen {self.filter_name!r}; expected one of "
                f"{list(FILTER_ORDER)}"
            )
            raise UniverseConsistencyError(msg)
        if min(self.considered, self.removed, self.also_failed) < 0:
            msg = (
                f"waterfall step {self.filter_name!r} has a negative count "
                f"(considered={self.considered}, removed={self.removed}, "
                f"also_failed={self.also_failed})"
            )
            raise UniverseConsistencyError(msg)
        if self.removed > self.considered:
            msg = (
                f"waterfall step {self.filter_name!r} removed {self.removed} of "
                f"{self.considered} names considered; a screen cannot remove a name that "
                f"an earlier screen already removed"
            )
            raise UniverseConsistencyError(msg)

    @property
    def remaining(self) -> int:
        """Names still standing after this screen (count): ``considered - removed``."""
        return self.considered - self.removed

    @property
    def failed_in_total(self) -> int:
        """Names failing this screen regardless of attribution (count).

        Returns:
            ``removed + also_failed`` — an upper bound on how many names
            loosening this screen could bring back, against which
            :attr:`removed` is the lower bound.
        """
        return self.removed + self.also_failed


@dataclass(frozen=True, slots=True)
class FilterWaterfall:
    """The filter-impact waterfall for one universe snapshot (§6.3).

    Attributes:
        rebalance_date: the date the universe was built for.
        criteria_hash: the criteria the snapshot was built under. Carried
            because the waterfall's shape depends on ``FILTER_ORDER``, which is
            part of this digest — two waterfalls under different hashes are not
            comparable and this is what says so.
        candidate_count: names considered, before any screen (count).
        member_count: names passing every applied screen (count).
        steps: one :class:`WaterfallStep` per **applied** screen, in
            :data:`~backend.universe.criteria.FILTER_ORDER`.
    """

    rebalance_date: dt.date
    criteria_hash: str
    candidate_count: int
    member_count: int
    steps: tuple[WaterfallStep, ...]

    def __post_init__(self) -> None:
        """Validate the chain of steps and the reconciliation identity.

        Raises:
            UniverseConsistencyError: if the steps are not a subsequence of
                :data:`~backend.universe.criteria.FILTER_ORDER`, if a step's
                ``considered`` does not equal the previous step's ``remaining``
                (the first step's must equal ``candidate_count``), if the last
                step's ``remaining`` does not equal ``member_count``, or if the
                removals plus the members do not sum back to the candidates.
                Each of these is the waterfall failing to be a waterfall.
        """
        names = [step.filter_name for step in self.steps]
        positions = [FILTER_ORDER.index(name) for name in names]
        if positions != sorted(set(positions)):
            msg = (
                f"waterfall steps {names} are not in FILTER_ORDER without repeats; the "
                f"chart's columns and the attribution rule would disagree"
            )
            raise UniverseConsistencyError(msg)
        standing = self.candidate_count
        for step in self.steps:
            if step.considered != standing:
                msg = (
                    f"waterfall step {step.filter_name!r} says it considered "
                    f"{step.considered} names, but {standing} were still standing when it "
                    f"was reached; the chain of steps is broken"
                )
                raise UniverseConsistencyError(msg)
            standing = step.remaining
        if standing != self.member_count:
            msg = (
                f"waterfall for {self.rebalance_date.isoformat()} leaves {standing} names "
                f"standing after the last screen, but the snapshot has {self.member_count} "
                f"members"
            )
            raise UniverseConsistencyError(msg)
        if self.total_removed + self.member_count != self.candidate_count:
            msg = (
                f"waterfall for {self.rebalance_date.isoformat()} does not reconcile: "
                f"{self.total_removed} removed + {self.member_count} members != "
                f"{self.candidate_count} candidates considered"
            )
            raise UniverseConsistencyError(msg)

    @property
    def total_removed(self) -> int:
        """Names removed by some screen (count) — the sum over the steps."""
        return sum(step.removed for step in self.steps)

    @property
    def applied_filters(self) -> tuple[str, ...]:
        """The screens the waterfall covers, in :data:`FILTER_ORDER`."""
        return tuple(step.filter_name for step in self.steps)

    def step_for(self, filter_name: str) -> WaterfallStep | None:
        """Return one screen's step, or ``None`` if that screen was not applied.

        Args:
            filter_name: the screen to look up.

        Returns:
            The step, or ``None``. ``None`` means the screen was not applied at
            all — which is not the same as having removed nobody, and callers
            rendering a chart must not draw a zero bar for it.
        """
        for step in self.steps:
            if step.filter_name == filter_name:
                return step
        return None

    def report(self) -> str:
        """Render the waterfall as text, with the reconciliation spelled out.

        Returns:
            A multi-line summary: one row per applied screen with the names it
            considered, removed, and left standing, plus the count that also
            failed it under a different attribution; then the identity that the
            removals and the members sum back to the candidates.
        """
        header = (
            f"Universe filter waterfall — {self.rebalance_date.isoformat()} "
            f"(criteria {self.criteria_hash[:12]}…)"
        )
        lines = [
            header,
            f"  candidates considered : {self.candidate_count}",
            "  screen        considered   removed   remaining   also failed",
        ]
        for step in self.steps:
            lines.append(
                f"  {step.filter_name:<14}{step.considered:>10}{step.removed:>10}"
                f"{step.remaining:>12}{step.also_failed:>14}"
            )
        lines.append(f"  members               : {self.member_count}")
        lines.append(
            f"  reconciliation        : {self.total_removed} removed + "
            f"{self.member_count} members = {self.candidate_count} candidates"
        )
        return "\n".join(lines)


def filter_waterfall(snapshot: UniverseSnapshot) -> FilterWaterfall:
    """Derive the filter-impact waterfall from a snapshot's stored outcomes.

    Every exclusion is attributed to
    :attr:`~backend.universe.criteria.FilterOutcome.attributed_filter` — the
    earliest screen in :data:`~backend.universe.criteria.FILTER_ORDER` the name
    failed — so the removals sum to the number of names removed exactly once.

    Args:
        snapshot: the built or loaded universe. Its ``outcomes`` must cover every
            candidate considered, which
            :class:`~backend.universe.snapshot.UniverseSnapshot` already
            guarantees.

    Returns:
        The :class:`FilterWaterfall`, with one step per applied screen.

    Raises:
        UniverseConsistencyError: if an outcome is attributed to a screen the
            criteria did not apply — the snapshot and its criteria disagree
            about what was run, and the waterfall built from it would count a
            removal in a column that should not exist — or if the resulting
            steps do not reconcile.
    """
    applied = snapshot.criteria.applied_filters()
    attributed: Counter[str] = Counter()
    failed_anywhere: Counter[str] = Counter()
    for outcome in snapshot.outcomes:
        attribution = outcome.attributed_filter
        if attribution is not None:
            attributed[attribution] += 1
        for name in outcome.failed_filters:
            failed_anywhere[name] += 1
    unexpected = sorted(set(failed_anywhere) - set(applied))
    if unexpected:
        msg = (
            f"snapshot for {snapshot.rebalance_date.isoformat()} records failures against "
            f"screen(s) {unexpected} that its criteria did not apply "
            f"(applied: {list(applied)}). The snapshot and its criteria disagree about what "
            f"was run"
        )
        raise UniverseConsistencyError(msg)
    steps: list[WaterfallStep] = []
    standing = snapshot.candidate_count
    for name in applied:
        removed = attributed[name]
        steps.append(
            WaterfallStep(
                filter_name=name,
                considered=standing,
                removed=removed,
                also_failed=failed_anywhere[name] - removed,
            )
        )
        standing -= removed
    return FilterWaterfall(
        rebalance_date=snapshot.rebalance_date,
        criteria_hash=snapshot.criteria_hash,
        candidate_count=snapshot.candidate_count,
        member_count=snapshot.member_count,
        steps=tuple(steps),
    )


def waterfall_series(snapshots: Iterable[UniverseSnapshot]) -> tuple[FilterWaterfall, ...]:
    """Derive one waterfall per snapshot, in rebalance-date order.

    Deliberately **not** an aggregate. Summing removals across dates would
    produce a single chart whose bars are dominated by whichever dates had the
    most candidates, and the question the waterfall answers — how hard is this
    screen biting — is a per-date question whose answer moves with the market.
    Callers wanting a trend plot one series per screen across these.

    Args:
        snapshots: the builds, in any order. Pass
            :attr:`~backend.universe.history.UniverseHistory.snapshots` for a
            reconstruction. This module deliberately does not import the history
            module: a waterfall is a property of a single snapshot, and taking
            the dependency the other way would let a change to the history's
            invariants alter what a waterfall means.

    Returns:
        One :class:`FilterWaterfall` per snapshot, ascending by rebalance date.

    Raises:
        UniverseConsistencyError: propagated from :func:`filter_waterfall`.
    """
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.rebalance_date)
    return tuple(filter_waterfall(snapshot) for snapshot in ordered)
