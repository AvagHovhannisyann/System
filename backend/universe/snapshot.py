"""The universe snapshot value object and its append-only persistence (P4.1).

A snapshot is the complete record of one universe build: the rebalance date it
was built for, the criteria (and their hash), the knowledge instant the inputs
were read at, the members, and the screening outcome of **every candidate
considered** — including the ones that were screened out, and why.

Keeping the exclusions is what makes the record useful rather than merely
correct. §6.3 asks for a filter-impact waterfall showing how many names each
screen removes, and that number cannot be recovered from a membership list: a
universe of 480 names says nothing about whether the market-cap floor removed 40
names or 4,000. Rebuilding the answer later is not an option either, because
rebuilding requires the store as it stood at the original ``as_of`` and the
answer would change as data arrives. So the exclusions are persisted with the
members, in :class:`backend.db.models.UniverseMember`, one row per candidate.

--------------------------------------------------------------------------
Identity, reproducibility, and why there is no seed
--------------------------------------------------------------------------

A snapshot is identified by ``(rebalance_date, criteria_hash, as_of)``, unique in
the database. Those three determine it completely given the contents of the
bitemporal store, which is the reproducibility claim I2 asks for. ``as_of`` is
load-bearing: the same criteria at the same rebalance date, read a month later
after a vendor backfill, can legitimately give a different universe. That
difference is information about the data, and a schema that overwrote the first
answer would destroy it. Re-running therefore **appends** — enforced by migration
0011's ``BEFORE UPDATE OR DELETE`` trigger, not by convention.

The full :class:`~backend.tracking.stamp.ReproducibilityStamp` is deliberately
not stored. It requires a seed, and a universe build draws no random numbers;
that module refuses partial stamps precisely so nobody records ``seed=0`` and
makes an artifact look stamped. What does carry over is the canonicalisation:
``criteria_hash`` comes from
:func:`~backend.tracking.stamp.canonical_config_hash`, the same function that
hashes a training run's config, so two subsystems hashing the same mapping agree.

--------------------------------------------------------------------------
Sessions and transactions
--------------------------------------------------------------------------

Neither ``universe_snapshot`` nor ``universe_member`` is bitemporal (see
:class:`backend.db.models.UniverseSnapshot`), so reading them needs no as-of
rewrite. The functions here take whatever ``AsyncSession`` the caller is already
holding — typically the ``as_of()``-scoped one that built the snapshot — and
**never commit**: the transaction boundary belongs to the caller, who is usually
persisting a run of snapshots and wants them to land together or not at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sqlalchemy as sa

from backend.db.models import UniverseMember as UniverseMemberRow
from backend.db.models import UniverseSnapshot as UniverseSnapshotRow
from backend.universe.criteria import FilterOutcome, UniverseCriteria
from backend.universe.errors import UniverseConsistencyError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "UniverseSnapshot",
    "load_snapshots",
    "persist_snapshot",
    "snapshot_from_rows",
    "snapshots_by_date",
]


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """One point-in-time universe, with the outcome of every candidate considered.

    The in-memory counterpart of :class:`backend.db.models.UniverseSnapshot`
    (same name, different layer: that one is the row, this one is the result).

    Attributes:
        rebalance_date: the date the universe was built for. An exchange-calendar
            date with no time component.
        criteria: the screens applied, carried in full so a snapshot states its
            own definition rather than pointing at a digest nobody can invert.
        criteria_hash: SHA-256 hex digest of ``criteria`` (64 lowercase hex
            characters), validated against :attr:`criteria` on construction —
            they cannot drift apart.
        as_of: the knowledge instant the inputs were read at, timezone-aware UTC.
            The as-of bound on the session that built the snapshot (I1), and the
            snapshot's data version.
        members: the securities passing every applied screen, by identity-anchor
            key, **ascending and unique**. Sorted rather than left in query order
            so two builds of the same universe compare equal as tuples.
        outcomes: one :class:`~backend.universe.criteria.FilterOutcome` per
            candidate considered — every security listed at the rebalance date,
            members and exclusions alike — in ascending ``security_id``.
    """

    rebalance_date: dt.date
    criteria: UniverseCriteria
    criteria_hash: str
    as_of: dt.datetime
    members: tuple[int, ...]
    outcomes: tuple[FilterOutcome, ...]

    def __post_init__(self) -> None:
        """Validate the as-of instant, the criteria hash, and members-vs-outcomes.

        Raises:
            UniverseConsistencyError: if ``as_of`` is naive or not UTC, if
                ``criteria_hash`` does not match ``criteria``, if two outcomes
                name the same security, if ``outcomes`` is not in ascending
                ``security_id`` order, or if ``members`` is not exactly the
                ascending list of included securities. Each of these would make
                some downstream count — the waterfall's, the turnover series' —
                quietly wrong rather than loudly absent.
        """
        if self.as_of.tzinfo is None or self.as_of.utcoffset() != dt.timedelta(0):
            msg = (
                f"as_of={self.as_of!r} must be a timezone-aware UTC instant; it records the "
                f"knowledge time the universe was read at (I1) and a naive or offset value "
                f"would make two snapshots incomparable"
            )
            raise UniverseConsistencyError(msg)
        expected_hash = self.criteria.criteria_hash()
        if self.criteria_hash != expected_hash:
            msg = (
                f"criteria_hash={self.criteria_hash!r} does not match the criteria it is "
                f"supposed to identify (expected {expected_hash!r}). A snapshot whose hash "
                f"and criteria disagree cannot be grouped with anything honestly"
            )
            raise UniverseConsistencyError(msg)
        outcome_ids = [outcome.security_id for outcome in self.outcomes]
        if outcome_ids != sorted(set(outcome_ids)):
            msg = (
                "outcomes must be in ascending security_id order without duplicates; "
                f"got {len(outcome_ids)} outcome(s) covering {len(set(outcome_ids))} "
                "security/securities"
            )
            raise UniverseConsistencyError(msg)
        included = tuple(outcome.security_id for outcome in self.outcomes if outcome.included)
        if self.members != included:
            msg = (
                f"members ({len(self.members)}) is not the ascending list of included "
                f"outcomes ({len(included)}); membership and the screening record must be "
                f"the same statement"
            )
            raise UniverseConsistencyError(msg)

    @property
    def candidate_count(self) -> int:
        """Securities considered — listed at the rebalance date, before any screen (count)."""
        return len(self.outcomes)

    @property
    def member_count(self) -> int:
        """Securities passing every applied screen (count)."""
        return len(self.members)

    @property
    def excluded_count(self) -> int:
        """Candidates removed by at least one screen (count)."""
        return self.candidate_count - self.member_count

    @property
    def member_set(self) -> frozenset[int]:
        """Members as a set, for the entry/exit arithmetic in the turnover series."""
        return frozenset(self.members)

    def outcome_for(self, security_id: int) -> FilterOutcome | None:
        """Return one security's screening outcome, or ``None`` if it was not a candidate.

        ``None`` means the security was not listed at the rebalance date under
        this snapshot's ``as_of`` — it was never screened. That is a different
        statement from "screened and excluded", which is an outcome with a
        non-empty ``failed_filters``, and the two must not be conflated when the
        operator asks why a name is missing.

        Args:
            security_id: identity-anchor key to look up.

        Returns:
            The outcome, or ``None`` when the security was not a candidate.
        """
        for outcome in self.outcomes:
            if outcome.security_id == security_id:
                return outcome
        return None


async def persist_snapshot(session: AsyncSession, snapshot: UniverseSnapshot) -> int:
    """Append one snapshot and one row per candidate considered.

    Writes a :class:`backend.db.models.UniverseSnapshot` header plus a
    :class:`backend.db.models.UniverseMember` row for **every** outcome, members
    and exclusions alike — the exclusions are what make the §6.3 waterfall
    reconstructible from the record (module docstring).

    Does **not commit**: the caller owns the transaction, because a historical
    reconstruction persists a run of snapshots that should land together or not
    at all. Flushes so the database-generated ``snapshot_id`` is available to
    the member rows and to the caller.

    Args:
        session: any writable ``AsyncSession``. Neither table is bitemporal, so
            no as-of scoping is required here; the caller is normally still
            holding the ``as_of()`` session that built the snapshot.
        snapshot: the built universe.

    Returns:
        The database-generated ``snapshot_id``.

    Raises:
        sqlalchemy.exc.IntegrityError: if a snapshot with the same
            ``(rebalance_date, criteria_hash, as_of)`` already exists. That
            triple is the snapshot's identity, so a duplicate is a re-run of an
            identical build, and the append-only store has no update to offer it.
    """
    header = UniverseSnapshotRow(
        rebalance_date=snapshot.rebalance_date,
        criteria_hash=snapshot.criteria_hash,
        criteria=snapshot.criteria.as_config(),
        as_of=snapshot.as_of,
        candidate_count=snapshot.candidate_count,
        member_count=snapshot.member_count,
    )
    session.add(header)
    await session.flush()
    snapshot_id = header.snapshot_id
    session.add_all(
        [
            UniverseMemberRow(
                snapshot_id=snapshot_id,
                security_id=outcome.security_id,
                included=outcome.included,
                failed_filters=list(outcome.failed_filters),
            )
            for outcome in snapshot.outcomes
        ]
    )
    await session.flush()
    return snapshot_id


def snapshot_from_rows(
    header: UniverseSnapshotRow,
    members: Iterable[UniverseMemberRow],
) -> UniverseSnapshot:
    """Rebuild a snapshot value object from its persisted rows.

    The inverse of :func:`persist_snapshot`. The criteria are reconstructed from
    the stored JSON and re-hashed by :class:`UniverseSnapshot`'s own validation,
    so a stored ``criteria_hash`` that does not match its stored ``criteria``
    fails here rather than propagating into a comparison that would silently
    group two different universes together.

    Args:
        header: the ``universe_snapshot`` row.
        members: that snapshot's ``universe_member`` rows, in any order.

    Returns:
        The reconstructed :class:`UniverseSnapshot`.

    Raises:
        UniverseConsistencyError: if the rows are mutually inconsistent, or if a
            member row belongs to a different snapshot.
        UniverseCriteriaError: if the stored criteria JSON does not describe
            usable criteria (propagated from
            :meth:`~backend.universe.criteria.UniverseCriteria.from_config`).
    """
    rows = sorted(members, key=lambda row: row.security_id)
    foreign = [row.security_id for row in rows if row.snapshot_id != header.snapshot_id]
    if foreign:
        msg = (
            f"{len(foreign)} member row(s) belong to a different snapshot than "
            f"snapshot_id={header.snapshot_id}"
        )
        raise UniverseConsistencyError(msg)
    stored_criteria: Mapping[str, object] = header.criteria
    criteria = UniverseCriteria.from_config(stored_criteria)
    outcomes = tuple(
        FilterOutcome(
            security_id=row.security_id,
            included=row.included,
            failed_filters=tuple(row.failed_filters),
        )
        for row in rows
    )
    snapshot = UniverseSnapshot(
        rebalance_date=header.rebalance_date,
        criteria=criteria,
        criteria_hash=header.criteria_hash,
        as_of=header.as_of,
        members=tuple(outcome.security_id for outcome in outcomes if outcome.included),
        outcomes=outcomes,
    )
    if (snapshot.candidate_count, snapshot.member_count) != (
        header.candidate_count,
        header.member_count,
    ):
        msg = (
            f"snapshot_id={header.snapshot_id} header counts "
            f"(candidates={header.candidate_count}, members={header.member_count}) disagree "
            f"with its member rows (candidates={snapshot.candidate_count}, "
            f"members={snapshot.member_count})"
        )
        raise UniverseConsistencyError(msg)
    return snapshot


async def load_snapshots(
    session: AsyncSession,
    *,
    criteria_hash: str,
    first_rebalance_date: dt.date | None = None,
    last_rebalance_date: dt.date | None = None,
    knowledge_cutoff: dt.datetime | None = None,
) -> tuple[UniverseSnapshot, ...]:
    """Load persisted snapshots for one set of criteria, latest build per date.

    Several builds can exist for one ``(rebalance_date, criteria_hash)`` — one
    per ``as_of`` the universe was reconstructed at. This returns **one snapshot
    per rebalance date**: the build with the greatest ``as_of`` at or before
    ``knowledge_cutoff``. That is the same latest-knowledge-wins rule the
    bitemporal store applies to facts (D-011), applied here to a derived
    artifact, and it is what makes ``knowledge_cutoff`` a point-in-time browser:
    asking for a past cutoff returns the universe as it was reconstructible then,
    not as it is reconstructible now.

    The reduction happens in Python rather than in SQL. Snapshots are hundreds of
    rows for a decade of monthly rebalances, so a window function would buy
    nothing and cost readability.

    Args:
        session: any readable ``AsyncSession``; neither table is bitemporal, so
            no as-of scoping is required.
        criteria_hash: the criteria digest to load. Snapshots under different
            criteria are different measurements and are never mixed.
        first_rebalance_date: earliest rebalance date to include (inclusive);
            ``None`` for no lower bound.
        last_rebalance_date: latest rebalance date to include (inclusive);
            ``None`` for no upper bound.
        knowledge_cutoff: ignore builds whose ``as_of`` is later than this
            timezone-aware UTC instant; ``None`` takes the latest build of each
            date.

    Returns:
        Snapshots in ascending ``rebalance_date``, at most one per date.

    Raises:
        UniverseConsistencyError: if ``knowledge_cutoff`` is naive or not UTC, or
            if the loaded rows are internally inconsistent (propagated from
            :func:`snapshot_from_rows`).
    """
    if knowledge_cutoff is not None and (
        knowledge_cutoff.tzinfo is None or knowledge_cutoff.utcoffset() != dt.timedelta(0)
    ):
        msg = f"knowledge_cutoff={knowledge_cutoff!r} must be a timezone-aware UTC instant"
        raise UniverseConsistencyError(msg)
    conditions = [UniverseSnapshotRow.criteria_hash == criteria_hash]
    if first_rebalance_date is not None:
        conditions.append(UniverseSnapshotRow.rebalance_date >= first_rebalance_date)
    if last_rebalance_date is not None:
        conditions.append(UniverseSnapshotRow.rebalance_date <= last_rebalance_date)
    if knowledge_cutoff is not None:
        conditions.append(UniverseSnapshotRow.as_of <= knowledge_cutoff)
    headers = (
        (
            await session.execute(
                sa.select(UniverseSnapshotRow)
                .where(*conditions)
                .order_by(UniverseSnapshotRow.rebalance_date, UniverseSnapshotRow.as_of)
            )
        )
        .scalars()
        .all()
    )
    latest: dict[dt.date, UniverseSnapshotRow] = {}
    for header in headers:
        latest[header.rebalance_date] = header  # ordered by as_of, so the last write wins
    if not latest:
        return ()
    selected = [latest[date] for date in sorted(latest)]
    member_rows = (
        (
            await session.execute(
                sa.select(UniverseMemberRow).where(
                    UniverseMemberRow.snapshot_id.in_([header.snapshot_id for header in selected])
                )
            )
        )
        .scalars()
        .all()
    )
    by_snapshot: dict[int, list[UniverseMemberRow]] = {
        header.snapshot_id: [] for header in selected
    }
    for row in member_rows:
        by_snapshot[row.snapshot_id].append(row)
    return tuple(snapshot_from_rows(header, by_snapshot[header.snapshot_id]) for header in selected)


def snapshots_by_date(snapshots: Sequence[UniverseSnapshot]) -> dict[dt.date, UniverseSnapshot]:
    """Index snapshots by rebalance date, refusing duplicates.

    Args:
        snapshots: the snapshots to index.

    Returns:
        Mapping of rebalance date to snapshot.

    Raises:
        UniverseConsistencyError: if two snapshots share a rebalance date. Two
            universes for one date have no defined ordering, and silently keeping
            one would make the size and turnover series depend on argument order.
    """
    indexed: dict[dt.date, UniverseSnapshot] = {}
    for snapshot in snapshots:
        if snapshot.rebalance_date in indexed:
            msg = (
                f"two snapshots supplied for rebalance_date={snapshot.rebalance_date}; "
                f"pick the build you mean (they differ by as_of)"
            )
            raise UniverseConsistencyError(msg)
        indexed[snapshot.rebalance_date] = snapshot
    return indexed
