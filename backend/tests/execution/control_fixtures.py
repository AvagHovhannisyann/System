"""Builders and an in-memory double for the P11.3/P11.5 control tables.

Every snapshot built here carries :attr:`~backend.execution.reconciliation.SnapshotOrigin.SIMULATED`
(I3). A fixture statement is a *different value on the row* from a paper-account
statement, so nothing constructed in a test can ever be mistaken for something a
venue said — and ``test_control_paper_only.py`` asserts that property over this
module rather than trusting the convention.

What the double models, and what it deliberately cannot
--------------------------------------------------------

It models the two tables' behaviour that :mod:`backend.execution.halt` and
:mod:`backend.execution.reconciliation` actually depend on, and takes P11.2's
lesson (D-034) as its starting point rather than rediscovering it:

- **The clearance guard fires before the unique index**, because a ``BEFORE
  INSERT`` row trigger runs ahead of every constraint. The double refuses in that
  order, so the path that is *likelier under contention* is the path it models.
- **Every refusal carries a real SQLSTATE.** The halt store decides
  "already cleared" versus "not an engagement" on the five-character code, so a
  double whose errors carry none could not tell a correct store from one that
  translates every failure into the same thing. :class:`DriverError` supplies the
  attribute asyncpg exposes.
- **Both routes to "already cleared" are available** —
  :attr:`ControlRows.clearance_refusal_route` selects the trigger or the index —
  so the claim that they are one condition is tested rather than assumed.
- **Nothing refuses an engagement.** The double has no branch that can reject an
  ``engaged`` row, mirroring the trigger. A double that could would be modelling a
  schema in which the kill switch can be prevented from firing.

What it does not prove: that *Postgres* enforces any of this. The CHECK
constraints, the real trigger, the append-only guards and true cross-connection
concurrency live in ``backend/tests/integration/test_reconciliation.py``.

Storage is separated from the session on purpose. :class:`ControlRows` holds the
rows; :class:`ControlSessionDouble` is the handle over them. A **restart** is
modelled by discarding the session and building a new one over the same rows,
which is exactly what a restarted process does to a database — and it is how the
"a halt survives a restart" test avoids proving anything about Python object
lifetimes instead.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Self, cast

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.sql import Insert, Select

from backend.execution.halt import (
    HALT_CLEARANCE_REFUSED_SQLSTATE,
    UNIQUE_VIOLATION_SQLSTATE,
    HaltEventKind,
)
from backend.execution.killswitch import (
    CycleObservation,
    DataFreshnessObservation,
    DrawdownObservation,
    ManualHaltRequest,
)
from backend.execution.reconciliation import (
    PositionSnapshot,
    ReconciliationResult,
    SnapshotOrigin,
    reconcile,
)
from backend.tests.execution.fixtures import make_stamp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from backend.tracking.stamp import ReproducibilityStamp

CYCLE_ID: Final = "2026-08-03T15:00:00Z/rebalance"
"""A fixed cycle identifier, so a test never depends on a clock."""

OBSERVED_AT: Final = dt.datetime(2026, 8, 3, 15, 0, tzinfo=dt.UTC)
"""A fixed, timezone-aware observation instant."""


class DriverError(Exception):
    """Stand-in for the driver exception SQLAlchemy wraps in a ``DBAPIError``.

    Carries ``sqlstate``, which is the attribute asyncpg exposes and the one
    :func:`backend.execution.halt._sqlstate` reads. Without it every refusal this
    double produces would look identical to code that inspects the code — the
    defect that made P11.2's trigger/index confusion invisible (D-034).
    """

    def __init__(self, sqlstate: str, message: str) -> None:
        """Build the error from a SQLSTATE and the message the server would send."""
        super().__init__(message)
        self.sqlstate = sqlstate


def internal_snapshot(
    *,
    positions: Mapping[int, int] | None = None,
    cash_usd: Decimal = Decimal("100000.00"),
    observed_at: dt.datetime = OBSERVED_AT,
) -> PositionSnapshot:
    """Build our own side of a reconciliation."""
    return PositionSnapshot(
        origin=SnapshotOrigin.INTERNAL_LEDGER,
        observed_at=observed_at,
        cash_usd=cash_usd,
        positions={} if positions is None else positions,
    )


def reported_snapshot(
    *,
    positions: Mapping[int, int] | None = None,
    cash_usd: Decimal = Decimal("100000.00"),
    observed_at: dt.datetime = OBSERVED_AT,
) -> PositionSnapshot:
    """Build the statement side of a reconciliation.

    Always ``SIMULATED``, never ``PAPER_BROKER``: no adapter exists (P11.1 is
    blocked on B2), so a fixture claiming to be a paper-account statement would be
    exactly the fabrication I3 forbids.
    """
    return PositionSnapshot(
        origin=SnapshotOrigin.SIMULATED,
        observed_at=observed_at,
        cash_usd=cash_usd,
        positions={} if positions is None else positions,
    )


def clean_result(*, cycle_id: str = CYCLE_ID) -> ReconciliationResult:
    """A reconciliation of two agreeing snapshots."""
    return reconcile(
        cycle_id=cycle_id,
        internal=internal_snapshot(positions={7: 100}),
        reported=reported_snapshot(positions={7: 100}),
        stamp=make_stamp(),
    )


def broken_result(*, cycle_id: str = CYCLE_ID) -> ReconciliationResult:
    """A reconciliation whose statement holds a position we do not know about."""
    return reconcile(
        cycle_id=cycle_id,
        internal=internal_snapshot(positions={7: 100}),
        reported=reported_snapshot(positions={7: 100, 99: 250}),
        stamp=make_stamp(),
    )


class Unset:
    """Sentinel distinguishing "not overridden" from an explicit ``None``.

    Load-bearing rather than stylistic: ``None`` is itself one of the conditions
    under test — an unmeasured drawdown, an absent reconciliation — so a helper
    that read ``None`` as "use the default" would make every fail-closed test
    silently pass on the safe value instead.
    """


UNSET: Final = Unset()
"""The sentinel instance; see :class:`Unset`."""


def observation(
    *,
    cycle_id: str = CYCLE_ID,
    observed_at: dt.datetime = OBSERVED_AT,
    drawdown: Unset | DrawdownObservation | None = UNSET,
    freshness: Unset | DataFreshnessObservation | None = UNSET,
    reconciliation: Unset | ReconciliationResult | None = UNSET,
    manual: ManualHaltRequest | None = None,
    stamp: ReproducibilityStamp | None = None,
) -> CycleObservation:
    """Build a cycle observation that is safe unless a field is overridden.

    The defaults are the *only* combination that permits trading, which is what
    makes every override in a test a single, named deviation from safety.
    """
    return CycleObservation(
        cycle_id=cycle_id,
        observed_at=observed_at,
        drawdown=healthy_drawdown() if isinstance(drawdown, Unset) else drawdown,
        freshness=fresh_data() if isinstance(freshness, Unset) else freshness,
        reconciliation=(
            clean_result(cycle_id=cycle_id) if isinstance(reconciliation, Unset) else reconciliation
        ),
        manual=manual,
        stamp=make_stamp() if stamp is None else stamp,
    )


def healthy_drawdown(
    *,
    equity_usd: Decimal | None = Decimal("95000.00"),
    peak_equity_usd: Decimal | None = Decimal("100000.00"),
    limit_fraction: Decimal | None = Decimal("0.10"),
) -> DrawdownObservation:
    """A 5% drawdown against a 10% limit — comfortably inside."""
    return DrawdownObservation(
        equity_usd=equity_usd,
        peak_equity_usd=peak_equity_usd,
        limit_fraction=limit_fraction,
    )


def fresh_data(
    *,
    age_seconds: Decimal | None = Decimal("60"),
    max_age_seconds: Decimal | None = Decimal("3600"),
) -> DataFreshnessObservation:
    """One-minute-old data against a one-hour limit."""
    return DataFreshnessObservation(age_seconds=age_seconds, max_age_seconds=max_age_seconds)


class ExplodingDecimal(Decimal):
    """A decimal whose finiteness cannot be established.

    Stands in for a measurement whose *probe* fails rather than whose value is
    wrong — a valuation service that raises, a metric backend that times out. The
    kill switch must convert that into a halt, because a check that raised has
    ruled nothing out.
    """

    def is_finite(self) -> bool:
        """Raise, as the failing probe would."""
        msg = "the valuation probe failed while reading this measurement"
        raise RuntimeError(msg)


class ExplodingResult(ReconciliationResult):
    """A reconciliation verdict that raises when asked whether it matched."""

    @property
    def matched(self) -> bool:
        """Raise, as a corrupted verdict would."""
        msg = "the verdict could not be evaluated"
        raise RuntimeError(msg)


class _Result:
    """The slice of ``Result`` the control stores use: ``scalar_one``, ``first``, ``all``."""

    def __init__(self, rows: Sequence[tuple[object, ...]]) -> None:
        self._rows = list(rows)

    def scalar_one(self) -> object:
        """Return the single scalar of the single row."""
        if len(self._rows) != 1:
            msg = f"expected exactly one row, got {len(self._rows)}"
            raise AssertionError(msg)
        return self._rows[0][0]

    def first(self) -> tuple[object, ...] | None:
        """Return the first row, or ``None``."""
        return self._rows[0] if self._rows else None

    def all(self) -> list[tuple[object, ...]]:
        """Return every row."""
        return list(self._rows)


class _Savepoint:
    """Stand-in for ``AsyncSessionTransaction``; the double's writes are atomic."""

    async def __aenter__(self) -> Self:
        """Enter the savepoint."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Leave the savepoint, propagating any exception."""
        return False


class ControlRows:
    """The rows of ``execution_halt`` and ``execution_reconciliation``.

    Separate from the session so a **restart** can be modelled honestly: discard
    the session, build a new one over the same rows. That is what a restarted
    process does, and it is the only way the "a halt survives a restart" claim can
    be about persistence rather than about Python object lifetimes.

    Attributes:
        halts: halt rows in insertion order.
        reconciliations: reconciliation rows in insertion order.
        read_failure: when set, every ``SELECT`` raises it. Models the database
            being unreachable, which the halt store must treat as *halted*.
        clearance_refusal_route: which of the two refusal routes an already-cleared
            halt takes — ``"trigger"`` (the ``BEFORE INSERT`` guard, the likelier
            path under contention) or ``"index"`` (the unique constraint). Both
            carry ``23505`` and both must reach the caller as the same condition.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.halts: list[dict[str, object]] = []
        self.reconciliations: list[dict[str, object]] = []
        self.read_failure: Exception | None = None
        self.clearance_refusal_route: str = "trigger"
        self.halt_insert_attempts = 0
        self.insert_barrier: asyncio.Barrier | None = None
        """Set to an ``asyncio.Barrier`` of N parties to force N writers to the
        contention point simultaneously. Without it the scheduler may happen to
        run callers one after another, and a test that passes because nothing
        raced has proved nothing."""
        self._next_halt_id = 1
        self._next_reconciliation_id = 1

    def open_halt_ids(self) -> list[int]:
        """Return the ids of engagements with no clearance, ascending."""
        cleared = {
            int(cast("int", row["clears_halt_id"]))
            for row in self.halts
            if row["event"] == HaltEventKind.CLEARED.value and row["clears_halt_id"] is not None
        }
        return sorted(
            int(cast("int", row["halt_id"]))
            for row in self.halts
            if row["event"] == HaltEventKind.ENGAGED.value
            and int(cast("int", row["halt_id"])) not in cleared
        )

    def halt(self, halt_id: int) -> dict[str, object]:
        """Return one halt row by id."""
        for row in self.halts:
            if row["halt_id"] == halt_id:
                return row
        msg = f"no halt row with halt_id={halt_id}"
        raise AssertionError(msg)

    def next_halt_id(self) -> int:
        """Allocate the next surrogate halt key."""
        allocated = self._next_halt_id
        self._next_halt_id += 1
        return allocated

    def next_reconciliation_id(self) -> int:
        """Allocate the next surrogate reconciliation key."""
        allocated = self._next_reconciliation_id
        self._next_reconciliation_id += 1
        return allocated


class ControlSessionDouble:
    """An ``AsyncSession`` stand-in over a :class:`ControlRows`.

    Implements exactly the two methods the control stores call — ``execute`` and
    ``begin_nested``. Yields to the event loop on entry to ``execute`` so
    concurrent callers interleave, while the check-and-write for a clearance
    happens with no ``await`` between the two, exactly as the database behaves.
    """

    def __init__(self, rows: ControlRows) -> None:
        """Open a handle over an existing set of rows."""
        self.rows = rows

    def begin_nested(self) -> _Savepoint:
        """Return a savepoint context manager."""
        return _Savepoint()

    async def execute(self, statement: Any) -> _Result:  # noqa: ANN401 - SQLAlchemy statements
        """Dispatch one statement against the in-memory tables."""
        await asyncio.sleep(0)
        if isinstance(statement, Insert):
            if self.rows.insert_barrier is not None:
                await self.rows.insert_barrier.wait()
            return self._insert(statement)
        if isinstance(statement, Select):
            if self.rows.read_failure is not None:
                raise self.rows.read_failure
            return self._select(statement)
        msg = f"the double does not model {type(statement).__name__} statements"
        raise AssertionError(msg)

    def _insert(self, statement: Insert) -> _Result:
        """Apply an INSERT atomically."""
        table = statement.table.name
        values = dict(statement.compile().params)
        if table == "execution_halt":
            return self._insert_halt(values)
        if table == "execution_reconciliation":
            return self._insert_reconciliation(values)
        msg = f"the double does not model inserts into {table}"
        raise AssertionError(msg)

    def _insert_halt(self, values: dict[str, object]) -> _Result:
        """Insert one halt row, refusing clearances the way the database refuses them.

        **An engagement is never examined.** There is deliberately no branch here
        that can reject one: a double that could refuse a halt-engage row would be
        modelling a schema in which the kill switch can be prevented from firing,
        and the real trigger returns ``NEW`` unconditionally for an engagement.
        """
        self.rows.halt_insert_attempts += 1
        if values.get("event") != HaltEventKind.CLEARED.value:
            halt_id = self.rows.next_halt_id()
            self.rows.halts.append({**values, "halt_id": halt_id})
            return _Result([(halt_id,)])
        target_id = values.get("clears_halt_id")
        # The chain of refusals in the database's own order: the BEFORE INSERT
        # trigger runs ahead of every constraint, so these come before the index.
        target = next((row for row in self.rows.halts if row["halt_id"] == target_id), None)
        if target is None:
            raise DBAPIError(
                "INSERT INTO execution_halt",
                values,
                DriverError(
                    HALT_CLEARANCE_REFUSED_SQLSTATE,
                    f"clearance names halt_id {target_id} which does not exist",
                ),
            )
        if target["event"] != HaltEventKind.ENGAGED.value:
            raise DBAPIError(
                "INSERT INTO execution_halt",
                values,
                DriverError(
                    HALT_CLEARANCE_REFUSED_SQLSTATE,
                    f"clearance names halt_id {target_id}, whose event is {target['event']}",
                ),
            )
        already = next(
            (
                row
                for row in self.rows.halts
                if row["event"] == HaltEventKind.CLEARED.value
                and row["clears_halt_id"] == target_id
            ),
            None,
        )
        if already is not None:
            raise self._already_cleared(values, target_id)
        halt_id = self.rows.next_halt_id()
        self.rows.halts.append({**values, "halt_id": halt_id})
        return _Result([(halt_id,)])

    def _already_cleared(self, values: dict[str, object], target_id: object) -> Exception:
        """Build the refusal for a halt that is already cleared, by the selected route.

        Both routes carry ``23505``. The trigger route is the likelier one under
        contention (``READ COMMITTED`` gives the trigger a fresh snapshot the
        loser's own earlier read did not have) and is the default; the index route
        exists so the claim that the caller cannot tell them apart is tested
        rather than asserted.
        """
        message = f"halt_id {target_id} was already cleared"
        if self.rows.clearance_refusal_route == "index":
            return IntegrityError(
                "INSERT INTO execution_halt",
                values,
                DriverError(
                    UNIQUE_VIOLATION_SQLSTATE,
                    "duplicate key value violates unique constraint "
                    '"uq_execution_halt_clears_halt_id"',
                ),
            )
        return DBAPIError(
            "INSERT INTO execution_halt",
            values,
            DriverError(UNIQUE_VIOLATION_SQLSTATE, message),
        )

    def _insert_reconciliation(self, values: dict[str, object]) -> _Result:
        """Insert one reconciliation row."""
        reconciliation_id = self.rows.next_reconciliation_id()
        self.rows.reconciliations.append({**values, "reconciliation_id": reconciliation_id})
        return _Result([(reconciliation_id,)])

    def _select(self, statement: Select[Any]) -> _Result:
        """Apply a SELECT, routed by the table it reads."""
        table = cast("sa.Table", statement.get_final_froms()[0]).name
        columns = list(statement.selected_columns.keys())
        params = dict(statement.compile().params)
        if table == "execution_halt":
            open_ids = set(self.rows.open_halt_ids())
            matches = [row for row in self.rows.halts if row["halt_id"] in open_ids]
            matches.sort(key=lambda row: int(cast("int", row["halt_id"])))
            return _Result([tuple(row.get(name) for name in columns) for row in matches])
        if table == "execution_reconciliation":
            wanted = params.get("reconciliation_id_1")
            matches = [
                row for row in self.rows.reconciliations if row["reconciliation_id"] == wanted
            ]
            return _Result([tuple(row.get(name) for name in columns) for row in matches])
        msg = f"the double does not model selects from {table}"
        raise AssertionError(msg)


def as_session(double: ControlSessionDouble) -> Any:  # noqa: ANN401 - deliberate duck typing
    """Present the double where an ``AsyncSession`` is annotated.

    The control stores are annotated against ``AsyncSession`` because that is what
    production passes. The double implements the two methods they call; this
    function states the substitution once rather than with a cast at every call
    site.
    """
    return double


def restart(rows: ControlRows) -> ControlSessionDouble:
    """Return a brand-new session over the same rows — a simulated process restart.

    Nothing from the previous session survives except the rows, which is the whole
    claim: if a halt were held in memory it would be gone, and if it is a row it is
    still there.
    """
    return ControlSessionDouble(rows)
