"""Builders shared by the execution suite: stamps, intents, and an in-memory store double.

The double models the *two tables*, not the ORM. It implements exactly the two
session methods :mod:`backend.execution.store` uses — ``execute`` and
``begin_nested`` — and enforces the one property the concurrency proof turns on:
the uniqueness check and the insert happen atomically, the way a unique index
does, with no window between them. Everything *around* that atomic step yields to
the event loop, so concurrent callers genuinely interleave up to the boundary.

What it does and does not prove is worth being exact about. It proves the control
flow in ``record_order`` and ``append_transition`` is correct when N coroutines
race: one winner, N-1 absorbed or refused, one order. It does **not** prove that
Postgres enforces anything — that is
``backend/tests/integration/test_order_lifecycle.py``.

Where this double was wrong, and what it cost
---------------------------------------------

The first version modelled a concurrent transition append as *the primary key*
refusing the loser, raising a bare ``IntegrityError`` whose ``orig`` carried no
SQLSTATE. Both halves were wrong, and the suite was green anyway until CI ran the
integration tests against a real daemon:

- **The trigger fires before the index.** ``execution_order_transition`` has a
  ``BEFORE INSERT`` chain guard, and a ``BEFORE`` row trigger runs ahead of every
  constraint. A writer whose ``INSERT`` starts after the winner committed is
  refused by the *trigger* — under ``READ COMMITTED`` its statement sees a tail
  its own earlier ``SELECT`` did not — and never reaches the index at all. That
  is the likelier path under contention, and the double had no model of it.
- **Refusals carry a code.** The store decides retryable-versus-malformed on
  SQLSTATE, so a double whose errors carry none cannot distinguish a correct
  store from one that translates every failure into a retry.

The double now refuses in the database's order, with the database's codes. The
lesson is narrower than "doubles lie": a double that models a *constraint* while
the schema also has a *trigger* is modelling the wrong layer.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.sql import Insert, Select

from backend.execution.orders import (
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
)
from backend.execution.store import UNIQUE_VIOLATION_SQLSTATE
from backend.tracking.stamp import ReproducibilityStamp

if TYPE_CHECKING:
    from collections.abc import Sequence

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
CONFIG_HASH_A = "1" * 64
CONFIG_HASH_B = "2" * 64

PAPER_VENUE = "paper"

RAISE_EXCEPTION_SQLSTATE = "P0001"
"""SQLSTATE of a plpgsql ``RAISE EXCEPTION`` that names no code of its own.

The chain guard's non-retryable refusals — a gap, a chain that does not connect,
an overfill — arrive under this code. The store must **not** translate them into
``ConcurrentTransitionError``: retrying a malformed row forever is worse than
failing once.
"""


class DriverError(Exception):
    """Stand-in for the driver exception SQLAlchemy wraps in a ``DBAPIError``.

    Carries a ``sqlstate`` attribute, which is what asyncpg exposes and what
    :func:`backend.execution.store._is_position_taken` reads. The double raised a
    plain ``Exception`` before, so every error it produced looked
    indistinguishable to code that inspects the SQLSTATE — which is precisely
    the distinction the store now depends on.
    """

    def __init__(self, sqlstate: str, message: str) -> None:
        """Build the error from a SQLSTATE and the message the server would send."""
        super().__init__(message)
        self.sqlstate = sqlstate


MOMENT = dt.datetime(2026, 8, 3, 14, 30, tzinfo=dt.UTC)
"""A fixed, timezone-aware instant for ``occurred_at`` in tests."""


def make_stamp(
    *,
    git_commit: str = COMMIT_A,
    git_dirty: bool = False,
    data_version: str = "sharadar-2026-08-01",
    config_hash: str = CONFIG_HASH_A,
    seed: int = 7,
) -> ReproducibilityStamp:
    """Build a valid I2 stamp without touching git."""
    return ReproducibilityStamp(
        git_commit=git_commit,
        git_dirty=git_dirty,
        data_version=data_version,
        config_hash=config_hash,
        seed=seed,
    )


def make_intent(
    *,
    security_id: int = 42,
    side: Side = Side.BUY,
    quantity_shares: int = 100,
    order_type: OrderType = OrderType.LIMIT,
    time_in_force: TimeInForce = TimeInForce.DAY,
    limit_price_usd: Decimal | None = Decimal("123.45"),
    rebalance_date: dt.date = dt.date(2026, 8, 3),
    slice_index: int = 0,
    slice_count: int = 1,
    stamp: ReproducibilityStamp | None = None,
) -> OrderIntent:
    """Build a valid order intent, overriding one field at a time."""
    return OrderIntent(
        security_id=security_id,
        side=side,
        quantity_shares=quantity_shares,
        order_type=order_type,
        time_in_force=time_in_force,
        limit_price_usd=limit_price_usd,
        rebalance_date=rebalance_date,
        slice_index=slice_index,
        slice_count=slice_count,
        stamp=make_stamp() if stamp is None else stamp,
    )


class _Result:
    """The slice of ``Result`` the store uses: ``scalar_one``, ``first``, ``all``."""

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
    """Stand-in for ``AsyncSessionTransaction``.

    Nothing to roll back: the double's writes are atomic, so a failed statement
    leaves nothing behind. It exists so ``async with session.begin_nested()``
    works, and so the store's use of a savepoint is exercised rather than
    bypassed.
    """

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


class PaperOrderStoreDouble:
    """In-memory stand-in for ``execution_order`` and ``execution_order_transition``.

    Enforces the constraints the store depends on, and only those:

    - ``UNIQUE (idempotency_key)`` on orders;
    - the per-order sequence on transitions, refused the way the *database*
      refuses it — see :meth:`_insert_transition`.

    All of them are enforced *atomically* — the check and the write happen with
    no ``await`` between them, exactly as a unique index behaves — while
    :meth:`execute` yields to the event loop on entry, so concurrent callers
    interleave freely up to the atomic boundary.

    Every refusal carries a :class:`DriverError` with a real SQLSTATE, because
    the store distinguishes retryable refusals from malformed ones on that code.
    A double whose errors carry no code cannot tell the store apart from a
    version of the store that translates everything.

    ``venue`` is defaulted here rather than accepted, mirroring the server
    default: the store never supplies it, and the double would have nowhere to
    put it if it tried.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.orders: dict[int, dict[str, object]] = {}
        self.key_index: dict[str, int] = {}
        self.transitions: dict[int, list[dict[str, object]]] = {}
        self.order_insert_attempts = 0
        self.transition_insert_attempts = 0
        self.insert_barrier: asyncio.Barrier | None = None
        """Set to an ``asyncio.Barrier`` of N parties to force N writers to arrive
        at the contention point simultaneously. Without it the scheduler may
        happen to run the callers one after another, and a test that passes
        because nothing actually raced has proved nothing."""
        self._next_order_id = 1

    def begin_nested(self) -> _Savepoint:
        """Return a savepoint context manager."""
        return _Savepoint()

    def seed_order(self, **columns: object) -> int:
        """Insert a row directly, bypassing the store — for corrupt-row tests.

        Returns:
            The new ``order_id``.
        """
        order_id = self._next_order_id
        self._next_order_id += 1
        row: dict[str, object] = {"venue": PAPER_VENUE, "order_id": order_id}
        row.update(columns)
        self.orders[order_id] = row
        self.key_index[str(row["idempotency_key"])] = order_id
        self.transitions[order_id] = []
        return order_id

    async def execute(self, statement: Any) -> _Result:  # noqa: ANN401 - SQLAlchemy statements
        """Dispatch one statement against the in-memory tables.

        Yields to the event loop first, so every caller reaches the atomic step
        by a different path on different runs.
        """
        await asyncio.sleep(0)
        if isinstance(statement, Insert):
            if self.insert_barrier is not None:
                await self.insert_barrier.wait()
            return self._insert(statement)
        if isinstance(statement, Select):
            return self._select(statement)
        msg = f"the double does not model {type(statement).__name__} statements"
        raise AssertionError(msg)

    def _insert(self, statement: Insert) -> _Result:
        """Apply an INSERT atomically, raising IntegrityError on a key clash."""
        table = statement.table.name
        values = dict(statement.compile().params)
        if table == "execution_order":
            return self._insert_order(values)
        if table == "execution_order_transition":
            return self._insert_transition(values)
        msg = f"the double does not model inserts into {table}"
        raise AssertionError(msg)

    def _insert_order(self, values: dict[str, object]) -> _Result:
        """Insert one order; the uniqueness check and the write are one step."""
        self.order_insert_attempts += 1
        key = str(values["idempotency_key"])
        # No await between the check and the write: that is the whole point.
        if key in self.key_index:
            raise IntegrityError(
                "INSERT INTO execution_order",
                values,
                DriverError(
                    UNIQUE_VIOLATION_SQLSTATE,
                    "duplicate key value violates unique constraint "
                    '"uq_execution_order_idempotency_key"',
                ),
            )
        order_id = self._next_order_id
        self._next_order_id += 1
        row: dict[str, object] = {"venue": PAPER_VENUE, "order_id": order_id}
        row.update(values)
        self.orders[order_id] = row
        self.key_index[key] = order_id
        self.transitions[order_id] = []
        return _Result([(order_id,)])

    def _insert_transition(self, values: dict[str, object]) -> _Result:
        """Insert one transition, modelling both refusals the database can give.

        **The chain guard first, the index second — that is the order Postgres
        uses and the order this double got wrong before CI ran it.** A
        ``BEFORE INSERT`` trigger fires ahead of every constraint, so a writer
        whose statement starts after the winner committed is refused by the
        *trigger* (``23505`` by migration 0014's ``USING ERRCODE``), never by the
        index. Modelling only the index made the double raise a bare
        ``IntegrityError`` with no SQLSTATE and hid the whole path.

        A sequence number at or below the tail is "position taken" (``23505``,
        retryable). A gap above it is a malformed chain (``P0001``, not
        retryable), and the store must be seen to keep them apart.
        """
        self.transition_insert_attempts += 1
        order_id = int(cast("int", values["order_id"]))
        sequence_number = int(cast("int", values["sequence_number"]))
        history = self.transitions.setdefault(order_id, [])
        tail = max((int(cast("int", row["sequence_number"])) for row in history), default=0)
        if sequence_number <= tail:
            raise IntegrityError(
                "INSERT INTO execution_order_transition",
                values,
                DriverError(
                    UNIQUE_VIOLATION_SQLSTATE,
                    f"order {order_id} is already at sequence {tail}, so position "
                    f"{sequence_number} is taken",
                ),
            )
        if sequence_number != tail + 1:
            raise DBAPIError(
                "INSERT INTO execution_order_transition",
                values,
                DriverError(
                    RAISE_EXCEPTION_SQLSTATE,
                    f"order {order_id} is at sequence {tail}, so the next transition is "
                    f"{tail + 1}, not {sequence_number}",
                ),
            )
        history.append(dict(values))
        return _Result([])

    def _select(self, statement: Select[Any]) -> _Result:
        """Apply a SELECT, routed by the columns it projects."""
        columns = list(statement.selected_columns.keys())
        params = dict(statement.compile().params)
        table = cast("sa.Table", statement.get_final_froms()[0]).name
        if table == "execution_order":
            return _Result(self._select_orders(columns, params))
        if table == "execution_order_transition":
            return _Result(self._select_transitions(columns, params))
        msg = f"the double does not model selects from {table}"
        raise AssertionError(msg)

    def _select_orders(
        self, columns: list[str], params: dict[str, object]
    ) -> list[tuple[object, ...]]:
        """Return matching order rows, projected to ``columns``."""
        matches = list(self.orders.values())
        if "idempotency_key_1" in params:
            wanted = params["idempotency_key_1"]
            matches = [row for row in matches if row.get("idempotency_key") == wanted]
        if "order_id_1" in params:
            wanted_id = params["order_id_1"]
            matches = [row for row in matches if row.get("order_id") == wanted_id]
        return [tuple(row.get(name) for name in columns) for row in matches]

    def _select_transitions(
        self, columns: list[str], params: dict[str, object]
    ) -> list[tuple[object, ...]]:
        """Return one order's transitions in sequence order, projected to ``columns``."""
        order_id = int(cast("int", params["order_id_1"]))
        history = sorted(
            self.transitions.get(order_id, []),
            key=lambda row: int(cast("int", row["sequence_number"])),
        )
        return [tuple(row.get(name) for name in columns) for row in history]


def as_session(double: PaperOrderStoreDouble) -> Any:  # noqa: ANN401 - deliberate duck typing
    """Present the double where an ``AsyncSession`` is annotated.

    The store's functions are annotated against ``AsyncSession`` because that is
    what production passes. The double implements the two methods they actually
    call; this function is where that substitution is stated once, rather than
    with a ``cast`` at every call site.
    """
    return double


def order_columns(intent: OrderIntent, *, idempotency_key: str, preimage: str) -> dict[str, object]:
    """Return the column values :func:`seed_order` needs for one intent."""
    return {
        "idempotency_key": idempotency_key,
        "idempotency_preimage": preimage,
        "security_id": intent.security_id,
        "side": intent.side.value,
        "quantity_shares": intent.quantity_shares,
        "order_type": intent.order_type.value,
        "time_in_force": intent.time_in_force.value,
        "limit_price_usd": intent.limit_price_usd,
        "rebalance_date": intent.rebalance_date,
        "slice_index": intent.slice_index,
        "slice_count": intent.slice_count,
    }


def statement_tables(statement: sa.Select[Any]) -> list[str]:
    """Return the table names a select reads, for assertions about read shape."""
    return [cast("sa.Table", source).name for source in statement.get_final_froms()]
