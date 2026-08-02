"""Bitemporal column mixin and mapper registry (DECISIONS.md D-011).

Every fact table carries three temporal columns, all ``TIMESTAMPTZ`` in UTC:

- ``valid_from`` / ``valid_to`` — the **event-time** interval the fact
  describes, half-open ``[valid_from, valid_to)``. Open-ended facts store
  ``'infinity'`` in ``valid_to`` (the server default) so range predicates
  stay uniform; Python-side that value is the aware sentinel
  :data:`INFINITY` (see :class:`InfinityDateTime`). ``valid_from <
  valid_to`` is a CHECK constraint.
- ``knowledge_time`` — when the information became knowable to the market.
  It has **no default of any kind**: writers must supply it explicitly under
  their connector's documented knowledge-time policy.

Separately, ``ingested_at`` (audit only, server-defaulted to ``now()``)
records when our pipeline wrote the row; it is never used in queries, so
backfilled history keeps an honest historical ``knowledge_time``.

Rows are never updated or deleted. A correction is a new row for the same
(logical key, ``valid_from``) with a later ``knowledge_time``; a retraction
is such a row with ``is_retraction = true``. The as-of read semantics
(latest ``knowledge_time <= as_of`` wins; a winning retraction hides the
fact) are enforced by :mod:`backend.db.asof`, which consumes the registry
kept here.

Retraction payloads (P2.10)
---------------------------

A retraction is a *row*, and a row has to fill every column the table
declares. Before this, that meant a retraction of a price bar carried a
``close_usd``, a ``volume_shares`` and an ``adjustment_factor`` invented by
whoever wrote it — numbers that no source ever stated, sitting in a fact
table, byte-identical to a real observation. That is fabricated data inside
the store invariant I1 reads from (I3, directive §9.1/§9.2). Documenting the
convention is not a fix: the storage layer still cannot tell the two apart,
and neither can anything reading it.

**The rule now enforced: a retraction's payload is NULL, and only a
retraction's payload may be NULL.** Concretely, for every registered fact
table :meth:`BitemporalMixin.__init_subclass__` derives

- ``__bitemporal_payload__`` — every mapped column that is neither one of
  the five temporal/audit columns above nor part of ``__bitemporal_key__``;
- ``__bitemporal_required_payload__`` — the payload columns an *observation*
  must state, taken from the model's own ``Mapped[...]`` annotations: a
  non-optional annotation means required, ``Mapped[X | None]`` means the
  source may legitimately not state it;

drops the ``NOT NULL`` on the payload columns (so a retraction can be
written at all) and puts the requirement back as two CHECK constraints that
say strictly more than ``NOT NULL`` did:

- ``ck_<table>_retraction_payload_absent`` — ``NOT is_retraction OR (every
  payload column IS NULL)``. A retraction that carries a value is refused by
  the database, on every role, every session and every write path including
  raw Core inserts and COPY.
- ``ck_<table>_observation_payload_present`` — ``is_retraction OR (every
  required payload column IS NOT NULL)``. Observations keep exactly the
  guarantee ``NOT NULL`` gave them; nothing was relaxed.

NULL is the right marker because it is the one value in SQL that is *not* a
value: it does not compare equal to anything, aggregates skip it, and no
arithmetic silently consumes it. There is nothing to mistake for a price.

Alternatives rejected (argued in full in the P2.10 report):

- **A sentinel payload** (``-1``, ``NaN``, ``0``) is the worst of the four
  and is rejected outright. It is fabricated data by construction — a number
  in a numeric column — so ``AVG(close_usd)`` consumes it, a leaked
  retraction hands a caller something that *looks* like a quote, and the
  sentinel has to be chosen per column, each choice a fresh chance to pick a
  legal value (``0`` is a legal volume, ``-1`` a legal return, ``1`` a legal
  adjustment factor). It fails I3 in the same breath it claims to serve it.
- **Nullable payloads with optional Python types** — the same storage shape,
  but re-annotating every payload column ``Mapped[X | None]``. Rejected on
  the read contract, not on effort: the as-of layer already masks
  retractions, so no caller can obtain a row whose payload is NULL, and
  spreading ``| None`` across every consumer would demand a ``None`` check
  at every use site for a state that path cannot produce. Checks that can
  never fire are how real ones stop being read.
- **A separate retraction table** keeps fact tables free of retraction rows,
  but a retraction has to *compete* in the latest-knowledge ordering — beat
  an earlier observation, lose to a later re-assertion — so the versioned
  read would have to UNION the two tables back into one stream and then
  split them again. That reconstructs this design with a worse query plan,
  splits the primary key that makes latest-wins deterministic across two
  tables where no single constraint can hold it, and needs either one extra
  table per fact table or an untypeable JSONB logical key.

Write-side hygiene is layered on top of the constraint rather than trusted
instead of it: :func:`backend.ingest.supersession.retract_fact` is the
sanctioned constructor and never sets a payload, and the ORM
``before_insert`` hook installed below clears the payload of any retraction
flushed through the ORM (logging what it dropped) so an older writer that
still fills those columns stops fabricating rather than starts failing.
Read-side enforcement — the masking predicate, its structural backstop, and
the row-level refusal to hand a loaded retraction to a caller — lives in
:mod:`backend.db.asof`.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, cast

from sqlalchemy import Boolean, CheckConstraint, Index, PrimaryKeyConstraint, Table, event, text
from sqlalchemy.orm import Mapped, Mapper, class_mapper, declared_attr, mapped_column, validates
from sqlalchemy.sql import func, null
from sqlalchemy.sql.elements import Null
from sqlalchemy.types import DateTime, TypeDecorator

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.engine.interfaces import Dialect

_logger = get_logger(__name__)

_registry: list[type[BitemporalMixin]] = []
"""Every mapped class carrying :class:`BitemporalMixin`, in definition order."""

TEMPORAL_COLUMNS: tuple[str, ...] = (
    "valid_from",
    "valid_to",
    "knowledge_time",
    "ingested_at",
    "is_retraction",
)
"""The five columns :class:`BitemporalMixin` contributes, in ``sort_order``.

Named as a constant because "payload" is defined by subtraction: a fact
table's payload is every mapped column that is neither one of these nor part
of the model's ``__bitemporal_key__``. Deriving it rather than listing it per
model is what makes the P2.10 constraints appear on fact tables added by later
phases without anyone remembering to add them.
"""

RETRACTION_PAYLOAD_ABSENT = "retraction_payload_absent"
"""Unprefixed name of the CHECK forbidding a payload on a retraction row.

Unprefixed because the metadata naming convention
(``ck_%(table_name)s_%(constraint_name)s``, :mod:`backend.db.base`) expands
it; the constraint reaches the database as
``ck_<table>_retraction_payload_absent``.
"""

OBSERVATION_PAYLOAD_PRESENT = "observation_payload_present"
"""Unprefixed name of the CHECK requiring a complete payload on an observation.

Carries the guarantee the payload columns' ``NOT NULL`` used to carry, minus
nothing: it is asserted for exactly the columns whose ``Mapped[...]``
annotation is non-optional.
"""

INFINITY = dt.datetime.max.replace(tzinfo=dt.UTC)
"""Timezone-aware sentinel for an open-ended ``valid_to`` (PG ``'infinity'``).

``datetime.max`` with ``tzinfo=UTC``: orderable against any aware datetime
(later than every representable instant), unlike the *naive* ``datetime.max``
asyncpg hands back for ``'infinity'``, which raises ``TypeError`` on any
aware comparison. :class:`InfinityDateTime` translates between this sentinel
and PG ``'infinity'`` in both directions.
"""

NEGATIVE_INFINITY = dt.datetime.min.replace(tzinfo=dt.UTC)
"""Timezone-aware sentinel for PG ``'-infinity'`` (symmetry; not used by any
current fact semantics, but decoded losslessly rather than leaking a naive
``datetime.min``)."""

# DTZ901 (naive datetime.max/min) is suppressed deliberately on the next two
# constants: their *naiveness is the specification*. They are not timestamps we
# choose — they are the exact naive values asyncpg produces for, and expects
# for, PG 'infinity'/'-infinity'. Making them aware would break the round-trip
# InfinityDateTime exists to perform.
_PG_INFINITY_NAIVE = dt.datetime.max  # noqa: DTZ901
"""The naive value asyncpg decodes PG ``'infinity'`` to (and encodes back)."""

_PG_NEGATIVE_INFINITY_NAIVE = dt.datetime.min  # noqa: DTZ901
"""The naive value asyncpg decodes PG ``'-infinity'`` to (and encodes back)."""


class InfinityDateTime(TypeDecorator[dt.datetime]):
    """``TIMESTAMPTZ`` that round-trips PG ``'infinity'`` as an aware sentinel.

    asyncpg maps ``'infinity'``/``'-infinity'`` to *naive*
    ``datetime.max``/``datetime.min``, which breaks every comparison against
    aware datetimes. This decorator (applied to ``valid_from`` and
    ``valid_to`` on :class:`BitemporalMixin`):

    - **decodes** those naive driver values to the aware module-level
      sentinels :data:`INFINITY` / :data:`NEGATIVE_INFINITY`;
    - **encodes** the sentinels back to the naive driver values, which
      asyncpg writes as PG ``'infinity'``/``'-infinity'`` — so the sentinel
      persists as true PG infinity, keeping range predicates uniform;
    - **rejects** any other naive bind value with ``TypeError`` (second net
      under the ORM-level validator: asyncpg would silently reinterpret a
      naive datetime in *host-local* time).

    Note the deliberate canonicalization: an explicit aware
    ``9999-12-31 23:59:59.999999+00`` equals :data:`INFINITY` and is
    therefore stored as ``'infinity'``. Assumptions: asyncpg driver; all
    stored values UTC (D-011).
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: dt.datetime | None,
        dialect: Dialect,  # noqa: ARG002 — TypeDecorator API signature
    ) -> dt.datetime | None:
        """Encode sentinels to the driver's infinity values; reject naive input."""
        if value is None:
            return None
        if value == INFINITY:
            return _PG_INFINITY_NAIVE
        if value == NEGATIVE_INFINITY:
            return _PG_NEGATIVE_INFINITY_NAIVE
        if value.tzinfo is None or value.utcoffset() is None:
            msg = (
                f"naive datetime {value!r} bound to a bitemporal temporal column: "
                "temporal values must be timezone-aware UTC (asyncpg silently "
                "reinterprets naive datetimes in host-local time; D-011)"
            )
            raise TypeError(msg)
        return value

    def process_result_value(
        self,
        value: dt.datetime | None,
        dialect: Dialect,  # noqa: ARG002 — TypeDecorator API signature
    ) -> dt.datetime | None:
        """Decode the driver's naive infinity values to the aware sentinels."""
        if value is None or value.tzinfo is not None:
            return value
        if value == _PG_INFINITY_NAIVE:
            return INFINITY
        if value == _PG_NEGATIVE_INFINITY_NAIVE:
            return NEGATIVE_INFINITY
        msg = (
            f"naive datetime {value!r} loaded from a bitemporal temporal column: "
            "TIMESTAMPTZ must decode timezone-aware (only PG infinity decodes "
            "naive, and it maps to the aware sentinels; D-011)"
        )
        raise ValueError(msg)


_INFINITY_DATETIME = InfinityDateTime()
"""Shared type instance for the event-time columns of every fact table."""


class BitemporalMixin:
    """Declarative mixin adding the D-011 bitemporal columns to a fact table.

    Subclasses must declare ``__bitemporal_key__``: the column names that
    identify the *logical entity* (e.g. ``("security_id",)``). Together with
    ``valid_from`` it addresses one fact; together with ``knowledge_time``
    it addresses one *version* of that fact (the composite primary key).
    Subclassing registers the model in the bitemporal registry consumed by
    the as-of query layer and by tests; a subclass without a valid key is
    rejected at class-definition time.

    Temporal hygiene (both enforced, neither is convention):

    - every temporal value written through the ORM **must be timezone-aware**
      — a naive ``valid_from``/``valid_to``/``knowledge_time`` raises
      ``TypeError`` at attribute-assignment time, before any I/O, because
      asyncpg would otherwise silently reinterpret it in host-local time;
    - an **open-ended ``valid_to``** is PG ``'infinity'`` in the database and
      the aware sentinel :data:`INFINITY` (``datetime.max`` with
      ``tzinfo=UTC``) in Python, in both directions
      (:class:`InfinityDateTime`), so loaded values always compare cleanly
      against aware datetimes. Leave ``valid_to`` unset to take the
      ``'infinity'`` server default, or set it to :data:`INFINITY`
      explicitly — both persist as PG infinity.
    """

    __bitemporal_key__: ClassVar[tuple[str, ...]]

    __bitemporal_payload__: ClassVar[tuple[str, ...]]
    """Derived: every mapped column that is neither temporal nor part of the key.

    Set by :meth:`__init_subclass__` in table-column order. These are the
    columns a retraction must leave NULL and an observation fills; the
    database enforces both directions (module docstring, P2.10).
    """

    __bitemporal_required_payload__: ClassVar[tuple[str, ...]]
    """Derived: the payload columns an **observation** must state.

    Read off the model's own annotations before the ``NOT NULL`` is dropped:
    ``Mapped[Decimal]`` is required, ``Mapped[Decimal | None]`` is not. A
    column absent from this tuple is one the *source* may legitimately not
    state (an unknown listing date, a filing with no reporting period) — not
    one the platform may invent.
    """

    valid_from: Mapped[dt.datetime] = mapped_column(
        _INFINITY_DATETIME,
        nullable=False,
        sort_order=91,
        doc="Event-time interval start (inclusive), UTC. For a daily bar of day D: D 00:00Z.",
    )
    valid_to: Mapped[dt.datetime] = mapped_column(
        _INFINITY_DATETIME,
        nullable=False,
        server_default=text("'infinity'::timestamptz"),
        sort_order=92,
        doc=(
            "Event-time interval end (exclusive), UTC; 'infinity' when open-ended "
            "(Python-side: the aware INFINITY sentinel)."
        ),
    )
    knowledge_time: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        sort_order=93,
        doc="When the fact became knowable to the market, UTC. Writer-supplied; no default.",
    )
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        sort_order=94,
        doc="Audit only: when our pipeline wrote the row. Never used in queries (D-011).",
    )
    is_retraction: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
        default=False,
        sort_order=95,
        doc=(
            "True marks a retraction version: it repeats the logical key and valid "
            "interval, carries NO payload (every payload column IS NULL, enforced by "
            "CHECK), and hides the fact from any as_of at or after its knowledge_time."
        ),
    )

    @validates("valid_from", "valid_to", "knowledge_time")
    def _require_timezone_aware(self, key: str, value: object) -> object:
        """Reject naive temporal values at assignment time, before any I/O.

        asyncpg silently reinterprets naive datetimes in **host-local** time,
        which would corrupt every temporal comparison in the store (D-011).
        Non-datetime values (e.g. SQL expressions) pass through for the
        database/type layer to judge.
        """
        if isinstance(value, dt.datetime) and (value.tzinfo is None or value.utcoffset() is None):
            msg = (
                f"{type(self).__name__}.{key} must be a timezone-aware UTC datetime; "
                f"got naive {value!r}. asyncpg silently reinterprets naive datetimes "
                "in host-local time, so naive temporal values are rejected before "
                "any I/O (D-011)"
            )
            raise TypeError(msg)
        return value

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate ``__bitemporal_key__`` and register the subclass.

        Raises ``TypeError`` — before the declarative machinery maps the
        class, so a rejected model leaves no trace in the metadata — when
        the key is missing, empty, or names an attribute the class does not
        define.
        """
        key = getattr(cls, "__bitemporal_key__", None)
        if not key or not isinstance(key, tuple):
            msg = (
                f"{cls.__name__} must declare __bitemporal_key__: a non-empty tuple of "
                "column names identifying the logical entity (D-011)"
            )
            raise TypeError(msg)
        missing = [column_name for column_name in key if not hasattr(cls, column_name)]
        if missing:
            msg = (
                f"{cls.__name__}.__bitemporal_key__ names undefined column(s): {', '.join(missing)}"
            )
            raise TypeError(msg)
        super().__init_subclass__(**kwargs)
        _apply_payload_contract(cls)
        _registry.append(cls)

    @declared_attr.directive
    @classmethod
    def __table_args__(cls) -> tuple[Any, ...]:
        """Derive per-table constraints and the as-of index from the logical key.

        - Primary key ``(*__bitemporal_key__, valid_from, knowledge_time)``:
          one row per version; equal-``knowledge_time`` duplicates are
          impossible, so latest-knowledge-wins has a unique winner.
        - ``CHECK (valid_from < valid_to)`` — non-empty half-open interval.
        - Composite index ``(*key, valid_from, knowledge_time DESC)`` matching
          the as-of access pattern exactly (D-011 physical layout; created in
          the database by migration 0003).
        """
        tablename = cast(str, cls.__tablename__)  # type: ignore[attr-defined]
        return (
            PrimaryKeyConstraint(*cls.__bitemporal_key__, "valid_from", "knowledge_time"),
            CheckConstraint("valid_from < valid_to", name="valid_interval"),
            Index(
                f"ix_{tablename}_asof_lookup",
                *cls.__bitemporal_key__,
                "valid_from",
                text("knowledge_time DESC"),
            ),
        )


def _apply_payload_contract(cls: type[BitemporalMixin]) -> None:
    """Derive the payload columns of a freshly mapped fact table and constrain them.

    Runs immediately after the declarative machinery has built ``cls.__table__``
    (so every column's nullability is already resolved from its ``Mapped[...]``
    annotation) and before the class is registered. Three effects, in order:

    1. records ``__bitemporal_payload__`` (all columns bar the temporal five
       and the logical key) and ``__bitemporal_required_payload__`` (those of
       them the annotations declared non-optional);
    2. drops ``NOT NULL`` from every payload column, because a retraction row
       has no payload to put there and the alternative is fabricating one
       (module docstring, I3);
    3. attaches the two CHECK constraints that make (2) safe — a retraction
       may hold *only* NULLs, an observation must hold every required value —
       so the guarantee ``NOT NULL`` gave observations survives intact and the
       retraction rule joins it at the same level of enforcement.

    A fact table whose columns are all key and temporal has no payload to
    constrain and gets neither constraint; both would be tautologies. Raises
    ``TypeError`` if the class has no ``__table__`` (single-table inheritance
    or a hand-supplied ``__table__`` — shapes the D-011 physical layout does
    not contemplate, refused loudly rather than silently left unconstrained).
    """
    table = cast(Table | None, getattr(cls, "__table__", None))
    if table is None:
        msg = (
            f"{cls.__name__} carries BitemporalMixin but has no __table__ of its own; "
            "every fact table must map to a dedicated table so the P2.10 payload "
            "constraints can be attached to it (D-011)"
        )
        raise TypeError(msg)
    excluded = {*TEMPORAL_COLUMNS, *cls.__bitemporal_key__}
    payload = tuple(column.name for column in table.columns if column.name not in excluded)
    cls.__bitemporal_payload__ = payload
    cls.__bitemporal_required_payload__ = tuple(
        name for name in payload if not table.c[name].nullable
    )
    for name in payload:
        table.c[name].nullable = True
    if not payload:
        return
    absent = " AND ".join(f"{name} IS NULL" for name in payload)
    table.append_constraint(
        CheckConstraint(f"NOT is_retraction OR ({absent})", name=RETRACTION_PAYLOAD_ABSENT)
    )
    if cls.__bitemporal_required_payload__:
        present = " AND ".join(
            f"{name} IS NOT NULL" for name in cls.__bitemporal_required_payload__
        )
        table.append_constraint(
            CheckConstraint(f"is_retraction OR ({present})", name=OBSERVATION_PAYLOAD_PRESENT)
        )


def _carries_value(value: object) -> bool:
    """True when a payload attribute holds something that would reach the database.

    Python ``None`` and SQL ``NULL`` (:class:`~sqlalchemy.sql.elements.Null`,
    what a correctly built retraction already carries) both mean "no value";
    anything else is a value that clearing would discard, and is therefore
    worth naming in the warning.
    """
    return value is not None and not isinstance(value, Null)


@event.listens_for(Mapper, "before_insert")
def _clear_retraction_payload(
    mapper: Mapper[Any],  # noqa: ARG001 — MapperEvents API signature
    connection: Connection,  # noqa: ARG001 — MapperEvents API signature
    target: object,
) -> None:
    """Blank the payload of any retraction being inserted through the ORM.

    Registered on the :class:`~sqlalchemy.orm.Mapper` *class*, so it covers
    every mapper in the process and cannot be dodged by mapping a fact table
    somewhere new; non-bitemporal targets and ordinary observations return
    immediately.

    Each payload column is set to SQL ``NULL`` explicitly
    (:func:`sqlalchemy.sql.null`) rather than to Python ``None``. The
    difference is load-bearing: for a column carrying a Python or server
    default — ``macro_observation.is_missing`` defaults to ``false`` — an
    attribute left at ``None`` makes SQLAlchemy *omit* the column from the
    INSERT and the default fills it in, which is precisely the fabricated
    value this task exists to remove. An explicit ``null()`` is sent as
    ``NULL`` and no default applies.

    Why clear rather than refuse: the payload of a retraction is defined as
    carrying no information (module docstring), so dropping it destroys
    nothing, whereas raising would break every writer that predates the rule —
    including two large property suites — for no gain in honesty. It is not
    silent, though: every column that actually held a value is named in a
    warning, so a caller who set ``is_retraction`` by mistake and lost a real
    payload sees it in the log rather than wondering where the row went. The
    canonicalization has a precedent one screen up, where an explicit
    ``9999-12-31T23:59:59.999999+00`` is stored as PG ``'infinity'``.

    Writers that bypass the ORM (Core ``insert()``, ``COPY``) never reach this
    hook and are refused outright by ``ck_<table>_retraction_payload_absent``.
    """
    if not isinstance(target, BitemporalMixin) or not target.is_retraction:
        return
    dropped = [
        name
        for name in type(target).__bitemporal_payload__
        if _carries_value(getattr(target, name, None))
    ]
    for name in type(target).__bitemporal_payload__:
        setattr(target, name, null())
    if dropped:
        _logger.warning(
            "retraction_payload_cleared",
            table=cast(Table, cast(Any, type(target)).__table__).name,
            columns=dropped,
            reason=(
                "a retraction records that a fact is no longer believed and carries no "
                "payload; values supplied for these columns were not written (P2.10, I3)"
            ),
        )


def bitemporal_classes() -> tuple[type[BitemporalMixin], ...]:
    """Return every registered bitemporal ORM class, in definition order."""
    return tuple(_registry)


def bitemporal_mappers() -> tuple[Mapper[Any], ...]:
    """Return the SQLAlchemy mappers of every registered bitemporal class."""
    return tuple(class_mapper(cls) for cls in _registry)


def bitemporal_tables() -> frozenset[Table]:
    """Return the set of tables backing bitemporal mappers.

    The as-of query layer uses this to decide whether a statement touches
    versioned data and therefore must carry a bound as-of timestamp.
    """
    return frozenset(cast(Table, mapper.local_table) for mapper in bitemporal_mappers())
