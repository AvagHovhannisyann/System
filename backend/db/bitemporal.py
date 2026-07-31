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
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, cast

from sqlalchemy import Boolean, CheckConstraint, Index, PrimaryKeyConstraint, Table, text
from sqlalchemy.orm import Mapped, Mapper, class_mapper, declared_attr, mapped_column, validates
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime, TypeDecorator

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import Dialect

_registry: list[type[BitemporalMixin]] = []
"""Every mapped class carrying :class:`BitemporalMixin`, in definition order."""

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
            "interval (payload columns are semantically ignored) and hides the fact "
            "from any as_of at or after its knowledge_time."
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
