"""Every generated database identifier must fit PostgreSQL's 63-byte limit.

This file exists because of a defect that cost a full CI run and 487 errors.

``monitoring_halt_event`` carries a **self-referential** foreign key, and the
project's naming convention renders a foreign key as
``fk_<table>_<column>_<referred table>``. When the table refers to itself its
name appears twice, and the result was 69 characters against PostgreSQL's
63-byte ``NAMEDATALEN`` limit.

The failure mode is what makes this worth a dedicated guard. SQLAlchemy raises
``IdentifierError`` while *rendering the metadata*, not while touching the
offending table — so a single over-long name in one new column took down every
test that touches ``Base.metadata``. The 487 errors named
``test_universe_db``, ``test_edgar_ingestion``, ``test_asof_layer`` and dozens
of other modules that have nothing to do with monitoring, which is a
maximally misleading signal to debug from.

It is also invisible without a database. Nothing in the unit suite renders
DDL, so ``ruff``, ``mypy --strict`` and 4,100 passing tests all agreed the
model was fine. Checking the metadata directly costs nothing and runs
everywhere — which is the whole point, because the environments where this bug
hides are the ones without a Docker daemon.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

import backend.db.models  # noqa: F401 — imported for its side effect: populating the metadata
from backend.db.base import Base

MAX_IDENTIFIER_BYTES: Final = 63
"""PostgreSQL's ``NAMEDATALEN - 1``. Longer names are truncated or refused."""


def _named_objects() -> list[tuple[str, str, str]]:
    """Return ``(kind, table, identifier)`` for everything the metadata names."""
    found: list[tuple[str, str, str]] = []
    for table in Base.metadata.tables.values():
        found.append(("table", table.name, table.name))
        found += [("column", table.name, column.name) for column in table.columns]
        for constraint in table.constraints:
            if constraint.name is not None and not isinstance(constraint.name, str):
                continue  # a sentinel placeholder, resolved at DDL time
            if isinstance(
                constraint, ForeignKeyConstraint | UniqueConstraint | CheckConstraint
            ) and isinstance(constraint.name, str):
                found.append((type(constraint).__name__, table.name, constraint.name))
        found += [
            ("index", table.name, index.name) for index in table.indexes if index.name is not None
        ]
    return found


def test_no_identifier_exceeds_the_postgres_limit() -> None:
    """A name over 63 bytes fails metadata rendering, taking every table with it.

    Measured in **bytes**, not characters, because the limit is
    ``NAMEDATALEN`` in bytes and identifiers are UTF-8.
    """
    over_limit = [
        (kind, table, name)
        for kind, table, name in _named_objects()
        if len(name.encode()) > MAX_IDENTIFIER_BYTES
    ]
    assert over_limit == [], (
        "these identifiers exceed PostgreSQL's 63-byte limit; SQLAlchemy raises "
        "IdentifierError while rendering the metadata, which fails every test that "
        f"touches Base.metadata rather than only the offending table: {over_limit}"
    )


def test_the_scan_reaches_the_self_referential_foreign_key_that_caused_this() -> None:
    """Non-vacuity, pinned to the actual defect rather than to a synthetic one.

    The convention would have generated
    ``fk_monitoring_halt_event_resolves_halt_event_id_monitoring_halt_event``
    (69 bytes) for this key. It is named explicitly instead, and this asserts
    both that the scan sees it and that the name it sees is within the limit —
    so reverting to the convention here fails, and a scan that silently stopped
    finding foreign keys fails too.
    """
    foreign_keys = [
        (table, name) for kind, table, name in _named_objects() if kind == "ForeignKeyConstraint"
    ]
    # This assertion has already paid for itself. Without the models import
    # above, `Base.metadata` is empty, so the length scan ran over nothing and
    # passed vacuously — a guard against a defect that had just cost 487 errors,
    # itself checking an empty set. The non-vacuity companion is what caught it.
    assert foreign_keys, "the scan found no foreign keys at all — it has stopped working"

    resolves = [
        name
        for table, name in foreign_keys
        if table == "monitoring_halt_event" and "resolv" in name
    ]
    assert resolves == ["fk_monitoring_halt_event_resolves"], resolves
    assert len(resolves[0].encode()) <= MAX_IDENTIFIER_BYTES


def test_the_limit_is_the_one_postgres_actually_enforces() -> None:
    """Guards the guard: a loosened constant would silently disarm the scan above."""
    assert MAX_IDENTIFIER_BYTES == 63
