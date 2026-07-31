"""Public surface of the database layer (D-011 bypass prevention, layer 1).

The only sanctioned handles on the database are re-exported here:

- :func:`backend.db.asof.as_of` — versioned reads (the only read path for
  bitemporal fact tables);
- :func:`backend.db.asof.ingest_writer_session` — the append-only ingestion
  write path;
- named admin helpers :func:`backend.db.engine.create_admin_engine` (health
  probes, migration tooling — never fact reads) and
  :func:`backend.db.engine.dispose_database` (shutdown / test isolation);
- the error types those paths raise (re-exported so callers can catch them;
  they grant no session capability).

The raw engine and sessionmakers stay module-private in
:mod:`backend.db.engine`, which the import contract (ruff TID251) bans
outside ``backend/db``. Importing this package installs the class-level
``do_orm_execute`` enforcement hook (layer 2) as a side effect — any use of
the ORM models triggers it, so app-constructed sessions cannot dodge it.
"""

from backend.db.asof import (
    AsOfTimestampError,
    BitemporalBypassError,
    BitemporalRewriteError,
    as_of,
    ingest_writer_session,
)
from backend.db.engine import create_admin_engine, dispose_database

__all__ = [
    "AsOfTimestampError",
    "BitemporalBypassError",
    "BitemporalRewriteError",
    "as_of",
    "create_admin_engine",
    "dispose_database",
    "ingest_writer_session",
]
