#!/bin/sh
# Apply migrations, then serve. Single-writer deployment makes start-time
# migration race-free (DECISIONS.md D-009). Venv binaries are invoked directly:
# the environment is fully synced at image build time, and uv itself would need
# a writable cache dir the non-root runtime user does not have.
#
# Two roles, and the split is enforced here rather than assumed (CC.9, D-017):
# alembic runs as the schema owner (MIGRATION_DATABASE_URL), the server runs as
# the application role (DATABASE_URL), and the owner credential is removed from
# the environment before the server is exec'd. A serving process that does not
# hold the owner credential cannot drop the append-only triggers even if
# something inside it tries.
set -e

: "${DATABASE_URL:?must be set — the application role's connection URL}"
: "${MIGRATION_DATABASE_URL:?must be set — the schema owner's connection URL; migrations do not run as the application role (CC.9)}"

if [ "$MIGRATION_DATABASE_URL" = "$DATABASE_URL" ]; then
    echo "refusing to start: MIGRATION_DATABASE_URL and DATABASE_URL name the same" >&2
    echo "credential, so the role the application connects as would own the schema and" >&2
    echo "could DISABLE or DROP the append-only triggers on itself. That makes the" >&2
    echo "immutable audit log a false claim (DECISIONS.md D-012, D-017)." >&2
    exit 1
fi

/app/.venv/bin/alembic upgrade head

# Past this line nothing may run DDL. Both the owner URL and the role-creation
# password are dropped from the environment; the server needs neither, and what
# a process does not hold it cannot leak or misuse (DIRECTIVE I5).
unset MIGRATION_DATABASE_URL
unset APP_DB_PASSWORD
unset POSTGRES_PASSWORD

exec /app/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
