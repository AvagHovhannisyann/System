#!/bin/sh
# Apply migrations, then serve. Single-writer deployment makes start-time
# migration race-free (DECISIONS.md D-009). Venv binaries are invoked directly:
# the environment is fully synced at image build time, and uv itself would need
# a writable cache dir the non-root runtime user does not have.
set -e
/app/.venv/bin/alembic upgrade head
exec /app/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
