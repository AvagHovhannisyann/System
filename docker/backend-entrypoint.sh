#!/bin/sh
# Apply migrations, then serve. Single-writer deployment makes start-time
# migration race-free (DECISIONS.md D-009).
set -e
uv run --no-sync alembic upgrade head
exec uv run --no-sync uvicorn backend.main:app --host 0.0.0.0 --port 8000
