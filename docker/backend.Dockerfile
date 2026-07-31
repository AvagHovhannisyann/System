# Backend image: uv-managed Python 3.12, runs Alembic migrations then uvicorn.
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer first so code changes don't bust the dep cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# README.md is project metadata (pyproject readme field) — hatchling needs it
# to build the package in the next sync step.
COPY alembic.ini README.md ./
COPY backend ./backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY docker/backend-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && groupadd -r app && useradd -r -g app app \
    && chown -R app:app /app
USER app

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
