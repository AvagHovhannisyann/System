# Systematic Equity Research Platform (paper-trading only)

A research platform that ingests market/fundamental/textual data with strict
point-in-time integrity, extracts structured features from documents with LLMs,
ranks equities cross-sectionally with a regularized gradient-boosted model,
constructs cost-aware portfolios, validates results with overfitting-resistant
statistics, and executes **only against a paper account**. Operated through a
web dashboard.

**There is no live-trading code path in this repository, by design. This is not
configurable.** (See `DIRECTIVE.md` §1.1, §5-P11.)

## Quickstart

```bash
cp .env.example .env   # fill in values
docker compose up
```

- Backend API: http://localhost:8000 (health: `/api/health`)
- Dashboard: http://localhost:3000

## Development

```bash
# backend (requires uv, Python 3.12)
uv sync
uv run pytest
uv run mypy --strict backend
uv run ruff check .

# frontend (requires Node 22)
cd frontend && npm ci
npm run lint && npx tsc --noEmit && npm run build
```

## Project state files

| File | What it is |
|---|---|
| `DIRECTIVE.md` | The build directive: mission, invariants, phases, gates. Read first |
| `PLAN.md` | Task breakdown with IDs, dependencies, status |
| `PROGRESS.md` | Append-only work log |
| `DECISIONS.md` | Engineering decisions with reasoning |
| `BLOCKERS.md` | Items requiring a human (credentials, vendor choices, labels) |
| `TESTING_LEDGER.md` | Append-only record of every model/strategy configuration ever evaluated |
| `BACKLOG.md` | Out-of-scope ideas parked for human review |
