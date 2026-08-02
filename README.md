# Systematic Equity Research Platform (paper-trading only)

A research platform that ingests market, fundamental and textual data with strict
point-in-time integrity, extracts structured features from documents with LLMs,
ranks equities cross-sectionally with a regularized gradient-boosted model,
constructs cost-aware portfolios, validates results with overfitting-resistant
statistics, and executes **only against a paper account**. Operated through a web
dashboard.

**There is no live-trading code path in this repository, by design. This is not
configurable.** (`DIRECTIVE.md` §1.1, §5-P11.)

## Quickstart

```bash
cp .env.example .env   # fill in values — see the comments in that file
docker compose up
```

- Backend API: http://localhost:8000 (health: `/api/health`)
- Dashboard: http://localhost:3000
- MLflow tracking UI: http://localhost:5000

`SEC_USER_AGENT` is required before the EDGAR connector will run — SEC's
fair-access policy requires a contact string, and the connector refuses rather
than sending an anonymous or invented one.

### Experiment tracking

The `mlflow` service is the store every run's reproducibility stamp is written
to (invariant I2). Its SQLite backing store and its artifacts both live on the
`mlflowdata` named volume, so history survives `docker compose down` and any
container replacement — `docker compose down -v` deletes that volume and with
it every recorded run. The server image is pinned to the same MLflow version
`uv.lock` resolves the client to, because the two share a store schema.

The backend container is given `MLFLOW_TRACKING_URI=http://mlflow:5000`, the
variable the MLflow client reads for its default tracking URI.
`backend.tracking.mlflow_run.tracked_run` still takes the URI as an explicit
argument — it never falls back to a default — so a caller inside the container
passes `mlflow.get_tracking_uri()`, which resolves to that address.

Tests and local scripts do **not** need this service:
`backend.tracking.mlflow_run.local_tracking_store` builds a serverless store
under any directory.

## Development

```bash
# backend (uv, Python 3.12) — run from the repository root
uv sync
uv run pytest --runxfail          # xfail is treated as failure (invariant I6)
uv run mypy backend               # strict
uv run ruff check . && uv run ruff format --check .
uv run python scripts/check_no_fabrication.py   # invariant I3

# frontend (Node 22)
cd frontend && npm ci
npm run lint && npx tsc --noEmit && npm run build
npm run test:ci                   # Vitest + coverage (gate: 70%)
npm run test:e2e                  # Playwright
```

Integration tests start real PostgreSQL/TimescaleDB and Redis containers via
Testcontainers, so a working Docker daemon is required. Tests are never skipped
when infrastructure is missing — they fail, per I6.

## Data versioning (DVC)

DVC is initialised at the repository root: `.dvc/config`, `.dvc/.gitignore` and
`.dvcignore` are committed, the content cache and `.dvc/config.local` are not.
It is what keeps data out of git while still producing the **data version** that
I2 demands next to the git commit, config hash and seed.

`dvc` is not yet a declared project dependency, so `uv sync` will not install
it. Install it into the environment you run it from, then version the data
directory:

```bash
uv tool install dvc            # or: pipx install dvc

dvc add data                   # writes data.dvc — a content hash, not the bytes
                               # and appends /data to .gitignore
git add data.dvc .gitignore    # commit the pointer. Never commit data/ itself
```

Record that hash so every run stamps the snapshot it actually read — either per
process, or in the committed `.data-version` pointer:

```bash
export DATA_VERSION=$(uv run python -c \
  "import yaml,pathlib;print(yaml.safe_load(pathlib.Path('data.dvc').read_text())['outs'][0]['md5'])")

uv run python -c "from backend.tracking.data_version import default_pointer_path, \
write_data_version; write_data_version(default_pointer_path(), '$DATA_VERSION')"
```

`backend.tracking.data_version.resolve_data_version` reads `DATA_VERSION` first
and the pointer file second. There is no third fallback: with neither present it
raises, because a result whose data version is unknown is not reproducible and
recording a guess would hide that.

No DVC remote is configured yet, so `dvc push` / `dvc pull` are unavailable and
the cache is local only — the storage location and its credentials are a human
decision (credentials go in `.dvc/config.local`, which is gitignored, never in
`.dvc/config`, which is tracked).

## What exists today

| Area | State |
|---|---|
| Foundation (compose, config, JSON logging, health, CI) | **Gate G1 passed** |
| Bitemporal store (`as_of()`, purging bypass prevention, hypertables) | **Gate G2 passed** |
| Ingestion framework + SEC EDGAR connector + data-quality report | Built |
| Purged K-fold CV with embargo | Built |
| Ledoit-Wolf covariance, transaction cost model | Built |
| CPCV, Deflated Sharpe, PBO, synthetic-truth harness | Built |
| LLM anonymization + leak detector | Built |
| Crypto/redaction, CSRF + rate limiting, audit log | Built |
| Universe, features, labels, predictor, execution, monitoring | Not yet — see `PLAN.md` |

Phases needing vendor market data are blocked on `BLOCKERS.md` B1; paper
execution on B2; the LLM golden set and provider keys on B3/B4.

## Reading order for a new session

1. **`DIRECTIVE.md`** — mission, the six invariants, the twelve phases, and the
   binary gate conditions. §9 (forbidden behaviors) is meant to be re-read every
   session.
2. **`PLAN.md`** — every task with an ID, dependencies and status.
3. **`PROGRESS.md`** — append-only log of what was done, what was verified, and
   what failed.
4. **`BLOCKERS.md`** — what needs a human. Nothing here is guessed past.
5. **`DECISIONS.md`** — every non-obvious decision with its reasoning and the
   alternatives that were rejected.
6. **`TESTING_LEDGER.md`** — append-only record of every model and strategy
   configuration ever evaluated. Never edited, never pruned: the trial count is
   what makes the Deflated Sharpe honest, so deleting a failed experiment would
   silently inflate every subsequent result.

## The invariants everything else serves

- **I1 temporal integrity** — no query returns a fact whose `knowledge_time` is
  later than the query's `as_of`. Enforced structurally, not by convention.
- **I2 reproducibility** — every result regenerable from git commit + data
  version + config hash + seed.
- **I3 no fabricated data** — an unavailable source raises; it never returns a
  plausible value. A missing connector is a blocker, not an excuse for a mock.
- **I4 cost realism** — no result is ever reported gross.
- **I5 secret isolation** — keys encrypted at rest, never logged, never returned
  in full by any endpoint.
- **I6 test honesty** — a skipped, xfailed or weakened test counts as a failure.

The hardest part of this project is not making a strategy look profitable. It is
building infrastructure rigorous enough that you know when it isn't.
