# DECISIONS.md — Engineering decision log

Append-only. Every non-obvious decision: context, decision, reasoning, rejected alternatives.

---

## D-001 — Repository root is the platform root (2026-07-31)

**Context.** Directive §4 names the tree `quant-research-platform/`; the actual GitHub repository is `AvagHovhannisyann/System`.
**Decision.** The repository root plays the role of `quant-research-platform/`. Layout inside the repo matches §4 exactly.
**Rejected.** Nested `quant-research-platform/` subdirectory — adds a pointless path segment to every command and breaks the "docker-compose.yml at root" expectation of a clean-clone `docker compose up`.

## D-002 — Python packaging: root `pyproject.toml`, single top-level package `backend` (2026-07-31)

**Context.** Directive §4 shows modules directly under `backend/` (`backend/core`, `backend/db`, …) but does not show where `pyproject.toml` lives.
**Decision.** One uv project at the repo root; `backend/` is itself the (single) importable package: `from backend.core.config import Settings`. Tests live in `backend/tests` per §4.
**Reasoning.** Matches the visible layout byte-for-byte; one lockfile; no generic top-level package names; docker build context = repo root works for both images; mypy/pytest configured once.
**Rejected.** (a) `backend/pyproject.toml` with flat multi-package layout (`core`, `db`, … as top-level packages) — generic top-level names invite collisions and confuse mypy namespace handling. (b) `backend/app/...` src-style layout — deviates from the prescribed tree.

## D-003 — structlog for JSON logging (2026-07-31)

**Context.** Directive requires structured JSON logging with correlation IDs (§5 Phase 1) and automatic redaction of key patterns (§7).
**Decision.** `structlog` with contextvars-based binding; ASGI middleware generates/propagates `X-Request-ID` and binds it; a custom processor chain emits JSON and applies secret-pattern redaction (redaction processor lands with CC.0).
**Rejected.** (a) stdlib `logging` + `python-json-logger` — no first-class context binding; correlation IDs end up threaded manually. (b) `loguru` — weaker typing story under `mypy --strict`.

## D-004 — TimescaleDB image pinned by digest-stable tag (2026-07-31)

**Decision.** `timescale/timescaledb:2.17.2-pg16` in `docker-compose.yml` (pin verified by pulling at G1 time; if the tag is unavailable the pin is corrected to the nearest verified 2.x-pg16 tag and this entry updated — never `latest`).
**Reasoning.** Reproducibility (I2) starts with pinned infrastructure.

## D-005 — Health endpoint semantics (2026-07-31)

**Decision.** `GET /api/health` returns per-component status (`db`, `redis`) with latency; HTTP 200 when all critical components are up, 503 with the same body when any is down. Compose healthcheck consumes it.
**Reasoning.** Operators need per-component truth (dashboard §6.1 health tiles); a bare 200 hides degradation, and lying about health violates the project's honesty ethos.
**Rejected.** Always-200 "liveness only" — hides a dead DB; separate live/ready endpoints — unneeded complexity for a single-process API at this stage (revisit if k8s ever appears, which is out of scope).

## D-006 — Visual design is delegated; this codebase ships functional UI only (2026-07-31)

**Context.** Project `CLAUDE.md` mandates: website/UI *design* tasks (mockups, visual styling direction, aesthetics) are handled by Fabel, not by this engineering effort.
**Decision.** All dashboard work here is functional: routes, data fetching, tables, charts wired to real endpoints, using stock shadcn/ui + Tailwind defaults. No bespoke visual design, no styling guidelines authored here. When a task would require design direction, it is built with default components and flagged for Fabel in `BACKLOG.md`.

## D-007 — Build directive committed as `DIRECTIVE.md` (2026-07-31)

**Reasoning.** Sessions start with zero context; §9 must be re-read every session and gates/invariants must never drift by paraphrase. The state files reference it instead of restating it.

## D-008 — Dependency scanning is a blocking CI job (2026-07-31)

**Context.** §7 requires dependency scanning in CI; scanners sometimes fail on new upstream advisories unrelated to the change being pushed.
**Decision.** `pip-audit` + `npm audit --audit-level=high` run as a blocking job. When an advisory fires, it is triaged in the PR that hits it (upgrade, or a documented ignore with justification committed alongside).
**Rejected.** `continue-on-error` — a scanner that cannot fail is decoration, same class of dishonesty as a weakened test (I6).

## D-009 — Migrations run on backend container start (2026-07-31)

**Decision.** Backend image entrypoint runs `alembic upgrade head` before `uvicorn`. Single-writer deployment (one backend container) makes start-time migration race-free.
**Rejected.** Separate migration job/service — right answer for multi-replica production, needless moving part for a single-operator research stack; revisit only if the deployment story changes.

## D-010 — Frontend dependency audit via audit-ci with a committed allowlist (2026-07-31)

**Context.** D-008 makes dependency scanning blocking. On day one, `npm audit --audit-level=high` fails on 12 transitive advisories: postcss (≤8.5.17) and sharp (<0.35.0) are *bundled inside next itself* — the advisory ranges cover every published next release, so no upgrade fixes them — plus a brace-expansion DoS reachable only through the eslint lint-time chain. npm's only offered "fixes" are breaking downgrades (`next@9.3.3`, `eslint-config-next@12`), and raw `npm audit` has no ignore mechanism.
**Decision.** CI runs `audit-ci --config frontend/audit-ci.jsonc`: fails on any high/critical advisory *not* in the committed allowlist; each allowlisted GHSA carries a written justification and a removal condition in the config itself. Current entries: GHSA-mh99-v99m-4gvg (brace-expansion, dev-time only), GHSA-6g55-p6wh-862q + GHSA-r28c-9q8g-f849 (postcss, build-time on first-party CSS only), GHSA-f88m-g3jw-g9cj (sharp/libvips, no untrusted image processing in this app).
**Rejected.** (a) `continue-on-error` on the audit job — decoration, violates D-008. (b) npm `overrides` — cannot reach dependencies *bundled* inside next's package. (c) Downgrades npm proposes — strictly worse security posture.

## D-011 — Bitemporal store design (P2.1) (2026-07-31)

The design that Phase 2 implements. Invariant I1 (no query returns a fact whose
`knowledge_time` exceeds the query's `as_of`) is the property everything below serves.

### Column semantics

Every fact table carries exactly three temporal columns, all `TIMESTAMPTZ`, all UTC:

- `valid_from` / `valid_to` — the **event-time** interval the fact describes, half-open
  `[valid_from, valid_to)`. A daily bar for trading day *D* has `valid_from = D 00:00Z`,
  `valid_to = D+1 00:00Z`. A fundamental applies from its fiscal-period end until
  superseded by the next period (`valid_to = NULL` ⇒ open-ended, represented as
  `'infinity'::timestamptz` so range predicates stay uniform). `valid_from < valid_to`
  is a CHECK constraint.
- `knowledge_time` — when the information **became knowable to the market**, per the
  source: EDGAR acceptance timestamp; vendor-provided availability timestamp; for
  sources that give only a date (e.g. a fundamentals report date with no intraday
  time), a **documented conservative lag** is applied per source (default: next trading
  day 00:00Z after the report date — never the report date itself, because same-day
  availability at the open cannot be assumed). Each connector declares its
  `knowledge_time` policy in code and the DQ report displays it.

Separately, `ingested_at` (audit only, never used in queries): when our pipeline wrote
the row. Kept distinct because backfilled history has `ingested_at = today` but honest
historical `knowledge_time` — conflating them would make all backfilled data invisible
to every historical `as_of` and destroy backtesting; using ingestion time as knowledge
time is only correct for live (non-backfill) flow, where the two roughly coincide.

### Corrections and restatements

Rows are **never updated or deleted**. A correction/restatement is a new row for the
same logical key and valid interval with a later `knowledge_time`. A retraction is a
new row with `is_retraction = true`. Read semantics: for each (logical key,
valid interval), the visible version at `as_of` is the row with the greatest
`knowledge_time <= as_of`; if that row is a retraction, the fact is invisible. This
yields exactly the honest behavior: a backtest dated before a restatement sees the
original (wrong-but-then-believed) number; one dated after sees the restated one.
Unrestated fundamentals (directive P3 requirement) fall out naturally: query with
`as_of` shortly after the original release.

### Query layer (`as_of`)

`as_of(session_factory, as_of_ts) -> AsyncSession` produces a session with the as-of
timestamp bound; a SQLAlchemy `do_orm_execute` hook rewrites every SELECT that touches
a bitemporal mapper to the versioned form (`knowledge_time <= :as_of`, latest-version
wins via `DISTINCT ON (key, valid_from) ... ORDER BY knowledge_time DESC`, retractions
masked). `as_of_ts` may not be in the future (guard: `as_of <= now()`); "current view"
is simply `as_of(now())` — there is no separate unversioned read path.

Write path: live connectors let the DB stamp nothing — `knowledge_time` is always
supplied explicitly by the connector under its declared policy; a CHECK forbids
`knowledge_time > ingested_at + skew_allowance` for live flow is NOT imposed (backfills
legitimately violate it); instead the ingestion-run record marks each run
`backfill | live`, and a DQ check flags live runs whose knowledge_times trail
ingestion by more than the source's declared lag.

### Bypass prevention (P2.4) — three layers, tested independently

1. **Session-factory guard:** the only exported way to obtain a read session is
   `as_of(...)`; the raw `async_sessionmaker` is module-private to `backend/db`.
   Writer sessions (ingest) come from a distinct factory that permits INSERT/COPY but
   installs the same hook to reject un-versioned SELECTs on bitemporal tables.
2. **Runtime assertion:** the `do_orm_execute` hook raises `BitemporalBypassError` if
   a statement referencing a bitemporal table executes on a session with no bound
   as-of timestamp (defense against sessions constructed by future code paths).
3. **Import contract:** lint rule (import-linter or ruff ban) forbidding
   `backend.db.engine` / raw sessionmaker imports outside `backend/db`.
The P2.7 test exercises all three: direct sessionmaker use fails at import-contract
CI; a hand-built session raises at runtime; `as_of` sessions pass.

### Physical layout

- Hypertables partition on **`valid_from`** (the event-time axis). Reasoning: research
  queries always bound event time (a training window, a rebalance date's lookback), so
  chunk pruning bites on every query; `knowledge_time <= as_of` predicates are
  half-unbounded and prune poorly as a partition key (they only exclude
  future-of-as_of chunks, which the `valid_from` bound usually excludes anyway).
- Composite index per fact table: `(entity_id, valid_from, knowledge_time DESC)` —
  matches the `DISTINCT ON` + order exactly; secondary partial index on retractions if
  they prove common (deferred until data shows need).
- No UPDATE/DELETE grants for the application role on fact tables (append-only
  enforced in the DB, not just convention); Alembic migrations run as a separate role.

### Rejected alternatives

- **Full SQL:2011-style transaction-time intervals (`tx_from`/`tx_to`) per row** —
  classic bitemporal texts use them to support physical deletion visibility; we never
  physically delete, so a single `knowledge_time` (assertion start) plus retraction
  rows carries the same information with half the update surface and no closing of
  intervals on supersede (which would require UPDATEs, breaking append-only).
- **Postgres temporal extensions / system versioning** — not composable with
  TimescaleDB hypertables and hides `knowledge_time` semantics (system time = commit
  time = our `ingested_at`, which is exactly the wrong axis for backfills).
- **Application-level filtering by convention** (every query author remembers the
  predicate) — this is how lookahead bias actually happens; rejected on I1.
- **Partitioning on `knowledge_time`** — see Physical layout.
- **Views per as-of date** — unbounded view proliferation, and dynamic as-of
  parameters don't fit static views.
