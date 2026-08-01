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

## D-012 — Phase 2 implementation amendments to D-011 (2026-07-31)

D-011 was written before implementation. Four things changed on contact with the
code; each is recorded here so the design doc and the code cannot silently diverge.

**1. Append-only is enforced by triggers, not role grants.** D-011 specified "no
UPDATE/DELETE grants for the application role" plus "Alembic migrations run as a
separate role". The compose stack runs a single `POSTGRES_USER`, and migrations run
in the backend entrypoint as that same role (D-009), so grants would bind nothing.
Migration 0003 (fact tables) and 0004 (identity anchor) install `BEFORE UPDATE OR
DELETE` row triggers instead: role-independent, and they propagate to Timescale
chunks so direct-chunk mutation is refused too. **Residual weakness, stated plainly:**
the single owning role can `ALTER TABLE ... DISABLE TRIGGER` or `DROP TRIGGER`, which
true role separation would have prevented. Closing it requires a second DB role — a
deployment change, logged as a backlog item rather than done silently here.

**2. Enforcement is two-surface, not one.** D-011's layer 2 assumed the ORM
`do_orm_execute` hook sufficed. An adversarial review proved it does not: raw Core
access via `session.connection()` and the publicly exported admin engine read fact
tables unversioned with no error, and SELECTs embedded in INSERT/UPDATE statements
skipped the hook entirely (the exact "materialize features FROM price_bar" pattern
Phases 4–6 will write). A **Core-level guard** now runs on every engine `backend.db`
creates, vetting compiled statements and textual SQL. Sanctioned executions carry a
module-private token object compared by identity — the option *name* is public and
useless without the token. Migrations and test resets run on a module-private
unguarded engine, itself banned from import outside `backend/db`.

**3. Fail-closed beats analyze-harder.** Textual SQL cannot be structurally analyzed,
so it is name-scanned with a conservative word-boundary regex and *rejected* on
match. This has known false positives — a column literal spelling a fact-table name,
or `TRUNCATE price_bar` — which are accepted deliberately: the failure mode is a loud
error, never a silent leak. The same principle governs aliased entities and compound
selects, which raise rather than execute unversioned.

**4. Temporal values are validated aware at the boundary.** Not in D-011, added
because asyncpg silently reinterprets naive datetimes in *host-local* time — a
silent hours-scale I1 violation invisible to every test. Writer-supplied temporal
values must be timezone-aware or raise before I/O, and PG `'infinity'` round-trips
through an aware sentinel instead of leaking the naive `datetime.max` asyncpg
returns (which raises `TypeError` on any aware comparison).

## D-013 — Paper fills are a lower bound on slippage, never an estimate (2026-07-31)

**Context.** Directive §5-P9 says the cost model is "calibrated against paper fills once
Phase 11 exists". Operator correction: IBKR paper fills are **optimistic** — they fill at
the touch far more readily than reality and model **no queue position**.

**Decision.** P11.8 does not calibrate the cost model directly to paper fills. Realized
paper slippage is treated as a **lower bound**; the fitted parameters carry a documented
haircut, and any backtest using them displays the calibration basis. The haircut's
magnitude and derivation are recorded here when P11.8 lands.

**Reasoning.** Calibrating directly to optimistic fills understates costs, and
understated costs are the single easiest way to turn a losing strategy into a winning
backtest — the exact failure mode I4 and the whole validation framework exist to
prevent. A cost model that is wrong in the *conservative* direction loses money on
paper; one wrong in the optimistic direction loses it for real.

**Rejected.** (a) Calibrate to paper fills and note the caveat in prose — the number
would still flow into every reported Sharpe. (b) Wait for live fills — out of scope
forever (§1.1), so the lower-bound treatment is permanent, not transitional.

## D-014 — Golden-set threshold is set relative to the human noise floor (2026-07-31)

**Context.** Directive §5-P7 sets the Gate G7 bar at "golden-set agreement ≥ 85%".
Operator correction: that number is unjustifiable until the labeller's own **intra-rater
agreement** is measured.

**Decision.** Measure the noise floor first (B3 protocol: 60-document pilot → rubric
fix → 20 blind re-labels a week later → intra-rater agreement). The model gate is then
set **relative to that floor**. If self-agreement is 82%, an 85% model gate is measuring
noise, not capability, and the reported figure would be meaningless precision.

**Reasoning.** An agreement threshold above the measurement instrument's own
reproducibility cannot be satisfied by any model, and a threshold near it is measuring
label noise. Reporting "87% agreement" against an unmeasured floor is the kind of
false precision this project exists to avoid. The floor also bounds what any downstream
extraction-quality drift signal (P12.3) can legitimately claim to detect.

**Consequence.** Gate G7's numeric bar is *derived*, not literal, and the derivation is
recorded in `TESTING_LEDGER.md` alongside the floor measurement. The directive's 85% is
treated as the operator's prior, superseded by measurement.

## D-015 — LLM backfill is gated on proven incremental IC (2026-07-31)

**Context.** Steady-state extraction is ~60k calls/year (negligible). The historical
backfill is ~600k calls, order **$1,000–1,500**, and is the only material LLM cost.

**Decision.** No full backfill before **Phase 8 demonstrates incremental IC** over the
Phase 5 baseline factors. The gating experiment is a pilot at **200 names, 5 years,
2-model ensemble**; its result is a `TESTING_LEDGER.md` row like any other trial.
Cost levers, in order of preference: universe size, history depth, ensemble width.

**Reasoning.** A negative pilot and a negative full backfill teach exactly the same
thing at a fraction of the cost. Spending first would also create sunk-cost pressure to
find the LLM features useful — precisely the bias the validation framework is built to
resist.

**Ledger note.** The pilot is a trial and counts toward the Deflated Sharpe trial count
(§9.7). Running it and *not* logging it because it was "just a pilot" is exactly the
omission that makes DSR dishonest.

## D-016 — P2.9 (pushdown verification) runs at first real data volume (2026-07-31)

**Context.** P2.5 promised an EXPLAIN sanity check; it was not done. The versioned
subquery carries no entity/event-time quals, so index use and hypertable chunk pruning
depend on Postgres pushing outer quals down through two subquery layers past
`DISTINCT ON` — safe only for quals on the distinct-on columns.

**Decision.** Run P2.9 **the moment the first hypertable holds meaningful volume** —
not now (an empty table's plan proves nothing), and **not after Phase 4**.

**Reasoning.** This is a performance gate with **design consequences**, not a loose end.
If predicates do not reach chunk exclusion, every as-of query full-scans the hypertable;
at Phase 6 volumes — daily bars × universe × a decade, with triple-barrier labels
resolved per name per date — that is the difference between a backtest in minutes and
one in days. Critically, **if the fix requires changing how predicates are injected, that
must be known before Phases 4–6 write against the current shape** — otherwise the
rewrite lands after there is code depending on it. Empty-table EXPLAIN output cannot
detect any of this, which is why it waits for volume rather than running now.

## D-017 — Deadline for role separation: before the audit log is trusted (2026-07-31)

**Context.** D-012 records that append-only is enforced by triggers under a single owning
role, which can `DISABLE`/`DROP` its own triggers, and backlogged true role separation.
A backlog item with no deadline is a deferral without end.

**Decision.** Role separation ships **before Phase 11**, and specifically **before the
§6.11 immutable audit log is presented as trustworthy**.

**Reasoning.** "Immutable audit log" is a **false claim** while the owning role can drop
the triggers enforcing immutability — and the dashboard states it as fact to the
operator. The same applies to `TESTING_LEDGER.md` integrity, which is what makes the
Deflated Sharpe honest: a ledger that can be silently edited makes the trial count
unverifiable, and DSR is only as honest as its trial count (§9.7). The deadline is
Phase 11 — not "before real capital", because there is no real capital in this system by
design (§1.1); the binding constraint is the point at which the platform starts making
integrity claims to its operator.

## D-018 — The as-of *value* is verified at the boundary, not trusted (P2.11) (2026-08-01)

**Amends D-012, which listed four mechanisms. This is the fifth, and it exists because
the first four were all satisfied while an I1 leak ran in production shape.**

**Context.** D-012's mechanisms all guarantee properties of the emitted **SQL**: that
every fact-table reference sits inside the versioned form, that nothing unsanctioned
reaches the driver. The depth suite (P2.12) found a leak where **the SQL was entirely
correct and only the bind value was wrong**: SQLAlchemy pairs cache-key binds to compiled
binds by `.key` across the clone lineage, and `BindParameter._clone` regenerates the key
of *anonymous* binds — which `sa.literal()` produces. A second clause-adapter pass over
an already-substituted tree therefore produced a compiled bind whose key lineage was
disjoint from the cache key; `construct_params` fell back to the value frozen at first
compile. Re-executing one statement at a second as-of instant read the store **as of the
first**. Structural verification cannot see this class of defect at all.

**Decision.** Two additions, both fail-closed:
1. **One bind identity.** Every versioned subquery in a rewrite shares a single
   explicitly-keyed, non-unique as-of bindparam. An explicit key survives `_clone`, so
   all copies collapse to one parameter. A post-rewrite invariant asserts exactly one
   distinct as-of bind key carrying the session's value — a regression to anonymous
   per-table binds is *detected*, not tolerated.
2. **Boundary verification.** At the cursor boundary, the as-of values **actually being
   sent** are compared against ground truth carried with the sanction token, read from
   `context.compiled_parameters` (recomputed per execution, so a cache hit's stale value
   is visible). Mismatch, expectation-without-bind, bind-without-expectation, and a
   context exposing no resolved parameters all raise `AsOfBindIntegrityError`.

**Reasoning.** Same principle as the default-deny inversion in D-012: *do not trust that
the rewrite produced the right thing — verify against ground truth at the boundary where
truth is observable.* The rewriter is a heuristic over clause trees interacting with a
compiled-statement cache neither we nor SQLAlchemy documents as bind-stable under
repeated adaptation. Verifying the value that actually reaches the driver makes the
whole class unshippable, rather than patching the one instance found.

**Rejected.** `execution_options={'compiled_cache': None}` — the obvious "fix". It works,
and it is wrong: measured **+1.75 ms (+16.6 %) per as-of query, permanently**, roughly
six times the cost of the verification checks (~0.3 ms against a 5.96 ms rewrite, below
the end-to-end noise floor), and it treats the symptom while leaving the bind-identity
defect in place for any future code path that rebuilds the cache.

**Testing note (why the regression test is not vacuous).** The test proving the backstop
fires reinstalls the pre-fix mechanism and asserts the second execution raises having
executed nothing. A companion test *disables* the backstop and asserts the leak actually
reappears — so if SQLAlchemy ever stops producing this behavior, the companion fails
loudly instead of the backstop test silently passing for the wrong reason.

**Standing lesson for later phases.** A property suite that randomizes *data* while
holding *query shape* fixed measures breadth, not correctness. 75,936 breadth cases
passed while this leak was live; it took nesting depth as a first-class strategy
dimension to reach the shape. Any future property suite (labels P6.4, optimizer P9.4,
cost model, backtest engine) must randomize the *structure* of what it exercises, not
only the values fed through one fixed structure.

## D-019 — API hardening: default-on CSRF, fail-closed rate limiting (CC.2) (2026-08-01)

Landed **before** the first mutating endpoint (P3.10's re-sync trigger), not retrofitted
onto one. Three choices worth recording, because each has a defensible opposite.

**1. CSRF enforcement is default-on, not opt-in.** A middleware rejects every unsafe
method unless the path carries an explicit, exact-match exemption (the exemption set is
currently empty; every future entry must carry a written justification).
**Reasoning:** a route that forgets an opt-in dependency is *silently unprotected* — the
failure is invisible until someone exploits it. A route that needs a missing exemption
fails loudly the first time it runs. Given a choice between a silent security failure
and a noisy functional one, take the noisy one. Tokens are seeded on any safe-method
response so there is no token endpoint to forget either.
**Accepted, consciously:** the CSRF cookie is **not** `HttpOnly`. That is inherent to
double-submit — the client must read the cookie to populate the header. The token
authenticates nothing on its own; it only proves same-origin script access. Flagged here
so it is a deliberate call rather than a discovery.
**Rejected.** Per-route opt-in dependency (silent-failure mode above); `SameSite` cookies
alone (no defense-in-depth, and browser-version dependent).

**2. Rate limiting fails CLOSED when Redis is unavailable — 503, not 429.**
**Reasoning, in the order it actually mattered:** safe methods are exempt, so the entire
dashboard *read* surface is unaffected by a Redis outage; Redis is already a D-005
critical component and the Celery broker, so the first consumer of these endpoints could
not have functioned during that outage anyway; the endpoints being throttled spend real
money (LLM caps, §6.5) and consume EDGAR fair-access quota, and retry storms happen
*precisely* during infrastructure failures; and this is a single-operator research tool
with no availability SLA to weigh against that. 503 rather than 429 because the client
did nothing wrong — an outage must not be disguised as throttling.
**Rejected.** Fail-open (accept-and-drop is a *silent* outage, the failure mode this
project rejects everywhere else); an env var to switch the behavior — `fail_open` exists
as a constructor argument for tests but deliberately not as configuration, on exactly the
reasoning D-008 used to reject `continue-on-error`: a safety control that can be turned
off from the environment will be, at the worst moment.
**Also:** `X-Forwarded-For` is deliberately not consulted when keying. Without a
trusted-proxy allowlist it is attacker-controlled and turns the limiter into decoration.

**3. Parameterized-query check tolerates interpolation only under
`migrations/versions/`, and only for DDL.** Identifiers have no bind-parameter form, and
revisions run offline with no untrusted input.
**Rejected.** A per-statement allowlist — it was written first, then replaced: a
concurrently-landing migration reused the same trigger DDL and would have broken the
build. An allowlist keyed to specific statements makes correct work fail.
**Residual risk, stated rather than hidden:** a *new* interpolated DDL shape in a future
revision passes unreviewed. Documented at the check itself.
