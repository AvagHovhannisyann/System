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

## D-020 — Covariance refuses rather than invents; cost defaults document their error direction (P9) (2026-08-01)

**1. Ledoit-Wolf raises below three observations instead of flooring the intensity.**
At `n = 2` the analytic shrinkage intensity is identically zero for *every* panel at every
width — demeaning makes the two rows exact negatives, so the error term the estimator
measures vanishes by construction (verified numerically for p ∈ {1,2,5,50}; scikit-learn
agrees, its unclamped value even going slightly negative). This is a degeneracy of the
formula, not of the data.
**Rejected — flooring the intensity:** it would report a shrinkage this code *invented*
rather than derived (forbidden behavior #2), and the intensity is precisely the number
that tells an operator how much of the risk model is assumption rather than data.
Corrupting it to avoid an error message trades a loud failure for a quiet lie.
**Rejected — warning and continuing:** the consumer is an optimizer, which answers a
near-singular covariance with an unbounded position along the null space. A warning in a
log does not stop that.
**Hole this closed:** the draft relied on a downstream singularity error firing
incidentally, but at `n_assets == 1` a two-observation panel is invertible — so nothing
raised and the caller silently received an unshrunk estimate reporting `intensity = 0.0`.
**Explicitly not claimed:** the minimum of 3 is where the *formula* stops being valid, not
where a risk model becomes trustworthy. The real minimum (60? 252?) is a policy that
belongs upstream where it is visible — logged as an open item.

**2. Constant panels are detected against the demeaning noise floor, not against zero.**
Demeaning a constant panel leaves residue around `eps · level` (~1e-38), never exact zero,
so a `<= 0.0` guard never fires and the panel falls through to a misleading singularity
error.

**3. A short position must state its holding period.** A short with no holding period
silently paid zero borrow. The draft documented that as deliberate; it is an
understatement of the short book in exactly the direction D-013 forbids. Intraday is out
of scope (§1.1), so a zero-day short is always a caller error and is now refused.

**4. Cost defaults are documented by the DIRECTION of their error, not presented as
accurate.** Flat half-spread overcharges megacaps roughly fivefold (conservative — it can
kill a viable large-cap strategy but cannot flatter a bad one); commission understates
low-priced names as bps rises when price falls; borrow understates hard-to-borrow names by
one to two orders of magnitude until per-name borrow data arrives from Phase 11 (B2).
Every result carries `uncalibrated=True` and a calibration basis so a backtest can declare
what its numbers rest on — G9 requires that flag be visible, not merely present.

## D-021 — Validation-framework decisions and two metric defects (P10) (2026-08-01)

**Trial count is a required argument.** `deflated_sharpe_ratio` refuses to default
`trials` or `trial_sharpe_variance`, pinned by a signature-introspection test that fails
if either ever acquires a default or becomes positional. An empty `TESTING_LEDGER.md`
yields zero trials and the DSR **refuses** rather than rounding to one. Reason: a DSR
computed as though one thing was tried when two hundred were is the single most
misleading number this system could emit (§9.7).

**Two defects found by the new tests, both silent:**
- `kurtosis` computed `m4 / variance**2`. For deviations near 1e-100 the squared variance
  underflows to exactly zero, so it raised *after* the zero-variance guard had already
  passed. Found by Hypothesis. Replaced with a scale-free formulation (rescale deviations
  onto [-1,1] before taking powers) — mathematically identical, immune to underflow and
  overflow.
- `sharpe_ratio` returned a silent `0.0` when the variance overflowed (`mean / inf`). It
  now refuses. A zero Sharpe reads as "no edge"; the truth was "this number is unusable".

**Kurtosis floor tolerance.** The guard `kurtosis < 1.0` rejected the *attainable*
minimum — a two-point symmetric series evaluates to 0.9999999999999999. Relaxed by 1e-9.
The mix-up the guard exists to catch (excess vs non-excess kurtosis) is off by ~3, so the
tolerance cannot mask it.

**Deferred, deliberately.** DSR assumes `N` independent trials; a ledger of near-duplicate
perturbations over-deflates. That is the conservative direction and is documented, but a
narrowly-rejected strategy deserves a look at trial correlation before it is discarded.
`deflated_sharpe_ratio_from_trials` cannot yet read trial Sharpes from the ledger — there
is no structured column, and the reader refuses to mine free text for one, because
guessing a grammar would silently understate the variance that makes deflation work.
Until P8.4 emits a structured trial Sharpe, callers supply that variance from elsewhere.

## D-022 — A broken CI workflow reports nothing, so its validity is now tested (2026-08-01)

**What happened.** A step name containing a colon-space (`I3: no mock…`) parses as a
nested YAML mapping. GitHub rejected the workflow before starting any job, reporting the
run as *failure with zero jobs* — which in the commit view looks like a build that has not
started rather than one that failed. It went unnoticed for fifteen commits, during which
CI results were reported as green when no workflow had run at all.
**Decision.** A test parses every workflow file and asserts it declares jobs with steps.
**Reasoning.** The failure is invisible exactly where people look for it, and no amount of
care while editing fixes a mode that produces no signal. The local gate (full suite, ruff,
`mypy --strict`) was run before each of those commits, so nothing shipped untested — but
"CI is green" was not a true statement, and the fix is a check rather than more diligence.

## D-023 — Label construction mathematics (P6) (2026-08-01)

Named by §0.3 as an extended-reasoning component. The choices below are all places
where a defensible-looking default would have quietly degraded every downstream model.

**Barrier width = `multiple × σ_trailing × √horizon`.** The textbook fixed-daily-σ barrier
is touched almost surely well before a 63-day deadline, which collapses the vertical class
and makes the 63d label a slower copy of the 5d one — three horizons that are nominally
different and substantively the same. `√H` scaling makes them comparable. Nothing is
lost: `upper_multiple = k/√H` recovers the daily convention exactly.

**The vertical barrier is its own class, not `sign(return)`.** Signing an unresolved
return labels a +0.05σ drift identically to a +2σ breakout — maximum noise precisely
where the path carried the least information. The realized return is retained so Phase 8
can take the sign if it chooses, but as a decision rather than a default.

**Both barriers pierced in one bar ⇒ `AMBIGUOUS`, flagged, never tie-broken.** Daily bars
do not contain the intrabar ordering; inferring it from the close direction fabricates the
missing fact (I3). Ambiguous rows are **not** folded into the vertical class — "we cannot
tell" is not "neither was touched". A `LOWER_FIRST` policy exists but is opt-in and named
for the assumption it makes, since "conservative" is only true for a long. **The ambiguity
rate is a diagnostic:** if it is material on real data, the barrier widths are wrong, not
the tie-break.

**Volatility: fixed 20-bar rolling sample std, full window required.** Fixed rather than
EWMA so "which bars produced this number" has a finite answer. Partial windows are
refused because a 3-observation and a 20-observation std are different statistics, and
mixing them makes early and late barriers incomparable *silently*.

**Residualization freezes trailing OLS coefficients and applies them forward**, with the
intercept **estimated but not subtracted**: omitting it biases the betas, while
subtracting a trailing alpha from future returns would remove the very idiosyncratic drift
the label exists to capture. Residual-path barriers are sized by trailing **residual**
volatility — sizing them by total volatility gives a mostly-market stock a wide barrier
that reads VERTICAL for reasons unrelated to itself.

**Two guards that exist because the failure is silent:**
- *Near-collinear factors raise*, judged on the **column-normalized** condition number so
  the threshold measures collinearity rather than the scale gap between an intercept
  column and a returns column. The residual is well defined under collinearity; the
  *coefficients* are not, and the coefficients are what get applied forward. A rank check
  alone would pass a near-duplicate sector.
- *Residual-volatility noise floor.* A stock that is an exact combination of its factors
  leaves residuals of order 1e-16 — a positive number that would size a barrier resolved
  by the sign of rounding error. This caught a real defect in the track's own fixtures.

**Uniqueness spans differ from `PurgedKFold`'s by half a bar, deliberately.** Purging
includes the event bar because purging one extra observation is conservative; uniqueness
must not, because there it would invent overlap and over-shrink the weights. Same-looking
interval, opposite direction of safety.

**Why this matters, measured:** on a constructed two-regime path, 518 labels at the 63d
horizon carry an effective sample size of **11.15** (ratio 0.0215, peak concurrency 63).
Training on 518 nominal observations as if they were independent would overstate the
evidence by roughly fortyfold.

**Not claimed:** G6 remains open. Regime invariance is proven on constructed paths, but
"label distribution sane across regimes" is a *measurement on real equities* and is
blocked on B1. Defaults (barrier multiples 1.0/1.0, 20-bar volatility, 252-bar estimation,
condition limit 1e8, noise-floor ratio 1e-8) are defensible, **not calibrated** — and
every variant tried in calibrating them belongs in `TESTING_LEDGER.md`, because they feed
the Deflated Sharpe trial count (§9.7).

## D-024 — CSRF token tampering: test exhaustively by position, document the tail equivalence class (2026-08-01)

**Trigger.** CI went red on `test_tampered_cookie_and_header_are_refused` — a *security*
test asserting a tampered CSRF token is refused, which instead saw `200`. The obvious
readings were "CSRF is broken" or "flaky, re-run it". Both are wrong.

**What is actually true.** The test took a genuine token and flipped its **last**
character to `"A"`. An HMAC-SHA1 signature is 20 bytes = 160 bits, base64-encoded into 27
characters = 162 bits; the two surplus bits are padding and Python's decoder ignores them.
So the four characters sharing the final character's top four bits **all decode to the
same signature** and all verify. Whenever the token happened to end in `"A"`, the test
flipped it to `"B"` — inside the same equivalence class — and the "tampered" token was
genuinely valid. Measured directly: **117 of 2,000 tokens (5.85%)**, against a predicted
1/16 = 6.25%, and the observed final characters were exactly the 16 alphabet positions
divisible by four, as the arithmetic requires.

**Decision: fix the test, not the library.**
- Tampering is now asserted **at every position of the token**, not one sampled position.
  The substitution steps a whole equivalence class at the final position and one alphabet
  position everywhere else, so every edit provably changes the decoded bytes. Measured
  after the change: **0 accepted tampered tokens across 500 fresh tokens** (~25,000 edits),
  where the old assertion failed 5.9% of runs.
- The tail equivalence class is now asserted in its own named test, so the property is
  *recorded* rather than lurking. A future reader who flips a trailing character will find
  the explanation instead of rediscovering it through a red build.

**Why not harden the application to reject non-canonical encodings.** It is not a forgery
route: producing one of the four variants requires already holding a valid token, and for
double-submit CSRF an attacker holding the token has already won. The alternative —
hand-rolling canonical-base64 enforcement around `itsdangerous` — adds bespoke
crypto-adjacent parsing to defend against a non-threat. Documented wart beats custom
crypto.

**Standing lesson.** A security test that fails ~6% of the time reads exactly like flake,
and the cheapest response — re-run until green — would have preserved a test that could
not distinguish "tampering is refused" from "tampering is accepted" in one case in sixteen.
An intermittently-failing assertion is a claim about the system that is *sometimes false*;
it deserves the same investigation as a hard failure, and I3/I6 forbid the re-run.

## D-025 — The private-engine source scan parses instead of greps (2026-08-02)

**Trigger.** `test_no_source_outside_db_layer_imports_engine_internals` failed against
`backend/features/compute.py`, which had *documented* the D-011 ban in its module
docstring — explaining that it accepts a session already scoped by `as_of()` and is given
no way to build an engine. The scan was a substring search over file text, so prose about
the rule read identically to a violation of it.

**Decision.** The scan now parses each file and reports three routes that actually grant a
handle: `import <engine>`, `from <engine> import ...` / `from backend.db import engine`,
and the module's name appearing as a **string constant** (the dynamic-import route).
Docstrings and comments are exempt; nothing else is.

**Why this is not a weakening.** Dynamic access still needs the module name as a string
*value*, and string constants are still checked — only docstrings and comments, which
cannot import anything, are skipped. A companion test asserts every one of the six
reachable forms is still caught and that the three prose forms are not, so replacing a
noisy check with a silent one would fail immediately.

**The reason it was worth changing rather than rewording one docstring.** A scan that
cannot distinguish a sentence from an import teaches authors to placate it by editing
prose. The existing file already carried the scar — its own constants are split across a
string concatenation with a comment telling future editors to *keep the literal out of
comments*. That workaround is a smell, not a solution: the next author writes the
docstring, watches a security test go red, and learns that the way to satisfy the security
test is to stop describing the security property.

## D-026 — Phase 4 universe construction: four conventions and one consequence (2026-08-02)

**The consequence first, because it is the load-bearing part.** `UniverseCriteria` makes a
strictly positive market-cap floor **mandatory**, and no shares-outstanding source exists
(B1). So **every `build_universe` call refuses today**, before it reads anything, with
`UniverseInputUnavailableError(blocker="B1")`. Phase 4 is built and tested; it cannot
produce a universe until the fundamentals connector lands. That is the I3-correct
behaviour and it is stated here so nobody reads "P4.1 DONE" as "we can build universes".
The borrow screen refuses on B2 the same way.

**Turnover is normalised by the sum of both sides' counts**: `(entered + exited) /
(prev_count + count)`, a fraction in [0, 1]. The obvious alternative — divide by the later
count — reports turnover above 1 whenever the universe shrinks, which is precisely when a
reader is most likely to be looking at it. At constant size the two agree, so the
convention costs nothing and removes a nonsense reading.

**The ADV window is a calendar span (1.5x + 10 days), not an exchange calendar**, because
no trading calendar exists yet. A window holding fewer bars than required yields
`adv = None` — the name is refused, not estimated from a shorter window. Estimating would
let a name enter the universe on the strength of three quiet days.

**`delisted_on == D` means still listed on D** (the delisting date is the last tradeable
day). Arbitrary but it must be written down: the survivorship-bias test turns on this
comparison, and an off-by-one here silently drops every name on its final day.

**The waterfall attributes each exclusion once, to the earliest screen in `FILTER_ORDER`.**
A name failing three screens is not counted three times, so `sum(removed) + members ==
candidates` holds exactly. This is why `FILTER_ORDER` is part of the criteria hash —
reordering the screens does not change *who* is in the universe but does change the
waterfall, and an artefact whose numbers move without a hash change is unreproducible (I2).

## D-027 — Phase 5 factor lags: zero for prices, a 7-day margin for fundamentals (2026-08-02)

**Price factors (momentum 12-1, short-term reversal, low volatility) declare a lag of
zero.** A daily close is knowable *at* that close — 16:00 ET, hours before the midnight
UTC that opens the next date. There is no filing and no publication step to wait for.

A defensive margin here would be worse than useless, for a reason worth stating: D-011
already makes the connector stamp `knowledge_time` honestly and the as-of session filter
on it. A lag *on top* would discard bars that genuinely were available, and if a vendor
file lands late an honest connector stamps it later, the bar goes invisible, and the
window shortens by itself. The margin would be double-counting a control that already
exists.

The residual risk is therefore **detected rather than absorbed**: a connector that reused
`valid_from` (`D 00:00Z`, the trading day's *open*) as its knowledge time would make the
compute date's own bar visible. `load_adjusted_closes` raises
`PriceTemporalIntegrityError` on any visible bar dated at or after the compute date. A
one-day lag would have hidden that defect while leaving it in the store for every other
consumer to trip over.

**Fundamentals factors (book-to-price, earnings yield, gross profitability, ROIC,
accruals, asset growth) declare 7 days**, as a margin on top of the store's
`knowledge_time`, sized from three uncertainties in the P3.5 connector that has not been
written: a date-only vendor `datekey` resolved to the next trading day (≤4 days across a
holiday weekend), filing date versus acceptance instant (1 day — `EdgarFiling` records a
real accession whose filing date *precedes* acceptance in UTC, which is the lookahead
direction), and vendor delivery after filing (2 days, **labelled a guess**). Revisit when
P3.5 lands and the real distribution is measurable.

**Known cost, accepted knowingly.** `FeatureSpec` carries one lag per feature, so
book-to-price applies its 7 days to the *price* leg too — B/P on `D` uses the close from
`D−7d`. That is signal loss, not lookahead, and it is the safe direction. A per-source lag
is not expressible in the wave-1 schema; changing the schema for it is backlog, not now.

**Standing lesson, from a defect that lived for minutes.** Mid-build `knowledge_cutoff`
read `instant + availability_lag` while its docstring, error message and doctest all said
minus — a lookahead of twice the lag. Its own unit tests passed, because they compared the
method against a recomputation of the same formula. What caught it was a *different*
suite asserting the cutoff against an independently stated expected instant. A test that
recomputes the formula it is checking cannot see a sign error in that formula.
`test_the_availability_lag_moves_the_cutoff_backwards_and_never_forwards` now pins the
direction against literals for that reason.

## D-028 — Overflow in a cross-sectional transform is a data condition, not a number (2026-08-02)

**What was wrong.** `cross_sectional_zscore` guarded only the *bottom* of the float64
range. `np.std` averages *squared* deviations, so a cross-section with spread above
~1.3e154 overflows to `inf`, and `(x - mean) / inf` returned `[-0., -0., 0., 0.]` —
"every security is precisely average". Finite, plausible, silent, and the exact output the
function's own docstring says it refuses to produce. `neutralize` and `beta_neutralize`
had the mirror defect, emitting `±inf`; worse, the module *rejects* infinite inputs, so
passing that output to the next stage raised a `ValueError` blaming a caller who had
supplied nothing infinite.

**Decision.** An overflow is a fact about the arithmetic on one date, so it is a **data
condition: NaN — never `inf`, never `0.0`.** Two helpers in `_stats.py` implement it:
`statistics_are_representable()` (a non-finite summary statistic voids the whole date) and
`nan_where_overflowed()` (per-entry, where only some results overflowed). NaN already
means "not available" everywhere else in the pipeline, so downstream code needs no new
case.

**`numpy`'s overflow `RuntimeWarning` is deliberately left unsuppressed.** It is a truthful
signal that a date reached the edge of float64; silencing it inside the library would hide
the condition from the operator. It fires only on genuine overflow, which real feature data
does not reach.

**Two guards were added and then deleted, on I6 grounds.** A mutation harness showed no
input can distinguish the guarded from the unguarded version across 60,000 extreme-magnitude
cases: a zero regression denominator is unreachable past the dispersion check, because it is
the same sum `np.std(…, ddof=1)` takes and `0 <= tolerance * scale` holds for every
non-negative scale. Unreachable code is untested code; the reasoning survives as a comment
where the guard was.

**Three docstring claims were false and are corrected.** (1) "Standard deviation is at most
1" after the full pipeline — it is not; a singleton sector drops the name carrying the
spread and *stretches* the survivors (measured 1.25). (2) The projection inequality does
not chain across steps: a later step can drop the name that absorbed an earlier step's
energy. (3) The z-score idempotence gap is `eps · κ` where `κ = max|x| / σ`, not `eps` —
measured at 5.1e-6 for κ=5.7e10, and the module accepts κ up to 1e12.

**`transform_cross_section` is NOT idempotent**, and not merely as a floating-point caveat.
Two independent mechanisms, each with its own counterexample: re-standardizing rescales
(`twice == once / σ_once`, with **ranks unchanged**, which is why it is easy to miss), and
the two projections do not commute — beta neutralization puts a sector bet back that sector
demeaning had removed.

## D-029 — Factor premia expectations are pre-registered, signed, and hashed (2026-08-02)

Three judgement calls in the P5.4 harness, each chosen against a plausible alternative.

**Expectations are written down before the data exists, and hashed into the artefact
stamp.** Each of the nine factors carries a sign, a magnitude range, a cited source, the
source's own published estimate with the arithmetic to an annualized fraction, the
construction the range assumes, and the caveats a reviewer needs — HML's dead 2007-2020
decade, accruals' post-2003 decay, low volatility being an *alpha* rather than a raw
return, short-term reversal's gross-to-net gap, and the equal- versus value-weighted factor
of ~2.5 on asset growth. `expectations_config()` feeds `canonical_config_hash`, so editing
a range, a citation, or a caveat changes the `config_hash` of every report produced
afterwards. Tuning the expectation to the result is not forbidden by exhortation; it is
*visible*, which is the only enforcement that survives contact with a disappointing
backtest.

**Ranges are signed intervals, not a magnitude plus a direction flag.** Accruals is
`−0.12 … −0.02`, not `0.02 … 0.12` with `sign = negative`. Under the flag encoding, a
premium of the *wrong* sign but plausible size can satisfy the magnitude test and fail only
the sign test — two independent checks over one fact, and a reader who glances at
"magnitude: within range" is misled. Under signed intervals a wrong-signed premium is
simply not in the interval, and cannot be.

**Too little data is a special case of no data, and significance is evaluated before
sign.** A short sample raises `InsufficientHistoryError` rather than emitting a weak
verdict, and the verdict logic asks "is there a premium at all" before "does it point the
right way". The alternative — reporting `SIGN_CONTRADICTS` on a statistically
indistinguishable-from-zero estimate — would let noise masquerade as a refutation of the
literature, which is the most expensive possible false positive for this project: it is the
finding a researcher *wants*, and therefore the one least likely to be audited.

**Consequence today.** No code path in this repository can build a non-empty return panel,
so the only reachable outcome of running the harness is `FactorReturnsUnavailableError`.
No premia table exists and none was fabricated.

## D-030 — A retraction carries no payload, enforced by two CHECKs (P2.10, 2026-08-02)

**The defect.** A retraction is a row: same logical key, same event-time interval, later
`knowledge_time`, `is_retraction = true`. Every payload column was `NOT NULL`, so writing
one meant inventing a `close_usd`, a `volume_shares` and an `adjustment_factor` that no
source ever stated. Those numbers sat in a fact table **byte-identical to real
observations** — fabricated data inside the store that I1 reads from. Nothing downstream
could tell them apart, and neither could the storage layer.

**Decision: make the absence structural, in both directions.** Migration 0012 drops
`NOT NULL` from every payload column — payload meaning every column that is neither part of
the logical key nor one of the five D-011 temporal/audit columns — and adds *two*
constraints per table:

- `ck_<table>_retraction_payload_absent` — `NOT is_retraction OR (every payload column IS
  NULL)`. A retraction carrying a value is refused.
- `ck_<table>_observation_payload_present` — `is_retraction OR (every required payload
  column IS NOT NULL)`. An observation missing a value is refused.

**The second constraint is the one that makes this a fix rather than a loosening.** Merely
dropping `NOT NULL` would trade a fabrication bug for a worse one: silently admitting
observations with missing payloads, which is the same class of defect pointing the other
way and harder to notice, because an absent number reads as a gap rather than as a lie.
Enforced at the database, so it holds on every role, session and write path — ORM, raw
`INSERT`, or `COPY` — not only where application code remembers to check.

### D-030 addendum — the rejected alternatives, and why SQL NULL is not Python None

Recorded after the fact: the track's own reasoning arrived after the constraint work was
verified and committed, and it is the part worth keeping.

**Sentinel value (`-1`, `NaN`, `0`) — rejected outright, and it is the tempting one.** It
*is* fabricated data by construction: a number in a numeric column, so `AVG(close_usd)`
consumes it and a leaked retraction hands a caller something that reads as a quote. Worse,
the sentinel must be chosen per column, and each choice is a fresh chance to pick a legal
value — `0` is a legal volume, `-1` a legal return, `1` a legal adjustment factor. It fails
I3 in the same breath it claims to serve it.

**Nullable payloads with `Mapped[Decimal | None]` throughout — rejected on the read
contract, not on effort.** Same storage shape. But the as-of layer masks retractions, so no
caller can obtain a row with a NULL payload; spreading `| None` would demand a `None` check
at every use site for a state that path cannot produce, and *checks that can never fire are
how real ones stop being read*.

**Separate retraction table — rejected on the versioned read.** A retraction must *compete*
in the latest-knowledge ordering: beat an earlier observation, lose to a later re-assertion.
So the read would UNION the two tables back into one stream and re-split them — this design
with a worse plan, a primary key split where no single constraint makes latest-wins
deterministic, and P2.9's chunk exclusion fragmented across an anti-join.

**`retract_fact` writes `sqlalchemy.null()`, not Python `None`, and the difference is the
whole point.** `macro_observation.is_missing` carries a `false` server default. An *unset*
attribute makes SQLAlchemy omit the column from the INSERT, so the default fills it in — and
the row ends up asserting `is_missing = false` about a fact nobody observed, reintroducing
exactly the fabricated claim this decision removes. Explicit SQL NULL is what actually
stores an absence.

**Two further enforcement points beyond the CHECKs**, so the guarantee does not rest on the
schema alone: `_assert_retraction_mask` re-derives from the *rewritten* statement that every
path to a fact table sits under a `NOT is_retraction` scope (swept offline against P2.12's
shape algebra — 3,064 shapes, zero false positives), and `_refuse_loaded_retraction` raises
if a retraction instance is ever loaded, whatever query produced it.

**The downgrade refuses to run if any retraction row exists.** The pre-0012 schema cannot
represent one without a fabricated payload, and inventing values so a downgrade can succeed
is precisely the behaviour this revision removes.

## D-031 — Drift reporting: bands are data, refusals are rows, and NaN is its own axis (P12.2, 2026-08-02)

**PSI's classic silent failure, demonstrated rather than asserted.** If the quantile bins are
recomputed from each period's own distribution instead of being fixed at reference time, PSI
is not merely attenuated — it is **identically 0.0 at a fifty-sigma shift**. Measured:
`shift=0.5, 1.0, 2.0, 50.0 → PSI = 0.0000000000` in every case. A detector built that way
reports perfect health forever, and nothing about its output looks wrong. The broken
implementation is kept permanently in the test file, asserted to return exactly 0.0, plus a
structural AST test that no public callable accepts two array-like samples — which is the
shape that would let a caller supply both distributions and reintroduce it.

**The 0.1 / 0.25 bands are convention, and the type system says so.** They live in
`drift.py` as a validated, replaceable `DriftBands` ladder carrying a mandatory `basis`
field, never as literals in the detector. They are folklore from credit scoring, not
derived, and a threshold whose provenance is a magic number in a comparison is one nobody
will revisit.

**A feature that cannot be measured becomes a row, not a gap.** `UnmeasurableFeature` has
**no renderable numeric attribute** — it cannot be accidentally charted as zero — and the
report's `complete` flag goes false. The alternative, dropping refused features, makes a
monitoring dashboard that is silently blind to exactly the features whose data broke.
Mutating the report to swallow refusals and claim completeness fails 5 tests.

**NaN rate is measured as its own two-category PSI, never summed into the distributional
value.** A change in availability usually means an upstream source broke — the most
important drift there is — and averaging it into a distributional number hides it. The
decisive fixture draws survivors from the reference distribution itself: distribution PSI
**0.0065** (correctly stable) while availability PSI is **5.364** at a 40-point rate change.
An implementation that dropped NaNs reports that fixture as perfectly healthy.

**Caller error and data condition are different exits.** `MonitoringInputError` propagates
out of `drift_report`; `InsufficientSampleError` becomes a row. And the refusal is
deliberately *not* a `ValueError` — asserted by test — so a stray `except ValueError`
cannot swallow it.

## D-032 — The cost governor is a client decorator, and there is no `check()` to forget (P7.7, 2026-08-02)

**Enforcement is a `ModelClient` decorator whose inner client is a private attribute**, so
no reachable path to a provider skips `authorize()`. The alternative — a
`governor.check()` the pipeline calls first — is identical when written correctly and one
refactor away from a call site that no longer checks. Consequence worth noting:
`ExtractionPipeline` already takes a client, so handing it a governed one governs every
chunk of every document with **zero edits** to `backend/extraction/tasks/`. Governance sits
behind the cache, so a hit spends nothing.

**The concurrency answer is that the unsafe operation does not exist.** There is no `check`
on the ledger interface. `reserve()` reads committed spend, compares both windows, and
writes the reservation as one atomic step, returning a reservation or raising. A caller
cannot use it wrongly by forgetting to hold something, because there is nothing to hold.
Postgres takes `pg_advisory_xact_lock(provider)` **before** the read, in the transaction
that writes — a lock around the write alone still lets two transactions read the same
headroom, and a test asserts the statement ordering.

*Sharp detail:* the in-memory ledger's critical section contains a deliberate cooperative
yield, placed where Postgres does its round trip. Without it the coroutine would be
**accidentally atomic**, the lock would be untestable, and deleting the lock would pass
every test.

**The estimate is an upper bound, and reconciliation is what makes that affordable.**
Output is `max_tokens` (exact — the provider cannot exceed the request's own cap). Input is
UTF-8 byte length plus a declared framing margin: byte-level BPE tokens each cover ≥1 byte,
so byte count dominates any token count. It over-states ~4x, and that pessimism is returned
to the window at settlement rather than shaved by a fudge factor — a ratio-based estimate
would be right on average and wrong on exactly the token-dense inputs that cost the most.

**A failed call settles at the bound; it does not release.** Nothing observable says whether
the request left the host. Over-counting refuses calls that would have fit; under-counting
lets real money out. The asymmetry is deliberate and documented rather than left to
inference.

**Degradation is per provider and must be strictly cheaper for *this call*** — whole-call
estimates, not headline rates, since a lower output rate can still lose on a long prompt.
Cross-provider degradation is refused at construction: it spends a different budget under a
different credential and sends the document to a different vendor. That is a configuration
change, not a degradation. Halt forbids a target, because a fallback that can never fire
reads as protection that does not exist.

**I3 on prices:** `CATALOG_PRICES` is an explicitly empty mapping and every lookup raises
`ModelPriceUnknownError`. There is no average, no cheapest-configured, no zero fallback — a
cap enforced against a made-up price enforces nothing.

## D-033 — Order lifecycle: paper-only with no seam, and idempotency from content (P11.2, 2026-08-02)

**Paper-only is six independent structural facts, none of them a switch.** The one that
actually closes the door the brief warned about — "a slot where a live endpoint could later
be dropped in by configuration" — is that **no function anywhere in the package takes a
parameter named** `venue`, `adapter`, `broker`, `client`, `endpoint`, `host`, `port`,
`transport`, `url`, `settings` or `live`. Asserted from the AST *and* again through
`inspect.signature` on the runtime objects. There is nothing to configure a live endpoint
into. Alongside: no transport imports at all (token-scanned, strings and comments dropped);
no `Protocol`/`ABC`/`Callable`/`import_module` routing seam; `ExecutionVenue` has exactly
one member, so a second is a code change plus a migration; `OrderIntent` takes no venue
argument and the column has a server default under `CHECK (venue = 'paper')`, so no writer —
ORM, raw `INSERT`, or `COPY` — can set it.

**A live fill has no representation.** `FillSource` is `SIMULATED` or `PAPER_BROKER` and
nothing else, bound in SQL by CHECK. That is I3 at the schema: a simulated fill and a
paper-broker fill are *distinct values on the row*, and a real one is not a value at all.
D-013 rides along as `fill_cost_basis = 'lower_bound'` under its own CHECK — a label that
travels into every query and export rather than living in a document, so P11.8 cannot
relabel a paper fill as a calibrated estimate without a migration.

**The idempotency key is content, never an identifier.** SHA-256 over canonical JSON of the
order's own fields *plus all four I2 stamp components*. Every timestamp, counter, UUID and
attempt number is excluded, and the reason is the failure it prevents: a counter must be
*remembered* across the retry, so the worker that restarted mints a new one and sends the
order twice. Content means the retry recomputes the identical key with nothing to remember.
The limit price renders at fixed scale so `Decimal("1.5")` and `Decimal("1.50")` cannot
produce two keys. Uniqueness is enforced by the database; `record_order` deliberately does
not look before it inserts, but inserts inside a savepoint and lets the constraint decide.

**`execution_order` has no state column.** Append-only tables cannot update one, and a
denormalized state that drifts from its history is exactly the condition the log exists to
prevent. Current state is `replay()` over the transition table, which validates the chain on
the way through.

**Two pending-cancel states, so the table stays a pure function of `(state, event)`.** A
venue's cancel-*rejection* must return the order to where it actually was; with a single
pending-cancel state that target depends on cumulative filled quantity, which would make the
machine a function of its own arithmetic.

## D-034 — A plpgsql RAISE is not an IntegrityError, and a double that omits SQLSTATE hides that (2026-08-02)

**What CI found.** P11.2's 25 integration tests had never executed — no Docker daemon here.
Their first real run failed five, and none was a typo. The asyncpg driver maps Postgres
conditions to SQLAlchemy classes like this:

| Postgres condition | asyncpg | SQLAlchemy |
|---|---|---|
| `P0001` plpgsql `RAISE EXCEPTION` | `RaiseError` | generic **`DBAPIError`** |
| `23505` unique_violation | `UniqueViolationError` | `IntegrityError` |
| `23514` check_violation | `CheckViolationError` | `IntegrityError` |

That one table explains all five. Four tests caught `IntegrityError` while a `BEFORE INSERT`
trigger — which fires ahead of every CHECK — refused the row with `P0001`. And the
concurrency test's five losers were refused by **the trigger, not the index**: under
`READ COMMITTED` a losing writer's `INSERT` takes a fresh snapshot, so the chain guard sees
the winner's committed row that the loser's own earlier `SELECT` did not.

**Decision: decide on SQLSTATE, never on the exception class.** `store.py` now catches
`DBAPIError` and reads `sqlstate`/`pgcode` off `exc.orig`. This also removes a *latent* bug
the old code had: a CHECK violation is an `IntegrityError` too, so `except IntegrityError`
would have reported a permanently-invalid row as a retryable conflict. Migration 0014 splits
the trigger's sequence check — position at or below the tail raises
`USING ERRCODE = 'unique_violation'` (the same refusal the primary key gives, arriving by
another route), while a *gap* keeps `P0001`, because nobody holds that position and retrying
it would loop forever.

**Deliberate widening, flagged as the one judgement call:** the trigger now returns `23505`
for any position at or below the tail, so a malformed manual insert reusing an old sequence
number is reported as a conflict rather than a chain error. The position *is* taken and the
primary key would say the same — but it is a widening, not a neutral refactor.

**The lesson about the double is not "doubles lie".** It is narrower and more useful.
`execution_order_transition` has a `BEFORE INSERT` trigger *and* a primary key; the double
modelled only the key — that is, **it stood in for the wrong layer**, simulating the path
that is *less* likely under contention. And its errors carried no SQLSTATE, so every refusal
it produced looked identical to code that inspects the code. That second defect is what made
the first invisible: no test could distinguish a correct store from one translating every
failure into a retry. The double is now itself under test, because infrastructure nothing
asserts is free to drift back.

**A CHECK that no input can isolate is a property of the schema, not a gap in the tests.** A
terminal `from_state` is necessarily also absent from the enumerated-transition table, so
`from_state_not_terminal` cannot be reached alone; the test names both constraints and
requires one. By contrast `fill_payload_present` *is* reachable — not via `fill_complete`
(a NULL quantity cannot move the running sum, while `filled` demands a full trade) but via
`partial_fill`, which carries no completeness requirement. The claim was kept and the input
corrected, rather than the claim weakened to fit the first input tried.

## D-035 — The coverage ratchet, and what it is not evidence of (CC.6, 2026-08-02)

**The number was wrong before it was a floor.** The backend measured "~97%" only because
`backend/tests` sat in the denominator — rows that are ~100% covered by construction. On
the same data, tests-in-denominator reads 93.86% against 88.20% shipped-only: a **4.21-point
pad**. Tests are now omitted, which *lowers* the reported figure and points the gate at the
code §8 is about. A coverage number that flatters itself is worse than none, because it is
quoted.

**Two holes in the mechanism, both silent.**
- `precision` defaults to **0**, and coverage compares `round(total, precision) <
  fail_under`. A real **84.6% rounds to 85 and clears an 85 floor.** Set to 2.
- Vitest's `thresholds.autoUpdate` rewrites the floor to the last run's number. Now
  explicitly `false` **and asserted**, so it is not available to reach for the next time the
  gate fails — which is exactly when someone would.

**The floor is a committed constant, pinned in two files.** Raising it is a visible
two-file diff; lowering it quietly is impossible. The frontend stays at **70** rather than
today's 85.96, because pinning to the current number ratifies a lucky run and punishes every
commit after it. An auto-ratchet is not a ratchet — it is a record of the best weather.

**What this gate does not prove, documented at the config sites and asserted by test** so
the caveat cannot be deleted from the thing it describes: not that anything was *asserted*
(an import with no assertions moves the number); not that assertions are about outcomes; not
that the uncovered residue is random — the cheapest way to lift a ratio is to test easy
code, so the remainder drifts toward the error paths; not comparable across commits, since
deleting untested code raises it. **Every property this project actually rests on — I1
temporal integrity, the append-only triggers, no-alpha-on-pure-noise — is asserted directly,
and not one of them would be caught by this gate.** It is a decay alarm, not a quality
measure, and it is written down that way so a future reader does not mistake 88% for a
statement about correctness.

**`# pragma: no cover` is budgeted at 9 with zero headroom**, each required to carry a
reason. The next one added fails loudly and explains itself. Two existing pragmas in
`backend/portfolio/optimizer.py` say only "defensive" and are too thin — flagged for that
file's owner rather than edited.

## D-036 — Role separation closes D-012's residual weakness, and two routes D-012 never named (CC.9, 2026-08-02)

**D-017's deadline is met, late.** Phase 11 had already started (P11.2 landed first), which is
recorded in `BLOCKERS.md` rather than smoothed over.

**The two routes D-012 did not name are the reason this could not stay backlogged.** D-012
described the weakness as "the owning role can `DROP`/`DISABLE` its own triggers". That
understates it, and both omissions are worse than the one it named because neither produces
an error:

- **`SET session_replication_role = 'replica'`** silences every `ORIGIN` trigger for the
  session while leaving `DELETE` working normally. Demonstrated live during this work: with
  `SUPERUSER`, the application role deleted **every `price_bar` row straight through the
  append-only trigger**. It requires superuser — which is precisely why the new role carries
  `NOSUPERUSER`.
- **`TRUNCATE` fires no row trigger at all**, and migrations 0003/0004 deliberately left it
  unblocked for the test reset. Under a single role, `TRUNCATE config_change_event` erased
  the "immutable" audit log **with no error and no trace**.

Neither is closed by a trigger. Only privilege separation closes them, which is the argument
for why D-012's mitigation was never sufficient on its own.

**The grants.** The application role owns nothing, holds no role memberships, and has every
attribute flag false except `rolcanlogin`. `SELECT` + `INSERT` on all tables; `UPDATE` on
`ingestion_run` alone (a run must close `running → succeeded/failed`); `UPDATE`/`DELETE` on
`llm_provider_credential` alone (a rotation must overwrite ciphertext and a deletion must
remove it, or a KEK compromise widens from "every key in use" to "every key ever used").
Those are exactly the two tables that revisions 0003–0017 left without an append-only
trigger, so no immutability claim is weakened by the exceptions. `alembic_version` is
read-only. `ALTER DEFAULT PRIVILEGES` makes the default for future tables `SELECT`+`INSERT`,
so a future *mutable* table must grant in its own migration — default-deny, and the failure
is `permission denied`, never silent.

**I5 at the boundary.** The password reaches PostgreSQL only as a **bind parameter** to
`set_config(..., is_local => true)`; the `CREATE ROLE` text is assembled server-side by
`format(%L)`. It is therefore in no statement string this process holds and cannot reach a
log, an echoed statement, or a `DBAPIError`. The handler re-raises with SQLSTATE only, never
`SQLERRM`. Verified by forcing a `CREATE ROLE` failure and confirming the message, detail and
context carried no password and no statement text.

**Two mutations initially survived, and both were D-025's defect recurring.** A substring
scan placated by prose: flipping `NOSUPERUSER` to `SUPERUSER` passed because the module
*docstring* contains the word, and reverting the migration engine to `database_url` passed
because its docstring mentions `migration_url`. Fixed by parsing — the SQL constants and the
AST, where docstrings contribute nothing. **The same lesson has now cost three separate
tracks**; a scan over file text cannot distinguish a rule from a description of the rule.

**Still unverified:** TimescaleDB chunk ACL propagation — that grants on `price_bar` and
`edgar_filing` reach their chunks. 0015 re-issues a grant per hypertable *by name* precisely
because the `ALL TABLES` form may not propagate, but no Timescale instance was reachable.
The integration suite's insert-and-read test is what will confirm it.

## D-037 — Reconciliation tolerances are derived, and nothing may refuse a halt (P11.3/P11.5, 2026-08-02)

**Cash tolerance: 1 US cent, absolute — derived, not chosen.** The only legitimate
divergence between two correctly-maintained USD balances is representation: a statement
renders at 2 dp while our balance is carried at scale 6, and quantising scale-6 to scale-2
moves a value by at most **half a cent**. One cent is two quantisation ticks. The
load-bearing half is the second one: the tolerance is *smaller than the smallest real
break* — the cheapest thing that can actually go wrong (a commission, one share of any name
a price screen admits, a dropped fill) is larger by orders of magnitude. A tolerance that
cannot be exceeded by rounding and cannot absorb a real error is doing its whole job.

**Positions: zero tolerance, and no parameter through which to widen it.** Share counts are
integers compared with `==`. No arithmetic legitimately produces an off-by-one share, so a
quantity tolerance could only ever hide a break. That asymmetry with cash is exactly why they
are not one knob.

**The tolerance itself has a ceiling** (5 cents = ten ticks), restated as a CHECK in
migration 0016 so no Python can waive it. Tightening to zero is always allowed; widening past
the point where the derivation holds is refused by the database.

**Contemporaneity: 300 seconds**, because the mirror failure of a too-wide tolerance is a
comparison that *manufactures* breaks. A "mismatch" that means "a fill landed in the gap" is
how a real alarm gets ignored. Snapshots further apart yield **no verdict**, and a cycle with
no verdict is an unknown condition the kill switch halts on — failing towards a halt.

**Four mismatch kinds, because the operational response differs.** Shares the venue holds
and we do not is the most serious: capital at the venue that the risk model, optimizer and
drawdown monitor do not know exists. Its converse is serious *differently* — no unmanaged
capital, but our books overstate the account and the next order can instruct a sale of shares
that are not there. And "both flat, one side silent" is **not a break but is recorded
anyway**: an explicit zero is a statement, an absence is silence, and coercing absence to zero
makes a truncated statement indistinguishable from a confirmed flat book.

**The halt log is deliberately asymmetric: nothing may refuse an engagement.** `evaluate` is
pure and never raises — each probe's exception becomes `UNKNOWN_CONDITION`, which exists as a
fifth trigger precisely because an exhaustive trigger list fails open on everything not on
it. Halting is the default; *not* halting must be earned by four well-formed, in-limit
measurements. A missing measurement halts. An unconfigured limit halts — "no limit set" is
not "no limit". A `Decimal("NaN")` halts, because it compares false against every threshold,
which is the textbook fail-open. `engage_halt` has no pre-read, no savepoint and no
uniqueness check. Only *clearances* are constrained: explicit, attributed, by id, at most
once. And `assert_not_halted` raises if the log cannot be **read** — a database outage must
not do what no operator is permitted to do.

**A tolerance-shaped lesson about the tests themselves.** An aborted mutation run left
`MAX_SNAPSHOT_SKEW_SECONDS = 10**9` in the tree and the suite did not notice, because the
skew tests were written *relative to the constant*. A test that derives its expectation from
the value under test cannot detect that value being wrong — the same shape as D-027's sign
inversion, which survived its own unit tests for the same reason. All three thresholds are
now asserted as literals.
