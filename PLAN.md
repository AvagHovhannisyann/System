# PLAN.md — Systematic Equity Research Platform

> Read `DIRECTIVE.md` first. This plan decomposes it into tasks. Session protocol:
> read `PLAN.md`, `PROGRESS.md`, `BLOCKERS.md` → pick the next unstarted task on the
> active track → work one task at a time → test → commit → log to `PROGRESS.md`.

**Legend**
- Status: `TODO` · `WIP` · `DONE` · `BLOCKED(Bn)` · `GATE-PASSED`
- Complexity: `S` (≤1h) · `M` (half-day) · `L` (1–2 days) · `XL` (multi-day, split before starting)
- `[UI]` = dashboard task. `[ER]` = use extended reasoning (directive §0.3).
- A phase is complete only when its gate task is `GATE-PASSED` (directive §9.10).

---

## Gate index

| Gate | Phase | Binary condition |
|---|---|---|
| G1 | 1 | `docker compose up` produces a working stack from a clean clone; CI green |
| G2 | 2 | 10,000-case Hypothesis as-of suite, zero `knowledge_time > as_of` leaks; direct-table-access-impossible test. **Evidence must include query-shape DEPTH, not only data breadth** — a breadth-only suite passed 87k cases while a live I1 leak sat in a two-fact-table shape (P2.11) |
| G3 | 3 | Historical snapshot matches independent reference within tolerance; delisted names present; DQ report per source |
| G4 | 4 | Past universe contains later-delisted names; size/turnover history inspected for discontinuities |
| G5 | 5 | Known factor premia reproduce (sign + plausible magnitude); correlation matrix reviewed; no lookahead in any feature |
| G6 | 6 | Label distribution sane across regimes; uniqueness weights sum correctly; overlap down-weighting proven with ESS report |
| G7 | 7 | Golden-set agreement **above the measured intra-rater noise floor** (threshold derived, not the literal 85% — D-014); contamination divergence below threshold; 1,000-doc re-run within budget; cache hit >90% |
| G8 | 8 | Purged K-fold + embargo unit-tested; IC with t-stat reported; every config in `TESTING_LEDGER.md` |
| G9 | 9 | Optimizer solves across historical dates; turnover penalty demonstrably reduces turnover; cost defaults flagged conservative |
| G10 | 10 | Framework recovers injected signal on synthetic data AND reports near-zero alpha on pure noise |
| G11 | 11 | End-to-end paper cycle; injected reconciliation mismatch caught; kill switch halts within one cycle |
| G12 | 12 | Injected drift detected; injected performance deviation triggers halt |

## Dependency graph (phase level)

```
P1 ─→ P2 ─→ P3 ─→ P4 ─→ P5 ─→ P6 ─→ P8 ─→ P9 ─→ P10 ─→ P11 ─→ P12
             │                 ↑
             └───→ P7 (LLM) ───┘        (P7 features feed P8 *when G7 passes*; P7 needs
                                         P2+P3 docs, independent of P4–P6 — parallel
                                         track. P8 does NOT block on P7: B3/B4 could
                                         stall it indefinitely. Every training run
                                         records its feature-set version in the ledger,
                                         so baseline-only runs and LLM-augmented runs
                                         are distinct experiments.)
CC.* cross-cutting tasks slot in where their "after" column says.
```

Blocker-independent tracks (directive §10): if the data-vendor blocker (B1) halts P3,
work P2 hardening, P7 pipeline scaffolding (EDGAR is keyless), or CC tasks.

---

## Phase 0 — Bootstrap (session 1)

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P0.1 | Commit `DIRECTIVE.md` + all state files (`PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md`, `TESTING_LEDGER.md`, `BACKLOG.md`, `README.md`, `.gitignore`) | — | M | DONE |

## Phase 1 — Foundation

Deliverables: compose stack (TimescaleDB, Redis, backend, frontend), Pydantic Settings config, structured JSON logging with correlation IDs, Alembic wired, CI (lint+typecheck+test), health endpoint.

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P1.1 | Backend scaffold: root `pyproject.toml` (uv), `backend` package (`core/`, `db/`, `api/`, `tests/`), FastAPI app factory, Pydantic Settings (`core/config.py`), structlog JSON logging + correlation-ID middleware (`core/logging.py`), `/api/health` (DB+Redis component checks), SQLAlchemy 2.0 async engine + session, Alembic baseline migration, ruff (incl. pydocstyle) + `mypy --strict` + pytest all passing | P0.1 | L | DONE |
| P1.2 | Frontend scaffold: Next.js 15 App Router, TS strict, Tailwind, shadcn/ui, TanStack Query; app shell + system-status page reading `/api/health`; `tsc --noEmit`, eslint, `next build` all passing. Functional UI only — visual design is Fabel's lane (see D-006) | P0.1 | M | DONE |
| P1.3 | Containers: backend Dockerfile (uv, non-root, entrypoint runs `alembic upgrade head`), frontend Dockerfile (standalone), `docker-compose.yml` (timescaledb-pg16, redis, backend, frontend; healthchecks, `depends_on: condition: service_healthy`), `.env.example` | P1.1, P1.2 | M | DONE |
| P1.4 | CI + hooks: GitHub Actions — backend job (ruff, `mypy --strict`, pytest+coverage, I6 skip-guard), frontend job (eslint, `tsc --noEmit`, `next build`), dependency-scan job (pip-audit, audit-ci w/ justified allowlist — D-010); pre-commit config (.env-guard script, ruff, key-pattern grep) | P1.1, P1.2 | M | DONE |
| P1.5 | **Gate G1:** clean-clone `docker compose up` verified (health endpoint answers, frontend serves, migrations applied); CI green on the pushed branch | P1.3, P1.4 | S | GATE-PASSED (2026-07-31, see PROGRESS) |

## Phase 2 — Bitemporal store  [ER]

Design first, then enforce. This is the layer every backtest's honesty rests on.

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P2.1 | [ER] Bitemporal design doc → `DECISIONS.md`: column semantics (`valid_from`, `valid_to`, `knowledge_time`), restatement representation, delete/correction handling, timezone policy (all UTC, `TIMESTAMPTZ`), index strategy for as-of access, hypertable partitioning choices | G1 | M | DONE (D-011) |
| P2.2 | `BitemporalMixin` + declarative base; first fact tables (securities master, prices) carrying the three columns; Alembic migration | P2.1 | M | DONE |
| P2.3 | Query layer: `as_of(as_of_ts)` async context yielding a scoped session that transparently rewrites every SELECT on bitemporal tables to the versioned form (`knowledge_time <= as_of`, latest-knowledge-wins, retraction masking). Write path: `knowledge_time` is **writer-supplied under each connector's declared policy — never server-stamped** (D-011; server stamping would falsify backfills) | P2.2 | L | DONE |
| P2.4 | Bypass prevention: mechanism making direct table reads outside the query layer fail from application code (session-factory guard + SQLAlchemy event assertion + lint rule banning raw `select()` on fact tables outside `db/`), decision logged | P2.3 | L | DONE |
| P2.5 | Hypertables on time-series fact tables partitioned on `valid_from` + composite index `(entity_id, valid_from, knowledge_time DESC)` (D-011 physical layout); DB-level append-only enforcement; migration + EXPLAIN sanity check that outer quals push down into the versioned subquery | P2.2 | M | DONE |
| P2.6 | Hypothesis property suite: random facts (random valid intervals, knowledge times) + random as-of queries; 10,000 cases; assert no returned row has `knowledge_time > as_of`; runs against real Postgres via Testcontainers | P2.3 | L | DONE (81,121 cases, 0 failures) |
| P2.7 | Bypass-impossibility test: prove application code cannot read fact tables without the query layer (import-time + runtime enforcement both exercised) | P2.4 | M | DONE |
| P2.8 | **Gate G2:** P2.6 zero failures + P2.7 passing, in CI | P2.6, P2.7 | S | **GATE-PASSED (re-evaluated 2026-08-01)** — first passed on breadth evidence alone, REOPENED when P2.12's depth suite found a live I1 leak (stale as-of bind), re-passed only after P2.11 fixed it. Evidence now: 221 tests green, 75,936 breadth cases + 8,944 depth cases to nesting depth 14, 330 shapes rewritten / 140 refused, zero structural violations |
| P2.11 | **DONE (D-018).** Fix stale as-of bind (I1 leak). `_rewrite_select` runs one adapter pass per fact table over already-rewritten trees, so the as-of literal becomes several distinct `BindParameter` objects; two rewrites of one shape at different as-of values then share a compiled-cache key whose bind count disagrees with the compiled object, and a cache hit resolves the first param to the **stale** value. Trigger: entity-level `select(<fact>)` + a second fact table + the first referenced ≥2 places — the exact Phase 4–6 shape (join `price_bar`↔`security_master`, re-run across a rebalance calendar). Survives every existing defence because the **SQL text is correct**; only the bind value is wrong. Fix = single shared bind identity + a **structural backstop** verifying every as-of bind actually sent equals the session's as_of (fail-closed), so the class closes, not just the instance | P2.12 | L | DONE |
| P2.12 | Depth-parameterized as-of property suite: recursive shape algebra (4 base × 13 wrapper × 2 top families + 4 refusal families), depth drawn first as a first-class dimension, identity-preserving wrappers so the oracle is exact set equality; asserts (a) per-row `knowledge_time <= as_of` + set equality, (b) every fact-table reference in the **final compiled SQL** sits inside the versioned `DISTINCT ON` form at every level, (c) refused shapes raise and send no fact-table SQL. Measured depth 1–13 | P2.8 | L | DONE (found the P2.11 leak) |
| P2.9 | EXPLAIN verification that outer quals push down into the versioned subquery past `DISTINCT ON` into **chunk exclusion**. **Scheduled: the moment the first hypertable holds meaningful volume — not before (empty-table plans prove nothing), not after Phase 4** (D-016). Performance gate with design consequences: no pushdown ⇒ every as-of query full-scans; at Phase 6 volumes that is minutes vs days, and if the fix changes how predicates are injected it must land before P4–P6 write against the current shape | P3.4 data loaded | S | SCHEDULED (D-016) |
| P2.10 | Retraction-row payload hygiene: retractions must fabricate NOT NULL payload columns; mark/enforce placeholder payloads so they cannot be mistaken for data (audit finding, I3-adjacent) | P2.8 | S | DONE (2026-08-02) — migration 0012, two CHECKs per table, see D-030; integration unrun (no Docker) |

## Phase 3 — Data ingestion

Every connector: retry w/ backoff, rate limiting, incremental sync, data-quality report. **B1 (vendor selection + keys) gates most connectors — EDGAR needs no key and goes first.**

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P3.1 | Connector framework: base class (retry, rate-limit, incremental checkpoint, DQ metrics emission), Celery + beat wiring, ingestion-run tracking table marking each run `backfill\|live` with a DQ check flagging live runs whose knowledge_times trail ingestion beyond the source's declared lag (D-011 compensating control), rejection/flagging of future `knowledge_time` at write, and the **open-interval supersession contract**: a connector closing an open-ended fact writes a later-knowledge correction row with the same `valid_from` and a bounded `valid_to` (audit finding — until written, consecutive open intervals overlap) | G2 | L | DONE |
| P3.2 | SEC EDGAR connector — **the reference implementation of the P3.1 contracts** (operator direction): filings index + documents, **acceptance timestamps from the daily index files** as `knowledge_time`, incremental sync. Chosen first because it needs no key (B1-independent), because acceptance timestamps are the hardest temporal-correctness problem in Phase 3 — exercising the bitemporal layer against a real adversarial case rather than a synthetic one — and because it is the Phase 7 input anyway. Sharadar drops in behind the same interface once B1 clears | P3.1 | L | DONE |
| P3.3 | Corporate actions connector (splits, dividends, delistings, ticker changes, M&A) — **must include delisted securities** | P3.1, B1 | L | BLOCKED(B1) |
| P3.4 | Daily OHLCV connector with full adjustment history (raw + adjustment factors stored separately) | P3.1, B1 | L | BLOCKED(B1) |
| P3.5 | Point-in-time fundamentals connector: unrestated, original report dates → `knowledge_time` | P3.1, B1 | L | BLOCKED(B1) |
| P3.6 | Earnings call transcripts connector | P3.1, B1 | M | BLOCKED(B1) |
| P3.11 | **Gate G3** prerequisites note: EDGAR (P3.2) is done and gives filings coverage, but G3's snapshot-vs-reference check needs price/fundamental data — still B1-blocked |  | S | BLOCKED(B1) |
| P3.7 | Borrow availability + rates — **from IBKR in Phase 11, not a data vendor** (operator decision, B1). Rescoped: no vendor connector; the interface is defined here and fed by the Phase 11 adapter | P3.1, B2 | M | BLOCKED(B2) |
| P3.8 | Macro series connector (FRED — keyless tier available) | P3.1 | M | DONE (2026-08-01, see PROGRESS) |
| P3.9 | Data-quality report: per-source coverage, gaps, staleness; persisted per ingestion run | P3.2 | M | DONE |
| P3.10 | [UI] Data Health page: coverage heatmap, gap list w/ severity, staleness monitor, run history, manual re-sync trigger (rate-limited, CSRF-protected — CC.2) | P3.9, CC.2 | L | TODO |
| P3.11 | **Gate G3:** fixed-historical-date snapshot vs independently sourced reference within tolerance; delisted names present; DQ report live | P3.3–P3.9 | M | BLOCKED(B1) |

## Phase 4 — Universe construction

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P4.1 | PIT universe builder: ADV floor, price floor, market-cap floor, borrow-availability filter, exchange filter; all inputs read through `as_of()`; per-rebalance-date snapshots persisted | G3 | L | DONE (2026-08-02) — 224 tests; refuses on B1 until fundamentals land, see D-026 |
| P4.2 | Historical reconstruction + size/turnover series + filter-impact waterfall data | P4.1 | M | DONE (2026-08-02) — history, size/turnover, waterfall; 224-test suite shared with P4.1 |
| P4.3 | [UI] Universe page: constituents w/ entry/exit, size & turnover history, filter waterfall, PIT browser (pick any past date) | P4.2 | L | TODO |
| P4.4 | **Gate G4:** past universe contains later-delisted names (test); size/turnover history rendered and manually inspected — inspection notes to `PROGRESS.md` | P4.2 | S | TODO |

## Phase 5 — Feature library

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P5.1 | Feature framework: declaration schema (name, definition, **units**, availability lag, source tables), registry with **hard 30-feature cap** (enforced in code + test), lag enforcement in the compute path | G4 | L | DONE (2026-08-02) — 137 tests, 16/16 mutations killed, 3 defects fixed |
| P5.2 | Transform pipeline: winsorize 1/99 → cross-sectional z-score → sector neutralize → optional beta neutralize; property-tested (idempotence, NaN policy, no cross-date leakage) | P5.1 | L | DONE (2026-08-02) — 134 tests, 19/19 mutations killed, 6 defects fixed, see D-028 |
| P5.3 | Baseline factors: momentum 12-1, book-to-price, earnings yield, gross profitability, ROIC, accruals, asset growth, low volatility, size, short interest, Amihud illiquidity — each with docstring stating units + assumptions | P5.1, P5.2 | XL (split per-factor at start) | DONE (2026-08-02) — 12 factors, all 11 PLAN names; 250 tests, 43/43 mutations; see D-027, B7 |
| P5.4 | Factor premia validation harness: long-sample sign + magnitude check per factor vs published stylized facts | P5.3 | L | DONE (2026-08-02) — 140 tests, 27/27 mutations; measurement blocked on B1, see D-029 |
| P5.5 | [UI] Features page: catalog (definitions, lags, coverage), correlation heatmap, rolling IC, distributions, enable/disable toggles writing versioned config (CC.1 events, never mutating a live model) | P5.3 | L | TODO |
| P5.6 | **Gate G5:** premia reproduce; correlation matrix reviewed (notes to `PROGRESS.md`); lookahead test per feature | P5.4 | M | TODO |

## Phase 6 — Labels  [ER]

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P6.1 | [ER] Triple-barrier engine: upper/lower barriers sized by trailing volatility, vertical barrier per horizon (5d, 21d, 63d); math documented in `DECISIONS.md` before code | G4 | L | DONE |
| P6.2 | Residualization of returns vs market + sector prior to labeling | P6.1 | M | DONE |
| P6.3 | Sample-uniqueness weights from label overlap; effective-sample-size computation | P6.1 | L | DONE |
| P6.4 | Property tests: barrier-touch correctness on constructed paths, weight normalization, overlap ⇒ down-weighting proof, ESS report artifact | P6.2, P6.3 | L | DONE |
| P6.5 | **Gate G6:** distribution-sanity across regimes + P6.4 suite green | P6.4 | S | TODO |

## Phase 7 — LLM extraction pipeline (parallel track after P3.2)

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P7.1 | Provider registry backend: Fernet-encrypted keys (KEK from env — CC.0), masked display only, no reveal endpoint, test-connection probe; audit-logged config events | G2, CC.0 | L | DONE (2026-08-01, see PROGRESS) |
| P7.2 | Anonymization pipeline: mask company names/tickers/executive names, strip all dates; adversarial test set proving masking on EDGAR text | P3.2 | L | DONE (2026-08-01, see PROGRESS) |
| P7.3 | Extraction task framework: chunking, schema-validated outputs (Pydantic), temperature 0, raw responses stored, prompt version hash on every extraction | P7.2 | L | DONE (2026-08-01, see PROGRESS) |
| P7.4 | Delta-oriented extraction tasks: risk-factor language change, guidance tone vs magnitude, Q&A evasiveness, accounting-language shift, added/removed risk factors | P7.3 | XL (split per-task at start) | TODO |
| P7.5 | Ensemble: 2–3 cost-tier models, parallel calls, median aggregation, disagreement score, review flag above threshold. **Backfill is pilot-gated: 200 names / 5 years / 2-model ensemble until Phase 8 proves incremental IC over the P5 baseline (D-015)** — the pilot is a ledger row | P7.3, B4 | L | BLOCKED(B4) |
| P7.6 | Cache keyed `hash(document + prompt_version + model)`; prompt change auto-invalidates affected documents | P7.3 | M | DONE (2026-08-01, see PROGRESS) |
| P7.7 | Cost governor: per-provider daily+monthly caps enforced **before** each call; halt vs degrade-to-cheaper-model behavior; spend tracking | P7.5 | L | DONE (2026-08-02) — 101 tests, 17/17 mutations; live enforcement still BLOCKED(B4): no keys, caps or prices configured. See D-032 |
| P7.8 | Golden set harness: storage, scoring, per-prompt-version score history, **blind re-label mode** (same document served without prior label, for the intra-rater measurement) and **single-construct-across-all-documents** ordering — never all constructs per document (D-014, B3 protocol). Labels are human — B3 | P7.4 | L | BLOCKED(B3 for labels; harness TODO) |
| P7.9 | Contamination probe: anonymized-vs-named scoring divergence, threshold, wired into CI as deploy gate | P7.5 | L | TODO |
| P7.10 | Prompt management: content-addressed versions, history, diff data, golden-score attachment, rollback | P7.3 | L | DONE (2026-08-01, see PROGRESS) |
| P7.11 | [UI] Agents & Extraction page (operator's home — split): registry & key mgmt; per-task model assignment (versioned); prompt history/diff/rollback; ensemble config; cost gauges; quality trends; document inspector (anonymized text sent, raw responses, aggregate, disagreement) | P7.1–P7.10 | XL (split at start) | TODO |
| P7.12 | **Gate G7:** ≥85% golden agreement; contamination divergence under documented threshold; 1,000-doc re-run within budget; >90% cache hit on repeat | P7.5–P7.10 | M | BLOCKED(B3,B4) |

## Phase 8 — Predictor

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P8.1 | Purged K-fold with embargo; unit tests on exact fold boundaries incl. overlap/embargo edge cases | G5, G6 | L | DONE |
| P8.2 | LightGBM pipeline: `max_depth ≤ 4`, high `min_child_samples`, L1+L2, `feature_fraction ≈ 0.6`, early stop on purged fold; MLflow tracking; config hash + seed + data version + git commit stored per run (I2) | P8.1 | L | TODO |
| P8.3 | Time-window ensembling (no stacking); output = cross-sectional rank only | P8.2 | M | TODO |
| P8.4 | IC + t-stat reporting; **automatic append of every configuration to `TESTING_LEDGER.md`** wired into the training entrypoint so a run cannot complete unlogged | P8.2 | M | TODO |
| P8.5 | [UI] Models page: run history w/ full config, SHAP importances + instability warning, IC/t-stat, purged-fold visualization, registry promote/archive | P8.4 | L | TODO |
| P8.6 | **Gate G8:** fold-boundary tests green; IC+t-stat reported; ledger completeness check (runs in DB == rows in ledger) | P8.4 | S | TODO |

## Phase 9 — Portfolio and costs  [ER]

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P9.1 | Ledoit-Wolf shrinkage covariance; PSD + shrinkage-intensity property tests | G8 | M | DONE |
| P9.2 | [ER] cvxpy optimizer: max ER − risk penalty − **explicit turnover penalty**; sector & beta neutrality, 2% position cap, 20% sector cap, full investment; infeasibility diagnosis + documented relaxation ladder | P9.1 | XL (split at start) | DONE (2026-08-01, see PROGRESS) |
| P9.3 | Cost model: half-spread + commission + sqrt-impact (order size / ADV) + borrow on shorts; **units in bps documented on every function**; conservative defaults flagged `UNCALIBRATED` until calibrated from paper fills (P11.8 owns clearing the flag) | G8 | L | DONE |
| P9.4 | Hypothesis suites: optimizer constraint satisfaction on random inputs; cost-model monotonicity/scaling properties | P9.2, P9.3 | L | DONE (2026-08-02) — 45 tests; found the cost float-resolution and bypass-scan defects |
| P9.5 | [UI] Portfolio page: current vs target, drift, sector/factor exposures, planned trades w/ est. cost, constraint-binding indicators | P9.2 | L | TODO |
| P9.6 | **Gate G9:** optimizer solves across historical dates; turnover-penalty A/B in backtest shows reduced realized turnover; uncalibrated-cost flag visible | P9.4 | M | TODO |

## Phase 10 — Validation framework  [ER — maximum care]

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P10.1 | [ER] Backtest engine: event-driven daily loop, all reads through `as_of()`, net-of-cost only (I4), artifact stamped with git commit + data version + config hash + seed (I2) | G9 | XL (split at start) | DONE (2026-08-01, see PROGRESS) |
| P10.2 | [ER] CPCV: combinatorial purged splits → **distribution** of OOS Sharpe ratios | P10.1 | L | DONE |
| P10.3 | [ER] Deflated Sharpe Ratio using full trial count parsed from `TESTING_LEDGER.md` | P10.2 | L | DONE |
| P10.4 | [ER] Probability of Backtest Overfitting | P10.2 | L | DONE |
| P10.5 | Walk-forward analysis + buy-and-hold benchmark comparison shown with every result; CIs on every metric; UI contract: no point estimate without interval | P10.1 | L | DONE (2026-08-01, see PROGRESS) |
| P10.6 | [ER] **Synthetic-truth harness:** injected signal of known strength recovered; pure noise reports near-zero alpha. The noise test is the critical one | P10.2–P10.5 | L | TODO |
| P10.7 | [UI] Backtests explorer: equity curve vs benchmark, **CPCV Sharpe histogram with point estimate marked** (the most important visualization in the app), DSR, PBO, drawdowns, turnover, cost waterfall, per-year attribution, run comparison, ledger trial count on every view | P10.5 | XL (split at start) | TODO |
| P10.8 | **Gate G10:** synthetic-truth harness green both directions, in CI | P10.6 | S | TODO |

## Phase 11 — Execution (paper only)

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P11.1 | IBKR paper adapter via `ib_insync`: **paper endpoint hard-coded** (port 7497/paper gateway only; no live config path — I-check test asserts no such flag exists in codebase), needs B2 creds | G10, B2 | L | BLOCKED(B2) |
| P11.2 | OMS: order lifecycle, idempotency keys, persistence | P11.1 | L | DONE (2026-08-02) — 195 tests, 56/56 mutations; paper-only structural, see D-033. Integration (25 tests) unrun: no Docker |
| P11.3 | Reconciliation each cycle: positions/cash vs broker; mismatch ⇒ halt + alert | P11.2 | L | TODO |
| P11.4 | VWAP/TWAP slicing | P11.2 | M | TODO |
| P11.5 | Kill switch: drawdown breach, stale data, reconciliation mismatch, manual trigger; halts within one cycle | P11.3 | L | TODO |
| P11.6 | [UI] Execution page: blotter, fills w/ slippage vs arrival, reconciliation status, cost calibration (predicted vs realized), kill-switch status + manual trigger | P11.3, P11.5 | L | TODO |
| P11.7 | **Gate G11:** end-to-end paper cycle; injected mismatch caught; kill switch halts within one cycle | P11.5 | M | BLOCKED(B2) |
| P11.8 | Cost-model calibration from paper fills: fit half-spread/impact parameters against realized slippage, clear the `UNCALIBRATED` flag (closes G9's deferred clause), scheduled recalibration cadence; predicted-vs-realized feeds the 6.9 UI (P11.6). **IBKR paper fills are optimistic (fill at the touch, no queue position) — they are a LOWER BOUND on slippage, never an estimate; a documented haircut is mandatory (D-013)**. Post-gate task — needs accumulated fill history | P11.7 | L | BLOCKED(B2) |

## Phase 12 — Monitoring

| ID | Task | Depends | Cx | Status |
|---|---|---|---|---|
| P12.1 | Live-vs-expected vs CPCV distribution; auto-halt outside expected band | G11 | L | TODO |
| P12.2 | Feature drift via PSI | G10 | M | DONE (2026-08-02) — 252 tests, 17/17 mutations; see D-031 |
| P12.3 | Extraction-quality drift on golden set | G7 | M | TODO |
| P12.4 | Alerting: rules, delivery, acknowledgement, halt history | P12.1 | M | TODO |
| P12.5 | [UI] Monitoring page + finalize Overview (live-vs-expected w/ CPCV bands, drawdown vs limit, health tiles, alerts, next job, spend vs cap) | P12.1–P12.4 | L | TODO |
| P12.6 | **Gate G12:** injected drift detected; injected deviation triggers halt | P12.5 | M | TODO |

## Cross-cutting (CC) — slotted between phases

| ID | Task | After | Cx | Status |
|---|---|---|---|---|
| CC.0 | Crypto foundation: Fernet key encryption, KEK from env, log-redaction processor for key patterns (structlog processor + tests) | G1 | M | DONE |
| CC.1 | Immutable audit log + config-as-events: append-only table (who, when, field, old, new), every config write goes through it; versioned-config helper reused by features/extraction/settings | G2 | L | DONE |
| CC.2 | API hardening **before** the first mutating endpoint ships: CSRF protection, rate limiting on mutating routes, parameterized-queries-only lint check. P3.10's re-sync trigger is the first consumer and depends on this | P3.1 | M | DONE |
| CC.3 | MLflow service in compose + DVC init (data versioning for I2) | G2 | M | TODO |
| CC.4 | [UI] Settings & Audit page: config viewer, scheduler management, backup status, immutable audit trail browser | CC.1 | L | TODO |
| CC.5 | Playwright e2e harness + first critical-path test (loads dashboard, health visible); grows with each [UI] task | G1 | M | DONE |
| CC.10 | **Frontend unit-test runner + coverage measurement** — §8 requires ≥70% frontend coverage, and CC.5 does not close it: there is no unit-test runner in `frontend/` at all (no Vitest/Jest/Testing Library) and Playwright as configured emits no coverage number. Three e2e tests are not a coverage measurement. Needs a component-test runner or V8 coverage collection wired into the Playwright run, before CC.6's ratchet can honestly enforce the frontend half | CC.5 | M | DONE |
| CC.6 | Coverage ratchet: enforce ≥85% backend / ≥70% frontend in CI once each stack has meaningful surface (do not fake with trivial tests) | P2.8 | S | TODO |
| CC.7 | Test-honesty guard (I6) in CI: pytest runs with `--runxfail` and a junit-based check fails the build on any skipped test; extend to the frontend runner when frontend tests exist | G1 | S | DONE (backend side; frontend extension when tests exist) |
| CC.9 | **Database role separation** — migration-owner role owns schema and triggers; app role holds INSERT/SELECT only, so append-only cannot be disabled by the credential the app runs under. **Deadline: before Phase 11, and before the §6.11 audit log or `TESTING_LEDGER.md` integrity is presented as trustworthy** — 'immutable' is a false claim while the owning role can drop its own triggers, and DSR is only as honest as a ledger nobody can silently edit (D-017) | G10 | M | TODO |
| CC.8 | No-fabrication guard (I3) as code: connector base-class contract test — an unavailable source must raise, never return placeholder data; lint ban on mock/synthetic identifiers in `backend/ingest` production paths (`scripts/check_no_fabrication.py`, wired into pre-commit **and** CI) | P3.1 | M | DONE |

---

## Standing blockers (details in `BLOCKERS.md`)

- **B1** — Data-vendor selection + API keys (OHLCV w/ delistings, PIT fundamentals, transcripts, borrow). Human decision + credentials.
- **B2** — IBKR paper-account credentials + gateway hosting decision.
- **B3** — Golden-set human labels (300–500 docs). Harness can be built first; labels are human work.
- **B4** — LLM provider keys (2–3 cost-tier models) + approved spend caps.

## Session cadence

1. Session start: read state files; re-read `DIRECTIVE.md` §9.
2. Pick the lowest-numbered `TODO` task on the active track whose deps are met; if blocked, switch tracks (P7 scaffolding and CC tasks are the standing alternates).
3. Test-first; full suite before commit; one commit per task; `PROGRESS.md` entry per task.
4. Session end: state files current, `docker compose up` works, suite green, session summary appended to `PROGRESS.md`.
