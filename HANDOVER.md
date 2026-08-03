# HANDOVER.md — what is built, what is not, and what only you can do

**Status as of 2026-08-03.** Everything buildable without a human decision is built. P7.4 was the last such task and it landed on 2026-08-03.
Every remaining item is either blocked on you or is a `[UI]` page routed to Fabel.

Read `DIRECTIVE.md` for the rules this was built under, `PLAN.md` for task-level status,
`DECISIONS.md` for why things are the way they are, and `BLOCKERS.md` for the full text of
each blocker. This file is the short version.

---

## 1. The five things only you can do

Ordered by how much they unblock. **B1 is worth more than the other four combined.**

### B1 — Buy the market data · unblocks 4 gates

Decided already (D-015): **Sharadar via Nasdaq Data Link** — SF1 as-reported (ARQ/ARY),
SEP, SFP, ACTIONS. Not purchased.

Blocks P3.3–P3.6, P3.11, and therefore **G3, G4, G5, G6**, the entire predictor
(P8.2–P8.6), and **G9**. Phase 4 is built, tested, and **cannot produce a single
universe**: the market-cap floor is mandatory and its input does not exist, so
`build_universe` refuses before it reads anything (D-026). That refusal is correct
behaviour, not a bug.

> **Before you pay:** the operator decision on record is to validate point-in-time
> correctness on a trial first, and to check whether WRDS access is available through an
> institution — it may cover the same ground. That check has not been done.

### B2 — IBKR paper credentials · unblocks G11

Blocks P11.1 (broker adapter), P11.7, P11.8, and **G11**. Everything beneath the adapter
is built: order lifecycle, idempotency, reconciliation, kill switch, VWAP/TWAP slicing.
The system is **paper-only permanently and structurally** — no function in
`backend/execution/` accepts a `venue`, `broker`, `endpoint` or `live` parameter, asserted
by AST *and* `inspect.signature`, so there is nothing to point at a live endpoint (D-033).

### B3 — Label the golden set yourself · unblocks G7

**15–20 hours of your time, and nobody else's.** The protocol agreed (D-014): measure your
own **intra-rater agreement first** by re-labelling a sample blind, because the extraction
threshold is derived from that noise floor rather than from a made-up 85%. Label one
construct at a time.

### B4 — LLM API keys and spend caps · unblocks G7

2–3 **cost-tier** models (the directive forbids frontier models for extraction), plus a
confirmed monthly cap. The cost governor **refuses to run without configured caps and
prices** — deliberately: a cap enforced against a made-up price enforces nothing (D-032).
Blocks P7.5, P7.9, P7.12.

### B7 — Pick a short-interest source · unblocks G5's coverage clause

**New, and easy to miss: none of Sharadar's four feeds carries short interest.** No table,
no connector task, and no blocker entry existed until it was found. Source of record is
FINRA Rule 4560 — semi-monthly, published ~8 business days after the settlement date.

> The trap, if you pick a vendor: the settlement date is the obvious join key, the obvious
> `valid_from`, and **the obvious wrong `knowledge_time`**. A connector that substitutes one
> for the other grants eight business days of foresight twice a month, and the resulting
> factor looks *better* — right sign, right magnitude, better backtest.

### B8 — Decide how filing *text* is stored · new, has an I2 consequence

`edgar_filing_document` stores a **manifest and a URL, never a body**, so nothing in the
platform can hand an extraction task the text of a filing it just resolved. P7.4's delta
tasks work around it honestly (a caller-supplied text source; absent text yields
`NoComparisonPossible(PRIOR_TEXT_UNAVAILABLE)`, kept distinct from "the issuer filed
nothing"), but there is no shipped implementation.

The choice: **store bodies** (new table, fetch step, storage cost) or **fetch on demand**
(cheaper, but the text is then not point-in-time and an extraction cannot be reproduced from
the store alone — which collides with I2). A design decision, not a coding task.

### Also yours, smaller

- **B5** — a one-line ruling: for SEC `CORRESP`/`UPLOAD` forms, acceptance precedes
  dissemination by ~a month, so using acceptance as `knowledge_time` is *anti-conservative*.
  Most likely answer is "use the dissemination date", but it should be a recorded decision.
- **B6** — **hand the nine `[UI]` pages to Fabel.** `CLAUDE.md` routes all UI design away
  from this agent. **G4 and G5 both name a *rendered, manually inspected* artefact in their
  pass condition**, so neither can be signed off without them.

---

## 2. What is built

**4 of 12 gates passed.** ~4,700 tests. `mypy --strict` clean across 374 files. CI green on
all 8 checks.

| Gate | Status |
|---|---|
| **G1** stack | **PASSED** |
| **G2** bitemporal | **PASSED** — re-opened once when a depth suite found a live I1 leak, re-passed with evidence to nesting depth 14 |
| **G10** synthetic truth | **PASSED** — noise `t = −0.055` over 128 seeds; signal recovery *proportional* (2.559×/1.574× measured against 2.378×/1.562× predicted) |
| **G12** monitoring | **PASSED** — injected drift detected, injected deviation halts |
| G3 G4 G5 G6 G9 | blocked on **B1** |
| G7 | blocked on **B3 + B4** |
| G8 | blocked on **B1** via G5/G6 |
| G11 | blocked on **B2** |

Built and green: bitemporal store with default-deny at the SQL boundary; EDGAR and FRED
connectors; PIT universe construction; the feature framework with a hard 30-feature cap and
lag enforcement; twelve baseline factors; the premia validation harness; triple-barrier
labels with uniqueness weights; purged K-fold; Ledoit-Wolf covariance and a cvxpy optimizer;
the full validation stack (CPCV, DSR, PBO, walk-forward, synthetic truth); LLM extraction
with anonymization, caching, prompt versioning and a cost governor; order lifecycle,
reconciliation, kill switch and slicing; PSI drift, live-vs-expected halting and alerting;
database role separation.

---

## 3. Three things you should not misread

**A passing gate is narrower than it sounds.** G12 proves the pipeline reacts to a condition
it was *handed*. Its own measured power is **0.11 / 0.41 / 0.78** at 1σ/2σ/3σ — roughly
**2.27 years** to catch a 1σ decay at quarterly cadence. That is written into the gate's own
tests, not hidden in a footnote.

**A live result inside the expectation band is weak evidence.** The CPCV distribution it is
compared against was computed on the sample the strategy was selected on, so its centre is
biased upward. A result *below* the band is **stronger** evidence than the tail mass
suggests. `ExpectationComparison` deliberately has no `validated` or `passed` attribute so
this cannot be rendered as a green tick (D-031).

**Coverage is 88.20%, not the ~97% you may have seen.** The higher number had `backend/tests`
in the denominator. The gate's own config carries a literal `WHAT THIS GATE DOES NOT PROVE`
heading, asserted by test: an import with no assertions moves the number, and the properties
this project actually rests on — I1, append-only, no-alpha-in-noise — are asserted directly
and none would be caught by a coverage gate.

---

## 4. Two things unverified in the build environment

Neither is a defect; both are environmental and need a Docker daemon.

1. **Whether CC.3's MLflow compose service still lets G1 pass from a clean clone.** G1 is a
   passed gate and CC.3 changed the thing it tested. Run `docker compose up` from a fresh
   clone and confirm the health endpoint answers.
2. **TimescaleDB chunk ACL propagation** — that CC.9's per-hypertable grants reach their
   chunks. Migration 0015 grants per hypertable *by name* precisely because the `ALL TABLES`
   form may not propagate. CI's insert-and-read tests under the app role passed, which is
   good evidence, but it was not asserted directly.

---

## 5. If you do only one thing

**Resolve B1.** It is the difference between a platform that is provably correct on
synthetic data and one that can tell you something about markets. Four gates, the predictor,
and the entire question the project exists to answer are behind it.
