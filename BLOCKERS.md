# BLOCKERS.md — Requires a human. Never guess past these (directive §0.2, §9.8)

Format: ID, opened date, what is blocked, what is needed, status.

---

## B1 — Data vendor selection + API keys — DECIDED, ACTION PENDING (2026-07-31)

**Blocks:** P3.3–P3.7, P3.11 (Gate G3); transitively P4–P6, P8–P12 full-data runs.
**Not blocked:** P3.1 (connector framework), P3.2 (EDGAR — keyless), P3.8 (FRED), all of Phase 2, Phase 7 scaffolding, CC tasks.

**Decision (operator, 2026-07-31): Sharadar via Nasdaq Data Link.**

| Feed | Table | Use |
|---|---|---|
| Fundamentals | SF1, **as-reported dimensions ARQ/ARY** | Point-in-time fundamentals (P3.5) |
| Prices | SEP | Daily OHLCV (P3.4) |
| Delisted | SFP | Survivorship-bias elimination (P3.4, P4) |
| Corporate actions | ACTIONS | Splits, dividends, delistings, ticker changes, M&A (P3.3) |

Rationale: the only genuinely point-in-time fundamental source at non-institutional
pricing. **EODHD and FMP were rejected** — both serve *restated* fundamentals, which is
disqualifying for the core feed (restated data is exactly the lookahead I1 exists to
prevent).

**Sourced elsewhere, deliberately not from Sharadar:**
- **SEC filings — direct from EDGAR**, free. Acceptance timestamps are in the daily
  index files (this is the `knowledge_time` source of record for filings, per D-011).
- **Borrow availability/rates — skip the vendor entirely**; take IBKR's feed in
  Phase 11 (P3.7 rescoped accordingly).

**⚠ Do before paying for a year — validate the PIT claim on trial data.** Vendors
describe things as point-in-time that are not. Reconstruct a fundamental snapshot for a
date *before a known restatement* and confirm the system returns the **original**
figures, not the restated ones. This is the Phase 3 gate (G3) run against trial data,
and it is the go/no-go for the annual spend. **Do not purchase before this passes.**

**⚠ Check first — WRDS.** If any university affiliation is available, WRDS access to
**CRSP/Compustat** changes this decision entirely: academic-grade PIT, and it is the
reference standard the Phase 3 gate should validate against *regardless* of which feed
is bought. Establish this before committing to Sharadar.

**Still needed from the human:** confirmation on WRDS availability; Nasdaq Data Link
credentials as environment variables (never committed); approved annual spend after the
trial-data gate passes.

## B2 — IBKR paper account — ACTION STARTED, CREDENTIALS PENDING (2026-07-31)

**Blocks:** P11.1, P11.7 (Gate G11); cost-model calibration P11.8.
**Operator note:** a paper account requires a **completed live application first**, and
approval takes days — it is on the critical path for Phase 11, so the application is
being started now rather than at Phase 11.

**⚠ Spec correction (operator, 2026-07-31) — paper fills are a lower bound, not an
estimate.** IBKR paper fills are optimistic: they fill at the touch far more readily
than reality and model no queue position. Calibrating the cost model directly against
them **understates costs**, which is the single easiest way to turn a losing strategy
into a winning backtest. P11.8 must apply a documented haircut, or treat paper fills as
a *lower bound on slippage* rather than an estimate of it. See D-013.

**Still needed from the human:** paper-account credentials once approved; decision on
where IB Gateway runs. (The platform hard-codes paper endpoints only — directive §5-P11.)

## B3 — Golden-set human labels — PROTOCOL DECIDED, LABELLING PENDING (2026-07-31)

**Blocks:** P7.8 (labels), P7.12 (Gate G7).
**Not blocked:** the labelling harness itself (P7.8 harness), P7.2–P7.4, P7.6, P7.10.

**⚠ Spec correction (operator, 2026-07-31) — the 85% agreement gate is unjustifiable
until the human noise floor is measured.** An 85% threshold means nothing if the
labeller's own self-agreement is 82%; it would be measuring noise. See D-014.

**Labelling protocol (in order):**
1. Label **60 documents** as a pilot. The rubric *will* turn out ambiguous. Fix it.
2. A week later, **re-label 20 of those 60 blind** → **intra-rater agreement = the
   noise floor**.
3. **Set the model gate relative to that floor**, not to a pre-picked number.
4. Label the remainder against the corrected rubric.

**Method:** label **one construct at a time across all documents**, never all constructs
per document — consistency is materially higher that way. Budget **2–4 min per
judgment**; 300 documents ≈ **15–20 hours** of human time. **Start with a single
high-value construct** rather than four mediocre ones.

**Still needed from the human:** the pilot labels, then the blind re-label, then the
remainder. Labels must not be model-generated (I3 — a model-labelled "golden" set makes
the regression suite circular).

## B4 — LLM provider keys + spend caps — SIZED, KEYS PENDING (2026-07-31)

**Blocks:** P7.5, P7.7 live enforcement, P7.9, P7.12 (Gate G7).

**⚠ Load recomputed (operator, 2026-07-31).** Extraction is **event-driven, not daily** —
the earlier ~12k/day figure assumed a news feed that is explicitly out of scope (§1.1).

| | Volume | Cost |
|---|---|---|
| **Steady state** | ~500 names × 8 docs/yr × 5 chunks × 3 models ≈ **60k calls/year** | Negligible — a **$50/month** cap is generous |
| **Historical backfill** | 10 years × that volume ≈ **600k calls** | **~$1,000–1,500** at cost-tier pricing, depending on chunk size |

The backfill is the number to control. Levers: **universe size, history depth, ensemble
width**.

**⚠ Do not pay for the full backfill before Phase 8 proves incremental IC.** Pilot at
**200 names, 5 years, 2-model ensemble**. If the LLM features do not beat the Phase 5
baseline factors, the money is saved and the finding is identical. See D-015.

**Still needed from the human:** API keys for 2–3 **cost-tier** models (directive §5-P7
forbids frontier models for extraction); confirmation of the $50/month steady-state cap;
separate approval for the pilot backfill spend. The cost governor refuses to run without
configured caps.

---

*Resolved blockers move to a "Resolved" section below with resolution notes — never deleted.*
