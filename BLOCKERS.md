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

## B5 — CORRESP/UPLOAD knowledge-time policy — OPEN (2026-08-01)

**Blocks:** ingesting SEC correspondence forms (`CORRESP`, `UPLOAD`). Not in the
connector's default form selection, so nothing else is blocked today.
**The problem, found empirically during P3.2:** accession `0000950170-24-012183` was
**accepted 2024-02-07** but only **disseminated in the 2024-03-11 index**. For these
forms acceptance is therefore *earlier* than knowability — using the acceptance
timestamp as `knowledge_time` would be **anti-conservative**, making the filing appear
knowable roughly a month before the market could see it. That is a lookahead bug of
exactly the kind I1 exists to prevent, and it is invisible unless someone looks.
**Not resolved in code.** The connector measures the divergence instead of guessing at
it (`filings_accepted_before_index_date`, `max_acceptance_to_index_lag` in the DQ
report), so the size of the effect is observable before a policy is chosen.
**Needed from the human:** a policy decision before these forms are ingested — most
likely `knowledge_time = dissemination/index date` for correspondence forms, but that
should be a deliberate choice recorded in `DECISIONS.md`, not a silent default.

## B6 — Dashboard UI design is routed away from this agent — OPEN (2026-08-01)

**Blocks:** every `[UI]` task — P3.10, **P4.3**, **P5.5**, P7.11, P8.5, P10.7, P11.6,
P12.5, CC.4. Nine pages; only the Overview page exists.
**Why this is a blocker and not a task:** the project instructions in `CLAUDE.md` route
all website/UI/UX design — layout, visual wireframes, mockups, styling — to **Fabel**,
and forbid this agent from attempting them. That is a standing routing rule, not a
capacity limit, so no amount of build time here clears it.
**Consequence for the phase gates.** Two gates name a UI artefact in their binary
condition: **G4** requires size/turnover history *rendered and manually inspected*, and
**G5** requires the correlation matrix reviewed. The backend data those pages consume is
being built now (P4.2 supplies the waterfall and size/turnover series; P5.3/P5.4 supply
the correlation and premia data), so the gates are reachable the moment the pages exist.
Until then G4 and G5 can only be *partially* evidenced — the programmatic half (tests,
computed series) passes; the human-inspection half cannot be signed off. **Neither gate
may be marked GATE-PASSED on the backend half alone** (directive §9.10).
**Needed from the human:** hand the nine `[UI]` tasks to Fabel, or explicitly re-scope
the two gates' inspection clauses. The backend exposes the data through the API either
way, so the two tracks are independent and can proceed in parallel.

### B1 escalation (2026-08-02) — it now blocks Phase 4 *entirely*, not just full-data runs

P4.1 landed and the scope of B1 is wider than the original entry says. `UniverseCriteria`
makes a strictly positive **market-cap floor mandatory**, and the shares-outstanding source
is the Sharadar SF1 feed nobody has purchased yet. So **every `build_universe` call refuses
before reading anything**, with `UniverseInputUnavailableError(blocker="B1")`. Phase 4's
code and its 224 tests are complete; Phase 4 cannot produce a single universe.

The borrow screen refuses on **B2** (no locate feed; B1's decision rescoped borrow to IBKR
in Phase 11), so `require_borrow=True` is unavailable for the same structural reason.

**G4 now needs three things, not one:** (a) B1 resolved so a universe can be built at all;
(b) a Docker daemon, so `backend/tests/integration/test_universe_db.py` (20 tests, written
and *not* skipped) can actually run — migration 0011's runtime behaviour is currently
unverified, only its structural match to the ORM; (c) B6's UI half, since G4's binary
condition names size/turnover history *rendered and manually inspected*.
`UniverseHistory.report()` supplies the programmatic half.

### B1 and G5's first clause — a concrete five-item checklist (2026-08-02)

P5.4 landed the premia validation harness, so what G5's first clause ("known factor premia
reproduce — sign and plausible magnitude") still needs is now enumerable rather than vague.
The harness encodes it as `WHAT_G5_STILL_NEEDS`:

1. **Price history in `price_bar`** (P3.4, B1). The three price factors compute but return
   NaN for every security, because the table is empty.
2. **Point-in-time fundamentals** (P3.5, B1). The six fundamentals factors raise; no
   cross-section exists at any date.
3. **Point-in-time universe membership across the long sample** (Phase 4, itself blocked on
   B1 per D-026). This one is easy to overlook and matters most: a survivorship-biased
   universe would reproduce the literature *for the wrong reason*, and the reproduction
   would look like success.
4. **A portfolio-construction step** producing the decile long-short spread the published
   ranges are calibrated for, with a gross/net decision recorded per series (I4).
5. **Expectations for the three newest factors.** `factors_without_expectations()` returns
   `('amihud_illiquidity', 'short_interest', 'size')` today. **G5's first clause cannot pass
   while that tuple is non-empty**, by construction: `reproduces` is True only when every
   declared expectation was exercised. Suggested sources when P5.6 extends the table —
   Banz (1981) *JFE* 9(1) for size; Asquith, Pathak & Ritter (2005) *JFE* 78(2) and
   Boehmer, Huszár & Jordan (2010) for short interest; Amihud (2002) *Journal of Financial
   Markets* 5(1) for illiquidity.

Item 5 is not blocked on a human — it is ordinary work for P5.6. Items 1–4 are B1.

## B7 — Short-interest feed: no source selected — OPEN (2026-08-02)

**Blocks:** the `short_interest` factor (P5.3), and G5's coverage clause through it.
**Found by:** the P5.3 wave-2 track, which declined to borrow B1 for it.

**Why this is its own blocker and not part of B1.** B1 is DECIDED — Sharadar SF1 / SEP /
SFP / ACTIONS. **None of those four carries short interest.** There is no
`short_interest_report` table, **no Phase 3 connector task, and no prior BLOCKERS entry**.
The gap was unregistered until now, and the factor's error message says so explicitly
rather than reporting "blocked on B1", which would have quietly attached it to a decision
already taken and made it look owned. `require_short_interest_source` distinguishes three
states, most specific first: source unregistered → `ShortInterestSourceUnavailableError`;
short-interest feed present but fundamentals absent → the existing B1 error; both present →
`ShortInterestComputationNotWrittenError`, so the task cannot go missing once the data
arrives. Mutating the gate to report B1 first fails 7 tests.

**Source of record:** FINRA Rule 4560 — semi-monthly, settlement dates on the 15th and the
last business day, disseminated roughly eight business days after the settlement date.

**The temporal trap, which is the reason to be careful about the vendor choice.** The
settlement date is the obvious join key, the obvious `valid_from`, and **the obvious wrong
`knowledge_time`**. A connector that substitutes one for the other grants eight business
days of foresight twice a month, forever — and the resulting factor looks *better*: right
sign, right magnitude, healthy distribution, improved backtest. The declared 17-day lag
(14 publication + 1 intraday + 2 vendor redistribution) covers the whole gap rather than a
residual around it, deliberately. That margin is close to free here because the observation
is semi-monthly: extra days change *which* observation is read on a handful of dates rather
than discarding a bar of signal — the asymmetry that makes D-027 refuse a margin on daily
prices.

**Needed from the human:** pick a source (the FINRA file directly, or a vendor that
redistributes it), confirm whether its cost is acceptable, and add the connector as a Phase
3 task. Until then the factor refuses and G5's coverage clause stays open.

### D-017's deadline has passed (2026-08-02) — recorded so it cannot pass quietly twice

D-017 set a hard condition: **"Role separation ships before Phase 11."** P11.2 landed on
2026-08-02, so Phase 11 has started and **CC.9 was still TODO**. The deadline was missed.

Recorded here rather than only fixed, because the failure mode D-017 was written to prevent
is now live: append-only is enforced by triggers owned by the same role the application
connects as, and that role can `DISABLE`/`DROP` its own triggers. Every claim resting on
that is currently overstated —

- the §6.11 dashboard **states "immutable audit log" to the operator as fact**;
- `TESTING_LEDGER.md` integrity is what makes the Deflated Sharpe honest, and DSR is only as
  good as its trial count (§9.7). A ledger that can be silently edited makes the trial count
  unverifiable.

CC.9 is now first in the current wave. Until it lands, **neither claim should be presented
to an operator without this caveat attached** — which is the substance of what D-017 was
protecting, and the reason a backlog item was given a deadline in the first place.
