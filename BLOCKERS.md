# BLOCKERS.md — Requires a human. Never guess past these (directive §0.2, §9.8)

Format: ID, opened date, what is blocked, what is needed, status.

---

## B1 — Data vendor selection + API keys — OPEN (2026-07-31)

**Blocks:** P3.3–P3.7, P3.11 (Gate G3); transitively P4–P6, P8–P12 full-data runs.
**Needed from human:**
1. Choice of vendor(s) for: daily OHLCV **with delisted securities and full adjustment history**, point-in-time unrestated fundamentals with original report dates, earnings-call transcripts, borrow availability/rates. (Candidates worth evaluating when you decide: Sharadar/Nasdaq Data Link for OHLCV+fundamentals+corporate actions incl. delistings; alternatives: EODHD, Norgate, FMP. The PIT-fundamentals and delisting requirements disqualify most free sources — I3 forbids papering over this with synthetic data.)
2. API credentials for the chosen vendor(s), provided as environment variables (never committed).
3. Approved monthly data budget.
**Not blocked meanwhile:** SEC EDGAR (P3.2) and FRED macro (P3.8) are keyless (FRED's key is free-tier and still needed from you eventually for rate limits); connector framework (P3.1); all of Phase 2; Phase 7 scaffolding; CC tasks.

## B2 — IBKR paper account — OPEN (2026-07-31)

**Blocks:** P11.1, P11.7 (Gate G11); cost-model calibration part of G9.
**Needed from human:** IBKR paper-account credentials; decision on where IB Gateway runs (the platform hard-codes paper ports only, per directive §5-P11).

## B3 — Golden-set human labels — OPEN (2026-07-31)

**Blocks:** P7.8 (labels), P7.12 (Gate G7 agreement threshold).
**Needed from human:** 300–500 documents labelled by a human (or human-verified) for the chosen extraction tasks. The labelling UI/harness (P7.8) will be built first so labelling is as cheap as possible; the labels themselves must not be model-fabricated (I3 — a model-labelled "golden" set would make the regression suite circular).

## B4 — LLM provider keys + spend caps — OPEN (2026-07-31)

**Blocks:** P7.5, P7.7 live enforcement, P7.9, P7.12 (Gate G7).
**Needed from human:** API keys for 2–3 **cost-tier** models (directive §5-P7 forbids frontier models for extraction), plus approved daily and monthly spend caps per provider (the cost governor refuses to run without configured caps).

---

*Resolved blockers move to a "Resolved" section below with resolution notes — never deleted.*
