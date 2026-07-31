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
