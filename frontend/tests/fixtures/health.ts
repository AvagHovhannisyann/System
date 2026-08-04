/**
 * `GET /api/health` payloads for the component/unit suite, pinned at compile
 * time to the production client contract in `lib/api.ts`.
 *
 * `satisfies HealthResponse` is what makes these fixtures trustworthy: if the
 * contract gains, loses or retypes a field and these do not follow, the build
 * fails at `tsc --noEmit` rather than the suite staying green while asserting
 * against a payload the backend no longer sends.
 *
 * The Playwright harness keeps its own copy (`e2e/fixtures/health-contract.ts`)
 * on purpose — the two suites run independently, and neither should be able to
 * break the other's fixtures by editing its own.
 */

import type { HealthResponse } from "@/lib/api";

/**
 * Backend version string served by every fixture below. Deliberately
 * distinctive so an assertion on it cannot be satisfied by other page text.
 */
export const FIXTURE_BACKEND_VERSION = "0.1.0-unit-fixture";

/**
 * Every critical component up — the HTTP 200 case (DECISIONS.md D-005).
 *
 * Latencies carry one decimal place because the Overview page renders them
 * with `.toFixed(1)`, so the expected strings below are exact rather than
 * rounded approximations.
 */
export const HEALTH_OK = {
  status: "ok",
  version: FIXTURE_BACKEND_VERSION,
  components: {
    db: { up: true, latency_ms: 1.4 },
    redis: { up: true, latency_ms: 0.7 },
  },
} as const satisfies HealthResponse;

/**
 * Redis down, database up — the HTTP 503 case, which per D-005 carries the
 * *same body shape* as a 200 and must therefore be read as data, not as a
 * transport failure.
 */
export const HEALTH_DEGRADED_REDIS_DOWN = {
  status: "degraded",
  version: FIXTURE_BACKEND_VERSION,
  components: {
    db: { up: true, latency_ms: 2.3 },
    redis: { up: false, latency_ms: 1000.0 },
  },
} as const satisfies HealthResponse;

/** Render a fixture latency exactly as the Overview page formats it. */
export function renderedLatency(latencyMs: number): string {
  return `${latencyMs.toFixed(1)} ms`;
}
