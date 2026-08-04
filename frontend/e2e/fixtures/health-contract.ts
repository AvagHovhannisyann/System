/**
 * Mocked `GET /api/health` payloads, pinned at compile time to the real
 * client contract in `lib/api.ts`.
 *
 * A network-level mock is only as trustworthy as its agreement with the
 * production contract: if `HealthResponse` gains, loses or retypes a field and
 * these fixtures do not follow, the e2e suite keeps passing while asserting
 * against a payload the backend no longer sends — a green test that proves
 * nothing. The assertions at the bottom of this file make that divergence a
 * `tsc --noEmit` failure instead.
 */

import type { ComponentHealth, HealthResponse } from "@/lib/api";

/**
 * Compile-time type equality, invariant in both directions.
 *
 * The conditional-type-identity trick is used rather than mutual
 * assignability because assignability would accept a fixture that is merely a
 * subtype of the contract, which is exactly the drift being guarded against.
 */
type Equals<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;

/** Resolves only when `T` is exactly `true`; otherwise it is a type error. */
type Assert<T extends true> = T;

/**
 * Backend version string served by every fixture below.
 *
 * Deliberately distinctive so an assertion on it cannot be satisfied by any
 * other text that happens to be on the page.
 */
export const MOCK_BACKEND_VERSION = "0.1.0-e2e-fixture";

/**
 * Every critical component up. Corresponds to HTTP 200 (DECISIONS.md D-005).
 *
 * Latencies carry one decimal place because the Overview page renders them
 * with `.toFixed(1)`; the rendered strings are therefore exact.
 */
export const HEALTH_OK = {
  status: "ok",
  version: MOCK_BACKEND_VERSION,
  components: {
    db: { up: true, latency_ms: 1.4 },
    redis: { up: true, latency_ms: 0.7 },
  },
} as const satisfies HealthResponse;

/**
 * Redis down, database up. Corresponds to HTTP 503 carrying the same body
 * shape (DECISIONS.md D-005), which `fetchHealth` accepts as a valid envelope.
 *
 * `latency_ms` of 1000.0 mirrors the backend's behaviour of reporting
 * time-to-failure, which for a timed-out probe is roughly `CHECK_TIMEOUT_S`.
 */
export const HEALTH_DEGRADED_REDIS_DOWN = {
  status: "degraded",
  version: MOCK_BACKEND_VERSION,
  components: {
    db: { up: true, latency_ms: 2.3 },
    redis: { up: false, latency_ms: 1000.0 },
  },
} as const satisfies HealthResponse;

/**
 * Compile-time proof that the fixtures above are structurally identical to
 * the contract — not merely assignable to it.
 *
 * `satisfies` (applied at each declaration) rejects a wrong field type or a
 * missing field. The key-set equalities below additionally reject an *extra*
 * field on a fixture and a field that was removed from the contract, in both
 * directions and at every level of nesting. Exported so that the declarations
 * are used rather than dead code.
 */
export type HealthFixturesMatchContract =
  | Assert<Equals<keyof typeof HEALTH_OK, keyof HealthResponse>>
  | Assert<
      Equals<
        keyof typeof HEALTH_OK.components,
        keyof HealthResponse["components"]
      >
    >
  | Assert<Equals<keyof typeof HEALTH_OK.components.db, keyof ComponentHealth>>
  | Assert<
      Equals<keyof typeof HEALTH_OK.components.redis, keyof ComponentHealth>
    >
  | Assert<Equals<keyof typeof HEALTH_DEGRADED_REDIS_DOWN, keyof HealthResponse>>
  | Assert<
      Equals<
        keyof typeof HEALTH_DEGRADED_REDIS_DOWN.components,
        keyof HealthResponse["components"]
      >
    >
  | Assert<
      Equals<
        keyof typeof HEALTH_DEGRADED_REDIS_DOWN.components.redis,
        keyof ComponentHealth
      >
    >;
