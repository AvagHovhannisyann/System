/**
 * API client for the platform backend.
 *
 * The base URL comes from NEXT_PUBLIC_API_URL (inlined at build time),
 * defaulting to the local backend dev address.
 */

export const API_BASE_URL: string =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Overall system status as reported by the backend health endpoint. */
export type HealthStatus = "ok" | "degraded";

/** Health of a single backend component (latency in milliseconds). */
export interface ComponentHealth {
  up: boolean;
  latency_ms: number;
}

/**
 * Payload of `GET /api/health`.
 *
 * Per DECISIONS.md D-005 the backend returns HTTP 200 when every critical
 * component is up and HTTP 503 with the *same body* when any is down, so a
 * 503 here still carries a valid, parseable payload.
 */
export interface HealthResponse {
  status: HealthStatus;
  version: string;
  components: {
    db: ComponentHealth;
    redis: ComponentHealth;
  };
}

/** Error thrown when the backend cannot be reached or answers malformed data. */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isComponentHealth(value: unknown): value is ComponentHealth {
  return (
    isRecord(value) &&
    typeof value.up === "boolean" &&
    typeof value.latency_ms === "number"
  );
}

/**
 * Validated boundary between the untyped wire payload and the typed client.
 * This is the single place where unknown JSON is checked and narrowed.
 */
function isHealthResponse(value: unknown): value is HealthResponse {
  if (!isRecord(value)) return false;
  if (value.status !== "ok" && value.status !== "degraded") return false;
  if (typeof value.version !== "string") return false;
  if (!isRecord(value.components)) return false;
  return (
    isComponentHealth(value.components.db) &&
    isComponentHealth(value.components.redis)
  );
}

/**
 * Fetch backend health from `GET /api/health`.
 *
 * Treats HTTP 200 and 503 as valid health envelopes (D-005); any other HTTP
 * status, a network failure, a timeout, or a malformed body raises ApiError.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    throw new ApiError(`Backend unreachable at ${API_BASE_URL}: ${detail}`);
  }

  if (!response.ok && response.status !== 503) {
    throw new ApiError(
      `Health endpoint answered HTTP ${response.status}`,
      response.status,
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError(
      "Health endpoint returned a non-JSON body",
      response.status,
    );
  }

  if (!isHealthResponse(payload)) {
    throw new ApiError(
      "Health endpoint returned an unexpected payload shape",
      response.status,
    );
  }

  return payload;
}
