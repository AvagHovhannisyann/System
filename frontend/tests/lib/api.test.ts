/**
 * Unit coverage for the backend API client (`lib/api.ts`).
 *
 * `fetchHealth` is the only place in the frontend where untyped wire JSON
 * becomes a typed value the dashboard is willing to display, so it is the
 * choke point for the project's honesty rule: the operator must never see a
 * status the client cannot vouch for. Every test below asserts one of two
 * properties —
 *
 *   1. a *valid* envelope is passed through unchanged, including the HTTP 503
 *      degraded envelope (DECISIONS.md D-005), which must be read as data
 *      rather than flattened into "backend unreachable"; and
 *   2. anything the type guard cannot fully verify is rejected outright, so a
 *      partially-understood payload can never reach the UI.
 *
 * The network is stubbed at `globalThis.fetch`; no backend, database or Redis
 * is required or contacted.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { HEALTH_DEGRADED_REDIS_DOWN, HEALTH_OK } from "@/tests/fixtures/health";
import { API_BASE_URL, ApiError, fetchHealth } from "@/lib/api";

const HEALTH_URL = `${API_BASE_URL}/api/health`;

const fetchMock = vi.fn<typeof fetch>();

/** Build a real `Response` so the code under test parses a genuine body. */
function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("fetchHealth — valid envelopes", () => {
  it("returns the parsed payload on HTTP 200 and requests the health endpoint uncached", async () => {
    fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK));

    await expect(fetchHealth()).resolves.toEqual(HEALTH_OK);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(HEALTH_URL);
    // A cached health read is a stale health read; the operator would be
    // looking at a status that is no longer true.
    expect(init?.cache).toBe("no-store");
    expect(init?.signal).toBeInstanceOf(AbortSignal);
  });

  it("treats an HTTP 503 health envelope as data, not as a transport failure (D-005)", async () => {
    fetchMock.mockResolvedValue(jsonResponse(HEALTH_DEGRADED_REDIS_DOWN, 503));

    const health = await fetchHealth();

    // The whole point of D-005: the 503 still carries per-component truth, and
    // discarding it would downgrade "Redis is down" into "we have no idea".
    expect(health).toEqual(HEALTH_DEGRADED_REDIS_DOWN);
    expect(health.status).toBe("degraded");
    expect(health.components.redis.up).toBe(false);
    expect(health.components.db.up).toBe(true);
  });

  it("targets the backend named by NEXT_PUBLIC_API_URL", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://backend.unit.invalid");
    vi.resetModules();

    const api = await import("@/lib/api");
    expect(api.API_BASE_URL).toBe("https://backend.unit.invalid");

    fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK));
    await api.fetchHealth();

    expect(fetchMock).toHaveBeenCalledWith(
      "https://backend.unit.invalid/api/health",
      expect.anything(),
    );
  });
});

describe("fetchHealth — transport failures", () => {
  it("reports the configured backend and the underlying cause when the host is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    const error = await fetchHealth().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBeNull();
    // The operator has to be able to tell *which* backend was not answering.
    expect((error as ApiError).message).toContain(API_BASE_URL);
    expect((error as ApiError).message).toContain("fetch failed");
  });

  it("surfaces a non-Error rejection instead of swallowing it", async () => {
    fetchMock.mockRejectedValue("socket hang up");

    await expect(fetchHealth()).rejects.toThrow(/socket hang up/);
  });

  it("gives up on a backend that never answers, rather than hanging forever", async () => {
    // The stub honours the abort signal exactly as a real fetch does, so this
    // exercises the genuine `AbortSignal.timeout(5_000)` in `lib/api.ts`. If
    // that timeout were removed the promise would never settle and this test
    // would fail on its own 15s budget instead of passing quietly.
    fetchMock.mockImplementation(
      (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(init.signal?.reason as Error);
          });
        }),
    );

    const startedAt = Date.now();
    const error = await fetchHealth().catch((cause: unknown) => cause);
    const elapsedMs = Date.now() - startedAt;

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toMatch(/timeout/i);
    // Proves the abort came from the 5s deadline and not from an immediate
    // rejection that would make the assertion above vacuous.
    expect(elapsedMs).toBeGreaterThanOrEqual(4_000);
  }, 15_000);

  it("rejects an HTTP status that is neither 200 nor the 503 health envelope", async () => {
    for (const status of [400, 404, 418, 500, 502] as const) {
      fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK, status));

      const error = await fetchHealth().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(status);
      expect((error as ApiError).message).toContain(`HTTP ${status}`);
    }
  });

  it("rejects a body that is not JSON at all", async () => {
    fetchMock.mockResolvedValue(
      new Response("<html>502 Bad Gateway</html>", {
        status: 200,
        headers: { "content-type": "text/html" },
      }),
    );

    const error = await fetchHealth().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain("non-JSON body");
    expect((error as ApiError).status).toBe(200);
  });
});

describe("fetchHealth — type-guard boundary", () => {
  /**
   * Payloads that are syntactically valid JSON but that the client cannot
   * fully verify. Each must be rejected: a health envelope that is only
   * partly understood is exactly the input that would let the dashboard
   * render a confident status it has no evidence for.
   */
  const malformed: ReadonlyArray<readonly [string, unknown]> = [
    ["null", null],
    ["a JSON string", "ok"],
    ["a number", 200],
    ["an array", [HEALTH_OK]],
    ["an empty object", {}],
    [
      "an unrecognised status value",
      { ...HEALTH_OK, status: "healthy" },
    ],
    ["a missing status", { version: "1", components: HEALTH_OK.components }],
    ["a non-string version", { ...HEALTH_OK, version: 1 }],
    ["missing components", { status: "ok", version: "1" }],
    [
      "components as a JSON string",
      { ...HEALTH_OK, components: "db=up,redis=up" },
    ],
    [
      "a missing redis component",
      { ...HEALTH_OK, components: { db: HEALTH_OK.components.db } },
    ],
    [
      "a missing db component",
      { ...HEALTH_OK, components: { redis: HEALTH_OK.components.redis } },
    ],
    [
      // The dangerous one: a truthy string would read as "up" under a loose
      // check, so a down component could be displayed as healthy.
      "an `up` flag sent as the string \"false\"",
      {
        ...HEALTH_OK,
        components: {
          db: { up: "false", latency_ms: 1.4 },
          redis: HEALTH_OK.components.redis,
        },
      },
    ],
    [
      "a latency sent as a string",
      {
        ...HEALTH_OK,
        components: {
          db: HEALTH_OK.components.db,
          redis: { up: true, latency_ms: "0.7" },
        },
      },
    ],
    [
      "a component missing its latency",
      {
        ...HEALTH_OK,
        components: {
          db: HEALTH_OK.components.db,
          redis: { up: true },
        },
      },
    ],
    [
      "a null component",
      {
        ...HEALTH_OK,
        components: { db: HEALTH_OK.components.db, redis: null },
      },
    ],
  ];

  it.each(malformed)(
    "rejects a 200 response whose body is %s",
    async (_description, payload) => {
      fetchMock.mockResolvedValue(jsonResponse(payload));

      const error = await fetchHealth().catch((cause: unknown) => cause);

      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toContain("unexpected payload shape");
      expect((error as ApiError).status).toBe(200);
    },
  );

  it("applies the same validation to the 503 envelope, leaving no unchecked path", async () => {
    // 503 is the one non-2xx status the client accepts as data; it must not
    // become a hole through which an unvalidated body reaches the UI.
    fetchMock.mockResolvedValue(
      jsonResponse({ status: "degraded", version: 7 }, 503),
    );

    const error = await fetchHealth().catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain("unexpected payload shape");
    expect((error as ApiError).status).toBe(503);
  });

  it("accepts a valid envelope carrying extra unknown fields", async () => {
    // Forward compatibility: a backend that adds a field must not take the
    // dashboard down. Only the fields the client relies on are enforced.
    const withExtras = {
      ...HEALTH_OK,
      uptime_s: 1234,
      components: {
        ...HEALTH_OK.components,
        celery: { up: true, latency_ms: 3.2 },
      },
    };
    fetchMock.mockResolvedValue(jsonResponse(withExtras));

    const health = await fetchHealth();

    expect(health.status).toBe("ok");
    expect(health.components.db).toEqual(HEALTH_OK.components.db);
  });
});

describe("ApiError", () => {
  it("is a named Error carrying an optional HTTP status", () => {
    const withStatus = new ApiError("boom", 503);
    expect(withStatus).toBeInstanceOf(Error);
    expect(withStatus.name).toBe("ApiError");
    expect(withStatus.message).toBe("boom");
    expect(withStatus.status).toBe(503);

    // Transport failures have no HTTP status at all — `null`, never a
    // plausible-looking stand-in such as 0 or 500.
    expect(new ApiError("boom").status).toBeNull();
  });
});
