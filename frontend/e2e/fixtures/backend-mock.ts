/**
 * Network-layer mock of the platform backend.
 *
 * Every backend response the dashboard sees in an e2e run is produced here,
 * via Playwright request interception. Nothing is stubbed inside the app: the
 * real `lib/api.ts` client, the real TanStack Query cache and the real page
 * components all run unmodified, and only the wire is substituted. That keeps
 * the tests deterministic without a live Postgres, Redis or FastAPI process,
 * and it means a regression in the client's parsing or error handling still
 * fails the suite.
 *
 * Adding a page: give `BackendMock` one typed helper per endpoint that page
 * calls (modelled on `health*` below) and reuse `fulfilJson`. Nothing else in
 * the harness needs to change.
 */

import type { Page, Route } from "@playwright/test";

import type { HealthResponse } from "@/lib/api";
import { E2E_API_GLOB, E2E_HEALTH_URL } from "../env";

/** HTTP status the backend uses when a critical component is down (D-005). */
export const HTTP_SERVICE_UNAVAILABLE = 503;

export class BackendMock {
  /**
   * Requests that reached the backend origin with no explicit mock installed.
   *
   * Asserted empty after every test. A non-empty list means the app talked to
   * an endpoint the test did not control, so whatever the test observed was
   * not fully determined by the fixture.
   */
  readonly unhandled: string[] = [];

  /** URLs of health requests actually served by this mock, in order. */
  readonly healthRequests: string[] = [];

  private constructor(private readonly page: Page) {}

  /**
   * Attach the mock to a page and install the catch-all guard.
   *
   * The guard is registered first on purpose: Playwright matches route
   * handlers in reverse registration order, so any endpoint mock installed
   * later by a test takes precedence, and everything else is refused and
   * recorded rather than escaping to the network.
   */
  static async install(page: Page): Promise<BackendMock> {
    const mock = new BackendMock(page);
    await page.route(E2E_API_GLOB, async (route: Route) => {
      mock.unhandled.push(route.request().url());
      await route.abort("failed");
    });
    return mock;
  }

  /** Serve `GET /api/health` with a healthy envelope (HTTP 200). */
  async healthOk(payload: HealthResponse): Promise<void> {
    await this.serveHealth(payload, 200);
  }

  /**
   * Serve `GET /api/health` with a degraded envelope.
   *
   * Uses HTTP 503 with a full, valid body, exactly as the backend does when a
   * component is down (D-005) — `fetchHealth` treats 503 as a valid health
   * envelope rather than an error, and this is the only way to test that.
   */
  async healthDegraded(payload: HealthResponse): Promise<void> {
    await this.serveHealth(payload, HTTP_SERVICE_UNAVAILABLE);
  }

  /**
   * Refuse every connection to `GET /api/health`.
   *
   * Models an entirely unreachable backend (process down, wrong host, DNS
   * failure), which surfaces in the client as an `ApiError` rather than as a
   * parsed payload.
   */
  async healthUnreachable(): Promise<void> {
    await this.page.route(E2E_HEALTH_URL, async (route: Route) => {
      this.healthRequests.push(route.request().url());
      await route.abort("connectionrefused");
    });
  }

  private async serveHealth(
    payload: HealthResponse,
    status: number,
  ): Promise<void> {
    await this.page.route(E2E_HEALTH_URL, async (route: Route) => {
      this.healthRequests.push(route.request().url());
      await fulfilJson(route, payload, status);
    });
  }
}

/** Fulfil a route with a JSON body, matching how FastAPI answers. */
async function fulfilJson(
  route: Route,
  payload: unknown,
  status: number,
): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}
