/**
 * Critical-path end-to-end coverage for the Overview page
 * (DIRECTIVE.md section 6.1, quality bar section 8).
 *
 * The Overview page is the operator's first read on whether the platform is
 * trustworthy right now, so the suite covers the truthful-failure paths as
 * carefully as the happy path. A dashboard that renders "ok" while a
 * component is down, or that keeps showing a stale healthy state when the
 * backend is unreachable, is precisely the dishonest display DECISIONS.md
 * D-005 exists to prevent — so each failure test asserts the *absence* of a
 * reassuring status, not merely the presence of a warning.
 *
 * The backend is mocked at the network layer (see `fixtures/backend-mock.ts`);
 * no live backend, database or Redis is required or contacted.
 */

import { E2E_API_BASE_URL } from "./env";
import {
  HEALTH_DEGRADED_REDIS_DOWN,
  HEALTH_OK,
  MOCK_BACKEND_VERSION,
} from "./fixtures/health-contract";
import { expect, test } from "./fixtures/test";
import { COMPONENT_LABELS } from "./pages/overview-page";

/** Render a fixture latency exactly as the page formats it (`.toFixed(1)`). */
function renderedLatency(latencyMs: number): string {
  return `${latencyMs.toFixed(1)} ms`;
}

test.describe("Overview page — System Health", () => {
  test("renders per-component status and backend version when all components are up", async ({
    page,
    backend,
    overviewPage,
  }) => {
    await backend.healthOk(HEALTH_OK);
    await overviewPage.goto();

    await expect(page).toHaveTitle("Quant Research Platform");
    await expect(overviewPage.heading).toBeVisible();

    await expect(overviewPage.healthCard).toBeVisible();
    await expect(overviewPage.healthCardTitle).toHaveText("System Health");
    await expect(overviewPage.loadingSkeleton).toHaveCount(0);

    await expect(overviewPage.overallStatusBadge).toHaveText(HEALTH_OK.status);

    await expect(
      overviewPage.componentStatusBadge(COMPONENT_LABELS.db),
    ).toHaveText("up");
    await expect(overviewPage.componentRow(COMPONENT_LABELS.db)).toContainText(
      renderedLatency(HEALTH_OK.components.db.latency_ms),
    );

    await expect(
      overviewPage.componentStatusBadge(COMPONENT_LABELS.redis),
    ).toHaveText("up");
    await expect(
      overviewPage.componentRow(COMPONENT_LABELS.redis),
    ).toContainText(renderedLatency(HEALTH_OK.components.redis.latency_ms));

    await expect(overviewPage.backendVersion).toHaveText(
      `Backend version ${MOCK_BACKEND_VERSION}`,
    );

    // The exact, complete set of status claims the page makes.
    await expect(overviewPage.allStatusBadges).toHaveText(["ok", "up", "up"]);

    expect(
      backend.healthRequests.length,
      "the page must actually call GET /api/health, not render from thin air",
    ).toBeGreaterThan(0);
  });

  test("shows a down component as down, not as healthy, when the backend reports degraded", async ({
    backend,
    overviewPage,
  }) => {
    // HTTP 503 carrying a full body is the real degraded envelope (D-005);
    // `fetchHealth` must treat it as data rather than as a transport error.
    await backend.healthDegraded(HEALTH_DEGRADED_REDIS_DOWN);
    await overviewPage.goto();

    await expect(overviewPage.healthCardTitle).toHaveText("System Health");

    await expect(overviewPage.overallStatusBadge).toHaveText("degraded");
    await expect(overviewPage.overallStatusBadge).toHaveAttribute(
      "data-variant",
      "destructive",
    );

    await expect(
      overviewPage.componentStatusBadge(COMPONENT_LABELS.db),
    ).toHaveText("up");

    const redisBadge = overviewPage.componentStatusBadge(
      COMPONENT_LABELS.redis,
    );
    await expect(redisBadge).toHaveText("down");
    await expect(redisBadge).toHaveAttribute("data-variant", "destructive");
    await expect(
      overviewPage.componentRow(COMPONENT_LABELS.redis),
    ).toContainText(
      renderedLatency(HEALTH_DEGRADED_REDIS_DOWN.components.redis.latency_ms),
    );

    // The version is still reported: a degraded backend is still answering.
    await expect(overviewPage.backendVersion).toHaveText(
      `Backend version ${MOCK_BACKEND_VERSION}`,
    );

    // Exactly one component is claimed down and nothing claims overall health.
    await expect(overviewPage.allStatusBadges).toHaveText([
      "degraded",
      "up",
      "down",
    ]);
  });

  test("reports the backend as unreachable and claims no component health when the connection is refused", async ({
    backend,
    overviewPage,
  }) => {
    await backend.healthUnreachable();
    await overviewPage.goto();

    await expect(overviewPage.healthCardTitle).toHaveText("System Health");
    await expect(overviewPage.unreachableNotice).toBeVisible();

    // Proves the app under test is pointed at the mocked origin, so the
    // absence of health data below is caused by the fixture and nothing else.
    await expect(overviewPage.healthCard).toContainText(
      `Could not reach the backend at ${E2E_API_BASE_URL}`,
    );
    await expect(overviewPage.healthCard).toContainText(
      `Backend unreachable at ${E2E_API_BASE_URL}`,
    );

    // The honest-failure assertions: no status is invented for anything.
    await expect(overviewPage.allStatusBadges).toHaveText(["unreachable"]);
    await expect(
      overviewPage.componentRow(COMPONENT_LABELS.db),
    ).toHaveCount(0);
    await expect(
      overviewPage.componentRow(COMPONENT_LABELS.redis),
    ).toHaveCount(0);
    await expect(overviewPage.backendVersion).toHaveCount(0);
    await expect(overviewPage.healthCard).not.toContainText(
      MOCK_BACKEND_VERSION,
    );
    await expect(overviewPage.loadingSkeleton).toHaveCount(0);

    expect(
      backend.healthRequests.length,
      "the page must have attempted the health call before reporting it unreachable",
    ).toBeGreaterThan(0);
  });
});
