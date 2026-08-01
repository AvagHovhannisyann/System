/**
 * Component coverage for the Overview page's System Health card
 * (DIRECTIVE.md section 6.1).
 *
 * The single property under test is that the card cannot claim a health state
 * it has no evidence for. Every test therefore asserts the *complete* set of
 * status badges the card renders, not merely that the expected one is
 * present — an assertion of the form "the word degraded appears somewhere"
 * would still pass if the card simultaneously displayed a reassuring "ok".
 *
 * Only `globalThis.fetch` is stubbed, so these tests drive the real
 * `lib/api.ts` parsing and the real TanStack Query wiring in jsdom. The
 * Playwright suite proves the same property in a real browser against a real
 * Next.js server; this suite is the fast, exhaustive counterpart and covers
 * states the browser suite does not reach (the loading skeleton, and a
 * well-formed HTTP 200 carrying a malformed body).
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OverviewPage from "@/app/page";
import { API_BASE_URL } from "@/lib/api";
import {
  FIXTURE_BACKEND_VERSION,
  HEALTH_DEGRADED_REDIS_DOWN,
  HEALTH_OK,
  renderedLatency,
} from "@/tests/fixtures/health";

const fetchMock = vi.fn<typeof fetch>();

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/**
 * Render the page with retries disabled.
 *
 * Production retries once (see `components/providers.tsx`); here a failure
 * must surface on the first attempt so the error-state assertions describe a
 * settled page rather than an intermediate one.
 */
function renderOverview(): QueryClient {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <OverviewPage />
    </QueryClientProvider>,
  );
  return queryClient;
}

/** The System Health card, located the way the e2e page object locates it. */
function healthCard(): HTMLElement {
  const title = screen.getByText("System Health");
  const card = title.closest('[data-slot="card"]');
  if (card === null) {
    throw new Error("System Health card not found");
  }
  return card as HTMLElement;
}

/** Text of every status badge in the card, in document order. */
function statusBadges(): string[] {
  return Array.from(
    healthCard().querySelectorAll('[data-slot="badge"]'),
  ).map((badge) => badge.textContent ?? "");
}

/** The row rendered for a named component, or null when absent. */
function componentRow(label: string): HTMLElement | null {
  const cell = within(healthCard()).queryByText(label);
  return cell === null ? null : (cell.parentElement as HTMLElement);
}

function skeletonCount(): number {
  return healthCard().querySelectorAll('[data-slot="skeleton"]').length;
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Overview page — loading state", () => {
  it("shows a skeleton and claims no health at all while the request is in flight", () => {
    // Never settles: the page stays in its pending state for the whole test.
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));

    renderOverview();

    expect(screen.getByRole("heading", { name: "Overview", level: 1 })).toBeInTheDocument();
    expect(skeletonCount()).toBeGreaterThan(0);

    // The assertion that matters: "loading" must not be dressed up as a
    // status. No badge, no component row and no version may appear before the
    // backend has actually answered.
    expect(statusBadges()).toEqual([]);
    expect(componentRow("Database")).toBeNull();
    expect(componentRow("Redis")).toBeNull();
    expect(screen.queryByText(/^Backend version /)).toBeNull();
  });
});

describe("Overview page — healthy backend", () => {
  it("renders exactly the statuses the backend reported", async () => {
    fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK));

    renderOverview();

    await waitFor(() => expect(skeletonCount()).toBe(0));

    expect(statusBadges()).toEqual(["ok", "up", "up"]);

    const overall = within(healthCard()).getByText("Overall").parentElement;
    expect(
      overall?.querySelector('[data-slot="badge"]'),
    ).toHaveAttribute("data-variant", "secondary");

    expect(componentRow("Database")).toHaveTextContent(
      renderedLatency(HEALTH_OK.components.db.latency_ms),
    );
    expect(componentRow("Redis")).toHaveTextContent(
      renderedLatency(HEALTH_OK.components.redis.latency_ms),
    );

    expect(
      screen.getByText(`Backend version ${FIXTURE_BACKEND_VERSION}`),
    ).toBeInTheDocument();

    // Rendering health without asking for it would be fabrication (I3).
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/health`,
      expect.anything(),
    );
  });
});

describe("Overview page — degraded backend", () => {
  it("shows the down component as down and never as healthy (HTTP 503 envelope)", async () => {
    fetchMock.mockResolvedValue(jsonResponse(HEALTH_DEGRADED_REDIS_DOWN, 503));

    renderOverview();

    await waitFor(() =>
      expect(statusBadges()).toEqual(["degraded", "up", "down"]),
    );

    // Nothing anywhere in the card says the system is fine.
    expect(within(healthCard()).queryByText("ok")).toBeNull();

    const redisBadge = componentRow("Redis")?.querySelector(
      '[data-slot="badge"]',
    );
    expect(redisBadge).toHaveTextContent("down");
    expect(redisBadge).toHaveAttribute("data-variant", "destructive");

    const overallBadge = within(healthCard())
      .getByText("Overall")
      .parentElement?.querySelector('[data-slot="badge"]');
    expect(overallBadge).toHaveTextContent("degraded");
    expect(overallBadge).toHaveAttribute("data-variant", "destructive");

    // A degraded backend is still answering, so its version is still known.
    expect(
      screen.getByText(`Backend version ${FIXTURE_BACKEND_VERSION}`),
    ).toBeInTheDocument();
  });
});

describe("Overview page — unreachable backend", () => {
  it("reports the failure and invents no component status", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    renderOverview();

    await waitFor(() => expect(statusBadges()).toEqual(["unreachable"]));

    expect(
      screen.getByText(`Could not reach the backend at ${API_BASE_URL}.`),
    ).toBeInTheDocument();
    expect(
      screen.getByText(new RegExp(`Backend unreachable at ${API_BASE_URL}`)),
    ).toBeInTheDocument();

    expect(componentRow("Database")).toBeNull();
    expect(componentRow("Redis")).toBeNull();
    expect(screen.queryByText(/^Backend version /)).toBeNull();
    expect(skeletonCount()).toBe(0);
  });

  it("treats a malformed 200 body as a failure rather than displaying a guess", async () => {
    // HTTP 200, valid JSON, wrong shape: the tempting failure mode is to show
    // a partially-parsed card. `lib/api.ts` rejects it and the page must show
    // its error state instead of a fabricated status.
    fetchMock.mockResolvedValue(
      jsonResponse({ status: "ok", version: 3, components: {} }),
    );

    renderOverview();

    await waitFor(() => expect(statusBadges()).toEqual(["unreachable"]));

    expect(
      screen.getByText(/unexpected payload shape/),
    ).toBeInTheDocument();
    expect(componentRow("Database")).toBeNull();
    expect(componentRow("Redis")).toBeNull();
  });
});

describe("Overview page — refresh", () => {
  it("drops a previously healthy display once the backend stops answering", async () => {
    // Staleness is the subtle dishonesty: a card that keeps showing the last
    // good read after the backend dies is worse than one that shows nothing,
    // because the operator has no way to tell the difference.
    fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK));

    const queryClient = renderOverview();
    await waitFor(() => expect(statusBadges()).toEqual(["ok", "up", "up"]));

    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    await queryClient.refetchQueries({ queryKey: ["health"] });

    await waitFor(() => expect(statusBadges()).toEqual(["unreachable"]));
    expect(screen.queryByText(/^Backend version /)).toBeNull();
  });

  it("actually re-polls at the interval it advertises to the operator", async () => {
    // The card tells the operator how fresh its data is ("polled every 10s").
    // That claim is only true if the poll fires on that period, so the
    // interval is read back off the rendered text and then used to drive the
    // clock: a card that advertised 10s but polled every 60s — or stopped
    // polling altogether and froze on a stale status — fails here.
    vi.useFakeTimers();
    try {
      fetchMock.mockResolvedValue(jsonResponse(HEALTH_OK));

      renderOverview();

      const description = screen.getByText(
        /Backend component status from \/api\/health, polled every/,
      ).textContent;
      const advertisedSeconds = Number(
        /polled every (\d+)s$/.exec(description ?? "")?.[1],
      );
      expect(advertisedSeconds).toBeGreaterThan(0);

      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

      // Just short of the advertised period: no extra request yet.
      await vi.advanceTimersByTimeAsync(advertisedSeconds * 1_000 - 100);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(100);
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

      await vi.advanceTimersByTimeAsync(advertisedSeconds * 1_000);
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    } finally {
      vi.useRealTimers();
    }
  });
});
