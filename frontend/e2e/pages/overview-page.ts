/**
 * Page object for the Overview dashboard page (DIRECTIVE.md section 6.1).
 *
 * Locators are derived from what the page already renders — accessible roles,
 * user-visible text and the `data-slot` / `data-variant` attributes that
 * shadcn/ui emits. Nothing here requires a change to the components, in
 * keeping with DECISIONS.md D-006 (visual design is delegated; this track is
 * functional testing only).
 */

import type { Locator, Page } from "@playwright/test";

/** Row labels the Overview page renders for each backend component. */
export const COMPONENT_LABELS = {
  db: "Database",
  redis: "Redis",
} as const;

export class OverviewPage {
  /** Route this page is served at. */
  static readonly path = "/";

  readonly heading: Locator;
  readonly healthCard: Locator;
  readonly healthCardTitle: Locator;
  readonly loadingSkeleton: Locator;
  readonly unreachableNotice: Locator;

  constructor(private readonly page: Page) {
    this.heading = page.getByRole("heading", { name: "Overview", level: 1 });
    this.healthCard = page
      .locator('[data-slot="card"]')
      .filter({
        has: page.locator('[data-slot="card-title"]', {
          hasText: "System Health",
        }),
      });
    this.healthCardTitle = this.healthCard.locator('[data-slot="card-title"]');
    this.loadingSkeleton = this.healthCard.locator('[data-slot="skeleton"]');
    this.unreachableNotice = this.healthCard.locator(
      '[data-slot="badge"]',
      { hasText: "unreachable" },
    );
  }

  async goto(): Promise<void> {
    await this.page.goto(OverviewPage.path);
  }

  /** The "Overall" row: the label plus the aggregate status badge. */
  get overallRow(): Locator {
    return this.healthCard.getByText("Overall", { exact: true }).locator("..");
  }

  /** The aggregate status badge ("ok" / "degraded"). */
  get overallStatusBadge(): Locator {
    return this.overallRow.locator('[data-slot="badge"]');
  }

  /** The row for one component, e.g. `componentRow("Database")`. */
  componentRow(label: string): Locator {
    return this.healthCard.getByText(label, { exact: true }).locator("..");
  }

  /** The up/down badge inside a component's row. */
  componentStatusBadge(label: string): Locator {
    return this.componentRow(label).locator('[data-slot="badge"]');
  }

  /** The paragraph reporting the backend version. */
  get backendVersion(): Locator {
    return this.healthCard.getByText(/^Backend version /);
  }

  /**
   * Every status badge rendered inside the health card.
   *
   * Used to assert the *absence* of a healthy-looking badge when the backend
   * is unreachable: the point of these tests is that the dashboard cannot
   * display a reassuring status it has no evidence for.
   */
  get allStatusBadges(): Locator {
    return this.healthCard.locator('[data-slot="badge"]');
  }
}
