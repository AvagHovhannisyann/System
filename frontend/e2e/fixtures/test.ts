/**
 * The e2e harness entry point: `import { test, expect } from "../fixtures/test"`.
 *
 * Two patterns are combined deliberately.
 *
 * 1. **Page objects** (`e2e/pages/*`) own every locator, so a markup change is
 *    a one-file fix rather than a sweep through specs, and specs read as
 *    operator behaviour rather than CSS.
 * 2. **A worker-agnostic `test.extend` fixture** installs the backend mock
 *    *before* any spec body runs and verifies afterwards that no request
 *    escaped it. Interception cannot be forgotten by a future spec author,
 *    which is the property that keeps this suite from silently depending on a
 *    live backend.
 *
 * Adding a dashboard page later: write `e2e/pages/<name>-page.ts`, add one
 * entry to `Fixtures` and one factory line below, and add typed endpoint
 * helpers to `BackendMock`. No spec, config or harness rework.
 */

import { test as base, expect } from "@playwright/test";

import { BackendMock } from "./backend-mock";
import { OverviewPage } from "../pages/overview-page";

interface Fixtures {
  /** Network-layer backend mock, pre-installed and auto-verified. */
  backend: BackendMock;
  /** Page object for the Overview page (DIRECTIVE.md section 6.1). */
  overviewPage: OverviewPage;
}

export const test = base.extend<Fixtures>({
  backend: async ({ page }, use) => {
    const backend = await BackendMock.install(page);

    await use(backend);

    expect(
      backend.unhandled,
      "the app issued a backend request that no mock handled, so this test was " +
        "not fully determined by its fixtures",
    ).toEqual([]);
  },

  overviewPage: async ({ page }, use) => {
    await use(new OverviewPage(page));
  },
});

export { expect };
