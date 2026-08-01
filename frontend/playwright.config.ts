import { defineConfig } from "@playwright/test";

import { E2E_API_BASE_URL, E2E_BASE_URL, E2E_PORT } from "./e2e/env";

/**
 * Playwright end-to-end configuration (DIRECTIVE.md section 8: "Playwright
 * end-to-end on critical dashboard paths").
 *
 * The suite drives a real Next.js dev server in a real Chromium and mocks the
 * backend at the network layer, so it needs no database, no Redis and no
 * FastAPI process — but it exercises the genuine app bundle, the genuine
 * TanStack Query wiring and the genuine `lib/api.ts` parsing code.
 *
 * Honesty rules (DIRECTIVE.md section 2, I6): no retries, no `test.skip`, no
 * conditional execution. A machine without Chromium fails this suite; it does
 * not quietly pass it.
 *
 * `@playwright/test` is pinned to an exact version in package.json because the
 * runner and the browser build ship as a matched pair: 1.56.1 wants Chromium
 * revision 1194. Bumping the version means the CI/dev machines must fetch the
 * matching browser (`npx playwright install --with-deps chromium`) in the same
 * change, or every run fails on a missing executable.
 */
export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",

  /* Specs are independent; the mock is per-page so parallelism is safe. */
  fullyParallel: true,

  /* `test.only` silently skips every other test — never allow it in CI. */
  forbidOnly: !!process.env.CI,

  /*
   * Zero retries, everywhere. Every input to these tests is mocked, so a
   * flake is a real defect in the app or in the harness; retrying would hide
   * exactly the class of bug this suite exists to surface (I6).
   */
  retries: 0,

  workers: process.env.CI ? 1 : undefined,

  reporter: process.env.CI
    ? [["github"], ["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],

  /* Generous enough for a cold Turbopack compile of the route under test. */
  timeout: 60_000,
  expect: { timeout: 10_000 },

  use: {
    baseURL: E2E_BASE_URL,
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },

  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium", viewport: { width: 1280, height: 800 } },
    },
  ],

  /*
   * Playwright owns the dev server's lifecycle, so `npm run test:e2e` is a
   * single self-contained command in CI and locally.
   *
   * `reuseExistingServer` is false unconditionally: `NEXT_PUBLIC_API_URL` is
   * inlined into the client bundle, so reusing a server started with a
   * different value would point the app at a real backend and make the mock a
   * no-op. A slower start is worth a suite that cannot lie about what it hit.
   */
  webServer: {
    command: `npm run dev -- --port ${E2E_PORT}`,
    url: E2E_BASE_URL,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
    env: { NEXT_PUBLIC_API_URL: E2E_API_BASE_URL },
  },
});
