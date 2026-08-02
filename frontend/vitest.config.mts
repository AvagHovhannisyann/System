import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const rootDir = dirname(fileURLToPath(import.meta.url));

/**
 * Vitest configuration for the frontend component/unit suite
 * (DIRECTIVE.md section 8: "Test coverage ... >= 70% frontend").
 *
 * Vitest rather than Jest because this is a Vite-compatible TypeScript
 * project: it consumes the `tsconfig` path alias and TSX through the same
 * esbuild pipeline Next.js already relies on, with no Babel config, no
 * `ts-jest` transform and no second module-resolution story to keep in sync.
 *
 * This runner and the Playwright harness are strictly separate:
 *
 *   - `include` is limited to `tests/**`, and `e2e/**` is excluded outright,
 *     so `vitest run` can never pick up a Playwright spec (whose fixtures
 *     would fail under Vitest and produce a confusing red build).
 *   - Playwright's `testDir` is `./e2e`, so `playwright test` can never pick
 *     up a Vitest file.
 *   - The two have distinct npm scripts (`npm test` / `npm run test:e2e`).
 *
 * The suites are complementary, not redundant: Playwright proves the Overview
 * page behaves correctly in a real browser against a real Next.js server;
 * this suite covers the same honesty properties exhaustively and cheaply, plus
 * states the browser suite does not reach (the in-flight loading state,
 * malformed-payload rejection, the poll interval, per-field type-guard
 * behaviour).
 *
 * Honesty rules (DIRECTIVE.md section 2, I6): no retries, no conditional
 * execution, and Vitest's `allowOnly` defaults to false under CI so a stray
 * `.only` cannot silently shrink the suite.
 */
export default defineConfig({
  plugins: [react()],

  resolve: {
    // Mirrors the `@/*` path alias in tsconfig.json.
    alias: { "@": rootDir },
  },

  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],

    /* Unit/component specs live only here; e2e specs live only in e2e/. */
    include: ["tests/**/*.test.{ts,tsx}"],
    exclude: ["node_modules/**", ".next/**", "e2e/**", "coverage/**"],

    /* A suite that finds nothing must fail, not report success. */
    passWithNoTests: false,

    /* Every input is stubbed, so a flake is a real defect — never retry it. */
    retry: 0,

    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      reportsDirectory: "./coverage",

      /*
       * The whole first-party surface, with nothing carved out.
       * `app/layout.tsx` and the as-yet-unused `components/ui/button.tsx` are
       * currently uncovered and are deliberately left in the denominator:
       * excluding the awkward files is how a coverage number stops meaning
       * anything.
       */
      include: [
        "app/**/*.{ts,tsx}",
        "components/**/*.{ts,tsx}",
        "lib/**/*.{ts,tsx}",
      ],
      exclude: ["**/*.d.ts"],

      /*
       * CC.6 — the frontend half of the coverage ratchet.
       *
       * DIRECTIVE.md section 8's frontend bar, applied to all four metrics so
       * the gate cannot be cleared by covering statements while leaving
       * branches — the failure paths — untested.
       *
       * WHAT THIS GATE DOES NOT PROVE. That the covered lines ran, and no
       * more. It is satisfied by rendering a component and asserting nothing,
       * and it is satisfied faster by deleting a hard file from `include`
       * than by testing it — which is why `include` above carves out nothing
       * and why backend/tests/test_coverage_ratchet.py pins both lists.
       *
       * The denominator is also small enough that the percentage is not yet a
       * quality signal in its own right: at the time of writing the whole
       * first-party surface is 57 statements — one route (the Overview page),
       * the shell, and four UI primitives — measuring 85.96% statements /
       * 93.47% branches / 83.33% functions / 84.90% lines. The gate is still
       * worth having, because it is at the margin that it bites: adding an
       * untested page of any real size drops the ratio straight through 70%.
       * Read a pass as "no page was added without tests", not as "the
       * frontend is tested".
       *
       * HOW THE FLOOR MOVES. Up, by hand. It stays at the directive's 70
       * rather than at today's 85.96 on purpose — a floor pinned to the
       * current number ratifies whatever the last run happened to reach and
       * then punishes every commit after it. `autoUpdate` is the Vitest
       * setting that would do exactly that, so it is set to false explicitly
       * rather than left to its default. The same 70 is pinned in
       * backend/tests/test_coverage_ratchet.py, so a change here is a
       * deliberate two-file edit.
       */
      thresholds: {
        autoUpdate: false,
        statements: 70,
        branches: 70,
        functions: 70,
        lines: 70,
      },
    },
  },
});
