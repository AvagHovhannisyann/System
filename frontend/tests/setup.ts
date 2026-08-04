/**
 * Global setup for the Vitest component/unit suite.
 *
 * `globals` is left off in `vitest.config.mts`, so React Testing Library
 * cannot auto-register its own teardown — the unmount hook is wired here
 * explicitly. Without it, components from one test would leak into the next
 * and a DOM assertion could pass against a stale render.
 */

import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
