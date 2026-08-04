/**
 * Shared constants for the end-to-end harness.
 *
 * Imported by both `playwright.config.ts` (which starts the dev server and
 * sets `baseURL`) and the test fixtures (which build request-interception
 * patterns), so the app under test and the mock can never disagree about
 * which origins are in play. Deliberately free of any Playwright import so
 * the config file can load it without pulling in the test runtime.
 */

/**
 * TCP port the e2e dev server binds to.
 *
 * Not 3000: a developer's own `npm run dev` must never be mistaken for the
 * server under test, because that server would be built against a different
 * `NEXT_PUBLIC_API_URL` and would silently defeat the network mock.
 */
export const E2E_PORT = 3210;

/**
 * Origin the browser loads the dashboard from.
 *
 * `localhost` rather than `127.0.0.1` so `next dev` treats the requests as
 * same-origin and does not emit its cross-origin dev-resource warning.
 */
export const E2E_BASE_URL = `http://localhost:${E2E_PORT}`;

/**
 * Backend origin the app under test is pointed at via `NEXT_PUBLIC_API_URL`.
 *
 * `.invalid` is reserved by RFC 2606 and is guaranteed never to resolve, so a
 * request that escapes interception fails loudly instead of quietly reaching a
 * real backend that happens to be running on the machine. Determinism here
 * comes from the mock, never from a live service being up or down.
 */
export const E2E_API_BASE_URL = "http://backend.e2e.invalid";

/** Glob matching every request the app can make to the backend origin. */
export const E2E_API_GLOB = `${E2E_API_BASE_URL}/**`;

/** Absolute URL of the health endpoint the Overview page polls. */
export const E2E_HEALTH_URL = `${E2E_API_BASE_URL}/api/health`;
