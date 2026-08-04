/**
 * Coverage for the client-side provider shell (`components/providers.tsx`).
 *
 * `useQuery` throws when no `QueryClientProvider` is above it, so a consumer
 * that resolves is proof the provider really supplies a working client rather
 * than merely rendering its children. The retry setting is asserted because
 * the dashboard's failure behaviour depends on it: with retries left at the
 * TanStack default of 3, a dead backend would sit on a loading skeleton for
 * several seconds before admitting it is unreachable.
 */

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Providers } from "@/components/providers";

function QueryConsumer() {
  const client = useQueryClient();
  const { data } = useQuery({
    queryKey: ["providers-probe"],
    queryFn: () => Promise.resolve("client is live"),
  });
  const defaults = client.getDefaultOptions().queries;

  return (
    <div>
      <span data-testid="query-result">{data ?? "pending"}</span>
      <span data-testid="retry">{String(defaults?.retry)}</span>
      <span data-testid="refetch-on-focus">
        {String(defaults?.refetchOnWindowFocus)}
      </span>
    </div>
  );
}

describe("Providers", () => {
  it("supplies a live query client to its children", async () => {
    render(
      <Providers>
        <QueryConsumer />
      </Providers>,
    );

    await waitFor(() =>
      expect(screen.getByTestId("query-result")).toHaveTextContent(
        "client is live",
      ),
    );
  });

  it("bounds retries so a dead backend is reported instead of hidden behind a spinner", () => {
    render(
      <Providers>
        <QueryConsumer />
      </Providers>,
    );

    expect(screen.getByTestId("retry")).toHaveTextContent("1");
    expect(screen.getByTestId("refetch-on-focus")).toHaveTextContent("false");
  });
});
