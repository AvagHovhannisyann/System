"use client";

import { useQuery } from "@tanstack/react-query";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { API_BASE_URL, fetchHealth, type ComponentHealth } from "@/lib/api";

const HEALTH_POLL_INTERVAL_MS = 10_000;

function ComponentRow({
  name,
  health,
}: {
  name: string;
  health: ComponentHealth;
}) {
  return (
    <div className="flex items-center justify-between border-b py-2 last:border-b-0">
      <span className="text-sm font-medium">{name}</span>
      <span className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">
          {health.latency_ms.toFixed(1)} ms
        </span>
        <Badge variant={health.up ? "secondary" : "destructive"}>
          {health.up ? "up" : "down"}
        </Badge>
      </span>
    </div>
  );
}

function HealthSkeleton() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-5 w-24" />
      <Skeleton className="h-8 w-full" />
      <Skeleton className="h-8 w-full" />
      <Skeleton className="h-4 w-40" />
    </div>
  );
}

export default function OverviewPage() {
  const { data, error, isPending, isError } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: HEALTH_POLL_INTERVAL_MS,
  });

  return (
    <div className="max-w-xl space-y-6">
      <h1 className="text-2xl font-semibold">Overview</h1>
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>
            Backend component status from /api/health, polled every{" "}
            {HEALTH_POLL_INTERVAL_MS / 1000}s
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isPending ? (
            <HealthSkeleton />
          ) : isError ? (
            <div className="space-y-2">
              <Badge variant="destructive">unreachable</Badge>
              <p className="text-sm text-muted-foreground">
                Could not reach the backend at {API_BASE_URL}.
              </p>
              <p className="text-sm text-destructive">
                {error instanceof Error ? error.message : "Unknown error"}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Overall</span>
                <Badge
                  variant={data.status === "ok" ? "secondary" : "destructive"}
                >
                  {data.status}
                </Badge>
              </div>
              <div>
                <ComponentRow name="Database" health={data.components.db} />
                <ComponentRow name="Redis" health={data.components.redis} />
              </div>
              <p className="text-sm text-muted-foreground">
                Backend version {data.version}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
