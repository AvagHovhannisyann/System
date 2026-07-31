import Link from "next/link";

/**
 * Dashboard sections per DIRECTIVE.md section 6. Only Overview is routed
 * today; sections without an href render as inert entries until their
 * backend phase lands (no stub pages with fake content).
 */
const NAV_ITEMS: ReadonlyArray<{ label: string; href?: string }> = [
  { label: "Overview", href: "/" },
  { label: "Data Health" },
  { label: "Universe" },
  { label: "Features" },
  { label: "Agents" },
  { label: "Models" },
  { label: "Backtests" },
  { label: "Portfolio" },
  { label: "Execution" },
  { label: "Monitoring" },
  { label: "Settings" },
];

/** Minimal sidebar navigation for the operator dashboard shell. */
export function SidebarNav() {
  return (
    <nav aria-label="Dashboard sections" className="flex flex-col gap-1 p-2">
      {NAV_ITEMS.map((item) =>
        item.href !== undefined ? (
          <Link
            key={item.label}
            href={item.href}
            className="rounded-md px-3 py-2 text-sm font-medium text-foreground hover:bg-accent hover:text-accent-foreground"
          >
            {item.label}
          </Link>
        ) : (
          <span
            key={item.label}
            aria-disabled="true"
            title="Not yet available"
            className="cursor-not-allowed rounded-md px-3 py-2 text-sm font-medium text-muted-foreground/60"
          >
            {item.label}
          </span>
        ),
      )}
    </nav>
  );
}
