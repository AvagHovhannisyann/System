/**
 * Component coverage for the dashboard sidebar (`components/sidebar-nav.tsx`).
 *
 * The property under test is the inert-entry contract: a dashboard section
 * whose backend phase has not landed must be visible but *not navigable*. The
 * alternative — routing it to a placeholder page — would put invented content
 * in front of an operator, which is the failure mode DIRECTIVE.md section 2
 * I3 and section 9.2 exist to prevent. A regression here is silent by nature
 * (the sidebar still looks right), so it is asserted directly.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SidebarNav } from "@/components/sidebar-nav";

/** Every section listed in the sidebar, in the order DIRECTIVE.md section 6 lists them. */
const ALL_SECTIONS = [
  "Overview",
  "Data Health",
  "Universe",
  "Features",
  "Agents",
  "Models",
  "Backtests",
  "Portfolio",
  "Execution",
  "Monitoring",
  "Settings",
] as const;

/** The only section with a route today; everything else must be inert. */
const ROUTED_SECTIONS: ReadonlyArray<{ label: string; href: string }> = [
  { label: "Overview", href: "/" },
];

const INERT_SECTIONS = ALL_SECTIONS.filter(
  (label) => !ROUTED_SECTIONS.some((routed) => routed.label === label),
);

/**
 * The sidebar landmark.
 *
 * `getByRole` throws unless there is exactly one navigation landmark carrying
 * that accessible name, so every test below implicitly asserts the sidebar is
 * a properly labelled `<nav>` as well as whatever it checks explicitly.
 */
function nav(): HTMLElement {
  return screen.getByRole("navigation", { name: "Dashboard sections" });
}

describe("SidebarNav", () => {
  it("lists every dashboard section in order", () => {
    render(<SidebarNav />);

    const rendered = Array.from(nav().children).map(
      (child) => child.textContent,
    );
    expect(rendered).toEqual([...ALL_SECTIONS]);
  });

  it("routes only the sections that actually exist", () => {
    render(<SidebarNav />);

    const links = within(nav()).getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual(
      ROUTED_SECTIONS.map((routed) => routed.label),
    );
    for (const { label, href } of ROUTED_SECTIONS) {
      expect(within(nav()).getByRole("link", { name: label })).toHaveAttribute(
        "href",
        href,
      );
    }
  });

  it.each(INERT_SECTIONS)(
    "renders %s as an inert entry with no way to navigate to a page that does not exist",
    (label) => {
      render(<SidebarNav />);

      // Visible to the operator, so the roadmap is honest about what exists…
      const entry = within(nav()).getByText(label);
      expect(entry).toBeInTheDocument();

      // …but not a link, and not reachable by keyboard or assistive tech.
      expect(entry.tagName).toBe("SPAN");
      expect(entry).not.toHaveAttribute("href");
      expect(entry).toHaveAttribute("aria-disabled", "true");
      expect(entry).not.toHaveAttribute("tabindex");
      expect(within(nav()).queryByRole("link", { name: label })).toBeNull();

      // And it says why, rather than looking like a dead link.
      expect(entry).toHaveAttribute("title", "Not yet available");
    },
  );
});
