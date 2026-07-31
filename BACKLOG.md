# BACKLOG.md — Ideas and mid-build feature requests. Not scope.

Directive §1.2: feature requests arriving mid-build land here, not in the code.
Nothing here is worked on without the human promoting it into `PLAN.md`.

| # | date | idea | origin | notes |
|---|---|---|---|---|
| 1 | 2026-07-31 | Visual design pass on the dashboard (aesthetics, theming, layout polish) | CLAUDE.md routing rule | Explicitly Fabel's lane (see D-006). Engineering ships functional UI only; hand off to Fabel when the operator wants design work. |
| 2 | 2026-07-31 | Second DB role so append-only cannot be disabled by the app role | D-012 residual weakness | Append-only is enforced by triggers under a single owning role, which can `DROP`/`DISABLE TRIGGER` on itself. True prevention needs a separate migration-owner role with the app role holding only INSERT/SELECT — a deployment change (compose, entrypoint, Alembic config), not a code change. Not urgent for a single-operator research stack; matters if this ever runs where the app credential is reachable by anything else. |
