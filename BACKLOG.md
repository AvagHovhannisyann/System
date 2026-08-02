# BACKLOG.md — Ideas and mid-build feature requests. Not scope.

Directive §1.2: feature requests arriving mid-build land here, not in the code.
Nothing here is worked on without the human promoting it into `PLAN.md`.

| # | date | idea | origin | notes |
|---|---|---|---|---|
| 1 | 2026-07-31 | Visual design pass on the dashboard (aesthetics, theming, layout polish) | CLAUDE.md routing rule | Explicitly Fabel's lane (see D-006). Engineering ships functional UI only; hand off to Fabel when the operator wants design work. |
| 2 | 2026-07-31 | Second DB role so append-only cannot be disabled by the app role | D-012 residual weakness | Append-only is enforced by triggers under a single owning role, which can `DROP`/`DISABLE TRIGGER` on itself. True prevention needs a separate migration-owner role with the app role holding only INSERT/SELECT — a deployment change (compose, entrypoint, Alembic config), not a code change. Not urgent for a single-operator research stack; matters if this ever runs where the app credential is reachable by anything else. |
| 3 | 2026-08-02 | `FeatureVector` is `frozen=True` with an `ndarray` field, so `hash()` raises `TypeError` and `==` between two distinct-but-equal vectors raises numpy's ambiguous-truth-value error | P5.1 review | Outside P5.1's claim set, so tested-around rather than changed. Bites whoever puts vectors in a set/dict key or asserts equality — likely P5.3 factors or P8 feature-matrix assembly. Fix is either `eq=False` with an explicit `equals()` using `np.array_equal`, or storing an immutable read-only view plus a cached content hash. Choose when there is a real consumer, not now. |
