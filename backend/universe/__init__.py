"""Point-in-time universe construction (directive §5 Phase 4).

A *universe* is the set of securities a strategy is allowed to hold on a given
rebalance date, screened on exchange, price, market capitalisation, average
dollar volume, and borrow availability. Reconstructing it correctly for a **past**
date is the whole difficulty: the answer must be built out of what was knowable
then, and it must contain the names that later stopped existing. A universe that
quietly restricts itself to today's survivors turns every downstream backtest
into fiction, and it does so without producing a single failing test — the Sharpe
ratio simply goes up.

Curated surface, by module:

- :mod:`~backend.universe.criteria` — the screens' parameters
  (:class:`UniverseCriteria`), the per-name screening inputs
  (:class:`UniverseCandidate`), the per-name decision (:class:`FilterOutcome`),
  the frozen :data:`FILTER_ORDER`, and the pure screen
  (:func:`evaluate_candidate`).
- :mod:`~backend.universe.builder` — :func:`build_universe`, the P4.1 entry
  point, plus the read models and pure steps it is assembled from.
- :mod:`~backend.universe.snapshot` — the result value object
  (:class:`UniverseSnapshot`) and its append-only persistence.
- :mod:`~backend.universe.history` — P4.2 historical reconstruction across a
  schedule of rebalance dates, with the size and turnover series gate G4 asks
  for.
- :mod:`~backend.universe.waterfall` — P4.2 filter-impact waterfall (§6.3):
  how many names each screen removed, reconstructed from the stored outcomes.
- :mod:`~backend.universe.errors` — the failure taxonomy. Read that module's
  docstring first: the errors are the design.

Two things a caller must know before using any of this.

**Every read goes through a session the caller has already scoped with**
``as_of()`` (invariant I1). Nothing here opens a session, and a session with no
as-of bound is refused rather than used, because an unscoped read would
reconstruct a past universe out of facts that did not exist at the time.

**Two of the five screens cannot run today and the build refuses.**
``market_cap`` needs a shares-outstanding history and ``borrow`` needs a locate
feed; neither source exists in this repository (``BLOCKERS.md`` B1 and B2). Since
a strictly positive market-cap floor is mandatory, *every* build currently raises
:class:`UniverseInputUnavailableError`. That is the honest state of the system:
skipping an unevaluable screen would produce a plausible universe that is not the
one the criteria describe, which is the exact failure invariant I3 exists to
prevent. Everything either side of the refusal — the reads, the listing
predicate, the ADV arithmetic, the screen, the snapshot, the history, the
waterfall — is complete and tested.
"""

from backend.universe.builder import (
    DailyBar,
    ListedSecurity,
    adv_window_calendar_days,
    adv_window_start,
    assemble_candidates,
    build_universe,
    median_dollar_volume_usd,
    read_listings,
    read_price_window,
    require_available_inputs,
    screen_candidates,
)
from backend.universe.criteria import (
    FILTER_ORDER,
    FilterOutcome,
    UniverseCandidate,
    UniverseCriteria,
    evaluate_candidate,
)
from backend.universe.errors import (
    UniverseConsistencyError,
    UniverseCriteriaError,
    UniverseError,
    UniverseInputUnavailableError,
    UniverseSessionError,
)
from backend.universe.history import (
    MembershipSpan,
    UniverseHistory,
    UniverseSizePoint,
    UniverseTurnoverPoint,
    build_history,
    load_history,
    persist_history,
)
from backend.universe.snapshot import (
    UniverseSnapshot,
    load_snapshots,
    persist_snapshot,
    snapshot_from_rows,
    snapshots_by_date,
)
from backend.universe.waterfall import (
    FilterWaterfall,
    WaterfallStep,
    filter_waterfall,
    waterfall_series,
)

__all__ = [
    "FILTER_ORDER",
    "DailyBar",
    "FilterOutcome",
    "FilterWaterfall",
    "ListedSecurity",
    "MembershipSpan",
    "UniverseCandidate",
    "UniverseConsistencyError",
    "UniverseCriteria",
    "UniverseCriteriaError",
    "UniverseError",
    "UniverseHistory",
    "UniverseInputUnavailableError",
    "UniverseSessionError",
    "UniverseSizePoint",
    "UniverseSnapshot",
    "UniverseTurnoverPoint",
    "WaterfallStep",
    "adv_window_calendar_days",
    "adv_window_start",
    "assemble_candidates",
    "build_history",
    "build_universe",
    "evaluate_candidate",
    "filter_waterfall",
    "load_history",
    "load_snapshots",
    "median_dollar_volume_usd",
    "persist_history",
    "persist_snapshot",
    "read_listings",
    "read_price_window",
    "require_available_inputs",
    "screen_candidates",
    "snapshot_from_rows",
    "snapshots_by_date",
    "waterfall_series",
]
