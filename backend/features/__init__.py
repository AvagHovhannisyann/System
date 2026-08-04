"""Feature library: declared, capped, and lag-enforced factors (Phase 5).

A feature here is a **declaration** with a computation attached, not a function
that happens to produce a column. The declaration
(:class:`~backend.features.spec.FeatureSpec`) states the feature's name, what it
is in prose, its **units**, the wall-clock delay before its value is knowable,
and the fact tables it reads. Two of those exist because of invariants the rest
of the platform cannot re-derive on its own:

**Units** (directive §8). A float64 column carries no unit. Whether a number is
a fraction, a percent, basis points or dollars is knowable only from the
declaration, and unit confusion in this domain is silent — a cost model fed
percent where it expected basis points produces a backtest that is wrong by two
orders of magnitude and looks fine.

**Availability lag** (invariant I1). A quarterly fundamental is not knowable on
the quarter end; it is knowable when the filing is accepted. A feature that uses
the quarter-end number on the quarter-end date is prescient, and no distribution
check, type check or unit test reveals it — only a backtest that is worthless.
:mod:`backend.features.compute` turns the declared lag into the only as-of
instant a computation is ever handed.

**The cap is thirty** (directive §5 Phase 5), including the Phase 7 LLM
features. :class:`~backend.features.registry.FeatureRegistry` refuses the 31st.
Adding one therefore means removing one and logging the swap in
``DECISIONS.md``. Feature proliferation is the main route to overfitting a
cross-sectional ranker, and the count is the one complexity measure a reviewer
can verify at a glance.

Modules:

- :mod:`backend.features.spec` — the declaration schema, the cap constant, and
  the ``compute_date → instant`` convention;
- :mod:`backend.features.registry` — the catalog, the cap, and the ``@feature``
  decorator that P5.3's baseline factors and P7's LLM features register with;
- :mod:`backend.features.compute` — the single entry point that runs a feature,
  pinned to the as-of its declaration permits;
- :mod:`backend.features.errors` — the failure taxonomy, which is where the cap
  and the lag stop being documentation;
- :mod:`backend.features.transforms` (P5.2) — the winsorize → z-score →
  neutralize pipeline. Imported from its own module rather than re-exported
  here: per the frozen wave-1 contract those functions operate on plain
  ``numpy`` arrays and know nothing about the registry, the database or an
  as-of instant, and that separation is easier to keep honest when the import
  path says so.

Nothing in this package computes a factor. The registry's extension point is an
async callable and it is P5.3 that supplies the arithmetic; a plausible-looking
stub would be exactly the fabrication invariant I3 forbids.
"""

from __future__ import annotations

from backend.features.compute import (
    FeatureComputation,
    FeatureComputeRequest,
    FeatureVector,
    compute_feature,
    resolve_as_of,
)
from backend.features.errors import (
    AvailabilityLagViolationError,
    DuplicateFeatureError,
    FeatureCapExceededError,
    FeatureComputeError,
    FeatureError,
    FeatureSpecError,
    MalformedFeatureVectorError,
    UnknownFeatureError,
)
from backend.features.registry import FeatureRegistry, default_registry, feature
from backend.features.spec import (
    MAX_AVAILABILITY_LAG,
    MAX_FEATURES,
    FeatureSpec,
    compute_instant,
)

__all__ = [
    "MAX_AVAILABILITY_LAG",
    "MAX_FEATURES",
    "AvailabilityLagViolationError",
    "DuplicateFeatureError",
    "FeatureCapExceededError",
    "FeatureComputation",
    "FeatureComputeError",
    "FeatureComputeRequest",
    "FeatureError",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureSpecError",
    "FeatureVector",
    "MalformedFeatureVectorError",
    "UnknownFeatureError",
    "compute_feature",
    "compute_instant",
    "default_registry",
    "feature",
    "resolve_as_of",
]
