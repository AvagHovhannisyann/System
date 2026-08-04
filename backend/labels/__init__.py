"""Label construction — turning price paths into supervised targets (Phase 6).

Directive §0.3 names label construction as one of the components where "a silent
error invalidates the entire system and will not surface as a test failure".
That is not rhetoric about this package specifically; it is a description of how
labelling fails. A label that peeks at the future by one bar produces a model
with a beautiful information coefficient, a beautiful backtest, and no edge. No
distribution check catches it, because the distribution is fine. Every design
choice here is made against that failure mode, and the tests are written to
catch it by *recomputation on truncated data* rather than by inspection.

Three pieces, in the order they are applied:

- :mod:`backend.labels.residualize` (P6.2) — strip market and sector from the
  return path first, using betas fitted on a **trailing** window and frozen
  before the labelling window opens, so the label describes the security's own
  movement rather than its beta.
- :mod:`backend.labels.barriers` (P6.1) — the triple barrier itself: upper and
  lower barriers sized by **trailing** volatility, plus a vertical (time)
  barrier at 5, 21 or 63 bars. The label is whichever barrier the path touches
  first.
- :mod:`backend.labels.uniqueness` (P6.3) — overlapping labels share
  information, so they are down-weighted by average uniqueness, and the
  resulting **effective sample size** is reported. Ten thousand overlapping
  21-day labels can carry the information of a few hundred independent ones.

:mod:`backend.labels.volatility` holds the trailing estimator that sizes the
barriers, and :mod:`backend.labels.errors` the failure taxonomy — which is
substantial on purpose, because every situation this package cannot handle
honestly raises rather than defaults.

--------------------------------------------------------------------------
Standing conventions for the whole package
--------------------------------------------------------------------------

**Log returns, as fractions, per bar.** ``0.02`` is a 2% move. No annualization
anywhere; nothing multiplies by ``sqrt(252)``.

**Bars, not timestamps.** Every index is a dimensionless position into the
caller's series, so all interval arithmetic is exact integer arithmetic. Mapping
bars to timestamps (for Phase 8's purged cross-validation, which does work in
wall-clock time) is the caller's step: ``timestamps[labels.event_index]`` and
``timestamps[labels.resolution_index]``.

**One security at a time.** These functions take a single price path. A
cross-sectional panel is labelled security by security and stacked; the array
helpers refuse two-dimensional input rather than flattening it, because a
flattened panel labels beautifully and means nothing.

**No data source.** Nothing here reads the database or the network. Labels are
computed from arrays the caller supplies, which is what makes the mathematics
testable against constructed paths whose correct answer is derivable by hand.
"""

from __future__ import annotations

from backend.labels.barriers import (
    BarrierSpec,
    IntrabarPolicy,
    LabelBasis,
    LabelSet,
    TripleBarrierOutcome,
    barrier_log_return_width,
    resolve_barrier_touch,
    triple_barrier_labels,
    usable_event_indices,
)
from backend.labels.errors import (
    AmbiguousLabelError,
    DegenerateVolatilityError,
    InsufficientHistoryError,
    LabelConfigurationError,
    LabelError,
    LabelInputError,
    RankDeficientFactorError,
)
from backend.labels.residualize import (
    ResidualFit,
    ResidualizedLabels,
    ResidualSpec,
    fit_residual_model,
    residualized_triple_barrier_labels,
    usable_residual_event_indices,
)
from backend.labels.uniqueness import (
    UniquenessResult,
    effective_sample_size,
    sample_uniqueness,
    sample_uniqueness_from_labels,
)
from backend.labels.volatility import daily_log_returns, trailing_volatility

__all__ = [
    "AmbiguousLabelError",
    "BarrierSpec",
    "DegenerateVolatilityError",
    "InsufficientHistoryError",
    "IntrabarPolicy",
    "LabelBasis",
    "LabelConfigurationError",
    "LabelError",
    "LabelInputError",
    "LabelSet",
    "RankDeficientFactorError",
    "ResidualFit",
    "ResidualSpec",
    "ResidualizedLabels",
    "TripleBarrierOutcome",
    "UniquenessResult",
    "barrier_log_return_width",
    "daily_log_returns",
    "effective_sample_size",
    "fit_residual_model",
    "residualized_triple_barrier_labels",
    "resolve_barrier_touch",
    "sample_uniqueness",
    "sample_uniqueness_from_labels",
    "trailing_volatility",
    "triple_barrier_labels",
    "usable_event_indices",
    "usable_residual_event_indices",
]
