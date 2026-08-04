"""The fundamentals source gate: six factors that must refuse to compute (P5.3).

Book-to-price, earnings yield, gross profitability, ROIC, accruals and asset
growth are all ratios of accounting quantities. Their input is the
point-in-time fundamentals feed — Sharadar SF1 on the as-reported ARQ/ARY
dimensions, per the operator's B1 decision — and **that connector does not
exist**. P3.5 is blocked on B1; there is no table, no column names, no
knowledge-time policy.

So these six factors declare themselves fully and then raise. That is the whole
design, and it is deliberate:

**A plausible number here would be the worst possible outcome.** Directive §2
I3 and §9.1-9.2: never fabricate data, never stub for appearance, *"a function
that returns a plausible value without doing the work is worse than one that
raises"*. A book-to-price computed from invented equity would flow into the
transform pipeline, the design matrix, the model, the optimizer and the
backtest, and every one of those would report healthy numbers. The result would
be a backtest that looks real. Nothing downstream can tell a fabricated ratio
from a measured one — that is precisely why it must not exist.

**Declaring the factor is not the same as faking it.** The declaration is real
work and it is the part that has to be right: the units, the availability lag,
and the source-table contract are what P3.5 will be built against, and they are
reviewable now (directive §5's feature catalog, §6.4's Features page). The
computation is the part that cannot be honest yet, so it refuses.

**Two distinct refusals, because they mean different things.** Today the table
is absent and the answer is "blocked on B1"
(:class:`FundamentalsSourceUnavailableError`). Once P3.5 lands, the table will
exist while the arithmetic below still does not, and a factor that kept
reporting "blocked on B1" would be lying about which work remains
(:class:`FundamentalsComputationNotWrittenError`). Which of the two fires is
decided by looking at the mapped schema rather than by a flag someone has to
remember to flip.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**The table name is provisional.** :data:`FUNDAMENTALS_TABLE` is this package's
single statement of what P3.5 will be called, chosen to match the store's
existing convention (singular snake_case: ``price_bar``, ``security_master``,
``edgar_filing``). It appears in six ``FeatureSpec.source_tables`` declarations
and in the schema probe, so if P3.5 chooses another name this constant is the
only edit — and the probe below will not silently start believing the feed
arrived.

**The knowledge-time policy is the connector's, and it is undecided.** Under
D-011 each connector declares when its facts became knowable; for fundamentals
that is the original filing's public instant, never the fiscal period end. The
availability lags declared by the six factors are a *margin on top of* that
policy, sized in :mod:`backend.features.factors.value` where the reasoning
belongs to a declaration a reader can check.

**No fallback exists and none should be added.** There is no "estimate from the
last 10-K", no "carry forward the previous quarter", no default. A feature with
no input has no value, and ``NaN`` is not available here either: ``NaN`` claims
the pipeline asked the store and the store had nothing, which would be a claim
about a table that does not exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NoReturn

from backend.db.base import Base
from backend.features.errors import FeatureComputeError

if TYPE_CHECKING:
    from collections.abc import Container

__all__ = [
    "FUNDAMENTALS_BLOCKER",
    "FUNDAMENTALS_CONNECTOR_TASK",
    "FUNDAMENTALS_TABLE",
    "FundamentalsComputationNotWrittenError",
    "FundamentalsSourceUnavailableError",
    "fundamentals_source_present",
    "require_fundamentals_source",
]

FUNDAMENTALS_TABLE: Final = "fundamental_report"
"""Physical table the point-in-time fundamentals feed will land in (P3.5).

**Provisional**: the connector is blocked on B1 and the table does not exist.
The name follows the store's convention (singular snake_case, one row per
as-reported fiscal period per security) and is declared here once so that the
six fundamental factors, their ``source_tables`` sets and the schema probe in
:func:`fundamentals_source_present` cannot drift apart. If P3.5 lands under a
different name, this constant is the only place that changes.
"""

FUNDAMENTALS_BLOCKER: Final = "B1"
"""Blocker ID in ``BLOCKERS.md``: data-vendor selection and API keys."""

FUNDAMENTALS_CONNECTOR_TASK: Final = "P3.5"
"""Task ID in ``PLAN.md``: the point-in-time fundamentals connector."""


class FundamentalsSourceUnavailableError(FeatureComputeError):
    """Raised when a fundamental factor is asked to compute with no feed to read.

    The honest state of the platform today: the fundamentals connector (P3.5) is
    blocked on B1, so :data:`FUNDAMENTALS_TABLE` is not in the schema. The
    factor's declaration is real — its units, availability lag and source tables
    are the contract P3.5 will be built to — but its value is not computable and
    is not approximated.

    Attributes:
        feature: name of the feature that could not be computed.
        table: the missing source table.
        blocker: the blocker ID that must clear first.
        task: the task ID that will supply the feed.
    """

    def __init__(self, feature: str) -> None:
        """Build the error from the feature that was asked for.

        Args:
            feature: name of the feature that could not be computed.
        """
        self.feature = feature
        self.table = FUNDAMENTALS_TABLE
        self.blocker = FUNDAMENTALS_BLOCKER
        self.task = FUNDAMENTALS_CONNECTOR_TASK
        super().__init__(
            f"feature {feature!r} reads the point-in-time fundamentals feed, which "
            f"does not exist: table {FUNDAMENTALS_TABLE!r} is not in the schema "
            f"because connector {FUNDAMENTALS_CONNECTOR_TASK} is blocked on "
            f"{FUNDAMENTALS_BLOCKER} (data-vendor selection and credentials — see "
            f"BLOCKERS.md). The declaration is complete and reviewable; the value "
            f"is not computable. Refusing rather than returning a plausible number: "
            f"a fabricated fundamental ratio is indistinguishable from a measured "
            f"one everywhere downstream, and it would produce a backtest that looks "
            f"real (I3, directive §9.1-9.2)."
        )


class FundamentalsComputationNotWrittenError(FeatureComputeError):
    """Raised when the fundamentals feed exists but a factor's arithmetic does not.

    The state after P3.5 lands and before this package is finished. It is a
    separate error from :class:`FundamentalsSourceUnavailableError` because it
    names different outstanding work: there, the operator must clear a blocker;
    here, an engineer must write a query against a schema that now exists. A
    single error covering both would keep reporting "blocked on B1" long after
    B1 cleared, which is how a task goes quietly missing.

    Attributes:
        feature: name of the feature whose computation is unwritten.
        table: the source table that is now present.
    """

    def __init__(self, feature: str) -> None:
        """Build the error from the feature that was asked for.

        Args:
            feature: name of the feature whose computation is unwritten.
        """
        self.feature = feature
        self.table = FUNDAMENTALS_TABLE
        super().__init__(
            f"feature {feature!r} has a complete declaration and no computation: "
            f"table {FUNDAMENTALS_TABLE!r} is now mapped, so blocker "
            f"{FUNDAMENTALS_BLOCKER} no longer explains the gap and the query "
            f"against it must be written (P5.3 completion). Refusing rather than "
            f"returning a plausible number (I3)."
        )


def fundamentals_source_present(tables: Container[str] | None = None) -> bool:
    """Return whether the point-in-time fundamentals table is mapped.

    Reads the live SQLAlchemy metadata rather than a hand-maintained flag, so
    the answer changes when P3.5 actually lands and cannot be left stale by
    someone forgetting to update it.

    Args:
        tables: table names to probe. Defaults to the mapped schema
            (``Base.metadata.tables``). The parameter exists so the
            "feed has arrived" branch of :func:`require_fundamentals_source` is
            testable without mutating process-wide metadata — a test that
            registered a fake table would corrupt every other test in the run.

    Returns:
        ``True`` when :data:`FUNDAMENTALS_TABLE` is among the names.
    """
    known: Container[str] = Base.metadata.tables if tables is None else tables
    return FUNDAMENTALS_TABLE in known


def require_fundamentals_source(feature: str, *, tables: Container[str] | None = None) -> NoReturn:
    """Refuse to compute a fundamental factor, naming what is actually missing.

    Always raises. There is no success path, because there is no fundamentals
    feed and no substitute for one; the return type says so, so a caller that
    tried to continue afterwards would not type-check.

    Args:
        feature: name of the feature being computed, for the message.
        tables: table names to probe; defaults to the mapped schema. See
            :func:`fundamentals_source_present`.

    Raises:
        FundamentalsComputationNotWrittenError: if the fundamentals table is
            mapped — the feed arrived and the arithmetic is the missing piece.
        FundamentalsSourceUnavailableError: otherwise — the feed itself is
            missing, blocked on B1.
    """
    if fundamentals_source_present(tables):
        raise FundamentalsComputationNotWrittenError(feature)
    raise FundamentalsSourceUnavailableError(feature)
