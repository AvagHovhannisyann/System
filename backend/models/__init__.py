"""Predictor package (directive §5 Phase 8): training and purged cross-validation.

Currently holds P8.1 only — :mod:`backend.models.purged_cv`, the time-aware
splitter every later task in this phase depends on. The LightGBM pipeline
(P8.2), time-window ensembling (P8.3) and IC reporting (P8.4) land beside it and
all consume :class:`~backend.models.purged_cv.PurgedKFold`; Phase 10's
combinatorial purged CV builds on the same purge/embargo rules.

Nothing here touches the database. The splitter operates on label spans handed
to it, which keeps the correctness of fold boundaries testable in isolation —
the directive's Phase 8 gate is exactly that: fold boundaries unit-tested for
correctness rather than assumed.
"""

from __future__ import annotations

from backend.models.purged_cv import (
    EmptyTrainingSetError,
    Fold,
    PurgedCVError,
    PurgedKFold,
)

__all__ = [
    "EmptyTrainingSetError",
    "Fold",
    "PurgedCVError",
    "PurgedKFold",
]
