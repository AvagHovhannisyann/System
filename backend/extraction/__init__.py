"""LLM extraction pipeline — anonymization and its verification (Phase 7).

Phase 7 turns documents into numeric features by reading them with a model. The
danger specific to doing that on *historical* documents is that the model may
already know how the story ended. If it can name the company and date the
filing, an extracted "signal" can be recall of the outcome rather than a
reading of the text, and the feature it produces is a lookahead — I1's failure
mode arriving through the prompt rather than through a query, and just as
invisible in a backtest.

Anonymization is the control. This package currently holds P7.2:

- :mod:`backend.extraction.anonymize` — the pass itself, and the honest
  statement of what it does not catch. Read that docstring before relying on it.
- :mod:`backend.extraction.entities` — what the caller declares, and the
  reversible mapping, which is kept out of the payload by construction.
- :mod:`backend.extraction.surface` — one declared name to the many writings a
  filing actually uses.
- :mod:`backend.extraction.temporal` — every date writing, stripped.
- :mod:`backend.extraction.rules` — the deterministic arbitration between
  overlapping rules.
- :mod:`backend.extraction.leak` — the detector that checks the result, written
  independently of the masker so that it can disagree with it.

A control nobody measures is a claim. P7.9's contamination probe is what turns
this one into a measurement, and it can only do that if the leaks it finds are
the model's and not ours — which is what :mod:`backend.extraction.leak` is for.
"""

from __future__ import annotations

from backend.extraction.anonymize import AnonymizerConfig, anonymize, build_rules
from backend.extraction.entities import (
    AnonymizedDocument,
    Entity,
    EntityKind,
    EntityMapping,
    company,
    identifier,
    person,
    person_from_edgar_conformed_name,
    ticker,
)
from backend.extraction.leak import LeakFinding, LeakKind, LeakReport, Severity, detect_leaks
from backend.extraction.rules import (
    AllocatedPlaceholder,
    MaskKind,
    MaskRule,
    Replacement,
    RuleApplication,
    apply_rules,
)
from backend.extraction.surface import core_tokens, entity_rules
from backend.extraction.temporal import (
    accession_rules,
    temporal_rules,
    year_exemption_reason,
)

__all__ = [
    "AllocatedPlaceholder",
    "AnonymizedDocument",
    "AnonymizerConfig",
    "Entity",
    "EntityKind",
    "EntityMapping",
    "LeakFinding",
    "LeakKind",
    "LeakReport",
    "MaskKind",
    "MaskRule",
    "Replacement",
    "RuleApplication",
    "Severity",
    "accession_rules",
    "anonymize",
    "apply_rules",
    "build_rules",
    "company",
    "core_tokens",
    "detect_leaks",
    "entity_rules",
    "identifier",
    "person",
    "person_from_edgar_conformed_name",
    "temporal_rules",
    "ticker",
    "year_exemption_reason",
]
