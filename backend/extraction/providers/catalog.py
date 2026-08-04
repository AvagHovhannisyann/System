"""The closed vocabulary of LLM providers the platform can talk to (P7.1, §6.5).

Why an enum and not free text
-----------------------------

A provider name is not a label: it selects the code that builds an
authenticated request (:mod:`backend.extraction.providers.probe`). A provider
the platform cannot construct a request for is a provider it cannot use, so
there is no state in which storing an unrecognized name would be useful — it
would only defer the failure from configuration time to call time, which is
exactly the direction §9 tells this system not to move failures in.

The consequence is stated rather than hidden: **adding a provider is a code and
migration change.** The vocabulary appears here and in
``backend.db.models.PROVIDER_NAMES_SQL``, and drift between the two fails a unit
test rather than reaching a database. In the schema it is a CHECK constraint,
first declared by migration 0009 and *widened* by a later revision each time a
member is added — 0018 for :attr:`Provider.GROQ`. Earlier revisions keep the
vocabulary they shipped with, because a migration records what the schema was at
its point in the chain, not what it is now.

What is *not* here
------------------

No default model, no pricing, no rate limit, and no per-provider token budget.
§5-P7 requires cost-tier rather than frontier models for extraction, and B4
(provider keys and spend caps) is unresolved, so any figure written here today
would be an invented number in a module whose readers would reasonably treat it
as fact (I3). Model choice is per-task operator configuration
(:mod:`backend.extraction.providers.assignments`); the cost governor is P7.7.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Provider", "provider_from_name"]


class Provider(StrEnum):
    """An LLM provider the platform holds a credential for and can probe.

    ``StrEnum`` so a member is usable directly as the text stored in
    ``llm_provider_credential.provider`` and as a FastAPI path parameter — an
    unrecognized provider is then rejected at the HTTP boundary with a 422
    naming the accepted values, before any handler code runs.
    """

    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENAI = "openai"


def provider_from_name(name: str) -> Provider:
    """Return the :class:`Provider` spelled *name*, or refuse.

    Args:
        name: candidate provider name, matched exactly (no case folding, no
            whitespace trimming). Both are refused rather than repaired so that
            ``"OpenAI "`` cannot become a second spelling of one provider in a
            log line, a dashboard, or a SQL query a human writes.

    Returns:
        The matching enum member.

    Raises:
        ValueError: *name* is not one of the known providers. The message lists
            the accepted values, because a caller that got this wrong needs to
            know what would have been right.
    """
    try:
        return Provider(name)
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in Provider))
        msg = f"unknown provider {name!r}; known providers are: {known}"
        raise ValueError(msg) from exc
