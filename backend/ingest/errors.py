"""Error taxonomy for the ingestion framework (P3.1, invariant I3).

The whole point of this module is that **failure has a type and failure is
loud**. Directive §2 I3 and §9.1—9.2 forbid returning plausible values when a
source cannot be reached: an unavailable source raises, and the exception says
whether retrying could possibly help.

The retry boundary is the only distinction the framework acts on
automatically:

- :class:`TransientSourceError` — the request was well-formed and the source
  simply did not answer usefully *this time* (connection reset, timeout, 429,
  5xx). Retrying the identical request may succeed, so
  :func:`backend.ingest.retry.retry_async` retries it with exponential backoff
  and jitter.
- :class:`PermanentSourceError` — the source answered, and its answer means
  the request itself is wrong (400, 401, 403, 404, 422) or the payload cannot
  be parsed. Retrying the identical request cannot succeed; retrying would
  only burn rate-limit budget and delay the operator seeing a real defect, so
  it is **never** retried.

Both derive from :class:`SourceUnavailableError`, because from the caller's
point of view the outcome is identical and is never a value: no data came
back, so no data is written and no run is recorded as successful.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ConnectorDeclarationError",
    "FutureKnowledgeTimeError",
    "IngestError",
    "PermanentSourceError",
    "RunRecordError",
    "SourceUnavailableError",
    "SupersessionError",
    "TransientSourceError",
    "source_error_for_status",
]

_MIN_ERROR_STATUS: Final = 400
"""Lowest HTTP status that denotes an error (dimensionless)."""

_MAX_ERROR_STATUS: Final = 599
"""Highest defined HTTP status (dimensionless)."""

_MIN_SERVER_ERROR_STATUS: Final = 500
"""Lowest HTTP status in the 5xx server-error range (dimensionless)."""

_TRANSIENT_CLIENT_STATUSES: Final = frozenset({408, 425, 429})
"""4xx statuses that are about *timing*, not about a malformed request.

408 Request Timeout, 425 Too Early, 429 Too Many Requests: the identical
request may well succeed later, so these are transient despite being 4xx.
Every other 4xx says the request is wrong and is permanent.
"""


class IngestError(RuntimeError):
    """Base class for every error raised by the ingestion framework."""


class ConnectorDeclarationError(IngestError):
    """A connector subclass does not declare the contract the framework requires.

    Raised at **class-definition time** (so a misdeclared connector cannot be
    instantiated, let alone scheduled) when a concrete
    :class:`~backend.ingest.base.Connector` omits or mis-types its
    ``source_name``, ``knowledge_time_policy`` or ``rate_limit``.
    """


class SourceUnavailableError(IngestError):
    """A data source did not yield usable data.

    The framework raises this (or a subclass) instead of returning a
    placeholder, a default, or an empty result that would be mistaken for
    "the source says there is nothing" — invariant I3. Catching it must never
    lead to substituting a value; the only sanctioned responses are retrying
    (transient only), failing the ingestion run, and reporting the failure.
    """


class TransientSourceError(SourceUnavailableError):
    """The source failed in a way that retrying the same request might fix.

    Connection errors, read timeouts, HTTP 408/425/429 and 5xx. This is the
    **only** error class :func:`backend.ingest.retry.retry_async` retries.
    """


class PermanentSourceError(SourceUnavailableError):
    """The source failed in a way that retrying the same request cannot fix.

    HTTP 4xx other than 408/425/429 (bad request, unauthorized, forbidden,
    not found, unprocessable), and responses whose payload cannot be parsed
    under the connector's documented contract. Never retried: a retry loop on
    a malformed request wastes the source's rate-limit budget and hides a real
    defect behind a delay.
    """


class FutureKnowledgeTimeError(IngestError):
    """A row was written whose ``knowledge_time`` lies in the future.

    Knowledge time means "when this became knowable to the market" (D-011). A
    value later than the writing host's clock asserts knowledge of something
    not yet knowable — which is precisely what invariant I1 exists to make
    impossible — so the write is refused rather than flagged. See
    :mod:`backend.ingest.write` for the enforcement point and the reasoning
    behind rejecting instead of flagging.
    """


class SupersessionError(IngestError):
    """An open-interval supersession was attempted incorrectly.

    Raised by :mod:`backend.ingest.supersession` when the row to close is not
    open-ended, when the correction's ``knowledge_time`` is not strictly later
    than the row it corrects, or when the proposed bounded ``valid_to`` does
    not lie strictly inside the open interval.
    """


class RunRecordError(IngestError):
    """The ingestion-run record could not be transitioned as requested.

    Raised when finishing a run that is not (or no longer) in the ``running``
    state — a double-finish, or a run id that does not exist. Signals a
    framework/bookkeeping defect, never a source problem.
    """


def source_error_for_status(status: int, *, source: str, detail: str) -> SourceUnavailableError:
    """Return the error to raise for an HTTP status, classified for retry.

    Args:
        status: HTTP response status code (dimensionless integer, 100—599).
        source: connector source name, used in the message.
        detail: short human-readable context (URL, response excerpt). Callers
            must not put secrets in it — messages reach logs (invariant I5).

    Returns:
        A :class:`TransientSourceError` for 408, 425, 429 and any 5xx (the
        identical request may succeed later); a :class:`PermanentSourceError`
        for every other 4xx (the request itself is wrong).

    Raises:
        ValueError: if ``status`` is not an error status (< 400 or > 599).
            Asking for the error of a successful response is a programming
            mistake, not a source failure, and must not be silently converted
            into one.
    """
    if status < _MIN_ERROR_STATUS or status > _MAX_ERROR_STATUS:
        msg = (
            f"source_error_for_status called with non-error HTTP status {status} "
            f"for source {source!r}; only 4xx/5xx statuses describe a failure"
        )
        raise ValueError(msg)
    message = f"{source}: HTTP {status} ({detail})"
    if status >= _MIN_SERVER_ERROR_STATUS or status in _TRANSIENT_CLIENT_STATUSES:
        return TransientSourceError(message)
    return PermanentSourceError(message)
