"""Connector registry: source name -> connector class (P3.1).

The scheduler needs to turn a string (a beat-schedule entry, an operator's
manual re-sync request from the Data Health page) into a connector class. This
is that lookup, and nothing more.

Registration is **explicit**, via the :func:`register_connector` decorator,
rather than automatic on subclassing. Automatic registration would enrol every
subclass ever defined — including the test doubles that exist precisely to
exercise failure paths — and a test double sitting in the production registry
under a plausible source name is exactly the kind of thing invariant I3 exists
to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.ingest.base import Connector

__all__ = ["connector_class", "register_connector", "registered_sources"]

_REGISTRY: dict[str, type[Connector]] = {}


def register_connector[ConnectorT: type[Connector]](cls: ConnectorT) -> ConnectorT:
    """Register a connector class under its ``source_name`` and return it.

    Args:
        cls: a concrete :class:`~backend.ingest.base.Connector` subclass.

    Returns:
        ``cls`` unchanged, so the function works as a class decorator.

    Raises:
        ValueError: if another class is already registered under the same
            source name. Silently overwriting would let a later import decide
            which code actually runs for a source.
    """
    existing = _REGISTRY.get(cls.source_name)
    if existing is not None and existing is not cls:
        msg = (
            f"source {cls.source_name!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}; refusing to replace it "
            f"with {cls.__module__}.{cls.__qualname__}"
        )
        raise ValueError(msg)
    _REGISTRY[cls.source_name] = cls
    return cls


def connector_class(source: str) -> type[Connector]:
    """Return the connector class registered for ``source``.

    Args:
        source: source name as recorded on ingestion runs.

    Returns:
        The registered connector class.

    Raises:
        KeyError: if no connector is registered for that source. The message
            lists what *is* registered, so a typo in a schedule entry is
            obvious rather than mysterious.
    """
    try:
        return _REGISTRY[source]
    except KeyError:
        msg = (
            f"no connector registered for source {source!r}; "
            f"registered sources: {sorted(_REGISTRY)}"
        )
        raise KeyError(msg) from None


def registered_sources() -> tuple[str, ...]:
    """Return every registered source name, sorted."""
    return tuple(sorted(_REGISTRY))
