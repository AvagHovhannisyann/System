"""Extraction cache, keyed on ``hash(document + prompt_version + model)`` (P7.6).

The directive gives the key and Gate G7 gives the bar: above 90% hit rate on
repeat runs. The reason the key is spelled out rather than left to
implementation is that it is also the *invalidation* mechanism, and this module
is written so that nobody ever has to remember that.

Invalidation is structural, not a procedure
-------------------------------------------

§6.5 requires that "a prompt change invalidates the cache for affected documents
automatically". There is no invalidation code here, and no scheduled sweep,
because there is nothing to invalidate: a prompt version is content-addressed
(:mod:`backend.extraction.prompts.versioning`), so a changed prompt is a
different ``prompt_version`` component, which is a different key, which
addresses an entry that does not exist. The old entries remain in the store,
untouched and unreachable by the new prompt — which is also what makes a
rollback instant: pointing the task back at the previous hash makes yesterday's
entries addressable again, at full hit rate, with no re-spend.

Two properties make that guarantee hold rather than merely sound true, and both
are pinned by tests:

* :meth:`CacheKey.build` takes a :class:`~backend.extraction.prompts.versioning.PromptVersion`
  **object**, not a hash string. The hash is a property derived from the text on
  every read, so no caller can supply an address that no longer matches the
  prompt beside it — the failure mode this design exists to prevent is precisely
  "the prompt changed and the recorded version string didn't";
* the ``document`` component is the **payload actually sent** — the anonymized
  chunk, after masking. Anything that changes what the model reads changes the
  key: a different document, a different chunking, a different anonymizer
  configuration. A key built from the pre-masking text would let a
  masking change silently reuse responses to text that no longer exists.

What is *not* in the key: the task name. Two tasks with byte-identical prompts,
schemas and payloads would legitimately share an entry, because the model call
would be identical in every respect; and since the output schema is part of the
prompt's address, they cannot differ in what the response has to satisfy. The
task name is stored on the entry for provenance, never keyed on.

Also not in the key: temperature. It is 0 everywhere (§5-P7,
:mod:`backend.extraction.tasks.client`), and the pipeline rejects anything else,
so including it would be encoding a constant. If that ever changes it must go
into the key in the same commit — a comment nobody reads is not that guarantee,
so :func:`assert_temperature_is_cacheable` is called on the pipeline's path and
fails loudly instead.

Storage
-------

:class:`CacheStore` is the interface; two implementations ship:

* :class:`InMemoryCacheStore` — process-local, unbounded, fully exercised by the
  tests. This is what a single-process backfill run uses;
* :class:`RedisCacheStore` — durable across processes and runs, on the cache
  the stack already specifies (§3). **Its client interaction is unexercised.**
  Its serialization is a pure function and is tested as one; ``get``, ``put``,
  ``discard`` and ``clear`` have never run against a server, because there is
  no Redis in this test environment and standing up a fake would mean editing
  the dependency set, which this change does not do. Treat the store itself as
  unverified until an integration test covers it — a tested codec is not a
  tested store.

Entries never expire by default. A TTL would quietly convert a hit into a spend
and make Gate G7's hit-rate figure a function of how long the operator waited,
which is not a property of the cache.

Units: hit rate is a **fraction in [0, 1]**, not a percentage. Digests are
lowercase hex. Timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from backend.extraction.prompts.versioning import PromptVersion

__all__ = [
    "CACHE_HASH_BYTES",
    "CacheEntry",
    "CacheKey",
    "CacheStats",
    "CacheStore",
    "ExtractionCache",
    "InMemoryCacheStore",
    "RedisCacheStore",
    "assert_temperature_is_cacheable",
    "payload_digest",
]

CACHE_HASH_BYTES: Final = 32
"""Digest width of a cache key in bytes (32 bytes == 64 hex characters).

Wider than a prompt address (16 bytes) on purpose. A prompt library holds
hundreds of items and its digests are read by humans; this keyspace holds one
entry per (document chunk x prompt x model) across a multi-year backfill —
hundreds of millions — and its digests are read by machines. A collision here
would return one document's response for another document, which is a
fabricated extraction (I3) and would be invisible.
"""

_KEY_DOMAIN: Final = b"quant-research-platform/extraction-cache/1"
"""Domain-separation prefix for cache keys.

Bumping the trailing number invalidates the entire cache — it is a re-address
of every entry, which is a deliberate, expensive act, not a refactor.
"""

_PAYLOAD_DOMAIN: Final = b"quant-research-platform/extraction-payload/1"
"""Domain-separation prefix for the payload digest."""

_REDIS_PREFIX: Final = "extraction:cache:1:"
"""Redis key prefix. Carries the same schema version as the key domain."""


def payload_digest(payload: str) -> str:
    """Return the digest of a payload — the ``document`` component of a key.

    Args:
        payload: The exact text that would be sent to the model (anonymized,
            already chunked). Not the raw document: see the module docstring.

    Returns:
        ``2 * CACHE_HASH_BYTES`` lowercase hex characters (dimensionless).
    """
    digest = hashlib.blake2b(digest_size=CACHE_HASH_BYTES)
    digest.update(_PAYLOAD_DOMAIN)
    digest.update(payload.encode())
    return digest.hexdigest()


def assert_temperature_is_cacheable(temperature: float) -> None:
    """Refuse to cache a call whose temperature is not 0.

    The cache key does not include temperature, because §5-P7 fixes it at 0 for
    all extraction and a constant in a key is noise. That omission is only sound
    while the constant holds, so it is checked rather than commented: a non-zero
    temperature makes the response non-deterministic, at which point one cached
    sample would be silently reused as though it were *the* answer.

    Args:
        temperature: The dimensionless sampling temperature of the call.

    Raises:
        ValueError: if ``temperature`` is not exactly 0. The fix is to add
            temperature to :meth:`CacheKey.build` in the same change that
            allows a non-zero value — not to relax this check.
    """
    if temperature != 0.0:
        msg = (
            f"refusing to cache a call at temperature {temperature!r}: the cache key omits "
            "temperature because §5-P7 fixes it at 0, so caching a sampled response would "
            "reuse one draw as though it were the answer. Allowing non-zero temperature "
            "means adding it to CacheKey.build in the same change"
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The address of one model call: document, prompt version and model.

    Attributes:
        payload_digest: Digest of the anonymized text actually sent.
        prompt_version_hash: The prompt's content address, taken from the
            :class:`~backend.extraction.prompts.versioning.PromptVersion`
            object — never accepted as a loose string, see :meth:`build`.
        model: The model identifier the call is made against, verbatim (e.g.
            ``"anthropic:claude-3-5-haiku-20241022"``). Two models are two
            different answers to the same question, so this is part of the
            address rather than metadata.
    """

    payload_digest: str
    prompt_version_hash: str
    model: str

    @classmethod
    def build(cls, *, payload: str, prompt: PromptVersion, model: str) -> CacheKey:
        """Build the key for one call.

        Takes the prompt *object* rather than its hash. That is the whole
        invalidation guarantee: the hash is derived from the prompt's text on
        every read, so a caller physically cannot combine new prompt text with
        an old address.

        Args:
            payload: The exact anonymized text that will be sent.
            prompt: The prompt version that will be used.
            model: The model identifier. Must be non-empty — an unnamed model
                would collapse two providers' answers into one address.

        Returns:
            The :class:`CacheKey`.

        Raises:
            ValueError: if ``model`` is empty or blank.
        """
        if not model.strip():
            msg = (
                "model must be a non-empty identifier: without it, two models' answers to "
                "the same prompt share one cache address"
            )
            raise ValueError(msg)
        return cls(
            payload_digest=payload_digest(payload),
            prompt_version_hash=prompt.version_hash,
            model=model,
        )

    @property
    def digest(self) -> str:
        """The single-string address, ``2 * CACHE_HASH_BYTES`` lowercase hex chars.

        Length-prefixed over the three components, so no two distinct triples
        can serialize to the same bytes (there is no delimiter that cannot occur
        in a model identifier).
        """
        digest = hashlib.blake2b(digest_size=CACHE_HASH_BYTES)
        digest.update(_KEY_DOMAIN)
        for part in (self.payload_digest, self.prompt_version_hash, self.model):
            encoded = part.encode()
            digest.update(f"{len(encoded)}:".encode())
            digest.update(encoded)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One stored raw model response.

    §5-P7 requires raw responses to be stored, so this holds the response
    **exactly as the provider returned it** — not the parsed object. A response
    that failed schema validation is still cached: re-asking the same model the
    same question at temperature 0 would produce the same malformed answer and
    cost money to rediscover. The failure is recorded on the extraction result
    (:mod:`backend.extraction.tasks.pipeline`), not by omitting the entry.

    Attributes:
        digest: The cache address (:attr:`CacheKey.digest`).
        payload_digest: Digest of the text that was sent.
        prompt_version_hash: The prompt version used.
        model: The model identifier used.
        raw_response: The provider's response text, verbatim.
        stored_at: When the entry was written, UTC.
        task: Extraction task the call served, for provenance only — never part
            of the address (module docstring).
        input_tokens: Prompt tokens the provider reported (count), or ``None``
            when it reported none. Never estimated: an invented token count
            feeds an invented cost figure (I3).
        output_tokens: Response tokens the provider reported (count), or ``None``.
        latency_ms: Wall-clock duration of the call in **milliseconds**, or
            ``None`` when not measured.
    """

    digest: str
    payload_digest: str
    prompt_version_hash: str
    model: str
    raw_response: str
    stored_at: dt.datetime
    task: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class CacheStats:
    """A snapshot of lookup outcomes.

    Attributes:
        hits: Lookups answered from the store (count).
        misses: Lookups that had to call the model (count).
    """

    hits: int
    misses: int

    @property
    def lookups(self) -> int:
        """Total lookups (count)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hits over lookups, a **fraction in [0, 1]**.

        Zero when there have been no lookups. That is a reporting choice, not a
        measurement: a run with no lookups has no hit rate, and returning 0
        rather than raising keeps a dashboard tile from exploding — but Gate G7
        must not be read off an empty run, so callers check
        :attr:`lookups` before believing this number.
        """
        return self.hits / self.lookups if self.lookups else 0.0


@runtime_checkable
class CacheStore(Protocol):
    """Where cache entries live.

    Content-addressed: an entry's key is a digest of everything that produced
    it, so a second write under one address carries the same content as the
    first. There is deliberately no ``update``.
    """

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry at ``digest``, or ``None``."""
        ...

    async def put(self, entry: CacheEntry) -> None:
        """Store ``entry`` at its own digest."""
        ...

    async def discard(self, digest: str) -> bool:
        """Remove one entry. Returns whether it was there."""
        ...

    async def clear(self) -> int:
        """Remove every entry. Returns how many were removed (count)."""
        ...


class InMemoryCacheStore:
    """A :class:`CacheStore` held in process memory.

    Unbounded and never evicting, deliberately: eviction under memory pressure
    would make the hit rate a function of how much else the process was doing,
    and Gate G7 asks a question about the cache, not about the host. A run large
    enough to make that a problem is a run that should be using
    :class:`RedisCacheStore`.

    Not thread-safe, and not safe across event loops. The async methods never
    await.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._entries: dict[str, CacheEntry] = {}

    def __len__(self) -> int:
        """Number of entries held (count)."""
        return len(self._entries)

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry at ``digest``, or ``None`` if there is none.

        Args:
            digest: The cache address.

        Returns:
            The stored :class:`CacheEntry`, or ``None``.
        """
        return self._entries.get(digest)

    async def put(self, entry: CacheEntry) -> None:
        """Store ``entry``, overwriting any entry at the same address.

        Overwriting is safe precisely because the address is a digest of the
        content: an entry at this address was produced by the same payload,
        prompt and model, so the two responses differ only if the provider is
        non-deterministic at temperature 0. The later one wins.

        Args:
            entry: The entry to store.
        """
        self._entries[entry.digest] = entry

    async def discard(self, digest: str) -> bool:
        """Remove one entry.

        Not needed for prompt-change invalidation, which is structural (module
        docstring). This exists for deliberate eviction — a response known to
        be corrupt, or a spend audit.

        Args:
            digest: The cache address.

        Returns:
            True when an entry was removed.
        """
        return self._entries.pop(digest, None) is not None

    async def clear(self) -> int:
        """Remove every entry.

        Returns:
            How many entries were removed (count).
        """
        removed = len(self._entries)
        self._entries.clear()
        return removed


class RedisCacheStore:
    """A :class:`CacheStore` backed by Redis — the stack's cache (§3).

    **Unverified against a server.** ``_encode``/``_decode`` are pure and are
    round-tripped by ``backend/tests/extraction/test_cache.py``; ``get``,
    ``put``, ``discard`` and ``clear`` have never been executed against Redis,
    because the test environment has none and standing up a fake would mean
    changing the dependency set. A tested codec is not a tested store, and this
    docstring says so rather than letting the class's existence imply coverage.

    Entries are stored as JSON under :data:`_REDIS_PREFIX` plus the digest.
    JSON rather than pickle because a cache entry crossing a process boundary is
    data, and pickle would make it code.

    No TTL by default. See the module docstring: an expiring extraction cache
    turns a hit into a spend on a schedule nobody chose.
    """

    def __init__(self, client: Redis, *, ttl_s: int | None = None) -> None:
        """Bind the store to a Redis client.

        Args:
            client: An ``redis.asyncio.Redis`` instance, configured and owned by
                the caller. Not created here — connection lifecycle belongs to
                the application, not to a cache adapter.
            ttl_s: Expiry in **seconds**, or ``None`` for no expiry (the
                default, and what Gate G7 assumes).

        Raises:
            ValueError: if ``ttl_s`` is not positive.
        """
        if ttl_s is not None and ttl_s <= 0:
            msg = f"ttl_s must be positive seconds or None; got {ttl_s!r}"
            raise ValueError(msg)
        self._client = client
        self._ttl_s = ttl_s

    @staticmethod
    def _redis_key(digest: str) -> str:
        """Return the Redis key holding the entry at ``digest``."""
        return f"{_REDIS_PREFIX}{digest}"

    @staticmethod
    def _encode(entry: CacheEntry) -> str:
        """Serialize an entry to JSON text."""
        return json.dumps(
            {
                "digest": entry.digest,
                "payload_digest": entry.payload_digest,
                "prompt_version_hash": entry.prompt_version_hash,
                "model": entry.model,
                "raw_response": entry.raw_response,
                "stored_at": entry.stored_at.isoformat(),
                "task": entry.task,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "latency_ms": entry.latency_ms,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _decode(raw: str | bytes) -> CacheEntry:
        """Deserialize an entry from JSON text.

        Raises:
            ValueError: if the stored value is not a JSON object with the
                expected fields. A malformed entry is reported rather than
                repaired: silently substituting defaults would hand the caller
                an entry whose ``raw_response`` never came from a model.
        """
        text = raw.decode() if isinstance(raw, bytes) else raw
        loaded: Any = json.loads(text)
        if not isinstance(loaded, dict):
            msg = f"cache entry is not a JSON object; got {type(loaded).__name__}"
            raise ValueError(msg)
        return CacheEntry(
            digest=str(loaded["digest"]),
            payload_digest=str(loaded["payload_digest"]),
            prompt_version_hash=str(loaded["prompt_version_hash"]),
            model=str(loaded["model"]),
            raw_response=str(loaded["raw_response"]),
            stored_at=dt.datetime.fromisoformat(str(loaded["stored_at"])),
            task=None if loaded["task"] is None else str(loaded["task"]),
            input_tokens=None if loaded["input_tokens"] is None else int(loaded["input_tokens"]),
            output_tokens=None if loaded["output_tokens"] is None else int(loaded["output_tokens"]),
            latency_ms=None if loaded["latency_ms"] is None else float(loaded["latency_ms"]),
        )

    async def get(self, digest: str) -> CacheEntry | None:
        """Return the entry at ``digest``, or ``None`` if Redis has none.

        Args:
            digest: The cache address.

        Returns:
            The stored :class:`CacheEntry`, or ``None``.
        """
        raw = await self._client.get(self._redis_key(digest))
        if raw is None:
            return None
        return self._decode(raw)

    async def put(self, entry: CacheEntry) -> None:
        """Store ``entry`` under its digest, with the configured TTL if any.

        Args:
            entry: The entry to store.
        """
        await self._client.set(self._redis_key(entry.digest), self._encode(entry), ex=self._ttl_s)

    async def discard(self, digest: str) -> bool:
        """Remove one entry.

        Args:
            digest: The cache address.

        Returns:
            True when Redis reported a key was deleted.
        """
        return bool(await self._client.delete(self._redis_key(digest)))

    async def clear(self) -> int:
        """Remove every entry under this store's prefix.

        Scans and deletes by prefix rather than issuing ``FLUSHDB``: the Redis
        instance is shared with Celery (§3), and flushing it would silently
        destroy queued work.

        Returns:
            How many keys were deleted (count).
        """
        removed = 0
        async for key in self._client.scan_iter(match=f"{_REDIS_PREFIX}*"):
            removed += int(await self._client.delete(key))
        return removed


class ExtractionCache:
    """A store plus the hit/miss accounting Gate G7 is measured on.

    Separated from :class:`CacheStore` because they answer different questions.
    A store answers "is this response here"; this answers "how often did we not
    have to spend money", which is a property of a *run* and has to be resettable
    at a run boundary — a lifetime hit rate blended across every run since
    process start is not the number G7 asks for.
    """

    def __init__(self, store: CacheStore) -> None:
        """Bind the accounting to a store.

        Args:
            store: Where entries live.
        """
        self._store = store
        self._hits = 0
        self._misses = 0

    @property
    def store(self) -> CacheStore:
        """The underlying store, for direct inspection in tests and tooling."""
        return self._store

    @property
    def stats(self) -> CacheStats:
        """Lookup outcomes since construction or the last :meth:`reset_stats`."""
        return CacheStats(hits=self._hits, misses=self._misses)

    def reset_stats(self) -> CacheStats:
        """Zero the counters and return what they were.

        Called at a run boundary so each run reports its own hit rate.

        Returns:
            The :class:`CacheStats` snapshot taken before zeroing.
        """
        snapshot = self.stats
        self._hits = 0
        self._misses = 0
        return snapshot

    async def lookup(self, key: CacheKey) -> CacheEntry | None:
        """Look up one call and record the outcome.

        Args:
            key: The address of the call.

        Returns:
            The stored entry on a hit, ``None`` on a miss.
        """
        entry = await self._store.get(key.digest)
        if entry is None:
            self._misses += 1
        else:
            self._hits += 1
        return entry

    async def store_response(
        self,
        key: CacheKey,
        raw_response: str,
        *,
        task: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> CacheEntry:
        """Store a raw response at ``key`` and return the entry written.

        Args:
            key: The address of the call.
            raw_response: The provider's response text, verbatim (§5-P7).
            task: Extraction task served, for provenance. Never keyed on.
            input_tokens: Prompt tokens as *reported by the provider* (count),
                or ``None``. Never estimated (I3).
            output_tokens: Response tokens as reported (count), or ``None``.
            latency_ms: Measured wall-clock duration in milliseconds, or
                ``None``.

        Returns:
            The :class:`CacheEntry` written.
        """
        entry = CacheEntry(
            digest=key.digest,
            payload_digest=key.payload_digest,
            prompt_version_hash=key.prompt_version_hash,
            model=key.model,
            raw_response=raw_response,
            stored_at=dt.datetime.now(tz=dt.UTC),
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        await self._store.put(entry)
        return entry
