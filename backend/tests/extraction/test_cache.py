"""P7.6 cache tests: the address, structural invalidation, and the G7 hit rate.

Three things are proved here, in order of how much they matter:

1. **The key is what §5-P7 says it is** — ``hash(document + prompt_version +
   model)`` — and each of the three components genuinely moves the address.
2. **A prompt change invalidates by construction, not by a cleanup step.** There
   is no invalidation function to call and no sweep to schedule: an edited
   prompt has a different content address, so it addresses entries that do not
   exist. The corollary is tested too, because it is the reason rollback is
   cheap: pointing a task back at the previous prompt makes yesterday's entries
   addressable again, at full hit rate, with no re-spend.
3. **The hit rate G7 asks about is measured**, over a batch run through the real
   pipeline rather than by poking the counters.

What the hit-rate measurement is and is not
--------------------------------------------

It measures the *addressing mechanism*: given a repeated batch, does the cache
answer without calling the model. It is **not** a forecast of a production
backfill's hit rate, which depends on how often documents and prompts actually
change and cannot be known before real runs exist (B4). Reporting the number
below as though it settled Gate G7 would be exactly the false precision this
project exists to avoid.

The model client throughout is a deliberate test double. Its responses are this
file's input, not a provider's output — B4 leaves the live path unexercised, and
nothing here claims otherwise (I3).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from backend.extraction.anonymize import AnonymizerConfig
from backend.extraction.cache import (
    CACHE_HASH_BYTES,
    CacheEntry,
    CacheKey,
    CacheStats,
    ExtractionCache,
    InMemoryCacheStore,
    RedisCacheStore,
    assert_temperature_is_cacheable,
    payload_digest,
)
from backend.extraction.entities import company
from backend.extraction.prompts.versioning import PromptVersion
from backend.extraction.tasks import (
    ExtractionOutput,
    ExtractionPipeline,
    ExtractionTask,
    ModelRequest,
    ModelResponse,
    SourceDocument,
)

_MODEL = "anthropic:a-cost-tier-model"
_OTHER_MODEL = "openai:another-cost-tier-model"

_GOLDEN_HIT_RATE = 0.90
"""Gate G7's bar: cache hit rate above 90% on repeat runs. A fraction, not a percent."""


def _prompt(system: str = "Report the change.", template: str = "$document\n") -> PromptVersion:
    """Build a prompt version for addressing tests."""
    return PromptVersion(
        name="probe_task", system=system, template=template, schema_digest="deadbeef"
    )


class _Score(ExtractionOutput):
    """A minimal output model for pipeline-level cache tests."""

    tone_shift: float


_VALID = '{"tone_shift": 0.0}'


class CountingClient:
    """A model client that counts calls and replays one canned response."""

    def __init__(self) -> None:
        """Start at zero calls."""
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Count the call and answer."""
        self.calls += 1
        return ModelResponse(text=_VALID, model=request.model, input_tokens=5, output_tokens=3)


def _task(system: str = "Report the change.") -> ExtractionTask:
    """Build a task whose prompt address follows its system text."""
    return ExtractionTask(
        name="probe_task", output_model=_Score, system=system, template="$document\n"
    )


# --------------------------------------------------------------------------
# The address is document + prompt version + model, and nothing else
# --------------------------------------------------------------------------


def test_the_key_carries_exactly_the_three_components_the_directive_names() -> None:
    """§5-P7 spells the key out; the dataclass must not quietly hold a fourth thing."""
    key = CacheKey.build(payload="some anonymized text", prompt=_prompt(), model=_MODEL)
    assert set(CacheKey.__dataclass_fields__) == {
        "payload_digest",
        "prompt_version_hash",
        "model",
    }
    assert key.payload_digest == payload_digest("some anonymized text")
    assert key.prompt_version_hash == _prompt().version_hash
    assert key.model == _MODEL
    assert len(key.digest) == 2 * CACHE_HASH_BYTES


@pytest.mark.parametrize(
    ("payload", "prompt", "model"),
    [
        ("different text", _prompt(), _MODEL),
        ("text", _prompt(system="Report the change. Be brief."), _MODEL),
        ("text", _prompt(template="Document:\n$document\n"), _MODEL),
        ("text", _prompt(), _OTHER_MODEL),
    ],
)
def test_changing_any_component_changes_the_address(
    payload: str, prompt: PromptVersion, model: str
) -> None:
    """Each of the three is load-bearing; a component that did not move the key is noise."""
    baseline = CacheKey.build(payload="text", prompt=_prompt(), model=_MODEL)
    assert CacheKey.build(payload=payload, prompt=prompt, model=model).digest != baseline.digest


def test_the_schema_is_part_of_the_address_even_when_the_prompt_text_is_not_touched() -> None:
    """The easy field to leave out, and the expensive one.

    If the output schema gains a field while the prompt text is unchanged, a
    response cached under the old schema would still be addressable and would be
    re-validated — possibly re-*accepted* — under the new one.
    """
    same_text_new_schema = PromptVersion(
        name="probe_task", system="Report the change.", template="$document\n", schema_digest="cafe"
    )
    assert (
        CacheKey.build(payload="text", prompt=same_text_new_schema, model=_MODEL).digest
        != CacheKey.build(payload="text", prompt=_prompt(), model=_MODEL).digest
    )


def test_the_address_is_built_from_a_prompt_object_and_not_from_a_loose_hash() -> None:
    """The invalidation guarantee rests on this: no caller can pass a stale address.

    ``version_hash`` is a property derived from the text on every read, so
    combining new prompt text with an old address is not something a caller can
    physically express.
    """
    with pytest.raises(AttributeError):
        CacheKey.build(payload="text", prompt="a-hash-string", model=_MODEL)  # type: ignore[arg-type]


def test_two_distinct_component_triples_cannot_collapse_to_one_address() -> None:
    """Length prefixing: there is no delimiter that cannot occur in a model identifier."""
    left = CacheKey(payload_digest="ab", prompt_version_hash="c", model="d")
    right = CacheKey(payload_digest="a", prompt_version_hash="bc", model="d")
    assert left.digest != right.digest


def test_an_unnamed_model_is_refused() -> None:
    """Without it, two providers' answers to one prompt would share an address."""
    with pytest.raises(ValueError, match="model must be a non-empty identifier"):
        CacheKey.build(payload="text", prompt=_prompt(), model="   ")


@pytest.mark.parametrize("temperature", [0.1, 0.7, 2.0])
def test_caching_a_sampled_call_is_refused(temperature: float) -> None:
    """Temperature is absent from the key because it is 0; the check is why that is sound.

    At a sampling temperature one draw would be reused as though it were *the*
    answer. The fix is to add temperature to the key in the same change that
    permits a non-zero value — not to relax this.
    """
    with pytest.raises(ValueError, match="refusing to cache a call at temperature"):
        assert_temperature_is_cacheable(temperature)


def test_temperature_zero_is_cacheable() -> None:
    """The only value §5-P7 permits passes without comment."""
    assert_temperature_is_cacheable(0.0)


# --------------------------------------------------------------------------
# Invalidation is structural: no sweep, no cleanup step, no window
# --------------------------------------------------------------------------


async def test_editing_a_prompt_makes_previously_cached_responses_unreachable() -> None:
    """§6.5: a prompt change invalidates affected documents *automatically*.

    Nothing is deleted and nothing is scanned. The edited prompt simply
    addresses an entry that was never written.
    """
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    document = SourceDocument("d1", "some filing text")

    await pipeline.run(_task(), document, model=_MODEL)
    await pipeline.run(_task(), document, model=_MODEL)
    assert client.calls == 1, "the unchanged prompt must hit"

    await pipeline.run(_task("Report the change. Be brief."), document, model=_MODEL)
    assert client.calls == 2, "the edited prompt must miss"


async def test_rolling_a_prompt_back_makes_the_old_entries_addressable_again() -> None:
    """Rollback is selection, so it costs nothing: yesterday's entries are still there.

    This is the corollary of structural invalidation and the reason §6.5 can
    offer one-click rollback without a re-spend. The old entries were never
    deleted — they were only unaddressable while the edited prompt was in force.
    """
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    document = SourceDocument("d1", "some filing text")
    original, edited = _task(), _task("Report the change. Be brief.")

    await pipeline.run(original, document, model=_MODEL)
    await pipeline.run(edited, document, model=_MODEL)
    assert client.calls == 2

    run = await pipeline.run(original, document, model=_MODEL)
    assert client.calls == 2, "returning to the earlier prompt must not spend again"
    assert run.chunks[0].cache_hit


async def test_two_models_do_not_share_one_answer() -> None:
    """Two models are two different answers to the same question, not one cached value."""
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    document = SourceDocument("d1", "some filing text")

    await pipeline.run(_task(), document, model=_MODEL)
    await pipeline.run(_task(), document, model=_OTHER_MODEL)

    assert client.calls == 2


async def test_a_masking_change_re_addresses_the_document() -> None:
    """The keyed payload is the text *actually sent*, so a masking change must miss.

    A key built from the pre-masking text would let an anonymizer change
    silently reuse responses to text that no longer exists — and a date left in
    by a relaxed configuration is precisely the contamination channel §5-P7's
    "strip all dates" exists to close.
    """
    document = SourceDocument(
        "d1", "Acme Corp. reported on March 3, 2024 that a risk exists.", (company("Acme Corp."),)
    )
    client = CountingClient()
    strict = ExtractionPipeline(client=client)
    relaxed = ExtractionPipeline(
        client=client,
        cache=strict.cache,
        anonymizer=AnonymizerConfig(mask_dates=False),
    )

    await strict.run(_task(), document, model=_MODEL)
    await relaxed.run(_task(), document, model=_MODEL)

    assert client.calls == 2


# --------------------------------------------------------------------------
# Gate G7's hit rate, measured over a repeated batch
# --------------------------------------------------------------------------


async def test_a_repeated_batch_hits_above_the_gate_g7_bar() -> None:
    """G7: cache hit rate above 90% on repeat runs.

    Measured through the real pipeline over a real batch, not by incrementing
    counters: twenty documents cold, then the same twenty again with one new
    document mixed in, which is the shape a steady-state run actually has.

    This measures the *addressing mechanism*. It is not a forecast of a
    production backfill's hit rate — that depends on how often documents and
    prompts change, which cannot be known before real runs exist (B4).
    """
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    task = _task()
    batch = [SourceDocument(f"doc-{i}", f"filing text number {i}") for i in range(20)]

    cold = await pipeline.run_batch(task, batch, model=_MODEL)
    assert sum(run.stats.hits for run in cold) == 0
    assert client.calls == 20

    repeat_batch = [*batch, SourceDocument("doc-new", "a filing nobody has read yet")]
    repeat = await pipeline.run_batch(task, repeat_batch, model=_MODEL)

    hits = sum(run.stats.hits for run in repeat)
    lookups = sum(run.stats.lookups for run in repeat)
    hit_rate = hits / lookups
    assert lookups == 21
    assert hit_rate > _GOLDEN_HIT_RATE, f"repeat-run hit rate {hit_rate:.3f}"
    assert client.calls == 21, "only the unseen document cost a call"


async def test_an_identical_repeat_run_hits_every_time() -> None:
    """The degenerate case the gate's wording describes: nothing changed, nothing spent."""
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    task = _task()
    batch = [SourceDocument(f"doc-{i}", f"filing text number {i}") for i in range(10)]

    await pipeline.run_batch(task, batch, model=_MODEL)
    repeat = await pipeline.run_batch(task, batch, model=_MODEL)

    assert all(run.stats.hit_rate == 1.0 for run in repeat)
    assert client.calls == 10


def test_a_run_with_no_lookups_reports_zero_rather_than_dividing_by_zero() -> None:
    """A dashboard tile must not explode, and G7 must not be read off an empty run."""
    empty = CacheStats(hits=0, misses=0)
    assert empty.lookups == 0
    assert empty.hit_rate == 0.0


async def test_run_statistics_are_the_runs_own_and_not_the_processs() -> None:
    """A lifetime rate blended across every run since process start is not G7's number."""
    client = CountingClient()
    pipeline = ExtractionPipeline(client=client)
    task = _task()
    first = SourceDocument("d1", "text one")
    second = SourceDocument("d2", "text two")

    await pipeline.run(task, first, model=_MODEL)
    repeat_of_first = await pipeline.run(task, first, model=_MODEL)
    fresh_second = await pipeline.run(task, second, model=_MODEL)

    assert repeat_of_first.stats == CacheStats(hits=1, misses=0)
    assert fresh_second.stats == CacheStats(hits=0, misses=1)
    # The shared counter has seen all three; the per-run figures have not.
    assert pipeline.cache.stats.lookups == 3


# --------------------------------------------------------------------------
# The in-memory store
# --------------------------------------------------------------------------


def _entry(digest: str = "abc") -> CacheEntry:
    """Build a stored entry."""
    return CacheEntry(
        digest=digest,
        payload_digest="pd",
        prompt_version_hash="pv",
        model=_MODEL,
        raw_response='{"tone_shift": 0.5}',
        stored_at=dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC),
        task="probe_task",
        input_tokens=11,
        output_tokens=7,
        latency_ms=42.5,
    )


async def test_the_in_memory_store_round_trips_and_discards() -> None:
    """The basic contract, including that a miss is ``None`` rather than an exception."""
    store = InMemoryCacheStore()
    assert await store.get("abc") is None

    await store.put(_entry())
    assert await store.get("abc") == _entry()
    assert len(store) == 1

    assert await store.discard("abc") is True
    assert await store.discard("abc") is False
    assert await store.get("abc") is None


async def test_clearing_the_store_reports_how_much_it_removed() -> None:
    """A deliberate eviction is an act with a size, and the caller should see it."""
    store = InMemoryCacheStore()
    for index in range(3):
        await store.put(_entry(digest=f"d{index}"))
    assert await store.clear() == 3
    assert len(store) == 0


async def test_a_lookup_records_the_outcome_it_observed() -> None:
    """Hit and miss counting is the whole reason the accounting is separate from the store."""
    cache = ExtractionCache(InMemoryCacheStore())
    key = CacheKey.build(payload="text", prompt=_prompt(), model=_MODEL)

    assert await cache.lookup(key) is None
    entry = await cache.store_response(key, '{"tone_shift": 0.0}', task="probe_task")
    assert await cache.lookup(key) == entry

    assert cache.stats == CacheStats(hits=1, misses=1)
    assert cache.reset_stats() == CacheStats(hits=1, misses=1)
    assert cache.stats == CacheStats(hits=0, misses=0)


async def test_a_stored_response_is_kept_verbatim() -> None:
    """§5-P7 requires raw responses; normalizing one would destroy the evidence."""
    cache = ExtractionCache(InMemoryCacheStore())
    key = CacheKey.build(payload="text", prompt=_prompt(), model=_MODEL)
    raw = '  {"tone_shift": 0.5}\n\n'

    entry = await cache.store_response(key, raw, task="probe_task")

    assert entry.raw_response == raw
    assert entry.digest == key.digest


# --------------------------------------------------------------------------
# The Redis store: its codec only. The client interaction is unexercised.
# --------------------------------------------------------------------------


def test_the_redis_codec_round_trips_an_entry_without_loss() -> None:
    """Covers the serialization and **nothing else about Redis**.

    There is no Redis in the test environment, so
    :class:`~backend.extraction.cache.RedisCacheStore`'s get/put/discard/clear
    have never been executed against a server and are unverified. The codec is a
    pure function and is tested as one; calling that coverage of the store would
    be a claim this suite cannot support (I3).
    """
    entry = _entry()
    assert RedisCacheStore._decode(RedisCacheStore._encode(entry)) == entry


def test_the_redis_codec_produces_data_rather_than_code() -> None:
    """JSON, not pickle: a cache entry crossing a process boundary must not be executable."""
    encoded = RedisCacheStore._encode(_entry())
    decoded = json.loads(encoded)
    assert decoded["raw_response"] == '{"tone_shift": 0.5}'
    assert decoded["latency_ms"] == 42.5


def test_a_malformed_redis_entry_is_reported_rather_than_repaired() -> None:
    """Substituting defaults would hand the caller a response that never came from a model."""
    with pytest.raises(ValueError, match="cache entry is not a JSON object"):
        RedisCacheStore._decode("[1, 2, 3]")


def test_a_redis_store_refuses_a_meaningless_expiry() -> None:
    """A non-positive TTL is a configuration mistake, not an instruction."""
    with pytest.raises(ValueError, match="ttl_s must be positive"):
        RedisCacheStore(client=None, ttl_s=0)  # type: ignore[arg-type]
