"""Tests for backend.core.crypto: Fernet at rest, KEK discipline, masked display.

Every credential-shaped literal here is obviously fake and generated or
hand-written for the test; nothing in this file is or ever was a real key.
"""

from __future__ import annotations

import base64
import inspect
import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import SecretStr

from backend.core import crypto
from backend.core.config import Settings, get_settings
from backend.core.crypto import (
    MASK_TAIL_CHARS,
    KeyEncryptionKeyMalformedError,
    KeyEncryptionKeyMissingError,
    SecretCipher,
    SecretDecryptionError,
    SecretsCryptoError,
    decrypt_secret,
    encrypt_secret,
    generate_key_encryption_key,
    load_cipher,
    mask_secret,
    verify_key_encryption_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_KEK = generate_key_encryption_key()
_OTHER_KEK = generate_key_encryption_key()

# Obviously-fake, correctly-shaped provider credentials.
_OPENAI_SHAPED = "sk-proj-FAKE0000example1111example2222example3333"
_GITHUB_SHAPED = "ghp_FAKEfakeFAKEfakeFAKEfakeFAKEfake0000"
_AWS_SHAPED = "AKIAFAKEEXAMPLE00000"

_FULLY_MASKED_RENDERING = "..."
"""What mask_secret emits when the secret is too short to reveal a tail from."""

_SHORTEST_MASKABLE = 12
"""Shortest secret that reveals anything: MASK_TAIL_CHARS shown, 8 still hidden."""


def _settings_with(kek: str | None) -> Settings:
    """Build Settings with an explicit KEK, bypassing environment discovery."""
    return Settings(secrets_kek=None if kek is None else SecretStr(kek))


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plaintext",
    [_OPENAI_SHAPED, _GITHUB_SHAPED, "x", "a" * 4096, "ünïcode-ключ-🔐"],
)
def test_round_trip_encrypt_decrypt(plaintext: str) -> None:
    """Whatever goes in comes back out byte-for-byte, including non-ASCII."""
    cipher = SecretCipher(_KEK)
    assert cipher.decrypt(cipher.encrypt(plaintext)) == plaintext


def test_round_trip_through_module_level_helpers() -> None:
    """The convenience wrappers agree with the cipher object."""
    config = _settings_with(_KEK)
    token = encrypt_secret(_OPENAI_SHAPED, settings=config)
    assert decrypt_secret(token, settings=config) == _OPENAI_SHAPED


def test_ciphertext_does_not_contain_the_plaintext() -> None:
    """Storage never holds the credential in a recoverable-by-eye form."""
    token = SecretCipher(_KEK).encrypt(_OPENAI_SHAPED)
    assert _OPENAI_SHAPED not in token
    assert base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).find(b"sk-proj") == -1


def test_encrypting_twice_yields_different_tokens() -> None:
    """Fernet embeds an IV and timestamp, so ciphertext is not a stable fingerprint."""
    cipher = SecretCipher(_KEK)
    assert cipher.encrypt(_OPENAI_SHAPED) != cipher.encrypt(_OPENAI_SHAPED)


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def test_tampered_ciphertext_fails_to_decrypt() -> None:
    """A single flipped character breaks authentication rather than decoding to garbage."""
    cipher = SecretCipher(_KEK)
    token = cipher.encrypt(_OPENAI_SHAPED)
    flipped = "B" if token[20] != "B" else "C"
    tampered = token[:20] + flipped + token[21:]
    with pytest.raises(SecretDecryptionError, match="failed authentication"):
        cipher.decrypt(tampered)


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda t: t[:-8], id="truncated"),
        pytest.param(lambda t: t[:0], id="emptied"),
        pytest.param(lambda t: "Z" + t[1:], id="version-byte-mangled"),
        pytest.param(lambda t: t[:-8][::-1] + t[-8:], id="reordered"),
    ],
)
def test_mangled_ciphertext_fails_to_decrypt(mangle: Callable[[str], str]) -> None:
    """Truncation, emptiness, a bad version byte and reordering all raise."""
    cipher = SecretCipher(_KEK)
    tampered = mangle(cipher.encrypt(_OPENAI_SHAPED))
    with pytest.raises(SecretDecryptionError):
        cipher.decrypt(tampered)


def test_characters_appended_after_the_base64_padding_are_inert() -> None:
    """Documents a real base64 property so nobody mistakes it for a bypass.

    A Fernet token ends in ``=`` padding and Python's base64 decoder discards
    everything after it, so appended characters change neither the
    authenticated bytes nor the plaintext. They cannot be used to alter what
    decryption returns; only edits *inside* the token break authentication,
    which the tampering tests above pin down.
    """
    cipher = SecretCipher(_KEK)
    token = cipher.encrypt(_OPENAI_SHAPED)
    assert token.endswith("=")
    assert cipher.decrypt(token + "AAAA") == _OPENAI_SHAPED


def test_ciphertext_written_under_another_kek_does_not_decrypt() -> None:
    """A database dump is useless without the environment KEK that wrapped it (§7)."""
    token = SecretCipher(_KEK).encrypt(_OPENAI_SHAPED)
    with pytest.raises(SecretDecryptionError):
        SecretCipher(_OTHER_KEK).decrypt(token)


def test_decrypt_never_returns_its_input_on_failure() -> None:
    """The failure mode is an exception, never a pass-through of the stored bytes."""
    cipher = SecretCipher(_KEK)
    with pytest.raises(SecretDecryptionError) as excinfo:
        cipher.decrypt("not-a-fernet-token")
    assert "not-a-fernet-token" not in str(excinfo.value)


# --------------------------------------------------------------------------
# KEK discipline: missing or malformed must raise, never degrade
# --------------------------------------------------------------------------


def test_missing_kek_raises() -> None:
    """An unset SECRETS_KEK is a hard failure, not a reason to store plaintext (I5)."""
    with pytest.raises(KeyEncryptionKeyMissingError, match="not configured"):
        load_cipher(_settings_with(None))


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_kek_is_treated_as_missing(blank: str) -> None:
    """`SECRETS_KEK=` in a .env file must read as unset, not as a usable key."""
    with pytest.raises(KeyEncryptionKeyMissingError):
        load_cipher(_settings_with(blank))


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        ("not base64 at all!!", "not urlsafe-base64"),
        ("short", "not decodable urlsafe-base64"),
        (base64.urlsafe_b64encode(b"\x00" * 16).decode(), "decodes to 16 bytes"),
        (base64.urlsafe_b64encode(b"\x00" * 64).decode(), "decodes to 64 bytes"),
        ("a" * 43, "not decodable urlsafe-base64"),
    ],
)
def test_malformed_kek_raises_with_a_precise_reason(bad: str, reason: str) -> None:
    """Each malformation names what is wrong rather than surfacing a base64 stack trace."""
    with pytest.raises(KeyEncryptionKeyMalformedError, match=reason):
        load_cipher(_settings_with(bad))


def test_a_valid_kek_survives_surrounding_whitespace() -> None:
    """A trailing newline from a .env file is a formatting artifact, not a malformed key."""
    assert SecretCipher(f"  {_KEK}\n").decrypt(SecretCipher(_KEK).encrypt("v")) == "v"


def test_kek_is_validated_at_load_not_at_first_use() -> None:
    """Construction itself rejects a bad key, so startup fails before any secret is handled."""
    with pytest.raises(KeyEncryptionKeyMalformedError):
        SecretCipher("definitely-not-a-fernet-key")
    with pytest.raises(KeyEncryptionKeyMalformedError):
        verify_key_encryption_key(_settings_with("definitely-not-a-fernet-key"))
    verify_key_encryption_key(_settings_with(_KEK))  # valid key: silent


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda c: encrypt_secret(_OPENAI_SHAPED, settings=c), id="encrypt"),
        pytest.param(lambda c: decrypt_secret("anything", settings=c), id="decrypt"),
    ],
)
def test_no_plaintext_fallback_when_the_kek_is_missing(
    operation: Callable[[Settings], str],
) -> None:
    """Neither entry point has a code path that proceeds unencrypted."""
    config = _settings_with(None)
    with pytest.raises(KeyEncryptionKeyMissingError):
        operation(config)


def test_every_kek_failure_shares_one_base_class() -> None:
    """Callers can fail closed on SecretsCryptoError without enumerating subclasses."""
    assert issubclass(KeyEncryptionKeyMissingError, SecretsCryptoError)
    assert issubclass(KeyEncryptionKeyMalformedError, SecretsCryptoError)
    assert issubclass(SecretDecryptionError, SecretsCryptoError)


def test_kek_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default path sources the KEK from the environment only (§7)."""
    monkeypatch.setenv("SECRETS_KEK", _KEK)
    get_settings.cache_clear()
    cipher = load_cipher()
    assert cipher.decrypt(cipher.encrypt("v")) == "v"


def test_error_messages_never_echo_the_kek() -> None:
    """A malformed-key diagnostic must not print the key material into logs."""
    almost = base64.urlsafe_b64encode(b"\x11" * 31).decode()
    with pytest.raises(KeyEncryptionKeyMalformedError) as excinfo:
        SecretCipher(almost)
    assert almost not in str(excinfo.value)


def test_cipher_repr_does_not_leak_the_key() -> None:
    """The cipher is safe to interpolate into a diagnostic."""
    rendered = repr(SecretCipher(_KEK))
    assert _KEK not in rendered
    assert "redacted" in rendered


# --------------------------------------------------------------------------
# Masked display (§6.5): last four characters, no reveal path
# --------------------------------------------------------------------------


def test_masked_display_shows_public_prefix_and_last_four() -> None:
    """The directive's rendering: `sk-...4f2a`."""
    assert mask_secret("sk-abcdefghijklmnopqrstuvwx4f2a") == "sk-...4f2a"


@pytest.mark.parametrize(
    ("full", "expected"),
    [
        (_OPENAI_SHAPED, "sk-proj-...3333"),
        (_GITHUB_SHAPED, "ghp_...0000"),
        (_AWS_SHAPED, "AKIA...0000"),
        ("xoxb-1111111111-2222222222-abcdefghijkl", "xoxb-...ijkl"),
        ("some-unknown-vendor-credential-zzzz", "...zzzz"),
    ],
)
def test_masked_display_only_reveals_known_vendor_prefixes(full: str, expected: str) -> None:
    """Prefixes come from a closed published vocabulary; anything else shows only the tail."""
    assert mask_secret(full) == expected


@pytest.mark.parametrize("short", ["abc", "abcdefghijk", "sk-abcdef"])
def test_masked_display_reveals_nothing_for_short_secrets(short: str) -> None:
    """Too short to hide 8 characters behind the tail: reveal nothing at all."""
    assert mask_secret(short) == _FULLY_MASKED_RENDERING


@pytest.mark.parametrize("blank", ["", "   "])
def test_masking_an_empty_secret_raises(blank: str) -> None:
    """Masking a value that is not there is a caller bug, not a display case."""
    with pytest.raises(ValueError, match="empty secret"):
        mask_secret(blank)


@given(secret=st.text(min_size=_SHORTEST_MASKABLE).filter(lambda s: len(s.strip()) >= 12))
@hypothesis_settings(max_examples=500, deadline=None)
def test_masked_display_never_contains_the_full_secret(secret: str) -> None:
    """Property: for any secret long enough to reveal a tail, the full value never appears."""
    candidate = secret.strip()
    masked = mask_secret(secret)
    assert secret not in masked
    assert candidate not in masked
    assert masked.endswith(candidate[-MASK_TAIL_CHARS:])
    assert len(masked) < len(candidate)


@given(secret=st.text(min_size=1).filter(lambda s: 0 < len(s.strip()) < _SHORTEST_MASKABLE))
@hypothesis_settings(max_examples=200, deadline=None)
def test_short_secrets_all_render_to_one_constant(secret: str) -> None:
    """Property: below the reveal threshold the rendering does not depend on the input.

    This is the stronger guarantee, and it is why the containment property
    above is scoped to longer secrets: a one-character secret like ``.`` *is*
    trivially a substring of the constant ``...``, but the constant is emitted
    for every short secret alike, so observing it tells an attacker nothing.
    """
    assert mask_secret(secret) == _FULLY_MASKED_RENDERING


def test_module_offers_no_reveal_path() -> None:
    """§6.5: nothing here un-masks a secret for display, under any name or flag."""
    forbidden = re.compile(r"reveal|unmask|unredact|plaintext_of|full_secret|show_secret", re.I)
    public = [name for name in dir(crypto) if not name.startswith("_")]
    assert public, "module exposes nothing at all — the guard would be vacuous"
    assert [name for name in public if forbidden.search(name)] == []
    # ...and the one rendering helper takes the secret and nothing else, so no
    # keyword argument can widen the window.
    assert list(inspect.signature(mask_secret).parameters) == ["secret"]
