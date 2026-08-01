"""Fernet encryption at rest for operator-supplied credentials (DIRECTIVE §7, I5).

Provider API keys are stored as Fernet tokens. The key-encryption key (KEK)
that wraps them comes from the environment (``SECRETS_KEK``) and **only** from
the environment — never from the database — so a database dump on its own
cannot decrypt anything it contains.

Two rules this module enforces rather than documents:

* **No plaintext fallback.** Every entry point raises when the KEK is unset or
  malformed. Storing a credential unencrypted "just this once" is precisely the
  failure I5 forbids, so there is no code path that does it.
* **No reveal path.** :func:`mask_secret` is the only rendering helper and it
  has no unmasking counterpart, no ``reveal=True`` switch, and no
  full-value variant (§6.5: no endpoint returns a full key under any
  circumstance). :func:`decrypt_secret` exists solely so the backend can *use*
  a credential when calling a provider; its result must never reach a response
  body, a template, or a log line.

Units: KEK and ciphertext are urlsafe-base64 ASCII text; plaintext secrets are
``str`` and are encoded as UTF-8 before encryption.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import TYPE_CHECKING, Final

from cryptography.fernet import Fernet, InvalidToken

if TYPE_CHECKING:
    from backend.core.config import Settings

__all__ = [
    "FERNET_KEY_BYTES",
    "MASK_TAIL_CHARS",
    "KeyEncryptionKeyMalformedError",
    "KeyEncryptionKeyMissingError",
    "SecretCipher",
    "SecretDecryptionError",
    "SecretsCryptoError",
    "decrypt_secret",
    "encrypt_secret",
    "generate_key_encryption_key",
    "load_cipher",
    "mask_secret",
    "verify_key_encryption_key",
]

FERNET_KEY_BYTES: Final = 32
"""Decoded length of a Fernet key in bytes (128-bit signing + 128-bit encryption half)."""

MASK_TAIL_CHARS: Final = 4
"""Number of trailing characters :func:`mask_secret` reveals (§6.5 ``sk-...4f2a``)."""

_MIN_HIDDEN_CHARS: Final = 8
"""Characters that must stay hidden for a masked rendering to reveal anything at all."""

_MASK_ELLIPSIS: Final = "..."
_FULLY_MASKED: Final = "..."
"""Rendering for a secret too short to reveal a tail from without leaking most of it."""

_URLSAFE_B64 = re.compile(r"\A[A-Za-z0-9_-]+={0,2}\Z")

_DISPLAY_PREFIXES: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            "github_pat_",
            "sk-ant-api",
            "sk-ant-",
            "sk-proj-",
            "sk_live_",
            "sk_test_",
            "xoxb-",
            "xoxp-",
            "ghp_",
            "gho_",
            "ghu_",
            "ghs_",
            "ghr_",
            "sk-",
            "AKIA",
            "ASIA",
            "AIza",
        ),
        key=len,
        reverse=True,
    )
)
"""Public vendor scheme markers safe to show in a masked rendering.

Deliberately a closed vocabulary rather than "everything before the first
separator": only strings already published by the vendor as a fixed prefix can
appear, so the visible part of a mask can never carry secret entropy. Longest
match wins, hence the length-descending order.
"""


class SecretsCryptoError(RuntimeError):
    """Base class for every credential-encryption failure raised here."""


class KeyEncryptionKeyMissingError(SecretsCryptoError):
    """``SECRETS_KEK`` is unset (or empty), so nothing can be encrypted or decrypted."""


class KeyEncryptionKeyMalformedError(SecretsCryptoError):
    """``SECRETS_KEK`` is set but is not urlsafe-base64-encoded 32-byte key material."""


class SecretDecryptionError(SecretsCryptoError):
    """A ciphertext did not decrypt: wrong KEK, tampering, truncation, or corruption."""


def _validated_key(key: str) -> bytes:
    """Return ASCII key bytes for *key*, raising precisely when it is unusable.

    Validation happens here — at load — rather than on first encryption, so a
    misconfigured deployment fails at startup instead of at the moment an
    operator saves their first API key.

    Args:
        key: Candidate KEK as configured, urlsafe-base64 text. Surrounding
            whitespace (a common ``.env`` artifact) is stripped first.

    Returns:
        The stripped key encoded as ASCII, ready for :class:`~cryptography.fernet.Fernet`.

    Raises:
        KeyEncryptionKeyMissingError: The value is empty or whitespace only.
        KeyEncryptionKeyMalformedError: The value is not urlsafe-base64, or it
            decodes to something other than :data:`FERNET_KEY_BYTES` bytes.
    """
    candidate = key.strip()
    if not candidate:
        raise KeyEncryptionKeyMissingError(
            "SECRETS_KEK is empty. Set it in the environment to a urlsafe-base64 "
            "32-byte Fernet key; credentials are never stored unencrypted (I5)."
        )
    if not _URLSAFE_B64.match(candidate):
        raise KeyEncryptionKeyMalformedError(
            "SECRETS_KEK is not urlsafe-base64 text (expected characters A-Z a-z 0-9 "
            "- _ with optional '=' padding)."
        )
    try:
        raw = base64.urlsafe_b64decode(candidate)
    except (binascii.Error, ValueError) as exc:
        raise KeyEncryptionKeyMalformedError(
            f"SECRETS_KEK is not decodable urlsafe-base64: {exc}."
        ) from exc
    if len(raw) != FERNET_KEY_BYTES:
        raise KeyEncryptionKeyMalformedError(
            f"SECRETS_KEK decodes to {len(raw)} bytes; a Fernet key must decode to "
            f"exactly {FERNET_KEY_BYTES}. Generate one with "
            '`python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"`.'
        )
    return candidate.encode("ascii")


class SecretCipher:
    """Fernet cipher bound to a validated key-encryption key.

    Construction validates the key material, so holding an instance is proof
    that the KEK is usable. The key itself is not exposed by any attribute,
    property, or ``repr``.
    """

    __slots__ = ("_fernet",)

    def __init__(self, key: str) -> None:
        """Build a cipher from *key*, a urlsafe-base64 32-byte Fernet key.

        Raises:
            KeyEncryptionKeyMissingError: *key* is empty or whitespace only.
            KeyEncryptionKeyMalformedError: *key* is not a valid Fernet key.
        """
        validated = _validated_key(key)
        try:
            self._fernet = Fernet(validated)
        except ValueError as exc:  # pragma: no cover - _validated_key covers the cases
            raise KeyEncryptionKeyMalformedError(f"SECRETS_KEK rejected by Fernet: {exc}.") from exc

    def __repr__(self) -> str:
        """Return a representation that never contains the key material."""
        return f"{type(self).__name__}(key=<redacted>)"

    def encrypt(self, plaintext: str) -> str:
        """Return the Fernet token for *plaintext* (UTF-8 encoded before sealing).

        The token is urlsafe-base64 ASCII and is safe to persist; it carries its
        own IV and timestamp, so encrypting the same input twice yields
        different tokens.
        """
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Return the plaintext for a Fernet token produced by :meth:`encrypt`.

        For **outbound provider calls only** — never for display or logging;
        the masked rendering is :func:`mask_secret`.

        Raises:
            SecretDecryptionError: The token is not authentic under this KEK —
                it was tampered with, truncated, corrupted, or written under a
                different key. Never falls back to returning the input.
        """
        try:
            plaintext = self._fernet.decrypt(ciphertext.encode("utf-8"))
        except (InvalidToken, TypeError, ValueError) as exc:
            raise SecretDecryptionError(
                "Ciphertext failed authentication: wrong SECRETS_KEK, or the stored "
                "value was tampered with or corrupted."
            ) from exc
        return plaintext.decode("utf-8")


def load_cipher(settings: Settings | None = None) -> SecretCipher:
    """Return a :class:`SecretCipher` built from ``settings.secrets_kek``.

    The KEK is read from configuration, which sources it from the environment
    only (§7) — never from the database. When *settings* is omitted the cached
    application settings are used.

    Raises:
        KeyEncryptionKeyMissingError: ``SECRETS_KEK`` is unset or empty.
        KeyEncryptionKeyMalformedError: ``SECRETS_KEK`` is not a Fernet key.
    """
    if settings is None:
        from backend.core.config import get_settings

        settings = get_settings()
    kek = settings.secrets_kek
    if kek is None:
        raise KeyEncryptionKeyMissingError(
            "SECRETS_KEK is not configured. Credential encryption is mandatory (I5); "
            "there is no unencrypted fallback. Set SECRETS_KEK in the environment."
        )
    return SecretCipher(kek.get_secret_value())


def verify_key_encryption_key(settings: Settings | None = None) -> None:
    """Validate the configured KEK and discard the cipher.

    Intended for application startup: a deployment with a missing or malformed
    ``SECRETS_KEK`` should fail immediately and visibly, not when an operator
    first saves a provider key.

    Raises:
        KeyEncryptionKeyMissingError: ``SECRETS_KEK`` is unset or empty.
        KeyEncryptionKeyMalformedError: ``SECRETS_KEK`` is not a Fernet key.
    """
    load_cipher(settings)


def encrypt_secret(plaintext: str, *, settings: Settings | None = None) -> str:
    """Encrypt *plaintext* under the environment KEK and return the Fernet token.

    Convenience wrapper over :meth:`SecretCipher.encrypt` that rebuilds the
    cipher per call (cheap: a base64 decode and a key split). Callers doing bulk
    work should hold a :func:`load_cipher` result instead.

    Raises:
        KeyEncryptionKeyMissingError: ``SECRETS_KEK`` is unset or empty.
        KeyEncryptionKeyMalformedError: ``SECRETS_KEK`` is not a Fernet key.
    """
    return load_cipher(settings).encrypt(plaintext)


def decrypt_secret(ciphertext: str, *, settings: Settings | None = None) -> str:
    """Decrypt a Fernet token produced by :func:`encrypt_secret`.

    For outbound provider calls only — see :meth:`SecretCipher.decrypt`.

    Raises:
        KeyEncryptionKeyMissingError: ``SECRETS_KEK`` is unset or empty.
        KeyEncryptionKeyMalformedError: ``SECRETS_KEK`` is not a Fernet key.
        SecretDecryptionError: The token is not authentic under this KEK.
    """
    return load_cipher(settings).decrypt(ciphertext)


def generate_key_encryption_key() -> str:
    """Return a fresh urlsafe-base64 Fernet key for an operator to place in the environment.

    Generation only: this does not read, store, or install the key anywhere.
    """
    return Fernet.generate_key().decode("ascii")


def mask_secret(secret: str) -> str:
    """Render *secret* for display as a public prefix, an ellipsis, and its last 4 characters.

    Example: an OpenAI-shaped key renders as ``sk-...4f2a``. The prefix is shown
    only when it is one of the closed set of published vendor markers
    (:data:`_DISPLAY_PREFIXES`), so the visible characters can never carry secret
    entropy, and only when at least :data:`_MIN_HIDDEN_CHARS` characters stay
    hidden. Secrets too short for that render as ``...`` with nothing revealed.

    This is the only rendering helper in the module: there is no unmasking
    counterpart and no option to widen the window (§6.5).

    Args:
        secret: The full secret. Surrounding whitespace is ignored.

    Returns:
        A masked rendering, always strictly shorter than *secret* itself.

    Raises:
        ValueError: *secret* is empty or whitespace only — masking a value that
            is not there is a caller bug, not a display case.
    """
    candidate = secret.strip()
    if not candidate:
        raise ValueError("cannot mask an empty secret")
    if len(candidate) - MASK_TAIL_CHARS < _MIN_HIDDEN_CHARS:
        return _FULLY_MASKED
    prefix = next((p for p in _DISPLAY_PREFIXES if candidate.startswith(p)), "")
    if len(candidate) - len(prefix) - MASK_TAIL_CHARS < _MIN_HIDDEN_CHARS:
        prefix = ""
    return f"{prefix}{_MASK_ELLIPSIS}{candidate[-MASK_TAIL_CHARS:]}"
