"""Redaction helpers for run-detail output (R12.3) and log records (R26.1).

``Repository.get_run`` can return raw model outputs and state payloads to an
evaluator. Everything that leaves the store on those paths passes through
:func:`redact` first, which always strips credential-shaped and PII-shaped
substrings and, when ``config.logging.redact_candidate_text`` is enabled,
replaces free-text content with a placeholder so candidate-provided text never
leaves the database.

The same redactor guards the log stream (R26.1): :class:`SecretLogFilter` is a
:mod:`logging` filter that rewrites every record through :func:`redact` before a
handler can format it, additionally masking the *literal* values of
secret-shaped environment variables (:func:`secret_values`) so an API key is
scrubbed even when it carries no recognisable prefix.

The helpers are pure and idempotent: redacting already-redacted text is a no-op,
so write-time and read-time redaction can both be applied safely.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
REDACTED_KEY = "[REDACTED_KEY]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"

#: Payload keys whose values are free text (prompts, model responses, candidate
#: utterances). Only these are dropped wholesale when candidate-text redaction is
#: enabled; structural fields (ids, purposes, latencies) stay inspectable.
TEXT_KEYS: tuple[str, ...] = (
    "prompt",
    "raw_response",
    "raw_text",
    "response_text",
    "text",
    "utterance",
)

_BEARER = re.compile(r"(?i)\bbearer\s+[\w.\-]{8,}")
_KEY_TOKEN = re.compile(r"(?i)\b(?:sk|rk|pk|api)[-_][A-Za-z0-9]{8,}\b")
_LABELLED_SECRET = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password)"
    r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',}]+)"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w")
# Either an international number (leading ``+``) or a national one (leading
# trunk ``0``); a digit-count guard keeps years/ids/timestamps untouched.
# The trailing guard rejects a following word character outright and rejects a
# following ``.`` only when a digit comes after it, so a sentence-final number
# ("call +60 12-345 6789.") is still redacted while decimals, dates and version
# strings ("4000.50", "2026-01-01", "1.2.3") are left intact.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d[\d\s().\-]{7,}\d|0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})(?!\w)(?!\.\d)"
)


#: Environment variable *names* whose values are treated as secrets (R26.1).
#: Matched on the suffix so provider-specific names (``JOBREC_LLM_API_KEY``,
#: ``OPENAI_API_KEY``, ``PGPASSWORD``) are covered without an allow-list.
_SECRET_ENV_NAME = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)$"
)

#: Literal env values shorter than this are NOT masked. Real API keys are long;
#: the floor keeps a short, word-like value (a local dev password) from blanking
#: unrelated substrings in every log line.
MIN_SECRET_LENGTH = 12


def _phone_replacement(match: re.Match[str]) -> str:
    digits = sum(1 for ch in match.group(0) if ch.isdigit())
    return REDACTED_PHONE if digits >= 9 else match.group(0)


def secret_env_names(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the secret-shaped variable names present in ``env`` (default: os.environ)."""
    source = os.environ if env is None else env
    return tuple(sorted(name for name in source if _SECRET_ENV_NAME.search(str(name))))


def secret_values(
    env: Mapping[str, str] | None = None, names: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Return the literal secret values to mask, longest first.

    Values are read from ``env`` (default: the process environment) for either the
    explicitly requested ``names`` or every secret-shaped name
    (:func:`secret_env_names`). Values below :data:`MIN_SECRET_LENGTH` are
    dropped. Longest-first ordering makes masking stable when one value is a
    substring of another.
    """
    source = os.environ if env is None else env
    wanted = list(names) if names is not None else list(secret_env_names(source))
    found = {
        value
        for name in wanted
        if isinstance(value := source.get(name), str) and len(value) >= MIN_SECRET_LENGTH
    }
    return tuple(sorted(found, key=lambda value: (-len(value), value)))


def _mask_secret_values(text: str, secrets: Sequence[str]) -> str:
    out = text
    for value in secrets:
        if value and len(value) >= MIN_SECRET_LENGTH:
            out = out.replace(value, REDACTED_KEY)
    return out


def redact(
    text: str, *, redact_candidate_text: bool = False, secrets: Sequence[str] = ()
) -> str:
    """Return ``text`` with sensitive content removed.

    Always removes credential-shaped substrings (bearer tokens, ``api_key=...``
    assignments, ``sk-``/``pk-`` style keys) and PII patterns (e-mail addresses,
    phone numbers). When ``redact_candidate_text`` is True (mirroring
    ``config.logging.redact_candidate_text``) the whole string is replaced with
    :data:`REDACTED`, because free text may quote the candidate verbatim.

    ``secrets`` are literal values (typically from :func:`secret_values`) masked
    before the pattern pass, so a key that matches no known credential shape is
    still removed.
    """
    if not isinstance(text, str) or not text:
        return text
    if redact_candidate_text:
        return REDACTED
    out = _mask_secret_values(text, secrets) if secrets else text
    out = _BEARER.sub(f"Bearer {REDACTED_KEY}", out)
    out = _LABELLED_SECRET.sub(rf"\1{REDACTED_KEY}", out)
    out = _KEY_TOKEN.sub(REDACTED_KEY, out)
    out = _EMAIL.sub(REDACTED_EMAIL, out)
    return _PHONE.sub(_phone_replacement, out)


def redact_payload(
    value: Any,
    *,
    redact_candidate_text: bool = False,
    in_text_field: bool = False,
    secrets: Sequence[str] = (),
) -> Any:
    """Recursively :func:`redact` every string inside a JSON-shaped payload.

    Credential/PII stripping applies to every string. Candidate-text redaction
    applies only to values reached through a :data:`TEXT_KEYS` key, so a redacted
    payload keeps its structure (ids, counts, latencies) while the free text is
    gone. ``secrets`` is forwarded to :func:`redact` unchanged.
    """
    if isinstance(value, str):
        return redact(
            value,
            redact_candidate_text=redact_candidate_text and in_text_field,
            secrets=secrets,
        )
    if isinstance(value, dict):
        return {
            key: redact_payload(
                item,
                redact_candidate_text=redact_candidate_text,
                in_text_field=in_text_field or (isinstance(key, str) and key in TEXT_KEYS),
                secrets=secrets,
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [
            redact_payload(
                item,
                redact_candidate_text=redact_candidate_text,
                in_text_field=in_text_field,
                secrets=secrets,
            )
            for item in value
        ]
    return value


#: Marker attribute used to recognise (and not duplicate) an installed filter.
_FILTER_MARKER = "_jobrec_secret_filter"


class SecretLogFilter(logging.Filter):
    """A :mod:`logging` filter guaranteeing no API key reaches a log record (R26.1).

    Installed on the JSON handler by
    :func:`jobrec.utils.observability.configure_logging` and directly on the
    remote-provider logger, so a credential cannot be logged regardless of which
    handler is attached. The filter never drops a record; it rewrites it:

    * ``record.msg``/``record.args`` are collapsed into one already-redacted
      message when redaction changed anything;
    * a structured payload (``record.structured``, set by
      :class:`~jobrec.utils.observability.RunTrace`) is replaced by a redacted
      *copy*, so the in-memory run trace is not mutated behind its owner's back;
    * literal values of secret-shaped environment variables are masked in
      addition to the credential shapes :func:`redact` already knows.

    The environment is read per record, so a key exported after logging was
    configured is still masked.
    """

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        names: Iterable[str] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name)
        self._env = env
        self._names = tuple(names) if names is not None else None
        setattr(self, _FILTER_MARKER, True)

    def secrets(self) -> tuple[str, ...]:
        return secret_values(self._env, self._names)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib API
        secrets = self.secrets()
        structured = getattr(record, "structured", None)
        if isinstance(structured, Mapping):
            record.structured = redact_payload(dict(structured), secrets=secrets)
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not lose the record
            message = str(record.msg)
        cleaned = redact(message, secrets=secrets)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def install_secret_log_filter(
    logger: logging.Logger | logging.Handler,
    *,
    env: Mapping[str, str] | None = None,
    names: Iterable[str] | None = None,
) -> SecretLogFilter:
    """Attach a :class:`SecretLogFilter` to ``logger`` (or a handler), idempotently.

    Returns the installed filter, reusing an existing one so repeated calls (module
    import, re-configuration) never stack duplicates.
    """
    existing = next(
        (f for f in logger.filters if getattr(f, _FILTER_MARKER, False)), None
    )
    if isinstance(existing, SecretLogFilter):
        return existing
    log_filter = SecretLogFilter(env=env, names=names)
    logger.addFilter(log_filter)
    return log_filter
