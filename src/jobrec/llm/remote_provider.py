"""Remote LLM provider (OpenAI-compatible chat completions via httpx).

Only used in ``hybrid`` mode when explicitly configured. This provider is
intentionally not exercised in deterministic CI.

Secret handling (R26.1): the API key is read **only** from the environment
(:data:`API_KEY_ENV`). There is no constructor argument, config field or file
that can carry it — ``AppConfig`` forbids extra keys, so a key cannot even be
smuggled in through YAML. The key is stored in a private attribute, never placed
in :meth:`RemoteLLMProvider.manifest` (which is persisted into run artifacts),
never rendered by ``repr``, and the module logger carries
:class:`~jobrec.utils.redaction.SecretLogFilter` so nothing this module logs —
including a transport error that quoted a request — can contain it.

Call accounting (R11.1): every returned :class:`~jobrec.llm.provider.LLMCallRecord`
carries a ``metadata`` dict with the token usage reported by the server, the
request parameters *as actually sent* and the retry/fallback trace, which the run
bundle exporter surfaces as ``request_params``/``response_metadata``. That dict
deliberately holds no prompt, no response text and no credential material.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from ..utils.hashing import content_id
from ..utils.redaction import install_secret_log_filter, redact, secret_values
from .provider import LLMCallRecord, LLMError, LLMInvalidJSON, LLMTimeout

#: Environment variables this provider reads. Keys live ONLY in the environment.
API_KEY_ENV = "JOBREC_LLM_API_KEY"
BASE_URL_ENV = "JOBREC_LLM_BASE_URL"
MODEL_ENV = "JOBREC_LLM_MODEL"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)
#: R26.1 — every record from this module is scrubbed before any handler sees it.
install_secret_log_filter(logger)


def _scrub(text: str) -> str:
    """Redact credential material from a message before it is raised or logged."""
    return redact(str(text), secrets=secret_values())


#: Token-count aliases: OpenAI-compatible servers report either the classic
#: ``prompt_tokens``/``completion_tokens`` spelling or the newer
#: ``input_tokens``/``output_tokens`` one. Both are normalised to the classic name
#: so the exported artifacts have a single, comparable shape.
_TOKEN_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prompt_tokens", ("prompt_tokens", "input_tokens")),
    ("completion_tokens", ("completion_tokens", "output_tokens")),
    ("total_tokens", ("total_tokens",)),
)


def _token_usage(data: Any) -> dict[str, Any]:
    """Normalise a response's ``usage`` block into flat, comparable token counts.

    Returns ``{}`` when the response carries no usable ``usage`` object (some
    proxies omit it entirely), so a missing usage block is recorded as *absent*
    rather than as a fabricated zero. ``total_tokens`` is derived by addition only
    when the server did not report it but reported both halves.
    """
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {"usage": dict(usage)}
    for canonical, names in _TOKEN_ALIASES:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[canonical] = value
                break
    if "total_tokens" not in out and "prompt_tokens" in out and "completion_tokens" in out:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


#: Server-reported identity of a single completion, copied verbatim when present.
#:
#: ``model`` is the model the SERVER says answered, which is not always the one that was
#: asked for: an alias like ``gpt-4o-mini`` resolves to a dated build, and a gateway may
#: route to a different deployment entirely. ``system_fingerprint`` is the backend
#: configuration OpenAI-compatible servers expose so two identical requests can be told
#: apart when the serving stack changed underneath. ``id`` identifies the individual
#: completion and is what a provider-side log can be matched against.
#:
#: Without these the artifacts recorded only what was REQUESTED, so "same model" was an
#: assumption rather than a record.
_PROVENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("response_id", "id"),
    ("system_fingerprint", "system_fingerprint"),
    ("response_model", "model"),
)


def _response_provenance(data: Any) -> dict[str, Any]:
    """Server-reported completion identity, omitting whatever the server did not send.

    Absent fields are LEFT OUT rather than recorded as null or filled with the requested
    value: many OpenAI-compatible proxies omit ``system_fingerprint`` entirely, and a
    fabricated value would be indistinguishable from a real one when reading the bundle.
    """
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for name, key in _PROVENANCE_FIELDS:
        value = data.get(key)
        if isinstance(value, str) and value:
            out[name] = value
    return out


def _status_reason(exc: httpx.HTTPStatusError) -> str:
    """A short, non-sensitive label for the status that triggered the retry."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status is not None else "http_error"


def _finish_reason(data: Any) -> str | None:
    """The first choice's ``finish_reason``, or ``None`` when the server omits it."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None
    return None


def _content_of(data: Any) -> str:
    """The first choice's message content, as an :class:`LLMError` on any other shape.

    ``data["choices"][0]["message"]["content"]`` used to be indexed directly and outside
    every exception handler. Gateways routinely return HTTP 200 with an error envelope, a
    content-filter verdict, or an empty ``choices`` list; each of those raises
    ``KeyError`` / ``IndexError`` / ``TypeError``, none of which is an ``LLMError``, so it
    would bypass the bounded retry AND the rule-extractor fallback and abort a
    multi-hundred-call experiment outright. Raising ``LLMError`` puts these on the same
    degrade-this-one-run path as a timeout.
    """
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        keys = ", ".join(sorted(data)[:6]) if isinstance(data, dict) else type(data).__name__
        raise LLMError(f"remote model response carried no choices (keys: {keys})")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError(
            f"remote model response carried no message content "
            f"({type(content).__name__})")
    return content


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from a model response, tolerating code fences/prose."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


class RemoteLLMProvider:
    """Thin OpenAI-compatible chat client."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = API_KEY_ENV,
        timeout_seconds: int = 30,
        extraction_temperature: float = 0.0,
        response_temperature: float = 0.2,
    ) -> None:
        self.name = "remote"
        self.api_key_env = api_key_env
        self.model = model or os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL)
        # Env-only (R26.1): the caller passes the variable NAME, never a value.
        self._api_key = os.environ.get(api_key_env) or None
        self.timeout = timeout_seconds
        self.extraction_temperature = extraction_temperature
        self.response_temperature = response_temperature
        if self._api_key is None:
            logger.warning(
                "remote provider constructed without a key: export %s", api_key_env
            )

    @property
    def has_api_key(self) -> bool:
        """Whether a key was found in the environment. The value stays private."""
        return self._api_key is not None

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"RemoteLLMProvider(model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key_env={self.api_key_env!r}, api_key_present={self.has_api_key})"
        )

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            try:
                payload = resp.json()
            except ValueError as exc:
                # A 200 whose body is not JSON -- an HTML error page from a proxy, a
                # truncated response. ``json.JSONDecodeError`` is a ``ValueError``, NOT an
                # httpx error, so without this it escapes every handler below, escapes the
                # bounded retry (which only catches ``LLMError``) and escapes the
                # rule-extractor fallback: one malformed reply would abort the whole
                # experiment mid-flight and force a complete re-run.
                raise LLMInvalidJSON(
                    f"remote model returned a non-JSON body (HTTP {resp.status_code})"
                ) from exc
        if not isinstance(payload, dict):
            raise LLMInvalidJSON(
                f"remote model returned {type(payload).__name__}, expected a JSON object")
        return payload

    def _request_params(self, body: dict[str, Any]) -> dict[str, Any]:
        """Request parameters exactly as they were sent on the successful attempt.

        Reads them back off ``body`` rather than off ``self``, so a parameter the
        code deliberately omitted (temperature for the ``gpt-5`` family) or dropped
        on the retry is recorded as absent instead of as a value that was never
        transmitted.
        """
        params: dict[str, Any] = {
            "timeout_seconds": self.timeout,
            "json_mode": "response_format" in body,
        }
        if "temperature" in body:
            params["temperature"] = body["temperature"]
        if "response_format" in body:
            params["response_format"] = body["response_format"]
        return params

    def _chat(
        self, prompt: str, temperature: float, json_mode: bool
    ) -> tuple[str, float, dict[str, Any]]:
        """Post one chat completion and return ``(content, latency_ms, metadata)``.

        ``metadata`` carries the request parameters as actually sent, the token
        usage / finish reason from the response and the retry-fallback trace. It
        never carries the prompt, the response text or credential material.
        """
        if not self._api_key:
            raise LLMTimeout(
                f"no API key configured for remote provider: set {self.api_key_env}"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # GPT-5 family typically only accepts the default temperature; omit it.
        if not str(self.model).lower().startswith("gpt-5"):
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        attempts = 1
        retried = False
        retry_reason: str | None = None
        try:
            data = self._post(body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(_scrub(str(exc))) from exc
        except httpx.HTTPStatusError as exc:
            # Fallback: retry once without response_format / temperature, which
            # some proxies or newer models reject.
            body.pop("response_format", None)
            body.pop("temperature", None)
            retried = True
            attempts = 2
            retry_reason = _status_reason(exc)
            try:
                data = self._post(body)
            except httpx.HTTPError as exc2:
                raise LLMError(f"remote model error: {_scrub(str(exc2))}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"remote model error: {_scrub(str(exc))}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        metadata: dict[str, Any] = {
            **self._request_params(body),
            **_token_usage(data),
            "attempts": attempts,
            "retried_without_response_format": retried,
        }
        reason = _finish_reason(data)
        if reason is not None:
            metadata["finish_reason"] = reason
        if retry_reason is not None:
            metadata["retry_reason"] = retry_reason
        metadata.update(_response_provenance(data))
        return _content_of(data), latency_ms, metadata

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        raw, latency, metadata = self._chat(
            prompt, self.extraction_temperature, json_mode=True)
        payload = _extract_json(raw)
        if payload is None:
            raise LLMInvalidJSON("could not parse JSON object from model response")
        record = LLMCallRecord(
            call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
            raw_response=raw, parsed_ok=True, latency_ms=latency,
            provider=self.name, model=self.model, metadata=metadata,
        )
        return payload, record

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = "") -> tuple[str, LLMCallRecord]:
        try:
            raw, latency, metadata = self._chat(
                prompt, self.response_temperature, json_mode=False)
        except Exception as exc:  # noqa: BLE001 - phrasing must never break the pipeline
            # Record WHY we fell back (the exception class only, never its message,
            # which can quote the request) so the fallback is visible in artifacts.
            return fallback, LLMCallRecord(
                call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
                raw_response=fallback, parsed_ok=False, latency_ms=0.0,
                provider=self.name, model=self.model,
                metadata={"fell_back": True, "error": type(exc).__name__},
            )
        record = LLMCallRecord(
            call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
            raw_response=raw, parsed_ok=True, latency_ms=latency,
            provider=self.name, model=self.model, metadata=metadata,
        )
        return raw, record

    def manifest(self) -> dict[str, Any]:
        """Reproducibility metadata. Records the key's SOURCE, never its value."""
        return {"provider": self.name, "model": self.model, "mode": "hybrid",
                "base_url": self.base_url, "api_key_env": self.api_key_env,
                "api_key_present": self.has_api_key}
