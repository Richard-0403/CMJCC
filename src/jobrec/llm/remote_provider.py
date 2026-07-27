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
            return resp.json()

    def _chat(self, prompt: str, temperature: float, json_mode: bool) -> tuple[str, float]:
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
        try:
            data = self._post(body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout(_scrub(exc)) from exc
        except httpx.HTTPStatusError as exc:
            # Fallback: retry once without response_format / temperature, which
            # some proxies or newer models reject.
            body.pop("response_format", None)
            body.pop("temperature", None)
            try:
                data = self._post(body)
            except httpx.HTTPError as exc2:
                raise LLMError(f"remote model error: {_scrub(exc2)}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"remote model error: {_scrub(exc)}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return data["choices"][0]["message"]["content"], latency_ms

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        raw, latency = self._chat(prompt, self.extraction_temperature, json_mode=True)
        payload = _extract_json(raw)
        if payload is None:
            raise LLMInvalidJSON("could not parse JSON object from model response")
        record = LLMCallRecord(
            call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
            raw_response=raw, parsed_ok=True, latency_ms=latency,
            provider=self.name, model=self.model,
        )
        return payload, record

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = "") -> tuple[str, LLMCallRecord]:
        try:
            raw, latency = self._chat(prompt, self.response_temperature, json_mode=False)
        except Exception:  # noqa: BLE001 - phrasing must never break the pipeline
            return fallback, LLMCallRecord(
                call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
                raw_response=fallback, parsed_ok=False, latency_ms=0.0,
                provider=self.name, model=self.model, metadata={"fell_back": True},
            )
        record = LLMCallRecord(
            call_id=content_id("call", purpose, prompt), purpose=purpose, prompt=prompt,
            raw_response=raw, parsed_ok=True, latency_ms=latency,
            provider=self.name, model=self.model,
        )
        return raw, record

    def manifest(self) -> dict[str, Any]:
        """Reproducibility metadata. Records the key's SOURCE, never its value."""
        return {"provider": self.name, "model": self.model, "mode": "hybrid",
                "base_url": self.base_url, "api_key_env": self.api_key_env,
                "api_key_present": self.has_api_key}
