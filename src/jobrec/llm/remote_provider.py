"""Remote LLM provider (OpenAI-compatible chat completions via httpx).

Only used in ``hybrid`` mode when explicitly configured. API keys are read from
the environment, never hard-coded or logged. This provider is intentionally not
exercised in deterministic CI.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ..utils.hashing import content_id
from .provider import LLMCallRecord, LLMInvalidJSON, LLMTimeout


class RemoteLLMProvider:
    """Thin OpenAI-compatible chat client."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "JOBREC_LLM_API_KEY",
        timeout_seconds: int = 30,
        extraction_temperature: float = 0.0,
        response_temperature: float = 0.2,
    ) -> None:
        self.name = "remote"
        self.model = model or os.environ.get("JOBREC_LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("JOBREC_LLM_BASE_URL", "https://api.openai.com/v1")
        self._api_key = os.environ.get(api_key_env)
        self.timeout = timeout_seconds
        self.extraction_temperature = extraction_temperature
        self.response_temperature = response_temperature

    def _chat(self, prompt: str, temperature: float, json_mode: bool) -> tuple[str, float]:
        if not self._api_key:
            raise LLMTimeout("no API key configured for remote provider")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMTimeout(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return data["choices"][0]["message"]["content"], latency_ms

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        raw, latency = self._chat(prompt, self.extraction_temperature, json_mode=True)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMInvalidJSON(str(exc)) from exc
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
        return {"provider": self.name, "model": self.model, "mode": "hybrid",
                "base_url": self.base_url}
