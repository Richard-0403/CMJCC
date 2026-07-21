"""Replay provider: returns previously recorded model responses.

Used in ``replay`` mode so experiments can be reproduced without calling a
remote model. Records are keyed by (purpose, prompt) content id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.hashing import content_id
from .provider import LLMCallRecord, LLMError


class ReplayProvider:
    """Serves recorded responses from a model_calls.jsonl file."""

    def __init__(self, records_path: str | Path) -> None:
        self.name = "replay"
        self.model = "replay"
        self._by_key: dict[str, dict] = {}
        path = Path(records_path)
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = content_id("call", rec.get("purpose", ""), rec.get("prompt", ""))
                    self._by_key[key] = rec

    def _lookup(self, prompt: str, purpose: str) -> dict:
        key = content_id("call", purpose, prompt)
        if key not in self._by_key:
            raise LLMError(f"no recorded response for purpose={purpose}")
        return self._by_key[key]

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        rec = self._lookup(prompt, purpose)
        payload = rec.get("parsed") or json.loads(rec["raw_response"])
        record = LLMCallRecord(
            call_id=rec.get("call_id", "replay"), purpose=purpose, prompt=prompt,
            raw_response=rec.get("raw_response", ""), parsed_ok=True,
            latency_ms=0.0, provider=self.name, model=self.model,
        )
        return payload, record

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = "") -> tuple[str, LLMCallRecord]:
        try:
            rec = self._lookup(prompt, purpose)
            text = rec.get("raw_response", fallback)
        except LLMError:
            text = fallback
        record = LLMCallRecord(
            call_id="replay", purpose=purpose, prompt=prompt, raw_response=text,
            parsed_ok=True, latency_ms=0.0, provider=self.name, model=self.model,
        )
        return text, record

    def manifest(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "mode": "replay",
                "records": len(self._by_key)}
