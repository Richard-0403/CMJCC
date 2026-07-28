"""Replay provider: returns previously recorded model responses.

Used in ``replay`` mode so experiments can be reproduced without calling a
remote model. Lookup is by ``content_id("call", purpose, prompt)``.

Records are indexed under two keys so both artifact shapes resolve:

* the recomputed ``content_id("call", purpose, prompt)``, for files that persist
  the prompt (older or externally produced records); and
* the record's own ``call_id``, because
  :class:`~jobrec.llm.remote_provider.RemoteLLMProvider` builds that id as exactly
  ``content_id("call", purpose, prompt)``. A run bundle therefore replays from
  ``call_id`` + ``raw_response`` alone, without the prompt ever being persisted.

An empty index is legitimate: deterministic bundles contain no model calls at all
(the mock provider issues none), so their ``model_calls.jsonl`` is empty and every
lookup misses. :meth:`ReplayProvider.complete_text` degrades to its fallback in
that case; :meth:`ReplayProvider.complete_json` raises so a missing recording is
never silently substituted for a real one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.hashing import content_id
from .provider import LLMCallRecord, LLMError


def _record_keys(rec: dict) -> list[str]:
    """Every lookup key one recorded call answers to (see the module docstring)."""
    keys: list[str] = []
    purpose = str(rec.get("purpose", "") or "")
    if rec.get("prompt") is not None:
        keys.append(content_id("call", purpose, str(rec["prompt"])))
    call_id = rec.get("call_id")
    if isinstance(call_id, str) and call_id:
        keys.append(call_id)
    if not keys:
        # Neither a prompt nor an id: keep the historical empty-prompt key so an
        # otherwise unusable record is at least indexed the way it always was.
        keys.append(content_id("call", purpose, ""))
    return keys


class ReplayProvider:
    """Serves recorded responses from a model_calls.jsonl file."""

    def __init__(self, records_path: str | Path) -> None:
        self.name = "replay"
        self.model = "replay"
        self._by_key: dict[str, dict] = {}
        self._record_count = 0
        path = Path(records_path)
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._record_count += 1
                    for key in _record_keys(rec):
                        self._by_key[key] = rec

    def _lookup(self, prompt: str, purpose: str) -> dict:
        key = content_id("call", purpose, prompt)
        if key not in self._by_key:
            raise LLMError(f"no recorded response for purpose={purpose}")
        return self._by_key[key]

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        rec = self._lookup(prompt, purpose)
        payload = rec.get("parsed")
        if not payload:
            raw = rec.get("raw_response")
            if not raw:
                # A row written with ``save_raw_responses`` off carries no body;
                # that is a missing recording, not a malformed one.
                raise LLMError(f"recorded call for purpose={purpose} has no response body")
            payload = json.loads(raw)
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
                # ``records`` counts loaded calls; ``keys`` counts index entries,
                # which is larger when a record is reachable by both prompt and id.
                "records": self._record_count, "keys": len(self._by_key)}
