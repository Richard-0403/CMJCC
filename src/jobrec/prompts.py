"""Prompt template loading and prompt hashing for reproducibility."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .utils.hashing import stable_hash

# Resolve the prompts directory relative to the repository root.
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _read(name: str) -> str:
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text()
    return ""


@lru_cache(maxsize=1)
def prompt_templates() -> dict[str, str]:
    return {
        "intent_extraction": _read("intent_extraction.md"),
        "clarification": _read("clarification.md"),
        "grounded_explanation": _read("grounded_explanation.md"),
    }


@lru_cache(maxsize=1)
def prompt_hash() -> str:
    """Stable hash over all prompt templates."""
    return stable_hash(prompt_templates())


def render_intent_extraction(utterance: str) -> str:
    tmpl = prompt_templates()["intent_extraction"] or "Utterance:\n{utterance}"
    return tmpl.replace("{utterance}", utterance)
