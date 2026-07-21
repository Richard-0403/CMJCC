"""Deterministic hashing and id helpers.

Stable content hashes guarantee reproducibility: identical inputs produce
identical ids and identical config/catalog/prompt hashes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def sha256_of_text(text: str) -> str:
    """Return the hex sha256 digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    """Return the hex sha256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Return a stable sha256 over a JSON-serialisable object.

    Keys are sorted so that dict ordering does not affect the digest.
    """
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return sha256_of_text(payload)


def content_id(prefix: str, *parts: Any) -> str:
    """Deterministic content-addressed id built from ``parts``.

    Used where reproducibility matters (evidence, states) so that replaying the
    same inputs yields the same identifiers.
    """
    digest = stable_hash([str(p) for p in parts])[:16]
    return f"{prefix}-{digest}"


def random_id(prefix: str) -> str:
    """Return a random UUID-backed id with a human-readable prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
