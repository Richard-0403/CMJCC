"""Build and persist a retrieval index snapshot for the catalog.

For TF-IDF the "index" is the fitted vectorizer + matrix held in memory by the
retriever; this module records an index manifest so runs can reference which
catalog snapshot an index was built from (reproducibility).
"""

from __future__ import annotations

from pathlib import Path

from ..catalog import catalog_hash, load_catalog, write_json
from ..utils.time import to_iso, utcnow


def build_index(catalog_path: str | Path, out_dir: str | Path) -> dict:
    """Load the catalog and write an index manifest (no heavy artifacts needed)."""
    jobs = load_catalog(catalog_path)
    manifest = {
        "index_type": "tfidf+structured",
        "record_count": len(jobs),
        "catalog_hash": catalog_hash(jobs),
        "built_at": to_iso(utcnow()),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest, out_dir / "index_manifest.json")
    return manifest
