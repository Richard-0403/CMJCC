"""Build a retrieval index manifest for a catalog.

Usage: python scripts/build_index.py --catalog data/processed/jobs.jsonl --out-dir artifacts/indexes
"""

from __future__ import annotations

import argparse

from jobrec.retrieval.index_builder import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    parser.add_argument("--out-dir", default="artifacts/indexes")
    args = parser.parse_args()
    manifest = build_index(args.catalog, args.out_dir)
    print(f"Index: {manifest['record_count']} records, catalog_hash={manifest['catalog_hash'][:12]}")


if __name__ == "__main__":
    main()
