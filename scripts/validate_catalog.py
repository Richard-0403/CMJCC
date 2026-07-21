"""Validate that every record in a normalised catalog parses against the schema.

Usage: python scripts/validate_catalog.py --catalog data/processed/jobs.jsonl
"""

from __future__ import annotations

import argparse
import sys

from jobrec.catalog import load_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    args = parser.parse_args()
    try:
        jobs = load_catalog(args.catalog)
    except Exception as exc:  # noqa: BLE001
        print(f"INVALID catalog: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(jobs)} valid job postings in {args.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
