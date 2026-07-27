"""Validate a normalised catalog, optionally with full data-quality checks.

Schema check only (the original behaviour, used by CI):

    python scripts/validate_catalog.py --catalog data/processed/jobs.jsonl

Schema check plus the data-quality validation of R17 (duplicate ids, salary
min>max, unknown currencies, invalid enums, expired deadlines, empty required
fields, scenario reference metadata and true no-match scenarios). Exits non-zero
on any error-severity finding:

    python scripts/validate_catalog.py --catalog data/processed/jobs.jsonl \
        --scenarios evaluation/data/scenarios.jsonl --report-dir artifacts
"""

from __future__ import annotations

import argparse
import sys

from jobrec.catalog import load_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    parser.add_argument("--scenarios", default=None,
                        help="scenario JSONL; enables the R17 data-quality checks")
    parser.add_argument("--config", default=None,
                        help="config supplying the reference date and constraint policies")
    parser.add_argument("--report-dir", default=None,
                        help="write data_quality_report.json into this directory")
    args = parser.parse_args()
    try:
        jobs = load_catalog(args.catalog)
    except Exception as exc:  # noqa: BLE001
        print(f"INVALID catalog: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(jobs)} valid job postings in {args.catalog}")

    if args.scenarios is None:
        return 0

    # Delegate the semantic checks to the single validator implementation.
    from jobrec_eval.cli import run_validate

    return run_validate(
        args.scenarios, args.catalog,
        config_path=args.config, report_dir=args.report_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
