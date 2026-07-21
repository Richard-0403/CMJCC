"""Normalise a raw job CSV into jobs.jsonl + jobs.csv + catalog_manifest.json.

Usage:
    python scripts/prepare_catalog.py --input data/raw/jobs.csv \
        --out-dir data/processed --snapshot-id catalog-2026-01-v1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jobrec.catalog import build_manifest, normalize_job, write_catalog, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare (normalise) the job catalog")
    parser.add_argument("--input", default="data/raw/jobs.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--snapshot-id", default="catalog-2026-01-v1")
    parser.add_argument("--reference-date", default="2026-01-01")
    args = parser.parse_args()

    in_path = Path(args.input)
    with in_path.open() as fh:
        rows = list(csv.DictReader(fh))

    jobs = [normalize_job(row, args.snapshot_id) for row in rows]

    out_dir = Path(args.out_dir)
    write_catalog(jobs, out_dir / "jobs.jsonl")

    # Also emit a flat normalised CSV for human inspection.
    csv_path = out_dir / "jobs.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "job_id", "title", "company", "role_family", "experience_level",
            "city", "country", "work_mode", "salary_min", "salary_currency",
            "salary_period", "salary_min_monthly_myr", "is_active", "application_deadline",
        ])
        for j in jobs:
            writer.writerow([
                j.job_id, j.title, j.company, j.role_family, j.experience_level,
                j.city, j.country, j.work_mode, j.salary_min, j.salary_currency,
                j.salary_period, j.salary_min_monthly_myr, j.is_active, j.application_deadline,
            ])

    manifest = build_manifest(jobs, args.snapshot_id, [in_path.name], args.reference_date)
    write_json(manifest, out_dir / "catalog_manifest.json")

    print(f"Normalised {len(jobs)} jobs -> {out_dir}/jobs.jsonl")
    print(f"catalog_hash={manifest['catalog_hash'][:12]} missingness={manifest['field_missingness']}")


if __name__ == "__main__":
    main()
