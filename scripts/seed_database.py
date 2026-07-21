"""Seed PostgreSQL with the catalog snapshot and job postings.

Requires DATABASE_URL to point at a reachable PostgreSQL instance.
Usage: python scripts/seed_database.py --catalog data/processed/jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobrec.catalog import catalog_hash, load_catalog
from jobrec.storage.db import create_all, make_engine, make_session_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    args = parser.parse_args()

    jobs = load_catalog(args.catalog)
    manifest_path = Path(args.catalog).parent / "catalog_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    snapshot_id = manifest.get("catalog_snapshot_id", "catalog-unknown")

    engine = make_engine()
    create_all(engine)
    from jobrec.storage.models import CatalogSnapshotRow, JobPostingRow

    with make_session_factory(engine)() as s:
        s.merge(CatalogSnapshotRow(catalog_snapshot_id=snapshot_id,
                catalog_hash=catalog_hash(jobs), record_count=len(jobs), payload=manifest))
        for job in jobs:
            s.merge(JobPostingRow(job_id=job.job_id, catalog_snapshot_id=snapshot_id,
                    title=job.title, role_family=job.role_family, payload=job.model_dump(mode="json")))
        s.commit()
    print(f"Seeded {len(jobs)} jobs (snapshot {snapshot_id}) into the database.")


if __name__ == "__main__":
    main()
