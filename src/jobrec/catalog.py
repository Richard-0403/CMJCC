"""Job catalog normalisation, loading and manifest generation."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .domain.job import JobPosting
from .taxonomy import canonical_role, canonical_skill
from .utils.hashing import sha256_of_text, stable_hash
from .utils.money import to_monthly_myr
from .utils.text import normalize_token
from .utils.time import utcnow

NORMALIZATION_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


def _split_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split("|")
    return [i.strip() for i in items if str(i).strip()]


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def normalize_job(raw: dict[str, Any], catalog_snapshot_id: str) -> JobPosting:
    """Normalise a raw job record into a typed :class:`JobPosting`.

    Missing values are represented as ``None`` (never empty-string as if known).
    """
    required = [canonical_skill(s) for s in _split_list(raw.get("required_skills"))]
    preferred = [canonical_skill(s) for s in _split_list(raw.get("preferred_skills"))]

    currency = (raw.get("salary_currency") or None)
    period = raw.get("salary_period") or "unknown"
    smin = _to_float(raw.get("salary_min"))
    smax = _to_float(raw.get("salary_max"))

    smin_myr = to_monthly_myr(smin, currency, period) if (smin is not None and currency) else None
    smax_myr = to_monthly_myr(smax, currency, period) if (smax is not None and currency) else None

    title = str(raw.get("title", "")).strip()
    payload_hash = stable_hash(raw)

    return JobPosting(
        job_id=str(raw["job_id"]).strip(),
        title=title,
        company=str(raw.get("company", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        normalized_title=normalize_token(title),
        role_family=canonical_role(raw.get("role_family") or title) if title else None,
        industry=(normalize_token(raw["industry"]) if raw.get("industry") else None),
        employment_type=(normalize_token(raw["employment_type"]) if raw.get("employment_type") else None),
        required_skills=required,
        preferred_skills=preferred,
        responsibilities=_split_list(raw.get("responsibilities")),
        salary_min=smin,
        salary_max=smax,
        salary_currency=currency.upper() if currency else None,
        salary_period=period if period in {"hour", "month", "year", "unknown"} else "unknown",
        salary_min_monthly_myr=smin_myr,
        salary_max_monthly_myr=smax_myr,
        country=(str(raw["country"]).strip() if raw.get("country") else None),
        city=(str(raw["city"]).strip() if raw.get("city") else None),
        region=(str(raw["region"]).strip() if raw.get("region") else None),
        work_mode=(raw.get("work_mode") or "unspecified"),
        min_years_experience=_to_float(raw.get("min_years_experience")),
        max_years_experience=_to_float(raw.get("max_years_experience")),
        experience_level=(normalize_token(raw["experience_level"]) if raw.get("experience_level") else None),
        required_work_authorization=_split_list(raw.get("required_work_authorization")),
        application_deadline=_parse_date(raw.get("application_deadline")),
        is_active=_to_bool(raw.get("is_active", True)),
        source_uri=(raw.get("source_uri") or None),
        source_snapshot_id=catalog_snapshot_id,
        ingested_at=utcnow(),
        raw_payload_hash=payload_hash,
    )


def load_catalog(path: str | Path) -> list[JobPosting]:
    """Load a normalised catalog from a JSONL file."""
    path = Path(path)
    jobs: list[JobPosting] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            jobs.append(JobPosting.model_validate_json(line))
    return jobs


def write_catalog(jobs: list[JobPosting], path: str | Path) -> None:
    """Write jobs to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for job in jobs:
            fh.write(job.model_dump_json())
            fh.write("\n")


def catalog_hash(jobs: list[JobPosting]) -> str:
    """Stable hash over the catalog content (ignores volatile ingest times)."""
    payload = [j.raw_payload_hash for j in jobs]
    return stable_hash(sorted(payload))


def field_missingness(jobs: list[JobPosting], fields: list[str]) -> dict[str, float]:
    total = max(len(jobs), 1)
    out: dict[str, float] = {}
    for field in fields:
        missing = sum(1 for j in jobs if getattr(j, field, None) in (None, [], ""))
        out[field] = round(missing / total, 4)
    return out


def build_manifest(
    jobs: list[JobPosting],
    catalog_snapshot_id: str,
    source_files: list[str],
    reference_date: str,
) -> dict[str, Any]:
    """Build a ``catalog_manifest.json`` payload."""
    chash = catalog_hash(jobs)
    return {
        "catalog_snapshot_id": catalog_snapshot_id,
        "record_count": len(jobs),
        "created_at": datetime.now().astimezone().isoformat(),
        "source_files": source_files,
        "sha256": sha256_of_text(chash),
        "catalog_hash": chash,
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "reference_date": reference_date,
        "field_missingness": field_missingness(
            jobs,
            ["salary_min", "work_mode", "experience_level", "application_deadline", "city"],
        ),
    }


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, default=str)
